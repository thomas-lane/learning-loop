"""A tool-calling agent for Harbor that talks to any OpenAI-compatible chat
completions endpoint (llama.cpp's llama-server, vLLM, SGLang, ...).

Run it with:
    evaluation/run.sh                     # uses configs/local-llama.yaml
    harbor run -p evaluation/tasks -a evaluation.agents.tool_agent:ToolAgent \
        -m <model-id> --ak api_base=http://localhost:8080/v1

Where it runs: this class runs in the Harbor process on the *host*. Only tool
executions happen in the task container (via `environment.exec` /
`upload_file`), so `localhost` in `api_base` means your machine.

The loop itself lives in `learning_loop.episode.run_episode` (policy <->
environment, stop reasons, replay/branching); the tools are
`evaluation/agents/tools.py`; the endpoint client is
`learning_loop.policy.OpenAIChatPolicy`.

Two modes:
  * CLI mode (no `episode_plan_path`): options below, exactly as before. The
    system prompt is system_prompt.md with {workdir} filled in, the user message
    is the task's instruction.md verbatim, no request seed is sent.
  * Plan mode (`episode_plan_path=<EpisodePlan JSON>`): used by the learning
    loop for collection, evaluation and branch continuations (including
    scripted fixture policies). The plan fixes the policy, seeds, budgets,
    prompts, tools, replay prefix and state-fingerprint contract.

Outputs:
  record dir (host-only; `record_dir` option, default <trial>/learning_loop/):
    events.jsonl   append-only lossless record: every request as sent, raw
                   response, parse errors, repairs, tool calls (requested vs
                   executed arguments), raw tool output + exact truncated
                   observation, fingerprints, replay checks
    episode.json   episode core (stop reason/category, usage, counts, timing)
    trajectory.json, messages.json  Harbor-facing views, rewritten every turn
  agent log dir (`self.logs_dir`, also mounted *writable* in the container at /logs/agent):
    trajectory.json  ATIF trajectory (`harbor view` / `harbor analyze`)
    messages.json    final request history + tools (NOT a complete record)
    events.jsonl, episode.json  copies of the host-only records
  Nothing is written into the agent log dir while the episode runs: the
  container could plant symlinks there. At the end, the four files are copied
  in by `copy_into_untrusted_dir()`, which never follows a symlink (temp file
  opened with O_CREAT|O_EXCL|O_NOFOLLOW inside a directory fd, then renamed
  over the name, which replaces rather than follows a planted symlink).
"""

from __future__ import annotations

import asyncio
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.options import AgentOptions, Env
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from learning_loop.envs.harbor_session import HarborSession
from learning_loop.episode import EpisodeRun, run_episode, write_episode_files
from learning_loop.events import EventLog, load_turns
from learning_loop.interfaces import EpisodePlan, StateSpec
from learning_loop.policy import OpenAIChatPolicy, make_policy
from learning_loop.records import EpisodeBudgets, EpisodeRole, PolicySpec, SamplingConfig
from learning_loop.storage import atomic_write_json

from .tools import ToolConfig, tool_schemas

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")
NO_WALL_CLOCK_LIMIT_SEC = 1e9  # CLI mode: Harbor's [agent].timeout_sec is the wall-clock limit


class ToolAgentOptions(AgentOptions):
    """Options passed with `--ak key=value` (or `kwargs:` in a job config)."""

    api_base: Annotated[str, Env("LLM_API_BASE", fallback="LLM_API_BASE")] = Field(
        default="http://localhost:8080/v1",
        description="OpenAI-compatible base URL (falls back to $LLM_API_BASE).",
    )
    api_key: Annotated[str, Env("LLM_API_KEY", fallback="LLM_API_KEY")] = Field(
        default="sk-no-key", description="API key (falls back to $LLM_API_KEY)."
    )
    max_turns: int = Field(default=30, description="Max LLM calls per trial.")
    temperature: float | None = Field(default=None)
    max_tokens: int | None = Field(
        default=8192,
        description="Max completion tokens per LLM call (reasoning included).",
    )
    workdir: str = Field(default="/app", description="cwd for tool execution.")
    command_timeout_sec: int = Field(default=60)
    max_output_chars: int = Field(
        default=8000, description="Per-tool-result truncation limit."
    )
    system_prompt_path: str | None = Field(
        default=None, description="Override system_prompt.md."
    )
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Extra request fields, e.g. llama.cpp sampling params.",
    )
    episode_plan_path: str | None = Field(
        default=None,
        description="Learning-loop EpisodePlan JSON; when set, it overrides the options above.",
    )
    record_dir: str | None = Field(
        default=None,
        description="Host directory for events.jsonl/episode.json (default: <trial dir>/learning_loop).",
    )


class ToolAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)
    options_model = ToolAgentOptions
    options: ToolAgentOptions

    @staticmethod
    def name() -> str:
        return "tool-agent"

    def version(self) -> str:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing to install: the agent loop runs on the host.
        pass

    # -- plan construction -------------------------------------------------- #

    def _cli_plan(self, instruction: str) -> EpisodePlan:
        opts = self.options
        prompt_path = Path(opts.system_prompt_path or SYSTEM_PROMPT_PATH)
        return EpisodePlan(
            episode_id=self.session_id or "tool-agent",
            role=EpisodeRole.EVAL,
            instance_id="harbor-cli",
            policy=PolicySpec(
                kind="openai",
                served_model_name=self.model_name or "default",
                api_base=opts.api_base,
                api_key_env="LLM_API_KEY",
                send_seed=False,
                sampling=SamplingConfig(temperature=opts.temperature, max_output_tokens=opts.max_tokens, extra_body=opts.extra_body or {}),
            ),
            budgets=EpisodeBudgets(
                max_turns=opts.max_turns,
                max_episode_tokens=None,
                tool_timeout_sec=opts.command_timeout_sec,
                max_output_chars=opts.max_output_chars,
                agent_timeout_sec=NO_WALL_CLOCK_LIMIT_SEC,
            ),
            system_prompt=prompt_path.read_text().format(workdir=opts.workdir),
            instruction=instruction,
            tools=tool_schemas(),
            state_spec=StateSpec(),
            record_fingerprints=False,
            workdir=opts.workdir,
        )

    # -- run ------------------------------------------------------------------ #

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        opts = self.options
        if opts.episode_plan_path:
            plan = EpisodePlan.model_validate_json(Path(opts.episode_plan_path).read_text())
            if plan.instruction.strip() != instruction.strip():
                raise ValueError("episode plan instruction differs from the task's instruction.md; refusing to run")
            policy = make_policy(plan.policy)
        else:
            plan = self._cli_plan(instruction)
            policy = OpenAIChatPolicy(plan.policy, api_key=opts.api_key)
        record_dir = Path(opts.record_dir) if opts.record_dir else self.logs_dir.parent / "learning_loop"
        record_dir.mkdir(parents=True, exist_ok=True)
        events_path = record_dir / "events.jsonl"
        log = EventLog(events_path, plan.episode_id)
        session = HarborSession(
            environment,
            ToolConfig(workdir=plan.workdir, command_timeout_sec=plan.budgets.tool_timeout_sec, max_output_chars=plan.budgets.max_output_chars),
            restore=plan.state_spec.restore,
        )

        def flush(messages: list[dict[str, Any]]) -> None:
            """Persist Harbor-facing views every turn (host-only dir) so a crash still leaves logs."""
            atomic_write_json(record_dir / "messages.json", {"tools": plan.tools, "messages": messages})
            (record_dir / "trajectory.json.tmp").write_text(self._trajectory(plan, events_path).model_dump_json(indent=2, exclude_none=True))
            os.replace(record_dir / "trajectory.json.tmp", record_dir / "trajectory.json")

        def finish(run: EpisodeRun) -> None:
            write_episode_files(record_dir, run, plan.tools)
            core = run.core
            context.n_input_tokens = core.usage.input_tokens
            context.n_output_tokens = core.usage.output_tokens
            context.n_cache_tokens = core.usage.cached_input_tokens
            context.metadata = {
                "stop_reason": core.stop_reason,
                "stop_category": core.stop_category.value,
                "turns": core.n_turns,
                "n_requests": core.n_requests,
                "episode_id": core.episode_id,
                "record_dir": str(record_dir),
                "api_base": plan.policy.api_base,
                "usage_source": core.usage.source,
            }
            flush(run.messages)
            if record_dir.resolve() == Path(self.logs_dir).resolve():
                return
            refused = {}
            for name in ("trajectory.json", "messages.json", "events.jsonl", "episode.json"):
                src = record_dir / name
                if src.exists():
                    err = copy_into_untrusted_dir(src, Path(self.logs_dir), name)
                    if err:
                        refused[name] = err
            if refused:
                context.metadata["agent_log_copy_refused"] = refused

        try:
            await run_episode(plan, session, policy, log, on_turn=flush, on_end=finish)
        finally:
            await asyncio.shield(policy.aclose())

    # -- ATIF ------------------------------------------------------------------ #

    def _trajectory(self, plan: EpisodePlan, events_path: Path) -> Trajectory:
        turns = load_turns(events_path)
        steps: list[Step] = []
        prefix = plan.replay.history_prefix[:2] if plan.replay else None
        system = prefix[0]["content"] if prefix else plan.system_prompt
        user = prefix[1]["content"] if prefix else plan.instruction
        steps.append(Step(step_id=1, timestamp=_now(), source="system", message=system))
        steps.append(Step(step_id=2, timestamp=_now(), source="user", message=user))
        tot_in = tot_out = tot_cached = 0
        known = {"in": True, "out": True, "cached": True}
        for t in turns:
            msg = t.assistant_message
            calls = [
                ToolCall(
                    tool_call_id=te.call_id,
                    function_name=te.name,
                    arguments=te.executed_arguments if te.executed_arguments is not None else {"_raw": te.requested_arguments_raw},
                )
                for te in t.tool_executions
            ]
            results = [ObservationResult(source_call_id=te.call_id, content=te.observation) for te in t.tool_executions]
            u = t.usage
            if t.origin == "model":
                for key, val in (("in", u.input_tokens if u else None), ("out", u.output_tokens if u else None), ("cached", u.cached_input_tokens if u else None)):
                    if val is None:
                        known[key] = False
                tot_in += (u.input_tokens or 0) if u else 0
                tot_out += (u.output_tokens or 0) if u else 0
                tot_cached += (u.cached_input_tokens or 0) if u else 0
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=_now(),
                    source="agent",
                    model_name=self.model_name,
                    message=msg.get("content") or "",
                    reasoning_content=msg.get("reasoning_content") or None,
                    tool_calls=calls or None,
                    observation=Observation(results=results) if results else None,
                    metrics=Metrics(
                        prompt_tokens=u.input_tokens if u else None,
                        completion_tokens=u.output_tokens if u else None,
                        cached_tokens=u.cached_input_tokens if u else None,
                        extra={"latency_sec": t.latency_sec, "finish_reason": t.finish_reason, "origin": t.origin, "turn_index": t.turn_index},
                    ),
                )
            )
        return Trajectory(
            session_id=self.session_id,
            agent=Agent(
                name=self.name(),
                version=self.version(),
                model_name=self.model_name,
                tool_definitions=plan.tools,
                extra={"api_base": plan.policy.api_base, "episode_id": plan.episode_id},
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=tot_in if known["in"] else None,
                total_completion_tokens=tot_out if known["out"] else None,
                total_cached_tokens=tot_cached if known["cached"] else None,
                total_steps=len(steps),
            ),
        )


def copy_into_untrusted_dir(src: Path, dest_dir: Path, name: str) -> str | None:
    """Copy `src` to `dest_dir/name` without following symlinks; returns an error or None.

    `dest_dir` is writable by the task container (bind mount), so any entry in
    it may be a planted symlink. The directory is opened with O_NOFOLLOW, the
    data goes to a fresh temp name opened O_CREAT|O_EXCL|O_NOFOLLOW relative to
    that directory fd, and os.replace() renames it over `name` (a rename
    replaces a symlink instead of writing through it). A planted directory at
    `name` makes the copy fail (reported, never followed).
    """
    if "/" in name or name in ("", ".", ".."):
        return f"invalid name {name!r}"
    data = Path(src).read_bytes()
    try:
        dfd = os.open(dest_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        return f"cannot open {dest_dir} without following links: {e}"
    tmp = None
    try:
        for _ in range(5):
            cand = f".{name}.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
            except FileExistsError:
                continue
            tmp = cand
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            break
        if tmp is None:
            return "could not create a temp file"
        os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        tmp = None
        return None
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
