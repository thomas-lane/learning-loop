"""The episode loop: policy <-> environment, with deterministic replay/branching.

`run_episode(plan, session, policy, event_log)` runs one episode and returns
an `EpisodeRun` (the `EpisodeCore` written to `episode.json`, plus the final
message history). Grading is not part of the loop: backends grade afterwards
in a separate environment and turn the core into an `EpisodeSummary`.

Loop: [fingerprint] -> request -> response -> execute each tool call in order
-> append observations -> repeat, until a stop condition:

    model_finished                MODEL        reply without tool calls
    budget:max_turns              BUDGET       assistant turns (incl. replayed prefix + intervention)
    budget:max_episode_tokens     BUDGET       input+output across requests (incl. prefix accounting)
    budget:usage_unavailable      BUDGET       max_episode_tokens is set but a response reported no usage
    budget:output_truncated       BUDGET       finish_reason=length and no valid tool call
    safety:agent_timeout          SAFETY       wall-clock limit (budgets.agent_timeout_sec)
    safety:cancelled              SAFETY       cancelled from outside (e.g. Harbor's agent timeout)
    model_error:<msg>             MODEL_ERROR  endpoint rejected the request (4xx, e.g. context overflow)
    infra:<msg>                   INFRA        endpoint transport/5xx/timeouts, or the environment
                                               transport (EnvInfraError: exec/upload failures,
                                               backstop timeouts); retried by the coordinator
    replay:<mismatch>             REPLAY       restored state differs from the source (fail closed),
                                               incl. replay:image_mismatch (different environment image)

A turn is malformed (PARSE_ERROR events, `n_malformed_turns`) when any of its
tool calls had unusable arguments or the server reported a tool-call block it
could not parse, even if other calls in the same turn were valid.

Environment image identity (`session.image_identity()`, when available) is
recorded in `episode_start` and `extra["image_identity"]`. A branch whose
image identity differs from the source's stops as `replay:image_mismatch`;
when the source recorded none, the check is skipped and that is recorded in
`extra["image_identity_check"]`. After the intervention executes,
`extra["intervention_tool"] = {executed, error, exit_code, timed_out}`.

Replay (`plan.replay`): no model calls for the prefix. The conversation starts
from the exact historical `history_prefix`; each prefix turn's *executed*
arguments are re-run in the fresh environment and compared with the source
observations (after only the task-declared normalizers) and fingerprints. On
any mismatch the episode stops before the intervention is executed or the
model is called. Otherwise the fixed intervention (one tool call) is executed,
its fresh observation appended, and the policy continues with fresh
per-request seeds derived from `plan.seed`. Budgets are shared with the
source: the prefix + intervention turns count against `max_turns`, and the
source prefix usage + the intervention request's input tokens count against
`max_episode_tokens`, identically for both branches.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from ..core.interfaces import EpisodePlan, Policy, ReplaySpec, ReplayTurn, StateSpec
from ..core.records import (
    Event,
    EpisodeRole,
    EpisodeSummary,
    EventKind,
    Message,
    RestoreCapability,
    StopCategory,
    Timing,
    ToolExecution,
    Usage,
)
from ..core.seeds import request_seed
from .envs.base import EnvInfraError
from .events import EventLog, first_request_messages, load_turns, read_events
from .policy import parse_arguments

CORE_SCHEMA_VERSION = 1


class EpisodeCore(BaseModel):
    """Everything about an episode except grading (`episode.json`)."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = CORE_SCHEMA_VERSION
    episode_id: str
    role: EpisodeRole
    instance_id: str
    attempt_index: int | None = None
    seed: int | None = None
    checkpoint_id: str | None = None
    stop_reason: str
    stop_category: StopCategory
    usage: Usage  # sum over this episode's answered requests (a replayed prefix has none)
    n_requests: int  # requests sent, including rejected/failed ones
    n_failed_requests: int = 0  # request_error / infra_error (no usage reported)
    n_tool_calls: int  # calls issued by the learner or the intervention in this episode
    n_replayed_calls: int = 0
    n_malformed_turns: int = 0
    n_turns: int | None  # assistant turns in the final history (None: unavailable, e.g. no record)
    n_prefix_turns: int = 0
    timing: Timing = Field(default_factory=Timing)
    tool_cpu_sec: float | None = None
    tool_peak_memory_bytes: int | None = None
    infra_error: str | None = None
    replay_ok: bool | None = None  # None when not a replay episode
    replay_mismatches: list[str] = Field(default_factory=list)
    seeds_sent: bool | None = None  # were request seeds sent to the endpoint (None: no requests)
    fingerprint_errors: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_summary(
        self,
        *,
        reward: dict[str, float] | None,
        state_spec: StateSpec,
        trial_dir: str | None = None,
        events_path: str | None = None,
        stop_override: tuple[str, StopCategory] | None = None,
        infra_error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EpisodeSummary:
        partial = None
        if reward is not None and state_spec.reward_key in reward:
            partial = float(reward[state_spec.reward_key])
        stop_reason, stop_category = stop_override or (self.stop_reason, self.stop_category)
        ex = {
            "n_failed_requests": self.n_failed_requests,
            "n_replayed_calls": self.n_replayed_calls,
            "n_turns": self.n_turns,
            "n_prefix_turns": self.n_prefix_turns,
            "seeds_sent": self.seeds_sent,
            "replay_ok": self.replay_ok,
            "replay_mismatches": self.replay_mismatches,
            "fingerprint_errors": self.fingerprint_errors,
            **self.extra,
            **(extra or {}),
        }
        if stop_override:
            ex["episode_stop_reason"] = self.stop_reason
        return EpisodeSummary(
            episode_id=self.episode_id,
            role=self.role,
            instance_id=self.instance_id,
            attempt_index=self.attempt_index,
            seed=self.seed,
            checkpoint_id=self.checkpoint_id or "unknown",
            stop_reason=stop_reason,
            stop_category=stop_category,
            reward=reward,
            partial_reward=partial,
            success=(partial >= state_spec.success_threshold) if partial is not None else None,
            usage=self.usage,
            n_requests=self.n_requests,
            n_tool_calls=self.n_tool_calls,
            n_malformed_turns=self.n_malformed_turns,
            timing=self.timing,
            tool_cpu_sec=self.tool_cpu_sec,
            tool_peak_memory_bytes=self.tool_peak_memory_bytes,
            infra_error=infra_error if infra_error is not None else self.infra_error,
            trial_dir=trial_dir,
            events_path=events_path,
            extra=ex,
        )


@dataclass
class EpisodeRun:
    core: EpisodeCore
    messages: list[Message] = field(default_factory=list)


Callback = Callable[..., Any]


async def _maybe_await(x: Any) -> None:
    if inspect.isawaitable(x):
        await x


def apply_normalizers(text: str, normalizers: list[dict[str, str]]) -> str:
    """Only task-declared normalizers; each is {pattern, replacement, reason}."""
    for n in normalizers:
        text = re.sub(n["pattern"], n["replacement"], text)
    return text


def checkpoint_id_of(plan: EpisodePlan) -> str | None:
    p = plan.policy
    if p.checkpoint is not None:
        return p.checkpoint.checkpoint_id
    return p.served_model_name


class _StopEpisode(Exception):
    def __init__(self, reason: str, category: StopCategory):
        self.reason = reason
        self.category = category


class _Episode:
    def __init__(self, plan: EpisodePlan, session: Any, policy: Policy, log: EventLog, on_turn: Callback | None):
        self.plan = plan
        self.session = session
        self.policy = policy
        self.log = log
        self.on_turn = on_turn
        self.messages: list[Message] = []
        self.usages: list[Usage] = []
        self.n_requests = 0
        self.n_failed = 0
        self.n_tool_calls = 0
        self.n_replayed_calls = 0
        self.n_malformed = 0
        self.n_prefix_turns = 0
        self.turn = 0
        self.tokens_used: int | None = 0
        self.endpoint_sec = 0.0
        self.tool_sec = 0.0
        self.cpu: list[float | None] = []
        self.infra_error: str | None = None
        self.replay_ok: bool | None = None
        self.replay_mismatches: list[str] = []
        self.fp_errors: list[str] = []
        self.seed_flags: list[bool] = []
        self.extra: dict[str, Any] = {}
        self.image_identity: dict[str, Any] | None = None

    # -- helpers ------------------------------------------------------------ #

    def emit(self, kind: EventKind, data: dict[str, Any], turn: int | None = None) -> None:
        self.log.emit(kind, data, turn_index=turn)

    async def fingerprint(self, turn: int) -> str | None:
        spec = self.plan.state_spec
        if not spec.fingerprint_paths or not getattr(self.session, "capabilities", None) or not self.session.capabilities.fingerprint:
            return None
        try:
            if hasattr(self.session, "fingerprint_with_detail"):
                digest, detail = await self.session.fingerprint_with_detail(spec)
            else:
                digest, detail = await self.session.fingerprint(spec), None
        except EnvInfraError:
            raise  # the environment failed, not the state: INFRA, never a replay mismatch
        except Exception as e:  # recorded; makes this state non-replayable
            msg = f"turn {turn}: {type(e).__name__}: {e}"
            self.fp_errors.append(msg)
            self.emit(EventKind.FINGERPRINT, {"fingerprint": None, "detail": None, "error": msg, "paths": spec.fingerprint_paths, "exclude": spec.fingerprint_exclude}, turn)
            return None
        self.emit(EventKind.FINGERPRINT, {"fingerprint": digest, "detail": detail, "paths": spec.fingerprint_paths, "exclude": spec.fingerprint_exclude}, turn)
        return digest

    def record_tool(self, turn: int, order: int, te: ToolExecution, replay: bool = False) -> None:
        self.emit(
            EventKind.TOOL_CALL,
            {
                "tool_call_id": te.call_id,
                "name": te.name,
                "requested_arguments_raw": te.requested_arguments_raw,
                "executed_arguments": te.executed_arguments,
                "executed": te.executed,
                "order": order,
                "replay": replay,
            },
            turn,
        )
        self.emit(EventKind.TOOL_RESULT, te.model_dump(mode="json"), turn)
        if te.duration_sec is not None:
            self.tool_sec += te.duration_sec
        if te.executed:
            self.cpu.append(te.cpu_time_sec)

    async def run_call(self, turn: int, order: int, call: dict[str, Any], error: str | None, original_raw: str | None, finish_reason: str | None) -> ToolExecution:
        cid = call["id"]
        name = call["function"]["name"]
        if error is not None:
            if finish_reason == "length":
                limit = self.plan.policy.sampling.max_output_tokens
                obs = (
                    f"[error] your response hit the max_tokens limit ({limit}) and this tool call was cut off, "
                    "so it was not run. Keep reasoning shorter or split large writes into smaller pieces."
                )
            else:
                obs = f"[error] {error}"
            te = ToolExecution(call_id=cid, name=name, requested_arguments_raw=original_raw, executed_arguments=None, executed=False, raw_output=obs, observation=obs, error=error)
        else:
            args, _ = parse_arguments(call["function"]["arguments"])
            te = await self.session.execute(cid, name, args or {}, original_raw)
        self.record_tool(turn, order, te)
        self.n_tool_calls += 1
        self.messages.append({"role": "tool", "tool_call_id": cid, "content": te.observation})
        return te

    async def turn_done(self) -> None:
        if self.on_turn is not None:
            await _maybe_await(self.on_turn(self.messages))

    # -- replay ------------------------------------------------------------- #

    def _prefix_consistency(self, r: ReplaySpec) -> list[str]:
        """history_prefix must be system/user + exactly the prefix turns and their observations."""
        errs: list[str] = []
        if [t.turn_index for t in r.prefix_turns] != list(range(r.intervention_turn)):
            errs.append("prefix_turns_not_contiguous")
            return errs
        hist = r.history_prefix
        asst_idx = [i for i, m in enumerate(hist) if m.get("role") == "assistant"]
        if len(asst_idx) != len(r.prefix_turns):
            return ["history_prefix_turn_count"]
        for rt, i in zip(r.prefix_turns, asst_idx):
            if hist[i] != rt.assistant_message:
                errs.append(f"history_prefix_assistant:turn={rt.turn_index}")
                continue
            calls = rt.assistant_message.get("tool_calls") or []
            tools = hist[i + 1 : i + 1 + len(calls)]
            got = [(m.get("role"), m.get("tool_call_id"), m.get("content")) for m in tools]
            want = [("tool", c["id"], o) for c, o in zip(calls, rt.expected_observations)]
            if len(calls) != len(rt.expected_observations) or len(calls) != len(rt.executed_arguments) or got != want:
                errs.append(f"history_prefix_observations:turn={rt.turn_index}")
        return errs

    async def replay(self) -> None:
        r = self.plan.replay
        assert r is not None
        spec = self.plan.state_spec
        b = self.plan.budgets
        self.messages = copy.deepcopy(r.history_prefix)
        k = r.intervention_turn
        self.n_prefix_turns = k
        mismatches: list[str] = []
        details: list[dict[str, Any]] = []

        if spec.restore != RestoreCapability.DETERMINISTIC_REPLAY:
            mismatches.append(f"restore_unsupported:{spec.restore.value}")
        elif self.session.capabilities.restore != RestoreCapability.DETERMINISTIC_REPLAY:
            mismatches.append(f"environment_restore:{self.session.capabilities.restore.value}")
        elif not spec.fingerprint_paths:
            mismatches.append("no_fingerprint_paths_declared")
        exp_img = r.expected_image_identity
        if exp_img is None:
            self.extra["image_identity_check"] = "skipped: source has no recorded image identity"
        elif self.image_identity is None or self.image_identity.get("identity") != exp_img.get("identity"):
            self.extra["image_identity_check"] = "mismatch"
            mismatches.append("image_mismatch")
            details.append({"image_identity": {"expected": exp_img, "actual": self.image_identity}})
        else:
            self.extra["image_identity_check"] = "ok"
        if b.max_episode_tokens is not None and (r.prefix_usage.total is None or r.intervention_request_input_tokens is None):
            mismatches.append("prefix_usage_unavailable")
        icalls = r.intervention_message.get("tool_calls") or []
        if r.intervention_message.get("role") != "assistant" or len(icalls) != 1:
            mismatches.append("invalid_intervention")
        elif parse_arguments(icalls[0].get("function", {}).get("arguments"))[1] is not None:
            mismatches.append("invalid_intervention_arguments")
        if not mismatches:
            mismatches += self._prefix_consistency(r)

        if not mismatches:
            for rt in r.prefix_turns:
                i = rt.turn_index
                if rt.expected_fingerprint_before is not None:
                    got = await self.fingerprint(i)
                    if got != rt.expected_fingerprint_before:
                        mismatches.append(f"fingerprint:turn={i}")
                        break
                self.emit(EventKind.REPLAY_ACTION, {"assistant_message": rt.assistant_message}, i)
                calls = rt.assistant_message.get("tool_calls") or []
                for j, (call, args, expected) in enumerate(zip(calls, rt.executed_arguments, rt.expected_observations)):
                    fn = call["function"]
                    if args is None:  # the source did not execute this call (e.g. malformed arguments)
                        te = ToolExecution(
                            call_id=call["id"], name=fn["name"], requested_arguments_raw=fn.get("arguments"), executed_arguments=None,
                            executed=False, raw_output=expected, observation=expected, error="not executed in the source episode",
                        )
                    else:
                        te = await self.session.execute(call["id"], fn["name"], args, fn.get("arguments"))
                    self.record_tool(i, j, te, replay=True)
                    self.n_replayed_calls += 1
                    if apply_normalizers(te.observation, spec.observation_normalizers) != apply_normalizers(expected, spec.observation_normalizers):
                        mismatches.append(f"observation:turn={i}:call={j}")
                        details.append({"turn": i, "call": j, "expected": expected[:4000], "actual": te.observation[:4000]})
                        break
                if mismatches:
                    break
            if not mismatches:
                got = await self.fingerprint(k)
                exp = r.expected_fingerprint_before_intervention
                if exp is None:
                    mismatches.append("missing_source_fingerprint")
                elif got is None:
                    mismatches.append("fingerprint_unavailable")
                elif got != exp:
                    mismatches.append("fingerprint:before_intervention")
                    details.append({"turn": k, "expected": exp, "actual": got})

        self.replay_ok = not mismatches
        self.replay_mismatches = mismatches
        self.emit(EventKind.REPLAY_CHECK, {"ok": self.replay_ok, "mismatches": mismatches, "normalizers": spec.observation_normalizers, "details": details}, k)
        self.turn = k
        if mismatches:
            raise _StopEpisode(f"replay:{mismatches[0]}", StopCategory.REPLAY)

        msg = r.intervention_message
        c = msg["tool_calls"][0]
        args, _ = parse_arguments(c["function"].get("arguments"))
        self.emit(EventKind.INTERVENTION, {"label": r.intervention_label, "assistant_message": msg}, k)
        self.messages.append(copy.deepcopy(msg))
        try:
            te = await self.session.execute(c["id"], c["function"]["name"], args or {}, c["function"]["arguments"])
        except EnvInfraError as e:
            self.extra["intervention_tool"] = {"executed": False, "error": f"infra: {e}", "exit_code": None, "timed_out": False}
            raise
        self.extra["intervention_tool"] = {"executed": te.executed, "error": te.error, "exit_code": te.exit_code, "timed_out": self._timed_out(te)}
        self.record_tool(k, 0, te)
        self.n_tool_calls += 1
        self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": te.observation})
        self.turn = k + 1
        pt = r.prefix_usage.total
        self.tokens_used = None if pt is None or r.intervention_request_input_tokens is None else pt + r.intervention_request_input_tokens
        self.extra["budget_prefix_tokens"] = self.tokens_used
        await self.turn_done()

    def _timed_out(self, te: ToolExecution) -> bool:
        """The call hit its timeout: exit 124 under an in-container timeout wrapper, or a timeout error."""
        if te.timed_out:
            return True
        target = getattr(self.session, "target", None)
        if te.exit_code == 124 and getattr(target, "enforces_timeout", False):
            return True
        return bool(te.error and re.search(r"timed out|TimeoutError", te.error))

    # -- main loop ---------------------------------------------------------- #

    async def loop(self) -> None:
        plan, b = self.plan, self.plan.budgets
        while True:
            if self.turn >= b.max_turns:
                raise _StopEpisode("budget:max_turns", StopCategory.BUDGET)
            if b.max_episode_tokens is not None:
                if self.tokens_used is None:  # a response without usage: the budget cannot be enforced
                    raise _StopEpisode("budget:usage_unavailable", StopCategory.BUDGET)
                if self.tokens_used >= b.max_episode_tokens:
                    raise _StopEpisode("budget:max_episode_tokens", StopCategory.BUDGET)
            t = self.turn
            if plan.record_fingerprints:
                await self.fingerprint(t)
            req_idx = self.n_requests
            seed = request_seed(plan.seed, req_idx)
            will_send = seed if (plan.policy.send_seed and seed is not None) else None
            self.emit(
                EventKind.REQUEST,
                {
                    "request_index": req_idx,
                    "model": plan.policy.served_model_name,
                    "messages": self.messages,
                    "tools": plan.tools,
                    "sampling": plan.policy.sampling.model_dump(mode="json"),
                    "seed": will_send,
                    "seed_requested": seed,
                },
                t,
            )
            self.n_requests += 1
            d = await self.policy.decide(list(self.messages), plan.tools, seed)
            self.seed_flags.append(d.seed_sent is not None)
            if d.latency_sec is not None:
                self.endpoint_sec += d.latency_sec
            if d.infra_error is not None:
                self.n_failed += 1
                self.infra_error = d.infra_error
                self.emit(EventKind.INFRA_ERROR, {"where": "policy", "error": d.infra_error, "request_index": req_idx, "raw": d.raw_response}, t)
                raise _StopEpisode(f"infra:{d.infra_error}", StopCategory.INFRA)
            if d.request_error is not None:
                self.n_failed += 1
                self.emit(
                    EventKind.RESPONSE,
                    {"request_index": req_idx, "raw": d.raw_response, "history_message": None, "finish_reason": None, "usage": None, "latency_sec": d.latency_sec, "seed_sent": d.seed_sent, "error": d.request_error},
                    t,
                )
                raise _StopEpisode(f"model_error:{d.request_error}", StopCategory.MODEL_ERROR)
            self.emit(
                EventKind.RESPONSE,
                {
                    "request_index": req_idx,
                    "raw": d.raw_response,
                    "history_message": d.history_message,
                    "finish_reason": d.finish_reason,
                    "usage": d.usage.model_dump(mode="json"),
                    "latency_sec": d.latency_sec,
                    "seed_sent": d.seed_sent,
                },
                t,
            )
            for cid, err in d.call_errors.items():
                self.emit(EventKind.PARSE_ERROR, {"request_index": req_idx, "tool_call_id": cid, "error": err}, t)
            original_raw: dict[str, Any] = {}
            for rep in d.repairs:
                self.emit(EventKind.REPAIR, {"request_index": req_idx, **rep}, t)
                if rep.get("field") == "function.arguments":
                    original_raw[rep["tool_call_id"]] = rep["original"]
            self.usages.append(d.usage)
            if self.tokens_used is not None:
                tot = d.usage.total
                self.tokens_used = None if tot is None else self.tokens_used + tot
            self.messages.append(d.history_message)
            self.turn += 1
            calls = d.history_message.get("tool_calls") or []
            # Tool-call blocks the server could not parse are malformed calls even when other
            # calls in the same turn parsed: visible PARSE_ERRORs, and the turn is malformed.
            unparsed = [e for e in d.parse_errors if e.startswith("unparsed_tool_call:")]
            for e in unparsed:
                self.emit(EventKind.PARSE_ERROR, {"request_index": req_idx, "tool_call_id": None, "error": e}, t)
            if d.call_errors or unparsed:
                self.n_malformed += 1
            if not calls and unparsed:
                # Only unparseable attempts: an explicit error, not a successful action and not a normal finish.
                await self.turn_done()
                raise _StopEpisode("model_error:unparsed_tool_call", StopCategory.MODEL_ERROR)
            if not calls:
                await self.turn_done()
                if d.finish_reason == "length":
                    raise _StopEpisode("budget:output_truncated", StopCategory.BUDGET)
                raise _StopEpisode("model_finished", StopCategory.MODEL)
            for j, c in enumerate(calls):
                raw = original_raw.get(c["id"], c["function"]["arguments"])
                await self.run_call(t, j, c, d.call_errors.get(c["id"]), raw, d.finish_reason)
            await self.turn_done()
            if d.finish_reason == "length" and all(c["id"] in d.call_errors for c in calls):
                raise _StopEpisode("budget:output_truncated", StopCategory.BUDGET)

    # -- result ------------------------------------------------------------- #

    def core(self, reason: str, category: StopCategory, total_sec: float) -> EpisodeCore:
        p = self.plan
        cpu = None if not self.cpu or any(c is None for c in self.cpu) else round(sum(c for c in self.cpu if c is not None), 6)
        if self.n_failed:
            self.extra["failed_requests_usage"] = "not reported by the endpoint; excluded from usage"
        return EpisodeCore(
            episode_id=p.episode_id,
            role=p.role,
            instance_id=p.instance_id,
            attempt_index=p.attempt_index,
            seed=p.seed,
            checkpoint_id=checkpoint_id_of(p),
            stop_reason=reason,
            stop_category=category,
            # answered requests only; failed requests report no usage (counted in n_failed_requests)
            usage=Usage.sum(self.usages) if self.usages else (Usage(source="none") if self.n_failed else Usage(input_tokens=0, output_tokens=0, source="none")),
            n_requests=self.n_requests,
            n_failed_requests=self.n_failed,
            n_tool_calls=self.n_tool_calls,
            n_replayed_calls=self.n_replayed_calls,
            n_malformed_turns=self.n_malformed,
            n_turns=sum(1 for m in self.messages if m.get("role") == "assistant"),
            n_prefix_turns=self.n_prefix_turns,
            timing=Timing(total_sec=round(total_sec, 6), endpoint_sec=round(self.endpoint_sec, 6), tool_sec=round(self.tool_sec, 6)),
            tool_cpu_sec=cpu,
            tool_peak_memory_bytes=None,
            infra_error=self.infra_error,
            replay_ok=self.replay_ok,
            replay_mismatches=self.replay_mismatches,
            seeds_sent=(all(self.seed_flags) if self.seed_flags else None),
            fingerprint_errors=self.fp_errors,
            extra=self.extra,
        )


async def run_episode(
    plan: EpisodePlan,
    session: Any,
    policy: Policy,
    event_log: EventLog,
    *,
    on_turn: Callback | None = None,
    on_end: Callback | None = None,
) -> EpisodeRun:
    """Run one episode. `on_turn(messages)` after every turn; `on_end(run)` once,
    also when the episode is cancelled from outside (before re-raising)."""
    ep = _Episode(plan, session, policy, event_log, on_turn)
    ep.extra["seed_supported"] = plan.policy.seed_supported  # declared by the model profile; None = unknown
    t0 = time.monotonic()
    if hasattr(session, "image_identity"):
        try:
            ep.image_identity = await session.image_identity()
        except Exception as e:  # recorded; a replay against this episode then fails closed
            ep.extra["image_identity_error"] = f"{type(e).__name__}: {e}"
    ep.extra["image_identity"] = ep.image_identity
    ep.emit(EventKind.EPISODE_START, {"plan": plan.model_dump(mode="json"), "image_identity": ep.image_identity})
    if plan.replay is None:
        ep.messages = [{"role": "system", "content": plan.system_prompt}, {"role": "user", "content": plan.instruction}]
    reason, category = "", StopCategory.MODEL
    cancelled: BaseException | None = None
    try:
        async with asyncio.timeout(plan.budgets.agent_timeout_sec) as cm:
            try:
                if plan.replay is not None:
                    await ep.replay()
                await ep.loop()
            except _StopEpisode as s:
                reason, category = s.reason, s.category
    except TimeoutError as e:
        if cm.expired():
            reason, category = "safety:agent_timeout", StopCategory.SAFETY
        else:
            ep.infra_error = f"{type(e).__name__}: {e}"
            reason, category = f"infra:{ep.infra_error}", StopCategory.INFRA
    except asyncio.CancelledError as e:
        reason, category = "safety:cancelled", StopCategory.SAFETY
        cancelled = e
    except EnvInfraError as e:  # the environment transport failed (also during replay)
        ep.infra_error = f"{type(e).__name__}: {e}"
        ep.emit(EventKind.INFRA_ERROR, {"where": "environment", "error": ep.infra_error}, ep.turn)
        reason, category = f"infra:{ep.infra_error}", StopCategory.INFRA
    except Exception as e:  # environment/transport failure outside the tool handlers
        ep.infra_error = f"{type(e).__name__}: {e}"
        ep.emit(EventKind.INFRA_ERROR, {"where": "episode", "error": ep.infra_error}, ep.turn)
        reason, category = f"infra:{ep.infra_error}", StopCategory.INFRA
    core = ep.core(reason, category, time.monotonic() - t0)
    ep.emit(
        EventKind.EPISODE_END,
        {
            "stop_reason": core.stop_reason,
            "stop_category": core.stop_category.value,
            "usage": core.usage.model_dump(mode="json"),
            "n_requests": core.n_requests,
            "n_tool_calls": core.n_tool_calls,
            "timing": core.timing.model_dump(mode="json"),
            "replay_ok": core.replay_ok,
        },
    )
    run = EpisodeRun(core=core, messages=ep.messages)
    if on_end is not None:
        await _maybe_await(on_end(run))
    if cancelled is not None:
        raise cancelled
    return run


# --------------------------------------------------------------------------- #
# Replay specs from a source episode's events
# --------------------------------------------------------------------------- #


def build_replay_spec(events: list[Event] | Path | str, intervention_turn: int, label: str, intervention_message: Message) -> ReplaySpec:
    """Restore the decision state before `intervention_turn` of a source episode.

    Uses only the source prefix: the exact request messages of turn k, the
    executed arguments/observations/fingerprints of turns < k, and usage of
    requests < k (+ the input tokens of request k). Nothing after turn k.
    """
    evs = read_events(Path(events)) if isinstance(events, (str, Path)) else events
    turns = load_turns(evs)
    k = intervention_turn
    by_index = {t.turn_index: t for t in turns}
    if k not in by_index:
        raise ValueError(f"source has no assistant turn {k}")
    target = by_index[k]
    if target.origin != "model" or target.request_seq is None or target.usage is None:
        raise ValueError(f"turn {k} is not an answered model turn")
    prefix = [by_index[i] for i in range(k) if i in by_index]
    if len(prefix) != k:
        raise ValueError("source prefix has missing turns")
    history = first_request_messages(evs, k)
    if history is None:
        raise ValueError(f"no request recorded for turn {k}")
    prefix_turns = [
        ReplayTurn(
            turn_index=t.turn_index,
            assistant_message=t.assistant_message,
            executed_arguments=[te.executed_arguments for te in t.tool_executions],
            expected_observations=[te.observation for te in t.tool_executions],
            expected_fingerprint_before=t.fingerprint_before,
        )
        for t in prefix
    ]
    source_episode_id = evs[0].episode_id if evs else ""
    return ReplaySpec(
        expected_image_identity=source_image_identity(evs),
        source_episode_id=source_episode_id,
        history_prefix=copy.deepcopy(history),
        prefix_turns=prefix_turns,
        intervention_turn=k,
        intervention_label=label,
        intervention_message=intervention_message,
        expected_fingerprint_before_intervention=target.fingerprint_before,
        # A prefix turn without usage (e.g. itself replayed) makes the prefix usage unavailable.
        prefix_usage=Usage.sum([t.usage if t.usage is not None else Usage(source="none") for t in prefix]),
        intervention_request_input_tokens=target.usage.input_tokens,
    )


def source_image_identity(events: list[Event]) -> dict[str, Any] | None:
    """The image identity a source episode recorded in `episode_start` (None if absent)."""
    for e in events:
        if e.kind == EventKind.EPISODE_START:
            ident = e.data.get("image_identity")
            return ident if isinstance(ident, dict) else None
    return None


def write_episode_files(out_dir: Path, run: EpisodeRun, tools: list[dict[str, Any]]) -> None:
    """`episode.json` (core) and `messages.json` (final request history + tools)."""
    from ..core.storage import atomic_write_json

    atomic_write_json(out_dir / "episode.json", run.core)
    atomic_write_json(out_dir / "messages.json", {"tools": tools, "messages": run.messages})


def load_core(path: Path) -> EpisodeCore:
    return EpisodeCore.model_validate(json.loads(Path(path).read_text()))
