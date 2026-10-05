"""Synthetic source episodes and a fake EpisodeBackend for Workstream C unit tests.

Nothing here touches Docker or a model. The source episode mimics the event
contract in `learning_loop.episodes.events` (episode_start plan, request/response per
model turn, tool_call/tool_result, fingerprints before each model turn).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from learning_loop.core.interfaces import EpisodePlan, EpisodeResult, StateSpec
from learning_loop.core.records import (
    EpisodeBudgets,
    EpisodeRole,
    EpisodeSummary,
    EventKind,
    PolicySpec,
    RestoreCapability,
    StopCategory,
    TaskInstance,
    ToolExecution,
    Usage,
)
from learning_loop.episodes.events import EventLog

SYSTEM = "You are an autonomous agent. The working directory is `/app`."
INSTRUCTION = (
    "The nginx access logs are in `/app/logs/`. Find the client IP address responsible for the most "
    "HTTP 5xx responses across all of the logs there. Write just that IP address to `/app/answer.txt`."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout_sec": {"type": "integer"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "num_lines": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
]

PIPELINE = "zcat -f /app/logs/* | awk '$9 ~ /^5/ {print $1}' | sort | uniq -c | sort -rn | head -1"

# (tool name, arguments, observation) per tool turn; the final turn is text only.
DEFAULT_TURNS: list[tuple[str, dict[str, Any], str]] = [
    ("bash", {"command": "ls /app/logs"}, "[exit code 0]\naccess.log\naccess.log.1\naccess.log.2.gz"),
    (
        "bash",
        {"command": "zcat -f /app/logs/* | awk '$9 ~ /^5/' | head -2"},
        '[exit code 0]\n10.0.0.7 - - [01/Jan/2026] "GET /a" 503 12\n192.168.4.20 - - [01/Jan/2026] "GET /b" 500 9',
    ),
    ("bash", {"command": PIPELINE}, "[exit code 0]\n     42 10.0.0.7"),
    ("write_file", {"path": "/app/answer.txt", "content": "10.0.0.7"}, "Wrote 8 characters to /app/answer.txt"),
]
FINAL_ANSWER = "Done: 10.0.0.7 caused the most 5xx responses."


def usage_for(i: int) -> Usage:
    return Usage(input_tokens=200 + 100 * i, output_tokens=20 + i, source="provider")


def state_spec(**kw: Any) -> StateSpec:
    base = dict(restore=RestoreCapability.DETERMINISTIC_REPLAY, fingerprint_paths=["/app"], success_threshold=1.0)
    base.update(kw)
    return StateSpec(**base)


def make_plan(episode_id: str = "src-ep", spec: StateSpec | None = None) -> EpisodePlan:
    return EpisodePlan(
        episode_id=episode_id,
        role=EpisodeRole.COLLECT,
        instance_id="log-triage/easy/s0",
        attempt_index=0,
        seed=1234,
        policy=PolicySpec(kind="scripted", scripted_path="fixture.yaml"),
        budgets=EpisodeBudgets(max_turns=10, max_episode_tokens=20000, tool_timeout_sec=30, max_output_chars=4000, agent_timeout_sec=300),
        system_prompt=SYSTEM,
        instruction=INSTRUCTION,
        tools=TOOLS,
        state_spec=spec or state_spec(),
    )


def make_instance(tmp: Path, split_family: str = "log-triage", **kw: Any) -> TaskInstance:
    base = dict(
        instance_id="log-triage/easy/s0",
        family=split_family,
        difficulty="easy",
        skills=["logs", "gzip"],
        task_dir=str(tmp / "task"),
        content_hash="c" * 64,
        restore=RestoreCapability.DETERMINISTIC_REPLAY,
    )
    base.update(kw)
    return TaskInstance(**base)


def assistant_msg(call_id: str, name: str, args: dict[str, Any], content: str = "", reasoning: str | None = None, extra_calls: int = 0) -> dict[str, Any]:
    calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, separators=(",", ":"))}}]
    for j in range(extra_calls):
        calls.append({"id": f"{call_id}-x{j}", "type": "function", "function": {"name": "bash", "arguments": '{"command":"true"}'}})
    msg: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": calls}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return msg


def write_source_episode(
    out_dir: Path,
    turns: list[tuple[str, dict[str, Any], str]] | None = None,
    *,
    episode_id: str = "src-ep",
    spec: StateSpec | None = None,
    overrides: dict[int, dict[str, Any]] | None = None,
    missing_usage_turns: tuple[int, ...] = (),
    success: bool = True,
    fingerprints: list[str] | None = None,
) -> tuple[EpisodeSummary, Path]:
    """Write events.jsonl for a successful source episode. `overrides[i]` may set
    content / reasoning / extra_calls / malformed / repaired for turn i."""
    turns = DEFAULT_TURNS if turns is None else turns
    overrides = overrides or {}
    plan = make_plan(episode_id, spec)
    path = out_dir / "events.jsonl"
    log = EventLog(path, episode_id)
    log.emit(EventKind.EPISODE_START, {"plan": plan.model_dump(mode="json")})
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": INSTRUCTION}]
    usages = []
    for i, (name, args, obs) in enumerate(turns + [("", {}, "")]):
        final = i == len(turns)
        fp = fingerprints[i] if fingerprints is not None else f"fp-{i}"
        log.emit(EventKind.FINGERPRINT, {"fingerprint": fp, "detail": []}, turn_index=i)
        log.emit(EventKind.REQUEST, {"request_index": i, "model": "m", "messages": json.loads(json.dumps(messages)), "tools": TOOLS, "sampling": {}, "seed": 7}, turn_index=i)
        o = overrides.get(i, {})
        if final:
            msg: dict[str, Any] = {"role": "assistant", "content": FINAL_ANSWER}
        else:
            msg = assistant_msg(f"call_{i}", name, args, content=o.get("content", ""), reasoning=o.get("reasoning"), extra_calls=o.get("extra_calls", 0))
        u = None if i in missing_usage_turns else usage_for(i)
        usages.append(u or Usage(input_tokens=None, output_tokens=None))
        log.emit(
            EventKind.RESPONSE,
            {"request_index": i, "raw": {"fixture": True}, "history_message": msg, "finish_reason": "stop" if final else "tool_calls", "usage": u.model_dump() if u else None, "latency_sec": 0.1, "seed_sent": 7},
            turn_index=i,
        )
        if o.get("malformed"):
            log.emit(EventKind.PARSE_ERROR, {"request_index": i, "tool_call_id": f"call_{i}", "error": "bad json"}, turn_index=i)
        if o.get("repaired"):
            log.emit(EventKind.REPAIR, {"request_index": i, "tool_call_id": f"call_{i}", "field": "arguments", "original": "{", "replacement": "{}", "reason": "invalid json"}, turn_index=i)
        messages.append(msg)
        if final:
            break
        for j, call in enumerate(msg["tool_calls"]):
            te = ToolExecution(
                call_id=call["id"],
                name=call["function"]["name"],
                requested_arguments_raw=call["function"]["arguments"],
                executed_arguments=json.loads(call["function"]["arguments"]) if not o.get("malformed") else None,
                executed=not o.get("malformed"),
                raw_output=obs if j == 0 else "",
                observation=obs if j == 0 else "",
            )
            log.emit(EventKind.TOOL_CALL, {"tool_call_id": te.call_id, "name": te.name, "requested_arguments_raw": te.requested_arguments_raw, "executed_arguments": te.executed_arguments, "executed": te.executed, "order": j}, turn_index=i)
            log.emit(EventKind.TOOL_RESULT, te.model_dump(), turn_index=i)
            messages.append({"role": "tool", "tool_call_id": te.call_id, "content": te.observation})
    total = Usage.sum(usages)
    log.emit(EventKind.EPISODE_END, {"stop_reason": "model_finished", "stop_category": "model", "usage": total.model_dump(), "n_requests": len(turns) + 1, "n_tool_calls": len(turns), "timing": {}})
    summary = EpisodeSummary(
        episode_id=episode_id,
        role=EpisodeRole.COLLECT,
        instance_id="log-triage/easy/s0",
        attempt_index=0,
        seed=1234,
        checkpoint_id="base",
        stop_reason="model_finished",
        stop_category=StopCategory.MODEL,
        reward={"reward": 1.0 if success else 0.0},
        partial_reward=1.0 if success else 0.0,
        success=success,
        usage=total,
        n_requests=len(turns) + 1,
        n_tool_calls=len(turns),
        events_path=str(path),
    )
    return summary, path


# --------------------------------------------------------------------------- #
# Fake backend
# --------------------------------------------------------------------------- #

Outcome = dict[str, Any]  # {success, requests: [(in, out)|None], stop_category, stop_reason, replay_ok, intervention_tool}


class FakeBackend:
    """Simulates branch episodes: replays nothing real, but writes contract-shaped
    events (replay_action, intervention, model request/response) and returns a
    summary whose outcome is chosen by `outcome_fn(label, seed, plan)`."""

    name = "fake"

    def __init__(self, outcome_fn: Callable[[str, int, EpisodePlan], Outcome], restore: RestoreCapability = RestoreCapability.DETERMINISTIC_REPLAY):
        self.outcome_fn = outcome_fn
        self.restore = restore
        self.calls: list[tuple[EpisodePlan, Path]] = []

    def restore_capability(self, instance: TaskInstance) -> RestoreCapability:
        return self.restore

    async def run(self, instance: TaskInstance, plan: EpisodePlan, out_dir: Path) -> EpisodeResult:
        assert plan.replay is not None
        self.calls.append((plan, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        label = plan.replay.intervention_label
        o = self.outcome_fn(label, plan.seed, plan)
        log = EventLog(out_dir / "events.jsonl", plan.episode_id)
        log.emit(EventKind.EPISODE_START, {"plan": plan.model_dump(mode="json")})
        for t in plan.replay.prefix_turns:
            log.emit(EventKind.REPLAY_ACTION, {"assistant_message": t.assistant_message}, turn_index=t.turn_index)
        replay_ok = o.get("replay_ok", True)
        log.emit(EventKind.REPLAY_CHECK, {"ok": replay_ok, "mismatches": [] if replay_ok else ["observation:turn0:call0"], "normalizers": []})
        k = plan.replay.intervention_turn
        usages = []
        if replay_ok:
            log.emit(EventKind.INTERVENTION, {"label": label, "assistant_message": plan.replay.intervention_message}, turn_index=k)
            for j, req in enumerate(o.get("requests", [])):
                ti = k + 1 + j
                log.emit(EventKind.REQUEST, {"request_index": j, "model": "m", "messages": [], "tools": [], "sampling": {}, "seed": plan.seed}, turn_index=ti)
                u = Usage(input_tokens=req[0], output_tokens=req[1]) if req is not None else Usage(input_tokens=None, output_tokens=None)
                usages.append(u)
                log.emit(EventKind.RESPONSE, {"request_index": j, "raw": {}, "history_message": {"role": "assistant", "content": "done"}, "finish_reason": "stop", "usage": u.model_dump(), "latency_sec": 0.0, "seed_sent": plan.seed}, turn_index=ti)
        cat = StopCategory(o.get("stop_category", "model" if replay_ok else "replay"))
        summary = EpisodeSummary(
            episode_id=plan.episode_id,
            role=plan.role,
            instance_id=instance.instance_id,
            seed=plan.seed,
            checkpoint_id="base",
            stop_reason=o.get("stop_reason", "model_finished" if replay_ok else "replay:observation"),
            stop_category=cat,
            reward={"reward": 1.0 if o.get("success") else 0.0} if replay_ok else None,
            partial_reward=(1.0 if o.get("success") else 0.0) if replay_ok else None,
            success=bool(o.get("success")) if replay_ok else None,
            usage=Usage.sum(usages),
            n_requests=len(usages),
            n_tool_calls=k + 1,
            events_path=str(out_dir / "events.jsonl"),
        )
        if replay_ok:
            # What the episode loop records after executing the fixed action at turn k.
            # Outcome key "intervention_tool": a dict to override, None to omit the record.
            rec = o.get("intervention_tool", {"executed": True, "error": None, "exit_code": 0, "timed_out": False})
            if rec is not None:
                summary.extra["intervention_tool"] = rec
        return EpisodeResult(summary=summary, out_dir=out_dir, replay_ok=replay_ok, replay_mismatches=[] if replay_ok else ["observation:turn0:call0"])
