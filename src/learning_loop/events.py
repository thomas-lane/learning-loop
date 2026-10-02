"""Append-only per-episode event log (`events.jsonl`) and the derived turn view.

Event payload contracts (`Event.data`), by kind:

  episode_start  {"plan": EpisodePlan.model_dump()}
  request        {"request_index", "model", "messages", "tools", "sampling", "seed"}
                 -- the full request exactly as sent (messages as supplied)
  response       {"request_index", "raw", "history_message", "finish_reason",
                  "usage", "latency_sec", "seed_sent"}
  parse_error    {"request_index", "tool_call_id", "error"}
  repair         {"request_index", "tool_call_id", "field", "original", "replacement", "reason"}
  tool_call      {"tool_call_id", "name", "requested_arguments_raw", "executed_arguments",
                  "executed", "order"}
  tool_result    ToolExecution.model_dump()
  fingerprint    {"fingerprint", "detail"}          (before the turn in `turn_index`)
  replay_action  {"assistant_message"}              (fixed prefix turn; no model call)
  replay_check   {"ok", "mismatches", "normalizers"}
  intervention   {"label", "assistant_message"}
  infra_error    {"where", "error"}
  episode_end    {"stop_reason", "stop_category", "usage", "n_requests", "n_tool_calls",
                  "timing"}

`turn_index` on an event is the 0-based assistant-turn index it belongs to.
`load_turns()` reconstructs TurnRecords; it is the only sanctioned reader for
replay and editing, so both see the same view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .records import Event, EventKind, ToolExecution, TurnRecord, Usage
from .storage import JsonlAppender, now_iso, read_jsonl


class EventLog:
    def __init__(self, path: Path, episode_id: str):
        self.path = Path(path)
        self.episode_id = episode_id
        self._out = JsonlAppender(self.path)
        self._seq = len(read_jsonl(self.path))  # continue numbering if reopened

    def emit(self, kind: EventKind, data: dict[str, Any], turn_index: int | None = None) -> Event:
        ev = Event(seq=self._seq, kind=kind, ts=now_iso(), episode_id=self.episode_id, turn_index=turn_index, data=data)
        self._out.append(ev)
        self._seq += 1
        return ev


def read_events(path: Path) -> list[Event]:
    return [Event.model_validate(e) for e in read_jsonl(path)]


def load_turns(events: list[Event] | Path) -> list[TurnRecord]:
    """Derive one TurnRecord per assistant turn, in turn order."""
    if isinstance(events, (str, Path)):
        events = read_events(Path(events))
    turns: dict[int, dict[str, Any]] = {}

    def slot(i: int) -> dict[str, Any]:
        return turns.setdefault(
            i,
            {
                "turn_index": i,
                "request_seq": None,
                "origin": "model",
                "assistant_message": None,
                "finish_reason": None,
                "usage": None,
                "malformed": False,
                "repaired": False,
                "tool_executions": [],
                "fingerprint_before": None,
                "latency_sec": None,
            },
        )

    for ev in events:
        if ev.turn_index is None:
            continue
        t = slot(ev.turn_index)
        d = ev.data
        if ev.kind == EventKind.REQUEST:
            t["request_seq"] = ev.seq
        elif ev.kind == EventKind.RESPONSE:
            t["assistant_message"] = d["history_message"]
            t["finish_reason"] = d.get("finish_reason")
            t["usage"] = Usage.model_validate(d["usage"]) if d.get("usage") else None
            t["latency_sec"] = d.get("latency_sec")
        elif ev.kind == EventKind.PARSE_ERROR:
            t["malformed"] = True
        elif ev.kind == EventKind.REPAIR:
            t["repaired"] = True
        elif ev.kind == EventKind.TOOL_RESULT:
            t["tool_executions"].append(ToolExecution.model_validate(d))
        elif ev.kind == EventKind.FINGERPRINT:
            t["fingerprint_before"] = d["fingerprint"]
        elif ev.kind == EventKind.REPLAY_ACTION:
            t["origin"] = "replayed"
            t["assistant_message"] = d["assistant_message"]
        elif ev.kind == EventKind.INTERVENTION:
            t["origin"] = "intervention_edited" if d["label"] == "edited" else "intervention_original"
            t["assistant_message"] = d["assistant_message"]
    out = []
    for i in sorted(turns):
        t = turns[i]
        if t["assistant_message"] is None:
            continue  # request sent but no response (infra error / timeout)
        out.append(TurnRecord.model_validate(t))
    return out


def first_request_messages(events: list[Event], turn_index: int) -> list[dict[str, Any]] | None:
    """The exact message list sent for the model request of `turn_index`."""
    for ev in events:
        if ev.kind == EventKind.REQUEST and ev.turn_index == turn_index:
            return ev.data["messages"]
    return None
