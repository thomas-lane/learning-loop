"""Repo-defined retrospective editor: trajectory view, proposals, validation.

The editor is a component of this repository, never a developer's
Claude/Codex configuration. It sees only `build_trajectory_view()`:

  - the task instruction (and the learner's own system prompt),
  - the learner's tool schemas,
  - the learner's assistant turns exactly as they entered the history, with
    tool-call ids and the truncated observations the learner was shown,
  - optionally a few scalar outcome metrics (success, partial reward, tokens).

It never sees tests/, solution/, verifier output text, other trajectories,
file paths of run artifacts, or held-out data. It executes no code; it returns
one structured JSON proposal (or abstains). It cannot approve its own edit,
change budgets or the grader, or touch the source environment: verification
(`verify.py`) is a separate step on fresh environments.

`validate_proposal()` runs BEFORE any execution. Its grounding check is a
heuristic (constants that only appear in hindsight are rejected); it is not a
proof that no hindsight leaked into the replacement. Rejected proposals are
kept with every reason for audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import jsonschema
import yaml

from ..core.interfaces import EpisodePlan, Policy
from ..core.records import (
    CheckpointRef,
    EditProposal,
    EpisodeRole,
    EpisodeSummary,
    Event,
    EventKind,
    PolicySpec,
    Message,
    ProposedCall,
    StopCategory,
    TaskInstance,
    ToolSchema,
    TurnRecord,
    Usage,
)
from ..core.seeds import derive_seed, stable_id
from ..episodes.events import load_turns, read_events

if TYPE_CHECKING:
    from ..core.config import EditorConfig

VIEW_VERSION = 1
ASSISTANT_TEXT_POLICIES = ("reject_nonempty",)

# --------------------------------------------------------------------------- #
# Small helpers shared with verify.py / preferences.py
# --------------------------------------------------------------------------- #


def tool_calls_of(message: Message) -> list[dict[str, Any]]:
    return list(message.get("tool_calls") or [])


def parse_call_arguments(call: dict[str, Any]) -> dict[str, Any] | None:
    """Arguments of an OpenAI tool call as a dict; None if not a JSON object."""
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, dict) else None


def serialize_arguments(arguments: dict[str, Any]) -> str:
    """Canonical JSON-string form used for edited tool-call arguments in history."""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def make_edited_message(original: Message, replacement: ProposedCall) -> Message:
    """The fixed edited assistant turn: identical to the original message except
    its single tool call's name/arguments. Content (empty under the strict policy)
    and the original tool_call_id are kept, so the tool-result message that
    follows stays well-formed."""
    calls = tool_calls_of(original)
    if len(calls) != 1:
        raise ValueError("edited messages are only defined for single-tool-call turns")
    edited = json.loads(json.dumps(original))  # deep copy, JSON-safe
    call = edited["tool_calls"][0]
    call.setdefault("type", "function")
    call["function"] = {"name": replacement.name, "arguments": serialize_arguments(replacement.arguments)}
    return edited


def tool_schema_map(tools: list[ToolSchema]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in tools:
        fn = t.get("function") or {}
        if "name" in fn:
            out[fn["name"]] = fn.get("parameters") or {"type": "object"}
    return out


def validate_call(tools: list[ToolSchema], name: str, arguments: Any) -> list[str]:
    """Tool exists and arguments validate against its JSON schema."""
    schemas = tool_schema_map(tools)
    if name not in schemas:
        return [f"unknown_tool:{name}"]
    if not isinstance(arguments, dict):
        return ["schema_violation:arguments are not a JSON object"]
    validator_cls = jsonschema.validators.validator_for(schemas[name])
    errors = sorted(validator_cls(schemas[name]).iter_errors(arguments), key=lambda e: list(e.path))
    return [f"schema_violation:{e.message}" for e in errors]


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield str(obj)


# --------------------------------------------------------------------------- #
# Source trajectories (read through events.jsonl only)
# --------------------------------------------------------------------------- #


def find_events_path(summary: EpisodeSummary, source_dir: str | Path | None = None) -> Path:
    """Locate a source episode's events.jsonl: summary.events_path (absolute, or
    relative to source_dir), else <source_dir>/events.jsonl, else a unique match below it."""
    base = Path(source_dir) if source_dir is not None else None
    if summary.events_path:
        p = Path(summary.events_path)
        if not p.is_absolute() and base is not None:
            p = base / p
        if p.exists():
            return p
    if base is not None:
        if (base / "events.jsonl").exists():
            return base / "events.jsonl"
        found = sorted(q for q in base.rglob("events.jsonl") if ".interrupted-" not in str(q))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise FileNotFoundError(f"{base}: several events.jsonl files; set EpisodeSummary.events_path")
    raise FileNotFoundError(f"no events.jsonl for episode {summary.episode_id}")


@dataclass
class SourceContext:
    """Everything the editor/verifier/dataset builder may read about a source episode."""

    summary: EpisodeSummary
    instance: TaskInstance
    events: list[Event]
    plan: EpisodePlan
    turns: list[TurnRecord] = field(default_factory=list)

    @classmethod
    def load(cls, summary: EpisodeSummary, instance: TaskInstance, events_path: str | Path | None = None) -> "SourceContext":
        path = Path(events_path) if events_path is not None else find_events_path(summary)
        return cls.from_events(summary, instance, read_events(path))

    @classmethod
    def from_dir(cls, source_dir: str | Path, instance: TaskInstance, summary: EpisodeSummary | None = None) -> "SourceContext":
        """A coordinator item dir holding summary.json and the episode's events.jsonl."""
        source_dir = Path(source_dir)
        if summary is None:
            summary = EpisodeSummary.model_validate(json.loads((source_dir / "summary.json").read_text()))
        return cls.from_events(summary, instance, read_events(find_events_path(summary, source_dir)))

    @classmethod
    def from_events(cls, summary: EpisodeSummary, instance: TaskInstance, events: list[Event]) -> "SourceContext":
        start = next((e for e in events if e.kind == EventKind.EPISODE_START), None)
        if start is None or "plan" not in start.data:
            raise ValueError(f"{summary.episode_id}: events lack an episode_start plan")
        return cls(summary, instance, events, EpisodePlan.model_validate(start.data["plan"]), load_turns(events))


# --------------------------------------------------------------------------- #
# Hidden paths and grounding heuristics
# --------------------------------------------------------------------------- #

# Absolute container roots the learner must never touch (Harbor mounts / task dirs).
_HIDDEN_ROOT = re.compile(r"(?<![\w-])/(tests|solution|oracle)(?![\w-])")
_HIDDEN_ANY = [
    re.compile(r"(?<![\w-])/logs/verifier(?![\w-])"),
    re.compile(r"(?<![\w-])/logs/agent(?![\w-])"),
    re.compile(r"\breward\.(txt|json)\b"),
]


def hidden_path_references(arguments: dict[str, Any]) -> list[str]:
    """Sorted unique hidden-path references in any string argument value.

    `/tests` counts only as an absolute root path (or reached via `..`); a
    relative `./tests` or `/app/tests` is a normal workdir path."""
    found: set[str] = set()
    for s in _strings(arguments):
        for m in _HIDDEN_ROOT.finditer(s):
            start = m.start()
            if start >= 1 and s[start - 1] == "." and not (start >= 2 and s[start - 2] == "."):
                continue  # "./tests" is relative to the workdir
            found.add(m.group(0))
        for pat in _HIDDEN_ANY:
            for m in pat.finditer(s):
                found.add(m.group(0))
    return sorted(found)


_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d)")
# Numbers in output contexts: `$2` / `$1` (awk fields, positional args) are code, not data.
_OUT_NUM = re.compile(r"(?<![\w.$])\d+(?:\.\d+)?(?![\w]|\.\d)")
_QUOTED = [re.compile(r"'([^'\n]{3,})'"), re.compile(r'"([^"\n]{3,})"')]
_QUOTED_SHORT_NUM = re.compile(r"""(['"])(\d{1,2})\1""")
_PATH = re.compile(r"(?<![\w.~-])(?:/[\w.\-]+)+/?|(?<![\w/.-])[\w.\-]+(?:/[\w.\-]+)+/?")

# Argument keys whose string value is a program (shell command / script). Their
# small numbers count as constants only in output contexts (see `_output_segments`);
# every other string argument (e.g. write_file `content`) is written verbatim and
# is an output context as a whole.
COMMAND_ARGUMENT_KEYS = frozenset({"command", "cmd", "script", "code"})
# echo/printf in shell command position (not e.g. `print` inside an awk program).
_ECHO_SEGMENT = re.compile(r"(?:^|[;&|({`\n]|\$\(|\b(?:then|do|else)\b)\s*(?:echo|printf)\b([^|;&\n)]*)")
_PRINT_CALL = re.compile(r"(?:\bprint|\.write)\(([^)\n]*)\)")  # python -c "print(17)" / f.write('17')
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)\n\s*\2\b", re.S)


def _keyed_strings(arguments: dict[str, Any]) -> Iterable[tuple[str, str, bool]]:
    """(top-level key, text, is_string_value) for every scalar below each argument."""

    def walk(v: Any) -> Iterable[tuple[str, bool]]:
        if isinstance(v, str):
            yield v, True
        elif isinstance(v, dict):
            for x in v.values():
                yield from walk(x)
        elif isinstance(v, list):
            for x in v:
                yield from walk(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield str(v), False

    for key, value in arguments.items():
        for text, is_str in walk(value):
            yield key, text, is_str


def _output_segments(key: str, text: str) -> list[str]:
    """Parts of an argument that are written/printed verbatim: the whole value of a
    non-command string argument; for commands, the arguments of echo/printf in
    command position (up to the next `|`, `;`, `&`, `)` or newline), of
    `print(...)` / `.write(...)` calls, and here-document bodies. awk `{print $1}` is not one."""
    if key not in COMMAND_ARGUMENT_KEYS:
        return [text]
    segs = [m.group(1) for m in _ECHO_SEGMENT.finditer(text)]
    segs += [m.group(1) for m in _PRINT_CALL.finditer(text)]
    segs += [m.group(3) for m in _HEREDOC.finditer(text)]
    return segs


def _general_constants(s: str) -> set[str]:
    out: set[str] = set(_IP.findall(s))
    masked = _IP.sub(" ", s)
    for n in _NUM.findall(masked):
        if sum(c.isdigit() for c in n) >= 3:
            out.add(n)
    for pat in _QUOTED:
        out |= {q for q in pat.findall(s) if q.strip()}
    for p in _PATH.findall(s):
        p = p.rstrip("/") or p
        if len(p) >= 3 and not _NUM.fullmatch(p):
            out.add(p)
    return out


def _output_constants(s: str) -> set[str]:
    """Like `_general_constants` but numbers of ANY length count (a written count
    such as `echo 17 > /app/answer.txt`), including short quoted numbers."""
    out = _general_constants(s)
    out |= set(_OUT_NUM.findall(_IP.sub(" ", s)))
    out |= {m.group(2) for m in _QUOTED_SHORT_NUM.finditer(s)}
    return out


def _classified_constants(arguments: dict[str, Any]) -> dict[str, bool]:
    """{constant: appears in an output context} for replacement arguments."""
    out: dict[str, bool] = {}
    for key, s, is_text in _keyed_strings(arguments):
        for c in _general_constants(s):
            out.setdefault(c, False)
        if is_text:
            for seg in _output_segments(key, s):
                for c in _output_constants(seg):
                    out[c] = True
    return out


def extract_constants(arguments: dict[str, Any]) -> list[str]:
    """Constants in replacement arguments that could encode hindsight. Sorted, unique:

    - IPv4 addresses, numbers with >= 3 digits, quoted literals (>= 3 chars) and
      file paths (>= 3 chars) anywhere in the arguments;
    - numbers of ANY length (1-2 digit counts included) that are written or
      printed: echo/printf arguments, print(...) calls and here-doc bodies in commands, and
      the whole value of non-command string arguments (e.g. write_file content).
      `$N` field references are not numbers. Small numbers elsewhere in a command
      (`head -1`, `sort -k2`) are not extracted, so they are never checked."""
    return sorted(_classified_constants(arguments))


def _is_numeric(c: str) -> bool:
    return bool(_IP.fullmatch(c) or _NUM.fullmatch(c))


def _boundary_pattern(c: str) -> re.Pattern[str]:
    """Token-boundary match: `123` does not match inside `1234` or `0.123`,
    `10.0.0.9` not inside `10.0.0.99`, `/app/x` not inside `/app/x.csv`."""
    if _is_numeric(c):
        return re.compile(r"(?<![\w.])" + re.escape(c) + r"(?![\w]|\.\d)")
    left = r"(?<![\w.~-])" if (c[0].isalnum() or c[0] in "_/") else ""
    right = r"(?![\w-]|\.\w)" if (c[-1].isalnum() or c[-1] == "_") else ""
    return re.compile(left + re.escape(c) + right)


def contains_token(text: str, constant: str) -> bool:
    return _boundary_pattern(constant).search(text) is not None


def _message_text(message: Message) -> str:
    return "\n".join([_message_prose(message), _message_calls_text(message)])


def _message_prose(message: Message) -> str:
    parts = []
    for key in ("content", "reasoning_content"):
        v = message.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in v)
    return "\n".join(parts)


def _message_calls_text(message: Message) -> str:
    parts = []
    for call in tool_calls_of(message):
        fn = call.get("function") or {}
        parts.append(str(fn.get("name", "")))
        args = fn.get("arguments")
        parts.append(args if isinstance(args, str) else json.dumps(args))
        parsed = parse_call_arguments(call)
        if parsed is not None:
            parts.extend(_strings(parsed))  # unescaped values too
    return "\n".join(parts)


def grounding_violations(
    replacement_arguments: dict[str, Any],
    *,
    turns: list[TurnRecord],
    turn_index: int,
    instruction: str,
    tools: list[ToolSchema],
    system_prompt: str | None = None,
) -> list[str]:
    """`ungrounded_constant:<value>` for each constant that appears AFTER the
    decision but nowhere in the information available at the decision (system
    prompt, instruction, tool schemas, turns before k with their observations,
    and the original action at turn k itself). All matching is on token
    boundaries (`123` is not grounded by `1234`, nor `10.0.0.9` by `10.0.0.99`).

    "After the decision" = observations of turn k and later, later assistant
    text (e.g. the final answer) and, for numbers/IPs and for anything the
    replacement writes or prints, later turns' tool-call arguments too (an answer
    the learner computed in its head and only wrote in a later call). Quoted
    literals and paths elsewhere in a command are not checked against later tool
    calls: re-using the learner's own later procedure (an awk program, a path it
    guessed) earlier is the intended kind of edit.

    Heuristic only. False negatives: a constant computed at runtime by the
    replacement leaves no literal trace; paraphrased hindsight is not detected;
    small numbers outside output contexts are not checked. False positives
    (accepted, documented): a small written number that happens to appear as a
    token after the decision but not before is rejected even if it was not
    hindsight (e.g. `echo 0 > f` when a later observation shows `0`)."""
    allowed_parts = [instruction, system_prompt or "", json.dumps(tools)]
    # Written/printed data values (numbers, IPs) are the answer-hardcoding risk. A small number
    # or an IP almost always occurs *somewhere* in a log-heavy prefix, so for these the prefix
    # observations only ground a value the learner was actually shown as a result: a whole
    # observation line (e.g. `wc -l` printing `10`). Coincidental occurrences inside longer lines
    # (dates, sizes, other log fields) do not count.
    output_allowed_parts = [instruction, system_prompt or "", json.dumps(tools)]
    result_lines: set[str] = set()
    later_obs: list[str] = []
    later_calls: list[str] = []
    for t in turns:
        if t.turn_index < turn_index:
            allowed_parts.append(_message_text(t.assistant_message))
            allowed_parts.extend(te.observation for te in t.tool_executions)
            output_allowed_parts.append(_message_text(t.assistant_message))
            for te in t.tool_executions:
                result_lines.update(ln.strip() for ln in te.observation.splitlines() if ln.strip())
        elif t.turn_index == turn_index:
            output_allowed_parts.append(_message_text(t.assistant_message))
            allowed_parts.append(_message_text(t.assistant_message))
            later_obs.extend(te.observation for te in t.tool_executions)
        else:
            later_obs.extend(te.observation for te in t.tool_executions)
            later_obs.append(_message_prose(t.assistant_message))
            later_calls.append(_message_calls_text(t.assistant_message))
    allowed = "\n".join(allowed_parts)
    later_narrow = "\n".join(later_obs)
    later_full = "\n".join([later_narrow, *later_calls])
    output_allowed = "\n".join(output_allowed_parts)
    out = []
    for c, is_output in sorted(_classified_constants(replacement_arguments).items()):
        later = later_full if (is_output or _is_numeric(c)) else later_narrow
        if not contains_token(later, c):
            continue
        if is_output and _is_numeric(c):
            grounded = contains_token(output_allowed, c) or c in result_lines
        else:
            grounded = contains_token(allowed, c)
        if not grounded:
            out.append(f"ungrounded_constant:{c}")
    return out


# Arguments that set an execution budget. The editor cannot change budgets
# (spec section 7): a replacement must keep the original call's value (absent
# stays absent), whatever the tool.
BUDGET_ARGUMENT_KEYS = ("timeout_sec", "timeout", "max_output_chars", "max_tokens", "max_turns")


def budget_violations(original_call: dict[str, Any] | None, replacement_arguments: dict[str, Any]) -> list[str]:
    """`budget_argument_changed:<key>` for every budget-like argument whose value
    differs from the original call's (setting one the original did not set counts)."""
    orig = (parse_call_arguments(original_call) if original_call is not None else None) or {}
    return [
        f"budget_argument_changed:{k}"
        for k in BUDGET_ARGUMENT_KEYS
        if (k in orig or k in replacement_arguments) and orig.get(k) != replacement_arguments.get(k)
    ]


# --------------------------------------------------------------------------- #
# Turn eligibility and proposal validation
# --------------------------------------------------------------------------- #


def turn_ineligibility(turn: TurnRecord, assistant_text_policy: str = "reject_nonempty") -> list[str]:
    """Reasons a turn cannot be edited (empty list = eligible)."""
    if assistant_text_policy not in ASSISTANT_TEXT_POLICIES:
        raise ValueError(f"unsupported assistant_text_policy {assistant_text_policy!r}")
    reasons: list[str] = []
    if turn.origin != "model":
        reasons.append(f"not_model_turn:{turn.origin}")
    calls = tool_calls_of(turn.assistant_message)
    if len(calls) == 0:
        reasons.append("no_tool_call_turn")
    elif len(calls) > 1:
        reasons.append(f"multi_tool_call_turn:{len(calls)}")
    if turn.malformed:
        reasons.append("malformed_turn")
    if turn.repaired:
        reasons.append("repaired_turn")
    if len(calls) == 1:
        if parse_call_arguments(calls[0]) is None:
            reasons.append("malformed_turn:arguments")
        execs = [te for te in turn.tool_executions if te.call_id == calls[0].get("id")]
        if not execs or not execs[0].executed:
            reasons.append("original_not_executed")
    content = turn.assistant_message.get("content")
    if (isinstance(content, str) and content.strip()) or (isinstance(content, list) and len(content) > 0):
        reasons.append("nonempty_assistant_content")
    reasoning = turn.assistant_message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        reasons.append("nonempty_reasoning")
    return reasons


def validate_proposal(
    proposal: EditProposal,
    *,
    turns: list[TurnRecord],
    tools: list[ToolSchema],
    instruction: str,
    source_episode_id: str,
    system_prompt: str | None = None,
    assistant_text_policy: str = "reject_nonempty",
) -> list[str]:
    """Every reason the proposal must not be executed (empty = valid)."""
    reasons: list[str] = []
    if proposal.source_episode_id != source_episode_id:
        reasons.append("source_trajectory_mismatch")
    if proposal.turn_index is None or proposal.replacement is None or proposal.tool_call_id is None:
        return reasons + ["incomplete_proposal"]
    by_index = {t.turn_index: t for t in turns}
    turn = by_index.get(proposal.turn_index)
    if turn is None:
        return reasons + [f"turn_not_found:{proposal.turn_index}"]
    reasons += turn_ineligibility(turn, assistant_text_policy)
    calls = tool_calls_of(turn.assistant_message)
    if len(calls) == 1 and calls[0].get("id") != proposal.tool_call_id:
        reasons.append("tool_call_id_mismatch")
    rep = proposal.replacement
    reasons += validate_call(tools, rep.name, rep.arguments)
    if len(calls) == 1:
        orig_name = (calls[0].get("function") or {}).get("name")
        if orig_name == rep.name and parse_call_arguments(calls[0]) == rep.arguments:
            reasons.append("identical_replacement")
    reasons += [f"hidden_path:{p}" for p in hidden_path_references(rep.arguments)]
    if isinstance(rep.arguments, dict):
        reasons += budget_violations(calls[0] if len(calls) == 1 else None, rep.arguments)
    reasons += grounding_violations(
        rep.arguments,
        turns=turns,
        turn_index=proposal.turn_index,
        instruction=instruction,
        tools=tools,
        system_prompt=system_prompt,
    )
    return reasons


def apply_validation(proposal: EditProposal, **kwargs: Any) -> EditProposal:
    """Validate a `proposed` proposal; mark it invalid with all reasons if needed."""
    if proposal.status != "proposed":
        return proposal
    reasons = validate_proposal(proposal, **kwargs)
    if reasons:
        return proposal.model_copy(update={"status": "invalid", "rejection_reasons": proposal.rejection_reasons + reasons})
    return proposal


def select_sources(summaries: list[EpisodeSummary]) -> list[EpisodeSummary]:
    """Initial preferences target efficiency on successful collection episodes."""
    return [
        s for s in summaries if s.role == EpisodeRole.COLLECT and s.success is True and s.stop_category == StopCategory.MODEL
    ]


# --------------------------------------------------------------------------- #
# The editor's view
# --------------------------------------------------------------------------- #


def build_trajectory_view(
    turns: list[TurnRecord],
    instruction: str,
    tools: list[ToolSchema],
    outcome: EpisodeSummary | dict[str, Any] | None = None,
    include_later_observations: bool = True,
    *,
    source_trajectory_id: str,
    system_prompt: str | None = None,
    assistant_text_policy: str = "reject_nonempty",
) -> dict[str, Any]:
    """The only data an editor receives. JSON-serializable.

    include_later_observations=False: the view stops at the earliest eligible
    turn (the learner's information state there, plus its original action) and
    only that turn is eligible; nothing observed after it is shown.
    """
    eligible_idx = [t.turn_index for t in turns if not turn_ineligibility(t, assistant_text_policy)]
    cutoff = None if include_later_observations or not eligible_idx else min(eligible_idx)
    view_turns = []
    for t in turns:
        if cutoff is not None and t.turn_index > cutoff:
            break
        msg = t.assistant_message
        entry: dict[str, Any] = {
            "turn_index": t.turn_index,
            "assistant": {k: msg[k] for k in ("content", "reasoning_content", "tool_calls") if k in msg},
        }
        if cutoff is None or t.turn_index < cutoff:
            entry["observations"] = [{"tool_call_id": te.call_id, "observation": te.observation} for te in t.tool_executions]
        why_not = turn_ineligibility(t, assistant_text_policy)
        entry["eligible"] = not why_not and (cutoff is None or t.turn_index == cutoff)
        if why_not:
            entry["ineligible_reasons"] = why_not
        view_turns.append(entry)
    view: dict[str, Any] = {
        "view_version": VIEW_VERSION,
        "source_trajectory_id": source_trajectory_id,
        "instruction": instruction,
        "system_prompt": system_prompt,
        "tools": tools,
        "turns": view_turns,
        "later_observations_included": cutoff is None,
    }
    if outcome is not None:
        if isinstance(outcome, EpisodeSummary):
            outcome = {
                "success": outcome.success,
                "partial_reward": outcome.partial_reward,
                "total_tokens": outcome.usage.total,
                "n_requests": outcome.n_requests,
                "n_tool_calls": outcome.n_tool_calls,
            }
        view["outcome"] = {k: v for k, v in outcome.items() if v is None or isinstance(v, (int, float, bool))}
    return view


# --------------------------------------------------------------------------- #
# Editors
# --------------------------------------------------------------------------- #


def editor_identity(mode: str, checkpoint_id: str | None, prompt_sha256: str | None, decoding: dict[str, Any]) -> str:
    return stable_id("editor", mode, checkpoint_id, prompt_sha256, decoding)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_editor_response(text: str) -> dict[str, Any]:
    """Extract the single JSON object from the editor's reply (fences tolerated)."""
    text = (text or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in editor response") from None
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("editor response is not a JSON object")
    return obj


def proposal_from_response(
    obj: dict[str, Any],
    *,
    proposal_id: str,
    source_episode_id: str,
    instance_id: str,
    editor_id: str,
) -> EditProposal:
    """Map the structured response onto an EditProposal (status proposed/abstained/invalid)."""
    base = dict(proposal_id=proposal_id, source_episode_id=source_episode_id, instance_id=instance_id, editor_id=editor_id)
    decision = obj.get("decision")
    justification = obj.get("justification") if isinstance(obj.get("justification"), str) else None
    if decision == "abstain":
        return EditProposal(**base, status="abstained", justification=justification)
    if decision != "edit":
        return EditProposal(**base, status="invalid", justification=justification, rejection_reasons=[f"response_schema:decision={decision!r}"])
    reasons = []
    if obj.get("source_trajectory_id") not in (None, source_episode_id):
        reasons.append("source_trajectory_mismatch")
    turn_index = obj.get("turn_index")
    if not isinstance(turn_index, int) or isinstance(turn_index, bool):
        reasons.append("response_schema:turn_index")
        turn_index = None
    call_id = obj.get("tool_call_id")
    if not isinstance(call_id, str):
        reasons.append("response_schema:tool_call_id")
        call_id = None
    rep = obj.get("replacement")
    replacement = None
    if isinstance(rep, dict) and isinstance(rep.get("name"), str):
        args = rep.get("arguments")
        if isinstance(args, str):  # tolerate a JSON-string encoding, visibly
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if isinstance(args, dict):
            replacement = ProposedCall(name=rep["name"], arguments=args)
    if replacement is None:
        reasons.append("response_schema:replacement")
    return EditProposal(
        **base,
        status="invalid" if reasons else "proposed",
        turn_index=turn_index,
        tool_call_id=call_id,
        replacement=replacement,
        justification=justification,
        rejection_reasons=reasons,
    )


class _EditorBase:
    editor_id: str
    assistant_text_policy: str = "reject_nonempty"

    async def propose(self, source: EpisodeSummary, turns: list[TurnRecord], instance: TaskInstance, instruction: str, tools: list[ToolSchema], **kw: Any) -> EditProposal:
        raise NotImplementedError

    async def propose_for(
        self,
        *,
        proposal_id: str,
        source: EpisodeSummary,
        source_dir: str | Path,
        instance: TaskInstance,
        seed: int | None = None,
    ) -> EditProposal:
        """Coordinator entry point: read the source trajectory from its item dir
        (events.jsonl only; instruction/tools/system prompt from its plan)."""
        ctx = SourceContext.from_dir(source_dir, instance, summary=source)
        return await self.propose(
            source,
            ctx.turns,
            instance,
            ctx.plan.instruction,
            ctx.plan.tools,
            system_prompt=ctx.plan.system_prompt,
            proposal_id=proposal_id,
            seed=seed,
        )

    def _validate(self, proposal: EditProposal, turns: list[TurnRecord], tools: list[ToolSchema], instruction: str, source: EpisodeSummary, system_prompt: str | None) -> EditProposal:
        return apply_validation(
            proposal,
            turns=turns,
            tools=tools,
            instruction=instruction,
            source_episode_id=source.episode_id,
            system_prompt=system_prompt,
            assistant_text_policy=self.assistant_text_policy,
        )


class LLMEditor(_EditorBase):
    """One request per source trajectory against a served model (policy)."""

    def __init__(
        self,
        policy: Policy,
        *,
        mode: str,
        checkpoint_id: str | None,
        prompt_path: str | Path,
        root_seed: int = 0,
        include_outcome_metrics: bool = True,
        include_later_observations: bool = True,
        assistant_text_policy: str = "reject_nonempty",
        proposals_per_source: int = 1,
    ):
        self.policy = policy
        self.mode = mode
        self.checkpoint_id = checkpoint_id
        self.prompt_text = Path(prompt_path).read_text()
        self.prompt_sha256 = hashlib.sha256(self.prompt_text.encode()).hexdigest()
        self.prompt_path = str(prompt_path)
        self.root_seed = root_seed
        self.include_outcome_metrics = include_outcome_metrics
        self.include_later_observations = include_later_observations
        self.assistant_text_policy = assistant_text_policy
        self.proposals_per_source = proposals_per_source
        self.decoding = {
            "sampling": policy.spec.sampling.model_dump(),
            "send_seed": policy.spec.send_seed,
            "view": {
                "version": VIEW_VERSION,
                "outcome": include_outcome_metrics,
                "later_observations": include_later_observations,
                "assistant_text_policy": assistant_text_policy,
            },
            "proposals_per_source": proposals_per_source,
        }
        self.editor_id = editor_identity(mode, checkpoint_id, self.prompt_sha256, self.decoding)

    def request_messages(self, view: dict[str, Any]) -> list[Message]:
        return [
            {"role": "system", "content": self.prompt_text},
            {"role": "user", "content": json.dumps(view, ensure_ascii=False, indent=1)},
        ]

    async def propose(
        self,
        source: EpisodeSummary,
        turns: list[TurnRecord],
        instance: TaskInstance,
        instruction: str,
        tools: list[ToolSchema],
        *,
        system_prompt: str | None = None,
        proposal_index: int = 0,
        proposal_id: str | None = None,
        seed: int | None = None,
    ) -> EditProposal:
        view = build_trajectory_view(
            turns,
            instruction,
            tools,
            outcome=source if self.include_outcome_metrics else None,
            include_later_observations=self.include_later_observations,
            source_trajectory_id=source.episode_id,
            system_prompt=system_prompt,
            assistant_text_policy=self.assistant_text_policy,
        )
        proposal_id = proposal_id or stable_id("prop", self.editor_id, source.episode_id, proposal_index)
        messages = self.request_messages(view)
        if seed is None:
            seed = derive_seed(self.root_seed, "editor_proposal", self.editor_id, source.episode_id, proposal_index)
        # The editor answers in JSON text; it gets no tools (it executes nothing).
        decision = await self.policy.decide(messages, [], seed)
        raw = {
            "request_messages": messages,
            "seed": seed,
            "seed_sent": decision.seed_sent,
            "response": decision.raw_response,
            "history_message": decision.history_message,
            "finish_reason": decision.finish_reason,
            "request_error": decision.request_error,
            "infra_error": decision.infra_error,
        }
        common = dict(proposal_id=proposal_id, source_episode_id=source.episode_id, instance_id=instance.instance_id, editor_id=self.editor_id)
        if decision.infra_error or decision.request_error:
            err = f"editor_infra_error:{decision.infra_error}" if decision.infra_error else f"editor_request_error:{decision.request_error}"
            return EditProposal(**common, status="invalid", rejection_reasons=[err], raw_response=raw, usage=decision.usage, duration_sec=decision.latency_sec)
        text = decision.history_message.get("content") or ""
        try:
            obj = parse_editor_response(text if isinstance(text, str) else json.dumps(text))
        except (ValueError, json.JSONDecodeError) as e:
            reasons = [f"unparseable_response:{e}"]
            if decision.finish_reason == "length":
                reasons.append("editor_output_truncated")
            return EditProposal(**common, status="invalid", rejection_reasons=reasons, raw_response=raw, usage=decision.usage, duration_sec=decision.latency_sec)
        proposal = proposal_from_response(obj, **common).model_copy(update={"raw_response": raw, "usage": decision.usage, "duration_sec": decision.latency_sec})
        return self._validate(proposal, turns, tools, instruction, source, system_prompt)


class ScriptedEditor(_EditorBase):
    """Fixture editor: proposals come from a YAML/JSON file.

    File format::

        editor_id: scripted-fixture-v1        # optional
        proposals:
          - instance_id: fixture/easy/s0      # or `family: fixture` to match every instance of a family
            attempt_index: 0                  # optional; matches any attempt if absent
            decision: edit                    # edit | abstain (default edit)
            turn_index: 0
            tool_call_id: call_0              # optional; defaults to the turn's call id
            replacement: {name: bash, arguments: {command: "..."}}
            justification: "..."

    The first matching entry is used; no entry => abstain. Proposals are still validated.
    """

    def __init__(self, path: str | Path, assistant_text_policy: str = "reject_nonempty"):
        self.path = Path(path)
        text = self.path.read_text()
        data = yaml.safe_load(text) or {}
        self.entries: list[dict[str, Any]] = list(data.get("proposals") or [])
        sha = hashlib.sha256(text.encode()).hexdigest()
        self.assistant_text_policy = assistant_text_policy
        self.editor_id = data.get("editor_id") or editor_identity("scripted", None, sha, {})
        self.mode = "scripted"

    def _entry(self, source: EpisodeSummary, instance: TaskInstance) -> dict[str, Any] | None:
        for e in self.entries:
            if "instance_id" in e and e["instance_id"] != source.instance_id:
                continue
            if "family" in e and e["family"] != instance.family:
                continue
            if "attempt_index" in e and e["attempt_index"] != source.attempt_index:
                continue
            return e
        return None

    async def propose(
        self,
        source: EpisodeSummary,
        turns: list[TurnRecord],
        instance: TaskInstance,
        instruction: str,
        tools: list[ToolSchema],
        *,
        system_prompt: str | None = None,
        proposal_index: int = 0,
        proposal_id: str | None = None,
        seed: int | None = None,
    ) -> EditProposal:
        common = dict(
            proposal_id=proposal_id or stable_id("prop", self.editor_id, source.episode_id, proposal_index),
            source_episode_id=source.episode_id,
            instance_id=instance.instance_id,
            editor_id=self.editor_id,
        )
        no_model = Usage(input_tokens=0, output_tokens=0, source="none")  # no model call was made
        e = self._entry(source, instance)
        if e is None:
            return EditProposal(**common, status="abstained", justification="no scripted proposal", usage=no_model)
        obj = {k: v for k, v in e.items() if k not in ("instance_id", "family", "attempt_index")}
        obj.setdefault("decision", "edit")
        obj.setdefault("source_trajectory_id", source.episode_id)
        if obj["decision"] == "edit" and "tool_call_id" not in obj:
            turn = next((t for t in turns if t.turn_index == obj.get("turn_index")), None)
            calls = tool_calls_of(turn.assistant_message) if turn else []
            obj["tool_call_id"] = calls[0].get("id") if len(calls) == 1 else None
        proposal = proposal_from_response(obj, **common).model_copy(update={"raw_response": {"scripted_entry": e, "script": str(self.path)}, "usage": no_model})
        return self._validate(proposal, turns, tools, instruction, source, system_prompt)


def make_editor(
    cfg: "EditorConfig",
    checkpoint: CheckpointRef | None,
    policy_spec: PolicySpec | None,
    *,
    policy: Policy | None = None,
    root_seed: int = 0,
) -> LLMEditor | ScriptedEditor:
    """Build the configured editor. `checkpoint` is the editor's own (fixed) model
    identity, e.g. the initial policy for mode=initial_policy; it enters editor_id."""
    from ..core.config import repo_path

    if cfg.mode == "scripted":
        assert cfg.scripted_path is not None
        return ScriptedEditor(repo_path(cfg.scripted_path), assistant_text_policy=cfg.assistant_text_policy)
    if policy is None:
        if policy_spec is None:
            raise ValueError(f"editor mode {cfg.mode} needs a policy spec")
        from ..episodes.policy import make_policy

        policy = make_policy(policy_spec)
    return LLMEditor(
        policy,
        mode=cfg.mode,
        checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
        prompt_path=repo_path(cfg.prompt),
        root_seed=root_seed,
        include_outcome_metrics=cfg.include_outcome_metrics,
        include_later_observations=cfg.include_later_observations,
        assistant_text_policy=cfg.assistant_text_policy,
        proposals_per_source=cfg.proposals_per_source,
    )
