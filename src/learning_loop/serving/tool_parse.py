"""Parse a model's native completion text into an OpenAI-style assistant message.

Only formats that have a tested renderer/parser pair are supported; others raise.

qwen3_xml:  [<think>reasoning</think>] content <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call> ...
            Arguments are re-serialized with json.dumps (default separators), which is
            exactly how the Qwen3 template's `tojson` renders a parsed history object, so a
            parsed call round-trips to the same prompt text on the next turn.
            A block that is not valid JSON (or lacks name/arguments object) is NOT turned into
            a tool call: its text stays in `content` and the error is reported, so the caller
            sees a malformed turn instead of a silently repaired one.

gemma4:     [<|channel>thought ...<channel|>] content <|tool_call>call:NAME{key:value,...}<tool_call|> ...
            The generation stops at <|tool_response> (an end-of-turn id), which the server strips.
            Values use the template's own syntax: strings between <|"|> delimiters, bare numbers,
            true/false, null/None, {...} objects (keys bare or <|"|>-quoted) and [...] lists. A
            block that does not parse completely stays in `content` with the error reported.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

_QWEN_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_QWEN_OPEN_UNCLOSED = re.compile(r"<tool_call>(?![\s\S]*</tool_call>)", re.S)


@dataclass
class ParsedCompletion:
    content: str
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    def message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


def _call_id(seed: str, index: int) -> str:
    return "call_" + hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:24]


def parse_qwen3(text: str, id_seed: str) -> ParsedCompletion:
    reasoning = None
    body = text
    if "</think>" in body:
        head, _, body = body.partition("</think>")
        reasoning = head.split("<think>", 1)[-1].strip("\n")
        body = body.lstrip("\n")
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    kept: list[str] = []
    pos = 0
    for m in _QWEN_CALL.finditer(body):
        kept.append(body[pos : m.start()])
        pos = m.end()
        raw = m.group(1)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                raise ValueError("tool call must be an object with a string 'name'")
            args = obj.get("arguments", {})
            if isinstance(args, str):  # some models emit a JSON string; accept only if it holds an object
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("'arguments' must be a JSON object")
        except (ValueError, json.JSONDecodeError) as e:
            errors.append(f"tool_call {len(calls) + len(errors)}: {e}")
            kept.append(m.group(0))  # leave the malformed block visible in content
            continue
        calls.append(
            {
                "id": _call_id(id_seed, len(calls)),
                "type": "function",
                "function": {"name": obj["name"], "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    kept.append(body[pos:])
    content = "".join(kept)
    if _QWEN_OPEN_UNCLOSED.search(content):
        errors.append("unterminated <tool_call> block")
    return ParsedCompletion(content=content.strip(), reasoning_content=reasoning, tool_calls=calls, parse_errors=errors)


_GEMMA_QUOTE = '<|"|>'
_GEMMA_CALL = re.compile(r"<\|tool_call>(.*?)<tool_call\|>", re.S)
_GEMMA_THOUGHT = re.compile(r"<\|channel>(.*?)<channel\|>", re.S)
_GEMMA_NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
_GEMMA_KEY = re.compile(r"[A-Za-z_][\w.\-]*")
_GEMMA_LITERALS = {"true": True, "false": False, "True": True, "False": False, "null": None, "None": None}


class _GemmaArgs:
    """Recursive-descent parser for one Gemma 4 argument object; the whole text must be consumed."""

    def __init__(self, text: str):
        self.t, self.i = text, 0

    def fail(self, what: str) -> ValueError:
        return ValueError(f"{what} at offset {self.i}: {self.t[self.i:self.i + 30]!r}")

    def value(self) -> Any:
        t = self.t
        if t.startswith(_GEMMA_QUOTE, self.i):
            end = t.find(_GEMMA_QUOTE, self.i + len(_GEMMA_QUOTE))
            if end < 0:
                raise self.fail("unterminated string")
            out, self.i = t[self.i + len(_GEMMA_QUOTE) : end], end + len(_GEMMA_QUOTE)
            return out
        if t.startswith("{", self.i):
            return self.obj()
        if t.startswith("[", self.i):
            self.i += 1
            items: list[Any] = []
            if t.startswith("]", self.i):
                self.i += 1
                return items
            while True:
                items.append(self.value())
                if t.startswith(",", self.i):
                    self.i += 1
                elif t.startswith("]", self.i):
                    self.i += 1
                    return items
                else:
                    raise self.fail("expected ',' or ']'")
        for lit, v in _GEMMA_LITERALS.items():
            if t.startswith(lit, self.i) and not t[self.i + len(lit) : self.i + len(lit) + 1].isalnum():
                self.i += len(lit)
                return v
        m = _GEMMA_NUMBER.match(t, self.i)
        if m:
            self.i = m.end()
            return float(m.group()) if (m.group(1) or m.group(2)) else int(m.group())
        raise self.fail("expected a value")

    def obj(self) -> dict[str, Any]:
        t = self.t
        if not t.startswith("{", self.i):
            raise self.fail("expected '{'")
        self.i += 1
        out: dict[str, Any] = {}
        if t.startswith("}", self.i):
            self.i += 1
            return out
        while True:
            if t.startswith(_GEMMA_QUOTE, self.i):
                key = self.value()
            else:
                m = _GEMMA_KEY.match(t, self.i)
                if not m:
                    raise self.fail("expected a key")
                key, self.i = m.group(), m.end()
            if not t.startswith(":", self.i):
                raise self.fail("expected ':'")
            self.i += 1
            out[key] = self.value()
            if t.startswith(",", self.i):
                self.i += 1
            elif t.startswith("}", self.i):
                self.i += 1
                return out
            else:
                raise self.fail("expected ',' or '}'")


def _gemma_call(block: str) -> tuple[str, dict[str, Any]]:
    if not block.startswith("call:"):
        raise ValueError("tool call must start with 'call:'")
    brace = block.find("{")
    name = block[len("call:") : brace] if brace >= 0 else ""
    if not name or not _GEMMA_KEY.fullmatch(name):
        raise ValueError(f"invalid tool name {name!r}")
    p = _GemmaArgs(block[brace:])
    args = p.obj()
    if p.i != len(p.t):
        raise p.fail("trailing text after arguments")
    return name, args


def parse_gemma4(text: str, id_seed: str) -> ParsedCompletion:
    thoughts = [m.group(1) for m in _GEMMA_THOUGHT.finditer(text)]
    reasoning = "\n".join(t.removeprefix("thought").strip("\n") for t in thoughts) if thoughts else None
    body = _GEMMA_THOUGHT.sub("", text)
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    kept: list[str] = []
    pos = 0
    for m in _GEMMA_CALL.finditer(body):
        kept.append(body[pos : m.start()])
        pos = m.end()
        try:
            name, args = _gemma_call(m.group(1))
        except ValueError as e:
            errors.append(f"tool_call {len(calls) + len(errors)}: {e}")
            kept.append(m.group(0))
            continue
        calls.append(
            {
                "id": _call_id(id_seed, len(calls)),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    kept.append(body[pos:])
    content = "".join(kept)
    if re.search(r"<\|tool_call>(?![\s\S]*<tool_call\|>)", content):
        errors.append("unterminated <|tool_call> block")
    return ParsedCompletion(content=content.strip(), reasoning_content=reasoning, tool_calls=calls, parse_errors=errors)


PARSERS = {"qwen3_xml": parse_qwen3, "gemma4": parse_gemma4}


def parse_bare_call(text: str, names: set[str]) -> tuple[str, dict[str, Any]] | None:
    """A reply that is exactly one tool call written without the format's call markers, to one of
    `names`: Gemma style `name{key:value,...}` (optionally prefixed `call:`) or `name {json}`.
    Used only for the editor's answer (its replies are proposals, not learner turns); returns None
    unless the whole text is consumed."""
    t = (text or "").strip()
    t = t.removeprefix("<|tool_call>").removesuffix("<tool_call|>").removesuffix("<").strip()
    t = t.removeprefix("call:")
    brace = t.find("{")
    name = t[:brace].strip() if brace > 0 else ""
    if name not in names:
        return None
    try:
        return _gemma_call("call:" + name + t[brace:])
    except ValueError:
        pass
    try:
        args = json.loads(t[brace:])
    except json.JSONDecodeError:
        return None
    return (name, args) if isinstance(args, dict) else None


def parse_completion(tool_call_format: str, text: str, id_seed: str) -> ParsedCompletion:
    if tool_call_format in PARSERS:
        return PARSERS[tool_call_format](text, id_seed)
    raise NotImplementedError(f"tool_call_format {tool_call_format!r} has no tested parser in hf_server")
