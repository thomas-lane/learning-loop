"""Policies: an OpenAI-compatible endpoint, and a scripted fixture policy.

Both return `PolicyDecision`s built by the same normalization code, so parse
errors and repairs are handled identically:

- The history message is `{"role": "assistant", "content": <str>, ["reasoning_content"],
  ["tool_calls"]}`. Absent content is represented as "" (canonical form, not a
  repair). `reasoning_content` is copied verbatim when the provider returned it
  and never fabricated.
- Tool-call arguments that are not a JSON object are a parse error for that
  call (`call_errors`), and the call is not executed. Because llama.cpp
  re-parses the arguments of past calls to render the chat template, such
  arguments are replaced by "{}" in the history; that replacement is recorded
  as a visible repair. A missing tool-call id is replaced by a deterministic id
  (also a repair).
- Endpoint failures: 4xx (except 408/429) -> `request_error` (e.g. context
  overflow; the episode stops as a model error). Connection errors, timeouts,
  408/429 and 5xx -> `infra_error`. The client does not retry: a retry would
  be invisible, and infrastructure retries belong to the coordinator.
- `PolicySpec.request_timeout_sec` is the client timeout per request (default
  600 s). `PolicySpec.max_concurrent_requests` bounds in-flight requests per
  endpoint with a process-wide semaphore keyed by (api_base, limit) and event
  loop; latency is measured after the semaphore is acquired.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import weakref
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .interfaces import Policy, PolicyDecision
from .records import Message, PolicySpec, ToolSchema, Usage

REPAIR_INVALID_ARGS_REASON = (
    "arguments were not a JSON object; replaced by '{}' in the history because "
    "llama.cpp re-parses past tool-call arguments when rendering the chat template"
)


# --------------------------------------------------------------------------- #
# Shared normalization
# --------------------------------------------------------------------------- #


def usage_from_payload(u: dict[str, Any] | None) -> Usage:
    if not u:
        return Usage(source="none")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return Usage(
        input_tokens=u.get("prompt_tokens"),
        output_tokens=u.get("completion_tokens"),
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
        source="provider",
    )


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """(arguments, error). Arguments must be a JSON object encoded as a string."""
    if not isinstance(raw, str):
        return None, f"arguments must be a JSON string, got {type(raw).__name__}"
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"could not parse arguments as JSON: {e}"
    if not isinstance(val, dict):
        return None, "arguments must be a JSON object"
    return val, None


def normalize_message(msg: dict[str, Any], id_prefix: str) -> tuple[Message, dict[str, str], list[dict[str, Any]]]:
    """Provider message -> (history message, call_errors, repairs)."""
    content = msg.get("content")
    history: Message = {"role": "assistant", "content": content if isinstance(content, str) else ("" if content is None else content)}
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        history["reasoning_content"] = reasoning
    call_errors: dict[str, str] = {}
    repairs: list[dict[str, Any]] = []
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        cid = tc.get("id")
        if not isinstance(cid, str) or not cid:
            new_id = f"call_{id_prefix}_{i}"
            repairs.append({"tool_call_id": new_id, "field": "id", "original": cid, "replacement": new_id, "reason": "missing tool-call id"})
            cid = new_id
        name = fn.get("name")
        raw_args = fn.get("arguments")
        _, err = parse_arguments(raw_args)
        if not isinstance(name, str) or not name:
            err = "missing function name" + (f"; {err}" if err else "")
            name = name if isinstance(name, str) else ""
        hist_args = raw_args
        if err is not None:
            call_errors[cid] = err
            if not (isinstance(raw_args, str) and parse_arguments(raw_args)[1] is None):
                hist_args = "{}"
                repairs.append(
                    {"tool_call_id": cid, "field": "function.arguments", "original": raw_args, "replacement": "{}", "reason": REPAIR_INVALID_ARGS_REASON}
                )
        calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": hist_args}})
    if calls:
        history["tool_calls"] = calls
    return history, call_errors, repairs


def _decision(
    raw_request: dict[str, Any],
    raw_response: dict[str, Any] | None,
    msg: dict[str, Any],
    finish_reason: str | None,
    usage: Usage,
    latency: float | None,
    seed_sent: int | None,
    id_prefix: str,
) -> PolicyDecision:
    history, call_errors, repairs = normalize_message(msg, id_prefix)
    # Servers that fail to parse a native tool-call block (e.g. invalid JSON escapes) may return
    # it as plain content; ours reports that under `learning_loop.parse_errors`. Surface it so the
    # turn is an explicit malformed call, never a silent "finished" reply.
    server_errors = ((raw_response or {}).get("learning_loop") or {}).get("parse_errors") or []
    return PolicyDecision(
        raw_request=raw_request,
        raw_response=raw_response,
        history_message=history,
        finish_reason=finish_reason,
        usage=usage,
        latency_sec=latency,
        parse_errors=[f"{cid}: {e}" for cid, e in call_errors.items()] + [f"unparsed_tool_call: {e}" for e in server_errors],
        repairs=repairs,
        seed_sent=seed_sent,
        call_errors=call_errors,
    )


def _error_decision(raw_request: dict[str, Any], seed_sent: int | None, latency: float | None, *, request_error: str | None = None, infra_error: str | None = None, raw: dict[str, Any] | None = None) -> PolicyDecision:
    return PolicyDecision(
        raw_request=raw_request,
        raw_response=raw,
        history_message={},
        finish_reason=None,
        usage=Usage(source="none"),
        latency_sec=latency,
        request_error=request_error,
        infra_error=infra_error,
        seed_sent=seed_sent,
    )


def _n_assistant(messages: list[Message]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoint
# --------------------------------------------------------------------------- #

DEFAULT_REQUEST_TIMEOUT_SEC = 600.0
# loop -> {(api_base, limit): Semaphore}. Semaphores bind to one event loop, so they are per loop.
_ENDPOINT_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, int], asyncio.Semaphore]]" = weakref.WeakKeyDictionary()


def endpoint_semaphore(api_base: str, limit: int) -> asyncio.Semaphore:
    """The process-wide semaphore bounding concurrent requests to `api_base` (this event loop)."""
    if limit < 1:
        raise ValueError(f"max_concurrent_requests must be >= 1, got {limit}")
    per_loop = _ENDPOINT_SEMAPHORES.setdefault(asyncio.get_running_loop(), {})
    key = (api_base.rstrip("/"), int(limit))
    if key not in per_loop:
        per_loop[key] = asyncio.Semaphore(int(limit))
    return per_loop[key]


class OpenAIChatPolicy:
    def __init__(self, spec: PolicySpec, api_key: str | None = None, timeout_sec: float | None = None, client: Any = None):
        if spec.kind != "openai":
            raise ValueError(f"OpenAIChatPolicy needs kind=openai, got {spec.kind}")
        if not spec.api_base:
            raise ValueError("PolicySpec.api_base is required for an OpenAI-compatible policy")
        self.spec = spec
        if spec.max_concurrent_requests is not None and spec.max_concurrent_requests < 1:
            raise ValueError(f"max_concurrent_requests must be >= 1, got {spec.max_concurrent_requests}")
        # explicit argument > spec (machine profile) > default
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else (spec.request_timeout_sec or DEFAULT_REQUEST_TIMEOUT_SEC))
        if client is None:
            from openai import AsyncOpenAI

            key = api_key or (os.environ.get(spec.api_key_env) if spec.api_key_env else None) or "sk-no-key"
            client = AsyncOpenAI(base_url=spec.api_base, api_key=key, max_retries=0, timeout=self.timeout_sec)
        self.client = client

    @property
    def model(self) -> str:
        if self.spec.served_model_name:
            return self.spec.served_model_name
        if self.spec.checkpoint:
            return self.spec.checkpoint.checkpoint_id
        return "default"

    def build_request(self, messages: list[Message], tools: list[ToolSchema], seed: int | None) -> tuple[dict[str, Any], int | None]:
        s = self.spec.sampling
        req: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:  # some servers reject an empty tools array (e.g. editor requests)
            req["tools"] = tools
        if s.temperature is not None:
            req["temperature"] = s.temperature
        if s.top_p is not None:
            req["top_p"] = s.top_p
        if s.max_output_tokens is not None:
            req["max_tokens"] = s.max_output_tokens
        seed_sent = seed if (self.spec.send_seed and seed is not None) else None
        if seed_sent is not None:
            req["seed"] = seed_sent
        if s.extra_body:
            req["extra_body"] = dict(s.extra_body)
        return req, seed_sent

    async def decide(self, messages: list[Message], tools: list[ToolSchema], seed: int | None) -> PolicyDecision:
        req, seed_sent = self.build_request(messages, tools, seed)
        limit = self.spec.max_concurrent_requests
        if limit is not None:
            async with endpoint_semaphore(self.spec.api_base or "", limit):
                return await self._send(req, seed_sent, messages)
        return await self._send(req, seed_sent, messages)

    async def _send(self, req: dict[str, Any], seed_sent: int | None, messages: list[Message]) -> PolicyDecision:
        import openai

        t0 = time.monotonic()
        try:
            resp = await self.client.chat.completions.create(**req)
        except openai.APIStatusError as e:
            latency = time.monotonic() - t0
            status = e.status_code
            body = e.body if isinstance(e.body, (dict, list, str)) else None
            raw = {"error": {"status": status, "message": e.message, "body": body}}
            text = f"{status} {e.message}"
            if 400 <= status < 500 and status not in (408, 429):
                return _error_decision(req, seed_sent, latency, request_error=text, raw=raw)
            return _error_decision(req, seed_sent, latency, infra_error=text, raw=raw)
        except openai.APIConnectionError as e:  # includes APITimeoutError
            latency = time.monotonic() - t0
            return _error_decision(req, seed_sent, latency, infra_error=f"{type(e).__name__}: {e}", raw={"error": {"type": type(e).__name__, "message": str(e)}})
        latency = time.monotonic() - t0
        raw = resp.model_dump(mode="json")
        choices = raw.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            return _error_decision(req, seed_sent, latency, infra_error="provider response has no choices[0].message", raw=raw)
        choice = choices[0]
        return _decision(req, raw, choice["message"], choice.get("finish_reason"), usage_from_payload(raw.get("usage")), latency, seed_sent, id_prefix=str(_n_assistant(messages)))

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            await close()


# --------------------------------------------------------------------------- #
# Scripted fixture policy
# --------------------------------------------------------------------------- #


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScriptedCall(_Strict):
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str | None = None  # exact argument text (e.g. deliberately malformed)
    id: str | None = None


class ScriptedError(_Strict):
    kind: Literal["request_error", "infra_error"]
    message: str


class ScriptedResponse(_Strict):
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ScriptedCall] = Field(default_factory=list)
    finish_reason: str | None = None
    error: ScriptedError | None = None
    delay_sec: float = 0.0  # simulated inference wait (timing-independence tests)


class ScriptedWhen(_Strict):
    turn: int | None = None  # assistant-turn index (= assistant messages so far)
    min_turn: int | None = None
    max_turn: int | None = None
    last_observation_regex: str | None = None  # search in the last tool message
    any_observation_regex: str | None = None  # search in any tool message
    no_observation_regex: str | None = None  # no tool message matches
    last_tool_name: str | None = None


class ScriptedRule(_Strict):
    name: str
    when: ScriptedWhen = Field(default_factory=ScriptedWhen)
    respond: ScriptedResponse


class PolicyScript(_Strict):
    schema_: Literal["scripted_policy/v1"] = Field(alias="schema")
    label: str = "fixture"
    description: str = ""
    delay_sec: float = 0.0
    rules: list[ScriptedRule]


def load_script(path: str | Path) -> PolicyScript:
    p = Path(path)
    data = yaml.safe_load(p.read_text()) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text())
    return PolicyScript.model_validate(data)


def fixture_token_estimate(text: str) -> int:
    """Labeled fixture estimate (ceil(chars/4)); never presented as provider usage."""
    return math.ceil(len(text) / 4)


_CAPTURE = re.compile(r"\$\{(\d+)\}")


def _subst(value: Any, groups: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _CAPTURE.sub(lambda m: groups[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(groups) else m.group(0), value)
    if isinstance(value, dict):
        return {k: _subst(v, groups) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v, groups) for v in value]
    return value


class ScriptedPolicy:
    """Deterministic, rule-based fixture policy (first matching rule wins).

    Rules look at the conversation (turn index, tool observations), so a
    continuation after a *different* intervention still produces sensible
    actions. `${n}` in a response's strings is replaced by capture group n of
    `last_observation_regex`.
    """

    def __init__(self, spec: PolicySpec, script: PolicyScript | None = None, delay_sec: float | None = None):
        if script is None:
            if not spec.scripted_path:
                raise ValueError("scripted policy needs PolicySpec.scripted_path")
            script = load_script(_resolve(spec.scripted_path))
        self.spec = spec
        self.script = script
        self.delay_sec = script.delay_sec if delay_sec is None else delay_sec

    def _match(self, messages: list[Message]) -> tuple[int, ScriptedRule, tuple[str, ...]] | None:
        turn = _n_assistant(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        last = messages[-1] if messages else {}
        last_obs = last.get("content") if last.get("role") == "tool" else None
        last_tool_name = None
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                last_tool_name = m["tool_calls"][-1]["function"]["name"]
                break
        for i, rule in enumerate(self.script.rules):
            w = rule.when
            groups: tuple[str, ...] = ()
            if w.turn is not None and turn != w.turn:
                continue
            if w.min_turn is not None and turn < w.min_turn:
                continue
            if w.max_turn is not None and turn > w.max_turn:
                continue
            if w.last_tool_name is not None and last_tool_name != w.last_tool_name:
                continue
            if w.last_observation_regex is not None:
                m = re.search(w.last_observation_regex, last_obs or "", re.M) if last_obs is not None else None
                if m is None:
                    continue
                groups = tuple(g or "" for g in m.groups())
            if w.any_observation_regex is not None and not any(re.search(w.any_observation_regex, str(t.get("content")), re.M) for t in tool_msgs):
                continue
            if w.no_observation_regex is not None and any(re.search(w.no_observation_regex, str(t.get("content")), re.M) for t in tool_msgs):
                continue
            return i, rule, groups
        return None

    async def decide(self, messages: list[Message], tools: list[ToolSchema], seed: int | None) -> PolicyDecision:
        seed_sent = seed if (self.spec.send_seed and seed is not None) else None
        raw_request = {"model": self.spec.served_model_name or "scripted", "messages": messages, "tools": tools, "seed": seed_sent, "scripted": True}
        t0 = time.monotonic()
        matched = self._match(messages)
        if matched is None:
            return _error_decision(raw_request, seed_sent, 0.0, request_error="scripted policy: no rule matches the conversation")
        idx, rule, groups = matched
        r = rule.respond
        delay = r.delay_sec or self.delay_sec
        if delay:
            await asyncio.sleep(delay)
        if r.error is not None:
            kw = {r.error.kind: r.error.message}
            return _error_decision(raw_request, seed_sent, time.monotonic() - t0, raw={"scripted_rule": rule.name}, **kw)
        turn = _n_assistant(messages)
        calls = []
        for j, c in enumerate(r.tool_calls):
            args = c.raw_arguments if c.raw_arguments is not None else json.dumps(_subst(c.arguments or {}, groups))
            calls.append({"id": c.id or f"call_{turn}_{j}", "type": "function", "function": {"name": c.name, "arguments": args}})
        msg: dict[str, Any] = {"role": "assistant", "content": _subst(r.content, groups)}
        if r.reasoning_content:
            msg["reasoning_content"] = r.reasoning_content
        if calls:
            msg["tool_calls"] = calls
        finish = r.finish_reason or ("tool_calls" if calls else "stop")
        out_text = msg["content"] + (msg.get("reasoning_content") or "") + "".join(c["function"]["name"] + c["function"]["arguments"] for c in calls)
        usage = Usage(
            input_tokens=fixture_token_estimate(json.dumps(messages) + json.dumps(tools)),
            output_tokens=fixture_token_estimate(out_text),
            source="fixture_estimate",
        )
        raw = {"scripted_rule": rule.name, "rule_index": idx, "message": msg, "finish_reason": finish}
        return _decision(raw_request, raw, msg, finish, usage, time.monotonic() - t0, seed_sent, id_prefix=str(turn))

    async def aclose(self) -> None:
        return None


def _resolve(path: str) -> Path:
    from .config import repo_path

    return repo_path(path)


def make_policy(spec: PolicySpec, **kwargs: Any) -> Policy:
    if spec.kind == "openai":
        return OpenAIChatPolicy(spec, **kwargs)
    if spec.kind == "scripted":
        return ScriptedPolicy(spec, **kwargs)
    raise ValueError(f"unknown policy kind {spec.kind!r}")
