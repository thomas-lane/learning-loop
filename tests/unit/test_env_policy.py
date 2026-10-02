"""OpenAI-compatible policy against a mocked endpoint; scripted policy rules."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from learning_loop.core.records import PolicySpec, SamplingConfig
from learning_loop.episodes.policy import OpenAIChatPolicy, ScriptedPolicy, fixture_token_estimate, load_script

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def _completion(message: dict, usage: dict | None = None, finish: str = "tool_calls") -> dict:
    body = {"id": "x", "object": "chat.completion", "created": 1, "model": "m", "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _policy(handler, **spec_kw) -> tuple[OpenAIChatPolicy, list[dict]]:
    seen: list[dict] = []

    def _h(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    client = AsyncOpenAI(base_url="http://test/v1", api_key="k", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(_h)))
    spec = PolicySpec(kind="openai", api_base="http://test/v1", served_model_name="m", sampling=SamplingConfig(temperature=0.5, max_output_tokens=64, extra_body={"top_k": 20}), **spec_kw)
    return OpenAIChatPolicy(spec, client=client), seen


async def test_request_seed_reasoning_usage():
    msg = {"role": "assistant", "content": None, "reasoning_content": "think  verbatim\n", "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "prompt_tokens_details": {"cached_tokens": 64}, "completion_tokens_details": {"reasoning_tokens": 5}}
    pol, seen = _policy(lambda r: httpx.Response(200, json=_completion(msg, usage)))
    d = await pol.decide(MSGS, TOOLS, seed=42)
    assert seen[0]["seed"] == 42 and seen[0]["temperature"] == 0.5 and seen[0]["max_tokens"] == 64 and seen[0]["top_k"] == 20
    assert d.seed_sent == 42 and d.raw_request["seed"] == 42
    assert d.history_message == {"role": "assistant", "content": "", "reasoning_content": "think  verbatim\n", "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    assert (d.usage.input_tokens, d.usage.output_tokens, d.usage.cached_input_tokens, d.usage.reasoning_tokens, d.usage.source) == (100, 20, 64, 5, "provider")
    assert d.raw_response["choices"][0]["message"]["reasoning_content"] == "think  verbatim\n"
    assert d.parse_errors == [] and d.repairs == [] and d.finish_reason == "tool_calls"


async def test_seed_not_sent_when_disabled_and_missing_usage_is_none():
    msg = {"role": "assistant", "content": "done"}
    pol, seen = _policy(lambda r: httpx.Response(200, json=_completion(msg, None, "stop")), send_seed=False)
    d = await pol.decide(MSGS, TOOLS, seed=42)
    assert "seed" not in seen[0] and d.seed_sent is None
    assert d.usage.input_tokens is None and d.usage.output_tokens is None and d.usage.source == "none"
    assert "reasoning_content" not in d.history_message  # never fabricated
    u = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    pol2, _ = _policy(lambda r: httpx.Response(200, json=_completion(msg, u, "stop")))
    d2 = await pol2.decide(MSGS, TOOLS, None)
    assert d2.usage.cached_input_tokens is None and d2.usage.reasoning_tokens is None and d2.seed_sent is None


async def test_malformed_arguments_and_missing_id_are_visible_repairs():
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls'}},
        {"id": "b", "type": "function", "function": {"name": "bash", "arguments": '["ls"]'}},
        {"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pwd"}'}},
    ]}
    pol, _ = _policy(lambda r: httpx.Response(200, json=_completion(msg, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})))
    d = await pol.decide(MSGS, TOOLS, 1)
    calls = d.history_message["tool_calls"]
    assert calls[0]["id"] == "call_0_0" and calls[0]["function"]["arguments"] == "{}"
    assert calls[1]["function"]["arguments"] == "{}" and calls[2]["function"]["arguments"] == '{"command": "pwd"}'
    assert set(d.call_errors) == {"call_0_0", "b"}
    assert "could not parse" in d.call_errors["call_0_0"] and d.call_errors["b"] == "arguments must be a JSON object"
    fields = [(r["tool_call_id"], r["field"], r["original"]) for r in d.repairs]
    assert ("call_0_0", "id", "") in fields
    assert ("call_0_0", "function.arguments", '{"command": "ls') in fields and ("b", "function.arguments", '["ls"]') in fields


@pytest.mark.parametrize(
    "status,field",
    [(400, "request_error"), (422, "request_error"), (429, "infra_error"), (500, "infra_error"), (503, "infra_error")],
)
async def test_http_errors_are_classified(status, field):
    pol, _ = _policy(lambda r: httpx.Response(status, json={"error": {"message": "the request exceeds the available context size"}}))
    d = await pol.decide(MSGS, TOOLS, 1)
    assert getattr(d, field) is not None
    other = "infra_error" if field == "request_error" else "request_error"
    assert getattr(d, other) is None
    assert d.raw_response["error"]["status"] == status and d.history_message == {}


async def test_connection_error_is_infra():
    def boom(r):
        raise httpx.ConnectError("refused")

    pol, _ = _policy(boom)
    d = await pol.decide(MSGS, TOOLS, 1)
    assert d.infra_error and "APIConnectionError" in d.infra_error and d.request_error is None


def test_real_client_construction_uses_env_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-value")
    pol = OpenAIChatPolicy(PolicySpec(kind="openai", api_base="http://127.0.0.1:1/v1", api_key_env="MY_KEY"))
    assert pol.client.api_key == "secret-value" and pol.client.max_retries == 0
    pol2 = OpenAIChatPolicy(PolicySpec(kind="openai", api_base="http://127.0.0.1:1/v1"))
    assert pol2.client.api_key == "sk-no-key"


async def test_scripted_policy_rules_captures_and_labels(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("""
schema: scripted_policy/v1
rules:
  - {name: echo, when: {last_observation_regex: 'count=(\\d+)'}, respond: {tool_calls: [{name: write_file, arguments: {path: a, content: "n=${1}"}}]}}
  - {name: first, when: {turn: 0}, respond: {reasoning_content: "r", tool_calls: [{name: bash, arguments: {command: ls}}]}}
""")
    spec = PolicySpec(kind="scripted", scripted_path=str(p))
    pol = ScriptedPolicy(spec)
    d0 = await pol.decide(MSGS, TOOLS, 5)
    assert d0.history_message["tool_calls"][0] == {"id": "call_0_0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    assert d0.history_message["reasoning_content"] == "r" and d0.usage.source == "fixture_estimate"
    assert d0.usage.input_tokens == fixture_token_estimate(json.dumps(MSGS) + json.dumps(TOOLS))
    hist = MSGS + [d0.history_message, {"role": "tool", "tool_call_id": "call_0_0", "content": "count=17"}]
    d1 = await pol.decide(hist, TOOLS, 5)
    assert json.loads(d1.history_message["tool_calls"][0]["function"]["arguments"]) == {"path": "a", "content": "n=17"}
    d2 = await pol.decide(MSGS + [d0.history_message, {"role": "tool", "tool_call_id": "call_0_0", "content": "nothing"}], TOOLS, 5)
    assert d2.request_error and "no rule matches" in d2.request_error
    assert load_script(p).label == "fixture"


async def test_server_reported_unparsed_tool_call_is_a_parse_error():
    # hf_server leaves an unparseable native <tool_call> block in content and reports it.
    msg = {"role": "assistant", "content": '<tool_call>\n{"name": "bash", "arguments": {"command": "ls \\*.log"}}\n</tool_call>'}
    body = _completion(msg, {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}, "stop")
    body["learning_loop"] = {"parse_errors": ["invalid JSON in <tool_call> block"]}
    pol, _ = _policy(lambda r: httpx.Response(200, json=body))
    d = await pol.decide(MSGS, TOOLS, None)
    assert "tool_calls" not in d.history_message
    assert d.parse_errors == ["unparsed_tool_call: invalid JSON in <tool_call> block"]
