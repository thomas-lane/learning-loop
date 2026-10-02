"""Tool-call parsing and the OpenAI-compatible HTTP layer of hf_server (fake engine, no model)."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from learning_loop.serving.hf_server import ApiError, make_server, validate_request
from learning_loop.serving.managed import http_json, port_in_use
from learning_loop.serving.tool_parse import parse_completion, parse_gemma4, parse_qwen3


def test_qwen3_single_call_roundtrips_canonical_json():
    p = parse_qwen3('<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>', "seed")
    assert p.parse_errors == [] and p.content == "" and p.reasoning_content is None
    (tc,) = p.tool_calls
    assert tc["type"] == "function" and tc["id"].startswith("call_")
    assert tc["function"] == {"name": "bash", "arguments": '{"command": "ls -la"}'}
    assert p.message() == {"role": "assistant", "content": "", "tool_calls": p.tool_calls}


def test_qwen3_reasoning_content_and_multiple_calls():
    text = ('<think>\nlook first\n</think>\n\nSure.\n<tool_call>\n{"name": "bash", "arguments": {"command": "a"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read_file", "arguments": "{\\"path\\": \\"/x\\"}"}\n</tool_call>')
    p = parse_qwen3(text, "s")
    assert p.reasoning_content == "look first"
    assert p.content == "Sure."
    assert [c["function"]["name"] for c in p.tool_calls] == ["bash", "read_file"]
    assert json.loads(p.tool_calls[1]["function"]["arguments"]) == {"path": "/x"}
    assert len({c["id"] for c in p.tool_calls}) == 2


def test_qwen3_malformed_call_stays_visible():
    bad = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}\n</tool_call>'
    p = parse_qwen3(bad, "s")
    assert p.tool_calls == [] and p.parse_errors and "<tool_call>" in p.content
    p2 = parse_qwen3('<tool_call>\n{"name": "bash", "arguments": {"comm', "s")
    assert p2.tool_calls == [] and any("unterminated" in e for e in p2.parse_errors)
    p3 = parse_qwen3('<tool_call>\n{"name": "bash", "arguments": ["ls"]}\n</tool_call>', "s")
    assert p3.tool_calls == [] and p3.parse_errors


def test_plain_answer_and_unsupported_format():
    p = parse_qwen3("The answer is 4.", "s")
    assert p.content == "The answer is 4." and p.tool_calls == [] and "reasoning_content" not in p.message()
    with pytest.raises(NotImplementedError):
        parse_completion("unknown_format", "x", "s")


Q = '<|"|>'


def test_gemma4_call_values_and_reasoning():
    text = (f"<|channel>thought\nlook first<channel|>Checking.<|tool_call>call:bash{{command:{Q}grep -c \"x\" a, {{b}}{Q},"
            f"flags:[{Q}-a{Q},2],n:3,o:{{x:true,y:None,z:-1.5e2,{Q}k k{Q}:false}}}}<tool_call|>"
            f"<|tool_call>call:read_file{{path:{Q}/x{Q}}}<tool_call|>")
    p = parse_gemma4(text, "s")
    assert p.parse_errors == [] and p.content == "Checking." and p.reasoning_content == "look first"
    a, b = p.tool_calls
    assert a["function"]["name"] == "bash" and json.loads(a["function"]["arguments"]) == {
        "command": 'grep -c "x" a, {b}', "flags": ["-a", 2], "n": 3, "o": {"x": True, "y": None, "z": -150.0, "k k": False}}
    assert b["function"] == {"name": "read_file", "arguments": '{"path": "/x"}'} and a["id"] != b["id"]
    assert parse_completion("gemma4", text, "s").tool_calls == p.tool_calls


def test_gemma4_malformed_calls_stay_visible():
    for bad in [f"<|tool_call>call:bash{{command:{Q}ls}}<tool_call|>",       # unterminated string
                f"<|tool_call>call:bash{{command:{Q}ls{Q}}} extra<tool_call|>",  # trailing text
                "<|tool_call>bash{}<tool_call|>"]:                              # no call: prefix
        p = parse_gemma4(bad, "s")
        assert p.tool_calls == [] and p.parse_errors and bad in p.content
    p = parse_gemma4(f"<|tool_call>call:bash{{command:{Q}ls", "s")
    assert p.tool_calls == [] and any("unterminated" in e for e in p.parse_errors)
    p = parse_gemma4("There are 5 errors.", "s")
    assert (p.content, p.tool_calls, p.reasoning_content) == ("There are 5 errors.", [], None)


def test_validate_request_defaults_and_rejections():
    r = validate_request({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "m")
    assert (r["temperature"], r["top_p"], r["max_tokens"], r["seed"]) == (1.0, 1.0, None, None)
    assert validate_request({"messages": [{"role": "user"}], "max_completion_tokens": 5}, "m")["max_tokens"] == 5
    for body, status in [
        ({"model": "other", "messages": [{}]}, 404),
        ({"messages": []}, 400),
        ({"messages": [{}], "stream": True}, 400),
        ({"messages": [{}], "n": 2}, 400),
        ({"messages": [{}], "tool_choice": "required"}, 400),
        ({"messages": [{}], "seed": "7"}, 400),
    ]:
        with pytest.raises(ApiError) as e:
            validate_request(body, "m")
        assert e.value.status == status


class FakeEngine:
    served_model = "c001-fake"

    def __init__(self):
        self.closed = False

    def info(self):
        return {"model": self.served_model, "device": "none"}

    def complete(self, body):
        req = validate_request(body, self.served_model)
        if req["messages"][0].get("content") == "boom":
            raise RuntimeError("engine failure")
        return {"id": "x", "object": "chat.completion", "model": self.served_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def close(self):
        self.closed = True


@pytest.fixture()
def server():
    srv = make_server(FakeEngine(), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_http_routes(server):
    assert http_json(server + "/health")[1]["status"] == "ok"
    st, models = http_json(server + "/v1/models")
    assert st == 200 and models["data"][0]["id"] == "c001-fake"
    st, r = http_json(server + "/v1/chat/completions", {"model": "c001-fake", "messages": [{"role": "user", "content": "hi"}]})
    assert st == 200 and r["choices"][0]["message"]["content"] == "ok"
    st, r = http_json(server + "/v1/chat/completions", {"model": "latest", "messages": [{"role": "user", "content": "hi"}]})
    assert st == 404 and r["error"]["code"] == "model_not_found"
    st, r = http_json(server + "/v1/chat/completions", {"messages": [{"role": "user", "content": "boom"}]})
    assert st == 500 and "engine failure" in r["error"]["message"]
    req = urllib.request.Request(server + "/v1/chat/completions", data=b"{not json", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert http_json(server + "/nope")[0] == 404
    host, port = server.rsplit(":", 1)
    assert port_in_use("127.0.0.1", int(port))
