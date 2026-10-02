"""`evaluation/run.sh` + `configs/local-llama.yaml` still work (CLI mode, marker `docker`).

A tiny stub OpenAI-compatible server (stdlib, free port) stands in for the model,
so this checks the Harbor job path, kwargs and output files without a model.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from _env_helpers import count_errors_instance  # noqa: E402

from learning_loop.core.config import REPO_ROOT  # noqa: E402

pytestmark = pytest.mark.docker

SOLVE = "find /app/data -type f -name '*.log' -exec cat {} + | grep -c ERROR > /app/answer.txt"


class _Stub(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append(body)
        turn = sum(1 for m in body["messages"] if m["role"] == "assistant")
        if turn == 0:
            msg = {"role": "assistant", "content": "", "reasoning_content": "count them", "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": SOLVE})}}]}
            finish = "tool_calls"
        else:
            msg, finish = {"role": "assistant", "content": "done"}, "stop"
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
               "usage": {"prompt_tokens": 100 + turn, "completion_tokens": 10, "total_tokens": 110 + turn}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_sh_cli_mode(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    port = _free_port()
    assert port != 9931
    srv = HTTPServer(("127.0.0.1", port), _Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        proc = subprocess.run(
            [str(REPO_ROOT / "evaluation" / "run.sh"), "-p", inst.task_dir, "-o", str(tmp_path / "jobs"), "--ak", f"api_base=http://127.0.0.1:{port}/v1"],
            capture_output=True, text=True, timeout=600,
        )
    finally:
        srv.shutdown()
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    trials = [p for p in (tmp_path / "jobs").glob("*/*") if p.is_dir() and (p / "result.json").exists()]
    assert len(trials) == 1
    t = trials[0]
    result = json.loads((t / "result.json").read_text())
    assert result["verifier_result"]["rewards"] == {"reward": 1.0}
    assert result["agent_result"]["metadata"]["stop_reason"] == "model_finished"
    assert result["agent_result"]["n_input_tokens"] == 201 and result["agent_result"]["n_output_tokens"] == 20
    for f in ("trajectory.json", "messages.json", "events.jsonl", "episode.json"):
        assert (t / "agent" / f).exists(), f
    assert (t / "learning_loop" / "events.jsonl").exists()  # host-only record
    req = _Stub.requests[0]
    assert "seed" not in req and req["max_tokens"] == 8192 and req["model"] == "ggml-org/gemma-4-E4B-it-GGUF:Q8_0"
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert _Stub.requests[1]["messages"][2]["reasoning_content"] == "count them"  # sent back verbatim
