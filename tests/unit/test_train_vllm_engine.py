"""vLLM as a token engine behind hf_server (`--engine vllm`), against a fake vLLM server process:
our rendering goes in as token ids, every sampling parameter is explicit, generated ids come back
and are decoded, parsed and counted by us; the zero adapter matches what PEFT would build."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from learning_loop.core.config import HostRef, InferenceProfile, load_model_profile
from learning_loop.core.records import CheckpointRef
from learning_loop.serving import lifecycle, vllm_engine

torch = pytest.importorskip("torch")

FAKE_VLLM = textwrap.dedent('''
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    loras = [a.split("=")[0] for a in args[args.index("--lora-modules") + 1:] if "=" in a] if "--lora-modules" in args else []
    script = json.load(open(os.environ["FAKE_VLLM_SCRIPT"]))
    class H(BaseHTTPRequestHandler):
        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self._send({"data": [{"id": n} for n in ["base", *loras]]})
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(os.environ["FAKE_VLLM_LOG"], "a") as f:
                f.write(json.dumps({"argv": args, "request": req}) + "\\n")
            self._send({"choices": [{"token_ids": script["token_ids"], "finish_reason": "stop", "stop_reason": script["stop"]}]})
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture
def qwen():
    from learning_loop.training.render import load_tokenizer

    prof = load_model_profile("qwen3-0.6b")
    try:
        return prof, load_tokenizer(prof, local_files_only=True)
    except OSError:
        pytest.skip("qwen3-0.6b tokenizer not in the local HF cache")


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    pkg = tmp_path / "fake" / "vllm" / "entrypoints" / "openai"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "fake" / "vllm", tmp_path / "fake" / "vllm" / "entrypoints", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "api_server.py").write_text(FAKE_VLLM)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake"))
    monkeypatch.setenv("FAKE_VLLM_LOG", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("FAKE_VLLM_SCRIPT", str(tmp_path / "script.json"))
    monkeypatch.setattr(vllm_engine, "ensure_engine", lambda package: __import__("pathlib").Path(sys.executable))
    return tmp_path


def _requests(tmp_path):
    return [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]


def test_vllm_engine_round_trip_uses_our_rendering_parser_and_counts(qwen, fake_vllm):
    from learning_loop.training.render import encode, render_text

    prof, tok = qwen
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    gen = encode(tok, '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>')
    (fake_vllm / "script.json").write_text(json.dumps({"token_ids": gen, "stop": eos}))  # stop token not echoed
    spec = {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"], "exclude_modules": None}
    eng = vllm_engine.VllmEngine(prof, None, "vllm==0.30.0", base_checkpoint_id="base:x", zero_lora=spec,
                                 zero_adapter_root=fake_vllm / "zero", start_timeout_sec=30)
    try:
        msgs = [{"role": "user", "content": "list files"}]
        tools = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
        r = eng.complete({"model": "base:x", "messages": msgs, "tools": tools, "temperature": 0.7, "top_p": 0.9, "max_tokens": 64, "seed": 9})
        (req,) = _requests(fake_vllm)
        want = encode(tok, render_text(tok, msgs, tools, add_generation_prompt=True, chat_template_kwargs=prof.chat_template_kwargs))
        body = req["request"]
        assert body["prompt"] == want and body["model"] == "base:x"  # token ids in; the served LoRA module
        assert {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed", "max_tokens")} == {
            "temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "seed": 9, "max_tokens": 64}
        assert body["skip_special_tokens"] is False and body["return_token_ids"] is True and eos in body["stop_token_ids"]
        argv = req["argv"]
        assert "--generation-config" in argv and argv[argv.index("--generation-config") + 1] == "vllm"
        assert "--no-enable-prefix-caching" in argv and argv[argv.index("--max-num-seqs") + 1] == "1"
        assert any(a.startswith("base:x=") for a in argv)  # the zero adapter is served as a LoRA module
        ch = r["choices"][0]
        assert ch["finish_reason"] == "tool_calls" and json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
        assert r["usage"] == {"prompt_tokens": len(want), "completion_tokens": len(gen) + 1, "total_tokens": len(want) + len(gen) + 1}
        assert eng.info()["engine"] == "vllm" and r["learning_loop"]["engine"] == "vllm"
    finally:
        eng.close()


def test_completion_ids_end_with_the_stop_token_either_way():
    eot = {7}
    assert vllm_engine.completion_ids({"choices": [{"token_ids": [1, 2], "finish_reason": "stop", "stop_reason": 7}]}, eot) == [1, 2, 7]
    assert vllm_engine.completion_ids({"choices": [{"token_ids": [1, 2, 7], "finish_reason": "stop", "stop_reason": 7}]}, eot) == [1, 2, 7]
    assert vllm_engine.completion_ids({"choices": [{"token_ids": [1, 2], "finish_reason": "length", "stop_reason": None}]}, eot) == [1, 2]


def test_zero_adapter_has_exactly_the_modules_and_shapes_peft_creates():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=96, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=32, vocab_size=128)
    spec = {"r": 4, "alpha": 8, "target_modules": ["q_proj", "k_proj", "down_proj"], "exclude_modules": r".*layers\.1\.self_attn\..*"}
    with torch.device("meta"):
        meta = LlamaForCausalLM(cfg)
    ours = vllm_engine.zero_adapter_tensors(meta, spec)
    peft = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=spec["target_modules"],
                                                           exclude_modules=spec["exclude_modules"], task_type="CAUSAL_LM"))
    ref = get_peft_model_state_dict(peft)
    assert {k: tuple(v.shape) for k, v in ours.items()} == {k: tuple(v.shape) for k, v in ref.items()}
    assert all(float(v.abs().max()) == 0.0 for v in ours.values())


def test_lifecycle_launches_hf_server_with_the_vllm_engine(tmp_path):
    prof = InferenceProfile(mode="managed", backend="vllm", port=8765, device="cuda", request_concurrency=4, startup_timeout_sec=1800,
                            host=HostRef(kind="local"))
    spec = {"r": 16, "alpha": 32, "target_modules": ["q_proj"], "exclude_modules": None}
    mgr = lifecycle.InferenceManager(prof, load_model_profile("gemma-4-e4b-it"), tmp_path, zero_lora=spec)
    ckpt = CheckpointRef(checkpoint_id="base:gemma", model_profile="gemma-4-e4b-it", base_model="google/gemma-4-E4B-it",
                         base_revision="fee6332c1abaafb77f6f9624236c63aa2f1d0187", adapter_path=None)
    argv = mgr._server_argv(ckpt)
    assert argv[:2] == ["-m", "learning_loop.serving.hf_server"]
    assert argv[argv.index("--engine") + 1] == "vllm" and argv[argv.index("--engine-package") + 1] == "vllm==0.30.0"
    assert argv[argv.index("--concurrency") + 1] == "4" and json.loads(argv[argv.index("--zero-lora") + 1]) == spec


def test_equivalence_comparisons():
    from learning_loop.serving.equivalence import _corr, _prefix_agreement, compare

    c = compare([[-1.0, -2.0], [-0.5]], [[-1.1, -2.0], [-0.4]])
    assert c["n_tokens"] == 3 and abs(c["mean_abs"] - 0.2 / 3) < 1e-9 and abs(c["max_abs"] - 0.1) < 1e-9
    assert abs(c["max_seq_abs_per_token"] - 0.1) < 1e-9
    assert abs(_corr([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12 and abs(_corr([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    assert _corr([1, 1], [1, 2]) == 0.0
    assert _prefix_agreement([1, 2, 3], [1, 2, 4]) == 2 and _prefix_agreement([], [1]) == 0


def test_concurrent_requests_share_the_engine_safely(qwen, fake_vllm):
    import threading

    from learning_loop.training.render import encode

    prof, tok = qwen
    gen = encode(tok, "done")
    (fake_vllm / "script.json").write_text(json.dumps({"token_ids": gen, "stop": tok.convert_tokens_to_ids("<|im_end|>")}))
    eng = vllm_engine.VllmEngine(prof, None, "vllm==0.30.0", base_checkpoint_id="base:x", concurrency=8, start_timeout_sec=30)
    results, errors = [], []

    def one(i):
        try:
            results.append(eng.complete({"model": "base:x", "messages": [{"role": "user", "content": f"task {i} " * 50}], "temperature": 0, "max_tokens": 8}))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    try:
        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        eng.close()
    assert not errors and len(results) == 8
    assert {r["choices"][0]["message"]["content"] for r in results} == {"done"}
    assert len({tuple(r["request"]["prompt"]) for r in _requests(fake_vllm)}) == 8  # each request rendered its own prompt
    assert "--max-num-seqs" in _requests(fake_vllm)[0]["argv"] and _requests(fake_vllm)[0]["argv"][_requests(fake_vllm)[0]["argv"].index("--max-num-seqs") + 1] == "8"
