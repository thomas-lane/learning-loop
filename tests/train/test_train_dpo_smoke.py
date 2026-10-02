"""REAL tiny LoRA DPO smoke on Qwen/Qwen3-0.6B (pinned) with labeled FIXTURE preferences.

Marked `train` (excluded by default). Run: uv run pytest -m train tests/train -v
Needs the `train` extra, the pinned model in the HF cache (or network), and MPS or CUDA.
Budget: 2 optimizer steps per run, batch 1, 3 fixture pairs; four short runs + one server.

What is exercised (fixture data is NOT evidence of learning):
- cycle 1 from the base model (new LoRA), cycle 2 continuing cycle 1's adapter;
- reference = incoming learner incl. adapter (vs. adapter-disabled base);
- interrupted cycle-2 stage resumed from its own Trainer checkpoint equals an
  uninterrupted run (optimizer/scheduler state restored);
- reference-cache hits/invalidation, save/reload identity, idempotent publication;
- serving the trained adapter and a two-turn tool-calling exchange through hf_server.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.train

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "train" / "fixture_prefs"
PROFILE = "qwen3-0.6b"


def _device() -> str:
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("no MPS/CUDA device (CPU training is only allowed as an explicit fallback)")


def _deterministic_cuda() -> None:
    """CUDA's efficient attention kernels accumulate gradients in a nondeterministic order, and a
    LoRA's first Adam steps amplify that noise past the resume tolerance (measured on an A100:
    2-9e-5 between identical runs). These tests check resume logic, not kernels, so on CUDA they
    use the math attention kernel and fail on any other nondeterministic op."""
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from learning_loop.core.config import TrainingConfig
    from learning_loop.core.interfaces import TrainRequest
    from learning_loop.training.dpo import TrlDpoTrainer
    from learning_loop.training.fixture import SimulatedInterruption, base_checkpoint_ref

    device = _device()
    if device == "cuda":
        _deterministic_cuda()
    root = tmp_path_factory.mktemp("dpo_smoke")
    cache = root / "ref_cache"
    tc = TrainingConfig(
        trainer="trl_dpo", optimizer_steps=2,
        dpo={"learning_rate": 5e-4, "max_length": 1024, "beta": 0.1},
        lora={"r": 8, "alpha": 16, "dropout": 0.0},
    ).model_dump()

    def req(cycle, incoming, out, seed):
        return TrainRequest(run_id="smoke-test", cycle=cycle, dataset_dir=str(FIX), incoming=incoming, model_profile=PROFILE,
                            training_config=tc, seed=seed, output_root=str(root / out), device=device)

    r1 = req(1, base_checkpoint_ref(PROFILE), "ckpts", 11)
    c1 = TrlDpoTrainer(ref_cache_dir=cache).train(r1, root / "w1")

    r2 = req(2, c1.checkpoint, "ckpts", 22)
    with pytest.raises(SimulatedInterruption):
        TrlDpoTrainer(ref_cache_dir=cache, _test_interrupt_after_step=1).train(r2, root / "w2")
    ck_files = sorted(p.name for p in (root / "w2" / "trainer" / "checkpoint-1").iterdir())
    published_after_interrupt = sorted(p.name for p in (root / "ckpts").iterdir())
    c2 = TrlDpoTrainer(ref_cache_dir=cache).train(r2, root / "w2")

    r2f = req(2, c1.checkpoint, "ckpts-fresh", 22)
    c2f = TrlDpoTrainer(ref_cache_dir=cache).train(r2f, root / "w2-fresh")
    return {"root": root, "device": device, "c1": c1, "c2": c2, "c2f": c2f, "r1": r1, "cache": cache,
            "ck_files": ck_files, "published_after_interrupt": published_after_interrupt}


def _adapter(rec):
    from learning_loop.training.modeling import load_adapter_file_state

    return load_adapter_file_state(Path(rec.checkpoint.adapter_path))


def test_cycle1_real_dpo_step(runs):
    c1 = runs["c1"]
    m = c1.metrics
    assert c1.optimizer_steps == 2 and len(m["losses"]) == 2 and all(math.isfinite(x) for x in m["losses"])
    assert abs(m["losses"][0] - math.log(2)) < 1e-3  # policy == reference before the first update
    assert m["device"]["device"] == runs["device"] and m["device"]["cpu_fallback"] is False
    assert m["device"]["supported_device_check"] == "ok" and runs["device"] in m["device"]["supported_train_devices"]
    # one log-prob path: no Trainer mixed precision, fp32 logits hook, recorded
    prec = m["precision"]
    assert prec["trainer_mixed_precision"] == "no" and prec["native_amp"] is False
    assert prec["forward_wrapped_by_accelerate"] is False and prec["logits"].startswith("float32")
    assert m["adapter_init"] == "new_lora_on_base" and c1.reference_checkpoint_id == "base"
    assert m["reference"]["reference_minus_base_max_abs"] == 0.0  # no incoming adapter yet
    assert m["reference_identity_first_step"]["checked"] is True
    # LoRA B starts at zero; after training it moved
    b = [v for k, v in _adapter(c1).items() if "lora_B" in k]
    assert b and max(float(t.abs().max()) for t in b) > 0
    lc = c1.load_check
    assert lc["ok"] and lc["param_max_abs_diff"] == 0.0 and lc["logp_max_abs_diff"] <= lc["logp_atol"]
    assert c1.tokenizer_sha256 and c1.chat_template_sha256 == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    assert c1.n_train_examples == 3 and c1.metrics["dataset"]["provenance_kinds"] == {"fixture": 3}


def test_cycle2_continues_adapter_with_incoming_reference(runs):
    c1, c2 = runs["c1"], runs["c2"]
    m = c2.metrics
    assert c2.checkpoint.parent_checkpoint_id == c1.checkpoint.checkpoint_id == c2.reference_checkpoint_id
    assert m["adapter_init"] == "continue_incoming" and m["initial_adapter_matches_incoming"] is True
    ref = m["reference"]
    assert ref["reference_includes_adapter"] is True
    # the reference is the incoming learner WITH its adapter: it differs from the adapter-disabled base ...
    assert ref["reference_minus_base_max_abs"] > 1e-2
    # ... and equals cycle 1's published model, as measured independently by cycle 1's reload check
    for a, b in zip(ref["reference_logps"], c1.load_check["reloaded_logps"]):
        assert a == pytest.approx(b, abs=1e-3)
    assert m["adapter_max_abs_change_vs_incoming"] > 0
    ident = m["reference_identity_first_step"]
    assert ident["checked"] and abs(ident["first_step_logratio_chosen"]) <= ident["atol"]
    assert c2.load_check["ok"]


def test_reference_fixed_while_policy_moves(runs):
    """Reload the incoming checkpoint after cycle 2 trained: its log-probs still equal the cached reference."""
    from learning_loop.core.config import load_model_profile
    from learning_loop.training.modeling import free_memory, load_adapter, load_base_model, sequence_logps
    from learning_loop.training.render import end_of_turn_ids, load_tokenizer, render_pairs
    from learning_loop.training.common import load_preference_dataset

    c1, c2 = runs["c1"], runs["c2"]
    prof = load_model_profile(PROFILE)
    tok = load_tokenizer(prof)
    exs, _, _ = load_preference_dataset(FIX)
    pairs = render_pairs(exs, tok, chat_template_kwargs=prof.chat_template_kwargs, eot_ids=end_of_turn_ids(tok, prof), max_length=1024).kept[:2]
    model = load_adapter(load_base_model(prof, runs["device"]), Path(c1.checkpoint.adapter_path), trainable=False)
    try:
        now = sequence_logps(model, pairs, runs["device"], tok.pad_token_id)
    finally:
        del model
        free_memory()
    for a, b in zip(now, c2.metrics["reference"]["reference_logps"]):
        assert a == pytest.approx(b, abs=1e-3)
    moved = max(abs(x - y) for p, q in zip(c2.metrics["trained_logps"], now) for x, y in zip(p, q))
    assert moved > 1e-2


def test_resume_restores_trainer_state(runs):
    c2, c2f = runs["c2"], runs["c2f"]
    assert {"optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"} <= set(runs["ck_files"])
    assert runs["published_after_interrupt"] == [runs["c1"].checkpoint.checkpoint_id]  # nothing half-published
    assert c2.metrics["resumed_from"].endswith("checkpoint-1") and c2f.metrics["resumed_from"] is None
    assert c2.checkpoint.checkpoint_id == c2f.checkpoint.checkpoint_id  # same inputs, same id
    assert len(c2.metrics["losses"]) == 2
    a, b = _adapter(c2), _adapter(c2f)
    assert set(a) == set(b)
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst <= 1e-5, f"resumed run diverged from uninterrupted run by {worst}"


def test_reference_cache_hits_and_invalidation(runs):
    assert runs["c1"].metrics["reference_cache"]["misses"] == 3
    assert runs["c2"].metrics["reference_cache"] == {**runs["c2"].metrics["reference_cache"], "hits": 3, "misses": 0}
    assert runs["c2f"].metrics["reference_cache"]["hits"] == 3
    entries = [json.loads(p.read_text()) for p in runs["cache"].glob("*.json")]
    assert len(entries) == 6  # 3 keyed on reference "base" + 3 keyed on the cycle-1 checkpoint
    assert {e["key"]["reference_checkpoint_id"] for e in entries} == {"base", runs["c1"].checkpoint.checkpoint_id}


def test_publication_is_idempotent_and_cli_reports_path(runs):
    from learning_loop.training.dpo import TrlDpoTrainer

    root, c1, r1 = runs["root"], runs["c1"], runs["r1"]
    cj = Path(c1.checkpoint.adapter_path) / "checkpoint.json"
    before = cj.read_bytes()
    again = TrlDpoTrainer().train(r1, root / "w1-again")
    assert again == c1 and cj.read_bytes() == before
    rp = root / "r1.json"
    rp.write_text(r1.model_dump_json())
    out = subprocess.run([sys.executable, "-m", "learning_loop.training.run", "--request", str(rp), "--work-dir", str(root / "w1-cli")],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == str(cj)


def test_serve_trained_adapter_tool_calling(runs):
    from learning_loop.core.config import load_model_profile
    from learning_loop.serving.managed import ManagedHFServer, free_port, http_json
    from learning_loop.training.common import load_preference_dataset
    from learning_loop.training.render import encode, load_tokenizer, render_text

    c2 = runs["c2"]
    exs, _, _ = load_preference_dataset(FIX)
    ex = exs[0]
    prof = load_model_profile(PROFILE)
    tok = load_tokenizer(prof)
    srv = ManagedHFServer(PROFILE, free_port(), runs["root"] / "serve.log", checkpoint_dir=Path(c2.checkpoint.adapter_path), device=runs["device"])
    info = srv.start(timeout_sec=300)
    try:
        assert info["model"] == c2.checkpoint.checkpoint_id and info["adapter_sha256"] == c2.checkpoint.adapter_sha256
        body = {"model": c2.checkpoint.checkpoint_id, "messages": ex.prompt, "tools": ex.tools, "temperature": 0, "max_tokens": 96, "seed": 3}
        st, r = http_json(srv.api_base + "/chat/completions", body, timeout=300)
        assert st == 200, r
        choice = r["choices"][0]
        msg = choice["message"]
        assert r["usage"]["prompt_tokens"] == len(encode(tok, render_text(tok, ex.prompt, ex.tools, add_generation_prompt=True,
                                                                           chat_template_kwargs=prof.chat_template_kwargs)))
        assert r["usage"]["completion_tokens"] >= 1 and "raw_completion_text" in r["learning_loop"]
        assert choice["finish_reason"] in ("tool_calls", "stop", "length")
        print("turn-1 finish_reason:", choice["finish_reason"], "raw:", r["learning_loop"]["raw_completion_text"][:200])
        if choice["finish_reason"] == "tool_calls":
            for tc in msg["tool_calls"]:
                assert tc["function"]["name"] in {t["function"]["name"] for t in ex.tools}
                assert isinstance(json.loads(tc["function"]["arguments"]), dict)
            # second turn: feed a fixed tool observation back through the parsed call id
            turn2 = ex.prompt + [msg] + [{"role": "tool", "tool_call_id": tc["id"], "content": "[exit code 0]\n3"} for tc in msg["tool_calls"]]
            st2, r2 = http_json(srv.api_base + "/chat/completions", {**body, "messages": turn2}, timeout=300)
            assert st2 == 200, r2
            assert r2["usage"]["prompt_tokens"] > r["usage"]["prompt_tokens"]
            print("turn-2 finish_reason:", r2["choices"][0]["finish_reason"], "raw:", r2["learning_loop"]["raw_completion_text"][:200])
        # same seed + sampling -> same output on this server
        sampled = {**body, "temperature": 0.8, "top_p": 0.95, "max_tokens": 24, "seed": 1234}
        s1 = http_json(srv.api_base + "/chat/completions", sampled, timeout=300)[1]
        s2 = http_json(srv.api_base + "/chat/completions", sampled, timeout=300)[1]
        assert s1["learning_loop"]["raw_completion_text"] == s2["learning_loop"]["raw_completion_text"]
        assert http_json(srv.api_base + "/chat/completions", {**body, "model": "base"})[0] == 404
    finally:
        rc = srv.stop()
    assert rc == 0, srv.log_tail()
