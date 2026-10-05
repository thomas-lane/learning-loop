"""Model profiles, reference-cache keys and preflight result semantics (no model weights loaded)."""

from __future__ import annotations

import re

import pytest

from learning_loop.core.config import load_model_profile
from learning_loop.core.records import CheckpointRef
from learning_loop.hosts import preflight
from learning_loop.training.dpo import reference_cache_key
from learning_loop.training.render import RenderedPair


def test_profiles_pin_exact_identities():
    q = load_model_profile("qwen3-0.6b")
    assert q.base_model == "Qwen/Qwen3-0.6B" and re.fullmatch(r"[0-9a-f]{40}", q.base_revision)
    assert q.chat_template_kwargs == {"enable_thinking": False} and q.tool_call_format == "qwen3_xml"
    assert q.serving["hf_transformers"].adapter_formats == ["peft_lora"]
    g = load_model_profile("gemma-4-e4b-it")
    assert g.base_model == "google/gemma-4-E4B-it" and g.base_revision == "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
    lc = g.serving["llama_cpp"]
    # the GGUF is a quantized serving artifact, not the trainable source, and adapter serving is untested
    assert lc.artifact != g.base_model and lc.quantization == "Q8_0" and lc.status == "untested"
    assert lc.adapter_formats == ["gguf_lora"]
    assert all(s.status != "tested" for s in g.serving.values())
    assert g.supported_train_devices == ["cuda"]


def _pair(**kw) -> RenderedPair:
    base = dict(pair_id="p", example_sha256="e" * 64, prompt_ids=[1, 2], chosen_ids=[3, 9], rejected_ids=[4, 9],
                prompt_text="", chosen_text="", rejected_text="")
    base.update(kw)
    return RenderedPair(**base)


def test_reference_cache_key_invalidation():
    ref = CheckpointRef(checkpoint_id="c001-a", model_profile="qwen3-0.6b", base_model="Qwen/Qwen3-0.6B",
                        base_revision="c" * 40, adapter_path="/x", adapter_sha256="s1")
    k = reference_cache_key(ref, _pair(), "tok", "tmpl", {"enable_thinking": False}, "float32")
    assert k == reference_cache_key(ref, _pair(), "tok", "tmpl", {"enable_thinking": False}, "float32")
    variants = [
        reference_cache_key(ref.model_copy(update={"checkpoint_id": "c002-b"}), _pair(), "tok", "tmpl", {"enable_thinking": False}, "float32"),
        reference_cache_key(ref.model_copy(update={"adapter_sha256": "s2"}), _pair(), "tok", "tmpl", {"enable_thinking": False}, "float32"),
        reference_cache_key(ref, _pair(example_sha256="f" * 64), "tok", "tmpl", {"enable_thinking": False}, "float32"),
        reference_cache_key(ref, _pair(chosen_ids=[3, 8]), "tok", "tmpl", {"enable_thinking": False}, "float32"),
        reference_cache_key(ref, _pair(), "tok2", "tmpl", {"enable_thinking": False}, "float32"),
        reference_cache_key(ref, _pair(), "tok", "tmpl2", {"enable_thinking": False}, "float32"),
        reference_cache_key(ref, _pair(), "tok", "tmpl", {"enable_thinking": True}, "float32"),
        reference_cache_key(ref, _pair(), "tok", "tmpl", {"enable_thinking": False}, "bfloat16"),
    ]
    assert all(v != k for v in variants)


def test_reference_cache_roundtrip(tmp_path):
    from learning_loop.training.dpo import ReferenceCache

    c = ReferenceCache(tmp_path)
    key = {"a": 1}
    assert c.get(key) is None
    c.put(key, -1.5, -2.5, {})
    assert c.get(key) == (-1.5, -2.5) and (c.hits, c.misses) == (1, 1)
    assert c.get({"a": 2}) is None


def test_preflight_skips_are_not_success(tmp_path):
    res = {r.name: r for r in preflight.run_preflight(None, docker=False, train=False, disk_path=tmp_path, min_free_gb=0)}
    assert res["docker"].status == "skipped"
    assert res["model_access"].status == "skipped"
    assert res["adapter_trainability"].status == "skipped"
    assert res["disk"].status == "ok"


def test_preflight_failures_are_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    assert preflight.check_docker().status == "fail"
    assert preflight.check_disk(tmp_path, min_free_gb=10**9).status == "fail"
    assert preflight.check_memory(min_total_gb=10**9).status in ("fail", "skipped")
    prof = tmp_path / "missing.yaml"
    prof.write_text(
        "name: missing\nbase_model: nobody/does-not-exist\nbase_revision: '" + "0" * 40 + "'\n"
        "serving: {scripted: {backend: scripted}}\n"
    )
    r = preflight.check_model_access(str(prof), allow_network=False)
    assert r.status == "fail" and "not in the local HF cache" in r.detail


GB = 2**30


@pytest.mark.parametrize(
    ("v2", "v1", "total_gb", "limit_gb"),
    [
        (str(8 * GB), None, 8.0, 8.0),  # cgroup v2 limit below the host's RAM: a pod container
        (None, str(12 * GB), 12.0, 12.0),  # cgroup v1
        ("max", None, 64.0, None),  # v2 without a limit
        (None, str(2**63 - 4096), 64.0, None),  # v1 without a limit (page-rounded LONG_MAX)
        (str(128 * GB), None, 64.0, 128.0),  # a limit above physical memory does not raise it
        (None, None, 64.0, None),  # no cgroup files (macOS, bare metal)
    ],
)
def test_preflight_memory_respects_cgroup_limit(tmp_path, monkeypatch, v2, v1, total_gb, limit_gb):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal:       {64 * GB // 1024} kB\nMemFree:        1024 kB\n")
    files = (tmp_path / "memory.max", tmp_path / "memory" / "memory.limit_in_bytes")
    for f, val in zip(files, (v2, v1)):
        if val is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(val + "\n")
    monkeypatch.setattr(preflight.sys, "platform", "linux")
    monkeypatch.setattr(preflight, "MEMINFO", meminfo)
    monkeypatch.setattr(preflight, "CGROUP_MEMORY_LIMIT_FILES", files)
    r = preflight.check_memory(min_total_gb=10)
    assert r.data["total_gb"] == total_gb and r.data["machine_gb"] == 64.0 and r.data["cgroup_limit_gb"] == limit_gb
    assert r.status == ("fail" if total_gb < 10 else "ok")
    assert ("cgroup limit" in r.detail) == (total_gb == limit_gb)


def test_preflight_cli_exit_code(tmp_path, capsys):
    rc = preflight.main(["--no-docker", "--no-train", "--disk-path", str(tmp_path), "--min-free-gb", "0", "--json"])
    assert rc == 0
    assert '"status": "skipped"' in capsys.readouterr().out
    rc = preflight.main(["--no-docker", "--no-train", "--disk-path", str(tmp_path), "--min-free-gb", str(10**9)])
    assert rc == 1


@pytest.mark.parametrize("profile", ["qwen3-0.6b"])
def test_resolve_device_requires_explicit_cpu_fallback(profile, monkeypatch):
    torch = pytest.importorskip("torch")
    from learning_loop.training.common import TrainingRequestError, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(TrainingRequestError):
        resolve_device("auto", allow_cpu_fallback=False)
    info = resolve_device("mps", allow_cpu_fallback=True)
    assert info["device"] == "cpu" and info["cpu_fallback"] is True
    assert resolve_device("cpu", allow_cpu_fallback=False)["cpu_fallback"] is False


def test_qwen3_1_7b_profile_shares_the_0_6b_template_pin():
    small, big = load_model_profile("qwen3-0.6b"), load_model_profile("qwen3-1.7b")
    assert big.base_model == "Qwen/Qwen3-1.7B" and big.base_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert big.chat_template_sha256 == small.chat_template_sha256
    assert big.chat_template_kwargs == small.chat_template_kwargs and big.tool_call_format == small.tool_call_format
    assert big.serving["hf_transformers"].status == "tested"  # base model served live on MPS and CUDA (Runpod)


def _no_accelerators(monkeypatch, cuda=False, mps=False):
    torch = pytest.importorskip("torch")
    # import the lazy train stack first: torch._dynamo (pulled in by transformers/peft) breaks
    # when torch.cuda.is_available is already a plain lambda at import time
    import peft  # noqa: F401
    import transformers
    from transformers import AutoModelForCausalLM, set_seed  # noqa: F401
    from transformers.trainer_utils import get_last_checkpoint  # noqa: F401

    transformers.AutoConfig  # noqa: B018
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_resolve_device_enforces_supported_train_devices(monkeypatch):
    from learning_loop.training.common import TrainingRequestError, resolve_device

    _no_accelerators(monkeypatch, mps=True)
    # a cuda-only profile on an MPS host: neither "auto", an explicit "mps", nor a CPU fallback is accepted
    for requested, fallback in (("auto", False), ("auto", True), ("mps", False), ("cpu", False), ("cuda", True)):
        with pytest.raises(TrainingRequestError, match="supported"):
            resolve_device(requested, fallback, ["cuda"])
    info = resolve_device("auto", False, ["mps", "cuda", "cpu"])
    assert info["device"] == "mps" and info["supported_device_check"] == "ok"
    assert info["supported_train_devices"] == ["mps", "cuda", "cpu"]
    # "auto" skips an available but unsupported accelerator and falls back to CPU only when allowed
    info = resolve_device("auto", True, ["cuda", "cpu"])
    assert info["device"] == "cpu" and info["cpu_fallback"] is True
    info = resolve_device("auto", False, [])
    assert info["device"] == "mps" and info["supported_device_check"] == "not_declared"


def test_preflight_fails_on_unsupported_train_device(monkeypatch):
    _no_accelerators(monkeypatch, mps=True)
    r = preflight.check_adapter_trainability("gemma-4-e4b-it", device="auto", allow_cpu_fallback=True)
    assert r.status == "fail" and "supported_train_devices" in r.detail
    assert r.data["supported_train_devices"] == ["cuda"]


def test_dpo_trainer_refuses_unsupported_device_before_loading(monkeypatch, tmp_path):
    pytest.importorskip("trl")
    from pathlib import Path

    from learning_loop.core.config import TrainingConfig
    from learning_loop.core.interfaces import TrainRequest
    from learning_loop.training import TrainingRequestError, base_checkpoint_ref
    from learning_loop.training import dpo

    _no_accelerators(monkeypatch, mps=True)
    monkeypatch.setattr(dpo, "load_tokenizer", lambda *a, **k: pytest.fail("tokenizer loaded before the device check"))
    monkeypatch.setattr(dpo, "load_base_model", lambda *a, **k: pytest.fail("model loaded before the device check"))
    req = TrainRequest(
        run_id="r", cycle=1, dataset_dir=str(Path(__file__).resolve().parents[1] / "fixtures" / "train" / "fixture_prefs"),
        incoming=base_checkpoint_ref("gemma-4-e4b-it"), model_profile="gemma-4-e4b-it",
        training_config=TrainingConfig(trainer="trl_dpo", optimizer_steps=1).model_dump(), seed=1,
        output_root=str(tmp_path / "out"), device="auto", allow_cpu_fallback=True,
    )
    with pytest.raises(TrainingRequestError, match="supported_train_devices"):
        dpo.TrlDpoTrainer().train(req, tmp_path / "work")
