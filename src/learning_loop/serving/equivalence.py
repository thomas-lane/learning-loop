"""Equivalence gate: does vLLM (`hf_server --engine vllm`) serve the same model as the training
path? Run on a CUDA host before trusting the vLLM backend for a model profile:

    uv run --extra train python -m learning_loop.serving.equivalence --profile gemma-4-e4b-it \\
        --adapter runs/<run>/checkpoints/<ckpt> [--out equivalence.json]

On the profile's fixture-style conversations (tests/fixtures/train/fixture_prefs, rendered with
training/render.py exactly as for training) it compares, token by token, the log-probabilities of
the completion tokens under

- the reference: transformers + PEFT, float32 logits (the DPO/reference log-prob path), and
- vLLM, scoring the same token ids (`prompt_logprobs`) with the profile's `serving.vllm`
  engine package and launch arguments, through an all-zero LoRA of the experiment shape, a
  trained adapter (`--adapter`, e.g. a published checkpoint of the model), and a strong random
  LoRA.

Checks (all must pass; the report has every number):
1. vLLM echoes exactly the prompt token ids it was sent (no re-tokenization or template).
2. Zero LoRA and the trained adapter: mean per-token |delta log-prob| <= `--max-mean-abs` and
   per-sequence |delta sum| <= `--max-seq-abs` per completion token (bf16 kernel noise only).
3. Random LoRA: its effect (random - zero) is large compared with the numerical noise and agrees
   between the two paths, per sequence within `--max-effect-rel` of its size plus twice the
   zero-LoRA noise and per token with correlation >= `--min-effect-corr`: vLLM applies adapters
   with PEFT's scaling on the same modules. Its absolute log-probs are reported but not bounded:
   vLLM computes LoRA in bf16 (PEFT in float32), which shows only for adapters far stronger than
   trained ones.
4. Greedy decoding from each prompt: the token-for-token agreement with transformers is reported
   (informational: bf16 rounding can flip a near-tie and then the continuations diverge).
Sampling defaults cannot leak in: the server runs with `--generation-config vllm` and every
request carries all sampling parameters explicitly (see vllm_engine.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..core.config import REPO_ROOT, load_model_profile
from .managed import http_json

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "train" / "fixture_prefs"


def _sequences(profile: Any, tok: Any, max_pairs: int) -> list[tuple[list[int], int]]:
    """(token ids of prompt + chosen completion, prompt length) per fixture pair."""
    from ..training.common import load_preference_dataset
    from ..training.render import end_of_turn_ids, render_pairs

    exs, _, _ = load_preference_dataset(FIXTURE)
    rep = render_pairs(exs[:max_pairs], tok, chat_template_kwargs=profile.chat_template_kwargs, eot_ids=end_of_turn_ids(tok, profile), max_length=None)
    return [(p.prompt_ids + p.chosen_ids, len(p.prompt_ids)) for p in rep.kept]


def _hf_token_logps(model: Any, seqs: list[tuple[list[int], int]], device: str) -> list[list[float]]:
    import torch

    from ..training.modeling import ensure_fp32_logits

    ensure_fp32_logits(model)
    out = []
    with torch.no_grad():
        for ids, plen in seqs:
            x = torch.tensor([ids], device=device)
            logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1).gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)
            out.append(lp[plen - 1 :].cpu().tolist())  # log-probs of completion tokens
    return out


def _hf_greedy(model: Any, tok: Any, eot: set[int], seqs: list[tuple[list[int], int]], device: str, n: int) -> list[list[int]]:
    import torch

    out = []
    with torch.no_grad():
        for ids, plen in seqs:
            x = torch.tensor([ids[:plen]], device=device)
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                               eos_token_id=sorted(eot), pad_token_id=next(iter(eot)))
            out.append(g[0, plen:].tolist())
    return out


def _vllm_token_logps(base_url: str, model: str, seqs: list[tuple[list[int], int]]) -> tuple[list[list[float]], bool]:
    logps, echo_ok = [], True
    for ids, plen in seqs:
        st, r = http_json(base_url + "/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                                       "prompt_logprobs": 0, "return_token_ids": True}, timeout=600)
        if st != 200:
            raise RuntimeError(f"vLLM scoring failed: HTTP {st}: {str(r)[:300]}")
        ch = r["choices"][0]
        echo_ok &= ch.get("prompt_token_ids") == ids
        per = []
        for pos in range(plen, len(ids)):
            entry = ch["prompt_logprobs"][pos] or {}
            tok_entry = entry.get(str(ids[pos]), entry.get(ids[pos]))
            per.append(float(tok_entry["logprob"]) if tok_entry else float("nan"))
        logps.append(per)
    return logps, echo_ok


def _vllm_greedy(base_url: str, model: str, eot: set[int], seqs: list[tuple[list[int], int]], n: int) -> list[list[int]]:
    from .vllm_engine import completion_ids

    out = []
    for ids, plen in seqs:
        st, r = http_json(base_url + "/completions", {"model": model, "prompt": ids[:plen], "max_tokens": n, "temperature": 0.0,
                                                       "top_k": 0, "stop_token_ids": sorted(eot), "skip_special_tokens": False,
                                                       "return_token_ids": True}, timeout=600)
        if st != 200:
            raise RuntimeError(f"vLLM generation failed: HTTP {st}: {str(r)[:300]}")
        out.append(completion_ids(r, eot))
    return out


def compare(ref: list[list[float]], got: list[list[float]]) -> dict[str, Any]:
    diffs = [abs(a - b) for r, g in zip(ref, got) for a, b in zip(r, g)]
    seq = [abs(sum(r) - sum(g)) / max(len(r), 1) for r, g in zip(ref, got)]
    return {"n_tokens": len(diffs), "mean_abs": sum(diffs) / max(len(diffs), 1), "max_abs": max(diffs, default=0.0),
            "max_seq_abs_per_token": max(seq, default=0.0)}


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def _prefix_agreement(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run(profile_name: str, package: str | None, engine_args: list[str] | None, adapter: Path | None, max_pairs: int,
        greedy_tokens: int, thresholds: dict[str, float]) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model

    from ..training.modeling import apply_zero_lora, free_memory, load_base_model
    from ..training.render import end_of_turn_ids, load_tokenizer
    from .vllm_engine import VllmServer, ensure_engine, write_zero_adapter

    profile = load_model_profile(profile_name)
    sb = profile.serving.get("vllm")
    package = package or (sb.engine_package if sb else None)
    if not package:
        raise SystemExit(f"{profile_name}: no serving.vllm.engine_package; pass --engine-package")
    engine_args = list(sb.launch_args if sb else []) if engine_args is None else engine_args
    tok = load_tokenizer(profile)
    eot = end_of_turn_ids(tok, profile)
    seqs = _sequences(profile, tok, max_pairs)
    spec = {"r": 16, "alpha": 32, "target_modules": sorted(profile.lora_target_modules), "exclude_modules": profile.lora_exclude_modules}
    work = Path(tempfile.mkdtemp(prefix="equivalence-"))
    zero_dir = write_zero_adapter(profile, spec, work)

    # reference: transformers + PEFT (zero LoRA, then a random LoRA saved for vLLM)
    base = load_base_model(profile, "cuda")
    model = apply_zero_lora(base, spec).eval()
    hf_zero = _hf_token_logps(model, seqs, "cuda")
    hf_greedy = _hf_greedy(model, tok, eot, seqs, "cuda", greedy_tokens)
    model = model.unload()  # the plain base model again
    model = get_peft_model(model, LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], target_modules=spec["target_modules"],
                                             exclude_modules=spec["exclude_modules"], lora_dropout=0.0, task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(0)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.copy_((torch.randn(p.shape, generator=gen) * 0.1).to(p.dtype))  # an effect well above bf16 noise
    model.eval()
    hf_rand = _hf_token_logps(model, seqs, "cuda")
    rand_dir = work / "random-lora"
    model.save_pretrained(str(rand_dir))
    hf_trained = None
    if adapter is not None:
        from ..training.modeling import load_adapter

        model = load_adapter(model.unload(), Path(adapter), trainable=False).eval()
        hf_trained = _hf_token_logps(model, seqs, "cuda")
    del model, base
    free_memory()

    # vLLM, same token ids
    loras = {"zero": zero_dir, "random": rand_dir, **({"trained": Path(adapter)} if adapter is not None else {})}
    server = VllmServer(ensure_engine(package), profile, loras, max_context=32768,
                        engine_args=[*engine_args, "--gpu-memory-utilization", "0.6"])  # the reference model may still hold memory here
    try:
        v_zero, echo_zero = _vllm_token_logps(server.base_url, "zero", seqs)
        v_rand, echo_rand = _vllm_token_logps(server.base_url, "random", seqs)
        v_trained = _vllm_token_logps(server.base_url, "trained", seqs)[0] if adapter is not None else None
        v_plain, _ = _vllm_token_logps(server.base_url, "base", seqs)
        v_greedy = _vllm_greedy(server.base_url, "zero", eot, seqs, greedy_tokens)
    finally:
        server.close()

    zero_cmp, rand_cmp = compare(hf_zero, v_zero), compare(hf_rand, v_rand)
    trained_cmp = compare(hf_trained, v_trained) if hf_trained is not None else None
    eff_hf = [sum(r) - sum(z) for r, z in zip(hf_rand, hf_zero)]
    eff_v = [sum(r) - sum(z) for r, z in zip(v_rand, v_zero)]
    # tolerance: a fraction of the effect plus twice the per-sequence noise measured on the zero LoRA
    noise = [abs(sum(a) - sum(b)) for a, b in zip(hf_zero, v_zero)]
    eff_ok = [abs(a - b) <= thresholds["max_effect_rel"] * abs(a) + 2 * n for a, b, n in zip(eff_hf, eff_v, noise)]
    eff_rel = [abs(a - b) / max(abs(a), 1e-6) for a, b in zip(eff_hf, eff_v)]
    tok_hf = [x - y for r, z in zip(hf_rand, hf_zero) for x, y in zip(r, z)]
    tok_v = [x - y for r, z in zip(v_rand, v_zero) for x, y in zip(r, z)]
    corr = _corr(tok_hf, tok_v)
    checks = {
        "prompt_ids_echoed": echo_zero and echo_rand,
        "zero_lora_matches": zero_cmp["mean_abs"] <= thresholds["max_mean_abs"] and zero_cmp["max_seq_abs_per_token"] <= thresholds["max_seq_abs"],
        "trained_adapter_matches": trained_cmp is not None and trained_cmp["mean_abs"] <= thresholds["max_mean_abs"]
        and trained_cmp["max_seq_abs_per_token"] <= thresholds["max_seq_abs"],
        "adapter_effect_agrees": all(eff_ok) and corr >= thresholds["min_effect_corr"]
        and sum(abs(a) for a in eff_hf) / len(eff_hf) > 10 * max(sum(noise) / len(noise), 1e-3),  # the adapter must matter
    }
    return {
        "profile": profile_name,
        "engine_package": package,
        "engine_args": engine_args,
        "trained_adapter": str(adapter) if adapter is not None else None,
        "n_sequences": len(seqs),
        "completion_tokens": sum(len(s[0]) - s[1] for s in seqs),
        "thresholds": thresholds,
        "zero_lora_vs_hf": zero_cmp,
        "trained_adapter_vs_hf": trained_cmp,
        "random_lora_vs_hf": rand_cmp,
        "vllm_zero_lora_vs_vllm_base": compare(v_plain, v_zero),
        "adapter_effect_sum_logp": {"hf": eff_hf, "vllm": eff_v, "max_rel_diff": max(eff_rel), "zero_lora_noise": noise,
                                    "within_tolerance": eff_ok, "per_token_correlation": corr},
        "greedy_prefix_agreement": [{"agree": _prefix_agreement(a, b), "hf_len": len(a), "vllm_len": len(b)} for a, b in zip(hf_greedy, v_greedy)],
        "checks": checks,
        "ok": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learning_loop.serving.equivalence", description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True, help="model profile name (configs/models/<name>.yaml)")
    ap.add_argument("--engine-package", default=None, help="pip requirement of the vLLM environment (default: the profile's serving.vllm.engine_package)")
    ap.add_argument("--engine-arg", action="append", default=None, help="vLLM server flag (repeatable; default: the profile's serving.vllm.launch_args)")
    ap.add_argument("--adapter", type=Path, default=None, help="a trained PEFT adapter of this model (e.g. a published checkpoint dir); required for the check to pass")
    ap.add_argument("--max-pairs", type=int, default=3, help="fixture pairs to compare")
    ap.add_argument("--greedy-tokens", type=int, default=48, help="tokens of greedy decoding to compare per prompt")
    ap.add_argument("--max-mean-abs", type=float, default=0.05, help="bound on mean per-token |delta log-prob| (nats)")
    ap.add_argument("--max-seq-abs", type=float, default=0.05, help="bound on per-sequence |delta sum log-prob| per completion token")
    ap.add_argument("--max-effect-rel", type=float, default=0.1, help="relative part of the bound on the random adapter's effect disagreement (plus twice the zero-LoRA noise)")
    ap.add_argument("--min-effect-corr", type=float, default=0.95, help="minimum per-token correlation of the adapter's effect between the two paths")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    a = ap.parse_args(argv)
    report = run(a.profile, a.engine_package, a.engine_arg, a.adapter, a.max_pairs, a.greedy_tokens,
                 {"max_mean_abs": a.max_mean_abs, "max_seq_abs": a.max_seq_abs, "max_effect_rel": a.max_effect_rel,
                  "min_effect_corr": a.min_effect_corr})
    text = json.dumps(report, indent=2)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
