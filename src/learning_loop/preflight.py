"""Preflight checks for a local/remote host before a run or smoke test.

Each check returns a structured `CheckResult` with status ok | warn | fail | skipped.
"skipped" means the check did not run (e.g. the train extra is absent) and is never
reported as success.

    python -m learning_loop.preflight --profile qwen3-0.6b [--no-docker] [--no-train] [--json]

Exit status 1 when any check failed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["ok", "warn", "fail", "skipped"]


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


def _gb(n: int | float | None) -> float | None:
    return None if n is None else round(n / 2**30, 1)


# --------------------------------------------------------------------------- #
# Host
# --------------------------------------------------------------------------- #


def check_host() -> CheckResult:
    return CheckResult("host", "ok", f"{platform.system()} {platform.machine()} python {platform.python_version()}",
                       {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()})


def check_docker(timeout_sec: float = 20.0) -> CheckResult:
    exe = shutil.which("docker")
    if exe is None:
        return CheckResult("docker", "fail", "docker CLI not found on PATH")
    try:
        out = subprocess.run([exe, "info", "--format", "{{json .}}"], capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return CheckResult("docker", "fail", f"`docker info` timed out after {timeout_sec}s")
    if out.returncode != 0:
        return CheckResult("docker", "fail", f"docker daemon not reachable: {out.stderr.strip()[:300]}")
    try:
        info = json.loads(out.stdout)
    except json.JSONDecodeError:
        return CheckResult("docker", "fail", "could not parse `docker info` output")
    arch = str(info.get("Architecture") or "")
    host = platform.machine().lower()
    norm = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}
    data = {"server_version": info.get("ServerVersion"), "architecture": arch, "os_type": info.get("OSType"),
            "ncpu": info.get("NCPU"), "mem_total_gb": _gb(info.get("MemTotal")), "host_machine": host}
    if norm.get(arch.lower()) != norm.get(host):
        return CheckResult("docker", "warn", f"docker arch {arch} != host {host} (emulation is slow and may change timing)", data)
    return CheckResult("docker", "ok", f"docker {data['server_version']} {arch}, {data['ncpu']} CPUs, {data['mem_total_gb']} GB", data)


def check_disk(path: str | Path = ".", min_free_gb: float = 20.0) -> CheckResult:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    u = shutil.disk_usage(p)
    data = {"path": str(p.resolve()), "free_gb": _gb(u.free), "total_gb": _gb(u.total), "min_free_gb": min_free_gb}
    status: Status = "ok" if u.free / 2**30 >= min_free_gb else "fail"
    return CheckResult("disk", status, f"{data['free_gb']} GB free at {data['path']} (need {min_free_gb})", data)


def total_memory_bytes() -> int | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip())
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return None


def check_memory(min_total_gb: float = 16.0) -> CheckResult:
    total = total_memory_bytes()
    if total is None:
        return CheckResult("memory", "skipped", "total memory not measurable on this platform")
    data = {"total_gb": _gb(total), "min_total_gb": min_total_gb}
    status: Status = "ok" if total / 2**30 >= min_total_gb else "fail"
    return CheckResult("memory", status, f"{data['total_gb']} GB total (need {min_total_gb})", data)


# --------------------------------------------------------------------------- #
# Model access
# --------------------------------------------------------------------------- #


def check_model_access(profile_name: str, allow_network: bool = False) -> CheckResult:
    from .config import load_model_profile

    prof = load_model_profile(profile_name)
    data: dict[str, Any] = {"base_model": prof.base_model, "base_revision": prof.base_revision}
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(prof.base_model, revision=prof.base_revision, local_files_only=True,
                                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"]))
        weights = sorted(p.name for p in snap.glob("*.safetensors"))
        data.update(snapshot=str(snap), weights=weights)
        cached = bool(weights)
    except Exception as e:  # noqa: BLE001 - LocalEntryNotFoundError and friends
        data["cache_error"] = f"{type(e).__name__}: {e}"[:300]
        cached = False
    if cached:
        try:
            from .training.render import chat_template_sha256, load_tokenizer

            tok = load_tokenizer(prof, local_files_only=True)
            data["chat_template_sha256"] = chat_template_sha256(tok)
        except Exception as e:  # noqa: BLE001
            return CheckResult("model_access", "fail", f"cached, but tokenizer/template check failed: {e}", data)
        return CheckResult("model_access", "ok", f"{prof.base_model}@{prof.base_revision[:12]} cached; template hash verified", data)
    if not allow_network:
        return CheckResult("model_access", "fail", f"{prof.base_model}@{prof.base_revision[:12]} not in the local HF cache (network check disabled)", data)
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(prof.base_model, revision=prof.base_revision, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings or [] if s.rfilename.endswith(".safetensors"))
        data["download_gb"] = _gb(size)
        return CheckResult("model_access", "warn", f"not cached; downloadable from the hub (~{data['download_gb']} GB)", data)
    except Exception as e:  # noqa: BLE001
        return CheckResult("model_access", "fail", f"not cached and hub lookup failed: {type(e).__name__}: {e}"[:300], data)


# --------------------------------------------------------------------------- #
# Torch / MPS / adapter trainability
# --------------------------------------------------------------------------- #


def check_accelerator() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("accelerator", "skipped", "torch not installed (install the `train` extra)")
    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps = bool(mps_built and torch.backends.mps.is_available())
    cuda = torch.cuda.is_available()
    data = {"torch": torch.__version__, "mps_built": mps_built, "mps_available": mps, "cuda_available": cuda,
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []}
    if cuda:
        return CheckResult("accelerator", "ok", f"CUDA: {data['cuda_devices']}", data)
    if mps:
        return CheckResult("accelerator", "ok", "MPS available", data)
    return CheckResult("accelerator", "warn", "no CUDA/MPS; training needs an explicit CPU fallback", data)


def check_adapter_trainability(profile_name: str, device: str = "auto", allow_cpu_fallback: bool = False) -> CheckResult:
    """Tiny randomly initialised model of the profile's architecture family: LoRA forward/backward/step."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
    except ImportError:
        return CheckResult("adapter_trainability", "skipped", "train extra (torch/peft) not installed")
    from .config import load_model_profile
    from .training.common import TrainingRequestError, resolve_device

    prof = load_model_profile(profile_name)
    try:
        dev = resolve_device(device, allow_cpu_fallback, prof.supported_train_devices)
    except TrainingRequestError as e:  # unavailable, or not in the profile's supported_train_devices
        return CheckResult("adapter_trainability", "fail", str(e),
                           {"supported_train_devices": prof.supported_train_devices or None, "requested": device})
    small = dict(num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                 num_key_value_heads=1, head_dim=32, vocab_size=512, max_position_embeddings=128)
    arch = "llama_tiny"
    try:
        cfg = AutoConfig.from_pretrained(prof.base_model, revision=prof.base_revision, local_files_only=True)
        if cfg.model_type in ("qwen3", "qwen2", "llama", "mistral"):
            for k, v in small.items():
                setattr(cfg, k, v)
            arch = f"{cfg.model_type}_tiny"
        else:
            cfg = LlamaConfig(**small)
    except OSError:
        cfg = LlamaConfig(**small)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(dev["device"])
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in prof.lora_target_modules if t in names] or ["q_proj", "v_proj"]
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    opt = torch.optim.SGD(params, lr=0.1)
    ids = torch.randint(0, 512, (1, 16), device=dev["device"])
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    opt.step()
    changed = sum(int(not torch.equal(b, p.detach())) for b, p in zip(before, params))
    data = {"device": dev, "arch": arch, "targets": targets, "loss": loss.item(), "n_lora_tensors": len(params),
            "n_changed": changed, "scope": "tiny random model; checks device + PEFT stack, not the full model"}
    del model, opt
    ok = grad_ok and changed > 0 and torch.isfinite(loss).item()
    return CheckResult("adapter_trainability", "ok" if ok else "fail",
                       f"LoRA step on {dev['device']} ({arch}): loss={loss.item():.3f}, {changed}/{len(params)} tensors changed", data)


# --------------------------------------------------------------------------- #


def run_preflight(
    profile: str | None,
    *,
    docker: bool = True,
    train: bool = True,
    device: str = "auto",
    allow_cpu_fallback: bool = False,
    allow_network: bool = False,
    disk_path: str | Path = "runs",
    min_free_gb: float = 20.0,
    min_memory_gb: float = 16.0,
) -> list[CheckResult]:
    results = [check_host()]
    results.append(check_docker() if docker else CheckResult("docker", "skipped", "not requested"))
    results.append(check_disk(disk_path, min_free_gb))
    results.append(check_memory(min_memory_gb))
    if profile:
        results.append(check_model_access(profile, allow_network=allow_network))
    else:
        results.append(CheckResult("model_access", "skipped", "no model profile given"))
    if train:
        results.append(check_accelerator())
        results.append(check_adapter_trainability(profile, device, allow_cpu_fallback) if profile
                       else CheckResult("adapter_trainability", "skipped", "no model profile given"))
    else:
        results.append(CheckResult("accelerator", "skipped", "not requested"))
        results.append(CheckResult("adapter_trainability", "skipped", "not requested"))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learning_loop.preflight")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--allow-network", action="store_true", help="look the model up on the hub when not cached")
    ap.add_argument("--disk-path", default=os.environ.get("LOOP_RUNS_DIR", "runs"))
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = run_preflight(a.profile, docker=not a.no_docker, train=not a.no_train, device=a.device,
                        allow_cpu_fallback=a.allow_cpu_fallback, allow_network=a.allow_network,
                        disk_path=a.disk_path, min_free_gb=a.min_free_gb)
    if a.json:
        print(json.dumps([asdict(r) for r in res], indent=2, default=str))
    else:
        for r in res:
            print(f"{r.status.upper():8s} {r.name:22s} {r.detail}")
    return 1 if any(r.status == "fail" for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())
