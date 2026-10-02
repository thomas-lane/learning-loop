"""vLLM as a token engine behind hf_server's API (`hf_server --engine vllm`).

hf_server keeps everything that defines what the learner sees and what is counted: prompts are
rendered with training/render.py and sent to vLLM as token ids, the generated token ids come
back and are decoded and parsed by our tool-call parser, and usage is counted with the training
tokenizer. vLLM only generates. It runs as a child process of hf_server (same process group, so
stopping the server stops it), from its own environment (`.engines/<package>/`, installed on the
serving host on first use: vLLM pins its own torch), with:

- `--generation-config vllm`: none of the model repository's sampling defaults; every sampling
  parameter is sent explicitly with each request (top_k disabled, as in the HF engine);
- prefix caching off: a cached prefix takes a different numerical path than a recomputed one;
- the served checkpoint as a LoRA module: a published adapter (hash-checked by hf_server), or for
  base checkpoints an all-zero adapter of the experiment's LoRA shape written by
  `write_zero_adapter` (the same adapter overhead for every checkpoint, base outputs unchanged);
- `--max-num-seqs` = the requested concurrency (1 = unbatched).

Equivalence with the HF engine (prompt ids, log-probs of base and adapter, adapter applied) is
checked on a GPU host with `python -m learning_loop.serving.equivalence`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.config import REPO_ROOT, ModelProfile
from ..core.storage import sha256_json
from .hf_server import ApiError, ServingCore
from .managed import free_port, http_json

ENGINES_DIR = REPO_ROOT / ".engines"
INSTALLED_MARKER = ".installed"


def engine_dir(package: str) -> Path:
    return ENGINES_DIR / re.sub(r"[^A-Za-z0-9._-]", "_", package)


def ensure_engine(package: str) -> Path:
    """Python of a separate environment with `package` installed (created on first use)."""
    d = engine_dir(package)
    py = d / "bin" / "python"
    marker = d / INSTALLED_MARKER
    if py.exists() and marker.exists() and marker.read_text().strip() == package:
        return py
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is needed to install the serving engine (scripts/setup_gpu_host.sh installs it)")
    if d.exists():
        shutil.rmtree(d)  # a previous install did not finish
    d.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "engine_install", "package": package, "dir": str(d)}), file=sys.stderr, flush=True)
    subprocess.run([uv, "venv", "--python", "3.12", str(d)], check=True, stdout=sys.stderr, stderr=sys.stderr)
    # ninja: vLLM/FlashInfer compile kernels at startup and call the `ninja` binary from PATH
    subprocess.run([uv, "pip", "install", "--python", str(py), package, "ninja"], check=True, stdout=sys.stderr, stderr=sys.stderr)
    marker.write_text(package)
    return py


def zero_adapter_tensors(model: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """PEFT-format LoRA tensors, all zero, for every module of `model` that PEFT would adapt with
    `spec` (nn.Linear whose name ends in a target and does not fully match `exclude_modules`)."""
    import torch

    targets = set(spec["target_modules"])
    exclude = re.compile(spec["exclude_modules"]) if spec.get("exclude_modules") else None
    r = int(spec["r"])
    out: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        if exclude is not None and exclude.fullmatch(name):
            continue
        out[f"base_model.model.{name}.lora_A.weight"] = torch.zeros(r, mod.in_features, dtype=torch.float32)
        out[f"base_model.model.{name}.lora_B.weight"] = torch.zeros(mod.out_features, r, dtype=torch.float32)
    if not out:
        raise ValueError(f"zero LoRA {spec}: no module matches the target modules")
    return out


def zero_adapter_config(profile: ModelProfile, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": profile.base_model,
        "revision": profile.base_revision,
        "r": int(spec["r"]),
        "lora_alpha": int(spec["alpha"]),
        "lora_dropout": 0.0,
        "target_modules": sorted(spec["target_modules"]),
        "exclude_modules": spec.get("exclude_modules"),
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "init_lora_weights": True,
        "modules_to_save": None,
    }


def write_zero_adapter(profile: ModelProfile, spec: dict[str, Any], root: Path) -> Path:
    """An all-zero PEFT adapter directory for `profile` with LoRA shape `spec` (module shapes come
    from the model config on the meta device: no weights are loaded). Cached by content."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    key = sha256_json({"base": profile.base_model, "revision": profile.base_revision, **spec})[:16]
    d = Path(root) / f"zero-lora-{key}"
    if (d / "adapter_model.safetensors").exists() and (d / "adapter_config.json").exists():
        return d
    try:
        cfg = AutoConfig.from_pretrained(profile.base_model, revision=profile.base_revision, local_files_only=True)
    except OSError:
        cfg = AutoConfig.from_pretrained(profile.base_model, revision=profile.base_revision)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    tensors = zero_adapter_tensors(model, spec)
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(tensors, str(tmp / "adapter_model.safetensors"))
    (tmp / "adapter_config.json").write_text(json.dumps(zero_adapter_config(profile, spec), indent=2, sort_keys=True))
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    return d


def adapter_rank(adapter_dir: Path) -> int:
    return int(json.loads((Path(adapter_dir) / "adapter_config.json").read_text())["r"])


class VllmServer:
    """A vLLM OpenAI-compatible server child process on a free loopback port, serving the base
    model and the given LoRA modules {name: adapter dir}. It shares the caller's process group."""

    def __init__(self, python: Path, profile: ModelProfile, loras: dict[str, Path], max_context: int, concurrency: int = 1,
                 engine_args: list[str] | None = None, start_timeout_sec: float = 1800.0):
        self.port = free_port()
        self.argv = [
            str(python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", profile.base_model, "--revision", profile.base_revision, "--served-model-name", "base",
            "--host", "127.0.0.1", "--port", str(self.port), "--dtype", profile.training_dtype,
            "--generation-config", "vllm", "--no-enable-prefix-caching", "--max-num-seqs", str(concurrency),
            "--max-model-len", str(max_context), "--seed", "0",
            "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0, "video": 0}),
        ]
        if loras:
            self.argv += ["--enable-lora", "--max-loras", str(len(loras)), "--max-lora-rank", str(max(adapter_rank(d) for d in loras.values())),
                          "--lora-modules", *[f"{n}={d}" for n, d in loras.items()]]
        self.argv += list(engine_args or [])
        self.names = set(loras) | {"base"}
        print(json.dumps({"event": "engine_start", "argv": self.argv}), file=sys.stderr, flush=True)
        # The engine environment's bin/ first on PATH (as if activated): its tools (ninja) are found.
        env = {**os.environ, "PATH": f"{Path(python).parent}{os.pathsep}{os.environ.get('PATH', '')}", "VIRTUAL_ENV": str(Path(python).parent.parent)}
        self.proc = subprocess.Popen(self.argv, stdout=sys.stderr, stderr=sys.stderr, env=env)
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        deadline = time.time() + start_timeout_sec
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode} while starting")
            try:
                st, data = http_json(self.base_url + "/models", timeout=5)
                if st == 200 and self.names <= {m.get("id") for m in data.get("data", [])}:
                    return
            except OSError:
                pass
            time.sleep(2)
        self.close()
        raise RuntimeError(f"vLLM not ready after {start_timeout_sec}s")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)


class VllmEngine(ServingCore):
    """Generation by a vLLM child process; everything else as in ServingCore."""

    engine_name = "vllm"

    def __init__(
        self,
        profile: ModelProfile,
        checkpoint_dir: Path | None,
        package: str,
        base_checkpoint_id: str = "base",
        zero_lora: dict[str, Any] | None = None,
        default_max_tokens: int = 1024,
        max_context: int | None = None,
        concurrency: int = 1,
        engine_args: list[str] | None = None,
        start_timeout_sec: float = 1800.0,
        zero_adapter_root: Path | None = None,
    ):
        super().__init__(profile, checkpoint_dir, base_checkpoint_id, zero_lora, default_max_tokens)
        self.package = package
        self.concurrency = concurrency
        self.max_context = max_context or 65536
        python = ensure_engine(package)
        if self.adapter_dir is not None:
            lora_dir: Path | None = self.adapter_dir
        elif zero_lora:
            lora_dir = write_zero_adapter(profile, zero_lora, zero_adapter_root or (REPO_ROOT / "runs" / "_servers"))
        else:
            lora_dir = None
        self.lora_name = self.served_model if lora_dir is not None else None
        self.server = VllmServer(python, profile, {self.lora_name: lora_dir} if lora_dir is not None else {},
                                 self.max_context, concurrency, engine_args, start_timeout_sec)
        self.base_url = self.server.base_url
        self.argv = self.server.argv

    def engine_info(self) -> dict[str, Any]:
        return {"engine_package": self.package, "device": "cuda", "concurrency": self.concurrency, "lora_module": self.lora_name,
                "engine_argv": self.argv}

    def request_body(self, ids: list[int], req: dict[str, Any], max_new: int) -> dict[str, Any]:
        greedy = req["temperature"] == 0
        return {
            "model": self.lora_name or "base",
            "prompt": ids,
            "max_tokens": max_new,
            "n": 1,
            "temperature": 0.0 if greedy else req["temperature"],
            "top_p": 1.0 if greedy else req["top_p"],
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": req["seed"],
            "stop_token_ids": sorted(self.eot),
            "skip_special_tokens": False,
            "return_token_ids": True,
        }

    def generate(self, ids: list[int], req: dict[str, Any], max_new: int) -> tuple[list[int], dict[str, Any]]:
        body = self.request_body(ids, req, max_new)
        st, r = http_json(self.base_url + "/completions", body, timeout=3600)
        if st != 200:
            raise ApiError(502, f"vLLM returned HTTP {st}: {str(r)[:500]}", "engine_error")
        return completion_ids(r, self.eot), {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens")}

    def close(self) -> None:
        self.server.close()


def completion_ids(response: dict[str, Any], eot: set[int]) -> list[int]:
    """Generated ids from a vLLM completion, ending with the end-of-turn id when generation stopped
    on one (whether or not vLLM included the stop token in `token_ids`)."""
    ch = response["choices"][0]
    new = list(ch.get("token_ids") or [])
    stop = ch.get("stop_reason")
    if ch.get("finish_reason") == "stop" and isinstance(stop, int) and stop in eot and (not new or new[-1] != stop):
        new.append(stop)
    return new
