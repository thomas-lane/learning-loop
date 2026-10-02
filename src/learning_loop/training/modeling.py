"""Model loading, adapter identity and log-probability helpers (torch; `train` extra).

Used by the DPO trainer, the reference HF server and preflight so that every
consumer loads base@revision + adapter the same way.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from ..core.config import ModelProfile
from ..core.records import CheckpointRef
from .common import TrainingRequestError, adapter_sha256
from .render import RenderedPair

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def torch_dtype(name: str) -> Any:
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_base_model(profile: ModelProfile, device: str, dtype: str | None = None) -> Any:
    """base_model@base_revision on `device`, cache first (no network when cached). On CUDA the
    weights are placed on the GPU as they are read (`device_map`), never assembled in CPU memory
    first; MPS/CPU load to CPU and move (transformers' threaded loader segfaults with
    `device_map="mps"` on macOS with transformers 5.17)."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"revision": profile.base_revision, "dtype": torch_dtype(dtype or profile.training_dtype)}
    if device == "cuda":
        kwargs["device_map"] = device
    try:
        model = AutoModelForCausalLM.from_pretrained(profile.base_model, local_files_only=True, **kwargs)
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(profile.base_model, **kwargs)
    return model if device == "cuda" else model.to(device)


def apply_zero_lora(base: Any, spec: dict[str, Any]) -> Any:
    """Wrap `base` in a LoRA whose B matrices are exactly zero (PEFT's default init), so every
    adapted layer adds exactly 0.0: outputs equal the base model's while each forward runs the same
    adapter operations as a trained checkpoint. Serving base checkpoints this way keeps timing
    comparable across cycles. `spec`: r, alpha, target_modules, exclude_modules."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=0.0, target_modules=list(spec["target_modules"]),
        exclude_modules=spec.get("exclude_modules"), task_type="CAUSAL_LM", init_lora_weights=True,
    )
    model = get_peft_model(base, cfg)
    b = [p for n, p in model.named_parameters() if "lora_B" in n]
    if not b or any(bool(p.detach().abs().max() != 0) for p in b):
        raise TrainingRequestError("zero LoRA: expected adapter B matrices that are all exactly zero")
    return model


def verify_adapter_dir(ref: CheckpointRef) -> Path:
    """The published adapter directory of `ref`, after checking its recorded hash."""
    if ref.adapter_path is None:
        raise TrainingRequestError(f"{ref.checkpoint_id} has no adapter")
    d = Path(ref.adapter_path)
    missing = [f for f in ADAPTER_FILES if not (d / f).exists()]
    if missing:
        raise TrainingRequestError(f"{d}: not a PEFT adapter directory (missing {missing})")
    if ref.adapter_sha256 is not None and adapter_sha256(d, ADAPTER_FILES) != ref.adapter_sha256:
        raise TrainingRequestError(f"{d}: adapter files do not match the recorded adapter_sha256 (mutated checkpoint?)")
    return d


def load_adapter(base: Any, adapter_dir: Path, trainable: bool) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="default", is_trainable=trainable)


def adapter_state_cpu(model: Any) -> dict[str, Any]:
    """LoRA weights of the "default" adapter (PEFT save-format keys), cloned to CPU."""
    from peft import get_peft_model_state_dict

    return {k: v.detach().to("cpu").clone() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}


def load_adapter_file_state(adapter_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(Path(adapter_dir) / "adapter_model.safetensors"), device="cpu")


def max_abs_diff(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Max |a-b| over identical key sets; None if the key sets differ."""
    if set(a) != set(b):
        return None
    worst = 0.0
    for k in a:
        if a[k].shape != b[k].shape:
            return None
        worst = max(worst, float((a[k].float() - b[k].float()).abs().max()) if a[k].numel() else 0.0)
    return worst


FP32_LOGITS_HOOK_ATTR = "_learning_loop_fp32_logits_hook"


def ensure_fp32_logits(model: Any) -> Any:
    """Make every forward of `model` return float32 logits (idempotent; returns the head module).

    A forward hook on the output embedding (lm_head) upcasts its output, so the DPO loss
    inside TRL, our reference/trained log-probs and the reload check all run log-softmax
    on the same float32 logits whatever the weight dtype. Upcasting bf16 -> fp32 is exact,
    and model-specific post-processing after lm_head (e.g. Gemma's final logit
    soft-capping) then also runs in fp32 on every path. This replaces Trainer mixed
    precision (accelerate autocast + convert_outputs_to_fp32), which is never enabled."""
    head = model.get_output_embeddings()
    if head is None:
        raise TrainingRequestError(f"{type(model).__name__} has no output embedding to hook")
    if getattr(head, FP32_LOGITS_HOOK_ATTR, None) is None:
        handle = head.register_forward_hook(lambda _m, _inp, out: out.float())
        setattr(head, FP32_LOGITS_HOOK_ATTR, handle)
    return head


def completion_window(completion_mask: Any) -> int:
    """How many trailing positions' logits the completion log-probs need: from the last prompt
    token of the row whose completion starts earliest to the end of the batch. Only these
    positions go through lm_head (`logits_to_keep`), so a long prompt never materializes
    float32 logits over the whole vocabulary for every prompt token."""
    first = int(completion_mask.int().argmax(dim=1).min())
    return min(completion_mask.shape[1], completion_mask.shape[1] - first + 1)


def completion_logps(logits: Any, input_ids: Any, completion_mask: Any) -> Any:
    """Summed completion log-probs per row (float32), the same shift / masked selective
    log-softmax / sum as TRL's DPO loss (trl/trainer/dpo_trainer.py `_compute_loss`).

    Logits are cast to float32 BEFORE the log-softmax (a no-op when the fp32 hook is
    installed) and the sum is taken in float32, so a bf16 model never produces bf16
    per-token log-probs or a bf16 sequence sum."""
    from trl.trainer.utils import selective_log_softmax

    logits = logits[..., :-1, :].float()
    labels = input_ids[..., 1:]
    mask = completion_mask[..., 1:]
    lp = selective_log_softmax(logits, labels, row_mask=mask)
    return (lp.float() * mask).sum(dim=1)


def sequence_logps(model: Any, pairs: list[RenderedPair], device: str, pad_token_id: int) -> list[tuple[float, float]]:
    """Summed completion log-probs (chosen, rejected) per pair, computed exactly like the DPO
    loss (same collator, same completion window, shift and masked log-softmax, float32 logits),
    in eval mode without grad. Installs the fp32-logits hook on `model` (see `ensure_fp32_logits`)."""
    import torch
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    ensure_fp32_logits(model)
    collator = DataCollatorForPreference(pad_token_id=pad_token_id)
    was_training = model.training
    model.eval()
    out: list[tuple[float, float]] = []
    try:
        with torch.no_grad():
            for p in pairs:
                b = collator([p.row()])
                ids = b["input_ids"].to(device)
                am = b["attention_mask"].to(device)
                cm = b["completion_mask"].to(device)
                k = completion_window(cm)
                logits = model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=k).logits
                s = completion_logps(logits, ids[:, -k:], cm[:, -k:]).cpu()
                out.append((float(s[0]), float(s[1])))
    finally:
        if was_training:
            model.train()
    return out


def free_memory(device: str | None = None) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
