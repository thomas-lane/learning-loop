"""One numerical path for DPO log-probs: fp32 logits, fp32 sum, no Trainer mixed precision.

CPU (and MPS when available) only; no model download. The CUDA path (TRL's fused Triton
log-prob kernel on the same fp32 logits) is not exercised here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from learning_loop.training.common import TrainingRequestError  # noqa: E402
from learning_loop.training.modeling import (  # noqa: E402
    FP32_LOGITS_HOOK_ATTR,
    completion_logps,
    completion_window,
    ensure_fp32_logits,
    sequence_logps,
)
from learning_loop.training.render import RenderedPair  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else [])


def _reference_fp64(logits, ids, cm):
    logits, ids, cm = logits.cpu(), ids.cpu(), cm.cpu()  # MPS has no float64
    lp = torch.log_softmax(logits[..., :-1, :].double(), dim=-1).gather(-1, ids[..., 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * cm[..., 1:]).sum(dim=1)


@pytest.mark.parametrize("device", DEVICES)
def test_completion_logps_upcasts_bf16_logits_and_sums_in_fp32(device):
    g = torch.Generator().manual_seed(0)
    V, T = 4096, 96
    logits32 = (torch.randn(2, T, V, generator=g) * 4).to(device)
    logits16 = logits32.to(torch.bfloat16)
    ids = torch.randint(0, V, (2, T), generator=g).to(device)
    cm = torch.zeros(2, T, dtype=torch.long)
    cm[:, 20:] = 1
    cm = cm.to(device)

    out16 = completion_logps(logits16, ids, cm)
    assert out16.dtype == torch.float32
    # bf16 input == its exact fp32 upcast: the bf16 weights never change the log-softmax path
    out_up = completion_logps(logits16.float(), ids, cm)
    assert torch.equal(out16, out_up)
    ref = _reference_fp64(logits16.float(), ids, cm).float()
    assert torch.allclose(out16.cpu(), ref.cpu(), atol=1e-3, rtol=0)
    # what a bf16 log-softmax + bf16 sum would give is visibly off (the defect this guards against)
    lp16 = torch.log_softmax(logits16[..., :-1, :], dim=-1).gather(-1, ids[..., 1:].unsqueeze(-1)).squeeze(-1)
    naive16 = (lp16 * cm[..., 1:]).sum(dim=1)
    assert (naive16.float().cpu() - ref.cpu()).abs().max() > 1e-2
    # prompt positions contribute nothing
    cm0 = torch.zeros_like(cm)
    assert torch.equal(completion_logps(logits16, ids, cm0), torch.zeros(2, device=device))


def _tiny_model(dtype, device):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=32, vocab_size=512, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).to(dtype=dtype, device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_logits_hook_is_idempotent_and_shared_by_every_forward(device):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_tiny_model(torch.bfloat16, device),
                           LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ids = torch.randint(0, 512, (1, 12), device=device)
    assert model(input_ids=ids).logits.dtype == torch.bfloat16
    head = ensure_fp32_logits(model)
    handle = getattr(head, FP32_LOGITS_HOOK_ATTR)
    assert ensure_fp32_logits(model) is head and getattr(head, FP32_LOGITS_HOOK_ATTR) is handle
    assert len(head._forward_hooks) == 1
    out = model(input_ids=ids).logits  # the forward TRL's DPO loss calls
    assert out.dtype == torch.float32
    handle.remove()
    raw = model(input_ids=ids).logits
    assert torch.equal(out, raw.float())  # exact upcast, no other change


@pytest.mark.parametrize("device", DEVICES)
def test_sequence_logps_matches_trl_loss_path_on_bf16_model(device):
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    from trl.trainer.utils import selective_log_softmax_and_entropy

    model = _tiny_model(torch.bfloat16, device)
    pair = RenderedPair(pair_id="p", example_sha256="e" * 64, prompt_ids=list(range(5, 25)),
                        chosen_ids=[30, 31, 32, 33, 2], rejected_ids=[40, 41, 2],
                        prompt_text="", chosen_text="", rejected_text="")
    ours = sequence_logps(model, [pair], device, pad_token_id=0)[0]
    # TRL DPOTrainer._compute_loss (non-liger): selective_log_softmax_and_entropy on shifted logits, sum(dim=1)
    b = DataCollatorForPreference(pad_token_id=0)([pair.row()])
    ids, am, cm = (b[k].to(device) for k in ("input_ids", "attention_mask", "completion_mask"))
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
        assert logits.dtype == torch.float32  # hook installed by sequence_logps
        lp, _ = selective_log_softmax_and_entropy(logits[..., :-1, :], ids[..., 1:], entropy_requires_grad=False,
                                                  row_mask=cm[..., 1:])
        trl = lp.sum(dim=1).cpu()
    assert lp.dtype == torch.float32
    # ours runs lm_head on the completion window only; same values up to matmul rounding
    assert abs(ours[0] - float(trl[0])) < 1e-3 and abs(ours[1] - float(trl[1])) < 1e-3


def test_completion_window_covers_every_completion_token():
    cm = torch.tensor([[0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1, 1]])
    k = completion_window(cm)
    assert k == 5  # from the last prompt token of row 0 (position 2) to the end
    assert cm[:, -k:][:, 1:].sum() == cm[:, 1:].sum()  # no completion label lost after the shift
    assert completion_window(torch.tensor([[0, 1, 1]])) == 3


def _pair():
    return RenderedPair(pair_id="p", example_sha256="e" * 64, prompt_ids=list(range(5, 45)),
                        chosen_ids=[30, 31, 32, 33, 2], rejected_ids=[40, 41, 2],
                        prompt_text="", chosen_text="", rejected_text="")


def test_windowed_trainer_loss_equals_trl_full_logits_loss(tmp_path):
    """The trainer's loss (completion window, `logits_to_keep`) equals TRL's own `_compute_loss`
    on full logits, gradients included, and lm_head never sees the prompt positions."""
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from trl import DPOConfig, DPOTrainer
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    from learning_loop.training.dpo import _make_trainer_class

    model = get_peft_model(_tiny_model(torch.float32, "cpu"),
                           LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ensure_fp32_logits(model)
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token="<unk>")),
                                  pad_token="<pad>", eos_token="<unk>")
    from datasets import Dataset

    row = {**_pair().row(), "ref_chosen_logps": -10.0, "ref_rejected_logps": -12.0}
    args = DPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False, report_to="none",
                     precompute_ref_log_probs=True, max_length=None, disable_dropout=True)
    trainer = _make_trainer_class()(model=model, args=args, train_dataset=Dataset.from_list([row]), processing_class=tok)
    b = DataCollatorForPreference(pad_token_id=0)([row])
    b["ref_chosen_logps"], b["ref_rejected_logps"] = torch.tensor([-10.0]), torch.tensor([-12.0])
    seen = []
    hook = model.get_output_embeddings().register_forward_hook(lambda _m, _i, out: seen.append(out.shape[1]))

    def loss_and_grads(fn):
        model.zero_grad()
        loss = fn(trainer, model, b, False)
        loss.backward()
        return float(loss.detach()), [p.grad.clone() for p in model.parameters() if p.requires_grad]

    ours, g_ours = loss_and_grads(type(trainer)._compute_loss)
    full, g_full = loss_and_grads(DPOTrainer._compute_loss)
    hook.remove()
    assert seen[0] == completion_window(b["completion_mask"]) < b["input_ids"].shape[1] == seen[1]
    assert abs(ours - full) < 1e-5
    assert all(torch.allclose(a, c, atol=1e-6) for a, c in zip(g_ours, g_full))


def test_trainer_mixed_precision_is_refused(monkeypatch):
    from learning_loop.training.dpo import LOGP_PATH, _precision_path

    ok = SimpleNamespace(accelerator=SimpleNamespace(mixed_precision="no", native_amp=False), model=torch.nn.Linear(1, 1))
    info = _precision_path(ok, "bfloat16")
    assert info["trainer_mixed_precision"] == "no" and info["logp_path"] == LOGP_PATH
    bad = SimpleNamespace(accelerator=SimpleNamespace(mixed_precision="bf16", native_amp=True), model=torch.nn.Linear(1, 1))
    with pytest.raises(TrainingRequestError, match="mixed precision"):
        _precision_path(bad, "bfloat16")
    wrapped = torch.nn.Linear(1, 1)
    wrapped._original_forward = wrapped.forward  # what accelerate.prepare_model leaves behind under native_amp
    with pytest.raises(TrainingRequestError):
        _precision_path(SimpleNamespace(accelerator=SimpleNamespace(mixed_precision="no", native_amp=False), model=wrapped), "bfloat16")


def test_dpo_config_as_built_has_no_mixed_precision(monkeypatch, tmp_path):
    """bf16/fp16 are passed False explicitly: TRL's own default is bf16=True when fp16 is unset."""
    from trl import DPOConfig

    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    assert DPOConfig(output_dir=str(tmp_path), bf16=False, fp16=False, use_cpu=True, report_to="none").mixed_precision == "no"
    # an environment override would re-enable it; _precision_path catches that at Trainer construction
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    assert DPOConfig(output_dir=str(tmp_path), bf16=False, fp16=False, use_cpu=True, report_to="none").mixed_precision == "bf16"


def test_reference_cache_key_carries_logp_path():
    from learning_loop.core.records import CheckpointRef
    from learning_loop.training.dpo import LOGP_PATH, reference_cache_key

    ref = CheckpointRef(checkpoint_id="c", model_profile="qwen3-0.6b", base_model="Qwen/Qwen3-0.6B",
                        base_revision="c" * 40, adapter_path=None, adapter_sha256=None)
    pair = RenderedPair(pair_id="p", example_sha256="e" * 64, prompt_ids=[1], chosen_ids=[2], rejected_ids=[3],
                        prompt_text="", chosen_text="", rejected_text="")
    assert reference_cache_key(ref, pair, "t", "m", {}, "bfloat16")["logp_path"] == LOGP_PATH


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_lora_is_applied_and_changes_no_output(dtype):
    """Base checkpoints are served through an all-zero LoRA (same adapter overhead as trained
    checkpoints): the adapter modules are really present and the outputs are bit-identical."""
    from learning_loop.training.modeling import apply_zero_lora

    base = _tiny_model(dtype, "cpu").eval()
    ids = torch.randint(0, 512, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = base(input_ids=ids).logits.clone()
        greedy = base.generate(input_ids=ids[:1], max_new_tokens=8, do_sample=False).clone()
    model = apply_zero_lora(base, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj", "down_proj"], "exclude_modules": None}).eval()
    lora = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    assert len(lora) == 3  # one layer x 3 targets
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, want)
        assert torch.equal(model.generate(input_ids=ids[:1], max_new_tokens=8, do_sample=False), greedy)
