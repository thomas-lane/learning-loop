"""dpo.split_pair_backward: running each sequence of a pair under its own checkpoint changes only
when activations are alive, never the log-probs or the gradients."""

import pytest

torch = pytest.importorskip("torch")

from learning_loop.training.dpo import pair_token_logps  # noqa: E402


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=96, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=32, vocab_size=128)
    m = LlamaForCausalLM(cfg).to(torch.float64)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # as the trainer does
    m.train()
    return m


@pytest.mark.parametrize("k", [3, 9])
def test_split_pair_forward_matches_the_batched_forward(k):
    model = _model()
    ids = torch.randint(0, 128, (2, 9))  # [chosen, rejected], padded to one length
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0  # right padding on one row, as TRL's collator pads
    cm = torch.zeros_like(ids)
    cm[0, 9 - k:], cm[1, 9 - k:7] = 1, 1
    kw = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
    weights = torch.randn(2, 8, dtype=torch.float64)

    def grads(split):
        model.zero_grad()
        lp, ent = pair_token_logps(model, kw, ids, cm, split)
        ((lp.double() * weights).sum() + ent.double().sum()).backward()
        return lp.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    logits_b, grads_b = grads(False)
    logits_s, grads_s = grads(True)
    assert logits_s.shape == (2, 8)
    assert torch.allclose(logits_s, logits_b, atol=1e-6)
    assert grads_s.keys() == grads_b.keys()
    assert all(torch.allclose(grads_s[n], grads_b[n], atol=1e-6) for n in grads_b)


def _gemma(num_kv_shared_layers):
    from transformers import AutoModelForCausalLM, Gemma4UnifiedTextConfig

    torch.manual_seed(0)
    cfg = Gemma4UnifiedTextConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=8,
                                  layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                                  num_kv_shared_layers=num_kv_shared_layers)
    m = AutoModelForCausalLM.from_config(cfg).to(torch.float64)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    return m


@pytest.mark.parametrize("shared", [0, 2])
def test_gemma4_split_matches_batched_and_releases_the_step(shared):
    """Gemma 4's layers keep K/V in a shared_kv_states dict; under the split's nested checkpoints
    that dict closes a reference cycle that kept every layer input of a step alive. Each row gets its
    own dict (also when layers share K/V), cleared after the backward pass: same gradients, no
    tensors of the step left behind."""
    import gc

    from learning_loop.training.dpo import takes_shared_kv_states

    model = _gemma(shared)
    assert takes_shared_kv_states(model)
    ids = torch.randint(0, 128, (2, 40))
    cm = torch.zeros_like(ids)
    cm[:, 30:] = 1
    kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}

    def live():
        gc.collect()
        return sum(1 for o in gc.get_objects() if torch.is_tensor(o) and o.dim() >= 3 and 40 in o.shape)

    def step(split, dicts):
        model.zero_grad()
        lp, _ = pair_token_logps(model, kw, ids, cm, split, dicts)
        lp.sum().backward()
        logits = lp.detach()
        for d in dicts or []:
            d.clear()
        return logits, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    before = live()
    logits_b, grads_b = step(False, None)
    logits_s, grads_s = step(True, [])
    # Gemma's RMSNorm computes in float32 even in a float64 model, so batch-1 and batch-2 runs
    # round differently (measured: 2e-7 relative).
    assert torch.allclose(logits_s, logits_b, rtol=1e-5, atol=1e-6)
    assert grads_s.keys() == grads_b.keys()
    assert all(float((grads_s[n] - grads_b[n]).abs().max()) <= 1e-5 * float(grads_b[n].abs().max()) + 1e-12 for n in grads_b)
    del logits_b, logits_s
    assert live() == before
    step(True, None)  # without clearing, the step stays alive (the cycle the dicts break)
    assert live() > before


def test_trainer_clears_the_steps_kv_dicts_after_the_backward_pass(monkeypatch):
    from collections import UserDict

    import trl

    from learning_loop.training.dpo import _make_trainer_class

    cls = _make_trainer_class(split_pair_backward=True)
    tr = object.__new__(cls)  # training_step only: no model or data
    d = UserDict(k=1)
    tr._kv_dicts = [d]

    def step(self, *a, **kw):
        assert d  # still populated during the step (recomputation reads it in the backward pass)
        raise RuntimeError("step failed")

    monkeypatch.setattr(trl.DPOTrainer, "training_step", step)
    with pytest.raises(RuntimeError):
        tr.training_step(None, {})
    assert not d and tr._kv_dicts == []  # cleared even when the step raised
