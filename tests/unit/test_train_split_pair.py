"""dpo.split_pair_backward: running each sequence of a pair under its own checkpoint changes only
when activations are alive, never the logits or the gradients."""

import pytest

torch = pytest.importorskip("torch")

from learning_loop.training.dpo import pair_forward  # noqa: E402


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
    mask[1, :2] = 0  # left padding on one row
    kw = {"attention_mask": mask, "use_cache": False}
    weights = torch.randn(2, k, 128, dtype=torch.float64)

    def grads(split):
        model.zero_grad()
        out = pair_forward(model, kw, ids, k, split)
        (out.logits * weights).sum().backward()
        return out.logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}

    logits_b, grads_b = grads(False)
    logits_s, grads_s = grads(True)
    assert logits_s.shape == (2, k, 128)
    assert torch.allclose(logits_s, logits_b, atol=1e-12)
    assert grads_s.keys() == grads_b.keys()
    assert all(torch.allclose(grads_s[n], grads_b[n], atol=1e-10) for n in grads_b)
