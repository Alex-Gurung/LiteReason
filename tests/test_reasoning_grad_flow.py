"""Gradient-flow tests for LiteReason latent reasoning (tiny, CPU-only).

These lock in the paper's claim that projector SFT backpropagates through the
*rollout* of latent reasoning steps. With the base model frozen (the SFT setup),
the gradient from a later latent step must still flow back through the base model
to earlier latent steps (backprop-through-time). A historical port bug wrapped
the iterative base forwards in ``torch.no_grad()`` whenever the base was frozen,
which silently detached the steps from each other and trained only the final
projector application per marker.

No GPU and no model download required: a randomly-initialised tiny Qwen2 is built
from config on CPU.
"""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from litereason.causal_lm_with_reasoning import (
    _make_reasoning_forward_fn,
    attach_litereason_to_causal_lm,
)


def _tiny_frozen_litereason_model():
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    attach_litereason_to_causal_lm(model, num_reasoning_layers=2, max_reasoning_steps=5)
    # Paper SFT setup: freeze the whole base model, train only the projector.
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("reasoning_projector."))
    return model


def _run_rollout(model, complexity):
    reasoning_fn = _make_reasoning_forward_fn(model)
    hidden = model.config.hidden_size
    # Stand-in for token embeddings from a frozen embedding layer: no grad on input.
    section = torch.randn(1, 3, hidden, requires_grad=False)
    with torch.enable_grad():
        embeds, _ = reasoning_fn(
            all_embeddings=[section],
            inputs_embeds=section,
            past_key_values=None,
            use_cache=False,
            complexity=complexity,
        )
    return embeds


def test_bptt_through_latent_rollout():
    """A later latent step must depend on earlier ones (no detach between steps)."""
    model = _tiny_frozen_litereason_model()
    embeds = _run_rollout(model, complexity=2)

    assert len(embeds) == 2
    assert embeds[1].requires_grad, "latent step should carry projector gradient"

    # Decisive check: d(step2)/d(step1) must exist. Under the port bug the base
    # forward between steps ran in torch.no_grad(), so step 2 was detached from
    # step 1 and this gradient would be None.
    grad = torch.autograd.grad(
        embeds[1].sum(), embeds[0], retain_graph=True, allow_unused=True
    )[0]
    assert grad is not None, (
        "latent step 2 is detached from step 1: no backprop through the rollout "
        "(base forward is wrapped in torch.no_grad() when the base is frozen)"
    )
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_projector_receives_finite_gradient():
    """Smoke test: supervising the kept (last) latent embedding produces finite,
    non-trivial gradients on the projector under a frozen base."""
    model = _tiny_frozen_litereason_model()
    embeds = _run_rollout(model, complexity=3)
    loss = embeds[-1].pow(2).sum()
    loss.backward()

    proj_params = list(model.reasoning_projector.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in proj_params
    )
    assert any(p.grad.abs().sum() > 0 for p in proj_params)


def test_no_grad_context_freezes_projector():
    """In the RL rollout, shadow_forward runs under torch.no_grad, so the projector
    produces no-grad embeddings and PPO cannot update it (only the periodic SFT does)."""
    model = _tiny_frozen_litereason_model()
    reasoning_fn = _make_reasoning_forward_fn(model)
    hidden = model.config.hidden_size
    section = torch.randn(1, 3, hidden)
    with torch.no_grad():
        embeds, _ = reasoning_fn(
            all_embeddings=[section], inputs_embeds=section,
            past_key_values=None, use_cache=False, complexity=2,
        )
    assert not embeds[-1].requires_grad


def _active_projector_model():
    """Frozen-base model whose projector is perturbed off the near-zero warm-start,
    so latent outputs and gradients are at realistic (non-degenerate) magnitudes.
    float64 for a clean numeric cache-vs-no-cache comparison."""
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).double()
    attach_litereason_to_causal_lm(model, num_reasoning_layers=2, max_reasoning_steps=5)
    torch.manual_seed(123)
    for name, p in model.named_parameters():
        if name.startswith("reasoning_projector."):
            p.data = p.data + 0.2 * torch.randn_like(p)
        p.requires_grad_(name.startswith("reasoning_projector."))
    return model


def _projector_grad(use_cache, complexity=3):
    model = _active_projector_model()
    torch.manual_seed(1)
    section = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float64)
    reasoning_fn = _make_reasoning_forward_fn(model)
    with torch.enable_grad():
        embeds, _ = reasoning_fn(
            all_embeddings=([section] if not use_cache else None),
            inputs_embeds=section, past_key_values=None,
            use_cache=use_cache, complexity=complexity,
        )
    torch.manual_seed(7)
    weights = [torch.randn_like(e) for e in embeds]  # loss depends on every step
    loss = sum((e * w).sum() for e, w in zip(embeds, weights, strict=True))
    model.zero_grad()
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.reasoning_projector.parameters()])
    return torch.stack([e.detach().flatten() for e in embeds]), grad


def test_cache_mode_bptt_matches_no_cache():
    """The periodic RP-SFT during RL runs the rollout in cache mode (use_cache=True,
    the Path-C default). Its projector gradient must equal the no-cache full
    recompute, i.e. the KV cache stays graph-connected (no detach between steps)."""
    e_nc, g_nc = _projector_grad(use_cache=False)
    e_c, g_c = _projector_grad(use_cache=True)
    assert torch.allclose(e_nc, e_c, atol=1e-10)  # forward identical
    assert g_nc.norm() > 1.0  # gradient is non-trivial, so the match means something
    assert (g_c - g_nc).norm() / g_nc.norm() < 1e-6  # equal up to fp reassociation


def test_cache_mode_backprops_across_latent_steps():
    """In cache mode a later latent step must still depend on an earlier one through
    the cached KV, not just the direct input embedding."""
    model = _active_projector_model()
    torch.manual_seed(1)
    section = torch.randn(1, 3, model.config.hidden_size, dtype=torch.float64)
    reasoning_fn = _make_reasoning_forward_fn(model)
    with torch.enable_grad():
        embeds, _ = reasoning_fn(
            all_embeddings=None, inputs_embeds=section, past_key_values=None,
            use_cache=True, complexity=3,
        )
    grad = torch.autograd.grad(
        embeds[2].sum(), embeds[0], retain_graph=True, allow_unused=True
    )[0]
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


if __name__ == "__main__":
    test_bptt_through_latent_rollout()
    test_projector_receives_finite_gradient()
    test_no_grad_context_freezes_projector()
    test_cache_mode_bptt_matches_no_cache()
    test_cache_mode_backprops_across_latent_steps()
    print("OK: latent-rollout gradient-flow tests passed")
