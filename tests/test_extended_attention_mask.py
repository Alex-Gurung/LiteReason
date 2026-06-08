"""Regression test for the RL-time extended attention mask.

`_build_extended_attention_mask` inserts one attention position per latent step
so the mask lines up with the embeddings `shadow_forward` produces. Both sides
must clamp the per-marker step count to `max_reasoning_steps`; if the mask used
the raw (generated) complexity while the embeddings used the clamped count, a
marker like `<implicit_thought>9</implicit_thought>` with max_steps=5 would make
the mask longer than the embeddings and overflow the per-sample assignment in
the patched forward. CPU-only; no model or OpenRLHF needed.
"""
import torch

from litereason.patches.openrlhf.actor_patch import _build_extended_attention_mask


# marker tuple: (start_idx, gap_len, complexity, start_len, end_len)
# end_idx = start_idx + start_len + gap_len + end_len = 0 + 1 + 1 + 1 = 3
def _marker(complexity):
    return (0, 1, complexity, 1, 1)


def test_mask_length_matches_complexity_when_under_cap():
    am = torch.ones(5, dtype=torch.long)
    out = _build_extended_attention_mask(am, [_marker(3)], max_steps=5)
    # head(3) + 3 inserted + tail(2)
    assert int(out.numel()) == 5 + 3


def test_mask_clamps_complexity_above_cap():
    am = torch.ones(5, dtype=torch.long)
    out = _build_extended_attention_mask(am, [_marker(9)], max_steps=5)
    # clamped to 5 inserted steps, NOT 9: head(3) + 5 + tail(2) == 10
    assert int(out.numel()) == 5 + 5
    assert int(out.numel()) != 5 + 9


def test_mask_floors_complexity_at_one():
    am = torch.ones(4, dtype=torch.long)
    out = _build_extended_attention_mask(am, [_marker(0)], max_steps=5)
    # complexity 0 floors to 1 inserted step (matches shadow_forward's clamp)
    assert int(out.numel()) == 4 + 1
