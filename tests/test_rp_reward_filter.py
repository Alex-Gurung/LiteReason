"""Tests for the periodic projector-SFT reward filter (rejection sampling).

The paper's SFT keeps only positive-reward traces. `extract_reasoning_traces_
from_experiences(..., positive_reward_only=True)` (the default) must drop
non-positive-reward sequences before any parsing. CPU-only; no model.
"""
import torch

from litereason.patches.openrlhf.projector_sft_patch import (
    _reward_for_seq,
    extract_reasoning_traces_from_experiences,
)


def test_reward_for_seq_handles_tensor_list_and_missing():
    assert _reward_for_seq(torch.tensor([0.0, 1.0, -2.0]), 1) == 1.0
    assert _reward_for_seq([0.5, 0.0], 0) == 0.5
    assert _reward_for_seq(None, 0) is None
    assert _reward_for_seq(torch.tensor([1.0]), 5) is None  # index out of range


class _FakeExp:
    """Minimal PPO Experience stand-in: only fields the reward filter reads."""

    def __init__(self, rewards):
        n = len(rewards)
        self.sequences = torch.zeros((n, 4), dtype=torch.long)
        self.info = {"reward": torch.tensor(rewards)}
        self.prompts = ["p"] * n
        self.labels = ["a"] * n


def test_reward_filter_drops_nonpositive_before_parsing():
    # All non-positive -> dropped by the reward filter before any tokenizer use,
    # so tokenizer=None is never touched.
    stats = {}
    traces = extract_reasoning_traces_from_experiences(
        [_FakeExp([0.0, -1.0, 0.0])], tokenizer=None, stats=stats, positive_reward_only=True
    )
    assert traces == []
    assert stats.get("dropped_nonpositive_reward") == 3
    assert "reward_unavailable" not in stats


def test_missing_reward_is_kept_not_dropped():
    # No info dict -> reward unavailable -> counted, not silently dropped.
    exp = _FakeExp([1.0])
    exp.info = None
    stats = {}
    # Filter on, but reward unavailable: the sequence is NOT dropped by the filter
    # (it proceeds to parsing). We only assert the filter accounted for it.
    try:
        extract_reasoning_traces_from_experiences(
            [exp], tokenizer=None, stats=stats, positive_reward_only=True
        )
    except Exception:
        pass  # parsing may fail with tokenizer=None; we only care about the filter accounting
    assert stats.get("reward_unavailable") == 1
    assert "dropped_nonpositive_reward" not in stats
