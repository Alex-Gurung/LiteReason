"""Marker-aware collator masking tests (CPU-only; stubbed detector + tokenizer).

Pins the A4 decision: SFT supervision starts after the FIRST implicit-thought
marker (``mask_before="first"``, the default). With multiple latent spans, the
tokens after the first span (including later spans) are therefore supervised.
The reference implementation uses ``"last"``.
"""
import pytest
import torch

from litereason.training.collate import MarkerAwareCollator


class _StubDetector:
    """Returns a fixed marker list, bypassing tokenizer-dependent detection."""

    def __init__(self, markers):
        self._markers = markers

    def find_thought_markers(self, input_ids_2d):
        return list(self._markers)


class _StubTok:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return "<preview>"


# marker tuple = (start_idx, num_gap_tokens, complexity, start_len, end_len)
# end position = start_idx + start_len + num_gap_tokens + end_len
_MARKER_A = (1, 1, 2, 1, 0)  # ends at index 3
_MARKER_B = (5, 1, 3, 1, 0)  # ends at index 7


def _labels_for(mask_before, markers):
    col = MarkerAwareCollator(
        _StubTok(), thought_detector=_StubDetector(markers), mask_before=mask_before
    )
    ids = torch.arange(1, 11)  # 10 tokens, none equal pad_token_id (0)
    return col([{"input_ids": ids}])["labels"][0]


def test_mask_before_first_supervises_after_first_marker():
    labels = _labels_for("first", [_MARKER_A, _MARKER_B])
    assert (labels[:3] == -100).all()       # masked up to the first marker end (idx 3)
    assert (labels[3:] != -100).all()       # everything after IS supervised (incl. 2nd span)


def test_mask_before_last_supervises_only_after_last_marker():
    labels = _labels_for("last", [_MARKER_A, _MARKER_B])
    assert (labels[:7] == -100).all()       # masked up to the last marker end (idx 7)
    assert (labels[7:] != -100).all()


def test_require_marker_raises_when_absent():
    col = MarkerAwareCollator(
        _StubTok(), thought_detector=_StubDetector([]), require_marker=True
    )
    with pytest.raises(ValueError, match="Missing <implicit_thought> marker"):
        col([{"input_ids": torch.arange(1, 6)}])


if __name__ == "__main__":
    test_mask_before_first_supervises_after_first_marker()
    test_mask_before_last_supervises_only_after_last_marker()
    test_require_marker_raises_when_absent()
    print("OK: collate masking tests passed")
