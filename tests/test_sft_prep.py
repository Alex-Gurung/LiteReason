"""SFT-data prep tests (CPU-only, no model download).

Pins two paper-fidelity fixes:
- A2: latent-step count is based on the number of *tokens* in the replaced
  sentence (paper), not word count.
- A3: markers are placed on the correct sentence by index, even when a sentence
  repeats or is a substring of another (a plain str.replace would mis-target).
"""
from unittest import mock

from litereason.pipeline.prepare_sft import (
    calculate_reasoning_steps,
    mask_sentences,
)


class CharTok:
    """Stub tokenizer: one token per non-space character.

    Deliberately differs from a word count so a test can prove the token-based
    path is actually used (10 chars in one word -> 2 steps, not 1).
    """

    def encode(self, text, add_special_tokens=False):
        return [c for c in text if not c.isspace()]


def test_latent_count_is_token_based():
    tok = CharTok()
    # 10 tokens * 0.2 = 2 steps. A word count would give floor(1*0.2)=1 -> proves tokens used.
    assert calculate_reasoning_steps("abcdefghij", tok, token_ratio=0.2) == 2
    assert calculate_reasoning_steps("a" * 100, tok, token_ratio=0.2) == 5   # capped at 5
    assert calculate_reasoning_steps("ab", tok, token_ratio=0.2) == 1        # at least 1


def test_marker_targets_correct_sentence_with_substring_dupes():
    # sentence[1] ("ok") is a substring of sentence[0] ("ok ok"). The buggy
    # str.replace(s, ..., 1) would corrupt sentence[0]; index-based rebuild must not.
    sentences = ["ok ok", "ok"]
    with mock.patch("random.sample", return_value=[1]):
        out = mask_sentences("ok ok\nok", sentences, mask_ratio=0.5, tokenizer=CharTok())
    assert "ok ok" in out                          # sentence 0 left intact
    assert out.count("<implicit_thought>") == 1     # exactly one marker, on sentence 1


def test_no_masking_returns_text_unchanged():
    sentences = ["a sentence.", "another one."]
    out = mask_sentences(
        "a sentence.\nanother one.", sentences, mask_ratio=0.0, tokenizer=CharTok()
    )
    assert out == "a sentence.\nanother one."


if __name__ == "__main__":
    test_latent_count_is_token_based()
    test_marker_targets_correct_sentence_with_substring_dupes()
    test_no_masking_returns_text_unchanged()
    print("OK: sft-prep tests passed")
