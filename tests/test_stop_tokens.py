"""Unit tests for litereason.stop_tokens (CPU-only, no model downloads).

These lock the behavior the SFT and RL stop-id backfill paths share, so the
shared helpers can't silently drift.
"""
from litereason.stop_tokens import assistant_suffix_stop_token_ids, normalize_token_ids


class TestNormalizeTokenIds:
    def test_flatten_dedup_and_order(self):
        # Nested groups + duplicates across groups; first-appearance order kept.
        assert normalize_token_ids([3, 1], 1, [[2, 3]], 4) == [3, 1, 2, 4]

    def test_skips_none_and_non_numeric(self):
        assert normalize_token_ids(None, [None, "x"], "7", 8) == [7, 8]

    def test_empty_inputs(self):
        assert normalize_token_ids() == []
        assert normalize_token_ids(None, [], set()) == []

    def test_coerces_numeric_strings_and_floats(self):
        assert normalize_token_ids("5", 6.0) == [5, 6]


class _StubTokenizer:
    """Minimal tokenizer whose chat template appends <|im_end|> (special id 100)."""

    eos_token_id = 0
    unk_token_id = 1
    all_special_ids = (0, 1, 100)

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return f"<user>u</user>{messages[-1]['content']}<|im_end|>"

    def __call__(self, text, add_special_tokens=False):
        # Only the isolated suffix is passed in.
        return {"input_ids": [100] if "<|im_end|>" in text else []}

    def convert_ids_to_tokens(self, token_id):
        return {0: "</s>", 1: "<unk>", 100: "<|im_end|>"}.get(token_id)


class TestAssistantSuffixStopTokenIds:
    def test_extracts_control_token_after_assistant_turn(self):
        assert assistant_suffix_stop_token_ids(_StubTokenizer()) == [100]

    def test_missing_apply_chat_template(self):
        assert assistant_suffix_stop_token_ids(object()) == []

    def test_template_error_degrades_to_empty(self):
        class Raises(_StubTokenizer):
            def apply_chat_template(self, messages, tokenize=False):
                raise ValueError("no chat template set")

        assert assistant_suffix_stop_token_ids(Raises()) == []

    def test_plain_eos_suffix_is_excluded(self):
        class EosOnly(_StubTokenizer):
            def apply_chat_template(self, messages, tokenize=False):
                return f"{messages[-1]['content']}</s>"

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [0] if "</s>" in text else []}

        assert assistant_suffix_stop_token_ids(EosOnly()) == []
