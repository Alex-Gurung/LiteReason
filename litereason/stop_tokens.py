"""Infer EOS / stop-token ids from a tokenizer and its chat template.

Both the SFT save path (``training/train_sft.py``) and the RL rollout backfill
(``patches/openrlhf/train_ppo_ray.py``) need to reconstruct the full set of stop
ids a model should generate against: the tokenizer EOS, any ids already in the
checkpoint's ``generation_config``, and the control tokens a chat template
appends after an assistant turn (e.g. Qwen's ``<|im_end|>``, Gemma's
``<end_of_turn>``). Centralizing the logic here keeps the two call sites from
drifting apart.
"""

from typing import Iterator


def _iter_token_ids(values) -> Iterator[int]:
    """Recursively flatten ``values`` into ints, skipping anything non-coercible."""
    if values is None:
        return
    if isinstance(values, (list, tuple, set)):
        for value in values:
            yield from _iter_token_ids(value)
        return
    try:
        yield int(values)
    except (TypeError, ValueError):
        # Leaf isn't coercible to an int (None slipped through, or a non-numeric
        # string): skip it rather than poison the merged id list.
        return


def normalize_token_ids(*groups) -> list[int]:
    """Merge one or more (possibly nested) id groups into a deduped, ordered list.

    Accepts scalars, lists/tuples/sets, and arbitrary nesting thereof; ``None``
    and non-numeric leaves are dropped. Order of first appearance is preserved.
    """
    merged: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for token_id in _iter_token_ids(group):
            if token_id not in seen:
                merged.append(token_id)
                seen.add(token_id)
    return merged


def assistant_suffix_stop_token_ids(tokenizer) -> list[int]:
    """Infer the control-token ids a chat template appends after an assistant turn.

    Renders a single assistant message through the tokenizer's chat template,
    isolates whatever the template emitted *after* the assistant content, and
    returns the control-token ids found there (special ids, or tokens that look
    like ``<...>`` markers), excluding plain EOS/UNK.

    Best-effort: returns ``[]`` if the tokenizer has no usable chat template or
    the suffix can't be re-tokenized; callers fall back to their existing stop
    ids rather than failing.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return []

    assistant_sentinel = "__litereason_assistant_suffix__"
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": assistant_sentinel},
            ],
            tokenize=False,
        )
    except Exception:
        # Chat-template rendering raises varied types (no template -> ValueError,
        # malformed template -> jinja errors); degrade to no suffix.
        return []

    idx = rendered.rfind(assistant_sentinel)
    if idx < 0:
        return []
    suffix = rendered[idx + len(assistant_sentinel):]
    if not suffix:
        return []

    try:
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    except Exception:
        # If the isolated suffix can't be re-tokenized, skip the backfill.
        return []

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    def _looks_like_control_token(token_id: int) -> bool:
        try:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:
            # Heuristic only: any failure mapping the id back to a token string
            # means "not a control token".
            return False
        return isinstance(token, str) and token.startswith("<") and token.endswith(">")

    return normalize_token_ids(
        [
            token_id
            for token_id in suffix_ids
            if token_id not in {eos_id, unk_id}
            and (token_id in special_ids or _looks_like_control_token(token_id))
        ]
    )
