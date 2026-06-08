"""Token pattern detection utilities for implicit thought markers.

We represent the marker:
    <implicit_thought>N</implicit_thought>

as a token-level pattern:
    <start_tokens> <gap_tokens> <end_tokens>

`start_tokens` and `end_tokens` depend on tokenizer behavior. In particular, some
tokenizers merge preceding or trailing characters into the same token as "<" / ">".
To stay robust, we auto-detect multiple valid start/end tokenizations by encoding the
markers in different string contexts.
"""


import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


def get_implicit_thought_patterns(
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """Generate all token patterns for <implicit_thought> markers.

    Auto-detects patterns by encoding the markers and extracting just the
    marker tokens (without context prefixes/suffixes that might merge).

    Args:
        tokenizer: The tokenizer to use for encoding.

    Returns:
        Tuple of (start_patterns, end_patterns) where each is a list of
        token ID tuples that encode the start/end markers.
    """
    start_patterns: Set[Tuple[int, ...]] = set()
    end_patterns: Set[Tuple[int, ...]] = set()

    start_text = "<implicit_thought>"
    end_text = "</implicit_thought>"

    def _enc(s: str) -> Tuple[int, ...]:
        return tuple(tokenizer.encode(s, add_special_tokens=False))

    def _dec(ids: Iterable[int]) -> str:
        return tokenizer.decode(list(ids))

    base_start_ids = _enc(start_text)
    base_end_ids = _enc(end_text)

    if start_text in _dec(base_start_ids):
        start_patterns.add(base_start_ids)
    if end_text in _dec(base_end_ids):
        end_patterns.add(base_end_ids)

    # Some tokenizers merge adjacent characters with angle brackets into a single
    # token (e.g. ".<" becomes one token). We probe multiple string contexts to
    # discover all valid tokenizations of the marker tags.
    test_prefixes = [" ", "\n", ".", ",", "!", "?", ":", ";", '"', "'", ")", "(", "[", "]", ">"]
    for prefix in test_prefixes:
        prefix_ids = _enc(prefix)
        combined_ids = _enc(f"{prefix}{start_text}")

        # If prefix tokens are separate, the marker tokenization is the suffix which
        # should match `base_start_ids`, so there's nothing new to add.
        if len(base_start_ids) > 0 and combined_ids[-len(base_start_ids) :] == base_start_ids:
            continue

        # Otherwise the prefix likely merged with the "<...>" tokens. Keep the minimal
        # suffix that still contains the marker text.
        for i in range(len(combined_ids)):
            candidate = combined_ids[i:]
            if start_text in _dec(candidate):
                # Avoid patterns that are just the prefix (paranoia, but cheap).
                if candidate != prefix_ids:
                    start_patterns.add(candidate)
                break

    # Suffixes: try to catch merged ">" tokens (e.g. ">.", ">\n", "></", ">\"", etc.).
    test_suffixes = [" ", " </", "\n", "\n\n", "\n\n\n", ".", ",", "!", "?", "</", '"']
    for suffix in test_suffixes:
        suffix_ids = _enc(suffix)
        combined_ids = _enc(f"{end_text}{suffix}")

        # If suffix tokens are separate, the marker tokenization is the prefix which
        # should match `base_end_ids`, so there's nothing new to add.
        if len(base_end_ids) > 0 and len(combined_ids) > len(base_end_ids) and combined_ids[: len(base_end_ids)] == base_end_ids:
            continue

        # Otherwise the suffix likely merged with the "</...>" tokens. Keep the minimal
        # prefix that still contains the marker text.
        for i in range(len(combined_ids), 0, -1):
            candidate = combined_ids[:i]
            if end_text in _dec(candidate):
                if candidate != suffix_ids:
                    end_patterns.add(candidate)
                break

    validated_start = [p for p in start_patterns if start_text in _dec(p)]
    validated_end = [p for p in end_patterns if end_text in _dec(p)]

    # Deterministic ordering for reproducibility and tests.
    validated_start.sort(key=lambda p: (len(p), p))
    validated_end.sort(key=lambda p: (len(p), p))
    return validated_start, validated_end


def _match_special_sequence(
    input_ids: torch.Tensor,
    special_start_sequences: List[Tuple[int, ...]],
    special_end_sequences: List[Tuple[int, ...]],
    max_gap_tokens: int = 3,
) -> Optional[Tuple[torch.Tensor, int, int]]:
    """Match a marker sequence at the start of `input_ids`.

    Returns:
        (gap_tokens, start_len, end_len) if match found, else None.
    """
    if input_ids.dim() != 1:
        input_ids = input_ids.squeeze(0)

    # Try each start pattern at position 0, then for each valid gap length
    # (0 to max_gap_tokens), check if any end pattern follows immediately.
    # The gap tokens contain the complexity number digits.
    window = input_ids.tolist()
    for start in special_start_sequences:
        ls = len(start)
        if ls == 0 or ls > len(window):
            continue
        if tuple(window[:ls]) != start:
            continue
        for gap in range(0, max_gap_tokens + 1):
            j = ls + gap
            if j > len(window):
                break
            for end in special_end_sequences:
                le = len(end)
                if le == 0 or j + le > len(window):
                    continue
                if tuple(window[j : j + le]) == end:
                    return input_ids[ls:j], ls, le
    return None


class ImplicitThoughtDetector:
    """Detector for <implicit_thought>N</implicit_thought> patterns.

    Caches tokenizer patterns. `find_thought_markers` scans the sequence with an
    O(seq_len) python loop, but takes a fast early-out via `torch.isin`: if the
    sequence contains no start token at all, it returns immediately without
    looping.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_gap_tokens: int = 3):
        self.tokenizer = tokenizer

        start_patterns, end_patterns = get_implicit_thought_patterns(tokenizer)
        if not start_patterns or not end_patterns:
            raise ValueError(
                "Could not detect <implicit_thought> token patterns for this tokenizer. "
                "If you added the tags as special tokens, ensure the tokenizer can encode/decode "
                "them without special tokens."
            )

        self.start_patterns = start_patterns
        self.end_patterns = end_patterns
        log.info("ImplicitThoughtDetector: detected %d start and %d end token patterns", len(start_patterns), len(end_patterns))

        self.start_seq_len = max(len(p) for p in start_patterns)
        self.end_seq_len = max(len(p) for p in end_patterns)
        self._start_first_tokens = {p[0] for p in start_patterns if len(p) > 0}
        self._max_gap_tokens = int(max_gap_tokens)
        self._scan_ahead = self.start_seq_len + self._max_gap_tokens + self.end_seq_len
        if self._scan_ahead > 25:
            raise ValueError(
                "Implicit thought marker tokenization is unexpectedly long "
                f"(start<= {self.start_seq_len} + gap<= {self._max_gap_tokens} + end<= {self.end_seq_len} "
                f"= {self._scan_ahead} tokens). Increase the scan window if you really need this tokenizer."
            )

    def find_thought_markers(
        self,
        input_ids: torch.Tensor,
    ) -> List[Tuple[int, int, int, int, int]]:
        """Find all implicit thought markers in input sequence.

        Args:
            input_ids: Token IDs tensor of shape [batch_size, seq_len] or [seq_len].

        Returns:
            List of tuples:
            `(start_idx, num_complexity_tokens, complexity, start_seq_len, end_seq_len)`.
        """
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("Currently only batch size 1 is supported")
            input_seq = input_ids[0]
        else:
            input_seq = input_ids

        if not self._start_first_tokens:
            return []

        start_tokens = torch.tensor(sorted(self._start_first_tokens), device=input_seq.device)
        has_any_start_token = torch.isin(input_seq, start_tokens).any()
        if not has_any_start_token:
            return []
        results: List[Tuple[int, int, int, int, int]] = []
        input_len = int(input_seq.shape[0])

        # Advance past a matched marker's span so overlapping/nested markers are
        # not emitted (a later marker whose start_idx < an earlier marker's
        # end_idx). Adjacent markers (next starts exactly at the previous end)
        # are still detected because next_scan_start equals the previous end_idx.
        i = 0
        while i < input_len:
            if int(input_seq[i].item()) not in self._start_first_tokens:
                i += 1
                continue

            window_end = min(i + self._scan_ahead, input_len)
            token_window = input_seq[i:window_end]

            matched = _match_special_sequence(
                token_window,
                self.start_patterns,
                self.end_patterns,
                max_gap_tokens=self._max_gap_tokens,
            )
            if matched is None:
                i += 1
                continue
            gap_tokens, start_len, end_len = matched

            complexity_str = self.tokenizer.decode(gap_tokens.tolist()).strip()
            if not complexity_str or not complexity_str.isdigit():
                i += 1
                continue
            complexity = int(complexity_str)
            if complexity <= 0:
                i += 1
                continue

            results.append((i, int(gap_tokens.numel()), complexity, start_len, end_len))
            # Skip past the matched marker's end so we don't re-match overlapping
            # markers inside its span.
            i += start_len + int(gap_tokens.numel()) + end_len

        return results

    def find_markers_or_none(
        self,
        input_ids: torch.Tensor,
    ) -> Optional[List[Tuple[int, int, int, int, int]]]:
        """Fast-path marker find: returns markers list or None.

        Combines fast token-level checks with batch>1 validation into a single
        call, replacing the 4-stage cascade previously duplicated in
        ``causal_lm_with_reasoning.py`` and ``actor_patch.py``.

        Args:
            input_ids: Token IDs ``[batch_size, seq_len]`` or ``[seq_len]``.

        Returns:
            List of marker tuples if markers found (batch_size must be 1),
            or None if no markers are present.

        Raises:
            ValueError: If batch_size > 1 and actual markers are detected.
        """
        start_first = sorted(self._start_first_tokens)
        if not start_first:
            return None

        start_tokens = torch.tensor(start_first, device=input_ids.device)
        if not torch.isin(input_ids, start_tokens).any():
            return None

        # Marker expansion only works with batch_size=1. If batch>1 and markers
        # exist, raise so the caller uses per-sample shadow_forward instead.
        if input_ids.dim() == 2 and input_ids.shape[0] != 1:
            for i in range(input_ids.shape[0]):
                if torch.isin(input_ids[i], start_tokens).any():
                    if self.find_thought_markers(input_ids[i : i + 1]):
                        raise ValueError(
                            "Marker expansion supports batch_size=1. "
                            "Use shadow_forward per sample and then do a single batched "
                            "forward with inputs_embeds + given_is_reasoning_embedding_mask."
                        )
            return None

        markers = self.find_thought_markers(input_ids)
        return markers if markers else None
