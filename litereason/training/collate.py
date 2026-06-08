"""Marker-aware data collation for LiteReason SFT training.

Labels are masked up to the first marker boundary by default. The reference
implementation masks up to the last marker. For single-marker data (the common
case) these are identical. Use ``mask_before="last"`` to match reference behavior.
"""


from typing import Any, Dict, List, Literal, Optional

import torch
from transformers import PreTrainedTokenizer

from litereason.token_utils import ImplicitThoughtDetector


class MarkerAwareCollator:
    """Collator that masks labels before the first implicit thought marker.

    During SFT, we only want to compute loss on tokens after the first
    <implicit_thought>N</implicit_thought> marker, since that's where
    the model's reasoning begins.

    Args:
        tokenizer: The tokenizer to use for padding.
        thought_detector: Pre-built detector (shared with model). If None, one
            is created from the tokenizer.
        mask_before: Which marker boundary to mask up to. ``"first"`` (default)
            masks before the first marker end; ``"last"`` masks before the last.
        require_marker: If True (default), raise an error when a sample has no
            implicit-thought marker.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        thought_detector: Optional[ImplicitThoughtDetector] = None,
        mask_before: Literal["first", "last"] = "first",
        require_marker: bool = True,
    ):
        self.tokenizer = tokenizer
        # Explicit None check: a legitimate pad_token_id of 0 is falsy and must
        # NOT be replaced by eos_token_id.
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        self._detector = thought_detector or ImplicitThoughtDetector(tokenizer)
        self._mask_before = mask_before
        self._require_marker = require_marker

    def _find_marker_mask_boundary(self, input_ids: torch.Tensor) -> int:
        """Find the position after the selected marker end sequence.

        Returns the index of the first token AFTER the marker ends,
        or 0 if no marker is found.
        """
        markers = self._detector.find_thought_markers(input_ids.unsqueeze(0))
        if not markers:
            return 0

        def _marker_end(m):
            start_idx, num_gap_tokens, _complexity, start_len, end_len = m
            return start_idx + start_len + num_gap_tokens + end_len

        if self._mask_before == "last":
            return _marker_end(markers[-1])
        else:
            return _marker_end(markers[0])

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch with masked labels.

        Expects examples to have 'input_ids' key with pre-tokenized sequences.
        """
        # Extract input_ids
        input_ids_list = []
        for example in examples:
            if "input_ids" in example:
                ids = example["input_ids"]
                if isinstance(ids, list):
                    ids = torch.tensor(ids, dtype=torch.long)
                input_ids_list.append(ids)
            else:
                raise ValueError("Examples must have 'input_ids' key")

        # Pad sequences
        if len(input_ids_list) == 1:
            # No padding needed for batch size 1
            input_ids = input_ids_list[0].unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
        else:
            # Pad to longest sequence
            max_len = max(ids.shape[0] for ids in input_ids_list)
            padded = []
            masks = []
            for ids in input_ids_list:
                pad_len = max_len - ids.shape[0]
                if pad_len > 0:
                    padded.append(
                        torch.cat(
                            [ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
                        )
                    )
                    masks.append(
                        torch.cat(
                            [torch.ones_like(ids), torch.zeros(pad_len, dtype=torch.long)]
                        )
                    )
                else:
                    padded.append(ids)
                    masks.append(torch.ones_like(ids))
            input_ids = torch.stack(padded)
            attention_mask = torch.stack(masks)

        labels = input_ids.clone()
        # Mask only PADDING positions, not real EOS. When pad_token_id == eos
        # (train_sft sets pad = eos when there's no pad), masking by token id
        # would also hide the legitimate final EOS so the model never learns to
        # stop. The attention_mask is 0 exactly on padded positions.
        labels[attention_mask == 0] = -100

        for batch_idx in range(labels.shape[0]):
            marker_end_pos = self._find_marker_mask_boundary(input_ids[batch_idx])
            if marker_end_pos > 0:
                labels[batch_idx, :marker_end_pos] = -100
            elif self._require_marker:
                preview_ids = input_ids[batch_idx, : min(256, input_ids.shape[1])].tolist()
                preview_text = self.tokenizer.decode(preview_ids, skip_special_tokens=False)
                preview_text = preview_text.replace("\n", "\\n")
                raise ValueError(
                    "Missing <implicit_thought> marker in SFT sample while require_marker=True. "
                    "This usually means dataset prep produced unmasked samples or prompt/data mismatch. "
                    f"sample_preview={preview_text[:320]!r}"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
