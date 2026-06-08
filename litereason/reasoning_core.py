"""Shared reasoning logic for marker expansion, reasoning forward, and output masking.

The generic HF path (``causal_lm_with_reasoning.py``) delegates to these
functions so the core algorithms live in one place.
"""


import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

log = logging.getLogger(__name__)

# Rate-limit flag so a divergent projector rollout warns once rather than on
# every latent step of every forward.
_WARNED_NONFINITE = False


def _warn_nonfinite_once(where: str) -> None:
    global _WARNED_NONFINITE
    if not _WARNED_NONFINITE:
        log.warning(
            "LiteReason: non-finite values in %s at a latent step; clamping to 0",
            where,
        )
        _WARNED_NONFINITE = True


# --- Projector stack ---

def apply_projector_stack(projector: nn.ModuleList, hidden: torch.Tensor) -> torch.Tensor:
    """Run hidden state through the reasoning projector stack.

    Args:
        projector: ``nn.ModuleList`` of MLP-like layers.
        hidden: Hidden state tensor (any shape accepted by the layers).

    Returns:
        Projected tensor, same shape as input.
    """
    out = hidden
    for layer in projector:
        out = layer(out)
    return out


# --- Output masking ---

def apply_reasoning_mask(
    outputs: Union[BaseModelOutputWithPast, CausalLMOutputWithPast],
    mask: torch.BoolTensor,
) -> Union[BaseModelOutputWithPast, CausalLMOutputWithPast]:
    """Remove positions where ``mask=True`` from model outputs.

    Works for both ``BaseModelOutputWithPast`` (masks ``last_hidden_state``)
    and ``CausalLMOutputWithPast`` (masks ``logits``).

    Args:
        outputs: Model output dataclass.
        mask: Boolean mask (True = remove). Shape ``[T]`` or ``[B, T]``.

    Returns:
        The *same* outputs object, mutated in place with masked positions removed.
    """
    if mask.sum().item() <= 0:
        return outputs

    # Determine which primary tensor to mask
    if hasattr(outputs, "logits") and outputs.logits is not None:
        primary_field = "logits"
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        primary_field = "last_hidden_state"
    else:
        return outputs

    primary = getattr(outputs, primary_field)

    if mask.dim() == 1:
        _apply_1d_mask(outputs, mask, primary_field, primary)
    elif mask.dim() == 2:
        _apply_2d_mask(outputs, mask, primary_field, primary)
    else:
        raise ValueError(f"Expected mask dim 1 or 2, got {mask.dim()}")

    return outputs


def _apply_1d_mask(
    outputs: Union[BaseModelOutputWithPast, CausalLMOutputWithPast],
    mask: torch.BoolTensor,
    primary_field: str,
    primary: torch.Tensor,
) -> None:
    """Apply a 1-D mask (batch size 1 or single sample)."""
    keep = ~mask
    setattr(outputs, primary_field, primary[:, keep, :])

    if outputs.hidden_states is not None:
        outputs.hidden_states = tuple(hs[:, keep, :] for hs in outputs.hidden_states)

    if outputs.attentions is not None:
        outputs.attentions = tuple(
            attn[:, :, keep, :][:, :, :, keep] for attn in outputs.attentions
        )


def _apply_2d_mask(
    outputs: Union[BaseModelOutputWithPast, CausalLMOutputWithPast],
    mask: torch.BoolTensor,
    primary_field: str,
    primary: torch.Tensor,
) -> None:
    """Apply a 2-D mask (batched, all samples padded to same output length).

    All samples must have the same number of kept (non-masked) positions,
    because the output tensor must be a regular [B, T_out, D] shape.
    """
    B, T_ext = mask.shape
    keep_counts = (~mask).sum(dim=1)

    if int(keep_counts.min().item()) <= 0:
        return

    uniq = torch.unique(keep_counts)
    if uniq.numel() != 1:
        raise ValueError(
            "Inconsistent unmasked (kept) lengths across the batch. "
            "Pad so that after removing reasoning/pad positions (mask=True), every sample has "
            "the same kept length. "
            f"Got kept lengths: {keep_counts.tolist()}."
        )

    T_out = int(uniq.item())
    keep_idxs = [(~mask[b]).nonzero(as_tuple=True)[0] for b in range(B)]

    # Primary tensor
    D = primary.size(-1)
    new_primary = primary.new_empty((B, T_out, D))
    for b in range(B):
        new_primary[b] = primary[b, keep_idxs[b], :]
    setattr(outputs, primary_field, new_primary)

    # Hidden states
    if outputs.hidden_states is not None:
        new_hs = []
        for hs in outputs.hidden_states:
            H = hs.size(-1)
            nlayer = hs.new_empty((B, T_out, H))
            for b in range(B):
                nlayer[b] = hs[b, keep_idxs[b], :]
            new_hs.append(nlayer)
        outputs.hidden_states = tuple(new_hs)

    # Attentions
    if outputs.attentions is not None:
        new_atts = []
        for attn in outputs.attentions:
            heads = attn.size(1)
            nlayer = attn.new_empty((B, heads, T_out, T_out))
            for b in range(B):
                idx = keep_idxs[b]
                nlayer[b] = attn[b][:, idx, :][:, :, idx]
            new_atts.append(nlayer)
        outputs.attentions = tuple(new_atts)


# --- Iterative reasoning ---

def reasoning_forward_core(
    base_forward_fn: Callable[..., object],
    projector_fn: Callable[[torch.Tensor], torch.Tensor],
    extract_hidden_fn: Callable[[object], torch.Tensor],
    all_embeddings: Optional[List[torch.Tensor]],
    inputs_embeds: torch.Tensor,
    past_key_values: object,
    use_cache: bool,
    complexity: int,
    forward_kwargs: dict,
) -> Tuple[List[torch.Tensor], object]:
    """Iterative reasoning embedding generation.

    Runs *complexity* iterations of:
    1. Forward pass with current context (cache or full sequence)
    2. Extract last hidden state via ``extract_hidden_fn``
    3. Project through ``projector_fn`` (the reasoning projector)
    4. Append reasoning embedding to context for next iteration

    Args:
        base_forward_fn: Calls the model's base forward (Qwen2Model.forward or
            CausalLM._litereason_base_forward).
        projector_fn: ``hidden -> projected`` (the reasoning projector stack).
        extract_hidden_fn: ``outputs -> hidden_state_at_last_position``.
        all_embeddings: Accumulated embedding list (non-cache mode).
        inputs_embeds: Current section embeddings.
        past_key_values: KV cache (cache mode) or None.
        use_cache: Whether to use KV caching.
        complexity: Number of reasoning steps.
        forward_kwargs: Extra kwargs passed to base_forward_fn.

    Returns:
        ``(reasoning_embeddings, updated_past_key_values)``
    """
    if complexity <= 0:
        return [], past_key_values

    log.debug("Starting %d reasoning iterations (use_cache=%s)", complexity, use_cache)

    # For non-cache mode, materialise the embedding list into a tensor
    if not use_cache and all_embeddings is not None:
        all_embeddings_tensor = torch.cat(all_embeddings, dim=1)
    else:
        all_embeddings_tensor = None

    reasoning_embeddings: List[torch.Tensor] = []
    cur_past = past_key_values

    # Each iteration: forward pass to get context -> project hidden state through
    # the reasoning projector -> append the projected embedding as the next "token"
    # in the context for the following iteration.
    for _ in range(complexity):
        if use_cache:
            out = base_forward_fn(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=cur_past,
                use_cache=True,
                **forward_kwargs,
            )
        else:
            out = base_forward_fn(
                inputs_embeds=all_embeddings_tensor,
                use_cache=False,
                **forward_kwargs,
            )

        context_hidden = extract_hidden_fn(out)
        # Clamp NaNs/Infs for numerical stability. These appear most often during
        # early training when projector weights are randomly initialized, but the
        # clamps run on every latent step in every mode by design: they are a cheap,
        # permanent defense against rare numerical blowups in the projector rollout
        # (a single Inf would otherwise propagate through all subsequent steps).
        if not torch.isfinite(context_hidden).all():
            _warn_nonfinite_once("context_hidden")
        context_hidden = torch.nan_to_num(context_hidden, nan=0.0, posinf=0.0, neginf=0.0)

        reasoning_embedding = projector_fn(context_hidden)
        if not torch.isfinite(reasoning_embedding).all():
            _warn_nonfinite_once("reasoning_embedding")
        reasoning_embedding = torch.nan_to_num(reasoning_embedding, nan=0.0, posinf=0.0, neginf=0.0)
        reasoning_embeddings.append(reasoning_embedding)

        inputs_embeds = reasoning_embedding.unsqueeze(1)
        if use_cache:
            cur_past = out.past_key_values
        else:
            all_embeddings_tensor = torch.cat(
                [all_embeddings_tensor, reasoning_embedding.unsqueeze(1)], dim=1
            )

    # Run one more forward to fold the last reasoning embedding into the KV cache,
    # so that subsequent tokens can attend to it. No gradients needed here.
    if use_cache and reasoning_embeddings:
        with torch.no_grad():
            final_out = base_forward_fn(
                input_ids=None,
                inputs_embeds=reasoning_embeddings[-1].unsqueeze(1),
                past_key_values=cur_past,
                use_cache=True,
                **forward_kwargs,
            )
            cur_past = final_out.past_key_values

    return reasoning_embeddings, cur_past


# --- Marker expansion ---

def expand_markers_to_embeddings(
    input_ids: torch.LongTensor,
    markers: List[Tuple[int, int, int, int, int]],
    embed_fn: Callable[[torch.LongTensor], torch.Tensor],
    reasoning_forward_fn: Callable[
        ..., Tuple[List[torch.Tensor], object]
    ],
    use_cache: bool,
    past_key_values: object = None,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Expand ``<implicit_thought>`` markers into reasoning embeddings.

    Iterates over detected markers, embeds each section, generates reasoning
    embeddings, and builds the remove-mask.

    Args:
        input_ids: Token IDs ``[1, seq_len]``.
        markers: Output from ``ImplicitThoughtDetector.find_thought_markers``.
        embed_fn: ``input_ids -> embeddings`` (the token embedding layer).
        reasoning_forward_fn: Callable with signature
            ``(all_embeddings, inputs_embeds, past_key_values, use_cache, complexity)
            -> (reasoning_embeddings_list, updated_past_key_values)``.
        use_cache: Whether to use KV caching.
        past_key_values: Initial KV cache (or None).

    Returns:
        ``(all_embeddings_tensor, is_reasoning_mask)`` where mask is True
        for positions to be removed from the final output.
    """
    log.debug("Expanding %d marker(s) into reasoning embeddings", len(markers))

    all_embeds: List[torch.Tensor] = []
    remove_mask: List[bool] = []
    cur_past = past_key_values
    processed = 0

    for start_idx, gap_len, complexity, start_len, end_len in markers:
        end_idx = min(start_idx + start_len + gap_len + end_len, input_ids.shape[1])

        # Guard against an empty section (end_idx <= processed), which can arise
        # from overlapping markers. A zero-length section would still append a
        # stray remove_mask entry below, desyncing len(mask) from the expanded
        # embeddings length. Skip such markers entirely.
        if end_idx <= processed:
            continue

        section_ids = input_ids[:, processed:end_idx]
        section_embeds = embed_fn(section_ids)

        all_embeds.append(section_embeds)

        # Mask logic: section tokens are kept (False) except the marker's end
        # token which is removed (True). For reasoning embeddings, all are
        # removed (True) except the last one (False), which carries the final
        # reasoning state into the visible output sequence.
        keep = max(section_embeds.shape[1] - 1, 0)
        remove_mask.extend([False] * keep)
        remove_mask.append(True)

        # Generate reasoning embeddings
        reasoning, cur_past = reasoning_forward_fn(
            all_embeddings=all_embeds if not use_cache else None,
            inputs_embeds=section_embeds,
            past_key_values=cur_past,
            use_cache=use_cache,
            complexity=complexity,
        )

        for e in reasoning:
            all_embeds.append(e.unsqueeze(1))

        # Mask reasoning embeddings: all except last are removed
        remove_mask.extend([True] * max(len(reasoning) - 1, 0))
        remove_mask.append(False)

        processed = end_idx

    # Process remaining tokens after the last marker
    if processed < input_ids.shape[1]:
        rest_ids = input_ids[:, processed:]
        rest_embeds = embed_fn(rest_ids)
        all_embeds.append(rest_embeds)
        remove_mask.extend([False] * rest_embeds.shape[1])

    all_embeddings = torch.cat([e.to(all_embeds[0].device) for e in all_embeds], dim=1)
    mask = torch.tensor(remove_mask, dtype=torch.bool, device=all_embeddings.device)
    return all_embeddings, mask
