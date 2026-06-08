"""Monkeypatch OpenRLHF's Actor and rollout executor for LiteReason.

Works with upstream OpenRLHF (no fork):
1) Swap ``AutoModelForCausalLM`` → ``AutoModelForCausalLMWithReasoning``
2) Attach a tokenizer so marker detection works
3) Patch ``Actor.forward`` for batched marker expansion via ``shadow_forward``
4) Strip dummy token IDs from vLLM rollouts so training sees marker-only tokens

To activate, call ``patch_openrlhf_actor()`` before running OpenRLHF,
or use: ``python -m litereason.patches.openrlhf.train_ppo_ray ...``

Additional safety env vars:
- ``LITEREASON_GUARD_NONFINITE=1`` (default): skip optimizer step when loss is non-finite.
- ``LITEREASON_MAX_LOSS=1e6``: default threshold value; skip optimizer step when finite
  loss magnitude exceeds it.
- ``LITEREASON_DIAGNOSTIC_LOGGING=1``: enable debug diagnostics for gradients/embeddings
  (default off).
"""


import functools
import logging
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from litereason.causal_lm_with_reasoning import (
    _detect_checkpoint_projector_keys,
    _extract_hub_kwargs,
    _infer_num_reasoning_layers_from_keys,
    _maybe_load_projector_weights,
    attach_litereason_to_causal_lm,
)
from litereason.token_utils import ImplicitThoughtDetector

log = logging.getLogger("litereason.patches.openrlhf")


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if not v:
        return bool(default)
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    if not v:
        return int(default)
    try:
        return int(v.strip())
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    if not v:
        return float(default)
    try:
        return float(v.strip())
    except (TypeError, ValueError):
        return float(default)


def _get_or_create_detector(tokenizer: PreTrainedTokenizerBase) -> ImplicitThoughtDetector:
    det = getattr(tokenizer, "_litereason_thought_detector", None)
    if det is None:
        det = ImplicitThoughtDetector(tokenizer)
        tokenizer._litereason_thought_detector = det
    return det


def _strip_dummy_tokens_after_markers(
    token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    prompt_len: int,
    action_ranges: Sequence[Tuple[int, int]],
) -> Tuple[List[int], int, List[int]]:
    """Remove latent-step dummy tokens emitted by the vLLM plugin.

    vLLM emits a dummy token ID for each latent step, but those tokens are not
    "real" actions. We remove them so the training model can reconstruct latent
    steps from the marker tags alone.

    Returns:
        cleaned_ids, cleaned_prompt_len, kept_old_indices
    """
    det = _get_or_create_detector(tokenizer)
    ids = list(token_ids)
    if not ids:
        return [], 0, []

    try:
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        markers = det.find_thought_markers(input_ids)
    except (RuntimeError, ValueError) as e:
        log.warning("LiteReason: marker detection failed in dummy-token stripping: %s", e)
        return ids, int(prompt_len), list(range(len(ids)))

    to_remove: set[int] = set()
    L = len(ids)
    in_action_span = [False] * L
    for start, end in action_ranges:
        s = max(0, int(start))
        e = min(L, int(end))
        for i in range(s, e):
            in_action_span[i] = True

    max_steps = max(1, _env_int("LITEREASON_MAX_REASONING_STEPS", 32))
    for start_idx, gap_len, complexity, start_len, end_len in markers:
        end_idx = int(start_idx + start_len + gap_len + end_len)
        n = max(1, min(int(complexity), max_steps))
        if n <= 0:
            continue
        for j in range(end_idx, min(end_idx + n, L)):
            # vLLM inserts latent-step dummies only during decode.
            # Never strip prompt-side tokens even if prompt contains markers.
            if in_action_span[j]:
                to_remove.add(j)

    kept_old = [i for i in range(L) if i not in to_remove]
    cleaned = [ids[i] for i in kept_old]
    cleaned_prompt_len = sum(1 for i in kept_old if i < int(prompt_len))
    return cleaned, cleaned_prompt_len, kept_old


def _build_extended_attention_mask(
    attention_mask_1d: torch.Tensor,
    markers: Sequence[Tuple[int, int, int, int, int]],
    max_steps: int,
) -> torch.Tensor:
    """Insert 1s for latent steps to align with shadow_forward embeddings.

    After shadow_forward expands markers into reasoning embeddings, the
    sequence is longer than the original. This inserts 1s into the attention
    mask at each marker position to match the expanded length. The inserted
    count per marker must use the SAME clamp shadow_forward applies,
    max(1, min(complexity, max_steps)); otherwise a generated marker whose
    complexity exceeds max_steps would make the mask longer than the
    embeddings, overflowing the per-sample assignment in the caller.
    """
    chunks: List[torch.Tensor] = []
    processed = 0
    for start_idx, gap_len, complexity, start_len, end_len in markers:
        end_idx = int(start_idx + start_len + gap_len + end_len)
        end_idx = min(end_idx, int(attention_mask_1d.numel()))
        n = max(1, min(int(complexity), max_steps))
        chunks.append(attention_mask_1d[processed:end_idx])
        chunks.append(torch.ones(n, dtype=attention_mask_1d.dtype, device=attention_mask_1d.device))
        processed = end_idx
    if processed < int(attention_mask_1d.numel()):
        chunks.append(attention_mask_1d[processed:])
    return torch.cat(chunks, dim=0)


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.LongTensor:
    # OpenRLHF uses this convention for left padding.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def patch_openrlhf_actor(tokenizer_name: Optional[str] = None) -> None:
    """Monkeypatch OpenRLHF to run LiteReason with upstream OpenRLHF.

    Env vars:
    - ``LITEREASON_ENABLE_OPENRLHF=1`` (default): enable patched forward + dummy stripping.
    - ``LITEREASON_OPENRLHF_INIT_PROJECTOR=1``: init projector from pretrained after load.
    """
    # imported lazily so this module imports without openrlhf
    from openrlhf.models import actor as actor_module
    from openrlhf.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
    from openrlhf.models.utils import compute_entropy, log_probs_from_logits
    from openrlhf.utils import agent as agent_module

    if getattr(actor_module, "_litereason_actor_patched", False):
        return

    enable_openrlhf = _env_flag("LITEREASON_ENABLE_OPENRLHF", True)
    if not enable_openrlhf:
        return

    # --- Patch DeepspeedStrategy: skip optimizer step on non-finite loss ---
    # imported lazily so this module imports without openrlhf
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    if not getattr(DeepspeedStrategy, "_litereason_nonfinite_guard_patched", False):
        _orig_strategy_backward = DeepspeedStrategy.backward
        _orig_strategy_optimizer_step = DeepspeedStrategy.optimizer_step

        def _zero_grads(target_model, optimizer):
            model_engine = getattr(target_model, "model", target_model)
            # set_to_none is unsupported on some older zero_grad signatures; fall back.
            try:
                model_engine.zero_grad(set_to_none=True)
            except TypeError:
                model_engine.zero_grad()
            if optimizer is not None:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()

        def _has_nonfinite_gradients(target_model) -> Tuple[bool, float]:
            model_engine = getattr(target_model, "model", target_model)
            total_sq = 0.0
            has_any = False
            for p in model_engine.parameters():
                g = getattr(p, "grad", None)
                if g is None:
                    continue
                has_any = True
                if not torch.isfinite(g).all():
                    return True, float("nan")
                gn = float(g.detach().float().norm().item())
                total_sq += gn * gn
            grad_norm = (total_sq ** 0.5) if has_any else 0.0
            return False, grad_norm

        def _reduce_any_flag(local_bad: bool, ref_tensor: torch.Tensor) -> bool:
            global_bad = local_bad
            if dist.is_initialized():
                reduce_device = ref_tensor.device
                if (not ref_tensor.is_cuda) and torch.cuda.is_available():
                    reduce_device = torch.device("cuda", torch.cuda.current_device())
                bad = torch.tensor(
                    1 if local_bad else 0,
                    device=reduce_device,
                    dtype=torch.int32,
                )
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                global_bad = bool(bad.item())
            return global_bad

        @functools.wraps(_orig_strategy_backward)
        def _patched_strategy_backward(self, loss: torch.Tensor, model, optimizer, **kwargs):
            if _env_flag("LITEREASON_GUARD_NONFINITE", True) and torch.is_tensor(loss):
                diag = _env_flag("LITEREASON_DIAGNOSTIC_LOGGING", False)
                max_loss = _env_float("LITEREASON_MAX_LOSS", 1e6)
                finite_ok = bool(torch.isfinite(loss.detach()).all().item())
                abs_loss = float(loss.detach().float().abs().max().item())
                local_nonfinite = not finite_ok
                local_extreme = finite_ok and (abs_loss > max_loss)
                global_nonfinite = _reduce_any_flag(local_nonfinite, ref_tensor=loss)
                global_extreme = _reduce_any_flag(local_extreme, ref_tensor=loss)

                if global_nonfinite or global_extreme:
                    self._litereason_skip_next_optim_step = True
                    reason = (
                        "non-finite loss"
                        if global_nonfinite
                        else f"extreme loss (|loss|>{max_loss:.3e})"
                    )
                    self._litereason_skip_reason = reason
                    _zero_grads(model, optimizer)
                    if self.is_rank_0():
                        if global_nonfinite:
                            if loss.numel() == 1:
                                v = float(loss.detach().float().item())
                                log.warning(
                                    "LiteReason: detected non-finite loss (%s); skipping optimizer step.",
                                    v,
                                )
                            else:
                                log.warning(
                                    "LiteReason: detected non-finite loss tensor; skipping optimizer step."
                                )
                        else:
                            log.warning(
                                "LiteReason: detected extreme finite loss (|loss|=%s, max=%s); skipping optimizer step.",
                                f"{abs_loss:.3e}",
                                f"{max_loss:.3e}",
                            )
                    return

                _orig_strategy_backward(self, loss, model, optimizer, **kwargs)

                local_bad_grad, grad_norm = _has_nonfinite_gradients(model)
                global_bad_grad = _reduce_any_flag(local_bad_grad, ref_tensor=loss)

                if diag and self.is_rank_0() and loss.numel() == 1:
                    log.debug(
                        "LiteReason diag: backward loss=%s grad_norm=%s bad_grad=%s",
                        f"{float(loss.detach().float().item()):.3e}",
                        f"{grad_norm:.3e}",
                        global_bad_grad,
                    )

                if global_bad_grad:
                    self._litereason_skip_next_optim_step = True
                    self._litereason_skip_reason = "non-finite gradients"
                    _zero_grads(model, optimizer)
                    if self.is_rank_0():
                        log.warning(
                            "LiteReason: detected non-finite gradients; skipping optimizer step."
                        )
                return

            return _orig_strategy_backward(self, loss, model, optimizer, **kwargs)

        @functools.wraps(_orig_strategy_optimizer_step)
        def _patched_strategy_optimizer_step(
            self,
            optimizer,
            model,
            scheduler,
            name="model",
            **kwargs,
        ):
            if _env_flag("LITEREASON_GUARD_NONFINITE", True) and getattr(
                self, "_litereason_skip_next_optim_step", False
            ):
                self._litereason_skip_next_optim_step = False
                reason = getattr(self, "_litereason_skip_reason", "non-finite loss")
                self._litereason_skip_reason = ""
                _zero_grads(model, optimizer)
                if self.is_rank_0():
                    log.warning("LiteReason: skipped %s optimizer step (%s).", name, reason)
                return

            return _orig_strategy_optimizer_step(
                self,
                optimizer,
                model,
                scheduler,
                name=name,
                **kwargs,
            )

        DeepspeedStrategy.backward = _patched_strategy_backward
        DeepspeedStrategy.optimizer_step = _patched_strategy_optimizer_step
        DeepspeedStrategy._litereason_nonfinite_guard_patched = True
        log.info("LiteReason: patched DeepspeedStrategy with non-finite loss guard")

    # --- Global patch: AutoModelForCausalLM.from_pretrained ---
    # AutoLigerKernelForCausalLM inherits from AutoModelForCausalLM and calls
    # super().from_pretrained(), so a single patch on the base class handles
    # both the Liger and non-Liger code paths automatically.
    _orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    @classmethod  # type: ignore[misc]
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = _orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
        cfg = getattr(model, "config", None)
        has_litereason_config = cfg and getattr(cfg, "is_litereason_model", False)

        if has_litereason_config or _env_flag("LITEREASON_ENABLE_OPENRLHF", True):
            if not getattr(model, "_litereason_attached", False):
                hub_kwargs = _extract_hub_kwargs(kwargs)

                if has_litereason_config:
                    # Existing litereason checkpoint: recover layers from config or weights.
                    num_layers = getattr(cfg, "litereason_num_reasoning_layers", None)
                    if num_layers is None:
                        keys = _detect_checkpoint_projector_keys(
                            pretrained_model_name_or_path, hub_kwargs=hub_kwargs
                        )
                        num_layers = _infer_num_reasoning_layers_from_keys(keys)
                    if num_layers is None:
                        num_layers = 3
                    max_steps = getattr(cfg, "litereason_max_reasoning_steps", 32)
                else:
                    # Base model (no litereason config): use env vars or defaults.
                    num_layers = int(os.environ.get("LITEREASON_NUM_REASONING_LAYERS", "3"))
                    max_steps = 32

                attach_litereason_to_causal_lm(
                    model, num_reasoning_layers=num_layers, max_reasoning_steps=max_steps
                )

                if has_litereason_config:
                    # Load saved projector weights from the checkpoint.
                    _maybe_load_projector_weights(
                        model,
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        hub_kwargs=hub_kwargs,
                    )
                else:
                    # Warm-start projector from last MLP block for base models.
                    try:
                        model.init_reasoning_projector_from_pretrained()
                        log.info(
                            "LiteReason: warm-started projector from last MLP for base model %s",
                            pretrained_model_name_or_path,
                        )
                    except (AttributeError, ValueError, RuntimeError) as e:
                        log.warning("LiteReason: could not warm-start projector from last MLP: %s", e)

            # attach_litereason_to_causal_lm sets `_litereason_attached=True` on
            # success, so reaching this branch means the attach above genuinely
            # failed (not merely that we skipped an already-attached model).
            if _env_flag("LITEREASON_ENABLE_OPENRLHF", True) and not getattr(model, "_litereason_attached", False):
                log.error(
                    "LiteReason: LITEREASON_ENABLE_OPENRLHF=1 but model was not attached after from_pretrained. "
                    "This is unexpected; projector will be missing from saved checkpoints."
                )
        return model

    AutoModelForCausalLM.from_pretrained = _patched_from_pretrained
    log.info("LiteReason: patched AutoModelForCausalLM.from_pretrained (global)")

    # --- Patch Actor.__init__ to attach tokenizer ---
    orig_init = actor_module.Actor.__init__

    @functools.wraps(orig_init)
    def patched_init(self, pretrain_or_model, *args, **kwargs):
        orig_init(self, pretrain_or_model, *args, **kwargs)

        if not isinstance(pretrain_or_model, str):
            return

        tok_name = tokenizer_name or pretrain_or_model
        tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

        if hasattr(self, "model") and hasattr(self.model, "set_tokenizer"):
            self.model.set_tokenizer(tok)
            log.info("LiteReason: attached tokenizer to Actor model (%s)", type(self.model).__name__)

        if _env_flag("LITEREASON_OPENRLHF_INIT_PROJECTOR", False):
            if hasattr(self, "model") and hasattr(self.model, "init_reasoning_projector_from_pretrained"):
                try:
                    self.model.init_reasoning_projector_from_pretrained()
                except (AttributeError, RuntimeError) as e:
                    log.warning("LiteReason: init_reasoning_projector_from_pretrained failed: %s", e)

    actor_module.Actor.__init__ = patched_init

    # --- Patch Actor.forward for marker expansion via shadow_forward ---
    orig_forward = actor_module.Actor.forward

    @functools.wraps(orig_forward)
    def patched_forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        allgather_logits: bool = False,
        return_logprobs: bool = False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy: bool = False,
    ):
        def _delegate():
            return orig_forward(
                self, sequences, action_mask,
                attention_mask=attention_mask, return_output=return_output,
                allgather_logits=allgather_logits, return_logprobs=return_logprobs,
                ring_attn_group=ring_attn_group, packed_seq_lens=packed_seq_lens,
                return_entropy=return_entropy,
            )

        if not hasattr(getattr(self, "model", None), "shadow_forward"):
            return _delegate()

        if attention_mask is None:
            attention_mask = torch.ones_like(sequences, dtype=torch.long)

        # Fast check: if no marker start tokens appear anywhere, we can delegate.
        tokenizer = getattr(self.model, "_litereason_tokenizer", None)
        det = getattr(self.model, "_litereason_thought_detector", None)
        if tokenizer is None or det is None:
            raise RuntimeError(
                "LiteReason tokenizer is not attached to the OpenRLHF Actor model. "
                "Ensure `patch_openrlhf_actor()` runs before constructing the Actor."
            )

        if not det._start_first_tokens:
            return _delegate()

        start_tokens = torch.tensor(sorted(det._start_first_tokens), device=sequences.device)
        diagnostic_logging = _env_flag("LITEREASON_DIAGNOSTIC_LOGGING", False)
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        # --- Packed path: multiple samples concatenated into one sequence ---
        if getattr(self, "packing_samples", False):
            if not bool(torch.isin(sequences, start_tokens).any()):
                return _delegate()
            log.debug("patched_forward: markers detected in packed sequences, running shadow_forward")

            if ring_attn_group is not None:
                # In ring attention, OpenRLHF slices packed sequences per rank, so we
                # cannot reconstruct per-sample boundaries robustly here yet.
                raise NotImplementedError(
                    "LiteReason OpenRLHF patch currently supports `packing_samples` only with "
                    "`ring_attn_size==1` (ring_attn_group=None)."
                )

            batch, seqlen = sequences.size()
            packed_ids, packed_pos, rolled_ids, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )

            if not bool(torch.isin(packed_ids, start_tokens).any()):
                return _delegate()

            # In packed mode, OpenRLHF concatenates multiple samples into one
            # sequence. We find sample boundaries by looking for position_id
            # resets (pos == 0), then run shadow_forward on each segment independently.
            pos_1d = packed_pos[0]
            starts = (pos_1d == 0).nonzero(as_tuple=True)[0].tolist()
            if not starts or starts[0] != 0:
                starts = [0] + starts
            ends = starts[1:] + [int(packed_ids.size(1))]

            per_embeds: List[torch.Tensor] = []
            per_remove_masks: List[torch.Tensor] = []
            per_pos_ids: List[torch.Tensor] = []

            for s, e in zip(starts, ends, strict=True):
                seg_ids = packed_ids[:, s:e]
                seg_emb, seg_rm = self.model.shadow_forward(
                    input_ids=seg_ids, use_cache=False,
                )
                per_embeds.append(seg_emb)
                per_remove_masks.append(seg_rm.to(device=seg_emb.device))
                per_pos_ids.append(
                    torch.arange(int(seg_emb.size(1)), dtype=torch.long, device=seg_emb.device).unsqueeze(0)
                )

            input_embeds = torch.cat(per_embeds, dim=1)
            remove_mask_1d = torch.cat(per_remove_masks, dim=0)
            pos_ids = torch.cat(per_pos_ids, dim=1)

            output = self.model(
                input_ids=None,
                inputs_embeds=input_embeds,
                attention_mask=None,
                position_ids=pos_ids,
                use_cache=False,
                return_dict=True,
                given_is_reasoning_embedding_mask=remove_mask_1d,
                output_attentions=False,
                output_hidden_states=False,
            )
            output["logits"] = output["logits"].to(torch.float32)

            if return_entropy:
                assert return_output
                entropy = compute_entropy(output["logits"])
                entropy = gather_and_pad_tensor(
                    entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
                output.entropy = entropy[:, :-1]

            return_action_log_probs = action_mask is not None
            if not return_action_log_probs and not return_logprobs:
                assert return_output
                if allgather_logits:
                    output["logits"] = gather_and_pad_tensor(
                        output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                    )
                return output

            log_probs = log_probs_from_logits(
                output["logits"], rolled_ids, temperature=getattr(self, "temperature", 1.0)
            )
            log_probs = gather_and_pad_tensor(
                log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
            )
            log_probs = log_probs[:, :-1]

            if not return_action_log_probs and return_logprobs:
                return (log_probs, output) if return_output else log_probs

            action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
            if diagnostic_logging and is_rank0 and self.training and self.model.training:
                emb_norm = float(per_embeds[0].detach().float().norm().item()) if per_embeds else 0.0
                log.debug(
                    "LiteReason diag: packed_forward emb_norm0=%s",
                    f"{emb_norm:.3e}",
                )
            return (action_log_probs, output) if return_output else action_log_probs

        # --- Non-packed path: one sample per batch element ---
        if not bool(torch.isin(sequences, start_tokens).any()):
            return _delegate()
        log.debug("patched_forward: markers detected in batch of %d sequences, running shadow_forward", sequences.shape[0])

        B, _ = sequences.shape
        # Clamp latent-step counts with the same bound shadow_forward uses, so the
        # extended attention mask matches the expanded embeddings exactly.
        mask_max_steps = max(
            1, int(getattr(self.model, "_litereason_max_reasoning_steps",
                           _env_int("LITEREASON_MAX_REASONING_STEPS", 32)))
        )
        per_embeds: List[torch.Tensor] = []
        per_remove_masks: List[torch.Tensor] = []
        per_attn: List[torch.Tensor] = []

        for i in range(B):
            seq_i = sequences[i : i + 1]
            seg_emb, seg_rm = self.model.shadow_forward(
                input_ids=seq_i, use_cache=False,
            )
            per_embeds.append(seg_emb)
            per_remove_masks.append(seg_rm.to(device=seg_emb.device))

            # If markers expanded the sequence, extend the attention mask.
            if seg_emb.size(1) != seq_i.size(1):
                markers_i = det.find_thought_markers(seq_i)
                per_attn.append(_build_extended_attention_mask(attention_mask[i], markers_i, mask_max_steps))
            else:
                per_attn.append(attention_mask[i])

        # Each sample may expand to a different length after marker expansion.
        # Pad all samples to the longest expanded length and build a 2D remove
        # mask so apply_reasoning_mask can strip the right positions per sample.
        L_ext_max = max(int(e.size(1)) for e in per_embeds)
        H = int(per_embeds[0].size(2))
        device = per_embeds[0].device
        dtype = per_embeds[0].dtype

        input_embeds = torch.zeros((B, L_ext_max, H), dtype=dtype, device=device)
        remove_mask = torch.ones((B, L_ext_max), dtype=torch.bool, device=device)
        ext_attn = torch.zeros((B, L_ext_max), dtype=attention_mask.dtype, device=device)

        for i, (emb_i, rm_i, attn_i) in enumerate(zip(per_embeds, per_remove_masks, per_attn, strict=True)):
            L_i = int(emb_i.size(1))
            input_embeds[i, :L_i, :] = emb_i[0]  # shadow_forward returns (1, L, H)
            remove_mask[i, :L_i] = rm_i
            ext_attn[i, : int(attn_i.numel())] = attn_i

        ext_pos = _position_ids_from_attention_mask(ext_attn)

        output = self.model(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=ext_attn,
            position_ids=ext_pos,
            use_cache=False,
            return_dict=True,
            given_is_reasoning_embedding_mask=remove_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            output.entropy = entropy[:, :-1]

        if action_mask is None and not return_logprobs:
            assert return_output
            return output

        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=getattr(self, "temperature", 1.0))
        log_probs = log_probs[:, :-1]

        if action_mask is None and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        if diagnostic_logging and is_rank0 and self.training and self.model.training:
            emb_norm = float(per_embeds[0].detach().float().norm().item()) if per_embeds else 0.0
            log.debug(
                "LiteReason diag: forward batch=%d emb_norm0=%s",
                B,
                f"{emb_norm:.3e}",
            )
        return (action_log_probs, output) if return_output else action_log_probs

    actor_module.Actor.forward = patched_forward

    # --- Patch rollout executor to strip dummy tokens ---
    SingleTurnAgentExecutor = agent_module.SingleTurnAgentExecutor
    orig_execute = SingleTurnAgentExecutor.execute

    @functools.wraps(orig_execute)
    async def patched_execute(
        self, prompt, label, sampling_params, max_length: int,
        hf_tokenizer, llm_engine, images=None,
    ):
        # Delegate generation + reward to the upstream executor (so we track its
        # API exactly across versions), then strip the latent-reasoning dummy
        # tokens vLLM inserted during the rollout so PPO token alignment stays
        # deterministic. Reward is computed upstream on the pre-strip observation,
        # matching prior behavior.
        out = await orig_execute(
            self, prompt, label, sampling_params, max_length,
            hf_tokenizer, llm_engine, images=images,
        )

        # Strip dummy tokens inserted by vLLM during latent reasoning steps.
        obs = out["observation_tokens"]
        prompt_len = out["action_ranges"][0][0]
        cleaned, cleaned_prompt_len, kept_old = _strip_dummy_tokens_after_markers(
            obs,
            tokenizer=hf_tokenizer,
            prompt_len=prompt_len,
            action_ranges=out["action_ranges"],
        )
        if len(cleaned) != len(obs):
            log.debug(
                "Stripped %d dummy tokens from rollout (%d -> %d)",
                len(obs) - len(cleaned), len(obs), len(cleaned),
            )
            out["observation_tokens"] = cleaned
            out["action_ranges"] = [(cleaned_prompt_len, len(cleaned))]

            rlp = out.get("rollout_log_probs")
            if isinstance(rlp, list) and len(rlp) == len(obs):
                out["rollout_log_probs"] = [rlp[i] for i in kept_old]

        return out

    SingleTurnAgentExecutor.execute = patched_execute

    # --- Misconfiguration warnings ---
    # train_ppo_ray.py (the wrapped entrypoint) does
    # os.environ.setdefault("LITEREASON_USE_WRAPPED_ENTRYPOINT", "1") at import,
    # so this warning only fires for someone bypassing it via bare openrlhf.cli.
    if _env_flag("LITEREASON_ENABLE_PROJECTOR_SFT", False) and not _env_flag(
        "LITEREASON_USE_WRAPPED_ENTRYPOINT", False
    ):
        log.warning(
            "LITEREASON_ENABLE_PROJECTOR_SFT=1 requires using "
            "'python -m litereason.patches.openrlhf.train_ppo_ray' as the entry point. "
            "If using 'openrlhf.cli.train_ppo_ray', projector SFT will NOT run."
        )

    actor_module._litereason_actor_patched = True
