"""vLLM plugin for latent reasoning during decoding.

Loaded via vLLM's plugin system (``vllm.general_plugins`` entry point).
Enables LiteReason rollouts inside vLLM without a fork.

Only triggers latent reasoning when the generated sequence ends with
``</implicit_thought>``; mid-prompt markers are handled by the HF
training model, not vLLM decode.

Targets the vLLM v1 runner (>= 0.15); validated against vLLM 0.19.1. Set
``LITEREASON_ENABLE_VLLM=1`` to activate.
"""


import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .causal_lm_with_reasoning import SimpleProjectorMLP
from .reasoning_core import apply_projector_stack
from .token_utils import get_implicit_thought_patterns

log = logging.getLogger(__name__)

# Evaluated once at import time: the plugin only patches vLLM when explicitly
# enabled, so importing this module on a CPU box (no vllm installed) is a no-op.
_ENV_ENABLED = os.getenv("LITEREASON_ENABLE_VLLM", "").lower() in {"1", "true", "yes", "y", "on"}


def _env_enabled() -> bool:
    return _ENV_ENABLED


@dataclass(frozen=True)
class _MarkerMatcher:
    tokenizer: PreTrainedTokenizerBase
    start_patterns: Tuple[Tuple[int, ...], ...]
    end_patterns: Tuple[Tuple[int, ...], ...]
    max_gap_tokens: int
    scan: int
    end_last_tokens: frozenset[int]

    @classmethod
    def build(
        cls, tokenizer: PreTrainedTokenizerBase, max_gap_tokens: int = 3, max_scan: int = 15
    ) -> "_MarkerMatcher":
        start, end = get_implicit_thought_patterns(tokenizer)
        if not start or not end:
            raise RuntimeError("Could not auto-detect <implicit_thought> token patterns.")

        scan = max(len(p) for p in start) + int(max_gap_tokens) + max(len(p) for p in end)
        if scan > max_scan:
            raise ValueError(
                "Implicit thought marker tokenization is unexpectedly long "
                f"(scan={scan} > max_scan={max_scan})."
            )
        end_last = frozenset(int(p[-1]) for p in end if len(p) > 0)
        return cls(
            tokenizer=tokenizer,
            start_patterns=tuple(start),
            end_patterns=tuple(end),
            max_gap_tokens=int(max_gap_tokens),
            scan=int(scan),
            end_last_tokens=end_last,
        )

    def match_end(self, token_history: Sequence[int]) -> Optional[int]:
        """Check if the most recently generated tokens form a complete marker.

        Scans only the last ``scan`` tokens for efficiency. Returns the
        complexity N if a complete ``<implicit_thought>N</implicit_thought>``
        marker is found at the end, else None.
        """
        if not token_history:
            return None

        last = int(token_history[-1])
        if last not in self.end_last_tokens:
            return None

        # Only need a small suffix to match: <= scan tokens.
        window = token_history[-self.scan :]
        win_len = len(window)

        for end in self.end_patterns:
            le = len(end)
            if le == 0 or le > win_len:
                continue
            if tuple(window[win_len - le :]) != end:
                continue

            for gap in range(0, self.max_gap_tokens + 1):
                j = win_len - le - gap
                if j < 0:
                    continue
                gap_tokens = window[j : win_len - le]

                for start in self.start_patterns:
                    ls = len(start)
                    i = j - ls
                    if ls == 0 or i < 0:
                        continue
                    if tuple(window[i:j]) != start:
                        continue

                    # Decode the gap tokens to get the integer complexity. A
                    # non-digit gap or non-positive value means THIS alignment is
                    # not a valid marker; keep trying other gap/end/start patterns
                    # instead of giving up on the whole match.
                    complexity_str = self.tokenizer.decode(list(gap_tokens)).strip()
                    if not complexity_str.isdigit():
                        continue
                    n = int(complexity_str)
                    if n > 0:
                        return n

        return None


def _pick_dummy_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    for s in (" ", "\n", "\t"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    # Fallback: first vocab token is always valid.
    return 0


_GELU_FAMILY = frozenset({"gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"})


def _get_scalar_act_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    _MAP = {"relu": F.relu, "silu": F.silu, "gelu": F.gelu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
    fn = _MAP.get(name.lower())
    if fn is None:
        raise ValueError(f"Unsupported activation for vLLM projector: {name!r}")
    return fn


class _VllmProjectorMLP(nn.Module):
    """Projector MLP using vLLM's parallel linear layers for tensor parallelism.

    Mirrors the structure of ``SimpleProjectorMLP`` but uses vLLM's
    ``ColumnParallelLinear`` / ``RowParallelLinear`` so it participates in
    TP sharding. Supports SwiGLU (silu), GeGLU (gelu family), and a generic
    fallback for any other gated activation (act(gate) * up).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[object],
        activation: str = "silu",
        approximate: str = "tanh",
        prefix: str,
    ) -> None:
        super().__init__()
        # imported lazily so this module imports without vllm
        from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

        self.gate_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self._activation = activation
        if activation == "silu":
            # imported lazily so this module imports without vllm
            from vllm.model_executor.layers.activation import SiluAndMul
            self.act_fn = SiluAndMul()
            self._scalar_act = None
        elif activation in _GELU_FAMILY:
            # imported lazily so this module imports without vllm
            from vllm.model_executor.layers.activation import GeluAndMul
            self.act_fn = GeluAndMul(approximate=approximate)
            self._scalar_act = None
        else:
            # Generic fallback: apply activation element-wise to gate, multiply with up
            self.act_fn = None
            self._scalar_act = _get_scalar_act_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        if self.act_fn is not None:
            x = self.act_fn(torch.cat([gate, up], dim=-1))
        else:
            x = self._scalar_act(gate) * up
        x, _ = self.down_proj(x)
        return x


def _get_projector_shape_config(cfg: object) -> object:
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None):
        return text_cfg
    return cfg


def _infer_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _ensure_reasoning_projector_module(
    model: nn.Module,
    num_reasoning_layers: int,
    device: Optional[torch.device] = None,
) -> bool:
    if hasattr(model, "reasoning_projector"):
        return False

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("vLLM model has no `.config`; cannot attach reasoning projector.")

    shape_cfg = _get_projector_shape_config(cfg)
    hidden_size = int(getattr(shape_cfg, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(shape_cfg, "intermediate_size", 0) or 0)
    if hidden_size <= 0 or intermediate_size <= 0:
        raise RuntimeError("Could not infer hidden/intermediate size from vLLM model config.")

    activation = str(
        getattr(shape_cfg, "hidden_act", "")
        or getattr(shape_cfg, "hidden_activation", "")
        or "silu"
    )

    model.reasoning_projector = nn.ModuleList(
        [SimpleProjectorMLP(hidden_size, intermediate_size, activation) for _ in range(int(num_reasoning_layers))]
    )
    model.NUM_REASONING_LAYERS = int(num_reasoning_layers)
    model.reasoning_projector.to(device or _infer_module_device(model))
    return True


def _attach_projector_to_vllm_model(
    model: nn.Module,
    num_reasoning_layers: int,
    device: torch.device,
) -> None:
    if hasattr(model, "reasoning_projector"):
        return

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("vLLM model has no `.config`; cannot attach reasoning projector.")

    shape_cfg = _get_projector_shape_config(cfg)
    hidden_size = int(getattr(shape_cfg, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(shape_cfg, "intermediate_size", 0) or 0)
    if hidden_size <= 0 or intermediate_size <= 0:
        raise RuntimeError("Could not infer hidden/intermediate size from vLLM model config.")

    quant_config = getattr(model, "quant_config", None)

    model_type = str(getattr(shape_cfg, "model_type", "") or getattr(cfg, "model_type", "") or "")
    hidden_act = str(getattr(shape_cfg, "hidden_act", "") or "")
    hidden_activation = str(getattr(shape_cfg, "hidden_activation", "") or "")

    layers: List[nn.Module] = []
    for i in range(int(num_reasoning_layers)):
        prefix = f"reasoning_projector.{i}"
        if model_type.startswith("gemma"):
            approximate = "none" if hidden_activation == "gelu" else "tanh"
            layers.append(
                _VllmProjectorMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    quant_config=quant_config,
                    activation="gelu",
                    approximate=approximate,
                    prefix=prefix,
                )
            )
        else:
            act = hidden_act if hidden_act else "silu"
            layers.append(
                _VllmProjectorMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    quant_config=quant_config,
                    activation=act,
                    prefix=prefix,
                )
            )

    model.reasoning_projector = nn.ModuleList(layers)
    model.NUM_REASONING_LAYERS = int(num_reasoning_layers)
    model.reasoning_projector.to(device)


def _init_projector_from_last_mlp(model: nn.Module) -> None:
    """Warm-start projector weights from the last transformer block's MLP."""
    if not hasattr(model, "reasoning_projector"):
        return

    # Probe known model layouts in order; the wrong-layout ones legitimately raise
    # AttributeError/IndexError/TypeError, so we swallow those and try the next.
    last_mlp = None
    for get_last_mlp in [
        lambda m: m.language_model.model.layers[-1].mlp,
        lambda m: m.model.layers[-1].mlp,
        lambda m: m.transformer.h[-1].mlp,
    ]:
        try:
            last_mlp = get_last_mlp(model)
            break
        except (AttributeError, IndexError, TypeError):
            log.debug("last-MLP lookup failed, trying next layout")
    if last_mlp is None:
        return

    down = getattr(last_mlp, "down_proj", None)
    if down is None:
        log.warning(
            "LiteReason vLLM: last MLP has no down_proj; leaving projector at "
            "random init (warm-start skipped)."
        )
        return

    # Two layouts: a fused gate_up_proj (vLLM's MergedColumnParallelLinear, as in
    # Qwen2/Qwen3) or separate gate_proj/up_proj (some HF/base architectures).
    fused = getattr(last_mlp, "gate_up_proj", None)
    if fused is not None:
        w = fused.weight.data
        if w.dim() != 2 or w.size(0) % 2 != 0:
            log.warning(
                "LiteReason vLLM: fused gate_up_proj has unexpected shape %s; "
                "leaving projector at random init.", tuple(w.shape),
            )
            return
        w_gate, w_up = w.chunk(2, dim=0)
    else:
        gate = getattr(last_mlp, "gate_proj", None)
        up = getattr(last_mlp, "up_proj", None)
        if gate is None or up is None:
            log.warning(
                "LiteReason vLLM: last MLP exposes neither a fused gate_up_proj nor "
                "separate gate_proj/up_proj; leaving projector at random init "
                "(warm-start skipped)."
            )
            return
        w_gate = gate.weight.data
        w_up = up.weight.data

    with torch.no_grad():
        for proj in model.reasoning_projector:
            try:
                proj.gate_proj.weight.data.copy_(w_gate)
                proj.up_proj.weight.data.copy_(w_up)
                proj.down_proj.weight.data.copy_(down.weight.data)
            except AttributeError:
                log.debug("projector layer missing weight attrs, skipping")
                continue


def _clear_litereason_state(rs: object) -> None:
    """Drop all per-request latent reasoning state from a CachedRequestState.

    vLLM recycles ``CachedRequestState`` objects across requests, so leaving a
    stale ``_litereason_next_embed`` (or a nonzero ``_litereason_remaining``) on a
    finished request could leak a reasoning embedding into an unrelated request
    that later occupies the same slot. Resetting ``remaining`` to 0 and clearing
    the pending embedding makes the slot look like a fresh, non-reasoning request.
    """
    rs._litereason_remaining = 0
    rs._litereason_next_embed = None


def _ensure_runner_state(self) -> None:
    """Lazy initialization on first forward pass.

    Loads the tokenizer, builds the marker matcher, and either finds an
    existing projector (loaded from checkpoint) or creates one from scratch
    with warm-started weights from the last transformer MLP.
    """
    if getattr(self, "_litereason_ready", False):
        return

    tok_name = getattr(self.model_config, "tokenizer", None) or self.model_config.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=self.model_config.trust_remote_code)
    matcher = _MarkerMatcher.build(tokenizer)
    dummy_token_id = _pick_dummy_token_id(tokenizer)

    # Use the unwrapped model (not CUDAGraphWrapper / UBatchWrapper) so the
    # projector lives on the raw nn.Module that apply_model() also returns.
    raw_model = self.get_model() if callable(getattr(self, "get_model", None)) else self.model

    if hasattr(raw_model, "reasoning_projector"):
        # Projector already exists, loaded from checkpoint by _patch_weight_loading.
        # Just ensure it's on the correct device.
        raw_model.reasoning_projector.to(self.device)
    else:
        # Base model without projector in checkpoint: create with vLLM parallel
        # layers and warm-start from the last transformer MLP.
        # imported lazily so this module imports without vllm
        from vllm.config import set_current_vllm_config

        cfg = getattr(raw_model, "config", None)
        num_layers = int(getattr(cfg, "litereason_num_reasoning_layers", 0) or 0)
        if num_layers <= 0:
            num_layers = int(os.getenv("LITEREASON_NUM_REASONING_LAYERS", "3"))
        with set_current_vllm_config(self.vllm_config):
            _attach_projector_to_vllm_model(raw_model, num_reasoning_layers=num_layers, device=self.device)
            _init_projector_from_last_mlp(raw_model)
    self._litereason_model = raw_model
    log.info("LiteReason vLLM: projector attached to model (%d layers)", len(raw_model.reasoning_projector))

    self._litereason_tokenizer = tokenizer
    self._litereason_matcher = matcher
    self._litereason_dummy_token_id = int(dummy_token_id)
    self._litereason_override_mask: List[bool] = []
    self._litereason_step_skip_sampling: List[bool] = []
    self._litereason_ready = True


def _patch_vllm_runner() -> None:
    # imported lazily so this module imports without vllm
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_litereason_patched", False):
        return

    # --- Patch _preprocess to inject reasoning embeddings ---
    orig_preprocess = GPUModelRunner._preprocess

    def patched_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        out = orig_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors)
        if not _env_enabled():
            return out

        _ensure_runner_state(self)

        input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output = out

        num_reqs = self.input_batch.num_reqs
        self._litereason_override_mask = [False] * num_reqs

        # Only support text-only path (input_ids provided, inputs_embeds None) for now.
        if input_ids is None or inputs_embeds is not None:
            # If we have pending reasoning embeds but can't override in this path, fail loudly.
            req_ids = self.input_batch.req_ids
            for i in range(num_reqs):
                rs = self.requests[req_ids[i]]
                if getattr(rs, "_litereason_next_embed", None) is not None or int(
                    getattr(rs, "_litereason_remaining", 0) or 0
                ) > 0:
                    raise NotImplementedError(
                        "LiteReason vLLM plugin currently supports only the text-only (token-id) "
                        "input path. Pending reasoning state exists, but vLLM is providing "
                        "`inputs_embeds` (e.g. multimodal/prompt-embeds path)."
                    )
            return out

        override_mask = [False] * num_reqs

        req_ids = self.input_batch.req_ids
        for i in range(num_reqs):
            rs = self.requests[req_ids[i]]
            if getattr(rs, "_litereason_next_embed", None) is not None:
                override_mask[i] = True

        self._litereason_override_mask = override_mask

        if not any(override_mask):
            return out

        # When a request is in the middle of latent reasoning, we replace the
        # last token's embedding with the projector's output from the previous
        # step. This injects the reasoning embedding into the KV cache without
        # going through the normal token embedding path.
        embed_fn = getattr(self.model, "embed_input_ids", None)
        if embed_fn is None:
            # Some models expose embedding on the inner `.model`.
            embed_fn = getattr(getattr(self.model, "model", None), "embed_input_ids", None)
        if embed_fn is None:
            raise RuntimeError(
                "LiteReason vLLM plugin requires `embed_input_ids` to build inputs_embeds "
                "when overriding the last token embedding."
            )
        embeds = embed_fn(input_ids)
        qsl = self.query_start_loc.np  # CPU-side prefix sums; indices correspond to unpadded tokens.

        for i in range(num_reqs):
            if not override_mask[i]:
                continue
            rs = self.requests[req_ids[i]]
            e = rs._litereason_next_embed
            if e is None:
                raise RuntimeError("LiteReason internal error: override_mask True but next_embed is None.")
            rs._litereason_next_embed = None  # consume

            last_idx = int(qsl[i + 1] - 1)
            if last_idx < 0:
                continue
            embeds[last_idx].copy_(e.to(device=embeds.device, dtype=embeds.dtype).view(-1))

        self.inputs_embeds.gpu[:num_input_tokens].copy_(embeds)
        inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
        input_ids = None
        return (input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output)

    GPUModelRunner._preprocess = patched_preprocess

    # --- Patch execute_model to detect markers and start reasoning ---
    orig_execute_model = GPUModelRunner.execute_model

    def patched_execute_model(self, scheduler_output, intermediate_tensors=None):
        out = orig_execute_model(self, scheduler_output, intermediate_tensors)
        if not _env_enabled():
            return out

        # Only v1 text generation path sets execute_model_state and returns None.
        if self.execute_model_state is None:
            return out

        _ensure_runner_state(self)

        state = self.execute_model_state
        sample_hs = state.sample_hidden_states  # [num_reqs, hidden]
        raw_model = getattr(self, "_litereason_model", self.model)
        if not hasattr(raw_model, "reasoning_projector"):
            raise RuntimeError("LiteReason projector missing on vLLM model (unexpected).")

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        override_mask = getattr(self, "_litereason_override_mask", [False] * num_reqs)
        skip_mask = [False] * num_reqs

        # Only run projector when at least one request needs it:
        # either in reasoning mode (override_mask) or last token could be marker end.
        needs_projector = any(override_mask)
        if not needs_projector:
            matcher = self._litereason_matcher
            for i in range(num_reqs):
                cur_len = int(self.input_batch.num_tokens_no_spec[i])
                if cur_len > 0:
                    last_tok = int(self.input_batch.token_ids_cpu[i, cur_len - 1])
                    if last_tok in matcher.end_last_tokens:
                        needs_projector = True
                        break

        if not needs_projector:
            self._litereason_step_skip_sampling = skip_mask
            return out

        with torch.no_grad():
            proj = apply_projector_stack(raw_model.reasoning_projector, sample_hs)

        for i in range(num_reqs):
            rs = self.requests[req_ids[i]]
            remaining = int(getattr(rs, "_litereason_remaining", 0) or 0)

            if override_mask[i]:
                if remaining <= 0:
                    raise RuntimeError(
                        "LiteReason internal error: consumed a reasoning embedding "
                        "but `_litereason_remaining` is not set (>0)."
                    )
                remaining -= 1
                rs._litereason_remaining = remaining
                if remaining > 0:
                    rs._litereason_next_embed = proj[i].detach()
                    skip_mask[i] = True
                else:
                    # Reasoning complete for this request: drop all per-request
                    # latent state so a stale embedding can never be injected if
                    # this CachedRequestState object is recycled for a new request
                    # in a future scheduler step.
                    _clear_litereason_state(rs)
                continue

            if remaining != 0:
                # If we are in reasoning mode but did not consume an embed, something is wrong.
                raise RuntimeError(
                    "LiteReason internal error: `_litereason_remaining` > 0 but no embed was consumed "
                    "in this forward. This implementation assumes no PP and 1-token decode steps."
                )

            # Defensive: this request is in normal-decode mode (remaining == 0 and
            # not overriding). If a pending reasoning embedding is somehow still
            # attached (e.g. a recycled CachedRequestState slot), drop it so it
            # cannot be injected on a later step for an unrelated request.
            if getattr(rs, "_litereason_next_embed", None) is not None:
                rs._litereason_next_embed = None

            # After each decode step, check if the generated sequence now ends
            # with a complete marker. If so, initiate latent reasoning: set
            # _litereason_remaining and store the first projected embedding for
            # injection on the next step.
            cur_len = int(self.input_batch.num_tokens_no_spec[i])
            hist = self.input_batch.token_ids_cpu[i, max(0, cur_len - self._litereason_matcher.scan) : cur_len]
            complexity = self._litereason_matcher.match_end(hist.tolist())
            if complexity is None:
                continue

            log.debug("LiteReason vLLM: request entered reasoning mode (complexity=%d)", complexity)
            # The checkpoint's configured budget is authoritative; fall back to the
            # env var only for a base model whose config carries no value, then 32.
            cfg = getattr(raw_model, "config", None)
            cfg_steps = getattr(cfg, "litereason_max_reasoning_steps", None)
            if cfg_steps is not None:
                # Config-first: 0 is a *configured* value (markers present, latent
                # reasoning disabled) and must be honored, not coerced to env/32.
                max_steps = int(cfg_steps)
            else:
                env_max_steps = os.getenv("LITEREASON_MAX_REASONING_STEPS", "").strip()
                max_steps = int(env_max_steps) if env_max_steps else 32

            # A configured budget of <= 0 means "no latent reasoning for this
            # request", matching the HF training forward (reasoning_forward_core
            # returns no embeddings when complexity <= 0). Do not start reasoning;
            # leave the request in normal-decode mode (remaining stays 0).
            if max_steps <= 0:
                continue

            complexity = max(1, min(int(complexity), max_steps))

            rs._litereason_remaining = complexity
            rs._litereason_next_embed = proj[i].detach()
            skip_mask[i] = True

        self._litereason_step_skip_sampling = skip_mask
        return out

    GPUModelRunner.execute_model = patched_execute_model

    # --- Patch _update_states to emit dummy tokens during reasoning ---
    orig_update_after = GPUModelRunner._update_states_after_model_execute

    def patched_update_after(self, output_token_ids: torch.Tensor, scheduler_output):
        # During latent reasoning steps, the model produces a hidden state but
        # we don't want a real sampled token. Overwrite the output with a dummy
        # (whitespace) token. The training pipeline strips these dummies later.
        if _env_enabled() and getattr(self, "_litereason_ready", False):
            skip_mask = getattr(self, "_litereason_step_skip_sampling", None)
            if skip_mask:
                if output_token_ids.dim() != 2 or output_token_ids.size(1) != 1:
                    raise NotImplementedError(
                        "LiteReason vLLM plugin currently supports only 1-token sampling steps "
                        "(no speculative decoding / jump decoding)."
                    )
                dummy = int(self._litereason_dummy_token_id)
                for i, skip in enumerate(skip_mask):
                    if skip:
                        output_token_ids[i, 0] = dummy
            self._litereason_step_skip_sampling = []
        return orig_update_after(self, output_token_ids, scheduler_output)

    GPUModelRunner._update_states_after_model_execute = patched_update_after

    GPUModelRunner._litereason_patched = True


def _patch_weight_loading() -> None:
    """Attach a reasoning projector before vLLM loads weights.

    If the HF config contains ``litereason_num_reasoning_layers > 0``, the
    checkpoint was trained with a projector and will contain
    ``reasoning_projector.*`` weights.  We attach ``SimpleProjectorMLP``
    modules (regular ``nn.Linear``) so that vLLM's ``AutoWeightsLoader``
    can load those weights through the standard path.
    """
    # imported lazily so this module imports without vllm
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if getattr(DefaultModelLoader, "_litereason_weight_patched", False):
        return

    orig_load_weights = DefaultModelLoader.load_weights

    def patched_load_weights(self, model, model_config):
        cfg = model_config.hf_config
        # Publish the checkpoint's latent-step budget to the env so the in-process
        # dummy-token stripper (which reads LITEREASON_MAX_REASONING_STEPS, default
        # 32) matches the config-first rollout clamp; otherwise it over-strips real
        # tokens whenever a generated marker's count exceeds the env default.
        cfg_steps = getattr(cfg, "litereason_max_reasoning_steps", None)
        if cfg_steps is not None:
            os.environ["LITEREASON_MAX_REASONING_STEPS"] = str(int(cfg_steps))
        num_layers = int(getattr(cfg, "litereason_num_reasoning_layers", 0) or 0)
        if num_layers > 0:
            _ensure_reasoning_projector_module(
                model,
                num_reasoning_layers=num_layers,
                device=_infer_module_device(model),
            )
        orig_load_weights(self, model, model_config)
        if num_layers > 0 and hasattr(model, "reasoning_projector"):
            norms = {}
            for i, layer in enumerate(model.reasoning_projector):
                for pname, p in layer.named_parameters():
                    norms[f"{i}.{pname}"] = p.float().norm().item()
            log.info("LiteReason vLLM: reasoning_projector loaded, norms: %s", norms)
        elif _env_enabled() and not hasattr(model, "reasoning_projector"):
            # Base model without projector in checkpoint: create with vLLM
            # parallel layers so it's allocated before memory profiling.
            #
            # `_attach_projector_to_vllm_model` builds vLLM Column/RowParallelLinear
            # layers plus SiluAndMul/GeluAndMul CustomOps, all of which read the
            # *current* vLLM config (get_current_vllm_config()) at construction
            # time. That config is set during `initialize_model`, but load_weights
            # runs *after* that context exits, so it is no longer active here. The
            # lazy `_ensure_runner_state` path wraps the identical build in
            # `set_current_vllm_config(self.vllm_config)` for exactly this reason;
            # the eager build must do the same or it can fail (notably under TP>1).
            from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config

            env_layers = int(os.getenv("LITEREASON_NUM_REASONING_LAYERS", "3"))
            vllm_config = get_current_vllm_config_or_none()
            if vllm_config is None:
                # No vLLM config is obtainable at this point in load_weights, so we
                # cannot reproduce the construction context the parallel layers need.
                # Rather than build under a missing/incorrect config (which can raise
                # under TP), skip the eager allocation and let the lazy
                # `_ensure_runner_state` path build it on the first forward, where it
                # *does* have `self.vllm_config` to wrap the build. Logged (not
                # silent) so the deferral is visible.
                log.info(
                    "LiteReason vLLM: no current vLLM config during load_weights; "
                    "deferring base-model projector build to first forward "
                    "(_ensure_runner_state)."
                )
            else:
                try:
                    device = next(model.parameters()).device
                    with set_current_vllm_config(vllm_config):
                        _attach_projector_to_vllm_model(
                            model, num_reasoning_layers=env_layers, device=device,
                        )
                        _init_projector_from_last_mlp(model)
                    log.info(
                        "LiteReason vLLM: reasoning_projector created for base model "
                        "(%d layers, before memory profiling)", env_layers,
                    )
                except (RuntimeError, ValueError, StopIteration) as e:
                    # Boundary guard: building the projector eagerly here is only an
                    # optimization (allocate before memory profiling). On failure the
                    # model still has no `reasoning_projector`, so _ensure_runner_state
                    # takes its `else` branch on the first forward and rebuilds it via
                    # _attach_projector_to_vllm_model + _init_projector_from_last_mlp.
                    # Hence we log loudly and continue rather than abort weight loading.
                    log.warning(
                        "LiteReason vLLM: failed to create projector during load_weights "
                        "(will retry in _ensure_runner_state): %s", e,
                    )

    DefaultModelLoader.load_weights = patched_load_weights
    DefaultModelLoader._litereason_weight_patched = True


def _patch_gemma3_mm_load_weights() -> None:
    # imported lazily so this module imports without vllm
    from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration

    if getattr(Gemma3ForConditionalGeneration, "_litereason_weight_patched", False):
        return

    orig_load_weights = Gemma3ForConditionalGeneration.load_weights

    def patched_load_weights(self, weights):
        cfg = getattr(self, "config", None)
        num_layers = int(getattr(cfg, "litereason_num_reasoning_layers", 0) or 0)
        if num_layers > 0:
            _ensure_reasoning_projector_module(
                self,
                num_reasoning_layers=num_layers,
                device=_infer_module_device(self),
            )
        return orig_load_weights(self, weights)

    Gemma3ForConditionalGeneration.load_weights = patched_load_weights
    Gemma3ForConditionalGeneration._litereason_weight_patched = True


def register() -> None:
    """Entry point for vllm.general_plugins.

    Never fatal: if a patch fails here (e.g. pynvml race when multiple Ray
    actors start simultaneously), the actor stays alive and the patch will
    succeed when ``load_general_plugins()`` runs again in vLLM's EngineCore
    or worker subprocess (each has its own ``plugins_loaded`` flag).
    """
    if not _env_enabled():
        return
    log.info("LiteReason vLLM plugin: registering patches")
    # Broad catches are intentional: register() is a vLLM plugin entry point
    # that may fire before the vLLM worker is fully initialized (e.g. pynvml
    # race). Patches will succeed on retry in the subprocess.
    try:
        _patch_weight_loading()
        _patch_gemma3_mm_load_weights()
        log.info("LiteReason vLLM: weight loading patch applied")
    except Exception as e:
        log.warning("LiteReason: weight patch failed (will retry in subprocess): %s", e)
    try:
        _patch_vllm_runner()
        log.info("LiteReason vLLM: runner patch applied")
    except Exception as e:
        log.warning("LiteReason: runner patch failed (will retry in subprocess): %s", e)
