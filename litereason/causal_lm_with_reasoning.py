"""Model-agnostic latent reasoning for HuggingFace causal LMs.

This module implements LiteReason as a *runtime patch* that can be attached to
any HuggingFace `AutoModelForCausalLM` model instance (we tested with Qwen2.5, Mistral and Gemma).

Design goals:
- Works across architectures (Mistral/Gemma/Llama/GPT-style...) without forking
  or subclassing every model.
- Keeps the underlying model class unchanged, so `.generate()` and other HF
  features keep working.
- Adds two new capabilities:
  1) Automatic marker expansion in `forward(input_ids=...)`.
  2) `shadow_forward(...)` for RL rollouts (returns embeddings + remove-mask).
"""


import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .reasoning_core import (
    apply_projector_stack,
    apply_reasoning_mask,
    expand_markers_to_embeddings,
    reasoning_forward_core,
)
from .token_utils import ImplicitThoughtDetector

log = logging.getLogger(__name__)


class SimpleProjectorMLP(nn.Module):
    """Fallback SwiGLU projector matching Qwen2/Llama MLP architecture.

    Uses the same gate_proj/up_proj/down_proj parameter names as HF
    Qwen2MLP so that weights are interchangeable with deepcopy'd MLPs
    and loadable by the vLLM projector.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = _get_activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (.., H)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an activation callable compatible with HF config strings."""
    name = (name or "").lower()
    if name in ACT2FN:
        return ACT2FN[name]
    raise ValueError(f"Unsupported activation: {name!r}")


def _try_get_last_block_mlp(model: nn.Module) -> Optional[nn.Module]:
    """Try to extract the last transformer block's MLP."""
    # Llama-family (Llama/Mistral/Gemma/...) typically uses: model.model.layers[-1].mlp
    try:
        layers = model.model.layers
        if layers:
            mlp = getattr(layers[-1], "mlp", None)
            if isinstance(mlp, nn.Module):
                return mlp
    except (AttributeError, IndexError, TypeError):
        log.debug("model.model.layers[-1].mlp not found, trying next pattern")

    # GPT-2 style: model.transformer.h[-1].mlp
    try:
        layers = model.transformer.h
        if layers:
            mlp = getattr(layers[-1], "mlp", None)
            if isinstance(mlp, nn.Module):
                return mlp
    except (AttributeError, IndexError, TypeError):
        log.debug("model.transformer.h[-1].mlp not found")

    return None


def _infer_config_int(cfg: object, names: Sequence[str], default: int = 0) -> int:
    for candidate in _config_candidates(cfg):
        for name in names:
            v = getattr(candidate, name, None)
            if v is None:
                continue
            try:
                iv = int(v)
            except (TypeError, ValueError):
                log.debug("config attr %s not parseable as int", name)
                continue
            if iv > 0:
                return iv
    return int(default)


def _config_candidates(cfg: object) -> list[object]:
    """Return likely config objects that may hold model-dimension attributes."""
    if cfg is None:
        return []

    candidates: list[object] = [cfg]
    for nested_name in ("text_config", "language_config", "llm_config", "model_config", "decoder_config"):
        nested = getattr(cfg, nested_name, None)
        if nested is not None:
            candidates.append(nested)
    return candidates


def _infer_hidden_size(cfg: object) -> int:
    return _infer_config_int(cfg, ("hidden_size", "n_embd", "d_model", "dim"), default=0)


def _infer_intermediate_size(cfg: object, hidden_size: int) -> int:
    # Configs name this differently across archs. The `4 * hidden_size` default is
    # a last-resort fallback (effectively unreachable for supported archs, which
    # all expose one of these keys; it would only trigger if the key were missing
    # and the last-block-MLP clone also failed).
    return _infer_config_int(cfg, ("intermediate_size", "n_inner", "ffn_dim"), default=4 * hidden_size)


def _infer_activation_name(cfg: object) -> str:
    for candidate in _config_candidates(cfg):
        for name in ("hidden_act", "hidden_activation", "activation_function", "act_fn"):
            v = getattr(candidate, name, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return "silu"


def _build_reasoning_projector(model: nn.Module, num_layers: int) -> nn.ModuleList:
    if num_layers <= 0:
        raise ValueError(f"num_reasoning_layers must be > 0, got {num_layers}")

    last_mlp = _try_get_last_block_mlp(model)
    if last_mlp is not None:
        log.info("Building reasoning projector: cloning last-block MLP (%s) x %d layers", type(last_mlp).__name__, num_layers)
        return nn.ModuleList([copy.deepcopy(last_mlp) for _ in range(num_layers)])

    cfg = getattr(model, "config", None)
    hidden_size = _infer_hidden_size(cfg)
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden_size from model.config; cannot build projector.")

    intermediate_size = _infer_intermediate_size(cfg, hidden_size)
    activation = _infer_activation_name(cfg)
    log.info("Building reasoning projector: SimpleProjectorMLP (hidden=%d, inter=%d, act=%s) x %d layers", hidden_size, intermediate_size, activation, num_layers)
    return nn.ModuleList([SimpleProjectorMLP(hidden_size, intermediate_size, activation) for _ in range(num_layers)])


def _with_loss(
    model: nn.Module, out: CausalLMOutputWithPast, labels: Optional[torch.LongTensor],
) -> CausalLMOutputWithPast:
    """Compute loss and return a *new* CausalLMOutputWithPast with loss in the constructor.

    We must construct a new output (rather than ``out.loss = tensor``) because
    ``ModelOutput.__post_init__`` only adds non-None fields to its underlying
    OrderedDict. Mutating via attribute assignment doesn't update the dict,
    so the HF Trainer (which reads via dict iteration) would see ``loss=None``.
    """
    if labels is None or out.loss is not None:
        return out

    # Some multimodal wrappers (e.g. Gemma3) expose vocab size on nested
    # text config rather than top-level config.
    cfg = getattr(model, "config", None)
    vocab_size = _infer_config_int(cfg, ("vocab_size",), default=0)
    if vocab_size <= 0:
        raise ValueError("Could not infer vocab_size from model config for loss computation.")

    loss = model.loss_function(
        logits=out.logits, labels=labels, vocab_size=vocab_size,
    )
    return CausalLMOutputWithPast(
        loss=loss,
        logits=out.logits,
        past_key_values=out.past_key_values,
        hidden_states=out.hidden_states,
        attentions=out.attentions,
    )


def _get_or_raise_tokenizer(model: nn.Module) -> PreTrainedTokenizerBase:
    tok = getattr(model, "_litereason_tokenizer", None)
    if tok is None:
        raise ValueError(
            "Tokenizer is not set for LiteReason marker detection. "
            "Call `model.set_tokenizer(tokenizer)` or pass `tokenizer=...` to "
            "`AutoModelForCausalLMWithReasoning.from_pretrained(...)`."
        )
    return tok


def _get_or_create_detector(model: nn.Module) -> ImplicitThoughtDetector:
    det = getattr(model, "_litereason_thought_detector", None)
    if det is None:
        det = ImplicitThoughtDetector(_get_or_raise_tokenizer(model))
        model._litereason_thought_detector = det
    return det


def _get_embed_layer(model: nn.Module) -> nn.Module:
    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError("Model has no input embeddings (get_input_embeddings() returned None).")
    return emb


def _is_gemma3_model(model: nn.Module) -> bool:
    """Return True if this model uses Gemma3 causal-mask semantics."""
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None)
    if isinstance(model_type, str) and model_type.lower() == "gemma3":
        return True
    return "gemma3" in model.__class__.__name__.lower()


def _infer_batch_seq_from_forward_kwargs(
    kwargs: Dict[str, Any]
) -> Optional[Tuple[int, int, torch.device]]:
    """Infer `(batch, seq, device)` for token_type_ids from forward kwargs."""
    input_ids = kwargs.get("input_ids", None)
    if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2:
        bsz, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        return bsz, seq, input_ids.device

    inputs_embeds = kwargs.get("inputs_embeds", None)
    if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.dim() == 3:
        bsz, seq = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        return bsz, seq, inputs_embeds.device

    attention_mask = kwargs.get("attention_mask", None)
    if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
        bsz, seq = int(attention_mask.shape[0]), int(attention_mask.shape[1])
        return bsz, seq, attention_mask.device

    return None


def _ensure_gemma3_training_token_type_ids(model: nn.Module, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Populate text-only token_type_ids for Gemma3 training when missing.

    Transformers >=5.2 requires `token_type_ids` for Gemma3 during training.
    For text-only SFT we can safely provide all-zeros token_type_ids.
    """
    if not model.training or not _is_gemma3_model(model):
        return kwargs
    if kwargs.get("token_type_ids", None) is not None:
        return kwargs

    inferred = _infer_batch_seq_from_forward_kwargs(kwargs)
    if inferred is None:
        return kwargs

    bsz, seq, device = inferred
    patched = dict(kwargs)
    patched["token_type_ids"] = torch.zeros((bsz, seq), dtype=torch.long, device=device)
    return patched


# kwargs from .generate() that are sized for the original sequence and must
# NOT be forwarded into the iterative reasoning sub-forwards or the final
# expanded-embeddings forward (Path C).  These would cause shape mismatches
# because marker expansion changes the sequence length.
_GENERATE_SEQ_KWARGS = frozenset({"cache_position", "num_logits_to_keep"})


def _make_reasoning_forward_fn(
    model: nn.Module, **extra_kwargs
) -> Callable[..., Tuple[List[torch.Tensor], Optional[Any]]]:
    """Create a reasoning forward callable for ``expand_markers_to_embeddings``.

    Wraps ``reasoning_forward_core`` with the generic-patch specifics:
    grad-context management for frozen base models, max-step validation,
    and ``hidden_states[-1]`` extraction (CausalLMOutputWithPast).
    """
    base_forward = getattr(model, "_litereason_base_forward", None)
    if base_forward is None:
        raise RuntimeError("LiteReason is not attached (missing _litereason_base_forward).")

    max_steps = int(model._litereason_max_reasoning_steps)

    # Build the autograd graph through the iterative latent rollout whenever
    # gradients are enabled, so that gradients flow back through earlier latent
    # steps into the Reasoning Projector (backprop-through-time). This is
    # required for projector SFT: each latent step depends on the previous one,
    # and the paper trains the projector through a rollout of several steps. The
    # base model's (frozen) parameters still receive no gradient -- only the
    # activations carry it -- so this trains the projector alone.
    # An outer torch.no_grad()/torch.inference_mode() (e.g. the RL shadow_forward
    # rollout, or inference) disables grad here, as expected.
    if torch.is_grad_enabled():
        base_ctx = torch.enable_grad()
    else:
        base_ctx = torch.no_grad()

    # Strip kwargs that are sized for the original sequence length and would
    # cause shape mismatches in the iterative reasoning sub-forwards (e.g.
    # cache_position from .generate()).
    forward_kwargs = {
        k: v for k, v in extra_kwargs.items() if k not in _GENERATE_SEQ_KWARGS
    }
    forward_kwargs["labels"] = None

    # Prefer the inner model (e.g. model.model = Qwen2Model) to avoid
    # storing all hidden states.  The inner model returns
    # BaseModelOutputWithPast whose .last_hidden_state is exactly what we
    # need: only the final layer output, no intermediate storage.
    inner_model = getattr(model, "model", None)
    if inner_model is not None and callable(getattr(inner_model, "forward", None)):
        forward_kwargs["output_hidden_states"] = False

        def _wrapped_base_forward(**kwargs):
            kwargs.pop("labels", None)
            kwargs = _ensure_gemma3_training_token_type_ids(model, kwargs)
            with base_ctx:
                return inner_model(**kwargs)

        def extract_hidden_fn(out):
            return out.last_hidden_state[:, -1, :]
    else:
        # Fallback: CausalLM forward with all hidden states stored.
        forward_kwargs["output_hidden_states"] = True

        def _wrapped_base_forward(**kwargs):
            with base_ctx:
                return base_forward(**kwargs)

        def extract_hidden_fn(out):
            return out.hidden_states[-1][:, -1, :]

    def reasoning_fn(
        all_embeddings, inputs_embeds, past_key_values, use_cache, complexity
    ):
        complexity = max(1, min(complexity, max_steps))
        return reasoning_forward_core(
            base_forward_fn=_wrapped_base_forward,
            projector_fn=lambda h: apply_projector_stack(model.reasoning_projector, h),
            extract_hidden_fn=extract_hidden_fn,
            all_embeddings=all_embeddings,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            complexity=complexity,
            forward_kwargs=forward_kwargs,
        )

    return reasoning_fn


_PROJECTOR_PREFIX = "reasoning_projector."
_PROJECTOR_LAYER_RE = re.compile(r"^reasoning_projector\.(\d+)\.")


def _extract_hub_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the subset of kwargs relevant for hf_hub_download."""
    hub_kwargs: Dict[str, Any] = {}
    for key in ("revision", "subfolder", "cache_dir", "local_files_only"):
        if key in kwargs and kwargs[key] is not None:
            hub_kwargs[key] = kwargs[key]

    token = kwargs.get("token", None)
    if token is None:
        token = kwargs.get("use_auth_token", None)
    if token is not None:
        hub_kwargs["token"] = token

    return hub_kwargs


def _infer_num_reasoning_layers_from_keys(keys: Iterable[str]) -> Optional[int]:
    max_idx = -1
    for k in keys:
        m = _PROJECTOR_LAYER_RE.match(k)
        if m is None:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return (max_idx + 1) if max_idx >= 0 else None


def _detect_checkpoint_projector_keys(
    pretrained_model_name_or_path: str, hub_kwargs: Dict[str, Any]
) -> List[str]:
    """Return reasoning_projector.* keys present in a checkpoint (no tensor loads)."""
    if os.path.isdir(pretrained_model_name_or_path):
        ckpt = Path(pretrained_model_name_or_path)
        safe_index = ckpt / "model.safetensors.index.json"
        if safe_index.exists():
            idx = json.loads(safe_index.read_text(encoding="utf-8"))
            weight_map = idx.get("weight_map", {})
            return [k for k in weight_map.keys() if k.startswith(_PROJECTOR_PREFIX)]

        safe_single = ckpt / "model.safetensors"
        if safe_single.exists():
            with safe_open(str(safe_single), framework="pt", device="cpu") as f:
                return [k for k in f.keys() if k.startswith(_PROJECTOR_PREFIX)]

        bin_index = ckpt / "pytorch_model.bin.index.json"
        if bin_index.exists():
            idx = json.loads(bin_index.read_text(encoding="utf-8"))
            weight_map = idx.get("weight_map", {})
            return [k for k in weight_map.keys() if k.startswith(_PROJECTOR_PREFIX)]

        return []

    # Hub repo id.
    idx_path = _try_hub_download(
        pretrained_model_name_or_path, filename="model.safetensors.index.json", hub_kwargs=hub_kwargs
    )
    if idx_path is not None and idx_path.exists():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        return [k for k in weight_map.keys() if k.startswith(_PROJECTOR_PREFIX)]

    single_path = _try_hub_download(
        pretrained_model_name_or_path, filename="model.safetensors", hub_kwargs=hub_kwargs
    )
    if single_path is not None and single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            return [k for k in f.keys() if k.startswith(_PROJECTOR_PREFIX)]

    bin_idx = _try_hub_download(
        pretrained_model_name_or_path, filename="pytorch_model.bin.index.json", hub_kwargs=hub_kwargs
    )
    if bin_idx is not None and bin_idx.exists():
        idx = json.loads(bin_idx.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        return [k for k in weight_map.keys() if k.startswith(_PROJECTOR_PREFIX)]

    return []


def _load_projector_keys_from_safetensors_file(
    file_path: Path, keys: Optional[Sequence[str]] = None
) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        use_keys = keys if keys is not None else list(f.keys())
        for k in use_keys:
            if not k.startswith(_PROJECTOR_PREFIX):
                continue
            sd[k] = f.get_tensor(k)
    return sd


def _load_projector_state_from_local_dir(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load only reasoning_projector.* tensors from a local HF checkpoint dir."""
    safetensors_index = checkpoint_dir / "model.safetensors.index.json"
    if safetensors_index.exists():
        idx = json.loads(safetensors_index.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        proj_keys = [k for k in weight_map.keys() if k.startswith(_PROJECTOR_PREFIX)]
        if not proj_keys:
            return {}
        by_file: Dict[str, List[str]] = {}
        for k in proj_keys:
            by_file.setdefault(weight_map[k], []).append(k)

        out: Dict[str, torch.Tensor] = {}
        for fname, keys in by_file.items():
            shard = checkpoint_dir / fname
            if not shard.exists():
                raise FileNotFoundError(f"Missing shard file referenced by index: {shard}")
            out.update(_load_projector_keys_from_safetensors_file(shard, keys=keys))
        return out

    safetensors_single = checkpoint_dir / "model.safetensors"
    if safetensors_single.exists():
        return _load_projector_keys_from_safetensors_file(safetensors_single)

    # If an index exists for .bin and contains projector weights, fail loudly:
    bin_index = checkpoint_dir / "pytorch_model.bin.index.json"
    if bin_index.exists():
        idx = json.loads(bin_index.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        has_proj = any(k.startswith(_PROJECTOR_PREFIX) for k in weight_map.keys())
        if has_proj:
            raise RuntimeError(
                "Checkpoint contains reasoning_projector weights, but only sharded PyTorch "
                "checkpoints were found. For memory-efficient loading, save with safetensors "
                "(e.g. `model.save_pretrained(..., safe_serialization=True)`)."
            )

    return {}


def _try_hub_download(repo_id: str, filename: str, hub_kwargs: Dict[str, Any]) -> Optional[Path]:
    try:
        return Path(hf_hub_download(repo_id, filename=filename, **hub_kwargs))
    except (OSError, ValueError):
        # File not found on hub, network error, or invalid repo: expected
        # when probing for optional checkpoint files.
        return None


def _load_projector_state_from_hub(
    repo_id: str, hub_kwargs: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Load only reasoning_projector.* tensors from a HF Hub repo."""
    idx_path = _try_hub_download(repo_id, filename="model.safetensors.index.json", hub_kwargs=hub_kwargs)
    if idx_path is not None and idx_path.exists():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        proj_keys = [k for k in weight_map.keys() if k.startswith(_PROJECTOR_PREFIX)]
        if not proj_keys:
            return {}

        by_file: Dict[str, List[str]] = {}
        for k in proj_keys:
            by_file.setdefault(weight_map[k], []).append(k)

        out: Dict[str, torch.Tensor] = {}
        for fname, keys in by_file.items():
            shard_path = _try_hub_download(repo_id, filename=fname, hub_kwargs=hub_kwargs)
            if shard_path is None:
                raise FileNotFoundError(f"Missing shard file referenced by index: {fname}")
            out.update(_load_projector_keys_from_safetensors_file(shard_path, keys=keys))
        return out

    single_path = _try_hub_download(repo_id, filename="model.safetensors", hub_kwargs=hub_kwargs)
    if single_path is not None and single_path.exists():
        return _load_projector_keys_from_safetensors_file(single_path)

    # If only .bin exists but it contains projector weights, fail loudly.
    bin_idx = _try_hub_download(repo_id, filename="pytorch_model.bin.index.json", hub_kwargs=hub_kwargs)
    if bin_idx is not None and bin_idx.exists():
        idx = json.loads(bin_idx.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        has_proj = any(k.startswith(_PROJECTOR_PREFIX) for k in weight_map.keys())
        if has_proj:
            raise RuntimeError(
                "Repo contains reasoning_projector weights, but only sharded PyTorch checkpoints "
                "were found. For memory-efficient loading, save/upload safetensors weights."
            )

    return {}


def _maybe_load_projector_weights(
    model: nn.Module,
    pretrained_model_name_or_path: str,
    hub_kwargs: Dict[str, Any],
) -> bool:
    """Load trained projector weights if present in the checkpoint."""
    if not hasattr(model, "reasoning_projector"):
        return False

    if os.path.isdir(pretrained_model_name_or_path):
        proj_full = _load_projector_state_from_local_dir(Path(pretrained_model_name_or_path))
    else:
        proj_full = _load_projector_state_from_hub(pretrained_model_name_or_path, hub_kwargs=hub_kwargs)

    if not proj_full:
        return False

    proj_sd = {k[len(_PROJECTOR_PREFIX) :]: v for k, v in proj_full.items()}
    missing, unexpected = model.reasoning_projector.load_state_dict(proj_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load reasoning_projector weights from checkpoint. "
            f"missing={missing} unexpected={unexpected}. "
            "Most likely `num_reasoning_layers` does not match the checkpoint."
        )

    norms = {k: v.float().norm().item() for k, v in proj_sd.items()}
    log.info("LiteReason: loaded reasoning_projector weights, norms: %s", norms)
    return True


def attach_litereason_to_causal_lm(
    model: nn.Module,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    num_reasoning_layers: int = 3,
    max_reasoning_steps: int = 32,
) -> nn.Module:
    """Attach LiteReason behavior to an existing CausalLM instance.

    This mutates the model in-place (adds modules + replaces forward + adds helpers).
    """
    if getattr(model, "_litereason_attached", False):
        # OpenRLHF's global from_pretrained patch can attach LiteReason before
        # the caller has a tokenizer object. If a later caller retries
        # attachment with an explicit tokenizer, refresh marker-detection state.
        if tokenizer is not None:
            if hasattr(model, "set_tokenizer"):
                model.set_tokenizer(tokenizer)
            else:
                model._litereason_tokenizer = tokenizer
                model._litereason_thought_detector = ImplicitThoughtDetector(tokenizer)
        return model

    log.info("Attaching LiteReason to %s with %d projector layers", model.__class__.__name__, num_reasoning_layers)
    model.reasoning_projector = _build_reasoning_projector(model, num_reasoning_layers)
    model.NUM_REASONING_LAYERS = int(num_reasoning_layers)
    model._litereason_max_reasoning_steps = int(max_reasoning_steps)

    # `max_reasoning_steps` is authoritative here: it is the checkpoint's configured
    # value when loading a LiteReason model, or the value the caller resolved for a
    # base model (from LITEREASON_MAX_REASONING_STEPS or the default). Publish it to
    # the env var so env-only readers -- notably the actor-side dummy stripper, which
    # has no model handle -- use the same number, overwriting any stale value. A
    # disagreeing env var (e.g. left over in the shell, or a training script forcing
    # a value) is ignored rather than allowed to silently corrupt a loaded checkpoint.
    _env_steps = os.getenv("LITEREASON_MAX_REASONING_STEPS", "").strip()
    if _env_steps and _env_steps != str(int(max_reasoning_steps)):
        log.warning(
            "Ignoring LITEREASON_MAX_REASONING_STEPS=%s; using the model's "
            "max_reasoning_steps=%d.",
            _env_steps,
            int(max_reasoning_steps),
        )
    os.environ["LITEREASON_MAX_REASONING_STEPS"] = str(int(max_reasoning_steps))

    cfg = getattr(model, "config", None)
    if cfg is not None:
        # Persist on save_pretrained() so future loads can recover defaults.
        cfg.is_litereason_model = True
        cfg.litereason_num_reasoning_layers = int(num_reasoning_layers)
        # Persist the resolved value (after any env override) so a saved+reloaded
        # checkpoint and the runtime model agree.
        cfg.litereason_max_reasoning_steps = int(model._litereason_max_reasoning_steps)

    if tokenizer is not None:
        model._litereason_tokenizer = tokenizer
        model._litereason_thought_detector = ImplicitThoughtDetector(tokenizer)
    else:
        model._litereason_tokenizer = None
        model._litereason_thought_detector = None

    # Capture the original class's forward as an unbound function.
    # This is used by the dynamic subclass's _litereason_base_forward method.
    _OrigCls = model.__class__
    _orig_forward = _OrigCls.forward

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._litereason_tokenizer = tokenizer
        self._litereason_thought_detector = ImplicitThoughtDetector(tokenizer)

    def init_reasoning_projector_from_pretrained(self) -> None:
        last_mlp = _try_get_last_block_mlp(self)
        if last_mlp is None:
            raise ValueError("Could not find a last-block MLP to init from for this architecture.")
        sd = last_mlp.state_dict()
        for layer in self.reasoning_projector:
            layer.load_state_dict(sd, strict=True)

    def shadow_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        # shadow_forward assumes a single unpadded sequence (batch=1), so a full
        # attention mask / position_ids are not used by the marker-expansion path;
        # positions and causal masking are handled implicitly for the lone sequence.
        del attention_mask, position_ids
        with torch.no_grad():
            if input_ids is None:
                if inputs_embeds is None:
                    raise ValueError("shadow_forward requires input_ids or inputs_embeds")
                mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device)
                return inputs_embeds, mask

            if input_ids.shape[0] != 1:
                raise ValueError("shadow_forward supports batch size 1 (call per-sample and pad externally).")

            det = _get_or_create_detector(self)
            markers = det.find_markers_or_none(input_ids)
            if not markers:
                emb = _get_embed_layer(self)(input_ids)
                return emb, torch.zeros(emb.shape[1], dtype=torch.bool, device=emb.device)

            reasoning_fn = _make_reasoning_forward_fn(self, **kwargs)
            return expand_markers_to_embeddings(
                input_ids=input_ids,
                markers=markers,
                embed_fn=_get_embed_layer(self),
                reasoning_forward_fn=reasoning_fn,
                use_cache=bool(use_cache) if use_cache is not None else True,
                past_key_values=past_key_values,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        given_is_reasoning_embedding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        base_fwd_kwargs = dict(
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            **kwargs,
        )

        # Path A: pre-computed embeddings (RL rollouts via shadow_forward)
        if input_ids is None or inputs_embeds is not None:
            log.debug("forward: Path A (pre-computed embeddings)")

            out = self._litereason_base_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                labels=None, **base_fwd_kwargs,
            )
            if given_is_reasoning_embedding_mask is not None:
                apply_reasoning_mask(out, given_is_reasoning_embedding_mask.to(out.logits.device))
            return _with_loss(self, out, labels)

        # Path B: no markers, pass through to the base model unchanged
        det = _get_or_create_detector(self)
        markers = det.find_markers_or_none(input_ids)
        if not markers:
            log.debug("forward: Path B (no markers, base model pass-through)")

            out = self._litereason_base_forward(
                input_ids=input_ids, inputs_embeds=None,
                labels=labels, **base_fwd_kwargs,
            )
            # When the base model is frozen and no markers are present, the
            # loss has no gradient path through the projector. DDP requires
            # all parameters to participate in backward, so we add a
            # zero-contribution term from projector params to satisfy it.
            # Paths A and C already route real gradients through the projector,
            # so only this path (B, no markers) needs the dummy term.
            if (out.loss is not None
                    and not out.loss.requires_grad
                    and hasattr(self, "reasoning_projector")):
                dummy = sum(
                    0.0 * p.float().sum()
                    for p in self.reasoning_projector.parameters()
                )
                out = CausalLMOutputWithPast(
                    loss=out.loss + dummy,
                    logits=out.logits,
                    past_key_values=out.past_key_values,
                    hidden_states=out.hidden_states,
                    attentions=out.attentions,
                )
            return out

        # Path C: marker expansion (batch_size=1)
        log.debug("forward: Path C (marker expansion, %d markers)", len(markers))
        reasoning_fn = _make_reasoning_forward_fn(self, **kwargs)
        all_embeddings, mask = expand_markers_to_embeddings(
            input_ids=input_ids,
            markers=markers,
            embed_fn=_get_embed_layer(self),
            reasoning_forward_fn=reasoning_fn,
            use_cache=bool(use_cache) if use_cache is not None else False,
            past_key_values=past_key_values,
        )
        # Strip sequence-length-dependent kwargs (cache_position etc.)
        # that .generate() passes: they're sized for the original input,
        # not the expanded embeddings.
        path_c_kwargs = {
            k: v for k, v in base_fwd_kwargs.items() if k not in _GENERATE_SEQ_KWARGS
        }
        out = self._litereason_base_forward(
            input_ids=None, inputs_embeds=all_embeddings, labels=None,
            **{**path_c_kwargs, "use_cache": False,
               "attention_mask": None, "position_ids": None,
               "past_key_values": None},
        )
        apply_reasoning_mask(out, mask)
        return _with_loss(self, out, labels)

    # Define _litereason_base_forward as a class method that dispatches to
    # the original (pre-patch) forward via the captured _orig_forward closure.
    # This ensures DataParallel replicas call the correct instance's forward.
    def _litereason_base_forward(self, **kwargs):
        # Gemma3 requires token_type_ids while training even for text-only SFT.
        # Centralizing this here ensures all LiteReason paths (A/B/C + iterative
        # reasoning sub-forwards) pass a valid mask.
        patched_kwargs = _ensure_gemma3_training_token_type_ids(self, kwargs)
        return _orig_forward(self, **patched_kwargs)

    # Override save_pretrained so that saved checkpoints use the base
    # architecture name (e.g. "Qwen2ForCausalLM") instead of the dynamic
    # subclass name ("LiteReasonQwen2ForCausalLM").  HF's save_pretrained
    # unconditionally sets config.architectures = [type(self).__name__]
    # before writing, so we must re-save config after the parent finishes.
    def save_pretrained(self, save_directory, *args, **kwargs):
        result = _OrigCls.save_pretrained(self, save_directory, *args, **kwargs)
        # Fix architecture name and re-save config
        self.config.architectures = [_OrigCls.__name__]
        self.config.save_pretrained(save_directory)
        return result

    # Create a dynamic subclass (e.g. LiteReasonQwen2ForCausalLM) rather than
    # using types.MethodType. DataParallel creates replicas by copying the module,
    # and MethodType bindings would point to the original instance, not the replica.
    PatchedCls = type(
        f"LiteReason{_OrigCls.__name__}",
        (_OrigCls,),
        {
            "forward": forward,
            "shadow_forward": shadow_forward,
            "set_tokenizer": set_tokenizer,
            "init_reasoning_projector_from_pretrained": init_reasoning_projector_from_pretrained,
            "_litereason_base_forward": _litereason_base_forward,
            "save_pretrained": save_pretrained,
        },
    )
    model.__class__ = PatchedCls

    model._litereason_attached = True
    return model


class AutoModelForCausalLMWithReasoning:
    """Factory for a patched HF causal LM with latent reasoning enabled."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        num_reasoning_layers: Optional[int] = None,
        max_reasoning_steps: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

        hub_kwargs = _extract_hub_kwargs(kwargs)
        projector_keys = _detect_checkpoint_projector_keys(
            pretrained_model_name_or_path, hub_kwargs=hub_kwargs
        )

        if num_reasoning_layers is None:
            num_reasoning_layers = getattr(model.config, "litereason_num_reasoning_layers", None)
        if num_reasoning_layers is None:
            num_reasoning_layers = _infer_num_reasoning_layers_from_keys(projector_keys)
        if num_reasoning_layers is None:
            num_reasoning_layers = 3

        if max_reasoning_steps is None:
            max_reasoning_steps = getattr(model.config, "litereason_max_reasoning_steps", None)
        if max_reasoning_steps is None:
            # Base model with no config value: let the env seed the budget (config
            # always wins above, so this never overrides a real checkpoint).
            _env_steps = os.getenv("LITEREASON_MAX_REASONING_STEPS", "").strip()
            if _env_steps.isdigit():
                max_reasoning_steps = int(_env_steps)
        if max_reasoning_steps is None:
            max_reasoning_steps = 32

        attach_litereason_to_causal_lm(
            model,
            tokenizer=tokenizer,
            num_reasoning_layers=int(num_reasoning_layers),
            max_reasoning_steps=int(max_reasoning_steps),
        )

        loaded = _maybe_load_projector_weights(
            model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            hub_kwargs=hub_kwargs,
        )
        if loaded:
            log.info("Loaded reasoning_projector weights from checkpoint")
        else:
            log.info("No reasoning_projector weights in checkpoint, using warm-started projector")
        if projector_keys and not loaded:
            raise RuntimeError(
                "Checkpoint appears to contain reasoning_projector weights, but they were not loaded. "
                "This is unexpected; please report this with your checkpoint layout."
            )
        return model
