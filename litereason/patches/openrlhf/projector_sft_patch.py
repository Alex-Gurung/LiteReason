"""Projector SFT utilities for OpenRLHF integration.

Provides the training loop, data processing, and trace extraction functions
used by ``LiteReasonPolicyModelActor`` (GPU worker) and
``LiteReasonPPOTrainer`` (coordinator) in ``train_ppo_ray.py``.

This is a **utility module**: it does NOT monkeypatch anything.

Set ``LITEREASON_ENABLE_PROJECTOR_SFT=1`` to activate projector SFT.

Configuration (via env vars):
- ``LITEREASON_PROJECTOR_SFT_EPOCHS`` (int, default 1)
- ``LITEREASON_PROJECTOR_SFT_LR`` (float, default 1e-4)
- ``LITEREASON_PROJECTOR_SFT_BATCH_SIZE`` (int, default: OpenRLHF micro_train_batch_size)
- ``LITEREASON_PROJECTOR_SFT_SWAP_RATIOS`` (comma list, default ``0.10,0.25``) the
  equal mixture of sentence-replacement ratios; each trace is emitted once per ratio
- ``LITEREASON_PROJECTOR_SFT_TOKEN_RATIO`` (float, default 0.2) latent tokens per
  original token in a replaced sentence
- ``LITEREASON_PROJECTOR_SFT_DATA_RATIO`` (float, default 0.1)
- ``LITEREASON_PROJECTOR_SFT_REWARD_FILTER`` (bool, default true) keep only
  positive-reward traces (the paper's rejection-sampling SFT); set 0 to train on
  every extracted trace regardless of reward
- ``LITEREASON_PROJECTOR_SFT_MAX_LEN`` (int, default 2048) tokenized length cap
- ``LITEREASON_PROJECTOR_SFT_CLIP_NORM`` (float, default 0.0) grad clipping (0=off)
- ``LITEREASON_PROJECTOR_SFT_BINARY_LABEL_STYLE`` (str, default ``literal``)
  optional answer canonicalization; ``yes_no`` maps 0/1-style labels to
  ``\\boxed{Yes}`` / ``\\boxed{No}``
- ``LITEREASON_PROJECTOR_SFT_DROP_LAST`` (bool, default false) whether RP
  worker dataloaders drop tail batches
- ``LITEREASON_PROJECTOR_SFT_DIAGNOSTICS`` (bool, default false) extra RP diagnostics
- ``LITEREASON_PROJECTOR_SFT_DIAG_SAMPLES`` (int, default 2) number of sample previews to log
"""


import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from litereason.token_utils import ImplicitThoughtDetector

log = logging.getLogger("litereason.patches.openrlhf")


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _env(name: str, default, type_fn=str):
    v = os.getenv(name, "")
    if not v:
        return bool(default) if type_fn is bool else type_fn(default)
    if type_fn is bool:
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return type_fn(v.strip())
    except (ValueError, TypeError):
        log.debug("env var %s unparseable, using default %r", name, default)
        return type_fn(default)


def _parse_float_list(value: str, default) -> tuple:
    """Parse a comma-separated float list (e.g. ``0.10,0.25``); fall back to default."""
    try:
        parsed = tuple(float(x) for x in str(value).split(",") if x.strip())
    except (ValueError, TypeError):
        # Malformed list (non-numeric token / non-string value): use the default.
        parsed = ()
    return parsed or tuple(default)


def _diag_enabled() -> bool:
    return _env("LITEREASON_PROJECTOR_SFT_DIAGNOSTICS", False, bool)


def _diag_sample_limit() -> int:
    return max(0, _env("LITEREASON_PROJECTOR_SFT_DIAG_SAMPLES", 2, int))


def _state_probe_enabled() -> bool:
    return _env("LITEREASON_PROJECTOR_SFT_STATE_PROBE", False, bool)


def _state_probe_max_new_tokens() -> int:
    return max(1, _env("LITEREASON_PROJECTOR_SFT_STATE_PROBE_MAX_NEW_TOKENS", 96, int))


def _state_probe_prompt_limit() -> int:
    return max(1, _env("LITEREASON_PROJECTOR_SFT_STATE_PROBE_PROMPTS", 1, int))


def _bump(stats: Optional[Dict[str, int]], key: str, n: int = 1) -> None:
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + int(n)


def _shorten(text: str, limit: int = 180) -> str:
    text = (text or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _tail(text: str, limit: int = 180) -> str:
    text = (text or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return "..." + text[-max(0, limit - 3):]


def _state_probe_path(trainer: Any) -> Optional[str]:
    override = os.getenv("LITEREASON_PROJECTOR_SFT_STATE_PROBE_PATH", "").strip()
    if override:
        return override
    save_path = getattr(getattr(trainer, "args", None), "save_path", None)
    if not save_path:
        return None
    return os.path.join(save_path, "rp_state_probe.jsonl")


def _append_state_probe_record(trainer: Any, record: Dict[str, Any]) -> None:
    path = _state_probe_path(trainer)
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        # Diagnostic probe write must never crash training: OSError covers the
        # filesystem write, TypeError/ValueError cover non-serializable records.
        log.warning("[RP/state] failed to append probe record to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# spaCy sentence splitting pipeline
# ---------------------------------------------------------------------------

# Tempered pattern: don't cross the closing tag
IMPLICIT_RE = re.compile(
    r"<implicit_thought\b[^>]*>(?:(?!</implicit_thought>).)*</implicit_thought>",
    flags=re.DOTALL,
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", flags=re.IGNORECASE)


def _setup_spacy_pipeline():
    """Build a spaCy NLP pipeline with implicit-thought-aware sentence splitting."""
    # imported lazily so this module imports without the optional spaCy dependency
    import spacy
    from spacy.language import Language
    from spacy.util import filter_spans

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    @Language.component("merge_implicit_thought")
    def merge_implicit_thought(doc):
        # IMPLICIT_RE matches each <implicit_thought>...</implicit_thought>
        # independently, so run it directly on doc.text. char_span() below
        # interprets offsets against doc.text, so we must NOT preclean here
        # (inserting a space would shift every later match by +1 and make
        # char_span land on the wrong tokens / return None).
        matches = list(IMPLICIT_RE.finditer(doc.text))
        if not matches:
            return doc
        spans = []
        for m in matches:
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if span is not None:
                spans.append(span)
        spans = filter_spans(spans)
        with doc.retokenize() as retok:
            for span in sorted(spans, key=lambda s: s.start, reverse=True):
                try:
                    retok.merge(span)
                except ValueError:
                    log.debug("spaCy retokenize merge failed for span")
                    continue
        return doc

    @Language.component("split_around_implicit_thought")
    def split_around_implicit_thought(doc):
        for i, tok in enumerate(doc):
            if tok.text.startswith("<implicit_thought"):
                tok.is_sent_start = True
                if i + 1 < len(doc):
                    doc[i + 1].is_sent_start = True
        return doc

    nlp.add_pipe("merge_implicit_thought", after="sentencizer")
    nlp.add_pipe("split_around_implicit_thought", last=True)
    return nlp


# Lazy-init: only built when projector SFT actually runs.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _setup_spacy_pipeline()
    return _nlp


def _safe_nlp(text: str):
    """Sentence-split a trace with the spaCy pipeline, or return None on error.

    spaCy is a required dependency, but the rule-based pipeline can still raise
    (ValueError/IndexError) on rare malformed-Unicode or retokenize edge cases in
    a rollout trace. Returning None lets the caller skip just that one trace
    instead of crashing the whole projector-SFT step mid-RL. (Retrying on
    marker-stripped text is pointless: the offsets no longer match the original,
    so the result would have to be discarded anyway.)
    """
    try:
        return _get_nlp()(text)
    except (ValueError, IndexError):
        log.warning("spaCy failed on a rollout trace; skipping it for projector SFT")
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasoningProjectorBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class ReasoningTraceExample:
    prompt: str
    answer: str
    trace: str


class ReasoningProjectorDataset(Dataset):
    """Simple dataset wrapper for distributed projector training."""

    def __init__(self, samples: List[ReasoningProjectorBatch]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReasoningProjectorBatch:
        return self.samples[idx]


def collate_reasoning_samples(batch):
    """Collate ReasoningProjectorBatch objects into a single batch.

    Pads variable-length sequences to the max length in the batch so that
    ``torch.cat(dim=0)`` succeeds when ``per_gpu_batch_size > 1``.
    """
    max_len = max(s.input_ids.shape[1] for s in batch)
    padded_ids, padded_mask, padded_labels = [], [], []
    for s in batch:
        pad_len = max_len - s.input_ids.shape[1]
        padded_ids.append(F.pad(s.input_ids, (0, pad_len), value=0))
        padded_mask.append(F.pad(s.attention_mask, (0, pad_len), value=0))
        padded_labels.append(F.pad(s.labels, (0, pad_len), value=-100))
    return ReasoningProjectorBatch(
        input_ids=torch.cat(padded_ids, dim=0),
        attention_mask=torch.cat(padded_mask, dim=0),
        labels=torch.cat(padded_labels, dim=0),
    )


# ---------------------------------------------------------------------------
# Trace extraction and processing (coordinator side)
# ---------------------------------------------------------------------------

def _unwrap_boxed_answer(answer: str) -> tuple[str, bool]:
    answer = (answer or "").strip()
    match = re.fullmatch(r"\\boxed\{([^}]*)\}", answer, flags=re.IGNORECASE)
    if match:
        return (match.group(1).strip(), True)
    return (answer, False)


def _canonical_yes_no_answer(answer: str) -> Optional[str]:
    token = (answer or "").strip().lower()
    if token in {"1", "1.0", "yes", "true"}:
        return "Yes"
    if token in {"0", "0.0", "no", "false"}:
        return "No"

    alpha = re.sub(r"[^a-z]+", "", token)
    if alpha == "yes":
        return "Yes"
    if alpha == "no":
        return "No"
    return None


def _format_boxed_answer_with_style(answer: str, binary_label_style: str) -> str:
    raw_answer = (answer or "").strip()
    inner_answer, was_boxed = _unwrap_boxed_answer(raw_answer)
    style = (binary_label_style or "literal").strip().lower()

    if style in {"yesno", "yes_no", "yes-no", "ff", "flawed_fictions"}:
        canonical = _canonical_yes_no_answer(inner_answer)
        if canonical is not None:
            return f"\\boxed{{{canonical}}}"

    if was_boxed:
        return raw_answer
    return f"\\boxed{{{inner_answer}}}"


def _prompt_token_prefix_len(tokenizer, prompt_text: str, full_input_ids: torch.Tensor) -> Optional[int]:
    try:
        prompt_tok = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
    except Exception as exc:
        # Tokenizers raise heterogeneous errors on edge-case text; this is only a
        # prefix-alignment check, so degrade to "unknown prefix" (None).
        log.debug("prompt_text tokenization failed while checking RP prefix: %s", exc)
        return None

    prompt_ids = prompt_tok.get("input_ids")
    if prompt_ids is None or prompt_ids.dim() != 2:
        return None
    prompt_ids_1d = prompt_ids[0]
    if prompt_ids_1d.numel() > full_input_ids.numel():
        return None
    if not torch.equal(full_input_ids[: prompt_ids_1d.numel()], prompt_ids_1d.to(full_input_ids.device)):
        return None
    return int(prompt_ids_1d.numel())


def _decode_supervised_span(tokenizer, input_ids: torch.Tensor, labels: torch.Tensor) -> str:
    keep = labels != -100
    if keep.dim() != 1 or not bool(keep.any().item()):
        return ""
    ids = input_ids[keep].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False).strip()


def _grad_stats(
    named_params: List[tuple[str, torch.nn.Parameter]],
    param_id_filter: Optional[set[int]] = None,
) -> Dict[str, Any]:
    selected: List[tuple[str, torch.nn.Parameter]] = []
    for name, param in named_params:
        if param_id_filter is not None and id(param) not in param_id_filter:
            continue
        selected.append((name, param))

    with_grad = 0
    nonzero = 0
    total_sq = 0.0
    sample_names: List[str] = []
    for name, param in selected:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        with_grad += 1
        gn = float(grad.detach().float().norm().item())
        total_sq += gn * gn
        if gn > 0.0:
            nonzero += 1
            if len(sample_names) < 5:
                sample_names.append(name)

    return {
        "param_count": len(selected),
        "with_grad": with_grad,
        "nonzero_grad": nonzero,
        "grad_norm": total_sq ** 0.5,
        "sample_nonzero_names": sample_names,
    }


def _allreduce_average_grads(params: List[torch.nn.Parameter], world_size: int) -> None:
    if world_size <= 1 or not dist.is_initialized():
        return
    inv_world_size = 1.0 / float(world_size)
    for p in params:
        grad = getattr(p, "grad", None)
        if grad is None:
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.mul_(inv_world_size)


def _shadow_tie_status(actor, sample_limit: int = 5) -> Dict[str, Any]:
    train_model = getattr(actor, "model", None)
    shadow_model = getattr(actor, "shadow_model", None)
    if train_model is None or shadow_model is None:
        return {"available": False}

    train = train_model.module if hasattr(train_model, "module") else train_model
    t_params = dict(train.named_parameters())
    s_params = dict(shadow_model.named_parameters())
    common = sorted(set(t_params).intersection(s_params))
    if not common:
        return {"available": True, "checked": 0, "ok": True, "mismatches": []}

    mismatches = []
    checked = 0
    for name in common[: max(1, sample_limit)]:
        checked += 1
        if t_params[name].data.data_ptr() != s_params[name].data.data_ptr():
            mismatches.append(name)
    return {
        "available": True,
        "checked": checked,
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def _assistant_turn_suffix_text(tokenizer) -> str:
    """Return the template suffix after assistant content, e.g. `<end_of_turn>\n`."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return ""

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
        # Chat-template rendering can fail for tokenizers without an assistant
        # role / Jinja template; no usable suffix, so emit none.
        return ""

    idx = rendered.rfind(assistant_sentinel)
    if idx < 0:
        return ""
    return rendered[idx + len(assistant_sentinel):]


def _find_marker_mask_boundary(detector: ImplicitThoughtDetector, input_ids: torch.Tensor) -> int:
    markers = detector.find_thought_markers(input_ids.unsqueeze(0))
    if not markers:
        return 0
    start_idx, num_gap_tokens, _complexity, start_len, end_len = markers[0]
    return start_idx + start_len + num_gap_tokens + end_len


def _token_id_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        return [int(token_id) for token_id in value if token_id is not None]
    return []


def _resolve_sequence_end_ids(tokenizer, model_name_or_path: Optional[str] = None) -> List[int]:
    """Return all known turn/sequence terminators for response-boundary inference."""
    generation_eos = None
    if model_name_or_path:
        try:
            # imported here to keep transformers out of this module's import path
            from transformers import GenerationConfig

            generation_config = GenerationConfig.from_pretrained(model_name_or_path)
            generation_eos = getattr(generation_config, "eos_token_id", None)
        except Exception as exc:
            # Best-effort: a missing/unparseable generation_config (OSError, HF
            # parse errors) just falls back to the tokenizer's eos below.
            log.debug("Could not load generation config from %s: %s", model_name_or_path, exc)

    merged: List[int] = []
    seen = set()
    for token_id in _token_id_list(generation_eos) + _token_id_list(getattr(tokenizer, "eos_token_id", None)):
        if token_id not in seen:
            merged.append(token_id)
            seen.add(token_id)
    return merged


def _collect_state_probe_cases(experiences: List[Any], limit: int) -> List[Dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    cases: List[Dict[str, str]] = []
    for exp in experiences:
        prompts = list(getattr(exp, "prompts", []) or [])
        labels = list(getattr(exp, "labels", []) or [])
        for prompt, label in zip(prompts, labels, strict=False):
            prompt_text = (prompt or "").strip()
            label_text = (label or "").strip()
            if not prompt_text:
                continue
            key = (prompt_text, label_text)
            if key in seen:
                continue
            seen.add(key)
            cases.append({"prompt": prompt_text, "label": label_text})
            if len(cases) >= max(1, limit * 8):
                break
        if len(cases) >= max(1, limit * 8):
            break

    cases.sort(key=lambda item: len(item["prompt"]))
    return cases[:limit]


def _run_state_probe(
    trainer: Any,
    episode: int,
    stage: str,
    probe_cases: List[Dict[str, str]],
    include_vllm: bool,
) -> None:
    # imported lazily so this module imports without the optional ray dependency
    import ray

    if not _state_probe_enabled() or not probe_cases:
        return

    actor0 = getattr(trainer.actor_model_group, "_actor_handlers", [None])[0]
    if actor0 is None:
        return

    actor_probe = ray.get(
        actor0.debug_state_probe.remote(
            [case["prompt"] for case in probe_cases],
            max_new_tokens=_state_probe_max_new_tokens(),
        )
    )

    vllm_probe = None
    if include_vllm and getattr(trainer, "vllm_engines", None):
        # imported lazily so this module imports without the optional vllm dependency
        from vllm import SamplingParams

        sequence_end_ids = _resolve_sequence_end_ids(
            trainer.tokenizer,
            getattr(trainer.args, "pretrain", None),
        )
        probe_case = probe_cases[0]
        max_tokens = _state_probe_max_new_tokens()
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            min_tokens=1,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop_token_ids=sequence_end_ids or None,
        )
        ray.get(
            [
                engine.wake_up.remote(tags=["weights", "kv_cache"])
                for engine in trainer.vllm_engines
            ]
        )
        refs = [
            engine.generate_responses.remote(
                prompt=probe_case["prompt"],
                label=probe_case["label"],
                sampling_params=sampling_params,
                max_length=getattr(trainer.args, "max_len", 4096),
                hf_tokenizer=trainer.tokenizer,
                num_samples=1,
            )
            for engine in trainer.vllm_engines
        ]
        raw_outputs = ray.get(refs)
        prompt_ids = trainer.tokenizer(
            probe_case["prompt"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        vllm_probe = []
        for engine_idx, engine_outputs in enumerate(raw_outputs):
            payload = engine_outputs[0] if engine_outputs else {}
            observation_tokens = list(payload.get("observation_tokens") or [])
            response_tokens = observation_tokens[len(prompt_ids) :]
            vllm_probe.append(
                {
                    "engine_idx": engine_idx,
                    "truncated": bool(payload.get("truncated", False)),
                    "new_token_count": len(response_tokens),
                    "response_text": trainer.tokenizer.decode(
                        response_tokens,
                        skip_special_tokens=False,
                    ),
                }
            )

    actor_generations = actor_probe.get("generations", [])
    record = {
        "episode": episode,
        "stage": stage,
        "probe_cases": [
            {
                "label": case["label"],
                "prompt_tail": _tail(case["prompt"], 220),
            }
            for case in probe_cases
        ],
        "actor_rank": actor_probe.get("rank"),
        "actor_model_type": actor_probe.get("model_type"),
        "actor_stop_token_ids": actor_probe.get("stop_token_ids"),
        "actor_base_digest": actor_probe.get("base_digest"),
        "actor_projector_digest": actor_probe.get("projector_digest"),
        "actor_generations": [
            {
                "new_token_count": int(item.get("new_token_count", 0)),
                "response_text": _shorten(item.get("response_text", ""), 320),
            }
            for item in actor_generations
        ],
        "vllm_generations": [
            {
                "engine_idx": item["engine_idx"],
                "truncated": item["truncated"],
                "new_token_count": item["new_token_count"],
                "response_text": _shorten(item["response_text"], 320),
            }
            for item in (vllm_probe or [])
        ],
    }
    log.info("[RP/state] %s", json.dumps(record, ensure_ascii=False, sort_keys=True))
    _append_state_probe_record(trainer, record)


def _find_model_response(
    sequence: torch.Tensor,
    tokenizer,
    prompt_len: Optional[int] = None,
    sequence_end_ids: Optional[List[int]] = None,
) -> str:
    """Find the model response text from a full sequence.

    Preferred boundary:
    - ``prompt_len`` when available from PPO metadata (model-agnostic).

    Fallback boundary:
    - Token after last EOS in the sequence.
    """
    ids = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
    if not ids:
        return ""

    start_idx = 0
    if prompt_len is not None:
        try:
            p = int(prompt_len)
            if 0 <= p <= len(ids):
                start_idx = p
        except (TypeError, ValueError):
            log.debug("prompt_len parse failed in _find_model_response")
    else:
        end_ids = sequence_end_ids or _token_id_list(getattr(tokenizer, "eos_token_id", None))
        if end_ids:
            end_id_set = set(end_ids)
            for i in range(len(ids) - 1, -1, -1):
                if ids[i] in end_id_set:
                    start_idx = i + 1
                    break

    return tokenizer.decode(ids[start_idx:], skip_special_tokens=True).strip()


def _infer_prompt_len_from_experience(exp: Any, seq_idx: int) -> Optional[int]:
    """Infer prompt length for a sequence from PPO Experience metadata."""
    sequences = getattr(exp, "sequences", None)
    action_mask = getattr(exp, "action_mask", None)

    if isinstance(sequences, torch.Tensor) and isinstance(action_mask, torch.Tensor):
        if (
            sequences.dim() == 2
            and action_mask.dim() == 2
            and seq_idx < sequences.shape[0]
            and seq_idx < action_mask.shape[0]
        ):
            total_len = int(sequences.shape[1])
            action_len = int(action_mask[seq_idx].long().sum().item())
            prompt_len = total_len - action_len
            if 0 <= prompt_len <= total_len:
                return prompt_len

    info = getattr(exp, "info", None)
    if isinstance(info, dict):
        for key in ("prompt_length", "prompt_len", "input_length"):
            value = info.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 1 and seq_idx < value.shape[0]:
                prompt_len = int(value[seq_idx].item())
                if prompt_len >= 0:
                    return prompt_len
            if isinstance(value, (list, tuple)) and seq_idx < len(value):
                try:
                    prompt_len = int(value[seq_idx])
                except (TypeError, ValueError):
                    continue
                if prompt_len >= 0:
                    return prompt_len

    return None


def _infer_prompt_len_from_prompt_text(prompt_text: Optional[str], tokenizer) -> Optional[int]:
    if not prompt_text:
        return None
    try:
        tokenized = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
    except Exception as exc:
        # Tokenizers raise heterogeneous errors on edge-case text; prompt-length
        # inference is best-effort, so degrade to None.
        log.debug("prompt text tokenization failed in projector SFT: %s", exc)
        return None
    input_ids = tokenized.get("input_ids")
    if input_ids is None or input_ids.dim() != 2:
        return None
    return int(input_ids.shape[1])


def _reward_for_seq(rewards: Any, seq_idx: int) -> Optional[float]:
    """Scalar task reward for a sequence index, or None if unavailable."""
    if rewards is None:
        return None
    try:
        v = rewards[seq_idx]
    except (IndexError, KeyError, TypeError):
        return None
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except (ValueError, RuntimeError):
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_reasoning_traces_from_experiences(
    experiences: list,
    tokenizer,
    sequence_end_ids: Optional[List[int]] = None,
    stats: Optional[Dict[str, int]] = None,
    positive_reward_only: bool = True,
) -> List[ReasoningTraceExample]:
    """Extract prompt/answer/reasoning examples from PPO Experience objects.

    The prompt field already contains the chat-templated generation prompt
    from OpenRLHF's PromptDataset, so downstream projector SFT can append
    assistant text and the correct template-specific turn suffix directly.

    When ``positive_reward_only`` is True (the paper's rejection-sampling SFT),
    only sequences whose task reward (``exp.info["reward"]``) is strictly
    positive are kept, so the projector learns from successful reasoning only.
    A sequence whose reward cannot be read is kept (counted as
    ``reward_unavailable``) rather than silently dropped.
    """
    traces: List[ReasoningTraceExample] = []
    diag = _diag_enabled()
    diag_budget = _diag_sample_limit()
    diag_logged = 0
    for exp in experiences:
        if not hasattr(exp, "sequences") or exp.sequences is None:
            _bump(stats, "dropped_missing_sequences")
            continue
        prompts = list(getattr(exp, "prompts", []) or [])
        labels = list(getattr(exp, "labels", []) or [])
        info = getattr(exp, "info", None)
        rewards = info.get("reward") if isinstance(info, dict) else None
        for seq_idx in range(exp.sequences.shape[0]):
            if positive_reward_only:
                reward = _reward_for_seq(rewards, seq_idx)
                if reward is None:
                    _bump(stats, "reward_unavailable")
                elif reward <= 0.0:
                    _bump(stats, "dropped_nonpositive_reward")
                    continue
            prompt_len = _infer_prompt_len_from_experience(exp, seq_idx)
            if prompt_len is None:
                prompt_len = _infer_prompt_len_from_prompt_text(
                    prompts[seq_idx] if seq_idx < len(prompts) else None,
                    tokenizer,
                )
            text = _find_model_response(
                exp.sequences[seq_idx],
                tokenizer,
                prompt_len=prompt_len,
                sequence_end_ids=sequence_end_ids,
            )
            matches = list(_BOXED_RE.finditer(text or ""))
            if not matches:
                _bump(stats, "dropped_no_boxed_answer")
                continue
            last_match = matches[-1]
            model_boxed_answer = last_match.group(1).strip() or None
            prefix = text[: last_match.start()].strip()
            if not prefix:
                _bump(stats, "dropped_empty_reasoning_prefix")
                continue

            prompt_text = (prompts[seq_idx] if seq_idx < len(prompts) else None) or ""
            answer_text = (labels[seq_idx] if seq_idx < len(labels) else None) or ""
            prompt_text = prompt_text.strip()
            answer_text = answer_text.strip()
            if not prompt_text:
                _bump(stats, "dropped_empty_prompt")
                continue
            if not answer_text:
                _bump(stats, "dropped_empty_answer_label")
                continue

            example = ReasoningTraceExample(
                prompt=prompt_text,
                answer=answer_text,
                trace=prefix,
            )
            traces.append(example)
            _bump(stats, "extracted_traces")
            if diag and diag_logged < diag_budget:
                diag_logged += 1
                log.info(
                    "[RP/diag] extracted trace #%d: prompt_len=%s model_boxed=%r label_answer=%r "
                    "response=%s trace=%s",
                    diag_logged,
                    prompt_len,
                    model_boxed_answer,
                    answer_text,
                    _shorten(text),
                    _shorten(prefix),
                )
    return traces


def process_reasoning_for_training(
    traces: List[ReasoningTraceExample],
    tokenizer,
    swap_ratio: float,
    max_len: int,
    token_ratio: float = 0.2,
    max_depth: int = 5,
    rng: Optional[random.Random] = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[ReasoningProjectorBatch]:
    """Convert reasoning traces to tokenized training samples with marker injection.

    Uses spaCy for sentence splitting, preserves original whitespace via char spans.
    """
    if rng is None:
        rng = random

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    samples: List[ReasoningProjectorBatch] = []
    detector = ImplicitThoughtDetector(tokenizer)
    assistant_suffix = _assistant_turn_suffix_text(tokenizer)
    diag = _diag_enabled()
    diag_budget = _diag_sample_limit()
    diag_logged = 0
    binary_label_style = _env("LITEREASON_PROJECTOR_SFT_BINARY_LABEL_STYLE", "literal", str)

    for trace_example in traces:
        prompt_text = trace_example.prompt
        boxed_answer = _format_boxed_answer_with_style(
            trace_example.answer,
            binary_label_style=binary_label_style,
        )
        trace = trace_example.trace

        # _safe_nlp returns None if spaCy raised on this trace; skip just that one.
        doc = _safe_nlp(trace)
        if doc is None:
            _bump(stats, "dropped_nlp_error")
            continue
        sents = list(doc.sents)
        if not sents:
            _bump(stats, "dropped_no_sentences")
            continue

        # Classify sentences
        non_special = []
        special_count = 0
        for i, s in enumerate(sents):
            txt = s.text
            if "<implicit_thought>" in txt and "</implicit_thought>" in txt:
                special_count += 1
            else:
                non_special.append((i, s))

        # Decide path based on ratio
        if swap_ratio <= 0.0:
            if special_count == 0:
                _bump(stats, "dropped_ratio0_without_existing_markers")
                continue
            modified_text = trace
        else:
            if not non_special:
                _bump(stats, "dropped_no_swappable_sentences")
                continue
            # At least one marker per trace (matches standalone SFT prep) so short
            # traces still produce a periodic-SFT sample.
            num_to_swap = max(1, int(len(non_special) * swap_ratio))
            chosen = {
                idx for idx, _ in rng.sample(non_special, k=min(num_to_swap, len(non_special)))
            }

            # Build replacement via char spans to preserve whitespace/newlines
            out_chunks = []
            cursor = 0
            for i, s in enumerate(sents):
                start, end = s.start_char, s.end_char
                if cursor < start:
                    out_chunks.append(trace[cursor:start])
                if i in chosen:
                    n_tokens = max(1, len(tokenizer.encode(s.text, add_special_tokens=False)))
                    depth = int(n_tokens * token_ratio)
                    depth = min(max_depth, max(1, depth))
                    out_chunks.append(f"<implicit_thought>{depth}</implicit_thought>")
                else:
                    out_chunks.append(trace[start:end])
                cursor = end
            if cursor < len(trace):
                out_chunks.append(trace[cursor:])
            modified_text = "".join(out_chunks)

        # Tokenize
        assistant_text = f"{modified_text.strip()}\n{boxed_answer}".strip()
        full_text = f"{prompt_text}{assistant_text}{assistant_suffix}"
        tokenized = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        if tokenized["input_ids"].shape[1] > max_len:
            _bump(stats, "dropped_too_long")
            continue

        labels = tokenized["input_ids"].clone()
        marker_end_pos = _find_marker_mask_boundary(detector, tokenized["input_ids"][0])
        if marker_end_pos <= 0:
            _bump(stats, "dropped_no_marker_boundary")
            continue
        labels[:, :marker_end_pos] = -100

        assistant_start = _prompt_token_prefix_len(
            tokenizer,
            prompt_text,
            tokenized["input_ids"][0],
        )
        if assistant_start is None:
            _bump(stats, "prompt_prefix_token_mismatch")
        else:
            supervised_before_assistant = int((labels[0, :assistant_start] != -100).sum().item())
            if supervised_before_assistant > 0:
                # A marker boundary landed inside the prompt region, so prompt
                # tokens would become supervised targets. Drop the sample rather
                # than training the projector on prompt tokens.
                _bump(stats, "supervised_non_assistant_tokens", supervised_before_assistant)
                _bump(stats, "dropped_supervised_prompt")
                continue

        samples.append(
            ReasoningProjectorBatch(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=labels,
            )
        )
        _bump(stats, "created_training_samples")

        if diag and diag_logged < diag_budget:
            diag_logged += 1
            supervised_count = int((labels[0] != -100).sum().item())
            supervised_text = _decode_supervised_span(
                tokenizer,
                tokenized["input_ids"][0],
                labels[0],
            )
            log.info(
                "[RP/diag] sample #%d build: label_answer=%r boxed_answer=%r total_tokens=%d "
                "prompt_tail=%s assistant=%s assistant_suffix=%r",
                diag_logged,
                trace_example.answer,
                boxed_answer,
                int(tokenized["input_ids"].shape[1]),
                _tail(prompt_text),
                _shorten(assistant_text),
                assistant_suffix,
            )
            log.info(
                "[RP/diag] sample #%d: prompt_prefix=%s marker_end=%d supervised_tokens=%d "
                "assistant_start=%s supervised=%s",
                diag_logged,
                "ok" if assistant_start is not None else "mismatch",
                marker_end_pos,
                supervised_count,
                assistant_start,
                _shorten(supervised_text),
            )

    return samples


# ---------------------------------------------------------------------------
# GPU worker side: projector training loop
# ---------------------------------------------------------------------------

def train_reasoning_projector_distributed(
    trainer: Any,
    dataset: ReasoningProjectorDataset,
    per_gpu_batch_size: int,
    epochs: int,
    learning_rate: float,
) -> Dict[str, float]:
    """DeepSpeed-safe projector SFT with per-batch forward.

    Called from LiteReasonPolicyModelActor.fit_reasoning_projector.
    trainer is the ActorPPOTrainer instance on the GPU worker.
    """
    empty = {
        "loss": 0.0,
        "learning_rate": float(learning_rate),
        "total_steps": 0,
        "epochs": int(epochs),
        "samples_processed": 0,
        "skipped_nonfinite_steps": 0,
    }

    # Find projector parameters
    model_engine = trainer.actor.model
    engine_mod = model_engine.module if hasattr(model_engine, "module") else model_engine
    proj_mod = getattr(model_engine.model, "reasoning_projector", None)
    if proj_mod is None:
        proj_mod = getattr(engine_mod, "reasoning_projector", None)
    if proj_mod is None:
        log.warning("[RP] No reasoning_projector module. Skipping training.")
        return empty

    proj_params = list(proj_mod.parameters())
    if not proj_params:
        log.warning("[RP] Projector has 0 parameters. Skipping training.")
        return empty
    proj_param_ids = {id(p) for p in proj_params}

    if len(dataset) == 0:
        return empty

    # Rank/world
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    is_rank0 = rank == 0
    drop_last = _env("LITEREASON_PROJECTOR_SFT_DROP_LAST", False, bool)

    device = torch.cuda.current_device()
    model_engine.train()
    engine_mod.train()
    diag = _diag_enabled()
    strict_base_grad = _env("LITEREASON_PROJECTOR_SFT_STRICT_BASE_GRAD", False, bool)
    named_params = list(engine_mod.named_parameters())
    base_param_ids = {id(p) for _, p in named_params if id(p) not in proj_param_ids}

    # Snapshot and freeze: projector-only trainable
    rg_mask = {p: p.requires_grad for p in engine_mod.parameters()}
    for p in engine_mod.parameters():
        p.requires_grad = False
    for p in proj_params:
        p.requires_grad = True

    # Invariant: only the projector is trainable. We assert this here, where
    # requires_grad is actually set, because the previous .grad-based guard in
    # the training loop was dead (torch.autograd.grad never populates .grad on
    # base params, and they were zeroed/frozen, so it could never fire).
    if strict_base_grad:
        bad_base = [
            name
            for name, p in named_params
            if id(p) not in proj_param_ids and p.requires_grad
        ]
        bad_proj = [name for name, p in named_params if id(p) in proj_param_ids and not p.requires_grad]
        if bad_base or bad_proj:
            raise RuntimeError(
                "Projector-only trainability invariant violated: "
                f"non-projector params with requires_grad=True: {bad_base[:8]}; "
                f"projector params with requires_grad=False: {bad_proj[:8]}"
            )

    # DataLoader with DistributedSampler
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last)
        if world_size > 1
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(per_gpu_batch_size),
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        num_workers=2,
        drop_last=drop_last,
        collate_fn=collate_reasoning_samples,
    )

    # Dedicated projector-only optimizer. After each projector step we refresh
    # ZeRO's fp32 master weights from the updated bf16 copies so later PPO
    # steps do not overwrite the projector change.
    try:
        base_lr = float(trainer.actor_optim.param_groups[0]["lr"])
    except (AttributeError, KeyError, IndexError, TypeError):
        # base_lr is diagnostic-only (logging + returned metrics); fall back if
        # the optimizer / its param groups aren't shaped as expected.
        base_lr = float(learning_rate)
    rp_lr = float(learning_rate)
    rp_optimizer = torch.optim.AdamW(
        proj_params,
        lr=rp_lr,
        betas=tuple(getattr(trainer.strategy.args, "adam_betas", (0.9, 0.95))),
        weight_decay=float(getattr(trainer.args, "l2", 0.0) or 0.0),
    )
    rp_grad_scale = 1.0
    clip_norm = _env("LITEREASON_PROJECTOR_SFT_CLIP_NORM", 0.0, float)
    log_every = 10
    lr_before_first_step: Optional[float] = None
    lr_after_last_step: Optional[float] = None

    def _clear_rp_grads() -> None:
        for target in (
            getattr(model_engine, "optimizer", None),
            getattr(trainer, "actor_optim", None),
            model_engine,
            engine_mod,
        ):
            if target is None:
                continue
            try:
                target.zero_grad(set_to_none=True)
            except TypeError:
                # Older optimizers/engines lack the set_to_none kwarg.
                target.zero_grad()

    if hasattr(engine_mod, "model") and hasattr(engine_mod.model, "_litereason_max_reasoning_steps"):
        max_reasoning_steps = int(engine_mod.model._litereason_max_reasoning_steps)
    else:
        max_reasoning_steps = getattr(engine_mod, "_litereason_max_reasoning_steps", None)
        if max_reasoning_steps is not None:
            max_reasoning_steps = int(max_reasoning_steps)
    tie_before = _shadow_tie_status(trainer.actor)

    if is_rank0:
        log.info(
            f"[RP] rank={rank} epochs={epochs} steps/epoch={len(dataloader)} "
            f"rp_lr={rp_lr:.2e} base_lr={base_lr:.2e} grad_scale={rp_grad_scale:.3g} "
            f"clip_norm={clip_norm if clip_norm > 0 else 'off'} per_gpu_bs={per_gpu_batch_size} "
            f"drop_last={drop_last} proj_params={sum(p.numel() for p in proj_params):,d}"
        )
        if diag:
            log.info(
                "[RP/diag] max_reasoning_steps=%s shadow_tie_before=%s",
                max_reasoning_steps,
                tie_before,
            )

    # Training loop
    total_loss = 0.0
    total_steps = 0
    skipped_nonfinite_steps = 0
    for epoch in range(int(epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0

        for bidx, batch in enumerate(dataloader):
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            _clear_rp_grads()

            # Per-batch forward through the full model (model handles marker expansion)
            outputs = engine_mod(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss

            # If projector path inactive, anchor a zero loss so the step is well-defined.
            if (not torch.is_tensor(loss)) or (not loss.requires_grad) or (loss.numel() == 0):
                dummy_dtype = loss.dtype if torch.is_tensor(loss) else torch.float32
                dummy = torch.zeros([], device=device, dtype=dummy_dtype)
                for p in proj_params:
                    dummy = dummy + 0.0 * p.float().sum()
                loss = dummy
                if is_rank0 and diag and (bidx % log_every == 0):
                    log.info(
                        "[RP/diag] epoch %d step %d/%d: projector path inactive, anchoring zero loss",
                        epoch,
                        bidx,
                        len(dataloader),
                    )
            assert loss.requires_grad
            has_nonfinite_loss = not bool(torch.isfinite(loss).all().item())
            if torch.distributed.is_initialized() and world_size > 1:
                bad = torch.tensor(
                    1 if has_nonfinite_loss else 0,
                    device=device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
                has_nonfinite_loss = bool(bad.item())
            if has_nonfinite_loss:
                skipped_nonfinite_steps += 1
                if is_rank0:
                    log.warning(
                        "[RP] non-finite loss at epoch=%d step=%d; skipping optimizer step",
                        epoch,
                        bidx,
                    )
                _clear_rp_grads()
                continue

            # Compute grads explicitly for projector params only. This keeps
            # the base model out of the RP graph while preserving the full
            # dependency chain from reasoning embeddings back into projector
            # weights.
            proj_grads = torch.autograd.grad(
                loss,
                proj_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for p, grad in zip(proj_params, proj_grads, strict=True):
                p.grad = None if grad is None else grad.detach()
            # NB: there is no per-step base-gradient guard here. torch.autograd.grad
            # only returns grads for proj_params and never writes .grad on any
            # parameter, so a .grad-based base check would always see None. The
            # real invariant ("only the projector is trainable") is asserted once
            # at setup via the requires_grad check above.
            _allreduce_average_grads(proj_params, world_size)
            proj_grad_stats_after_scale = _grad_stats(named_params, param_id_filter=proj_param_ids)

            has_nonfinite_grad = False
            for p in proj_params:
                grad = p.grad
                if grad is not None and (not torch.isfinite(grad).all()):
                    has_nonfinite_grad = True
                    break
            if torch.distributed.is_initialized() and world_size > 1:
                bad = torch.tensor(
                    1 if has_nonfinite_grad else 0,
                    device=device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
                has_nonfinite_grad = bool(bad.item())
            if has_nonfinite_grad:
                skipped_nonfinite_steps += 1
                if is_rank0:
                    log.warning(
                        "[RP] non-finite gradients at epoch=%d step=%d; skipping optimizer step",
                        epoch,
                        bidx,
                    )
                _clear_rp_grads()
                continue

            # Optional grad clipping
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    proj_params,
                    max_norm=clip_norm,
                    error_if_nonfinite=False,
                )

            # Projector-only step. Afterwards refresh ZeRO's fp32 masters from
            # the updated bf16 weights so future PPO steps see the same state.
            lr_before_step = float(rp_optimizer.param_groups[0]["lr"])
            grad_norm = proj_grad_stats_after_scale["grad_norm"]
            if lr_before_first_step is None:
                lr_before_first_step = lr_before_step
            rp_optimizer.step()
            ds_optim = getattr(model_engine, "optimizer", None)
            if ds_optim is not None and hasattr(ds_optim, "refresh_fp32_params"):
                ds_optim.refresh_fp32_params()
            if hasattr(trainer.actor, "ensure_still_tied"):
                trainer.actor.ensure_still_tied(every_n=50, deep_every=1000)
            lr_after_step = float(rp_optimizer.param_groups[0]["lr"])
            lr_after_last_step = lr_after_step

            # Clear grads
            _clear_rp_grads()

            lval = float(loss.item())
            epoch_loss += lval
            epoch_steps += 1
            total_loss += lval
            total_steps += 1

            if is_rank0 and (bidx % log_every == 0):
                if diag:
                    # Diagnostic-only base grad stats (not a correctness guard):
                    # base params are frozen and never receive .grad here, so this
                    # is expected to report all zeros.
                    base_grad_stats = _grad_stats(named_params, param_id_filter=base_param_ids)
                    tie_after = _shadow_tie_status(trainer.actor)
                    log.info(
                        "[RP] epoch %d step %d/%d: loss=%.4f grad_scale=%.3g lr_eff=%.2e "
                        "lr_before=%.3e lr_after=%.3e proj_grad_norm=%.3e "
                        "proj_with_grad=%d/%d proj_nonzero=%d base_with_grad=%d/%d "
                        "base_nonzero=%d base_samples=%s shadow_tie=%s",
                        epoch,
                        bidx,
                        len(dataloader),
                        lval,
                        rp_grad_scale,
                        rp_lr,
                        lr_before_step,
                        lr_after_step,
                        grad_norm,
                        proj_grad_stats_after_scale["with_grad"],
                        proj_grad_stats_after_scale["param_count"],
                        proj_grad_stats_after_scale["nonzero_grad"],
                        base_grad_stats["with_grad"],
                        base_grad_stats["param_count"],
                        base_grad_stats["nonzero_grad"],
                        base_grad_stats["sample_nonzero_names"],
                        tie_after,
                    )
                else:
                    log.info(
                        f"[RP] epoch {epoch} step {bidx}/{len(dataloader)}: "
                        f"loss={lval:.4f} grad_scale={rp_grad_scale:.3g} lr_eff={rp_lr:.2e}"
                    )

        # Per-epoch average
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        if torch.distributed.is_initialized() and world_size > 1:
            t = torch.tensor(avg_epoch_loss, device=device)
            torch.distributed.all_reduce(t)
            avg_epoch_loss = t.item() / world_size

        if is_rank0:
            log.info(f"[RP] epoch {epoch}: avg_loss={avg_epoch_loss:.4f} steps={epoch_steps}")

    # Restore requires_grad
    for p, rg in rg_mask.items():
        p.requires_grad = rg
    if hasattr(trainer.actor, "ensure_still_tied"):
        trainer.actor.ensure_still_tied(every_n=0)
    tie_after_restore = _shadow_tie_status(trainer.actor)
    if is_rank0 and diag:
        log.info("[RP/diag] shadow_tie_after_restore=%s", tie_after_restore)

    # Final averaged loss across ranks
    avg_total_loss = total_loss / max(1, total_steps)
    if torch.distributed.is_initialized() and world_size > 1:
        t = torch.tensor(avg_total_loss, device=device)
        torch.distributed.all_reduce(t)
        avg_total_loss = t.item() / world_size

    return {
        "loss": float(avg_total_loss),
        "learning_rate": float(rp_lr),
        "total_steps": int(total_steps),
        "epochs": int(epochs),
        "samples_processed": int(len(dataset)),
        "skipped_nonfinite_steps": int(skipped_nonfinite_steps),
        "gpu_memory_allocated": torch.cuda.memory_allocated(device) / 1024**3,
        "gpu_memory_reserved": torch.cuda.memory_reserved(device) / 1024**3,
        "grad_scale": float(rp_grad_scale),
        "lr_before_first_step": float(lr_before_first_step or base_lr),
        "lr_after_last_step": float(lr_after_last_step or base_lr),
        "max_reasoning_steps": int(max_reasoning_steps) if max_reasoning_steps is not None else -1,
    }


# ---------------------------------------------------------------------------
# Coordinator side: end-of-episode projector SFT orchestration
# ---------------------------------------------------------------------------


def _sleep_unused_components(trainer: Any) -> None:
    # imported lazily so this module imports without ray / openrlhf installed
    import ray
    from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

    sleep_refs = []

    if trainer.vllm_engines and getattr(trainer.args, "vllm_enable_sleep", False):
        log.info("[LiteReason/RP] Sleeping vLLM engines to free GPU memory")
        batch_vllm_engine_call(trainer.vllm_engines, "sleep")

    if getattr(trainer.args, "deepspeed_enable_sleep", False):
        if trainer.critic_model_group is not None:
            log.info("[LiteReason/RP] Offloading critic model states to CPU")
            sleep_refs.append(trainer.critic_model_group.async_run_method(method_name="offload_states"))
        if trainer.reward_model_group is not None:
            log.info("[LiteReason/RP] Offloading reward model states to CPU")
            sleep_refs.append(trainer.reward_model_group.async_run_method(method_name="offload_states"))

    if sleep_refs:
        ray.get(sleep_refs)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _wake_unused_components(trainer: Any) -> None:
    # imported lazily so this module imports without ray / openrlhf installed
    import ray
    from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

    wake_refs = []

    if getattr(trainer.args, "deepspeed_enable_sleep", False):
        if trainer.critic_model_group is not None:
            log.info("[LiteReason/RP] Reloading critic model states from CPU")
            wake_refs.append(trainer.critic_model_group.async_run_method(method_name="reload_states"))
        if trainer.reward_model_group is not None:
            log.info("[LiteReason/RP] Reloading reward model states from CPU")
            wake_refs.append(trainer.reward_model_group.async_run_method(method_name="reload_states"))

    if wake_refs:
        ray.get(wake_refs)

    if trainer.vllm_engines and getattr(trainer.args, "vllm_enable_sleep", False):
        log.info("[LiteReason/RP] Waking vLLM engines")
        batch_vllm_engine_call(trainer.vllm_engines, "wake_up")


def _projector_per_gpu_batch_size(trainer: Any) -> int:
    raw = os.getenv("LITEREASON_PROJECTOR_SFT_BATCH_SIZE", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            log.warning(
                "[LiteReason/RP] invalid LITEREASON_PROJECTOR_SFT_BATCH_SIZE=%r; "
                "falling back to micro_train_batch_size",
                raw,
            )
    return max(1, int(getattr(trainer.args, "micro_train_batch_size", 1)))


def train_projector_after_episode(trainer: Any, episode: int) -> None:
    """Run projector SFT at the end of an episode.

    Called from ``LiteReasonPPOTrainer.fit()``.
    trainer is the PPOTrainer instance (coordinator process).
    """
    # imported lazily so this module imports without the optional ray dependency
    import ray

    if not _env("LITEREASON_ENABLE_PROJECTOR_SFT", False, bool):
        log.info(
            "[LiteReason/RP] Episode %d: projector SFT disabled "
            "(LITEREASON_ENABLE_PROJECTOR_SFT != 1)",
            episode,
        )
        return

    experiences = getattr(trainer, "_litereason_rp_experiences", [])
    if not experiences:
        log.warning(
            "[LiteReason/RP] Episode %d: no experiences collected, skipping projector SFT",
            episode,
        )
        return

    probe_cases = _collect_state_probe_cases(experiences, _state_probe_prompt_limit())
    _run_state_probe(
        trainer,
        episode=episode,
        stage="pre_rp",
        probe_cases=probe_cases,
        include_vllm=True,
    )

    _sleep_unused_components(trainer)
    try:
        if getattr(trainer.args, "deepspeed_enable_sleep", False):
            log.info("[LiteReason/RP] Reloading actor model states from CPU")
            ray.get(trainer.actor_model_group.async_run_method(method_name="reload_states"))

        # Step 1: Extract reasoning traces from experiences
        log.info(
            "[LiteReason/RP] Episode %d: extracting reasoning traces from %d experience batches",
            episode,
            len(experiences),
        )
        sequence_end_ids = _resolve_sequence_end_ids(
            trainer.tokenizer,
            getattr(trainer.args, "pretrain", None),
        )
        if _diag_enabled():
            log.info("[LiteReason/RP] Episode %d: sequence_end_ids=%s", episode, sequence_end_ids)
        # Rejection sampling (paper default): keep only positive-reward traces so the
        # projector learns from successful reasoning. Disable to train on all traces.
        positive_reward_only = _env("LITEREASON_PROJECTOR_SFT_REWARD_FILTER", True, bool)
        trace_stats: Dict[str, int] = {}
        traces = extract_reasoning_traces_from_experiences(
            experiences,
            trainer.tokenizer,
            sequence_end_ids=sequence_end_ids,
            stats=trace_stats,
            positive_reward_only=positive_reward_only,
        )
        if trace_stats:
            log.info(
                "[LiteReason/RP] Episode %d: extract stats %s (reward_filter=%s)",
                episode,
                trace_stats,
                positive_reward_only,
            )

        if not traces:
            log.warning(
                "[LiteReason/RP] Episode %d: %d experiences but 0 reasoning traces "
                "(no \\boxed{} prefixes found), skipping projector SFT",
                episode,
                len(experiences),
            )
            return

        log.info("[LiteReason/RP] Episode %d: found %d reasoning traces", episode, len(traces))

        # Step 2: Subsample traces based on data_ratio
        data_ratio = _env("LITEREASON_PROJECTOR_SFT_DATA_RATIO", 0.1, float)
        # data_ratio is a fraction in [0, 1]; a negative value is the opt-out
        # sentinel that disables subsampling and keeps every extracted trace.
        if data_ratio >= 0.0:
            # Keep at least one trace so a small episode still updates the
            # projector instead of silently skipping periodic SFT.
            max_samples = max(1, int(len(traces) * data_ratio))
            if max_samples < len(traces):
                # Reproducible subsample: seed a local RNG from the trainer seed
                # and the episode so identical config + identical rollouts pick
                # the same traces (the bare global random.sample was unseeded).
                trainer_seed = int(getattr(getattr(trainer, "args", None), "seed", 0) or 0)
                rng = random.Random(trainer_seed * 1_000_003 + episode)
                traces = rng.sample(traces, k=max_samples)
                log.info(
                    "[LiteReason/RP] Episode %d: sampled %d traces (ratio=%.2f)",
                    episode,
                    len(traces),
                    data_ratio,
                )
        if not traces:
            log.warning(
                "[LiteReason/RP] Episode %d: trace subsampling created 0 samples, skipping projector SFT",
                episode,
            )
            return

        # Step 3: Process traces into training samples (marker injection + tokenization).
        # The paper uses an equal mixture of sentence-replacement ratios (s_r=10%
        # and 25%); each trace is emitted once per ratio.
        swap_ratios = _parse_float_list(
            _env("LITEREASON_PROJECTOR_SFT_SWAP_RATIOS", "0.10,0.25", str),
            default=(0.10, 0.25),
        )
        token_ratio = _env("LITEREASON_PROJECTOR_SFT_TOKEN_RATIO", 0.2, float)
        max_len = _env("LITEREASON_PROJECTOR_SFT_MAX_LEN", 2048, int)
        binary_label_style = _env("LITEREASON_PROJECTOR_SFT_BINARY_LABEL_STYLE", "literal", str) or "literal"
        drop_last = _env("LITEREASON_PROJECTOR_SFT_DROP_LAST", False, bool)
        process_stats: Dict[str, int] = {}

        log.info(
            "[LiteReason/RP] Episode %d: config data_ratio=%.2f swap_ratios=%s "
            "token_ratio=%.2f max_len=%d label_style=%s drop_last=%s",
            episode,
            data_ratio,
            swap_ratios,
            token_ratio,
            max_len,
            binary_label_style,
            drop_last,
        )

        training_samples: List[ReasoningProjectorBatch] = []
        for swap_ratio in swap_ratios:
            training_samples.extend(
                process_reasoning_for_training(
                    traces,
                    trainer.tokenizer,
                    swap_ratio=swap_ratio,
                    token_ratio=token_ratio,
                    max_len=max_len,
                    stats=process_stats,
                )
            )
        if process_stats:
            log.info("[LiteReason/RP] Episode %d: process stats %s", episode, process_stats)

        if not training_samples:
            log.warning(
                "[LiteReason/RP] Episode %d: processing created no training samples, "
                "skipping projector SFT",
                episode,
            )
            return

        # Step 4: Send pre-tokenized dataset to GPU workers
        dataset = ReasoningProjectorDataset(training_samples)
        rp_epochs = _env("LITEREASON_PROJECTOR_SFT_EPOCHS", 1, int)
        lr = _env("LITEREASON_PROJECTOR_SFT_LR", 1e-4, float)
        bs = _projector_per_gpu_batch_size(trainer)

        log.info(
            f"[LiteReason/RP] Episode {episode}: projector SFT on {len(training_samples)} samples "
            f"(epochs={rp_epochs} bs={bs} lr={lr:g})"
        )
        refs = trainer.actor_model_group.async_run_method(
            method_name="fit_reasoning_projector",
            dataset=dataset,
            per_gpu_batch_size=bs,
            epochs=rp_epochs,
            learning_rate=lr,
        )
        results = ray.get(refs)
        if results:
            log.info(f"[LiteReason/RP] Episode {episode}: actor0 metrics: {results[0]}")
        _run_state_probe(
            trainer,
            episode=episode,
            stage="post_fit_pre_broadcast",
            probe_cases=probe_cases,
            include_vllm=False,
        )
    finally:
        if getattr(trainer.args, "deepspeed_enable_sleep", False):
            log.info("[LiteReason/RP] Offloading actor model states to CPU")
            ray.get(trainer.actor_model_group.async_run_method(method_name="offload_states"))
        _wake_unused_components(trainer)
        if trainer.vllm_engines is not None:
            trainer.broadcast_to_vllm()
            _run_state_probe(
                trainer,
                episode=episode,
                stage="post_broadcast",
                probe_cases=probe_cases,
                include_vllm=True,
            )
        trainer._litereason_rp_experiences = []
