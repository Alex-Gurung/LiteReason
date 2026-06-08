#!/usr/bin/env python3
"""Validate that a LiteReason checkpoint loads and expands implicit-thought markers.

This script checks:
1) The model is a LiteReason checkpoint (`config.is_litereason_model`).
2) `reasoning_projector` exists and has layers.
3) Marker detection finds `<implicit_thought>N</implicit_thought>`.
4) `shadow_forward()` expands sequence length when a marker is present.
"""


import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer

from litereason import AutoModelForCausalLMWithReasoning


@dataclass(frozen=True)
class CheckResult:
    model_name_or_path: str
    marker_complexity: int
    original_length: int
    expanded_length: int
    inserted_reasoning_steps: int
    projector_layers: int


DTYPE_BY_NAME = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LiteReason SFT checkpoint (local path or HF repo id).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path/repo. Defaults to --model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to run the sanity forward on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=tuple(DTYPE_BY_NAME.keys()),
        help="Torch dtype for loading.",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=3,
        help="Complexity N in <implicit_thought>N</implicit_thought>.",
    )
    parser.add_argument(
        "--max-reasoning-steps",
        type=int,
        default=5,
        help="Upper bound sanity check for marker complexity.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_projector_layer_count(model) -> int:
    projector = getattr(model, "reasoning_projector", None)
    if projector is None and hasattr(model, "model"):
        projector = getattr(model.model, "reasoning_projector", None)
    if projector is None:
        return 0
    try:
        return len(projector)
    except TypeError:
        return 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_check(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str],
    device: str,
    dtype_name: str,
    marker_complexity: int,
    max_reasoning_steps: int,
) -> CheckResult:
    _require(1 <= marker_complexity <= max_reasoning_steps, "Marker complexity is out of allowed bounds.")

    device_obj = _resolve_device(device)
    tokenizer_path = tokenizer_name_or_path or model_name_or_path
    print(f"[check] loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"[check] loading model: {model_name_or_path}")
    model = AutoModelForCausalLMWithReasoning.from_pretrained(
        model_name_or_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        torch_dtype=DTYPE_BY_NAME[dtype_name],
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[check] moving model to device: {device_obj}")
    model.to(device_obj)

    is_litereason = bool(getattr(model.config, "is_litereason_model", False))
    _require(is_litereason, "Checkpoint is not marked as a LiteReason model (config.is_litereason_model != True).")

    projector_layers = _resolve_projector_layer_count(model)
    _require(projector_layers > 0, "reasoning_projector is missing or empty.")

    detector = getattr(model, "_litereason_thought_detector", None)
    _require(detector is not None, "Implicit-thought detector is missing on model.")

    sample_text = (
        "Continuity check reasoning. "
        f"<implicit_thought>{marker_complexity}</implicit_thought> "
        "Final answer."
    )
    encoded = tokenizer(sample_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device_obj)

    markers = detector.find_thought_markers(input_ids)
    _require(bool(markers), "No implicit-thought marker detected in sanity text.")

    print(f"[check] running shadow_forward on device: {device_obj}")
    with torch.no_grad():
        expanded_embeddings, keep_mask = model.shadow_forward(input_ids=input_ids, use_cache=False)

    _require(expanded_embeddings.ndim == 3, "shadow_forward returned malformed embeddings.")
    _require(keep_mask.ndim == 1, "shadow_forward returned malformed keep mask.")

    original_length = int(input_ids.shape[1])
    expanded_length = int(expanded_embeddings.shape[1])
    inserted_reasoning_steps = expanded_length - original_length

    _require(expanded_length > original_length, "Sequence was not expanded; projector path likely inactive.")
    _require(inserted_reasoning_steps >= marker_complexity, "Expanded length did not include expected reasoning steps.")

    return CheckResult(
        model_name_or_path=model_name_or_path,
        marker_complexity=marker_complexity,
        original_length=original_length,
        expanded_length=expanded_length,
        inserted_reasoning_steps=inserted_reasoning_steps,
        projector_layers=projector_layers,
    )


def main() -> None:
    args = parse_args()

    result = run_check(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        dtype_name=args.dtype,
        marker_complexity=args.complexity,
        max_reasoning_steps=args.max_reasoning_steps,
    )

    print("LiteReason projector load check passed.")
    print(f"model: {result.model_name_or_path}")
    print(f"projector_layers: {result.projector_layers}")
    print(f"marker_complexity: {result.marker_complexity}")
    print(f"original_length: {result.original_length}")
    print(f"expanded_length: {result.expanded_length}")
    print(f"inserted_reasoning_steps: {result.inserted_reasoning_steps}")


if __name__ == "__main__":
    main()
