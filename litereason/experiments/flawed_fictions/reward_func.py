"""Reward function for Flawed Fictions GRPO training.

Matches the reference reward function for Flawed Fictions.
"""


import re
from typing import Any, Dict, List, Optional

import torch

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_last_boxed(text: str) -> Optional[str]:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _parse_binary_token(text: str) -> Optional[float]:
    normalized = re.sub(r"[^a-z]+", "", (text or "").lower())
    if normalized == "yes":
        return 1.0
    if normalized == "no":
        return 0.0
    return None


def parse_prediction(raw_text: str) -> Optional[float]:
    # Flawed Fictions expects explicit final format: \boxed{Yes} / \boxed{No}.
    # Unboxed or malformed outputs are treated as incorrect.
    candidate = extract_last_boxed(raw_text)
    if candidate is None:
        return None
    return _parse_binary_token(candidate)


def parse_label(label_text: str) -> Optional[float]:
    token = (label_text or "").strip().lower()
    if token in {"1", "1.0", "yes", "true"}:
        return 1.0
    if token in {"0", "0.0", "no", "false"}:
        return 0.0
    return _parse_binary_token(token)


def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[str],
    **kwargs,
) -> Dict[str, Any]:
    results = []
    for response_text, label_text in zip(queries, labels, strict=True):
        pred = parse_prediction(response_text)
        gold = parse_label(label_text)
        results.append(float(pred is not None and gold is not None and pred == gold))

    t = torch.tensor(results, dtype=torch.float32)
    return {"rewards": t, "scores": t}
