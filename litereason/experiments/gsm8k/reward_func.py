"""Custom reward function for GSM8K GRPO training."""

import logging
import re
from typing import Any, Dict, List

import torch
from math_verify import parse, verify

log = logging.getLogger(__name__)

def extract_answer_text(text: str) -> str:
    """Extract final answer from model output (boxed or #### format)."""
    text = text or ""
    text = text.replace("\\boxed{<final_integer>}", "fake")
    # Try \\boxed{...} format
    boxed_matches = list(re.finditer(r'\\boxed\{([^}]*)\}', text))
    if boxed_matches:
        return boxed_matches[-1].group(1).strip()

    # Try #### format
    hash_matches = re.findall(r"####\s*([^\n]+)", text)
    if hash_matches:
        return hash_matches[-1].strip()

    return text


def is_equiv(pred: str, gold: str) -> bool:
    """Compare predicted and gold answers via math_verify parse+verify."""
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if not pred or not gold:
        return False

    # Match common math-eval usage where expressions are parsed in math mode.
    if "$" not in pred:
        pred = f"${pred}$"
    if "$" not in gold:
        gold = f"${gold}$"

    gold_parsed = parse(gold, parsing_timeout=None)
    pred_parsed = parse(pred, parsing_timeout=None)
    return bool(verify(gold_parsed, pred_parsed, timeout_seconds=None))

def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> Dict[str, Any]:
    """
    Compute rewards for GSM8K responses using math_verify.

    Args:
        queries: List of the model's generated responses (OpenRLHF names the
            first positional arg ``queries``; it is the model output, not the prompt)
        prompts: List of input prompts
        labels: List of gold answers

    Returns:
        Dictionary with:
            - rewards: List of reward scores for training
            - scores: List of scores for dynamic filtering (0-1 range)
            - extra_logs: Dictionary of additional metrics
    """
    rewards = []
    scores = []
    for response, label in zip(queries, labels, strict=True):
        pred_answer = extract_answer_text(response)
        try:
            is_correct = is_equiv(pred_answer, label)
        except Exception:
            # math_verify (sympy/latex parsing) raises a wide, unpredictable set
            # of errors on malformed model output; treat unparseable as incorrect.
            log.warning("math_verify failed for pred=%r gold=%r", pred_answer, label, exc_info=True)
            is_correct = False

        reward = score = 1.0 if is_correct else 0.0
        rewards.append(reward)
        scores.append(score)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    return {
        "rewards": rewards,
        "scores": scores,
    }
