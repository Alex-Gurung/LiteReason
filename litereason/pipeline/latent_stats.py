"""Detect latent-reasoning activity in generated text.

Used by evaluation to *prove* latent reasoning actually fired: the model emits
``<implicit_thought>N</implicit_thought>`` markers in its output, and each marker
runs N latent Reasoning-Projector steps.
"""

import re
from typing import Dict, List

_MARKER_RE = re.compile(r"<implicit_thought>\s*(\d+)\s*</implicit_thought>")


def num_markers_in_text(text: str) -> int:
    """Number of implicit-thought markers in ``text``."""
    return len(_MARKER_RE.findall(text or ""))


def latent_steps_in_text(text: str) -> int:
    """Total latent steps requested by the markers in ``text``."""
    return sum(int(n) for n in _MARKER_RE.findall(text or ""))


def aggregate_latent_stats(texts: List[str]) -> Dict[str, float]:
    """Aggregate latent-firing stats over a set of generations.

    Returns ``latent_trigger_rate`` (fraction with >=1 marker), ``mean_latent_steps``
    (mean total latent steps per generation), ``n_with_latents`` (count with >=1
    latent), and ``n_latent_samples`` (total generations scored). The latter is
    deliberately *not* named ``n`` so it does not collide with a dataset-size ``n``
    when this dict is merged into an evaluation summary.
    """
    n = len(texts)
    steps = [latent_steps_in_text(t) for t in texts]
    n_with = sum(1 for s in steps if s > 0)
    denom = n or 1
    return {
        "latent_trigger_rate": n_with / denom,
        "mean_latent_steps": sum(steps) / denom,
        "n_with_latents": n_with,
        "n_latent_samples": n,
    }
