"""Tests for the dataset-agnostic pipeline helpers (CPU-only)."""
from litereason.pipeline.latent_stats import (
    aggregate_latent_stats,
    latent_steps_in_text,
    num_markers_in_text,
)
from litereason.pipeline.metrics import compute_rep_stats, wilson_interval
from litereason.pipeline.reward import load_reward_func, score_responses


def test_load_reward_func_and_score(tmp_path):
    rf = tmp_path / "reward_func.py"
    rf.write_text(
        "def reward_func(queries, prompts, labels, **k):\n"
        "    return {'rewards': [1.0 if q == l else 0.0 for q, l in zip(queries, labels)]}\n"
    )
    fn = load_reward_func(str(rf))
    scores = score_responses(fn, responses=["yes", "no"], prompts=["p", "p"], labels=["yes", "x"])
    assert scores == [1.0, 0.0]


def test_load_reward_func_missing(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_reward_func(str(tmp_path / "nope.py"))


def test_latent_stats():
    text = "think <implicit_thought>3</implicit_thought> more <implicit_thought> 2 </implicit_thought> done"
    assert num_markers_in_text(text) == 2
    assert latent_steps_in_text(text) == 5
    agg = aggregate_latent_stats([text, "no markers here", ""])
    assert agg["n_with_latents"] == 1
    assert agg["latent_trigger_rate"] == 1 / 3
    assert abs(agg["mean_latent_steps"] - 5 / 3) < 1e-9


def test_metrics():
    s = compute_rep_stats([0.8, 0.9, 0.85])
    assert abs(s["mean"] - 0.85) < 1e-9 and s["k"] == 3
    assert compute_rep_stats([0.5]) == {"mean": 0.5, "std": 0.0, "ci_lo": 0.5, "ci_hi": 0.5, "k": 1}
    lo, hi = wilson_interval(8, 10)
    assert 0.0 < lo < 0.8 < hi < 1.0


if __name__ == "__main__":
    import pathlib
    import tempfile

    d = pathlib.Path(tempfile.mkdtemp())
    test_load_reward_func_and_score(d)
    test_latent_stats()
    test_metrics()
    print("OK: pipeline helper tests passed")
