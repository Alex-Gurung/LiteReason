"""LITEREASON_MAX_REASONING_STEPS and the model's configured cap are reconciled at
attach so all paths agree. The configured cap is authoritative: a disagreeing env
var is ignored (not allowed to silently desync HF training vs. the vLLM rollout),
and the authoritative value is published to the env for config-less readers.

CPU-only: a tiny Qwen2 from config, no download.
"""
import os

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from litereason.causal_lm_with_reasoning import attach_litereason_to_causal_lm


def _tiny_model():
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


def test_unset_env_publishes_configured_cap(monkeypatch):
    monkeypatch.delenv("LITEREASON_MAX_REASONING_STEPS", raising=False)
    model = attach_litereason_to_causal_lm(_tiny_model(), max_reasoning_steps=5)
    assert model._litereason_max_reasoning_steps == 5
    # Published to the env so the (config-less) dummy-stripper stops defaulting to 32.
    assert os.environ["LITEREASON_MAX_REASONING_STEPS"] == "5"


def test_env_does_not_override_configured_cap(monkeypatch):
    monkeypatch.setenv("LITEREASON_MAX_REASONING_STEPS", "7")
    model = attach_litereason_to_causal_lm(_tiny_model(), max_reasoning_steps=5)
    # The configured cap is authoritative; a disagreeing env var is ignored so it
    # cannot silently desync the HF training forward from the vLLM rollout.
    assert model._litereason_max_reasoning_steps == 5
    assert model.config.litereason_max_reasoning_steps == 5
    # The authoritative value is published to the env (overwriting the stale one),
    # so the config-less dummy stripper uses it too.
    assert os.environ["LITEREASON_MAX_REASONING_STEPS"] == "5"


if __name__ == "__main__":
    os.environ.pop("LITEREASON_MAX_REASONING_STEPS", None)
    m = attach_litereason_to_causal_lm(_tiny_model(), max_reasoning_steps=5)
    assert m._litereason_max_reasoning_steps == 5
    assert os.environ["LITEREASON_MAX_REASONING_STEPS"] == "5"
    os.environ["LITEREASON_MAX_REASONING_STEPS"] = "7"
    m2 = attach_litereason_to_causal_lm(_tiny_model(), max_reasoning_steps=5)
    assert m2._litereason_max_reasoning_steps == 5  # env ignored, config wins
    assert os.environ["LITEREASON_MAX_REASONING_STEPS"] == "5"
    print("OK: max-steps resolution tests passed")
