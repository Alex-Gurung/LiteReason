#!/usr/bin/env python3
"""Check that the LiteReason OpenRLHF patches' hook points exist in the installed openrlhf.

`litereason/patches/openrlhf/` monkeypatches and subclasses private OpenRLHF internals by
name, so an openrlhf upgrade can move them out from under the patches. Run this after
bumping openrlhf (and WITHOUT LITEREASON_ENABLE_OPENRLHF set, so it inspects vanilla
openrlhf rather than the already-patched classes):

    python check_openrlhf_hooks.py

A moved class/function fails its import with a precise error; a changed method fails an
assert naming what it lost. The Ray-actor classes (PPOTrainer, PolicyModelActor,
LLMRayActor) are unwrapped via __ray_metadata__.modified_class exactly as the patches do.
"""
import inspect

import openrlhf
from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from openrlhf.models.utils import compute_entropy, log_probs_from_logits
from openrlhf.trainer.ppo_trainer import PPOTrainer
from openrlhf.trainer.ppo_utils import balance_experiences
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.vllm_engine import RolloutRayActor
from openrlhf.utils.agent import SingleTurnAgentExecutor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger

# Imported above only to confirm they still live where the patches expect them;
# referenced here so the imports aren't flagged as unused.
_PATCH_DEPS = (
    gather_and_pad_tensor, unpad_and_slice_tensor, compute_entropy,
    log_probs_from_logits, balance_experiences, init_logger,
)


def require_params(cls, method, *params):
    """Assert cls.method exists and still accepts the params the patch passes/overrides."""
    sig = inspect.signature(getattr(cls, method))
    missing = [p for p in params if p not in sig.parameters]
    assert not missing, f"{cls.__name__}.{method} changed: missing {missing} (has {list(sig.parameters)})"


def _actor_class(remote_cls):
    """Unwrap a @ray.remote actor to its underlying class, as the patches do."""
    return remote_cls.__ray_metadata__.modified_class


# Actor.__init__ and Actor.forward are monkeypatched in place; the forward wrapper
# delegates to the original with these keyword args, so they must still exist.
require_params(Actor, "__init__")
require_params(
    Actor, "forward",
    "action_mask", "attention_mask", "return_output", "allgather_logits",
    "return_logprobs", "ring_attn_group", "packed_seq_lens", "return_entropy",
)

# DeepspeedStrategy.backward / optimizer_step are wrapped for the non-finite guard.
# The 3rd positional arg is always passed positionally by openrlhf's trainers, so its
# name (optimizer vs _optimizer across versions) doesn't matter — assert only the
# stable leading params.
require_params(DeepspeedStrategy, "backward", "loss", "model")
require_params(DeepspeedStrategy, "optimizer_step")

# SingleTurnAgentExecutor.execute is wrapped to strip vLLM latent-step dummy tokens.
require_params(SingleTurnAgentExecutor, "execute")

# The Ray-actor classes are subclassed via their modified_class; check the methods
# LiteReason overrides still exist with the expected parameters.
require_params(
    _actor_class(PolicyModelActor), "init_model_from_pretrained",
    "strategy", "pretrain", "max_steps", "vllm_engines",
)
require_params(_actor_class(PPOTrainer), "fit")
require_params(_actor_class(RolloutRayActor), "__init__")

print(f"OK: openrlhf {getattr(openrlhf, '__version__', '(unknown)')} — patch hook points present.")
