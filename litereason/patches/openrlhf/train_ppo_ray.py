"""Entry point for OpenRLHF PPO training with LiteReason projector SFT.

Usage::

    python -m litereason.patches.openrlhf.train_ppo_ray [OpenRLHF args...]

Defines ``LiteReasonPolicyModelActor`` and ``LiteReasonPPOTrainer`` subclasses
that add projector SFT support without monkeypatching:

- ``LiteReasonPolicyModelActor``: adds ``fit_reasoning_projector`` as a proper
  class method (visible in Ray metadata, dispatched to GPU workers).
- ``LiteReasonPPOTrainer``: overrides ``train_step`` (experience collection)
  and ``fit`` (per-episode projector SFT orchestration).
"""


import hashlib
import logging
import os
import random
import runpy
import time
from functools import lru_cache

import openrlhf.trainer.ppo_trainer as _trainer_mod
import openrlhf.trainer.ray.ppo_actor as _actor_mod
import openrlhf.trainer.ray.vllm_engine as _vllm_engine_mod
import ray
import torch
from openrlhf.trainer.ppo_trainer import PPOTrainer as _RemotePPOTrainer
from openrlhf.trainer.ppo_utils import balance_experiences
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor as _RemotePolicyModelActor
from openrlhf.utils.logging_utils import init_logger
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from litereason.patches.openrlhf.projector_sft_patch import (
    _env,
    train_projector_after_episode,
    train_reasoning_projector_distributed,
)
from litereason.stop_tokens import assistant_suffix_stop_token_ids, normalize_token_ids

log = logging.getLogger(__name__)

# Marker used by actor_patch.py to avoid false-positive entrypoint warnings.
os.environ.setdefault("LITEREASON_USE_WRAPPED_ENTRYPOINT", "1")

# Unwrap to get the base classes before @ray.remote decoration.
_BasePolicyModelActor = _RemotePolicyModelActor.__ray_metadata__.modified_class
_BasePPOTrainer = _RemotePPOTrainer.__ray_metadata__.modified_class


@lru_cache(maxsize=None)
def _resolve_vllm_generation_override(pretrain: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
    except (OSError, ValueError, EnvironmentError) as e:
        # Can't load a tokenizer for this checkpoint (bad path, network, custom
        # code): skip the backfill entirely; vLLM keeps its own stop ids.
        log.debug("LiteReason: tokenizer load failed for %r; skipping vLLM backfill: %s", pretrain, e)
        return None

    extra_stop_ids = assistant_suffix_stop_token_ids(tokenizer)
    if not extra_stop_ids:
        return None

    try:
        existing_eos = GenerationConfig.from_pretrained(pretrain).eos_token_id
    except (OSError, ValueError, EnvironmentError) as e:
        # Checkpoint may ship no generation_config.json; fall back to the
        # tokenizer's eos so we still merge against a sensible baseline.
        log.debug(
            "LiteReason: generation_config load failed for %r; using tokenizer eos: %s",
            pretrain,
            e,
        )
        existing_eos = getattr(tokenizer, "eos_token_id", None)

    merged_eos = normalize_token_ids(
        getattr(tokenizer, "eos_token_id", None),
        existing_eos,
        extra_stop_ids,
    )
    if not merged_eos:
        return None

    existing_eos_norm = normalize_token_ids(existing_eos)
    if merged_eos == existing_eos_norm:
        return None

    return {"eos_token_id": merged_eos}


def _patch_vllm_generation_config_backfill() -> None:
    """Backfill chat stop ids for vLLM when checkpoint generation_config is incomplete."""
    remote_cls = _vllm_engine_mod.RolloutRayActor.__ray_metadata__.modified_class
    if getattr(remote_cls, "_litereason_generation_backfill_patched", False):
        return

    orig_init = remote_cls.__init__

    async def patched_init(self, *args, **kwargs):
        model_name = kwargs.get("model")
        override = (
            _resolve_vllm_generation_override(model_name)
            if isinstance(model_name, str)
            else None
        )
        if override is not None:
            merged_override = dict(kwargs.get("override_generation_config") or {})
            merged_override["eos_token_id"] = normalize_token_ids(
                merged_override.get("eos_token_id"),
                override["eos_token_id"],
            )
            kwargs["override_generation_config"] = merged_override
            log.info(
                "LiteReason: backfilled vLLM generation eos_token_id=%s for %s",
                merged_override["eos_token_id"],
                model_name,
            )

        # Log loudly on init failure (the vLLM engine actor is otherwise hard to
        # diagnose from the driver) and re-raise so startup still fails fast.
        try:
            return await orig_init(self, *args, **kwargs)
        except Exception:
            log.exception("LiteReason: vLLM engine __init__ failed for model=%r", model_name)
            raise

    remote_cls.__init__ = patched_init
    remote_cls._litereason_generation_backfill_patched = True
    log.info(
        "LiteReason: armed vLLM generation backfill patch on %s.%s",
        remote_cls.__module__,
        getattr(remote_cls, "__qualname__", remote_cls.__name__),
    )


# ---------------------------------------------------------------------------
# GPU worker: PolicyModelActor + fit_reasoning_projector
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class LiteReasonPolicyModelActor(_BasePolicyModelActor):
    """PolicyModelActor with ``fit_reasoning_projector`` for projector SFT."""

    def __init__(self, *args, **kwargs):
        # Local import: actor_patch monkeypatches OpenRLHF on import; keep it out of
        # module load so patching is driven by the explicit call below, not import order.
        from litereason.patches.openrlhf.actor_patch import patch_openrlhf_actor

        super().__init__(*args, **kwargs)
        patch_openrlhf_actor()  # idempotent: ensures projector is attached in GPU workers

    def init_model_from_pretrained(self, strategy, pretrain, max_steps=None, vllm_engines=None):
        ray_gpu_ids = [int(gid) for gid in ray.get_gpu_ids()]
        if torch.cuda.is_available():
            try:
                visible_count = torch.cuda.device_count()
            except RuntimeError:  # pragma: no cover
                # Best-effort GPU count; fall back to None so the mapping below
                # treats it as "unknown" rather than crashing the actor init.
                # A failed probe silently degrades the GPU->LOCAL_RANK mapping,
                # so warn rather than swallowing it entirely.
                visible_count = None
                log.warning(
                    "[LiteReason/ActorEnv] torch.cuda.device_count() probe failed; "
                    "LOCAL_RANK mapping falls back to 0 for ray_gpu_ids=%s",
                    ray_gpu_ids,
                )
            # Some Ray worker processes import torch before CUDA_VISIBLE_DEVICES is
            # narrowed, so each actor still sees all node-local GPUs. In that case we
            # must map LOCAL_RANK to the Ray-assigned physical GPU id explicitly.
            if ray_gpu_ids:
                mapped_local_rank = ray_gpu_ids[0] if (visible_count or 0) > 1 else 0
                os.environ["LOCAL_RANK"] = str(mapped_local_rank)
                torch.cuda.set_device(mapped_local_rank)

        def _diag_snapshot():
            info = {
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "world_size": os.environ.get("WORLD_SIZE"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ray_gpu_ids": ray_gpu_ids,
                "pid": os.getpid(),
            }
            if torch.cuda.is_available():
                try:
                    info["device_count"] = torch.cuda.device_count()
                    info["current_device"] = torch.cuda.current_device()
                    info["current_uuid"] = str(
                        torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
                    )
                except Exception as exc:  # pragma: no cover - diagnostics only
                    info["cuda_probe_error"] = repr(exc)
            return info

        log.info("[LiteReason/ActorEnv] before_init %s", _diag_snapshot())
        try:
            result = super().init_model_from_pretrained(strategy, pretrain, max_steps, vllm_engines)
        except Exception:
            # Surface the GPU/rank snapshot alongside the traceback; init failures
            # in a Ray actor are otherwise hard to attribute to a device/rank.
            log.exception("[LiteReason/ActorEnv] init_failed %s", _diag_snapshot())
            raise

        log.info("[LiteReason/ActorEnv] after_init %s", _diag_snapshot())
        return result

    def fit_reasoning_projector(self, dataset, per_gpu_batch_size, epochs, learning_rate):
        torch.cuda.empty_cache()
        status = train_reasoning_projector_distributed(
            self.trainer,
            dataset,
            int(per_gpu_batch_size),
            int(epochs),
            float(learning_rate),
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def debug_state_probe(self, prompts, max_new_tokens=96):
        prompts = list(prompts or [])
        model = getattr(self.trainer.actor, "model", None)
        if model is None:
            raise RuntimeError("trainer.actor.model missing")

        probe_model = model.module if hasattr(model, "module") else model
        tokenizer = self.tokenizer
        device = torch.cuda.current_device()
        stop_token_ids = normalize_token_ids(
            getattr(tokenizer, "eos_token_id", None),
            assistant_suffix_stop_token_ids(tokenizer),
        )

        def _group_digest(include_projector: bool) -> dict:
            h = hashlib.sha1()
            param_count = 0
            sampled_values = 0
            for name, param in probe_model.named_parameters():
                is_projector = "reasoning_projector" in name
                if is_projector != include_projector:
                    continue
                tensor = param.detach()
                if tensor.numel() == 0:
                    continue
                flat_cpu = tensor.float().reshape(-1).cpu()
                steps = min(16, int(flat_cpu.numel()))
                if steps <= 0:
                    continue
                if steps == 1:
                    sample = flat_cpu[:1]
                else:
                    idx = torch.linspace(
                        0,
                        flat_cpu.numel() - 1,
                        steps=steps,
                        dtype=torch.float64,
                    ).round().long()
                    sample = flat_cpu[idx]
                h.update(name.encode("utf-8"))
                h.update(sample.numpy().tobytes())
                param_count += 1
                sampled_values += steps
            return {
                "sha1": h.hexdigest(),
                "param_count": param_count,
                "sampled_values": sampled_values,
            }

        was_training = bool(getattr(probe_model, "training", False))
        if hasattr(probe_model, "eval"):
            probe_model.eval()

        generations = []
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = attention_mask.to(device)

                generated = []
                cur_input_ids = input_ids
                cur_attention_mask = attention_mask
                for _ in range(int(max_new_tokens)):
                    outputs = probe_model(
                        input_ids=cur_input_ids,
                        attention_mask=cur_attention_mask,
                        return_dict=True,
                    )
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                    token_id = int(next_token[0].item())
                    generated.append(token_id)
                    cur_input_ids = torch.cat([cur_input_ids, next_token[:, None]], dim=-1)
                    cur_attention_mask = torch.cat(
                        [cur_attention_mask, torch.ones_like(next_token[:, None])],
                        dim=-1,
                    )
                    if stop_token_ids and token_id in stop_token_ids:
                        break

                generations.append(
                    {
                        "prompt_chars": len(prompt),
                        "new_token_count": len(generated),
                        "response_text": tokenizer.decode(
                            generated,
                            skip_special_tokens=False,
                        ),
                    }
                )

        if was_training and hasattr(probe_model, "train"):
            probe_model.train()

        return {
            "rank": int(os.environ.get("RANK", "0")),
            "model_type": type(probe_model).__name__,
            "stop_token_ids": stop_token_ids,
            "base_digest": _group_digest(include_projector=False),
            "projector_digest": _group_digest(include_projector=True),
            "generations": generations,
        }


# ---------------------------------------------------------------------------
# Coordinator: PPOTrainer + experience collection + projector SFT
# ---------------------------------------------------------------------------


@ray.remote
class LiteReasonPPOTrainer(_BasePPOTrainer):
    """PPOTrainer with experience collection and per-episode projector SFT."""

    def __init__(self, *args, **kwargs):
        # Local import: actor_patch monkeypatches OpenRLHF on import; keep it out of
        # module load so patching is driven by the explicit call below, not import order.
        from litereason.patches.openrlhf.actor_patch import patch_openrlhf_actor

        patch_openrlhf_actor()  # safety net: ensures patches in coordinator process
        super().__init__(*args, **kwargs)

    def train_step(self, rollout_samples, global_step: int):
        # --- metric-equivalent to upstream BasePPOTrainer.train_step, plus
        #     LiteReason experience buffering for projector SFT ---
        t0 = time.time()
        experiences = self.experience_maker.make_experience_batch(rollout_samples)
        make_experience_time = time.time() - t0

        sample0 = [
            self.tokenizer.batch_decode(
                experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
            )[0],
            experiences[0].info["reward"][0].item(),
        ]
        log.debug("sample0: %s", sample0)

        # Ground-truth rollout stats BEFORE dynamic batch splitting (upstream parity).
        # Guarded in case the method name differs across OpenRLHF versions.
        rollout_stats = {}
        if hasattr(self, "_compute_rollout_stats"):
            rollout_stats = self._compute_rollout_stats(experiences)

        # LiteReason: store experiences for projector SFT, with a bounded buffer.
        if _env("LITEREASON_ENABLE_PROJECTOR_SFT", False, bool):
            self._buffer_rp_experiences(experiences)

        if self.args.train.dynamic_batch_enable:
            experiences = balance_experiences(experiences, self.args)

        refs = self.actor_model_group.async_run_method_batch(
            method_name="append", experience=experiences
        )
        if self.critic_model_group is not None:
            refs.extend(
                self.critic_model_group.async_run_method_batch(
                    method_name="append", experience=experiences
                )
            )
        ray.get(refs)

        t0 = time.time()
        status = self.ppo_train(global_step)
        ppo_train_time = time.time() - t0

        t0 = time.time()
        if self.vllm_engines is not None:
            self.broadcast_to_vllm()
        broadcast_time = time.time() - t0

        if "kl" in status:
            self.kl_ctl.update(
                status["kl"],
                self.args.rollout.batch_size * self.args.rollout.n_samples_per_prompt,
            )

        # Per-phase timing breakdown (upstream parity).
        status["timing/make_experience"] = make_experience_time
        status["timing/ppo_train"] = ppo_train_time
        status["timing/broadcast"] = broadcast_time

        # Merge rollout stats (ground-truth, pre-dynamic-batch) so downstream
        # readers of status["rollout/..."] don't KeyError and wandb keeps metrics.
        status.update(rollout_stats)

        status["generated_samples"] = sample0
        return status, global_step + 1

    def _buffer_rp_experiences(self, experiences) -> None:
        """Append experiences to the projector-SFT buffer with a bounded size.

        A long episode can otherwise accumulate every Experience (sequences,
        logprobs, advantages) on the coordinator and OOM the driver. Since
        ``train_projector_after_episode`` later keeps only ``data_ratio`` (~10%)
        of traces, a uniform reservoir sample of the experiences is sufficient.

        The cap is controlled by ``LITEREASON_PROJECTOR_SFT_MAX_BUFFER`` (number
        of buffered Experience objects; default 4096). Once the buffer is full,
        new experiences replace existing ones with probability that keeps the
        retained sample uniform over all experiences seen this episode.
        """
        max_buffer = _env("LITEREASON_PROJECTOR_SFT_MAX_BUFFER", 4096, int)
        buf = getattr(self, "_litereason_rp_experiences", [])
        seen = getattr(self, "_litereason_rp_experiences_seen", 0)
        dropped = getattr(self, "_litereason_rp_experiences_dropped", 0)

        if max_buffer <= 0:
            # No cap: preserve previous unbounded behaviour.
            buf.extend(experiences)
            self._litereason_rp_experiences = buf
            return

        for exp in experiences:
            seen += 1
            if len(buf) < max_buffer:
                buf.append(exp)
            else:
                # Reservoir sampling: keep the retained set uniform.
                j = random.randint(0, seen - 1)
                if j < max_buffer:
                    buf[j] = exp
                dropped += 1

        self._litereason_rp_experiences = buf
        self._litereason_rp_experiences_seen = seen
        if dropped and dropped != getattr(self, "_litereason_rp_experiences_dropped", 0):
            log.warning(
                "LiteReason/RP: projector-SFT experience buffer capped at %d "
                "(LITEREASON_PROJECTOR_SFT_MAX_BUFFER); reservoir-sampling, "
                "%d experiences dropped so far this episode (%d seen)",
                max_buffer,
                dropped,
                seen,
            )
        self._litereason_rp_experiences_dropped = dropped

    def fit(self) -> None:
        logger = init_logger(__name__)

        # --- identical to upstream PPOTrainer.fit, with LiteReason additions ---
        checkpoint_states = self.init_checkpoint_states()
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states["total_consumed_prompts"]
        if global_step:
            self.broadcast_to_vllm()
            state_dict = checkpoint_states["data_loader_state_dict"]
            if state_dict:
                self.prompts_dataloader.load_state_dict(state_dict)

        for episode in range(start_episode, self.args.train.num_episodes):
            # LiteReason: init (bounded) experience buffer for projector SFT
            if _env("LITEREASON_ENABLE_PROJECTOR_SFT", False, bool):
                self._litereason_rp_experiences = []
                self._litereason_rp_experiences_seen = 0
                self._litereason_rp_experiences_dropped = 0

            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.train.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(**self.generate_kwargs)
                )
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                status, global_step = self.train_step(rollout_samples, global_step)

                if self.args.algo.dynamic_filtering_enable:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                log_status = {
                    k: v for k, v in status.items() if k not in ["generated_samples"]
                }
                logger.info(f"✨ Global step {global_step}: {log_status}")

                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                if global_step % self.args.eval.steps == 0 and self.eval_dataloader:
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval.temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = (
                        self.args.eval.n_samples_per_prompt
                    )
                    self.evaluate(global_step, **eval_generate_kwargs)

                pbar.update(prompts_consumed)

            # LiteReason: projector SFT at end of episode
            train_projector_after_episode(self, episode)

        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Local import: actor_patch monkeypatches OpenRLHF on import, and patching must
    # happen before runpy launches the upstream CLI below, hence not at module load.
    from litereason.patches.openrlhf.actor_patch import patch_openrlhf_actor

    patch_openrlhf_actor()
    if os.getenv("LITEREASON_DISABLE_VLLM_GENCFG_BACKFILL") == "1":
        log.info("LiteReason: skipping vLLM generation-config backfill by env override.")
    else:
        _patch_vllm_generation_config_backfill()

    # Substitute our subclasses into the upstream modules so that upstream's
    # train() function picks them up via ``from ... import ...``.
    _actor_mod.PolicyModelActor = LiteReasonPolicyModelActor
    _trainer_mod.PPOTrainer = LiteReasonPPOTrainer

    # Run the upstream CLI (handles argparse, Ray init, model init, training).
    runpy.run_module("openrlhf.cli.train_ppo_ray", run_name="__main__")


if __name__ == "__main__":
    # Under `python -m ...` this module is "__main__", so the actor subclasses and
    # patch closures defined here would be cloudpickled with __main__ references
    # that Ray workers (whose __main__ is default_worker.py) cannot resolve. Re-enter
    # through the stable module name so every symbol Ray serializes is importable.
    from litereason.patches.openrlhf.train_ppo_ray import main as _stable_main

    _stable_main()
