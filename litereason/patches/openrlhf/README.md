# LiteReason OpenRLHF Integration

Monkeypatch helpers for integrating LiteReason into **upstream** OpenRLHF
without maintaining a long-lived fork.

## Architecture: Two Deferred Import Hooks

OpenRLHF uses Ray, which spawns separate Python processes for each actor.
LiteReason patches must be applied in the right process:

| Hook trigger | Patches applied | Target Ray process |
|---|---|---|
| `openrlhf.models.actor` | `from_pretrained`, `Actor.__init__`, `Actor.forward`, rollout stripping | PolicyModelActor (GPU workers) |
| `openrlhf.trainer.ppo_trainer` | `PPOTrainer.fit`, `BasePPOTrainer.train_step`, projector SFT methods | PPOTrainer (coordinator) |

Both hooks apply via the wrapper entrypoint
`litereason.patches.openrlhf.train_ppo_ray`: the driver applies them in
`main()` before delegating to upstream OpenRLHF, and each Ray worker
re-applies them in `LiteReasonPolicyModelActor.__init__` (idempotent). Each
hook intercepts the target module import, performs the real import, then
applies patches.

**Why two hooks?** The projector SFT patch (`openrlhf.trainer.ppo_trainer` hook)
imports `openrlhf.trainer.ray.ppo_actor`, which does
`from openrlhf.models import Actor`. If called from inside the
`openrlhf.models.actor` hook, `openrlhf.models.__init__` is still
mid-execution, causing a circular import. The separate hook avoids this
by deferring until `openrlhf.trainer.ppo_trainer` is imported on its own.

## What Gets Patched

### Actor patches (`actor_patch.py`, via `openrlhf.models.actor` hook)

1. **Global `AutoModelForCausalLM.from_pretrained` patch** checks for
   `config.is_litereason_model` and attaches the reasoning projector +
   loads projector weights. Works for both Liger and non-Liger code paths
   (Liger's `AutoLigerKernelForCausalLM` calls `super().from_pretrained()`).

2. **Tokenizer attachment** wraps `Actor.__init__` to call
   `model.set_tokenizer(tokenizer)` so marker detection works.

3. **Batched marker expansion in `Actor.forward`**: for each sample with
   markers, runs `model.shadow_forward()` (no-grad) to expand markers into
   reasoning embeddings, then runs one batched forward on `inputs_embeds`
   with `given_is_reasoning_embedding_mask`.

4. **vLLM rollout post-processing** wraps
   `SingleTurnAgentExecutor.execute` to strip dummy tokens emitted by the
   vLLM plugin after each marker (adjusts `action_ranges` / `rollout_log_probs`).

### Projector SFT patches (`projector_sft_patch.py`, via `openrlhf.trainer.ppo_trainer` hook)

1. **Response collection** wraps `BasePPOTrainer.train_step` to buffer all
   episode experiences during training, so their responses can feed the
   periodic projector SFT.

2. **Per-episode projector SFT** replaces `PPOTrainer.fit` with a version
   that calls `_train_projector_after_episode()` at the end of each episode.
   Extracts reasoning prefixes (text before `\boxed{}`), injects
   `<implicit_thought>N</implicit_thought>` markers, and runs projector-only
   SFT on the resulting training data.

3. **Remote training method** adds `fit_reasoning_projector` to
   `PolicyModelActor` so the coordinator can trigger projector SFT on GPU
   workers via Ray remote calls.

## Setup

Install the package once, then run training through the wrapper entrypoint.
There is no separate install step for the patches: the entrypoint applies
them in the driver, and each Ray worker re-applies them on init.

```bash
pip install -e ".[rl]"
ray start --head --num-gpus=4
```

### Running training

```bash
export VLLM_PLUGINS=litereason            # enable vLLM plugin
export LITEREASON_ENABLE_VLLM=1
export LITEREASON_ENABLE_PROJECTOR_SFT=1  # optional: periodic projector SFT

python -m litereason.patches.openrlhf.train_ppo_ray [OpenRLHF args...]
```

For a runnable end-to-end example (with projector SFT enabled), see
`litereason/experiments/flawed_fictions/lr_scripts/smoke_rl_pipeline.sh`, or
`run_smoke.sh` at the repo root.

## Environment Variables

### Core toggles
| Variable | Default | Description |
|---|---|---|
| `LITEREASON_ENABLE_OPENRLHF` | `1` | Enable actor-side monkeypatches |
| `LITEREASON_OPENRLHF_INIT_PROJECTOR` | `0` | Init projector from pretrained after model load |

### Projector SFT configuration
| Variable | Default | Description |
|---|---|---|
| `LITEREASON_ENABLE_PROJECTOR_SFT` | `0` | Enable periodic projector SFT after each episode |
| `LITEREASON_PROJECTOR_SFT_EPOCHS` | `1` | SFT epochs per episode |
| `LITEREASON_PROJECTOR_SFT_LR` | `1e-4` | Projector SFT learning rate |
| `LITEREASON_PROJECTOR_SFT_BATCH_SIZE` | micro_train_batch_size | Per-GPU batch size |
| `LITEREASON_PROJECTOR_SFT_SWAP_RATIOS` | `0.10,0.25` | Equal mixture of sentence-replacement ratios (each trace emitted once per ratio) |
| `LITEREASON_PROJECTOR_SFT_TOKEN_RATIO` | `0.2` | Latent tokens per original token for marker depth |
| `LITEREASON_PROJECTOR_SFT_DATA_RATIO` | `0.1` | Fraction of collected traces used per episode |
| `LITEREASON_PROJECTOR_SFT_MAX_LEN` | `2048` | Max tokenized length for SFT samples |

## Known Limitations

- `--packing_samples` is supported for `ring_attn_size==1` only (no ring-attention slicing).
- Projector SFT micro-batches per sample (marker expansion requires batch size 1).
- Patches depend on OpenRLHF internal APIs and may need updating when OpenRLHF is updated.
- GPU validation should be done on the target OpenRLHF + vLLM versions (TP=1 and TP>1).
