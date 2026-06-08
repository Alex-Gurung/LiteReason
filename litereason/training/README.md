# Training

LiteReason uses a two-stage training pipeline:

1. **Stage 1: SFT**: Train the reasoning projector on data with `<implicit_thought>` markers (base model frozen).
2. **Stage 2: RL**: GRPO fine-tuning via OpenRLHF, with optional periodic projector SFT.

## Stage 1: Supervised Fine-Tuning (SFT)

Train the reasoning projector while freezing the base model:

```bash
# Single GPU
python -m litereason.training.train_sft \
    --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml

# Multi-GPU
accelerate launch -m litereason.training.train_sft \
    --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml
```

The SFT trainer uses `MarkerAwareCollator` to mask labels before the first marker, so loss is only computed on post-reasoning tokens. Configuration is via `ScriptArguments` and `SFTConfig` dataclasses (see `configs.py`), which extend TRL's defaults with LiteReason-specific options like `num_reasoning_layers` and `max_reasoning_steps`.

**Files:**
- `train_sft.py`: Main SFT training script
- `collate.py`: `MarkerAwareCollator` for marker-aware loss masking
- `configs.py`: `ScriptArguments` and `SFTConfig` dataclasses

## Stage 2: Reinforcement Learning (GRPO)

RL training uses OpenRLHF with runtime patches (in `patches/openrlhf/`). The patches handle marker expansion during the Actor forward pass, vLLM dummy token stripping, and optional projector SFT between episodes (to update the reasoning projector on the traces generated during training).

### Prerequisites

```bash
pip install -e ".[rl]"       # installs vLLM, Ray, OpenRLHF deps
ray start --head --num-gpus=N
```

### Running

A skeleton script is provided in this directory:

```bash
# Edit run_grpo.sh to set MODEL, DATA_DIR, REWARD_FUNC, NUM_GPUS
bash litereason/training/run_grpo.sh
```

For experiment-specific scripts with tuned hyperparameters, see `experiments/flawed_fictions/scripts/`.

### How it works

LiteReason patches OpenRLHF at runtime via `patches/openrlhf/actor_patch.py` (see `patches/openrlhf/README.md` for full details). Activation goes through the wrapper entrypoint:

```bash
python -m litereason.patches.openrlhf.train_ppo_ray [OpenRLHF args...]
```

The driver applies the patches before delegating to upstream OpenRLHF, and each Ray worker re-applies them in `LiteReasonPolicyModelActor.__init__` (idempotent). This relies only on the package being installed in the environment Ray launches workers in, so no `PYTHONPATH` edits are needed. `run_grpo.sh` and the experiment scripts call this entrypoint.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_PLUGINS` | (none) | Set to `litereason` for vLLM plugin |
| `LITEREASON_ENABLE_VLLM` | `0` | Enable vLLM plugin |
| `LITEREASON_ENABLE_OPENRLHF` | `1` | Enable actor patches |
| `LITEREASON_ENABLE_PROJECTOR_SFT` | `0` | Periodic projector SFT after each episode |

See `patches/openrlhf/README.md` for the full environment variable reference including projector SFT tuning knobs.

### Projector SFT during RL

When `LITEREASON_ENABLE_PROJECTOR_SFT=1`, the trainer collects reasoning traces during rollouts and runs projector-only SFT at the end of each episode. This keeps the projector aligned as the base model's policy evolves. Controlled via `LITEREASON_PROJECTOR_SFT_*` env vars (documented in `patches/openrlhf/README.md`).
