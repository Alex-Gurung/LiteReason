# LiteReason RL Scripts

This folder contains practical scripts for validating and running LiteReason RL on Flawed Fictions.

## Scripts

- `check_projector_load.py`
  - Loads a LiteReason checkpoint and verifies:
    - `reasoning_projector` exists.
    - The tokenizer/detector can find `<implicit_thought>N</implicit_thought>`.
    - `shadow_forward()` expands the sequence when a marker is present.

- `smoke_rl_pipeline.sh`
  - Runs a very short end-to-end RL smoke test with:
    - LiteReason vLLM plugin enabled.
    - OpenRLHF actor patches enabled.
    - Periodic projector SFT enabled.
  - Keeps full-run topology (same PPO/Ray wiring), but uses smoke-safe
    length/runtime defaults (`PROMPT_MAX_LEN=512`, `MAX_LEN=2560`,
    `NUM_EPISODES=1`, smaller `MAX_SAMPLES`).
  - Uses the LiteReason OpenRLHF wrapper entrypoint:
    `python -m litereason.patches.openrlhf.train_ppo_ray`.
  - Captures Ray submit logs to `OUTPUT_DIR/ray_job_submit.log` and, by default,
    asserts projector SFT ran with non-zero steps.
  - Key tunables (env): `ROLLOUT_BATCH_SIZE`, `TRAIN_BATCH_SIZE`,
    `MICRO_ROLLOUT_BATCH_SIZE`, `MICRO_TRAIN_BATCH_SIZE`,
    `N_SAMPLES_PER_PROMPT`, `GENERATE_MAX_LEN`, `PROMPT_MAX_LEN`,
    `ENABLE_GRADIENT_CHECKPOINTING`, `ENABLE_ADAM_OFFLOAD`,
    `ASSERT_PROJECTOR_SFT`, `ASSERT_PROJECTOR_SFT_STEPS`.

- `train_full_rl.sh`
  - Full paper-aligned GRPO training script for small models.
  - Uses the same LiteReason wrapper entrypoint and projector-SFT hooks.
  - Paper defaults are preserved, but you can override all major memory knobs
    via env vars (batch sizes, lengths, samples/prompt, gradient checkpointing,
    Adam offload, prefix caching, packing).

## Recommended order

1. `python check_projector_load.py --model <your_sft_checkpoint>`
2. `bash smoke_rl_pipeline.sh`
3. `bash train_full_rl.sh`

## One-time setup note

Ensure LiteReason is installed in the environment you use to run these scripts
(`python` on your `PATH`, unless overridden via `PYTHON_BIN`):

`pip install -e .`

## Defaults

- Python: `python` (or override `PYTHON_BIN`)
- Ray CLI: from the same environment as `PYTHON_BIN` when available
  (override via `RAY_BIN` if needed)
- Data: `litereason/experiments/flawed_fictions/data`
- Reward func: `litereason/experiments/flawed_fictions/reward_func.py`
- Ray address: `http://127.0.0.1:8265`
- Ray runtime `working_dir`: `.`
