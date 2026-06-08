#!/usr/bin/env bash
# Tiny end-to-end LiteReason RL smoke: a short GRPO run on a small model that
# exercises the full training path. Use it to validate the RL fidelity changes
# end-to-end -- it touches all of these:
#   - OpenRLHF + Ray job submission and the LiteReason actor patches
#   - the LiteReason vLLM plugin (latent reasoning during rollouts)
#   - the end-of-episode projector SFT hook (the 10/25% swap mixture, LR 1e-4),
#     which the assertions at the end confirm actually ran with >0 steps
#
# Prerequisites:
#   - 1+ GPU and `uv pip install -e ".[rl]"` (vLLM + OpenRLHF + Ray). For RL
#     training, we recommend installing a prebuilt flash-attention wheel matching
#     your torch/CUDA/Python from
#     https://github.com/mjun0812/flash-attention-prebuild-wheels first.
#   - a running Ray cluster, or pass START_RAY_HEAD=1 to start one here
#   - FF data at $DATA_DIR/{train,val}.jsonl (see the pre-flight hint below)
#
# Run (single GPU, from the repo root):
#   NUM_GPUS=1 START_RAY_HEAD=1 \
#     bash litereason/experiments/flawed_fictions/lr_scripts/smoke_rl_pipeline.sh

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Base model keeps the smoke turnkey (no prior SFT needed); can also point at an
# SFT checkpoint to smoke-test the SFT -> RL handoff.
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXP_DIR/smoke_outputs/rl_pipeline_smoke_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-$EXP_DIR/data}"
REWARD_FUNC="${REWARD_FUNC:-$EXP_DIR/reward_func.py}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
WORKING_DIR="${WORKING_DIR:-$REPO_ROOT/working}"
NUM_GPUS="${NUM_GPUS:-4}"
START_RAY_HEAD="${START_RAY_HEAD:-0}"
RING_ATTN_SIZE="${RING_ATTN_SIZE:-1}"
ASSERT_PROJECTOR_SFT="${ASSERT_PROJECTOR_SFT:-1}"
ASSERT_PROJECTOR_SFT_STEPS="${ASSERT_PROJECTOR_SFT_STEPS:-1}"

# Smoke defaults: keep the same training topology as `train_full_rl.sh`, but use
# shorter prompt length + bounded sample count to reduce memory/time.
NUM_EPISODES="${NUM_EPISODES:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-48}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-512}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-2048}"
MAX_LEN="${MAX_LEN:-2560}"

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_PACKING_SAMPLES="${ENABLE_PACKING_SAMPLES:-1}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-0}"
ENABLE_ADAM_OFFLOAD="${ENABLE_ADAM_OFFLOAD:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
SAVE_STEPS="${SAVE_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-16}"

USE_WANDB_TOKEN="${USE_WANDB_TOKEN:-}"
WANDB_PROJECT="${WANDB_PROJECT:-flawed_fictions_rl}"
WANDB_GROUP="${WANDB_GROUP:-grpo_flawed_fictions_litereason_smoke}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"

# LiteReason integration toggles.
export VLLM_PLUGINS="${VLLM_PLUGINS:-litereason}"
export LITEREASON_ENABLE_VLLM="${LITEREASON_ENABLE_VLLM:-1}"
export LITEREASON_ENABLE_OPENRLHF="${LITEREASON_ENABLE_OPENRLHF:-1}"
export LITEREASON_ENABLE_PROJECTOR_SFT="${LITEREASON_ENABLE_PROJECTOR_SFT:-1}"
export LITEREASON_USE_WRAPPED_ENTRYPOINT="${LITEREASON_USE_WRAPPED_ENTRYPOINT:-1}"

# Keep projector-SFT defaults close to full run; use paper-style 25% swap for smoke.
export LITEREASON_PROJECTOR_SFT_EPOCHS="${LITEREASON_PROJECTOR_SFT_EPOCHS:-1}"
export LITEREASON_PROJECTOR_SFT_LR="${LITEREASON_PROJECTOR_SFT_LR:-1e-4}"
# Marker-expansion forward is batch_size=1 (per-sample), so projector SFT uses bs=1.
export LITEREASON_PROJECTOR_SFT_BATCH_SIZE="${LITEREASON_PROJECTOR_SFT_BATCH_SIZE:-1}"
export LITEREASON_PROJECTOR_SFT_SWAP_RATIOS="${LITEREASON_PROJECTOR_SFT_SWAP_RATIOS:-0.25}"
export LITEREASON_PROJECTOR_SFT_TOKEN_RATIO="${LITEREASON_PROJECTOR_SFT_TOKEN_RATIO:-0.2}"
export LITEREASON_PROJECTOR_SFT_DATA_RATIO="${LITEREASON_PROJECTOR_SFT_DATA_RATIO:-0.1}"
export LITEREASON_PROJECTOR_SFT_MAX_LEN="${LITEREASON_PROJECTOR_SFT_MAX_LEN:-2048}"

prep_hint="Prepare it with: python -m litereason.experiments.flawed_fictions.download_dataset --output-dir $DATA_DIR"
[[ -f "$DATA_DIR/train.jsonl" ]] || { echo "Missing $DATA_DIR/train.jsonl. $prep_hint"; exit 1; }
[[ -f "$DATA_DIR/val.jsonl" ]]   || { echo "Missing $DATA_DIR/val.jsonl. $prep_hint"; exit 1; }
[[ -f "$REWARD_FUNC" ]]          || { echo "Missing reward function: $REWARD_FUNC"; exit 1; }
mkdir -p "$WORKING_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ -z "${RAY_BIN:-}" ]]; then
  PYTHON_PATH="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
  if [[ -n "$PYTHON_PATH" ]] && [[ -x "$(dirname "$PYTHON_PATH")/ray" ]]; then
    RAY_BIN="$(dirname "$PYTHON_PATH")/ray"
  else
    RAY_BIN="ray"
  fi
fi

if [[ "$START_RAY_HEAD" == "1" ]]; then
  "$RAY_BIN" start --head --num-gpus="$NUM_GPUS"
fi

RUNTIME_ENV_JSON="$("$PYTHON_BIN" -c "
import json, os
env = {
    k: v
    for k, v in os.environ.items()
    if k.startswith('LITEREASON_') or k in {'VLLM_PLUGINS', 'LD_LIBRARY_PATH'}
}
print(json.dumps({'working_dir': '$WORKING_DIR', 'env_vars': env}))
")"

# OpenRLHF 0.10.3 uses a namespaced (dotted) CLI; flags map to args.<ns>.<name>.
COMMON_ARGS=(
  --algo.advantage.estimator group_norm
  --actor.model_name_or_path "$MODEL"
  --ckpt.output_dir "$OUTPUT_DIR/"
  --ckpt.path "$OUTPUT_DIR/ckpt/"
  --data.prompt_dataset "$DATA_DIR/train.jsonl"
  --eval.dataset "$DATA_DIR/val.jsonl"
  --data.prompt_probs 1.0
  --data.input_key prompt
  --data.label_key answer
  --reward.remote_url "$REWARD_FUNC"
  --train.num_episodes "$NUM_EPISODES"
  --train.max_epochs 1
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE"
  --rollout.micro_batch_size "$MICRO_ROLLOUT_BATCH_SIZE"
  --train.batch_size "$TRAIN_BATCH_SIZE"
  --train.micro_batch_size "$MICRO_TRAIN_BATCH_SIZE"
  --rollout.n_samples_per_prompt "$N_SAMPLES_PER_PROMPT"
  --data.max_samples "$MAX_SAMPLES"
  --rollout.max_new_tokens "$GENERATE_MAX_LEN"
  --data.max_len "$MAX_LEN"
  --ds.ring_attn_size "$RING_ATTN_SIZE"
  --actor.adam.lr 5e-7
  --algo.kl.init_coef 0.0
  --reward.normalize_enable
  --ds.param_dtype bf16
  --ds.zero_stage 2
  --actor.num_nodes 1
  --actor.num_gpus_per_node "$NUM_GPUS"
  --ref.num_nodes 0
  --ref.num_gpus_per_node 0
  --vllm.num_engines "$NUM_GPUS"
  --vllm.tensor_parallel_size 1
  --data.apply_chat_template
  --ds.attn_implementation flash_attention_2
  --ds.use_liger_kernel
  --vllm.enable_sleep
  --ds.enable_sleep
  --train.colocate_all
  --vllm.gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --ckpt.save_steps "$SAVE_STEPS"
  --eval.steps "$EVAL_STEPS"
  --ckpt.max_num 1
  --ckpt.save_hf
)

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  COMMON_ARGS+=(--vllm.enable_prefix_caching)
fi
if [[ "$ENABLE_PACKING_SAMPLES" == "1" ]]; then
  COMMON_ARGS+=(--ds.packing_samples)
fi
if [[ "$ENABLE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  COMMON_ARGS+=(--actor.gradient_checkpointing_enable)
fi
if [[ "$ENABLE_ADAM_OFFLOAD" == "1" ]]; then
  COMMON_ARGS+=(--ds.adam_offload)
fi
if [[ "$LOAD_CHECKPOINT" == "1" ]]; then
  COMMON_ARGS+=(--ckpt.load_enable)
fi

if [[ "$ENABLE_PACKING_SAMPLES" == "1" && "$RING_ATTN_SIZE" != "1" ]]; then
  echo "Invalid config: packing_samples requires ring_attn_size=1 with current LiteReason actor patch."
  exit 1
fi

USE_RAY_JOB_SUBMIT="${USE_RAY_JOB_SUBMIT:-1}"
JOB_LOG="$OUTPUT_DIR/ray_job_submit.log"
if [[ "$USE_RAY_JOB_SUBMIT" == "1" ]]; then
  # Canonical path: submit to the running Ray cluster via the dashboard job API.
  if ! "$RAY_BIN" job submit --address="$RAY_ADDRESS" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "$PYTHON_BIN" -m litereason.patches.openrlhf.train_ppo_ray \
    "${COMMON_ARGS[@]}" 2>&1 | tee "$JOB_LOG"; then
    echo "Smoke RL run failed. See log: $JOB_LOG"
    exit 1
  fi
else
  # Direct path (USE_RAY_JOB_SUBMIT=0): no Ray dashboard / job API needed. The
  # driver starts a local Ray cluster and workers inherit this process's env
  # (LITEREASON_*, VLLM_PLUGINS). Use with START_RAY_HEAD=0.
  unset RAY_ADDRESS
  if ! "$PYTHON_BIN" -m litereason.patches.openrlhf.train_ppo_ray \
    "${COMMON_ARGS[@]}" 2>&1 | tee "$JOB_LOG"; then
    echo "Smoke RL run failed. See log: $JOB_LOG"
    exit 1
  fi
fi

if [[ "$ASSERT_PROJECTOR_SFT" == "1" ]]; then
  if ! grep -Eq "\[RP\].*avg_loss" "$JOB_LOG"; then
    echo "Smoke assertion failed: projector SFT metrics not found in job log."
    echo "Set ASSERT_PROJECTOR_SFT=0 to disable this assertion."
    exit 1
  fi
fi

if [[ "$ASSERT_PROJECTOR_SFT_STEPS" == "1" ]]; then
  if ! grep -Eq "\[RP\].*steps=[1-9]" "$JOB_LOG"; then
    echo "Smoke assertion failed: projector SFT reported zero training steps."
    echo "Set ASSERT_PROJECTOR_SFT_STEPS=0 to disable this assertion."
    exit 1
  fi
fi

echo "Smoke RL run submitted/completed. Output dir: $OUTPUT_DIR"
