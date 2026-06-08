#!/usr/bin/env bash
# Full LiteReason GRPO training script (paper-aligned settings for small models).
#
# Main differences from older scripts:
# 1) Uses LiteReason wrapper entrypoint:
#      python -m litereason.patches.openrlhf.train_ppo_ray
#    so periodic projector SFT is actually wired into PPOTrainer.
# 2) Uses a LiteReason SFT checkpoint as --pretrain by default.
# 3) Enables projector SFT by default via env vars.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-$EXP_DIR/outputs/sft_qwen3_4b_all}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXP_DIR/outputs/rl_qwen3_4b_litereason}"
DATA_DIR="${DATA_DIR:-$EXP_DIR/data}"
REWARD_FUNC="${REWARD_FUNC:-$EXP_DIR/reward_func.py}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
WORKING_DIR="${WORKING_DIR:-$REPO_ROOT/working}"
RING_ATTN_SIZE="${RING_ATTN_SIZE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"  # Gemma may need 1.
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-2}"      # Gemma may need 1.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-48}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-2048}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-2048}"
MAX_LEN="${MAX_LEN:-4096}"
NUM_EPISODES="${NUM_EPISODES:-30}"

USE_WANDB_TOKEN="${USE_WANDB_TOKEN:-}"
WANDB_PROJECT="${WANDB_PROJECT:-flawed_fictions_rl}"
WANDB_GROUP="${WANDB_GROUP:-grpo_flawed_fictions_litereason}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_PACKING_SAMPLES="${ENABLE_PACKING_SAMPLES:-1}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-0}"
ENABLE_ADAM_OFFLOAD="${ENABLE_ADAM_OFFLOAD:-0}"

# LiteReason integration toggles.
export VLLM_PLUGINS="${VLLM_PLUGINS:-litereason}"
export LITEREASON_ENABLE_VLLM="${LITEREASON_ENABLE_VLLM:-1}"
export LITEREASON_ENABLE_OPENRLHF="${LITEREASON_ENABLE_OPENRLHF:-1}"
export LITEREASON_ENABLE_PROJECTOR_SFT="${LITEREASON_ENABLE_PROJECTOR_SFT:-1}"
export LITEREASON_USE_WRAPPED_ENTRYPOINT="${LITEREASON_USE_WRAPPED_ENTRYPOINT:-1}"
# Ensure `python -m litereason...` resolves even without editable install.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Projector-SFT defaults aligned with implementation defaults.
export LITEREASON_PROJECTOR_SFT_EPOCHS="${LITEREASON_PROJECTOR_SFT_EPOCHS:-1}"
export LITEREASON_PROJECTOR_SFT_LR="${LITEREASON_PROJECTOR_SFT_LR:-1e-4}"
# Marker-expansion forward is batch_size=1 (per-sample), so projector SFT uses bs=1.
export LITEREASON_PROJECTOR_SFT_BATCH_SIZE="${LITEREASON_PROJECTOR_SFT_BATCH_SIZE:-1}"
export LITEREASON_PROJECTOR_SFT_SWAP_RATIOS="${LITEREASON_PROJECTOR_SFT_SWAP_RATIOS:-0.10,0.25}"
export LITEREASON_PROJECTOR_SFT_TOKEN_RATIO="${LITEREASON_PROJECTOR_SFT_TOKEN_RATIO:-0.2}"
export LITEREASON_PROJECTOR_SFT_DATA_RATIO="${LITEREASON_PROJECTOR_SFT_DATA_RATIO:-0.1}"
export LITEREASON_PROJECTOR_SFT_MAX_LEN="${LITEREASON_PROJECTOR_SFT_MAX_LEN:-2048}"

[[ -f "$DATA_DIR/train.jsonl" ]] || { echo "Missing train split: $DATA_DIR/train.jsonl"; exit 1; }
[[ -f "$DATA_DIR/val.jsonl" ]] || { echo "Missing val split: $DATA_DIR/val.jsonl"; exit 1; }
[[ -f "$REWARD_FUNC" ]] || { echo "Missing reward function: $REWARD_FUNC"; exit 1; }
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

RUNTIME_ENV_JSON="$("$PYTHON_BIN" -c "
import json, os
env = {
    k: v
    for k, v in os.environ.items()
    if k.startswith('LITEREASON_') or k in {'VLLM_PLUGINS', 'PYTHONPATH'}
}
print(json.dumps({'working_dir': '$WORKING_DIR', 'env_vars': env}))
")"

TRAIN_ARGS=(
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
  --logger.wandb.project "$WANDB_PROJECT"
  --logger.wandb.group "$WANDB_GROUP"
  --vllm.enable_sleep
  --ds.enable_sleep
  --train.colocate_all
  --vllm.gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --ckpt.save_steps 16
  --eval.steps 16
  --ckpt.max_num 1
  --ckpt.save_hf
)

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  TRAIN_ARGS+=(--vllm.enable_prefix_caching)
fi
if [[ "$ENABLE_PACKING_SAMPLES" == "1" ]]; then
  TRAIN_ARGS+=(--ds.packing_samples)
fi
if [[ "$ENABLE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  TRAIN_ARGS+=(--actor.gradient_checkpointing_enable)
fi
if [[ "$ENABLE_ADAM_OFFLOAD" == "1" ]]; then
  TRAIN_ARGS+=(--ds.adam_offload)
fi

if [[ "$LOAD_CHECKPOINT" == "1" ]]; then
  TRAIN_ARGS+=(--ckpt.load_enable)
fi

if [[ "$ENABLE_PACKING_SAMPLES" == "1" && "$RING_ATTN_SIZE" != "1" ]]; then
  echo "Invalid config: packing_samples requires ring_attn_size=1 with current LiteReason actor patch."
  exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] RAY_BIN=$RAY_BIN"
  echo "[DRY_RUN] RAY_ADDRESS=$RAY_ADDRESS"
  echo "[DRY_RUN] RUNTIME_ENV_JSON=$RUNTIME_ENV_JSON"
  echo "[DRY_RUN] Env (LiteReason + vLLM):"
  env | sort | rg -n '^(LITEREASON_|VLLM_PLUGINS|PYTHONPATH)='
  echo "[DRY_RUN] Command:"
  echo "$RAY_BIN job submit --address=\"$RAY_ADDRESS\" --runtime-env-json=\"$RUNTIME_ENV_JSON\" -- \"$PYTHON_BIN\" -m litereason.patches.openrlhf.train_ppo_ray ${TRAIN_ARGS[*]}"
  exit 0
fi

"$RAY_BIN" job submit --address="$RAY_ADDRESS" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- "$PYTHON_BIN" -m litereason.patches.openrlhf.train_ppo_ray \
  "${TRAIN_ARGS[@]}"
