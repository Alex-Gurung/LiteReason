#!/usr/bin/env bash
# GRPO training script for GSM8K (2 GPUs, OpenRLHF current args).

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs_gsm8k_2gpu}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
REWARD_FUNC="${REWARD_FUNC:-$SCRIPT_DIR/reward_func.py}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
WANDB_TOKEN="${WANDB_TOKEN:-}"

# GRPO hyperparameters
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-8}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-2048}"
MAX_LEN="${MAX_LEN:-3072}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
KL_COEF="${KL_COEF:-0.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
SAVE_EVAL_STEPS="${SAVE_EVAL_STEPS:-16}"

# Start Ray cluster (if not already running)
# ray start --head --node-ip-address=0.0.0.0 --num-gpus=2

RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {k: v for k, v in os.environ.items()
       if k.startswith('LITEREASON_') or k == 'VLLM_PLUGINS'}
print(json.dumps({'working_dir': '$SCRIPT_DIR/working', 'env_vars': env}))
")

ray job submit --address="$RAY_ADDRESS" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --algo.advantage.estimator group_norm \
  --actor.model_name_or_path "$MODEL" \
  --ckpt.output_dir "$OUTPUT_DIR/" \
  --ckpt.path "$OUTPUT_DIR/ckpt/" \
  --data.prompt_dataset "$DATA_DIR/train.jsonl" \
  --eval.dataset "$DATA_DIR/val.jsonl" \
  --data.prompt_probs 1.0 \
  --data.input_key prompt \
  --data.label_key answer \
  --reward.remote_url "$REWARD_FUNC" \
  --train.num_episodes "$NUM_EPISODES" \
  --train.max_epochs "$MAX_EPOCHS" \
  --rollout.batch_size "$ROLLOUT_BATCH_SIZE" \
  --rollout.micro_batch_size "$MICRO_ROLLOUT_BATCH_SIZE" \
  --train.batch_size "$TRAIN_BATCH_SIZE" \
  --train.micro_batch_size "$MICRO_TRAIN_BATCH_SIZE" \
  --rollout.n_samples_per_prompt "$NUM_SAMPLES_PER_PROMPT" \
  --data.max_samples "$MAX_SAMPLES" \
  --rollout.max_new_tokens "$GENERATE_MAX_LEN" \
  --data.max_len "$MAX_LEN" \
  --actor.adam.lr "$LEARNING_RATE" \
  --algo.kl.init_coef "$KL_COEF" \
  --reward.normalize_enable \
  --ds.param_dtype bf16 \
  --ds.zero_stage 2 \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 2 \
  --ref.num_nodes 0 \
  --ref.num_gpus_per_node 0 \
  --vllm.num_engines 2 \
  --vllm.tensor_parallel_size 1 \
  --data.apply_chat_template \
  --ds.attn_implementation flash_attention_2 \
  --ds.use_liger_kernel \
  --logger.wandb.project gsm8k_baseline \
  --logger.wandb.group grpo_gsm8k_qwen3_17b \
  --logger.wandb.key "$WANDB_TOKEN" \
  --vllm.enable_prefix_caching \
  --vllm.enable_sleep \
  --ds.enable_sleep \
  --train.colocate_all \
  --vllm.gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --ckpt.save_steps "$SAVE_EVAL_STEPS" \
  --eval.steps "$SAVE_EVAL_STEPS" \
  --ckpt.max_num 1 \
  --ds.packing_samples \
  --actor.lr_scheduler constant \
  --ckpt.save_hf
