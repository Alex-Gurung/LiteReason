#!/usr/bin/env bash
# Skeleton GRPO training script for LiteReason.
#
# Customize MODEL, DATA_DIR, REWARD_FUNC, and OpenRLHF hyperparameters
# for your task. For a complete working example, see:
#   litereason/experiments/flawed_fictions/lr_scripts/train_full_rl.sh
#
# Prerequisites:
#   pip install -e ".[rl]"   # makes litereason importable; patching is via the wrapper entrypoint below
#   ray start --head --num-gpus=$NUM_GPUS

set -euo pipefail
set -x

# ---- Customize these ----
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs_grpo}"
DATA_DIR="${DATA_DIR:-./data}"                     # must contain train.jsonl and val.jsonl
REWARD_FUNC="${REWARD_FUNC:-./reward_func.py}"     # OpenRLHF remote reward function
NUM_GPUS="${NUM_GPUS:-1}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

# ---- LiteReason environment ----
export VLLM_PLUGINS="${VLLM_PLUGINS:-litereason}"
export LITEREASON_ENABLE_VLLM="${LITEREASON_ENABLE_VLLM:-1}"
export LITEREASON_ENABLE_OPENRLHF="${LITEREASON_ENABLE_OPENRLHF:-1}"
# max_reasoning_steps comes from the model checkpoint's config; set LITEREASON_MAX_REASONING_STEPS only to override the latent budget at test time.
# Uncomment to enable projector SFT between episodes:
# export LITEREASON_ENABLE_PROJECTOR_SFT=1

# ---- Ray runtime env ----
# Propagates LiteReason env vars (LITEREASON_*, VLLM_PLUGINS) and working_dir
# to all Ray workers and vLLM subprocesses.
RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {k: v for k, v in os.environ.items()
       if k.startswith('LITEREASON_') or k == 'VLLM_PLUGINS'}
print(json.dumps({'working_dir': '$(pwd)', 'env_vars': env}))
")

ray job submit --address="$RAY_ADDRESS" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 -m litereason.patches.openrlhf.train_ppo_ray \
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
  \
  `# ---- Training hyperparameters (adjust for your task) ----` \
  --train.num_episodes 1 \
  --train.max_epochs 1 \
  --rollout.batch_size 48 \
  --rollout.micro_batch_size 2 \
  --train.batch_size 48 \
  --train.micro_batch_size 2 \
  --rollout.n_samples_per_prompt 8 \
  --data.max_samples 10000 \
  --rollout.max_new_tokens 2048 \
  --data.max_len 4096 \
  --actor.adam.lr 5e-7 \
  --algo.kl.init_coef 0.0 \
  --reward.normalize_enable \
  --ds.param_dtype bf16 \
  \
  `# ---- Infrastructure ----` \
  --ds.zero_stage 2 \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$NUM_GPUS" \
  --ref.num_nodes 0 \
  --ref.num_gpus_per_node 0 \
  --vllm.num_engines "$NUM_GPUS" \
  --vllm.tensor_parallel_size 1 \
  --data.apply_chat_template \
  --ds.attn_implementation flash_attention_2 \
  --ds.use_liger_kernel \
  --vllm.enable_prefix_caching \
  --vllm.enable_sleep \
  --ds.enable_sleep \
  --train.colocate_all \
  --vllm.gpu_memory_utilization 0.5 \
  \
  `# ---- Checkpointing ----` \
  --ckpt.save_steps 16 \
  --eval.steps 16 \
  --ckpt.max_num 1 \
  --ds.packing_samples \
  --ckpt.save_hf
