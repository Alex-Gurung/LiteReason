# GRPO Training for GSM8K

Minimal setup for running GRPO on GSM8K with OpenRLHF.

## Files

- `prepare_dataset.py` - Builds OpenRLHF-format `data/train.jsonl` and `data/val.jsonl`
- `reward_func.py` - Reward function using `math_verify.verify` for correctness
- `train_grpo.sh` - 2-GPU training script (current OpenRLHF args)
- `eval_vllm_server.py` - Scores a checkpoint against a running vLLM OpenAI server

## Setup

1. Install dependencies:
```bash
python3 -m pip install -U openrlhf ray datasets math-verify wandb
```

2. Prepare dataset:
```bash
# Auto mode:
# - uses local raw files if data/gsm8k/train.jsonl + data/gsm8k/val.jsonl exist
# - otherwise downloads openai/gsm8k and creates a deterministic split
python3 prepare_dataset.py --source auto

# Alternative source: reasoning-machines/gsm-hard
# (writes separate files so you can keep gsm8k data untouched)
# default is deterministic 80/10/10 train/val/test split
# uses prompt instruction: \boxed{<final_number>}
python3 prepare_dataset.py --source hf_gsm_hard --output-dir data_gsm_hard
```

Note: default `hf` mode uses a deterministic split of `openai/gsm8k` `train` (not the official `test` split).
For `hf_gsm_hard`, the prep script also writes `test.jsonl`.

3. Start Ray (2 GPUs):
```bash
ray stop --force
ray start --head --node-ip-address=0.0.0.0 --num-gpus=2
```

4. Run training:
```bash
bash train_grpo.sh
```

To train on the GSM-hard split instead, prepare it with
`prepare_dataset.py --source hf_gsm_hard --output-dir data_gsm_hard` (see above)
and point training at it with `DATA_DIR="$PWD/data_gsm_hard" bash train_grpo.sh`.

## Evaluation

If a vLLM OpenAI server is running on `http://127.0.0.1:8000/v1`, score a
checkpoint against it with:
```bash
python eval_vllm_server.py
```

Or use the **generic evaluator** (local vLLM, no server); it scores with this
experiment's `reward_func.py` and additionally reports whether latent reasoning
fired (`latent_trigger_rate`, `mean_latent_steps`):
```bash
LITEREASON_ENABLE_VLLM=1 VLLM_PLUGINS=litereason \
python -m litereason.pipeline.evaluate \
    --model <checkpoint> --test-file data_gsm_hard/test.jsonl \
    --reward-func litereason/experiments/gsm8k/reward_func.py \
    --use-chat-template --save-preds preds.jsonl
```

Key defaults in `eval_vllm_server.py`:
- `model-name=Qwen/Qwen3-1.7B`
- `temperature=1.0`
- `max-new-tokens=2048`
- `concurrency=32`

Override eval defaults (example):
```bash
python eval_vllm_server.py \
  --model-name Qwen/Qwen3-1.7B \
  --temperature 1.0 \
  --max-new-tokens 2048 \
  --concurrency 32
```

## 2-GPU Defaults (`train_grpo.sh`)

- `num_episodes=5`
- `rollout_batch_size=128`
- `train_batch_size=128`
- `n_samples_per_prompt=8`
- `actor_learning_rate=5e-7`
- `param_dtype=bf16`
- HF checkpoints enabled (`--save_hf_ckpt`, `--ckpt_path`, tags/steps configured)
- W&B enabled (`--use_wandb`, `--wandb_project`, `--wandb_group`)

All key values are overrideable via environment variables, e.g.:
```bash
NUM_EPISODES=5 ROLLOUT_BATCH_SIZE=64 TRAIN_BATCH_SIZE=64 bash train_grpo.sh

# Run against gsm-hard-prepared files
DATA_DIR="$PWD/data_gsm_hard" bash train_grpo.sh
```
