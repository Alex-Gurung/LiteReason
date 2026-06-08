#!/usr/bin/env bash
# One-command smoke test for LiteReason. Run from the repo root:
#
#   bash run_smoke.sh
#
# On a single GPU it runs the full path end-to-end:
#   1. prepares the Flawed Fictions data (if missing)
#   2. a short GRPO run -- validates the actor/vLLM patches and the periodic
#      projector-SFT 10/25% mixture (the script asserts it actually trained)
#   3. evaluates the trained checkpoint with the generic pipeline -- validates the
#      dataset-agnostic evaluator and reports whether latent reasoning fired
#
# Install the RL stack once before running. For RL training, we recommend
# installing a prebuilt flash-attention wheel matching your torch/CUDA/Python from
# https://github.com/mjun0812/flash-attention-prebuild-wheels first.
#   uv pip install <matching flash_attn prebuilt wheel>   # e.g. cu128torch2.x-cp312
#   uv pip install -e ".[rl]"
#   python -m spacy download en_core_web_sm
#
# By default the GRPO run submits to a Ray cluster via the dashboard job API
# (OpenRLHF's canonical path). If the Ray dashboard fails to start on your host
# (a known Ray 2.45+ issue, common when /dev/shm is small -- e.g. the 64 MB Docker
# default starves the dashboard under load), use the direct path instead, which
# starts a local Ray cluster in-process and needs no dashboard:
#   USE_RAY_JOB_SUBMIT=0 START_RAY_HEAD=0 bash run_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
FF="litereason/experiments/flawed_fictions"
DATA_DIR="${DATA_DIR:-$FF/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/smoke_outputs/run_$(date +%Y%m%d_%H%M%S)}"

# --- pre-flight ---------------------------------------------------------------
if ! "$PYTHON_BIN" -c "import vllm, openrlhf, ray, spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
  echo "Missing the RL dependencies. For RL training, we recommend installing a prebuilt"
  echo "flash-attention wheel matching your torch/CUDA/Python from https://github.com/mjun0812/flash-attention-prebuild-wheels first, then:"
  echo "  uv pip install <matching flash_attn prebuilt wheel> && uv pip install -e \".[rl]\""
  echo "  python -m spacy download en_core_web_sm"
  exit 1
fi

# --- pre-flight: verify the monkeypatch hooks still match the installed libs ---
# A vLLM/OpenRLHF bump can move the private internals the plugin/patches target.
# These static checks fail in seconds; without them a moved hook only surfaces
# minutes into the GRPO run.
echo ">>> Verifying patch hook points against the installed vLLM / OpenRLHF"
if ! "$PYTHON_BIN" check_vllm_hooks.py; then
  echo "vLLM hook check failed: litereason/vllm_plugin.py needs updating for this vLLM (see above)."
  exit 1
fi
if ! LITEREASON_ENABLE_OPENRLHF= "$PYTHON_BIN" check_openrlhf_hooks.py; then
  echo "OpenRLHF hook check failed: litereason/patches/openrlhf/ needs updating for this openrlhf (see above)."
  exit 1
fi

# --- 1. data ------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
  echo ">>> Preparing Flawed Fictions data in $DATA_DIR"
  # Implicit prompt so the model emits <implicit_thought> markers -> latent reasoning
  # actually fires during rollout + eval (the basic prompt never exercises that path).
  "$PYTHON_BIN" -m litereason.experiments.flawed_fictions.download_dataset \
    --output-dir "$DATA_DIR" --prompt-file "$FF/prompts/implicit_prompt.txt"
fi

# --- 2. RL smoke (the lr_scripts smoke, single GPU) ---------------------------
echo ">>> Running the RL smoke -> $OUTPUT_DIR"
NUM_GPUS="${NUM_GPUS:-1}" START_RAY_HEAD="${START_RAY_HEAD:-1}" \
  OUTPUT_DIR="$OUTPUT_DIR" DATA_DIR="$DATA_DIR" \
  bash "$FF/lr_scripts/smoke_rl_pipeline.sh"

# --- 3. evaluate the trained checkpoint ---------------------------------------
# OpenRLHF (--save_hf_ckpt) writes an HF checkpoint under OUTPUT_DIR; locate it by
# its config.json (excluding the DeepSpeed ckpt/ dir).
CKPT="$(find "$OUTPUT_DIR" -maxdepth 3 -name config.json -not -path '*/ckpt/*' 2>/dev/null | head -1 | xargs -r dirname)"
if [[ -z "$CKPT" || ! -f "$CKPT/config.json" ]]; then
  echo "Could not find the saved HF checkpoint under $OUTPUT_DIR; skipping eval."
  echo "Once you locate it, run: python -m litereason.pipeline.evaluate --model <ckpt> ..."
  exit 0
fi

echo ">>> Evaluating $CKPT"
LITEREASON_ENABLE_VLLM=1 VLLM_PLUGINS=litereason \
  "$PYTHON_BIN" -m litereason.pipeline.evaluate \
    --model "$CKPT" \
    --test-file "$DATA_DIR/val.jsonl" \
    --reward-func "$FF/reward_func.py" \
    --use-chat-template --num-samples 1 --limit 8 \
    --save-preds "$OUTPUT_DIR/eval_preds.jsonl"

echo ""
echo "=== SMOKE COMPLETE ==="
echo "Eval summary (metric + latent-firing stats):"
cat "$OUTPUT_DIR/eval_preds.summary.json"
