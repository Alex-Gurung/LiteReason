# Flawed Fictions

Continuity error detection. Given a short story, the model predicts whether it
contains a continuity error (`\boxed{Yes}` or `\boxed{No}`). This directory is
the worked example for the full LiteReason pipeline.

For any task, the two artifacts you supply are:

1. A `{prompt, answer}` dataset (train/val/test JSONL).
2. A `reward_func.py` that scores generated responses against the gold answers.

Everything else below is the canonical pipeline, here instantiated with
`Qwen/Qwen3-4B-Instruct-2507` (the model used in the paper run).

## Pipeline

### 1. Data

Build the train/val/test JSONL of `{prompt, answer}` rows:

```bash
python -m litereason.experiments.flawed_fictions.download_dataset \
    --output-dir litereason/experiments/flawed_fictions/data \
    --prompt-file litereason/experiments/flawed_fictions/prompts/implicit_prompt.txt
```

The implicit prompt is what makes the model emit `<implicit_thought>` markers,
so use it rather than `basicprompt.txt` for the LiteReason path.

### 2. Generate reasoning traces (rejection sampled)

Start a vLLM OpenAI server for `Qwen/Qwen3-4B-Instruct-2507`, then:

```bash
python -m litereason.pipeline.generate_traces \
    --input-path litereason/experiments/flawed_fictions/data/train.jsonl \
    --output-path litereason/experiments/flawed_fictions/working/traces/train.json \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --reward-func litereason/experiments/flawed_fictions/reward_func.py \
    --k 5
```

This samples k=5 traces per example and keeps only the correct ones (scored by
`reward_func.py`). The surviving traces become the SFT data. Run it for the val
split too.

### 3. Prepare SFT data (inject markers)

Replace trace sentences with `<implicit_thought>` markers to produce the
projector training set:

```bash
python -m litereason.pipeline.prepare_sft \
    --mode ff \
    --input-dir litereason/experiments/flawed_fictions/working/traces \
    --output-dir litereason/experiments/flawed_fictions/working/sft_data/qwen3_4b/all \
    --strategies ten_percent twenty_five_percent \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507
```

### 4. Stage-1 SFT (train the Reasoning Projector)

```bash
python -m litereason.training.train_sft \
    --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml
```

This is `projector_only` on the `twenty_five_percent_masked` strategy for 1
epoch, writing the checkpoint to `outputs/sft_qwen3_4b_all`. Optionally validate
that the projector loaded and fires:

```bash
python litereason/experiments/flawed_fictions/lr_scripts/check_projector_load.py \
    --model outputs/sft_qwen3_4b_all
```

### 5. Stage-2 GRPO (LiteReason RL with periodic projector SFT)

```bash
bash litereason/experiments/flawed_fictions/lr_scripts/train_full_rl.sh
```

`MODEL` defaults to the Stage-1 checkpoint `outputs/sft_qwen3_4b_all`, and the
hyperparameters are paper-aligned. For a quick end-to-end check first, the repo
root `bash run_smoke.sh` runs the same pipeline at tiny scale.

### 6. Evaluate

```bash
python -m litereason.pipeline.evaluate \
    --model <rl checkpoint> \
    --test-file litereason/experiments/flawed_fictions/data/test.jsonl \
    --reward-func litereason/experiments/flawed_fictions/reward_func.py \
    --use-chat-template \
    --save-preds preds.jsonl
```

The evaluator scores responses with this experiment's `reward_func.py` (here the
metric is mean reward = accuracy) and reports latent-firing stats. It writes the
predictions plus a `<preds>.summary.json` with `latent_trigger_rate` and
`mean_latent_steps`.
