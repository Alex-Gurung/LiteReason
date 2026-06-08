# LiteReason

**Lightweight Latent Reasoning for Narrative Tasks**

[Paper (arXiv)](https://arxiv.org/abs/2512.02240) &nbsp;·&nbsp; [Project page](https://alexgurung.me/litereason/)

LiteReason implements a latent reasoning architecture where explicit chain-of-thought tokens are replaced with continuous hidden-state embeddings via a learned projector. In order to combine this method into RL training and fast inference, we patch OpenRLHF and VLLM, and provide a wrapper around the transformers AutoModelForCausalLM class.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training) (run on your own dataset, SFT, GRPO)
- [Evaluation](#evaluation)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Integration](#integration) (vLLM plugin, OpenRLHF monkeypatch)
- [Token Patterns](#token-patterns)
- [Citation](#citation)

## Overview

Instead of generating explicit reasoning text like:
```
Let me think step by step... First, I need to consider...
```

LiteReason uses special markers that trigger latent reasoning:
```
It seems likely that the real culprit was wearing red gloves. <implicit_thought>3</implicit_thought> The answer is...
```

The `<implicit_thought>3</implicit_thought>` marker triggers 3 iterations of latent reasoning through a learned projector, producing continuous embeddings instead of discrete tokens.

## Installation

We use `uv` for installs. In a `uv` virtualenv, a tool that shells out to `pip`
may need `uv pip install pip` first.

### Basic Installation (Core Library)
```bash
uv pip install -e .
```

### Training (SFT/Fine-tuning)
```bash
uv pip install -e ".[training]"
# projector-SFT sentence splitting needs the spaCy English model:
python -m spacy download en_core_web_sm
```

### Full RL Training (OpenRLHF + vLLM)
For RL training, we recommend installing a prebuilt flash-attention wheel matching your torch/CUDA/Python from https://github.com/mjun0812/flash-attention-prebuild-wheels first.
```bash
# Step 1: flash-attn from a prebuilt wheel for YOUR stack (example: cu128 / torch 2.x / cp312)
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl

# Step 2: LiteReason + RL extras (openrlhf[vllm,liger,ring] + ray; reuses the wheel above)
uv pip install -e ".[rl]"
```

### Flawed Fictions Experiments
The SFT / trace-generation path needs spaCy (sentence splitting) and `openai`
(used by `litereason.pipeline.generate_traces`); both come in via the extras:
```bash
uv pip install -e ".[all]"
python -m spacy download en_core_web_sm
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.8.0 (required for vLLM 0.15+)
- transformers >= 4.47.0 (compatible with v5+)

Validated stack (matches `pyproject.toml`): vLLM 0.19.1, torch 2.10, openrlhf 0.10.3, transformers 5.x. The lower bounds above still hold; the versions here are what the patches were last verified against.

## Quick Start

### Using the Model (Any HF Causal LM)

```python
from litereason import AutoModelForCausalLMWithReasoning
from transformers import AutoTokenizer

# Load tokenizer (any causal LM tokenizer works)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Load and patch any causal LM
model = AutoModelForCausalLMWithReasoning.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer=tokenizer,
    num_reasoning_layers=3,
)

# The model automatically handles <implicit_thought> markers during forward pass
text = "Question? <implicit_thought>3</implicit_thought> Answer."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

To resume from a LiteReason checkpoint that already contains trained
`reasoning_projector.*` weights, point `from_pretrained(...)` at the saved
directory. Projector weights are auto-loaded from safetensors (including sharded
checkpoints):

```python
model = AutoModelForCausalLMWithReasoning.from_pretrained(
    "./checkpoints/litereason_sft",
    tokenizer=tokenizer,
)
```

## Training

LiteReason uses a two-stage training pipeline:
1. **Stage 1: SFT** - Train the reasoning projector on data with implicit thought markers
2. **Stage 2: RL** - Fine-tune with GRPO (and, optionally, periodic projector SFT)

### Run on your own dataset

You supply two things:

1. **Train/val/test JSONL** where each row is `{"prompt": <str>, "answer": <str>}`. `prompt` is the input handed to the model; `answer` is the gold label string passed to your reward function as `labels`.
2. **A `reward_func.py`** that scores generated responses against those labels.

**Reward function contract** (OpenRLHF convention). Your file must expose a callable:

```python
def reward_func(queries, prompts, labels, **kwargs) -> dict:
    # returns {"rewards": <list/tensor>, "scores": <list/tensor>}
    ...
```

The surprising part: the first positional arg `queries` is the model's
generated responses, not the inputs. `prompts` are the inputs and `labels`
are the gold answers (your `answer` strings). Score by zipping `queries`
with `labels`. See `litereason/experiments/gsm8k/reward_func.py` for a clean
math-verify example, and `litereason/experiments/flawed_fictions/reward_func.py`
for another.

**Keep the three answer formats in agreement.** The SFT target answer, the RL
`answer` label, and what your `reward_func` parses must all use the same
format. For example, gsm8k uses `\boxed{N}` in all three.

**Run order:**

1. Stage-1 SFT, to initialize the Reasoning Projector from your traces (see below).
2. GRPO via the generic `litereason/training/run_grpo.sh` (set `MODEL`, `DATA_DIR`, `REWARD_FUNC`).
3. Evaluate with `python -m litereason.pipeline.evaluate`.

**Latent-step budget (`LITEREASON_MAX_REASONING_STEPS`).** You normally do not set
this. The number of latent steps per marker is read from the checkpoint's
`litereason_max_reasoning_steps` config attribute (set during SFT), and every path
- HF training, the vLLM rollout, and the dummy-token stripper - uses that one
value. The env var only sets the budget when starting from a base model that has
no config value; if it disagrees with a loaded checkpoint it is ignored (with a
warning), so it can no longer silently desync training and rollout.

### Preparing Training Data

For your own dataset, use `--mode jsonl`. Each input row is
`{"prompt": <str>, "answer": <str>, "trace": <str>}` (or `"steps": [<str>, ...]`
instead of `trace`): `prompt`/`answer` use the same schema as the RL stage and
reward function, and `trace`/`steps` is the reasoning text whose sentences are
replaced with `<implicit_thought>` markers. The default `--mode ff` is the
Flawed Fictions format (`train.json`/`val.json`/`test.json` arrays), not JSONL.

```bash
python -m litereason.pipeline.prepare_sft \
    --mode jsonl \
    --input-dir ./data \
    --output-dir ./processed \
    --strategies twenty_five_percent fifty_percent \
    --tokenizer Qwen/Qwen2.5-7B-Instruct
```

### Stage 1: Supervised Fine-Tuning (SFT)

Train the reasoning projector while freezing the base model:

```bash
# Single GPU
python -m litereason.training.train_sft \
    --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml

# Multi-GPU with accelerate
accelerate launch -m litereason.training.train_sft \
    --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml
```

The SFT trainer:
- Freezes base model weights by default (`--projector-only`)
- Uses marker-aware collation (loss computed only after markers)
- Supports hierarchical masking strategies (ten_percent, twenty_five_percent, etc.)

### Stage 2: Reinforcement Learning (RL)

Fine-tune with GRPO using OpenRLHF:

```bash
# Start Ray cluster
ray start --head --num-gpus=4

# Run GRPO training
bash litereason/experiments/flawed_fictions/lr_scripts/train_full_rl.sh
```

The example scripts set `VLLM_PLUGINS`, `LITEREASON_ENABLE_VLLM`, and
`LITEREASON_ENABLE_OPENRLHF` themselves (via `:-` defaults), so you do not
need to export them by hand. Periodic projector SFT is off by default; to
turn it on, set `LITEREASON_ENABLE_PROJECTOR_SFT=1` before launching (the
`lr_scripts/smoke_rl_pipeline.sh` example enables it).

See `lr_scripts/train_full_rl.sh` and the
[flawed_fictions README](litereason/experiments/flawed_fictions/README.md)
for the full worked training pipeline.

## Evaluation

### Evaluating a Trained Checkpoint

```bash
python -m litereason.pipeline.evaluate \
    --model ./outputs/sft_projector \
    --test-file ./data/test.jsonl \
    --reward-func path/to/reward_func.py \
    --num-samples 10 \
    --save-preds preds.jsonl
```

For instruct models with a chat template:

```bash
python -m litereason.pipeline.evaluate \
    --model Qwen/Qwen2.5-7B-Instruct \
    --test-file ./data/test.jsonl \
    --reward-func path/to/reward_func.py \
    --use-chat-template \
    --save-preds preds.jsonl
```

### Summarizing Results

`evaluate` writes `<preds>.summary.json` alongside the predictions, with the metric
(mean reward), mean completion tokens, and latent-firing stats
(`latent_trigger_rate`, `mean_latent_steps`, `n_with_latents`).

### Generating Reasoning Traces

Generate explicit reasoning traces from a teacher model, then use them to prepare SFT data:

```bash
# Start a vLLM server, then:
python -m litereason.pipeline.generate_traces \
    --input-path ./data/train.jsonl \
    --output-path ./traces/train.json \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --reward-func path/to/reward_func.py \
    --k 5
```

## Architecture

### Model Structure (Patched Causal LM)

```
<Your HF CausalLM>
|-- (original modules...)
`-- reasoning_projector: ModuleList
    |-- MLP[0]
    |-- MLP[1]
    `-- ...
```

The reasoning projector is a stack of MLP modules. When possible, LiteReason clones the
architecture's native final-block MLP module (e.g. Llama/Mistral/Gemma MLP).

### Forward Pass

1. **Token Pattern Detection**: Scans input for `<implicit_thought>N</implicit_thought>` patterns
2. **Section Processing**: For each section ending with a marker:
   - Embed tokens up to and including the marker
   - Generate N reasoning embeddings through the projector
   - Each reasoning step uses context from all previous embeddings
3. **Mask Application**: Filter reasoning embeddings from output using `given_is_reasoning_embedding_mask`

### Shadow Forward (for RL)

For RL training, `shadow_forward()` returns `(embeddings, mask)` for efficient batched processing:
- Uses KV caching for speed
- Runs with `torch.no_grad()` since gradients flow through the main forward only

`shadow_forward` runs entirely under `torch.no_grad()`, so during RL it produces
the latent embeddings as constants: there is no gradient through it, and the KV
cache is purely a speed optimization (it cannot bias a gradient). The Reasoning
Projector is not updated during RL; it is trained only by the periodic projector
SFT, whose latent rollout runs under `enable_grad` and backpropagates through the
(graph-connected) cached states exactly.

## Project Structure

```
litereason/
|-- litereason/                        # Python package
|   |-- __init__.py
|   |-- causal_lm_with_reasoning.py    # Generic HF causal LM patch + factory
|   |-- reasoning_core.py             # Shared reasoning forward / masking logic
|   |-- token_utils.py                 # Marker pattern detection utilities
|   |-- vllm_plugin.py                 # vLLM plugin (rollout-time latent steps)
|   |-- training/                      # Training utilities
|   |   |-- train_sft.py               # SFT trainer script
|   |   |-- configs.py                 # Training configuration classes
|   |   `-- collate.py                 # Marker-aware data collation
|   |-- patches/                       # Integration helpers (OpenRLHF, vLLM)
|   |-- pipeline/                       # Task-agnostic data/eval pipeline
|   |   |-- evaluate.py                 # vLLM-based evaluation
|   |   |-- generate_traces.py         # Reasoning trace generation
|   |   |-- prepare_sft.py             # SFT data preparation with markers
|   |   `-- metrics.py                 # Shared statistical utilities
|   `-- experiments/                   # Experiment scripts
|       `-- flawed_fictions/
|           |-- download_dataset.py    # Download HF dataset for GRPO
|           |-- reward_func.py         # GRPO reward function
|           |-- configs/
|           |   `-- sft_qwen3_4b.yaml  # Stage-1 SFT training config
|           `-- lr_scripts/            # LiteReason RL launch + smoke scripts
`-- pyproject.toml
```

## Integration

### vLLM Plugin

LiteReason registers a [vLLM general plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) via a Python entry point in `pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
litereason = "litereason.vllm_plugin:register"
```

When `LITEREASON_ENABLE_VLLM=1` is set, vLLM discovers and activates the plugin at startup. During decoding, the plugin monitors the generated token sequence. When it detects a complete `<implicit_thought>N</implicit_thought>` marker at the end of the sequence, it:

1. Parses the complexity `N` from the marker
2. Runs the reasoning projector on the current hidden state to produce a latent embedding
3. Injects that embedding into the KV cache (skipping normal sampling) and emits a dummy token ID
4. Repeats for `N` latent steps, then resumes normal autoregressive decoding

This works anywhere `litereason` is pip-installed (no special working directory required).

**Limitations:**
- Requires `vllm>=0.15.0` (v1 runner + plugin system); last validated on vLLM 0.19.1.
- No speculative decoding (assumes 1-token decode steps).
- No pipeline parallel (PP=1). Tensor parallel is fine.
- Marker detection is decoding only (markers in the middle of a prompt are not expanded).

### OpenRLHF Monkeypatch

LiteReason's RL training runs through a thin wrapper entrypoint that monkeypatches OpenRLHF's actor classes before training starts. Install the package once, then run the entrypoint from anywhere:

```bash
pip install -e ".[rl]"                    # makes litereason importable like any package
python -m litereason.patches.openrlhf.train_ppo_ray [OpenRLHF args...]
```

The wrapper calls `patch_openrlhf_actor()` in the driver, then delegates to `openrlhf.cli.train_ppo_ray`. Each Ray worker re-applies the patch from `LiteReasonPolicyModelActor.__init__` (idempotent), so the integration is active in the driver and every worker. This relies only on the package being installed in the environment Ray launches workers in - no `PYTHONPATH` edits and no need to run from a particular directory. The example script `litereason/experiments/flawed_fictions/lr_scripts/train_full_rl.sh` invokes this entrypoint directly.

**What gets patched** (`patch_openrlhf_actor()` in `litereason/patches/openrlhf/actor_patch.py`):

| Target | Patch | Purpose |
|--------|-------|---------|
| `openrlhf.models.actor.AutoModelForCausalLM` | Replaced with `AutoModelForCausalLMWithReasoning` | Actor loads LiteReason-enabled model |
| `Actor.__init__` | Wrapped to call `model.set_tokenizer()` | Marker detection needs the tokenizer |
| `Actor.forward()` | Wrapped with per-sample `shadow_forward()` + batched masked forward | Marker expansion during PPO/GRPO training |
| `SingleTurnAgentExecutor.execute` | Wrapped to strip dummy vLLM tokens | Training sees marker tokens only, not latent-step dummies |

`LITEREASON_*` and `VLLM_PLUGINS` reach the Ray workers through the runtime-env block the training scripts pass to `ray job submit` (or, in the direct-launch path, by inheritance from the driver process).

### Verifying patch compatibility

The vLLM plugin and OpenRLHF patches hook **private internals by name**, so a vLLM or
OpenRLHF upgrade can move them out from under the hooks. Two static checkers verify the
hook points against whatever is installed; run them after bumping either library:

```bash
python check_vllm_hooks.py       # GPUModelRunner / DefaultModelLoader / Gemma3 hook points
python check_openrlhf_hooks.py   # Actor, DeepspeedStrategy, the Ray-actor classes, helpers
```

A clean `OK:` means the targets are intact; otherwise the checker prints exactly which
class, method, or parameter moved. `run_smoke.sh` runs both as a preflight check.

### Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `LITEREASON_ENABLE_VLLM` | `0` | Enable vLLM plugin |
| `VLLM_PLUGINS` | (none) | Set to `litereason` for vLLM to discover the plugin |
| `LITEREASON_ENABLE_OPENRLHF` | `1` | Enable OpenRLHF monkeypatches |
| `LITEREASON_ENABLE_PROJECTOR_SFT` | `0` | Periodic projector SFT during RL |
| `LITEREASON_PROJECTOR_SFT_REWARD_FILTER` | `1` | Train periodic projector SFT only on positive-reward traces (rejection sampling) |
| `LITEREASON_OPENRLHF_INIT_PROJECTOR` | `0` | Init projector from pretrained after Actor load |
| `LITEREASON_MAX_REASONING_STEPS` | (from checkpoint config) | Latent steps per marker. Read from the checkpoint's `litereason_max_reasoning_steps`; only set to seed the value for a base model (ignored if it disagrees with a loaded checkpoint) |

See also:
- [litereason/patches/openrlhf/README.md](litereason/patches/openrlhf/README.md): OpenRLHF patch details

## Token Patterns

The `<implicit_thought>N</implicit_thought>` markers are tokenized differently depending on context (e.g., `".<implicit_thought>"` can merge into a single token in some BPE vocabularies). LiteReason auto-detects multiple valid start/end tokenizations from the tokenizer at runtime.
If pattern detection fails for your tokenizer, LiteReason will raise an error; in that case, add the tags as special tokens or use a tokenizer/vocab that can represent them.

Pattern structure: `<start_sequence> [0-3 gap tokens] <end_sequence>`
- Start sequence: tokenizer-dependent encoding of `<implicit_thought>` (plus any merged prefix character)
- Gap tokens: 1-3 tokens encoding the complexity number (e.g. `3`)
- End sequence: tokenizer-dependent encoding of `</implicit_thought>` (plus any merged suffix character)

## Citation

If you use LiteReason, please cite the paper:

```bibtex
@article{gurung2026lightweightlatentreasoning,
  title     = {Lightweight Latent Reasoning for Narrative Tasks},
  author    = {Alexander Gurung and Esmeralda S. Whitammer and Mirella Lapata},
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.02240}
}
```
