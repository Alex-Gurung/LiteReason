# Experiments

Experiment scripts for reproducing paper results.

- `flawed_fictions/`: Continuity-error detection in short stories (binary Yes/No with `\boxed{}` answers). The narrative task and the worked example for the full LiteReason pipeline (data, trace generation, Stage-1 SFT, Stage-2 GRPO, evaluation). See its [README](flawed_fictions/README.md) for the single canonical pipeline.
- `gsm8k/`: GRPO on GSM8K / GSM-Hard (the math task). `prepare_dataset.py --source hf_gsm_hard` builds the GSM-Hard split; includes a `math_verify`-based reward, trace generation/filtering, and CoLaR/CoCoNut export.
