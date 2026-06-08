#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) script for LiteReason.

This script trains the reasoning projector on data with implicit thought markers.
The base model is frozen by default, only the projector weights are updated.

Usage:
    # Single GPU
    python -m litereason.training.train_sft --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml

    # Multi-GPU with accelerate
    accelerate launch -m litereason.training.train_sft --config litereason/experiments/flawed_fictions/configs/sft_qwen3_4b.yaml

    # With DeepSpeed ZeRO-2
    accelerate launch --config_file zero2.yaml -m litereason.training.train_sft --config sft.yaml
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import datasets
import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, SFTTrainer, TrlParser, get_kbit_device_map, get_quantization_config

from litereason import AutoModelForCausalLMWithReasoning
from litereason.stop_tokens import assistant_suffix_stop_token_ids, normalize_token_ids

from .collate import MarkerAwareCollator
from .configs import ScriptArguments, SFTConfig

log = logging.getLogger(__name__)


def _requires_post_load_use_cache(model_name_or_path: str) -> bool:
    """Return True for model families that reject `use_cache` in __init__ kwargs.

    Gemma3 uses a conditional-generation class whose constructor does not accept
    `use_cache`, even though forward/config behavior still supports cache control.
    """
    normalized = model_name_or_path.lower()
    return "gemma-3" in normalized or "gemma3" in normalized


def _preserve_generation_eos_tokens(model, tokenizer) -> None:
    """Keep existing generation EOS ids while ensuring tokenizer EOS is present.

    Gemma-family checkpoints may already carry multiple valid stop ids in
    ``generation_config.eos_token_id`` (for example ``<eos>`` and
    ``<end_of_turn>``). Overwriting that field with ``tokenizer.eos_token_id``
    collapses the stop set and can change generation behavior after save.
    """
    generation_config = getattr(model, "generation_config", None)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if generation_config is None:
        return

    existing = getattr(generation_config, "eos_token_id", None)
    merged = []

    if tokenizer_eos is not None:
        merged.append(int(tokenizer_eos))

    if isinstance(existing, int):
        merged.append(int(existing))
    elif existing is not None:
        for token_id in existing:
            if token_id is not None:
                merged.append(int(token_id))

    merged.extend(assistant_suffix_stop_token_ids(tokenizer))

    deduped = []
    seen = set()
    for token_id in merged:
        if token_id not in seen:
            deduped.append(token_id)
            seen.add(token_id)

    if not deduped:
        return

    generation_config.eos_token_id = deduped[0] if len(deduped) == 1 else deduped


def _stage_processor_artifacts(
    source_model_name_or_path: str,
    trust_remote_code: bool,
) -> dict[str, bytes]:
    """Load processor metadata from the base model and stage the required files."""
    try:
        processor = AutoProcessor.from_pretrained(
            source_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        # Broad: text-only models have no processor (ValueError) and AutoProcessor
        # surfaces several load-failure types; staging is optional, so skip.
        # Logged at warning level so a genuinely broken processor config is visible.
        log.warning("Skipping processor artifact staging: %s", exc)
        return {}

    wanted = ("preprocessor_config.json", "processor_config.json")
    artifacts: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="litereason_processor_") as tmpdir:
        tmp_path = Path(tmpdir)
        processor.save_pretrained(tmp_path)
        for name in wanted:
            file_path = tmp_path / name
            if file_path.is_file():
                artifacts[name] = file_path.read_bytes()
    return artifacts


def _repair_saved_model_dir(
    target_dir: Path,
    source_model_name_or_path: str,
    tokenizer,
    processor_artifacts: dict[str, bytes],
) -> None:
    """Backfill processor files and normalize generation_config for a saved dir."""
    if not target_dir.is_dir():
        return

    existing_eos = None
    try:
        existing_eos = GenerationConfig.from_pretrained(str(target_dir)).eos_token_id
    except OSError:
        # No (or unreadable) generation_config in the saved dir; nothing to merge.
        pass

    source_eos = None
    try:
        source_eos = GenerationConfig.from_pretrained(source_model_name_or_path).eos_token_id
    except OSError:
        # Source model has no generation_config; fall back to other eos sources.
        pass

    merged_eos = normalize_token_ids(
        getattr(tokenizer, "eos_token_id", None),
        source_eos,
        existing_eos,
        assistant_suffix_stop_token_ids(tokenizer),
    )
    if merged_eos:
        try:
            generation_config = GenerationConfig.from_pretrained(str(target_dir))
        except OSError:
            # Saved dir has no generation_config yet; start from a fresh one.
            generation_config = GenerationConfig()
        generation_config.eos_token_id = (
            merged_eos[0] if len(merged_eos) == 1 else merged_eos
        )
        generation_config.save_pretrained(str(target_dir))

    for name, payload in processor_artifacts.items():
        (target_dir / name).write_bytes(payload)


def _repair_saved_artifacts(
    output_dir: str,
    source_model_name_or_path: str,
    tokenizer,
    trust_remote_code: bool,
) -> None:
    """Repair saved model dirs so local Gemma checkpoints load in vLLM."""
    root = Path(output_dir)
    if not root.is_dir():
        return

    processor_artifacts = _stage_processor_artifacts(
        source_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    target_dirs = [root]
    target_dirs.extend(
        sorted(
            path
            for path in root.glob("checkpoint-*")
            if path.is_dir()
        )
    )
    for target_dir in target_dirs:
        _repair_saved_model_dir(
            target_dir,
            source_model_name_or_path=source_model_name_or_path,
            tokenizer=tokenizer,
            processor_artifacts=processor_artifacts,
        )
        log.info("Repaired saved artifacts in %s", target_dir)


def init_wandb(training_args: SFTConfig) -> None:
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group


def get_model(
    model_args: ModelConfig,
    training_args: SFTConfig,
    tokenizer: AutoTokenizer,
    script_args: ScriptArguments,
) -> AutoModelForCausalLMWithReasoning:
    raw_dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    torch_dtype = (
        raw_dtype
        if raw_dtype in ["auto", None]
        else getattr(torch, raw_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    desired_use_cache = not training_args.gradient_checkpointing
    set_use_cache_post_load = _requires_post_load_use_cache(model_args.model_name_or_path)

    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": getattr(model_args, "attn_implementation", None),
        "torch_dtype": torch_dtype,
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
    }
    if set_use_cache_post_load:
        log.info("Skipping constructor `use_cache` for Gemma3; setting it on config after load.")
    else:
        model_kwargs["use_cache"] = desired_use_cache

    model = AutoModelForCausalLMWithReasoning.from_pretrained(
        model_args.model_name_or_path,
        tokenizer=tokenizer,
        num_reasoning_layers=script_args.num_reasoning_layers,
        max_reasoning_steps=script_args.max_reasoning_steps,
        **model_kwargs,
    )
    if set_use_cache_post_load and hasattr(model, "config"):
        model.config.use_cache = desired_use_cache

    return model


def freeze_base_model(model: AutoModelForCausalLMWithReasoning) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.reasoning_projector.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )


def load_datasets(script_args: ScriptArguments) -> tuple:
    if script_args.dataset_dir is None:
        raise ValueError("dataset_dir must be specified")

    dataset_dir = script_args.dataset_dir
    dataset_name = script_args.dataset_name or "train"

    # Inclusive strategy support:
    # - "<name>"          includes no_masking + ... + <name>
    # - "<name>_masked"   includes ten_percent + ... + <name> (no no_masking)
    strategy_hierarchy_all = [
        "no_masking",
        "ten_percent",
        "twenty_five_percent",
        "fifty_percent",
        "seventy_five_percent",
        "full_masking",
    ]
    strategy_hierarchy_masked = [
        "ten_percent",
        "twenty_five_percent",
        "fifty_percent",
        "seventy_five_percent",
        "full_masking",
    ]

    included_strategies = None
    if dataset_name.endswith("_masked"):
        target = dataset_name[: -len("_masked")]
        if target not in strategy_hierarchy_masked:
            raise ValueError(
                "Invalid masked inclusive dataset_name. "
                f"Got {dataset_name!r}; expected <strategy>_masked where strategy in "
                f"{strategy_hierarchy_masked}"
            )
        selected_idx = strategy_hierarchy_masked.index(target)
        included_strategies = strategy_hierarchy_masked[: selected_idx + 1]
        log.info(
            "Using masked inclusive strategy: %s includes %s",
            dataset_name,
            included_strategies,
        )
    elif dataset_name in strategy_hierarchy_all:
        selected_idx = strategy_hierarchy_all.index(dataset_name)
        included_strategies = strategy_hierarchy_all[: selected_idx + 1]
        log.info(
            "Using inclusive strategy: %s includes %s",
            dataset_name,
            included_strategies,
        )

    if included_strategies is not None:
        train_datasets = []
        val_datasets = []
        for strategy in included_strategies:
            train_file = f"{dataset_dir}/train_{strategy}.jsonl"
            val_file = f"{dataset_dir}/val_{strategy}.jsonl"
            if not os.path.exists(train_file):
                raise FileNotFoundError(
                    f"Missing required training file for strategy '{strategy}': {train_file}"
                )
            train_datasets.append(
                datasets.load_dataset("json", data_files=train_file, split="train")
            )
            if os.path.exists(val_file):
                val_datasets.append(
                    datasets.load_dataset("json", data_files=val_file, split="train")
                )

        if not train_datasets:
            raise ValueError(
                f"No training datasets loaded for dataset_name={dataset_name!r} from {dataset_dir}"
            )
        train_dataset = (
            train_datasets[0]
            if len(train_datasets) == 1
            else datasets.concatenate_datasets(train_datasets)
        )
        val_dataset = (
            None
            if not val_datasets
            else val_datasets[0]
            if len(val_datasets) == 1
            else datasets.concatenate_datasets(val_datasets)
        )
    else:
        # Single dataset
        requested_train_file = f"{dataset_dir}/train_{dataset_name}.jsonl"
        train_file = requested_train_file
        val_file = f"{dataset_dir}/val_{dataset_name}.jsonl"

        if not os.path.exists(train_file):
            # Try without prefix. This masks a dataset_name mismatch, so be loud
            # about which file we actually chose vs. what was requested.
            fallback_train = f"{dataset_dir}/train.jsonl"
            fallback_val = f"{dataset_dir}/val.jsonl"
            if os.path.exists(fallback_train):
                log.warning(
                    "Requested train file %r not found; falling back to %r "
                    "(this may indicate a --dataset_name mismatch).",
                    requested_train_file, fallback_train,
                )
                train_file = fallback_train
                val_file = fallback_val
            else:
                raise FileNotFoundError(
                    f"No training data found in {dataset_dir!r}: expected "
                    f"{requested_train_file!r} (for --dataset_name={dataset_name!r}) "
                    f"or {fallback_train!r}."
                )
        else:
            log.info("Loading training data from %r", train_file)

        train_dataset = datasets.load_dataset("json", data_files=train_file, split="train")
        val_dataset = (
            datasets.load_dataset("json", data_files=val_file, split="train")
            if os.path.exists(val_file)
            else None
        )

    train_dataset = train_dataset.shuffle(seed=42)
    if val_dataset:
        val_dataset = val_dataset.shuffle(seed=42)
        # Limit validation size for efficiency
        if script_args.max_val_samples is not None and len(val_dataset) > script_args.max_val_samples:
            val_dataset = val_dataset.select(range(script_args.max_val_samples))

    return train_dataset, val_dataset


def main(script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    log.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    log.info(f"Model parameters: {model_args}")
    log.info(f"Script parameters: {script_args}")
    log.info(f"Training parameters: {training_args}")

    # Marker expansion is a per-sample forward (it raises
    # ValueError("Marker expansion supports batch_size=1...") otherwise), so the
    # train/eval batch size must be 1. Assert up front rather than crashing
    # mid-run. Use gradient_accumulation_steps to get a larger effective batch.
    if training_args.per_device_train_batch_size != 1:
        raise ValueError(
            "LiteReason SFT requires per_device_train_batch_size == 1 because "
            "marker expansion is a per-sample forward (batch_size=1 only). "
            f"Got per_device_train_batch_size={training_args.per_device_train_batch_size}. "
            "Increase gradient_accumulation_steps for a larger effective batch size."
        )
    if training_args.do_eval and training_args.per_device_eval_batch_size != 1:
        raise ValueError(
            "LiteReason SFT requires per_device_eval_batch_size == 1 because "
            "marker expansion is a per-sample forward (batch_size=1 only). "
            f"Got per_device_eval_batch_size={training_args.per_device_eval_batch_size}."
        )

    # Check for checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        log.info(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Initialize WandB
    if "wandb" in training_args.report_to:
        init_wandb(training_args)

    # Load tokenizer (honor the same revision / trust_remote_code as the model,
    # so a revision-pinned or custom checkpoint does not get a mismatched tokenizer).
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = get_model(model_args, training_args, tokenizer, script_args)

    # Freeze base model if requested
    if script_args.projector_only:
        log.info("Freezing base model, training projector only")
        freeze_base_model(model)

    # Load datasets
    train_dataset, val_dataset = load_datasets(script_args)
    log.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        log.info(f"Val dataset size: {len(val_dataset)}")

    # Create collator
    collator = MarkerAwareCollator(
        tokenizer,
        require_marker=script_args.require_implicit_markers,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # Train
    log.info("*** Starting training ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save model
    log.info("*** Saving model ***")
    _preserve_generation_eos_tokens(trainer.model, tokenizer)
    trainer.save_model(training_args.output_dir)
    log.info(f"Model saved to {training_args.output_dir}")

    # Create model card
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        _repair_saved_artifacts(
            training_args.output_dir,
            source_model_name_or_path=model_args.model_name_or_path,
            tokenizer=tokenizer,
            trust_remote_code=model_args.trust_remote_code,
        )

    # Evaluate
    if training_args.do_eval and val_dataset:
        log.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push to hub
    if training_args.push_to_hub:
        log.info("Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
