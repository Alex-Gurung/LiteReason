#!/usr/bin/env python3
"""Prepare GSM8K data for OpenRLHF GRPO training.

Outputs `train.jsonl` and `val.jsonl` in OpenRLHF format with keys:
  - prompt
  - answer

Data source modes:
  1) Local raw JSONL files (default auto-detected): `data/gsm8k/train.jsonl`,
     `data/gsm8k/val.jsonl`, each with `question` and `answer`.
  2) HuggingFace `openai/gsm8k` train split with deterministic random split.
  3) HuggingFace `reasoning-machines/gsm-hard` train split with deterministic
     80/10/10 train/val/test split by default.
"""


import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

INSTRUCTION_INTEGER = (
    "Solve the problem step by step. End your solution with the final answer as "
    "\\boxed{<final_integer>}."
)
INSTRUCTION_NUMBER = (
    "Solve the problem step by step. End your solution with the final answer as "
    "\\boxed{<final_number>}."
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def extract_gold_answer(answer_text: str) -> str:
    text = (answer_text or "").strip()
    if "####" in text:
        return text.rsplit("####", maxsplit=1)[-1].strip()
    return text


def normalize_answer(value: Any) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return extract_gold_answer(str(value))


def convert_row(row: dict, instruction: str = INSTRUCTION_INTEGER) -> dict[str, str]:
    if "prompt" in row and "answer" in row:
        return {
            "prompt": str(row["prompt"]).strip(),
            "answer": normalize_answer(row["answer"]),
        }

    if "question" in row and "answer" in row:
        prompt = f"{instruction}\n\nProblem:\n{str(row['question']).strip()}"
        return {
            "prompt": prompt,
            "answer": normalize_answer(row["answer"]),
        }

    if "input" in row and "target" in row:
        prompt = f"{instruction}\n\nProblem:\n{str(row['input']).strip()}"
        return {
            "prompt": prompt,
            "answer": normalize_answer(row["target"]),
        }

    raise ValueError(f"Row is missing required keys: {sorted(row.keys())}")


def build_from_local_raw(
    raw_dir: Path, instruction: str = INSTRUCTION_INTEGER
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_path = raw_dir / "train.jsonl"
    val_path = raw_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        missing = [str(p) for p in (train_path, val_path) if not p.exists()]
        raise FileNotFoundError(f"Missing local raw files: {', '.join(missing)}")

    train_rows = [convert_row(row, instruction=instruction) for row in read_jsonl(train_path)]
    val_rows = [convert_row(row, instruction=instruction) for row in read_jsonl(val_path)]
    return train_rows, val_rows


def _count_from_ratio(total: int, ratio: float) -> int:
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    return max(1, int(round(total * ratio)))


def build_from_hf(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    val_size: int | None,
    seed: int,
    instruction: str = INSTRUCTION_INTEGER,
    test_size: int | None = 0,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    from datasets import load_dataset

    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config)
    else:
        ds = load_dataset(dataset_name)

    if split not in ds:
        raise ValueError(f"Split '{split}' not found in {dataset_name}. Available: {list(ds.keys())}")

    split_rows = list(ds[split])
    total = len(split_rows)

    val_size_used = val_size
    if val_size_used is None:
        if val_ratio is None:
            raise ValueError("val_size is None but val_ratio is not provided")
        val_size_used = _count_from_ratio(total, val_ratio)

    test_size_used = test_size
    if test_size_used is None:
        if test_ratio is None:
            raise ValueError("test_size is None but test_ratio is not provided")
        test_size_used = _count_from_ratio(total, test_ratio)

    if val_size_used < 1:
        raise ValueError(f"val_size must be >= 1, got {val_size_used}")
    if test_size_used < 0:
        raise ValueError(f"test_size must be >= 0, got {test_size_used}")

    # Always leave at least one row for train.
    max_non_train = total - 1
    non_train = val_size_used + test_size_used
    if non_train > max_non_train:
        overflow = non_train - max_non_train
        reduce_test = min(test_size_used, overflow)
        test_size_used -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            reduce_val = min(max(val_size_used - 1, 0), overflow)
            val_size_used -= reduce_val
            overflow -= reduce_val
        if overflow > 0:
            raise ValueError(
                f"Not enough rows ({total}) for val_size={val_size_used} and test_size={test_size_used}"
            )

    rng = random.Random(seed)
    rng.shuffle(split_rows)
    val_raw = split_rows[:val_size_used]
    test_raw = split_rows[val_size_used : val_size_used + test_size_used]
    train_raw = split_rows[val_size_used + test_size_used :]

    train_samples = [convert_row(row, instruction=instruction) for row in train_raw]
    val_samples = [convert_row(row, instruction=instruction) for row in val_raw]
    test_samples = [convert_row(row, instruction=instruction) for row in test_raw]
    return train_samples, val_samples, test_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset in OpenRLHF format")
    parser.add_argument(
        "--source",
        choices=("auto", "local", "hf", "hf_gsm_hard"),
        default="auto",
        help="Input data source mode",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/gsm8k",
        help="Directory containing raw train/val jsonl (used by local/auto modes)",
    )
    parser.add_argument("--output-dir", default="data", help="Where to write train/val jsonl")
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help=(
            "Validation size for HF sources. Defaults: hf=1000, "
            "hf_gsm_hard=10%% of split size"
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Test size for HF sources. Default: hf=0, "
            "hf_gsm_hard=10%% of split size (writes test.jsonl)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for HF source splits",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_used = args.source
    if args.source == "auto":
        if (raw_dir / "train.jsonl").exists() and (raw_dir / "val.jsonl").exists():
            source_used = "local"
        else:
            source_used = "hf"

    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_split: str | None = None
    instruction_used = INSTRUCTION_INTEGER
    val_size_used: int | None = None
    test_size_used: int | None = None
    val_ratio_used: float | None = None
    test_ratio_used: float | None = None

    if source_used == "local":
        train_samples, val_samples = build_from_local_raw(raw_dir, instruction=instruction_used)
        test_samples: list[dict[str, str]] = []
    else:
        if source_used == "hf":
            hf_dataset = "openai/gsm8k"
            hf_config = "main"
            hf_split = "train"
            instruction_used = INSTRUCTION_INTEGER
            val_size_used = args.val_size if args.val_size is not None else 1000
            test_size_used = args.test_size if args.test_size is not None else 0
            val_ratio_used = None
            test_ratio_used = None
        elif source_used == "hf_gsm_hard":
            hf_dataset = "reasoning-machines/gsm-hard"
            hf_config = None
            hf_split = "train"
            instruction_used = INSTRUCTION_NUMBER
            # Default deterministic 80/10/10 split.
            val_size_used = args.val_size
            test_size_used = args.test_size
            val_ratio_used = 0.1 if val_size_used is None else None
            test_ratio_used = 0.1 if test_size_used is None else None
        else:
            raise ValueError(f"Unsupported source: {source_used}")

        train_samples, val_samples, test_samples = build_from_hf(
            dataset_name=hf_dataset,
            dataset_config=hf_config,
            split=hf_split,
            val_size=val_size_used,
            instruction=instruction_used,
            test_size=test_size_used,
            val_ratio=val_ratio_used,
            test_ratio=test_ratio_used,
            seed=args.seed,
        )

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    write_jsonl(train_file, train_samples)
    write_jsonl(val_file, val_samples)
    test_file = output_dir / "test.jsonl"
    if test_samples:
        write_jsonl(test_file, test_samples)

    print(f"Source: {source_used}")
    if hf_dataset:
        print(
            f"HF dataset: {hf_dataset}"
            + (f" (config={hf_config})" if hf_config else "")
            + f", split={hf_split}, val_size={len(val_samples)}, test_size={len(test_samples)}, seed={args.seed}"
        )
    if instruction_used == INSTRUCTION_NUMBER:
        print("Prompt instruction: final_number")
    else:
        print("Prompt instruction: final_integer")
    print(f"Prepared {len(train_samples)} training samples -> {train_file}")
    print(f"Prepared {len(val_samples)} validation samples -> {val_file}")
    if test_samples:
        print(f"Prepared {len(test_samples)} test samples -> {test_file}")


if __name__ == "__main__":
    main()
