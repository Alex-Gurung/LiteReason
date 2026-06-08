#!/usr/bin/env python3
"""
Prepare the Flawed Fictions dataset for OpenRLHF GRPO training.

This script formats the HuggingFace dataset into train/validation JSONL files
with `prompt` and `answer` fields, matching the expectations of the paper
training scripts.
"""


import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Flawed Fictions data for GRPO.")
    parser.add_argument(
        "--split",
        type=str,
        default="flawed_fictions",
        help="Which split of kahuja/flawed-fictions to use.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(Path(__file__).resolve().parent / "prompts" / "basicprompt.txt"),
        help="Prompt template file with a {story} placeholder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Where to write train/val jsonl files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of examples to allocate to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    return parser.parse_args()


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, story: str) -> str:
    if "{story}" not in template:
        raise ValueError("Prompt template must contain a '{story}' placeholder.")
    return template.replace("{story}", story)


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    dataset = dataset.shuffle(seed=seed)
    valtest_len = int(len(dataset) * val_ratio)
    train_end = len(dataset) - (2 * valtest_len)
    train_slice = dataset.select(range(train_end))
    val_slice = dataset.select(range(train_end, train_end + valtest_len))
    test_slice = dataset.select(range(train_end + valtest_len, len(dataset)))
    return train_slice, val_slice, test_slice


def build_samples(dataset: Dataset, template: str) -> list[dict[str, Any]]:
    samples = []
    for row in dataset:
        prompt = format_prompt(template, row["story"])
        label = str(int(row["cont_error"]))
        samples.append(
            {
                "prompt": prompt,
                "answer": label,
                "example_id": row.get("example_id"),
            }
        )
    return samples


def write_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    template = load_prompt_template(args.prompt_file)

    dataset = load_dataset("kahuja/flawed-fictions", split=args.split)
    print(f"len(dataset): {len(dataset)}")
    train_slice, val_slice, test_slice = split_dataset(dataset, args.val_ratio, args.seed)

    train_samples = build_samples(train_slice, template)
    val_samples = build_samples(val_slice, template)
    test_samples = build_samples(test_slice, template)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    output_dir = Path(args.output_dir)
    write_jsonl(train_samples, output_dir / "train.jsonl")
    write_jsonl(val_samples, output_dir / "val.jsonl")
    write_jsonl(test_samples, output_dir / "test.jsonl")

    print(f"Wrote {len(train_samples)} training samples to {output_dir / 'train.jsonl'}")
    print(f"Wrote {len(val_samples)} validation samples to {output_dir / 'val.jsonl'}")
    print(f"Wrote {len(test_samples)} test samples to {output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
