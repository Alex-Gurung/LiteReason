#!/usr/bin/env python3
"""Generate reasoning traces for Flawed Fictions via a vLLM server.

Calls a vLLM OpenAI-compatible endpoint to generate reasoning traces,
filters by correctness using the reward function, and outputs in the
format expected by prepare_sft.py (--mode ff).

Prerequisites:
    pip install openai spacy
    python -m spacy download en_core_web_sm

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct --port 8000

    # Generate traces:
    python -m litereason.pipeline.generate_traces \
        --input-path ./ff_data/train.jsonl \
        --output-path ./ff_traces/train.json \
        --model-name Qwen/Qwen2.5-1.5B-Instruct \
        --limit 20 --k 5
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI
from tqdm import tqdm

from litereason.pipeline.prepare_sft import sentence_split_spacy
from litereason.pipeline.reward import load_reward_func, score_responses

BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batch_iter(xs: List[Any], bs: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def get_chat_samples(
    client: OpenAI,
    model_name: str,
    prompt: str,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    """Call vLLM /chat/completions and return n response texts."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"top_k": top_k},
    )
    # Fall back to ch.text when content is None (some vLLM versions).
    # Always append (even empty) so len(outs) == n for downstream zipping.
    outs: List[str] = []
    for ch in resp.choices:
        content = getattr(ch.message, "content", None)
        if content is None and hasattr(ch, "text"):
            content = ch.text
        outs.append(content or "")
    return outs


def split_trace(trace: str):
    """Split a model trace into (reasoning_steps, boxed_answer).

    Returns (steps, answer) where steps is a list of sentences and answer
    is the full \\boxed{...} string. If no boxed answer found, answer is
    empty and the full trace is sentence-split.
    """
    # Find last \boxed{...} occurrence
    matches = list(BOXED_RE.finditer(trace))
    if matches:
        last = matches[-1]
        answer = last.group(0)  # e.g. "\\boxed{Yes}"
        reasoning = trace[: last.start()].strip()
    else:
        answer = ""
        reasoning = trace.strip()

    steps = sentence_split_spacy(reasoning) if reasoning else []
    return steps, answer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and filter reasoning traces for Flawed Fictions via vLLM."
    )
    ap.add_argument(
        "--input-path", type=str, required=True,
        help="JSONL file with {prompt, answer} fields (from download_dataset.py).",
    )
    ap.add_argument(
        "--output-path", type=str, required=True,
        help="Output JSON file ({question, steps, answer}) consumed by prepare_sft.",
    )
    ap.add_argument(
        "--reward-func", type=str, required=True,
        help="Path to a reward_func.py; only generations with reward > 0 are kept.",
    )
    ap.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--api-key", type=str, default="EMPTY")
    ap.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    ap.add_argument("--k", type=int, default=5, help="Traces per example.")
    ap.add_argument(
        "--batch-size", type=int, default=8,
        help="Dataset items per progress-bar tick (API calls are per-prompt).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N items (0 = all).",
    )
    args = ap.parse_args()

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    reward_func = load_reward_func(args.reward_func)

    data = read_jsonl(Path(args.input_path))
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    print(f"Loaded {len(data)} examples from {args.input_path}")

    correct_records: List[Dict[str, Any]] = []
    total_traces = 0
    idx = 0

    for batch in tqdm(list(batch_iter(data, args.batch_size)), desc="Generating"):
        for ex in batch:
            prompt = str(ex.get("prompt", ""))
            gold = str(ex.get("answer", "")).strip()

            generations = get_chat_samples(
                client=client,
                model_name=args.model_name,
                prompt=prompt,
                n=args.k,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            rewards = score_responses(
                reward_func,
                responses=generations,
                prompts=[prompt] * len(generations),
                labels=[gold] * len(generations),
            )
            total_traces += len(generations)

            for g, r in zip(generations, rewards, strict=True):
                if r > 0:
                    steps, answer = split_trace(g)
                    correct_records.append({
                        "idx": idx,
                        "question": prompt,
                        "steps": steps,
                        "answer": answer,
                    })
                    idx += 1

    # Write as JSON array (prepare_sft.py --mode ff format)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(correct_records, f, ensure_ascii=False, indent=2)

    print(
        f"Generated {total_traces} traces, kept {len(correct_records)} correct "
        f"({len(correct_records)/max(total_traces,1):.1%})"
    )
    print(f"Wrote {len(correct_records)} records to {out_path}")


if __name__ == "__main__":
    main()
