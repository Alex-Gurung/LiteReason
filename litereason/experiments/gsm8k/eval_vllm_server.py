#!/usr/bin/env python3
"""Evaluate GSM-style JSONL data against a running vLLM OpenAI server.

Expected input JSONL rows:
  - prompt: str
  - answer: str
"""


import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_text(choice: Any) -> str:
    content = getattr(choice.message, "content", None)
    if content is None and hasattr(choice, "text"):
        content = choice.text
    return content or ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model via vLLM OpenAI API server")
    p.add_argument("--input", default="data_gsm_hard/val.jsonl", help="JSONL file with {prompt, answer}")
    p.add_argument(
        "--output",
        default="working/vllm_dp4_eval/preds_qwen3_1p7b_t1_2048.jsonl",
        help="Where to save per-example predictions JSONL",
    )
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0, help="Evaluate first N rows only (0 = all)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent requests to issue",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from litereason.experiments.gsm8k.reward_func import extract_answer_text, is_equiv

    rows = read_jsonl(Path(args.input))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No rows loaded from {args.input}")

    # Ensure default output dir exists even for default relative path.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Lazily create one OpenAI client per worker thread.
    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(base_url=args.api_base, api_key=args.api_key)
            thread_local.client = client
        return client

    def eval_one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        prompt = str(row.get("prompt", ""))
        gold = str(row.get("answer", "")).strip()
        if not prompt:
            raise ValueError(f"Row {idx} missing prompt")
        if not gold:
            raise ValueError(f"Row {idx} missing answer")

        response = None
        last_err: Exception | None = None
        for attempt in range(args.max_retries):
            try:
                response = get_client().chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    extra_body={"top_k": args.top_k} if args.top_k >= 0 else {},
                )
                break
            except Exception as exc:  # noqa: BLE001
                # Broad: OpenAI/httpx surface many transient error types; retry,
                # and re-raise on the final attempt.
                last_err = exc
                if attempt + 1 == args.max_retries:
                    raise
                time.sleep(args.retry_sleep)

        if response is None:
            raise RuntimeError(f"No response for row {idx}: {last_err!r}")

        text = get_text(response.choices[0])
        pred = extract_answer_text(text)
        try:
            is_correct = bool(is_equiv(pred, gold))
        except Exception:
            # math_verify raises a wide set of parse errors on malformed model
            # output; an unverifiable prediction counts as incorrect.
            is_correct = False

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        return {
            "idx": idx,
            "gold": gold,
            "pred": pred,
            "correct": int(is_correct),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "response": text,
        }

    out_by_idx: dict[int, dict[str, Any]] = {}
    t0 = time.time()

    workers = max(1, min(args.concurrency, len(rows)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(eval_one, i, row) for i, row in enumerate(rows)]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"eval (concurrency={workers})",
        ):
            r = fut.result()
            out_by_idx[r["idx"]] = r

    out_rows = [out_by_idx[i] for i in range(len(rows))]

    elapsed = time.time() - t0
    n = len(out_rows)
    correct = sum(int(r["correct"]) for r in out_rows)
    total_prompt_tokens = sum(int(r["prompt_tokens"]) for r in out_rows)
    total_completion_tokens = sum(int(r["completion_tokens"]) for r in out_rows)
    acc = correct / n
    avg_prompt = total_prompt_tokens / n if n else 0.0
    avg_completion = total_completion_tokens / n if n else 0.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, out_rows)

    print(f"examples={n}")
    print(f"correct={correct}")
    print(f"accuracy={acc:.6f}")
    print(f"avg_prompt_tokens={avg_prompt:.2f}")
    print(f"avg_completion_tokens={avg_completion:.2f}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"saved_predictions={output_path}")


if __name__ == "__main__":
    main()
