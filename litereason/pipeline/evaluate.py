#!/usr/bin/env python3
"""Dataset-agnostic evaluation via vLLM's local Python API.

Runs a model on a ``{prompt, answer}`` JSONL dataset, scores each response with a
user ``reward_func.py`` (the task metric = mean reward), and reports confidence
intervals, token usage, AND whether latent reasoning fired
(``latent_trigger_rate`` / ``mean_latent_steps`` / ``n_with_latents``). Writes a
``summary.json`` next to the predictions.

To evaluate a LiteReason checkpoint with latent reasoning active::

    LITEREASON_ENABLE_VLLM=1 VLLM_PLUGINS=litereason \
    python -m litereason.pipeline.evaluate \
        --model ./outputs/litereason_ckpt \
        --test-file ./data/test.jsonl \
        --reward-func litereason/experiments/flawed_fictions/reward_func.py \
        --use-chat-template \
        --save-preds preds.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from litereason.pipeline.latent_stats import aggregate_latent_stats, latent_steps_in_text
from litereason.pipeline.metrics import compute_rep_stats
from litereason.pipeline.reward import load_reward_func, score_responses


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a model on a {prompt, answer} dataset via local vLLM.")
    p.add_argument("--model", required=True, help="HF model ID or local checkpoint path.")
    p.add_argument("--test-file", required=True, help="JSONL with {prompt, answer} fields.")
    p.add_argument("--reward-func", required=True, help="Path to a reward_func.py (OpenRLHF convention).")
    p.add_argument("--save-preds", required=True, help="Output JSONL path for predictions.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--num-samples", type=int, default=10, help="Repetitions per test item.")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Evaluate only first N items (0 = all).")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-model-len", type=int, default=0, help="Override max model length (0 = auto).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.0)
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--revision", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    reward_func = load_reward_func(args.reward_func)
    rows = read_jsonl(args.test_file)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    assert rows, f"No rows found in {args.test_file}"
    N, R = len(rows), args.num_samples

    llm_kwargs: Dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization > 0:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)

    tokenizer = None
    if args.use_chat_template:
        tok_kwargs = {"trust_remote_code": args.trust_remote_code}
        if args.revision:
            tok_kwargs["revision"] = args.revision
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)

    def render(text: str) -> str:
        if tokenizer is None:
            return text
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )

    prompts_all = [str(r.get("prompt", "")) for r in rows]
    labels_all = [r.get("answer") for r in rows]
    results: List[Dict[str, Any]] = []

    for rep in range(R):
        sp = SamplingParams(
            n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            max_tokens=args.max_tokens, seed=args.seed_base + rep,
        )
        for i in tqdm(range(0, N, args.batch_size), desc=f"rep {rep + 1}/{R}"):
            j = min(i + args.batch_size, N)
            prompts = prompts_all[i:j]
            labels = labels_all[i:j]
            outs = llm.generate([render(p) for p in prompts], sp, use_tqdm=False)
            responses = [o.outputs[0].text if o.outputs else "" for o in outs]
            rewards = score_responses(reward_func, responses=responses, prompts=prompts, labels=labels)
            for k, (o, resp, rew) in enumerate(zip(outs, responses, rewards, strict=True)):
                n_completion = len(o.outputs[0].token_ids) if o.outputs else 0
                n_prompt = len(o.prompt_token_ids) if o.prompt_token_ids else 0
                results.append({
                    "idx_original": i + k,
                    "rep": rep,
                    "answer": labels[k],
                    "response": resp,
                    "reward": rew,
                    "latent_steps": latent_steps_in_text(resp),
                    "num_completion_tokens": n_completion,
                    "num_prompt_tokens": n_prompt,
                })

    assert len(results) == N * R, f"Expected {N * R} results, got {len(results)}"

    # Task metric = mean reward; report per-repetition mean + CI.
    reward_per_rep = [
        sum(r["reward"] for r in results if r["rep"] == rep) / N for rep in range(R)
    ]
    stats = compute_rep_stats(reward_per_rep)
    latent = aggregate_latent_stats([r["response"] for r in results])
    tokens = [r["num_completion_tokens"] for r in results]
    mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0

    # With a single repetition there is no sampling distribution: std/CI would be
    # a degenerate [mean, mean], so report them as null rather than misleading.
    metric_std = stats["std"] if R > 1 else None
    metric_ci = [stats["ci_lo"], stats["ci_hi"]] if R > 1 else None
    summary = {
        "model": args.model,
        "metric_name": "mean_reward",
        "metric": stats["mean"],
        "metric_std": metric_std,
        "metric_ci": metric_ci,
        "mean_tokens": mean_tokens,
        "n": N,
        "num_samples": R,
        **latent,
    }
    print(f"Mean reward: {stats['mean']:.4f} (std {stats['std']:.4f})")
    if R > 1:
        print(f"95% t-CI: [{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}]")
    print(f"Avg completion tokens: {mean_tokens:.1f}")
    print(
        f"Latent reasoning fired: trigger_rate={latent['latent_trigger_rate']:.2%} "
        f"mean_latent_steps={latent['mean_latent_steps']:.2f} "
        f"n_with_latents={latent['n_with_latents']}/{N * R}"
    )

    out_path = Path(args.save_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({**r, "idx": i}, ensure_ascii=False) + "\n")
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} predictions to {out_path} and summary to {summary_path}")


if __name__ == "__main__":
    main()
