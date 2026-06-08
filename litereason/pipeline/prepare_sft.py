#!/usr/bin/env python3
"""Prepare reasoning data for SFT training with implicit thought markers.

This script processes a reasoning dataset and creates training data with
various levels of sentence masking, replacing masked sentences with
`<implicit_thought>N</implicit_thought>` markers where N indicates the
number of latent reasoning steps to generate.

It works with any bring-your-own dataset via the `jsonl` mode; the
Flawed-Fictions `ff` mode is just one example input format.

Input formats:
- ff mode: reads {split}.json with {idx, question, steps[], answer}
- jsonl mode (bring-your-own): reads {split}.jsonl, one JSON object per line.
  Required keys per record:
    - "prompt" (or "question"): the input question/prompt text.
    - "answer": the final (boxed) answer text.
  Reasoning steps are taken from "steps" (a list of step strings) if present;
  otherwise "trace" (a single string) is sentence-split into steps. "steps"
  takes precedence over "trace" when both are present.

Output format:
- JSONL files with chat messages format compatible with SFT training

Usage:
    python -m litereason.pipeline.prepare_sft --input-dir ./data --output-dir ./processed
    python -m litereason.pipeline.prepare_sft --input-dir ./data --output-dir ./processed --strategies twenty_five_percent fifty_percent
"""


import argparse
import json
import os
import random
import re
from functools import cache
from typing import Any, Dict, List, Tuple, Union

import spacy
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm
from transformers import AutoTokenizer


@cache
def _get_nlp():
    # Loaded lazily so importing this module (and the no-download CPU tests) does
    # not require the en_core_web_sm model, which ships via a separate
    # `python -m spacy download`, not as a pip dependency.
    return spacy.load("en_core_web_sm")


IMPLICIT_THOUGHT_PATTERN = re.compile(r"<implicit_thought>(\d+)</implicit_thought>")
IMPLICIT_PROMPT_INJECTION = (
    "In addition to your normal reasoning, you may format your reasoning with "
    "implicit thought tags of the following format:\n\n"
    " <implicit_thought>number</implicit_thought> \n\n"
    "Where number is a number between 1 and 5, and corresponds to the complexity "
    "of the thought. When you use this tool, you can skip the sentence that you "
    "would have said and move on with your reasoning. This will allow you to think "
    "a lot more about the story, while skipping over obvious conclusions/reasoning "
    "steps. We recommend doing this in the center of your reasoning, and not at the "
    "beginning or end."
)


def ensure_implicit_prompt(question: str) -> str:
    """Inject implicit-thought prompt guidance if it is missing.

    This is idempotent: if the prompt already mentions `<implicit_thought>`,
    it is returned unchanged.
    """
    text = question.strip()
    if not text or "<implicit_thought>" in text:
        return text
    return f"{text}\n\n{IMPLICIT_PROMPT_INJECTION}"


def sentence_split_spacy(text: str) -> List[str]:
    doc = _get_nlp()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def join_sentences(sentences: List[str]) -> str:
    return "\n".join(sentences)


def calculate_reasoning_steps(sentence: str, tokenizer, token_ratio: float = 0.2) -> int:
    """Number of latent reasoning steps for a replaced sentence.

    Per the paper, the count is a proportion ``token_ratio`` (t_r) of the number
    of *tokens* in the replaced sentence ("roughly one latent token for every
    five original reasoning tokens"), clamped to [1, 5].

    Args:
        sentence: The sentence being replaced by a latent span.
        tokenizer: HF tokenizer used to count tokens (via ``.encode``).
        token_ratio: Latent tokens per original token (default 0.2 = 1 per 5).

    Returns:
        Number of reasoning steps (1-5).
    """
    n_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
    return max(1, min(5, int(n_tokens * token_ratio)))


def mask_sentences(
    original_text: str,
    sentences: List[str],
    mask_ratio: float,
    tokenizer,
    token_ratio: float = 0.2,
    return_indices: bool = False,
) -> Union[str, Tuple[str, List[int]]]:
    """Replace randomly selected sentences with implicit-thought markers.

    The masked text is rebuilt from ``sentences`` by index, so a sentence that
    repeats (or is a substring of another) is replaced at the correct position.
    A plain ``str.replace`` would instead hit the first textual match and could
    corrupt an earlier sentence.

    Args:
        original_text: The text containing all sentences (returned unchanged when
            ``mask_ratio <= 0``); equals ``"\\n".join(non-empty sentences)``.
        sentences: List of sentences extracted from the trace.
        mask_ratio: Fraction of sentences to mask (0.0 to 1.0).
        tokenizer: HF tokenizer used to size each latent span (see
            ``calculate_reasoning_steps``).
        token_ratio: Latent tokens per original token (default 0.2).
        return_indices: If True, also return the indices of masked sentences.

    Returns:
        Masked text (and optionally the sorted indices of masked sentences).
    """
    sents = [s for s in sentences if s.strip()]
    if mask_ratio <= 0 or not sents:
        return (original_text, []) if return_indices else original_text

    k = max(1, int(len(sents) * mask_ratio))
    chosen = set(random.sample(range(len(sents)), k))

    # Rebuild by index: masked sentences -> markers, others kept verbatim.
    pieces: List[str] = []
    for i, s in enumerate(sents):
        if i in chosen:
            complexity = calculate_reasoning_steps(s, tokenizer, token_ratio)
            pieces.append(f"<implicit_thought>{complexity}</implicit_thought>")
        else:
            pieces.append(s)
    text = "\n".join(pieces)

    # Normalize whitespace around markers
    text = re.sub(r"\s*<implicit_thought>", " <implicit_thought>", text)
    text = re.sub(r"</implicit_thought>\s*", "</implicit_thought> ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = text.strip()

    return (text, sorted(chosen)) if return_indices else text


def load_ff_split(input_dir: str, split: str) -> List[Dict[str, Any]]:
    """Load Flawed Fictions JSON split."""
    path = os.path.join(input_dir, f"{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_split(input_dir: str, split: str) -> List[Dict[str, Any]]:
    """Load JSONL split file."""
    path = os.path.join(input_dir, f"{split}.jsonl")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_messages(question: str, thinking_text: str, boxed_answer: str) -> List[Dict[str, str]]:
    """Build chat messages in the standard format."""
    return [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": f"{thinking_text.strip()}\n{boxed_answer.strip()}"},
    ]


def get_masking_strategies() -> Dict[str, Tuple[float, int]]:
    """Get available masking strategies.

    Returns:
        Dict mapping strategy name to (mask_ratio, num_samples).
    """
    return {
        "no_masking": (0.0, 1),
        "ten_percent": (0.10, 5),
        "twenty_five_percent": (0.25, 5),
        "fifty_percent": (0.50, 5),
        "seventy_five_percent": (0.75, 5),
        "full_masking": (1.0, 1),
    }


def process_split(
    records: List[Dict[str, Any]],
    strategies: Dict[str, Tuple[float, int]],
    tokenizer_name: str,
    token_ratio: float = 0.2,
    seed: int = 42,
    paired: bool = False,
    preview_n: int = 0,
    prompt_mode: str = "implicit",
    require_markers: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process a data split with the specified masking strategies.

    Args:
        records: List of data records.
        strategies: Dict of strategy name to (mask_ratio, num_samples).
        tokenizer_name: HF model name for tokenization (e.g. Qwen/Qwen2.5-7B-Instruct).
        token_ratio: Latent tokens per original token (see calculate_reasoning_steps).
        seed: Random seed for reproducibility.
        paired: If True, emit paired masked/original examples.
        preview_n: Number of examples to preview per strategy.
        prompt_mode: Prompt handling mode: "as_is" or "implicit".
        require_markers: If True, fail when a masked sample has no marker.

    Returns:
        Dict mapping strategy name to list of processed examples.
    """
    random.seed(seed)
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in strategies.keys()}
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    recs = list(records)
    random.shuffle(recs)

    previews: Dict[str, List[Dict[str, Any]]] = {k: [] for k in strategies.keys()} if preview_n > 0 else {}

    n_skipped = 0
    n_dropped_no_sentences = 0
    for rec in tqdm(recs, desc="Processing examples"):
        # Accept both "question" (ff mode) and "prompt" (bring-your-own jsonl
        # mode, matching the key used everywhere else in the repo).
        question = (rec.get("question") or rec.get("prompt") or "").strip()
        if prompt_mode == "implicit":
            question = ensure_implicit_prompt(question)
        # "steps" takes precedence over "trace" when both are present.
        steps = rec.get("steps") or []
        if not steps and rec.get("trace"):
            steps = sentence_split_spacy(str(rec["trace"]))
        boxed_answer = (rec.get("answer") or "").strip()

        if not question or not boxed_answer:
            n_skipped += 1
            continue

        # Compute the usable (non-empty) reasoning sentences once. A record with
        # a valid prompt+answer but empty/whitespace steps (or a trace that
        # spaCy splits into zero non-empty sentences) has nothing to mask:
        # mask_sentences would return the marker-less original and the
        # require-markers check downstream would raise, aborting the whole job.
        # Skip such records instead.
        usable_steps = [s for s in steps if s.strip()]
        if not usable_steps:
            n_dropped_no_sentences += 1
            continue
        steps = usable_steps

        thinking_text = join_sentences(steps)
        base_messages = build_messages(question, thinking_text, boxed_answer)

        for strat_name, (mask_ratio, n_samples) in strategies.items():
            for _ in range(n_samples):
                want_preview = preview_n > 0 and len(previews.get(strat_name, [])) < preview_n
                res = mask_sentences(
                    thinking_text, steps, mask_ratio,
                    tokenizer=tok, token_ratio=token_ratio,
                    return_indices=want_preview
                )

                if want_preview:
                    masked_text, idxs = res  # type: ignore
                    previews[strat_name].append({
                        "question": question,
                        "original": thinking_text,
                        "masked": masked_text,
                        "answer": boxed_answer,
                        "masked_indices": idxs,
                        "num_steps": len(steps),
                        "mask_ratio": mask_ratio,
                    })
                else:
                    masked_text = res  # type: ignore

                if require_markers and mask_ratio > 0 and "<implicit_thought>" not in masked_text:
                    q_excerpt = question.replace("\n", " ")[:180]
                    raise ValueError(
                        "Masked sample is missing implicit thought markers. "
                        f"strategy={strat_name} ratio={mask_ratio} question_excerpt={q_excerpt!r}"
                    )

                messages = build_messages(question, masked_text, boxed_answer)
                # return_dict=False yields a plain List[int] (transformers >=5 returns
                # a BatchEncoding by default, which is not JSON-serializable).
                ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
                rec_out: Dict[str, Any] = {"messages": messages, "input_ids": ids}

                if paired and mask_ratio > 0:
                    pair_obj: Dict[str, Any] = {
                        "masked": {"messages": messages, "input_ids": ids},
                        "original": {
                            "messages": base_messages,
                            "input_ids": tok.apply_chat_template(base_messages, tokenize=True, return_dict=False),
                        },
                    }
                    out[strat_name].append(pair_obj)
                else:
                    out[strat_name].append(rec_out)

    n_total = len(recs)
    if n_total > 0 and (n_skipped + n_dropped_no_sentences) == n_total:
        raise RuntimeError(
            f"No usable records: all {n_total} were missing a question/prompt "
            "or answer, or had no usable reasoning sentences. Expected keys: "
            "'prompt' (or 'question') and 'answer', plus non-empty 'steps'/'trace'."
        )
    if n_skipped:
        print(f"Skipped {n_skipped}/{n_total} records missing a question/prompt or answer.")
    if n_dropped_no_sentences:
        print(
            f"Skipped {n_dropped_no_sentences}/{n_total} records with no usable "
            "reasoning sentences (empty/whitespace steps or trace)."
        )

    # Print previews
    if preview_n > 0:
        _print_previews(previews)

    return out


def _print_previews(previews: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print preview examples."""
    console = Console()
    for strat_name, samples in previews.items():
        if not samples:
            continue
        console.print(f"\n[bold]Preview: {strat_name}[/bold] ({len(samples)} examples)")
        for i, ex in enumerate(samples, 1):
            left = Panel(ex["original"], title=f"Original (steps={ex['num_steps']})")
            right = Panel(ex["masked"], title=f"Masked (ratio={ex['mask_ratio']:.2f})")
            console.print(Columns([left, right]))
            console.print(Panel(
                f"Answer: {ex['answer']}\nMasked indices: {ex['masked_indices']}",
                title=f"Example {i}"
            ))


def save_outputs(
    outputs: Dict[str, List[Dict[str, Any]]],
    split: str,
    out_dir: str,
    paired: bool
) -> None:
    """Save processed outputs to JSONL files."""
    os.makedirs(out_dir, exist_ok=True)

    for name, items in outputs.items():
        if not items:
            continue

        suffix = f"{split}_{name}_paired.jsonl" if paired and name != "no_masking" else f"{split}_{name}.jsonl"
        path = os.path.join(out_dir, suffix)

        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"Saved {len(items)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reasoning data for SFT with implicit thought markers "
        "(works with any bring-your-own jsonl dataset; ff mode is one example)."
    )
    parser.add_argument(
        "--input-dir",
        default="ff_data",
        help="Input directory containing data files (default 'ff_data' is just an example)"
    )
    parser.add_argument(
        "--mode",
        choices=["ff", "jsonl"],
        default="ff",
        help=(
            "Input format: 'ff' (JSON with {idx, question, steps[], answer}) or "
            "'jsonl' (one JSON object per line; required keys: 'prompt' (or "
            "'question') and 'answer'; reasoning steps from 'steps' if present, "
            "else sentence-split from 'trace' -- 'steps' takes precedence)."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for processed JSONL files"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Data splits to process"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(get_masking_strategies().keys()),
        help="Masking strategies to use (default: all)"
    )
    parser.add_argument(
        "--token-ratio",
        type=float,
        default=0.2,
        help="Latent tokens per original token in a replaced sentence (t_r; default: 0.2).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF model name for tokenization (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Emit paired masked/original examples"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of examples to preview per strategy"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["as_is", "implicit"],
        default="implicit",
        help="Prompt handling mode. 'implicit' injects implicit-thought instructions.",
    )
    parser.add_argument(
        "--require-markers",
        dest="require_markers",
        action="store_true",
        default=True,
        help="Fail if a masked sample has no <implicit_thought> marker (default: on).",
    )
    parser.add_argument(
        "--allow-missing-markers",
        dest="require_markers",
        action="store_false",
        help="Allow masked samples without <implicit_thought> markers.",
    )

    args = parser.parse_args()

    all_strategies = get_masking_strategies()
    strategies = {
        k: v for k, v in all_strategies.items()
        if not args.strategies or k in args.strategies
    }
    print(f"Strategies: {list(strategies.keys())}")

    for split in args.splits:
        print(f"\nProcessing {split} split...")

        if args.mode == "ff":
            records = load_ff_split(args.input_dir, split)
        else:
            records = load_jsonl_split(args.input_dir, split)

        outputs = process_split(
            records,
            strategies,
            tokenizer_name=args.tokenizer,
            token_ratio=args.token_ratio,
            seed=args.seed,
            paired=args.paired,
            preview_n=args.preview,
            prompt_mode=args.prompt_mode,
            require_markers=args.require_markers,
        )

        save_outputs(outputs, split, args.output_dir, paired=args.paired)

    print("\nDone!")


if __name__ == "__main__":
    main()
