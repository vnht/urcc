#!/usr/bin/env python3
"""Investigate WHY gpt-oss rows come back with an empty parsed completion.

`generate_greedy` keeps only the harmony `final`-channel text, so the results
JSON never shows what the model actually emitted on an empty row. This script
finds the rows whose stored completion is empty in an existing results file,
regenerates them with the SAME stored prompt, and dumps the RAW decoded stream
(channel markers included) plus a per-row diagnosis:

    budget_exhausted_in_analysis  — analysis channel ran to the token cap and
                                    the final header never appeared (the
                                    expected common case; fix = larger budget)
    empty_final_channel           — a final header appeared but with no text
                                    after it (model "answered" with nothing)
    no_channel_header             — generation produced no harmony channel
                                    marker at all (template/adapter damage —
                                    investigate the adapter if frequent)
    stopped_early_no_final        — generation hit <|return|>/<|end|> before
                                    any final channel (model terminated inside
                                    analysis; adapter-induced degeneration)
    now_non_empty                 — produced a final answer this time (only
                                    possible if --max-new-tokens here exceeds
                                    the budget the eval ran with)

Raw streams are written to
    step5_evaluate/data/debug/<run_name>/<dataset>_empty_rows.jsonl
and a summary histogram is printed at the end.

Run (same model-loading flags as the eval scripts)
---
    # trained run
    python3 step5_evaluate/debug_empty_completions.py \
        --run-dir step4_train/data/runs/<run_name> --datasets kuq

    # baseline
    python3 step5_evaluate/debug_empty_completions.py \
        --model gptoss_instruct --datasets kuq simpleqa
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import (
    Stopwatch,
    _build_generation_input_ids,
    log,
    parse_harmony_final,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # type: ignore[import]
    _load_adapter_model,
    _load_base_model,
    _load_dataset_json,
)

_CHANNEL_RE = re.compile(r"<\|channel\|>\s*(\w+)")
_FINAL_HEADER_RE = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>")
_STOP_RE = re.compile(r"<\|return\|>|<\|end\|>\s*$")


def _generate_raw(model, tokenizer, model_key: str, prompt: str,
                  max_new_tokens: int) -> tuple[str, int]:
    """Greedy generation returning the RAW decoded stream (special tokens
    kept) and the number of new tokens — bypasses the final-channel parse."""
    ids = _build_generation_input_ids(tokenizer, model_key, prompt)
    input_ids = torch.tensor([ids], dtype=torch.long)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or
                          getattr(tokenizer, "eos_token_id", 0),
        )
    new_tokens = out[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=False), len(new_tokens)


def _diagnose(raw: str, n_new: int, budget: int) -> str:
    channels = _CHANNEL_RE.findall(raw)
    final_text = parse_harmony_final(raw)
    if final_text:
        return "now_non_empty"
    if not channels:
        return "no_channel_header"
    if _FINAL_HEADER_RE.search(raw):
        return "empty_final_channel"
    if n_new >= budget:
        return "budget_exhausted_in_analysis"
    return "stopped_early_no_final"


def run(args: argparse.Namespace) -> None:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        result_name = run_dir.name
        if args.adapter_scale != 1.0:
            result_name = f"{result_name}_scale{args.adapter_scale:g}"
    else:
        result_name = f"baseline_{args.model}"

    results_dir = cfg.RESULTS_DIR / result_name
    debug_dir = cfg.STEP5_DIR / "data" / "debug" / result_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Collect empty rows per dataset BEFORE paying for the model load.
    todo: dict[str, list[dict]] = {}
    for dataset in args.datasets:
        rec = _load_dataset_json(results_dir / f"{dataset}.json")
        if not rec:
            log.warning("  [%s] no readable results in %s — skipping",
                        dataset, results_dir)
            continue
        empty = [r for r in (rec.get("rows") or [])
                 if not (r.get("completion") or "").strip()
                 and (r.get("prompt") or "").strip()]
        if args.max_rows is not None:
            empty = empty[: args.max_rows]
        log.info("  [%s] %d empty rows to regenerate", dataset, len(empty))
        if empty:
            todo[dataset] = empty
    if not todo:
        log.info("Nothing to investigate — no empty completions found.")
        return

    with Stopwatch("model load"):
        if args.run_dir is not None:
            model, tokenizer, model_key = _load_adapter_model(
                args.run_dir.resolve(), adapter_scale=args.adapter_scale)
        else:
            model, tokenizer, model_key = _load_base_model(args.model)

    budget = args.max_new_tokens
    for dataset, rows in todo.items():
        out_path = debug_dir / f"{dataset}_empty_rows.jsonl"
        counts: dict[str, int] = {}
        t0 = time.time()
        with out_path.open("w") as f:
            for i, r in enumerate(rows, 1):
                raw, n_new = _generate_raw(
                    model, tokenizer, model_key, r["prompt"], budget)
                verdict = _diagnose(raw, n_new, budget)
                counts[verdict] = counts.get(verdict, 0) + 1
                f.write(json.dumps({
                    "id":           r.get("id"),
                    "question":     r.get("question"),
                    "verdict":      verdict,
                    "n_new_tokens": n_new,
                    "budget":       budget,
                    "channels":     _CHANNEL_RE.findall(raw),
                    "final_text":   parse_harmony_final(raw),
                    "raw_tail":     raw[-600:],
                    "raw":          raw,
                }, ensure_ascii=False) + "\n")
                f.flush()
                log.info("  [%s] %d/%d id=%s -> %s (%d tok)",
                         dataset, i, len(rows), r.get("id"), verdict, n_new)
        log.info("  [%s] verdicts: %s   (%.1fs)  -> %s",
                 dataset,
                 ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                 time.time() - t0, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate empty-completion rows and dump raw harmony streams.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, help="Trained run dir (LoRA adapter).")
    g.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()),
                   help="Model key for the zero-shot baseline results.")
    p.add_argument("--datasets", nargs="+", default=["kuq"],
                   help="Datasets whose results JSONs to scan (default: kuq).")
    p.add_argument("--adapter-scale", type=float, default=1.0,
                   help="Must match the eval run if it used a scale suffix.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap regenerated rows per dataset (default: all).")
    p.add_argument("--max-new-tokens", type=int,
                   default=cfg.GPTOSS_MAX_NEW_TOKENS_ESCALATED,
                   help="Decode budget for the re-generation (default: the "
                        "escalated gpt-oss cap, 1024).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
