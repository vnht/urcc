#!/usr/bin/env python3
"""Step 0 — Mine the model's over-commitment behaviour on unanswerable questions.

For one model, runs greedy generation on the raw unanswerable pools (KUQ +
SQuAD), judges each completion via the Cerebras gpt-oss-120b judge, and writes:

    step0_mine/data/mined/<model>_<dataset>.jsonl    all judged rows
    step0_mine/data/forget/<model>_<dataset>.jsonl   COMMIT-only subset
                                                     = forget pool D_F (category A)

Each row contains:
    example_id, dataset, source_index, source_id,
    question, context, generation_prompt,
    full_completion_clean, y_com_prefix_k8,
    judge_label, judge_model, judge_raw_output,
    gen_time_s, judge_time_s

`y_com_prefix_k8` is the first 8 tokens of the model's completion, used by
step 1 (activation extraction) to align with the K_ANSWER_TOKENS window.

Crash-safe & resumable
----------------------
Every row is appended to `step0_mine/data/mined/...jsonl` and flushed as
soon as it's been generated AND judged. Re-running this script skips any
example_id already present.

Run
---
    python step0_mine/mine.py --model qwen_instruct
    python step0_mine/mine.py --model qwen_instruct --max-per-dataset 50  # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import (
    Progress,
    Stopwatch,
    append_jsonl,
    build_unanswerable_prompt,
    first_k_token_prefix,
    format_duration,
    free_model,
    generate_greedy,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    write_jsonl,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_unanswerable(dataset: str, max_n: int | None) -> list[dict]:
    path = cfg.sampled_unanswerable_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Sampled file not found: {path}")
    rows = load_jsonl(path)
    out = []
    for i, r in enumerate(rows):
        out.append({
            "dataset":      dataset,
            "source_index": i,
            "source_id":    r.get("id", i),
            "question":     r["question"],
            "context":      r.get("context"),
        })
    if max_n is not None:
        out = out[:max_n]
    return out


def _example_id(model_key: str, dataset: str, source_index: int) -> str:
    return f"{dataset}_{source_index}_{model_key}"


def _existing_mined_ids(model_key: str, dataset: str) -> set[str]:
    """Read the on-disk mined file and return the set of completed example_ids."""
    path = cfg.mined_path(model_key, dataset)
    if not path.exists():
        return set()
    return {
        r["example_id"] for r in load_jsonl(path)
        if r.get("example_id") and r.get("judge_label") is not None
    }


# ── Judge ─────────────────────────────────────────────────────────────────────

def _import_judge():
    """Local import of the judge module (lives at repo root)."""
    from judge import (  # type: ignore[import]
        ABSTAIN, COMMIT, JUDGE_MODEL_ID, build_judge_prompt,
        call_judge, make_cerebras_client, normalise_label,
    )
    return {
        "ABSTAIN": ABSTAIN, "COMMIT": COMMIT,
        "JUDGE_MODEL_ID": JUDGE_MODEL_ID,
        "build_judge_prompt": build_judge_prompt,
        "call_judge": call_judge,
        "make_cerebras_client": make_cerebras_client,
        "normalise_label": normalise_label,
    }


# ── Forget filter ─────────────────────────────────────────────────────────────

def _write_forget_files(model_key: str, judge_mod: dict) -> None:
    """Filter mined rows by judge_label == COMMIT and (re)write forget files."""
    norm = judge_mod["normalise_label"]
    COMMIT = judge_mod["COMMIT"]

    for ds in ("kuq", "squad"):
        mined_p = cfg.mined_path(model_key, ds)
        if not mined_p.exists():
            continue
        rows = load_jsonl(mined_p)
        forget = [
            r for r in rows
            if norm(r.get("judge_label")) == COMMIT
            and (r.get("y_com_prefix_k8") or "").strip()
        ]
        forget.sort(key=lambda r: r["source_index"])
        out_path = cfg.forget_path(model_key, ds)
        write_jsonl(out_path, forget)
        log.info("  forget %s: %d / %d COMMIT rows -> %s",
                 ds, len(forget), len(rows), out_path)


# ── Main run ──────────────────────────────────────────────────────────────────

def run(model_key: str, max_per_dataset: int | None,
        max_new_tokens: int = cfg.DEFAULT_MAX_NEW_TOKENS) -> None:
    pipeline_t0 = time.time()
    log.info("STEP 0 — MINE OVER-COMMITMENT  model=%s  max_per_dataset=%s",
             model_key, max_per_dataset)

    # Load all unanswerable instances per dataset.
    datasets: dict[str, list[dict]] = {}
    for ds in ("kuq", "squad"):
        datasets[ds] = _load_unanswerable(ds, max_per_dataset)
        log.info("  %s: %d unanswerable instances", ds, len(datasets[ds]))

    # Resume: drop any example_ids already finished on disk.
    existing_ids: dict[str, set[str]] = {
        ds: _existing_mined_ids(model_key, ds) for ds in datasets
    }
    todo: list[tuple[str, dict]] = []
    skipped = 0
    for ds, rows in datasets.items():
        for r in rows:
            ex_id = _example_id(model_key, ds, r["source_index"])
            if ex_id in existing_ids[ds]:
                skipped += 1
            else:
                todo.append((ds, r))
    if skipped:
        log.info("  resume: %d examples already mined, %d remaining",
                 skipped, len(todo))

    # Always rebuild forget files at the end, even if there's nothing to mine.
    judge_mod = _import_judge()
    if not todo:
        log.info("  nothing to do — all examples already mined for %s.", model_key)
        _write_forget_files(model_key, judge_mod)
        log.info("STEP 0 done in %s", format_duration(time.time() - pipeline_t0))
        return

    # Load model + judge client.
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=True)
    client = judge_mod["make_cerebras_client"]()

    # Single-pass gen+judge with per-row append for crash safety.
    counts = {"COMMIT": 0, "ABSTAIN": 0, "ERROR": 0}
    gen_total = 0.0
    judge_total = 0.0
    progress = Progress(total=len(todo), desc="step 0 mine", log_every=10)

    for ds, inst in todo:
        prompt = build_unanswerable_prompt(ds, inst)

        # Generate
        gen_t0 = time.time()
        try:
            completion = generate_greedy(
                model, tokenizer, model_key, prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            log.warning("  gen error: %s", exc)
            completion = ""
        gen_dt = time.time() - gen_t0
        gen_total += gen_dt

        prefix_k8 = first_k_token_prefix(tokenizer, completion,
                                         k=cfg.K_ANSWER_TOKENS) if completion else ""

        # Judge
        judge_t0 = time.time()
        if completion:
            jp = judge_mod["build_judge_prompt"](
                question=inst["question"], completion=completion,
                context=inst.get("context"),
            )
            try:
                label, raw = judge_mod["call_judge"](client, jp)
            except Exception as exc:
                log.warning("  judge error: %s", exc)
                label, raw = "ERROR", str(exc)
        else:
            label, raw = None, None
        judge_dt = time.time() - judge_t0
        judge_total += judge_dt

        row = {
            "example_id":            _example_id(model_key, ds, inst["source_index"]),
            "dataset":               ds,
            "source_index":          inst["source_index"],
            "source_id":             inst["source_id"],
            "question":              inst["question"],
            "context":               inst.get("context"),
            "generator_model":       cfg.MODEL_REGISTRY[model_key],
            "model_key":             model_key,
            "generation_prompt":     prompt,
            "full_completion_clean": completion,
            "y_com_prefix_k8":       prefix_k8,
            "judge_label":           label,
            "judge_model":           judge_mod["JUDGE_MODEL_ID"],
            "judge_raw_output":      raw,
            "gen_time_s":            round(gen_dt, 3),
            "judge_time_s":          round(judge_dt, 3),
        }
        append_jsonl(cfg.mined_path(model_key, ds), row)

        counts[label if label in counts else "ERROR"] += 1
        n_done = progress.n + 1   # tick() is called next; this is the count after it
        progress.tick(extras={
            "C": counts["COMMIT"], "A": counts["ABSTAIN"], "E": counts["ERROR"],
            "gen_avg": f"{gen_total / max(n_done, 1):.2f}s",
        })

    progress.done(extras={
        "C": counts["COMMIT"], "A": counts["ABSTAIN"], "E": counts["ERROR"],
    })
    log.info("  gen total=%s avg=%.2fs/inst   judge total=%s avg=%.2fs/inst",
             format_duration(gen_total), gen_total / max(progress.n, 1),
             format_duration(judge_total), judge_total / max(progress.n, 1))

    free_model(model)

    # Build COMMIT-only forget files from the freshly-updated mined files.
    _write_forget_files(model_key, judge_mod)

    log.info("STEP 0 done in %s. mined=%s forget=%s",
             format_duration(time.time() - pipeline_t0),
             cfg.MINED_DIR, cfg.FORGET_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 0: mine the model's over-commitment behaviour.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--max-per-dataset", type=int, default=None,
                   help="Cap per dataset (smoke testing)")
    p.add_argument("--max-new-tokens", type=int, default=cfg.DEFAULT_MAX_NEW_TOKENS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, max_per_dataset=args.max_per_dataset,
        max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
