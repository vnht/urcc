#!/usr/bin/env python3
"""Step 5 — Evaluate a trained UOC run on held-out data.

Loads the LoRA adapter from a step-4 run directory, generates one greedy
completion per held-out prompt for KUQ + SQuAD, judges each completion via
`judge.py` (Cerebras gpt-oss-120b), and writes per-run metrics.

Reads:  step5_evaluate/data/heldout/{kuq,squad}.jsonl
        step4_train/data/runs/<run_name>/        (adapter + training_config.json)
Writes: step5_evaluate/data/results/<run_name>/
            generations.jsonl              (incrementally appended)
            answerability_metrics.jsonl
            comparison.json                (optional, vs --baseline)

Crash-safe & resumable
----------------------
Each row is appended to `generations.jsonl` and flushed as soon as it has
both a completion and a judge label. Re-running this script skips any row
already present (same dataset + id).

Run
---
    python step5_evaluate/evaluate.py --run-dir step4_train/data/runs/<run_name>
    python step5_evaluate/evaluate.py --run-dir step4_train/data/runs/<run_name> \\
        --max-per-dataset 200            # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import (
    Progress,
    Stopwatch,
    append_jsonl,
    build_answerable_prompt,
    build_unanswerable_prompt,
    format_duration,
    generate_greedy,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    write_jsonl,
)


# ── Adapter loading ───────────────────────────────────────────────────────────

def _load_adapter_model(run_dir: Path):
    """Load base model from training_config.json + apply LoRA adapter."""
    cfg_path = run_dir / "training_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"training_config.json not found in {run_dir}")
    train_cfg = json.loads(cfg_path.read_text())
    model_key = train_cfg["model_key"]

    model, tokenizer = load_model_and_tokenizer(model_key, eval_only=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(run_dir))
    model.eval()
    return model, tokenizer, model_key, train_cfg


# ── Judging ───────────────────────────────────────────────────────────────────

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


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(rows: list[dict], dataset: str, judge_mod: dict) -> dict:
    norm = judge_mod["normalise_label"]
    COMMIT, ABSTAIN = judge_mod["COMMIT"], judge_mod["ABSTAIN"]

    subset = [r for r in rows if r["dataset"] == dataset]
    answerable   = [r for r in subset if r["answerable"]]
    unanswerable = [r for r in subset if not r["answerable"]]

    def label_of(r):
        return norm(r.get("judge_label"))

    def valid(g):
        return [r for r in g if label_of(r) in (COMMIT, ABSTAIN)]

    av  = valid(answerable)
    uv  = valid(unanswerable)
    tot = len(subset)
    err = tot - len(av) - len(uv)

    def rate(g, label):
        if not g:
            return float("nan")
        return sum(1 for r in g if label_of(r) == label) / len(g)

    tc = sum(1 for r in av if label_of(r) == COMMIT)
    ta = sum(1 for r in uv if label_of(r) == ABSTAIN)
    dec_acc = (tc + ta) / tot if tot else float("nan")

    return {
        "dataset":              dataset,
        "num_instances":        tot,
        "num_answerable":       len(answerable),
        "num_unanswerable":     len(unanswerable),
        "num_judge_errors":     err,
        "true_commitment_rate":  round(rate(av, COMMIT),   4),
        "false_abstention_rate": round(rate(av, ABSTAIN),  4),
        "true_abstention_rate":  round(rate(uv, ABSTAIN),  4),
        "false_commitment_rate": round(rate(uv, COMMIT),   4),
        "decision_accuracy":     round(dec_acc, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _row_key(dataset: str, row: dict) -> str:
    return f"{dataset}::{row.get('id', row.get('source_index', '?'))}"


def run(args: argparse.Namespace) -> None:
    pipeline_t0 = time.time()
    run_dir: Path = args.run_dir.resolve()
    out_dir = cfg.results_dir_for(run_dir.name)
    gen_path = out_dir / "generations.jsonl"

    log.info("STEP 5 — EVALUATE  run=%s", run_dir.name)
    log.info("  results_dir: %s", out_dir)

    # Resume: collect keys of already-completed rows on disk.
    existing_rows: list[dict] = load_jsonl(gen_path) if gen_path.exists() else []
    done_keys: set[str] = {
        _row_key(r["dataset"], r) for r in existing_rows
        if r.get("completion") is not None and r.get("judge_label") is not None
    }
    if existing_rows:
        log.info("  resume: %d existing rows in %s, %d already complete",
                 len(existing_rows), gen_path, len(done_keys))

    with Stopwatch("model load"):
        model, tokenizer, model_key, _train_cfg = _load_adapter_model(run_dir)
    log.info("  Adapter loaded for %s (%s)", model_key, cfg.MODEL_REGISTRY[model_key])

    judge_mod = _import_judge()
    client = judge_mod["make_cerebras_client"]()

    # Build the to-do list across datasets.
    todo: list[tuple[str, dict]] = []
    for dataset in ("kuq", "squad"):
        eval_path = cfg.heldout_path(dataset)
        if not eval_path.exists():
            log.warning("  eval pool missing: %s — skipping", eval_path)
            continue
        pool = load_jsonl(eval_path)
        if args.max_per_dataset is not None:
            pool = pool[:args.max_per_dataset]
        for r in pool:
            if _row_key(dataset, r) in done_keys:
                continue
            todo.append((dataset, r))
        log.info("  %s: %d in pool, %d to do",
                 dataset, len(pool), sum(1 for d, _ in todo if d == dataset))

    counts = {"COMMIT": 0, "ABSTAIN": 0, "ERROR": 0, "OTHER": 0}
    gen_total = 0.0
    judge_total = 0.0
    progress = Progress(total=len(todo), desc="step 5 eval", log_every=10)

    for dataset, r in todo:
        row = dict(r)
        row["dataset"] = dataset
        if not r.get("answerable"):
            row["prompt"] = build_unanswerable_prompt(dataset, r)
        else:
            row["prompt"] = build_answerable_prompt(dataset, r)

        # Generate
        gen_t0 = time.time()
        try:
            row["completion"] = generate_greedy(
                model, tokenizer, model_key, row["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            log.warning("  gen error: %s", exc)
            row["completion"] = ""
        gen_dt = time.time() - gen_t0
        gen_total += gen_dt
        row["gen_time_s"] = round(gen_dt, 3)

        # Judge
        judge_t0 = time.time()
        if row["completion"]:
            jp = judge_mod["build_judge_prompt"](
                question=row["question"],
                completion=row["completion"],
                context=row.get("context"),
            )
            try:
                label, raw = judge_mod["call_judge"](client, jp)
            except Exception as exc:
                log.warning("  judge error: %s", exc)
                label, raw = "ERROR", str(exc)
        else:
            label, raw = "ERROR", "empty completion"
        judge_dt = time.time() - judge_t0
        judge_total += judge_dt

        row["judge_label"] = label
        row["judge_model"] = judge_mod["JUDGE_MODEL_ID"]
        row["judge_raw_output"] = raw
        row["judge_time_s"] = round(judge_dt, 3)
        row["model"] = cfg.MODEL_REGISTRY[model_key]
        row["run"] = run_dir.name

        bucket = label if label in counts else "OTHER"
        counts[bucket] += 1

        append_jsonl(gen_path, row)
        existing_rows.append(row)

        progress.tick(extras={
            "C": counts["COMMIT"], "A": counts["ABSTAIN"],
            "E": counts["ERROR"], "O": counts["OTHER"],
        })

    progress.done(extras={"C": counts["COMMIT"], "A": counts["ABSTAIN"],
                          "E": counts["ERROR"]})
    if progress.n:
        log.info("  gen total=%s avg=%.2fs/inst   judge total=%s avg=%.2fs/inst",
                 format_duration(gen_total), gen_total / progress.n,
                 format_duration(judge_total), judge_total / progress.n)

    # Metrics
    metrics: list[dict] = []
    for dataset in ("kuq", "squad"):
        m = _compute_metrics(existing_rows, dataset, judge_mod)
        m["model"] = cfg.MODEL_REGISTRY[model_key]
        m["run"]   = run_dir.name
        metrics.append(m)
        log.info("  %-6s acc=%.3f TCR=%.3f FAR=%.3f TAR=%.3f FCR=%.3f",
                 dataset, m["decision_accuracy"], m["true_commitment_rate"],
                 m["false_abstention_rate"], m["true_abstention_rate"],
                 m["false_commitment_rate"])

    metrics_path = out_dir / "answerability_metrics.jsonl"
    write_jsonl(metrics_path, metrics)
    log.info("  Wrote -> %s", metrics_path)

    # Compare against baseline if provided
    if args.baseline:
        baseline_rows = load_jsonl(args.baseline)
        comparison = []
        for m in metrics:
            base = next((b for b in baseline_rows if b["dataset"] == m["dataset"]
                         and b.get("model") == m["model"]), None)
            if base is None:
                continue
            comparison.append({
                "dataset": m["dataset"],
                "deltas": {
                    k: round(m[k] - base[k], 4)
                    for k in ("decision_accuracy", "true_commitment_rate",
                              "false_abstention_rate", "true_abstention_rate",
                              "false_commitment_rate")
                    if k in base
                },
                "baseline": {k: base[k] for k in m if k in base},
                "post":     {k: m[k]    for k in m},
            })
        comp_path = out_dir / "comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2))
        log.info("  Wrote comparison -> %s", comp_path)

    log.info("STEP 5 done in %s. Outputs in %s",
             format_duration(time.time() - pipeline_t0), out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5: evaluate UOC run.")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run directory (step4_train/data/runs/<run_name>)")
    p.add_argument("--max-new-tokens", type=int, default=cfg.DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-per-dataset", type=int, default=None,
                   help="Cap eval to N rows per dataset (smoke testing)")
    p.add_argument("--baseline", type=Path, default=None,
                   help="Optional baseline metrics file for delta comparison")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
