#!/usr/bin/env python3
"""
Judge baseline generations and compute answerability metrics.

Reads:  eval_baseline_generations_<model>.jsonl (from generate.py)
Writes: eval_baseline_generations_<model>.jsonl  (adds judge_label in-place)
        eval_baseline_answerability_metrics.jsonl

Metrics per (model, dataset):
    true_commitment_rate  = P(COMMIT  | answerable)
    false_abstention_rate = P(ABSTAIN | answerable)
    true_abstention_rate  = P(ABSTAIN | unanswerable)
    false_commitment_rate = P(COMMIT  | unanswerable)
    decision_accuracy     = (true_commitments + true_abstentions) / total

Usage:
    python judge_outputs.py                          # all generation files in evaluation/
    python judge_outputs.py --model qwen-9b
    python judge_outputs.py --results-dir training/pre-training-baselines
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from judge import (  # type: ignore[import]
    build_judge_prompt, call_judge, make_cerebras_client,
    normalise_label, COMMIT, ABSTAIN, JUDGE_MODEL_ID,
)
from llms.constants import SHORTCUTS  # type: ignore[import]

SHORTCUT_TO_ID = SHORTCUTS
ID_TO_SHORTCUT = {v: k for k, v in SHORTCUTS.items()}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ─── Judging ──────────────────────────────────────────────────────────────────

def judge_file(gen_path: Path, client) -> list[dict]:
    """Judge unjudged rows in gen_path, write back in-place, return all rows."""
    rows: list[dict] = []
    with open(gen_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    needs_judging = [r for r in rows if normalise_label(r.get("judge_label")) is None]
    if not needs_judging:
        log.info("  All %d rows already judged in %s", len(rows), gen_path.name)
        return rows

    log.info("  Judging %d / %d rows in %s", len(needs_judging), len(rows), gen_path.name)

    judged_map: dict = {r["id"]: r for r in rows}
    committed = abstained = errors = 0

    bar = tqdm(needs_judging, desc=gen_path.stem, unit="inst", dynamic_ncols=True)
    for row in bar:
        prompt = build_judge_prompt(
            question=row["question"],
            completion=row["completion"],
            context=row.get("context"),
        )
        t_start = time.time()
        label, raw_output = call_judge(client, prompt)
        elapsed = time.time() - t_start

        judged_map[row["id"]].update({
            "judge_label": label,
            "judge_model": JUDGE_MODEL_ID,
            "judge_raw_output": raw_output,
            "judge_time_s": round(elapsed, 3),
        })

        if label == COMMIT:
            committed += 1
        elif label == ABSTAIN:
            abstained += 1
        else:
            errors += 1

        bar.set_postfix(C=committed, A=abstained, E=errors, t=f"{elapsed:.2f}s")

    all_rows = list(judged_map.values())
    with open(gen_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    log.info("  Saved → %s", gen_path.name)
    return all_rows


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], model_id: str, dataset: str) -> dict:
    subset       = [r for r in rows if r["dataset"] == dataset]
    answerable   = [r for r in subset if r["answerable"]]
    unanswerable = [r for r in subset if not r["answerable"]]

    # Only rows with a valid label contribute to rates (so rates sum to 1.0).
    # Errors are tracked separately and count as wrong in decision_accuracy.
    def label_of(r):
        return normalise_label(r.get("judge_label"))

    def valid(group):
        return [r for r in group if label_of(r) in (COMMIT, ABSTAIN)]

    ans_valid   = valid(answerable)
    unans_valid = valid(unanswerable)
    num_errors  = len(subset) - len(ans_valid) - len(unans_valid)

    def rate(group, label):
        if not group:
            return float("nan")
        return sum(1 for r in group if label_of(r) == label) / len(group)

    true_commitments = sum(1 for r in ans_valid   if label_of(r) == COMMIT)
    true_abstentions = sum(1 for r in unans_valid if label_of(r) == ABSTAIN)
    decision_accuracy = (true_commitments + true_abstentions) / len(subset) if subset else float("nan")

    return {
        "model":                model_id,
        "dataset":              dataset,
        "num_instances":        len(subset),
        "num_answerable":       len(answerable),
        "num_unanswerable":     len(unanswerable),
        "num_judge_errors":     num_errors,
        "true_commitment_rate":  round(rate(ans_valid,   COMMIT),  4),
        "false_abstention_rate": round(rate(ans_valid,   ABSTAIN), 4),
        "true_abstention_rate":  round(rate(unans_valid, ABSTAIN), 4),
        "false_commitment_rate": round(rate(unans_valid, COMMIT),  4),
        "decision_accuracy":     round(decision_accuracy, 4),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Judge baseline generations and compute metrics.")
    parser.add_argument("--model", help="Model shortcut or full HF ID (default: all)")
    parser.add_argument("--results-dir", type=Path, default=HERE,
                        help="Directory containing generation files and where output is written")
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    client = make_cerebras_client()

    if args.model:
        model_id = SHORTCUT_TO_ID.get(args.model, args.model)
        slug = ID_TO_SHORTCUT.get(model_id, model_id.replace("/", "__"))
        gen_files = [results_dir / f"eval_baseline_generations_{slug}.jsonl"]
    else:
        gen_files = sorted(results_dir.glob("eval_baseline_generations_*.jsonl"))

    if not gen_files:
        log.error("No generation files found in %s. Run generate.py first.", results_dir)
        sys.exit(1)

    all_metrics: list[dict] = []

    for gen_path in gen_files:
        log.info("")
        log.info("=" * 64)
        log.info("FILE  %s", gen_path.name)

        rows = judge_file(gen_path, client)
        if not rows:
            continue

        model_id = rows[0]["model"]
        for dataset in ("kuq", "squad"):
            metrics = compute_metrics(rows, model_id, dataset)
            all_metrics.append(metrics)
            log.info(
                "  %-8s  acc=%.3f  TCR=%.3f  FAR=%.3f  TAR=%.3f  FCR=%.3f",
                dataset,
                metrics["decision_accuracy"],
                metrics["true_commitment_rate"],
                metrics["false_abstention_rate"],
                metrics["true_abstention_rate"],
                metrics["false_commitment_rate"],
            )

    out_path = results_dir / "eval_baseline_answerability_metrics.jsonl"
    with open(out_path, "w") as f:
        for row in all_metrics:
            f.write(json.dumps(row) + "\n")
    log.info("Saved → %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
