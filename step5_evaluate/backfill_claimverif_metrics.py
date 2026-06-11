#!/usr/bin/env python3
"""Backfill the new claim-verification distribution metrics into saved results.

evaluate-ext.py now records, for scifact / averitec, how the SUPPORTS / REFUTES /
NOT ENOUGH INFO (/ CONFLICTING) predictions are distributed and — for trained
runs — how that distribution shifts vs the baseline:

    metrics.pred_counts / pred_rates          overall predicted-label mix
    metrics.confusion_by_gold[gold]           where each gold class's preds land
    pred_rate_deltas                          overall mix shift vs baseline
    confusion_rate_deltas[gold]               per-gold-class shift vs baseline

Generation + judging are already persisted per row, so we only recompute the
summary from the saved `judge_label`s (no GPU, no judge API). Baselines are
recomputed first so trained-run deltas use the refreshed baseline metrics.

Usage:
    python3 step5_evaluate/backfill_claimverif_metrics.py
    python3 step5_evaluate/backfill_claimverif_metrics.py --datasets scifact
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backfill")

_HERE = Path(__file__).resolve().parent

# evaluate-ext.py has a hyphen, so import it by file location.
_spec = importlib.util.spec_from_file_location("evaluate_ext", _HERE / "evaluate-ext.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import config as cfg  # noqa: E402  (sys.path is set up while loading evaluate-ext)

DATASETS = ("scifact", "averitec")


def _recompute(path: Path, dataset: str) -> bool:
    rec = ev._load_dataset_json(path)
    if not rec or "rows" not in rec:
        return False
    kind = ev.DATASET_KIND[dataset]
    rec["metrics"] = ev._SUMMARISE[kind](rec["rows"])
    ev._save_dataset_json(path, rec)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args()

    results_dir = cfg.RESULTS_DIR
    run_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    baselines = [p for p in run_dirs if p.name.startswith("baseline_")]
    trained = [p for p in run_dirs if not p.name.startswith("baseline_")]

    # Pass 1: recompute metrics for every run (baselines first).
    for run in baselines + trained:
        for dataset in args.datasets:
            path = run / f"{dataset}.json"
            if path.exists() and _recompute(path, dataset):
                log.info("  metrics recomputed: %s", path.relative_to(results_dir))

    # Pass 2: re-attach baseline deltas (incl. the new distribution-shift deltas)
    # for trained runs, now that baselines carry the new metrics.
    for run in trained:
        rec0 = ev._load_dataset_json(run / f"{args.datasets[0]}.json")
        model_key = (rec0 or {}).get("model_key")
        base_dir = results_dir / f"baseline_{model_key}" if model_key else None
        if not base_dir or not base_dir.exists():
            log.warning("  [%s] no baseline dir (model_key=%s) — deltas skipped",
                        run.name, model_key)
            continue
        for dataset in args.datasets:
            path = run / f"{dataset}.json"
            if not path.exists():
                continue
            rec = ev._load_dataset_json(path)
            ev._attach_baseline(rec, base_dir, dataset)
            ev._save_dataset_json(path, rec)
            log.info("  deltas attached: %s (vs %s)",
                     path.relative_to(results_dir), base_dir.name)


if __name__ == "__main__":
    main()
