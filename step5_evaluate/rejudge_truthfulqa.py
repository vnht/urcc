#!/usr/bin/env python3
"""Offline re-judge of saved TruthfulQA results with the updated judge.

The TruthfulQA judge in evaluate-ext.py was rewritten to grade each answer into
the SimpleQA verdict {CORRECT, INCORRECT, NOT_ATTEMPTED} against the reference
TRUE/FALSE answer sets (a refusal like "I have no comment" is NOT_ATTEMPTED,
never CORRECT). Existing result files were produced by the OLD two-flag
(truthful/informative) judge, so their labels need replaying.

Generation is greedy / deterministic and the per-row `completion` is already
persisted, so we do NOT need to re-run the models on GPU — we only replay the
judge over the saved completions. This script:

  1. Re-judges every TruthfulQA result file from its saved completions using the
     new judge (resumable: rows already carrying a CORRECT/INCORRECT/
     NOT_ATTEMPTED label are skipped).
  2. Recomputes metrics with the updated summariser.
  3. Re-attaches baseline deltas for trained runs (baselines re-judged first so
     the deltas use the refreshed baseline metrics).

Usage:
    CEREBRAS_TOKEN=... python3 step5_evaluate/rejudge_truthfulqa.py
    CEREBRAS_TOKEN=... python3 step5_evaluate/rejudge_truthfulqa.py --max 50  # smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rejudge")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# evaluate-ext.py has a hyphen, so import it by file location.
_spec = importlib.util.spec_from_file_location("evaluate_ext", _HERE / "evaluate-ext.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

import config as cfg  # noqa: E402  (after sys.path is set up via repo layout)

_TQA_LABELS = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}


def _rejudge_file(path: Path, client, max_rows: int | None, save_every: int) -> dict:
    rec = ev._load_dataset_json(path)
    if not rec or "rows" not in rec:
        log.warning("  [%s] no rows — skipping", path)
        return {}
    rows = rec["rows"]
    if max_rows is not None:
        rows = rows[:max_rows]

    todo = [r for r in rows if r.get("judge_label") not in _TQA_LABELS]
    log.info("  [%s] %d rows, %d to (re)judge", path.parent.name, len(rows), len(todo))

    if not todo:
        rec["rows"] = rows
        rec["metrics"] = ev._summarise_truthfulqa(rows)
        ev._save_dataset_json(path, rec)
        return rec["metrics"]

    allowed = ev._ALLOWED["truthfulqa"]
    progress = ev.Progress(total=len(todo), desc=path.parent.name, log_every=10)
    done = 0
    for r in rows:
        if r.get("judge_label") in _TQA_LABELS:
            continue
        # Drop stale flags from the old two-flag judge.
        r.pop("truthful", None)
        r.pop("informative", None)
        completion = (r.get("completion") or "").strip()
        if not completion:
            r["judge_label"], r["judge_raw_output"] = "NOT_ATTEMPTED", "empty completion"
        else:
            jp = ev.TRUTHFULQA_JUDGE_TEMPLATE.format(
                question=r["question"],
                correct="\n".join(f"- {a}" for a in r.get("correct_answers", [])),
                incorrect="\n".join(f"- {a}" for a in r.get("incorrect_answers", [])),
                completion=completion,
            )
            label, raw = ev._judge(client, jp, allowed)
            r["judge_label"], r["judge_raw_output"] = label, raw
        done += 1
        progress.tick(extras={
            "C":  sum(1 for x in rows if x.get("judge_label") == "CORRECT"),
            "I":  sum(1 for x in rows if x.get("judge_label") == "INCORRECT"),
            "NA": sum(1 for x in rows if x.get("judge_label") == "NOT_ATTEMPTED"),
        })
        if done % save_every == 0:
            rec["metrics"] = ev._summarise_truthfulqa(rows)
            ev._save_dataset_json(path, rec)

    progress.done()
    rec["rows"] = rows
    rec["metrics"] = ev._summarise_truthfulqa(rows)
    ev._save_dataset_json(path, rec)
    return rec["metrics"]


def _baseline_for(trained_dir: Path, baseline_dirs: list[Path]) -> Path | None:
    """Match a trained run dir to its baseline dir by model-key prefix."""
    name = trained_dir.name
    best = None
    for b in baseline_dirs:
        model = b.name[len("baseline_"):]
        if name.startswith(model) and (best is None or len(model) > len(best.name)):
            best = b
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max", type=int, default=None, help="Cap rows per file (smoke test).")
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    from judge import make_cerebras_client  # type: ignore[import]
    client = make_cerebras_client()

    results_root = cfg.RESULTS_DIR
    files = sorted(results_root.glob("*/truthfulqa.json"))
    if not files:
        log.warning("No truthfulqa.json files under %s", results_root)
        return

    baseline_files = [f for f in files if f.parent.name.startswith("baseline_")]
    trained_files = [f for f in files if not f.parent.name.startswith("baseline_")]
    baseline_dirs = [f.parent for f in baseline_files]

    # Baselines first so trained-run deltas use the refreshed baseline metrics.
    log.info("Re-judging %d baseline file(s)...", len(baseline_files))
    for f in baseline_files:
        m = _rejudge_file(f, client, args.max, args.save_every)
        if m:
            log.info("  [%s] halluc=%.3f acc=%.3f na=%.3f c|att=%.3f",
                     f.parent.name, m["hallucination_rate"], m["accuracy"],
                     m["not_attempted_rate"], m["correct_given_attempted"])

    log.info("Re-judging %d trained file(s)...", len(trained_files))
    for f in trained_files:
        m = _rejudge_file(f, client, args.max, args.save_every)
        if not m:
            continue
        bdir = _baseline_for(f.parent, baseline_dirs)
        rec = ev._load_dataset_json(f)
        ev._attach_baseline(rec, bdir, "truthfulqa")
        ev._save_dataset_json(f, rec)
        log.info("  [%s] halluc=%.3f acc=%.3f na=%.3f c|att=%.3f (vs %s)",
                 f.parent.name, m["hallucination_rate"], m["accuracy"],
                 m["not_attempted_rate"], m["correct_given_attempted"],
                 bdir.name if bdir else "—")


if __name__ == "__main__":
    main()
