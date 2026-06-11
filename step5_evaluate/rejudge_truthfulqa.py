#!/usr/bin/env python3
"""Re-judge existing TruthfulQA results with the updated judge prompt.

Completions are already saved in each truthfulqa.json; this script re-runs
only the judge API call, overwrites judge_label / judge_raw_output in place,
and recomputes the summary metrics. No GPU / model inference needed.

Usage
-----
    # All result dirs that have a truthfulqa.json:
    python3 step5_evaluate/rejudge_truthfulqa.py

    # Specific result dirs only:
    python3 step5_evaluate/rejudge_truthfulqa.py \\
        step5_evaluate/data/results/baseline_gptoss_instruct \\
        step5_evaluate/data/results/gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rejudge_truthfulqa")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("evaluate_ext", Path(__file__).parent / "evaluate-ext.py")
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
TRUTHFULQA_JUDGE_TEMPLATE = _mod.TRUTHFULQA_JUDGE_TEMPLATE
_judge                    = _mod._judge
_summarise_truthfulqa     = _mod._summarise_truthfulqa
_attach_baseline          = _mod._attach_baseline
from evaluate import _load_dataset_json, _save_dataset_json  # type: ignore[import]

ALLOWED = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}
RESULTS_ROOT = REPO_ROOT / "step5_evaluate" / "data" / "results"


def _rejudge_file(path: Path, client, baseline_dir: Path | None, dry_run: bool) -> None:
    data = _load_dataset_json(path)
    if not data:
        log.warning("  skip (empty/missing): %s", path)
        return

    rows: list[dict] = data.get("rows") or []
    if not rows:
        log.warning("  skip (no rows): %s", path)
        return

    log.info("  %s  (%d rows)", path, len(rows))

    changed = 0
    for i, row in enumerate(rows):
        completion = row.get("completion", "").strip()
        question   = row.get("question", "")
        correct    = row.get("correct_answers", [])
        incorrect  = row.get("incorrect_answers", [])

        if not completion:
            new_label = "NOT_ATTEMPTED"
            new_raw   = "empty completion"
        else:
            jp = TRUTHFULQA_JUDGE_TEMPLATE.format(
                question=question,
                correct="\n".join(f"- {a}" for a in correct),
                incorrect="\n".join(f"- {a}" for a in incorrect),
                completion=completion,
            )
            new_label, new_raw = _judge(client, jp, ALLOWED)

        old_label = row.get("judge_label")
        if old_label != new_label:
            changed += 1

        row["judge_label"]      = new_label
        row["judge_raw_output"] = new_raw

        if (i + 1) % 50 == 0:
            log.info("    %d / %d  (changed so far: %d)", i + 1, len(rows), changed)

    log.info("  done — %d / %d labels changed", changed, len(rows))

    if dry_run:
        log.info("  [dry-run] not writing")
        return

    data["rows"]    = rows
    data["metrics"] = _summarise_truthfulqa(rows)

    # Re-attach baseline deltas if a baseline_dir is known.
    for k in ("baseline", "baseline_run", "deltas"):
        data.pop(k, None)

    run_name = data.get("run", "")
    # Infer baseline dir from run name if not provided explicitly.
    inferred_baseline: Path | None = baseline_dir
    if inferred_baseline is None and not run_name.startswith("baseline_"):
        model_key = data.get("model_key", "")
        candidate = RESULTS_ROOT / f"baseline_{model_key}"
        if candidate.is_dir():
            inferred_baseline = candidate

    if inferred_baseline is not None:
        _attach_baseline(data, inferred_baseline, "truthfulqa")

    _save_dataset_json(path, data)
    m = data["metrics"]
    log.info(
        "  saved  halluc=%.3f  acc=%.3f  not_att=%.3f  c|att=%.3f",
        m.get("hallucination_rate", float("nan")),
        m.get("accuracy",           float("nan")),
        m.get("not_attempted_rate", float("nan")),
        m.get("correct_given_attempted", float("nan")),
    )
    if data.get("deltas"):
        log.info(
            "  vs baseline -> %s",
            ", ".join(f"Δ{k}={v:+.3f}" for k, v in data["deltas"].items()),
        )


def _find_result_dirs() -> list[Path]:
    return sorted(p.parent for p in RESULTS_ROOT.rglob("truthfulqa.json"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-judge TruthfulQA rows with the updated prompt (no model inference)."
    )
    p.add_argument(
        "result_dirs", nargs="*", type=Path,
        help="Result dirs to process (default: all dirs under results/ with a truthfulqa.json).",
    )
    p.add_argument(
        "--baseline", type=Path, default=None,
        help="Override baseline dir for delta computation.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Re-judge and print stats but do not write back to disk.",
    )
    args = p.parse_args()

    result_dirs = [d.resolve() for d in args.result_dirs] if args.result_dirs else _find_result_dirs()
    baseline_dir = args.baseline.resolve() if args.baseline else None

    # Baselines first so trained-run deltas use refreshed baseline metrics.
    result_dirs = sorted(result_dirs, key=lambda d: (0 if d.name.startswith("baseline_") else 1, d.name))

    log.info("Re-judging TruthfulQA in %d result dir(s)%s",
             len(result_dirs), "  [dry-run]" if args.dry_run else "")
    for d in result_dirs:
        log.info("  %s", d.name)

    from judge import make_cerebras_client  # type: ignore[import]
    client = make_cerebras_client()

    t0 = time.time()
    for result_dir in result_dirs:
        path = result_dir / "truthfulqa.json"
        if not path.exists():
            log.warning("missing: %s", path)
            continue
        log.info("\n[%s]", result_dir.name)
        _rejudge_file(path, client, baseline_dir, dry_run=args.dry_run)

    elapsed = time.time() - t0
    log.info("\nDone in %.1fs", elapsed)


if __name__ == "__main__":
    main()
