#!/usr/bin/env python3
"""Re-judge cached mined completions without regenerating them.

Reads each `step0_mine/data/mined/<model>_<dataset>.jsonl` row, calls the
Cerebras judge again on `full_completion_clean` (using the cached question
and context), overwrites `judge_label` / `judge_raw_output` /
`judge_model` / `judge_time_s` in place, and rebuilds the COMMIT-only
forget files at `step0_mine/data/forget/`.

Use when you've tweaked the judge prompt or model and want to refresh the
labels without paying the GPU cost of re-generating ~4k completions.

Run
---
    python step0_mine/rejudge.py --model qwen_instruct
    python step0_mine/rejudge.py --model qwen_instruct --dataset kuq
    python step0_mine/rejudge.py --model qwen_instruct --only-missing
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import Progress, Stopwatch, format_duration, load_jsonl, log
from judge import (
    JUDGE_MODEL_ID,
    build_judge_prompt,
    call_judge,
    make_cerebras_client,
    normalise_label,
)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write `rows` to `path` atomically (write to a tempfile in the same
    directory, then rename). Avoids corrupting the mined file if the process
    is interrupted mid-write."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _rejudge_dataset(
    client,
    model_key: str,
    dataset: str,
    only_missing: bool,
) -> tuple[int, dict[str, int]]:
    path = cfg.mined_path(model_key, dataset)
    if not path.exists():
        log.warning("  %s: %s not found, skipping", dataset, path)
        return 0, {}

    rows = load_jsonl(path)
    counts = {"COMMIT": 0, "ABSTAIN": 0, "judge_error": 0, "skipped_empty": 0,
              "skipped_existing": 0}

    todo_idx = []
    for i, r in enumerate(rows):
        completion = (r.get("full_completion_clean") or "").strip()
        if not completion:
            counts["skipped_empty"] += 1
            continue
        if only_missing and normalise_label(r.get("judge_label")) in {"COMMIT", "ABSTAIN"}:
            counts["skipped_existing"] += 1
            continue
        todo_idx.append(i)

    log.info("  %s: %d rows, %d to re-judge (skipped: %d empty, %d already labelled)",
             dataset, len(rows), len(todo_idx),
             counts["skipped_empty"], counts["skipped_existing"])

    if not todo_idx:
        return 0, counts

    judge_total = 0.0
    progress = Progress(total=len(todo_idx), desc=f"rejudge {dataset}", log_every=25)
    save_every = 100   # checkpoint the file every N rows to bound data loss on crash

    for n_done, idx in enumerate(todo_idx, start=1):
        r = rows[idx]
        prompt = build_judge_prompt(
            question=r["question"],
            completion=r["full_completion_clean"],
            context=r.get("context"),
        )
        t0 = time.time()
        try:
            label, raw = call_judge(client, prompt)
        except Exception as exc:
            log.warning("  judge error on row %d: %s", idx, exc)
            label, raw = "judge_error", str(exc)
        dt = time.time() - t0
        judge_total += dt

        r["judge_label"]      = label
        r["judge_raw_output"] = raw
        r["judge_model"]      = JUDGE_MODEL_ID
        r["judge_time_s"]     = round(dt, 3)

        counts[label if label in counts else "judge_error"] += 1
        progress.tick(extras={
            "C": counts["COMMIT"], "A": counts["ABSTAIN"], "E": counts["judge_error"],
            "judge_avg": f"{judge_total / n_done:.2f}s",
        })

        if n_done % save_every == 0:
            _atomic_write_jsonl(path, rows)

    progress.done(extras={
        "C": counts["COMMIT"], "A": counts["ABSTAIN"], "E": counts["judge_error"],
    })
    _atomic_write_jsonl(path, rows)
    log.info("  %s: wrote updated labels -> %s (judge total=%s avg=%.2fs)",
             dataset, path, format_duration(judge_total),
             judge_total / max(len(todo_idx), 1))
    return len(todo_idx), counts


def _rewrite_forget_files(model_key: str) -> None:
    """Re-derive COMMIT-only forget files from the freshly-relabelled mined
    files. Mirrors mine.py::_write_forget_files."""
    from _common import write_jsonl

    for ds in ("kuq", "squad"):
        mined_p = cfg.mined_path(model_key, ds)
        if not mined_p.exists():
            continue
        rows = load_jsonl(mined_p)
        forget = [
            r for r in rows
            if normalise_label(r.get("judge_label")) == "COMMIT"
            and (r.get("y_com_prefix_k8") or "").strip()
        ]
        forget.sort(key=lambda r: r["source_index"])
        out_path = cfg.forget_path(model_key, ds)
        write_jsonl(out_path, forget)
        log.info("  forget %s: %d / %d COMMIT rows -> %s",
                 ds, len(forget), len(rows), out_path)


def run(model_key: str, datasets: list[str], only_missing: bool) -> None:
    t0 = time.time()
    log.info("REJUDGE  model=%s  datasets=%s  only_missing=%s",
             model_key, datasets, only_missing)

    with Stopwatch("cerebras client init"):
        client = make_cerebras_client()

    grand_total = 0
    for ds in datasets:
        n, _ = _rejudge_dataset(client, model_key, ds, only_missing)
        grand_total += n

    if grand_total > 0:
        _rewrite_forget_files(model_key)

    log.info("REJUDGE done in %s  (rejudged %d rows)",
             format_duration(time.time() - t0), grand_total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-judge cached mined completions without re-generating.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--dataset", choices=("kuq", "squad", "all"), default="all",
                   help="Which mined file(s) to re-judge (default: all).")
    p.add_argument("--only-missing", action="store_true",
                   help="Only re-judge rows whose judge_label is missing / "
                        "'judge_error' / unrecognised (default: re-judge every row).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["kuq", "squad"] if args.dataset == "all" else [args.dataset]
    run(args.model, datasets, only_missing=args.only_missing)


if __name__ == "__main__":
    main()
