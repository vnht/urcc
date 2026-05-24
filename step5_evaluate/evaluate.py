#!/usr/bin/env python3
"""Step 5 — Evaluate a UOC run (or zero-shot baseline) on held-out data.

Two modes:
  • Trained run:  --run-dir <dir>   loads the LoRA adapter from step 4 and
                                    evaluates the post-UOC model.
  • Baseline:     --model <key>     loads the bare base model with no
                                    adapter — the zero-shot reference. Run
                                    this BEFORE training so the trained
                                    eval can use --baseline against it.

Two evaluations run in one model load:

  (1) Answerability — KUQ + SQuAD
      For each held-out prompt, generate one greedy completion, judge it via
      judge.py (Cerebras gpt-oss-120b), and report TCR / FCR / TAR / FAR /
      decision accuracy.

  (2) UltraChat preservation (perplexity)
      For each held-out (prompt, response) pair, compute per-token
      cross-entropy of the response (capped to --max-response-tokens,
      default 256). Healthy preservation: PPL_post / PPL_base ≈ 1.

Output layout (one JSON file per dataset)
-----------------------------------------
    step5_evaluate/data/results/<name>/
        kuq.json         metrics + per-row details (+ baseline deltas if --baseline)
        squad.json       metrics + per-row details (+ baseline deltas if --baseline)
        ultrachat.json   PPL aggregate + per-row details (+ baseline ratios if --baseline)

Each file is the complete record for that dataset.

`<name>` is the run directory's name in trained mode, or
`baseline_<model_key>` in baseline mode.

Crash-safe & resumable
----------------------
The per-dataset JSON file is rewritten atomically every `--summary-every`
rows (default 5) and at the end of each dataset. On restart, rows whose id
already appears in `rows` are skipped. The atomic rewrite (write to .tmp,
then rename) guarantees the on-disk file is never half-written.

Run
---
    # Baseline (zero-shot, run this first):
    python3 step5_evaluate/evaluate.py --model qwen_instruct

    # Trained UOC run (with delta vs baseline):
    python3 step5_evaluate/evaluate.py --run-dir step4_train/data/runs/<run_name> \\
        --baseline step5_evaluate/data/results/baseline_qwen_instruct

    # Smoke:
    python3 step5_evaluate/evaluate.py --model qwen_instruct \\
        --max-per-dataset 50 --max-ppl-rows 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import (
    Progress,
    Stopwatch,
    build_answerable_prompt,
    build_unanswerable_prompt,
    format_duration,
    generate_greedy,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    tokenise_chat_prompt_response,
)


DEFAULT_MAX_RESPONSE_TOKENS = 256
DEFAULT_SUMMARY_EVERY = 5


# ── Atomic dataset-JSON I/O ───────────────────────────────────────────────────

def _load_dataset_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("  cannot read %s: %s — starting fresh", path, exc)
        return None


def _save_dataset_json(path: Path, data: dict) -> None:
    """Atomic write: serialize to .tmp, fsync, then rename over the target."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# ── Model loading ─────────────────────────────────────────────────────────────

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
    return model, tokenizer, model_key


def _load_base_model(model_key: str):
    """Load the bare base model with no adapter — zero-shot baseline."""
    model, tokenizer = load_model_and_tokenizer(model_key, eval_only=True)
    model.eval()
    return model, tokenizer, model_key


# ── Judging (answerability) ───────────────────────────────────────────────────

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


def _summarise_answerability(rows: list[dict], judge_mod: dict) -> dict:
    norm = judge_mod["normalise_label"]
    COMMIT, ABSTAIN = judge_mod["COMMIT"], judge_mod["ABSTAIN"]

    answerable   = [r for r in rows if r.get("answerable")]
    unanswerable = [r for r in rows if not r.get("answerable")]

    def label_of(r):
        return norm(r.get("judge_label"))

    def valid(g):
        return [r for r in g if label_of(r) in (COMMIT, ABSTAIN)]

    av  = valid(answerable)
    uv  = valid(unanswerable)
    tot = len(rows)
    err = tot - len(av) - len(uv)

    def rate(g, label):
        if not g:
            return float("nan")
        return sum(1 for r in g if label_of(r) == label) / len(g)

    tc = sum(1 for r in av if label_of(r) == COMMIT)
    ta = sum(1 for r in uv if label_of(r) == ABSTAIN)
    dec_acc = (tc + ta) / tot if tot else float("nan")

    def _round(x):
        return round(x, 4) if isinstance(x, float) and x == x else x

    return {
        "num_instances":         tot,
        "num_answerable":        len(answerable),
        "num_unanswerable":      len(unanswerable),
        "num_judge_errors":      err,
        "true_commitment_rate":  _round(rate(av, COMMIT)),
        "false_abstention_rate": _round(rate(av, ABSTAIN)),
        "true_abstention_rate":  _round(rate(uv, ABSTAIN)),
        "false_commitment_rate": _round(rate(uv, COMMIT)),
        "decision_accuracy":     _round(dec_acc),
    }


# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _row_nll(model, tokenizer, model_key: str, prompt: str, response: str,
             max_response_tokens: int) -> tuple[float, int]:
    """Returns (sum_nll, n_response_tokens). Cross-entropy on response tokens only."""
    full_ids, prompt_len = tokenise_chat_prompt_response(
        tokenizer, model_key, prompt, response,
    )
    response_len = len(full_ids) - prompt_len
    if response_len <= 0:
        return 0.0, 0
    if response_len > max_response_tokens:
        full_ids = full_ids[: prompt_len + max_response_tokens]
        response_len = max_response_tokens

    device = next(model.parameters()).device
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
    )
    logits = outputs.logits

    pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
    targets     = input_ids[:, prompt_len : prompt_len + response_len]

    ce = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="sum",
    )
    return float(ce.item()), response_len


def _summarise_ppl(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("n_tokens", 0) > 0 and r.get("ppl") is not None]
    if not valid:
        return {"num_instances": len(rows), "num_valid": 0}

    sum_nll = sum(r["sum_nll"] for r in valid)
    sum_tok = sum(r["n_tokens"] for r in valid)
    token_nll = sum_nll / sum_tok
    token_ppl = math.exp(token_nll)

    per_row_ppl = sorted(r["ppl"] for r in valid)

    def pct(p: float) -> float:
        i = max(0, min(len(per_row_ppl) - 1,
                       int(round((p / 100) * (len(per_row_ppl) - 1)))))
        return per_row_ppl[i]

    return {
        "num_instances":  len(rows),
        "num_valid":      len(valid),
        "n_tokens":       sum_tok,
        "token_nll":      round(token_nll, 6),
        "token_ppl":      round(token_ppl, 4),
        "row_ppl_mean":   round(mean(r["ppl"] for r in valid), 4),
        "row_ppl_median": round(median(r["ppl"] for r in valid), 4),
        "row_ppl_p95":    round(pct(95), 4),
        "row_ppl_max":    round(per_row_ppl[-1], 4),
    }


# ── Baseline comparison ───────────────────────────────────────────────────────

_ANSWERABILITY_KEYS = (
    "decision_accuracy", "true_commitment_rate",
    "false_abstention_rate", "true_abstention_rate",
    "false_commitment_rate",
)
_PPL_KEYS = ("token_ppl", "row_ppl_mean", "row_ppl_median")


def _attach_baseline_answerability(record: dict, baseline_dir: Path | None,
                                   dataset: str) -> None:
    if baseline_dir is None:
        return
    base_path = baseline_dir / f"{dataset}.json"
    base = _load_dataset_json(base_path)
    if base is None or "metrics" not in base:
        return
    bm = base["metrics"]
    cm = record["metrics"]
    deltas = {}
    for k in _ANSWERABILITY_KEYS:
        if k in bm and k in cm and isinstance(bm[k], (int, float)) and isinstance(cm[k], (int, float)):
            deltas[k] = round(cm[k] - bm[k], 4)
    record["baseline"]     = {k: bm.get(k) for k in bm}
    record["baseline_run"] = base.get("run")
    record["deltas"]       = deltas


def _attach_baseline_ppl(record: dict, baseline_dir: Path | None) -> None:
    if baseline_dir is None:
        return
    base_path = baseline_dir / "ultrachat.json"
    base = _load_dataset_json(base_path)
    if base is None or "metrics" not in base:
        return
    bm = base["metrics"]
    cm = record["metrics"]
    ratios = {}
    deltas = {}
    for k in _PPL_KEYS:
        bv, pv = bm.get(k), cm.get(k)
        if isinstance(bv, (int, float)) and isinstance(pv, (int, float)) and bv:
            ratios[f"{k}_ratio_post_over_base"]  = round(pv / bv, 4)
            deltas[f"{k}_delta_post_minus_base"] = round(pv - bv, 4)
    record["baseline"]     = {k: bm.get(k) for k in bm}
    record["baseline_run"] = base.get("run")
    record["ratios"]       = ratios
    record["deltas"]       = deltas


# ── Per-dataset answerability pass ────────────────────────────────────────────

def _build_answerability_record(rows: list[dict], judge_mod: dict, model_key: str,
                                result_name: str, dataset: str,
                                eval_path: Path) -> dict:
    return {
        "dataset":   dataset,
        "model":     cfg.MODEL_REGISTRY[model_key],
        "model_key": model_key,
        "run":       result_name,
        "pool":      str(eval_path.relative_to(cfg.REPO_ROOT)),
        "metrics":   _summarise_answerability(rows, judge_mod),
        "rows":      rows,
    }


def _run_dataset_answerability(args, model, tokenizer, model_key, result_name,
                               out_dir: Path, dataset: str, judge_mod: dict,
                               baseline_dir: Path | None) -> dict | None:
    eval_path = cfg.heldout_path(dataset)
    if not eval_path.exists():
        log.warning("  [%s] eval pool missing: %s — skipping", dataset, eval_path)
        return None

    out_path = out_dir / f"{dataset}.json"

    pool = load_jsonl(eval_path)
    if args.max_per_dataset is not None:
        pool = pool[: args.max_per_dataset]

    prior = _load_dataset_json(out_path) or {}
    rows: list[dict] = list(prior.get("rows") or [])
    done_ids = {
        r.get("id") for r in rows
        if r.get("completion") is not None and r.get("judge_label") is not None
    }
    if rows:
        log.info("  [%s] resume: %d rows on disk, %d already complete",
                 dataset, len(rows), len(done_ids))

    todo = [r for r in pool if r.get("id") not in done_ids]
    log.info("  [%s] pool: %d   to do: %d", dataset, len(pool), len(todo))

    def flush():
        rec = _build_answerability_record(
            rows, judge_mod, model_key, result_name, dataset, eval_path,
        )
        _attach_baseline_answerability(rec, baseline_dir, dataset)
        _save_dataset_json(out_path, rec)

    if not todo:
        flush()
        return _load_dataset_json(out_path)

    client = judge_mod["make_cerebras_client"]()
    counts = {"COMMIT": 0, "ABSTAIN": 0, "ERROR": 0, "OTHER": 0}
    gen_total = 0.0
    judge_total = 0.0
    progress = Progress(total=len(todo), desc=dataset, log_every=10)
    rows_since_save = 0

    for r in todo:
        row = dict(r)
        row["dataset"] = dataset
        if not r.get("answerable"):
            row["prompt"] = build_unanswerable_prompt(dataset, r)
        else:
            row["prompt"] = build_answerable_prompt(dataset, r)

        gen_t0 = time.time()
        try:
            row["completion"] = generate_greedy(
                model, tokenizer, model_key, row["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            log.warning("  [%s] gen error: %s", dataset, exc)
            row["completion"] = ""
        gen_dt = time.time() - gen_t0
        gen_total += gen_dt
        row["gen_time_s"] = round(gen_dt, 3)

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
                log.warning("  [%s] judge error: %s", dataset, exc)
                label, raw = "ERROR", str(exc)
        else:
            label, raw = "ERROR", "empty completion"
        judge_dt = time.time() - judge_t0
        judge_total += judge_dt

        row["judge_label"]      = label
        row["judge_model"]      = judge_mod["JUDGE_MODEL_ID"]
        row["judge_raw_output"] = raw
        row["judge_time_s"]     = round(judge_dt, 3)
        row["model"]            = cfg.MODEL_REGISTRY[model_key]
        row["run"]              = result_name

        bucket = label if label in counts else "OTHER"
        counts[bucket] += 1

        rows.append(row)
        rows_since_save += 1

        progress.tick(extras={
            "C": counts["COMMIT"], "A": counts["ABSTAIN"],
            "E": counts["ERROR"], "O": counts["OTHER"],
        })

        if rows_since_save >= args.summary_every:
            flush()
            rows_since_save = 0

    progress.done(extras={"C": counts["COMMIT"], "A": counts["ABSTAIN"],
                          "E": counts["ERROR"]})
    if progress.n:
        log.info("  [%s] gen total=%s avg=%.2fs/inst   judge total=%s avg=%.2fs/inst",
                 dataset, format_duration(gen_total), gen_total / progress.n,
                 format_duration(judge_total), judge_total / progress.n)

    flush()
    rec = _load_dataset_json(out_path) or {}
    m = rec.get("metrics", {})
    log.info("  [%s] acc=%.3f TCR=%.3f FAR=%.3f TAR=%.3f FCR=%.3f -> %s",
             dataset, m.get("decision_accuracy"), m.get("true_commitment_rate"),
             m.get("false_abstention_rate"), m.get("true_abstention_rate"),
             m.get("false_commitment_rate"), out_path)
    if "deltas" in rec and rec["deltas"]:
        log.info("  [%s] vs baseline -> %s",
                 dataset,
                 ", ".join(f"Δ{k}={v:+.3f}" for k, v in rec["deltas"].items()))
    return rec


# ── UltraChat perplexity pass ─────────────────────────────────────────────────

def _build_ppl_record(rows: list[dict], model_key: str, result_name: str,
                      eval_path: Path, max_response_tokens: int) -> dict:
    return {
        "dataset":             "ultrachat",
        "model":               cfg.MODEL_REGISTRY[model_key],
        "model_key":           model_key,
        "run":                 result_name,
        "pool":                str(eval_path.relative_to(cfg.REPO_ROOT)),
        "max_response_tokens": max_response_tokens,
        "metrics":             _summarise_ppl(rows),
        "rows":                rows,
    }


def _run_ultrachat_ppl(args, model, tokenizer, model_key, result_name,
                       out_dir: Path, baseline_dir: Path | None) -> dict | None:
    eval_path = cfg.heldout_path("ultrachat")
    if not eval_path.exists():
        log.warning("  [ultrachat] held-out missing: %s — skipping", eval_path)
        return None

    out_path = out_dir / "ultrachat.json"

    pool = load_jsonl(eval_path)
    if args.max_ppl_rows is not None:
        pool = pool[: args.max_ppl_rows]

    prior = _load_dataset_json(out_path) or {}
    rows: list[dict] = list(prior.get("rows") or [])
    done_ids = {r.get("id") for r in rows if r.get("n_tokens") is not None}
    if rows:
        log.info("  [ultrachat] resume: %d rows on disk", len(rows))

    todo = [r for r in pool if r.get("id") not in done_ids]
    log.info("  [ultrachat] pool: %d  to do: %d   max-response-tokens=%d",
             len(pool), len(todo), args.max_response_tokens)

    def flush():
        rec = _build_ppl_record(rows, model_key, result_name, eval_path,
                                args.max_response_tokens)
        _attach_baseline_ppl(rec, baseline_dir)
        _save_dataset_json(out_path, rec)

    if not todo:
        flush()
        return _load_dataset_json(out_path)

    progress = Progress(total=len(todo), desc="ultrachat", log_every=25)
    fwd_total = 0.0
    skipped_empty = 0
    rows_since_save = 0

    for r in todo:
        prompt   = (r.get("prompt") or "").strip()
        response = (r.get("response") or "").strip()
        if not prompt or not response:
            skipped_empty += 1
            progress.tick()
            continue

        t0 = time.time()
        try:
            sum_nll, n_tok = _row_nll(
                model, tokenizer, model_key, prompt, response,
                max_response_tokens=args.max_response_tokens,
            )
        except Exception as exc:
            log.warning("  [ultrachat] error on id=%s: %s", r.get("id"), exc)
            sum_nll, n_tok = 0.0, 0
        dt = time.time() - t0
        fwd_total += dt

        if n_tok > 0:
            mean_nll = sum_nll / n_tok
            ppl = math.exp(mean_nll)
        else:
            mean_nll = float("nan")
            ppl = float("nan")

        out_row = {
            "id":         r.get("id"),
            "n_tokens":   n_tok,
            "sum_nll":    round(sum_nll, 4),
            "mean_nll":   round(mean_nll, 6) if n_tok > 0 else None,
            "ppl":        round(ppl, 4) if n_tok > 0 else None,
            "fwd_time_s": round(dt, 3),
            "model":      cfg.MODEL_REGISTRY[model_key],
            "run":        result_name,
        }
        rows.append(out_row)
        rows_since_save += 1
        progress.tick(extras={"ppl": f"{ppl:.2f}" if n_tok > 0 else "—"})

        if rows_since_save >= args.summary_every:
            flush()
            rows_since_save = 0

    progress.done()
    if progress.n:
        log.info("  [ultrachat] forward total=%s avg=%.3fs/row   skipped (empty)=%d",
                 format_duration(fwd_total), fwd_total / max(progress.n, 1),
                 skipped_empty)

    flush()
    rec = _load_dataset_json(out_path) or {}
    m = rec.get("metrics", {})
    log.info("  [ultrachat] token_ppl=%s row_ppl_mean=%s row_ppl_p95=%s n=%s -> %s",
             m.get("token_ppl"), m.get("row_ppl_mean"), m.get("row_ppl_p95"),
             m.get("num_valid"), out_path)
    if "ratios" in rec and rec["ratios"]:
        log.info("  [ultrachat] vs baseline -> %s",
                 ", ".join(f"{k}={v}" for k, v in rec["ratios"].items()))
    return rec


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    pipeline_t0 = time.time()
    if args.run_dir is not None:
        run_dir: Path = args.run_dir.resolve()
        result_name = run_dir.name
        mode = "trained"
    else:
        run_dir = None
        result_name = f"baseline_{args.model}"
        mode = "baseline"

    out_dir = cfg.results_dir_for(result_name)
    baseline_dir: Path | None = None
    if args.baseline:
        baseline_dir = args.baseline.resolve()
        if not baseline_dir.is_dir():
            log.warning("  --baseline expects a results directory; got %s — ignoring",
                        baseline_dir)
            baseline_dir = None

    log.info("STEP 5 — EVALUATE  mode=%s  name=%s", mode, result_name)
    log.info("  results_dir: %s", out_dir)
    if baseline_dir:
        log.info("  baseline_dir: %s", baseline_dir)

    with Stopwatch("model load"):
        if mode == "trained":
            model, tokenizer, model_key = _load_adapter_model(run_dir)
            log.info("  Adapter loaded for %s (%s)",
                     model_key, cfg.MODEL_REGISTRY[model_key])
        else:
            model, tokenizer, model_key = _load_base_model(args.model)
            log.info("  Base model loaded: %s (%s) — no adapter",
                     model_key, cfg.MODEL_REGISTRY[model_key])

    if not args.skip_judge:
        judge_mod = _import_judge()
        for dataset in ("kuq", "squad"):
            _run_dataset_answerability(
                args, model, tokenizer, model_key, result_name, out_dir,
                dataset, judge_mod, baseline_dir,
            )
    else:
        log.info("  [judge] skipped (--skip-judge)")

    if not args.skip_ppl:
        _run_ultrachat_ppl(
            args, model, tokenizer, model_key, result_name, out_dir, baseline_dir,
        )
    else:
        log.info("  [ultrachat] skipped (--skip-ppl)")

    log.info("STEP 5 done in %s. Outputs in %s",
             format_duration(time.time() - pipeline_t0), out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5: evaluate UOC run or zero-shot baseline.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path,
                   help="Run directory (step4_train/data/runs/<run_name>) for trained eval.")
    g.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()),
                   help="Model key for zero-shot baseline eval (no adapter).")
    p.add_argument("--max-new-tokens", type=int, default=cfg.DEFAULT_MAX_NEW_TOKENS,
                   help="Greedy decode cap for the answerability eval.")
    p.add_argument("--max-per-dataset", type=int, default=None,
                   help="Cap answerability eval to N rows per dataset (smoke).")
    p.add_argument("--max-response-tokens", type=int, default=DEFAULT_MAX_RESPONSE_TOKENS,
                   help=f"Cap UltraChat response token length (default {DEFAULT_MAX_RESPONSE_TOKENS}).")
    p.add_argument("--max-ppl-rows", type=int, default=None,
                   help="Cap UltraChat ppl eval to N rows (smoke).")
    p.add_argument("--summary-every", type=int, default=DEFAULT_SUMMARY_EVERY,
                   help=f"Atomically rewrite the dataset JSON every N rows "
                        f"(default {DEFAULT_SUMMARY_EVERY}).")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip the answerability (KUQ + SQuAD) eval.")
    p.add_argument("--skip-ppl", action="store_true",
                   help="Skip the UltraChat perplexity eval.")
    p.add_argument("--baseline", type=Path, default=None,
                   help="Baseline RESULTS DIRECTORY for delta comparison "
                        "(e.g. step5_evaluate/data/results/baseline_qwen_instruct).")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
