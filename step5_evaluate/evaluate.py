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

  (1) Answerability (Categories A/B/C/D)
      For each held-out KUQ + SQuAD prompt, generate one greedy completion,
      judge it via judge.py (Cerebras gpt-oss-120b), and report TCR / FCR /
      TAR / FAR / decision accuracy.

  (2) UltraChat preservation (Category E, perplexity)
      For each held-out UltraChat (prompt, response) pair, compute the
      per-token cross-entropy of the response (capped to --max-response-tokens,
      default 256) under the model. Report token-level corpus perplexity.
      Healthy preservation: PPL_post / PPL_base ≈ 1.

Reads:  step5_evaluate/data/heldout/{kuq,squad,ultrachat}.jsonl
        step4_train/data/runs/<run_name>/        (adapter + training_config.json, trained mode only)
Writes: step5_evaluate/data/results/<name>/
            generations.jsonl                    answerability per-row
            answerability_metrics.jsonl          answerability summary
            perplexity_ultrachat.jsonl           ppl per-row
            perplexity_ultrachat_summary.json    ppl summary
            comparison.json                      (optional, vs --baseline)

`<name>` is the run directory's name in trained mode, or
`baseline_<model_key>` in baseline mode.

Crash-safe & resumable
----------------------
Each row is appended to its file as soon as it has a result. Re-running
this script skips any row already complete on disk (matched by id within
its file). The two evaluations are independent — you can use --skip-judge
or --skip-ppl to run only one.

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
    append_jsonl,
    build_answerable_prompt,
    build_unanswerable_prompt,
    format_duration,
    generate_greedy,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    tokenise_chat_prompt_response,
    write_jsonl,
)


DEFAULT_MAX_RESPONSE_TOKENS = 256


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
    return model, tokenizer, model_key, train_cfg


def _load_base_model(model_key: str):
    """Load the bare base model with no adapter — zero-shot baseline."""
    model, tokenizer = load_model_and_tokenizer(model_key, eval_only=True)
    model.eval()
    train_cfg = {"model_key": model_key, "method": "baseline (no adapter)"}
    return model, tokenizer, model_key, train_cfg


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


def _compute_answerability_metrics(rows: list[dict], dataset: str, judge_mod: dict) -> dict:
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


def _row_key(dataset: str, row: dict) -> str:
    return f"{dataset}::{row.get('id', row.get('source_index', '?'))}"


def _run_answerability(args, model, tokenizer, model_key, result_name, out_dir, judge_mod):
    """Generate + judge KUQ + SQuAD held-out prompts. Returns metrics list."""
    gen_path = out_dir / "generations.jsonl"
    existing_rows: list[dict] = load_jsonl(gen_path) if gen_path.exists() else []
    done_keys: set[str] = {
        _row_key(r["dataset"], r) for r in existing_rows
        if r.get("completion") is not None and r.get("judge_label") is not None
    }
    if existing_rows:
        log.info("  [judge] resume: %d rows on disk, %d already complete",
                 len(existing_rows), len(done_keys))

    client = judge_mod["make_cerebras_client"]()

    todo: list[tuple[str, dict]] = []
    for dataset in ("kuq", "squad"):
        eval_path = cfg.heldout_path(dataset)
        if not eval_path.exists():
            log.warning("  [judge] eval pool missing: %s — skipping", eval_path)
            continue
        pool = load_jsonl(eval_path)
        if args.max_per_dataset is not None:
            pool = pool[: args.max_per_dataset]
        for r in pool:
            if _row_key(dataset, r) in done_keys:
                continue
            todo.append((dataset, r))
        log.info("  [judge] %s: %d in pool, %d to do",
                 dataset, len(pool), sum(1 for d, _ in todo if d == dataset))

    counts = {"COMMIT": 0, "ABSTAIN": 0, "ERROR": 0, "OTHER": 0}
    gen_total = 0.0
    judge_total = 0.0
    progress = Progress(total=len(todo), desc="answerability", log_every=10)

    for dataset, r in todo:
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
            log.warning("  [judge] gen error: %s", exc)
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
                log.warning("  [judge] judge error: %s", exc)
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

        append_jsonl(gen_path, row)
        existing_rows.append(row)

        progress.tick(extras={
            "C": counts["COMMIT"], "A": counts["ABSTAIN"],
            "E": counts["ERROR"], "O": counts["OTHER"],
        })

    progress.done(extras={"C": counts["COMMIT"], "A": counts["ABSTAIN"],
                          "E": counts["ERROR"]})
    if progress.n:
        log.info("  [judge] gen total=%s avg=%.2fs/inst   judge total=%s avg=%.2fs/inst",
                 format_duration(gen_total), gen_total / progress.n,
                 format_duration(judge_total), judge_total / progress.n)

    metrics: list[dict] = []
    for dataset in ("kuq", "squad"):
        m = _compute_answerability_metrics(existing_rows, dataset, judge_mod)
        m["model"] = cfg.MODEL_REGISTRY[model_key]
        m["run"]   = result_name
        metrics.append(m)
        log.info("  [judge] %-6s acc=%.3f TCR=%.3f FAR=%.3f TAR=%.3f FCR=%.3f",
                 dataset, m["decision_accuracy"], m["true_commitment_rate"],
                 m["false_abstention_rate"], m["true_abstention_rate"],
                 m["false_commitment_rate"])

    metrics_path = out_dir / "answerability_metrics.jsonl"
    write_jsonl(metrics_path, metrics)
    log.info("  [judge] Wrote -> %s", metrics_path)
    return metrics


# ── Perplexity (UltraChat preservation) ───────────────────────────────────────

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
    valid = [r for r in rows if r.get("n_tokens", 0) > 0]
    if not valid:
        return {"n_rows": 0}

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
        "n_rows":         len(valid),
        "n_tokens":       sum_tok,
        "token_nll":      round(token_nll, 6),
        "token_ppl":      round(token_ppl, 4),
        "row_ppl_mean":   round(mean(r["ppl"] for r in valid), 4),
        "row_ppl_median": round(median(r["ppl"] for r in valid), 4),
        "row_ppl_p95":    round(pct(95), 4),
        "row_ppl_max":    round(per_row_ppl[-1], 4),
    }


def _run_perplexity(args, model, tokenizer, model_key, result_name, out_dir):
    """Compute UltraChat per-token PPL; writes per-row jsonl + summary json."""
    pool_path = cfg.heldout_path("ultrachat")
    if not pool_path.exists():
        log.warning("  [ppl] held-out UltraChat missing: %s — skipping", pool_path)
        return None

    pool = load_jsonl(pool_path)
    if args.max_ppl_rows is not None:
        pool = pool[: args.max_ppl_rows]

    rows_path = out_dir / "perplexity_ultrachat.jsonl"
    existing: list[dict] = load_jsonl(rows_path) if rows_path.exists() else []
    done_ids = {r.get("id") for r in existing if r.get("n_tokens") is not None}
    if existing:
        log.info("  [ppl] resume: %d rows on disk", len(existing))

    todo = [r for r in pool if r.get("id") not in done_ids]
    log.info("  [ppl] pool: %d rows  to do: %d   max-response-tokens=%d",
             len(pool), len(todo), args.max_response_tokens)

    progress = Progress(total=len(todo), desc="ultrachat ppl", log_every=25)
    fwd_total = 0.0
    skipped_empty = 0

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
            log.warning("  [ppl] error on id=%s: %s", r.get("id"), exc)
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
        append_jsonl(rows_path, out_row)
        existing.append(out_row)
        progress.tick(extras={"ppl": f"{ppl:.2f}" if n_tok > 0 else "—"})

    progress.done()
    if progress.n:
        log.info("  [ppl] forward total=%s avg=%.3fs/row   skipped (empty)=%d",
                 format_duration(fwd_total), fwd_total / max(progress.n, 1),
                 skipped_empty)

    summary = _summarise_ppl(existing)
    summary["model"]               = cfg.MODEL_REGISTRY[model_key]
    summary["run"]                 = result_name
    summary["max_response_tokens"] = args.max_response_tokens
    summary["pool"]                = str(pool_path.relative_to(cfg.REPO_ROOT))

    summary_path = out_dir / "perplexity_ultrachat_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("  [ppl] token_ppl=%s   row_ppl_mean=%s   row_ppl_p95=%s   n_rows=%s",
             summary.get("token_ppl"), summary.get("row_ppl_mean"),
             summary.get("row_ppl_p95"), summary.get("n_rows"))
    log.info("  [ppl] Wrote -> %s", summary_path)
    return summary


# ── Comparison vs baseline ────────────────────────────────────────────────────

def _write_comparison(args, out_dir, post_metrics: list[dict] | None,
                      post_ppl: dict | None) -> None:
    if not args.baseline:
        return
    base_dir: Path = args.baseline
    if not base_dir.is_dir():
        log.warning("  [cmp] --baseline expects a results directory; got %s — skipping comparison",
                    base_dir)
        return

    comparison: dict = {"baseline_dir": str(base_dir)}

    # Answerability deltas
    base_metrics_path = base_dir / "answerability_metrics.jsonl"
    if post_metrics and base_metrics_path.exists():
        base_metrics = load_jsonl(base_metrics_path)
        deltas_per_dataset = []
        for m in post_metrics:
            base = next((b for b in base_metrics if b["dataset"] == m["dataset"]
                         and b.get("model") == m["model"]), None)
            if base is None:
                continue
            deltas_per_dataset.append({
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
        comparison["answerability"] = deltas_per_dataset

    # PPL deltas / ratios
    base_ppl_path = base_dir / "perplexity_ultrachat_summary.json"
    if post_ppl and base_ppl_path.exists():
        try:
            base_ppl = json.loads(base_ppl_path.read_text())
            ppl_block = {"baseline": base_ppl, "post": post_ppl}
            for key in ("token_ppl", "row_ppl_mean", "row_ppl_median"):
                bv, pv = base_ppl.get(key), post_ppl.get(key)
                if bv and pv:
                    ppl_block[f"{key}_ratio_post_over_base"]  = round(pv / bv, 4)
                    ppl_block[f"{key}_delta_post_minus_base"] = round(pv - bv, 4)
            comparison["perplexity_ultrachat"] = ppl_block
        except Exception as exc:
            log.warning("  [cmp] could not read baseline ppl summary: %s", exc)

    if "answerability" not in comparison and "perplexity_ultrachat" not in comparison:
        log.warning("  [cmp] no comparable artefacts found in %s — skipping", base_dir)
        return

    comp_path = out_dir / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    log.info("  [cmp] Wrote comparison -> %s", comp_path)


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

    log.info("STEP 5 — EVALUATE  mode=%s  name=%s", mode, result_name)
    log.info("  results_dir: %s", out_dir)

    with Stopwatch("model load"):
        if mode == "trained":
            model, tokenizer, model_key, _ = _load_adapter_model(run_dir)
            log.info("  Adapter loaded for %s (%s)",
                     model_key, cfg.MODEL_REGISTRY[model_key])
        else:
            model, tokenizer, model_key, _ = _load_base_model(args.model)
            log.info("  Base model loaded: %s (%s) — no adapter",
                     model_key, cfg.MODEL_REGISTRY[model_key])

    post_metrics: list[dict] | None = None
    post_ppl: dict | None = None

    # 1) Answerability
    if not args.skip_judge:
        judge_mod = _import_judge()
        post_metrics = _run_answerability(
            args, model, tokenizer, model_key, result_name, out_dir, judge_mod,
        )
    else:
        log.info("  [judge] skipped (--skip-judge)")

    # 2) UltraChat perplexity
    if not args.skip_ppl:
        post_ppl = _run_perplexity(
            args, model, tokenizer, model_key, result_name, out_dir,
        )
    else:
        log.info("  [ppl] skipped (--skip-ppl)")

    # 3) Comparison vs baseline
    _write_comparison(args, out_dir, post_metrics, post_ppl)

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
