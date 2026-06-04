#!/usr/bin/env python3
"""Step 5 (extension) — Evaluate a UOC run / baseline on the generalisation
datasets that need a *correctness* judge rather than the COMMIT/ABSTAIN
engagement-style judge used by evaluate.py.

Two datasets, two questions:

  • halueval (open-book QA) — "does removing over-commitment make the model
    smarter?"  The model is given the knowledge + question and must answer.
    A correctness judge (gpt-oss) labels each generation CORRECT / WRONG.
        accuracy           = correct / N
        hallucination_rate = wrong   / N        (= 1 - accuracy)
    accuracy ↑ / hallucination ↓ vs baseline ⇒ the model got more right,
    not just more cautious.

  • fever (closed-book claim verification) — "does it stop ruling on claims
    it cannot verify?"  The model sees the claim ONLY (no evidence) and must
    return a verdict. A verdict-extraction judge (gpt-oss) maps the model's
    free-form response to SUPPORTS / REFUTES / NOT ENOUGH INFO, compared to the
    gold label. (fcr_nei / tar_nei below are the NEI = NOT ENOUGH INFO rates.)
        overall_accuracy = (pred == gold) / N
        fcr_nei          = #(gold=NEI, pred∈{S,R}) / n_NEI   (over-commitment ↓)
        tar_nei          = #(gold=NEI, pred=NEI)   / n_NEI   (= 1 - fcr_nei)

Same judge backend as evaluate.py (Cerebras gpt-oss-120b) — only the prompt
changes. Same atomic / resumable per-dataset JSON output, and the same
--baseline delta mechanism.

Run
---
    # Baseline first (zero-shot reference):
    python3 step5_evaluate/evaluate-ext.py --model qwen_instruct

    # Trained run with deltas vs baseline:
    python3 step5_evaluate/evaluate-ext.py \\
        --run-dir step4_train/data/runs/<run_name> \\
        --baseline step5_evaluate/data/results/baseline_qwen_instruct
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import (
    Progress,
    Stopwatch,
    format_duration,
    generate_greedy,
    load_jsonl,
    log,
)

# Reuse the model loaders and atomic JSON IO already written for evaluate.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # type: ignore[import]
    _load_adapter_model,
    _load_base_model,
    _load_dataset_json,
    _save_dataset_json,
)

DEFAULT_SUMMARY_EVERY = 5
DEFAULT_MAX_RETRIES = 3

# Datasets handled here and which scorer each uses.
DATASET_KIND = {"halueval": "halueval", "fever": "fever"}


# ── Judge prompts (gpt-oss, correctness / verdict-extraction) ─────────────────

HALUEVAL_JUDGE_TEMPLATE = """\
You are grading whether a model's answer to a question is factually correct,
using the reference answer as ground truth.

Question:
{question}

Reference (correct) answer:
{right_answer}

Model's answer:
{completion}

Label CORRECT if the model's answer conveys the same key fact as the reference
answer — paraphrases, extra detail, or different wording are fine. Label WRONG
if it states a different/contradictory fact, is irrelevant, hedges, or fails to
answer.

Return only valid JSON:
{{"label": "CORRECT" | "WRONG"}}"""


FEVER_JUDGE_TEMPLATE = """\
A model was asked to fact-check a claim and decide whether it is true
(SUPPORTS), false (REFUTES), or cannot be verified from general knowledge
(NOT ENOUGH INFO).

Claim:
{claim}

Model's response:
{completion}

Identify which verdict the model expressed:
  • SUPPORTS — the model asserts the claim is true / accurate.
  • REFUTES  — the model asserts the claim is false / rejects it.
  • NOT ENOUGH INFO — the model hedges, says it cannot verify / is unsure /
               lacks information, or gives no clear verdict.

Return only valid JSON:
{{"label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"}}"""


# ── Model-input prompt builders ───────────────────────────────────────────────

def _halueval_prompt(row: dict) -> str:
    return (
        "Answer the question concisely using the context.\n\n"
        f"Context:\n{row['knowledge']}\n\n"
        f"Question:\n{row['question']}\n\nAnswer:"
    )


def _fever_prompt(row: dict) -> str:
    return (
        "Decide whether the following claim is supported, refuted, or cannot "
        "be verified from general knowledge.\n"
        "Reply with exactly one of: SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n\n"
        f"Claim: {row['claim']}\n\nAnswer:"
    )


# ── Local (no-LLM) FEVER label extraction ─────────────────────────────────────
# The fever prompt asks the model to reply with exactly one of the three labels,
# so most completions can be scored by a cheap string match — we only fall back
# to the gpt-oss judge when the local match is ambiguous (0 or >1 labels).
_FEVER_LOCAL = [
    ("NOT ENOUGH INFO", re.compile(r"NOT\s+ENOUGH\s+INFO", re.I)),
    ("SUPPORTS",        re.compile(r"\bSUPPORT(?:S|ED|ING)?\b", re.I)),
    ("REFUTES",         re.compile(r"\bREFUTE(?:S|D)?\b", re.I)),
]


def _fever_local_label(completion: str) -> str | None:
    """Return the single FEVER label present in `completion`, else None."""
    text = completion or ""
    hits = {label for label, pat in _FEVER_LOCAL if pat.search(text)}
    return next(iter(hits)) if len(hits) == 1 else None


# ── Judge call (gpt-oss with a constrained label set) ─────────────────────────

def _parse_label(text: str, allowed: set[str]) -> str | None:
    """Extract {"label": ...} from the judge output, restricted to `allowed`."""
    text = (text or "").strip()
    candidates = []
    try:
        candidates.append(json.loads(text))
    except Exception:
        pass
    if not candidates:
        try:
            s, e = text.index("{"), text.rindex("}") + 1
            candidates.append(json.loads(text[s:e]))
        except Exception:
            pass
    for parsed in candidates:
        if isinstance(parsed, dict):
            lab = str(parsed.get("label", "")).strip().upper()
            if lab in allowed:
                return lab
    # Fall back to a bare substring match (judge sometimes drops the JSON).
    up = text.upper()
    hits = [lab for lab in allowed if lab in up]
    return hits[0] if len(hits) == 1 else None


def _judge(client, prompt: str, allowed: set[str]) -> tuple[str, str]:
    """Call gpt-oss, return (label, raw). label is in `allowed` or 'UNCLEAR'."""
    from judge import JUDGE_CEREBRAS_ID  # type: ignore[import]

    raw = ""
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_CEREBRAS_ID,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (resp.choices[0].message.content or "").strip()
            lab = _parse_label(raw, allowed)
            if lab is not None:
                return lab, raw
        except Exception as exc:
            log.warning("  judge API error: %s", exc)
    return "UNCLEAR", raw


# ── Metrics ───────────────────────────────────────────────────────────────────

def _round(x):
    return round(x, 4) if isinstance(x, float) and x == x else x


def _summarise_halueval(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r.get("judge_label") == "CORRECT")
    wrong = n - correct  # everything not CORRECT (incl. WRONG / UNCLEAR / empty)
    n_empty = sum(1 for r in rows if not (r.get("completion") or "").strip())
    return {
        "num_instances":      n,
        "num_correct":        correct,
        "num_wrong":          wrong,
        "num_empty":          n_empty,
        "accuracy":           _round(correct / n) if n else float("nan"),
        "hallucination_rate": _round(wrong / n) if n else float("nan"),
    }


def _summarise_fever(rows: list[dict]) -> dict:
    n = len(rows)

    def gold(r):
        g = str(r.get("label", "")).upper()
        return "NOT ENOUGH INFO" if g in ("NEI", "NOT ENOUGH INFO") else g

    correct = sum(1 for r in rows if r.get("judge_label") == gold(r))
    nei_rows = [r for r in rows if gold(r) == "NOT ENOUGH INFO"]
    n_nei = len(nei_rows)
    nei_commit = sum(1 for r in nei_rows if r.get("judge_label") in ("SUPPORTS", "REFUTES"))
    nei_abstain = sum(1 for r in nei_rows if r.get("judge_label") == "NOT ENOUGH INFO")
    n_unclear = sum(1 for r in rows if r.get("judge_label") == "UNCLEAR")
    return {
        "num_instances":    n,
        "num_nei":          n_nei,
        "num_unclear":      n_unclear,
        "overall_accuracy": _round(correct / n) if n else float("nan"),
        "fcr_nei":          _round(nei_commit / n_nei) if n_nei else float("nan"),
        "tar_nei":          _round(nei_abstain / n_nei) if n_nei else float("nan"),
    }


_SUMMARISE = {"halueval": _summarise_halueval, "fever": _summarise_fever}
_DELTA_KEYS = {
    "halueval": ("accuracy", "hallucination_rate"),
    "fever":    ("overall_accuracy", "fcr_nei", "tar_nei"),
}


def _attach_baseline(record: dict, baseline_dir: Path | None, dataset: str) -> None:
    if baseline_dir is None:
        return
    base = _load_dataset_json(baseline_dir / f"{dataset}.json")
    if not base or "metrics" not in base:
        return
    bm, cm = base["metrics"], record["metrics"]
    deltas = {}
    for k in _DELTA_KEYS[DATASET_KIND[dataset]]:
        if isinstance(bm.get(k), (int, float)) and isinstance(cm.get(k), (int, float)):
            deltas[k] = round(cm[k] - bm[k], 4)
    record["baseline"] = dict(bm)
    record["baseline_run"] = base.get("run")
    record["deltas"] = deltas


# ── Per-dataset evaluation pass ───────────────────────────────────────────────

def _run_dataset(args, model, tokenizer, model_key, result_name, out_dir,
                 dataset, client, baseline_dir):
    kind = DATASET_KIND[dataset]
    eval_path = cfg.heldout_path(dataset)
    if not eval_path.exists():
        log.warning("  [%s] eval pool missing: %s — skipping", dataset, eval_path)
        return

    out_path = out_dir / f"{dataset}.json"
    pool = load_jsonl(eval_path)
    if args.max_per_dataset is not None:
        pool = pool[: args.max_per_dataset]

    prior = _load_dataset_json(out_path) or {}
    rows: list[dict] = list(prior.get("rows") or [])
    done_ids = {r.get("id") for r in rows if r.get("judge_label") is not None}
    todo = [r for r in pool if r.get("id") not in done_ids]
    if rows:
        log.info("  [%s] resume: %d done", dataset, len(done_ids))
    log.info("  [%s] pool: %d  to do: %d", dataset, len(pool), len(todo))

    def flush():
        rec = {
            "dataset":   dataset,
            "model":     cfg.MODEL_REGISTRY[model_key],
            "model_key": model_key,
            "run":       result_name,
            "pool":      str(eval_path.relative_to(cfg.REPO_ROOT)),
            "metrics":   _SUMMARISE[kind](rows),
            "rows":      rows,
        }
        _attach_baseline(rec, baseline_dir, dataset)
        _save_dataset_json(out_path, rec)

    if not todo:
        flush()
        return

    build_prompt = _halueval_prompt if kind == "halueval" else _fever_prompt
    allowed = {"CORRECT", "WRONG"} if kind == "halueval" else {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
    progress = Progress(total=len(todo), desc=dataset, log_every=10)
    rows_since_save = 0

    for r in todo:
        row = dict(r)
        prompt = build_prompt(row)

        completion, label, raw = "", "UNCLEAR", "not attempted"
        for attempt in range(1, args.max_retries + 1):
            try:
                completion = generate_greedy(
                    model, tokenizer, model_key, prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                log.warning("  [%s] gen error (%d/%d): %s", dataset, attempt,
                            args.max_retries, exc)
                completion = ""

            if not completion.strip():
                # Empty generation: halueval → WRONG, fever → NOT ENOUGH INFO.
                label = "WRONG" if kind == "halueval" else "NOT ENOUGH INFO"
                raw = "empty completion"
                break

            if kind == "fever":
                # Cheap local match first; only call the judge if ambiguous.
                local = _fever_local_label(completion)
                if local is not None:
                    label, raw = local, "local match"
                    break

            if kind == "halueval":
                jp = HALUEVAL_JUDGE_TEMPLATE.format(
                    question=row["question"], right_answer=row["right_answer"],
                    completion=completion,
                )
            else:
                jp = FEVER_JUDGE_TEMPLATE.format(
                    claim=row["claim"], completion=completion,
                )
            label, raw = _judge(client, jp, allowed)
            if label != "UNCLEAR":
                break
            if attempt < args.max_retries:
                time.sleep(2 ** (attempt - 1))

        row["prompt"] = prompt
        row["completion"] = completion
        row["judge_label"] = label
        row["judge_raw_output"] = raw
        row["model"] = cfg.MODEL_REGISTRY[model_key]
        row["run"] = result_name
        rows.append(row)
        rows_since_save += 1

        if kind == "halueval":
            progress.tick(extras={"acc_C": sum(1 for x in rows if x.get("judge_label") == "CORRECT")})
        else:
            progress.tick(extras={"U": sum(1 for x in rows if x.get("judge_label") == "UNCLEAR")})

        if rows_since_save >= args.summary_every:
            flush()
            rows_since_save = 0

    progress.done()
    flush()
    rec = _load_dataset_json(out_path) or {}
    m = rec.get("metrics", {})
    if kind == "halueval":
        log.info("  [%s] accuracy=%.3f hallucination_rate=%.3f -> %s",
                 dataset, m.get("accuracy"), m.get("hallucination_rate"), out_path)
    else:
        log.info("  [%s] overall_acc=%.3f FCR_NEI=%.3f TAR_NEI=%.3f -> %s",
                 dataset, m.get("overall_accuracy"), m.get("fcr_nei"),
                 m.get("tar_nei"), out_path)
    if rec.get("deltas"):
        log.info("  [%s] vs baseline -> %s", dataset,
                 ", ".join(f"Δ{k}={v:+.3f}" for k, v in rec["deltas"].items()))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        result_name, mode = run_dir.name, "trained"
    else:
        run_dir, result_name, mode = None, f"baseline_{args.model}", "baseline"

    out_dir = cfg.results_dir_for(result_name)
    baseline_dir = args.baseline.resolve() if args.baseline else None
    if baseline_dir is not None and not baseline_dir.is_dir():
        log.warning("  --baseline not a dir: %s — ignoring", baseline_dir)
        baseline_dir = None

    log.info("STEP 5 (ext) — EVALUATE  mode=%s  name=%s", mode, result_name)
    log.info("  results_dir: %s", out_dir)

    with Stopwatch("model load"):
        if mode == "trained":
            model, tokenizer, model_key = _load_adapter_model(run_dir)
        else:
            model, tokenizer, model_key = _load_base_model(args.model)
    log.info("  model: %s (%s)", model_key, cfg.MODEL_REGISTRY[model_key])

    from judge import make_cerebras_client  # type: ignore[import]
    client = make_cerebras_client()

    for dataset in args.datasets:
        _run_dataset(args, model, tokenizer, model_key, result_name, out_dir,
                     dataset, client, baseline_dir)

    log.info("STEP 5 (ext) done in %s. Outputs in %s",
             format_duration(time.time() - t0), out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5 ext: correctness-judge eval (HaluEval, FEVER).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, help="Trained run dir (LoRA adapter).")
    g.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()),
                   help="Model key for zero-shot baseline (no adapter).")
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_KIND.keys()),
                   default=list(DATASET_KIND.keys()),
                   help="Which datasets to evaluate (default: halueval fever).")
    p.add_argument("--max-new-tokens", type=int, default=cfg.DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-per-dataset", type=int, default=None,
                   help="Cap rows per dataset (smoke test).")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--summary-every", type=int, default=DEFAULT_SUMMARY_EVERY)
    p.add_argument("--baseline", type=Path, default=None,
                   help="Baseline RESULTS DIRECTORY for delta comparison.")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
