#!/usr/bin/env python3
"""Step 6 — Extract late-layer hidden states from a trained UOC model.

Loads a base model + saved LoRA adapter (from a step-4 run directory) and
runs the same five forward-pass sets as step 1, saving an activations bundle
in the identical format so that all downstream visualization tools (a_slides/)
work without modification.

Output
------
step6_extract_trained_activations/data/activations_<run_name>.pt

  Same keys as step-1 output:
    "model_key", "run_name", "layers", "k_answer_tokens",
    "h_A"  (N_F, L, D)   over-commitment
    "h_B"  (N_F, L, D)   legitimate-abstention
    "h_C"  (N_A, L, D)   legitimate-commitment
    "h_D"  (N_A, L, D)   over-abstention
    "h_E"  (N_R, L, D)   general utility
    "meta_A", "meta_B", "meta_C", "meta_D"

Run
---
    python step6_extract_trained_activations/extract_trained.py \\
        --run-dir step4_train/data/runs/qwen_instruct_uoc_r32_lam2_ep3_lr3e-05

    # limit to 200 examples per set for a quick smoke-test
    python step6_extract_trained_activations/extract_trained.py \\
        --run-dir step4_train/data/runs/qwen_instruct_uoc_r32_lam2_ep3_lr3e-05 \\
        --max-per-set 200

    # force re-extraction even if output already exists
    python step6_extract_trained_activations/extract_trained.py \\
        --run-dir step4_train/data/runs/qwen_instruct_uoc_r32_lam2_ep3_lr3e-05 \\
        --rebuild
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from _common import (
    Progress,
    Stopwatch,
    build_answerable_prompt,
    build_unanswerable_prompt,
    format_duration,
    forward_hidden_states,
    free_model,
    layer_indices_for,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    mean_answer_activation,
    tokenise_chat_prompt_response,
    tokenise_prompt_plus_answer,
)

OUT_DIR = ROOT / "step6_extract_trained_activations" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers (mirrors step 1 helpers exactly) ──────────────────────────────────

def _judge_label(row: dict) -> str:
    raw = (row.get("judge_label") or "").strip().upper()
    if raw in ("COMMIT", "COMMITTED"):
        return "COMMIT"
    if raw in ("ABSTAIN", "ABSTAINED", "ABSTANTED"):
        return "ABSTAIN"
    return raw or "UNLABELLED"


def _load_forget_pool(model_key: str) -> list[dict]:
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.forget_path(model_key, dataset)
        if not path.exists():
            log.warning("  forget pool missing: %s", path)
            continue
        for row in load_jsonl(path):
            row["dataset"] = dataset
            row["judge_label"] = _judge_label(row)
            pool.append(row)
    return pool


def _load_answerable_pool() -> list[dict]:
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.sampled_answerable_path(dataset)
        if not path.exists():
            log.warning("  sampled answerable missing: %s", path)
            continue
        for row in load_jsonl(path):
            row["dataset"] = dataset
            pool.append(row)
    return pool


def _load_retain_general() -> list[dict]:
    return load_jsonl(cfg.sampled_general_path())


def _partial_dir(run_name: str) -> Path:
    p = OUT_DIR / f"_partial_{run_name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _partial_path(run_name: str, set_name: str) -> Path:
    return _partial_dir(run_name) / f"{set_name}.pt"


def _load_or_extract_set(set_name: str, extractor, *, run_name: str,
                         rebuild: bool) -> dict:
    pp = _partial_path(run_name, set_name)
    if not rebuild and pp.exists():
        log.info("  set %-22s   (cached) loading from %s", set_name, pp)
        return torch.load(pp, map_location="cpu", weights_only=False)
    with Stopwatch(f"set {set_name}"):
        result = extractor()
    torch.save(result, pp)
    log.info("  set %-22s   checkpoint -> %s", set_name, pp)
    return result


# ── per-set forward passes (identical to step 1) ─────────────────────────────

def _extract_means_for(rows, *, model, tokenizer, model_key, layer_indices,
                       prompt_fn, answer_fn, k_answer_tokens, desc) -> dict:
    means: list[torch.Tensor] = []
    meta:  list[dict] = []
    skipped = 0
    progress = Progress(total=len(rows), desc=desc, log_every=25)

    for row in rows:
        prompt = prompt_fn(row) or ""
        answer = answer_fn(row) or ""
        if not prompt.strip() or not answer.strip():
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        try:
            full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
                tokenizer, prompt, answer, k_answer_tokens=k_answer_tokens,
            )
        except Exception as exc:
            log.debug("  %s tokenise error: %s", desc, exc)
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        if n_ans == 0:
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue

        ids = torch.tensor([full_ids], dtype=torch.long)
        try:
            with torch.no_grad():
                _, hiddens = forward_hidden_states(model, ids, layer_indices)
        except Exception as exc:
            log.warning("  %s forward error: %s", desc, exc)
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue

        if p_len < 1:
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        m = mean_answer_activation(hiddens, prompt_len=p_len - 1,
                                   n_answer_tokens=n_ans)
        means.append(m)
        meta.append({
            "dataset":     row.get("dataset", "?"),
            "id":          row.get("example_id") or row.get("id"),
            "judge_label": row.get("judge_label"),
        })
        progress.tick(extras={"kept": len(means), "skip": skipped})

    progress.done(extras={"kept": len(means), "skipped": skipped})

    if not means:
        return {"means": torch.zeros(0, len(layer_indices), 1), "meta": meta}
    return {"means": torch.stack(means, dim=0).float(), "meta": meta}


def _extract_retain_general_means(rows, *, model, tokenizer, model_key,
                                  layer_indices, k_answer_tokens, desc) -> dict:
    means: list[torch.Tensor] = []
    skipped = 0
    progress = Progress(total=len(rows), desc=desc, log_every=25)

    for row in rows:
        prompt   = row.get("prompt") or ""
        response = row.get("response") or ""
        if not prompt.strip() or not response.strip():
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        try:
            full_ids, resp_start = tokenise_chat_prompt_response(
                tokenizer, model_key, prompt, response,
            )
        except Exception as exc:
            log.debug("  %s tokenise error: %s", desc, exc)
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue

        id_tensor = torch.tensor([full_ids], dtype=torch.long)
        try:
            with torch.no_grad():
                _, hiddens = forward_hidden_states(model, id_tensor,
                                                   layer_indices)
        except Exception as exc:
            log.warning("  %s forward error: %s", desc, exc)
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue

        if resp_start < 1:
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        m = mean_answer_activation(hiddens, prompt_len=resp_start - 1,
                                   n_answer_tokens=k_answer_tokens)
        means.append(m)
        progress.tick(extras={"kept": len(means), "skip": skipped})

    progress.done(extras={"kept": len(means), "skipped": skipped})

    if not means:
        return {"means": torch.zeros(0, len(layer_indices), 1)}
    return {"means": torch.stack(means, dim=0).float()}


# ── main ──────────────────────────────────────────────────────────────────────

def run(run_dir: Path, max_per_set: int | None, rebuild: bool = False) -> Path:
    pipeline_t0 = time.time()

    # Infer model_key from training_config.json saved by step 4
    training_cfg_path = run_dir / "training_config.json"
    if not training_cfg_path.exists():
        raise FileNotFoundError(
            f"training_config.json not found in {run_dir}. "
            "Is this a valid step-4 run directory?"
        )
    with open(training_cfg_path) as f:
        training_cfg = json.load(f)
    model_key = training_cfg["model_key"]
    run_name   = run_dir.name

    log.info("=" * 60)
    log.info("Step 6 — extract activations from trained model")
    log.info("  run_dir   : %s", run_dir)
    log.info("  model_key : %s", model_key)
    log.info("  run_name  : %s", run_name)

    out_path = OUT_DIR / f"activations_{run_name}.pt"
    if not rebuild and out_path.exists():
        log.info("  output already exists (%s) — skipping. Use --rebuild to re-run.", out_path)
        return out_path

    layer_indices = layer_indices_for(model_key)
    K = cfg.K_ANSWER_TOKENS

    # Load data pools (same as step 1)
    with Stopwatch("load data pools"):
        forget_pool   = _load_forget_pool(model_key)
        answer_pool   = _load_answerable_pool()
        retain_pool   = _load_retain_general()

    forget_committed = [r for r in forget_pool if r.get("judge_label") == "COMMIT"]
    log.info("  forget pool       : %d total, %d committed",
             len(forget_pool), len(forget_committed))
    log.info("  answer pool       : %d", len(answer_pool))
    log.info("  retain general    : %d", len(retain_pool))

    if max_per_set:
        forget_committed = forget_committed[:max_per_set]
        answer_pool      = answer_pool[:max_per_set]
        retain_pool      = retain_pool[:max_per_set]
        log.info("  (capped to %d per set)", max_per_set)

    # Lazy model loader — loads once and caches
    _model_cache: list = []

    def model_and_tok():
        if not _model_cache:
            with Stopwatch("load base model"):
                m, t = load_model_and_tokenizer(model_key, eval_only=True)
            # Find the adapter checkpoint (prefer _best, fall back to _final)
            adapter_dir = None
            for subdir in ("_best", "_final"):
                candidate = run_dir / subdir
                if (candidate / "adapter_config.json").exists():
                    adapter_dir = candidate
                    break
            if adapter_dir is None:
                # adapter files are directly in run_dir
                if (run_dir / "adapter_config.json").exists():
                    adapter_dir = run_dir
            if adapter_dir is None:
                raise FileNotFoundError(
                    f"No adapter_config.json found under {run_dir}. "
                    "Checked: _best/, _final/, and run_dir itself."
                )
            log.info("  loading LoRA adapter from: %s", adapter_dir)
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, str(adapter_dir), is_trainable=False)
            m.eval()
            log.info("  LoRA adapter loaded and merged into eval mode")
            _model_cache.extend([m, t])
        return _model_cache[0], _model_cache[1]

    # Extract sets A–E
    def extract_A():
        m, t = model_and_tok()
        return _extract_means_for(
            forget_committed,
            model=m, tokenizer=t, model_key=model_key,
            layer_indices=layer_indices, k_answer_tokens=K,
            prompt_fn=lambda r: build_unanswerable_prompt(r["dataset"], r),
            answer_fn=lambda r: r.get("y_com_prefix_k8") or r.get("full_completion_clean") or "",
            desc="A. over_commit",
        )

    def extract_B():
        m, t = model_and_tok()
        return _extract_means_for(
            forget_committed,
            model=m, tokenizer=t, model_key=model_key,
            layer_indices=layer_indices, k_answer_tokens=K,
            prompt_fn=lambda r: build_unanswerable_prompt(r["dataset"], r),
            answer_fn=lambda r: cfg.abstain_template_for(r.get("dataset")),
            desc="B. legit_abstain (per-domain templated)",
        )

    def extract_C():
        m, t = model_and_tok()
        return _extract_means_for(
            answer_pool,
            model=m, tokenizer=t, model_key=model_key,
            layer_indices=layer_indices, k_answer_tokens=K,
            prompt_fn=lambda r: build_answerable_prompt(r["dataset"], r),
            answer_fn=lambda r: r.get("correct_answer") or "",
            desc="C. legit_commit",
        )

    def extract_D():
        m, t = model_and_tok()
        return _extract_means_for(
            answer_pool,
            model=m, tokenizer=t, model_key=model_key,
            layer_indices=layer_indices, k_answer_tokens=K,
            prompt_fn=lambda r: build_answerable_prompt(r["dataset"], r),
            answer_fn=lambda r: cfg.abstain_template_for(r.get("dataset")),
            desc="D. over_abstain (per-domain templated)",
        )

    def extract_E():
        m, t = model_and_tok()
        return _extract_retain_general_means(
            retain_pool,
            model=m, tokenizer=t, model_key=model_key,
            layer_indices=layer_indices, k_answer_tokens=K,
            desc="E. general_utility",
        )

    out_A = _load_or_extract_set("A_over_commit",     extract_A, run_name=run_name, rebuild=rebuild)
    out_B = _load_or_extract_set("B_legit_abstain",   extract_B, run_name=run_name, rebuild=rebuild)
    out_C = _load_or_extract_set("C_legit_commit",    extract_C, run_name=run_name, rebuild=rebuild)
    out_D = _load_or_extract_set("D_over_abstain",    extract_D, run_name=run_name, rebuild=rebuild)
    out_E = _load_or_extract_set("E_general_utility", extract_E, run_name=run_name, rebuild=rebuild)

    # Bundle (identical schema to step-1 output)
    bundle = {
        "model_key":       model_key,
        "run_name":        run_name,
        "run_dir":         str(run_dir),
        "layers":          layer_indices,
        "k_answer_tokens": K,
        "h_A":   out_A["means"],
        "h_B":   out_B["means"],
        "h_C":   out_C["means"],
        "h_D":   out_D["means"],
        "h_E":   out_E["means"],
        "meta_A": out_A.get("meta", []),
        "meta_B": out_B.get("meta", []),
        "meta_C": out_C.get("meta", []),
        "meta_D": out_D.get("meta", []),
    }

    torch.save(bundle, out_path)

    def _shape(t):
        return tuple(t.shape) if isinstance(t, torch.Tensor) else (0,)

    log.info("  saved -> %s", out_path)
    log.info("    h_A (over-commit)     : %s", _shape(bundle["h_A"]))
    log.info("    h_B (legit-abstain)   : %s", _shape(bundle["h_B"]))
    log.info("    h_C (legit-commit)    : %s", _shape(bundle["h_C"]))
    log.info("    h_D (over-abstain)    : %s", _shape(bundle["h_D"]))
    log.info("    h_E (general utility) : %s", _shape(bundle["h_E"]))
    log.info("  total time: %s", format_duration(time.time() - pipeline_t0))

    # Clean up partial checkpoints now that merged bundle is durable
    shutil.rmtree(_partial_dir(run_name), ignore_errors=True)

    # Free GPU memory
    if _model_cache:
        free_model(_model_cache[0])

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 6: extract UOC activations from a trained model.")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Path to a step-4 run directory containing adapter_config.json.")
    p.add_argument("--max-per-set", type=int, default=None,
                   help="Cap each set at N examples (useful for smoke-tests).")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore existing partial/full checkpoints and re-extract.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.exists():
        log.error("run-dir not found: %s", run_dir)
        sys.exit(1)
    out = run(run_dir, max_per_set=args.max_per_set, rebuild=args.rebuild)
    print(f"✓  activations saved to {out}")
