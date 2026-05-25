#!/usr/bin/env python3
"""Step 1 — Extract late-layer hidden states for the five UOC behaviour sets.

For one model, runs forward passes on five (prompt, completion) pairs per
example and saves the mean late-layer hidden states per example. Everything
downstream (subspace V, anchors μ⁻ μ⁺, and the training loss) is built from
this single bundle.

Sets extracted (vocabulary in README.md)
----------------------------------------
A. Over-commitment        — D_F prompts + the model's own over-commit prefix
B. Legitimate abstention  — D_F prompts + templated legitimate-abstention text
C. Legitimate commitment  — D_R_A prompts + gold answer
D. Over-abstention        — D_R_A prompts + templated abstention text
E. General utility        — D_R_G (prompt, response) pairs

The contrast (A − B) isolates the over-commit direction; the contrast (C − D)
isolates the legitimate-commit direction. Both subtract a shared "abstain
mode" baseline so the eigenproblem in step 2 sees only the commit-mode signal.

Answer-token window (K positions, indexing convention)
-----------------------------------------------------
For each row we take the mean late-layer hidden state over a window of K
positions starting one token *before* the first answer token:

    window = [p_len - 1, p_len, p_len + 1, …, p_len + K - 2]

Position ``p_len - 1`` is the prompt-final residual stream — the state from
which the LM head decides the *first* generated token. Including it inside
the window is what lets the retain loss intrinsically discourage degenerate
solutions where the first-token logit collapses to a chat-end token; the
remaining K-1 positions cover the body of the answer (or the start of the
abstention text, for sets B and D).

For set E (UltraChat) the same shift applies: the window starts at
``resp_start - 1`` rather than ``resp_start``.

Output
------
step1_extract_activations/data/activations_<model>.pt  with keys:
    "model_key", "model_id", "layers", "k_answer_tokens",
    "h_A":      (N_F, L, D)   over-commitment            (D_F prompts, model's over-commit prefix)
    "h_B":      (N_F, L, D)   legitimate-abstention      (D_F prompts, templated abstention)
    "h_C":      (N_A, L, D)   legitimate-commitment      (D_R_A prompts, gold answer)
    "h_D":      (N_A, L, D)   over-abstention            (D_R_A prompts, templated abstention)
    "h_E":      (N_R, L, D)   general utility            (D_R_G prompt+response)
    "meta_A":   list[{"dataset", "judge_label", "id"}]   (one entry per row of h_A)
    "meta_B":   list[{"dataset", "id"}]                  (one entry per row of h_B)
    "meta_C":   list[{"dataset", "id"}]                  (one entry per row of h_C)
    "meta_D":   list[{"dataset", "id"}]                  (one entry per row of h_D)

The per-set ``meta`` lists track each row's source dataset (kuq / squad) so
that downstream steps (notably step 3 anchors) can compute per-domain poles
without re-loading the source jsonl files.

Crash-safe & resumable
----------------------
After each set finishes, its tensor + meta are written to a per-set checkpoint
under `step1_extract_activations/data/_partial_<model>/<setname>.pt`. On
restart, any set whose checkpoint already exists is skipped. After all sets
are done the merged bundle is written and the partial directory is removed.
Use `--rebuild` to ignore cached partial sets.

Run
---
    python step1_extract_activations/extract.py --model qwen_instruct
    python step1_extract_activations/extract.py --model qwen_instruct --max-per-set 200   # smoke
    python step1_extract_activations/extract.py --model qwen_instruct --rebuild           # ignore cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _judge_label(row: dict) -> str:
    """Best-effort normalisation of legacy judge labels."""
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


def _partial_dir(model_key: str) -> Path:
    p = cfg.ACTIVATIONS_DIR / f"_partial_{model_key}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _partial_path(model_key: str, set_name: str) -> Path:
    return _partial_dir(model_key) / f"{set_name}.pt"


def _load_or_extract_set(
    set_name: str,
    extractor,
    *,
    model_key: str,
    rebuild: bool,
) -> dict:
    """If a partial checkpoint exists for this set, load it; otherwise run the
    extractor and save the result."""
    pp = _partial_path(model_key, set_name)
    if not rebuild and pp.exists():
        log.info("  set %-22s   (cached) loading from %s", set_name, pp)
        return torch.load(pp, map_location="cpu", weights_only=False)
    with Stopwatch(f"set {set_name}"):
        result = extractor()
    torch.save(result, pp)
    log.info("  set %-22s   checkpoint -> %s", set_name, pp)
    return result


# ── Per-set forward passes ────────────────────────────────────────────────────

def _extract_means_for(
    rows: list[dict],
    *,
    model,
    tokenizer,
    model_key: str,
    layer_indices: list[int],
    prompt_fn,
    answer_fn,
    k_answer_tokens: int,
    desc: str,
) -> dict:
    """Forward each (prompt, answer) pair; return mean activations + metadata.

    Returns ``{"means": tensor[N, L, D], "meta": list[dict]}``.
    """
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

        # Window starts at p_len - 1 so it includes the prompt-final hidden
        # state (the residual stream that decides the first generated token).
        # The window has K positions: [p_len - 1, …, p_len + K - 2].
        if p_len < 1:
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        m = mean_answer_activation(hiddens, prompt_len=p_len - 1, n_answer_tokens=n_ans)
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


def _extract_retain_general_means(
    rows: list[dict],
    *,
    model,
    tokenizer,
    model_key: str,
    layer_indices: list[int],
    k_answer_tokens: int,
    desc: str,
) -> dict:
    """Mean late-layer activation over the first k response tokens (UltraChat)."""
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
        n_ans = min(k_answer_tokens, max(0, len(full_ids) - resp_start))
        if n_ans == 0:
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue

        # Causal attention ⇒ tokens past resp_start + n_ans don't influence
        # the activations we keep. Truncate to skip wasted compute.
        full_ids = full_ids[: resp_start + n_ans]
        ids = torch.tensor([full_ids], dtype=torch.long)
        try:
            with torch.no_grad():
                _, hiddens = forward_hidden_states(model, ids, layer_indices)
        except Exception as exc:
            log.warning("  %s forward error: %s", desc, exc)
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue

        # Window starts at resp_start - 1 to include the prompt-to-response
        # transition state (mirrors sets A–D).
        if resp_start < 1:
            skipped += 1
            progress.tick(extras={"kept": len(means), "skip": skipped})
            continue
        m = mean_answer_activation(hiddens, prompt_len=resp_start - 1, n_answer_tokens=n_ans)
        means.append(m)
        progress.tick(extras={"kept": len(means), "skip": skipped})

    progress.done(extras={"kept": len(means), "skipped": skipped})

    if not means:
        return {"means": torch.zeros(0, len(layer_indices), 1)}
    return {"means": torch.stack(means, dim=0).float()}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_key: str, max_per_set: int | None, rebuild: bool = False) -> Path:
    pipeline_t0 = time.time()
    layer_indices = layer_indices_for(model_key)
    K = cfg.K_ANSWER_TOKENS

    log.info("=" * 64)
    log.info("STEP 1 — EXTRACT ACTIVATIONS  model=%s  K=%d  layers=%s",
             model_key, K, layer_indices)
    log.info("  partial dir: %s   (rebuild=%s)", _partial_dir(model_key), rebuild)

    forget_pool = _load_forget_pool(model_key)
    answer_pool = _load_answerable_pool()
    retain_pool = _load_retain_general()
    forget_committed = [r for r in forget_pool if r["judge_label"] == "COMMIT"]
    forget_abstained = [r for r in forget_pool if r["judge_label"] == "ABSTAIN"]
    log.info("  D_F (forget pool): %d total (%d COMMIT, %d ABSTAIN)",
             len(forget_pool), len(forget_committed), len(forget_abstained))
    log.info("  D_R_A (retain-answerable): %d   D_R_G (retain-general): %d",
             len(answer_pool), len(retain_pool))

    if max_per_set is not None:
        forget_committed = forget_committed[:max_per_set]
        answer_pool      = answer_pool[:max_per_set]
        retain_pool      = retain_pool[:max_per_set]
        log.info("  capped each set to max-per-set=%d", max_per_set)

    # Defer loading the model until we actually need it — avoids re-loading
    # when every set is already cached.
    _model = {"obj": None, "tok": None}

    def model_and_tok():
        if _model["obj"] is None:
            _model["obj"], _model["tok"] = load_model_and_tokenizer(
                model_key, eval_only=True,
            )
        return _model["obj"], _model["tok"]

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
            answer_fn=lambda _: cfg.ABSTAIN_TEMPLATE,
            desc="B. legit_abstain (templated)",
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
            answer_fn=lambda _: cfg.ABSTAIN_TEMPLATE,
            desc="D. over_abstain (templated)",
        )

    def extract_E():
        m, t = model_and_tok()
        return _extract_retain_general_means(
            retain_pool,
            model=m, tokenizer=t, model_key=model_key,
            layer_indices=layer_indices, k_answer_tokens=K,
            desc="E. general_utility",
        )

    out_A = _load_or_extract_set("A_over_commit",       extract_A, model_key=model_key, rebuild=rebuild)
    out_B = _load_or_extract_set("B_legit_abstain",     extract_B, model_key=model_key, rebuild=rebuild)
    out_C = _load_or_extract_set("C_legit_commit",      extract_C, model_key=model_key, rebuild=rebuild)
    out_D = _load_or_extract_set("D_over_abstain",      extract_D, model_key=model_key, rebuild=rebuild)
    out_E = _load_or_extract_set("E_general_utility",   extract_E, model_key=model_key, rebuild=rebuild)

    # Final bundle
    bundle = {
        "model_key":        model_key,
        "model_id":         cfg.MODEL_REGISTRY[model_key],
        "layers":           layer_indices,
        "k_answer_tokens":  K,
        "h_A":      out_A["means"],
        "h_B":      out_B["means"],
        "h_C":      out_C["means"],
        "h_D":      out_D["means"],
        "h_E":      out_E["means"],
        "meta_A":   out_A.get("meta", []),
        "meta_B":   out_B.get("meta", []),
        "meta_C":   out_C.get("meta", []),
        "meta_D":   out_D.get("meta", []),
    }

    out_path = cfg.activations_path(model_key)
    torch.save(bundle, out_path)

    def _shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(t.shape) if isinstance(t, torch.Tensor) else (0,)

    log.info("  saved -> %s", out_path)
    log.info("    h_A (over-commit)        : %s", _shape(bundle["h_A"]))
    log.info("    h_B (legit-abstain)      : %s", _shape(bundle["h_B"]))
    log.info("    h_C (legit-commit)       : %s", _shape(bundle["h_C"]))
    log.info("    h_D (over-abstain)       : %s", _shape(bundle["h_D"]))
    log.info("    h_E (general utility)    : %s", _shape(bundle["h_E"]))

    # Drop the partial dir now that the merged bundle is durable.
    shutil.rmtree(_partial_dir(model_key), ignore_errors=True)

    if _model["obj"] is not None:
        free_model(_model["obj"])

    log.info("STEP 1 done in %s", format_duration(time.time() - pipeline_t0))
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: extract UOC activations (sets A/B/C/D/E).")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--max-per-set", type=int, default=None,
                   help="Cap each set to N rows (smoke testing)")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore any cached per-set checkpoints and re-run all sets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, max_per_set=args.max_per_set, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
