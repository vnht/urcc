#!/usr/bin/env python3
"""Step 3 — Build the two UOC poles along V, per answerability domain.

For each answerability domain d ∈ {kuq, squad} we compute two layer-aligned
poles in late-layer hidden-state space:

μ_l⁻(d)   (legitimate-abstention pole, domain d)
    Mean late-layer hidden state over the answer-token window of templated
    legitimate-abstention completions on domain-d unanswerable prompts.
    Drawn from set B in step 1, restricted to rows whose dataset == d.

μ_l⁺(d)   (legitimate-commitment pole, domain d)
    Mean late-layer hidden state over the answer-token window of gold answers
    on domain-d answerable prompts. Drawn from set C, restricted to rows whose
    dataset == d.

Why per-domain
--------------
KUQ prompts (no context) and SQuAD prompts (long context) sit in very
different regions of late-layer hidden-state space. A single mean over both
yields a blended pole that is well-aligned with neither domain's natural
abstention or commitment region. Per-domain poles localise the target inside
the shared discriminative subspace V (which stays domain-shared) so each
training example is pulled toward a target that lives in *its own* prompt
distribution.

Both poles are fixed, layer-aligned constants — no gradient flows through
them. They are the per-pole targets the UOC loss anchors each example to:

    D_F[d]   (forget, category A, domain d)        → μ_l⁻(d)
    D_R_A[d] (retain-answerable, category C, dom d) → μ_l⁺(d)
    D_R_G    (retain-general, category E)          → frozen reference (computed at
                                                     training time, not here)

Reads:  step1_extract_activations/data/activations_<model>.pt
Writes: step3_build_anchors/data/anchors_<model>.pt with keys:
    "model_key", "layers", "k_answer_tokens", "datasets",
    "mu_minus":      tensor[L, D]               grand-mean abstain pole (legacy)
    "mu_plus":       tensor[L, D]               grand-mean commit  pole (legacy)
    "mu_minus_per":  dict[str, tensor[L, D]]    {kuq: …, squad: …} abstain poles
    "mu_plus_per":   dict[str, tensor[L, D]]    {kuq: …, squad: …} commit  poles
    "n_minus_per":   dict[str, int]             rows used per domain for μ⁻
    "n_plus_per":    dict[str, int]             rows used per domain for μ⁺
    "n_minus":       int                        total rows used for grand μ⁻
    "n_plus":        int                        total rows used for grand μ⁺

Run
---
    python step3_build_anchors/build_anchors.py --model qwen_instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import format_duration, log


DOMAINS = ("kuq", "squad")


def _datasets_for(bundle: dict, hidden_key: str) -> list[str]:
    """Return a per-row dataset label list aligned to `bundle[hidden_key]`.

    Looks for `meta_<X>` first; falls back to a sibling meta when the
    set was extracted from the same row pool (B↔A, D↔C). Raises if no
    alignment is possible.
    """
    n = bundle[hidden_key].shape[0]
    set_letter = hidden_key.split("_")[-1]   # "h_B" -> "B"
    meta_key   = f"meta_{set_letter}"
    fallback   = {"B": "meta_A", "D": "meta_C"}.get(set_letter)

    meta = bundle.get(meta_key) or (bundle.get(fallback) if fallback else None)
    if not meta or len(meta) != n:
        raise RuntimeError(
            f"Cannot align dataset labels for {hidden_key}: "
            f"len(meta)={len(meta) if meta else 0} != n_rows={n}. "
            f"Re-run step 1 with the latest extract.py."
        )
    return [str(m.get("dataset") or "?") for m in meta]


def _per_domain_mean(h: torch.Tensor, datasets: list[str]) -> dict[str, torch.Tensor]:
    """Group rows of h by dataset label and return {dataset: mean[L, D]}."""
    out: dict[str, torch.Tensor] = {}
    for d in DOMAINS:
        idx = [i for i, ds in enumerate(datasets) if ds == d]
        if not idx:
            raise RuntimeError(f"No rows found for dataset '{d}'. Check step 1 output.")
        out[d] = h[idx].mean(dim=0).float()       # (L, D)
    return out


def run(model_key: str, overwrite: bool = False) -> Path:
    pipeline_t0 = time.time()
    act_path = cfg.activations_path(model_key)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Activations bundle not found: {act_path}. Run step 1 first."
        )

    out_path = cfg.anchors_path(model_key)
    if out_path.exists() and not overwrite:
        log.info("STEP 3 — BUILD ANCHORS  (cached) %s", out_path)
        log.info("  use --overwrite to recompute. Skipping.")
        return out_path

    bundle = torch.load(act_path, map_location="cpu", weights_only=False)

    h_minus = bundle["h_B"]       # (N_F, L, D) — legitimate-abstention pool μ⁻
    h_plus  = bundle["h_C"]       # (N_A, L, D) — legitimate-commitment pool μ⁺

    if h_minus.shape[0] == 0:
        raise RuntimeError("No abstain examples available — check step 1 output.")
    if h_plus.shape[0] == 0:
        raise RuntimeError("No answerable correct examples — check step 1 output.")

    ds_minus = _datasets_for(bundle, "h_B")
    ds_plus  = _datasets_for(bundle, "h_C")

    mu_minus      = h_minus.mean(dim=0).float()                  # (L, D)  grand mean
    mu_plus       = h_plus.mean(dim=0).float()                   # (L, D)  grand mean
    mu_minus_per  = _per_domain_mean(h_minus, ds_minus)          # {d: (L, D)}
    mu_plus_per   = _per_domain_mean(h_plus,  ds_plus)           # {d: (L, D)}
    n_minus_per   = {d: int(sum(1 for x in ds_minus if x == d)) for d in DOMAINS}
    n_plus_per    = {d: int(sum(1 for x in ds_plus  if x == d)) for d in DOMAINS}

    out = {
        "model_key":       model_key,
        "model_id":        bundle["model_id"],
        "layers":          bundle["layers"],
        "k_answer_tokens": bundle["k_answer_tokens"],
        "datasets":        list(DOMAINS),
        "mu_minus":        mu_minus,
        "mu_plus":         mu_plus,
        "mu_minus_per":    mu_minus_per,
        "mu_plus_per":     mu_plus_per,
        "n_minus":         int(h_minus.shape[0]),
        "n_plus":          int(h_plus.shape[0]),
        "n_minus_per":     n_minus_per,
        "n_plus_per":      n_plus_per,
    }
    torch.save(out, out_path)

    log.info("STEP 3 — BUILD ANCHORS  model=%s", model_key)
    log.info("  μ⁻ from %d abstain examples (per-domain: %s)",
             out["n_minus"], n_minus_per)
    log.info("  μ⁺ from %d answerable examples (per-domain: %s)",
             out["n_plus"], n_plus_per)
    log.info("  Layer-wise norms:")
    for li, l in enumerate(out["layers"]):
        m_minus_kuq   = mu_minus_per["kuq"][li]
        m_minus_squad = mu_minus_per["squad"][li]
        m_plus_kuq    = mu_plus_per["kuq"][li]
        m_plus_squad  = mu_plus_per["squad"][li]
        log.info(
            "    L%-3d  ||μ⁻_kuq||=%.2f  ||μ⁻_squad||=%.2f  "
            "||μ⁻_kuq−μ⁻_squad||=%.2f   "
            "||μ⁺_kuq||=%.2f  ||μ⁺_squad||=%.2f  "
            "||μ⁺_kuq−μ⁺_squad||=%.2f",
            l,
            float(m_minus_kuq.norm()), float(m_minus_squad.norm()),
            float((m_minus_kuq - m_minus_squad).norm()),
            float(m_plus_kuq.norm()),  float(m_plus_squad.norm()),
            float((m_plus_kuq  - m_plus_squad).norm()),
        )
    log.info("  Saved -> %s", out_path)
    log.info("STEP 3 done in %s", format_duration(time.time() - pipeline_t0))
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: build per-domain anchors μ⁻ and μ⁺.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
