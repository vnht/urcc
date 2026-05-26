#!/usr/bin/env python3
"""Step 3 — Build the two UOC poles along V, per answerability domain.

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
the shared discriminative subspace V (built once in step 2 from KUQ + SQuAD
contrasts combined) so each training example is pulled toward a target that
lives in *its own* prompt distribution.

Note: the poles themselves are points in 4096-D activation space and do not
depend on V — V is just the projection used by the loss in step 4. So this
step is V-agnostic.

Both poles are fixed, layer-aligned constants — no gradient flows through
them. They are the per-pole targets the UOC loss anchors each example to:

    D_F[d]   (forget, category A, domain d)        → μ_l⁻(d)
    D_R_A[d] (retain-answerable, category C, dom d) → μ_l⁺(d)
    D_R_G    (retain-general, category E)          → frozen reference (computed at
                                                     training time, not here)

Reads:  step1_extract_activations/data/activations_<model>.pt
        step2_build_subspace/data/subspace_<model>_r<rank>.pt   (for V)
Writes: step3_build_anchors/data/anchors_<model>.pt with keys:
    "model_key", "layers", "k_answer_tokens", "datasets",
    "mu_minus":        tensor[L, D]               grand-mean abstain pole (legacy)
    "mu_plus":         tensor[L, D]               grand-mean commit  pole (legacy)
    "mu_minus_per":    dict[str, tensor[L, D]]    {kuq: …, squad: …} abstain poles
    "mu_plus_per":     dict[str, tensor[L, D]]    {kuq: …, squad: …} commit  poles
    "n_minus_per":     dict[str, int]             rows used per domain for μ⁻
    "n_plus_per":      dict[str, int]             rows used per domain for μ⁺
    "n_minus":         int                        total rows used for grand μ⁻
    "n_plus":          int                        total rows used for grand μ⁺
    "init_scale_per":  dict[str, float]           per-domain initial L_forget magnitude:
                                                  E_{x ∈ D_F[d]} ⟨ ‖Vᵀ(h_A(x) − μ⁻(d))‖² ⟩_l
                                                  Used by step 4 to divide each forget
                                                  example by its own domain's scale, so
                                                  KUQ and SQuAD start at L_forget ≈ 1.0
                                                  per example regardless of intrinsic
                                                  contrast magnitude.
    "init_scale":      float                      mean over domains of init_scale_per
                                                  (shared scalar, kept for compatibility)

Run
---
    python step3_build_anchors/build_anchors.py --model qwen_instruct --rank 32
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


def _per_domain_init_scale(
    h_A: torch.Tensor,                         # (N_F, L, D)
    ds_A: list[str],                           # per-row dataset labels
    mu_minus_per: dict[str, torch.Tensor],     # {d: (L, D)}
    V: torch.Tensor,                           # (L, D, r)
) -> dict[str, float]:
    """Per-domain initial L_forget magnitude:

        s(d) = E_{x ∈ D_F[d]} ⟨ ‖V_lᵀ (h_A(x) − μ⁻_l(d)) ‖² ⟩_l

    Computed once at step 3, used by step 4 to divide each forget example
    by its own domain's scale so KUQ and SQuAD start at L_forget ≈ 1.0
    per example regardless of intrinsic contrast magnitude.
    """
    V = V.float()
    out: dict[str, float] = {}
    for d in DOMAINS:
        idx = [i for i, ds in enumerate(ds_A) if ds == d]
        if not idx:
            raise RuntimeError(f"No D_F rows for dataset '{d}'.")
        h_d = h_A[idx].float()                                 # (n_d, L, D)
        diff = h_d - mu_minus_per[d].unsqueeze(0)              # (n_d, L, D)
        proj = torch.einsum("nld,ldr->nlr", diff, V)           # (n_d, L, r)
        sq   = (proj ** 2).sum(dim=-1)                         # (n_d, L)
        out[d] = float(sq.mean().item())                       # mean over rows AND layers
    return out


def run(model_key: str, rank: int, overwrite: bool = False) -> Path:
    pipeline_t0 = time.time()
    act_path = cfg.activations_path(model_key)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Activations bundle not found: {act_path}. Run step 1 first."
        )

    sub_path = cfg.subspace_path(model_key, rank=rank)
    if not sub_path.exists():
        raise FileNotFoundError(
            f"Subspace bundle not found: {sub_path}. Run step 2 first."
        )

    out_path = cfg.anchors_path(model_key)
    if out_path.exists() and not overwrite:
        log.info("STEP 3 — BUILD ANCHORS  (cached) %s", out_path)
        log.info("  use --overwrite to recompute. Skipping.")
        return out_path

    bundle = torch.load(act_path, map_location="cpu", weights_only=False)
    sub    = torch.load(sub_path, map_location="cpu", weights_only=False)

    if sub["layers"] != bundle["layers"]:
        raise RuntimeError(
            f"Subspace layers {sub['layers']} != activations layers {bundle['layers']}"
        )

    h_minus = bundle["h_B"]       # (N_F, L, D) — legitimate-abstention pool μ⁻
    h_plus  = bundle["h_C"]       # (N_A, L, D) — legitimate-commitment pool μ⁺
    h_A     = bundle["h_A"]       # (N_F, L, D) — over-commit pool (for s(d))

    if h_minus.shape[0] == 0:
        raise RuntimeError("No abstain examples available — check step 1 output.")
    if h_plus.shape[0] == 0:
        raise RuntimeError("No answerable correct examples — check step 1 output.")
    if h_A.shape[0] == 0:
        raise RuntimeError("No D_F examples available — check step 1 output.")

    ds_minus = _datasets_for(bundle, "h_B")
    ds_plus  = _datasets_for(bundle, "h_C")
    ds_A     = _datasets_for(bundle, "h_A")

    mu_minus      = h_minus.mean(dim=0).float()                  # (L, D)  grand mean
    mu_plus       = h_plus.mean(dim=0).float()                   # (L, D)  grand mean
    mu_minus_per  = _per_domain_mean(h_minus, ds_minus)          # {d: (L, D)}
    mu_plus_per   = _per_domain_mean(h_plus,  ds_plus)           # {d: (L, D)}
    n_minus_per   = {d: int(sum(1 for x in ds_minus if x == d)) for d in DOMAINS}
    n_plus_per    = {d: int(sum(1 for x in ds_plus  if x == d)) for d in DOMAINS}

    init_scale_per = _per_domain_init_scale(h_A, ds_A, mu_minus_per, sub["V"])
    init_scale     = sum(init_scale_per.values()) / len(init_scale_per)

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
        "init_scale_per":  init_scale_per,
        "init_scale":      init_scale,
        "subspace_rank":   rank,
    }
    torch.save(out, out_path)

    log.info("STEP 3 — BUILD ANCHORS  model=%s  rank=%d", model_key, rank)
    log.info("  μ⁻ from %d abstain examples (per-domain: %s)",
             out["n_minus"], n_minus_per)
    log.info("  μ⁺ from %d answerable examples (per-domain: %s)",
             out["n_plus"], n_plus_per)
    log.info("  init_scale_per: %s   shared init_scale: %.2f",
             {d: round(s, 2) for d, s in init_scale_per.items()}, init_scale)
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
    p.add_argument("--rank", type=int, default=cfg.SUBSPACE_RANK,
                   help="Subspace rank (must match step 2 output)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, rank=args.rank, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
