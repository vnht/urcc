#!/usr/bin/env python3
"""
Part 1 — Compute cleaned unsupported contrasts.

For each model and selected layer l, projects out the supported-answering
subspace (rank-r PCA of c_supported_l) from c_unsupported_l:

    c_unsupported_clean_l = c_unsupported_l - V_r V_r^T c_unsupported_l

Then verifies:
    rho_clean  ≈ 0            (supported directions are gone)
    energy_retained ≈ 1 - rho_raw   (expected energy after projection)

Output per model:
    mining-data/activations/cleaned_unsupported_contrasts_{model}_r8.pt

Contents:
    {
        "model":                str,
        "layers":               list[int],   # selected layers only
        "supported_rank":       int,
        "k":                    int,
        "datasets":             list[str],
        "prompt_ids":           list[str],
        "c_unsupported_clean":  tensor[N, L, D],
        "V_supported":          tensor[L, D, r],  # projection bases used
    }

Usage:
    python3 mining-data/compute_cleaned_contrasts.py
    python3 mining-data/compute_cleaned_contrasts.py --model qwen_instruct
    python3 mining-data/compute_cleaned_contrasts.py --rank 8
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent
ACTIVATIONS_DIR = BASE_DIR / "activations"

MODEL_KEYS = ["qwen_instruct", "qwen_base", "ministral_instruct", "ministral_base"]
DEFAULT_RANK = 8


# ─── PCA helpers ──────────────────────────────────────────────────────────────

def fit_pca_basis(X: torch.Tensor, r: int) -> torch.Tensor:
    """
    Fit rank-r PCA on X (N, D). Returns V_r (D, r) — top-r right singular
    vectors. Centres X before decomposition.
    """
    X_f = X.float()
    X_c = X_f - X_f.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X_c, full_matrices=False)
    return Vh[:r].T   # (D, r)


def project_out(X: torch.Tensor, V_r: torch.Tensor) -> torch.Tensor:
    """
    Remove the component of X (N, D) that lies in the column space of V_r (D, r).
    Returns X - V_r V_r^T X  (N, D).
    """
    X_f  = X.float()
    proj = (X_f @ V_r) @ V_r.T   # (N, D)
    return X_f - proj


def projection_ratio(X: torch.Tensor, V_r: torch.Tensor) -> float:
    """Mean fraction of squared norm of X lying in col(V_r)."""
    X_f   = X.float()
    proj  = (X_f @ V_r) @ V_r.T
    num   = proj.norm(dim=-1).pow(2)
    denom = X_f.norm(dim=-1).pow(2).clamp(min=1e-12)
    return float((num / denom).mean())


# ─── Per-model cleaning ───────────────────────────────────────────────────────

def clean_model(model_key: str, rank: int) -> None:
    sup_path   = ACTIVATIONS_DIR / f"supported_answer_contrasts_{model_key}.pt"
    unsup_path = ACTIVATIONS_DIR / f"unsupported_commitment_contrasts_{model_key}.pt"
    out_path   = ACTIVATIONS_DIR / f"cleaned_unsupported_contrasts_{model_key}_last25_r{rank}.pt"

    if not sup_path.exists():
        log.error("Missing: %s", sup_path); return
    if not unsup_path.exists():
        log.error("Missing: %s", unsup_path); return

    sup   = torch.load(sup_path,   map_location="cpu", weights_only=False)
    unsup = torch.load(unsup_path, map_location="cpu", weights_only=False)

    c_sup   = sup["c_supported"].float()     # (N_sup,  L_sel, D)
    c_unsup = unsup["c_unsupported"].float() # (N_unsup, L_all, D)

    sup_layers   = sup["layers"]    # e.g. [20, 28, 29, 30]
    unsup_layers = unsup["layers"]  # 0..31 or 0..33
    unsup_pos    = {l: i for i, l in enumerate(unsup_layers)}

    N_sup, L_sel, D = c_sup.shape
    N_unsup = c_unsup.shape[0]

    log.info("")
    log.info("Model: %s  rank=%d", model_key, rank)
    log.info("  c_supported  N=%d, L=%d, D=%d", N_sup,   L_sel, D)
    log.info("  c_unsupported N=%d, L=%d, D=%d", N_unsup, len(unsup_layers), D)

    c_clean_layers: list[torch.Tensor] = []
    V_layers:       list[torch.Tensor] = []

    log.info("  %-8s  %10s  %10s  %10s  %10s",
             "Layer", "rho_raw", "rho_clean", "energy_ret", "expected")

    for li, layer_idx in enumerate(sup_layers):
        pos = unsup_pos.get(layer_idx)
        if pos is None:
            log.warning("  Layer %d not in unsupported bundle — skipping", layer_idx)
            continue

        c_sup_l   = c_sup[:, li, :]       # (N_sup, D)
        c_unsup_l = c_unsup[:, pos, :]    # (N_unsup, D)

        r_eff = min(rank, N_sup - 1, D)
        V_r   = fit_pca_basis(c_sup_l, r_eff)                  # (D, r)
        rho_raw   = projection_ratio(c_unsup_l, V_r)
        c_clean_l = project_out(c_unsup_l, V_r)                # (N_unsup, D)
        rho_clean = projection_ratio(c_clean_l, V_r)
        energy_ret = float(
            c_clean_l.norm(dim=-1).pow(2).mean() /
            c_unsup_l.float().norm(dim=-1).pow(2).mean().clamp(min=1e-12)
        )
        expected = 1.0 - rho_raw

        log.info("  L%-7d  %10.4f  %10.6f  %10.4f  %10.4f",
                 layer_idx, rho_raw, rho_clean, energy_ret, expected)

        c_clean_layers.append(c_clean_l.cpu())
        V_layers.append(V_r.cpu())

    c_clean_tensor = torch.stack(c_clean_layers, dim=1)  # (N, L_sel, D)
    V_tensor       = torch.stack(V_layers,       dim=0)  # (L_sel, D, r)

    bundle = {
        "model":                unsup["model"],
        "layers":               sup_layers,
        "supported_rank":       rank,
        "k":                    unsup.get("k", 8),
        "datasets":             unsup["datasets"],
        "prompt_ids":           unsup["prompt_ids"],
        "c_unsupported_clean":  c_clean_tensor,
        "V_supported":          V_tensor,
    }

    torch.save(bundle, out_path)
    log.info("  Saved → %s  shape=%s", out_path.name, list(c_clean_tensor.shape))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cleaned unsupported contrasts")
    p.add_argument("--model", choices=MODEL_KEYS, nargs="*", default=None)
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    keys  = args.model or MODEL_KEYS
    log.info("Computing cleaned contrasts  rank=%d", args.rank)
    for key in keys:
        clean_model(key, args.rank)
    log.info("")
    log.info("Done → %s", ACTIVATIONS_DIR)


if __name__ == "__main__":
    main()
