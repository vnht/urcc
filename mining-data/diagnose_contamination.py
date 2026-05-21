#!/usr/bin/env python3
"""
Diagnose supported-direction contamination in unsupported-commitment contrasts.

For each model and selected layer l, fits a rank-r PCA subspace S_l on
c_supported (the supported-answering direction), then measures what fraction
of the unsupported-commitment contrast c_unsupported lies in that subspace:

    rho_l(r) = E[ ||P_{S_l} c_unsupported_l||² / ||c_unsupported_l||² ]

where P_{S_l} = V_r V_r^T  (V_r are the top-r PCA components of c_supported_l).

A high rho means raw c_unsupported contains legitimate supported-answering
directions that should be projected out before unlearning.

Usage:
    python3 mining-data/diagnose_contamination.py
    python3 mining-data/diagnose_contamination.py --model qwen_instruct
    python3 mining-data/diagnose_contamination.py --r 4 8 16 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent
ACTIVATIONS_DIR = BASE_DIR / "activations"

MODEL_KEYS = ["qwen_instruct", "qwen_base", "ministral_instruct", "ministral_base"]

# Must match extract_supported_activations.py
SELECTED_LAYERS = {
    "qwen_instruct":      [20, 28, 29, 30],
    "qwen_base":          [20, 28, 29, 30],
    "ministral_instruct": [22, 31, 32, 33],
    "ministral_base":     [22, 31, 32, 33],
}

DEFAULT_R_VALUES = [4, 8, 16, 32]


# ─── PCA subspace projection ──────────────────────────────────────────────────

def fit_pca_subspace(X: torch.Tensor, r: int) -> torch.Tensor:
    """
    Fit rank-r PCA on X (N, D). Returns V_r (D, r) — top-r right singular
    vectors (principal components). Uses torch.linalg.svd for efficiency.
    """
    X_f = X.float()
    X_c = X_f - X_f.mean(dim=0, keepdim=True)   # centre
    # Thin SVD: U (N,k), S (k,), Vh (k,D)  where k = min(N,D)
    _, _, Vh = torch.linalg.svd(X_c, full_matrices=False)
    return Vh[:r].T   # (D, r)


def projection_ratio(
    c_unsup: torch.Tensor,    # (N, D)  — vectors to probe
    V_r: torch.Tensor,        # (D, r)  — subspace basis
) -> float:
    """
    Mean fraction of squared norm of c_unsup that lies in the subspace spanned
    by V_r:   E[ ||V_r V_r^T x||² / ||x||² ]
    """
    c_f    = c_unsup.float()
    proj   = c_f @ V_r @ V_r.T                          # (N, D)
    num    = proj.norm(dim=-1).pow(2)                    # (N,)
    denom  = c_f.norm(dim=-1).pow(2).clamp(min=1e-12)   # (N,)
    return float((num / denom).mean())


# ─── Per-model diagnostic ─────────────────────────────────────────────────────

def diagnose_model(model_key: str, r_values: list[int]) -> None:
    sup_path   = ACTIVATIONS_DIR / f"supported_answer_contrasts_{model_key}.pt"
    unsup_path = ACTIVATIONS_DIR / f"unsupported_commitment_contrasts_{model_key}.pt"

    if not sup_path.exists():
        log.warning("  Missing supported file: %s — skipping %s", sup_path.name, model_key)
        return
    if not unsup_path.exists():
        log.warning("  Missing unsupported file: %s — skipping %s", unsup_path.name, model_key)
        return

    sup   = torch.load(sup_path,   map_location="cpu", weights_only=False)
    unsup = torch.load(unsup_path, map_location="cpu", weights_only=False)

    c_sup   = sup["c_supported"].float()    # (N_sup,  L_sup, D)
    c_unsup = unsup["c_unsupported"].float() # (N_unsup, L_all, D)

    sup_layers   = sup["layers"]    # e.g. [20, 28, 29, 30]
    unsup_layers = unsup["layers"]  # 0..31 or 0..33 (all layers)

    # Build a mapping: selected layer index → position in c_unsup
    unsup_layer_pos = {l: i for i, l in enumerate(unsup_layers)}

    N_sup,   L_sup,  D = c_sup.shape
    N_unsup, L_all, _  = c_unsup.shape

    log.info("")
    log.info("Model: %s", model_key)
    log.info("  c_supported  : N=%d, L=%d (selected), D=%d", N_sup, L_sup, D)
    log.info("  c_unsupported: N=%d, L=%d (all), D=%d", N_unsup, L_all, D)

    # Header
    header = f"  {'Layer':<8}" + "".join(f"  r={r:<4}" for r in r_values)
    log.info(header)
    log.info("  " + "-" * (8 + 8 * len(r_values)))

    results: list[dict] = []
    for li, layer_idx in enumerate(sup_layers):
        c_sup_l   = c_sup[:, li, :]   # (N_sup, D)

        unsup_pos = unsup_layer_pos.get(layer_idx)
        if unsup_pos is None:
            log.warning("  Layer %d not found in unsupported bundle — skipping", layer_idx)
            continue
        c_unsup_l = c_unsup[:, unsup_pos, :]   # (N_unsup, D)

        row: dict = {"layer": layer_idx}
        parts = [f"  L{layer_idx:<6}"]
        for r in r_values:
            r_eff = min(r, N_sup, D)
            V_r   = fit_pca_subspace(c_sup_l, r_eff)
            rho   = projection_ratio(c_unsup_l, V_r)
            row[f"rho_r{r}"] = round(rho, 5)
            parts.append(f"  {rho:.4f} ")
        log.info("".join(parts))
        results.append(row)

    return results


# ─── Summary across models ────────────────────────────────────────────────────

def print_summary(all_results: dict[str, list[dict]], r_values: list[int]) -> None:
    log.info("")
    log.info("=" * 64)
    log.info("Summary — mean rho across layers, per model")
    log.info("=" * 64)

    header = f"  {'Model':<24}" + "".join(f"  r={r:<6}" for r in r_values)
    log.info(header)
    log.info("  " + "-" * (24 + 9 * len(r_values)))

    for model_key, rows in sorted(all_results.items()):
        if not rows:
            continue
        parts = [f"  {model_key:<24}"]
        for r in r_values:
            key = f"rho_r{r}"
            vals = [row[key] for row in rows if key in row]
            mean_rho = sum(vals) / len(vals) if vals else float("nan")
            parts.append(f"  {mean_rho:.4f}  ")
        log.info("".join(parts))

    log.info("")
    log.info("Interpretation:")
    log.info("  rho < 0.05  → minimal contamination; c_unsupported is largely orthogonal")
    log.info("  rho 0.05–0.15 → moderate — consider projecting out S_l before unlearning")
    log.info("  rho > 0.15  → high — raw c_unsupported contains substantial supported-")
    log.info("                answering signal; projection recommended")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose supported-direction contamination")
    p.add_argument(
        "--model",
        choices=MODEL_KEYS,
        nargs="*",
        default=None,
        help="One or more model keys (default: all four)",
    )
    p.add_argument(
        "--r",
        type=int,
        nargs="+",
        default=DEFAULT_R_VALUES,
        metavar="R",
        help="PCA rank values to test (default: 4 8 16 32)",
    )
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    keys  = args.model or MODEL_KEYS
    r_vals = sorted(set(args.r))

    log.info("Contamination diagnostic")
    log.info("  Models : %s", keys)
    log.info("  r vals : %s", r_vals)
    log.info("  Activations dir: %s", ACTIVATIONS_DIR)

    all_results: dict[str, list[dict]] = {}
    for model_key in keys:
        rows = diagnose_model(model_key, r_vals)
        all_results[model_key] = rows or []

    print_summary(all_results, r_vals)


if __name__ == "__main__":
    main()
