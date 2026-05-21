#!/usr/bin/env python3
"""
Retain-normalised residual commitment subspace.

Solves the generalised eigenproblem

    Sigma_C v = gamma Sigma_R v

per layer, projected into the retain span to avoid the null-space
collapse that occurs when N_retain << D.

Algorithm per layer:
  1. SVD of centred R_l -> retain basis W (D, k), k = min(N_retain-1, retain_rank)
  2. Project both data matrices into this k-space (avoids D x D covariances)
  3. Solve eigh(Sigma_C_proj, Sigma_R_proj_reg) in k-space -> full-rank, well-conditioned
  4. Map top-rank eigenvectors back to D-space and normalise

Inputs (per model):
    cleaned_unsupported_contrasts_<model>_last25_r8.pt   (or _r8.pt fallback)
    retain_activations_<model>_last25.pt

Output (per model):
    retain_normalised_subspace_<model>_last25_r8.pt

Usage:
    python3 mining-data/retain_normalised_subspace.py
    python3 mining-data/retain_normalised_subspace.py --model qwen_instruct
    python3 mining-data/retain_normalised_subspace.py --rank 8 --ridge 1e-3
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

try:
    import scipy.linalg as sla
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
ACTIVATIONS_DIR = BASE_DIR / "activations"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_KEYS = [
    "qwen_instruct",
    "qwen_base",
    "ministral_instruct",
    "ministral_base",
]

DEFAULT_RANK        = 8
DEFAULT_RIDGE       = 1e-3
DEFAULT_RETAIN_RANK = 512   # retain PCA components to keep (capped at N_retain-1)


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _load_cleaned_contrasts(model_key: str) -> dict:
    """Load cleaned contrasts; tries _last25_r8 first, falls back to _r8."""
    p1 = ACTIVATIONS_DIR / f"cleaned_unsupported_contrasts_{model_key}_last25_r8.pt"
    p2 = ACTIVATIONS_DIR / f"cleaned_unsupported_contrasts_{model_key}_r8.pt"
    if p1.exists():
        log.info("  Loading contrasts: %s", p1.name)
        return torch.load(p1, map_location="cpu", weights_only=False)
    if p2.exists():
        log.warning("  _last25_r8 not found; falling back to %s", p2.name)
        return torch.load(p2, map_location="cpu", weights_only=False)
    raise FileNotFoundError(
        f"No cleaned contrast file found for {model_key}. "
        f"Expected {p1.name} or {p2.name}."
    )


def _load_retain_activations(model_key: str) -> dict:
    p = ACTIVATIONS_DIR / f"retain_activations_{model_key}_last25.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"Retain activation file not found: {p.name}\n"
            "Run extract_retain_activations.py first."
        )
    log.info("  Loading retain: %s", p.name)
    return torch.load(p, map_location="cpu", weights_only=False)


# ─── Per-layer computation ─────────────────────────────────────────────────────

def _whitening_eigh(
    Sigma_C: np.ndarray,
    Sigma_R_reg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: whiten by Sigma_R_reg, solve standard eigenproblem."""
    vals_r, vecs_r = np.linalg.eigh(Sigma_R_reg)
    vals_r = np.clip(vals_r, 1e-12, None)
    W_w    = vecs_r @ np.diag(1.0 / np.sqrt(vals_r)) @ vecs_r.T
    M      = W_w @ Sigma_C @ W_w
    vals_m, vecs_m = np.linalg.eigh(M)
    vecs_orig = W_w @ vecs_m
    idx = np.argsort(vals_m)[::-1]
    return vals_m[idx], vecs_orig[:, idx]


def _process_layer(
    C_l: np.ndarray,    # (N_commit, D)
    R_l: np.ndarray,    # (N_retain, D)
    rank: int,
    ridge: float,
    retain_rank: int,
    layer_idx: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Project into the retain span to fix null-space collapse, then solve
    the generalised eigenproblem in that low-dimensional subspace.

    Returns (V_l, gamma_top, commitment_proj, retain_proj)
    where V_l has shape (D, rank).
    """
    N_r, D = R_l.shape
    N_c    = C_l.shape[0]

    # Step 1: retain basis via SVD of centred R_l
    R_c = R_l - R_l.mean(axis=0, keepdims=True)        # (N_r, D)
    _, _, Vt = np.linalg.svd(R_c, full_matrices=False)  # Vt: (min(N_r,D), D)
    k = min(retain_rank, N_r - 1, D, Vt.shape[0])
    W = Vt[:k].T                                         # (D, k)

    # Step 2: project data matrices into k-space (never form D x D covariances)
    C_c = C_l - C_l.mean(axis=0, keepdims=True)
    C_p = C_c @ W   # (N_c, k)
    R_p = R_c @ W   # (N_r, k)

    Sigma_C_proj = (C_p.T @ C_p) / max(N_c - 1, 1)   # (k, k)
    Sigma_R_proj = (R_p.T @ R_p) / max(N_r - 1, 1)   # (k, k) -- now full rank

    # Step 3: regularise and solve in k-space
    ridge_scale = float(np.diag(Sigma_R_proj).mean()) * ridge
    Sigma_R_reg = Sigma_R_proj + ridge_scale * np.eye(k, dtype=np.float64)

    if _SCIPY:
        try:
            vals, vecs = sla.eigh(Sigma_C_proj, Sigma_R_reg)
            idx = np.argsort(vals)[::-1]
            vals, vecs = vals[idx], vecs[:, idx]
        except Exception as exc:
            log.warning("  scipy.linalg.eigh failed (%s); using whitening fallback", exc)
            vals, vecs = _whitening_eigh(Sigma_C_proj, Sigma_R_reg)
    else:
        vals, vecs = _whitening_eigh(Sigma_C_proj, Sigma_R_reg)

    # Step 4: map top-rank back to D-space and normalise
    a     = vecs[:, :rank]   # (k, rank)
    V_top = W @ a            # (D, rank)
    norms = np.linalg.norm(V_top, axis=0, keepdims=True).clip(min=1e-12)
    V_top /= norms

    gamma_top = vals[:rank]

    # Diagnostics computed in k-space (equivalent to D-space)
    commit_proj = float(np.trace(a.T @ Sigma_C_proj @ a))
    retain_proj = float(np.trace(a.T @ Sigma_R_proj @ a))

    return V_top, gamma_top, commit_proj, retain_proj


# ─── Per-model run ─────────────────────────────────────────────────────────────

def run_model(model_key: str, rank: int, ridge: float, retain_rank: int) -> None:
    log.info("")
    log.info("=" * 64)
    log.info("MODEL  %s  rank=%d  ridge=%g  retain_rank=%d",
             model_key, rank, ridge, retain_rank)

    contrast_bundle = _load_cleaned_contrasts(model_key)
    retain_bundle   = _load_retain_activations(model_key)

    c_clean = contrast_bundle["c_unsupported_clean"]   # (N_c, L_c, D)
    r_acts  = retain_bundle["retain_activations"]       # (N_r, L_r, D)

    c_layers = contrast_bundle["layers"]
    r_layers = retain_bundle["layers"]

    r_layer_pos   = {l: i for i, l in enumerate(r_layers)}
    common_layers = [l for l in c_layers if l in r_layer_pos]
    if not common_layers:
        log.error("  No common layers: contrasts=%s retain=%s", c_layers, r_layers)
        return
    if len(common_layers) < len(c_layers):
        missing = [l for l in c_layers if l not in r_layer_pos]
        log.warning("  Layers in contrasts but not retain: %s -- skipping", missing)

    N_c, _, D = c_clean.shape
    N_r       = r_acts.shape[0]
    k_eff     = min(retain_rank, N_r - 1, D)
    log.info("  N_commit=%d  N_retain=%d  D=%d  k_eff=%d  layers=%s",
             N_c, N_r, D, k_eff, common_layers)

    log.info("")
    log.info("  %-8s  %10s  %10s  %10s  %10s  %10s",
             "Layer", "gamma_1", "gamma_8", "C_proj", "R_proj", "C/R")
    log.info("  " + "-" * 68)

    V_all      = []
    gamma_all  = []
    c_proj_all = []
    r_proj_all = []

    for layer_idx in common_layers:
        ci = c_layers.index(layer_idx)
        ri = r_layer_pos[layer_idx]

        C_l = c_clean[:, ci, :].numpy().astype(np.float64)
        R_l = r_acts[:, ri, :].numpy().astype(np.float64)

        V_l, gamma_top, c_proj, r_proj = _process_layer(
            C_l, R_l, rank=rank, ridge=ridge,
            retain_rank=retain_rank, layer_idx=layer_idx,
        )

        ratio = c_proj / max(r_proj, 1e-12)
        log.info(
            "  L%-7d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
            layer_idx,
            float(gamma_top[0]),
            float(gamma_top[min(rank - 1, len(gamma_top) - 1)]),
            c_proj, r_proj, ratio,
        )

        V_all.append(torch.tensor(V_l,        dtype=torch.float32))
        gamma_all.append(torch.tensor(gamma_top, dtype=torch.float32))
        c_proj_all.append(c_proj)
        r_proj_all.append(r_proj)

    V_tensor     = torch.stack(V_all,     dim=0)  # (L, D, rank)
    gamma_tensor = torch.stack(gamma_all, dim=0)  # (L, rank)

    log.info("")
    log.info("  Mean C_proj=%.4f  Mean R_proj=%.4f  Mean C/R=%.4f",
             np.mean(c_proj_all), np.mean(r_proj_all),
             np.mean(c_proj_all) / max(np.mean(r_proj_all), 1e-12))

    out_path = ACTIVATIONS_DIR / f"retain_normalised_subspace_{model_key}_last25_r{rank}.pt"
    bundle = {
        "model":                   contrast_bundle["model"],
        "layers":                  common_layers,
        "rank":                    rank,
        "ridge":                   ridge,
        "retain_rank":             k_eff,
        "V_retain_normalised":     V_tensor,
        "generalized_eigenvalues": gamma_tensor,
        "retain_projection":       torch.tensor(r_proj_all, dtype=torch.float32),
        "commitment_projection":   torch.tensor(c_proj_all, dtype=torch.float32),
    }
    torch.save(bundle, out_path)
    log.info("  Saved -> %s  V shape=%s", out_path.name, list(V_tensor.shape))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retain-normalised residual commitment subspace."
    )
    p.add_argument("--model",       choices=MODEL_KEYS, nargs="*", default=None)
    p.add_argument("--rank",        type=int,   default=DEFAULT_RANK)
    p.add_argument("--ridge",       type=float, default=DEFAULT_RIDGE)
    p.add_argument("--retain-rank", type=int,   default=DEFAULT_RETAIN_RANK,
                   help="Number of retain PCA components to keep (default: 512)")
    p.add_argument("--layers",      default="last25",
                   help="Layer selection tag (informational only)")
    return p.parse_args()


def main() -> None:
    if not _SCIPY:
        log.warning("scipy not found; using numpy whitening fallback for eigenproblem")

    args   = parse_args()
    models = args.model or MODEL_KEYS

    log.info("Retain-normalised subspace  (retain-span projection)")
    log.info("  Models      : %s", models)
    log.info("  Rank        : %d", args.rank)
    log.info("  Ridge       : %g", args.ridge)
    log.info("  Retain rank : %d", args.retain_rank)

    for model_key in models:
        try:
            run_model(model_key, rank=args.rank, ridge=args.ridge,
                      retain_rank=args.retain_rank)
        except FileNotFoundError as exc:
            log.error("  SKIP %s -- %s", model_key, exc)

    log.info("")
    log.info("Done -> %s", ACTIVATIONS_DIR)


if __name__ == "__main__":
    main()
