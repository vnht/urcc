#!/usr/bin/env python3
"""
Raw retain-normalised subspace.

Same generalised eigenproblem and retain-span projection as
``retain_normalised_subspace.py``, but uses **raw** unsupported contrasts
(``c_unsupported``) directly instead of the cleaned version
(``c_unsupported_clean`` from ``compute_cleaned_contrasts.py``).

Hypothesis under test: the cleaning step that projects out the
supported-answering subspace also removes the commitment signal we want
to suppress. If so, eigenvalues built from raw contrasts should decay
more sharply, indicate stronger commitment-vs-retain ratios, and / or
yield a more behaviourally effective forget direction once trained on.

Inputs (per model, slice = last-25 % layers, matching the cleaned pipeline):
    unsupported_commitment_contrasts_<model>.pt  (all 32 layers; we slice)
    retain_activations_<model>_last25.pt

Output:
    raw_subspace_<model>_last25_r<rank>.pt

Usage:
    python3 mining-data/raw_subspace.py
    python3 mining-data/raw_subspace.py --model qwen_instruct --rank 32
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
DEFAULT_RETAIN_RANK = 512


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _load_raw_contrasts(model_key: str) -> dict:
    p = ACTIVATIONS_DIR / f"unsupported_commitment_contrasts_{model_key}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"Raw contrast file not found: {p.name}\n"
            "Run extract_activations.py first."
        )
    log.info("  Loading raw contrasts: %s", p.name)
    return torch.load(p, map_location="cpu", weights_only=False)


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
    vals_r, vecs_r = np.linalg.eigh(Sigma_R_reg)
    vals_r = np.clip(vals_r, 1e-12, None)
    W_w    = vecs_r @ np.diag(1.0 / np.sqrt(vals_r)) @ vecs_r.T
    M      = W_w @ Sigma_C @ W_w
    vals_m, vecs_m = np.linalg.eigh(M)
    vecs_orig = W_w @ vecs_m
    idx = np.argsort(vals_m)[::-1]
    return vals_m[idx], vecs_orig[:, idx]


def _process_layer(
    C_l: np.ndarray, R_l: np.ndarray,
    rank: int, ridge: float, retain_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Returns (V_top, gamma_top, gamma_full_in_k, c_proj, r_proj)."""
    N_r, D = R_l.shape
    N_c    = C_l.shape[0]

    R_c = R_l - R_l.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(R_c, full_matrices=False)
    k = min(retain_rank, N_r - 1, D, Vt.shape[0])
    W = Vt[:k].T

    C_c = C_l - C_l.mean(axis=0, keepdims=True)
    C_p = C_c @ W
    R_p = R_c @ W

    Sigma_C_proj = (C_p.T @ C_p) / max(N_c - 1, 1)
    Sigma_R_proj = (R_p.T @ R_p) / max(N_r - 1, 1)

    ridge_scale = float(np.diag(Sigma_R_proj).mean()) * ridge
    Sigma_R_reg = Sigma_R_proj + ridge_scale * np.eye(k, dtype=np.float64)

    if _SCIPY:
        try:
            vals, vecs = sla.eigh(Sigma_C_proj, Sigma_R_reg)
            idx = np.argsort(vals)[::-1]
            vals, vecs = vals[idx], vecs[:, idx]
        except Exception as exc:
            log.warning("  scipy.linalg.eigh failed (%s); using whitening", exc)
            vals, vecs = _whitening_eigh(Sigma_C_proj, Sigma_R_reg)
    else:
        vals, vecs = _whitening_eigh(Sigma_C_proj, Sigma_R_reg)

    a     = vecs[:, :rank]
    V_top = W @ a
    norms = np.linalg.norm(V_top, axis=0, keepdims=True).clip(min=1e-12)
    V_top /= norms

    gamma_top = vals[:rank]

    commit_proj = float(np.trace(a.T @ Sigma_C_proj @ a))
    retain_proj = float(np.trace(a.T @ Sigma_R_proj @ a))

    return V_top, gamma_top, vals, commit_proj, retain_proj


# ─── Per-model run ─────────────────────────────────────────────────────────────

def run_model(
    model_key: str, rank: int, ridge: float,
    retain_rank: int, last_pct: float = 0.25,
) -> None:
    log.info("")
    log.info("=" * 64)
    log.info("MODEL  %s  rank=%d (RAW contrasts)", model_key, rank)

    raw_bundle    = _load_raw_contrasts(model_key)
    retain_bundle = _load_retain_activations(model_key)

    c_raw    = raw_bundle["c_unsupported"]              # (N_c, L_total, D)
    r_acts   = retain_bundle["retain_activations"]      # (N_r, L_last25, D)
    r_layers = retain_bundle["layers"]

    L_total = c_raw.shape[1]
    last_n  = max(1, int(round(L_total * last_pct)))
    expected_layers = list(range(L_total - last_n, L_total))

    common_layers = [l for l in expected_layers if l in r_layers]
    if not common_layers:
        log.error("  No common layers between contrasts (%s) and retain (%s)",
                  expected_layers, r_layers)
        return

    N_c, _, D = c_raw.shape
    N_r       = r_acts.shape[0]
    k_eff     = min(retain_rank, N_r - 1, D)
    log.info("  N_commit=%d  N_retain=%d  D=%d  k_eff=%d  layers=%s",
             N_c, N_r, D, k_eff, common_layers)

    log.info("")
    log.info("  %-8s  %10s  %10s  %10s  %10s  %10s",
             "Layer", "gamma_1", f"gamma_{rank}", "C_proj", "R_proj", "C/R")
    log.info("  " + "-" * 68)

    V_all, gamma_all = [], []
    full_eig_all = []
    c_proj_all, r_proj_all = [], []

    r_pos = {l: i for i, l in enumerate(r_layers)}

    for layer_idx in common_layers:
        C_l = c_raw[:, layer_idx, :].numpy().astype(np.float64)
        R_l = r_acts[:, r_pos[layer_idx], :].numpy().astype(np.float64)

        V_l, gamma_top, gamma_full, c_proj, r_proj = _process_layer(
            C_l, R_l, rank=rank, ridge=ridge, retain_rank=retain_rank,
        )

        ratio = c_proj / max(r_proj, 1e-12)
        log.info(
            "  L%-7d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
            layer_idx,
            float(gamma_top[0]),
            float(gamma_top[min(rank - 1, len(gamma_top) - 1)]),
            c_proj, r_proj, ratio,
        )

        V_all.append(torch.tensor(V_l, dtype=torch.float32))
        gamma_all.append(torch.tensor(gamma_top, dtype=torch.float32))
        full_eig_all.append(torch.tensor(gamma_full, dtype=torch.float32))
        c_proj_all.append(c_proj)
        r_proj_all.append(r_proj)

    V_tensor = torch.stack(V_all, dim=0)
    gamma_tensor = torch.stack(gamma_all, dim=0)

    log.info("")
    log.info("  Mean C_proj=%.4f  Mean R_proj=%.4f  Mean C/R=%.4f",
             np.mean(c_proj_all), np.mean(r_proj_all),
             np.mean(c_proj_all) / max(np.mean(r_proj_all), 1e-12))

    out_path = ACTIVATIONS_DIR / f"raw_subspace_{model_key}_last25_r{rank}.pt"
    bundle = {
        "model":                   raw_bundle["model"],
        "layers":                  common_layers,
        "rank":                    rank,
        "ridge":                   ridge,
        "retain_rank":             k_eff,
        "source":                  "raw c_unsupported (no cleaning)",
        # ── train-compatible keys (drop-in replacement for retain_normalised_subspace_*) ──
        "V_retain_normalised":     V_tensor,
        "generalized_eigenvalues": gamma_tensor,
        "commitment_projection":   torch.tensor(c_proj_all, dtype=torch.float32),
        "retain_projection":       torch.tensor(r_proj_all, dtype=torch.float32),
        # ── extra: full eigenvalue spectrum per layer for spectrum analysis ──
        "full_eigenvalues":        full_eig_all,
    }
    torch.save(bundle, out_path)
    log.info("  Saved -> %s  V shape=%s", out_path.name, list(V_tensor.shape))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw (uncleaned) retain-normalised commitment subspace."
    )
    p.add_argument("--model",       choices=MODEL_KEYS, nargs="*", default=None)
    p.add_argument("--rank",        type=int,   default=DEFAULT_RANK)
    p.add_argument("--ridge",       type=float, default=DEFAULT_RIDGE)
    p.add_argument("--retain-rank", type=int,   default=DEFAULT_RETAIN_RANK)
    return p.parse_args()


def main() -> None:
    if not _SCIPY:
        log.warning("scipy not found; using numpy whitening fallback")

    args   = parse_args()
    models = args.model or MODEL_KEYS

    log.info("Raw subspace (unCLEANED contrasts)")
    log.info("  Models : %s", models)
    log.info("  Rank   : %d", args.rank)
    log.info("  Ridge  : %g", args.ridge)
    log.info("  Retain : %d", args.retain_rank)

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
