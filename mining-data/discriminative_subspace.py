#!/usr/bin/env python3
"""
Discriminative retain-normalised commitment subspace (Option B).

Instead of finding directions of high commitment variance vs. retain, this
finds directions where **unsupported** commitment varies *more than*
**supported** commitment, while still avoiding retain. Three-way
discrimination instead of two-way.

Generalised eigenproblem (per layer, projected into the retain span):

    (Sigma_C - Sigma_A) v = gamma Sigma_R v

where
    Sigma_C = cov(c_unsupported)       # raw unsupported commitment contrasts
    Sigma_A = cov(c_supported)         # supported answerable contrasts
    Sigma_R = cov(retain_activations)  # retain (capability) activations

Eigenvalues can be negative (the LHS is indefinite). We rank by descending
gamma and keep the top-rank directions; large positive gamma = direction
where unsupported commitment dominates supported answering and avoids retain.

Inputs (per model, slice = last-25 % layers):
    unsupported_commitment_contrasts_<model>.pt   (all 32 layers; we slice)
    supported_answer_contrasts_<model>.pt         (already last-25%)
    retain_activations_<model>_last25.pt          (already last-25%)

Output:
    discriminative_subspace_<model>_last25_r<rank>.pt

Also prints a comparison vs. the cleaned and raw subspaces if those files
exist (no analysis file is written; results are stdout only).

Usage:
    python3 mining-data/discriminative_subspace.py --model qwen_instruct --rank 32
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
    "qwen_instruct", "qwen_base",
    "ministral_instruct", "ministral_base",
]

DEFAULT_RANK        = 8
DEFAULT_RIDGE       = 1e-3
DEFAULT_RETAIN_RANK = 512


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _load(path: Path, kind: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path.name}")
    log.info("  Loading %-9s %s", kind + ":", path.name)
    return torch.load(path, map_location="cpu", weights_only=False)


# ─── Per-layer computation ─────────────────────────────────────────────────────

def _whitening_eigh(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fallback for indefinite A, SPD B."""
    vals_b, vecs_b = np.linalg.eigh(B)
    vals_b = np.clip(vals_b, 1e-12, None)
    W_w    = vecs_b @ np.diag(1.0 / np.sqrt(vals_b)) @ vecs_b.T
    M      = W_w @ A @ W_w
    vals_m, vecs_m = np.linalg.eigh(M)
    vecs_orig = W_w @ vecs_m
    idx = np.argsort(vals_m)[::-1]
    return vals_m[idx], vecs_orig[:, idx]


def _process_layer(
    C_l: np.ndarray, A_l: np.ndarray, R_l: np.ndarray,
    rank: int, ridge: float, retain_rank: int,
) -> dict:
    N_r, D = R_l.shape
    N_c    = C_l.shape[0]
    N_a    = A_l.shape[0]

    # Step 1: retain basis (avoid null-space collapse)
    R_c = R_l - R_l.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(R_c, full_matrices=False)
    k = min(retain_rank, N_r - 1, D, Vt.shape[0])
    W = Vt[:k].T

    # Step 2: project all three matrices into retain span
    C_c = C_l - C_l.mean(axis=0, keepdims=True)
    A_c = A_l - A_l.mean(axis=0, keepdims=True)
    C_p = C_c @ W
    A_p = A_c @ W
    R_p = R_c @ W

    Sigma_C = (C_p.T @ C_p) / max(N_c - 1, 1)
    Sigma_A = (A_p.T @ A_p) / max(N_a - 1, 1)
    Sigma_R = (R_p.T @ R_p) / max(N_r - 1, 1)
    LHS     = Sigma_C - Sigma_A   # may be indefinite

    ridge_scale = float(np.diag(Sigma_R).mean()) * ridge
    Sigma_R_reg = Sigma_R + ridge_scale * np.eye(k, dtype=np.float64)

    # Step 3: solve generalised eigenproblem (B SPD, A indefinite OK for eigh)
    if _SCIPY:
        try:
            vals, vecs = sla.eigh(LHS, Sigma_R_reg)
            idx = np.argsort(vals)[::-1]
            vals, vecs = vals[idx], vecs[:, idx]
        except Exception as exc:
            log.warning("  scipy.linalg.eigh failed (%s); using whitening", exc)
            vals, vecs = _whitening_eigh(LHS, Sigma_R_reg)
    else:
        vals, vecs = _whitening_eigh(LHS, Sigma_R_reg)

    a     = vecs[:, :rank]
    V_top = W @ a
    V_top /= np.linalg.norm(V_top, axis=0, keepdims=True).clip(min=1e-12)

    # Diagnostics on the chosen top-rank subspace
    c_proj = float(np.trace(a.T @ Sigma_C @ a))
    a_proj = float(np.trace(a.T @ Sigma_A @ a))
    r_proj = float(np.trace(a.T @ Sigma_R @ a))

    return {
        "V":              V_top,
        "gamma":          vals[:rank],
        "gamma_full":     vals,
        "commit_proj":    c_proj,
        "answer_proj":    a_proj,
        "retain_proj":    r_proj,
    }


# ─── Per-model run ─────────────────────────────────────────────────────────────

def run_model(
    model_key: str, rank: int, ridge: float,
    retain_rank: int, last_pct: float = 0.25,
) -> Path:
    log.info("")
    log.info("=" * 64)
    log.info("MODEL  %s  rank=%d (DISCRIMINATIVE: C - A vs R)", model_key, rank)

    raw_bundle      = _load(ACTIVATIONS_DIR / f"unsupported_commitment_contrasts_{model_key}.pt", "raw_C")
    supported_bndl  = _load(ACTIVATIONS_DIR / f"supported_answer_contrasts_{model_key}.pt",       "raw_A")
    retain_bundle   = _load(ACTIVATIONS_DIR / f"retain_activations_{model_key}_last25.pt",        "retain")

    c_raw    = raw_bundle["c_unsupported"]                # (N_c, L_total, D)
    a_raw    = supported_bndl["c_supported"]              # (N_a, L_last25, D)
    r_acts   = retain_bundle["retain_activations"]        # (N_r, L_last25, D)

    a_layers = supported_bndl["layers"]
    r_layers = retain_bundle["layers"]

    L_total = c_raw.shape[1]
    last_n  = max(1, int(round(L_total * last_pct)))
    expected_layers = list(range(L_total - last_n, L_total))
    common_layers = [
        l for l in expected_layers if l in r_layers and l in a_layers
    ]
    if not common_layers:
        raise RuntimeError(f"No common layers across C/A/R for {model_key}")

    a_pos = {l: i for i, l in enumerate(a_layers)}
    r_pos = {l: i for i, l in enumerate(r_layers)}

    N_c, _, D = c_raw.shape
    log.info("  N_C=%d  N_A=%d  N_R=%d  D=%d  layers=%s",
             N_c, a_raw.shape[0], r_acts.shape[0], D, common_layers)
    log.info("")
    log.info("  %-8s  %10s  %10s  %10s  %10s  %10s  %10s",
             "Layer", "gamma_1", f"gamma_{rank}", "C_proj", "A_proj", "R_proj", "(C-A)/R")
    log.info("  " + "-" * 80)

    V_all, gamma_all, full_eig_all = [], [], []
    c_all, a_all, r_all = [], [], []

    for layer_idx in common_layers:
        C_l = c_raw[:, layer_idx, :].numpy().astype(np.float64)
        A_l = a_raw[:, a_pos[layer_idx], :].numpy().astype(np.float64)
        R_l = r_acts[:, r_pos[layer_idx], :].numpy().astype(np.float64)

        out = _process_layer(C_l, A_l, R_l,
                             rank=rank, ridge=ridge, retain_rank=retain_rank)

        ratio = (out["commit_proj"] - out["answer_proj"]) / max(out["retain_proj"], 1e-12)
        log.info(
            "  L%-7d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
            layer_idx,
            float(out["gamma"][0]),
            float(out["gamma"][min(rank - 1, len(out["gamma"]) - 1)]),
            out["commit_proj"], out["answer_proj"], out["retain_proj"], ratio,
        )

        V_all.append(torch.tensor(out["V"], dtype=torch.float32))
        gamma_all.append(torch.tensor(out["gamma"], dtype=torch.float32))
        full_eig_all.append(torch.tensor(out["gamma_full"], dtype=torch.float32))
        c_all.append(out["commit_proj"])
        a_all.append(out["answer_proj"])
        r_all.append(out["retain_proj"])

    V_tensor = torch.stack(V_all, dim=0)
    gamma_tensor = torch.stack(gamma_all, dim=0)

    log.info("")
    log.info("  Mean C_proj=%.4f  A_proj=%.4f  R_proj=%.4f  (C-A)/R=%.4f",
             np.mean(c_all), np.mean(a_all), np.mean(r_all),
             (np.mean(c_all) - np.mean(a_all)) / max(np.mean(r_all), 1e-12))

    out_path = ACTIVATIONS_DIR / f"discriminative_subspace_{model_key}_last25_r{rank}.pt"
    bundle = {
        "model":                   raw_bundle["model"],
        "layers":                  common_layers,
        "rank":                    rank,
        "ridge":                   ridge,
        "retain_rank":             min(retain_rank, r_acts.shape[0] - 1, D),
        "source":                  "discriminative: (Sigma_C - Sigma_A) v = gamma Sigma_R v",
        # train-compatible (drop-in) keys
        "V_retain_normalised":     V_tensor,
        "generalized_eigenvalues": gamma_tensor,
        "commitment_projection":   torch.tensor(c_all, dtype=torch.float32),
        "retain_projection":       torch.tensor(r_all, dtype=torch.float32),
        # extras for analysis
        "answer_projection":       torch.tensor(a_all, dtype=torch.float32),
        "full_eigenvalues":        full_eig_all,
    }
    torch.save(bundle, out_path)
    log.info("  Saved -> %s  V shape=%s", out_path.name, list(V_tensor.shape))
    return out_path


# ─── Comparison vs cleaned / raw ──────────────────────────────────────────────

def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Q_a, _ = np.linalg.qr(A)
    Q_b, _ = np.linalg.qr(B)
    s = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def compare_three_way(model_key: str, rank: int) -> None:
    paths = {
        "clean": ACTIVATIONS_DIR / f"retain_normalised_subspace_{model_key}_last25_r{rank}.pt",
        "raw":   ACTIVATIONS_DIR / f"raw_subspace_{model_key}_last25_r{rank}.pt",
        "disc":  ACTIVATIONS_DIR / f"discriminative_subspace_{model_key}_last25_r{rank}.pt",
    }
    bundles = {}
    for name, p in paths.items():
        if not p.exists():
            log.info("  (skip 3-way comparison: %s missing)", p.name)
            return
        bundles[name] = torch.load(p, map_location="cpu", weights_only=False)

    log.info("")
    log.info("=" * 64)
    log.info("3-way comparison (clean vs raw vs discriminative)  rank=%d", rank)
    log.info("")
    log.info("  Mean across last-25%% layers:")
    for name, b in bundles.items():
        cp = b["commitment_projection"].numpy().mean()
        rp = b["retain_projection"].numpy().mean()
        g1 = b["generalized_eigenvalues"][:, 0].numpy().mean()
        gk = b["generalized_eigenvalues"][:, -1].numpy().mean()
        extra = ""
        if "answer_projection" in b:
            ap = b["answer_projection"].numpy().mean()
            extra = f"  A_proj={ap:7.3f}  (C-A)/R={(cp-ap)/max(rp,1e-12):.2f}"
        log.info("    %-5s  gamma_1=%7.3f  gamma_%d=%6.3f  C_proj=%7.3f  R_proj=%6.3f  C/R=%6.2f%s",
                 name, g1, rank, gk, cp, rp, cp / max(rp, 1e-12), extra)

    log.info("")
    log.info("  Subspace alignment (mean cos(theta) over rank dims, averaged over layers):")
    V_clean = bundles["clean"]["V_retain_normalised"].numpy()
    V_raw   = bundles["raw"]["V_retain_normalised"].numpy()
    V_disc  = bundles["disc"]["V_retain_normalised"].numpy()
    L = V_clean.shape[0]

    def _avg_overlap(A_all, B_all):
        return float(np.mean([_principal_angles(A_all[i], B_all[i]).mean() for i in range(L)]))

    log.info("    clean ↔ raw  : %.3f", _avg_overlap(V_clean, V_raw))
    log.info("    clean ↔ disc : %.3f", _avg_overlap(V_clean, V_disc))
    log.info("    raw   ↔ disc : %.3f", _avg_overlap(V_raw,   V_disc))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discriminative (Option B) retain-normalised subspace."
    )
    p.add_argument("--model",       choices=MODEL_KEYS, nargs="*", default=None)
    p.add_argument("--rank",        type=int,   default=DEFAULT_RANK)
    p.add_argument("--ridge",       type=float, default=DEFAULT_RIDGE)
    p.add_argument("--retain-rank", type=int,   default=DEFAULT_RETAIN_RANK)
    p.add_argument("--no-compare",  action="store_true",
                   help="Skip 3-way comparison vs cleaned and raw bundles")
    return p.parse_args()


def main() -> None:
    if not _SCIPY:
        log.warning("scipy not found; using numpy whitening fallback")

    args   = parse_args()
    models = args.model or MODEL_KEYS

    log.info("Discriminative subspace  (Sigma_C - Sigma_A) v = gamma Sigma_R v")
    log.info("  Models : %s", models)
    log.info("  Rank   : %d", args.rank)

    for model_key in models:
        try:
            run_model(model_key, rank=args.rank, ridge=args.ridge,
                      retain_rank=args.retain_rank)
            if not args.no_compare:
                compare_three_way(model_key, rank=args.rank)
        except FileNotFoundError as exc:
            log.error("  SKIP %s -- %s", model_key, exc)

    log.info("")
    log.info("Done -> %s", ACTIVATIONS_DIR)


if __name__ == "__main__":
    main()
