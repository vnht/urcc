#!/usr/bin/env python3
"""Step 2 — Build the discriminative commitment subspace V.

For each late layer l, solve the generalised eigenproblem in the
general-utility span:

    (Σ_OC - Σ_LC) v = γ · Σ_E v

where
    Σ_OC = cov(c_OC)   c_OC = h_A − h_B   (over-commit minus its abstain baseline)
    Σ_LC = cov(c_LC)   c_LC = h_C − h_D   (legit-commit minus its abstain baseline)
    Σ_E  = cov(h_E)                       (general utility)

V_l ∈ ℝ^{D × r} are the top-r generalised eigenvectors. Large positive γ ⇒
direction along which over-commit varies more than legitimate-commit (after
subtracting the shared abstain-mode baseline), normalised against general
utility. V is shared across answerability domains; the per-domain
specialisation lives in step 3 (per-domain abstain pole μ⁻(d)) and in the
per-domain abstain templates used to build h_B and h_D in step 1.

Reads:  step1_extract_activations/data/activations_<model>.pt
Writes: step2_build_subspace/data/subspace_<model>_r<rank>.pt with keys:
    "model_key", "layers", "rank", "ridge", "retain_basis_rank",
    "V":           tensor[L, D, r]
    "gamma":       tensor[L, r]
    "diag":        list per-layer projection diagnostics

Run
---
    python step2_build_subspace/build_subspace.py --model qwen_instruct --rank 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import format_duration, log

try:
    import scipy.linalg as sla
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Per-layer eigenproblem ────────────────────────────────────────────────────

def _whitening_eigh(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generalised eigh fallback for indefinite A, SPD B (no scipy)."""
    vals_b, vecs_b = np.linalg.eigh(B)
    vals_b = np.clip(vals_b, 1e-12, None)
    W      = vecs_b @ np.diag(1.0 / np.sqrt(vals_b)) @ vecs_b.T
    M      = W @ A @ W
    vals_m, vecs_m = np.linalg.eigh(M)
    V_back = W @ vecs_m
    idx = np.argsort(vals_m)[::-1]
    return vals_m[idx], V_back[:, idx]


def _solve_layer(
    OC: np.ndarray, LC: np.ndarray, E: np.ndarray,
    *, rank: int, ridge: float, retain_basis_rank: int,
) -> dict:
    """One layer's generalised eigenproblem  (Σ_OC − Σ_LC) v = γ Σ_E v
    projected into the general-utility (E) span."""
    N_e, D = E.shape
    # 1. SVD basis from set E (avoid null-space collapse when N < D)
    E_c = E - E.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(E_c, full_matrices=False)
    k = int(min(retain_basis_rank, N_e - 1, D, Vt.shape[0]))
    W = Vt[:k].T  # (D, k)

    # 2. Centre and project into E-span
    OC_c = OC - OC.mean(axis=0, keepdims=True)
    LC_c = LC - LC.mean(axis=0, keepdims=True)
    OC_p, LC_p, E_p = OC_c @ W, LC_c @ W, E_c @ W

    Sigma_OC = (OC_p.T @ OC_p) / max(OC.shape[0] - 1, 1)
    Sigma_LC = (LC_p.T @ LC_p) / max(LC.shape[0] - 1, 1)
    Sigma_E  = (E_p.T  @ E_p)  / max(N_e - 1, 1)
    LHS      = Sigma_OC - Sigma_LC          # may be indefinite

    ridge_scale = float(np.diag(Sigma_E).mean()) * ridge
    Sigma_E_reg = Sigma_E + ridge_scale * np.eye(k, dtype=np.float64)

    # 3. Solve
    if _SCIPY:
        try:
            vals, vecs = sla.eigh(LHS, Sigma_E_reg)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
        except Exception as exc:
            log.warning("  scipy eigh failed (%s); using whitening fallback", exc)
            vals, vecs = _whitening_eigh(LHS, Sigma_E_reg)
    else:
        vals, vecs = _whitening_eigh(LHS, Sigma_E_reg)

    a = vecs[:, :rank]
    V = W @ a                                     # (D, rank)
    V /= np.linalg.norm(V, axis=0, keepdims=True).clip(min=1e-12)

    diag = {
        "OC_proj": float(np.trace(a.T @ Sigma_OC @ a)),
        "LC_proj": float(np.trace(a.T @ Sigma_LC @ a)),
        "E_proj":  float(np.trace(a.T @ Sigma_E  @ a)),
    }
    return {"V": V, "gamma": vals[:rank], "diag": diag}


# ── Per-domain init_scale (baked into the subspace bundle) ───────────────────

def _per_domain_init_scale(
    *,
    h_A: torch.Tensor,           # (N_F, L, D) — over-commit activations
    h_B: torch.Tensor,           # (N_F, L, D) — legit-abstain activations
    meta_A: list[dict],
    meta_B: list[dict],
    V_layers: list[torch.Tensor],  # list of L tensors (D, r)
) -> dict[str, float]:
    """Per-domain expected step-0 L_forget per example.

    Matches the actual loss formula
        L_forget_per_ex(x) = ‖V_l⊤ (h_A(x) − μ⁻(d_x))‖²
    where μ⁻(d) = mean over rows of h_B restricted to domain d.

    Returns ``{"kuq": float, "squad": float, "general": float}`` where
    ``general`` is the arithmetic mean of the two domain scales (used to
    normalise category-E loss in step 4, which has no source domain).

    Step 4 reads these values directly from the subspace bundle so it does
    not need to load the (1+ GB) activations bundle at training time.
    """
    if len(meta_A) != h_A.shape[0] or len(meta_B) != h_B.shape[0]:
        log.warning("  init_scales: meta_A/meta_B misaligned with h_A/h_B; "
                    "falling back to {kuq:1, squad:1, general:1}")
        return {"kuq": 1.0, "squad": 1.0, "general": 1.0}

    scales: dict[str, float] = {}
    for domain in ("kuq", "squad"):
        idx_A = [i for i, m in enumerate(meta_A) if m.get("dataset") == domain]
        idx_B = [i for i, m in enumerate(meta_B) if m.get("dataset") == domain]
        if not idx_A or not idx_B:
            log.warning("  init_scales: no examples for domain '%s'; defaulting to 1.0",
                        domain)
            scales[domain] = 1.0
            continue
        mu_minus_d = h_B[idx_B].mean(dim=0)        # (L, D)
        c = h_A[idx_A] - mu_minus_d.unsqueeze(0)   # (n_d, L, D)
        per_layer: list[float] = []
        for li, V_l in enumerate(V_layers):
            proj = c[:, li, :] @ V_l               # (n_d, r)
            per_layer.append(float((proj ** 2).sum(dim=-1).mean()))
        scales[domain] = max(sum(per_layer) / len(per_layer), 1e-6)

    scales["general"] = (scales["kuq"] + scales["squad"]) / 2.0
    return scales


# ── Driver ────────────────────────────────────────────────────────────────────

def run(model_key: str, rank: int, ridge: float, retain_basis_rank: int,
        overwrite: bool = False) -> Path:
    pipeline_t0 = time.time()
    act_path = cfg.activations_path(model_key)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Activations bundle not found: {act_path}. Run step 1 first."
        )

    out_path = cfg.subspace_path(model_key, rank=rank)
    if out_path.exists() and not overwrite:
        log.info("STEP 2 — BUILD SUBSPACE  (cached) %s", out_path)
        log.info("  use --overwrite to recompute. Skipping.")
        return out_path

    bundle = torch.load(act_path, map_location="cpu", weights_only=False)
    layers = bundle["layers"]

    h_A = bundle["h_A"]   # (N_F, L, D)   over-commitment
    h_B = bundle["h_B"]   # (N_F, L, D)   legitimate-abstention
    h_C = bundle["h_C"]   # (N_A, L, D)   legitimate-commitment
    h_D = bundle["h_D"]   # (N_A, L, D)   over-abstention
    h_E = bundle["h_E"]   # (N_R, L, D)   general utility

    if min(h_A.shape[0], h_C.shape[0], h_E.shape[0]) == 0:
        raise RuntimeError("Empty activation set(s). Check step 1 output.")

    c_OC = h_A - h_B          # (N_F, L, D)   over-commit contrast
    c_LC = h_C - h_D          # (N_A, L, D)   legit-commit contrast

    log.info("STEP 2 — BUILD SUBSPACE  model=%s  rank=%d  ridge=%.0e  retain_basis_rank=%d",
             model_key, rank, ridge, retain_basis_rank)
    log.info("  N_OC=%d  N_LC=%d  N_E=%d  L=%d  D=%d",
             c_OC.shape[0], c_LC.shape[0], h_E.shape[0],
             c_OC.shape[1], c_OC.shape[2])
    log.info("")
    log.info("  %-8s %10s %10s %10s %10s %10s %8s",
             "layer", "γ_1", f"γ_{rank}", "OC_proj", "LC_proj", "E_proj", "secs")
    log.info("  " + "-" * 78)

    V_layers, gamma_layers, diag_per_layer = [], [], []
    for li, layer_idx in enumerate(layers):
        layer_t0 = time.time()
        OC = c_OC[:, li, :].numpy().astype(np.float64)
        LC = c_LC[:, li, :].numpy().astype(np.float64)
        E  = h_E [:, li, :].numpy().astype(np.float64)
        out = _solve_layer(OC, LC, E, rank=rank, ridge=ridge,
                           retain_basis_rank=retain_basis_rank)
        V_layers.append(torch.tensor(out["V"], dtype=torch.float32))
        gamma_layers.append(torch.tensor(out["gamma"], dtype=torch.float32))
        diag_per_layer.append({"layer": layer_idx, **out["diag"]})

        log.info("  L%-7d %10.4f %10.4f %10.4f %10.4f %10.4f %8.2f",
                 layer_idx,
                 float(out["gamma"][0]),
                 float(out["gamma"][min(rank - 1, len(out["gamma"]) - 1)]),
                 out["diag"]["OC_proj"],
                 out["diag"]["LC_proj"],
                 out["diag"]["E_proj"],
                 time.time() - layer_t0)

    V_tensor = torch.stack(V_layers, dim=0)        # (L, D, rank)
    gamma_tensor = torch.stack(gamma_layers, dim=0)

    # Per-domain init_scale = expected step-0 value of L_forget per example, i.e.
    #     mean_l E_{x ∈ A[d]} ‖V_l⊤ (h_A(x) − μ⁻(d))‖²
    # baked into the subspace bundle so step 4 does not need to re-load the
    # activations bundle just to compute it. μ⁻(d) is the per-domain mean of
    # h_B (legitimate-abstention activations) — matches step 3's per-domain
    # pole definition.
    init_scales = _per_domain_init_scale(
        h_A=h_A.float(),
        h_B=h_B.float(),
        meta_A=bundle.get("meta_A") or [],
        meta_B=bundle.get("meta_B") or [],
        V_layers=V_layers,
    )

    out_bundle = {
        "model_key":           model_key,
        "model_id":            bundle["model_id"],
        "layers":              layers,
        "rank":                rank,
        "ridge":               ridge,
        "retain_basis_rank":   retain_basis_rank,
        "V":                   V_tensor,
        "gamma":               gamma_tensor,
        "diag":                diag_per_layer,
        "init_scales":         init_scales,
    }
    torch.save(out_bundle, out_path)
    log.info("")
    log.info("  init_scales (per-domain, baked into bundle): %s",
             {k: round(float(v), 2) for k, v in init_scales.items()})
    log.info("  Saved -> %s   V shape=%s", out_path, list(V_tensor.shape))
    log.info("STEP 2 done in %s", format_duration(time.time() - pipeline_t0))
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2: build subspace V.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--rank",  type=int, default=cfg.SUBSPACE_RANK)
    p.add_argument("--ridge", type=float, default=cfg.SUBSPACE_RIDGE)
    p.add_argument("--retain-basis-rank", type=int, default=cfg.RETAIN_BASIS_RANK)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, rank=args.rank, ridge=args.ridge,
        retain_basis_rank=args.retain_basis_rank, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
