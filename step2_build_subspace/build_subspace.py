#!/usr/bin/env python3
"""Step 2 — Build the per-domain discriminative commitment subspaces V(d).

For each domain d ∈ {kuq, squad} and each late layer l, solve the generalised
eigenproblem in the general-utility span using only domain-d contrasts:

    (Σ_OC(d) - Σ_LC(d)) v = γ · Σ_E v

where
    Σ_OC(d) = cov(c_OC | dataset==d)   c_OC = h_A − h_B
    Σ_LC(d) = cov(c_LC | dataset==d)   c_LC = h_C − h_D
    Σ_E     = cov(h_E)                 (general utility — domain-shared)

V_l(d) ∈ ℝ^{D × r} are the top-r generalised eigenvectors for domain d. Large
positive γ ⇒ direction along which over-commit varies more than legitimate-
commit (after subtracting the shared abstain-mode baseline), normalised
against general utility.

Why per-domain V (with CE retain)
---------------------------------
KUQ (no-context) and SQuAD (with-context) prompts sit in different regions of
late-layer hidden-state space and have different commitment-vs-abstention
decision directions. A shared V is a compromise basis dominated by whichever
domain has the stronger contrast (KUQ at OC/LC≈14× vs SQuAD at OC/LC≈5×).
Empirically V_kuq and V_squad share zero highly-aligned dimensions and
≈half their basis is near-orthogonal — they are genuinely different bases.

V is used only by L_forget in step 4 (retain is CE on response tokens, no V).
There is no V/retain interaction, so per-domain V here is a pure isolation
of the two forget regions.

Reads:  step1_extract_activations/data/activations_<model>.pt
Writes: step2_build_subspace/data/subspace_<model>_r<rank>.pt with keys:
    "model_key", "layers", "rank", "ridge", "retain_basis_rank",
    "datasets":    ["kuq", "squad"]
    "V_per":       dict[str, tensor[L, D, r]]   per-domain subspaces (training target)
    "gamma_per":   dict[str, tensor[L, r]]      per-domain eigenvalues
    "diag_per":    dict[str, list]              per-domain layer diagnostics
    "V":           tensor[L, D, r]              grand-mixed (legacy / ablation)
    "gamma":       tensor[L, r]                 grand-mixed (legacy)
    "diag":        list                          grand-mixed (legacy)

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


# ── Driver ────────────────────────────────────────────────────────────────────

DOMAINS = ("kuq", "squad")


def _datasets_for(bundle: dict, hidden_key: str) -> list[str]:
    """Per-row dataset labels aligned to bundle[hidden_key]; falls back to
    the sibling meta when set was extracted from the same row pool (B↔A, D↔C).
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


def _build_one_subspace(
    *, c_OC: torch.Tensor, c_LC: torch.Tensor, h_E: torch.Tensor,
    layers: list[int], rank: int, ridge: float, retain_basis_rank: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Solve the per-layer eigenproblem for one (filtered) contrast set."""
    log.info("")
    log.info("  [%s]  N_OC=%d  N_LC=%d  N_E=%d", label,
             c_OC.shape[0], c_LC.shape[0], h_E.shape[0])
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
    return (
        torch.stack(V_layers, dim=0),       # (L, D, rank)
        torch.stack(gamma_layers, dim=0),   # (L, rank)
        diag_per_layer,
    )


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

    c_OC = h_A - h_B
    c_LC = h_C - h_D

    ds_OC = _datasets_for(bundle, "h_A")
    ds_LC = _datasets_for(bundle, "h_C")

    log.info("STEP 2 — BUILD SUBSPACE  model=%s  rank=%d  ridge=%.0e  retain_basis_rank=%d",
             model_key, rank, ridge, retain_basis_rank)
    log.info("  total: N_OC=%d  N_LC=%d  N_E=%d  L=%d  D=%d",
             c_OC.shape[0], c_LC.shape[0], h_E.shape[0],
             c_OC.shape[1], c_OC.shape[2])
    log.info("  domains: OC=%s   LC=%s",
             {d: sum(1 for x in ds_OC if x == d) for d in DOMAINS},
             {d: sum(1 for x in ds_LC if x == d) for d in DOMAINS})

    # Per-domain subspaces (the training targets used by L_forget)
    V_per: dict[str, torch.Tensor] = {}
    gamma_per: dict[str, torch.Tensor] = {}
    diag_per: dict[str, list[dict]] = {}
    for d in DOMAINS:
        oc_idx = [i for i, x in enumerate(ds_OC) if x == d]
        lc_idx = [i for i, x in enumerate(ds_LC) if x == d]
        if not oc_idx or not lc_idx:
            raise RuntimeError(f"No rows for dataset '{d}'. Check step 1 output.")
        V_d, gamma_d, diag_d = _build_one_subspace(
            c_OC=c_OC[oc_idx], c_LC=c_LC[lc_idx], h_E=h_E,
            layers=layers, rank=rank, ridge=ridge,
            retain_basis_rank=retain_basis_rank, label=f"V_{d}",
        )
        V_per[d] = V_d
        gamma_per[d] = gamma_d
        diag_per[d] = diag_d

    # Grand V (legacy / ablation only)
    V_grand, gamma_grand, diag_grand = _build_one_subspace(
        c_OC=c_OC, c_LC=c_LC, h_E=h_E, layers=layers,
        rank=rank, ridge=ridge, retain_basis_rank=retain_basis_rank,
        label="V_grand (legacy / ablation)",
    )

    out_bundle = {
        "model_key":           model_key,
        "model_id":            bundle["model_id"],
        "layers":              layers,
        "rank":                rank,
        "ridge":               ridge,
        "retain_basis_rank":   retain_basis_rank,
        "datasets":            list(DOMAINS),
        "V_per":               V_per,
        "gamma_per":           gamma_per,
        "diag_per":            diag_per,
        "V":                   V_grand,
        "gamma":               gamma_grand,
        "diag":                diag_grand,
    }
    torch.save(out_bundle, out_path)
    log.info("")
    log.info("  Saved -> %s", out_path)
    for d in DOMAINS:
        log.info("    V_%s shape=%s", d, list(V_per[d].shape))
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
