#!/usr/bin/env python3
"""Diagnostic — is the discriminative subspace V behaviorally usable per domain?

For each answerability domain (kuq | squad) this reports, from the saved
activations + subspace + anchors (no model load, runs anywhere the .pt files
are):

  pole_sep_in_V   ‖Vᵀ(μ⁻−μ⁺)‖ / ‖μ⁻−μ⁺‖
      How much of the abstain↔commit pole separation survives the projection
      into V. This is the direction the forget loss optimizes along; if μ⁻≈μ⁺
      *inside V* the loss can converge while behavior never changes. (Random
      floor ≈ sqrt(rank/D).)
  frac_axis_in_V  ‖V Vᵀ(mean_A−mean_B)‖ / ‖mean_A−mean_B‖
      Fraction of the over-commit→abstain mean shift captured by V.
  fisher_full     ‖mean_A−mean_B‖ / sqrt(mean within-class var)   [FULL space]
      Whether the commit/abstain signal even *exists* in the raw activations,
      independent of V. High here but low pole_sep ⇒ V is the problem, not the
      data.
  E_overlap       vᵀΣ_E v   and  frac of (μ⁺−μ⁻) in top-k UltraChat PCs
      How much the abstain↔commit direction looks like general-utility (set E)
      answering. High overlap ⇒ the Σ_E whitening in step 2 rotates V away from
      that direction to protect utility (the kuq failure mode).

Run
---
    python3 step2_build_subspace/diagnose_subspace.py --model gptoss_instruct
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import log

DOMAINS = ("kuq", "squad")


def _didx(meta, d):
    return [i for i, m in enumerate(meta) if (m or {}).get("dataset") == d]


def run(model_key: str, top_pcs: int = 64) -> None:
    act = torch.load(cfg.activations_path(model_key), map_location="cpu", weights_only=False)
    sub = torch.load(cfg.subspace_path(model_key, rank=cfg.SUBSPACE_RANK),
                     map_location="cpu", weights_only=False)
    anc = torch.load(cfg.anchors_path(model_key), map_location="cpu", weights_only=False)

    V = sub["V"].float()                      # (L, D, r)
    layers = sub["layers"]; L = len(layers); D = V.shape[1]; r = V.shape[2]
    hA, hB, hE = act["h_A"].float(), act["h_B"].float(), act["h_E"].float()
    mA = act["meta_A"]
    rand_floor = (r / D) ** 0.5

    log.info("model=%s  layers=%s  rank=%d  random pole-sep floor≈%.3f",
             model_key, layers, r, rand_floor)
    log.info("init_scales: %s", {k: round(float(v), 1) for k, v in sub.get("init_scales", {}).items()})
    log.info("")
    log.info("%-6s %12s %12s %12s %14s %16s",
             "domain", "pole_sep_V", "frac_axis_V", "fisher_full", "E_rayleigh(k/s)", f"E_in_top{top_pcs}PC")
    log.info("-" * 80)

    rayleigh = {}
    for d in DOMAINS:
        ia = _didx(mA, d)
        if not ia:
            log.warning("  no rows for domain %s", d); continue
        pole_sep, frac_axis, fisher, ray, epc = [], [], [], [], []
        for li in range(L):
            Vl = V[li]
            mm = anc["mu_minus_per"][d][li]; mp = anc["mu_plus_per"][d][li]
            dpole = mm - mp
            pole_sep.append(((Vl.t() @ dpole).norm() / (dpole.norm() + 1e-9)).item())

            A = hA[ia, li]; B = hB[ia, li]
            dAB = A.mean(0) - B.mean(0)
            frac_axis.append(((Vl @ (Vl.t() @ dAB)).norm() / (dAB.norm() + 1e-9)).item())
            scat = (((A.var(0) + B.var(0)) / 2).sum().sqrt())
            fisher.append((dAB.norm() / (scat + 1e-9)).item())

            Ec = hE[:, li] - hE[:, li].mean(0, keepdim=True)
            SE = (Ec.t() @ Ec) / (Ec.shape[0] - 1)
            v = dpole / (dpole.norm() + 1e-9)
            ray.append((v @ (SE @ v)).item())
            U, S, Vt = torch.linalg.svd(Ec, full_matrices=False)
            P = Vt[:top_pcs].t()
            epc.append(((P @ (P.t() @ dpole)).norm() / (dpole.norm() + 1e-9)).item())
        rayleigh[d] = st.mean(ray)
        log.info("%-6s %12.3f %12.3f %12.3f %14.0f %16.3f",
                 d, st.mean(pole_sep), st.mean(frac_axis), st.mean(fisher),
                 st.mean(ray), st.mean(epc))

    if "kuq" in rayleigh and "squad" in rayleigh:
        log.info("")
        log.info("E_rayleigh ratio kuq/squad = %.2f  (>1 ⇒ kuq abstain dir looks more "
                 "like general-utility answering ⇒ Σ_E whitening suppresses it in V)",
                 rayleigh["kuq"] / max(rayleigh["squad"], 1e-9))
    log.info("")
    log.info("READ: low pole_sep_V with HIGH fisher_full ⇒ signal exists but V drops it "
             "(V-construction problem → try larger SUBSPACE_RIDGE). low pole_sep_V AND "
             "low fisher_full ⇒ the signal isn't at these answer tokens (→ CoT/option-B "
             "extraction). High E_rayleigh/E_in_topPC for kuq ⇒ utility-whitening is the "
             "culprit.")


def ridge_sweep(model_key: str, ridges: list[float]) -> None:
    """Rebuild V (pooled, exactly as step 2 does) at several ridge values and
    report per-domain pole_sep_in_V, so the Σ_E-whitening relaxation can be
    tuned before committing a training run. Higher ridge ⇒ Σ_E_reg → identity
    ⇒ V keeps high-utility-overlap directions (recovers the kuq axis) at the
    cost of weaker general-utility protection (guarded by the retain loss)."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_subspace import _solve_layer

    act = torch.load(cfg.activations_path(model_key), map_location="cpu", weights_only=False)
    anc = torch.load(cfg.anchors_path(model_key), map_location="cpu", weights_only=False)
    layers = act["layers"]; L = len(layers)
    hA, hB, hC, hD, hE = [act[k].float() for k in ("h_A", "h_B", "h_C", "h_D", "h_E")]
    c_OC, c_LC = hA - hB, hC - hD

    log.info("RIDGE SWEEP (current pipeline ridge = %.0e). squad worked at pole_sep≈0.25–0.28;"
             " target kuq into that range.", cfg.SUBSPACE_RIDGE)
    log.info("%12s %10s %10s", "ridge", "kuq", "squad")
    log.info("-" * 34)
    for rg in ridges:
        Vs = []
        for li in range(L):
            out = _solve_layer(c_OC[:, li].numpy().astype(np.float64),
                               c_LC[:, li].numpy().astype(np.float64),
                               hE[:, li].numpy().astype(np.float64),
                               rank=cfg.SUBSPACE_RANK, ridge=rg,
                               retain_basis_rank=cfg.RETAIN_BASIS_RANK)
            Vs.append(torch.tensor(out["V"], dtype=torch.float32))
        row = {}
        for d in DOMAINS:
            ps = []
            for li in range(L):
                dpole = anc["mu_minus_per"][d][li] - anc["mu_plus_per"][d][li]
                ps.append(((Vs[li].t() @ dpole).norm() / (dpole.norm() + 1e-9)).item())
            row[d] = st.mean(ps)
        log.info("%12.0e %10.3f %10.3f", rg, row["kuq"], row["squad"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose per-domain usability of subspace V.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--top-pcs", type=int, default=64)
    p.add_argument("--ridge-sweep", action="store_true",
                   help="Rebuild V at several ridge values and report pole_sep per "
                        "domain (to tune SUBSPACE_RIDGE before training).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ridge_sweep:
        ridge_sweep(args.model, [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0])
    else:
        run(args.model, top_pcs=args.top_pcs)


if __name__ == "__main__":
    main()
