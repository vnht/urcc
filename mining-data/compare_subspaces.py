#!/usr/bin/env python3
"""
Compare cleaned vs raw retain-normalised subspaces.

Loads:
    activations/retain_normalised_subspace_<model>_last25_r<rank>.pt   (cleaned)
    activations/raw_subspace_<model>_last25_r<rank>.pt                  (raw)

Reports per layer:
    1. Top-rank eigenvalue magnitudes and decay shape
    2. Cumulative variance captured at rank 8, 16, 32 (uses raw bundle's
       full eigenvalues if available; cleaned bundle only stores top-rank)
    3. Commitment / retain projection trace and C/R ratio
    4. Subspace alignment: principal angles between V_clean and V_raw

Writes a markdown report to:
    activations/subspace_comparison_<model>_last25_r<rank>.md

Usage:
    python3 mining-data/compare_subspaces.py --model qwen_instruct --rank 32
    python3 mining-data/compare_subspaces.py --model qwen_instruct --rank 8
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

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


def _principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosines of principal angles between subspaces spanned by columns of A, B.
    Both expected to have orthonormal columns.
    """
    Q_a, _ = np.linalg.qr(A)
    Q_b, _ = np.linalg.qr(B)
    s = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def _format_eigs(eigs: np.ndarray, n: int = 8) -> str:
    return "  ".join(f"{e:8.3f}" for e in eigs[:n])


def _cumulative_capture(eigs: np.ndarray, ks: list[int]) -> dict:
    total = float(eigs.clip(min=0).sum())
    if total <= 0:
        return {k: 0.0 for k in ks}
    cum = eigs.clip(min=0).cumsum() / total
    return {k: float(cum[min(k - 1, len(cum) - 1)]) for k in ks}


def compare(model_key: str, rank: int) -> str:
    cleaned_path = ACTIVATIONS_DIR / f"retain_normalised_subspace_{model_key}_last25_r{rank}.pt"
    raw_path     = ACTIVATIONS_DIR / f"raw_subspace_{model_key}_last25_r{rank}.pt"

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing: {cleaned_path.name}")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing: {raw_path.name}\n"
            f"Run: python3 mining-data/raw_subspace.py --model {model_key} --rank {rank}"
        )

    cleaned = torch.load(cleaned_path, map_location="cpu", weights_only=False)
    raw     = torch.load(raw_path,     map_location="cpu", weights_only=False)

    layers = cleaned["layers"]
    assert layers == raw["layers"], (
        f"Layer mismatch: cleaned={layers} raw={raw['layers']}"
    )

    V_clean = cleaned["V_retain_normalised"].numpy()    # (L, D, rank)
    V_raw   = raw["V_retain_normalised"].numpy()
    g_clean = cleaned["generalized_eigenvalues"].numpy()  # (L, rank)
    g_raw   = raw["generalized_eigenvalues"].numpy()
    cp_clean = cleaned["commitment_projection"].numpy()
    cp_raw   = raw["commitment_projection"].numpy()
    rp_clean = cleaned["retain_projection"].numpy()
    rp_raw   = raw["retain_projection"].numpy()
    full_raw = raw.get("full_eigenvalues", None)

    out_lines: list[str] = []
    p = lambda s="": out_lines.append(s)

    p(f"# Subspace comparison — {model_key} (last-25%, rank={rank})")
    p()
    p("Cleaned: variance from `c_unsupported_clean` (post supported-PCA subtraction).")
    p("Raw:     variance from `c_unsupported` (no cleaning step).")
    p()

    # ── 1. Per-layer top eigenvalues ────────────────────────────────────
    p("## 1. Top eigenvalues per layer (γ = vᵀΣ_C v / vᵀΣ_R v)")
    p()
    p("### Cleaned")
    p()
    p("```")
    for i, l in enumerate(layers):
        p(f"L{l}: {_format_eigs(g_clean[i])}")
    p("```")
    p()
    p("### Raw")
    p()
    p("```")
    for i, l in enumerate(layers):
        p(f"L{l}: {_format_eigs(g_raw[i])}")
    p("```")
    p()

    # ── 2. Cumulative variance capture (only available from full spectrum) ──
    if full_raw is not None:
        p("## 2. Cumulative commitment variance captured (raw)")
        p()
        p("Computed from the full generalised-eigenvalue spectrum, projecting"
          " into the retain span (k_eff dims) per layer. Cleaned bundle stores"
          " only top-rank eigenvalues so a full comparison is not possible —"
          " these numbers come from the raw subspace.")
        p()
        p("| Layer | rank-1 | rank-8 | rank-16 | rank-32 | rank-64 | rank-128 |")
        p("|-------|-------:|-------:|--------:|--------:|--------:|---------:|")
        for i, l in enumerate(layers):
            full = full_raw[i].numpy()
            cum  = _cumulative_capture(full, [1, 8, 16, 32, 64, 128])
            p(f"| L{l} | {cum[1]*100:5.1f}% | {cum[8]*100:5.1f}% | "
              f"{cum[16]*100:5.1f}% | {cum[32]*100:5.1f}% | "
              f"{cum[64]*100:5.1f}% | {cum[128]*100:5.1f}% |")
        p()

    # ── 3. C / R projection traces ──────────────────────────────────────
    p("## 3. Commitment vs retain projection on top-rank subspace")
    p()
    p("`C_proj = tr(VᵀΣ_C V)` — commitment energy captured by the subspace")
    p("`R_proj = tr(VᵀΣ_R V)` — retain energy captured (lower is better)")
    p("`C/R`   = signal-to-interference ratio")
    p()
    p("| Layer | C_proj (clean) | C_proj (raw) | R_proj (clean) | R_proj (raw) | C/R clean | C/R raw |")
    p("|-------|---------------:|-------------:|---------------:|-------------:|----------:|--------:|")
    for i, l in enumerate(layers):
        ratio_c = cp_clean[i] / max(rp_clean[i], 1e-12)
        ratio_r = cp_raw[i]   / max(rp_raw[i],   1e-12)
        p(f"| L{l} | {cp_clean[i]:13.3f} | {cp_raw[i]:11.3f} | "
          f"{rp_clean[i]:13.3f} | {rp_raw[i]:11.3f} | "
          f"{ratio_c:8.2f} | {ratio_r:6.2f} |")
    p()
    p(f"**Mean C_proj (clean) = {cp_clean.mean():.3f}**, "
      f"**(raw) = {cp_raw.mean():.3f}**")
    p(f"Mean R_proj (clean) = {rp_clean.mean():.3f}, (raw) = {rp_raw.mean():.3f}")
    p(f"Mean C/R   (clean) = {cp_clean.mean()/max(rp_clean.mean(),1e-12):.2f}, "
      f"(raw) = {cp_raw.mean()/max(rp_raw.mean(),1e-12):.2f}")
    p()

    # ── 4. Subspace alignment ───────────────────────────────────────────
    p("## 4. Subspace alignment between cleaned and raw")
    p()
    p("Principal angles (cos θ) between the rank-`r` subspaces spanned by"
      " V_clean and V_raw at each layer. Values close to 1 = subspaces are"
      " nearly identical; values close to 0 = orthogonal.")
    p()
    p(f"| Layer | mean cos θ | min cos θ | max cos θ | top-{min(rank,8)} cos θ |")
    p(f"|-------|-----------:|----------:|----------:|{'-'*22}:|")
    for i, l in enumerate(layers):
        cosines = _principal_angles(V_clean[i], V_raw[i])
        head = "  ".join(f"{c:.3f}" for c in cosines[:min(rank, 8)])
        p(f"| L{l} | {cosines.mean():.3f} | {cosines.min():.3f} | "
          f"{cosines.max():.3f} | {head} |")
    p()

    # ── 5. Interpretation hints ─────────────────────────────────────────
    p("## 5. Interpretation hints")
    p()
    sharper_decay = (g_raw[:, 0].mean() / max(g_raw[:, -1].mean(), 1e-12)) > \
                    (g_clean[:, 0].mean() / max(g_clean[:, -1].mean(), 1e-12))
    p(f"- Eigenvalue decay sharper in **{'raw' if sharper_decay else 'cleaned'}** "
      f"(γ_1/γ_{rank} ratio: clean={g_clean[:,0].mean()/max(g_clean[:,-1].mean(),1e-12):.2f}, "
      f"raw={g_raw[:,0].mean()/max(g_raw[:,-1].mean(),1e-12):.2f})")
    p(f"- Commitment-vs-retain ratio: "
      f"raw is **{'higher' if cp_raw.mean()/rp_raw.mean() > cp_clean.mean()/rp_clean.mean() else 'lower'}** "
      f"({cp_raw.mean()/rp_raw.mean():.2f} vs {cp_clean.mean()/rp_clean.mean():.2f})")
    avg_overlap = float(np.mean([
        _principal_angles(V_clean[i], V_raw[i]).mean()
        for i in range(len(layers))
    ]))
    p(f"- Mean subspace overlap: {avg_overlap:.3f}  (1.0 = identical, 0.0 = orthogonal)")
    p()
    p("**Reading the diagnostic:**")
    p()
    p("- If **raw eigenvalues decay much more sharply** AND **C/R ratio is much"
      " higher** for raw than cleaned → the cleaning step was destroying signal.")
    p("- If **eigenvalues are similar** AND **C/R is similar** AND **subspace"
      " overlap is low (<0.7)** → cleaning isn't hurting magnitude but is"
      " pointing at different directions; may not matter much.")
    p("- If **eigenvalues are similar** AND **subspace overlap is high (>0.95)**"
      " → cleaning is essentially a no-op; the supported-answer subspace doesn't"
      " overlap with the unsupported-contrast subspace much in the first place.")
    p("- If **raw also has slow decay** with no clean elbow → commitment is"
      " genuinely high-rank; URC's low-rank framing is the wrong primitive.")

    return "\n".join(out_lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare cleaned vs raw subspaces.")
    p.add_argument("--model", choices=MODEL_KEYS, required=True)
    p.add_argument("--rank",  type=int, default=8)
    args = p.parse_args()

    report = compare(args.model, args.rank)

    out_path = ACTIVATIONS_DIR / \
               f"subspace_comparison_{args.model}_last25_r{args.rank}.md"
    out_path.write_text(report)

    print(report)
    print()
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
