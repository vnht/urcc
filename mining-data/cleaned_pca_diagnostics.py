#!/usr/bin/env python3
"""
Part 2 — Cleaned-PCA diagnostics.

Fits PCA on c_unsupported_clean (output of compute_cleaned_contrasts.py),
per model and selected layer. Reports explained variance ratio (EVR) at ranks
4, 8, 16, 32 and saves the PCA bases for use in the projection intervention.

Answers: is the residual unsupported-commitment signal low-rank enough to target?

Output per model:
    mining-data/activations/cleaned_pca_{model}_r8.pt
    mining-data/activations/cleaned_pca_explained_variance_{model}.csv

.pt contents:
    {
        "model":    str,
        "layers":   list[int],
        "supported_rank": int,     # rank used to clean (e.g. 8)
        "clean_pca_rank": int,     # max rank fitted (e.g. 32)
        "V_clean":  tensor[L, D, R],  # top-R PCA components of c_clean per layer
        "singular_values": tensor[L, R],
        "explained_variance_ratio": tensor[L, R],
    }

Usage:
    python3 mining-data/cleaned_pca_diagnostics.py
    python3 mining-data/cleaned_pca_diagnostics.py --model qwen_instruct
    python3 mining-data/cleaned_pca_diagnostics.py --rank 8 --evr-ranks 4 8 16 32
"""

import argparse
import csv
import logging
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

MODEL_KEYS   = ["qwen_instruct", "qwen_base", "ministral_instruct", "ministral_base"]
DEFAULT_RANK = 8
DEFAULT_EVR_RANKS = [4, 8, 16, 32]


# ─── PCA ─────────────────────────────────────────────────────────────────────

def fit_pca(X: torch.Tensor, max_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fit PCA on X (N, D). Returns:
        V    (D, max_rank) — top right singular vectors
        S    (max_rank,)   — singular values
        EVR  (max_rank,)   — explained variance ratios (cumulative not taken)
    """
    X_f = X.float()
    X_c = X_f - X_f.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
    k   = min(max_rank, S.shape[0])
    S_k = S[:k]
    Vh_k = Vh[:k]   # (k, D)
    total_var = (S ** 2).sum()
    evr = (S_k ** 2) / total_var.clamp(min=1e-12)
    return Vh_k.T, S_k, evr   # (D, k), (k,), (k,)


# ─── Per-model diagnostics ────────────────────────────────────────────────────

def run_model(model_key: str, supported_rank: int, evr_ranks: list[int]) -> list[dict]:
    in_path  = ACTIVATIONS_DIR / f"cleaned_unsupported_contrasts_{model_key}_last25_r{supported_rank}.pt"
    pt_out   = ACTIVATIONS_DIR / f"cleaned_pca_{model_key}_last25_r{supported_rank}.pt"
    csv_out  = ACTIVATIONS_DIR / f"cleaned_pca_explained_variance_{model_key}.csv"

    if not in_path.exists():
        log.error("Missing: %s — run compute_cleaned_contrasts.py first", in_path)
        return []

    bundle = torch.load(in_path, map_location="cpu", weights_only=False)
    c_clean = bundle["c_unsupported_clean"].float()   # (N, L, D)
    layers  = bundle["layers"]
    N, L, D = c_clean.shape
    max_rank = max(evr_ranks)

    log.info("")
    log.info("Model: %s  (N=%d, L=%d, D=%d)", model_key, N, L, D)

    # Header
    evr_header = "".join(f"  EVR@{r:<4}" for r in evr_ranks)
    cum_header = "".join(f"  CUM@{r:<4}" for r in evr_ranks)
    log.info("  %-8s%s%s", "Layer", evr_header, cum_header)
    log.info("  " + "-" * (8 + 10 * len(evr_ranks) * 2))

    V_all   = []
    S_all   = []
    EVR_all = []
    csv_rows: list[dict] = []

    for li, layer_idx in enumerate(layers):
        X_l = c_clean[:, li, :]   # (N, D)
        k   = min(max_rank, N - 1, D)
        V, S, evr = fit_pca(X_l, k)     # (D,k), (k,), (k,)

        evr_at = []
        cum_at = []
        for r in evr_ranks:
            r_eff = min(r, k)
            evr_at.append(float(evr[r_eff - 1]))          # individual component
            cum_at.append(float(evr[:r_eff].sum()))        # cumulative

        parts = [f"  L{layer_idx:<6}"]
        parts += [f"  {v:.5f}  " for v in evr_at]
        parts += [f"  {v:.5f}  " for v in cum_at]
        log.info("".join(parts))

        row = {"model": model_key, "layer": layer_idx}
        for r, ev, cv in zip(evr_ranks, evr_at, cum_at):
            row[f"EVR@{r}"]  = round(ev, 6)
            row[f"CUM@{r}"]  = round(cv, 6)
        csv_rows.append(row)

        # Pad to max_rank if svd returned fewer components
        pad = max_rank - k
        if pad > 0:
            V   = torch.cat([V,   torch.zeros(D, pad)], dim=1)
            S   = torch.cat([S,   torch.zeros(pad)])
            evr = torch.cat([evr, torch.zeros(pad)])

        V_all.append(V[:, :max_rank])
        S_all.append(S[:max_rank])
        EVR_all.append(evr[:max_rank])

    V_tensor   = torch.stack(V_all,   dim=0)   # (L, D, R)
    S_tensor   = torch.stack(S_all,   dim=0)   # (L, R)
    EVR_tensor = torch.stack(EVR_all, dim=0)   # (L, R)

    # Save .pt with PCA bases (needed by projection_intervention.py)
    pt_bundle = {
        "model":                    bundle["model"],
        "layers":                   layers,
        "supported_rank":           supported_rank,
        "clean_pca_rank":           max_rank,
        "V_clean":                  V_tensor,
        "singular_values":          S_tensor,
        "explained_variance_ratio": EVR_tensor,
    }
    torch.save(pt_bundle, pt_out)
    log.info("  Saved bases → %s", pt_out.name)

    # Save CSV
    fieldnames = ["model", "layer"] + [f"EVR@{r}" for r in evr_ranks] + [f"CUM@{r}" for r in evr_ranks]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("  Saved EVR   → %s", csv_out.name)

    return csv_rows


# ─── Cross-model summary ──────────────────────────────────────────────────────

def print_summary(all_rows: dict[str, list[dict]], evr_ranks: list[int]) -> None:
    log.info("")
    log.info("=" * 70)
    log.info("Summary — mean cumulative EVR across layers, per model")
    log.info("=" * 70)

    header = f"  {'Model':<24}" + "".join(f"  CUM@{r:<4}" for r in evr_ranks)
    log.info(header)
    log.info("  " + "-" * (24 + 10 * len(evr_ranks)))

    for model_key, rows in sorted(all_rows.items()):
        if not rows:
            continue
        parts = [f"  {model_key:<24}"]
        for r in evr_ranks:
            vals = [row[f"CUM@{r}"] for row in rows if f"CUM@{r}" in row]
            mean = sum(vals) / len(vals) if vals else float("nan")
            parts.append(f"  {mean:.5f}  ")
        log.info("".join(parts))

    log.info("")
    log.info("Interpretation (cumulative EVR):")
    log.info("  < 0.20 at r=8  → signal is diffuse; rank-8 subspace covers little variance")
    log.info("  0.20–0.50 at r=8  → moderately low-rank; good target for projection unlearning")
    log.info("  > 0.50 at r=8  → strongly low-rank; unlearning will be precise and efficient")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cleaned-PCA diagnostics")
    p.add_argument("--model", choices=MODEL_KEYS, nargs="*", default=None)
    p.add_argument("--rank",  type=int, default=DEFAULT_RANK,
                   help="supported_rank used when cleaning (default: 8)")
    p.add_argument("--evr-ranks", type=int, nargs="+", default=DEFAULT_EVR_RANKS,
                   metavar="R", help="Ranks at which to report EVR (default: 4 8 16 32)")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    keys      = args.model or MODEL_KEYS
    evr_ranks = sorted(set(args.evr_ranks))

    log.info("Cleaned-PCA diagnostics")
    log.info("  Models     : %s", keys)
    log.info("  Clean rank : %d", args.rank)
    log.info("  EVR ranks  : %s", evr_ranks)

    all_rows: dict[str, list[dict]] = {}
    for key in keys:
        rows = run_model(key, args.rank, evr_ranks)
        all_rows[key] = rows

    print_summary(all_rows, evr_ranks)
    log.info("Done → %s", ACTIVATIONS_DIR)


if __name__ == "__main__":
    main()
