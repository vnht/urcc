#!/usr/bin/env python3
"""Step 3 — Build the two UOC poles along V.

μ_l⁻  (legitimate-abstention pole)
    Mean late-layer hidden state over the answer-token window of templated
    legitimate-abstention completions on unanswerable prompts.
    Drawn from set B in step 1.

μ_l⁺  (legitimate-commitment pole)
    Mean late-layer hidden state over the answer-token window of gold answers
    on answerable prompts.
    Drawn from set C in step 1.

Both anchors are fixed, layer-aligned constants — no gradient flows through
them. They are the per-pole targets the UOC loss anchors each example to:

    D_F   (forget, category A)        → μ_l⁻
    D_R_A (retain-answerable, cat. C) → μ_l⁺
    D_R_G (retain-general, cat. E)    → frozen reference (computed at
                                        training time, not here)

Reads:  step1_extract_activations/data/activations_<model>.pt
Writes: step3_build_anchors/data/anchors_<model>.pt with keys:
    "model_key", "layers", "k_answer_tokens",
    "mu_minus":   tensor[L, D]   abstain pole
    "mu_plus":    tensor[L, D]   commit pole
    "n_minus":    int            examples used for μ⁻
    "n_plus":     int            examples used for μ⁺

Run
---
    python step3_build_anchors/build_anchors.py --model qwen_instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import format_duration, log


def run(model_key: str, overwrite: bool = False) -> Path:
    pipeline_t0 = time.time()
    act_path = cfg.activations_path(model_key)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Activations bundle not found: {act_path}. Run step 1 first."
        )

    out_path = cfg.anchors_path(model_key)
    if out_path.exists() and not overwrite:
        log.info("STEP 3 — BUILD ANCHORS  (cached) %s", out_path)
        log.info("  use --overwrite to recompute. Skipping.")
        return out_path

    bundle = torch.load(act_path, map_location="cpu", weights_only=False)

    h_minus = bundle["h_B"]   # (N_F, L, D) — legitimate-abstention pole μ⁻
    h_plus  = bundle["h_C"]   # (N_A, L, D) — legitimate-commitment pole μ⁺

    if h_minus.shape[0] == 0:
        raise RuntimeError("No abstain examples available — check step 1 output.")
    if h_plus.shape[0] == 0:
        raise RuntimeError("No answerable correct examples — check step 1 output.")

    mu_minus = h_minus.mean(dim=0)         # (L, D)
    mu_plus  = h_plus.mean(dim=0)          # (L, D)

    out = {
        "model_key":       model_key,
        "model_id":        bundle["model_id"],
        "layers":          bundle["layers"],
        "k_answer_tokens": bundle["k_answer_tokens"],
        "mu_minus":        mu_minus.float(),
        "mu_plus":         mu_plus.float(),
        "n_minus":         int(h_minus.shape[0]),
        "n_plus":          int(h_plus.shape[0]),
    }

    torch.save(out, out_path)

    log.info("STEP 3 — BUILD ANCHORS  model=%s", model_key)
    log.info("  μ⁻ from %d abstain examples,  μ⁺ from %d answerable examples",
             out["n_minus"], out["n_plus"])
    log.info("  Layer-wise norms:")
    for li, l in enumerate(out["layers"]):
        log.info("    L%-3d   ||μ⁻||=%.3f   ||μ⁺||=%.3f   ||μ⁻−μ⁺||=%.3f",
                 l, float(mu_minus[li].norm()), float(mu_plus[li].norm()),
                 float((mu_minus[li] - mu_plus[li]).norm()))
    log.info("  Saved -> %s", out_path)
    log.info("STEP 3 done in %s", format_duration(time.time() - pipeline_t0))
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: build anchors μ⁻ and μ⁺.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
