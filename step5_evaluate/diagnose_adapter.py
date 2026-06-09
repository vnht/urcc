#!/usr/bin/env python3
"""Cheap inference-time diagnostic for a degenerate UOC adapter (no retraining).

Motivation: the trained ministral-14B adapter produces repetition-collapse
gibberish (proj_norm driven 24 -> 1.37 — over-suppression). Before spending GPU
on a retrain, we validate *whether* gentler intervention restores fluency by
manipulating the EXISTING adapter at inference time:

  1. SCALE SWEEP — multiply the LoRA contribution by a factor in [0, 1] (0 = base
     model, 1 = trained). If fluency returns below 1.0, over-suppression is
     confirmed and a gentler retrain (higher proj-norm floor) will work; the
     factor where it recovers tells us roughly how much gentler.
  2. LAST-LAYER ABLATION — zero the adapter on the final K layers (which feed
     lm_head) at full strength. If that alone restores fluency, the top-layer
     edits are the culprit and the retrain should skip the last layers.

Generation is greedy/deterministic, so a handful of prompts per setting is
enough. This is minutes of GPU, not a retrain.

Usage:
    python3 step5_evaluate/diagnose_adapter.py \
        --run-dir step4_train/data/runs/ministral14b_instruct_uoc_r32_lam2_ep3_lr3e-05
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))                       # config, _common (repo root)
sys.path.insert(0, str(_REPO / "step5_evaluate"))    # evaluate
import config as cfg  # noqa: E402
from _common import generate_greedy, log  # noqa: E402
from evaluate import _load_adapter_model  # noqa: E402  (loads base + PEFT adapter)

# Fixed probe prompts: closed-book factual (answerable) + a couple unanswerable.
PROBES = [
    "What is the capital of France?",
    "Why do veins appear blue?",
    "Where did fortune cookies originate?",
    "What is the spiciest part of a chili pepper?",
    "On what date was the Declaration of Independence officially signed?",
    "What U.S. state produces the most peaches?",
    "Who is the bell-ringer of Notre Dame?",
    "What happens if you eat watermelon seeds?",
]


def _prompt(q: str) -> str:
    return f"Answer concisely in a sentence.\n\nQuestion:\n{q}\n\nAnswer:"


def _is_degenerate(c: str) -> bool:
    c = c or ""
    if not c.strip():
        return True
    if re.search(r"(\*\s*){6,}", c) or re.search(r"(—\s*){4,}", c):
        return True
    toks = c.split()
    return len(toks) >= 10 and len(set(toks)) <= 3


def _lora_modules(model):
    """Yield (name, module) for every PEFT LoRA layer (has a `scaling` dict)."""
    for name, mod in model.named_modules():
        if isinstance(getattr(mod, "scaling", None), dict) and hasattr(mod, "lora_A"):
            yield name, mod


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _capture_base_scaling(model) -> dict:
    base = {}
    for name, mod in _lora_modules(model):
        for ad, val in mod.scaling.items():
            base[(name, ad)] = float(val)
    return base


def _apply_scaling(model, base: dict, factor: float, zero_layers: set[int] | None = None):
    """Set each LoRA layer's scaling to base*factor, or 0 for zero_layers."""
    zero_layers = zero_layers or set()
    for name, mod in _lora_modules(model):
        m = _LAYER_RE.search(name)
        layer = int(m.group(1)) if m else -1
        for ad in mod.scaling:
            mod.scaling[ad] = 0.0 if layer in zero_layers else base[(name, ad)] * factor


def _run_setting(model, tokenizer, model_key, label: str, max_new_tokens: int):
    deg = 0
    print(f"\n===== {label} =====")
    for q in PROBES:
        out = generate_greedy(model, tokenizer, model_key, _prompt(q),
                              max_new_tokens=max_new_tokens)
        bad = _is_degenerate(out)
        deg += bad
        flag = "DEGENERATE" if bad else "ok"
        print(f"  [{flag:10s}] Q: {q[:45]:45s} -> {out.strip()[:90]!r}")
    print(f"  --> {deg}/{len(PROBES)} degenerate")
    return deg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--ablate-last", type=int, nargs="+", default=[1, 2],
                   help="Also test full-strength with the last K layers ablated.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    args = p.parse_args()

    model, tokenizer, model_key = _load_adapter_model(args.run_dir.resolve())
    mnt = args.max_new_tokens or cfg.max_new_tokens_for(model_key)
    log.info("  model_key=%s  max_new_tokens=%d", model_key, mnt)

    base = _capture_base_scaling(model)
    layers = sorted({int(_LAYER_RE.search(n).group(1))
                     for n, _ in _lora_modules(model) if _LAYER_RE.search(n)})
    log.info("  adapter touches %d LoRA modules across layers %s",
             len(base), layers)
    if not base:
        log.warning("  No PEFT LoRA layers found — was the adapter merged? "
                    "This diagnostic needs an unmerged PeftModel.")
        return
    max_layer = max(layers)

    results = {}
    # 1) scale sweep
    for f in args.scales:
        _apply_scaling(model, base, f)
        results[f"scale={f:.2f}"] = _run_setting(
            model, tokenizer, model_key, f"SCALE x{f:.2f}", mnt)

    # 2) last-K-layer ablation at full strength
    for k in args.ablate_last:
        zero = {max_layer - i for i in range(k)}
        _apply_scaling(model, base, 1.0, zero_layers=zero)
        results[f"full,ablate_last_{k}"] = _run_setting(
            model, tokenizer, model_key, f"FULL, ablate last {k} layers {sorted(zero)}", mnt)

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"  {k:24s} degenerate {v}/{len(PROBES)}")
    print("\nReading: scale=0.00 should be clean (base). If a scale<1.0 is clean "
          "but 1.0 is not, over-suppression is confirmed. If 'ablate_last_*' is "
          "clean at full strength, the final-layer edits are the culprit.")


if __name__ == "__main__":
    main()
