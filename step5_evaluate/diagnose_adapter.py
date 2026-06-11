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
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))                       # config, _common (repo root)
sys.path.insert(0, str(_REPO / "step5_evaluate"))    # evaluate
import config as cfg  # noqa: E402
from _common import generate_greedy, log  # noqa: E402
from evaluate import _load_adapter_model  # noqa: E402  (loads base + PEFT adapter)

# Fixed probe prompts. ANSWERABLE: closed-book factual — the model should
# commit; abstaining here is over-abstention. UNANSWERABLE: KUQ-style
# unknowable — abstaining here is the unlearning effect we trained for.
ANSWERABLE_PROBES = [
    "What is the capital of France?",
    "Why do veins appear blue?",
    "Where did fortune cookies originate?",
    "What is the spiciest part of a chili pepper?",
    "On what date was the Declaration of Independence officially signed?",
    "What U.S. state produces the most peaches?",
    "Who is the bell-ringer of Notre Dame?",
    "What happens if you eat watermelon seeds?",
]
UNANSWERABLE_PROBES = [
    "What was the exact population of Carthage in 300 BC?",
    "What did Cleopatra eat for breakfast on her 20th birthday?",
    "Will it rain in Paris exactly one year from today?",
    "What song was playing on the radio when my grandfather was born?",
    "How many fish are in the Pacific Ocean right now?",
    "What will be the closing price of Apple stock ten years from today?",
]

# Cheap lexical abstention detector (no judge call) — good enough to rank
# scales relative to each other; the real numbers come from evaluate.py.
_ABSTAIN_RE = re.compile(
    r"(?i)\b(i\s+(?:do\s*n[o']t|can\s*n[o']?t)\s+know|cannot\s+(?:be\s+)?"
    r"(?:known|determined|answered|predicted)|unknown|unknowable|unanswerable|"
    r"no\s+(?:one|way)\s+(?:knows|to\s+know)|not\s+(?:possible|known|recorded|"
    r"documented|publicly\s+(?:known|disclosed))|"
    r"impossible\s+to\s+(?:know|say|determine|predict)|"
    r"there\s+is\s+no\s+(?:way|record|definitive\s+answer)|"
    r"i\s+can'?t\s+(?:answer|predict|know)|"
    r"(?:context|passage|text)\s+does\s+not\s+"
    r"(?:mention|provide|specify|say|state|include)|"
    r"no\s+(?:concise\s+)?answer\s+(?:exists|can\s+be)|"
    r"no\s+definitive\s+(?:answer|consensus)|"
    r"(?:exact|precise)\s+[\w\s,'-]{0,40}?\bis\s+(?:uncertain|unknown|unclear))\b")


def _is_abstain(c: str) -> bool:
    # Markdown emphasis (**not**, **unknown**) would otherwise split phrases.
    return bool(_ABSTAIN_RE.search((c or "").replace("*", "").strip()))


def _prompt(q: str) -> str:
    return f"Answer concisely in a sentence.\n\nQuestion:\n{q}\n\nAnswer:"


def _gather_degenerate_probes(results_dir: Path, n: int) -> list[dict]:
    """Pull the REAL prompts that degenerated in this run's saved results.

    Uses the exact stored `prompt` so we reproduce the actual failure, not a
    synthetic stand-in, and carries the row's ground-truth `answerable` label
    so the sweep can also score abstention. Samples up to n, spread across
    datasets.
    """
    files = sorted(results_dir.glob("*.json"))
    by_ds: dict[str, list[dict]] = {}
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        rows = rec.get("rows") or []
        degs = [r for r in rows
                if r.get("prompt") and _is_degenerate(r.get("completion"))]
        if degs:
            by_ds[f.stem] = degs
    if not by_ds:
        return []
    # Round-robin across datasets so the probe set isn't dominated by one.
    probes: list[dict] = []
    idx = 0
    while len(probes) < n and any(idx < len(v) for v in by_ds.values()):
        for ds, degs in by_ds.items():
            if idx < len(degs) and len(probes) < n:
                r = degs[idx]
                q = (r.get("question") or r.get("claim") or r.get("prompt") or "")[:45]
                kind = "answerable" if r.get("answerable") else "unanswerable"
                probes.append({"label": f"{ds}:{q}", "prompt": r["prompt"],
                               "kind": kind,
                               "old": (r.get("completion") or "").strip()[:60]})
        idx += 1
    kinds = {"answerable": 0, "unanswerable": 0}
    for pr in probes:
        kinds[pr["kind"]] += 1
    log.info("  pulled %d real degenerate probes from %s (datasets: %s, "
             "answerable: %d, unanswerable: %d)",
             len(probes), results_dir.name,
             {k: len(v) for k, v in by_ds.items()},
             kinds["answerable"], kinds["unanswerable"])
    return probes


def _is_degenerate(c: str) -> bool:
    c = (c or "").strip()
    if not c:
        return True
    if re.search(r"(\*\s*){6,}", c) or re.search(r"(—\s*){4,}", c):
        return True
    # Generic short-unit loop: any 2-12 char unit repeated 5+ times back to
    # back (catches '**&**&**&', '**& **& **&', ' sentence& sentence&', …
    # which the patterns above miss).
    if re.search(r"(.{2,12}?)\1{4,}", c):
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


def _apply_scaling(model, base: dict, factor: float,
                   zero_layers: set[int] | None = None,
                   zero_types: set[str] | None = None):
    """Set each LoRA layer's scaling to base*factor, or 0 for modules in
    zero_layers (by layer index) or zero_types (by module-name suffix, e.g.
    'down_proj')."""
    zero_layers = zero_layers or set()
    zero_types = zero_types or set()
    for name, mod in _lora_modules(model):
        m = _LAYER_RE.search(name)
        layer = int(m.group(1)) if m else -1
        suffix = name.split(".")[-1]
        zero = layer in zero_layers or suffix in zero_types
        for ad in mod.scaling:
            mod.scaling[ad] = 0.0 if zero else base[(name, ad)] * factor


def _run_setting(model, tokenizer, model_key, label: str, probes: list[dict],
                 max_new_tokens: int) -> dict:
    """Returns counts: degenerate (all probes), abstain on unanswerable
    (unlearning effect — want HIGH), abstain on answerable (over-abstention —
    want LOW)."""
    stats = {"deg": 0, "n": len(probes),
             "abst_unans": 0, "n_unans": 0, "abst_ans": 0, "n_ans": 0}
    print(f"\n===== {label} =====")
    for pr in probes:
        out = generate_greedy(model, tokenizer, model_key, pr["prompt"],
                              max_new_tokens=max_new_tokens)
        bad = _is_degenerate(out)
        abst = _is_abstain(out)
        stats["deg"] += bad
        kind = pr.get("kind", "real")
        if kind == "unanswerable":
            stats["n_unans"] += 1
            stats["abst_unans"] += (abst and not bad)
        elif kind == "answerable":
            stats["n_ans"] += 1
            stats["abst_ans"] += (abst and not bad)
        flag = "DEGENERATE" if bad else ("abstain" if abst else "commit")
        print(f"  [{flag:10s}] {pr['label'][:50]:50s} -> {out.strip()[:80]!r}")
    parts = [f"{stats['deg']}/{stats['n']} degenerate"]
    if stats["n_unans"]:
        parts.append(f"abstain {stats['abst_unans']}/{stats['n_unans']} unanswerable (want high)")
    if stats["n_ans"]:
        parts.append(f"abstain {stats['abst_ans']}/{stats['n_ans']} answerable (want low)")
    print(f"  --> {', '.join(parts)}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Adapter/training dir (must contain training_config.json).")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0])
    p.add_argument("--ablate-last", type=int, nargs="+", default=[],
                   help="Also test full-strength with the last K layers ablated.")
    p.add_argument("--ablate-types", type=str, nargs="+", default=[],
                   help="Also test full-strength with whole module TYPES ablated. "
                        "Each item is a comma-separated group, e.g. "
                        "'gate_proj,up_proj,down_proj' (zero the MLP adapters, "
                        "keep attention) and 'q_proj,k_proj,v_proj,o_proj' "
                        "(zero attention, keep MLP). Localises which module "
                        "family enacts the collapse.")
    p.add_argument("--synthetic", action="store_true",
                   help="Force the synthetic probe set. Default behaviour is "
                        "to probe with the REAL degenerate prompts from this "
                        "run's saved results (falling back to synthetic when "
                        "none are found).")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Where to read saved results for the real degenerate "
                        "probes. Defaults to RESULTS_DIR/<run-dir name>.")
    p.add_argument("--n-probes", type=int, default=100,
                   help="How many real degenerate probes to sample.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    synthetic = (
        [{"label": q, "prompt": _prompt(q), "kind": "answerable"}
         for q in ANSWERABLE_PROBES] +
        [{"label": q, "prompt": _prompt(q), "kind": "unanswerable"}
         for q in UNANSWERABLE_PROBES])
    probes: list[dict] = []
    if not args.synthetic:
        results_dir = (args.results_dir or cfg.RESULTS_DIR / run_dir.name).resolve()
        probes = _gather_degenerate_probes(results_dir, args.n_probes)
        if not probes:
            log.warning("  No degenerate rows found in saved results at %s — "
                        "falling back to synthetic probes.", results_dir)
    if not probes:
        probes = synthetic

    # adapter_scale=1.0 (explicit): the sweep must capture the TRAINED scaling
    # as its baseline — don't let the per-model config default pre-scale it.
    model, tokenizer, model_key = _load_adapter_model(run_dir, adapter_scale=1.0)
    mnt = args.max_new_tokens or cfg.max_new_tokens_for(model_key)
    log.info("  model_key=%s  max_new_tokens=%d  probes=%d", model_key, mnt, len(probes))

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
    scale_stats: dict[float, dict] = {}
    for f in args.scales:
        _apply_scaling(model, base, f)
        st = _run_setting(model, tokenizer, model_key, f"SCALE x{f:.2f}", probes, mnt)
        results[f"scale={f:.2f}"] = st
        scale_stats[f] = st

    # 2) last-K-layer ablation at full strength (optional)
    for k in args.ablate_last:
        zero = {max_layer - i for i in range(k)}
        _apply_scaling(model, base, 1.0, zero_layers=zero)
        results[f"full,ablate_last_{k}"] = _run_setting(
            model, tokenizer, model_key, f"FULL, ablate last {k} layers {sorted(zero)}",
            probes, mnt)

    # 3) module-type ablation at full strength (optional)
    for group in args.ablate_types:
        types = {t.strip() for t in group.split(",") if t.strip()}
        _apply_scaling(model, base, 1.0, zero_types=types)
        label = "+".join(sorted(types))
        results[f"full,ablate_{label}"] = _run_setting(
            model, tokenizer, model_key, f"FULL, ablate types {sorted(types)}",
            probes, mnt)

    print("\n================ SUMMARY ================")
    print(f"  {'setting':24s} {'degen':>8s} {'abst-unans':>11s} {'abst-ans':>9s}")
    for k, v in results.items():
        au = f"{v['abst_unans']}/{v['n_unans']}" if v["n_unans"] else "-"
        aa = f"{v['abst_ans']}/{v['n_ans']}" if v["n_ans"] else "-"
        print(f"  {k:24s} {v['deg']:>4d}/{v['n']:<3d} {au:>11s} {aa:>9s}")

    # Sweet spot: the LARGEST scale that is fully fluent (0 degenerate) —
    # maximises the retained unlearning signal subject to fluency.
    clean = [f for f, st in scale_stats.items() if st["deg"] == 0 and f > 0]
    if clean:
        best = max(clean)
        st = scale_stats[best]
        au = f"{st['abst_unans']}/{st['n_unans']}" if st["n_unans"] else "n/a"
        print(f"\n  SWEET SPOT: scale={best:g} — largest fully-fluent scale "
              f"(abstains {au} on unanswerable probes).")
        print(f"  Set ADAPTER_SCALE_OVERRIDES[model_key] = {best:g} in config.py, "
              "then confirm with the full evaluate.py run.")
    else:
        print("\n  No fully-fluent scale > 0 found — sweep finer/lower scales.")
    print("\nReading: scale=0.00 should be clean (base). If a scale<1.0 is clean "
          "but 1.0 is not, over-suppression is confirmed. abst-unans is the "
          "unlearning effect (want high); abst-ans is over-abstention (want "
          "low). The lexical abstain detector is approximate — confirm the "
          "chosen scale with evaluate.py before reporting numbers.")


if __name__ == "__main__":
    main()
