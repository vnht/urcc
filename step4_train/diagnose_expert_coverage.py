#!/usr/bin/env python3
"""Diagnostic — per-expert routing coverage over the forget set (MoE models).

Why
---
gpt-oss-20b routes each token to top-4 of 32 experts. The UOC forget loss only
touches the ~8-token answer window of ~1-2k forget rows, so LoRA gradient only
reaches the experts that actually *fire* on those tokens. Any (layer, expert)
slot that never fires keeps its LoRA delta at its zero init — i.e. it is never
trained, and at eval a prompt that routes there gets no intervention. This is a
prime suspect for why the kuq (parametric-recall) intervention is weak while
squad's is strong: kuq facts are spread across many "knowledge" experts, so the
narrow forget set likely leaves large coverage holes.

This script forwards every forget row (harmony-wrapped, exactly as step1/step4
tokenise it), hooks each layer's MoE router, and counts how often each expert is
selected at the answer-window positions — separately for kuq and squad. It then
reports, per layer and overall, how many (layer, expert) slots are starved.

It is read-only (no training, no writes besides an optional JSON report) and
needs the full model, so run it on the GPU box (Colab), not locally.

Run
---
    python3 step4_train/diagnose_expert_coverage.py --model gptoss_instruct
    python3 step4_train/diagnose_expert_coverage.py --model gptoss_instruct \
        --max-per-domain 300 --starved-threshold 5 --out coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import (
    Progress,
    build_unanswerable_prompt,
    format_duration,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    tokenise_prompt_plus_answer,
)


def _load_forget_commit_pool(model_key: str, max_per_domain: int | None) -> list[dict]:
    """Forget rows the forget loss actually trains on: COMMIT-labelled (set A),
    one list per domain, optionally capped per domain for speed."""
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.forget_path(model_key, dataset)
        if not path.exists():
            log.warning("  forget pool missing: %s", path)
            continue
        rows = []
        for row in load_jsonl(path):
            label = (row.get("judge_label") or "").strip().upper()
            if label not in ("COMMIT", "COMMITTED"):
                continue
            row["dataset"] = dataset
            rows.append(row)
            if max_per_domain is not None and len(rows) >= max_per_domain:
                break
        log.info("  %-6s forget COMMIT rows: %d", dataset, len(rows))
        pool.extend(rows)
    return pool


def _extract_router_indices(out, num_experts: int, top_k: int):
    """Best-effort extraction of selected expert indices from a router module's
    output. gpt-oss' GptOssTopKRouter returns (router_scores, router_indices);
    we prefer the integer indices tensor and fall back to top-k over float
    router scores so the hook is robust across transformers versions."""
    cands = out if isinstance(out, (tuple, list)) else (out,)
    # Preferred: an integer [tokens, top_k] tensor (router_indices).
    for o in cands:
        if torch.is_tensor(o) and not torch.is_floating_point(o) and o.dim() == 2:
            return o.reshape(-1, o.shape[-1])
    # Fallback: float router_scores [tokens, num_experts] -> top-k.
    for o in cands:
        if (torch.is_tensor(o) and torch.is_floating_point(o)
                and o.dim() == 2 and o.shape[-1] == num_experts):
            return o.topk(top_k, dim=-1).indices
    return None


def run(model_key: str, *, max_per_domain: int | None, starved_threshold: int,
        out_path: Path | None) -> None:
    t0 = time.time()
    if cfg.lora_target_parameters(model_key) is None:
        log.warning("Model %s is not a fused-expert MoE (no target_parameters); "
                    "expert-coverage diagnostic is a no-op for dense models.",
                    model_key)
        return

    model, tokenizer = load_model_and_tokenizer(model_key, eval_only=True)
    device = next(model.parameters()).device

    num_experts = (getattr(model.config, "num_local_experts", None)
                   or getattr(model.config, "num_experts", 32))
    top_k = (getattr(model.config, "num_experts_per_tok", None)
             or getattr(model.config, "experts_per_token", 4))
    log.info("  num_experts=%d  top_k=%d", num_experts, top_k)

    # counts[domain][layer_name] = LongTensor[num_experts]
    counts: dict[str, dict[str, torch.Tensor]] = {"kuq": {}, "squad": {}}
    ctx = {"domain": None, "start": 0, "end": 0}

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            idx = _extract_router_indices(out, num_experts, top_k)
            if idx is None or ctx["domain"] is None:
                return
            idx = idx[ctx["start"]: ctx["end"]]
            if idx.numel() == 0:
                return
            bc = torch.bincount(idx.reshape(-1).cpu(), minlength=num_experts)
            tbl = counts[ctx["domain"]].setdefault(
                name, torch.zeros(num_experts, dtype=torch.long))
            tbl += bc[:num_experts]
        return hook

    handles = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp.router") or type(mod).__name__.endswith("Router"):
            handles.append(mod.register_forward_hook(make_hook(name)))
    log.info("  hooked %d router modules", len(handles))
    if not handles:
        log.error("  no router modules found — cannot diagnose coverage. "
                  "Check the model's MoE module naming.")
        return

    pool = _load_forget_commit_pool(model_key, max_per_domain)
    progress = Progress(total=len(pool), desc="forward forget rows", log_every=50)
    used = 0
    for row in pool:
        ds = row["dataset"]
        prompt = build_unanswerable_prompt(ds, row) or ""
        answer = row.get("y_com_prefix_k8") or row.get("full_completion_clean") or ""
        if not prompt.strip() or not answer.strip():
            progress.tick(); continue
        try:
            full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
                tokenizer, model_key, prompt, answer,
                k_answer_tokens=cfg.K_ANSWER_TOKENS,
            )
        except Exception as exc:
            log.debug("  tokenise error: %s", exc)
            progress.tick(); continue
        if n_ans == 0 or p_len < 1:
            progress.tick(); continue
        ctx["domain"], ctx["start"], ctx["end"] = ds, p_len - 1, p_len - 1 + n_ans
        ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        used += 1
        progress.tick(extras={"used": used})
    progress.done(extras={"used": used})
    for h in handles:
        h.remove()

    _report(counts, num_experts=num_experts, starved_threshold=starved_threshold,
            out_path=out_path, model_key=model_key)
    log.info("DIAGNOSTIC done in %s", format_duration(time.time() - t0))


def _report(counts, *, num_experts, starved_threshold, out_path, model_key):
    log.info("")
    log.info("=" * 78)
    log.info("EXPERT COVERAGE OVER THE FORGET SET  (threshold for 'starved' = %d "
             "selections)", starved_threshold)
    log.info("  A (layer, expert) slot below threshold gets little/no LoRA "
             "gradient → stays ~at init → no intervention there at eval.")
    report: dict = {"model_key": model_key, "num_experts": num_experts,
                    "starved_threshold": starved_threshold, "domains": {}}

    for domain in ("kuq", "squad"):
        per_layer = counts[domain]
        if not per_layer:
            log.info("  [%s] no routing captured", domain)
            continue
        layer_names = sorted(per_layer.keys(),
                             key=lambda n: int(n.split(".layers.")[-1].split(".")[0])
                             if ".layers." in n else 0)
        n_layers = len(layer_names)
        total_slots = n_layers * num_experts
        zero_slots = 0
        starved_slots = 0
        all_counts = torch.zeros(num_experts, dtype=torch.long)
        worst = []  # (layer_idx, n_zero)
        for name in layer_names:
            c = per_layer[name]
            all_counts += c
            nz = int((c == 0).sum())
            ns = int((c < starved_threshold).sum())
            zero_slots += nz
            starved_slots += ns
            li = name.split(".layers.")[-1].split(".")[0] if ".layers." in name else "?"
            worst.append((li, nz, ns, int(c.max()), int(c.sum())))

        # union over layers: experts never hit at ANY layer
        union_never = int((all_counts == 0).sum())
        tot = int(all_counts.sum())
        nonzero = all_counts[all_counts > 0]
        cv = (float(nonzero.float().std() / nonzero.float().mean())
              if nonzero.numel() > 1 else 0.0)

        log.info("")
        log.info("  ── domain: %s  (%d layers × %d experts = %d slots) ──",
                 domain, n_layers, num_experts, total_slots)
        log.info("     never-selected slots : %d / %d  (%.1f%%)",
                 zero_slots, total_slots, 100.0 * zero_slots / total_slots)
        log.info("     starved (<%d) slots  : %d / %d  (%.1f%%)",
                 starved_threshold, starved_slots, total_slots,
                 100.0 * starved_slots / total_slots)
        log.info("     experts never hit at ANY layer : %d / %d",
                 union_never, num_experts)
        log.info("     load imbalance (CV of per-expert totals): %.2f", cv)
        # show the 5 layers with the most never-selected experts
        worst_sorted = sorted(worst, key=lambda x: -x[1])[:5]
        log.info("     worst layers (idx: #never / #starved / max / total):")
        for li, nz, ns, mx, sm in worst_sorted:
            log.info("       L%-3s  %3d never  %3d starved   max=%-5d total=%d",
                     li, nz, ns, mx, sm)

        report["domains"][domain] = {
            "n_layers": n_layers,
            "total_slots": total_slots,
            "never_selected_slots": zero_slots,
            "never_selected_pct": round(100.0 * zero_slots / total_slots, 2),
            "starved_slots": starved_slots,
            "starved_pct": round(100.0 * starved_slots / total_slots, 2),
            "experts_never_hit_any_layer": union_never,
            "load_cv": round(cv, 3),
        }

    log.info("")
    log.info("  INTERPRETATION: high never/starved %% (esp. for kuq vs squad) ⇒ the "
             "intervention can't reach most experts; broaden the forget set or "
             "widen the answer window so routing spreads.")
    log.info("=" * 78)

    if out_path is not None:
        out_path.write_text(json.dumps(report, indent=2))
        log.info("  report -> %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-expert routing coverage over the forget set.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--max-per-domain", type=int, default=None,
                   help="Cap forget rows per domain (speed). Default: all.")
    p.add_argument("--starved-threshold", type=int, default=5,
                   help="A slot with fewer than this many selections counts as "
                        "starved (default 5).")
    p.add_argument("--out", type=str, default="",
                   help="Optional path to write a JSON coverage report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model, max_per_domain=args.max_per_domain,
        starved_threshold=args.starved_threshold,
        out_path=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
