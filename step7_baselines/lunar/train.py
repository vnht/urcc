#!/usr/bin/env python3
"""LUNAR baseline for the URC project.

Paper / repo: facebookresearch/LUNAR (NeurIPS 2025)
  https://github.com/facebookresearch/LUNAR

Faithful port
-------------
LUNAR: LLM Unlearning via Neural Activation Redirection.

The algorithm in three phases (faithful to the original):

  Phase 0 — Direction (offline, base model, no LoRA)
    For each domain d ∈ {kuq, squad} and each selected layer L:
      μ_forget[d][L] = mean hidden state over forget examples, answer-token window
      direction[d][L] = normalize( μ⁻[d][L]  −  μ_forget[d][L] )
    This is the exact counterpart of LUNAR's `generate_directions` step:
      d = mean(harmful_activations) − mean(forget_activations)
    using μ⁻ (the abstention pole from step3_build_anchors) as the "refusal"
    centroid instead of a separate harmful.json file.

  Phase 1 — LoRA application
    Apply PEFT LoRA to the model (same config as step4_train).

  Phase 2 — Online training (per-mini-batch)
    Forget loss:   MSE( h_updated[L][span],  h_frozen[L][span] + coeff × d[d][L] )
    Retain loss:   MSE( h_updated[L][span],  h_frozen[L][span]                  )
    L = L_forget + λ · L_retain

    The forget loss is a *relative shift*: it steers each example's activations
    by `coeff × d` toward the abstention direction from wherever they currently
    sit — exactly as LUNAR trains the estimated_net to output
    `down_proj_orig(x) + coeff × direction`.

URC vs. original LUNAR
----------------------
| Aspect            | Original LUNAR               | This implementation            |
|-------------------|------------------------------|--------------------------------|
| Refusal centroid  | harmful.json (separate file) | μ⁻ from step3_build_anchors    |
| Checkpoint format | full model weights           | PEFT LoRA (loadable by eval)   |
| Training          | offline: estimated net MSE   | online: per-batch MSE via LoRA |
| Layer target      | single layer (e.g. 22)       | LAYER_SLICE (multi-layer)      |

The "harmful" data in the original TOFU/PISTOL setting corresponds to refusal
or uncertainty responses. μ⁻ plays exactly that role in URC: it is the mean
activation of legitimate-abstain (ABSTAIN-labeled) examples.

Run
---
    python step7_baselines/lunar/train.py --model qwen_instruct
    python step7_baselines/lunar/train.py --model ministral14b_instruct \
        --lora-exclude-last 1
    python step7_baselines/lunar/train.py --model gptoss_instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config as cfg
from _common import (
    Progress,
    Stopwatch,
    build_answerable_prompt,
    build_unanswerable_prompt,
    format_duration,
    forward_hidden_states,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    tokenise_chat_prompt_response,
    tokenise_prompt_plus_answer,
)

RUNS_DIR = Path(__file__).resolve().parent / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

K_ANSWER_TOKENS = cfg.K_ANSWER_TOKENS   # 8, same as UOC
DIRECTION_SAMPLES = 200                  # max examples used for direction computation


# ── Model helpers ────────────────────────────────────────────────────────────

def _num_text_layers(model) -> int:
    """Number of transformer layers; multimodal-safe (Ministral wraps under text_config)."""
    mcfg = model.config
    tc = getattr(mcfg, "text_config", None)
    return int(getattr(tc if tc is not None else mcfg, "num_hidden_layers"))


def _apply_lora(model, model_key: str, lora_alpha: int | None = None,
                exclude_last: int = 0) -> object:
    """Apply LoRA adapter — mirrors step4_train._apply_lora exactly."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_alpha = cfg.LORA_ALPHA if lora_alpha is None else lora_alpha
    expert_params = cfg.lora_target_parameters(model_key)

    if expert_params:
        # gptoss: fused MoE expert LoRA + attention LoRA
        attn_targets = cfg.lora_attn_targets(model_key)
        lcfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=attn_targets,
            target_parameters=expert_params,
            rank_pattern={p.split(".")[-1]: cfg.LORA_R for p in expert_params},
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    else:
        layers_to_transform: list[int] | None = None
        if exclude_last > 0:
            n = _num_text_layers(model)
            layers_to_transform = list(range(n - exclude_last))
            log.info("  LoRA layers_to_transform: 0..%d  (last %d of %d excluded)",
                     n - exclude_last - 1, exclude_last, n)
        lcfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=lora_alpha,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.lora_dense_targets(model_key),
            layers_to_transform=layers_to_transform,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def _mean_lora_delta_norm(model) -> float:
    """Mean Frobenius norm of (α/r)·B·A per LoRA module (from step4_train)."""
    norms: list[float] = []
    for mod in model.modules():
        A_dict = getattr(mod, "lora_A", None)
        B_dict = getattr(mod, "lora_B", None)
        scaling = getattr(mod, "scaling", None)
        if not (isinstance(A_dict, torch.nn.ModuleDict)
                and isinstance(B_dict, torch.nn.ModuleDict)
                and isinstance(scaling, dict)):
            continue
        for ad in A_dict:
            try:
                A = A_dict[ad].weight           # (r, in)
                B = B_dict[ad].weight           # (out, r)
                s = float(scaling.get(ad, 1.0))
            except Exception:
                continue
            AAt = A @ A.T                       # (r, r)
            BtB = B.T @ B                       # (r, r)
            fro2 = (AAt * BtB).sum()
            norms.append(s * float(fro2.clamp_min(0).sqrt()))
    return sum(norms) / len(norms) if norms else 0.0


# ── Anchor loading ────────────────────────────────────────────────────────────

def _load_anchors(model_key: str, layer_indices: list[int]):
    """Load μ⁻ per domain per layer from step3_build_anchors.

    Returns:
        mu_minus_per  — {"kuq": [tensor(D)×L], "squad": [...]}
        layer_indices — reconciled list (intersection with anchor layers)
    """
    ap = cfg.anchors_path(model_key)
    if not ap.exists():
        raise FileNotFoundError(f"Anchors bundle not found: {ap}.  Run step 3 first.")
    ab = torch.load(ap, map_location="cpu", weights_only=False)
    anchor_layers: list[int] = ab["layers"]

    if anchor_layers != layer_indices:
        shared = [l for l in layer_indices if l in anchor_layers]
        if not shared:
            raise ValueError(
                f"No overlap between anchor layers {anchor_layers} and "
                f"LAYER_SLICE {layer_indices}."
            )
        log.warning("  anchor layers %s ≠ LAYER_SLICE %s; using intersection %s",
                    anchor_layers, layer_indices, shared)
        idx_in_anchor = [anchor_layers.index(l) for l in shared]
        layer_indices = shared
    else:
        idx_in_anchor = list(range(len(layer_indices)))

    def _split(t: torch.Tensor) -> list[torch.Tensor]:
        return [t[i].float() for i in idx_in_anchor]

    minus_per_t = (ab.get("mu_minus_per")
                   or {"kuq": ab["mu_minus"], "squad": ab["mu_minus"]})
    mu_minus_per = {d: _split(t) for d, t in minus_per_t.items()}

    if "mu_minus_per" not in ab:
        log.warning("  anchors bundle has no per-domain poles; using grand mean for "
                    "both domains.  Re-run step 3 to build per-domain anchors.")

    log.info("  anchors loaded from %s  (domains: %s, layers: %s)",
             ap.name, list(mu_minus_per.keys()), layer_indices)
    return mu_minus_per, layer_indices


# ── Direction computation (Phase 0, faithful to LUNAR generate_directions) ───

@torch.no_grad()
def _compute_directions(
    model, tokenizer, model_key: str,
    forget_data: list[dict],
    mu_minus_per: dict[str, list[torch.Tensor]],
    layer_indices: list[int],
    n_samples: int,
) -> dict[str, dict[int, torch.Tensor]]:
    """Compute per-domain, per-layer forget direction.

    Mirrors LUNAR's generate_directions / get_mean_diff:
        d[domain][L] = normalize( μ⁻[domain][L]  −  μ_forget[domain][L] )

    μ_forget[domain][L] is the mean hidden state over forget examples from that
    domain, measured over the answer-token window at the BASE model (no LoRA).

    Returns:
        directions — {"kuq": {layer_id: tensor(D)}, "squad": {...}}
    """
    model.eval()
    log.info("  LUNAR Phase 0: computing forget directions (%d samples per domain)", n_samples)

    domains = list(mu_minus_per.keys())
    n_layers = len(layer_indices)
    hidden_dim = mu_minus_per[domains[0]][0].shape[0]

    # Accumulate per-domain sums (float64 for numerical stability)
    sums:  dict[str, list[torch.Tensor]] = {d: [torch.zeros(hidden_dim, dtype=torch.float64)
                                                  for _ in range(n_layers)]
                                             for d in domains}
    counts: dict[str, int] = {d: 0 for d in domains}

    # Iterate over forget data, grouped by domain
    by_domain: dict[str, list[dict]] = {d: [] for d in domains}
    for row in forget_data:
        ds = row.get("__dataset__", "kuq")
        if ds in by_domain:
            by_domain[ds].append(row)

    for ds in domains:
        pool = by_domain[ds][:n_samples]
        for row in pool:
            prompt = build_unanswerable_prompt(ds, row)
            answer = row.get("y_com_prefix_k8") or row.get("full_completion_clean") or ""
            if not prompt.strip() or not answer.strip():
                continue
            try:
                full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
                    tokenizer, model_key, prompt, answer,
                    k_answer_tokens=K_ANSWER_TOKENS,
                )
            except Exception:
                continue
            if n_ans == 0 or p_len < 1:
                continue

            ids = torch.tensor([full_ids], dtype=torch.long)
            _, hiddens = forward_hidden_states(model, ids, layer_indices)

            span_lo, span_hi = p_len - 1, p_len - 1 + n_ans
            for li, h in enumerate(hiddens):
                h_slice = h[0, span_lo:span_hi, :].float().mean(dim=0)  # (D,)
                sums[ds][li] += h_slice.double()
            counts[ds] += 1

    directions: dict[str, dict[int, torch.Tensor]] = {}
    for ds in domains:
        n = counts[ds]
        if n == 0:
            log.warning("  no valid forget examples for domain %s; "
                        "falling back to μ⁻ direction alone", ds)
            # Degenerate fallback: direction = μ⁻ normalised
            directions[ds] = {}
            for li, layer_id in enumerate(layer_indices):
                d = mu_minus_per[ds][li].float()
                norm = d.norm()
                directions[ds][layer_id] = (d / norm) if norm > 1e-8 else d
            continue

        directions[ds] = {}
        for li, layer_id in enumerate(layer_indices):
            mu_forget_l = (sums[ds][li] / n).float()       # (D,)
            mu_minus_l  = mu_minus_per[ds][li].float()      # (D,)

            # Direction: FROM forget cluster TOWARD abstention pole
            d = mu_minus_l - mu_forget_l
            norm = d.norm()
            d = (d / norm) if norm > 1e-8 else d
            directions[ds][layer_id] = d

        log.info("  direction[%s]:  %d samples used  "
                 "‖μ⁻−μ_forget‖=%.2f (layer %d)",
                 ds, n,
                 float((mu_minus_per[ds][-1] -
                        (sums[ds][-1] / n).float()).norm()),
                 layer_indices[-1])

    model.train()
    return directions


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_forget(model_key: str) -> list[dict]:
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.forget_path(model_key, dataset)
        if not path.exists():
            log.warning("  forget pool missing: %s", path)
            continue
        for r in load_jsonl(path):
            if (r.get("judge_label") or "").upper() not in ("COMMIT", "COMMITTED"):
                continue
            r["__type__"] = "forget"
            r["__dataset__"] = dataset
            pool.append(r)
    log.info("  forget: %d examples", len(pool))
    return pool


def _load_retain(model_key: str) -> list[dict]:  # noqa: ARG001
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.sampled_answerable_path(dataset)
        if not path.exists():
            log.warning("  retain answerable missing: %s", path)
            continue
        for r in load_jsonl(path):
            r["__type__"] = "answerable"
            r["__dataset__"] = dataset
            pool.append(r)
    gen_path = cfg.sampled_general_path()
    if gen_path.exists():
        for r in load_jsonl(gen_path):
            r["__type__"] = "general"
            pool.append(r)
    else:
        log.warning("  retain general (UltraChat) missing: %s", gen_path)
    log.info("  retain: %d examples", len(pool))
    return pool


# ── Loss functions ────────────────────────────────────────────────────────────

def _forget_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
    layer_indices: list[int],
    directions: dict[str, dict[int, torch.Tensor]],
    coeff: float,
) -> torch.Tensor | None:
    """L_forget — LUNAR faithful: MSE(h_updated[span], h_frozen[span] + coeff × d[L])

    This is the direct translation of LUNAR's estimated_net training objective.
    In the original, down_proj is trained to output `down_proj_orig(x) + coeff × d`.
    Equivalently at the block output level:
        target[L] = block_output_frozen[L] + coeff × d[L]
    The LoRA drives the updated block output toward this shifted target.

    Key difference from UOC's L_forget:
        UOC:   MSE( V_L^T (h - μ⁻) )  — projected, absolute pull to pole
        LUNAR: MSE( h_updated - (h_frozen + coeff × d) )  — full space, relative shift
    """
    ds = row.get("__dataset__", "kuq")
    d_domain = directions.get(ds)
    if d_domain is None:
        return None

    prompt = build_unanswerable_prompt(ds, row)
    answer = row.get("y_com_prefix_k8") or row.get("full_completion_clean") or ""
    if not prompt.strip() or not answer.strip():
        return None

    try:
        full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
            tokenizer, model_key, prompt, answer,
            k_answer_tokens=K_ANSWER_TOKENS,
        )
    except Exception:
        return None
    if n_ans == 0 or p_len < 1:
        return None

    ids = torch.tensor([full_ids], dtype=torch.long)
    span_lo, span_hi = p_len - 1, p_len - 1 + n_ans

    # Frozen reference (adapters disabled, no grad) — same window as updated
    model.eval()
    model.disable_adapter_layers()
    with torch.no_grad():
        _, frozen_hiddens = forward_hidden_states(model, ids, layer_indices)
    model.enable_adapter_layers()
    model.train()

    # Updated forward (with grad)
    _, hiddens = forward_hidden_states(model, ids, layer_indices)

    layer_losses: list[torch.Tensor] = []
    for li, layer_id in enumerate(layer_indices):
        h_u = hiddens[li][0, span_lo:span_hi, :]            # (n_ans, D) updated
        h_f = frozen_hiddens[li][0, span_lo:span_hi, :]     # (n_ans, D) frozen

        d = d_domain.get(layer_id)
        if d is None:
            continue
        d = d.to(h_f.device).to(h_f.dtype)                  # (D,)
        target = (h_f + coeff * d.unsqueeze(0).expand_as(h_f)).detach()
        layer_losses.append(F.mse_loss(h_u, target))

    if not layer_losses:
        return None
    return sum(layer_losses) / len(layer_losses)


def _retain_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
    layer_indices: list[int],
) -> torch.Tensor | None:
    """L_retain — full-space MSE to frozen activations (identical to UOC but no V).

    Same structure as step4_train._compute_retain_loss per example.
    """
    kind = row.get("__type__", "answerable")

    if kind == "answerable":
        ds = row.get("__dataset__", "kuq")
        prompt = build_answerable_prompt(ds, row)
        answer = row.get("correct_answer") or ""
        if not prompt.strip() or not answer.strip():
            return None
        try:
            full_ids, resp_start, n_ans = tokenise_prompt_plus_answer(
                tokenizer, model_key, prompt, answer,
                k_answer_tokens=K_ANSWER_TOKENS,
            )
        except Exception:
            return None
    elif kind == "general":
        prompt   = row.get("prompt")   or ""
        response = row.get("response") or ""
        if not prompt.strip() or not response.strip():
            return None
        try:
            full_ids, resp_start = tokenise_chat_prompt_response(
                tokenizer, model_key, prompt, response,
            )
        except Exception:
            return None
        n_ans = min(K_ANSWER_TOKENS, max(0, len(full_ids) - resp_start))
    else:
        return None

    if n_ans == 0 or resp_start < 1:
        return None

    ids = torch.tensor([full_ids], dtype=torch.long)
    span_lo, span_hi = resp_start - 1, resp_start - 1 + n_ans

    model.eval()
    model.disable_adapter_layers()
    with torch.no_grad():
        _, ref_hiddens = forward_hidden_states(model, ids, layer_indices)
    model.enable_adapter_layers()
    model.train()

    _, hiddens = forward_hidden_states(model, ids, layer_indices)

    layer_losses: list[torch.Tensor] = []
    for li, h in enumerate(hiddens):
        h_s   = h[0,               span_lo:span_hi, :]
        ref_s = ref_hiddens[li][0, span_lo:span_hi, :]
        layer_losses.append(F.mse_loss(h_s, ref_s.to(h_s.device).to(h_s.dtype)))

    if not layer_losses:
        return None
    return sum(layer_losses) / len(layer_losses)


# ── LR scheduler (mirrors step4_train) ───────────────────────────────────────

def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_adapter(out_dir: Path, model, tokenizer, model_key: str, args,
                  layer_indices: list[int]) -> None:
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(tmp))
    try:
        tokenizer.save_pretrained(str(tmp))
    except Exception as exc:
        log.warning("  tokenizer.save_pretrained failed (%s); skipping", exc)

    (tmp / "training_config.json").write_text(json.dumps({
        "model_key":         model_key,
        "method":            "lunar",
        "coeff":             args.coeff,
        "lambda_retain":     args.lambda_retain,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "forget_batch":      args.forget_batch,
        "retain_batch":      args.retain_batch,
        "lora_r":            cfg.LORA_R,
        "lora_alpha":        args.lora_alpha if args.lora_alpha else cfg.LORA_ALPHA,
        "lora_exclude_last": args.lora_exclude_last,
        "k_answer_tokens":   K_ANSWER_TOKENS,
        "layer_indices":     layer_indices,
    }, indent=2))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    log.info("  adapter saved to %s", out_dir)


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    model_key = args.model
    t0_total  = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    forget_data = _load_forget(model_key)
    retain_data = _load_retain(model_key)
    if not forget_data:
        raise RuntimeError("Forget pool empty. Run step0_mine first.")
    if not retain_data:
        raise RuntimeError("Retain pool empty. Check step0_mine/data/sampled/.")

    if args.dry_run:
        forget_data = forget_data[:8]
        retain_data = retain_data[:8]
        args.epochs = 1
        log.info("  DRY RUN: 8 examples, 1 epoch")

    # ── Anchors (μ⁻) and layer indices ───────────────────────────────────────
    layer_indices: list[int] = cfg.LAYER_SLICE.get(model_key, [])
    if not layer_indices:
        raise ValueError(f"No LAYER_SLICE for {model_key!r} in config.py.")

    # ── Run name ──────────────────────────────────────────────────────────────
    run_name = (
        f"{model_key}_lunar"
        f"_coeff{args.coeff:g}"
        f"_lam{args.lambda_retain:g}"
        f"_ep{args.epochs}"
        f"_lr{args.lr:g}"
    )
    if args.lora_exclude_last:
        run_name = f"{run_name}_excl{args.lora_exclude_last}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run: %s", run_name)
    log.info("Output dir: %s", out_dir)

    # ── Phase 0: Load BASE model, compute directions (no LoRA) ───────────────
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=False)

    mu_minus_per, layer_indices = _load_anchors(model_key, layer_indices)

    with Stopwatch("direction computation"):
        directions = _compute_directions(
            model, tokenizer, model_key,
            forget_data, mu_minus_per, layer_indices,
            n_samples=args.direction_samples,
        )

    # ── Phase 1: Apply LoRA ───────────────────────────────────────────────────
    model = _apply_lora(model, model_key,
                        lora_alpha=args.lora_alpha,
                        exclude_last=args.lora_exclude_last)
    model.train()

    # gptoss fused-expert MoE: gradient checkpointing prevents OOM from
    # materialising all expert deltas in memory simultaneously (see step4_train).
    if cfg.lora_target_parameters(model_key):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.enable_input_require_grads()
        log.info("  gradient checkpointing enabled (fused-expert MoE)")

    log.info("  layer_indices: %s  coeff=%.2f  λ_retain=%.2f",
             layer_indices, args.coeff, args.lambda_retain)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    log.info("  AdamW lr=%g  weight_decay=%g", args.lr, args.weight_decay)

    f_steps_per_epoch = math.ceil(len(forget_data) / args.forget_batch)
    total_steps = math.ceil(f_steps_per_epoch / args.grad_accum) * args.epochs
    warmup = max(1, int(total_steps * cfg.DEFAULT_WARMUP_RATIO))
    scheduler = _linear_warmup_decay(optimizer, warmup, total_steps)
    log.info("  total_steps=%d  warmup=%d  epochs=%d", total_steps, warmup, args.epochs)

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "loss_log.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_fh,
        fieldnames=["step", "L_total", "L_forget", "L_retain",
                    "lora_delta_norm", "lr", "grad_norm", "elapsed_s"],
    )
    csv_writer.writeheader()

    # ── Retain cycling ────────────────────────────────────────────────────────
    retain_cycle = retain_data.copy()
    random.shuffle(retain_cycle)
    retain_pos = 0

    def _next_retain_batch() -> list[dict]:
        nonlocal retain_cycle, retain_pos
        batch: list[dict] = []
        for _ in range(args.retain_batch):
            if retain_pos >= len(retain_cycle):
                retain_cycle = retain_data.copy()
                random.shuffle(retain_cycle)
                retain_pos = 0
            batch.append(retain_cycle[retain_pos])
            retain_pos += 1
        return batch

    # ── Phase 2: Training loop ────────────────────────────────────────────────
    optim_step = 0
    accum_forget = torch.tensor(0.0)
    accum_retain = torch.tensor(0.0)

    prog = Progress(total=total_steps, desc="lunar", log_every=5)
    t0 = time.time()

    for ep in range(args.epochs):
        random.shuffle(forget_data)
        f_pos = 0

        while f_pos < len(forget_data):
            forget_batch = forget_data[f_pos: f_pos + args.forget_batch]
            retain_batch = _next_retain_batch()
            f_pos += args.forget_batch

            # ── Forget loss (LUNAR: relative shift toward μ⁻ direction) ───────
            forget_losses: list[torch.Tensor] = []
            for row in forget_batch:
                loss_ex = _forget_loss_per_example(
                    model, row, tokenizer, model_key,
                    layer_indices, directions, args.coeff,
                )
                if loss_ex is not None:
                    forget_losses.append(loss_ex)

            l_forget = (sum(forget_losses) / len(forget_losses)
                        if forget_losses else None)

            # ── Retain loss (MSE to frozen activations — no V) ─────────────
            retain_losses: list[torch.Tensor] = []
            for row in retain_batch:
                loss_ex = _retain_loss_per_example(
                    model, row, tokenizer, model_key, layer_indices,
                )
                if loss_ex is not None:
                    retain_losses.append(loss_ex)

            l_retain = (sum(retain_losses) / len(retain_losses)
                        if retain_losses else None)

            # Skip backward if no usable examples in this micro-batch
            if l_forget is None and l_retain is None:
                accum_forget = accum_forget + torch.tensor(0.0)
                accum_retain = accum_retain + torch.tensor(0.0)
            else:
                lf = l_forget if l_forget is not None else torch.tensor(0.0)
                lr_ = l_retain if l_retain is not None else torch.tensor(0.0)
                l_total = lf + args.lambda_retain * lr_
                (l_total / args.grad_accum).backward()
                accum_forget = accum_forget + (l_forget.detach() if l_forget is not None else torch.tensor(0.0))
                accum_retain = accum_retain + (l_retain.detach() if l_retain is not None else torch.tensor(0.0))

            inner = (f_pos // args.forget_batch) % args.grad_accum
            if inner == 0 or f_pos >= len(forget_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.DEFAULT_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                avg_forget = float(accum_forget) / max(args.grad_accum, 1)
                avg_retain = float(accum_retain) / max(args.grad_accum, 1)
                avg_total  = avg_forget + args.lambda_retain * avg_retain
                lr_now     = scheduler.get_last_lr()[0]
                delta_norm = _mean_lora_delta_norm(model)
                elapsed    = time.time() - t0

                csv_writer.writerow({
                    "step":           optim_step,
                    "L_total":        round(avg_total,   6),
                    "L_forget":       round(avg_forget,  6),
                    "L_retain":       round(avg_retain,  6),
                    "lora_delta_norm": round(delta_norm, 4),
                    "lr":             lr_now,
                    "grad_norm":      round(float(grad_norm), 4),
                    "elapsed_s":      round(elapsed, 1),
                })
                csv_fh.flush()

                prog.tick(extras={
                    "ep":  f"{ep+1}/{args.epochs}",
                    "step": f"{optim_step}/{total_steps}",
                    "L":    f"{avg_total:.4f}",
                    "L_F":  f"{avg_forget:.4f}",
                    "L_R":  f"{avg_retain:.4f}",
                    "|Δ|":  f"{delta_norm:.3f}",
                    "lr":   f"{lr_now:.2e}",
                    "gn":   f"{float(grad_norm):.3f}",
                })

                accum_forget = torch.tensor(0.0)
                accum_retain = torch.tensor(0.0)

    csv_fh.close()
    prog.done()

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_adapter(out_dir, model, tokenizer, model_key, args, layer_indices)

    elapsed_total = time.time() - t0_total
    (out_dir / "train_summary.json").write_text(json.dumps({
        "num_steps":  optim_step,
        "elapsed_s":  round(elapsed_total, 1),
        "elapsed":    format_duration(elapsed_total),
    }, indent=2))

    log.info("LUNAR done in %s.  Outputs: %s", format_duration(elapsed_total), out_dir)
    log.info("")
    log.info("Evaluate with:")
    log.info("  !python3 step5_evaluate/evaluate.py --run-dir %s \\", out_dir)
    log.info("      --datasets kuq squad selfaware faitheval nomiracl \\")
    log.info("      --heldout-dir step5_evaluate/data2/heldout")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 7: LUNAR baseline training.")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)

    # LUNAR-specific
    p.add_argument("--coeff", type=float, default=2.0,
                   help="Scaling coefficient for the forget direction shift. "
                        "Original LUNAR default: 2.0 (coeff_list=[+2.0] in forget.yaml)")
    p.add_argument("--direction-samples", type=int, default=DIRECTION_SAMPLES,
                   help="Max forget examples per domain used for direction computation "
                        "(Phase 0).  Mirrors LUNAR's n_train knob.")

    # Loss
    p.add_argument("--lambda-retain", type=float, default=cfg.DEFAULT_LAMBDA_RETAIN,
                   help="Weight on L_retain (default 1.0, same as UOC)")

    # Training schedule (mirror UOC defaults)
    p.add_argument("--epochs",       type=int,   default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=cfg.DEFAULT_LR)
    p.add_argument("--forget-batch", type=int,   default=cfg.DEFAULT_FORGET_BATCH)
    p.add_argument("--retain-batch", type=int,   default=cfg.DEFAULT_RETAIN_BATCH)
    p.add_argument("--grad-accum",   type=int,   default=cfg.DEFAULT_GRAD_ACCUM)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (0.0 default; >0 caps LoRA magnitude)")

    # LoRA
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha override (default cfg.LORA_ALPHA)")
    p.add_argument("--lora-exclude-last", type=int, default=0,
                   help="Exclude last N layers from LoRA. "
                        "Recommended: 1 for ministral14b_instruct.")

    # Utilities
    p.add_argument("--tag",     type=str, default="",
                   help="Optional tag appended to the run name")
    p.add_argument("--dry-run", action="store_true",
                   help="8 examples, 1 epoch — smoke test")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
