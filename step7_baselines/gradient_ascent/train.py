#!/usr/bin/env python3
"""Gradient Ascent (GA) baseline for the URC project.

Paper: Yao, Xu, Liu — "Large Language Model Unlearning" (NeurIPS 2024)
  https://proceedings.neurips.cc/paper_files/paper/2024/file/be52acf6bccf4a8c0a90fe2f5cfcead3-Paper-Conference.pdf
  arXiv: https://arxiv.org/abs/2310.10683

Algorithm (faithful to the paper's Equation 2)
-----------------------------------------------
At each step the model parameters update via:

    θ ← θ  −  ε₁ · ∇ L_fgt  −  ε₂ · ∇ L_rdn  −  ε₃ · ∇ L_nor

  L_fgt — *gradient ascent* on the forget examples.
    L_fgt = − CE( model(x_fgt), y_fgt )
    Maximises the NLL of the over-committed continuations the model produces
    on unanswerable prompts.  In implementation: compute CE then negate.

  L_rdn — *random mismatch* on the forget prompts.
    L_rdn = CE( model(x_fgt), y_rdn )
    y_rdn is a random response sampled from the retain pool (a correct answer
    to some other question, or a UltraChat response).  Forces the model to
    produce something unrelated to its committed answer for x_fgt.

  L_nor — *forward KL preservation* on retain examples.
    L_nor = KL( P(y|x; θ_frozen) ∥ P(y|x; θ) )  on (x_nor, y_nor)
    Keeps the current model's output distribution close to the frozen base
    (adapters disabled) to prevent catastrophic forgetting.

All losses operate on **output tokens only** (response positions), not the
prompt — following the paper's key design finding (Section 4).

URC adaptation
--------------
  x_fgt  = unanswerable prompt (same format as UOC forget pass)
  y_fgt  = mined over-committed completion (first K_ANSWER_TOKENS tokens)
  y_rdn  = random response from retain pool (sampled per forget example)
  D_nor  = KUQ answerable + SQuAD answerable + UltraChat (same as UOC retain)
  θ_frozen = base model with LoRA adapters disabled

Compared to UOC / LUNAR / AdaptiveRMU:
  GA works in *token space* (logits / cross-entropy) rather than
  representation space (hidden-state MSE).  No subspace projection, no
  direction vectors, no hooks — just standard language-model losses.

Run
---
    python step7_baselines/gradient_ascent/train.py --model qwen_instruct
    python step7_baselines/gradient_ascent/train.py --model ministral14b_instruct \\
        --lora-exclude-last 1
    python step7_baselines/gradient_ascent/train.py --model gptoss_instruct
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


# ── Model helpers ─────────────────────────────────────────────────────────────

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


@torch.no_grad()
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
                A = A_dict[ad].weight
                B = B_dict[ad].weight
                s = float(scaling.get(ad, 1.0))
            except Exception:
                continue
            AAt = A @ A.T
            BtB = B.T @ B
            fro2 = (AAt * BtB).sum()
            norms.append(s * float(fro2.clamp_min(0).sqrt()))
    return sum(norms) / len(norms) if norms else 0.0


# ── Data loading ───────────────────────────────────────────────────────────────

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


def _rdn_response_text(row: dict) -> str | None:
    """Extract the response text from a retain row (used as y_rdn for L_rdn)."""
    if row.get("__type__") == "general":
        return row.get("response") or None
    return row.get("correct_answer") or None


# ── Loss functions ─────────────────────────────────────────────────────────────

def _logit_ce_on_response(
    model, ids: torch.Tensor, p_len: int, n_ans: int,
) -> torch.Tensor | None:
    """CE loss over response tokens [p_len : p_len+n_ans].

    Performs a single forward pass; returns a scalar CE tensor or None if
    the response window is empty.
    """
    if n_ans == 0 or p_len < 1:
        return None
    logits, _ = forward_hidden_states(model, ids, layer_indices=None)
    # logits: (1, T, V) float32 CPU
    # predict ids[p_len..p_len+n_ans] from logits[p_len-1..p_len+n_ans-1]
    logit_slice = logits[0, p_len - 1: p_len + n_ans - 1, :]   # (n_ans, V)
    targets = ids[0, p_len: p_len + n_ans].cpu()                 # (n_ans,)
    if targets.shape[0] == 0:
        return None
    return F.cross_entropy(logit_slice, targets)


def _forget_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
) -> torch.Tensor | None:
    """L_fgt = − CE( model(x_fgt), y_fgt ) — gradient ascent."""
    ds = row.get("__dataset__", "kuq")
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
    ids = torch.tensor([full_ids], dtype=torch.long)
    ce = _logit_ce_on_response(model, ids, p_len, n_ans)
    if ce is None:
        return None
    return -ce   # negate: we ascend (maximise) the NLL


def _rdn_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
    rdn_response: str,
) -> torch.Tensor | None:
    """L_rdn = CE( model(x_fgt), y_rdn ) — random mismatch.

    Same forget prompt x_fgt, but target is a random retain response y_rdn.
    Forces the model to predict something unrelated to y_fgt when given x_fgt.
    """
    ds = row.get("__dataset__", "kuq")
    prompt = build_unanswerable_prompt(ds, row)
    if not prompt.strip() or not rdn_response.strip():
        return None
    try:
        full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
            tokenizer, model_key, prompt, rdn_response,
            k_answer_tokens=K_ANSWER_TOKENS,
        )
    except Exception:
        return None
    ids = torch.tensor([full_ids], dtype=torch.long)
    return _logit_ce_on_response(model, ids, p_len, n_ans)


def _nor_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
) -> torch.Tensor | None:
    """L_nor = KL( P_frozen ∥ P_current ) on retain response tokens.

    Forward KL divergence — penalises the current model for drifting away
    from the frozen base (adapters disabled), exactly as the paper describes.
    """
    kind = row.get("__type__", "answerable")

    if kind == "answerable":
        ds = row.get("__dataset__", "kuq")
        prompt = build_answerable_prompt(ds, row)
        answer = row.get("correct_answer") or ""
        if not prompt.strip() or not answer.strip():
            return None
        try:
            full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
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
            full_ids, p_len = tokenise_chat_prompt_response(
                tokenizer, model_key, prompt, response,
            )
        except Exception:
            return None
        n_ans = min(K_ANSWER_TOKENS, max(0, len(full_ids) - p_len))
    else:
        return None

    if n_ans == 0 or p_len < 1:
        return None

    ids = torch.tensor([full_ids], dtype=torch.long)
    span_lo = p_len - 1
    span_hi = p_len + n_ans - 1   # logit position range [span_lo, span_hi)

    # Frozen pass (adapters off, no grad)
    model.eval()
    model.disable_adapter_layers()
    with torch.no_grad():
        frozen_logits, _ = forward_hidden_states(model, ids, layer_indices=None)
    model.enable_adapter_layers()
    model.train()

    # Current pass (adapters on, with grad)
    current_logits, _ = forward_hidden_states(model, ids, layer_indices=None)

    # Slice to response window
    p_frz = frozen_logits[0, span_lo:span_hi, :].softmax(dim=-1)    # (n_ans, V) no grad
    lp_cur = current_logits[0, span_lo:span_hi, :].log_softmax(dim=-1)  # (n_ans, V) w grad

    # KL( P_frozen ∥ P_current ) = sum( P_frozen · log(P_frozen / P_current) )
    # F.kl_div(log_input, target) = sum( target · (log target − log_input) )
    return F.kl_div(lp_cur, p_frz, reduction="batchmean")


# ── LR scheduler (mirrors step4_train) ────────────────────────────────────────

def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Save helper ────────────────────────────────────────────────────────────────

def _save_adapter(out_dir: Path, model, tokenizer, model_key: str, args,
                  note: str = "") -> None:
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
        "method":            "gradient_ascent",
        "note":              note,
        "eps_forget":        args.eps_forget,
        "eps_rdn":           args.eps_rdn,
        "eps_retain":        args.eps_retain,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "forget_batch":      args.forget_batch,
        "retain_batch":      args.retain_batch,
        "lora_r":            cfg.LORA_R,
        "lora_alpha":        args.lora_alpha if args.lora_alpha else cfg.LORA_ALPHA,
        "lora_exclude_last": args.lora_exclude_last,
        "k_answer_tokens":   K_ANSWER_TOKENS,
    }, indent=2))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    log.info("  adapter saved to %s", out_dir)


# ── Main training loop ─────────────────────────────────────────────────────────

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

    # ── Run name ──────────────────────────────────────────────────────────────
    run_name = (
        f"{model_key}_ga"
        f"_ef{args.eps_forget:g}"
        f"_er{args.eps_rdn:g}"
        f"_en{args.eps_retain:g}"
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

    # ── Model + LoRA ───────────────────────────────────────────────────────────
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=False)

    model = _apply_lora(model, model_key,
                        lora_alpha=args.lora_alpha,
                        exclude_last=args.lora_exclude_last)
    model.train()

    # gptoss fused-expert MoE: gradient checkpointing prevents OOM
    if cfg.lora_target_parameters(model_key):
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.enable_input_require_grads()
        log.info("  gradient checkpointing enabled (fused-expert MoE)")

    log.info("  ε_forget=%.2f  ε_rdn=%.2f  ε_retain=%.2f",
             args.eps_forget, args.eps_rdn, args.eps_retain)

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
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

    # ── CSV log ────────────────────────────────────────────────────────────────
    csv_path = out_dir / "loss_log.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_fh,
        fieldnames=["step", "L_total", "L_fgt", "L_rdn", "L_nor",
                    "lora_delta_norm", "lr", "grad_norm", "elapsed_s"],
    )
    csv_writer.writeheader()

    # ── Retain cycling (for both L_nor and L_rdn random responses) ────────────
    retain_cycle = retain_data.copy()
    random.shuffle(retain_cycle)
    retain_pos = 0

    rdn_cycle = retain_data.copy()
    random.shuffle(rdn_cycle)
    rdn_pos = 0

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

    def _next_rdn_text() -> str:
        """Return the next random response text from the retain cycle."""
        nonlocal rdn_cycle, rdn_pos
        while True:
            if rdn_pos >= len(rdn_cycle):
                rdn_cycle = retain_data.copy()
                random.shuffle(rdn_cycle)
                rdn_pos = 0
            row = rdn_cycle[rdn_pos]
            rdn_pos += 1
            text = _rdn_response_text(row)
            if text and text.strip():
                return text

    # ── Training loop ──────────────────────────────────────────────────────────
    optim_step   = 0
    accum_fgt    = torch.tensor(0.0)
    accum_rdn    = torch.tensor(0.0)
    accum_nor    = torch.tensor(0.0)

    prog = Progress(total=total_steps, desc="ga", log_every=5)
    t0 = time.time()

    for ep in range(args.epochs):
        random.shuffle(forget_data)
        f_pos = 0

        while f_pos < len(forget_data):
            forget_batch = forget_data[f_pos: f_pos + args.forget_batch]
            retain_batch = _next_retain_batch()
            f_pos += args.forget_batch

            # ── L_fgt: gradient ascent on forget completions ──────────────
            fgt_losses: list[torch.Tensor] = []
            for row in forget_batch:
                loss_ex = _forget_loss_per_example(model, row, tokenizer, model_key)
                if loss_ex is not None:
                    fgt_losses.append(loss_ex)

            # ── L_rdn: random mismatch on forget prompts ──────────────────
            rdn_losses: list[torch.Tensor] = []
            if args.eps_rdn > 0:
                for row in forget_batch:
                    rdn_text = _next_rdn_text()
                    loss_ex = _rdn_loss_per_example(
                        model, row, tokenizer, model_key, rdn_text,
                    )
                    if loss_ex is not None:
                        rdn_losses.append(loss_ex)

            # ── L_nor: KL preservation on retain examples ─────────────────
            nor_losses: list[torch.Tensor] = []
            if args.eps_retain > 0:
                for row in retain_batch:
                    loss_ex = _nor_loss_per_example(model, row, tokenizer, model_key)
                    if loss_ex is not None:
                        nor_losses.append(loss_ex)

            l_fgt = (sum(fgt_losses) / len(fgt_losses) if fgt_losses else None)
            l_rdn = (sum(rdn_losses) / len(rdn_losses) if rdn_losses else None)
            l_nor = (sum(nor_losses) / len(nor_losses) if nor_losses else None)

            if l_fgt is None and l_rdn is None and l_nor is None:
                # All examples failed — skip, zero accumulators
                accum_fgt = accum_fgt + torch.tensor(0.0)
                accum_rdn = accum_rdn + torch.tensor(0.0)
                accum_nor = accum_nor + torch.tensor(0.0)
            else:
                zero = torch.tensor(0.0)
                lf = l_fgt if l_fgt is not None else zero
                lr_ = l_rdn if l_rdn is not None else zero
                ln = l_nor if l_nor is not None else zero

                l_total = (args.eps_forget * lf
                           + args.eps_rdn    * lr_
                           + args.eps_retain * ln)
                (l_total / args.grad_accum).backward()

                accum_fgt = accum_fgt + (l_fgt.detach() if l_fgt is not None else zero)
                accum_rdn = accum_rdn + (l_rdn.detach() if l_rdn is not None else zero)
                accum_nor = accum_nor + (l_nor.detach() if l_nor is not None else zero)

            inner = (f_pos // args.forget_batch) % args.grad_accum
            if inner == 0 or f_pos >= len(forget_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.DEFAULT_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                avg_fgt = float(accum_fgt) / max(args.grad_accum, 1)
                avg_rdn = float(accum_rdn) / max(args.grad_accum, 1)
                avg_nor = float(accum_nor) / max(args.grad_accum, 1)
                avg_total = (args.eps_forget * avg_fgt
                             + args.eps_rdn    * avg_rdn
                             + args.eps_retain * avg_nor)
                lr_now    = scheduler.get_last_lr()[0]
                delta_norm = _mean_lora_delta_norm(model)
                elapsed    = time.time() - t0

                csv_writer.writerow({
                    "step":           optim_step,
                    "L_total":        round(avg_total,   6),
                    "L_fgt":          round(avg_fgt,     6),
                    "L_rdn":          round(avg_rdn,     6),
                    "L_nor":          round(avg_nor,     6),
                    "lora_delta_norm": round(delta_norm, 4),
                    "lr":             lr_now,
                    "grad_norm":      round(float(grad_norm), 4),
                    "elapsed_s":      round(elapsed, 1),
                })
                csv_fh.flush()

                prog.tick(extras={
                    "ep":   f"{ep+1}/{args.epochs}",
                    "step": f"{optim_step}/{total_steps}",
                    "L":    f"{avg_total:.4f}",
                    "↑F":   f"{avg_fgt:.4f}",
                    "R":    f"{avg_rdn:.4f}",
                    "KL":   f"{avg_nor:.4f}",
                    "|Δ|":  f"{delta_norm:.3f}",
                    "lr":   f"{lr_now:.2e}",
                    "gn":   f"{float(grad_norm):.3f}",
                })

                accum_fgt = torch.tensor(0.0)
                accum_rdn = torch.tensor(0.0)
                accum_nor = torch.tensor(0.0)

    csv_fh.close()
    prog.done()

    # ── Save ───────────────────────────────────────────────────────────────────
    _save_adapter(out_dir, model, tokenizer, model_key, args)

    elapsed_total = time.time() - t0_total
    (out_dir / "train_summary.json").write_text(json.dumps({
        "num_steps":  optim_step,
        "elapsed_s":  round(elapsed_total, 1),
        "elapsed":    format_duration(elapsed_total),
    }, indent=2))

    log.info("GA done in %s.  Outputs: %s", format_duration(elapsed_total), out_dir)
    log.info("")
    log.info("Evaluate with:")
    log.info("  !python3 step5_evaluate/evaluate.py --run-dir %s \\", out_dir)
    log.info("      --datasets kuq squad selfaware faitheval nomiracl \\")
    log.info("      --heldout-dir step5_evaluate/data2/heldout")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 7: Gradient Ascent (GA) baseline training (Yao et al., NeurIPS 2024).")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)

    # Loss weights (ε₁, ε₂, ε₃ in the paper's Equation 2)
    p.add_argument("--eps-forget", type=float, default=1.0,
                   help="ε₁: weight for L_fgt gradient-ascent loss (default 1.0)")
    p.add_argument("--eps-rdn", type=float, default=1.0,
                   help="ε₂: weight for L_rdn random-mismatch loss (default 1.0). "
                        "Set to 0 to disable (reduces to pure GA + KL)")
    p.add_argument("--eps-retain", type=float, default=1.0,
                   help="ε₃: weight for L_nor KL-preservation loss (default 1.0). "
                        "Set to 0 to disable (no retain constraint)")

    # Training schedule (mirror UOC defaults)
    p.add_argument("--epochs",       type=int,   default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=cfg.DEFAULT_LR)
    p.add_argument("--forget-batch", type=int,   default=cfg.DEFAULT_FORGET_BATCH)
    p.add_argument("--retain-batch", type=int,   default=cfg.DEFAULT_RETAIN_BATCH)
    p.add_argument("--grad-accum",   type=int,   default=cfg.DEFAULT_GRAD_ACCUM)
    p.add_argument("--weight-decay", type=float, default=0.0)

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
