#!/usr/bin/env python3
"""Negative Preference Optimization (NPO) baseline for the URC project.

Paper: Zhang, Lin, Bai, Mei — "Negative Preference Optimization: From
  Catastrophic Collapse to Effective Unlearning" (ICML 2024)
  arXiv: https://arxiv.org/abs/2404.05868
  Code:  https://github.com/licong-lin/negative-preference-optimization

Algorithm (faithful to TOFU/dataloader.py → npo_grad_diff)
----------------------------------------------------------
NPO treats each forget example as a *negative-only* preference pair.
Derived from DPO by dropping the positive (preferred) response:

    L_NPO(θ) = −(2/β) · E_{D_fgt}[ log σ(−β · log(π_θ(y|x) / π_ref(y|x))) ]

Equivalently (used for stability in practice):

    L_NPO(θ) = (2/β) · E_{D_fgt}[ log(1 + (π_θ(y|x) / π_ref(y|x))^β) ]

Key implementation detail (faithful to get_batch_loss in the original code):

    NLL(θ, x, y) = Σ_t  −log P_θ(y_t | y_{<t}, x)    [SUM over tokens, not mean]

    neg_log_ratios = NLL(θ, x, y) − NLL(ref, x, y)
                   = −log(π_θ(y|x) / π_ref(y|x))

    L_NPO = −logsigmoid(β · neg_log_ratios) · (2/β)   [per-example, then .mean()]

The gradient of L_NPO down-weights the forget response proportionally to how
much the current model already agrees with the reference, providing a natural
lower bound that prevents catastrophic collapse (unlike pure gradient ascent).

Two retain variants (--retain-loss), using original variant names:

  npo_grad_diff  [default]  Standard cross-entropy on retain responses.
                             Faithful to npo_grad_diff in original code.

  npo_KL                     KL(π_ref ∥ π_θ) on retain responses.
                             Faithful to npo_KL in original code.

Combined loss:

    L_total = npo_coeff · L_NPO  +  lambda_retain · L_retain

URC adaptation
--------------
  x_fgt   = unanswerable prompt (same format as UOC forget pass)
  y_fgt   = mined over-committed completion (first K_ANSWER_TOKENS tokens)
  D_nor   = KUQ answerable + SQuAD answerable + UltraChat (same as UOC retain)
  π_ref   = frozen base with LoRA adapters disabled (disable_adapter_layers)
  Labels  = −100 for prompt tokens, token IDs for answer tokens (per original)

Run
---
    python step7_baselines/npo/train.py --model qwen_instruct
    python step7_baselines/npo/train.py --model ministral14b_instruct
    python step7_baselines/npo/train.py --model gptoss_instruct
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


# ── Core NPO primitive ─────────────────────────────────────────────────────────

def _npo_batch_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence NLL (sum over non-ignored response tokens).

    Faithful to get_batch_loss() in the original NPO repository:

        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    Args:
        logits: (1, T, V) raw logits from model forward pass
        labels: (1, T) token IDs; prompt tokens should be set to -100

    Returns:
        (1,) tensor — summed NLL over response tokens (not mean)
    """
    shifted_labels = labels[..., 1:].contiguous()         # (1, T-1)
    shifted_logits = logits[..., :-1, :].contiguous()     # (1, T-1, V)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_tok = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels)  # (1, T-1)
    return per_tok.sum(dim=-1)   # (1,) — sum over tokens, per sequence


# ── Loss functions ─────────────────────────────────────────────────────────────

def _npo_loss_per_example(
    model,
    row: dict,
    tokenizer,
    model_key: str,
    beta: float,
) -> torch.Tensor | None:
    """NPO forget loss for one example.

    Faithful to npo / npo_grad_diff in TOFU/dataloader.py:

        forget_loss_current = get_batch_loss(outputs.logits, labels)
        forget_loss_oracle  = get_batch_loss(forget_logits_oracle, labels)
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    neg_log_ratios = NLL_θ − NLL_ref = −log(π_θ(y|x) / π_ref(y|x))
    """
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
    if n_ans == 0 or p_len < 1:
        return None

    # Labels: -100 for prompt tokens, actual token IDs for answer tokens
    raw_labels = [-100] * p_len + full_ids[p_len:]
    ids    = torch.tensor([full_ids],   dtype=torch.long)   # (1, T)
    labels = torch.tensor([raw_labels], dtype=torch.long)   # (1, T)

    # Current model (adapters ON, with gradients)
    logits_cur, _ = forward_hidden_states(model, ids, layer_indices=None)
    nll_cur = _npo_batch_nll(logits_cur, labels)    # (1,)

    # Reference model (adapters OFF, frozen — mirrors oracle_model in original)
    model.eval()
    model.disable_adapter_layers()
    with torch.no_grad():
        logits_ref, _ = forward_hidden_states(model, ids, layer_indices=None)
    model.enable_adapter_layers()
    model.train()

    nll_ref = _npo_batch_nll(logits_ref, labels).detach()   # (1,), no grad

    # neg_log_ratios = NLL_θ − NLL_ref = −log(π_θ/π_ref) per sequence
    neg_log_ratios = nll_cur - nll_ref                       # (1,), keeps grad

    # L_NPO = -(2/β) · log σ(-β · log(π_θ/π_ref))
    #       = -logsigmoid(β · neg_log_ratios) · 2/β
    return (-F.logsigmoid(beta * neg_log_ratios) * (2.0 / beta)).squeeze()


def _ce_retain_per_example(
    model, row: dict, tokenizer, model_key: str,
) -> torch.Tensor | None:
    """CE on retain examples — the grad_diff retain term in npo_grad_diff.

    Faithful to retain_loss = retain_outputs.loss in the original.
    Uses response tokens only (K_ANSWER_TOKENS), consistent with URC.
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

    # Forward pass and CE over response tokens [p_len : p_len+n_ans]
    logits, _ = forward_hidden_states(model, ids, layer_indices=None)
    logit_slice = logits[0, p_len - 1: p_len + n_ans - 1, :]   # (n_ans, V)
    targets = ids[0, p_len: p_len + n_ans].cpu()                # (n_ans,)
    if targets.shape[0] == 0:
        return None
    return F.cross_entropy(logit_slice, targets)


def _kl_retain_per_example(
    model, row: dict, tokenizer, model_key: str,
) -> torch.Tensor | None:
    """KL(π_ref ∥ π_θ) on retain examples — the npo_KL retain term.

    Faithful to the KL computation in TOFU/dataloader.py npo_KL:

        retain_probs   = F.log_softmax(retain_outputs.logits, dim=-1)
        current_probs  = F.log_softmax(current_outputs.logits, dim=-1)
        retain_loss = F.kl_div(current_probs, retain_probs,
                               reduction='batchmean', log_target=True)
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
    span_hi = p_len + n_ans - 1   # logit window [span_lo, span_hi)

    # Reference (frozen, adapters off)
    model.eval()
    model.disable_adapter_layers()
    with torch.no_grad():
        logits_ref, _ = forward_hidden_states(model, ids, layer_indices=None)
    model.enable_adapter_layers()
    model.train()

    # Current (adapters on, with grad)
    logits_cur, _ = forward_hidden_states(model, ids, layer_indices=None)

    # Slice to response window
    lp_ref = logits_ref[0, span_lo:span_hi, :].log_softmax(dim=-1).detach()  # (n, V)
    lp_cur = logits_cur[0, span_lo:span_hi, :].log_softmax(dim=-1)           # (n, V)

    # KL(π_ref ∥ π_θ) = Σ π_ref · log(π_ref / π_θ)
    # F.kl_div(log_input, target, log_target=True) = Σ target · (target - log_input)
    # → pass lp_cur as input (log), lp_ref as target (log) with log_target=True
    return F.kl_div(lp_cur, lp_ref, reduction="batchmean", log_target=True)


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
        "model_key":        model_key,
        "method":           "npo",
        "note":             note,
        "beta":             args.beta,
        "npo_coeff":        args.npo_coeff,
        "lambda_retain":    args.lambda_retain,
        "retain_loss":      args.retain_loss,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "forget_batch":     args.forget_batch,
        "retain_batch":     args.retain_batch,
        "lora_r":           cfg.LORA_R,
        "lora_alpha":       args.lora_alpha if args.lora_alpha else cfg.LORA_ALPHA,
        "lora_exclude_last": args.lora_exclude_last,
        "k_answer_tokens":  K_ANSWER_TOKENS,
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
        f"{model_key}_npo"
        f"_b{args.beta:g}"
        f"_nc{args.npo_coeff:g}"
        f"_lam{args.lambda_retain:g}"
        f"_{args.retain_loss}"
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

    log.info("  beta=%.4f  npo_coeff=%.2f  lambda_retain=%.2f  retain_loss=%s",
             args.beta, args.npo_coeff, args.lambda_retain, args.retain_loss)

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
        fieldnames=["step", "L_total", "L_npo", "L_retain",
                    "lora_delta_norm", "lr", "grad_norm", "elapsed_s"],
    )
    csv_writer.writeheader()

    # ── Retain data cycling ────────────────────────────────────────────────────
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

    # ── Retain loss function dispatch ──────────────────────────────────────────
    if args.retain_loss == "npo_KL":
        _retain_loss_fn = _kl_retain_per_example
    else:  # "npo_grad_diff" (default)
        _retain_loss_fn = _ce_retain_per_example

    # ── Training loop ──────────────────────────────────────────────────────────
    optim_step   = 0
    accum_npo    = torch.tensor(0.0)
    accum_retain = torch.tensor(0.0)

    prog = Progress(total=total_steps, desc="npo", log_every=5)
    t0 = time.time()

    for ep in range(args.epochs):
        random.shuffle(forget_data)
        f_pos = 0

        while f_pos < len(forget_data):
            forget_batch = forget_data[f_pos: f_pos + args.forget_batch]
            retain_batch = _next_retain_batch() if args.lambda_retain > 0 else []
            f_pos += args.forget_batch

            # ── NPO forget loss ───────────────────────────────────────────────
            npo_losses: list[torch.Tensor] = []
            for row in forget_batch:
                loss_ex = _npo_loss_per_example(
                    model, row, tokenizer, model_key, args.beta,
                )
                if loss_ex is not None:
                    npo_losses.append(loss_ex)

            # ── Retain loss (CE or KL) ────────────────────────────────────────
            retain_losses: list[torch.Tensor] = []
            if args.lambda_retain > 0:
                for row in retain_batch:
                    loss_ex = _retain_loss_fn(model, row, tokenizer, model_key)
                    if loss_ex is not None:
                        retain_losses.append(loss_ex)

            l_npo    = (sum(npo_losses)    / len(npo_losses)    if npo_losses    else None)
            l_retain = (sum(retain_losses) / len(retain_losses) if retain_losses else None)

            if l_npo is None and l_retain is None:
                accum_npo    = accum_npo    + torch.tensor(0.0)
                accum_retain = accum_retain + torch.tensor(0.0)
            else:
                zero = torch.tensor(0.0)
                ln  = l_npo    if l_npo    is not None else zero
                lr_ = l_retain if l_retain is not None else zero

                l_total = args.npo_coeff * ln + args.lambda_retain * lr_
                (l_total / args.grad_accum).backward()

                accum_npo    = accum_npo    + (l_npo.detach()    if l_npo    is not None else zero)
                accum_retain = accum_retain + (l_retain.detach() if l_retain is not None else zero)

            inner = (f_pos // args.forget_batch) % args.grad_accum
            if inner == 0 or f_pos >= len(forget_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.DEFAULT_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                avg_npo    = float(accum_npo)    / max(args.grad_accum, 1)
                avg_retain = float(accum_retain) / max(args.grad_accum, 1)
                avg_total  = args.npo_coeff * avg_npo + args.lambda_retain * avg_retain
                lr_now     = scheduler.get_last_lr()[0]
                delta_norm = _mean_lora_delta_norm(model)
                elapsed    = time.time() - t0

                csv_writer.writerow({
                    "step":            optim_step,
                    "L_total":         round(avg_total,    6),
                    "L_npo":           round(avg_npo,      6),
                    "L_retain":        round(avg_retain,   6),
                    "lora_delta_norm": round(delta_norm,   4),
                    "lr":              lr_now,
                    "grad_norm":       round(float(grad_norm), 4),
                    "elapsed_s":       round(elapsed, 1),
                })
                csv_fh.flush()

                prog.tick(extras={
                    "ep":    f"{ep+1}/{args.epochs}",
                    "step":  f"{optim_step}/{total_steps}",
                    "L":     f"{avg_total:.4f}",
                    "NPO":   f"{avg_npo:.4f}",
                    "Ret":   f"{avg_retain:.4f}",
                    "|Δ|":   f"{delta_norm:.3f}",
                    "lr":    f"{lr_now:.2e}",
                    "gn":    f"{float(grad_norm):.3f}",
                })

                accum_npo    = torch.tensor(0.0)
                accum_retain = torch.tensor(0.0)

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

    log.info("NPO done in %s.  Outputs: %s", format_duration(elapsed_total), out_dir)
    log.info("")
    log.info("Evaluate with:")
    log.info("  !python3 step5_evaluate/evaluate.py --run-dir %s \\", out_dir)
    log.info("      --datasets kuq squad selfaware faitheval nomiracl \\")
    log.info("      --heldout-dir step5_evaluate/data2/heldout")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 7: NPO (Negative Preference Optimization) baseline training "
                    "(Zhang et al., ICML 2024).")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)

    # NPO-specific hyperparameters (faithful to original config)
    p.add_argument("--beta", type=float, default=0.1,
                   help="Inverse temperature β (default 0.1, from the original TOFU config). "
                        "Controls the log-ratio weighting. Larger β → more aggressive unlearning.")
    p.add_argument("--npo-coeff", type=float, default=1.0,
                   help="Weight on the NPO forget term (default 1.0, from npo_coeff in original).")
    p.add_argument("--lambda-retain", type=float, default=1.0,
                   help="Weight on the retain loss (default 1.0, from grad_diff_coeff / KL_coeff). "
                        "Set to 0.0 to run pure NPO without any retain constraint.")
    p.add_argument("--retain-loss", choices=["npo_grad_diff", "npo_KL"], default="npo_grad_diff",
                   help="Retain loss variant: 'npo_grad_diff' → CE on retain (default, from original); "
                        "'npo_KL' → KL(π_ref ∥ π_θ) on retain (from original npo_KL variant).")

    # Training schedule (mirror UOC defaults)
    p.add_argument("--epochs",       type=int,   default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=cfg.DEFAULT_LR)
    p.add_argument("--forget-batch", type=int,   default=cfg.DEFAULT_FORGET_BATCH)
    p.add_argument("--retain-batch", type=int,   default=cfg.DEFAULT_RETAIN_BATCH)
    p.add_argument("--grad-accum",   type=int,   default=cfg.DEFAULT_GRAD_ACCUM)
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay (default 0.01, from original TOFU config).")

    # LoRA
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha override (default cfg.LORA_ALPHA)")
    p.add_argument("--lora-exclude-last", type=int, default=0,
                   help="Exclude last N layers from LoRA.")

    # Utilities
    p.add_argument("--tag",     type=str, default="",
                   help="Optional tag appended to the run name")
    p.add_argument("--dry-run", action="store_true",
                   help="8 examples, 1 epoch — smoke test")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
