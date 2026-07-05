#!/usr/bin/env python3
"""R-Tuning baseline for the URC project.

Paper: Zhang et al. — "R-Tuning: Instructing Large Language Models to Say
  'I Don't Know'" (NAACL 2024 Outstanding Paper)
  arXiv: https://arxiv.org/abs/2311.09677
  Code:  https://github.com/shizhediao/R-Tuning

Algorithm (faithful to training/*/run_*.py → method="unknown")
-------------------------------------------------------------
R-Tuning constructs a supervised fine-tuning (SFT) dataset where the model
is *explicitly taught* when to abstain. For each training example:

  1. Run the base model on the question.
  2. If the model gets the answer WRONG → replace the answer with a randomly
     sampled refusal response from a fixed list (FALSE_RESPONSES).
  3. If the model gets the answer RIGHT → keep the original correct answer.
  4. Fine-tune on the resulting mixed dataset with standard next-token CE.

URC adaptation
--------------
Step 2 is already done by step0_mine: the forget pool is exactly R-Tuning's
"questions the model gets wrong" (mined COMMIT completions on unanswerable
prompts). So no pre-training inference step is needed.

  forget examples (WRONG) → pair with refusal response  → SFT teaches ABSTAIN
  retain examples (RIGHT)  → pair with correct answer    → SFT preserves answers
  retain general            → pair with chat response     → SFT preserves utility

Loss: standard cross-entropy next-token SFT on both forget and retain.
      No frozen reference, no preference optimization, no hooks — pure SFT.

FALSE_RESPONSES list (16 entries, directly from run_pararel.py):
  "The answer is unknown.", "The answer is uncertain.", ... (see below)

Run
---
    python step7_baselines/r_tuning/train.py --model qwen_instruct
    python step7_baselines/r_tuning/train.py --model ministral14b_instruct
    python step7_baselines/r_tuning/train.py --model gptoss_instruct
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

# Refusal response pool — directly from run_pararel.py's FALSE_RESPONSES list
# (https://github.com/shizhediao/R-Tuning/blob/main/training/pararel/run_pararel.py)
FALSE_RESPONSES: list[str] = [
    "The answer is unknown.",
    "The answer is uncertain.",
    "The answer is unclear.",
    "It is not known.",
    "I do not know the answer.",
    "I'm not sure.",
    "There is no definitive answer.",
    "There is much debate.",
    "There is no concrete answer to this question.",
    "It is impossible to answer.",
    "There is no known case.",
    "There is no public information available.",
    "There is no scientific evidence.",
    "There is no right answer.",
    "It is impossible to know.",
    "It is difficult to predict.",
]


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
                A = A_dict[ad].weight.detach()
                B = B_dict[ad].weight.detach()
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
    """Load mined COMMIT examples — these are R-Tuning's 'wrong' examples."""
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


# ── Loss functions ─────────────────────────────────────────────────────────────

def _sft_ce_loss(
    model,
    ids: torch.Tensor,
    p_len: int,
    n_ans: int,
) -> torch.Tensor | None:
    """Standard SFT cross-entropy loss over response tokens [p_len : p_len+n_ans].

    This is the only loss in R-Tuning — applied to both forget (with refusal
    response) and retain (with correct answer) examples identically.
    """
    if n_ans == 0 or p_len < 1:
        return None
    logits, _ = forward_hidden_states(model, ids, layer_indices=None)
    # Predict token at position t from logits at position t-1
    logit_slice = logits[0, p_len - 1: p_len + n_ans - 1, :]   # (n_ans, V)
    targets = ids[0, p_len: p_len + n_ans].cpu()                # (n_ans,)
    if targets.shape[0] == 0:
        return None
    return F.cross_entropy(logit_slice, targets)


def _forget_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
) -> torch.Tensor | None:
    """SFT loss on forget example with a randomly sampled refusal response.

    Faithful to R-Tuning's 'unknown' method:
        text = f"Question: {q} Answer: {FALSE_RESPONSES[random_int]}"
    Here we use the model's existing prompt format (unanswerable template)
    and pair with a randomly chosen FALSE_RESPONSE.
    """
    ds = row.get("__dataset__", "kuq")
    prompt = build_unanswerable_prompt(ds, row)
    refusal = random.choice(FALSE_RESPONSES)
    if not prompt.strip():
        return None
    try:
        full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
            tokenizer, model_key, prompt, refusal,
            k_answer_tokens=K_ANSWER_TOKENS,
        )
    except Exception:
        return None
    ids = torch.tensor([full_ids], dtype=torch.long)
    return _sft_ce_loss(model, ids, p_len, n_ans)


def _retain_loss_per_example(
    model, row: dict, tokenizer, model_key: str,
) -> torch.Tensor | None:
    """SFT loss on retain example with the correct answer / chat response.

    Faithful to R-Tuning's treatment of 'correct' examples:
        text = f"Question: {q} Answer: {answer}."
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
    return _sft_ce_loss(model, ids, p_len, n_ans)


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
        "method":            "r_tuning",
        "note":              note,
        "lambda_forget":     args.lambda_forget,
        "lambda_retain":     args.lambda_retain,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "forget_batch":      args.forget_batch,
        "retain_batch":      args.retain_batch,
        "lora_r":            cfg.LORA_R,
        "lora_alpha":        args.lora_alpha if args.lora_alpha else cfg.LORA_ALPHA,
        "lora_exclude_last": args.lora_exclude_last,
        "k_answer_tokens":   K_ANSWER_TOKENS,
        "num_false_responses": len(FALSE_RESPONSES),
    }, indent=2))

    if out_dir.exists():
        for fname in ("loss_log.csv", "train_summary.json"):
            src = out_dir / fname
            if src.exists():
                shutil.copy2(src, tmp / fname)
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
        f"{model_key}_rtuning"
        f"_lf{args.lambda_forget:g}"
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

    # ── Model + LoRA ───────────────────────────────────────────────────────────
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=False)

    model = _apply_lora(model, model_key,
                        lora_alpha=args.lora_alpha,
                        exclude_last=args.lora_exclude_last)
    model.train()

    # Gradient checkpointing for all models (reduces activation memory)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.enable_input_require_grads()
    if cfg.lora_target_parameters(model_key):
        log.info("  gradient checkpointing enabled (fused-expert MoE)")
    else:
        log.info("  gradient checkpointing enabled")

    log.info("  λ_forget=%.2f  λ_retain=%.2f", args.lambda_forget, args.lambda_retain)
    log.info("  %d FALSE_RESPONSES templates", len(FALSE_RESPONSES))

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
        fieldnames=["step", "L_total", "L_forget", "L_retain",
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

    # ── Training loop ──────────────────────────────────────────────────────────
    optim_step   = 0
    accum_forget = torch.tensor(0.0)
    accum_retain = torch.tensor(0.0)

    prog = Progress(total=total_steps, desc="rtuning", log_every=5)
    t0 = time.time()

    for ep in range(args.epochs):
        random.shuffle(forget_data)
        f_pos = 0

        while f_pos < len(forget_data):
            forget_batch = forget_data[f_pos: f_pos + args.forget_batch]
            retain_batch = _next_retain_batch()
            f_pos += args.forget_batch

            # ── L_forget: SFT on forget prompts with refusal responses ────────
            # Faithful to R-Tuning "unknown" method: replace model answer with
            # a random FALSE_RESPONSE, then do standard CE SFT.
            fgt_losses: list[torch.Tensor] = []
            for row in forget_batch:
                loss_ex = _forget_loss_per_example(model, row, tokenizer, model_key)
                if loss_ex is not None:
                    fgt_losses.append(loss_ex)

            # ── L_retain: SFT on retain prompts with correct answers ──────────
            # Faithful to R-Tuning "correct" examples: keep original answer.
            ret_losses: list[torch.Tensor] = []
            if args.lambda_retain > 0:
                for row in retain_batch:
                    loss_ex = _retain_loss_per_example(model, row, tokenizer, model_key)
                    if loss_ex is not None:
                        ret_losses.append(loss_ex)

            l_fgt = (sum(fgt_losses) / len(fgt_losses) if fgt_losses else None)
            l_ret = (sum(ret_losses) / len(ret_losses) if ret_losses else None)

            if l_fgt is None and l_ret is None:
                accum_forget = accum_forget + torch.tensor(0.0)
                accum_retain = accum_retain + torch.tensor(0.0)
            else:
                zero = torch.tensor(0.0)
                lf = l_fgt if l_fgt is not None else zero
                lr_ = l_ret if l_ret is not None else zero

                l_total = args.lambda_forget * lf + args.lambda_retain * lr_
                (l_total / args.grad_accum).backward()

                accum_forget = accum_forget + (l_fgt.detach() if l_fgt is not None else zero)
                accum_retain = accum_retain + (l_ret.detach()  if l_ret  is not None else zero)

            inner = (f_pos // args.forget_batch) % args.grad_accum
            if inner == 0 or f_pos >= len(forget_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.DEFAULT_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                avg_fgt   = float(accum_forget) / max(args.grad_accum, 1)
                avg_ret   = float(accum_retain) / max(args.grad_accum, 1)
                avg_total = args.lambda_forget * avg_fgt + args.lambda_retain * avg_ret
                lr_now    = scheduler.get_last_lr()[0]
                delta_norm = _mean_lora_delta_norm(model)
                elapsed    = time.time() - t0

                csv_writer.writerow({
                    "step":            optim_step,
                    "L_total":         round(avg_total,    6),
                    "L_forget":        round(avg_fgt,      6),
                    "L_retain":        round(avg_ret,      6),
                    "lora_delta_norm": round(delta_norm,   4),
                    "lr":              lr_now,
                    "grad_norm":       round(float(grad_norm), 4),
                    "elapsed_s":       round(elapsed, 1),
                })
                csv_fh.flush()

                prog.tick(extras={
                    "ep":   f"{ep+1}/{args.epochs}",
                    "step": f"{optim_step}/{total_steps}",
                    "L":    f"{avg_total:.4f}",
                    "F":    f"{avg_fgt:.4f}",
                    "R":    f"{avg_ret:.4f}",
                    "|Δ|":  f"{delta_norm:.3f}",
                    "lr":   f"{lr_now:.2e}",
                    "gn":   f"{float(grad_norm):.3f}",
                })

                accum_forget = torch.tensor(0.0)
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

    log.info("R-Tuning done in %s.  Outputs: %s", format_duration(elapsed_total), out_dir)
    log.info("")
    log.info("Evaluate with:")
    log.info("  !python3 step5_evaluate/evaluate.py --run-dir %s \\", out_dir)
    log.info("      --datasets kuq squad selfaware faitheval nomiracl \\")
    log.info("      --heldout-dir step5_evaluate/data2/heldout")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 7: R-Tuning baseline training (Zhang et al., NAACL 2024).")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)

    # Loss weights
    p.add_argument("--lambda-forget", type=float, default=1.0,
                   help="Weight on forget SFT loss — CE on refusal responses (default 1.0).")
    p.add_argument("--lambda-retain", type=float, default=1.0,
                   help="Weight on retain SFT loss — CE on correct answers (default 1.0). "
                        "Set to 0.0 to train on forget-only (ablation).")

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
                   help="Exclude last N layers from LoRA.")

    # Utilities
    p.add_argument("--tag",     type=str, default="",
                   help="Optional tag appended to the run name")
    p.add_argument("--dry-run", action="store_true",
                   help="8 examples, 1 epoch — smoke test")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
