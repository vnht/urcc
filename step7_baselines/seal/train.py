#!/usr/bin/env python3
"""SEAL baseline for the URC project.

Paper: Huang et al. — "Alleviating Hallucinations from Knowledge Misalignment
  in Large Language Models via Selective Abstention Learning"
  ACL 2025 — https://aclanthology.org/2025.acl-long.1199/
  No official code release; this implementation replicates the paper.

Algorithm (Section 3)
----------------------
SEAL adds a special [REJ] token to the vocabulary. The training objective
dynamically shifts probability mass to [REJ] at positions where the model
predicts poorly on the ground-truth token (knowledge misalignment).

Abstention tuning loss (Equations 3-6):
  For each response token position t with ground-truth token y_t:

    α_t = τ · (1 − p_θ(y_t | ctx) / max_w p_θ(w | ctx))
                  ↑ large when model fails to predict y_t

    L_nll_t = −(1−α_t)·log p_θ(y_t) − α_t·log p_θ([REJ])
                  ↑ allows [REJ] to absorb the uncertainty

    L_reg_t = −I_correct · log(1 − p_θ([REJ]))
                  ↑ penalises [REJ] when model can predict y_t correctly

    L_total = mean_t(L_nll_t) + mean_t(L_reg_t)

  α_t is computed from detached probabilities (it is a target coefficient,
  not a differentiable function of θ in the paper's formulation).

Hyperparameters (Appendix E of the paper):
  τ = 0.5   — upper bound for probability shift to [REJ]
  epochs = 3, lr = 5e-6 (paper default, kept as-is for LoRA).
  Paper beam size B=8, penalty λ=1.0 — used in abstention-aware decoding only.

URC adaptation
--------------
  forget pool (COMMIT rows from step0_mine):
    • Prompt:   build_unanswerable_prompt(dataset, row)
    • Response: randomly sampled refusal from FALSE_RESPONSES (same as R-Tuning)
    • Mechanism: model can't predict refusal tokens well → high α_t → [REJ]
                 learns to signal knowledge misalignment

  retain-answerable pool (KUQ/SQuAD answerable):
    • Prompt:   build_answerable_prompt(dataset, row)
    • Response: row["correct_answer"]
    • Mechanism: model knows these → α_t ≈ 0 → essentially vanilla SFT

  retain-general pool (UltraChat):
    • Standard SFT on chat turns (prompt/response pairs)

[REJ] token and LoRA
--------------------
[REJ] is added to the tokenizer vocabulary and the embedding matrices are
resized. The new token embedding is initialised from the mean of common
abstention-related sub-word tokens (e.g. "unknown", "unsure") to give it a
semantically meaningful starting direction.

Embedding layers (embed_tokens, lm_head) are NOT marked trainable to keep the
parameter budget in the LoRA regime. The LoRA parameters learn to modulate
hidden states so that their projection through the fixed [REJ] lm_head row
yields appropriately high/low [REJ] probability — this is the standard LoRA
mechanism applied to a new vocabulary token.

Evaluation note
---------------
After training, if the model generates [REJ] at any position, the raw
completion contains "[REJ]" which the judge will classify as ABSTAIN.
Standard greedy decoding (as used by evaluate.py) is compatible; no
abstention-aware beam search is needed in the evaluation pipeline.

Run
---
    python step7_baselines/seal/train.py --model qwen_instruct
    python step7_baselines/seal/train.py --model ministral14b_instruct
    python step7_baselines/seal/train.py --model gptoss_instruct

    # lr=5e-6 is the paper's default (Appendix E); override if needed:
    python step7_baselines/seal/train.py --model qwen_instruct --lr 3e-5
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
    load_jsonl,
    load_model_and_tokenizer,
    log,
    tokenise_chat_prompt_response,
    tokenise_prompt_plus_answer,
)

RUNS_DIR = Path(__file__).resolve().parent / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

REJ_TOKEN = "[REJ]"
TAU = 0.5           # upper bound for α_t; paper's optimal value (Appendix F.1)
K_ANSWER_TOKENS = cfg.K_ANSWER_TOKENS   # 8, consistent with other baselines

# Refusal response pool — same as R-Tuning (Zhang et al., NAACL 2024)
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

# Tokens used to initialise [REJ] embedding from semantically related content
_ABSTENTION_INIT_WORDS = ["unknown", "unsure", "uncertain", "sorry", "cannot"]


# ── Model helpers ─────────────────────────────────────────────────────────────

def _num_text_layers(model) -> int:
    """Number of transformer layers; multimodal-safe (Ministral wraps text_config)."""
    mcfg = model.config
    tc = getattr(mcfg, "text_config", None)
    return int(getattr(tc if tc is not None else mcfg, "num_hidden_layers"))


def _apply_lora(model, model_key: str, lora_alpha: int | None = None,
                exclude_last: int = 0) -> object:
    """Apply LoRA adapter — mirrors r_tuning._apply_lora exactly.

    Handles both dense models (Qwen, Ministral) via target_modules and the
    fused-expert MoE model (gpt-oss-20b) via target_parameters.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lora_alpha    = cfg.LORA_ALPHA if lora_alpha is None else lora_alpha
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


def _add_rej_token(tokenizer, model) -> int:
    """Add [REJ] to the tokenizer vocabulary and resize model embeddings.

    Initialises the [REJ] embedding from the mean of abstention-related
    sub-word tokens so it starts with a meaningful semantic direction.
    Embedding weights remain frozen (LoRA trains the hidden-state mapping).
    Returns the integer token-id for [REJ].
    """
    # Add special token
    tokenizer.add_tokens([REJ_TOKEN], special_tokens=False)
    rej_id = tokenizer.convert_tokens_to_ids(REJ_TOKEN)
    old_vocab = model.get_input_embeddings().weight.shape[0]

    # Resize both embed_tokens and lm_head
    model.resize_token_embeddings(len(tokenizer))
    new_vocab = model.get_input_embeddings().weight.shape[0]
    log.info("  [REJ] id=%d  vocab %d → %d", rej_id, old_vocab, new_vocab)

    # Initialise from average of abstention-related tokens
    abstention_ids: list[int] = []
    for word in _ABSTENTION_INIT_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        abstention_ids.extend(ids)

    if abstention_ids:
        with torch.no_grad():
            embed_w = model.get_input_embeddings().weight
            avg = embed_w[abstention_ids].mean(dim=0)
            embed_w[rej_id] = avg

            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                avg_lm = lm_head.weight[abstention_ids].mean(dim=0)
                lm_head.weight[rej_id] = avg_lm

        log.info("  [REJ] embedding initialised from %d abstention tokens",
                 len(abstention_ids))
    else:
        log.warning("  could not find abstention init tokens; [REJ] uses random init")

    return rej_id


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

def _build_mixed_dataset(model_key: str) -> list[dict]:
    """Build the unified training dataset (same structure as R-Tuning).

    forget  → prompt + refusal target  (high α_t → [REJ] learns uncertainty)
    retain  → prompt + correct answer  (low α_t  → standard SFT signal)
    general → UltraChat prompt/response
    """
    mixed: list[dict] = []

    for dataset in ("kuq", "squad"):
        path = cfg.forget_path(model_key, dataset)
        if not path.exists():
            log.warning("  forget pool missing: %s", path)
            continue
        for r in load_jsonl(path):
            if (r.get("judge_label") or "").upper() not in ("COMMIT", "COMMITTED"):
                continue
            r["__kind__"] = "forget"
            r["__dataset__"] = dataset
            mixed.append(r)

    for dataset in ("kuq", "squad"):
        path = cfg.sampled_answerable_path(dataset)
        if not path.exists():
            log.warning("  retain answerable missing: %s", path)
            continue
        for r in load_jsonl(path):
            r["__kind__"] = "retain"
            r["__dataset__"] = dataset
            mixed.append(r)

    gen_path = cfg.sampled_general_path()
    if gen_path.exists():
        for r in load_jsonl(gen_path):
            r["__kind__"] = "general"
            mixed.append(r)
    else:
        log.warning("  UltraChat missing: %s", gen_path)

    n_forget  = sum(1 for r in mixed if r["__kind__"] == "forget")
    n_retain  = sum(1 for r in mixed if r["__kind__"] == "retain")
    n_general = sum(1 for r in mixed if r["__kind__"] == "general")
    log.info("  mixed dataset: %d forget + %d retain + %d general = %d total",
             n_forget, n_retain, n_general, len(mixed))
    return mixed


# ── SEAL loss ──────────────────────────────────────────────────────────────────

def _seal_loss(
    model,
    ids: torch.Tensor,
    p_len: int,
    n_ans: int,
    rej_id: int,
    tau: float,
) -> tuple[torch.Tensor, float, float] | None:
    """Compute SEAL abstention-tuning loss for one tokenised example.

    Equations 3–6 from the paper.

    ids:   (1, p_len + n_ans)  long tensor
    p_len: number of prompt tokens
    n_ans: number of answer tokens to supervise
    rej_id: token id of [REJ]
    tau:   upper bound for α_t (paper: 0.5)
    """
    if n_ans == 0 or p_len < 1:
        return None

    device = next(model.parameters()).device
    ids_dev = ids.to(device)

    outputs = model(
        input_ids=ids_dev,
        attention_mask=torch.ones_like(ids_dev),
    )
    logits = outputs.logits   # (1, seq_len, vocab)

    # Shift: position p_len-1 predicts the first answer token, etc.
    resp_logits  = logits[0, p_len - 1 : p_len + n_ans - 1, :]  # (n, V)
    resp_targets = ids_dev[0, p_len : p_len + n_ans]             # (n,)

    n = resp_targets.shape[0]
    if n == 0:
        return None

    # ── Compute α_t with detached probabilities (target coefficient, not grad) ──
    with torch.no_grad():
        det_probs = F.softmax(resp_logits.detach().float(), dim=-1)   # (n, V)
        p_yt   = det_probs[torch.arange(n, device=device), resp_targets]  # (n,)
        p_max  = det_probs.max(dim=-1).values                              # (n,)
        alpha  = (tau * (1.0 - p_yt / (p_max + 1e-8))).clamp(0.0, tau)    # Eq. 3
        i_corr = (det_probs.argmax(dim=-1) == resp_targets).float()        # I_correct

    # ── Differentiable log-probs for loss computation ──
    log_probs = F.log_softmax(resp_logits.float(), dim=-1)                # (n, V)
    log_p_yt  = log_probs[torch.arange(n, device=device), resp_targets]  # (n,)
    log_p_rej = log_probs[:, rej_id]                                      # (n,)

    # L_nll (Equation 4)
    l_nll = -((1.0 - alpha) * log_p_yt + alpha * log_p_rej).mean()

    # L_reg (Equation 5) — penalise [REJ] when correct prediction is feasible
    p_rej_grad = log_p_rej.exp()
    log_1_mp   = torch.log((1.0 - p_rej_grad).clamp(min=1e-8))
    l_reg      = -(i_corr * log_1_mp).mean()

    return l_nll + l_reg, l_nll.detach().item(), l_reg.detach().item()


def _seal_loss_for_example(
    model,
    row: dict,
    tokenizer,
    model_key: str,
    rej_id: int,
    tau: float,
) -> tuple[torch.Tensor, float, float] | None:
    """Dispatch one training example to _seal_loss.

    Returns (L_total, l_nll_val, l_reg_val) or None if the example is skipped.
    """
    kind = row.get("__kind__", "forget")

    if kind == "forget":
        ds     = row.get("__dataset__", "kuq")
        prompt = build_unanswerable_prompt(ds, row)
        resp   = random.choice(FALSE_RESPONSES)
        if not prompt.strip():
            return None
        try:
            full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
                tokenizer, model_key, prompt, resp,
                k_answer_tokens=K_ANSWER_TOKENS,
            )
        except Exception:
            return None

    elif kind == "retain":
        ds     = row.get("__dataset__", "kuq")
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
        prompt = row.get("prompt") or ""
        resp   = row.get("response") or ""
        if not prompt.strip() or not resp.strip():
            return None
        try:
            full_ids, p_len = tokenise_chat_prompt_response(
                tokenizer, model_key, prompt, resp,
            )
        except Exception:
            return None
        n_ans = min(K_ANSWER_TOKENS, max(0, len(full_ids) - p_len))

    else:
        return None

    if n_ans == 0 or p_len < 1:
        return None

    ids = torch.tensor([full_ids], dtype=torch.long)
    return _seal_loss(model, ids, p_len, n_ans, rej_id=rej_id, tau=tau)


# ── LR scheduler (mirrors step4_train) ────────────────────────────────────────

def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Save helper ────────────────────────────────────────────────────────────────

def _save_adapter(out_dir: Path, model, tokenizer, model_key: str,
                  args: argparse.Namespace, rej_id: int, note: str = "") -> None:
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
        "model_key":           model_key,
        "method":              "seal",
        "note":                note,
        "epochs":              args.epochs,
        "lr":                  args.lr,
        "batch_size":          args.batch_size,
        "grad_accum":          args.grad_accum,
        "lora_r":              cfg.LORA_R,
        "lora_alpha":          args.lora_alpha if args.lora_alpha else cfg.LORA_ALPHA,
        "lora_exclude_last":   args.lora_exclude_last,
        "k_answer_tokens":     K_ANSWER_TOKENS,
        "tau":                 args.tau,
        "rej_token_id":        rej_id,
        "rej_token":           REJ_TOKEN,
        "paper":               "Huang et al., ACL 2025",
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
    mixed_data = _build_mixed_dataset(model_key)
    if not mixed_data:
        raise RuntimeError("Mixed dataset empty. Check step0_mine data.")

    if args.dry_run:
        mixed_data = mixed_data[:16]
        args.epochs = 1
        log.info("  DRY RUN: 16 examples, 1 epoch")

    # ── Run name ──────────────────────────────────────────────────────────────
    run_name = (
        f"{model_key}_seal"
        f"_tau{args.tau:g}"
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

    # ── Model + LoRA + [REJ] token ─────────────────────────────────────────────
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=False)

    # Add [REJ] BEFORE LoRA so resize_token_embeddings sets the correct vocab size
    rej_id = _add_rej_token(tokenizer, model)

    model = _apply_lora(model, model_key,
                        lora_alpha=args.lora_alpha,
                        exclude_last=args.lora_exclude_last)
    model.train()

    # Gradient checkpointing
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.enable_input_require_grads()
    log.info("  gradient checkpointing enabled")
    log.info("  τ=%.2f  |  [REJ] token id=%d", args.tau, rej_id)

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    log.info("  AdamW lr=%g  weight_decay=%g", args.lr, args.weight_decay)

    steps_per_epoch = math.ceil(len(mixed_data) / args.batch_size)
    total_steps = math.ceil(steps_per_epoch / args.grad_accum) * args.epochs
    warmup = max(1, int(total_steps * cfg.DEFAULT_WARMUP_RATIO))
    scheduler = _linear_warmup_decay(optimizer, warmup, total_steps)
    log.info("  total_steps=%d  warmup=%d  epochs=%d", total_steps, warmup, args.epochs)

    # ── CSV log ────────────────────────────────────────────────────────────────
    csv_path = out_dir / "loss_log.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_fh,
        fieldnames=["step", "L_seal", "L_nll", "L_reg",
                    "lora_delta_norm", "lr", "grad_norm", "elapsed_s"],
    )
    csv_writer.writeheader()

    # ── Training loop ──────────────────────────────────────────────────────────
    # Shuffle each epoch, then iterate in micro-batches. Each example produces
    # a scalar SEAL loss (L_nll + L_reg); losses are averaged per batch before
    # backprop, matching the paper's per-token mean formulation.
    optim_step = 0
    accum_loss = 0.0
    accum_nll  = 0.0
    accum_reg  = 0.0

    prog = Progress(total=total_steps, desc="seal", log_every=5)
    t0   = time.time()

    for ep in range(args.epochs):
        random.shuffle(mixed_data)
        pos = 0

        while pos < len(mixed_data):
            batch = mixed_data[pos : pos + args.batch_size]
            pos  += args.batch_size

            batch_losses:   list[torch.Tensor] = []
            batch_nll_vals: list[float]         = []
            batch_reg_vals: list[float]         = []
            for row in batch:
                result = _seal_loss_for_example(
                    model, row, tokenizer, model_key,
                    rej_id=rej_id, tau=args.tau,
                )
                if result is not None:
                    l_total, l_nll_val, l_reg_val = result
                    batch_losses.append(l_total)
                    batch_nll_vals.append(l_nll_val)
                    batch_reg_vals.append(l_reg_val)

            if batch_losses:
                n_ex   = len(batch_losses)
                l_seal = sum(batch_losses) / n_ex
                (l_seal / args.grad_accum).backward()
                accum_loss += l_seal.detach().item()
                accum_nll  += sum(batch_nll_vals) / n_ex
                accum_reg  += sum(batch_reg_vals) / n_ex

            inner = (pos // args.batch_size) % args.grad_accum
            if inner == 0 or pos >= len(mixed_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.DEFAULT_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                denom      = max(args.grad_accum, 1)
                avg_loss   = accum_loss / denom
                avg_nll    = accum_nll  / denom
                avg_reg    = accum_reg  / denom
                lr_now     = scheduler.get_last_lr()[0]
                delta_norm = _mean_lora_delta_norm(model)
                elapsed    = time.time() - t0

                csv_writer.writerow({
                    "step":            optim_step,
                    "L_seal":          round(avg_loss,   6),
                    "L_nll":           round(avg_nll,    6),
                    "L_reg":           round(avg_reg,    6),
                    "lora_delta_norm": round(delta_norm, 4),
                    "lr":              lr_now,
                    "grad_norm":       round(float(grad_norm), 4),
                    "elapsed_s":       round(elapsed, 1),
                })
                csv_fh.flush()

                prog.tick(extras={
                    "ep":    f"{ep+1}/{args.epochs}",
                    "step":  f"{optim_step}/{total_steps}",
                    "L":     f"{avg_loss:.4f}",
                    "nll":   f"{avg_nll:.4f}",
                    "reg":   f"{avg_reg:.4f}",
                    "|Δ|":   f"{delta_norm:.3f}",
                    "lr":    f"{lr_now:.2e}",
                    "gn":    f"{float(grad_norm):.3f}",
                })

                accum_loss = 0.0
                accum_nll  = 0.0
                accum_reg  = 0.0

    csv_fh.close()
    prog.done()

    # ── Save ───────────────────────────────────────────────────────────────────
    _save_adapter(out_dir, model, tokenizer, model_key, args, rej_id=rej_id)

    elapsed_total = time.time() - t0_total
    (out_dir / "train_summary.json").write_text(json.dumps({
        "num_steps":   optim_step,
        "elapsed_s":   round(elapsed_total, 1),
        "elapsed":     format_duration(elapsed_total),
        "rej_token":   REJ_TOKEN,
        "rej_id":      rej_id,
        "tau":         args.tau,
    }, indent=2))

    log.info("SEAL done in %s.  Outputs: %s", format_duration(elapsed_total), out_dir)
    log.info("")
    log.info("Evaluate with:")
    log.info("  !python3 step5_evaluate/evaluate.py --run-dir %s \\", out_dir)
    log.info("      --datasets kuq squad selfaware faitheval nomiracl \\")
    log.info("      --heldout-dir step5_evaluate/data2/heldout")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 7: SEAL baseline training (Huang et al., ACL 2025).")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)

    # SEAL-specific
    p.add_argument("--tau", type=float, default=TAU,
                   help=("Upper bound for [REJ] probability shift α_t. "
                         "Paper: 0.5 (default)."))

    # Training schedule
    p.add_argument("--epochs",       type=int,   default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=5e-6,
                   help=("Learning rate. Paper (Appendix E): 5e-6 (default). "
                         "Full-FT value kept as default; LoRA runs at this rate "
                         "to match the paper's training regime as closely as possible."))
    p.add_argument("--batch-size",   type=int,   default=cfg.DEFAULT_FORGET_BATCH,
                   help="Examples per gradient-accumulation micro-step.")
    p.add_argument("--grad-accum",   type=int,   default=cfg.DEFAULT_GRAD_ACCUM)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha override (default cfg.LORA_ALPHA).")
    p.add_argument("--lora-exclude-last", type=int, default=0,
                   help="Exclude last N layers from LoRA.")

    # Utilities
    p.add_argument("--tag",     type=str, default="",
                   help="Optional tag appended to the run name.")
    p.add_argument("--dry-run", action="store_true",
                   help="16 examples, 1 epoch — smoke test.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
