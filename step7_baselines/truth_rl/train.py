#!/usr/bin/env python3
"""TruthRL baseline for the URC project.

Paper: Wei et al. — "TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning"
  arXiv: https://arxiv.org/abs/2509.25760
  Code:  https://github.com/facebookresearch/TruthRL

Algorithm (faithful to Section 3.2 + Appendix A / train_grpo.sh)
-----------------------------------------------------------------
GRPO (Group Relative Policy Optimization) with a ternary reward.

For each training step:
  1. Sample G completions per prompt from the current policy (rollout phase).
  2. Judge each completion with gpt-oss-120b → COMMIT or ABSTAIN.
  3. Compute ternary reward:
       r(x, y) = +1   if x is forget  (unanswerable) AND judge(y) == ABSTAIN
       r(x, y) = -1   if x is forget  (unanswerable) AND judge(y) == COMMIT
       r(x, y) = +1   if x is retain  (answerable)   AND judge(y) == COMMIT
       r(x, y) =  0   if x is retain  (answerable)   AND judge(y) == ABSTAIN
  4. Group advantage: A_i = (r_i − mean(r)) / (std(r) + ε)
  5. GRPO loss with importance-ratio clipping + KL regularisation:
       L = −mean_i [ mean_t [ min(w_it·A_i, clip(w_it, 1−ε, 1+ε)·A_i) ] ]
           + β · KL(π_θ ∥ π_ref)
  6. Update LoRA parameters.

URC adaptation
--------------
  • forget pool  = unanswerable questions (mined COMMIT examples, same as UOC)
  • retain pool  = answerable QA (KUQ/SQuAD answerable, same as UOC)
  • judge        = gpt-oss-120b via Cerebras API (same judge as evaluation)
  • reference π  = frozen base model (LoRA adapters disabled via disable_adapter_layers)
  • LoRA rank/targets identical to all other URC baselines

Hyperparameters (from paper Appendix A)
  G      = 4 rollouts/prompt   (paper: 8; halved for single-GPU memory)
  T      = 1.0  (rollout sampling temperature)
  top_p  = 1.0  (nucleus sampling — paper: top-p=1.0 for rollouts)
  β      = 0.001 (KL coefficient)
  ε      = 0.2   (clip ratio)
  lr     = 1e-6  (paper: 1e-6)
  LoRA dropout = 0.0  (forced; non-zero would corrupt importance ratios)

Run
---
    python step7_baselines/truth_rl/train.py --model qwen_instruct
    python step7_baselines/truth_rl/train.py --model ministral14b_instruct
    python step7_baselines/truth_rl/train.py --model gptoss_instruct
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config as cfg
from _common import (
    Progress,
    _build_generation_input_ids,
    build_answerable_prompt,
    build_unanswerable_prompt,
    format_duration,
    load_jsonl,
    load_model_and_tokenizer,
    log,
    parse_harmony_final,
)
from judge import ABSTAIN, COMMIT, build_judge_prompt, call_judge, make_cerebras_client

RUNS_DIR = Path(__file__).resolve().parent / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── GRPO hyperparameters (from paper Appendix A) ──────────────────────────────

GRPO_G        = 4       # group size (rollouts per prompt); paper uses 8
KL_COEF       = 0.001   # β — KL regularisation coefficient (paper: 0.001)
CLIP_EPS      = 0.2     # ε — importance ratio clip (paper: 0.2)
ADV_EPS       = 1e-8    # numerical stability for advantage normalisation
ROLLOUT_TEMP  = 1.0     # sampling temperature (paper Appendix A: temperature=1.0)
ROLLOUT_TOP_P = 1.0     # top-p for rollouts (paper Appendix A: top-p=1.0)


# ── Model helpers (mirrors step4_train.py) ────────────────────────────────────

def _num_text_layers(model) -> int:
    mcfg = model.config
    tc = getattr(mcfg, "text_config", None)
    return int(getattr(tc if tc is not None else mcfg, "num_hidden_layers"))


def _apply_lora(model, model_key: str, lora_alpha: int | None = None,
                exclude_last: int = 0):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_alpha = cfg.LORA_ALPHA if lora_alpha is None else lora_alpha
    expert_params = cfg.lora_target_parameters(model_key)

    # GRPO requires deterministic log probs for stable importance ratios:
    #   ratio = exp(lp_new - lp_old) must equal 1.0 at initialisation.
    # Non-zero LoRA dropout applies a different random mask each forward pass,
    # breaking this invariant. Force dropout=0.0 for all paths.
    LORA_DROPOUT = 0.0

    if expert_params:
        attn_targets = cfg.lora_attn_targets(model_key)
        lcfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
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
        lcfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_forget(model_key: str) -> list[dict]:
    rows: list[dict] = []
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
            rows.append(r)
    log.info("  forget: %d rows", len(rows))
    return rows


def _load_retain(model_key: str) -> list[dict]:
    rows: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.sampled_answerable_path(dataset)
        if not path.exists():
            log.warning("  retain answerable missing: %s", path)
            continue
        for r in load_jsonl(path):
            r["__type__"] = "retain"
            r["__dataset__"] = dataset
            rows.append(r)
    log.info("  retain: %d rows", len(rows))
    return rows


# ── Rollout generation ────────────────────────────────────────────────────────

def _get_max_new_tokens(model_key: str) -> int:
    """Use the same per-model token budget as the rest of the pipeline."""
    return cfg.max_new_tokens_for(model_key)


@torch.no_grad()
def _sample_rollouts(
    model,
    tokenizer,
    model_key: str,
    prompt: str,
    G: int,
) -> list[tuple[str, torch.Tensor, int, int]]:
    """Generate G sampled completions for a prompt in one batched generate() call.

    Returns list of (completion_text, full_ids_cpu, p_len, n_ans) per rollout.
    full_ids_cpu is (1, p_len+n_ans) — no padding, ready for log-prob computation.
    """
    ids = _build_generation_input_ids(tokenizer, model_key, prompt)
    p_len = len(ids)
    device = next(model.parameters()).device

    # Broadcast the same prompt G times for a single batched decode
    input_ids = torch.tensor([ids], dtype=torch.long, device=device).expand(G, -1)
    attention_mask = torch.ones(G, p_len, dtype=torch.long, device=device)
    pad_id = (getattr(tokenizer, "pad_token_id", None)
              or getattr(tokenizer, "eos_token_id", 0))
    eos_id = getattr(tokenizer, "eos_token_id", pad_id)
    max_tok = _get_max_new_tokens(model_key)

    # Temporarily re-enable KV cache for autoregressive generation.
    # model.config.use_cache was set to False for gradient checkpointing, but
    # generation needs it. Some models gate caching on config, not the arg.
    prev_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.eval()
    try:
        # One batched call: prompt KV-cache computed once, G responses decoded in parallel
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tok,
            do_sample=True,
            temperature=ROLLOUT_TEMP,
            top_p=ROLLOUT_TOP_P,
            pad_token_id=pad_id,
            use_cache=True,
        )
        # out: (G, p_len + actual_or_padded_len)  — padded with pad_id after EOS
    finally:
        model.train()
        model.config.use_cache = prev_use_cache

    prompt_ids_cpu = torch.tensor([ids], dtype=torch.long)  # (1, p_len) on CPU

    rollouts: list[tuple[str, torch.Tensor, int, int]] = []
    for i in range(G):
        new_tok = out[i, p_len:].tolist()
        # Strip right-padding: keep only tokens up to and including the first EOS
        if eos_id in new_tok:
            new_tok = new_tok[: new_tok.index(eos_id) + 1]
        if not new_tok:
            continue
        n_ans = len(new_tok)

        # Reconstruct clean (unpadded) full sequence for log-prob computation
        full_ids_cpu = torch.cat([
            prompt_ids_cpu,
            torch.tensor([new_tok], dtype=torch.long),
        ], dim=1)  # (1, p_len+n_ans)

        raw = tokenizer.decode(new_tok, skip_special_tokens=False)
        if "gptoss" in model_key:
            completion_text = parse_harmony_final(raw) or ""
        else:
            completion_text = tokenizer.decode(new_tok, skip_special_tokens=True).strip()

        rollouts.append((completion_text, full_ids_cpu, p_len, n_ans))

    return rollouts


# ── Reward function ───────────────────────────────────────────────────────────

def _ternary_reward(row_type: str, label: str) -> float:
    """URC-adapted TruthRL ternary reward (Section 3.2).

    forget + ABSTAIN = +1  (correctly abstains on unanswerable)
    forget + COMMIT  = -1  (hallucination on unanswerable)
    retain + COMMIT  = +1  (correctly engages on answerable)
    retain + ABSTAIN =  0  (neutral: over-cautious but not harmful)

    If judge fails → 0 (neutral; no learning signal from broken calls).
    """
    if label == "judge_error":
        return 0.0
    if row_type == "forget":
        return +1.0 if label == ABSTAIN else -1.0
    else:  # retain
        return +1.0 if label == COMMIT else 0.0


# ── Judge batching ────────────────────────────────────────────────────────────

def _judge_batch(
    client,
    items: list[tuple[int, str, str, str | None]],
    max_workers: int = 8,
) -> dict[int, str]:
    """Parallel-judge a list of (idx, question, completion, context) tuples.

    Returns dict[idx] -> label ('COMMIT' | 'ABSTAIN' | 'judge_error').
    """
    results: dict[int, str] = {}

    def _judge_one(args):
        idx, question, completion, context = args
        if not completion:
            return idx, ABSTAIN  # empty response treated as abstain
        jp = build_judge_prompt(question, completion, context)
        label, _ = call_judge(client, jp)
        return idx, label

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, item): item[0] for item in items}
        for fut in as_completed(futures):
            idx, label = fut.result()
            results[idx] = label

    return results


# ── Log-prob computation ──────────────────────────────────────────────────────

def _response_log_probs(
    model,
    full_ids: torch.Tensor,
    p_len: int,
    n_ans: int,
    with_grad: bool = True,
) -> torch.Tensor:
    """Per-token log probs of response tokens [p_len : p_len+n_ans].

    full_ids: (1, p_len+n_ans) on CPU.
    Returns: Tensor shape (n_ans,) on model device (or CPU if no_grad).
    """
    device = next(model.parameters()).device
    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab)

    # Predict token t using logit at position t-1 (causal LM offset)
    response_logits = logits[0, p_len - 1: p_len + n_ans - 1, :]  # (n_ans, V)
    log_probs = F.log_softmax(response_logits, dim=-1)             # (n_ans, V)
    targets = input_ids[0, p_len: p_len + n_ans]                   # (n_ans,)
    return log_probs[torch.arange(n_ans, device=device), targets]  # (n_ans,)


# ── GRPO loss for one prompt group ────────────────────────────────────────────

def _grpo_loss_for_group(
    model,
    rollouts: list[tuple[str, torch.Tensor, int, int]],
    log_probs_old_list: list[torch.Tensor],
    rewards: list[float],
    kl_coef: float,
    clip_eps: float,
) -> torch.Tensor | None:
    """Compute the GRPO loss for one prompt's rollout group.

    Paper equation (train_grpo.sh):
      L = −(1/G) Σ_i (1/|y_i|) Σ_t min(w_it·A_i, clip(w_it, 1−ε, 1+ε)·A_i)
          + β · KL(π_θ ∥ π_ref)

    KL formula: low_var_kl (kl_loss_type=low_var_kl in train_grpo.sh)
      KL ≈ mean_t [ exp(log π_ref(t) − log π_θ(t)) − (log π_ref(t) − log π_θ(t)) − 1 ]
    This is Schulman's lower-variance KL approximation, which equals the true
    KL(π_θ ∥ π_ref) to first order around π_θ ≈ π_ref.
    """
    if len(rollouts) == 0:
        return None

    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    if r_tensor.std() < ADV_EPS:
        # All rewards identical → zero advantage → skip group (no gradient signal)
        return None
    advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + ADV_EPS)

    losses: list[torch.Tensor] = []

    for (_text, full_ids, p_len, n_ans), lp_old, A_i in zip(
        rollouts, log_probs_old_list, advantages.tolist()
    ):
        if n_ans == 0:
            continue

        # π_new log probs (with gradient)
        lp_new = _response_log_probs(model, full_ids, p_len, n_ans, with_grad=True)

        # π_ref log probs (frozen base; no gradient — mirrors NPO reference pattern)
        model.eval()
        model.disable_adapter_layers()
        try:
            lp_ref = _response_log_probs(model, full_ids, p_len, n_ans, with_grad=False)
        finally:
            model.enable_adapter_layers()
            model.train()

        lp_old_dev = lp_old.to(lp_new.device)
        lp_ref_dev = lp_ref.to(lp_new.device)

        # Importance ratio (starts at 1.0; diverges from 1 across gradient updates)
        ratio = (lp_new - lp_old_dev).exp()  # (n_ans,)

        # Clipped surrogate objective (Eq. 3 from paper)
        surr1 = ratio * A_i
        surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * A_i
        policy_loss = -torch.min(surr1, surr2).mean()

        # Low-variance KL approximation (faithful to kl_loss_type=low_var_kl
        # in train_grpo.sh). Schulman formula:
        #   KL(π_θ ∥ π_ref) ≈ exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1
        log_diff = lp_ref_dev - lp_new  # (n_ans,)  log(π_ref / π_θ)
        kl_loss = (log_diff.exp() - log_diff - 1.0).mean()

        losses.append(policy_loss + kl_coef * kl_loss)

    if not losses:
        return None
    return sum(losses) / len(losses)


# ── Checkpoint save ───────────────────────────────────────────────────────────

def _save_adapter(model, out_dir: Path, tokenizer, training_config: dict,
                  train_summary: dict) -> None:
    """Atomically save LoRA adapter: write to .tmp, copy logs, rename."""
    tmp_dir = out_dir.with_suffix(".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    model.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))

    with open(tmp_dir / "training_config.json", "w") as fh:
        json.dump(training_config, fh, indent=2)
    with open(tmp_dir / "train_summary.json", "w") as fh:
        json.dump(train_summary, fh, indent=2)

    # Preserve log files written by the training loop
    for logfile in ("loss_log.csv",):
        src = out_dir / logfile
        if src.exists():
            shutil.copy2(src, tmp_dir / logfile)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    log.info("  adapter saved → %s", out_dir)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    t_total = time.time()

    # ── run name ──────────────────────────────────────────────────────────────
    run_name = (
        f"{args.model}_truthrl"
        f"_g{args.grpo_g}"
        f"_kl{args.kl_coef:g}"
        f"_ep{args.epochs}"
        f"_lr{args.lr:g}"
    )
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== TruthRL | %s ===", run_name)

    # ── load data ─────────────────────────────────────────────────────────────
    forget_data = _load_forget(args.model)
    retain_data = _load_retain(args.model)
    if not forget_data:
        raise RuntimeError("empty forget pool")
    if not retain_data:
        raise RuntimeError("empty retain pool")

    # Balance retain size to forget size (over-sample if needed)
    if len(retain_data) < len(forget_data):
        retain_data = retain_data * math.ceil(len(forget_data) / len(retain_data))
    retain_data = retain_data[: len(forget_data)]

    # Interleave forget + retain so each batch is 50/50
    mixed_data: list[tuple[dict, dict]] = list(zip(forget_data, retain_data))

    # ── load model ────────────────────────────────────────────────────────────
    log.info("Loading model: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)
    model = _apply_lora(model, args.model)

    # Enable gradient checkpointing for all models (memory efficiency)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    # ── judge client ──────────────────────────────────────────────────────────
    judge_client = make_cerebras_client()

    # ── training ──────────────────────────────────────────────────────────────
    training_config: dict = {
        "method": "truth_rl",
        "model": args.model,
        "grpo_g": args.grpo_g,
        "kl_coef": args.kl_coef,
        "clip_eps": args.clip_eps,
        "rollout_temperature": ROLLOUT_TEMP,
        "rollout_top_p": ROLLOUT_TOP_P,
        "max_new_tokens": _get_max_new_tokens(args.model),
        "lr": args.lr,
        "epochs": args.epochs,
        "grad_accum_steps": args.grad_accum,
        "lora_r": cfg.LORA_R,
        "lora_alpha": cfg.LORA_ALPHA,
    }

    n_pairs = len(mixed_data)
    total_steps = args.epochs * n_pairs
    pairs_per_opt_step = args.grad_accum  # one forget+retain pair per micro-step
    n_opt_steps = math.ceil(total_steps / pairs_per_opt_step)
    log.info("  %d pairs × %d epochs = %d micro-steps / %d opt steps",
             n_pairs, args.epochs, total_steps, n_opt_steps)

    log_path = out_dir / "loss_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_csv = csv.writer(log_fh)
    log_csv.writerow(["step", "loss", "n_rollouts_judged", "mean_reward",
                      "mean_advantage_std", "lora_delta_norm", "elapsed_s"])

    global_step = 0
    opt_step = 0
    accum_loss: torch.Tensor | None = None
    accum_count = 0

    # Flatten all (forget, retain) pairs across epochs
    all_pairs: list[tuple[dict, dict]] = []
    for _ in range(args.epochs):
        shuffled = mixed_data[:]
        random.shuffle(shuffled)
        all_pairs.extend(shuffled)

    prog = Progress(len(all_pairs), desc="TruthRL")

    for pair_idx, (fgt_row, ret_row) in enumerate(all_pairs):
        t_step = time.time()
        pair_loss: list[torch.Tensor] = []
        total_rollouts = 0
        total_reward = 0.0
        total_adv_std = 0.0
        n_groups = 0

        for row in (fgt_row, ret_row):
            row_type = row["__type__"]
            ds = row.get("__dataset__", "kuq")
            question = row.get("question", row.get("q", ""))
            context = row.get("context", row.get("passage", None))
            prompt = (build_unanswerable_prompt(ds, row) if row_type == "forget"
                      else build_answerable_prompt(ds, row))
            if not prompt.strip():
                continue

            # ── Phase 1: Rollout ──────────────────────────────────────────────
            rollouts = _sample_rollouts(model, tokenizer, args.model, prompt,
                                        G=args.grpo_g)
            if not rollouts:
                continue

            # ── Phase 2: Compute π_old log probs (no grad) ───────────────────
            lp_old_list: list[torch.Tensor] = []
            for (_text, full_ids, p_len, n_ans) in rollouts:
                lp_old = _response_log_probs(model, full_ids, p_len, n_ans,
                                             with_grad=False)
                lp_old_list.append(lp_old.detach().cpu())

            # ── Phase 3: Judge rollouts in parallel ───────────────────────────
            judge_items = [
                (j, question, text, context)
                for j, (text, _ids, _p, _n) in enumerate(rollouts)
            ]
            label_map = _judge_batch(judge_client, judge_items)
            rewards = [_ternary_reward(row_type, label_map.get(j, "judge_error"))
                       for j in range(len(rollouts))]

            total_rollouts += len(rollouts)
            total_reward += sum(rewards)

            # ── Phase 4: GRPO loss ────────────────────────────────────────────
            r_t = torch.tensor(rewards, dtype=torch.float32)
            if r_t.std() > ADV_EPS:
                total_adv_std += float(r_t.std())
                n_groups += 1
                grp_loss = _grpo_loss_for_group(
                    model, rollouts, lp_old_list, rewards,
                    kl_coef=args.kl_coef, clip_eps=args.clip_eps,
                )
                if grp_loss is not None:
                    pair_loss.append(grp_loss)

        if not pair_loss:
            prog.tick()
            global_step += 1
            continue

        # Average loss over the two groups in this pair
        step_loss = sum(pair_loss) / len(pair_loss)

        if accum_loss is None:
            accum_loss = step_loss / args.grad_accum
        else:
            accum_loss = accum_loss + step_loss / args.grad_accum
        accum_count += 1

        if accum_count >= args.grad_accum or pair_idx == len(all_pairs) - 1:
            accum_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1
            accum_loss = None
            accum_count = 0

        mean_reward = total_reward / max(total_rollouts, 1)
        mean_adv_std = total_adv_std / max(n_groups, 1)
        lora_norm = _mean_lora_delta_norm(model)
        elapsed = time.time() - t_step

        log_csv.writerow([opt_step, float(step_loss.detach()), total_rollouts,
                          f"{mean_reward:.3f}", f"{mean_adv_std:.3f}",
                          f"{lora_norm:.4f}", f"{elapsed:.1f}"])
        log_fh.flush()

        if global_step % 20 == 0:
            log.info(
                "  step %d/%d | loss=%.4f | mean_r=%.3f | lora_δ=%.4f | %s",
                opt_step, n_opt_steps, float(step_loss.detach()),
                mean_reward, lora_norm, format_duration(time.time() - t_total),
            )

        global_step += 1
        prog.tick()

    prog.done()
    log_fh.close()

    # ── save ──────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    train_summary = {
        "run_name": run_name,
        "opt_steps": opt_step,
        "global_steps": global_step,
        "n_forget": len(forget_data),
        "n_retain": len(retain_data),
        "epochs": args.epochs,
        "elapsed_s": elapsed_total,
        "elapsed_human": format_duration(elapsed_total),
    }
    _save_adapter(model, out_dir, tokenizer, training_config, train_summary)
    log.info("=== done in %s ===", format_duration(elapsed_total))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TruthRL GRPO training")
    p.add_argument("--model", required=True,
                   choices=list(cfg.MODEL_REGISTRY),
                   help="Model key (e.g. qwen_instruct)")
    p.add_argument("--epochs", type=int, default=1,
                   help="Number of passes over the training data (default: 1)")
    p.add_argument("--lr", type=float, default=1e-6,
                   help="AdamW learning rate (default: 1e-6, as in paper)")
    p.add_argument("--grpo-g", type=int, default=GRPO_G,
                   help=f"GRPO group size — rollouts per prompt (default: {GRPO_G})")
    p.add_argument("--kl-coef", type=float, default=KL_COEF,
                   help=f"KL regularisation coefficient β (default: {KL_COEF})")
    p.add_argument("--clip-eps", type=float, default=CLIP_EPS,
                   help=f"Importance ratio clip ε (default: {CLIP_EPS})")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
