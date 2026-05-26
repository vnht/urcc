#!/usr/bin/env python3
"""Step 4 — UOC training (two-component subspace-anchored loss).

Method
------
Two losses, both of the form  ‖ V_lᵀ (h_l - target) ‖²   averaged over
late layers and answer-token positions. The forget loss *changes* the
representation of over-commitment (category A) by pulling it toward the
legitimate-abstention pole μ⁻(d). The retain loss *preserves* the
representation of legitimate commitment (C) and general utility (E) by
anchoring each example, per token, to its frozen-base activation. The
two operations have clearly opposite roles — change vs. preserve — and
both are per-example specific (not pulled toward an averaged anchor).

    L_forget = E_{(x,y) ∈ D_F}             ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ) − μ_l⁻(d_x)         ) ‖² ⟩_{l, t ∈ T(x)}
    L_retain = E_{(x,y) ∈ D_R_A ∪ D_R_G}   ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ) − h_l(x, y; θ_frozen)) ‖² ⟩_{l, t ∈ T(x)}

    L = L_forget + λ · L_retain

μ⁻(d) is per answerability domain d ∈ {kuq, squad} so the forget target
lives in the same prompt distribution (KUQ no-context vs SQuAD with-
context) as the example being trained. The discriminative subspace V is
shared across domains.

For general-utility examples (UltraChat — no domain), preservation is
measured in *both* subspaces and averaged: any direction the forget pull
acts on (V_kuq or V_squad) must be preserved on retain-general inputs.

The answer-token window T(x) starts one position *before* the first answer
token: T(x) = {p_len-1, p_len, …, p_len+K-2}. Position p_len-1 is the
prompt-final residual stream from which the LM head decides the first
generated token. Including it in the retain side means the frozen-base
reference protects each individual answerable / general-utility input's
first-token decision, per-example — which is what prevents the LoRA from
finding degenerate solutions that collapse first-token logits to a
chat-end token on retain inputs.

μ⁺(d) is computed in step 3 and saved in the anchors bundle but is no
longer a training target — it is kept as a geometric *diagnostic*
confirming V meaningfully separates the legit-commit cluster from the
legit-abstain cluster.

LoRA on `{q,k,v,o,up,down,gate}` projections of the late-layer set; base
weights frozen.

Reads:  step0_mine/data/forget/<model>_<dataset>.jsonl
        step0_mine/data/sampled/{kuq,squad}_answerable.jsonl, ultrachat.jsonl
        step2_build_subspace/data/subspace_<model>_r<rank>.pt
        step3_build_anchors/data/anchors_<model>.pt
Writes: step4_train/data/runs/<run_name>/{adapter, loss_log.csv,
        training_config.json, train_summary.json, subspace_config.json}

Run
---
    python step4_train/train.py --model qwen_instruct --rank 32 \
        --lambda-retain 1.0 --epochs 3 --lr 3e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_forget(model_key: str) -> list[dict]:
    """Forget pool D_F = COMMIT-labeled rows (category A — over-commitment)
    from step0_mine/data/forget/<model>_<dataset>.jsonl"""
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.forget_path(model_key, dataset)
        if not path.exists():
            log.warning("  D_F (forget) pool missing: %s", path)
            continue
        for r in load_jsonl(path):
            label = (r.get("judge_label") or "").upper()
            if label not in ("COMMIT", "COMMITTED"):
                continue
            r["__type__"] = "forget"
            r["__dataset__"] = dataset
            pool.append(r)
    log.info("  D_F (over-commitment): %d examples", len(pool))
    return pool


def _load_retain(model_key: str) -> list[dict]:
    """Retain pool D_R_A ∪ D_R_G = answerable QA (category C) + UltraChat
    (category E). Each row is tagged with __type__ ('answerable' | 'general')."""
    pool: list[dict] = []
    for dataset in ("kuq", "squad"):
        path = cfg.sampled_answerable_path(dataset)
        if not path.exists():
            log.warning("  sampled answerable missing: %s", path)
            continue
        for r in load_jsonl(path):
            r["__type__"] = "answerable"
            r["__dataset__"] = dataset
            pool.append(r)

    gpath = cfg.sampled_general_path()
    if gpath.exists():
        for r in load_jsonl(gpath):
            r["__type__"] = "general"
            pool.append(r)
    else:
        log.warning("  sampled general missing: %s", gpath)

    n_ans = sum(1 for r in pool if r["__type__"] == "answerable")
    n_gen = sum(1 for r in pool if r["__type__"] == "general")
    log.info("  retain pool: %d  (D_R_A=%d legitimate-commit + D_R_G=%d general-utility)",
             len(pool), n_ans, n_gen)
    random.shuffle(pool)
    return pool


def _layer_split(t: torch.Tensor) -> list[torch.Tensor]:
    """Slice a (L, D) tensor into a list of L float (D,) tensors."""
    return [t[i].float() for i in range(t.shape[0])]


def _load_subspace_and_anchors(model_key: str, rank: int):
    """Returns (V_layers, layer_indices, mu_minus_per, mu_plus_per, init_scale).

    V_layers is the shared discriminative subspace from step 2 (one V for
    both domains): list of L float tensors of shape (D, r).

    mu_minus_per is the per-domain forget pole, used by L_forget.
    mu_plus_per  is loaded for backward-compat / diagnostic but is **no longer
                 a training target** — L_retain uses the frozen-base forward as
                 its per-example, per-token reference (see _compute_retain_loss).

    Both are dicts keyed by dataset:
        {"kuq": [tensor(D), …L layers…], "squad": [tensor(D), …L layers…]}

    For backwards-compat with old anchor bundles (no per-domain poles): the
    grand mean is replicated under both keys.
    """
    sp = cfg.subspace_path(model_key, rank=rank)
    ap = cfg.anchors_path(model_key)
    if not sp.exists():
        raise FileNotFoundError(f"Subspace bundle not found: {sp}. Run step 2.")
    if not ap.exists():
        raise FileNotFoundError(f"Anchors bundle not found: {ap}. Run step 3.")

    sb = torch.load(sp, map_location="cpu", weights_only=False)
    ab = torch.load(ap, map_location="cpu", weights_only=False)

    if sb["layers"] != ab["layers"]:
        raise RuntimeError(
            f"Subspace layers {sb['layers']} != anchor layers {ab['layers']}"
        )

    V = sb["V"]                     # (L, D, r)
    V_layers = [V[i].float() for i in range(V.shape[0])]

    minus_per_t = ab.get("mu_minus_per") or {"kuq": ab["mu_minus"], "squad": ab["mu_minus"]}
    plus_per_t  = ab.get("mu_plus_per")  or {"kuq": ab["mu_plus"],  "squad": ab["mu_plus"]}
    mu_minus_per = {d: _layer_split(t) for d, t in minus_per_t.items()}
    mu_plus_per  = {d: _layer_split(t) for d, t in plus_per_t.items()}

    if "mu_minus_per" not in ab:
        log.warning("  anchors bundle has no per-domain poles; falling back to grand mean. "
                    "Re-run step 3 to build per-domain anchors.")

    # diag.OC_proj averages roughly the initial value of L_forget per layer ⇒
    # use it to scale both losses so they start at O(1). Constant scaling does
    # not change the optimum or the relative weight λ; it only affects the
    # effective learning rate.
    init_scale = 1.0
    if sb.get("diag"):
        oc_means = [d["OC_proj"] for d in sb["diag"]]
        init_scale = max(sum(oc_means) / len(oc_means), 1e-6)

    return V_layers, sb["layers"], mu_minus_per, mu_plus_per, float(init_scale)


# ── LoRA ──────────────────────────────────────────────────────────────────────

def _apply_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model
    lcfg = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


# ── Loss components (always the same operation: ‖ Vᵀ(h - target) ‖²) ───────────

def _project_to_pole(
    h_seq: torch.Tensor,                      # (1, seq_len, D)
    span: tuple[int, int],                    # (start, end) in seq dim
    V_l: torch.Tensor,                        # (D, r)
    target: torch.Tensor,                     # (D,) constant pole or (n_ans, D) per-token
) -> torch.Tensor:
    """Returns mean over span of ||Vᵀ(h_t - target_t)||²."""
    s, e = span
    h = h_seq[0, s:e, :]                      # (n_ans, D)
    if target.dim() == 1:
        h = h - target.to(h.device).to(h.dtype)
    else:
        h = h - target.to(h.device).to(h.dtype)
    proj = h @ V_l.to(h.device).to(h.dtype)   # (n_ans, r)
    return (proj ** 2).sum(dim=-1).mean()


def _compute_forget_loss(
    *, model, batch: list[dict], tokenizer, model_key: str,
    V_layers, layer_indices, mu_minus_per: dict, k_answer_tokens: int,
) -> tuple[torch.Tensor, dict]:
    """L_forget — pull category-A activations toward μ⁻(d) along V, where d is
    the example's source dataset (kuq | squad)."""
    total = torch.tensor(0.0, requires_grad=True)
    layer_norms: list[float] = []
    n_used = 0
    by_dataset = {d: 0 for d in mu_minus_per.keys()}
    for r in batch:
        ds = r["__dataset__"]
        mu_minus = mu_minus_per.get(ds)
        if mu_minus is None:
            continue
        prompt = build_unanswerable_prompt(ds, r)
        answer = r.get("y_com_prefix_k8") or r.get("full_completion_clean") or ""
        if not prompt.strip() or not answer.strip():
            continue
        try:
            full_ids, p_len, n_ans = tokenise_prompt_plus_answer(
                tokenizer, prompt, answer, k_answer_tokens=k_answer_tokens,
            )
        except Exception:
            continue
        if n_ans == 0:
            continue

        ids = torch.tensor([full_ids], dtype=torch.long)
        _, hiddens = forward_hidden_states(model, ids, layer_indices)

        # Window starts at p_len - 1 (prompt-final state) so the loss anchors
        # the first-token-decision residual stream.
        if p_len < 1:
            continue
        span = (p_len - 1, p_len - 1 + n_ans)
        per_ex = torch.tensor(0.0, requires_grad=True)
        for li, h in enumerate(hiddens):
            l_loss = _project_to_pole(
                h, span, V_layers[li], mu_minus[li],
            )
            per_ex = per_ex + l_loss
            layer_norms.append(float(l_loss.detach().sqrt()))
        per_ex = per_ex / len(hiddens)
        total = total + per_ex
        n_used += 1
        by_dataset[ds] = by_dataset.get(ds, 0) + 1

    if n_used > 0:
        total = total / n_used
    mean_norm = sum(layer_norms) / len(layer_norms) if layer_norms else 0.0
    return total, {"n": n_used, "proj_norm": mean_norm, **by_dataset}


def _compute_retain_loss(
    *, model, batch: list[dict], tokenizer, model_key: str,
    V_layers, layer_indices, k_answer_tokens: int,
) -> tuple[torch.Tensor, dict]:
    """L_retain — preserve each retain example at its frozen-base location along V.

    For both branches (D_R_A category C and D_R_G category E), the target is
    h_l(x, y; θ_frozen) per-token: the model's own current activation under
    the frozen base on (x, y). The window is the same K-token transition
    window starting at the prompt-final position. This is per-example and
    per-token, so the LoRA pays a sharp price for any drift on a specific
    retain input — unlike a shared scalar pole anchor where many retain
    examples can collectively move together while the loss only measures
    the cluster's variance.

    μ⁺(d) is no longer used in training; it is kept in the anchors bundle as a
    geometric diagnostic confirming V separates legit-commit from legit-abstain.
    """
    total = torch.tensor(0.0, requires_grad=True)
    n_used = 0
    breakdown = {"answerable": 0, "general": 0}

    for r in batch:
        kind = r["__type__"]
        if kind == "answerable":
            ds = r["__dataset__"]
            prompt = build_answerable_prompt(ds, r)
            answer = r.get("correct_answer") or ""
            if not prompt.strip() or not answer.strip():
                continue
            try:
                full_ids, resp_start, n_ans = tokenise_prompt_plus_answer(
                    tokenizer, prompt, answer, k_answer_tokens=k_answer_tokens,
                )
            except Exception:
                continue
        elif kind == "general":
            prompt   = r.get("prompt") or ""
            response = r.get("response") or ""
            if not prompt.strip() or not response.strip():
                continue
            try:
                full_ids, resp_start = tokenise_chat_prompt_response(
                    tokenizer, model_key, prompt, response,
                )
            except Exception:
                continue
            n_ans = min(k_answer_tokens, max(0, len(full_ids) - resp_start))
        else:
            continue

        if n_ans == 0 or resp_start < 1:
            continue

        ids = torch.tensor([full_ids], dtype=torch.long)

        # Frozen reference (adapters disabled, no_grad)
        model.eval()
        model.disable_adapter_layers()
        with torch.no_grad():
            _, ref_hiddens = forward_hidden_states(model, ids, layer_indices)
        model.enable_adapter_layers()
        model.train()

        # Trainable forward
        _, hiddens = forward_hidden_states(model, ids, layer_indices)

        # Window includes resp_start - 1 (prompt-final state) so the loss
        # protects first-token-decision residual stream per-example.
        span = (resp_start - 1, resp_start - 1 + n_ans)
        per_ex = torch.tensor(0.0, requires_grad=True)
        for li, h in enumerate(hiddens):
            ref_h = ref_hiddens[li][0, span[0]: span[1], :]
            per_ex = per_ex + _project_to_pole(
                h, span, V_layers[li], ref_h,
            )
        per_ex = per_ex / len(hiddens)
        total = total + per_ex
        breakdown[kind if kind == "general" else "answerable"] += 1
        n_used += 1

    if n_used > 0:
        total = total / n_used
    return total, {"n": n_used, **breakdown}


# ── LR scheduler ──────────────────────────────────────────────────────────────

def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
    return LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

CKPT_SUBDIR = "checkpoint"
TRAINER_STATE_FILE = "trainer_state.pt"


def _save_checkpoint(out_dir: Path, model, optimizer, scheduler,
                     step: int, summary_initial: dict) -> None:
    """Save adapter + optim/scheduler state to <run_dir>/checkpoint/.

    Atomic w.r.t. interrupts: writes to a sibling .tmp dir then renames.
    """
    final = out_dir / CKPT_SUBDIR
    tmp   = out_dir / (CKPT_SUBDIR + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save({
        "step":             step,
        "optimizer":        optimizer.state_dict(),
        "scheduler":        scheduler.state_dict(),
        "summary_initial":  summary_initial,
    }, tmp / TRAINER_STATE_FILE)
    if final.exists():
        import shutil
        shutil.rmtree(final)
    tmp.rename(final)


def _load_checkpoint_if_any(out_dir: Path):
    """Return (step, optimizer_state, scheduler_state, summary_initial) or None."""
    ckpt = out_dir / CKPT_SUBDIR / TRAINER_STATE_FILE
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    return state


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    pipeline_t0 = time.time()
    model_key = args.model

    # Load all the prerequisites
    forget_data = _load_forget(model_key)
    retain_data = _load_retain(model_key)
    if not forget_data:
        raise RuntimeError("Forget pool is empty.")
    if not retain_data:
        raise RuntimeError("Retain pool is empty.")

    V_layers, layer_indices, mu_minus_per, mu_plus_per, init_scale = \
        _load_subspace_and_anchors(model_key, rank=args.rank)
    log.info("  V layers=%s  rank=%d  init_scale=%.2f", layer_indices, args.rank, init_scale)
    log.info("  forget pole μ⁻ keys=%s   "
             "(μ⁺ kept as diagnostic only; retain uses frozen-base reference)",
             list(mu_minus_per.keys()))
    _ = mu_plus_per  # diagnostic; not used by L_retain (frozen-ref preservation instead)

    if args.dry_run:
        forget_data = forget_data[:8]
        retain_data = retain_data[:8]
        log.info("  DRY RUN: trimmed to 8 examples each")

    run_name = (
        f"{model_key}_uoc_r{args.rank}"
        f"_lam{args.lambda_retain:g}_ep{args.epochs}_lr{args.lr:.0e}"
    )
    out_dir = cfg.RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run name: %s", run_name)
    log.info("Output dir: %s", out_dir)

    # Model + LoRA (resume from checkpoint adapter if present)
    ckpt_state = _load_checkpoint_if_any(out_dir)
    resume = ckpt_state is not None
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=False)
    if resume:
        from peft import PeftModel
        log.info("  resuming from checkpoint: %s", out_dir / CKPT_SUBDIR)
        model = PeftModel.from_pretrained(model, str(out_dir / CKPT_SUBDIR),
                                          is_trainable=True)
        model.print_trainable_parameters()
    else:
        model = _apply_lora(model)
    model.train()

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    f_steps_per_epoch = math.ceil(len(forget_data) / args.forget_batch)
    total_optim_steps = math.ceil(f_steps_per_epoch / args.grad_accum) * args.epochs
    warmup = max(1, int(total_optim_steps * cfg.DEFAULT_WARMUP_RATIO))
    scheduler = _linear_warmup_decay(optimizer, warmup, total_optim_steps)

    start_step = 0
    summary_initial: dict = {}
    if resume:
        try:
            optimizer.load_state_dict(ckpt_state["optimizer"])
            scheduler.load_state_dict(ckpt_state["scheduler"])
            start_step = int(ckpt_state.get("step", 0))
            summary_initial = ckpt_state.get("summary_initial", {}) or {}
            log.info("  resumed at optim_step=%d", start_step)
        except Exception as exc:
            log.warning("  failed to load optim/scheduler state (%s); restarting fresh", exc)
            start_step = 0

    if start_step >= total_optim_steps:
        log.info("  checkpoint already at final step (%d / %d). Nothing to do.",
                 start_step, total_optim_steps)
        log.info("STEP 4 done in %s", format_duration(time.time() - pipeline_t0))
        return

    log.info("  optim steps: %d  (warmup %d)  forget batches/epoch: %d",
             total_optim_steps, warmup, f_steps_per_epoch)

    # CSV log — append on resume so prior history is preserved
    log_path = out_dir / "loss_log.csv"
    log_fields = ["step", "L_total", "L_forget", "L_retain",
                  "mean_proj_norm", "learning_rate", "grad_norm",
                  "step_time_s", "elapsed_s"]
    is_new_log = not log_path.exists() or not resume
    log_file = open(log_path, "a" if resume and log_path.exists() else "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if is_new_log:
        log_writer.writeheader()
        log_file.flush()

    # Retain cycle
    retain_cycle = retain_data.copy()
    random.shuffle(retain_cycle)
    retain_pos = 0

    def next_retain_batch() -> list[dict]:
        nonlocal retain_cycle, retain_pos
        out = []
        for _ in range(args.retain_batch):
            if retain_pos >= len(retain_cycle):
                retain_cycle = retain_data.copy()
                random.shuffle(retain_cycle)
                retain_pos = 0
            out.append(retain_cycle[retain_pos])
            retain_pos += 1
        return out

    optim_step = start_step
    accum_forget = torch.tensor(0.0)
    accum_retain = torch.tensor(0.0)
    accum_proj: list[float] = []
    summary_final: dict = {}

    progress = Progress(total=total_optim_steps - start_step,
                        desc="step 4 train", log_every=1)
    train_t0 = time.time()
    last_step_t = train_t0

    skipped_steps = 0  # count optim steps to skip during resume warm-up

    for ep in range(args.epochs):
        random.shuffle(forget_data)
        f_pos = 0
        while f_pos < len(forget_data):
            forget_batch = forget_data[f_pos:f_pos + args.forget_batch]
            retain_batch = next_retain_batch()
            f_pos += args.forget_batch

            l_forget_raw, finfo = _compute_forget_loss(
                model=model, batch=forget_batch, tokenizer=tokenizer,
                model_key=model_key, V_layers=V_layers,
                layer_indices=layer_indices, mu_minus_per=mu_minus_per,
                k_answer_tokens=cfg.K_ANSWER_TOKENS,
            )
            l_retain_raw, _rinfo = _compute_retain_loss(
                model=model, batch=retain_batch, tokenizer=tokenizer,
                model_key=model_key, V_layers=V_layers,
                layer_indices=layer_indices,
                k_answer_tokens=cfg.K_ANSWER_TOKENS,
            )
            l_forget = l_forget_raw / init_scale
            l_retain = l_retain_raw / init_scale

            l_total = l_forget + args.lambda_retain * l_retain
            (l_total / args.grad_accum).backward()

            accum_forget = accum_forget + l_forget.detach()
            accum_retain = accum_retain + l_retain.detach()
            if finfo.get("proj_norm"):
                accum_proj.append(finfo["proj_norm"])

            inner = (f_pos // args.forget_batch) % args.grad_accum
            if inner == 0 or f_pos >= len(forget_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.DEFAULT_MAX_GRAD_NORM,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                # Skip already-completed optim steps when resuming
                if optim_step <= start_step:
                    skipped_steps += 1
                    accum_forget = torch.tensor(0.0)
                    accum_retain = torch.tensor(0.0)
                    accum_proj = []
                    continue

                now = time.time()
                step_dt = now - last_step_t
                last_step_t = now

                avg_forget = float(accum_forget) / max(args.grad_accum, 1)
                avg_retain = float(accum_retain) / max(args.grad_accum, 1)
                avg_total  = avg_forget + args.lambda_retain * avg_retain
                mean_proj  = sum(accum_proj) / len(accum_proj) if accum_proj else 0.0
                lr_now = scheduler.get_last_lr()[0]

                row = {
                    "step":            optim_step,
                    "L_total":         round(avg_total,  6),
                    "L_forget":        round(avg_forget, 6),
                    "L_retain":        round(avg_retain, 6),
                    "mean_proj_norm":  round(mean_proj,  6),
                    "learning_rate":   lr_now,
                    "grad_norm":       round(float(grad_norm), 4),
                    "step_time_s":     round(step_dt, 3),
                    "elapsed_s":       round(now - train_t0, 1),
                }
                log_writer.writerow(row)
                log_file.flush()

                if not summary_initial:
                    summary_initial = {f"initial_{k}": v for k, v in row.items()
                                       if k not in ("step", "learning_rate",
                                                    "grad_norm", "step_time_s",
                                                    "elapsed_s")}
                summary_final = {f"final_{k}": v for k, v in row.items()
                                 if k not in ("step", "learning_rate",
                                              "grad_norm", "step_time_s",
                                              "elapsed_s")}

                progress.tick(extras={
                    "ep":   ep + 1,
                    "step": f"{optim_step}/{total_optim_steps}",
                    "L":    f"{avg_total:.4f}",
                    "L_F":  f"{avg_forget:.4f}",
                    "L_R":  f"{avg_retain:.4f}",
                    "lr":   f"{lr_now:.2e}",
                    "gn":   f"{float(grad_norm):.3f}",
                    "dt":   f"{step_dt:.2f}s",
                })

                accum_forget = torch.tensor(0.0)
                accum_retain = torch.tensor(0.0)
                accum_proj   = []

                # Periodic checkpoint
                if args.checkpoint_every > 0 and \
                   optim_step % args.checkpoint_every == 0 and \
                   optim_step != total_optim_steps:
                    _save_checkpoint(out_dir, model, optimizer, scheduler,
                                     step=optim_step,
                                     summary_initial=summary_initial)
                    log.info("  checkpoint @ step %d  (resume from here on restart)",
                             optim_step)

    log_file.close()
    progress.done(extras={"final_step": optim_step,
                          "skipped_resume_steps": skipped_steps})

    # Save final adapter (run-dir level, distinct from checkpoint/)
    log.info("Saving final adapter to %s", out_dir)
    model.save_pretrained(out_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)

    (out_dir / "training_config.json").write_text(json.dumps({
        "model_key":        model_key,
        "model_id":         cfg.MODEL_REGISTRY[model_key],
        "method":           "UOC two-component (shared V, per-domain μ⁻ forget pole, frozen-base retain preservation, transition-window)",
        "per_domain_poles": True,
        "pole_domains":     list(mu_minus_per.keys()),
        "retain_target":    "frozen-base h(x, y; θ_frozen)  (per-token, per-example) for both C and E",
        "answer_window":    "transition-shifted: [p_len-1, p_len+K-2]",
        "subspace_rank":    args.rank,
        "lambda_retain":    args.lambda_retain,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "forget_batch":     args.forget_batch,
        "retain_batch":     args.retain_batch,
        "grad_accum":       args.grad_accum,
        "checkpoint_every": args.checkpoint_every,
        "warmup_ratio":     cfg.DEFAULT_WARMUP_RATIO,
        "lora_r":           cfg.LORA_R,
        "lora_alpha":       cfg.LORA_ALPHA,
        "lora_dropout":     cfg.LORA_DROPOUT,
        "lora_target_modules": cfg.LORA_TARGET_MODULES,
        "k_answer_tokens":  cfg.K_ANSWER_TOKENS,
        "init_scale":       round(float(init_scale), 4),
        "forget_examples":  len(forget_data),
        "retain_examples":  len(retain_data),
        "total_optim_steps": total_optim_steps,
    }, indent=2))

    (out_dir / "subspace_config.json").write_text(json.dumps({
        "subspace_file": cfg.subspace_path(model_key, rank=args.rank).name,
        "anchors_file":  cfg.anchors_path(model_key).name,
        "layers":        layer_indices,
        "rank":          args.rank,
    }, indent=2))

    (out_dir / "train_summary.json").write_text(json.dumps(
        {**summary_initial, **summary_final,
         "num_steps":        optim_step,
         "elapsed_s":        round(time.time() - train_t0, 1),
         "elapsed_human":    format_duration(time.time() - train_t0),
         "step_avg_s":       round((time.time() - train_t0) / max(progress.n, 1), 3)},
        indent=2,
    ))

    # Clean up the rolling checkpoint now that the final adapter is in place
    import shutil
    shutil.rmtree(out_dir / CKPT_SUBDIR, ignore_errors=True)

    log.info("STEP 4 done in %s. Outputs in %s",
             format_duration(time.time() - pipeline_t0), out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: train UOC (two-component loss).")
    p.add_argument("--model", choices=list(cfg.MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--rank", type=int, default=cfg.SUBSPACE_RANK)
    p.add_argument("--lambda-retain", type=float, default=cfg.DEFAULT_LAMBDA_RETAIN,
                   help="Weight on L_retain (default: 1.0)")
    p.add_argument("--epochs",       type=int,   default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--lr",           type=float, default=cfg.DEFAULT_LR)
    p.add_argument("--forget-batch", type=int,   default=cfg.DEFAULT_FORGET_BATCH)
    p.add_argument("--retain-batch", type=int,   default=cfg.DEFAULT_RETAIN_BATCH)
    p.add_argument("--grad-accum",   type=int,   default=cfg.DEFAULT_GRAD_ACCUM)
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save a resumable checkpoint every N optim steps (0 to disable)")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
