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
import shutil
import sys
import time
from collections import deque
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
    """Returns (V_layers, layer_indices, mu_minus_per, mu_plus_per).

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

    Per-domain init_scale is read from the subspace bundle (baked in by step 2).
    For old subspace bundles that don't carry ``init_scales``, this falls back
    to recomputing from the activations bundle via ``_per_domain_init_scale``.
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

    init_scales = sb.get("init_scales")
    return V_layers, sb["layers"], mu_minus_per, mu_plus_per, init_scales


def _per_domain_init_scale(
    model_key: str,
    V_layers: list[torch.Tensor],
) -> dict[str, float]:
    """LEGACY FALLBACK: compute init_scale from the activations bundle.

    Step 2 now bakes ``init_scales`` directly into the subspace bundle, so
    in normal use ``train()`` reads it from there and this function is not
    called. It is kept as a fallback for old subspace bundles that pre-date
    that change and do not carry ``init_scales``.

    Matches the actual loss formula
        L_forget_per_ex(x) = ‖V_l⊤ (h_A(x) − μ⁻(d_x))‖²
    by using μ⁻(d) = mean over rows of h_B restricted to domain d as the
    baseline (not per-row h_B), then averaging over rows and layers.

    Returns ``{"kuq": float, "squad": float, "general": float}`` where
    ``general`` is the arithmetic mean of the two domain scales — used to
    normalise the retain-general (UltraChat) loss, which has no source
    domain.

    Each forget / retain example is divided per-example by its domain's
    init_scale inside the loss so that L_forget starts at ≈ 1.0 for both
    KUQ and SQuAD regardless of how much larger one domain's contrast
    magnitude is in late-layer hidden space. This is a constant per-example
    rescaling — it does not change the optimum or the relative weight λ,
    only the relative contribution of each domain to the gradient. Without
    it the larger-magnitude domain dominates the update.
    """
    act_path = cfg.activations_path(model_key)
    if not act_path.exists():
        raise FileNotFoundError(
            f"Activations bundle not found: {act_path}. Run step 1 first."
        )
    bundle = torch.load(act_path, map_location="cpu", weights_only=False)
    h_A = bundle["h_A"].float()           # (N_F, L, D)
    h_B = bundle["h_B"].float()
    meta_A = bundle.get("meta_A") or []
    meta_B = bundle.get("meta_B") or meta_A
    if len(meta_A) != h_A.shape[0] or len(meta_B) != h_B.shape[0]:
        raise RuntimeError(
            f"meta_A/meta_B misaligned with h_A/h_B "
            f"(len_meta_A={len(meta_A)} vs N_A={h_A.shape[0]}, "
            f"len_meta_B={len(meta_B)} vs N_B={h_B.shape[0]}). "
            f"Re-run step 1 with the latest extract.py."
        )

    scales: dict[str, float] = {}
    for domain in ("kuq", "squad"):
        idx_A = [i for i, m in enumerate(meta_A) if m.get("dataset") == domain]
        idx_B = [i for i, m in enumerate(meta_B) if m.get("dataset") == domain]
        if not idx_A or not idx_B:
            log.warning("  no examples for domain '%s' in activations; "
                        "init_scale defaulting to 1.0", domain)
            scales[domain] = 1.0
            continue
        mu_minus_d = h_B[idx_B].mean(dim=0)        # (L, D)  per-domain abstain pole
        c = h_A[idx_A] - mu_minus_d.unsqueeze(0)   # (n_d, L, D)
        per_layer: list[float] = []
        for li, V_l in enumerate(V_layers):
            proj = c[:, li, :] @ V_l               # (n_d, r)
            per_layer.append(float((proj ** 2).sum(dim=-1).mean()))
        scales[domain] = max(sum(per_layer) / len(per_layer), 1e-6)

    scales["general"] = (scales["kuq"] + scales["squad"]) / 2.0
    return scales


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
    V_layers, layer_indices, mu_minus_per: dict,
    init_scales: dict[str, float], k_answer_tokens: int,
) -> tuple[torch.Tensor, dict]:
    """L_forget — pull category-A activations toward μ⁻(d) along V, where d is
    the example's source dataset (kuq | squad). Each example is divided by
    ``init_scales[d]`` so KUQ and SQuAD contribute on the same O(1) scale at
    step 0 regardless of their domain-specific contrast magnitude."""
    total = torch.tensor(0.0, requires_grad=True)
    layer_norms: list[float] = []
    n_used = 0
    by_dataset = {d: 0 for d in mu_minus_per.keys()}
    for r in batch:
        ds = r["__dataset__"]
        mu_minus = mu_minus_per.get(ds)
        scale = init_scales.get(ds)
        if mu_minus is None or scale is None:
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
        per_ex = per_ex / scale
        total = total + per_ex
        n_used += 1
        by_dataset[ds] = by_dataset.get(ds, 0) + 1

    if n_used > 0:
        total = total / n_used
    mean_norm = sum(layer_norms) / len(layer_norms) if layer_norms else 0.0
    return total, {"n": n_used, "proj_norm": mean_norm, **by_dataset}


def _compute_retain_loss(
    *, model, batch: list[dict], tokenizer, model_key: str,
    V_layers, layer_indices, init_scales: dict[str, float], k_answer_tokens: int,
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

    Per-example normalisation uses ``init_scales[d_x]`` for category C
    (answerable, has a source domain) and ``init_scales["general"]`` for
    category E (UltraChat, no source domain). The retain loss starts at 0
    by construction, so the per-example division mostly affects the
    relative gradient magnitude between the two branches across domains —
    matching the forget side's per-domain scaling so the loss landscape
    is consistently scaled across the (forget, retain) × (kuq, squad,
    general) grid.

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
            scale = init_scales.get(ds, init_scales.get("general", 1.0))
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
            scale = init_scales.get("general", 1.0)
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
        per_ex = per_ex / scale
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


# ── Checkpoint + early-stop adapter helpers ──────────────────────────────────

CKPT_SUBDIR        = "checkpoint"
BEST_SUBDIR        = "_best"           # best-step adapter snapshot (early stop)
FINAL_SUBDIR       = "_final"          # last-step adapter snapshot (always written)
TRAINER_STATE_FILE = "trainer_state.pt"

# Adapter files PEFT writes via save_pretrained that we promote to the run dir
# (tokenizer files are written once and stay at the root regardless of which
# snapshot is the primary).
_ADAPTER_PROMOTE_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "README.md",
)


def _save_adapter_snapshot(dest: Path, model) -> None:
    """Atomic PEFT adapter save to ``dest``: write into ``dest.tmp`` and rename.

    Tokenizer files are NOT written here; they're saved once at the run-dir
    root by the main loop and don't change across snapshots.
    """
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def _promote_adapter(src: Path, out_dir: Path) -> None:
    """Copy adapter files from a snapshot dir (``_best`` / ``_final``) to the
    run-dir root, which is what ``step5_evaluate`` loads via
    ``PeftModel.from_pretrained(model, str(run_dir))``."""
    for fname in _ADAPTER_PROMOTE_FILES:
        sp = src / fname
        if sp.exists():
            shutil.copy2(sp, out_dir / fname)


def _save_checkpoint(out_dir: Path, model, optimizer, scheduler,
                     step: int, summary_initial: dict,
                     loss_window: deque, best_smoothed_loss: float,
                     best_step_recorded: int) -> None:
    """Save adapter + optim/scheduler state + early-stop tracking state to
    ``<run_dir>/checkpoint/``. Atomic w.r.t. interrupts."""
    final = out_dir / CKPT_SUBDIR
    tmp   = out_dir / (CKPT_SUBDIR + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save({
        "step":                step,
        "optimizer":           optimizer.state_dict(),
        "scheduler":           scheduler.state_dict(),
        "summary_initial":     summary_initial,
        "loss_window":         list(loss_window),
        "best_smoothed_loss":  float(best_smoothed_loss),
        "best_step_recorded":  int(best_step_recorded),
    }, tmp / TRAINER_STATE_FILE)
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def _load_checkpoint_if_any(out_dir: Path):
    """Return saved trainer-state dict (optim/scheduler/step/early-stop) or None."""
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

    V_layers, layer_indices, mu_minus_per, mu_plus_per, init_scales = \
        _load_subspace_and_anchors(model_key, rank=args.rank)
    log.info("  V layers=%s  rank=%d", layer_indices, args.rank)
    if init_scales is None:
        log.warning("  subspace bundle has no baked init_scales; "
                    "recomputing from activations bundle (re-run step 2 to bake it in)")
        init_scales = _per_domain_init_scale(model_key, V_layers)
    log.info("  init_scales (per-domain): %s",
             {k: round(v, 2) for k, v in init_scales.items()})
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
    # Early-stop / best-step tracking — survives across resume via trainer_state.
    loss_window: deque = deque(maxlen=max(int(args.early_stop_window), 1))
    best_smoothed_loss = float("inf")
    best_step_recorded = -1
    if resume:
        try:
            optimizer.load_state_dict(ckpt_state["optimizer"])
            scheduler.load_state_dict(ckpt_state["scheduler"])
            start_step = int(ckpt_state.get("step", 0))
            summary_initial = ckpt_state.get("summary_initial", {}) or {}
            for v in ckpt_state.get("loss_window", []) or []:
                loss_window.append(float(v))
            best_smoothed_loss = float(ckpt_state.get("best_smoothed_loss", float("inf")))
            best_step_recorded = int(ckpt_state.get("best_step_recorded", -1))
            log.info("  resumed at optim_step=%d   best_step=%d  best_smoothed_loss=%.4f",
                     start_step, best_step_recorded, best_smoothed_loss)
        except Exception as exc:
            log.warning("  failed to load optim/scheduler state (%s); restarting fresh", exc)
            start_step = 0
            loss_window.clear()
            best_smoothed_loss = float("inf")
            best_step_recorded = -1

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

            # Per-example scaling by domain-specific init_scale happens inside
            # the loss helpers, so both terms come out as ≈ O(1) at step 0.
            l_forget, finfo = _compute_forget_loss(
                model=model, batch=forget_batch, tokenizer=tokenizer,
                model_key=model_key, V_layers=V_layers,
                layer_indices=layer_indices, mu_minus_per=mu_minus_per,
                init_scales=init_scales,
                k_answer_tokens=cfg.K_ANSWER_TOKENS,
            )
            l_retain, _rinfo = _compute_retain_loss(
                model=model, batch=retain_batch, tokenizer=tokenizer,
                model_key=model_key, V_layers=V_layers,
                layer_indices=layer_indices, init_scales=init_scales,
                k_answer_tokens=cfg.K_ANSWER_TOKENS,
            )

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

                # Early-stop tracking — smoothed L_total minimum, with optional
                # minimum-improvement threshold so we don't write the adapter
                # to disk on tiny numerical fluctuations.
                loss_window.append(avg_total)
                new_best = False
                if args.early_stop and len(loss_window) >= loss_window.maxlen:
                    smoothed = sum(loss_window) / len(loss_window)
                    threshold = best_smoothed_loss * (1.0 - args.early_stop_min_improvement)
                    if smoothed < threshold:
                        best_smoothed_loss = smoothed
                        best_step_recorded = optim_step
                        _save_adapter_snapshot(out_dir / BEST_SUBDIR, model)
                        new_best = True

                progress.tick(extras={
                    "ep":   ep + 1,
                    "step": f"{optim_step}/{total_optim_steps}",
                    "L":    f"{avg_total:.4f}",
                    "L_F":  f"{avg_forget:.4f}",
                    "L_R":  f"{avg_retain:.4f}",
                    "lr":   f"{lr_now:.2e}",
                    "gn":   f"{float(grad_norm):.3f}",
                    "dt":   f"{step_dt:.2f}s",
                    **({"best": f"{best_smoothed_loss:.4f}@{best_step_recorded}"}
                       if args.early_stop and best_step_recorded > 0 else {}),
                    **({"*": "new-best"} if new_best else {}),
                })

                accum_forget = torch.tensor(0.0)
                accum_retain = torch.tensor(0.0)
                accum_proj   = []

                # Periodic checkpoint
                if args.checkpoint_every > 0 and \
                   optim_step % args.checkpoint_every == 0 and \
                   optim_step != total_optim_steps:
                    _save_checkpoint(
                        out_dir, model, optimizer, scheduler,
                        step=optim_step,
                        summary_initial=summary_initial,
                        loss_window=loss_window,
                        best_smoothed_loss=best_smoothed_loss,
                        best_step_recorded=best_step_recorded,
                    )
                    log.info("  checkpoint @ step %d  (resume from here on restart)",
                             optim_step)

    log_file.close()
    progress.done(extras={"final_step": optim_step,
                          "skipped_resume_steps": skipped_steps})

    # Always snapshot the final-step adapter into _final/ for diagnostics, then
    # decide whether the run-dir root (the primary load location used by
    # step5_evaluate) should hold the best-step or the final-step adapter.
    _save_adapter_snapshot(out_dir / FINAL_SUBDIR, model)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)

    promoted_source = "final"
    if args.early_stop and best_step_recorded > 0 and (out_dir / BEST_SUBDIR).exists():
        log.info("Promoting BEST adapter (step %d, smoothed L_total=%.4f) "
                 "to primary at %s", best_step_recorded, best_smoothed_loss, out_dir)
        _promote_adapter(out_dir / BEST_SUBDIR, out_dir)
        promoted_source = "best"
    else:
        log.info("Promoting FINAL-step adapter to primary at %s "
                 "(early_stop=%s, best_step=%d)",
                 out_dir, args.early_stop, best_step_recorded)
        _promote_adapter(out_dir / FINAL_SUBDIR, out_dir)

    (out_dir / "training_config.json").write_text(json.dumps({
        "model_key":        model_key,
        "model_id":         cfg.MODEL_REGISTRY[model_key],
        "method":           ("UOC two-component (shared V, per-domain μ⁻ forget pole, "
                             "per-domain init_scale, frozen-base retain preservation, "
                             "transition-window, best-step early-stop)"),
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
        "init_scales":      {k: round(float(v), 4) for k, v in init_scales.items()},
        "early_stop":                  bool(args.early_stop),
        "early_stop_window":           int(args.early_stop_window),
        "early_stop_min_improvement":  float(args.early_stop_min_improvement),
        "primary_adapter_source":      promoted_source,   # "best" or "final"
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

    elapsed = time.time() - train_t0
    (out_dir / "train_summary.json").write_text(json.dumps(
        {**summary_initial, **summary_final,
         "num_steps":           optim_step,
         "best_step":           int(best_step_recorded) if best_step_recorded > 0 else None,
         "best_smoothed_L_total": (round(float(best_smoothed_loss), 6)
                                   if best_smoothed_loss < float("inf") else None),
         "primary_adapter_source": promoted_source,
         "elapsed_s":           round(elapsed, 1),
         "elapsed_human":       format_duration(elapsed),
         "step_avg_s":          round(elapsed / max(progress.n, 1), 3)},
        indent=2,
    ))

    # Clean up the rolling resume checkpoint now that primary + _best/_final
    # snapshots are in place.
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
    p.add_argument("--early-stop", dest="early_stop", action="store_true", default=True,
                   help="Track smoothed L_total and save the best-step adapter as the "
                        "primary load target at <run_dir>/. Default: enabled.")
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   help="Disable best-step tracking; primary adapter is the final step.")
    p.add_argument("--early-stop-window", type=int, default=5,
                   help="Smoothing window (in optim steps) for the L_total minimum "
                        "tracked by --early-stop. Default: 5.")
    p.add_argument("--early-stop-min-improvement", type=float, default=0.0,
                   help="Minimum relative improvement (e.g. 0.01 = 1%%) for a step to "
                        "count as a new best and trigger a best-adapter snapshot. "
                        "Default: 0.0 (any improvement counts).")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
