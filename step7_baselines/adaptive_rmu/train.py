#!/usr/bin/env python3
"""AdaptiveRMU baseline for the URC project.

Faithful port of the AAAI-25 implementation:
  https://github.com/RebelsNLU-jaist/llm-unlearning
Paper: "On Effects of Steering Latent Representation for Large Language
        Model Unlearning" — Dang, Pham, Hoang, Inoue, AAAI 2025

The original AdaptiveRMU trains on WMDP (bio/cyber forget + wikitext retain)
and directly updates model layer weights. This port adapts it to URC's data
and infrastructure while preserving the core algorithm faithfully:

  Preserved from original
  -----------------------
  • Per-domain random unit control vectors with adaptive magnitude scaling.
    Scaling coefficient c_d is computed once from the first forget batch's
    mean activation norm, multiplied by `scale`. After that it is fixed.
  • forward_with_cache: forward hook at a single transformer layer captures
    hidden states (batch, seq_len, hidden_dim) for both forget and retain.
  • Forget loss:  MSE(h_forget_updated, c_d * u_d)
  • Retain loss:  MSE(h_retain_updated, h_retain_frozen) * alpha_d
  • Total loss:   L_forget + L_retain  (one optimizer step per batch)
  • Alternating-domain training loop: topic_idx = step % n_topics
  • Single measurement layer (layer_id) with multi-layer parameter update
    (layer_ids), matching the original's distinct measurement / update split.

  URC adaptations
  ---------------
  • LoRA adapter (PEFT) instead of direct weight update, so the checkpoint
    is loadable by step5_evaluate/evaluate.py via --run-dir.
  • Two domains: KUQ (no-context) and SQuAD (context-grounded), replacing
    the original's bio/cyber topics. Both use URC's prompt templates.
  • Data loaded from URC's JSONL format (step0_mine outputs + sampled pools).
  • Frozen reference via model.disable_adapter_layers() instead of a second
    frozen model instance, since the LoRA base weights are already frozen.
  • Output written to step7_baselines/adaptive_rmu/data/runs/<run_name>/ in
    the PEFT adapter format that evaluate.py expects.

Run
---
    python step7_baselines/adaptive_rmu/train.py --model qwen_instruct
    python step7_baselines/adaptive_rmu/train.py --model qwen_instruct \\
        --scale 5.0 --alpha 1200,1200 --max-num-batches 500 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
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

# Topics mirror the original's bio/cyber split: two forget domains
DOMAINS = ["kuq", "squad"]


# ── Faithful port: original utils.py helpers ─────────────────────────────────

def forward_with_cache(model, input_ids: torch.Tensor, module, no_grad: bool = True):
    """Hook-based hidden-state extraction — faithful to original utils.py.

    Registers a forward hook on `module`, runs the forward pass, removes
    the hook, and returns the captured activation tensor
    (batch_size, seq_len, hidden_dim).

    no_grad=True:  detaches the output (used for frozen reference).
    no_grad=False: keeps the full computation graph (used for unlearn loss).
    """
    cache: list[torch.Tensor] = []

    def _hook(mod, inp, output):
        tensor = output[0] if isinstance(output, tuple) else output
        cache.append(tensor.detach().clone() if no_grad else tensor)

    handle = module.register_forward_hook(_hook)
    attention_mask = torch.ones_like(input_ids)
    try:
        if no_grad:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    return cache[0]


def get_params(model, layer_ids: list[int]) -> list[torch.nn.Parameter]:
    """Return trainable parameters in the given transformer layer indices.

    Faithful to original get_params, adapted for LoRA: only parameters with
    requires_grad=True (i.e. LoRA A/B matrices) are returned, not frozen
    base weights. This preserves the original's concept of updating specific
    layer parameters while keeping everything else fixed.
    """
    params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for lid in layer_ids:
            if f".layers.{lid}." in name:
                params.append(p)
                break
    return params


def _get_layer_module(model, layer_id: int):
    """Find the transformer layer module at `layer_id`.

    Walks named_modules to locate the exact layer regardless of PEFT wrapping
    depth (PeftModel → LoraModel → base model → transformer stack → layers).
    Returns the first match where the parent container is named "layers" and
    the child key equals layer_id — this handles Qwen, Llama, Mistral, and
    gptoss all of which use model.model.layers[i] as the transformer stack.
    """
    for name, module in model.named_modules():
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_tail, child = parts
            if child == str(layer_id) and parent_tail.endswith("layers"):
                return module
        elif name == str(layer_id):
            return module
    raise RuntimeError(
        f"Cannot find transformer layer {layer_id} in model. "
        "Layer names: " + str([n for n, _ in model.named_modules()][:30])
    )


def _get_hidden_size(model) -> int:
    """Return hidden_size handling Ministral's text_config nesting.

    Ministral uses Mistral3ForConditionalGeneration whose hidden_size lives
    under config.text_config (same nesting as num_hidden_layers). Qwen,
    Llama, and gptoss expose hidden_size directly on model.config.
    """
    mcfg = model.config
    tc = getattr(mcfg, "text_config", None)
    return int(getattr(tc if tc is not None else mcfg, "hidden_size"))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_forget_by_domain(model_key: str) -> dict[str, list[dict]]:
    """Load mined COMMIT examples per domain from step0_mine outputs."""
    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for dataset in DOMAINS:
        path = cfg.forget_path(model_key, dataset)
        if not path.exists():
            log.warning("  forget pool missing: %s", path)
            continue
        rows = [
            r for r in load_jsonl(path)
            if (r.get("judge_label") or "").upper() in ("COMMIT", "COMMITTED")
        ]
        for r in rows:
            r["__dataset__"] = dataset
        by_domain[dataset] = rows
        log.info("  forget[%s]: %d examples", dataset, len(rows))
    return by_domain


def _load_retain(model_key: str) -> list[dict]:
    """Load retain pool: answerable KUQ/SQuAD + UltraChat."""
    pool: list[dict] = []
    for dataset in DOMAINS:
        path = cfg.sampled_answerable_path(dataset)
        if not path.exists():
            log.warning("  retain answerable missing: %s", path)
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
    log.info("  retain: %d examples", len(pool))
    random.shuffle(pool)
    return pool


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize_forget_row(
    tokenizer, model_key: str, row: dict, max_len: int = 512,
) -> torch.Tensor | None:
    """Tokenize a forget row to a (1, seq_len) input_ids tensor."""
    ds     = row["__dataset__"]
    prompt = build_unanswerable_prompt(ds, row)
    compl  = row.get("y_com_prefix_k8") or row.get("full_completion_clean") or ""
    if not prompt.strip() or not compl.strip():
        return None
    try:
        full_ids, _, _ = tokenise_prompt_plus_answer(
            tokenizer, model_key, prompt, compl,
        )
    except Exception:
        return None
    return torch.tensor([full_ids[:max_len]], dtype=torch.long)


def _tokenize_retain_row(
    tokenizer, model_key: str, row: dict, max_len: int = 512,
) -> torch.Tensor | None:
    """Tokenize a retain row to a (1, seq_len) input_ids tensor."""
    kind = row.get("__type__")
    try:
        if kind == "answerable":
            ds     = row["__dataset__"]
            prompt = build_answerable_prompt(ds, row)
            answer = row.get("correct_answer") or ""
            if not prompt.strip() or not answer.strip():
                return None
            full_ids, _, _ = tokenise_prompt_plus_answer(
                tokenizer, model_key, prompt, answer,
            )
        elif kind == "general":
            prompt   = row.get("prompt") or ""
            response = row.get("response") or ""
            if not prompt.strip() or not response.strip():
                return None
            full_ids, _ = tokenise_chat_prompt_response(
                tokenizer, model_key, prompt, response,
            )
        else:
            return None
    except Exception:
        return None
    return torch.tensor([full_ids[:max_len]], dtype=torch.long)


# ── Core training loop ────────────────────────────────────────────────────────

def run_adaptive_rmu(
    model,
    tokenizer,
    model_key: str,
    forget_by_domain: dict[str, list[dict]],
    retain_data: list[dict],
    args,
    out_dir: Path,
) -> None:
    """Faithful port of run_adaptive_rmu from the original.

    Key correspondences to original code
    -------------------------------------
    forget_data_list[topic_idx][batch_idx]  →  forget_by_domain[domain][rows]
    retain_data_list[topic_idx][batch_idx]  →  retain_data (shared, cycled)
    frozen_model                            →  model with adapters disabled
    updated_module / frozen_module          →  same layer, hook captures both
    coeffs                                  →  per-topic adaptive float dict
    """
    model.train()
    n_topics = len(DOMAINS)

    # Layer module for hook-based activation extraction (single layer, faithful)
    updated_module = _get_layer_module(model, args.layer_id)
    log.info("  measurement layer: %d  (module type: %s)",
             args.layer_id, type(updated_module).__name__)

    # Trainable LoRA parameters in the update layer range
    params = get_params(model, args.layer_ids)
    if not params:
        log.warning("  get_params: no trainable params matched layer_ids %s; "
                    "falling back to all trainable", args.layer_ids)
        params = [p for p in model.parameters() if p.requires_grad]
    log.info("  trainable param tensors: %d", len(params))

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # Per-topic random control vectors (unit vectors, faithful to original)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Use the first trainable parameter only for dtype; device is not reliable
    # for multi-GPU models where device_map="auto" spreads layers across GPUs.
    # Control vectors are kept on CPU and moved to each activation's device at
    # MSE time, so they are always co-located with what they are compared to.
    dtype      = next(p for p in model.parameters() if p.requires_grad).dtype
    # Input ids must go to the embedding layer's device (always the first GPU).
    device     = next(model.parameters()).device
    hidden_dim = _get_hidden_size(model)

    control_vectors: list[torch.Tensor] = []
    for i in range(n_topics):
        # Build on CPU; moved to acts.device when computing MSE (Bug 2 fix)
        v = torch.rand(1, 1, hidden_dim, dtype=dtype)
        unit_v = (v / v.norm()) * args.steering_coeff_list[i]
        control_vectors.append(unit_v)
        log.info("  control_vec[%s] init norm: %.4f", DOMAINS[i], unit_v.norm().item())

    # Adaptive scaling coefficients per topic (set on first batch, then fixed)
    coeffs: dict[int, float] = {i: 1.0 for i in range(n_topics)}

    # Batched forget pools per topic — matches original's forget_data_list
    forget_batches: list[list[list[dict]]] = []
    for domain in DOMAINS:
        rows = list(forget_by_domain.get(domain, []))
        random.shuffle(rows)
        batches = [rows[i:i + args.batch_size]
                   for i in range(0, len(rows), args.batch_size)
                   if rows[i:i + args.batch_size]]
        forget_batches.append(batches)
        log.info("  forget_batches[%s]: %d batches of up to %d",
                 domain, len(batches), args.batch_size)

    # Retain pool cycling (original uses separate per-topic retain; here shared)
    retain_cycle = list(retain_data)
    random.shuffle(retain_cycle)
    retain_pos = 0

    def _next_retain_batch() -> list[dict]:
        nonlocal retain_cycle, retain_pos
        batch: list[dict] = []
        for _ in range(args.batch_size):
            if retain_pos >= len(retain_cycle):
                retain_cycle = list(retain_data)
                random.shuffle(retain_cycle)
                retain_pos = 0
            batch.append(retain_cycle[retain_pos])
            retain_pos += 1
        return batch

    # ── CSV log ──────────────────────────────────────────────────────────────
    csv_path   = out_dir / "loss_log.csv"
    csv_fh     = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_fh,
        fieldnames=["step", "domain", "L_total", "L_unlearn", "L_retain",
                    "coeff", "elapsed_s"],
    )
    csv_writer.writeheader()
    csv_fh.flush()

    # ── Main loop (faithful to original) ─────────────────────────────────────
    prog = Progress(total=args.max_num_batches, desc="adaptive_rmu", log_every=25)
    t0   = time.time()

    for idx in range(args.max_num_batches):
        topic_idx = idx % n_topics          # alternate topics, original: idx % len(forget_data_list)
        domain    = DOMAINS[topic_idx]
        pool      = forget_batches[topic_idx]

        if not pool:
            prog.tick()
            continue

        batch_idx    = (idx // n_topics) % len(pool)
        forget_rows  = pool[batch_idx]
        retain_rows  = _next_retain_batch()
        control_vec  = control_vectors[topic_idx]   # (1, 1, D) scaled unit vector

        # ── Forget activations (with grad) ───────────────────────────────────
        forget_acts: list[torch.Tensor] = []
        for row in forget_rows:
            ids = _tokenize_forget_row(tokenizer, model_key, row,
                                       max_len=args.max_len)
            if ids is None:
                continue
            ids = ids.to(device)
            acts = forward_with_cache(model, ids, updated_module, no_grad=False)
            forget_acts.append(acts)          # (1, seq_len, D) with grad

        if not forget_acts:
            prog.tick()
            continue

        # Adaptive scaling coefficient (faithful: set once per topic from the
        # first batch's mean activation norm × scale, then frozen)
        # Original: idx==0 sets coeff "0", idx==1 sets coeff "1"
        if idx == topic_idx:
            mean_norm = float(
                torch.stack([
                    a.detach().norm(dim=-1).mean() for a in forget_acts
                ]).mean()
            )
            coeffs[topic_idx] = mean_norm * args.scale
            log.info("  adaptive coeff[%s] = %.4f  (mean_norm=%.4f × scale=%.1f)",
                     domain, coeffs[topic_idx], mean_norm, args.scale)

        # Unlearn loss: MSE(h_updated, control_vec * coeff)  — faithful
        # target stays on CPU; moved to a.device at each MSE call so this
        # works regardless of which GPU the measurement layer lives on.
        coeff_val = coeffs[topic_idx]
        target    = control_vec * coeff_val              # (1, 1, D) on CPU
        unlearn_loss = sum(
            F.mse_loss(a, target.to(a.device).to(a.dtype).expand_as(a))
            for a in forget_acts
        )
        unlearn_loss = unlearn_loss / len(forget_acts)

        # ── Retain activations (updated vs frozen) ────────────────────────────
        # Accumulate into a list to avoid pre-allocating on a specific device.
        # Both updated_acts and frozen_acts come from the same hook on the same
        # module so they are always co-located — no cross-device MSE.
        retain_parts: list[torch.Tensor] = []
        for row in retain_rows:
            ids = _tokenize_retain_row(tokenizer, model_key, row,
                                       max_len=args.max_len)
            if ids is None:
                continue
            ids = ids.to(device)

            # Updated activations (adapter on, with grad)
            updated_acts = forward_with_cache(
                model, ids, updated_module, no_grad=False,
            )

            # Frozen reference (adapter off, no grad) — faithful to original's
            # separate frozen_model instance but via disable/enable
            model.eval()
            model.disable_adapter_layers()
            frozen_acts = forward_with_cache(
                model, ids, updated_module, no_grad=True,
            )
            model.enable_adapter_layers()
            model.train()

            retain_parts.append(F.mse_loss(updated_acts, frozen_acts))

        if retain_parts:
            retain_loss = sum(retain_parts) / len(retain_parts)
        else:
            retain_loss = torch.tensor(0.0)
        retain_loss = retain_loss * args.alpha[topic_idx]  # faithful: alpha per topic

        # ── Gradient step (faithful to original) ─────────────────────────────
        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        row_log = {
            "step":      idx,
            "domain":    domain,
            "L_total":   round(float(loss.detach()),         6),
            "L_unlearn": round(float(unlearn_loss.detach()), 6),
            "L_retain":  round(float(retain_loss.detach()),  6),
            "coeff":     round(coeff_val,                    4),
            "elapsed_s": round(elapsed,                      2),
        }
        csv_writer.writerow(row_log)
        csv_fh.flush()

        prog.tick(extras={
            "dom": domain,
            "Lu": f"{float(unlearn_loss.detach()):.3g}",
            "Lr": f"{float(retain_loss.detach()):.3g}",
        })

    csv_fh.close()
    prog.done()
    log.info("  run_adaptive_rmu finished in %s", format_duration(time.time() - t0))


# ── LoRA application (reused from step4_train) ────────────────────────────────

def _num_text_layers(model) -> int:
    """Number of transformer layers in the text stack, multimodal-safe.

    Ministral uses Mistral3ForConditionalGeneration whose num_hidden_layers
    lives under config.text_config. Qwen and gptoss expose it directly.
    """
    mcfg = model.config
    tc = getattr(mcfg, "text_config", None)
    return int(getattr(tc if tc is not None else mcfg, "num_hidden_layers"))


def _apply_lora(model, model_key: str, exclude_last: int = 0) -> object:
    """Apply LoRA adapter.  Matches step4_train._apply_lora.

    exclude_last: exclude the last N transformer layers from LoRA adaptation.
    Motivation (Ministral-14B): the final layer is an output funnel — LoRA
    edits there write straight into the pre-lm_head state with no downstream
    layers to absorb the perturbation, which causes repetition collapse.
    Default is 0 (no exclusion).  Passed via --lora-exclude-last.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    expert_params = cfg.lora_target_parameters(model_key)
    if expert_params:
        # Fused-expert MoE (gptoss): expert FFNs are 3-D nn.Parameter tensors,
        # not nn.Linear, so they're targeted via `target_parameters`.
        # lora_dropout must be 0 for PEFT's ParamWrapper.
        attn_targets = cfg.lora_attn_targets(model_key)
        lcfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
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
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.lora_dense_targets(model_key),
            layers_to_transform=layers_to_transform,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_run(model, tokenizer, out_dir: Path, model_key: str, args) -> None:
    """Save LoRA adapter + metadata in the format expected by evaluate.py."""
    # Atomic save: write to .tmp, rename over out_dir
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(tmp))
    try:
        tokenizer.save_pretrained(str(tmp))
    except Exception as exc:
        log.warning("  tokenizer.save_pretrained failed (%s); skipping", exc)

    train_cfg = {
        "model_key":        model_key,
        "method":           "adaptive_rmu",
        "scale":            args.scale,
        "alpha":            args.alpha,
        "steering_coeffs":  args.steering_coeff_list,
        "layer_id":         args.layer_id,
        "layer_ids":        args.layer_ids,
        "max_num_batches":  args.max_num_batches,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "seed":             args.seed,
        "max_len":          args.max_len,
        "lora_r":           cfg.LORA_R,
        "lora_alpha":       cfg.LORA_ALPHA,
        "lora_exclude_last": args.lora_exclude_last,
    }
    (tmp / "training_config.json").write_text(
        json.dumps(train_cfg, indent=2)
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    log.info("  saved adapter to %s", out_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    model_key = args.model
    t0 = time.time()

    # Data
    forget_by_domain = _load_forget_by_domain(model_key)
    retain_data      = _load_retain(model_key)
    if all(len(v) == 0 for v in forget_by_domain.values()):
        raise RuntimeError(
            "All forget pools are empty. Run step0_mine first."
        )
    if not retain_data:
        raise RuntimeError("Retain pool is empty. Check step0_mine/data/sampled/.")

    if args.dry_run:
        for d in forget_by_domain:
            forget_by_domain[d] = forget_by_domain[d][:8]
        retain_data = retain_data[:8]
        args.max_num_batches = 4
        log.info("  DRY RUN: limited to 8 examples, 4 batches")

    # Resolve layer_id / layer_ids from LAYER_SLICE before building the run
    # name so the directory reflects the actual layer used, not the sentinel -1.
    layer_slice = cfg.LAYER_SLICE.get(model_key, [])
    if args.layer_id < 0:
        args.layer_id = layer_slice[-1] if layer_slice else -1
        log.info("  layer_id defaulted to last in LAYER_SLICE: %d", args.layer_id)
    if not args.layer_ids:
        args.layer_ids = layer_slice
        log.info("  layer_ids defaulted to LAYER_SLICE: %s", args.layer_ids)

    if args.layer_id < 0:
        raise ValueError(
            f"layer_id={args.layer_id} invalid and no LAYER_SLICE for {model_key!r}."
        )

    # Run name — mirrors original's checkpoint naming convention
    alpha_str = "-".join(str(int(a)) for a in args.alpha)
    run_name  = (
        f"{model_key}_adap_rmu"
        f"_scale{args.scale:g}"
        f"_alpha{alpha_str}"
        f"_layer{args.layer_id}"
        f"_batches{args.max_num_batches}"
    )
    if args.lora_exclude_last:
        run_name = f"{run_name}_excl{args.lora_exclude_last}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"

    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run: %s", run_name)
    log.info("Output dir: %s", out_dir)

    # Model + LoRA
    with Stopwatch("model load"):
        model, tokenizer = load_model_and_tokenizer(model_key, eval_only=False)

    model = _apply_lora(model, model_key, exclude_last=args.lora_exclude_last)
    model.train()

    # Fused-expert MoE: gradient checkpointing needed to avoid OOM
    if cfg.lora_target_parameters(model_key):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.enable_input_require_grads()
        log.info("  gradient checkpointing enabled (MoE)")

    log.info("  measurement layer_id:  %d", args.layer_id)
    log.info("  update     layer_ids:  %s", args.layer_ids)
    log.info("  scale=%.2f  alpha=%s  steering_coeffs=%s",
             args.scale, args.alpha, args.steering_coeff_list)
    log.info("  max_num_batches=%d  batch_size=%d  lr=%.2e",
             args.max_num_batches, args.batch_size, args.lr)

    run_adaptive_rmu(
        model          = model,
        tokenizer      = tokenizer,
        model_key      = model_key,
        forget_by_domain = forget_by_domain,
        retain_data    = retain_data,
        args           = args,
        out_dir        = out_dir,
    )

    _save_run(model, tokenizer, out_dir, model_key, args)
    log.info("ADAPTIVE RMU done in %s", format_duration(time.time() - t0))
    log.info("")
    log.info("Evaluate with:")
    log.info("  python step5_evaluate/evaluate.py --run-dir %s \\", out_dir)
    log.info("      --datasets kuq squad selfaware faitheval nomiracl \\")
    log.info("      --baseline step5_evaluate/data/results/baseline_%s", model_key)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AdaptiveRMU baseline (AAAI-25) for URC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model", required=True,
                   help="Model key from config.py (e.g. qwen_instruct)")

    # Original AdaptiveRMU hyperparameters (faithful names)
    p.add_argument("--scale", type=float, default=5.0,
                   help="Steering scale: coeff = mean_act_norm × scale")
    p.add_argument("--alpha", type=str, default="1200,1200",
                   help="Retain loss weight per topic (comma-separated)")
    p.add_argument("--steering-coeffs", type=str, default="1,1",
                   help="Initial steering vector magnitude per topic")
    p.add_argument("--max-num-batches", type=int, default=500,
                   help="Total optimizer steps (alternates over topics)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Examples per topic per step")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="AdamW learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for control vector init and data shuffle")

    # Layer selection (defaults derived from LAYER_SLICE)
    p.add_argument("--layer-id", type=int, default=-1,
                   help="Layer to measure activations at (-1 = last in LAYER_SLICE)")
    p.add_argument("--layer-ids", type=str, default="",
                   help="Comma-separated layer indices to update (-1 = all in LAYER_SLICE)")

    # Data / sequence
    p.add_argument("--max-len", type=int, default=512,
                   help="Max token length per example (truncated)")

    # LoRA
    p.add_argument("--lora-exclude-last", type=int, default=0,
                   help="Exclude last N layers from LoRA adaptation (dense models only). "
                        "Recommended: 1 for ministral14b_instruct to avoid the output-funnel "
                        "repetition collapse (matches step4_train --lora-exclude-last behaviour).")

    # URC utilities
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix appended to the run name")
    p.add_argument("--dry-run", action="store_true",
                   help="Trim data to 8 examples and 4 steps for smoke testing")

    args = p.parse_args()

    # Parse comma-separated lists (faithful to original arg parsing)
    args.alpha              = [float(x) for x in args.alpha.split(",")]
    args.steering_coeff_list = [float(x) for x in args.steering_coeffs.split(",")]
    args.layer_ids           = (
        [int(x) for x in args.layer_ids.split(",") if x.strip()]
        if args.layer_ids.strip() else []
    )

    # Pad alpha and steering_coeff_list to n_topics if needed
    n = len(DOMAINS)
    while len(args.alpha) < n:
        args.alpha.append(args.alpha[-1])
    while len(args.steering_coeff_list) < n:
        args.steering_coeff_list.append(args.steering_coeff_list[-1])

    return args


if __name__ == "__main__":
    train(_parse_args())
