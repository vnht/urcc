#!/usr/bin/env python3
"""
URC training: retain-normalised subspace projection unlearning.

Forget loss:  minimise ||r_l @ V_l||^2 over the first k answer tokens
              at each selected layer.
Retain loss:  KL(p_frozen || p_trainable) over response tokens on UltraChat.
Total loss:   L_forget + beta * L_retain

Both frozen and trainable logits come from the same PEFT model instance;
frozen logits use adapter layers disabled (equivalent to the original weights),
trainable logits use adapters enabled.

Usage:
    python3 training/train_urc.py --model qwen_instruct
    python3 training/train_urc.py --model ministral_base --beta 2.0
    python3 training/train_urc.py --model qwen_base --rank 8 --beta 1.0 --dry-run
"""

# ── Triton mock — must be first ───────────────────────────────────────────────
import sys
from unittest.mock import MagicMock

if sys.platform == "darwin":
    import importlib.abc
    import importlib.machinery

    _TRITON_MODS = (
        "triton", "triton.language", "triton.runtime", "triton.runtime.jit",
        "triton.backends", "triton.backends.compiler", "triton.backends.cuda",
        "triton.compiler", "triton.compiler.compiler",
    )
    for _m in _TRITON_MODS:
        sys.modules.setdefault(_m, MagicMock())

    class _TritonMockLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mock = sys.modules.get(spec.name) or MagicMock()
            mock.__name__ = spec.name
            mock.__package__ = spec.name.rpartition(".")[0] or spec.name
            mock.__path__ = []
            mock.__spec__ = spec
            mock.__loader__ = self
            sys.modules[spec.name] = mock
            return mock
        def exec_module(self, module):
            pass

    class _TritonMockFinder(importlib.abc.MetaPathFinder):
        _loader = _TritonMockLoader()
        def find_spec(self, fullname, path, target=None):
            if fullname == "triton" or fullname.startswith("triton."):
                return importlib.machinery.ModuleSpec(fullname, self._loader, is_package=True)
            return None

    sys.meta_path.insert(0, _TritonMockFinder())

# ── Standard imports ──────────────────────────────────────────────────────────
import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

if sys.platform == "darwin":
    torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*apply_chat_template.*tokenize=False.*")
log = logging.getLogger(__name__)

BASE_DIR          = Path(__file__).parent
REPO_ROOT         = BASE_DIR.parent
ACTIVATIONS_DIR   = REPO_ROOT / "mining-data" / "activations"
MINING_SEL_DIR    = REPO_ROOT / "mining-data" / "mining-selected"
RETAIN_DATA_PATH      = REPO_ROOT / "mining-data" / "sampled" / "ultrachat_retain_1000.jsonl"
RETAIN_KUQ_PATH       = REPO_ROOT / "mining-data" / "sampled" / "kuq_answerable_500.jsonl"
RETAIN_SQUAD_PATH     = REPO_ROOT / "mining-data" / "sampled" / "squad_answerable_500.jsonl"
RUNS_DIR          = BASE_DIR / "runs"

sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ── Model / layer registry ────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}

# File-prefix used for mining-selected JSONL names (e.g. "qwen-9b_kuq.jsonl")
MODEL_KEY_TO_FILE_PREFIX = {
    "qwen_instruct":      "qwen-9b",
    "qwen_base":          "qwen-9b-base",
    "ministral_instruct": "ministral-8b",
    "ministral_base":     "ministral-8b-base",
}

KUQ_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence.\n\nQuestion:\n{question}\n\nAnswer:"
)
SQUAD_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence using only the context."
    "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

# ── Training hyper-parameters ─────────────────────────────────────────────────

DEFAULT_RANK        = 8
DEFAULT_BETA        = 1.0
K_ANSWER_TOKENS     = 8
TOP_K_KL            = 100

LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"]

LEARNING_RATE       = 1e-5
EPOCHS              = 1
WARMUP_RATIO        = 0.03
MAX_GRAD_NORM       = 1.0
FORGET_BATCH_SIZE   = 4
RETAIN_BATCH_SIZE   = 4
GRAD_ACCUM_STEPS    = 4
LOG_EVERY           = 1


# ── Data loading ──────────────────────────────────────────────────────────────

def load_forget_data(model_key: str) -> list[dict]:
    prefix = MODEL_KEY_TO_FILE_PREFIX[model_key]
    records = []
    for dataset in ("kuq", "squad"):
        path = MINING_SEL_DIR / f"{prefix}_{dataset}.jsonl"
        if not path.exists():
            log.warning("  Forget data not found: %s", path.name)
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    log.info("  Loaded %d forget examples (%s)", len(records), prefix)
    return records


def load_retain_data() -> list[dict]:
    records = [json.loads(l) for l in RETAIN_DATA_PATH.read_text().splitlines() if l.strip()]

    # Add answerable QA pairs so the retain loss penalises "always say No"
    for path, template in (
        (RETAIN_KUQ_PATH,   KUQ_PROMPT_TEMPLATE),
        (RETAIN_SQUAD_PATH, SQUAD_PROMPT_TEMPLATE),
    ):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = template.format(
                question=row["question"],
                context=row.get("context", ""),
            )
            records.append({"prompt": prompt, "response": row["correct_answer"]})

    random.shuffle(records)
    log.info("  Loaded %d retain examples (ultrachat + answerable QA)", len(records))
    return records


# ── Subspace loading ──────────────────────────────────────────────────────────

SUBSPACE_VARIANTS = {
    "clean": {
        "filename": "retain_normalised_subspace_{model}_last25_r{rank}.pt",
        "build_cmd": "python3 mining-data/retain_normalised_subspace.py",
        "label":     "retainnorm",
        "method":    "retain_normalised_URC",
    },
    "raw": {
        "filename": "raw_subspace_{model}_last25_r{rank}.pt",
        "build_cmd": "python3 mining-data/raw_subspace.py",
        "label":     "raw",
        "method":    "raw_unsupported_URC",
    },
    "disc": {
        "filename": "discriminative_subspace_{model}_last25_r{rank}.pt",
        "build_cmd": "python3 mining-data/discriminative_subspace.py",
        "label":     "disc",
        "method":    "discriminative_URC_C-A",
    },
}


def load_subspace(
    model_key: str, rank: int, variant: str = "clean",
) -> tuple[list[torch.Tensor], list[int], float, Path]:
    """
    Returns (V_layers, layer_indices, proj_norm_scale, source_path).

    variant: which subspace bundle to load
        - "clean" : retain_normalised_subspace_*.pt   (cleaned contrasts; original)
        - "raw"   : raw_subspace_*.pt                  (uncleaned c_unsupported)
        - "disc"  : discriminative_subspace_*.pt       (Sigma_C - Sigma_A vs Sigma_R)
    """
    if variant not in SUBSPACE_VARIANTS:
        raise ValueError(
            f"Unknown subspace variant {variant!r}. "
            f"Choose from {list(SUBSPACE_VARIANTS)}."
        )
    spec = SUBSPACE_VARIANTS[variant]
    path = ACTIVATIONS_DIR / spec["filename"].format(model=model_key, rank=rank)
    if not path.exists():
        raise FileNotFoundError(
            f"Subspace file not found: {path}\n"
            f"Run: {spec['build_cmd']} --model {model_key} --rank {rank}"
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    V = bundle.get("V_retain_normalised")
    if V is None:
        V = bundle.get("V_retain_normalized")
    if V is None:
        raise KeyError("Subspace bundle missing V_retain_normalised / V_retain_normalized key.")
    layer_indices = bundle["layers"]
    V_layers = [V[i].float() for i in range(V.shape[0])]
    proj_norm_scale = float(bundle["commitment_projection"].mean())
    log.info("  Subspace[%s]: %s", variant, path.name)
    log.info("  Subspace: %d layers, rank=%d, D=%d  proj_norm_scale=%.4f",
             len(V_layers), rank, V.shape[1], proj_norm_scale)
    return V_layers, layer_indices, proj_norm_scale, path


# ── Model loading + LoRA ──────────────────────────────────────────────────────

def load_model_and_tokenizer(model_key: str, hf_token: str):
    model_id = MODEL_REGISTRY[model_key]
    log.info("  Loading %s ...", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )

    elif model_id.startswith("mistralai/"):
        from transformers import (
            MistralCommonBackend, Mistral3ForConditionalGeneration,
        )
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
        if sys.platform == "darwin":
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu", token=hf_token,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                tie_word_embeddings=False,
            )
            model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", token=hf_token,
                tie_word_embeddings=False,
            )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    log.info("  Loaded in %.1fs", time.time() - t0)
    return model, tokenizer


def apply_lora(model, model_key: str):
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA training.  pip install peft")

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def _encode(tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def tokenise_forget(
    tokenizer,
    prompt: str,
    answer_prefix: str,
    model_key: str,
) -> tuple[list[int], int, int]:
    """
    Returns (full_ids, answer_start, num_answer_tokens).
    For instruct models prompt is treated as plain text (same as mining templates).
    """
    prompt_ids = _encode(tokenizer, prompt)
    answer_ids = _encode(tokenizer, answer_prefix, add_special_tokens=False)
    answer_ids = answer_ids[:K_ANSWER_TOKENS]
    full_ids   = prompt_ids + answer_ids
    return full_ids, len(prompt_ids), len(answer_ids)


def tokenise_retain(
    tokenizer,
    prompt: str,
    response: str,
    model_key: str,
) -> tuple[list[int], int]:
    """
    Returns (full_ids, response_start).
    Matches extraction pipeline tokenisation.
    """
    is_base = "base" in model_key

    if is_base:
        prompt_ids   = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response, add_special_tokens=False)
        full_ids     = prompt_ids + response_ids
        return full_ids, len(prompt_ids)
    else:
        user_msg = [{"role": "user", "content": prompt}]
        if "ministral" in model_key:
            # MistralCommonBackend: return_dict=False returns a plain list[int]
            prompt_ids = tokenizer.apply_chat_template(
                user_msg, tokenize=True, add_generation_prompt=True, return_dict=False,
            )
        else:
            # Qwen: tokenize=False avoids BatchEncoding issues
            prompt_fmt = tokenizer.apply_chat_template(
                user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_fmt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_ids     = list(prompt_ids) + response_ids
        return full_ids, len(prompt_ids)


# ── Forward pass ──────────────────────────────────────────────────────────────

def forward_with_hiddens(
    model,
    input_ids: torch.Tensor,    # (1, seq_len)
    layer_indices: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Returns (logits, hidden_states_per_layer).
    logits: (1, seq_len, vocab)
    hidden per layer: (1, seq_len, D)
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
    )
    # hidden_states[0] = embedding, hidden_states[l+1] = layer l output
    all_hidden = outputs.hidden_states
    hiddens = [all_hidden[l + 1].float().cpu() for l in layer_indices]
    logits  = outputs.logits.float().cpu()
    return logits, hiddens


# ── Forget loss ───────────────────────────────────────────────────────────────

def compute_forget_loss(
    model,
    batch: list[dict],
    V_layers: list[torch.Tensor],
    layer_indices: list[int],
    tokenizer,
    model_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    For each example, project answer-token residuals onto V and minimise norm.
    Returns scalar loss and per-layer projection norm dict.
    """
    total_loss  = torch.tensor(0.0, requires_grad=True)
    layer_norms: dict[int, list[float]] = {l: [] for l in layer_indices}
    skipped = 0

    for rec in batch:
        prompt = rec["generation_prompt"]
        answer = rec.get("y_com_prefix_k8", "")
        if not prompt or not answer:
            skipped += 1
            continue

        try:
            full_ids, ans_start, n_ans = tokenise_forget(
                tokenizer, prompt, answer, model_key
            )
        except Exception as exc:
            log.debug("  Forget tokenisation error: %s", exc)
            skipped += 1
            continue

        if n_ans == 0:
            skipped += 1
            continue

        input_tensor = torch.tensor([full_ids], dtype=torch.long)

        _, hiddens = forward_with_hiddens(model, input_tensor, layer_indices)

        ex_loss = torch.tensor(0.0, requires_grad=True)
        for li, (l_idx, h) in enumerate(zip(layer_indices, hiddens)):
            # h: (1, seq_len, D)
            ans_h = h[0, ans_start:ans_start + n_ans, :]  # (n_ans, D)
            V_l   = V_layers[li].to(ans_h.device)          # (D, rank)
            proj  = ans_h @ V_l                             # (n_ans, rank)
            # Expected ‖proj_vec‖² per answer token: sum across rank dims, mean across tokens.
            # Matches `proj_norm_scale = tr(VᵀΣ_C V)` so L_forget_scaled ≈ 1 at start.
            l_loss = (proj ** 2).sum(dim=-1).mean()
            ex_loss = ex_loss + l_loss
            layer_norms[l_idx].append(float(proj.detach().norm(dim=-1).mean()))

        ex_loss = ex_loss / len(layer_indices)
        total_loss = total_loss + ex_loss

    n_valid = len(batch) - skipped
    if n_valid > 0:
        total_loss = total_loss / n_valid

    mean_norms = {l: (sum(v) / len(v) if v else 0.0) for l, v in layer_norms.items()}
    return total_loss, {"layer_proj_norms": mean_norms, "skipped": skipped}


# ── Retain loss ───────────────────────────────────────────────────────────────

def compute_retain_loss(
    model,
    batch: list[dict],
    tokenizer,
    model_key: str,
    device: torch.device,
    top_k: int = TOP_K_KL,
) -> tuple[torch.Tensor, dict]:
    """
    top-100 KL(p_frozen || p_trainable) over response tokens.

    Frozen forward:    eval mode + adapters disabled + no_grad
                       (deterministic; dropout fully off)
    Trainable forward: train mode + adapters enabled
                       (LoRA dropout active — acceptable for training)
    """
    total_kl = torch.tensor(0.0, requires_grad=True)
    n_valid  = 0

    for rec in batch:
        prompt   = rec.get("prompt", "")
        response = rec.get("response", "")
        if not prompt or not response:
            continue

        try:
            full_ids, resp_start = tokenise_retain(
                tokenizer, prompt, response, model_key
            )
        except Exception as exc:
            log.debug("  Retain tokenisation error: %s", exc)
            continue

        n_resp = len(full_ids) - resp_start
        if n_resp <= 0:
            continue

        input_tensor = torch.tensor([full_ids], dtype=torch.long)

        # Frozen forward: eval mode kills all dropout; adapters off = original weights
        model.eval()
        model.disable_adapter_layers()
        with torch.no_grad():
            frozen_logits, _ = forward_with_hiddens(model, input_tensor, [])
        model.enable_adapter_layers()
        model.train()

        # Trainable forward: adapters active, gradient flows; dropout active (train mode)
        trainable_logits, _ = forward_with_hiddens(model, input_tensor, [])

        # Slice response tokens only: positions [resp_start : end - 1]
        # We predict token t+1 from position t, so take logits at [resp_start-1 : -1]
        # and compare to targets at [resp_start : end]
        # For KL distillation we compare distributions at each response position
        resp_logits_frozen    = frozen_logits[0, resp_start - 1:-1, :]    # (n_resp, V)
        resp_logits_trainable = trainable_logits[0, resp_start - 1:-1, :] # (n_resp, V)

        # Top-k KL: pick top-k tokens from frozen, restrict both distributions to those
        topk_vals, topk_idx = resp_logits_frozen.topk(top_k, dim=-1)  # (n_resp, top_k)

        frozen_topk    = topk_vals                                           # (n_resp, top_k)
        trainable_topk = resp_logits_trainable.gather(-1, topk_idx)         # (n_resp, top_k)

        log_p_frozen    = F.log_softmax(frozen_topk,    dim=-1)
        log_p_trainable = F.log_softmax(trainable_topk, dim=-1)
        p_frozen        = log_p_frozen.exp()

        # KL(p_frozen || p_trainable) = sum(p_f * (log_pf - log_pt))
        kl = (p_frozen * (log_p_frozen - log_p_trainable)).sum(dim=-1).mean()
        total_kl = total_kl + kl
        n_valid += 1

    if n_valid > 0:
        total_kl = total_kl / n_valid

    return total_kl, {"n_retain_valid": n_valid}


# ── LR scheduler ─────────────────────────────────────────────────────────────

def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model_key: str, beta: float, rank: int, dry_run: bool = False,
          epochs: int = EPOCHS, lr: float = LEARNING_RATE,
          es_patience: int = 20, es_delta: float = 0.005, kl_max: float = 0.15,
          subspace: str = "clean") -> None:
    hf_token = os.environ.get("HF_TOKEN", "")

    label = SUBSPACE_VARIANTS[subspace]["label"]
    run_name = (
        f"{model_key}_{label}_urc_last25_r{rank}"
        f"_beta{beta:g}_ep{epochs}_lr{lr:.0e}"
    )
    out_dir  = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run: %s", run_name)
    log.info("Output dir: %s", out_dir)

    # ── Load data ────────────────────────────────────────────────────────────
    forget_data = load_forget_data(model_key)
    retain_data = load_retain_data()

    if dry_run:
        forget_data = forget_data[:8]
        retain_data = retain_data[:8]
        log.info("  Dry run: trimmed to 8 examples each")

    # ── Load subspace ────────────────────────────────────────────────────────
    V_layers, layer_indices, proj_norm_scale, subspace_path = load_subspace(
        model_key, rank, variant=subspace,
    )

    # ── Load model + apply LoRA ──────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_key, hf_token)
    model = apply_lora(model, model_key)
    model.train()

    device = next(model.parameters()).device

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    forget_steps_per_epoch = math.ceil(len(forget_data) / FORGET_BATCH_SIZE)
    total_optim_steps = math.ceil(forget_steps_per_epoch / GRAD_ACCUM_STEPS) * epochs
    warmup_steps = max(1, int(total_optim_steps * WARMUP_RATIO))
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_optim_steps)
    log.info("  Total optimiser steps: %d  (warmup: %d)", total_optim_steps, warmup_steps)

    # ── Logging setup ────────────────────────────────────────────────────────
    log_path = out_dir / "loss_log.csv"
    log_fields = [
        "step", "L_total", "L_forget", "top100_retain_KL",
        "mean_proj_norm", "learning_rate", "grad_norm",
    ]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    # ── Training loop ────────────────────────────────────────────────────────
    random.shuffle(forget_data)
    retain_cycle = retain_data.copy()
    random.shuffle(retain_cycle)
    retain_pos = 0

    def next_retain_batch() -> list[dict]:
        nonlocal retain_cycle, retain_pos
        batch = []
        for _ in range(RETAIN_BATCH_SIZE):
            if retain_pos >= len(retain_cycle):
                retain_cycle = retain_data.copy()
                random.shuffle(retain_cycle)
                retain_pos = 0
            batch.append(retain_cycle[retain_pos])
            retain_pos += 1
        return batch

    optim_step = 0
    forget_pos = 0
    accum_forget = torch.tensor(0.0)
    accum_retain = torch.tensor(0.0)
    accum_proj_norms: list[float] = []

    # Track initial and final logged values for train_summary
    summary_initial: dict = {}
    summary_final:   dict = {}

    # Early stopping state
    es_best_forget: float = float("inf")
    es_steps_no_improve: int = 0
    stopped_early: bool = False

    # Previous step values for delta reporting
    prev_forget: float | None = None
    prev_retain: float | None = None
    prev_proj:   float | None = None
    prev_proj:   float | None = None

    pbar = tqdm(total=total_optim_steps, desc=run_name, unit="step", dynamic_ncols=True)

    for _epoch in range(epochs):
        random.shuffle(forget_data)
        forget_pos = 0

        while forget_pos < len(forget_data):
            forget_batch = forget_data[forget_pos:forget_pos + FORGET_BATCH_SIZE]
            retain_batch = next_retain_batch()
            forget_pos += FORGET_BATCH_SIZE

            # Forget step
            l_forget, finfo = compute_forget_loss(
                model, forget_batch, V_layers, layer_indices,
                tokenizer, model_key, device,
            )
            # Retain step
            l_retain, rinfo = compute_retain_loss(
                model, retain_batch, tokenizer, model_key, device,
            )

            l_forget_scaled = l_forget / proj_norm_scale
            l_total = l_forget_scaled + beta * l_retain
            l_total_scaled = l_total / GRAD_ACCUM_STEPS
            l_total_scaled.backward()

            accum_step = (forget_pos // FORGET_BATCH_SIZE)
            log.info(
                "  accum %3d/%d  Lf=%.4f  Lr=%.4f",
                accum_step, math.ceil(len(forget_data) / FORGET_BATCH_SIZE),
                float(l_forget_scaled.detach()), float(l_retain.detach()),
            )

            accum_forget = accum_forget + l_forget_scaled.detach()
            accum_retain = accum_retain + l_retain.detach()
            layer_norms  = finfo.get("layer_proj_norms", {})
            if layer_norms:
                accum_proj_norms.append(sum(layer_norms.values()) / len(layer_norms))

            # Optimiser step after accumulation
            inner_step = (forget_pos // FORGET_BATCH_SIZE) % GRAD_ACCUM_STEPS
            if inner_step == 0 or forget_pos >= len(forget_data):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                # Logging
                if optim_step % LOG_EVERY == 0 or optim_step == 1:
                    n_accum    = max(GRAD_ACCUM_STEPS, 1)
                    avg_forget = float(accum_forget) / n_accum
                    avg_retain = float(accum_retain) / n_accum
                    avg_total  = avg_forget + beta * avg_retain
                    mean_proj  = (sum(accum_proj_norms) / len(accum_proj_norms)
                                  if accum_proj_norms else 0.0)
                    lr_now = scheduler.get_last_lr()[0]

                    def _delta(cur, prev):
                        if prev is None:
                            return "     —"
                        d = cur - prev
                        return f"{d:+.4f}"

                    log.info(
                        "  step=%d/%d  L_total=%.4f  "
                        "L_forget=%.4f(%s)  KL=%.4f(%s)  "
                        "proj=%.4f(%s)  lr=%.2e  grad=%.3f",
                        optim_step, total_optim_steps, avg_total,
                        avg_forget, _delta(avg_forget, prev_forget),
                        avg_retain, _delta(avg_retain, prev_retain),
                        mean_proj,  _delta(mean_proj,  prev_proj),
                        lr_now, float(grad_norm),
                    )

                    prev_forget = avg_forget
                    prev_retain = avg_retain
                    prev_proj   = mean_proj
                    row = {
                        "step":              optim_step,
                        "L_total":           round(avg_total,  6),
                        "L_forget":          round(avg_forget, 6),
                        "top100_retain_KL":  round(avg_retain, 6),
                        "mean_proj_norm":    round(mean_proj,  6),
                        "learning_rate":     lr_now,
                        "grad_norm":         round(float(grad_norm), 4),
                    }
                    log_writer.writerow(row)
                    log_file.flush()

                    if not summary_initial:
                        summary_initial = {
                            "initial_L_total":       round(avg_total,  6),
                            "initial_L_forget":      round(avg_forget, 6),
                            "initial_top100_KL":     round(avg_retain, 6),
                            "initial_mean_proj_norm": round(mean_proj, 6),
                        }
                    summary_final = {
                        "final_L_total":       round(avg_total,  6),
                        "final_L_forget":      round(avg_forget, 6),
                        "final_top100_KL":     round(avg_retain, 6),
                        "final_mean_proj_norm": round(mean_proj, 6),
                    }

                accum_forget    = torch.tensor(0.0)
                accum_retain    = torch.tensor(0.0)
                accum_proj_norms = []
                pbar.update(1)
                pbar.set_postfix(
                    Lf=f"{float(l_forget_scaled.detach()):.3f}",
                    Lr=f"{float(l_retain.detach()):.3f}",
                )

                # ── Early stopping checks ─────────────────────────────────
                if es_patience > 0:
                    if avg_forget < es_best_forget - es_delta:
                        es_best_forget = avg_forget
                        es_steps_no_improve = 0
                    else:
                        es_steps_no_improve += 1
                    if es_steps_no_improve >= es_patience:
                        log.info(
                            "  Early stop: L_forget hasn't improved by %.4f "
                            "in %d steps (best=%.4f)",
                            es_delta, es_patience, es_best_forget,
                        )
                        stopped_early = True
                        break

                if kl_max > 0 and avg_retain > kl_max:
                    log.info(
                        "  Early stop: KL=%.4f exceeded kl_max=%.4f",
                        avg_retain, kl_max,
                    )
                    stopped_early = True
                    break

        if stopped_early:
            break

    pbar.close()
    log_file.close()

    # ── Save train_summary.json ───────────────────────────────────────────────
    # Read the full loss_log to compute deltas from first to last logged step
    loss_log_rows: list[dict] = []
    try:
        with open(log_path) as f:
            reader = csv.DictReader(f)
            loss_log_rows = list(reader)
    except Exception:
        pass

    def _total_delta(field: str) -> float | None:
        if len(loss_log_rows) < 2:
            return None
        try:
            return round(float(loss_log_rows[-1][field]) - float(loss_log_rows[0][field]), 6)
        except (KeyError, ValueError):
            return None

    train_summary = {
        **summary_initial,
        **summary_final,
        "proj_norm_scale":     round(proj_norm_scale, 6),
        "delta_L_forget":       _total_delta("L_forget"),
        "delta_top100_KL":      _total_delta("top100_retain_KL"),
        "delta_mean_proj_norm": _total_delta("mean_proj_norm"),
        "num_steps":            optim_step,
        "num_forget_examples":  len(forget_data),
        "num_retain_examples":  len(retain_data),
        "stopped_early":        stopped_early,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2)
    )
    log.info("  train_summary.json written")

    # ── Save outputs ─────────────────────────────────────────────────────────
    log.info("Saving adapter weights -> %s", out_dir)
    model.save_pretrained(out_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)

    training_config = {
        "model_key":            model_key,
        "model_id":             MODEL_REGISTRY[model_key],
        "method":               SUBSPACE_VARIANTS[subspace]["method"],
        "subspace_variant":     subspace,
        "lora_rank":            LORA_R,
        "lora_alpha":           LORA_ALPHA,
        "lora_dropout":         LORA_DROPOUT,
        "target_modules":       LORA_TARGET_MODULES,
        "learning_rate":        lr,
        "epochs":               epochs,
        "warmup_ratio":         WARMUP_RATIO,
        "max_grad_norm":        MAX_GRAD_NORM,
        "forget_batch_size":    FORGET_BATCH_SIZE,
        "retain_batch_size":    RETAIN_BATCH_SIZE,
        "grad_accum_steps":     GRAD_ACCUM_STEPS,
        "beta":                 beta,
        "k_answer_tokens":      K_ANSWER_TOKENS,
        "top_k_kl":             TOP_K_KL,
        "proj_norm_scale":      proj_norm_scale,
        "forget_examples":      len(forget_data),
        "retain_examples":      len(retain_data),
        "total_optim_steps":    total_optim_steps,
    }
    (out_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2)
    )

    subspace_config = {
        "method":            SUBSPACE_VARIANTS[subspace]["method"],
        "variant":           subspace,
        "basis_file":        subspace_path.name,
        "basis_key":         "V_retain_normalised",
        "rank":              rank,
        "layers":            "last_25_percent",
        "layer_indices":     layer_indices,
        "k_answer_tokens":   K_ANSWER_TOKENS,
        "beta":              beta,
        "proj_norm_scale":   proj_norm_scale,
        "proj_norm_scale_source": "commitment_projection mean from subspace bundle (N=1000)",
    }
    (out_dir / "subspace_config.json").write_text(
        json.dumps(subspace_config, indent=2)
    )

    log.info("Done. Outputs in %s", out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="URC training: retain-normalised subspace unlearning."
    )
    p.add_argument("--model",   choices=list(MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--beta",    type=float, default=DEFAULT_BETA,
                   help="Weight on retain KL loss (default: 1.0)")
    p.add_argument("--rank",    type=int,   default=DEFAULT_RANK,
                   help="Subspace rank to load (default: 8)")
    p.add_argument("--subspace", choices=list(SUBSPACE_VARIANTS), default="clean",
                   help=("Which subspace bundle to suppress: "
                         "'clean' = retain_normalised_subspace_*.pt (default), "
                         "'raw' = raw_subspace_*.pt (uncleaned contrasts), "
                         "'disc' = discriminative_subspace_*.pt (Sigma_C - Sigma_A vs Sigma_R)"))
    p.add_argument("--epochs",  type=int,   default=EPOCHS,
                   help=f"Number of training epochs (default: {EPOCHS})")
    p.add_argument("--lr",      type=float, default=LEARNING_RATE,
                   help=f"Peak learning rate (default: {LEARNING_RATE})")
    p.add_argument("--es-patience", type=int,   default=20,
                   help="Early stopping: stop if L_forget doesn't improve by "
                        "--es-delta over this many steps (default: 20, 0=off)")
    p.add_argument("--es-delta",    type=float, default=0.05,
                   help="Early stopping: minimum improvement in L_forget (default: 0.05)")
    p.add_argument("--kl-max",      type=float, default=0.15,
                   help="Stop if retain KL exceeds this value (default: 0.15, 0=off)")
    p.add_argument("--dry-run", action="store_true",
                   help="Use only 8 examples per split for a quick smoke test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("URC training")
    log.info("  Model    : %s", args.model)
    log.info("  Beta     : %g", args.beta)
    log.info("  Rank     : %d", args.rank)
    log.info("  Subspace : %s", args.subspace)
    log.info("  Epochs   : %d", args.epochs)
    log.info("  LR       : %g", args.lr)
    if args.dry_run:
        log.info("  Mode     : DRY RUN")
    train(args.model, beta=args.beta, rank=args.rank, dry_run=args.dry_run,
          epochs=args.epochs, lr=args.lr,
          es_patience=args.es_patience, es_delta=args.es_delta, kl_max=args.kl_max,
          subspace=args.subspace)


if __name__ == "__main__":
    main()
