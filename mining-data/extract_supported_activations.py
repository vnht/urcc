#!/usr/bin/env python3
"""
Extract teacher-forced activations for supported-answer contrasts.

For each record in a tokenised supported-answer file:
    - Forward pass with  prompt + correct_prefix
    - Forward pass with  prompt + abstained_prefix
    - Mean residual activation over answer-prefix tokens, selected layers only
    - Contrast: c_supported_l = r_correct_l - r_abstained_l

Selected layer sets (0-indexed transformer layers):
    Qwen:      [20, 28, 29, 30]
    Ministral: [22, 31, 32, 33]

Saves one .pt bundle per model:
    mining-data/activations/supported_answer_contrasts_{model_key}.pt

Contents:
    {
        "model":       str,
        "layers":      list[int],
        "k":           int,
        "datasets":    list[str],
        "prompt_ids":  list[str],
        "r_correct":   tensor[N, L, D],
        "r_abstained": tensor[N, L, D],
        "c_supported": tensor[N, L, D],
    }

Usage:
    python3 mining-data/extract_supported_activations.py --model qwen_instruct
    python3 mining-data/extract_supported_activations.py --model qwen_base
    python3 mining-data/extract_supported_activations.py --model ministral_instruct
    python3 mining-data/extract_supported_activations.py --model ministral_base
"""

# ── Triton mock — must be first, before any other import ─────────────────────
import sys
from unittest.mock import MagicMock

if sys.platform == "darwin":
    import importlib.abc
    import importlib.machinery

    _TRITON_MODS = (
        "triton",
        "triton.language",
        "triton.runtime",
        "triton.runtime.jit",
        "triton.backends",
        "triton.backends.compiler",
        "triton.backends.cuda",
        "triton.compiler",
        "triton.compiler.compiler",
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
                return importlib.machinery.ModuleSpec(
                    fullname, self._loader, is_package=True
                )
            return None

    sys.meta_path.insert(0, _TritonMockFinder())

# ── Standard imports ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

if sys.platform == "darwin":
    torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

TOKENISED_DIR  = BASE_DIR / "tokenised"
ACTIVATIONS_DIR = BASE_DIR / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)

# ─── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}

# Selected layers (0-indexed transformer blocks, excluding embedding layer)
SELECTED_LAYERS = {
    "qwen_instruct":      [24, 25, 26, 27, 28, 29, 30, 31],
    "qwen_base":          [24, 25, 26, 27, 28, 29, 30, 31],
    "ministral_instruct": [25, 26, 27, 28, 29, 30, 31, 32, 33],
    "ministral_base":     [25, 26, 27, 28, 29, 30, 31, 32, 33],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_key: str, model_id: str, hf_token: str):
    log.info("Loading model %s …", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM  # type: ignore[import]
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
    elif model_id.startswith("mistralai/"):
        from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config  # type: ignore[import]
        if sys.platform == "darwin":
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="cpu",
                token=hf_token,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                tie_word_embeddings=False,
            )
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                token=hf_token,
                tie_word_embeddings=False,
            )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    model.eval()
    log.info("  Loaded in %.1fs", time.time() - t0)
    return model


# ─── Activation extraction ────────────────────────────────────────────────────

def forward_hidden_states(
    model,
    input_ids: torch.Tensor,   # (1, seq_len)
    layer_indices: list[int],
) -> list[torch.Tensor]:
    """
    Run one forward pass. Returns a list of hidden-state tensors for the
    requested layer indices only, each shape (seq_len, hidden_dim).
    Layer indices are 0-based transformer block indices (excluding embedding).
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # hidden_states[0] = embedding; [1..] = transformer layer outputs
    all_hidden = outputs.hidden_states  # tuple of (1, seq, D)
    return [all_hidden[i + 1][0].float() for i in layer_indices]


def mean_answer_activation(
    hidden_states: list[torch.Tensor],  # each (seq_len, D)
    prompt_len: int,
    k: int,
) -> torch.Tensor:
    """Mean activation over [prompt_len : prompt_len+k]. Returns (L, D)."""
    return torch.stack([
        h[prompt_len: prompt_len + k, :].mean(dim=0)
        for h in hidden_states
    ])


# ─── Quality stats ────────────────────────────────────────────────────────────

def _p(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(int(len(s) * pct), len(s) - 1)
    return s[idx]


def log_quality_stats(
    records: list[dict],
    kept_mask: list[bool],
    layer_indices: list[int],
    r_correct_list: list[torch.Tensor],
    r_abstained_list: list[torch.Tensor],
) -> None:
    by_ds: dict[str, list[int]] = defaultdict(list)  # dataset → kept indices
    kept_idx = 0
    for i, (rec, kept) in enumerate(zip(records, kept_mask)):
        if kept:
            by_ds[rec["dataset"]].append(kept_idx)
            kept_idx += 1

    log.info("")
    log.info("Quality stats:")
    log.info("  %-10s %6s %6s %6s %8s %8s %8s %8s %8s",
             "dataset", "in", "out", "drop%",
             "cor_mean", "cor_med", "cor_p90",
             "abs_mean", "abs_p90")

    total_in  = len(records)
    total_out = sum(kept_mask)

    for ds_label, indices in sorted(by_ds.items()):
        n_in  = sum(1 for r in records if r["dataset"] == ds_label)
        n_out = len(indices)
        drop  = (n_in - n_out) / n_in * 100 if n_in else 0
        cor_lens = [records[i]["k"] for i in range(total_in)
                    if kept_mask[i] and records[i]["dataset"] == ds_label]
        abs_lens = [len(records[i].get("abstained_prefix_token_ids", [])) for i in range(total_in)
                    if kept_mask[i] and records[i]["dataset"] == ds_label]
        # Use actual token counts from records
        cor_k = [len(records[i]["correct_prefix_token_ids"]) for i in range(total_in)
                 if kept_mask[i] and records[i]["dataset"] == ds_label]
        abs_k = [len(records[i]["abstained_prefix_token_ids"]) for i in range(total_in)
                 if kept_mask[i] and records[i]["dataset"] == ds_label]
        log.info(
            "  %-10s %6d %6d %5.1f%% %8.1f %8.1f %8.1f %8.1f %8.1f",
            ds_label, n_in, n_out, drop,
            statistics.mean(cor_k) if cor_k else 0,
            statistics.median(cor_k) if cor_k else 0,
            _p(cor_k, 0.9),
            statistics.mean(abs_k) if abs_k else 0,
            _p(abs_k, 0.9),
        )

    log.info("  %-10s %6d %6d %5.1f%%", "TOTAL", total_in, total_out,
             (total_in - total_out) / total_in * 100 if total_in else 0)

    # Layer-wise contrast norm
    if r_correct_list and r_abstained_list:
        r_cor = torch.stack(r_correct_list)    # (N, L, D)
        r_abs = torch.stack(r_abstained_list)  # (N, L, D)
        c = r_cor - r_abs
        layer_norms = c.norm(dim=-1).mean(dim=0).tolist()  # (L,)
        log.info("")
        log.info("  Layer-wise mean ‖c_supported‖:")
        for li, norm_val in zip(layer_indices, layer_norms):
            log.info("    L%-3d  %.4f", li, norm_val)


# ─── Main extraction loop ─────────────────────────────────────────────────────

def run(model_key: str) -> None:
    model_id     = MODEL_REGISTRY[model_key]
    layer_indices = SELECTED_LAYERS[model_key]
    hf_token     = os.environ.get("HF_TOKEN", "")

    tok_path = TOKENISED_DIR / f"tokenised_supported_answer_contrasts_{model_key}.jsonl"
    if not tok_path.exists():
        log.error("Tokenised file not found: %s", tok_path)
        log.error("Run build_supported_contrasts.py first.")
        sys.exit(1)

    records = load_jsonl(tok_path)
    log.info("Loaded %d records from %s", len(records), tok_path.name)
    log.info("Selected layers: %s", layer_indices)

    model = load_model(model_key, model_id, hf_token)

    r_correct_list:  list[torch.Tensor] = []
    r_abstained_list: list[torch.Tensor] = []
    kept_mask: list[bool] = []
    meta_datasets: list[str] = []
    meta_prompt_ids: list[str] = []

    t0 = time.time()
    for i, rec in enumerate(records):
        p_ids  = rec["prompt_token_ids"]
        c_ids  = rec["correct_prefix_token_ids"]
        a_ids  = rec["abstained_prefix_token_ids"]
        k      = rec["k"]
        p_len  = len(p_ids)

        cor_input = torch.tensor([p_ids + c_ids], dtype=torch.long)
        abs_input = torch.tensor([p_ids + a_ids], dtype=torch.long)

        try:
            cor_hidden = forward_hidden_states(model, cor_input, layer_indices)
            abs_hidden = forward_hidden_states(model, abs_input, layer_indices)
        except Exception as exc:
            log.warning("  [%d] Forward pass error: %s — skipping", i + 1, exc)
            kept_mask.append(False)
            continue

        r_cor = mean_answer_activation(cor_hidden, p_len, k)   # (L, D)
        r_abs = mean_answer_activation(abs_hidden, p_len, k)   # (L, D)

        r_correct_list.append(r_cor.cpu())
        r_abstained_list.append(r_abs.cpu())
        kept_mask.append(True)
        meta_datasets.append(rec["dataset"])
        meta_prompt_ids.append(rec["prompt_id"])

        elapsed = time.time() - t0
        if (i + 1) % 50 == 0 or i == 0:
            log.info("  [%d/%d] %.0fs elapsed", i + 1, len(records), elapsed)

    r_correct  = torch.stack(r_correct_list)    # (N, L, D)
    r_abstained = torch.stack(r_abstained_list)  # (N, L, D)
    c_supported = r_correct - r_abstained        # (N, L, D)

    bundle = {
        "model":       model_id,
        "layers":      layer_indices,
        "k":           K,
        "datasets":    meta_datasets,
        "prompt_ids":  meta_prompt_ids,
        "r_correct":   r_correct,
        "r_abstained": r_abstained,
        "c_supported": c_supported,
    }

    out_path = ACTIVATIONS_DIR / f"supported_answer_contrasts_{model_key}.pt"
    torch.save(bundle, out_path)
    log.info("")
    log.info("Saved → %s", out_path.name)
    log.info("  Shape: N=%d  L=%d  D=%d", *r_correct.shape)

    log_quality_stats(records, kept_mask, layer_indices, r_correct_list, r_abstained_list)

    del model, r_correct_list, r_abstained_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─── CLI ─────────────────────────────────────────────────────────────────────

K = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract supported-answer activation contrasts")
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        nargs="*",
        default=None,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = args.model or list(MODEL_REGISTRY.keys())
    log.info("Extract supported activations  models=%s", models)
    log.info("Tokenised dir  : %s", TOKENISED_DIR)
    log.info("Activations dir: %s", ACTIVATIONS_DIR)
    for model_key in models:
        run(model_key)


if __name__ == "__main__":
    main()
