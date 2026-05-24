#!/usr/bin/env python3
"""
Extract teacher-forced activations and compute unsupported-commitment contrasts.

For each record in a tokenised contrast file:
    - Run forward pass with  prompt + committed_prefix
    - Run forward pass with  prompt + abstained_prefix
    - Compute mean residual activation over the k answer-prefix tokens per layer
    - Contrast: c_u_l = r_committed_l - r_abstained_l

Saves one .pt bundle per model:
    mining-data/activations/unsupported_commitment_contrasts_{model_key}.pt

Contents:
    {
        "model":       str,
        "layers":      list[int],        # 0-indexed transformer layers
        "k":           int,
        "datasets":    list[str],
        "prompt_ids":  list[str],
        "r_committed": tensor[N, L, D],
        "r_abstained": tensor[N, L, D],
        "c_unsupported": tensor[N, L, D],
    }

Usage:
    python3 mining-data/extract_activations.py --model qwen_instruct
    python3 mining-data/extract_activations.py --model qwen_base
    python3 mining-data/extract_activations.py --model ministral_instruct
    python3 mining-data/extract_activations.py --model ministral_base
"""

# ── Triton mock — must be first, before any other import ─────────────────────
# On macOS, triton (CUDA-only) is not installed. Several libraries
# (torch._inductor, transformers.integrations.finegrained_fp8, etc.) do
# `import triton` or `import triton.*` at module load time, before we have a
# chance to intercept them. Pre-populate sys.modules with MagicMocks for every
# known triton path, then install a MetaPathFinder as a catch-all for any
# submodule names that vary across torch/transformers versions.
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
import time
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

TOKENISED_DIR = BASE_DIR / "tokenised"
ACTIVATIONS_DIR = BASE_DIR / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)

# ─── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "qwen_instruct":     "Qwen/Qwen3.5-9B",
    "qwen_base":         "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral_base":    "mistralai/Ministral-3-8B-Base-2512",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Model loading ────────────────────────────────────────────────────────────


def load_model(model_key: str, model_id: str, hf_token: str):
    """Load model for inference (bfloat16, device_map=auto)."""
    log.info("Loading model %s …", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM  # type: ignore[import]
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
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
    input_ids: torch.Tensor,   # shape: (1, seq_len)
) -> list[torch.Tensor]:
    """
    Run one forward pass and return hidden states for each transformer layer.
    Returns list of tensors, each shape (seq_len, hidden_dim).
    Excludes the embedding layer (index 0 of model output).
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

    # outputs.hidden_states: tuple of (1, seq_len, hidden_dim), len = num_layers + 1
    # Index 0 is the embedding; indices 1..N are transformer layer outputs.
    hidden = [h[0].float() for h in outputs.hidden_states[1:]]  # list of (seq, D)
    return hidden


def mean_answer_activation(
    hidden_states: list[torch.Tensor],
    prompt_len: int,
    k: int,
) -> torch.Tensor:
    """
    Mean activation over the answer-prefix positions [prompt_len : prompt_len+k].
    Returns tensor of shape (num_layers, hidden_dim).
    """
    layer_means = []
    for h in hidden_states:
        # h: (seq_len, hidden_dim)
        ans_slice = h[prompt_len: prompt_len + k, :]   # (k, D)
        layer_means.append(ans_slice.mean(dim=0))       # (D,)
    return torch.stack(layer_means)   # (L, D)


# ─── Quality stats ────────────────────────────────────────────────────────────


def log_quality_stats(records: list[dict], kept_mask: list[bool]) -> None:
    from collections import Counter
    import statistics

    by_ds: dict[str, list] = defaultdict_list()
    for rec, kept in zip(records, kept_mask):
        if kept:
            by_ds[rec["dataset"]].append(rec)

    log.info("")
    log.info("Quality stats:")
    log.info("  %-12s %8s %8s %8s %8s %8s %8s %8s %8s",
             "dataset",
             "in", "out", "drop%",
             "com_mean", "com_med", "com_p90",
             "abs_mean", "abs_p90")

    total_in = len(records)
    total_out = sum(kept_mask)
    for ds_label, ds_recs in sorted(by_ds.items()):
        com_lens = [len(r["committed_prefix_token_ids"]) for r in ds_recs]
        abs_lens = [len(r["abstained_prefix_token_ids"]) for r in ds_recs]
        n_in  = sum(1 for r in records if r["dataset"] == ds_label)
        n_out = len(ds_recs)
        drop  = (n_in - n_out) / n_in * 100 if n_in else 0
        log.info(
            "  %-12s %8d %8d %7.1f%% %8.1f %8.1f %8.1f %8.1f %8.1f",
            ds_label, n_in, n_out, drop,
            statistics.mean(com_lens), statistics.median(com_lens),
            _p90(com_lens),
            statistics.mean(abs_lens),
            _p90(abs_lens),
        )

    log.info("  %-12s %8d %8d %7.1f%%", "TOTAL", total_in, total_out,
             (total_in - total_out) / total_in * 100 if total_in else 0)


def _p90(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * 0.9)
    return s[min(idx, len(s) - 1)]


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)


# ─── Main extraction loop ─────────────────────────────────────────────────────


def run(model_key: str) -> None:
    model_id = MODEL_REGISTRY[model_key]
    hf_token = os.environ.get("HF_TOKEN", "")

    tok_path = TOKENISED_DIR / f"tokenised_unsupported_contrasts_{model_key}.jsonl"
    if not tok_path.exists():
        log.error("Tokenised file not found: %s", tok_path)
        log.error("Run build_contrasts.py first.")
        sys.exit(1)

    records = load_jsonl(tok_path)
    log.info("Loaded %d records from %s", len(records), tok_path.name)

    model = load_model(model_key, model_id, hf_token)
    # Mistral3ForConditionalGeneration wraps text blocks as model.language_model.layers;
    # Qwen / plain Mistral expose them as model.layers directly.
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    num_layers = len(list(inner.layers))
    log.info("  Transformer layers: %d", num_layers)

    r_committed_list: list[torch.Tensor] = []
    r_abstained_list: list[torch.Tensor] = []
    kept_mask: list[bool] = []
    meta_datasets: list[str] = []
    meta_prompt_ids: list[str] = []

    t0 = time.time()
    for i, rec in enumerate(records):
        p_ids = rec["prompt_token_ids"]
        c_ids = rec["committed_prefix_token_ids"]
        a_ids = rec["abstained_prefix_token_ids"]
        k = rec["k"]

        # Build full input sequences
        com_input = torch.tensor([p_ids + c_ids], dtype=torch.long)
        abs_input = torch.tensor([p_ids + a_ids], dtype=torch.long)
        p_len = len(p_ids)

        try:
            com_hidden = forward_hidden_states(model, com_input)
            abs_hidden = forward_hidden_states(model, abs_input)
        except Exception as exc:
            log.warning("  [%d] Forward pass error: %s — skipping", i + 1, exc)
            kept_mask.append(False)
            continue

        r_com = mean_answer_activation(com_hidden, p_len, k)   # (L, D)
        r_abs = mean_answer_activation(abs_hidden, p_len, k)   # (L, D)

        r_committed_list.append(r_com.cpu())
        r_abstained_list.append(r_abs.cpu())
        kept_mask.append(True)
        meta_datasets.append(rec["dataset"])
        meta_prompt_ids.append(rec["prompt_id"])

        elapsed = time.time() - t0
        if (i + 1) % 50 == 0 or i == 0:
            log.info("  [%d/%d] %.0fs elapsed", i + 1, len(records), elapsed)

    # Stack tensors
    r_committed = torch.stack(r_committed_list)   # (N, L, D)
    r_abstained = torch.stack(r_abstained_list)   # (N, L, D)
    c_unsupported = r_committed - r_abstained      # (N, L, D)

    bundle = {
        "model": model_id,
        "layers": list(range(num_layers)),
        "k": records[0]["k"] if records else K,
        "datasets": meta_datasets,
        "prompt_ids": meta_prompt_ids,
        "r_committed": r_committed,
        "r_abstained": r_abstained,
        "c_unsupported": c_unsupported,
    }

    out_path = ACTIVATIONS_DIR / f"unsupported_commitment_contrasts_{model_key}.pt"
    torch.save(bundle, out_path)
    log.info("")
    log.info("Saved → %s", out_path.name)
    log.info("  Shape: N=%d  L=%d  D=%d", *r_committed.shape)

    log_quality_stats(records, kept_mask)

    # Clean up
    del model, r_committed_list, r_abstained_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─── CLI ─────────────────────────────────────────────────────────────────────

K = 8  # default, also read from tokenised file per record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract activation contrasts")
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Model key to process",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Extract activations  model=%s", args.model)
    log.info("Tokenised dir : %s", TOKENISED_DIR)
    log.info("Activations dir: %s", ACTIVATIONS_DIR)
    run(args.model)


if __name__ == "__main__":
    main()
