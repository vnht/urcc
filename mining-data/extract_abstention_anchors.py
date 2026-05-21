#!/usr/bin/env python3
"""
Extract abstention anchors mu_abstain per layer.

For each model we want a single fixed reference point per last-25 % layer:

    mu_l = mean over (i, t) of  h_l(prompt_i + abstention_completion_i)[t]

where t ranges over the first K answer-position tokens (matching the K=8
used by the forget loss in train_urc.py). The mean is taken across all
(prompt, completion) pairs from the mining results where the base model
successfully abstained on an unanswerable question.

This vector is used as the new origin in train_urc.py's forget loss:

    L_forget = || V^T (h_forget - mu_l) ||^2

so that the optimiser pushes forget activations toward the abstention
manifold instead of toward zero.

Inputs (per model):
    mining-data/mining-results/<model_tag>_kuq.jsonl
    mining-data/mining-results/<model_tag>_squad.jsonl
        keep rows with judge_label == "ABSTAINED"
        use record["generation_prompt"] + record["full_completion_clean"]

Outputs (per model, last-25 % layer slice):
    mining-data/activations/abstention_anchors_<model_key>_last25.pt
        {
          "model":       str,
          "model_key":   str,
          "layers":      list[int],
          "k":           int,
          "n_examples":  int,
          "mu_abstain":  tensor[L, D]   # the anchor
          "datasets":    list[str],     # which mining files contributed
        }

Usage:
    python3 mining-data/extract_abstention_anchors.py --model qwen_instruct
    python3 mining-data/extract_abstention_anchors.py
    python3 mining-data/extract_abstention_anchors.py --model qwen_instruct --max-per-dataset 500
"""

# ── Triton mock (must precede any other import on macOS) ─────────────────────
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

MINING_RESULTS_DIR = BASE_DIR / "mining-results"
ACTIVATIONS_DIR    = BASE_DIR / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)

# ── Model registry (matches extract_supported_activations.py) ─────────────────

MODEL_REGISTRY = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}

MODEL_FILE_TAG = {
    "qwen_instruct":      "qwen-9b",
    "qwen_base":          "qwen-9b-base",
    "ministral_instruct": "ministral-8b",
    "ministral_base":     "ministral-8b-base",
}

# Last-25 % layer slice — must match the slice used by retain activations
# and the subspace bundles, otherwise mu_abstain won't align with V at training.
SELECTED_LAYERS = {
    "qwen_instruct":      [24, 25, 26, 27, 28, 29, 30, 31],
    "qwen_base":          [24, 25, 26, 27, 28, 29, 30, 31],
    "ministral_instruct": [25, 26, 27, 28, 29, 30, 31, 32, 33],
    "ministral_base":     [25, 26, 27, 28, 29, 30, 31, 32, 33],
}

K_ANSWER_TOKENS = 8   # matches train_urc.py K_ANSWER_TOKENS


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gather_abstention_examples(
    model_key: str,
    max_per_dataset: int | None,
) -> tuple[list[dict], list[str]]:
    """
    Read mining-results JSONLs for this model. Keep rows where the base model
    successfully abstained on an unanswerable question. Return list of
    {prompt, completion, dataset, example_id} plus the list of source datasets.
    """
    tag = MODEL_FILE_TAG[model_key]
    paths = [
        MINING_RESULTS_DIR / f"{tag}_kuq.jsonl",
        MINING_RESULTS_DIR / f"{tag}_squad.jsonl",
    ]
    out: list[dict] = []
    sources: list[str] = []

    for p in paths:
        if not p.exists():
            log.warning("  Missing mining-results file: %s — skipping", p.name)
            continue
        rows = load_jsonl(p)
        kept: list[dict] = []
        for r in rows:
            if r.get("judge_label") != "ABSTAINED":
                continue
            prompt = r.get("generation_prompt") or ""
            completion = r.get("full_completion_clean") or r.get("full_completion_raw") or ""
            if not prompt.strip() or not completion.strip():
                continue
            kept.append({
                "prompt":     prompt,
                "completion": completion,
                "dataset":    r.get("dataset") or "?",
                "example_id": r.get("example_id"),
            })

        if max_per_dataset is not None and len(kept) > max_per_dataset:
            kept = kept[:max_per_dataset]

        log.info("  %-30s %d abstention rows -> using %d",
                 p.name, sum(1 for r in rows if r.get("judge_label") == "ABSTAINED"), len(kept))
        out.extend(kept)
        sources.append(p.name)

    return out, sources


# ── Model loading (mirrors extract_supported_activations.py) ─────────────────

def load_model_and_tokenizer(model_key: str, model_id: str, hf_token: str):
    log.info("Loading model %s ...", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
    elif model_id.startswith("mistralai/"):
        from transformers import (
            Mistral3ForConditionalGeneration, FineGrainedFP8Config,
            MistralCommonBackend,
        )  # type: ignore[import]
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
        if sys.platform == "darwin":
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu", token=hf_token,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                tie_word_embeddings=False,
            )
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(device)
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", token=hf_token, tie_word_embeddings=False,
            )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    model.eval()
    log.info("  Loaded in %.1fs", time.time() - t0)
    return model, tokenizer


# ── Tokenisation (must match train_urc.py tokenise_forget) ───────────────────

def _encode(tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    """Match train_urc.py _encode behaviour."""
    if hasattr(tokenizer, "encode") and not isinstance(tokenizer, type):
        try:
            ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(ids, list):
                return ids
            if hasattr(ids, "tolist"):
                return ids.tolist()
            return list(ids)
        except TypeError:
            ids = tokenizer.encode(text)
            return list(ids) if not isinstance(ids, list) else ids
    raise RuntimeError(f"Tokenizer {type(tokenizer)} has no usable encode method")


def tokenise_forget(tokenizer, prompt: str, completion: str) -> tuple[list[int], int, int]:
    """Returns (full_ids, answer_start, num_answer_tokens). Matches train_urc.tokenise_forget."""
    prompt_ids = _encode(tokenizer, prompt)
    answer_ids = _encode(tokenizer, completion, add_special_tokens=False)
    answer_ids = answer_ids[:K_ANSWER_TOKENS]
    full_ids   = prompt_ids + answer_ids
    return full_ids, len(prompt_ids), len(answer_ids)


# ── Activation extraction ────────────────────────────────────────────────────

def forward_hidden_states(
    model, input_ids: torch.Tensor, layer_indices: list[int],
) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    all_hidden = outputs.hidden_states  # tuple, [0]=embed, [i+1]=block i output
    return [all_hidden[i + 1][0].float() for i in layer_indices]


# ── Main extraction loop ─────────────────────────────────────────────────────

def run(model_key: str, max_per_dataset: int | None) -> None:
    model_id      = MODEL_REGISTRY[model_key]
    layer_indices = SELECTED_LAYERS[model_key]
    hf_token      = os.environ.get("HF_TOKEN", "")

    log.info("")
    log.info("=" * 64)
    log.info("MODEL %s  layers=%s  K=%d", model_key, layer_indices, K_ANSWER_TOKENS)

    examples, sources = gather_abstention_examples(model_key, max_per_dataset)
    if not examples:
        log.error("  No abstention examples found for %s — skipping", model_key)
        return
    log.info("  Total abstention examples: %d  (sources: %s)", len(examples), sources)

    model, tokenizer = load_model_and_tokenizer(model_key, model_id, hf_token)

    L = len(layer_indices)
    D: int | None = None
    sum_per_layer: torch.Tensor | None = None
    n_used = 0
    skipped = 0

    t0 = time.time()
    for i, ex in enumerate(examples):
        try:
            full_ids, ans_start, n_ans = tokenise_forget(
                tokenizer, ex["prompt"], ex["completion"]
            )
        except Exception as exc:
            log.debug("  [%d] tokenisation error: %s", i, exc)
            skipped += 1
            continue
        if n_ans == 0:
            skipped += 1
            continue

        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        try:
            hiddens = forward_hidden_states(model, input_tensor, layer_indices)
        except Exception as exc:
            log.warning("  [%d] forward error: %s", i, exc)
            skipped += 1
            continue

        # Per-example: mean over the K answer-token positions, per layer.
        # h: (seq_len, D); slice [ans_start : ans_start + n_ans] then mean.
        per_layer = []
        for h in hiddens:
            ans_h = h[ans_start: ans_start + n_ans, :]   # (n_ans, D)
            per_layer.append(ans_h.mean(dim=0).cpu())
        ex_mean = torch.stack(per_layer)                  # (L, D)

        if D is None:
            D = ex_mean.shape[-1]
            sum_per_layer = torch.zeros(L, D, dtype=torch.float32)
        sum_per_layer += ex_mean.float()
        n_used += 1

        if (i + 1) % 50 == 0 or i == 0:
            log.info("  [%d/%d] used=%d skipped=%d  %.0fs",
                     i + 1, len(examples), n_used, skipped, time.time() - t0)

    if n_used == 0 or sum_per_layer is None:
        log.error("  No valid examples produced activations — aborting %s", model_key)
        return

    mu_abstain = sum_per_layer / n_used   # (L, D)

    bundle = {
        "model":       model_id,
        "model_key":   model_key,
        "layers":      layer_indices,
        "k":           K_ANSWER_TOKENS,
        "n_examples":  n_used,
        "mu_abstain":  mu_abstain.float(),
        "datasets":    sources,
    }

    out_path = ACTIVATIONS_DIR / f"abstention_anchors_{model_key}_last25.pt"
    torch.save(bundle, out_path)
    log.info("")
    log.info("  Saved -> %s", out_path.name)
    log.info("  shape=%s  n=%d  layers=%s",
             tuple(mu_abstain.shape), n_used, layer_indices)
    log.info("  Layer-wise mu norms:")
    for li, l in enumerate(layer_indices):
        log.info("    L%-3d  ||mu||=%.4f", l, float(mu_abstain[li].norm()))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract abstention anchors mu_abstain per layer."
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                   nargs="*", default=None,
                   help="One or more model keys (default: all)")
    p.add_argument("--max-per-dataset", type=int, default=None,
                   help="Cap rows used per mining-results file (default: no cap)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = args.model or list(MODEL_REGISTRY.keys())
    log.info("Extract abstention anchors  models=%s", models)
    log.info("Mining-results dir : %s", MINING_RESULTS_DIR)
    log.info("Activations dir    : %s", ACTIVATIONS_DIR)
    if args.max_per_dataset:
        log.info("Per-dataset cap    : %d", args.max_per_dataset)
    for model_key in models:
        run(model_key, max_per_dataset=args.max_per_dataset)


if __name__ == "__main__":
    main()
