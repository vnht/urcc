#!/usr/bin/env python3
"""
Extract UltraChat retain activations.

For each model, runs teacher-forced forward passes on 1000 UltraChat
retain examples and saves the mean hidden state over response tokens
at the last-25% transformer layers.

Output per model:
    mining-data/activations/retain_activations_<model>_last25.pt

Contents:
    {
        "model":              str,          # HF model ID
        "model_key":          str,          # e.g. "qwen_instruct"
        "layers":             list[int],    # selected layer indices
        "source":             "ultrachat_retain",
        "num_examples":       int,
        "hidden_dim":         int,
        "retain_activations": tensor[N, L, D],
        "prompt_ids":         list[str],
    }

Usage:
    python3 mining-data/extract_retain_activations.py --model qwen_instruct
    python3 mining-data/extract_retain_activations.py          # all four models
"""

# ── Triton mock — must be first, before any other import ─────────────────────
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
import gc
import json
import logging
import os
import statistics
import time
import warnings
from pathlib import Path

import torch
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

BASE_DIR        = Path(__file__).parent
REPO_ROOT       = BASE_DIR.parent
SAMPLED_DIR     = BASE_DIR / "sampled"
ACTIVATIONS_DIR = BASE_DIR / "activations"
ACTIVATIONS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}

LAYER_SETS = {
    "qwen_instruct":      list(range(24, 32)),   # 8 layers
    "qwen_base":          list(range(24, 32)),
    "ministral_instruct": list(range(25, 34)),   # 9 layers
    "ministral_base":     list(range(25, 34)),
}

MAX_RESPONSE_TOKENS = 256
INPUT_FILE = SAMPLED_DIR / "ultrachat_retain_1000.jsonl"


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_key: str, model_id: str, hf_token: str):
    log.info("Loading %s …", model_id)
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )

    elif model_id.startswith("mistralai/"):
        from transformers import (  # type: ignore[import]
            MistralCommonBackend, Mistral3ForConditionalGeneration, FineGrainedFP8Config,
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
                model_id, device_map="auto", token=hf_token, tie_word_embeddings=False,
            )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    model.eval()
    log.info("  Loaded in %.1fs", time.time() - t0)
    return model, tokenizer


# ─── Tokenisation ─────────────────────────────────────────────────────────────

def _tokenise_example(
    tokenizer,
    prompt: str,
    response: str,
    model_key: str,
) -> tuple[list[int], list[int]]:
    """
    Returns (full_ids, response_ids).
    full_ids    — token IDs for the complete prompt+response sequence
    response_ids — token IDs corresponding to the response span only
    """
    is_base = "base" in model_key

    if is_base:
        full_text    = prompt + "\n\n" + response
        full_ids     = tokenizer.encode(full_text, add_special_tokens=True)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        # In rare cases the tokeniser may re-merge tokens across the boundary;
        # use suffix matching as ground truth.
        if len(full_ids) >= len(response_ids):
            return full_ids, response_ids
        return full_ids, full_ids  # fallback: use all tokens

    else:
        # Instruct: encode prompt with generation prompt, then response separately
        # (avoids chat-template errors on models that reject assistant-final turns)
        user_msg = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids   = tokenizer.encode(prompt_formatted, add_special_tokens=False)
        response_ids = tokenizer.encode(response,         add_special_tokens=False)
        full_ids     = prompt_ids + response_ids
        return full_ids, response_ids


# ─── Forward pass ─────────────────────────────────────────────────────────────

def forward_selected_layers(
    model,
    input_ids: torch.Tensor,   # (1, seq_len)
    layer_indices: list[int],
) -> list[torch.Tensor]:
    """
    Run a forward pass with output_hidden_states=True.
    Returns one tensor per selected layer, each shape (seq_len, D), float32.
    hidden_states[0] = embedding; hidden_states[i+1] = layer i output.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
        )

    # hidden_states tuple: index 0 = embedding, index l+1 = layer l output
    all_hidden = outputs.hidden_states
    return [all_hidden[l + 1][0].float().cpu() for l in layer_indices]


def mean_response_activation(
    layer_hiddens: list[torch.Tensor],   # each (seq_len, D)
    response_start: int,
    response_len: int,
    max_tokens: int,
) -> torch.Tensor:
    """
    Mean over response token positions [response_start : response_start + capped_len].
    Returns (num_layers, D).
    """
    cap   = min(response_len, max_tokens)
    end   = response_start + cap
    means = []
    for h in layer_hiddens:
        resp_h = h[response_start:end, :]   # (cap, D)
        means.append(resp_h.mean(dim=0))     # (D,)
    return torch.stack(means)  # (L, D)


# ─── Per-model extraction ─────────────────────────────────────────────────────

def run(model_key: str) -> None:
    model_id     = MODEL_REGISTRY[model_key]
    layer_indices = LAYER_SETS[model_key]
    hf_token     = os.environ.get("HF_TOKEN", "")
    out_path     = ACTIVATIONS_DIR / f"retain_activations_{model_key}_last25.pt"

    examples = [json.loads(l) for l in INPUT_FILE.read_text().splitlines() if l.strip()]
    log.info("")
    log.info("=" * 64)
    log.info("MODEL  %s", model_key)
    log.info("  Input examples : %d", len(examples))
    log.info("  Layers         : %s", layer_indices)

    model, tokenizer = load_model_and_tokenizer(model_key, model_id, hf_token)

    activations: list[torch.Tensor] = []  # each (L, D)
    prompt_ids_out: list[str] = []
    response_token_counts: list[int] = []
    dropped = 0

    bar = tqdm(examples, desc=model_key, unit="ex", dynamic_ncols=True)
    for ex in bar:
        prompt   = ex.get("prompt", "")
        response = ex.get("response", "")
        ex_id    = str(ex.get("id", ""))

        if not prompt or not response:
            dropped += 1
            continue

        try:
            full_ids, resp_ids = _tokenise_example(tokenizer, prompt, response, model_key)
        except Exception as exc:
            log.warning("  Tokenisation error [%s]: %s", ex_id, exc)
            dropped += 1
            continue

        if not resp_ids:
            dropped += 1
            continue

        n_resp = len(resp_ids)
        resp_start = len(full_ids) - n_resp

        input_tensor = torch.tensor([full_ids], dtype=torch.long)

        try:
            layer_hiddens = forward_selected_layers(model, input_tensor, layer_indices)
        except Exception as exc:
            log.warning("  Forward pass error [%s]: %s", ex_id, exc)
            dropped += 1
            continue

        act = mean_response_activation(layer_hiddens, resp_start, n_resp, MAX_RESPONSE_TOKENS)
        activations.append(act.cpu())
        prompt_ids_out.append(ex_id)
        response_token_counts.append(min(n_resp, MAX_RESPONSE_TOKENS))

        bar.set_postfix(kept=len(activations), drop=dropped)

    if not activations:
        log.error("  No examples retained — aborting.")
        return

    retain_tensor = torch.stack(activations).float()  # (N, L, D)
    N, L, D = retain_tensor.shape

    # ── Logging ───────────────────────────────────────────────────────────────
    rtc = sorted(response_token_counts)
    log.info("")
    log.info("  num_input_examples  : %d", len(examples))
    log.info("  num_kept            : %d", N)
    log.info("  num_dropped         : %d", dropped)
    log.info("  layers              : %s", layer_indices)
    log.info("  hidden_dim          : %d", D)
    log.info("  retain_activations  : %s", list(retain_tensor.shape))
    log.info("  response_tokens mean: %.1f", statistics.mean(rtc))
    log.info("  response_tokens med : %.1f", statistics.median(rtc))
    log.info("  response_tokens p90 : %.1f", rtc[int(len(rtc) * 0.9)])

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert not torch.isnan(retain_tensor).any(),  "NaNs in retain activations!"
    assert not torch.isinf(retain_tensor).any(),  "Infs in retain activations!"
    for li, layer_idx in enumerate(layer_indices):
        std = retain_tensor[:, li, :].std().item()
        assert std > 0, f"Zero std at layer {layer_idx}"
    log.info("  Sanity checks passed (no NaN/Inf, std > 0 per layer)")

    # ── Save ──────────────────────────────────────────────────────────────────
    bundle = {
        "model":              model_id,
        "model_key":          model_key,
        "layers":             layer_indices,
        "source":             "ultrachat_retain",
        "num_examples":       N,
        "hidden_dim":         D,
        "retain_activations": retain_tensor,
        "prompt_ids":         prompt_ids_out,
    }
    torch.save(bundle, out_path)
    log.info("  Saved → %s", out_path.name)

    del model, activations, retain_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract UltraChat retain activations")
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    models = args.model or list(MODEL_REGISTRY.keys())

    log.info("Extract retain activations")
    log.info("  Input : %s", INPUT_FILE)
    log.info("  Output: %s", ACTIVATIONS_DIR)
    log.info("  Models: %s", models)

    for model_key in models:
        run(model_key)

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
