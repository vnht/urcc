#!/usr/bin/env python3
"""
Compute teacher-forced perplexity on UltraChat for each baseline model.

PPL is measured over the assistant response tokens only, conditioned on the
full conversation context (system + user turns preceding it).

Reads:  training/held-out-eval-data/ultrachat_1000.jsonl
Writes: eval_baseline_ultrachat_ppl.jsonl

Usage:
    python perplexity.py                              # all four models
    python perplexity.py --model qwen-9b              # one model
    python perplexity.py --n 50                       # smoke test
    python perplexity.py --results-dir path/to/dir   # custom output directory
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
EVAL_DIR  = REPO_ROOT / "training" / "held-out-eval-data"

sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from llms.constants import SHORTCUTS  # type: ignore[import]

SHORTCUT_TO_ID = SHORTCUTS
ID_TO_SHORTCUT = {v: k for k, v in SHORTCUTS.items()}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)

MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Base-2512",
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_ultrachat(n: int | None = None) -> list[dict]:
    rows = []
    with open(EVAL_DIR / "ultrachat_1000.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n is not None:
        rows = rows[:n]
    log.info("Loaded %d UltraChat examples", len(rows))
    return rows


MAX_SEQ_LEN = 256


# ─── Tokenisation helpers ─────────────────────────────────────────────────────

def _get_response_token_ids(tokenizer, ex: dict, is_base: bool) -> tuple[list[int], list[int]]:
    """
    Returns (full_ids, response_ids) using only the first user/assistant exchange.
    Sequences are truncated to MAX_SEQ_LEN tokens from the left (keeping the response).
    """
    prompt_text   = ex.get("prompt", "")
    response_text = ex.get("response", "")

    if not prompt_text or not response_text:
        # fall back to first two messages
        messages = ex.get("messages", [])
        prompt_text   = next((m["content"] for m in messages if m["role"] == "user"),      "")
        response_text = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    first_exchange = [
        {"role": "user",      "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]

    if is_base:
        full_ids = tokenizer.encode(prompt_text + " " + response_text, add_special_tokens=True)
        resp_ids = tokenizer.encode(response_text, add_special_tokens=False)
    else:
        # Encode prompt with generation prompt, then response separately.
        # Avoids chat-template errors on models that reject assistant-final turns.
        prompt_formatted = tokenizer.apply_chat_template(
            first_exchange[:1], tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)
        resp_ids   = tokenizer.encode(response_text,    add_special_tokens=False)
        full_ids   = prompt_ids + resp_ids

    # Truncate from the left if over limit, preserving the response
    if len(full_ids) > MAX_SEQ_LEN:
        excess    = len(full_ids) - MAX_SEQ_LEN
        keep_resp = min(len(resp_ids), MAX_SEQ_LEN)
        resp_ids  = resp_ids[-keep_resp:]
        full_ids  = full_ids[-MAX_SEQ_LEN:]

    return full_ids, resp_ids


# ─── PPL computation ──────────────────────────────────────────────────────────

def compute_ppl(model_id: str, examples: list[dict]) -> dict:
    hf_token = os.environ.get("HF_TOKEN", "")
    is_base  = "Base" in model_id

    log.info("")
    log.info("=" * 64)
    log.info("PPL  %s", model_id)

    t0 = time.time()
    if model_id.startswith("Qwen/"):
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
    elif model_id.startswith("mistralai/"):
        import sys as _sys
        from unittest.mock import MagicMock
        if _sys.platform == "darwin":
            torch.compile = lambda fn=None, **kwargs: (fn if fn is not None else lambda f: f)
            for _mod in ("triton", "triton.language", "triton.runtime", "triton.runtime.jit"):
                _sys.modules.setdefault(_mod, MagicMock())
        from transformers import MistralCommonBackend, Mistral3ForConditionalGeneration, FineGrainedFP8Config  # type: ignore[import]
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
        if _sys.platform == "darwin":
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu", token=hf_token,
                quantization_config=FineGrainedFP8Config(dequantize=True),
            )
            model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", token=hf_token,
            )
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    log.info("  Model loaded in %.1fs", time.time() - t0)
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    skipped = 0
    instance_rows: list[dict] = []

    bar = tqdm(examples, desc=ID_TO_SHORTCUT.get(model_id, model_id), unit="ex", dynamic_ncols=True)
    for i, ex in enumerate(bar):
        try:
            full_ids, resp_ids = _get_response_token_ids(tokenizer, ex, is_base)
        except Exception as exc:
            log.warning("  Tokenisation error [%d]: %s", i, exc)
            skipped += 1
            continue

        if not resp_ids:
            skipped += 1
            continue

        n_prompt  = len(full_ids) - len(resp_ids)
        labels    = [-100] * n_prompt + list(resp_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        label_ids = torch.tensor([labels],  dtype=torch.long, device=device)

        with torch.no_grad():
            nll = model(input_ids=input_ids, labels=label_ids).loss.item()

        ppl_i = math.exp(nll)
        instance_rows.append({
            "id":         ex.get("id", i),
            "model":      ID_TO_SHORTCUT.get(model_id, model_id.replace("/", "__")),
            "prompt":     ex.get("prompt", ""),
            "response":   ex.get("response", ""),
            "num_tokens": len(resp_ids),
            "nll":        round(nll, 4),
            "ppl":        round(ppl_i, 3),
        })

        total_nll    += nll * len(resp_ids)
        total_tokens += len(resp_ids)

        running_ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
        bar.set_postfix(ppl=f"{running_ppl:.2f}", tokens=total_tokens, skip=skipped)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    ppl      = math.exp(mean_nll) if total_tokens else float("nan")

    log.info("  mean_nll=%.4f  ppl=%.3f  tokens=%d  skipped=%d", mean_nll, ppl, total_tokens, skipped)

    summary = {
        "model":        ID_TO_SHORTCUT.get(model_id, model_id.replace("/", "__")),
        "num_examples": len(examples) - skipped,
        "num_tokens":   total_tokens,
        "mean_nll":     round(mean_nll, 4),
        "ppl":          round(ppl, 3),
    }
    return summary, instance_rows


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced PPL on UltraChat eval set.")
    parser.add_argument("--model", help="Model shortcut or full HF ID (default: all four)")
    parser.add_argument("--n", type=int, help="Limit examples (smoke test)")
    parser.add_argument("--results-dir", type=Path, default=HERE,
                        help="Directory where output JSONL is written")
    args = parser.parse_args()

    models = [SHORTCUT_TO_ID.get(args.model, args.model)] if args.model else MODELS
    examples = load_ultrachat(n=args.n)

    summaries = []
    for model_id in models:
        summary, instance_rows = compute_ppl(model_id, examples)
        summaries.append(summary)

        inst_path = args.results_dir / f"eval_baseline_ultrachat_ppl_{summary['model']}.jsonl"
        with open(inst_path, "w") as f:
            for row in instance_rows:
                f.write(json.dumps(row) + "\n")
        log.info("Saved instances → %s", inst_path)

    out_path = args.results_dir / "eval_baseline_ultrachat_ppl.jsonl"
    with open(out_path, "w") as f:
        for row in summaries:
            f.write(json.dumps(row) + "\n")
    log.info("Saved summary → %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
