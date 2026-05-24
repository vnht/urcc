#!/usr/bin/env python3
"""
Baseline generation for KUQ and SQuAD eval sets.

Runs each model on training/held-out-eval-data/kuq_1000.jsonl and
training/held-out-eval-data/squad_1000.jsonl with greedy decoding and saves
one output file per model.

Usage:
    python generate.py                         # all four models
    python generate.py --model qwen-9b         # one model only
    python generate.py --model qwen-9b --n 10  # smoke test
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
EVAL_DIR  = REPO_ROOT / "training" / "held-out-eval-data"
OUT_DIR   = HERE

sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)

# ─── Config ───────────────────────────────────────────────────────────────────

MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Base-2512",
]

KUQ_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence.\n\nQuestion:\n{question}\n\nAnswer:"
)
SQUAD_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence using only the context."
    "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

MAX_NEW_TOKENS = 64

from llms.constants import SHORTCUTS  # type: ignore[import]
SHORTCUT_TO_ID = SHORTCUTS
ID_TO_SHORTCUT = {v: k for k, v in SHORTCUTS.items()}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def model_slug(model_id: str) -> str:
    return ID_TO_SHORTCUT.get(model_id, model_id.replace("/", "__"))


def is_base_model(model_id: str) -> bool:
    return "Base" in model_id


def build_prompt(row: dict) -> str:
    if row["dataset"] == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=row["question"])
    return SQUAD_PROMPT_TEMPLATE.format(
        context=row["context"],
        question=row["question"],
    )


def set_greedy_decoding(pipe) -> None:
    cfg = pipe.gen_config
    cfg.do_sample = False
    cfg.temperature = 1.0
    cfg.top_p = None
    cfg.top_k = None
    cfg.max_new_tokens = MAX_NEW_TOKENS


def generate_completion(pipe, prompt: str, base: bool) -> str:
    if base:
        result = pipe(prompt)
        full: str = result[0]["generated_text"]
        return full[len(prompt):] if full.startswith(prompt) else full
    else:
        messages = [{"role": "user", "content": prompt}]
        result = pipe(messages)
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)


def clean_completion(raw: str, prompt: str) -> str:
    text = raw.strip()
    prompt_stripped = prompt.strip()
    if text.startswith(prompt_stripped):
        text = text[len(prompt_stripped):].strip()
    return text


def free_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_eval_instances(n: int | None = None) -> list[dict]:
    instances = []
    for dataset in ("kuq", "squad"):
        path = EVAL_DIR / f"{dataset}_1000.jsonl"
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                instances.append({
                    "dataset": dataset,
                    "id": row["id"],
                    "answerable": row["answerable"],
                    "question": row["question"],
                    "context": row.get("context"),
                })
    if n is not None:
        instances = instances[:n]
    log.info("Loaded %d eval instances", len(instances))
    return instances


# ─── Generation ───────────────────────────────────────────────────────────────

def run_model(model_id: str, instances: list[dict]) -> None:
    slug = model_slug(model_id)
    out_path = OUT_DIR / f"eval_baseline_generations_{slug}.jsonl"

    # Resume: skip already-generated instance IDs
    done_ids: set = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
        log.info("Resuming %s — %d already done", slug, len(done_ids))

    remaining = [inst for inst in instances if inst["id"] not in done_ids]
    if not remaining:
        log.info("All instances complete for %s", slug)
        return

    log.info("")
    log.info("=" * 64)
    log.info("GENERATE  %s", model_id)
    log.info("  remaining: %d / %d", len(remaining), len(instances))

    hf_token = os.environ.get("HF_TOKEN", "")
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from llms import qwen as qwen_module  # type: ignore[import]
        pipe = qwen_module.load(model_id, hf_token)
    elif model_id.startswith("mistralai/"):
        from llms import mistral as mistral_module  # type: ignore[import]
        pipe = mistral_module.load(model_id, hf_token)
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    log.info("  Model loaded in %.1fs", time.time() - t0)
    set_greedy_decoding(pipe)
    base = is_base_model(model_id)

    with open(out_path, "a") as fout:
        bar = tqdm(remaining, desc=slug, unit="inst", dynamic_ncols=True)
        for inst in bar:
            prompt = build_prompt(inst)
            t_start = time.time()
            try:
                raw = generate_completion(pipe, prompt, base)
                clean = clean_completion(raw, prompt)
            except Exception as exc:
                log.warning("  Generation error [%s %s]: %s", inst["dataset"], inst["id"], exc)
                raw = ""
                clean = ""

            elapsed = time.time() - t_start
            row = {
                "id": inst["id"],
                "dataset": inst["dataset"],
                "answerable": inst["answerable"],
                "question": inst["question"],
                "context": inst.get("context"),
                "model": model_id,
                "generation_prompt": prompt,
                "completion_raw": raw,
                "completion": clean,
                "gen_time_s": round(elapsed, 3),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            preview = clean[:60].replace("\n", " ")
            bar.set_postfix_str(f"{elapsed:.2f}s | {preview}{'…' if len(clean) > 60 else ''}")

    log.info("  Saved → %s", out_path)
    del pipe
    free_memory()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline generation for KUQ/SQuAD eval sets.")
    parser.add_argument("--model", help="Model shortcut or full HF ID (default: all four)")
    parser.add_argument("--n", type=int, help="Limit instances per run (smoke test)")
    args = parser.parse_args()

    if args.model:
        model_id = SHORTCUT_TO_ID.get(args.model, args.model)
        models = [model_id]
    else:
        models = MODELS

    instances = load_eval_instances(n=args.n)

    for model_id in models:
        run_model(model_id, instances)

    log.info("Done.")


if __name__ == "__main__":
    main()
