#!/usr/bin/env python3
"""
Build unsupported-commitment contrast files from selected mining results.

Step 1 — contrasts:
    Read mining-data/mining-selected/*.jsonl, add templated y_abstained, and write
    one contrast JSONL per (model × dataset) slice to mining-data/contrasts/.

Step 2 — tokenise:
    For each of the four models, load the tokenizer, tokenise
    generation_prompt + y_committed  and  generation_prompt + y_abstained,
    and write one tokenised JSONL per model to mining-data/tokenised/.
    Records with fewer than 4 usable answer tokens on either side are dropped.

Usage:
    python3 mining-data/build_contrasts.py                 # both steps
    python3 mining-data/build_contrasts.py --step contrasts
    python3 mining-data/build_contrasts.py --step tokenise
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

SELECTED_DIR = BASE_DIR / "mining-selected"
RESULTS_DIR = BASE_DIR / "mining-results"
CONTRASTS_DIR = BASE_DIR / "contrasts"
TOKENISED_DIR = BASE_DIR / "tokenised"
CONTRASTS_DIR.mkdir(exist_ok=True)
TOKENISED_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────

# selected filename stem prefix → (model_key, full HF model ID)
SELECTED_TO_MODEL = {
    "qwen-9b":           ("qwen_instruct",     "Qwen/Qwen3.5-9B"),
    "qwen-9b-base":      ("qwen_base",          "Qwen/Qwen3.5-9B-Base"),
    "ministral-8b":      ("ministral_instruct", "mistralai/Ministral-3-8B-Instruct-2512"),
    "ministral-8b-base": ("ministral_base",     "mistralai/Ministral-3-8B-Base-2512"),
}

DATASET_LABEL = {"kuq": "KUQ", "squad": "SQuAD2"}

Y_ABS = {
    "kuq":   "I do not have enough information to answer that.",
    "squad": "The context does not provide enough information to answer that.",
}

K = 8
MIN_ANSWER_TOKENS = 4


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_prompt(prompt: str, dataset: str) -> tuple[str, str | None]:
    """Extract (question, context) from the generation prompt string."""
    if dataset == "kuq":
        q = prompt.split("Question:\n", 1)[1].rsplit("\n\nAnswer:", 1)[0]
        return q.strip(), None
    # SQuAD
    after_ctx = prompt.split("Context:\n", 1)[1]
    context = after_ctx.split("\n\nQuestion:\n", 1)[0].strip()
    q = after_ctx.split("\n\nQuestion:\n", 1)[1].rsplit("\n\nAnswer:", 1)[0].strip()
    return q, context


def build_results_lookup() -> dict[tuple, dict]:
    """
    Count, per (dataset, source_index), how many of the 4 model runs were
    COMMITTED vs other. Used to fill num_samples / num_committed / num_other.
    """
    lookup: dict[tuple, dict] = defaultdict(lambda: {"total": 0, "committed": 0})
    for path in RESULTS_DIR.glob("*.jsonl"):
        for row in load_jsonl(path):
            key = (row["dataset"], row["source_index"])
            lookup[key]["total"] += 1
            if row.get("judge_label") == "COMMITTED":
                lookup[key]["committed"] += 1
    return lookup


# ─── Step 1: Contrast files ───────────────────────────────────────────────────

def build_contrasts(lookup: dict) -> None:
    log.info("=" * 64)
    log.info("STEP 1 — building contrast files")

    # Group selected files by model_key
    by_model: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(SELECTED_DIR.glob("*.jsonl")):
        stem = path.stem                  # e.g. "qwen-9b_kuq"
        model_prefix = stem.rsplit("_", 1)[0]   # e.g. "qwen-9b"
        if model_prefix in SELECTED_TO_MODEL:
            by_model[model_prefix].append(path)

    for model_prefix, paths in sorted(by_model.items()):
        model_key, model_id = SELECTED_TO_MODEL[model_prefix]

        for path in sorted(paths):
            dataset = path.stem.split("_")[-1]   # "kuq" or "squad"
            y_abs = Y_ABS[dataset]
            ds_label = DATASET_LABEL[dataset]

            rows = load_jsonl(path)
            contrast_name = f"unsupported_contrasts_{dataset}_{model_key}.jsonl"
            out_path = CONTRASTS_DIR / contrast_name

            records = []
            for row in rows:
                source_idx = row["source_index"]
                prompt = row["generation_prompt"]
                question, context = parse_prompt(prompt, dataset)
                stats = lookup.get((dataset, source_idx), {"total": 4, "committed": 1})

                record = {
                    "dataset": ds_label,
                    "model": model_id,
                    "prompt_id": f"{dataset}_{source_idx}",
                    "question": question,
                    "context": context,
                    "generation_prompt": prompt,
                    "x_unanswerable": {
                        "question": question,
                        "context": context,
                    },
                    "y_committed": row["full_completion_clean"],
                    "y_abstained": y_abs,
                    "abstention_type": "template",
                    "num_samples": stats["total"],
                    "num_committed": stats["committed"],
                    "num_other": stats["total"] - stats["committed"],
                }
                records.append(record)

            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            log.info("  Wrote %d rows → %s", len(records), contrast_name)


# ─── Step 2: Tokenised files ─────────────────────────────────────────────────

def tokenise_for_model(model_key: str, model_id: str) -> None:
    log.info("")
    log.info("Tokenising for %s (%s)", model_key, model_id)

    hf_token = os.environ.get("HF_TOKEN", "")

    # Load tokenizer only (no model weights)
    if model_id.startswith("Qwen/"):
        from transformers import AutoTokenizer  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    elif model_id.startswith("mistralai/"):
        from transformers import MistralCommonBackend  # type: ignore[import]
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    # Gather all contrast files for this model (KUQ + SQuAD)
    contrast_files = sorted(CONTRASTS_DIR.glob(f"unsupported_contrasts_*_{model_key}.jsonl"))
    if not contrast_files:
        log.warning("  No contrast files found for %s", model_key)
        return

    out_path = TOKENISED_DIR / f"tokenised_unsupported_contrasts_{model_key}.jsonl"
    kept = 0
    dropped = 0

    # Pre-tokenise the fixed abstention templates to get k-prefix IDs
    abs_prefix_cache: dict[str, list[int]] = {}

    with open(out_path, "w") as fout:
        for cpath in contrast_files:
            for row in load_jsonl(cpath):
                prompt = row["generation_prompt"]
                y_com = row["y_committed"]
                y_abs = row["y_abstained"]

                # Encode full sequences
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

                full_com_ids = tokenizer.encode(prompt + y_com, add_special_tokens=True)
                full_abs_ids = tokenizer.encode(prompt + y_abs, add_special_tokens=True)

                p_len = len(prompt_ids)

                # Answer token IDs
                com_answer_ids = full_com_ids[p_len:]
                abs_answer_ids = full_abs_ids[p_len:]

                # Drop if either side has fewer than MIN_ANSWER_TOKENS
                if len(com_answer_ids) < MIN_ANSWER_TOKENS or len(abs_answer_ids) < MIN_ANSWER_TOKENS:
                    dropped += 1
                    continue

                # Take first K tokens of each answer side
                com_prefix_ids = com_answer_ids[:K]
                abs_prefix_ids = abs_answer_ids[:K]

                com_prefix_text = tokenizer.decode(com_prefix_ids, skip_special_tokens=True)
                abs_prefix_text = tokenizer.decode(abs_prefix_ids, skip_special_tokens=True)

                out = {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "prompt_id": row["prompt_id"],
                    "prompt_token_ids": prompt_ids,
                    "committed_prefix_token_ids": com_prefix_ids,
                    "abstained_prefix_token_ids": abs_prefix_ids,
                    "committed_prefix_text": com_prefix_text,
                    "abstained_prefix_text": abs_prefix_text,
                    "k": K,
                }
                fout.write(json.dumps(out) + "\n")
                kept += 1

    log.info("  Kept %d / dropped %d → %s", kept, dropped, out_path.name)
    del tokenizer


def build_tokenised() -> None:
    log.info("=" * 64)
    log.info("STEP 2 — tokenising")

    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(REPO_ROOT / ".env")

    import gc
    for model_prefix, (model_key, model_id) in sorted(SELECTED_TO_MODEL.items()):
        tokenise_for_model(model_key, model_id)
        gc.collect()


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build contrast and tokenised files")
    p.add_argument(
        "--step",
        choices=["contrasts", "tokenise", "all"],
        default="all",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    lookup = build_results_lookup()
    log.info("Results lookup built: %d (dataset, source_index) pairs", len(lookup))

    if args.step in ("contrasts", "all"):
        build_contrasts(lookup)

    if args.step in ("tokenise", "all"):
        build_tokenised()

    log.info("")
    log.info("Done.")
    log.info("  Contrasts  : %s", CONTRASTS_DIR)
    log.info("  Tokenised  : %s", TOKENISED_DIR)


if __name__ == "__main__":
    main()
