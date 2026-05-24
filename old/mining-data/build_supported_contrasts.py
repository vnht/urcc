#!/usr/bin/env python3
"""
Build supported-answer contrast files from answerable KUQ and SQuAD instances.

Step 1 — instances:
    Sample 500 answerable instances from each dataset, attach y_correct (gold
    answer) and y_abstained (fixed template), and write:
        mining-data/supported/supported_answer_instances_kuq.jsonl
        mining-data/supported/supported_answer_instances_squad.jsonl

Step 2 — tokenise:
    For each of the four models, load the tokenizer, tokenise
        prompt + " " + y_correct
        prompt + " " + y_abstained
    and write one tokenised JSONL per model to mining-data/tokenised/:
        tokenised_supported_answer_contrasts_{model_key}.jsonl
    Records where y_correct is empty or tokenisation fails are dropped.
    Short gold answers (fewer than 8 tokens) are kept with all available tokens.

Usage:
    python3 mining-data/build_supported_contrasts.py          # both steps
    python3 mining-data/build_supported_contrasts.py --step instances
    python3 mining-data/build_supported_contrasts.py --step tokenise
"""

import argparse
import json
import logging
import os
import sys
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

SAMPLED_DIR   = BASE_DIR / "sampled"
CONTRASTS_DIR = BASE_DIR / "contrasts"
TOKENISED_DIR = BASE_DIR / "tokenised"
CONTRASTS_DIR.mkdir(exist_ok=True)
TOKENISED_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────

K = 8  # answer-prefix cap

KUQ_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence.\n\nQuestion:\n{question}\n\nAnswer:"
)
SQUAD_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence using only the context."
    "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

Y_ABS = {
    "kuq":   "I do not have enough information to answer that.",
    "squad": "The context does not provide enough information to answer that.",
}

MODEL_REGISTRY = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ─── Step 1: Build instance files ─────────────────────────────────────────────

def build_instances() -> None:
    log.info("=" * 64)
    log.info("STEP 1 — building supported-answer instance files")

    # ── KUQ ──────────────────────────────────────────────────────────────────
    kuq_sampled_path = SAMPLED_DIR / "kuq_answerable_500.jsonl"
    if not kuq_sampled_path.exists():
        log.error("Missing %s — run sampling first", kuq_sampled_path)
        sys.exit(1)
    kuq_sample = load_jsonl(kuq_sampled_path)
    log.info("KUQ answerable: %d instances from %s", len(kuq_sample), kuq_sampled_path.name)

    kuq_records = []
    for row in kuq_sample:
        q = row["question"].strip()
        prompt = KUQ_PROMPT_TEMPLATE.format(question=q)
        kuq_records.append({
            "dataset": "KUQ",
            "prompt_id": f"kuq_{row['id']}",
            "question": q,
            "context": None,
            "generation_prompt": prompt,
            "x_answerable": {"question": q, "context": None},
            "y_correct": row["correct_answer"].strip(),
            "y_abstained": Y_ABS["kuq"],
            "abstention_type": "template",
        })

    kuq_out = CONTRASTS_DIR / "supported_answer_instances_kuq.jsonl"
    write_jsonl(kuq_out, kuq_records)
    log.info("  Wrote %d rows → %s", len(kuq_records), kuq_out.name)

    # ── SQuAD ─────────────────────────────────────────────────────────────────
    squad_sampled_path = SAMPLED_DIR / "squad_answerable_500.jsonl"
    if not squad_sampled_path.exists():
        log.error("Missing %s — run sampling first", squad_sampled_path)
        sys.exit(1)
    squad_sample = load_jsonl(squad_sampled_path)
    log.info("SQuAD answerable: %d instances from %s", len(squad_sample), squad_sampled_path.name)

    squad_records = []
    for row in squad_sample:
        q = row["question"].strip()
        ctx = row["context"].strip()
        prompt = SQUAD_PROMPT_TEMPLATE.format(context=ctx, question=q)
        squad_records.append({
            "dataset": "SQuAD2",
            "prompt_id": f"squad_{row['id']}",
            "question": q,
            "context": ctx,
            "generation_prompt": prompt,
            "x_answerable": {"question": q, "context": ctx},
            "y_correct": row["correct_answer"].strip(),
            "y_abstained": Y_ABS["squad"],
            "abstention_type": "template",
        })

    squad_out = CONTRASTS_DIR / "supported_answer_instances_squad.jsonl"
    write_jsonl(squad_out, squad_records)
    log.info("  Wrote %d rows → %s", len(squad_records), squad_out.name)


# ─── Step 2: Tokenise per model ───────────────────────────────────────────────

def tokenise_for_model(model_key: str, model_id: str, records: list[dict]) -> None:
    log.info("")
    log.info("Tokenising for %s (%s)", model_key, model_id)

    hf_token = os.environ.get("HF_TOKEN", "")

    if model_id.startswith("Qwen/"):
        from transformers import AutoTokenizer  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    elif model_id.startswith("mistralai/"):
        from transformers import MistralCommonBackend  # type: ignore[import]
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    out_path = TOKENISED_DIR / f"tokenised_supported_answer_contrasts_{model_key}.jsonl"
    kept = dropped_empty = dropped_tok_err = 0

    with open(out_path, "w") as fout:
        for row in records:
            prompt = row["generation_prompt"]
            y_cor  = row["y_correct"]
            y_abs  = row["y_abstained"]

            if not y_cor:
                dropped_empty += 1
                continue

            try:
                prompt_ids    = tokenizer.encode(prompt, add_special_tokens=True)
                full_cor_ids  = tokenizer.encode(prompt + " " + y_cor, add_special_tokens=True)
                full_abs_ids  = tokenizer.encode(prompt + " " + y_abs, add_special_tokens=True)
            except Exception as exc:
                log.warning("  Tokenisation error for %s: %s — skipping", row["prompt_id"], exc)
                dropped_tok_err += 1
                continue

            p_len = len(prompt_ids)
            cor_answer_ids = full_cor_ids[p_len:]
            abs_answer_ids = full_abs_ids[p_len:]

            # Cap at K; keep all if shorter (do not drop short gold answers)
            cor_prefix_ids = cor_answer_ids[:K]
            abs_prefix_ids = abs_answer_ids[:K]

            if not cor_prefix_ids or not abs_prefix_ids:
                dropped_empty += 1
                continue

            cor_prefix_text = tokenizer.decode(cor_prefix_ids, skip_special_tokens=True)
            abs_prefix_text = tokenizer.decode(abs_prefix_ids, skip_special_tokens=True)

            out = {
                "dataset":                  row["dataset"],
                "model":                    model_id,
                "prompt_id":                row["prompt_id"],
                "prompt_token_ids":         prompt_ids,
                "correct_prefix_token_ids": cor_prefix_ids,
                "abstained_prefix_token_ids": abs_prefix_ids,
                "correct_prefix_text":      cor_prefix_text,
                "abstained_prefix_text":    abs_prefix_text,
                "k":                        len(cor_prefix_ids),
            }
            fout.write(json.dumps(out) + "\n")
            kept += 1

    log.info(
        "  Kept %d  |  dropped empty=%d  tok_err=%d  →  %s",
        kept, dropped_empty, dropped_tok_err, out_path.name,
    )
    del tokenizer


def build_tokenised() -> None:
    log.info("=" * 64)
    log.info("STEP 2 — tokenising per model")

    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(REPO_ROOT / ".env")

    # Combine KUQ + SQuAD instance files
    records: list[dict] = []
    for fname in ("supported_answer_instances_kuq.jsonl",
                  "supported_answer_instances_squad.jsonl"):
        path = SUPPORTED_DIR / fname
        if not path.exists():
            log.error("Missing instance file: %s — run --step instances first", fname)
            sys.exit(1)
        records.extend(load_jsonl(path))
    log.info("Total instances: %d", len(records))

    import gc
    for model_key, model_id in sorted(MODEL_REGISTRY.items()):
        tokenise_for_model(model_key, model_id, records)
        gc.collect()


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build supported-answer contrast files")
    p.add_argument(
        "--step",
        choices=["instances", "tokenise", "all"],
        default="all",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.step in ("instances", "all"):
        build_instances()

    if args.step in ("tokenise", "all"):
        build_tokenised()

    log.info("")
    log.info("Done.")
    log.info("  Instances  : %s", CONTRASTS_DIR)
    log.info("  Tokenised  : %s", TOKENISED_DIR)


if __name__ == "__main__":
    main()
