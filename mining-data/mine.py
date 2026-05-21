#!/usr/bin/env python3
"""
URC training data mining script.

Generates one deterministic completion per model per instance from four models,
judges each with gpt-oss-120b, and retains COMMITTED completions as URC training data.

Usage:
    python mine.py                        # full run, all four models
    python mine.py --model qwen-9b        # one model only
    python mine.py --n 10                 # smoke test (10 instances per dataset)
    python mine.py --model qwen-9b --n 5  # smoke test with one model
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

# ─── Path setup ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Silence noisy HTTP request logs from HF hub and Cerebras SDK
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub._commit_api").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Silence NumExpr thread-count notice
logging.getLogger("numexpr").setLevel(logging.WARNING)

# ─── Configuration ────────────────────────────────────────────────────────────

GENERATION_MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Base-2512",
]

from judge import (  # type: ignore[import]
    JUDGE_MODEL_ID, JUDGE_CEREBRAS_ID, JUDGE_TEMPLATE,
    build_judge_prompt as _build_judge_prompt_shared,
    _parse_judge_response, call_judge, make_cerebras_client,
)

# Recorded in output metadata; do_sample=False is the actual decoding mode
DECODING = {
    "do_sample": False,
    "temperature": 0,
    "max_new_tokens": 64,
    "num_samples_per_model": 1,
}
MAX_NEW_TOKENS = 64

KUQ_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence.\n\nQuestion:\n{question}\n\nAnswer:"
)
SQUAD_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence using only the context."
    "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

Y_ABS_KUQ = "I do not have enough information to answer."
Y_ABS_SQUAD = "The provided context does not contain enough information."

SAMPLED_DIR = BASE_DIR / "sampled"
RESULTS_DIR = BASE_DIR / "mining-results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── I/O helpers ──────────────────────────────────────────────────────────────


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def model_short_name(model_id: str) -> str:
    """Return the human-readable shortcut name (e.g. 'qwen-9b'), falling back to slug."""
    from llms.constants import SHORTCUTS  # type: ignore[import]
    _reverse = {v: k for k, v in SHORTCUTS.items()}
    return _reverse.get(model_id, model_slug(model_id))


def make_example_id(dataset: str, source_index: int, model_id: str) -> str:
    return f"{dataset}_{source_index}_{model_slug(model_id)}"



# ─── Data loading ─────────────────────────────────────────────────────────────


def load_instances() -> list[dict]:
    instances: list[dict] = []

    kuq_path = SAMPLED_DIR / "kuq_unanswerable_2000.jsonl"
    with open(kuq_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instances.append({
                "dataset": "kuq",
                "source_index": i,
                "source_id": row.get("id", i),
                "question": row["question"],
                "context": None,
            })

    squad_path = SAMPLED_DIR / "squad_unanswerable_2000.jsonl"
    with open(squad_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instances.append({
                "dataset": "squad",
                "source_index": i,
                "source_id": row.get("id", i),
                "question": row["question"],
                "context": row.get("context"),
            })

    n_kuq = sum(1 for x in instances if x["dataset"] == "kuq")
    n_squad = sum(1 for x in instances if x["dataset"] == "squad")
    log.info("Loaded %d KUQ + %d SQuAD instances (%d total)", n_kuq, n_squad, len(instances))
    return instances


def build_generation_prompt(inst: dict) -> str:
    if inst["dataset"] == "kuq":
        return KUQ_PROMPT_TEMPLATE.format(question=inst["question"])
    return SQUAD_PROMPT_TEMPLATE.format(
        context=inst["context"],
        question=inst["question"],
    )


# ─── Generation ───────────────────────────────────────────────────────────────


def is_base_model(model_id: str) -> bool:
    return "Base" in model_id


def set_deterministic_decoding(pipe: object) -> None:
    """Override a pipeline's gen_config for greedy decoding."""
    cfg = pipe.gen_config  # type: ignore[attr-defined]
    cfg.do_sample = False
    # Keep temperature at 1.0 to avoid transformers warning; do_sample=False
    # makes temperature irrelevant (greedy decoding is used).
    cfg.temperature = 1.0
    cfg.top_p = None
    cfg.top_k = None
    cfg.max_new_tokens = MAX_NEW_TOKENS


def generate_completion(pipe: object, prompt: str, base: bool) -> str:
    """Return only the newly generated text (no prompt echo)."""
    if base:
        result = pipe(prompt)  # type: ignore[operator]
        full: str = result[0]["generated_text"]
        return full[len(prompt):] if full.startswith(prompt) else full
    else:
        messages = [{"role": "user", "content": prompt}]
        result = pipe(messages)  # type: ignore[operator]
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)


def clean_completion(raw: str, prompt: str) -> str:
    text = raw.strip()
    # Remove prompt echo (base models sometimes repeat the prompt verbatim)
    prompt_stripped = prompt.strip()
    if text.startswith(prompt_stripped):
        text = text[len(prompt_stripped):].strip()
    return text


def _free_model_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_generation_phase(
    instances: list[dict],
    model_id: str,
) -> list[dict]:
    """Generate one completion per instance for model_id."""
    log.info("")
    log.info("=" * 64)
    log.info("GENERATE  %s", model_id)
    log.info("  instances  : %d", len(instances))
    log.info("  Loading model …")

    hf_token = os.environ.get("HF_TOKEN", "")
    t0 = time.time()

    if model_id.startswith("Qwen/"):
        from llms import qwen as qwen_module
        pipe = qwen_module.load(model_id, hf_token)
    elif model_id.startswith("mistralai/"):
        from llms import mistral as mistral_module
        pipe = mistral_module.load(model_id, hf_token)
    else:
        raise ValueError(f"Unsupported generator model: {model_id}")

    log.info("  Model loaded in %.1fs", time.time() - t0)
    set_deterministic_decoding(pipe)

    base = is_base_model(model_id)
    new_rows: list[dict] = []

    for i, inst in enumerate(instances):
        prompt = build_generation_prompt(inst)
        t_start = time.time()
        try:
            raw = generate_completion(pipe, prompt, base)
            clean = clean_completion(raw, prompt)
        except Exception as exc:
            log.warning("  Generation error [%s %d]: %s", inst["dataset"], inst["source_index"], exc)
            raw = ""
            clean = ""

        elapsed = time.time() - t_start
        row = {
            "example_id": make_example_id(inst["dataset"], inst["source_index"], model_id),
            "dataset": inst["dataset"],
            "source_index": inst["source_index"],
            "source_id": inst.get("source_id"),
            "question": inst["question"],
            "context": inst.get("context"),
            "generator_model": model_id,
            "sample_index": 0,
            "generation_prompt": prompt,
            "decoding": DECODING,
            "full_completion_raw": raw,
            "full_completion_clean": clean,
            "gen_time_s": round(elapsed, 3),
        }
        new_rows.append(row)

        preview = clean[:80].replace("\n", " ")
        log.info(
            "  [%d/%d] %.2fs | %s%s",
            i + 1, len(instances), elapsed,
            preview, "…" if len(clean) > 80 else "",
        )

    log.info("  Unloading model …")
    del pipe
    _free_model_memory()

    return new_rows


# ─── Judging ──────────────────────────────────────────────────────────────────


def build_judge_prompt(row: dict) -> str:
    return _build_judge_prompt_shared(
        question=row["question"],
        completion=row["full_completion_clean"],
        context=row.get("context"),
    )


def run_judging_phase(all_gen_rows: list[dict]) -> list[dict]:
    """Judge every generated completion."""
    log.info("")
    log.info("=" * 64)
    log.info("JUDGE  model=%s", JUDGE_MODEL_ID)
    log.info("  completions : %d", len(all_gen_rows))

    client = make_cerebras_client()

    committed = 0
    abstained = 0
    errors = 0
    judged_rows: list[dict] = []

    for i, gen_row in enumerate(all_gen_rows):
        clean = gen_row.get("full_completion_clean", "")

        if not clean:
            judged_rows.append({
                **gen_row,
                "judge_label": None,
                "judge_model": JUDGE_MODEL_ID,
                "judge_raw_output": None,
                "judge_time_s": 0.0,
            })
            continue

        prompt = build_judge_prompt(gen_row)
        t_start = time.time()
        label, raw_output = call_judge(client, prompt)
        elapsed = time.time() - t_start

        judged_rows.append({
            **gen_row,
            "judge_label": label,
            "judge_model": JUDGE_MODEL_ID,
            "judge_raw_output": raw_output,
            "judge_time_s": round(elapsed, 3),
        })

        if label == "COMMITTED":
            committed += 1
        elif label == "ABSTAINED":
            abstained += 1
        else:
            errors += 1

        log.info(
            "  [%d/%d] %.2fs  %-9s  C=%d A=%d E=%d",
            i + 1, len(all_gen_rows), elapsed, label,
            committed, abstained, errors,
        )

    return judged_rows


# ─── Prefix extraction ────────────────────────────────────────────────────────


def is_valid_prefix(text: str | None) -> bool:
    """A prefix is valid if it contains at least one word character."""
    if not text or not text.strip():
        return False
    return bool(re.search(r"\w", text.strip()))


def extract_prefixes_for_model(rows: list[dict], model_id: str) -> dict[str, dict]:
    """
    Load only the tokenizer for model_id, extract k=4,8,16 prefixes.
    Returns {example_id: {y_com_prefix_k4, k8, k16, prefix_valid_*}}.
    """
    log.info("  Tokenizer: %s", model_id)
    hf_token = os.environ.get("HF_TOKEN", "")

    if model_id.startswith("Qwen/"):
        from transformers import AutoTokenizer  # type: ignore[import]
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    elif model_id.startswith("mistralai/"):
        from transformers import MistralCommonBackend  # type: ignore[import]
        tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
    else:
        raise ValueError(f"Unsupported model for prefix extraction: {model_id}")

    results: dict[str, dict] = {}

    for row in rows:
        example_id = row["example_id"]
        completion = row["full_completion_clean"]
        try:
            token_ids = tokenizer.encode(completion, add_special_tokens=False)

            def decode_k(k: int) -> str | None:
                if len(token_ids) < k:
                    return None
                return tokenizer.decode(token_ids[:k], skip_special_tokens=True)

            k4 = decode_k(4)
            k8 = decode_k(8)
            k16 = decode_k(16)

            results[example_id] = {
                "y_com_prefix_k4": k4 or "",
                "y_com_prefix_k8": k8 or "",
                "y_com_prefix_k16": k16,
                "prefix_valid_k4": is_valid_prefix(k4),
                "prefix_valid_k8": is_valid_prefix(k8),
                "prefix_valid_k16": is_valid_prefix(k16),
            }
        except Exception as exc:
            log.warning("  Prefix error for %s: %s", example_id, exc)
            results[example_id] = {
                "y_com_prefix_k4": "",
                "y_com_prefix_k8": "",
                "y_com_prefix_k16": None,
                "prefix_valid_k4": False,
                "prefix_valid_k8": False,
                "prefix_valid_k16": False,
            }

    del tokenizer
    gc.collect()
    log.info("  Extracted prefixes for %d completions", len(results))
    return results


def is_prompt_echo_only(completion: str, prompt: str) -> bool:
    """True only if the completion is literally a substring of the prompt (verbatim echo)."""
    if not completion or not completion.strip():
        return False
    return completion.strip() in prompt


# ─── Output files ─────────────────────────────────────────────────────────────


def write_output_files(
    judged_rows: list[dict],
    prefix_map: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    Write one JSONL file per (model × dataset) pair.
    Each row contains both retained and discarded examples, distinguished by
    the `retained` boolean and `rejection_reason` (null when retained).
    Filename: mining-results/{model_slug}_{dataset}.jsonl
    """
    retained: list[dict] = []
    rejected: list[dict] = []
    seen: dict[tuple, set] = {}  # (dataset, source_index, model) -> {clean_completions}

    # Build all output rows first
    output_rows: list[dict] = []
    for row in judged_rows:
        example_id = row["example_id"]
        clean = row.get("full_completion_clean", "")
        label = row.get("judge_label")
        rejection_reason: str | None = None

        if not clean:
            rejection_reason = "empty_completion"
        elif label == "judge_error":
            rejection_reason = "invalid_json_from_judge"
        elif label in ("ABSTAINED", None) or label != "COMMITTED":
            rejection_reason = "abstained"
        elif is_prompt_echo_only(clean, row["generation_prompt"]):
            rejection_reason = "prompt_echo_only"

        if rejection_reason is None:
            prefixes = prefix_map.get(example_id, {})
            if not prefixes.get("prefix_valid_k8", False):
                rejection_reason = "invalid_prefix_k8"

        if rejection_reason is None:
            dedup_key = (row["dataset"], row["source_index"], row["generator_model"])
            existing = seen.setdefault(dedup_key, set())
            if clean in existing:
                rejection_reason = "duplicate_completion"
            else:
                existing.add(clean)

        is_retained = rejection_reason is None
        y_abs = Y_ABS_KUQ if row["dataset"] == "kuq" else Y_ABS_SQUAD
        prefixes = prefix_map.get(example_id, {}) if is_retained else {}

        out = {
            "example_id": example_id,
            "dataset": row["dataset"],
            "source_index": row["source_index"],
            "generator_model": row["generator_model"],
            "sample_index": row["sample_index"],
            "generation_prompt": row["generation_prompt"],
            "decoding": row["decoding"],
            "full_completion_raw": row.get("full_completion_raw", ""),
            "full_completion_clean": clean,
            "judge_label": row.get("judge_label"),
            "judge_model": JUDGE_MODEL_ID,
            "judge_raw_output": row.get("judge_raw_output"),
            "retained": is_retained,
            "rejection_reason": rejection_reason,
        }
        if is_retained:
            out.update({
                "y_com_prefix_k4": prefixes.get("y_com_prefix_k4", ""),
                "y_com_prefix_k8": prefixes.get("y_com_prefix_k8", ""),
                "y_com_prefix_k16": prefixes.get("y_com_prefix_k16"),
                "y_abs": y_abs,
                "prefix_valid_k4": prefixes.get("prefix_valid_k4", False),
                "prefix_valid_k8": prefixes.get("prefix_valid_k8", False),
                "prefix_valid_k16": prefixes.get("prefix_valid_k16", False),
            })
            retained.append(out)
        else:
            rejected.append(out)
        output_rows.append(out)

    # Group by (model_slug, dataset) and write one file per pair
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for out in output_rows:
        key = (model_short_name(out["generator_model"]), out["dataset"])
        groups[key].append(out)

    for (slug, dataset), rows in sorted(groups.items()):
        path = RESULTS_DIR / f"{slug}_{dataset}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_ret = sum(1 for r in rows if r["retained"])
        log.info("  Wrote %s  (%d retained, %d discarded)", path.name, n_ret, len(rows) - n_ret)

    return retained, rejected


def write_summary(
    all_gen_rows: list[dict],
    retained: list[dict],
    rejected: list[dict],
    judged_rows: list[dict],
) -> dict:
    summary_path = RESULTS_DIR / "urc_mining_summary.json"

    def _ds(rows: list[dict], dataset: str) -> list[dict]:
        return [r for r in rows if r.get("dataset") == dataset]

    def _model(rows: list[dict], model_id: str) -> list[dict]:
        return [r for r in rows if r.get("generator_model") == model_id]

    def _label(rows: list[dict], label: str) -> int:
        return sum(1 for r in rows if r.get("judge_label") == label)

    def dataset_counts(dataset: str) -> dict:
        j = _ds(judged_rows, dataset)
        return {
            "generated": len(j),
            "committed": _label(j, "COMMITTED"),
            "abstained": _label(j, "ABSTAINED"),
            "judge_errors": _label(j, "judge_error"),
            "retained": len(_ds(retained, dataset)),
            "rejected": len(_ds(rejected, dataset)),
        }

    def model_counts(model_id: str) -> dict:
        j = _model(judged_rows, model_id)
        return {
            "generated": len(j),
            "committed": _label(j, "COMMITTED"),
            "abstained": _label(j, "ABSTAINED"),
            "judge_errors": _label(j, "judge_error"),
            "retained": len(_model(retained, model_id)),
            "rejected": len(_model(rejected, model_id)),
        }

    def ds_model_counts(dataset: str, model_id: str) -> dict:
        j = _model(_ds(judged_rows, dataset), model_id)
        r = _model(_ds(retained, dataset), model_id)
        return {
            "committed": _label(j, "COMMITTED"),
            "abstained": _label(j, "ABSTAINED"),
            "retained": len(r),
        }

    rejection_counts: dict[str, int] = {}
    for r in rejected:
        reason = r.get("rejection_reason", "unknown")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    summary = {
        "num_kuq_instances": 2000,
        "num_squad_instances": 2000,
        "generation_models": GENERATION_MODELS,
        "judge_model": JUDGE_MODEL_ID,
        "decoding": DECODING,
        "total_generations": len(all_gen_rows),
        "total_committed": _label(judged_rows, "COMMITTED"),
        "total_abstained": _label(judged_rows, "ABSTAINED"),
        "total_rejected": len(rejected),
        "total_retained": len(retained),
        "counts_by_dataset": {
            "kuq": dataset_counts("kuq"),
            "squad": dataset_counts("squad"),
        },
        "counts_by_generator_model": {m: model_counts(m) for m in GENERATION_MODELS},
        "counts_by_dataset_and_model": {
            ds: {m: ds_model_counts(ds, m) for m in GENERATION_MODELS}
            for ds in ("kuq", "squad")
        },
        "judge_error_count": _label(judged_rows, "judge_error"),
        "valid_prefix_k4_count": sum(1 for r in retained if r.get("prefix_valid_k4")),
        "valid_prefix_k8_count": sum(1 for r in retained if r.get("prefix_valid_k8")),
        "valid_prefix_k16_count": sum(1 for r in retained if r.get("prefix_valid_k16")),
        "rejection_reasons": {
            "abstained": rejection_counts.get("abstained", 0),
            "empty_completion": rejection_counts.get("empty_completion", 0),
            "malformed_completion": rejection_counts.get("malformed_completion", 0),
            "invalid_json_from_judge": rejection_counts.get("invalid_json_from_judge", 0),
            "invalid_prefix_k8": rejection_counts.get("invalid_prefix_k8", 0),
            "prompt_echo_only": rejection_counts.get("prompt_echo_only", 0),
            "duplicate_completion": rejection_counts.get("duplicate_completion", 0),
        },
        "generated_at": datetime.now().isoformat(),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Wrote summary → %s", summary_path.name)
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URC mining pipeline")
    p.add_argument(
        "--model",
        default=None,
        help="Run generation for a single model only (shortcut or full ID)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N instances per dataset for smoke-testing",
    )
    p.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-apply retention rules and prefix extraction to existing result files "
             "(no generation or judging — fast)",
    )
    return p.parse_args()


def reprocess() -> None:
    """
    Read existing per-(model×dataset) result files, re-apply retention rules
    and prefix extraction, and overwrite the files in place.
    No models are loaded for generation; only tokenizers are loaded for prefixes.
    """
    log.info("REPROCESS MODE — reading existing result files")
    log.info("Results dir : %s", RESULTS_DIR)
    t0 = time.time()

    existing = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not existing:
        log.error("No result files found in %s", RESULTS_DIR)
        sys.exit(1)

    # Load all rows from all files; strip old retention fields so we recompute
    all_rows: list[dict] = []
    for path in existing:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        all_rows.extend(rows)
    log.info("Loaded %d rows from %d files", len(all_rows), len(existing))

    # Re-extract prefixes for committed rows, per model
    committed_rows = [r for r in all_rows if r.get("judge_label") == "COMMITTED"]
    log.info("")
    log.info("=" * 64)
    log.info("PREFIX EXTRACTION")
    log.info("  Committed completions: %d", len(committed_rows))

    prefix_map: dict[str, dict] = {}
    for model_id in GENERATION_MODELS:
        model_rows = [r for r in committed_rows if r.get("generator_model") == model_id]
        if model_rows:
            prefix_map.update(extract_prefixes_for_model(model_rows, model_id))

    log.info("")
    log.info("=" * 64)
    log.info("WRITING OUTPUT FILES")
    retained, rejected = write_output_files(all_rows, prefix_map)
    summary = write_summary(all_rows, retained, rejected, all_rows)

    log.info("")
    log.info("=" * 64)
    log.info("DONE in %.0fs", time.time() - t0)
    log.info("  Retained  : %d", summary["total_retained"])
    log.info("  Rejected  : %d", summary["total_rejected"])


def main() -> None:
    args = parse_args()

    if args.reprocess:
        reprocess()
        return

    from llms.constants import SHORTCUTS  # type: ignore[import]
    model_filter: str | None = SHORTCUTS.get(args.model, args.model) if args.model else None
    if model_filter and model_filter not in GENERATION_MODELS:
        log.error("Unknown model: %s", model_filter)
        sys.exit(1)

    log.info("URC Mining Pipeline")
    if args.n:
        log.info("Smoke-test mode  : first %d instances per dataset", args.n)
    log.info("Results dir      : %s", RESULTS_DIR)
    t_total = time.time()

    instances = load_instances()
    if args.n:
        kuq = [x for x in instances if x["dataset"] == "kuq"][: args.n]
        squad = [x for x in instances if x["dataset"] == "squad"][: args.n]
        instances = kuq + squad
        log.info("Truncated to %d instances (%d KUQ + %d SQuAD)", len(instances), len(kuq), len(squad))

    # ── Phase 1: Generation ──────────────────────────────────────────────────
    models_to_run = [model_filter] if model_filter else GENERATION_MODELS
    all_gen_rows: list[dict] = []
    for model_id in models_to_run:
        gen_rows = run_generation_phase(instances, model_id)
        all_gen_rows.extend(gen_rows)
    log.info("")
    log.info("Total generations: %d", len(all_gen_rows))

    # ── Phase 2: Judging ─────────────────────────────────────────────────────
    judged_rows = run_judging_phase(all_gen_rows)

    # ── Phase 3: Prefix extraction ───────────────────────────────────────────
    log.info("")
    log.info("=" * 64)
    log.info("PREFIX EXTRACTION")

    committed_rows = [r for r in judged_rows if r.get("judge_label") == "COMMITTED"]
    log.info("  Committed completions: %d", len(committed_rows))

    prefix_map: dict[str, dict] = {}
    for model_id in GENERATION_MODELS:
        model_rows = [r for r in committed_rows if r["generator_model"] == model_id]
        if model_rows:
            prefix_map.update(extract_prefixes_for_model(model_rows, model_id))

    # ── Phase 4: Write output files ──────────────────────────────────────────
    log.info("")
    log.info("=" * 64)
    log.info("WRITING OUTPUT FILES")
    retained, rejected = write_output_files(judged_rows, prefix_map)

    # ── Phase 5: Summary ─────────────────────────────────────────────────────
    summary = write_summary(all_gen_rows, retained, rejected, judged_rows)

    total_time = time.time() - t_total
    log.info("")
    log.info("=" * 64)
    log.info("DONE in %.2fh (%.0fs)", total_time / 3600, total_time)
    log.info("  Generations : %d", summary["total_generations"])
    log.info("  Committed   : %d", summary["total_committed"])
    log.info("  Abstained   : %d", summary["total_abstained"])
    log.info("  Judge errors: %d", summary["judge_error_count"])
    log.info("  Retained    : %d", summary["total_retained"])
    log.info("  Rejected    : %d", summary["total_rejected"])


if __name__ == "__main__":
    main()
