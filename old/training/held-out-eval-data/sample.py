"""
Sample eval data for training/held-out-eval-data.

Outputs:
  kuq_1000.jsonl                        — 354 answerable + 646 unanswerable
  squad_1000.jsonl                      — 500 answerable + 500 unanswerable
  ultrachat_1000.jsonl                  — 1000 held-out UltraChat pairs
  mining-data/sampled/ultrachat_retain_1000.jsonl — 1000 retain pairs (disjoint from held-out)

All instances are disjoint from mining-data/sampled/ answerable pool and mining-data/mining-selected/.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).parent.parent.parent
MINING_DATA = REPO_ROOT / "mining-data"
SAMPLED_DIR = MINING_DATA / "sampled"
MINING_SELECTED_DIR = MINING_DATA / "mining-selected"
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
OUT_DIR = Path(__file__).parent

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_excluded_source_indices(dataset: str) -> set[int]:
    """Source indices (line positions in unanswerable file) used in mining-selected."""
    excluded: set[int] = set()
    for path in MINING_SELECTED_DIR.glob(f"*_{dataset}.jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded.add(json.loads(line)["source_index"])
    return excluded


def load_training_answerable_ids(dataset: str) -> set:
    """IDs already used in the training answerable pool (sampled/)."""
    ids: set = set()
    with open(SAMPLED_DIR / f"{dataset}_answerable_500.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


# ---------------------------------------------------------------------------
# KUQ — 354 answerable (all remaining) + 646 unanswerable
# ---------------------------------------------------------------------------

def sample_kuq() -> None:
    training_ids = load_training_answerable_ids("kuq")
    excluded_indices = load_excluded_source_indices("kuq")

    # Answerable: all remaining triviaqa instances not in training pool
    answerable: list[dict] = []
    with open(PROCESSED_DIR / "kuq.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("answerable") and row["id"] not in training_ids:
                if row.get("source") in {"turk", "triviaqa"}:
                    answerable.append(row)

    n_answerable = len(answerable)
    n_unanswerable = 1000 - n_answerable
    print(f"KUQ: {n_answerable} answerable remaining → topping up with {n_unanswerable} unanswerable")

    # Unanswerable: sample from pool excluding mining-selected
    unans_pool: list[dict] = []
    with open(SAMPLED_DIR / "kuq_unanswerable_2000.jsonl") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if i not in excluded_indices and row.get("source") in {"turk", "triviaqa"}:
                unans_pool.append(row)

    assert len(unans_pool) >= n_unanswerable, f"Not enough unanswerable: {len(unans_pool)}"
    unanswerable = random.sample(unans_pool, n_unanswerable)

    all_rows = answerable + unanswerable
    random.shuffle(all_rows)

    out_path = OUT_DIR / "kuq_1000.jsonl"
    with open(out_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out_path}  ({n_answerable} answerable, {n_unanswerable} unanswerable)")


# ---------------------------------------------------------------------------
# SQuAD — 500 answerable (disjoint) + 500 unanswerable
# ---------------------------------------------------------------------------

def sample_squad() -> None:
    training_ids = load_training_answerable_ids("squad")
    excluded_indices = load_excluded_source_indices("squad")

    # Answerable: sample 500 from full dataset excluding training pool
    ans_pool: list[dict] = []
    with open(PROCESSED_DIR / "squad.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("answerable") and row["id"] not in training_ids:
                ans_pool.append(row)

    assert len(ans_pool) >= 500, f"Not enough answerable: {len(ans_pool)}"
    answerable = random.sample(ans_pool, 500)

    # Unanswerable: sample 500 from pool excluding mining-selected
    unans_pool: list[dict] = []
    with open(SAMPLED_DIR / "squad_unanswerable_2000.jsonl") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i not in excluded_indices:
                unans_pool.append(json.loads(line))

    assert len(unans_pool) >= 500, f"Not enough unanswerable: {len(unans_pool)}"
    unanswerable = random.sample(unans_pool, 500)

    all_rows = answerable + unanswerable
    random.shuffle(all_rows)

    out_path = OUT_DIR / "squad_1000.jsonl"
    with open(out_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out_path}  (500 answerable, 500 unanswerable)")


# ---------------------------------------------------------------------------
# UltraChat — 1000 instruction/QA pairs
# ---------------------------------------------------------------------------

def sample_ultrachat() -> None:
    print("Loading UltraChat 200k (train_sft split)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    # Load held-out eval IDs to exclude
    held_out_ids: set = set()
    held_out_path = OUT_DIR / "ultrachat_1000.jsonl"
    if held_out_path.exists():
        with open(held_out_path) as f:
            for line in f:
                if line.strip():
                    held_out_ids.add(json.loads(line)["id"])

    pool = []
    for example in ds:
        messages = example.get("messages", [])
        if len(messages) < 2:
            continue
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        pool.append({
            "id": example.get("prompt_id", example.get("id", "")),
            "prompt": messages[0]["content"],
            "response": messages[1]["content"],
            "messages": messages,
        })

    # Split into held-out (1000) and retain (2000), both disjoint
    non_held_out = [r for r in pool if r["id"] not in held_out_ids]
    assert len(non_held_out) >= 3000, f"Not enough examples: {len(non_held_out)}"

    # Held-out eval set (only write if not already present)
    if not held_out_path.exists():
        held_out = random.sample(non_held_out, 1000)
        with open(held_out_path, "w") as f:
            for row in held_out:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {held_out_path}")
        held_out_ids = {r["id"] for r in held_out}
    else:
        print(f"Skipping {held_out_path} (already exists)")

    # Retain sample — disjoint from held-out eval
    retain_pool = [r for r in non_held_out if r["id"] not in held_out_ids]
    assert len(retain_pool) >= 1000, f"Not enough retain examples: {len(retain_pool)}"
    retain = random.sample(retain_pool, 1000)
    retain_path = SAMPLED_DIR / "ultrachat_retain_1000.jsonl"
    with open(retain_path, "w") as f:
        for row in retain:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {retain_path}  (disjoint from held-out eval)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_kuq()
    sample_squad()
    sample_ultrachat()
    print("Done.")
