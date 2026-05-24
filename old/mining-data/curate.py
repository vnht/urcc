#!/usr/bin/env python3
"""
Select the most obviously committed retained examples per (model × dataset) pair.

Scores each retained completion on how directly and confidently it commits —
penalising hedging language, source deferrals, and very long or uncertain answers.
Writes top-N per pair to mining-data/mining-selected/.

Usage:
    python3 mining-data/select.py            # 500 per pair (default)
    python3 mining-data/select.py --n 200    # custom count
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "mining-results"
SELECTED_DIR = BASE_DIR / "mining-selected"
SELECTED_DIR.mkdir(exist_ok=True)

# ─── Scoring ──────────────────────────────────────────────────────────────────

# Borderline / hedging signals — lower score
HEDGING = re.compile(
    r"\b(might|may|could|perhaps|possibly|probably|i think|i believe|i'm not sure"
    r"|unclear|unknown|it depends|not certain|not clear|it is not known"
    r"|no one knows|nobody knows|cannot be determined|hard to say)\b",
    re.IGNORECASE,
)

# Source / context deferral signals — lower score
DEFERRAL = re.compile(
    r"\b(the text|the context|the passage|the article|the excerpt|the document"
    r"|based on|according to|as mentioned|as stated|as described"
    r"|not mentioned|does not mention|doesn't mention|no mention"
    r"|not provided|not stated|not specified|cannot be found)\b",
    re.IGNORECASE,
)

# Uncertainty punctuation
UNCERTAINTY = re.compile(r"\?")


def score(row: dict) -> float:
    """
    Higher score = more obviously committed.
    Range roughly 0–1; borderline examples cluster near 0.
    """
    text = row.get("full_completion_clean", "")
    words = text.split()
    n_words = len(words)

    s = 1.0

    # Penalise hedging language (each match -0.25, capped)
    s -= min(len(HEDGING.findall(text)) * 0.25, 0.75)

    # Penalise source deferral language (each match -0.3, capped)
    s -= min(len(DEFERRAL.findall(text)) * 0.30, 0.90)

    # Penalise question marks in the answer
    s -= UNCERTAINTY.search(text) is not None and 0.2 or 0.0

    # Reward concise, direct answers (< 30 words gets a bonus; very long answers penalised)
    if n_words <= 5:
        s += 0.30
    elif n_words <= 15:
        s += 0.20
    elif n_words <= 30:
        s += 0.10
    elif n_words > 80:
        s -= 0.20

    # Reward valid k4 prefix (commitment starts within 4 tokens)
    if row.get("prefix_valid_k4"):
        s += 0.10

    # Penalise completions that start with "I " (often hedging or first-person dodge)
    if text.strip().startswith("I "):
        s -= 0.15

    return s


def is_borderline(row: dict) -> bool:
    """Hard filter: exclude rows that contain any deferral language."""
    text = row.get("full_completion_clean", "")
    return bool(DEFERRAL.search(text))


# ─── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select top committed examples per model×dataset pair")
    p.add_argument("--n", type=int, default=500, metavar="N",
                   help="Number of examples to select per pair (default: 500)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = args.n

    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"No result files found in {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"{'pair':<32} {'retained':>8} {'eligible':>9} {'selected':>9}")
    print("-" * 62)

    for path in files:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        retained = [r for r in rows if r.get("retained")]

        # Hard-filter borderline examples
        eligible = [r for r in retained if not is_borderline(r)]

        # Score and rank
        eligible.sort(key=score, reverse=True)

        selected = eligible[:n]

        # If still short, fill remaining slots from borderline pool (best-scored first)
        if len(selected) < n:
            borderline = [r for r in retained if is_borderline(r)]
            borderline.sort(key=score, reverse=True)
            needed = n - len(selected)
            selected += borderline[:needed]

        out_path = SELECTED_DIR / path.name
        with open(out_path, "w") as f:
            for r in selected:
                f.write(json.dumps(r) + "\n")

        print(f"{path.stem:<32} {len(retained):>8} {len(eligible):>9} {len(selected):>9}")

    print(f"\nWrote selected examples to {SELECTED_DIR}/")


if __name__ == "__main__":
    main()
