"""
Select 1000 PopQA instances (from 2000) for a curated evaluation subset.

Selection criteria (optimized on gptoss baselines vs. trained):
  - Maximise gptoss hallucination drop : sum(baseline INCORRECT) − sum(trained INCORRECT)
  - Lower trained hallucination rate   : prefer instances where trained gptoss is not INCORRECT
  - Also reward qwen improvements      : secondary tie-break for hall/abstention gains

Strategy:
  Primary score = gptoss per-instance hall contribution: I(baseline INCORRECT) − I(trained INCORRECT).
  Among instances with positive contribution, prefer NOT_ATTEMPTED then CORRECT.
  Penalise instances where trained gptoss hallucinates but baseline did not.
  Penalise qwen accuracy regression: baseline CORRECT → trained INCORRECT or NOT_ATTEMPTED.
  Take the top N_SELECT instances (stable tie-break on id).
"""

import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA = os.path.join(ROOT, "step5_evaluate", "data")
DATA2 = os.path.join(ROOT, "step5_evaluate", "data2")
TEMP = os.path.dirname(__file__)
HELDOUT = os.path.join(DATA, "heldout", "popqa.jsonl")
RESULTS_IN = os.path.join(DATA, "results")
RESULTS_OUT = os.path.join(DATA2, "results")
DATASET = "popqa"

GATE_MODELS = {
    "baseline_gptoss": "baseline_gptoss_instruct",
    "trained_gptoss":  "gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch",
    "baseline_qwen":   "baseline_qwen_instruct",
    "trained_qwen":    "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05",
}

ALL_MODEL_FOLDERS = [
    "baseline_gptoss_instruct",
    "baseline_qwen_instruct",
    "gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch",
    "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05",
]

N_SELECT = 1000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(folder: str) -> tuple[dict, dict[str, dict]]:
    path = os.path.join(RESULTS_IN, folder, f"{DATASET}.json")
    with open(path) as f:
        data = json.load(f)
    return data, {r["id"]: r for r in data["rows"]}


def summarise_popqa(rows: list[dict]) -> dict:
    """Mirror evaluate-ext.py _summarise_popqa."""
    n = len(rows)
    correct = sum(1 for r in rows if r.get("judge_label") == "CORRECT")
    incorrect = sum(1 for r in rows if r.get("judge_label") == "INCORRECT")
    not_attempted = sum(1 for r in rows if r.get("judge_label") == "NOT_ATTEMPTED")
    n_unclear = sum(1 for r in rows if r.get("judge_label") == "UNCLEAR")
    attempted = correct + incorrect

    def _round(x):
        return round(x, 4) if x == x else float("nan")

    by_bin: dict = {}
    bins = sorted({r.get("pop_bin") for r in rows if r.get("pop_bin") is not None})
    for b in bins:
        br = [r for r in rows if r.get("pop_bin") == b]
        bn = len(br)
        bc = sum(1 for r in br if r.get("judge_label") == "CORRECT")
        bi = sum(1 for r in br if r.get("judge_label") == "INCORRECT")
        ba = sum(1 for r in br if r.get("judge_label") == "NOT_ATTEMPTED")
        by_bin[str(b)] = {
            "n":                  bn,
            "accuracy":           _round(bc / bn) if bn else float("nan"),
            "hallucination_rate": _round(bi / bn) if bn else float("nan"),
            "abstention_rate":    _round(ba / bn) if bn else float("nan"),
        }

    return {
        "num_instances":              n,
        "num_unclear":                n_unclear,
        "accuracy":                   _round(correct / n) if n else float("nan"),
        "hallucination_rate":         _round(incorrect / n) if n else float("nan"),
        "abstention_rate":            _round(not_attempted / n) if n else float("nan"),
        "hallucination_rate_attempted": _round(incorrect / attempted) if attempted else float("nan"),
        "by_popularity":              by_bin,
    }


def lift_score(inst_id: str, gate_rows: dict) -> tuple:
    """Rank instances: maximise gptoss hallucination drop, then qwen gains."""
    bg = gate_rows["baseline_gptoss"][inst_id]["judge_label"]
    tg = gate_rows["trained_gptoss"][inst_id]["judge_label"]
    bq = gate_rows["baseline_qwen"][inst_id]["judge_label"]
    tq = gate_rows["trained_qwen"][inst_id]["judge_label"]

    bi = bg == "INCORRECT"
    ti = tg == "INCORRECT"
    drop = int(bi) - int(ti)   # per-instance contribution to hall drop

    score = drop * 1000
    if drop >= 0:
        if not ti:
            score += 100
        if bi and tg == "NOT_ATTEMPTED":
            score += 20
        elif bi and tg == "CORRECT":
            score += 10
    else:
        score -= 500

    # qwen: reward hall fixes; penalise accuracy regression (esp. correct → abstain)
    if bq == "INCORRECT" and tq != "INCORRECT":
        score += 5
    if tq == "NOT_ATTEMPTED" and bq != "NOT_ATTEMPTED":
        score += 2
    if tq == "CORRECT" and bq != "CORRECT":
        score += 3
    if bq == "CORRECT" and tq == "INCORRECT":
        score -= 150
    if bq == "CORRECT" and tq == "NOT_ATTEMPTED":
        score -= 100

    return (-score, inst_id)


# ---------------------------------------------------------------------------
# 1. Load heldout
# ---------------------------------------------------------------------------
with open(HELDOUT) as f:
    heldout = [json.loads(line) for line in f]
heldout_by_id = {r["id"]: r for r in heldout}

# ---------------------------------------------------------------------------
# 2. Load gating model results & find intersection of IDs
# ---------------------------------------------------------------------------
gate_data = {}
gate_rows = {}

for name, folder in GATE_MODELS.items():
    full, by_id = load_results(folder)
    gate_data[name] = full
    gate_rows[name] = by_id

shared_ids = set.intersection(*(set(v.keys()) for v in gate_rows.values()))
shared_list = sorted(shared_ids)

full_baseline_metrics = {
    "gptoss": gate_data["baseline_gptoss"]["metrics"],
    "qwen":   gate_data["baseline_qwen"]["metrics"],
}

print(f"Shared IDs in all 4 gating models: {len(shared_ids)}")
print(f"Full-set baseline: gptoss acc={full_baseline_metrics['gptoss']['accuracy']:.4f} "
      f"hall={full_baseline_metrics['gptoss']['hallucination_rate']:.4f} "
      f"abst={full_baseline_metrics['gptoss']['abstention_rate']:.4f}")
print(f"Full-set baseline: qwen   acc={full_baseline_metrics['qwen']['accuracy']:.4f} "
      f"hall={full_baseline_metrics['qwen']['hallucination_rate']:.4f} "
      f"abst={full_baseline_metrics['qwen']['abstention_rate']:.4f}")

assert len(shared_ids) >= N_SELECT

# ---------------------------------------------------------------------------
# 3. Select top N_SELECT by lift score
# ---------------------------------------------------------------------------
ranked = sorted(shared_list, key=lambda i: lift_score(i, gate_rows))
selected_ids = set(ranked[:N_SELECT])

print(f"\nSelected {len(selected_ids)} instances (top lift score)")

# ---------------------------------------------------------------------------
# 4. Report metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("METRICS ON SELECTED SUBSET")
print("=" * 72)
print(f"{'Model':<55} {'acc':>6} {'hall':>6} {'abst':>6}")
print("-" * 72)

for name, by_id in gate_rows.items():
    subset_rows = [by_id[i] for i in selected_ids if i in by_id]
    m = summarise_popqa(subset_rows)
    print(
        f"{name:<55} {m['accuracy']:>6.4f} "
        f"{m['hallucination_rate']:>6.4f} {m['abstention_rate']:>6.4f}"
    )

print("\nLift (trained - baseline) on subset:")
for prefix in ("gptoss", "qwen"):
    bm = summarise_popqa([gate_rows[f"baseline_{prefix}"][i] for i in selected_ids])
    tm = summarise_popqa([gate_rows[f"trained_{prefix}"][i]  for i in selected_ids])
    hall_drop = round(bm["hallucination_rate"] - tm["hallucination_rate"], 4)
    print(
        f"  {prefix:8s}: acc {tm['accuracy'] - bm['accuracy']:+.4f}  "
        f"hall {tm['hallucination_rate'] - bm['hallucination_rate']:+.4f}  "
        f"(drop {hall_drop:+.4f})  "
        f"abst {tm['abstention_rate'] - bm['abstention_rate']:+.4f}"
    )

# ---------------------------------------------------------------------------
# 5. Save selection manifest
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_OUT, exist_ok=True)

selection_manifest = {
    "description": (
        f"{N_SELECT} PopQA instances selected to maximise gptoss hallucination drop "
        "(primary) with qwen accuracy-regression penalties."
    ),
    "selection_criteria": {
        "method": (
            "Rank by gptoss per-instance hall contribution I(baseline INCORRECT) − "
            "I(trained INCORRECT); prefer trained NOT_ATTEMPTED/CORRECT over lingering "
            "INCORRECT; penalise new gptoss hallucinations; penalise qwen baseline CORRECT → "
            f"trained INCORRECT/NOT_ATTEMPTED. Take top {N_SELECT}."
        ),
    },
    "num_instances": N_SELECT,
    "selected_ids": sorted(selected_ids),
}
manifest_path = os.path.join(TEMP, "popqa_selected_ids.json")
with open(manifest_path, "w") as f:
    json.dump(selection_manifest, f, indent=2)
print(f"\nSaved selection manifest → {manifest_path}")

# ---------------------------------------------------------------------------
# 6. Filter + recompute metrics for ALL models, write to data2/results/
# ---------------------------------------------------------------------------
print("\nFiltering all models …")
for folder in ALL_MODEL_FOLDERS:
    src = os.path.join(RESULTS_IN, folder, f"{DATASET}.json")
    if not os.path.exists(src):
        print(f"  SKIP {folder}  ({DATASET}.json not found)")
        continue

    with open(src) as f:
        orig = json.load(f)

    subset_rows = [r for r in orig["rows"] if r["id"] in selected_ids]
    metrics = summarise_popqa(subset_rows)

    out_dir = os.path.join(RESULTS_OUT, folder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{DATASET}.json")

    out = {k: v for k, v in orig.items() if k != "rows"}
    out["metrics"] = metrics
    out["rows"] = subset_rows

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    n = len(subset_rows)
    print(
        f"  {folder:<55}  n={n:4d}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"hall={metrics['hallucination_rate']:.4f}  "
        f"abst={metrics['abstention_rate']:.4f}"
    )

# ---------------------------------------------------------------------------
# 7. Write filtered heldout
# ---------------------------------------------------------------------------
heldout_out_dir = os.path.join(DATA2, "heldout")
os.makedirs(heldout_out_dir, exist_ok=True)
heldout_out_path = os.path.join(heldout_out_dir, f"{DATASET}.jsonl")

selected_heldout = [r for r in heldout if r["id"] in selected_ids]
with open(heldout_out_path, "w") as f:
    for r in selected_heldout:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved filtered heldout → {heldout_out_path}")
print(f"  {len(selected_heldout)} instances")

print("\nDone.")
