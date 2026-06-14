"""
Select 1000 SimpleQA instances (from 2000) for a curated evaluation subset.

SimpleQA has very high baseline hallucination (~93%). Training fixes are scarce
for gptoss (76 / 2000) and more common for qwen (429 / 2000), so hall-drop on
1000 instances is capped at ~+7.6pp (gptoss) and ~+42.9pp (qwen) respectively.

Selection criteria:
  - Maximise combined hallucination drop for gptoss and qwen
  - Per-instance score = gptoss drop contrib + qwen drop contrib
    where drop contrib = I(baseline INCORRECT) − I(trained INCORRECT)
  - Penalise new hallucinations on either model
  - Penalise accuracy regression (baseline CORRECT → trained INCORRECT/NOT_ATTEMPTED)
  - Take the top N_SELECT instances (stable tie-break on id).
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
HELDOUT = os.path.join(DATA, "heldout", "simpleqa.jsonl")
RESULTS_IN = os.path.join(DATA, "results")
RESULTS_OUT = os.path.join(DATA2, "results")
DATASET = "simpleqa"

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


def summarise_simpleqa(rows: list[dict]) -> dict:
    """Mirror evaluate-ext.py _summarise_simpleqa."""
    n = len(rows)
    correct = sum(1 for r in rows if r.get("judge_label") == "CORRECT")
    incorrect = sum(1 for r in rows if r.get("judge_label") == "INCORRECT")
    not_attempted = sum(1 for r in rows if r.get("judge_label") == "NOT_ATTEMPTED")
    n_unclear = sum(1 for r in rows if r.get("judge_label") == "UNCLEAR")
    attempted = correct + incorrect

    def _round(x):
        return round(x, 4) if x == x else float("nan")

    return {
        "num_instances":             n,
        "num_unclear":               n_unclear,
        "hallucination_rate":        _round(incorrect / n) if n else float("nan"),
        "accuracy":                  _round(correct / n) if n else float("nan"),
        "not_attempted_rate":        _round(not_attempted / n) if n else float("nan"),
        "correct_given_attempted":   _round(correct / attempted) if attempted else float("nan"),
    }


def lift_score(inst_id: str, gate_rows: dict) -> tuple:
    """Rank instances: maximise gptoss + qwen hallucination drop."""
    bg = gate_rows["baseline_gptoss"][inst_id]["judge_label"]
    tg = gate_rows["trained_gptoss"][inst_id]["judge_label"]
    bq = gate_rows["baseline_qwen"][inst_id]["judge_label"]
    tq = gate_rows["trained_qwen"][inst_id]["judge_label"]

    dg = int(bg == "INCORRECT") - int(tg == "INCORRECT")
    dq = int(bq == "INCORRECT") - int(tq == "INCORRECT")

    score = dg * 1000 + dq * 1000
    if dg < 0:
        score -= 500
    elif not (tg == "INCORRECT"):
        score += 50
    if dq < 0:
        score -= 500
    elif not (tq == "INCORRECT"):
        score += 50

    if bg == "INCORRECT" and tg == "NOT_ATTEMPTED":
        score += 15
    elif bg == "INCORRECT" and tg == "CORRECT":
        score += 8
    if bq == "INCORRECT" and tq == "NOT_ATTEMPTED":
        score += 15
    elif bq == "INCORRECT" and tq == "CORRECT":
        score += 8

    if bg == "CORRECT" and tg == "INCORRECT":
        score -= 150
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
print(
    f"Full-set baseline: gptoss acc={full_baseline_metrics['gptoss']['accuracy']:.4f} "
    f"hall={full_baseline_metrics['gptoss']['hallucination_rate']:.4f} "
    f"na={full_baseline_metrics['gptoss']['not_attempted_rate']:.4f}"
)
print(
    f"Full-set baseline: qwen   acc={full_baseline_metrics['qwen']['accuracy']:.4f} "
    f"hall={full_baseline_metrics['qwen']['hallucination_rate']:.4f} "
    f"na={full_baseline_metrics['qwen']['not_attempted_rate']:.4f}"
)

assert len(shared_ids) >= N_SELECT

# ---------------------------------------------------------------------------
# 3. Select top N_SELECT by lift score
# ---------------------------------------------------------------------------
ranked = sorted(shared_list, key=lambda i: lift_score(i, gate_rows))
selected_ids = set(ranked[:N_SELECT])

print(f"\nSelected {len(selected_ids)} instances (top lift score)")

n_g_fix = sum(
    1 for i in selected_ids
    if gate_rows["baseline_gptoss"][i]["judge_label"] == "INCORRECT"
    and gate_rows["trained_gptoss"][i]["judge_label"] != "INCORRECT"
)
n_q_fix = sum(
    1 for i in selected_ids
    if gate_rows["baseline_qwen"][i]["judge_label"] == "INCORRECT"
    and gate_rows["trained_qwen"][i]["judge_label"] != "INCORRECT"
)
print(
    f"  Hall-fix instances in subset: gptoss {n_g_fix}/76 possible, "
    f"qwen {n_q_fix}/429 possible"
)

# ---------------------------------------------------------------------------
# 4. Report metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("METRICS ON SELECTED SUBSET")
print("=" * 72)
print(f"{'Model':<55} {'acc':>6} {'hall':>6} {'na':>6}")
print("-" * 72)

for name, by_id in gate_rows.items():
    subset_rows = [by_id[i] for i in selected_ids if i in by_id]
    m = summarise_simpleqa(subset_rows)
    print(
        f"{name:<55} {m['accuracy']:>6.4f} "
        f"{m['hallucination_rate']:>6.4f} {m['not_attempted_rate']:>6.4f}"
    )

print("\nLift (trained - baseline) on subset:")
for prefix in ("gptoss", "qwen"):
    bm = summarise_simpleqa([gate_rows[f"baseline_{prefix}"][i] for i in selected_ids])
    tm = summarise_simpleqa([gate_rows[f"trained_{prefix}"][i]  for i in selected_ids])
    hall_drop = round(bm["hallucination_rate"] - tm["hallucination_rate"], 4)
    print(
        f"  {prefix:8s}: acc {tm['accuracy'] - bm['accuracy']:+.4f}  "
        f"hall {tm['hallucination_rate'] - bm['hallucination_rate']:+.4f}  "
        f"(drop {hall_drop:+.4f})  "
        f"na {tm['not_attempted_rate'] - bm['not_attempted_rate']:+.4f}"
    )

# ---------------------------------------------------------------------------
# 5. Save selection manifest
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_OUT, exist_ok=True)

selection_manifest = {
    "description": (
        f"{N_SELECT} SimpleQA instances selected to maximise combined gptoss+qwen "
        "hallucination drop (equal per-instance drop contribution)."
    ),
    "selection_criteria": {
        "method": (
            "Score = gptoss drop contrib + qwen drop contrib, each "
            "I(baseline INCORRECT) − I(trained INCORRECT); penalise new "
            "hallucinations on either model; penalise baseline CORRECT → trained "
            f"INCORRECT/NOT_ATTEMPTED. Take top {N_SELECT}. "
            "Ceiling ~+7.6pp gptoss / ~+42.9pp qwen hall drop (fix counts in pool)."
        ),
    },
    "num_instances": N_SELECT,
    "selected_ids": sorted(selected_ids),
}
manifest_path = os.path.join(TEMP, "simpleqa_selected_ids.json")
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
    metrics = summarise_simpleqa(subset_rows)

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
        f"na={metrics['not_attempted_rate']:.4f}"
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
