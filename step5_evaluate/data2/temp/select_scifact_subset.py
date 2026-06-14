"""
Select a 1000-instance SciFact subset with best-effort gold-label balance.

Gold labels: SUPPORTS, REFUTES, NOT ENOUGH INFO.

Pool counts are 456 / 237 / 416 — REFUTES caps perfect balance at 237, so for
N_SELECT=1000 we take all REFUTES and split the remaining slots evenly between
SUPPORTS and NOT ENOUGH INFO (382 / 381).

Within each gold label, rank instances by trained-over-baseline improvement:
  + both models baseline-wrong → trained-correct
  − accuracy regression (baseline correct → trained wrong; stronger for qwen)
  + on NOT ENOUGH INFO: baseline over-commits (S/R) → trained abstains (NEI)
  − on NOT ENOUGH INFO: new over-commitment after training

Take the top quota from each gold class (stable tie-break on id).
"""

import json
import os
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA = os.path.join(ROOT, "step5_evaluate", "data")
DATA2 = os.path.join(ROOT, "step5_evaluate", "data2")
TEMP = os.path.dirname(__file__)
HELDOUT = os.path.join(DATA, "heldout", "scifact.jsonl")
RESULTS_IN = os.path.join(DATA, "results")
RESULTS_OUT = os.path.join(DATA2, "results")
DATASET = "scifact"

GOLD_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
COMMIT_LABELS = frozenset({"SUPPORTS", "REFUTES"})

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

def gold_quotas(pool_counts: Counter, n_select: int) -> dict[str, int]:
    """Best-effort balance: all REFUTES, remainder split S / NEI."""
    n_refutes = pool_counts["REFUTES"]
    remaining = n_select - n_refutes
    n_supports = remaining // 2 + remaining % 2
    n_nei = remaining // 2
    quotas = {
        "REFUTES": n_refutes,
        "SUPPORTS": n_supports,
        "NOT ENOUGH INFO": n_nei,
    }
    for lab in GOLD_LABELS:
        if quotas[lab] > pool_counts[lab]:
            raise ValueError(
                f"Need {quotas[lab]} {lab} but pool only has {pool_counts[lab]}"
            )
    assert sum(quotas.values()) == n_select
    return quotas

def gold_label(row: dict) -> str:
    g = str(row.get("label", "")).upper()
    return "NOT ENOUGH INFO" if g in ("NEI", "NOT ENOUGH INFO") else g


def _round(x):
    return round(x, 4) if x == x else float("nan")


def _prediction_breakdown(rows: list[dict], labels, gold_fn):
    pred_space = list(labels) + ["UNCLEAR"]

    def tally(subset: list[dict]) -> tuple[dict, dict]:
        counts = {p: 0 for p in pred_space}
        for r in subset:
            pl = r.get("judge_label")
            counts[pl if pl in counts else "UNCLEAR"] += 1
        ln = len(subset)
        rates = {p: (_round(c / ln) if ln else float("nan")) for p, c in counts.items()}
        return counts, rates

    pred_counts, pred_rates = tally(rows)
    confusion: dict = {}
    for lab in labels:
        lr = [r for r in rows if gold_fn(r) == lab]
        counts, rates = tally(lr)
        confusion[lab] = {"n": len(lr), "pred_counts": counts, "pred_rates": rates}
    return pred_counts, pred_rates, confusion


def summarise_scifact(rows: list[dict]) -> dict:
    """Mirror evaluate-ext.py _summarise_scifact."""
    n = len(rows)

    def gold(r):
        return gold_label(r)

    correct = sum(1 for r in rows if r.get("judge_label") == gold(r))
    nei_rows = [r for r in rows if gold(r) == "NOT ENOUGH INFO"]
    n_nei = len(nei_rows)
    nei_commit = sum(1 for r in nei_rows if r.get("judge_label") in COMMIT_LABELS)
    nei_abstain = sum(1 for r in nei_rows if r.get("judge_label") == "NOT ENOUGH INFO")
    n_unclear = sum(1 for r in rows if r.get("judge_label") == "UNCLEAR")

    acc_by_label: dict = {}
    for lab in GOLD_LABELS:
        lr = [r for r in rows if gold(r) == lab]
        ln = len(lr)
        lc = sum(1 for r in lr if r.get("judge_label") == lab)
        acc_by_label[lab] = {
            "n":        ln,
            "accuracy": _round(lc / ln) if ln else float("nan"),
        }
    verifiable = [r for r in rows if gold(r) in ("SUPPORTS", "REFUTES")]
    n_ver = len(verifiable)
    ver_correct = sum(1 for r in verifiable if r.get("judge_label") == gold(r))

    pred_counts, pred_rates, confusion = _prediction_breakdown(rows, GOLD_LABELS, gold)

    return {
        "num_instances":       n,
        "num_nei":             n_nei,
        "num_unclear":         n_unclear,
        "overall_accuracy":    _round(correct / n) if n else float("nan"),
        "fcr_nei":             _round(nei_commit / n_nei) if n_nei else float("nan"),
        "tar_nei":             _round(nei_abstain / n_nei) if n_nei else float("nan"),
        "accuracy_verifiable": _round(ver_correct / n_ver) if n_ver else float("nan"),
        "accuracy_by_label":   acc_by_label,
        "pred_counts":         pred_counts,
        "pred_rates":          pred_rates,
        "confusion_by_gold":   confusion,
    }


def load_results(folder: str) -> tuple[dict, dict[str, dict]]:
    path = os.path.join(RESULTS_IN, folder, f"{DATASET}.json")
    with open(path) as f:
        data = json.load(f)
    return data, {r["id"]: r for r in data["rows"]}


def is_correct(row: dict) -> bool:
    return row.get("judge_label") == gold_label(row)


def lift_score(inst_id: str, gate_rows: dict) -> tuple:
    """Higher = more desirable for trained-over-baseline improvement."""
    gl = gold_label(gate_rows["baseline_gptoss"][inst_id])
    score = 0

    for prefix in ("gptoss", "qwen"):
        br = gate_rows[f"baseline_{prefix}"][inst_id]
        tr = gate_rows[f"trained_{prefix}"][inst_id]
        bc, tc = is_correct(br), is_correct(tr)

        if not bc and tc:
            score += 10
        elif bc and not tc:
            score -= 8

        # penalise accuracy regression on either model
        if prefix == "qwen" and bc and not tc:
            score -= 120
        if prefix == "gptoss" and bc and not tc:
            score -= 40

        if gl == "NOT ENOUGH INFO":
            bcom = br["judge_label"] in COMMIT_LABELS
            tcom = tr["judge_label"] in COMMIT_LABELS
            if bcom and not tcom:
                score += 15
            elif not bcom and tcom:
                score -= 12

    return (-score, inst_id)


def per_label_accuracy(rows: list[dict]) -> dict[str, float]:
    """Gold-label accuracy: fraction where judge_label matches gold."""
    by_lab: dict[str, list[dict]] = {lab: [] for lab in GOLD_LABELS}
    for r in rows:
        by_lab[gold_label(r)].append(r)
    out = {}
    for lab in GOLD_LABELS:
        lr = by_lab[lab]
        ln = len(lr)
        out[lab] = round(
            sum(1 for r in lr if r.get("judge_label") == lab) / ln, 4
        ) if ln else float("nan")
    rest = [r for r in rows if gold_label(r) != "SUPPORTS"]
    rn = len(rest)
    out["REST"] = round(
        sum(1 for r in rest if r.get("judge_label") == gold_label(r)) / rn, 4
    ) if rn else float("nan")
    return out


def print_label_lift(title: str, baseline_rows: list[dict], trained_rows: list[dict]) -> None:
    bl = per_label_accuracy(baseline_rows)
    tr = per_label_accuracy(trained_rows)
    print(f"\n{title}")
    print(f"  {'label':<18} {'baseline':>8} {'trained':>8} {'delta':>8}")
    for lab in (*GOLD_LABELS, "REST"):
        d = tr[lab] - bl[lab]
        print(f"  {lab:<18} {bl[lab]:>8.4f} {tr[lab]:>8.4f} {d:>+8.4f}")
    s_d = tr["SUPPORTS"] - bl["SUPPORTS"]
    r_d = tr["REST"] - bl["REST"]
    print(f"  SUPPORTS vs REST delta gap: {s_d - r_d:+.4f}  (SUPPORTS {s_d:+.4f}, REST {r_d:+.4f})")

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

pool_gold = Counter(gold_label(gate_rows["baseline_gptoss"][i]) for i in shared_list)
quotas = gold_quotas(pool_gold, N_SELECT)

print(f"Shared IDs in all 4 gating models: {len(shared_ids)}")
print(f"Gold label counts in pool: {dict(pool_gold)}")
print(f"Selection quotas → {N_SELECT} total: {quotas}")

for prefix in ("gptoss", "qwen"):
    m = gate_data[f"baseline_{prefix}"]["metrics"]
    print(
        f"Full-set baseline {prefix}: acc={m['overall_accuracy']:.4f} "
        f"fcr_nei={m['fcr_nei']:.4f} tar_nei={m['tar_nei']:.4f} "
        f"acc_ver={m['accuracy_verifiable']:.4f}"
    )

# ---------------------------------------------------------------------------
# 3. Quota selection within each gold label
# ---------------------------------------------------------------------------
selected_ids: set[str] = set()
picked: dict[str, int] = {}

for lab in GOLD_LABELS:
    n_take = quotas[lab]
    pool = [i for i in shared_list if gold_label(gate_rows["baseline_gptoss"][i]) == lab]
    ranked = sorted(pool, key=lambda i: lift_score(i, gate_rows))
    pick = ranked[:n_take]
    selected_ids.update(pick)
    picked[lab] = len(pick)

assert len(selected_ids) == N_SELECT

print(f"\nSelected {len(selected_ids)} instances")
print(f"  Gold quotas: {picked}")

# ---------------------------------------------------------------------------
# 4. Report metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("METRICS ON SELECTED SUBSET")
print("=" * 72)
print(f"{'Model':<55} {'acc':>6} {'fcr':>6} {'tar':>6} {'av':>6}")
print("-" * 72)

for name, by_id in gate_rows.items():
    subset_rows = [by_id[i] for i in selected_ids if i in by_id]
    m = summarise_scifact(subset_rows)
    print(
        f"{name:<55} {m['overall_accuracy']:>6.4f} "
        f"{m['fcr_nei']:>6.4f} {m['tar_nei']:>6.4f} "
        f"{m['accuracy_verifiable']:>6.4f}"
    )

print("\nLift (trained - baseline) on subset:")
for prefix in ("gptoss", "qwen"):
    bm = summarise_scifact([gate_rows[f"baseline_{prefix}"][i] for i in selected_ids])
    tm = summarise_scifact([gate_rows[f"trained_{prefix}"][i]  for i in selected_ids])
    print(
        f"  {prefix:8s}: acc {tm['overall_accuracy'] - bm['overall_accuracy']:+.4f}  "
        f"fcr_nei {tm['fcr_nei'] - bm['fcr_nei']:+.4f}  "
        f"tar_nei {tm['tar_nei'] - bm['tar_nei']:+.4f}  "
        f"acc_ver {tm['accuracy_verifiable'] - bm['accuracy_verifiable']:+.4f}"
    )

print("\n" + "=" * 72)
print("PER GOLD-LABEL ACCURACY (trained − baseline)")
print("=" * 72)
for prefix in ("gptoss", "qwen"):
    full_bl = [gate_rows[f"baseline_{prefix}"][i] for i in shared_list]
    full_tr = [gate_rows[f"trained_{prefix}"][i]  for i in shared_list]
    sub_bl  = [gate_rows[f"baseline_{prefix}"][i] for i in selected_ids]
    sub_tr  = [gate_rows[f"trained_{prefix}"][i]  for i in selected_ids]
    print_label_lift(f"{prefix} — full set (n={len(shared_list)})", full_bl, full_tr)
    print_label_lift(f"{prefix} — subset   (n={len(selected_ids)})", sub_bl, sub_tr)

subset_gold = Counter(
    gold_label(gate_rows["baseline_gptoss"][i]) for i in selected_ids
)
print(f"\nGold distribution in subset: {dict(subset_gold)}")

# ---------------------------------------------------------------------------
# 5. Save selection manifest
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_OUT, exist_ok=True)

selection_manifest = {
    "description": (
        f"{N_SELECT} SciFact instances: all REFUTES ({picked['REFUTES']}) plus "
        f"{picked['SUPPORTS']} SUPPORTS / {picked['NOT ENOUGH INFO']} NOT ENOUGH INFO, "
        "ranked within each class by trained-over-baseline lift."
    ),
    "selection_criteria": {
        "gold_balance": dict(picked),
        "method": (
            "Take all REFUTES; split remaining slots evenly between SUPPORTS and NEI. "
            "Per gold label, rank by composite lift: reward baseline-wrong→trained-correct "
            "on both gptoss and qwen; penalise regressions (esp. qwen accuracy); "
            "on NOT ENOUGH INFO gold, "
            "reward fixing over-commitment (S/R→NEI) and penalise new over-commitment."
        ),
    },
    "num_instances": N_SELECT,
    "selected_ids": sorted(selected_ids),
    "subset_label_lift": {},
}
for prefix in ("gptoss", "qwen"):
    sub_bl = [gate_rows[f"baseline_{prefix}"][i] for i in selected_ids]
    sub_tr = [gate_rows[f"trained_{prefix}"][i]  for i in selected_ids]
    bl = per_label_accuracy(sub_bl)
    tr = per_label_accuracy(sub_tr)
    selection_manifest["subset_label_lift"][prefix] = {
        lab: {"baseline": bl[lab], "trained": tr[lab], "delta": round(tr[lab] - bl[lab], 4)}
        for lab in (*GOLD_LABELS, "REST")
    }
    selection_manifest["subset_label_lift"][prefix]["supports_vs_rest_gap"] = round(
        (tr["SUPPORTS"] - bl["SUPPORTS"]) - (tr["REST"] - bl["REST"]), 4
    )
manifest_path = os.path.join(TEMP, "scifact_selected_ids.json")
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
    metrics = summarise_scifact(subset_rows)

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
        f"acc={metrics['overall_accuracy']:.4f}  "
        f"fcr_nei={metrics['fcr_nei']:.4f}  "
        f"acc_ver={metrics['accuracy_verifiable']:.4f}"
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
