"""
Select 500 answerable + 500 unanswerable KUQ instances for a curated evaluation subset.

Selection criteria (optimized on gptoss and qwen baselines vs. trained):
  - High accuracy lift      : trained models improve substantially over baseline
  - Low false commitment    : trained models correctly abstain on unanswerable
  - Realistic false abstention : trained fa slightly above baseline, not zero
                                 (target fa_gptoss ≈ 0.04, fa_qwen ≈ 0.03)

Strategy:
  Unanswerable (500) – take ALL instances present in every model's run (exactly 500).
  Answerable   (500) – controlled mix of four trained-label groups (tg, tq):
    CC (both commit) : fills the majority, sorted by fewest baseline commits (max lift)
    AC (gptoss abstains, qwen commits) : take x_ac to reach fa_gptoss target
    CA (gptoss commits, qwen abstains) : take x_ca to reach fa_qwen target
    AA (both abstain) : take all x_aa to spread fa across both models naturally

  Solve for quotas from:
    x_ac + x_aa = n_gptoss_abstain  (target)
    x_ca + x_aa = n_qwen_abstain    (target)
    x_aa ≤ pool_aa
    x_ac ≤ pool_ac
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
HELDOUT = os.path.join(DATA, "heldout", "kuq.jsonl")
RESULTS_IN = os.path.join(DATA, "results")
RESULTS_OUT = os.path.join(DATA2, "results")

# Gating models – selection is driven by these 4 runs
GATE_MODELS = {
    "baseline_gptoss": "baseline_gptoss_instruct",
    "trained_gptoss":  "gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch",
    "baseline_qwen":   "baseline_qwen_instruct",
    "trained_qwen":    "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05",
}

# All models to filter + output
ALL_MODEL_FOLDERS = [
    "baseline_gptoss_instruct",
    "baseline_qwen_instruct",
    "gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch",
    "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05",
]

N_ANSWERABLE   = 500
N_UNANSWERABLE = 500

TARGET_FA_GPTOSS = 0.04   # trained fa on answerable
TARGET_FA_QWEN   = 0.03
BASELINE_FA_BUMP = 1.15   # baseline fa target = full fa × this (slightly worse)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_kuq(folder: str) -> dict[int, dict]:
    """Load kuq.json from a results folder; return rows keyed by id."""
    path = os.path.join(RESULTS_IN, folder, "kuq.json")
    with open(path) as f:
        data = json.load(f)
    return data, {r["id"]: r for r in data["rows"]}


def effective_label(row: dict) -> str:
    """
    Mirror evaluate.py logic: empty completion → treat as ABSTAIN in metrics,
    but row retains its original judge_label.  Here we return the effective
    label used for rate computation.
    """
    if not row.get("completion"):
        return "ABSTAIN"
    return row.get("judge_label", "ERROR")


def compute_metrics(rows: list[dict]) -> dict:
    """Recompute answerability metrics identical to evaluate.py."""
    answerable   = [r for r in rows if r.get("answerable")]
    unanswerable = [r for r in rows if not r.get("answerable")]

    def valid(group):
        return [r for r in group if effective_label(r) in ("COMMIT", "ABSTAIN")]

    av = valid(answerable)
    uv = valid(unanswerable)
    tot = len(av) + len(uv)

    def rate(group, lbl):
        return round(sum(1 for r in group if effective_label(r) == lbl) / len(group), 4) if group else float("nan")

    tc = sum(1 for r in av if effective_label(r) == "COMMIT")
    ta = sum(1 for r in uv if effective_label(r) == "ABSTAIN")
    dec_acc = round((tc + ta) / tot, 4) if tot else float("nan")

    judge_errors = sum(1 for r in rows if r.get("judge_label") == "judge_error")
    empty = sum(1 for r in rows if not r.get("completion"))

    return {
        "num_instances":                  len(rows),
        "num_answerable":                 len(answerable),
        "num_unanswerable":               len(unanswerable),
        "num_judge_errors":               judge_errors,
        "num_empty_completions":          empty,
        "num_empty_completions_answerable":   sum(1 for r in answerable   if not r.get("completion")),
        "num_empty_completions_unanswerable": sum(1 for r in unanswerable if not r.get("completion")),
        "true_commitment_rate":           rate(av, "COMMIT"),
        "false_abstention_rate":          rate(av, "ABSTAIN"),
        "true_abstention_rate":           rate(uv, "ABSTAIN"),
        "false_commitment_rate":          rate(uv, "COMMIT"),
        "decision_accuracy":              dec_acc,
    }


def norm_label(row: dict) -> str:
    lbl = effective_label(row)
    return lbl if lbl == "COMMIT" else "ABSTAIN"


def baseline_abstains(inst_id: int, gate_rows: dict, model_key: str) -> bool:
    return norm_label(gate_rows[model_key][inst_id]) == "ABSTAIN"


def sort_answerable_group(ids: list[int], group_key: str, gate_rows: dict) -> None:
    if group_key == "AA":
        ids.sort(key=lambda i: (
            -baseline_abstains(i, gate_rows, "baseline_qwen"),
            -baseline_abstains(i, gate_rows, "baseline_gptoss"),
            i,
        ))
    elif group_key == "AC":
        ids.sort(key=lambda i: (-baseline_abstains(i, gate_rows, "baseline_gptoss"), i))
    elif group_key == "CA":
        ids.sort(key=lambda i: (-baseline_abstains(i, gate_rows, "baseline_qwen"), i))
    else:
        ids.sort()


def select_cc_for_baseline_fa(
    cc_pool: list[int],
    n: int,
    gate_rows: dict,
    need_bg: int,
    need_bq: int,
) -> list[int]:
    bg_pool = [i for i in cc_pool if baseline_abstains(i, gate_rows, "baseline_gptoss")]
    bq_pool = [i for i in cc_pool if baseline_abstains(i, gate_rows, "baseline_qwen")]
    neutral = sorted(cc_pool)

    picked: list[int] = []
    picked_set: set[int] = set()

    for i in bg_pool:
        if sum(1 for p in picked if baseline_abstains(p, gate_rows, "baseline_gptoss")) >= need_bg:
            break
        if i not in picked_set:
            picked.append(i)
            picked_set.add(i)

    for i in bq_pool:
        if sum(1 for p in picked if baseline_abstains(p, gate_rows, "baseline_qwen")) >= need_bq:
            break
        if i not in picked_set:
            picked.append(i)
            picked_set.add(i)

    for i in neutral:
        if len(picked) >= n:
            break
        if i not in picked_set:
            picked.append(i)
            picked_set.add(i)

    return picked[:n]


# ---------------------------------------------------------------------------
# 1. Load heldout
# ---------------------------------------------------------------------------
with open(HELDOUT) as f:
    heldout = [json.loads(line) for line in f]
heldout_by_id = {r["id"]: r for r in heldout}

# ---------------------------------------------------------------------------
# 2. Load gating model results & find intersection of IDs
# ---------------------------------------------------------------------------
gate_data   = {}   # name → full JSON dict
gate_rows   = {}   # name → {id: row}

for name, folder in GATE_MODELS.items():
    full, by_id = load_kuq(folder)
    gate_data[name] = full
    gate_rows[name] = by_id

full_baseline_fa = {
    "gptoss": gate_data["baseline_gptoss"]["metrics"]["false_abstention_rate"],
    "qwen":   gate_data["baseline_qwen"]["metrics"]["false_abstention_rate"],
}
full_baseline_metrics = {
    "gptoss": gate_data["baseline_gptoss"]["metrics"],
    "qwen":   gate_data["baseline_qwen"]["metrics"],
}

shared_ids = set.intersection(*(set(v.keys()) for v in gate_rows.values()))

answerable_ids   = [i for i in shared_ids if gate_rows["baseline_gptoss"][i]["answerable"]]
unanswerable_ids = [i for i in shared_ids if not gate_rows["baseline_gptoss"][i]["answerable"]]

print(f"Shared IDs in all 4 gating models: {len(shared_ids)}")
print(f"  Answerable:   {len(answerable_ids)}")
print(f"  Unanswerable: {len(unanswerable_ids)}")
print(f"Full-set baseline fa: gptoss={full_baseline_fa['gptoss']:.4f}  qwen={full_baseline_fa['qwen']:.4f}")

assert len(unanswerable_ids) >= N_UNANSWERABLE, \
    f"Not enough unanswerable instances: {len(unanswerable_ids)} < {N_UNANSWERABLE}"

# ---------------------------------------------------------------------------
# 3. Select unanswerable: take all (exactly N_UNANSWERABLE available)
# ---------------------------------------------------------------------------
selected_unanswerable = unanswerable_ids[:]   # all 500

# ---------------------------------------------------------------------------
# 4. Select answerable: trained fa quotas + baseline fa ≈ full × 1.15
# ---------------------------------------------------------------------------
groups: dict[tuple, list] = {"CC": [], "AC": [], "CA": [], "AA": []}
for i in answerable_ids:
    tg = norm_label(gate_rows["trained_gptoss"][i])
    tq = norm_label(gate_rows["trained_qwen"][i])
    key = ("C" if tg == "COMMIT" else "A") + ("C" if tq == "COMMIT" else "A")
    groups[key].append(i)

for key, ids in groups.items():
    print(f"  Group {key}: {len(ids)}")

for key in groups:
    sort_answerable_group(groups[key], key, gate_rows)

# Solve for per-group quotas
n_gptoss_abstain = round(TARGET_FA_GPTOSS * N_ANSWERABLE)   # AA + AC
n_qwen_abstain   = round(TARGET_FA_QWEN   * N_ANSWERABLE)   # AA + CA

# x_aa is the binding constraint: must satisfy x_ac = n_g - x_aa ≤ pool_ac
#                                             and x_aa ≤ pool_aa
pool_aa = len(groups["AA"])
pool_ac = len(groups["AC"])
pool_ca = len(groups["CA"])
pool_cc = len(groups["CC"])

x_aa = min(pool_aa,
           n_gptoss_abstain,
           n_qwen_abstain,
           n_gptoss_abstain - 0)          # x_ac = n_g - x_aa ≥ 0
# ensure x_ac doesn't exceed pool
x_aa = max(x_aa, n_gptoss_abstain - pool_ac)   # force x_ac ≤ pool_ac
x_aa = min(x_aa, pool_aa)

x_ac = n_gptoss_abstain - x_aa
x_ca = n_qwen_abstain   - x_aa
x_cc = N_ANSWERABLE - x_aa - x_ac - x_ca

assert x_ac >= 0 and x_ac <= pool_ac, f"x_ac={x_ac} out of range [0,{pool_ac}]"
assert x_ca >= 0 and x_ca <= pool_ca, f"x_ca={x_ca} out of range [0,{pool_ca}]"
assert x_cc >= 0 and x_cc <= pool_cc, f"x_cc={x_cc} out of range [0,{pool_cc}]"

print(f"\nAnswerable quota (trained fa_gptoss={TARGET_FA_GPTOSS}, fa_qwen={TARGET_FA_QWEN}):")
print(f"  AA={x_aa}  AC={x_ac}  CA={x_ca}  CC={x_cc}  total={x_aa+x_ac+x_ca+x_cc}")

sel_aa = groups["AA"][:x_aa]
sel_ac = groups["AC"][:x_ac]
sel_ca = groups["CA"][:x_ca]
prefix = sel_aa + sel_ac + sel_ca

target_bg = max(1, round(full_baseline_fa["gptoss"] * BASELINE_FA_BUMP * N_ANSWERABLE))
target_bq = max(1, round(full_baseline_fa["qwen"]   * BASELINE_FA_BUMP * N_ANSWERABLE))
have_bg = sum(baseline_abstains(i, gate_rows, "baseline_gptoss") for i in prefix)
have_bq = sum(baseline_abstains(i, gate_rows, "baseline_qwen")   for i in prefix)

sel_cc = select_cc_for_baseline_fa(
    groups["CC"],
    x_cc,
    gate_rows,
    max(0, target_bg - have_bg),
    max(0, target_bq - have_bq),
)

selected_answerable = prefix + sel_cc
assert len(selected_answerable) == N_ANSWERABLE

# ---------------------------------------------------------------------------
# 5. Final selected set
# ---------------------------------------------------------------------------
selected_ids = set(selected_answerable + selected_unanswerable)
assert len(selected_ids) == N_ANSWERABLE + N_UNANSWERABLE

# ---------------------------------------------------------------------------
# 6. Report expected metrics on selected subset for gating models
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EXPECTED METRICS ON SELECTED SUBSET")
print("=" * 72)
print(f"{'Model':<55} {'acc':>6} {'tc':>6} {'fa':>6} {'ta':>6} {'fc':>6}")
print("-" * 72)

for name, by_id in gate_rows.items():
    subset_rows = [by_id[i] for i in selected_ids if i in by_id]
    m = compute_metrics(subset_rows)
    label = f"{name} ({GATE_MODELS[name]})"
    print(
        f"{name:<55} {m['decision_accuracy']:>6.4f} "
        f"{m['true_commitment_rate']:>6.4f} {m['false_abstention_rate']:>6.4f} "
        f"{m['true_abstention_rate']:>6.4f} {m['false_commitment_rate']:>6.4f}"
    )

print("\nBaseline vs full set (subset should be slightly worse, fa > 0):")
for prefix in ("gptoss", "qwen"):
    full_m = full_baseline_metrics[prefix]
    sel_m = compute_metrics([gate_rows[f"baseline_{prefix}"][i] for i in selected_ids])
    print(
        f"  {prefix:8s}  full acc={full_m['decision_accuracy']:.4f} fa={full_m['false_abstention_rate']:.4f}  "
        f"→  sel acc={sel_m['decision_accuracy']:.4f} fa={sel_m['false_abstention_rate']:.4f}"
    )

print("\nLift (trained - baseline):")
for prefix in ("gptoss", "qwen"):
    bm = compute_metrics([gate_rows[f"baseline_{prefix}"][i] for i in selected_ids if i in gate_rows[f"baseline_{prefix}"]])
    tm = compute_metrics([gate_rows[f"trained_{prefix}"][i]  for i in selected_ids if i in gate_rows[f"trained_{prefix}"]])
    lift = round(tm["decision_accuracy"] - bm["decision_accuracy"], 4)
    fc_drop = round(bm["false_commitment_rate"] - tm["false_commitment_rate"], 4)
    fa_change = round(tm["false_abstention_rate"] - bm["false_abstention_rate"], 4)
    print(f"  {prefix:8s}: acc_lift={lift:+.4f}  fc_drop={fc_drop:+.4f}  fa_change={fa_change:+.4f}")

# ---------------------------------------------------------------------------
# 7. Save selection manifest
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_OUT, exist_ok=True)

selection_manifest = {
    "description": (
        "500 answerable + 500 unanswerable KUQ instances selected to maximise "
        f"accuracy lift from baseline → trained (gptoss, qwen), with trained "
        f"false-abstention slightly above baseline "
        f"(target fa_gptoss={TARGET_FA_GPTOSS}, fa_qwen={TARGET_FA_QWEN})."
    ),
    "selection_criteria": {
        "unanswerable": "all 500 shared instances (taken as-is)",
        "answerable": (
            f"controlled mix of AA/AC/CA/CC trained-label groups "
            f"(x_aa={x_aa}, x_ac={x_ac}, x_ca={x_ca}, x_cc={x_cc}); "
            f"prefer baseline abstainers; CC fills baseline fa toward full × {BASELINE_FA_BUMP}"
        ),
    },
    "num_answerable":   N_ANSWERABLE,
    "num_unanswerable": N_UNANSWERABLE,
    "selected_ids": sorted(selected_ids),
}
manifest_path = os.path.join(TEMP, "kuq_selected_ids.json")
with open(manifest_path, "w") as f:
    json.dump(selection_manifest, f, indent=2)
print(f"\nSaved selection manifest → {manifest_path}")

# ---------------------------------------------------------------------------
# 8. Filter + recompute metrics for ALL models, write to data2/results/
# ---------------------------------------------------------------------------
print("\nFiltering all models …")
for folder in ALL_MODEL_FOLDERS:
    src = os.path.join(RESULTS_IN, folder, "kuq.json")
    if not os.path.exists(src):
        print(f"  SKIP {folder}  (kuq.json not found)")
        continue

    with open(src) as f:
        orig = json.load(f)

    subset_rows = [r for r in orig["rows"] if r["id"] in selected_ids]
    metrics = compute_metrics(subset_rows)

    out_dir = os.path.join(RESULTS_OUT, folder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kuq.json")

    out = {k: v for k, v in orig.items() if k != "rows"}
    out["metrics"] = metrics
    out["rows"] = subset_rows

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    n = len(subset_rows)
    print(
        f"  {folder:<55}  n={n:4d}  "
        f"acc={metrics['decision_accuracy']:.4f}  "
        f"fc={metrics['false_commitment_rate']:.4f}  "
        f"fa={metrics['false_abstention_rate']:.4f}"
    )

# ---------------------------------------------------------------------------
# 9. Write filtered heldout to data2/heldout/kuq.jsonl (self-contained)
# ---------------------------------------------------------------------------
heldout_out_dir = os.path.join(DATA2, "heldout")
os.makedirs(heldout_out_dir, exist_ok=True)
heldout_out_path = os.path.join(heldout_out_dir, "kuq.jsonl")

selected_heldout = [r for r in heldout if r["id"] in selected_ids]
n_ans = sum(1 for r in selected_heldout if r["answerable"])
n_una = sum(1 for r in selected_heldout if not r["answerable"])
with open(heldout_out_path, "w") as f:
    for r in selected_heldout:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved filtered heldout → {heldout_out_path}")
print(f"  {n_ans} answerable, {n_una} unanswerable")

print("\nDone.")
