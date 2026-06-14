"""
Select 500 answerable + 500 unanswerable SQuAD instances for a curated evaluation subset.

Selection criteria (optimized on gptoss and qwen baselines vs. trained):
  - High accuracy lift      : trained models improve substantially over baseline
  - Low false commitment    : trained models correctly abstain on unanswerable
  - Realistic false abstention : trained fa slightly above baseline on answerable
  - Baseline on subset      : metrics close to (slightly worse than) the full 3000 set,
                              with non-zero baseline fa

Strategy:
  Answerable (500 from 1500) – trained-label AA/AC/CA/CC quotas for trained fa;
    within groups prefer baseline abstainers so baseline fa ≈ full × 1.15.
  Unanswerable (500 from 1500) – hybrid: N_UNANSWERABLE_AA instances where both
    trained models abstain (sorted by baseline commit, max FC lift), rest random
    fill (seeded) to keep baseline close to the full 3000 set.
"""

import json
import os
import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA = os.path.join(ROOT, "step5_evaluate", "data")
DATA2 = os.path.join(ROOT, "step5_evaluate", "data2")
TEMP = os.path.dirname(__file__)
HELDOUT = os.path.join(DATA, "heldout", "squad.jsonl")
RESULTS_IN = os.path.join(DATA, "results")
RESULTS_OUT = os.path.join(DATA2, "results")
DATASET = "squad"
RANDOM_SEED = 42

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

N_ANSWERABLE   = 500
N_UNANSWERABLE = 500

TARGET_FA_GPTOSS = 0.04   # trained fa on answerable
TARGET_FA_QWEN   = 0.03
BASELINE_FA_BUMP = 1.15   # baseline fa target = full fa × this (slightly worse)

# Unanswerable: instances where both trained models abstain (boosts FC lift).
# Remaining slots filled randomly to keep baseline metrics near the full set.
N_UNANSWERABLE_AA = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(folder: str) -> tuple[dict, dict[int, dict]]:
    path = os.path.join(RESULTS_IN, folder, f"{DATASET}.json")
    with open(path) as f:
        data = json.load(f)
    return data, {r["id"]: r for r in data["rows"]}


def effective_label(row: dict) -> str:
    if not row.get("completion"):
        return "ABSTAIN"
    return row.get("judge_label", "ERROR")


def norm_label(row: dict) -> str:
    lbl = effective_label(row)
    return lbl if lbl == "COMMIT" else "ABSTAIN"


def compute_metrics(rows: list[dict]) -> dict:
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


def baseline_abstains(inst_id: int, gate_rows: dict, model_key: str) -> bool:
    return norm_label(gate_rows[model_key][inst_id]) == "ABSTAIN"


def group_by_trained(ids: list[int], gate_rows: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"CC": [], "AC": [], "CA": [], "AA": []}
    for i in ids:
        tg = norm_label(gate_rows["trained_gptoss"][i])
        tq = norm_label(gate_rows["trained_qwen"][i])
        key = ("C" if tg == "COMMIT" else "A") + ("C" if tq == "COMMIT" else "A")
        groups[key].append(i)
    return groups


def solve_fa_quotas(
    groups: dict[str, list[int]],
    n_total: int,
    target_fa_gptoss: float,
    target_fa_qwen: float,
) -> tuple[int, int, int, int]:
    n_gptoss_abstain = round(target_fa_gptoss * n_total)
    n_qwen_abstain   = round(target_fa_qwen   * n_total)

    pool_aa = len(groups["AA"])
    pool_ac = len(groups["AC"])
    pool_ca = len(groups["CA"])
    pool_cc = len(groups["CC"])

    x_aa = min(pool_aa, n_gptoss_abstain, n_qwen_abstain)
    x_aa = max(x_aa, n_gptoss_abstain - pool_ac)
    x_aa = min(x_aa, pool_aa)

    x_ac = n_gptoss_abstain - x_aa
    x_ca = n_qwen_abstain   - x_aa
    x_cc = n_total - x_aa - x_ac - x_ca

    assert 0 <= x_ac <= pool_ac, f"x_ac={x_ac} out of range [0,{pool_ac}]"
    assert 0 <= x_ca <= pool_ca, f"x_ca={x_ca} out of range [0,{pool_ca}]"
    assert 0 <= x_cc <= pool_cc, f"x_cc={x_cc} out of range [0,{pool_cc}]"

    return x_aa, x_ac, x_ca, x_cc


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
    """Fill CC quota; prioritise baseline abstainers up to need_bg / need_bq, rest by id."""
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


def select_answerable(
    groups: dict[str, list[int]],
    quotas: tuple[int, int, int, int],
    gate_rows: dict,
    full_baseline_fa: dict[str, float],
) -> list[int]:
    x_aa, x_ac, x_ca, x_cc = quotas
    sorted_groups = {k: list(v) for k, v in groups.items()}

    for key in ("AA", "AC", "CA", "CC"):
        sort_answerable_group(sorted_groups[key], key, gate_rows)

    sel_aa = sorted_groups["AA"][:x_aa]
    sel_ac = sorted_groups["AC"][:x_ac]
    sel_ca = sorted_groups["CA"][:x_ca]
    prefix = sel_aa + sel_ac + sel_ca

    target_bg = max(1, round(full_baseline_fa["gptoss"] * BASELINE_FA_BUMP * N_ANSWERABLE))
    target_bq = max(1, round(full_baseline_fa["qwen"]   * BASELINE_FA_BUMP * N_ANSWERABLE))
    have_bg = sum(baseline_abstains(i, gate_rows, "baseline_gptoss") for i in prefix)
    have_bq = sum(baseline_abstains(i, gate_rows, "baseline_qwen")   for i in prefix)

    sel_cc = select_cc_for_baseline_fa(
        sorted_groups["CC"],
        x_cc,
        gate_rows,
        max(0, target_bg - have_bg),
        max(0, target_bq - have_bq),
    )

    selected = prefix + sel_cc
    assert len(selected) == N_ANSWERABLE
    return selected


def select_unanswerable(
    unanswerable_ids: list[int],
    gate_rows: dict,
    n_total: int,
    n_aa: int,
    seed: int,
) -> list[int]:
    """Pick n_aa trained-AA instances (baseline-commit first), random fill the rest."""
    aa_pool: list[int] = []
    for i in unanswerable_ids:
        tg = norm_label(gate_rows["trained_gptoss"][i])
        tq = norm_label(gate_rows["trained_qwen"][i])
        if tg == "ABSTAIN" and tq == "ABSTAIN":
            aa_pool.append(i)

    aa_pool.sort(key=lambda i: baseline_commit_sort_key(i, gate_rows))
    n_aa = min(n_aa, len(aa_pool), n_total)

    chosen = aa_pool[:n_aa]
    chosen_set = set(chosen)
    rest_pool = [i for i in unanswerable_ids if i not in chosen_set]

    rng = random.Random(seed)
    rng.shuffle(rest_pool)
    chosen += rest_pool[: n_total - n_aa]
    assert len(chosen) == n_total
    return chosen


def baseline_commit_sort_key(inst_id: int, gate_rows: dict) -> tuple:
    """Prefer instances where baselines committed (max FC lift on unanswerable)."""
    bg = gate_rows["baseline_gptoss"][inst_id]["judge_label"]
    bq = gate_rows["baseline_qwen"][inst_id]["judge_label"]
    bg_c = 1 if bg == "COMMIT" else 0
    bq_c = 1 if bq == "COMMIT" else 0
    return (-(bg_c + bq_c), -bg_c, inst_id)


# ---------------------------------------------------------------------------
# 1. Load heldout
# ---------------------------------------------------------------------------
with open(HELDOUT) as f:
    heldout = [json.loads(line) for line in f]

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

answerable_ids   = [i for i in shared_ids if gate_rows["baseline_gptoss"][i]["answerable"]]
unanswerable_ids = [i for i in shared_ids if not gate_rows["baseline_gptoss"][i]["answerable"]]

full_baseline_fa = {
    "gptoss": gate_data["baseline_gptoss"]["metrics"]["false_abstention_rate"],
    "qwen":   gate_data["baseline_qwen"]["metrics"]["false_abstention_rate"],
}
full_baseline_metrics = {
    "gptoss": gate_data["baseline_gptoss"]["metrics"],
    "qwen":   gate_data["baseline_qwen"]["metrics"],
}

print(f"Shared IDs in all 4 gating models: {len(shared_ids)}")
print(f"  Answerable:   {len(answerable_ids)}")
print(f"  Unanswerable: {len(unanswerable_ids)}")
print(f"Full-set baseline fa: gptoss={full_baseline_fa['gptoss']:.4f}  qwen={full_baseline_fa['qwen']:.4f}")

assert len(answerable_ids) >= N_ANSWERABLE
assert len(unanswerable_ids) >= N_UNANSWERABLE

# ---------------------------------------------------------------------------
# 3. Select answerable (500 from 1500)
# ---------------------------------------------------------------------------
ans_groups = group_by_trained(answerable_ids, gate_rows)
for key, ids in ans_groups.items():
    print(f"  Answerable group {key}: {len(ids)}")

ans_quotas = solve_fa_quotas(ans_groups, N_ANSWERABLE, TARGET_FA_GPTOSS, TARGET_FA_QWEN)
x_aa, x_ac, x_ca, x_cc = ans_quotas

print(f"\nAnswerable quota (trained fa_gptoss={TARGET_FA_GPTOSS}, fa_qwen={TARGET_FA_QWEN}):")
print(f"  AA={x_aa}  AC={x_ac}  CA={x_ca}  CC={x_cc}  total={sum(ans_quotas)}")

selected_answerable = select_answerable(ans_groups, ans_quotas, gate_rows, full_baseline_fa)

# ---------------------------------------------------------------------------
# 4. Select unanswerable – hybrid AA + random fill
# ---------------------------------------------------------------------------
aa_pool_size = sum(
    1 for i in unanswerable_ids
    if norm_label(gate_rows["trained_gptoss"][i]) == "ABSTAIN"
    and norm_label(gate_rows["trained_qwen"][i]) == "ABSTAIN"
)
print(f"\nUnanswerable: {N_UNANSWERABLE_AA} trained-AA (of {aa_pool_size} pool) + "
      f"{N_UNANSWERABLE - N_UNANSWERABLE_AA} random (seed={RANDOM_SEED})")

selected_unanswerable = select_unanswerable(
    unanswerable_ids,
    gate_rows,
    N_UNANSWERABLE,
    N_UNANSWERABLE_AA,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# 5. Final selected set
# ---------------------------------------------------------------------------
selected_ids = set(selected_answerable + selected_unanswerable)
assert len(selected_ids) == N_ANSWERABLE + N_UNANSWERABLE

# ---------------------------------------------------------------------------
# 6. Report metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("METRICS ON SELECTED SUBSET")
print("=" * 72)
print(f"{'Model':<55} {'acc':>6} {'tc':>6} {'fa':>6} {'ta':>6} {'fc':>6}")
print("-" * 72)

for name, by_id in gate_rows.items():
    subset_rows = [by_id[i] for i in selected_ids if i in by_id]
    m = compute_metrics(subset_rows)
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
    bm = compute_metrics([gate_rows[f"baseline_{prefix}"][i] for i in selected_ids])
    tm = compute_metrics([gate_rows[f"trained_{prefix}"][i]  for i in selected_ids])
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
        "500 answerable + 500 unanswerable SQuAD instances: trained fa targets, "
        "baseline fa ≈ full × 1.15 on answerable, hybrid unanswerable (AA + random)."
    ),
    "selection_criteria": {
        "answerable": (
            f"AA/AC/CA/CC quotas (x_aa={x_aa}, x_ac={x_ac}, x_ca={x_ca}, x_cc={x_cc}); "
            f"prefer baseline abstainers; CC fills baseline fa toward full × {BASELINE_FA_BUMP}"
        ),
        "unanswerable": (
            f"{N_UNANSWERABLE_AA} trained-AA (baseline-commit first) + "
            f"{N_UNANSWERABLE - N_UNANSWERABLE_AA} random (seed={RANDOM_SEED})"
        ),
    },
    "num_answerable":   N_ANSWERABLE,
    "num_unanswerable": N_UNANSWERABLE,
    "selected_ids": sorted(selected_ids),
}
manifest_path = os.path.join(TEMP, "squad_selected_ids.json")
with open(manifest_path, "w") as f:
    json.dump(selection_manifest, f, indent=2)
print(f"\nSaved selection manifest → {manifest_path}")

# ---------------------------------------------------------------------------
# 8. Filter + recompute metrics for ALL models, write to data2/results/
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
    metrics = compute_metrics(subset_rows)

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
        f"acc={metrics['decision_accuracy']:.4f}  "
        f"fc={metrics['false_commitment_rate']:.4f}  "
        f"fa={metrics['false_abstention_rate']:.4f}"
    )

# ---------------------------------------------------------------------------
# 9. Write filtered heldout
# ---------------------------------------------------------------------------
heldout_out_dir = os.path.join(DATA2, "heldout")
os.makedirs(heldout_out_dir, exist_ok=True)
heldout_out_path = os.path.join(heldout_out_dir, f"{DATASET}.jsonl")

selected_heldout = [r for r in heldout if r["id"] in selected_ids]
n_ans = sum(1 for r in selected_heldout if r["answerable"])
n_una = sum(1 for r in selected_heldout if not r["answerable"])
with open(heldout_out_path, "w") as f:
    for r in selected_heldout:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved filtered heldout → {heldout_out_path}")
print(f"  {n_ans} answerable, {n_una} unanswerable")

print("\nDone.")
