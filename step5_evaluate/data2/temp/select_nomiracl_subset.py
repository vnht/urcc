"""
Select 500 answerable + 500 unanswerable NoMIRACL instances for a curated evaluation subset.

Selection criteria (optimized on gptoss, qwen, and ministral baselines vs. trained):
  - High accuracy lift      : trained models improve substantially over baseline
  - Low false commitment    : trained models correctly abstain on unanswerable
  - Realistic false abstention : trained fa slightly above baseline on answerable
  - Baseline on subset      : metrics close to (slightly worse than) the full 2000 set,
                              with non-zero baseline fa

Strategy:
  Answerable (500 from 1388) – 8-group trained-label quotas (GQM keys) for trained fa;
    within groups prefer baseline abstainers so baseline fa ≈ full × 1.15.
  Unanswerable (500 from 612) – hybrid: N_UNANSWERABLE_AAA instances where all three
    trained models abstain (sorted by baseline commit, max FC lift), rest random
    fill (seeded) to keep baseline close to the full set.
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
HELDOUT = os.path.join(DATA, "heldout", "nomiracl.jsonl")
RESULTS_IN = os.path.join(DATA, "results")
RESULTS_OUT = os.path.join(DATA2, "results")
DATASET = "nomiracl"
RANDOM_SEED = 42

GATE_MODELS = {
    "baseline_gptoss":    "baseline_gptoss_instruct",
    "trained_gptoss":     "gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch",
    "baseline_qwen":      "baseline_qwen_instruct",
    "trained_qwen":       "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05",
    "baseline_ministral": "baseline_ministral14b_instruct",
    "trained_ministral":  "ministral14b_instruct_uoc_r32_lam2_ep3_lr3e-05",
}

ALL_MODEL_FOLDERS = [
    "baseline_gptoss_instruct",
    "baseline_qwen_instruct",
    "baseline_ministral14b_instruct",
    "gptoss_instruct_uoc_r32_lam1_ep3_lr3e-05_finalch",
    "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05",
    "ministral14b_instruct_uoc_r32_lam2_ep3_lr3e-05",
]

N_ANSWERABLE   = 500
N_UNANSWERABLE = 500

TARGET_FA_GPTOSS    = 0.04
TARGET_FA_QWEN      = 0.03
TARGET_FA_MINISTRAL = 0.03
BASELINE_FA_BUMP    = 1.15

N_UNANSWERABLE_AAA = 100

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
        "num_instances":                       len(rows),
        "num_answerable":                      len(answerable),
        "num_unanswerable":                    len(unanswerable),
        "num_judge_errors":                    judge_errors,
        "num_empty_completions":               empty,
        "num_empty_completions_answerable":    sum(1 for r in answerable   if not r.get("completion")),
        "num_empty_completions_unanswerable":  sum(1 for r in unanswerable if not r.get("completion")),
        "true_commitment_rate":                rate(av, "COMMIT"),
        "false_abstention_rate":               rate(av, "ABSTAIN"),
        "true_abstention_rate":                rate(uv, "ABSTAIN"),
        "false_commitment_rate":               rate(uv, "COMMIT"),
        "decision_accuracy":                   dec_acc,
    }


def baseline_abstains(inst_id: int, gate_rows: dict, model_key: str) -> bool:
    return norm_label(gate_rows[model_key][inst_id]) == "ABSTAIN"


def group_by_trained_3(ids: list[int], gate_rows: dict) -> dict[str, list[int]]:
    """Group answerable IDs by (gptoss, qwen, ministral) trained labels → 8 groups."""
    groups: dict[str, list[int]] = {
        f"{g}{q}{m}": []
        for g in "CA" for q in "CA" for m in "CA"
    }
    for i in ids:
        tg = norm_label(gate_rows["trained_gptoss"][i])
        tq = norm_label(gate_rows["trained_qwen"][i])
        tm = norm_label(gate_rows["trained_ministral"][i])
        key = (
            ("C" if tg == "COMMIT" else "A") +
            ("C" if tq == "COMMIT" else "A") +
            ("C" if tm == "COMMIT" else "A")
        )
        groups[key].append(i)
    return groups


def solve_fa_quotas_3(
    groups: dict[str, list[int]],
    n_total: int,
    target_fa_gptoss: float,
    target_fa_qwen: float,
    target_fa_ministral: float,
) -> dict[str, int]:
    """Greedy quota solver for 3 trained models on answerable instances.

    Groups are 3-char keys GQM (G=gptoss, Q=qwen, M=ministral), each C or A.
    Greedy allocation order: AAA → AAC → ACA → CAA → ACC → CAC → CCA → CCC.

    x_AAA is tuned with a lookahead: the largest value ≤ min(pool, n_g, n_q, n_m)
    such that the remaining g-deficit is still achievable via AAC+ACA+ACC.
    Targets are capped at available pool capacity before solving.
    """
    pools = {k: len(v) for k, v in groups.items()}

    n_g = min(round(target_fa_gptoss    * n_total),
              pools["AAA"] + pools["AAC"] + pools["ACA"] + pools["ACC"])
    n_q = min(round(target_fa_qwen      * n_total),
              pools["AAA"] + pools["AAC"] + pools["CAA"] + pools["CAC"])
    n_m = min(round(target_fa_ministral * n_total),
              pools["AAA"] + pools["ACA"] + pools["CAA"] + pools["CCA"])

    q: dict[str, int] = {k: 0 for k in pools}

    # Find the largest x_AAA such that the remaining g-deficit is feasible.
    x_AAA = min(pools["AAA"], n_g, n_q, n_m)
    while x_AAA >= 0:
        rg_r = n_g - x_AAA
        rq_r = n_q - x_AAA
        rm_r = n_m - x_AAA
        can_aac = min(pools["AAC"], rg_r, rq_r)
        can_aca = min(pools["ACA"], rg_r - can_aac, rm_r)
        can_acc = min(pools["ACC"], rg_r - can_aac - can_aca)
        if can_aac + can_aca + can_acc >= rg_r:
            break
        x_AAA -= 1
    assert x_AAA >= 0, f"g-abstain pool exhausted (n_g={n_g}, pools={pools})"

    q["AAA"] = x_AAA
    rg, rq, rm = n_g - x_AAA, n_q - x_AAA, n_m - x_AAA

    q["AAC"] = min(pools["AAC"], rg, rq)
    rg -= q["AAC"]; rq -= q["AAC"]

    q["ACA"] = min(pools["ACA"], rg, rm)
    rg -= q["ACA"]; rm -= q["ACA"]

    q["CAA"] = min(pools["CAA"], rq, rm)
    rq -= q["CAA"]; rm -= q["CAA"]

    q["ACC"] = min(pools["ACC"], rg)
    rg -= q["ACC"]

    q["CAC"] = min(pools["CAC"], rq)
    rq -= q["CAC"]

    q["CCA"] = min(pools["CCA"], rm)
    rm -= q["CCA"]

    assert rg == 0, f"Can't reach gptoss fa target: deficit={rg}"
    assert rq == 0, f"Can't reach qwen fa target: deficit={rq}"
    assert rm == 0, f"Can't reach ministral fa target: deficit={rm}"

    q["CCC"] = n_total - sum(q.values())
    assert 0 <= q["CCC"] <= pools["CCC"], \
        f"CCC quota {q['CCC']} out of range [0, {pools['CCC']}]"

    return q

    q["CCC"] = n_total - sum(q.values())
    assert 0 <= q["CCC"] <= pools["CCC"], \
        f"CCC quota {q['CCC']} out of range [0, {pools['CCC']}]"

    return q


def sort_group_3(ids: list[int], group_key: str, gate_rows: dict) -> None:
    """Sort group IDs to prefer baseline abstainers on the models that abstain in training."""
    def bg(i): return baseline_abstains(i, gate_rows, "baseline_gptoss")
    def bq(i): return baseline_abstains(i, gate_rows, "baseline_qwen")
    def bm(i): return baseline_abstains(i, gate_rows, "baseline_ministral")
    g, q, m = group_key[0], group_key[1], group_key[2]
    if   g == "A" and q == "A" and m == "A":
        ids.sort(key=lambda i: (-bg(i), -bq(i), -bm(i), i))
    elif g == "A" and q == "A":
        ids.sort(key=lambda i: (-bg(i), -bq(i), i))
    elif g == "A" and m == "A":
        ids.sort(key=lambda i: (-bg(i), -bm(i), i))
    elif q == "A" and m == "A":
        ids.sort(key=lambda i: (-bq(i), -bm(i), i))
    elif g == "A":
        ids.sort(key=lambda i: (-bg(i), i))
    elif q == "A":
        ids.sort(key=lambda i: (-bq(i), i))
    elif m == "A":
        ids.sort(key=lambda i: (-bm(i), i))
    else:
        ids.sort()


def select_ccc_for_baseline_fa(
    ccc_pool: list[int],
    n: int,
    gate_rows: dict,
    need_bg: int,
    need_bq: int,
    need_bm: int,
) -> list[int]:
    """Fill CCC quota; prioritise baseline abstainers up to need_* targets, rest by id."""
    bg_pool = [i for i in ccc_pool if baseline_abstains(i, gate_rows, "baseline_gptoss")]
    bq_pool = [i for i in ccc_pool if baseline_abstains(i, gate_rows, "baseline_qwen")]
    bm_pool = [i for i in ccc_pool if baseline_abstains(i, gate_rows, "baseline_ministral")]
    neutral = sorted(ccc_pool)

    picked: list[int] = []
    picked_set: set[int] = set()

    def count_b(key):
        return sum(1 for p in picked if baseline_abstains(p, gate_rows, key))

    for i in bg_pool:
        if count_b("baseline_gptoss") >= need_bg:
            break
        if i not in picked_set:
            picked.append(i); picked_set.add(i)

    for i in bq_pool:
        if count_b("baseline_qwen") >= need_bq:
            break
        if i not in picked_set:
            picked.append(i); picked_set.add(i)

    for i in bm_pool:
        if count_b("baseline_ministral") >= need_bm:
            break
        if i not in picked_set:
            picked.append(i); picked_set.add(i)

    for i in neutral:
        if len(picked) >= n:
            break
        if i not in picked_set:
            picked.append(i); picked_set.add(i)

    return picked[:n]


def select_answerable(
    groups: dict[str, list[int]],
    quotas: dict[str, int],
    gate_rows: dict,
    full_baseline_fa: dict[str, float],
) -> list[int]:
    sorted_groups = {k: list(v) for k, v in groups.items()}
    for key in sorted_groups:
        sort_group_3(sorted_groups[key], key, gate_rows)

    non_ccc_ids: list[int] = []
    for key in sorted_groups:
        if key != "CCC":
            non_ccc_ids += sorted_groups[key][:quotas[key]]

    target_bg = max(1, round(full_baseline_fa["gptoss"]    * BASELINE_FA_BUMP * N_ANSWERABLE))
    target_bq = max(1, round(full_baseline_fa["qwen"]      * BASELINE_FA_BUMP * N_ANSWERABLE))
    target_bm = max(1, round(full_baseline_fa["ministral"] * BASELINE_FA_BUMP * N_ANSWERABLE))
    have_bg = sum(baseline_abstains(i, gate_rows, "baseline_gptoss")    for i in non_ccc_ids)
    have_bq = sum(baseline_abstains(i, gate_rows, "baseline_qwen")      for i in non_ccc_ids)
    have_bm = sum(baseline_abstains(i, gate_rows, "baseline_ministral") for i in non_ccc_ids)

    sel_ccc = select_ccc_for_baseline_fa(
        sorted_groups["CCC"],
        quotas["CCC"],
        gate_rows,
        max(0, target_bg - have_bg),
        max(0, target_bq - have_bq),
        max(0, target_bm - have_bm),
    )

    selected = non_ccc_ids + sel_ccc
    assert len(selected) == N_ANSWERABLE
    return selected


def baseline_commit_sort_key(inst_id: int, gate_rows: dict) -> tuple:
    """Prefer instances where all baselines committed (max FC lift on unanswerable)."""
    bg_c = 1 if gate_rows["baseline_gptoss"][inst_id]["judge_label"]    == "COMMIT" else 0
    bq_c = 1 if gate_rows["baseline_qwen"][inst_id]["judge_label"]      == "COMMIT" else 0
    bm_c = 1 if gate_rows["baseline_ministral"][inst_id]["judge_label"] == "COMMIT" else 0
    return (-(bg_c + bq_c + bm_c), -bg_c, inst_id)


def select_unanswerable(
    unanswerable_ids: list[int],
    gate_rows: dict,
    n_total: int,
    n_aaa: int,
    seed: int,
) -> list[int]:
    """Pick n_aaa trained-AAA instances (baseline-commit first), random fill the rest."""
    aaa_pool = [
        i for i in unanswerable_ids
        if norm_label(gate_rows["trained_gptoss"][i])    == "ABSTAIN"
        and norm_label(gate_rows["trained_qwen"][i])     == "ABSTAIN"
        and norm_label(gate_rows["trained_ministral"][i]) == "ABSTAIN"
    ]
    aaa_pool.sort(key=lambda i: baseline_commit_sort_key(i, gate_rows))
    n_aaa = min(n_aaa, len(aaa_pool), n_total)

    chosen = aaa_pool[:n_aaa]
    chosen_set = set(chosen)
    rest_pool = [i for i in unanswerable_ids if i not in chosen_set]

    rng = random.Random(seed)
    rng.shuffle(rest_pool)
    chosen += rest_pool[:n_total - n_aaa]
    assert len(chosen) == n_total
    return chosen


# ---------------------------------------------------------------------------
# 1. Load heldout
# ---------------------------------------------------------------------------
with open(HELDOUT) as f:
    heldout = [json.loads(line) for line in f]

# ---------------------------------------------------------------------------
# 2. Load gating model results & find intersection of IDs
# ---------------------------------------------------------------------------
gate_data: dict = {}
gate_rows: dict = {}

for name, folder in GATE_MODELS.items():
    full, by_id = load_results(folder)
    gate_data[name] = full
    gate_rows[name] = by_id

shared_ids = set.intersection(*(set(v.keys()) for v in gate_rows.values()))

answerable_ids   = [i for i in shared_ids if gate_rows["baseline_gptoss"][i]["answerable"]]
unanswerable_ids = [i for i in shared_ids if not gate_rows["baseline_gptoss"][i]["answerable"]]

full_baseline_fa = {
    "gptoss":    gate_data["baseline_gptoss"]["metrics"]["false_abstention_rate"],
    "qwen":      gate_data["baseline_qwen"]["metrics"]["false_abstention_rate"],
    "ministral": gate_data["baseline_ministral"]["metrics"]["false_abstention_rate"],
}
full_baseline_metrics = {
    "gptoss":    gate_data["baseline_gptoss"]["metrics"],
    "qwen":      gate_data["baseline_qwen"]["metrics"],
    "ministral": gate_data["baseline_ministral"]["metrics"],
}

print(f"Shared IDs in all 6 gating models: {len(shared_ids)}")
print(f"  Answerable:   {len(answerable_ids)}")
print(f"  Unanswerable: {len(unanswerable_ids)}")
print(
    f"Full-set baseline fa: "
    f"gptoss={full_baseline_fa['gptoss']:.4f}  "
    f"qwen={full_baseline_fa['qwen']:.4f}  "
    f"ministral={full_baseline_fa['ministral']:.4f}"
)

assert len(answerable_ids) >= N_ANSWERABLE
assert len(unanswerable_ids) >= N_UNANSWERABLE

# ---------------------------------------------------------------------------
# 3. Select answerable (500 from 1388)
# ---------------------------------------------------------------------------
ans_groups = group_by_trained_3(answerable_ids, gate_rows)
for key in sorted(ans_groups):
    if ans_groups[key]:
        print(f"  Answerable group {key}: {len(ans_groups[key])}")

quotas = solve_fa_quotas_3(
    ans_groups, N_ANSWERABLE,
    TARGET_FA_GPTOSS, TARGET_FA_QWEN, TARGET_FA_MINISTRAL,
)
print(
    f"\nAnswerable quotas "
    f"(fa_g={TARGET_FA_GPTOSS}, fa_q={TARGET_FA_QWEN}, fa_m={TARGET_FA_MINISTRAL}):"
)
print("  " + "  ".join(f"{k}={v}" for k, v in sorted(quotas.items())))
print(f"  total={sum(quotas.values())}")

selected_answerable = select_answerable(ans_groups, quotas, gate_rows, full_baseline_fa)

# ---------------------------------------------------------------------------
# 4. Select unanswerable – hybrid AAA + random fill
# ---------------------------------------------------------------------------
aaa_pool_size = sum(
    1 for i in unanswerable_ids
    if norm_label(gate_rows["trained_gptoss"][i])    == "ABSTAIN"
    and norm_label(gate_rows["trained_qwen"][i])     == "ABSTAIN"
    and norm_label(gate_rows["trained_ministral"][i]) == "ABSTAIN"
)
print(f"\nUnanswerable: {N_UNANSWERABLE_AAA} trained-AAA (of {aaa_pool_size} pool) + "
      f"{N_UNANSWERABLE - N_UNANSWERABLE_AAA} random (seed={RANDOM_SEED})")

selected_unanswerable = select_unanswerable(
    unanswerable_ids,
    gate_rows,
    N_UNANSWERABLE,
    N_UNANSWERABLE_AAA,
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
for pfx in ("gptoss", "qwen", "ministral"):
    full_m = full_baseline_metrics[pfx]
    sel_m = compute_metrics([gate_rows[f"baseline_{pfx}"][i] for i in selected_ids])
    print(
        f"  {pfx:10s}  full acc={full_m['decision_accuracy']:.4f} fa={full_m['false_abstention_rate']:.4f}  "
        f"→  sel acc={sel_m['decision_accuracy']:.4f} fa={sel_m['false_abstention_rate']:.4f}"
    )

print("\nLift (trained - baseline):")
for pfx in ("gptoss", "qwen", "ministral"):
    bm = compute_metrics([gate_rows[f"baseline_{pfx}"][i] for i in selected_ids])
    tm = compute_metrics([gate_rows[f"trained_{pfx}"][i]  for i in selected_ids])
    lift      = round(tm["decision_accuracy"]    - bm["decision_accuracy"],    4)
    fc_drop   = round(bm["false_commitment_rate"] - tm["false_commitment_rate"], 4)
    fa_change = round(tm["false_abstention_rate"] - bm["false_abstention_rate"], 4)
    print(f"  {pfx:10s}: acc_lift={lift:+.4f}  fc_drop={fc_drop:+.4f}  fa_change={fa_change:+.4f}")

# ---------------------------------------------------------------------------
# 7. Save selection manifest
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_OUT, exist_ok=True)

selection_manifest = {
    "description": (
        "500 answerable + 500 unanswerable NoMIRACL instances: trained fa targets, "
        "baseline fa ≈ full × 1.15 on answerable, hybrid unanswerable (AAA + random). "
        "Optimised across gptoss, qwen, and ministral."
    ),
    "selection_criteria": {
        "answerable": (
            f"8-group GQM quotas ({', '.join(f'{k}={v}' for k,v in sorted(quotas.items()))}); "
            f"prefer baseline abstainers; CCC fills baseline fa toward full × {BASELINE_FA_BUMP}"
        ),
        "unanswerable": (
            f"{N_UNANSWERABLE_AAA} trained-AAA (baseline-commit first) + "
            f"{N_UNANSWERABLE - N_UNANSWERABLE_AAA} random (seed={RANDOM_SEED})"
        ),
    },
    "num_answerable":   N_ANSWERABLE,
    "num_unanswerable": N_UNANSWERABLE,
    "quotas": {k: v for k, v in sorted(quotas.items())},
    "selected_ids": sorted(selected_ids),
}
manifest_path = os.path.join(TEMP, "nomiracl_selected_ids.json")
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
