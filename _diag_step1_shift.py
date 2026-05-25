"""Quick diagnostic: compare new (transition-shifted) step 1 bundle
against the recorded numbers from the old (unshifted) bundle.

Reports:
  1. Per-set norm statistics (compare to old: A=117.55, B=111.17, C=118.08,
     D=105.75, E=108.57)
  2. Contrast geometry: ‖c_OC‖, ‖c_LC‖, cos(mean c_OC, mean c_LC)
     (old: 208.4, 213.0, +0.842)
  3. Per-layer covariance traces tr Σ_OC, tr Σ_LC, tr Σ_E
  4. Per-domain contrast separation: ‖c_OC[kuq]‖ vs ‖c_OC[squad]‖
  5. Sanity: A vs B mean separation per layer (broad anchor signal)

If the contrast geometry is preserved (cos < 1, ‖c‖ comparable), the V
subspace will still be discriminative. If norms collapse or cos→1, the
shift broke the signal.
"""
from __future__ import annotations
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "step1_extract_activations" / "data" / "activations_qwen_instruct.pt"

bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
print(f"keys: {sorted(bundle.keys())}\n")

# ── (1) per-set norms ─────────────────────────────────────────────────────────
print("=" * 78)
print("(1) Per-set norm statistics  [compare old / new]")
print("=" * 78)
print(f"{'Set':<3} {'shape':<22} {'mean ‖h‖':>10} {'std':>8}   old")
old = {"A": (117.55, 27.5), "B": (111.17, 24.0), "C": (118.08, 29.8),
       "D": (105.75, 23.2), "E": (108.57, 24.6)}
for tag in ["A", "B", "C", "D", "E"]:
    key = {"A": "h_A", "B": "h_B", "C": "h_C", "D": "h_D", "E": "h_E"}[tag]
    if key not in bundle:
        # fallback names
        candidates = [k for k in bundle if k.lower().endswith(tag.lower()) or k.endswith(f"_{tag.lower()}") or tag.lower() in k.lower()]
        print(f"  {tag}: missing key (candidates: {candidates})")
        continue
    H = bundle[key]
    if isinstance(H, dict):
        H = H["means"]
    norms = H.flatten(1).norm(dim=1) / (H.shape[1] ** 0.5)  # per-row mean over layers
    # Match the old metric: per-row ‖h‖ averaged across layers via flatten/sqrt(L)
    print(f"  {tag} {str(tuple(H.shape)):<22} {norms.mean().item():>10.2f} {norms.std().item():>8.2f}   "
          f"old: {old[tag][0]:.2f} ± {old[tag][1]:.1f}")

# ── (2) contrast geometry ────────────────────────────────────────────────────
print()
print("=" * 78)
print("(2) Contrast geometry  [compare: ‖c_OC‖=208.4, ‖c_LC‖=213.0, cos=+0.842]")
print("=" * 78)
def get(key: str) -> torch.Tensor:
    h = bundle[key]
    return h["means"] if isinstance(h, dict) else h

A, B, C_, D = get("h_A"), get("h_B"), get("h_C"), get("h_D")
# (N, L, D)
c_oc = (A - B)
c_lc = (C_ - D)
def per_row_norm(t):  # (N, L, D)
    return t.flatten(1).norm(dim=1)
print(f"  per-row mean ‖c_OC‖ = {per_row_norm(c_oc).mean().item():.2f}   (old 208.4)")
print(f"  per-row mean ‖c_LC‖ = {per_row_norm(c_lc).mean().item():.2f}   (old 213.0)")
mean_oc = c_oc.mean(0).flatten()
mean_lc = c_lc.mean(0).flatten()
cos = (mean_oc @ mean_lc) / (mean_oc.norm() * mean_lc.norm())
print(f"  cos(mean c_OC, mean c_LC) = {cos.item():+.4f}   (old +0.8420)")

# ── (3) per-layer trace stats ────────────────────────────────────────────────
print()
print("=" * 78)
print("(3) Per-layer covariance traces  [old in interpretation/qwen_instruct.md]")
print("=" * 78)
E = get("h_E")
def trace_cov(X):  # X: (N, L, D) → per-layer trace
    Xc = X - X.mean(0, keepdim=True)
    # tr Σ = (1/N) Σ_n ‖x_n - μ‖²
    return Xc.pow(2).sum(-1).mean(0)  # (L,)

t_oc = trace_cov(c_oc); t_lc = trace_cov(c_lc); t_e = trace_cov(E)
old_oc = [3091, 3592, 3891, 5091, 6341, 7460, 8977, 3441]
old_lc = [3388, 3937, 4343, 5810, 7192, 8750, 10851, 4481]
old_e  = [3820, 4492, 5065, 5984, 7434, 8862, 11056, 6072]
print(f"  {'layer':>5} | {'tr Σ_OC':>10} {'(old)':>8} | {'tr Σ_LC':>10} {'(old)':>8} | {'tr Σ_E':>10} {'(old)':>8} | OC/LC  OC/E")
for li in range(t_oc.shape[0]):
    print(f"  {24+li:>5} | {t_oc[li].item():>10.0f} {old_oc[li]:>8.0f} | "
          f"{t_lc[li].item():>10.0f} {old_lc[li]:>8.0f} | "
          f"{t_e[li].item():>10.0f} {old_e[li]:>8.0f} | "
          f"{t_oc[li].item()/t_lc[li].item():.2f}   {t_oc[li].item()/t_e[li].item():.2f}")

# ── (4) per-domain contrast magnitudes ───────────────────────────────────────
print()
print("=" * 78)
print("(4) Per-domain pole prerequisites: ‖c_OC[d]‖ separation by dataset")
print("=" * 78)
meta_b = bundle.get("meta_B") or bundle.get("meta_b")
meta_a = bundle.get("meta_A") or bundle.get("meta_a")
meta_c = bundle.get("meta_C") or bundle.get("meta_c")
meta_d = bundle.get("meta_D") or bundle.get("meta_d")

if meta_b is None or meta_a is None:
    print("  (no meta — per-domain analysis skipped)")
else:
    da = [m.get("dataset", "?") for m in meta_a]
    db = [m.get("dataset", "?") for m in meta_b]
    # quick sanity: A and B should be aligned by row (forget pool order)
    aligned = all(x == y for x, y in zip(da, db))
    print(f"  A/B dataset alignment by row: {aligned}")
    for d in sorted(set(da)):
        idx = [i for i, x in enumerate(da) if x == d]
        if not idx: continue
        c_oc_d = c_oc[idx]
        print(f"  c_OC[{d:>5}]  N={len(idx):>3}  per-row ‖c‖={per_row_norm(c_oc_d).mean().item():.2f}  "
              f"per-row ‖c‖ std={per_row_norm(c_oc_d).std().item():.2f}")

if meta_c is None or meta_d is None:
    print("  (no meta_C/D — skipping)")
else:
    dc = [m.get("dataset", "?") for m in meta_c]
    dd = [m.get("dataset", "?") for m in meta_d]
    aligned = all(x == y for x, y in zip(dc, dd))
    print(f"  C/D dataset alignment by row: {aligned}")
    for d in sorted(set(dc)):
        idx = [i for i, x in enumerate(dc) if x == d]
        if not idx: continue
        c_lc_d = c_lc[idx]
        print(f"  c_LC[{d:>5}]  N={len(idx):>3}  per-row ‖c‖={per_row_norm(c_lc_d).mean().item():.2f}  "
              f"per-row ‖c‖ std={per_row_norm(c_lc_d).std().item():.2f}")

# ── (5) μ⁻(d) vs μ⁺(d) preview (these are what step 3 will compute) ──────────
print()
print("=" * 78)
print("(5) Pole preview: per-layer ‖μ⁻(d) − μ⁺(d)‖  (what the loss anchors against)")
print("=" * 78)
if meta_b is not None and meta_c is not None:
    db = [m.get("dataset", "?") for m in meta_b]
    dc = [m.get("dataset", "?") for m in meta_c]
    print(f"  {'layer':>5} | {'‖μ⁻_kuq − μ⁺_kuq‖':>20} | {'‖μ⁻_squad − μ⁺_squad‖':>22} | grand-mean")
    for li in range(B.shape[1]):
        per_d = []
        for d in ["kuq", "squad"]:
            ib = [i for i, x in enumerate(db) if x == d]
            ic = [i for i, x in enumerate(dc) if x == d]
            if not ib or not ic:
                per_d.append(float("nan")); continue
            mu_minus = B[ib, li].mean(0)
            mu_plus = C_[ic, li].mean(0)
            per_d.append((mu_minus - mu_plus).norm().item())
        # grand mean
        gm = (B[:, li].mean(0) - C_[:, li].mean(0)).norm().item()
        print(f"  {24+li:>5} | {per_d[0]:>20.2f} | {per_d[1]:>22.2f} | {gm:>10.2f}")

print()
print("Diagnostic done.")
