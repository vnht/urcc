"""
arch5_results.py
Figure: "Training outcome"

Left  — Training curves: L_forget and L_retain over steps
Right — Before / After V-subspace scatter (baseline vs trained model)
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from pathlib import Path

OUT = Path("a_slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

RED  = "#C0392B";  RED_LT  = "#FDF2F2"
BLU  = "#1060C0";  BLU_LT  = "#EBF5FB"
GRN  = "#1E8449"
ORG  = "#D35400"
PUR  = "#6A1B9A"
DARK = "#1C1C2E"
GREY = "#888888"

RUN  = "qwen_instruct_uoc_r32_lam2_ep3_lr3e-05"
LAYER = 28

# ── load data ──────────────────────────────────────────────────────────────────
sub  = torch.load("step2_build_subspace/data/subspace_qwen_instruct_r32.pt",
                  map_location="cpu", weights_only=False)

acts_base    = torch.load("step1_extract_activations/data/activations_qwen_instruct.pt",
                          map_location="cpu", weights_only=False)
acts_trained = torch.load(f"step6_extract_trained_activations/data/activations_{RUN}.pt",
                          map_location="cpu", weights_only=False)

loss_df = pd.read_csv(f"step4_train/data/runs/{RUN}/loss_log.csv")

# ── subspace projection ────────────────────────────────────────────────────────
l28s  = list(sub["layers"]).index(LAYER)
V     = sub["V"][l28s].numpy()
gamma = sub["gamma"][l28s].numpy()

mA    = acts_base["meta_A"]
mC    = acts_base["meta_C"]
kuq_A = [i for i, m in enumerate(mA) if m["dataset"] == "kuq"]
kuq_C = [i for i, m in enumerate(mC) if m["dataset"] == "kuq"]
l28   = acts_base["layers"].index(LAYER)

def proj2(acts, kuq_idx, key):
    h = acts[key][kuq_idx, l28, :].numpy()
    return (h @ V)[:, :2]

# baseline
b_A = proj2(acts_base, kuq_A, "h_A")
b_B = proj2(acts_base, kuq_A, "h_B")
b_C = proj2(acts_base, kuq_C, "h_C")

# trained  (same kuq indices)
mA_t  = acts_trained["meta_A"]
mC_t  = acts_trained["meta_C"]
kuq_At = [i for i, m in enumerate(mA_t) if m["dataset"] == "kuq"]
kuq_Ct = [i for i, m in enumerate(mC_t) if m["dataset"] == "kuq"]

t_A = proj2(acts_trained, kuq_At, "h_A")
t_B = proj2(acts_trained, kuq_At, "h_B")   # legit abstain (target μ)
t_C = proj2(acts_trained, kuq_Ct, "h_C")

mu_bA = b_A.mean(0);  mu_tA = t_A.mean(0)
mu_bB = b_B.mean(0);  mu_tB = t_B.mean(0)
mu_bC = b_C.mean(0);  mu_tC = t_C.mean(0)

# ── helpers ────────────────────────────────────────────────────────────────────
def cov_ellipse(ax, mean, cov, n_std=2.0, color="grey", lw=1.8,
                fill_alpha=0.10, ls="-"):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.abs(vals)
    order = vals.argsort()[::-1]
    angle = np.degrees(np.arctan2(vecs[1, order[0]], vecs[0, order[0]]))
    w = 2 * n_std * np.sqrt(vals[order[0]])
    h = 2 * n_std * np.sqrt(vals[order[1]])
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                         fc=color, alpha=fill_alpha, ec="none", zorder=2))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                         fc="none", ec=color, lw=lw, ls=ls, zorder=3))

def common_lims(*arrs, pad=0.20):
    all_p = np.concatenate(arrs)
    lo = np.percentile(all_p, 2, axis=0)
    hi = np.percentile(all_p, 98, axis=0)
    margin = (hi - lo) * pad
    return (lo[0]-margin[0], hi[0]+margin[0]), (lo[1]-margin[1]*1.5, hi[1]+margin[1]*1.5)

# ── figure ─────────────────────────────────────────────────────────────────────
FW, FH = 28, 11
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs  = GridSpec(1, 2, figure=fig,
               left=0.06, right=0.97, bottom=0.12, top=0.86,
               wspace=0.10, width_ratios=[1.0, 1.6])

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Training curves
# ══════════════════════════════════════════════════════════════════════════════
ax_loss = fig.add_subplot(gs[0])
ax_loss.set_facecolor("#FAFAFA")

steps = loss_df["step"].values
# smooth with rolling mean
w = 8
lf = pd.Series(loss_df["L_forget"].values).rolling(w, min_periods=1, center=True).mean().values
lr = pd.Series(loss_df["L_retain"].values).rolling(w, min_periods=1, center=True).mean().values

ax_loss.plot(steps, lf, color=RED, lw=2.5, label="L_forget")
ax_loss.plot(steps, lr, color=GRN, lw=2.5, label="L_retain")
ax_loss.scatter(steps, loss_df["L_forget"].values, c=RED, s=6, alpha=0.18, linewidths=0, zorder=2)
ax_loss.scatter(steps, loss_df["L_retain"].values, c=GRN, s=6, alpha=0.18, linewidths=0, zorder=2)

# epoch boundaries (375 steps / 3 epochs)
steps_per_epoch = len(steps) // 3
for ep in range(1, 3):
    ax_loss.axvline(ep * steps_per_epoch, color=GREY, lw=1.0, ls="--", alpha=0.5)
    ax_loss.text(ep * steps_per_epoch + 4, ax_loss.get_ylim()[1] if ax_loss.get_ylim()[1] > 0 else 1.4,
                 f"epoch {ep+1}", fontsize=14, color=GREY, va="top")

ax_loss.set_xlabel("training step", fontsize=18, color=GREY)
ax_loss.set_ylabel("loss", fontsize=18, color=GREY)
ax_loss.tick_params(labelsize=14)
ax_loss.spines[["top", "right"]].set_visible(False)
ax_loss.legend(fontsize=16, framealpha=0.85)
ax_loss.set_title("① Training curves", fontsize=26, fontweight="bold", color=DARK, pad=10)

# annotate final values
ax_loss.annotate(f"final: {lf[-1]:.3f}", xy=(steps[-1], lf[-1]),
                 xytext=(-60, 12), textcoords="offset points",
                 fontsize=13, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
ax_loss.annotate(f"final: {lr[-1]:.3f}", xy=(steps[-1], lr[-1]),
                 xytext=(-60, -20), textcoords="offset points",
                 fontsize=13, color=GRN, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=GRN, lw=1.2))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Before / After scatter  (two side-by-side axes)
# ══════════════════════════════════════════════════════════════════════════════
from matplotlib.gridspec import GridSpecFromSubplotSpec
gs_r = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.06)
ax_b = fig.add_subplot(gs_r[0])   # before
ax_a = fig.add_subplot(gs_r[1])   # after

xlim, ylim = common_lims(b_A, b_B, b_C, t_A, t_B, t_C)

for ax, proj_A, proj_B, proj_C, mu_A, mu_B, mu_C, title in [
    (ax_b, b_A, b_B, b_C, mu_bA, mu_bB, mu_bC, "BEFORE  (baseline)"),
    (ax_a, t_A, t_B, t_C, mu_tA, mu_tB, mu_tC, "AFTER   (UOC trained)"),
]:
    ax.set_facecolor("#FAFAFA")
    ax.scatter(proj_A[:, 0], proj_A[:, 1], c=RED, s=18, alpha=0.28, linewidths=0, zorder=3)
    ax.scatter(proj_B[:, 0], proj_B[:, 1], c=GRN, s=18, alpha=0.28, linewidths=0, zorder=3)
    ax.scatter(proj_C[:, 0], proj_C[:, 1], c=BLU, s=18, alpha=0.28, linewidths=0, zorder=3)

    cov_ellipse(ax, mu_A, np.cov(proj_A.T), color=RED, lw=2.0)
    cov_ellipse(ax, mu_B, np.cov(proj_B.T), color=GRN, lw=2.0)
    cov_ellipse(ax, mu_C, np.cov(proj_C.T), color=BLU, lw=2.0)

    # μ star
    ax.scatter([mu_B[0]], [mu_B[1]], c=GRN, s=280, marker="*",
               zorder=10, edgecolors="white", linewidths=1.2)

    # cluster labels above
    for mu, label, color in [(mu_A, "OC", RED), (mu_B, "μ", GRN), (mu_C, "LC", BLU)]:
        std_y = np.sqrt(np.cov(proj_A.T if color == RED else
                               proj_B.T if color == GRN else
                               proj_C.T)[1, 1])
        ax.text(mu[0], mu[1] + 2.2 * std_y, label, ha="center", va="bottom",
                fontsize=18, color=color, fontweight="bold")

    ax.set_xlim(*xlim);  ax.set_ylim(*ylim)
    ax.axhline(0, color=GREY, lw=0.4, alpha=0.3)
    ax.axvline(0, color=GREY, lw=0.4, alpha=0.3)
    ax.set_xticks([]);  ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    color = RED if "BEFORE" in title else GRN
    ax.set_title(title, fontsize=20, fontweight="bold",
                 color=RED if "BEFORE" in title else ORG, pad=8)

ax_b.set_xlabel(f"V₁  (γ₁ = {gamma[0]:.1f})", fontsize=16, color=GREY)
ax_b.set_ylabel(f"V₂  (γ₂ = {gamma[1]:.1f})", fontsize=16, color=GREY)
ax_a.set_xlabel(f"V₁  (γ₁ = {gamma[0]:.1f})", fontsize=16, color=GREY)

# Arrow showing OC cluster shift between panels
ax_a.annotate("",
    xy=mu_tA, xytext=mu_bA,
    xycoords=ax_a.transData, textcoords=ax_b.transData,
    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5,
                    mutation_scale=20,
                    connectionstyle="arc3,rad=-0.25"),
    annotation_clip=False)

# overall section title
ax_b.set_title("BEFORE  (baseline)", fontsize=20, fontweight="bold", color=RED, pad=8)
ax_a.set_title("AFTER   (UOC trained)", fontsize=20, fontweight="bold", color=ORG, pad=8)

# shared right-panel title
fig.text(0.685, 0.935, "② V-subspace geometry  (V₁×V₂, L28, KUQ)",
         ha="center", va="bottom", fontsize=24, fontweight="bold", color=DARK)

# suptitle
fig.suptitle(
    "UOC training outcome  —  OC cluster shifts toward μ  while  LC stays anchored",
    fontsize=24, fontweight="bold", color=DARK, y=0.99)

fig.savefig(OUT / "arch5_results.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("✓ arch5_results.png")
