"""
arch3_eigenproblem.py
3-panel figure: "How we solve the generalised eigenproblem"

Panel 1 — Raw 3D scatter (3 classes)
Panel 2 — Matrix equation: Σ_OC − Σ_LC = S → whiten → M → eigen → V
Panel 3 — Projection onto V₁×V₂ → separation  +  eigenspectrum
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

OUT = Path("a_slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

RED  = "#C0392B";  RED_LT = "#FDEDEC"
BLU  = "#1060C0";  BLU_LT = "#BBDEFB"
GRN  = "#1E8449";  GRN_LT = "#EAFAF1"
PUR  = "#6A1B9A";  PUR_LT = "#EDE7F6"
ORG  = "#E65000";  ORG_LT = "#FFF3E0"
DARK = "#1C1C2E"
GREY = "#888888"

# ── load data ─────────────────────────────────────────────────────────────────
acts = torch.load("step1_extract_activations/data/activations_qwen_instruct.pt",
                  map_location="cpu", weights_only=False)
sub  = torch.load("step2_build_subspace/data/subspace_qwen_instruct_r32.pt",
                  map_location="cpu", weights_only=False)

l28   = acts["layers"].index(28)
l28s  = list(sub["layers"]).index(28)
mA    = acts["meta_A"];  mC = acts["meta_C"]
kuq_A = [i for i, m in enumerate(mA) if m["dataset"] == "kuq"]
kuq_C = [i for i, m in enumerate(mC) if m["dataset"] == "kuq"]

h_A = acts["h_A"][kuq_A, l28, :].numpy()
h_B = acts["h_B"][kuq_A, l28, :].numpy()
h_C = acts["h_C"][kuq_C, l28, :].numpy()
h_E = acts["h_E"][:,     l28, :].numpy()

V     = sub["V"][l28s].numpy()
gamma = sub["gamma"][l28s].numpy()

# ── matrices for Panel 2 (16×16 submatrix of real data) ───────────────────────
K = 16
cov_A_k = np.cov(h_A[:, :K].T)
cov_C_k = np.cov(h_C[:, :K].T)
cov_E_k = np.cov(h_E[:, :K].T)
S_k     = cov_A_k - cov_C_k
vals_Ek, vecs_Ek = np.linalg.eigh(cov_E_k)
vals_Ek = np.maximum(vals_Ek, 1e-10)
W_k = vecs_Ek @ np.diag(vals_Ek ** -0.5) @ vecs_Ek.T
M_k = W_k @ S_k @ W_k.T
V_k = V[:K, :]

proj_A = h_A @ V
proj_C = h_C @ V

# ── covariance ellipse helper ──────────────────────────────────────────────────
def cov_ellipse(ax, mean, cov, n_std=2.0, color="grey", lw=2.5, ls="-",
                fill_alpha=0.15, zorder=3):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.abs(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2 * n_std * np.sqrt(vals[0])
    h = 2 * n_std * np.sqrt(vals[1])
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                         fc=color, alpha=fill_alpha, ec="none", zorder=zorder))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                         fc="none", ec=color, lw=lw, ls=ls, zorder=zorder + 1))

# ── matrix heatmap helper ─────────────────────────────────────────────────────
def mat_ax(ax, data, cmap, border_color, title, subtitle, dim_str):
    vabs = np.percentile(np.abs(data), 99)
    ax.imshow(data, cmap=cmap, vmin=-vabs, vmax=vabs,
              aspect=1, interpolation="nearest")
    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color);  sp.set_linewidth(2.8)
    ax.set_title(f"{title}\n{subtitle}", fontsize=20, color=border_color,
                 fontweight="bold", pad=8, linespacing=1.35)
    ax.text(0.5, -0.22, dim_str, transform=ax.transAxes,
            ha="center", va="top", fontsize=16, color=GREY, fontstyle="italic")

def vec_to_mat_draw(ax, color):
    """Draw  h (tall column) × hᵀ (wide row) = hhᵀ (square)  in the given axis."""
    ax.set_xlim(0, 12);  ax.set_ylim(0, 10)
    ax.axis("off")

    col_w = 0.6;  col_h = 6.0
    row_w = 6.0;  row_h = 0.6
    sq    = 2.8
    yc    = 5.0

    ax.add_patch(plt.Rectangle((0.8, yc - col_h/2), col_w, col_h,
                                fc=color, alpha=0.80, ec=color, lw=1.2))
    ax.text(1.1, yc + col_h/2 + 0.3, "h",
            ha="center", va="bottom", fontsize=22, color=color, fontweight="bold")
    ax.text(1.1, yc - col_h/2 - 0.6, "4096×1",
            ha="center", va="top", fontsize=16, color=GREY, fontstyle="italic")

    ax.text(2.0, yc, "×", ha="center", va="center", fontsize=28, color=DARK)

    ax.add_patch(plt.Rectangle((2.5, yc - row_h/2), row_w, row_h,
                                fc=color, alpha=0.80, ec=color, lw=1.2))
    ax.text(5.5, yc + row_h/2 + 0.5, "hᵀ",
            ha="center", va="bottom", fontsize=22, color=color, fontweight="bold")
    ax.text(5.5, yc - row_h/2 - 0.6, "1×4096",
            ha="center", va="top", fontsize=16, color=GREY, fontstyle="italic")

    ax.text(9.2, yc, "=", ha="center", va="center", fontsize=28, color=DARK)

    ax.add_patch(plt.Rectangle((9.8, yc - sq/2), sq, sq,
                                fc=color, alpha=0.38, ec=color, lw=2.0))
    ax.text(11.2, yc + sq/2 + 0.4, "hhᵀ",
            ha="center", va="bottom", fontsize=20, color=color, fontweight="bold")
    ax.text(11.2, yc - sq/2 - 0.6, "4096×4096",
            ha="center", va="top", fontsize=16, color=GREY, fontstyle="italic")
    ax.text(11.2, yc + sq/2 + 1.4, "avg over N",
            ha="center", va="bottom", fontsize=16, color=GREY, fontstyle="italic")

def sign_ax(ax, txt, fontsize=36):
    ax.axis("off")
    ax.text(0.5, 0.52, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=DARK, fontweight="bold", linespacing=1.3)

# ── figure ─────────────────────────────────────────────────────────────────────
FW, FH = 32, 18
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs  = GridSpec(2, 2, figure=fig, left=0.02, right=0.98,
               bottom=0.06, top=0.89,
               wspace=0.20, hspace=0.28,
               width_ratios=[2.2, 1.0],
               height_ratios=[1.0, 1.6])

# ─── PANEL 1: split into [3D scatter | V direction explainer] ─────────────────
gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0],
                               wspace=0.10, width_ratios=[2.6, 1.0])
ax1 = fig.add_subplot(gs1[0], projection="3d")
ax1.set_facecolor("#F8F8FC")

A3 = h_A[:, :3];  B3 = h_B[:, :3];  C3 = h_C[:, :3]
ax1.scatter(A3[:, 0], A3[:, 1], A3[:, 2], c=RED, s=14, alpha=0.22,
            linewidths=0, depthshade=True)
ax1.scatter(B3[:, 0], B3[:, 1], B3[:, 2], c=GRN, s=14, alpha=0.22,
            linewidths=0, depthshade=True)
ax1.scatter(C3[:, 0], C3[:, 1], C3[:, 2], c=BLU, s=14, alpha=0.22,
            linewidths=0, depthshade=True)

for mu, col in [(A3.mean(0), RED), (B3.mean(0), GRN), (C3.mean(0), BLU)]:
    ax1.quiver(0, 0, 0, mu[0], mu[1], mu[2], color=col, lw=2.0,
               arrow_length_ratio=0.14, alpha=0.90)
    ax1.scatter([mu[0]], [mu[1]], [mu[2]], c=col, s=90, depthshade=False, zorder=10)
ax1.scatter([0], [0], [0], c="#444444", s=40, depthshade=False, zorder=10)

for c, lbl in [(RED, "overcommit"), (BLU, "legit commit"), (GRN, "legit abstain")]:
    ax1.scatter([], [], c=c, s=40, label=lbl)
ax1.legend(loc="upper left", fontsize=18, framealpha=0.88)

ax1.set_xlabel("dim 1", fontsize=16, color=GREY, labelpad=1)
ax1.set_ylabel("dim 2", fontsize=16, color=GREY, labelpad=1)
ax1.set_zlabel("dim 3", fontsize=16, color=GREY, labelpad=1)
ax1.set_xlim(-3, 3);  ax1.set_ylim(-4.5, 3);  ax1.set_zlim(-2, 3)
ax1.set_xticks([]);   ax1.set_yticks([]);       ax1.set_zticks([])
for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#DDDDEE")
ax1.grid(True, alpha=0.18, color="#CCCCDD")
ax1.view_init(elev=30, azim=150)
ax1.set_title("① Raw dims 1–3\nclouds overlap — no separation",
              fontsize=22, color=DARK, fontweight="bold", pad=6, linespacing=1.5)

# ── V direction explainer (standalone plot next to 3D scatter) ────────────────
ax_ins = fig.add_subplot(gs1[1])
ax_ins.set_facecolor("#FFFFF0")
ax_ins.set_xlim(-2.5, 2.5);  ax_ins.set_ylim(-2.5, 2.5)
ax_ins.set_xticks([]);  ax_ins.set_yticks([])
ax_ins.set_aspect("equal")
for sp in ax_ins.spines.values():
    sp.set_edgecolor("#BBBBBB");  sp.set_linewidth(0.8)

np.random.seed(7)
_cov_demo = np.array([[2.4, 1.2], [1.2, 0.8]])
_pts = np.random.multivariate_normal([0, 0], _cov_demo, 120)
ax_ins.scatter(_pts[:, 0], _pts[:, 1], c=GREY, s=7, alpha=0.35, linewidths=0)

for dx, dy, lbl in [(1, 0, "dim i"), (0, 1, "dim j")]:
    ax_ins.annotate("", xy=(dx*2.0, dy*2.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#BBBBBB",
                                   lw=1.2, mutation_scale=9))
    ax_ins.text(dx*2.15, dy*2.15, lbl, color="#BBBBBB", fontsize=12, ha="center")

_vals, _vecs = np.linalg.eigh(_cov_demo)
_order = _vals.argsort()[::-1]
_vals, _vecs = _vals[_order], _vecs[:, _order]
_pc1, _pc2 = _vecs[:, 0], _vecs[:, 1]
for pc, col, lbl, sc in [(_pc1, ORG, "V₁\n(max OC−LC)", 1.85),
                          (_pc2, PUR, "V₂\n(⊥ V₁)", 1.05)]:
    ax_ins.annotate("", xy=pc*sc, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=2.2, mutation_scale=12))
    ax_ins.text(*(pc * sc * 1.25), lbl, color=col, fontsize=12,
                fontweight="bold", ha="center", linespacing=1.2)

ax_ins.set_title("V = discriminative direction\n(linear combo of all 4096 dims)",
                 fontsize=12, color=DARK, fontweight="bold", pad=3, linespacing=1.4)

# ─── PANEL 2: matrix equation ─────────────────────────────────────────────────
# 3-row: [title] / [h×hᵀ mini-diagrams] / [matrix heatmaps + signs]
W_RATIOS = [3, 0.45, 3, 0.45, 3, 0.80, 3, 0.80, 1.8]
gs2 = GridSpecFromSubplotSpec(3, 9, subplot_spec=gs[1, 0],
                               height_ratios=[0.22, 0.72, 1.0],
                               hspace=0.14, wspace=0.08,
                               width_ratios=W_RATIOS)

# Row 0: panel title
ax_p2title = fig.add_subplot(gs2[0, :])
ax_p2title.axis("off")
ax_p2title.text(0.5, 0.5, "② How we solve  (Σ_OC − Σ_LC) v = γ Σ_E v",
                transform=ax_p2title.transAxes, ha="center", va="center",
                fontsize=26, fontweight="bold", color=DARK)

# Row 1: h×hᵀ mini-diagrams
ax_htop_oc = fig.add_subplot(gs2[1, 0])
ax_htop_lc = fig.add_subplot(gs2[1, 2])
for col in [1, 3, 4, 5, 6, 7, 8]:
    fig.add_subplot(gs2[1, col]).axis("off")

vec_to_mat_draw(ax_htop_oc, RED)
vec_to_mat_draw(ax_htop_lc, BLU)

# Row 2: matrix heatmaps and sign operators
ax_oc  = fig.add_subplot(gs2[2, 0])
ax_s1  = fig.add_subplot(gs2[2, 1])
ax_lc  = fig.add_subplot(gs2[2, 2])
ax_s2  = fig.add_subplot(gs2[2, 3])
ax_s   = fig.add_subplot(gs2[2, 4])
ax_s3  = fig.add_subplot(gs2[2, 5])
ax_m   = fig.add_subplot(gs2[2, 6])
ax_s4  = fig.add_subplot(gs2[2, 7])
ax_v   = fig.add_subplot(gs2[2, 8])

mat_ax(ax_oc, cov_A_k, "RdBu_r",   RED, "Σ_OC",            "how OC activations\nspread",  "4096×4096\n(16×16 shown)")
mat_ax(ax_lc, cov_C_k, "RdBu_r",   BLU, "Σ_LC",            "how LC activations\nspread",  "4096×4096\n(16×16 shown)")
mat_ax(ax_s,  S_k,     "PiYG",     PUR, "S = Σ_OC − Σ_LC", "their spread\ndifference",    "4096×4096")
mat_ax(ax_m,  M_k,     "coolwarm", ORG, "M = W S W",        "difference ÷\ngeneral utility", "4096×4096")

vabs_v = np.percentile(np.abs(V_k), 99)
ax_v.imshow(V_k, cmap="PRGn", vmin=-vabs_v, vmax=vabs_v,
            aspect=1, interpolation="nearest")
ax_v.set_xticks([]);  ax_v.set_yticks([])
for sp in ax_v.spines.values():
    sp.set_edgecolor(GRN);  sp.set_linewidth(2.8)
ax_v.set_title("V\ntop 32 discriminative\ndirections", fontsize=20, color=GRN,
               fontweight="bold", pad=8, linespacing=1.35)
ax_v.text(0.5, -0.22, "4096×32", transform=ax_v.transAxes,
          ha="center", va="top", fontsize=16, color=GREY, fontstyle="italic")

sign_ax(ax_s1, "−", fontsize=38)
sign_ax(ax_s2, "=", fontsize=38)
sign_ax(ax_s3, "→\n÷ general\nutility\n(Σ_E→I)\n→", fontsize=16)
sign_ax(ax_s4, "→\ntop r\neigen-\nvectors\n→", fontsize=16)

# ─── PANEL 3: V₁×V₂ projection + eigenspectrum ───────────────────────────────
gs3  = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2, 1],
                               hspace=0.44, height_ratios=[2.0, 1.0])
ax3  = fig.add_subplot(gs3[0])
axC  = fig.add_subplot(gs3[1])
ax3.set_facecolor("#FAFAFA")

ax3.scatter(proj_A[:, 0], proj_A[:, 1], c=RED, s=35, alpha=0.35,
            linewidths=0, label="overcommit", zorder=3)
ax3.scatter(proj_C[:, 0], proj_C[:, 1], c=BLU, s=35, alpha=0.35,
            linewidths=0, label="legit commit", zorder=3)

mu_pA = proj_A[:, :2].mean(0);  mu_pC = proj_C[:, :2].mean(0)
cov_ellipse(ax3, mu_pA, np.cov(proj_A[:, :2].T), n_std=2.0,
            color=RED, lw=2.8, fill_alpha=0.14)
cov_ellipse(ax3, mu_pC, np.cov(proj_C[:, :2].T), n_std=2.0,
            color=BLU, lw=2.8, fill_alpha=0.14)

for mu, col, lbl in [(mu_pA, RED, "Σ_OC"), (mu_pC, BLU, "Σ_LC")]:
    ax3.text(mu[0], mu[1], lbl, ha="center", va="center", fontsize=20,
             color=col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col, lw=1.8, alpha=0.92),
             zorder=8)

all_p = np.concatenate([proj_A[:, :2], proj_C[:, :2]], axis=0)
lo4   = np.percentile(all_p, 2, axis=0);  hi4 = np.percentile(all_p, 98, axis=0)
pad4  = (hi4 - lo4) * 0.14
ax3.set_xlim(lo4[0] - pad4[0], hi4[0] + pad4[0])
ax3.set_ylim(lo4[1] - pad4[1], hi4[1] + pad4[1])
ax3.axhline(0, color=GREY, lw=0.5, alpha=0.25)
ax3.axvline(0, color=GREY, lw=0.5, alpha=0.25)
ax3.set_xlabel(f"V₁  (γ₁ = {gamma[0]:.1f})", fontsize=18, color=GREY)
ax3.set_ylabel(f"V₂  (γ₂ = {gamma[1]:.1f})", fontsize=18, color=GREY)
ax3.set_xticks([]);  ax3.set_yticks([])
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(fontsize=18, loc="upper right", framealpha=0.88)
ax3.set_title("③ Project onto V₁×V₂\nOC and LC now separated",
              fontsize=22, color=DARK, fontweight="bold", linespacing=1.5)

# ── eigenspectrum ──────────────────────────────────────────────────────────────
axC.set_facecolor("#FAFAFA")
ranks = np.arange(1, len(gamma) + 1)
bar_colors = [RED if i == 0 else (PUR if i < 3 else "#C8C8D8") for i in range(len(gamma))]
axC.bar(ranks, gamma, color=bar_colors, edgecolor="none", width=0.85, zorder=3)
axC.axvline(32.5, color=DARK, lw=1.8, ls="--", alpha=0.65, zorder=4)
axC.text(31.5, gamma.max() * 0.55, "r=32\ncutoff", color=DARK,
         fontsize=16, va="center", ha="right")
axC.text(1, gamma[0] + 0.8, f"γ₁ = {gamma[0]:.1f}",
         color=RED, fontsize=18, fontweight="bold", ha="center")
axC.text(2, gamma[1] + 0.5, f"γ₂ = {gamma[1]:.1f}",
         color=PUR, fontsize=16, ha="center")
axC.set_xlabel("eigenvector rank", fontsize=18, color=GREY)
axC.set_ylabel("eigenvalue  γ", fontsize=18, color=GREY)
axC.set_xlim(0.5, 32.5)
axC.spines[["top", "right"]].set_visible(False)
axC.tick_params(labelsize=14)
axC.grid(axis="y", alpha=0.22, color="#CCCCDD", zorder=0)
axC.set_title("Eigenspectrum  (real data, r=32)",
              fontsize=20, color=DARK, fontweight="bold")

# ── overall title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "How we solve the generalised eigenproblem  "
    "(Σ_OC − Σ_LC) v = γ Σ_E v   —   L28, KUQ, 1000 instances",
    fontsize=24, fontweight="bold", color=DARK, y=0.98)

# ── vertical separator between left panels and panel 3 ────────────────────────
from matplotlib.lines import Line2D
_xmid = (gs[0, 0].get_position(fig).x1 + gs[0, 1].get_position(fig).x0) / 2
fig.add_artist(Line2D([_xmid, _xmid], [0.02, 0.96],
                      transform=fig.transFigure,
                      color="#BBBBBB", lw=1.5, zorder=0))

fig.savefig(OUT / "arch3_eigenproblem.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("✓ arch3_eigenproblem.png")
