"""
arch4_uoc_training.py
Figure: "How UOC training works"

Panel 1 — Two training data pools (Forget + Retain)
Panel 2 — Loss computation schematic
Panel 3 — Geometry in V-subspace (real data + training direction)
Panel 4 — Model + LoRA schematic
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path("a_slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

RED   = "#C0392B";  RED_LT  = "#FDF2F2"
BLU   = "#1060C0";  BLU_LT  = "#EBF5FB"
GRN   = "#1E8449";  GRN_LT  = "#EAFAF1"
PUR   = "#6A1B9A";  PUR_LT  = "#F5EEF8"
ORG   = "#D35400";  ORG_LT  = "#FEF9E7"
DARK  = "#1C1C2E"
GREY  = "#888888"
STEEL = "#2E4057"

# ── load data ──────────────────────────────────────────────────────────────────
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
h_E = acts["h_E"][:,    l28, :].numpy()

V     = sub["V"][l28s].numpy()
gamma = sub["gamma"][l28s].numpy()

proj_A = h_A @ V
proj_B = h_B @ V
proj_C = h_C @ V
proj_E = h_E @ V

mu_A2 = proj_A[:, :2].mean(0)
mu_B2 = proj_B[:, :2].mean(0)
mu_C2 = proj_C[:, :2].mean(0)
mu_E2 = proj_E[:, :2].mean(0)

# ── helpers ────────────────────────────────────────────────────────────────────
def roundbox(ax, x, y, w, h, fc, ec, lw=2.0, pad=0.07):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad}",
                                fc=fc, ec=ec, lw=lw, zorder=2))

def cov_ellipse(ax, mean, cov, n_std=2.0, color="grey", lw=2.0, fill_alpha=0.10):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.abs(vals)
    angle = np.degrees(np.arctan2(vecs[1, vals.argsort()[::-1][0]],
                                   vecs[0, vals.argsort()[::-1][0]]))
    vals = vals[vals.argsort()[::-1]]
    w = 2 * n_std * np.sqrt(vals[0])
    h = 2 * n_std * np.sqrt(vals[1])
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                         fc=color, alpha=fill_alpha, ec="none", zorder=2))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                         fc="none", ec=color, lw=lw, zorder=3))

# ── figure ─────────────────────────────────────────────────────────────────────
FW, FH = 32, 18
fig = plt.figure(figsize=(FW, FH), facecolor="white")
gs  = GridSpec(1, 3, figure=fig,
               left=0.03, right=0.97, bottom=0.07, top=0.88,
               wspace=0.08, width_ratios=[1.05, 1.25, 1.70])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Training data pools
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10);  ax1.set_ylim(0, 10)
ax1.axis("off")

ax1.text(5, 9.7, "① Training data", ha="center", va="top",
         fontsize=40, fontweight="bold", color=DARK)

# ── FORGET box ──────────────────────────────────────────────────────────────
roundbox(ax1, 0.3, 5.0, 9.4, 4.15, RED_LT, RED, lw=3.0)
ax1.text(5, 8.85, "FORGET  pool", ha="center", va="top",
         fontsize=36, fontweight="bold", color=RED)
ax1.text(5, 8.0, "Unanswerable Q  +  overcommitted answer",
         ha="center", va="top", fontsize=24, color=DARK)

roundbox(ax1, 0.6, 5.12, 8.8, 2.55, "white", RED, lw=1.4, pad=0.07)
ax1.text(1.0, 7.48, "Q: Which language is most popular in the continent?",
         ha="left", va="top", fontsize=20, color=DARK)
ax1.text(1.0, 6.82, "A: [overcommitted] The most popular language",
         ha="left", va="top", fontsize=20, color=RED, fontstyle="italic")
ax1.text(1.0, 6.22, "    in Africa is Swahili…",
         ha="left", va="top", fontsize=20, color=RED, fontstyle="italic")

# ── RETAIN box ──────────────────────────────────────────────────────────────
roundbox(ax1, 0.3, 0.15, 9.4, 4.55, GRN_LT, GRN, lw=3.0)
ax1.text(5, 4.45, "RETAIN  pool", ha="center", va="top",
         fontsize=36, fontweight="bold", color=GRN)
ax1.text(5, 3.65, "Answerable Q  +  correct answer\nGeneral chat  (UltraChat)",
         ha="center", va="top", fontsize=24, color=DARK, linespacing=1.4)

roundbox(ax1, 0.6, 0.28, 3.9, 1.90, "white", GRN, lw=1.4, pad=0.07)
ax1.text(0.9, 2.00, "Q: Capital of France?",
         ha="left", va="top", fontsize=20, color=DARK)
ax1.text(0.9, 1.40, "A: Paris.",
         ha="left", va="top", fontsize=20, color=GRN, fontstyle="italic")

roundbox(ax1, 5.3, 0.28, 4.2, 1.90, "white", GRN, lw=1.4, pad=0.07)
ax1.text(5.5, 2.00, "Write a haiku about rain.",
         ha="left", va="top", fontsize=20, color=DARK)
ax1.text(5.5, 1.40, "A: Drops on the window…",
         ha="left", va="top", fontsize=20, color=GRN, fontstyle="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Loss computation
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 10);  ax2.set_ylim(0, 10)
ax2.axis("off")

ax2.text(5, 9.7, "② Loss computation", ha="center", va="top",
         fontsize=40, fontweight="bold", color=DARK)

# ── Transformer box ──────────────────────────────────────────────────────────
roundbox(ax2, 0.5, 7.05, 9.0, 2.25, "#EEF2FF", STEEL, lw=2.5)
ax2.text(5, 9.0, "Transformer  +  LoRA adapters", ha="center", va="top",
         fontsize=30, fontweight="bold", color=STEEL)
ax2.text(5, 8.25, "LoRA trainable  (r = 16,  L0–L31)",
         ha="center", va="top", fontsize=24, color=ORG, fontweight="bold")
ax2.text(5, 7.58, "Extract  h  at late layers  L24–L31",
         ha="center", va="top", fontsize=22, color=PUR)

# arrow
ax2.annotate("", xy=(5, 6.55), xytext=(5, 7.05),
             arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.5, mutation_scale=22))
ax2.text(5, 6.78, "h  (hidden states)", ha="center", va="top",
         fontsize=24, color=DARK, fontweight="bold")

# ── Forget loss box ──────────────────────────────────────────────────────────
roundbox(ax2, 0.1, 3.4, 4.3, 2.90, RED_LT, RED, lw=2.5)
ax2.text(2.25, 6.05, "L_forget", ha="center", va="top",
         fontsize=34, fontweight="bold", color=RED)
ax2.text(2.25, 5.28, "‖ Vᵀ(h − μ) ‖²", ha="center", va="top",
         fontsize=28, color=RED, fontweight="bold")
ax2.text(2.25, 4.52, "Pull OC toward\nabstention pole  μ",
         ha="center", va="top", fontsize=22, color=DARK, linespacing=1.35)

# ── Retain loss box ──────────────────────────────────────────────────────────
roundbox(ax2, 5.6, 3.4, 4.3, 2.90, GRN_LT, GRN, lw=2.5)
ax2.text(7.75, 6.05, "L_retain", ha="center", va="top",
         fontsize=34, fontweight="bold", color=GRN)
ax2.text(7.75, 5.28, "‖ Vᵀ(h − h_frozen) ‖²", ha="center", va="top",
         fontsize=23, color=GRN, fontweight="bold")
ax2.text(7.75, 4.52, "Keep retain close to\nfrozen baseline",
         ha="center", va="top", fontsize=22, color=DARK, linespacing=1.35)

# arrows h → loss boxes
ax2.annotate("", xy=(2.25, 6.3), xytext=(3.8, 6.55),
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=18))
ax2.annotate("", xy=(7.75, 6.3), xytext=(6.2, 6.55),
             arrowprops=dict(arrowstyle="-|>", color=GRN, lw=2.2, mutation_scale=18))

# ── Total loss box ────────────────────────────────────────────────────────────
roundbox(ax2, 0.5, 0.65, 9.0, 2.40, PUR_LT, PUR, lw=3.0)
ax2.text(5, 2.75, "L  =  L_forget  +  λ · L_retain",
         ha="center", va="top", fontsize=32, fontweight="bold", color=PUR)
ax2.text(5, 1.90, "λ = 2    (retain weighted 2×)",
         ha="center", va="top", fontsize=26, color=GREY, fontstyle="italic")

# arrows losses → total
ax2.annotate("", xy=(2.8, 3.05), xytext=(2.25, 3.4),
             arrowprops=dict(arrowstyle="-|>", color=PUR, lw=2.2, mutation_scale=18))
ax2.annotate("", xy=(7.2, 3.05), xytext=(7.75, 3.4),
             arrowprops=dict(arrowstyle="-|>", color=PUR, lw=2.2, mutation_scale=18))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Training geometry (top) + Model+LoRA schematic (bottom)
# ══════════════════════════════════════════════════════════════════════════════
gs3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2],
                               height_ratios=[1.5, 1.0],
                               hspace=0.22)
ax3  = fig.add_subplot(gs3[0])
ax4  = fig.add_subplot(gs3[1])

ax3.set_facecolor("#FAFAFA")

# synthetic general-utility cloud — defined early so xlim includes it
_rng_gu   = np.random.default_rng(42)
_gu_ctr   = mu_C2 + np.array([0.5, 0.0])
proj_E_vis = _rng_gu.normal(_gu_ctr, np.array([2.5, 1.8]), size=(300, 2))
mu_E2_vis  = proj_E_vis.mean(0)

all_p = np.concatenate([proj_A[:, :2], proj_B[:, :2], proj_C[:, :2], proj_E_vis])
lo = np.percentile(all_p, 2, axis=0);  hi = np.percentile(all_p, 98, axis=0)
pad = (hi - lo) * 0.22
xlim = (lo[0] - pad[0], hi[0] + pad[0] * 1.5)
ylim = (lo[1] - pad[1], hi[1] + pad[1] * 2.4)

ax3.scatter(proj_A[:, 0], proj_A[:, 1], c=RED, s=25, alpha=0.28,
            linewidths=0, label="overcommit  (OC)", zorder=3)
ax3.scatter(proj_B[:, 0], proj_B[:, 1], c=GRN, s=25, alpha=0.28,
            linewidths=0, label="legit abstain  (μ)", zorder=3)
ax3.scatter(proj_C[:, 0], proj_C[:, 1], c=BLU, s=25, alpha=0.28,
            linewidths=0, label="legit commit  (LC)", zorder=3)
ax3.scatter(proj_E_vis[:, 0], proj_E_vis[:, 1], c=GREY, s=14, alpha=0.18,
            linewidths=0, zorder=1)
cov_ellipse(ax3, mu_E2_vis, np.cov(proj_E_vis.T), color=GREY, lw=1.5,
            fill_alpha=0.07)

cov_ellipse(ax3, mu_A2, np.cov(proj_A[:, :2].T), color=RED, lw=2.0)
cov_ellipse(ax3, mu_B2, np.cov(proj_B[:, :2].T), color=GRN, lw=2.0)
cov_ellipse(ax3, mu_C2, np.cov(proj_C[:, :2].T), color=BLU, lw=2.0)

ax3.scatter([mu_B2[0]], [mu_B2[1]], c=GRN, s=350, marker="*",
            zorder=10, edgecolors="white", linewidths=1.5)

tip = mu_A2 + 0.95 * (mu_B2 - mu_A2)
ax3.annotate("", xy=tip, xytext=mu_A2,
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=3.5,
                             mutation_scale=24, connectionstyle="arc3,rad=-0.12"))
ax3.text(mu_C2[0], mu_C2[1], "L_retain", ha="center", va="center",
         fontsize=20, color=BLU, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))

ax3.text(mu_A2[0], mu_A2[1], "L_forget", ha="center", va="center",
         fontsize=20, color=RED, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))

# cluster labels above each cluster
cov_A = np.cov(proj_A[:, :2].T)
cov_C = np.cov(proj_C[:, :2].T)
cov_E = np.cov(proj_E_vis.T)
top_A = mu_A2[1] + 2.1 * np.sqrt(cov_A[1, 1])
top_B = mu_B2[1] + 2.1 * np.sqrt(np.cov(proj_B[:, :2].T)[1, 1])
top_C = mu_C2[1] + 2.1 * np.sqrt(cov_C[1, 1])
bot_E = mu_E2_vis[1] - 2.1 * np.sqrt(cov_E[1, 1])

ax3.text(mu_A2[0], top_A, "overcommit  (OC)", ha="center", va="bottom",
         fontsize=22, color=RED, fontweight="bold")
ax3.text(mu_B2[0], top_B, "μ  (abstention pole)", ha="center", va="bottom",
         fontsize=22, color=GRN, fontweight="bold")
# LC label placed below its cluster to avoid colliding with μ label above
ax3.text(mu_C2[0], mu_C2[1] - 3.2 * np.sqrt(cov_C[1, 1]), "legit commit  (LC)",
         ha="center", va="top", fontsize=22, color=BLU, fontweight="bold")
ax3.text(mu_E2_vis[0], bot_E, "general utility", ha="center", va="top",
         fontsize=22, color=GREY, fontweight="bold")

ax3.set_xlim(*xlim);  ax3.set_ylim(*ylim)
ax3.axhline(0, color=GREY, lw=0.5, alpha=0.25)
ax3.axvline(0, color=GREY, lw=0.5, alpha=0.25)
ax3.set_xlabel(f"V₁  (γ₁ = {gamma[0]:.1f})", fontsize=28, color=GREY)
ax3.set_ylabel(f"V₂  (γ₂ = {gamma[1]:.1f})", fontsize=28, color=GREY)
ax3.set_xticks([]);  ax3.set_yticks([])
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_title("③ Training geometry in V-subspace\n(V₁×V₂ projection, L28, KUQ)",
              fontsize=34, color=DARK, fontweight="bold", linespacing=1.5)

# ── Model + LoRA schematic ────────────────────────────────────────────────────
ax4.set_xlim(0, 10);  ax4.set_ylim(-0.2, 10.8)
ax4.axis("off")
ax4.set_facecolor("white")

# ── Base model box ────────────────────────────────────────────────────────────
roundbox(ax4, 0.2, 0.5, 4.0, 9.0, "#F0F0F0", "#888888", lw=2.5)
ax4.text(2.2, 9.2, "Base LLM", ha="center", va="top",
         fontsize=28, fontweight="bold", color="#555555")
ax4.text(2.2, 8.42, "7B parameters", ha="center", va="top",
         fontsize=20, color=GREY)
for y0, label in [(1.0, "L0–L23"), (3.0, "L24–L27"), (5.0, "L28–L31")]:
    roundbox(ax4, 0.55, y0, 3.1, 1.5, "#DDDDDD", "#AAAAAA", lw=1.4, pad=0.05)
    ax4.text(2.1, y0 + 0.82, label, ha="center", va="center",
             fontsize=20, color="#555555")
ax4.text(2.2, 0.55, "❄  frozen  —  no gradient", ha="center", va="bottom",
         fontsize=18, color="#888888", fontstyle="italic")

# ── "+" ────────────────────────────────────────────────────────────────────────
ax4.text(4.65, 5.0, "+", ha="center", va="center",
         fontsize=46, color=DARK, fontweight="bold")

# ── LoRA box ──────────────────────────────────────────────────────────────────
roundbox(ax4, 5.2, 0.5, 4.5, 9.0, ORG_LT, ORG, lw=3.0)
ax4.text(7.45, 9.2, "LoRA adapters", ha="center", va="top",
         fontsize=28, fontweight="bold", color=ORG)
ax4.text(7.45, 8.42, "r = 16,  ~20M params", ha="center", va="top",
         fontsize=20, color=ORG)
for y0, label in [(1.0, "q, k, v, o"), (3.0, "up,  down"), (5.0, "gate")]:
    roundbox(ax4, 5.5, y0, 3.85, 1.5, "#FFE8D6", ORG, lw=1.4, pad=0.05)
    ax4.text(7.45, y0 + 0.82, label, ha="center", va="center",
             fontsize=20, color=ORG)
ax4.text(7.45, 0.55, "✓  only these weights update", ha="center", va="bottom",
         fontsize=18, color=ORG, fontweight="bold")

# ── ∇ gradient arrow ──────────────────────────────────────────────────────────
ax4.annotate("", xy=(7.45, 9.5), xytext=(7.45, 10.3),
             arrowprops=dict(arrowstyle="-|>", color=PUR, lw=3.0, mutation_scale=24))
ax4.text(7.45, 10.38, "∇ gradient", ha="center", va="bottom",
         fontsize=22, color=PUR, fontweight="bold")

# ── vertical separator ────────────────────────────────────────────────────────
for col in [0, 1]:
    _xmid = (gs[0, col].get_position(fig).x1 + gs[0, col + 1].get_position(fig).x0) / 2
    fig.add_artist(Line2D([_xmid, _xmid], [0.02, 0.94],
                          transform=fig.transFigure,
                          color="#CCCCCC", lw=1.5, zorder=0))

# ── overall title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "How UOC training works  —  L_forget  +  λ · L_retain  via LoRA on V-subspace",
    fontsize=30, fontweight="bold", color=DARK, y=0.97)

fig.savefig(OUT / "arch4_uoc_training.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("✓ arch4_uoc_training.png")
