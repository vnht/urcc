"""
arch1_diagram.py — clean two-path transformer diagram.
All vertical positions are computed top-to-bottom to guarantee zero overlap.
"""
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path

OUT = Path("a_slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
RED    = "#C0392B";  RED_LT  = "#FDEDEC";  RED_MK  = "#D44444"
GRN    = "#1E8449";  GRN_LT  = "#EAFAF1";  GRN_MK  = "#2EAA5C"
BLUE   = "#2471A3";  BLU_LT  = "#EBF5FB"
ORG    = "#E67E22"
PUR    = "#7D3C98"
DARK   = "#1C1C2E"
GREY   = "#888888"

# ── figure ────────────────────────────────────────────────────────────────────
FW, FH = 24, 14
fig = plt.figure(figsize=(FW, FH), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

def box(x, y, w, h, fc, ec, lw=1.8, r=0.10, alpha=1.0, z=2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))

def arrow(x0, y0, x1, y1, c, lw=2.2, ms=14, ls="solid"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                linestyle=ls, mutation_scale=ms), zorder=8)

def t(x, y, s, **kw):
    ax.text(x, y, s, **kw)

# ─────────────────────────────────────────────────────────────────────────────
# HORIZONTAL GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
LC_X   = 0.35          # left column x
RC_X   = 15.35         # right column x  (leaves 8.3 wide each, 6.65 middle gap)
COL_W  = 8.30
MID_X  = LC_X + COL_W  # 8.65  (left edge of middle gap)
MID_W  = RC_X - MID_X  # 6.70

TWR_OFF = 0.42         # inset of bar-chart towers inside each column
TWR_W   = COL_W - 2*TWR_OFF   # 7.46
LC_TW   = LC_X + TWR_OFF      # 0.77
RC_TW   = RC_X + TWR_OFF      # 15.77

# ─────────────────────────────────────────────────────────────────────────────
# VERTICAL GEOMETRY — computed top → bottom, no overlaps possible
# Each section is allocated a height; y-positions are derived in sequence.
# ─────────────────────────────────────────────────────────────────────────────
# Reserve heights for each section
H_TOP    = 0.20   # top margin
H_TITLE  = 0.65   # title text block
H_G1     = 0.18   # gap: title → headers
H_HDR    = 0.62   # PATH header boxes
H_G2     = 0.12   # gap: headers → prompt
H_PRO    = 0.68   # prompt boxes
H_G3     = 0.12   # gap: prompt → continuation
H_CON    = 0.84   # continuation boxes
H_G4     = 0.60   # gap: continuation → column top (visual breath before layers)
H_EB     = 1.10   # per-column early-layer block (Zone B)
H_G5     = 0.00   # NO gap — early rows flow directly into late rows
H_BOT    = 0.22   # bottom margin (must exceed box rounding r=0.10)
H_INS    = 0.55   # insight banner
H_GINS   = 0.14   # gap: insight → result boxes
H_RES    = 1.12   # result boxes
H_GARR   = 0.22   # gap: result box top → late column bottom (arrow crosses)

FIXED = (H_TOP + H_TITLE + H_G1 + H_HDR + H_G2 + H_PRO + H_G3 + H_CON +
         H_G4 + H_EB + H_G5 + H_BOT + H_INS + H_GINS + H_RES + H_GARR)
H_LATE = FH - FIXED          # height for all 8 late-layer rows
late_h  = H_LATE / 8         # per-layer row height

# Derive y-positions (variable = bottom edge of each section unless noted)
y = FH
y -= H_TOP
TITLE_CY = y - H_TITLE / 2   # title text centre
y -= H_TITLE
y -= H_G1
HDR_Y = y - H_HDR;  y = HDR_Y
y -= H_G2
PRO_Y = y - H_PRO;  y = PRO_Y
y -= H_G3
CON_Y = y - H_CON;  y = CON_Y
y -= H_G4
ZB_TOP = y
ZB_BOT = ZB_TOP - H_EB;  y = ZB_BOT
# Gap 5: diverge text at mid-point, late column labels just above ZC_TOP
DIV_Y  = ZB_BOT - H_G5 / 2   # centre of "paths split here" text
ZC_TOP = ZB_BOT - H_G5;  y = ZC_TOP
ZC_BOT = ZC_TOP - H_LATE;  y = ZC_BOT
y -= H_GARR
RES_TOP = y                  # top of result boxes  (arrow ends here)
RES_Y   = y - H_RES          # bottom of result boxes
y = RES_Y
y -= H_GINS
INS_TOP = y
INSIGHT_Y = INS_TOP - H_INS

# ── quick sanity check ────────────────────────────────────────────────────────
assert INSIGHT_Y >= H_BOT - 0.01, \
    f"Layout overflow: insight_y={INSIGHT_Y:.3f}, need >= {H_BOT}"

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
t(FW/2, TITLE_CY,
  '"Which language is the most popular in the continent?"',
  ha="center", va="center", fontsize=22, fontweight="bold", color=DARK,
  bbox=dict(boxstyle="round,pad=0.38", fc="#FEF9E7", ec=ORG, lw=2.8), zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# ZONE A — PATH headers + input token boxes
# ─────────────────────────────────────────────────────────────────────────────
# Headers
box(LC_X, HDR_Y, COL_W, H_HDR, RED_LT, RED, lw=2.5)
box(RC_X, HDR_Y, COL_W, H_HDR, GRN_LT, GRN, lw=2.5)
t(LC_X + COL_W/2, HDR_Y + H_HDR/2,
  "PATH ①  —  Let the model answer freely",
  ha="center", va="center", fontsize=16, color=RED, fontweight="bold")
t(RC_X + COL_W/2, HDR_Y + H_HDR/2,
  "PATH ②  —  We set the answer: abstain",
  ha="center", va="center", fontsize=16, color=GRN, fontweight="bold")
t(MID_X + MID_W/2, HDR_Y + H_HDR/2, "vs",
  ha="center", va="center", fontsize=22, color=GREY,
  fontweight="bold", style="italic", alpha=0.4)

# Prompt boxes (same in both columns — full question)
for cx in [LC_X, RC_X]:
    box(cx + 0.15, PRO_Y, COL_W - 0.30, H_PRO, BLU_LT, BLUE, lw=1.8)
    t(cx + COL_W/2, PRO_Y + H_PRO/2,
      '"Which language is the most popular in the continent?  Answer:"',
      ha="center", va="center", fontsize=13, color=BLUE)

# Left continuation box
box(LC_X + 0.15, CON_Y, COL_W - 0.30, H_CON, RED_LT, RED, lw=2.0)
t(LC_X + COL_W/2, CON_Y + H_CON/2,
  '"The most popular language in Africa is Swahili..."',
  ha="center", va="center", fontsize=14, color=RED)

# Right continuation box
box(RC_X + 0.15, CON_Y, COL_W - 0.30, H_CON, GRN_LT, GRN, lw=2.2)
# Abstain text (left-aligned, leaves room for badge)
t(RC_X + 0.55, CON_Y + H_CON/2,
  '"I do not have enough information to answer that."',
  ha="left", va="center", fontsize=14, color=GRN)
# FORCED badge — inside right side of right continuation box
BW, BH = 2.20, 0.55
BX = RC_X + COL_W - BW - 0.20
BY = CON_Y + (H_CON - BH) / 2
box(BX, BY, BW, BH, GRN, GRN, lw=0, r=0.08, z=9)
t(BX + BW/2, BY + BH/2,
  "SET BY US\n(not generated)",
  ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=10)

# Bar-chart / row constants (used in both Zone B and C)
layers  = [24, 25, 26, 27, 28, 29, 30, 31]
N_BARS  = 200
BAR_GAP = 0.08
LBL_W   = 0.60   # width reserved for layer label
CPAD    = 0.07
ROW_PAD = 0.06

# ─────────────────────────────────────────────────────────────────────────────
# ZONE B + C unified — one tall column box per path, L0→L31 inside
# ─────────────────────────────────────────────────────────────────────────────
N_EARLY = 24
er_h    = H_EB / N_EARLY

# Arrow: continuation box → top of unified column
arrow(LC_X + COL_W/2, CON_Y, LC_X + COL_W/2, ZB_TOP, RED, lw=2.2)
arrow(RC_X + COL_W/2, CON_Y, RC_X + COL_W/2, ZB_TOP, GRN, lw=2.2)

# Unified column borders (span from ZC_BOT all the way up to ZB_TOP)
COL_H = ZB_TOP - ZC_BOT
box(LC_TW - 0.22, ZC_BOT - 0.10, TWR_W + 0.44, COL_H + 0.20,
    RED_LT, RED, lw=2.2, r=0.15)
box(RC_TW - 0.22, ZC_BOT - 0.10, TWR_W + 0.44, COL_H + 0.20,
    GRN_LT, GRN, lw=2.2, r=0.15)

# Column title labels (top of each box)
t(LC_TW + TWR_W/2, ZB_TOP - 0.06, "Commit path",
  ha="center", va="top", fontsize=14, color=RED, fontweight="bold")
t(RC_TW + TWR_W/2, ZB_TOP - 0.06, "Set-answer abstain path",
  ha="center", va="top", fontsize=14, color=GRN, fontweight="bold")

# Early rows L0-L23 (compact gradient rows, same column colour, no bar charts)
for (tx, bg, lc) in [(LC_TW, RED_LT, RED), (RC_TW, GRN_LT, GRN)]:
    for i in range(N_EARLY):
        row_bot = ZB_BOT + i * er_h
        alpha   = 0.10 + 0.25 * (i / N_EARLY)
        box(tx, row_bot, TWR_W, er_h,
            lc, lc, lw=0, alpha=alpha, r=0.01, z=4)
    t(tx + LBL_W/2, ZB_BOT + H_EB/2, "L0\n…\nL23",
      ha="center", va="center", fontsize=11, color=DARK, fontweight="bold", zorder=6)
    t(tx + TWR_W/2, ZB_BOT + H_EB/2,
      "Layers 0 – 23\nTokenisation, fact recall,\nearly reasoning",
      ha="center", va="center", fontsize=12, color=DARK, zorder=6,
      bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.88, ec="none"))

# "identical computation" dashed connector between the two early sections
mid_y = ZB_BOT + H_EB / 2
ax.plot([LC_TW + TWR_W + 0.25, RC_TW - 0.25], [mid_y, mid_y],
        color="#AAAAAA", lw=1.5, linestyle="--", zorder=3)
t(MID_X + MID_W/2, mid_y + 0.10,
  "identical computation",
  ha="center", va="bottom", fontsize=12, color=GREY, style="italic")

# Divider line + inline "late layers" label inside each column at L23/L24 boundary
for (tx, lc) in [(LC_TW, RED), (RC_TW, GRN)]:
    ax.plot([tx + 0.08, tx + TWR_W - 0.08], [ZC_TOP, ZC_TOP],
            color=lc, lw=1.8, alpha=0.55, linestyle="-", zorder=6)
    t(tx + TWR_W/2, ZC_TOP + 0.04,
      "▼  late layers (L24–L31)",
      ha="center", va="bottom", fontsize=11, color=lc,
      fontweight="bold", style="italic", zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# ZONE C — Late layers L24-L31 with activation bar charts
# ─────────────────────────────────────────────────────────────────────────────
# (column borders and labels are drawn in the Zone B+C unified section above)

for li, l in enumerate(layers):
    row_bot = ZC_TOP - (li + 1) * late_h + ROW_PAD
    row_h   = late_h - 2*ROW_PAD
    y_ctr   = row_bot + row_h / 2
    y_base  = row_bot + row_h * 0.08
    max_bar = row_h * 0.44
    chart_w = TWR_W - LBL_W - 2*CPAD

    for (tx, bg, lc, bc, side) in [
            (LC_TW, RED_LT, RED, RED_MK, "left"),
            (RC_TW, GRN_LT, GRN, GRN_MK, "right")]:

        box(tx, row_bot, TWR_W, row_h, bg, lc, lw=1.0, r=0.06)
        t(tx + LBL_W/2, y_ctr, f"L{l}",
          ha="center", va="center", fontsize=15, color=DARK, fontweight="bold")
        ax.plot([tx + LBL_W, tx + LBL_W],
                [row_bot + 0.04, row_bot + row_h - 0.04],
                color=lc, lw=0.8, alpha=0.4, zorder=6)

        cx0   = tx + LBL_W + CPAD
        pitch = chart_w / N_BARS
        bw    = pitch * (1 - BAR_GAP)

        np.random.seed(42 + li*97 + (0 if side == "left" else 1000))
        vals = np.random.randn(N_BARS) * 0.35
        pks  = np.random.choice(N_BARS, 18, replace=False)
        vals[pks] += np.random.choice([-1,1], 18) * (0.7 + np.random.rand(18)*0.3)
        vals = np.clip(vals, -1, 1)

        for bi, v in enumerate(vals):
            bx = cx0 + bi*pitch
            bh = abs(v) * max_bar
            by = y_base if v >= 0 else y_base - bh
            ax.add_patch(mpatches.Rectangle(
                (bx, by), bw, bh,
                fc=bc if v > 0 else "#BBBBBB",
                ec="none", alpha=0.88, zorder=7))

        ax.plot([cx0, cx0 + chart_w], [y_base, y_base],
                color=lc, lw=0.5, alpha=0.30, zorder=8)

        if side == "left":
            arrow(tx, y_ctr, tx - 0.55, y_ctr, RED, lw=1.8, ms=12)
        else:
            arrow(tx + TWR_W, y_ctr, tx + TWR_W + 0.55, y_ctr, GRN, lw=1.8, ms=12)

# ── 3D PCA vector illustration — centred in the middle gap ───────────────────
MID_CX = MID_X + MID_W / 2

# Load real activations — first 3 dims of mean commit/abstain at L28
_d      = torch.load("step1_extract_activations/data/activations_qwen_instruct.pt",
                     map_location="cpu", weights_only=False)
_layers = _d["layers"]
_l28    = _layers.index(28)
_mu_A   = _d["h_A"][:, _l28, :].mean(0).numpy()   # commit
_mu_B   = _d["h_B"][:, _l28, :].mean(0).numpy()   # abstain
_norm_A = float(np.linalg.norm(_mu_A))
_norm_B = float(np.linalg.norm(_mu_B))
_dist   = float(np.linalg.norm(_mu_A - _mu_B))

_r3_A = _mu_A[:3]; _r3_B = _mu_B[:3]

# 3-D projections (dims 1-3) — raw actual values
p_c  = _r3_A.copy()   # commit point
p_a  = _r3_B.copy()   # abstain point
_d3  = float(np.linalg.norm(p_c - p_a))

# Recompute best viewing angle for raw vectors
_best_sep, _best_az, _best_el = 0, 150, 30
for _az in range(-180, 180, 5):
    for _el in [15, 20, 25, 30]:
        _a, _b = np.radians(_az), np.radians(_el)
        def _proj(v):
            u  =  v[0]*np.cos(_a) - v[1]*np.sin(_a)
            vv = -(v[0]*np.sin(_a) + v[1]*np.cos(_a))*np.sin(_b) + v[2]*np.cos(_b)
            return np.array([u, vv])
        _pc2, _pa2 = _proj(p_c/np.linalg.norm(p_c)), _proj(p_a/np.linalg.norm(p_a))
        _cos = np.dot(_pc2, _pa2) / (np.linalg.norm(_pc2)*np.linalg.norm(_pa2) + 1e-9)
        _sep = np.degrees(np.arccos(np.clip(_cos, -1, 1)))
        if _sep > _best_sep:
            _best_sep, _best_az, _best_el = _sep, _az, _el

# Title using main 2D axes
t(MID_CX, ZC_TOP - 0.08,
  "Same question, same position (p−1)\n→ different hidden states at L28",
  ha="center", va="top", fontsize=14, color=DARK,
  fontweight="bold", linespacing=1.45, zorder=9)
t(MID_CX, ZC_TOP - 0.98,
  "dims 1–3 of 4096  (real data, L28)",
  ha="center", va="top", fontsize=11, color=GREY,
  style="italic", zorder=9)

# Embed 3D subplot (figure-fraction coordinates)
ax3_left = (MID_X + 0.30) / FW
ax3_bot  = (ZC_BOT + 0.70) / FH
ax3_w    = (MID_W - 0.60) / FW
ax3_h    = (H_LATE * 0.70) / FH
ax3 = fig.add_axes([ax3_left, ax3_bot, ax3_w, ax3_h], projection="3d")

# Arrows from origin to each dot
ax3.quiver(0, 0, 0, p_c[0], p_c[1], p_c[2],
           color=RED, linewidth=2.8, arrow_length_ratio=0.18, alpha=0.75)
ax3.quiver(0, 0, 0, p_a[0], p_a[1], p_a[2],
           color=GRN, linewidth=2.8, arrow_length_ratio=0.18, alpha=0.75)

# Origin dot + label
ax3.scatter([0], [0], [0], color="#444444", s=45, zorder=10, depthshade=False)
ax3.text(0.02, 0.02, -0.10, "(0, 0, 0)", color="#555555", fontsize=8)

# Big dots at each endpoint
ax3.scatter([p_c[0]], [p_c[1]], [p_c[2]], color=RED, s=180, zorder=10, depthshade=False)
ax3.scatter([p_a[0]], [p_a[1]], [p_a[2]], color=GRN, s=180, zorder=10, depthshade=False)

# Coordinate labels next to each dot
ax3.text(p_c[0] + 0.02, p_c[1] - 0.02, p_c[2] - 0.10,
         f"commit\n({p_c[0]:.2f}, {p_c[1]:.2f}, {p_c[2]:.2f})",
         color=RED, fontsize=8.5, fontweight="bold", ha="left", va="top")
ax3.text(p_a[0] + 0.02, p_a[1] - 0.02, p_a[2] + 0.04,
         f"abstain\n({p_a[0]:.2f}, {p_a[1]:.2f}, {p_a[2]:.2f})",
         color=GRN, fontsize=8.5, fontweight="bold", ha="left", va="bottom")

# Dashed distance line between the two dots + label at midpoint
_mid = (p_c + p_a) / 2
ax3.plot([p_c[0], p_a[0]], [p_c[1], p_a[1]], [p_c[2], p_a[2]],
         color=PUR, lw=0.9, ls="--", alpha=0.6, zorder=8)
ax3.text(_mid[0] + 0.03, _mid[1], _mid[2],
         f"d = {_d3:.2f}", color=PUR, fontsize=10, fontweight="bold")

# Arc between the two vectors + angle label
_u1  = p_c / np.linalg.norm(p_c)
_u2t = p_a - np.dot(p_a, _u1) * _u1
_u2  = _u2t / np.linalg.norm(_u2t)
_cos_ang = float(np.dot(p_c, p_a) / (np.linalg.norm(p_c) * np.linalg.norm(p_a)))
_ang_rad  = np.arccos(np.clip(_cos_ang, -1, 1))
_ang_deg  = np.degrees(_ang_rad)
_arc_r    = min(np.linalg.norm(p_c), np.linalg.norm(p_a)) * 0.55
_arc_ts   = np.linspace(0, _ang_rad, 40)
_arc_pts  = _arc_r * (np.outer(np.cos(_arc_ts), _u1) + np.outer(np.sin(_arc_ts), _u2))
ax3.plot(_arc_pts[:, 0], _arc_pts[:, 1], _arc_pts[:, 2],
         color=PUR, lw=1.8, alpha=0.7, zorder=9)
_mp = _arc_pts[len(_arc_pts)//2] * 1.6
ax3.text(_mp[0], _mp[1], _mp[2],
         f"≈{_ang_deg:.0f}°", color=PUR, fontsize=10, fontweight="bold", ha="center")

# Axis style
ax3.set_xlabel("dim 1", fontsize=10, color="#888888", labelpad=1)
ax3.set_ylabel("dim 2", fontsize=10, color="#888888", labelpad=1)
ax3.set_zlabel("dim 3", fontsize=10, color="#888888", labelpad=1)
ax3.set_xlim(-0.05, max(p_c[0], p_a[0]) + 0.25)
ax3.set_ylim(min(p_c[1], p_a[1]) - 0.12, 0.15)
ax3.set_zlim(-0.10, max(p_c[2], p_a[2]) + 0.12)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("#DDDDEE")
ax3.yaxis.pane.set_edgecolor("#DDDDEE")
ax3.zaxis.pane.set_edgecolor("#DDDDEE")
ax3.set_facecolor("#F5F5FC")
ax3.grid(True, alpha=0.25, color="#CCCCDD")
ax3.view_init(elev=_best_el, azim=_best_az)

# Distance annotation below the 3D plot (back on main 2D axes)
t(MID_CX, ZC_BOT + 0.55,
  f"‖commit‖={_norm_A:.1f}   ‖abstain‖={_norm_B:.1f}\ndistance={_dist:.1f}  ({100*_dist/_norm_A:.0f}% of ‖commit‖)",
  ha="center", va="top", fontsize=12, color=GREY,
  linespacing=1.4, zorder=9)

# ─────────────────────────────────────────────────────────────────────────────
# ZONE D — Result boxes + key insight
# ─────────────────────────────────────────────────────────────────────────────
arrow(LC_TW + TWR_W/2, ZC_BOT - 0.10, LC_TW + TWR_W/2, RES_TOP, RED, lw=2.5)
arrow(RC_TW + TWR_W/2, ZC_BOT - 0.10, RC_TW + TWR_W/2, RES_TOP, GRN, lw=2.5)

box(LC_X + 0.18, RES_Y, COL_W - 0.36, H_RES, RED_LT, RED, lw=2.5, r=0.15)
t(LC_X + COL_W/2, RES_Y + H_RES*0.67, "COMMIT STATE",
  ha="center", va="center", fontsize=19, color=RED, fontweight="bold")
t(LC_X + COL_W/2, RES_Y + H_RES*0.25, "8 vectors  ×  4096 dims  (L24 → L31)",
  ha="center", va="center", fontsize=13, color=DARK)

box(RC_X + 0.18, RES_Y, COL_W - 0.36, H_RES, GRN_LT, GRN, lw=2.5, r=0.15)
t(RC_X + COL_W/2, RES_Y + H_RES*0.67, "ABSTAIN STATE",
  ha="center", va="center", fontsize=19, color=GRN, fontweight="bold")
t(RC_X + COL_W/2, RES_Y + H_RES*0.25, "8 vectors  ×  4096 dims  (L24 → L31)",
  ha="center", va="center", fontsize=13, color=DARK)

H_INS = 0.90   # taller to fit two lines comfortably
box(6.5, INSIGHT_Y, 11.0, H_INS, "#FEF9E7", ORG, lw=2.2, r=0.06)
t(FW/2, INSIGHT_Y + H_INS*0.78, "Key insight:",
  ha="center", va="center", fontsize=15, color=ORG, fontweight="bold")
t(FW/2, INSIGHT_Y + H_INS*0.35,
  f"Same question  →  two very different vectors at L28\n"
  f"‖commit‖ = {_norm_A:.1f}     ‖abstain‖ = {_norm_B:.1f}     distance = {_dist:.1f}  ({100*_dist/_norm_A:.0f}% of ‖commit‖)",
  ha="center", va="center", fontsize=14, color=DARK, linespacing=1.6)

# ─────────────────────────────────────────────────────────────────────────────
fig.savefig(OUT / "arch1_transformer_diagram.png", bbox_inches="tight", dpi=160)
plt.close(fig)
print("✓ arch1_transformer_diagram.png")
