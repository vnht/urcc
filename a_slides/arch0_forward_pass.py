"""
arch0_forward_pass.py
"How a Transformer Produces Hidden State Vectors"

Shows:
  - Prompt tokens enter from the TOP
  - Answer tokens appear at the BOTTOM (they are also fed in, just shown separately)
  - Layer-by-layer processing (L0 → L31) in between
  - We read the vector at p-1 (last prompt token) × late layers (L24-L31)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("a_slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
BLUE   = "#1060C0";  BLU_LT = "#BBDEFB"
ORG    = "#E65000";  ORG_LT = "#FFD9B0"
RED    = "#C62828"
PUR    = "#6A1B9A";  PUR_LT = "#DDB8F0"
DARK   = "#1C1C2E"
GREY   = "#666666"
LGREY  = "#EBEBEB"
WHITE  = "#FFFFFF"

FW, FH = 24, 14
fig = plt.figure(figsize=(FW, FH), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

def box(x, y, w, h, fc, ec, lw=1.5, r=0.08, alpha=1.0, z=2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))

def arr(x0, y0, x1, y1, c, lw=2.0, ms=13, ls="solid"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                linestyle=ls, mutation_scale=ms), zorder=8)

def t(x, y, s, **kw):
    ax.text(x, y, s, **kw)

# ─────────────────────────────────────────────────────────────────────────────
# TOKENS TO DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
# kind: "prompt" | "p1" (last prompt token) | "answer"
tokens = [
    ("Which",     "prompt"),
    ("language",  "prompt"),
    ("···",       "prompt"),
    ("continent", "prompt"),
    ("?",         "p1"),      # ← p-1: last prompt token
    ("The",       "answer"),
    ("most",      "answer"),
    ("popular",   "answer"),
    ("Swahili",   "answer"),
    ("···",       "answer"),
]
N_TOK  = len(tokens)
P1_IDX = 4

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
LBL_W  = 1.65           # left strip for layer labels
GRID_X = LBL_W + 0.25   # left edge of cell grid
GRID_W = 16.5            # total grid width
TOK_W  = GRID_W / N_TOK # per-token column width  = 1.65
RPN_X  = GRID_X + GRID_W + 0.45   # right panel x
RPN_W  = FW - RPN_X - 0.20         # right panel width ≈ 5.1

# Vertical
TITLE_Y   = 13.65
TOK_Y     = 12.25          # bottom of TOP token row (prompt + p-1)
TOK_H     = 0.72
GRID_TOP  = 11.25          # top of layer grid
GRID_BOT  = 3.10           # bottom of layer grid (raised to make room below)
BOT_TOK_Y = 2.10           # bottom of BOTTOM token row (answer tokens)
BOT_TOK_H = 0.72

# Row heights (must fill GRID_TOP - GRID_BOT exactly)
H_THIN  = 0.40
H_BREAK = 0.45
H_EMBED = 0.52
remaining = (GRID_TOP - GRID_BOT) - 2*H_EMBED - 3*H_THIN - H_BREAK - 2*H_THIN
H_THICK = remaining / 8

# Build row y-positions (from top down)
EARLY_SHOW  = [0, 1, 2]
LATE_EARLY  = [22, 23]
LATE_LAYERS = [24, 25, 26, 27, 28, 29, 30, 31]

row_ys = {}   # layer → (y_bottom, height)
y = GRID_TOP

# Top embed row (above L0)
embed_y = y - H_EMBED; y -= H_EMBED

for l in EARLY_SHOW:
    row_ys[l] = (y - H_THIN, H_THIN); y -= H_THIN
break_top = y; y -= H_BREAK
for l in LATE_EARLY:
    row_ys[l] = (y - H_THIN, H_THIN); y -= H_THIN
for l in LATE_LAYERS:
    row_ys[l] = (y - H_THICK, H_THICK); y -= H_THICK

# Bottom embed row (below L31)
embed_bot_y = y - H_EMBED; y -= H_EMBED

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
t(FW/2, TITLE_Y, "How the Transformer Produces Hidden State Vectors",
  ha="center", va="center", fontsize=20, fontweight="bold", color=DARK,
  bbox=dict(boxstyle="round,pad=0.38", fc=WHITE, ec=DARK, lw=2.2), zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# TOP TOKEN ROW — prompt tokens + p-1
# ─────────────────────────────────────────────────────────────────────────────
t(0.25, TOK_Y + TOK_H/2, "Prompt\ntokens:",
  ha="left", va="center", fontsize=13, color=DARK, fontweight="bold")

for i, (tok, kind) in enumerate(tokens):
    if kind == "answer":
        continue
    tx = GRID_X + i * TOK_W
    fc = BLU_LT if kind == "prompt" else ORG_LT
    ec = BLUE   if kind == "prompt" else ORG
    lw = 2.4 if kind == "p1" else 1.4
    box(tx + 0.06, TOK_Y, TOK_W - 0.12, TOK_H, fc, ec, lw=lw, r=0.07)
    t(tx + TOK_W/2, TOK_Y + TOK_H/2, tok,
      ha="center", va="center", fontsize=12, color=ec,
      fontweight="bold" if kind == "p1" else "normal")

# "← prompt tokens →" label under top row
p_end = GRID_X + P1_IDX * TOK_W
t((GRID_X + p_end)/2, TOK_Y - 0.15, "← prompt tokens →",
  ha="center", va="top", fontsize=11, color=BLUE)

# p-1 callout above top-row token
p1_cx = GRID_X + P1_IDX * TOK_W + TOK_W/2
t(p1_cx, TOK_Y + TOK_H + 0.10,
  "p−1\n(last prompt\ntoken)",
  ha="center", va="bottom", fontsize=11, color=ORG, fontweight="bold",
  bbox=dict(boxstyle="round,pad=0.18", fc=ORG_LT, ec=ORG, lw=1.5))

# Arrow: p-1 top callout → top of grid
arr(p1_cx, TOK_Y, p1_cx, GRID_TOP, ORG, lw=1.8, ms=11)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM TOKEN ROW — answer tokens
# ─────────────────────────────────────────────────────────────────────────────
t(0.25, BOT_TOK_Y + BOT_TOK_H/2, "Answer\ntokens:",
  ha="left", va="center", fontsize=13, color=DARK, fontweight="bold")

for i, (tok, kind) in enumerate(tokens):
    if kind != "answer":
        continue
    tx = GRID_X + i * TOK_W
    box(tx + 0.06, BOT_TOK_Y, TOK_W - 0.12, BOT_TOK_H,
        LGREY, GREY, lw=1.4, r=0.07)
    t(tx + TOK_W/2, BOT_TOK_Y + BOT_TOK_H/2, tok,
      ha="center", va="center", fontsize=12, color=GREY)

# "← answer tokens →" label under the bottom row
a_start = GRID_X + (P1_IDX + 1) * TOK_W
a_cx = (a_start + GRID_X + GRID_W) / 2
t(a_cx, BOT_TOK_Y - 0.15, "← answer tokens →",
  ha="center", va="top", fontsize=11, color=GREY)

# Dashed connectors from GRID_BOT down to each answer token box
for i, (tok, kind) in enumerate(tokens):
    if kind != "answer" or tok == "···":
        continue
    col_cx = GRID_X + i * TOK_W + TOK_W/2
    ax.plot([col_cx, col_cx], [GRID_BOT, BOT_TOK_Y + BOT_TOK_H],
            color=GREY, lw=0.8, ls="dashed", alpha=0.4, zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# VERTICAL HIGHLIGHT STRIP for p-1 column
# ─────────────────────────────────────────────────────────────────────────────
p1_x = GRID_X + P1_IDX * TOK_W
ax.add_patch(mpatches.Rectangle(
    (p1_x + 0.05, GRID_BOT), TOK_W - 0.10, GRID_TOP - GRID_BOT,
    fc=ORG_LT, ec="none", alpha=0.30, zorder=1))
for xv in [p1_x + 0.05, p1_x + TOK_W - 0.05]:
    ax.plot([xv, xv], [GRID_BOT, GRID_TOP],
            color=ORG, lw=1.0, alpha=0.45, zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: draw activation bar chart inside a cell
# ─────────────────────────────────────────────────────────────────────────────
def barchart(x0, y_bot, w, h, color, seed):
    n = 60; pitch = w / n; bw = pitch * 0.85
    max_h = h * 0.42
    np.random.seed(seed)
    vals = np.random.randn(n) * 0.32
    pks  = np.random.choice(n, 10, replace=False)
    vals[pks] += np.random.choice([-1,1], 10) * (0.65 + np.random.rand(10)*0.30)
    vals = np.clip(vals, -1, 1)
    yb = y_bot + h * 0.10
    for bi, v in enumerate(vals):
        bh = abs(v) * max_h
        by = yb if v >= 0 else yb - bh
        ax.add_patch(mpatches.Rectangle(
            (x0 + bi*pitch, by), bw, bh,
            fc=color if v > 0 else "#BBBBBB", ec="none", alpha=0.88, zorder=7))
    ax.plot([x0, x0+w], [yb, yb], color=color, lw=0.5, alpha=0.30, zorder=8)

# ─────────────────────────────────────────────────────────────────────────────
# EMBED ROW
# ─────────────────────────────────────────────────────────────────────────────
GRNN    = "#2E7D32";  GRNN_LT = "#C8E6C9"

embed_top_w = (P1_IDX + 1) * TOK_W   # prompt + p-1 columns only
ax.add_patch(mpatches.Rectangle(
    (GRID_X, embed_y + H_EMBED*0.06), embed_top_w, H_EMBED * 0.88,
    fc="#F1F8E9", ec="#A5D6A7", lw=0.9, zorder=2))
t(LBL_W - 0.12, embed_y + H_EMBED*0.50, "Embed",
  ha="right", va="center", fontsize=12, color=GRNN, fontweight="bold")

for i, (tok, kind) in enumerate(tokens):
    if kind == "answer" or tok == "···": continue
    tx  = GRID_X + i * TOK_W
    cw  = TOK_W - 0.14; cx = tx + 0.07
    ch  = H_EMBED * 0.80; cy = embed_y + H_EMBED * 0.10
    ec  = ORG   if kind == "p1" else GRNN
    lw  = 2.0   if kind == "p1" else 1.0
    barchart(cx, cy, cw, ch, ec, seed=99 + i)
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - 0.03, cy - 0.02), cw + 0.06, ch + 0.04,
        boxstyle="round,pad=0.03", fc="none", ec=ec, lw=lw, zorder=8))

# ─────────────────────────────────────────────────────────────────────────────
# EARLY LAYER ROWS (L0, L1, L2 + break + L22, L23)
# ─────────────────────────────────────────────────────────────────────────────
for l in EARLY_SHOW + LATE_EARLY:
    ry, rh = row_ys[l]
    ax.add_patch(mpatches.Rectangle(
        (GRID_X, ry + rh*0.06), GRID_W, rh * 0.86,
        fc="#F8FBFF", ec="#C8DAEA", lw=0.6, zorder=2))
    t(LBL_W - 0.12, ry + rh*0.50, f"L{l}",
      ha="right", va="center", fontsize=13, color=DARK, fontweight="bold")
    for i, (tok, kind) in enumerate(tokens):
        tx = GRID_X + i * TOK_W
        cw = TOK_W - 0.14; cx = tx + 0.07; ch = rh * 0.80; cy = ry + rh * 0.10
        if kind == "p1":
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, cy), cw, ch, boxstyle="round,pad=0.03",
                fc=ORG_LT, ec=ORG, lw=1.4, zorder=4))
            t(cx + cw/2, cy + ch/2, "h",
              ha="center", va="center", fontsize=10,
              color=ORG, fontweight="bold", style="italic", zorder=5)
        elif tok != "···":
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, cy), cw, ch, boxstyle="round,pad=0.03",
                fc=LGREY, ec="#CCCCCC", lw=0.6, zorder=3))

# break row
ax.add_patch(mpatches.Rectangle(
    (GRID_X, break_top - H_BREAK + H_BREAK*0.15), GRID_W, H_BREAK * 0.70,
    fc="#F5F5F5", ec="#DDDDDD", lw=0.4, zorder=2))
t(LBL_W - 0.12, break_top - H_BREAK/2, "···",
  ha="right", va="center", fontsize=15, color=GREY)
t(GRID_X + GRID_W/2, break_top - H_BREAK/2, "···",
  ha="center", va="center", fontsize=15, color=GREY)

# ─────────────────────────────────────────────────────────────────────────────
# LATE LAYER ROWS (L24-L31) with bar charts
# ─────────────────────────────────────────────────────────────────────────────
for li, l in enumerate(LATE_LAYERS):
    ry, rh = row_ys[l]
    ax.add_patch(mpatches.Rectangle(
        (GRID_X, ry + rh*0.04), GRID_W, rh * 0.91,
        fc="#FFF8F8", ec="#FFB8B8", lw=0.7, zorder=2))
    t(LBL_W - 0.12, ry + rh*0.50, f"L{l}",
      ha="right", va="center", fontsize=13, color=RED, fontweight="bold")
    for i, (tok, kind) in enumerate(tokens):
        if tok == "···": continue
        tx   = GRID_X + i * TOK_W
        cw   = TOK_W - 0.14; cx = tx + 0.07
        ch   = rh * 0.85;    cy = ry + rh * 0.07
        c    = ORG if kind == "p1" else "#AAAAAA"
        barchart(cx, cy, cw, ch, c, seed=li*17 + i)
        if kind == "p1":
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - 0.04, cy - 0.02), cw + 0.08, ch + 0.04,
                boxstyle="round,pad=0.03", fc="none", ec=ORG,
                lw=2.4, zorder=8))

# Arrow from p-1 column to right panel (mid-height of late layers)
l_mid = LATE_LAYERS[len(LATE_LAYERS)//2]
lm_ry, lm_rh = row_ys[l_mid]
lm_cy = lm_ry + lm_rh/2
arr(p1_x + TOK_W + 0.10, lm_cy, RPN_X - 0.10, lm_cy, ORG, lw=2.2)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM EMBED ROW (answer token embeddings, below L31)
# ─────────────────────────────────────────────────────────────────────────────
embed_bot_x = GRID_X + (P1_IDX + 1) * TOK_W   # answer columns only
embed_bot_w = GRID_W - (P1_IDX + 1) * TOK_W
ax.add_patch(mpatches.Rectangle(
    (embed_bot_x, embed_bot_y + H_EMBED*0.06), embed_bot_w, H_EMBED * 0.88,
    fc="#F1F8E9", ec="#A5D6A7", lw=0.9, zorder=2))
t(LBL_W - 0.12, embed_bot_y + H_EMBED*0.50, "Embed",
  ha="right", va="center", fontsize=12, color=GRNN, fontweight="bold")

for i, (tok, kind) in enumerate(tokens):
    if kind not in ("answer",) or tok == "···": continue
    tx  = GRID_X + i * TOK_W
    cw  = TOK_W - 0.14; cx = tx + 0.07
    ch  = H_EMBED * 0.80; cy = embed_bot_y + H_EMBED * 0.10
    barchart(cx, cy, cw, ch, GRNN, seed=200 + i)
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - 0.03, cy - 0.02), cw + 0.06, ch + 0.04,
        boxstyle="round,pad=0.03", fc="none", ec=GRNN, lw=1.0, zorder=8))

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — "What we get"
# ─────────────────────────────────────────────────────────────────────────────
rp_bot = GRID_BOT + 0.4
rp_top = GRID_TOP - 0.4
box(RPN_X, rp_bot, RPN_W - 0.20, rp_top - rp_bot, PUR_LT, PUR, lw=2.2, r=0.16)

t(RPN_X + (RPN_W-0.20)/2, rp_top - 0.10,
  "What we get",
  ha="center", va="top", fontsize=15, color=PUR, fontweight="bold")

t(RPN_X + (RPN_W-0.20)/2, rp_top - 0.55,
  "At each late layer:\none 4096-dim vector\nat position p−1",
  ha="center", va="top", fontsize=13, color=DARK, linespacing=1.50)

va_y = (rp_bot + rp_top) / 2
ax.annotate("", xy=(RPN_X + RPN_W - 0.50, va_y),
            xytext=(RPN_X + 0.25, va_y),
            arrowprops=dict(arrowstyle="-|>", color=PUR, lw=3.2,
                            mutation_scale=20), zorder=5)
t(RPN_X + (RPN_W-0.20)/2, va_y - 0.30,
  "[ 2.3,  −0.7,  1.1,  … 4096 ]",
  ha="center", fontsize=11, color=PUR)

box(RPN_X + 0.20, rp_bot + 0.20, RPN_W - 0.60, 1.40,
    WHITE, PUR, lw=1.6, r=0.10, z=4)
t(RPN_X + (RPN_W-0.20)/2, rp_bot + 1.52,
  "8 late layers",
  ha="center", va="center", fontsize=13, color=PUR, fontweight="bold", zorder=5)
t(RPN_X + (RPN_W-0.20)/2, rp_bot + 1.08,
  "×  4096 dims",
  ha="center", va="center", fontsize=13, color=DARK, zorder=5)
t(RPN_X + (RPN_W-0.20)/2, rp_bot + 0.60,
  "=  8 vectors",
  ha="center", va="center", fontsize=14, color=PUR,
  fontweight="bold", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM RESULT STRIP
# ─────────────────────────────────────────────────────────────────────────────
box(2.5, 0.10, 19.0, 0.85, ORG_LT, ORG, lw=2.0, r=0.07)
t(FW/2, 0.10 + 0.62, "Key point:",
  ha="center", va="center", fontsize=13, color=ORG, fontweight="bold")
t(FW/2, 0.10 + 0.25,
  "We concatenate (prompt + answer) and run ONE forward pass.  "
  "At each late layer, we read off the 4096-dim vector at p−1  "
  "→  this is the model's internal state when processing that answer.",
  ha="center", va="center", fontsize=12, color=DARK)

# ─────────────────────────────────────────────────────────────────────────────
fig.savefig(OUT / "arch0_forward_pass.png", bbox_inches="tight", dpi=160)
plt.close(fig)
print("✓ arch0_forward_pass.png")
