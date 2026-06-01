"""
arch2_kuq_scatter3d.py
"1000 KUQ instances — commit vs abstain in 3D (dims 1–3, L28)"

Shows 1000 paired commit / abstain hidden-state vectors as scatter dots in
the raw first 3 dimensions at layer L28.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path

OUT = Path("a_slides/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
RED   = "#C0392B";  RED_LT  = "#FDEDEC"
GRN   = "#1E8449";  GRN_LT  = "#EAFAF1"
PUR   = "#6A1B9A"
DARK  = "#1C1C2E"
GREY  = "#888888"

# ── load data ─────────────────────────────────────────────────────────────────
acts = torch.load(
    "step1_extract_activations/data/activations_qwen_instruct.pt",
    map_location="cpu", weights_only=False)

LAYER_IDX = acts["layers"].index(28)
meta      = acts["meta_A"]
kuq_idx   = [i for i, m in enumerate(meta) if m["dataset"] == "kuq"]

h_A = acts["h_A"][kuq_idx, LAYER_IDX, :].numpy()   # (1000, 4096) commit
h_B = acts["h_B"][kuq_idx, LAYER_IDX, :].numpy()   # (1000, 4096) abstain

# First 3 dims only
A3 = h_A[:, :3]   # (1000, 3)
B3 = h_B[:, :3]

mu_A = A3.mean(0)
mu_B = B3.mean(0)
_dist_mean = float(np.linalg.norm(mu_A - mu_B))

# Fixed viewing angle — matches arch1_transformer_diagram.py
best_el, best_az = 30, 150

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor="white")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#F8F8FC")

# Scatter — all 1000 instances
ax.scatter(A3[:, 0], A3[:, 1], A3[:, 2],
           c=RED, s=18, alpha=0.30, label="overcommit (1000)", depthshade=True,
           linewidths=0)
ax.scatter(B3[:, 0], B3[:, 1], B3[:, 2],
           c=GRN, s=18, alpha=0.30, label="legit abstain (1000)", depthshade=True,
           linewidths=0)

# Mean dots + arrows from origin
ax.scatter([mu_A[0]], [mu_A[1]], [mu_A[2]], c=RED, s=180, zorder=10,
           depthshade=False, edgecolors="white", linewidths=1.0)
ax.scatter([mu_B[0]], [mu_B[1]], [mu_B[2]], c=GRN, s=180, zorder=10,
           depthshade=False, edgecolors="white", linewidths=1.0)

ax.quiver(0, 0, 0, mu_A[0], mu_A[1], mu_A[2],
          color=RED, linewidth=2.2, arrow_length_ratio=0.12, alpha=0.85)
ax.quiver(0, 0, 0, mu_B[0], mu_B[1], mu_B[2],
          color=GRN, linewidth=2.2, arrow_length_ratio=0.12, alpha=0.85)

# Origin dot
ax.scatter([0], [0], [0], c="#444444", s=40, zorder=10, depthshade=False)
ax.text(0.05, 0.05, -0.20, "(0, 0, 0)", color="#555555", fontsize=8)

# Mean coordinate labels
ax.text(mu_A[0] + 0.05, mu_A[1] - 0.05, mu_A[2] - 0.25,
        f"mean overcommit\n({mu_A[0]:.2f}, {mu_A[1]:.2f}, {mu_A[2]:.2f})",
        color=RED, fontsize=9, fontweight="bold", va="top")
ax.text(mu_B[0] + 0.05, mu_B[1] - 0.05, mu_B[2] + 0.10,
        f"mean legit abstain\n({mu_B[0]:.2f}, {mu_B[1]:.2f}, {mu_B[2]:.2f})",
        color=GRN, fontsize=9, fontweight="bold", va="bottom")

# Distance line between means
_mid = (mu_A + mu_B) / 2
ax.plot([mu_A[0], mu_B[0]], [mu_A[1], mu_B[1]], [mu_A[2], mu_B[2]],
        color=PUR, lw=0.9, ls="--", alpha=0.6)
ax.text(_mid[0] + 0.05, _mid[1], _mid[2],
        f"d = {_dist_mean:.2f}", color=PUR, fontsize=10, fontweight="bold")

# Angle arc between mean vectors
_u1  = mu_A / np.linalg.norm(mu_A)
_u2t = mu_B - np.dot(mu_B, _u1) * _u1
_u2  = _u2t / np.linalg.norm(_u2t)
_cos_ang = float(np.dot(mu_A, mu_B) / (np.linalg.norm(mu_A)*np.linalg.norm(mu_B)))
_ang_rad  = np.arccos(np.clip(_cos_ang, -1, 1))
_ang_deg  = np.degrees(_ang_rad)
_arc_r    = min(np.linalg.norm(mu_A), np.linalg.norm(mu_B)) * 0.50
_arc_ts   = np.linspace(0, _ang_rad, 40)
_arc_pts  = _arc_r * (np.outer(np.cos(_arc_ts), _u1) + np.outer(np.sin(_arc_ts), _u2))
ax.plot(_arc_pts[:, 0], _arc_pts[:, 1], _arc_pts[:, 2],
        color=PUR, lw=1.8, alpha=0.7)
_mp = _arc_pts[len(_arc_pts)//2] * 1.7
ax.text(_mp[0], _mp[1], _mp[2],
        f"≈{_ang_deg:.0f}°", color=PUR, fontsize=11, fontweight="bold", ha="center")

# Legend
ax.legend(loc="upper left", fontsize=11, framealpha=0.85,
          markerscale=1.8, handlelength=1.0)

# Axes
ax.set_xlabel("dim 1", fontsize=11, color=GREY, labelpad=4)
ax.set_ylabel("dim 2", fontsize=11, color=GREY, labelpad=4)
ax.set_zlabel("dim 3", fontsize=11, color=GREY, labelpad=4)
ax.set_xlim(-3, 3); ax.set_ylim(-4.5, 3); ax.set_zlim(-2, 3)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#DDDDEE")
ax.yaxis.pane.set_edgecolor("#DDDDEE")
ax.zaxis.pane.set_edgecolor("#DDDDEE")
ax.grid(True, alpha=0.20, color="#CCCCDD")
ax.view_init(elev=best_el, azim=best_az)

# Title
ax.set_title(
    "1000 KUQ instances — commit vs abstain\n"
    "hidden state at position p−1, layer L28  (dims 1–3 of 4096)",
    fontsize=13, color=DARK, fontweight="bold", pad=12, linespacing=1.5)

fig.tight_layout()
fig.savefig(OUT / "arch2_kuq_scatter3d.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("✓ arch2_kuq_scatter3d.png")
