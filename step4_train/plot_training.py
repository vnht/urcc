"""Plot UOC training progress from a run directory.

Reads the CSV columns produced by `step4_train.py`:

    step, L_total, L_forget, L_retain, mean_proj_norm, learning_rate, grad_norm

Run
---
    python step4_train/plot_training.py step4_train/data/runs/<run_name>
    python step4_train/plot_training.py step4_train/data/runs/<run_name> --log-loss
    python step4_train/plot_training.py step4_train/data/runs/<run_name> --out custom.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_loss_log(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        for c in reader.fieldnames:
            cols[c] = []
        for row in reader:
            for c, v in row.items():
                if v == "" or v is None:
                    continue
                try:
                    cols[c].append(float(v))
                except ValueError:
                    cols[c].append(float("nan"))
    return cols


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def plot_run(run_dir: Path, out_path: Path | None = None,
             log_loss: bool = False) -> Path:
    csv_path = run_dir / "loss_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"loss_log.csv not found in {run_dir}")
    data = _load_loss_log(csv_path)
    train_cfg = _load_json(run_dir / "training_config.json")
    summary   = _load_json(run_dir / "train_summary.json")

    steps = data.get("step", [])
    if not steps:
        raise ValueError(f"No 'step' column in {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    title = run_dir.name
    if train_cfg is not None:
        title = (
            f"{run_dir.name}\n"
            f"{train_cfg.get('method','UOC')}  |  "
            f"r={train_cfg.get('subspace_rank','?')}  "
            f"λ={train_cfg.get('lambda_retain','?')}  "
            f"lr={train_cfg.get('lr','?')}  "
            f"epochs={train_cfg.get('epochs','?')}"
        )
    fig.suptitle(title, fontsize=12)

    color_total  = "tab:blue"
    color_forget = "tab:orange"
    color_retain = "tab:green"

    ax_total = axes[0, 0]
    if "L_total" in data:
        ax_total.plot(steps, data["L_total"], color=color_total,
                      linewidth=1.5, label="L_total")
    ax_total.set_title("Total loss  (L_forget + λ·L_retain)")
    ax_total.set_ylabel("Loss")
    ax_total.grid(True, alpha=0.3)
    ax_total.legend(loc="best")
    if log_loss:
        ax_total.set_yscale("log")

    ax_terms = axes[0, 1]
    if "L_forget" in data:
        ax_terms.plot(steps, data["L_forget"], color=color_forget,
                      linewidth=1.5, label="L_forget")
    if "L_retain" in data:
        ax_terms.plot(steps, data["L_retain"], color=color_retain,
                      linewidth=1.5, label="L_retain")
    ax_terms.set_title("Forget vs retain")
    ax_terms.set_ylabel("Loss")
    ax_terms.grid(True, alpha=0.3)
    ax_terms.legend(loc="best")
    if log_loss:
        ax_terms.set_yscale("log")

    ax_proj = axes[1, 0]
    if "mean_proj_norm" in data:
        ax_proj.plot(steps, data["mean_proj_norm"], color="tab:red",
                     linewidth=1.4, label="mean ‖Vᵀ(h-μ⁻)‖")
    ax_proj.set_title("Mean projection norm (forget)")
    ax_proj.set_ylabel("‖Vᵀ(h-μ⁻)‖")
    ax_proj.set_xlabel("Step")
    ax_proj.grid(True, alpha=0.3)
    ax_proj.legend(loc="best")

    ax_aux = axes[1, 1]
    plotted = False
    if "grad_norm" in data:
        ax_aux.plot(steps, data["grad_norm"], color="tab:purple",
                    linewidth=1.0, alpha=0.8, label="grad norm")
        ax_aux.set_ylabel("Grad norm", color="tab:purple")
        ax_aux.tick_params(axis="y", labelcolor="tab:purple")
        plotted = True
    if "learning_rate" in data:
        ax_lr = ax_aux.twinx() if plotted else ax_aux
        ax_lr.plot(steps, data["learning_rate"], color="tab:gray",
                   linewidth=1.2, linestyle="--", label="learning rate")
        ax_lr.set_ylabel("Learning rate", color="tab:gray")
        ax_lr.tick_params(axis="y", labelcolor="tab:gray")
    ax_aux.set_title("Grad norm & learning rate")
    ax_aux.set_xlabel("Step")
    ax_aux.grid(True, alpha=0.3)

    if summary is not None:
        info_lines = [f"steps: {summary.get('num_steps', '?')}"]
        for k_init, k_final in [("initial_L_forget", "final_L_forget"),
                                 ("initial_L_retain", "final_L_retain"),
                                 ("initial_mean_proj_norm",
                                  "final_mean_proj_norm")]:
            if k_init in summary and k_final in summary:
                info_lines.append(
                    f"{k_init.replace('initial_', '')}:"
                    f" {summary[k_init]:.3f} → {summary[k_final]:.3f}"
                )
        fig.text(0.99, 0.01, "\n".join(info_lines), ha="right", va="bottom",
                 fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor="lightgray", alpha=0.85))

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    if out_path is None:
        out_path = run_dir / "training_progress.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--log-loss", action="store_true")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    out = plot_run(run_dir, out_path=args.out, log_loss=args.log_loss)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
