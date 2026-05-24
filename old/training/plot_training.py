"""Plot training progress (loss curves, KL, projection norm, LR, grad norm) from a run directory.

Reads ``loss_log.csv`` (and optionally ``train_summary.json`` / ``training_config.json``)
inside a training run directory and writes ``training_progress.png`` next to it.

Usage:
    python training/plot_training.py training/runs/<run_name>
    python training/plot_training.py training/runs/<run_name> --out custom_name.png
    python training/plot_training.py training/runs/<run_name> --log-loss
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_loss_log(csv_path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")
        for col in reader.fieldnames:
            columns[col] = []
        for row in reader:
            for col, val in row.items():
                if val == "" or val is None:
                    continue
                try:
                    columns[col].append(float(val))
                except ValueError:
                    columns[col].append(float("nan"))
    return columns


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def plot_run(run_dir: Path, out_path: Path | None = None, log_loss: bool = False) -> Path:
    csv_path = run_dir / "loss_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"loss_log.csv not found in {run_dir}")

    data = load_loss_log(csv_path)
    summary = load_json_if_exists(run_dir / "train_summary.json")
    config = load_json_if_exists(run_dir / "training_config.json")

    steps = data.get("step", [])
    if not steps:
        raise ValueError(f"No 'step' column found in {csv_path}")

    beta = float(config["beta"]) if config is not None and "beta" in config else 1.0

    # Total loss = L_forget + beta * top100_retain_KL (the actual term that gets backprop'd)
    retain_loss: list[float] | None = None
    if "top100_retain_KL" in data:
        retain_loss = [beta * v for v in data["top100_retain_KL"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    title = run_dir.name
    if config is not None:
        method = config.get("method", "")
        lr = config.get("learning_rate", "")
        epochs = config.get("epochs", "")
        title = f"{run_dir.name}\n{method}  |  β={beta}  lr={lr}  epochs={epochs}"
    fig.suptitle(title, fontsize=12)

    color_total = "tab:blue"
    color_forget = "tab:orange"
    color_retain = "tab:green"

    ax_total = axes[0, 0]
    if "L_total" in data:
        ax_total.plot(steps, data["L_total"], color=color_total, linewidth=1.5, label="L_total")
    ax_total.set_title("Total loss  (L_forget + β·KL)")
    ax_total.set_ylabel("Loss")
    ax_total.grid(True, alpha=0.3)
    ax_total.legend(loc="best")
    if log_loss:
        ax_total.set_yscale("log")

    ax_terms = axes[0, 1]
    if "L_forget" in data:
        ax_terms.plot(
            steps, data["L_forget"], color=color_forget, linewidth=1.5, label="L_forget"
        )
    if retain_loss is not None:
        ax_terms.plot(
            steps,
            retain_loss,
            color=color_retain,
            linewidth=1.5,
            label=f"β·KL  (retain, β={beta:g})",
        )
    ax_terms.set_title("Forget vs retain loss")
    ax_terms.set_ylabel("Loss")
    ax_terms.grid(True, alpha=0.3)
    ax_terms.legend(loc="best")
    if log_loss:
        ax_terms.set_yscale("log")

    ax_proj = axes[1, 0]
    if "mean_proj_norm" in data:
        ax_proj.plot(
            steps, data["mean_proj_norm"], color="tab:red", linewidth=1.4, label="mean proj norm"
        )
    ax_proj.set_title("Mean projection norm")
    ax_proj.set_ylabel("‖Vᵀ(h−μ)‖")
    ax_proj.set_xlabel("Step")
    ax_proj.grid(True, alpha=0.3)
    ax_proj.legend(loc="best")

    ax_aux = axes[1, 1]
    plotted_aux = False
    if "grad_norm" in data:
        ax_aux.plot(
            steps, data["grad_norm"], color="tab:purple", linewidth=1.0, alpha=0.8, label="grad norm"
        )
        ax_aux.set_ylabel("Grad norm", color="tab:purple")
        ax_aux.tick_params(axis="y", labelcolor="tab:purple")
        plotted_aux = True
    if "learning_rate" in data:
        ax_lr = ax_aux.twinx() if plotted_aux else ax_aux
        ax_lr.plot(
            steps,
            data["learning_rate"],
            color="tab:gray",
            linewidth=1.2,
            linestyle="--",
            label="learning rate",
        )
        ax_lr.set_ylabel("Learning rate", color="tab:gray")
        ax_lr.tick_params(axis="y", labelcolor="tab:gray")
    ax_aux.set_title("Grad norm & learning rate")
    ax_aux.set_xlabel("Step")
    ax_aux.grid(True, alpha=0.3)

    if summary is not None:
        info_lines = [
            f"steps: {summary.get('num_steps', '?')}"
            + (" (early-stopped)" if summary.get("stopped_early") else ""),
            f"L_forget: {summary.get('initial_L_forget', float('nan')):.3f}"
            f" → {summary.get('final_L_forget', float('nan')):.3f}",
            f"top100 KL: {summary.get('initial_top100_KL', float('nan')):.3f}"
            f" → {summary.get('final_top100_KL', float('nan')):.3f}",
            f"proj norm: {summary.get('initial_mean_proj_norm', float('nan')):.2f}"
            f" → {summary.get('final_mean_proj_norm', float('nan')):.2f}",
        ]
        fig.text(
            0.99,
            0.01,
            "\n".join(info_lines),
            ha="right",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.85),
        )

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    if out_path is None:
        out_path = run_dir / "training_progress.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to a training run directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <run_dir>/training_progress.png)",
    )
    parser.add_argument(
        "--log-loss",
        action="store_true",
        help="Use a log scale on the loss y-axis (useful when loss spans several orders of magnitude)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    out_path = plot_run(run_dir, out_path=args.out, log_loss=args.log_loss)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
