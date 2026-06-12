#!/usr/bin/env python3
"""Export a pruned copy of a trained LoRA adapter.

Zeroing selected LoRA module deltas *after* training is identical to having
trained with those deltas absent (zeroing is the final op either way), so this
needs no GPU and no retraining. It exists to remove the MLP write-path deltas
(`up_proj`, `down_proj`, `gate_proj`) that collapse generation into degenerate
repetition while leaving the attention/residual path — which carries the
abstain-vs-commit *decision* — untouched.

The new run directory is a full, self-contained copy that `step5_evaluate`
loads exactly like any other run (`PeftModel.from_pretrained(model, run_dir)`).

Run
---
    python step4_train/export_pruned_adapter.py \
        --src llama3b_instruct_uoc_r32_lam2_ep3_lr3e-05 \
        --modules up_proj down_proj gate_proj \
        --tag mlpzero
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

DEFAULT_MLP_MODULES = ("up_proj", "down_proj", "gate_proj")
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def _resolve_run_dir(src: str) -> Path:
    """Accept either a run name (under RUNS_DIR) or an explicit path."""
    p = Path(src)
    if p.is_dir():
        return p
    cand = cfg.RUNS_DIR / src
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"Run dir not found: {src!r} (tried {p} and {cand})")


def _key_targets_module(key: str, modules: tuple[str, ...]) -> bool:
    """True if a safetensors key belongs to one of the target projections.

    PEFT keys look like
    `base_model.model.model.layers.27.mlp.down_proj.lora_B.weight`, so we match
    the module name as a dotted path segment to avoid accidental substring hits.
    """
    return any(f".{m}." in key for m in modules)


def prune_adapter_weights(src_file: Path, dst_file: Path,
                          modules: tuple[str, ...]) -> tuple[int, int]:
    """Copy `src_file` to `dst_file`, zeroing every LoRA tensor that belongs to
    one of `modules`. Returns (n_zeroed, n_total)."""
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    n_zeroed = 0
    with safe_open(str(src_file), framework="pt") as f:
        md = f.metadata()
        if md:
            metadata = dict(md)
        for key in f.keys():
            t = f.get_tensor(key)
            if _key_targets_module(key, modules):
                t = torch.zeros_like(t)
                n_zeroed += 1
            tensors[key] = t
    if not metadata:
        metadata = {"format": "pt"}
    save_file(tensors, str(dst_file), metadata=metadata)
    return n_zeroed, len(tensors)


def export(src_dir: Path, dst_dir: Path, modules: tuple[str, ...]) -> None:
    if dst_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_dir}")
    src_weights = src_dir / ADAPTER_WEIGHTS
    if not src_weights.exists():
        raise FileNotFoundError(f"No {ADAPTER_WEIGHTS} in {src_dir}")

    tmp = dst_dir.with_name(dst_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    # Copy everything except the weights + snapshot subdirs; weights are written
    # pruned, and the eval pipeline only reads the run-dir root.
    shutil.copytree(
        src_dir, tmp,
        ignore=shutil.ignore_patterns(ADAPTER_WEIGHTS, "_best", "_final",
                                      "checkpoint", "*.tmp"),
    )
    n_zeroed, n_total = prune_adapter_weights(src_weights, tmp / ADAPTER_WEIGHTS,
                                              modules)

    # Record what was pruned so the run is self-describing.
    cfg_path = tmp / "training_config.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            tcfg = json.load(fh)
    else:
        tcfg = {}
    tcfg["pruned_from"] = src_dir.name
    tcfg["pruned_lora_modules"] = list(modules)
    tcfg["pruned_tensor_count"] = n_zeroed
    with open(cfg_path, "w") as fh:
        json.dump(tcfg, fh, indent=2)

    tmp.rename(dst_dir)
    print(f"  zeroed {n_zeroed}/{n_total} LoRA tensors "
          f"({', '.join(modules)})")
    print(f"  wrote pruned run -> {dst_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Source run name (under RUNS_DIR) or path.")
    ap.add_argument("--modules", nargs="+", default=list(DEFAULT_MLP_MODULES),
                    help="LoRA projection names to zero "
                         f"(default: {' '.join(DEFAULT_MLP_MODULES)}).")
    ap.add_argument("--tag", default="mlpzero",
                    help="Suffix for the new run dir name (default: mlpzero).")
    ap.add_argument("--dest", default=None,
                    help="Explicit destination run name/path "
                         "(overrides --tag-derived name).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = _resolve_run_dir(args.src)
    modules = tuple(args.modules)
    if args.dest:
        dst = Path(args.dest)
        if not dst.is_absolute() and dst.parent == Path("."):
            dst = cfg.RUNS_DIR / args.dest
    else:
        dst = cfg.RUNS_DIR / f"{src_dir.name}_{args.tag}"
    print(f"Pruning adapter: {src_dir.name}")
    export(src_dir, dst, modules)


if __name__ == "__main__":
    main()
