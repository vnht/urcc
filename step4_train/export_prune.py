#!/usr/bin/env python3
"""Post-hoc adapter prune — zero selected LoRA module deltas in a *trained* run.

Why
---
On low-redundancy backbones (Llama-3B, Ministral-14B) the UOC forget loss is
geometric only: it pulls the late-layer residual onto the fluent-refusal pole
μ⁻ but says nothing about free-running generation. The MLP "write path"
(gate/up/down_proj LoRA) can satisfy that geometric target while simultaneously
corrupting the decode path, which decodes as the repetition / empty-completion
collapse — even though the abstain target itself is a fluent sentence.

Zeroing the MLP-write LoRA delta after training reproduces the validated
"Full LoRA + zero MLP at inference" fix. It is a pure adapter edit: the UOC
loss and the trained attention LoRA are untouched. This avoids retraining the
existing Qwen-recipe runs that already collapsed.

What it does
------------
Copies an existing run dir's adapter + tokenizer + config files into a NEW
sibling run dir ``<run>_<out_tag>`` and zeros the chosen LoRA module deltas in
the copy. ``step5_evaluate`` loads the new dir exactly like any other run, so
you can A/B the pruned vs full adapter without touching the original.

Run
---
    # Zero the MLP write path of an existing Llama run (no retrain):
    python step4_train/export_prune.py \
        --run llama3b_instruct_uoc_r32_lam2_ep3_lr3e-05_fluency

    # Explicit modules + custom output tag:
    python step4_train/export_prune.py --run <run_name> \
        --modules down_proj --out-tag downzero
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from _common import log
from step4_train.train import _resolve_prune_modules, prune_lora_modules


def _resolve_run_dir(run: str) -> Path:
    """Accept either a run name (under cfg.RUNS_DIR) or an explicit path."""
    p = Path(run)
    if p.is_dir():
        return p
    cand = cfg.RUNS_DIR / run
    if cand.is_dir():
        return cand
    raise FileNotFoundError(
        f"Run dir not found: tried '{p}' and '{cand}'. Pass a run name under "
        f"{cfg.RUNS_DIR} or an explicit path."
    )


def _copy_run_files(src: Path, dst: Path) -> None:
    """Copy all top-level files (adapter, tokenizer, config json) from src to
    dst. Snapshot subdirs (_best/_final/checkpoint) are intentionally skipped
    — the pruned dir only needs a loadable primary adapter."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Zero selected LoRA module deltas in a trained run "
                    "(no retraining).")
    p.add_argument("--run", required=True,
                   help="Source run name (under step4_train/data/runs) or path.")
    p.add_argument("--modules", default="mlp",
                   help="'mlp' (default) = backbone MLP write-path projections, "
                        "or a comma list e.g. 'down_proj'.")
    p.add_argument("--model", default="",
                   help="Model key for resolving 'mlp' module names. If omitted, "
                        "read from the run's training_config.json.")
    p.add_argument("--out-tag", default="prune",
                   help="Suffix for the new run dir: <run>_<out_tag>. "
                        "Default: 'prune'.")
    args = p.parse_args()

    src = _resolve_run_dir(args.run)
    cfg_path = src / "training_config.json"
    train_cfg: dict = {}
    if cfg_path.exists():
        train_cfg = json.loads(cfg_path.read_text())

    model_key = args.model or train_cfg.get("model_key", "")
    if not model_key and args.modules.strip().lower() == "mlp":
        raise SystemExit(
            "Cannot resolve 'mlp' module names: no --model and no model_key in "
            f"{cfg_path}. Pass --model or an explicit --modules list.")

    module_names = _resolve_prune_modules(args.modules, model_key)
    if not module_names:
        raise SystemExit(f"--modules '{args.modules}' resolved to nothing.")

    dst = src.with_name(f"{src.name}_{args.out_tag}")
    if dst.exists():
        raise SystemExit(
            f"Output dir already exists: {dst}. Remove it or pick a different "
            f"--out-tag.")

    log.info("Pruning run: %s", src)
    log.info("  modules to zero: %s", module_names)
    log.info("  output dir: %s", dst)

    _copy_run_files(src, dst)
    matched = prune_lora_modules(dst, module_names)
    if not matched:
        shutil.rmtree(dst, ignore_errors=True)
        raise SystemExit(
            "No LoRA keys matched the requested modules; nothing written. "
            f"Check the module names against {src/'adapter_model.safetensors'}.")

    # Record provenance so eval/analysis can tell pruned adapters apart.
    train_cfg["export_pruned_modules"] = matched
    train_cfg["pruned_from_run"] = src.name
    (dst / "training_config.json").write_text(json.dumps(train_cfg, indent=2))

    log.info("Done. Pruned adapter written to %s", dst)
    log.info("Evaluate with: python step5_evaluate/evaluate.py --run-dir %s "
             "--datasets kuq squad", dst)


if __name__ == "__main__":
    main()
