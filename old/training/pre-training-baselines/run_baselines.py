#!/usr/bin/env python3
"""
Run all pre-training baseline evaluations and write results to this directory.

Steps:
  1. generate.py   — run each model on KUQ/SQuAD eval sets
  2. judge_outputs — judge completions, compute answerability metrics
  3. perplexity    — compute UltraChat PPL for each model

All output files are written to training/pre-training-baselines/.

Usage:
    python run_baselines.py                    # all four models
    python run_baselines.py --model qwen-9b    # one model
    python run_baselines.py --n 10             # smoke test
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
EVAL_DIR  = REPO_ROOT / "evaluation"


def run(script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n{'=' * 64}")
    print(f"RUN  {script.relative_to(REPO_ROOT)}")
    print(f"{'=' * 64}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"ERROR: {script.name} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all pre-training baseline evaluations.")
    parser.add_argument("--model", help="Model shortcut (default: all four)")
    parser.add_argument("--n", type=int, help="Limit instances/examples (smoke test)")
    parser.add_argument("--skip-generate",    action="store_true", help="Skip generation step")
    parser.add_argument("--skip-judge",       action="store_true", help="Skip judging step")
    parser.add_argument("--skip-perplexity",  action="store_true", help="Skip PPL step")
    args = parser.parse_args()

    shared = []
    if args.model:
        shared += ["--model", args.model]
    if args.n:
        shared += ["--n", str(args.n)]

    results_dir_args = ["--results-dir", str(HERE)]

    if not args.skip_generate:
        run(HERE / "generate.py", shared)

    if not args.skip_judge:
        run(EVAL_DIR / "judge_outputs.py", shared + results_dir_args)

    if not args.skip_perplexity:
        run(EVAL_DIR / "perplexity.py", shared + results_dir_args)

    print("\nAll steps complete. Results written to:", HERE)


if __name__ == "__main__":
    main()
