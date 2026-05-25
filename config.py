"""Central configuration for the UOC (Unlearning Over-Commitment) pipeline.

Single source of truth for model IDs, layer slices, paths, and method defaults.
Imported by every step script.

Layout
------
Each step lives in its own folder (step0_mine, step1_extract_activations, ...)
and owns its `data/` subdirectory. This module knows where each step's data
lives and exposes path helpers that the scripts use.

    repo/
    ├── config.py          (this file)
    ├── _common.py
    ├── judge.py
    ├── step0_mine/data/{sampled,mined,forget}/
    ├── step1_extract_activations/data/
    ├── step2_build_subspace/data/
    ├── step3_build_anchors/data/
    ├── step4_train/data/runs/
    └── step5_evaluate/data/{heldout,results}/
"""

from __future__ import annotations

from pathlib import Path

# ── Paths (root + per-step folders) ───────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent

STEP0_DIR = REPO_ROOT / "step0_mine"
STEP1_DIR = REPO_ROOT / "step1_extract_activations"
STEP2_DIR = REPO_ROOT / "step2_build_subspace"
STEP3_DIR = REPO_ROOT / "step3_build_anchors"
STEP4_DIR = REPO_ROOT / "step4_train"
STEP5_DIR = REPO_ROOT / "step5_evaluate"

# Step 0 — mining inputs and outputs
SAMPLED_DIR = STEP0_DIR / "data" / "sampled"   # raw inputs (questions, retain pairs)
MINED_DIR   = STEP0_DIR / "data" / "mined"     # all judged completions
FORGET_DIR  = STEP0_DIR / "data" / "forget"    # COMMIT-only subset (forget set)

# Step 1 — activations bundles
ACTIVATIONS_DIR = STEP1_DIR / "data"

# Step 2 — subspace bundles
SUBSPACE_DIR = STEP2_DIR / "data"

# Step 3 — anchor bundles
ANCHORS_DIR = STEP3_DIR / "data"

# Step 4 — training runs
RUNS_DIR = STEP4_DIR / "data" / "runs"

# Step 5 — held-out eval inputs and per-run results
HELDOUT_DIR  = STEP5_DIR / "data" / "heldout"
RESULTS_DIR  = STEP5_DIR / "data" / "results"

# Make sure output directories exist (input ones are populated externally)
for _d in (MINED_DIR, FORGET_DIR, ACTIVATIONS_DIR, SUBSPACE_DIR,
           ANCHORS_DIR, RUNS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, str] = {
    "qwen_instruct":      "Qwen/Qwen3.5-9B",
    "qwen_base":          "Qwen/Qwen3.5-9B-Base",
    # `-BF16` is the pre-dequantised release of Ministral-3-8B-Instruct-2512,
    # used because we train through the model (no FP8 dequant detour).
    "ministral_instruct": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "ministral_base":     "mistralai/Ministral-3-8B-Base-2512",
}

# Last-25% of transformer layers per model (where the commitment subspace lives)
LAYER_SLICE: dict[str, list[int]] = {
    "qwen_instruct":      [24, 25, 26, 27, 28, 29, 30, 31],
    "qwen_base":          [24, 25, 26, 27, 28, 29, 30, 31],
    "ministral_instruct": [25, 26, 27, 28, 29, 30, 31, 32, 33],
    "ministral_base":     [25, 26, 27, 28, 29, 30, 31, 32, 33],
}


# ── Prompt templates (must match mining-time templates) ───────────────────────

KUQ_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence.\n\nQuestion:\n{question}\n\nAnswer:"
)
SQUAD_PROMPT_TEMPLATE = (
    "Answer concisely in a sentence using only the context."
    "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

# Templated abstention used to build μ⁻ contrasts.
ABSTAIN_TEMPLATE = "I do not have enough information to answer that."


# ── Method defaults ───────────────────────────────────────────────────────────

K_ANSWER_TOKENS    = 8
SUBSPACE_RANK      = 32
SUBSPACE_RIDGE     = 1e-3
RETAIN_BASIS_RANK  = 512

LORA_R              = 32
LORA_ALPHA          = 64
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "up_proj", "down_proj", "gate_proj"]

DEFAULT_LR              = 3e-5
DEFAULT_EPOCHS          = 3
DEFAULT_LAMBDA_RETAIN   = 1.0
DEFAULT_FORGET_BATCH    = 4
DEFAULT_RETAIN_BATCH    = 4
DEFAULT_GRAD_ACCUM      = 4
DEFAULT_WARMUP_RATIO    = 0.03
DEFAULT_MAX_GRAD_NORM   = 1.0

DEFAULT_MAX_NEW_TOKENS  = 64    # greedy decoding cap during mining/eval


# ── Path helpers ──────────────────────────────────────────────────────────────

def sampled_unanswerable_path(dataset: str) -> Path:
    """Raw unanswerable questions, input to step 0 (mining)."""
    return SAMPLED_DIR / f"{dataset}_unanswerable.jsonl"


def sampled_answerable_path(dataset: str) -> Path:
    """Raw answerable QA pairs (with gold answers). Used as the retain-answerable
    pool D_R_A (category C) and as the source for the legitimate-commitment
    pole μ⁺."""
    return SAMPLED_DIR / f"{dataset}_answerable.jsonl"


def sampled_general_path() -> Path:
    """Raw UltraChat retain pairs (general retain pool)."""
    return SAMPLED_DIR / "ultrachat.jsonl"


def mined_path(model_key: str, dataset: str) -> Path:
    """All judged completions for (model, dataset) — output of step 0."""
    return MINED_DIR / f"{model_key}_{dataset}.jsonl"


def forget_path(model_key: str, dataset: str) -> Path:
    """COMMIT-only subset of mined rows — the forget set used in steps 1, 4."""
    return FORGET_DIR / f"{model_key}_{dataset}.jsonl"


def activations_path(model_key: str) -> Path:
    """Step 1 output: bundle with all forward-pass means."""
    return ACTIVATIONS_DIR / f"activations_{model_key}.pt"


def subspace_path(model_key: str, rank: int = SUBSPACE_RANK) -> Path:
    """Step 2 output: discriminative subspace V_l."""
    return SUBSPACE_DIR / f"subspace_{model_key}_r{rank}.pt"


def anchors_path(model_key: str) -> Path:
    """Step 3 output: μ⁻ (abstain pole) and μ⁺ (commit pole)."""
    return ANCHORS_DIR / f"anchors_{model_key}.pt"


def heldout_path(dataset: str) -> Path:
    """Step 5 input: held-out evaluation pool."""
    return HELDOUT_DIR / f"{dataset}.jsonl"


def results_dir_for(run_name: str) -> Path:
    """Step 5 output: per-run evaluation results."""
    p = RESULTS_DIR / run_name
    p.mkdir(parents=True, exist_ok=True)
    return p
