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
    # Ministral-3-14B: same `mistral3` architecture as the 8B (40 text layers).
    # The `-BF16` instruct release is used for the same no-FP8-dequant reason as
    # the 8B (the unsuffixed 14B instruct ships FP8 weights).
    "ministral14b_instruct": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    # OpenAI gpt-oss-20b. Sparse Mixture-of-Experts (24 layers, hidden 2880,
    # 32 experts top-4) whose expert FFNs are stored as FUSED 3-D nn.Parameter
    # tensors (`mlp.experts.gate_up_proj` / `mlp.experts.down_proj`), not
    # nn.Linear — so LoRA targets them via `target_parameters` (see
    # LORA_TARGET_PARAMETERS below), while attention stays standard q/k/v/o.
    # Ships MXFP4-quantised experts: loaded with FineGrained MXFP4 dequant to
    # bf16 (~40GB). A reasoning model on the harmony chat format — the analysis
    # (CoT) channel cannot be disabled, only its effort lowered, so generation
    # is parsed down to the `final` channel (see _common.generate_greedy).
    "gptoss_instruct":    "openai/gpt-oss-20b",
    # Meta Llama 3.1 8B pre-trained (base, no instruction tuning).
    # Dense decoder: 32 layers, hidden 4096, GQA (32 query / 8 KV heads),
    # SwiGLU FFN — LORA_TARGET_MODULES applies without changes.
    "llama_base":         "meta-llama/Llama-3.1-8B",
    # Microsoft Phi-4 (14.7B). Dense decoder (Phi3ForCausalLM): 40 layers,
    # hidden 5120, trained predominantly on synthetic/distilled data. Uses
    # FUSED projections — `qkv_proj` (attention) and `gate_up_proj` (MLP) —
    # instead of split q/k/v and gate/up, so the LoRA targets are remapped in
    # LORA_DENSE_TARGET_OVERRIDES (same full attn+MLP coverage as qwen, just
    # fused module names; the training recipe is otherwise identical).
    "phi4_instruct":      "microsoft/phi-4",
}

# Last-25% of transformer layers per model (where the commitment subspace lives)
LAYER_SLICE: dict[str, list[int]] = {
    "qwen_instruct":      [24, 25, 26, 27, 28, 29, 30, 31],
    "qwen_base":          [24, 25, 26, 27, 28, 29, 30, 31],
    "ministral_instruct": [25, 26, 27, 28, 29, 30, 31, 32, 33],
    "ministral_base":     [25, 26, 27, 28, 29, 30, 31, 32, 33],
    # Ministral-3-14B: 40 text layers → last 25% = layers 30-39.
    "ministral14b_instruct": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    # gpt-oss-20b: 24 text layers → last 25% = layers 18-23.
    "gptoss_instruct":    [18, 19, 20, 21, 22, 23],
    # Llama 3.1 8B: 32 text layers → last 25% = layers 24-31.
    "llama_base":         [24, 25, 26, 27, 28, 29, 30, 31],
    # Phi-4: 40 layers → last 25% = layers 30-39.
    "phi4_instruct":      [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
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
#
# Per-domain templates: each was chosen by analysing the base model's natural
# abstain phrasings on the held-out unanswerable set (see baseline_qwen_instruct
# evals). KUQ abstentions are semantically diverse (no single dominant template),
# so a generic refusal is used. SQuAD abstentions are dominated by the
# "the provided context does not [contain|state|mention] X" family — so a
# context-grounded template lives much closer to the base model's natural abstain
# region in late-layer hidden-state space, which makes μ⁻_squad a target the
# forget loss can actually reach without dragging unrelated activations along.
ABSTAIN_TEMPLATE_PER_DATASET = {
    "kuq":   "I do not have enough information to answer that.",
    "squad": "The provided context does not contain information about that.",
}

# Backward-compat: keep a single fallback string for any caller that doesn't
# know the row's dataset (none in the current pipeline).
ABSTAIN_TEMPLATE = ABSTAIN_TEMPLATE_PER_DATASET["kuq"]


# Map every dataset name to one of the two trained domains so prompt builders,
# abstain templates, and per-domain V/μ⁻ can all be looked up generically.
# "kuq"   = no-context (KUQ_PROMPT_TEMPLATE)
# "squad" = with-context (SQUAD_PROMPT_TEMPLATE)
#
# The two trained domains map to themselves. Held-out / unseen datasets added
# in step 5 are registered here so evaluate.py can route them without changes
# to its prompt-building code path.
DOMAIN_OF: dict[str, str] = {
    # Trained domains
    "kuq":        "kuq",
    "squad":      "squad",
    # New held-out, no-context
    "selfaware":  "kuq",
    # New held-out, with-context
    "faitheval":  "squad",
    "nomiracl":   "squad",
}


def domain_of(dataset: str | None) -> str:
    """Return the trained domain ('kuq' | 'squad') for a dataset name."""
    key = str(dataset or "").lower()
    return DOMAIN_OF.get(key, key)


def abstain_template_for(dataset: str | None) -> str:
    """Return the abstention template aligned to the row's dataset domain.
    Falls back to the KUQ generic template if the domain is unknown.
    """
    return ABSTAIN_TEMPLATE_PER_DATASET.get(domain_of(dataset), ABSTAIN_TEMPLATE)


# ── Method defaults ───────────────────────────────────────────────────────────

K_ANSWER_TOKENS    = 8
SUBSPACE_RANK      = 32
SUBSPACE_RIDGE     = 1e-3
RETAIN_BASIS_RANK  = 512

# Per-model overrides of the step-2 eigenproblem ridge. Higher ridge relaxes
# the Σ_E whitening so V keeps more of the behavioral abstainↈcommit axis
# (pole_sep_in_V) at the cost of weaker utility protection (guarded by the
# retain loss). Tuned per model via
# step2_build_subspace/diagnose_subspace.py --ridge-sweep:
#   gptoss_instruct: 10.0  (kuq axis suppressed at default ridge)
# ministral14b_instruct uses the default 1e-3 (same as qwen).
SUBSPACE_RIDGE_OVERRIDES: dict[str, float] = {
    "gptoss_instruct": 10.0,
}


def subspace_ridge(model_key: str) -> float:
    """Step-2 ridge for `model_key` (per-model override or SUBSPACE_RIDGE)."""
    return SUBSPACE_RIDGE_OVERRIDES.get(model_key, SUBSPACE_RIDGE)

LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "up_proj", "down_proj", "gate_proj"]

# Per-model LoRA module suffixes zeroed at inference (train full like qwen,
# deploy without). ministral14b: MLP LoRA writes enact repetition collapse;
# diagnose_adapter --ablate-types on the full trained adapter showed
# gate/up/down zeroed = 0/8 degenerate at scale 1.0, attention zeroed = 4/8.
# Training attention-only from scratch does NOT replicate that — attention
# must compensate alone and still collapses. Correct recipe: train full LoRA,
# zero MLP suffixes when loading for eval.
LORA_INFERENCE_ZERO_SUFFIXES: dict[str, list[str]] = {
    "ministral14b_instruct": ["gate_proj", "up_proj", "down_proj"],
}


def lora_inference_zero_suffixes(model_key: str) -> list[str]:
    """Module-name suffixes to zero at inference for `model_key` (empty = none)."""
    return LORA_INFERENCE_ZERO_SUFFIXES.get(model_key, [])


# Per-model remap of the dense LoRA targets for backbones whose nn.Linear
# names differ from the Llama-style default. Coverage is kept IDENTICAL to
# qwen (all attention + all MLP projections) — only the module names change,
# so the training recipe stays consistent across models.
#   phi4: fused attention (`qkv_proj` = q+k+v in one Linear) and fused MLP
#   input (`gate_up_proj` = gate+up in one Linear).
LORA_DENSE_TARGET_OVERRIDES: dict[str, list[str]] = {
    "phi4_instruct": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


def lora_dense_targets(model_key: str) -> list[str]:
    """Dense-model LoRA target_modules for `model_key` (name remap only —
    coverage is the same full attn+MLP set for every dense model). Not
    consulted for fused-expert MoE models (see LORA_TARGET_PARAMETERS)."""
    return LORA_DENSE_TARGET_OVERRIDES.get(model_key, LORA_TARGET_MODULES)

# ── Fused-expert MoE LoRA (gpt-oss et al.) ────────────────────────────────────
# Some MoE models store expert FFNs as fused 3-D nn.Parameter tensors instead of
# nn.Linear modules, so standard `target_modules` can't reach them. For these
# models PEFT (>=0.17) targets the raw parameters via `target_parameters`, and
# attention is targeted separately with the (Linear) names in LORA_ATTN_TARGETS.
# A model_key present in LORA_TARGET_PARAMETERS routes step4's _apply_lora down
# the MoE path; absent keys use the standard dense LORA_TARGET_MODULES path.
LORA_TARGET_PARAMETERS: dict[str, list[str]] = {
    # MoE experiment: force the *MoE path* (routed experts + the router gate) to
    # carry the whole unlearning intervention, instead of letting the dense
    # attention LoRA absorb the geometric forget loss without changing behaviour.
    #   - experts.gate_up_proj / down_proj: per-expert fused FFN tensors (3-D).
    #   - router.weight: the GptOssTopKRouter gate is a raw 2-D nn.Parameter
    #     (num_experts × hidden), NOT an nn.Linear, so it is also targeted here
    #     via target_parameters. LoRA on it re-weights the contribution of the
    #     selected experts; note top-4 selection is argmax (non-differentiable),
    #     so this changes how much each routed expert speaks, not WHICH experts
    #     are routed.
    "gptoss_instruct": [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.router.weight",
    ],
}
LORA_ATTN_TARGETS: dict[str, list[str] | str] = {
    # Empty = no attention LoRA for gpt-oss (drop q/k/v/o). PEFT supports an
    # empty target_modules alongside target_parameters; this makes the experts +
    # router the ONLY adaptable params so they must carry the intervention.
    "gptoss_instruct": [],
}


def lora_target_parameters(model_key: str) -> list[str] | None:
    """Fused-expert parameter targets for `model_key`, or None for dense models."""
    return LORA_TARGET_PARAMETERS.get(model_key)


def lora_attn_targets(model_key: str) -> list[str] | str:
    """Attention LoRA targets used alongside target_parameters. Either a suffix
    list or a regex string (the latter scopes multimodal models to their text
    stack so the vision tower is excluded)."""
    return LORA_ATTN_TARGETS.get(model_key, ["q_proj", "k_proj", "v_proj", "o_proj"])


# ── Harmony / reasoning (gpt-oss) ─────────────────────────────────────────────
# gpt-oss always emits an `analysis` (CoT) channel before the `final` answer;
# it can't be disabled, only lowered. We use the lowest effort to minimise the
# CoT length (and the generation budget it consumes) since URC only needs the
# committal `final`-channel answer.
GPTOSS_REASONING_EFFORT = "low"

# Reasoning models burn part of the decode budget on the analysis channel before
# the final answer appears, so they need a larger cap than the dense default.
GPTOSS_MAX_NEW_TOKENS = 512

# gpt-oss attention backend for GENERATION/eval/mining (training always uses
# eager — see load_model_and_tokenizer). The model uses attention SINKS
# (learnable per-head logits added before softmax), so plain `sdpa` is NOT
# supported — it silently drops the sink term and corrupts results. Options:
#   "eager"          — correct everywhere, no compile. Slowest, but the only
#       backend that reliably works on the Colab Blackwell (sm_120) image:
#       flex_attention there fails to compile with Inductor "NoValidChoicesError"
#       (the Triton flex template has no valid sm_120 config and the HOP has no
#       ATEN fallback). Default.
#   "flex_attention" — sink-aware and much faster than eager IF Inductor/Triton
#       can compile it (works on Hopper/Ampere with a matching torch); broken on
#       the current Colab Blackwell+Triton build, so do NOT use it there.
#   "kernels-community/vllm-flash-attn3" — fastest, but Hopper-only (H100/H200).
GPTOSS_ATTN_IMPLEMENTATION = "eager"


def max_new_tokens_for(model_key: str) -> int:
    """Greedy-decode cap for a model: larger for reasoning models whose CoT
    precedes the final answer."""
    if model_key == "gptoss_instruct":
        return GPTOSS_MAX_NEW_TOKENS
    return DEFAULT_MAX_NEW_TOKENS

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
