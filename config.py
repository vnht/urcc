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
    # Google Gemma-4-26B-A4B-it. Sparse MoE (30 layers, 128 experts top-8,
    # 25.2B total / 3.8B active). Loads as `Gemma4ForConditionalGeneration` (a
    # multimodal wrapper whose text stack is at `.model.language_model`); text-
    # only forward/generate works without pixel inputs. Like gpt-oss its routed
    # expert FFNs are fused 3-D nn.Parameter tensors (`...layers.N.experts.
    # gate_up_proj` / `down_proj`, targeted via LORA_TARGET_PARAMETERS), but
    # unlike gpt-oss it ships native bf16 (no MXFP4 dequant). Configurable
    # thinking: we disable it (chat template pre-fills an empty
    # `<|channel>thought\n<channel|>` scaffold into the prompt), so the greedy
    # answer is whatever follows the final `<channel|>` close token (see
    # _common.parse_gemma_final).
    "gemma_instruct":     "google/gemma-4-26B-A4B-it",
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
    # gemma-4-26B-A4B: 30 text layers → last 25% (ceil) = layers 22-29.
    "gemma_instruct":     [22, 23, 24, 25, 26, 27, 28, 29],
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
    "qaqa":       "kuq",
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

LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "up_proj", "down_proj", "gate_proj"]

# ── Fused-expert MoE LoRA (gpt-oss et al.) ────────────────────────────────────
# Some MoE models store expert FFNs as fused 3-D nn.Parameter tensors instead of
# nn.Linear modules, so standard `target_modules` can't reach them. For these
# models PEFT (>=0.17) targets the raw parameters via `target_parameters`, and
# attention is targeted separately with the (Linear) names in LORA_ATTN_TARGETS.
# A model_key present in LORA_TARGET_PARAMETERS routes step4's _apply_lora down
# the MoE path; absent keys use the standard dense LORA_TARGET_MODULES path.
LORA_TARGET_PARAMETERS: dict[str, list[str]] = {
    "gptoss_instruct": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    # gemma-4-26B-A4B routed experts (Gemma4TextExperts). Confirmed on host: the
    # fused expert tensors hang directly off the decoder layer as
    # `...layers.N.experts.gate_up_proj` (128, 1408, 2816) and
    # `...layers.N.experts.down_proj` (128, 2816, 704) — no `mlp.` prefix. There
    # is no separate shared-expert module in the HF impl (only `experts` +
    # `router.per_expert_scale`), so attention is the only nn.Linear target.
    "gemma_instruct":  ["experts.gate_up_proj", "experts.down_proj"],
}
LORA_ATTN_TARGETS: dict[str, list[str] | str] = {
    "gptoss_instruct": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # gemma-4 is multimodal: a bare ["q_proj", ...] suffix list also matches the
    # SigLIP VISION tower's attention, whose projections are Gemma4ClippableLinear
    # (a clipping wrapper PEFT can't adapt). A regex string (PEFT re.fullmatch)
    # scopes LoRA to the TEXT stack's self-attention only — the text q/k/v/o are
    # plain nn.Linear. The fused text experts are handled separately via
    # target_parameters; the vision tower gets no adapters.
    "gemma_instruct":  r".*language_model\.layers\.\d+\.self_attn\."
                       r"(?:q_proj|k_proj|v_proj|o_proj)",
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

# ── Gemma 4 (configurable thinking) ───────────────────────────────────────────
# Gemma-4 thinking is disabled by omitting the `<|think|>` control token from
# the system prompt. The 26B/31B variants still emit an (empty) `thought`
# channel scaffold before the answer, so we parse the final answer out of the
# decoded output (see _common.parse_gemma_final) and give a small budget bump
# over the dense default to absorb the channel-scaffold tokens.
GEMMA_ENABLE_THINKING = False
GEMMA_MAX_NEW_TOKENS = 128


def max_new_tokens_for(model_key: str) -> int:
    """Greedy-decode cap for a model: larger for reasoning models whose CoT
    precedes the final answer."""
    if model_key == "gptoss_instruct":
        return GPTOSS_MAX_NEW_TOKENS
    if model_key == "gemma_instruct":
        return GEMMA_MAX_NEW_TOKENS
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
