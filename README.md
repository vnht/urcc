# UOC: Unlearning Over-Commitment

Self-contained pipeline that unlearns **over-commitment** (confidently
answering inputs that should be abstained from) by anchoring late-layer
hidden states along a behaviorally-discriminative subspace `V`.

## Vocabulary

The five behaviour categories the pipeline reasons about:

| Symbol | Term                       | Definition                                                                                                                                    | Dataset condition         | Desired? |
|--------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|----------|
| **A**  | Over-commitment            | The model gives or attempts a substantive answer when it should abstain, clarify, or reject the premise.                                       | Unanswerable input        | No       |
| **B**  | Legitimate abstention      | The model abstains, says it cannot determine the answer, requests missing information, or rejects the premise when the input is unanswerable. | Unanswerable input        | Yes      |
| **C**  | Legitimate commitment      | The model gives a substantive answer when the input calls for answering.                                                                       | Answerable input          | Yes      |
| **D**  | Over-abstention            | The model abstains when the input is answerable and it should provide a substantive answer.                                                    | Answerable input          | No       |
| **E**  | General utility            | The model preserves ordinary instruction-following and language-modelling behaviour outside the answerability setting.                         | UltraChat / utility data  | Yes      |

The unlearning target is **A**. We must protect **B**, **C**, and **E**, and
avoid creating **D**.

## Data pools

- **Forget pool `D_F`** — unanswerable prompts paired with the model's own
  over-commit prefix. Mined by judging the base model's greedy output and
  keeping COMMIT (category A) examples.
- **Retain-answerable pool `D_R_A`** — answerable prompts paired with their
  gold answers (category C).
- **Retain-general pool `D_R_G`** — `(prompt, response)` pairs from a general
  instruction corpus (category E, UltraChat).

## Activation sets (step 1)

For each example, take the mean late-layer hidden state over the first `K=8`
answer-token positions. Five sets:

| Set | Source | Forward signal |
|-----|--------|----------------|
| **A** activations | `D_F` prompts, model's over-commit completion          | `h_l(x, y_A)` |
| **B** activations | `D_F` prompts, **templated** legitimate-abstention text | `h_l(x, y_⊥)` |
| **C** activations | `D_R_A` prompts, gold answer                            | `h_l(x, y_C)` |
| **D** activations | `D_R_A` prompts, **templated** abstention text          | `h_l(x, y_⊥)` |
| **E** activations | `D_R_G` (prompt, response) pairs                        | `h_l(x, y_E)` |

`y_⊥ = "I do not have enough information to answer that."` is the same
templated string in B and D — it acts as the "abstain mode" baseline that
gets subtracted out of the contrasts below.

## Subspace V (step 2)

For each late layer `l`, form the over-commit contrast and the
legitimate-commit contrast:

```
c_OC(x) = h_l(A) − h_l(B)        for x ∈ D_F        (over-commit minus its abstain baseline)
c_LC(x) = h_l(C) − h_l(D)        for x ∈ D_R_A      (legit-commit minus its abstain baseline)
```

Form covariances `Σ_OC, Σ_LC` of those contrasts plus the general-utility
covariance `Σ_E = cov(h_l(E))`, and solve the generalised eigenproblem in
the retain span:

```
(Σ_OC − Σ_LC) v = γ Σ_E v
```

`V_l ∈ ℝ^{D × r}` are the top-`r` generalised eigenvectors. Large positive
`γ` ⇒ direction along which over-commit varies *more* than legitimate-commit
(after subtracting the shared abstain baseline), normalised against general
utility. `V_l` is computed once and frozen.

## Anchors (step 3)

```
μ_l⁻ = mean over D_F   of  h_l(B)         (legitimate-abstention pole)
μ_l⁺ = mean over D_R_A of  h_l(C)         (legitimate-commitment pole)
```

Both fixed.

## Loss (step 4)

LoRA adapter `δθ` on `f_θ`. Two components, both of the form
`‖V_lᵀ(h_l − target)‖²`, averaged over late layers `{l}` and the first
`K=8` answer-token positions `{t}`:

```
L = L_forget + λ · L_retain

L_forget = E_{(x, y) ∈ D_F}                ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ)  −  μ_l⁻      ) ‖² ⟩_{l, t}

L_retain = E_{(x, y) ∈ D_R_A ∪ D_R_G}      ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ)  −  τ_l(x, y) ) ‖² ⟩_{l, t}

τ_l(x, y) = μ_l⁺                  if (x, y) ∈ D_R_A     (legitimate commitment)
          = h_l(x, y; θ_frozen)   if (x, y) ∈ D_R_G     (general utility, per-token frozen reference)
```

Effect, by category:

| Category | Forward inputs | Anchor target | Effect |
|---|---|---|---|
| A (over-commit)         | unanswerable + over-commit prefix | `μ⁻` (B-pole)         | pulled toward legitimate abstention |
| B (legit-abstain)       | not trained on directly           | —                     | preserved (anchor is fixed)         |
| C (legit-commit)        | answerable + gold answer          | `μ⁺` (C-pole)         | held in place                       |
| D (over-abstain)        | not trained on directly           | —                     | not encouraged (no path to D)       |
| E (general utility)     | UltraChat (prompt, response)      | `h_l^ref` per token   | held at frozen-base reference       |

LoRA on `{q,k,v,o,up,down,gate}_proj`; base weights frozen.

## Layout

Each step lives in its own folder and owns its `data/` subdirectory.

```
.
├── README.md                       (this file)
├── config.py                       paths, model registry, defaults
├── _common.py                      shared utils (loading, forward, tokenisation, generation)
├── judge.py                        Cerebras gpt-oss-120b judge (used by step 0 & 5)
├── requirements.txt
│
├── step0_mine/
│   ├── mine.py
│   └── data/
│       ├── sampled/                raw inputs (questions, retain pairs)
│       │   ├── kuq_unanswerable.jsonl
│       │   ├── squad_unanswerable.jsonl
│       │   ├── kuq_answerable.jsonl
│       │   ├── squad_answerable.jsonl
│       │   └── ultrachat.jsonl
│       ├── mined/                  step 0 output: full judged completions
│       │   └── <model>_<dataset>.jsonl
│       └── forget/                 step 0 output: COMMIT-only subset (D_F, category A)
│           └── <model>_<dataset>.jsonl
│
├── step1_extract_activations/
│   ├── extract.py
│   └── data/
│       └── activations_<model>.pt          A, B, C, D, E activation sets
│
├── step2_build_subspace/
│   ├── build_subspace.py
│   └── data/
│       └── subspace_<model>_r<rank>.pt
│
├── step3_build_anchors/
│   ├── build_anchors.py
│   └── data/
│       └── anchors_<model>.pt              μ⁻, μ⁺
│
├── step4_train/
│   ├── train.py
│   ├── plot_training.py
│   └── data/
│       └── runs/<run_name>/
│           ├── adapter/                LoRA weights
│           ├── training_config.json
│           ├── loss_log.csv
│           └── train_summary.json
│
└── step5_evaluate/
    ├── evaluate.py
    └── data/
        ├── heldout/
        │   ├── kuq.jsonl
        │   └── squad.jsonl
        └── results/<run_name>/
            ├── generations.jsonl
            └── answerability_metrics.jsonl
```

## Pipeline

| Step | Script                                    | Reads                                                                    | Writes                                                              |
|------|-------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------|
| 0    | `step0_mine/mine.py`                      | `step0_mine/data/sampled/{kuq,squad}_unanswerable.jsonl`                 | `step0_mine/data/{mined,forget}/<model>_<dataset>.jsonl`            |
| 1    | `step1_extract_activations/extract.py`    | `step0_mine/data/{forget,sampled}/`                                      | `step1_extract_activations/data/activations_<model>.pt`             |
| 2    | `step2_build_subspace/build_subspace.py`  | `step1_extract_activations/data/activations_<model>.pt`                  | `step2_build_subspace/data/subspace_<model>_r<rank>.pt`             |
| 3    | `step3_build_anchors/build_anchors.py`    | `step1_extract_activations/data/activations_<model>.pt`                  | `step3_build_anchors/data/anchors_<model>.pt`                       |
| 4    | `step4_train/train.py`                    | `step0_mine/`, `step2_*/`, `step3_*/` outputs                            | `step4_train/data/runs/<run_name>/`                                 |
| 5    | `step5_evaluate/evaluate.py`              | `step5_evaluate/data/heldout/{kuq,squad,ultrachat}.jsonl`                | `step5_evaluate/data/results/<name>/{kuq,squad,ultrachat}.json` (one JSON per dataset, metrics + per-row + baseline deltas) |

## Run end-to-end

```bash
# Step 0 — mine the model's over-commitment (fresh judge calls; needs CEREBRAS_TOKEN)
python3 step0_mine/mine.py                       --model qwen_instruct

# Steps 1–3 — build the subspace and anchors (one-time, per model)
python3 step1_extract_activations/extract.py     --model qwen_instruct
python3 step2_build_subspace/build_subspace.py   --model qwen_instruct --rank 32
python3 step3_build_anchors/build_anchors.py     --model qwen_instruct

# Step 5 (baseline first) — zero-shot reference
# Runs answerability (KUQ + SQuAD) AND UltraChat perplexity in one model load.
python3 step5_evaluate/evaluate.py               --model qwen_instruct

# Step 4 — train with the two-component UOC loss
python3 step4_train/train.py                     --model qwen_instruct \
    --rank 32 --lambda-retain 1.0 --epochs 3 --lr 3e-5

# Step 5 (trained) — evaluate the LoRA adapter and compare against the baseline
python3 step5_evaluate/evaluate.py               --run-dir step4_train/data/runs/<run_name> \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct

# Plot training curves
python3 step4_train/plot_training.py             step4_train/data/runs/<run_name>
```

## Smoke test

```bash
python3 step0_mine/mine.py                       --model qwen_instruct --max-per-dataset 50
python3 step1_extract_activations/extract.py     --model qwen_instruct --max-per-set 200
python3 step2_build_subspace/build_subspace.py   --model qwen_instruct --rank 16
python3 step3_build_anchors/build_anchors.py     --model qwen_instruct
python3 step5_evaluate/evaluate.py               --model qwen_instruct \
    --max-per-dataset 100 --max-ppl-rows 100
python3 step4_train/train.py                     --model qwen_instruct \
    --rank 16 --max-train-steps 50 --epochs 1
python3 step5_evaluate/evaluate.py               --run-dir step4_train/data/runs/<run> \
    --max-per-dataset 100 --max-ppl-rows 100 \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct
```

## Why this design

- **Two components, one geometry.** Forget and retain are both
  projection-distance terms along the same subspace `V`, just with different
  targets. One coefficient `λ` controls the trade-off; nothing else.
- **Categories drive ablations.** Each piece of the loss is named by which
  behaviour it is moving (A, C, E) and which anchor it is moving toward
  (B-pole, C-pole, frozen ref). Drop μ⁻ → "no abstain pole". Drop μ⁺ → "no
  commit pole". Set rank `r=0` → "no subspace". Each ablation removes
  exactly one geometric component.
- **Per-step folders, per-step data.** Every script's inputs and outputs
  are located in predictable paths; outputs of step `k` live under
  `step_k/data/` and downstream steps reach back through `config.py`
  helpers.
- **No external runtime dependencies on `old/`.** The judge module
  (`judge.py`) is colocated at the project root; everything the pipeline
  needs is in the steps it ships with.

## Configuration

All paths, model IDs, layer slices, and method defaults live in `config.py`.
To target a different model:

1. Add the HF id to `MODEL_REGISTRY`.
2. Add its late-layer indices to `LAYER_SLICE`.
3. Re-run from step 0 (mining is per-model).
