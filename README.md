# UOC: Unlearning Over-Commitment

Self-contained pipeline that unlearns **over-commitment** (confidently
answering inputs that should be abstained from) with a two-component LoRA
loss: a geometric forget term that pulls late-layer hidden states along a
behaviorally-discriminative subspace `V` toward a per-domain abstain pole
`μ⁻(d)`, and a supervised CE retain term that preserves the model's output
distribution on legitimate-commit and general-utility inputs..

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

For each example, take the mean late-layer hidden state over a window of
`K=8` token positions starting **one token before** the first answer token:

```
window T(x) = { p_len − 1, p_len, p_len + 1, …, p_len + K − 2 }
```

Position `p_len − 1` is the prompt-final residual stream — the state from
which the LM head decides the *first* generated token. Including it inside
the window is what lets the retain loss intrinsically discourage degenerate
solutions where the LoRA satisfies the geometric loss by collapsing the
first-token logit to a chat-end token. The remaining `K − 1` positions
cover the body of the answer (or the start of the abstention text, for sets
B and D).

Five sets:

| Set | Source | Forward signal |
|-----|--------|----------------|
| **A** activations | `D_F` prompts, model's over-commit completion          | `h_l(x, y_A)` |
| **B** activations | `D_F` prompts, **templated** legitimate-abstention text | `h_l(x, y_⊥)` |
| **C** activations | `D_R_A` prompts, gold answer                            | `h_l(x, y_C)` |
| **D** activations | `D_R_A` prompts, **templated** abstention text          | `h_l(x, y_⊥)` |
| **E** activations | `D_R_G` (prompt, response) pairs                        | `h_l(x, y_E)` |

`y_⊥(d)` is a **per-domain** templated abstention string used in B and D.
KUQ and SQuAD have very different natural abstention phrasings (no-context
vs. context-grounded), so each domain gets its own template:

```
y_⊥(kuq)   = "I do not have enough information to answer that."
y_⊥(squad) = "The provided context does not contain information about that."
```

Both templates were chosen by inspecting the base model's natural abstentions
on held-out unanswerable inputs. Using the right template per domain is what
makes `μ⁻(d)` (the abstain pole, step 3) sit in a region the model can
actually reach with small LoRA updates, rather than dragging unrelated
activations along.

## Subspace V (step 2)

For each late layer `l`, form the over-commit and legitimate-commit
contrasts:

```
c_OC(x) = h_l(A) − h_l(B)     for x ∈ D_F      (over-commit minus its abstain baseline)
c_LC(x) = h_l(C) − h_l(D)     for x ∈ D_R_A    (legit-commit minus its abstain baseline)
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
utility. `V_l` is computed once and frozen, and is shared across both
answerability domains. Domain specialisation lives in the abstention
template `y_⊥(d)` (which shapes the contrasts going into V) and in the
per-domain abstain pole `μ⁻(d)` (step 3).

## Anchors (step 3)

Per answerability domain `d ∈ {kuq, squad}`:

```
μ_l⁻(d) = mean over D_F[d]   of  h_l(B[d])         (legitimate-abstention pole, domain d)
μ_l⁺(d) = mean over D_R_A[d] of  h_l(C[d])         (legitimate-commitment pole, domain d, diagnostic only)
```

The poles are points in 4096-D activation space — they don't depend on V.
`μ⁻(d)` is the forget target in step 4. `μ⁺(d)` is *not used in training*;
it's kept as a geometric diagnostic confirming `V` separates the legit-
commit cluster from the legit-abstain cluster.

KUQ (no context) and SQuAD (long context) sit in genuinely different regions
of late-layer hidden space — `||μ⁻_kuq − μ⁻_squad||` ≈ 50–85 across layers —
so a single grand-mean pole would miss each domain's true location.

Step 3 also computes the **per-domain initial L_forget magnitude**

```
s(d) = E_{x ∈ D_F[d]} ⟨ ‖ V_lᵀ (h_A(x) − μ_l⁻(d)) ‖² ⟩_l
```

and saves it alongside the poles. Step 4 divides each forget example by
its own domain's `s(d)` so KUQ and SQuAD enter the optimiser with equal
per-example pressure regardless of intrinsic contrast magnitude.

## Loss (step 4)

LoRA adapter `δθ` on `f_θ`. Two components with **role-appropriate signals**:

- `L_forget` is **geometric** — it changes the representation of category A
  in the discriminative subspace `V` by pulling it toward the per-domain
  abstain pole `μ⁻(d)`.
- `L_retain` is **supervised next-token cross-entropy** — it preserves the
  model's output distribution on retain examples (the gold answer for
  category C, the natural response for category E). It directly protects
  the LM head's decisions, which is what determines whether the model
  abstains or commits at inference time.

```
L = L_forget + λ · L_retain

L_forget = E_{(x,y) ∈ D_F}             ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ) − μ_l⁻(d_x)) ‖² ⟩_{l, t ∈ T(x)} / s(d_x)
L_retain = E_{(x,y) ∈ D_R_A ∪ D_R_G}   − ⟨ log p_{θ+δθ}(y_t | x, y_<t) ⟩_{t ∈ y_resp}
```

`d_x ∈ {kuq, squad}` is the source dataset of example `x`. The forget pole
`μ⁻(d_x)` is per-domain; `V` is shared; `s(d_x)` is the per-domain initial
forget magnitude (a constant from data, not a hyperparameter). `T(x)` is
the K-token transition window from above. `y_resp` is the response span:
the gold answer for D_R_A, the natural UltraChat response for D_R_G,
capped at `MAX_RETAIN_RESPONSE_TOKENS`. Prompt tokens are masked from CE
(`label = −100`) so the loss only fires on response positions.

**Why CE for retain.** A geometric retain term can preserve where activations
sit in `V`-space, but the final decision (commit vs. abstain) is made by
the LM head's output distribution. Geometric retain is invariant to drifts
in that distribution — the head can shift the next-token probability mass
toward EOS / abstention text while satisfying any geometric anchor. CE
constrains `p(y_t | x, y_<t)` directly, which is the right preservation
signal for *behavioural* unlearning. (V is whitened against `Σ_E` in step 2,
so the forget pull along `V` already does not push general-utility
directions; CE on D_R_G is the second line of defence.)

Effect, by category:

| Category | Training inputs | Loss target | Effect |
|---|---|---|---|
| A (over-commit)        | unanswerable + over-commit prefix | `μ⁻(d_x)` along `V`     | pulled toward legitimate abstention in its own domain |
| B (legit-abstain)      | not trained on directly           | —                        | preserved (μ⁻(d) is a fixed target)                  |
| C (legit-commit)       | answerable + gold answer          | gold-answer tokens (CE) | answer distribution held at the gold |
| D (over-abstain)       | not trained on directly           | —                        | not encouraged (no path to D)                        |
| E (general utility)    | UltraChat (prompt, response)      | response tokens (CE)    | next-token distribution held at the natural response |

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
│       └── anchors_<model>.pt              μ⁻(d), μ⁺(d), s(d) for d ∈ {kuq, squad} (μ⁺ diagnostic)
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
| 3    | `step3_build_anchors/build_anchors.py`    | `step1_extract_activations/`, `step2_build_subspace/` outputs            | `step3_build_anchors/data/anchors_<model>.pt`                       |
| 4    | `step4_train/train.py`                    | `step0_mine/`, `step2_*/`, `step3_*/` outputs                            | `step4_train/data/runs/<run_name>/`                                 |
| 5    | `step5_evaluate/evaluate.py`              | `step5_evaluate/data/heldout/{kuq,squad,ultrachat}.jsonl`                | `step5_evaluate/data/results/<name>/{kuq,squad,ultrachat}.json` (one JSON per dataset, metrics + per-row + baseline deltas) |

## Run end-to-end

```bash
# Step 0 — mine the model's over-commitment (fresh judge calls; needs CEREBRAS_TOKEN)
python3 step0_mine/mine.py                       --model qwen_instruct

# Steps 1–3 — build the subspace and anchors (one-time, per model)
python3 step1_extract_activations/extract.py     --model qwen_instruct
python3 step2_build_subspace/build_subspace.py   --model qwen_instruct --rank 32
python3 step3_build_anchors/build_anchors.py     --model qwen_instruct --rank 32

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
python3 step3_build_anchors/build_anchors.py     --model qwen_instruct --rank 16
python3 step5_evaluate/evaluate.py               --model qwen_instruct \
    --max-per-dataset 100 --max-ppl-rows 100
python3 step4_train/train.py                     --model qwen_instruct \
    --rank 16 --max-train-steps 50 --epochs 1
python3 step5_evaluate/evaluate.py               --run-dir step4_train/data/runs/<run> \
    --max-per-dataset 100 --max-ppl-rows 100 \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct
```

## Why this design

- **Role-appropriate losses.** Forget is geometric because *changing*
  behaviour is naturally formulated as moving the representation along a
  discriminative direction (`V` toward `μ⁻(d)`). Retain is supervised CE
  because *preserving* behaviour is naturally formulated as keeping the
  output token distribution unchanged — the LM head's decisions are what
  matters at inference. Two terms, one λ, each in the right space for its
  job.
- **Per-domain abstention template, pole, and init scale.** The two
  answerability conditions (no-context KUQ vs. with-context SQuAD) live in
  genuinely different regions of late-layer hidden space and abstain in
  genuinely different ways. The abstention template `y_⊥(d)`, the abstain
  pole `μ⁻(d)`, and the per-domain initial forget magnitude `s(d)` are all
  per-domain so the forget pull is natural in each domain — short,
  behaviorally-aligned, and equally weighted regardless of intrinsic
  contrast magnitude.
- **CE retain as the right preservation signal.** Geometric retain anchors
  hidden states; CE retain anchors output distributions. The diagnosed
  failure mode of pure-geometric retain is that the LoRA can shift LM-head
  logits toward EOS / abstention while still satisfying any geometric
  anchor, producing empty completions and false abstentions on retain
  inputs. CE on response tokens fixes that directly; geometric retain
  cannot, at any λ.
- **Categories drive ablations.** Each piece of the loss is named by which
  behaviour it is moving (A) or holding (C, E) and which anchor it uses.
  Drop `μ⁻(d)` → "no abstain pole". Set rank `r=0` → "no subspace". Drop
  `s(d)` → "shared init scale". Replace CE retain with geometric retain →
  "geometric-only retain". Drop CE on D_R_G → "no general-utility
  preservation". Each ablation removes exactly one component.
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
