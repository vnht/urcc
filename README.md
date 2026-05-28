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
`μ⁻(d)` is the forget target in step 4. `μ⁺(d)` is *no longer used in
training*; it's kept as a geometric diagnostic confirming `V` separates
the legit-commit cluster from the legit-abstain cluster (retain on category
C uses a per-example frozen-base reference instead, see below).

KUQ (no context) and SQuAD (long context) sit in genuinely different regions
of late-layer hidden space — `||μ⁻_kuq − μ⁻_squad||` ≈ 50–85 across layers —
so a single grand-mean pole would miss each domain's true location.

## Loss (step 4)

LoRA adapter `δθ` on `f_θ`. Two components, both projection-distance terms
along the shared subspace `V`, averaged over late layers `{l}` and the
`K=8`-position window `T(x) = {p_len − 1, …, p_len + K − 2}` defined above:

```
L = L_forget + λ · L_retain

L_forget = E_{(x,y) ∈ D_F}             ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ) − μ_l⁻(d_x)         ) ‖² ⟩_{l, t}
L_retain = E_{(x,y) ∈ D_R_A ∪ D_R_G}   ⟨ ‖ V_lᵀ (h_l(x, y; θ+δθ) − h_l(x, y; θ_frozen)) ‖² ⟩_{l, t}
```

`d_x ∈ {kuq, squad}` is the source dataset of example `x`. The forget pole
`μ⁻(d_x)` is per-domain so the target lives in the same prompt distribution
as the example; `V` is shared. Each example is divided by a per-domain
`init_scale[d]` (the expected step-0 value of `L_forget` for examples in
domain `d`, baked into the subspace bundle at step 2). Category-E retain
examples (UltraChat, no source domain) use the mean of the two domain
scales. This per-example rescaling makes KUQ and SQuAD contribute on the
same `O(1)` magnitude regardless of their domain-specific contrast size;
it does not change the optimum or `λ`, only the effective learning rate.

**Why retain uses a frozen-base reference, not μ⁺(d).** A pole-style anchor
(`μ⁺(d)`) is a *cluster-mean* target: many retain examples can collectively
drift, and the loss only measures the cluster's variance. The frozen-base
reference is a *per-example, per-token* target: each retain example pays a
sharp price for any drift on its own activation. This is a strict "do not
move from where you started" force per example, which is the symmetric
counterpart to the strong per-example "change behaviour" force in
`L_forget`. Empirically this asymmetry (per-example forget vs. per-cluster
retain) was what produced empty / degenerate completions in earlier runs;
making both sides per-example fixes it.

Effect, by category:

| Category | Forward inputs | Anchor target | Effect |
|---|---|---|---|
| A (over-commit)         | unanswerable + over-commit prefix | `μ⁻(d_x)` along `V`         | pulled toward legitimate abstention in its own domain |
| B (legit-abstain)       | not trained on directly           | —                            | preserved (anchor is fixed)                          |
| C (legit-commit)        | answerable + gold answer          | `h_l^frozen(x, y)` along `V` | held at own frozen-base activation, per-token       |
| D (over-abstain)        | not trained on directly           | —                            | not encouraged (no path to D)                        |
| E (general utility)     | UltraChat (prompt, response)      | `h_l^frozen(x, y)` along `V` | held at own frozen-base activation, per-token       |

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
│   ├── rejudge.py                  re-judge cached completions without regenerating
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
│       └── anchors_<model>.pt              μ⁻(d), μ⁺(d) for d ∈ {kuq, squad} (μ⁺ kept as diagnostic)
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
        │   ├── kuq.jsonl              trained domain — no-context
        │   ├── squad.jsonl            trained domain — with-context
        │   ├── selfaware.jsonl        held-out — no-context
        │   ├── falseqa.jsonl          held-out — no-context (false-premise)
        │   ├── qaqa.jsonl             held-out — no-context (false-premise)
        │   ├── truthfulqa.jsonl       held-out — no-context
        │   ├── faitheval.jsonl        held-out — with-context
        │   ├── musique.jsonl          held-out — with-context (multi-hop)
        │   ├── nomiracl.jsonl         held-out — with-context (no-answer retrieval)
        │   └── ultrachat.jsonl        utility — (prompt, response) for perplexity
        └── results/<run_name>/
            ├── kuq.json               answerability metrics + per-row (+ baseline deltas)
            ├── squad.json             (same shape, one JSON per dataset evaluated)
            ├── …                      one file per held-out set passed to --datasets
            └── ultrachat.json         perplexity metrics + per-row (+ baseline ratios)
```

## Pipeline

| Step | Script                                    | Reads                                                                    | Writes                                                              |
|------|-------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------|
| 0    | `step0_mine/mine.py`                      | `step0_mine/data/sampled/{kuq,squad}_unanswerable.jsonl`                 | `step0_mine/data/{mined,forget}/<model>_<dataset>.jsonl`            |
| 1    | `step1_extract_activations/extract.py`    | `step0_mine/data/{forget,sampled}/`                                      | `step1_extract_activations/data/activations_<model>.pt`             |
| 2    | `step2_build_subspace/build_subspace.py`  | `step1_extract_activations/data/activations_<model>.pt`                  | `step2_build_subspace/data/subspace_<model>_r<rank>.pt`             |
| 3    | `step3_build_anchors/build_anchors.py`    | `step1_extract_activations/data/activations_<model>.pt`                  | `step3_build_anchors/data/anchors_<model>.pt`                       |
| 4    | `step4_train/train.py`                    | `step0_mine/`, `step2_*/`, `step3_*/` outputs                            | `step4_train/data/runs/<run_name>/`                                 |
| 5    | `step5_evaluate/evaluate.py`              | `step5_evaluate/data/heldout/<dataset>.jsonl` (`--datasets` selection, default `kuq squad`) + `ultrachat.jsonl` | `step5_evaluate/data/results/<name>/<dataset>.json` (one JSON per dataset, metrics + per-row + baseline deltas) + `ultrachat.json` (perplexity + ratios) |

## Run end-to-end

```bash
# Step 0 — mine the model's over-commitment (fresh judge calls; needs CEREBRAS_TOKEN)
python3 step0_mine/mine.py                       --model qwen_instruct

# Steps 1–3 — build the subspace and anchors (one-time, per model)
python3 step1_extract_activations/extract.py     --model qwen_instruct
python3 step2_build_subspace/build_subspace.py   --model qwen_instruct --rank 32
python3 step3_build_anchors/build_anchors.py     --model qwen_instruct

# Step 5 (baseline first) — zero-shot reference
# Runs answerability (KUQ + SQuAD by default) AND UltraChat perplexity in one model load.
# Pass --datasets to also run the held-out generalisation sets.
python3 step5_evaluate/evaluate.py               --model qwen_instruct \
    --datasets kuq squad selfaware falseqa qaqa truthfulqa faitheval musique nomiracl

# Step 4 — train with the two-component UOC loss
python3 step4_train/train.py                     --model qwen_instruct \
    --rank 32 --lambda-retain 1.0 --epochs 3 --lr 3e-5

# Step 5 (trained) — evaluate the LoRA adapter and compare against the baseline
python3 step5_evaluate/evaluate.py               --run-dir step4_train/data/runs/<run_name> \
    --datasets kuq squad selfaware falseqa qaqa truthfulqa faitheval musique nomiracl \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct

# Plot training curves
python3 step4_train/plot_training.py             step4_train/data/runs/<run_name>
```

> Re-judging cached mining outputs (e.g. after tweaking the judge prompt)
> without paying GPU cost to re-generate:
> `python3 step0_mine/rejudge.py --model qwen_instruct`

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
  projection-distance terms along the same shared subspace `V`, just with
  different targets. One coefficient `λ` controls the trade-off; nothing else.
- **Per-domain abstention template and pole.** The two answerability
  conditions (no-context KUQ vs. with-context SQuAD) live in genuinely
  different regions of late-layer hidden space and abstain in genuinely
  different ways. The abstention template `y_⊥(d)` and the abstain pole
  `μ⁻(d)` are per-domain so the forget pull is *natural* in each domain —
  short, behaviorally-aligned, and the model's path to abstain at inference
  matches the path the loss trained it on.
- **Per-example retain, not per-cluster retain.** The retain loss anchors
  each legit-commit (C) and general-utility (E) example to its **own
  frozen-base activation** at every (layer, token) position. Unlike a
  pole-style mean target, this gives a strong, specific gradient against
  drift on each example. This per-example symmetry between forget (change)
  and retain (preserve) is what prevents the LoRA from finding degenerate
  solutions that collapse first-token logits to a chat-end token.
- **Categories drive ablations.** Each piece of the loss is named by which
  behaviour it is moving (A) or holding (C, E) and which anchor it uses
  (μ⁻(d), frozen-base reference). Drop μ⁻(d) → "no abstain pole". Set rank
  `r=0` → "no subspace". Replace frozen-base retain with μ⁺(d) → "cluster-
  anchor retain". Each ablation removes exactly one geometric component.
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

To add a new held-out evaluation dataset (step 5 only — no retraining):

1. Drop `step5_evaluate/data/heldout/<name>.jsonl` with rows shaped like
   the existing held-out sets (`{id, answerable, question, [context],
   correct_answer, ...}`).
2. Register it in `DOMAIN_OF` (`config.py`) mapping the dataset name to
   one of the two trained domains, `"kuq"` (no-context) or `"squad"`
   (with-context). This routes the dataset through the right prompt
   template and the right per-domain abstention template / pole at
   evaluation time, without any retraining.
3. Pass `--datasets ... <name>` to `step5_evaluate/evaluate.py`.
