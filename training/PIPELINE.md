# URC Training Pipeline

End-to-end walkthrough of the URC (Unlearning Residual Commitment) pipeline,
from the three raw datasets through to post-training evaluation.

---

## Datasets at a glance

| Dataset    | Role                                           | File                                               | Size   |
|------------|------------------------------------------------|----------------------------------------------------|--------|
| KUQ        | Open-domain unanswerable / answerable questions | `mining-data/sampled/kuq_*_*.jsonl`                | 2 500  |
| SQuAD      | Context-grounded unanswerable / answerable     | `mining-data/sampled/squad_*_*.jsonl`              | 2 500  |
| UltraChat  | General-purpose retain examples                | `mining-data/sampled/ultrachat_retain_1000.jsonl`  | 1 000  |

Each "answerable" file pairs a `question` (and optional `context`) with
`correct_answer`. Each "unanswerable" file pairs a `question` with
`answerable=False`. The UltraChat retain file is `(prompt, response)` pairs.

---

## Stage 0 — Sampling raw evaluation pools

Files under `training/held-out-eval-data/` (`kuq_1000.jsonl`,
`squad_1000.jsonl`, `ultrachat_1000.jsonl`) are the **held-out 1 000-instance
evaluation pools**. They are generated once via
`training/held-out-eval-data/sample.py` and never seen during training.

The training pools under `mining-data/sampled/` are disjoint from these
evaluation pools.

---

## Stage 1 — Mine committed completions

Script: `mining-data/mine.py`

For each model in the registry:

1. Load the model and tokenizer.
2. For every question in `kuq_unanswerable_2000.jsonl` and
   `squad_unanswerable_2000.jsonl`, generate one **deterministic** completion
   using the model's chat template (or raw prompt for base models).
3. Send each completion to the Cerebras `gpt-oss-120b` judge with the prompt
   defined in `evaluation/judge.py` to label it `COMMITTED` or `ABSTAINED`.
4. Save **all** rows to `mining-data/mining-results/<model>_<dataset>.jsonl`,
   tagged with the judge label and the first 8 tokens of the committed
   completion (`y_com_prefix_k8`).

The result is a per-model record of which unanswerable questions the model
**confidently committed to a wrong answer for**. Those are the rows we want
to unlearn.

---

## Stage 2 — Curate top examples per model

Script: `mining-data/curate.py`

1. Filter `COMMITTED` rows.
2. Score each row by judge confidence and answer length.
3. Keep the top 500 per `(model, dataset)` pair, with optional borderline
   backfill if fewer than 500 are eligible.
4. Save to `mining-data/mining-selected/<model>_<dataset>.jsonl`.

These selected JSONLs are the **forget set** for training.

---

## Stage 3 — Extract residual-stream activations

Three scripts, all writing to `mining-data/activations/`:

| Script                              | Inputs                                                           | Output                                                          | Purpose                                          |
|-------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------|--------------------------------------------------|
| `extract_activations.py`            | mining-selected forget rows                                      | `unsupported_commitment_contrasts_<model>.pt`                   | Forget contrasts: `c = h_committed_answer − h_prompt`  |
| `extract_supported_activations.py`  | answerable QA pairs                                              | `supported_answer_contrasts_<model>.pt`                         | Supported contrasts: same shape, but the model is *correctly* answering an answerable question |
| `extract_retain_activations.py`     | `ultrachat_retain_1000.jsonl`                                    | `retain_activations_<model>_last25.pt`                          | Retain activations on UltraChat responses        |

All scripts hook the model's residual stream at the **last 25 % of layers**
(e.g. layers 24–31 for an 8-layer slice of a 32-layer model) and stack the
hidden states into tensors of shape `(N, L, D)`.

---

## Stage 4 — Clean the forget contrasts

Script: `mining-data/compute_cleaned_contrasts.py`

The raw forget contrasts contain both **commitment** signal (what we want to
suppress) **and** **answer-content** signal (what we want to preserve). To
isolate commitment we project out the supported-answer subspace from the
unsupported contrasts:

```
c_clean_l = c_unsupported_l − V_r V_rᵀ c_unsupported_l
```

where `V_r` is the rank-`r` PCA basis of the supported-answer contrasts at
layer `l`. Saved to:

```
mining-data/activations/cleaned_unsupported_contrasts_<model>_last25_r8.pt
```

A diagnostics PCA is also written to
`cleaned_pca_<model>_last25_r8.pt` and used later to inspect how much
commitment variance the chosen rank captures.

---

## Stage 5 — Build the retain-normalised commitment subspace

Three subspace formulations are available; all share the same retain-span
projection plumbing and produce drop-in compatible bundles for training.

| Variant | Script | Σ_C source | What it picks |
|---|---|---|---|
| `clean` (default) | `mining-data/retain_normalised_subspace.py` | `c_unsupported_clean` (post supported-PCA cleaning) | Σ_C v = γ Σ_R v |
| `raw`             | `mining-data/raw_subspace.py`               | `c_unsupported` (no cleaning) | Σ_C v = γ Σ_R v |
| `disc`            | `mining-data/discriminative_subspace.py`    | raw `c_unsupported` and `c_supported` | (Σ_C − Σ_A) v = γ Σ_R v |

`disc` is the most behaviourally targeted: it explicitly subtracts the
supported-answerable covariance, so the resulting directions suppress
unsupported commitment far more than legitimate answer-giving (C/A ≈ 4
vs 0.96 for `raw`). Side-by-side analysis lives in
`mining-data/compare_subspaces.py` and the build scripts.

For each layer `l` we want directions `v` along which **commitment is large
but retain is small** (and, for `disc`, also large *relative to supported
answering*). That is the generalised eigenproblem

```
Σ_C v = γ Σ_R v          # clean / raw
(Σ_C − Σ_A) v = γ Σ_R v  # disc
```

Naively solving this in `D × D` collapses into the null space of `Σ_R`
because retain only spans `~ N_retain` dimensions. The scripts fix that
with retain-span projection:

1. SVD of centred `R_l` → retain basis `W ∈ ℝ^{D × k}`,
   `k = min(N_retain − 1, retain_rank)`.
2. Project both `C_l` and `R_l` into `W`-space.
3. Solve the well-conditioned `eigh(Σ_C_proj, Σ_R_proj + ridge·I)` in
   `k`-space.
4. Map the top-`rank` eigenvectors back to `ℝ^D` and unit-normalise.

Output bundle (per model, per rank, per variant):

```
mining-data/activations/retain_normalised_subspace_<model>_last25_r<rank>.pt   # clean
mining-data/activations/raw_subspace_<model>_last25_r<rank>.pt                  # raw
mining-data/activations/discriminative_subspace_<model>_last25_r<rank>.pt       # disc
```

Keys:

| Key                         | Shape          | Meaning                                                                  |
|-----------------------------|----------------|--------------------------------------------------------------------------|
| `V_retain_normalised`       | `(L, D, rank)` | The orthonormal commitment-direction matrix per layer                    |
| `generalized_eigenvalues`   | `(L, rank)`    | γ values; bigger = more commitment per unit retain                       |
| `commitment_projection`     | `(L,)`         | `tr(Vᵀ Σ_C V)` per layer — averaged across layers to give `proj_norm_scale`, the denominator of `L_forget` |
| `retain_projection`         | `(L,)`         | `tr(Vᵀ Σ_R V)` per layer — sanity-check; should be ~1 after retain-span normalisation |
| `layers`                    | list           | Layer indices (the last 25 % slice)                                      |

---

## Stage 5b — Build the abstention anchor (optional but recommended)

Script: `mining-data/extract_abstention_anchors.py`

Pure projection-magnitude suppression (`L = ||VᵀH||²`) is a *one-sided*
loss: it tells the model what *not* to be but provides no preferred
destination. Empirically the optimiser collapses by globally damping
activations, which manifests as the "answer 'No' to everything" failure
mode.

The anchor is a single fixed reference point per layer:

```
μ_l = mean residual activation at the answer-token position when the
      original model successfully abstained on an unanswerable question
```

It is computed by filtering the existing mining results for
`judge_label == "ABSTAINED"`, replaying those `(prompt, completion)`
pairs through the base model, and averaging the residual stream at the
last-25 % layers over the first `K=8` answer-token positions.

Output:

```
mining-data/activations/abstention_anchors_<model>_last25.pt
    {
      "model":      str,
      "model_key":  str,
      "layers":     list[int],
      "k":          8,
      "n_examples": int,
      "mu_abstain": tensor[L, D],
      "datasets":   list[str],
    }
```

When training uses `--anchor`, the forget loss becomes:

```
L_forget(l, t) = || V_lᵀ ( h_l(t) − μ_l ) ||²
```

`μ` is a constant (no gradient, no learnable target). Geometrically
this changes the **origin** of the forget-loss coordinate system: the
optimiser now drives `h_forget` toward `μ_abstain` along the
commitment subspace, instead of toward zero. This stays purely
representation-engineering — no token-level supervision is added.

```bash
python3 mining-data/extract_abstention_anchors.py --model qwen_instruct
```

Counts of available abstention examples per model (from current
mining-results) — all sufficient to estimate `μ`:

| Model | KUQ | SQuAD | Total |
|---|---:|---:|---:|
| qwen_instruct | 242 | 986 | 1228 |
| qwen_base | 196 | 594 | 790 |
| ministral_instruct | 79 | 118 | 197 |
| ministral_base | 45 | 46 | 91 |

---

## Stage 6 — Pre-training baselines

Folder: `training/pre-training-baselines/`

Before any unlearning, each model is evaluated against the held-out pools to
record reference behaviour:

1. `generate.py` runs the model on `kuq_1000.jsonl` and `squad_1000.jsonl`
   → `eval_baseline_generations_<model>.jsonl`.
2. The completions are judged with the same Cerebras judge → labels merged
   back into the same files.
3. `compute_metrics` (in `evaluation/judge_outputs.py`) aggregates per-dataset
   answerability metrics → `eval_baseline_answerability_metrics.jsonl`.
4. Teacher-forced perplexity on `ultrachat_1000.jsonl` →
   `eval_baseline_ultrachat_ppl.jsonl`.

These baselines are what every post-training run is compared against.

---

## Stage 7 — URC training

Script: `training/train_urc.py`

```bash
python3 training/train_urc.py \
    --model qwen_instruct \
    --rank 32 \
    --subspace disc \
    --anchor \
    --epochs 3 \
    --lr 3e-5 \
    --beta 1.0
```

`--subspace {clean,raw,disc}` selects which bundle to suppress (default
`clean`). `--anchor` re-centres the forget loss on `μ_abstain`
(requires Stage 5b to have run). The run name embeds both choices
(`{model}_{label}[_anchor]_urc_...`) so variants don't collide in
`training/runs/`.

### 7.1 Data assembled at start-up

| Source                                                 | Used for                                |
|--------------------------------------------------------|------------------------------------------|
| `mining-selected/<model>_kuq.jsonl` + `_squad.jsonl`   | **Forget set** (committed unanswerable rows) |
| `sampled/ultrachat_retain_1000.jsonl`                  | **Retain set**, general capability       |
| `sampled/kuq_answerable_500.jsonl`                     | **Retain set**, factual commitment       |
| `sampled/squad_answerable_500.jsonl`                   | **Retain set**, context-grounded commitment |
| `activations/retain_normalised_subspace_<model>_last25_r<rank>.pt` | Forget-loss target subspace |

The two answerable QA files are mixed into the retain pool to prevent the
trivial degenerate solution of always saying "No" — the retain KL then
penalises any drift from the model's correct factual output on real
questions.

### 7.2 Model and adapter

1. Base model loaded in `bfloat16` with `device_map="auto"`.
2. **LoRA adapter** applied via `peft` to all attention and MLP projections
   (`q,k,v,o,up,down,gate`):
   - LoRA rank: `LORA_R = 16` (separate from subspace rank)
   - LoRA α: `32`
   - LoRA dropout: `0.05`
3. Only adapter parameters are trainable (~30 M for Qwen-9B,
   ~50 M for Ministral-8B). Base weights stay frozen.

### 7.3 The two losses

#### Forget loss `L_forget`

For each forget example we tokenise `prompt + y_com_prefix_k8`, identify the
last `K_ANSWER_TOKENS = 8` positions (the answer tokens), and at every layer
in the last-25 % slice project the residual stream onto the commitment
subspace. The per-layer loss is the expected squared projection-vector norm
per answer token (sum over rank dims, mean over tokens):

```
proj    = h_answer_tokens @ V_l                          # (n_ans, rank)
L_l     = mean_token(‖proj_vec‖²) = (proj²).sum(-1).mean()
L_forget_raw    = mean over selected layers of L_l
L_forget_scaled = L_forget_raw / proj_norm_scale
```

`proj_norm_scale = mean_layers(tr(Vᵀ Σ_C V))` is the expected per-token
‖proj_vec‖² at initialisation, computed from the same contrasts used to
build the subspace. It comes directly out of the subspace bundle as
`commitment_projection.mean()`.

**Same units, same meaning** — both numerator and denominator measure
expected per-token commitment magnitude, so `L_forget_scaled ≈ 1` at the
start of training and trends toward 0 as projections shrink. This holds
across rank settings: rank-8, rank-32 and rank-64 all start near 1.

#### Retain loss `L_retain`

For each retain example we tokenise `prompt + response` (using the model's
chat template for instruct models) and run **two** forward passes through
the same PEFT model:

1. **Frozen reference** — `model.eval()` + `disable_adapter_layers()` +
   `torch.no_grad()`. This produces the original-weights distribution.
2. **Trainable** — `model.train()` + `enable_adapter_layers()`. Gradients
   flow through the adapter only.

We then compute **top-100 KL divergence** at each response position:

```
top_k_idx        = top_k(frozen_logits, 100)
log_p_frozen     = log_softmax(gather(frozen_logits, top_k_idx))
log_p_trainable  = log_softmax(gather(trainable_logits, top_k_idx))
L_retain         = mean( p_frozen · (log_p_frozen − log_p_trainable) )
```

Top-100 truncation keeps the loss memory-cheap and eliminates the long
low-probability vocab tail without changing the qualitative behaviour.

#### Total

```
L_total = L_forget_scaled + β · L_retain
```

### 7.4 Optimiser and schedule

| Hyper-parameter      | Value                                          |
|----------------------|------------------------------------------------|
| Optimiser            | AdamW                                          |
| Learning rate        | `--lr` (default `1e-5`; `3e-5` works well in practice) |
| Weight decay         | 0                                              |
| Warmup ratio         | 3 % of total steps                             |
| Schedule             | linear warmup → linear decay to 0              |
| Forget batch         | 4 examples / step                              |
| Retain batch         | 4 examples / step                              |
| Gradient accumulation| 4 micro-steps → 1 optimiser step               |
| Max grad norm        | 1.0                                            |

Total optimiser steps per epoch ≈ `ceil(N_forget / 4) / 4`, e.g. 63 for
1 000 forget rows.

### 7.5 Inner loop

For each optimiser step:

1. Pull next forget batch (4 rows) and retain batch (4 rows).
2. Compute `L_forget` on the forget batch.
3. Compute `L_retain` on the retain batch.
4. Sum to `L_total`, divide by `GRAD_ACCUM_STEPS`, backward.
5. After 4 micro-steps: clip to `MAX_GRAD_NORM`, optimiser step, scheduler
   step, zero grads.
6. Log `L_total`, `L_forget`, `top100_retain_KL`, `mean_proj_norm`,
   `learning_rate`, `grad_norm` to `loss_log.csv`.

### 7.6 Early stopping

Two checks run **after every optimiser step**:

| Trigger                                                                         | Default        |
|---------------------------------------------------------------------------------|----------------|
| `L_forget` does not improve by at least `--es-delta` over `--es-patience` steps | 0.05 / 20      |
| Retain `KL` exceeds `--kl-max`                                                  | 0.15           |

Either check can be disabled by passing `0`.

### 7.7 Output artefacts

```
training/runs/<run_name>/
├── adapter_config.json          # PEFT config
├── adapter_model.safetensors    # LoRA weights
├── training_config.json         # all hyper-params + paths
├── subspace_config.json         # subspace bundle metadata used
├── loss_log.csv                 # per-step losses
└── train_summary.json           # initial / final / delta values + early-stop flag
```

The run name encodes the key knobs:
`<model>_retainnorm_urc_last25_r<rank>_beta<β>_ep<epochs>_lr<lr>`.

---

## Stage 8 — Post-training evaluation

Script: `training/eval_urc.py`

```bash
python3 training/eval_urc.py \
    --run-dir training/runs/<run_name>
```

The script writes everything to `<run_dir>/eval/`.

### 8.1 Generation

1. Load the base model and apply the saved LoRA adapter.
2. Generate one completion per row of `held-out-eval-data/kuq_1000.jsonl`
   and `squad_1000.jsonl` using greedy decoding and the model's chat
   template (with `<think>` blocks stripped post-decode for Qwen3).
3. Append rows to `eval/eval_generations_<model>.jsonl`. The file is
   resumable — re-running picks up where it left off.

### 8.2 Judging

Each row is sent to the Cerebras `gpt-oss-120b` judge with the same
template as the baseline. Labels are merged back into the same generations
file (`judge_label`, `judge_raw`).

### 8.3 Answerability metrics

Per dataset (KUQ, SQuAD):

| Metric                  | Definition                                              |
|-------------------------|---------------------------------------------------------|
| `true_commitment_rate`  | answerable & COMMITTED / answerable                     |
| `false_abstention_rate` | answerable & ABSTAINED / answerable                     |
| `true_abstention_rate`  | unanswerable & ABSTAINED / unanswerable                 |
| `false_commitment_rate` | unanswerable & COMMITTED / unanswerable                 |
| `decision_accuracy`     | (TC + TA) / total non-error                             |

Saved to `eval/eval_answerability_metrics.jsonl`.

### 8.4 Perplexity (optional)

If `--skip-ppl` is **not** passed, the script computes teacher-forced
perplexity on `ultrachat_1000.jsonl` (using the same chat template + token
masking as the baseline) and writes:

```
eval/eval_ultrachat_ppl_<model>.jsonl     # per-instance NLL
eval/eval_ultrachat_ppl.jsonl              # aggregate summary
```

### 8.5 Comparison to baseline

For every metric, the script computes
`delta = post − baseline` and writes `eval/eval_comparison.json`. It also
prints a compact table to the console:

```
dataset     metric    baseline      post      delta
----------------------------------------------------
kuq         true_…    0.184         0.674    +0.490
…
```

---

## End-to-end flow (visual)

```
sampled/{kuq,squad}_*.jsonl
        │
        ▼
mine.py  ──►  mining-results/
        │
        ▼
curate.py  ──►  mining-selected/
        │
        ├── extract_activations.py            ──►  unsupported_commitment_contrasts_*.pt
        ├── extract_supported_activations.py  ──►  supported_answer_contrasts_*.pt
        └── extract_retain_activations.py     ──►  retain_activations_*.pt
                       │
                       ▼
            compute_cleaned_contrasts.py ──►  cleaned_unsupported_contrasts_*.pt
                       │
                       ▼
            retain_normalised_subspace.py ──► retain_normalised_subspace_*.pt
                       │
                       ▼
                 train_urc.py  (uses subspace + forget + retain pools)
                       │
                       ▼
                  runs/<run_name>/adapter_model.safetensors
                       │
                       ▼
                 eval_urc.py  ──► runs/<run_name>/eval/
```

---

## Quick reference: minimal runbook for a new model

1. `python3 mining-data/mine.py --model <model>`
2. `python3 mining-data/curate.py`
3. `python3 mining-data/extract_activations.py --model <model>`
4. `python3 mining-data/extract_supported_activations.py --model <model>`
5. `python3 mining-data/extract_retain_activations.py --model <model>`
6. `python3 mining-data/compute_cleaned_contrasts.py --model <model>`
7. `python3 mining-data/retain_normalised_subspace.py --model <model> --rank <r>`
8. `python3 training/pre-training-baselines/run_baselines.py --model <model>`  *(once)*
9. `python3 training/train_urc.py --model <model> --rank <r> --epochs <e> --lr <lr>`
10. `python3 training/eval_urc.py --run-dir training/runs/<run_name>`
