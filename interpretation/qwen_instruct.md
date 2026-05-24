# UOC pipeline — interpretation for `qwen_instruct`

Geometric diagnostics from steps 0–3 of the Unlearning Over-Commitment (UOC)
pipeline run against `Qwen/Qwen3.5-9B`. Each section reports raw numbers
from the saved bundles and what they imply for the training step that follows.

---

## 0. Run configuration

| Item | Value |
|------|-------|
| Model key            | `qwen_instruct` |
| Model id             | `Qwen/Qwen3.5-9B` |
| Hidden dim `D`       | 4096 |
| Layer slice (last 25%) | `[24, 25, 26, 27, 28, 29, 30, 31]`  → `L = 8` |
| Answer-token window `K` | 8 |
| Subspace rank `r`    | 32 |
| Subspace ridge       | `1e-3` |
| Retain basis rank    | 512 |
| Forget templates     | `KUQ_PROMPT_TEMPLATE`, `SQUAD_PROMPT_TEMPLATE` |
| Templated abstention | `"I do not have enough information to answer that."` |

Datasets:

| Pool | Source | Size | Role |
|------|--------|------|------|
| `D_F` (forget)         | KUQ + SQuAD unanswerable, COMMIT-only | 500 + 500 = **1000** | Category A |
| `D_R_A` (retain QA)    | KUQ + SQuAD answerable + gold         | 500 + 500 = **1000** | Categories C, D, μ⁺ |
| `D_R_G` (retain general) | UltraChat (prompt, response)         | **1000**            | Category E |

---

## 1. Step 0 — mining

Greedy completions on 2 000 unanswerable prompts per dataset, judged by
Cerebras `gpt-oss-120b` into `COMMIT` / `ABSTAIN`. The COMMIT subset is then
filtered down to the 500 most obvious overcommits per dataset (length-ranked
after hard-excluding hedge / premise-rejection / source-deferral patterns and
deduplicating on `(question, completion)`).

| Dataset | Mined | COMMIT | ABSTAIN | judge_error | Forget kept |
|---------|------:|-------:|--------:|------------:|------------:|
| KUQ     | 2 000 | 1 251 | 749     | 0           | 500 |
| SQuAD   | 2 000 |  698 | 1 301   | 1           | 500 |

**Reading:** Qwen3.5-9B over-commits on **62.5%** of KUQ prompts but only
**34.9%** of SQuAD prompts. KUQ is harder for the model — it tests
common-knowledge questions where the model is unwilling to abstain — while
SQuAD requires copying from context, where saying *"the context does not say"*
comes more naturally to the instruct model. After hard-exclude filtering,
both forget files are clean enough to use as Category A training data: a
manual sweep at the top of each forget file shows substantive answers
attempting to resolve the question, with no leftover ABSTAIN-shaped behaviour.

---

## 2. Step 1 — activation extraction

`step1_extract_activations/data/activations_qwen_instruct.pt`

Per-set tensors of shape `(N, L, D)` where the answer-token window of size
`K=8` has already been mean-collapsed inside extraction.

| Set | Description                          | `N`  | Shape           | Mean per-row `‖h‖` | Std |
|-----|--------------------------------------|-----:|-----------------|-------------------:|----:|
| A   | Over-commit                          | 1 000 | (1000, 8, 4096) | 117.55             | 27.5 |
| B   | Legit abstention (templated)         | 1 000 | (1000, 8, 4096) | 111.17             | 24.0 |
| C   | Legit commitment (gold answer)       | 1 000 | (1000, 8, 4096) | 118.08             | 29.8 |
| D   | Over-abstention (templated on `D_R_A`) | 1 000 | (1000, 8, 4096) | 105.75             | 23.2 |
| E   | General utility (UltraChat)          | 1 000 | (1000, 8, 4096) | 108.57             | 24.6 |

Wall-clock: 15 m 45 s.

### 2.1 Contrast geometry

The two cross-set contrasts that drive the eigenproblem in step 2:

```
c_OC(x) = h_A(x) − h_B(x)        per-row over-commit contrast
c_LC(x) = h_C(x) − h_D(x)        per-row legit-commit contrast
```

| Quantity | Value |
|----------|------:|
| Per-row mean `‖c_OC‖` | 208.4 |
| Per-row mean `‖c_LC‖` | 213.0 |
| `cos(mean c_OC, mean c_LC)` | **+0.842** |

`cos = +0.842` is the central diagnostic. It says **most** of what differs
between A and B is the same direction that differs between C and D — the
"commit-vs-abstain mode". The gap below 1.0 is the residual signal that is
*uniquely about over-committing*. Step 2's eigenproblem
`(Σ_OC − Σ_LC) v = γ Σ_E v` is built to amplify exactly that residual.

### 2.2 Per-layer covariance traces (proxy for eigen-signal)

| Layer | `tr Σ_OC` | `tr Σ_LC` | `tr Σ_E` | OC / LC | OC / E |
|------:|----------:|----------:|---------:|--------:|-------:|
| 24    |  3 091    |  3 388    |  3 820   | 0.91    | 0.81   |
| 25    |  3 592    |  3 937    |  4 492   | 0.91    | 0.80   |
| 26    |  3 891    |  4 343    |  5 065   | 0.90    | 0.77   |
| 27    |  5 091    |  5 810    |  5 984   | 0.88    | 0.85   |
| 28    |  6 341    |  7 192    |  7 434   | 0.88    | 0.85   |
| 29    |  7 460    |  8 750    |  8 862   | 0.85    | 0.84   |
| 30    |  8 977    | 10 851    | 11 056   | 0.83    | 0.81   |
| 31    |  3 441    |  4 481    |  6 072   | 0.77    | 0.57   |

`Σ_OC < Σ_LC` everywhere by 9–23%. Counter-intuitive at first, but
consistent with: over-commits are stylistically stereotyped (a few canonical
confident-but-wrong shapes) while legit-commits span every topic in the
retain pool, so total variance is lower for OC. Trace is the sum of
eigenvalues; what matters for V is whether **specific directions** have
`Σ_OC > Σ_LC`. They do — see step 2 below.

L31 contracts sharply (everything drops ~½×). Standard last-layer
"residual-stream collapses into LM-head logit space" pattern.

---

## 3. Step 2 — discriminative subspace V

`step2_build_subspace/data/subspace_qwen_instruct_r32.pt`,
`V` shape `[8, 4096, 32]`.

Solves the generalized eigenproblem per layer:
```
(Σ_OC − Σ_LC) v = γ (Σ_E + ridge·I) v       picking top r=32 v's
```

| Layer | γ_1   | γ_32  | OC_proj | LC_proj | E_proj | OC/LC | OC/E |
|------:|------:|------:|--------:|--------:|-------:|------:|-----:|
| 24    | 27.80 | 1.145 | 112.69  | 15.80   | 31.92  | **7.13** | 3.53 |
| 25    | 23.97 | 1.092 | 105.12  | 14.61   | 31.92  | **7.20** | 3.29 |
| 26    | 21.22 | 1.060 |  99.71  | 13.39   | 31.92  | **7.45** | 3.12 |
| 27    | 19.92 | 0.991 |  93.70  | 13.84   | 31.93  | 6.77    | 2.94 |
| 28    | 19.60 | 1.005 |  93.40  | 13.89   | 31.93  | 6.72    | 2.93 |
| 29    | 20.19 | 0.998 |  93.02  | 14.52   | 31.93  | 6.41    | 2.91 |
| 30    | 19.40 | 1.013 |  92.20  | 14.43   | 31.93  | 6.39    | 2.89 |
| 31    | 16.02 | 0.743 |  71.97  | 12.72   | 31.92  | 5.66    | 2.25 |

Wall-clock: 2 s.

`OC_proj`, `LC_proj`, `E_proj` are `tr(V⊤ Σ_X V)` per set — the variance each
set carries inside V.

### Reading

* **`γ_32 ≈ 1.0` at every layer.** Even the 32nd-best direction still has
  `Σ_OC > Σ_LC` after E-whitening. Rank 32 is real budget, not padding.
* **OC / LC ≈ 6 – 7×** in V vs ≈ 0.85 in full space. The eigenproblem
  produces an **8× selectivity gain** with a single linear projection.
* **OC / E ≈ 3×.** General-utility activations move ~⅓ as much in V as
  over-commit contrasts do — modest margin; main risk for retain integrity.
* **`E_proj ≈ 31.92` everywhere by construction** — V is whitened against
  `Σ_E + ridge·I`, so `tr(V⊤ Σ_E V) ≈ r = 32`.
* **Layer pattern.** L24–L26 lead in selectivity (γ_1 = 28, 24, 21); L31
  trails (γ_1 = 16). Over-commit-specific geometry lives in the early-late
  layers; the very last layer is too close to LM-head logit space to keep
  behaviours distinct. Best of both: L26 / L28 (high γ_1 *and* high
  OC_proj).
* **Smooth spectrum.** `γ_1 / γ_32 ≈ 22–24` everywhere. No abrupt knee →
  rank=32 is reasonable; ablations at rank ∈ {16, 64} should be informative
  but neither change is forced by the spectrum.

---

## 4. Step 3 — pole anchors μ⁻, μ⁺

`step3_build_anchors/data/anchors_qwen_instruct.pt`, both poles shape `[8, 4096]`.

```
μ_l⁻ = mean over rows of  h_B   (templated abstention on D_F prompts)
μ_l⁺ = mean over rows of  h_C   (gold answer on D_R_A prompts)
```

`n_minus = 1000`, `n_plus = 1000`. Wall-clock: < 1 s.

### Per-layer pole geometry

| Layer | `‖μ⁻‖` | `‖μ⁺‖` | `‖μ⁻ − μ⁺‖` | `‖V⊤(μ⁻ − μ⁺)‖` | shrinkage | `cos(μ⁻, μ⁺)` |
|------:|-------:|-------:|------------:|----------------:|----------:|--------------:|
| 24    |  75.72 |  61.13 |  60.22      | 10.38           | 17.2%     | +0.6313       |
| 25    |  84.34 |  67.63 |  64.65      | 11.14           | 17.2%     | +0.6582       |
| 26    |  97.53 |  79.37 |  68.77      | 11.37           | 16.5%     | +0.7159       |
| 27    | 103.79 |  82.46 |  72.86      | 12.04           | 16.5%     | +0.7165       |
| 28    | 118.41 | 105.90 |  78.89      | 14.65           | 18.6%     | +0.7581       |
| 29    | 132.02 | 119.51 |  86.22      | 15.47           | 17.9%     | +0.7694       |
| 30    | 149.26 | 126.26 |  97.31      | 17.74           | 18.2%     | +0.7628       |
| 31    | 101.67 |  80.57 |  64.51      | 14.09           | 21.8%     | +0.7731       |

### Reading

* **Real, well-separated anchors.** Full-space `‖μ⁻ − μ⁺‖` = 60–97 across
  layers. Both poles are large (60–149).
* **`‖μ⁻‖ > ‖μ⁺‖` consistently** by 14–24. Templated abstention is the same
  text on every example, so its mean is a sharp representation. Gold answers
  vary across topics, so averaging suppresses the answer-specific component
  and leaves mostly the layer's "I'm answering substantively" mode. μ⁻ is
  the cleaner of the two anchors.
* **`cos(μ⁻, μ⁺) ≈ 0.7`** — moderate-strong overlap. Healthy: cos near 1
  would mean the loss has nothing to push on; cos near 0 would mean abstain
  and commit are unrelated axes (also workable but surprising). 0.7 says
  abstain and commit share a layer-typical content envelope but differ in a
  stable, learnable direction.
* **Projected target distance `‖V⊤(μ⁻ − μ⁺)‖ ≈ 10–18`.** This is what the
  loss actually sees in V. Largest at L30 (17.7); smallest at L24 (10.4).
* **Shrinkage 17–22% ≈ 2× random baseline.** A random 32-dim subspace in
  4096-dim would shrink the pole separation by `√(32/4096) ≈ 8.8%`. We see
  ~18%, meaning V *partially* aligns with the pole direction — enough to
  give the forget loss a clear signal toward μ⁻, but not so much that
  pulling A toward μ⁻ also drags C away from μ⁺. The orthogonal 80% of the
  pole separation lives outside V where the loss can't reach it.

---

## 5. Synthesis — what the geometry says about the method

The three things UOC's loss needs are all empirically present for this
model.

| Requirement | Observed | Verdict |
|-------------|----------|---------|
| A unique low-rank direction where over-commit dominates | `OC/LC = 7×` in `V`, every `γ_k > 0` for `k ≤ 32` | ✓ strong |
| A target point inside that direction to pull A toward    | `‖V⊤(μ⁻ − μ⁺)‖ ≈ 10–18` per layer                | ✓ strong |
| Headroom so the pull doesn't damage C and E              | `OC/LC = 7×`, `OC/E = 3×`; `LC_proj ≈ 14`, `E_proj ≡ 32` | ✓ moderate (E is the watch) |

### Consequence for the loss

```
L = L_forget + λ · L_retain
L_forget        = mean_x∈D_F   ‖V⊤(h_A(x) − μ⁻)‖²
L_retain[C]     = mean_x∈D_R_A ‖V⊤(h_C(x) − μ⁺)‖²
L_retain[E]     = mean_x∈D_R_G ‖V⊤(h_E(x) − h_E_frozen(x))‖²
```

Initial-step expectations:

* **`L_forget`.** `V⊤ h_A` has RMS `≈ √OC_proj ≈ 10` per layer. `V⊤ μ⁻`
  sits ~10–18 away. Initial loss is on the order of `100–300` per layer;
  should fall fast — quadratic with strong gradient.
* **`L_retain[C]`.** `V⊤ h_C − V⊤ μ⁺` is the projection of legit-commit
  fluctuation around its own pole. Expected initial loss ~`13–16` per
  layer. Should stay near initial value throughout training; a rise means λ
  is too low or β too high.
* **`L_retain[E]`.** Reference-anchored, so initial value is 0 by
  construction. Will grow as LoRA changes the model. Capped by E_proj ≈ 32
  per layer — any rise above ~8 should trigger an eyebrow.

### Predicted layer roles

* **L24–L26.** Cleanest selectivity (γ_1 = 21–28). Forget gradient comes
  with the least collateral on legit-commits.
* **L28–L30.** Largest absolute pole gap (`‖V⊤(μ⁻ − μ⁺)‖` = 14.6–17.7).
  Forget loss has the most signal to drop here.
* **L31.** Weaker on every metric; useful as a sanity-anchor but not the
  workhorse layer.

### Predicted step 5 metrics (qualitative)

* **FCR (false-commit rate, unanswerable held-out).** This is what V is
  built to suppress; substantial drop expected.
* **Decision accuracy (answerable held-out).** Should hold within a few
  points of the base model — `LC_proj` is small in V (only ~14 worth of
  variance budget for the model to disturb).
* **UltraChat preservation.** The hardest call from geometry alone.
  `OC/E = 3×` is a modest margin; if UltraChat outputs visibly degrade
  before FCR drops, β is too high.

### Risks the data flags

* **E-set margin is the bottleneck.** OC/E ≈ 3× vs OC/LC ≈ 7×. Don't push
  β beyond ~1.5 on the first run; UltraChat will be the first thing to
  show damage.
* **Stylistic over-commit ≠ all over-commit.** Σ_OC has lower trace than
  Σ_LC because over-commits are stereotyped. The forget set captures the
  most obvious cases (KUQ + SQuAD). Real-world over-commit on out-of-domain
  unanswerable prompts may vary in shapes V doesn't index. Step 5 results
  on the held-out dataset are the test.

---

## 6. Reproduction

All artefacts referenced above:

| Step | File |
|------|------|
| 0    | `step0_mine/data/mined/qwen_instruct_{kuq,squad}.jsonl`<br>`step0_mine/data/forget/qwen_instruct_{kuq,squad}.jsonl` |
| 1    | `step1_extract_activations/data/activations_qwen_instruct.pt` |
| 2    | `step2_build_subspace/data/subspace_qwen_instruct_r32.pt` |
| 3    | `step3_build_anchors/data/anchors_qwen_instruct.pt` |

Commands used (all `python3`):

```bash
python3 step0_mine/mine.py                  --model qwen_instruct
python3 step1_extract_activations/extract.py --model qwen_instruct
python3 step2_build_subspace/build_subspace.py --model qwen_instruct --rank 32
python3 step3_build_anchors/build_anchors.py --model qwen_instruct
```

Next:

```bash
python3 step4_train/train.py --model qwen_instruct --rank 32 --beta 1.0 --epochs 3 --lr 3e-5
python3 step5_evaluate/evaluate.py --run <run_name>
```
