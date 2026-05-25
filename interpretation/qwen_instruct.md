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
| Answer-token window `K` | 8 (transition-shifted: `[p_len − 1, p_len + K − 2]`) |
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

The window is shifted one position earlier than naive answer-token indexing:

```
T(x) = { p_len − 1, p_len, p_len + 1, …, p_len + K − 2 }
```

`p_len − 1` is the prompt-final residual stream — the state that determines
the *first* generated token via the LM head. Including it inside the
averaging window gives the retain loss a direct grip on first-token
generation, which is what intrinsically prevents the LoRA from satisfying
the geometric loss by collapsing first-token logits to a chat-end token
(the failure mode that produced empty completions on SQuAD with the
unshifted window). The remaining `K − 1` positions still cover the body of
the answer (or abstention) so the per-pole geometry is barely perturbed.

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

## 4. Step 3 — pole anchors μ⁻(d), μ⁺(d), forget margin m²(d)   [per answerability domain]

`step3_build_anchors/data/anchors_qwen_instruct.pt`. Each pole is **per
answerability domain** d ∈ {kuq, squad}, shape `[8, 4096]` per domain.
A new per-(layer, domain) constant `m²_l(d)` is also stored — the
V-projected variance of legitimate-abstention examples around their own
pole, used as a hinge threshold by step 4's forget loss.

The shared subspace V from step 2 is unchanged — V captures the
abstain-vs-commit *axis*, the poles localise the *target on that axis*
inside each domain's prompt distribution, and `m²(d)` defines the natural
"done" boundary around each pole.

```
μ_l⁻(d) = mean over rows of  h_B  whose dataset == d                (templated abstention on D_F[d] prompts)
μ_l⁺(d) = mean over rows of  h_C  whose dataset == d                (gold answer on D_R_A[d] prompts)
m²_l(d) = mean over rows of  h_B(d)  of  ||V_l⊤(h_B(d) − μ_l⁻(d))||²   (V-projected abstain spread, layer l, domain d)
```

`n_minus_per = {kuq: 500, squad: 500}`, `n_plus_per = {kuq: 500, squad: 500}`. Wall-clock: < 1 s.

### Per-layer pole geometry (per domain)

| Layer | `‖μ⁻_kuq‖` | `‖μ⁻_squad‖` | `‖μ⁻_kuq − μ⁻_squad‖` | `‖μ⁺_kuq‖` | `‖μ⁺_squad‖` | `‖μ⁺_kuq − μ⁺_squad‖` |
|------:|-----------:|-------------:|----------------------:|-----------:|-------------:|----------------------:|
| 24    |    74.52   |    73.04     |        30.69          |    47.13   |    70.96     |        38.83          |
| 25    |    83.43   |    80.81     |        33.62          |    49.65   |    78.28     |        43.71          |
| 26    |   102.01   |    93.66     |        37.95          |    74.19   |    88.03     |        32.78          |
| 27    |   103.97   |    99.31     |        39.41          |    63.44   |    97.06     |        48.94          |
| 28    |   120.08   |   113.23     |        44.08          |    79.17   |   118.32     |        55.62          |
| 29    |   134.02   |   127.27     |        47.38          |    96.40   |   132.69     |        55.96          |
| 30    |   149.51   |   144.41     |        51.75          |   112.71   |   134.23     |        52.45          |
| 31    |   100.39   |    96.85     |        32.40          |    77.70   |    80.33     |        32.30          |

### Per-layer forget margin `m²_l(d)`  (V-projected abstain spread, used as a hinge by step 4)

| Layer | `m²_kuq` | `m²_squad` | `OC_proj` (init L_forget proxy) | margin / init  KUQ | margin / init  SQuAD |
|------:|---------:|-----------:|--------------------------------:|-------------------:|---------------------:|
| 24    |   13.26  |    3.87    |        112.69                   |        12%         |          3%          |
| 25    |   15.51  |    4.38    |        105.12                   |        15%         |          4%          |
| 26    |   18.34  |    4.82    |         99.71                   |        18%         |          5%          |
| 27    |   21.52  |    6.60    |         93.70                   |        23%         |          7%          |
| 28    |   28.56  |    9.00    |         93.40                   |        31%         |         10%          |
| 29    |   35.91  |   11.41    |         93.02                   |        39%         |         12%          |
| 30    |   46.02  |   15.55    |         92.20                   |        50%         |         17%          |
| 31    |   20.73  |    9.38    |         71.97                   |        29%         |         13%          |

### Reading

* **Domains are 30–53 units apart.** `‖μ⁻_kuq − μ⁻_squad‖` = 32–53 across
  layers, on the order of `‖μ⁻_kuq − μ⁺_kuq‖` itself. The two domains' abstain
  representations live in genuinely different regions of late-layer hidden
  space, not slight perturbations of each other.
* **The same is true for μ⁺.** `‖μ⁺_kuq − μ⁺_squad‖` = 32–54. The "answering
  substantively" mode also looks different with vs. without context — SQuAD
  prompts have a long context that biases the late-layer representation
  toward extracted-from-context content, while KUQ has none.
* **A grand-mean μ would be ~16–27 units from each domain's true pole** —
  on the order of 20–25% of pole magnitude. Per-domain poles eliminate this
  cross-domain mismatch; KUQ examples are pulled toward `μ⁻_kuq`, SQuAD
  examples toward `μ⁻_squad`. The training pull stays within each example's
  natural prompt distribution.
* **`‖μ⁻‖ > ‖μ⁺‖` per domain.** Templated abstention is the same text every
  time so its mean is a sharp representation; gold answers vary in topic so
  averaging suppresses answer-specific content. μ⁻ is the cleaner anchor.
* **The shared V still captures the axis.** Its construction in step 2 used
  the contrasts `(h_A − h_B)` and `(h_C − h_D)`; subtracting templated-
  abstention baselines from each set already removed most of the shared
  domain-specific envelope, so V points toward the abstain-vs-commit
  direction common to both domains. Per-domain poles are the *positions on
  that axis* that differ.

### Reading the forget margin

* **`m²_squad ≈ ⅓ × m²_kuq` at every layer.** SQuAD's natural abstain
  region is ~3× tighter in V than KUQ's. Long-context prompts force the
  late-layer state into a narrow band when the model knows it cannot
  answer; KUQ's open-ended unanswerable prompts allow more spread. The
  hinge therefore stops SQuAD's forget pull much earlier — exactly where
  empirically it was overshooting and producing empty completions.
* **margin / init ≈ 12–50 % (KUQ), 3–17 % (SQuAD).** The hinge sits well
  below the initial forget loss everywhere. Training will get full
  gradient at step 0 and the hinge will progressively kick in only as
  examples actually reach their domain's abstain region.
* **Layer 30 has the largest absolute margin (46 / 16) but also the
  largest absolute pole gap (52 / 52).** The cap there is generous,
  consistent with this layer carrying a lot of behavioural geometry that
  we don't want to over-pull through.
* **Layer 24 has the tightest cap (margin ≈ 12 % / 3 % of init).** The
  earliest of the eight late layers — selective `γ_1 = 27.8` but the
  abstain region itself is small. The hinge will saturate fastest here.

---

## 5. Synthesis — what the geometry says about the method

The three things UOC's loss needs are all empirically present for this
model.

| Requirement | Observed | Verdict |
|-------------|----------|---------|
| A unique low-rank direction where over-commit dominates | `OC/LC = 7×` in `V`, every `γ_k > 0` for `k ≤ 32` | ✓ strong |
| A target point inside that direction to pull A toward    | per-domain `‖V⊤(μ⁻(d) − μ⁺(d))‖` order-of `10–18` per layer; KUQ and SQuAD targets sit `30–50` units apart in full hidden space | ✓ strong |
| A geometric "done" signal that stops over-pulling once an example reaches the abstain region | per-(layer, domain) margin `m²_l(d)` with margin/init = 3–50 % depending on layer & domain | ✓ strong |
| Headroom so the pull doesn't damage C and E              | `OC/LC = 7×`, `OC/E = 3×`; `LC_proj ≈ 14`, `E_proj ≡ 32` | ✓ moderate (E is the watch) |

### Consequence for the loss

```
L = L_forget + λ · L_retain
L_forget        = mean_x∈D_F   relu( ‖V⊤(h_A(x) − μ⁻(d_x))‖² − m²(d_x) )    # d_x = dataset(x)
L_retain[C]     = mean_x∈D_R_A      ‖V⊤(h_C(x) − μ⁺(d_x))‖²
L_retain[E]     = mean_x∈D_R_G      ‖V⊤(h_E(x) − h_E_frozen(x))‖²
```

Initial-step expectations:

* **`L_forget`.** Each KUQ row is pulled toward `μ⁻_kuq`, each SQuAD row
  toward `μ⁻_squad`. `V⊤ h_A(x) − V⊤ μ⁻(d_x)` is now a within-domain
  distance instead of a partly-cross-domain one — initial loss falls
  sooner because the target lives in the example's own prompt
  distribution. The hinge is inactive at step 0 (margin ≪ init L_forget)
  and progressively saturates per-token as activations enter the natural
  abstain region.
* **`L_retain[C]`.** Same routing: KUQ legit-commits anchored to
  `μ⁺_kuq`, SQuAD to `μ⁺_squad`. Expected initial loss is *lower* than
  the grand-mean version because each pole sits at the centre of its own
  domain's commit cluster. **No margin is applied on the retain side** —
  the goal there is exact anchoring, not "close enough".
* **`L_retain[E]`.** Reference-anchored, so initial value is 0 by
  construction. Domain routing does not apply (UltraChat has no
  answerability domain).

### Predicted layer roles

* **L24–L26.** Cleanest selectivity (γ_1 = 21–28). Forget gradient comes
  with the least collateral on legit-commits.
* **L28–L30.** Largest absolute pole gaps for both domains. Forget loss
  has the most signal to drop here.
* **L31.** Weaker on every metric; useful as a sanity-anchor but not the
  workhorse layer.

### Predicted step 5 metrics (qualitative)

* **FCR (false-commit rate, unanswerable held-out).** This is what V is
  built to suppress; substantial drop expected.
* **Empty completions on SQuAD.** Previously the dominant failure mode
  with grand-mean poles and a plain squared `L_forget`. The hinge `m²` is
  expected to remove most of them: by the time activations enter the
  domain's natural abstain region, the forget term is zero and stops
  perturbing the prompt-final residual stream.
* **Decision accuracy (answerable held-out).** Should hold within a few
  points of the base model — `LC_proj` is small in V (only ~14 worth of
  variance budget for the model to disturb), and the hinge does not
  affect retain.
* **UltraChat preservation.** The hardest call from geometry alone.
  `OC/E = 3×` is a modest margin; if UltraChat outputs visibly degrade
  before FCR drops, λ is too high.

### Risks the data flags

* **E-set margin is the bottleneck.** OC/E ≈ 3× vs OC/LC ≈ 7×. Don't push
  λ beyond ~7 on the first run; UltraChat will be the first thing to
  show damage if anything does.
* **Stylistic over-commit ≠ all over-commit.** Σ_OC has lower trace than
  Σ_LC because over-commits are stereotyped. The forget set captures the
  most obvious cases (KUQ + SQuAD). Real-world over-commit on out-of-domain
  unanswerable prompts may vary in shapes V doesn't index. Step 5 results
  on the held-out dataset are the test.
* **Margin generalisation is an empirical claim.** `m²(d)` is calibrated
  on `h_B(d)` — templated abstention activations. The hinge therefore
  encodes "indistinguishable from real abstain in V". If A activations
  reach a *different* corner of V that happens to be ≤ m² from μ⁻ but
  isn't actually abstain-shaped, the hinge will mistakenly deactivate.
  Step 5 generations are the check.

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
python3 step4_train/train.py --model qwen_instruct --rank 32 --lambda-retain 1 --epochs 3 --lr 3e-5
python3 step5_evaluate/evaluate.py --run-dir step4_train/data/runs/<run_name> \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct
```
