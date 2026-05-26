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
| Subspace rank `r`    | 32 (shared `V`) |
| Subspace ridge       | `1e-3` |
| Retain basis rank    | 512 |
| Forget templates     | `KUQ_PROMPT_TEMPLATE`, `SQUAD_PROMPT_TEMPLATE` |
| Templated abstention (KUQ)   | `"I do not have enough information to answer that."` |
| Templated abstention (SQuAD) | `"The provided context does not contain information about that."` |

Datasets:

| Pool | Source | Size | Role |
|------|--------|------|------|
| `D_F` (forget)         | KUQ + SQuAD unanswerable, COMMIT-only | 500 + 500 = **1000** | Category A |
| `D_R_A` (retain QA)    | KUQ + SQuAD answerable + gold         | 500 + 500 = **1000** | Categories C, D (μ⁺ kept as diagnostic only) |
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

| Set | Description                                  | `N`  | Shape           | Mean per-row `‖h‖` | Std |
|-----|----------------------------------------------|-----:|-----------------|-------------------:|----:|
| A   | Over-commit                                  | 1 000 | (1000, 8, 4096) | 116.17             | 27.0 |
| B   | Legit abstention (per-domain templated)      | 1 000 | (1000, 8, 4096) | 112.32             | 24.5 |
| C   | Legit commitment (gold answer)               | 1 000 | (1000, 8, 4096) | 112.94             | 28.8 |
| D   | Over-abstention (per-domain templated on `D_R_A`) | 1 000 | (1000, 8, 4096) | 107.60             | 24.9 |
| E   | General utility (UltraChat)                  | 1 000 | (1000, 8, 4096) | 105.42             | 23.8 |

Wall-clock: 40 m 30 s. Sets B and D each use the row's own dataset
template (`y_⊥(kuq)` or `y_⊥(squad)`); sets A, C, E are domain-agnostic
since they use the model's own / gold / corpus completions.

### 2.1 Contrast geometry

The two cross-set contrasts that drive the eigenproblem in step 2:

```
c_OC(x) = h_A(x) − h_B(x)        per-row over-commit contrast
c_LC(x) = h_C(x) − h_D(x)        per-row legit-commit contrast
```

| Quantity | Value |
|----------|------:|
| Per-row mean `‖c_OC‖` | 82.8 |
| Per-row mean `‖c_LC‖` | 90.3 |
| `cos(mean c_OC, mean c_LC)` | **+0.805** |

`cos = +0.805` is the central diagnostic. It says **most** of what differs
between A and B is the same direction that differs between C and D — the
"commit-vs-abstain mode". The gap below 1.0 is the residual signal that is
*uniquely about over-committing*. Step 2's eigenproblem
`(Σ_OC − Σ_LC) v = γ Σ_E v` is built to amplify exactly that residual.

### 2.2 Per-layer covariance traces (proxy for eigen-signal)

| Layer | `tr Σ_OC` | `tr Σ_LC` | `tr Σ_E` | OC / LC | OC / E |
|------:|----------:|----------:|---------:|--------:|-------:|
| 24    |  2 733    |  2 784    |  3 410   | 0.98    | 0.80   |
| 25    |  3 175    |  3 305    |  4 003   | 0.96    | 0.79   |
| 26    |  3 593    |  3 897    |  4 514   | 0.92    | 0.80   |
| 27    |  4 464    |  4 966    |  5 339   | 0.90    | 0.84   |
| 28    |  5 465    |  6 147    |  6 697   | 0.89    | 0.82   |
| 29    |  6 463    |  7 487    |  8 004   | 0.86    | 0.81   |
| 30    |  7 844    |  9 412    | 10 040   | 0.83    | 0.78   |
| 31    |  2 913    |  3 752    |  5 518   | 0.78    | 0.53   |

`Σ_OC < Σ_LC` everywhere by 2–22%. Counter-intuitive at first, but
consistent with: over-commits are stylistically stereotyped (a few canonical
confident-but-wrong shapes) while legit-commits span every topic in the
retain pool, so total variance is lower for OC. Trace is the sum of
eigenvalues; what matters for V is whether **specific directions** have
`Σ_OC > Σ_LC`. They do — see step 2 below.

L31 contracts sharply (everything drops ~½×). Standard last-layer
"residual-stream collapses into LM-head logit space" pattern.

---

## 3. Step 2 — discriminative subspace `V`

`step2_build_subspace/data/subspace_qwen_instruct_r32.pt`,
`V` shape `[8, 4096, 32]`.

For each layer, solves the generalized eigenproblem on the cross-domain
contrasts (KUQ + SQuAD pooled) against the general-utility covariance:

```
(Σ_OC − Σ_LC) v = γ (Σ_E + ridge·I) v       picking top r=32 v's
```

`V_l ∈ ℝ^{D × r}` are the top-32 generalised eigenvectors per layer. The
contrasts that go into Σ_OC and Σ_LC are themselves shaped by the
**per-domain** abstention template `y_⊥(d)` used to build sets B and D
in step 1, so domain structure enters V through the data even though V
itself is shared. Domain specialisation in step 4 lives in the per-domain
forget pole `μ⁻(d)`, not in the projection axis.

### 3.1 Per-layer spectrum

| Layer | γ_1   | γ_32  | OC_proj | LC_proj | E_proj | OC/LC |
|------:|------:|------:|--------:|--------:|-------:|------:|
| 24    | 41.02 | 1.04  | 137.25  | 31.88   | 31.91  | 4.30  |
| 25    | 36.90 | 1.01  | 129.50  | 30.50   | 31.92  | 4.25  |
| 26    | 36.06 | 0.98  | 125.88  | 29.19   | 31.92  | 4.31  |
| 27    | 32.41 | 0.91  | 116.18  | 28.64   | 31.93  | 4.06  |
| 28    | 32.43 | 0.88  | 114.25  | 27.50   | 31.93  | 4.16  |
| 29    | 31.26 | 0.85  | 109.81  | 26.03   | 31.93  | 4.22  |
| 30    | 31.78 | 0.85  | 109.60  | 25.88   | 31.93  | 4.24  |
| 31    | 24.48 | 0.58  |  80.92  | 19.56   | 31.92  | 4.14  |

`E_proj ≈ 31.92` ≈ `r = 32` everywhere by construction (V is whitened
against `Σ_E + ridge·I`, so `tr(V⊤ Σ_E V) ≈ r`). Wall-clock: 2 s.

`OC_proj` is a useful diagnostic but is not the divisor used at training.
Step 4 uses the **per-domain `s(d)`** computed in step 3 from the actual
forget target `‖Vᵀ(h_A − μ⁻(d))‖²` (see §4 below).

### Reading

* **`OC_proj / LC_proj ≈ 4.1–4.3×` everywhere.** Strong, layer-stable
  separation. The subspace is built specifically to amplify directions
  along which over-commit varies more than legit-commit (after subtracting
  each side's abstain baseline), and the projection diagnostic confirms
  that on every layer the over-commit cluster spreads ~4× wider in V than
  the legit-commit cluster.
* **`OC_proj / E_proj ≈ 2.5–4.3×`.** Headroom against general utility:
  every direction in V already has `E_proj ≈ 32` by whitening, so the fact
  that `OC_proj` is 80–137 says the over-commit signal lives in a region
  of late-layer space that is not just "anywhere general activations vary".
  This is what the retain loss exploits — moving along V to suppress A is
  not equivalent to randomly perturbing E.
* **Smooth spectrum.** `γ_1 / γ_r=32 ≈ 35–42`. No abrupt knee; rank `r=32`
  is comfortably in the discriminative regime, well clear of the noise
  floor.
* **Layer pattern.** L24–L26 lead in `OC_proj` (≥ 125); L31 trails (~80).
  Standard "early-late layers carry behavioural geometry, last layer
  collapses into logit space" pattern.

---

## 4. Step 3 — pole anchors μ⁻(d), μ⁺(d), and per-domain init scale s(d)

`step3_build_anchors/data/anchors_qwen_instruct.pt`. Each pole is **per
answerability domain** d ∈ {kuq, squad}, shape `[8, 4096]` per domain.
The poles are points in 4096-D activation space and do not depend on V —
V is the projection used by the loss in step 4.

```
μ_l⁻(d) = mean over rows of  h_B  whose dataset == d                (per-domain templated abstention on D_F[d] prompts)
μ_l⁺(d) = mean over rows of  h_C  whose dataset == d                (gold answer on D_R_A[d] prompts)
```

`n_minus_per = {kuq: 500, squad: 500}`, `n_plus_per = {kuq: 500, squad: 500}`. Wall-clock: < 1 s.

`μ⁻(d)` is the forget target in step 4. **`μ⁺(d)` is not used in
training** — it is kept in the anchors bundle as a geometric diagnostic
confirming that `V` separates the legit-commit cluster from the
legit-abstain cluster. The retain side uses supervised CE on response
tokens (see §5).

Step 3 also computes the **per-domain initial L_forget magnitude**

```
s(d) = E_{x ∈ D_F[d]} ⟨ ‖ V_lᵀ (h_A(x) − μ_l⁻(d)) ‖² ⟩_l
```

which is the actual quantity step 4's L_forget computes per example before
averaging. For this model:

| Domain | `s(d)` |
|---|---:|
| KUQ   | **952.95** |
| SQuAD | **381.51** |

KUQ's pole-distance squared norm is **2.5× SQuAD's** — KUQ has a sharper
commit-vs-abstain contrast. With a shared init scale (mean ≈ 667), KUQ
examples would dominate the gradient average. Step 4 divides each forget
example by its own domain's `s(d)` so KUQ and SQuAD enter the optimiser
at L_forget ≈ 1.0 per example — equal per-example forget pressure
regardless of intrinsic contrast magnitude.

### Per-layer pole geometry (per domain)

| Layer | `‖μ⁻_kuq‖` | `‖μ⁻_squad‖` | `‖μ⁻_kuq − μ⁻_squad‖` | `‖μ⁺_kuq‖` | `‖μ⁺_squad‖` | `‖μ⁺_kuq − μ⁺_squad‖` |
|------:|-----------:|-------------:|----------------------:|-----------:|-------------:|----------------------:|
| 24    |    74.52   |    79.20     |        51.90          |    47.13   |    70.96     |        38.83          |
| 25    |    83.43   |    88.73     |        56.01          |    49.65   |    78.28     |        43.71          |
| 26    |   102.01   |    93.88     |        59.24          |    74.19   |    88.03     |        32.78          |
| 27    |   103.97   |   106.18     |        63.16          |    63.44   |    97.06     |        48.94          |
| 28    |   120.08   |   126.67     |        70.35          |    79.17   |   118.32     |        55.62          |
| 29    |   134.02   |   135.63     |        75.82          |    96.40   |   132.69     |        55.96          |
| 30    |   149.51   |   148.12     |        85.65          |   112.71   |   134.23     |        52.45          |
| 31    |   100.39   |    91.15     |        54.03          |    77.70   |    80.33     |        32.30          |

### Reading

* **Domains are 50–86 units apart for μ⁻.** `‖μ⁻_kuq − μ⁻_squad‖` = 52–86
  across layers, on the order of `‖μ⁻_kuq‖` itself. The two domains'
  abstain representations live in genuinely different regions of
  late-layer hidden space, not slight perturbations of each other.
* **The per-domain abstain template makes the gap larger.** With a single
  shared template `"I do not have enough information to answer that."`
  the gap was 30–53 units. Switching SQuAD to its own context-grounded
  template `"The provided context does not contain information about
  that."` (chosen by inspection of the base model's natural SQuAD
  abstentions) increased the gap by ~+60% at every layer. The new μ⁻_squad
  sits in a region the model can actually reach with small LoRA updates.
* **The same separation pattern holds for μ⁺.** `‖μ⁺_kuq − μ⁺_squad‖` =
  32–56. Long-context prompts bias the late-layer representation toward
  extracted-from-context content; KUQ has none. Both poles need to be
  per-domain.
* **`‖μ⁻‖ > ‖μ⁺‖` per domain.** Templated abstention is the same text
  every time so its mean is a sharp representation; gold answers vary in
  topic so averaging suppresses answer-specific content. μ⁻ is the cleaner
  anchor, which is why the forget side uses it as a pole target. The
  retain side does not use any pole — it is supervised CE on response
  tokens, which directly anchors the LM head's output distribution.

---

## 5. Synthesis — what the geometry says about the method

The five things UOC's loss needs are all empirically present for this
model.

| Requirement | Observed | Verdict |
|-------------|----------|---------|
| A low-rank discriminative direction where over-commit dominates legit-commit | `OC_proj / LC_proj ≈ 4.1–4.3×` per layer; every `γ_k > 0` for `k ≤ 32` | ✓ strong |
| Per-domain abstain target inside that direction to pull A toward | `μ⁻(d)` for `d ∈ {kuq, squad}`; KUQ and SQuAD μ⁻ poles `52–86` units apart in full hidden space | ✓ strong |
| The per-domain abstention text matches the model's natural abstention region | KUQ: `"I do not have enough information…"`; SQuAD: `"The provided context does not contain…"`; per-domain template widened μ⁻ separation by ~+60% over a shared template | ✓ explicit |
| Equalised per-example forget pressure across domains | per-domain `s(d)` (KUQ ≈ 953, SQuAD ≈ 382); each forget example divided by its own `s(d)` so KUQ and SQuAD start at L_forget ≈ 1.0 per example | ✓ explicit |
| Direct preservation of LM-head decisions on retain inputs | supervised next-token CE on response tokens for both D_R_A (gold answer) and D_R_G (UltraChat response); prompt tokens masked | ✓ strong |
| Headroom so the forget pull doesn't damage E (UltraChat) | `OC_proj / E_proj ≈ 2.5–4.3×` (V is whitened against Σ_E so `E_proj ≡ 32`); CE on D_R_G also explicitly anchors UltraChat output distributions | ✓ strong |

### Consequence for the loss

```
L = L_forget + λ · L_retain

L_forget = mean_x∈D_F             ⟨ ‖V_l⊤ (h_A(x) − μ⁻(d_x)) ‖² ⟩_{l, t∈T(x)} / s(d_x)
L_retain = mean_x∈D_R_A ∪ D_R_G   − ⟨ log p_{θ+δθ}(y_t | x, y_<t) ⟩_{t ∈ y_resp}

s(kuq) ≈ 952.95     s(squad) ≈ 381.51       (constants from data, not hyperparameters)
d_x ∈ {kuq, squad}                           (forget pole and init scale are per-domain)
y_resp = gold answer (D_R_A)  or  UltraChat response (D_R_G), capped at 128 tokens
```

The two losses operate in different spaces by design. `L_forget` is a
geometric pull along `V` — the right space for *changing* a behaviour
(over-commitment). `L_retain` is a token-distribution constraint — the
right space for *preserving* a behaviour (the LM head's decision on a
retain prompt). One coefficient `λ` controls the trade-off between
change and preservation; nothing else.

Domain structure enters the loss through (a) the per-domain abstention
template that shaped the contrasts going into V in step 2, (b) the
per-domain forget pole `μ⁻(d_x)` selected per example, and (c) the
per-domain init scale `s(d_x)` that equalises per-example forget
pressure across domains.

Initial-step expectations:

* **`L_forget`.** Each forget example is projected through the shared `V`
  and pulled toward its own domain's `μ⁻(d_x)`. After dividing by `s(d_x)`,
  `L_forget ≈ 1.0` at step 0 — the same for KUQ and SQuAD examples,
  regardless of intrinsic contrast magnitude.
* **`L_retain`.** Initial CE = the base model's negative log-likelihood on
  the response tokens. Empirically `~1.5–3` nats per token; gradients
  start small (the LoRA is identity at step 0 so logits match the base).
  The retain loss grows only as the LoRA starts to perturb the logits.

### Predicted layer roles

* **L24–L26.** Highest `OC_proj` (≥ 125) — most forget signal per unit
  weight update; the forget gradient is well-conditioned here.
* **L28–L30.** Largest absolute pole gaps `‖μ⁻_kuq − μ⁻_squad‖ ≈ 70–86`.
  Forget loss has the most domain-specific signal here, so the per-domain
  `μ⁻(d)` switching is most consequential at these layers.
* **L31.** Lower `OC_proj` (~80) and tighter pole gap (~54). Useful as a
  sanity-anchor but not the workhorse layer.

### Predicted step 5 metrics (qualitative)

* **FCR (false-commit rate, unanswerable held-out).** This is what `V` is
  built to suppress. Substantial drop expected on both domains. SQuAD
  benefits from per-domain `s(d)` because each SQuAD forget example now
  contributes equal optimisation pressure to KUQ examples — KUQ no
  longer dominates the gradient average.
* **TCR (true-commit rate, answerable held-out).** Should hold close to
  baseline. CE on gold answer tokens directly preserves
  `p(y_gold | x)` for legit-commit prompts; the LoRA pays a token-level
  price for any drift on a specific answerable input.
* **Empty completions.** Should be 0. CE on response tokens directly
  penalises mass on EOS at answer-start positions, which was the
  mechanism producing empties under geometric-only retain.
* **UltraChat preservation.** Should be at or above baseline.
  CE on UltraChat response tokens explicitly anchors the LM head's
  next-token distribution; combined with V's `Σ_E`-whitening (forget
  pull doesn't push general-utility directions to begin with), the LoRA
  has both an explicit and an implicit reason to leave UltraChat alone.

### Risks the data flags

* **SQuAD's contrast is partly entangled with the in-context
  evidence-vs-memory conflict.** When the gold answer is in the context,
  the model's parametric memory and retrieval prompts can pull in opposite
  directions. The per-domain abstention template `"The provided context
  does not contain information about that."` aligns the abstain pole with
  the model's natural in-context abstention region, but the underlying
  contrast still has ~5× rather than ~14× selectivity in domain-restricted
  diagnostics — i.e., SQuAD is intrinsically harder to unlearn cleanly
  than KUQ.
* **Stylistic over-commit ≠ all over-commit.** Σ_OC has lower trace than
  Σ_LC because over-commits are stereotyped. The forget set captures the
  most obvious cases (KUQ + SQuAD top-500). Real-world over-commit on
  out-of-domain unanswerable prompts may vary in shapes V doesn't index.
  Step 5 results on held-out / OOD evaluations are the test.
* **One V, two pole regions.** A single `V` has to be useful for both
  KUQ-style and SQuAD-style commit-vs-abstain decisions. The cross-domain
  cosine of mean contrasts (+0.805) confirms most of the abstain mode is
  shared, so a single V is reasonable; the per-domain pole `μ⁻(d)` and
  per-domain init scale `s(d)` then pick up the remaining domain-specific
  localisation and pressure equalisation. Per-domain `V(d)` was tried in
  development and did not improve end-to-end metrics beyond this.

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
python3 step3_build_anchors/build_anchors.py --model qwen_instruct --rank 32
```

Next:

```bash
python3 step4_train/train.py --model qwen_instruct --rank 32 --lambda-retain 1 --epochs 3 --lr 3e-5
python3 step5_evaluate/evaluate.py --run-dir step4_train/data/runs/<run_name> \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct
```
