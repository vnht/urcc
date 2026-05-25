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
| Subspace rank `r`    | 32 (per-domain: `V_kuq`, `V_squad`) |
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
`(Σ_OC(d) − Σ_LC(d)) v = γ Σ_E v` is built (per-domain) to amplify
exactly that residual.

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

## 3. Step 2 — per-domain discriminative subspaces `V(d)`

`step2_build_subspace/data/subspace_qwen_instruct_r32.pt`,
`V_per`: `{"kuq": [8, 4096, 32], "squad": [8, 4096, 32]}`.

For each domain `d ∈ {kuq, squad}` and each layer, solves the generalized
eigenproblem on **domain-restricted contrasts** while keeping the
general-utility covariance shared:

```
(Σ_OC(d) − Σ_LC(d)) v = γ (Σ_E + ridge·I) v       picking top r=32 v's
```

A legacy grand `V` is also saved (built from KUQ+SQuAD mixed contrasts) for
backward-compat / ablations, but training uses the per-domain pair.

### 3.1 Per-domain spectra

| Layer | γ_1(kuq) | γ_1(sq) | OC_proj(kuq) | OC_proj(sq) | LC_proj(kuq) | LC_proj(sq) | OC/LC(kuq) | OC/LC(sq) |
|------:|---------:|--------:|-------------:|------------:|-------------:|------------:|-----------:|----------:|
| 24    | 12.42    | 11.32   | 122.34       | 65.50       | 8.92         | 12.25       | **13.72**  | 5.35      |
| 25    | 12.07    | 11.23   | 118.30       | 62.09       | 8.64         | 11.38       | **13.69**  | 5.45      |
| 26    | 11.76    | 10.68   | 116.03       | 59.91       | 8.81         | 11.11       | **13.17**  | 5.39      |
| 27    | 10.42    |  9.81   | 104.55       | 58.63       | 8.03         | 11.02       | **13.01**  | 5.32      |
| 28    | 10.01    |  9.56   | 102.51       | 57.94       | 7.96         | 10.68       | **12.88**  | 5.43      |
| 29    |  9.28    |  9.45   |  98.56       | 56.76       | 7.97         | 10.53       | **12.37**  | 5.39      |
| 30    |  9.49    |  9.23   |  97.57       | 56.98       | 8.07         | 10.60       | **12.09**  | 5.38      |
| 31    |  6.20    |  7.66   |  68.95       | 46.52       | 6.50         | 10.27       | 10.61      | 4.53      |

`E_proj ≈ 31.92` everywhere in both subspaces by construction (V is whitened
against `Σ_E + ridge·I`, so `tr(V⊤ Σ_E V) ≈ r = 32`). Wall-clock: 5 s.

### 3.2 Subspace overlap diagnostic

Principal angles between `V_kuq` and `V_squad` per layer (cosines of the 32
principal angles between the two rank-32 subspaces):

| Layer | cos_max | cos_min | #cos > 0.9 | #cos < 0.3 |
|------:|--------:|--------:|-----------:|-----------:|
| 24    | 0.701   | 0.006   | **0**      | **15**     |
| 25    | 0.703   | 0.005   | 0          | 15         |
| 26    | 0.686   | 0.011   | 0          | 16         |
| 27    | 0.686   | 0.020   | 0          | 16         |
| 28    | 0.692   | 0.013   | 0          | 15         |
| 29    | 0.704   | 0.015   | 0          | 14         |
| 30    | 0.683   | 0.001   | 0          | 14         |
| 31    | 0.679   | 0.002   | 0          | 15         |

### Reading

* **`OC/LC = 13.7×` for KUQ, `5.4×` for SQuAD.** KUQ has a much sharper
  decision direction than SQuAD: the "commit vs. abstain" contrast is
  cleaner without context, while SQuAD's contrast is partly entangled
  with the in-context evidence-vs-memory conflict (the model's training
  pulls toward answering even when context says otherwise). This is
  intrinsic to the two tasks, not a deficiency of the method.
* **Per-domain V is much more selective than the legacy grand V.** Grand V
  achieves `OC/LC = 4.3×` (mixed); KUQ alone is **3.2× more selective**
  inside its own subspace (13.7 vs 4.3) and SQuAD is **1.3× more
  selective** (5.4 vs 4.3). The shared V was a compromise that under-served
  both domains, KUQ more so because its stronger contrast was diluted.
* **`V_kuq` and `V_squad` share zero highly-aligned dimensions** (cos > 0.9)
  at every layer, and **15–16 of 32 are near-orthogonal** (cos < 0.3). The
  top shared direction has cosine ≈ 0.70 (a ~45° angle). The two
  subspaces are genuinely different bases, not minor perturbations of
  each other.
* **Roughly half of each subspace is unique to its domain.** The shared V
  was forcing both domains through a basis that missed ~half of each
  domain's decision-relevant directions. This is the structural reason
  per-domain V is needed.
* **Smooth spectra.** `γ_1 / γ_r=32 ≈ 8 – 14` per domain. No abrupt knee;
  rank=32 is reasonable for both.
* **Layer pattern.** L24–L26 lead in selectivity for both domains; L31
  trails. Same "early late layers carry the behavioural geometry, very
  last layer is logit-space" pattern.

---

## 4. Step 3 — pole anchors μ⁻(d), μ⁺(d)   [per answerability domain]

`step3_build_anchors/data/anchors_qwen_instruct.pt`. Each pole is **per
answerability domain** d ∈ {kuq, squad}, shape `[8, 4096]` per domain.
The poles are points in 4096-D activation space and do not depend on V —
V is just the projection used by the loss in step 4.

```
μ_l⁻(d) = mean over rows of  h_B  whose dataset == d                (per-domain templated abstention on D_F[d] prompts)
μ_l⁺(d) = mean over rows of  h_C  whose dataset == d                (gold answer on D_R_A[d] prompts)
```

`n_minus_per = {kuq: 500, squad: 500}`, `n_plus_per = {kuq: 500, squad: 500}`. Wall-clock: < 1 s.

`μ⁻(d)` is the forget target in step 4. **`μ⁺(d)` is no longer used in
training** — it is kept in the anchors bundle as a geometric diagnostic
confirming that `V(d)` separates the legit-commit cluster from the
legit-abstain cluster. The retain side uses a per-example, per-token
frozen-base reference instead (see §5).

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
  anchor — which is one reason the forget side uses a pole target while
  the retain side uses per-example frozen-base references.

---

## 5. Synthesis — what the geometry says about the method

The four things UOC's loss needs are all empirically present for this
model.

| Requirement | Observed | Verdict |
|-------------|----------|---------|
| A low-rank discriminative direction *per domain* where over-commit dominates legit-commit | `OC/LC = 13.7×` in V_kuq, `5.4×` in V_squad; every `γ_k > 0` for `k ≤ 32` | ✓ strong |
| The two domains' decision directions are genuinely different | 0/32 highly-aligned dims, 14–16/32 near-orthogonal between V_kuq and V_squad | ✓ strong |
| A target inside each direction to pull A toward          | per-domain `μ⁻(d)`; KUQ and SQuAD μ⁻ poles `52–86` units apart in full hidden space | ✓ strong |
| Per-example preservation reference for retain (no cluster-mean drift) | frozen-base activation `h_l(x, y; θ_frozen)` for every retain example, per-token | ✓ strong |
| Headroom so the pull doesn't damage E (UltraChat)        | `OC_proj / E_proj ≈ 1.5–4×` per domain (V is whitened against Σ_E so E_proj ≡ 32) | ✓ moderate (E is the watch) |
| Equal optimisation pressure across domains              | per-domain init scale `s(kuq)=104`, `s(squad)=58`; loss divided by `s(d)` so both domains start at `L_forget ≈ 1.0` | ✓ explicit |

### Consequence for the loss

```
L = L_forget + λ · L_retain

L_forget   = mean_x∈D_F      ⟨ ‖V_l(d_x)⊤ (h_A(x) − μ⁻(d_x)         )‖² ⟩_{l, t∈T(x)} / s(d_x)
L_retain^C = mean_x∈D_R_A    ⟨ ‖V_l(d_x)⊤ (h_C(x) − h_C^frozen(x)   )‖² ⟩_{l, t∈T(x)} / s(d_x)
L_retain^E = mean_x∈D_R_G    ½ Σ_d  ⟨ ‖V_l(d)⊤ (h_E(x) − h_E^frozen(x))‖² ⟩_{l, t∈T(x)} / s(d)

L_retain   = ½ ( L_retain^C + L_retain^E )                  d ∈ {kuq, squad}, d_x = dataset(x)
s(d)       = mean_l OC_proj(V_l(d))                         (per-domain init scale)
```

For this model: `s(kuq) ≈ 104`, `s(squad) ≈ 58`. The asymmetry directly
reflects KUQ's stronger intrinsic discriminative signal (`OC/LC = 13.7×`
vs SQuAD's `5.4×`). Without per-domain normalisation, KUQ would
contribute ~1.8× the forget pressure of SQuAD per batch — so SQuAD
would be under-budgeted purely because its underlying geometry has less
variance. Dividing by `s(d)` equalises this so both domains start at
`L_forget(d) ≈ 1.0` and contribute equal optimisation pressure.

Initial-step expectations:

* **`L_forget`.** Each KUQ row is projected through `V_kuq` and pulled toward
  `μ⁻_kuq`, each SQuAD row through `V_squad` toward `μ⁻_squad`. With per-domain
  V the projection axis is itself domain-specialised (15/32 dims unique to
  each domain), so initial loss is a *within-domain* distance along a
  *within-domain* direction. Initial L_forget magnitude depends on
  `OC_proj`: ~110 KUQ / ~60 SQuAD per layer.
* **`L_retain^C`.** Each legit-commit example is anchored to its own frozen-
  base activation, projected through its domain's `V(d)`. This is a
  per-example, per-token target — sharp counter-gradient against any drift
  on a specific retain input. Initial loss = 0 by construction (trainable
  forward = frozen forward at step 0).
* **`L_retain^E`.** UltraChat has no domain, so preservation is averaged
  across both `V_kuq` and `V_squad`: any direction either domain's forget
  pull acts on must be preserved on retain-general inputs. Initial loss = 0
  by construction.

### Predicted layer roles

* **L24–L26.** Cleanest selectivity for both domains (`OC/LC ≈ 13×` KUQ, `5×`
  SQuAD). Forget gradient comes with the least collateral on legit-commit
  drift.
* **L28–L30.** Largest absolute pole gaps for both domains
  (`‖μ⁻_kuq − μ⁻_squad‖ ≈ 70–86`). Forget loss has the most signal to drop
  here.
* **L31.** Weaker on every metric; useful as a sanity-anchor but not the
  workhorse layer.

### Predicted step 5 metrics (qualitative)

* **FCR (false-commit rate, unanswerable held-out).** This is what V(d) is
  built to suppress. Substantial drop expected on both domains; SQuAD
  should benefit most relative to the shared-V baseline because V_squad
  finally captures SQuAD-specific decision directions that the shared V
  missed.
* **TCR (true-commit rate, answerable held-out).** Should hold within a
  few points of the base model — per-example frozen-base retain anchors
  provide strong, specific preservation pressure; per-domain V keeps
  forget pull from leaking into legit-commit territory.
* **Empty completions.** Should remain at 0 (a property of the per-example
  frozen-base retain — first-token decisions on every retain example are
  pinned per-token).
* **UltraChat preservation.** Should be near-baseline. Retain^E projects
  through both V(d) and uses frozen-base targets; the LoRA's only path to
  drift is through directions outside both subspaces, which the loss
  doesn't constrain — but those directions also don't carry the
  forget-pull signal, so there's no incentive to drift there.

### Risks the data flags

* **SQuAD's intrinsic decision direction is weaker than KUQ's** (`OC/LC = 5×`
  vs `13×`). This is a property of the task, not the method: in-context
  unanswerability is harder for the model because parametric memory and
  retrieval prompts pull in opposite directions. Per-domain V gives SQuAD
  its own axis, but the axis itself has lower selectivity.
* **Stylistic over-commit ≠ all over-commit.** Σ_OC has lower trace than
  Σ_LC because over-commits are stereotyped. The forget set captures the
  most obvious cases (KUQ + SQuAD). Real-world over-commit on out-of-domain
  unanswerable prompts may vary in shapes neither V indexes. Step 5
  results on the held-out dataset are the test.
* **Per-domain V doubles the geometry to specify but not the model.** One
  shared LoRA adapter still has to satisfy two projection objectives
  simultaneously. This is fine when the two directions don't conflict
  (15/32 dims orthogonal — they don't compete for the adapter's capacity)
  but worth verifying that L_forget on both domains drops together during
  training.

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
