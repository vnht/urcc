# Step 7 — Baselines

<!-- Plan: not yet implemented -->

Comparative baselines for Overcommitment Unlearning (OU). Every baseline uses
the same training data, test data, judge, and evaluation metrics as OU so that
results are directly comparable.

---

## Shared Infrastructure

### Training data (identical to OU, Section 4.1)

| Pool | Source | Role | # examples |
|------|--------|------|------------|
| Forget — KUQ | `step0_mine/data/forget/<model>_kuq.jsonl` | Mined COMMIT completions on no-context unanswerable questions | 1,000 |
| Forget — SQuAD | `step0_mine/data/forget/<model>_squad.jsonl` | Mined COMMIT completions on context-grounded unanswerable questions | 1,000 |
| Retain — KUQ answerable | `step0_mine/data/sampled/kuq_answerable.jsonl` | Gold answers on answerable KUQ questions | 1,000 |
| Retain — SQuAD answerable | `step0_mine/data/sampled/squad_answerable.jsonl` | Gold answers on answerable SQuAD questions | 1,000 |
| Retain — UltraChat | `step0_mine/data/sampled/ultrachat.jsonl` | General instruction-following | 1,000 |

The forget pool is **model-specific**: each baseline is trained separately per
backbone using that backbone's own mined completions, exactly as OU is.

### Evaluation (identical to OU, Section 4.2)

All baselines are evaluated by reusing `step5_evaluate/evaluate.py` with
`--run-dir` pointing to the baseline's adapter checkpoint, so the same judge
(`gpt-oss-120b` via Cerebras), the same prompts, and the same metrics apply.

**In-distribution test sets:** KUQ, SQuAD (unanswerable split)

**Out-of-distribution test sets:**
- `SelfAware` (no-context, maps to KUQ template)
- `FaithEval` (context-grounded, maps to SQuAD template)
- `NoMIRACL` (context-grounded, maps to SQuAD template)

**Metrics:** TCR, FCR, TAR, FAR, decision accuracy, UltraChat perplexity ratio.

Results land in `step5_evaluate/data/results/<method>_<model>/` following the
existing convention.

### Folder layout per baseline

```
step7_baselines/
  <method>/
    train.py          # training script (Group 1 & 2) or run.py (Group 3)
    README.md         # paper reference, key hyperparameters, run commands
    data/
      runs/           # saved adapter checkpoints (LoRA or full delta)
```

Adapter checkpoints must be loadable by `step5_evaluate/evaluate.py` — i.e.
they must be PEFT LoRA dirs with an `adapter_config.json`. Baselines that
cannot produce a LoRA checkpoint (inference-time methods) receive a thin
adapter-like wrapper that applies the inference-time intervention inside the
forward pass, or they output results directly into
`step5_evaluate/data/results/<method>_<model>/`.

### Shared hyperparameters (match OU where possible)

- LoRA rank 16, α 32, dropout 0.05
- AdamW, lr 3e-5, cosine decay, 3% warmup
- 3 epochs, batch size 4 forget + 4 retain, gradient accumulation ×4
- Early stopping on smoothed loss (window 5)

---

## Group 1 — Unlearning Baselines

These methods treat the mined COMMIT instances as the **forget set** — exactly
the same data OU uses for its forget loss — but apply different forgetting
objectives.

---

### 1a. Gradient Ascent (GA)

**Folder:** `step7_baselines/gradient_ascent/`

**Paper:** Maini et al., "TOFU: A Task of Fictitious Unlearning for LLMs",
NeurIPS 2024. ([PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/be52acf6bccf4a8c0a90fe2f5cfcead3-Paper-Conference.pdf))

**Method:** Maximise cross-entropy on forget examples (ascend the loss),
simultaneously minimise cross-entropy on retain examples to prevent
catastrophic forgetting. This is the simplest unlearning baseline and serves
as the floor comparison.

**Loss:**
```
L_GA = -L_CE(model, D_forget) + λ · L_CE(model, D_retain)
```

**Adaptation to this project:**
- `D_forget`: mined COMMIT completions on unanswerable prompts (same as OU)
- `D_retain`: KUQ answerable + SQuAD answerable + UltraChat (same as OU)
- The forget loss pushes the model away from its own overcommitted completions
- The retain loss preserves answerable and general behaviour
- No subspace projection — gradient ascent operates in the full weight space

**Key hyperparameters:** λ = 1.0 (sweep: 0.5, 1.0, 2.0). Small learning rate
(1e-5) often needed to prevent collapse — sweep with same AdamW/cosine setup.

**Implementation notes:**
- Standard CE loss with sign flipped on the forget batch
- Watch for instability: the gradient ascent direction can cause the model to
  degenerate. Clip gradients (norm 1.0) and monitor both forget-loss magnitude
  and retain PPL at each checkpoint.
- Export best checkpoint by retain-PPL (not by total loss, since forget loss
  descending is not meaningful as a stopping criterion)

---

### 1b. NPO — Negative Preference Optimization

**Folder:** `step7_baselines/npo/`

**Paper / repo:** Lin et al., "Negative Preference Optimization: How to Make
LLMs Forget" ([GitHub](https://github.com/licong-lin/negative-preference-optimization))

**Method:** DPO-style loss that only needs negative (dispreferred) examples.
Unlike DPO which requires (chosen, rejected) pairs, NPO treats the forget
completions as "rejected" and optimises the model away from them relative to a
frozen reference, without needing explicit positive pairs.

**Loss (NPO forget term):**
```
L_NPO = -log σ( -β · log(π_θ(y_oc | x) / π_ref(y_oc | x)) )
```
Combined with a standard SFT retain loss on `D_retain`.

**Adaptation to this project:**
- Rejected responses `y_oc`: the mined overcommitted completions
- No explicit "chosen" responses needed — NPO only requires the negative side
- Reference model `π_ref`: the frozen base model (same as OU's frozen base)
- β = 0.1 (NPO default); sweep 0.05, 0.1, 0.2

**Implementation notes:**
- Load the frozen base as the reference and the trainable LoRA model as π_θ
- Compute log-probs of `y_oc` under both; take the NPO gradient on forget
- Add λ · SFT loss on the retain set (same retain data as OU, λ = 1.0)
- No subspace projection — operates directly on token probabilities

---

### 1c. AdaptiveRMU — Adaptive Random Mismatch Unlearning

**Folder:** `step7_baselines/adaptive_rmu/`

**Paper / repo:** RebelsNLU, "LLM Unlearning via Representation Misdirection"
([GitHub](https://github.com/RebelsNLU-jaist/llm-unlearning))

**Method:** The original RMU (Representation Misdirection Unlearning) steers
hidden states for forget examples toward random unit vectors, making the
model's internal representations for those inputs incoherent. AdaptiveRMU
adds a retain regulariser that preserves representations on the retain set.

**Loss:**
```
L_RMU  = ‖ h_l(x_oc, y_oc) − c · u ‖²        # steer forget toward random unit u
L_retain = ‖ h_l(x_r, y_r)_adapted − h_l(x_r, y_r)_frozen ‖²   # anchor retain
L = L_RMU + λ · L_retain
```
where `u` is a fixed random unit vector (sampled once at init) and `c` is a
scaling coefficient matched to the mean forget-representation norm.

**Adaptation to this project:**
- `h_l` is the mean hidden state over the same K=8 token window used in OU
  (last layer in the trained layer range, e.g. layer 31 for Qwen)
- `u`: one random unit vector per domain (KUQ, SQuAD) to avoid a common
  misdirection target that could be learned away; alternatively use a single u
- The random direction is the key difference from OU's abstention-pole target

**Implementation notes:**
- Use the same layer extraction logic as `step1_extract_activations/extract.py`
- λ = 1.0; c = mean ‖h_forget‖ computed on a small warm-up batch
- This is a representation-space baseline closest to OU but without the
  eigenproblem subspace and without a meaningful abstention target

---

### 1d. LUNAR — LLM Unlearning with Activation Redirection

**Folder:** `step7_baselines/lunar/`

**Paper / repo:** Facebook Research, LUNAR
([GitHub](https://github.com/facebookresearch/LUNAR))

**Method:** Steers forget-example representations toward refusal/abstention
representations extracted from the model's own refusal completions. Differs
from OU in that it redirects to a simple refusal centroid rather than
estimating a low-rank overcommitment subspace.

**Loss:**
```
L_LUNAR = ‖ h_l(x_oc, y_oc)_adapted − μ_refusal ‖²
L_retain = ‖ h_l(x_r, y_r)_adapted − h_l(x_r, y_r)_frozen ‖²
L = L_LUNAR + λ · L_retain
```
where `μ_refusal` is the mean hidden state of abstention-style completions
(equivalent to OU's `μ⁻` anchor poles).

**Adaptation to this project:**
- `μ_refusal` = `μ⁻(kuq)` and `μ⁻(squad)` from `step3_build_anchors`
  (i.e. the same anchors already computed for OU, just used without the
  subspace projection matrix V)
- Full representation space — no eigenproblem — so the loss operates on the
  raw D-dimensional hidden state instead of the projected z = Vᵀh
- This directly isolates the contribution of the subspace estimation step in OU

**Implementation notes:**
- Load `step3_build_anchors/data/anchors_<model>.pt` and extract `mu_neg_kuq`
  and `mu_neg_squad`; use the same domain-routing as OU (KUQ rows → kuq pole,
  SQuAD rows → squad pole)
- The retain loss is identical to OU's retain loss but in full space (no Vᵀ)
- This baseline is the ablation that isolates whether OU's subspace matters

---

## Group 2 — Training for Abstention

These methods fine-tune the model to produce abstention outputs on the
unanswerable split. They treat the problem as supervised abstention rather
than unlearning.

---

### 2a. R-Tuning

**Folder:** `step7_baselines/r_tuning/`

**Paper / repo:** Zhang et al., "R-Tuning: Instructing Large Language Models
to Say 'I Don't Know'" ([GitHub](https://github.com/shizhediao/R-Tuning))

**Method:** Constructs a supervised fine-tuning dataset where unanswerable
prompts are paired with "I don't know" (IDK) responses. The model is then
fine-tuned with standard SFT on the mixed dataset (IDK responses on
unanswerable, gold responses on answerable, UltraChat for utility).

**Training data construction:**
- Forget → SFT pairs: `(prompt_unanswerable, abstention_response)` — the
  abstention response is the same template used in OU's abstention reference
  completions (e.g. "The provided information is insufficient to answer this
  question.")
- Retain answerable pairs: `(prompt_answerable, gold_answer)` — same KUQ/SQuAD
  answerable pairs used in OU
- Retain general pairs: `(prompt_chat, response_chat)` — same UltraChat pairs

**Loss:** Standard next-token cross-entropy SFT on the full mixed dataset.

**Adaptation to this project:**
- Use the same unanswerable prompts from the forget pool (same 2,000 examples
  as OU), but pair them with abstention references rather than the mined COMMIT
  completions
- R-Tuning's original formulation filters to only those questions the model
  *gets wrong* (analogous to OU's mined-COMMIT requirement); here we already
  have that filter from step 0 — so the forget pool is exactly R-Tuning's
  "questions the model can't answer correctly"
- λ balance: mix forget:retain at 1:1.5 ratio by upsampling (matching OU's
  2,000 forget / 3,000 retain balance)

**Implementation notes:**
- No subspace, no anchors, no preference model — purely SFT
- Use the same model prompt templates (KUQ template, SQuAD template) so the
  model sees the same input format as OU
- This is the SFT abstention training baseline

---

### 2b. TruthRL

**Folder:** `step7_baselines/truth_rl/`

**Paper / repo:** Facebook Research, "TruthRL"
([GitHub](https://github.com/facebookresearch/TruthRL))

**Method:** Reinforcement learning from a truth-reward signal. The reward
model assigns +1 for a correct ABSTAIN on unanswerable inputs and +1 for a
correct COMMIT on answerable inputs. PPO or GRPO training optimises the policy
toward this reward.

**Reward function (adapted):**
```
r(x, y) = +1   if x is unanswerable and judge(y) == ABSTAIN
r(x, y) = +1   if x is answerable   and judge(y) == COMMIT
r(x, y) =  0   otherwise
```
The judge is the same Cerebras `gpt-oss-120b` used in OU and step5_evaluate,
ensuring reward labels are consistent with evaluation labels.

**Adaptation to this project:**
- Training prompts: same unanswerable (forget) + answerable (retain) pools,
  with UltraChat prompts added for general-utility reward (judge: any coherent
  response earns +1)
- Online rollout: generate completions from the current policy, judge each,
  update with PPO (or GRPO for simplicity)
- Use `trl` library (already a dependency) for the RL loop
- KL penalty coefficient β_KL = 0.05 to prevent policy collapse

**Implementation notes:**
- TruthRL is the most expensive baseline — each training step requires one
  generation + one judge call per example
- Approximate with GRPO (group relative policy optimisation) if PPO is too
  slow: sample k=4 completions per prompt, score each, use group-normalised
  rewards
- The reward signal is the same judge already used for mining in step 0, so
  no new infrastructure is needed beyond connecting judge.py into the RL loop

---

## Group 3 — Inference-Time Baselines

These methods leave the model weights unchanged and intervene at inference
time. They do not produce LoRA adapters. Evaluation is run by a standalone
`run.py` that wraps `step5_evaluate/evaluate.py`'s generation + judge loop,
writing results to `step5_evaluate/data/results/<method>_<model>/` in the same
JSON format.

---

### 3a. Multi-LLM Collaboration (AbstainQA — Cooperate)

**Folder:** `step7_baselines/multi_llm_collab/`

**Paper / repo:** Feng et al., "Don't Hallucinate, Abstain: Identifying LLM
Knowledge Gaps via Multi-LLM Collaboration", ACL 2024
([GitHub](https://github.com/BunsenFeng/AbstainQA))

**Method:** The primary model proposes an answer; a set of feedback LLMs
evaluate whether the proposed answer is consistent with their own knowledge.
If the feedbacks disagree with the proposal (or flag low confidence), the
system abstains. No training required.

**Adapted pipeline for this project:**
1. **Propose:** Generate a greedy completion from the primary model (same
   generation call as step5_evaluate)
2. **Judge externally:** Send the (prompt, proposed_answer) to the Cerebras
   `gpt-oss-120b` judge asking "Does this answer the question reliably, or
   should the model abstain?" — this repurposes the existing judge.py
   infrastructure as the "area-chair" LLM
3. **Decide:** If the judge signals ABSTAIN, replace the completion with the
   standard abstention template; otherwise keep the original completion
4. **Evaluate:** Feed the final completion through the standard COMMIT/ABSTAIN
   evaluation, so the external judge's decision is the only intervention

**Adaptation notes:**
- This is a two-call approach (generate + judge-as-advisor) rather than a
  full multi-LLM setup; it is a faithful lightweight adaptation given the
  available infrastructure
- A more faithful multi-model version would use two different model families
  (e.g. query Qwen and Ministral on the same prompt and abstain if they
  disagree) — document both variants and implement the lightweight one
- No training data required; the test-time decision threshold is the judge

**Expected output:** `step5_evaluate/data/results/multi_llm_collab_<model>/`
using the same JSON schema as other results.

---

### 3b. AbstentionReasoning (Probe-Based Intervention)

**Folder:** `step7_baselines/abstention_reasoning/`

**Paper / repo:** NJU WebSoft, "Answering the Unanswerable Is to Err
Knowingly", AAAI 2026
([GitHub](https://github.com/nju-websoft/AbstentionReasoning))

**Method:** Trains a lightweight linear probe on hidden states at a specific
layer to predict COMMIT vs. ABSTAIN. At inference, the probe fires when it
detects an incipient overcommitment; an intervention prompt ("I cannot answer
this question") is injected, and generation continues from that point.

**Two-stage pipeline:**

**Stage 1 — Train probe (`train_probe.py`):**
- Extract hidden states at the K=8 token window from `step1_extract_activations`
  (already computed for OU) — no re-extraction needed
- Train a logistic regression (or thin 2-layer MLP) on `h_A` (overcommit) vs
  `h_B` (abstention) per selected layer
- Select the probe layer by cross-validation; expected range: last 25% of
  layers (same range OU uses for subspace estimation)
- Training data: same activation extractions from step 1, so the probe is
  trained on the same examples as OU

**Stage 2 — Inference intervention (`run.py`):**
- At generation time, extract the hidden state at the probe layer after the
  prompt is processed (before the first answer token)
- If probe(h) > threshold, insert the abstention string and return it without
  further generation
- Otherwise, let generation proceed normally
- Threshold is calibrated on the validation split of KUQ/SQuAD (same heldout
  split used for early stopping in OU)

**Adaptation notes:**
- The original AbstentionReasoning operates on reasoning models with explicit
  `<think>` trajectories; for non-reasoning models (Qwen dense, Ministral) the
  intervention point is simply the position after the full prompt
- For GPT-OSS-20B (reasoning model), the intervention can be placed after the
  first reasoning step, more faithfully matching the paper's design
- `train_probe.py` is model-specific (one probe per model per layer); reuse
  `_common.py`'s forward-pass utilities for hidden-state extraction

**Expected output:** `step5_evaluate/data/results/abstention_reasoning_<model>/`

---

## Evaluation Protocol Summary

All seven baselines are evaluated on the same five datasets and the same six
metrics as OU:

| Dataset | Split | Condition | Metric |
|---------|-------|-----------|--------|
| KUQ | unanswerable heldout | in-distribution (no-context) | FCR, FAR, acc |
| SQuAD | unanswerable heldout | in-distribution (context) | FCR, FAR, acc |
| SelfAware | unanswerable | OOD (no-context) | FCR, FAR, acc |
| FaithEval | unanswerable | OOD (context) | FCR, FAR, acc |
| NoMIRACL | unanswerable | OOD (context, retrieval) | FCR, FAR, acc |
| UltraChat | — | utility preservation | PPL ratio |

**TCR** = true commitment rate (answerable → COMMIT); **FCR** = false
commitment rate (unanswerable → COMMIT, should be low); **TAR** = true
abstention rate; **FAR** = false abstention rate (answerable → ABSTAIN,
should be low).

Run sequence per baseline:

```bash
# Group 1 & 2: train, then evaluate with the standard script
python3 step7_baselines/<method>/train.py --model qwen_instruct

python3 step5_evaluate/evaluate.py \
    --run-dir step7_baselines/<method>/data/runs/<run_name> \
    --datasets kuq squad selfaware faitheval nomiracl \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct

# Group 3: run directly (no training)
python3 step7_baselines/<method>/run.py --model qwen_instruct \
    --datasets kuq squad selfaware faitheval nomiracl
```

---

## Summary Table

| # | Method | Group | Training required | Core mechanic | Key difference from OU |
|---|--------|-------|-------------------|---------------|----------------------|
| 1a | Gradient Ascent | Unlearning | Yes (LoRA) | Ascend CE loss on forget set | No subspace; no abstention target |
| 1b | NPO | Unlearning | Yes (LoRA) | DPO-style negative preference on forget | No subspace; prob-space not rep-space |
| 1c | AdaptiveRMU | Unlearning | Yes (LoRA) | Steer forget reps toward random unit | Random target vs. abstention pole |
| 1d | LUNAR | Unlearning | Yes (LoRA) | Steer forget reps toward abstention centroid | No subspace projection (full space) |
| 2a | R-Tuning | SFT abstention | Yes (LoRA) | SFT on (unanswerable → IDK) pairs | No unlearning; direct supervision |
| 2b | TruthRL | RL abstention | Yes (LoRA) | RL with COMMIT/ABSTAIN reward signal | Online reward signal; no forget pool |
| 3a | Multi-LLM Collab | Inference-time | No | External judge advises abstention | No weight change; judge-in-the-loop |
| 3b | AbstentionReasoning | Inference-time | Probe only | Linear probe fires → inject abstention | No LoRA; lightweight probe |
