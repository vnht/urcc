# Gradient Ascent (GA) Baseline

**Paper:** Yao, Xu, Liu — *Large Language Model Unlearning* (NeurIPS 2024)
- PDF: <https://proceedings.neurips.cc/paper_files/paper/2024/file/be52acf6bccf4a8c0a90fe2f5cfcead3-Paper-Conference.pdf>
- arXiv: <https://arxiv.org/abs/2310.10683>

---

## Algorithm

GA unlearning uses three losses (paper's Equation 2):

```
θ ← θ  −  ε₁ · ∇ L_fgt  −  ε₂ · ∇ L_rdn  −  ε₃ · ∇ L_nor
```

| Loss | Formula | Effect |
|------|---------|--------|
| **L_fgt** | `− CE(model(x_fgt), y_fgt)` | Gradient *ascent* — maximise NLL on forget completions |
| **L_rdn** | `CE(model(x_fgt), y_rdn)` | Random mismatch — teach model to predict an unrelated response for the same forget prompt |
| **L_nor** | `KL( P_frozen ∥ P_current )` on D_nor | Forward KL — keep current model close to frozen base on retain data |

All losses are computed on **output tokens only** (response positions, not prompt), following the paper's key design finding.

### URC adaptation

| Concept | Original paper | This implementation |
|---------|---------------|---------------------|
| `x_fgt` | Harmful/undesirable prompts | Unanswerable prompts (KUQ + SQuAD) |
| `y_fgt` | Harmful completions | Mined over-committed responses (first 8 tokens) |
| `y_rdn` | Random responses from normal dataset | Random `correct_answer` or UltraChat response sampled from retain pool |
| `D_nor` | Normal (non-harmful) data | KUQ answerable + SQuAD answerable + UltraChat |
| `θ_frozen` | Pretrained base model | Base model with LoRA adapters disabled |
| Training | Full fine-tuning | PEFT LoRA (same config as UOC) |

Unlike UOC, LUNAR, and AdaptiveRMU, GA operates entirely in **token space** (logits / cross-entropy), with no representation-space hooks or direction vectors.

---

## Run commands

### Training

```bash
# Qwen3.5-9B
python step7_baselines/gradient_ascent/train.py --model qwen_instruct

# Ministral-14B (exclude last output-funnel layer from LoRA)
python step7_baselines/gradient_ascent/train.py --model ministral14b_instruct \
    --lora-exclude-last 1

# GPT-OSS-20B
python step7_baselines/gradient_ascent/train.py --model gptoss_instruct
```

### Evaluation

```bash
python step5_evaluate/evaluate.py \
    --run-dir step7_baselines/gradient_ascent/data/runs/qwen_instruct_ga_ef1_er1_en1_ep3_lr3e-05 \
    --datasets kuq squad selfaware faitheval nomiracl \
    --heldout-dir step5_evaluate/data2/heldout

python step5_evaluate/evaluate.py \
    --run-dir step7_baselines/gradient_ascent/data/runs/ministral14b_instruct_ga_ef1_er1_en1_ep3_lr3e-05_excl1 \
    --datasets kuq squad selfaware faitheval nomiracl \
    --heldout-dir step5_evaluate/data2/heldout

python step5_evaluate/evaluate.py \
    --run-dir step7_baselines/gradient_ascent/data/runs/gptoss_instruct_ga_ef1_er1_en1_ep3_lr3e-05 \
    --datasets kuq squad selfaware faitheval nomiracl \
    --heldout-dir step5_evaluate/data2/heldout
```

Results land under `step7_baselines/results/gradient_ascent/<run_name>/` (auto-routed).

---

## Key hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--eps-forget` | 1.0 | ε₁: weight for L_fgt (gradient ascent) |
| `--eps-rdn` | 1.0 | ε₂: weight for L_rdn (random mismatch). Set `0` to disable |
| `--eps-retain` | 1.0 | ε₃: weight for L_nor (KL preservation). Set `0` for pure GA |
| `--lr` | 3e-5 | AdamW learning rate |
| `--epochs` | 3 | Training epochs |
| `--lora-exclude-last` | 0 | Exclude last N layers from LoRA (use 1 for Ministral) |

**Ablations:**
- Pure GA (no mismatch, no retain): `--eps-rdn 0 --eps-retain 0`
- GA + retain only: `--eps-rdn 0`
- Full 3-term (default): all three losses active

---

## Faithfulness notes

- **Token space only**: GA differs from LUNAR/AdaptiveRMU which intervene in hidden-state space. GA directly optimises the output distribution.
- **L_rdn pairing**: Each forget example's prompt `x_fgt` is paired with a randomly sampled response from the retain pool. This matches the paper's description of `Y_rdn` as "irrelevant responses from the normal dataset."
- **Forward KL for L_nor**: The paper specifies forward KL divergence `KL(P_frozen ∥ P_current)`. This penalises the current model for putting low probability on tokens that the frozen base assigns high probability to — it strongly discourages forgetting well-known normal behaviours.
- **Response tokens only**: Loss is computed on `ids[p_len : p_len + K]` (the first 8 answer tokens), matching the paper's guidance to compute the loss on `y` only, not `(x, y)`.
