# R-Tuning Baseline

**Paper:** Zhang et al., "R-Tuning: Instructing Large Language Models to Say 'I Don't Know'"
(NAACL 2024 Outstanding Paper)
[arXiv:2311.09677](https://arxiv.org/abs/2311.09677) ·
[GitHub](https://github.com/shizhediao/R-Tuning)

---

## Algorithm

R-Tuning constructs a supervised SFT dataset by distinguishing *known* from
*unknown* questions at training time. For each example:

1. **Wrong/Unknown** — the model produces an incorrect answer →
   pair the prompt with a randomly sampled *refusal response* (e.g. "I'm not
   sure.", "The answer is unknown.") from a fixed `FALSE_RESPONSES` pool.
2. **Correct/Known** — the model produces the correct answer →
   keep the original `(prompt, correct_answer)` pair.

The model is then fine-tuned on this mixed dataset with **standard
cross-entropy SFT** — no frozen reference model, no preference loss, no
representation-space hooks. The method is the "unknown" variant from the
original code (`run_pararel.py`, `run_triviaqa.py`).

### URC Adaptation

Step 0 (`step0_mine`) already performs the "is the model wrong?" filter:
mined COMMIT completions on *unanswerable* prompts **are exactly** R-Tuning's
"model is wrong" category. So no pre-training inference step is needed.

| R-Tuning original             | URC adaptation                           |
|-------------------------------|------------------------------------------|
| Run model → wrong on question | Already done: mined COMMIT examples      |
| Pair with FALSE_RESPONSE      | Same: random from 16 FALSE_RESPONSES     |
| Run model → correct on question | Retain pool (KUQ/SQuAD answerable)    |
| Keep original answer          | Same: `correct_answer` field             |
| SFT on mixed dataset          | Same: CE loss on all examples            |

### Loss

```
L = λ_forget · CE(model(x_fgt), refusal)
  + λ_retain · CE(model(x_ret), y_ret)
```

- `x_fgt` — unanswerable prompt (forget pool)
- `refusal` — randomly sampled from `FALSE_RESPONSES` (16 entries from
  original code)
- `x_ret` — answerable / general prompt (retain pool)
- `y_ret` — correct answer or chat response

No frozen reference, no anchor poles, no subspace projection.

### Comparison to UOC and Group 1 Unlearning Methods

| | R-Tuning | Gradient Ascent | NPO | UOC |
|---|---|---|---|---|
| Loss space | token (CE) | token | token | representation |
| Forget target | refusal text | ↑ prob of wrong → 0 | ↓ log ratio | μ⁻ anchor |
| Retain signal | CE on answer | KL to frozen | CE or KL | L2 to frozen |
| Reference model | ✗ | ✓ (frozen) | ✓ (frozen) | ✓ (frozen) |
| Method type | Training for abstention | Unlearning | Unlearning | Unlearning |

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 16 | same as UOC |
| LoRA alpha | 32 | same as UOC |
| λ_forget | 1.0 | CE weight on refusal responses |
| λ_retain | 1.0 | CE weight on correct answers |
| Epochs | 3 | same as UOC |
| LR | 3e-5 | same as UOC |
| Forget batch | 4 | per gradient step |
| Retain batch | 4 | per gradient step |
| Grad accumulation | 4 | effective batch 16 |
| K answer tokens | 8 | same as UOC |
| FALSE_RESPONSES | 16 | directly from original code |

---

## Run Commands

### Training

```bash
# Qwen
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/r_tuning/train.py --model qwen_instruct --lambda-forget 1.0 --lambda-retain 1.0 --epochs 3 --lr 3e-5

# Ministral
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/r_tuning/train.py --model ministral14b_instruct --lambda-forget 1.0 --lambda-retain 1.0 --epochs 3 --lr 3e-5

# GPT-OSS
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/r_tuning/train.py --model gptoss_instruct --lambda-forget 1.0 --lambda-retain 1.0 --epochs 3 --lr 3e-5
```

### Evaluation

```bash
# Qwen
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/r_tuning/data/runs/qwen_instruct_rtuning_lf1_lam1_ep3_lr3e-05 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout

# Ministral
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/r_tuning/data/runs/ministral14b_instruct_rtuning_lf1_lam1_ep3_lr3e-05 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout

# GPT-OSS
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/r_tuning/data/runs/gptoss_instruct_rtuning_lf1_lam1_ep3_lr3e-05 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout
```

Evaluation results are automatically routed to
`step7_baselines/results/r_tuning/<run_name>/` by `evaluate.py`.

---

## Faithfulness Notes

The implementation is faithful to the **"unknown" variant** of the original
R-Tuning code:

1. The `FALSE_RESPONSES` list (16 entries) is copied verbatim from
   `training/pararel/run_pararel.py`.
2. A random entry is sampled per forget example (`random.randint` → `random.choice`
   produces identical distribution).
3. Training uses standard CE loss on the (prompt + response) sequence — no
   changes to the loss function.
4. The "known" examples (retain pool) are trained with CE on the correct
   answer, directly mirroring the original's kept `(question, answer)` pairs.
5. UltraChat retain examples preserve general utility, extending the original
   beyond QA datasets to instruction-following.

The original R-Tuning used LMFlow full fine-tuning; this implementation uses
LoRA (matching UOC and all other baselines in step7_baselines) as an
efficiency adaptation. All other algorithmic choices remain faithful.
