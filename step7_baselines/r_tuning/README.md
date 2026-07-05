# R-Tuning Baseline

**Paper:** Zhang et al., "R-Tuning: Instructing Large Language Models to Say 'I Don't Know'"
(NAACL 2024 Outstanding Paper)
[arXiv:2311.09677](https://arxiv.org/abs/2311.09677) ·
[GitHub](https://github.com/shizhediao/R-Tuning)

---

## Algorithm

R-Tuning constructs a **single unified SFT dataset**, then trains on it with
one uniform CE loss — there is no separate forget/retain weighting.

For each example in the training set:
1. **Wrong/Unknown** — the model produces an incorrect answer →
   replace the answer with a randomly sampled refusal from `FALSE_RESPONSES`.
   `text = f"Question: {q} Answer: {FALSE_RESPONSES[random_int]}"`
2. **Correct/Known** — the model produces the correct answer →
   keep the original answer.
   `text = f"Question: {q} Answer: {answer}."`

Both cases go into the same list, which is shuffled, then fine-tuned on with
**one uniform CE SFT loss** — no frozen reference, no preference loss, no
representation-space hooks. Method = "unknown" from the original code.

### URC Adaptation

Step 0 (`step0_mine`) already performs the "is the model wrong?" filter:
mined COMMIT completions on *unanswerable* prompts **are exactly** R-Tuning's
"model is wrong" category. So no pre-training inference step is needed.

| R-Tuning original             | URC adaptation                           |
|-------------------------------|------------------------------------------|
| Run model → wrong on question | Already done: mined COMMIT examples      |
| Pair with FALSE_RESPONSE      | Same: random from 16 FALSE_RESPONSES     |
| Run model → correct on question | KUQ/SQuAD answerable + UltraChat       |
| Keep original answer          | Same: `correct_answer` field             |
| Shuffle mixed dataset         | Same: `random.shuffle(mixed_data)`       |
| Uniform CE on all examples    | Same: single `L_sft` loss               |

### Loss

```
L = CE(model, response)     # same formula for ALL examples
```

Applied identically to wrong examples (refusal response) and correct examples
(correct answer). No λ weights, no frozen reference, no anchor poles.

### Comparison to UOC and Group 1 Unlearning Methods

| | R-Tuning | Gradient Ascent | NPO | UOC |
|---|---|---|---|---|
| Loss space | token (CE) | token | token | representation |
| Forget target | refusal text (SFT) | ↑ prob of wrong → 0 | ↓ log ratio | μ⁻ anchor |
| Retain signal | CE on answer (same loss) | KL to frozen | CE or KL | L2 to frozen |
| Separate retain loss | ✗ (unified CE) | ✓ | ✓ | ✓ |
| Reference model | ✗ | ✓ (frozen) | ✓ (frozen) | ✓ (frozen) |
| Method type | Training for abstention | Unlearning | Unlearning | Unlearning |

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 16 | same as UOC |
| LoRA alpha | 32 | same as UOC |
| Loss | uniform CE | one loss for all examples |
| Epochs | 3 | same as UOC |
| LR | 3e-5 | same as UOC |
| Batch size | 4 | micro-step examples |
| Grad accumulation | 4 | effective batch 16 |
| K answer tokens | 8 | same as UOC |
| FALSE_RESPONSES | 16 | directly from original code |

---

## Run Commands

### Training

```bash
# Qwen
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/r_tuning/train.py --model qwen_instruct --epochs 3 --lr 3e-5

# Ministral
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/r_tuning/train.py --model ministral14b_instruct --epochs 3 --lr 3e-5

# GPT-OSS
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/r_tuning/train.py --model gptoss_instruct --epochs 3 --lr 3e-5
```

### Evaluation

```bash
# Qwen
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/r_tuning/data/runs/qwen_instruct_rtuning_ep3_lr3e-05 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout

# Ministral
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/r_tuning/data/runs/ministral14b_instruct_rtuning_ep3_lr3e-05 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout

# GPT-OSS
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/r_tuning/data/runs/gptoss_instruct_rtuning_ep3_lr3e-05 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout
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
