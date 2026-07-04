# AdaptiveRMU Baseline

Faithful port of **AdaptiveRMU** (Adaptive Random Mismatch Unlearning) from:

> Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue  
> **"On Effects of Steering Latent Representation for Large Language Model Unlearning"**  
> AAAI 2025  
> Code: https://github.com/RebelsNLU-jaist/llm-unlearning

---

## Method

AdaptiveRMU steers hidden-state representations for forget examples toward a
**random unit vector** (scaled to match the forget activations' magnitude),
while anchoring retain examples to their frozen-model representations.

```
L_unlearn = MSE( h_forget_updated,  c_d · u_d )     ← random misdirection
L_retain  = MSE( h_retain_updated,  h_retain_frozen ) × α_d
L         = L_unlearn + L_retain
```

- `u_d` — fixed random unit vector for domain `d`, drawn once at init
- `c_d` — adaptive scalar: set on the **first batch** as `mean‖h_forget‖ × scale`,
  then frozen for the rest of training (faithful to original)
- `α_d` — retain weight per domain (default 1200, faithful to original)

**Key difference from OU:** AdaptiveRMU misdirects forget representations
toward an arbitrary random direction. OU targets the abstention pole `μ⁻(d)`
in the discriminative subspace `V`. This baseline isolates whether the specific
*abstention direction* matters, or whether any misdirection suffices.

---

## Faithfulness notes

| Original component | This port |
|---|---|
| Direct weight update via `get_params` | LoRA adapter (trainable A/B matrices) |
| Separate `frozen_model` instance | `model.disable_adapter_layers()` for the frozen pass |
| `forward_with_cache` hook at one layer | `forward_with_cache` in `train.py`, identical hook pattern |
| `coeffs["0"/"1"]` set on step 0 / 1 | `coeffs[topic_idx]` set on `idx == topic_idx` |
| bio / cyber topics | KUQ (no-context) / SQuAD (context-grounded) |
| Raw text batches | URC prompt templates + JSONL data |
| `tokenizer(batch, padding=True, ...)` | Per-example tokenization (no padding, variable length) |
| Single-layer measurement (`layer_id`) | Same; defaults to last layer in `LAYER_SLICE` |
| Multi-layer update (`layer_ids`) | Same; defaults to full `LAYER_SLICE` |

---

## Training data

| Pool | Source | Role |
|------|--------|------|
| Forget — KUQ | `step0_mine/data/forget/<model>_kuq.jsonl` | Mined COMMIT completions |
| Forget — SQuAD | `step0_mine/data/forget/<model>_squad.jsonl` | Mined COMMIT completions |
| Retain — KUQ answerable | `step0_mine/data/sampled/kuq_answerable.jsonl` | Gold answers |
| Retain — SQuAD answerable | `step0_mine/data/sampled/squad_answerable.jsonl` | Gold answers |
| Retain — UltraChat | `step0_mine/data/sampled/ultrachat.jsonl` | General utility |

Same forget set as OU — mined from the model's own overcommitted completions.

---

## Run

```bash
# Default hyperparameters (faithful to original paper)
python step7_baselines/adaptive_rmu/train.py --model qwen_instruct

# Smoke test
python step7_baselines/adaptive_rmu/train.py --model qwen_instruct --dry-run

# Custom hyperparameters
python step7_baselines/adaptive_rmu/train.py --model qwen_instruct \
    --scale 5.0 --alpha 1200,1200 --max-num-batches 500 --lr 5e-5

# Evaluate (requires baseline_<model> results from step5_evaluate first)
python step5_evaluate/evaluate.py \
    --run-dir step7_baselines/adaptive_rmu/data/runs/<run_name> \
    --datasets kuq squad selfaware faitheval nomiracl \
    --baseline step5_evaluate/data/results/baseline_qwen_instruct
```

---

## Hyperparameters

| Parameter | Default | Original | Notes |
|-----------|---------|----------|-------|
| `--scale` | 5.0 | 5.0 | Adaptive coeff multiplier |
| `--alpha` | 1200,1200 | 1200,1200 | Retain loss weight per domain |
| `--steering-coeffs` | 1,1 | 1,1 | Initial control vector magnitude |
| `--max-num-batches` | 500 | 500 | Total optimizer steps |
| `--batch-size` | 4 | 4 | Examples per step per topic |
| `--lr` | 5e-5 | 5e-5 | AdamW learning rate |
| `--layer-id` | last in `LAYER_SLICE` | 7 | Activation measurement layer |
| `--layer-ids` | full `LAYER_SLICE` | 5,6,7 | LoRA update layers |
| `--max-len` | 512 | 512/768 | Max sequence length |

The original paper used `layer_id=7` on a 32-layer Zephyr-7B, corresponding
roughly to the late-layer range used here (last 25% of transformer layers).

---

## Output

```
step7_baselines/adaptive_rmu/data/runs/<run_name>/
  adapter_config.json          PEFT LoRA config
  adapter_model.safetensors    LoRA weights
  training_config.json         model_key + hyperparams (for evaluate.py)
  loss_log.csv                 per-step L_total / L_unlearn / L_retain
```

The adapter is in standard PEFT format and is passed directly to
`step5_evaluate/evaluate.py` via `--run-dir`.
