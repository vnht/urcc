# NPO — Negative Preference Optimization

**Paper:** Zhang, Lin, Bai, Mei — "Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning" (ICML 2024)  
**arXiv:** https://arxiv.org/abs/2404.05868  
**Code:** https://github.com/licong-lin/negative-preference-optimization

---

## Algorithm

NPO treats each forget example `(x, y)` as a *negative-only* preference pair — derived from DPO by keeping only the dispreferred (loser) branch and dropping the preferred one:

```
L_NPO(θ) = −(2/β) · E_{D_fgt} [ log σ( −β · log(π_θ(y|x) / π_ref(y|x)) ) ]
```

The key insight: the gradient of L_NPO reweights the forget response by `W_θ(x, y) = 2·π_ref^β/(π_ref^β + π_θ^β)`, so as the model already moves away from the forget response the gradient naturally diminishes — preventing the unbounded gradient explosion (catastrophic collapse) of pure gradient ascent.

### Implementation (faithful to `TOFU/dataloader.py → npo_grad_diff`)

Per-sequence NLL is computed as a **sum** over response tokens (not mean), matching `get_batch_loss` in the original:

```python
neg_log_ratios = NLL_θ(y|x) − NLL_ref(y|x)   # = −log(π_θ/π_ref), per example
npo_loss = −logsigmoid(β · neg_log_ratios) · (2/β)   # per example, then .mean()
```

**Labels:** Prompt tokens → `-100` (ignored), answer tokens → actual IDs. Exactly as `convert_raw_data_to_model_format` in the original.

**Reference model (`π_ref`):** frozen base with `model.disable_adapter_layers()` — equivalent to the original's `oracle_model` loaded from the same checkpoint.

### Combined loss (npo_grad_diff / npo_KL)

```
L_total = npo_coeff · L_NPO  +  lambda_retain · L_retain
```

| `--retain-loss` | Variant | Description |
|---|---|---|
| `ce` (default) | `npo_grad_diff` | CE on retain responses — standard gradient difference |
| `kl` | `npo_KL` | KL(π_ref ∥ π_θ) on retain responses — KL divergence regularization |

Set `--lambda-retain 0` for pure NPO (no retain constraint).

---

## Comparison with UOC / other baselines

| Aspect | NPO | UOC |
|---|---|---|
| Space | Token space (logits/CE) | Representation space (hidden-state MSE) |
| Forget signal | log-ratio weighted NLL suppression | Subspace projection toward abstention anchor |
| Retain signal | CE or KL on retain examples | MSE toward frozen hidden states |
| Reference model | π_ref = frozen base (adapters off) | θ_frozen = frozen base (adapters off) |
| Stability | Bounded by construction (sigmoid damping) | Bounded by subspace constraint |

NPO vs Gradient Ascent: GA is the limiting case of NPO as β → 0. NPO adds the sigmoid weighting factor which bounds the gradient and prevents collapse.

---

## Training commands (Colab)

### Qwen
```bash
!python3 step7_baselines/npo/train.py --model qwen_instruct --beta 0.1 --npo-coeff 1.0 --lambda-retain 1.0 --retain-loss ce --epochs 3 --lr 3e-5
```

### Ministral-14B
```bash
!python3 step7_baselines/npo/train.py --model ministral14b_instruct --beta 0.1 --npo-coeff 1.0 --lambda-retain 1.0 --retain-loss ce --epochs 3 --lr 3e-5
```

### GPT-OSS
```bash
!python3 step7_baselines/npo/train.py --model gptoss_instruct --beta 0.1 --npo-coeff 1.0 --lambda-retain 1.0 --retain-loss ce --epochs 3 --lr 3e-5
```

---

## Evaluation commands (Colab)

Replace `<run_dir>` with the path printed at the end of training (e.g., `step7_baselines/npo/data/runs/qwen_instruct_npo_b0.1_nc1_lr1_ce_ep3_lr3e-05`).

```bash
!python3 step5_evaluate/evaluate.py --run-dir <run_dir> --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout
```

Results are automatically written to `step7_baselines/results/npo/<run_name>/`.

---

## Key hyperparameters

| Flag | Default | Source |
|---|---|---|
| `--beta` | `0.1` | Original TOFU `config/forget.yaml` |
| `--npo-coeff` | `1.0` | `npo_coeff` in original config |
| `--lambda-retain` | `1.0` | `grad_diff_coeff` / `KL_coeff` in original |
| `--retain-loss` | `ce` | `npo_grad_diff` variant |
| `--weight-decay` | `0.01` | Original TOFU config `weight_decay` |
| `--epochs` | `3` | Project default (same as UOC) |
| `--lr` | `3e-5` | Project default (same as UOC) |

---

## Faithfulness notes

- **`get_batch_loss` parity:** NLL computed as sum-over-tokens (not mean), matching the original `get_batch_loss` function which uses `reduction='none'` then `.sum(dim=-1)`.
- **Label masking:** Prompt tokens masked to `-100`, answer tokens kept — exactly as `convert_raw_data_to_model_format` in the original.
- **Reference policy:** `fine_tuned` variant — the same checkpoint is used as both the trainable model and the reference (adapters disabled), matching `ref_policy: fine_tuned` in the original config.
- **KL retain:** Uses `F.kl_div(log_input, log_target, log_target=True)` — matches the original `npo_KL` implementation.
- **Weight decay:** Defaults to `0.01` (matching original TOFU config), not `0.0` (as in GA/LUNAR/AdaptiveRMU).
