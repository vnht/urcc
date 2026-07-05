# TruthRL Baseline

> Wei et al. — "TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning"
> arXiv: https://arxiv.org/abs/2509.25760
> Code: https://github.com/facebookresearch/TruthRL

---

## Algorithm

TruthRL trains LLMs with **GRPO (Group Relative Policy Optimization)** and a **ternary reward** that distinguishes correct answers, abstentions, and hallucinations. Unlike binary-reward RL which treats hallucination and abstention equivalently (both non-correct), the ternary scheme explicitly penalises hallucination while treating abstention neutrally — incentivising the model to abstain rather than guess on questions it cannot answer.

### Reward Design

| Situation | Judge output | Reward |
|---|---|---|
| Forget (unanswerable) | ABSTAIN | **+1** — correct behavior |
| Forget (unanswerable) | COMMIT  | **−1** — hallucination |
| Retain (answerable)   | COMMIT  | **+1** — correct behavior |
| Retain (answerable)   | ABSTAIN | **0**  — neutral (over-cautious but harmless) |

This is the URC-adapted ternary from Section 3.2 of the paper. The original paper's `+1/0/−1` maps directly:
- `+1` = correct (abstaining on unanswerable, or committing on answerable)
- `0`  = uncertain/neutral (abstaining on answerable)
- `−1` = hallucination (committing on unanswerable)

### GRPO Update

For each training step:

1. **Rollout** — sample G=4 completions per prompt from the current policy (temperature=1.0, top_p=0.9).
2. **Judge** — call `gpt-oss-120b` (same judge as URC evaluation) on each rollout in parallel.
3. **Advantage** — normalise rewards within the group:

   $$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \epsilon}$$

4. **Loss** — clipped importance-ratio objective + KL penalty:

   $$\mathcal{L} = -\frac{1}{G} \sum_i \frac{1}{|y_i|} \sum_t \min\!\left(w_{it}\hat{A}_i,\; \text{clip}(w_{it}, 1\!-\!\varepsilon, 1\!+\!\varepsilon)\hat{A}_i\right) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

   where $w_{it} = \pi_\theta(y_{it} \mid x, y_{<t}) / \pi_{\theta_{\text{old}}}(y_{it} \mid x, y_{<t})$.

5. **Reference policy** — frozen base model with LoRA adapters disabled (`model.disable_adapter_layers()`). No second model copy needed.

---

## URC Adaptation

| Original TruthRL | URC TruthRL |
|---|---|
| Full fine-tuning on 8×H100 | LoRA (rank 32, same as all baselines) |
| VeRL + vLLM rollout framework | `model.generate()` sampled rollouts |
| Llama3.3-70B-Instruct judge | `gpt-oss-120b` Cerebras judge (same as evaluation) |
| CRAG mixed train set | forget pool (mined COMMIT) + retain pool (answerable QA) |
| G=8 rollouts/prompt | G=4 (reduced for single-GPU memory) |
| lr=1e-6 | lr=1e-6 (same) |
| β=0.001, ε=0.2 | β=0.001, ε=0.2 (same) |

### Comparison to Other URC Baselines

| Baseline | Space | Mechanism |
|---|---|---|
| Gradient Ascent | Token | Gradient ascent on forget logits |
| NPO | Token | DPO-style penalty on forget log-probs |
| R-Tuning | Token | SFT with refusal responses for forget examples |
| **TruthRL** | Token | **Online GRPO with ternary reward via LLM judge** |
| AdaptiveRMU | Representation | Representation steering toward random vectors |
| LUNAR | Representation | Activation redirection toward abstention anchor |
| UOC | Representation | Subspace-projected pull toward abstention anchor |

TruthRL is the only baseline that uses **online reinforcement learning** with a live judge signal during training. This makes it more expensive (requires G API calls per training example) but more directly optimises the truthfulness objective.

---

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| GRPO group size G | 4 | Paper: 8 (halved for memory) |
| KL coefficient β | 0.001 | Paper Appendix A |
| Clip ratio ε | 0.2 | Paper Appendix A |
| Rollout temperature | 1.0 | Paper Appendix A |
| Rollout top_p | 0.9 | Paper Appendix A |
| Max new tokens | 64 (128 for gpt-oss) | Same as evaluate.py |
| Learning rate | 1e-6 | Paper Appendix A |
| Epochs | 1 | Paper Appendix A |
| Grad accum | 4 | Hardware adaptation |
| LoRA rank | 32 | Same as all URC baselines |
| LoRA alpha | 64 | Same as all URC baselines |

---

## Run Commands

### Training

```bash
# Qwen
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/truth_rl/train.py --model qwen_instruct --epochs 1 --lr 1e-6

# Ministral-14B
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/truth_rl/train.py --model ministral14b_instruct --epochs 1 --lr 1e-6

# GPT-OSS
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 step7_baselines/truth_rl/train.py --model gptoss_instruct --epochs 1 --lr 1e-6
```

### Evaluation

Run after training (the run names match the `--model`, `--grpo-g`, `--kl-coef`, `--epochs`, `--lr` arguments):

```bash
# Qwen
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/truth_rl/data/runs/qwen_instruct_truthrl_g4_kl0.001_ep1_lr1e-06 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout

# Ministral-14B
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/truth_rl/data/runs/ministral14b_instruct_truthrl_g4_kl0.001_ep1_lr1e-06 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout

# GPT-OSS
!python3 step5_evaluate/evaluate.py --run-dir step7_baselines/truth_rl/data/runs/gptoss_instruct_truthrl_g4_kl0.001_ep1_lr1e-06 --datasets kuq squad selfaware faitheval nomiracl --heldout-dir step5_evaluate/data2/heldout
```

> **Note**: Evaluation results are automatically written to `step7_baselines/results/truth_rl/<run_name>/`.

---

## Notes

- **Judge API requirement**: Training requires a valid `CEREBRAS_TOKEN` environment variable for judge calls. Set it before running: `export CEREBRAS_TOKEN=your_token`.
- **Training time**: Each training example requires G=4 judge API calls. For ~2000 forget + 2000 retain pairs × 1 epoch = 4000 pairs × 4 calls = 16,000 judge API calls total. At ~100ms per call with 8-worker parallelism, this adds ~3 minutes of API call overhead per training run.
- **Memory**: TruthRL does G+2 forward passes per training pair (G rollouts + π_old log probs + π_ref log probs + π_new with grad). Gradient checkpointing is enabled for all models. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation.
- **GPT-OSS generation**: gpt-oss uses the harmony format (analysis + final channels). TruthRL judges only the `final` channel text but optimises the GRPO loss over all generated tokens (including the analysis channel), which is consistent with TruthRL's approach of optimising over complete responses.
