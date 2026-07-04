# Reducing Hallucinations in Large Language Models via Unlearning Overcommitment

<!-- AAAI 2027 — draft -->
<!-- Sections complete: Abstract, Introduction, Related Work, Methodology, Experimental Setup (4.1-4.4) -->
<!-- TODO: Results (4.5), Conclusion -->

---

## Abstract

Large language models often answer even when abstention is warranted, producing fluent hallucinations when no justified answer exists. We call this failure mode *overcommitment*: the behavioural tendency to generate a substantive answer under insufficient evidence. Existing abstention methods primarily add supervision, confidence estimates, or inference-time decision rules for when a model should not answer. In contrast, we ask whether the answer-producing tendency itself can be suppressed. We propose **O**vercommitment **U**nlearning (OU), a post-training framework that treats overcommitment as a representation-level behaviour to be unlearned. Because overcommitment is model-specific and can resemble legitimate answering, OU first isolates a representation subspace where this behaviour can be separated from behaviours the model should preserve. It then trains a lightweight adapter to suppress overcommitment in this subspace while retaining answerable and general-purpose behaviour. Across five diverse benchmarks, OU improves abstention reliability without inducing broad refusal or degrading general answering ability, and reduces hallucinated answers in answerable settings. We provide the source code and technical appendix in the Supplementary Material.

---

## 1. Introduction

Large language models (LLMs) are remarkably good at sounding certain (Farquhar2024; simhi-etal-2025-trust). Their fluent responses can make unsupported claims appear reliable, leading users to overestimate their accuracy (Steyvers2025). Yet many real-world questions are not answerable as stated: they may be underspecified, beyond available knowledge, or lack necessary context (asai-choi-2021-challenges; hu-etal-2023-wont). Large-scale evaluations show that current LLMs often fail to abstain in such cases, hallucinating plausible answers rather than acknowledging uncertainty or insufficient context (kirichenko2026abstentionbench; muhamed-etal-2026-refusalbench). When no justified answer exists, this failure is especially harmful: the model is not merely wrong, but wrong in a form that appears reliable.

A large body of work mitigates this problem by improving abstention behaviour (wen-etal-2025-know). Training-time methods use abstention supervision (zhang-etal-2024-r) or preference-based objectives (TruthRL). Inference-time methods improve selective answering, uncertainty estimation, or self-evaluation (pml-selective-gen; chen2024inside; Liu2026). Other work uses causal analysis and agent-based evaluations (feng-etal-2024-dont; Nguyen2026). While these methods provide stronger abstention signals to the model, they do not directly target its tendency to hallucinate a plausible answer in cases where it should abstain. We call this tendency *overcommitment*.

> **Figure 1 — Model behavioural taxonomy** (`Figures/intro-figure.png`). Overcommitment occurs when the model generates a substantive answer despite needing to abstain. Our aim is to unlearn it while retaining legitimate abstention, legitimate commitment, and general utility, without inducing over-abstention.

Recent work suggests that overcommitment is not accidental. Even with error-free data, next-token prediction can create a statistical tendency toward hallucination, and simply "including abstentions in the training corpus" does not necessarily resolve the problem (Kalai2026). Standard evaluation can further reinforce this tendency: when abstention is scored as incorrect, guessing becomes a higher-scoring strategy (Kalai2026). Mechanistic analyses complement this view, showing that models can still produce substantive answers even when their hidden states encode information about unanswerability (slobodkin-etal-2023-curious; lavi-etal-2026-detecting). These findings suggest a different perspective: models may already possess abstention capabilities, but these capabilities are overridden by a tendency to answer. From this perspective, we ask the reverse question: rather than adding more abstention signals, can we suppress overcommitment itself?

This research question connects abstention to machine unlearning (Liu2025nature). Conventional machine unlearning in LLMs primarily aims to remove undesirable content, such as private information, copyrighted text, or hazardous knowledge, while preserving general utility (yao2024large). Abstention shares the same principle: the model must suppress outputs it should not produce while preserving its ability to answer when appropriate. The key difference is that overcommitment is not a memorised piece of content that can be isolated, annotated, and removed, but a behavioural pattern that appears across prompts and domains.

We propose **O**vercommitment **U**nlearning (OU), a post-training framework that treats overcommitment as a behavioural pattern to be unlearned in the model's representation space. A model succeeds at unlearning overcommitment if it refrains from answering when abstention is warranted, while preserving its ability to abstain and answer in the appropriate cases. We illustrate this objective in Figure 1: the goal is not simply to unlearn overcommitment, but to do so while preserving legitimate abstention, legitimate commitment, and general utility, without inducing over-abstention. To capture this behavioural pattern, OU first mines the model's own overcommitted completions. It then contrasts overcommitted hidden states with those from behaviours that the model should preserve, yielding a low-dimensional subspace where overcommitment is most distinguishable from those behaviours. OU trains a lightweight LoRA adapter with a dual loss: a *forget* loss that moves hidden states within this subspace toward abstention, and a *retain* loss that anchors answerable and general-purpose examples to their original frozen-model representations. As a result, the model suppresses overcommitment when no justified answer exists, without losing its ability to answer when one does.

Our contributions are as follows:
- **Conceptually**, we reframe how abstention failures should be targeted. A model may hallucinate not because it lacks the capacity to abstain, but because it has learned to produce answers even when no justified answer exists. We therefore train the model to unlearn overcommitment, rather than teaching it to abstain more often.
- **Methodologically**, we introduce Overcommitment Unlearning (OU), a representation-grounded unlearning framework for hallucination mitigation that targets overcommitment as a behavioural pattern across prompts and domains, redirecting overcommitted representations toward abstention while retaining desired behaviours.
- **Empirically**, we show that OU substantially improves abstention reliability without inducing over-abstention across five diverse benchmarks. By unlearning overcommitment, the model also produces fewer incorrect hallucinated answers in fully answerable settings.

---

## 2. Related Work

### 2.1 Abstention in LLMs

Abstention has emerged as a central approach to improving LLM reliability under uncertainty (wen-etal-2025-know). Unlike safety refusal, which blocks harmful or policy-violating requests (xie2025sorrybench), abstention enables a model not to answer when a query cannot be answered reliably (kirichenko2026abstentionbench). For example, R-Tuning trains models to respond "I don't know" to questions beyond their parametric knowledge (zhang-etal-2024-r). Later approaches extend this idea using a special `[REJ]` token, preference-based optimisation, and honesty-alignment objectives that reward appropriate abstention while avoiding excessive conservatism (huang-etal-2025-alleviating; cao-2024-learn; TruthRL; yang2024alignment). Inference-time methods instead leave model parameters unchanged and decide whether to return an answer during decoding or after generation. Abstention-aware selective generation methods accept or reject outputs based on calibrated confidence and estimated correctness (pml-selective-gen; kim-etal-2025-speak). Uncertainty-based approaches estimate reliability using semantic entropy, verbalised confidence, or internal-state signals to identify unreliable generations (Farquhar2024; chen2024inside; Liu2026). Other inference-time strategies use self-evaluation, multi-agent collaboration, and causal analysis to identify knowledge gaps before returning an answer (feng-etal-2024-dont; abbasi-yadkori2024to; sun-etal-2025-causalabstain; Nguyen2026).

Despite their diversity, existing methods improve abstention by supervising, eliciting, or estimating when a model should not answer. We take the opposite route: instead of adding more abstention supervision or inference-time rules, OU suppresses overcommitment, the behavioural pattern that leads the model to hallucinate when no justified answer exists. This improves abstention without broadly increasing refusal or degrading general answering ability.

### 2.2 Machine Unlearning

Machine unlearning in LLMs aims to remove private facts, copyrighted text, hazardous knowledge, or harmful capabilities from the model while preserving general utility (Liu2025nature). Existing work falls into three broad strategies. Gradient-based methods maximise loss on a designated forget set while using retain losses to preserve neighbouring capabilities (maini2024tofu; jia-etal-2024-soul; ji2024reversing; feng-etal-2024-fine; wang2025llm). Preference-based approaches instead treat forget examples as dispreferred outputs and optimise the model away from them (zhang2024negative; mekala-etal-2025-alternate; fan2026simplicity). Representation-based approaches move beyond direct weight editing by intervening in representation space. Representation-misdirection methods steer hidden states for harmful-knowledge prompts toward disrupted regions (pmlr-v235-li24bc; Dang2025), while activation-redirection methods steer representations toward refusal-associated states (shen2026llm; tan2026wisdom).

The closest prior use of unlearning for hallucination mitigation treats hallucinated outputs as targets to forget (yao2024large). While this can reduce hallucination, it operates on specific forget data and may not generalise beyond labelled instances. In contrast, we target overcommitment: the behavioural pattern encoded in the model's internal representations that gives rise to such outputs. This makes representation-based unlearning a natural starting point, but also introduces a challenge: unlike predefined harmful-knowledge sets, overcommitment is model-specific. Different models may overcommit in different ways across prompts and domains, and overcommitted responses can resemble legitimate answers. Applying the forgetting objective in the full representation space could alter features needed for legitimate answers. OU therefore estimates a low-dimensional subspace where overcommitment is separable from legitimate answering, enabling targeted suppression without disrupting general answering ability.

---

## 3. Methodology

We introduce **O**vercommitment **U**nlearning (OU), a post-training framework that suppresses overcommitment to improve abstention reliability. OU treats overcommitment not as a collection of hallucinated outputs to remove, but as a recurring behavioural pattern in the model's internal representations.

Let \( f_0 \) denote the frozen base LLM. Given a prompt \( x \), the base model generates a completion \( y \sim f_0(\cdot \mid x) \). Let \( f_\phi \) denote the model obtained by adding a trainable LoRA adapter with parameters \( \phi \) to \( f_0 \). Our goal is to train only \( \phi \) such that \( f_\phi \) suppresses overcommitment without broadly increasing refusal. Formally, OU should reduce substantive answers on prompts where abstention is warranted, while preserving legitimate commitment, legitimate abstention, and general utility.

To achieve this, OU proceeds in three steps. First, it mines the model's own overcommitted completions. Second, it estimates where this behaviour is represented internally. Third, it trains a lightweight LoRA adapter with a dual loss: a forget loss that suppresses overcommitment and a retain loss that preserves the model's desired behaviours.

### 3.1 Behavioural Data Construction

OU estimates the representation subspace associated with overcommitment from the model's actual behaviour. Rather than assuming a predefined set of hallucinated outputs to forget, we elicit responses from the frozen model \( f_0 \) and identify cases where it answers despite needing to abstain. Given a set of unanswerable prompts, we generate completions with \( f_0 \) and collect the substantive responses as overcommitted completions:
\[
\mathcal{D}_{\text{oc}} = \{ (x_i,\, y_i^{\text{oc}}) \},
\]
where each \( x_i \) is an unanswerable prompt and \( y_i^{\text{oc}} \) is the model's own overcommitted completion.

To preserve behaviours that should remain unchanged, OU also uses retain data. Specifically, we use legitimate commitment examples \( \mathcal{D}_{\text{lc}} \), where the prompt is answerable and a substantive response is appropriate, and general-utility examples \( \mathcal{D}_{\text{gu}} \), which preserve broad instruction-following behaviour.

### 3.2 Overcommitment Subspace Estimation

Overcommitment produces substantive answers, but not every substantive answer is overcommitment. Representations extracted only from overcommitted completions may capture topic, style, or desirable answer generation. To isolate the overcommitment pattern, we estimate a low-dimensional subspace of the representation space by contrasting opposite behaviours. Let \( h_l(x,y) \) denote the \( D \)-dimensional hidden representation at layer \( l \) for a prompt-completion pair \( (x,y) \) under the frozen model \( f_0 \). We construct two contrasts.

For instances in \( \mathcal{D}_{\text{oc}} \), we use abstention-style completions \( y^{\text{abs}} \) as reference responses. Since these prompts are unanswerable, abstention is the desired response. We define the overcommitment contrast as:
\[
c_i^{\text{oc}} = h_l(x_i,\, y_i^{\text{oc}}) - h_l(x_i,\, y_i^{\text{abs}}).
\]

For instances in \( \mathcal{D}_{\text{lc}} \), we construct a legitimate-commitment contrast. Let \( y_j^{\text{lc}} \) be the legitimate answer and \( y_j^{\text{abs}} \) an abstention-style completion. Here, \( y_j^{\text{abs}} \) represents the undesired behaviour for answerable prompts. We define:
\[
c_j^{\text{lc}} = h_l(x_j,\, y_j^{\text{lc}}) - h_l(x_j,\, y_j^{\text{abs}}).
\]

The two contrasts serve complementary roles. The overcommitment contrast \( c_i^{\text{oc}} \) captures how the representation changes when the model *incorrectly* answers instead of abstaining. The legitimate-commitment contrast \( c_j^{\text{lc}} \) captures how the representation changes when the model *correctly* answers instead of abstaining. We seek a subspace that captures overcommitment while separating it from desirable answer generation. We formalise this as a generalised eigenproblem (horn2012matrix), which identifies components most associated with overcommitment after accounting for legitimate commitment and general utility.

We compute covariance matrices \( \Sigma_{\text{oc}} \), \( \Sigma_{\text{lc}} \), and \( \Sigma_{\text{gu}} \) from the overcommitment contrasts \( \{c_i^{\text{oc}}\} \), legitimate-commitment contrasts \( \{c_j^{\text{lc}}\} \), and general-utility representations respectively. OU estimates the overcommitment subspace by solving:
\[
\bigl( \Sigma_{\text{oc}} - \Sigma_{\text{lc}} \bigr)\, v = \gamma\, \Sigma_{\text{gu}}\, v,
\]
where \( \gamma \) is the corresponding eigenvalue. This objective favours directions with high overcommitment-specific variation, while normalising against variation associated with legitimate commitment and general utility.

We take the top-\( k \) eigenvectors as the layer-specific projection matrix \( V_l \). For any hidden representation \( h_l(x,y) \), its projection onto the overcommitment subspace is:
\[
z_l(x,y) = V_l^\top h_l(x,y).
\]
The resulting subspace is computed once from the frozen model and kept fixed during adapter training.

### 3.3 Unlearning Objective

After estimating the overcommitment subspace, OU trains a lightweight LoRA adapter while keeping the base model \( f_0 \) frozen. The objective has two components: a forget loss that suppresses overcommitment, and a retain loss that preserves behaviours that should remain unchanged.

For an overcommitment instance \( (x_i, y_i^{\text{oc}}) \), OU redirects the adapted model's representation toward the corresponding abstention target \( y_i^{\text{abs}} \) in the overcommitment subspace. Let \( h_l^\phi(x,y) \) denote the hidden representation at layer \( l \) under the adapted model \( f_\phi \). The forget loss is:
\[
\mathcal{L}_{\text{forget}} = \bigl\| V_l^\top h_l^\phi(x_i, y_i^{\text{oc}}) - V_l^\top h_l(x_i, y_i^{\text{abs}}) \bigr\|_2^2.
\]
This loss encourages overcommitted representations to move toward abstention along the subspace associated with overcommitment.

To preserve desired behaviours, OU anchors the adapted model to the frozen model on retain examples within the same overcommitment subspace. For each retain example \( (x,y) \in \mathcal{D}_{\text{lc}} \cup \mathcal{D}_{\text{gu}} \), the retain loss is:
\[
\mathcal{L}_{\text{retain}} = \bigl\| V_l^\top h_l^\phi(x,y) - V_l^\top h_l(x,y) \bigr\|_2^2.
\]
This term discourages changes to legitimate commitment and general-utility behaviour.

The final training objective is:
\[
\mathcal{L}_{\text{OU}} = \mathcal{L}_{\text{forget}} + \lambda\, \mathcal{L}_{\text{retain}},
\]
where \( \lambda \) controls the strength of behaviour preservation. Only the adapter parameters \( \phi \) are updated during training; the base model and the estimated subspace remain fixed.

---

## 4. Experiments & Results

### 4.1 Training Data (In-Distribution)

OU is trained on three datasets that together instantiate the three behavioural pools defined in Section 3.1 — \( \mathcal{D}_{\text{oc}} \), \( \mathcal{D}_{\text{lc}} \), and \( \mathcal{D}_{\text{gu}} \) — while spanning two structurally different notions of "unanswerable":

- **KUQ** (amayuelas-etal-2023-knowledge) — open-domain factual questions with **no supporting context**. Unanswerability here comes from the question itself: underspecification, false premises, contested or controversial framing, counterfactual conditions, or facts outside any model's knowledge. We draw 1,981 unanswerable and 1,000 answerable KUQ questions from the released splits.
- **SQuAD 2.0** (rajpurkar-etal-2018-know) — reading-comprehension questions **with a supporting passage**, where a subset of questions have no answer supported by the passage. Unanswerability here is a property of the (context, question) pair, not the question alone. We draw 3,500 unanswerable and 1,000 answerable SQuAD pairs.
- **UltraChat** (ding-etal-2023-enhancing) — general-purpose, multi-turn instruction-following dialogue with no notion of answerability at all. We sample 1,000 (prompt, response) pairs to instantiate \( \mathcal{D}_{\text{gu}} \), the pool that anchors ordinary chat behaviour untouched by the forget signal.

KUQ and SQuAD give OU two structurally distinct answerability conditions (no-context vs. context-grounded) from which to mine overcommitment, while UltraChat contributes a domain with no abstention semantics at all, so that the retain loss protects general instruction-following independently of the abstention decision. This diversity is what lets a single shared subspace \( V_l \) (Section 3.2) be tested for whether it captures a general overcommitment direction rather than a KUQ- or SQuAD-specific artifact.

**How the training pools are sampled.** \( \mathcal{D}_{\text{oc}} \) is not curated a priori — it is *mined* from each model's own behaviour (Section 3.1). For each of KUQ and SQuAD, we run each frozen backbone under greedy decoding on the unanswerable pool and label every completion COMMIT/ABSTAIN with an automatic LLM judge (Cerebras-hosted `gpt-oss-120b`). COMMIT-labelled completions are kept, one dataset-model pair capped at 1,000 examples, giving \( |\mathcal{D}_{\text{oc}}| = 2{,}000 \) per model (1,000 KUQ + 1,000 SQuAD). \( \mathcal{D}_{\text{lc}} \) uses the answerable KUQ/SQuAD splits paired with their gold answers (1,000 each), and \( \mathcal{D}_{\text{gu}} \) uses the 1,000 sampled UltraChat pairs — giving \( |\mathcal{D}_{\text{lc}} \cup \mathcal{D}_{\text{gu}}| = 3{,}000 \) retain examples per model. The same mining and sampling procedure is repeated independently for every backbone in Section 4.3, so \( \mathcal{D}_{\text{oc}} \) reflects each model's own overcommitment tendencies rather than a shared, model-agnostic forget set.

| Pool | Dataset | Role | Answerability | # examples |
|---|---|---|---|---|
| \( \mathcal{D}_{\text{oc}} \) | KUQ | forget (mined, COMMIT-labelled) | no-context | 1,000 |
| \( \mathcal{D}_{\text{oc}} \) | SQuAD 2.0 | forget (mined, COMMIT-labelled) | context-grounded | 1,000 |
| \( \mathcal{D}_{\text{lc}} \) | KUQ | retain (gold answers) | no-context, answerable | 1,000 |
| \( \mathcal{D}_{\text{lc}} \) | SQuAD 2.0 | retain (gold answers) | context-grounded, answerable | 1,000 |
| \( \mathcal{D}_{\text{gu}} \) | UltraChat | retain (general utility) | n/a | 1,000 |

### 4.2 Out-of-Distribution Evaluation

To test whether OU unlearns a general overcommitment *behaviour* rather than memorising KUQ- or SQuAD-specific phrasing, we evaluate on three additional benchmarks that are never seen during mining, subspace estimation, anchor construction, or adapter training. Each is mapped to one of the two trained answerability conditions so that it is probed with the matching prompt template and abstention reference, without any retraining or adapter change:

- **SelfAware** (yin-etal-2023-large) — no-context questions probing models' awareness of their own knowledge limits, sourced from different community Q&A data than KUQ and constructed around intrinsic unanswerability rather than KUQ's mix of ambiguity/false-premise/controversy categories.
- **FaithEval** (ming2024faitheval) — context-grounded questions where the passage is constructed to be unfaithful to (rather than merely silent on) the question, a different failure mode than SQuAD's "answer not present" unanswerability.
- **NoMIRACL** (thakur2024nomiracl) — context-grounded, retrieval-augmented QA where the retrieved passages may not support an answer, simulating a realistic RAG setting rather than the curated single-passage construction of SQuAD.

These three sets are diverse along two axes at once: they preserve the same two structural answerability conditions used in training (no-context vs. context-grounded), so success on them is attributable to the *decision* OU trains rather than to a new task format, while each is built from an independent source and a distinct mechanism of unanswerability (self-knowledge limits, context unfaithfulness, retrieval failure) that none of KUQ/SQuAD/UltraChat instantiates. Generalising across all three is therefore evidence that OU suppresses overcommitment as a behaviour, not as a memorised response to a fixed set of prompts.

| Dataset | Trained domain it maps to | Unanswerability mechanism | # examples |
|---|---|---|---|
| SelfAware | KUQ (no-context) | intrinsic knowledge limits | 1,000 |
| FaithEval | SQuAD (context-grounded) | context unfaithfulness | 1,000 |
| NoMIRACL | SQuAD (context-grounded) | retrieval non-support | 2,000 |

### 4.3 Models

We evaluate OU on three instruction-tuned backbones chosen to vary along axes that are orthogonal to the method itself — architecture family, scale, and routing style — so that generalisation of the method is not confounded with any single design choice:

- **Qwen 3.5 9B Instruct** — dense decoder, 32 transformer layers, standard split attention/MLP projections (\(q,k,v,o\)-proj, \(\text{gate},\text{up},\text{down}\)-proj).
- **Ministral-3-14B Instruct** — dense decoder from a different model family (Mistral), 40 transformer layers, larger than Qwen 3.5 9B by parameter count while remaining a standard dense architecture.
- **GPT-OSS-20B** — sparse Mixture-of-Experts decoder (OpenAI), 24 transformer layers, hidden size 2880, 32 experts with top-4 routing, and a *reasoning* model whose hidden analysis (chain-of-thought) channel cannot be disabled. Its expert feed-forward blocks are fused 3-D parameter tensors rather than `nn.Linear` modules, so the adapter is routed through the experts and the router gate instead of through attention (Section 4.4).

The three backbones therefore differ in (i) model family (Qwen / Mistral / OpenAI), (ii) parameter count and depth (9B/32L, 14B/40L, 20B/24L), (iii) dense vs. sparse-MoE computation, and (iv) whether generation includes a hidden reasoning channel. Testing OU across all three is a direct test of whether "overcommitment lives in a low-rank subspace of the late-layer residual stream" is a property of transformer-based LLMs in general, rather than an artifact of one attention/MLP implementation or one training recipe.

| Model | Family | Params | Layers | Routing | LoRA surface |
|---|---|---|---|---|---|
| Qwen 3.5 9B Instruct | Qwen | 9B | 32 | dense | attention + MLP proj. |
| Ministral-3-14B Instruct | Mistral | 14B | 40 | dense | attention + MLP proj. |
| GPT-OSS-20B | OpenAI | 20B (32 experts, top-4) | 24 | sparse MoE | fused expert proj. + router |

### 4.4 Framework and Implementation Settings

**Subspace estimation.** For every backbone, \( V_l \) is estimated once over the last 25% of transformer layers (Qwen: layers 24–31; Ministral-14B: layers 30–39; GPT-OSS: layers 18–23) with rank \( k=32 \). The generalised eigenproblem of Section 3.2 is solved with a ridge term added to \( \Sigma_{\text{gu}} \) to keep it invertible; the ridge coefficient is \( 10^{-3} \) by default, overridden per model when the default over-whitens the overcommitment-vs-legitimate-commitment contrast (GPT-OSS: 10.0; Ministral-14B: 1.0).

**Adapter.** OU trains a LoRA adapter with rank 16, \( \alpha = 32 \), dropout 0.05, applied to all attention and MLP projections for the two dense backbones. For the MoE backbone (GPT-OSS), attention is left un-adapted and LoRA instead targets the fused expert projections (`experts.gate_up_proj`, `experts.down_proj`) and the router weight, so the intervention is forced through the same computation path that performs expert selection rather than being absorbed by attention alone.

**Optimisation.** AdamW, learning rate \( 3\times10^{-5} \) with cosine decay and 3% warmup, 3 epochs, batch size 4 for forget and 4 for retain with gradient accumulation \( \times 4 \), max gradient norm 1.0. Training uses early stopping on a smoothed total loss (window 5, no minimum-improvement threshold) and exports the best-loss checkpoint. \( \lambda \) is set to 2.0 for Qwen and Ministral-14B and 1.0 for GPT-OSS. Each run trains for 375 optimisation steps over the 2,000 forget / 3,000 retain examples described in Section 4.1.

**Judge.** All COMMIT/ABSTAIN labelling in Section 4.1 uses the same automatic judge, a Cerebras-hosted `gpt-oss-120b`, so labelling is consistent across backbones and datasets.

### 4.5 Results

<!-- TODO: results tables and discussion (decision accuracy / false-commitment / false-abstention on
     KUQ, SQuAD, SelfAware, FaithEval, NoMIRACL; UltraChat perplexity ratio; collateral effects on
     hallucination/misinformation benchmarks) -->

---

<!-- ============================================================ -->
<!-- TODO: Section 5 — Conclusion                                 -->
<!-- ============================================================ -->
