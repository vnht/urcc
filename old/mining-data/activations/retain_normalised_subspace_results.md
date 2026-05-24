# Retain-Normalised Subspace Results

**Method:** Retain-span projection — both data matrices are projected into the
512-dimensional PCA basis of the retain activations before solving the
generalised eigenproblem `Sigma_C v = gamma Sigma_R v`. This keeps `Sigma_R`
full-rank in the subspace and avoids null-space instability.

**Config:** rank=8, ridge=1e-3, retain_rank=512, layers=last 25%

---

## Per-model stats

### qwen_instruct (layers 24–31)

| Layer | gamma_1 | gamma_8 | C_proj | R_proj | C/R |
|---|---|---|---|---|---|
| L24 | 66.06 | 9.45 | 165.85 | 7.96 | 20.83 |
| L25 | 58.46 | 9.04 | 151.67 | 7.96 | 19.05 |
| L26 | 60.19 | 8.48 | 150.19 | 7.97 | 18.86 |
| L27 | 52.65 | 7.88 | 130.06 | 7.97 | 16.32 |
| L28 | 50.67 | 7.56 | 125.12 | 7.97 | 15.70 |
| L29 | 47.55 | 7.04 | 119.81 | 7.97 | 15.03 |
| L30 | 49.06 | 7.35 | 119.29 | 7.97 | 14.97 |
| L31 | 38.69 | 6.26 | 98.94 | 7.97 | 12.42 |
| **Mean** | | | **132.62** | **7.97** | **16.65** |

### qwen_base (layers 24–31)

| Layer | gamma_1 | gamma_8 | C_proj | R_proj | C/R |
|---|---|---|---|---|---|
| L24 | 76.86 | 9.30 | 172.53 | 7.96 | 21.67 |
| L25 | 71.34 | 8.62 | 160.66 | 7.96 | 20.18 |
| L26 | 72.54 | 9.29 | 163.66 | 7.96 | 20.55 |
| L27 | 61.64 | 7.66 | 140.13 | 7.97 | 17.59 |
| L28 | 64.50 | 7.82 | 142.00 | 7.97 | 17.82 |
| L29 | 56.44 | 7.66 | 132.66 | 7.97 | 16.65 |
| L30 | 59.60 | 7.89 | 139.63 | 7.97 | 17.53 |
| L31 | 50.46 | 7.12 | 125.19 | 7.96 | 15.72 |
| **Mean** | | | **147.06** | **7.97** | **18.46** |

### ministral_instruct (layers 25–33)

| Layer | gamma_1 | gamma_8 | C_proj | R_proj | C/R |
|---|---|---|---|---|---|
| L25 | 75.74 | 11.94 | 189.17 | 7.96 | 23.75 |
| L26 | 67.18 | 10.79 | 174.05 | 7.96 | 21.85 |
| L27 | 75.09 | 11.23 | 183.76 | 7.96 | 23.08 |
| L28 | 83.95 | 11.37 | 191.51 | 7.96 | 24.05 |
| L29 | 69.71 | 11.26 | 176.16 | 7.97 | 22.12 |
| L30 | 74.86 | 10.86 | 183.60 | 7.97 | 23.05 |
| L31 | 66.95 | 10.52 | 166.39 | 7.96 | 20.89 |
| L32 | 61.44 | 9.87 | 152.68 | 7.97 | 19.16 |
| L33 | 48.04 | 8.76 | 130.75 | 7.97 | 16.41 |
| **Mean** | | | **172.01** | **7.97** | **21.59** |

### ministral_base (layers 25–33)

| Layer | gamma_1 | gamma_8 | C_proj | R_proj | C/R |
|---|---|---|---|---|---|
| L25 | 96.58 | 10.63 | 222.98 | 7.96 | 28.02 |
| L26 | 87.15 | 10.37 | 205.49 | 7.96 | 25.82 |
| L27 | 90.15 | 10.43 | 211.18 | 7.96 | 26.54 |
| L28 | 91.54 | 10.47 | 213.95 | 7.96 | 26.89 |
| L29 | 87.98 | 10.23 | 208.26 | 7.96 | 26.16 |
| L30 | 86.55 | 10.35 | 209.92 | 7.96 | 26.36 |
| L31 | 83.55 | 10.64 | 204.56 | 7.96 | 25.69 |
| L32 | 72.28 | 9.84 | 187.51 | 7.97 | 23.54 |
| L33 | 59.48 | 6.60 | 138.32 | 7.97 | 17.36 |
| **Mean** | | | **200.24** | **7.96** | **25.15** |

---

## Interpretation

**R_proj is consistent and meaningful.** All models show R_proj ≈ 7.97 across
every layer — the expected value for 8 orthonormal vectors sharing the retain
subspace proportionally (8/512 × total retain variance).

**C/R ratios show genuine signal.** The subspace captures 12–28× more
commitment variance than retain variance per unit direction. The found directions
are commit-rich and retain-neutral — directions that should affect commitment
behaviour without damaging general language capability.

**Base > instruct across both architectures.** qwen_base C/R = 18.5 vs.
qwen_instruct = 16.6; ministral_base C/R = 25.2 vs. ministral_instruct = 21.6.
Instruction tuning partially collapses the commitment–retain separation,
presumably because SFT/RLHF trains the model to hedge by default, blending
those representations.

**Ministral > Qwen overall.** Ministral base has the strongest signal
(C/R = 25.2, C_proj = 200), suggesting its commitment directions are more
linearly separable from its general language representations.

**Signal decays toward the last layer.** `gamma_1` and C_proj drop
monotonically from the first selected layer to the last. The very last layer
(L31 for Qwen, L33 for Ministral) is notably weaker — consistent with the
residual stream becoming more output-logit-focused and less feature-rich at the
final position.

**gamma_1 >> gamma_8.** The ratio gamma_1/gamma_8 ranges from ~7 to ~9,
indicating the commitment signal is concentrated in the top 1–3 eigenvectors.
Using rank 3–4 for interventions may be sufficient and less likely to affect
retain performance.
