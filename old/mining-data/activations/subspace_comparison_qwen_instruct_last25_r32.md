# Subspace comparison — qwen_instruct (last-25%, rank=32)

Cleaned: variance from `c_unsupported_clean` (post supported-PCA subtraction).
Raw:     variance from `c_unsupported` (no cleaning step).

## 1. Top eigenvalues per layer (γ = vᵀΣ_C v / vᵀΣ_R v)

### Cleaned

```
L24:   66.065    26.808    17.498    13.335    12.050    10.910     9.738     9.447
L25:   58.458    25.243    15.652    12.363    10.579    10.383     9.958     9.038
L26:   60.188    23.909    15.284    12.180    10.458    10.210     9.479     8.484
L27:   52.654    18.984    13.711    10.333     9.966     8.424     8.106     7.880
L28:   50.667    17.913    13.041    10.039     9.449     8.359     8.097     7.559
L29:   47.554    16.865    12.857     9.789     9.006     8.655     8.036     7.045
L30:   49.064    15.781    12.645     9.613     8.946     8.278     7.619     7.350
L31:   38.695    14.527    10.652     8.055     7.540     6.722     6.490     6.261
```

### Raw

```
L24:  100.347    30.691    18.123    13.992    12.833    11.440    10.831    10.079
L25:   91.561    28.341    16.154    12.822    11.465    10.936    10.231     9.674
L26:   87.924    26.111    16.001    12.514    11.454    11.033    10.144     9.277
L27:   79.542    21.268    14.327    10.910    10.588     9.692     8.507     8.385
L28:   76.796    19.581    13.600    10.650    10.263    10.049     8.278     7.921
L29:   75.692    18.218    13.606    10.415    10.082     9.901     8.484     8.156
L30:   77.558    17.350    13.362    11.094    10.114     9.199     8.481     7.885
L31:   63.178    17.201    13.365    11.053     8.716     8.438     7.384     7.013
```

## 2. Cumulative commitment variance captured (raw)

Computed from the full generalised-eigenvalue spectrum, projecting into the retain span (k_eff dims) per layer. Cleaned bundle stores only top-rank eigenvalues so a full comparison is not possible — these numbers come from the raw subspace.

| Layer | rank-1 | rank-8 | rank-16 | rank-32 | rank-64 | rank-128 |
|-------|-------:|-------:|--------:|--------:|--------:|---------:|
| L24 |  13.7% |  28.4% |  37.0% |  48.7% |  63.4% |  80.0% |
| L25 |  13.4% |  27.9% |  36.6% |  48.4% |  63.1% |  79.9% |
| L26 |  13.1% |  27.4% |  36.1% |  47.9% |  62.8% |  79.7% |
| L27 |  12.4% |  25.5% |  33.7% |  45.0% |  59.8% |  77.2% |
| L28 |  12.2% |  25.0% |  33.2% |  44.5% |  59.4% |  76.9% |
| L29 |  12.4% |  25.2% |  33.5% |  44.8% |  59.6% |  77.0% |
| L30 |  12.8% |  25.6% |  33.9% |  45.0% |  59.6% |  77.0% |
| L31 |  12.0% |  26.0% |  34.2% |  45.0% |  59.6% |  77.2% |

## 3. Commitment vs retain projection on top-rank subspace

`C_proj = tr(VᵀΣ_C V)` — commitment energy captured by the subspace
`R_proj = tr(VᵀΣ_R V)` — retain energy captured (lower is better)
`C/R`   = signal-to-interference ratio

| Layer | C_proj (clean) | C_proj (raw) | R_proj (clean) | R_proj (raw) | C/R clean | C/R raw |
|-------|---------------:|-------------:|---------------:|-------------:|----------:|--------:|
| L24 |       305.942 |     356.629 |        31.845 |      31.847 |     9.61 |  11.20 |
| L25 |       283.533 |     331.254 |        31.851 |      31.853 |     8.90 |  10.40 |
| L26 |       280.536 |     322.745 |        31.857 |      31.859 |     8.81 |  10.13 |
| L27 |       249.168 |     288.191 |        31.870 |      31.873 |     7.82 |   9.04 |
| L28 |       242.133 |     279.709 |        31.870 |      31.873 |     7.60 |   8.78 |
| L29 |       234.122 |     274.463 |        31.870 |      31.873 |     7.35 |   8.61 |
| L30 |       230.761 |     272.331 |        31.871 |      31.874 |     7.24 |   8.54 |
| L31 |       192.046 |     236.017 |        31.854 |      31.860 |     6.03 |   7.41 |

**Mean C_proj (clean) = 252.280**, **(raw) = 295.167**
Mean R_proj (clean) = 31.861, (raw) = 31.864
Mean C/R   (clean) = 7.92, (raw) = 9.26

## 4. Subspace alignment between cleaned and raw

Principal angles (cos θ) between the rank-`r` subspaces spanned by V_clean and V_raw at each layer. Values close to 1 = subspaces are nearly identical; values close to 0 = orthogonal.

| Layer | mean cos θ | min cos θ | max cos θ | top-8 cos θ |
|-------|-----------:|----------:|----------:|----------------------:|
| L24 | 0.929 | 0.078 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L25 | 0.941 | 0.337 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L26 | 0.929 | 0.236 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L27 | 0.947 | 0.430 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L28 | 0.938 | 0.502 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L29 | 0.934 | 0.239 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L30 | 0.922 | 0.103 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |
| L31 | 0.897 | 0.165 | 1.000 | 1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000 |

## 5. Interpretation hints

- Eigenvalue decay sharper in **raw** (γ_1/γ_32 ratio: clean=14.39, raw=21.62)
- Commitment-vs-retain ratio: raw is **higher** (9.26 vs 7.92)
- Mean subspace overlap: 0.930  (1.0 = identical, 0.0 = orthogonal)

**Reading the diagnostic:**

- If **raw eigenvalues decay much more sharply** AND **C/R ratio is much higher** for raw than cleaned → the cleaning step was destroying signal.
- If **eigenvalues are similar** AND **C/R is similar** AND **subspace overlap is low (<0.7)** → cleaning isn't hurting magnitude but is pointing at different directions; may not matter much.
- If **eigenvalues are similar** AND **subspace overlap is high (>0.95)** → cleaning is essentially a no-op; the supported-answer subspace doesn't overlap with the unsupported-contrast subspace much in the first place.
- If **raw also has slow decay** with no clean elbow → commitment is genuinely high-rank; URC's low-rank framing is the wrong primitive.