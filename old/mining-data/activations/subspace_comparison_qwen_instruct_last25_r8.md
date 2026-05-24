# Subspace comparison — qwen_instruct (last-25%, rank=8)

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
| L24 |       165.852 |     208.335 |         7.962 |       7.963 |    20.83 |  26.16 |
| L25 |       151.673 |     191.184 |         7.963 |       7.964 |    19.05 |  24.01 |
| L26 |       150.192 |     184.457 |         7.965 |       7.966 |    18.86 |  23.16 |
| L27 |       130.059 |     163.219 |         7.969 |       7.970 |    16.32 |  20.48 |
| L28 |       125.125 |     157.138 |         7.969 |       7.970 |    15.70 |  19.72 |
| L29 |       119.807 |     154.556 |         7.969 |       7.970 |    15.03 |  19.39 |
| L30 |       119.294 |     155.043 |         7.969 |       7.971 |    14.97 |  19.45 |
| L31 |        98.942 |     136.348 |         7.966 |       7.969 |    12.42 |  17.11 |

**Mean C_proj (clean) = 132.618**, **(raw) = 168.785**
Mean R_proj (clean) = 7.967, (raw) = 7.968
Mean C/R   (clean) = 16.65, (raw) = 21.18

## 4. Subspace alignment between cleaned and raw

Principal angles (cos θ) between the rank-`r` subspaces spanned by V_clean and V_raw at each layer. Values close to 1 = subspaces are nearly identical; values close to 0 = orthogonal.

| Layer | mean cos θ | min cos θ | max cos θ | top-8 cos θ |
|-------|-----------:|----------:|----------:|----------------------:|
| L24 | 0.886 | 0.480 | 0.997 | 0.997  0.991  0.982  0.979  0.939  0.912  0.810  0.480 |
| L25 | 0.911 | 0.596 | 0.998 | 0.998  0.991  0.984  0.976  0.951  0.931  0.861  0.596 |
| L26 | 0.848 | 0.151 | 0.999 | 0.999  0.995  0.989  0.976  0.954  0.920  0.800  0.151 |
| L27 | 0.923 | 0.677 | 0.996 | 0.996  0.991  0.974  0.966  0.963  0.940  0.875  0.677 |
| L28 | 0.875 | 0.528 | 0.995 | 0.995  0.989  0.985  0.961  0.943  0.900  0.701  0.528 |
| L29 | 0.910 | 0.713 | 0.995 | 0.995  0.989  0.975  0.967  0.954  0.915  0.773  0.713 |
| L30 | 0.828 | 0.165 | 0.994 | 0.994  0.989  0.988  0.964  0.910  0.869  0.743  0.165 |
| L31 | 0.757 | 0.136 | 0.995 | 0.995  0.969  0.939  0.908  0.830  0.730  0.551  0.136 |

## 5. Interpretation hints

- Eigenvalue decay sharper in **raw** (γ_1/γ_8 ratio: clean=6.71, raw=9.54)
- Commitment-vs-retain ratio: raw is **higher** (21.18 vs 16.65)
- Mean subspace overlap: 0.867  (1.0 = identical, 0.0 = orthogonal)

**Reading the diagnostic:**

- If **raw eigenvalues decay much more sharply** AND **C/R ratio is much higher** for raw than cleaned → the cleaning step was destroying signal.
- If **eigenvalues are similar** AND **C/R is similar** AND **subspace overlap is low (<0.7)** → cleaning isn't hurting magnitude but is pointing at different directions; may not matter much.
- If **eigenvalues are similar** AND **subspace overlap is high (>0.95)** → cleaning is essentially a no-op; the supported-answer subspace doesn't overlap with the unsupported-contrast subspace much in the first place.
- If **raw also has slow decay** with no clean elbow → commitment is genuinely high-rank; URC's low-rank framing is the wrong primitive.