# Solvability of the UOC Subspace Eigenproblem

This note proves that the generalised eigenproblem used to build the
discriminative subspace `V` (step 2) is **always solvable for any model**, with
a complete set of real eigenpairs, provided the regularisation the pipeline
already applies is in place.

## The problem

For each late layer we solve

$$
(\Sigma_{OC} - \Sigma_{LC})\, v = \gamma\, \Sigma_E\, v ,
$$

where

- $\Sigma_{OC}$ is the covariance of the over-commit contrasts,
- $\Sigma_{LC}$ is the covariance of the legit-commit contrasts,
- $\Sigma_E$ is the covariance of the general-utility activations.

Write $A := \Sigma_{OC} - \Sigma_{LC}$ and $B := \Sigma_E$. This is a
**generalised symmetric eigenproblem** $A v = \gamma B v$.

## Why it is not trivially solvable

A generalised eigenproblem $A v = \gamma B v$ is guaranteed to have a complete
set of real eigenpairs **iff $B$ is symmetric positive definite (SPD)**. The two
ways this can fail here:

1. **$B = \Sigma_E$ is singular.** A covariance estimated from $N$ samples in
   $\mathbb{R}^{D}$ has rank at most $N-1$. Since $N \approx 1000 < D = 4096$,
   $\Sigma_E$ is rank-deficient, hence only positive *semi*-definite. Then
   $B^{-1/2}$ does not exist and eigenvalues can diverge along the null space.
2. **$A$ is indefinite.** $A$ is a *difference* of covariances, so it has both
   positive and negative eigenvalues. This is fine — it does **not** threaten
   solvability — but it rules out naive Cholesky-on-$A$ approaches.

The pipeline removes failure mode (1) by construction (see below), which is what
makes the proof go through for any model.

## What the pipeline enforces (`step2_build_subspace/build_subspace.py`)

Two safeguards, both unconditional:

**(a) Project into the general-utility span.** Lines 76–85 take the top-$k$ right
singular vectors $W \in \mathbb{R}^{D\times k}$ of the centred set-E matrix, with

$$
k = \min(\texttt{retain\_basis\_rank},\, N_e - 1,\, D),
$$

and project all three covariances into that $k$-dimensional span. On this span
the restricted $\Sigma_E$ has no zero directions by construction.

**(b) Ridge the retain covariance.** Lines 92–93:

```python
ridge_scale = float(np.diag(Sigma_E).mean()) * ridge
Sigma_E_reg = Sigma_E + ridge_scale * np.eye(k, dtype=np.float64)
```

so the matrix actually used is $B := \Sigma_E + \epsilon I$ with
$\epsilon = \texttt{ridge\_scale} > 0$. (The scipy-free fallback
`_whitening_eigh` additionally clips $B$'s eigenvalues to $\ge 10^{-12}$,
line 60.)

## Lemma

Let $A \in \mathbb{R}^{k\times k}$ be real symmetric and let
$B = \Sigma_E + \epsilon I$ with $\Sigma_E \succeq 0$ symmetric and
$\epsilon > 0$. Then the generalised eigenproblem $A v = \gamma B v$ has $k$
real eigenvalues and a $B$-orthogonal basis of real eigenvectors.

### Proof

**Step 1 — $A$ is symmetric.** Each $\Sigma$ is a covariance, hence symmetric;
a difference of symmetric matrices is symmetric, so $A^\top = A$. This is an
algebraic property of the covariance estimator and holds for *any* model and
*any* data.

**Step 2 — $B$ is SPD.** For any $x \neq 0$,

$$
x^\top B x = x^\top \Sigma_E x + \epsilon\, \lVert x \rVert^2 \;\ge\; 0 + \epsilon \lVert x \rVert^2 \;>\; 0,
$$

using $\Sigma_E \succeq 0$. So $B \succ 0$ regardless of how rank-deficient
$\Sigma_E$ is. This is the crucial step: the ridge makes positive-definiteness
an **enforced** property, not a data-dependent hope.

**Step 3 — reduce to a standard symmetric problem.** Since $B \succ 0$ it has a
unique SPD square root $B^{1/2}$, which is invertible. Substituting
$w = B^{1/2} v$,

$$
A v = \gamma B v
\iff
\underbrace{B^{-1/2} A B^{-1/2}}_{=:M}\, w = \gamma\, w .
$$

**Step 4 — $M$ is symmetric.**

$$
M^\top = (B^{-1/2})^\top A^\top (B^{-1/2})^\top = B^{-1/2} A B^{-1/2} = M ,
$$

using $A^\top = A$ and $(B^{-1/2})^\top = B^{-1/2}$.

**Step 5 — apply the spectral theorem.** Every real symmetric
$M \in \mathbb{R}^{k\times k}$ has $k$ real eigenvalues
$\gamma_1 \ge \dots \ge \gamma_k$ and an orthonormal eigenbasis $\{w_i\}$.
Transforming back, $v_i = B^{-1/2} w_i$ are real eigenvectors of the original
problem, and they are $B$-orthogonal:

$$
v_i^\top B v_j = w_i^\top w_j = \delta_{ij}. \qquad \blacksquare
$$

## What this proves — and what it does not

- **Existence and computability for any model.** Independent of architecture,
  layer, or sample count, the regularised eigenproblem has a complete, real,
  well-defined solution. There is no model for which it fails to be solvable.
- **It does *not* prove usefulness.** Solvability $\neq$ separability. The lemma
  guarantees $r$ real eigenvectors with real $\gamma$; it does **not** guarantee
  $\gamma_1$ is large or that over-commit and legit-commit actually separate
  along $V$. That separation (e.g. $\gamma_1 = 27.5$ for Qwen at L28) is an
  empirical property of the model's representations, not a theorem.

## Practical condition

The proof needs $\epsilon > 0$ **strictly**. With $\epsilon = 0$ and singular
$\Sigma_E$ (which is the case whenever $N < D$), $B$ is only PSD, $B^{-1/2}$ does
not exist, and solvability is not guaranteed. The ridge term on lines 92–93 (and
the eigenvalue floor on line 60) are exactly the $\epsilon > 0$ that turns
"usually works" into "provably always works".

## Optimality: the solution is the best possible subspace *within the search space*

Solvability says a solution exists. A stronger statement is that the
eigenvectors are **optimal** for the objective — but the optimality is scoped,
and stating that scope precisely matters.

Two reductions happen before the eigenproblem is solved, and both restrict what
"optimal" means:

1. **Retain-span projection.** The solver does not work in $\mathbb{R}^{D}$. It
   first projects onto the top-$k$ right singular vectors
   $W \in \mathbb{R}^{D\times k}$ of the centred set-E matrix
   (`build_subspace.py` lines 76–85), with
   $k = \min(\texttt{retain\_basis\_rank},\, N_e - 1,\, D)$. All optimality
   below is over the **column span of $W$**, denoted $\mathcal{S} = \operatorname{range}(W)$.
   Directions lying (partly) in the discarded complement of $\mathcal{S}$ are
   **not** candidates.
2. **Ridge.** The matrix used on the right is the **regularised**
   $\tilde B = \Sigma_E + \epsilon I$, not $\Sigma_E$. The optimum is therefore
   for the pencil $(A, \tilde B)$, not $(A, \Sigma_E)$; for large $\epsilon$ the
   solution shifts toward the unweighted top directions of $A$.

Within $\mathcal{S}$, each eigenvalue is the **generalised Rayleigh quotient**

$$
R(v) = \frac{v^\top A\, v}{v^\top \tilde B\, v}
     = \frac{\operatorname{Var}_{OC}(v) - \operatorname{Var}_{LC}(v)}{\operatorname{Var}_E(v) + \epsilon\lVert v\rVert^2},
\qquad v \in \mathcal{S},
$$

the excess of over-commit over legit-commit variance along $v$, measured in
units of (regularised) general-utility variance.

**Courant–Fischer / Rayleigh–Ritz (restricted).** For the symmetric pencil
$(A, \tilde B)$ with $\tilde B \succ 0$, both restricted to $\mathcal{S}$, the
ordered eigenvalues $\gamma_1 \ge \dots \ge \gamma_k$ satisfy

$$
\gamma_1 = \max_{0 \neq v \in \mathcal{S}} R(v),
\qquad
\gamma_{j} = \max_{\substack{0 \neq v \in \mathcal{S} \\ v \perp_{\tilde B} v_1,\dots,v_{j-1}}} R(v),
$$

and the top-$r$ eigenvectors maximise the trace objective over $\mathcal{S}$:

$$
V_\star = \arg\max_{\substack{V^\top \tilde B V = I_r \\ \operatorname{range}(V) \subseteq \mathcal{S}}} \ \operatorname{tr}\!\big(V^\top A\, V\big).
$$

### Consequence (precise)

- $\gamma_1$ is the **largest achievable** discriminative ratio over every
  direction **in the retain span $\mathcal{S}$**, for the regularised pencil. It
  is the single most over-commit-specific direction *that the solver can
  reach* — not necessarily over all of $\mathbb{R}^{D}$.
- The top-$r$ subspace $V_\star$ is the $\tilde B$-orthonormal $r$-frame in
  $\mathcal{S}$ that captures the maximum total OC-minus-LC variance. No other
  rank-$r$ subspace **inside $\mathcal{S}$** scores higher on the (regularised)
  objective.

So the method is guaranteed to find the best discriminative direction **among
those expressible in the retain span**. This is a theorem and holds for any
model. The projection is deliberate (it removes set-E null-space directions that
are unidentifiable from the data), but it does mean optimality is **conditional
on $\mathcal{S}$ containing the informative direction**; a discriminative
direction orthogonal to the top-$k$ utility span would be missed.

It does **not** guarantee that a meaningful direction exists at all — that
depends on $A = \Sigma_{OC} - \Sigma_{LC}$ being nonzero in informative
directions of $\mathcal{S}$, which is an empirical property of the model
(certified after the fact by $\gamma_1$ and the projected cluster separation).
The provable claim is the sharp one: *constrained optimality is a theorem;
meaningfulness is a measurement.*
