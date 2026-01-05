---
title: "rethinking model complexity in overparameterized models"
date: 2025-01-04
draft: false
tags: [singular learning theory, bayesian inference, model selection]
math: true
---

## 1. why model complexity is not the number of parameters

model selection criteria often penalize models according to the number of parameters they use. this is reasonable when every parameter direction independently affects predictions. but many modern models violate this assumption: different parameter values can represent the same function. low-rank regression, overcomplete models, and neural networks with symmetries all have redundant coordinates.

a simple thought experiment makes this issue clear. consider two models that represent the same family of functions:

- model a uses a minimal coordinate system
- model b uses an inflated coordinate system with redundant directions

functionally, these models are identical. yet parameter-count penalties will prefer model a, appealing to occam's razor. this mismatch motivates a basic question:

**what does bayesian evidence actually count when parameters are redundant?**

---

## 2. where the bic penalty comes from and where the assumption enters

before diving into formulas, let's build intuition for why curvature matters at all. imagine standing in a valley:

- in some directions the ground slopes steeply upward
- in others it is almost flat

if we only know our location up to measurement noise, steep directions pin us down precisely, while flat directions leave us uncertain. bayesian evidence is, at its core, a calculation of how much volume of plausible parameter space remains after seeing data.

curvature determines how fast this volume shrinks as we collect more data.

we observe a dataset
\[
D_n = \{(x_i, y_i)\}_{i=1}^n
\[
of size $n$. a model is parameterized by $\theta \in \mathbb{R}^d$ with prior distribution $\pi(\theta)$.

the likelihood is
\[
p(D_n \mid \theta) = \prod_{i=1}^n p(y_i \mid x_i, \theta),
\[
and the marginal likelihood (evidence) is
\[
p(D_n) = \int p(D_n \mid \theta)\, \pi(\theta)\, d\theta.
\[

throughout this post, we consider the asymptotic regime where the sample size $n$ grows while the model class is fixed.

geometrically, it is useful to distinguish two kinds of directions in parameter space:

- **curved directions**: small changes in $\theta$ lead to quadratic changes in the loss or log-likelihood
- **flat directions**: changes in $\theta$ leave the model's predictions unchanged (or only change them at higher order)

the central theme is that bayesian evidence is sensitive to curved directions, not the raw number of parameters.

for large sample size $n$, the marginal likelihood is approximated using laplace's method. the key steps are:

- the posterior concentrates near a maximizer $\hat{\theta}$
- near $\hat{\theta}$, the log posterior is approximated by a quadratic
- integrating a gaussian in $d$ dimensions yields a volume factor

writing $H_n$ for the hessian of the negative log posterior at $\hat{\theta}$,
\[
\log p(D_n)
\approx
\log p(D_n \mid \hat{\theta})
- \tfrac{1}{2} \log \det H_n + O(1).
\[

at this point, it is useful to introduce the fisher information. it measures how sensitive the likelihood is to small parameter changes:

- large fisher information means predictions change rapidly
- small fisher information means predictions barely change

in regular models, the fisher information matrix is full rank. mathematically,
\[
H_n \approx n\, I(\theta^\star),
\[
where $I(\theta^\star)$ is the fisher information at a kl minimizer.

if $I(\theta^\star)$ is full rank with dimension $d$,
\[
\log \det H_n \sim d \log n,
\[
leading to
\[
\log p(D_n)
\approx
\log p(D_n \mid \hat{\theta})
- \frac{d}{2} \log n + O(1),
\[
which is the bayesian information criterion (bic).

crucially, this derivation assumes every parameter direction contributes curvature. if some directions are flat, this scaling breaks.

---

## 3. what goes wrong in overparameterized models

consider fitting a straight line using two parameterizations:

- a minimal one with slope and intercept
- an inflated one using five constrained parameters

both describe the same functions, but the second has directions where moving does nothing. these directions have zero fisher information; the data cannot constrain them.

more generally, many models have likelihoods that are flat along some directions. laplace's method treats all directions as curved, overestimating posterior contraction and over-penalizing the model.

this error scales as $\log n$, so it diverges with more data. to see this explicitly, we turn to a minimal example.

---

## 4. a minimal example: rank-deficient linear regression

consider the linear–gaussian model
\[
y_i = x_i^\top B\theta + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal N(0, \sigma^2),
\[
with prior $\theta \sim \mathcal N(0, \tau^2 I_d)$.

here:

- $x_i \in \mathbb{R}^p$
- $B \in \mathbb{R}^{p \times d}$
- $\theta \in \mathbb{R}^d$
- $\varepsilon_i$ is gaussian noise
- $I_d$ is the identity matrix
- $\tau^2$ controls prior scale

if $\mathrm{rank}(B) = r < d$, only an $r$-dimensional projection of $\theta$ affects predictions.

### 4.1 exact marginal likelihood

because the model is gaussian, the marginal likelihood is exact. writing $A_n = X_n B$ and $S_n = A_n^\top A_n$,
\[
\log p(D_n)
=
-\tfrac{1}{2}\Big(
n\log(2\pi)
+ n\log \sigma^2
+ \log \det(I + \alpha S_n)
+ \text{data-fit terms}
\Big),
\[
with $\alpha = \tau^2/\sigma^2$.

the spectrum of $S_n$ has:

- $r$ eigenvalues scaling as $n$
- $d-r$ eigenvalues remaining $O(1)$

### 4.2 effective dimension from eigenvalues

\[
\log \det(I + \alpha S_n)
= r \log n + O(1),
\[
so
\[
\log p(D_n)
=
\log p(D_n \mid \theta^\star)
- \frac{r}{2} \log n + O(1).
\[

### 4.3 quantifying the bic error

bic predicts a penalty of $\frac{d}{2} \log n$. the error is
\[
\Big(\frac{d}{2} - \frac{r}{2}\Big)\log n + O(1),
\[
which diverges as $n$ grows.

---

**bayesian evidence penalizes the number of directions that actually change the model, not the number of parameters used to describe it.**

---

## 5. a primer on singular learning theory (slt)

laplace's method and bic assume the model is *regular*: the fisher information is full rank, the posterior concentrates at a unique point, and every parameter direction contributes curvature. many modern models violate this.

mixture models, neural networks, and low-rank representations all exhibit redundant parameters and flat directions in the likelihood. in such *singular* models, the set of kl minimizers
\[
\{\theta : \mathrm{KL}(p^* \| p_\theta) = 0\}
\[
can have nontrivial geometry—curves, surfaces, or more complex singularities—rather than being isolated points.

singular learning theory (slt), developed by watanabe, shows that in these models the marginal likelihood is governed by the *real log canonical threshold* (rlct) $\lambda$, a birational invariant that measures the local singularity structure:
\[
\log p(D_n) = \log p(D_n \mid \theta^*) - \lambda \log n + (m-1)\log\log n + O(1),
\[
where $m \in \mathbb{N}$ is the multiplicity. for regular models, $\lambda = d/2$ and $m = 1$, recovering bic. for singular models, typically $\lambda < d/2$.

the key insight is that effective complexity depends on the intrinsic geometry of the likelihood—not on the number of coordinates used to describe it.

---

## 6. effective dimension and the real log canonical threshold (rlct)

slt shows that
\[
\log p(D_n)
=
\log p(D_n \mid \theta^\star)
- \lambda \log n + O(\log \log n).
\[

$\lambda$, the rlct, is a coordinate-free effective dimension.

in regular models, $\lambda = d/2$. in singular models, $\lambda < d/2$.

for rank-deficient regression, $\lambda = r/2$.

---

## 7. representation invariance

rlct depends only on the function class, not the parameterization. bic does not satisfy this invariance.

overcomplete representations introduce near-zero curvature directions without changing the intrinsic subspace. both parameterizations span the same subspace, but the overcomplete model introduces additional near-zero eigenvalues corresponding to redundant coordinates.

---

## 8. why this matters

model complexity is a geometric property of the likelihood, not a bookkeeping exercise over parameters. slt provides the correct language for understanding generalization in overparameterized models.

---

## appendix a: exact marginal likelihood and evidence slopes in linear–gaussian models

this appendix records the exact marginal likelihood calculation underlying section 4, and explains why the leading $\log n$ term depends on intrinsic rank rather than parameter count.

we consider the linear–gaussian regression model
\[
y_i = x_i^\top B\theta + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
\[
with prior $\theta \sim \mathcal{N}(0, \tau^2 I_d)$. writing $X_n \in \mathbb{R}^{n \times p}$ for the design matrix and
\[
A_n := X_n B \in \mathbb{R}^{n \times d},
\[
the likelihood can be written compactly as
\[
p(y \mid \theta) = \mathcal{N}(y \mid A_n \theta, \sigma^2 I_n).
\[

because both likelihood and prior are gaussian, the marginal likelihood can be computed in closed form by integrating out $\theta$. a standard gaussian completion of the square yields
\[
p(y) = \mathcal{N}(y \mid 0, \sigma^2 I_n + \tau^2 A_n A_n^\top).
\[

taking logarithms, the log evidence is
\[
\log p(D_n) = -\frac{1}{2}\Big(n \log(2\pi) + n \log \sigma^2 + \log \det(I_d + \alpha S_n) + \text{data-fit terms}\Big),
\[
where
\[
S_n := A_n^\top A_n, \quad \alpha := \tau^2 / \sigma^2.
\[

the key object controlling the complexity penalty is the gram matrix $S_n$.

### spectrum and intrinsic rank

assume the rows $x_i$ are i.i.d. with nondegenerate covariance $\Sigma_x$, and that $\mathrm{rank}(B) = r < d$. then
\[
\frac{1}{n} S_n = \frac{1}{n} B^\top X_n^\top X_n B \longrightarrow B^\top \Sigma_x B \quad \text{almost surely}.
\[

as a consequence:

- $r$ eigenvalues of $S_n$ scale linearly with $n$
- the remaining $d - r$ eigenvalues remain $O(1)$

taking determinants,
\[
\log \det(I_d + \alpha S_n) = \sum_{j=1}^{r} \log(\alpha n \lambda_j + 1) + O(1) = r \log n + O(1).
\[

substituting back, the leading asymptotic form of the evidence is
\[
\log p(D_n) = \log p(D_n \mid \theta^\star) - \frac{r}{2} \log n + O(1).
\[

this shows explicitly that the effective dimension of the model is $r$, not the ambient parameter count $d$. in the language of singular learning theory, the real log canonical threshold (rlct) is
\[
\lambda = r/2.
\[

by contrast, laplace's approximation (and bic) always insert $d/2$ in place of $\lambda$, leading to a systematic error of
\[
\Big(\frac{d}{2} - \frac{r}{2}\Big) \log n
\[
in singular models. this is the precise sense in which bic "over-penalizes" overparameterized representations.

---

## further reading

- kalyaan rao, "evidence slopes and effective dimension in singular linear models," preprint
- sumio watanabe, *algebraic geometry and statistical learning theory*, cambridge university press, 2009
- sumio watanabe, *mathematical theory of bayesian statistics*, chapman and hall/crc, 2018
- gideon schwarz, "estimating the dimension of a model," *the annals of statistics*, 1978

