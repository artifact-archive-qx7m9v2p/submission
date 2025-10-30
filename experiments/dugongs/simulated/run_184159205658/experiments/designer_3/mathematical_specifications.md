# Mathematical Specifications for Designer 3 Models

## Precise Distributional Specifications

---

## Model 1: B-Spline Regression with Hierarchical Shrinkage

### Full Hierarchical Model

**Sampling Distribution:**
```
Y_i | mu_i, sigma ~ Normal(mu_i, sigma^2)  for i = 1, ..., 27
```

**Mean Structure:**
```
mu_i = beta_0 + sum_{k=1}^9 beta_k * B_k(x_i)
```

**Basis Functions:**
```
B_k(x) = cubic B-spline basis functions (degree 3)
Knots: tau_1, ..., tau_5 at quantiles [0.2, 0.4, 0.6, 0.8, 0.95] of observed x
Boundary knots: [min(x), max(x)] = [1.0, 31.5]
Total basis functions: K = 9
```

**Prior Hierarchy:**
```
beta_0 ~ Normal(2.3, 0.5^2)
beta_k | tau_k ~ Normal(0, tau_k^2)  for k = 1, ..., 9
tau_k | tau_global ~ Half-Cauchy(0, 1) * tau_global
tau_global ~ Half-Cauchy(0, 0.2)
sigma ~ Half-Normal(0, 0.3^2)
```

**Half-Cauchy Density:**
```
p(tau | scale) = (2 / (pi * scale)) * (1 / (1 + (tau/scale)^2))  for tau > 0
```

**Half-Normal Density:**
```
p(sigma | scale) = sqrt(2/pi) * (1/scale) * exp(-sigma^2 / (2*scale^2))  for sigma > 0
```

### Posterior Target

**Joint Posterior (up to proportionality):**
```
p(beta_0, beta, tau, tau_global, sigma | Y, X) ∝

  [Product_{i=1}^27 Normal(Y_i | mu_i, sigma^2)] *
  Normal(beta_0 | 2.3, 0.5^2) *
  [Product_{k=1}^9 Normal(beta_k | 0, (tau_k * tau_global)^2)] *
  [Product_{k=1}^9 Half-Cauchy(tau_k | 0, 1)] *
  Half-Cauchy(tau_global | 0, 0.2) *
  Half-Normal(sigma | 0, 0.3^2)

where mu_i = beta_0 + sum_{k=1}^9 beta_k * B_k(x_i)
```

### Effective Degrees of Freedom

**Expected Effective DF:**
```
df_eff = trace(S)
where S = B * (B'B + Lambda^{-1})^{-1} * B'
      Lambda = diag((tau_1 * tau_global)^2, ..., (tau_9 * tau_global)^2)
```

With strong shrinkage (tau_global small), df_eff << 9.
Expected: df_eff ~ 3-5 based on hierarchical prior structure.

---

## Model 2: Gaussian Process Regression

### Full Model Specification

**Sampling Distribution:**
```
Y | f, sigma ~ Normal(f, sigma^2 * I)
where f = [f(x_1), ..., f(x_27)]'
```

**Gaussian Process Prior:**
```
f | beta_0, eta^2, ell ~ MVN(m, K)
where:
  m = beta_0 * 1  (constant mean vector)
  K = covariance matrix with K_ij = k(x_i, x_j)
```

**Squared Exponential Covariance:**
```
k(x_i, x_j) = eta^2 * exp(-(x_i - x_j)^2 / (2 * ell^2))
```

**Hyperpriors:**
```
beta_0 ~ Normal(2.3, 0.3^2)
eta^2 ~ Half-Normal(0, 0.5^2)
ell ~ Inverse-Gamma(alpha=5, beta=10)
sigma ~ Half-Normal(0, 0.2^2)
```

**Inverse-Gamma Density:**
```
p(ell | alpha, beta) = (beta^alpha / Gamma(alpha)) * ell^{-(alpha+1)} * exp(-beta/ell)
```

For IG(5, 10):
- Mode = 10/(5+1) = 1.67
- Mean = 10/(5-1) = 2.5
- Variance = 10^2 / ((5-1)^2 * (5-2)) = 100/48 = 2.08

### Marginal Likelihood

**Integrating out f (Marginal GP):**
```
Y | beta_0, eta^2, ell, sigma ~ MVN(beta_0 * 1, K + sigma^2 * I)
```

**Log Marginal Likelihood:**
```
log p(Y | beta_0, eta^2, ell, sigma) =
  -0.5 * (Y - beta_0*1)' * (K + sigma^2*I)^{-1} * (Y - beta_0*1)
  -0.5 * log|K + sigma^2*I|
  -(27/2) * log(2*pi)
```

### Posterior Predictive Distribution

**For new x_star:**
```
Y_star | Y, x_star, theta ~ Normal(mu_star, sigma^2_star)

where:
  mu_star = beta_0 + k_star' * (K + sigma^2*I)^{-1} * (Y - beta_0*1)
  sigma^2_star = k_star_star - k_star' * (K + sigma^2*I)^{-1} * k_star + sigma^2

  k_star = [k(x_star, x_1), ..., k(x_star, x_27)]'
  k_star_star = k(x_star, x_star) = eta^2
  theta = (beta_0, eta^2, ell, sigma)
```

**Posterior Predictive with Parameter Uncertainty:**
```
p(Y_star | Y, x_star) = integral p(Y_star | theta, Y, x_star) * p(theta | Y) dtheta
```
Approximated by Monte Carlo: average over posterior samples of theta.

---

## Model 3: Horseshoe Polynomial Regression

### Full Hierarchical Model

**Standardization:**
```
x_std_i = (x_i - mean(x)) / sd(x)
where mean(x) = 10.94, sd(x) = 7.87
```

**Sampling Distribution:**
```
Y_i | mu_i, sigma ~ Normal(mu_i, sigma^2)
```

**Mean Structure (on standardized scale):**
```
mu_i = beta_0 + sum_{j=1}^6 beta_j * (x_std_i)^j
```

**Horseshoe Prior:**
```
beta_0 ~ Normal(2.3, 0.5^2)
beta_j | lambda_j, tau ~ Normal(0, (lambda_j * tau)^2)  for j = 1, ..., 6
lambda_j ~ Half-Cauchy(0, 1)  for j = 1, ..., 6
tau ~ Half-Cauchy(0, tau_0^2)
sigma ~ Half-Normal(0, 0.3^2)
```

**Global Shrinkage Parameter:**
```
tau_0 = (p_0 / (J - p_0)) * (sigma_y / sqrt(N))
     = (3 / (6 - 3)) * (0.28 / sqrt(27))
     = 1 * (0.28 / 5.196)
     = 0.054
```

where:
- p_0 = 3 (expected number of non-zero coefficients)
- J = 6 (maximum polynomial degree)
- sigma_y = 0.28 (observed standard deviation of Y)
- N = 27 (sample size)

### Joint Posterior

```
p(beta_0, beta, lambda, tau, sigma | Y, X) ∝

  [Product_{i=1}^27 Normal(Y_i | mu_i, sigma^2)] *
  Normal(beta_0 | 2.3, 0.5^2) *
  [Product_{j=1}^6 Normal(beta_j | 0, (lambda_j * tau)^2)] *
  [Product_{j=1}^6 Half-Cauchy(lambda_j | 0, 1)] *
  Half-Cauchy(tau | 0, 0.054) *
  Half-Normal(sigma | 0, 0.3^2)
```

### Posterior Shrinkage

**Effective Shrinkage Factor:**
```
kappa_j = (lambda_j * tau)^2 / ((lambda_j * tau)^2 + sigma^2 / sum_i (x_std_i^j)^2)
```

For coefficient j:
- If kappa_j ~ 0: strong shrinkage, beta_j ~ 0
- If kappa_j ~ 1: no shrinkage, beta_j escapes prior

**Posterior Mean (approximate):**
```
E[beta_j | Y] ≈ kappa_j * beta_j_MLE
```

### Back-Transformation to Original Scale

**To get interpretable coefficients on original x scale:**

For polynomial sum_{j=0}^J beta_j * x_std^j on standardized scale,
back-transform to sum_{j=0}^J gamma_j * x^j:

```
Define: a = mean(x), b = sd(x)
Then: x_std = (x - a) / b

(x - a) / b = sum_{j=0}^J beta_j * ((x - a) / b)^j

Expand using binomial theorem to get gamma_j in terms of beta_j, a, b.
```

**Simpler approach:** Report predictions on original x scale, don't interpret transformed coefficients.

---

## Comparative Properties

### Effective Model Complexity

**Spline:**
```
Complexity ~ df_eff = trace(S)
Expected: 3-5 (data determines via shrinkage)
```

**GP:**
```
Complexity ~ trace(S) where S = K * (K + sigma^2*I)^{-1}
Expected: 15-20 (relatively smooth but flexible)
Maximum: 27 (interpolation)
```

**Polynomial:**
```
Complexity ~ number of non-zero beta_j
Expected: 2-4 (horseshoe selects order)
Maximum: 6 (all terms used)
```

### Extrapolation Behavior

**Spline (x > 31.5):**
```
mu(x) = beta_0 + sum_{k=1}^9 beta_k * B_k(x)
Behavior: Linear extrapolation from last segment
Slope: Determined by rightmost active basis functions
```

**GP (x > 31.5):**
```
mu(x) → beta_0 as x → infinity
sigma^2(x) → eta^2 + sigma^2 as x → infinity
Behavior: Reverts to prior mean with high uncertainty
```

**Polynomial (x > 31.5):**
```
mu(x) = beta_0 + sum_{j=1}^6 beta_j * x_std^j
Behavior: Depends on sign and magnitude of highest non-zero term
  If beta_6 > 0: diverges to +infinity
  If beta_6 < 0: diverges to -infinity
  Highly unreliable for extrapolation
```

---

## Computational Considerations

### Spline

**Design Matrix:**
```
B = [B_1(x_1) ... B_9(x_1)]
    [   ...   ...    ...   ]
    [B_1(x_27) ... B_9(x_27)]
Dimension: 27 x 9
```

**Computation per MCMC iteration:** O(N * K + K^3) = O(27*9 + 9^3) = O(1000)

**Memory:** O(N*K) = O(243) floats

### GP

**Covariance Matrix:**
```
K = [k(x_1,x_1) ... k(x_1,x_27)]
    [    ...    ...     ...     ]
    [k(x_27,x_1) ... k(x_27,x_27)]
Dimension: 27 x 27
```

**Computation per MCMC iteration:** O(N^3) = O(19,683) for Cholesky decomposition

**Memory:** O(N^2) = O(729) floats

**Speed:** ~10x slower than spline per iteration

### Polynomial

**Design Matrix (on standardized x):**
```
X_poly = [x_std_1^1 ... x_std_1^6]
         [   ...    ...    ...    ]
         [x_std_27^1 ... x_std_27^6]
Dimension: 27 x 6
```

**Computation per MCMC iteration:** O(N * J + J^3) = O(27*6 + 6^3) = O(378)

**Memory:** O(N*J) = O(162) floats

**Speed:** Fastest of the three

---

## Prior Sensitivity Analysis Plan

### Spline

**Vary tau_global scale:** {0.1, 0.2, 0.4}
- Test: Do posteriors change substantially?
- Metric: KL divergence between posterior predictives

**Vary beta_0 scale:** {0.25, 0.5, 1.0}
- Test: Does intercept posterior change?
- Expect: Minimal change (data informative)

### GP

**Vary ell prior:**
- Alternative 1: ell ~ Half-Normal(0, 5^2)
- Alternative 2: ell ~ Inverse-Gamma(3, 5)
- Test: Posterior length scale stability

**Vary eta^2 prior:**
- Alternative: eta^2 ~ Inverse-Gamma(2, 0.5)
- Test: Signal variance stability

### Polynomial

**Vary tau_0:** {0.027, 0.054, 0.108} (0.5x, 1x, 2x)
- Test: Does degree selection change?
- Expect: Some sensitivity but same active terms

**Vary p_0:** {2, 3, 4}
- Test: Prior on sparsity level
- Expect: Minimal impact if data is informative

---

## Posterior Predictive Checks - Test Statistics

### Universal Checks (All Models)

**Location:**
```
T_1(Y) = mean(Y)
Observed: 2.32
Check: P(T_1(Y_rep) > T_1(Y_obs) | Y_obs)
Accept if: 0.05 < p-value < 0.95
```

**Scale:**
```
T_2(Y) = sd(Y)
Observed: 0.28
Check: P(T_2(Y_rep) > T_2(Y_obs) | Y_obs)
Accept if: 0.05 < p-value < 0.95
```

**Range:**
```
T_3(Y) = max(Y) - min(Y)
Observed: 2.63 - 1.71 = 0.92
```

**Skewness:**
```
T_4(Y) = E[(Y - mean(Y))^3] / sd(Y)^3
Observed: -0.88
```

### Model-Specific Checks

**Spline:**
```
T_5 = Number of sign changes in second derivative of fitted curve
Expect: 2-4 (moderate wiggliness)
Reject if: > 6 (over-wiggling)
```

**GP:**
```
T_6 = Correlation between Y values at replicated x points
Expect: High correlation at same x
```

**Polynomial:**
```
T_7 = Number of local extrema in fitted curve
Expect: 0-2 for low-degree polynomial
Reject if: > 3 (Runge phenomenon)
```

---

## Decision Thresholds (Quantitative)

### Convergence
```
R-hat < 1.01  (MANDATORY)
ESS_bulk > 400  (GOOD)
ESS_tail > 400  (EXCELLENT)
```

### Predictive Performance
```
LOO-ELPD difference > 2*SE  (MEANINGFUL)
Pareto-k < 0.5  (EXCELLENT)
Pareto-k < 0.7  (ACCEPTABLE)
Pareto-k > 0.7  (PROBLEMATIC)
```

### Prior-Posterior Overlap
```
KL(Prior || Posterior) > 2 nats  (Data dominates prior - GOOD)
KL(Prior || Posterior) < 0.5 nats  (Prior dominates data - BAD)
```

### Residual Diagnostics
```
Shapiro-Wilk p-value > 0.05  (Normality acceptable)
Breusch-Pagan p-value > 0.05  (Homoscedasticity acceptable)
Durbin-Watson in [1.5, 2.5]  (No autocorrelation)
```

---

## Implementation Notes

### Reparameterization Options

**Spline (non-centered):**
```
Original: beta_k ~ Normal(0, tau_k)
Non-centered:
  beta_k_raw ~ Normal(0, 1)
  beta_k = tau_k * beta_k_raw
```
Use if: Funnel geometry detected (divergences with low tau_k)

**GP (whitening):**
```
Original: f ~ MVN(m, K)
Whitened:
  f_white ~ Normal(0, 1)  independently
  f = m + L * f_white  where L*L' = K
```
Use if: Slow mixing in f parameters

**Polynomial (QR decomposition):**
```
Original: X_poly * beta
QR: X_poly = Q * R
Then: beta_tilde ~ priors
      beta = R^{-1} * beta_tilde
```
Use if: High posterior correlation between beta_j

### Software-Specific Syntax

**PyMC:**
```python
# Spline
mu = pm.math.dot(B_matrix, beta)

# GP
gp = pm.gp.Marginal(mean_func=beta_0, cov_func=cov)

# Polynomial
mu = beta_0 + pm.math.dot(X_poly, beta)
```

**Stan:**
```stan
// Spline
mu = B_matrix * beta;

// GP
mu = gp_exp_quad_cov(x, alpha=eta, rho=ell) * weights;

// Polynomial
mu = beta_0 + X_poly * beta;
```

---

## Expected Posterior Quantities

### Spline
```
E[beta_0 | Y] ~ 2.3 (close to prior, as intercept well-identified)
E[sum |beta_k| | Y] ~ 0.3-0.5 (shrinkage reduces total coefficient mass)
E[tau_global | Y] ~ 0.1-0.3 (data determines smoothness)
E[sigma | Y] ~ 0.15-0.20 (residual SD after accounting for trend)
```

### GP
```
E[beta_0 | Y] ~ 2.3
E[eta^2 | Y] ~ 0.05-0.15 (marginal GP variance)
E[ell | Y] ~ 3-10 (characteristic length scale in x units)
E[sigma | Y] ~ 0.10-0.18 (nugget variance)
```

### Polynomial
```
E[beta_0 | Y] ~ 2.3
E[beta_1 | Y] ~ 0.1-0.3 (linear term, positive)
E[beta_2 | Y] ~ -0.02 to -0.01 (quadratic term, negative for concavity)
E[beta_j | Y] ~ 0 for j > 3 (shrunk away by horseshoe)
E[sigma | Y] ~ 0.15-0.20
```

---

This document provides complete mathematical specifications for implementation and verification. All formulas are computationally explicit and can be directly translated to Stan or PyMC code.
