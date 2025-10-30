# Mathematical Specifications: Three Model Classes

## Notation

- **J = 8**: Number of schools
- **y_i**: Observed effect for school i
- **sigma_i**: Known standard error for school i (not estimated)
- **theta_i**: True effect for school i (latent parameter)
- **mu**: Overall population mean effect
- **tau**: Between-school standard deviation

---

## Model 1: Standard Hierarchical Model

### Full Specification

```
Likelihood:     y_i | theta_i, sigma_i  ~ Normal(theta_i, sigma_i)     i = 1, ..., 8

School level:   theta_i | mu, tau       ~ Normal(mu, tau)              i = 1, ..., 8

Hyperpriors:    mu                       ~ Normal(0, 50)
                tau                      ~ HalfCauchy(0, 25)
```

### Parameterization Options

**Centered** (as above):
- Direct parameterization
- Can have funnel geometry if tau near 0
- Use when tau expected to be moderate to large

**Non-centered** (recommended):
```
theta_raw_i     ~ Normal(0, 1)
theta_i         = mu + tau * theta_raw_i
```
- Better geometry when tau near 0
- More efficient sampling
- Recommended for this dataset given low I²

### Derived Quantities

**Shrinkage factor** for school i:
```
B_i = tau² / (tau² + sigma_i²)
E[theta_i | y, mu, tau] ≈ (1 - B_i) * y_i + B_i * mu
```

**Posterior predictive** for new school:
```
theta_new | mu, tau ~ Normal(mu, tau)
```

---

## Model 2: Near-Complete Pooling Model

### Full Specification

```
Likelihood:     y_i | theta_i, sigma_i  ~ Normal(theta_i, sigma_i)     i = 1, ..., 8

School level:   theta_i | mu, tau       ~ Normal(mu, tau)              i = 1, ..., 8

Hyperpriors:    mu                       ~ Normal(10, 15)
                tau                      ~ Exponential(0.5)
```

### Key Differences from Model 1

1. **Prior on mu**: Centered at 10 (near weighted mean) vs. 0
   - More informative
   - Reflects belief effects are positive and moderate

2. **Prior on tau**: Exponential(0.5) vs. HalfCauchy(0, 25)
   - Mean = 2, mode = 0
   - Strong regularization toward homogeneity
   - Much less heavy-tailed than half-Cauchy

### Non-Centered Parameterization (Essential)

```
theta_raw_i     ~ Normal(0, 1)
theta_i         = mu + tau * theta_raw_i

mu              ~ Normal(10, 15)
tau             ~ Exponential(0.5)
```
- Avoids funnel when tau → 0
- Necessary given strong prior on tau

### Asymptotic Behavior

As tau → 0:
```
theta_i → mu  for all i
y_i | mu ~ Normal(mu, sigma_i)
```
Reduces to pure fixed-effect model (complete pooling)

---

## Model 3: Mixture Model (K=2 Components)

### Full Specification

```
Likelihood:     y_i | theta_i, sigma_i   ~ Normal(theta_i, sigma_i)     i = 1, ..., 8

Latent class:   z_i | pi                 ~ Categorical(pi)               i = 1, ..., 8
                pi = (pi_1, pi_2)        ~ Dirichlet(2, 2)

School level:   theta_i | z_i=k, mu, tau ~ Normal(mu_k, tau_k)          k = 1, 2

Hyperpriors:    mu_k                     ~ Normal(0, 50)                 k = 1, 2
                tau_k                    ~ HalfCauchy(0, 15)             k = 1, 2

Constraint:     mu_1 < mu_2              (ordered to avoid label switching)
```

### Marginalized Form (No Explicit z_i)

The marginal distribution of theta_i is a mixture:
```
p(theta_i | mu, tau, pi) = sum_{k=1}^{2} pi_k * Normal(theta_i | mu_k, tau_k)
```

In Stan, use log_sum_exp for numerical stability:
```stan
for (j in 1:J) {
  vector[2] log_pi;
  for (k in 1:2) {
    log_pi[k] = log(pi[k]) + normal_lpdf(theta[j] | mu[k], tau[k]);
  }
  target += log_sum_exp(log_pi);
}
```

### Derived Quantities

**Responsibility matrix** (posterior cluster probabilities):
```
gamma_ik = P(z_i = k | theta_i, mu, tau, pi)
         = pi_k * Normal(theta_i | mu_k, tau_k) / sum_{k'} pi_{k'} * Normal(theta_i | mu_{k'}, tau_{k'})
```

**Component separation**:
```
Delta = |mu_2 - mu_1|
```
Large Delta (> 10) indicates well-separated components

---

## Model Comparison

### Parameters to Estimate

| Model | Parameters | Count |
|-------|-----------|-------|
| Model 1 | mu, tau, theta_1, ..., theta_8 | 10 |
| Model 2 | mu, tau, theta_1, ..., theta_8 | 10 |
| Model 3 | mu_1, mu_2, tau_1, tau_2, pi_1, pi_2, theta_1, ..., theta_8 | 14 |

Note: theta_i are nuisance parameters (not of primary interest)

### Effective Parameters (WAIC)

Effective number of parameters p_WAIC accounts for:
- Posterior variance
- Shrinkage (reduces effective parameters)
- Hierarchical structure

Expect: p_eff < 10 for Models 1-2 due to shrinkage

### Model Selection Criteria

**WAIC (Watanabe-Akaike Information Criterion)**:
```
WAIC = -2 * (lppd - p_WAIC)

lppd = sum_{i=1}^{J} log(E[p(y_i | theta_i)])     (log pointwise predictive density)
p_WAIC = sum_{i=1}^{J} Var(log p(y_i | theta_i))  (effective parameters)
```

**LOO-CV (Leave-One-Out Cross-Validation)**:
```
LOO = sum_{i=1}^{J} log p(y_i | y_{-i})

Computed via Pareto Smoothed Importance Sampling (PSIS)
Check Pareto-k diagnostic: k > 0.7 indicates influential observation
```

**Comparison**:
- Difference in WAIC or LOO-ELPD
- SE of difference (accounts for correlation)
- Rule: |Delta| > 5 considered meaningful

---

## Prior Predictive Checks

Before fitting, simulate from priors to verify they're reasonable.

### Model 1 Prior Predictive

```
mu_sim       ~ Normal(0, 50)
tau_sim      ~ HalfCauchy(0, 25)
theta_i_sim  ~ Normal(mu_sim, tau_sim)      for i = 1, ..., 8
y_i_sim      ~ Normal(theta_i_sim, sigma_i)
```

Check:
- Are simulated y values in plausible range [-100, 100]?
- Do they show reasonable variation?

### Model 2 Prior Predictive

```
mu_sim       ~ Normal(10, 15)
tau_sim      ~ Exponential(0.5)
theta_i_sim  ~ Normal(mu_sim, tau_sim)
y_i_sim      ~ Normal(theta_i_sim, sigma_i)
```

Check:
- More concentrated around mu = 10
- Less between-school variation due to tight tau prior

### Model 3 Prior Predictive

```
pi_sim       ~ Dirichlet(2, 2)
mu_k_sim     ~ Normal(0, 50)     for k = 1, 2 with mu_1 < mu_2
tau_k_sim    ~ HalfCauchy(0, 15) for k = 1, 2
z_i_sim      ~ Categorical(pi_sim)
theta_i_sim  ~ Normal(mu_{z_i_sim}, tau_{z_i_sim})
y_i_sim      ~ Normal(theta_i_sim, sigma_i)
```

Check:
- Does prior allow for both unimodal and bimodal?
- Are component separations reasonable?

---

## Posterior Predictive Checks

After fitting, simulate replicated data and compare to observed.

### Test Statistics

```
T_1(y) = mean(y)              # Location
T_2(y) = sd(y)                # Spread
T_3(y) = min(y)               # Lower tail
T_4(y) = max(y)               # Upper tail
T_5(y) = |min(y)|/max(y)      # Asymmetry
```

For each:
```
p_value = P(T(y_rep) > T(y_obs))
```

Well-calibrated model: p-values should be uniform on [0, 1]

### Visual Checks

1. **Overlay density**: Plot histogram of y_obs vs. density of y_rep
2. **QQ plot**: Quantiles of y_obs vs. y_rep
3. **Individual school checks**: Compare y_i to posterior predictive interval

---

## Sensitivity Analysis

### Prior Sensitivity for Model 1

Vary tau prior, keep mu prior fixed:

| Prior | Scale | Mean | Heavy-tailed? |
|-------|-------|------|---------------|
| HalfCauchy(0, 25) | 25 | ∞ | Yes |
| HalfNormal(0, 25) | 25 | 19.9 | No |
| HalfStudent_t(3, 0, 25) | 25 | - | Moderate |
| Exponential(0.1) | 10 | 10 | No |

Compare:
- Posterior mean of tau
- Posterior 95% credible interval
- Shrinkage factors

### Influence Analysis

**Leave-One-Out**:
For each school j:
1. Refit model excluding school j
2. Compare posterior of mu, tau to full model
3. Compute change: Delta_mu_j = mu_{-j} - mu_full

Identify influential schools: |Delta_mu_j| > 1

**School 5 Specific**:
Given it's the only negative effect with high precision:
1. Fit with all 8 schools
2. Fit with schools 1-4, 6-8 (exclude 5)
3. Compare posteriors

---

## Computational Details

### MCMC Settings

**Recommended**:
- Chains: 4
- Warmup: 1000
- Iterations: 2000 (1000 post-warmup per chain)
- Total: 4000 posterior draws
- adapt_delta: 0.95 (increase to 0.99 if divergences)
- max_treedepth: 10 (increase to 15 if needed)

### Convergence Diagnostics

**R-hat** (potential scale reduction):
- Target: < 1.01
- Warning: > 1.05
- Failure: > 1.1

**Effective Sample Size (ESS)**:
- Target: > 400 (10% of total draws)
- Warning: < 100
- Failure: < 50

**Divergent transitions**:
- Target: 0
- Acceptable: < 1% of post-warmup
- Failure: > 5%

### Reparameterization if Needed

If divergences persist, use **non-centered parameterization**:

**Centered** (current):
```stan
theta ~ normal(mu, tau);
```

**Non-centered**:
```stan
theta_raw ~ std_normal();
theta = mu + tau * theta_raw;
```

Non-centered is better when:
- tau posterior concentrated near 0
- J is small
- Funnel geometry suspected

---

## Software Implementation

### Stan

Primary choice for:
- Full HMC sampling
- Gradient-based inference
- Excellent diagnostics

### PyMC

Alternative for:
- Python-native workflow
- Similar features to Stan
- NUTS sampler

### Model Comparison Tools

**Stan/PyMC + arviz**:
- WAIC, LOO via arviz.waic(), arviz.loo()
- Plotting: arviz.plot_posterior(), plot_trace()
- Diagnostics: arviz.summary()

---

## Expected Computational Time

On modern laptop (4 cores):

- Model 1: ~30 seconds
- Model 2: ~30 seconds
- Model 3: ~2-5 minutes (slower due to mixture)

Bottlenecks:
- Model 3 may have label switching
- If divergences, need to increase adapt_delta (slower)
- LOO-CV: ~10 seconds per model

---

## Interpretation Guidelines

### Shrinkage Interpretation

**High shrinkage** (B_i → 1):
- theta_i pulled strongly toward mu
- Individual estimate unreliable
- Occurs when: tau small OR sigma_i large

**Low shrinkage** (B_i → 0):
- theta_i ≈ y_i (no pooling)
- Individual estimate reliable
- Occurs when: tau large OR sigma_i small

### Credible Intervals

**Posterior interval for theta_i**:
- Typically narrower than y_i ± 2*sigma_i
- Reflects information from other schools
- More honest uncertainty (accounts for tau uncertainty)

**Posterior interval for mu**:
- Population-average effect
- Not conditional on any specific school
- Useful for: "What effect should we expect in a new school?"

---

## Summary: Key Equations

**Model 1 (Standard)**:
```
y_i ~ N(theta_i, sigma_i)
theta_i ~ N(mu, tau)
mu ~ N(0, 50), tau ~ HC(0, 25)
```

**Model 2 (Near-Complete)**:
```
y_i ~ N(theta_i, sigma_i)
theta_i ~ N(mu, tau)
mu ~ N(10, 15), tau ~ Exp(0.5)
```

**Model 3 (Mixture)**:
```
y_i ~ N(theta_i, sigma_i)
theta_i ~ sum_k pi_k * N(mu_k, tau_k)
mu_k ~ N(0, 50), tau_k ~ HC(0, 15)
pi ~ Dir(2, 2)
```

**Shrinkage**:
```
B_i = tau² / (tau² + sigma_i²)
```

**Model Comparison**:
```
WAIC = -2 * (lppd - p_WAIC)
LOO = sum_i log p(y_i | y_{-i})
```

---

**Implementation ready**: These specifications can be directly translated to Stan/PyMC code.
