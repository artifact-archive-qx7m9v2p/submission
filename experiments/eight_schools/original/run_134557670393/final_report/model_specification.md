# Complete Model Specification
## Bayesian Hierarchical Meta-Analysis

**Report**: Bayesian Meta-Analysis of Eight Studies
**Date**: October 28, 2025
**Purpose**: Complete technical specification for reproducibility

---

## 1. Mathematical Specification

### 1.1 Full Generative Model

**Data Model** (Likelihood):
```
y_i | theta_i, sigma_i ~ Normal(theta_i, sigma_i^2)   for i = 1, ..., J
```

Where:
- `y_i` = observed effect size for study i (data)
- `theta_i` = true underlying effect for study i (latent parameter)
- `sigma_i` = known standard error for study i (data, fixed)
- `J = 8` = number of studies

**Hierarchical Model**:
```
theta_i | mu, tau ~ Normal(mu, tau^2)   for i = 1, ..., J
```

Where:
- `mu` = population mean effect (primary parameter of interest)
- `tau` = between-study standard deviation (heterogeneity parameter)

**Prior Distributions**:
```
mu ~ Normal(0, 50)
tau ~ Half-Cauchy(0, 5)
```

**Parameterization**: Non-centered for computational efficiency
```
theta_raw_i ~ Normal(0, 1)
theta_i = mu + tau * theta_raw_i
```

### 1.2 Joint Posterior Distribution

The target of inference is the joint posterior:

```
p(mu, tau, theta | y, sigma) ∝
  p(mu) * p(tau) *
  prod_{i=1}^J [ p(theta_i | mu, tau) * p(y_i | theta_i, sigma_i) ]
```

Explicitly:

```
p(mu, tau, theta | y, sigma) ∝
  Normal(mu | 0, 50) *
  HalfCauchy(tau | 0, 5) *
  prod_{i=1}^8 [ Normal(theta_i | mu, tau) * Normal(y_i | theta_i, sigma_i) ]
```

### 1.3 Marginal Posteriors of Interest

**Primary Parameters**:
- `p(mu | y, sigma)`: Population mean effect
- `p(tau | y, sigma)`: Between-study heterogeneity
- `p(theta_i | y, sigma)`: Study-specific effects

**Derived Quantities**:
- `P(mu > 0 | y, sigma)`: Probability effect is positive
- `P(mu > c | y, sigma)`: Probability effect exceeds threshold c
- `P(tau > 0 | y, sigma)`: Probability heterogeneity exists
- `p(theta_new | y, sigma)`: Predictive distribution for new study

### 1.4 Posterior Predictive Distribution

For predicting a new study:

```
p(y_new | y, sigma_new) =
  integral [ p(y_new | theta_new, sigma_new) *
             p(theta_new | mu, tau) *
             p(mu, tau | y, sigma) ] d(mu, tau, theta_new)
```

Simplified:
```
theta_new | y ~ integrated over p(mu, tau | y) of Normal(mu, tau)
y_new | theta_new, sigma_new ~ Normal(theta_new, sigma_new)
```

---

## 2. Prior Specifications and Justifications

### 2.1 Prior on mu (Population Mean Effect)

**Distribution**: `mu ~ Normal(0, 50)`

**Parameterization**:
- Location: 0 (no directional bias)
- Scale: 50 (standard deviation)

**95% Prior Coverage**: [-98, 98]

**Justification**:
1. **Weakly informative**: Covers observed effect range (-3 to 28) with ample margin (3.5x range)
2. **Centered at zero**: Reflects no prior belief about direction
3. **Broad tails**: Allows data to dominate inference
4. **Not flat**: Provides gentle regularization against extreme values
5. **Standard practice**: Similar to recommendations in Gelman et al. (2013)

**Prior Predictive Implications**:
- P(|mu| > 100) ≈ 0.05 (excludes implausibly large effects)
- P(-50 < mu < 50) ≈ 0.68 (reasonable effect sizes likely)
- Compatible with observed data (no prior-data conflict)

**Sensitivity**: Posterior largely insensitive to this prior given data informativeness. Could use Normal(0, 25) or Normal(0, 100) with minimal impact on conclusions.

### 2.2 Prior on tau (Between-Study Heterogeneity)

**Distribution**: `tau ~ Half-Cauchy(0, 5)`

**Parameterization**:
- Location: 0 (lower bound, since tau ≥ 0)
- Scale: 5

**Properties**:
- **Median**: 5
- **Mean**: Undefined (Cauchy has no mean)
- **Mode**: 0
- **95% Prior Coverage**: [0.13, 31.8]
- **Heavy tails**: P(tau > 100) = 0.03 (allows extreme heterogeneity if data support it)

**Justification** (Gelman 2006):
1. **Standard recommendation**: Half-Cauchy(0, scale) is the default for hierarchical variance parameters
2. **Mode at zero**: Compatible with I²=0% finding from classical analysis
3. **Heavy tails**: Allows large tau without strong prior penalty
4. **Scale = 5 rationale**:
   - Approximately 0.5 × mean within-study SE (12.5)
   - Conservative for small samples (doesn't force strong pooling)
   - If tau > 5, studies differ substantially; this prior allows discovering that

**Prior Predictive Implications**:
- P(tau < 1) ≈ 0.16 (low heterogeneity possible)
- P(1 ≤ tau < 5) ≈ 0.34 (moderate heterogeneity)
- P(tau ≥ 5) ≈ 0.50 (high heterogeneity possible)
- Flexible across full range of plausible heterogeneity levels

**Why Not Alternatives**:
- **Uniform**: Improper, computational issues, not recommended
- **Half-Normal**: Less heavy-tailed, may underestimate heterogeneity
- **Inverse-Gamma**: Can induce bias toward zero (Gelman 2006)
- **Half-t(df=3)**: Similar to Half-Cauchy, also acceptable

**Sensitivity Check** (recommended): Refit with Half-Cauchy(0, 2.5) and Half-Cauchy(0, 10) to confirm robustness.

### 2.3 Implicit Priors

**Standard Errors (sigma_i)**: Treated as known data, not estimated
- Standard meta-analytic assumption
- Assumes primary studies report accurate uncertainty
- No prior distribution (fixed at observed values)

**Study Effects (theta_i)**: Conditionally on mu, tau
- Prior: p(theta_i | mu, tau) = Normal(mu, tau)
- This is the hierarchical prior, not a top-level prior
- Induces partial pooling automatically

---

## 3. Non-Centered Parameterization

### 3.1 Motivation

Standard ("centered") parameterization:
```
theta_i ~ Normal(mu, tau)
```

**Problem**: When tau is small (near zero), posterior has funnel geometry:
- When tau ≈ 0, theta_i must be near mu (strong dependence)
- Creates "funnel" in (tau, theta) space
- Causes divergences in HMC sampling

### 3.2 Non-Centered Transformation

**Reparameterization**:
```
theta_raw_i ~ Normal(0, 1)           # Uncorrelated with mu, tau
theta_i = mu + tau * theta_raw_i      # Deterministic transformation
```

**Advantages**:
1. **Decorrelates parameters**: theta_raw independent of tau in prior
2. **Better geometry**: No funnel, smoother posterior
3. **Faster sampling**: Higher ESS per iteration
4. **Fewer divergences**: Numerical stability improved

**Equivalence**: Mathematically equivalent to centered parameterization
- Same marginal posteriors for mu, tau, theta_i
- Only reparameterizes for computational efficiency

### 3.3 Implementation

**PyMC Code**:
```python
with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=50)
    tau = pm.HalfCauchy('tau', beta=5)

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)
```

**Stan Code**:
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;  // Non-centered
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;  // Deterministic
}
model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta_raw ~ normal(0, 1);  // Uncorrelated

  // Likelihood
  y ~ normal(theta, sigma);
}
```

---

## 4. Sampling Configuration

### 4.1 MCMC Settings

**Sampler**: NUTS (No-U-Turn Sampler)
- Adaptive Hamiltonian Monte Carlo
- Automatically tunes step size and trajectory length
- No manual tuning required

**Chains**: 4 independent chains
- Allows convergence assessment via R-hat
- Parallelizable for speed

**Iterations per chain**:
- Warmup: 1000 (tuning phase, discarded)
- Sampling: 1000 (post-warmup, retained)
- **Total retained samples**: 4 × 1000 = 4000

**Target acceptance probability**: 0.95
- Higher than default (0.8) for better accuracy
- Reduces divergences at cost of slightly slower sampling

**Random seed**: 12345 (for reproducibility)

### 4.2 Convergence Diagnostics

**R-hat (Potential Scale Reduction Factor)**:
- **Target**: < 1.01 (excellent), < 1.05 (acceptable)
- **Achieved**: 1.000 for all parameters (perfect)
- **Interpretation**: Chains have converged to same distribution

**Effective Sample Size (ESS)**:
- **ESS bulk**: Sampling efficiency for central moments (mean, variance)
- **ESS tail**: Sampling efficiency for tail quantiles
- **Target**: > 400 per parameter
- **Achieved**:
  - mu: ESS bulk = 2047, ESS tail = 2341
  - tau: ESS bulk = 1878, ESS tail = 2156
  - theta[i]: ESS bulk > 2000, ESS tail > 2100
- **Interpretation**: High-quality samples, low autocorrelation

**Divergences**:
- **Target**: < 1% of post-warmup samples
- **Achieved**: 0 divergences (0%)
- **Interpretation**: No numerical instabilities, good posterior geometry

**Energy Diagnostic**:
- **E-BFMI** (Energy Bayesian Fraction of Missing Information): 0.21
- **Target**: > 0.2
- **Interpretation**: HMC effectively explores posterior

### 4.3 Computational Performance

**Hardware**: Standard desktop/laptop (no special requirements)
- No GPU needed
- 8-16 GB RAM sufficient

**Runtime**: ~40 seconds total
- ~10 seconds per chain (parallel execution)
- 4000 samples / 40 sec = 100 samples/sec

**Sampling Efficiency**: 61 ESS/sec
- ESS bulk (2047) / runtime (33.6 sec) ≈ 61
- Excellent efficiency for hierarchical model

**Memory**: ~200 MB for InferenceData object

---

## 5. Generated Quantities

### 5.1 Quantities Computed in Model

**Posterior Predictive**:
```python
y_rep = pm.Normal('y_rep', mu=theta, sigma=sigma, shape=J)
```
- Replicated data for posterior predictive checks
- Same structure as observed y
- One draw per posterior sample

**Log-Likelihood**:
```python
pm.Deterministic('log_lik', pm.logp(pm.Normal.dist(mu=theta, sigma=sigma), y))
```
- Point-wise log-likelihood for each observation
- Required for LOO-CV calculation
- Shape: (n_samples, J)

### 5.2 Post-Processing Computations

**Probability Statements**:
```python
P_mu_positive = (mu_samples > 0).mean()
P_mu_gt_5 = (mu_samples > 5).mean()
P_tau_positive = (tau_samples > 0).mean()
P_tau_lt_5 = (tau_samples < 5).mean()
```

**Credible Intervals**:
```python
mu_CI_95 = np.percentile(mu_samples, [2.5, 97.5])
mu_CI_90 = np.percentile(mu_samples, [5, 95])
tau_CI_95 = np.percentile(tau_samples, [2.5, 97.5])
```

**Shrinkage**:
```python
shrinkage = theta_mean - y_observed
relative_shrinkage = shrinkage / sigma
```

**Leave-One-Out Predictions**:
```python
loo = az.loo(idata)
loo_predictions = loo.loo
pareto_k = loo.pareto_k
```

---

## 6. Model Variants and Extensions

### 6.1 Alternative Priors (Sensitivity Analysis)

**More Informative on tau**:
```
tau ~ Half-Normal(0, 3)
```
- Forces moderate heterogeneity
- Less heavy-tailed than Half-Cauchy
- Use if prior knowledge suggests limited heterogeneity

**Less Informative on tau**:
```
tau ~ Half-Cauchy(0, 10)
```
- Allows larger heterogeneity more easily
- Use if concerned about underestimating tau

**More Informative on mu**:
```
mu ~ Normal(5, 10)
```
- Centers prior on positive effect
- Use if external evidence suggests direction

### 6.2 Robust Likelihood (Student-t)

**Modification**:
```
y_i | theta_i, sigma_i, nu ~ Student-t(nu, theta_i, sigma_i)
nu ~ Gamma(2, 0.1)
```

**Purpose**: Accommodate outliers via heavy-tailed likelihood

**When to use**:
- Suspected outliers (e.g., Study 1)
- Robustness check
- If posterior predictive shows outliers

**Trade-off**: Additional parameter (nu) to estimate

### 6.3 Fixed-Effects Variant

**Modification**: Set tau = 0 (no heterogeneity)

```
# Remove tau and hierarchical structure
y_i | mu, sigma_i ~ Normal(mu, sigma_i)
mu ~ Normal(0, 15)
```

**Purpose**: Simplicity benchmark, test if hierarchical structure needed

**When to use**: If posterior tau ≈ 0, fixed-effects may be adequate

**Comparison**: Use LOO-CV to compare to hierarchical model

### 6.4 Meta-Regression (with Covariates)

**Extension**: Include study-level covariates

```
theta_i | mu, tau, X_i, beta ~ Normal(mu + X_i * beta, tau)
beta ~ Normal(0, 10)
```

Where X_i are study characteristics (e.g., year, sample size, quality)

**Purpose**: Explain heterogeneity via moderators

**Requirement**: Covariate data (not available in current analysis)

---

## 7. Inference and Interpretation

### 7.1 Point Estimates

**Posterior Mean vs Median**:
- **Mean**: Expected value, influenced by tails
- **Median**: 50th percentile, robust to skewness
- **Recommendation**: Report both; prefer median for skewed parameters (tau)

**Maximum A Posteriori (MAP)**:
- Mode of posterior distribution
- Not commonly used in Bayesian inference
- Point estimates less informative than full distributions

### 7.2 Interval Estimates

**Credible Intervals** (Equal-Tailed):
- 95% CI: [2.5th percentile, 97.5th percentile]
- **Interpretation**: "95% probability parameter lies in interval given data"
- Direct probability statement (unlike frequentist CI)

**Highest Density Intervals (HDI)**:
- Shortest interval containing 95% of probability mass
- For symmetric posteriors, equal to credible interval
- For skewed posteriors, may differ slightly

**Recommendation**: Report equal-tailed CI for consistency with literature

### 7.3 Probability Statements

**Direct Posterior Probabilities**:
```
P(mu > 0 | data) = integral from 0 to inf of p(mu | data) d(mu)
```

**Advantages**:
- Directly interpretable
- Answers policy-relevant questions
- Avoids p-value misinterpretations

**Examples**:
- P(mu > 0): Probability effect is positive
- P(mu > 5): Probability effect exceeds threshold
- P(tau < 3): Probability heterogeneity is small

### 7.4 Prediction for New Studies

**Predictive Distribution**:
```
theta_new | data ~ integrated posterior of Normal(mu, tau)
```

**Procedure**:
1. Sample (mu, tau) from posterior
2. For each sample, draw theta_new ~ Normal(mu, tau)
3. Summarize theta_new distribution

**Interpretation**: Distribution of true effects in a new study from same population

**Uncertainty Components**:
- Estimation uncertainty (mu, tau)
- Between-study variability (tau)
- Does NOT include within-study measurement error (sigma_new)

**For observed outcome**:
```
y_new | data, sigma_new ~ Normal(theta_new, sigma_new)
```

---

## 8. Model Assumptions

### 8.1 Core Assumptions

**A1: Normality of Likelihoods**:
- Within-study: y_i ~ Normal(theta_i, sigma_i)
- **Justification**: Central limit theorem, standard meta-analysis assumption
- **Violation**: If primary outcomes non-normal, CLT may not apply
- **Test**: Posterior predictive checks

**A2: Normality of Study Effects**:
- Between-study: theta_i ~ Normal(mu, tau)
- **Justification**: Reasonable first approximation, convenient
- **Violation**: True effects may be skewed or multi-modal
- **Alternative**: Mixture models, non-parametric approaches

**A3: Known Standard Errors**:
- sigma_i treated as fixed (known)
- **Justification**: Standard meta-analysis practice
- **Violation**: If sigma_i underestimated, CIs too narrow
- **Test**: Compare reported sigma_i to uncertainty from raw data if available

**A4: Independence of Studies**:
- y_i conditionally independent given theta_i
- **Justification**: Different studies, different participants
- **Violation**: Overlapping samples, same research group, geographic clustering
- **Test**: Examine study characteristics for dependence

**A5: Exchangeability**:
- theta_i are exchangeable draws from p(theta | mu, tau)
- **Justification**: Studies from same target population
- **Violation**: Systematic differences (e.g., by year, quality)
- **Alternative**: Meta-regression with covariates

### 8.2 Prior Assumptions

**A6: Weakly Informative Priors**:
- Priors regularize but don't dominate data
- **Check**: Prior predictive generates reasonable data
- **Test**: Prior sensitivity analysis

**A7: Heavy-Tailed tau Prior**:
- Half-Cauchy allows large heterogeneity
- **Justification**: Conservative, data-driven learning
- **Alternative**: Half-Normal for more informative prior

### 8.3 Testing Assumptions

**Posterior Predictive Checks**: Test A1, A2
- Generate y_rep from posterior
- Compare to observed y
- Check for outliers, systematic deviations

**Leave-One-Out**: Test influence and A4
- Ensure no single study dominates
- Check for overly influential observations

**Residual Analysis**: Test A1, A2
- Plot residuals vs fitted values
- Check for patterns, non-normality

**Prior Sensitivity**: Test A6
- Vary priors, check robustness of conclusions
- If conclusions change dramatically, priors too influential

---

## 9. Software Implementation

### 9.1 PyMC Version (Used in This Analysis)

**Environment**:
```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd

# Version requirements
# pymc >= 5.26.1
# arviz >= 0.19.0
# numpy >= 1.26.0
```

**Model Code**:
```python
with pm.Model() as model:
    # Data
    y_obs_data = pm.MutableData('y_obs_data', y)
    sigma_data = pm.MutableData('sigma_data', sigma)

    # Priors
    mu = pm.Normal('mu', mu=0, sigma=50)
    tau = pm.HalfCauchy('tau', beta=5)

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma_data,
                       observed=y_obs_data)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=theta, sigma=sigma_data, shape=J)

    # Sampling
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        random_seed=12345,
        idata_kwargs={'log_likelihood': True}
    )
```

### 9.2 Stan Version (Alternative)

**Model Code** (`meta_analysis.stan`):
```stan
data {
  int<lower=1> J;              // Number of studies
  vector[J] y;                 // Observed effects
  vector<lower=0>[J] sigma;    // Known SEs
}
parameters {
  real mu;                     // Overall mean
  real<lower=0> tau;           // Between-study SD
  vector[J] theta_raw;         // Non-centered effects
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta_raw ~ normal(0, 1);

  // Likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] log_lik;
  vector[J] y_rep;

  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    y_rep[j] = normal_rng(theta[j], sigma[j]);
  }
}
```

**Fitting** (Python, CmdStanPy):
```python
import cmdstanpy as csp

# Compile
model = csp.CmdStanModel(stan_file='meta_analysis.stan')

# Data
data = {'J': 8, 'y': y, 'sigma': sigma}

# Sample
fit = model.sample(
    data=data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    seed=12345
)

# Convert to ArviZ
idata = az.from_cmdstanpy(fit)
```

---

## 10. Reproducibility Checklist

**For Complete Reproducibility**:

1. **Data**: `/workspace/data/data.csv`
   - 8 rows (studies)
   - Columns: study_id, y (effect), sigma (SE)

2. **Software Versions**:
   - Python: 3.11+
   - PyMC: 5.26.1
   - ArviZ: 0.19+
   - NumPy: 1.26+
   - Pandas: 2.2+

3. **Random Seed**: 12345 (set before sampling)

4. **Model Code**: See Section 9.1 or `/workspace/experiments/experiment_1/posterior_inference/code/fit_model_pymc.py`

5. **Sampling Configuration**:
   - 4 chains
   - 1000 warmup, 1000 sampling per chain
   - Target accept = 0.95
   - Non-centered parameterization

6. **Hardware**: Standard desktop (no special requirements)

7. **Operating System**: Linux, macOS, or Windows (cross-platform)

8. **Runtime**: ~40 seconds

**Verification**: Rerun should produce R-hat=1.00, ESS>2000, 0 divergences, and posterior means/medians within Monte Carlo error (~0.1 units for mu, ~0.2 for tau).

---

**Specification Document Prepared**: October 28, 2025
**Purpose**: Complete technical reference for model implementation and reproduction
**Status**: Validated and Used in Analysis
