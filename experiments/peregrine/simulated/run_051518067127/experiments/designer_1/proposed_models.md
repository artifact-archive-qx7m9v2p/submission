# Bayesian Count Model Designs for Exponentially Growing Time Series
## Model Designer 1: Count Likelihood Families

**Designer**: Model Designer 1
**Date**: 2025-10-30
**Focus**: Count data likelihood families (Negative Binomial, overdispersed Poisson, hierarchical structures)

---

## Executive Summary

This document proposes **three competing Bayesian model classes** for count time series data with exponential growth (2.37x/year) and severe overdispersion (variance/mean = 70.43). Each model makes fundamentally different assumptions about the data-generating process. **All models must use Stan/PyMC for posterior inference and include log-likelihood calculations for LOO-CV.**

**Critical mindset**: Each model is designed with explicit **falsification criteria**. The goal is to discover which models fail quickly and pivot to better alternatives.

### Data Context (40 observations)
- Counts range: [21, 269] (no zeros)
- Exponential growth: 2.37x multiplier per standardized year
- Severe overdispersion: Var/Mean = 70.43 (but heterogeneous across time)
- Strong autocorrelation: ACF lag-1 = 0.971 (0.754 after detrending)
- Evidence of regime shift at series midpoint

### Competing Hypotheses
1. **Overdispersion is intrinsic** → Negative Binomial likelihood with fixed dispersion
2. **Overdispersion is time-varying** → Hierarchical model with evolving dispersion
3. **Autocorrelation dominates** → State-space model with latent AR process

---

## Model Class 1: Negative Binomial GLM with Quadratic Trend

### Theoretical Justification

**Core hypothesis**: The data arise from a **contagion or clustering process** where events occur in bursts. The Negative Binomial naturally arises from Poisson processes with Gamma-distributed rates (Poisson-Gamma mixture).

**Why this might work**:
- NB explicitly models overdispersion via parameter θ (dispersion parameter)
- Log-link captures exponential growth mechanistically
- Quadratic term allows for changing growth rates (acceleration/deceleration)
- Standard GLM framework, computationally stable

**Why this might FAIL** (falsification criteria):
1. **Temporal structure**: If autocorrelation remains high in posterior predictive checks (ACF > 0.5 at lag-1)
2. **Homogeneous dispersion assumption**: If period-specific dispersion parameters differ drastically (early: 0.68, middle: 13.11, late: 7.23)
3. **Regime shifts**: If residual patterns show clear structural breaks
4. **Computational warnings**: Divergent transitions or R-hat > 1.01
5. **Prior-posterior conflict**: If posterior concentrates far from prior mode (suggests misspecification)

### Full Bayesian Specification

```
Likelihood:
  C_t ~ NegativeBinomial2(mu_t, phi)

  where mu_t = expected count at time t
        phi = overdispersion parameter (larger phi = less overdispersion)

  NegativeBinomial2 parameterization:
    mean = mu
    variance = mu + mu^2/phi

Link function:
  log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2

Parameters:
  beta_0: intercept (log-scale mean at year=0)
  beta_1: linear growth rate (log-scale slope)
  beta_2: acceleration/deceleration parameter
  phi: dispersion parameter (inverse overdispersion)
```

### Prior Specification

**Priors chosen to be weakly informative** but constrain predictions to plausible range [10, 500]:

```stan
parameters {
  real beta_0;          // Intercept
  real beta_1;          // Linear effect
  real beta_2;          // Quadratic effect
  real<lower=0> phi;    // Overdispersion (reciprocal parameterization)
}

model {
  // Priors
  beta_0 ~ normal(4.5, 1.0);       // Centered at log(90) ≈ data mean
  beta_1 ~ normal(0.9, 0.5);       // Prior for ~2.5x growth, but uncertain
  beta_2 ~ normal(0, 0.3);         // Weak prior on curvature
  phi ~ gamma(2, 0.1);             // Mean=20, allows wide range of dispersion
}
```

**Prior justification**:

1. **beta_0 ~ N(4.5, 1.0)**:
   - Prior predictive at year=0: exp(4.5) = 90 (close to data mean 109)
   - 95% interval: [exp(2.5), exp(6.5)] = [12, 665] (covers data range)
   - Rationale: Weakly informative around observed center

2. **beta_1 ~ N(0.9, 0.5)**:
   - Prior: 95% interval [0, 1.9] on log-scale slope
   - Translates to multiplicative effects: [1.0x, 6.7x] per unit year
   - Data suggest 2.37x, so prior is centered slightly below but allows higher
   - Rationale: Growth is expected, but magnitude uncertain

3. **beta_2 ~ N(0, 0.3)**:
   - Prior: 95% interval [-0.6, 0.6] on quadratic coefficient
   - Centered at 0 (no curvature), but allows acceleration or deceleration
   - Conservative: Prefers linear unless data strongly suggest otherwise
   - Rationale: EDA shows quadratic adds 8% R², but may be overfitting

4. **phi ~ Gamma(2, 0.1)**:
   - Prior mean: 20, variance: 200
   - NB variance = mu + mu²/phi, so phi=20 at mu=100 gives var ≈ 600
   - Data shows var ≈ 7700, suggesting phi closer to 1-2
   - Wide prior allows data to dominate
   - Rationale: Expect strong overdispersion, but let data determine magnitude

### Prior Predictive Checks

**Before seeing data, this prior generates**:
- Count range: ~[5, 800] with 95% probability
- Covers observed range [21, 269]
- Allows both under- and overdispersion
- Permits various growth patterns (linear to exponential with curvature)

**Red flag if**: Prior predictive generates >10% of samples outside [1, 1000]

### Likelihood Calculation for LOO-CV

```stan
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;  // Posterior predictive samples

  for (t in 1:N) {
    real mu_t = exp(beta_0 + beta_1 * year[t] + beta_2 * year[t]^2);
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu_t, phi);
    y_rep[t] = neg_binomial_2_rng(mu_t, phi);
  }
}
```

### Posterior Predictive Checks (Falsification Tests)

**Model passes if**:
1. **Coverage**: 95% posterior intervals contain ~38/40 observations
2. **Mean-variance**: Predicted var/mean ratio ≈ 60-80
3. **Growth pattern**: Predicted log-linear trend R² > 0.90
4. **Tail behavior**: Maximum predicted count ≈ 250-300

**Model FAILS if**:
1. **Systematic underprediction** of variance (predicted var/mean < 30)
2. **Autocorrelation structure not captured**: ACF of standardized residuals > 0.6 at lag-1
3. **Poor calibration**: <90% or >98% coverage (suggests wrong uncertainty quantification)
4. **Regime shift visible in residuals**: Clear pattern change at t ≈ 20

### Convergence Requirements

**Strict MCMC diagnostics**:
- R-hat < 1.01 for all parameters
- ESS_bulk > 400 per chain (1600 total for 4 chains)
- ESS_tail > 400 per chain
- No divergent transitions (0 out of 4000 post-warmup)
- BFMI > 0.3 (energy diagnostic)

**If convergence fails**: Try non-centered parameterization or stronger priors

### Implementation Plan (Stan via CmdStanPy)

```python
import cmdstanpy
import numpy as np
import pandas as pd
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Stan data dictionary
stan_data = {
    'N': len(data),
    'year': data['year'].values,
    'C': data['C'].values.astype(int)
}

# Compile and fit
model = cmdstanpy.CmdStanModel(stan_file='nb_quadratic.stan')
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.95,  # Increase for better exploration
    max_treedepth=12,
    seed=42
)

# Extract posterior
posterior = fit.draws_pd()

# Compute LOO-CV
log_lik = fit.stan_variable('log_lik')
import arviz as az
loo = az.loo(log_lik, pointwise=True)
print(f"LOO-CV ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

# Posterior predictive checks
y_rep = fit.stan_variable('y_rep')
# Check autocorrelation
residuals = data['C'].values[:, None] - y_rep
acf_resid = np.array([np.corrcoef(residuals[:-k, i], residuals[k:, i])[0, 1]
                       for k in range(1, 6)
                       for i in range(y_rep.shape[1])]).mean()
print(f"Mean ACF lag-1: {acf_resid:.3f}")
```

### Computational Considerations

**Expected issues**:
1. **High posterior correlation** between beta_1 and beta_2 (quadratic terms)
   - Solution: Center and scale predictors (already done)

2. **Overdispersion parameter phi** may have long tail
   - Solution: Use gamma prior to constrain, monitor ESS

3. **Potential for overflow** in exp(log_mu) if beta values extreme
   - Solution: Use `exp()` in generated quantities with bounds checking

**Computational cost**: ~30 seconds for 4 chains x 2000 samples on modern CPU

### When to Abandon This Model

**Immediate abandonment if**:
1. Posterior predictive ACF > 0.5 at lag-1 (temporal structure not captured)
2. LOO-CV Pareto-k > 0.7 for >10% of observations (influential points, poor model)
3. Posterior variance consistently underpredicted by >50%
4. Visual inspection shows clear regime change in residuals

**Consider switching to**:
- **Hierarchical model** if dispersion clearly varies by period
- **State-space model** if autocorrelation dominates
- **Piecewise model** if structural break is evident

---

## Model Class 2: Hierarchical Negative Binomial with Time-Varying Dispersion

### Theoretical Justification

**Core hypothesis**: Overdispersion is **not constant**—it evolves over time due to changing underlying processes. EDA shows early period has var/mean=0.68 (underdispersed!), middle has 13.11, late has 7.23.

**Why this might work**:
- Accounts for non-stationarity in variance structure
- More flexible than fixed-dispersion model
- Can capture regime changes implicitly
- Hierarchical pooling prevents overfitting

**Why this might FAIL**:
1. **Overfitting**: Small sample (n=40) split into periods may not support parameter estimates
2. **Computational complexity**: Hierarchical models prone to divergences
3. **Arbitrary periodization**: Dividing into 3 periods is ad-hoc, true structure may be continuous
4. **Still ignores autocorrelation**: Time dependencies not modeled
5. **Identification issues**: If period effects are confounded with trend

### Full Bayesian Specification

```
Hierarchical structure:
  C_t ~ NegativeBinomial2(mu_t, phi[period[t]])

  log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2

  phi[p] ~ Gamma(alpha_phi, beta_phi)  for p in {1, 2, 3}

Hyperpriors:
  alpha_phi ~ Gamma(2, 0.5)   // Shape parameter for phi distribution
  beta_phi ~ Gamma(2, 0.5)    // Rate parameter for phi distribution

Period assignments:
  Period 1: t = 1-14  (early, underdispersed)
  Period 2: t = 15-27 (middle, highly overdispersed)
  Period 3: t = 28-40 (late, moderately overdispersed)
```

### Prior Specification

```stan
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  vector<lower=0>[3] phi;        // Period-specific dispersion
  real<lower=0> alpha_phi;       // Hyperparameter for phi distribution
  real<lower=0> beta_phi;        // Hyperparameter for phi distribution
}

model {
  // Hyperpriors
  alpha_phi ~ gamma(2, 0.5);     // Mean = 4, allows variability
  beta_phi ~ gamma(2, 0.5);      // Mean = 4, allows variability

  // Hierarchical prior on dispersion
  phi ~ gamma(alpha_phi, beta_phi);

  // Trend parameters (same as Model 1)
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);

  // Likelihood
  for (t in 1:N) {
    real mu_t = exp(beta_0 + beta_1 * year[t] + beta_2 * year[t]^2);
    C[t] ~ neg_binomial_2(mu_t, phi[period[t]]);
  }
}
```

**Key innovation**: Period-specific `phi` parameters allow dispersion to vary systematically. Hierarchical structure pools information across periods to prevent overfitting.

### Prior Justification

**Hierarchical structure rationale**:
- Expect some variation in dispersion, but not completely independent
- Gamma hyperpriors allow data to determine degree of pooling
- If alpha_phi and beta_phi posteriors are wide, suggests periods differ substantially
- If posteriors are tight, suggests pooling is appropriate

**Prior predictive**: Generates diverse dispersion patterns across periods while maintaining overall structure.

### Falsification Criteria

**Model FAILS if**:
1. **Phi posteriors overlap >80%**: No evidence for time-varying dispersion, simpler Model 1 preferred
2. **Within-period residuals still overdispersed**: Even after period-specific phi, variance not captured
3. **Computational failure**: Divergent transitions >1% despite tuning
4. **LOO-CV worse than Model 1**: Additional complexity not justified
5. **Posterior predictive still shows autocorrelation** > 0.5

**Model SUCCEEDS if**:
- Phi[1] < Phi[3] < Phi[2] (matches EDA pattern)
- Within-period posterior predictive variance matches data
- LOO-CV significantly better than Model 1 (ΔELPD > 3)

### Stress Test: Period Sensitivity

**Test**: Fit model with alternative period boundaries:
- Alternative 1: t=1-13, 14-26, 27-40
- Alternative 2: t=1-15, 16-28, 29-40

**If results change drastically**, this suggests:
- Periodization is arbitrary and misleading
- Need continuous change model instead
- **Action**: Abandon hierarchical period model, switch to random walk dispersion

### Implementation Considerations

**Computational challenges**:
1. **Hierarchical models require careful initialization**
   - Initialize phi near observed period-specific estimates

2. **Potential non-identifiability** if periods too similar
   - Monitor effective sample size for phi parameters
   - Check posterior correlations

3. **Longer sampling time**: ~2-3x Model 1 due to hierarchical structure

**Stan implementation notes**:
```stan
data {
  int<lower=0> N;
  int<lower=0> C[N];
  vector[N] year;
  int<lower=1, upper=3> period[N];  // Period assignments
}
```

### When to Abandon This Model

**Abandon immediately if**:
1. ESS for any phi parameter < 100 (poor mixing)
2. Divergent transitions > 5% despite max_adapt_delta=0.99
3. Phi posteriors indistinguishable (suggests model is overly complex)
4. LOO-CV has many Pareto-k > 0.7

**Pivot to**:
- **Model 1** if periods don't differ meaningfully
- **Random walk dispersion model** if continuous change apparent
- **State-space model** if temporal dependence dominates

---

## Model Class 3: Negative Binomial with AR(1) Latent Process

### Theoretical Justification

**Core hypothesis**: The **autocorrelation is the primary feature**, not just a nuisance. The observed counts are Negative Binomial draws from a latent mean process that follows AR(1) dynamics.

**Why this might work**:
- ACF lag-1 = 0.971 (0.754 after detrending) is extremely high
- Temporal dependencies are mechanistically plausible (cumulative processes)
- Separates systematic temporal structure from overdispersion
- More appropriate for time series than assuming independence

**Why this might FAIL**:
1. **Computational complexity**: Latent AR process adds N parameters
2. **Identifiability**: Hard to separate AR correlation from trend + overdispersion
3. **Small sample**: n=40 may be insufficient to reliably estimate AR parameter
4. **Misspecified dynamics**: AR(1) may be too simple (should be AR(p), ARMA, etc.)
5. **Non-Gaussian latent process**: AR(1) typically assumes Gaussian errors, but we have counts

### Full Bayesian Specification

```
Latent process (log-scale):
  eta_t = beta_0 + beta_1 * year_t + beta_2 * year_t^2 + z_t

  where z_t follows AR(1):
    z_1 ~ Normal(0, sigma_z / sqrt(1 - rho^2))  // Stationary initial state
    z_t ~ Normal(rho * z_{t-1}, sigma_z)        // for t > 1

Observation model:
  log(mu_t) = eta_t
  C_t ~ NegativeBinomial2(mu_t, phi)

Parameters:
  beta_0, beta_1, beta_2: Trend parameters
  phi: Overdispersion parameter
  rho: AR(1) coefficient, |rho| < 1
  sigma_z: Innovation variance in latent process
```

### Prior Specification

```stan
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;
  real<lower=-1, upper=1> rho;   // AR(1) coefficient
  real<lower=0> sigma_z;          // AR innovation SD
  vector[N] z;                    // Latent AR process (non-centered)
}

transformed parameters {
  vector[N] eta;
  eta[1] = beta_0 + beta_1 * year[1] + beta_2 * year[1]^2 + z[1];
  for (t in 2:N) {
    eta[t] = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2 + z[t];
  }
}

model {
  // Priors on trend (same as Model 1)
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ gamma(2, 0.1);

  // Priors on AR process
  rho ~ beta(8, 2);              // Prior favors high autocorrelation (mean=0.8)
  sigma_z ~ exponential(5);      // Prior: small innovations (mean=0.2)

  // Latent AR(1) process (non-centered parameterization)
  z[1] ~ normal(0, sigma_z / sqrt(1 - rho^2));  // Stationary initial condition
  for (t in 2:N) {
    z[t] ~ normal(rho * z[t-1], sigma_z);
  }

  // Likelihood
  for (t in 1:N) {
    C[t] ~ neg_binomial_2(exp(eta[t]), phi);
  }
}
```

### Prior Justification

**rho ~ Beta(8, 2)**:
- Prior mean: 0.8, SD: 0.12
- 95% interval: [0.53, 0.96]
- Rationale: EDA shows very high ACF, so prior concentrates on high values
- Still allows moderate autocorrelation if data suggest it

**sigma_z ~ Exponential(5)**:
- Prior mean: 0.2, SD: 0.2
- Most mass on small innovations (latent process is smooth)
- Rationale: Large sigma_z would overwhelm trend, keep innovations small

**Non-centered parameterization**:
- Improves sampling efficiency for latent variables
- Reduces posterior correlation between rho and z

### Falsification Criteria

**Model FAILS if**:
1. **Posterior rho < 0.3**: Autocorrelation not as important as thought, use simpler model
2. **Posterior sigma_z > 1.0**: Innovations too large, suggests misspecification
3. **ESS for z parameters < 50**: Latent process not well-identified
4. **Posterior predictive ACF still > 0.5**: AR(1) insufficient, need AR(p) or ARMA
5. **LOO-CV worse than Model 1**: Complexity not justified

**Model SUCCEEDS if**:
- Posterior rho ≈ 0.7-0.9 (consistent with EDA residual ACF)
- Posterior predictive ACF matches data ACF structure
- Residuals (C_t - mu_t) are approximately uncorrelated
- LOO-CV significantly better than independence models

### Posterior Predictive Checks: Autocorrelation Focus

**Specific checks for temporal structure**:

1. **ACF plot comparison**:
   - Compute ACF of data
   - Compute ACF of posterior predictive samples
   - Should match for lags 1-5

2. **One-step-ahead predictions**:
   - Use t=1:39 to predict t=2:40
   - Should be better than independence model

3. **Runs test**:
   - Test for randomness in sign of residuals
   - Should NOT reject (residuals are random)

### Implementation Challenges

**Major computational concerns**:

1. **Latent variable identifiability**:
   - 40 latent z parameters + correlation structure
   - Can be hard to disentangle z from beta parameters
   - **Solution**: Use informative priors, monitor ESS

2. **Sampling efficiency**:
   - Latent AR processes can have slow mixing
   - **Solution**: Non-centered parameterization (implemented above)
   - May need longer chains (4000+ samples)

3. **Computational cost**:
   - ~5-10x slower than Model 1
   - Expect 2-5 minutes for full inference

**Stan implementation note**:
```stan
// Alternative: Centered parameterization (if non-centered doesn't work)
// Replace z declaration with:
// vector[N] z_raw;
// In transformed parameters:
// z = z_raw * sigma_z;
// z[1] = z_raw[1] * sigma_z / sqrt(1 - rho^2);
```

### When to Abandon This Model

**Abandon immediately if**:
1. After 10,000 iterations, ESS < 100 for >5 z parameters (poor identification)
2. Rhat > 1.05 for z parameters after tuning (non-convergence)
3. Posterior rho very uncertain (95% credible interval covers [0.2, 0.95])
4. LOO-CV has Pareto-k warnings for >20% of observations

**Pivot to**:
- **Model 1** if autocorrelation not well-identified (simpler is better)
- **State-space model** with different dynamics (random walk, GP, etc.)
- **ARMA(p,q) extension** if AR(1) insufficient but temporal structure clear

---

## Model Comparison Strategy

### Phase 1: Fit All Three Models (Parallel)

**Fit all models simultaneously** to avoid confirmation bias:

1. Model 1 (NB Quadratic): Baseline, fast
2. Model 2 (Hierarchical): Test time-varying dispersion hypothesis
3. Model 3 (AR1): Test temporal dependence hypothesis

**Initial assessment criteria**:
- All models must converge (Rhat < 1.01, ESS > 400)
- All models must produce reasonable posterior predictive samples
- Eliminate any models with computational failures

### Phase 2: Quantitative Comparison

**Metrics** (in order of importance):

1. **LOO-CV ELPD** (with SE):
   - Primary model selection criterion
   - Difference >3 is meaningful
   - Watch for Pareto-k warnings

2. **Posterior predictive checks**:
   - Mean-variance relationship
   - ACF structure (lags 1-5)
   - Growth pattern (R² on log scale)
   - Coverage (proportion in 95% intervals)

3. **Parameter interpretability**:
   - Are posteriors consistent with EDA?
   - Do parameters have reasonable magnitudes?
   - Are uncertainties appropriately quantified?

### Phase 3: Diagnostic Decision Tree

```
IF Model 1 LOO-CV best AND ACF residuals < 0.3:
  → USE Model 1 (simplicity preferred)

ELIF Model 2 LOO-CV best AND phi parameters well-separated:
  → USE Model 2, but VERIFY with period sensitivity analysis
  → IF sensitive to period choice: ABANDON, try continuous dispersion model

ELIF Model 3 LOO-CV best AND rho well-identified:
  → USE Model 3
  → CHECK: Does sigma_z make sense? Is AR(1) sufficient?

ELIF Multiple models have similar LOO-CV (ΔELPD < 3):
  → ENSEMBLE or AVERAGE predictions
  → Report sensitivity to model choice

ELSE:
  → ALL MODELS FAILING
  → Red flags present, need different model class
  → Pivot to state-space, GP, or piecewise models
```

### Phase 4: Falsification Synthesis

**Cross-cutting checks** (apply to all models):

1. **Prior-posterior conflict**:
   - If posterior concentrates far from prior (>3 SD), suggests misspecification
   - Plot prior vs posterior densities

2. **Influential observations**:
   - LOO Pareto-k > 0.7 identifies observations poorly predicted
   - Investigate: Are they outliers? Regime changes? Measurement errors?

3. **Posterior predictive extremes**:
   - Does model predict values outside [0, 400]? (Red flag)
   - Does model predict impossible patterns? (e.g., negative growth in late period)

4. **Computational health**:
   - Any divergences? (Even 1% suggests geometry issues)
   - Energy diagnostic (BFMI > 0.3)
   - Trace plots look like "hairy caterpillars"?

---

## Red Flags That Would Trigger Complete Model Class Change

### When to Abandon ALL Proposed Models

**Critical failure modes**:

1. **All models show posterior predictive ACF > 0.6**:
   - Count likelihoods insufficient for temporal structure
   - **PIVOT TO**: State-space models (dynamic linear models, structural time series)
   - Alternative: Gaussian process on latent intensity

2. **Residuals show clear structural break at t≈20**:
   - Regime shift not captured by smooth trends or hierarchical dispersion
   - **PIVOT TO**: Piecewise models with changepoint detection
   - Alternative: Hidden Markov model with discrete states

3. **Overdispersion increases with mean despite NB**:
   - NB mean-variance relationship (mu + mu²/phi) may be wrong functional form
   - **PIVOT TO**: Conway-Maxwell-Poisson (flexible mean-variance)
   - Alternative: Beta-Binomial if there's an implicit "trials" interpretation

4. **Posterior predictive systematically over/underpredicts in specific periods**:
   - Missing covariate or external driver
   - **PIVOT TO**: Need additional predictors (if available)
   - Alternative: Semi-parametric models (splines, GAMs)

5. **All models fail to converge despite extensive tuning**:
   - Fundamental identifiability issues
   - **PIVOT TO**: Simpler baseline models (Poisson, log-normal OLS)
   - Alternative: Frequentist methods to establish baseline, then revisit Bayesian

### Decision Points for Major Strategy Pivots

**After fitting all 3 models, if**:

- **Scenario A**: Model 1 wins clearly (ΔELPD > 5, good diagnostics)
  - Conclusion: Overdispersion is primary issue, autocorrelation secondary
  - **Next step**: Refine Model 1 (try splines, test for breakpoints)

- **Scenario B**: Model 2 wins clearly
  - Conclusion: Time-varying dispersion is real
  - **Next step**: Test continuous dispersion evolution (random walk on log(phi))

- **Scenario C**: Model 3 wins clearly
  - Conclusion: Temporal dependence dominates
  - **Next step**: Test AR(2), ARMA(1,1), or more complex dynamics

- **Scenario D**: No model clearly superior (all ΔELPD < 2)
  - Conclusion: Multiple mechanisms at play, models capture different aspects
  - **Next step**: Consider hybrid models or ensemble approaches

- **Scenario E**: All models fail diagnostics
  - Conclusion: Wrong model class entirely
  - **Next step**: Emergency pivot to exploratory models (GP, changepoint, etc.)

---

## Alternative Approaches If Initial Models Fail

### Backup Plan 1: State-Space Model with Random Walk

If autocorrelation cannot be captured by AR(1):

```
Latent state:
  mu_t ~ NegativeBinomial2(exp(eta_t), phi)
  eta_t = eta_{t-1} + beta_1 + epsilon_t
  epsilon_t ~ Normal(0, sigma_epsilon)

Initial state:
  eta_1 ~ Normal(log(mean(C)), 1)
```

**Rationale**: Random walk may be more flexible than AR(1) for non-stationary growth.

### Backup Plan 2: Conway-Maxwell-Poisson (CMP)

If NB mean-variance relationship is wrong:

```
C_t ~ CMP(lambda_t, nu)
log(lambda_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2
```

**Rationale**: CMP has flexible mean-variance relationship via nu parameter.
**Implementation**: Use `brms` with `family=cmp()` or custom Stan code.

### Backup Plan 3: Gaussian Process on Latent Intensity

If temporal structure is complex and non-parametric:

```
eta ~ MVNormal(m, K)
where K is covariance matrix with squared exponential kernel
C_t ~ NegativeBinomial2(exp(eta_t), phi)
```

**Rationale**: GP captures arbitrary smooth temporal patterns.
**Computational cost**: Much higher, O(N³) covariance operations.

---

## Implementation Roadmap

### Week 1: Core Model Fitting

**Day 1-2**: Implement and fit Model 1 (NB Quadratic)
- Write Stan code
- Prior predictive checks
- Fit model, check convergence
- Posterior predictive checks
- Compute LOO-CV

**Day 3-4**: Implement and fit Model 2 (Hierarchical)
- Write Stan code with period indicators
- Fit model, monitor divergences
- Compare phi posteriors across periods
- Sensitivity analysis (alternative period boundaries)
- Compute LOO-CV

**Day 5-6**: Implement and fit Model 3 (AR1)
- Write Stan code with latent AR process
- Use non-centered parameterization
- Fit model, check ESS for z parameters
- Posterior predictive ACF checks
- Compute LOO-CV

**Day 7**: Model comparison and synthesis
- Compare LOO-CV scores
- Posterior predictive check synthesis
- Identify best model(s)
- Document failures and pivots

### Week 2: Refinement and Robustness

**If Model 1 wins**:
- Test cubic term
- Test for structural break
- Sensitivity to priors

**If Model 2 wins**:
- Test continuous dispersion change
- Alternative period specifications
- Hierarchical vs fixed effects comparison

**If Model 3 wins**:
- Test AR(2)
- Test ARMA(1,1)
- Test different AR prior specifications

**If no clear winner**:
- Ensemble predictions
- Model averaging via stacking
- Report sensitivity

### Week 3: Emergency Pivots (If Needed)

**If all models fail**:
- Implement state-space model
- Try GP model
- Consider changepoint detection
- Revisit EDA for missed patterns

---

## Summary: Models Prioritized by Theoretical Justification

### Priority 1: Model 1 (NB Quadratic)
**Why first**: Simplest model addressing core issues (overdispersion, exponential growth). Fast to fit, easy to diagnose. If this fails, we learn a lot about what's wrong.

**Expected outcome**: Will likely fit well but show residual autocorrelation. This tells us whether temporal structure is critical or secondary.

### Priority 2: Model 3 (AR1)
**Why second**: Addresses the most striking EDA finding (ACF=0.971). If temporal dependence is primary mechanism, this should win decisively.

**Expected outcome**: May have identification issues with small sample. Success would be strong evidence for temporal process models.

### Priority 3: Model 2 (Hierarchical)
**Why third**: Most complex, most assumptions (period boundaries). If this wins, suggests very specific time-varying structure.

**Expected outcome**: Likely sensitive to period specification. Success would require follow-up with continuous dispersion models.

---

## Critical Success Criteria

**A successful model**:
1. Converges reliably (Rhat < 1.01, no divergences)
2. Posterior predictive variance matches data (within 20%)
3. LOO-CV ELPD within 5 of best model
4. Residual ACF < 0.4 (or explicitly models temporal structure)
5. Posterior intervals well-calibrated (empirical coverage ≈ nominal)
6. Parameters interpretable and consistent with domain knowledge

**A failed model**:
1. Cannot achieve convergence despite tuning
2. Systematic posterior predictive failures (>30% outside 95% intervals)
3. LOO-CV ELPD > 10 worse than best model
4. Pareto-k warnings for >20% of observations
5. Posterior concentrates in implausible parameter regions

---

## Final Note: Embrace Falsification

**These models are hypotheses to be tested, not solutions to be forced.**

If all three models fail, that is **success**—we've learned the data generation process is more complex than these model classes allow. The goal is to fail fast, learn what's wrong, and pivot intelligently.

**Key principle**: Model comparison is not about picking the "winner" but about understanding which aspects of the data each model captures and which it misses. The residuals tell the story.

---

## Appendix: Stan Code Templates

### Model 1: NB Quadratic

```stan
data {
  int<lower=0> N;
  int<lower=0> C[N];
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;
}

model {
  // Priors
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ gamma(2, 0.1);

  // Likelihood
  for (t in 1:N) {
    real log_mu = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2;
    C[t] ~ neg_binomial_2_log(log_mu, phi);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;

  for (t in 1:N) {
    real log_mu = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2;
    real mu = exp(log_mu);
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | log_mu, phi);
    y_rep[t] = neg_binomial_2_rng(mu, phi);
  }
}
```

### Model 2: Hierarchical (Sketch)

```stan
data {
  int<lower=0> N;
  int<lower=0> C[N];
  vector[N] year;
  int<lower=1, upper=3> period[N];
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  vector<lower=0>[3] phi;
  real<lower=0> alpha_phi;
  real<lower=0> beta_phi;
}

model {
  // Hyperpriors
  alpha_phi ~ gamma(2, 0.5);
  beta_phi ~ gamma(2, 0.5);

  // Hierarchical prior
  phi ~ gamma(alpha_phi, beta_phi);

  // Trend priors
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);

  // Likelihood
  for (t in 1:N) {
    real log_mu = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2;
    C[t] ~ neg_binomial_2_log(log_mu, phi[period[t]]);
  }
}

generated quantities {
  vector[N] log_lik;
  // ... (similar to Model 1)
}
```

### Model 3: AR(1) (Sketch)

```stan
data {
  int<lower=0> N;
  int<lower=0> C[N];
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;
  real<lower=-1, upper=1> rho;
  real<lower=0> sigma_z;
  vector[N] z;
}

model {
  // Priors
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ gamma(2, 0.1);
  rho ~ beta(8, 2);
  sigma_z ~ exponential(5);

  // AR(1) process
  z[1] ~ normal(0, sigma_z / sqrt(1 - rho^2));
  for (t in 2:N) {
    z[t] ~ normal(rho * z[t-1], sigma_z);
  }

  // Likelihood
  for (t in 1:N) {
    real log_mu = beta_0 + beta_1 * year[t] + beta_2 * year[t]^2 + z[t];
    C[t] ~ neg_binomial_2_log(log_mu, phi);
  }
}

generated quantities {
  vector[N] log_lik;
  // ... (similar to Model 1)
}
```

---

**Document prepared by**: Model Designer 1
**Date**: 2025-10-30
**Output location**: `/workspace/experiments/designer_1/proposed_models.md`
**Implementation**: Stan (CmdStanPy) with LOO-CV via ArviZ
