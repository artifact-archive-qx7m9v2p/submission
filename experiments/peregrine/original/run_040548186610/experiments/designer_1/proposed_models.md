# Parametric Bayesian GLM Models for Time Series Count Data

**Designer:** Designer 1 (Parametric Focus)
**Date:** 2025-10-29
**Dataset:** 40 observations, years -1.67 to +1.67, counts 19-272

---

## Executive Summary

I propose **3 parametric Bayesian GLM models** that address the key challenges in this data:
1. **Negative Binomial with Quadratic Trend** - addresses overdispersion + non-linear growth
2. **Negative Binomial with Exponential Trend** - exponential growth hypothesis
3. **Quasi-Poisson with Quadratic Trend + Observation-Level Random Effects** - flexible overdispersion

All models use Stan for posterior inference and include explicit falsification criteria.

**Key Design Decisions:**
- Focus on negative binomial family (overdispersion = 68 demands this)
- Test both polynomial and exponential trend hypotheses
- Use weakly informative priors centered on EDA findings
- Plan for model failure by defining clear rejection criteria

---

## Core Data Challenges (from EDA)

1. **Extreme overdispersion:** Variance/Mean = 68 (Poisson expects 1.0)
2. **Strong non-linearity:** Quadratic R² = 0.961 vs Linear R² = 0.885
3. **Accelerating growth:** 6x rate increase from early to late period
4. **High autocorrelation:** lag-1 r = 0.989 (temporal dependence)
5. **Heteroscedastic variance:** Different dispersion across time periods

**What parametric GLMs can and cannot address:**
- Can handle: Overdispersion, non-linear trends, count distributions
- Cannot handle well: Complex temporal dependencies (leave to Designer 2's state-space models)
- Limitation: Assumes constant dispersion parameter (may be violated)

---

## Model 1: Negative Binomial with Quadratic Trend (BASELINE)

### Theoretical Motivation

The EDA shows accelerating growth (6x rate increase) and extreme overdispersion. A negative binomial distribution naturally handles count data with variance exceeding the mean, while a quadratic trend captures acceleration without assuming exponential growth continues indefinitely.

**Data generation hypothesis:** Counts follow a growth process with:
- Systematic acceleration (positive feedback loop, e.g., word-of-mouth)
- Stochastic variation larger than mean (heterogeneous population)
- Possible eventual saturation (quadratic can flex downward)

### Mathematical Specification

**Likelihood:**
```
C_i ~ NegativeBinomial(mu_i, phi)
```

**Link function (log):**
```
log(mu_i) = beta_0 + beta_1 * year_i + beta_2 * year_i^2
```

**Parameter interpretation:**
- `mu_i`: Expected count at time i
- `phi`: Dispersion parameter (larger = more overdispersion)
- `beta_0`: Intercept (log count at year=0)
- `beta_1`: Linear growth rate
- `beta_2`: Acceleration (if positive, growth is accelerating)

**Equivalent mean-variance form:**
```
E[C_i] = mu_i
Var[C_i] = mu_i + mu_i^2 / phi
```

As phi → ∞, approaches Poisson. As phi → 0, variance grows quadratically.

### Prior Distributions

**Strategy:** Weakly informative priors centered on EDA findings but allowing uncertainty.

```stan
// Intercept: center on observed mean at year=0
beta_0 ~ Normal(4.7, 0.5)  // log(109) ≈ 4.7, count around 100

// Linear term: positive growth expected
beta_1 ~ Normal(0.8, 0.3)  // Exponential fit suggested 0.85

// Quadratic term: small acceleration expected
beta_2 ~ Normal(0.3, 0.2)  // Positive but constrained

// Dispersion: weakly informative, allowing heavy overdispersion
phi ~ Gamma(2, 0.5)  // Mode around 2-4, allows range 0.5-20
```

**Prior justification:**
- `beta_0`: EDA shows mean ~109, log(109) ≈ 4.7. SD=0.5 allows 50-250 counts.
- `beta_1`: Exponential model had growth rate 0.85. Allowing +/- 0.6.
- `beta_2`: Quadratic term from OLS was 0.30. Allowing flexibility.
- `phi`: Gamma(2, 0.5) is weakly informative, modes around 2-4, but allows wide range.

**Prior predictive check:** These priors should generate count trajectories ranging from 10-500, encompassing observed data but not wildly unrealistic.

### Stan Implementation Outline

```stan
data {
  int<lower=0> N;           // Number of observations
  array[N] int<lower=0> y;  // Count observations
  vector[N] year;           // Standardized year
}

transformed data {
  vector[N] year_sq = year .* year;  // Precompute squared term
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;  // Dispersion
}

transformed parameters {
  vector[N] log_mu = beta_0 + beta_1 * year + beta_2 * year_sq;
  vector[N] mu = exp(log_mu);
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(0.8, 0.3);
  beta_2 ~ normal(0.3, 0.2);
  phi ~ gamma(2, 0.5);

  // Likelihood
  y ~ neg_binomial_2(mu, phi);
}

generated quantities {
  array[N] int y_rep;  // Posterior predictive samples
  vector[N] log_lik;   // For LOO-CV

  for (n in 1:N) {
    y_rep[n] = neg_binomial_2_rng(mu[n], phi);
    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu[n], phi);
  }
}
```

### Expected Strengths

1. **Handles overdispersion directly** via phi parameter
2. **Captures acceleration** without assuming exponential explosion
3. **Interpretable parameters:** beta_1 = growth rate, beta_2 = acceleration
4. **Computationally stable:** Well-defined posteriors, should sample efficiently
5. **Flexible shape:** Quadratic can capture slight slowdown if present

### Expected Weaknesses

1. **Ignores temporal autocorrelation** (lag-1 = 0.989)
   - Will underestimate uncertainty
   - Standard errors too narrow
   - Residuals will show autocorrelation

2. **Constant dispersion assumption:**
   - EDA shows variance changes across time periods
   - Q3 has much higher overdispersion (13.5x) than other periods
   - May underfit variance structure

3. **Extrapolation danger:**
   - Quadratic can go negative outside data range
   - No saturation mechanism
   - Predictions beyond year=2 unreliable

4. **Symmetric about peak:**
   - Quadratic is symmetric around turning point
   - Real growth processes often asymmetric (S-curves)

5. **No mechanism for changepoints:**
   - Assumes smooth acceleration
   - Misses if there are regime changes

### Falsification Criteria: I will abandon this model if...

**Critical failures (must switch model class):**
1. **Phi posterior includes zero:** Model cannot resolve overdispersion → Switch to zero-inflated models
2. **Residual autocorrelation > 0.9:** Temporal structure dominates → Switch to AR/state-space models (Designer 2)
3. **Posterior predictive checks fail badly:** <50% coverage → Fundamental distributional misspecification
4. **Beta_2 posterior includes large negative values:** Suggests downturn, need logistic/saturation model

**Warning signs (try model extensions first):**
1. **Dispersion inadequacy:** Observed variance still exceeds predicted by 2x → Try time-varying dispersion
2. **Systematic residual patterns by time period:** → Add period-specific effects or splines
3. **Poor out-of-sample predictions:** RMSE > 40 on last 20% → Model overfitting or wrong functional form
4. **Prior-posterior conflict:** Posterior modes far from priors → Priors misspecified or model wrong

### Diagnostic Plan

**Must check:**
1. **Posterior predictive checks:** Overlay y_rep draws on observed data
2. **Residual plots:** Plot Pearson residuals vs year, check for patterns
3. **Dispersion adequacy:** Compare observed vs predicted variance by time period
4. **LOO-CV:** Check for influential points (Pareto k > 0.7)
5. **Autocorrelation of residuals:** ACF plot, should be near zero

**Success criteria:**
- Posterior predictive 95% intervals cover 90-98% of observed data
- Residual autocorrelation < 0.5
- LOO-IC improvement over simpler models
- Posterior distributions well-identified (ESS > 400, Rhat < 1.01)

---

## Model 2: Negative Binomial with Exponential Trend

### Theoretical Motivation

The EDA exponential model fit extremely well (R² = 0.929), suggesting the data may follow exponential growth. This is consistent with population growth, epidemic spread, or compound growth processes. The log-linear relationship on transformed data (R² = 0.935 on log scale) supports this hypothesis.

**Data generation hypothesis:** Multiplicative growth process where:
- Growth rate is proportional to current level (dC/dt ∝ C)
- This generates exponential trajectories: C(t) = C_0 * exp(r*t)
- Stochastic variation is multiplicative (log-normal errors)

This is more mechanistically justified than polynomial models for many real processes.

### Mathematical Specification

**Likelihood:**
```
C_i ~ NegativeBinomial(mu_i, phi)
```

**Link function (log):**
```
log(mu_i) = beta_0 + beta_1 * year_i
```

**This implies:**
```
mu_i = exp(beta_0) * exp(beta_1 * year_i)
     = mu_0 * exp(r * year_i)
```

**Parameter interpretation:**
- `beta_0 = log(mu_0)`: Log count at year=0
- `beta_1 = r`: Instantaneous growth rate (per standardized year)
- `exp(beta_1)`: Multiplicative factor per unit time
- `phi`: Overdispersion parameter

### Prior Distributions

```stan
// Intercept: observed mean at year=0
beta_0 ~ Normal(4.7, 0.5)  // log(109) ≈ 4.7

// Growth rate: exponential model suggested 0.85
beta_1 ~ Normal(0.85, 0.2)  // Constrain to positive growth

// Dispersion
phi ~ Gamma(2, 0.5)
```

**Prior justification:**
- `beta_0`: Same as Model 1, centers on observed mean
- `beta_1`: EDA exponential fit gave 0.85. SD=0.2 allows 0.45-1.25 (reasonable range)
- Could use `beta_1 ~ Normal(0.85, 0.2) T[0, ]` to enforce positivity if desired

**Growth rate interpretation:**
- beta_1 = 0.85 → multiply by exp(0.85) ≈ 2.34 per standardized year
- Over 40 observations (3.34 standardized years), factor of exp(0.85 * 3.34) ≈ 16x growth
- This matches observed data: 29 → 245 ≈ 8.4x (same order of magnitude)

### Stan Implementation Outline

```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;  // Could constrain <lower=0> for positive growth
  real<lower=0> phi;
}

transformed parameters {
  vector[N] log_mu = beta_0 + beta_1 * year;
  vector[N] mu = exp(log_mu);
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(0.85, 0.2);
  phi ~ gamma(2, 0.5);

  // Likelihood
  y ~ neg_binomial_2(mu, phi);
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  real growth_multiplier = exp(beta_1);  // Interpretable quantity

  for (n in 1:N) {
    y_rep[n] = neg_binomial_2_rng(mu[n], phi);
    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu[n], phi);
  }
}
```

### Expected Strengths

1. **Mechanistically motivated:** Exponential growth is the natural model for many processes
2. **Parsimonious:** Only 3 parameters (vs 4 in quadratic model)
3. **Strong empirical fit:** EDA showed exponential R² = 0.929
4. **Natural interpretation:** Growth multiplier = exp(beta_1)
5. **Handles overdispersion** via negative binomial

### Expected Weaknesses

1. **Same temporal correlation issue** as Model 1
   - Ignores lag-1 autocorrelation = 0.989
   - Will underestimate uncertainty

2. **Worse fit in tails:**
   - EDA showed quadratic outperformed exponential (AIC: 232 vs 254)
   - May underpredict early observations
   - May overpredict or underpredict late observations

3. **Lacks flexibility:**
   - Cannot capture acceleration changes
   - If growth rate changes over time, model will fail

4. **Explosive extrapolation:**
   - Even worse than quadratic for extrapolation
   - Exponential explodes to infinity
   - No saturation mechanism

5. **Constant dispersion assumption:**
   - Same issue as Model 1
   - Variance structure may be time-varying

### Falsification Criteria: I will abandon this model if...

**Critical failures:**
1. **LOO-IC substantially worse than quadratic** (delta > 10): → Model 1 or more flexible alternatives
2. **Systematic residual patterns:** Clear U-shape in residuals vs fitted → Need polynomial terms
3. **Beta_1 posterior spans negative values:** No consistent exponential growth → Wrong model class
4. **Posterior predictive coverage < 70%:** Fundamental distributional failure

**Warning signs:**
1. **Poor fit in specific time periods:** Early or late observations systematically missed → Time-varying growth rate
2. **Wide posterior for beta_1:** Data not informative about growth rate → May need regularization or different form
3. **Comparison to Model 1 shows large AIC difference:** If delta > 20, exponential hypothesis rejected

### Diagnostic Plan

**Model comparison with Model 1:**
1. **LOO-IC comparison:** Should be within 10 points, otherwise one model clearly better
2. **Residual patterns:** Check if exponential shows systematic bias that quadratic fixes
3. **Posterior predictive:** Which model better captures data variability?

**Hypothesis test:** Is exponential growth sufficient, or do we need acceleration term (beta_2)?
- If Model 1's beta_2 posterior excludes zero, exponential is insufficient
- If Model 1 and Model 2 have similar LOO-IC, prefer simpler exponential (parsimony)

---

## Model 3: Quasi-Poisson with Quadratic Trend + Observation-Level Random Effects

### Theoretical Motivation

Both Models 1 and 2 assume constant overdispersion (phi). However, the EDA revealed that variance structure changes dramatically across time:
- Q1: Var/Mean = 1.65
- Q2: Var/Mean = 2.39
- Q3: Var/Mean = 13.55 (extreme!)
- Q4: Var/Mean = 2.33

This suggests the overdispersion is not constant. An alternative approach: model each observation as having its own random deviation, allowing the dispersion to vary flexibly.

**Data generation hypothesis:**
- Counts follow a Poisson-like process with systematic trend
- Each observation has idiosyncratic variation beyond Poisson
- This variation may be time-dependent or observation-dependent

This is a "hierarchical overdispersion" model: we add a random effect at the observation level.

### Mathematical Specification

**Likelihood:**
```
C_i ~ Poisson(lambda_i * exp(epsilon_i))
```

**Or equivalently (using negative binomial parameterization):**
```
C_i ~ Poisson(mu_i * zeta_i)
where zeta_i ~ Gamma(shape=nu, rate=nu)  // Mean=1, Variance=1/nu
```

**Linear predictor:**
```
log(mu_i) = beta_0 + beta_1 * year_i + beta_2 * year_i^2

// Observation-level random effect
epsilon_i ~ Normal(0, sigma_obs)

// Final intensity
lambda_i = mu_i * exp(epsilon_i)
```

**Parameter interpretation:**
- `mu_i`: Expected count from trend
- `epsilon_i`: Observation-specific deviation (on log scale)
- `sigma_obs`: Amount of extra-Poisson variability
- If sigma_obs = 0, reduces to Poisson
- If sigma_obs large, substantial overdispersion

### Prior Distributions

```stan
// Trend parameters (same as Model 1)
beta_0 ~ Normal(4.7, 0.5)
beta_1 ~ Normal(0.8, 0.3)
beta_2 ~ Normal(0.3, 0.2)

// Observation-level variance
sigma_obs ~ Exponential(1)  // Weakly informative, favors smaller values

// Individual deviations (in model block)
epsilon ~ Normal(0, sigma_obs)  // Vectorized
```

**Why this prior for sigma_obs?**
- Exponential(1) has mean=1, SD=1, mode=0
- Favors simpler models (small sigma_obs) but allows large values
- If data needs large dispersion, posterior will shift sigma_obs up
- More flexible than fixed dispersion parameter

### Stan Implementation Outline

```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] year;
}

transformed data {
  vector[N] year_sq = year .* year;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> sigma_obs;          // Observation-level SD
  vector[N] epsilon_raw;             // Non-centered parameterization
}

transformed parameters {
  vector[N] epsilon = sigma_obs * epsilon_raw;
  vector[N] log_mu_trend = beta_0 + beta_1 * year + beta_2 * year_sq;
  vector[N] log_lambda = log_mu_trend + epsilon;  // Add random effect
  vector[N] lambda = exp(log_lambda);
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(0.8, 0.3);
  beta_2 ~ normal(0.3, 0.2);
  sigma_obs ~ exponential(1);

  // Non-centered parameterization for random effects
  epsilon_raw ~ normal(0, 1);

  // Likelihood
  y ~ poisson(lambda);
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    // For predictions, include random effect uncertainty
    real epsilon_new = normal_rng(0, sigma_obs);
    real lambda_pred = exp(log_mu_trend[n] + epsilon_new);
    y_rep[n] = poisson_rng(lambda_pred);
    log_lik[n] = poisson_lpmf(y[n] | lambda[n]);
  }
}
```

**Technical note:** Using non-centered parameterization (`epsilon = sigma_obs * epsilon_raw`) improves MCMC sampling efficiency when sigma_obs is small.

### Expected Strengths

1. **Flexible overdispersion:** Each observation can deviate independently
2. **Can capture time-varying dispersion:** No constraint that variance is constant
3. **Observation-level inference:** Can identify specific unusual observations
4. **Mathematically equivalent to negative binomial** but more flexible
5. **Interpretable:** Separates trend (mu) from noise (epsilon)

### Expected Weaknesses

1. **Many more parameters:** 40 epsilon values vs 1 phi parameter
   - Risk of overfitting with n=40
   - May be poorly identified
   - Computational cost higher

2. **Still ignores temporal correlation:**
   - epsilon_i are independent by assumption
   - But lag-1 autocorrelation is 0.989!
   - This is a MAJOR issue

3. **May absorb trend into random effects:**
   - If trend is misspecified, epsilon will compensate
   - Could hide model inadequacy
   - Hard to detect systematic patterns

4. **Interpretation ambiguity:**
   - Is large epsilon_i due to true randomness or misspecified trend?
   - Cannot distinguish without more data

5. **Prior sensitivity:**
   - Results may depend heavily on sigma_obs prior
   - Need to check prior sensitivity

### Falsification Criteria: I will abandon this model if...

**Critical failures:**
1. **Epsilon values show strong autocorrelation:** ACF(epsilon) > 0.7 → Need correlated random effects (AR structure)
2. **Sigma_obs posterior near zero:** Model reduces to Poisson, overdispersion not captured → Back to Model 1
3. **Divergent transitions or poor sampling:** Model too complex for n=40 → Use simpler Model 1
4. **LOO Pareto k > 0.7 for many observations:** Model overfitting → Too many parameters

**Warning signs:**
1. **Epsilon shows systematic time pattern:** Plot epsilon vs year, if trend visible → Misspecified trend function
2. **LOO-IC worse than Model 1 by >5:** Flexibility not helping → Stick with simpler negative binomial
3. **Wide posteriors for beta parameters:** Random effects absorbing signal → Identification problem

### Diagnostic Plan

**Key checks:**
1. **ACF of epsilon:** Should be near zero if model is correctly specified
2. **Plot epsilon vs year:** Should show no pattern (random scatter)
3. **Posterior of sigma_obs:** Should be clearly away from zero (0.2-2.0 range)
4. **Compare to Model 1:** LOO-IC, posterior predictive coverage
5. **Prior sensitivity:** Refit with sigma_obs ~ Exponential(2) and Exponential(0.5)

**Success criteria:**
- LOO-IC comparable to or better than Model 1
- epsilon shows no temporal structure
- Better captures time-varying variance (check Q3 period specifically)

---

## Model Comparison Strategy

### Primary Comparison Metrics

**1. LOO-IC (Leave-One-Out Information Criterion):**
- Bayesian cross-validation approximation
- Lower is better
- Differences > 4 are meaningful, > 10 are substantial
- Use `loo` package in R or `arviz.loo` in Python

**2. Posterior Predictive Coverage:**
- What % of observed data falls in 95% posterior predictive intervals?
- Should be 90-98% (perfect calibration = 95%)
- If < 85%, model is missing important structure
- If > 99%, model may be overfitting

**3. Overdispersion Adequacy:**
```
Check: Var(y_rep) / Mean(y_rep) vs Var(y) / Mean(y)
Target: Ratio should be close to 1.0
```
By time period: Compute this ratio for Q1, Q2, Q3, Q4 separately

**4. Out-of-Sample Prediction:**
- Fit models on first 32 observations (80%)
- Predict last 8 observations (20%)
- Compute RMSE and MAE
- Check if true values fall in predictive intervals

### Model Comparison Table (to be filled after fitting)

| Metric | Model 1 (NB Quad) | Model 2 (NB Exp) | Model 3 (Quasi-Pois) |
|--------|-------------------|------------------|----------------------|
| LOO-IC | ? | ? | ? |
| Posterior Coverage (%) | ? | ? | ? |
| Var/Mean Ratio | ? | ? | ? |
| Out-Sample RMSE | ? | ? | ? |
| Residual ACF(1) | ? | ? | ? |
| Sampling Efficiency | ? | ? | ? |
| Interpretability | High | Highest | Medium |
| Parsimony | Good | Best | Poor |

### Decision Rules

**If Model 1 (NB Quadratic) wins:**
- Acceleration is real and important
- Constant dispersion is adequate approximation
- **Next step:** Try adding AR(1) errors for temporal correlation

**If Model 2 (NB Exponential) wins:**
- Exponential growth hypothesis confirmed
- Simpler model is sufficient
- **Next step:** Consider time-varying growth rate or logistic model for saturation

**If Model 3 (Quasi-Poisson) wins:**
- Time-varying dispersion is crucial
- Observation-level variation important
- **Next step:** Try correlated random effects (epsilon with AR structure)

**If all models fail similarly:**
- Parametric GLMs insufficient
- **Pivot to:** Designer 2's state-space models or Designer 3's hierarchical/temporal models
- Problem: Temporal correlation (r=0.989) dominates signal

### What Would Make Me Abandon Parametric GLMs Entirely?

**I will recommend switching to non-parametric or fully temporal models if:**

1. **All three models show residual autocorrelation > 0.8**
   - Signal: Temporal structure dominates
   - Action: Need AR, state-space, or GP models

2. **Posterior predictive coverage < 70% for all models**
   - Signal: Fundamental distributional misspecification
   - Action: Try zero-inflated, hurdle, or mixture models

3. **Systematic residual patterns persist in all models**
   - Signal: Trend form is wrong (need splines, GPs, or non-parametric)
   - Action: Switch to Designer 2's approaches

4. **LOO-IC differences < 2 between all models**
   - Signal: Models are equally inadequate
   - Action: Data doesn't distinguish parametric forms, need more flexible approach

5. **Extreme parameter values or numerical instability**
   - Signal: Model class incompatible with data
   - Action: Rethink entire modeling framework

---

## Prior Predictive Checks (To Be Done)

Before fitting, I will:

1. **Simulate from priors only** (no data)
   - Generate 1000 datasets from prior predictive distribution
   - Check if simulated counts are in reasonable range (10-500)
   - Visualize: Do simulated trends look plausible?

2. **Prior sensitivity analysis:**
   - Refit with more diffuse priors: SD × 2
   - Refit with more informative priors: SD / 2
   - Check if posteriors change substantially

3. **Prior-data conflict checks:**
   - Compare prior predictive to observed data
   - If prior puts < 1% mass on observed data, prior is too strong

**Expected prior predictive ranges:**
- Model 1 & 2: Counts from 10 to 500, mostly 50-200
- Model 3: Similar, with more variability due to observation-level effects

---

## Implementation Roadmap

### Phase 1: Model Fitting (Priority Order)

1. **Start with Model 1 (NB Quadratic):**
   - Most likely to succeed
   - Baseline for comparisons
   - Diagnose key issues

2. **Fit Model 2 (NB Exponential):**
   - Test exponential hypothesis
   - Compare to Model 1

3. **Fit Model 3 (Quasi-Poisson) IF Models 1/2 show issues:**
   - Only if dispersion inadequacy detected
   - More complex, save for later

### Phase 2: Diagnostics (Critical)

For each model:
1. **MCMC diagnostics:** Rhat < 1.01, ESS > 400, no divergences
2. **Posterior predictive checks:** Visual overlays, coverage statistics
3. **Residual analysis:** ACF, plots vs time, variance by period
4. **LOO-CV:** Pareto k values, LOO-IC comparison
5. **Out-of-sample validation:** Last 20% predictions

### Phase 3: Decision Point

**After fitting all models:**
1. Create comparison table (above)
2. Identify best parametric GLM
3. **Critical decision:** Are parametric GLMs sufficient?
   - If residual ACF > 0.7: → Recommend temporal models
   - If coverage < 80%: → Recommend more flexible models
   - If no clear winner: → Parametric GLMs inadequate

### Phase 4: Model Extensions (If Needed)

**If parametric GLMs show promise but have issues:**
- Add AR(1) errors to best model
- Try time-varying dispersion parameter: phi(t) = phi_0 + phi_1 * year
- Consider breaking into periods with changepoints

**If parametric GLMs fundamentally fail:**
- Document why they failed
- Recommend specific alternatives from other designers
- Pivot strategy entirely

---

## What Each Model Will Teach Us

### Model 1 (NB Quadratic):
**Hypothesis:** Growth is accelerating systematically (polynomial trend)

**If it works well:**
- Confirms acceleration hypothesis
- Suggests deterministic growth process
- Dispersion is approximately constant

**If it fails:**
- Residual ACF > 0.7 → Temporal structure dominates
- Poor fit in Q3 → Time-varying dispersion needed
- Beta_2 near zero → Linear or exponential sufficient

**We learn:** Is acceleration (quadratic term) necessary?

### Model 2 (NB Exponential):
**Hypothesis:** Growth is multiplicative/exponential (constant % growth rate)

**If it works well:**
- Confirms exponential growth hypothesis
- Simpler explanation (parsimony wins)
- Growth rate constant over time

**If it fails:**
- Worse LOO-IC than Model 1 → Acceleration not captured by exponential
- Systematic residuals → Growth rate changes over time
- Poor fit at extremes → Need polynomial flexibility

**We learn:** Is constant growth rate sufficient, or does it change?

### Model 3 (Quasi-Poisson):
**Hypothesis:** Dispersion varies across observations (flexible overdispersion)

**If it works well:**
- Time-varying dispersion is important
- Q3 period needs special treatment
- Observation-level variation is large

**If it fails:**
- Too complex for n=40 → Overfitting
- Epsilon shows autocorrelation → Need correlated structure
- Similar to Model 1 → Constant dispersion adequate

**We learn:** Is constant dispersion assumption violated?

### Meta-Learning from Model Comparison:

**Model 1 vs Model 2:** Tests polynomial vs exponential growth hypothesis
- If Model 1 wins by >10 LOO-IC: Acceleration is real
- If Model 2 wins by >10 LOO-IC: Exponential is sufficient
- If difference <4: Data doesn't distinguish (need more data or flexibility)

**Model 1 vs Model 3:** Tests constant vs flexible dispersion
- If Model 3 wins: Time-varying variance matters
- If Model 1 wins: Constant dispersion adequate (simpler is better)

**All models show high residual ACF:** Temporal correlation dominates parametric form
- Implies: Need to move beyond GLMs to temporal models
- Action: Recommend Designer 2's state-space models

---

## Integration with Other Designers

### Designer 2 (Non-Parametric / State-Space):
**If my models fail due to:**
- High residual autocorrelation (>0.7)
- Systematic temporal patterns
- Inability to capture dynamics

**Then Designer 2 should explore:**
- AR(p) models for temporal correlation
- State-space models with time-varying parameters
- Gaussian Process regression (non-parametric trends)

**My models provide:** Baseline parametric performance, quantify how much temporal structure matters

### Designer 3 (Hierarchical / Temporal):
**If my models show:**
- Time-varying dispersion (Model 3 wins)
- Different behavior in different periods (Q1, Q2, Q3, Q4)
- Changepoint evidence

**Then Designer 3 should explore:**
- Hierarchical models by time period
- Changepoint models
- Random effects at period level

**My models provide:** Evidence for/against constant dispersion, period-specific diagnostics

### Synthesis Strategy:

**Best case:** One of my models works well
- Provides interpretable parametric baseline
- Clear parameter estimates (growth rate, acceleration, dispersion)
- Other designers can compare flexibility vs complexity tradeoff

**Likely case:** Models work OK but show temporal correlation issues
- Hybrid approach: Parametric trend + AR errors
- Or: Handoff to Designer 2 for full temporal modeling

**Worst case:** All parametric GLMs fail
- Strong evidence that parametric forms are insufficient
- Points toward non-parametric, state-space, or mixture models
- Informs what NOT to do

---

## Falsification Summary: Global Rejection Criteria

**I will declare parametric GLMs INADEQUATE if:**

1. **All models show residual ACF(1) > 0.80**
   - Temporal correlation cannot be ignored
   - GLM independence assumption fatally violated

2. **All models have posterior predictive coverage < 75%**
   - Distributional assumptions wrong
   - Missing key structural features

3. **All models show systematic residual patterns**
   - U-shape, trend, or period effects remaining
   - Functional form fundamentally wrong

4. **LOO-IC differences between models < 3**
   - Models equally bad
   - Data doesn't distinguish parametric forms
   - Need more flexibility

5. **Out-of-sample RMSE > 50 for all models**
   - Poor predictive performance (>20% error)
   - Models not learning useful structure

**If ANY TWO of above criteria are met → Recommend switching to alternative model classes entirely**

---

## Technical Details: Stan vs PyMC

All three models can be implemented in either framework:

### Stan Advantages:
- Faster sampling (HMC/NUTS highly optimized)
- Better for complex hierarchical models (Model 3)
- Strong diagnostics (divergences, energy plots)
- Recommended for production use

### PyMC Advantages:
- More Pythonic syntax
- Easier prior/posterior visualization
- Better integration with Python ML ecosystem
- Recommended for exploratory analysis

**Recommendation:** Implement in Stan for performance, use `arviz` for visualization.

### Expected Computational Cost:
- Model 1 & 2: 1-2 minutes per model (4 chains, 2000 iterations)
- Model 3: 5-10 minutes (more parameters, longer warmup)

### Convergence Expectations:
- Model 1 & 2: Should converge easily (simple structure)
- Model 3: May need non-centered parameterization, longer warmup

---

## Appendix: Mathematical Details

### Negative Binomial Parameterizations

**Stan's `neg_binomial_2(mu, phi)` uses:**
```
E[Y] = mu
Var[Y] = mu + mu^2 / phi
```

**Relationship to traditional NB(r, p):**
```
r = phi
p = phi / (phi + mu)
```

**As phi → ∞:** Variance → mu (Poisson limit)
**As phi → 0:** Variance → ∞ (extreme overdispersion)

### Quadratic Trend Interpretation

```
log(mu) = beta_0 + beta_1 * t + beta_2 * t^2

Taking derivatives:
d(log(mu))/dt = beta_1 + 2 * beta_2 * t

Growth rate at time t:
r(t) = beta_1 + 2 * beta_2 * t
```

**Interpretation:**
- If beta_2 > 0: Growth rate increases over time (acceleration)
- If beta_2 < 0: Growth rate decreases over time (deceleration)
- If beta_2 = 0: Constant exponential growth rate

**Turning point:** t_max = -beta_1 / (2 * beta_2)
(Only relevant if beta_2 < 0)

### Exponential Trend Interpretation

```
log(mu) = beta_0 + beta_1 * t
mu = exp(beta_0) * exp(beta_1 * t)

Growth multiplier per unit time:
mu(t+1) / mu(t) = exp(beta_1)

Doubling time:
t_double = log(2) / beta_1
```

**For beta_1 = 0.85:**
- Multiplier = exp(0.85) = 2.34 per standardized year
- Doubling time = 0.82 standardized years

---

## Summary: Key Deliverables

**I am proposing:**
1. Three parametric Bayesian GLM models with different hypotheses
2. Complete mathematical specifications with priors
3. Clear falsification criteria for each model
4. Comparison strategy to identify best parametric approach
5. Decision rules for when to abandon parametric GLMs entirely

**Expected outcomes:**
- **Best case:** One model fits well, provides interpretable parameters
- **Likely case:** Models work reasonably but show temporal correlation issues → Hybrid approach
- **Challenging case:** All models show high residual ACF → Handoff to Designer 2/3

**Critical questions to answer:**
1. Is growth polynomial (accelerating) or exponential (constant rate)?
2. Is constant dispersion adequate, or is it time-varying?
3. Can we ignore temporal correlation, or does it dominate?

**Timeline:**
- Phase 1 (Fitting): 2-3 hours
- Phase 2 (Diagnostics): 2-3 hours
- Phase 3 (Comparison): 1-2 hours
- **Total:** 5-8 hours of computation and analysis

---

**Document prepared by:** Designer 1 (Parametric GLM Focus)
**Ready for:** Implementation and empirical testing
**Collaboration:** Awaiting Designer 2 and Designer 3 proposals for comparison
