# Hierarchical/Compositional Bayesian Model Designs
## Designer 3: Structured Decomposition Perspective

**Date**: 2025-10-28
**Perspective**: Hierarchical and compositional modeling with structured variance
**Data Context**: N=27, x∈[1,31.5], Y∈[1.71,2.63], replicates at 6 x-values, data gap at x∈[23,29]

---

## Executive Summary

From a hierarchical/compositional perspective, I propose **three distinct model classes** that decompose the Y-x relationship into interpretable components:

1. **Additive Decomposition Model**: Separates trend (parametric saturation) + smooth deviations (GP) + measurement error
2. **Hierarchical Replicate Model**: Explicitly models replicate structure with group-level and observation-level variance
3. **Compositional Variance Model**: Decomposes total variance into systematic heteroscedasticity + residual noise

Each model targets different structural features evident in the data (non-linearity, replicates, variance patterns) and provides clear falsification criteria.

---

## Critical Analysis of EDA Findings

### What the EDA Tells Us
- **Strong saturation pattern**: Spearman ρ=0.78, logarithmic fit R²=0.83
- **Replicate structure**: 6 x-values with 2-3 replicates (21 total observations across these)
- **Variance heterogeneity**: 4.6:1 ratio (low:high x), but not statistically significant
- **Data gap**: x∈[23,29] creates interpolation challenge
- **One influential point**: x=31.5 with high leverage

### What the EDA Might Be Missing
- **Hidden structure in replicates**: Do replicates represent true measurement error or systematic grouping (e.g., batches, time)?
- **Non-stationary processes**: Variance decrease might indicate regime change, not just noise reduction
- **Compositional effects**: Multiple underlying processes (biological growth + saturation + measurement) conflated
- **Gap artifacts**: Interpolation across gap may hide discontinuity or regime shift

### My Modeling Philosophy
From a hierarchical perspective, I believe the data generation process is **compositional**:
- **Level 1**: Systematic trend (saturation function)
- **Level 2**: Local deviations (may be structured or smooth)
- **Level 3**: Replicate variability (if replicates are true replicates)
- **Level 4**: Measurement error

I will design models that **explicitly separate these levels** and test whether the structure is necessary.

---

## Model Class 1: Additive Decomposition Model

### Conceptual Framework

**Hypothesis**: The relationship is composed of:
1. A **parametric saturation trend** (captures primary non-linearity)
2. **Smooth local deviations** from trend (captures unmodeled structure)
3. **Independent measurement noise**

This is a **semiparametric additive model** that combines interpretable parameters with flexible components.

### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(μ_i, σ_noise)
μ_i = f_trend(x_i) + f_smooth(x_i)

Trend component (parametric saturation):
f_trend(x) = α + β·log(x)

Smooth component (Gaussian Process):
f_smooth ~ GP(0, k(x, x'))
k(x, x') = η² · exp(-ρ² · (x - x')²)  [Squared Exponential kernel]

Priors:
α ~ Normal(1.75, 0.5)           # Intercept from EDA
β ~ Normal(0.27, 0.15)          # Log-slope from EDA (positive)
σ_noise ~ HalfNormal(0.15)     # Residual noise scale
η ~ HalfNormal(0.1)             # GP amplitude (smaller than trend)
ρ ~ InverseGamma(5, 5)          # GP length scale (moderate smoothness)
```

### Rationale

**Why this decomposition?**
- **Trend captures saturation**: Logarithmic term models the primary diminishing returns pattern (R²=0.83)
- **GP captures local structure**: Smoothly interpolates deviations, useful for gap x∈[23,29]
- **Separates scales**: Trend operates at data-wide scale, GP at local scale, noise at observation scale
- **Interpolation safety**: GP provides uncertainty-aware interpolation in data gap

**Prior Justification**:
- **α, β**: Centered on EDA point estimates with moderate uncertainty
- **η small**: GP should explain residual variance (~0.12), not dominate trend
- **ρ moderate**: Length scale ~5 allows smooth but not rigid interpolation
- **σ_noise**: Based on observed residual std (~0.11)

### Expected Behavior

**If model is correct**:
- Posterior for β should remain positive and significant
- GP amplitude η should be << β (trend dominates)
- GP should show structure in middle range, less at extremes
- Predictive uncertainty should increase in gap region

**Parameter Interpretations**:
- **α**: Baseline Y at x=1 (exp(0)=1)
- **β**: Saturation rate (how fast Y increases per log-unit of x)
- **η**: Magnitude of systematic deviations from log trend
- **ρ**: Spatial correlation scale (how locally similar are deviations)
- **σ_noise**: Pure measurement error

### Falsification Criteria

**I will ABANDON this model if**:

1. **η posterior >> β**: GP dominates, meaning trend is wrong (switch to pure GP or different parametric form)
2. **Posterior α or β shifts drastically**: Priors conflict with likelihood (trend misspecified)
3. **GP shows high-frequency oscillations**: ρ → 0, indicating overfitting (simplify to parametric only)
4. **Poor predictive on replicates**: Can't capture within-replicate variance (need hierarchical structure)
5. **Computational failure**: GP inversion unstable (try sparse approximation or abandon GP)

**Red flags**:
- Divergent transitions in Stan (reparameterize or simplify)
- Very wide posterior credible intervals for trend parameters (trend not identifiable)
- GP predictions go wild in gap region (no information to constrain)

### Computational Implementation

**Stan Code** (Recommended):

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
}

parameters {
  real alpha;
  real<lower=0> beta;  // Enforce positivity
  real<lower=0> sigma_noise;
  real<lower=0> eta;
  real<lower=0> rho;
  vector[N] z_gp;  // Non-centered parameterization
}

transformed parameters {
  vector[N] f_smooth;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(x, eta, rho);

    // Add jitter for numerical stability
    for (n in 1:N) {
      K[n, n] = K[n, n] + 1e-9;
    }

    L_K = cholesky_decompose(K);
    f_smooth = L_K * z_gp;
  }
}

model {
  vector[N] mu = alpha + beta * log_x + f_smooth;

  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  sigma_noise ~ normal(0, 0.15);
  eta ~ normal(0, 0.1);
  rho ~ inv_gamma(5, 5);
  z_gp ~ std_normal();  // Non-centered

  // Likelihood
  Y ~ normal(mu, sigma_noise);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha + beta * log(x[n]) + f_smooth[n];
    Y_rep[n] = normal_rng(mu_n, sigma_noise);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma_noise);
  }
}
```

**Computational Considerations**:
- **GP computational cost**: O(N³) for N=27 is trivial (~20k flops)
- **Non-centered parameterization**: z_gp helps with sampling efficiency
- **Cholesky stability**: Add small jitter (1e-9) to diagonal
- **Expected runtime**: ~30 seconds for 4000 iterations

**Alternative: PyMC Implementation** (if Stan fails):
```python
import pymc as pm
import numpy as np

with pm.Model() as additive_model:
    # Priors
    alpha = pm.Normal("alpha", mu=1.75, sigma=0.5)
    beta = pm.Normal("beta", mu=0.27, sigma=0.15)
    sigma_noise = pm.HalfNormal("sigma_noise", sigma=0.15)

    # GP component
    eta = pm.HalfNormal("eta", sigma=0.1)
    rho = pm.InverseGamma("rho", alpha=5, beta=5)
    cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=rho)
    gp = pm.gp.Latent(cov_func=cov_func)
    f_smooth = gp.prior("f_smooth", X=x[:, None])

    # Mean function
    mu = alpha + beta * np.log(x) + f_smooth

    # Likelihood
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_noise, observed=Y)

    # Posterior sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.95)
```

### Success Criteria

**Model is SUCCESSFUL if**:
- LOO-CV score improves over pure logarithmic model by >5 (meaningful)
- GP component shows interpretable structure (not just noise fitting)
- Predictive intervals contain ~95% of held-out data
- Posterior convergence: Rhat < 1.01, ESS > 400 for all parameters
- Predictions in gap x∈[23,29] are reasonable (Y∈[2.4, 2.6])

---

## Model Class 2: Hierarchical Replicate Model

### Conceptual Framework

**Hypothesis**: The 6 x-values with replicates represent **true grouping structure**:
- **Between-group variance**: Systematic variation in mean Y at each x
- **Within-group variance**: Measurement error or true biological variability

This is a **hierarchical mixed-effects model** where replicate groups share information.

### Mathematical Specification

```
Likelihood:
Y_ij ~ Normal(μ_ij, σ_within)
μ_ij = f(x_j) + u_j

Where:
- j indexes x-value groups (20 unique x values)
- i indexes replicates within group (1 to n_j)
- f(x) is the population-level trend
- u_j is group-level random effect

Population trend:
f(x) = α + β·log(x)

Group-level effects:
u_j ~ Normal(0, σ_between)   for groups with replicates
u_j = 0                       for singleton observations

Priors:
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
σ_between ~ HalfNormal(0.1)    # Between-group variability
σ_within ~ HalfNormal(0.1)     # Within-group variability
```

### Rationale

**Why hierarchical structure?**
- **Explicit replicate modeling**: 21/27 observations are replicates - this structure should be leveraged
- **Partial pooling**: Groups with replicates inform variance estimate for all predictions
- **Variance decomposition**: Separates systematic group variation from pure noise
- **Realistic uncertainty**: Acknowledges that repeated measures at same x may differ

**EDA Support**:
- Replicate std varies from 0.019 (x=5) to 0.157 (x=15.5)
- Mean replicate std ≈ 0.07, suggesting σ_within ≈ 0.05-0.10
- If σ_between << σ_within, replicates are pure noise (model reduces to simple regression)
- If σ_between ≈ σ_within, true grouping structure exists

**Prior Justification**:
- **σ_between prior**: Centered small (0.1) because replicates should cluster
- **σ_within prior**: Similar scale, based on observed replicate variability
- **Population trend**: Same as Model 1 (logarithmic saturation)

### Expected Behavior

**If model is correct**:
- σ_between posterior should be < σ_within (measurement noise dominates)
- Groups with high replicate variance (x=15.5) should have larger u_j posterior width
- Singleton observations should "borrow strength" from groups via f(x)
- Predictions at new x should use population trend + uncertainty from both variance components

**Parameter Interpretations**:
- **α, β**: Same as Model 1 (population-level trend)
- **σ_between**: How much systematic variation exists between "replicates" at same x
- **σ_within**: Pure measurement error or unmodeled individual variation
- **u_j**: Group-specific deviation from population trend

### Falsification Criteria

**I will ABANDON this model if**:

1. **σ_between ≈ 0**: No group structure, replicates are pure noise (simplify to Model 1 or 3)
2. **σ_between >> σ_within**: Replicates aren't true replicates (batches? experimental conditions? need covariates)
3. **u_j posteriors don't correlate with replicate variance**: Model structure is wrong
4. **Population trend (f(x)) poorly estimated**: Hierarchical structure is conflating levels
5. **Computational issues**: Many divergences, poor mixing (overparameterized)

**Red flags**:
- Posterior for σ_between has mode at 0 (variance component not identified)
- Group effects u_j show systematic trends with x (missing covariate or wrong functional form)
- Predictive performance worse than simple regression (unnecessary complexity)

### Computational Implementation

**Stan Code** (Recommended):

```stan
data {
  int<lower=1> N;                      // Total observations
  int<lower=1> J;                      // Number of unique x groups
  vector[N] Y;                         // Response
  int<lower=1,upper=J> group_id[N];   // Group membership
  vector[J] x_group;                   // Unique x values
  int<lower=0,upper=1> has_replicates[J];  // Indicator: 1 if group has replicates
}

transformed data {
  vector[J] log_x_group = log(x_group);
}

parameters {
  real alpha;
  real<lower=0> beta;
  real<lower=0> sigma_between;
  real<lower=0> sigma_within;
  vector[J] u_raw;  // Non-centered group effects
}

transformed parameters {
  vector[J] u;

  // Group effects: only for groups with replicates
  for (j in 1:J) {
    if (has_replicates[j] == 1) {
      u[j] = sigma_between * u_raw[j];
    } else {
      u[j] = 0;
    }
  }
}

model {
  vector[N] mu;

  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  sigma_between ~ normal(0, 0.1);
  sigma_within ~ normal(0, 0.1);
  u_raw ~ std_normal();  // Non-centered

  // Mean model
  for (n in 1:N) {
    mu[n] = alpha + beta * log_x_group[group_id[n]] + u[group_id[n]];
  }

  // Likelihood
  Y ~ normal(mu, sigma_within);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;
  real<lower=0,upper=1> ICC;  // Intraclass correlation

  // Intraclass correlation: proportion of variance between groups
  ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2);

  for (n in 1:N) {
    real mu_n = alpha + beta * log_x_group[group_id[n]] + u[group_id[n]];
    Y_rep[n] = normal_rng(mu_n, sigma_within);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma_within);
  }
}
```

**Data Preparation** (Python):
```python
import numpy as np
import pandas as pd

# Prepare data for Stan
df = pd.DataFrame({'x': x, 'Y': Y})
df['group_id'] = pd.factorize(df['x'])[0] + 1  # 1-indexed for Stan

group_info = df.groupby('group_id').agg({
    'x': 'first',
    'Y': 'count'
}).rename(columns={'Y': 'n_replicates'})

group_info['has_replicates'] = (group_info['n_replicates'] > 1).astype(int)

stan_data = {
    'N': len(df),
    'J': len(group_info),
    'Y': df['Y'].values,
    'group_id': df['group_id'].values,
    'x_group': group_info['x'].values,
    'has_replicates': group_info['has_replicates'].values
}
```

**Computational Considerations**:
- **Non-centered parameterization**: Crucial for small σ_between (avoids funnel)
- **J=20 groups**: Small number, hierarchical benefits limited but present
- **Expected runtime**: ~1 minute for 4000 iterations
- **Convergence**: May need target_accept=0.95 due to hierarchical structure

### Success Criteria

**Model is SUCCESSFUL if**:
- ICC > 0.1 (meaningful between-group variance)
- σ_between and σ_within both identified (posteriors away from 0)
- Group effects u_j show reasonable patterns (not wildly varying)
- LOO-CV improves over non-hierarchical model
- Rhat < 1.01, ESS > 400 for all parameters

---

## Model Class 3: Compositional Variance Model

### Conceptual Framework

**Hypothesis**: Total variance has **two components**:
1. **Systematic heteroscedasticity**: Variance changes with x (decreasing as observed)
2. **Residual noise**: Constant measurement error

This is a **location-scale model** that explicitly models both mean and variance as functions of x.

### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(μ_i, σ_i)
μ_i = α + β·log(x_i)
log(σ_i) = γ_0 + γ_1·log(x_i)

Priors:
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
γ_0 ~ Normal(log(0.12), 0.5)    # Log-scale intercept
γ_1 ~ Normal(0, 0.3)            # Variance trend with x
```

### Rationale

**Why model variance structure?**
- **EDA evidence**: 4.6:1 variance ratio (low vs high x), clear decreasing trend
- **Statistical significance**: BP test non-significant, but n=27 has low power
- **Better uncertainty**: If heteroscedasticity exists, constant variance underestimates uncertainty at low x
- **Physical plausibility**: Many processes have variance decrease with maturity/saturation

**Variance Parameterization**:
- **Log-scale**: Ensures σ_i > 0 without constraints
- **log(x) covariate**: Matches mean function scale
- **γ_1 < 0**: Would indicate decreasing variance with increasing x
- **γ_1 = 0**: Reduces to constant variance (model comparison tests this)

**Prior Justification**:
- **γ_0 centered at log(0.12)**: Matches overall residual std ≈ 0.12
- **γ_1 centered at 0**: Weakly informative, allows increase or decrease
- **γ_1 std = 0.3**: Moderate uncertainty (on log scale)

### Expected Behavior

**If model is correct**:
- γ_1 posterior should be < 0 (variance decreasing with x)
- Credible interval for γ_1 should exclude 0 if heteroscedasticity is real
- Variance estimates should follow observed pattern: higher at low x
- Predictions should have wider intervals at low x, narrower at high x

**Parameter Interpretations**:
- **α, β**: Same as previous models (mean trend)
- **γ_0**: Baseline log-variance at x=1
- **γ_1**: Rate of variance change per log-unit of x
- **exp(γ_0 + γ_1·log(x))**: Predicted standard deviation at x

### Falsification Criteria

**I will ABANDON this model if**:

1. **γ_1 credible interval includes 0**: No evidence for heteroscedasticity (use constant variance)
2. **Posterior γ_1 > 0**: Variance increases with x (contradicts EDA - wrong functional form)
3. **Very wide posterior for γ_1**: Variance structure not identifiable with n=27
4. **Poor predictive on replicates**: Can't capture within-replicate variance patterns
5. **Model comparison favors constant variance**: LOO/WAIC decisively prefer simpler model

**Red flags**:
- Extreme variance predictions (σ_i → 0 or σ_i → ∞)
- Variance function crosses observed replicate variances systematically
- Computational issues (log-scale variance poorly identified)

### Computational Implementation

**Stan Code** (Recommended):

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
}

parameters {
  real alpha;
  real<lower=0> beta;
  real gamma_0;
  real gamma_1;
}

transformed parameters {
  vector[N] mu;
  vector[N] sigma;

  for (n in 1:N) {
    mu[n] = alpha + beta * log_x[n];
    sigma[n] = exp(gamma_0 + gamma_1 * log_x[n]);
  }
}

model {
  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  gamma_0 ~ normal(log(0.12), 0.5);
  gamma_1 ~ normal(0, 0.3);

  // Likelihood with heteroscedastic variance
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    Y_rep[n] = normal_rng(mu[n], sigma[n]);
    log_lik[n] = normal_lpdf(Y[n] | mu[n], sigma[n]);
  }
}
```

**Computational Considerations**:
- **Simple model**: No hierarchical structure, fast sampling
- **Log-variance**: Avoids constrained parameters, aids HMC
- **Expected runtime**: ~20 seconds for 4000 iterations
- **Convergence**: Should be straightforward, no special tuning needed

**Model Comparison Setup**:
```python
# Fit both constant and varying variance models
models = {
    'constant_variance': fit_model_1(),  # σ constant
    'varying_variance': fit_model_3()     # σ(x)
}

# Compare using LOO-CV
import arviz as az
comparison = az.compare(models, ic='loo')

# Check if ΔLOO > 4 (meaningful difference)
if comparison.iloc[1]['loo'] - comparison.iloc[0]['loo'] > 4:
    print("Heteroscedasticity model preferred")
```

### Success Criteria

**Model is SUCCESSFUL if**:
- γ_1 posterior 95% CI excludes 0 (evidence for heteroscedasticity)
- LOO-CV improves over constant variance by ΔLOO > 4
- Variance predictions match replicate variances reasonably
- Predictive intervals better calibrated (contain ~95% of data)
- Rhat < 1.01, ESS > 400 for all parameters

---

## Model Prioritization and Selection Strategy

### Priority Ranking

**1. Model 2 (Hierarchical Replicate) - HIGHEST PRIORITY**

**Reasoning**:
- **Directly addresses data structure**: 21/27 observations are replicates
- **Most scientifically justified**: Replicates likely represent true experimental structure
- **Provides variance decomposition**: Separates biological from measurement variation
- **Conservative**: If σ_between ≈ 0, reduces to simple regression (model comparison tests necessity)

**Expected outcome**: This model should reveal whether replicates contain information or are pure noise.

---

**2. Model 3 (Compositional Variance) - MEDIUM PRIORITY**

**Reasoning**:
- **EDA evidence exists**: 4.6:1 variance ratio, clear visual trend
- **Low statistical power**: n=27 may not detect heteroscedasticity even if present
- **Important for uncertainty**: If real, constant variance model gives wrong prediction intervals
- **Easy to test**: Model comparison directly assesses if complexity is warranted

**Expected outcome**: Posterior for γ_1 will likely include 0, but magnitude will inform future experiments.

---

**3. Model 1 (Additive Decomposition) - LOWEST PRIORITY**

**Reasoning**:
- **Most complex**: GP component adds significant complexity
- **Weakest justification**: No strong evidence for structured deviations from log trend
- **Computational risk**: GP with small n can overfit
- **Gap interpolation**: Only clear advantage over simpler models

**Expected outcome**: GP component will likely be small (η ≈ 0), suggesting simpler model sufficient.

---

### Decision Tree for Model Selection

```
START: Fit all three models in parallel

CHECK 1: Hierarchical structure (Model 2)
├─ If ICC > 0.1 AND σ_between CI excludes 0
│  └─> Use Model 2 as base, consider adding variance structure
├─ If ICC ≈ 0
│  └─> Replicates are pure noise, continue with Models 1 or 3

CHECK 2: Variance structure (Model 3)
├─ If γ_1 CI excludes 0 AND ΔLOO > 4
│  └─> Use heteroscedastic variance in final model
├─ If γ_1 CI includes 0
│  └─> Use constant variance (simpler)

CHECK 3: Additive structure (Model 1)
├─ If η posterior >> 0 AND LOO improves
│  └─> Structured deviations exist, use additive model
├─ If η ≈ 0
│  └─> Parametric trend sufficient, use simpler model

FINAL MODEL: Combination based on above checks
- Most likely: Hierarchical with constant variance
- Alternative: Simple log trend with heteroscedastic variance
- Unlikely: Full additive decomposition needed
```

---

## Cross-Model Diagnostics and Stress Tests

### Diagnostic 1: Replicate Prediction Test

**Objective**: Which model best predicts held-out replicates?

**Method**:
1. For each x with replicates, hold out one observation
2. Fit model on remaining data
3. Check if prediction interval contains held-out point
4. Repeat for all replicates

**Success**: Model should capture ≥90% of held-out replicates (close to 95% nominal)

**Expected**:
- Model 2 should excel (explicit replicate structure)
- Models 1 and 3 should perform similarly
- If all fail, replicates contain structure not captured by x alone

---

### Diagnostic 2: Gap Interpolation Test

**Objective**: How reasonable are predictions in x∈[23,29]?

**Method**:
1. Fit each model
2. Predict Y at x = 25 (center of gap)
3. Compare prediction means and intervals

**Success**:
- Predictions should be Y ∈ [2.45, 2.55] (interpolating neighbors)
- Uncertainty should be higher than regions with data

**Expected**:
- Model 1 (GP) should have widest intervals (explicitly models uncertainty)
- Model 2 should have moderate intervals (hierarchical uncertainty)
- Model 3 should have narrowest intervals (no spatial correlation)

---

### Diagnostic 3: Influential Point Sensitivity

**Objective**: How sensitive are conclusions to x=31.5 point?

**Method**:
1. Fit each model with full data
2. Fit each model excluding x=31.5
3. Compare posterior means and widths for all parameters

**Success**:
- Parameter estimates should shift < 20%
- Qualitative conclusions (e.g., γ_1 < 0) should not change

**Expected**:
- All models somewhat sensitive (high leverage)
- Model 2 least sensitive (partial pooling provides robustness)
- Model 1 most sensitive (GP endpoint strongly influences predictions)

---

### Stress Test 1: Prior-Posterior Conflict

**Check for each model**:
```python
# Prior predictive
prior_pred = sample_prior_predictive(model, samples=1000)

# Posterior predictive
post_pred = sample_posterior_predictive(trace, model, samples=1000)

# Compare distributions
from scipy.stats import ks_2samp
ks_stat, p_value = ks_2samp(prior_pred['Y'].flatten(),
                            post_pred['Y'].flatten())

# If p < 0.01, strong prior-posterior conflict (model fighting data)
```

**Red flag**: KS statistic > 0.5 indicates major conflict (respecify priors or model)

---

### Stress Test 2: Posterior Predictive Calibration

**Check**: Do 95% prediction intervals contain 95% of data?

**Method**:
```python
# For each observation
coverage = []
for i in range(N):
    lower, upper = np.percentile(Y_rep[:, i], [2.5, 97.5])
    coverage.append(lower <= Y[i] <= upper)

empirical_coverage = np.mean(coverage)

# Should be 0.93-0.97 for well-calibrated model
```

**Red flag**: Coverage < 0.85 or > 0.99 indicates miscalibration

---

## Alternative Model Pivot Plans

### Pivot 1: If All Models Fail Replicate Test

**Symptom**: None of the models adequately capture replicate variability

**Diagnosis**: Replicates may represent **hidden grouping** (batches, experimental conditions, time)

**Pivot Strategy**:
- **Add batch effects**: If experimental metadata available
- **Random effects by replicate set**: Each set of replicates gets its own effect
- **Time-series structure**: If replicates were collected sequentially

**New Model** (Batch-aware hierarchical):
```
Y_ij ~ Normal(μ_ij, σ)
μ_ij = f(x_j) + batch_effect[b_j]
batch_effect[b] ~ Normal(0, σ_batch)
```

---

### Pivot 2: If Variance Models Fail

**Symptom**: Neither constant nor log-linear variance captures pattern

**Diagnosis**: Variance structure may be **non-monotonic** or have **regime change**

**Pivot Strategy**:
- **Piecewise variance**: Different σ for x < 15 vs x ≥ 15
- **Quadratic variance**: log(σ) = γ_0 + γ_1·log(x) + γ_2·log(x)²
- **Student-t likelihood**: Robust to outliers and heavy tails

**New Model** (Robust location-scale):
```
Y_i ~ StudentT(ν, μ_i, σ_i)
ν ~ Gamma(2, 0.1)  # Degrees of freedom (allows heavy tails)
```

---

### Pivot 3: If Trend Misspecified

**Symptom**: Large GP component (η >> β) or systematic residual patterns

**Diagnosis**: Logarithmic trend is **wrong functional form**

**Pivot Strategy**:
- **Try alternative saturating functions**: Michaelis-Menten, asymptotic exponential
- **Regime change model**: Different trend before/after breakpoint
- **Pure Gaussian Process**: Fully non-parametric (abandon parametric trend)

**New Model** (Michaelis-Menten):
```
μ_i = θ_max · x_i / (θ_half + x_i)

Priors:
θ_max ~ Normal(2.6, 0.2)     # Asymptotic Y value
θ_half ~ Gamma(5, 0.5)        # Half-saturation x value
```

---

## Implementation Roadmap

### Phase 1: Parallel Fitting (Day 1)

**Tasks**:
1. Implement all three models in Stan
2. Fit each independently with 4 chains, 2000 iterations (1000 warmup)
3. Check convergence diagnostics (Rhat, ESS, divergences)
4. Generate posterior predictive samples

**Deliverables**:
- Three fitted models with trace diagnostics
- Posterior summaries for all parameters
- Initial visual checks (trace plots, posterior densities)

---

### Phase 2: Model Comparison (Day 1-2)

**Tasks**:
1. Compute LOO-CV for each model
2. Compare ΔLOO (differences > 4 are meaningful)
3. Run three cross-model diagnostics
4. Run two stress tests

**Deliverables**:
- Model comparison table (LOO, WAIC, predictive accuracy)
- Diagnostic results (replicate prediction, gap interpolation, sensitivity)
- Stress test results (prior-posterior, calibration)

---

### Phase 3: Model Selection and Refinement (Day 2)

**Tasks**:
1. Select best model based on comparison
2. If needed, implement hybrid model (e.g., hierarchical + heteroscedastic)
3. Re-fit final model with extended sampling (4000 iterations)
4. Generate publication-quality posterior predictive checks

**Deliverables**:
- Final selected model with justification
- Comprehensive posterior predictive checks
- Prediction intervals across full x range
- Uncertainty decomposition (if hierarchical)

---

### Phase 4: Sensitivity and Reporting (Day 2-3)

**Tasks**:
1. Refit final model excluding x=31.5 (sensitivity check)
2. Explore alternative prior specifications
3. Generate predictions for held-out scenarios
4. Document findings and limitations

**Deliverables**:
- Sensitivity analysis report
- Prior sensitivity checks
- Final model specification document
- Recommendations for future data collection

---

## Stopping Rules and Success Criteria

### When to Stop and Declare Success

**Criteria for SUCCESS**:
1. At least one model achieves Rhat < 1.01, ESS > 400 for all parameters
2. Selected model has LOO-CV predictive accuracy RMSE < 0.15
3. Posterior predictive checks show good calibration (coverage 0.93-0.97)
4. Parameter posteriors are interpretable and scientifically plausible
5. Model comparison clearly favors one approach (ΔLOO > 4) OR multiple models perform similarly (no clear winner is also informative)

**If SUCCESS**: Document model, generate predictions, provide recommendations

---

### When to Stop and Pivot

**Criteria for PIVOT**:
1. All models show poor convergence (many divergences, low ESS)
2. All models fail replicate prediction test (< 85% coverage)
3. All models show major prior-posterior conflict
4. Selected model has implausible parameter estimates
5. No model provides better predictions than simple mean(Y) = 2.32

**If PIVOT**: Implement alternative model class (see Pivot Plans above)

---

### When to Stop and Abandon Bayesian Approach

**Criteria for ABANDON** (unlikely but important to define):
1. Sample size too small for any Bayesian model (n < 10 after quality checks)
2. Data quality issues discovered (e.g., systematic measurement errors)
3. Computational resources exhausted (Stan won't converge after extensive tuning)
4. All models give nonsensical predictions (indicates fundamental misunderstanding)

**If ABANDON**: Recommend data collection, exploratory analysis only, or frequentist approach

---

## Expected Outcomes and Uncertainty

### Most Likely Scenario

**Prediction**: Model 2 (Hierarchical Replicate) will be selected

**Reasoning**:
- Replicates are the most salient data structure feature
- Variance decomposition is scientifically valuable
- If ICC ≈ 0, model reduces gracefully to simple regression
- Conservative and interpretable

**Expected Posteriors**:
- α ∈ [1.6, 1.9], β ∈ [0.20, 0.34] (similar to EDA point estimates)
- σ_between ∈ [0.02, 0.08], σ_within ∈ [0.08, 0.14]
- ICC ∈ [0.02, 0.20] (small but non-zero between-group variance)

---

### Alternative Scenario

**Prediction**: Model 3 (Compositional Variance) selected, but γ_1 CI includes 0

**Reasoning**:
- Heteroscedasticity exists but n=27 insufficient to detect
- Model provides better prediction intervals even if not statistically significant
- Conservative approach: model the uncertainty we observe

**Expected Posteriors**:
- γ_1 ∈ [-0.4, 0.1] (wide CI, negative mode but includes 0)
- Variance decreases with x, but evidence is weak

---

### Surprising Scenario (Low Probability)

**Prediction**: Model 1 (Additive Decomposition) shows large GP component

**Interpretation**:
- Logarithmic trend is **wrong** - systematic deviations exist
- Possible regime change or missing covariate
- Need to investigate what GP is capturing

**Action**:
- Examine GP predictions spatially
- Look for breakpoints or non-monotonicity
- Consider pivot to alternative parametric form

---

## Summary: Key Falsification Criteria Across All Models

### Global Falsification (All Models)

**Abandon ALL models if**:
1. **Computational failure**: Stan won't sample after extensive tuning
2. **Nonsensical predictions**: All models predict Y outside [1, 3]
3. **Severe miscalibration**: All models have coverage < 0.80 or > 0.99
4. **Parameter non-identifiability**: All models have Rhat > 1.1 for key parameters

---

### Model-Specific Falsification

**Model 1** (Additive Decomposition):
- Abandon if η ≈ 0 (no additive structure)
- Abandon if GP dominates trend (wrong parametric form)

**Model 2** (Hierarchical Replicate):
- Abandon if σ_between ≈ 0 (no hierarchy needed)
- Abandon if σ_between >> σ_within (replicates aren't replicates)

**Model 3** (Compositional Variance):
- Abandon if γ_1 CI includes 0 AND LOO doesn't improve (no heteroscedasticity)
- Abandon if variance predictions contradict replicate observations

---

## Final Notes on Scientific Integrity

### What I'm Committing To

1. **Honest reporting**: If all models fail, I will say so clearly
2. **Model comparison transparency**: Will report LOO for all models, not just "winner"
3. **Uncertainty acknowledgment**: Will clearly state what is NOT identified with n=27
4. **Limitation documentation**: Will list all assumptions and where they might fail

### What Would Make Me Reconsider Everything

1. **High influential point sensitivity**: If removing x=31.5 changes conclusions drastically, need more data at high x
2. **Replicate structure anomaly**: If Model 2 shows σ_between >> σ_within, replicates represent something else
3. **Systematic GP structure**: If Model 1 GP shows clear patterns, trend is misspecified
4. **All models equivalent**: If ΔLOO < 2 for all pairs, data insufficient to distinguish models

### Success is Learning

- **Best outcome**: One model clearly preferred, parameters interpretable, predictions good
- **Good outcome**: Multiple models similar, but we learn what data CAN'T distinguish
- **Acceptable outcome**: All models inadequate, but we learn what's missing (more data? covariates?)
- **Failure outcome**: None - as long as I report honestly, we learn something valuable

---

## File Outputs for Implementation

### Code Files to Create

1. `/workspace/experiments/designer_3/model_1_additive.stan` - Additive decomposition Stan code
2. `/workspace/experiments/designer_3/model_2_hierarchical.stan` - Hierarchical replicate Stan code
3. `/workspace/experiments/designer_3/model_3_variance.stan` - Compositional variance Stan code
4. `/workspace/experiments/designer_3/fit_all_models.py` - Python driver script
5. `/workspace/experiments/designer_3/diagnostics.py` - Cross-model diagnostic functions
6. `/workspace/experiments/designer_3/comparison.py` - Model comparison and selection

### Expected Output Files

1. `model_comparison_table.csv` - LOO/WAIC comparison
2. `posterior_summaries.txt` - Parameter estimates for all models
3. `diagnostic_results.json` - Results from all diagnostic tests
4. `final_model_selection.md` - Justification and specification
5. `predictions_with_uncertainty.csv` - Predictions across x range

---

**END OF PROPOSAL - Designer 3: Hierarchical/Compositional Perspective**

---

## Appendix: Quick Reference

### Model Comparison at a Glance

| Feature | Model 1 (Additive) | Model 2 (Hierarchical) | Model 3 (Variance) |
|---------|-------------------|----------------------|-------------------|
| **Complexity** | High (GP) | Medium (random effects) | Low (location-scale) |
| **Parameters** | 5 + N (GP) | 4 + J (group effects) | 4 |
| **Target** | Structured deviations | Replicate structure | Heteroscedasticity |
| **Computational** | O(N³) for GP | O(J) hierarchical | O(N) standard |
| **Falsifiable** | η ≈ 0 | ICC ≈ 0 | γ₁ includes 0 |
| **Priority** | Lowest | Highest | Medium |

### Prior Summary

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| α | Normal(1.75, 0.5) | EDA intercept estimate |
| β | Normal(0.27, 0.15) | EDA slope estimate |
| σ_noise | HalfNormal(0.15) | Residual scale ~0.12 |
| η | HalfNormal(0.1) | GP smaller than trend |
| ρ | InverseGamma(5, 5) | Moderate smoothness |
| σ_between | HalfNormal(0.1) | Small group effects |
| σ_within | HalfNormal(0.1) | Replicate variability |
| γ_0 | Normal(log(0.12), 0.5) | Log-variance intercept |
| γ_1 | Normal(0, 0.3) | Variance trend |

### Decision Thresholds

- **Meaningful LOO difference**: ΔLOO > 4
- **Convergence**: Rhat < 1.01, ESS > 400
- **Calibration**: Coverage ∈ [0.93, 0.97]
- **ICC threshold**: ICC > 0.1 (meaningful hierarchy)
- **Sensitivity**: Parameter shift < 20% without x=31.5
