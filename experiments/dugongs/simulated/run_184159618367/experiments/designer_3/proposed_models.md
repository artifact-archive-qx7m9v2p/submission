# Bayesian Model Design: Designer 3
## Alternative Approaches - Transformations, Robust Methods, and Simplicity

**Designer Focus**: Practical, robust models emphasizing transformation approaches, alternative likelihoods, and simple effective solutions

**Design Philosophy**: Start with simple, well-understood models that are robust to assumptions. Use transformations to achieve linearity when possible. Build in robustness through likelihood choice.

---

## Critical Context from EDA

### Data Pattern
- **N = 27** observations, excellent quality, 6 replicated x-values
- **Clear saturation**: Rapid increase (x: 1-10), plateau (x > 10)
- **Nonlinearity confirmed**: Linear R² = 0.52, Nonlinear R² = 0.81-0.90
- **Log-log transformation**: Achieves r = 0.92 (strong linearity)
- **Pure error estimate**: σ ≈ 0.075-0.12 from replicates
- **Regime shift**: Breakpoint near x = 9.5

### Strategic Insight
The log-log transformation achieving r=0.92 is a major finding. This suggests:
1. Power law structure may be fundamental
2. Transformation enables simple Gaussian inference
3. Heteroscedasticity may exist on original scale but not log-scale

---

## Model Design Strategy

### Philosophy: Robustness Through Simplicity
1. **Start with transformations** - if data becomes linear, use simple models
2. **Build in robustness** - Student-t likelihoods protect against outliers
3. **Polynomial pragmatism** - quadratic is simple, effective, interpretable
4. **Test assumptions explicitly** - validate homoscedasticity, normality

### Falsification Mindset
Each model has explicit failure modes and escape routes:
- **If transformation models fail**: Return to nonlinear models (Designer 1/2)
- **If homoscedasticity fails**: Switch to heteroscedastic models
- **If Gaussian fails**: Student-t or alternative error structures
- **If all fail**: Consider mixture models or changepoint models

---

## MODEL 1: Log-Log Power Law (Primary Recommendation)

### Model Specification

```
# Transformed scale (log-log)
log(Y_i) ~ Normal(μ_i, σ)
μ_i = α + β * log(x_i)

# Equivalent original scale
Y_i = exp(α) * x_i^β * ε_i,  where log(ε_i) ~ Normal(0, σ)
```

**Priors**:
```
α ~ Normal(0.6, 0.3)      # Centered at log(1.8) ≈ 0.59
β ~ Normal(0.12, 0.1)     # Centered at OLS estimate, weakly informative
σ ~ Half-Cauchy(0, 0.1)   # Log-scale residual variation
```

**Prior Justification**:
- α: log(Y) ranges [0.54, 0.97], center prior at 0.6 with SD covering plausible range
- β: OLS log-log fit gives β ≈ 0.121, prior allows [−0.08, 0.32] at 95%
- σ: On log scale, tight variation expected; half-Cauchy provides robustness

### Theoretical Justification

**Why this model?**
1. **Empirical**: Log-log transformation achieves r=0.92 (strongest linearity observed)
2. **Theoretical**: Power laws ubiquitous in natural phenomena (allometry, scaling)
3. **Mathematical**: Transforms nonlinear problem to simple linear regression
4. **Computational**: Fast inference, no nonlinear optimization in likelihood
5. **Robust**: Multiplicative errors often more realistic than additive

**Power Law Interpretation**:
- Y = 1.798 * x^0.121 (from EDA OLS fit)
- β = 0.121: 10% increase in x → 1.2% increase in Y
- Diminishing returns naturally emerge from 0 < β < 1

**Scientific Plausibility**:
Power laws appear when:
- Systems exhibit scale-free properties
- Cumulative processes with diminishing returns
- Biological growth/scaling relationships
- Saturation via multiplicative rather than additive constraints

### Falsification Criteria

**I will abandon this model if**:
1. **Residual patterns on log-scale**: If log(Y) residuals show systematic curvature
2. **Heteroscedasticity remains**: If log-scale variance is not constant
3. **Poor back-transformation**: If exp(predictions) systematically miss observed Y
4. **Extreme extrapolation bias**: If low-x or high-x predictions are systematically off
5. **Prior-posterior conflict**: If posterior pushes α or β far from reasonable ranges

**Red Flags**:
- Pareto-k > 0.7 for multiple observations (influential points)
- Posterior predictive p-value < 0.05 for variance test
- Systematic bias in replicated x-values (should average out)
- LOO R² < 0.80 (worse than nonlinear models)

**Stress Test**:
- **Test**: Predict at x = 0.5 and x = 50 (extrapolation)
- **Expectation**: Wide uncertainty, but monotonic increase
- **Failure**: If predictions go negative or explode unrealistically

### Expected Performance

**Best Case**:
- R² ≈ 0.81 (EDA log-log R²)
- RMSE ≈ 0.12 (back-transformed)
- Converges in < 1000 iterations
- All Pareto-k < 0.5

**Realistic**:
- R² = 0.78-0.82 (accounting for bias in exp transformation)
- RMSE = 0.12-0.14
- Minor Jensen's inequality correction needed for predictions

**Worst Case (still acceptable)**:
- R² = 0.75 (if back-transformation introduces bias)
- Heteroscedasticity on original scale (but acceptable if log-scale is good)

**Comparison to Other Models**:
- Likely worse than asymptotic/piecewise (R² ~ 0.89-0.90)
- But simpler, faster, more robust
- If ΔLOO < 2 SE, prefer this for parsimony

### Implementation Notes

#### Stan Implementation
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
  vector[N] log_Y = log(Y);
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(0.6, 0.3);
  beta ~ normal(0.12, 0.1);
  sigma ~ cauchy(0, 0.1);

  // Likelihood on log-log scale
  log_Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] Y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    real mu_log = alpha + beta * log_x[i];
    Y_pred[i] = exp(mu_log + 0.5 * sigma^2);  // Bias correction
    log_lik[i] = normal_lpdf(log_Y[i] | mu_log, sigma);
  }
}
```

**Key Considerations**:
1. **Bias correction**: exp(E[log Y]) ≠ E[Y]; add 0.5*σ² for median prediction
2. **Back-transformation**: Generate predictions on both log and original scale
3. **LOO calculation**: Compute on log-scale likelihood for proper scoring
4. **Zero handling**: Ensure x > 0 (satisfied in this dataset)

#### PyMC Implementation
```python
import pymc as pm

with pm.Model() as log_log_model:
    # Data
    log_x = pm.Data('log_x', np.log(x_obs))
    log_Y = pm.Data('log_Y', np.log(Y_obs))

    # Priors
    alpha = pm.Normal('alpha', mu=0.6, sigma=0.3)
    beta = pm.Normal('beta', mu=0.12, sigma=0.1)
    sigma = pm.HalfCauchy('sigma', beta=0.1)

    # Linear model on log-log scale
    mu_log = alpha + beta * log_x

    # Likelihood
    Y_obs_log = pm.Normal('Y_obs', mu=mu_log, sigma=sigma, observed=log_Y)

    # Predictions with bias correction
    Y_pred = pm.Deterministic('Y_pred', pm.math.exp(mu_log + 0.5 * sigma**2))

    # Sample
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)
```

**Computational Expectations**:
- **Speed**: Very fast (linear model in parameters)
- **Convergence**: Should converge in < 500 iterations
- **ESS**: Expect ESS > 1000 for all parameters
- **Warnings**: None expected (well-behaved geometry)

---

## MODEL 2: Robust Quadratic with Student-t Likelihood

### Model Specification

```
Y_i ~ Student-t(ν, μ_i, σ)
μ_i = β₀ + β₁*x_i + β₂*x_i²
```

**Priors**:
```
β₀ ~ Normal(1.75, 0.3)        # Intercept near observed low-x Y
β₁ ~ Normal(0.08, 0.05)       # Positive slope, weakly informative
β₂ ~ Normal(-0.002, 0.001)    # Negative curvature, weakly informative
σ ~ Half-Cauchy(0, 0.15)      # Residual scale
ν ~ Gamma(2, 0.1)             # Degrees of freedom (mean ≈ 20, allows heavy tails)
```

**Prior Justification**:
- β₀: Y-intercept extrapolated to x=0 from EDA is ~1.75
- β₁, β₂: Centered at OLS quadratic estimates from EDA
- ν: Gamma(2, 0.1) weakly favors ν ≈ 20 (near-Gaussian) but allows ν < 10 (heavy tails)

### Theoretical Justification

**Why this model?**
1. **Simplicity**: Quadratic is simplest nonlinear form, widely understood
2. **Flexibility**: Captures curvature with only 3 parameters
3. **Robustness**: Student-t likelihood protects against outliers (e.g., x=31.5)
4. **Fast inference**: Still linear in β parameters, only ν adds complexity
5. **Interpretable**: Coefficients have clear meaning (diminishing slope via β₂)

**When to prefer this**:
- If interpretability > perfect fit
- If outlier robustness is priority
- If computational speed matters (e.g., for cross-validation)
- If stakeholders understand polynomials

**Quadratic Saturation**:
- β₂ < 0 ensures downward curvature
- Peak occurs at x = -β₁/(2β₂) ≈ 20 (well within data range)
- Saturation emerges naturally from parabolic shape

### Falsification Criteria

**I will abandon this model if**:
1. **Cubic needed**: If residuals show systematic cubic pattern (S-shape)
2. **Non-monotonic**: If quadratic predicts Y decreasing at high x (unphysical)
3. **Poor tail behavior**: If Student-t ν posterior hits prior boundary (ν → ∞ or ν → 2)
4. **Underfitting**: If LOO R² < 0.80 (clearly worse than alternatives)
5. **Heteroscedasticity**: If residual variance increases with x despite Student-t

**Red Flags**:
- Posterior for ν concentrated at ν < 4 (extremely heavy tails needed)
- Posterior for β₂ includes 0 (no curvature detected)
- Predictions decrease at high x (quadratic turns down)
- Poor fit at mid-range x (7 < x < 13)

**Stress Test**:
- **Test**: Leave out all x > 20 and predict
- **Expectation**: Reasonable extrapolation with wide uncertainty
- **Failure**: If predictions go negative or wildly off

### Expected Performance

**Best Case**:
- R² ≈ 0.86 (EDA quadratic R² = 0.862)
- RMSE ≈ 0.10
- ν posterior centered at 15-25 (near Gaussian)
- No influential points (all Pareto-k < 0.5)

**Realistic**:
- R² = 0.83-0.86
- RMSE = 0.10-0.12
- ν = 10-30 (robust but not extreme)
- 1-2 points with Pareto-k ∈ [0.5, 0.7]

**Worst Case (still acceptable)**:
- R² = 0.80 (slight underfitting)
- ν < 10 (heavy tails needed, suggests outliers)
- Systematic residual pattern at extremes

**Comparison to Models**:
- Likely worse than asymptotic model (~0.89 R²)
- Comparable to log-log model (0.81 R²)
- **Advantage**: Robustness and simplicity
- **Disadvantage**: Less theoretically motivated

### Implementation Notes

#### Stan Implementation
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] x_sq = x .* x;  // Precompute x²
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> sigma;
  real<lower=1> nu;  // DOF constrained > 1
}

model {
  // Priors
  beta_0 ~ normal(1.75, 0.3);
  beta_1 ~ normal(0.08, 0.05);
  beta_2 ~ normal(-0.002, 0.001);
  sigma ~ cauchy(0, 0.15);
  nu ~ gamma(2, 0.1);

  // Likelihood
  Y ~ student_t(nu, beta_0 + beta_1 * x + beta_2 * x_sq, sigma);
}

generated quantities {
  vector[N] Y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    real mu = beta_0 + beta_1 * x[i] + beta_2 * x_sq[i];
    Y_pred[i] = mu;  // Mean prediction
    log_lik[i] = student_t_lpdf(Y[i] | nu, mu, sigma);
  }
}
```

**Key Considerations**:
1. **ν constraint**: Must have ν > 1 for variance to exist; ν > 2 for mean to exist
2. **Reparameterization**: If convergence issues, center x to reduce correlation between β₀ and β₁
3. **Posterior predictive**: Use student_t_rng() for full uncertainty including tail risk
4. **Comparison to Gaussian**: Run both, compare via LOO to assess need for robustness

#### PyMC Implementation
```python
import pymc as pm

with pm.Model() as robust_quadratic_model:
    # Data
    x_data = pm.Data('x', x_obs)

    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.75, sigma=0.3)
    beta_1 = pm.Normal('beta_1', mu=0.08, sigma=0.05)
    beta_2 = pm.Normal('beta_2', mu=-0.002, sigma=0.001)
    sigma = pm.HalfCauchy('sigma', beta=0.15)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # Mean function
    mu = beta_0 + beta_1 * x_data + beta_2 * x_data**2

    # Likelihood
    Y_obs = pm.StudentT('Y_obs', nu=nu, mu=mu, sigma=sigma, observed=Y_obs)

    # Sample
    trace = pm.sample(2000, tune=1000, target_accept=0.95)
```

**Computational Expectations**:
- **Speed**: Moderate (Student-t slower than Gaussian)
- **Convergence**: May need target_accept=0.95 for ν
- **ESS**: Expect ESS > 500 for all parameters
- **Warnings**: Possible divergences if ν has poor initialization

**Centering Trick** (if needed):
```python
# Center x to reduce correlation
x_mean = x_obs.mean()
x_centered = x_obs - x_mean
mu = beta_0 + beta_1 * x_centered + beta_2 * x_centered**2
```

---

## MODEL 3: Logarithmic Model with Heteroscedastic Errors

### Model Specification

```
Y_i ~ Normal(μ_i, σ_i)
μ_i = β₀ + β₁ * log(x_i + 1)  # +1 handles x=0 if needed
σ_i = σ₀ * exp(α * x_i)       # Variance grows with x
```

**Priors**:
```
β₀ ~ Normal(1.70, 0.2)        # Intercept at x→0
β₁ ~ Normal(0.65, 0.2)        # Log slope from EDA
σ₀ ~ Half-Cauchy(0, 0.1)      # Baseline residual SD
α ~ Normal(0, 0.01)           # Heteroscedasticity parameter (0 = homoscedastic)
```

**Prior Justification**:
- β₀, β₁: From EDA log model Y = 1.754 + 0.648*log(x)
- σ₀: Similar to replicate SD (~0.075-0.12)
- α: Weakly informative, centered at 0 (homoscedastic null)

### Theoretical Justification

**Why this model?**
1. **EDA support**: Log model achieved R² = 0.829 (third best in EDA)
2. **Simple transformation**: Logarithmic is simpler than exponential
3. **Parsimony**: Only 2 parameters for mean function
4. **Theoretical**: Log models common for diminishing returns (economics, psychology)
5. **Heteroscedasticity**: Explicitly models potential variance increase

**When to prefer this**:
- If simplicity is paramount (2 parameters vs 3)
- If log interpretation is natural for domain
- If variance clearly increases with x
- If need alternative to power law

**Scientific Interpretation**:
- β₁ = 0.65: Doubling x increases Y by 0.65*0.693 ≈ 0.45 units
- Logarithmic returns: Equal percentage changes in x → equal absolute changes in Y
- Natural for perception, dose-response, stimulus-response

### Falsification Criteria

**I will abandon this model if**:
1. **Underfitting**: R² < 0.75 (clearly worse than alternatives)
2. **Residual curvature**: Systematic pattern remains after log transformation
3. **Homoscedasticity sufficient**: If α posterior includes 0 with high probability
4. **Poor high-x fit**: If model fails to capture plateau adequately
5. **Computational issues**: If heteroscedastic model doesn't converge

**Red Flags**:
- α posterior far from 0 (e.g., |α| > 0.05) suggests wrong variance model
- Systematic residuals at low x or high x
- LOO R² < 0.75
- Multiple Pareto-k > 0.7

**Stress Test**:
- **Test**: Predict at x = 100 (extreme extrapolation)
- **Expectation**: Continued slow increase, wide uncertainty
- **Failure**: If predictions unbounded or variance explodes

**Decision Point**:
If α posterior is tightly centered at 0, **switch to homoscedastic version** (simpler):
```
Y_i ~ Normal(β₀ + β₁ * log(x_i), σ)
```
This is a clear pivot - use LOO comparison to decide.

### Expected Performance

**Best Case**:
- R² ≈ 0.83 (EDA log R²)
- RMSE ≈ 0.11-0.12
- α ≈ 0 (homoscedastic sufficient)
- Converges easily

**Realistic**:
- R² = 0.80-0.83
- RMSE = 0.12-0.14
- Small but nonzero α (mild heteroscedasticity)
- Good fit at low x, slight underfitting at plateau

**Worst Case (still acceptable)**:
- R² = 0.75-0.80
- α far from 0 but variance model helps
- Worse than quadratic/log-log but still reasonable

**Comparison to Models**:
- Likely worse than log-log (0.81) and quadratic (0.86)
- **Advantage**: Explicit variance modeling
- **Disadvantage**: May not capture plateau as well

### Implementation Notes

#### Stan Implementation
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] log_x = log(x + 1);  // +1 for numerical safety
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma_0;
  real alpha;  // Heteroscedasticity parameter
}

model {
  vector[N] mu;
  vector[N] sigma;

  // Priors
  beta_0 ~ normal(1.70, 0.2);
  beta_1 ~ normal(0.65, 0.2);
  sigma_0 ~ cauchy(0, 0.1);
  alpha ~ normal(0, 0.01);

  // Mean and variance functions
  mu = beta_0 + beta_1 * log_x;
  sigma = sigma_0 * exp(alpha * x);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    real mu = beta_0 + beta_1 * log_x[i];
    real sigma = sigma_0 * exp(alpha * x[i]);
    Y_pred[i] = mu;
    log_lik[i] = normal_lpdf(Y[i] | mu, sigma);
  }
}
```

**Key Considerations**:
1. **Variance explosion**: Monitor exp(α*x) to ensure σ doesn't explode at high x
2. **Identifiability**: α and σ₀ can be weakly identified; strong priors help
3. **Model comparison**: Compare to homoscedastic version via LOO
4. **Alternative variance**: Could use σ(x) = σ₀ * (1 + α*x) for milder growth

#### PyMC Implementation
```python
import pymc as pm

with pm.Model() as log_heteroscedastic_model:
    # Data
    x_data = pm.Data('x', x_obs)
    log_x = pm.math.log(x_data + 1)

    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.70, sigma=0.2)
    beta_1 = pm.Normal('beta_1', mu=0.65, sigma=0.2)
    sigma_0 = pm.HalfCauchy('sigma_0', beta=0.1)
    alpha = pm.Normal('alpha', mu=0, sigma=0.01)

    # Mean and variance functions
    mu = beta_0 + beta_1 * log_x
    sigma = sigma_0 * pm.math.exp(alpha * x_data)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y_obs)

    # Sample
    trace = pm.sample(2000, tune=1000, target_accept=0.9)
```

**Computational Expectations**:
- **Speed**: Moderate (heteroscedasticity adds complexity)
- **Convergence**: May need higher target_accept
- **ESS**: Expect ESS > 400 for all parameters
- **Warnings**: Possible if variance grows too fast

**Simplified Homoscedastic Version** (if α ≈ 0):
```python
with pm.Model() as log_simple_model:
    beta_0 = pm.Normal('beta_0', mu=1.70, sigma=0.2)
    beta_1 = pm.Normal('beta_1', mu=0.65, sigma=0.2)
    sigma = pm.HalfCauchy('sigma', beta=0.15)

    mu = beta_0 + beta_1 * pm.math.log(x_data + 1)
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y_obs)
```

---

## Cross-Model Validation Strategy

### Phase 1: Individual Model Checks (Per Model)

For each of the 3 models:

1. **Prior Predictive Checks**
   - Sample 1000 datasets from priors
   - Check: Y ∈ [1.0, 3.5] for most samples
   - Check: Saturation pattern plausible
   - **Reject prior if**: >10% of samples are nonsensical

2. **MCMC Diagnostics**
   - Target: R-hat < 1.01 for all parameters
   - Target: ESS > 400 for all parameters
   - Check: No divergences
   - **Reject model if**: Cannot achieve convergence after tuning

3. **Posterior Predictive Checks**
   - Visual: Overlay Y_rep vs Y_obs
   - Replicate check: For 6 replicated x-values, check coverage
   - Residual check: Plot residuals vs x, should be random
   - **Reject model if**: <80% of observations in 95% intervals

4. **LOO-CV**
   - Compute LOO and Pareto-k diagnostics
   - Target: All Pareto-k < 0.7
   - **Flag model if**: Multiple Pareto-k > 0.7

### Phase 2: Model Comparison

1. **Predictive Performance**
   ```
   Compare:
   - LOO ELPD (higher is better)
   - LOO R² (target > 0.80)
   - RMSE from posterior mean predictions
   ```

2. **Decision Rules**
   ```
   If |ΔLOO| > 2*SE_diff:
     - Prefer model with higher ELPD

   If |ΔLOO| < 2*SE_diff:
     - Models equivalent
     - Prefer simpler model (fewer parameters)
     - Prefer more interpretable model
   ```

3. **Sensitivity Analysis**
   - Run with alternative priors (wider/narrower)
   - Check if posterior changes substantially
   - **Reject if**: Results highly prior-dependent

### Phase 3: Stress Tests

1. **Extrapolation Test**
   - Predict at x = [0.1, 0.5, 50, 100]
   - Check: Monotonic, reasonable uncertainty
   - Compare across models

2. **Leave-Out-Regime Test**
   - Fit on x < 10, predict x > 10
   - Fit on x > 10, predict x < 10
   - Check: Reasonable performance

3. **Replicate Test**
   - For 6 replicated x-values, compute posterior predictive SD
   - Compare to observed variation
   - Target: Posterior SD ≈ observed SD

### Phase 4: Decision Points

**Checkpoint 1** (After individual fits):
- If all 3 models fail basic checks → STOP, reconsider model class entirely
- If 1-2 pass → Continue with passing models
- If all pass → Continue to comparison

**Checkpoint 2** (After LOO comparison):
- If log-log clearly best (ΔLOO > 4) → Declare winner, report
- If quadratic clearly best → Winner
- If equivalent → Report all, recommend simplest

**Checkpoint 3** (After stress tests):
- If extrapolation fails for all → Report uncertainty, don't extrapolate
- If replicate test fails → Revisit error model

---

## Model Selection Criteria

### Quantitative Thresholds

| Criterion | Minimum Acceptable | Good | Excellent |
|-----------|-------------------|------|-----------|
| R-hat | < 1.01 | < 1.01 | < 1.001 |
| ESS | > 400 | > 800 | > 2000 |
| LOO R² | > 0.75 | > 0.80 | > 0.85 |
| Pareto-k | < 0.7 | < 0.5 | < 0.3 |
| Coverage (95% PPI) | > 90% | > 93% | > 95% |

### Qualitative Considerations

**Prefer Model A over Model B if**:
1. ΔLOO(A,B) > 2*SE and LOO_A > LOO_B
2. ΔLOO ≈ 0 but A is simpler (fewer parameters)
3. ΔLOO ≈ 0 but A is more interpretable
4. A has better extrapolation behavior
5. A has better computational properties

**Red Flags** (Any of these triggers reconsideration):
- Prior-posterior conflict (posterior far from prior mode)
- Computational difficulties despite tuning
- Systematic residual patterns
- Poor replicate fit (observed variation >> predicted)
- Extreme parameter values (e.g., ν < 3 in Student-t)

---

## Expected Outcomes & Falsification

### Predicted Model Ranking

**Most Likely**:
1. Log-log power law (R² ≈ 0.81, simplest, theoretically grounded)
2. Robust quadratic (R² ≈ 0.85, good fit, robust)
3. Log heteroscedastic (R² ≈ 0.82, if variance matters)

**Reasoning**: EDA shows log-log achieves highest linearity (r=0.92), suggesting fundamental power law structure.

### Alternative Scenarios

**Scenario A: Quadratic Dominates**
- If LOO R² > 0.85 and clearly beats log-log
- **Interpretation**: Parabolic form better captures plateau
- **Action**: Accept quadratic, report as pragmatic choice

**Scenario B: All Models Equivalent**
- If ΔLOO < 2*SE for all pairs
- **Interpretation**: Data insufficient to discriminate
- **Action**: Report model averaging or ensemble

**Scenario C: All Models Fail**
- If best LOO R² < 0.75
- **Interpretation**: Transformation/polynomial insufficient
- **Action**: Pivot to Designer 1/2 models (asymptotic, piecewise)

### Global Falsification Criteria

**Abandon ALL three models if**:
1. Best LOO R² < 0.70 (clearly inadequate)
2. All models show systematic residual patterns
3. Replicate test fails for all (observed SD << predicted SD)
4. Computational issues persist despite extensive tuning
5. Extrapolation is wildly implausible for all models

**Escape Routes**:
- → Designer 1: Asymptotic exponential model
- → Designer 2: Piecewise/hierarchical models
- → Gaussian Process: If need maximum flexibility
- → Mixture models: If heterogeneity suspected

---

## Implementation Roadmap

### Stage 1: Setup (30 min)
1. Load data, preprocess (log transformations)
2. Set up Stan/PyMC models for all 3
3. Run prior predictive checks
4. Adjust priors if needed

### Stage 2: Fitting (1-2 hours)
1. Fit Model 1 (log-log) - expect fast
2. Fit Model 2 (robust quadratic) - moderate
3. Fit Model 3 (log heteroscedastic) - moderate
4. Check convergence for each

### Stage 3: Diagnostics (1 hour)
1. MCMC diagnostics (traceplots, R-hat, ESS)
2. Posterior predictive checks
3. LOO-CV computation
4. Residual analysis

### Stage 4: Comparison (30 min)
1. LOO comparison table
2. Visual comparison of fits
3. Stress tests
4. Decision

### Stage 5: Reporting (1 hour)
1. Document chosen model(s)
2. Report performance metrics
3. Visualizations
4. Recommendations

**Total Time Estimate**: 4-5 hours for complete analysis

---

## Key Insights & Design Rationale

### Why These Models?

1. **Log-Log Power Law**
   - EDA finding (r=0.92) is too strong to ignore
   - Transformation simplifies inference dramatically
   - Power laws are ubiquitous in nature
   - **This is the strategic bet**

2. **Robust Quadratic**
   - Polynomial is the "safe" choice (everyone understands it)
   - Student-t provides robustness insurance
   - Good balance of simplicity and flexibility
   - **This is the pragmatic fallback**

3. **Log Heteroscedastic**
   - Tests explicit assumption about variance
   - Simpler than quadratic (2 vs 3 mean parameters)
   - Easy to simplify if α ≈ 0
   - **This is the assumption tester**

### What Makes This Design Robust?

1. **Multiple transformation approaches**: Power law, polynomial, logarithmic
2. **Explicit robustness**: Student-t likelihood in Model 2
3. **Variance modeling**: Heteroscedasticity in Model 3
4. **Clear falsification**: Each model has explicit failure criteria
5. **Escape routes**: Know what to do if all fail

### What Could Go Wrong?

1. **Transformation inadequate**: If saturation is truly asymptotic (not power law)
   - **Pivot to**: Designer 1 asymptotic models

2. **Polynomial unstable**: If extrapolation is critical
   - **Pivot to**: Constrained models (asymptotic)

3. **All simple models fail**: If relationship is truly complex
   - **Pivot to**: Gaussian Process or mixture models

4. **Heterogeneity present**: If subgroups have different patterns
   - **Pivot to**: Designer 2 hierarchical models

---

## Communication Plan

### For Technical Audience
- Report LOO comparison table
- Show posterior distributions for key parameters
- Display residual diagnostics
- Provide Stan/PyMC code

### For General Audience
- Focus on R² and RMSE
- Show observed vs predicted plot
- Explain chosen model in plain language
- Highlight uncertainty in predictions

### Key Messages
1. "We tested 3 fundamentally different approaches to modeling the saturation pattern"
2. "The log-log transformation achieved strong linearity (r=0.92), suggesting a power law"
3. "All models show Y increases rapidly at low x, then plateaus at high x"
4. "Uncertainty grows substantially when extrapolating beyond x=32"

---

## Final Recommendations

### Primary Strategy
1. **Fit all 3 models** in parallel
2. **Compare via LOO** - let data decide
3. **Report winner(s)** with full diagnostics
4. **Document uncertainty** especially for extrapolation

### If Time Constrained
1. **Fit log-log model first** (highest prior probability of success)
2. **If satisfactory** (R² > 0.80, good diagnostics) → DONE
3. **If not** → Fit robust quadratic
4. **If still not** → Pivot to Designer 1/2 models

### Success Metrics
- At least one model achieves LOO R² > 0.80
- Convergence for all parameters (R-hat < 1.01)
- Residuals show no systematic patterns
- Predictions reasonable for replicated x-values

### Definition of Complete Failure
- No model achieves R² > 0.75
- Persistent convergence issues
- Systematic residual patterns remain
- Extrapolation wildly implausible

**In case of failure**: Do not force it. Acknowledge limitations, pivot to alternative model classes, or declare data insufficient for reliable inference.

---

## Appendix: Quick Reference

### Model Comparison Table

| Feature | Log-Log | Robust Quadratic | Log Heteroscedastic |
|---------|---------|------------------|---------------------|
| Parameters | 3 (α, β, σ) | 5 (β₀, β₁, β₂, σ, ν) | 4 (β₀, β₁, σ₀, α) |
| Transformation | Yes (log-log) | No | No |
| Robustness | Multiplicative errors | Student-t | Heteroscedastic |
| Speed | Fast | Moderate | Moderate |
| Interpretability | High | High | Medium |
| Expected R² | 0.81 | 0.85 | 0.82 |
| Extrapolation | Power law | Polynomial | Logarithmic |
| Best For | Power law processes | General saturation | Variance modeling |

### Prior Summary Table

| Parameter | Model | Distribution | Rationale |
|-----------|-------|--------------|-----------|
| α (intercept, log-log) | 1 | Normal(0.6, 0.3) | log(1.8) ≈ 0.59 |
| β (slope, log-log) | 1 | Normal(0.12, 0.1) | OLS estimate |
| β₀ (intercept) | 2,3 | Normal(1.70-1.75, 0.2-0.3) | Low-x extrapolation |
| β₁ (linear term) | 2 | Normal(0.08, 0.05) | Positive slope |
| β₂ (quadratic term) | 2 | Normal(-0.002, 0.001) | Negative curvature |
| σ | 1,2,3 | Half-Cauchy(0, 0.1-0.15) | Replicate SD |
| ν (DOF) | 2 | Gamma(2, 0.1) | Weakly favors ν≈20 |
| α (hetero) | 3 | Normal(0, 0.01) | Null = homoscedastic |

### Convergence Troubleshooting

| Issue | Solution |
|-------|----------|
| High R-hat | Increase warmup, check initialization |
| Low ESS | Increase samples, check for multimodality |
| Divergences | Increase target_accept (0.9 or 0.95) |
| Slow mixing | Reparameterize (center x), non-centered |
| ν instability | Stronger prior on ν, fix ν=5 if needed |

---

**Document Status**: Complete
**Designer**: 3 (Alternative Approaches)
**Models Proposed**: 3 (Log-Log Power Law, Robust Quadratic, Log Heteroscedastic)
**Ready for**: Implementation and testing
**Contact**: Reference this document for all design decisions

