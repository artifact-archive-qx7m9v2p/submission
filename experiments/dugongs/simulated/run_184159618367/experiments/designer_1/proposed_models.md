# Bayesian Model Design: Parametric Saturation Models
## Designer 1: Mechanistically-Interpretable Approaches

**Date**: 2025-10-27
**Focus**: Parametric models with clear mechanistic interpretation
**Design Philosophy**: Competing hypotheses with explicit falsification criteria

---

## Executive Summary

I propose **3 fundamentally different model classes** that represent distinct data generation hypotheses:

1. **Asymptotic Growth Models** (smooth saturation via rate parameter)
2. **Threshold Models** (sharp regime shift at breakpoint)
3. **Diminishing Returns Models** (monotonic deceleration via power law)

**Critical Insight**: These models embody different scientific hypotheses about *how* saturation occurs:
- Is it a **smooth exponential approach** to equilibrium?
- Is it a **sharp phase transition** at a threshold?
- Is it **continuous deceleration** following a power law?

Each model will be implemented in Stan with explicit falsification criteria. I will abandon models early if evidence contradicts their core assumptions.

---

## Competing Hypotheses Framework

### Hypothesis 1: Smooth Exponential Saturation
**Mechanism**: System approaches equilibrium asymptotically (e.g., enzyme kinetics, learning curves, resource depletion)
**Prediction**: Rate of change decreases exponentially with x
**Would fail if**: Transition is truly discontinuous, or multiple regimes exist

### Hypothesis 2: Sharp Threshold Transition
**Mechanism**: System operates in distinct regimes separated by threshold (e.g., phase change, capacity constraint, switch-like behavior)
**Prediction**: Slope change occurs at specific breakpoint
**Would fail if**: Transition is gradual/smooth, or breakpoint varies systematically

### Hypothesis 3: Power Law Diminishing Returns
**Mechanism**: Constant elasticity process (e.g., scaling laws, marginal utility, allometric relationships)
**Prediction**: Log-log linearity, constant percentage returns
**Would fail if**: Returns diminish faster than power law, or plateau is sharp

---

## Model 1: Asymptotic Exponential (Michaelis-Menten Family)

### Model Specification

**Likelihood**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = α - β * exp(-γ * x_i)
```

**Parameters**:
- `α`: Asymptote (upper limit as x → ∞)
- `β`: Amplitude (difference from minimum to asymptote)
- `γ`: Rate parameter (speed of saturation, units: 1/x)
- `σ`: Residual standard deviation

**Priors** (informed by EDA):
```stan
// Asymptote: plateau observed at 2.5-2.6
α ~ Normal(2.55, 0.1)

// Amplitude: range from x→0 to plateau
// Back-extrapolation suggests Y_min ≈ 1.65, so β ≈ 2.55 - 1.65 = 0.9
β ~ Normal(0.9, 0.2)  // weakly informative, allows 0.5-1.3

// Rate: transition occurs over ~10 x-units, so γ ≈ 0.1-0.3
γ ~ Gamma(4, 20)  // E[γ]=0.2, SD=0.1, restricted to positive

// Residual: pure error from replicates ≈ 0.075-0.12
σ ~ Half-Cauchy(0, 0.15)  // weakly informative, heavy tail
```

**Prior Predictive Check Expectations**:
- 95% of prior samples should produce Y ∈ [1.0, 3.5]
- Curves should saturate within x ∈ [10, 40]
- Initial slope at x=1 should be positive but < 0.3

### Theoretical Justification

**Why this functional form?**
1. **Mechanistic Plausibility**: Ubiquitous in natural processes
   - Enzyme kinetics: Michaelis-Menten equation
   - Learning curves: rapid improvement early, plateau with practice
   - Resource depletion: exponential approach to carrying capacity
   - Market saturation: diminishing marginal returns

2. **Mathematical Properties**:
   - Monotonically increasing (if β, γ > 0)
   - Bounded above by α (guaranteed saturation)
   - Has interpretable inflection point at x = ln(2)/γ
   - First derivative: dY/dx = β*γ*exp(-γ*x) (exponentially decreasing)

3. **EDA Alignment**:
   - Achieved R² = 0.889 in frequentist fit
   - Captures observed saturation pattern
   - Only 3 parameters (parsimonious)

**Parameter Interpretation**:
- `α = 2.55`: System plateaus at Y ≈ 2.55
- `β = 0.9`: Total change from minimum to maximum is 0.9 units
- `γ = 0.2`: Half-saturation occurs at x ≈ 3.5
- `σ = 0.09`: Typical prediction error is ±0.09 units

### Falsification Criteria

**I will abandon this model if**:

1. **Prior-Posterior Conflict**:
   - Posterior for α or β shifts >3 SD from prior
   - Suggests misspecified functional form or poor prior

2. **Computational Pathology**:
   - R-hat > 1.01 after 4000 iterations with good initialization
   - Effective sample size < 100 for any parameter
   - Indicates model is unidentifiable or misspecified

3. **Systematic Residual Patterns**:
   - Residuals show clear structure vs x (e.g., U-shape, heteroscedasticity)
   - Autocorrelation in residuals for replicated x-values
   - Suggests missing model component

4. **Poor Predictive Performance**:
   - LOO-R² < 0.80 (worse than EDA benchmark)
   - >10% of observations outside 95% posterior predictive intervals
   - Multiple Pareto-k > 0.7 (many influential points)

5. **Asymptote Implausibility**:
   - Posterior for α has >10% mass above max(Y) + 0.3
   - Suggests model is trying to fit plateau that doesn't exist yet
   - May need longer-range model or threshold model

6. **Rate Parameter Extremes**:
   - Posterior for γ includes substantial mass near 0 (γ < 0.05)
   - Suggests saturation is too slow, model forcing flat line
   - OR γ > 1.0 suggesting near-instantaneous saturation (threshold model better)

**Red Flags Requiring Investigation**:
- Posterior for β becomes bimodal (identifiability issue)
- Strong negative correlation between α and β (>0.9) in posterior
- Predictions diverge wildly for x > 31.5 (extrapolation risk)

**Decision Point**: If >2 falsification criteria met, pivot to Model 2 (Threshold)

### Expected Performance

**Convergence**:
- Should converge in 2000-4000 iterations with warmup
- May need non-centered parameterization if correlation issues arise
- Use OLS fit (α=2.565, β=1.019, γ=0.204) for initialization

**Fit Quality**:
- Posterior predictive R²: 0.85-0.90 (matching EDA benchmark)
- RMSE: 0.08-0.10
- Coverage: 90-95% of observations in 95% credible intervals

**Uncertainty Quantification**:
- Narrow credible intervals for α (±0.05) due to plateau data
- Wider intervals for β (±0.1) due to extrapolation to x→0
- Moderate uncertainty in γ (±0.05) controlling transition speed
- Widest intervals for predictions at x<2 and x>25 (sparse data)

**LOO-CV**:
- Expect ELPD ≈ -20 to -30
- All Pareto-k < 0.5 (ideally)
- Observation at x=31.5 may have k ∈ [0.5, 0.7] (monitor)

### Implementation Notes (Stan)

**Parameterization**:
```stan
parameters {
  real<lower=2.3> alpha;           // asymptote must exceed max(Y)
  real<lower=0> beta;              // amplitude must be positive
  real<lower=0> gamma;             // rate must be positive
  real<lower=0> sigma;             // residual SD must be positive
}

model {
  // Priors
  alpha ~ normal(2.55, 0.1);
  beta ~ normal(0.9, 0.2);
  gamma ~ gamma(4, 20);
  sigma ~ cauchy(0, 0.15);

  // Likelihood
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = alpha - beta * exp(-gamma * x[i]);
  }
  Y ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive samples
  vector[N] Y_rep;
  vector[N] log_lik;
  real Y_at_x0 = alpha - beta;  // predicted Y when x→0
  real half_sat_x = log(2) / gamma;  // x where half-saturation occurs

  for (i in 1:N) {
    real mu_i = alpha - beta * exp(-gamma * x[i]);
    Y_rep[i] = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu_i, sigma);
  }
}
```

**Potential Issues**:
1. **Correlation between α and β**: If severe (>0.95), use non-centered parameterization
2. **Initialization**: Use `init=user_init` with OLS values to avoid poor starting points
3. **Identifiability**: With limited low-x data, β may be weakly identified
   - Monitor effective sample size
   - Check prior sensitivity for β

**Computational Efficiency**:
- Simple exponential operations (fast)
- Expect ~1000 effective samples per second
- Total runtime: <1 minute for 4000 iterations

---

## Model 2: Piecewise Linear (Broken Stick / Threshold Model)

### Model Specification

**Likelihood**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = β_0 + β_1 * x_i                         if x_i ≤ τ
μ_i = β_0 + β_1 * τ + β_2 * (x_i - τ)        if x_i > τ
```

**Continuous at breakpoint**: Enforced by construction

**Parameters**:
- `β_0`: Intercept
- `β_1`: Slope in low-x regime (x ≤ τ)
- `β_2`: Slope in high-x regime (x > τ)
- `τ`: Breakpoint location
- `σ`: Residual standard deviation

**Priors** (informed by EDA):
```stan
// Intercept: Y at x=0 extrapolates to ~1.6-1.7
β_0 ~ Normal(1.65, 0.2)

// Slope 1: strong positive slope at low x (observed ~0.08)
β_1 ~ Normal(0.08, 0.03)  // weakly informative, forces positive

// Slope 2: near-zero or negative at high x (observed ~0.00)
β_2 ~ Normal(0.0, 0.02)  // allows small positive or negative

// Breakpoint: visual and segmented analysis suggests 9-10
τ ~ Normal(9.5, 1.5)  // allows range 6-13 with 95% probability

// Residual: same as Model 1
σ ~ Half-Cauchy(0, 0.15)
```

**Prior Predictive Check Expectations**:
- 95% of prior samples should show breakpoint at x ∈ [6, 13]
- Slope should decrease (β_1 > β_2)
- Model should produce saturation-like pattern (not V-shape or hump)

### Theoretical Justification

**Why this functional form?**
1. **Mechanistic Plausibility**: Common in threshold-driven systems
   - Phase transitions (solid → liquid at melting point)
   - Capacity constraints (system saturates when capacity reached)
   - Regulatory thresholds (policy/biological switches)
   - Tipping points in complex systems

2. **Mathematical Properties**:
   - Continuous but non-differentiable at τ
   - Two distinct linear regimes
   - Explicitly tests hypothesis of regime shift
   - Breakpoint τ is scientifically interpretable

3. **EDA Alignment**:
   - Best frequentist fit (R² = 0.904)
   - Segmented analysis found strong evidence for breakpoint
   - Slope changes dramatically: 0.080 → ~0.000
   - Visual inspection confirms apparent threshold at x ≈ 9-10

**Parameter Interpretation**:
- `τ = 9.5`: System transitions from growth to saturation at x = 9.5
- `β_1 = 0.08`: Below threshold, Y increases by 0.08 per unit x
- `β_2 ≈ 0.00`: Above threshold, Y is essentially constant
- Transition sharpness is perfect (discontinuous derivative)

### Falsification Criteria

**I will abandon this model if**:

1. **Breakpoint Uncertainty Too Large**:
   - Posterior for τ spans >50% of x range (e.g., 95% CI wider than 10 x-units)
   - Indicates no clear breakpoint exists, smooth model preferred

2. **Slopes Not Significantly Different**:
   - Posterior P(β_1 > β_2) < 0.95
   - No evidence for regime shift, simpler model (quadratic?) sufficient

3. **Breakpoint at Data Boundary**:
   - Posterior mode for τ < 3 or τ > 25
   - Model trying to fit simple monotonic curve with fake breakpoint

4. **Discontinuity Artifacts**:
   - Residuals show systematic jump at τ
   - Observations near breakpoint consistently under/over-predicted
   - Suggests true transition is smooth, not sharp

5. **Poor Predictive Performance**:
   - LOO-R² < 0.85 (shouldn't happen given EDA benchmark of 0.90)
   - Pareto-k for observations near τ are > 0.7
   - Model is overfitting to specific observations

6. **Posterior Bimodality for τ**:
   - Two distinct peaks in τ posterior
   - Suggests data supports multiple breakpoints (complex model needed)
   - Or insufficient data to localize breakpoint

**Red Flags Requiring Investigation**:
- Strong correlation between τ and β_1 (>0.8)
- Posterior for β_2 doesn't include zero (may need non-zero plateau slope)
- Predictions become non-monotonic (crossing segments)

**Decision Point**: If falsification criteria met, pivot to Model 1 (smooth saturation) or consider 2-breakpoint model

### Expected Performance

**Convergence**:
- May be slower than Model 1 (discrete breakpoint can cause sampling issues)
- Expect 3000-5000 iterations needed
- Use continuous approximation if severe convergence problems
- Monitor τ chain carefully for good mixing

**Fit Quality**:
- Posterior predictive R²: 0.88-0.92 (should match/exceed EDA)
- RMSE: 0.07-0.09 (best of all models)
- Coverage: 92-96% (excellent given model flexibility)

**Uncertainty Quantification**:
- τ uncertainty is key diagnostic (expect SD ≈ 1-2 x-units)
- Narrow intervals for β_1 (well-constrained by low-x data)
- Wider intervals for β_2 (fewer high-x observations)
- Predictions highly uncertain exactly AT breakpoint (structural uncertainty)

**LOO-CV**:
- Expect ELPD ≈ -18 to -28 (should be best)
- Observations near τ may have elevated Pareto-k (0.5-0.7)
- If multiple k > 0.7, suggests overfitting to breakpoint region

### Implementation Notes (Stan)

**Parameterization** (tricky due to breakpoint):
```stan
parameters {
  real beta_0;
  real<lower=0> beta_1;       // force positive slope before breakpoint
  real beta_2;                 // allow negative/zero after breakpoint
  real<lower=1, upper=30> tau; // breakpoint constrained to reasonable range
  real<lower=0> sigma;
}

model {
  // Priors
  beta_0 ~ normal(1.65, 0.2);
  beta_1 ~ normal(0.08, 0.03);
  beta_2 ~ normal(0.0, 0.02);
  tau ~ normal(9.5, 1.5);
  sigma ~ cauchy(0, 0.15);

  // Likelihood (vectorized with conditional indexing)
  vector[N] mu;
  for (i in 1:N) {
    if (x[i] <= tau) {
      mu[i] = beta_0 + beta_1 * x[i];
    } else {
      mu[i] = beta_0 + beta_1 * tau + beta_2 * (x[i] - tau);
    }
  }
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;
  real slope_change = beta_2 - beta_1;  // negative indicates deceleration
  real Y_at_tau = beta_0 + beta_1 * tau;  // predicted Y at breakpoint

  for (i in 1:N) {
    real mu_i;
    if (x[i] <= tau) {
      mu_i = beta_0 + beta_1 * x[i];
    } else {
      mu_i = beta_0 + beta_1 * tau + beta_2 * (x[i] - tau);
    }
    Y_rep[i] = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu_i, sigma);
  }
}
```

**Potential Issues**:
1. **Discrete breakpoint causes sampling difficulties**:
   - Use tight prior on τ to improve mixing
   - Consider continuous approximation: `sigmoid((x - τ)/scale)`
   - May need more warmup iterations (1500-2000)

2. **Label switching**: β_1 and β_2 could swap if prior is too weak
   - Enforce β_1 > 0 with constraint
   - Monitor posterior to ensure β_1 > β_2 (expected pattern)

3. **Sensitivity to τ prior**:
   - Run prior sensitivity analysis (try N(9.5, 1.0) and N(9.5, 2.5))
   - If posterior highly sensitive, breakpoint is weakly identified

**Computational Efficiency**:
- Conditional statement prevents vectorization (slower)
- Expect ~300-500 effective samples per second
- Total runtime: 2-4 minutes for 4000 iterations

**Alternative Parameterization** (if convergence issues):
Use smooth approximation with logistic transition:
```stan
mu[i] = beta_0 + beta_1 * x[i] +
        (beta_2 - beta_1) * (x[i] - tau) * inv_logit((x[i] - tau) / scale);
```
This creates nearly-identical fit but removes discontinuity in gradient.

---

## Model 3: Power Law (Allometric / Diminishing Returns)

### Model Specification

**Likelihood** (on original scale):
```
Y_i ~ LogNormal(log(μ_i), σ)
μ_i = α * x_i^β
```

**Equivalent log-log formulation**:
```
log(Y_i) ~ Normal(log(α) + β * log(x_i), σ)
```

**Parameters**:
- `α`: Scaling coefficient (Y when x=1)
- `β`: Power law exponent (elasticity, dimensionless)
- `σ`: Log-scale residual standard deviation

**Priors** (informed by EDA):
```stan
// Scaling: log(Y) at log(x)=0 suggests α ≈ 1.8-2.0
α ~ Normal(1.9, 0.2)  // weakly informative on original scale

// Exponent: log-log fit found β ≈ 0.12, bounded [0, 1]
β ~ Beta(3, 15)  // E[β]=0.167, allows range 0.05-0.35 with 95% probability
                 // Skewed toward small values (diminishing returns)

// Log-scale residual: smaller than original scale
σ ~ Half-Cauchy(0, 0.1)
```

**Prior Predictive Check Expectations**:
- 95% of prior samples should produce β ∈ [0.05, 0.35]
- Curves should show diminishing returns (concave)
- Predictions at x=30 should not exceed 3.0

### Theoretical Justification

**Why this functional form?**
1. **Mechanistic Plausibility**: Universal in scaling phenomena
   - Allometric scaling (biological size relationships)
   - Marginal utility (economics)
   - Power laws in physics (e.g., drag, radiation)
   - Self-similar processes (fractals, networks)

2. **Mathematical Properties**:
   - Log-log linearity (correlation = 0.92 in EDA)
   - Constant elasticity: % change in Y per % change in x
   - Monotonically increasing if β > 0
   - Concave (diminishing returns) if β < 1
   - Scale-free (no characteristic scale)

3. **EDA Alignment**:
   - Achieved R² = 0.81 (decent but not best)
   - Log-log transformation linearizes relationship
   - Simpler than exponential (only 2 structural parameters)

**Parameter Interpretation**:
- `α = 1.9`: When x=1, predicted Y ≈ 1.9
- `β = 0.12`: A 10% increase in x yields 1.2% increase in Y
- Elasticity is constant across all x (key assumption)
- No explicit saturation mechanism (implicit via β < 1)

### Falsification Criteria

**I will abandon this model if**:

1. **Log-Log Non-Linearity**:
   - Residuals on log-log scale show clear curvature vs log(x)
   - Suggests power law is inadequate, exponential saturation preferred
   - Specifically: residuals positive at extremes, negative in middle (U-shape)

2. **Poor High-x Fit**:
   - Model systematically under-predicts Y for x > 20
   - Power law doesn't saturate fast enough
   - Asymptotic model (Model 1) would fit better

3. **Exponent Near Boundary**:
   - Posterior for β has >20% mass near 0 (β < 0.05)
   - Suggests essentially flat relationship (no model needed)
   - OR β > 0.5 suggests faster-than-expected growth (exponential better)

4. **Lognormal Assumption Violated**:
   - Posterior predictive checks show poor coverage
   - Log(Y) residuals are skewed or heavy-tailed
   - May need Student-t or different transformation

5. **Predictive Performance Gap**:
   - LOO-R² < 0.75 (substantially worse than Models 1-2)
   - ΔELPD > 10 compared to best model (decisive difference)
   - Not worth the simplicity trade-off

6. **Scale-Free Assumption Fails**:
   - Different elasticity in low-x vs high-x regimes
   - Check via segmented log-log regressions
   - Would favor threshold or multi-regime model

**Red Flags Requiring Investigation**:
- High leverage for x=31.5 (only high-x point)
- Predictions diverge wildly for x > 40 (no saturation bound)
- Posterior for α is bimodal (identifiability issue)

**Decision Point**: If >2 falsification criteria met, this model is wrong for this data

### Expected Performance

**Convergence**:
- Should converge easily (linear in log-log space)
- Expect 1500-2500 iterations sufficient
- No sampling pathologies expected
- Fast mixing (parameters weakly correlated)

**Fit Quality**:
- Posterior predictive R²: 0.78-0.83 (matching EDA)
- RMSE: 0.11-0.13 (worse than Models 1-2)
- Coverage: 88-93% (adequate but not excellent)

**Uncertainty Quantification**:
- Narrow intervals for β (log-log relationship is strong)
- Moderate uncertainty in α (depends on low-x extrapolation)
- Widest prediction intervals at high x (no saturation bound)

**LOO-CV**:
- Expect ELPD ≈ -25 to -35 (worse than Models 1-2)
- Observation at x=31.5 likely has elevated Pareto-k (high leverage)
- May show systematic prediction errors at plateau region

### Implementation Notes (Stan)

**Parameterization** (log-log formulation is more stable):
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
  vector[N] log_x;  // pre-compute to avoid repeated log() calls
  vector[N] log_Y;
}

parameters {
  real log_alpha;           // log-scale intercept
  real<lower=0, upper=1> beta;  // constrain to diminishing returns region
  real<lower=0> sigma;      // log-scale residual SD
}

transformed parameters {
  real alpha = exp(log_alpha);  // back-transform for interpretation
}

model {
  // Priors (on log scale for better geometry)
  log_alpha ~ normal(log(1.9), 0.1);  // implies α ~ LogNormal(log(1.9), 0.1)
  beta ~ beta(3, 15);
  sigma ~ cauchy(0, 0.1);

  // Likelihood on log-log scale (Gaussian)
  vector[N] log_mu = log_alpha + beta * log_x;
  log_Y ~ normal(log_mu, sigma);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;
  real elasticity = beta;  // interpretable quantity

  for (i in 1:N) {
    real log_mu_i = log_alpha + beta * log_x[i];
    real Y_pred = exp(log_mu_i);

    // Sample on log scale then back-transform
    Y_rep[i] = exp(normal_rng(log_mu_i, sigma));

    // Log-likelihood on log scale
    log_lik[i] = normal_lpdf(log_Y[i] | log_mu_i, sigma);
  }
}
```

**Potential Issues**:
1. **Back-transformation bias**:
   - LogNormal mean ≠ exp(log-scale mean)
   - Use exp(μ + σ²/2) for unbiased predictions
   - Or simulate on log-scale and exponentiate samples

2. **Zero or negative Y**:
   - Log transform undefined for Y ≤ 0
   - Not an issue here (all Y > 1.7)
   - But could add small constant if needed

3. **Extrapolation**:
   - Power law has no upper bound (unlike Models 1-2)
   - Predictions for x >> 31.5 will be overly optimistic
   - Emphasize this limitation in interpretation

**Computational Efficiency**:
- Fastest of all models (linear regression on log-log scale)
- Expect >2000 effective samples per second
- Total runtime: <30 seconds for 4000 iterations

---

## Model Comparison Strategy

### Decision Framework

**Phase 1: Individual Model Validation** (Do in Parallel)
For each model:
1. Prior predictive checks (do priors generate reasonable data?)
2. Fit model to observed data
3. MCMC diagnostics (convergence, effective sample size)
4. Posterior predictive checks (does model capture patterns?)
5. Check falsification criteria (should we abandon this model?)

**Phase 2: Head-to-Head Comparison** (Only for models that pass Phase 1)
1. LOO-CV: Compare ELPD with standard errors
2. Pareto-k diagnostics: Identify influential observations
3. Visual comparison: Overlay predictions from all models
4. Interpretability: Which parameters are most scientifically meaningful?

**Phase 3: Model Selection** (Evidence-Based)
- If |ΔELPD| > 2×SE: Choose model with highest ELPD
- If |ΔELPD| < 2×SE: Choose simpler/more interpretable model
- Consider model averaging if models are statistically equivalent

### Key Comparisons

**Model 1 vs Model 2**: Smooth vs Sharp Transition
- If LOO prefers Model 2 (piecewise): Strong evidence for threshold
- If equivalent: Prefer Model 1 (simpler, more interpretable)
- Look at predictions near x=9-10 (critical region)

**Model 1 vs Model 3**: Asymptotic vs Power Law
- If Model 3 residuals show curvature in log-log space: Model 1 wins
- If Model 1 fits high-x region much better: Strong evidence for saturation
- Power law cannot capture true plateau

**Model 2 vs Model 3**: Threshold vs Continuous Diminishing Returns
- If breakpoint uncertainty is large (Model 2): Model 3 may be preferred
- If power law exponent is near 0: Model 2 (flat plateau) preferred
- These are philosophically very different hypotheses

### Expected Outcome

**Most Likely Scenario**: Models 1 and 2 are statistically equivalent
- Both capture saturation well
- ΔELPD < 4 (within 2 SE)
- Model 1 preferred for interpretability and extrapolation
- Report uncertainty in mechanism (smooth vs sharp)

**Alternative Scenario 1**: Model 2 (Piecewise) decisively wins
- ΔELPD > 6 compared to Models 1 and 3
- Strong evidence for true threshold at x ≈ 9-10
- Scientific interpretation: system has capacity constraint or phase transition

**Alternative Scenario 2**: Model 1 (Exponential) decisively wins
- Model 2 breakpoint uncertainty too large (>5 x-units)
- Model 3 shows systematic log-log curvature
- Scientific interpretation: smooth approach to equilibrium

**Unlikely Scenario**: Model 3 (Power Law) wins
- Would require excellent fit at high-x despite no saturation mechanism
- Would suggest visual "plateau" is artifact of log-log perception
- Would require R² ≈ 0.85+ (unlikely given EDA benchmark of 0.81)

---

## Stress Tests and Red Flags

### Stress Test 1: High-x Extrapolation
**Goal**: Break the models by extrapolating to x=50, 100

**Predictions**:
- Model 1: Should saturate at α ± small uncertainty
- Model 2: Should remain flat at β_0 + β_1*τ + β_2*(x-τ)
- Model 3: Will predict unbounded growth (fails)

**Success Criterion**: Models 1-2 should have stable, bounded predictions

### Stress Test 2: Low-x Interpolation
**Goal**: Predict Y at x=0.1, 0.5 (below observed range)

**Predictions**:
- Model 1: Predicts Y ≈ α - β (well-defined)
- Model 2: Extrapolates linearly (may go below observed minimum)
- Model 3: Power law explodes as x→0 if β>0 (fails for x<1)

**Success Criterion**: Predictions should be plausible (Y ∈ [1.0, 2.0])

### Stress Test 3: Replicate Consistency
**Goal**: Check if model predictions are consistent across replicated x-values

**Method**:
- For each of 6 replicated x-values, compute posterior predictive variance
- Compare to observed variance within replicates
- Should be similar (calibrated uncertainty)

**Success Criterion**: Observed variance within 95% posterior predictive interval

### Stress Test 4: Leave-One-Out by Regime
**Goal**: Test sensitivity to removing data from specific x-regions

**Method**:
- Remove all data with x < 5 (9 observations)
- Remove all data with x > 15 (8 observations)
- Refit models and check parameter stability

**Success Criterion**: Parameters should shift < 1 SD from full-data posterior

### Red Flags (Global)

**Abandon Entire Modeling Approach If**:
1. **All three models fail falsification criteria**
   - Suggests fundamental misunderstanding of data
   - Consider: heteroscedastic models, mixture models, changepoint models, Gaussian process

2. **Posterior predictive checks consistently fail**
   - Coverage < 85% for all models
   - Systematic residual patterns remain
   - Suggests missing covariate or non-normal errors

3. **LOO-CV shows many influential points**
   - >5 observations with Pareto-k > 0.7
   - Model is overfitting to specific data points
   - Need more robust model class

4. **Parameters have implausible values**
   - E.g., α < max(Y) in Model 1 (asymptote below data)
   - E.g., β > 1 in Model 3 (accelerating returns)
   - Model fighting the data (misspecification)

5. **Prior-posterior conflict across all models**
   - Data strongly contradicts all reasonable priors
   - Either data is wrong or I misunderstood the problem

**Pivot Options If All Models Fail**:
- Heteroscedastic models: σ(x) = σ_0 * f(x)
- Mixture models: Multiple latent populations
- Student-t likelihood: Robust to outliers
- Gaussian Process: Fully nonparametric
- Time-series models: If x represents time and autocorrelation exists

---

## Implementation Timeline

### Phase 1: Model Implementation (Day 1)
- [ ] Implement Model 1 (Asymptotic) in Stan
- [ ] Implement Model 2 (Piecewise) in Stan
- [ ] Implement Model 3 (Power Law) in Stan
- [ ] Write prior predictive check script (shared across models)
- [ ] Write posterior predictive check script (shared)
- [ ] Write LOO-CV comparison script

### Phase 2: Prior Validation (Day 1-2)
- [ ] Run prior predictive checks for each model
- [ ] Visualize prior predictive samples (n=1000)
- [ ] Adjust priors if needed (too informative or too diffuse)
- [ ] Document prior choices and justifications

### Phase 3: Model Fitting (Day 2)
- [ ] Fit Model 1 (expect 2-5 minutes)
- [ ] Fit Model 2 (expect 5-10 minutes)
- [ ] Fit Model 3 (expect 1-3 minutes)
- [ ] Check MCMC diagnostics for all models
- [ ] Re-fit with adjusted parameters if needed

### Phase 4: Validation (Day 2-3)
- [ ] Posterior predictive checks (visual and quantitative)
- [ ] Check all falsification criteria
- [ ] Run stress tests
- [ ] Identify red flags
- [ ] LOO-CV for models that pass validation

### Phase 5: Comparison and Reporting (Day 3)
- [ ] Compare ELPD across models
- [ ] Visualize predictions from all models
- [ ] Interpret parameters scientifically
- [ ] Document model strengths/weaknesses
- [ ] Make recommendation with uncertainty

### Phase 6: Synthesis (Day 3-4)
- [ ] Compare with other designers' models
- [ ] Identify consensus and disagreements
- [ ] Propose ensemble/averaging strategy if appropriate
- [ ] Document lessons learned

**Total Estimated Time**: 3-4 days

---

## Success Criteria

This modeling effort will be considered **successful** if:

1. **At least one model passes all falsification criteria**
   - Converges properly (R-hat < 1.01)
   - Captures saturation pattern
   - Achieves LOO-R² > 0.80
   - Passes posterior predictive checks

2. **Parameter estimates are scientifically interpretable**
   - Credible intervals are reasonably narrow (< 50% of parameter value)
   - Estimates align with EDA findings
   - Posterior is not fighting the prior

3. **Uncertainty is well-calibrated**
   - 90-95% coverage in posterior predictive checks
   - Wider intervals where data is sparse
   - Narrower intervals at replicated x-values

4. **Model comparison is decisive or informative**
   - Either one model clearly wins (ΔELPD > 2×SE)
   - Or equivalence is informative (multiple mechanisms plausible)

5. **I can explain why rejected models failed**
   - Clear evidence against falsified models
   - Not just "this one had better LOO"
   - Understand mechanism of failure

---

## Anticipated Findings

### Most Likely Outcome
**Model 1 (Asymptotic Exponential) is preferred**
- Fits data well (R² ≈ 0.88)
- Scientifically interpretable parameters
- Robust to extrapolation
- LOO equivalent or slightly better than Model 2

**Model 2 (Piecewise) is statistically equivalent but less preferred**
- Slightly better fit (R² ≈ 0.90)
- Breakpoint creates identifiability issues
- Less robust to extrapolation
- Threshold interpretation may be overly strong

**Model 3 (Power Law) is inadequate**
- Fails to capture plateau (R² ≈ 0.81)
- Log-log residuals show curvature
- Poor high-x fit
- Falsification criteria met

**Conclusion**: Smooth saturation mechanism (Model 1) is most consistent with data

### Alternative Outcome
**Model 2 (Piecewise) decisively wins**
- Breakpoint clearly identified at x ≈ 9.5 ± 1
- Much better LOO (ΔELPD > 6)
- Sharp transition is real, not artifact

**Scientific Interpretation**: System has capacity constraint or regulatory threshold
- Below x=9.5: System is undersaturated, Y increases linearly
- Above x=9.5: System is saturated, Y is constant
- Suggests physical/biological mechanism with discrete switch

**Implication**: Would recommend investigating what happens at x≈9.5 (mechanistic studies)

---

## Key Interpretations by Model

### If Model 1 Wins: Smooth Saturation Process
**Mechanism**: System approaches equilibrium asymptotically
- Examples: enzyme saturation, learning curves, resource depletion
- Continuous rate of change (no discontinuities)
- Half-saturation at x ≈ log(2)/γ ≈ 3-4

**Parameters to Report**:
- Asymptote α with 95% CI (ultimate system capacity)
- Rate γ with 95% CI (speed of saturation)
- Derived: half-saturation point, initial slope

### If Model 2 Wins: Threshold-Driven Transition
**Mechanism**: System operates in distinct regimes
- Examples: phase transitions, capacity constraints, tipping points
- Discontinuous derivative at τ (sudden change)
- Binary interpretation: pre-threshold vs post-threshold

**Parameters to Report**:
- Breakpoint τ with 95% CI (critical threshold)
- Slope change β_1 - β_2 (magnitude of transition)
- Derived: Y at threshold, regime definitions

### If Model 3 Wins: Scale-Free Diminishing Returns
**Mechanism**: Constant elasticity across all scales
- Examples: allometric scaling, power law utilities
- No characteristic scale or saturation point
- Returns diminish but never cease

**Parameters to Report**:
- Exponent β with 95% CI (elasticity)
- Scaling α with 95% CI (reference point)
- Derived: predicted Y at reference x-values

---

## Final Notes

### Philosophy
- **Truth-seeking over task-completion**: I will pivot if evidence demands
- **Falsification mindset**: Each model has explicit failure modes
- **Bayesian honesty**: Report uncertainty, don't pretend we know more than we do
- **Mechanism matters**: Parameters should have scientific meaning

### Unknowns I'm Comfortable With
- Exact form of saturation (smooth vs sharp) - data will tell us
- Optimal prior strength - will tune based on prior predictive checks
- Whether Models 1-2 are distinguishable - that's a scientific finding

### Unknowns That Worry Me
- Extrapolation beyond x=31.5 (limited high-x data)
- Identifiability with only 27 observations (may need stronger priors)
- Sensitivity to single influential point (x=31.5)
- Possibility that relationship is more complex than any model captures

### Commitment to Pivot
I will abandon this entire approach and recommend alternative models if:
1. All three models fail validation
2. Posterior predictive checks show systematic failures
3. LOO diagnostics reveal fundamental issues
4. Another designer discovers evidence contradicting saturation hypothesis

**Success is learning the truth about this relationship, not defending my initial models.**

---

## File Outputs

This analysis will generate:
- `designer_1_model1_asymptotic.stan` - Stan code for Model 1
- `designer_1_model2_piecewise.stan` - Stan code for Model 2
- `designer_1_model3_powerlaw.stan` - Stan code for Model 3
- `designer_1_prior_predictive_checks.py` - Prior validation
- `designer_1_posterior_predictive_checks.py` - Model validation
- `designer_1_loo_comparison.py` - Model comparison
- `designer_1_results.md` - Findings and recommendations
- `designer_1_visualizations/` - All plots and diagnostics

---

**Designer**: Bayesian Model Designer 1
**Date**: 2025-10-27
**Status**: Ready for implementation
**Next Step**: Begin Stan model implementation and prior predictive checks
