# Parametric Bayesian Models for Y vs x Relationship
## Model Designer 1: Parametric Modeling Perspective

**Date**: 2025-10-28
**Designer**: Parametric Modeling Specialist
**Data**: N=27, x ∈ [1, 31.5], Y ∈ [1.71, 2.63]

---

## Executive Summary

Based on EDA findings showing strong non-linear saturation behavior, I propose **three distinct parametric model classes** representing fundamentally different data-generating mechanisms:

1. **Logarithmic Model** (Primary) - Unbounded slow growth
2. **Michaelis-Menten Model** (Alternative) - True asymptotic saturation
3. **Polynomial Model** (Baseline) - Flexible empirical fit

Each model makes different assumptions about the limiting behavior and should be evaluated not just on fit, but on **scientific plausibility** and **extrapolation behavior**.

---

## Critical Context from EDA

### Strong Evidence
- Spearman ρ=0.78 (p<0.001): Strong monotonic relationship
- Logarithmic model: R²=0.829, RMSE=0.115
- Quadratic model: R²=0.862, RMSE=0.103
- Saturation pattern: Y increases 0.27 units from low to high x region
- Variance appears approximately constant (BP test p=0.546)

### Key Uncertainties
- **Data gap** at x∈[23, 29]: 6.5-unit gap in predictor space
- **Influential point** at x=31.5 (Cook's D=0.84): 4% of data
- **Limited high-x data**: Only 7 observations for x>15
- **Variance trend**: Suggestive 4.6:1 decrease (low to high x) but not significant
- **Functional form ambiguity**: Only 3% R² difference between log and quadratic

### Critical Questions
1. Does Y have a true asymptote or just slow unbounded growth?
2. Is the variance truly constant or decreasing with x?
3. What is the true behavior in the gap region [23, 29]?

---

## Philosophy: Competing Hypotheses

I adopt a **falsificationist approach**: Each model represents a distinct scientific hypothesis about the data-generating process. Success means discovering which hypothesis is most consistent with data, not completing a predetermined plan.

### Three Competing Hypotheses

**H1 (Logarithmic)**: Y grows without bound following logarithmic law, common in information-theoretic or psychophysical processes (Weber-Fechner law).

**H2 (Michaelis-Menten)**: Y asymptotically approaches a finite upper limit, common in enzyme kinetics, resource-limited growth, or physical constraints.

**H3 (Polynomial)**: Y follows a polynomial trajectory, possibly quadratic with eventual downturn (though not observed yet).

These are **mutually exclusive in their limiting behavior**:
- Log: lim(x→∞) Y = ∞ (slow)
- MM: lim(x→∞) Y = Y_max (finite)
- Poly: lim(x→∞) Y = -∞ if negative leading coefficient

---

# Model Class 1: Logarithmic Regression (Primary Hypothesis)

## Scientific Justification

The logarithmic model represents processes where **response diminishes proportionally to the current level**. This is ubiquitous in:
- Psychophysics (Weber-Fechner law: perceived intensity ∝ log(stimulus))
- Information theory (bit complexity, Shannon entropy)
- Economics (marginal utility)
- Learning curves (time-on-task effects)

**Key assumption**: There is no fundamental upper bound on Y, but incremental gains require exponentially more x.

## Mathematical Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = α + β·log(x_i)
```

Where:
- `α`: Intercept (expected Y when x=1, since log(1)=0)
- `β`: Log-slope (change in Y per unit increase in log(x))
- `σ`: Residual standard deviation (constant across x)

### Prior Distributions

**Informed by EDA statistics**:

```stan
// Intercept: EDA estimate ≈ 1.75
// Y ranges [1.71, 2.63], so intercept should be in [1.5, 2.5]
// Use weakly informative prior centered at OLS estimate
α ~ Normal(1.75, 0.5)

// Log-slope: EDA estimate ≈ 0.27
// Must be positive (monotonic increase observed)
// Range: [0.1, 0.5] covers reasonable values
β ~ Normal(0.27, 0.15)

// Residual SD: EDA RMSE ≈ 0.115
// Observed Y std = 0.28, so residual should be < 0.3
σ ~ HalfNormal(0.2)
```

**Prior justification**:
- α prior: Allows ±1 SD = ±0.5 from OLS estimate, covering [1.25, 2.25] at 68% probability
- β prior: Centered at OLS but allows substantial deviation; ensures β>0 with ~96% probability
- σ prior: Mode at 0, median ≈ 0.135, 95th percentile ≈ 0.33; matches observed residual scale

### Stan Implementation

```stan
data {
  int<lower=0> N;               // Number of observations
  vector[N] x;                  // Predictor
  vector[N] Y;                  // Response
}

transformed data {
  vector[N] log_x;
  log_x = log(x);               // Precompute log(x)
}

parameters {
  real alpha;                   // Intercept
  real beta;                    // Log-slope
  real<lower=0> sigma;          // Residual SD
}

model {
  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  sigma ~ normal(0, 0.2);       // Half-normal via constraint

  // Likelihood
  Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] mu;                 // Mean predictions
  vector[N] y_rep;              // Posterior predictive samples
  vector[N] log_lik;            // Pointwise log-likelihood for LOO

  for (i in 1:N) {
    mu[i] = alpha + beta * log_x[i];
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }
}
```

## Expected Parameter Interpretations

Assuming posterior concentrates near OLS estimates:

- **α ≈ 1.75**: Expected Y at x=1 (baseline)
- **β ≈ 0.27**: Every 10-fold increase in x adds ~0.27·ln(10) ≈ 0.62 to Y
  - Example: x=1→10 increases Y by ~0.62, x=10→100 increases Y by another ~0.62
- **σ ≈ 0.12**: Typical deviation from mean is ~0.12, or ~5% of mean Y

## Prior Predictive Checks

**Before fitting**, simulate from priors to verify:

1. **Range constraint**: Y should stay roughly in [1, 3] for x∈[1, 32]
2. **Monotonicity**: All samples should be monotonic increasing
3. **Saturation**: Growth should slow as x increases
4. **No pathologies**: No extreme values, infinities, or negative Y

**Expected behavior**:
- At x=1: Y ~ Normal(1.75, 0.12) from mean, wider from prior uncertainty
- At x=10: Y ~ Normal(2.37, 0.12)
- At x=30: Y ~ Normal(2.67, 0.12)

## Falsification Criteria

**I will abandon this model if**:

1. **Posterior predictive p-value < 0.05** for test statistic = max(Y_rep) - max(Y_obs)
   - *Why*: If model systematically underestimates maximum Y, it suggests a true upper bound exists (favoring MM model)

2. **LOO-ELPD worse than MM model by >4** (strong evidence)
   - *Why*: If asymptotic model predicts better, data likely has true saturation

3. **Residuals show systematic pattern** in x>20 region (curved, not random)
   - *Why*: Log model assumes linear relationship with log(x); curvature suggests different form

4. **β posterior contains zero** (95% CI includes 0)
   - *Why*: Would indicate no log-linear relationship

5. **Extreme parameter values**: |α|>5 or β>2 or β<0
   - *Why*: Scientifically implausible; suggests model misspecification

6. **Prior-posterior conflict**: Posterior mass in tails of prior
   - *Why*: Priors fighting data indicates wrong model family

## Computational Considerations

**Stan vs PyMC**: Stan preferred for this simple model
- Faster for Gaussian likelihood
- HMC diagnostics well-developed
- Easy to compute LOO-CV

**Expected challenges**: None - this is a simple linear model in transformed space

**Diagnostics to check**:
- Rhat < 1.01 for all parameters
- Effective sample size > 400
- No divergent transitions (would be very surprising for this model)

**Sampling strategy**:
- 4 chains, 2000 iterations each (1000 warmup)
- Should complete in <10 seconds

## Stress Tests

1. **Influence analysis**: Refit without x=31.5; check if β changes by >20%
2. **Gap prediction**: Compute posterior predictive at x=25; check if 95% CI is reasonable
3. **Extrapolation**: Predict at x=50, x=100; check if growth rate is plausible
4. **Leave-high-x-out**: Fit on x<20 only; predict x>20; check coverage

---

# Model Class 2: Michaelis-Menten Saturation (Alternative Hypothesis)

## Scientific Justification

The Michaelis-Menten (MM) model represents processes with **true asymptotic limits**:
- Enzyme kinetics: Reaction rate saturates at maximum enzyme capacity
- Resource limitation: Growth limited by finite resource
- Physical constraints: System cannot exceed maximum capacity
- Dose-response curves: Effect plateaus at saturation

**Key assumption**: Y asymptotically approaches a finite maximum Y_max as x→∞.

## Mathematical Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = Y_max - (Y_max - Y_min)·K/(K + x_i)
```

Alternative parameterization (equivalent):
```
μ_i = Y_max·x_i/(K + x_i)  [if Y_min=0]
```

Where:
- `Y_max`: Asymptotic maximum (upper limit as x→∞)
- `Y_min`: Baseline minimum (Y at x=0)
- `K`: Half-saturation constant (x value where μ = (Y_max + Y_min)/2)
- `σ`: Residual standard deviation

**Why this form**: At x=K, exactly halfway between baseline and asymptote.

### Prior Distributions

**Informed by EDA**:

```stan
// Asymptotic maximum: Should be > max(Y)=2.63
// EDA suggests plateau around 2.5-2.6, but allow higher
Y_max ~ Normal(2.7, 0.3)  // Weakly informative, T+(2.63, ∞)

// Baseline: Should be ≈ min(Y) or lower
// min(Y)=1.71, but extrapolate to x→0
Y_min ~ Normal(1.5, 0.3)

// Half-saturation: x where Y = (Y_max + Y_min)/2 ≈ 2.1
// Looking at data, Y≈2.1 occurs around x≈3-5
K ~ Normal(5, 3)  // Positive support via constraint

// Residual SD: Same as log model
σ ~ HalfNormal(0.2)
```

**Prior justification**:
- Y_max: Centered slightly above max(Y), allowing uncertainty; truncate below max(Y) in Stan
- Y_min: Represents extrapolation to x=0; should be near min(Y) but uncertain
- K: From data, half-max appears around x=5; wide prior allows x∈[0, 15] at 95%
- Hierarchy: Y_max > Y_min enforced via constraint

### Stan Implementation

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}

parameters {
  real<lower=max(Y)> Y_max;         // Asymptote above max observed
  real Y_min;                        // Baseline
  real<lower=0> K;                   // Half-saturation
  real<lower=0> sigma;               // Residual SD
}

transformed parameters {
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = Y_max - (Y_max - Y_min) * K / (K + x[i]);
  }
}

model {
  // Priors
  Y_max ~ normal(2.7, 0.3);
  Y_min ~ normal(1.5, 0.3);
  K ~ normal(5, 3);
  sigma ~ normal(0, 0.2);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  real half_saturation_Y;

  half_saturation_Y = (Y_max + Y_min) / 2;

  for (i in 1:N) {
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }
}
```

## Expected Parameter Interpretations

Based on asymptotic model from EDA:

- **Y_max ≈ 2.6-2.8**: Maximum achievable Y (physical/biological limit)
- **Y_min ≈ 1.5-1.7**: Baseline Y at very low x
- **K ≈ 3-7**: x value where half saturation occurs
- **σ ≈ 0.12-0.14**: May be slightly higher than log model due to different functional form

**Interpretation aid**:
- At x=K: Y = (Y_max + Y_min)/2 (definition)
- At x=10K: Y ≈ Y_max - 0.1·(Y_max - Y_min) (90% saturated)
- At x=∞: Y = Y_max

## Prior Predictive Checks

1. **Asymptotic behavior**: All prior samples should approach Y_max as x→∞
2. **Monotonicity**: Should be monotonic increasing (guaranteed by form if Y_max > Y_min)
3. **Reasonable saturation**: Most prior samples should be >90% saturated by x=30
4. **Range**: Y should stay in [1, 3.5] for most samples

## Falsification Criteria

**I will abandon this model if**:

1. **Y_max posterior unbounded**: Posterior mean >10 or posterior keeps increasing
   - *Why*: Suggests no true asymptote exists (favoring log model)

2. **K posterior near or beyond max(x)**: Posterior mean K > 25
   - *Why*: Would mean saturation not yet observed in data; log model more appropriate

3. **Y_min posterior far from observed minimum**: Posterior mean Y_min < 1.0 or > 2.0
   - *Why*: Unphysical; suggests wrong functional form

4. **Divergent transitions or Rhat > 1.05**: Sampling problems
   - *Why*: MM models can have difficult geometry (correlations between Y_max and K); sampling issues suggest reparameterization needed or wrong model

5. **LOO-ELPD worse than log model by >4**
   - *Why*: Log model simpler and fits better = Occam's razor

6. **Posterior predictive shows systematic bias** at high x (consistently over/under)
   - *Why*: Model not capturing true saturation behavior

## Computational Considerations

**Stan vs PyMC**: Stan preferred but may need reparameterization
- MM models can have challenging posterior geometry
- Correlation between Y_max and K can cause issues
- **Alternative parameterization** if needed:
  ```
  θ1 = Y_max - Y_min  (range)
  θ2 = Y_min + θ1/2    (midpoint)
  θ3 = K               (scale)
  ```

**Expected challenges**:
- **Weak identifiability**: With only 7 high-x observations, Y_max may be poorly identified
- **Correlation**: Y_max and K typically negatively correlated
- **Divergences**: Possible in tails of posterior

**Diagnostics to check**:
- Pairs plot: Look for correlation between Y_max and K
- Divergent transitions: If >1%, try:
  - Increase adapt_delta to 0.95 or 0.99
  - Reparameterize (centered vs non-centered)
- Energy plot: Check for bias in MCMC

**Sampling strategy**:
- 4 chains, 3000 iterations (1500 warmup)
- adapt_delta = 0.9 (higher than default)
- May take 30-60 seconds

## Stress Tests

1. **Identifiability check**: Plot posterior samples of Y_max vs K; check if concentrated or diffuse
2. **High-x extrapolation**: Predict at x=100; should be very close to Y_max
3. **Prior sensitivity**: Refit with Y_max ~ Normal(3.0, 0.5); check if posterior changes substantially
4. **Asymptote test**: Compute P(Y_max < 2.8 | data); if low, asymptote may not exist

---

# Model Class 3: Quadratic Polynomial (Baseline/Comparison)

## Scientific Justification

The quadratic model represents a **flexible empirical fit** without strong mechanistic interpretation:
- Polynomial approximation to unknown smooth function
- Captures curvature parsimoniously (3 parameters)
- Local behavior (not for extrapolation)

**Key assumption**: Relationship is smooth and can be approximated by low-order polynomial in observed range.

**Warning**: Quadratic with negative x² coefficient will eventually decrease for large x, which is **scientifically implausible** unless we believe Y peaks then declines (no evidence for this).

## Mathematical Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = α + β₁·x_i + β₂·x_i²
```

Where:
- `α`: Intercept (Y at x=0)
- `β₁`: Linear coefficient (initial slope)
- `β₂`: Quadratic coefficient (curvature)
- `σ`: Residual standard deviation

**Expected sign**: β₁ > 0 (increasing), β₂ < 0 (diminishing returns)

### Prior Distributions

**Informed by EDA (quadratic OLS gave: α≈1.746, β₁≈0.086, β₂≈-0.002)**:

```stan
// Intercept: Should be near min(Y) or lower
α ~ Normal(1.7, 0.3)

// Linear term: Positive, but uncertainty high
// Initial slope dY/dx at x=0 is β₁
β₁ ~ Normal(0.1, 0.05)

// Quadratic term: Negative for saturation
// Magnitude controls curvature
β₂ ~ Normal(-0.002, 0.001)

// Residual SD
σ ~ HalfNormal(0.15)  // May be lower than log model
```

**Prior justification**:
- α: Extrapolated intercept, near lower range of Y
- β₁: Centered at OLS, wide enough to allow [0.01, 0.2] at 95%
- β₂: Centered at OLS; negative prior mean enforces expected curvature
- Note: These priors are weakly informative and allow data to dominate

### Stan Implementation

```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] x2;
  x2 = x .* x;  // Precompute x²
}

parameters {
  real alpha;
  real beta1;
  real beta2;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] mu;
  mu = alpha + beta1 * x + beta2 * x2;
}

model {
  // Priors
  alpha ~ normal(1.7, 0.3);
  beta1 ~ normal(0.1, 0.05);
  beta2 ~ normal(-0.002, 0.001);
  sigma ~ normal(0, 0.15);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  real vertex_x;  // x value at vertex (max/min of parabola)
  real vertex_Y;  // Y value at vertex

  // Vertex of parabola: x = -β₁/(2β₂)
  vertex_x = -beta1 / (2 * beta2);
  vertex_Y = alpha + beta1 * vertex_x + beta2 * vertex_x^2;

  for (i in 1:N) {
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }
}
```

## Expected Parameter Interpretations

- **α ≈ 1.75**: Extrapolated Y at x=0
- **β₁ ≈ 0.09**: Initial rate of increase (dY/dx at x=0)
- **β₂ ≈ -0.002**: Rate of deceleration; each unit increase in x reduces slope by 2·β₂·x
- **Vertex x ≈ -β₁/(2β₂) ≈ -0.09/(2·(-0.002)) ≈ 22.5**: Maximum Y occurs around x≈22
  - **Critical**: This is INSIDE the data range! If vertex_x < max(x), model predicts Y decreases beyond vertex

## Prior Predictive Checks

1. **Monotonicity in observed range**: For x∈[1, 32], Y should be increasing
2. **Vertex location**: Vertex should be well beyond max(x)=31.5, ideally >50
3. **Range**: Y should stay in [1.5, 3] for x∈[1, 32]
4. **No extreme curvature**: Parabola shouldn't be too steep

**Red flag**: If many prior samples show vertex at x<50, prior may be too informative or model inappropriate.

## Falsification Criteria

**I will abandon this model if**:

1. **Vertex in observed range**: Posterior P(vertex_x < 31.5) > 0.1
   - *Why*: Would predict Y decreases beyond observed x, contradicting monotonicity assumption

2. **LOO-ELPD worse than simpler log model by >2**
   - *Why*: Extra parameter not justified; Occam's razor favors simpler model

3. **β₂ posterior contains zero**: 95% CI includes 0
   - *Why*: No evidence for curvature; linear model sufficient

4. **Poor extrapolation behavior**: Predictions at x=40 or x=50 are wildly implausible
   - *Why*: Polynomial extrapolation unreliable; model only valid locally

5. **High posterior uncertainty**: Posterior SD for any parameter >3× prior SD
   - *Why*: Suggests weak identifiability or overfitting

6. **WAIC/LOO warnings**: High Pareto k values (k>0.7) for multiple points
   - *Why*: Influential points dominating fit; model unstable

## Computational Considerations

**Stan vs PyMC**: Either fine; Stan slightly faster

**Expected challenges**: Minimal - this is standard linear regression

**Diagnostics**:
- Standard HMC diagnostics should be clean
- Watch for correlation between β₁ and β₂

**Sampling strategy**:
- 4 chains, 2000 iterations (1000 warmup)
- Should complete in <10 seconds

## Stress Tests

1. **Extrapolation danger**: Plot posterior predictive mean and 95% CI for x∈[32, 50]; check for peak/downturn
2. **Vertex analysis**: Plot posterior distribution of vertex_x; should be far right of data
3. **Curvature test**: Compare posterior fit to spline fit; should match closely in observed range
4. **Leave-low-x-out**: Fit on x>5; predict x∈[1,5]; check if predictions reasonable (tests extrapolation in reverse)

---

# Model Comparison Strategy

## Multi-Stage Evaluation

### Stage 1: Individual Model Diagnostics (Within-Model)

**For each model separately**:

1. **Prior predictive checks**: Do priors generate reasonable data?
2. **MCMC diagnostics**: Rhat, ESS, divergences, energy
3. **Posterior predictive checks**: Does model reproduce observed patterns?
   - Y distribution (histogram, mean, SD, range)
   - X-Y relationship (does fit curve match data?)
   - Residual patterns (no systematic structure)
   - Replicate variability (can model explain within-x variation?)

**Pass criteria**: All diagnostics clean, PPC shows good calibration

### Stage 2: Between-Model Comparison

**Quantitative metrics**:

1. **LOO-CV (primary)**: Leave-one-out cross-validation using PSIS-LOO
   - Compare ELPD_loo (expected log pointwise predictive density)
   - Difference >4 = substantial evidence
   - Difference 2-4 = moderate evidence
   - Difference <2 = models roughly equivalent

2. **WAIC (secondary)**: Widely applicable information criterion
   - Should agree with LOO in direction
   - If disagree, investigate influential points

3. **Bayes Factor** (if needed): Direct comparison via bridge sampling
   - BF > 10 = strong evidence
   - BF > 100 = decisive evidence

**Model weights**: Compute stacking weights for model averaging if models are close.

### Stage 3: Scientific Plausibility

**Qualitative evaluation**:

1. **Parameter reasonableness**: Are estimates scientifically meaningful?
2. **Extrapolation behavior**: What does model predict for x=50, x=100, x=1000?
   - Log: Slow unbounded growth - plausible for many processes
   - MM: Approaches Y_max - plausible if physical limit exists
   - Poly: Eventually decreases - IMPLAUSIBLE without strong evidence

3. **Gap prediction**: What happens at x∈[23, 29]?
   - Do different models agree? If not, uncertainty is high
   - Are predictions consistent with extrapolation from both sides?

4. **Physical interpretation**: Can we tell a coherent story about the mechanism?

### Stage 4: Sensitivity Analysis

**Robustness checks**:

1. **Influence of x=31.5**: Refit all models excluding this point
   - Do conclusions change?
   - Does LOO ranking change?

2. **Prior sensitivity**: Refit best model with alternative priors
   - 2× wider priors
   - Different prior families
   - Check posterior robustness

3. **Subset analysis**:
   - Fit on x<15, predict x>15 (forward extrapolation)
   - Fit on x>10, predict x<10 (backward extrapolation)
   - Check predictive coverage

## Decision Rules

### Scenario A: Clear Winner (ΔLOO > 4)

**Action**: Adopt winning model as primary

**Additional checks**:
- Verify winner passes all diagnostics
- Ensure scientific plausibility
- Document why other models failed

### Scenario B: Close Competition (ΔLOO < 2)

**Action**: Use Bayesian model averaging
- Weight predictions by LOO stacking weights
- Report uncertainty across model classes
- Acknowledge structural uncertainty

**Interpretation**: Data insufficient to distinguish mechanisms

### Scenario C: All Models Fail Diagnostics

**Action**: STOP and reconsider
- Question: Are we missing a key feature (heteroscedasticity, outliers, nonlinearity type)?
- Consider:
  - Mixture models (regime changes)
  - Change-point models (structural break)
  - Robust regression (heavy-tailed errors)
  - Hierarchical variance structure

**This is success, not failure**: Learning that simple models are inadequate

### Scenario D: Best Model Fails Scientific Plausibility

**Action**: Report best-fitting model but flag concerns
- Example: Quadratic with vertex at x=25 fits best but predicts Y decreases beyond data
- Recommendation: Use for interpolation only, not extrapolation
- Consider: Constrained alternatives (monotonic GP, etc.)

---

# Red Flags Triggering Model Class Changes

## Evidence for More Complex Models

### Switch from constant to varying variance if:

1. **Residual analysis** shows clear fan shape (increasing or decreasing with x)
2. **Replicate variance** shows systematic pattern with x
3. **Posterior predictive checks** fail to capture variance structure
4. **LOO-PIT** (probability integral transform) shows mis-calibration

**New model**: Add heteroscedastic variance
```
σ_i = σ₀ + σ₁·f(x_i)
where f(x) could be: x, 1/x, log(x), exp(-x), etc.
```

### Switch to robust likelihood if:

1. **Outliers detected** in posterior predictive space
2. **Influential points** have Cook's D > 1
3. **Residuals** show heavy tails (|z| > 3 for multiple points)

**New model**: Student-t likelihood
```
Y_i ~ StudentT(ν, μ_i, σ)
ν ~ Gamma(2, 0.1)  // Degrees of freedom
```

### Switch to piecewise/change-point model if:

1. **Gap region** shows strong deviation from global model
2. **Low-x vs high-x** regions have different functional forms
3. **Residuals** show distinct patterns in different regions

**New model**: Piecewise linear or different functions in different regions

## Evidence for Simpler Models

### Drop to linear model if:

1. **Quadratic β₂** posterior contains zero
2. **Log β** posterior ≈ 0
3. **LOO** strongly favors linear over all nonlinear

**New model**: Simple linear regression
```
μ_i = α + β·x_i
```

### Drop to intercept-only if:

1. **All slope parameters** contain zero
2. **Correlation tests** become non-significant in Bayesian framework
3. **LOO** favors null model

**This would be shocking given EDA**, but would represent important discovery

---

# Expected Outcomes and Interpretation

## Most Likely Scenario

Based on EDA, I predict:

**Log model**: Best balance of fit, parsimony, and plausibility
- LOO-ELPD ≈ -5 to 5 (arbitrary scale)
- Posterior: α≈1.75±0.1, β≈0.27±0.05, σ≈0.12±0.02
- Clean diagnostics, good posterior predictive fit

**MM model**: Similar fit, possibly slightly worse due to extra parameter
- LOO-ELPD ≈ -2 to -8 (slightly worse than log)
- Posterior: Y_max≈2.7±0.15, K≈5±3, σ≈0.12±0.02
- May show weak identifiability (wide Y_max posterior)
- Some correlation between Y_max and K

**Quadratic**: Best fit but concerns about extrapolation
- LOO-ELPD ≈ 0 to 2 (slightly better than log)
- Posterior: β₂≈-0.002±0.0005
- Vertex at x≈20-30 (concerning)
- Good fit in range, dangerous outside

**Conclusion**: Adopt log model unless MM model shows strong evidence of true asymptote

## Alternative Scenario 1: MM Model Wins Decisively

**If**: LOO-ELPD(MM) - LOO-ELPD(log) > 4 AND Y_max posterior is well-identified

**Interpretation**:
- Strong evidence for true asymptotic limit
- Data sufficient to distinguish bounded vs unbounded growth
- Saturation is real, not just apparent

**Action**:
- Report Y_max with 95% credible interval
- Recommend focusing on mechanisms that produce saturation
- Consider implications for extrapolation (plateaus, not continues)

## Alternative Scenario 2: Quadratic Required Despite Concerns

**If**: Both log and MM fail posterior predictive checks, but quadratic passes

**Interpretation**:
- True relationship more complex than simple monotonic functions
- May indicate regime change or inflection point
- Polynomial approximation needed locally

**Action**:
- Use quadratic for interpolation only
- DO NOT extrapolate beyond x=35
- Flag need for more data at high x to understand limiting behavior
- Consider: Is there a scientific reason for non-monotonic or complex curvature?

## Alternative Scenario 3: All Models Inadequate

**If**: All models show LOO Pareto k > 0.7 for multiple points OR fail PPC badly

**Interpretation**:
- Missing key structure (heteroscedasticity, outliers, nonlinearity type)
- Simple parametric models insufficient
- Need more flexible approach

**Action**:
- Escalate to designer teams proposing:
  - Gaussian processes (designer 2, likely)
  - Hierarchical/robust models
  - Mixture models
  - State-space models

**This is success**: Discovered that standard toolkit is insufficient

---

# Computational Implementation Plan

## Software Choices

**Primary**: Stan via CmdStanPy
- Reason: Fast, excellent diagnostics, gold standard for simple models
- LOO-CV: Use arviz package

**Backup**: PyMC
- Reason: If Stan shows issues (unlikely for these models)
- Use NUTS sampler, same diagnostics

## Workflow

### Step 1: Prior Predictive Checks (Pre-Fitting)

```python
# For each model:
# 1. Sample from prior
# 2. Simulate Y_rep
# 3. Plot prior predictive envelope
# 4. Check ranges, monotonicity, plausibility
# 5. STOP if priors produce nonsense
```

### Step 2: Model Fitting

```python
# For each model:
# 1. Compile Stan model
# 2. Run MCMC: 4 chains, 2000 iter, 1000 warmup
# 3. Check diagnostics immediately
# 4. If divergences/Rhat issues:
#    - Increase adapt_delta
#    - Try reparameterization
#    - Check priors
# 5. Extract posterior samples
```

### Step 3: Posterior Diagnostics

```python
# For each model:
# 1. Check Rhat (all < 1.01)
# 2. Check ESS (all > 400)
# 3. Check divergences (< 1%)
# 4. Energy plot
# 5. Pairs plot for correlations
# 6. Prior-posterior overlap
```

### Step 4: Posterior Predictive Checks

```python
# For each model:
# 1. Generate posterior predictive samples
# 2. Compare distributions: Y_rep vs Y_obs
# 3. Test statistics:
#    - Mean, SD, min, max
#    - Skewness, kurtosis
#    - Correlation with x
# 4. Visual: overlay predicted vs observed
# 5. Residual analysis: any patterns?
```

### Step 5: Model Comparison

```python
# Compare all models:
# 1. Compute LOO-CV for each
# 2. Compare ELPD differences
# 3. Check Pareto k diagnostics
# 4. Compute model weights
# 5. If close: use stacking
```

### Step 6: Sensitivity Analysis

```python
# For best model(s):
# 1. Refit without x=31.5
# 2. Refit with wider priors
# 3. Subset cross-validation
# 4. Compare results
```

## Expected Runtime

- Prior predictive: ~1 min per model
- MCMC fitting: ~10-60 sec per model
- LOO computation: ~5 sec per model
- PPC: ~2 min per model

**Total**: ~30 minutes for all analyses including visualization

---

# Success Criteria and Stopping Rules

## Success Defined

**Not**: Completing all three models
**But**: Finding the model that best explains the data-generating process

**Markers of success**:
1. Understanding why rejected models failed
2. Clear evidence distinguishing competing hypotheses
3. Reliable predictions with well-calibrated uncertainty
4. Scientific story consistent with domain knowledge

## Stopping Rules

### Stop and Declare Success If:

1. **One model clearly superior**: ΔLOO>4, passes all diagnostics, scientifically plausible
2. **Multiple models equivalent**: ΔLOO<2, use model averaging, acknowledge uncertainty
3. **Learned something unexpected**: E.g., discovered heteroscedasticity, need for robust likelihood

### Stop and Escalate If:

1. **All models fail diagnostics**: Fundamental misspecification
2. **LOO shows high Pareto k** (>0.7) for >5 points: Influential points driving conclusions
3. **Posterior predictive p-value <0.05** for multiple test statistics across all models
4. **Scientific implausibility**: Best statistical model makes no domain sense

### Stop and Collect More Data If:

1. **High uncertainty in critical parameters**: Wide posteriors prevent conclusions
2. **Gap region**: Predictions at x∈[23,29] have >50% relative uncertainty
3. **Extrapolation needed**: If decisions require knowing behavior beyond x=40

## What Would Make Me Completely Change Approach?

**Abandon parametric models entirely if**:

1. **Residuals show complex patterns** not explainable by simple variance models
2. **Data suggests regime changes** or non-stationary behavior
3. **Strong local wiggles** that parametric curves cannot capture
4. **Evidence of periodicity or cycles** (would be very surprising)

**Pivot to**:
- Gaussian processes (flexible non-parametric)
- Splines with knots (semi-parametric)
- Mixture models (multiple regimes)
- State-space models (if temporal/sequential structure)

---

# Summary and Priority Ranking

## Recommended Priority Order

### Priority 1: Logarithmic Model (Primary)
**Why first**:
- Best balance of fit, parsimony, plausibility
- Strong theoretical foundation
- Robust to extrapolation
- Simple to fit and interpret
- EDA evidence: R²=0.829, clear saturation pattern

**Expected outcome**: This will likely be the final choice

### Priority 2: Michaelis-Menten Model (Key Alternative)
**Why second**:
- Directly tests hypothesis of true asymptote vs unbounded growth
- If distinguishable from log, very informative
- Scientific question: Is there a physical limit?

**Expected outcome**: Likely similar fit to log, harder to identify Y_max

### Priority 3: Quadratic Model (Baseline)
**Why third**:
- Empirically best fit from EDA (R²=0.862)
- But concerns about extrapolation
- Useful as sensitivity check
- If this is necessary, suggests both log and MM are inadequate

**Expected outcome**: Best ELPD but vertex concerns; use for interpolation only

## If Time/Resources Limited

**Minimum viable analysis**:
1. Fit log model thoroughly (diagnostics, PPC, sensitivity)
2. Fit MM model for comparison
3. Report LOO difference and interpretation

**Skip quadratic if**: Log and MM both adequate and distinguishable

## Final Thought: Embracing Uncertainty

The EDA shows ΔLOO between log and quadratic is only ~3%. With N=27, distinguishing these models conclusively may be impossible. **This is an important finding**, not a failure.

If models are equivalent (ΔLOO<2), the honest conclusion is: **"Data consistent with multiple mechanisms; structural uncertainty is high."**

This motivates:
- Model averaging for robust predictions
- Collecting targeted data to distinguish models
- Acknowledging limitations of current data

**Truth-seeking means admitting what we don't know.**

---

# Appendix: Prior Elicitation Details

## Logarithmic Model Priors

### α ~ Normal(1.75, 0.5)

**Reasoning**:
- Interpretation: Y value when x=1 (since log(1)=0)
- From data: Y(x=1) = 1.71 (single observation)
- From OLS: α̂ = 1.751
- Prior: Centers at OLS, SD=0.5 allows ±1 from estimate
- Implied range: [0.75, 2.75] at 95%, covering [min(Y), max(Y)]

**Prior predictive**: P(α < 1) ≈ 0.07, P(α > 2.5) ≈ 0.07

### β ~ Normal(0.27, 0.15)

**Reasoning**:
- Interpretation: Change in Y per unit change in log(x)
- From OLS: β̂ = 0.275
- From data: Y increases ~0.9 over log(x) range of ~3.45, so β≈0.26
- Prior: Centers at OLS, SD=0.15 allows substantial uncertainty
- Constraint: β must be positive (monotonic increase)

**Prior predictive**: P(β < 0) ≈ 0.04, P(β > 0.6) ≈ 0.01
- Mostly positive (correct sign)
- Rarely extreme

### σ ~ HalfNormal(0.2)

**Reasoning**:
- Interpretation: Residual standard deviation
- From OLS RMSE: σ̂ ≈ 0.115
- From data: SD(Y) = 0.28 (upper bound on residual SD)
- Prior: HalfNormal(0.2) has median≈0.135, 95%ile≈0.33

**Prior predictive**:
- P(σ < 0.05) ≈ 0.11 (too small - overfitting)
- P(σ > 0.3) ≈ 0.07 (too large - underfitting)
- Most mass in [0.05, 0.25], reasonable range

## Michaelis-Menten Priors

### Y_max ~ Normal(2.7, 0.3) truncated below max(Y)=2.63

**Reasoning**:
- Interpretation: Asymptotic maximum Y
- Constraint: Must be > max(Y) = 2.63
- From asymptotic model fit: Y∞ ≈ 2.52 (likely underestimate)
- Prior: Centers at 2.7, SD=0.3
- After truncation: Implies Y_max ∈ [2.63, 3.3] at 95%

**Prior predictive**:
- P(Y_max > 3.5) ≈ 0.004 (very high asymptote unlikely)
- Mode near 2.7 (slightly above observed max)

### Y_min ~ Normal(1.5, 0.3)

**Reasoning**:
- Interpretation: Baseline Y at x→0 (extrapolated)
- From data: min(Y) = 1.71 at x=1
- Extrapolation: At x=0, expect Y≈1.5-1.7
- Prior: Centers at 1.5, allows range [0.9, 2.1] at 95%

**Prior predictive**: P(Y_min > Y_max) handled by ordering constraint in Stan

### K ~ Normal(5, 3) with K>0

**Reasoning**:
- Interpretation: Half-saturation point
- From data: Y≈2.1 (half-max) occurs around x=3-5
- Prior: Centers at 5, SD=3 allows K∈[0, 11] at 95%
- Positive constraint enforced

**Prior predictive**:
- P(K > 15) ≈ 0.0004 (saturation far beyond data unlikely)
- P(K < 1) ≈ 0.09 (very rapid saturation possible but unlikely)

## Quadratic Model Priors

### α ~ Normal(1.7, 0.3)

**Reasoning**:
- Interpretation: Y at x=0 (extrapolated intercept)
- From OLS: α̂ = 1.746
- Should be near lower range of Y
- Prior: Tight around OLS estimate

### β₁ ~ Normal(0.1, 0.05)

**Reasoning**:
- Interpretation: Linear slope (initial rate)
- From OLS: β̂₁ = 0.086
- Must be positive (initial increase)
- Prior: Fairly tight, favors positivity

**Prior predictive**: P(β₁ < 0) ≈ 0.02 (rarely negative)

### β₂ ~ Normal(-0.002, 0.001)

**Reasoning**:
- Interpretation: Curvature (deceleration)
- From OLS: β̂₂ = -0.002
- Should be negative (concave down)
- Prior: Centered at OLS, allows range [-0.004, 0] at 95%

**Prior predictive**: P(β₂ > 0) ≈ 0.02 (rarely positive = accelerating growth, implausible)

---

# Implementation Checklist

Before beginning MCMC fitting:

- [ ] Read and visualize raw data
- [ ] Implement all three Stan models
- [ ] Run prior predictive checks for each model
- [ ] Verify priors produce reasonable Y ranges
- [ ] Check for prior-data conflicts (prior predictive vs observed data)

During fitting:

- [ ] Monitor MCMC diagnostics in real-time
- [ ] If divergences occur, increase adapt_delta before continuing
- [ ] Save all posterior samples
- [ ] Compute LOO-CV immediately after fitting

After fitting:

- [ ] Generate posterior predictive samples
- [ ] Create comparison plots (observed vs predicted)
- [ ] Compute LOO differences and standard errors
- [ ] Run sensitivity analyses
- [ ] Make final model selection decision

**Documentation**: Save all diagnostic plots, LOO results, and posterior summaries

---

**END OF PROPOSAL**

**Key Files**:
- This proposal: `/workspace/experiments/designer_1/proposed_models.md`
- Stan models: To be created in `/workspace/experiments/designer_1/stan_models/`
- Results: To be saved in `/workspace/experiments/designer_1/results/`

**Next Steps**: Await approval to proceed with implementation, or incorporate feedback from main agent/other designers.
