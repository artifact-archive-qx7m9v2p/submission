# Bayesian Model Design: Flexible & Adaptive Approaches
## Designer 2 - Piecewise and Hierarchical Models

**Date**: 2025-10-27
**Dataset**: 27 observations, x ∈ [1.0, 31.5], Y ∈ [1.71, 2.63]
**Core Pattern**: Nonlinear saturation with evidence of regime shift at x ≈ 9-10
**Design Philosophy**: Favor models that can adapt to sharp vs. smooth transitions

---

## Executive Summary

I propose **3 distinct model classes** that emphasize flexibility and adaptability:

1. **Bayesian Change-Point Model** - Explicitly infers breakpoint location (sharp transition)
2. **Hierarchical Spline Model** - Smooth transition with adaptive basis functions
3. **Mixture-of-Experts Model** - Data-driven regime weighting (hybrid approach)

Each model is designed with:
- Explicit falsification criteria
- Computational efficiency considerations
- Clear escape routes if evidence contradicts assumptions

**Critical Stance**: The apparent regime shift at x=9-10 might be an artifact of sparse sampling. I will abandon piecewise approaches if posterior breakpoint uncertainty is extreme (SD > 5 units) or if smooth models dramatically outperform in LOO-CV.

---

## Model 1: Bayesian Change-Point Regression

### Hypothesis
The data generation process involves a **sharp regime transition** at an unknown breakpoint, with distinct linear relationships before and after.

### Model Specification

#### Likelihood
```
Y_i ~ Normal(μ_i, σ)

μ_i = {
    β₀ + β₁ * x_i                           if x_i ≤ τ
    β₀ + β₁ * τ + β₂ * (x_i - τ)            if x_i > τ
}
```

**Constraint for continuity**: The function is continuous at τ, but the slope changes from β₁ to β₂.

#### Priors

```python
# Breakpoint location - informed by EDA
τ ~ Normal(9.5, 2.0)  # 95% CI: [5.5, 13.5], centered on visual break

# Intercept - informed by low-x observations
β₀ ~ Normal(1.7, 0.2)  # Near minimum observed Y

# Slope before breakpoint - positive, moderate
β₁ ~ Normal(0.08, 0.03)  # EDA shows ~0.08 slope in low regime

# Slope after breakpoint - near zero or slightly positive
β₂ ~ Normal(0.0, 0.02)  # Weakly centered at zero (plateau)

# Residual SD - from pure error estimates
σ ~ Half-Cauchy(0, 0.15)  # Weakly informative, compatible with replicates
```

#### Stan Implementation Notes

**Challenge**: Discontinuous indicator functions (x ≤ τ) can cause gradient issues.

**Solution**: Use continuous approximation via logistic function:
```stan
// Smooth approximation to indicator (sharpness parameter k=100)
real smooth_indicator(real x, real tau, real k) {
  return inv_logit(k * (x - tau));
}

// In model block:
mu[i] = beta0 + beta1 * x[i] +
        (beta2 - beta1) * smooth_indicator(x[i], tau, 100) * (x[i] - tau);
```

**Alternative**: Use discrete parameter (slower, requires marginalization):
```stan
// Discretize tau over grid [1, 31] and marginalize
vector[31] log_lik_tau;
for(t in 1:31) {
  log_lik_tau[t] = normal_lpdf(tau_prior | t, 9.5, 2.0);
  for(i in 1:N) {
    if(x[i] <= t) {
      log_lik_tau[t] += normal_lpdf(Y[i] | beta0 + beta1*x[i], sigma);
    } else {
      log_lik_tau[t] += normal_lpdf(Y[i] | beta0 + beta1*t + beta2*(x[i]-t), sigma);
    }
  }
}
tau = categorical_logit_rng(log_lik_tau);
```

### Theoretical Justification

**Why this approach?**
- EDA shows correlation drops from 0.94 (x<10) to -0.03 (x≥10) - suggestive of regime change
- Piecewise linear achieved best empirical fit (R²=0.904)
- Physically plausible: Some processes have threshold effects (e.g., saturation of enzyme, phase transitions)
- Bayesian approach quantifies breakpoint uncertainty (vs. fixed-grid search in frequentist methods)

**Scientific interpretation**:
- τ represents the "saturation threshold" where the underlying process changes
- β₁ represents the "active regime" rate
- β₂ represents the "saturated regime" rate (expected ≈ 0)

### Falsification Criteria

**I will abandon this model if:**

1. **Breakpoint is highly uncertain**: Posterior SD(τ) > 5 units
   - Indicates no clear regime shift; data consistent with smooth transition
   - Action: Switch to spline or smooth nonlinear model

2. **Breakpoint hits boundary**: Posterior median τ < 3 or τ > 25
   - Suggests model is trying to escape to a single regime
   - Action: Simplify to single linear or nonlinear model

3. **Slopes are similar**: Posterior P(|β₁ - β₂| < 0.02) > 0.5
   - No evidence for distinct regimes
   - Action: Use simpler model without breakpoint

4. **Poor predictive performance**: LOO-ELPD worse than smooth alternatives by >4 (2×SE)
   - Overfitting or wrong functional form
   - Action: Favor smooth models

5. **Divergent transitions**: >5% after tuning adapt_delta=0.95
   - Geometry issues indicate model misspecification
   - Action: Reparameterize or abandon

### Expected Performance

- **R²**: 0.88-0.92 (match or exceed piecewise OLS at 0.904)
- **Convergence**: Should converge with adapt_delta=0.9, 2000 iterations
- **Posterior τ**: Expect median ≈ 9-10, SD ≈ 1-2 units
- **Computational**: ~30-60 seconds for 4000 iterations (2000 warmup + 2000 sampling)

### Stress Tests

1. **Synthetic sharp transition**: Generate data with known τ=10, verify recovery within 95% CI
2. **Synthetic smooth transition**: Generate from exponential model, check if model forces artificial breakpoint
3. **Leave-one-out sensitivity**: Remove observations near suspected breakpoint, check stability

---

## Model 2: Hierarchical B-Spline Regression

### Hypothesis
The relationship is **smoothly nonlinear** without discrete regime changes. The apparent "break" is an artifact of sampling density.

### Model Specification

#### Likelihood
```
Y_i ~ Normal(μ_i, σ)

μ_i = Σⱼ₌₁ᴷ βⱼ * Bⱼ(x_i)

where Bⱼ(x) are B-spline basis functions (cubic, degree=3)
```

**Knot placement**: Use quantile-based knots to ensure even coverage:
- K = 5 basis functions (3 interior knots at 33rd, 50th, 67th percentiles)
- Interior knots at x ≈ [5.5, 9.0, 14.0] based on data quantiles

#### Priors

```python
# Spline coefficients - hierarchical shrinkage
β ~ MultivariateNormal(μ_β * 1, Σ_β)

# Global mean (overall level)
μ_β ~ Normal(2.3, 0.3)  # Data mean

# Hierarchical variance (controls smoothness)
Σ_β = diag(τ²)
τ ~ Half-Cauchy(0, 0.5)  # Regularization on coefficient variation

# Alternative: Second-order random walk prior (penalized smoothness)
β₁ ~ Normal(μ_β, τ)
βⱼ ~ Normal(2*βⱼ₋₁ - βⱼ₋₂, τ²)  for j=2,...,K

# Residual SD
σ ~ Half-Cauchy(0, 0.15)
```

#### Adaptive Variant: Data-Driven Knot Placement

```python
# Allow knot locations to vary (computationally expensive)
knot_k ~ Uniform(x_min_k, x_max_k)  # Constrained intervals

# Or: Automatic relevance determination on basis functions
β_j ~ Normal(0, λ_j)
λ_j ~ Half-Cauchy(0, 1)  # Sparse coefficients
```

#### Stan Implementation Notes

**Efficient approach**: Pre-compute B-spline basis matrix in R/Python, pass as data:

```stan
data {
  int<lower=1> N;
  int<lower=1> K;  // Number of basis functions
  vector[N] Y;
  matrix[N, K] B;  // Pre-computed B-spline basis
}

parameters {
  vector[K] beta;
  real<lower=0> tau;
  real<lower=0> sigma;
}

model {
  vector[N] mu = B * beta;

  // Priors
  beta ~ normal(0, tau);
  tau ~ cauchy(0, 0.5);
  sigma ~ cauchy(0, 0.15);

  // Likelihood
  Y ~ normal(mu, sigma);
}
```

**Libraries**: Use `splines2` (R) or `patsy` (Python) to generate B-spline basis.

### Theoretical Justification

**Why this approach?**
- Agnostic to exact functional form (data-driven flexibility)
- Can represent any smooth nonlinear relationship
- Hierarchical shrinkage prevents overfitting with limited data (N=27)
- No assumption of sharp breakpoint required
- Widely used in GAMs (Generalized Additive Models)

**Advantages over parametric nonlinear**:
- No initialization issues (linear in coefficients)
- Fast sampling (Gaussian posterior conditionals)
- Can capture unanticipated wiggles

**Smoothness control**:
- τ controls trade-off between fit and smoothness
- Low τ → smooth (rigid)
- High τ → flexible (wiggly)

### Falsification Criteria

**I will abandon this model if:**

1. **Overfitting**: Posterior predictive checks show wild oscillations between data points
   - Indicates insufficient regularization
   - Action: Increase shrinkage or reduce K

2. **Underfitting**: Model cannot capture saturation (R² < 0.80)
   - Too few knots or over-smoothed
   - Action: Increase K or relax shrinkage

3. **Knot sensitivity**: Results change drastically with different knot placements
   - Model instability
   - Action: Use more knots or switch to Gaussian Process

4. **Poor extrapolation**: Posterior predictions go wild outside [1, 31.5]
   - Splines notoriously bad at extrapolation
   - Action: Add boundary constraints or switch to parametric model

5. **Computational issues**: Fails to converge or takes >5 minutes
   - Geometry problems
   - Action: Simplify to fixed-knot or lower K

### Expected Performance

- **R²**: 0.85-0.90 (competitive with parametric nonlinear)
- **Convergence**: Should be fast (<30 sec) due to linear structure
- **Smoothness**: Expect posterior τ to favor moderate smoothness
- **Computational**: Very efficient (conjugate-like updates)

### Stress Tests

1. **Cross-validation**: 5-fold CV to check sensitivity to data subsets
2. **Varying K**: Test K ∈ {3, 5, 7} to assess knot number sensitivity
3. **Prior sensitivity**: Compare τ ~ Half-Cauchy(0, 0.5) vs. τ ~ Half-Normal(0, 0.3)

---

## Model 3: Mixture-of-Experts (Gating Network)

### Hypothesis
The data generation process involves **soft regime mixing** - both linear and saturating mechanisms contribute, with weights varying by x.

### Model Specification

#### Likelihood
```
Y_i ~ Normal(μ_i, σ)

μ_i = π(x_i) * μ₁(x_i) + [1 - π(x_i)] * μ₂(x_i)

where:
  μ₁(x) = β₀ + β₁ * x              # Linear component (active regime)
  μ₂(x) = α                         # Constant component (saturated regime)
  π(x) = logit⁻¹(γ₀ + γ₁ * x)      # Gating function (regime probability)
```

**Interpretation**:
- At low x: π(x) ≈ 1 → linear behavior dominates
- At high x: π(x) ≈ 0 → constant plateau dominates
- Transition smoothness controlled by γ₁

#### Priors

```python
# Linear expert parameters
β₀ ~ Normal(1.7, 0.2)  # Intercept
β₁ ~ Normal(0.08, 0.03)  # Slope (positive)

# Constant expert (plateau)
α ~ Normal(2.55, 0.1)  # Asymptote from EDA

# Gating network (controls transition)
γ₀ ~ Normal(0, 2)  # Unconstrained location parameter
γ₁ ~ Normal(-0.5, 0.3)  # Negative → decreasing π(x)

# Residual SD
σ ~ Half-Cauchy(0, 0.15)
```

#### Stan Implementation

```stan
parameters {
  real beta0;
  real<lower=0> beta1;  // Constrain slope positive
  real alpha;
  real gamma0;
  real<upper=0> gamma1;  // Constrain negative (decreasing weight)
  real<lower=0> sigma;
}

model {
  vector[N] mu1;
  vector[N] mu2;
  vector[N] pi_x;
  vector[N] mu;

  // Expert predictions
  mu1 = beta0 + beta1 * x;
  mu2 = rep_vector(alpha, N);

  // Gating weights
  pi_x = inv_logit(gamma0 + gamma1 * x);

  // Mixture
  mu = pi_x .* mu1 + (1 - pi_x) .* mu2;

  // Priors
  beta0 ~ normal(1.7, 0.2);
  beta1 ~ normal(0.08, 0.03);
  alpha ~ normal(2.55, 0.1);
  gamma0 ~ normal(0, 2);
  gamma1 ~ normal(-0.5, 0.3);
  sigma ~ cauchy(0, 0.15);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  // Compute effective breakpoint (where π=0.5)
  real tau_eff = -gamma0 / gamma1;
}
```

### Theoretical Justification

**Why this approach?**
- Combines strengths of both piecewise (interpretable regimes) and smooth (continuous transition)
- Gating network learns transition sharpness from data
- Can represent sharp breaks (large |γ₁|) or gradual shifts (small |γ₁|)
- Effective breakpoint τ_eff = -γ₀/γ₁ is derived, not imposed
- Used in machine learning for heterogeneous data

**Flexibility hierarchy**:
- If γ₁ → -∞: Sharp transition (approaches piecewise)
- If γ₁ ≈ -0.5: Smooth transition over ~10 units
- If γ₁ ≈ 0: No transition (global mixture)

### Falsification Criteria

**I will abandon this model if:**

1. **Gating is uninformative**: Posterior π(x) ≈ 0.5 for all x
   - Data doesn't support regime distinction
   - Action: Simplify to single-component model

2. **Identifiability issues**: Strong posterior correlations (ρ > 0.95) between β₁ and α
   - Cannot distinguish expert contributions
   - Action: Use stronger priors or simpler model

3. **Unrealistic gating**: τ_eff outside [1, 31.5] or posterior SD(τ_eff) > 10
   - Transition point is unconstrained by data
   - Action: Switch to fixed-breakpoint or parametric model

4. **Worse than components**: LOO-ELPD lower than both single linear and single constant models
   - Overfitting penalty exceeds mixture benefit
   - Action: Abandon mixture, use simpler expert

5. **Computational failure**: Divergences or multimodality in posterior
   - Label switching or geometry issues
   - Action: Use ordered constraints or abandon

### Expected Performance

- **R²**: 0.87-0.91 (should match or exceed parametric alternatives)
- **Convergence**: May be slower due to nonlinear gating (~60-120 sec)
- **Effective breakpoint**: Expect τ_eff ≈ 8-11 with SD ≈ 2-3
- **Transition sharpness**: Posterior γ₁ will reveal sharp vs. smooth

### Stress Tests

1. **Synthetic sharp mixture**: Generate with γ₁ = -5, verify recovery
2. **Synthetic smooth mixture**: Generate with γ₁ = -0.1, verify recovery
3. **Expert swap**: Check if model is invariant to expert label (identifiability)

---

## Model Comparison Strategy

### Philosophical Approach

**Key Question**: Does the data truly have a regime shift, or is the pattern smoothly nonlinear?

**Evidence for sharp transition** (favor Model 1):
- Posterior τ is well-constrained (SD < 2)
- Piecewise LOO-ELPD >> Spline LOO-ELPD (Δ > 4)
- Posterior predictive checks show clear "kink"

**Evidence for smooth transition** (favor Model 2):
- Spline LOO-ELPD ≥ Piecewise LOO-ELPD
- Piecewise posterior τ is highly uncertain
- Residuals don't cluster by regime

**Evidence for hybrid** (favor Model 3):
- Mixture outperforms both piecewise and pure smooth
- Effective breakpoint well-defined but transition gradual

### Decision Tree

```
1. Fit all three models
   ├─ Check convergence (R-hat < 1.01)
   └─ Run posterior predictive checks

2. Compare LOO-ELPD
   ├─ If ΔELPD < 2*SE: Models statistically tied
   │  └─ Prefer simpler/more interpretable (likely Model 1 or 2)
   └─ If ΔELPD > 2*SE: Clear winner
      └─ Report winner, but check falsification criteria

3. Examine posterior breakpoint (Models 1 & 3)
   ├─ If SD(τ) > 5: Reject piecewise, favor smooth
   └─ If SD(τ) < 2: Strong evidence for regime shift

4. Sensitivity analysis
   ├─ Prior robustness: Re-fit with diffuse priors
   ├─ Knot sensitivity (Model 2): Try K ∈ {3, 5, 7}
   └─ Outlier influence: Refit without x=31.5

5. Final recommendation
   └─ Report top model + honest assessment of uncertainty
```

---

## Red Flags & Decision Points

### Critical Warning Signs

**STOP and reconsider everything if:**

1. **All models fail LOO**: Pareto-k > 0.7 for multiple observations
   - Indicates fundamental model misspecification
   - Action: Consider heteroscedastic variance, outlier accommodation (Student-t), or transformations

2. **Prior-posterior conflict**: Posterior far from prior despite "informative" prior
   - EDA priors were wrong or data has unexpected feature
   - Action: Use diagnostic plots to identify conflict source

3. **Convergence failures persist**: Divergences or R-hat > 1.01 after extensive tuning
   - Model geometry fundamentally flawed
   - Action: Reparameterize (non-centered) or abandon model class

4. **Predictive failure**: Posterior predictive R² < 0.75 for all models
   - Missing key covariate or wrong likelihood family
   - Action: Return to EDA, consider alternative data generating processes

5. **Extrapolation disaster**: Predictions for x ∈ [25, 35] have 95% CI width > 1.0
   - Insufficient data in high-x region, model unreliable
   - Action: Report limited extrapolation ability, suggest more data

### Major Strategy Pivots

**Pivot Point 1**: After initial fits (Day 1)
- **Trigger**: If all three models have LOO-ELPD within 2*SE
- **Action**: Add Gaussian Process as "maximum flexibility" benchmark
- **Rationale**: Current models may all be too restrictive

**Pivot Point 2**: After posterior predictive checks (Day 2)
- **Trigger**: Systematic residual patterns (e.g., heteroscedasticity, non-normality)
- **Action**: Add Student-t likelihood or variance modeling
- **Rationale**: Gaussian likelihood may be inadequate

**Pivot Point 3**: After sensitivity analysis (Day 3)
- **Trigger**: Results highly sensitive to prior choice or knot placement
- **Action**: Switch to fully nonparametric (GP) or ensemble approach
- **Rationale**: Model structure uncertainty dominates parameter uncertainty

---

## Alternative Approaches (If All Fail)

### Escape Route 1: Gaussian Process Regression

**When to use**: If parametric forms cannot capture data features

```python
Y ~ Normal(f(x), σ)
f(x) ~ GP(0, K(x, x'))

# Kernel options:
# 1. Squared Exponential: K(x,x') = η² * exp(-ρ² * (x-x')²)
# 2. Matérn 3/2: More flexible, allows non-smooth functions
# 3. Composite: Sum of SE + Linear (captures global trend + local variation)

# Priors:
η ~ Half-Normal(0, 0.5)  # Amplitude
ρ ~ Inv-Gamma(5, 5)      # Length scale (E[ρ]=1.25)
σ ~ Half-Cauchy(0, 0.15) # Noise
```

**Pros**: Maximum flexibility, well-calibrated uncertainty
**Cons**: Slow for N>50, poor extrapolation, hard to interpret

### Escape Route 2: Robust Regression (Student-t)

**When to use**: If outliers or heavy tails evident

```python
# Replace Normal likelihood with Student-t:
Y ~ Student-t(ν, μ(x), σ)

# Prior on degrees of freedom:
ν ~ Gamma(2, 0.1)  # E[ν]=20 (near Normal), allows heavy tails
```

**When to trigger**: If standardized residuals show |z| > 3 for multiple points

### Escape Route 3: Transformation Models

**When to use**: If nonlinearity is removable by transformation

```python
# Box-Cox transformation
Y_transformed = (Y^λ - 1) / λ  if λ ≠ 0
              = log(Y)        if λ = 0

# Then fit simple linear model on transformed scale
Y_transformed ~ Normal(β₀ + β₁*x, σ)

# Prior on λ:
λ ~ Normal(1, 0.5)  # Weakly centered at identity
```

**When to trigger**: If log-log or sqrt transformations linearize relationship in EDA

---

## Computational Implementation Plan

### Software Stack
- **Primary**: Stan (via `cmdstanr` or `pystan`)
- **Alternative**: PyMC (if Stan has convergence issues)
- **Preprocessing**: R (`splines2`) or Python (`scipy.interpolate`) for basis functions

### Workflow

```python
# Pseudocode for Model 1 (Change-Point)

import cmdstanpy
import numpy as np

# 1. Prepare data
stan_data = {
    'N': 27,
    'x': x_obs,
    'Y': Y_obs
}

# 2. Compile model
model = cmdstanpy.CmdStanModel(stan_file='changepoint.stan')

# 3. Sample
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.9,
    max_treedepth=12,
    seed=42
)

# 4. Diagnostics
print(fit.diagnose())
print(fit.summary())

# 5. Posterior predictive checks
Y_rep = fit.stan_variable('Y_rep')
# Compare Y_rep distribution to Y_obs

# 6. LOO-CV
import arviz as az
loo = az.loo(fit, pointwise=True)
print(loo)

# 7. Visualize
# Plot posterior mu(x) with credible bands
# Plot posterior distribution of tau
```

### Tuning Strategy

**If divergences occur**:
1. Increase `adapt_delta`: 0.9 → 0.95 → 0.99
2. Increase `max_treedepth`: 10 → 12 → 15
3. Use non-centered parameterization
4. Add tighter priors

**If slow mixing**:
1. Check trace plots for multimodality
2. Increase warmup iterations: 2000 → 5000
3. Use better initialization (from OLS fits)

---

## Expected Outcomes Summary

| Model | R² | LOO-ELPD | Convergence | Interpretation | Computational |
|-------|-----|----------|-------------|----------------|---------------|
| **Model 1: Change-Point** | 0.88-0.92 | High | Moderate (may need tuning) | Excellent (τ, β₁, β₂) | 30-60 sec |
| **Model 2: B-Spline** | 0.85-0.90 | Moderate-High | Excellent | Poor (coefficients opaque) | <30 sec |
| **Model 3: Mixture** | 0.87-0.91 | High | Moderate (nonlinear) | Good (experts + gating) | 60-120 sec |

**Personal prediction**: Model 1 will likely win on interpretability and LOO-ELPD, but Model 2 will provide the "safest" fit with fastest convergence. Model 3 is my "dark horse" - if the transition is truly gradual, it could dominate.

---

## Reflection: Why These Models Might Fail

### Model 1 (Change-Point) - Failure Modes

**Why it might be wrong**:
- The "breakpoint" at x=9-10 might be a sampling artifact (no observations at x=8-9)
- Biology/physics rarely has truly sharp discontinuities
- Forcing a breakpoint when transition is smooth leads to poor uncertainty quantification

**Escape route**: If SD(τ) > 5, immediately switch to Model 2 (smooth spline)

### Model 2 (B-Spline) - Failure Modes

**Why it might be wrong**:
- May oversmooth the transition (obscure real regime shift)
- Sensitive to knot placement (results change with K)
- Poor extrapolation (splines go wild outside data range)
- Opaque interpretation (can't easily explain to domain scientist)

**Escape route**: If model shows wild oscillations or poor LOO, switch to parametric exponential model

### Model 3 (Mixture) - Failure Modes

**Why it might be wrong**:
- Overparameterized for N=27 (risk of overfitting)
- Identifiability issues (β₁ and α can trade off)
- Assumes linear+constant mixture (true components might be different)
- Computational difficulties (label switching, multimodality)

**Escape route**: If convergence fails or LOO worse than components, abandon mixture

---

## Success Criteria

**Minimum viable model**:
- R² > 0.85 (match EDA benchmark)
- R-hat < 1.01, ESS > 400 for all parameters
- 90-95% of data within posterior predictive 95% intervals
- No Pareto-k > 0.7 in LOO

**Excellent model**:
- R² > 0.90
- Interpretable parameters with narrow posteriors (CV < 0.2)
- Posterior predictive checks show no systematic residual patterns
- Outperforms alternatives in LOO by >2*SE

**Model selection**:
- If multiple models meet "excellent" criteria: Prefer simpler, more interpretable
- If all models only "viable": Report model uncertainty, consider ensemble

---

## Timeline & Stopping Rules

### Day 1: Initial Fits
- Fit all three models with default priors
- Check convergence and diagnostics
- **STOP if**: All models fail to converge → Rethink model classes

### Day 2: Validation
- Posterior predictive checks
- LOO cross-validation
- **STOP if**: All models have LOO-R² < 0.75 → Return to EDA

### Day 3: Sensitivity & Refinement
- Prior sensitivity analysis
- Structural sensitivity (knot placement, etc.)
- **STOP if**: Results highly unstable → Switch to GP or ensemble

### Day 4: Final Selection
- Compare models on all criteria
- Make recommendation with honest uncertainty assessment
- **STOP if**: Evidence is ambiguous → Report multiple plausible models

---

## Final Note: Commitment to Truth

I am designing these models to be **falsifiable**, not to guarantee success. The goal is to discover which model class (if any) genuinely explains the data, not to complete a predetermined plan.

**I will consider this exercise successful if**:
1. I discover that one model class is clearly superior → Recommend it confidently
2. I discover that multiple models are equivalent → Report honest uncertainty
3. I discover that all proposed models fail → Pivot to alternative approaches (GP, transformations)

**I will consider this exercise a failure if**:
- I report a "winning" model while ignoring red flags
- I complete all models without critical evaluation
- I force a model to fit when evidence suggests it shouldn't

**Truth over task completion.**

---

## Files to Generate

1. `/workspace/experiments/designer_2/models/model1_changepoint.stan`
2. `/workspace/experiments/designer_2/models/model2_spline.stan`
3. `/workspace/experiments/designer_2/models/model3_mixture.stan`
4. `/workspace/experiments/designer_2/implementation_code.py` (or `.R`)
5. `/workspace/experiments/designer_2/validation_plan.md`

---

**END OF PROPOSAL**

**Absolute paths for reference**:
- This document: `/workspace/experiments/designer_2/proposed_models.md`
- EDA report: `/workspace/eda/eda_report.md`
- Models directory: `/workspace/experiments/designer_2/models/` (to be created)
