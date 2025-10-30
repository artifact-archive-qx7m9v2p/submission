# Flexible & Nonparametric Bayesian Models for Y ~ x Relationship

**Designer**: Flexible/Nonparametric Modeling Specialist
**Date**: 2025-10-28
**Focus**: Splines, Gaussian Processes, and Smoothing Methods
**Dataset**: n=27, Y vs x with strong nonlinear asymptotic relationship

---

## Executive Summary

This document proposes **three competing Bayesian model classes** for the Y~x relationship, focusing on flexible approaches that avoid strong parametric assumptions. Each model is designed with **explicit falsification criteria** and **escape routes** for when (not if) assumptions fail.

**Critical Mindset**: The EDA suggests a "simple" logarithmic relationship fits well (R²=0.897), which makes me deeply suspicious. Simple patterns in small datasets often hide complexity that only emerges with more data or better models. My goal is to design models that can **discover structure the EDA missed** while avoiding overfitting.

**Red Flags I'm Watching For**:
- Prior-data conflict (model fights the evidence)
- Excessive wiggliness (overfitting with n=27)
- Poor out-of-sample prediction despite good in-sample fit
- Computational pathologies (divergences, poor mixing)
- Inconsistent results when excluding x=31.5 outlier

---

## Competing Hypotheses

Before proposing models, I frame three fundamentally different hypotheses about the data-generating process:

### Hypothesis 1: Smooth Monotonic Saturation
**Belief**: Y approaches an asymptote smoothly with no regime changes. The apparent "two-regime" structure is a sampling artifact.

**Model Class**: Gaussian Process with Matérn kernel (smooth interpolation)

**I will abandon this if**: GP shows sharp discontinuities in mean function or derivative, suggesting genuine regime change rather than smooth saturation.

---

### Hypothesis 2: True Regime Change
**Belief**: The data reflect a genuine physical/biological phase transition at x≈7. The process literally changes mechanism.

**Model Class**: Penalized B-splines with moderate knots (can capture sharp transitions while regularizing)

**I will abandon this if**: Posterior derivative estimates show continuous change rather than discontinuity, or if the "changepoint" location has huge posterior uncertainty (e.g., 95% CI spanning [3, 15]).

---

### Hypothesis 3: Locally Complex, Globally Simple
**Belief**: There's subtle local structure (bumps, plateaus, inflections) that logarithmic models miss, but it's neither purely smooth nor a simple two-regime system.

**Model Class**: Adaptive P-splines with automatic knot selection

**I will abandon this if**: The model degenerates to a near-linear or simple logarithmic fit, meaning the flexibility wasn't needed.

---

## Model 1: Gaussian Process with Matérn 3/2 Kernel (RECOMMENDED START)

### Mathematical Specification

**Likelihood**:
```
Y_i ~ Normal(f(x_i), σ)
```

**Mean Function**:
```
f(x) ~ GP(m(x), k(x, x'))
```

**Mean Function Prior**:
```
m(x) = β₀ + β₁ * log(x)   # Informative mean based on EDA
β₀ ~ Normal(2.3, 0.3)
β₁ ~ Normal(0.3, 0.15)
```

**Kernel (Covariance Function)**:
```
k(x, x') = α² * (1 + √3 * d/ℓ) * exp(-√3 * d/ℓ)
where d = |x - x'| / scale

Parameters:
α (marginal standard deviation) ~ Normal⁺(0, 0.15)   # How much deviation from mean trend
ℓ (length scale) ~ InvGamma(5, 5)                    # Correlation decay rate
scale = std(x) ≈ 7.87                                # Standardization
```

**Observation Noise**:
```
σ ~ Exponential(1/0.1)  # Based on EDA residuals
```

### Why Matérn 3/2 Instead of SE (Squared Exponential)?

The Matérn 3/2 kernel produces functions that are **once differentiable but not infinitely smooth**. This is critical because:

1. **Real physical processes are rarely infinitely smooth** (SE assumption)
2. **Matérn 3/2 is more robust** to violations of smoothness assumptions
3. **Better posterior geometry** (fewer computational issues)
4. **The EDA shows a potential kink at x≈7** - Matérn 3/2 can accommodate this

**I will abandon Matérn 3/2 if**: Posterior suggests infinitely smooth function (unlikely with n=27) or if I need Matérn 1/2 (discontinuous derivative) for sharp regime change.

### Prior Rationale

**Informative mean function**: I'm using the logarithmic trend as a prior mean because:
- Ignoring EDA findings is foolish when n=27 (need all available information)
- GP will deviate from this if data supports it
- Prevents overfitting to noise in low-data regions

**Length scale prior InvGamma(5, 5)**:
- Mode at 4, mean at 5/4 = 1.25 (on standardized scale)
- Corresponds to ~10 units on original scale
- Allows both smooth trends and local variation
- **Falsification test**: If posterior concentrates at extreme values (ℓ < 0.5 or ℓ > 20), model is struggling

**Marginal SD prior Normal⁺(0, 0.15)**:
- Allows up to ~0.3 units deviation from mean trend (95% CI)
- Matches residual variation from EDA logarithmic model
- If posterior pushes against this, need stronger prior or different model

### Expected Strengths
1. **Minimal assumptions** about functional form
2. **Automatic uncertainty quantification** including prediction intervals
3. **Handles non-uniform x spacing** naturally (sparse at high x)
4. **Derivative estimates** for free (rate of change analysis)
5. **Can discover structure EDA missed** (local bumps, subtle transitions)

### Expected Weaknesses
1. **Computational cost**: O(n³) for n=27 (manageable but not trivial)
2. **Extrapolation is poor**: GP uncertainty explodes outside [1, 31.5]
3. **Hyperparameter sensitivity**: Results depend heavily on kernel choice and priors
4. **Small sample overfitting risk**: GP might wiggle through noise
5. **Interpretability**: "Black box" - hard to explain to domain scientists

### Falsification Criteria: I Will Abandon This Model If...

**Critical failures** (immediate abandonment):
1. **More than 5% divergent transitions** after tuning → Prior-geometry mismatch
2. **Posterior mean function shows wild oscillations** (>10 local extrema) → Overfitting
3. **LOO-CV worse than simple logarithmic baseline** → Flexibility not helping
4. **Posterior length scale ℓ < 0.5** → Suggesting extreme local variation incompatible with smooth GP

**Warning signs** (reconsider but don't abandon immediately):
1. **Effective sample size < 100** for key parameters → Poor mixing
2. **Posterior predictive p-values extreme** (< 0.01 or > 0.99) → Model-data mismatch
3. **Wide posterior uncertainty on hyperparameters** → Data don't constrain model
4. **Inconsistent results when removing x=31.5** → Overly influenced by outlier

### Computational Strategy

**Implementation**: PyMC with `gp.Marginal`
```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as gp_model:
    # Mean function
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.3)
    beta_1 = pm.Normal('beta_1', mu=0.3, sigma=0.15)
    mean_func = beta_0 + beta_1 * pt.log(x)

    # GP hyperparameters
    alpha = pm.HalfNormal('alpha', sigma=0.15)
    ell_raw = pm.InverseGamma('ell_raw', alpha=5, beta=5)
    ell = ell_raw * x.std()  # Rescale to data scale

    # Kernel
    cov = alpha**2 * pm.gp.cov.Matern32(1, ls=ell)

    # GP
    gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov)

    # Noise
    sigma = pm.Exponential('sigma', lam=1/0.1)

    # Likelihood
    y_obs = gp.marginal_likelihood('y_obs', X=x[:, None], y=y, sigma=sigma)

    # Sampling
    trace = pm.sample(2000, tune=2000, cores=4,
                      target_accept=0.95,  # High for GP
                      init='adapt_diag')
```

**Expected runtime**: ~5-10 minutes for 4000 iterations

**Stress test**: Refit with x=31.5 removed to check sensitivity

---

## Model 2: Penalized B-Splines (P-Splines) with Fixed Knots

### Mathematical Specification

**Likelihood**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = Σⱼ βⱼ * Bⱼ(x_i)
```

**Basis Functions**: Cubic B-splines with K=6 interior knots
```
Knot locations: quantiles at [0.1, 0.25, 0.4, 0.55, 0.7, 0.85] of x distribution
Total basis functions: J = K + degree + 1 = 6 + 3 + 1 = 10
```

**Coefficient Prior (Random Walk Penalty)**:
```
β₁ ~ Normal(2.3, 0.5)  # Intercept-like term, informed by data mean

For j = 2, ..., J:
  Δβⱼ = βⱼ - βⱼ₋₁
  Δβⱼ ~ Normal(0, τ)  # Second-order differences (smoothness penalty)

Smoothness parameter:
τ ~ Exponential(10)  # Shrinks toward smooth functions
```

**Alternative Penalty (L2 on coefficients)**:
```
βⱼ ~ Normal(0, τ) for j=2,...,J
τ ~ Exponential(5)
```

**Observation Noise**:
```
σ ~ Exponential(1/0.1)
```

### Why B-Splines Instead of Natural Cubic Splines?

1. **Local support**: Changing one coefficient affects only nearby region (more stable)
2. **Numerical stability**: Better conditioned design matrix
3. **Flexibility**: Can add/remove knots more easily
4. **Computational efficiency**: Sparse basis representation

**I will use natural cubic splines if**: Extrapolation beyond [1, 31.5] is critical (not the case here)

### Prior Rationale

**6 interior knots** (10 basis functions):
- Rule of thumb: K ≈ n/5 to n/3 for n=27 gives 5-9 knots
- 6 knots allows regime change at x≈7 plus local variation
- **Too few knots (K<4)**: Can't capture two-regime structure
- **Too many knots (K>10)**: Overfitting risk with n=27
- **Falsification**: If posterior shows strong activation of >8 basis functions with conflicting signs, overfitting likely

**Knot placement at quantiles**:
- Ensures adequate coverage where data are dense
- Avoids sparse regions (e.g., x>20) dominating knot budget
- Alternative: uniform spacing (will test both)

**Random walk penalty (second-order differences)**:
- Shrinks toward smooth, low-curvature functions
- Allows sharp local changes if data demand it
- Stronger than independent Normal priors
- **Falsification**: If posterior τ → 0 (infinite penalty), model degenerates to linear

**τ ~ Exponential(10)**:
- Mean = 0.1, mode near 0
- Strong shrinkage toward smoothness
- Data can override if needed
- **Stress test**: Try Exponential(1) (less shrinkage) to check sensitivity

### Expected Strengths
1. **Interpretable flexibility**: Can visualize basis function contributions
2. **Computational efficiency**: Faster than GP (linear algebra is simpler)
3. **Can capture sharp transitions**: Unlike GP with smooth kernels
4. **Local control**: Knot placement can be informed by EDA (x≈7 changepoint)
5. **Derivative estimates**: Splines are differentiable, can compute slopes

### Expected Weaknesses
1. **Knot placement sensitivity**: Results may depend on where knots are placed
2. **Boundary behavior**: May be unstable at x=1 and x=31.5 extremes
3. **Less principled than GP**: Ad hoc choices (# knots, penalty order)
4. **Overfitting risk**: With 10 basis functions and n=27, can fit noise
5. **Prior-data conflict**: Penalty may fight against real structure

### Falsification Criteria: I Will Abandon This Model If...

**Critical failures**:
1. **Posterior predictive checks fail systematically**: E.g., can't capture variance structure
2. **Coefficients show wild oscillations**: Sign changes every basis function → overfitting
3. **LOO-CV much worse than GP**: Suggests knot placement is poor
4. **Extreme extrapolation behavior**: Function goes to ±∞ just outside [1, 31.5]
5. **Posterior τ concentrates near 0**: Model degenerating to over-smoothed linear

**Warning signs**:
1. **Basis function collinearity**: High posterior correlations (ρ > 0.9) between adjacent β
2. **Knot placement matters too much**: Results change drastically with quantile vs uniform knots
3. **Poor fit in "changepoint" region**: If data truly have regime change, smooth penalty may blur it
4. **Effective sample size < 200**: Suggests posterior geometry problems

### Computational Strategy

**Implementation**: PyMC with patsy for basis generation
```python
import pymc as pm
import numpy as np
from patsy import dmatrix

# Generate B-spline basis
knots_interior = np.quantile(x, [0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
B = dmatrix("bs(x, knots=knots_interior, degree=3, include_intercept=True)",
            {"x": x}, return_type='dataframe')

with pm.Model() as pspline_model:
    # First coefficient (intercept-like)
    beta = [pm.Normal('beta_0', mu=2.3, sigma=0.5)]

    # Smoothness parameter
    tau = pm.Exponential('tau', lam=10)

    # Remaining coefficients with random walk prior
    for j in range(1, 10):
        beta.append(pm.Normal(f'beta_{j}', mu=beta[j-1], sigma=tau))

    beta_vec = pt.stack(beta)

    # Mean function
    mu = pm.Deterministic('mu', B.values @ beta_vec)

    # Noise
    sigma = pm.Exponential('sigma', lam=1/0.1)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sampling
    trace = pm.sample(2000, tune=2000, cores=4, target_accept=0.9)
```

**Expected runtime**: ~2-3 minutes

**Sensitivity analyses**:
1. Try uniform knot spacing
2. Try K=4, 8, 12 knots
3. Try tau ~ Exponential(1) vs Exponential(10)
4. Try L2 penalty instead of random walk

---

## Model 3: Adaptive Smoothing with Heterogeneous Length Scales

### Mathematical Specification

This model uses a **piecewise Gaussian Process** approach, allowing different smoothness in different regions.

**Likelihood**:
```
Y_i ~ Normal(f(x_i), σ)
```

**Piecewise GP Mean**:
```
For x ≤ τ:  f(x) ~ GP(m₁(x), k₁(x, x'))
For x > τ:  f(x) ~ GP(m₂(x), k₂(x, x'))
```

**Changepoint** (unknown):
```
τ ~ Uniform(4, 12)  # Informed by EDA suggesting x≈7
```

**Mean Functions**:
```
m₁(x) = β₁₀ + β₁₁ * x
m₂(x) = β₂₀ + β₂₁ * x

β₁₀ ~ Normal(1.9, 0.3)  # Based on EDA regime 1 intercept ≈ 1.687
β₁₁ ~ Normal(0.11, 0.05)  # Based on EDA regime 1 slope ≈ 0.113
β₂₀ ~ Normal(2.2, 0.3)  # Based on EDA regime 2 intercept ≈ 2.231
β₂₁ ~ Normal(0.02, 0.02)  # Based on EDA regime 2 slope ≈ 0.017
```

**Kernels** (Matérn 3/2 with regime-specific length scales):
```
k₁(x, x') = α₁² * Matérn₃/₂(d/ℓ₁)
k₂(x, x') = α₂² * Matérn₃/₂(d/ℓ₂)

α₁ ~ Normal⁺(0, 0.10)  # Regime 1 marginal SD
α₂ ~ Normal⁺(0, 0.10)  # Regime 2 marginal SD
ℓ₁ ~ InverseGamma(3, 3)  # Regime 1 length scale (expect shorter)
ℓ₂ ~ InverseGamma(5, 10)  # Regime 2 length scale (expect longer/smoother)
```

**Continuity Constraint**:
To avoid discontinuity at τ, enforce:
```
m₁(τ) = m₂(τ)
⟹ β₂₀ = β₁₀ + τ*(β₁₁ - β₂₁)
```

**Observation Noise**:
```
σ ~ Exponential(1/0.1)
```

### Why This Complex Design?

This model directly tests Hypothesis 2 (true regime change) by asking:
- **Is there a genuine changepoint?** (τ posterior should be concentrated if yes)
- **Do the regimes have different smoothness?** (ℓ₁ vs ℓ₂)
- **Is the transition sharp or gradual?** (GP smooths across τ vs hard break)

**I will abandon this if**:
- Posterior for τ is nearly uniform (no evidence for changepoint)
- ℓ₁ ≈ ℓ₂ (no difference in smoothness across regimes)
- Simpler piecewise linear model (from Designer 1) fits just as well

### Prior Rationale

**τ ~ Uniform(4, 12)**:
- EDA suggests x≈7, but I'm allowing wide range
- Below x=4: only 6 observations (too few for regime 1)
- Above x=12: regime 2 would have only 9 obs (too few)
- **Falsification**: If posterior concentrates at boundaries (4 or 12), wrong model

**Regime-specific mean functions**:
- Using EDA piecewise linear estimates as informative priors
- GP deviations will capture local nonlinearity within regimes
- **Alternative**: Could use logarithmic means, but linear is more conservative

**Different length scale priors for each regime**:
- Hypothesis: Regime 1 (growth) is more variable → shorter ℓ₁
- Regime 2 (plateau) is smoother → longer ℓ₂
- **Test**: If ℓ₁ > ℓ₂ in posterior, hypothesis is wrong

**Continuity constraint**:
- Ensures smooth transition at τ
- Prevents implausible discontinuous jumps
- **Alternative**: Allow discontinuity (harder computationally)

### Expected Strengths
1. **Directly addresses EDA finding**: Two-regime structure with uncertainty on τ
2. **Flexible within regimes**: GP captures local deviations from linear trends
3. **Interpretable parameters**: Changepoint τ, regime-specific slopes
4. **Tests specific hypothesis**: Can falsify "regime change" claim
5. **Propagates uncertainty**: τ is uncertain, affects everything downstream

### Expected Weaknesses
1. **Computational complexity**: Two GPs + changepoint = slow sampling
2. **Small sample in each regime**: If τ≈7, only 9 vs 18 observations
3. **Identifiability issues**: τ vs slopes vs GP deviations may trade off
4. **Continuity constraint complicates prior**: β₂₀ is no longer independent
5. **May be overparameterized**: Simpler piecewise linear might suffice

### Falsification Criteria: I Will Abandon This Model If...

**Critical failures**:
1. **Divergent transitions >10%**: Model geometry is pathological
2. **τ posterior is uniform**: No evidence for changepoint
3. **LOO-CV worse than Model 1 (simple GP)**: Complexity not justified
4. **Posterior predictions are discontinuous at τ**: Continuity constraint failed
5. **ℓ₁ and ℓ₂ posteriors nearly identical**: No regime-specific smoothness

**Warning signs**:
1. **ESS(τ) < 50**: Changepoint poorly identified
2. **High posterior correlation** between τ and β parameters (ρ > 0.95)
3. **Prior-posterior overlap >90%** for τ: Data don't inform changepoint
4. **Posterior predictive checks fail in regime transition region** (x ∈ [5, 10])

### Computational Strategy

**Implementation**: PyMC with conditional GP construction
```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as adaptive_gp_model:
    # Changepoint
    tau = pm.Uniform('tau', lower=4, upper=12)

    # Regime indicators
    regime1_idx = x <= tau
    regime2_idx = x > tau

    # Regime 1 mean parameters
    beta_10 = pm.Normal('beta_10', mu=1.9, sigma=0.3)
    beta_11 = pm.Normal('beta_11', mu=0.11, sigma=0.05)

    # Regime 2 slope
    beta_21 = pm.Normal('beta_21', mu=0.02, sigma=0.02)

    # Regime 2 intercept (continuity constraint)
    beta_20 = pm.Deterministic('beta_20', beta_10 + tau*(beta_11 - beta_21))

    # Mean functions
    mean1 = beta_10 + beta_11 * x[regime1_idx]
    mean2 = beta_20 + beta_21 * x[regime2_idx]

    # GP hyperparameters
    alpha1 = pm.HalfNormal('alpha1', sigma=0.10)
    alpha2 = pm.HalfNormal('alpha2', sigma=0.10)
    ell1 = pm.InverseGamma('ell1', alpha=3, beta=3)
    ell2 = pm.InverseGamma('ell2', alpha=5, beta=10)

    # Kernels
    cov1 = alpha1**2 * pm.gp.cov.Matern32(1, ls=ell1)
    cov2 = alpha2**2 * pm.gp.cov.Matern32(1, ls=ell2)

    # GPs (conditional on tau)
    gp1 = pm.gp.Marginal(mean_func=mean1, cov_func=cov1)
    gp2 = pm.gp.Marginal(mean_func=mean2, cov_func=cov2)

    # Noise
    sigma = pm.Exponential('sigma', lam=1/0.1)

    # Likelihood (piecewise)
    y1_obs = gp1.marginal_likelihood('y1_obs', X=x[regime1_idx, None],
                                     y=y[regime1_idx], sigma=sigma)
    y2_obs = gp2.marginal_likelihood('y2_obs', X=x[regime2_idx, None],
                                     y=y[regime2_idx], sigma=sigma)

    # Sampling (will be challenging!)
    trace = pm.sample(3000, tune=3000, cores=4,
                      target_accept=0.99,  # Very high for complex model
                      init='adapt_diag',
                      max_treedepth=12)
```

**Expected runtime**: ~15-30 minutes (complex model)

**Warning**: This model may have sampling difficulties. If divergences persist:
- Try simplifying to fixed τ=7 (based on EDA)
- Remove GP components (reduce to piecewise linear)
- Use variational inference as fallback

---

## Model Comparison Strategy

### Within-Sample Fit
1. **Posterior predictive checks**:
   - Plot posterior mean ± 95% CI vs data
   - Check if observations fall within credible intervals appropriately
   - Test statistic: max|residual| (should not always be same point)

2. **Residual analysis**:
   - Plot residuals vs x (should be randomly scattered)
   - Q-Q plot of residuals (check normality assumption)
   - Autocorrelation of residuals (should be near zero)

### Out-of-Sample Prediction
1. **LOO-CV (Leave-One-Out Cross-Validation)**:
   - Primary comparison metric
   - Penalizes model complexity automatically
   - Look for k̂ > 0.7 (influential observations)
   - **Decision rule**: ΔLOO > 2*SE → significant difference

2. **WAIC (Widely Applicable Information Criterion)**:
   - Secondary metric
   - Should agree with LOO-CV
   - If LOO and WAIC disagree strongly → model issues

### Computational Diagnostics
1. **Convergence checks**:
   - R̂ < 1.01 for all parameters (strict threshold)
   - ESS > 400 for bulk, >400 for tail (n=27 is small, need good ESS)
   - No divergent transitions (target: 0%)
   - Energy diagnostic (E-BFMI > 0.3)

2. **Prior sensitivity**:
   - Refit with 2× wider priors
   - Posterior should not change drastically
   - If results flip, priors are dominating (bad with n=27)

### Scientific Validity
1. **Derivative analysis**:
   - Estimate dY/dx at multiple x values
   - Should be positive (monotonic increase) everywhere
   - Sharp drop at x≈7 would support regime change

2. **Extrapolation check**:
   - Predict at x=0.5 and x=40
   - Should give reasonable values (Y ∈ [1.5, 3.0])
   - GP and splines may fail here (flag in report)

3. **Outlier sensitivity**:
   - Refit all models without x=31.5
   - Compare posteriors (should be similar)
   - If conclusions change → models not robust

---

## Decision Points and Pivot Criteria

### Checkpoint 1: After Initial Fits (Week 1)
**Question**: Do any models have critical failures?

**Actions**:
- If GP has >5% divergences → Try different kernel or priors
- If P-splines overfit (oscillating) → Reduce knots or increase penalty
- If adaptive GP won't sample → Simplify to fixed changepoint

**Pivot if**: All three models fail diagnostics → Abandon flexible approaches, use parametric models from Designer 1

---

### Checkpoint 2: After LOO-CV Comparison (Week 1)
**Question**: Do flexible models beat parametric baselines?

**Baselines** (from Designer 1, expected):
- Logarithmic: LOO ≈ -10 (rough estimate)
- Piecewise linear: LOO ≈ -8

**Decision rules**:
- If all flexible models have LOO > -5 (worse than baselines) → **ABANDON FLEXIBLE APPROACH**
- If GP beats baselines by ΔLOO < 2 → Flexibility not justified, use logarithmic
- If Model 3 beats Model 1 by ΔLOO > 4 → Regime change hypothesis supported

**Pivot if**: Simple logarithmic model wins decisively → Accept that EDA was right, write "why I was wrong" section

---

### Checkpoint 3: After Posterior Predictive Checks (Week 2)
**Question**: Do models capture essential data features?

**Essential features** (from EDA):
- Monotonic increase
- Asymptotic behavior (plateau at high x)
- Replicates at same x have similar Y (low variance)
- No extreme outliers (except possibly x=31.5)

**Red flags**:
- Posterior predictive simulations show non-monotonic patterns
- Predictive variance much larger than observed variance
- Systematic bias in specific x regions

**Pivot if**: All models fail PPC → Reconsider likelihood (try Student-t) or mean structure

---

### Checkpoint 4: Scientific Interpretation (Week 2)
**Question**: Can we tell a coherent story?

**Good outcomes**:
- Clear winner emerges (ΔLOO > 4) with interpretable parameters
- Regime change hypothesis confirmed or rejected decisively
- Predictions make sense for domain (even without domain context here)

**Bad outcomes**:
- Models disagree on fundamental features (monotonicity, asymptote)
- Parameter estimates implausible (e.g., length scale >> range of x)
- Cannot distinguish between models (all LOO within 2 SE)

**Pivot if**: Models are indistinguishable → Use Bayesian model averaging

---

## Alternative Approaches If Initial Models Fail

### Escape Route 1: Student-t Likelihood
If outlier x=31.5 causes problems:
```
Y_i ~ StudentT(ν, μ_i, σ)
ν ~ Gamma(2, 0.1)  # Degrees of freedom (heavier tails if ν < 10)
```

### Escape Route 2: Heteroscedastic Noise
If variance increases with x (despite EDA saying no):
```
log(σ_i) = γ₀ + γ₁ * x_i
γ₀ ~ Normal(log(0.1), 0.5)
γ₁ ~ Normal(0, 0.1)
```

### Escape Route 3: Discrete Mixture Model
If data suggest multiple distinct populations (not just regimes):
```
Y_i ~ Σₖ πₖ * Normal(μₖ(x_i), σₖ)
K = 2 or 3 components
```

### Escape Route 4: Transform Y
If relationship is nonlinear in Y (not tested in EDA):
```
log(Y_i) ~ GP(...)  or  sqrt(Y_i) ~ GP(...)
```

### Escape Route 5: Abandon Flexibility
If all flexible models overfit:
- Fall back to Designer 1's parametric models
- Document why flexibility hurt rather than helped
- Write honest post-mortem on what went wrong

---

## Prioritization and Workflow

### Priority 1: Model 1 (GP with Matérn 3/2)
**Why first**:
- Most principled (fewest ad hoc choices)
- If it works well, provides strong baseline
- Fast to fit (relative to Model 3)

**Timeline**: Fit and diagnose in Day 1

---

### Priority 2: Model 2 (P-Splines)
**Why second**:
- Alternative flexible approach
- Faster than Model 1
- Tests sensitivity to GP assumptions

**Timeline**: Fit in Day 1, compare to Model 1

---

### Priority 3: Model 3 (Adaptive GP)
**Why last**:
- Most complex (may not converge)
- Only fit if Models 1-2 suggest regime change is real
- Skip entirely if simple GP (Model 1) wins decisively

**Timeline**: Fit in Day 2, only if warranted

---

## Expected Outcomes and Predictions

### My Predictions (Falsifiable!)

**Prediction 1**: Model 1 (GP) will beat parametric baselines by ΔLOO ≈ 2-3
- **If wrong**: Means logarithmic form is truly optimal, flexibility is overfitting

**Prediction 2**: Model 2 (P-splines) will show ≥8 active basis functions
- **If wrong**: Fewer active functions means we could use simpler model

**Prediction 3**: Model 3 (adaptive GP) will have τ posterior concentrated in [6, 8]
- **If wrong**: Either no changepoint exists, or it's elsewhere

**Prediction 4**: All models will struggle with x=31.5 (outlier)
- **If wrong**: Means it's not really an outlier, or models are too flexible

**Prediction 5**: GP length scale ℓ will be ≈5-10 (on original scale)
- **If wrong**: Much smaller → overfitting; much larger → underfitting

---

## Success Criteria

### Minimal Success (What I Must Achieve)
1. All models converge (R̂ < 1.01, ESS > 400, divergences < 1%)
2. LOO-CV computed successfully (no unstable k̂ values)
3. At least one model beats logarithmic baseline
4. Clear recommendation with evidence

### Target Success (What I Aim For)
1. All three models fit and compare cleanly
2. One model is clear winner (ΔLOO > 4)
3. Posterior predictive checks all pass
4. Outlier sensitivity analysis complete
5. Interpretable scientific story emerges

### Exceptional Success (What Would Delight Me)
1. Discover structure EDA missed (validated by PPC)
2. Definitively confirm or reject regime change hypothesis
3. Provide actionable predictions with quantified uncertainty
4. Models agree on fundamental features despite different approaches
5. Honest documentation of what I got wrong

---

## What Would Make Me Reconsider Everything

### Scenario 1: All Flexible Models Overfit
**Evidence**: LOO much worse than simple models, posterior predictive variance too large

**Action**: Write report titled "Why Flexibility Failed" and recommend parametric approach

---

### Scenario 2: Data Suggest Discrete Jumps, Not Smooth Curves
**Evidence**: Posterior derivatives show discontinuities, GP struggles with sharp features

**Action**: Pivot to changepoint regression (Designer 1's territory) or mixture models

---

### Scenario 3: Likelihood is Wrong
**Evidence**: Posterior predictive checks fail, residuals non-normal, heteroscedasticity emerges

**Action**: Try Student-t, heteroscedastic, or transformed-Y models

---

### Scenario 4: Sample Size is Too Small
**Evidence**: All models have huge posterior uncertainty, LOO is unstable (high k̂), priors dominate

**Action**: Document that n=27 is insufficient for flexible modeling, recommend data collection

---

### Scenario 5: EDA Was Exactly Right
**Evidence**: Logarithmic model beats all flexible approaches decisively

**Action**: Write "In Defense of Simple Models" section, explain why complexity didn't help

---

## Computational Resource Planning

### Estimated Runtimes (4 cores, modern CPU)
- Model 1 (GP): 5-10 minutes
- Model 2 (P-splines): 2-3 minutes
- Model 3 (Adaptive GP): 15-30 minutes
- Sensitivity analyses: +50% per model
- **Total**: ~2-3 hours of compute time

### Memory Requirements
- All models: <2 GB RAM (n=27 is small)
- GP covariance matrices: 27×27 = 729 elements (trivial)

### Failure Modes
- **Divergences**: Increase target_accept to 0.95-0.99
- **Low ESS**: Increase iterations, check for multimodality
- **Timeout**: Reduce max_treedepth or use variational inference

---

## Final Thoughts: Embracing Uncertainty

I've designed these models to **fail informatively**. Each has clear falsification criteria and escape routes. My goal is not to confirm that flexible models are better (they may not be!), but to **test whether the EDA's simple story is complete**.

Key mantras:
- **Simple models are often right** - I must prove complexity is justified
- **Small samples limit what we can learn** - I cannot discover infinite complexity with n=27
- **Overfitting is worse than underfitting** - Conservative models are better than overconfident ones
- **Computational problems indicate model problems** - Divergences are warnings, not nuisances

If at the end I conclude "use the logarithmic model," that's success, not failure. It means I tested alternatives rigorously and found them wanting.

---

## Appendix: Why These Models and Not Others?

### Models I Considered But Rejected

**Squared Exponential GP**: Too smooth, assumes infinite differentiability (unrealistic)

**Matérn 1/2 GP**: Too rough (discontinuous derivative), would suggest random walk

**Natural cubic splines**: Poor extrapolation beyond [1, 31.5], no better than B-splines

**Thin plate splines**: Require 2D smoothing (only have 1D here), overkill

**LOESS/Local regression**: Not Bayesian, no uncertainty quantification

**Polynomial splines**: Less stable than B-splines, boundary issues

**Reproducing kernel Hilbert space**: Computationally expensive, not clearly better than GP

### Why I Chose These Three

**Model 1 (GP)**: Gold standard for flexible nonparametric regression with UQ

**Model 2 (P-splines)**: Practical alternative to GP, more interpretable basis

**Model 3 (Adaptive GP)**: Direct test of regime change hypothesis (scientific question)

Together they span the space of reasonable flexible approaches while remaining computationally feasible.

---

**End of Proposal**

Next steps: Implement models in PyMC, run diagnostics, compare results, write honest assessment.
