# Bayesian Parametric Model Design
## Designer 1: Parametric Models Specialist

**Date**: 2025-10-28
**Dataset**: /workspace/data/data.csv (n=27)
**EDA Report**: /workspace/eda/eda_report.md
**Focus**: Parametric transformations, piecewise, and asymptotic models

---

## Executive Summary

Based on EDA findings, I propose **THREE competing Bayesian model classes** with fundamentally different assumptions about the data-generating process:

1. **Logarithmic Model**: Smooth saturation via log-transformation (simplest)
2. **Piecewise Linear Model**: Sharp regime change at unknown threshold (most flexible)
3. **Three-Parameter Asymptotic Model**: Smooth approach to biochemical equilibrium (most mechanistic)

**Critical stance**: Each model makes strong assumptions that are likely WRONG in some way. The goal is to identify which model fails least badly and under what conditions we should abandon the entire parametric approach.

---

## Design Philosophy: Adversarial Modeling

### Falsification Mindset

I am designing these models to **FAIL INFORMATIVELY**. Success means discovering why a model is wrong, not confirming it works.

**Red flags that would make me abandon parametric models entirely**:
- Residual patterns suggesting missing covariates or latent structure
- Systematic predictive failures in held-out validation
- Prior-posterior conflict (data strongly contradicting priors)
- Parameters hitting boundaries (suggests wrong functional form)
- Better performance from non-parametric alternatives

### Decision Framework

| Evidence | Action |
|----------|--------|
| All three models show similar systematic residual patterns | Switch to non-parametric (GP, splines) |
| Changepoint posterior is bimodal or very diffuse | Consider mixture model or latent regime |
| Outlier (x=31.5) has high posterior probability of being anomalous | Investigate measurement error model |
| Student-t degrees of freedom → 2 | Data has heavy tails; Gaussian assumption fails |
| LOO-CV shows multiple influential points | Dataset may be too heterogeneous for single model |
| Posterior predictive p-values < 0.05 or > 0.95 | Severe model misspecification |

---

## Problem Formulation: Competing Hypotheses

### Hypothesis 1: Smooth Saturation (Logarithmic)
**Claim**: Y increases logarithmically with x, representing a smooth, continuous saturation process.

**Physical interpretation**: Each doubling of x produces the same additive gain in Y (Weber-Fechner law, diminishing marginal returns).

**I will abandon this if**:
- Residuals show clear clustering into two groups
- Posterior predictive checks show systematic deviation in the x=5-10 range
- Changepoint model has decisively better LOO-CV (ΔELPD > 10)

### Hypothesis 2: Sharp Regime Shift (Piecewise)
**Claim**: A discrete threshold exists where the system transitions from one operating regime to another.

**Physical interpretation**: Phase transition, resource exhaustion, or biological threshold (e.g., receptor saturation at specific concentration).

**I will abandon this if**:
- Changepoint location posterior is uniform (no identifiable threshold)
- Intercept discontinuity at changepoint is large (>0.2 units; suggests measurement artifact)
- Slope difference between regimes is not credibly different from zero
- Simpler logarithmic model explains data equally well (Occam's razor)

### Hypothesis 3: Mechanistic Saturation (Asymptotic)
**Claim**: Y approaches a biochemical/physical maximum asymptotically, following exponential decay from initial state.

**Physical interpretation**: Michaelis-Menten kinetics, enzyme saturation, equilibrium binding, or learning curves.

**I will abandon this if**:
- Estimated asymptote `a` is far below maximum observed Y (model can't reach data)
- Rate parameter `c` → 0 (no saturation) or → ∞ (instantaneous saturation)
- Initial gap `b` is larger than physically plausible range
- Residuals show clear two-regime structure that smooth asymptote can't capture

---

## Model Specifications

### Model 1: Logarithmic Transformation (BASELINE)

#### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(μ_i, σ)
μ_i = β₀ + β₁ * log(x_i)

Priors:
β₀ ~ Normal(2.3, 0.3)     # Intercept: centered at observed mean Y
β₁ ~ Normal(0.3, 0.15)    # Slope: positive, based on EDA estimate ± 50%
σ ~ Exponential(10)        # Scale: λ = 10 → mean = 0.1 (conservative)
```

#### Prior Justification

**β₀ (Intercept)**:
- EDA shows mean(Y) = 2.33, range [1.77, 2.72]
- Normal(2.3, 0.3) gives 95% prior credible interval [1.7, 2.9]
- Covers observed range while being mildly informative
- Rationale: With n=27, we need some regularization to prevent overfitting

**β₁ (Log-slope)**:
- EDA logarithmic fit: β₁ ≈ 0.29
- Normal(0.3, 0.15) gives 95% CI [0.0, 0.6]
- Allows for both weaker and stronger effects
- Positivity not enforced (truncation would cause computational issues), but prior mass heavily favors positive
- Rationale: Mildly informative but allows data to substantially move estimate

**σ (Residual SD)**:
- EDA logarithmic RMSE = 0.087, linear RMSE = 0.153
- Exponential(10) → mean = 0.1, 95% interval [0.01, 0.30]
- Weakly informative: allows for better or worse fit than EDA suggests
- Rationale: Half-Cauchy(0.1) is common, but Exponential is more stable for Stan

#### Theoretical Strengths

1. **Parsimony**: Only 3 parameters
2. **Interpretability**: β₁ is "change in Y per doubling of x" (multiply by log(2) ≈ 0.69)
3. **Proven EDA fit**: R² = 0.897, RMSE = 0.087
4. **Extrapolation behavior**: Reasonable at high x (unlike polynomials)
5. **Computational tractability**: Linear in log(x), fast sampling

#### Expected Weaknesses

1. **Cannot capture sharp transitions**: Smooth by construction
2. **Sensitive to low x values**: log(x) → -∞ as x → 0 (not issue here since min(x)=1)
3. **Homoscedastic assumption**: May miss variance changes
4. **Single outlier leverage**: x=31.5 may pull fit disproportionately
5. **Ignores regime structure**: If true changepoint exists, model will smooth over it

#### Falsification Criteria

**I will reject this model if**:

1. **Residual patterns**: Systematic clustering of residuals into low-x and high-x groups (suggests piecewise)
2. **Predictive failure**: Posterior predictive p-value for "variance of residuals in x<7" vs "x>7" is extreme (<0.05 or >0.95)
3. **Information criteria**: LOO-CV strongly favors piecewise (ΔELPD > 10, SE ratio > 3)
4. **Parameter pathology**: β₁ posterior includes zero with high probability (no log relationship)
5. **Outlier sensitivity**: Excluding x=31.5 drastically changes β₁ estimate (>30% shift)

#### Stress Test

**Designed to break the model**:
- Posterior predictive check for "maximum local slope change" between consecutive points
- If piecewise is true, observed slope change at x≈7 should be in the tail of the predictive distribution
- Null hypothesis: slope changes are gradual (logarithmic)
- Alternative: slope change is concentrated at one location (piecewise)

---

### Model 2: Piecewise Linear with Unknown Changepoint

#### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(μ_i, σ)

μ_i = {
  α₁ + β₁ * x_i,           if x_i ≤ τ
  α₂ + β₂ * x_i,           if x_i > τ
}

Priors:
τ ~ Uniform(3, 12)           # Changepoint: broad but within data range
α₁ ~ Normal(1.8, 0.4)        # Intercept regime 1: near min(Y)
α₂ ~ Normal(2.2, 0.4)        # Intercept regime 2: near plateau
β₁ ~ Normal(0.1, 0.05)       # Slope regime 1: steep (EDA: 0.113)
β₂ ~ Normal(0.02, 0.02)      # Slope regime 2: shallow (EDA: 0.017)
σ ~ Exponential(15)          # Residual SD: tighter than log model

Constraint (optional, for continuity):
α₂ = α₁ + (β₁ - β₂) * τ      # Forces continuity at changepoint
```

#### Prior Justification

**τ (Changepoint)**:
- EDA F-test optimal: τ = 7.0
- Uniform(3, 12) is intentionally broad to avoid over-constraining
- Range excludes extreme x values where changepoint is implausible
- Lower bound: 3 (need ≥3 points in regime 1 for stable estimation)
- Upper bound: 12 (leaves ≥15 points in regime 2)
- Rationale: Let data determine location; avoid confirmation bias

**α₁, α₂ (Intercepts)**:
- Regime-specific intercepts, not global
- Normal(1.8, 0.4): regime 1 starts near minimum Y
- Normal(2.2, 0.4): regime 2 near plateau level
- Weakly informative but physically grounded
- Rationale: Prevents intercepts from wandering to implausible values

**β₁, β₂ (Slopes)**:
- Normal(0.1, 0.05): regime 1 steep, based on EDA estimate
- Normal(0.02, 0.02): regime 2 shallow, near-flat
- Both allow for substantially different values
- Positive mean but not truncated (allows negative if data demands)
- Rationale: Informative enough to stabilize with n=27, but not dogmatic

**σ (Residual SD)**:
- Exponential(15) → mean ≈ 0.067
- Tighter than logarithmic model (EDA shows better fit)
- 95% interval [0.005, 0.20]
- Rationale: Piecewise should explain more variance than smooth log

**Continuity constraint** (optional):
- Enforcing continuity reduces parameters from 5 to 4
- Prevents discontinuous jumps at τ
- May be too restrictive if true process has step change
- Recommendation: Fit both versions, compare

#### Theoretical Strengths

1. **Captures regime shift**: Explicit threshold at τ
2. **EDA support**: F-test highly significant (F=22.4, p<0.0001)
3. **Interpretable parameters**: Two distinct operating modes
4. **Flexibility**: Can model both continuous and discontinuous transitions
5. **Physical realism**: Many systems exhibit threshold behavior

#### Expected Weaknesses

1. **Parameter correlation**: τ, α, β likely highly correlated → slow mixing
2. **Computational cost**: Discontinuity in likelihood requires careful gradient handling (Stan)
3. **Overfitting risk**: 4-5 parameters for n=27 may be too many
4. **Changepoint uncertainty**: Posterior may be wide or multimodal if threshold is weak
5. **Extrapolation danger**: Piecewise breaks down outside [min(x), max(x)]

#### Falsification Criteria

**I will reject this model if**:

1. **Changepoint posterior is flat**: Uniform over prior range → no identifiable threshold
2. **Slope difference not credible**: P(β₁ > β₂) < 0.9 → regimes not distinct
3. **Discontinuity is large**: If not enforcing continuity, discontinuity > 0.2 units suggests artifact
4. **Worse LOO than log model**: Complexity not justified by improved fit
5. **Changepoint at boundary**: τ posterior mass at 3 or 12 → misspecified prior range
6. **High posterior correlation**: ρ(τ, β₁) > 0.95 → unidentifiable

#### Stress Test

**Designed to break the model**:
- **Posterior predictive check**: Generate datasets from posterior, refit, compare τ estimates
- If true changepoint exists, recovered τ should be precise (SD < 1.5)
- If smooth saturation (log model correct), recovered τ will wander randomly
- **Sensitivity analysis**: Fit with different prior ranges for τ (e.g., [5,9] vs [3,12])
- If posteriors agree, evidence is strong; if they differ, changepoint is weakly identified

---

### Model 3: Three-Parameter Asymptotic (Mechanistic)

#### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(μ_i, σ)
μ_i = a - b * exp(-c * x_i)

Priors:
a ~ Normal(2.7, 0.2)      # Asymptote: near max(Y) = 2.72
b ~ Normal(0.9, 0.3)      # Gap: distance from asymptote at x=0
c ~ Exponential(2)        # Rate: λ=2 → mean=0.5, favors moderate saturation
σ ~ Exponential(12)       # Residual SD: between log and piecewise

Alternative parameterization (for identifiability):
a ~ Normal(2.7, 0.2)
μ₀ = a - b ~ Normal(1.8, 0.3)  # Y-value at x=0 (more interpretable)
c ~ Exponential(2)
```

#### Prior Justification

**a (Asymptote)**:
- Theoretical maximum Y as x → ∞
- max(Y) observed = 2.72 at x=29
- Normal(2.7, 0.2) gives 95% CI [2.3, 3.1]
- Centered at observed max but allows for higher true asymptote
- Rationale: Asymptote should be ≥ max observed Y; tight prior prevents wandering

**b (Initial gap)**:
- At x=0: μ(0) = a - b
- Observed min(Y) ≈ 1.8, so b ≈ 2.7 - 1.8 = 0.9
- Normal(0.9, 0.3) gives 95% CI [0.3, 1.5]
- Must be positive (enforced by prior mass, or use lognormal)
- Rationale: Gap should span roughly the range of Y

**c (Rate parameter)**:
- Controls how quickly Y approaches asymptote
- Exponential(2) → mean = 0.5, 95% interval [0.02, 1.5]
- Larger c → faster saturation
- Must be positive
- Rationale: Weakly informative; lets data determine saturation speed

**Alternative parameterization**:
- Instead of (a, b), use (a, μ₀) where μ₀ = a - b
- μ₀ is Y-intercept (more interpretable)
- Reduces posterior correlation between a and b
- Recommended for better sampling efficiency

**σ (Residual SD)**:
- Exponential(12) → mean ≈ 0.083
- Between log model (looser) and piecewise (tighter)
- Rationale: Asymptotic should fit well but not overfit

#### Theoretical Strengths

1. **Mechanistic interpretation**: Models approach to equilibrium/saturation
2. **Smooth transition**: No discontinuity issues
3. **Biological plausibility**: Common in biochemistry (Michaelis-Menten, binding curves)
4. **Bounded predictions**: Y cannot exceed `a`
5. **Good EDA fit**: R² = 0.889, RMSE = 0.090

#### Expected Weaknesses

1. **Nonlinear parameters**: c appears in exponent → potential sampling difficulties
2. **Parameter correlations**: a, b, c likely correlated → slow mixing
3. **Local optima**: Nonlinear likelihood may have multiple modes
4. **Identifiability**: With n=27, three nonlinear parameters may be hard to pin down
5. **Cannot capture sharp regime change**: Exponential decay is smooth by construction

#### Falsification Criteria

**I will reject this model if**:

1. **Asymptote estimate is wrong**: a < max(Y) (model can't reach observed data)
2. **Rate parameter at extremes**: c < 0.05 (no saturation) or c > 5 (instantaneous saturation)
3. **Computational failure**: High divergences (>1%), low ESS (<100), Rhat > 1.05
4. **Prior-posterior conflict**: Posterior for `a` strongly contradicts prior (suggests misspecification)
5. **Residual regime structure**: Clear split in residuals at x≈7 (piecewise is better)
6. **Worse fit than log**: Similar complexity but lower LOO → simpler log model preferred

#### Stress Test

**Designed to break the model**:
- **Parameter recovery**: Simulate data from asymptotic model with known (a, b, c), refit
- If recovered parameters don't match true values within 95% credible intervals, model is unidentifiable
- **Alternative forms**: Fit other saturation curves (e.g., Michaelis-Menten: μ = a*x/(b+x))
- If alternative asymptotic form fits much better, exponential decay is wrong functional form
- **High-x extrapolation**: Predict Y at x=50, 100
- If predictions are wildly implausible, model structure is questionable

---

## Robustness Consideration: Student-t Likelihood

### Outlier at x=31.5

EDA identifies observation at x=31.5, Y=2.57 as highly influential (Cook's D = 1.51, standardized residual = -2.31).

### Robust Likelihood Variant

For each of the three models above, fit a **robust version** with Student-t errors:

```
Likelihood (Robust):
Y_i ~ StudentT(ν, μ_i, σ)

Additional prior:
ν ~ Gamma(2, 0.1)    # Degrees of freedom: mean=20, allows heavy tails
```

**ν (degrees of freedom)**:
- ν = ∞ → Normal distribution (Gaussian limit)
- ν = 1 → Cauchy distribution (very heavy tails)
- ν ≈ 5-10 → Moderately robust to outliers
- Gamma(2, 0.1): mean=20, SD=14, 95% interval [3, 50]
- Allows data to determine tail heaviness
- Rationale: If ν posterior << 10, outliers are present; if ν >> 30, Normal is sufficient

### Comparison Strategy

1. Fit all three models with **Normal likelihood** (6 total)
2. Fit all three models with **Student-t likelihood** (6 more, 12 total)
3. Compare via LOO-CV:
   - Do Student-t versions have better predictive performance?
   - What is posterior for ν? If ν < 10, heavy tails detected
4. Sensitivity analysis:
   - Refit all models **excluding x=31.5**
   - If conclusions change drastically, single point drives inference (problematic)

### Decision Rule

- If Student-t ν posterior has 95% CI entirely below 10: **Use Student-t**
- If Student-t ν posterior has 95% CI entirely above 30: **Use Normal**
- If uncertain: **Report both**, acknowledge structural uncertainty

---

## Model Comparison Framework

### Primary Criterion: Leave-One-Out Cross-Validation (LOO-CV)

Use PSIS-LOO via `loo` package (R) or `arviz` (Python).

**Metrics**:
- **ELPD_loo**: Expected log pointwise predictive density (higher is better)
- **SE**: Standard error of ELPD difference
- **Δ ELPD**: Difference from best model
- **Pareto k**: Diagnostic for influential points (k > 0.7 problematic)

**Decision rules**:
- If Δ ELPD < 4: Models are equivalent (use simplest)
- If 4 < Δ ELPD < 10: Weak preference
- If Δ ELPD > 10: Strong preference
- If SE(Δ ELPD) > |Δ ELPD|: Difference is not reliable

### Secondary Criteria

1. **WAIC** (Widely Applicable Information Criterion): Cross-check LOO
2. **Posterior Predictive Checks**:
   - Visual: Overlay posterior predictive draws on observed data
   - Quantitative: Test statistics (mean, SD, min, max, skewness)
   - P-values: P(T(Y_rep) > T(Y_obs)) for various test statistics T
3. **Residual Analysis**:
   - Plot residuals vs x
   - Plot residuals vs fitted values
   - Check for patterns, heteroscedasticity, outliers
4. **Parameter Interpretability**:
   - Are estimates scientifically plausible?
   - Are credible intervals reasonable?
   - Do parameters have clear meaning?

### Computational Diagnostics (MANDATORY)

All models must pass these checks or be rejected:

| Diagnostic | Threshold | Action if Failed |
|------------|-----------|------------------|
| R-hat | < 1.01 (strict) or < 1.05 (acceptable) | Increase warmup/iterations, reparameterize |
| ESS (bulk) | > 400 (strict) or > 100 (minimum) | Increase samples, check for multimodality |
| ESS (tail) | > 400 (strict) or > 100 (minimum) | Same as above |
| Divergences | 0 (strict) or < 1% (acceptable) | Increase adapt_delta, reparameterize |
| Max treedepth | < 5% hitting limit | Increase max_treedepth or simplify model |
| BFMI | > 0.3 | Reparameterize, check for funnel geometry |

**If diagnostics fail despite reparameterization**: Model is likely misspecified or unidentifiable. Consider abandoning that model class.

---

## Posterior Predictive Checks (PPCs)

### Critical Checks for All Models

1. **Overall distribution**:
   - Test statistic: mean(Y)
   - P-value: P(mean(Y_rep) > mean(Y_obs))
   - Should be near 0.5 if model is calibrated

2. **Spread**:
   - Test statistic: SD(Y)
   - Check if observed SD is in the bulk of posterior predictive distribution

3. **Range**:
   - Test statistics: min(Y), max(Y)
   - Are observed extremes unusually extreme?

4. **Regime-specific behavior** (critical for piecewise):
   - Test statistic: mean(Y | x ≤ 7) vs mean(Y | x > 7)
   - Does model capture the difference between regimes?

5. **Local smoothness**:
   - Test statistic: max|Y_i - Y_{i-1}| / |x_i - x_{i-1}| (maximum slope)
   - If piecewise is correct, observed max slope should be extreme
   - If smooth model is correct, observed max slope should be typical

6. **Outlier check**:
   - Test statistic: Number of observations beyond 2.5σ from mean
   - Should be ~1-2 for n=27 under Normal; fewer under Student-t

### Model-Specific Checks

**Logarithmic**:
- Posterior predictive for "correlation of residuals with log(x)"
- Should be near zero if model is correctly specified

**Piecewise**:
- Posterior predictive for "changepoint location"
- Generate Y_rep, find optimal changepoint in Y_rep
- Distribution of τ_rep should cover τ_obs (self-consistency check)

**Asymptotic**:
- Posterior predictive for "asymptote"
- max(Y_rep) should be consistent with max(Y_obs)
- If model predicts higher asymptote than ever observed, questionable

---

## Alternative Models if All Three Fail

### Escape Routes

If all parametric models show systematic failures:

1. **Power-law transformation**: Y ~ β₀ + β₁ * x^α, estimate α
   - Generalizes logarithmic (α → 0 limit)
   - More flexible but harder to interpret

2. **Michaelis-Menten**: μ = V_max * x / (K_M + x)
   - Common in biochemistry
   - Two parameters, interpretable (V_max = max velocity, K_M = half-saturation)

3. **Box-Cox transformation**: Model Y^λ or (Y^λ - 1)/λ
   - Estimate transformation parameter λ
   - Addresses non-normality in response

4. **Mixture model**: Y ~ π * N(μ₁, σ₁) + (1-π) * N(μ₂, σ₂)
   - Two latent populations
   - Explains heterogeneity not captured by x alone

5. **Measurement error model**: X_obs = X_true + ε_x
   - If x=31.5 is truly an error in x measurement, not Y
   - Requires assumptions about error distribution

6. **Non-parametric** (outside my scope, but recommend):
   - Gaussian Process regression
   - Spline-based models
   - Local regression (LOESS)

### When to Abandon Parametric Approach

**I will recommend switching to non-parametric methods if**:
- All three models have LOO-CV diagnostics showing multiple high Pareto k values (>0.7)
- Residual plots for all models show clear, systematic patterns
- PPCs consistently fail (p-values < 0.05) for multiple test statistics
- Data suggest latent structure not captured by single covariate x
- Domain expert feedback contradicts all parametric functional forms

---

## Prior Sensitivity Analysis

### Rationale

With n=27, priors may influence posteriors. Must check sensitivity.

### Protocol

For the **best-performing model** (determined by initial LOO-CV):

1. **Wide priors**: Increase all prior SDs by 3×
2. **Tight priors**: Decrease all prior SDs by 0.5×
3. **Alternative prior families**:
   - σ: Try Half-Cauchy(0.1) instead of Exponential
   - β₁ (log model): Try Cauchy(0.3, 0.15) instead of Normal (heavier tails)
   - τ (piecewise): Try Normal(7, 2) truncated to [3,12] instead of Uniform

### Evaluation

- Compare posterior means and 95% credible intervals
- If posteriors shift by > 20% of posterior SD, priors are influential
- If conclusions change (e.g., sign of effect), priors are too strong
- Document sensitivity and report uncertainty

### Decision Rule

- If prior-insensitive: Confidence in results increases
- If prior-sensitive: Report multiple prior specifications, acknowledge uncertainty
- If extremely prior-sensitive: Sample size is too small for reliable inference

---

## Implementation Plan

### Phase 1: Initial Fits (All Models, Normal Likelihood)

**Models to fit**:
1. Logarithmic (Normal)
2. Piecewise with continuity (Normal)
3. Piecewise without continuity (Normal)
4. Asymptotic (Normal)

**Stan/PyMC settings**:
- Warmup: 1000
- Sampling: 2000
- Chains: 4
- adapt_delta: 0.95 (start conservative)
- max_treedepth: 12

**Deliverables**:
- Convergence diagnostics table
- Parameter estimates with 95% credible intervals
- LOO-CV comparison table
- Residual plots for each model

### Phase 2: Robust Variants (Student-t Likelihood)

**Models to fit**:
1. Logarithmic (Student-t)
2. Piecewise with continuity (Student-t)
3. Piecewise without continuity (Student-t)
4. Asymptotic (Student-t)

**Comparison**:
- Compare Normal vs Student-t for each functional form
- Extract ν posteriors
- Identify if heavy tails are needed

**Decision**: Select best likelihood family for each functional form

### Phase 3: Detailed Analysis of Top 2 Models

**Based on Phase 1-2 LOO-CV**, select top 2 models for deep dive:

1. **Posterior Predictive Checks** (all PPCs listed above)
2. **Residual diagnostics** (detailed plots, tests)
3. **Parameter interpretation** (scientifically meaningful?)
4. **Prior sensitivity analysis**
5. **Sensitivity to x=31.5** (refit without outlier)
6. **Extrapolation analysis** (predict at x=40, 50, 100)

**Deliverables**:
- Comprehensive PPC report
- Sensitivity analysis results
- Final model recommendation with caveats

### Phase 4: Model Averaging (If Models are Close)

If top models have Δ ELPD < 4:

- Compute **Bayesian model averaging** weights: w_m ∝ exp(ELPD_m)
- Weighted posterior: p(θ | Y) = Σ w_m * p(θ | Y, M_m)
- Report averaged predictions with model uncertainty

**Rationale**: If models are equivalent, averaging accounts for structural uncertainty

---

## Success Criteria

### Minimum Standards for Publication/Deployment

1. **Convergence**: All chains converge (R-hat < 1.01)
2. **Effective samples**: ESS > 400 for all parameters
3. **No divergences**: Divergence rate < 0.1%
4. **Posterior predictive**: No PPC p-values < 0.05 or > 0.95 for critical test statistics
5. **LOO diagnostics**: All Pareto k < 0.7 (or at most 1-2 borderline points)
6. **Residuals**: No systematic patterns in residual plots
7. **Interpretability**: Parameter estimates are scientifically plausible
8. **Stability**: Prior sensitivity analysis shows robust conclusions

### Aspirational Standards

1. **Decisive model selection**: Best model has Δ ELPD > 10 from others
2. **Tight posteriors**: 95% credible intervals span < 50% of prior range (data informative)
3. **Perfect LOO**: All Pareto k < 0.5
4. **Accurate calibration**: PPC p-values uniformly distributed over replications

### Failure Criteria (Abandon Approach)

1. **Persistent divergences**: >5% even with adapt_delta=0.99
2. **Non-convergence**: R-hat > 1.05 after 10,000 iterations
3. **Extreme Pareto k**: >5 points with k > 0.7
4. **Systematic PPC failures**: Multiple test statistics with p < 0.01
5. **Parameter nonsense**: Posteriors at prior boundaries or physically impossible
6. **All models equivalent**: No model has Δ ELPD > 2 (suggests underfitting)

---

## Reporting Standards

### Required Outputs

1. **Model comparison table**:
   - All models with ELPD, SE, Δ ELPD, Pareto k diagnostics
2. **Parameter table**:
   - For each model: mean, SD, 2.5%, 97.5% quantiles
3. **Diagnostic table**:
   - R-hat, ESS_bulk, ESS_tail, divergences, max_treedepth
4. **Visualizations**:
   - Posterior predictive overlay on data
   - Residual vs x plot
   - Residual vs fitted plot
   - Trace plots for key parameters
   - Pairs plots showing parameter correlations
5. **PPC plots**:
   - At least 5 test statistics
   - Observed value marked on predictive distribution
6. **Sensitivity analysis**:
   - Prior sensitivity results
   - Outlier sensitivity (with/without x=31.5)

### Honesty Standards

- Report all models fit, not just best
- Acknowledge failures and limitations
- Quantify uncertainty (don't overstate confidence)
- If results are sensitive to choices, say so explicitly
- If no model is clearly best, admit it

---

## Summary: Models to Implement

| Model | Likelihood | Parameters | Prior Complexity | Expected Performance | Computational Risk |
|-------|------------|------------|------------------|----------------------|--------------------|
| Log (Normal) | Normal | 3 | Low | High | Low |
| Log (Student-t) | Student-t | 4 | Low | High | Low |
| Piecewise-Cont (Normal) | Normal | 4 | Medium | Very High | Medium |
| Piecewise-Cont (Student-t) | Student-t | 5 | Medium | Very High | Medium |
| Piecewise-Discont (Normal) | Normal | 5 | Medium | High | Medium-High |
| Piecewise-Discont (Student-t) | Student-t | 6 | Medium | High | High |
| Asymptotic (Normal) | Normal | 4 | Medium-High | Medium-High | Medium-High |
| Asymptotic (Student-t) | Student-t | 5 | Medium-High | Medium-High | High |

**Recommendation**: Start with 4 core models (Log/Normal, Piecewise-Cont/Normal, Asymptotic/Normal, best of those with Student-t). Expand if needed.

---

## Timeline Estimate

Assuming Stan/PyMC implementation:

- **Phase 1** (Initial fits): 2-4 hours (model coding + fitting + diagnostics)
- **Phase 2** (Robust variants): 1-2 hours (reuse code, change likelihood)
- **Phase 3** (Deep dive): 3-6 hours (PPCs, sensitivity, interpretation)
- **Phase 4** (Model averaging, if needed): 1-2 hours

**Total**: 7-14 hours for comprehensive analysis

**Fastest path to initial results**: 1 hour (fit Log/Normal and Piecewise-Cont/Normal, quick LOO comparison)

---

## Final Philosophical Note

These models are **simplifications of reality**. All are wrong; the question is which is useful. I expect:

- **Logarithmic** will be simplest and robust but miss regime structure
- **Piecewise** will fit best but may overfit and have identifiability issues
- **Asymptotic** will be theoretically elegant but computationally fragile

**The data will decide**. If all three models fail in the same way, I've missed something fundamental. If they succeed in different ways, we'll learn about structural uncertainty. Either outcome is valuable.

Success is learning where the models break, not confirming they work.

---

**End of Model Design Document**

**Next steps**: Implement in Stan/PyMC, fit models, report results.
