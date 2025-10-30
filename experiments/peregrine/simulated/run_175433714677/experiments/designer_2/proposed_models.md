# Bayesian Model Design: Flexible & Complexity-Embracing Approach

**Design Philosophy**: Capture the rich complexity revealed by EDA. Willing to add parameters when they capture genuine data features rather than noise.

**Designer**: Designer 2 (Flexibility-focused)
**Date**: 2025-10-29
**EDA Source**: `/workspace/eda/eda_report.md`, `/workspace/eda/synthesis.md`

---

## Executive Summary

The EDA reveals **rich, multi-faceted complexity**:
- **Severe overdispersion** (Var/Mean ≈ 70, φ ≈ 1.5)
- **Non-linear acceleration** (9.6x growth rate increase)
- **Regime shift** at year ≈ -0.21 (Chow test p < 0.000001)
- **Heteroscedasticity** (variance varies 20x across time)
- **Quadratic fit** achieves R² = 0.96

This is NOT simple exponential growth. I propose three models that progressively capture increasing complexity, each justified by specific EDA findings.

---

## Model 1: Quadratic Negative Binomial with Heteroscedastic Dispersion

### Mathematical Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
```

**Mean Function** (captures acceleration):
```
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```

**Dispersion Function** (captures heteroscedasticity):
```
log(φ[i]) = γ₀ + γ₁ × year[i]
```

**Priors**:
```stan
// Mean function
β₀ ~ Normal(4.3, 1.0)        // EDA: log(mean) ≈ 4.3
β₁ ~ Normal(0.85, 0.5)       // EDA: positive growth
β₂ ~ Normal(0.3, 0.3)        // EDA: positive acceleration (informed, not centered at 0)

// Dispersion function
γ₀ ~ Normal(0.4, 0.5)        // EDA: log(1.5) ≈ 0.4
γ₁ ~ Normal(0, 0.3)          // Allow dispersion to vary with time
```

### Theoretical Justification

**Why this captures the data**:
1. **Quadratic term**: EDA shows R² jumps from 0.88 (linear) to 0.96 (quadratic)
2. **Acceleration**: Growth rate increases 9.6x from early to late period
3. **Time-varying dispersion**: Var/Mean varies from 0.58 (early) to 11.85 (middle) to 4.4 (late)
4. **Smooth function**: Captures regime shift without imposing discontinuity

**Why simpler models fail**:
- **Log-linear model** (no β₂): Cannot capture the visible curvature in growth
- **Constant dispersion** (no γ₁): Fails to model 20x variance change across time
- Visual evidence: Quadratic fit closely tracks the data; linear fit shows systematic bias

### Expected Parameter Values (from EDA)

```
β₀: 4.3 ± 0.3   (intercept at standardized year = 0)
β₁: 0.85 ± 0.2  (linear growth term)
β₂: 0.3 ± 0.15  (acceleration term, positive)
γ₀: 0.4 ± 0.3   (baseline dispersion)
γ₁: -0.2 ± 0.3  (dispersion may decrease late, given U-shaped variance pattern)
```

### Falsification Criteria

**I will REJECT this model if**:

1. **β₂ credible interval includes 0** → Quadratic term not justified, revert to linear
2. **γ₁ credible interval includes 0** → Dispersion is constant, simplify to φ[i] = φ
3. **Posterior predictive checks fail**:
   - Cannot reproduce Var/Mean ≈ 70 globally
   - Cannot reproduce time-varying variance pattern (20x range)
   - Prediction intervals miss > 10% of observed data
4. **Prior-posterior conflict**:
   - Posterior β₂ < -0.5 (negative acceleration contradicts EDA)
   - Posterior γ₀ < -1.0 or > 2.0 (extreme dispersion values)
5. **LOO warnings**:
   - More than 5 observations with Pareto-k > 0.7 (influential points)
   - ELPD_loo worse than simpler nested models by > 2 SE
6. **Computational issues**:
   - R-hat > 1.01 for any parameter (non-convergence)
   - ESS < 400 despite 4 chains × 2000 iterations
   - Divergent transitions > 1% (misspecified geometry)

### How This Captures Patterns Simpler Models Miss

**vs. Log-Linear Model**:
- Captures the **visible curvature** in scatter plots
- Models the **9.6x acceleration** in growth rate
- Achieves R² = 0.96 vs. 0.88

**vs. Constant Dispersion**:
- Captures **U-shaped variance pattern**: high → low → high
- Models variance ranging from 23 (early) to 461 (middle) to 103 (late)
- Better calibration of prediction intervals across time

**What complexity this adds**:
- 2 extra parameters (β₂, γ₁) justified by strong EDA evidence
- Minimal risk: both nest to simpler models if data doesn't support them

---

## Model 2: Piecewise Negative Binomial (Explicit Regime Shift)

### Mathematical Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
```

**Mean Function** (explicit changepoint):
```
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × I(year[i] > τ) + β₃ × (year[i] - τ) × I(year[i] > τ)

where:
  τ = changepoint location (fixed at -0.21 from EDA)
  I(·) = indicator function
```

**Dispersion Function** (regime-specific):
```
log(φ[i]) = γ₀ + γ₁ × I(year[i] > τ)
```

**Priors**:
```stan
// Pre-regime parameters
β₀ ~ Normal(4.0, 1.0)        // Early baseline
β₁ ~ Normal(0.3, 0.3)        // Slow initial growth

// Regime shift parameters
β₂ ~ Normal(0, 0.5)          // Level shift at changepoint
β₃ ~ Normal(0.5, 0.5)        // Additional slope in regime 2 (expect positive)

// Dispersion parameters
γ₀ ~ Normal(0.4, 0.5)        // Pre-regime dispersion
γ₁ ~ Normal(0, 0.5)          // Dispersion change post-regime
```

### Theoretical Justification

**Why this captures the data**:
1. **Chow test significance**: Structural break at year = -0.21 (p < 0.000001)
2. **Growth acceleration**: 9.6x increase from regime 1 to regime 2
3. **Interpretability**: May reflect real external event (policy change, technological shift)
4. **Parsimony**: 5 parameters vs. infinite smoothness of quadratic

**Why Model 1 (quadratic) might fail**:
- **If shift is discontinuous**: Quadratic forces smooth transition
- **If physical process changed**: Real regime shifts are often abrupt
- **Extrapolation**: Polynomials behave poorly outside data range

### Expected Parameter Values

```
β₀: 4.0 ± 0.3   (log-mean in early regime)
β₁: 0.3 ± 0.2   (slow early growth)
β₂: 0.5 ± 0.3   (level jump at changepoint)
β₃: 2.5 ± 0.5   (much steeper slope post-regime, to achieve 9.6x acceleration)
γ₀: 0.5 ± 0.3   (early dispersion)
γ₁: -0.3 ± 0.4  (possibly lower dispersion in late regime)
```

### Falsification Criteria

**I will REJECT this model if**:

1. **No regime shift detected**:
   - β₃ credible interval includes 0 (no slope change)
   - β₂ credible interval includes 0 (no level shift)
   - Both would suggest smooth growth, not regime change

2. **Discontinuity artifacts**:
   - Large residuals clustered near changepoint
   - Posterior predictive draws show "jumps" not in data

3. **Arbitrary changepoint**:
   - Changepoint at τ = -0.21 has no scientific justification
   - Model fit not substantially better than quadratic

4. **Worse predictive performance**:
   - LOO-CV: ELPD_loo worse than quadratic by > 2 SE
   - More influential observations (Pareto-k > 0.7)

5. **Computational red flags**:
   - R-hat > 1.01, ESS < 400, divergences > 1%

### How This Captures Patterns Model 1 Misses

**vs. Quadratic**:
- **Interpretable regime shift**: Can attach meaning to year = -0.21
- **Discontinuous acceleration**: If growth truly "jumps", quadratic smooths artificially
- **Better extrapolation**: Piecewise linear safer than polynomial outside data

**What complexity this adds**:
- Explicit changepoint assumption (strong prior belief)
- 5 parameters total, same as Model 1
- Risk: If shift is actually smooth, forces artificial discontinuity

---

## Model 3: Hierarchical Spline with Time-Varying Dispersion (Maximum Flexibility)

### Mathematical Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
```

**Mean Function** (B-spline with 6 knots):
```
log(μ[i]) = ∑(k=1 to K) β_k × B_k(year[i])

where:
  B_k = B-spline basis functions (degree 3, K=6 knots)
  Knots placed at quantiles: 0.05, 0.25, 0.40, 0.60, 0.75, 0.95
```

**Dispersion Function** (cubic B-spline):
```
log(φ[i]) = ∑(j=1 to J) γ_j × B_j(year[i])

where:
  J = 4 knots for dispersion (fewer than mean)
```

**Hierarchical Priors** (regularization to prevent overfitting):
```stan
// Mean function coefficients (hierarchical shrinkage)
β ~ Normal(0, σ_β)
σ_β ~ Exponential(1)       // Adaptive shrinkage

// Dispersion coefficients
γ ~ Normal(0, σ_γ)
σ_γ ~ Exponential(2)       // Stronger shrinkage for dispersion

// Overall intercept (identified separately)
α ~ Normal(4.3, 1.0)       // Mean log-count
```

### Theoretical Justification

**Why this captures the data**:
1. **Maximum flexibility**: Can capture ANY smooth non-linear pattern
2. **Local features**: Can model U-shaped variance pattern naturally
3. **No functional form assumptions**: Doesn't impose quadratic or piecewise structure
4. **Regularization**: Hierarchical priors prevent overfitting with n=40

**Why Models 1-2 might fail**:
- **If pattern is neither quadratic nor piecewise**: Complex growth dynamics
- **Multiple changepoints**: EDA only tested one, but there could be more
- **Non-parametric insurance**: When you don't trust parametric forms

### Expected Parameter Values

```
β_k: Variable depending on knot location
  - Early knots (year < -0.5): β_k ≈ 3.5-4.0 (low baseline)
  - Middle knots: β_k ≈ 4.5-5.0 (moderate)
  - Late knots (year > 0.5): β_k ≈ 5.5-6.0 (high growth)

γ_j: Dispersion coefficients
  - Expect U-shaped pattern: high early, low middle, high late

σ_β: 0.3-0.8 (moderate regularization)
σ_γ: 0.2-0.5 (stronger regularization)
```

### Falsification Criteria

**I will REJECT this model if**:

1. **Overfitting detected**:
   - LOO-CV shows many Pareto-k > 0.7 warnings
   - ELPD_loo worse than simpler parametric models
   - Posterior predictive draws are "wiggly" and unrealistic

2. **No advantage over parametric**:
   - Spline fits are nearly identical to quadratic (wasted complexity)
   - σ_β very small (all β_k shrunk to zero difference)

3. **Computational failure**:
   - Cannot achieve convergence (R-hat > 1.01) despite tuning
   - ESS < 400 for spline coefficients
   - Divergent transitions > 5% (geometry issues with high-dimensional space)

4. **Poor calibration**:
   - Despite flexibility, still cannot capture variance structure
   - Prediction intervals worse than simpler models

5. **Interpretation failure**:
   - Cannot extract meaningful pattern from spline coefficients
   - Results not scientifically interpretable

### How This Captures Patterns Models 1-2 Miss

**vs. Quadratic**:
- **No symmetry assumption**: Quadratic is symmetric around peak
- **Local adaptation**: Can fit different shapes in different regions
- **Multiple inflection points**: Quadratic has only one

**vs. Piecewise**:
- **No artificial discontinuities**: Smooth everywhere
- **No need to specify changepoint**: Data determines shape
- **Multiple regimes**: Can capture more than 2 regimes if present

**What complexity this adds**:
- 10+ parameters (6 mean knots + 4 dispersion knots)
- Requires hierarchical shrinkage to avoid overfitting
- Much slower computation (higher-dimensional posterior)
- Risk: May overfit with n=40 despite regularization

---

## Model Comparison Strategy

### Testing Order (Progressive Complexity)

1. **Start with Model 1** (Quadratic + time-varying dispersion)
   - Most justified by EDA (R² = 0.96, clear heteroscedasticity)
   - 5 parameters, computationally tractable
   - **Checkpoint**: If β₂ and γ₁ both non-significant → simplify to log-linear

2. **If Model 1 shows systematic residual patterns** → Try Model 2 (Piecewise)
   - Red flag: Residuals clustered around year = -0.21
   - Red flag: Posterior predictive checks show "corners" in data

3. **If Models 1-2 both fail** → Try Model 3 (Spline)
   - Red flag: Neither parametric form captures growth dynamics
   - Red flag: Visual inspection suggests more complex shape

4. **If all fail** → Reconsider data generation process entirely

### Decision Rules

**Prefer simpler model if**:
- ΔELPD < 2 × SE (no significant predictive improvement)
- Pareto-k warnings increase with complexity
- Posterior predictive checks are similar

**Switch model class if**:
- Systematic residual patterns remain
- Prior-posterior conflict (model fighting the data)
- Scientific interpretation suggests different mechanism

### Diagnostics for ALL Models

**Convergence** (required):
- R-hat < 1.01 for all parameters
- ESS_bulk > 400 for all parameters
- ESS_tail > 400 for all parameters
- Divergent transitions < 1%

**Posterior Predictive Checks** (critical):
- **Overdispersion recovery**: Posterior Var/Mean ≈ 70 (observed)
- **Variance pattern**: Can reproduce 20x variance change
- **Growth pattern**: Captures acceleration visually
- **Coverage**: 90% prediction intervals contain ~90% of data

**Model Comparison**:
- LOO-CV with PSIS diagnostics
- Compare ELPD_loo ± SE across models
- Check Pareto-k statistics (want < 0.7)

---

## Red Flags That Trigger Major Pivots

### Abandon ALL proposed models if:

1. **Fundamental distributional failure**:
   - Negative Binomial cannot capture dispersion pattern
   - Posterior predictive checks consistently fail
   - → Consider: Zero-inflated models, Hurdle models, Conway-Maxwell-Poisson

2. **Missing covariate signal**:
   - Large unexplained residual variance despite complexity
   - Patterns correlate with time index (not year)
   - → Consider: Additional covariates may exist, measurement error models

3. **Non-count process**:
   - Residual patterns suggest continuous latent process
   - Discretization artifacts visible
   - → Consider: Rounded continuous models, latent Gaussian process

4. **Temporal dependence emerges**:
   - Residual autocorrelation despite complex mean function
   - → Consider: State-space models, GP-based approaches

5. **Computational impossibility**:
   - Cannot achieve convergence for ANY model despite extensive tuning
   - → Consider: Model misspecification, need different parameterization

### Alternative Model Classes (Escape Routes)

If proposed models fail, consider:

1. **Gaussian Process Regression**:
   - If spline still shows patterns, go fully non-parametric
   - GP with squared-exponential kernel + NB likelihood

2. **State-Space Model**:
   - If residual autocorrelation emerges
   - Dynamic linear model with NB observation equation

3. **Mixture Models**:
   - If data suggests multiple populations
   - Finite mixture of NB distributions

4. **Zero-Inflated / Hurdle**:
   - If zeros appear in residual analysis
   - Two-part model: occurrence + count

---

## Stress Tests (Designed to Break Models)

### Test 1: Holdout Prediction
- **Withheld**: Last 8 observations (20% of data)
- **Success criteria**: RMSE < 20, MAE < 15
- **Failure signal**: Predictions systematically too high/low (model doesn't extrapolate)

### Test 2: Leave-One-Out Extremes
- **Method**: Iteratively remove highest/lowest 5 observations
- **Success criteria**: Parameter estimates stable (< 20% change)
- **Failure signal**: Extreme observations drive all inference (model unstable)

### Test 3: Variance Reproduction
- **Method**: Compute Var/Mean in early/middle/late thirds
- **Success criteria**: Posterior predictive Var/Mean matches observed within 50%
- **Failure signal**: Cannot reproduce heteroscedasticity

### Test 4: Prior Sensitivity
- **Method**: Refit with 0.5× and 2× prior scales
- **Success criteria**: Posterior means change < 0.3 SD
- **Failure signal**: Prior dominates likelihood (weak data signal)

---

## Implementation Requirements (Stan/CmdStanPy)

### All models must include:

```stan
generated quantities {
  vector[N] log_lik;        // For LOO-CV
  array[N] int C_rep;       // Posterior predictive draws
  real var_mean_ratio;      // Overdispersion check

  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi[i]);
    C_rep[i] = neg_binomial_2_rng(mu[i], phi[i]);
  }

  var_mean_ratio = variance(to_vector(C_rep)) / mean(to_vector(C_rep));
}
```

### Sampling Configuration

```python
# Use CmdStanPy for all models
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.95,      # Prevent divergences
    max_treedepth=12,      # Allow complex geometry
    seed=42,
    show_progress=True
)
```

### Model Validation Pipeline

```python
# 1. Convergence diagnostics
check_rhat(fit)           # All < 1.01
check_ess(fit)            # All > 400
check_divergences(fit)    # < 1%

# 2. Posterior predictive checks
ppc_variance_ratio(fit, data)
ppc_coverage(fit, data, interval=0.9)
ppc_growth_pattern(fit, data)

# 3. LOO-CV comparison
loo_results = az.loo(idata, pointwise=True)
check_pareto_k(loo_results)  # < 0.7 for all points

# 4. Model comparison
az.compare({"model1": idata1, "model2": idata2, ...})
```

---

## Summary: Why These Models Over Simpler Alternatives

### Why Not Log-Linear (Designer 1's likely preference)?

**Evidence against log-linear**:
- R² = 0.88 vs. 0.96 (quadratic) — substantial improvement
- Visual inspection shows clear curvature
- 9.6x growth acceleration cannot be captured by constant exponential rate
- Chow test p < 0.000001 for structural break

**My position**: Log-linear is a **testable hypothesis**, not a default. The EDA provides strong evidence it's insufficient.

### Why Not Fixed Dispersion?

**Evidence for time-varying dispersion**:
- Var/Mean varies from 0.58 to 11.85 (20x range)
- Levene's test p < 0.01 for heteroscedasticity
- Constant dispersion will produce miscalibrated prediction intervals

**My position**: With clear heteroscedasticity, **modeling it improves inference quality**.

### What If I'm Wrong?

**Falsification protects against overconfidence**:
- If β₂ ≈ 0: Quadratic not justified → revert to linear
- If γ₁ ≈ 0: Time-varying dispersion not needed → revert to constant
- If LOO-CV prefers simpler model → accept simpler model

**The data will decide**, not my prior preference for complexity.

---

## Conclusion

I propose three models that progressively capture increasing complexity, each justified by specific EDA patterns:

1. **Model 1** (Quadratic + time-varying φ): Best balance of flexibility and parsimony
2. **Model 2** (Piecewise): If regime shift is scientifically meaningful
3. **Model 3** (Spline): If parametric forms prove inadequate

All models:
- Use Negative Binomial likelihood (φ ≈ 1.5)
- Use log link function
- Include time-varying dispersion (φ[i])
- Have clear falsification criteria
- Will be compared via LOO-CV

**My prediction**: Model 1 will perform best, but I'm ready to be proven wrong by the data.

**Success criterion**: Finding a model that genuinely explains the data, not completing this plan.

---

**Files**:
- This plan: `/workspace/experiments/designer_2/proposed_models.md`
- EDA report: `/workspace/eda/eda_report.md`
- Synthesis: `/workspace/eda/synthesis.md`
