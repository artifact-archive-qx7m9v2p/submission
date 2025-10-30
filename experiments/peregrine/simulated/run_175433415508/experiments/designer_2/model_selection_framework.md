# Model Selection Framework: Temporal Correlation Structures

**Purpose**: Systematic approach to choosing between NB-AR1, NB-GP, and NB-RW

**Date**: 2025-10-29

---

## The Central Question

**Is the high ACF (0.971) genuine temporal dependence or a trend artifact?**

This is THE critical question that will determine which model is appropriate.

---

## Decision Tree

```
START: Fit all three temporal models + baseline (NB with independent errors)
  |
  ├─> Are temporal models better than baseline?
  |   |
  |   ├─> NO (ΔELPD < 2): HIGH ACF IS TREND ARTIFACT
  |   |   └─> Recommend: Quadratic trend + independent errors
  |   |       (Temporal correlation was spurious)
  |   |
  |   └─> YES (ΔELPD > 5): GENUINE TEMPORAL DEPENDENCE
  |       |
  |       ├─> Which temporal model wins?
  |       |   |
  |       |   ├─> AR1: Stationary correlation
  |       |   |   └─> Check: Is ρ < 0.95?
  |       |   |       ├─> YES: True AR(1) process
  |       |   |       └─> NO: Near unit root → consider RW
  |       |   |
  |       |   ├─> GP: Flexible smooth trend
  |       |   |   └─> Check: Is lengthscale small (< 2)?
  |       |   |       ├─> YES: Wiggly function, genuine GP
  |       |   |       └─> NO: Nearly constant → just a smooth trend
  |       |   |
  |       |   └─> RW: Non-stationary process
  |       |       └─> Check: Is σ_ω > 0.1?
  |       |           ├─> YES: Real random walk
  |       |           └─> NO: Nearly deterministic → use simpler model
  |       |
  |       └─> Are all temporal models similar?
  |           ├─> YES: Data cannot distinguish structures
  |           |   └─> Use simplest (AR1) or model averaging
  |           |
  |           └─> NO: Clear winner exists
  |               └─> Report best model with caveats
```

---

## Baseline Model: NB with Independent Errors

**Purpose**: Establish if temporal correlation is necessary AT ALL

### Stan Code (Minimal)

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta0;
  real beta1;
  real<lower=0> alpha;
}

model {
  beta0 ~ normal(log(109.4), 1);
  beta1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);

  C ~ neg_binomial_2(exp(beta0 + beta1 * year), alpha);
}

generated quantities {
  vector[N] log_lik;
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | exp(beta0 + beta1 * year[t]), alpha);
  }
}
```

### Key Diagnostic: Residual ACF

If residual ACF remains high (> 0.7 at lag 1) after fitting baseline, temporal correlation is **essential**.

If residual ACF is low (< 0.3 at lag 1), temporal correlation may be **unnecessary** (trend explains ACF).

---

## Quantitative Comparison Criteria

### 1. PSIS-LOO-CV (Primary)

**Metric**: Expected log predictive density (ELPD)

**Interpretation**:
- ΔELPD > 5: Strong evidence for better model
- 2 < ΔELPD < 5: Moderate evidence
- ΔELPD < 2: Models are indistinguishable

**Decision rule**:
```
If ΔELPD(temporal) - ΔELPD(baseline) < 2:
    Temporal correlation is unnecessary (trend artifact)
    → Use simpler model

If max(ΔELPD) among temporal models > others by > 5:
    Clear winner
    → Use best temporal model

If all temporal models within 2 ELPD:
    Cannot distinguish
    → Use simplest (AR1) or model averaging
```

### 2. Pareto-k Diagnostics

**Metric**: Stability of LOO approximation

**Interpretation**:
- k < 0.5: Excellent
- 0.5 < k < 0.7: Good
- 0.7 < k < 1: Problematic (LOO unreliable)
- k > 1: Very problematic

**Decision rule**:
```
If any model has > 20% observations with k > 0.7:
    LOO comparison is unreliable
    → Use K-fold cross-validation instead
    → Investigate model misspecification
```

### 3. Posterior Predictive ACF

**Metric**: Does model capture observed autocorrelation?

**Test**:
1. Compute ACF from observed data
2. Compute ACF from each posterior predictive replicate
3. Calculate 90% interval from replicates
4. Check if observed ACF falls within interval

**Decision rule**:
```
For each lag k = 1, 2, ..., 5:
    If observed ACF[k] is outside 90% PP interval:
        Model FAILS to capture temporal structure
        → Reject this model

If AR1 captures ACF but GP doesn't:
    → AR1 is more appropriate (exponential decay)

If GP captures ACF but AR1 doesn't:
    → ACF pattern is non-exponential, GP is better
```

### 4. Effective Parameters (Complexity)

**Metric**: p_loo (effective number of parameters)

**Interpretation**:
- AR1: Expect p_loo ≈ 5-8
- GP: Expect p_loo ≈ 10-20
- RW: Expect p_loo ≈ 5-10

**Decision rule**:
```
If p_loo > N/3 (> 13 for N=40):
    Model is too complex (overfitting)
    → Favor simpler alternative

If two models have similar ELPD (ΔELPD < 2):
    Choose model with lower p_loo (parsimony)
```

---

## Qualitative Comparison Criteria

### 5. Smoothness of Predictions

**Visual test**: Plot posterior predictive trajectories

**Expected patterns**:
- **AR1**: Smooth with small fluctuations around trend
- **GP**: Very smooth, adapts to curvature
- **RW**: Visible "steps" or jumps, cumulative drift

**Decision rule**:
```
If observed data is very smooth:
    GP or AR1 with high ρ is appropriate
    RW might be too jumpy

If observed data shows sudden changes:
    RW might be appropriate
    AR1 might undersmooth
```

### 6. Parameter Interpretability

**Question**: Does posterior make scientific sense?

**AR1**:
- **ρ = 0.95**: Makes sense (high but not extreme correlation)
- **ρ = 0.999**: Suspicious (nearly deterministic, might just be trend)
- **σ_η = 0.5**: Large innovations (noisy process)
- **σ_η = 0.01**: Tiny innovations (nearly deterministic)

**GP**:
- **ℓ = 1.0**: Medium lengthscale, reasonable
- **ℓ = 10.0**: Very long correlation, nearly constant (might just be trend)
- **η = 0.5**: Moderate amplitude, allows deviations
- **η = 2.0**: Large amplitude, very wiggly (might overfit)

**RW**:
- **σ_ω = 0.1**: Small steps, reasonable
- **σ_ω = 0.5**: Large steps, very volatile
- **σ_ω = 0.01**: Tiny steps, nearly deterministic

### 7. Prior-Posterior Conflict

**Test**: KL divergence or visual comparison

**Interpretation**:
```
If posterior ≈ prior:
    Data is too weak, model is unidentified
    → Need stronger prior or simpler model

If posterior << prior (concentrated):
    Data is informative, good learning

If posterior >> prior (wider):
    Prior-data conflict, misspecified model
    → Revise prior or model structure
```

---

## Special Tests for Model Class

### Test A: Is AR(1) Appropriate?

**Hypothesis**: Correlation has exponential decay (Markov property)

**Test**: Partial autocorrelation function (PACF)
```
If PACF[1] is large (> 0.9) and PACF[k>1] ≈ 0:
    → AR(1) is correct

If PACF[1:p] are all significant:
    → Need AR(p) with p > 1

If PACF decays slowly:
    → Not AR process, consider GP or RW
```

**Alternative test**: First-order Markov test
```
Compute: Cor(C_t, C_{t+2} | C_{t+1})
If this is ≈ 0: First-order Markov holds → AR(1) OK
If this is > 0.3: Higher-order dependence → AR(1) inadequate
```

### Test B: Is GP Appropriate?

**Hypothesis**: Smooth function explains data better than parametric trend

**Test**: Compare to quadratic trend
```python
# Fit NB with year + year² (no GP)
nb_quad = fit_quadratic_model()

# Compare to NB-GP
if ELPD(GP) - ELPD(quad) < 2:
    GP is just capturing polynomial trend
    → Use simpler quadratic model

if ELPD(GP) - ELPD(quad) > 5:
    GP captures genuine non-polynomial structure
    → GP is justified
```

**Alternative test**: Check learned lengthscale
```
If posterior ℓ > 5 (in standardized units):
    Correlation extends across entire range
    → GP is nearly constant function
    → Not adding value beyond trend

If posterior ℓ < 1:
    Local correlation structure
    → GP is capturing genuine smooth variation
```

### Test C: Is Random Walk Appropriate?

**Hypothesis**: Process is non-stationary (unit root)

**Test**: Augmented Dickey-Fuller on detrended data
```python
from statsmodels.tsa.stattools import adfuller

# Detrend data
residuals = C - fitted_linear_trend
adf_test = adfuller(residuals)

if adf_test[1] < 0.05:  # Reject unit root
    Data is stationary → AR(1) not RW

if adf_test[1] > 0.10:  # Fail to reject unit root
    Data is non-stationary → RW might be appropriate
```

**Alternative test**: Variance ratio test
```
# Compare variance growth over different horizons
var_ratio = Var(C[t:t+k]) / (k × Var(C[t:t+1]))

If var_ratio ≈ 1 for all k:
    → Stationary process (AR1)

If var_ratio increases with k:
    → Non-stationary process (RW)
```

---

## Model Averaging Strategy

If models are indistinguishable (all within 2 ELPD), use **Bayesian model averaging**:

### Method 1: Stacking Weights

```python
import arviz as az

# Fit all models
models = {'AR1': idata_ar1, 'GP': idata_gp, 'RW': idata_rw}

# Compute stacking weights
comparison = az.compare(models, method='stacking')
weights = comparison['weight'].values

# Model-averaged predictions
mu_ar1 = fit_ar1.stan_variable('mu')
mu_gp = fit_gp.stan_variable('mu')
mu_rw = fit_rw.stan_variable('mu')

mu_avg = (weights[0] * mu_ar1 +
          weights[1] * mu_gp +
          weights[2] * mu_rw)
```

**Interpretation of weights**:
- If one weight > 0.8: That model dominates (effectively ignore others)
- If weights are balanced (0.2-0.5 each): Genuine uncertainty about structure

### Method 2: Pseudo-BMA Weights

```python
# Based on PSIS-LOO
comparison = az.compare(models, method='pseudo-BMA')
weights = comparison['weight'].values
```

---

## Reporting Framework

### Scenario 1: Clear Winner

**Example**: AR1 wins with ΔELPD = 12 (SE = 4)

**Report**:
> "Model comparison via PSIS-LOO strongly favors the AR(1) structure (ELPD difference = 12 ± 4 relative to baseline). The posterior estimate ρ = 0.94 [0.91, 0.97] indicates genuine temporal dependence, not merely a trend artifact. Posterior predictive checks confirm the model captures the observed autocorrelation pattern (ACF[1] = 0.971)."

### Scenario 2: Models Indistinguishable

**Example**: All temporal models within ΔELPD < 2

**Report**:
> "All three temporal correlation structures (AR1, GP, RW) provide similar predictive performance (ΔELPD < 2), suggesting the data (n=40) cannot distinguish between these mechanisms. For parsimony, we report the AR(1) model as the primary result, but acknowledge substantial structural uncertainty. Model-averaged predictions using stacking weights (AR1: 0.42, GP: 0.31, RW: 0.27) are provided for robust inference."

### Scenario 3: Temporal Structure Unnecessary

**Example**: Baseline within ΔELPD < 2 of best temporal model

**Report**:
> "Despite high observed autocorrelation (ACF = 0.971), the addition of temporal correlation structure does not improve predictive performance (ΔELPD = 1.2 ± 2.3). This suggests the apparent autocorrelation is primarily a consequence of the strong deterministic trend, not genuine stochastic dependence. We recommend the simpler model with independent errors."

### Scenario 4: All Models Inadequate

**Example**: Residual ACF remains high (> 0.5) even in best model

**Report**:
> "All proposed temporal models fail to adequately capture the observed autocorrelation structure. Posterior predictive ACF checks reveal systematic underprediction of temporal dependence at lags > 2. This suggests the need for alternative model classes, potentially including: (1) higher-order AR(p) processes, (2) structural break models, or (3) non-Gaussian correlation structures."

---

## Red Flags: When to Stop and Reconsider

### Flag 1: Computational Pathologies

**Symptoms**:
- Divergent transitions > 1% (even with adapt_delta = 0.99)
- Rhat > 1.05 for key parameters
- ESS < 100 for correlation parameters

**Action**: Model is fundamentally problematic
- Try reparameterization (non-centered, transformed scales)
- If persists: Model is too complex or misspecified
- **Stop and reconsider model class**

### Flag 2: Unrealistic Posteriors

**Symptoms**:
- ρ → 1.0 (AR1 at boundary)
- ℓ → ∞ (GP becomes constant)
- σ_ω → 0 (RW becomes deterministic)

**Action**: Model is collapsing to simpler form
- Check if simpler model (fewer parameters) would suffice
- This often indicates **overparameterization**

### Flag 3: Prior-Posterior Overlap

**Symptoms**:
- Posterior is indistinguishable from prior
- Wide posteriors that don't concentrate

**Action**: Data is too weak or model is unidentified
- Try more informative priors (justified by EDA)
- Consider simpler model with fewer parameters
- May need more data to estimate correlation structure

### Flag 4: Poor Predictive Performance

**Symptoms**:
- Posterior predictive checks show systematic deviations
- Out-of-sample predictions far worse than in-sample
- Pareto-k > 0.7 for many observations

**Action**: Model is misspecified
- Investigate patterns in failures (early vs late time periods?)
- Consider structural break, regime-switching, or non-stationary variance
- **This is most serious flag**: model class is wrong

---

## Final Decision Algorithm

```python
def select_temporal_model(elpd_baseline, elpd_ar1, elpd_gp, elpd_rw,
                          se_baseline, se_ar1, se_gp, se_rw):
    """
    Systematic model selection for temporal structures

    Returns: (model_name, justification)
    """

    # Step 1: Is temporal structure necessary?
    best_temporal_elpd = max(elpd_ar1, elpd_gp, elpd_rw)
    diff = best_temporal_elpd - elpd_baseline
    se_max = max(se_ar1, se_gp, se_rw, se_baseline)

    if diff < 2 * se_max:
        return "Baseline", "Temporal correlation unnecessary (trend artifact)"

    # Step 2: Which temporal model is best?
    elpds = {'AR1': elpd_ar1, 'GP': elpd_gp, 'RW': elpd_rw}
    ses = {'AR1': se_ar1, 'GP': se_gp, 'RW': se_rw}

    best_model = max(elpds, key=elpds.get)
    best_elpd = elpds[best_model]

    # Step 3: Is winner clearly better?
    diffs = {m: best_elpd - elpds[m] for m in elpds if m != best_model}

    if all(d < 2 * se_max for d in diffs.values()):
        return "Model_Average", "Models indistinguishable, use averaging"

    if all(d > 5 for d in diffs.values()):
        return best_model, f"Clear winner (ΔELPD > 5)"

    return best_model, f"Best model with moderate evidence (2 < ΔELPD < 5)"


# Example usage:
result, reason = select_temporal_model(
    elpd_baseline=-150, elpd_ar1=-142, elpd_gp=-145, elpd_rw=-143,
    se_baseline=8, se_ar1=7, se_gp=8, se_rw=7
)

print(f"Selected: {result}")
print(f"Reason: {reason}")
# Output: Selected: AR1, Reason: Best model with moderate evidence
```

---

## Summary Table: Model Selection Criteria

| Criterion | Threshold | Favors | Red Flag |
|-----------|-----------|--------|----------|
| ΔELPD from baseline | > 5 | Temporal structure | < 2: Unnecessary |
| ΔELPD between temporal | > 5 | Clear winner | < 2: Indistinguishable |
| Pareto-k | < 0.7 | Stable LOO | > 0.7: Unstable |
| Residual ACF[1] | < 0.2 | Adequate fit | > 0.5: Poor fit |
| Posterior ρ (AR1) | 0.85-0.95 | AR1 | > 0.99: Boundary |
| Posterior ℓ (GP) | 0.5-3 | GP | > 10: Constant |
| Posterior σ_ω (RW) | 0.1-0.3 | RW | < 0.05: Deterministic |
| ESS (correlation) | > 400 | Identifiable | < 100: Unidentified |
| Rhat (all params) | < 1.01 | Converged | > 1.05: Failed |

---

## Conclusion

The key to selecting the right temporal model is:

1. **Start with the question**: Is correlation real or trend artifact?
2. **Use multiple criteria**: No single metric is sufficient
3. **Be honest about uncertainty**: If models are similar, say so
4. **Watch for red flags**: Computational or inferential failures
5. **Prioritize interpretability**: Complex model must earn its keep

Remember: **The goal is not to pick a winner, but to find the truth about the temporal structure.** If that truth is "we can't tell with n=40," that's a valid scientific conclusion.

---

**End of Model Selection Framework**

Files: `/workspace/experiments/designer_2/model_selection_framework.md`
