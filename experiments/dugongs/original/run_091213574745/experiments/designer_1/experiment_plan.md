# Experiment Plan: Parametric Bayesian Models

**Designer**: Parametric Models Specialist (Designer 1)
**Date**: 2025-10-28
**Dataset**: /workspace/data/data.csv (n=27)
**Focus Area**: Logarithmic, piecewise, and asymptotic parametric models

---

## Quick Reference

**Three competing model classes**:
1. **Logarithmic**: Y ~ β₀ + β₁*log(x) - Simplest, smooth saturation
2. **Piecewise Linear**: Y ~ α₁ + β₁*x (if x≤τ), α₂ + β₂*x (if x>τ) - Sharp regime shift
3. **Asymptotic**: Y ~ a - b*exp(-c*x) - Mechanistic saturation

**Key questions**:
- Is the saturation smooth (log/asymptotic) or sharp (piecewise)?
- Do we need robust likelihood (Student-t) for outlier at x=31.5?
- Can we identify a changepoint location with n=27?

---

## Experimental Design

### Phase 1: Baseline Models (Normal Likelihood)

**Fit these 4 models first**:
1. Logarithmic (Normal): 3 parameters
2. Piecewise with continuity (Normal): 4 parameters
3. Piecewise without continuity (Normal): 5 parameters
4. Asymptotic (Normal): 4 parameters

**MCMC Settings**:
- Chains: 4
- Warmup: 1000
- Samples: 2000 per chain
- adapt_delta: 0.95
- max_treedepth: 12

**Outputs**: Convergence diagnostics, parameter estimates, LOO-CV comparison

### Phase 2: Robust Variants (Student-t Likelihood)

**For best 2 models from Phase 1**: Add Student-t likelihood variant

**Question**: Is ν posterior < 10? (If yes, heavy tails needed)

### Phase 3: Deep Dive on Top 2 Models

**Comprehensive analysis**:
- Posterior predictive checks (6+ test statistics)
- Residual diagnostics
- Prior sensitivity analysis
- Sensitivity to outlier at x=31.5
- Extrapolation behavior

### Phase 4: Final Selection & Reporting

**If models differ by Δ ELPD < 4**: Use Bayesian model averaging

**Final deliverable**: Model recommendation with uncertainty quantification

---

## Falsification Criteria

### Abandon Logarithmic Model If:
- Residuals cluster into two distinct groups (low-x vs high-x)
- Piecewise model has Δ ELPD > 10
- Posterior predictive check for "local slope change" shows extreme p-value (<0.05)

### Abandon Piecewise Model If:
- Changepoint posterior is flat/uniform (not identifiable)
- Slope difference P(β₁ > β₂) < 0.9 (regimes not distinct)
- Discontinuity at changepoint > 0.2 units (measurement artifact)
- Worse LOO than simpler logarithmic model

### Abandon Asymptotic Model If:
- Asymptote estimate a < max(Y) (can't reach observed data)
- Rate parameter c < 0.05 or c > 5 (extreme values)
- Persistent computational issues (divergences >1%, ESS < 100)
- Residuals show clear two-regime structure

### Abandon All Parametric Models If:
- All models have multiple Pareto k > 0.7 (influential points)
- Systematic PPC failures across all models (p < 0.05 for multiple tests)
- Residual patterns suggest missing covariates or latent structure

---

## Model Specifications

### Model 1: Logarithmic (Baseline)

```stan
// Likelihood
Y_i ~ Normal(mu[i], sigma)
mu[i] = beta0 + beta1 * log(x[i])

// Priors
beta0 ~ Normal(2.3, 0.3)     // Intercept at mean(Y)
beta1 ~ Normal(0.3, 0.15)    // Positive slope from EDA
sigma ~ Exponential(10)      // Mean = 0.1
```

**Why this model?**
- EDA shows R²=0.897, RMSE=0.087
- Theoretically motivated (diminishing returns, Weber-Fechner law)
- Simplest model with only 3 parameters

**Expected failure mode**: Cannot capture sharp regime transition at x≈7

---

### Model 2: Piecewise Linear (Continuous at Changepoint)

```stan
// Likelihood
Y_i ~ Normal(mu[i], sigma)
mu[i] = alpha1 + beta1*x[i]  if x[i] <= tau
mu[i] = alpha2 + beta2*x[i]  if x[i] > tau

// Continuity constraint
alpha2 = alpha1 + (beta1 - beta2) * tau

// Priors
tau ~ Uniform(3, 12)         // Changepoint location
alpha1 ~ Normal(1.8, 0.4)    // Regime 1 intercept
beta1 ~ Normal(0.1, 0.05)    // Regime 1 slope (steep)
beta2 ~ Normal(0.02, 0.02)   // Regime 2 slope (shallow)
sigma ~ Exponential(15)      // Mean = 0.067
```

**Why this model?**
- EDA F-test highly significant (F=22.4, p<0.0001)
- Captures sharp transition at x≈7
- Continuity prevents implausible jumps

**Expected failure mode**: Changepoint location may be weakly identified with n=27

---

### Model 3: Asymptotic (Mechanistic Saturation)

```stan
// Likelihood
Y_i ~ Normal(mu[i], sigma)
mu[i] = a - b * exp(-c * x[i])

// Priors
a ~ Normal(2.7, 0.2)         // Asymptote near max(Y)
b ~ Normal(0.9, 0.3)         // Initial gap from asymptote
c ~ Exponential(2)           // Rate, mean = 0.5
sigma ~ Exponential(12)      // Mean = 0.083

// Alternative parameterization (better for sampling):
// mu0 = a - b ~ Normal(1.8, 0.3)  // Y-intercept
```

**Why this model?**
- Biochemical interpretation (Michaelis-Menten, saturation kinetics)
- Smooth approach to equilibrium
- EDA shows R²=0.889, RMSE=0.090

**Expected failure mode**: Cannot capture sharp regime change; may have sampling issues with nonlinear parameters

---

### Model Variants: Robust Likelihood (Student-t)

For each model above, add Student-t variant:

```stan
// Robust likelihood
Y_i ~ StudentT(nu, mu[i], sigma)

// Additional prior
nu ~ Gamma(2, 0.1)  // Mean = 20, allows heavy tails
```

**Decision rule**:
- If ν posterior < 10: Heavy tails present, use Student-t
- If ν posterior > 30: Normal is adequate
- If uncertain: Report both

---

## Model Comparison Strategy

### Primary: Leave-One-Out Cross-Validation (LOO-CV)

**Metrics**:
- ELPD_loo: Expected log pointwise predictive density (higher is better)
- SE: Standard error of ELPD
- Δ ELPD: Difference from best model
- Pareto k: Diagnostic for influential points (k > 0.7 problematic)

**Decision rules**:
| Δ ELPD | Interpretation | Action |
|--------|----------------|--------|
| < 4 | Models equivalent | Use simplest or average |
| 4-10 | Weak preference | Prefer higher ELPD but report uncertainty |
| > 10 | Strong preference | Clear winner |

### Secondary: Posterior Predictive Checks (PPCs)

**Critical test statistics**:
1. mean(Y) - Overall calibration
2. SD(Y) - Spread
3. max(Y), min(Y) - Range
4. mean(Y|x≤7) vs mean(Y|x>7) - Regime difference
5. max|slope| - Local smoothness (critical for piecewise detection)
6. Number beyond 2.5σ - Outlier detection

**Failure**: Any PPC p-value < 0.05 or > 0.95 suggests misspecification

### Mandatory Convergence Diagnostics

| Diagnostic | Acceptable | Strict |
|------------|------------|--------|
| R-hat | < 1.05 | < 1.01 |
| ESS (bulk) | > 100 | > 400 |
| ESS (tail) | > 100 | > 400 |
| Divergences | < 1% | 0% |
| Max treedepth hits | < 5% | < 1% |

**If diagnostics fail**: Reparameterize or reject model as unidentifiable

---

## Sensitivity Analyses

### 1. Prior Sensitivity

For best model:
- **Wide priors**: 3× larger SD
- **Tight priors**: 0.5× smaller SD
- **Alternative families**: Half-Cauchy for σ, truncated Normal for τ

**Pass criterion**: Posterior means shift < 20% of posterior SD

### 2. Outlier Sensitivity

Refit best model **excluding x=31.5 (Y=2.57)**

**Pass criterion**: Parameter estimates shift < 10%, conclusions unchanged

### 3. Changepoint Prior Sensitivity (Piecewise Only)

Refit with:
- Uniform(5, 9) - tight around EDA estimate
- Uniform(1, 20) - very diffuse
- Normal(7, 2) truncated [3, 12] - informative

**Pass criterion**: Changepoint posteriors agree within 1.5 units

---

## Stress Tests

### Test 1: Parameter Recovery (Asymptotic Model)

1. Simulate data from asymptotic model with known (a, b, c)
2. Refit model to simulated data
3. Check if 95% credible intervals contain true values

**Pass**: ≥95% coverage in repeated simulations
**Fail**: Coverage << 95% → model is unidentifiable with n=27

### Test 2: Changepoint Self-Consistency (Piecewise Model)

1. Generate Y_rep from posterior predictive
2. Find optimal changepoint in Y_rep via grid search
3. Compare distribution of τ_rep to τ_posterior

**Pass**: τ_rep distribution overlaps τ_posterior substantially
**Fail**: τ_rep is flat/random → changepoint not real feature

### Test 3: Local Slope Test (Logarithmic Model)

1. Compute observed max slope change: max|Δy/Δx between consecutive points|
2. Generate Y_rep, compute max slope change in each replicate
3. Calculate P(max_slope_rep > max_slope_obs)

**Pass** (log correct): p-value ≈ 0.5
**Fail** (piecewise correct): p-value < 0.05 (observed slope change is extreme)

---

## Alternative Models (Escape Routes)

If all three primary models fail:

1. **Power-law**: Y ~ β₀ + β₁ * x^α (estimate α)
2. **Michaelis-Menten**: μ = V_max * x / (K_M + x)
3. **Box-Cox transformation**: Model Y^λ or (Y^λ-1)/λ
4. **Mixture model**: Two latent populations
5. **Measurement error in x**: Account for uncertainty in x=31.5

**Trigger for escape**: All models have Pareto k > 0.7 for multiple points, or systematic PPC failures

---

## Expected Outcomes & Interpretation

### Scenario A: Logarithmic Wins (Δ ELPD > 10)

**Interpretation**: Saturation is smooth and continuous; no sharp threshold

**Implication**: Y increases logarithmically with x (diminishing returns)

**Recommendation**: Use log-transformed predictor for future modeling

**Caveat**: May miss subtle regime structure; check PPCs carefully

### Scenario B: Piecewise Wins (Δ ELPD > 10)

**Interpretation**: Sharp threshold at x≈τ where system behavior changes

**Implication**: Two distinct operating regimes; resource exhaustion or phase transition

**Recommendation**: Investigate physical mechanism causing threshold

**Caveat**: Changepoint location has uncertainty; report credible interval for τ

### Scenario C: Asymptotic Wins (Δ ELPD > 10)

**Interpretation**: Y approaches biochemical/physical maximum via exponential decay

**Implication**: Process follows saturation kinetics (Michaelis-Menten-like)

**Recommendation**: Estimate asymptote `a` and saturation rate `c` for practical use

**Caveat**: Extrapolation beyond observed x is uncertain; asymptote may not be reached

### Scenario D: Models are Equivalent (All Δ ELPD < 4)

**Interpretation**: Data insufficient to distinguish functional forms

**Implication**: Structural uncertainty; multiple models explain data equally well

**Recommendation**: Use Bayesian model averaging; report predictions from all models

**Caveat**: Acknowledge uncertainty in functional form; collect more data if critical

### Scenario E: All Models Fail (Multiple PPC failures, high Pareto k)

**Interpretation**: Parametric forms are inadequate; missing structure in data

**Implication**: Need non-parametric approach or additional covariates

**Recommendation**: Switch to Gaussian Process or spline-based models

**Caveat**: May indicate measurement issues or heterogeneous data

---

## Deliverables

### Phase 1 (Initial Fits)
- **File**: `phase1_initial_fits.md`
- **Contents**: Convergence diagnostics, parameter tables, LOO comparison

### Phase 2 (Robust Variants)
- **File**: `phase2_robust_likelihood.md`
- **Contents**: Student-t fits, ν posteriors, Normal vs Student-t comparison

### Phase 3 (Deep Dive)
- **File**: `phase3_detailed_analysis.md`
- **Contents**: PPCs, residual diagnostics, sensitivity analyses

### Phase 4 (Final Report)
- **File**: `final_model_recommendation.md`
- **Contents**: Best model(s), parameter interpretation, predictions, limitations

### Visualizations
- **Directory**: `visualizations/`
- **Files**:
  - `posterior_predictive_overlay.png` - Data + posterior draws
  - `residual_vs_x.png` - Residual diagnostic
  - `residual_vs_fitted.png` - Homoscedasticity check
  - `trace_plots.png` - Convergence check
  - `pairs_plot.png` - Parameter correlations
  - `ppc_test_statistics.png` - PPC results
  - `loo_comparison.png` - Model comparison visual

### Code
- **Directory**: `code/`
- **Files**:
  - `model_logarithmic.stan` or `model_logarithmic.py` (PyMC)
  - `model_piecewise.stan` or `model_piecewise.py`
  - `model_asymptotic.stan` or `model_asymptotic.py`
  - `fit_all_models.py` or `fit_all_models.R` - Master script
  - `posterior_analysis.py` or `posterior_analysis.R` - PPCs, LOO, diagnostics
  - `sensitivity_analysis.py` or `sensitivity_analysis.R`

---

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Initial fits (4 models) | 2-4 hours | 2-4h |
| Phase 2: Robust variants (2 models) | 1-2 hours | 3-6h |
| Phase 3: Deep dive (2 models) | 3-6 hours | 6-12h |
| Phase 4: Final report & visuals | 1-2 hours | 7-14h |

**Fast track** (if needed): Fit Log/Normal and Piecewise-Cont/Normal only → 1-2 hours for initial comparison

---

## Success Criteria

### Minimum Acceptable

✓ All models converge (R-hat < 1.05)
✓ ESS > 100 for all parameters
✓ Divergence rate < 1%
✓ At least one model passes all PPCs (no p < 0.05)
✓ LOO diagnostics acceptable (Pareto k < 0.7 for most points)

### Ideal

✓ All diagnostics perfect (R-hat < 1.01, ESS > 400, 0 divergences)
✓ Clear winner (Δ ELPD > 10)
✓ All PPCs pass (0.05 < p < 0.95)
✓ Prior-insensitive (robust to prior choices)
✓ Outlier-insensitive (excluding x=31.5 doesn't change conclusions)

### Failure (Abandon Approach)

✗ Persistent divergences (>5%) despite reparameterization
✗ Non-convergence (R-hat > 1.05 after 10k iterations)
✗ Multiple Pareto k > 0.7 across all models
✗ Systematic PPC failures (p < 0.01) across all models
✗ Parameters at prior boundaries or physically impossible

---

## Red Flags & Contingencies

| Red Flag | Possible Cause | Action |
|----------|----------------|--------|
| Piecewise changepoint at boundary (3 or 12) | Prior range too narrow | Expand to [1, 20], refit |
| Asymptotic `c` → 0 | No saturation, linear better | Fit linear model for comparison |
| Asymptotic `a` < max(Y) | Model can't reach data | Reject asymptotic form |
| Student-t ν → 1 (Cauchy) | Extreme outliers | Investigate data quality |
| All models similar LOO | Underfitting | Add complexity or non-parametric |
| High parameter correlation (>0.95) | Unidentifiable | Reparameterize or simplify |

---

## Final Notes

**Philosophy**: These models are hypotheses, not truths. I expect them to be wrong in different ways. The goal is to discover how they fail and learn about the data-generating process.

**Honesty commitment**: I will report all models fit, not cherry-pick. If results are ambiguous, I will say so. If all models fail, I will recommend switching approaches.

**Bayesian advantage**: Full uncertainty quantification, interpretable parameters, principled model comparison, and natural handling of small sample size (n=27).

**Limitations acknowledged**:
- n=27 is small; posteriors will be wide
- One influential outlier (x=31.5) may drive results
- Extrapolation beyond x=31.5 is speculative
- Parametric forms are simplifications; reality is more complex

---

**Next Step**: Implement models in Stan or PyMC, begin Phase 1 fitting.
