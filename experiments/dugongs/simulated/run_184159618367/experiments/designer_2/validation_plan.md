# Validation & Falsification Plan
## Designer 2 - Flexible & Adaptive Models

**Purpose**: Systematic validation of three Bayesian model classes with explicit falsification criteria at each stage.

---

## Validation Philosophy

**Core Principle**: Models must prove they work, not be assumed to work.

**Mindset**: Each validation stage is an opportunity for the model to fail. Failure is information, not setback.

**Success Definition**: Not "model converged" but "model genuinely explains the data and passes adversarial tests."

---

## Validation Pipeline (4 Stages)

### Stage 1: Prior Predictive Checks (Before Seeing Data)
**Goal**: Ensure priors encode reasonable beliefs and don't constrain implausibly.

#### Procedure
1. Sample from prior: Generate θ ~ p(θ) for 1000 draws
2. Generate synthetic data: Y_prior ~ p(Y | θ, x)
3. Check if prior samples produce plausible data ranges

#### Pass Criteria
- ✓ 95% of prior predictive Y ∈ [0, 5] (observed range ± buffer)
- ✓ Prior captures saturation patterns (not all monotone increasing)
- ✓ Prior allows for both sharp and smooth transitions (Model 1 & 3)

#### Fail Actions
- If Y_prior has extreme values (>10 or <0): Tighten priors
- If all prior samples are linear: Adjust to allow curvature
- If prior is too restrictive: Use more diffuse priors

#### Implementation
```python
# For each model, generate prior predictive samples
prior_samples = model.sample_prior(n_draws=1000)

# Plot prior predictive distribution
plt.figure(figsize=(10, 6))
for i in range(100):
    plt.plot(x_grid, Y_prior[i, :], alpha=0.1, color='gray')
plt.scatter(x_obs, Y_obs, c='red', s=50, label='Observed (for reference)')
plt.title('Prior Predictive Check')
plt.xlabel('x')
plt.ylabel('Y')
plt.legend()
```

---

### Stage 2: MCMC Diagnostics (During Sampling)
**Goal**: Ensure posterior samples are valid (not garbage from failed sampling).

#### Convergence Diagnostics

| Diagnostic | Threshold | Action if Failed |
|------------|-----------|------------------|
| R-hat | < 1.01 | Increase warmup, tune adapt_delta |
| ESS (bulk) | > 400 | Longer chains or reparameterization |
| ESS (tail) | > 400 | Check for heavy tails, consider transform |
| Divergences | < 5% | Increase adapt_delta to 0.95-0.99 |
| Max treedepth | < 5% | Increase max_treedepth to 15 |

#### Parameter-Specific Checks

**Model 1 (Change-Point)**:
- Check τ (breakpoint) mixing: Should not get stuck at boundary
- Check β₁, β₂ (slopes) for extreme correlations (ρ < 0.9)
- Monitor energy diagnostics (BFMI > 0.2)

**Model 2 (B-Spline)**:
- Check all β coefficients converge (easier due to linearity)
- Check τ (smoothness) is not hitting zero (over-regularization)

**Model 3 (Mixture)**:
- Check γ₀, γ₁ (gating) mixing (nonlinear, may be slower)
- Check for label switching (unlikely with directional constraints)
- Monitor expert parameters (β₀, β₁, α) for identifiability

#### Procedure
```python
# 1. Run diagnostics
fit.diagnose()

# 2. Check summary statistics
summary = fit.summary()
print(summary[['R_hat', 'N_Eff']])

# 3. Visualize traces
import arviz as az
idata = az.from_cmdstanpy(fit)
az.plot_trace(idata, var_names=['tau', 'beta1', 'beta2', 'sigma'])

# 4. Check pairs plot for correlations
az.plot_pair(idata, var_names=['tau', 'beta1', 'beta2'],
             divergences=True, kind='hexbin')
```

#### Fail Actions
- If R-hat > 1.01 after tuning: **REJECT MODEL** (geometry fundamentally flawed)
- If persistent divergences (>5%): Reparameterize or switch to simpler model
- If multimodal posterior: Check for label switching or prior conflict

---

### Stage 3: Posterior Predictive Checks (After Sampling)
**Goal**: Ensure model captures data features and residuals are well-behaved.

#### Visual Checks

1. **Overlay Plot**: Y_obs vs. Y_rep samples
   - Should see observed data within posterior predictive cloud
   - Check for systematic misses (e.g., all high-x points outside CI)

2. **Residual Plot**: (Y_obs - Ŷ) vs. x
   - Should be centered at zero, no patterns
   - Check for heteroscedasticity (variance changing with x)

3. **QQ-Plot**: Residuals vs. Normal quantiles
   - Should be linear if Gaussian likelihood is appropriate
   - Deviations indicate need for Student-t or transformation

4. **Replicate Checks**: For 6 x-values with replicates
   - Observed spread should match posterior predictive spread
   - If mismatched: Variance model is wrong

#### Quantitative Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Coverage (95% CI) | 90-98% | Calibrated uncertainty |
| Posterior Predictive R² | > 0.85 | Good fit |
| RMSE | < 0.12 | Match pure error estimate |
| MAE | < 0.09 | Robust accuracy |

#### Test Statistics (Bayesian p-values)

Compute test statistics for each Y_rep draw:
- T1: max(Y) - min(Y) [range]
- T2: Correlation(Y, x) [overall association]
- T3: Variance(Y | x > 10) [plateau variance]
- T4: Mean(Y | x < 5) [low-x level]

Bayesian p-value: p = P(T(Y_rep) > T(Y_obs))
- Good calibration: p ∈ [0.1, 0.9]
- Extreme p (< 0.05 or > 0.95): Model missing key feature

#### Procedure
```python
# 1. Extract posterior predictive samples
Y_rep = fit.stan_variable('Y_rep')  # Shape: (n_draws, N)

# 2. Coverage check
Y_lower = np.percentile(Y_rep, 2.5, axis=0)
Y_upper = np.percentile(Y_rep, 97.5, axis=0)
coverage = np.mean((Y_obs >= Y_lower) & (Y_obs <= Y_upper))
print(f"Coverage: {coverage:.2%}")

# 3. Residual diagnostics
Y_pred = Y_rep.mean(axis=0)
residuals = Y_obs - Y_pred

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(Y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Y')
plt.ylabel('Residual')
plt.title('Residual Plot')

plt.subplot(1, 3, 2)
from scipy.stats import probplot
probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q Plot')

plt.subplot(1, 3, 3)
plt.hist(residuals, bins=15, density=True, alpha=0.6)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
plt.plot(x_norm, norm.pdf(x_norm, 0, residuals.std()), 'r-', linewidth=2)
plt.xlabel('Residual')
plt.ylabel('Density')
plt.title('Residual Distribution')

# 4. Test statistics
T_obs = np.max(Y_obs) - np.min(Y_obs)
T_rep = np.max(Y_rep, axis=1) - np.min(Y_rep, axis=1)
p_val = np.mean(T_rep > T_obs)
print(f"Bayesian p-value (range): {p_val:.3f}")
```

#### Fail Actions
- Coverage < 85%: Model uncertainty underestimated → Check variance model
- Coverage > 98%: Model too conservative → Priors too wide
- Systematic residual patterns: Wrong functional form → Try alternative model
- Extreme Bayesian p-values: Model missing feature → Add complexity or transform

---

### Stage 4: Out-of-Sample Validation (LOO-CV)
**Goal**: Assess predictive performance on held-out data (approximate via LOO).

#### LOO-ELPD (Expected Log Pointwise Predictive Density)

**Interpretation**:
- Higher ELPD = better predictive accuracy
- ΔELPD > 2×SE: Meaningful difference between models
- ΔELPD < 2×SE: Models statistically equivalent

**Comparison**:
```
Model A: ELPD = 50 ± 5
Model B: ELPD = 48 ± 6
ΔELPD = 2 ± 7.8 → No significant difference
```

#### Pareto-k Diagnostics

| Pareto-k | Interpretation | Action |
|----------|----------------|--------|
| < 0.5 | Good | LOO reliable |
| 0.5-0.7 | OK | LOO somewhat reliable |
| 0.7-1.0 | Bad | LOO unreliable, use K-fold CV |
| > 1.0 | Very bad | Point is highly influential |

**Red Flag**: If x=31.5 (most extreme point) has k > 0.7 → Model struggles with extrapolation

#### Model Selection Decision Tree

```
1. Check Pareto-k diagnostics
   ├─ If any k > 0.7: Run 5-fold CV instead of LOO
   └─ If all k < 0.7: Proceed with LOO

2. Compare LOO-ELPD across models
   ├─ If ΔELPD > 2*SE: Clear winner → Recommend best model
   └─ If ΔELPD < 2*SE: Statistically tied → Prefer simpler/interpretable

3. Check best model falsification criteria
   ├─ Change-Point: Is SD(τ) < 5? Breakpoint not at boundary?
   ├─ B-Spline: No wild oscillations? Reasonable smoothness?
   └─ Mixture: Well-defined τ_eff? No identifiability issues?

4. Make recommendation
   ├─ If passes all checks: Confidently recommend
   ├─ If marginal: Report with caveats
   └─ If fails: REJECT, try alternative (GP, transform, etc.)
```

#### Procedure
```python
# 1. Compute LOO for each model
import arviz as az

loo_results = {}
for name, fit in fits.items():
    log_lik = fit.stan_variable('log_lik')
    idata = az.from_cmdstanpy(fit, log_likelihood={'Y': log_lik})
    loo = az.loo(idata, pointwise=True)
    loo_results[name] = loo

    print(f"\n{name}:")
    print(f"  LOO-ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
    print(f"  Pareto-k > 0.7: {np.sum(loo.pareto_k > 0.7)}")

# 2. Compare models
compare = az.compare(loo_results)
print("\nModel Comparison:")
print(compare)

# 3. Visualize Pareto-k
az.plot_khat(loo_results['Change-Point'], show_bins=True)
plt.title('Pareto-k Diagnostics')
```

#### Fail Actions
- All models have LOO-R² < 0.75: Return to EDA, fundamental misspecification
- Pareto-k > 0.7 for many points: Model unstable, try robust likelihood (Student-t)
- Best model fails falsification: Try alternative model class (GP, transformations)

---

## Model-Specific Falsification Criteria

### Model 1: Change-Point Regression

**Reject if**:
1. SD(τ) > 5 → No clear breakpoint
2. τ median < 3 or > 25 → Breakpoint at boundary
3. P(|β₁ - β₂| < 0.02) > 0.5 → Slopes not distinct
4. LOO-ELPD < Spline by >4 → Overfitting
5. Divergences > 5% after adapt_delta=0.95 → Geometry issues

**Diagnostic Plot**:
```python
# Plot posterior distribution of breakpoint
tau_samples = fit.stan_variable('tau')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(tau_samples, bins=30, density=True, alpha=0.6)
plt.axvline(np.median(tau_samples), color='red', linestyle='--', label='Median')
plt.xlabel('Breakpoint τ')
plt.ylabel('Density')
plt.title(f'Posterior: τ ~ {np.median(tau_samples):.1f} ± {np.std(tau_samples):.1f}')
plt.legend()

plt.subplot(1, 2, 2)
# Plot piecewise function with uncertainty
x_grid = np.linspace(x_obs.min(), x_obs.max(), 200)
mu_samples = []
for i in range(100):
    tau_i = tau_samples[i]
    beta0_i = fit.stan_variable('beta0')[i]
    beta1_i = fit.stan_variable('beta1')[i]
    beta2_i = fit.stan_variable('beta2')[i]

    mu_i = np.where(x_grid <= tau_i,
                    beta0_i + beta1_i * x_grid,
                    beta0_i + beta1_i * tau_i + beta2_i * (x_grid - tau_i))
    plt.plot(x_grid, mu_i, 'b-', alpha=0.05)

plt.scatter(x_obs, Y_obs, c='red', s=50, zorder=5)
plt.xlabel('x')
plt.ylabel('Y')
plt.title('Posterior Predictive (100 draws)')
```

### Model 2: B-Spline Regression

**Reject if**:
1. Wild oscillations between data points (overfitting)
2. R² < 0.80 (underfitting, too smooth)
3. Predictions change drastically with different knot placements
4. Extrapolation goes wild (|Ŷ(x=35)| > 5)
5. Convergence fails (unlikely, but check)

**Diagnostic Plot**:
```python
# Sensitivity to number of knots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, n_knots in enumerate([3, 5, 7]):
    fit_k, ppc_k, loo_k = fit_spline_model(x_obs, Y_obs, n_knots=n_knots)

    ax = axes[idx]
    ax.scatter(x_obs, Y_obs, c='black', s=50, label='Observed')
    ax.plot(x_obs, ppc_k['Y_pred_mean'], 'b-', linewidth=2, label='Fit')
    ax.fill_between(x_obs, ppc_k['Y_lower'], ppc_k['Y_upper'],
                    alpha=0.3, color='blue')
    ax.set_title(f'K={n_knots} knots (R²={ppc_k["r2"]:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('Y')
    ax.legend()
```

### Model 3: Mixture-of-Experts

**Reject if**:
1. π(x) ≈ 0.5 for all x → Gating uninformative
2. Posterior corr(β₁, α) > 0.95 → Identifiability issues
3. SD(τ_eff) > 10 → Transition unconstrained
4. LOO-ELPD worse than both single experts → Overfitting
5. Divergences persist → Nonlinear geometry problems

**Diagnostic Plot**:
```python
# Plot gating function evolution
gamma0_samples = fit.stan_variable('gamma0')
gamma1_samples = fit.stan_variable('gamma1')

x_grid = np.linspace(x_obs.min(), x_obs.max(), 200)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(100):
    pi_x = 1 / (1 + np.exp(-(gamma0_samples[i] + gamma1_samples[i] * x_grid)))
    plt.plot(x_grid, pi_x, 'b-', alpha=0.05)

plt.axhline(0.5, color='red', linestyle='--', label='π=0.5 (transition)')
plt.xlabel('x')
plt.ylabel('π(x) = P(Linear Expert)')
plt.title('Gating Function (100 posterior draws)')
plt.legend()

plt.subplot(1, 2, 2)
tau_eff_samples = fit.stan_variable('tau_eff')
plt.hist(tau_eff_samples, bins=30, density=True, alpha=0.6)
plt.axvline(np.median(tau_eff_samples), color='red', linestyle='--')
plt.xlabel('Effective Breakpoint τ_eff')
plt.ylabel('Density')
plt.title(f'τ_eff ~ {np.median(tau_eff_samples):.1f} ± {np.std(tau_eff_samples):.1f}')
```

---

## Stress Tests & Adversarial Checks

### Stress Test 1: Synthetic Data Recovery

**Purpose**: Verify model can recover known parameters.

**Procedure**:
1. Generate synthetic data from known parameters (e.g., τ=10, β₁=0.08, β₂=0)
2. Fit model to synthetic data
3. Check if true parameters fall within 95% posterior credible intervals

**Pass Criterion**: >95% coverage across 10 synthetic datasets

**Implementation**:
```python
def synthetic_recovery_test(model_func, true_params, n_reps=10):
    """Test if model recovers known parameters."""
    coverage = []

    for rep in range(n_reps):
        # Generate synthetic data
        x_syn = x_obs  # Use same x values
        Y_syn = generate_data(x_syn, true_params) + np.random.normal(0, 0.1, len(x_syn))

        # Fit model
        fit = model_func(x_syn, Y_syn)

        # Check coverage
        for param_name, true_val in true_params.items():
            samples = fit.stan_variable(param_name)
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
            coverage.append((ci_lower <= true_val <= ci_upper))

    recovery_rate = np.mean(coverage)
    print(f"Parameter recovery rate: {recovery_rate:.1%}")

    if recovery_rate < 0.90:
        print("⚠ WARNING: Poor parameter recovery - model may be misspecified")

    return recovery_rate
```

### Stress Test 2: Leave-One-Out Sensitivity

**Purpose**: Check if model is sensitive to single observations.

**Procedure**:
1. Fit model to full dataset → Get baseline LOO-ELPD
2. For each observation i:
   - Fit model excluding observation i
   - Compare posterior parameters to baseline
3. Flag observations causing large posterior shifts (Cook's distance analog)

**Red Flag**: If removing observation near breakpoint drastically changes τ

**Implementation**:
```python
def leave_one_out_sensitivity(model_func, x_obs, Y_obs):
    """Check sensitivity to individual observations."""
    n = len(x_obs)

    # Baseline fit
    fit_full = model_func(x_obs, Y_obs)
    tau_full = fit_full.stan_variable('tau')
    tau_full_median = np.median(tau_full)

    # Leave-one-out fits
    tau_deltas = []

    for i in range(n):
        # Exclude observation i
        x_loo = np.delete(x_obs, i)
        Y_loo = np.delete(Y_obs, i)

        # Fit
        fit_loo = model_func(x_loo, Y_loo)
        tau_loo = fit_loo.stan_variable('tau')
        tau_loo_median = np.median(tau_loo)

        # Compute shift
        delta = tau_loo_median - tau_full_median
        tau_deltas.append(delta)

        if np.abs(delta) > 2:
            print(f"⚠ Observation {i} (x={x_obs[i]:.1f}) causes large shift: Δτ = {delta:.2f}")

    return tau_deltas
```

### Stress Test 3: Prior Sensitivity

**Purpose**: Ensure conclusions are not driven by prior choice.

**Procedure**:
1. Fit with "default" priors (as specified)
2. Fit with diffuse priors (SD × 3)
3. Fit with informative priors (SD / 2)
4. Compare posterior medians and 95% CIs

**Pass Criterion**: Posterior medians differ by < 10% across prior choices

**Red Flag**: If conclusions reverse with different priors → Data is weak

**Implementation**:
```python
def prior_sensitivity_analysis(model_func, x_obs, Y_obs):
    """Test sensitivity to prior specification."""
    prior_specs = {
        'Default': {'tau_sd': 2.0, 'beta1_sd': 0.03},
        'Diffuse': {'tau_sd': 6.0, 'beta1_sd': 0.09},
        'Informative': {'tau_sd': 1.0, 'beta1_sd': 0.015}
    }

    results = {}

    for name, priors in prior_specs.items():
        fit = model_func(x_obs, Y_obs, priors=priors)
        tau = fit.stan_variable('tau')
        results[name] = {
            'median': np.median(tau),
            'ci_lower': np.percentile(tau, 2.5),
            'ci_upper': np.percentile(tau, 97.5)
        }

    # Check sensitivity
    medians = [r['median'] for r in results.values()]
    max_diff = np.max(medians) - np.min(medians)

    print("\nPrior Sensitivity:")
    for name, res in results.items():
        print(f"  {name}: τ = {res['median']:.2f} [{res['ci_lower']:.2f}, {res['ci_upper']:.2f}]")

    if max_diff > 2:
        print(f"⚠ WARNING: Large sensitivity to priors (Δτ = {max_diff:.2f})")
        print("   → Data may be insufficient to overcome prior")

    return results
```

---

## Final Decision Framework

### Step 1: Initial Screening
- Run all three models with default settings
- Check convergence: If any fail → Debug or reject
- Run posterior predictive checks: If any fail basic checks → Reject or modify

### Step 2: Performance Comparison
- Compare LOO-ELPD across models
- Identify best model(s) (within 2×SE)
- Check Pareto-k diagnostics: If issues → Run K-fold CV

### Step 3: Falsification Checks
- For each candidate model, check specific falsification criteria
- If model fails criteria → Reject and document reason
- If all models fail → Pivot to alternative approaches (GP, transforms)

### Step 4: Sensitivity & Robustness
- Run stress tests on surviving models
- Check prior sensitivity, LOO sensitivity, synthetic recovery
- If model is fragile → Downgrade confidence or reject

### Step 5: Final Recommendation

**Recommendation Format**:

```
PRIMARY MODEL: [Model Name]
- LOO-ELPD: [value] ± [SE]
- Posterior Predictive R²: [value]
- Key Parameters: [summary with 95% CIs]
- Falsification Status: PASSED [list checks]
- Robustness: [summary of stress tests]
- Confidence Level: HIGH / MODERATE / LOW
- Caveats: [any limitations or concerns]

ALTERNATIVE MODELS:
- [If models are statistically tied, list alternatives]

MODELS REJECTED:
- [Model Name]: Failed [specific criterion]
```

**Confidence Levels**:
- **HIGH**: Passes all checks, LOO-ELPD > alternatives by >2×SE, robust to sensitivity
- **MODERATE**: Passes most checks, comparable LOO-ELPD, some sensitivity concerns
- **LOW**: Passes minimum criteria, but marginal performance or robustness issues

---

## Red Flags Triggering Major Pivot

**STOP and reconsider everything if**:

1. **All models fail convergence** → Fundamental geometry issues
   - Action: Try simpler model classes, check data quality

2. **All models have Pareto-k > 0.7 for many points** → Specification issues
   - Action: Try Student-t likelihood, check for outliers

3. **All models have LOO-R² < 0.75** → Missing key features
   - Action: Return to EDA, consider transformations or additional covariates

4. **Best model fails falsification checks** → Wrong model class
   - Action: Pivot to GP, ensemble, or alternative approaches

5. **Results highly sensitive to priors** → Data insufficient
   - Action: Collect more data, use ensemble, or report high uncertainty

6. **Posterior conflicts with domain knowledge** → Misspecification or data issues
   - Action: Investigate data quality, check for measurement error

---

## Timeline

| Day | Activity | Output |
|-----|----------|--------|
| 1 | Prior predictive checks, initial fits | Converged posteriors or rejection |
| 2 | Posterior predictive checks, LOO-CV | Performance metrics, model ranking |
| 3 | Stress tests, sensitivity analysis | Robustness assessment |
| 4 | Final decision, documentation | Recommendation report |

**Total Time**: 4 days (assuming no major issues)

If issues arise: Add 1-2 days for debugging, reparameterization, or alternative models.

---

## Success Metrics

**Minimum Viable Success**:
- At least one model converges (R-hat < 1.01)
- At least one model achieves R² > 0.85
- At least one model passes falsification criteria

**Excellent Success**:
- All three models converge
- Clear winner emerges (LOO-ELPD > others by >2×SE)
- Winner passes all stress tests with high confidence

**Learning Success** (even if models fail):
- Discover why models fail (informative)
- Identify path forward (GP, transforms, etc.)
- Gain insight into data generation process

---

**Truth over task completion. Honest assessment over forced success.**

---

**File Location**: `/workspace/experiments/designer_2/validation_plan.md`
