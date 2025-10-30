# Implementation Guide - Designer 2 Models
## Step-by-Step Instructions for Model Fitting

**Quick Start:** Read this file first, then refer to detailed specifications in other documents.

---

## Document Hierarchy

```
designer_2/
│
├── README.md                        ← START HERE (index, 3-min read)
├── IMPLEMENTATION_GUIDE.md          ← THIS FILE (practical steps, 5-min read)
├── design_summary.md                ← Quick reference (10-min read)
├── model_comparison_matrix.md       ← Side-by-side comparison (15-min read)
└── proposed_models.md               ← Full specification (30-min read, reference)
```

---

## Quick Model Overview

### Three Models, One Data Generation Question

**The Core Question:** What process generated 8.45× growth with 0.989 autocorrelation?

```
                         COMPETING HYPOTHESES
                                 |
        ┌────────────────────────┼────────────────────────┐
        |                        |                        |
    Model 1                  Model 2                  Model 3
 State-Space              Changepoint          Gaussian Process
        |                        |                        |
 "It's a smooth          "There was a          "We don't know
  random walk             discrete break         the function,
  with drift"             at t ≈ 0.3"           let data speak"
        |                        |                        |
   Tests: AR(1)          Tests: Regime shift   Tests: Nonparametric
  structure with         with level/slope        smooth alternative
  small innovations           changes           to both parametric
```

**Decision Rule:** Fit all three, use LOO-ELPD to select, validate with diagnostics.

---

## Implementation Workflow

### Phase 0: Setup (15 minutes)

**Install Required Packages:**
```bash
# Python
pip install pystan arviz pandas numpy matplotlib seaborn

# R (alternative)
install.packages(c("rstan", "loo", "bayesplot", "tidyverse"))
```

**Verify Data:**
```python
import pandas as pd
data = pd.read_csv('/workspace/data/data_designer_2.csv')
print(data.shape)  # Should be (40, 2)
print(data.describe())
```

### Phase 1: Implement Stan Models (2-3 hours)

**Step 1.1: Create Model Files**

Copy Stan code skeletons from `proposed_models.md` Appendix:
- Save as `model_1_state_space.stan`
- Save as `model_2_changepoint.stan`
- Save as `model_3_gaussian_process.stan`

**Step 1.2: Create Fitting Script**

```python
# fit_models.py
import pystan
import pandas as pd
import pickle

# Load data
data = pd.read_csv('/workspace/data/data_designer_2.csv')
stan_data = {
    'N': len(data),
    'C': data['C'].values.astype(int),
    'year': data['year'].values
}

# Compile and fit Model 1
model_1 = pystan.StanModel(file='model_1_state_space.stan')
fit_1 = model_1.sampling(data=stan_data,
                          iter=4000,
                          chains=4,
                          control={'adapt_delta': 0.95})

# Save results
with open('fit_1.pkl', 'wb') as f:
    pickle.dump(fit_1, f)

# Repeat for Models 2 and 3
# ...
```

**Step 1.3: Check Convergence**

```python
import arviz as az

# Convert to ArviZ InferenceData
idata_1 = az.from_pystan(posterior=fit_1)

# Convergence diagnostics
print(az.summary(idata_1, var_names=['delta', 'sigma_eta', 'phi']))
# Check: R-hat < 1.01, ESS > 400

# Visual checks
az.plot_trace(idata_1, var_names=['delta', 'sigma_eta', 'phi'])
az.plot_pair(idata_1, var_names=['delta', 'sigma_eta'])
```

**Expected Output:**
- Model 1: ~5-10 minutes, 8000 posterior samples
- Model 2: ~2-5 minutes, 8000 posterior samples
- Model 3: ~3-8 minutes, 8000 posterior samples

**If Convergence Fails:**
- Increase `adapt_delta` to 0.99
- Increase `max_treedepth` to 12
- Run longer: `iter=10000`
- Check for coding errors in Stan

---

### Phase 2: Model Comparison (1 hour)

**Step 2.1: Compute LOO-ELPD**

```python
# Compute LOO for all models
loo_1 = az.loo(idata_1, pointwise=True)
loo_2 = az.loo(idata_2, pointwise=True)
loo_3 = az.loo(idata_3, pointwise=True)

# Compare
loo_compare = az.compare({'Model_1': idata_1,
                          'Model_2': idata_2,
                          'Model_3': idata_3})
print(loo_compare)
```

**Interpretation:**
```
           elpd_loo  p_loo  elpd_diff   weight
Model_1     -148.2    5.3       0.0      0.82   ← Best model
Model_2     -156.7    6.1       8.5      0.15
Model_3     -153.4    7.8       5.2      0.03
```

**Decision Rules:**
- **elpd_diff > 10:** Clear winner (Model 1 here)
- **elpd_diff < 4:** Equivalent, use BMA
- **High p_loo (>10):** Possible overfitting

**Step 2.2: Posterior Predictive Checks**

```python
# Extract posterior predictive samples
C_rep_1 = fit_1.extract()['C_rep']  # Shape: (8000, 40)

# Compare to observed data
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Data vs predictive distribution
axes[0,0].hist(C_rep_1.flatten(), bins=50, alpha=0.5, label='Predicted')
axes[0,0].hist(data['C'], bins=20, alpha=0.5, label='Observed')
axes[0,0].legend()

# 2. Max value check
ppc_max = (C_rep_1.max(axis=1) > data['C'].max()).mean()
axes[0,1].hist(C_rep_1.max(axis=1), bins=50)
axes[0,1].axvline(data['C'].max(), color='red', label=f'Observed (p={ppc_max:.2f})')
axes[0,1].legend()

# 3. ACF check
from statsmodels.tsa.stattools import acf
acf_obs = acf(data['C'], nlags=10)
acf_rep = np.array([acf(C_rep_1[i], nlags=10) for i in range(100)])
axes[1,0].plot(acf_obs, 'ro-', label='Observed')
axes[1,0].plot(acf_rep.T, 'k-', alpha=0.1)
axes[1,0].legend()

# 4. Mean-variance relationship
plt.tight_layout()
plt.savefig('model_1_ppc.png', dpi=150)
```

**Red Flags:**
- Observed max >> all replicate maxes (right tail too light)
- Observed ACF >> all replicate ACFs (temporal structure missed)
- Systematic deviations in mean/variance

---

### Phase 3: Residual Diagnostics (1 hour)

**Step 3.1: Compute Residuals**

```python
# Extract posterior means
eta_mean = fit_1.extract()['eta'].mean(axis=0)  # Model 1
mu_mean = np.exp(eta_mean)

# Pearson residuals
phi_mean = fit_1.extract()['phi'].mean()
var_mean = mu_mean + mu_mean**2 / phi_mean
pearson_resid = (data['C'].values - mu_mean) / np.sqrt(var_mean)

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
plot_acf(pearson_resid, lags=20, ax=ax)
ax.set_title('ACF of Pearson Residuals (Model 1)')
plt.savefig('model_1_resid_acf.png', dpi=150)
```

**Target:** ACF(1) < 0.3, all lags < 0.5

**Step 3.2: One-Step-Ahead Predictions**

```python
# For Model 1 (State-Space), predict C_t from C_{t-1}
osa_predictions = []
osa_intervals = []

for t in range(1, len(data)):
    # Fit on data up to t-1
    stan_data_partial = {
        'N': t,
        'C': data['C'].values[:t].astype(int),
        'year': data['year'].values[:t]
    }
    fit_partial = model_1.sampling(data=stan_data_partial, iter=2000, chains=2)

    # Predict t (using state evolution)
    eta_t = fit_partial.extract()['eta'][:, -1]  # Last state
    delta = fit_partial.extract()['delta']
    phi = fit_partial.extract()['phi']

    eta_next = eta_t + delta + beta * data['year'].iloc[t]
    mu_next = np.exp(eta_next)

    # Sample from predictive distribution
    C_next_pred = np.random.negative_binomial(phi, phi/(phi+mu_next))

    osa_predictions.append(np.median(C_next_pred))
    osa_intervals.append(np.percentile(C_next_pred, [10, 90]))

# Compute coverage
coverage_80 = np.mean([
    osa_intervals[i][0] <= data['C'].iloc[i+1] <= osa_intervals[i][1]
    for i in range(len(osa_intervals))
])
print(f'80% Interval Coverage: {coverage_80:.1%}')  # Target: 75-85%
```

**Target:** 80% coverage = 75-85%, 95% coverage = 90-98%

---

### Phase 4: Parameter Interpretation (30 minutes)

**Model 1 (if winner):**

```python
# Extract parameters
samples = fit_1.extract()

delta = samples['delta']
sigma_eta = samples['sigma_eta']
phi = samples['phi']

# Summarize
print("Drift (δ):")
print(f"  Mean: {delta.mean():.3f}")
print(f"  95% CI: [{np.percentile(delta, 2.5):.3f}, {np.percentile(delta, 97.5):.3f}]")
print(f"  Interpretation: {(np.exp(delta.mean())-1)*100:.1f}% growth per period")

print("\nInnovation SD (σ_η):")
print(f"  Mean: {sigma_eta.mean():.3f}")
print(f"  Interpretation: State fluctuates ±{1.96*sigma_eta.mean():.2f} (95% range)")

print("\nDispersion (φ):")
print(f"  Mean: {phi.mean():.1f}")
print(f"  Interpretation: Var/Mean ≈ {1 + np.mean(data['C'])/phi.mean():.1f} after temporal correlation")
```

**Expected Output (Model 1):**
```
Drift (δ):
  Mean: 0.061
  95% CI: [0.042, 0.078]
  Interpretation: 6.3% growth per period

Innovation SD (σ_η):
  Mean: 0.076
  Interpretation: State fluctuates ±0.15 (95% range)

Dispersion (φ):
  Mean: 14.3
  Interpretation: Var/Mean ≈ 8.7 after temporal correlation
```

**Scientific Interpretation:**
"The data are best explained by a random walk model with positive drift. The system grows at approximately 6% per period in log-space (exponential growth), with small random fluctuations (σ_η ≈ 0.08) around this trend. After accounting for temporal correlation, the residual overdispersion is moderate (φ ≈ 14), much less than the unconditional estimate (67.99). This suggests most apparent overdispersion is due to autocorrelation, not intrinsic count variability."

---

### Phase 5: Visualization (1 hour)

**Create Publication-Quality Figures:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Data + Fitted Trend
ax1 = fig.add_subplot(gs[0, :2])
eta_samples = fit_1.extract()['eta']  # (8000, 40)
mu_samples = np.exp(eta_samples)

# Plot posterior median and credible bands
mu_median = np.median(mu_samples, axis=0)
mu_lower = np.percentile(mu_samples, 2.5, axis=0)
mu_upper = np.percentile(mu_samples, 97.5, axis=0)

ax1.scatter(data['year'], data['C'], color='black', s=50, zorder=3, label='Observed')
ax1.plot(data['year'], mu_median, 'r-', linewidth=2, label='Posterior Median')
ax1.fill_between(data['year'], mu_lower, mu_upper, alpha=0.3, color='red', label='95% CI')
ax1.set_xlabel('Year (standardized)')
ax1.set_ylabel('Count')
ax1.set_title('Model 1: State-Space Fit')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Posterior Distributions
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(delta, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax2.axvline(delta.mean(), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('δ (drift)')
ax2.set_title(f'Posterior: δ = {delta.mean():.3f}')

# 3. Residuals vs Fitted
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(mu_median, pearson_resid, alpha=0.6)
ax3.axhline(0, color='red', linestyle='--')
ax3.set_xlabel('Fitted μ')
ax3.set_ylabel('Pearson Residuals')
ax3.set_title('Residuals vs Fitted')

# 4. ACF of Residuals
ax4 = fig.add_subplot(gs[1, 1])
plot_acf(pearson_resid, lags=15, ax=ax4)
ax4.set_title('ACF of Residuals')

# 5. Q-Q Plot of Residuals
ax5 = fig.add_subplot(gs[1, 2])
from scipy import stats
stats.probplot(pearson_resid, dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot')

# 6. Posterior Predictive Check
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(C_rep_1.flatten(), bins=50, alpha=0.5, density=True, label='Posterior Predictive')
ax6.hist(data['C'], bins=20, alpha=0.5, density=True, label='Observed')
ax6.legend()
ax6.set_xlabel('Count')
ax6.set_title('Posterior Predictive Check')

# 7. One-Step-Ahead Predictions
ax7 = fig.add_subplot(gs[2, 1:])
ax7.scatter(range(1, len(data)), data['C'].iloc[1:], color='black', label='Observed', s=50)
ax7.plot(range(1, len(data)), osa_predictions, 'ro-', label='One-Step-Ahead Predictions')
for i, (low, high) in enumerate(osa_intervals):
    ax7.plot([i+1, i+1], [low, high], 'r-', alpha=0.3)
ax7.set_xlabel('Time Index')
ax7.set_ylabel('Count')
ax7.set_title(f'One-Step-Ahead Predictions (Coverage: {coverage_80:.0%})')
ax7.legend()

plt.savefig('model_1_comprehensive_diagnostics.png', dpi=150, bbox_inches='tight')
```

---

### Phase 6: Model Variants (if needed)

**When to Try Variants:**
- If ACF(1) > 0.5: Try Variant 1b (stochastic drift)
- If β parameter is important: Try Variant 1a (no time trend)
- If Model 2 is close (ΔLOO < 6): Try smooth changepoint (Variant 2a)
- If all models fail: Try alternatives in `proposed_models.md`

---

## Troubleshooting Guide

### Problem: Divergent Transitions

**Symptoms:** Stan reports "X divergent transitions after warmup"

**Solutions:**
1. Increase `adapt_delta`:
   ```python
   control={'adapt_delta': 0.99, 'max_treedepth': 12}
   ```
2. Check non-centered parameterization (already in skeletons)
3. Add tighter priors on problematic parameters
4. Reparameterize: e.g., log-transform σ_eta

### Problem: Low ESS (Effective Sample Size)

**Symptoms:** ESS < 100 for key parameters

**Solutions:**
1. Run longer: `iter=10000`
2. More chains: `chains=8`
3. Check for multimodality (plot pairs)
4. May indicate fundamental model issue (check diagnostics)

### Problem: R-hat > 1.01

**Symptoms:** Chains haven't converged

**Solutions:**
1. Run longer (warmup may be insufficient)
2. Check initialization: May be starting in bad regions
3. Verify Stan code (syntax errors can cause weird behavior)
4. Check for label switching (especially Model 2)

### Problem: Poor Residual ACF (>0.5)

**Symptoms:** Model not capturing temporal structure

**Solutions:**
1. If Model 1: Try Variant 1b (stochastic drift)
2. If Model 2: Temporal structure not explained by changepoint alone
3. Consider hybrid: Changepoint + AR structure
4. May need ARMA instead of AR(1)

### Problem: Poor Coverage (<70%)

**Symptoms:** Prediction intervals too narrow or too wide

**Solutions:**
1. Check φ parameter: Too small → intervals too narrow
2. Check innovation variance: Too small → overconfident
3. May indicate model misspecification
4. Try robustified predictive: e.g., Student-t instead of NegBin

---

## Expected Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 0. Setup | 15 min | 0:15 |
| 1. Implement & Fit | 2-3 hours | 2:15-3:15 |
| 2. Model Comparison | 1 hour | 3:15-4:15 |
| 3. Diagnostics | 1 hour | 4:15-5:15 |
| 4. Interpretation | 30 min | 4:45-5:45 |
| 5. Visualization | 1 hour | 5:45-6:45 |
| **Total** | **5-7 hours** | - |

**Variants (if needed):** +2-3 hours per variant

---

## Success Criteria Checklist

### Computational
- [ ] All R-hat < 1.01
- [ ] All ESS > 400
- [ ] Divergent transitions < 1%
- [ ] Visual inspection: Traceplots show mixing

### Statistical
- [ ] LOO comparison completed
- [ ] Residual ACF(1) < 0.3
- [ ] One-step-ahead coverage: 75-85% (80% interval)
- [ ] Posterior predictive p-values: 0.05-0.95

### Substantive
- [ ] Parameters have plausible values (see expected ranges)
- [ ] Scientific interpretation makes sense
- [ ] Predictions align with observed data patterns
- [ ] Limitations are documented

### Reporting
- [ ] Model comparison table
- [ ] Parameter summaries with interpretation
- [ ] Comprehensive diagnostic plots
- [ ] Discussion of assumptions and caveats

---

## Final Deliverables

### Code
- `model_1_state_space.stan`
- `model_2_changepoint.stan`
- `model_3_gaussian_process.stan`
- `fit_models.py` (or `.R`)
- `diagnostics.py`

### Results
- `loo_comparison.csv`
- `posterior_summaries_model_X.csv`
- `model_X_diagnostics.png` (comprehensive figure)
- `one_step_ahead_predictions.csv`

### Report
- `modeling_report.md` with:
  - Executive summary (which model won, why)
  - Model comparison table
  - Parameter interpretations
  - Diagnostic results
  - Limitations and future work

---

## Quick Reference: What Good Results Look Like

### Model 1 (State-Space) Success:
```
LOO-ELPD: -148 to -152
δ: 0.04-0.08 (4-8% growth per period)
σ_η: 0.05-0.12 (small innovations)
φ: 10-20 (moderate overdispersion)
ACF(1) residuals: < 0.3
Coverage: 78-84%
Interpretation: "Random walk with drift captures temporal structure"
```

### Model 2 (Changepoint) Success:
```
LOO-ELPD: -150 to -158
τ: 0.2-0.4 (changepoint well-identified)
β_2: 0.4-1.2 (significant level shift)
β_3: 0.2-0.8 (significant slope change)
φ: 12-25
ACF(1) residuals: < 0.4 (some residual correlation acceptable)
Interpretation: "Discrete regime shift at year ≈ 0.3"
```

### Model 3 (GP) Success:
```
LOO-ELPD: -148 to -156
ℓ: 0.3-1.0 (neither too smooth nor too wiggly)
α: 0.3-0.8 (moderate function variance)
φ: 8-18
ACF(1) residuals: < 0.3
Interpretation: "Smooth nonparametric trend fits best"
```

---

## Help and Support

**If stuck:** Refer back to detailed specifications in:
- `proposed_models.md` (full model details, priors, Stan code)
- `model_comparison_matrix.md` (side-by-side comparison)
- `design_summary.md` (quick reference, decision trees)

**Common Issues:**
- Computational → See Troubleshooting section above
- Statistical → See Falsification Criteria in `proposed_models.md`
- Substantive → See Scientific Interpretation sections

**Remember:**
- Success = Finding truth, not completing tasks
- Failure of a model is scientific progress
- Uncertainty is honest, not weakness

---

**Prepared by:** Designer 2 (Temporal Structure Specialist)
**Last Updated:** 2025-10-29
**Status:** READY TO IMPLEMENT
