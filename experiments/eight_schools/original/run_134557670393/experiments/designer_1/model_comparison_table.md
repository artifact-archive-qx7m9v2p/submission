# Model Comparison: Mathematical Specifications

**Quick reference for comparing the three hierarchical model classes**

---

## Side-by-Side Comparison

| Component | Model 1: Adaptive | Model 2: Robust | Model 3: Informative |
|-----------|------------------|-----------------|---------------------|
| **Likelihood** | y_i ~ Normal(θ_i, σ_i²) | y_i ~ Normal(θ_i, σ_i²) | y_i ~ Normal(θ_i, σ_i²) |
| **Hierarchy** | θ_i ~ Normal(μ, τ²) | θ_i ~ Student-t(ν, μ, τ²) | θ_i ~ Normal(μ, τ²) |
| **Prior: μ** | Normal(0, 50) | Normal(0, 50) | Normal(0, 50) |
| **Prior: τ** | Half-Cauchy(0, 5) | Half-Cauchy(0, 5) | Half-Normal(0, 3) |
| **Prior: ν** | — | Gamma(2, 0.1) | — |
| **Parameters** | 10 (μ, τ, θ_1,...,θ_8) | 11 (μ, τ, ν, θ_1,...,θ_8) | 10 (μ, τ, θ_1,...,θ_8) |
| **Complexity** | Baseline | +1 parameter | Baseline |

---

## Prior Predictive Distributions

### Model 1: Adaptive Hierarchical
- **Prior for τ**: Heavy-tailed, allows τ ∈ [0, ∞)
  - Median: 5.0
  - 95% interval: [0.13, 63.7]
  - Interpretation: Vague, lets data determine heterogeneity

### Model 2: Robust Hierarchical
- **Prior for ν**: Moderate regularization
  - Mean: 20.0, SD: 14.1
  - 95% interval: [3.8, 54.8]
  - Interpretation: Favors moderate tail heaviness (ν ≈ 10-30)

### Model 3: Informative Heterogeneity
- **Prior for τ**: Informative based on external evidence
  - Median: 2.0
  - 95% interval: [0, 5.9]
  - Interpretation: Expects moderate heterogeneity typical of meta-analyses

---

## Shrinkage Behavior

### When tau ≈ 0 (homogeneous effects)

| Model | Shrinkage Strength | Study-Specific θ_i |
|-------|-------------------|-------------------|
| Model 1 | Strong (near-complete pooling) | θ_i ≈ μ for all i |
| Model 2 | Strong (if ν > 30) | θ_i ≈ μ for all i |
| Model 3 | **Moderate** (prior resists τ→0) | θ_i partially pooled even if data suggests τ=0 |

### When tau > 0 (heterogeneous effects)

| Model | Shrinkage Strength | Study-Specific θ_i |
|-------|-------------------|-------------------|
| Model 1 | Moderate (partial pooling) | θ_i shrunk toward μ proportional to σ_i |
| Model 2 | **Weak for outliers** (heavy tails) | Extreme θ_i less influenced by μ |
| Model 3 | Moderate (same as Model 1) | θ_i shrunk toward μ proportional to σ_i |

---

## Expected Posterior Characteristics

### Model 1: Adaptive Hierarchical
**Given I²=0% observation**:
- **τ posterior**: Likely concentrates near 0 (potentially at boundary)
- **μ posterior**: Similar to precision-weighted frequentist estimate
- **θ_i posterior**: Strong shrinkage toward μ
- **Risk**: Funnel geometry if τ near zero (divergences)

### Model 2: Robust Hierarchical
**Given no clear outliers**:
- **ν posterior**: Likely > 30 (approaching normal)
- **τ posterior**: Similar to Model 1
- **μ posterior**: Similar to Model 1
- **θ_i posterior**: Similar to Model 1 if ν large
- **Risk**: Over-parameterization with J=8

### Model 3: Informative Heterogeneity
**Given I²=0% vs prior expecting τ > 0**:
- **τ posterior**: **Conflict** between data (τ≈0) and prior (τ≈2)
- **μ posterior**: May differ from Model 1 due to prior influence
- **θ_i posterior**: Less shrinkage than Model 1 (prior resists full pooling)
- **Risk**: Prior-data conflict, biased inference

---

## Computational Considerations

| Aspect | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| **Sampling speed** | Fast | Slower (Student-t) | Fast |
| **Convergence** | May need non-centered | May need more iterations | Similar to Model 1 |
| **Reparameterization** | Non-centered if τ→0 | Usually centered | Non-centered if τ→0 |
| **Typical iterations** | 4000 (2000 warmup) | 6000 (3000 warmup) | 4000 (2000 warmup) |
| **adapt_delta** | 0.95 | 0.98 | 0.95 |

---

## Model Selection Criteria

### When to Prefer Model 1
- ✓ No outliers detected
- ✓ Computational simplicity valued
- ✓ Standard meta-analysis assumptions hold
- ✓ LOO-CV not substantially worse than others

### When to Prefer Model 2
- ✓ Outliers suspected (Study 1 is 2.7 SD above median)
- ✓ LOO-CV improvement > 4 ELPD over Model 1
- ✓ ν posterior < 30 (heavy tails detected)
- ✓ Robustness to extreme values desired

### When to Prefer Model 3
- ✓ External evidence about heterogeneity is reliable
- ✓ J=8 too small for reliable τ estimation
- ✓ Prior predictive check passes
- ✓ No prior-data conflict detected
- ✗ **UNLIKELY for this dataset** (I²=0% conflicts with typical heterogeneity)

---

## Falsification Summary

| Model | Primary Failure Mode | Diagnostic Test | Threshold |
|-------|---------------------|-----------------|-----------|
| **Model 1** | Posterior predictive mismatch | PPC p-value | < 0.05 or > 0.95 |
| **Model 1** | Influential study | Leave-one-out Δμ | > 5 units |
| **Model 2** | Heavy tails unnecessary | ν posterior | > 50 |
| **Model 2** | No improvement | LOO-CV vs Model 1 | ELPD diff < 2 |
| **Model 3** | Prior-data conflict | Prior predictive Q | p < 0.05 |
| **Model 3** | Prior too strong | P(τ > 5 \| data) | > 0.2 |

---

## Interpretation Guide

### If Model 1 Wins (Most Likely)
**Interpretation**: Standard meta-analysis assumptions hold
- Studies are exchangeable
- No substantial heterogeneity (τ ≈ 0)
- No problematic outliers
- Borderline significance (P(μ > 0 | data) ≈ 0.51-0.53)

**Report**:
- Posterior for μ: mean, median, 95% CrI
- Posterior for τ: likely concentrates at 0
- P(μ > 0 | data): probability of positive effect
- Shrinkage plot: θ_i pulled strongly toward μ

### If Model 2 Wins (Possible Surprise)
**Interpretation**: Heavy-tailed effects detected
- Study 1 (or others) downweighted by model
- Robust to outliers
- ν posterior tells us degree of tail heaviness

**Report**:
- Same as Model 1, plus:
- ν posterior: degree of freedom estimate
- Comparison to Model 1: how much outliers affected inference

### If Model 3 Wins (Unlikely)
**Interpretation**: External evidence improves inference
- Data alone insufficient to estimate τ
- Informative prior reduces posterior uncertainty
- **BUT BEWARE**: Prior may be biasing results

**Critical check**:
- Prior-posterior conflict must NOT be present
- Sensitivity analysis essential

---

## Joint Posterior Relationships

### Correlation Structure

**Model 1 & 3** (Normal hierarchy):
- **Cor(μ, τ)**: Typically weak or negative
  - If τ large, μ uncertain (studies heterogeneous)
  - If τ small, μ more certain (studies agree)

**Model 2** (Student-t hierarchy):
- **Cor(μ, ν)**: Negative expected
  - If ν small (heavy tails), μ estimate more robust but uncertain
  - If ν large, μ similar to normal case

- **Cor(τ, ν)**: Potentially strong
  - If ν small, τ may be inflated (tail heaviness vs heterogeneity confounded)

### Funnel Geometry (Model 1 & 3)

When τ → 0, centered parameterization creates funnel:
- At τ=0: θ_i not identified (all equal to μ)
- Near τ=0: Highly correlated θ_i parameters
- **Solution**: Non-centered parameterization (θ = μ + τ·θ_raw)

---

## Posterior Predictive Checks

### What to Check

**For all models**:
1. **Observed y within predictive distribution?**
   - Generate y_rep ~ Normal(θ_i, σ_i) from posterior
   - Check: Is observed y_i in middle 95% of y_rep distribution?
   - Fail if > 1 study outside

2. **Q statistic in predictive distribution?**
   - Calculate Q_obs = 4.707 (from data)
   - Generate Q_rep from posterior predictive
   - Check: Is Q_obs typical of Q_rep?

3. **Effect size range plausible?**
   - Observed range: [-3, 28]
   - Predictive range: Should cover this
   - If observed range is extreme tail, model may be wrong

**Model 2 specific**:
- Check if Study 1 is downweighted appropriately
- Compare θ_1 posterior to Model 1 (should be shrunk more toward μ)

---

## Practical Recommendations

### Start with Model 1
1. Fit baseline hierarchical model
2. Check convergence (use non-centered if needed)
3. Apply all falsification criteria
4. If passes, this is your primary model

### Consider Model 2 if:
- Study 1 has high LOO Pareto-k (> 0.7)
- Leave-one-out changes μ substantially
- Domain knowledge suggests outliers likely

### Skip Model 3 unless:
- You have reliable external evidence about τ
- J=8 is too small for your needs
- Pilot studies suggest τ ≈ 2-3

### Model Averaging
If Model 1 and Model 2 both pass falsification:
- Use LOO-CV weights for Bayesian model averaging
- Report both models' results
- Averaged posterior: p_1 * P(μ|data, M1) + p_2 * P(μ|data, M2)

---

## Code Templates

### Prior Predictive Check (All Models)
```python
import pymc as pm
import numpy as np

# Generate from prior
with pm.Model():
    mu = pm.Normal('mu', 0, 50)
    tau = pm.HalfCauchy('tau', 5)  # Or HalfNormal(3) for Model 3
    theta = pm.Normal('theta', mu, tau, shape=8)
    y_prior = pm.Normal('y_prior', theta, sigma, shape=8)

    prior_samples = pm.sample_prior_predictive(samples=1000)

# Check if observed data is typical
Q_obs = 4.707
Q_prior = compute_Q(prior_samples['y_prior'])
p_value = (Q_prior > Q_obs).mean()
print(f"Prior predictive p-value: {p_value}")
# Reject if p < 0.05 or p > 0.95
```

### Leave-One-Out Analysis
```python
# Fit model 8 times, each dropping one study
results_loo = []
for i in range(8):
    y_loo = np.delete(y, i)
    sigma_loo = np.delete(sigma, i)

    # Fit model with N=7
    with pm.Model():
        # ... model specification
        trace_loo = pm.sample()

    mu_loo = trace_loo.posterior['mu'].mean()
    results_loo.append(mu_loo)

# Check influence
mu_full = trace.posterior['mu'].mean()
influence = [abs(mu_loo - mu_full) for mu_loo in results_loo]
max_influence = max(influence)

if max_influence > 5:
    print(f"REJECT MODEL: Study {influence.index(max_influence)+1} too influential")
```

---

## Expected Timeline

| Phase | Duration | Key Outputs |
|-------|----------|-------------|
| **Setup** | 1 hour | Stan/PyMC code, prior predictive checks |
| **Model 1 fitting** | 30 min | Posterior samples, diagnostics |
| **Model 2 fitting** | 1 hour | Posterior samples, diagnostics |
| **Model 3 fitting** | 30 min | Posterior samples, diagnostics |
| **Falsification checks** | 1 hour | Pass/fail for each model |
| **LOO-CV comparison** | 30 min | ELPD differences, model weights |
| **Sensitivity analysis** | 2 hours | Leave-one-out, prior sensitivity |
| **Reporting** | 2 hours | Visualizations, tables, interpretation |
| **TOTAL** | ~8 hours | Complete analysis with all checks |

---

**Files**: `/workspace/experiments/designer_1/model_comparison_table.md`
