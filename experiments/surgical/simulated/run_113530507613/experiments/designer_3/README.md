# Model Designer 3: Regression Models with Covariates

**Designer Role:** Bayesian regression specialist focusing on covariate structures and structured effects

**Date:** 2025-10-30

---

## Overview

This directory contains three Bayesian hierarchical regression models designed to test whether available covariates (sample size, group ordering) explain the observed heterogeneity in success rates across 12 groups.

### Key Philosophy

**We are TESTING hypotheses, not proving them.**

The EDA found no significant linear correlation between sample size and success rate (r = -0.34, p = 0.278). However, this does NOT mean covariates are uninformative. With J=12, we lack statistical power for frequentist tests. The Bayesian approach allows us to:

1. Quantify effect sizes with uncertainty (not just p-values)
2. Detect non-linear relationships (quadratic, varying slopes)
3. Formally compare models via LOO-CV
4. Make decisions under uncertainty

**If covariates don't help (ΔLOO < 2), that's a valid and important finding—it tells us to focus on other explanations (random effects, mixture models).**

---

## Files in This Directory

### Model Specifications

1. **`proposed_models.md`** - Complete theoretical documentation
   - Mathematical specifications
   - Stan implementation details
   - Falsification criteria
   - Computational considerations

### Stan Model Files

2. **`model1_size_covariate.stan`**
   - Hierarchical logistic regression with log(sample size) as covariate
   - Tests: Does sample size systematically predict success rate?
   - Key parameters: beta_0 (intercept), beta_1 (slope), tau (residual SD)

3. **`model2_quadratic_group.stan`**
   - Hierarchical logistic regression with quadratic group effect
   - Tests: Is there non-linear sequential structure?
   - Key parameters: beta_0 (intercept), beta_1 (linear), beta_2 (quadratic), tau

4. **`model3_random_slopes.stan`**
   - Hierarchical logistic regression with varying slopes
   - Tests: Does the size-response vary across groups?
   - Key parameters: beta_0, beta_1 (population effects), tau_alpha, tau_gamma (SDs), rho (correlation)

### Implementation

5. **`fit_models.py`** - Complete Python implementation
   - Compiles Stan models
   - Prepares data (centering, scaling)
   - Fits models with MCMC
   - Checks diagnostics (Rhat, ESS, divergences)
   - Computes LOO-CV for comparison
   - Generates plots and reports

---

## Model Summary

### Model 1: Sample Size Covariate (HIGHEST PRIORITY)

**Hypothesis:** Sample size (n_trials) has a systematic relationship with success rate.

**Why test this despite EDA?**
- Correlation test has low power (J=12)
- Log-transform may capture non-linear effects
- Bayesian approach quantifies uncertainty
- Large groups (e.g., Group 4, n=810) have strong influence

**Mathematical Form:**
```
logit(p_j) = alpha_j
alpha_j ~ Normal(mu_j, tau)
mu_j = beta_0 + beta_1 * log(n_j / n_mean)
```

**Falsification:** Abandon if 95% CI for beta_1 includes zero AND R² < 0.05

**Expected runtime:** ~30 seconds

---

### Model 2: Quadratic Group Effect (MEDIUM PRIORITY)

**Hypothesis:** Sequential group_id has non-linear (U-shape or inverted-U) structure.

**Why quadratic?**
- EDA found no linear trend (p = 0.69)
- But clusters suggest non-monotonic pattern (high at extremes, low in middle)
- Quadratic can detect this with just 2 parameters

**Mathematical Form:**
```
logit(p_j) = alpha_j
alpha_j ~ Normal(mu_j, tau)
mu_j = beta_0 + beta_1 * group_scaled_j + beta_2 * group_scaled_j^2
```

**Falsification:** Abandon if both beta_1 and beta_2 include zero

**Expected runtime:** ~30 seconds

**Critical caveat:** If group_id is truly arbitrary, this should find beta_1 ≈ beta_2 ≈ 0. That's a FEATURE—it tests the exchangeability assumption.

---

### Model 3: Random Slopes (LOWEST PRIORITY)

**Hypothesis:** The effect of sample size VARIES across groups (heterogeneous slopes).

**Why random slopes?**
- More realistic than fixed slopes
- Captures potential interactions (group × size)
- Allows each group to have its own size-response

**Mathematical Form:**
```
logit(p_j) = alpha_j + gamma_j * log(n_j / n_mean)
alpha_j ~ Normal(beta_0, tau_alpha)
gamma_j ~ Normal(beta_1, tau_gamma)
(alpha_j, gamma_j) correlated with rho
```

**Falsification:** Abandon if tau_gamma < 0.1 (slopes don't meaningfully vary)

**Expected runtime:** 2-5 minutes (more complex)

**Warning:** With J=12, estimating slope variation is ambitious. May not have power to detect.

---

## How to Use

### Prerequisites

```bash
pip install cmdstanpy arviz numpy pandas matplotlib seaborn
```

Also requires CmdStan installation (see: https://mc-stan.org/cmdstanpy/installation.html)

### Running the Analysis

**Basic usage:**
```bash
python fit_models.py --data /path/to/data.csv --output ./results
```

**With custom model directory:**
```bash
python fit_models.py \
    --data /workspace/data/binomial_data.csv \
    --output /workspace/experiments/designer_3/results \
    --models-dir /workspace/experiments/designer_3
```

### Expected Data Format

CSV file with columns:
- `group_id`: Integer (1 to J)
- `n_trials`: Integer (number of trials per group)
- `r_successes`: Integer (number of successes per group)

### Output Files

The script generates:

1. **Diagnostic plots:**
   - `model1_predictions.png` - Observed vs predicted
   - `model1_coefficients.png` - Posterior distributions

2. **Results files:**
   - `model1_results.json` - Summary statistics and LOO scores
   - `model_comparison.csv` - LOO comparison table

3. **Console output:**
   - MCMC diagnostics (Rhat, ESS, divergences)
   - LOO comparison with interpretation
   - Decision guidance

---

## Interpretation Guide

### LOO Comparison (ΔLOO)

**How to read the comparison table:**

- **ΔLOO < 2:** Models are practically equivalent → Prefer simpler model
- **2 < ΔLOO < 4:** Weak evidence → Slight preference for best model
- **ΔLOO > 4:** Substantial evidence → Clear preference for best model

**Example interpretation:**

```
         rank  elpd_loo   p_loo  elpd_diff     se
model1      0    -45.2    5.3       0.0      0.0
baseline    1    -47.8    3.2       2.6      1.8
model2      2    -48.1    6.1       2.9      2.3
```

**Interpretation:**
- Model 1 (size covariate) ranks best
- ΔLOO vs baseline = 2.6 ± 1.8 (borderline significant)
- Weak evidence that sample size helps
- Report beta_1 with large uncertainty

### Coefficient Interpretation

**Model 1 (Sample Size):**

- **beta_1 < 0:** Larger samples → lower success rates
- **beta_1 > 0:** Larger samples → higher success rates
- **|beta_1| > 0.5:** Substantial effect (1 unit change in log(n) ≈ large probability shift)
- **R² > 0.20:** Sample size explains > 20% of variance

**Model 2 (Quadratic):**

- **beta_2 > 0:** U-shape (low in middle, high at extremes)
- **beta_2 < 0:** Inverted-U (high in middle, low at extremes)
- **peak_location in [-1, 1]:** Peak within observed range
- **peak_location outside [-1, 1]:** Extrapolation (be cautious)

**Model 3 (Random Slopes):**

- **tau_gamma > 0.2:** Substantial slope variation
- **rho < 0:** High-baseline groups have weaker size effects
- **rho > 0:** High-baseline groups have stronger size effects
- **prop_var_slopes > 0.15:** Slopes explain > 15% of variance

### Decision Rules

**Scenario 1: Model 1 wins decisively (ΔLOO > 4)**
→ Sample size effect is real
→ Report beta_1 and R²
→ Investigate WHY large studies differ (confounding?)

**Scenario 2: Model 2 wins decisively (ΔLOO > 4)**
→ Sequential structure exists
→ Group ordering is NOT arbitrary
→ Investigate what group_id represents (time? space?)

**Scenario 3: Model 3 wins decisively (ΔLOO > 4)**
→ Slopes vary across groups
→ Heterogeneous size-response
→ Consider mixture model (Designer 2)

**Scenario 4: No model wins (all ΔLOO < 2)**
→ **Covariates are uninformative**
→ Prefer simple random effects model (Designer 1)
→ Heterogeneity is "pure noise" (or unmeasured covariates)

---

## Falsification Criteria

### When to Abandon Each Model

**Model 1 (Size Covariate):**
- [ ] 95% CI for beta_1 includes zero AND is narrow
- [ ] R² < 0.05
- [ ] ΔLOO < 2 vs baseline
- [ ] tau_covariate ≈ tau_baseline (no reduction in residual variance)

→ **Conclusion:** Sample size is not a meaningful predictor

**Model 2 (Quadratic Group):**
- [ ] Both beta_1 and beta_2 include zero
- [ ] R² < 0.05
- [ ] ΔLOO < 2 vs baseline
- [ ] Peak location outside observed range

→ **Conclusion:** No sequential structure, group_id is arbitrary

**Model 3 (Random Slopes):**
- [ ] tau_gamma < 0.1 (slopes don't vary)
- [ ] prop_var_slopes < 0.05
- [ ] ΔLOO < 0 vs Model 1 (simpler fixed slopes)
- [ ] Computational failures (divergences, poor mixing)

→ **Conclusion:** Slopes don't meaningfully vary, use Model 1

### When to Abandon ALL Regression Models

If ALL of the following hold:
- [ ] No model achieves ΔLOO > 2 vs baseline
- [ ] All R² < 0.05
- [ ] All coefficient CIs include zero
- [ ] Persistent computational issues across all models

→ **Conclusion:** Covariates (n_trials, group_id) are uninformative
→ **Pivot to:** Random effects only (Designer 1) or mixture models (Designer 2)

---

## Computational Notes

### Expected Performance

| Model | Compilation | Sampling | Total | Memory |
|-------|-------------|----------|-------|--------|
| Model 1 | ~10s | ~30s | ~40s | ~100 MB |
| Model 2 | ~10s | ~30s | ~40s | ~100 MB |
| Model 3 | ~15s | 2-5 min | ~5 min | ~150 MB |

*Tested on: 8-core CPU, 16GB RAM*

### Diagnostic Thresholds

**Acceptable:**
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- Divergences < 1% of samples
- Pareto-k < 0.7 for all observations

**If diagnostics fail:**
1. Increase adapt_delta to 0.99
2. Increase warmup iterations to 2000
3. Try tighter priors
4. Check for data issues
5. Consider model misspecification

### Troubleshooting

**Problem:** Divergent transitions
- **Solution:** Increase adapt_delta, check priors, verify data

**Problem:** Low ESS for tau_gamma (Model 3)
- **Solution:** Normal with J=12, need more iterations or fix rho=0

**Problem:** Rhat > 1.01
- **Solution:** Run longer, check for multimodality, verify model

**Problem:** High Pareto-k values
- **Solution:** Indicates influential observations (expected for outliers), may need robust model

---

## Integration with Other Designers

### Expected Relationships

**With Designer 1 (Random Effects):**
- If my models fail (ΔLOO < 2), supports Designer 1's approach
- Random effects-only is the "baseline" for comparison
- Complementary, not competing

**With Designer 2 (Mixture/Robust):**
- If Model 2 succeeds (sequential structure), may align with clusters
- If Model 3 succeeds (varying slopes), suggests mixture components
- Could combine: mixture model with covariates

### Model Averaging Strategy

If no single model dominates (all ΔLOO < 4):

1. **Bayesian Model Averaging:** Weight by LOO-IC
2. **Stacking:** Optimal linear combination
3. **Ensemble:** Report all plausible models

**Don't force a winner!** Multiple models may be equally plausible.

---

## Contact and Issues

This model design was created by Model Designer 3 as part of a parallel model design exercise.

**Key Questions to Ask:**

1. What is the substantive meaning of group_id? (temporal? spatial? arbitrary?)
2. Why do large samples have different rates? (confounding? selection?)
3. Are there unmeasured covariates that might explain more variance?
4. Is the binomial likelihood appropriate? (check for overdispersion)

**Red Flags:**

- Extreme parameter estimates (|beta| > 2)
- Posterior at boundaries (tau = 0, rho = ±1)
- Poor predictive performance despite good fit
- Results contradict EDA without explanation

---

## References

**Statistical Methods:**
- Gelman & Hill (2007): Data Analysis Using Regression and Multilevel/Hierarchical Models
- McElreath (2020): Statistical Rethinking (Chapter 13: Multilevel Models)
- Vehtari et al. (2017): Practical Bayesian model evaluation using LOO-CV

**Software:**
- Stan: https://mc-stan.org/
- ArviZ: https://arviz-devs.github.io/
- CmdStanPy: https://mc-stan.org/cmdstanpy/

**Related Files:**
- EDA Report: `/workspace/eda/eda_report.md`
- Designer 1 Models: `/workspace/experiments/designer_1/` (if exists)
- Designer 2 Models: `/workspace/experiments/designer_2/` (if exists)
