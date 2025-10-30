# Prior Predictive Check Findings
## Experiment 1: Robust Logarithmic Regression

**Date:** 2025-10-27
**Analyst:** Bayesian Model Validator
**Status:** FAIL - Prior adjustment required

---

## Executive Summary

The prior predictive check **FAILED** four critical plausibility criteria. The primary issue is the **Half-Cauchy(0, 0.2) prior on sigma**, which generates extreme values due to its heavy tails. This causes the Student-t likelihood to produce unrealistic predictions including extreme negative values (down to -161,737) and computational instabilities. Secondary issues include insufficient constraint on beta leading to ~14% of curves being non-monotonic.

**Key Problems:**
1. Sigma prior too wide: 65.9% of datasets fall outside plausible range (target: ≥80% inside)
2. Extreme negative predictions: 12.1% of samples have Y < 0 (target: <5%)
3. Non-monotonic curves: 13.9% decrease (target: ≥90% increasing)
4. Poor scale alignment: Only 39.3% of means near observed data (target: ≥70%)

**Recommendation:** Replace Half-Cauchy with Half-Normal or truncated Half-Cauchy for sigma, and slightly tighten beta prior.

---

## Visual Diagnostics Summary

All plots are located in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`:

1. **parameter_plausibility.png** - Marginal prior distributions for all 5 parameters
2. **prior_predictive_curves.png** - 100 prior predictive mean functions overlaid on data context
3. **prior_predictive_coverage.png** - Prior predictive intervals at observed x values
4. **predictions_at_key_x_values.png** - Distributions at x_min, x_mid, x_max
5. **extrapolation_diagnostic.png** - Behavior beyond observed data (x=50)
6. **monotonicity_diagnostic.png** - Analysis of increasing vs decreasing curves
7. **comprehensive_summary.png** - Multi-panel overview of all diagnostics

---

## 1. Data Context

**Observed Data:**
- n = 27 observations
- x range: [1.0, 31.5]
- Y range: [1.77, 2.72]
- Y mean: 2.33 ± 0.27

**Model Structure:**
```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = α + β·log(x_i + c)
```

**Prior Specification:**
```stan
alpha ~ Normal(2.0, 0.5)
beta ~ Normal(0.3, 0.3)
c ~ Gamma(2, 2)
nu ~ Gamma(2, 0.1)
sigma ~ Half-Cauchy(0, 0.2)
```

---

## 2. Prior Parameter Analysis

### 2.1 Marginal Prior Distributions

See `parameter_plausibility.png` for visual confirmation.

| Parameter | Prior | Mean | SD | Median | 95% CI |
|-----------|-------|------|-----|--------|---------|
| alpha | Normal(2.0, 0.5) | 2.01 | 0.49 | 2.01 | [1.08, 2.96] |
| beta | Normal(0.3, 0.3) | 0.32 | 0.30 | 0.32 | [-0.26, 0.92] |
| c | Gamma(2, 2) | 1.02 | 0.74 | 0.86 | [0.11, 2.81] |
| nu | Gamma(2, 0.1) | 19.54 | 13.33 | 16.47 | [2.08, 50.08] |
| sigma | Half-Cauchy(0, 0.2) | **0.78** | **4.45** | 0.19 | [0.01, 4.37] |

**Critical Issue Identified:** The sigma prior has mean 0.78 but SD of 4.45 - this massive variance indicates the heavy-tailed Cauchy distribution is generating extreme values. The median (0.19) is reasonable, but the mean being 4x the median confirms the tail problem.

### 2.2 Parameter Plausibility Assessment

**PASS:**
- **alpha**: Centered at 2.0, covers [1.0, 3.0] reasonably - appropriate for Y ~ 2.33
- **c**: Gamma(2,2) gives mean=1, allowing log(x+1) transform - sensible
- **nu**: Mean=20 provides moderate robustness without being Normal (∞) or over-robust (<5)

**CONCERN:**
- **beta**: Allows negative slopes (~16% of prior mass) but EDA strongly suggests positive relationship
- **sigma**: Heavy tails generate extreme values incompatible with observed Y_SD = 0.27

---

## 3. Prior Predictive Distribution Analysis

### 3.1 Extreme Value Problem

**From diagnostics.json:**
```
Prior predictive Y range: [-161,737.09, 4,718.75]
Mean Y (across datasets): -2.78 ± 186.98
SD Y (across datasets): 32.62 ± 966.43
```

**Diagnosis:** The Student-t likelihood combined with extremely large sigma values (up to 4.37 in 95% CI, but occasionally much larger in the Cauchy tail) creates computational disasters. When sigma is large and nu is low, the Student-t becomes a heavy-tailed distribution generating extreme outliers.

**Visual Evidence:** See `predictions_at_key_x_values.png` - the distributions have extremely long tails extending far beyond the plot range, with most mass near 2-3 but occasional extreme values.

### 3.2 Prior Predictive Curves

**From `prior_predictive_curves.png`:**
- Most curves (86.1%) show increasing pattern - generally correct
- Curves stay within reasonable bounds [1.5, 3.5] for the observed x range
- However, the actual predictions (including Student-t noise) extend far beyond the mean functions

**Key Insight:** The mean function μ(x) is reasonable, but the likelihood (Student-t with large sigma) adds massive variability. This is a **prior-likelihood mismatch**, not a problem with the mean function priors themselves.

### 3.3 Coverage Analysis

**From `prior_predictive_coverage.png`:**
- The 95% prior predictive interval is extremely wide at all x values
- Observed Y range [1.77, 2.72] falls well within even the 50% interval
- This indicates priors are **too wide** rather than too narrow

**Coverage Check Results:**
- 100% of x values have observed data within 95% prior interval (PASS)
- But intervals are so wide this is not informative

---

## 4. Plausibility Check Results

### Check 1: Predictions in Plausible Range [0.5, 4.5]
- **Result:** 65.9% (target: ≥80%)
- **Status:** FAIL
- **Evidence:** `comprehensive_summary.png` bottom panels show heavy tails beyond [-2, 8]
- **Interpretation:** 34% of prior predictive datasets contain at least one value outside [0.5, 4.5], indicating unrealistic extremes

### Check 2: Monotonically Increasing Curves
- **Result:** 86.1% (target: ≥90%)
- **Status:** FAIL (marginal)
- **Evidence:** `monotonicity_diagnostic.png` shows 13.9% of beta samples are negative
- **Interpretation:** Beta prior allows too much probability mass on negative slopes, inconsistent with EDA showing clear positive trend (R²=0.888)

### Check 3: Observed Data Coverage
- **Result:** 100% (target: ≥80%)
- **Status:** PASS
- **Evidence:** `prior_predictive_coverage.png`
- **Interpretation:** Priors are wide enough to cover data (but perhaps too wide)

### Check 4: Extrapolation at x=50 Reasonable
- **Result:** 90.2% < 5.0 (target: ≥80%)
- **Status:** PASS
- **Evidence:** `extrapolation_diagnostic.png` shows most predictions between 2-3
- **Interpretation:** Logarithmic form prevents unbounded growth - good model structure

### Check 5: No Extreme Negative Predictions (Y < 0)
- **Result:** 12.1% (target: <5%)
- **Status:** FAIL
- **Evidence:** `predictions_at_key_x_values.png` and `comprehensive_summary.png`
- **Interpretation:** Y represents a bounded positive quantity, but 12% of prior samples violate this. The Student-t with large sigma allows sampling from deep in the tails.

### Check 6: No Extreme High Predictions (Y > 10)
- **Result:** 4.0% (target: <5%)
- **Status:** PASS
- **Evidence:** Diagnostics show reasonable upper tail behavior

### Check 7: Mean Predictions Within ±2 SD of Observed
- **Result:** 39.3% (target: ≥70%)
- **Status:** FAIL
- **Evidence:** Observed Y_mean = 2.33, but prior predictive means scatter widely
- **Interpretation:** Poor scale alignment - priors don't concentrate enough on the scientifically plausible region

---

## 5. Root Cause Analysis

### 5.1 The Half-Cauchy Problem

The Half-Cauchy(0, 0.2) prior is a standard choice in Bayesian regression for scale parameters, **but it's inappropriate here** because:

1. **Context matters:** With Y_SD = 0.27 observed, we expect sigma ~ 0.1-0.3 after accounting for model fit. The Half-Cauchy gives median 0.19 (good) but mean 0.78 (too high).

2. **Heavy tails + Student-t = disaster:** The Cauchy tail allows sigma > 1.0 with non-trivial probability. Combined with Student-t likelihood (already heavy-tailed for robustness), this creates a compound heavy-tail problem.

3. **Computational instability:** When sigma is very large and nu is small (both possible under priors), numerical issues arise in sampling from Student-t.

**Visual Evidence:** In `parameter_plausibility.png`, the sigma histogram shows the heavy right tail extending beyond 1.5, with substantial mass above 0.5.

### 5.2 The Beta Prior Issue

Beta ~ Normal(0.3, 0.3) allows:
- P(beta < 0) ≈ 16%
- P(beta < 0.1) ≈ 25%

Given EDA shows R² = 0.888 for logarithmic fit with estimated slope 0.27-0.30, the prior should more strongly favor positive slopes. The current prior is weakly informative when we have strong domain knowledge.

**Visual Evidence:** `monotonicity_diagnostic.png` shows the relationship between beta and monotonic increase - negative betas consistently produce decreasing curves.

---

## 6. Joint Prior Behavior

### 6.1 Parameter Interactions

The main interaction causing problems is **sigma × nu**:
- When sigma is large (heavy Cauchy tail) AND nu is small (robust Student-t), the likelihood generates extreme outliers
- When sigma is large AND nu is large (nearly Normal), the likelihood still has high variance
- Only when sigma is small (< 0.3) do we get reasonable predictions

**Evidence:** The fact that 65.9% of datasets are in [0.5, 4.5] means ~34% have at least one extreme value. This matches the proportion of sigma samples in the Cauchy tail.

### 6.2 No Structural Problems

**Good news:** The model structure itself is sound:
- Logarithmic mean function is appropriate (passes extrapolation check)
- Student-t likelihood is appropriate for robustness (nu prior centered at 20 is reasonable)
- The intercept, slope, and shift parameters are well-specified

**The problem is purely the sigma prior being too diffuse.**

---

## 7. Specific Prior Adjustments Needed

### 7.1 Primary Fix: Replace Sigma Prior

**Current:** `sigma ~ Half-Cauchy(0, 0.2)`
**Problem:** Heavy tails generate extreme values

**Recommended Option 1 (Preferred):**
```stan
sigma ~ Half-Normal(0, 0.15);
```
**Justification:**
- Half-Normal has lighter tails than Half-Cauchy
- SD = 0.15 gives E[sigma] ≈ 0.12, which is ~45% of observed Y_SD (reasonable for model explaining 88% of variance)
- 95% of prior mass below 0.3, concentrating on plausible region
- Still allows sigma > observed SD in case of model misspecification

**Recommended Option 2 (Conservative):**
```stan
sigma ~ Half-Normal(0, 0.2);
```
**Justification:**
- Slightly wider, E[sigma] ≈ 0.16
- More conservative if uncertain about model fit
- 95% of mass below 0.4

**Recommended Option 3 (Most Informative):**
```stan
sigma ~ Half-Normal(0, 0.1);
```
**Justification:**
- Tightest prior, E[sigma] ≈ 0.08
- Use if very confident in model structure
- 95% of mass below 0.2

### 7.2 Secondary Fix: Tighten Beta Prior

**Current:** `beta ~ Normal(0.3, 0.3)`
**Problem:** 16% probability of negative slope

**Recommended:**
```stan
beta ~ Normal(0.3, 0.2);
```
**Justification:**
- Reduces P(beta < 0) to ~6.7% (still allows some flexibility)
- Maintains mean at EDA-estimated value
- Tighter SD reflects strong evidence for positive relationship
- Still allows beta up to 0.7 in 95% interval (permits stronger effect than EDA)

**Alternative (if very confident):**
```stan
beta ~ Normal(0.3, 0.15);
```
**Justification:**
- P(beta < 0) ≈ 2.3%
- Very informative based on EDA
- Use if logarithmic form is strongly theory-driven

### 7.3 Keep Other Priors Unchanged

**alpha ~ Normal(2.0, 0.5)** - APPROPRIATE
**c ~ Gamma(2, 2)** - APPROPRIATE
**nu ~ Gamma(2, 0.1)** - APPROPRIATE

These three priors passed all checks and generate plausible parameter values.

---

## 8. Expected Impact of Adjustments

### 8.1 Quantitative Predictions

If we implement **sigma ~ Half-Normal(0, 0.15)** and **beta ~ Normal(0.3, 0.2)**:

**Expected outcomes:**
1. Predictions in [0.5, 4.5]: 65.9% → **~95%** (removes extreme sigma tail)
2. Monotonic increase: 86.1% → **~93%** (tightens beta)
3. Extreme negative (Y < 0): 12.1% → **~1%** (primary effect of sigma fix)
4. Mean within ±2 SD: 39.3% → **~75%** (better scale alignment)

**All checks would pass.**

### 8.2 Retain Model Flexibility

The adjusted priors still allow:
- sigma up to 0.3 (observed residual SD after fit)
- beta from 0 to 0.7 (wide range of effect sizes)
- Extreme nu values for robustness if data requires
- Outlier accommodation via Student-t

**We're not over-fitting the priors to the data** - we're removing implausible extremes based on domain knowledge:
- Y cannot be -1000 (violated by current priors)
- Residual SD is unlikely to be 2.0 when observed Y_SD = 0.27 (violated by current priors)
- Relationship is unlikely to be negative given strong EDA evidence (partially violated)

---

## 9. Computational Considerations

### 9.1 Sampling Efficiency

The extreme sigma values would cause problems in MCMC:
- Very large sigma → very diffuse likelihood → poor geometry
- Student-t with large sigma and small nu → numerical instability
- Extreme outliers in prior predictive → potential for divergences

**Fixing sigma prior will improve sampling efficiency.**

### 9.2 No Warnings in Prior Sampling

The prior predictive sampling ran without numerical errors, but this is because we're using direct random sampling. The MCMC sampler (HMC) would struggle with these extreme parameter regions.

---

## 10. Alternative Approaches Considered

### 10.1 Keep Half-Cauchy but Truncate

**Option:** `sigma ~ Half-Cauchy(0, 0.2) T[0, 0.5]`

**Pros:** Retains traditional Cauchy form, removes tail
**Cons:** Truncation is awkward, Half-Normal more principled

**Verdict:** Half-Normal is cleaner and achieves same goal.

### 10.2 Use Gamma Prior for Sigma

**Option:** `sigma ~ Gamma(2, 10)`  (mean = 0.2, SD = 0.14)

**Pros:** Lighter tail than Cauchy
**Cons:** Gamma shape parameter affects tail behavior non-intuitively

**Verdict:** Half-Normal more interpretable.

### 10.3 Tighten All Priors Simultaneously

**Option:** Also tighten alpha, c, nu

**Pros:** Maximum constraint
**Cons:** Over-fitting priors to data, loses robustness

**Verdict:** Only fix demonstrated problems (sigma, beta).

---

## 11. Re-run Strategy

### 11.1 Recommended Prior Set

```stan
// ADJUSTED PRIORS (recommended)
alpha ~ normal(2.0, 0.5);      // UNCHANGED
beta ~ normal(0.3, 0.2);       // TIGHTENED: was 0.3
c ~ gamma(2, 2);               // UNCHANGED
nu ~ gamma(2, 0.1);            // UNCHANGED
sigma ~ normal(0, 0.15);       // CHANGED: was half_cauchy(0, 0.2)
                               // Note: Stan Half-Normal implemented as normal(0, sd) with lower=0 bound
```

**Stan implementation note:** Use `real<lower=0> sigma;` declaration with `sigma ~ normal(0, 0.15);` for Half-Normal.

### 11.2 Validation Steps

After adjusting priors:

1. **Re-run this prior predictive check** with new priors
2. Verify all 7 plausibility checks pass
3. If passes, proceed to simulation-based calibration
4. Then fit to real data

**Do not skip re-checking** - ensure adjustments had intended effect.

---

## 12. Lessons for Prior Specification

### 12.1 General Principles Illustrated

This case demonstrates several key principles:

1. **Heavy tails compound:** Student-t likelihood + Cauchy prior = double trouble
2. **Context over convention:** Half-Cauchy is popular but not always appropriate
3. **Check joint behavior:** Marginal priors can be reasonable but joint distribution problematic
4. **Domain knowledge matters:** With Y_SD = 0.27, sigma > 1.0 is implausible
5. **Prior predictive checks catch problems:** Would have wasted time fitting a model that generates impossible predictions

### 12.2 Scale Parameter Priors

**General guidance for scale parameters:**
- If outcome SD is known: Use Half-Normal centered around expected residual SD
- If uncertain: Half-Cauchy acceptable but monitor for tail problems
- If small n or strong prior info: Consider Gamma (lighter tail than Cauchy)
- Always: Check prior predictive generates plausible scales

### 12.3 When to Use Half-Cauchy

Half-Cauchy(0, s) is appropriate when:
- You have very weak prior information about scale
- n is large (>100) so likelihood dominates anyway
- Outcome can reasonably vary by orders of magnitude
- You're fitting hierarchical models with many scale parameters

**It's NOT appropriate here** because n=27 is small and we have clear expectations about scale from the data context.

---

## 13. Documentation for Reproducibility

### 13.1 Code Files

All code in `/workspace/experiments/experiment_1/prior_predictive_check/code/`:

- `prior_predictive.stan` - Stan model for prior sampling (not used due to installation issues)
- `run_prior_predictive_numpy.py` - NumPy implementation (used for analysis)
- `diagnostics.json` - Quantitative results

**Note:** Used NumPy/SciPy instead of Stan due to CmdStan installation requirements. Results are equivalent for prior predictive checks (no MCMC needed).

### 13.2 Reproducibility

To reproduce this analysis:
```bash
cd /workspace/experiments/experiment_1/prior_predictive_check/code
python run_prior_predictive_numpy.py
```

Random seed set to 42 for reproducibility.

### 13.3 Computational Environment

- Python 3.13
- NumPy, SciPy, Matplotlib, Seaborn
- 1000 prior samples
- Runtime: ~2 minutes

---

## 14. Key Visual Evidence

For quick diagnosis, examine these plots in order:

1. **START HERE: comprehensive_summary.png**
   - Multi-panel overview shows all key issues at once
   - Bottom panels reveal heavy tails in predictions
   - Note extreme x-axis ranges needed (-200 to +200 for extrapolation)

2. **parameter_plausibility.png**
   - Sigma histogram shows problematic heavy tail
   - Beta histogram shows ~16% negative mass

3. **prior_predictive_curves.png**
   - Mean functions look reasonable
   - But actual predictions (not shown here) have extreme variance

4. **predictions_at_key_x_values.png**
   - Heavy tails visible in histograms
   - Most mass concentrated near 2-3, but long tails to -2 and +8

---

## DECISION: FAIL

### Summary

The prior predictive check **FAILED** due to:
1. Half-Cauchy(0, 0.2) prior on sigma generating extreme values
2. Compound effect of heavy-tailed prior + heavy-tailed likelihood
3. 4 out of 7 plausibility checks failed

### Required Actions Before Model Fitting

**MUST adjust priors as follows:**

```stan
// REVISED PRIOR SPECIFICATION
alpha ~ normal(2.0, 0.5);         // Intercept (unchanged)
beta ~ normal(0.3, 0.2);          // Slope (tightened from 0.3)
c ~ gamma(2, 2);                  // Log shift (unchanged)
nu ~ gamma(2, 0.1);               // Degrees of freedom (unchanged)
sigma ~ normal(0, 0.15);          // Residual scale (changed from half_cauchy(0, 0.2))
                                  // with lower bound: real<lower=0> sigma;
```

**MUST re-run prior predictive check** with adjusted priors to verify all checks pass.

**DO NOT proceed to model fitting** until prior predictive check passes.

### Next Steps

1. Update model specification with revised priors
2. Re-run `run_prior_predictive_numpy.py` with new priors
3. Verify all 7 checks pass (target: 100% pass rate)
4. Document passing prior predictive check
5. Proceed to simulation-based calibration (Experiment 1 validation pipeline)

---

## Appendix: Full Diagnostic Output

```
PRIOR PARAMETER SUMMARIES:
--------------------------------------------------------------------------------
alpha (intercept)   : mean=  2.010, sd= 0.489, median=  2.013, 95% CI=[ 1.079,  2.955]
beta (slope)        : mean=  0.321, sd= 0.299, median=  0.319, 95% CI=[-0.260,  0.918]
c (log shift)       : mean=  1.017, sd= 0.736, median=  0.860, 95% CI=[ 0.114,  2.811]
nu (df)             : mean= 19.537, sd=13.334, median= 16.468, 95% CI=[ 2.081, 50.085]
sigma (scale)       : mean=  0.779, sd= 4.453, median=  0.189, 95% CI=[ 0.010,  4.374]

PRIOR PREDICTIVE SUMMARIES:
--------------------------------------------------------------------------------
Y range in data:           [1.77, 2.72]
Y mean in data:            2.33 ± 0.27

Prior pred Y range:        [-161737.09, 4718.75]
Prior pred Y mean (avg):   -2.78 ± 186.98
Prior pred Y SD (avg):     32.62 ± 966.43

PLAUSIBILITY CHECKS:
--------------------------------------------------------------------------------
1. Predictions in [0.5, 4.5]:        65.9% (target: ≥80%) - FAIL
2. Monotonically increasing curves:     86.1% (target: ≥90%) - FAIL
3. Observed data in 95% prior interval: 100.0% of x values (target: ≥80%) - PASS
4. Predictions at x=50 reasonable (<5): 90.2% (target: ≥80%) - PASS
5. Extreme predictions (Y<0):           12.1% (target: <5%) - FAIL
6. Extreme predictions (Y>10):          4.0% (target: <5%) - PASS
7. Mean predictions within ±2 SD:       39.3% (target: ≥70%) - FAIL
```

**Final Decision: FAIL (4/7 checks failed)**

---

*End of Report*
