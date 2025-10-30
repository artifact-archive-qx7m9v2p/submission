# Prior Predictive Check: Negative Binomial Quadratic Model

**Experiment:** 1
**Model:** Negative Binomial with Quadratic Time Trend
**Date:** 2025-10-29
**Decision:** ADJUST (Priors need tightening to reduce extreme predictions)

---

## Model Specification

```
C_i ~ NegativeBinomial(μ_i, φ)
log(μ_i) = β₀ + β₁·year_i + β₂·year_i²

Priors:
β₀ ~ Normal(4.7, 0.5)
β₁ ~ Normal(0.8, 0.3)
β₂ ~ Normal(0.3, 0.2)
φ ~ Gamma(2, 0.5)
```

**Data Context:**
- 40 observations: year ∈ [-1.67, 1.67], C ∈ [19, 272]
- Observed mean: 109.5, variance: 7441.7

---

## Visual Diagnostics Summary

All diagnostic plots are saved in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`:

1. **parameter_plausibility.png** - Marginal distributions of each prior
2. **prior_predictive_trajectories.png** - Spaghetti plot of simulated count trajectories with observed data overlay
3. **prior_predictive_coverage.png** - Four-panel diagnostic showing simulated counts at key time points, ranges, means, and maxima
4. **expected_value_trajectories.png** - Expected values (μ) with median and 90% credible intervals
5. **parameter_space_coverage.png** - Six pairwise parameter scatterplots showing prior independence
6. **growth_pattern_diversity.png** - Growth pattern classification and curvature direction distribution

---

## Diagnostic Results

### 1. Parameter Prior Behavior

**From `parameter_plausibility.png` and console output:**

| Parameter | Prior Specification | Sampled Mean | Sampled Std | Range |
|-----------|-------------------|--------------|-------------|-------|
| β₀ | Normal(4.7, 0.5) | 4.699 | 0.490 | [3.23, 6.66] |
| β₁ | Normal(0.8, 0.3) | 0.805 | 0.306 | [-0.32, 1.78] |
| β₂ | Normal(0.3, 0.2) | 0.302 | 0.200 | [-0.39, 0.83] |
| φ | Gamma(2, 0.5) | 3.929 | 2.618 | [0.08, 15.67] |

**Assessment:** All priors sample correctly from their specified distributions with no numerical issues.

### 2. Prior Predictive Coverage

**From `prior_predictive_trajectories.png` and `prior_predictive_coverage.png`:**

**Critical Finding - Extreme Predictions:**
- **Simulated mean counts:** range [28.1, 2063.9], mean = 284.0
- **Simulated max counts:** range [91, 42686], mean = 2047.8
- **Observed data:** mean = 109.5, max = 272

**Key Metrics:**
- Simulations with mean in plausible range [10, 500]: **89.4%**
- Simulations covering observed min (19): **65.8%**
- Simulations covering observed max (272): **97.1%**
- Counts > 10,000 generated: **21 instances**
- No negative counts (domain violations): **0** ✓

**Problem:** While the priors cover the observed data range, they generate a substantial tail of extreme predictions. The spaghetti plot shows most trajectories are reasonable, but a concerning minority explode to unrealistic values (>10,000 counts).

### 3. Expected Values at Key Time Points

**From `expected_value_trajectories.png`:**

| Time Point | Year | Observed | Median μ | 90% CI |
|------------|------|----------|----------|---------|
| Early | -1.67 | 29 | 64.0 | [16.8, 287.9] |
| Mid | 0.04 | 87 | 112.9 | [52.3, 252.7] |
| Late | 1.67 | 245 | 977.8 | [231.2, 4252.9] |

**Critical Issue:** At the late time point, the median expected value (977.8) is already **4× larger** than the observed maximum (272), and the upper 90% CI reaches 4252.9. This indicates the priors strongly favor explosive growth at the end of the observation period.

### 4. Growth Pattern Diversity

**From `growth_pattern_diversity.png`:**

**Curvature Strength:**
- Near-linear (|β₂| < 0.1): 13.5%
- Moderate curve (0.1 ≤ |β₂| < 0.4): 55.1%
- Strong curve (|β₂| ≥ 0.4): 31.4%

**Curvature Direction:**
- Positive β₂ (upward acceleration): **93.2%**
- Negative β₂ (downward curve): 6.8%

**Assessment:** The prior strongly favors upward-curving trajectories (93%), which combined with positive β₁ (linear term) creates the explosive growth problem. This is reasonable given the observed data shows accelerating growth, but the magnitude is unconstrained.

### 5. Parameter Independence

**From `parameter_space_coverage.png`:**

All six pairwise parameter plots show no spurious correlations - the parameters sample independently as expected from independent priors. The parameter space is well-explored.

---

## Key Visual Evidence

### Most Important Plot: `prior_predictive_trajectories.png`
This plot reveals the core issue: while most trajectories are reasonable and cover the observed data, a visible subset shoot up dramatically at late time points, exceeding 10,000 counts. The observed data (red points) cluster near the bottom of the prior predictive envelope.

### Second Critical Plot: `expected_value_trajectories.png`
The 90% credible interval fans out dramatically, with the upper bound exceeding 10,000 by year=1.67. The median trajectory alone exceeds observed data by 4×, indicating systematic overestimation bias in the late period.

### Supporting Evidence: `prior_predictive_coverage.png`
The bottom-left panel shows the distribution of simulated means is right-skewed with a heavy tail extending to 2000+, while the observed mean (109.5) sits well within the bulk but toward the left side of the distribution.

---

## Domain Violations and Computational Flags

### Domain Violations
- **Negative counts:** 0 ✓
- All simulated counts are non-negative integers as required

### Computational Flags
- **Extreme values:** 21 counts > 10,000 detected
- **Maximum simulated count:** 42,686 (157× observed maximum)
- **Risk:** Such extreme values may cause numerical overflow in log-likelihood calculations during MCMC sampling

### Scale Appropriateness
- **Under-coverage at early times:** Only 65.8% of simulations reach as low as the observed minimum (19)
- **Over-shooting at late times:** Expected values systematically exceed observed by large margins

---

## Structural Assessment

### Prior-Likelihood Compatibility
The model structure itself is appropriate:
- Quadratic form on log scale allows flexible growth curves
- Negative binomial handles overdispersion in count data
- No structural conflicts between priors and likelihood

### Root Cause of Issues
The problem is **prior scale**, not model structure:
1. **β₂ prior too wide:** σ=0.2 allows β₂ values up to 0.83, which on squared year values (~2.8 at extremes) creates explosive growth
2. **Quadratic amplification:** Even moderate β₂ values combined with year²≈2.8 produce large log(μ) changes
3. **Exponential explosion:** exp(β₀ + β₁·year + β₂·year²) converts linear changes in log scale to exponential changes in count scale

**Mathematical demonstration:**
- At year=1.67: year²=2.79
- With β₀=5.5, β₁=1.2, β₂=0.6 (all within prior range):
  - log(μ) = 5.5 + 1.2(1.67) + 0.6(2.79) = 9.18
  - μ = exp(9.18) = 9,739 (far beyond observed max of 272)

---

## Decision: ADJUST

### Rationale
The priors generate scientifically implausible extreme predictions (counts >10,000) despite the observed maximum being 272. While 89.4% of simulations fall in a reasonable range, the 10.6% tail of extreme values poses:
1. **Scientific implausibility:** Counts 100× larger than observed are unrealistic
2. **Computational risk:** Extreme values may cause MCMC sampling issues
3. **Inefficiency:** Prior mass in implausible regions wastes sampling effort

The priors are **too vague**, not fundamentally misspecified. The model structure is appropriate.

---

## Recommended Prior Adjustments

### Proposed New Priors

```
β₀ ~ Normal(4.7, 0.3)    # Tightened from 0.5
β₁ ~ Normal(0.8, 0.2)    # Tightened from 0.3
β₂ ~ Normal(0.3, 0.1)    # Tightened from 0.2 (critical change)
φ ~ Gamma(2, 0.5)        # Keep as is
```

### Justification

1. **β₀ (Intercept):** Reduce σ from 0.5→0.3
   - Current: exp(4.7±1.0) = [40, 442] for 2σ range
   - Proposed: exp(4.7±0.6) = [60, 242] for 2σ range
   - Better matches observed range [19, 272]

2. **β₁ (Linear term):** Reduce σ from 0.3→0.2
   - Maintains positive growth but reduces extreme linear trends
   - 95% prior mass: [0.4, 1.2] vs previous [0.2, 1.4]

3. **β₂ (Quadratic term) - CRITICAL:** Reduce σ from 0.2→0.1
   - This is the key change to prevent exponential explosion
   - 95% prior mass: [0.1, 0.5] vs previous [-0.1, 0.7]
   - Still allows substantial curvature but prevents extreme acceleration

4. **φ (Overdispersion):** No change needed
   - Current prior behaves well (mean≈4, handles observed overdispersion)

### Expected Impact

With tightened priors:
- Expected max at year=1.67: median ~400-600 instead of ~978
- Upper 90% CI: ~1500 instead of ~4250
- Simulations >10,000: expect <1% instead of 2.1%
- Coverage of observed range: should improve while reducing extreme tail

---

## Next Steps

1. **Re-run prior predictive check** with adjusted priors
2. **Verify improvements:**
   - Max simulated means should rarely exceed 1000
   - Late time point median μ should be closer to observed maximum
   - 95%+ of simulations should have means in [20, 500]
3. **If second attempt passes:** Proceed to SBC validation
4. **If still problematic:** Consider alternative parameterizations (e.g., log-year to reduce quadratic effect)

---

## Technical Notes

**Implementation:** NumPy/SciPy (CmdStan unavailable in environment)
- 1000 prior samples generated
- No convergence issues
- All computations completed successfully

**Files:**
- Code: `/workspace/experiments/experiment_1/prior_predictive_check/code/run_prior_check_numpy.py`
- Plots: `/workspace/experiments/experiment_1/prior_predictive_check/plots/*.png` (6 plots)

---

## Conclusion

The Negative Binomial Quadratic model has appropriate structure for the data, but the current priors are too vague, allowing scientifically implausible extreme predictions. The primary issue is the quadratic coefficient prior (β₂) being too wide, which combined with the exponential link function creates explosive growth at late time points.

**Recommendation:** Tighten all coefficient priors (especially β₂: 0.2→0.1) and re-run prior predictive check before proceeding to SBC.

**Status:** Model is viable with adjusted priors - do not skip this model class.
