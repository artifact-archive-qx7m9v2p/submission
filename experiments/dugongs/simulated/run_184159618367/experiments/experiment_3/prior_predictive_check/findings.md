# Prior Predictive Check Findings: Log-Log Power Law Model

**Date**: 2025-10-27
**Experiment**: Experiment 3 - Log-Log Power Law Model
**Analyst**: Bayesian Model Validator

---

## Visual Diagnostics Summary

All plots are located in `/workspace/experiments/experiment_3/prior_predictive_check/plots/`:

1. **parameter_plausibility.png** - Marginal prior distributions for α, β, σ
2. **prior_predictive_coverage.png** - Prior predictive trajectories overlaid with observed data
3. **behavior_diagnostics.png** - Monotonicity, distributional properties, parameter relationships
4. **heavy_tail_diagnostics.png** - Identification of problematic prior components
5. **pointwise_plausibility.png** - Pointwise assessment across x range

---

## Executive Summary

**RECOMMENDATION: REVISE PRIORS**

The current prior specification generates scientifically plausible predictions in the **majority of cases (89% of individual predictions are plausible)**, but exhibits **three critical issues** that warrant prior revision before proceeding to simulation-based calibration:

1. **Heavy-tailed σ prior** (5.7% of draws > 1.0) causes extreme predictions
2. **Negative β values** (11.8% of draws) create decreasing trends inconsistent with theory
3. **Low trajectory-level pass rate** (62.8%) fails the 80% threshold

While not fundamentally broken, these issues indicate the priors are **too permissive** and would benefit from tightening to better encode domain knowledge.

---

## Detailed Assessment

### 1. Prior Parameter Distributions

**Visual Evidence**: `parameter_plausibility.png`

#### α (Intercept on log scale) - GOOD ✓
- **Specification**: Normal(0.6, 0.3)
- **Observed behavior**: Mean=0.614, SD=0.296
- **Domain alignment**: Centers around log(1.8), well-aligned with observed log(Y) range [0.54, 0.97]
- **Assessment**: **Well-calibrated**. Covers plausible range without excessive spread.

#### β (Power law exponent) - PROBLEMATIC ⚠
- **Specification**: Normal(0.12, 0.1)
- **Observed behavior**: Mean=0.119, SD=0.101
- **Issue**: **11.8% of draws are negative**, implying decreasing power laws
- **Scientific implausibility**: The observed data shows a clear increasing trend. Negative β values contradict this.
- **Impact**: Visible in `behavior_diagnostics.png` Panel A - only 0.3% of trajectories are monotonically increasing (though this includes flat trends)
- **Assessment**: **Needs tightening**. Consider truncated normal or shift to β ~ Normal(0.12, 0.05) to reduce negative tail.

#### σ (Residual SD on log scale) - PROBLEMATIC ⚠⚠
- **Specification**: Half-Cauchy(0, 0.1)
- **Observed behavior**: Mean=0.471, Median much lower, 95th percentile=1.153
- **Issue**: **5.7% of draws exceed 1.0**, causing extreme variability
- **Impact on predictions**:
  - When σ > 1.0, predictions can span orders of magnitude (see `heavy_tail_diagnostics.png` Panel B)
  - Maximum predicted Y reached 1.99e41 (effectively infinite)
  - Creates 0.57% of predictions > 100 and 0.33% < 0.01
- **Assessment**: **Heavy tail too extreme**. Half-Cauchy(0, 0.1) is too permissive for log-scale residuals.

---

### 2. Prior Predictive Coverage

**Visual Evidence**: `prior_predictive_coverage.png`, `pointwise_plausibility.png`

#### Plausibility Bounds
- **Criteria**: Y should remain in [0.5, 5.0] (observed range [1.71, 2.63] with buffer)
- **Results**:
  - **Overall**: 89.0% of all 200,000 predictions fall in plausible range
  - **Trajectory-level**: Only 62.8% of 2,000 trajectories are *entirely* plausible
  - **Pointwise**: Ranges from 95.1% (at x=1) to 84.9% (at x=30)

#### Interpretation
The pointwise assessment (`pointwise_plausibility.png` top panel) shows that at any given x value, the vast majority (85-95%) of predictions are plausible. However, when we require an *entire trajectory* to be plausible across all x values, the pass rate drops to 62.8%.

**This indicates**: Extreme values are relatively rare at any single point, but when sampling 100 prediction points per trajectory, the probability that at least one point is extreme becomes substantial.

#### Coverage of Observed Data
The prior median (2.51) is well-centered on the observed data range [1.71, 2.63]. The 95% prior predictive interval [0.77, 8.48] provides:
- **Lower coverage**: Extends to 0.77 (below observed minimum of 1.71) ✓
- **Upper coverage**: Extends to 8.48 (well above observed maximum of 2.63) ✓

This is appropriate - priors should be wider than the observed data to allow for uncertainty.

---

### 3. Monotonicity and Shape

**Visual Evidence**: `behavior_diagnostics.png` Panel A & C, `heavy_tail_diagnostics.png` Panel C

#### Power Law Behavior
The power law model assumes Y = exp(α) × x^β, which should be:
- **Monotonic increasing** when β > 0
- **Monotonic decreasing** when β < 0
- **Constant** when β = 0

#### Observed Issues
- **Only 0.3% of trajectories are strictly monotonically increasing** (Panel A)
- This is because:
  1. 11.8% have β < 0 (inherently decreasing)
  2. The stochastic noise from σ creates local fluctuations even when β > 0

#### Is this a problem?
**Partially**. The model *on average* captures increasing trends:
- Median growth from x=1 to x=30 is 1.50× (reasonable for β ≈ 0.12)
- Expected growth 30^β has median 1.50× (consistent)

However, individual realizations with negative β are **scientifically implausible** given the observed positive trend.

---

### 4. Problem Identification: Heavy Tails

**Visual Evidence**: `heavy_tail_diagnostics.png`

#### Panel A: Half-Cauchy Heavy Tail
The log-scale histogram reveals extreme σ values:
- 5.7% of draws exceed σ = 1.0
- Maximum σ = 132.076 (absurdly large)
- The Half-Cauchy distribution with scale=0.1 has no finite variance, leading to occasional extreme draws

#### Panel B: Impact of Large σ
When σ > 1.0 (red curves), predictions become wildly variable:
- Extreme upward spikes extending beyond visualization range (clipped at Y=50)
- This occurs because on the log scale, large σ means exp(Normal(μ, σ)) has exponentially large variance

#### Panel C: Impact of Negative β
When β < 0 (red curves), trajectories are decreasing:
- Contradicts the observed increasing trend in the data
- 11.8% of prior samples exhibit this behavior

#### Panel D: Joint Distribution
The (β, σ) joint distribution shows:
- Most density concentrated near (β=0.12, σ=0.1) ✓
- But non-negligible mass in problematic regions (β<0, σ>0.3)

---

### 5. Computational Health

**Visual Evidence**: Numerical output from prior predictive check

#### Red Flags
- **No NaN or Inf values** in parameters ✓
- **No NaN or Inf values** in predictions ✓
- **BUT**: Maximum prediction reached 1.99×10^41
  - This suggests extreme but not numerically broken behavior
  - Caused by large σ values creating exp(very large numbers)

#### Numerical Stability
While the priors don't cause numerical instability (NaN/Inf), they do create:
- **0.57% of predictions > 100** - computationally fine but scientifically absurd for Y ∈ [1.71, 2.63]
- **0.33% of predictions < 0.01** - near-zero values

These extreme values won't break Stan but will:
1. Slow down sampling (exploration of extreme parameter regions)
2. Potentially cause divergences if likelihood strongly rejects these regions
3. Reduce effective sample size

---

## Key Visual Evidence

### Most Important Plots for Decision

1. **prior_predictive_coverage.png**: Shows that 62.8% trajectory pass rate fails the 80% criterion
   - Clear visual evidence of many red (implausible) trajectories
   - Median prediction well-aligned with data, but tails are problematic

2. **heavy_tail_diagnostics.png**: Identifies the root causes
   - Panel A: σ heavy tail clearly visible on log scale
   - Panel C: Negative β problem creates decreasing trends

3. **pointwise_plausibility.png**: Shows the problem is worse at extrapolation range
   - Pass rate decreases from 95% at x=1 to 85% at x=30
   - Indicates tail issues compound over the x range

---

## Recommendations for Prior Revision

### Option 1: Minimal Adjustment (Recommended)
Tighten the problematic priors while keeping the general structure:

```
α ~ Normal(0.6, 0.3)           # Keep as is - well calibrated
β ~ Normal(0.12, 0.05)         # Reduce SD: 0.1 → 0.05 (fewer negative values)
σ ~ Half-Cauchy(0, 0.05)       # Reduce scale: 0.1 → 0.05 (reduce heavy tail)
```

**Expected impact**:
- β: Negative values drop from 11.8% to ~0.8%
- σ: Values > 1.0 drop from 5.7% to ~0.5%
- Trajectory pass rate increases to ~85-90%

### Option 2: More Conservative
Use truncated or alternative distributions:

```
α ~ Normal(0.6, 0.3)           # Keep
β ~ TruncatedNormal(0.12, 0.1, lower=0)  # Force positive
σ ~ Exponential(1/0.1)         # Lighter tail than Half-Cauchy
```

**Expected impact**:
- Eliminates all negative β values by construction
- Exponential prior on σ has finite mean/variance (lighter tail)
- Trajectory pass rate likely >90%

### Option 3: Data-Driven (If EDA available)
If you have preliminary OLS log-log fit estimates:

```
α ~ Normal(α_ols, 2 × SE_α)    # Center on OLS estimate
β ~ Normal(β_ols, 2 × SE_β)    # Center on OLS estimate
σ ~ Half-Normal(0, s_residual) # Based on residual SD
```

This would be most informative but requires running the OLS first.

---

## Scientific Plausibility Assessment

### Domain Constraints
- **Y must be positive**: ✓ All predictions positive (log-normal ensures this)
- **Y should be in [1, 4] range**: ⚠ 11% of predictions violate this
- **Power law should be increasing**: ⚠ 11.8% of trajectories decreasing

### Scale Reasonableness
- **Median predictions**: Well-aligned with data ✓
- **Prediction spread**: 95% interval [0.77, 8.48] is reasonable but wide ✓
- **Extreme values**: <1% but problematic ⚠

### Structural Appropriateness
The log-log model structure is sound:
- Multiplicative errors appropriate for positive-valued data ✓
- Power law relationship theoretically justified ✓
- Linear model in log-log space computationally efficient ✓

**The issue is not the model structure but the prior parameterization**.

---

## Final Recommendation

### Decision: REVISE PRIORS

**Rationale**:
1. **Trajectory pass rate of 62.8% fails the 80% threshold**
2. **Two identifiable prior components need tightening** (β and σ)
3. **Revisions are straightforward** and will substantially improve plausibility
4. **No fundamental model misspecification** - structure is sound

### Recommended Action Plan
1. **Implement Option 1** (minimal adjustment): Reduce SD of β to 0.05 and scale of σ to 0.05
2. **Re-run prior predictive check** with revised priors
3. **If trajectory pass rate >80%**, proceed to SBC
4. **If still <80%**, implement Option 2 (truncated/alternative distributions)

### Why Not Proceed Anyway?
While 89% of individual predictions are plausible, the **trajectory-level failure rate of 37.2% is too high**. This means:
- **SBC will waste computation** exploring implausible parameter regions
- **Posterior might be sensitive** to the prior tail behavior
- **Results harder to interpret** if prior dominates in tail regions

**Fixing the priors now is cheap (5 minutes) compared to debugging posterior issues later.**

---

## Appendix: Quantitative Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Trajectory pass rate | 62.8% | >80% | ⚠ FAIL |
| Pointwise plausibility | 89.0% | >80% | ✓ PASS |
| Numerical issues | 0.0% | <1% | ✓ PASS |
| Negative β | 11.8% | <5% | ⚠ FAIL |
| Extreme σ (>1.0) | 5.7% | <5% | ⚠ FAIL |
| Extreme predictions (>100) | 0.57% | <1% | ✓ PASS |
| Extreme predictions (<0.01) | 0.33% | <1% | ✓ PASS |

### Parameter Statistics

| Parameter | Prior | Mean | SD | 5th %ile | 95th %ile |
|-----------|-------|------|----|----|-----|
| α | N(0.6, 0.3) | 0.614 | 0.296 | 0.131 | 1.106 |
| β | N(0.12, 0.1) | 0.119 | 0.101 | -0.045 | 0.286 |
| σ | HC(0, 0.1) | 0.471 | 3.711 | 0.000 | 1.153 |

### Prediction Statistics (Overall)

| Statistic | Value |
|-----------|-------|
| Median | 2.51 |
| Mean | 2.89 |
| 2.5th percentile | 0.77 |
| 97.5th percentile | 8.48 |
| Min | 0.000 |
| Max | 1.99×10^41 |

---

## Files Generated

### Code
- `/workspace/experiments/experiment_3/prior_predictive_check/code/prior_predictive_check.py` - Main analysis
- `/workspace/experiments/experiment_3/prior_predictive_check/code/visualize_priors.py` - Visualization generation
- `/workspace/experiments/experiment_3/prior_predictive_check/code/prior_samples.csv` - 2000 prior parameter draws
- `/workspace/experiments/experiment_3/prior_predictive_check/code/prior_predictions.npz` - Prior predictive samples

### Plots
- `/workspace/experiments/experiment_3/prior_predictive_check/plots/parameter_plausibility.png`
- `/workspace/experiments/experiment_3/prior_predictive_check/plots/prior_predictive_coverage.png`
- `/workspace/experiments/experiment_3/prior_predictive_check/plots/behavior_diagnostics.png`
- `/workspace/experiments/experiment_3/prior_predictive_check/plots/heavy_tail_diagnostics.png`
- `/workspace/experiments/experiment_3/prior_predictive_check/plots/pointwise_plausibility.png`

---

**Conclusion**: The Log-Log Power Law Model has sound structure and mostly reasonable priors, but requires minor tightening of β and σ priors to meet the 80% trajectory plausibility threshold. Revise priors as recommended and re-validate before proceeding to SBC.
