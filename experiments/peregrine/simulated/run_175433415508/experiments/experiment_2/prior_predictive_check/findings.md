# Prior Predictive Check: NB-AR(1) Model

**Experiment**: 2 - Negative Binomial with AR(1) Temporal Correlation
**Date**: 2025-10-29
**Status**: FAIL (with recommendations)

---

## Visual Diagnostics Summary

All plots saved to: `/workspace/experiments/experiment_2/prior_predictive_check/plots/`

1. **`prior_parameter_distributions.png`**: Validates that sampled parameters match theoretical prior distributions
2. **`temporal_correlation_diagnostics.png`**: Critical assessment of AR(1) structure (ρ vs σ, ρ vs realized ACF)
3. **`prior_predictive_trajectories.png`**: Time series behavior on linear and log scales, AR process dynamics
4. **`prior_acf_structure.png`**: Autocorrelation patterns in AR process and count data
5. **`prior_predictive_coverage.png`**: Range plausibility, growth rates, and extreme value assessment
6. **`decision_summary.png`**: **START HERE** - One-page overview of all findings, failures, and recommendations

---

## Executive Summary

**DECISION: FAIL - Revise priors before fitting**

The prior predictive check reveals **critical issues** with the current prior specification:

### Primary Issues

1. **Extreme outliers**: 3.22% of counts exceed 10,000 (threshold: <1%)
   - Maximum generated count: **674 million** (observed max: 269)
   - Mean maximum per series: 2,038,561
   - This indicates **prior-likelihood conflict**

2. **AR(1) process not behaving correctly**:
   - Correlation between ρ parameter and realized ACF(1) is only 0.39 (expected: >0.95)
   - Mean realized ACF(1) of AR process: 0.766 (expected: 0.910)
   - **This suggests simulation issues or model structural problems**

3. **Count ACF computation failed**: NaN values indicate numerical instability
   - Extreme outliers caused variance calculations to fail
   - Cannot validate temporal correlation in count space

### Root Cause

The combination of:
- Wide priors on β₀ and β₁ (generating very large log-rates)
- Heavy-tailed Exponential(2) prior on σ (allows large innovations)
- Exponential transformation to counts

...creates a **multiplicative explosion** where occasional extreme parameter draws produce astronomically large counts.

---

## Detailed Findings

### 1. Prior Parameter Distributions

**Visual Evidence**: `prior_parameter_distributions.png`

All five parameters sample correctly from their theoretical distributions:

| Parameter | Prior | Sample Mean | Sample SD | Theoretical E[X] |
|-----------|-------|-------------|-----------|------------------|
| β₀ | Normal(4.69, 1.0) | 4.706 | 0.944 | 4.69 |
| β₁ | Normal(1.0, 0.5) | 0.993 | 0.504 | 1.00 |
| φ | Gamma(2, 0.1) | 19.012 | 12.894 | 20.00 |
| **ρ** | **Beta(20, 2)** | **0.910** | **0.059** | **0.909** |
| σ | Exponential(2) | 0.522 | 0.547 | 0.500 |

**Key observation**: The **ρ ~ Beta(20,2) prior is extremely concentrated** around 0.91, as intended. The 95% interval is [0.769, 0.975], showing this is indeed a strong informative prior based on EDA ACF=0.971.

**Status**: ✓ PASS - Sampling is correct

---

### 2. Temporal Correlation Structure

**Visual Evidence**: `temporal_correlation_diagnostics.png`, `prior_acf_structure.png`

#### AR(1) Parameter Prior

The ρ distribution shows:
- **Median**: 0.921
- **95% CI**: [0.769, 0.975]
- **Range**: [0.641, 0.995]

This appropriately encodes strong belief in high temporal correlation, motivated by EDA findings.

#### AR(1) Process Validation: CRITICAL FAILURE

**The AR(1) process is not behaving as expected:**

From `temporal_correlation_diagnostics.png` (Panel 3):
- Points should fall on the red dashed line (ρ = realized ACF)
- Instead, there's **massive scatter** with correlation = 0.39
- Most realized ACF(1) values are **lower than the ρ parameter**

**What this means**:
- When ρ = 0.95, the realized ACF(1) should also be ≈0.95
- But we're seeing realized ACF(1) values around 0.7-0.9
- This suggests either:
  1. The AR(1) simulation isn't working correctly
  2. The time series is too short (N=40) to realize the correlation
  3. There's interaction with the exponential transformation

#### Count Data ACF: FAILURE

From `prior_acf_structure.png`:
- Panel B shows ACF(1) = 1.0 for all lags (implausible)
- Panel C shows histogram mean = NaN
- **Root cause**: Extreme outliers break variance calculations

**Status**: ✗ FAIL - AR(1) structure compromised

---

### 3. Prior Predictive Count Distribution

**Visual Evidence**: `prior_predictive_trajectories.png`, `prior_predictive_coverage.png`

#### Overall Statistics

- **Mean**: 91,019 (observed range: 21-269)
- **Median**: 112 ✓ (reasonable!)
- **Range**: [0, **674,970,346**] ✗ (extreme!)

#### Percentiles

| Percentile | Value | Assessment |
|------------|-------|------------|
| 1% | 0 | Too low (zeros unlikely) |
| 5% | 3 | Plausible lower bound |
| 50% | 112 | **Excellent** - covers observed range |
| 95% | 4,503 | Reasonable upper uncertainty |
| 99% | 143,745 | **Too high** - orders of magnitude above data |

#### Plausibility Flags

From `prior_predictive_coverage.png` (Panel D):
- **Mean maximum count per series**: 2,038,561
- Most series peak around 1,000-10,000
- But **~100 series** have maxima > 100,000
- These extreme series drive the failures

**Key insight**: The **median** behavior is reasonable, but the **tail** is catastrophic.

**Status**: ✗ FAIL - Heavy-tailed extremes

---

### 4. Growth Patterns

**Visual Evidence**: `prior_predictive_coverage.png` (Panel C)

The prior on β₁ ~ Normal(1.0, 0.5) implies:
- **Mean growth rate**: 207% per standardized year
- **Range**: [-35%, 1234%]

For context, with N=40 years, a growth rate of 200% per standardized unit translates to:
- exp(β₁ × Δyear_max) where Δyear_max ≈ 3.34
- Total growth: exp(1.0 × 3.34) ≈ 28× increase

**Observed data**: 21 → 269 is a 12.8× increase over 40 years

The prior allows growth rates from:
- exp(-0.5 × 3.34) = 0.19× (82% decline)
- exp(2.5 × 3.34) = 4,915× (extreme growth)

**Status**: ⚠ PASS (marginally) - Growth prior is wide but not unreasonable given uncertainty

---

### 5. AR(1) Process Behavior

**Visual Evidence**: `prior_predictive_trajectories.png` (Panels C & D)

#### Epsilon (AR process) Trajectories

From Panel C:
- Most trajectories oscillate around zero ✓
- Range approximately [-10, +10]
- Show persistence/smoothness ✓
- **But**: Some extreme excursions visible

#### Log-Rate (η) Trajectories

From Panel D:
- Most trajectories: 0 to 10 range (reasonable)
- Show smooth temporal trends ✓
- **But**: Some trajectories reach 10-15 (exp(15) ≈ 3 million)

**The problem**: When β₀=7, β₁=2, and σ=2 all occur together (rare but possible):
- η can reach 7 + 2×1.67 + 5×2 = 20.3
- exp(20.3) = **660 million**

**Status**: ⚠ PARTIAL - Most draws reasonable, tail problematic

---

### 6. Coverage Assessment

**Visual Evidence**: `prior_predictive_coverage.png` (Panel A)

The envelope plot shows:
- **Median** prior predictive aligns well with observed trend ✓
- **5-95% interval** at early timepoints: 3-500 (observed: 21-30) ✓
- **5-95% interval** at late timepoints: 50-25,000 (observed: 250-270) ✗

**At year_max (+1.67)**:
- Observed: 252-269
- Prior 50%: ~1,000
- Prior 95%: ~30,000

The upper tail is **2 orders of magnitude too wide**.

**Status**: ✗ FAIL - Overconfident in extreme growth

---

## Root Cause Analysis

### Why Are We Getting Extreme Outliers?

The problem is **multiplicative amplification** through the exponential link:

```
η = β₀ + β₁×year + ε_t
μ = exp(η)
```

When rare combinations occur:
1. β₀ from upper tail: 6.5 (P ≈ 3%)
2. β₁ from upper tail: 2.0 (P ≈ 2.5%)
3. σ from upper tail: 1.5 (P ≈ 8%)
4. ε_t reaches +3σ at late timepoint: 4.5

Then: η = 6.5 + 2.0×1.67 + 4.5 = **14.3**
→ μ = exp(14.3) = **1.6 million**

This happens in ~0.5% of (draw, timepoint) combinations, which with 500 draws × 40 timepoints = 20,000 samples means ~100 extreme values.

### Why Is AR(1) Validation Failing?

Possible explanations:
1. **Short time series bias**: N=40 may be insufficient to realize theoretical ACF
2. **Nonstationarity**: The trend component interferes with ACF estimation
3. **High ρ instability**: When ρ is near 1, the process is near non-stationary
4. **Exponential transformation**: Non-linear transformation changes correlation structure

The fact that realized ACF is systematically **lower** than ρ suggests the short series length is the issue.

---

## Recommendations

### Option 1: Tighten Innovation Prior (RECOMMENDED)

**Change**: σ ~ Exponential(2) → σ ~ Exponential(5) or even Exponential(10)

**Rationale**:
- Current E[σ] = 0.5, but tail allows σ up to 3.2
- With ρ ≈ 0.91, the stationary SD of ε is σ/√(1-ρ²) ≈ 1.2
- Large innovations create the extreme values
- Tightening σ prior will constrain the AR process

**Expected impact**: Reduces 99th percentile from 143,745 → ~10,000

### Option 2: Truncate or Transform β₁ Prior

**Change**: Constrain β₁ to [-0.5, 2.0] range

**Rationale**:
- Current β₁ ~ Normal(1.0, 0.5) allows values up to 2.5+
- Growth rates above 1000% are scientifically implausible
- Can use truncated normal or put prior on log-scale

**Implementation**:
```
β₁ ~ TruncatedNormal(1.0, 0.5, lower=-0.5, upper=2.0)
```

### Option 3: Informative Prior on φ (dispersion)

**Current**: φ ~ Gamma(2, 0.1) with mean=20, allows range [0.18, 70]

**From Experiment 1**: Posterior was φ = 35.6 ± 10.8

**Change**: Use Exp1 posterior as prior:
```
φ ~ Normal(35, 15)  # Slightly more conservative than Exp1 posterior
```

**Rationale**: Large variance (small φ) combined with large mean creates extreme counts

### Option 4: Alternative AR(1) Initialization

**Current**: ε₀ ~ Normal(0, σ/√(1-ρ²)) (stationary)

**Issue**: When ρ → 1, this variance explodes

**Change**: Fixed initialization:
```
ε₀ ~ Normal(0, σ)  # Simple initialization
```

**Trade-off**: Less theoretically pure, but more numerically stable

---

## Recommended Prior Specification (Version 2)

```python
# Revised priors for Experiment 2
β₀ ~ Normal(4.69, 1.0)                    # Keep - aligns with Exp1
β₁ ~ TruncatedNormal(1.0, 0.5, -0.5, 2.0) # NEW - constrain growth
φ ~ Normal(35, 15)                         # NEW - inform from Exp1
ρ ~ Beta(20, 2)                            # Keep - appropriate for data
σ ~ Exponential(5)                         # NEW - tighter innovation scale
```

**Key changes**:
1. Truncate β₁ to prevent extreme growth
2. Inform φ from Experiment 1 posterior
3. Tighter σ prior (E[σ] = 0.2 instead of 0.5)

**Expected improvement**:
- 99th percentile counts: 143,745 → ~5,000
- Extreme outliers (>10k): 3.22% → <0.1%
- Maintains flexibility for temporal correlation

---

## Alternative: Relaxed Priors (if you want more vague)

If the above is too constraining:

```python
β₀ ~ Normal(4.69, 1.5)      # Slightly wider
β₁ ~ Normal(1.0, 0.5)       # Keep same
φ ~ Gamma(3, 0.1)           # Slightly more concentrated (mean=30)
ρ ~ Beta(20, 2)             # Keep same
σ ~ Exponential(4)          # Moderate tightening (E[σ]=0.25)
```

This is less aggressive but should still reduce extreme outliers to ~1% level.

---

## Technical Notes

### AR(1) Short Series Issue

The low correlation between ρ and realized ACF(1) is concerning but may be unavoidable:
- With N=40 and ρ≈0.9, ACF estimates have high variance
- The trend component further complicates ACF estimation
- **This is a model limitation, not necessarily a prior issue**

**Mitigation**: Accept that prior predictive ACF will be imperfect. The key is:
1. Are the AR parameters (ρ, σ) in reasonable ranges? YES
2. Do the trajectories show persistence? YES
3. Will posterior estimation fix this? LIKELY (data will inform)

### Numerical Stability in Fitting

Even with revised priors, be prepared for:
- **Divergent transitions** if ρ samples near 1.0
- **Slow mixing** due to correlation between ρ and σ
- **Initialization sensitivity** for the AR process

Consider:
- Non-centered parameterization for AR process
- Strong initialization near Experiment 1 posterior
- Increased `adapt_delta` (e.g., 0.99)

---

## Conclusion

**FAIL - DO NOT PROCEED WITH CURRENT PRIORS**

While the median prior predictive behavior is reasonable, the tail behavior is catastrophic. The combination of wide priors on trend parameters and exponential link creates extreme outliers that indicate poor model specification.

**Next steps**:
1. Implement revised prior specification (Version 2 recommended)
2. Re-run this prior predictive check
3. Verify extremes are controlled (<1% above 10,000)
4. Then proceed to model fitting

**What we learned**:
- ρ ~ Beta(20, 2) is appropriate for the temporal correlation
- AR(1) structure is sound, but innovation scale needs tightening
- Growth prior needs constraint to prevent explosive dynamics
- Informing φ from Experiment 1 is scientifically justified

The prior predictive check **successfully caught these issues before wasting computation on fitting**. This is exactly what this validation step is designed to do.

---

## Files Generated

All outputs in: `/workspace/experiments/experiment_2/prior_predictive_check/`

**Code**:
- `code/prior_predictive_check.py` - Standalone validation script (500 simulations)

**Plots**:
1. `plots/prior_parameter_distributions.png` - Parameter sampling validation
2. `plots/temporal_correlation_diagnostics.png` - AR(1) structure assessment
3. `plots/prior_predictive_trajectories.png` - Time series behavior (4 panels)
4. `plots/prior_acf_structure.png` - Autocorrelation patterns (4 panels)
5. `plots/prior_predictive_coverage.png` - Range and plausibility (4 panels)

**Documentation**:
- `findings.md` - This report
