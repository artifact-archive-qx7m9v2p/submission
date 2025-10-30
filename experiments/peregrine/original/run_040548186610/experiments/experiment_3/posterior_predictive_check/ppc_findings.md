# Posterior Predictive Check Findings
## Experiment 3: Latent AR(1) Negative Binomial Model

**Date:** 2025-10-29
**Model:** Latent AR(1) Negative Binomial (α_t = β₀ + β₁·year + β₂·year² + ε_t, where ε_t ~ AR(1))
**Data:** 40 observations, count range [19, 272]
**Posterior Samples:** 6,000 replications (4 chains × 1,500 draws)

---

## Executive Summary

**OVERALL FIT QUALITY: POOR**

**CRITICAL FINDING: AR(1) MODEL FAILED TO RESOLVE TEMPORAL AUTOCORRELATION**

Despite perfect convergence diagnostics (R̂=1.000, ESS>1100, ρ=0.84 [0.69, 0.98]), the Latent AR(1) model **FAILS** to address the fundamental temporal autocorrelation problem identified in Experiment 1:

### Key Failures:
1. **Residual ACF(1) = 0.690** - UNCHANGED from Exp 1 (0.686), target was < 0.3
2. **Coverage = 100%** - NO IMPROVEMENT from Exp 1 (also 100%), target was 90-98%
3. **ACF(1) Bayesian p-value = 0.000** - IDENTICAL problem as Exp 1
4. **5 extreme test statistics** - Slight improvement from 7, but still poor
5. **Strong residual temporal patterns remain** - Visible in all diagnostic plots

### The Paradox:
- The model successfully estimated **ρ = 0.84** (strong AR coefficient)
- The AR(1) structure has **σ_η = 0.09** (very small innovations)
- Convergence was **PERFECT** (no technical issues)
- Yet **residual autocorrelation is UNCHANGED**

### Interpretation:
**The AR(1) structure on the latent state is NOT capturing the temporal dependencies in the OBSERVED counts.** The model is fitting AR(1) dynamics to the wrong level (log-scale latent state vs. count-scale observations), or the temporal structure in the data is more complex than simple AR(1) can capture.

### Decision:
**POOR FIT - AR(1) INSUFFICIENT**

**Next Steps:** Need fundamentally different temporal structures:
- Observation-level AR (not latent AR)
- Higher-order AR(p) models (p > 1)
- State-space models with different dynamics
- Non-stationary models (changepoint, time-varying parameters)
- Alternative link structures

---

## Plots Generated

### Comprehensive Diagnostics
1. **ppc_dashboard.png** - 12-panel comprehensive overview showing:
   - Observed vs predicted scatter (R² = 0.861)
   - Coverage plot (100%, unchanged from Exp 1)
   - Trajectory spaghetti plot (observed still systematically smoother)
   - Distribution comparison (marginal match reasonable)
   - Residual diagnostics showing U-shaped patterns
   - **Panel G: Residual ACF showing NO IMPROVEMENT** (0.690 vs 0.686)
   - **Panel K: ACF(1) test statistic still extreme** (p=0.000)

2. **acf_comparison_exp1_vs_exp3.png** - **CRITICAL COMPARISON** showing:
   - Side-by-side ACF plots for both experiments
   - Exp 1 ACF(1) = 0.686, Exp 3 ACF(1) = 0.690 (essentially identical)
   - Both far above target threshold (0.3) and Phase 2 trigger (0.5)
   - **Visual proof that AR(1) did not solve the problem**

3. **coverage_detailed.png** - Detailed coverage showing:
   - All 40 observations within 95% interval (100% coverage)
   - No improvement from Experiment 1
   - Still over-covering (too conservative)

4. **residual_diagnostics.png** - 6-panel residual suite revealing:
   - **Panel A: U-shaped pattern vs fitted** (systematic bias)
   - **Panel B: Strong temporal wave pattern** (U-shape over time)
   - **Panel C: ACF(1) = 0.690, unchanged from baseline**
   - Panel D: Residual distribution approximately normal
   - Panel E: Q-Q plot shows reasonable normality
   - Panel F: Scale-location shows heteroscedasticity (wave pattern)

5. **test_statistics.png** - Test statistics showing:
   - ACF(1) still extreme (p = 0.000)
   - Skewness, kurtosis still problematic
   - Mean and variance still well-captured
   - Maximum still extreme (p = 0.952)

---

## Visual Diagnosis Summary

| Aspect Tested | Plot File | Finding | Implication |
|---------------|-----------|---------|-------------|
| Overall prediction accuracy | ppc_dashboard.png (Panel A) | R² = 0.861 (was 0.883 in Exp 1) | Slightly worse mean fit |
| Uncertainty calibration | coverage_detailed.png | 100% coverage (unchanged from Exp 1) | **NO IMPROVEMENT** |
| Temporal patterns | ppc_dashboard.png (Panel C) | Observed still smoother than replications | **AR(1) not capturing dynamics** |
| Residual independence | acf_comparison_exp1_vs_exp3.png | ACF(1) = 0.690 (was 0.686) | **FAILED: NO IMPROVEMENT** |
| Residual homoscedasticity | residual_diagnostics.png (Panel A) | U-shaped pattern remains | Systematic bias unchanged |
| Time series structure | residual_diagnostics.png (Panel B) | **Strong U-shaped temporal pattern** | **Critical: temporal structure remains** |
| Extreme value behavior | test_statistics.png (Max panel) | p = 0.952 (was 0.994) | Minor improvement, still poor |
| Distribution shape | test_statistics.png | Skewness p=0.993, Kurtosis p=0.999 | Shape mismatch unchanged |
| Autocorrelation | test_statistics.png (ACF(1)) | **p = 0.000 (identical to Exp 1)** | **Most severe failure** |

**KEY VISUAL EVIDENCE:** `acf_comparison_exp1_vs_exp3.png` shows both experiments have nearly identical residual ACF patterns, proving the AR(1) structure did not resolve temporal dependence.

---

## Coverage Analysis

### Empirical Coverage Rates

| Prediction Interval | Expected | Observed (Exp 3) | Observed (Exp 1) | Change |
|---------------------|----------|------------------|------------------|--------|
| 50% PI | 50% | 75.0% (30/40) | 67.5% (27/40) | **+7.5%** (worse) |
| 80% PI | 80% | 97.5% (39/40) | 95.0% (38/40) | **+2.5%** (worse) |
| 95% PI | 95% | 100.0% (40/40) | 100.0% (40/40) | **No change** |

**Coverage Quality: POOR** (100% far exceeds 95%, no improvement)

### Coverage Findings

**All observations fall within 95% prediction intervals**, identical to Experiment 1. The AR(1) model did NOT improve calibration.

**Interpretation:**
- Adding AR(1) structure increased uncertainty without improving fit
- Prediction intervals still too wide (conservative)
- The model is uncertain about the wrong things
- Coverage remains excessive because the model doesn't understand the temporal dynamics

**Evidence:** `coverage_detailed.png` shows all red observed points comfortably within blue prediction bands, with no excursions outside 95% intervals. This is identical to Experiment 1's pattern.

**Comparison to Experiment 1:**
- 50% coverage WORSE (75.0% vs 67.5%)
- 80% coverage WORSE (97.5% vs 95.0%)
- 95% coverage UNCHANGED (100% vs 100%)

**Conclusion:** AR(1) model increased uncertainty without capturing structure, making calibration worse.

---

## Residual Diagnostics

### Residual Summary Statistics

- **Mean:** -9.87 (more biased than Exp 1: -4.21)
- **Std:** 39.81 (larger than Exp 1: 34.44)
- **Range:** [-175.01, 37.63] (wider than Exp 1)
- **Pattern:** Still approximately normal in distribution (Q-Q plot Panel E)

**Comparison to Exp 1:** Residuals are WORSE - larger bias, larger variance, wider range.

### Residual Autocorrelation (CRITICAL FAILURE)

**Residual ACF values** (evident in `residual_diagnostics.png` Panel C and `acf_comparison_exp1_vs_exp3.png`):

| Lag | Exp 3 (AR(1)) | Exp 1 (Baseline) | Change | Target |
|-----|---------------|------------------|--------|--------|
| 1 | **0.690** | 0.686 | **+0.004** | < 0.3 |
| 2 | 0.432 | 0.423 | +0.009 | - |
| 3 | 0.257 | 0.243 | +0.014 | - |

**CRITICAL FINDING: The AR(1) model achieved ZERO improvement in residual autocorrelation.**

- ACF(1) = 0.690 vs. 0.686 (0.6% INCREASE, not decrease!)
- Still far above target threshold (0.3)
- Still far above Phase 2 trigger (0.5)
- All higher lags also unchanged

**Decision: POOR - AR(1) FAILED TO RESOLVE TEMPORAL STRUCTURE**

**Visual Evidence:**
1. **`acf_comparison_exp1_vs_exp3.png`** - Side-by-side comparison shows virtually identical ACF patterns between experiments. The orange bars (Exp 1) and green bars (Exp 3) are nearly overlapping at all lags.
2. **`residual_diagnostics.png` Panel C** - Shows ACF(1) = 0.690 well above green target line (0.3) and orange Exp 1 line (0.686).
3. **`ppc_dashboard.png` Panel G** - Shows residual ACF has not improved from Exp 1 baseline.

### Residual Patterns (UNCHANGED FROM EXP 1)

**Pattern vs Fitted Values** (`residual_diagnostics.png` Panel A):
- **U-shaped nonlinear pattern PERSISTS** (green smooth trend line)
- Positive residuals at low fitted values
- Negative residuals at middle fitted values
- Large negative residuals at high fitted values
- **IDENTICAL pattern to Experiment 1**

**Pattern vs Time** (`residual_diagnostics.png` Panel B):
- **Strong U-shaped temporal pattern PERSISTS**
- Smooth trend shows clear systematic structure
- Large negative residuals at end of series (time 35-40)
- Positive residuals in early/middle periods
- **Wave pattern UNCHANGED from Experiment 1**

**Scale-Location** (`residual_diagnostics.png` Panel F):
- **Wave pattern in variance** (smooth red trend oscillates)
- Heteroscedasticity remains
- Not constant variance over fitted values

**Implication:** The AR(1) structure on the latent log-scale process does NOT translate to temporal correlation at the observation level. The residuals (observed - predicted) still show strong temporal dependence, meaning the model is missing something fundamental about the temporal dynamics.

---

## Test Statistics and Bayesian P-Values

### Overview

Computed 13 test statistics comparing observed vs 6,000 replicated datasets. Bayesian p-value = P(T_rep ≥ T_obs), where extreme values (< 0.05 or > 0.95) indicate poor fit.

### Problematic Statistics (p < 0.05 or p > 0.95)

| Statistic | Observed | P-value (Exp 3) | P-value (Exp 1) | Change | Assessment |
|-----------|----------|-----------------|-----------------|--------|------------|
| **ACF(1)** | **0.944** | **0.000***  | **0.000*** | **None** | **FAILED** |
| **Kurtosis** | -1.23 | 0.999*** | 1.000*** | Tiny | POOR |
| **Skewness** | 0.60 | 0.993*** | 0.999*** | Tiny | POOR |
| **Range** | 253 | 0.952*** | 0.995*** | Better | POOR |
| **Max** | 272 | 0.952*** | 0.994*** | Better | POOR |
| **IQR** | 160.75 | 0.089 | 0.017*** | **Fixed!** | GOOD |
| **Q75** | 195.50 | 0.118 | 0.020*** | **Fixed!** | GOOD |

**Summary:** 5 problematic statistics (down from 7 in Exp 1)
- **ACF(1): UNCHANGED** - still most severe problem (p = 0.000)
- **Kurtosis, skewness: UNCHANGED** - distribution shape still mismatched
- **Max, range: SLIGHTLY BETTER** - but still in extreme tail
- **IQR, Q75: FIXED** - no longer extreme (minor improvement)

### Well-Fitting Statistics (0.1 < p < 0.9)

| Statistic | Observed | P-value (Exp 3) | P-value (Exp 1) | Assessment |
|-----------|----------|-----------------|-----------------|------------|
| Mean | 109.45 | 0.612 | 0.668 | GOOD |
| Variance | 7441.74 | 0.772 | 0.910 | GOOD |
| Std | 86.27 | 0.772 | 0.910 | GOOD |
| Min | 19.00 | 0.445 | 0.244 | GOOD |
| Q25 | 34.75 | 0.784 | 0.776 | GOOD |
| Q50 | 74.50 | 0.418 | 0.371 | GOOD |

### Interpretation

**The AR(1) model made NO MEANINGFUL IMPROVEMENT:**

1. **Temporal structure (ACF(1) p = 0.000)**: IDENTICAL FAILURE to Exp 1. The observed ACF(1) of 0.944 is still far outside what the model can generate. This is the most critical finding. Evidence in `test_statistics.png` Panel K shows observed value (red line) in extreme right tail of distribution, with p-value annotation showing 0.000.

2. **Distribution shape (skewness p = 0.993, kurtosis p = 0.999)**: UNCHANGED. The model still cannot match the observed distribution shape. Minor improvement (p-values slightly less extreme) but still in extreme tails.

3. **Extreme values (max p = 0.952, range p = 0.952)**: SLIGHT IMPROVEMENT from Exp 1, but still problematic. The model still struggles to generate maximums as large as 272.

4. **Quantile behavior**: IQR and Q75 improved from extreme to healthy range - the only success story, but minor.

5. **Central tendency**: Mean, variance, median remain well-captured (as in Exp 1).

**Pattern:** The AR(1) model preserved Exp 1's strengths (central tendency) but FAILED to address its critical weakness (temporal dependence).

---

## Why Did the AR(1) Model Fail?

This is the key scientific question. The model converged perfectly and estimated strong temporal correlation (ρ = 0.84), yet residual autocorrelation is unchanged. Why?

### Evidence from Model Structure

**From `model.stan`:**
```stan
// Observation model
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = α_t

// Latent AR(1) state
α_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
ε_t = ρ·ε_{t-1} + η_t
η_t ~ Normal(0, σ_η)
```

**Key insight:** The AR(1) structure is on **α_t (log-scale)**, not on the observed counts **C_t** directly.

### Parameter Estimates Revealing the Problem

From Experiment 3 posterior inference:
- **ρ = 0.84** [0.69, 0.98] - Strong AR coefficient
- **σ_η = 0.09** [0.01, 0.16] - Very small innovation SD
- **φ = 20.26** [10.58, 30.71] - Large dispersion

**Diagnosis:**
1. **σ_η is too small** - With ρ = 0.84 and σ_η = 0.09, the AR(1) process has very little variation. The stationary variance is σ_η²/(1-ρ²) = 0.09²/(1-0.84²) ≈ 0.027. This means the AR(1) errors contribute almost nothing to the log-scale variation.

2. **Observation-level vs latent-level correlation** - The model creates AR(1) correlation on the log scale, but this doesn't guarantee the same correlation structure on the count scale after the nonlinear exponential transformation and negative binomial sampling.

3. **Nonlinear transformation issue** - Even if α_t has AR(1) structure, μ_t = exp(α_t) transforms it nonlinearly, and then adding NegBinomial noise can destroy the correlation structure at the observation level.

4. **Large dispersion φ** - The Negative Binomial variance is μ + μ²/φ. With φ ≈ 20, the observation noise is substantial relative to the signal, potentially washing out the latent AR(1) structure.

### What This Means

**The model is fitting AR(1) correlation to the wrong thing.** The temporal autocorrelation in the OBSERVED counts (ACF = 0.944) is not being captured by AR(1) errors on the log-scale latent state. The nonlinear link and observation noise break the connection.

### Mathematical Intuition

If α_t has ACF(1) = ρ = 0.84, then:
- E[α_t · α_{t-1}] ∝ ρ
- But E[exp(α_t) · exp(α_{t-1})] ≠ exp(ρ) due to nonlinearity
- And E[C_t · C_{t-1}] involves additional Negative Binomial covariance
- Result: Latent AR(1) with ρ = 0.84 does NOT produce observation-level ACF = 0.84

**The model is structurally unable to match the observed temporal autocorrelation**, even with perfect convergence and reasonable parameter estimates.

---

## Comparison to Experiment 1

### Summary Table

| Metric | Exp 1 (Baseline) | Exp 3 (AR(1)) | Change | Success? |
|--------|------------------|---------------|--------|----------|
| **Residual ACF(1)** | **0.686** | **0.690** | **+0.004** | **NO** |
| Coverage 95% | 100.0% | 100.0% | 0.0% | NO |
| Coverage 80% | 95.0% | 97.5% | +2.5% | NO (worse) |
| Coverage 50% | 67.5% | 75.0% | +7.5% | NO (worse) |
| Extreme p-values | 7 | 5 | -2 | Minor |
| R² | 0.883 | 0.861 | -0.022 | NO (worse) |
| Residual std | 34.44 | 39.81 | +5.37 | NO (worse) |
| ACF(1) p-value | 0.000 | 0.000 | 0.000 | NO |

**Overall: AR(1) model FAILED to improve fit quality.**

### What Changed?

**No meaningful improvements:**
- Temporal autocorrelation UNCHANGED (critical metric)
- Coverage UNCHANGED (still 100%)
- ACF(1) test statistic still extreme (p = 0.000)

**Minor improvements:**
- 2 fewer extreme p-values (7 → 5)
- IQR and Q75 moved from extreme to healthy range

**Things that got WORSE:**
- Coverage at 50% and 80% levels increased (more over-covering)
- R² decreased (0.883 → 0.861)
- Residual variance increased
- Residual bias increased (mean: -4.21 → -9.87)

### Visual Comparison

**`acf_comparison_exp1_vs_exp3.png`** provides definitive visual proof:
- Left panel: Exp 1 with ACF(1) = 0.686 (orange bars)
- Right panel: Exp 3 with ACF(1) = 0.690 (green bars)
- Both patterns nearly identical at all lags
- Both far above target threshold (green dotted line at 0.3)
- Both above Phase 2 trigger (red dotted line at 0.5)

**Conclusion from visual evidence:** The AR(1) model produced no improvement in temporal structure.

---

## Specific Model Deficiencies

### 1. Observation-Level Temporal Correlation Not Captured

**Evidence:**
- Residual ACF(1) = 0.690 unchanged from 0.686 (`acf_comparison_exp1_vs_exp3.png`)
- U-shaped temporal pattern persists (`residual_diagnostics.png` Panel B)
- Observed ACF(1) = 0.944 still extreme (p = 0.000) (`test_statistics.png` Panel K)
- Replicated trajectories still choppy vs smooth observed (`ppc_dashboard.png` Panel C)

**Implication:** The AR(1) structure on the latent log-scale state does NOT translate to temporal correlation at the observation level. The model architecture is fundamentally mismatched to the data's temporal structure.

**Why this matters:**
- The latent AR(1) is invisible at the observation level
- Predictions still treat observations as independent given latent state
- Cannot capture the smoothness/persistence in observed counts
- Forecasting will still be poor

### 2. Nonlinear Link Breaks Correlation Structure

**Evidence:**
- Model has ρ = 0.84 (strong latent correlation)
- But observation ACF only ~0.75 in replications vs. 0.944 observed
- Gap between latent and observation-level correlation
- σ_η = 0.09 too small to matter

**Implication:** The exponential link function exp(α_t) and Negative Binomial observation noise destroy the latent AR(1) correlation structure. Even perfect AR(1) on log scale doesn't produce AR(1) on count scale.

**Why this matters:**
- Model architecture fundamentally flawed for this data
- Adding AR(1) to latent state is not equivalent to observation-level AR
- Need models that directly specify observation-level correlation

### 3. Systematic Bias Unchanged

**Evidence:**
- U-shaped residual pattern vs fitted (`residual_diagnostics.png` Panel A)
- U-shaped residual pattern vs time (`residual_diagnostics.png` Panel B)
- Large negative residuals at end of series (time 35-40)
- Green smooth trend lines show systematic waves

**Implication:** The AR(1) model did not fix the mean structure problems. The quadratic trend + AR(1) errors still cannot match the observed acceleration pattern.

**Why this matters:**
- Late-period predictions systematically too high
- Bias is not random - it's structured and predictable
- Mean function may need different form (exponential, logistic, changepoint)

### 4. Over-Coverage Persists

**Evidence:**
- 100% of observations in 95% interval (`coverage_detailed.png`)
- 97.5% in 80% interval (target: 80%)
- 75.0% in 50% interval (target: 50%)
- Identical over-coverage to Experiment 1

**Implication:** Adding AR(1) structure increased uncertainty without improving fit. The model is uncertain about the wrong aspects - it has wide intervals but still misses the temporal structure.

**Why this matters:**
- Prediction intervals not useful (too wide)
- False sense of security from perfect coverage
- Uncertainty quantification is miscalibrated

### 5. Distribution Shape Mismatch Unchanged

**Evidence:**
- Skewness p = 0.993 (essentially unchanged from 0.999)
- Kurtosis p = 0.999 (essentially unchanged from 1.000)
- Observed data less skewed and flatter than model predicts
- Q-Q plot shows some tail deviation

**Implication:** The Negative Binomial distribution may not be the right family, or the mean function is wrong leading to wrong predictions of skewness/kurtosis.

---

## Overall Model Adequacy Assessment

### Fit Quality by Criterion

| Criterion | Quality | Value (Exp 3) | Value (Exp 1) | Target | Change |
|-----------|---------|---------------|---------------|--------|--------|
| Residual ACF(1) | **POOR** | **0.690** | **0.686** | **< 0.3** | **NONE** |
| Coverage (95%) | POOR | 100.0% | 100.0% | 90-98% | NONE |
| P-values | POOR | 5 extreme | 7 extreme | ≤ 2 | Minor |

### Strengths (Inherited from Exp 1)

1. **Convergence**: R̂ = 1.000, ESS > 1,100, minimal divergences (0.17%)
2. **Central tendency**: Mean still well-captured (p = 0.612)
3. **Overall variation**: Variance still well-captured (p = 0.772)
4. **Computational**: Stable MCMC, reasonable runtime
5. **Minor improvement**: Fixed IQR and Q75 extreme p-values

### Critical Weaknesses (Unchanged from Exp 1)

1. **Temporal dependence**: Residual ACF(1) = 0.690, NO IMPROVEMENT
2. **Observation correlation**: Model cannot generate ACF = 0.944
3. **Systematic patterns**: U-shaped temporal residual patterns persist
4. **Over-coverage**: 100% coverage unchanged
5. **Distribution shape**: Skewness and kurtosis still mismatched

### New Weaknesses (Created by AR(1) Model)

1. **Worse calibration**: 50% and 80% coverage got worse
2. **Worse predictions**: R² decreased, residual variance increased
3. **Larger bias**: Mean residual increased in magnitude
4. **Added complexity**: More parameters with no benefit
5. **Structural mismatch**: Latent AR(1) doesn't translate to observation AR

### Decision Matrix Position

```
                    Residual ACF(1)
                < 0.3    0.3-0.5    > 0.5
Coverage  90-98%  GOOD     ACCEPT   POOR
          85-90%  ACCEPT   ACCEPT   POOR
          < 85%   POOR     POOR     POOR
          > 98%   ACCEPT   ACCEPT   POOR ← Exp 3: (100%, 0.690)
                                     ↑ Exp 1: (100%, 0.686)
```

**Position: POOR FIT** (100% coverage, ACF(1) = 0.690)
- **IDENTICAL position to Experiment 1**
- **No movement in decision matrix**

---

## Scientific Interpretation

### What the AR(1) Model Gets Right

The model successfully:
- **Estimates temporal correlation parameters**: ρ = 0.84 [0.69, 0.98] is well-identified
- **Maintains trend capture**: Still captures long-term acceleration
- **Maintains dispersion**: Overdispersion still handled via Negative Binomial
- **Maintains central tendency**: Mean and variance matching preserved
- **Achieves convergence**: No technical MCMC issues

### What the AR(1) Model Fails to Do

The model FAILS to:
- **Translate latent correlation to observations**: ρ = 0.84 on log scale ≠ ACF = 0.944 on count scale
- **Reduce residual autocorrelation**: ACF(1) unchanged at 0.690
- **Improve coverage**: Still 100%, not the target 90-98%
- **Eliminate temporal patterns**: U-shaped residual patterns persist
- **Improve predictions**: R² actually decreased
- **Justify added complexity**: 3 extra parameters (ρ, σ_η, 39 ε_t values) with no benefit

### Why Latent AR(1) Doesn't Work Here

**The fundamental problem:** AR(1) on the log-scale latent state does not produce AR(1) on the count-scale observations.

**Mathematical reason:**
```
If:  α_t ~ AR(1) with Corr(α_t, α_{t-1}) = ρ
Then: μ_t = exp(α_t)
And:  C_t ~ NegBinom(μ_t, φ)

Does NOT imply: Corr(C_t, C_{t-1}) = ρ
```

The nonlinear transformation (exp) and observation noise (NegBinom) break the correlation structure.

**Evidence this is the issue:**
- Model estimates ρ = 0.84 (latent level)
- Replicated data has ACF ≈ 0.75 (observation level, from test statistic distribution)
- Observed data has ACF = 0.944 (target)
- Gap: 0.84 (latent) → 0.75 (observed replications) < 0.944 (observed data)

**Implication:** Even if the latent AR(1) were perfect, the model architecture prevents it from producing the observed correlation structure in the counts.

### What the Data Actually Exhibits

**The observed data ACF = 0.944 means:**
- 89% of variance at time t is explained by time t-1
- Consecutive observations are extremely similar
- Process has very high persistence/inertia
- Smooth trajectories, not independent jumps

**Possible mechanisms:**
1. **Observation-level AR**: Counts directly depend on previous counts (not just latent state)
2. **Cumulative process**: Counts are partial sums of underlying process
3. **Slow-varying unmodeled drivers**: External factors change very gradually
4. **Measurement overlap**: Observation windows overlap in time
5. **Different time scale**: Process operates on different time scale than observation intervals

**None of these are captured by latent AR(1) on log scale.**

---

## Recommendations for Next Steps

### What We Learned

**Critical insight:** Latent-scale temporal models are NOT equivalent to observation-scale temporal models. The AR(1) on log(μ) does not produce AR correlation in counts.

**This failure is informative:** It rules out a class of models and points toward different architectures.

### DO NOT Pursue

1. **Higher-order latent AR**: AR(2), AR(3) on latent state will have the same problem
2. **Different latent structures**: ARMA, random walk on latent state won't help
3. **More parameters on this architecture**: The structure is wrong, not the parameterization
4. **Different priors**: Not a prior specification issue, it's an architecture issue

### SHOULD Pursue

#### 1. Observation-Level Temporal Models

**Idea:** Model correlation directly on the count scale, not latent scale.

**Example structures:**
```
# Conditional AR
C_t | C_{t-1} ~ NegBinom(μ_t, φ)
log(μ_t) = β₀ + β₁·year_t + β₂·year_t² + γ·C_{t-1}

# Parameter-level AR
C_t ~ NegBinom(μ_t, φ)
log(μ_t) = α_t
α_t = ρ·α_{t-1} + β₀ + β₁·year_t + β₂·year_t² + η_t
(NOTE: This is different from latent AR - here α_t is deterministic given α_{t-1})
```

**Why this might work:** Direct dependence on previous observation preserves correlation through nonlinearity.

#### 2. Different Mean Function

**Idea:** The temporal patterns may be due to mean function misspecification, not autocorrelation.

**Try:**
- Exponential growth: log(μ_t) = β₀ + β₁·exp(β₂·year_t)
- Logistic growth: log(μ_t) = K·exp(r·year_t)/(K + exp(r·year_t) - 1)
- Piecewise/changepoint: Different quadratics in different periods
- Splines: Flexible smooth function of time

**Why this might work:** The U-shaped residual patterns suggest systematic mean function bias. Better mean function might eliminate apparent autocorrelation.

#### 3. Overdispersion Models

**Idea:** Maybe the issue is not temporal correlation but heterogeneous overdispersion.

**Try:**
- Time-varying dispersion: φ_t varies with time or fitted values
- Zero-inflated: Allow excess zeros with time-varying probability
- Mixture models: Some observations from different distribution

**Why this might work:** If dispersion is wrong, residuals can appear correlated even if they're not.

#### 4. Different Temporal Scale

**Idea:** Maybe one time step is not the right lag for dependence.

**Try:**
- Multi-step models: C_t depends on C_{t-2} or C_{t-3}
- Seasonal models: Dependence on earlier periods
- Trend + cycles: Separate smooth trend from cyclic component

**Why this might work:** The ACF pattern (high at lag 1, decaying slowly) might reflect longer-range dependence.

#### 5. Non-Stationary Models

**Idea:** Parameters or structure may change over time.

**Try:**
- Time-varying coefficients: β_t evolves (random walk, GP)
- Changepoint models: Different regimes in different periods
- Dynamic linear models: All parameters evolve

**Why this might work:** The U-shaped temporal pattern suggests non-stationarity. Early vs late period may have different dynamics.

### Validation Strategy for Next Models

**Primary metric:** Residual ACF(1) must be < 0.3 (ideally < 0.2)
**Secondary metrics:**
- Coverage in 90-98% range
- Bayesian p-values in 0.1-0.9 range
- No systematic temporal patterns in residuals
- R² improvement

**Tests:**
1. Residual ACF plot - must be flat within confidence bands
2. Residuals vs time - must be random scatter, no waves
3. One-step-ahead prediction - quantify temporal forecast accuracy
4. Information criteria - DIC, WAIC, LOO vs Exp 1 and Exp 3

### Most Promising Direction

**Recommendation:** Start with **observation-level conditional AR** or **improved mean function** (exponential/logistic growth).

**Rationale:**
1. These directly address the failure mode of Exp 3
2. Observation-level AR can match observation-level correlation
3. Better mean function might eliminate spurious autocorrelation
4. Both are interpretable and testable

**Specific next experiment:**
```stan
// Observation-level AR(1) Negative Binomial
C_t ~ NegBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁·year_t + β₂·year_t² + γ·log(C_{t-1} + 1)
```

This directly models count-on-count dependence while preserving trend structure.

---

## Technical Appendix

### Computation Details

- **Software**: Stan 2.26, ArviZ 0.22.0, Python 3.10
- **Sampling**: 4 chains, 1,500 draws each (6,000 total)
- **Posterior predictive**: Generated 6,000 replicated datasets with full AR(1) structure
- **Test statistics**: Computed for all 6,000 replications
- **Runtime**: ~15 minutes for PPC generation and analysis
- **Memory**: Generated ~240MB of replicated data (6,000 × 40 observations)

### Diagnostic Checks Performed

1. **Coverage analysis**: 50%, 80%, 95% prediction intervals
2. **Residual autocorrelation**: Up to lag 15
3. **Test statistics**: 13 summary statistics with Bayesian p-values
4. **Visual checks**: 5 comprehensive plots
5. **Comparison to Experiment 1**: Side-by-side ACF comparison

### Posterior Predictive Generation

**Method:** Manual generation accounting for full AR(1) structure:
```python
for each posterior sample i:
    # Generate AR(1) latent process
    epsilon[1] ~ Normal(0, sigma_eta / sqrt(1 - rho^2))
    for t in 2:N:
        epsilon[t] = rho * epsilon[t-1] + Normal(0, sigma_eta)

    # Compute latent state and mean
    alpha[t] = beta_0 + beta_1*year[t] + beta_2*year[t]^2 + epsilon[t]
    mu[t] = exp(alpha[t])

    # Generate observations
    C_rep[t] ~ NegBinomial(mu[t], phi)
```

This ensures replicated data has the same generative structure as the model assumes.

### Files Generated

**Code:**
- `code/posterior_predictive_checks.py` - Main analysis script (780 lines)
- `code/ppc_results.npz` - Numerical results archive

**Plots:**
- `plots/ppc_dashboard.png` - 12-panel comprehensive overview (1.0MB)
- `plots/acf_comparison_exp1_vs_exp3.png` - **CRITICAL comparison** (206KB)
- `plots/coverage_detailed.png` - Detailed coverage plot (359KB)
- `plots/residual_diagnostics.png` - 6-panel residual suite (713KB)
- `plots/test_statistics.png` - Test statistic distributions (431KB)

**Documentation:**
- `ppc_findings.md` - This document

### Key Statistics Summary

```
Experiment 3 (AR(1)) vs Experiment 1 (Baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric                  Exp 1    Exp 3    Change
────────────────────────────────────────────────
Residual ACF(1)         0.686    0.690    +0.004
Coverage 95%            100.0%   100.0%   +0.0%
Coverage 80%            95.0%    97.5%    +2.5%
Coverage 50%            67.5%    75.0%    +7.5%
Extreme p-values        7        5        -2
R²                      0.883    0.861    -0.022
Residual SD             34.44    39.81    +5.37
ACF(1) p-value          0.000    0.000    0.000
────────────────────────────────────────────────
Overall Fit             POOR     POOR     UNCHANGED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Conclusion

The Latent AR(1) Negative Binomial model represents a **FAILED attempt** to address the temporal autocorrelation problem identified in Experiment 1. Despite perfect convergence and successful estimation of temporal correlation parameters (ρ = 0.84), the model **does not improve posterior predictive performance** on any critical metric.

### Key Takeaways

1. **AR(1) on latent state ≠ AR(1) on observations**: Nonlinear link and observation noise break the correlation structure.

2. **Residual ACF unchanged**: 0.690 vs 0.686 - the primary success metric shows zero improvement.

3. **Coverage unchanged**: 100% vs 100% - the secondary success metric shows zero improvement.

4. **Some metrics worsened**: Calibration at 50%/80% levels, R², residual variance all got worse.

5. **Structural mismatch**: The model architecture is fundamentally unable to match the observed temporal dynamics.

6. **Informative failure**: This negative result rules out latent temporal models and points toward observation-level or different mean function specifications.

### Final Assessment

**OVERALL FIT: POOR**
**SUCCESS vs EXPERIMENT 1: FAILED**
**NEXT STEPS: Fundamentally different model architecture required**

The AR(1) model added complexity (3 extra parameters, 39 latent states) with **no benefit**. This is a clear case where posterior predictive checks reveal that good convergence and reasonable parameter estimates do NOT imply good model fit.

**Recommendation:** Abandon latent temporal correlation approaches. Pursue observation-level conditional models or improved mean functions that can directly produce the observed ACF = 0.944 in counts.

---

**Analysis completed:** 2025-10-29
**Analyst:** Claude (Model Validation Specialist)
**Status:** ✓ Complete - Critical failure documented, clear path forward identified
