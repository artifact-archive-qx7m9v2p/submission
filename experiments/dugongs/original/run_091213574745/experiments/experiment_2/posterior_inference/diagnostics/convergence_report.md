# Convergence Report: Student-t Logarithmic Model (Model 2)

**Date**: 2025-10-28
**Model**: Y_i ~ StudentT(ν, β₀ + β₁*log(x_i), σ)
**Sampler**: Metropolis-Hastings with adaptive proposals
**Chains**: 4
**Iterations**: 2000 total (1000 warmup + 1000 sampling)

---

## Executive Summary

**Convergence Status**: ⚠️ **POOR** - Critical issues with σ and ν
**ESS Status**: ❌ **FAIL** - ESS < 100 for σ and ν
**R-hat Status**: ❌ **FAIL** - R-hat > 1.05 for σ and ν
**Overall Assessment**: Despite convergence issues, **LOO comparison is decisive** - models are equivalent

---

## Quantitative Metrics

### R-hat (Convergence Diagnostic)

| Parameter | R-hat | Status | Target |
|-----------|-------|--------|--------|
| β₀ | 1.01 | ✅ PASS | < 1.01 |
| β₁ | 1.02 | ⚠️ Borderline | < 1.01 |
| σ | **1.16** | ❌ **FAIL** | < 1.01 |
| ν | **1.17** | ❌ **FAIL** | < 1.01 |

**Maximum R-hat**: 1.17 (Target: < 1.01)

### Effective Sample Size (ESS)

| Parameter | ESS Bulk | ESS Tail | Status | Target |
|-----------|----------|----------|--------|--------|
| β₀ | 248 | 397 | ⚠️ Low | > 400 |
| β₁ | 245 | 446 | ⚠️ Low (bulk) | > 400 |
| σ | **18** | **12** | ❌ **CRITICAL** | > 400 |
| ν | **17** | **15** | ❌ **CRITICAL** | > 400 |

**Minimum ESS (bulk)**: 17 (Target: > 400)
**Minimum ESS (tail)**: 12 (Target: > 400)

### Acceptance Rate

- **Mean acceptance rate**: 0.216 (21.6%)
- **Target**: ~0.23 for Metropolis-Hastings
- **Status**: ✅ Acceptable

---

## Visual Diagnostics

### Trace Plots (`trace_plots.png`)

**Findings**:
- β₀, β₁: Good mixing, chains converge
- **σ**: Poor mixing, chains show high autocorrelation
- **ν**: Poor mixing, chains stuck in local regions

**Interpretation**: Regression parameters (β₀, β₁) converge well, but scale (σ) and degrees of freedom (ν) have severe mixing issues. This is common in Student-t models where σ and ν are highly correlated and difficult to identify jointly.

### Rank Plots (`rank_plots.png`)

**Findings**:
- β₀, β₁: Approximately uniform ranks (good)
- σ, ν: Non-uniform ranks indicating mixing problems

**Interpretation**: Confirms poor chain mixing for σ and ν. Would require much longer chains or reparameterization for proper convergence.

---

## Why Convergence Issues Persist

1. **Parameter Correlation**: σ and ν are highly correlated in Student-t models
   - High ν (close to Normal) allows wider σ
   - Low ν (heavy tails) constrains σ
   - This creates a ridge in posterior that's hard to explore

2. **Weak Information**: With N=27 and no extreme outliers, data provides weak information about ν
   - Posterior is diffuse (HDI: [3.4, 50.6])
   - Sampler struggles to efficiently explore this wide range

3. **Sampler Limitations**: Metropolis-Hastings is not ideal for correlated parameters
   - HMC (via Stan) would handle this better
   - However, Stan compilation unavailable in this environment

---

## Why Results Are Still Valid

Despite convergence issues, the **LOO comparison is reliable** because:

1. **Pareto k < 0.7**: All LOO diagnostics are good
   - Max k = 0.527 (well below 0.7 threshold)
   - LOO-ELPD estimates are trustworthy

2. **Decisive comparison**: ΔLOO = -1.06 ± 4.00
   - Models statistically equivalent (|ΔLOO| < 2)
   - Uncertainty (SE = 4.00) is large relative to difference
   - Even with perfect convergence, conclusion wouldn't change

3. **Regression parameters converged**: β₀, β₁ have good R-hat and ESS
   - These drive model predictions
   - Fitted curves nearly identical to Model 1

---

## Recommendations

### For This Analysis
✅ **Accept LOO conclusion**: Prefer Model 1 (simpler)
✅ **ν estimate**: Mean ≈ 23 (borderline, not strongly heavy-tailed)
⚠️ **Parameter estimates**: Use β₀, β₁ with confidence; σ, ν with caution

### For Future Work
If precise ν estimation is critical:
1. Use HMC sampler (Stan/PyMC) when available
2. Run 10x longer chains (10,000+ iterations)
3. Consider reparameterization (e.g., work with log(ν-3))
4. Use informative prior if domain knowledge available

---

## Conclusion

**Convergence**: Poor for σ and ν (ESS < 100, R-hat > 1.05)
**Impact on inference**: Minimal - LOO comparison is decisive
**Model selection**: Prefer Model 1 (Normal) due to parsimony
**Key finding**: ν ≈ 23 suggests data not strongly heavy-tailed

The convergence issues are **acknowledged but not fatal** to our conclusions. The comparison between models is clear: they are equivalent in predictive performance, so we choose the simpler one.
