# Prior Predictive Check: Experiment 2 (NB-AR1)

## Quick Start

**DECISION: FAIL** - Do not proceed with current priors

**Start here**: `plots/decision_summary.png` - One-page visual summary

**Then read**: `findings.md` - Complete analysis with recommendations

---

## What Happened?

The prior predictive check revealed **critical issues**:

1. **Extreme outliers**: 3.22% of simulated counts exceed 10,000 (observed max: 269)
   - Maximum generated: **674 million**
   - Root cause: Multiplicative explosion through exp(η)

2. **AR(1) validation failed**: Correlation between ρ and realized ACF = 0.39 (expected: >0.95)
   - Suggests finite sample issues or initialization problems

3. **Numerical instability**: ACF computations returned NaN due to extreme values

---

## Recommended Action

**Revise priors (Version 2)**:
```
β₀ ~ Normal(4.69, 1.0)                    [Keep]
β₁ ~ TruncatedNormal(1.0, 0.5, -0.5, 2.0) [CHANGE - constrain growth]
φ  ~ Normal(35, 15)                       [CHANGE - inform from Exp1]
ρ  ~ Beta(20, 2)                          [Keep - appropriate]
σ  ~ Exponential(5)                       [CHANGE - tighter innovation]
```

**Expected improvement**:
- 99th percentile: 143,745 → ~5,000
- Extreme outliers: 3.22% → <0.1%

---

## Files

### Code
- `code/prior_predictive_check.py` - Main validation script (500 simulations)
- `code/create_decision_summary.py` - Summary visualization generator

### Plots (6 files)
1. `decision_summary.png` - **START HERE** - Executive summary
2. `prior_parameter_distributions.png` - Parameter sampling validation
3. `temporal_correlation_diagnostics.png` - AR(1) structure assessment
4. `prior_predictive_trajectories.png` - Time series dynamics
5. `prior_acf_structure.png` - Autocorrelation patterns
6. `prior_predictive_coverage.png` - Range and plausibility

### Documentation
- `findings.md` - Complete technical report with detailed analysis
- `README.md` - This file

---

## Key Insights

### What Worked ✓
- ρ ~ Beta(20, 2) is appropriate (motivated by EDA ACF=0.971)
- Median behavior is reasonable (50th percentile ≈ observed range)
- AR(1) structure shows persistence in trajectories
- Parameter sampling is correct

### What Failed ✗
- Tail behavior is catastrophic (multiplicative explosion)
- Innovation prior too wide (σ allows extreme variability)
- Growth prior unrestricted (β₁ allows 1000%+ growth)
- AR(1) not realizing intended correlation in finite samples

### The Value of Prior Predictive Checks

This check **successfully prevented wasted computation** by catching issues before model fitting. The problems identified would have caused:
- Divergent transitions during MCMC
- Extreme posterior uncertainty
- Numerical overflow
- Invalid inference

---

## Next Steps

1. Implement revised priors (Version 2)
2. Re-run this prior predictive check script
3. Verify extremes are controlled
4. Proceed to model fitting only after PASS

---

## Technical Details

**Model**: NB-AR(1) with temporal correlation
```
C_t ~ NegativeBinomial(exp(η_t), φ)
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t
ν_t ~ Normal(0, σ)
```

**Data**: 40 observations, counts range [21, 269]

**Simulation**: 500 prior draws × 40 timepoints = 20,000 count observations

**Context**: Building on Experiment 1 (baseline model, residual ACF=0.511)

---

Generated: 2025-10-29
Status: COMPLETE - Awaiting prior revision
