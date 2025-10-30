# Simulation-Based Calibration: Findings and Decision

**Model**: Beta-Binomial (Experiment 1)
**Date**: 2025-10-30
**Analyst**: Model Validation Specialist
**Status**: **FAILED** ⚠️

---

## Executive Summary

Simulation-based calibration reveals **critical failure** in recovering the concentration parameter φ, while mean parameter μ recovers well. The model cannot reliably estimate overdispersion with N=12 trials.

### Key Findings

✓ **μ (mean success probability)**: Recovers accurately with good calibration
✗ **φ (concentration/overdispersion)**: Catastrophic calibration failure (45.6% coverage vs 95% target)

**Decision**: **DO NOT PROCEED** to real data fitting until φ estimation is resolved.

---

## What is Simulation-Based Calibration?

SBC tests whether a model can "find the truth when we know it." We:

1. Draw true parameters from the prior
2. Generate synthetic data with these parameters
3. Fit the model and check if posteriors recover truth
4. Repeat 150 times to assess systematic patterns

**If the model can't recover known truth, it won't find unknown truth in real data.**

---

## The Problem: φ Cannot Be Recovered

### Visual Evidence

See `sbc_rank_histograms.png` (right panel):
- Expected: Uniform histogram (flat red line)
- Observed: **Extreme spike between ranks 1000-1500**
- Interpretation: Posterior is nearly identical regardless of true φ value

This is what "catastrophic miscalibration" looks like.

### The Numbers

| Metric | μ | φ | Status |
|--------|---|---|--------|
| **Coverage** (target: 95%) | 96.6% ✓ | **45.6%** ✗ | φ intervals miss truth >50% of time |
| **Bias** (target: ~0) | 0.013 ✓ | **-2.185** ✗ | φ systematically underestimated |
| **Rank uniformity** | Mild deviation | **Catastrophic** ✗ | χ² = 1104 vs expected ~19 |

### What This Means

When true φ = 8, the model estimates φ ≈ 1.5 with a 95% CI of [0.8, 2.5]. The truth is **outside the interval** most of the time.

---

## Why This Happens: Weak Identifiability

### The Core Issue

With only **12 trials**, the data doesn't provide enough information to reliably distinguish different φ values.

Consider:
- φ = 2: Moderate overdispersion
- φ = 8: Weak overdispersion

Both can generate similar-looking data with 12 trials. The model can't tell them apart.

### Evidence from Posterior Contraction

See `posterior_contraction.png`:
- **μ**: Posterior SD = 0.65 × Prior SD (strong learning)
- **φ**: Posterior SD = 0.91 × Prior SD (minimal learning)

The data barely updates beliefs about φ from the prior. This is **practical non-identifiability**.

---

## The Shrinkage Pattern

See `parameter_recovery_scatter.png` (top right):

The scatter plot shows posterior estimates "compressed" toward low values:
- True φ = 2 → Estimated φ ≈ 1.5 ✓
- True φ = 5 → Estimated φ ≈ 2.5 ✗ (underestimate by 2.5)
- True φ = 10 → Estimated φ ≈ 3.0 ✗ (underestimate by 7.0)

**Pattern**: The model "shrinks toward the prior" because the likelihood is too weak to overcome it.

---

## Coverage Failure: A Visual Demonstration

See `coverage_calibration.png` (top right):

- Green points: True φ inside 95% CI (should be 95% of points)
- Red points: True φ outside 95% CI (should be 5% of points)

**Actual**: ~55% red points, ~45% green points

The credible intervals are:
1. **Too narrow** (over-confident about uncertainty)
2. **Too low** (systematically underestimate φ)

---

## What About μ?

The good news: **μ recovery is excellent**.

### Why μ Works But φ Doesn't

- **μ** (mean success rate): Directly observable from data
  - Average of r_i/n_i ≈ μ
  - 12 trials provide strong constraint

- **φ** (overdispersion): Requires observing *variability* across trials
  - Need to distinguish "overdispersion" from "sampling variability"
  - 12 trials insufficient for reliable inference

**Analogy**: You can estimate a coin's bias from 12 flips, but you can't reliably tell if the coin's bias is *changing* between flips with only 12 observations.

---

## Technical Details

### Inference Method: Laplace Approximation

We used MAP + Laplace approximation (normal approximation to posterior) because Stan compilation failed. This could contribute to poor φ calibration if:

1. True φ posterior has heavy tails (normal approximation poor)
2. True φ posterior is skewed or multimodal
3. Hessian estimation at MAP is inaccurate

### Potential Fixes

#### 1. Use Full MCMC (Recommended)
- Switch from Laplace approximation to full MCMC sampling
- Would reveal true posterior geometry
- May improve calibration if approximation is the issue

#### 2. Reparameterize
- Use log(φ) instead of φ
- May have better posterior geometry (more normal)

#### 3. Stronger Prior
- Current: φ ~ Gamma(2, 2) with E[φ] = 1
- Alternative: Gamma(4, 4) (narrower, more informative)
- Trade-off: Reduces posterior variance but adds assumptions

#### 4. Collect More Data
- Need N >> 12 trials for reliable φ estimation
- Likely need N ≥ 50-100 trials
- Not always feasible

---

## Pass/Fail Criteria Applied

### PASS Criteria (from task)

- ✗ Coverage rates within [0.90, 0.98] for 95% intervals
  - μ: 0.966 ✓
  - φ: 0.456 ✗ **CRITICAL FAILURE**

- ✓ No systematic bias (mean error ≈ 0)
  - μ: 0.013 ✓
  - φ: -2.185 ✗ **SYSTEMATIC UNDERESTIMATION**

- ✗ Rank statistics roughly uniform
  - μ: Mild deviation (acceptable)
  - φ: χ² = 1104 ✗ **CATASTROPHIC NON-UNIFORMITY**

- ✓ Posteriors contract from priors
  - μ: Strong contraction ✓
  - φ: Minimal contraction ⚠️ **WEAK IDENTIFIABILITY**

- ✓ Convergence achieved in >90% of runs
  - 99.3% success rate ✓

### FAIL Criteria (from task)

- ✓ Systematic bias in recovery
  - φ has bias of -2.185 (severe)

- ✓ Poor calibration
  - φ intervals too narrow and too low

- ✗ Convergence failures
  - Only 1/150 failures (not an issue)

- ✓ Parameters unidentifiable
  - φ shows weak identifiability with N=12

**Result**: Multiple FAIL criteria triggered for φ parameter.

---

## Decision: FAILED

### Summary of Evidence

1. **Visual Evidence** (6 diagnostic plots)
   - Rank histograms show extreme non-uniformity for φ
   - Coverage plots show systematic under-coverage
   - Recovery scatter shows severe shrinkage pattern
   - Contraction plots show minimal learning for φ

2. **Quantitative Evidence**
   - Coverage: 45.6% vs 95% target (50% error rate)
   - Bias: -2.185 (large magnitude)
   - Rank test: χ² = 1104 vs expected 19 (58× worse)

3. **Consistency Across Diagnostics**
   - Every diagnostic points to same problem
   - Pattern is systematic, not random

### What This Means for Real Data

**If we fit this model to real data:**

1. φ estimates will be **systematically too low**
2. 95% credible intervals will **miss the true φ** more than half the time
3. We'll conclude "low overdispersion" even when overdispersion is high
4. Any scientific conclusions about overdispersion will be **unreliable**

**This is unacceptable for scientific inference.**

---

## Recommended Actions

### Immediate (Before Real Data Fitting)

1. **STOP**: Do not fit this model to real data
2. **Re-run SBC with full MCMC** (not Laplace approximation)
   - Test if approximation is causing calibration failure
   - Requires fixing Stan installation (make/compiler issues)

### If MCMC Doesn't Fix It

3. **Reparameterize**: Try log(φ) transformation
4. **Stronger prior**: Use more informative φ prior if domain knowledge available
5. **Simplify model**: Consider pooled binomial (ignore overdispersion) if φ not critical

### If Still Failing

6. **Accept limitation**: Document that φ not reliably estimable with N=12
7. **Collect more data**: Need N ≥ 50 trials for reliable φ inference
8. **Alternative model**: Investigate other approaches to overdispersion

---

## Positive Findings

Despite φ failure, important positives:

1. **μ estimation works well**
   - Can reliably estimate mean success rate
   - May be sufficient if overdispersion quantification not critical

2. **Computational stability**
   - 99.3% success rate excellent
   - No numerical issues or convergence failures

3. **SBC methodology validated**
   - Caught the problem *before* fitting real data
   - Saved time and prevented invalid conclusions

4. **Clear failure mode identified**
   - Problem is weak identifiability, not model misspecification
   - Provides direction for fixes (more data or simpler model)

---

## Comparison to Prior Expectations

From `metadata.md`, the model expected:
- φ ∈ [1, 5] based on observed φ_obs = 3.51

**Reality check**: If true φ = 3.51, SBC predicts:
- Estimated φ ≈ 1.5 (underestimate by 2)
- 95% CI ≈ [0.8, 2.5]
- True value (3.51) **outside interval**

This matches the "failure to recover φ > 3" pattern seen in SBC.

---

## Scientific Integrity Note

**This validation failure is exactly why we do SBC.**

It's tempting to:
- Ignore the problem
- Proceed with real data fitting
- Report φ estimates with false confidence

**We must not do this.**

The whole point of simulation-based calibration is to catch these issues *before* they lead to invalid scientific conclusions. The model failing SBC is the test working as intended.

---

## Final Verdict

**STATUS**: **FAILED** ⚠️

**μ Parameter**: PASS (96.6% coverage, negligible bias)
**φ Parameter**: FAIL (45.6% coverage, severe bias, non-uniform ranks)

**Overall**: Cannot proceed to real data analysis.

**Next Steps**:
1. Re-run with full MCMC (not Laplace approximation)
2. If still failing, consult recommendations above
3. Do not fit real data until validation passes

---

## Files Generated

All results saved to `/workspace/experiments/experiment_1/simulation_based_validation/`:

### Code
- `beta_binomial.stan`: Stan model (compilation failed)
- `run_sbc_scipy.py`: SBC implementation using Laplace approximation
- `visualize_sbc.py`: Diagnostic visualization code

### Results
- `results/sbc_results.csv`: Raw SBC results (149 iterations)
- `results/sbc_summary.json`: Quantitative metrics

### Plots (Visual Evidence)
- `plots/sbc_rank_histograms.png`: Primary calibration diagnostic
- `plots/coverage_calibration.png`: Interval coverage visualization
- `plots/parameter_recovery_scatter.png`: Bias and accuracy assessment
- `plots/posterior_contraction.png`: Learning vs prior
- `plots/parameter_space_identifiability.png`: Parameter space explored
- `plots/zscore_distribution.png`: Additional calibration check
- `plots/sbc_comprehensive_summary.png`: Combined overview

### Reports
- `recovery_metrics.md`: Detailed quantitative analysis (this document)
- `findings.md`: Summary and decision

---

**Validation Date**: 2025-10-30
**Validation Status**: FAILED
**Required Action**: Re-run with full MCMC before proceeding
