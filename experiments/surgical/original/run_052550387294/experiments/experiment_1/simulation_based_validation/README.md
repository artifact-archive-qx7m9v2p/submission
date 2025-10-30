# Simulation-Based Calibration Results

**Model**: Beta-Binomial (Experiment 1)
**Date**: 2025-10-30
**Status**: **FAILED** ⚠️

---

## Quick Summary

✗ **Overall Validation**: FAILED
✓ **μ Parameter**: PASS (96.6% coverage)
✗ **φ Parameter**: FAIL (45.6% coverage) - **CRITICAL ISSUE**

**Decision**: DO NOT proceed to real data fitting until φ recovery improves.

---

## Key Files

### Reports (Start Here)
- **`findings.md`**: Executive summary with decision and recommendations
- **`recovery_metrics.md`**: Detailed quantitative analysis with visual evidence

### Plots (Visual Evidence)
- **`plots/sbc_comprehensive_summary.png`**: Overview of all diagnostics
- **`plots/sbc_rank_histograms.png`**: Primary calibration test (shows φ failure)
- **`plots/coverage_calibration.png`**: Interval coverage visualization
- **`plots/parameter_recovery_scatter.png`**: Bias and accuracy assessment
- **`plots/posterior_contraction.png`**: Learning from data
- **`plots/parameter_space_identifiability.png`**: Parameter space explored
- **`plots/zscore_distribution.png`**: Additional calibration check

### Data
- **`results/sbc_results.csv`**: Raw results from 149 SBC iterations
- **`results/sbc_summary.json`**: Summary statistics

### Code
- **`code/run_sbc_scipy.py`**: SBC implementation (Laplace approximation)
- **`code/visualize_sbc.py`**: Plotting code
- **`code/beta_binomial.stan`**: Stan model (compilation failed, not used)

---

## The Problem in One Image

See `plots/sbc_rank_histograms.png` (right panel):

**Expected**: Flat histogram (uniform ranks)
**Observed**: Extreme spike between ranks 1000-1500
**Meaning**: Model cannot distinguish different φ values

---

## What Went Wrong?

### φ (Concentration Parameter)
- **Coverage**: 45.6% (should be 95%) - intervals miss truth >50% of time
- **Bias**: -2.185 (systematic underestimation)
- **Identifiability**: Weak with only N=12 trials
- **Conclusion**: Cannot reliably estimate overdispersion

### μ (Mean Success Rate)
- **Coverage**: 96.6% ✓
- **Bias**: 0.013 ✓
- **Identifiability**: Strong ✓
- **Conclusion**: Reliable estimation

---

## Why This Matters

If we ignored this validation and fit real data:

1. We'd report φ ≈ 1-2 with "95% confidence"
2. True φ could be 5-10 and we'd never know
3. Scientific conclusions about overdispersion would be **wrong**
4. We'd be over-confident in incorrect estimates

**SBC caught this before we wasted time on real data.**

---

## Root Cause

**Weak identifiability**: 12 trials insufficient to distinguish different φ values.

The data provides strong information about μ (mean success rate) but weak information about φ (variability in success rates). The posterior stays close to the prior for φ.

---

## Next Steps

### Option 1: Fix the Inference (Recommended)
- Re-run SBC with **full MCMC** (not Laplace approximation)
- Current method uses normal approximation which may be poor for φ
- Requires fixing Stan installation

### Option 2: Model Modifications
- Reparameterize using log(φ)
- Use stronger, more informative prior for φ
- Both may improve calibration but add assumptions

### Option 3: Simpler Model
- Use pooled binomial (ignore overdispersion)
- Accept we can only estimate μ reliably with N=12
- Document limitation clearly

### Option 4: Collect More Data
- Need N ≥ 50-100 trials for reliable φ estimation
- Not always feasible

---

## Validation Methodology

### What is SBC?

Simulation-Based Calibration tests: **Can the model recover known truth?**

1. Draw true parameters from prior: (μ_true, φ_true)
2. Generate synthetic data with these parameters
3. Fit model and check if posterior recovers truth
4. Repeat 150 times, check for systematic failures

### Why SBC?

- Catches problems **before** fitting real data
- Tests full inference pipeline (prior → data → posterior)
- Reveals identifiability issues, computational problems, miscalibration
- **If it can't find known truth, it won't find unknown truth**

---

## Technical Details

### Configuration
- **SBC Iterations**: 149/150 successful (99.3%)
- **Inference Method**: MAP + Laplace approximation
- **Sample Sizes**: N=12 trials with n=[47, 148, 119, ..., 360]
- **Priors**: μ ~ Beta(2, 25), φ ~ Gamma(2, 2)

### Why Laplace Approximation?

Stan compilation failed (no `make` available), so we used:
1. Find MAP (maximum a posteriori) via optimization
2. Approximate posterior as multivariate normal using Hessian
3. Sample from approximation

**Limitation**: If true φ posterior is non-normal (heavy tails, skewed, multimodal), approximation will be poor. This may explain calibration failure.

---

## Interpretation of Diagnostic Plots

### 1. Rank Histograms (Primary SBC Diagnostic)

**Well-calibrated**: Uniform histogram (ranks spread evenly 0-4000)
**μ**: Roughly uniform with slight central peak (acceptable)
**φ**: Extreme spike at ranks 1000-1500 (catastrophic failure)

**What ranks mean**: If posterior correctly captures uncertainty, the rank of the true value among posterior samples should be uniformly distributed across many SBC iterations.

### 2. Coverage Plots

**Well-calibrated**: 95% of points green (true value in 95% CI)
**μ**: 96.6% green ✓
**φ**: 45.6% green ✗ - most points RED (true value outside CI)

### 3. Recovery Scatter

**Well-calibrated**: Points cluster on y=x line
**μ**: Tight clustering around y=x ✓
**φ**: Points below y=x line, especially for large φ ✗ (systematic underestimation)

### 4. Posterior Contraction

**Learning from data**: Posterior SD < Prior SD
**μ**: Strong contraction (ratio=0.645) ✓
**φ**: Minimal contraction (ratio=0.908) ✗ (data provides little information)

### 5. Z-Score Distribution

**Well-calibrated**: N(0, 1) distribution
**μ**: Mean=0.378, SD=0.946 (close to N(0,1)) ✓
**φ**: Mean=-3.157, SD=2.806 (strongly negative, wide spread) ✗

Negative z-scores indicate (posterior_mean - true) / posterior_sd < 0, i.e., posterior underestimates true value.

---

## Lessons Learned

### What Worked
- SBC methodology successfully identified critical problem
- Computational stability excellent (99.3% success)
- μ parameter shows proper calibration
- Clear visual diagnostics made problem obvious

### What Didn't Work
- φ parameter poorly identifiable with N=12 trials
- Laplace approximation may be inadequate for φ
- Prior-likelihood balance problematic for φ

### Key Insight
**Data structure matters**: 12 independent trials provide enough information for mean estimation (μ) but not variance estimation (φ). This is a fundamental limitation, not just a modeling choice.

---

## Recommendations Summary

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **HIGH** | Re-run with full MCMC | Medium | May fix calibration if Laplace is the issue |
| **HIGH** | Document limitation | Low | If φ not critical, proceed with μ only |
| MEDIUM | Try log(φ) parameterization | Low | May improve posterior geometry |
| MEDIUM | Stronger φ prior | Low | Reduces uncertainty but adds assumptions |
| LOW | Collect more data | High | Definitive fix but may not be feasible |

---

## Conclusion

The Beta-Binomial model **failed simulation-based calibration** for the concentration parameter φ. While μ (mean success rate) is reliably estimated, φ (overdispersion) shows catastrophic miscalibration with only 45.6% coverage.

**This is not a subtle issue.** Every diagnostic consistently shows severe φ recovery failure. The model cannot be trusted for overdispersion inference with N=12 trials.

**Status**: Validation FAILED. Do not proceed to real data analysis until issue is resolved.

---

**For detailed analysis, see**: `findings.md` and `recovery_metrics.md`
**For visual evidence, see**: `plots/` directory (start with `sbc_comprehensive_summary.png`)
