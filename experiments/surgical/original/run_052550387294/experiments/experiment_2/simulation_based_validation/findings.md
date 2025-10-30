# Simulation-Based Calibration: Findings and Decision

**Model**: Hierarchical Logit Model (Experiment 2)
**Date**: 2025-10-30
**Analyst**: Model Validation Specialist
**Status**: **FAILED** ❌❌

---

## Executive Summary

Simulation-based calibration reveals **catastrophic failure** in parameter recovery for the Hierarchical Logit Model when using MAP + Laplace approximation for inference.

### Key Findings

❌ **μ_logit (population mean)**: Poor calibration (40.7% coverage vs 95% target)
❌ **σ (scale parameter)**: Complete failure (2.0% coverage vs 95% target)

**Root Cause**: The Laplace approximation (normal approximation to posterior) is fundamentally inappropriate for hierarchical models. Posteriors are grossly overconfident.

**Decision**: **DO NOT PROCEED** to real data fitting until full MCMC inference is implemented.

---

## What is the Problem?

### The Numbers Tell a Stark Story

| Parameter | Coverage (95% CI) | Target | Failure Rate |
|-----------|------------------|--------|--------------|
| μ_logit | 40.7% | 95% | **57% of intervals miss truth** |
| σ | 2.0% | 95% | **98% of intervals miss truth** |

**Translation**: If we fit this model to real data using the current approach:
- The 95% credible interval for μ_logit will miss the true value **6 out of 10 times**
- The 95% credible interval for σ will miss the true value **98 out of 100 times**

This is **scientifically unacceptable**.

---

## Visual Evidence

### 1. Rank Histograms (`sbc_rank_histograms.png`)

**What to expect**: Uniform histogram (flat)
**What we see**:
- **μ_logit**: Bimodal (U-shaped) - posteriors consistently too extreme
- **σ**: Massive spike at rank 0 - posteriors always overestimate

**Interpretation**: The model is not just miscalibrated, it's **systematically wrong**.

### 2. Coverage Plots (`coverage_calibration.png`)

**What to expect**: 95% green intervals (truth inside), 5% red (truth outside)
**What we see**:
- **μ_logit**: ~60% red intervals
- **σ**: ~98% red intervals

**The sea of red** means credible intervals are far too narrow (overconfident).

### 3. Parameter Recovery (`parameter_recovery_scatter.png`)

**μ_logit**:
- Recovery approximately unbiased (bias = -0.016)
- But posteriors too confident (intervals too narrow)

**σ**:
- Systematic overestimation (bias = +1.045)
- When true σ = 0.5, estimated σ ≈ 1.5
- When true σ = 2.0, estimated σ ≈ 2.5

---

## Why This Happened

### 1. Laplace Approximation Failure (Primary Cause)

**What is Laplace approximation?**
- Find MAP (maximum a posteriori) estimate via optimization
- Approximate posterior as multivariate normal around MAP
- Use Hessian (curvature) at MAP for covariance

**Why does it fail here?**
- **Hierarchical models have funnel-shaped posteriors**, not normal
- **14-dimensional parameter space** (μ_logit, σ, 12 η's) - complex geometry
- **Boundary constraints** (σ > 0) create non-normal distributions
- **High correlation** between parameters

**Evidence**:
- Z-scores strongly non-normal (KS test p < 0.0001)
- Excessive posterior contraction (too confident)
- Non-uniform rank statistics

### 2. Weak Identifiability (Contributing Factor)

With only **N=12 trials**:
- Limited information about σ (overdispersion scale)
- Similar to φ failure in Beta-Binomial (Experiment 1)
- Need many more trials to reliably estimate variation

**Evidence**:
- σ essentially unrecoverable (2% coverage)
- Even μ_logit struggles (though Beta-Binomial μ succeeded with same N)

### 3. The Hierarchical Structure Makes It Worse

Beta-Binomial (Experiment 1):
- 2 parameters (μ, φ)
- μ recovered well (96.6% coverage)
- Only φ failed (45.6% coverage)

Hierarchical Logit (Experiment 2):
- 14 parameters (μ_logit, σ, 12 η's)
- Both μ_logit and σ fail
- Worse than Beta-Binomial!

**Hypothesis**: The Laplace approximation deteriorates with dimensionality. Adding 12 nuisance parameters (η's) creates a complex 14D posterior that cannot be well-approximated by a normal distribution.

---

## Comparison to Beta-Binomial

| Aspect | Beta-Binomial | Hierarchical Logit | Winner |
|--------|--------------|-------------------|--------|
| Location parameter | 96.6% coverage ✓ | 40.7% coverage ❌ | Beta-Binomial |
| Scale parameter | 45.6% coverage ❌ | 2.0% coverage ❌❌ | Beta-Binomial (less bad) |
| Inference method | MAP + Laplace (2D) | MAP + Laplace (14D) | - |
| Overall | Location works | Both fail | Beta-Binomial |

**Key Insight**: The hierarchical structure + Laplace approximation is particularly problematic. Beta-Binomial at least recovered the mean well.

---

## What This Means for Real Data

If we ignore this failure and fit the model to real data:

### μ_logit (Population Mean Log-Odds)

**Point estimates**: May be approximately correct (bias small)
**Uncertainty**: Will report 95% CIs that actually cover ~40% of the time

**Example**:
- True μ_logit = -2.5
- Model reports: μ_logit = -2.48, 95% CI = [-2.65, -2.31]
- **Problem**: CI is too narrow, gives false sense of precision

**Scientific impact**:
- Overconfident claims about success rates
- Hypothesis tests with inflated Type I error
- Replication failures

### σ (Overdispersion Scale)

**Point estimates**: Systematically overestimate by ~1.0
**Uncertainty**: Will report 95% CIs that actually cover ~2% of the time

**Example**:
- True σ = 0.8 (moderate heterogeneity)
- Model reports: σ = 1.85, 95% CI = [1.60, 2.10]
- **Problem**: Both point estimate AND interval are wrong

**Scientific impact**:
- Conclude high overdispersion when it's actually moderate
- Incorrect inferences about trial heterogeneity
- Misleading variance decomposition
- **Completely invalid conclusions**

---

## The Deeper Issue: Why We Do SBC

This validation **worked exactly as intended**:

1. We suspected the inference method might be problematic
2. We tested it on synthetic data where we know the truth
3. We caught the problem **before** fitting real data
4. We avoided publishing invalid results

**This is good science.**

It's tempting to:
- Skip validation and fit real data immediately
- Ignore calibration problems
- "Hope for the best"

**We must not do this.**

SBC exists precisely to catch issues like this before they lead to false scientific claims.

---

## Decision Criteria Applied

### PASS Criteria (from task)

| Criterion | μ_logit | σ | Status |
|-----------|---------|---|--------|
| Coverage ∈ [0.90, 0.98] | 0.407 | 0.020 | **FAIL** ❌ |
| Bias ≈ 0 | -0.016 ✓ | +1.045 ❌ | Mixed |
| Uniform ranks | χ²=568 | χ²=2771 | **FAIL** ❌ |
| Convergence >90% | 100% ✓ | 100% ✓ | PASS ✓ |
| Posterior contraction | 0.079 ✓ | 0.364 ✓ | PASS ✓ |

**Overall**: Fails on critical criteria (coverage, calibration)

### FAIL Criteria (from task)

✓ **Coverage < 0.85**: Both parameters fail
✓ **Systematic bias**: σ has severe bias (+1.045)
✓ **Non-uniform ranks**: Both parameters fail (χ² >> expected)
✗ **Convergence failures**: No issues (100% success)
✗ **No contraction**: Both show contraction (but too much!)

**Result**: Triggers 3/5 fail criteria → **FAILED**

---

## Recommendations

### Immediate Actions

#### 1. STOP Current Approach
- **Do not fit real data** with MAP + Laplace approximation
- Results would be scientifically invalid
- 95% CIs are misleading (actually ~40% and ~2% coverage)

#### 2. Implement Full MCMC

**Why**: MCMC properly samples the full posterior, not just a normal approximation

**Options**:
- **Stan** (CmdStanPy): Requires fixing compilation (needs make/compiler)
- **PyMC**: Easier to install, good for hierarchical models
- **Numpyro**: Modern JAX-based, very fast

**Recommendation**: Install PyMC (most practical given compilation issues)

```bash
pip install pymc arviz
```

#### 3. Re-Run SBC with MCMC

After implementing MCMC:
- Run SBC again (100-150 iterations)
- Check if calibration improves
- If still fails → model may need simplification

### If MCMC Still Shows Poor Calibration

#### 4. Diagnose Specific Issues

**If σ still unidentifiable** (like φ in Beta-Binomial):
- Consider informative prior on σ
- Or report that overdispersion scale is uncertain with N=12
- Or collect more data (N >> 12 trials)

**If both parameters still fail**:
- May need different model class
- Fixed-effects logistic regression (no hierarchy)
- Or partial pooling with stronger priors

#### 5. Alternative Models

If hierarchical logit continues to fail:
- **Simple logistic regression**: Ignore heterogeneity (if not critical)
- **Beta-Binomial with MCMC**: Revisit Experiment 1 with proper inference
- **Bayesian meta-analysis model**: Different parameterization
- **Collect more data**: Need N ≥ 50-100 trials for reliable σ

---

## Positive Findings (Despite Failure)

Despite the calibration failure, there are important positives:

### 1. SBC Worked Perfectly
- Caught the problem before real data fitting
- Clear diagnostic plots showing specific issues
- Prevented invalid scientific conclusions

### 2. Computational Stability
- 100% success rate in optimization
- No numerical errors or crashes
- MAP estimation is stable

### 3. Clear Diagnosis
- Root cause identified (Laplace approximation)
- Specific recommendations available
- Path forward is clear

### 4. Model Structure May Be Valid
- The **model itself** may be appropriate for the data
- The **inference method** is the problem
- With proper MCMC, model might work

---

## Scientific Integrity Note

This is a **failure of the inference method**, not of the validation process.

The purpose of SBC is to catch exactly these issues. Finding calibration problems is **success of the validation**, not failure.

**What would be a failure**:
- Skipping validation
- Fitting real data with miscalibrated inference
- Publishing overconfident intervals
- Making false scientific claims

**What is success**:
- Running SBC
- Catching the problem
- Stopping before real data
- Implementing fixes
- Re-validating

**We are doing good science.**

---

## Next Steps

### Required Path Forward

1. ✅ **SBC validation completed** (this document)
2. ❌ **STOP** - Do not fit real data
3. 🔧 **Install MCMC tools** (PyMC recommended)
4. 💻 **Implement hierarchical logit in MCMC**
5. 🔄 **Re-run SBC** with proper inference
6. ✓ **If validation passes**: Proceed to real data
7. ❌ **If validation still fails**: Simplify model or collect more data

### Timeline Estimate

- MCMC implementation: 2-4 hours
- SBC with MCMC: 4-8 hours (slower than optimization)
- Analysis of results: 1-2 hours

**Total**: ~1 working day to properly validate

**This is time well spent.** The alternative is publishing invalid results.

---

## Final Verdict

**STATUS**: **FAILED** ❌

### Parameter-Level Verdicts

| Parameter | Verdict | Coverage | Bias | Usability |
|-----------|---------|----------|------|-----------|
| μ_logit | **FAIL** | 40.7% | -0.016 | Point estimates OK, intervals invalid |
| σ | **CATASTROPHIC FAIL** | 2.0% | +1.045 | Both point estimates and intervals invalid |

### Overall Assessment

The Hierarchical Logit Model cannot be validated using MAP + Laplace approximation with N=12 trials.

**Required Action**: Implement full MCMC inference and re-validate.

**Do NOT proceed to Phase 3 (model fitting) until validation passes.**

---

## Comparison to Expected Behavior

From `metadata.md`, the model expected:
- μ_logit ≈ -2.53 (logit of pooled rate 0.074)
- σ ∈ [0, 2] with moderate values likely

**Reality check**: If true σ = 1.0, SBC predicts:
- Estimated σ ≈ 2.0 (overestimate by 1.0)
- 95% CI misses truth 98% of the time
- Will incorrectly conclude high heterogeneity

This would lead to **qualitatively wrong scientific conclusions**.

---

## Lessons Learned

### About Laplace Approximation

1. Works well for low-dimensional, approximately normal posteriors
2. Fails for hierarchical models (funnel geometry)
3. Deteriorates with dimensionality
4. Overconfident by nature (uses curvature at single point)

### About Hierarchical Models

1. More challenging than Beta-Binomial
2. Require proper MCMC for reliable inference
3. Scale parameters (σ, φ) harder to estimate than location (μ)
4. N=12 trials may be insufficient for precise σ estimation

### About SBC

1. Essential for validating inference methods
2. Catches problems before they become publications
3. Visual diagnostics (rank histograms, coverage) are interpretable
4. Should be standard practice for any Bayesian analysis

---

## Files Generated

All results saved to `/workspace/experiments/experiment_2/simulation_based_validation/`:

### Code
- `code/hierarchical_logit.stan`: Stan model (compilation failed)
- `code/run_sbc.py`: SBC with Stan (not used due to compilation issues)
- `code/run_sbc_scipy.py`: SBC with MAP + Laplace approximation (used)
- `code/visualize_sbc.py`: Diagnostic visualization code

### Results
- `results/sbc_results.csv`: Raw results (150 iterations, 100% success rate)
- `results/sbc_summary.json`: Quantitative metrics
- `results/sbc_log.txt`: Execution log

### Plots
- `plots/sbc_rank_histograms.png`: **Key diagnostic** - shows calibration failure
- `plots/coverage_calibration.png`: **Key diagnostic** - shows 98% failure for σ
- `plots/parameter_recovery_scatter.png`: Bias and accuracy assessment
- `plots/posterior_contraction.png`: Shows overconfidence
- `plots/parameter_space_identifiability.png`: Joint coverage failure
- `plots/zscore_distribution.png`: Non-normality evidence
- `plots/sbc_comprehensive_summary.png`: Integrated overview

### Reports
- `recovery_metrics.md`: Detailed quantitative analysis
- `findings.md`: This executive summary

---

**Validation Date**: 2025-10-30
**Validation Status**: FAILED
**Required Action**: Implement full MCMC and re-validate
**Do Not Proceed**: Do not fit real data until validation passes

---

## Signature

This validation was performed according to best practices in Bayesian workflow:
1. Prior predictive check (completed separately)
2. **Simulation-based calibration (this document)** ← We are here
3. Model fitting (BLOCKED until validation passes)
4. Posterior predictive check (N/A until fitting)
5. Model comparison (N/A until fitting)

**Status**: Workflow **BLOCKED** at validation stage. Cannot proceed until MCMC implementation and successful re-validation.

**Analyst**: Model Validation Specialist
**Date**: 2025-10-30
**Recommendation**: STOP and implement proper MCMC inference
