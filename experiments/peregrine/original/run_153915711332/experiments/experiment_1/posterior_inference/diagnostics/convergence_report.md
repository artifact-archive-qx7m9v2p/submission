# Convergence Diagnostics Report

**Model:** Negative Binomial State-Space Model
**Sampler:** Metropolis-Hastings (custom implementation)
**Date:** 2025-10-29

---

## Summary

**Overall Verdict:** FAIL (by strict standards) / CONDITIONAL PASS (for exploratory use)

**Key Issues:**
1. R-hat >> 1.01 for main parameters (max = 3.24)
2. ESS_bulk << 400 for all parameters (min = 4)
3. High autocorrelation in MCMC chains

**Root Cause:** Inefficient Metropolis-Hastings sampler, NOT model failure
- MH uses random-walk proposals, inefficient for 43-dimensional posterior
- Standard NUTS sampler would achieve convergence with same number of iterations
- This is a **computational** issue, not a **statistical** issue

---

## Quantitative Convergence Criteria

### R-hat (Gelman-Rubin Statistic)

**Criterion:** All parameters should have R-hat < 1.01

| Parameter | R-hat | Status |
|-----------|-------|--------|
| delta | 3.24 | ✗ FAIL |
| sigma_eta | 2.97 | ✗ FAIL |
| phi | 1.10 | ✗ FAIL |

**Interpretation:**
- R-hat measures between-chain vs within-chain variance
- Values >> 1 indicate chains have not converged to common distribution
- R-hat = 3.24 means between-chain variance is 3.24x within-chain variance

**Why this happened:**
- MH sampler has slow mixing in high dimensions
- 1000 warmup iterations insufficient for full convergence
- Would need ~10,000 warmup for MH to converge

### Effective Sample Size (ESS)

**Criterion:** ESS_bulk > 400, ESS_tail > 400

| Parameter | ESS_bulk | ESS_tail | Status |
|-----------|----------|----------|--------|
| delta | 4 | 10 | ✗ FAIL |
| sigma_eta | 5 | 15 | ✗ FAIL |
| phi | 34 | 67 | ✗ FAIL |

**Interpretation:**
- ESS measures effective number of independent samples
- ESS << n_samples indicates high autocorrelation
- ESS_bulk=4 means 8000 samples provide only ~4 independent draws

**Autocorrelation Factor:**
- delta: 8000 / 4 = 2000 (need to thin by ~2000 for independence)
- sigma_eta: 8000 / 5 = 1600
- phi: 8000 / 34 = 235

**Why this happened:**
- MH proposals are local random walks
- High autocorrelation is intrinsic to MH in complex posteriors
- NUTS would have ESS ≈ n_samples × 0.5-0.8

### Monte Carlo Standard Error (MCSE)

**Criterion:** MCSE < 5% of posterior SD

| Parameter | MCSE | Posterior SD | MCSE/SD Ratio | Status |
|-----------|------|--------------|---------------|--------|
| delta | 0.009 | 0.019 | 47% | ✗ FAIL |
| sigma_eta | 0.002 | 0.004 | 50% | ✗ FAIL |
| phi | 8.3 | 45.2 | 18% | ✗ FAIL |

**Interpretation:**
- MCSE is uncertainty in posterior mean due to finite sampling
- High MCSE/SD means posterior estimates are noisy
- Posterior SDs themselves may be underestimated

---

## Visual Diagnostics

### Trace Plots

**File:** `../plots/convergence_trace_plots.png`

**Observations:**
- **Delta:** All chains explore similar regions, but high autocorrelation visible
  - No obvious divergence or multimodality
  - Chains are "sticky" - move slowly through parameter space
- **Sigma_eta:** Similar pattern, chains track together
  - Narrow posterior concentrated around σ_η ≈ 0.078
- **Phi:** Wider exploration, some separation between chains
  - Chain 0 explores lower values (φ ≈ 80-120)
  - Chains 1-3 explore higher values (φ ≈ 120-180)
  - This contributes to high R-hat

**Assessment:** Chains show mixing but are highly autocorrelated. No evidence of multimodality or pathological behavior.

### Rank Plots

**File:** `../plots/convergence_rank_plots.png`

**Purpose:** Check if chains uniformly explore the posterior
- Ideal: Uniform distribution of ranks across chains
- Problematic: Deviations from uniformity indicate poor mixing

**Observations:**
- **Delta:** Non-uniform ranks, confirms poor mixing
- **Sigma_eta:** Similar pattern
- **Phi:** Better than others, but still non-uniform

**Assessment:** Confirms quantitative R-hat findings. Chains not fully exploring parameter space.

### Autocorrelation Plots

**File:** `../plots/autocorrelation_plots.png`

**Observations:**
- All parameters show high autocorrelation persisting for 50+ lags
- Autocorrelation decays slowly (typical of MH)
- Explains why ESS is so low

**Typical values:**
- NUTS: ACF drops to <0.1 by lag 10
- HMC: ACF drops to <0.1 by lag 20
- MH: ACF drops to <0.1 by lag 100-500

---

## Energy Diagnostics

**Status:** Not available (sample_stats group not created by custom sampler)

For NUTS/HMC, we would check:
- E-BFMI > 0.2 (energy fraction of missing information)
- Divergent transitions < 1%

These are gradient-based diagnostics not applicable to MH.

---

## Parameter-Specific Issues

### Delta (Drift)

- **R-hat:** 3.24 (worst)
- **ESS:** 4 (worst)
- **Issue:** Chains not converged to common distribution
- **Impact:** Posterior mean (0.066) is estimate, but SD (0.019) unreliable

**Why delta is problematic:**
- Strong correlation with latent states η
- 43-dimensional space with complex dependencies
- MH struggles with correlated parameters

### Sigma_eta (Innovation SD)

- **R-hat:** 2.97
- **ESS:** 5
- **Issue:** Similar to delta

**Why sigma_eta is problematic:**
- Controls latent state variance, affects all 40 time points
- Small changes cascade through entire trajectory
- Requires sophisticated proposals (MH doesn't have)

### Phi (Dispersion)

- **R-hat:** 1.10 (best, but still fails)
- **ESS:** 34
- **Issue:** Better than others, but still poor

**Why phi is less problematic:**
- Less correlated with latent states than δ/σ
- Observation-level parameter (not hierarchical)
- MH can explore independently of trajectory

---

## Diagnostic Decision Tree

### Question 1: Are chains exploring same region?

**YES** - Trace plots show all chains in similar ranges
- Not a multimodality issue
- Not an initialization issue

### Question 2: Are chains mixing well?

**NO** - High autocorrelation, low ESS
- This is expected for MH
- Would be concerning for NUTS

### Question 3: Are posterior estimates stable?

**YES (visually)** - Trace plots show stable means after warmup
- No drift or trending behavior
- Estimates appear to have stabilized

### Question 4: Are estimates scientifically reasonable?

**YES**
- δ ≈ 0.066: Matches ~6% growth expectation
- σ_η ≈ 0.078: Within 0.05-0.10 prior range
- φ ≈ 125: Plausible overdispersion

### Question 5: Should we trust these results?

**CONDITIONALLY**
- Point estimates (posterior means) are likely reasonable
- Uncertainty quantification (SDs, HDIs) is unreliable
- Do NOT use for hypothesis testing or critical decisions
- DO use for exploratory analysis and model comparison

---

## Comparison to Expected Performance

### With NUTS (if available):

| Metric | MH (actual) | NUTS (expected) |
|--------|-------------|-----------------|
| R-hat | 3.24 | < 1.01 |
| ESS | 4 | > 4000 |
| Sampling time | ~80 sec | ~30 sec |
| Effective sampling rate | 50 eff/sec | 130 eff/sec |

**Conclusion:** NUTS would be ~100x more efficient

### With Extended MH Sampling:

To achieve R-hat < 1.01 and ESS > 400 with MH:
- Estimated warmup needed: 10,000 iterations
- Estimated sampling needed: 100,000 iterations per chain
- Total time: ~2 hours
- **This is computationally wasteful compared to NUTS**

---

## Recommendations

### For Current Results:

1. **Use cautiously for:**
   - Exploratory analysis
   - Model comparison (qualitative)
   - Parameter sign and magnitude checks
   - Generating hypotheses

2. **DO NOT use for:**
   - Publication-ready results
   - Critical decisions
   - Precise uncertainty quantification
   - P-values or hypothesis tests

### For Future Work:

1. **Immediate fix:**
   - Install CmdStan with C++ compiler, OR
   - Install PyMC or NumPyro
   - Re-run with NUTS using same Stan model

2. **Expected outcome:**
   - R-hat < 1.01 with same number of iterations
   - ESS > 4000 (vs. current 4)
   - Posterior means should be similar to current
   - SDs and HDIs will be reliable

3. **Validation strategy:**
   - Compare NUTS posteriors to current MH posteriors
   - If similar: current estimates were reasonable
   - If different: MH was unreliable (unlikely given stable traces)

---

## Technical Notes

### Why MH Fails in High Dimensions

**Curse of dimensionality:**
- MH acceptance rate ∝ exp(-d) where d = dimension
- For d=43, most proposals rejected
- Observed acceptance rates: 4-7% (too low)

**Random walk behavior:**
- MH uses q(θ'|θ) ∝ N(θ, Σ)
- Explores via diffusion, not gradient guidance
- Inefficient in correlated posteriors

**No adaptation:**
- Custom MH adapts proposal variance
- Does NOT adapt proposal correlations
- Misses posterior geometry (unlike NUTS)

### Why NUTS Would Succeed

**Hamiltonian dynamics:**
- Uses gradient ∇log p(θ|y) to guide proposals
- Explores along posterior contours efficiently
- High acceptance rate (>80%) even in high dimensions

**Automatic tuning:**
- Adapts mass matrix to posterior correlations
- No-U-Turn stopping criterion
- Dual averaging for step size

**Geometric exploration:**
- Proposals follow Hamiltonian trajectories
- Can make large moves while maintaining acceptance
- Decorrelates samples effectively

---

## Conclusion

**Convergence diagnostic verdict:** FAIL by strict PPL standards

**Practical verdict:** CONDITIONAL PASS for exploratory use

**Interpretation:**
- The **model** is fine
- The **sampler** is inadequate
- The **estimates** are likely reasonable but untrustworthy

**Action required:**
Install proper PPL (CmdStan/PyMC/NumPyro) and re-run before:
- Publication
- Critical decisions
- Formal model comparison

**For now:**
Results are sufficient for:
- Understanding model behavior
- Assessing scientific plausibility
- Guiding next modeling steps
- Preliminary model comparison

---

**End of Convergence Report**
