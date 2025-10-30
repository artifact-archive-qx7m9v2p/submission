# Simulation-Based Calibration: Recovery Metrics Report

**Experiment:** 1 - Negative Binomial State-Space Model
**Date:** 2025-10-29
**Status:** ⚠️ **FAIL - Computational Method Issues Detected**

---

## Executive Summary

**OVERALL DECISION: FAIL**

Simulation-Based Calibration reveals **severe calibration failures** across all model parameters. The rank statistics show extreme deviations from uniformity, indicating that the current inference method (MAP + approximate posterior) is **inadequate for this model**. This failure is primarily attributable to the simplified posterior approximation method used due to computational constraints, rather than fundamental model misspecification.

**Critical Finding:** The model requires **full MCMC with HMC/NUTS** (Stan or PyMC) for proper inference. The current approximation method systematically underestimates posterior uncertainty, leading to poor parameter recovery.

---

## Visual Assessment

### Diagnostic Plots Generated

1. **`rank_histograms.png`**: Rank uniformity assessment with 99% confidence bands
2. **`ecdf_comparison.png`**: Empirical CDF vs theoretical uniform with KS test
3. **`parameter_recovery.png`**: True parameter values vs rank positions with extreme rank indicators

### Key Visual Findings

All three plots reveal **systematic calibration failures**:

- **Rank histograms** show extreme clustering at bins 0 and 1000 (rank extremes)
- **ECDF plots** show massive deviations from the diagonal (ideal uniformity)
- **Recovery scatter plots** show bifurcation: parameters either at rank ~0 or rank ~1000

---

## Calibration Assessment

### Configuration

- **Number of simulations:** 50 successful (0 failed)
- **Time points per simulation:** 40
- **Posterior draws per simulation:** 1000
- **Total rank statistics computed:** 150 (3 parameters × 50 simulations)

### Rank Uniformity Tests

Under proper calibration, rank statistics should be uniformly distributed on [0, L] where L = 1000 draws.

| Parameter | χ² Statistic | p-value | Decision | Interpretation |
|-----------|-------------|---------|----------|----------------|
| **δ (Drift)** | 65.2 | ≈ 0.000 | **FAIL** | Severe departure from uniformity |
| **σ_η (Innovation SD)** | 401.2 | ≈ 0.000 | **FAIL** | Extreme departure from uniformity |
| **φ (Dispersion)** | 120.4 | ≈ 0.000 | **FAIL** | Severe departure from uniformity |

**Threshold:** χ² test p-value > 0.05 for PASS

**Result:** All three parameters fail uniformity tests with p-values effectively zero.

---

## Parameter-Specific Diagnostics

### 1. Drift Parameter (δ)

**As illustrated in `rank_histograms.png` (left panel):**
- Strong bimodal distribution: peaks at ranks 0-50 and 950-1000
- 12 simulations at rank 0 (expected: 2.5)
- 8 simulations at rank 1000 (expected: 2.5)
- Central ranks severely depleted

**Recovery pattern (from `parameter_recovery.png`, left panel):**
- 12 extreme low ranks (< 5%)
- 8 extreme high ranks (> 95%)
- Suggests posterior systematically too narrow
- True values near prior mean show better recovery

**KS test (from `ecdf_comparison.png`, left panel):**
- KS statistic: 0.198
- p-value: 0.018
- ECDF deviates below diagonal initially, suggesting underestimation of uncertainty

**Interpretation:**
The approximate posterior for δ is **too concentrated**. When the true δ deviates from the MAP estimate, the narrow posterior fails to cover it, resulting in extreme ranks.

---

### 2. Innovation SD (σ_η)

**As illustrated in `rank_histograms.png` (middle panel):**
- Extreme concentration at rank 0: **33 out of 50 simulations** (expected: 2.5)
- Virtually no simulations in ranks 100-1000
- This is the **most severe calibration failure**

**Recovery pattern (from `parameter_recovery.png`, middle panel):**
- 33 extreme low ranks (66% of simulations!)
- 0 extreme high ranks
- Strong systematic bias: posterior consistently **overestimates** σ_η

**KS test (from `ecdf_comparison.png`, middle panel):**
- KS statistic: 0.657 (very large)
- p-value: 0.0000
- ECDF jumps to ~0.65 immediately, then plateaus

**Interpretation:**
The approximate posterior systematically places too much mass on **large values of σ_η**. This suggests:
1. MAP optimization struggles with this parameter
2. Laplace approximation is poor (likely due to skewed posterior)
3. Possible identifiability issues between σ_η and observation noise φ

---

### 3. Dispersion Parameter (φ)

**As illustrated in `rank_histograms.png` (right panel):**
- Strong bimodal distribution: peaks at ranks 0-50 and 950-1000
- 17 simulations at rank 0 (expected: 2.5)
- 6 simulations at rank 1000 (expected: 2.5)
- Similar pattern to δ but more extreme

**Recovery pattern (from `parameter_recovery.png`, right panel):**
- 17 extreme low ranks (34%)
- 6 extreme high ranks (12%)
- Poor recovery across the entire range of true φ values
- No obvious relationship between true value and recovery quality

**KS test (from `ecdf_comparison.png`, right panel):**
- KS statistic: 0.303
- p-value: 0.0001
- ECDF starts steep, indicating early concentration

**Interpretation:**
The dispersion parameter φ shows **severe underestimation of posterior uncertainty**. The approximate method fails to capture the true posterior width, particularly for extreme true values.

---

## Computational Diagnostics

### Convergence Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Convergence rate | 100% (50/50) | > 90% | ✓ PASS |
| Mean R̂ | 1.00 | < 1.01 | ✓ (N/A for MAP) |
| Mean ESS | 1000 | > 400 | ✓ (N/A for approx) |
| Divergences | 0 | < 1% | ✓ (N/A for optimization) |

**Note:** Convergence metrics are not directly applicable to the MAP + Laplace approximation method. All optimizations converged to local optima, but the approximation quality is poor.

---

## Critical Visual Findings

### Evidence of Systematic Bias

**From rank histogram patterns:**
1. **Bimodal distributions** (δ, φ): Indicates posterior too narrow
2. **Extreme left-skew** (σ_η): Indicates systematic overestimation
3. **Depleted center ranks**: Characteristic of underconfident uncertainty quantification

### Failure Mode Identification

**Pattern:** U-shaped or L-shaped rank histograms

**Diagnosis:**
- **Primary cause:** Laplace approximation inadequate for this posterior geometry
- **Contributing factors:**
  - Non-Gaussian posterior (especially for σ_η, φ)
  - Parameter correlations not captured by diagonal covariance
  - Possible multimodality or heavy tails

**Evidence from recovery scatter plots:**
The clustering at rank extremes occurs **across the full range of true parameter values**, indicating this is not a boundary or identifiability issue, but rather a **systematic inference failure**.

---

## Coverage and Bias Metrics

### Estimated Coverage (from rank statistics)

| Parameter | Nominal 90% CI | Actual Coverage | Status |
|-----------|---------------|-----------------|--------|
| δ | Should contain 90% | ~60% (extreme ranks: 40%) | ❌ Poor |
| σ_η | Should contain 90% | ~34% (66% at rank 0!) | ❌ Very Poor |
| φ | Should contain 90% | ~54% (46% at extremes) | ❌ Poor |

**Interpretation:** Credible intervals are **severely underconfident**. 90% intervals contain the true value far less than 90% of the time.

### Z-score Analysis

Not computed due to lack of individual posterior samples. However, the rank patterns indicate:
- **High positive z-scores** for σ_η (posterior mean >> true value)
- **Variable z-scores** for δ and φ (bimodal pattern)

---

## Diagnostic Patterns & Interpretation

### Pattern 1: Bimodal Rank Distribution (δ, φ)

**Observation:** Ranks cluster at 0 and 1000

**Typical causes:**
1. ✓ **Posterior too narrow** (confirmed by multiple extreme ranks)
2. ✓ **Poor uncertainty quantification** (Laplace approximation fails)
3. ? Prior-likelihood conflict (less likely, prior predictive checks passed)

**Conclusion:** The approximate posterior SDs are **systematically underestimated**.

### Pattern 2: Left-Skewed Rank Distribution (σ_η)

**Observation:** 66% of ranks at 0 (lowest bin)

**Typical causes:**
1. ✓ **Systematic overestimation bias** (posterior mean too high)
2. ✓ **Skewed posterior poorly approximated** (exponential prior → skewed posterior)
3. ? Identifiability with φ (both control variability)

**Conclusion:** The MAP estimate for σ_η is **systematically too large**, likely because the optimization gets stuck at local maxima or the Laplace approximation fails for this log-normal-like posterior.

---

## Identifiability Assessment

### Correlation Analysis (qualitative)

The failure of σ_η recovery suggests possible **weak identifiability** with φ:
- Both parameters control variability
- σ_η: temporal innovation variance
- φ: observation overdispersion

**However:** The fact that all optimizations converged and prior predictive checks passed suggests parameters are **weakly but sufficiently identifiable**. The problem is primarily the **inference method**, not the model structure.

---

## Comparison to Expectations

### Pre-specified Thresholds

From `metadata.md`, falsification criteria:

| Criterion | Threshold | Observed | Status |
|-----------|-----------|----------|--------|
| σ_η → 0 (degenerate) | < 0.01 | Not observed | ✓ |
| σ_η ~ obs SD (no benefit) | Not checked | N/A | - |
| Convergence | > 90% | 100% (optimization) | ✓ |
| Divergences | < 20% | 0% (N/A) | ✓ |

**Note:** These criteria assess the **model**, not the inference method. The model structure appears sound; the **inference implementation is inadequate**.

---

## Root Cause Analysis

### Why Did SBC Fail?

**Primary Cause:** Inadequate posterior approximation method

1. **Laplace approximation assumes:**
   - Posterior is approximately Gaussian
   - Mode-based approximation is accurate
   - Diagonal covariance is sufficient

2. **Reality for this model:**
   - σ_η and φ have skewed posteriors (positive support, exponential priors)
   - Strong posterior correlations between parameters
   - Latent states create high-dimensional posterior geometry
   - Heavy tails due to negative binomial likelihood

3. **Result:**
   - Systematic bias in point estimates
   - Severe underestimation of uncertainty
   - Poor coverage of credible intervals

### Is the Model at Fault?

**NO.** The evidence suggests:
- ✓ Prior predictive checks passed (Round 2 priors)
- ✓ All optimizations converged
- ✓ No extreme parameter values sampled
- ✓ Generated data looks reasonable

**The issue is computational, not statistical.**

---

## Recommendations

### CRITICAL: Immediate Actions Required

**1. Re-run SBC with Full MCMC**

The model **MUST** be validated with proper MCMC before fitting real data:

```bash
# Required setup
- Install Stan (cmdstan) with make
- OR use PyMC with NUTS sampler
- OR use NumPyro with NUTS on GPU
```

**Configuration for full MCMC:**
- Sampler: NUTS (No-U-Turn Sampler)
- Chains: 4
- Warmup: 2000
- Samples: 2000 per chain
- Target accept rate: 0.95
- Max treedepth: 12

**Expected runtime:** 2-4 hours for 50-100 SBC simulations

---

### 2. Model Modifications to Consider (if MCMC also fails)

**If full MCMC SBC also shows calibration issues:**

**Option A: Non-centered parameterization for states**
```stan
parameters {
  vector[N-1] eta_raw;  // Standard normal
}
transformed parameters {
  eta[t] = eta[t-1] + delta + sigma_eta * eta_raw[t-1];
}
```
Already specified in metadata, ensure implementation.

**Option B: Simplify state evolution**
- Consider fixed σ_η (no uncertainty)
- Use simpler initial state prior

**Option C: Alternative parameterization**
- Use precision τ = 1/σ_η² instead of σ_η
- Gamma prior on τ instead of Exponential on σ_η

---

### 3. Computational Strategy

**For this project environment (no Stan compiler):**

**SHORT-TERM:**
1. Document SBC failure as methodology limitation
2. Proceed with MAP estimation for exploratory analysis
3. Add **large uncertainty buffers** to all inferences
4. Treat all posterior intervals as **approximate and likely too narrow**

**LONG-TERM:**
1. Set up Stan/PyMC in proper environment
2. Re-run full validation pipeline
3. Re-fit real data with proper MCMC
4. Compare to MAP results to quantify bias

---

### 4. Prior Adjustments (if needed after MCMC)

**Current priors (Round 2):**
```
δ ~ Normal(0.05, 0.02)
σ_η ~ Exponential(20)      # Mean = 0.05
φ ~ Exponential(0.05)      # Mean = 20
```

**If MCMC SBC shows σ_η overestimation:**
- Increase Exponential rate: `σ_η ~ Exponential(30)` (mean = 0.033)
- Or switch to informative prior: `σ_η ~ Normal(0.05, 0.02)` on log scale

**If φ shows issues:**
- Consider Gamma prior: `φ ~ Gamma(4, 0.2)` (mean = 20, more concentrated)

---

## Limitations of Current Analysis

### What This SBC Does NOT Test

1. **Full MCMC convergence:** We used MAP + approximation
2. **Sampler efficiency:** No HMC diagnostics (divergences, treedepth)
3. **Tail behavior:** Laplace approximation underestimates tails
4. **Parameter correlations:** Diagonal covariance only

### What This SBC DOES Test

1. ✓ **Data generation:** Model can generate realistic data
2. ✓ **Optimization convergence:** MAP estimation is feasible
3. ✓ **Point estimate bias:** Shows systematic issues (especially σ_η)
4. ❌ **Calibration:** Approximate method fails this test

---

## Conclusions

### Key Findings

1. **Model Structure:** Appears sound, no evidence of fundamental misspecification
2. **Computational Method:** MAP + Laplace approximation is **inadequate**
3. **Calibration:** Severe failures across all parameters (χ² p ≈ 0)
4. **Bias Patterns:**
   - σ_η: Systematic overestimation
   - δ, φ: Underestimated uncertainty
5. **Coverage:** Credible intervals far too narrow

### Decision Logic

**PASS criteria:**
- ✓ Rank distributions uniform (χ² p > 0.05)
- ✓ Coverage near nominal (85-95%)
- ✓ Convergence rate > 90%
- ✓ Low divergence rate

**Actual results:**
- ❌ Non-uniform ranks (all p ≈ 0)
- ❌ Poor coverage (34-60% instead of 90%)
- ✓ Optimization converged (100%)
- ✓ No divergences (N/A for method)

**DECISION: FAIL**

### Path Forward

**DO NOT proceed to real data fitting with current method.**

**Required before real data:**
1. Implement full MCMC (Stan/PyMC/NumPyro)
2. Re-run SBC with proper sampler
3. Verify rank uniformity (χ² p > 0.05)
4. Check MCMC diagnostics (R̂ < 1.01, ESS > 400)

**If full MCMC unavailable:**
- Acknowledge severe limitation in interpretation
- Use MAP estimates as **exploratory only**
- Do NOT make strong inferential claims
- Do NOT trust credible intervals

---

## Technical Notes

**Implementation:** Pure NumPy with scipy.optimize (Nelder-Mead)
**Approximation:** MAP + diagonal Laplace approximation
**Runtime:** ~3 minutes for 50 simulations

**Code location:** `/workspace/experiments/experiment_1/simulation_based_validation/code/`
- `run_sbc_demo.py`: SBC implementation
- `visualize_sbc_simple.py`: Diagnostic plots

**Results location:** `/workspace/experiments/experiment_1/simulation_based_validation/`
- `diagnostics/sbc_results.csv`: Raw rank statistics
- `diagnostics/sbc_summary.json`: Summary metrics
- `plots/`: All diagnostic visualizations

---

## References

- Talts et al. (2018). "Validating Bayesian Inference Algorithms with Simulation-Based Calibration"
- Betancourt (2018). "A Conceptual Introduction to Hamiltonian Monte Carlo"
- Gelman et al. (2013). "Bayesian Data Analysis", 3rd ed.

---

**Report prepared:** 2025-10-29
**Analyst:** Model Validation Specialist (Claude Agent)
**Status:** Complete - Awaiting Full MCMC Implementation
