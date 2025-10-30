# Simulation-Based Calibration Results

**Experiment 1:** Negative Binomial State-Space Model  
**Date:** 2025-10-29  
**Status:** ⚠️ **FAIL** - Requires Full MCMC Implementation

---

## Quick Summary

**Decision: FAIL**

The model **cannot be validated** with the current approximate inference method (MAP + Laplace approximation). All parameters show severe calibration failures:

- **δ (drift):** χ² = 65.2, p ≈ 0.000
- **σ_η (innovation SD):** χ² = 401.2, p ≈ 0.000  
- **φ (dispersion):** χ² = 120.4, p ≈ 0.000

**Root cause:** Laplace approximation inadequate for this posterior geometry.

**Required action:** Re-run with full MCMC (HMC/NUTS) using Stan or PyMC.

---

## Directory Contents

### Code (`code/`)

1. **`nb_state_space_model.stan`**
   - Stan model specification (requires cmdstan with make)
   - Non-centered parameterization
   - Ready for full MCMC when environment available

2. **`run_sbc_demo.py`**
   - Fast SBC implementation (used for current results)
   - MAP + Laplace approximation
   - 50 simulations, ~3 minutes runtime

3. **`visualize_sbc_simple.py`**
   - Generates 3 diagnostic plots
   - Rank histograms, ECDF comparison, recovery scatter

4. **`run_sbc.py`** (alternative)
   - Full MCMC-based SBC (requires CmdStanPy)
   - 100 simulations, proper posterior sampling
   - Use when Stan is available

5. **`run_sbc_numpy.py`** (alternative)
   - Custom MCMC implementation
   - Metropolis-Hastings sampler
   - Slow but works without external dependencies

### Diagnostics (`diagnostics/`)

- **`sbc_results.csv`**: Raw results (50 simulations × 3 parameters)
- **`sbc_summary.json`**: Statistical test results and decision

### Plots (`plots/`)

1. **`rank_histograms.png`**
   - Shows severe deviations from uniformity
   - Red bars = outside 99% confidence interval
   - All three parameters fail

2. **`ecdf_comparison.png`**
   - Empirical CDF vs theoretical uniform
   - Large KS statistics confirm poor calibration

3. **`parameter_recovery.png`**
   - True values vs rank positions
   - Extreme clustering at ranks 0 and 1000
   - Evidence of systematically narrow posteriors

### Reports

- **`recovery_metrics.md`**: Complete analysis (THIS DOCUMENT)
  - Detailed findings for each parameter
  - Visual evidence citations
  - Root cause analysis
  - Recommendations

---

## Key Findings

### Calibration Failures

**Pattern 1: Bimodal ranks (δ, φ)**
- Ranks cluster at 0 and 1000
- Indicates posterior too narrow
- ~40-45% of simulations at rank extremes

**Pattern 2: Left-skewed ranks (σ_η)**
- 66% of ranks at 0 (should be uniform!)
- Systematic overestimation of innovation SD
- Worst calibration failure

### Implications

1. **Current method unreliable:** Cannot trust credible intervals
2. **Coverage poor:** 90% CIs contain truth only ~35-60% of time
3. **Bias present:** Especially in σ_η (overestimated)
4. **Model OK:** Problem is inference method, not model structure

---

## Recommendations

### Immediate (Required)

**DO NOT fit real data with current method**

Options:
1. Set up Stan/PyMC in environment with compiler
2. Re-run SBC with full MCMC (HMC/NUTS)
3. Verify χ² p-values > 0.05 before proceeding

### If MCMC Unavailable

If forced to proceed with limitations:
1. Use MAP estimates for **exploratory analysis only**
2. Apply **large uncertainty buffers** (e.g., 3× reported SEs)
3. **Do not make inferential claims**
4. Clearly document limitation

### For Production Use

Required validation pipeline:
```
1. Prior predictive check → PASS ✓ (completed)
2. SBC with full MCMC → PENDING (current: FAIL)
3. Fit to real data → BLOCKED
4. Posterior predictive check → BLOCKED
5. Model critique → BLOCKED
```

---

## How to Re-run with Full MCMC

### Option 1: Stan (Recommended)

```bash
# Install cmdstan
conda install -c conda-forge cmdstan

# Or manual install
cd /tmp
wget https://github.com/stan-dev/cmdstan/releases/download/v2.33.0/cmdstan-2.33.0.tar.gz
tar -xzf cmdstan-2.33.0.tar.gz
cd cmdstan-2.33.0
make build

# Set environment variable
export CMDSTAN=/tmp/cmdstan-2.33.0

# Run SBC
cd /workspace/experiments/experiment_1/simulation_based_validation
python3 code/run_sbc.py
```

### Option 2: PyMC

```bash
# Install PyMC
pip install pymc pymc-experimental

# Create PyMC version of model
# (would need to write pymc_sbc.py)
```

### Option 3: NumPyro (if GPU available)

```bash
pip install numpyro jax
# Fastest option with proper NUTS implementation
```

---

## Technical Details

**SBC Theory:**
- For well-calibrated inference, rank statistics uniform on [0, L]
- L = number of posterior draws (1000 here)
- Test uniformity with χ² test (20 bins)
- p-value > 0.05 → PASS

**Our Results:**
- All p-values ≈ 0.000 → strong evidence of miscalibration
- Not due to random chance (signal is very strong)
- Requires methodological fix, not just more simulations

**Why Laplace Failed:**
- Assumes Gaussian posterior (false for σ_η, φ)
- Ignores parameter correlations
- Mode-based approximation poor for skewed distributions
- Underestimates tail probabilities

---

## Files Reference

All paths relative to: `/workspace/experiments/experiment_1/simulation_based_validation/`

**Main report:** `recovery_metrics.md` (detailed analysis)  
**This file:** `README.md` (quick reference)  
**Raw data:** `diagnostics/sbc_results.csv`  
**Visual evidence:** `plots/*.png`  

---

## Contact & Next Steps

**Current status:** Validation blocked by computational limitations

**Next step:** Obtain environment with Stan/PyMC compiler

**Timeline:**
- With Stan: ~2-4 hours for full SBC
- Expected outcome: Either PASS (proceed to data) or identify specific model issues

**Fallback:** If MCMC impossible, document as major limitation and proceed with caution

---

**Last updated:** 2025-10-29  
**Prepared by:** Model Validation Specialist
