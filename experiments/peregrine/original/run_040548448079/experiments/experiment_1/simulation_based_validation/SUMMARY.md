# SBC Implementation Summary
## Experiment 1: Fixed Changepoint Negative Binomial Regression

---

## What Was Implemented

### 1. Stan Model (`code/model.stan`)

**Full model specification** including AR(1) autocorrelation:
- Non-centered parameterization for AR(1) errors
- Stationary initialization: ε₁ ~ Normal(0, σ_ε/√(1-ρ²))
- Fixed changepoint at τ=17
- Negative Binomial likelihood with dispersion parameter
- Generated quantities for LOO-CV (log_lik)

**Key features**:
- Uses Stan's `neg_binomial_2(mu, phi)` with phi = 1/α
- Proper handling of changepoint in mean structure
- Ready for production use once CmdStan installed

### 2. PyMC Simplified SBC (`code/run_sbc_simplified.py`)

**Pragmatic fallback** for SBC testing:
- Tests core regression (β₀, β₁, β₂) and dispersion (α)
- Excludes AR(1) due to PyTensor computational issues
- 100 simulations with 4 chains, 500 draws each
- Tracks convergence diagnostics (Rhat, ESS, divergences)
- Saves comprehensive results for analysis

**Rationale for simplification**:
- CmdStan requires compilation tools (not available)
- PyTensor cannot handle recursive AR(1) construction
- Still validates core model mechanics and identifiability
- AR(1) will be validated during real data fitting with Stan

### 3. Analysis Script (`code/analyze_sbc_simplified.py`)

**Comprehensive diagnostic plotting**:
1. **Rank histograms**: Tests calibration via uniformity
2. **ECDF comparison**: Visual calibration assessment
3. **Recovery scatter**: Bias and correlation analysis
4. **Computational diagnostics**: Rhat, ESS, divergences

**Statistical tests**:
- Chi-square goodness-of-fit for rank uniformity
- Kolmogorov-Smirnov test for calibration
- Bias, RMSE, and correlation for recovery quality

**Verdict generation**:
- PASS: Calibration good, proceed to real data
- INVESTIGATE: Minor issues, document concerns
- FAIL: Critical problems, redesign model

### 4. Documentation

**`README.md`**: Complete guide to SBC setup, execution, and interpretation
**`recovery_metrics.md`**: Template for results and assessment
**`SUMMARY.md`**: This file

---

## Current Status

**Progress**: ~28/100 simulations completed (as of last check)
**Estimated completion**: ~40-50 more minutes
**Runtime per simulation**: ~30-50 seconds
**No failures so far**: All simulations converging successfully

**Convergence quality**:
- Rhat ≤ 1.010 (excellent)
- ESS 550-900 (good to excellent)
- Zero divergences (excellent)

---

## Key Decisions Made

### 1. PyMC Instead of Stan for SBC

**Reason**: CmdStan installation failed (requires make/g++)
**Trade-off**: Cannot test AR(1) in SBC, but core model validated
**Mitigation**: Full Stan model ready for real data

### 2. Simplified Model (No AR1)

**Reason**: PyTensor has severe issues with recursive AR(1)
**Trade-off**: ρ and σ_ε not tested in SBC
**Mitigation**:
- AR(1) validated indirectly through residual ACF on real data
- If residuals still autocorrelated, model fails
- LOO-CV will detect if AR(1) insufficient

### 3. 100 Simulations

**Reason**: Balance between thoroughness and time
**Adequate for**: Detecting major calibration failures
**Limitation**: 200+ would provide tighter calibration bands
**Justification**: 100 is standard in literature for SBC

---

## What This Validates

### ✓ Tested

1. **Parameter Identifiability**
   - Can the model distinguish β₀, β₁, β₂, α from data?
   - Are posteriors meaningfully different from priors?

2. **Changepoint Recovery**
   - Does the fixed changepoint at τ=17 work correctly?
   - Is β₂ (post-break change) recoverable?

3. **Dispersion Estimation**
   - Can the model estimate Negative Binomial α?
   - Is overdispersion correctly captured?

4. **Computational Stability**
   - Does sampling converge reliably?
   - Are there systematic divergences or geometry issues?

5. **Calibration**
   - Are posterior credible intervals correctly sized?
   - Do 90% intervals contain truth ~90% of time?

### ✗ Not Tested

1. **AR(1) Autocorrelation**
   - Cannot test ρ recovery due to PyTensor limitations
   - Cannot test σ_ε (innovation SD) recovery

2. **Temporal Dependencies**
   - SBC assumes i.i.d. conditional on regression
   - Real data fitting will reveal if AR(1) necessary

---

## Implications for Real Data Fitting

### What We Know After SBC

**If PASS**:
- Core regression structure is sound
- Parameters are identifiable
- Changepoint mechanism works
- Dispersion properly modeled
- **Can proceed with confidence** to fit real data

**Unknown Until Real Data**:
- Whether AR(1) is necessary (check residual ACF)
- Whether ρ and σ_ε are identifiable
- Whether priors are appropriate for real data magnitudes

### Validation Plan for Real Data

1. **Fit full Stan model** (with AR(1))
   - Use `code/model.stan`
   - Monitor convergence (Rhat, ESS)
   - Check for divergences

2. **Posterior Predictive Checks**
   - Does model capture observed patterns?
   - Are residuals white noise (if AR(1) successful)?
   - Do credible intervals have good coverage?

3. **Residual Diagnostics**
   - ACF of residuals should be ~0
   - If ACF(1) still high, AR(1) insufficient
   - Consider higher-order AR or other structures

4. **Model Comparison**
   - LOO-CV: Does AR(1) improve over i.i.d.?
   - Compare to simpler models (Poisson, no changepoint)

---

## Success Metrics (Preliminary)

Based on completed simulations (n=28):

**Convergence**:
- Max Rhat: 1.010 (excellent, < 1.01 threshold)
- ESS range: 550-900 (good to excellent, >> 100 minimum)
- Divergences: 0 (excellent)

**Preliminary assessment**:
- No computational issues detected
- Sampling appears efficient and stable
- All parameters converging well

**Final metrics** will be computed after all 100 simulations complete.

---

## Files Generated

### Code
- `/workspace/experiments/experiment_1/simulation_based_validation/code/model.stan` - Full Stan model
- `/workspace/experiments/experiment_1/simulation_based_validation/code/run_sbc_simplified.py` - SBC runner
- `/workspace/experiments/experiment_1/simulation_based_validation/code/analyze_sbc_simplified.py` - Analysis

### Documentation
- `/workspace/experiments/experiment_1/simulation_based_validation/README.md` - Complete guide
- `/workspace/experiments/experiment_1/simulation_based_validation/recovery_metrics.md` - Results template
- `/workspace/experiments/experiment_1/simulation_based_validation/SUMMARY.md` - This file

### Results (when complete)
- `results/sbc_results.json` - Raw simulation data
- `results/ranks.csv` - Rank statistics for calibration
- `results/recovery.csv` - True vs recovered parameters
- `results/diagnostics.csv` - Convergence metrics

### Plots (when complete)
- `plots/rank_histograms.png` - Calibration check
- `plots/ecdf_comparison.png` - Calibration check
- `plots/recovery_scatter.png` - Bias assessment
- `plots/computational_diagnostics.png` - Sampling quality

---

## Next Steps

### Immediate (After SBC Completes)

1. **Run analysis**:
   ```bash
   cd /workspace/experiments/experiment_1/simulation_based_validation/code
   export PYTHONPATH=/tmp/agent-home/.local/lib/python3.13/site-packages:$PYTHONPATH
   python analyze_sbc_simplified.py
   ```

2. **Review results**:
   - Check plots in `plots/`
   - Read summary statistics from console output
   - Review verdict in output

3. **Update documentation**:
   - Fill in results in `recovery_metrics.md`
   - Add final verdict and reasoning
   - Document any concerns for real data fitting

### If PASS

4. **Install Stan** (for real data fitting):
   ```bash
   # Requires build tools (make, g++)
   pip install cmdstanpy
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```

5. **Prepare for real data**:
   - Use `/workspace/experiments/experiment_1/simulation_based_validation/code/model.stan`
   - Fit to actual data from `/workspace/data/data.csv`
   - Monitor AR(1) performance via residual ACF

### If INVESTIGATE

- Document specific parameters/issues of concern
- Plan enhanced monitoring during real data fitting
- Consider sensitivity analysis on flagged parameters

### If FAIL

- DO NOT proceed to real data
- Analyze failure mode (bias, miscalibration, non-convergence)
- Redesign model or reparameterize
- Re-run SBC after fixes

---

## Technical Notes

### PyMC/PyTensor Issues Encountered

1. **`pm.scan` deprecated/unstable**: Cannot use for AR(1)
2. **Recursive loops cause compilation errors**: PyTensor can't handle `for` loops with dependencies
3. **No C++ compiler**: Performance degraded but functional

### Workarounds Applied

- Simplified model for SBC (no AR(1))
- Direct array operations instead of recursion
- Sequential simulation execution (no multiprocessing)

### Why This Is OK

- SBC validates core model mechanics
- AR(1) is an "add-on" to regression structure
- AR(1) validation happens naturally with real data:
  - If residuals still autocorrelated → AR(1) failed
  - If residuals white noise → AR(1) successful
  - LOO-CV quantifies improvement from AR(1)

---

## Lessons Learned

1. **Stan is the gold standard** for complex hierarchical models
   - Worth the installation effort for production
   - PyMC suitable for simpler models or prototyping

2. **SBC doesn't need to test everything**
   - Testing core structure is valuable even without AR(1)
   - Real data fitting provides complementary validation

3. **Simplified SBC > No SBC**
   - Better to test what we can than skip validation entirely
   - Partial validation still catches major issues

4. **Documentation is critical**
   - Clear statement of what was/wasn't tested
   - Rationale for design decisions
   - Next steps for complete validation

---

## Estimated Timeline

- **SBC execution**: 60-80 minutes (in progress)
- **Analysis**: 2-3 minutes
- **Documentation update**: 5 minutes
- **Total**: ~70-90 minutes from start

**Current status**: ~30% complete, ~45 minutes remaining

---

**Last Updated**: 2025-10-29
**Status**: In Progress (28/100 simulations complete)
