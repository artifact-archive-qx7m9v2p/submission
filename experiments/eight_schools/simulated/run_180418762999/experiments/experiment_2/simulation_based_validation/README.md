# Simulation-Based Calibration Results

**Experiment**: 2 - Hierarchical Partial Pooling Model
**Date**: 2025-10-28
**Status**: ✅ **PASSED**

---

## Quick Summary

Simulation-based calibration successfully validated the hierarchical partial pooling model with non-centered parameterization. The MCMC sampler correctly recovers known parameters from synthetic data.

### Key Results

✅ **Rank uniformity**: PASS (μ: p=0.407, τ: p=0.534)
✅ **Coverage**: PASS (μ: 86.7%, τ: 90.0% at 90% level)
✅ **Bias**: PASS (μ: -0.96, τ: -1.74, within acceptable bounds)
✅ **Convergence**: EXCELLENT (0.00% divergences, R̂ ≤ 1.01, high ESS)
✅ **Funnel geometry**: MITIGATED (non-centered parameterization working)

**Decision**: Model is computationally sound and ready for inference on real data.

---

## Files

### Diagnostics
- `diagnostics/sbc_results.csv` - Raw results from 30 SBC simulations
- `diagnostics/summary_stats.json` - Summary statistics (rank tests, coverage, bias, convergence)

### Visualizations
- `plots/rank_histogram.png` - Rank uniformity tests (primary validation criterion)
- `plots/parameter_recovery.png` - True vs recovered parameters, bias analysis
- `plots/coverage_analysis.png` - Credible interval coverage (overall and stratified)
- `plots/convergence_summary.png` - Divergences, R-hat, ESS diagnostics
- `plots/funnel_diagnostics.png` - Hierarchical model specific checks

### Code
- `code/sbc_validation_fixed.py` - Main SBC implementation (PyMC 5.x)
- `code/create_visualizations.py` - Visualization generation script
- `code/run_sbc.sh` - Shell script to run SBC with correct Python path

### Reports
- `recovery_metrics.md` - **Comprehensive metrics report with visual evidence** (READ THIS)

---

## Key Findings

### Strengths
1. **Excellent statistical calibration**: Both μ and τ show uniform rank distributions
2. **Minimal computational issues**: Only 2 divergences across 30 simulations (0.00%)
3. **High efficiency**: ESS of 3260 for μ, 1788 for τ
4. **Non-centered parameterization success**: No funnel geometry at τ → 0

### Limitations
1. **τ identifiability with n=8**: Limited precision for extreme heterogeneity (τ > 10)
2. **Slight shrinkage**: Negative bias of -1.74 for τ, driven by high true values
3. **Coverage degrades for extreme τ**: 70% coverage when τ > 10 (vs 100% for τ < 10)

These limitations are **inherent to small-sample hierarchical models** (n=8 groups), not computational failures. The model appropriately regularizes and expresses uncertainty.

---

## Recommendations for Real Data

1. **Use validated settings**: 1000+ draws, 4 chains, target_accept=0.95
2. **Expect smooth MCMC**: <1% divergences, excellent convergence
3. **Interpret τ cautiously**: If posterior suggests very high τ (>10), recognize identifiability limits
4. **Focus on key question**: Is τ clearly positive vs near-zero? (This IS testable with n=8)

---

## Interpretation

**This SBC validates the computational machinery is working correctly.**

The model will:
- Provide well-calibrated inference on population mean (μ)
- Identify whether between-group heterogeneity exists (τ > 0 vs τ ≈ 0)
- Shrink estimates of extreme heterogeneity toward moderate values (appropriate regularization)

**Now proceed to fit real data and see what it says about the 8-school problem.**

---

## Technical Details

- **Method**: Simulation-Based Calibration (Talts et al. 2018)
- **Simulations**: 30 successful iterations
- **MCMC**: PyMC 5.26.1, NUTS sampler, non-centered parameterization
- **Computational time**: 6.6 minutes (395 seconds)
- **Prior**: μ ~ N(10, 20), τ ~ Half-Normal(0, 10)

---

For complete analysis with visual evidence and detailed interpretation, see `recovery_metrics.md`.
