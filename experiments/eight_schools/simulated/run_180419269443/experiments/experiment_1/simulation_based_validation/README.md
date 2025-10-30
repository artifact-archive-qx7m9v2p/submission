# Simulation-Based Calibration - Experiment 1

**Status:** ✅ **PASS** (100/100 simulations successful)

## Quick Summary

The hierarchical normal model successfully passed all simulation-based calibration checks:

- **Coverage rates**: μ=94%, τ=95%, θ=93.5% (target: 95%)
- **Bias**: μ=-4.3%, τ=-1.4% (both negligible)
- **SBC rank uniformity**: PASS (χ²=13.6 for μ, 12.4 for τ)
- **MCMC efficiency**: ESS=415 (μ), 55 (τ)

**Decision: Safe to proceed with real data fitting.**

---

## Files in This Directory

### Code (`code/`)
- **sbc_gibbs_sampler.py** - Main SBC implementation (USED)
- **hierarchical_model.stan** - Stan model specification (reference)
- **create_visualizations.py** - Generate diagnostic plots
- **sbc_results.csv** - Full results from 100 simulations
- **summary_statistics.json** - Aggregate metrics
- **theta_recovery_examples.json** - Detailed recovery examples
- **rank_statistics.npz** - SBC rank data

### Plots (`plots/`)
1. **parameter_recovery.png** - Main recovery quality (R²=0.967 for μ, 0.530 for τ)
2. **sbc_rank_histograms.png** - Calibration uniformity (KEY: confirms proper calibration)
3. **shrinkage_recovery.png** - Hierarchical structure validation (6 examples)
4. **bias_and_coverage.png** - Systematic error detection
5. **mcmc_diagnostics.png** - Computational quality metrics

### Report
- **recovery_metrics.md** - Complete calibration report with detailed analysis

---

## Key Findings

### What Worked Well ✅
1. **Near-perfect μ recovery**: R²=0.967, slope≈1.0
2. **Uniform SBC ranks**: Strong evidence of proper calibration
3. **Appropriate hierarchical shrinkage**: Study effects correctly pooled toward population mean
4. **Stable computation**: 100% success rate, no divergences
5. **Adequate ESS**: Sufficient for reliable inference

### What to Monitor 🔍
1. **τ ESS is modest (55)**: Functional but shows autocorrelation (typical for variance parameters)
2. **MH acceptance rate (15.5%)**: Slightly below optimal but still effective
3. **τ shows some shrinkage bias**: Small (-1.4%), expected for hierarchical variance parameters

### What This Validates ✓
- Model can recover known parameters across full prior range
- 95% credible intervals have proper 95% coverage
- Hierarchical structure correctly implemented
- Inference method (Gibbs sampling) is reliable
- Computation is stable and efficient enough

---

## Method

**Model:**
```
y_i ~ Normal(θ_i, σ_i)     [known σ_i]
θ_i ~ Normal(μ, τ)
μ ~ Normal(0, 25)
τ ~ Half-Normal(0, 10)
```

**Inference:** Gibbs sampling with Metropolis-Hastings for τ
- 4,000 iterations (1,000 warmup, 3,000 retained)
- Custom implementation in Python

**Validation:** 100 SBC iterations
- Draw parameters from prior → Generate synthetic data → Fit model → Check recovery

---

## Next Steps

**✅ Proceed to real data fitting** using:
- Same model specification
- Same inference configuration (4,000 iterations)
- Monitor ESS and convergence diagnostics
- Expect reliable estimates with well-calibrated uncertainty

---

## Key Diagnostic Insights

### From SBC Rank Histograms (Most Important!)
The uniform distribution of ranks is the **gold standard** evidence that:
1. Posterior means are unbiased
2. Posterior uncertainties are correctly calibrated
3. No systematic over- or under-dispersion
4. Model is fit-for-purpose

### From Parameter Recovery Plots
- μ shows near-1:1 recovery (excellent)
- τ shows some shrinkage (expected for variance parameters)
- CI widths appropriately scale with true τ
- Coverage stable across parameter ranges

### From Shrinkage Plots
- Hierarchical pooling works correctly
- Study effects appropriately pulled toward population mean
- Extreme values show more shrinkage (correct behavior)
- Individual failures are random, not systematic

---

**Generated:** 2025-10-28
**Run time:** ~8 minutes
**Random seed:** 2025
**Validation status:** ✅ CLEARED FOR REAL DATA
