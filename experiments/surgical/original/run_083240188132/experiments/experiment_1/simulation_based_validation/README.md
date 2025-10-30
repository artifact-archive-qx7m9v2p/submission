# Simulation-Based Calibration Validation - Experiment 1

## Status: FAIL - DO NOT PROCEED TO REAL DATA

This directory contains the complete simulation-based calibration (SBC) validation for the Beta-Binomial hierarchical model.

## Critical Finding

**The model cannot reliably recover parameters when overdispersion is high (φ > 0.5), which is precisely the regime our data occupies.**

### Key Issues
- Only 52% of simulations converged (target: 80%)
- κ recovery error of 104-128% in moderate/high overdispersion scenarios
- Worst performance in Scenario C (κ=0.3, φ=0.77) which matches our data

### What This Means
**DO NOT fit this model to real data.** It will produce unreliable estimates, poor convergence, and untrustworthy inferences.

## Directory Structure

```
simulation_based_validation/
├── code/
│   └── sbc_validation.py          # Complete SBC implementation (50 sims + scenarios)
├── plots/
│   ├── sbc_rank_histograms.png    # Rank uniformity test (PASS)
│   ├── parameter_recovery.png     # True vs recovered scatter plots
│   ├── coverage_assessment.png    # Credible interval coverage (90-92%, PASS)
│   ├── bias_distribution.png      # Bias distributions (centered, PASS)
│   └── scenario_recovery.png      # Focused scenario results (FAIL)
├── sbc_results.csv                # 50 simulation results
├── scenario_results.csv           # 15 focused scenario results
├── sbc_output.log                 # Complete execution log
├── recovery_metrics.md            # COMPREHENSIVE REPORT - READ THIS
├── decision.txt                   # Contains "FAIL"
└── README.md                      # This file
```

## Quick Assessment

| Criterion | Result | Status |
|-----------|--------|--------|
| Coverage (90% CI) | 90-92% | PASS |
| Rank uniformity | p > 0.55 | PASS |
| Convergence rate | 52% | FAIL |
| Scenario recovery | 104-128% error | FAIL |
| **Overall** | **6/10 criteria** | **FAIL** |

## What to Read

1. **START HERE**: `recovery_metrics.md` - Comprehensive 400+ line report with:
   - Visual assessment of all plots
   - Detailed diagnostics for each failure mode
   - Root cause analysis
   - Recommended fixes

2. **PLOTS**: Visual evidence in `plots/` directory
   - Most critical: `scenario_recovery.png` shows catastrophic κ failure

3. **RAW DATA**: CSV files contain all numerical results for further analysis

## Recommended Next Steps

1. DO NOT proceed to fit real data with current model
2. Implement reparameterization (non-centered or mean-precision)
3. Re-run SBC validation on modified model
4. If still failing, consider alternative model class (logistic-normal hierarchy)

## Reproducibility

```bash
cd /workspace
python experiments/experiment_1/simulation_based_validation/code/sbc_validation.py
```

**Runtime**: ~2 hours
**Random seed**: 42 (fixed for reproducibility)

---

**Validation completed**: 2025-10-30
**Decision**: FAIL - Model unsuitable for high-overdispersion data
**Analyst**: Claude (SBC Validation Specialist)
