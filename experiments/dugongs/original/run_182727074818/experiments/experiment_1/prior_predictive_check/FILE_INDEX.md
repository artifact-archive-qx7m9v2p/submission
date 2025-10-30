# File Index: Prior Predictive Check - Experiment 1

All files use absolute paths for easy access.

## Main Decision Documents

### Original Check (FAILED)
- Findings: `/workspace/experiments/experiment_1/prior_predictive_check/findings.md`
- Code: `/workspace/experiments/experiment_1/prior_predictive_check/code/run_prior_predictive_numpy.py`
- Diagnostics: `/workspace/experiments/experiment_1/prior_predictive_check/code/diagnostics.json`

### Revised Check (CONDITIONAL PASS)
- **START HERE - Final Decision:** `/workspace/experiments/experiment_1/prior_predictive_check/revised/FINAL_DECISION.md`
- README: `/workspace/experiments/experiment_1/prior_predictive_check/revised/README.md`
- Detailed Findings: `/workspace/experiments/experiment_1/prior_predictive_check/revised/findings.md`
- Comparison Document: `/workspace/experiments/experiment_1/prior_predictive_check/revised/comparison.md`
- Executive Summary: `/workspace/experiments/experiment_1/prior_predictive_check/REVISED_SUMMARY.md`

## Code Files

### Revised Analysis
- Main Script: `/workspace/experiments/experiment_1/prior_predictive_check/revised/code/run_revised_prior_predictive.py`
- Check 7 Analysis: `/workspace/experiments/experiment_1/prior_predictive_check/revised/code/analyze_check7.py`
- Final Assessment: `/workspace/experiments/experiment_1/prior_predictive_check/revised/code/final_assessment.py`
- Diagnostics JSON: `/workspace/experiments/experiment_1/prior_predictive_check/revised/code/revised_diagnostics.json`

### Alternative Versions Tested
- Revision v2 (sigma=0.10): `/workspace/experiments/experiment_1/prior_predictive_check/revised_v2/code/run_revised_v2_prior_predictive.py`

## Visualization Files

### Original Check Plots
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/parameter_plausibility.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_predictive_curves.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_predictive_coverage.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/predictions_at_key_x_values.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/extrapolation_diagnostic.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/monotonicity_diagnostic.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/comprehensive_summary.png`

### Revised Check Plots (USE THESE)
- **Before/After Comparison:** `/workspace/experiments/experiment_1/prior_predictive_check/revised/plots/prior_comparison_before_after.png`
- **Revised Curves:** `/workspace/experiments/experiment_1/prior_predictive_check/revised/plots/prior_predictive_curves_revised.png`
- **Coverage Improvement:** `/workspace/experiments/experiment_1/prior_predictive_check/revised/plots/coverage_diagnostic_improvement.png`
- **Check Results Bar Chart:** `/workspace/experiments/experiment_1/prior_predictive_check/revised/plots/check_results_comparison.png`
- **Comprehensive Summary:** `/workspace/experiments/experiment_1/prior_predictive_check/revised/plots/comprehensive_revised_summary.png`

## Quick Access Commands

### View main decision
```bash
cat /workspace/experiments/experiment_1/prior_predictive_check/revised/FINAL_DECISION.md
```

### View executive summary
```bash
cat /workspace/experiments/experiment_1/prior_predictive_check/REVISED_SUMMARY.md
```

### Re-run revised analysis
```bash
cd /workspace/experiments/experiment_1/prior_predictive_check/revised/code
python run_revised_prior_predictive.py
```

### View quantitative results
```bash
cat /workspace/experiments/experiment_1/prior_predictive_check/revised/code/revised_diagnostics.json | python -m json.tool
```

## Recommended Reading Order

1. `/workspace/experiments/experiment_1/prior_predictive_check/REVISED_SUMMARY.md` (5 min)
2. `/workspace/experiments/experiment_1/prior_predictive_check/revised/FINAL_DECISION.md` (15 min)
3. `/workspace/experiments/experiment_1/prior_predictive_check/revised/plots/check_results_comparison.png` (visual)
4. `/workspace/experiments/experiment_1/prior_predictive_check/revised/comparison.md` (detailed comparison)
5. `/workspace/experiments/experiment_1/prior_predictive_check/revised/findings.md` (full technical details)

## Final Approved Priors

Located in all decision documents above. For quick reference:

```stan
alpha ~ normal(2.0, 0.5);
beta ~ normal(0.3, 0.2);
c ~ gamma(2, 2);
nu ~ gamma(2, 0.1);
sigma ~ normal(0, 0.15);  // Half-Normal with lower=0 constraint
```

## Status Summary

- **Original Check:** FAILED (3/7 checks passed)
- **Revised Check:** CONDITIONAL PASS (6/7 checks passed)
- **Decision:** Proceed to simulation-based calibration
- **Date:** 2025-10-27
