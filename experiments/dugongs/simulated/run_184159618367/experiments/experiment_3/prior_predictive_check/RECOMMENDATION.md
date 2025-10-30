# Prior Predictive Check Recommendation

**Date**: 2025-10-27
**Model**: Experiment 3 - Log-Log Power Law Model
**Status**: ⚠️ **REVISE PRIORS**

---

## Quick Summary

The prior predictive check **FAILED** the 80% trajectory plausibility threshold:
- **Trajectory pass rate**: 62.8% (need >80%)
- **Root causes**: Heavy-tailed σ prior (5.7% > 1.0) and negative β values (11.8%)
- **Good news**: Model structure is sound, only minor prior adjustments needed

---

## Recommended Prior Revisions

### Current (Problematic)
```stan
α ~ normal(0.6, 0.3);
β ~ normal(0.12, 0.1);
σ ~ cauchy(0, 0.1);  // Half-Cauchy
```

### Revised (Recommended)
```stan
α ~ normal(0.6, 0.3);      // Keep as is
β ~ normal(0.12, 0.05);    // Tighten: 0.1 → 0.05
σ ~ cauchy(0, 0.05);       // Tighten: 0.1 → 0.05
```

### Expected Improvement
- Negative β: 11.8% → ~0.9% (10× reduction)
- Extreme σ: 5.7% → ~0.5% (10× reduction)
- Trajectory pass rate: 62.8% → ~85-90% (PASS threshold)

---

## Key Findings

### What Works ✓
- **Model structure**: Log-log transformation appropriate
- **α prior**: Well-calibrated, centered on observed range
- **Overall plausibility**: 89% of individual predictions are reasonable
- **Computational health**: No NaN/Inf issues
- **Coverage**: Prior predictions appropriately bracket observed data

### What Needs Fixing ⚠️
1. **β prior too wide**: Allows 11.8% negative values (decreasing trends contradict data)
2. **σ heavy tail**: Half-Cauchy(0, 0.1) creates extreme predictions (max = 10^41)
3. **Compounding effects**: Wide priors interact to create implausible trajectories

---

## Visual Evidence

### Key Plots (all in `/workspace/experiments/experiment_3/prior_predictive_check/plots/`)

1. **prior_predictive_coverage.png**: Shows 37% of trajectories are implausible (red lines)
2. **heavy_tail_diagnostics.png**: Identifies σ and β problems clearly
3. **prior_revision_comparison.png**: Demonstrates impact of recommended changes
4. **parameter_plausibility.png**: Shows negative β mass and σ heavy tail
5. **pointwise_plausibility.png**: Pass rate degrades from 95% (x=1) to 85% (x=30)

---

## Next Steps

### 1. Revise Priors (5 minutes)
Update the Stan model file with revised prior specifications above.

### 2. Re-run Prior Predictive Check (5 minutes)
```bash
python experiments/experiment_3/prior_predictive_check/code/prior_predictive_check.py
python experiments/experiment_3/prior_predictive_check/code/visualize_priors.py
```

### 3. Verify Pass Criteria
Check that revised priors achieve:
- [ ] Trajectory pass rate >80%
- [ ] Negative β <5%
- [ ] Extreme σ (>1.0) <5%
- [ ] No numerical issues

### 4. Proceed to SBC
Once prior predictive check passes, proceed to simulation-based calibration.

---

## Detailed Analysis

See full findings in: `/workspace/experiments/experiment_3/prior_predictive_check/findings.md`

---

## Contact

For questions about this assessment, review the diagnostic plots and detailed findings document.
