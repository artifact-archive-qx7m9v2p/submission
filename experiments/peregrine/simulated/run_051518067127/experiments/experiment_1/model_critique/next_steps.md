# Next Steps After Model Rejection

**Decision**: REJECT
**Date**: 2025-10-30

---

## Why No Improvement Priorities Document?

The `improvement_priorities.md` file is created when the decision is **REVISE** - indicating that the model can be improved with specific modifications.

For this experiment, the decision is **REJECT**, which means:
- The core model structure (independence assumption) is fundamentally inadequate
- Minor revisions (adjusting priors, adding polynomial terms) will not address the root cause
- A different model class is needed (one with temporal dependence)

---

## Proceed to Experiment 2

**Next Experiment**: Experiment 2 - AR(1) Log-Normal with Regime-Switching

**Location**: `/workspace/experiments/experiment_2/` (to be created)

**Why This Model?**
1. **Addresses the autocorrelation issue** with explicit AR(1) structure:
   ```
   mu[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2 + phi * epsilon[t-1]
   ```

2. **Already designed and planned** in `experiment_plan.md` (lines 95-154)

3. **Expected to succeed** based on:
   - Data ACF lag-1 = 0.926 (strong persistence)
   - EDA showed log-scale R² = 0.937 (log transform works well)
   - Regime-switching addresses heterogeneity

**Key Differences from Experiment 1**:
- Likelihood: Log-Normal (continuous on log scale) vs Negative Binomial (count scale)
- Temporal structure: AR(1) errors vs independence
- Variance: Regime-specific sigma vs constant phi
- Complexity: 7 parameters vs 4 parameters

---

## What to Preserve from Experiment 1

### Keep for Comparison
1. **All outputs** in `/workspace/experiments/experiment_1/`
   - Posterior samples for LOO-CV comparison
   - Diagnostic plots showing failure mode
   - Parameter estimates as baseline

2. **Log-likelihood** (saved in `posterior_inference.netcdf`)
   - Needed for LOO-CV model comparison
   - Will compare ELPD against Experiment 2

3. **Lessons Learned**:
   - Prior structure is sound (can adapt for Experiment 2)
   - PyMC workflow is solid (runs fast, converges well)
   - Negative Binomial handles overdispersion (but is not enough)

### Scientific Value
- Demonstrates cost of independence assumption (p < 0.001)
- Establishes baseline performance (MAE = 16.41)
- Shows that mean trend alone is insufficient (temporal structure matters)

---

## Implementation Plan for Experiment 2

### Phase A: Prior Predictive Check
- Adapt priors from Experiment 1 (alpha, beta_1, beta_2 similar)
- Add priors for AR coefficient (phi ~ Uniform(-0.95, 0.95))
- Add priors for regime variances (sigma_regime ~ HalfNormal(0, 1))
- Test on log-scale (continuous, not count)

### Phase B: Simulation-Based Validation
- Generate data with known AR(1) structure
- Test recovery of phi parameter (critical for autocorrelation)
- Verify regime-specific variances are identifiable

### Phase C: Posterior Inference
- Fit to real data (N=40 observations)
- Expect residual ACF < 0.3 (below threshold)
- Check if phi is significantly different from 0

### Phase D: Posterior Predictive Check
- Test autocorrelation (expect p > 0.05)
- Test marginal distribution (should still match)
- Check if residual ACF < 0.5 (pass criterion)

### Phase E: Model Critique
- Compare to Experiment 1 via LOO-CV
- Assess if AR structure is sufficient
- Decide: ACCEPT / REVISE / REJECT

---

## Expected Outcome

**Most Likely**: Experiment 2 passes all diagnostics and is accepted
- Residual ACF < 0.3
- PPC autocorrelation test passes (p > 0.05)
- Better LOO than Experiment 1
- No falsification criteria met

**If Experiment 2 Also Fails**:
- Proceed to Experiment 3 (Changepoint Negative Binomial)
- Or Experiment 4 (Gaussian Process)
- Depends on nature of failure

**Stopping Rule** (from `experiment_plan.md`):
> "Stop early if Experiment 1 or 2 achieves: R-hat < 1.01, ESS > 400, Residual ACF < 0.3, Posterior predictive MAE < 20, No systematic bias in PPCs"

If Experiment 2 meets these criteria, we can stop and declare it adequate.

---

## Timeline

**Current Status**: Experiment 1 complete (rejected as expected)

**Next Steps**:
1. Model Designer creates Experiment 2 specification (if not already done)
2. Prior Predictive Checker runs Phase A
3. Simulation Validator runs Phase B
4. Model Fitter runs Phase C
5. Posterior Predictive Checker runs Phase D
6. Model Critic (this role) runs Phase E

**Estimated Time**: 1-2 days for all phases (Experiment 2 is higher complexity than 1)

---

## References

**Experiment Plan**: `/workspace/experiments/experiment_plan.md` (lines 95-154)

**Key Quote**:
> "Experiment 2: AR(1) Log-Normal with Regime-Switching
> **Priority**: 1 (MUST attempt)
> **Why This Model**: Addresses autocorrelation with AR(1) structure, addresses regime heterogeneity, leverages log-scale success"

**Falsification Criteria for Experiment 2**:
- Residual ACF lag-1 > 0.3 after fitting
- All sigma_regime posteriors overlap >80% (no regime effect)
- phi posterior centered near 0 (no autocorrelation benefit)
- Worse LOO than Experiment 1

If Experiment 2 meets any of these, it will also be rejected.

---

## Conclusion

**No improvement priorities needed** because the model class is fundamentally inadequate.

**Next action**: Proceed to Experiment 2 as planned in the experiment workflow.

This is not failure - this is the scientific process working correctly. We tested a hypothesis (independence adequate?), obtained a clear answer (no), and now move to test a more appropriate hypothesis (AR structure adequate?).

---

**Prepared by**: Model Criticism Specialist
**Date**: 2025-10-30
**Status**: Ready to initiate Experiment 2
