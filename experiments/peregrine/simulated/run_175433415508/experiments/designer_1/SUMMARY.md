# Designer 1 Summary: Baseline Bayesian Count Models

## Quick Reference

**Designer**: Model Designer 1 (Baseline GLM Focus)
**Date**: 2025-10-29
**Status**: COMPLETE - Models specified and ready for implementation

---

## Three Proposed Models

### Model 1: Negative Binomial (Linear) - RECOMMENDED BASELINE
- **Type**: NegBin GLM with exponential growth
- **Complexity**: Simplest (3 parameters)
- **Justification**: Strong log-linear fit (R² = 0.937), severe overdispersion (Var/Mean = 70)
- **Runtime**: ~30-60 seconds
- **When to use**: Default starting point

### Model 2: Negative Binomial (Quadratic) - TEST AGAINST MODEL 1
- **Type**: NegBin GLM with polynomial trend
- **Complexity**: Medium (4 parameters)
- **Justification**: Quadratic R² = 0.964 > linear, growth rate changes over time
- **Runtime**: ~45-90 seconds
- **When to use**: If Model 1 shows quadratic residual pattern

### Model 3: Gamma-Poisson - COMPUTATIONAL CHECK
- **Type**: Hierarchical Poisson with random effects
- **Complexity**: High (N+3 parameters)
- **Justification**: Equivalent to Model 1, alternative parameterization
- **Runtime**: ~2-5 minutes
- **When to use**: Only if Model 1 has convergence issues

---

## Key Design Decisions

### What These Models DO
1. Handle severe overdispersion (primary EDA finding)
2. Model exponential/accelerating growth
3. Provide clean baseline for comparison with temporal models
4. Test polynomial vs exponential functional forms

### What These Models DON'T DO (Intentionally)
1. Model temporal autocorrelation (ACF = 0.971)
   - **Reason**: Isolate overdispersion from correlation
   - **Action**: Pass residual correlation to Designer 2
2. Allow structural breaks
   - **Reason**: Start with smooth trends
   - **Action**: Designer 3 can add changepoints if needed
3. Time-varying dispersion
   - **Reason**: Simplest baseline assumption
   - **Action**: Add complexity only if PPCs fail

---

## Critical Falsification Criteria

### Reject ALL Models If:
- LOO worse than naive mean prediction
- Posterior for phi >100 (contradicts overdispersion finding)
- >50% observations have Pareto k >0.7
- PPCs show systematic misfit (Bayesian p <0.01)

### Choose Model 1 If:
- Model 2 beta_2 credible interval includes zero
- ΔLOO(Model 2 - Model 1) <2
- Model 2 has high Pareto k (>0.7 for >20% obs)

### Choose Model 2 If:
- ΔLOO(Model 2 - Model 1) >4
- Model 1 residuals show clear quadratic pattern
- beta_2 posterior clearly excludes zero

### Pass to Designer 2 If:
- Residual ACF >0.8 (high temporal correlation remains)
- Heteroscedastic residuals over time
- Good fit but autocorrelated errors

---

## Files Created

### Model Specifications
- `/workspace/experiments/designer_1/proposed_models.md` - Full specifications (28 pages)
- `/workspace/experiments/designer_1/models/model_1_negbin_linear.stan`
- `/workspace/experiments/designer_1/models/model_2_negbin_quadratic.stan`
- `/workspace/experiments/designer_1/models/model_3_gamma_poisson.stan`

### To Be Implemented (Next Phase)
- `scripts/prior_predictive_check.py` - Validate priors
- `scripts/fit_models.py` - Run MCMC inference
- `scripts/diagnostics.py` - Check convergence and fit
- `scripts/compare_models.py` - LOO comparison

---

## Expected Outcomes

### Success Scenario (Likely)
- Model 1 or 2 converges quickly (R-hat <1.01)
- Good predictive performance (LOO competitive)
- Overdispersion handled (phi ≈ 10-30)
- Residual ACF still high (0.6-0.8) → expected, pass to Designer 2

### Partial Success (Possible)
- Models fit well but residuals heteroscedastic
- Model 2 only marginally better than Model 1
- Some influential observations (Pareto k 0.5-0.7)
- Action: Report both models, note limitations

### Failure Scenario (Unlikely given EDA)
- All models fail to converge
- PPCs show systematic misfit
- Extreme parameter posteriors
- Action: Revisit likelihood family, consult Designer 3

---

## Key Parameters and Priors

| Parameter | Prior | Interpretation | Expected Posterior |
|-----------|-------|----------------|-------------------|
| beta_0 | Normal(4.69, 1.0) | Log count at year=0 | Near prior (4.5-4.9) |
| beta_1 | Normal(1.0, 0.5) | Growth rate | 0.7-1.2 (strong growth) |
| beta_2 | Normal(0, 0.5) | Acceleration | -0.5 to +0.5 (if needed) |
| phi | Gamma(2, 0.1) | Overdispersion | 10-40 (moderate) |

---

## Connection to EDA Findings

| EDA Finding | How Models Address | Limitation |
|-------------|-------------------|------------|
| Overdispersion (Var/Mean = 70) | NegBin distribution | Assumes constant phi |
| Exponential growth (R² = 0.937) | Log link, beta_1 | Linear on log scale |
| Quadratic trend (R² = 0.964) | Model 2 tests this | May be artifact |
| Autocorrelation (ACF = 0.971) | **NOT ADDRESSED** | Designer 2 job |
| Heteroscedastic variance | **NOT ADDRESSED** | Designer 2 job |
| Possible changepoint | **NOT ADDRESSED** | Designer 3 job |

---

## Computational Requirements

**Software**:
- CmdStanPy >= 1.2.0
- ArviZ >= 0.17.0
- NumPy, Pandas, SciPy, Matplotlib

**Hardware** (N=40):
- CPU: Any modern processor (4 cores recommended)
- RAM: 1 GB sufficient
- Storage: <100 MB for all results
- Time: <10 minutes total for all 3 models

**Scaling**:
- Model 1, 2: Scale to N=10,000 easily
- Model 3: Struggles beyond N=1,000 (too many parameters)

---

## Implementation Checklist

- [x] Model specifications written
- [x] Stan files created
- [x] Priors justified
- [x] Falsification criteria defined
- [ ] Prior predictive checks implemented
- [ ] Models fitted to data
- [ ] Convergence diagnostics run
- [ ] LOO comparison computed
- [ ] Posterior predictive checks performed
- [ ] Results documented

---

## Decision Tree

```
Start: Fit all 3 models
│
├─ Convergence issues?
│  ├─ Yes → Increase adapt_delta, try Model 3 parameterization
│  └─ No → Continue
│
├─ Compute LOO for converged models
│  ├─ Model 1 vs Model 2: |ΔLOO| <2?
│  │  ├─ Yes → Choose Model 1 (simpler)
│  │  └─ No → Choose better model
│  │
│  └─ Model 3 vs Model 1: Similar posterior?
│     ├─ Yes → Use Model 1 (faster)
│     └─ No → Investigate discrepancy
│
├─ Posterior predictive checks
│  ├─ Good fit (p ∈ [0.05, 0.95])?
│  │  ├─ Yes → SUCCESS, report best model
│  │  └─ No → Diagnose failure mode
│  │
│  └─ Residual ACF >0.8?
│     ├─ Yes → Pass to Designer 2 (expected)
│     └─ No → Baseline sufficient (unlikely)
│
└─ Final Report
   ├─ Recommended model: 1 or 2
   ├─ Overdispersion estimate: phi
   ├─ Growth rate: beta_1 (and beta_2)
   └─ Limitations: temporal correlation, etc.
```

---

## Critical Insights

### Why Baseline Models Matter
Even though we EXPECT these models to fail at capturing temporal correlation (ACF = 0.971), they serve crucial purposes:

1. **Isolate variance components**: Separate overdispersion from autocorrelation
2. **Establish lower bound**: Worst-case performance without correlation
3. **Diagnostic baseline**: What patterns remain after accounting for mean trend?
4. **Computational reference**: How much does complexity cost?

### The "Failure" is the Success
If these models:
- Fit the mean trend well (good mu estimates)
- Handle overdispersion (phi ≈ 20)
- But show residual ACF >0.8

Then we've SUCCEEDED by:
- Confirming NegBin family appropriate
- Quantifying growth parameters
- Isolating the correlation problem for Designer 2
- Providing clean comparison target

**Don't interpret high residual ACF as model failure** - it's expected and informative!

---

## Contact Information

**Questions for Designer 1**:
- "Is NegBin the right family?" → Check phi posterior and PPCs
- "Linear or quadratic?" → LOO comparison, beta_2 credible interval
- "Why ignore autocorrelation?" → Intentional, isolate components

**Questions for Other Designers**:
- "How to add temporal correlation?" → Designer 2
- "What about changepoints?" → Designer 3
- "Non-parametric smoothing?" → Designer 3

---

## References

- **Full specification**: `/workspace/experiments/designer_1/proposed_models.md`
- **EDA report**: `/workspace/eda/eda_report.md`
- **Stan files**: `/workspace/experiments/designer_1/models/`

---

**Status**: Ready for implementation
**Next step**: Run prior predictive checks
**Estimated time to results**: 3-4 hours
