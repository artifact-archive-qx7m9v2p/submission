# Model Designer 1: Experiment Outputs

**Designer:** Model Designer 1 - Bayesian Modeling Strategist
**Date:** 2025-10-28
**Dataset:** Hierarchical data with known measurement error (n=8)

---

## Quick Navigation

### Primary Documents

1. **`proposed_models.md`** (29 KB) - COMPREHENSIVE TECHNICAL SPECIFICATION
   - Full mathematical specifications for 3 model classes
   - Detailed falsification criteria for each model
   - Computational considerations (Stan implementations)
   - Stress tests and diagnostic strategies
   - Alternative approaches if all models fail
   - **Start here for technical details**

2. **`experiment_plan.md`** (11 KB) - ACTIONABLE ROADMAP
   - Implementation checklist
   - Decision rules and red flags
   - Timeline and deliverables
   - Success criteria
   - **Start here for implementation**

3. **`README.md`** (this file) - Quick reference

---

## Three Proposed Models

### Model 1: Complete Pooling (EDA-Supported)
```
y_i ~ Normal(mu, sigma_i)
mu ~ Normal(10, 20)
```
- **Hypothesis:** All groups share single true mean
- **EDA Support:** Strong (p=0.42, tau^2=0)
- **Expected:** ACCEPT - likely winner
- **Falsification:** Reject if posterior predictive checks fail OR Pareto k > 0.7

### Model 2: Hierarchical (Falsification Test)
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
mu ~ Normal(10, 20)
tau ~ Half-Cauchy(0, 5)
```
- **Hypothesis:** Groups have distinct means from common distribution
- **Purpose:** Challenge Model 1; test if EDA missed structure
- **Expected:** REJECT - tau will be ~0
- **Falsification:** Reject if tau < 1 (revert to Model 1)

### Model 3: Robust t-Distribution (Outlier Test)
```
y_i ~ Student_t(nu, mu, sigma_i)
mu ~ Normal(10, 20)
nu ~ Gamma(2, 0.1)
```
- **Hypothesis:** Group 4 (negative value) is outlier
- **Purpose:** Test robustness to potential contamination
- **Expected:** REJECT - nu will be >30 (data consistent with normal)
- **Falsification:** Reject if nu > 30 (revert to Model 1)

---

## Key Design Principles

1. **Falsification-First:** Each model has explicit criteria for rejection
2. **Challenge Assumptions:** Models 2-3 designed to falsify Model 1
3. **Computational Diagnostics:** Sampling issues indicate model problems
4. **Truth Over Tasks:** Success = finding right model, not fitting all three
5. **Ready to Pivot:** 5 alternative approaches if all models fail

---

## Decision Tree

```
Fit Model 1
    ├─ Posterior predictive checks PASS?
    │   ├─ YES + Pareto k < 0.7 → ACCEPT Model 1 ✓
    │   └─ NO → Try Model 2 or 3
    │
    ├─ If Model 2 fitted:
    │   ├─ tau < 1 → REVERT to Model 1 ✓
    │   └─ tau > 5 → ACCEPT Model 2
    │
    └─ If Model 3 fitted:
        ├─ nu > 30 → REVERT to Model 1 ✓
        └─ nu < 10 → ACCEPT Model 3
```

---

## Expected Outcome (Predictions)

Based on EDA findings:

- **Model 1:** Will pass all checks → ACCEPT
  - Posterior: mu ≈ 10 ± 4
  - Best LOO-CV score
  - No computational issues

- **Model 2:** Will estimate tau ≈ 0-2 → REJECT (revert to Model 1)
  - Computational challenges (funnel geometry)
  - Effectively reduces to Model 1

- **Model 3:** Will estimate nu > 30 → REJECT (revert to Model 1)
  - No improvement over Model 1
  - Data consistent with normal likelihood

**If these predictions are wrong, it means we learned something unexpected about the data.**

---

## Implementation Checklist

### Phase 1: Setup (2-4 hours)
- [ ] Write Stan code for Model 1
- [ ] Write Stan code for Model 2 (non-centered)
- [ ] Write Stan code for Model 3
- [ ] Test compilation

### Phase 2: Fitting (1-2 hours)
- [ ] Run Model 1: 4 chains, 2000 iterations
- [ ] Run Model 2: 4 chains, 2000 iterations
- [ ] Run Model 3: 4 chains, 2000 iterations

### Phase 3: Diagnostics (2-3 hours)
- [ ] Check convergence (R-hat, ESS, divergences)
- [ ] Posterior predictive checks
- [ ] LOO cross-validation
- [ ] Residual analysis

### Phase 4: Decisions (2-3 hours)
- [ ] Apply falsification criteria
- [ ] Document which models rejected/accepted
- [ ] Run sensitivity analyses
- [ ] Generate visualizations

### Phase 5: Reporting (2-3 hours)
- [ ] Write results summary
- [ ] Create comparison tables
- [ ] Explain decisions
- [ ] Provide recommendations

**Total Time:** 9-14 hours (comprehensive) or 5-7 hours (fast track)

---

## Red Flags (Major Issues)

### Immediate Pivot Required If:
1. All three models fail posterior predictive checks
2. Persistent computational issues across all models
3. Group 4 has Pareto k > 1.0 in all models
4. Posteriors dominated by priors (insufficient data)
5. Physically implausible parameter estimates

### Action:
- Investigate data quality
- Check measurement model assumptions
- Consider alternative model classes (see proposed_models.md)

---

## File Locations

**All outputs in:** `/workspace/experiments/designer_1/`

**Documents:**
- `proposed_models.md` - Technical specifications (29 KB)
- `experiment_plan.md` - Implementation plan (11 KB)
- `README.md` - This file

**To be created during implementation:**
- `model_1.stan` - Complete pooling Stan code
- `model_2.stan` - Hierarchical Stan code
- `model_3.stan` - Robust t-distribution Stan code
- `results_summary.md` - Final results and decisions
- `diagnostics/` - Convergence checks and plots
- `posteriors/` - Posterior samples (CSV/netCDF)

---

## Key Metrics

### Model Acceptance Criteria
- **R-hat:** < 1.01 for all parameters
- **ESS:** > 400 per chain (bulk and tail)
- **Divergences:** 0 (especially for Model 2)
- **Posterior predictive p:** 0.05 < p < 0.95
- **Pareto k:** < 0.7 for all observations

### Model Comparison
- **LOO-CV:** Delta-ELPD > 2*SE for meaningful difference
- **Parsimony:** Simpler model preferred if equivalent performance

---

## References

1. Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515-534.

2. Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

3. Gabry, J., et al. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A*, 182(2), 389-402.

4. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.

---

## Summary Statement

This experiment proposes three competing Bayesian model classes with explicit falsification criteria:

1. **Model 1 (Complete Pooling):** EDA-supported baseline
2. **Model 2 (Hierarchical):** Tests if group structure exists
3. **Model 3 (Robust):** Tests if outliers affect inference

**Strategy:** Models 2 and 3 are designed to CHALLENGE Model 1. If they fail (as expected based on EDA), it strengthens confidence in complete pooling. If they succeed unexpectedly, we've discovered hidden structure.

**Success is finding truth, not completing tasks.**

---

## Contact

For questions about:
- **Technical specifications:** See `proposed_models.md`
- **Implementation details:** See `experiment_plan.md`
- **Quick reference:** This file

**Next step:** Implement Stan models and begin Phase 1
