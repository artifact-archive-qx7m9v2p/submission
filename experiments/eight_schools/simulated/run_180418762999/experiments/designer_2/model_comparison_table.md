# Quick Model Comparison Table

## Three Competing Models at a Glance

| Aspect | Model A: Complete Pooling | Model B: Hierarchical | Model C: Inflated Errors |
|--------|--------------------------|----------------------|-------------------------|
| **Core Hypothesis** | Groups are identical (tau = 0) | Groups differ modestly (tau ~ 3-8) | Sigma values underestimated |
| **Likelihood** | `y_i ~ N(mu, sigma_i)` | `y_i ~ N(theta_i, sigma_i)` | `y_i ~ N(mu, sqrt(sigma_i^2 + tau_meas^2))` |
| **Key Parameters** | mu (1 param) | mu, tau, theta_1:8 (10 params) | mu, tau_meas (2 params) |
| **Prior on mu** | N(0, 30) | N(0, 30) | N(0, 30) |
| **Prior on variance** | - | tau ~ Half-Normal(0, 10) | tau_meas ~ Half-Normal(0, 5) |
| **What it tests** | EDA conclusion | Hidden heterogeneity | Measurement error misspec |
| **Expected if correct** | Narrow CI for mu, good fit | tau > 2, better LOO-CV | tau_meas > 5, better LOO-CV |
| **Expected if wrong** | High Pareto k, poor predictive fit | tau < 1, divergences | tau_meas < 2, unidentifiable |
| **Computational difficulty** | Easy (1 min) | Hard (boundary, funnel) | Moderate (correlation) |
| **Abandon if** | LOO k > 0.7 | Divergences > 5% OR tau < 1 | tau_meas < 2 OR corr > 0.7 |

---

## Decision Tree

```
Start: Fit Model A
    |
    v
    Pass diagnostics?
    |
    +--> NO ---> Red flag! Investigate data quality
    |
    +--> YES --> Fit Model B
                    |
                    v
                    LOO-CV better by > 2 SE AND tau > 2?
                    |
                    +--> YES --> Use Model B (heterogeneity detected)
                    |
                    +--> NO --> Is Model A LOO-CV best or tied?
                                |
                                +--> YES --> Use Model A (complete pooling)
                                |
                                +--> NO --> Fit Model C
                                            |
                                            v
                                            tau_meas > 5 AND LOO-CV better?
                                            |
                                            +--> YES --> Use Model C (errors underestimated)
                                            |
                                            +--> NO --> Use Model A (simplest adequate)
```

---

## Key Design Differences

### Prior on mu: N(0, 30) vs EDA's N(10, 20)

| Choice | Center | SD | Rationale |
|--------|--------|-----|-----------|
| **EDA** | 10 | 20 | Centers on observed weighted mean |
| **Mine** | 0 | 30 | More skeptical, tests if mu > 0 is data-driven |
| **Sensitivity test** | Compare posteriors | If differ by > 20%, prior matters | Diagnostic of data weakness |

### Prior on tau: Half-Normal(0, 10) vs Standard Half-Cauchy(0, 5)

| Choice | Distribution | Scale | Tail behavior |
|--------|-------------|-------|---------------|
| **Standard** | Half-Cauchy | 5 | Very heavy tails (allows tau >> SD) |
| **Mine** | Half-Normal | 10 | Lighter tails (weakly regularizing) |
| **Rationale** | With n=8, want mild regularization | | Prevents extreme tau values |

---

## Falsification Criteria Matrix

| Model | Computational Red Flag | Statistical Red Flag | Action if Triggered |
|-------|------------------------|---------------------|---------------------|
| **Model A** | ESS < 100 | Pareto k > 0.7 | Should never happen; investigate data |
| **Model A** | R-hat > 1.01 | Predictive variance check fails | Try Model B or C |
| **Model B** | Divergences > 5% | tau posterior < 1 | Use Model A (complete pooling) |
| **Model B** | R-hat > 1.05 | LOO-CV worse than A | Try non-centered, then abandon |
| **Model C** | Corr(mu, tau_meas) > 0.7 | tau_meas < 2 | Use Model A (reported errors OK) |
| **Model C** | ESS < 50 | Predictive var >> observed | Abandon (overfitting) |

---

## Expected Posterior Ranges

### If Model A is Correct (EDA Scenario)

| Parameter | Expected Posterior | 95% CI Width | Notes |
|-----------|-------------------|--------------|-------|
| mu | ~10 | ±8 | Centered on weighted mean |
| - | - | - | Simple, narrow CI |

### If Model B is Correct (Heterogeneity Scenario)

| Parameter | Expected Posterior | 95% CI Width | Notes |
|-----------|-------------------|--------------|-------|
| mu | ~10 | ±10 | Slightly wider than A |
| tau | 3-8 | ±5 | Moderate heterogeneity |
| theta_i | Group-specific | ±8-12 | Shrunk toward mu |

### If Model C is Correct (Error Misspec Scenario)

| Parameter | Expected Posterior | 95% CI Width | Notes |
|-----------|-------------------|--------------|-------|
| mu | ~10 | ±12 | Wider than A (more uncertainty) |
| tau_meas | 5-10 | ±5 | Additional measurement error |
| Total sigma | sqrt(sigma_i^2 + tau_meas^2) | - | Inflated uncertainties |

---

## Stress Tests Summary

| Test | What it Checks | Success Criterion | Failure Interpretation |
|------|---------------|-------------------|----------------------|
| **Leave-one-out (Model A)** | Stability to individual observations | Posterior shifts < 5 units | Influential obs, try Model B |
| **Prior sensitivity (tau)** | Data informativeness | Posteriors agree within 2x | Insufficient data for tau |
| **Predictive variance** | Model adequacy | Obs var in 50-95% PI | Over/underfitting |
| **Non-centered (Model B)** | Sampling geometry | Divergences < 2% | Funnel present, use non-centered |

---

## What Success Looks Like

### Best Case: Clear Winner

**Scenario**: Model B wins LOO-CV by 3 SE, tau ~ 6, no divergences
- **Report**: "Modest heterogeneity detected (tau ~ 6)"
- **Implication**: EDA missed structure due to low power
- **Deliver**: Group-specific estimates with shrinkage

### Good Case: EDA Confirmed

**Scenario**: Model A wins LOO-CV, Model B has tau < 1 and boundary issues
- **Report**: "Complete pooling confirmed, EDA validated"
- **Implication**: Groups truly homogeneous
- **Deliver**: Pooled estimate with narrow CI

### Honest Case: Inconclusive

**Scenario**: All models within 1 SE on LOO-CV, wide CIs, prior-sensitive
- **Report**: "Insufficient data to distinguish hypotheses"
- **Implication**: Need n > 20 for reliable inference
- **Deliver**: All three posteriors with uncertainty about choice

---

## Why This Approach is Different

### Standard Bayesian Workflow
1. Start with simplest model
2. Check diagnostics
3. Add complexity if needed
4. Report best model

**Risk**: Confirmation bias toward simple model

### My Workflow
1. Propose three competing hypotheses
2. Fit all in parallel
3. Use LOO-CV to arbitrate
4. Report model uncertainty if unclear

**Advantage**: Tests alternatives explicitly, avoids anchoring on first model

---

## Computational Roadmap

```
Phase 1 (30 min)
├── Fit Model A (2 min)
├── Check diagnostics (5 min)
├── Fit Model B (10 min)
└── Initial comparison (5 min)
    |
    v
Phase 2 (30 min)
├── Fit Model C (10 min)
├── Three-way LOO-CV (10 min)
└── Posterior predictive checks (10 min)
    |
    v
Phase 3 (30 min, if needed)
├── Stress tests (15 min)
├── Prior sensitivity (10 min)
└── Non-centered Model B (if divergences)
    |
    v
Phase 4 (30 min)
├── Generate plots (15 min)
├── Write reports (15 min)
└── Document code
```

---

## Critical Questions Each Model Answers

### Model A
**Question**: Can a single population mean explain all observations?
**Answer**: Yes if LOO-CV best and predictive checks pass

### Model B
**Question**: Is there detectable heterogeneity across groups?
**Answer**: Yes if tau > 2 and LOO-CV improvement > 2 SE

### Model C
**Question**: Are reported uncertainties adequate?
**Answer**: No if tau_meas > 5 and LOO-CV improvement

---

## Interpretation Guidelines

### If all models similar (within 1 SE)
- **Implication**: Data insufficient to distinguish
- **Report**: Model uncertainty
- **Recommendation**: Collect more data (n > 20)

### If Model A wins clearly
- **Implication**: EDA conclusion validated
- **Report**: Complete pooling, single mu estimate
- **Scientific story**: Groups truly homogeneous

### If Model B wins clearly
- **Implication**: Hidden heterogeneity exists
- **Report**: Group-specific estimates with shrinkage
- **Scientific story**: Groups differ despite low power in EDA

### If Model C wins clearly
- **Implication**: Measurement error misspecified
- **Report**: Inflated uncertainties, wider CIs
- **Scientific story**: Reported sigma values underestimate true uncertainty

---

## Final Checklist Before Reporting

- [ ] All chosen models: R-hat < 1.05
- [ ] All chosen models: ESS > 100
- [ ] Divergences < 2% (or non-centered used)
- [ ] Posterior predictive checks pass
- [ ] LOO-CV computed and interpreted with SEs
- [ ] Prior sensitivity < 30% for key parameters
- [ ] Trace plots reviewed
- [ ] Scientific interpretation makes sense
- [ ] Uncertainty honestly reported
- [ ] Code documented and reproducible

---

**Created**: 2025-10-28
**Purpose**: Quick reference for model comparison strategy
**Status**: Awaiting implementation
