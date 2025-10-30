# Unified Bayesian Model Experiment Plan
**Date**: 2025-10-28
**Dataset**: 8 observations with known measurement error
**Synthesis**: Combined recommendations from 3 independent model designers

---

## Overview

Three independent model designers proposed a total of 9 model variants. After removing duplicates and consolidating similar approaches, we have **4 distinct model classes** to implement, prioritized by theoretical justification and computational feasibility.

---

## Model Inventory (Consolidated from 3 Designers)

### Convergent Recommendations (All Designers)
All three designers independently recommended:
1. **Complete Pooling** - Single shared mean (baseline)
2. **Hierarchical/Partial Pooling** - Group-specific means with shrinkage

### Divergent Recommendations (Designer-Specific)
3. **Robust Models** (Designer 1) - t-distribution for outliers
4. **Measurement Error Misspecification** (Designers 2 & 3) - Challenge sigma assumptions

---

## Prioritized Model List

Based on theoretical strength, EDA support, and computational feasibility:

### **Model 1: Complete Pooling** [HIGH PRIORITY - BASELINE]
**Proposed by**: All 3 designers (convergent)
**EDA Support**: STRONG (p=0.42, tau²=0, SNR≈1)

**Mathematical Specification**:
```
Likelihood:  y_i ~ Normal(mu, sigma_i)    [known sigma_i]
Prior:       mu ~ Normal(10, 20)          [weakly informative]
```

**Justification**:
- Chi-square homogeneity test: p=0.42 (groups statistically indistinguishable)
- Between-group variance = 0 (observed < expected from measurement error alone)
- Signal-to-noise ratio ≈ 1 (measurement error dominates)
- Most parsimonious model consistent with data

**Falsification Criteria**:
- REJECT if: LOO Pareto k > 0.7 for any observation
- REJECT if: Posterior predictive checks fail (observed variance outside 95% interval)
- REJECT if: Systematic residual patterns indicate misfit

**Expected Outcome**: ACCEPT
- Posterior: mu ≈ 10 ± 4
- All diagnostics pass
- Best LOO-CV score

**PPL Implementation**: PyMC (simple, single parameter)

---

### **Model 2: Hierarchical Partial Pooling** [HIGH PRIORITY - COMPARISON]
**Proposed by**: All 3 designers (convergent)
**EDA Support**: WEAK (tests for hidden structure despite tau²=0)

**Mathematical Specification**:
```
Likelihood:       y_i ~ Normal(theta_i, sigma_i)    [known sigma_i]
Group level:      theta_i ~ Normal(mu, tau)         [partial pooling]
Hyperpriors:      mu ~ Normal(10, 20)
                  tau ~ Half-Normal(0, 10)           [regularizing, per Designer 2]
```

**Non-centered Parameterization** (to avoid funnel):
```
theta_i = mu + tau * theta_raw_i
theta_raw_i ~ Normal(0, 1)
```

**Justification**:
- Tests whether EDA missed group structure due to low power (n=8)
- Allows data to inform degree of pooling via tau
- If tau → 0, reduces to complete pooling (Model 1)
- Standard approach for hierarchical data

**Falsification Criteria**:
- REJECT if: tau posterior has 95% CI entirely below 1.0 (effectively complete pooling)
- REJECT if: Divergences > 5% (even with non-centered parameterization)
- REJECT if: LOO-CV worse than Model 1 by |ΔELPD| > 2×SE
- REJECT if: Funnel geometry persists despite non-centered parameterization

**Expected Outcome**: REJECT → revert to Model 1
- Posterior: tau ≈ 0-2 (near zero)
- LOO-CV: worse than or equal to Model 1
- Rationale: Insufficient data to estimate heterogeneity with SNR≈1

**PPL Implementation**: PyMC (handles non-centered parameterization well)

---

### **Model 3: Measurement Error Inflation** [MEDIUM PRIORITY - ADVERSARIAL]
**Proposed by**: Designers 2 & 3 (convergent adversarial approach)
**EDA Support**: NONE (challenges fundamental assumption)

**Mathematical Specification**:
```
Likelihood:       y_i ~ Normal(theta_i, sigma_i * lambda)    [inflate errors by lambda]
Group level:      theta_i ~ Normal(mu, tau)
Hyperpriors:      mu ~ Normal(0, 30)                         [skeptical, per Designer 2]
                  tau ~ Half-Normal(0, 10)
                  lambda ~ Uniform(0.5, 3.0)                 [test error misspecification]
```

**Justification**:
- Tests whether reported sigma values are systematically wrong
- Laboratories often underestimate measurement uncertainty
- Could explain "tau²=0" if true errors are 2-3× larger
- Directly falsifiable via lambda posterior

**Falsification Criteria**:
- REJECT if: lambda posterior 95% CI ∈ [0.9, 1.1] (errors accurate)
- REJECT if: Identifiability issues (tau and lambda confounded)
- REJECT if: LOO-CV worse than Model 1

**Expected Outcome**: REJECT → revert to Model 1
- Posterior: lambda ≈ 1.0 (0.85, 1.15)
- Rationale: No evidence of systematic error misspecification

**PPL Implementation**: PyMC with non-centered parameterization

**NOTE**: This is an **adversarial model** designed to challenge the EDA. Rejection strengthens confidence in Model 1.

---

### **Model 4: Robust t-Distribution** [LOW PRIORITY - IF NEEDED]
**Proposed by**: Designer 1
**EDA Support**: WEAK (no outliers detected, but tests robustness)

**Mathematical Specification**:
```
Likelihood:  y_i ~ StudentT(nu, mu, sigma_i)    [heavy-tailed alternative]
Priors:      mu ~ Normal(10, 20)
             nu ~ Gamma(2, 0.1)                  [weakly informative on df]
```

**Justification**:
- Tests whether "normal likelihood" assumption is critical
- Provides robustness to potential outliers
- If nu > 30, effectively reduces to normal (Model 1)

**Falsification Criteria**:
- REJECT if: nu posterior > 30 (normal distribution adequate)
- REJECT if: LOO-CV not significantly better than Model 1

**Expected Outcome**: REJECT → revert to Model 1
- Posterior: nu > 30 (effectively normal)
- Rationale: EDA found no outliers (LOO analysis)

**Implementation**: DEFERRED unless Models 1-3 show evidence for outliers

---

## Implementation Strategy

### Phase 1: Baseline Model (Required)
**Model 1: Complete Pooling**
- Must implement first (establishes baseline)
- Expected time: 30-45 minutes
- Expected outcome: ACCEPT

### Phase 2: Primary Comparison (Required per Minimum Attempt Policy)
**Model 2: Hierarchical Partial Pooling**
- Must implement to test hierarchical structure
- Expected time: 45-60 minutes (non-centered parameterization)
- Expected outcome: REJECT (tau ≈ 0)

**Decision Point 1**: After Models 1 & 2
- If Model 1 ACCEPT and Model 2 REJECT with tau < 1 → **Minimum attempts satisfied**
- Can proceed to Phase 4 (Assessment) OR continue to Phase 3 (Adversarial)

### Phase 3: Adversarial Testing (Recommended)
**Model 3: Measurement Error Inflation**
- Tests fundamental assumption about sigma values
- Expected time: 45-60 minutes
- Expected outcome: REJECT (lambda ≈ 1)
- **Rationale for implementing**: If REJECT, strongly confirms Model 1; if ACCEPT, critical discovery

### Phase 4: Robustness (Optional)
**Model 4: Robust t-Distribution**
- Only if Models 1-3 show evidence for outliers
- Otherwise: SKIP

---

## Decision Tree

```
START
  ↓
Fit Model 1 (Complete Pooling)
  ↓
[Passes validation?]
  ├─ NO → Document failure, try Model 3 (skip Model 2)
  └─ YES → Continue
      ↓
    Fit Model 2 (Hierarchical)
      ↓
    [tau < 1 AND LOO worse than Model 1?]
      ├─ YES → Model 2 REJECT, Model 1 ACCEPT
      │        Minimum attempts satisfied ✓
      │        ↓
      │      [Continue to Model 3?]
      │        ├─ YES → Fit Model 3 (adversarial test)
      │        └─ NO → Proceed to Phase 4 (Assessment)
      │
      └─ NO → Model 2 competitive
               ↓
             Fit Model 3 (if tau > 1, test if due to error misspec)
               ↓
             Compare Models 1, 2, 3 via LOO-CV
```

---

## Falsification Summary Table

| Model | Primary Falsification | Secondary Falsification | Expected Result |
|-------|----------------------|-------------------------|-----------------|
| 1. Complete Pooling | LOO Pareto k > 0.7 | PPC fail | **ACCEPT** |
| 2. Hierarchical | tau < 1 AND ΔLOO < 0 | Divergences > 5% | **REJECT** |
| 3. Error Inflation | lambda ∈ [0.9, 1.1] | Identifiability issues | **REJECT** |
| 4. Robust t | nu > 30 | ΔLOO ≤ 0 | **SKIP/REJECT** |

---

## Success Criteria

### Minimum Requirements (Per Guidelines)
- ✓ Attempt at least Models 1 and 2 (unless Model 1 fails pre-fit validation)
- ✓ Document all validation steps (prior predictive, SBC, posterior inference)
- ✓ Save InferenceData with log_likelihood for LOO-CV
- ✓ Posterior predictive checks for all fitted models

### Ideal Outcome
- Model 1 ACCEPT: Confirmed as adequate by all diagnostics
- Model 2 REJECT: tau ≈ 0, reduces to Model 1
- Model 3 REJECT: lambda ≈ 1, confirms measurement errors accurate
- **Conclusion**: Complete pooling strongly supported by:
  - EDA (p=0.42, tau²=0)
  - Model comparison (LOO-CV)
  - Adversarial testing (Models 2 & 3 fail to find issues)

---

## Computational Notes

### PyMC Implementation Details
All models use PyMC (CmdStan unavailable in current environment).

**Standard Configuration**:
```python
with pm.Model() as model:
    # ... model specification ...
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.95,  # For Model 2 (hierarchical)
        return_inferencedata=True
    )
    # Compute log_likelihood in model
    pm.compute_log_likelihood(trace)
```

**Saving Results**:
```python
trace.to_netcdf("experiments/experiment_N/posterior_inference/diagnostics/posterior_inference.netcdf")
```

### Expected Computation Times
- Model 1: ~1-2 minutes (single parameter)
- Model 2: ~5-10 minutes (non-centered, 8 group parameters)
- Model 3: ~10-15 minutes (potential identifiability issues)

---

## Designer Agreement Analysis

### Complete Agreement (3/3 designers)
✓ Model 1 (Complete Pooling) - All three independently recommended
✓ Model 2 (Hierarchical) - All three independently recommended

### Partial Agreement (2/3 designers)
✓ Model 3 (Measurement Error) - Designers 2 & 3 (adversarial)

### Unique Proposals (1/3 designers)
- Model 4 (Robust t) - Designer 1 only
- Mixture models - Designer 3 only (not prioritized due to n=8)
- Functional heteroscedasticity - Designer 3 only (not prioritized)

**Interpretation**: The convergent recommendations (Models 1 & 2) have strongest theoretical support and should be prioritized.

---

## Prior Sensitivity

### Designer Disagreement on mu Prior

**Designer 1**: `mu ~ Normal(10, 20)` [data-informed from EDA]
**Designer 2**: `mu ~ Normal(0, 30)` [skeptical, tests if mu > 0 is data-driven]
**Designer 3**: `mu ~ Normal(0, 30)` [adversarial stance]

**Resolution**:
- Use `Normal(10, 20)` for Models 1 & 2 (baseline)
- Use `Normal(0, 30)` for Model 3 (adversarial)
- If time permits, sensitivity analysis comparing both priors

### Designer Agreement on tau Prior

**Designer 1**: `Half-Cauchy(0, 5)` [standard weakly informative]
**Designer 2**: `Half-Normal(0, 10)` [regularizing for small n]
**Designer 3**: `Half-Cauchy(0, 10)` [standard but wider]

**Resolution**: Use `Half-Normal(0, 10)` per Designer 2's rationale (prevents extreme values with n=8)

---

## Documentation Requirements

For each fitted model, create:
```
experiments/experiment_N/
├── metadata.md                    # Model specification
├── prior_predictive_check/
│   ├── code/
│   ├── plots/
│   └── findings.md
├── simulation_based_validation/   # SBC
│   ├── code/
│   ├── plots/
│   └── recovery_metrics.md
├── posterior_inference/
│   ├── code/
│   ├── diagnostics/
│   │   ├── posterior_inference.netcdf  # WITH log_likelihood
│   │   ├── trace_plots.png
│   │   └── convergence_diagnostics.csv
│   ├── plots/
│   └── inference_summary.md
├── posterior_predictive_check/
│   ├── code/
│   ├── plots/
│   └── ppc_findings.md
└── model_critique/
    ├── critique_summary.md
    ├── decision.md                # ACCEPT/REVISE/REJECT
    └── improvement_priorities.md
```

---

## Timeline Estimate

| Phase | Activity | Expected Time |
|-------|----------|---------------|
| 1 | Model 1: Prior predictive check | 15 min |
| 1 | Model 1: Simulation-based validation | 30 min |
| 1 | Model 1: Fit + diagnostics | 30 min |
| 1 | Model 1: Posterior predictive check | 20 min |
| 1 | Model 1: Critique | 20 min |
| 2 | Model 2: Full validation pipeline | 2 hrs |
| 3 | Model 3: Full validation pipeline | 2 hrs |
| 4 | Model Assessment & Comparison | 1 hr |
| 5 | Adequacy Assessment | 30 min |
| **Total** | **Minimum (Models 1 & 2)** | **~4-5 hrs** |
| **Total** | **Full (Models 1-3)** | **~6-7 hrs** |

---

## Next Steps

1. Update `log.md` with synthesis completion
2. Launch `prior-predictive-checker` for Model 1
3. Upon success, proceed through validation pipeline
4. Continue per decision tree above

---

**End of Experiment Plan**
