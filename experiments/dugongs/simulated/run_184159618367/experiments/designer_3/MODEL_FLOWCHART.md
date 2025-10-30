# Designer 3: Model Selection Flowchart

```
START: EDA Finding - Log-log transformation achieves r=0.92
│
├─→ MODEL 1: Log-Log Power Law (PRIMARY BET)
│   │
│   ├─ Specification: log(Y) ~ Normal(α + β*log(x), σ)
│   ├─ Expected R²: 0.81
│   ├─ Strategy: Transform to linearity
│   │
│   ├─→ [SUCCESS if R² > 0.80 ∧ No residual curvature]
│   │   └─→ REPORT as winner
│   │
│   └─→ [FAIL if Log-scale residuals show curvature]
│       └─→ Continue to Model 2
│
├─→ MODEL 2: Robust Quadratic (PRAGMATIC FALLBACK)
│   │
│   ├─ Specification: Y ~ Student-t(ν, β₀+β₁x+β₂x², σ)
│   ├─ Expected R²: 0.85
│   ├─ Strategy: Simple nonlinear + robustness
│   │
│   ├─→ [SUCCESS if R² > 0.83 ∧ ν reasonable]
│   │   └─→ REPORT as winner
│   │
│   └─→ [FAIL if Needs cubic or ν extreme]
│       └─→ Continue to Model 3
│
├─→ MODEL 3: Log Heteroscedastic (ASSUMPTION TESTER)
│   │
│   ├─ Specification: Y ~ Normal(β₀+β₁log(x), σ₀exp(αx))
│   ├─ Expected R²: 0.82
│   ├─ Strategy: Test variance structure
│   │
│   ├─→ [SUCCESS if R² > 0.80]
│   │   └─→ REPORT as winner
│   │
│   └─→ [FAIL if All 3 models R² < 0.75]
│       └─→ GLOBAL FAILURE
│
└─→ GLOBAL FAILURE: All simple models inadequate
    │
    ├─→ PIVOT 1: Designer 1 Models
    │   └─ Asymptotic exponential (R² ≈ 0.89)
    │
    ├─→ PIVOT 2: Designer 2 Models
    │   └─ Piecewise linear (R² ≈ 0.90)
    │
    └─→ PIVOT 3: Advanced Models
        ├─ Gaussian Process
        └─ Mixture models

═══════════════════════════════════════════════════

COMPARISON STAGE (if multiple models succeed):
│
├─ Compute LOO ELPD for all passing models
│
├─→ [ΔLOO > 2*SE] → Clear winner exists
│   └─→ REPORT highest ELPD model
│
└─→ [ΔLOO < 2*SE] → Models equivalent
    └─→ Prefer simpler model
        Ranking: Log-Log > Log-Het > Quadratic
                (3 params)  (4 params)  (5 params)

═══════════════════════════════════════════════════

VALIDATION CHECKLIST (for any model):
│
├─ [✓] R-hat < 1.01 for all parameters
├─ [✓] ESS > 400 for all parameters
├─ [✓] No divergences
├─ [✓] Posterior predictive coverage > 90%
├─ [✓] No systematic residual patterns
├─ [✓] All Pareto-k < 0.7
├─ [✓] Replicate test passes
└─ [✓] Extrapolation plausible

═══════════════════════════════════════════════════

KEY DECISION POINTS:

Checkpoint 1 (After individual fits):
  - All fail (R² < 0.75) → PIVOT to Designer 1/2
  - 1-2 pass → Continue with passing models
  - All pass → Proceed to comparison

Checkpoint 2 (After LOO comparison):
  - Clear winner (ΔLOO > 4) → Report winner
  - Equivalent (ΔLOO < 2*SE) → Report simplest
  - All poor (R² < 0.75) → PIVOT

Checkpoint 3 (After stress tests):
  - Extrapolation fails → Flag uncertainty limits
  - Replicate test fails → Revisit error model
  - All tests pass → FINALIZE

═══════════════════════════════════════════════════

EXPECTED OUTCOME:

Most Likely:
  Log-Log wins with R² ≈ 0.81
  Reason: EDA r=0.92 is strong evidence

Alternative:
  Robust Quadratic wins with R² ≈ 0.85
  Reason: Better captures plateau shape

Unlikely:
  All three fail
  Action: Pivot to asymptotic models

═══════════════════════════════════════════════════
```

## Quick Reference

### Model Parameters

| Model | # Params | Transformation | Robustness | Complexity |
|-------|----------|----------------|------------|------------|
| Log-Log | 3 | Log-Log | Multiplicative | Low |
| Quadratic | 5 | None | Student-t | Medium |
| Log-Het | 4 | Log-X only | Hetero-σ | Medium |

### Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| R² | > 0.75 | > 0.80 | > 0.85 |
| R-hat | < 1.01 | < 1.01 | < 1.001 |
| ESS | > 400 | > 800 | > 2000 |
| Coverage | > 90% | > 93% | > 95% |
| Pareto-k | < 0.7 | < 0.5 | < 0.3 |

### Falsification Triggers

**Model-Specific**:
- Log-Log: Curvature remains on log-scale
- Quadratic: Needs cubic terms or ν extreme
- Log-Het: α far from 0 but still fails

**Global**:
- All R² < 0.70
- Convergence impossible
- Systematic residuals persist
- Replicate test fails

### Time Budget

- Setup: 30 min
- Fitting: 1-2 hours (all 3 models)
- Diagnostics: 1 hour
- Comparison: 30 min
- Reporting: 1 hour

**Total**: 4-5 hours
