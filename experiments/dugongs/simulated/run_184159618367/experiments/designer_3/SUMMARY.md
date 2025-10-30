# Designer 3 Summary: Alternative Approaches

## Core Strategy
Focus on **transformations, robustness, and simplicity** - use data transformations to achieve linearity when possible, build in robustness through likelihood choice, and prioritize interpretable models.

## Three Proposed Models

### 1. Log-Log Power Law (PRIMARY RECOMMENDATION)
**Specification**: `log(Y) ~ Normal(α + β*log(x), σ)`

**Key Insight**: EDA shows log-log transformation achieves r=0.92 (strongest linearity observed)

**Strengths**:
- Transforms nonlinear problem to linear regression
- Theoretically grounded (power laws ubiquitous)
- Fast inference, simple interpretation
- Expected R² ≈ 0.81

**Abandon if**: Residuals show curvature on log-scale, back-transformation systematically biased

---

### 2. Robust Quadratic with Student-t (PRAGMATIC FALLBACK)
**Specification**: `Y ~ Student-t(ν, β₀ + β₁*x + β₂*x², σ)`

**Key Insight**: Quadratic is simplest nonlinear form, Student-t provides outlier robustness

**Strengths**:
- Widely understood, interpretable
- Robust to outliers (e.g., x=31.5)
- Good balance of simplicity and flexibility
- Expected R² ≈ 0.85

**Abandon if**: Needs cubic terms, ν posterior at extreme values, predictions non-monotonic

---

### 3. Log Heteroscedastic (ASSUMPTION TESTER)
**Specification**: `Y ~ Normal(β₀ + β₁*log(x), σ₀*exp(α*x))`

**Key Insight**: Explicitly tests whether variance increases with x

**Strengths**:
- Simple mean function (2 parameters)
- Tests heteroscedasticity explicitly
- Can simplify to homoscedastic if α ≈ 0
- Expected R² ≈ 0.82

**Abandon if**: Underfits badly, α far from 0 but model still fails, variance explodes

---

## Validation Strategy

### Individual Model Checks
1. Prior predictive: Plausible data generation
2. MCMC diagnostics: R-hat < 1.01, ESS > 400
3. Posterior predictive: 90%+ coverage
4. LOO-CV: All Pareto-k < 0.7

### Model Comparison
- Compare LOO ELPD ± SE
- If ΔLOO > 2*SE: Prefer higher ELPD
- If ΔLOO < 2*SE: Prefer simpler model

### Stress Tests
- Extrapolation to x = [0.5, 50]
- Leave-out-regime validation
- Replicate test for 6 repeated x-values

---

## Decision Framework

### Success Thresholds
- **Minimum**: R² > 0.75, R-hat < 1.01, no systematic residuals
- **Good**: R² > 0.80, ESS > 800, 93%+ coverage
- **Excellent**: R² > 0.85, ESS > 2000, all Pareto-k < 0.3

### Checkpoint Decisions

**After individual fits**:
- All fail → Pivot to Designer 1/2 models
- 1-2 pass → Continue with passing models
- All pass → Proceed to comparison

**After LOO comparison**:
- Clear winner (ΔLOO > 4) → Report winner
- Equivalent (ΔLOO < 2*SE) → Report simplest

**After stress tests**:
- Extrapolation fails → Warn about uncertainty
- Replicate test fails → Revisit error model

---

## Expected Outcome

### Most Likely Ranking
1. **Log-log** (R² ≈ 0.81) - transformation achieves linearity
2. **Robust quadratic** (R² ≈ 0.85) - good fit, robust
3. **Log heteroscedastic** (R² ≈ 0.82) - if variance matters

### Reasoning
EDA's r=0.92 on log-log scale is strong evidence for fundamental power law structure. This is the strategic bet.

---

## Falsification: When to Abandon ALL Models

**Global failure criteria**:
- Best R² < 0.70 (clearly inadequate)
- All show systematic residual patterns
- Replicate test fails for all
- Convergence impossible despite tuning
- Extrapolation wildly implausible

**Escape routes**:
- Designer 1: Asymptotic exponential models
- Designer 2: Piecewise/hierarchical models
- Gaussian Process: Maximum flexibility
- Mixture models: If heterogeneity suspected

---

## Key Design Principles

1. **Transformation first**: If data becomes linear, use simple models
2. **Build in robustness**: Student-t protects against outliers
3. **Test assumptions**: Explicitly model heteroscedasticity
4. **Clear pivots**: Know when to switch model classes
5. **Falsification mindset**: Each model has explicit failure criteria

---

## Implementation Priority

### If unlimited time:
Fit all 3 models → Compare via LOO → Report best

### If time constrained:
1. Fit log-log (highest success probability)
2. If R² > 0.80 and diagnostics good → DONE
3. If not → Fit robust quadratic
4. If still not → Pivot to Designers 1/2

---

## Files

- **Full specification**: `/workspace/experiments/designer_3/proposed_models.md`
- **Summary**: `/workspace/experiments/designer_3/SUMMARY.md`

---

**Designer Focus**: Alternative approaches (transformations, robust methods, simple models)
**Models**: 3 (Log-Log, Robust Quadratic, Log Heteroscedastic)
**Primary Bet**: Log-log power law based on EDA r=0.92
**Pragmatic Fallback**: Robust quadratic Student-t
**Ready for**: Implementation
