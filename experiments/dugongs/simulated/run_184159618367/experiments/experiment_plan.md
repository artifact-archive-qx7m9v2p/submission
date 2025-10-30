# Bayesian Model Experiment Plan
## Relationship between Y and x

**Date**: 2025-10-27
**Data**: 27 observations, x ∈ [1.0, 31.5], Y ∈ [1.71, 2.63]
**Pattern**: Nonlinear saturation (rapid increase at low x, plateau at high x)

---

## Executive Summary

Three independent designers proposed 9 distinct models. After removing duplicates and prioritizing by theoretical justification and EDA evidence, this plan specifies **5 prioritized model classes** spanning different mechanistic hypotheses.

**Key Strategic Decision**: We will test multiple competing hypotheses about how saturation occurs:
1. **Smooth exponential** approach to asymptote
2. **Sharp threshold** regime shift
3. **Power law** diminishing returns
4. **Polynomial** flexibility
5. **Robust** variants for outlier protection

---

## Model Priority Rankings

### Tier 1: Must Attempt (Top Priority)

#### Experiment 1: Asymptotic Exponential Model
**Source**: Designer 1 (Model 1)
**Hypothesis**: Smooth saturation via exponential approach to asymptote
**Functional Form**: `Y = α - β * exp(-γ * x)`
**Priority Rationale**:
- Theoretically grounded (enzyme kinetics, learning curves)
- Most interpretable parameters (asymptote, rate, amplitude)
- EDA shows smooth saturation pattern
- Expected R² ≈ 0.88-0.89

**Falsification Criteria**:
- Abandon if R² < 0.80
- Abandon if γ posterior includes zero (no saturation)
- Abandon if convergence fails (R-hat > 1.01 after tuning)

**Implementation**: Stan (CmdStanPy)

#### Experiment 2: Piecewise Linear (Change-Point)
**Source**: Designer 2 (Model 1), also proposed by Designer 1 (Model 2)
**Hypothesis**: Sharp regime shift at breakpoint τ
**Functional Form**:
```
Y = β₀ + β₁*x              if x ≤ τ
Y = β₀ + β₁*τ + β₂*(x-τ)   if x > τ
```
**Priority Rationale**:
- Best empirical fit in EDA (R² = 0.904)
- Clear breakpoint evidence (correlation 0.94 → -0.03)
- Tests hypothesis of sharp vs smooth transition
- Expected R² ≈ 0.88-0.92

**Falsification Criteria**:
- Abandon if SD(τ) > 5 (breakpoint too uncertain)
- Abandon if β₁ ≈ β₂ (no regime difference)
- Abandon if R² < 0.85

**Implementation**: Stan with smooth approximation to avoid discontinuity

---

### Tier 2: High Priority

#### Experiment 3: Log-Log Power Law
**Source**: Designer 3 (Model 1), also mentioned by Designer 1 (Model 3)
**Hypothesis**: Power law with constant elasticity
**Functional Form**: `log(Y) = α + β*log(x)` (equivalent to `Y = exp(α) * x^β`)
**Priority Rationale**:
- Strongest linearization (r=0.92 on log-log scale)
- Transforms nonlinear problem to linear
- Simple, fast inference
- Expected R² ≈ 0.81

**Falsification Criteria**:
- Abandon if residuals on log-scale show curvature
- Abandon if back-transformation predictions poor
- Abandon if R² < 0.75

**Implementation**: Stan (linear model on log-transformed data)

#### Experiment 4: Quadratic Polynomial
**Source**: Designer 3 (Model 2), mentioned by all designers
**Hypothesis**: Parabolic curvature captures saturation
**Functional Form**: `Y = β₀ + β₁*x + β₂*x²`
**Priority Rationale**:
- Simple, widely understood
- Good balance of fit (R² ≈ 0.86) vs complexity
- Standard approach for comparison
- Fast inference (linear in parameters)

**Falsification Criteria**:
- Abandon if β₂ ≈ 0 (reduces to linear)
- Abandon if predictions unrealistic outside range
- Abandon if R² < 0.80

**Implementation**: Stan (fastest option)

---

### Tier 3: Consider if Time Permits

#### Experiment 5: Robust Quadratic (Student-t)
**Source**: Designer 3 (Model 2 variant)
**Hypothesis**: Same as quadratic but robust to outliers
**Functional Form**: `Y ~ Student-t(ν, β₀ + β₁*x + β₂*x², σ)`
**Priority Rationale**:
- Handles potential outlier at x=31.5
- Tests sensitivity to likelihood choice
- Provides insurance against deviations from normality

**Falsification Criteria**:
- Abandon if ν → ∞ (equivalent to Gaussian)
- Abandon if predictions identical to Gaussian quadratic

**Implementation**: Stan

---

### Tier 4: Alternative Approaches (If All Above Fail)

#### Backup Option 1: Hierarchical B-Spline
**Source**: Designer 2 (Model 2)
**Rationale**: Maximum flexibility while maintaining smoothness
**When to Use**: If both smooth and sharp models fail

#### Backup Option 2: Mixture-of-Experts
**Source**: Designer 2 (Model 3)
**Rationale**: Tests soft vs hard regime transition
**When to Use**: If changepoint has extreme uncertainty

---

## Unified Prior Specifications

Based on EDA synthesis, these priors apply across relevant models:

| Parameter | Domain Knowledge | Prior | Rationale |
|-----------|-----------------|-------|-----------|
| **Asymptote (α)** | Y plateaus at 2.5-2.6 | Normal(2.55, 0.1) | Observed plateau level |
| **Amplitude (β)** | Range from min to max ~0.9 | Normal(0.9, 0.2) | Back-extrapolation |
| **Rate (γ)** | Transition over ~10 units | Gamma(4, 20) → E[γ]=0.2 | EDA segmentation |
| **Breakpoint (τ)** | Visual break at x=9-10 | Normal(9.5, 2.0) | Correlation shift point |
| **Residual (σ)** | Pure error ~0.075-0.12 | Half-Cauchy(0, 0.15) | Replicates |
| **Poly coef (β₁)** | Positive initial slope | Normal(0, 1) | Weakly informative |
| **Poly coef (β₂)** | Negative curvature | Normal(0, 0.5) | Weakly informative |
| **Log-scale α** | log(Y) center ~0.6 | Normal(0.6, 0.3) | Log-transformed data |
| **Power (β)** | Exponent ~0.12 | Normal(0.12, 0.1) | OLS estimate |
| **DoF (ν)** | Moderate robustness | Gamma(2, 0.1) | Weakly favor ν≈20 |

---

## Validation Pipeline (All Models)

### Stage 1: Prior Predictive Checks
**Goals**:
- Y samples stay in [1.0, 3.5] (plausible range)
- Curves exhibit saturation (not monotone unbounded)
- At least 80% of prior samples look reasonable

**Action if Fails**: Revise priors, rerun

### Stage 2: Simulation-Based Calibration
**Goals**:
- Generate synthetic data from known parameters
- Recover parameters within 95% credible intervals
- Verify >95% coverage across 20+ simulations

**Action if Fails**: Check model implementation, adjust priors

### Stage 3: MCMC Diagnostics
**Goals**:
- R-hat < 1.01 for all parameters
- ESS > 400 for all parameters
- No divergent transitions
- Visual trace plots show good mixing

**Action if Fails**:
1st attempt: Increase adapt_delta, increase iterations
2nd attempt: Reparameterize or switch to PyMC
If still fails: Document and move to next model

### Stage 4: Posterior Predictive Checks
**Goals**:
- 90-95% of observed Y in 95% posterior predictive intervals
- Posterior predictive R² > 0.85
- No systematic patterns in residuals
- Good fit at replicated x-values

**Action if Fails**: Document issues, proceed to critique phase

### Stage 5: LOO Cross-Validation
**Goals**:
- All Pareto-k < 0.7 (no highly influential points)
- LOO-R² > 0.75
- Compute ELPD for model comparison

**Action if Fails**: Investigate influential points, consider robust likelihood

---

## Model Comparison Strategy

### Step 1: Individual Assessment
For each model, determine: **ACCEPT**, **REVISE**, or **REJECT**

**ACCEPT** if:
- ✓ Passes all 5 validation stages
- ✓ R² > 0.85
- ✓ Interpretable parameters
- ✓ No major violations

**REVISE** if:
- ~ Convergence issues but fixable
- ~ Moderate fit (0.80 < R² < 0.85) but improvable
- ~ Prior-data conflict resolvable
- ~ Clear path to improvement

**REJECT** if:
- ✗ R² < 0.75
- ✗ Fundamental misspecification evident
- ✗ Cannot converge after multiple attempts
- ✗ Fails falsification criteria

### Step 2: Model Comparison (if 2+ ACCEPT)
**Method**: LOO-CV via `az.compare()`

**Decision Rules**:
- If ΔELPD > 2×SE: Prefer higher ELPD model
- If ΔELPD < 2×SE: Models tied, prefer:
  1. Simpler model (fewer parameters)
  2. More interpretable model
  3. Theoretically grounded model

### Step 3: Adequacy Assessment
**Adequate** if:
- At least 1 ACCEPT model
- Best model R² > 0.85
- Posterior predictions reasonable
- Diminishing returns from further refinement

**Continue** if:
- No ACCEPT models yet
- Clear improvement path exists
- Haven't exhausted all Tier 1-2 models

**Stop** if:
- Multiple REJECT with no progress
- Data quality issues discovered
- Computational limits reached

---

## Minimum Attempt Policy

Per workflow guidelines:
- **Must attempt at least Experiments 1 and 2** (Asymptotic and Piecewise)
- If Experiment 1 fails pre-fit validation, still attempt Experiment 2
- Document reason in log.md if fewer than 2 models attempted
- After 2 attempts, may proceed to assessment if at least 1 ACCEPT

---

## Implementation Strategy

### Experiment Order
1. **Experiment 3 first** (Log-Log): Fastest, simplest, strong EDA evidence
2. **Experiment 1 next** (Asymptotic): Most interpretable, theoretically motivated
3. **Experiment 2 next** (Piecewise): Best empirical fit, tests sharp transition
4. **Experiment 4** if needed (Quadratic): Standard comparison baseline
5. **Experiment 5** if time permits (Robust): Outlier protection

**Rationale**: Start with simplest/fastest (log-log), then move to more complex

### Computational Setup
- **Primary PPL**: CmdStanPy (Stan)
- **Fallback PPL**: PyMC (if Stan convergence issues)
- **Sampling**: 4 chains, 2000 iterations (1000 warmup)
- **Diagnostics**: ArviZ for all posterior analysis
- **LOO**: Must save log_likelihood in generated quantities

### Documentation Requirements
Each experiment directory must contain:
- `metadata.md`: Model specification and rationale
- `prior_predictive_check/findings.md`: Prior validation results
- `simulation_based_validation/recovery_metrics.md`: Calibration results
- `posterior_inference/inference_summary.md`: MCMC diagnostics and estimates
- `posterior_inference/diagnostics/posterior_inference.netcdf`: ArviZ InferenceData with log_lik
- `posterior_predictive_check/ppc_findings.md`: Model adequacy assessment
- `model_critique/decision.md`: ACCEPT/REVISE/REJECT with justification

---

## Falsification Framework

### Global Red Flags (Abandon All Approaches)
If encountered, pivot to fundamentally different strategy:
- All 5 models converge but all R² < 0.70
- Systematic residual patterns across all models
- Data quality issues discovered (e.g., measurement error dominates signal)
- Saturation pattern is artifact of outliers

**Pivot Options**:
- Gaussian Process (non-parametric)
- Mixture models (multiple populations)
- Collect more data
- Consult domain expert on data generation process

### Model-Specific Falsification
See individual experiment sections above.

---

## Expected Outcomes

### Most Likely Scenario (70% confidence)
- **Winner**: Asymptotic exponential (Exp 1) or Piecewise (Exp 2)
- **Performance**: R² ≈ 0.88-0.92
- **ΔELPD**: < 2×SE (models tied)
- **Recommendation**: Choose asymptotic for smooth interpretation, piecewise if mechanistic breakpoint is real

### Alternative Scenario (20% confidence)
- **Winner**: Log-log power law (Exp 3)
- **Performance**: R² ≈ 0.81 (lower but adequate)
- **Advantage**: Simplicity, fast inference, theoretical grounding
- **Recommendation**: Accept if ΔELPD not significantly worse

### Failure Scenario (10% confidence)
- **Issue**: All models fail to converge or achieve R² > 0.80
- **Diagnosis**: Saturation pattern more complex than assumed
- **Action**: Pivot to hierarchical B-spline or Gaussian Process

---

## Success Criteria

This modeling effort will be deemed **successful** if:
1. ✓ At least 1 model achieves ACCEPT status
2. ✓ Best model R² > 0.85
3. ✓ Posterior predictive checks pass (>90% coverage)
4. ✓ Parameters interpretable and reasonable
5. ✓ Uncertainty quantified appropriately
6. ✓ Model comparison decisive or principled tie-breaking applied
7. ✓ Validation pipeline completed rigorously

**Note**: Success = discovering true model class, even if that means rejecting initial hypotheses.

---

## Next Steps

1. **Set up experiment directories** (experiment_1/ through experiment_5/)
2. **Start with Experiment 3** (log-log, fastest path to results)
3. **Follow validation pipeline rigorously** for each model
4. **Update log.md** after each phase
5. **Proceed to model assessment** once adequate model(s) identified

---

## Summary Table

| Exp | Model | Source | Priority | Expected R² | Speed | Interpretability | Status |
|-----|-------|--------|----------|-------------|-------|------------------|--------|
| 1 | Asymptotic Exponential | D1-M1 | **Tier 1** | 0.88 | Medium | ⭐⭐⭐⭐⭐ | Pending |
| 2 | Piecewise Linear | D2-M1 | **Tier 1** | 0.90 | Medium | ⭐⭐⭐⭐ | Pending |
| 3 | Log-Log Power Law | D3-M1 | **Tier 2** | 0.81 | Fast | ⭐⭐⭐⭐ | Pending |
| 4 | Quadratic Polynomial | D3-M2 | **Tier 2** | 0.86 | Fast | ⭐⭐⭐ | Pending |
| 5 | Robust Quadratic | D3-M2v | Tier 3 | 0.86 | Medium | ⭐⭐⭐ | Pending |

**Legend**: D1=Designer 1, D2=Designer 2, D3=Designer 3, M1=Model 1, etc.

---

## References

- **EDA Report**: `/workspace/eda/eda_report.md`
- **Designer 1**: `/workspace/experiments/designer_1/proposed_models.md`
- **Designer 2**: `/workspace/experiments/designer_2/proposed_models.md`
- **Designer 3**: `/workspace/experiments/designer_3/proposed_models.md`
- **Synthesis**: `/workspace/eda/synthesis.md`
