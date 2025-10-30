# Designer 2: Flexible & Complexity-Embracing Models

**Design Philosophy**: Capture complexity when EDA justifies it. Start with models that reflect genuine data patterns rather than defaulting to minimal parameterization.

---

## Quick Navigation

### 1. Main Proposal: `proposed_models.md`
**What**: Detailed specification of 3 Bayesian model classes
**Key sections**:
- Model 1: Quadratic + time-varying dispersion (primary)
- Model 2: Piecewise regime shift (alternative)
- Model 3: Hierarchical B-spline (maximum flexibility)
- Falsification criteria for each model
- Red flags that trigger major pivots
- Stress tests designed to break models

**Start here** for understanding the modeling strategy.

### 2. Stan Templates: `stan_model_templates.md`
**What**: Concrete Stan code for all proposed models
**Key sections**:
- Complete Stan implementations
- Python wrappers for CmdStanPy
- Diagnostic checklist
- Example usage code

**Use this** when implementing the models.

### 3. Design Philosophy: `design_philosophy.md`
**What**: Rationale for complexity-first approach
**Key sections**:
- Why quadratic term is justified
- Why time-varying dispersion is necessary
- Comparison to simpler approaches
- Success/failure criteria
- Meta-discussion on model complexity

**Read this** to understand the philosophical differences from Designer 1.

---

## Model Summary

### Model 1 (PRIMARY): Quadratic + Time-Varying Dispersion
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁×year[i] + β₂×year²[i]
log(φ[i]) = γ₀ + γ₁×year[i]
```
**Parameters**: 5 (β₀, β₁, β₂, γ₀, γ₁)
**Justification**: R² = 0.96 (vs 0.88 linear), 20x variance change
**Prediction**: Will outperform simpler models

### Model 2 (ALTERNATIVE): Piecewise Regime Shift
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁×year + β₂×I(year>-0.21) + β₃×(year+0.21)×I(year>-0.21)
log(φ[i]) = γ₀ + γ₁×I(year>-0.21)
```
**Parameters**: 5 (β₀, β₁, β₂, β₃, γ₀, γ₁)
**Justification**: Chow test p < 0.000001, 9.6x growth acceleration
**Use when**: Regime shift is scientifically interpretable

### Model 3 (BACKUP): Hierarchical B-Spline
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = α + Σ β_k × B_k(year[i])
log(φ[i]) = δ + Σ γ_j × B_j(year[i])
```
**Parameters**: 10+ (spline coefficients with hierarchical shrinkage)
**Justification**: Maximum flexibility, no functional form assumptions
**Use when**: Parametric models show systematic failures

---

## Key EDA Findings That Justify Complexity

1. **Severe overdispersion**: Var/Mean ≈ 70 (φ ≈ 1.5)
   - Negative Binomial required (not Poisson)

2. **Non-linear acceleration**: Growth rate increases 9.6x
   - Linear model inadequate (R² = 0.88 vs 0.96)

3. **Regime shift**: Chow test p < 0.000001 at year = -0.21
   - Structural break evident

4. **Heteroscedasticity**: Variance varies 20x over time
   - Constant dispersion wrong (Levene's test p < 0.01)

5. **No temporal autocorrelation**: After trend, residuals independent
   - No need for ARIMA/state-space

---

## Falsification Criteria (When to Simplify)

### Reject Model 1 if:
- β₂ credible interval includes 0 → No quadratic term needed
- γ₁ credible interval includes 0 → Constant dispersion sufficient
- LOO-CV: Simpler model better by > 2 SE
- Posterior predictive checks fail

### Reject Model 2 if:
- β₂ and β₃ both ≈ 0 → No regime shift
- Residuals cluster at changepoint → Artifact of discontinuity
- LOO-CV worse than quadratic

### Reject Model 3 if:
- Many Pareto-k > 0.7 → Overfitting
- LOO-CV worse than parametric models
- Cannot achieve convergence

### Abandon ALL models if:
- Negative Binomial cannot capture dispersion
- Systematic residual patterns persist
- Computational impossibility despite tuning

---

## Implementation Checklist

- [ ] Implement Model 1 (quadratic + time-varying φ)
- [ ] Run convergence diagnostics (R-hat, ESS)
- [ ] Posterior predictive checks (Var/Mean, coverage)
- [ ] Compute LOO-CV with Pareto-k diagnostics
- [ ] If Model 1 shows issues → Implement Model 2 or simplify
- [ ] Compare to Designer 1's simpler models via LOO-CV
- [ ] Document which approach better serves this dataset

---

## Expected Outcome

**Prediction**: Model 1 will succeed because:
- EDA provides strong evidence for β₂ ≠ 0 and γ₁ ≠ 0
- Visual inspection shows clear curvature and variance patterns
- Statistical tests highly significant

**Alternative outcomes**:
- If β₂ ≈ 0: Revert to log-linear (Designer 1 was right)
- If computational issues: Simplify to constant φ
- If LOO-CV favors simpler: Accept simpler model

**Meta-outcome**: Regardless of winner, we learn about data complexity vs model parsimony trade-off.

---

## Files in This Directory

```
/workspace/experiments/designer_2/
├── README.md                    (this file)
├── proposed_models.md           (detailed model specifications)
├── stan_model_templates.md      (Stan code implementations)
└── design_philosophy.md         (rationale & philosophy)
```

---

## Contact & Context

- **Designer**: Designer 2 (Flexibility-focused)
- **Date**: 2025-10-29
- **EDA Source**: `/workspace/eda/eda_report.md`
- **Parallel design**: Designer 1 likely proposes simpler log-linear models
- **Resolution**: Model comparison via LOO-CV will determine best approach

---

## Quick Start

1. **Read**: `proposed_models.md` for full specifications
2. **Understand**: `design_philosophy.md` for rationale
3. **Implement**: Use code from `stan_model_templates.md`
4. **Validate**: Run diagnostic checklist
5. **Compare**: LOO-CV against Designer 1's models

**Goal**: Find the model that genuinely explains the data, not prove my philosophy correct.
