# Bayesian Modeling Experiment Plan

## Overview

This plan synthesizes proposals from three parallel model designers to create a comprehensive, falsification-driven Bayesian modeling strategy for the count data.

---

## Design Team Summary

### Designer 1: Parsimony Focus
- **Philosophy**: Simplicity, interpretability, computational efficiency
- **Primary model**: Log-linear Negative Binomial (3 parameters)
- **Secondary model**: Quadratic Negative Binomial (4 parameters)
- **Strength**: Fast, interpretable, robust baseline

### Designer 2: Flexibility Focus
- **Philosophy**: Capture complexity revealed by EDA
- **Primary model**: Quadratic + time-varying dispersion (5 parameters)
- **Secondary models**: Piecewise regime shift, B-spline
- **Strength**: Captures acceleration and heteroscedasticity

### Designer 3: Alternative Perspectives
- **Philosophy**: Challenge standard approaches, robustness
- **Primary models**: Student-t on log-scale, Hierarchical Gamma-Poisson
- **Strength**: Tests distributional assumptions, robustness

---

## Synthesized Model Priority List

Based on EDA evidence, theoretical justification, and computational feasibility:

### **Model 1: Log-Linear Negative Binomial** (BASELINE - Must Test)
**Source**: Designer 1 primary recommendation

**Specification**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  φ ~ Exponential(0.667)  # E[φ] = 1.5
```

**Rationale**:
- Simplest model (3 parameters)
- Exponential growth interpretation
- Strong baseline for comparison
- EDA: R² = 0.92, growth rate 135% per year

**Falsification criteria**:
- LOO-CV: ΔELPD > 4 vs. quadratic model
- Posterior predictive: Var/Mean outside [50, 90]
- Calibration: <80% observations in 90% PI
- Residuals show systematic U-shaped curvature

---

### **Model 2: Quadratic Negative Binomial** (PRIMARY - Must Test)
**Source**: Designer 1 secondary, Designer 2 primary (without time-varying dispersion)

**Specification**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0, 0.3)     # Centered at 0 to test necessity
  φ ~ Exponential(0.667)
```

**Rationale**:
- Captures acceleration in growth (EDA: R² = 0.96)
- Tests 9.6x growth rate increase from early to late period
- Moderate complexity (4 parameters)
- Nested within Model 1 for comparison

**Falsification criteria**:
- β₂ credible interval includes 0 with |β₂| < 0.1
- LOO-CV: ΔELPD < 2 vs. linear (not worth complexity)
- >10% observations with Pareto-k > 0.7
- Divergent transitions >0.5%

---

### **Model 3: Student-t on Log-Counts** (ALTERNATIVE - Should Test)
**Source**: Designer 3 primary recommendation

**Specification**:
```
log(C[i] + 1) ~ Student_t(ν, μ[i], σ)
μ[i] = β₀ + β₁ × year[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  σ ~ Exponential(1)
  ν ~ Gamma(2, 0.1)      # E[ν] = 20, allows heavy tails
```

**Rationale**:
- Challenges assumption that count structure matters
- Robust to extreme values via heavy-tailed Student-t
- Direct interpretation on log-scale
- Tests if overdispersion is just heavy-tailed Normal

**Falsification criteria**:
- ν > 30 (too Normal-like, use standard GLM)
- Back-transformed predictions fail Var/Mean ≈ 70
- Negative predictions on original scale
- LOO-CV worse than NegBin by >4 ELPD

---

### **Model 4: Quadratic + Time-Varying Dispersion** (ADVANCED - Optional)
**Source**: Designer 2 primary recommendation

**Specification**:
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
log(φ[i]) = γ₀ + γ₁ × year[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0.3, 0.3)   # Weakly informative toward positive
  γ₀ ~ Normal(0.4, 0.5)   # log(1.5) ≈ 0.4
  γ₁ ~ Normal(0, 0.3)
```

**Rationale**:
- Captures heteroscedasticity (variance varies 20x)
- EDA: Dispersion varies from Var/Mean = 0.58 to 11.85
- Most flexible of standard models
- Test if time-varying dispersion improves fit

**Falsification criteria**:
- γ₁ credible interval includes 0 (dispersion is constant)
- Both β₂ and γ₁ not significant (too complex, revert to Model 1)
- LOO-CV worse than Model 2 (overfitting)
- Computational issues (>5% divergences)

---

### **Model 5: Hierarchical Gamma-Poisson** (ALTERNATIVE - Optional)
**Source**: Designer 3 alternative

**Specification**:
```
C[i] ~ Poisson(λ[i])
λ[i] ~ Gamma(α[i], β)
α[i] = μ[i] × β
log(μ[i]) = β₀ + β₁ × year[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.3)
  β ~ Gamma(2, 0.1)      # E[β] ≈ 1.5
```

**Rationale**:
- Mathematically equivalent to NegBin but hierarchical
- Explicit random effects interpretation
- Can examine λ[i] for patterns
- Tests if overdispersion is unobserved heterogeneity

**Falsification criteria**:
- λ[i] shows no interpretable pattern
- Performance identical to Model 1 (use simpler)
- Computational cost >2× Model 1 without insight

---

## Minimum Attempt Policy

**Required experiments**: At minimum, test **Models 1 and 2** (log-linear and quadratic).

**Rationale**:
- Model 1 is the baseline reference
- Model 2 tests the core EDA finding (acceleration)
- Together they address the primary modeling question

**Additional models** (3-5):
- Test if time permits and if Models 1-2 show inadequacies
- Model 3 provides robustness check
- Models 4-5 address specific diagnostic failures

---

## Model Comparison Strategy

### Primary Metric: LOO-CV (ELPD)

Use `az.compare()` for all fitted models:

```python
comparison = az.compare({
    'linear': idata1,
    'quadratic': idata2,
    'student_t': idata3,
    ...
})
```

**Decision rules**:
- **ΔELPD < 2**: Models equivalent → Choose simpler (parsimony)
- **ΔELPD 2-4**: Weak preference → Consider interpretability
- **ΔELPD > 4**: Strong preference → Choose better model

### Secondary Metrics

1. **Posterior Predictive Checks**:
   - Var/Mean ratio ≈ 70 ± 20
   - 90% prediction interval coverage ≈ 90% ± 5%
   - Visual fit to observed data

2. **Convergence Diagnostics**:
   - R̂ < 1.01 for all parameters
   - ESS > 400 for all parameters
   - Divergences <0.5%

3. **LOO Diagnostics**:
   - Pareto-k < 0.7 for >95% observations
   - No systematic LOO failures

4. **Interpretability**:
   - Parameters align with EDA estimates
   - Scientifically meaningful
   - No extreme or implausible values

---

## Falsification Framework

### Model-Specific Rejection Criteria

See individual model sections above for specific criteria.

### Global Red Flags (Reconsider All Models)

If ANY of these occur, stop and reassess:

1. **All models fail Var/Mean recovery**: Posterior predictive consistently <30 or >120
   - **Action**: Need fundamentally different approach (e.g., mixture models)

2. **All models show poor calibration**: <70% coverage in all models
   - **Action**: Check data quality, consider measurement error models

3. **Computational pathologies in all models**: Divergences, non-convergence
   - **Action**: Data-model pathology, reconsider likelihood family

4. **Prior-posterior conflict in all models**: Posteriors >>3 SD from EDA
   - **Action**: EDA estimates may be misleading, investigate data

5. **All LOO diagnostics poor**: Many Pareto-k > 0.7 across models
   - **Action**: Influential points not captured, need robust methods

### Escape Routes

If standard models fail:

1. **Time-varying dispersion** (Model 4) - If heteroscedasticity severe
2. **Piecewise/changepoint model** - If regime shift at year ≈ -0.21 is real
3. **Gaussian Process** - If functional form is too complex for polynomials
4. **Measurement error model** - If data quality issues discovered
5. **Simpler Bayesian approach** - If all complex models fail, return to basics

---

## Implementation Workflow

### Phase 3a: Prior Predictive Checks (All Models)

For each model:
1. Sample from priors only
2. Generate predicted counts
3. Check if prior predictions are reasonable:
   - Count range: Should include [21, 269]
   - Var/Mean: Should allow for ~70
   - No extreme values (e.g., counts >10,000)

**Decision**: If prior predictive is unreasonable, revise priors before fitting.

### Phase 3b: Simulation-Based Validation

For Models 1-2 (minimum):
1. Simulate data from model with known parameters
2. Fit model to simulated data
3. Check parameter recovery
4. If recovery fails, diagnose before fitting real data

### Phase 3c: Model Fitting

For each model that passes validation:
1. Fit using CmdStanPy with MCMC
2. Check convergence (R̂, ESS, divergences)
3. If convergence fails, try:
   - Increase iterations
   - Adjust step size / tree depth
   - Reparameterize
4. If still fails after tuning → Document and move to next model

### Phase 3d: Posterior Predictive Checks

For each converged model:
1. Generate posterior predictive samples
2. Check Var/Mean ratio recovery
3. Check prediction interval calibration
4. Visual diagnostics (observed vs. predicted)
5. Compute LOO for model comparison

### Phase 3e: Model Critique

For each model:
- Apply falsification criteria
- Make ACCEPT/REVISE/REJECT decision
- If REVISE, create refinement plan
- If REJECT, document why and move on

---

## Expected Timeline

Assuming all models tested:

| Phase | Model | Time | Cumulative |
|-------|-------|------|------------|
| Prior pred | Model 1 | 5 min | 5 min |
| SBC | Model 1 | 15 min | 20 min |
| Fitting | Model 1 | 3 min | 23 min |
| PPC | Model 1 | 5 min | 28 min |
| Prior pred | Model 2 | 5 min | 33 min |
| SBC | Model 2 | 15 min | 48 min |
| Fitting | Model 2 | 4 min | 52 min |
| PPC | Model 2 | 5 min | 57 min |
| Model 3 | All phases | 20 min | 77 min |
| Comparison | All models | 10 min | 87 min |

**Total**: ~90 minutes for Models 1-3
**Minimum**: ~60 minutes for Models 1-2 only

---

## Success Criteria

This experiment plan **succeeds** if:

1. At least 2 models (including Model 1) converge successfully
2. LOO-CV provides clear ranking or shows equivalence
3. Best model(s) pass posterior predictive checks
4. Results are scientifically interpretable
5. Findings robust across multiple model specifications

This experiment plan **teaches us even if models fail**:

- If all fail Var/Mean recovery → Need different distributional approach
- If Model 1 succeeds but Model 2 doesn't improve → Simplicity wins
- If Model 2 strongly outperforms → Acceleration is real
- If Student-t performs well → Count structure may not matter

---

## Summary of Recommended Testing Order

1. **Model 1** (Log-Linear NegBin) - REQUIRED baseline
2. **Model 2** (Quadratic NegBin) - REQUIRED to test acceleration
3. **Model 3** (Student-t) - RECOMMENDED for robustness
4. **Model 4** (Time-varying dispersion) - If Models 1-2 show calibration issues
5. **Model 5** (Hierarchical) - If time permits and interpretability desired

**Core question**: Is the growth pattern simple exponential (Model 1) or accelerating (Model 2)?

**Secondary questions**:
- Does distributional choice matter? (Model 3 vs. 1)
- Is heteroscedasticity important? (Model 4 vs. 2)

---

## References

- **EDA Report**: `eda/eda_report.md`
- **Designer 1 Details**: `experiments/designer_1/proposed_models.md`
- **Designer 2 Details**: `experiments/designer_2/proposed_models.md`
- **Designer 3 Details**: `experiments/designer_3/proposed_models.md`
- **EDA Synthesis**: `eda/synthesis.md`
