# Experiment Plan: Robust Bayesian Modeling
## Designer 3 - Focus: Robust & Alternative Approaches

**Date**: 2025-10-28
**Dataset**: Y vs x (n=27) with influential outlier
**Challenge**: Handle outlier (x=31.5, Cook's D=1.51) and test regime structure

---

## Problem Formulation: Competing Hypotheses

### Hypothesis 1: Smooth Saturation with Outlier Noise
**Model**: Student-t Logarithmic
**Claim**: Relationship is smooth logarithmic curve; outlier at x=31.5 is measurement error or heavy-tailed noise

**I will abandon this if:**
- nu posterior > 30 (Normal likelihood sufficient, robustness unnecessary)
- Residuals show systematic regime structure
- Outlier is actually start of downturn (check with synthetic data extension)

---

### Hypothesis 2: True Two-Regime Heterogeneity
**Model**: Mixture of Two Linear Regimes
**Claim**: Sharp transition at x~7 reflects real biological/physical regime change; observations belong to distinct populations

**I will abandon this if:**
- Regime probabilities are diffuse (all observations 40-60% in each regime)
- Estimated slopes beta1 ≈ beta2 (no real difference)
- Worse LOO-CV than simpler models (overfitting)

---

### Hypothesis 3: Hidden Variance Structure
**Model**: Hierarchical Variance with Spatial Trend
**Claim**: Variance increases with x (sparse high-x region); outlier reflects location-specific uncertainty, not error

**I will abandon this if:**
- zeta posterior includes 0 (no variance trend)
- tau ≈ 0 (no observation-level variance variation)
- All sigma_i are similar (hierarchy collapses)

---

## Model Specifications

### Model 1: Student-t Logarithmic (HIGHEST PRIORITY)

```
Y_i ~ StudentT(nu, mu_i, sigma)
mu_i = beta_0 + beta_1 * log(x_i)

Priors:
- beta_0 ~ Normal(2.3, 0.5)     # Intercept
- beta_1 ~ Normal(0.29, 0.15)   # Log-slope
- sigma ~ Exponential(10)       # Scale
- nu ~ Gamma(2, 0.1)            # DoF: mean=20, allows 3-100
```

**Falsification**: nu > 30 → refit with Normal; residual patterns → wrong mean function

**Stress test**: Add synthetic downturn data at x>35; does model break or adapt?

---

### Model 2: Mixture of Two Regimes (MEDIUM PRIORITY)

```
regime_i ~ Categorical(pi_i)
logit(pi_i[regime=1]) = gamma_0 + gamma_1 * x_i

If regime_i = 1: Y_i ~ N(alpha1 + beta1*x_i, sigma1)  # Growth
If regime_i = 2: Y_i ~ N(alpha2 + beta2*x_i, sigma2)  # Plateau

Priors:
- beta1 ~ Normal(0.11, 0.05)    # Steep slope
- beta2 ~ Normal(0.02, 0.02)    # Shallow slope
- gamma_1 ~ Normal(-1, 0.5)     # Higher x → plateau
```

**Falsification**: beta1 ≈ beta2 → no regimes; diffuse regime assignment → not learning

**Stress test**: Simulate smooth log data; does model collapse to single regime?

---

### Model 3: Hierarchical Variance (LOWER PRIORITY)

```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = eta + zeta*(x_i - x_mean) + epsilon_i
epsilon_i ~ Normal(0, tau)

Priors:
- zeta ~ Normal(0, 0.2)         # Variance trend
- tau ~ Exponential(2)          # Variance variability
```

**Falsification**: zeta ≈ 0 AND tau ≈ 0 → constant variance sufficient

**Stress test**: Fit to constant-variance data; does it correctly infer homoscedasticity?

---

## Red Flags: When to Pivot

### Immediate Red Flags (STOP and reconsider):

1. **All models fail diagnostics**
   → Mean function fundamentally wrong; try non-monotonic models (polynomial, spline)

2. **Systematic residual patterns persist**
   → Missing interaction terms or need non-parametric (Gaussian Process)

3. **x=31.5 has high Pareto-k in ALL models**
   → Observation incompatible; sensitivity analysis excluding it

4. **Prior-posterior conflict**
   → Priors wrong OR model class inappropriate

5. **Extreme parameter values**
   → Check data errors or alternative formulations

---

## Decision Points for Major Pivots

### Decision Point 1: After Model 1 Results

**IF nu < 10**: Robustness crucial, continue with Model 1
**IF nu > 30**: Normal sufficient, refit simpler version
**IF residuals patterned**: Try Model 2 (regime structure)

### Decision Point 2: After Model 2 Results

**IF mixture wins AND regimes clear**: Two-regime structure is real
**IF mixture wins BUT regimes diffuse**: Model overfitting, back to Model 1
**IF Model 1 wins**: Smooth curve better, regimes are artifact

### Decision Point 3: After All Three Models

**IF LOO-CV within 2 SE**: Models similar, use Bayesian Model Averaging
**IF one model clearly wins**: Report as primary, sensitivity checks
**IF all perform poorly**: Move to Backup Models (see below)

---

## Alternative Approaches If Initial Models Fail

### Backup 1: Gaussian Process Regression
**When**: All parametric models show poor fit
**Why**: Non-parametric, very flexible, no functional form assumption
**Cost**: Harder to interpret, computational expense

### Backup 2: Non-Monotonic Polynomial
**When**: Evidence of downturn after plateau (outlier signals this)
**Why**: Can capture turning points
**Cost**: Poor extrapolation, overfitting risk

### Backup 3: Changepoint with Student-t
**When**: Both robustness AND regime structure needed
**Why**: Combines Model 1 and Model 2 strengths
**Cost**: 5+ parameters, complexity

---

## Success Criteria

### MANDATORY Diagnostics (all must pass):

- R-hat < 1.01 for ALL parameters
- ESS_bulk > 400, ESS_tail > 400
- Divergences < 1% of samples
- Posterior predictive p-value in [0.05, 0.95]
- Residuals patternless vs x and fitted values
- LOO-CV Pareto-k < 0.7 for >90% of observations

### Model Comparison:

1. **Primary**: LOO-CV ELPD (higher better)
2. **Secondary**: Posterior predictive coverage (~95% for 95% intervals)
3. **Tertiary**: Simplicity (Occam's razor if performance similar)
4. **Qualitative**: Scientific interpretability

---

## Warning Signs for Model Misspecification

### Computational Red Flags:
- Divergences >1%: Model too complex or misspecified
- R-hat > 1.01: Poor convergence, try reparameterization
- ESS < 400: Inefficient sampling, longer chains needed

### Statistical Red Flags:
- Residuals show patterns: Wrong mean function
- Poor posterior predictive checks: Model doesn't capture data features
- High Pareto-k values: Influential observations, model sensitivity

### Scientific Red Flags:
- Parameters at prior boundaries: Prior too restrictive or model wrong
- Extreme estimates: Check data quality or model form
- Inconsistent results across subsets: Model instability

---

## Implementation Timeline

### Week 1: Baseline (Model 1)
- **Day 1-2**: Code Student-t Log model in Stan
- **Day 3-4**: Fit, diagnose convergence
- **Day 5**: Posterior predictive checks, residual analysis

### Week 2: Alternatives (Models 2 & 3)
- **Day 1-2**: Code Mixture model
- **Day 3-4**: Code Hierarchical Variance model
- **Day 5**: Fit all three, compare LOO-CV

### Week 3: Finalization
- **Day 1-2**: Sensitivity analysis (prior, outlier)
- **Day 3-4**: Generate predictions, visualizations
- **Day 5**: Write results, model selection rationale

---

## Critical Experiments to Run

### Experiment 1: Outlier Influence
**Protocol**: Fit Model 1 with and without x=31.5
**Question**: How much does outlier affect inferences?
**Success**: Results stable; outlier handled via nu parameter
**Failure**: Large changes; need to investigate measurement validity

### Experiment 2: Regime Reality Check
**Protocol**: Fit Model 2, examine regime probabilities
**Question**: Is changepoint at x=7 real or sampling artifact?
**Success**: Clear separation, transition zone narrow (x=6-8)
**Failure**: Diffuse probabilities, no clear structure → smooth model better

### Experiment 3: Extrapolation Behavior
**Protocol**: Generate predictions at x = 50, 100, 150
**Question**: What happens beyond observed range?
**Success**: Realistic asymptotic behavior for Models 1-2
**Failure**: Unrealistic (e.g., Y → infinity for log model) → need asymptotic model

---

## Key Scientific Questions

1. **Is outlier at x=31.5 measurement error or real?**
   - Answer via: nu posterior (Model 1), regime assignment (Model 2), sigma_i (Model 3)

2. **Is two-regime structure real or artifact?**
   - Answer via: Model 2 LOO-CV comparison, regime probabilities

3. **How uncertain are predictions at high x (>20)?**
   - Answer via: Posterior predictive intervals, widening at sparse regions

4. **What is asymptotic behavior?**
   - Model 1: Logarithmic growth continues
   - Model 2: Plateau at Y ≈ 2.5-2.7
   - Check: Predictions at x=50, 100

---

## Expected Outcome

**Most Likely**: Model 1 (Student-t Log) wins

**Reasoning**:
- Simplest robust model
- Log transformation well-justified (EDA R²=0.897)
- Outlier handled automatically via nu
- nu posterior likely 5-15 (moderately robust)

**Reporting**:
- Primary: Model 1 with nu, sigma, beta estimates
- Secondary: Compare to Normal likelihood (justify robustness)
- Sensitivity: Refit without x=31.5, show stability

**If unexpected**: Be ready to pivot to Model 2 (if regimes dominate) or Backups (if all fail)

---

## Falsification Summary

**Model 1**: Abandon if nu>30 (normal sufficient) OR residuals patterned
**Model 2**: Abandon if regimes diffuse OR beta1≈beta2 OR worse LOO-CV
**Model 3**: Abandon if zeta≈0 AND tau≈0 (no heteroscedasticity)

**ALL Models**: Abandon if convergence fails, residuals patterned, or LOO-CV poor

**Pivot to Backups**: If all models fail diagnostics or predictive checks

---

## What Success Looks Like

- At least ONE model passes all MCMC diagnostics
- LOO-CV clearly ranks models (or shows they're similar)
- Posterior predictive checks show good data fit
- Outlier handled principled way (not dropped)
- Uncertainty quantified appropriately
- Results scientifically interpretable

## What Failure Looks Like

- All models fail convergence
- Residual patterns persist
- Extreme posteriors or prior conflicts
- Poor out-of-sample prediction
- Results don't make scientific sense

**If we fail**: Don't force it. Move to non-parametric methods or collect more data.

---

**Prepared by Designer 3: Robust & Alternative Modeling**
**Full model details**: `/workspace/experiments/designer_3/proposed_models.md`
**Ready for implementation**
