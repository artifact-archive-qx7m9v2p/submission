# Quick Reference: Analyst 3 Findings

## TL;DR - The Bottom Line

**Use a GLM with log link (Poisson or Negative Binomial) with quadratic terms in year.**

This respects the count data structure, leverages the optimal log transformation, and captures the nonlinear growth pattern.

---

## Critical Numbers

| Metric | Value | Implication |
|--------|-------|-------------|
| **Variance ratio (original)** | 34.7x | SEVERE heteroscedasticity |
| **Variance ratio (log)** | 0.58x | Near-perfect stabilization |
| **Log correlation with year** | 0.967 | Excellent linearity |
| **Quadratic R²** | 0.961 | Strong fit |
| **Exponential R²** | 0.929 | Good fit, better residuals |
| **Growth rate** | 2.34x/year | Exponential growth parameter |

---

## The Transformation Winner: LOG

| Criterion | Original | Log | Winner |
|-----------|----------|-----|--------|
| Linearity (r) | 0.941 | **0.967** | Log ✓ |
| Variance ratio | 34.7 | **0.58** | Log ✓ |
| Residual normality (p) | 0.071 | **0.945** | Log ✓ |
| Interpretability | Moderate | **High** | Log ✓ |

**Box-Cox confirms**: Optimal λ = -0.036 ≈ 0 (log transformation)

---

## Model Recommendations (Ranked)

### 1. Poisson GLM with Log Link (RECOMMENDED)
```
C ~ Poisson(μ)
log(μ) = β₀ + β₁×year + β₂×year²
```
**Pros**: Respects count structure, optimal transformation, built-in variance modeling
**Cons**: More complex than OLS
**Use when**: Data are counts, proper inference needed

### 2. Negative Binomial GLM (if overdispersed)
```
C ~ NegBin(μ, θ)
log(μ) = β₀ + β₁×year + β₂×year²
```
**Pros**: Handles overdispersion, more robust
**Cons**: Extra parameter to estimate
**Use when**: Poisson shows overdispersion (deviance/df > 1.5)

### 3. Log-Linear OLS (simple alternative)
```
log(C) = β₀ + β₁×year + β₂×year² + ε
```
**Pros**: Simple, excellent residuals, easy to implement
**Cons**: Back-transformation bias, less principled for counts
**Use when**: Simplicity prioritized, careful with predictions

---

## Feature Engineering

### DO Use
- ✅ `year` (r = 0.941)
- ✅ `year²` (captures acceleration)
- ✅ `log(C)` as transformed response
- ✅ `exp(0.5×year)` as alternative feature (r = 0.973)

### DON'T Use
- ❌ `year³⁺` (overfitting risk, n=40)
- ❌ Inverse transformations (r = -0.90)
- ❌ Square transformations (variance ratio = 2062!)
- ❌ Power law specification (R² = 0.70)

---

## Polynomial Degree Selection

| Degree | R² | AIC | Verdict |
|--------|-----|-----|---------|
| 1 | 0.885 | 273.2 | Insufficient |
| **2** | **0.961** | **231.8** | **RECOMMENDED** ✓ |
| 3 | 0.976 | 214.9 | Good but may overfit |
| 4+ | 0.990+ | <182 | Overfitting risk |

**Recommendation**: Start with degree 2, test degree 3 if needed.

---

## Critical Visualizations

### Must-See Plots
1. **`01_transformation_comparison.png`** - See why log wins
2. **`03_variance_stabilization.png`** - See variance problem solved
3. **`05_all_models_comparison.png`** - Compare all model fits
4. **`08_model_selection_criteria.png`** - AIC/BIC guidance

### For Deep Dive
5. **`02_residual_diagnostics.png`** - Residual normality proof
6. **`06_boxcox_optimization.png`** - Box-Cox parameter justification
7. **`10_scale_location_plots.png`** - Homoscedasticity check

---

## Implementation Template

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Load
df = pd.read_csv('/workspace/data/data_analyst_3.csv')

# Create features
X = sm.add_constant(pd.DataFrame({
    'year': df['year'],
    'year2': df['year']**2
}))

# Fit Poisson GLM
model_poisson = sm.GLM(df['C'], X, family=sm.families.Poisson()).fit()

# Check overdispersion
od = model_poisson.deviance / model_poisson.df_resid
print(f"Overdispersion: {od:.2f}")

# Use NegBin if od > 1.5
if od > 1.5:
    model = sm.GLM(df['C'], X, family=sm.families.NegativeBinomial()).fit()
else:
    model = model_poisson

# Predictions (automatically on correct scale)
mu_hat = model.predict(X)
```

---

## Red Flags to Avoid

1. ❌ **Using original scale OLS without addressing heteroscedasticity** - variance increases 34x!
2. ❌ **Ignoring count data nature** - these are integers with variance-mean relationship
3. ❌ **Using linear model** - R² only 0.88, growth is clearly nonlinear
4. ❌ **High-degree polynomials** - sample size (n=40) too small for degree 4+
5. ❌ **Ignoring transformation** - residuals will be non-normal and heteroscedastic

---

## Key Insights

### What the Data Tell Us
- **Growth type**: Exponential with acceleration (2.34x per year, increasing)
- **Variance structure**: Proportional to mean (Poisson-like)
- **Scale**: Multiplicative errors, not additive
- **Best representation**: Log scale linearizes and stabilizes

### What This Means for Modeling
- GLM framework is natural fit
- Log link is theoretically justified
- Quadratic term captures acceleration
- Standard OLS inappropriate without transformation

### Confidence Levels
- **High confidence**: Log is optimal, variance heterogeneity severe, growth > linear
- **Medium confidence**: Exact form (quadratic vs exponential), optimal degree
- **Needs checking**: Overdispersion, temporal autocorrelation, influential points

---

## Decision Tree

```
Is your goal inference or prediction?
│
├─ Inference (p-values, CIs)
│  └─ Use Poisson/NegBin GLM ✓
│
└─ Prediction only
   │
   ├─ Want interpretability?
   │  └─ Yes → Log-linear OLS
   │  └─ No → Quadratic (highest R²)
   │
   └─ Want to respect count structure?
      └─ Yes → Poisson/NegBin GLM ✓
      └─ No → Log-linear OLS
```

**In most cases**: Use GLM (Tier 1 recommendation)

---

## Validation Checklist

Before finalizing model:
- [ ] Check deviance residuals for patterns
- [ ] Assess overdispersion parameter
- [ ] Compare Poisson vs NegBin via AIC
- [ ] Test polynomial degree (2 vs 3)
- [ ] Identify influential points
- [ ] Validate on holdout set if possible
- [ ] Check temporal autocorrelation if time series

---

## Questions Answered

**Q: What transformation should I use?**
A: Log transformation (λ≈0 by Box-Cox)

**Q: Linear or nonlinear?**
A: Nonlinear - quadratic or exponential (both fit well)

**Q: OLS or GLM?**
A: GLM preferred (respects count structure)

**Q: What distribution family?**
A: Start with Poisson, upgrade to NegBin if overdispersed

**Q: Include quadratic term?**
A: Yes - significant improvement (ΔAIC = -41)

**Q: Can I use original scale?**
A: Only with WLS or robust SE (heteroscedasticity is severe)

---

## Files to Read

1. **Quick overview**: This file
2. **Main findings**: `findings.md` (562 lines, comprehensive)
3. **Exploration details**: `eda_log.md` (381 lines, step-by-step)
4. **Full context**: `README.md` (directory guide)

## Files to Run

```bash
# Reproduce everything
python /workspace/eda/analyst_3/code/00_run_all_analyses.py

# Individual analyses
python /workspace/eda/analyst_3/code/01_initial_exploration.py
python /workspace/eda/analyst_3/code/02_transformation_analysis.py
python /workspace/eda/analyst_3/code/02b_polynomial_analysis.py
python /workspace/eda/analyst_3/code/03_visualization_transformations.py
python /workspace/eda/analyst_3/code/04_advanced_visualizations.py
```

---

**Last Updated**: 2025-10-29
**Analyst**: Analyst 3 (Feature Engineering & Transformations)
**Bottom Line**: GLM with log link + quadratic terms
