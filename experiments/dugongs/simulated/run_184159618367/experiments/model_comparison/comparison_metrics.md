# Model Comparison Metrics

**Date**: 2025-10-27
**Models**: Experiment 1 (Asymptotic Exponential) vs Experiment 3 (Log-Log Power Law)
**Data**: 27 observations

---

## Summary Table

| Metric | Exp1: Asymptotic | Exp3: Log-Log | Better Model |
|--------|------------------|---------------|--------------|
| **LOO-CV Metrics** | | | |
| ELPD_loo | 22.19 ± 2.91 | **38.85 ± 3.29** | **Exp3** |
| p_loo (effective params) | 2.91 | 2.79 | Exp3 |
| Pareto k > 0.7 (bad) | 0/27 (0%) | 0/27 (0%) | Tie |
| Pareto k max | 0.455 | **0.399** | **Exp3** |
| **Model Complexity** | | | |
| Number of Parameters | 4 | **3** | **Exp3** |
| **Point Predictions** | | | |
| RMSE | **0.0933** | 0.1217 | **Exp1** |
| MAE | **0.0782** | 0.0957 | **Exp1** |
| **Calibration** | | | |
| 90% PI Coverage | 33.33% (9/27) | 33.33% (9/27) | Tie |
| Target Coverage | 90% | 90% | - |
| Calibration Status | Under-calibrated | Under-calibrated | - |
| **MCMC Diagnostics** | | | |
| R-hat (all params) | 1.00-1.00 | 1.00-1.01 | Both good |
| Min ESS bulk | 1354 | 1383 | Comparable |

---

## Model Rankings

### By ArviZ compare():

| Rank | Model | ELPD_loo | SE | ELPD_diff | dSE | Weight |
|------|-------|----------|----|-----------|----|--------|
| **1** | **Exp3_LogLog** | **38.85** | 3.29 | 0.00 | 0.00 | **1.00** |
| 2 | Exp1_Asymptotic | 22.19 | 2.91 | 16.66 | 2.60 | 0.00 |

### Interpretation:
- **Exp3 ranked #1** with stacking weight of 1.00
- **ΔELPD = 16.66 ± 2.60** (Exp1 worse than Exp3)
- **Decision threshold (2×SE) = 5.21**
- **Ratio: 16.66 / 5.21 = 3.20** → Highly significant difference

---

## Statistical Decision

### Primary Criterion: ELPD_loo
- **Winner**: Experiment 3 (Log-Log Power Law)
- **Margin**: 16.66 ± 2.60
- **Significance**: YES (3.2× decision threshold)
- **Confidence**: HIGH

### Parsimony Consideration:
- Exp3 has **fewer parameters** (3 vs 4)
- Even if tied on ELPD, Exp3 would win on parsimony
- But not tied - Exp3 dominates on both criteria

---

## Where Each Model Excels

### Experiment 1 (Asymptotic) Excels At:
1. **Point prediction accuracy**: RMSE = 0.0933 (vs 0.1217)
2. **Mean absolute error**: MAE = 0.0782 (vs 0.0957)
3. **Capturing asymptotic behavior** with interpretable plateau

### Experiment 3 (Log-Log) Excels At:
1. **Out-of-sample prediction**: ELPD = 38.85 (vs 22.19) ← PRIMARY CRITERION
2. **Model simplicity**: 3 parameters (vs 4)
3. **LOO-CV reliability**: max k = 0.399 (vs 0.455)
4. **Generalization**: Better predictive distribution despite higher RMSE
5. **Parsimony**: Simpler model with better performance

---

## Model Specifications

### Experiment 1: Asymptotic Exponential
**Formula**: Y = α - β·exp(-γ·x) + ε, where ε ~ Normal(0, σ²)

**Parameters**:
- α (alpha): Asymptotic upper limit = 2.563 ± 0.038
- β (beta): Vertical range = 1.006 ± 0.077
- γ (gamma): Approach rate = 0.205 ± 0.034
- σ (sigma): Residual SD = 0.102 ± 0.016

**Interpretation**: Y approaches 2.563 as x increases, with transition around x ≈ 5.

### Experiment 3: Log-Log Power Law
**Formula**: log(Y) = α + β·log(x) + ε, where ε ~ Normal(0, σ²)
**Equivalent**: Y = exp(α)·x^β·exp(ε)

**Parameters**:
- α (alpha): Log-intercept = 0.572 ± 0.025 → Y(x=1) ≈ 1.77
- β (beta): Power exponent = 0.126 ± 0.011 (elasticity)
- σ (sigma): Residual SD (log scale) = 0.055 ± 0.008

**Interpretation**: Y = 1.77·x^0.126, power law with diminishing returns.

---

## Pareto k Diagnostics Detail

### Experiment 1:
- Good (k ≤ 0.5): 27/27 (100%)
- Moderate (0.5 < k ≤ 0.7): 0/27 (0%)
- Bad (k > 0.7): 0/27 (0%)
- Maximum k: 0.455
- **Status**: RELIABLE

### Experiment 3:
- Good (k ≤ 0.5): 27/27 (100%)
- Moderate (0.5 < k ≤ 0.7): 0/27 (0%)
- Bad (k > 0.7): 0/27 (0%)
- Maximum k: 0.399
- **Status**: RELIABLE (slightly better)

---

## Residual Analysis

### Experiment 1:
- Residual mean: ~0.0
- Residual SD: 0.093
- Pattern: Slight heteroscedasticity (fan pattern)
- Distribution: Approximately normal

### Experiment 3:
- Residual mean: ~0.0
- Residual SD: 0.122
- Pattern: Better homoscedasticity
- Distribution: Approximately normal

---

## Coverage Analysis (Warning)

Both models show **severe under-calibration**:
- **Observed**: 33.33% (9/27 observations in 90% PI)
- **Expected**: 90% (24-25/27 observations)
- **Discrepancy**: 56.7 percentage points

**Implications**:
- Posterior intervals are too narrow
- Models are overconfident in predictions
- Uncertainty quantification is unreliable
- **Action required** before deployment for uncertainty

**Possible causes**:
- Priors on σ too informative/narrow
- Wrong likelihood family (consider Student-t)
- Model misspecification
- Need hierarchical structure

---

## Computational Diagnostics

### Experiment 1:
- Chains: 4
- Draws per chain: 1000
- Total samples: 4000
- R-hat range: 1.00-1.00 (excellent)
- ESS bulk range: 1354-2642
- ESS tail range: 2025-2453
- **Convergence**: YES

### Experiment 3:
- Chains: 4
- Draws per chain: 1000
- Total samples: 4000
- R-hat range: 1.00-1.01 (excellent)
- ESS bulk range: 1383-1738
- ESS tail range: 1467-1731
- **Convergence**: YES

Both models show excellent MCMC convergence.

---

## Decision Matrix

| Criterion | Weight | Exp1 Score | Exp3 Score | Winner |
|-----------|--------|------------|------------|--------|
| ELPD_loo | HIGH | 22.19 | 38.85 | Exp3 |
| Simplicity | MEDIUM | 4 params | 3 params | Exp3 |
| LOO reliability | MEDIUM | k=0.455 | k=0.399 | Exp3 |
| RMSE | LOW | 0.0933 | 0.1217 | Exp1 |
| MAE | LOW | 0.0782 | 0.0957 | Exp1 |
| Calibration | HIGH | 33% | 33% | Tie |
| Interpretability | LOW | Good | Good | Tie |

**Overall Winner**: **Experiment 3**
- Wins on all HIGH and MEDIUM weighted criteria
- Loses only on LOW weighted criteria (in-sample fit)

---

## Comparison to User-Reported R²

The user reported:
- Exp1: R² = 0.887, RMSE = 0.093
- Exp3: R² = 0.81, RMSE = 0.12

Our analysis confirms:
- **RMSE matches**: Exp1 = 0.0933 ✓, Exp3 = 0.1217 ✓
- **R² aligns with fit quality**: Exp1 fits training data better

**But**: Higher R² ≠ better model for prediction
- R² measures in-sample fit
- ELPD measures out-of-sample prediction
- **ELPD is the gold standard** for Bayesian model selection

---

## Key Insights

### 1. RMSE vs ELPD Paradox
- Exp1 has better RMSE but worse ELPD
- This indicates **overfitting**: good training fit, poor generalization
- ELPD correctly identifies this issue

### 2. Simplicity Wins
- Exp3 is simpler (3 vs 4 params) AND better (higher ELPD)
- Perfect example of Occam's Razor in action

### 3. Calibration Crisis
- Both models critically under-calibrated (33% vs 90%)
- This is independent of model selection
- Needs immediate attention before deployment

### 4. Power of LOO-CV
- All Pareto k < 0.7 → reliable estimates
- Clear statistical separation (ΔELPD = 3.2×threshold)
- Stacking weights (1.0 vs 0.0) confirm dominance

---

## Recommendation

**USE EXPERIMENT 3 (LOG-LOG POWER LAW)**

**Reasoning**:
1. Significantly better out-of-sample prediction (primary criterion)
2. Simpler model (secondary criterion)
3. Better LOO-CV reliability
4. Trade-off of higher RMSE is justified by ELPD gain

**But first**:
- Fix calibration issues (investigate priors, likelihood)
- Validate on holdout data
- Run posterior predictive checks

---

## Files Reference

- **Full Report**: `/workspace/experiments/model_comparison/comparison_report.md`
- **Executive Summary**: `/workspace/experiments/model_comparison/recommendation.md`
- **This File**: `/workspace/experiments/model_comparison/comparison_metrics.md`
- **Analysis Code**: `/workspace/experiments/model_comparison/code/model_comparison.py`
- **Summary CSV**: `/workspace/experiments/model_comparison/comparison_summary.csv`
- **Plots**: `/workspace/experiments/model_comparison/plots/*.png`

---

**Analysis Date**: 2025-10-27
**Method**: Bayesian LOO-CV with Pareto-smoothed importance sampling (ArviZ)
**Decision Rule**: ΔELPD > 2×SE for statistical significance
**Result**: Experiment 3 wins decisively (ΔELPD = 16.66 ± 2.60, ratio = 3.20)
