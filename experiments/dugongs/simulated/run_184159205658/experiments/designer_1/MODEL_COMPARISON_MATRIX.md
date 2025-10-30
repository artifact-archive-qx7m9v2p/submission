# Model Comparison Matrix: Designer 1

**Quick decision guide for choosing between three proposed models**

---

## At-a-Glance Comparison

| Criterion | Model 1: Logarithmic | Model 2: Power Law | Model 3: Asymptotic |
|-----------|---------------------|-------------------|---------------------|
| **Parameters** | 2 (β₀, β₁) | 3 (β₀, β₁, β₂) | 3 (Y_min, Y_max, K) |
| **Functional Form** | β₀ + β₁·log(x) | β₀ + β₁·x^β₂ | Y_min + Y_range·x/(K+x) |
| **Complexity** | ★☆☆ Simple | ★★☆ Moderate | ★★★ Complex |
| **Interpretability** | ★★★ High | ★★☆ Medium | ★★★ High |
| **Computational Cost** | ★☆☆ Fast | ★★☆ Medium | ★★★ Slow |
| **Expected R²** | 0.83 | 0.85 | 0.82 |
| **EDA Support** | Strong | Moderate | Moderate |
| **Overfitting Risk** | Low (N/p = 13.5) | Medium (N/p = 9) | Medium (N/p = 9) |
| **Extrapolation** | Unbounded | Unbounded | Bounded (Y_max) |
| **Sampling Difficulty** | Easy | Medium | Hard |
| **Priority Rank** | **1st - START HERE** | 2nd - If needed | 3rd - If theory demands |

---

## Decision Tree

```
START
  │
  └─> Q1: Is Model 1 (Log) adequate?
       │
       ├─ YES (passes all checks) ──> DONE, use Model 1 ✓
       │
       └─ NO (residual patterns, poor fit)
            │
            └─> Q2: Do residuals show specific pattern?
                 │
                 ├─ Quadratic/complex ──> Try Model 2 (Power Law)
                 │                          │
                 │                          └─> Q3: LOO improvement?
                 │                               │
                 │                               ├─ YES (ΔELPD > 3) ──> Use Model 2 ✓
                 │                               └─ NO ──> Stick with Model 1
                 │
                 └─ Plateau at high x ──> Try Model 3 (Asymptotic)
                                           │
                                           └─> Q4: K well-identified?
                                                │
                                                ├─ YES ──> Use Model 3 ✓
                                                └─ NO ──> Revert to Model 1 or 2
```

---

## Detailed Comparison

### Strengths

| Model | Key Strengths |
|-------|---------------|
| **Logarithmic** | • Fewest parameters (2)<br>• Clear elasticity interpretation<br>• Fast sampling<br>• Strong EDA support (R²=0.83)<br>• Standard functional form |
| **Power Law** | • Flexible (nests linear)<br>• Can capture various curvatures<br>• Approaches log as β₂→0<br>• Better empirical fit potential |
| **Asymptotic** | • Bounded predictions (Y_max)<br>• Clear mechanistic interpretation<br>• Graceful extrapolation<br>• Theoretical motivation (saturation) |

### Weaknesses

| Model | Key Weaknesses |
|-------|----------------|
| **Logarithmic** | • Unbounded Y as x→∞<br>• Fixed curvature shape<br>• May miss complex patterns |
| **Power Law** | • Parameter correlation (β₁, β₂)<br>• Less interpretable exponent<br>• Overfitting risk (N=27)<br>• Computational challenges |
| **Asymptotic** | • Most complex (nonlinear)<br>• Slowest sampling<br>• May fail if no saturation<br>• Requires good priors for K |

---

## When to Choose Each Model

### Choose Model 1 (Logarithmic) if:
✓ Seeking simplest adequate model
✓ Interpretability is priority
✓ N=27 is concerning for overfitting
✓ Unbounded Y is acceptable
✓ EDA log fit was strong (R²=0.83)
✓ Computational resources limited

**Expected success rate: 80%**

### Choose Model 2 (Power Law) if:
✓ Model 1 shows systematic residual patterns
✓ Need more flexible curvature
✓ LOO-CV shows substantial improvement (ΔELPD > 3)
✓ Willing to accept 3-parameter complexity
✓ Have time for extended sampling/tuning

**Expected success rate: 15%**

### Choose Model 3 (Asymptotic) if:
✓ Domain theory predicts saturation
✓ High-x data (x>20) shows clear plateau
✓ Unbounded predictions are scientifically implausible
✓ Y_max has clear physical meaning
✓ Willing to handle computational challenges

**Expected success rate: 5%**

---

## Parameter Interpretation Guide

### Model 1: Y = β₀ + β₁·log(x)

| Parameter | Interpretation | Plausible Range | Prior |
|-----------|----------------|-----------------|-------|
| **β₀** | Y when log(x)=0 (x=1) | [0.7, 2.7] | Normal(1.73, 0.5) |
| **β₁** | Elasticity coefficient | [0, 0.6] | Normal(0.28, 0.15) |
| **σ** | Residual standard deviation | [0.1, 0.4] | Exponential(5) |

**Key relationship:** 1% increase in x → (β₁/100)% increase in Y

---

### Model 2: Y = β₀ + β₁·x^β₂

| Parameter | Interpretation | Plausible Range | Prior |
|-----------|----------------|-----------------|-------|
| **β₀** | Baseline Y | [1.3, 2.3] | Normal(1.8, 0.5) |
| **β₁** | Scale factor | [0, 1.5] | Normal(0.5, 0.3) |
| **β₂** | Exponent (curvature) | [0, 1] | Normal(0.3, 0.2) |
| **σ** | Residual SD | [0.1, 0.4] | Exponential(5) |

**Key relationship:**
- β₂ < 1 → diminishing returns (concave)
- β₂ = 1 → linear
- β₂ > 1 → accelerating returns (unlikely here)

---

### Model 3: Y = Y_min + Y_range·x/(K+x)

| Parameter | Interpretation | Plausible Range | Prior |
|-----------|----------------|-----------------|-------|
| **Y_min** | Minimum Y (x→0) | [1.1, 2.3] | Normal(1.7, 0.3) |
| **Y_range** | Y_max - Y_min | [0.3, 1.5] | Normal(0.9, 0.3) |
| **K** | Half-saturation constant | [0.5, 20] | LogNormal(log(5), 1) |
| **σ** | Residual SD | [0.1, 0.4] | Exponential(5) |

**Derived:** Y_max = Y_min + Y_range ≈ [2.0, 3.0]

**Key relationship:** At x=K, response is halfway between Y_min and Y_max

---

## Computational Comparison

### Sampling Efficiency

| Model | Expected Time | Chains | Tuning | Target Accept | Divergences Expected |
|-------|---------------|--------|--------|---------------|----------------------|
| **Log** | 30 sec | 4 | 1000 | 0.95 | ~0% |
| **Power** | 2 min | 4 | 2000 | 0.99 | ~1-2% |
| **Asymptotic** | 5 min | 4 | 3000 | 0.99 | ~2-5% |

*Times approximate on modern CPU with PyMC*

### Convergence Difficulty

| Model | R-hat Concerns | ESS Concerns | Reparameterization Needed? |
|-------|----------------|--------------|----------------------------|
| **Log** | None | None | No |
| **Power** | β₁-β₂ correlation | Moderate | Maybe (if corr > 0.9) |
| **Asymptotic** | K identification | High | Likely (non-centered) |

---

## Model Selection Criteria

### Primary Metrics (in order of importance)

1. **Scientific plausibility:** Do parameters make sense?
2. **Posterior predictive checks:** >90% coverage
3. **LOO-CV:** Best predictive performance
4. **Parsimony:** Fewest parameters that work
5. **Interpretability:** Can we explain it?
6. **Computational feasibility:** Converges reliably

### Quantitative Thresholds

| Criterion | Acceptable | Good | Excellent |
|-----------|-----------|------|-----------|
| **R-hat** | < 1.05 | < 1.02 | < 1.01 |
| **ESS (bulk)** | > 400 | > 1000 | > 2000 |
| **PPC coverage** | > 80% | > 90% | > 95% |
| **LOO Pareto k<0.7** | > 70% | > 80% | > 90% |
| **RMSE** | < 0.30 | < 0.25 | < 0.20 |
| **Shapiro p-value** | > 0.01 | > 0.05 | > 0.10 |

---

## Pairwise Comparisons

### Model 1 vs Model 2

**Choose Model 1 if:**
- LOO-CV ΔELPD < 3 (not substantially better)
- Model 2 has correlation |ρ(β₁,β₂)| > 0.9
- Model 2 has divergences
- Interpretability matters

**Choose Model 2 if:**
- LOO-CV ΔELPD > 3 (substantially better)
- Residuals from Model 1 show clear pattern
- Extra complexity justified by fit

---

### Model 1 vs Model 3

**Choose Model 1 if:**
- No evidence of saturation in data
- Y_max posterior very wide (unidentified)
- Computational resources limited
- Simpler is better

**Choose Model 3 if:**
- High-x data clearly plateau
- Domain theory predicts saturation
- Extrapolation beyond x=31.5 needed
- Y_max has physical meaning

---

### Model 2 vs Model 3

**Choose Model 2 if:**
- No clear saturation
- Better empirical fit (LOO-CV)
- Computational efficiency matters
- β₂ well-identified

**Choose Model 3 if:**
- Saturation theoretically expected
- Y_max interpretable
- Bounded predictions required
- Long-term extrapolation needed

---

## Expected Posterior Ranges

### Model 1 (Logarithmic)

| Parameter | Prior Mean | Expected Posterior Mean | Expected 95% CI |
|-----------|------------|------------------------|-----------------|
| β₀ | 1.73 | 1.70 - 1.75 | [1.50, 1.95] |
| β₁ | 0.28 | 0.26 - 0.30 | [0.20, 0.36] |
| σ | 0.20 | 0.18 - 0.21 | [0.14, 0.26] |

**Data informativeness:** High (prior → posterior shift)

---

### Model 2 (Power Law)

| Parameter | Prior Mean | Expected Posterior Mean | Expected 95% CI |
|-----------|------------|------------------------|-----------------|
| β₀ | 1.80 | 1.75 - 1.85 | [1.50, 2.10] |
| β₁ | 0.50 | 0.40 - 0.60 | [0.20, 1.00] |
| β₂ | 0.30 | 0.25 - 0.35 | [0.10, 0.50] |
| σ | 0.20 | 0.17 - 0.20 | [0.13, 0.25] |

**Data informativeness:** Moderate (wider CIs due to correlation)

---

### Model 3 (Asymptotic)

| Parameter | Prior Mean | Expected Posterior Mean | Expected 95% CI |
|-----------|------------|------------------------|-----------------|
| Y_min | 1.70 | 1.70 - 1.75 | [1.50, 1.95] |
| Y_range | 0.90 | 0.80 - 1.00 | [0.50, 1.30] |
| K | 5.00 | 2.00 - 8.00 | [0.50, 20.0] |
| σ | 0.20 | 0.18 - 0.21 | [0.14, 0.26] |

**Data informativeness:** Low for K (sparse high-x data)

---

## Stress Test Results (Predicted)

### Extrapolation to x=50

| Model | Prediction (mean) | 95% CI Width | Plausibility |
|-------|------------------|--------------|--------------|
| **Log** | 3.1 | 0.8 | Unbounded, wide |
| **Power** | 3.0 | 0.9 | Unbounded, wide |
| **Asymptotic** | 2.6 | 0.3 | Bounded, narrow |

**Caution:** All extrapolations uncertain (only 3 obs with x>20)

---

### Leave-High-X-Out CV (x>20 held out)

| Model | RMSE (high-x) | Coverage | Overconfident? |
|-------|---------------|----------|----------------|
| **Log** | 0.30 | 60% | Yes |
| **Power** | 0.28 | 65% | Yes |
| **Asymptotic** | 0.25 | 70% | Moderate |

**Expected:** All models struggle with sparse high-x region

---

## Failure Mode Summary

| Model | Most Likely Failure | Detection | Response |
|-------|---------------------|-----------|----------|
| **Log** | Systematic residual pattern | Residual plot U-shape | → Try Model 2 |
| **Power** | Parameter non-identification | High β₁-β₂ correlation | → Revert to Model 1 |
| **Asymptotic** | K unbounded posterior | Wide K posterior | → Revert to Model 1 or 2 |

---

## Final Recommendation

### Default Strategy (80% of cases)

1. **Fit Model 1** (Logarithmic)
2. **Check all diagnostics**
3. **If adequate → STOP and report**
4. **If not → Fit Model 2 and compare**

### Expected Outcome

**Most likely:** Model 1 adequate, simple, interpretable, fast

**Probability breakdown:**
- 80%: Model 1 sufficient
- 15%: Model 2 needed
- 5%: Model 3 or alternative needed

### Selection Hierarchy

```
Parsimony > Interpretability > Predictive Accuracy > Complexity
```

**Translation:** Choose simplest model that passes all checks, even if more complex model has slightly better LOO-CV (ΔELPD < 3).

---

## Quick Checklist for Model Selection

- [ ] Fitted Model 1 (Log)
- [ ] Model 1 converged (R-hat < 1.01)
- [ ] Model 1 residuals OK
- [ ] Model 1 PPC passed (>90%)
- [ ] Model 1 LOO OK (<20% k>0.7)

**If all checked → DONE, use Model 1**

**If any failed:**
- [ ] Identified failure mode
- [ ] Fitted appropriate alternative (Model 2 or 3)
- [ ] Compared LOO-CV (ΔELPD)
- [ ] Selected based on parsimony + fit

---

*This matrix designed for quick reference during model selection process*

**Designer 1 - Parsimonious & Interpretable Approach**
