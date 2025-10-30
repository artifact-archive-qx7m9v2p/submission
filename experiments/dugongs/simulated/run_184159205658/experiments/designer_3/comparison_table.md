# Quick Comparison Table: Three Bayesian Models

## At-a-Glance Model Comparison

| Feature | B-Spline (PRIMARY) | Gaussian Process (ALT-1) | Horseshoe Polynomial (ALT-2) |
|---------|-------------------|------------------------|---------------------------|
| **Philosophy** | Structured flexibility | Nonparametric smoothness | Variable selection |
| **Basis** | 9 cubic B-splines | Infinite (via kernel) | Monomials up to degree 6 |
| **Regularization** | Hierarchical shrinkage | Kernel smoothness | Horseshoe prior |
| **Priors (key)** | tau_global ~ HC(0,0.2) | ell ~ IG(5,10) | tau ~ HC(0,0.054) |
| **Parameters** | 12 (beta_0, 9 beta_k, tau_global, sigma) | 4 (beta_0, eta², ell, sigma) | 15 (beta_0, 6 beta_j, 6 lambda_j, tau, sigma) |
| **Effective DF** | 3-5 (adaptive) | 15-20 (data-driven) | 2-4 (sparse selection) |
| **Complexity/Iter** | O(1000) - FAST | O(20,000) - SLOW | O(400) - FASTEST |
| **Expected Time** | 2-5 min | 5-10 min | 3-7 min |
| **Extrapolation** | Linear from boundary | Reverts to prior mean | DIVERGES (unreliable) |
| **Interpretability** | Medium | Low | Medium |
| **Key Strength** | Optimal balance | Best uncertainty | Efficient selection |
| **Key Weakness** | Knot placement | Computational cost | Poor extrapolation |
| **Ranking** | 1st | 2nd | 3rd |

---

## Detailed Feature Comparison

### Likelihood and Variance

| Feature | All Models (Initial) | Backup Options |
|---------|---------------------|----------------|
| Likelihood | Y ~ Normal(mu, sigma) | Student-t for robustness |
| Variance | Constant sigma | log(sigma_i) = gamma_0 + gamma_1*x_i |
| Justification | EDA: normal residuals, constant variance | If heteroscedasticity emerges |

---

## Prior Specifications Side-by-Side

### Intercept (beta_0)

| Model | Prior | Justification |
|-------|-------|---------------|
| Spline | Normal(2.3, 0.5²) | Wider to allow GP to fit all variation |
| GP | Normal(2.3, 0.3²) | Narrower, GP handles structure |
| Polynomial | Normal(2.3, 0.5²) | Same as spline |

**All centered at observed mean(Y) = 2.32**

### Flexibility/Regularization Parameters

| Model | Parameter | Prior | Interpretation |
|-------|-----------|-------|----------------|
| Spline | tau_global | HC(0, 0.2) | Global shrinkage strength |
| Spline | tau_k | HC(0, 1) | Local shrinkage per basis |
| GP | ell | IG(5, 10) | Length scale (mode=1.67, mean=2.5) |
| GP | eta² | HN(0, 0.5²) | Marginal GP variance |
| Polynomial | tau | HC(0, 0.054) | Global shrinkage (very strong) |
| Polynomial | lambda_j | HC(0, 1) | Local shrinkage per degree |

### Noise Parameter (sigma)

| Model | Prior | Justification |
|-------|-------|---------------|
| Spline | HN(0, 0.3²) | Based on SD(Y)=0.28, allows space for residual |
| GP | HN(0, 0.2²) | Smaller, GP absorbs more signal |
| Polynomial | HN(0, 0.3²) | Same as spline |

---

## Success Criteria Comparison

| Criterion | Spline | GP | Polynomial |
|-----------|--------|----|-----------|
| **Convergence** | R-hat < 1.01, ESS > 400 | R-hat < 1.01, ESS > 300 | R-hat < 1.01, ESS > 400 |
| **Regularization** | 3-4 tau_k shrunk (<0.05) | ell in [3, 15] | 3-4 lambda_j near 0 |
| **LOO-CV** | ELPD > log baseline | ELPD > log baseline | ELPD > log baseline |
| **Pareto-k** | All < 0.7 | All < 0.7 | All < 0.7 |
| **Predictive** | Data in 95% intervals | Data in 95% intervals | Data in 95% intervals |
| **Residuals** | No patterns | No patterns | No patterns |

---

## Failure Criteria Comparison

| Red Flag | Spline Response | GP Response | Polynomial Response |
|----------|----------------|-------------|---------------------|
| **No regularization** | All tau_k similar → Abandon | - | All lambda_j similar → Abandon |
| **Overfitting** | Oscillations → Reduce knots | ell < 0.5 → Abandon | Runge → Abandon |
| **Underfitting** | Residual patterns → Add knots | ell > 50 → Abandon | All beta shrunk → Increase tau_0 |
| **Computational** | Divergences → Reparameterize | ESS < 100 → Variational | Divergences → Non-centered |
| **Predictive** | LOO worse → Abandon | LOO worse → Abandon | LOO worse → Abandon |

---

## Extrapolation Behavior (x > 31.5)

| Model | Behavior | Reliability | Use Case |
|-------|----------|-------------|----------|
| **Spline** | Linear from last segment | MEDIUM | Short-term extrapolation acceptable |
| **GP** | Reverts to beta_0 = 2.3 | HIGH | Honest uncertainty, conservative |
| **Polynomial** | Diverges (up or down) | VERY LOW | AVOID extrapolation |

**Critical Note:** Only GP provides theoretically sound extrapolation. Spline is pragmatic. Polynomial is dangerous.

---

## Expected Posterior Estimates

### Intercept

| Model | Expected E[beta_0 \| Y] | Rationale |
|-------|------------------------|-----------|
| Spline | 2.25-2.35 | Close to prior, well-identified |
| GP | 2.25-2.35 | Same, GP structure independent |
| Polynomial | 2.25-2.35 | Same, standardization preserves |

### Flexibility Parameters

| Model | Parameter | Expected Posterior | Interpretation |
|-------|-----------|-------------------|----------------|
| Spline | tau_global | 0.1-0.3 | Data determines smoothness |
| Spline | sum(\|beta_k\|) | 0.3-0.5 | Total coefficient shrinkage |
| GP | ell | 3-10 | Moderate smoothness scale |
| GP | eta² | 0.05-0.15 | GP explains moderate variance |
| Polynomial | Active betas | 2-4 coefficients | Quadratic or cubic likely |
| Polynomial | Selected degree | 2 or 3 | Matches EDA quadratic finding |

### Residual Variance

| Model | Expected E[sigma \| Y] | Comparison to Prior |
|-------|----------------------|---------------------|
| Spline | 0.15-0.20 | Below prior mode (good fit) |
| GP | 0.10-0.18 | Even lower (GP absorbs more) |
| Polynomial | 0.15-0.20 | Similar to spline |

---

## Computational Resource Requirements

| Resource | Spline | GP | Polynomial |
|----------|--------|----|-----------|
| **Warmup** | 1000 iter | 1000 iter | 2000 iter (horseshoe slow) |
| **Sampling** | 2000 iter | 1000 iter | 2000 iter |
| **Chains** | 4 | 4 | 4 |
| **Time/Chain** | ~1 min | ~2-3 min | ~1-2 min |
| **Total Time** | 2-5 min | 5-10 min | 3-7 min |
| **Memory** | Low (243 floats) | Medium (729 floats) | Low (162 floats) |
| **Disk** | ~50 MB | ~30 MB | ~60 MB |

**Assumptions:** Modern CPU, no GPU. Times for N=27 only. GP scales O(N³), others O(N).

---

## Model Selection Decision Tree

```
Start
  │
  ├─> All models converge? (R-hat < 1.01)
  │     ├─> NO → Simplify (reduce knots/degree, stronger priors)
  │     └─> YES → Continue
  │
  ├─> All pass posterior predictive checks?
  │     ├─> NO → Reconsider likelihood (Student-t? Heteroscedastic?)
  │     └─> YES → Continue
  │
  ├─> Compute LOO-CV for all models
  │
  ├─> Is there a clear winner? (ELPD diff > 2*SE)
  │     ├─> YES → Use best model
  │     │         └─> Validate with extrapolation test
  │     └─> NO → Models equivalent
  │               └─> Choose simplest (Polynomial if sparse, else Spline)
  │
  ├─> Does winner pass extrapolation test?
  │     ├─> YES → Finalize and report
  │     └─> NO → Restrict inference to observed x range
  │
  └─> All models fail LOO vs log baseline?
        └─> Abandon flexible approach
              └─> Use simple logarithmic/quadratic from EDA
```

---

## Integration with Other Designers

### If Designer 1 Proposed: Parametric Models

**My contribution:** Test if flexibility is justified
- **If my models win LOO:** Complexity necessary
- **If parametric wins:** EDA was right, simpler is better

### If Designer 2 Proposed: Robust/Hierarchical Models

**Possible integration:**
- Use my spline/GP mean function with their Student-t likelihood
- Combine my flexibility with their variance modeling
- Two-stage: Their model for structure, mine for residual smoothing

### If Another Designer Proposed: Similar Flexible Models

**Comparison axes:**
- Which regularization works better? (Mine vs theirs)
- LOO-CV is ultimate arbiter
- Can combine via Bayesian model averaging

---

## Sensitivity Analysis Plan

### Prior Robustness

| Model | Parameter | Alternative Priors | Test Metric |
|-------|-----------|-------------------|-------------|
| Spline | tau_global | HC(0, 0.1), HC(0, 0.4) | KL(Post1 \|\| Post2) |
| GP | ell | HN(0, 5²), IG(3, 5) | Posterior length scale shift |
| Polynomial | tau | HC(0, 0.027), HC(0, 0.108) | Number of active terms |

**Accept if:** KL < 1 nat (minimal sensitivity)
**Investigate if:** KL > 2 nats (strong sensitivity)

### Structural Robustness

| Model | Structure Variation | Test |
|-------|-------------------|------|
| Spline | Knots: 4, 5, 6 internal | LOO-CV comparison |
| GP | Kernel: SE vs Matern-3/2 | Posterior predictive checks |
| Polynomial | Max degree: 5, 6, 8 | Selected degree stability |

---

## Posterior Predictive Checks: Test Statistics

### Location and Scale (All Models)

| Statistic | Observed Value | Acceptable p-value Range |
|-----------|---------------|-------------------------|
| mean(Y) | 2.32 | [0.05, 0.95] |
| sd(Y) | 0.28 | [0.05, 0.95] |
| max(Y) - min(Y) | 0.92 | [0.05, 0.95] |
| skewness(Y) | -0.88 | [0.01, 0.99] (less critical) |

### Model-Specific

| Model | Statistic | Acceptable Range | Failure Signal |
|-------|-----------|-----------------|----------------|
| Spline | Sign changes in f''(x) | 2-4 | >6 indicates over-wiggling |
| GP | Correlation at replicated x | High (>0.8) | <0.5 indicates noise fitting |
| Polynomial | Local extrema | 0-2 | >3 indicates Runge phenomenon |

---

## Expected LOO-CV Outcomes

### Scenario 1: Spline Wins (Most Likely)

```
Model         ELPD_loo    SE      Δ ELPD    SE(Δ)
Spline        -5.2        2.1     0.0       0.0
GP            -7.8        2.3     -2.6      1.2    (Worse)
Polynomial    -6.5        2.2     -1.3      0.8    (Slightly worse)
Log baseline  -8.9        2.4     -3.7      1.5    (Significantly worse)
```

**Interpretation:** Spline's balance wins. GP overfits slightly. Polynomial underfits. Flexibility justified vs baseline.

### Scenario 2: All Equivalent (Possible)

```
Model         ELPD_loo    SE      Δ ELPD    SE(Δ)
Spline        -6.1        2.2     0.0       0.0
GP            -6.4        2.3     -0.3      0.6    (Not significant)
Polynomial    -6.0        2.1     +0.1      0.5    (Not significant)
Log baseline  -9.2        2.5     -3.1      1.4    (Worse)
```

**Interpretation:** All flexible models converge. Choose simplest (Polynomial if degree 2-3, else Spline). All beat baseline.

### Scenario 3: Baseline Wins (Surprising)

```
Model         ELPD_loo    SE      Δ ELPD    SE(Δ)
Log baseline  -5.8        2.0     0.0       0.0
Spline        -8.2        2.3     -2.4      1.3    (Worse - overfitting)
GP            -9.1        2.4     -3.3      1.5    (Worse - overfitting)
Polynomial    -7.9        2.2     -2.1      1.2    (Worse - overfitting)
```

**Interpretation:** N=27 too small for flexibility. All overfit. Use simple log model from EDA. Complexity not justified.

---

## Decision Thresholds (Quantitative)

### Convergence
```
EXCELLENT:  R-hat < 1.001, ESS_bulk > 1000, ESS_tail > 1000
GOOD:       R-hat < 1.01,  ESS_bulk > 400,  ESS_tail > 400
ACCEPTABLE: R-hat < 1.05,  ESS_bulk > 100,  ESS_tail > 100
FAILURE:    R-hat > 1.05   (do not use)
```

### Predictive Performance
```
EXCELLENT:  ΔELPD > 5*SE (strong evidence)
GOOD:       ΔELPD > 2*SE (meaningful difference)
WEAK:       ΔELPD > 1*SE (suggestive)
EQUIVALENT: ΔELPD < 1*SE (models tied)
```

### Pareto-k Diagnostics
```
EXCELLENT:  k < 0.5  (all observations)
GOOD:       k < 0.7  (all observations)
WARNING:    k > 0.7  (1-2 observations - investigate)
FAILURE:    k > 1.0  (any observation - refit model)
```

---

## Final Recommendation Logic

```python
def select_model(loo_results, convergence_checks, extrapolation_test):
    # Filter to converged models
    converged = [m for m in models if convergence_checks[m].Rhat < 1.01]

    if len(converged) == 0:
        return "FAILURE: No models converged. Simplify or use parametric baseline."

    # Find best LOO-CV
    best_model = max(converged, key=lambda m: loo_results[m].ELPD)
    best_ELPD = loo_results[best_model].ELPD

    # Check if significantly better than baseline
    if best_ELPD < loo_results['log_baseline'].ELPD + 2*loo_results['log_baseline'].SE:
        return "Use simple logarithmic baseline. Complexity not justified."

    # Check if clear winner among flexible models
    competitors = [m for m in converged if m != best_model]
    is_clear_winner = all(
        best_ELPD > loo_results[m].ELPD + 2*loo_results[m].SE
        for m in competitors
    )

    if is_clear_winner:
        if extrapolation_test[best_model] == 'PASS':
            return f"Use {best_model}. Clear winner with good extrapolation."
        else:
            return f"Use {best_model} but restrict to observed x range."
    else:
        # Models equivalent - choose simplest
        simplicity_order = ['polynomial', 'spline', 'gp']
        for m in simplicity_order:
            if m in converged and best_ELPD - loo_results[m].ELPD < 1*loo_results[m].SE:
                return f"Use {m}. Equivalent models, choosing simplest."

    return f"Use {best_model} (default)."
```

---

## Summary Statistics to Report

### For Winning Model

**Parameter Estimates:**
```
Parameter    Mean    SD      2.5%    97.5%   R-hat   ESS
beta_0       X.XX    X.XX    X.XX    X.XX    1.000   XXXX
sigma        X.XX    X.XX    X.XX    X.XX    1.000   XXXX
[flexibility parameters]
```

**Model Comparison:**
```
Model         ELPD_loo    SE      Pareto-k (max)
Winner        X.X         X.X     0.XX
Second        X.X         X.X     0.XX
Baseline      X.X         X.X     0.XX
```

**Posterior Predictive:**
```
Test Statistic    Observed    Posterior Mean    p-value
mean(Y)          2.32        2.XX              0.XX
sd(Y)            0.28        0.XX              0.XX
range(Y)         0.92        0.XX              0.XX
```

---

This comparison table provides quick lookup for all critical model features, decisions, and expected outcomes.
