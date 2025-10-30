# Model Comparison Table
## Designer 3: Robust & Alternative Approaches

---

| Aspect | Model 1: Student-t Log | Model 2: Mixture Regimes | Model 3: Hierarchical Variance |
|--------|------------------------|--------------------------|--------------------------------|
| **Core Idea** | Heavy-tailed likelihood | Two distinct populations | Spatial variance structure |
| **Parameters** | 4 (beta0, beta1, sigma, nu) | 7 (alpha1, beta1, sigma1, alpha2, beta2, sigma2, gamma) | 29+ (beta0, beta1, eta, zeta, tau, sigma_i) |
| **Complexity** | LOW | MEDIUM | HIGH |
| **Priority** | HIGHEST | MEDIUM | LOWER |
| | | | |
| **Mean Function** | beta_0 + beta_1*log(x) | Piecewise linear, soft transition | beta_0 + beta_1*log(x) |
| **Likelihood** | StudentT(nu, mu, sigma) | Mixture of 2 Normals | Normal(mu, sigma_i) |
| **Variance** | Constant (sigma) | Two constants (sigma1, sigma2) | Heterogeneous (sigma_i per obs) |
| | | | |
| **Outlier Handling** | Automatic downweighting via nu | Regime misclassification | High local variance |
| **Regime Structure** | Smooth curve (no regimes) | Explicit two regimes | Smooth curve (no regimes) |
| **Interpretability** | HIGH (simple, robust log) | MEDIUM (regime membership) | LOWER (many parameters) |
| | | | |
| **Strengths** | Simple, principled robustness | Tests regime hypothesis | Tests variance assumption |
| **Weaknesses** | Doesn't capture regime shift | 7 parameters with n=27 | Risk of overfitting |
| **Best For** | Outlier is noise/error | Regime shift is real | Variance changes with x |
| | | | |
| **Key Parameter** | nu (degrees of freedom) | gamma_1 (regime transition) | zeta (variance trend) |
| **Falsify If** | nu > 30 (Normal sufficient) | beta1 ≈ beta2 (no regimes) | zeta ≈ 0, tau ≈ 0 |
| **Expected nu/gamma/zeta** | 5-15 (moderately robust) | gamma_1 < 0 (x ↑ → plateau) | zeta > 0 (variance ↑ with x) |
| | | | |
| **Computational** | Fast (~30 sec) | Slower (~2-3 min) | Slowest (~5 min) |
| **Convergence** | Easy (well-behaved) | Moderate (label switching risk) | Harder (many parameters) |
| **Stan Difficulty** | LOW | MEDIUM | HIGH |
| | | | |
| **Scientific Question** | Is outlier just noise? | Are regimes real? | Is variance constant? |
| **If Wins, Implies** | Smooth saturation + noise | True biological regimes | Heteroscedasticity present |
| **If Loses, Implies** | Need regime structure | Smooth curve adequate | Constant variance OK |
| | | | |
| **Expected Outcome** | MOST LIKELY WINNER | Win if regimes real | Probably unnecessary |
| **Confidence** | 60% will win | 30% will win | 10% will win |

---

## Decision Tree

```
START: Fit all three models
│
├─> Check MCMC Diagnostics
│   ├─> ALL PASS → Continue to comparison
│   └─> ANY FAIL → Debug/reparameterize or pivot to backup
│
├─> Compare LOO-CV
│   ├─> Model 1 wins (ELPD > 2 SE) →
│   │   └─> Check nu: if >30, refit with Normal; else report Student-t
│   │
│   ├─> Model 2 wins (ELPD > 2 SE) →
│   │   └─> Examine regimes: clear separation? Report as real structure
│   │
│   ├─> Model 3 wins (ELPD > 2 SE) →
│   │   └─> Check zeta: variance trend real? Report heteroscedasticity
│   │
│   └─> Similar (within 2 SE) → Bayesian Model Averaging
│
└─> If ALL perform poorly → Pivot to Backup Models
    ├─> Gaussian Process (non-parametric)
    ├─> Non-monotonic polynomial
    └─> Changepoint + Student-t hybrid
```

---

## Prior Summary

| Parameter | Model 1 | Model 2 | Model 3 | Rationale |
|-----------|---------|---------|---------|-----------|
| **Intercept** | N(2.3, 0.5) | α1~N(1.7,0.3), α2~N(2.2,0.3) | N(2.3, 0.5) | Centered at observed mean |
| **Slope** | N(0.29, 0.15) | β1~N(0.11,0.05), β2~N(0.02,0.02) | N(0.29, 0.15) | Centered at EDA estimates |
| **Scale** | Exp(10) | Exp(10) for both | η~N(-2.3,0.5) | Mean ~0.1, matches RMSE |
| **Robust/Special** | nu~Gamma(2,0.1) | γ1~N(-1,0.5) | zeta~N(0,0.2), tau~Exp(2) | Key parameters for mechanism |

---

## Posterior Predictive Checks (All Models)

### Essential Checks:

1. **Histogram**: Y_rep vs Y_obs distribution
   - Should overlap well
   - Failure: distributions very different

2. **Scatter**: Y_rep vs x with Y_obs overlay
   - Should cover observed data
   - Failure: systematic over/under-prediction

3. **Extremes**: min(Y_rep), max(Y_rep) vs observed
   - Should include observed extremes
   - Failure: predicted range too narrow/wide

4. **Outlier**: Is x=31.5 observation within posterior predictive?
   - Model 1: Check if downweighted (large residual but small influence on predictions)
   - Model 2: Check regime assignment
   - Model 3: Check sigma_31.5 vs others

5. **Residuals**: Should be patternless vs x and fitted
   - Failure: curves, trends, fanning

---

## Sensitivity Analyses (Winner Only)

| Analysis | Protocol | Criterion |
|----------|----------|-----------|
| **Prior Sensitivity** | Refit with 2× wider priors | Posterior means should not change >20% |
| **Outlier Sensitivity** | Refit without x=31.5 | Key parameters should change <30% |
| **Data Subset** | Fit to x<20 only, predict x>20 | Predictions should include observed x>20 values |
| **Initialization** | Refit with different initial values | Should converge to same posterior |

---

## Reporting Checklist

### For Chosen Model:

- [ ] Posterior summaries (mean, SD, 95% CI for all parameters)
- [ ] MCMC diagnostics (R-hat, ESS, divergences, trace plots)
- [ ] LOO-CV comparison table with SE
- [ ] Posterior predictive plots (at least 4)
- [ ] Residual plots (vs x, vs fitted)
- [ ] Sensitivity analysis results
- [ ] Scientific interpretation
- [ ] Uncertainty on predictions at x = 1, 5, 10, 15, 20, 30

### Additional for Model 1 (if wins):
- [ ] nu posterior distribution plot
- [ ] Comparison to Normal likelihood (justify robustness)
- [ ] Influence of x=31.5 observation

### Additional for Model 2 (if wins):
- [ ] Regime probability plot vs x
- [ ] Regime assignments for each observation
- [ ] Transition point (where p(regime1)=0.5)
- [ ] Compare to EDA changepoint at x=7

### Additional for Model 3 (if wins):
- [ ] sigma_i vs x plot
- [ ] zeta and tau posteriors
- [ ] Variance structure interpretation

---

## Quick Reference: When to Use Each Model

**Use Model 1 when:**
- Outlier is suspected measurement error or heavy-tailed noise
- Want simple, interpretable model with automatic robustness
- EDA shows smooth pattern with no clear regime shift
- **Most common scenario**

**Use Model 2 when:**
- EDA changepoint is strongly significant (as in this data: F=22.4, p<0.0001)
- Scientific domain suggests distinct mechanisms/regimes
- Want to test if sharp transition at x=7 is real
- Have enough data in each regime (we have 9 and 18 obs)

**Use Model 3 when:**
- Suspect variance is not constant (despite EDA finding)
- Outlier might be due to location-specific uncertainty
- High-leverage observations (x=31.5) in sparse regions
- **Unlikely to win, but good for testing assumptions**

---

**This table should be used for:**
- Quick model selection
- Understanding trade-offs
- Explaining to collaborators
- Making go/no-go decisions during analysis
