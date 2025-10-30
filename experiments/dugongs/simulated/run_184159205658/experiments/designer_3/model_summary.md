# Quick Reference: Designer 3 Model Proposals

## Three Distinct Bayesian Approaches for Y vs x

---

## Model 1: B-Spline with Hierarchical Shrinkage (PRIMARY)

**Core Idea:** Flexible piecewise polynomial with automatic smoothing

**Key Specification:**
```
Y_i ~ Normal(mu_i, sigma)
mu_i = beta_0 + sum(beta_k * B_k(x_i))
beta_k ~ Normal(0, tau_k)
tau_k ~ HalfCauchy(0, tau_global)
```

**Strengths:**
- Local adaptation without global constraints
- Automatic smoothing via hierarchical priors
- Reasonable extrapolation (linear beyond boundary)
- Computational efficiency

**Weaknesses:**
- Requires knot placement decisions
- Less intuitive than parametric forms
- May be too flexible for N=27

**Abandon if:** No shrinkage occurs, oscillations appear, worse than log baseline

---

## Model 2: Gaussian Process with SE Kernel (ALTERNATIVE 1)

**Core Idea:** Nonparametric function with learned smoothness

**Key Specification:**
```
Y_i ~ Normal(f(x_i), sigma)
f ~ GP(beta_0, k(x,x'))
k(x,x') = eta^2 * exp(-(x-x')^2 / (2*ell^2))
ell ~ InverseGamma(5, 10)
```

**Strengths:**
- Superior uncertainty quantification
- No basis/knot decisions
- Learns local smoothness from data
- Principled extrapolation (reverts to prior mean)

**Weaknesses:**
- Computationally expensive (O(N^3))
- May be over-flexible for N=27
- Length scale prior sensitivity
- Less interpretable

**Abandon if:** Length scale <0.5 or >50, computational failure, no gain vs parametric

---

## Model 3: Horseshoe Polynomial (ALTERNATIVE 2)

**Core Idea:** Variable selection over polynomial degrees

**Key Specification:**
```
Y_i ~ Normal(mu_i, sigma)
mu_i = beta_0 + sum(beta_j * x_std^j)  for j=1..6
beta_j ~ Normal(0, lambda_j * tau)
lambda_j ~ HalfCauchy(0, 1)
tau ~ HalfCauchy(0, 0.054)
```

**Strengths:**
- Automatic degree selection
- Efficient computation
- Global smooth approximation
- EDA suggested quadratic works well

**Weaknesses:**
- Assumes polynomial form
- TERRIBLE extrapolation behavior
- Numerical instability risk
- Runge phenomenon possible

**Abandon if:** Extreme oscillations, no sparsity, divergent extrapolation, all coefficients zero

---

## Prior Specifications Summary

| Parameter | Spline | GP | Polynomial |
|-----------|--------|----|-----------|
| Intercept | N(2.3, 0.5) | N(2.3, 0.3) | N(2.3, 0.5) |
| Flexibility | tau_global ~ HC(0, 0.2) | ell ~ IG(5, 10) | tau ~ HC(0, 0.054) |
| Signal | tau_k ~ HC(0, 1) | eta^2 ~ HN(0.5) | lambda_j ~ HC(0, 1) |
| Noise | sigma ~ HN(0.3) | sigma ~ HN(0.2) | sigma ~ HN(0.3) |

**Key:** HC=HalfCauchy, HN=HalfNormal, N=Normal, IG=InverseGamma

---

## Comparison Strategy

### Step 1: Convergence
- R-hat < 1.01 for all parameters
- ESS > 400 (or >300 for GP)
- No excessive divergences (<5%)

### Step 2: Posterior Predictive Checks
- Visual: data within posterior predictive envelope
- Quantitative: test statistics match observed values

### Step 3: LOO-CV
- Compare ELPD_loo across models
- Check Pareto-k diagnostics (<0.7)
- Difference > 2*SE is meaningful

### Step 4: Residual Analysis
- No remaining systematic patterns
- Homoscedasticity maintained
- Normality preserved

### Step 5: Extrapolation Test
- Predict for x in [32, 35]
- Which model gives plausible values?
- Which has appropriate uncertainty?

---

## Expected Ranking (Most to Least Likely to Win)

1. **Spline** - Optimal balance for N=27
2. **GP** - If uncertainty quantification is paramount
3. **Polynomial** - If relationship is truly low-order polynomial

---

## Universal Red Flags (Apply to All Models)

**FLAG:** Posterior predictive failure
- **Action:** Switch to Student-t likelihood

**FLAG:** Heteroscedasticity emerges
- **Action:** Add variance model sigma(x)

**FLAG:** All models overfit identically
- **Action:** Use simple log/quadratic from EDA

**FLAG:** All models underfit identically
- **Action:** Consider mixture, change-point, or saturation models

**FLAG:** Extreme prior sensitivity
- **Action:** Use simpler model where data dominates

---

## Implementation Checklist

### Before Fitting
- [ ] Standardize x for polynomial model
- [ ] Create B-spline basis with quantile knots
- [ ] Prepare GP covariance function
- [ ] Set up proper sampler settings (chains=4, target_accept=0.95+)

### During Fitting
- [ ] Monitor convergence diagnostics in real-time
- [ ] Check for divergent transitions
- [ ] Verify ESS accumulation rate
- [ ] Watch for warnings about tail ESS

### After Fitting
- [ ] R-hat < 1.01 for all parameters
- [ ] Trace plots show good mixing
- [ ] Posterior predictive checks pass
- [ ] LOO-CV computed successfully
- [ ] Residuals examined for patterns

### Model Comparison
- [ ] All models converged
- [ ] LOO table created
- [ ] Posterior predictions plotted together
- [ ] Extrapolation behavior compared
- [ ] Decision documented with rationale

---

## Key Mathematical Insight

All three models can be viewed as regularized linear regression in different bases:

- **Spline:** Linear in B-spline basis {B_k(x)}
- **GP:** Linear in infinite basis (Mercer's theorem) with covariance-induced weights
- **Polynomial:** Linear in monomial basis {1, x, x^2, ...}

Differences are in:
1. **Basis choice:** Local vs global support
2. **Regularization:** Hierarchical vs kernel smoothness vs horseshoe
3. **Extrapolation:** Linear vs mean-reverting vs divergent

---

## Success Metrics

**Minimum Acceptable:**
- R-hat < 1.01
- LOO-ELPD > simple logarithmic baseline
- No systematic residual patterns
- Credible intervals contain ~95% of held-out data

**Good Performance:**
- Above + model selection yields sparse solution (spline/polynomial) or reasonable length scale (GP)
- Extrapolation behavior is plausible
- Interpretable to stakeholders

**Excellent Performance:**
- Above + clear winner in LOO-CV (delta ELPD > 2*SE)
- Posterior predictive p-values in [0.05, 0.95] for all test statistics
- Model explains why EDA patterns emerged

---

## Philosophical Note

These models embody the **falsification mindset:**

- Each has explicit **failure criteria** (abandon if X)
- Each has **stress tests** (extrapolation, knot sensitivity, prior robustness)
- Comparison is designed to **reveal weaknesses**, not confirm biases
- If all fail, that's **valuable information** pointing to model class error

The goal is not to fit the most complex model, but to find the **simplest model that genuinely captures the data generation process**.

If a simple logarithmic model from EDA outperforms all three sophisticated approaches, **that is the right answer**, not a failure of method.

---

## Contact Points with Other Designers

If other designers proposed:
- **Parametric models** (log, quadratic): My models should nest or outperform these
- **Other flexible models** (BART, neural nets): Compare regularization strategies
- **Hierarchical models:** Could combine with my mean structures
- **Robust models:** Student-t is compatible with all my mean functions

**Integration opportunity:** Use my spline as mean function in another designer's heteroscedastic/robust extension.

---

## Files Created

- `/workspace/experiments/designer_3/proposed_models.md` - Full detailed specification
- `/workspace/experiments/designer_3/model_summary.md` - This quick reference

**Next steps:** Implement models, compare, document results.
