# Model Comparison Table: Designer 3

## At-a-Glance Comparison

| Aspect | Model 1: Changepoint | Model 2: Gaussian Process | Model 3: State-Space |
|--------|----------------------|---------------------------|----------------------|
| **Scientific Hypothesis** | Discrete regime shift | Smooth acceleration | Latent process evolution |
| **Primary Test** | Is there a structural break? | Is growth smooth? | Is there signal vs noise? |
| **EDA Support** | CUSUM shows break at 0.3 | Polynomials fit well (R²=0.96) | ACF = 0.989 (strong autocorr) |
| **Complexity** | Low (6-7 params) | Medium (8-10 params) | High (40+ params) |
| **Interpretability** | ★★★★★ High | ★★★☆☆ Medium | ★★☆☆☆ Low |
| **Computational Cost** | Low (~10 min) | Medium (~20 min) | Medium-High (~15 min) |
| **Convergence Risk** | Low | Medium | High |
| **Overfitting Risk** | Low | Medium | High |
| **Priority** | 1 (Highest) | 2 | 3 |

---

## Model Architecture

### Model 1: Hierarchical Changepoint
```
C_t ~ NegBin(mu_t, phi_regime[t])

log(mu_t) = {
  beta0_1 + beta1_1 * year           if year < tau
  beta0_2 + beta1_2 * (year - tau)   if year >= tau
}
```
**Parameters:** tau, beta0_1, beta1_1, beta0_2, beta1_2, phi_1, phi_2 (7 total)

---

### Model 2: Gaussian Process
```
C_t ~ NegBin(mu_t, phi)

log(mu_t) = beta0 + beta1*year + beta2*year² + f(year)

f ~ GP(0, k(year, year'))
k = alpha² * exp(-rho² * distance²)
```
**Parameters:** beta0, beta1, beta2, alpha, rho, phi, f[1:40] (46 total, but f integrated out)

---

### Model 3: Latent State-Space
```
C_t ~ NegBin(exp(theta_t), phi)

State evolution:
theta_t = theta_{t-1} + gamma + w_t
w_t ~ Normal(0, sigma_w²)
```
**Parameters:** mu_0, sigma_0, gamma, sigma_w, phi, theta[1:40] (45 total)

---

## Strength/Weakness Analysis

### Model 1 Strengths
- ✓ Directly tests strongest EDA finding (changepoint at 0.3)
- ✓ Most interpretable: "Growth rate increased from β₁ to β₂ at time τ"
- ✓ Low parameter count, low overfitting risk
- ✓ Easy to communicate to non-statisticians
- ✓ Fast computation

### Model 1 Weaknesses
- ✗ Assumes discrete break (may be too rigid)
- ✗ Will fail if growth is actually smooth
- ✗ Doesn't directly model autocorrelation
- ✗ May generate unrealistic discontinuities in predictions

---

### Model 2 Strengths
- ✓ No need to assume functional form (data-driven)
- ✓ Naturally handles uncertainty about shape
- ✓ Can capture smooth acceleration without assumptions
- ✓ Autocorrelation emerges from kernel
- ✓ Can test smoothness (via length scale rho)

### Model 2 Weaknesses
- ✗ Computationally intensive (N×N covariance matrix)
- ✗ Hyperparameters may be hard to identify with N=40
- ✗ Less interpretable than parametric models
- ✗ Risk of overfitting despite flexibility
- ✗ May struggle if true discontinuity exists

---

### Model 3 Strengths
- ✓ Best theoretical model for ACF = 0.989 structure
- ✓ Separates measurement error from process evolution
- ✓ Natural for non-stationary I(1) processes
- ✓ Can make one-step-ahead predictions easily
- ✓ Scientifically meaningful (latent growth potential)

### Model 3 Weaknesses
- ✗ Most complex model (many latent states)
- ✗ Identification issues (sigma_w vs phi)
- ✗ High computational cost
- ✗ Latent state might just be smoothed observed data
- ✗ Hardest to explain and validate

---

## Falsification Criteria Summary

### Model 1 - Abandon If:
1. ❌ Posterior for tau is diffuse (SD > 1.0)
2. ❌ Regimes not significantly different (beta1_1 ≈ beta1_2)
3. ❌ LOO worse than smooth models by > 10 ELPD
4. ❌ Posterior predictive checks show unrealistic discontinuities

### Model 2 - Abandon If:
1. ❌ Posterior shows discontinuity at consistent locations
2. ❌ Length scale rho not identified (posterior ≈ prior)
3. ❌ GP reduces to simple parametric form (f ≈ trend)
4. ❌ LOO worse than changepoint by > 10 ELPD

### Model 3 - Abandon If:
1. ❌ Latent state is just smoothed data (corr > 0.99)
2. ❌ sigma_w not identified (posterior ≈ prior)
3. ❌ No better than simple AR(1) model
4. ❌ Computational failure (Rhat > 1.05, divergences > 10%)

---

## Expected Parameter Values

| Parameter | Model 1 | Model 2 | Model 3 | Interpretation |
|-----------|---------|---------|---------|----------------|
| tau | [0.0, 0.5] | - | - | Changepoint location |
| beta1_1 | [0.5, 1.5] | - | - | Pre-change growth rate |
| beta1_2 | [1.0, 2.0] | - | - | Post-change growth rate |
| beta2 | - | [-0.5, 0.5] | - | Quadratic curvature |
| alpha | - | [0.3, 1.5] | - | GP amplitude |
| rho | - | [0.5, 2.0] | - | GP length scale |
| gamma | - | - | [0.1, 0.2] | State drift |
| sigma_w | - | - | [0.05, 0.3] | State innovation SD |
| phi | [5, 50] | [5, 50] | [5, 50] | Overdispersion (all models) |

**Red flag:** If any parameter is outside these ranges, investigate model misspecification!

---

## Posterior Predictive Checks

### Required Tests for All Models

1. **Mean and Variance**
   - Does replicated data have similar mean and variance?
   - p-value should be in [0.05, 0.95]

2. **Autocorrelation**
   - Does replicated ACF(1) match observed (0.989)?
   - Critical test: All models must capture this!

3. **Growth Pattern**
   - Does replicated data show similar 8.45× increase?
   - Check early vs late period means

4. **Dispersion**
   - Does replicated Variance/Mean ratio match ~68?
   - Test of Negative Binomial adequacy

### Model-Specific Tests

**Model 1:** Generate data should show sharp transition at estimated tau

**Model 2:** Generated data should be smooth without jumps

**Model 3:** One-step-ahead predictions should be excellent (use theta_{t-1})

---

## LOO-CV Interpretation Guide

### ELPD Differences

| Δ ELPD | Interpretation | Action |
|--------|----------------|--------|
| > 10 | Decisive | Choose better model |
| 4-10 | Strong evidence | Prefer better model |
| 2-4 | Weak evidence | Consider both models |
| < 2 | Equivalent | Choose simpler/more interpretable |

### Pareto k Diagnostics

| Pareto k | Interpretation | Action |
|----------|----------------|--------|
| < 0.5 | Good | No issues |
| 0.5-0.7 | OK | Monitor |
| 0.7-1.0 | Bad | Influential points, investigate |
| > 1.0 | Very bad | LOO unreliable, use K-fold CV |

---

## Decision Tree

```
Start: Fit all three models
│
├─► All fail to converge?
│   └─► YES → Simplify to parametric NB GLM, or use frequentist methods
│   └─► NO → Continue
│
├─► Check LOO-ELPD differences
│   ├─► One model clearly best (Δ > 10)?
│   │   └─► YES → Verify with posterior predictive checks
│   │   └─► NO → All models equivalent (Δ < 4)
│   │       └─► Choose simplest: Model 1
│
├─► Does best model pass all falsification tests?
│   └─► YES → Final model found!
│   └─► NO → Reject and try next best
│
└─► Does any model pass all tests?
    └─► NO → None of these model classes work
        └─► Try alternative approaches (see backup plans)
```

---

## Contingency Plans

### If Model 1 Wins But Has Issues
- Try smooth transition variant (sigmoid instead of step)
- Test sensitivity to tau prior
- Consider multiple changepoints

### If Model 2 Wins But Has Issues
- Simplify to parametric cubic polynomial
- Try different kernels (Matern 3/2)
- Use sparse GP with inducing points

### If Model 3 Wins But Has Issues
- Simplify to deterministic trend + AR(1)
- Use stronger priors on sigma_w
- Try simpler random walk (drop time-varying trend)

### If All Three Fail
1. **Simplify:** NB GLM with quadratic trend (no GP, no latent states)
2. **Frequentist:** GEE with AR(1) correlation structure
3. **Different distribution:** Try Generalized Poisson or Poisson-lognormal
4. **Accept limitations:** N=40 may be too small for complex models

---

## Timeline and Resource Allocation

### Day 1: Core Implementation
- Hour 1-3: Model 1 (changepoint)
- Hour 4-7: Model 2 (GP)
- Hour 8: Initial LOO comparison

### Day 2: Advanced Model and Refinement
- Hour 1-5: Model 3 (state-space)
- Hour 6-8: Diagnose convergence issues

### Day 3: Comparison and Selection
- Hour 1-2: Comprehensive LOO-CV
- Hour 3-4: Posterior predictive checks
- Hour 5-6: Falsification tests
- Hour 7-8: Sensitivity analysis

### Day 4: Finalization
- Hour 1-2: Refine winning model
- Hour 3-4: Generate visualizations
- Hour 5-6: Document results
- Hour 7-8: Buffer for unexpected issues

**Total: ~28 hours spread over 4 days**

---

## Key Insights and Recommendations

### For Modeler 1 (Baseline/Observational)
Your simple models will establish the baseline. Focus on:
- NB GLM with quadratic trend
- AR(1) correction for autocorrelation
- These should be competitive!

### For Modeler 2 (Regularization/Sparsity)
Compare against my Model 2 (GP):
- Your horseshoe prior on polynomials vs my GP
- Your approach may win on interpretability
- My GP may win on flexibility

### For Modeler 3 (That's Me!)
Focus areas:
1. **Changepoint identification** - Is it real or artifact?
2. **Smooth vs discrete** - Which story fits better?
3. **Latent structure** - Is there signal extraction opportunity?

### Critical Success Factor
**The goal is not to "win" but to discover which hypothesis best explains the data.** If all three model classes fail, that's valuable scientific information!

---

## Final Checklist

Before declaring a model as final:

- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 100 for key parameters
- [ ] Divergent transitions < 1%
- [ ] LOO successful (Pareto k < 0.7)
- [ ] Passes posterior predictive checks (all p-values in [0.05, 0.95])
- [ ] Passes model-specific falsification tests
- [ ] Robust to prior specifications (sensitivity analysis)
- [ ] Predictions are scientifically plausible
- [ ] Results are interpretable and actionable
- [ ] Code is documented and reproducible

**Only when ALL boxes are checked: Model is ready for scientific use!**
