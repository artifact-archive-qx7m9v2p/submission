# Bayesian Model Design: Robust and Flexible Approaches
**Designer:** Model Designer #3
**Focus:** Robustness to outliers, flexible functional forms, model misspecification protection
**Date:** 2025-10-27

---

## Executive Summary

With n=27 observations and evidence of non-linearity, potential outliers (x=31.5), and structural uncertainty (change point at x≈7), I propose **three distinct Bayesian model classes** that prioritize robustness and adaptability:

1. **Robust Logarithmic Regression** (Student-t errors) - Primary recommendation
2. **Adaptive B-Spline Model** (flexible smooth function) - Secondary recommendation
3. **Robust Change-Point Model** (regime-switching with heavy tails) - Exploratory

Each model is designed to handle:
- Potential outliers via heavy-tailed likelihoods
- Functional form uncertainty via flexibility
- Small sample constraints via appropriate regularization

**Critical Insight:** With only 27 observations, we walk a tightrope between flexibility (to capture true patterns) and parsimony (to avoid overfitting). Our models embed robustness features that gracefully degrade if our assumptions fail.

---

## Problem Formulation: Competing Hypotheses

### Three Fundamental Questions

1. **Is the outlier (x=31.5, Y=2.57) genuine or erroneous?**
   - Genuine → need robust estimation that down-weights it automatically
   - Erroneous → need heavy-tailed likelihood to prevent corruption

2. **Is the logarithmic functional form truly optimal, or artifact of limited model search?**
   - Optimal → simple log model sufficient
   - Artifact → need flexible non-parametric approach

3. **Is the change point at x≈7 a real structural break or smooth transition?**
   - Real break → segmented model appropriate
   - Smooth transition → continuous flexible model better

### Why Robustness Matters Here

**EDA Red Flags:**
- One extreme x value (31.5) is 2.2x the second-largest (29.0)
- Varying replicate precision (CV=1.31 across replicate variances)
- Change point evidence strong (66% RSS reduction) but n=27 small for complex models
- Left-skewed Y distribution (skewness=-0.70) suggests potential for asymmetry in errors

**Robustness Strategy:**
Rather than assume Gaussian errors and hope for the best, we:
1. Use Student-t likelihoods to auto-adapt tail behavior
2. Employ flexible functions that don't force data into parametric straightjackets
3. Regularize via priors to prevent overfitting with small n

---

## Model Class 1: Robust Logarithmic Regression (PRIMARY)

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(nu, mu_i, sigma)

Mean function:
mu_i = alpha + beta * log(x_i + c)

Priors:
alpha ~ Normal(2.0, 1.0)           # Intercept: weakly informative
beta ~ Normal(0.3, 0.5)            # Slope: positive, moderate
c ~ Gamma(2, 2)                    # Log shift: mean=1, constrained positive
nu ~ Gamma(2, 0.1)                 # df: mean=20, allows heavy tails
sigma ~ HalfCauchy(0, 0.5)         # Error scale: weakly informative
```

### Why This Model?

**Robustness Features:**
1. **Student-t likelihood** with estimated degrees of freedom (nu)
   - If nu > 30: essentially Gaussian (data support normality)
   - If nu < 10: heavy tails (data suggest outliers/non-normality)
   - Data decide automatically via posterior of nu

2. **Flexible log shift (c)** instead of fixed log(x+1)
   - EDA used log(x+1) arbitrarily
   - Optimal shift may be c=0.5 or c=2, affecting fit
   - Gamma(2,2) prior: mean=1, mode=0.5, allows c ∈ [0.1, 5]

3. **Weakly informative priors** calibrated to data scale
   - Regularizes without imposing strong beliefs
   - Critical with n=27 to prevent overfitting

**Advantages:**
- Minimal complexity: 5 parameters (vs 3 for standard log model)
- Automatic outlier down-weighting (no ad-hoc exclusions)
- Interpretable: beta = rate of Y increase per log-unit of x
- Fast computation: conjugate-like structure in Stan/PyMC
- Conservative extrapolation: log doesn't explode like polynomials

**Limitations:**
- Still assumes smooth log relationship (misses potential breakpoint)
- Log form may not be true data-generating process
- c parameter may be weakly identified (high posterior correlation with alpha/beta)

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior of nu < 5 consistently**
   - Interpretation: Extreme heavy tails needed, suggesting systematic outliers or model mis-specification
   - Action: Switch to mixture model or investigate data quality issues

2. **Posterior predictive checks show systematic residual patterns**
   - Check: Plot residuals vs fitted, residuals vs x
   - Pattern: U-shape, heteroscedasticity, or clusters
   - Action: Move to flexible spline model (Model 2)

3. **Log shift parameter c hits prior boundary** (c > 4 or c < 0.2)
   - Interpretation: Log transformation fundamentally wrong
   - Action: Try power transformation Y ~ (x+c)^lambda or switch to splines

4. **Posterior intervals for Y at x=31.5 exclude observed value**
   - Despite robust likelihood, model still rejects that observation
   - Suggests structural mis-specification, not just outlier
   - Action: Investigate change-point model or non-monotonic function

5. **LOO diagnostic: Pareto k > 0.7 for multiple observations**
   - Indicates influential points that model can't accommodate
   - Action: More flexible model needed (splines or GP)

### Expected Posterior Behavior

**If model is correct:**
- nu posterior: 10-30 (moderate tails, handling x=31.5 gracefully)
- c posterior: 0.5-2.0 (refining log shift from EDA's +1)
- beta posterior: 0.2-0.4 (positive, consistent with EDA R²=0.89)
- sigma posterior: 0.1-0.2 (smaller than marginal SD(Y)=0.27)

**Warning signs:**
- nu < 5: Model struggling with multiple outliers
- c > 3: Log transformation inappropriate
- Posterior SD(beta) > 0.3: Weak identification, need more data
- High posterior correlation |rho(c, alpha)| > 0.9: Reparameterization needed

### Computational Considerations

**Sample Size Feasibility (n=27):**
- 5 parameters well within identifiability limits
- Each parameter identified by ~5 observations (rule of thumb: OK)
- Student-t adds minimal computational cost vs Normal

**Stan Implementation:**
- Use `student_t()` likelihood (built-in)
- Parameterize by mu, sigma (not location/scale) for efficiency
- Monitor: Rhat < 1.01, ESS_bulk > 400, ESS_tail > 400
- Expect: ~0.1 sec per iteration, 4000 iterations = ~400 sec total

**Potential Issues:**
1. **High posterior correlation** between c and alpha/beta
   - Solution: Consider non-centered parameterization or fix c=1 if weakly identified

2. **Divergent transitions** if nu near boundary (nu≈1)
   - Solution: Tighten nu prior to Gamma(3, 0.15) → mean=20, SD=11.5

3. **Slow mixing** if nu posterior multimodal
   - Solution: Initialize nu=10, increase adapt_delta to 0.95

### Prior Sensitivity Analysis Plan

**Test three prior scenarios:**

1. **Baseline (as specified above)**
2. **Tighter:** alpha~N(2,0.5), beta~N(0.3,0.25), c~Gamma(4,4), nu~Gamma(4,0.2)
3. **Wider:** alpha~N(2,2), beta~N(0.3,1), c~Gamma(1,1), nu~Gamma(2,0.05)

**Accept model if:** Posterior means change < 20% across scenarios

### Stress Test: Designed to Break the Model

**Test:** Simulate data from **piecewise linear** model with sharp breakpoint at x=7, then fit robust log model.

**Prediction:**
- Model will fit reasonably (log approximates piecewise smoothly)
- But: Posterior predictive p-value will be extreme (p < 0.05 or p > 0.95)
- Residuals will show systematic pattern around x=7

**If model passes** (p ∈ [0.05, 0.95], no pattern): Log model flexible enough for this data.
**If model fails**: Change-point model (Model 3) needed.

---

## Model Class 2: Adaptive B-Spline Model (SECONDARY)

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(nu, mu_i, sigma)

Mean function (B-spline basis):
mu_i = sum_{k=1}^K beta_k * B_k(x_i)

where B_k(x) are cubic B-spline basis functions with K knots

Knot placement:
- K = 4 knots at quantiles {0.25, 0.50, 0.75} of x, plus boundaries
- Yields K+3 = 7 basis functions (cubic splines)

Priors:
beta_k ~ Normal(2.3, 0.5) for k=1 (intercept-like)
beta_k ~ Normal(0, tau) for k=2,...,7 (smoothing)
tau ~ HalfCauchy(0, 0.2)         # Global smoothness parameter
nu ~ Gamma(2, 0.1)               # Heavy tails as before
sigma ~ HalfCauchy(0, 0.5)       # Error scale
```

### Why This Model?

**Flexibility Without Over-Commitment:**
- Logarithmic model assumes specific functional form
- Splines are **universal approximators**: can represent any smooth function
- B-splines provide **local control**: each basis function affects limited x range
- Regularization via **shared smoothness parameter (tau)** prevents overfitting

**Robustness Features:**
1. **Student-t likelihood** (same as Model 1)
2. **Adaptive smoothness:** tau learned from data
   - If data truly smooth (log-like): tau → 0 (strong shrinkage)
   - If data wiggly (change point): tau → 0.5 (flexible)
3. **Knot placement at quantiles:** ensures balanced data per segment

**Advantages Over Model 1:**
- No functional form assumption (log, quadratic, etc.)
- Can capture change point smoothly without explicit regime-switching
- Can detect non-monotonic relationships (if present)
- Still interpretable: plot posterior mean function with credible bands

**Advantages Over Full Gaussian Process:**
- Much faster: 7-9 parameters vs n×n covariance matrix
- More stable with n=27: GPs need more data for reliable length-scale inference
- Easier to specify informative priors on smoothness

**Limitations:**
- More parameters: 7 betas + nu + sigma + tau = 10 total
- With n=27, ratio n/p = 2.7 (marginal, but tau provides regularization)
- Less interpretable coefficients (basis functions not meaningful)
- Extrapolation risky: splines can behave badly outside knot range

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior of tau is extremely small** (tau < 0.05 consistently)
   - Interpretation: Extreme shrinkage needed, all betas ≈ 0
   - Meaning: Data linear/log, don't need flexibility
   - Action: Revert to Model 1 (simpler)

2. **Effective parameter count (p_eff from WAIC) > 8**
   - Suggests overfitting despite regularization
   - With n=27, p_eff > 8 is excessive (using 30% of data on parameters)
   - Action: Reduce knots (K=3 instead of 4) or use Model 1

3. **Posterior mean function shows wild oscillations**
   - Check: Does posterior curve wiggle up/down multiple times?
   - Especially between data points (interpolation instability)
   - Action: Increase shrinkage (tighter tau prior) or reduce knots

4. **Model not better than Model 1 by WAIC** (ΔWAIC < 2)
   - Flexibility not buying anything
   - Complexity not justified
   - Action: Use Model 1 (simpler, more interpretable)

5. **Extrapolation to x=35 produces absurd values**
   - Posterior mean Y(35) < 1.5 or > 3.0 (outside plausible range)
   - Splines untrustworthy beyond data
   - Action: Warn against extrapolation, use Model 1 for prediction

### Knot Selection Strategy

**Why 4 internal knots (K=4)?**
- Rule of thumb: K ≈ n^(1/5) = 27^0.2 ≈ 2.4 → round to 2-4
- With K=4 + boundary knots, get K+3=7 basis functions (cubic)
- Ratio n/p_basis = 27/7 ≈ 4 (adequate with strong regularization)

**Knot locations (quantile-based):**
```
Quantile:  0%    25%   50%   75%   100%
x value:   1.0   5.0   9.5   15.5  31.5  (approximate from data)
```

**Why not adaptive knots?**
- With n=27, estimating knot locations is unstable
- Fixed quantile-based placement is robust
- If model inadequate, add/remove knots manually (model comparison)

**Alternative to test:** K=3 (5 basis functions)
- More parsimonious: n/p = 27/5 = 5.4 (safer)
- Less flexible (may miss change point detail)
- Fit both, compare via WAIC

### Expected Posterior Behavior

**If model is correct:**
- tau posterior: 0.1-0.3 (moderate smoothness, not extreme)
- nu posterior: 10-30 (similar to Model 1)
- beta_k posteriors: varying, but no single beta dominates
- Posterior mean function: smooth curve, captures log-like trend + refinements

**Warning signs:**
- tau > 0.5: Overfitting risk, model too flexible for data
- Single beta_k has |posterior mean| > 3x others: basis function issue
- High correlation between adjacent betas: knot placement problem
- nu < 5: Same issue as Model 1 (multiple outliers)

### Computational Considerations

**Sample Size Feasibility (n=27):**
- 10 parameters total: 7 betas + tau + nu + sigma
- Regularization via tau provides effective ~4-5 parameters
- Hierarchical structure (betas ~ Normal(0, tau)) stabilizes estimation

**Stan Implementation:**
- Use `bs()` function from R's `splines` package to generate basis matrix
- Pass as data: `matrix[N, K+3] B` (basis functions evaluated at x_i)
- Vectorize: `mu = B * beta` (matrix multiplication)
- Computation: ~0.2 sec per iteration (2x slower than Model 1)

**Potential Issues:**
1. **Divergent transitions** due to funnel geometry in hierarchical prior
   - Solution: Non-centered parameterization
   ```
   beta_k = beta_mean + tau * beta_raw_k
   beta_raw_k ~ Normal(0, 1)
   ```

2. **Poor identifiability** if knots poorly placed
   - Solution: Check posterior predictive: do adjacent knots get pulled to same value?
   - If yes: reduce K

3. **Extrapolation instability** beyond [1, 31.5]
   - Solution: Add linear extrapolation constraints at boundaries
   - Or: only report predictions within data range

### Prior Sensitivity Analysis Plan

**Test three tau priors:**
1. **Baseline:** tau ~ HalfCauchy(0, 0.2) (moderate shrinkage)
2. **Stronger shrinkage:** tau ~ HalfCauchy(0, 0.1) (toward simpler model)
3. **Weaker shrinkage:** tau ~ HalfCauchy(0, 0.5) (more flexibility)

**Also test knot count:**
- K=3 (5 basis functions): more parsimonious
- K=5 (8 basis functions): more flexible

**Accept model if:**
- Posterior mean function stable across tau priors (shape doesn't change drastically)
- WAIC differences between K=3,4,5 small (ΔWAIC < 4)

### Stress Test: Designed to Break the Model

**Test 1 - Overfitting Check:**
- Simulate data from **simple linear** model: Y = 1.5 + 0.03*x + noise
- Fit spline model
- **Prediction:** Model should shrink toward linear (tau small, posterior flat)
- **Failure:** If posterior shows spurious wiggles with high posterior probability

**Test 2 - Extrapolation Check:**
- Fit model to data
- Predict at x=50 (far beyond x_max=31.5)
- **Prediction:** Wide posterior intervals (uncertainty grows)
- **Failure:** If posterior mean absurd (Y > 3.5 or Y < 2.0) or intervals too narrow (overconfidence)

---

## Model Class 3: Robust Change-Point Model (EXPLORATORY)

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(nu, mu_i, sigma)

Mean function (continuous piecewise linear):
mu_i = alpha + beta_1 * min(x_i, tau) + beta_2 * max(0, x_i - tau)

Equivalently:
mu_i = alpha + beta_1 * x_i                  if x_i <= tau
mu_i = alpha + beta_1 * tau + beta_2 * (x_i - tau)   if x_i > tau

Continuity constraint automatically satisfied.

Priors:
alpha ~ Normal(1.8, 0.5)              # Y at x=0 (extrapolated)
beta_1 ~ Normal(0.15, 0.1)            # Steeper initial slope
beta_2 ~ Normal(0.02, 0.05)           # Flatter later slope
tau ~ Uniform(5, 12)                  # Change point around x=7
nu ~ Gamma(2, 0.1)                    # Heavy tails
sigma ~ HalfCauchy(0, 0.5)            # Error scale
```

### Why This Model?

**Directly Tests Change-Point Hypothesis:**
- EDA found 66% RSS improvement with breakpoint at x≈7
- But: Is this robust to outliers? Student-t tests this.
- Two-regime interpretation:
  - **Regime 1 (x ≤ tau):** Rapid increase (learning, growth, response)
  - **Regime 2 (x > tau):** Slow increase (saturation, diminishing returns)

**Robustness Features:**
1. **Student-t likelihood:** protects against outlier at x=31.5 distorting breakpoint inference
2. **Continuous formulation:** no discontinuity at tau (scientifically implausible)
3. **Uniform prior on tau:** allows data to determine location in [5,12] range

**Advantages:**
- **Interpretable:** tau has clear meaning (when does regime shift?)
- **Parsimonious:** 6 parameters (fewer than spline model)
- **Falsifiable:** tau posterior should concentrate around 7 if EDA finding robust
- **Robust:** heavy tails prevent single outlier from moving breakpoint

**Limitations:**
- **Identifiability issues:** with n=27, tau may be poorly estimated (wide posterior)
- **Model complexity:** ratio n/p = 27/6 = 4.5 (marginal)
- **Assumption:** discrete regime switch (may be gradual transition)
- **Computation:** tau discrete parameter → slower mixing than continuous parameters
- **Extrapolation:** behavior beyond tau regime depends on beta_2 (uncertain if data sparse)

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior of tau is uniform** (not concentrating)
   - Check: Does 90% credible interval span entire prior range [5,12]?
   - Interpretation: Data don't support specific breakpoint
   - Action: Use smooth model (Model 1 or 2)

2. **Posterior of beta_1 and beta_2 overlap substantially**
   - Check: Are 90% CI(beta_1) and CI(beta_2) overlapping?
   - Interpretation: No evidence for different slopes
   - Action: Revert to single-slope model (Model 1)

3. **Posterior predictive checks show discontinuity at tau**
   - Despite continuous formulation, data might force jump
   - Check: Plot posterior mean with credible bands around tau
   - If jump visible: model mis-specified

4. **WAIC worse than Model 1** (simple log model)
   - Change-point complexity not justified
   - Occam's razor: prefer simpler model
   - Action: Abandon this model class

5. **LOO cross-validation unstable** (multiple Pareto k > 0.7)
   - Points near tau are highly influential
   - Model unstable with respect to data perturbations
   - Action: Use Model 2 (smooth spline) instead

6. **Prior-posterior conflict on tau**
   - Posterior mode at prior boundary (tau ≈ 5 or tau ≈ 12)
   - Suggests prior range mis-specified or change point outside range
   - Action: Expand tau prior range or abandon if no concentration

### Expected Posterior Behavior

**If model is correct:**
- tau posterior: Mode ≈ 6-8, 90% CI = [5.5, 9.5] (reasonably tight)
- beta_1 posterior: 0.10-0.20 (steeper than beta_2)
- beta_2 posterior: 0.00-0.06 (much flatter, possibly including 0)
- nu posterior: 10-30 (same as other models)
- Clear visual break in posterior mean function at tau

**Warning signs:**
- tau posterior bimodal: suggests multiple possible breakpoints (data ambiguous)
- beta_2 posterior includes negative values: slope reversal (implausible?)
- beta_1 ≈ beta_2 posteriors: no regime difference
- nu < 5: extreme outliers dominating inference

### Computational Considerations

**Sample Size Feasibility (n=27):**
- 6 parameters: manageable
- But: tau is discrete-ish (integer x values)
- Observations per regime: ~12 in regime 1, ~15 in regime 2 (if tau≈7)
- Marginal: each regime has only ~12-15 points to estimate 2 parameters

**Stan Implementation Challenges:**

1. **Discrete parameter (tau):**
   - Stan doesn't handle discrete parameters directly
   - **Solution:** Marginalize over tau grid or use continuous approximation

2. **Marginalization approach:**
   ```stan
   // Evaluate likelihood for each possible tau in grid
   vector[N_tau_grid] log_lik_tau;
   for (i in 1:N_tau_grid) {
     tau_candidate = tau_grid[i];
     mu = alpha + beta_1 * min(x, tau_candidate) + beta_2 * max(x - tau_candidate, 0);
     log_lik_tau[i] = student_t_lpdf(Y | nu, mu, sigma);
   }
   target += log_sum_exp(log_lik_tau);  // Marginalize
   ```

3. **Computational cost:**
   - If tau_grid has 50 values, each iteration evaluates 50 likelihoods
   - Expect ~1 sec per iteration (10x slower than Model 1)
   - For 4000 iterations: ~4000 sec = 67 min (manageable)

**Alternative: Smooth Approximation**
- Replace discrete tau with continuous sigmoid transition:
  ```
  w_i = inv_logit((x_i - tau) / delta)  # transition function
  mu_i = alpha + beta_1 * x_i * (1 - w_i) + (alpha + beta_1*tau + beta_2*(x_i - tau)) * w_i
  delta ~ Gamma(2, 5)  # transition width, small = sharp
  ```
- Faster sampling (continuous parameters)
- But: more parameters (7 total with delta)

### Prior Justification

**Why Uniform(5, 12) for tau?**
- EDA suggests tau ≈ 7
- Allow data to shift ±3 units (5-10 range core)
- Extend to 12 for robustness (test if tau > 10 plausible)
- Uniform = non-informative given range

**Why beta_1 ~ N(0.15, 0.1)?**
- Regime 1: x ∈ [1, 7], Y ∈ [1.77, 2.47]
- Slope ≈ (2.47-1.77)/(7-1) = 0.70/6 ≈ 0.12
- Prior centered at 0.15 (slightly steeper), SD=0.1 allows [0.05, 0.25]

**Why beta_2 ~ N(0.02, 0.05)?**
- Regime 2: x ∈ [7, 31.5], Y ∈ [2.47, 2.72]
- Slope ≈ (2.72-2.47)/(31.5-7) = 0.25/24.5 ≈ 0.01
- Prior centered at 0.02 (slightly steeper), SD=0.05 allows [-0.03, 0.07]
- Allows for slight negative slope (saturation overshoot)

### Prior Sensitivity Analysis Plan

**Test three scenarios:**
1. **Baseline** (as above)
2. **Wider tau:** tau ~ Uniform(3, 15) (test if breakpoint really at 7)
3. **More informative slopes:** beta_1 ~ N(0.15, 0.05), beta_2 ~ N(0.02, 0.02)

**Accept model if:**
- tau posterior mode stable across scenarios (±2 units)
- Slope estimates change < 30%

### Stress Test: Designed to Break the Model

**Test:** Simulate data from **smooth logarithmic** model (no breakpoint), then fit change-point model.

**Prediction:**
- Model will find spurious breakpoint (tau posterior may be bimodal or at boundary)
- beta_1 and beta_2 will be similar (CI overlap)
- WAIC will favor simpler model (Model 1)

**Success criterion:** Model doesn't force breakpoint when none exists (tau posterior diffuse or beta_1≈beta_2).

---

## Model Comparison Strategy

### Sequential Fitting Plan

**Phase 1: Baseline (Model 1)**
1. Fit robust logarithmic regression
2. Check diagnostics (Rhat, ESS, divergences)
3. Posterior predictive checks:
   - Residuals vs fitted
   - Replicate data distribution
   - Coverage of prediction intervals
4. Examine nu posterior: Is Student-t needed?
5. Examine c posterior: Is log(x+1) optimal?

**Decision Point 1:**
- If Model 1 adequate (PPC pass, nu > 20, c ≈ 1): STOP, use this model
- If nu < 10 or systematic residual pattern: Proceed to Phase 2
- If c posterior hits boundary: Reconsider functional form

**Phase 2: Flexibility Test (Model 2)**
1. Fit adaptive B-spline model
2. Check effective parameter count (p_eff from WAIC)
3. Compare WAIC: Model 2 vs Model 1
4. Examine tau posterior: Is flexibility used?
5. Visual comparison: Do curves differ?

**Decision Point 2:**
- If ΔWAIC(M2 - M1) > 6: Strong evidence for flexibility, use Model 2
- If ΔWAIC ∈ [2, 6]: Weak preference, check interpretability needs
- If ΔWAIC < 2: Models equivalent, prefer Model 1 (simpler)
- If p_eff > 8: Overfitting, simplify (fewer knots) or use Model 1

**Phase 3: Change-Point Test (Model 3) - OPTIONAL**
1. Fit only if:
   - Model 1 residuals show clear pattern at x≈7
   - Domain knowledge supports regime interpretation
   - Interpretability of tau valuable
2. Compare WAIC: Model 3 vs Model 1 and Model 2
3. Check tau posterior: Is it concentrated?

**Decision Point 3:**
- If tau posterior diffuse: Change point not supported, revert to Model 1 or 2
- If ΔWAIC(M3 - M1) > 6: Strong evidence for regimes, use Model 3
- If beta_1 ≈ beta_2: No regime difference, use Model 1
- If Model 2 better than Model 3: Smooth transition more plausible

### Comparison Metrics

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| **WAIC** | Predictive accuracy | ΔWAIC > 6 → strong preference |
| **LOO-CV** | Robustness to outliers | Prefer if Pareto k < 0.7 for all points |
| **p_eff** | Effective parameters | Prefer if p_eff/n < 0.3 (i.e., < 8 params) |
| **Posterior predictive p-value** | Calibration | p ∈ [0.05, 0.95] → good |
| **Coverage** | Uncertainty quantification | 90% PI should cover ~90% of held-out data |

### Decision Matrix

| Scenario | Evidence | Recommended Model | Rationale |
|----------|----------|-------------------|-----------|
| nu > 20, ΔWAIC < 2 | Gaussian adequate, simple best | **Model 1** | Parsimony wins |
| nu < 10, ΔWAIC(M2-M1) > 6 | Outliers + non-log | **Model 2** | Need flexibility + robustness |
| tau concentrated, beta_1 ≠ beta_2 | Clear breakpoint | **Model 3** | Interpretable regimes |
| All WAIC similar (< 4) | Model uncertainty | **Ensemble** | Average over models |
| p_eff > 10 for M2 or M3 | Overfitting | **Model 1** | Too complex for n=27 |

### Ensemble Option (If Models Comparable)

If ΔWAIC < 4 between models:

```
Posterior predictive:
Y_new | x_new ~ sum_m P(M=m | data) * P(Y_new | x_new, M=m, data)

Model weights:
P(M=m | data) ∝ exp(-0.5 * WAIC_m)
```

**Advantages:**
- Accounts for model uncertainty
- Better calibrated prediction intervals
- Robustness to model choice

**Disadvantages:**
- Less interpretable (no single mechanism)
- More complex to communicate

---

## Red Flags and Escape Routes

### Red Flag 1: All Models Have nu < 5

**Interpretation:** Multiple extreme outliers or fundamental likelihood mis-specification.

**Potential Causes:**
1. Y distribution actually non-Gaussian (bounded, discrete, etc.)
2. Multiple problematic observations (not just x=31.5)
3. Systematic measurement error (heteroscedasticity missed by EDA)

**Escape Route:**
- **Bounded likelihood:** Beta regression if Y ∈ (0, 3) bounded
- **Mixture model:** Y_i ~ p*Normal(mu_i, sigma_1) + (1-p)*Normal(mu_i, sigma_2)
- **Heteroscedastic model:** sigma_i = exp(gamma_0 + gamma_1 * x_i)

### Red Flag 2: Posterior Predictive p-values Extreme for All Models

**Interpretation:** All functional forms wrong, missing important feature.

**Potential Causes:**
1. Non-monotonic relationship (Y increases then decreases)
2. Interaction with unmeasured variable
3. Time trend or batch effect in data collection

**Escape Route:**
- **Non-monotonic spline:** allow downturn (quadratic B-splines)
- **Gaussian Process:** no functional form assumption
- **Data audit:** check for temporal patterns, experimental conditions

### Red Flag 3: Replicate Observations Strongly Disagree with Model

**Interpretation:** Replicates show more variability than model allows.

**Example:** At x=15.5, Y ∈ {2.47, 2.65}, difference = 0.18 (large relative to sigma).

**Escape Route:**
- **Hierarchical model:**
  ```
  Y_ij ~ Normal(mu_i, sigma_within)
  mu_i ~ Normal(f(x_i), sigma_between)
  ```
- **Observation-specific variance:**
  ```
  Y_i ~ Normal(mu_i, sigma_i)
  sigma_i ~ some model (perhaps constant within replicate groups)
  ```

### Red Flag 4: LOO Shows Many Influential Points (k > 0.7)

**Interpretation:** Model unstable, highly dependent on specific observations.

**Escape Route:**
- **Importance sampling correction** for high-k points
- **K-fold CV** instead of LOO (more stable)
- **Simplify model** (fewer parameters)
- **Collect more data** (if possible)

### Red Flag 5: Posterior Predictions Absurd for Extrapolation

**Interpretation:** All models unreliable beyond data range.

**Example:** At x=50, posterior mean Y > 4 or Y < 1 (both implausible).

**Escape Route:**
- **Do not extrapolate** (communicate limits clearly)
- **Expert elicitation:** informative priors on Y_max from domain knowledge
- **Asymptotic model with strong prior on Y_max:** constrain upper limit
- **Collect data at higher x** (experimental design)

---

## Balance: Flexibility vs Parsimony

### The n=27 Constraint

**Fundamental Tension:**
- **Too simple:** Miss important patterns (e.g., change point)
- **Too complex:** Overfit noise, unstable predictions

**Rule of Thumb:**
- Observations per parameter: n/p ≥ 5 is safe
- Model 1: n/p = 27/5 = 5.4 ✓
- Model 2: n/p = 27/10 = 2.7 (marginal, needs strong regularization) ⚠
- Model 3: n/p = 27/6 = 4.5 ✓

**Our Strategy:**
1. **Start simple** (Model 1): If adequate, stop
2. **Add complexity only if justified** by WAIC improvement > 6
3. **Regularize aggressively** (hierarchical priors, shrinkage)
4. **Cross-validate** (LOO-CV) to detect overfitting

### Where Flexibility Is Most Valuable

**Diminishing Returns Analysis:**

| Model Feature | Complexity Cost | Value for This Data |
|---------------|-----------------|---------------------|
| Robust likelihood (Student-t) | +2 params | **HIGH** (outlier at x=31.5) |
| Flexible log shift (c) | +1 param | **MEDIUM** (may refine fit) |
| B-spline flexibility | +4 params | **MEDIUM** (if change point confirmed) |
| Change-point regime | +3 params | **LOW-MEDIUM** (needs validation) |
| Heteroscedastic errors | +2 params | **LOW** (EDA shows homoscedasticity) |

**Prioritization:**
1. **Must have:** Robust likelihood (Student-t for nu parameter)
2. **Nice to have:** Flexible functional form (splines) IF Model 1 inadequate
3. **Exploratory:** Change-point interpretation IF scientifically meaningful

### Minimum Complexity Needed

**Given EDA findings:**
- Strong non-linearity: Need at least log transformation or equivalent
- Potential outlier: Need robust likelihood
- Homoscedastic: Don't need heteroscedastic model

**Minimum viable model:**
```
Y_i ~ StudentT(nu, alpha + beta * log(x_i + 1), sigma)
nu ~ Gamma(2, 0.1)
[standard priors for alpha, beta, sigma]
```

**This is our Model 1 without flexible c.**

If this fails posterior predictive checks, add complexity incrementally.

---

## Summary and Recommendations

### Model Selection Flowchart

```
START
  ↓
Fit Model 1 (Robust Log Regression)
  ↓
Posterior predictive checks pass? ────YES──→ USE MODEL 1 ✓
  ↓ NO
  ↓
Is nu < 10? ────YES──→ Keep Student-t likelihood
  ↓ NO                     ↓
  ↓                   Consider other likelihoods
  ↓
Systematic residual pattern at x≈7? ────YES──→ Fit Model 3 (Change-Point)
  ↓ NO                                              ↓
  ↓                                            tau concentrated?
Smooth residual pattern? ────YES──→ Fit Model 2 (Spline)       ↓ NO
  ↓ NO                                  ↓                        ↓
  ↓                              ΔWAIC > 6?              Use Model 1 or 2
Investigate data quality          ↓ YES
                                  ↓
                            USE MODEL 2 ✓
```

### Priority Ranking

**For typical analysis (no domain-specific constraints):**

1. **Model 1 (Robust Log)** - 80% probability this is sufficient
   - Fit first, check thoroughly
   - Only proceed if inadequate

2. **Model 2 (B-Spline)** - 15% probability needed
   - If Model 1 shows systematic residual pattern
   - If WAIC improvement substantial

3. **Model 3 (Change-Point)** - 5% probability needed
   - Only if regime interpretation critical
   - Requires domain validation

### Deliverables

For each model fitted:

1. **Posterior summaries:** Mean, SD, 90% CI for all parameters
2. **Diagnostics:** Rhat, ESS, trace plots, pairs plots
3. **Model fit:** WAIC, LOO-CV, p_eff
4. **Posterior predictive checks:**
   - Residual plots (vs fitted, vs x)
   - Replicate data overlay
   - Coverage checks
5. **Predictions:**
   - In-sample (fitted values)
   - Out-of-sample (at new x values)
   - With 90% credible/prediction intervals
6. **Sensitivity analyses:** Prior robustness checks
7. **Interpretation:** Scientific conclusions, limitations, uncertainties

### Critical Success Factors

**This modeling exercise succeeds if:**

1. ✓ Final model passes posterior predictive checks (p ∈ [0.05, 0.95])
2. ✓ Diagnostics clean (Rhat < 1.01, ESS > 400, no divergences)
3. ✓ Predictions reasonable (within plausible Y range)
4. ✓ Uncertainty properly quantified (intervals neither too wide nor narrow)
5. ✓ Robust to prior choices (sensitivity analysis confirms)
6. ✓ Interpretable and communicable to domain experts

**We accept failure if:**

1. All models have extreme posterior predictive p-values → data generation process not captured
2. All models show multiple high Pareto k values → fundamental instability
3. Posterior intervals include scientifically absurd values → model mis-specified
4. Sensitivity analyses show wild swings → data too sparse for inference

**In case of failure:** Document what was tried, why it failed, what additional data/information is needed.

---

## Implementation Plan

### Phase 1: Setup (30 min)

1. Load data (`/workspace/data/data.csv`)
2. Create Stan/PyMC model files for all 3 models
3. Set up prior specifications
4. Prepare visualization functions

### Phase 2: Model 1 Fitting (1 hour)

1. Fit robust log regression
2. Check convergence diagnostics
3. Run posterior predictive checks
4. Visualize results
5. Document findings

**Go/No-Go Decision:** If Model 1 adequate, stop. Otherwise, proceed.

### Phase 3: Model 2 Fitting (1.5 hours)

1. Generate B-spline basis functions
2. Fit adaptive spline model
3. Check diagnostics and p_eff
4. Compare to Model 1 via WAIC
5. Sensitivity to knot count (K=3,4,5)

**Go/No-Go Decision:** If Model 2 substantially better (ΔWAIC > 6) and not overfitting (p_eff < 8), use it. Otherwise, revert to Model 1.

### Phase 4: Model 3 Fitting (2 hours, OPTIONAL)

1. Set up change-point model (marginalization or smooth approximation)
2. Fit model
3. Check tau posterior concentration
4. Compare to Models 1 and 2

**Go/No-Go Decision:** If tau concentrated and slopes different and WAIC competitive, consider as interpretation tool. Otherwise, don't use for primary inference.

### Phase 5: Final Analysis (1 hour)

1. Select final model (or ensemble)
2. Generate predictions at x ∈ {1, 5, 10, 15, 20, 25, 30, 35}
3. Create publication-quality figures
4. Write interpretation document
5. Summarize limitations and uncertainties

**Total Estimated Time:** 4-6 hours depending on how many models fitted.

---

## Conclusion

This experiment plan proposes **three robust, flexible Bayesian model classes** designed for the specific challenges of this dataset:

- **Small sample (n=27):** Parsimonious models with strong regularization
- **Potential outlier:** Heavy-tailed likelihoods (Student-t)
- **Functional form uncertainty:** Flexible splines, adaptive smoothness
- **Change-point hypothesis:** Direct test via regime-switching model

**Key Philosophical Stance:**
- We **embrace uncertainty** about functional form and error distribution
- We **build in robustness** so models don't break with minor assumption violations
- We **prioritize falsifiability** with clear criteria for abandoning each approach
- We **balance flexibility and parsimony** appropriate for n=27

**Most Likely Outcome:**
Model 1 (robust log regression) will be sufficient, with nu posterior around 15-25 (moderate tails handling x=31.5), providing interpretable diminishing-returns relationship.

**But we're ready if:**
- Model 1 fails PPC → Model 2 (splines) ready
- Regime interpretation critical → Model 3 (change-point) ready
- All models fail → Documented escape routes and red flags guide next steps

**This is adaptive Bayesian modeling:** Start simple, add complexity only when data demand it, stop when model adequate.

---

**Files Generated:**
- `/workspace/experiments/designer_3/proposed_models.md` (this document)

**Next Steps:**
- Await approval or feedback
- Proceed to implementation phase
- Coordinate with other designer teams for model comparison
