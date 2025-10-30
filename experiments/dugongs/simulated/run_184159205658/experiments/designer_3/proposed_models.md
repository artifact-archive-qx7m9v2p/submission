# Bayesian Model Proposals for Y vs x Relationship
## Designer 3: Flexible Yet Principled Approaches

**Date:** 2025-10-27
**Designer:** Designer 3 (Flexible/Regularized Specialist)
**Dataset:** N=27 observations, Y range [1.71, 2.63], x range [1.0, 31.5]

---

## Executive Summary

Based on EDA findings showing strong nonlinear relationship with diminishing returns, I propose **three fundamentally different Bayesian model classes** that balance flexibility with regularization:

1. **B-Spline Regression with Shrinkage Priors** (PRIMARY) - Adaptive flexibility with automatic smoothing
2. **Gaussian Process Regression with Squared Exponential Kernel** (ALTERNATIVE 1) - Nonparametric with uncertainty quantification
3. **Bayesian Polynomial Regression with Horseshoe Prior** (ALTERNATIVE 2) - Variable selection approach to polynomial order

**Key Philosophy:** Given N=27 and sparse high-x coverage, I prioritize models that:
- Automatically regularize to prevent overfitting
- Provide honest uncertainty quantification in sparse regions
- Can adapt to local curvature without global constraints
- Have principled mechanisms to avoid spurious flexibility

**Critical Insight:** The EDA suggests logarithmic/quadratic forms perform well, but these impose strong global constraints. My proposals allow data to reveal local structure while preventing overfitting through Bayesian shrinkage.

---

## Model 1: B-Spline Regression with Shrinkage Priors (PRIMARY)

### Ranking: 1st Priority
**Expected adequacy:** HIGH - optimal balance of flexibility and regularization for this sample size

### Model Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma)
mu_i = beta_0 + sum_{k=1}^{K} beta_k * B_k(x_i)
```

where B_k(x_i) are B-spline basis functions.

**Basis Function Design:**
- Order: Cubic B-splines (degree = 3, smooth second derivatives)
- Knots: 5 internal knots placed at quantiles of x
  - Specifically: quantiles [0.2, 0.4, 0.6, 0.8, 0.95]
  - This gives denser knots in data-rich region (x < 20)
  - Total basis functions: K = 9 (5 internal + 4 from boundary conditions)

**Priors:**
```
# Global intercept (weakly informative)
beta_0 ~ Normal(2.3, 0.5)  # Centered at mean(Y) = 2.32

# Spline coefficients with hierarchical shrinkage
beta_k ~ Normal(0, tau_k) for k = 1, ..., K
tau_k ~ Half-Cauchy(0, tau_global)  # Local shrinkage parameters

# Global shrinkage parameter (controls overall wiggliness)
tau_global ~ Half-Cauchy(0, 0.2)  # Prior expected variation per basis

# Residual standard deviation
sigma ~ Half-Normal(0.3)  # Weakly informative, based on obs SD(Y) = 0.28
```

**Prior Justification:**

1. **beta_0 prior:** Centered at observed mean Y (2.32), SD=0.5 allows ~95% prior mass in [1.3, 3.3], wider than observed range [1.71, 2.63]. Weakly informative.

2. **Hierarchical shrinkage:** Horseshoe-like structure via tau_k allows:
   - Individual coefficients to be strongly shrunk to zero if not needed
   - Some coefficients to escape shrinkage if data demands flexibility
   - Automatic smoothing without hard constraints

3. **tau_global prior:** Half-Cauchy(0, 0.2) implies prior expectation that each basis contributes ~0.2 units to Y. With 9 bases, this allows substantial total variation but shrinks individual contributions.

4. **sigma prior:** Half-Normal(0.3) places most mass below observed SD(Y)=0.28, acknowledging we'll explain variance. Not too informative - 95% prior mass in [0, 0.6].

### Implementation Strategy

**Framework:** PyMC (preferred for this model)

**Stan Alternative:** Also tractable, though PyMC's Gaussian process interoperability is useful for comparison.

**Computational Approach:**
```python
import pymc as pm
import numpy as np
from patsy import dmatrix

# Create B-spline design matrix
knots = np.quantile(x_data, [0.2, 0.4, 0.6, 0.8, 0.95])
B = dmatrix("bs(x, knots=knots, degree=3, include_intercept=True)",
            {"x": x_data, "knots": knots})

with pm.Model() as spline_model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.5)
    tau_global = pm.HalfCauchy('tau_global', beta=0.2)
    tau_local = pm.HalfCauchy('tau_local', beta=1, shape=B.shape[1])
    tau_k = tau_global * tau_local  # Hierarchical shrinkage

    beta_k = pm.Normal('beta_k', mu=0, sigma=tau_k, shape=B.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=0.3)

    # Likelihood
    mu = beta_0 + pm.math.dot(B, beta_k)
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y_data)

    # Sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.95)
```

**Sampling Parameters:**
- Chains: 4
- Warmup: 1000 iterations
- Sampling: 2000 iterations post-warmup
- Target acceptance: 0.95 (high for hierarchical model)
- Expected time: ~2-5 minutes on modern CPU

### Success Criteria

**This model is ADEQUATE if:**

1. **Convergence:** R-hat < 1.01 for all parameters, ESS > 400 per chain
2. **Posterior predictive checks:**
   - Observed Y within 95% posterior predictive interval
   - Replicated datasets match observed Y distribution
   - No systematic residual patterns
3. **Regularization evidence:**
   - At least 3-4 tau_k values shrunk strongly (< 0.05)
   - Not all coefficients equally large (would indicate no shrinkage)
4. **Predictive performance:**
   - LOO-CV ELPD better than simple logarithmic baseline
   - No influential observations (Pareto-k < 0.7)
5. **Uncertainty calibration:**
   - Wide credible intervals for x > 20 (sparse data region)
   - Narrow intervals where data is dense
6. **Functional form:**
   - Posterior mean curve visually captures diminishing returns
   - No spurious oscillations between knots

### Failure Criteria - I Will Abandon This Model If:

1. **Computational failure:**
   - R-hat > 1.05 after reparameterization attempts
   - Divergent transitions > 5% even with high target_accept
   - This suggests model is too complex for data

2. **No effective shrinkage:**
   - All tau_k posteriors have similar magnitude
   - Equivalent to unregularized spline (overfitting)
   - Action: Switch to simpler parametric form (logarithmic/quadratic)

3. **Systematic prediction failures:**
   - LOO-CV worse than logarithmic baseline
   - Multiple observations with Pareto-k > 0.7 (influential outliers)
   - This indicates model not capturing true structure

4. **Overfitting evidence:**
   - Perfect in-sample fit but poor predictive intervals
   - Posterior mean curve oscillates wildly between sparse data points
   - Action: Reduce knots or switch to GP with stronger smoothness

5. **Prior-data conflict:**
   - Posterior concentrated far from prior (e.g., sigma >> 0.6)
   - Suggests prior was too informative or misspecified
   - Action: Refit with vaguer priors as sensitivity check

6. **Underfitting:**
   - Posterior predictive p-value extreme (< 0.01 or > 0.99)
   - Residuals show clear remaining nonlinear pattern
   - Action: Increase knots or switch to more flexible GP

### Expected Challenges

1. **Posterior geometry:**
   - Hierarchical model creates funnel-like geometry
   - May need non-centered parameterization:
     ```
     beta_k_raw ~ Normal(0, 1)
     beta_k = tau_k * beta_k_raw  # Non-centered
     ```

2. **Label switching:**
   - Spline coefficients not uniquely identified
   - Not a problem for predictions, only coefficient interpretation
   - Focus on posterior predictive, not individual beta_k values

3. **Knot placement sensitivity:**
   - Results may depend on knot locations
   - Mitigation: Quantile-based placement (adaptive to data density)
   - Sensitivity check: Refit with +/- 1 knot

4. **Extrapolation:**
   - Splines can behave poorly beyond boundary knots
   - For x > 31.5: predictions are linear extrapolation of last segment
   - Must report this limitation clearly

### Theoretical Justification

**Why this flexibility is appropriate:**

1. **Local adaptation:** B-splines are locally supported - coefficients affect only nearby regions. This prevents global constraints (like logarithm's monotonicity) from distorting fit.

2. **Effective degrees of freedom:** With hierarchical shrinkage, effective DF adapts to data. If true curve is simple (e.g., logarithmic), shrinkage will reduce effective DF automatically.

3. **Sample size consideration:** N=27 with K=9 bases gives ~3 observations per basis on average. Shrinkage priors prevent overfitting despite seemingly high dimensionality.

4. **Interpretability:** While individual beta_k are hard to interpret, posterior mean curve is directly interpretable as smoothed regression function.

5. **Uncertainty honesty:** Spline basis naturally produces wider intervals in sparse regions (x>20) because fewer basis functions are active there.

**Falsification mindset:** This model assumes smoothness (C2 continuous). If true relationship has discontinuities or sharp corners, splines will fail. Evidence: would see systematic residual patterns at change points.

---

## Model 2: Gaussian Process Regression (ALTERNATIVE 1)

### Ranking: 2nd Priority
**Expected adequacy:** MEDIUM-HIGH - excellent uncertainty quantification but potentially too flexible for N=27

### Model Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma)
mu_i = f(x_i)
f ~ GP(m(x), k(x, x'))
```

**Mean Function:**
```
m(x) = beta_0  # Constant mean (let GP handle all structure)
beta_0 ~ Normal(2.3, 0.3)
```

**Covariance Function (Squared Exponential):**
```
k(x_i, x_j) = eta^2 * exp(-(x_i - x_j)^2 / (2 * ell^2))

# Priors
eta^2 ~ Half-Normal(0.5)      # Marginal variance (signal strength)
ell ~ Inverse-Gamma(5, 10)    # Length scale (smoothness)
sigma ~ Half-Normal(0.2)      # Nugget/noise variance
```

**Prior Justification:**

1. **Constant mean prior:** beta_0 ~ Normal(2.3, 0.3) centered at mean(Y). Narrower than spline model because GP will capture all variation.

2. **eta^2 (signal variance):**
   - Half-Normal(0.5) gives 95% prior mass in [0, 1.0]
   - Observed Var(Y) = 0.28^2 = 0.078, so prior allows much more
   - Reflects belief that GP will explain substantial variance

3. **ell (length scale):**
   - Inverse-Gamma(5, 10) has mode at 10/(5+1) = 1.67
   - Mean = 10/(5-1) = 2.5
   - Interprets as "typical x distance over which Y values correlate"
   - Inverse-Gamma chosen over Half-Normal because:
     - Heavy tail allows long length scales if needed
     - Concentrates mass away from zero (prevents overfitting)
     - Common in GP literature for length scales

4. **sigma (noise):**
   - Half-Normal(0.2) smaller than spline model
   - GP will absorb more signal than basis expansion
   - Still weakly informative

**Alternative Kernel Consideration:**

Squared Exponential (SE) assumes infinite differentiability. For diminishing returns pattern, could consider:

- **Matern-3/2:** Allows less smoothness, once-differentiable
  ```
  k(x, x') = eta^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
  where r = |x - x'|
  ```
  - More robust to local non-smoothness
  - Use if SE shows evidence of over-smoothing

### Implementation Strategy

**Framework:** PyMC (excellent GP support) or Stan (more control but verbose)

**Computational Approach (PyMC):**
```python
import pymc as pm

with pm.Model() as gp_model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.3)
    eta_sq = pm.HalfNormal('eta_sq', sigma=0.5)
    ell = pm.InverseGamma('ell', alpha=5, beta=10)
    sigma = pm.HalfNormal('sigma', sigma=0.2)

    # GP covariance
    cov_func = eta_sq * pm.gp.cov.ExpQuad(1, ls=ell)
    gp = pm.gp.Marginal(mean_func=beta_0, cov_func=cov_func)

    # Likelihood
    Y_obs = gp.marginal_likelihood('Y_obs', X=x_data[:, None],
                                    y=Y_data, sigma=sigma)

    # Sampling (GPs can be slow)
    trace = pm.sample(1000, tune=1000, target_accept=0.99)
```

**Stan Alternative:** Use built-in `gp_exp_quad_cov` but requires manual Cholesky decomposition.

**Computational Concerns:**
- GPs scale O(N^3) in time, O(N^2) in memory
- N=27 is small enough for exact inference
- Expect ~5-10 minutes sampling time
- For N>100, would need sparse/variational approximations

### Success Criteria

**This model is ADEQUATE if:**

1. **Convergence:** R-hat < 1.01, ESS > 300 (GPs can have lower ESS)
2. **Posterior predictive checks:**
   - Data within posterior predictive envelope
   - Predictive distribution matches observed Y
3. **Length scale reasonableness:**
   - Posterior ell not at prior boundary (too small or too large)
   - ell ~ 5-15 would be reasonable given x range [1, 31.5]
   - ell < 1 suggests overfitting, ell > 30 suggests underfitting
4. **Uncertainty quantification:**
   - Credible intervals widen dramatically for x > 20
   - Narrowest intervals where data is replicated
5. **Predictive performance:**
   - LOO-CV competitive with spline model
   - Pareto-k diagnostics all < 0.7

### Failure Criteria - I Will Abandon This Model If:

1. **Length scale collapse:**
   - Posterior ell < 0.5 (overfitting to noise)
   - Function oscillates wildly between observations
   - Action: This indicates N=27 insufficient for nonparametric approach
   - Pivot to: Parametric models (logarithmic/spline)

2. **Length scale escape:**
   - Posterior ell > 50 (function too smooth)
   - Effectively fitting constant or linear function
   - Action: GP not providing value over simple parametric form
   - Pivot to: Logarithmic or quadratic regression

3. **Computational intractability:**
   - Divergences > 10% despite target_accept=0.99
   - ESS < 100 after extended sampling
   - Action: Poor posterior geometry for this data structure
   - Pivot to: Approximations (variational inference) or different model

4. **Predictive failure:**
   - Multiple Pareto-k > 0.7 in LOO-CV
   - Indicates some observations highly influential
   - Action: Model not robust to individual points
   - Pivot to: Student-t likelihood or robust regression

5. **Prior-posterior conflict:**
   - Posterior eta^2 >> 1.0 or sigma >> 0.4
   - Data fighting prior assumptions
   - Action: Refit with vaguer priors, compare to spline

6. **No improvement over parametric:**
   - LOO-CV score worse than logarithmic baseline
   - Complexity not justified by performance
   - Action: Abandon GP, use simpler model

### Expected Challenges

1. **Prior sensitivity:**
   - GP posteriors can be sensitive to length scale prior
   - Mitigation: Sensitivity analysis with alternative priors
   - Consider Half-Normal(5) for ell as alternative

2. **Computational cost:**
   - O(N^3) can strain MCMC even at N=27
   - May need more iterations than simpler models
   - Consider using GPU acceleration if available

3. **Extrapolation behavior:**
   - GP will revert to prior mean beyond data range
   - For x > 31.5, predictions will trend toward beta_0 = 2.3
   - This may be MORE reasonable than spline's linear extrapolation
   - But: may not capture true asymptotic behavior

4. **Interpretation:**
   - No simple equation for relationship
   - Must communicate via plots and predictions
   - Stakeholders may prefer parametric form

### Theoretical Justification

**Why this flexibility is appropriate:**

1. **Nonparametric uncertainty:** GPs naturally quantify uncertainty from both model and parameter uncertainty. Crucial for sparse high-x region.

2. **Automatic smoothness:** Squared exponential kernel enforces smooth functions without requiring knot placement decisions (unlike splines).

3. **Bayesian framework:** Full posterior over functions, not just point estimates. Natural for this probabilistic programming context.

4. **Covariance structure:** GP can learn that nearby x values have similar Y, implementing adaptive local smoothing.

5. **Sample size:** N=27 is on the edge for GPs - small enough to be tractable, large enough to learn hyperparameters. Below N=20 would be problematic.

**Why flexibility might be INAPPROPRIATE:**

1. **Occam's Razor:** If true function is simple (e.g., logarithmic), GP's infinite-dimensional flexibility is overkill.

2. **Overfitting risk:** Despite Bayesian regularization, GPs can fit noise when hyperpriors are weak.

3. **Computational inefficiency:** If parametric model suffices, GP wastes computation.

**Falsification test:** If posterior length scale collapses (ell < 1) or escapes (ell > 30), this indicates GP is wrong tool for this problem.

---

## Model 3: Bayesian Polynomial Regression with Horseshoe Prior (ALTERNATIVE 2)

### Ranking: 3rd Priority
**Expected adequacy:** MEDIUM - principled variable selection but assumes polynomial form

### Model Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma)
mu_i = beta_0 + sum_{j=1}^{J} beta_j * x_i^j
```

where J = maximum polynomial degree (set to 6).

**Priors (Horseshoe for Automatic Order Selection):**
```
# Intercept
beta_0 ~ Normal(2.3, 0.5)

# Horseshoe prior on polynomial coefficients
beta_j ~ Normal(0, lambda_j * tau) for j = 1, ..., J

# Local shrinkage parameters
lambda_j ~ Half-Cauchy(0, 1) for j = 1, ..., J

# Global shrinkage parameter
tau ~ Half-Cauchy(0, tau_0)
where tau_0 = p_0 / (J - p_0) * sigma_y / sqrt(N)
      p_0 = 3 (expected number of non-zero coefficients)
      sigma_y = 0.28 (observed SD of Y)
      N = 27
      => tau_0 = 3/(6-3) * 0.28/sqrt(27) = 0.054

# Residual variance
sigma ~ Half-Normal(0.3)
```

**Standardization (CRITICAL):**
```
# Standardize x to prevent numerical issues
x_std = (x - mean(x)) / sd(x)
# Work with x_std in all computations
# Back-transform predictions for interpretation
```

**Prior Justification:**

1. **Horseshoe structure:** Allows sparse selection - most beta_j shrunk to zero, few escape to fit signal. Ideal for unknown polynomial order.

2. **tau_0 calculation:** Based on Piironen & Vehtari (2017) formula:
   - Assumes 3 non-zero coefficients (linear, quadratic, maybe cubic)
   - Scales with residual SD and sample size
   - tau_0 = 0.054 is small, enforcing strong shrinkage by default

3. **lambda_j priors:** Half-Cauchy(0,1) is standard in horseshoe literature. Heavy tails allow coefficients to escape shrinkage if needed.

4. **Why J=6:**
   - Beyond cubic (J=3) for flexibility
   - But not too high to risk Runge phenomenon
   - With horseshoe, unused terms will shrink to zero
   - Could increase to J=8 but diminishing returns

5. **Standardization justification:**
   - x^6 with x ~ 30 gives values ~ 10^9, causing numerical instability
   - Standardized x has mean=0, SD=1, so x^6 ~ O(1)
   - CRITICAL for polynomial models

### Implementation Strategy

**Framework:** Stan (preferred for horseshoe efficiency) or PyMC

**Stan Approach:**
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
  int<lower=1> J;  // Max polynomial degree
}

transformed data {
  vector[N] x_std;
  real x_mean;
  real x_sd;

  x_mean = mean(x);
  x_sd = sd(x);
  x_std = (x - x_mean) / x_sd;
}

parameters {
  real beta_0;
  vector[J] beta_raw;
  vector<lower=0>[J] lambda;
  real<lower=0> tau;
  real<lower=0> sigma;
}

transformed parameters {
  vector[J] beta;
  beta = beta_raw .* lambda * tau;  // Horseshoe
}

model {
  vector[N] mu;
  matrix[N, J] X_poly;

  // Construct polynomial design matrix
  for (j in 1:J) {
    X_poly[:, j] = x_std ^ j;
  }

  mu = beta_0 + X_poly * beta;

  // Priors
  beta_0 ~ normal(2.3, 0.5);
  beta_raw ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, 0.054);
  sigma ~ normal(0, 0.3);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    real mu_i;
    vector[J] x_powers;

    for (j in 1:J) {
      x_powers[j] = x_std[i]^j;
    }
    mu_i = beta_0 + dot_product(x_powers, beta);

    Y_rep[i] = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu_i, sigma);
  }
}
```

**Sampling Parameters:**
- Chains: 4
- Warmup: 2000 (horseshoe can be slow to converge)
- Sampling: 2000
- Target accept: 0.95
- Expected time: 3-7 minutes

### Success Criteria

**This model is ADEQUATE if:**

1. **Convergence:** R-hat < 1.01, ESS > 400 for beta, lambda, tau
2. **Sparse selection:**
   - Most lambda_j have posterior concentrated near 0
   - 2-4 lambda_j clearly separated from zero (active coefficients)
   - Clear bimodal structure in joint (lambda_j, beta_j) posterior
3. **Polynomial order identification:**
   - Posterior probability(beta_j = 0) > 0.8 for j > 4
   - This would suggest quadratic or cubic suffices
4. **Predictive performance:**
   - LOO-CV competitive with spline model
   - No Pareto-k > 0.7
5. **Posterior predictive checks:**
   - No residual patterns
   - Observed data within predictive intervals
6. **Reasonable coefficient magnitudes:**
   - Non-zero beta_j have |beta_j| < 2.0 (on standardized scale)
   - No extreme values suggesting numerical issues

### Failure Criteria - I Will Abandon This Model If:

1. **No shrinkage:**
   - All lambda_j have similar posterior (no sparsity)
   - Equivalent to unregularized polynomial (disaster)
   - Action: tau_0 was too large or model inappropriate
   - Pivot to: Spline or GP with better regularization

2. **Runge phenomenon:**
   - Posterior mean oscillates wildly between observations
   - Common with high-degree polynomials
   - Evidence: inflection points exceed data support
   - Action: Even horseshoe can't save bad functional form
   - Pivot to: Spline (local support) or GP

3. **Computational failure:**
   - Divergences > 5% despite high target_accept
   - R-hat > 1.05 for tau or lambda parameters
   - Indicates difficult posterior geometry
   - Action: Try non-centered parameterization or abandon

4. **Extrapolation catastrophe:**
   - Predictions for x > 31.5 are absurd (e.g., Y < 0 or Y > 5)
   - Polynomials notoriously bad at extrapolation
   - This is EXPECTED but degree of badness matters
   - Action: If unusable, abandon for spline/GP with better extrapolation

5. **Underfitting:**
   - All coefficients shrunk to zero (tau collapsed)
   - Model effectively constant
   - Action: Increase tau_0 or abandon horseshoe

6. **Numerical instability:**
   - Despite standardization, large posterior correlations
   - Condition number of X_poly matrix too high
   - Action: Reduce J or switch to orthogonal polynomials

### Expected Challenges

1. **Posterior geometry:**
   - Horseshoe creates difficult funnel shapes
   - May need non-centered parameterization
   - Long warmup times expected

2. **Extrapolation:**
   - Polynomials behave TERRIBLY beyond data range
   - For x > 31.5, predictions will diverge (up or down)
   - Must restrict inference to observed x range
   - This is MAJOR limitation compared to GP

3. **Interpretation:**
   - Standardized coefficients hard to interpret
   - "Y increases by beta_2 per unit squared-standardized-x" is nonsense to stakeholders
   - Must back-transform to original scale for communication

4. **Multicollinearity:**
   - Even with standardization, x^j are correlated
   - Horseshoe helps but doesn't eliminate
   - Posterior correlations between beta_j expected

### Theoretical Justification

**Why this flexibility is appropriate:**

1. **Automatic model selection:** Horseshoe lets data choose polynomial degree without manual comparison of J models.

2. **Sparsity principle:** Most relationships don't need high-degree polynomials. Horseshoe encodes this belief.

3. **Computational efficiency:** Polynomial design matrix is simple compared to GP covariance.

4. **EDA support:** Quadratic fit had R²=0.86, suggesting low-degree polynomial may suffice. Horseshoe will identify this.

5. **Global approximation:** Unlike splines (local basis), polynomials are global. If true function is smooth and simple, polynomial is natural choice.

**Why flexibility might be INAPPROPRIATE:**

1. **Polynomial bias:** Assumes relationship is polynomial (or well-approximated by one). If true curve is exponential/logarithmic, high-degree polynomial is wrong tool.

2. **Extrapolation danger:** Polynomials diverge outside data. For x > 31.5, predictions are unreliable.

3. **Runge phenomenon:** Even with shrinkage, polynomials can oscillate between data points.

4. **Not truly flexible:** Horseshoe selects polynomial ORDER, but still constrained to polynomial FORM. Less flexible than GP/spline.

**Falsification test:** If model selects very high degree (J > 5 with many non-zero coefficients), this suggests polynomial form is wrong. Switch to spline or GP.

---

## Model Comparison and Selection Strategy

### Prioritization Rationale

**Why Spline is 1st:**
- Best balance of flexibility and regularization for N=27
- Local basis prevents global constraints from distorting fit
- Proven track record in similar settings
- Reasonable extrapolation behavior (linear beyond boundary)
- Computational efficiency

**Why GP is 2nd:**
- Superior uncertainty quantification (key for sparse x>20)
- No knot placement decisions required
- Principled nonparametric approach
- BUT: potentially over-flexible for N=27, higher computational cost

**Why Polynomial is 3rd:**
- Assumes polynomial form (strong constraint)
- Poor extrapolation behavior (critical limitation)
- If data is truly polynomial, horseshoe will discover it efficiently
- BUT: if not polynomial, forcing it is wrong approach
- Good as "stress test" - if polynomial fails badly, confirms need for local basis

### Cross-Model Comparison Plan

After fitting all three models:

1. **Convergence check:** Ensure all models converged (R-hat < 1.01)

2. **LOO-CV comparison:**
   ```
   Compare ELPD_loo and SE across models
   Difference > 2*SE is meaningful
   ```

3. **Posterior predictive checks:**
   - Visual: overlay posterior mean and credible bands
   - Quantitative: posterior predictive p-values for test statistics
     - Mean, variance, skewness of Y
     - Range of Y
     - Number of local maxima in fitted curve

4. **Residual analysis:**
   - Plot residuals from each model's posterior mean
   - Check for remaining patterns
   - Compare residual variance

5. **Extrapolation assessment:**
   - Predict for x in [32, 35] (beyond data)
   - Which model gives most reasonable predictions?
   - Which has widest credible intervals (appropriately uncertain)?

6. **Interpretability:**
   - Can we explain model to non-statisticians?
   - Spline: "smooth curve fit with automatic wiggliness control"
   - GP: "every point is smoothly related to nearby points"
   - Polynomial: "quadratic-ish relationship with variable selection"

### Decision Rules

**If Spline wins (expected):**
- Use as primary model for inference
- Report GP uncertainty bands as sensitivity check
- Dismiss polynomial if poor extrapolation

**If GP wins:**
- Computational cost was justified by better fit
- Emphasize uncertainty quantification in sparse regions
- Use spline as interpretable approximation

**If Polynomial wins:**
- Surprising! Suggests relationship is genuinely low-order polynomial
- Check which degree was selected
- Refit simpler model with just selected terms (no horseshoe)

**If all models fail:**
- Suggests fundamental misspecification
- Revisit EDA for missed patterns
- Consider:
  - Asymptotic model (Michaelis-Menten)
  - Segmented regression (different function for low/high x)
  - Mixture models (multiple subpopulations)
  - Robust regression (Student-t likelihood)

---

## Red Flags and Decision Points

### Red Flags That Trigger Model Class Change

**FLAG 1: Posterior Predictive Failure**
- **Symptom:** Observed Y systematically outside posterior predictive intervals
- **Diagnosis:** Likelihood misspecified (not just mean function)
- **Action:** Switch to Student-t likelihood for all models
- **Rationale:** EDA showed normal residuals, but small outliers could bias results

**FLAG 2: Heteroscedasticity Emerges**
- **Symptom:** Residual variance clearly increases/decreases with x or mu
- **Diagnosis:** Constant-sigma assumption violated
- **Action:** Add variance model: sigma_i = sigma_0 * exp(gamma * x_i)
- **Test:** Refit spline with both constant and varying sigma, compare LOO

**FLAG 3: All Models Overfit Identically**
- **Symptom:** All three models show same overfitting pattern (e.g., oscillation)
- **Diagnosis:** N=27 is too small for any flexible approach
- **Action:** Abandon all, use simple logarithmic/quadratic from EDA
- **Rationale:** Sometimes simplicity wins

**FLAG 4: All Models Underfit Identically**
- **Symptom:** All three models miss same pattern in residuals
- **Diagnosis:** Functional form class is wrong
- **Action:** Consider non-additive models:
  - Interaction terms (if there's a hidden covariate)
  - Mixture models (if multiple regimes)
  - Change-point models (if threshold effect)

**FLAG 5: Extreme Prior Sensitivity**
- **Symptom:** Results change dramatically with minor prior perturbations
- **Diagnosis:** Data is weak relative to model complexity
- **Action:** Use simpler model where data dominates prior
- **Test:** Refit with 2x and 0.5x prior scales, compare posteriors

### Decision Points for Major Strategy Pivots

**DECISION POINT 1: After Initial Fits (Day 1)**
- **Question:** Did any model converge with R-hat < 1.01?
- **If NO:** Simplify all models (fewer knots, lower J, stronger priors)
- **If YES:** Proceed to predictive checks

**DECISION POINT 2: After Posterior Predictive Checks (Day 1-2)**
- **Question:** Does any model pass visual and quantitative checks?
- **If NO:** Reconsider likelihood family (Student-t? heteroscedastic?)
- **If YES:** Proceed to LOO-CV

**DECISION POINT 3: After LOO-CV (Day 2)**
- **Question:** Is there a clear winner (ELPD diff > 2*SE)?
- **If NO:** Models are equivalent - choose simplest (spline or polynomial)
- **If YES:** Use best model but validate with sensitivity checks

**DECISION POINT 4: After Extrapolation Test (Day 2)**
- **Question:** Are predictions for x > 31.5 plausible?
- **If NO for all:** Restrict inference to x <= 31.5, acknowledge limitation
- **If YES for one:** That model has best functional form for this problem

**DECISION POINT 5: After Stakeholder Review (Final)**
- **Question:** Can we explain and defend this model?
- **If NO:** Revisit simpler models with better narrative
- **If YES:** Finalize and document

---

## Alternative Approaches If All Proposed Models Fail

### Backup Plan 1: Simplified Parametric Models

If all flexible models overfit or fail to converge:

**Option A: Bayesian Logarithmic Regression** (from EDA)
```
Y ~ Normal(beta_0 + beta_1 * log(x + c), sigma)
```
- Proven fit (R²=0.83)
- Two parameters, very stable
- Interpretable

**Option B: Bayesian Quadratic Regression** (from EDA)
```
Y ~ Normal(beta_0 + beta_1*x + beta_2*x^2, sigma)
```
- Best empirical fit (R²=0.86)
- Three parameters, manageable
- Constrain beta_2 < 0

### Backup Plan 2: Robust Extensions

If outliers or non-normality become issues:

**Option A: Student-t Likelihood**
```
Y ~ StudentT(nu, mu, sigma)
nu ~ Gamma(2, 0.1)  # Degrees of freedom
```
- Robust to outliers
- Reduces to Normal as nu → infinity
- Add to any mean function (log, spline, GP)

**Option B: Heteroscedastic Normal**
```
Y ~ Normal(mu, sigma_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i
```
- Allows variance to change with x
- Two additional parameters

### Backup Plan 3: Hybrid/Ensemble Approaches

**Option A: Bayesian Model Averaging**
- Fit logarithmic, quadratic, spline
- Weight predictions by LOO stacking weights
- Combines strengths of multiple models

**Option B: Two-Regime Model**
```
If x <= threshold: Y ~ Normal(mu_1(x), sigma_1)
If x > threshold: Y ~ Normal(mu_2(x), sigma_2)
```
- Threshold ~ Uniform(5, 20)
- Different functions for low/high x
- Captures potential regime change

### Stopping Rules

**STOP and use simple model if:**
1. Three days of effort yields no convergent flexible model
2. All models perform worse than logarithmic baseline
3. Computational cost exceeds benefit (e.g., GP takes hours)
4. Results are not interpretable to stakeholders

**STOP and collect more data if:**
1. All models show massive uncertainty for x > 20
2. Critical decisions depend on high-x predictions
3. Extrapolation is required but all models fail
4. N=27 is genuinely too small (multiple model classes fail identically)

---

## Summary of Falsification Criteria

### For Spline Model
- **Abandon if:** No shrinkage occurs, oscillations between knots, LOO worse than baseline, knot sensitivity is extreme

### For GP Model
- **Abandon if:** Length scale collapses (<0.5) or escapes (>50), computational intractability, no improvement over parametric

### For Polynomial Model
- **Abandon if:** Runge phenomenon, no sparsity, extrapolation is catastrophic, all coefficients shrunk to zero

### For ALL Models Collectively
- **Abandon flexible approach if:** All overfit identically, all underfit identically, all fail posterior predictive checks, extreme prior sensitivity

---

## Implementation Timeline

**Day 1: Core Fits**
- Morning: Implement and fit spline model
- Afternoon: Implement and fit GP model
- Evening: Implement and fit polynomial model
- Deliverable: Three converged posterior samples (or convergence diagnostics)

**Day 2: Comparison and Validation**
- Morning: LOO-CV comparison, posterior predictive checks
- Afternoon: Residual analysis, extrapolation tests
- Evening: Sensitivity analysis (prior robustness)
- Deliverable: Model comparison table, recommendation

**Day 3: Refinement or Pivot**
- If success: Finalize winning model, generate predictions, visualizations
- If failure: Implement backup plan, refit simpler models
- Deliverable: Final model with full diagnostics and interpretation

---

## Conclusion

These three models represent fundamentally different philosophies:

1. **Spline:** Structured flexibility via basis functions
2. **GP:** Unstructured flexibility via covariance functions
3. **Polynomial:** Constrained flexibility via variable selection

All three share:
- Normal likelihood (justified by EDA)
- Bayesian regularization (shrinkage/smoothness priors)
- Focus on honest uncertainty quantification

The key insight is that **all three might fail** if the true relationship is something else entirely (e.g., asymptotic saturation). I've designed each model with explicit falsification criteria to detect this early and pivot.

**Expected outcome:** Spline model will win due to optimal flexibility/parsimony balance for N=27. GP will provide superior uncertainty estimates but at higher computational cost. Polynomial will reveal effective degree (likely 2-3) and confirm EDA's quadratic finding, but poor extrapolation will limit its utility.

**Unexpected outcome:** If all three models converge to similar predictions, this is STRONG evidence that the relationship is well-determined by data despite model flexibility. Conversely, if all three fail identically, this signals fundamental misspecification requiring new model class.

---

**Files to be created during implementation:**
- `/workspace/experiments/designer_3/models/spline_model.py` (or `.stan`)
- `/workspace/experiments/designer_3/models/gp_model.py`
- `/workspace/experiments/designer_3/models/polynomial_model.py`
- `/workspace/experiments/designer_3/results/model_comparison.md`
- `/workspace/experiments/designer_3/results/diagnostics/*.png`
- `/workspace/experiments/designer_3/results/final_recommendation.md`

**Absolute paths for all outputs:**
- Base directory: `/workspace/experiments/designer_3/`
- Models: `/workspace/experiments/designer_3/models/`
- Results: `/workspace/experiments/designer_3/results/`
- Data: `/workspace/data/data.csv`
