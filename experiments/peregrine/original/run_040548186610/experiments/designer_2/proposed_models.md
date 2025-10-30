# Designer 2: Flexible/Non-parametric Bayesian Models for Time Series Count Data

**Designer:** Non-parametric/Flexible Modeling Specialist
**Date:** 2025-10-29
**Focus:** Flexible functional forms without strong parametric assumptions
**Implementation:** Stan/PyMC with full posterior inference

---

## Executive Summary

I propose **three flexible Bayesian models** that can adapt to complex non-linear patterns in count data without imposing rigid parametric forms. Each model tackles the key challenges differently:

1. **Gaussian Process Count Model (GP-NegBin)**: Flexible non-parametric trend learning
2. **Bayesian P-spline GLM**: Smooth curves with automatic complexity penalization
3. **Semi-parametric Mixture Model**: Combines structured growth with flexible deviations

**My adversarial stance**: These models are designed to FAIL if:
- The true process is actually simple (parametric models would win)
- The data pattern is driven by unmeasured discrete jumps (need changepoint models)
- N=40 is too small for the flexibility (overfitting)

---

## Data Context Recap

**Key EDA Findings:**
- N = 40 observations, counts range [19, 272]
- Extreme overdispersion: Var/Mean = 68 (vs 1.0 for Poisson)
- Strong non-linear growth: Quadratic R² = 0.961 vs Linear R² = 0.885
- Accelerating growth: 6× rate increase from early to late period
- Very high autocorrelation: lag-1 r = 0.989
- Non-constant overdispersion across time periods (Q3 shows 13.5× vs 1.6-2.4× elsewhere)

**My interpretation**: The data suggests a smooth, accelerating process with complex variance structure. But EDA can mislead - what looks "smooth" might be hiding discrete events, regime changes, or external shocks.

---

## Model 1: Gaussian Process Negative Binomial (GP-NegBin)

### Philosophy
Let the data tell us the functional form. Don't assume quadratic, exponential, or any parametric shape. Use a GP prior to learn arbitrarily complex smooth functions.

### Model Specification

**Likelihood:**
```
C_i ~ NegativeBinomial(μ_i, φ)
```

**Link function:**
```
log(μ_i) = f(year_i)
```

**Prior on function f:**
```
f ~ GP(m(x), k(x, x'))
```

**Mean function:**
```
m(x) = β₀ + β₁ · x
```
(Linear trend captures overall direction; GP captures deviations)

**Kernel (covariance function):**
```
k(x, x') = α² · exp(-ρ² · (x - x')²)
```
(Squared exponential / RBF kernel for smooth functions)

**Priors:**
```
β₀ ~ Normal(log(100), 1)           # Based on observed mean ~109
β₁ ~ Normal(0.8, 0.5)              # Positive growth expected
α² ~ Half-Normal(0, 1)             # GP amplitude (signal variance)
ρ ~ Inverse-Gamma(5, 5)            # GP lengthscale
φ ~ Half-Normal(0, 10)             # Overdispersion parameter
```

### Why This Might Work
- **No functional form assumption**: GP can represent any smooth function
- **Handles non-linearity naturally**: Can capture accelerating growth, S-curves, changepoints
- **Uncertainty quantification**: GP provides principled uncertainty about the function shape
- **Handles autocorrelation**: GP structure naturally induces correlation between nearby points
- **Overdispersion handled**: Negative binomial captures variance inflation

### Why This Might FAIL (Falsification Criteria)

**I will abandon this model if:**

1. **Posterior concentrates on nearly linear functions**
   - Evidence: GP lengthscale ρ → ∞ (flat prior correlation)
   - Interpretation: Data is too simple, wasting flexibility
   - Pivot to: Parametric GLM (Designer 1's models)

2. **Computational divergences despite tuning**
   - Evidence: >10% divergent transitions, poor mixing (R̂ > 1.1)
   - Interpretation: GP + count likelihood = difficult geometry, need reparameterization
   - Pivot to: Spline-based approach (Model 2) or state-space (Designer 3)

3. **Wild extrapolation outside data range**
   - Evidence: Posterior predictive intervals explode beyond reasonable values
   - Interpretation: GP is overfitting to noise, not learning structure
   - Pivot to: More constrained model with structural priors

4. **Poor predictive performance (LOO-CV)**
   - Evidence: LOO-IC worse than simple parametric models by >10 units
   - Interpretation: Flexibility not justified by data (n=40 too small)
   - Pivot to: Simpler models

5. **Prior-data conflict on lengthscale**
   - Evidence: Posterior on ρ hits prior boundaries
   - Interpretation: GP kernel inappropriate for this temporal structure
   - Pivot to: Different kernel (Matern, periodic) or abandon GP entirely

### Stress Test Design
**Test**: Fit model on first 30 observations, predict last 10
**Expectation**: If GP is learning true structure, predictions should be reasonable
**Failure mode**: If predictions are nonsense, GP is memorizing not learning

### Implementation Notes (Stan)

```stan
data {
  int<lower=1> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

transformed data {
  real delta = 1e-9;  // Jitter for numerical stability
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> alpha_sq;
  real<lower=0> rho;
  real<lower=0> phi;
  vector[N] eta;  // Non-centered parameterization
}

transformed parameters {
  vector[N] f;
  vector[N] mu;

  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(year, alpha_sq, rho);

    // Add jitter to diagonal for numerical stability
    for (n in 1:N) {
      K[n, n] = K[n, n] + delta;
    }

    L_K = cholesky_decompose(K);
    f = beta_0 + beta_1 * year + L_K * eta;
  }

  mu = exp(f);
}

model {
  // Priors
  beta_0 ~ normal(log(100), 1);
  beta_1 ~ normal(0.8, 0.5);
  alpha_sq ~ normal(0, 1);
  rho ~ inv_gamma(5, 5);
  phi ~ normal(0, 10);
  eta ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_lpmf(C[n] | mu[n], phi);
    C_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
```

**Key technical choices:**
- Non-centered parameterization (eta) for better sampling
- Cholesky decomposition for computational efficiency
- Jitter (delta) for numerical stability
- Generated quantities for LOO-CV and posterior predictive checks

### Expected Insights
- **If successful**: Learn the exact shape of non-linearity without parametric assumptions
- **Lengthscale parameter**: Tells us how quickly the function changes
- **Amplitude parameter**: Tells us how much deviation from linear trend
- **Comparison to parametric**: If GP ≈ quadratic, validates simpler model

---

## Model 2: Bayesian P-spline Negative Binomial

### Philosophy
Splines provide flexibility without full GP complexity. P-splines add automatic smoothness penalty, avoiding overfitting. This is a "Goldilocks" model - more flexible than polynomials, less complex than GPs.

### Model Specification

**Likelihood:**
```
C_i ~ NegativeBinomial(μ_i, φ)
```

**Link function:**
```
log(μ_i) = B(year_i)ᵀ θ
```
where B(year_i) is a vector of B-spline basis functions

**Spline setup:**
- K = 8 interior knots (evenly spaced quantiles)
- Degree = 3 (cubic B-splines)
- Total basis functions: D = K + degree + 1 = 12

**Smoothness prior (Random Walk on coefficients):**
```
θ₁ ~ Normal(0, τ_0)
θ₂ ~ Normal(0, τ_0)
Δθⱼ = θⱼ - θⱼ₋₁ ~ Normal(0, τ²)  for j = 3, ..., D
```
This is a "random walk" prior that penalizes wiggles

**Priors:**
```
τ_0 ~ Normal(0, 5)                 # Prior variance for first two coefficients
τ ~ Half-Normal(0, 1)              # Smoothness parameter (smaller = smoother)
φ ~ Half-Normal(0, 10)             # Overdispersion
```

### Why This Might Work
- **Flexible but controlled**: Splines can fit complex curves but smoothness prior prevents overfitting
- **Automatic complexity tuning**: τ parameter learns appropriate smoothness from data
- **Computational efficiency**: Simpler than GP, easier to fit
- **Interpretable basis**: Can visualize contribution of each spline basis function
- **Local adaptation**: Splines can have different curvature in different regions

### Why This Might FAIL (Falsification Criteria)

**I will abandon this model if:**

1. **Smoothness parameter τ → 0 (extreme smoothing)**
   - Evidence: Posterior on τ concentrates near zero
   - Interpretation: Model collapsing to nearly linear function
   - Pivot to: Simple parametric polynomial (waste of splines)

2. **Smoothness parameter τ → ∞ (no smoothing)**
   - Evidence: Posterior on τ has long right tail, not concentrating
   - Interpretation: Over-flexible, fitting noise
   - Pivot to: Fewer knots or stronger penalty

3. **Boundary artifacts**
   - Evidence: Posterior predictive shows unrealistic behavior at year extremes
   - Interpretation: Spline extrapolation failing (known issue)
   - Pivot to: GP with better extrapolation or parametric tail constraints

4. **Knot placement sensitivity**
   - Evidence: Results change dramatically with different knot locations
   - Interpretation: Model not robust, overfitting to specific configuration
   - Pivot to: Parametric model or GP

5. **Worse LOO-CV than GP-NegBin**
   - Evidence: LOO-IC difference > 10, large standard errors
   - Interpretation: Not capturing true complexity or over-smoothing
   - Pivot to: Different spline configuration or abandon splines

### Stress Test Design
**Test 1**: Fit with 6 knots vs 10 knots vs 15 knots
**Expectation**: Smoothness parameter τ should adapt, results shouldn't change drastically
**Failure**: High sensitivity = model not learning robust structure

**Test 2**: Check posterior predictive at boundaries (year = ±2)
**Expectation**: Should extrapolate reasonably within ±1 SD of training range
**Failure**: Wild predictions = spline extrapolation problem

### Implementation Notes (Stan)

```stan
functions {
  // B-spline basis function (cubic)
  vector bspline_basis(real x, vector knots, int degree) {
    int n_knots = num_elements(knots);
    int n_basis = n_knots + degree - 1;
    vector[n_basis] basis;

    // Recursive de Boor-Cox algorithm
    // (Simplified; full implementation would be ~50 lines)
    // For practical use: pre-compute in R/Python, pass as data

    return basis;
  }
}

data {
  int<lower=1> N;
  vector[N] year;
  array[N] int<lower=0> C;
  int<lower=1> D;              // Number of basis functions
  matrix[N, D] B;              // Pre-computed B-spline basis matrix
}

parameters {
  vector[D] theta;             // Spline coefficients
  real<lower=0> tau;           // Smoothness parameter
  real<lower=0> phi;           // Overdispersion
}

transformed parameters {
  vector[N] mu;
  vector[N] eta;

  eta = B * theta;  // Linear combination of basis functions
  mu = exp(eta);
}

model {
  // Priors
  theta[1] ~ normal(0, 5);
  theta[2] ~ normal(0, 5);

  // Random walk prior for smoothness
  for (j in 3:D) {
    theta[j] - theta[j-1] ~ normal(0, tau);
  }

  tau ~ normal(0, 1);
  phi ~ normal(0, 10);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_lpmf(C[n] | mu[n], phi);
    C_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
```

**Pre-processing (Python/R):**
```python
from patsy import dmatrix
import numpy as np

# Create B-spline basis with 8 interior knots
knots = np.quantile(year, np.linspace(0, 1, 10))  # 8 interior + 2 boundary
B = dmatrix("bs(year, knots=knots[1:-1], degree=3, include_intercept=True)",
            {"year": year, "knots": knots}, return_type="dataframe")
```

### Expected Insights
- **Smoothness parameter τ**: Tells us how much wiggle is justified by data
- **Coefficient pattern**: Monotonically increasing θⱼ suggests smooth growth
- **Visual fit**: Can plot fitted curve segment by segment
- **Comparison**: If spline ≈ quadratic, suggests simple polynomial sufficient

---

## Model 3: Semi-parametric Hierarchical Growth Model

### Philosophy
**Hybrid approach**: Combine structured parametric growth (biological/mechanistic) with flexible non-parametric deviations. This acknowledges that real processes often have known structure (e.g., logistic growth) plus unpredictable shocks.

### Model Specification

**Likelihood:**
```
C_i ~ NegativeBinomial(μ_i, φ_i)
```

**Decomposition:**
```
log(μ_i) = g(year_i; θ_growth) + δ_i
```
where:
- g(·) is a parametric growth function (logistic or Gompertz)
- δ_i are flexible deviations from parametric trend

**Parametric growth component (Logistic):**
```
g(year; θ_growth) = log(L / (1 + exp(-k · (year - t₀))))
```
Parameters:
- L: carrying capacity (upper asymptote)
- k: growth rate
- t₀: inflection point

**Non-parametric deviation component:**
```
δ ~ GP(0, k_δ(x, x'))
```
with Matern kernel:
```
k_δ(x, x') = σ² · Matern_ν(x, x'; ℓ)
```

**Time-varying overdispersion:**
```
log(φ_i) = φ_0 + φ_1 · year_i
```
(Allows variance to change over time, as EDA suggests)

**Priors:**
```
log(L) ~ Normal(log(300), 0.5)    # Carrying capacity (slightly above max observed)
k ~ Lognormal(0, 1)                # Growth rate (positive)
t₀ ~ Normal(0, 1)                  # Inflection around middle of range
σ² ~ Half-Normal(0, 0.5)           # Deviation amplitude (should be small)
ℓ ~ Inverse-Gamma(3, 3)            # Deviation lengthscale
φ_0 ~ Half-Normal(0, 10)           # Baseline overdispersion
φ_1 ~ Normal(0, 1)                 # Trend in overdispersion
```

### Why This Might Work
- **Best of both worlds**: Structured growth + flexibility for surprises
- **Scientific plausibility**: Many processes follow logistic/Gompertz with noise
- **Interpretable**: Can separate "what we expect" (g) from "surprises" (δ)
- **Handles time-varying variance**: φ_i models changing overdispersion
- **Can detect model failure**: If δ is large, parametric component is wrong

### Why This Might FAIL (Falsification Criteria)

**I will abandon this model if:**

1. **Deviations dominate growth function**
   - Evidence: Posterior on σ² is large relative to growth component magnitude
   - Interpretation: Parametric form is wrong, deviations doing all the work
   - Pivot to: Pure non-parametric (Model 1 or 2)

2. **Logistic doesn't fit (inflection point missing)**
   - Evidence: Posterior on t₀ pushes to boundary (±1.7), no clear inflection
   - Interpretation: Data in exponential phase, not S-curve
   - Pivot to: Exponential growth + GP deviations, or pure GP

3. **Time-varying overdispersion not needed**
   - Evidence: Posterior on φ_1 includes zero, wide credible interval
   - Interpretation: Over-parameterization, constant φ sufficient
   - Pivot to: Simpler model without time-varying dispersion

4. **Computational nightmare**
   - Evidence: Model won't converge, extreme divergences, days to fit
   - Interpretation: Too complex for n=40, geometry too difficult
   - Pivot to: Simpler components (parametric only or GP only)

5. **Worse fit than pure parametric or pure GP**
   - Evidence: LOO-IC not competitive with simpler alternatives
   - Interpretation: Hybrid complexity not justified
   - Pivot to: Best-performing simpler model

### Stress Test Design
**Test 1**: Fix parametric component at MLE, fit only deviations
**Expectation**: Deviations should be small, centered at zero
**Failure**: Large systematic deviations = parametric form wrong

**Test 2**: Compare to pure logistic (no deviations)
**Expectation**: Hybrid should improve fit modestly (LOO-IC better by 5-20)
**Failure**: Huge improvement = logistic is terrible; no improvement = deviations unnecessary

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

with pm.Model() as semi_parametric_model:
    # Data
    year_data = pm.Data("year", year)
    C_data = pm.Data("C", C)

    # Parametric growth component (logistic)
    log_L = pm.Normal("log_L", mu=np.log(300), sigma=0.5)
    L = pm.Deterministic("L", pm.math.exp(log_L))
    k = pm.Lognormal("k", mu=0, sigma=1)
    t0 = pm.Normal("t0", mu=0, sigma=1)

    logistic = pm.Deterministic(
        "logistic",
        pm.math.log(L / (1 + pm.math.exp(-k * (year_data - t0))))
    )

    # Non-parametric deviation component (GP)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=0.5)
    ell_delta = pm.InverseGamma("ell_delta", alpha=3, beta=3)

    # Matern 3/2 covariance
    cov_func = sigma_delta**2 * pm.gp.cov.Matern32(1, ls=ell_delta)
    gp = pm.gp.Latent(cov_func=cov_func)
    delta = gp.prior("delta", X=year_data[:, None])

    # Combined mean
    eta = logistic + delta
    mu = pm.Deterministic("mu", pm.math.exp(eta))

    # Time-varying overdispersion
    phi_0 = pm.HalfNormal("phi_0", sigma=10)
    phi_1 = pm.Normal("phi_1", mu=0, sigma=1)
    log_phi = phi_0 + phi_1 * year_data
    phi = pm.Deterministic("phi", pm.math.exp(log_phi))

    # Likelihood
    alpha = mu**2 / (pm.math.maximum(phi, 1e-6) - mu)  # NegBinomial parameterization
    obs = pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=C_data)

    # Sample
    trace = pm.sample(2000, tune=1000, target_accept=0.95,
                      return_inferencedata=True)
```

**Why PyMC here**: GP + complex transformations easier in PyMC's syntax than Stan

### Expected Insights
- **Deviation magnitude**: Is reality close to logistic or far from it?
- **Growth parameters**: L, k, t₀ tell us about underlying process
- **Variance evolution**: Does φ increase over time (multiplicative noise)?
- **Model adequacy**: Can we explain data with simple structure + small noise?

---

## Model Comparison Strategy

### Criteria for Success

**Primary metrics:**
1. **LOO-CV (Leave-One-Out Cross-Validation)**
   - LOO-IC: Lower is better
   - Pareto k diagnostic: All k < 0.7 (reliable LOO)
   - Standard error of difference between models

2. **Posterior Predictive Checks**
   - Visual: Observed data within 95% predictive intervals?
   - Quantitative: Bayesian p-values for key statistics (mean, variance, max)
   - Overdispersion captured: Compare observed vs predicted variance-to-mean ratio

3. **Predictive Performance**
   - Hold out last 10 observations
   - RMSE, MAE on held-out data
   - Coverage: % of holdout points in 95% prediction intervals

**Secondary metrics:**
4. **Computational efficiency**
   - Time to fit (should be < 30 minutes for n=40)
   - Convergence: R̂ < 1.01, ESS > 400
   - Divergences: < 1% of samples

5. **Scientific plausibility**
   - Parameter estimates make sense?
   - Extrapolation behavior reasonable?
   - Uncertainty calibrated appropriately?

### Expected Ranking (My Predictions)

**Best case scenario**: Model 2 (P-splines) wins
- Reason: Goldilocks flexibility for n=40
- GP might overfit, Semi-parametric might be too complex
- Splines computationally efficient, well-tested

**Surprising outcome 1**: Model 1 (GP) wins decisively
- Interpretation: True function is very complex, needs full flexibility
- Action: Trust the GP, but check for overfitting on holdout data

**Surprising outcome 2**: Simple parametric (Designer 1) wins
- Interpretation: Apparent complexity in EDA was noise/small sample artifact
- Action: Abandon flexible models, embrace simplicity

**Disaster scenario**: None of my models converge
- Interpretation: Flexible models + count likelihood + n=40 = bad geometry
- Action: Retreat to Designer 1's simpler models or Designer 3's structured approach

---

## Red Flags and Decision Points

### Stop and Reconsider Everything If:

1. **All three models show prior-data conflict**
   - Evidence: Posterior on key parameters at prior boundaries
   - Implication: My priors encode wrong information OR model class fundamentally wrong
   - Action: Examine data more carefully, consult domain experts

2. **Posterior predictive checks fail dramatically**
   - Evidence: Observed data far outside predictive distribution (Bayesian p-value < 0.01)
   - Implication: Missing crucial structure (changepoints? external shocks?)
   - Action: Exploratory analysis for discrete events, regime changes

3. **Holdout prediction is terrible (RMSE > 50)**
   - Evidence: Cannot predict even one step ahead accurately
   - Implication: Models are overfitting or missing key dynamics
   - Action: Try Designer 3's temporal models (state-space, AR)

4. **Lengthscales/smoothness suggest nearly linear**
   - Evidence: All flexible models collapse to straight lines
   - Implication: Non-linearity in EDA was spurious
   - Action: Use simple linear/quadratic model from Designer 1

5. **Computational issues across all implementations**
   - Evidence: Divergences, non-convergence, extreme run times
   - Implication: Problem is fundamentally hard for MCMC OR data is pathological
   - Action: Try variational inference, different parameterizations, or give up on Bayesian

### Decision Point: When to Pivot Model Classes

**After fitting all 3 models:**

- If Model 1 LOO-IC < Model 2 by > 20: True complexity needs full GP
- If Model 2 LOO-IC < Model 1 by > 20: Splines sufficient, GP overfitting
- If Model 3 deviations are tiny: Parametric model sufficient (Designer 1 wins)
- If Model 3 deviations are huge: Parametric form wrong (Models 1-2 win)
- If Designer 1's models beat all of mine: Abandon flexibility, embrace parametric

---

## Alternative Approaches If Initial Models Fail

### Backup Plan 1: Simpler GP
If GP-NegBin has computational issues:
- Use log-normal instead of negative binomial (Gaussian likelihood easier)
- Use sparse GP approximation (inducing points)
- Use squared exponential kernel with fixed lengthscale from CV

### Backup Plan 2: Linear Spline with Knots
If P-splines too complex:
- Reduce to linear splines (degree=1)
- Fewer knots (K=4 instead of 8)
- Stronger smoothness penalty (informative prior on τ)

### Backup Plan 3: Piecewise Parametric
If semi-parametric uninterpretable:
- Split time into 2-3 segments
- Fit separate parametric curves per segment
- Hierarchical structure on parameters across segments

### Backup Plan 4: Abandon Flexibility
If everything fails:
- Accept that n=40 is too small for complex models
- Use Designer 1's negative binomial with quadratic trend
- Focus on getting uncertainty quantification right

---

## What Each Model Will Teach Us

### Model 1 (GP-NegBin) Teaches:
- **Is the true function highly irregular?** (Short lengthscale suggests yes)
- **How uncertain are we about the trend shape?** (GP posterior width)
- **Can we predict out-of-sample?** (GP extrapolation test)
- **Is complexity warranted?** (Compare to simpler models)

### Model 2 (P-splines) Teaches:
- **How much smoothness does data support?** (τ parameter)
- **Where does curvature occur?** (Examine coefficients θⱼ near knots)
- **Is function locally adaptive?** (Different slope in different regions)
- **Is there a sweet spot for complexity?** (Knot count sensitivity)

### Model 3 (Semi-parametric) Teaches:
- **Is there a simple underlying process?** (Logistic component fit)
- **How surprising is reality?** (Deviation magnitude σ²)
- **Does variance increase over time?** (φ_1 parameter)
- **Can we decompose structure and noise?** (g vs δ contributions)

### Cross-Model Insights:
- **Model agreement**: If all 3 give similar predictions, high confidence
- **Model disagreement**: Where do they differ? That's where uncertainty lives
- **Complexity ladder**: Do more complex models improve fit enough to justify them?
- **Robustness**: Are conclusions sensitive to model choice?

---

## Prior Justification and Sensitivity

### Key Prior Choices

**GP lengthscale (ρ ~ InverseGamma(5,5)):**
- Mean ≈ 1.25, implies correlations decay over ~1 unit of standardized year
- Encodes weak belief that function is smooth but not constant
- Sensitivity test: Try Gamma(2,2) (shorter), compare posteriors

**Spline smoothness (τ ~ HalfNormal(0,1)):**
- Mode at 0 (prefers smooth), but allows wiggle if data demands
- Typical choice in spline literature
- Sensitivity test: Try τ ~ HalfNormal(0,0.5) (stronger smoothing)

**Logistic capacity (log(L) ~ Normal(log(300), 0.5)):**
- Mean of 300 is slightly above observed max (272)
- 0.5 SD means 95% CI ≈ [180, 500] - wide enough to adapt
- Sensitivity test: Try vague prior Normal(log(300), 2), see if posterior changes

**Overdispersion (φ ~ HalfNormal(0,10)):**
- Very vague (95% prior mass up to φ=20)
- Observed variance suggests φ ~ 5-15 range
- Sensitivity test: Try φ ~ HalfNormal(0,5), check if posterior hits boundary

### Prior Predictive Checks

**Before fitting, I will:**
1. Sample from prior predictive distribution
2. Check if generated counts are plausible (e.g., not negative, not millions)
3. Verify prior doesn't rule out observed data
4. If prior predictive is unreasonable, adjust priors

---

## Computational Considerations

### Expected Run Times (on modern laptop)
- **Model 1 (GP-NegBin)**: 15-30 minutes (GP Cholesky decomposition expensive)
- **Model 2 (P-splines)**: 5-10 minutes (simpler linear algebra)
- **Model 3 (Semi-parametric)**: 30-60 minutes (most complex)

### Sampling Strategy
- **Warmup**: 1000 iterations (tune step size, mass matrix)
- **Sampling**: 2000 iterations post-warmup
- **Chains**: 4 parallel chains
- **Thinning**: None (keep all samples for ESS calculation)
- **Target accept**: 0.90 for Models 1-2, 0.95 for Model 3

### Convergence Diagnostics
- **R̂ < 1.01** for all parameters (between-chain variance = within-chain)
- **ESS > 400** for all parameters (effective sample size after autocorrelation)
- **Divergences < 1%** (geometry not too pathological)
- **Tree depth maxed < 5%** (step size adequate)

### If Sampling Fails
1. Try non-centered parameterization (reparameterize GPs, splines)
2. Increase target_accept to 0.99 (smaller steps, more robust)
3. Run longer chains (more warmup, more samples)
4. Simplify model (fewer knots, simpler kernel, drop components)

---

## Summary: My Modeling Stance

**Philosophy**: Flexible models are tools for discovery, not destinations. If data is simple, flexible models should discover that simplicity (not force complexity).

**Success criteria**:
- Models converge and pass diagnostics
- LOO-CV better than simple alternatives
- Posterior predictive checks pass
- Predictions on holdout data reasonable
- Scientific interpretation plausible

**Failure criteria** (time to pivot):
- Computational disaster (divergences, non-convergence)
- Overfitting (worse out-of-sample than simple models)
- Collapse to simplicity (flexible components unused)
- Prior-data conflict (fighting the data)
- Nonsensical extrapolation

**Final thoughts**: I expect Model 2 (P-splines) to win because n=40 is small and splines offer a good flexibility/complexity tradeoff. But I'm prepared to be wrong. If Designer 1's simple quadratic negative binomial beats all my models, I'll happily admit that flexibility wasn't needed. The goal is truth, not complexity for its own sake.

---

## Files and Code References

**This document**: `/workspace/experiments/designer_2/proposed_models.md`

**Implementation code** (to be written during model fitting phase):
- `/workspace/experiments/designer_2/model1_gp_negbin.stan`
- `/workspace/experiments/designer_2/model2_pspline_negbin.stan`
- `/workspace/experiments/designer_2/model3_semiparametric.py` (PyMC)
- `/workspace/experiments/designer_2/fit_all_models.py` (orchestration script)

**Data**: `/workspace/data/data.csv`
**EDA report**: `/workspace/eda/eda_report.md`

---

**Designer 2 (Non-parametric Specialist) - Experiment Plan Complete**
**Date**: 2025-10-29
**Status**: Ready for model fitting phase
