# Bayesian Model Designs: Structural Change & Nonlinear Patterns
## Model Designer 3 - Experiment Plan

**Designer**: Model Designer 3 (Structural Change & Nonlinear Specialist)
**Date**: 2025-10-30
**Focus**: Changepoint detection, regime shifts, nonlinear trends, hierarchical time structure
**Data**: 40 count observations with exponential growth, severe overdispersion, regime shifts

---

## Executive Summary

The EDA reveals **three critical features** demanding structural change models:

1. **Regime shift evidence**: 7.8× increase early→late, heterogeneous dispersion (0.68 → 13.11 → 7.23 ratios across periods)
2. **Nonlinear acceleration**: Quadratic R² = 0.964 vs linear 0.881 (8.3% improvement)
3. **Time-varying processes**: Dispersion patterns change dramatically (underdispersed → severely overdispersed)

**Key insight**: The data-generating process **changes over time**. Models assuming stationarity will fail.

This document proposes **three Bayesian model classes**, each with explicit falsification criteria and computational plans.

---

## Critical Assumptions & Falsification Philosophy

### What Would Make Me Abandon This Entire Approach?

1. **If changepoints are artifacts of measurement changes** (e.g., counting method changed)
   - Evidence: Check data provenance, collection methods
   - Test: If "changepoint" aligns exactly with known protocol change

2. **If apparent regime shift is just smooth polynomial**
   - Evidence: BIC/LOO strongly favors polynomial over changepoint
   - Test: Posterior changepoint has uniform distribution (no information)

3. **If overdispersion is measurement error, not process**
   - Evidence: Negative binomial theta → ∞ (converges to Poisson)
   - Test: Variance shrinks dramatically with different counting method

4. **If temporal dependence dominates structural change**
   - Evidence: AR(1) model outperforms all regime models
   - Test: Changepoint posteriors are diffuse and uninformative

### Red Flags That Trigger Model Class Change

- **Prior-posterior conflict**: Posterior pushed to extreme values despite reasonable priors
- **Divergent transitions in Stan**: Often indicates model misspecification, not just tuning
- **Extreme parameter estimates**: e.g., changepoint at t=1 or t=40 (boundary)
- **LOO warnings**: High Pareto-k values (> 0.7) indicate poor pointwise predictions
- **Posterior predictive checks fail**: Generated data looks nothing like observed

### Decision Points for Major Pivots

1. **After Model 1 (Piecewise NegBin)**: If no evidence for discrete changepoint, move to smooth transitions (Model 2)
2. **After Model 2 (Spline)**: If still poor fit, consider latent state-space or GP
3. **After Model 3 (Hierarchical)**: If all fail, reconsider data quality or measurement model

---

## Problem Formulation: Competing Hypotheses

### Hypothesis 1: Discrete Regime Shift
**Claim**: The process changed abruptly at unknown time τ (e.g., policy change, environmental shift)

**Evidence FOR**:
- Period means: 28.57 → 83.00 → 222.85 (not smooth!)
- Dispersion ratios: 0.68 → 13.11 → 7.23 (structural change)
- Visual inspection shows "elbow" around observation 20-24

**Evidence AGAINST**:
- Gradual transition could mimic discrete shift with n=40
- Human perception biased toward seeing discrete changes
- Polynomial might be simpler explanation (Occam's Razor)

**Falsification test**:
- Posterior for changepoint location should be **concentrated** (not diffuse)
- Model comparison: LOO(changepoint) >> LOO(polynomial) by >10 ELPD
- Posterior predictive: sharp transition in generated data

**I will abandon this if**: Changepoint posterior is uniform, or LOO favors polynomial

---

### Hypothesis 2: Smooth Nonlinear Acceleration
**Claim**: Growth rate increases smoothly (no discrete change), like adoption curves or epidemic spread

**Evidence FOR**:
- Quadratic R² = 0.964 (excellent fit)
- Log-scale suggests compounding growth
- Biological/social processes often smooth

**Evidence AGAINST**:
- Heterogeneous dispersion across periods (why would this change smoothly?)
- Period analysis shows **discontinuity**, not smooth transition
- Residual autocorrelation remains high even with quadratic term

**Falsification test**:
- Residual plots should show no regime-specific patterns
- Posterior predictive: smooth acceleration, no "jumps"
- Dispersion parameter should be stable over time (or smoothly varying)

**I will abandon this if**: Posterior predictive shows regime structure not captured by smooth trend

---

### Hypothesis 3: Hierarchical Time Structure with Latent States
**Claim**: Observable counts driven by latent "intensity" that evolves; dispersion varies with state

**Evidence FOR**:
- Time-varying dispersion (can't be captured by fixed theta)
- High autocorrelation even after detrending (latent structure?)
- Occasional large jumps (see changes plot)

**Evidence AGAINST**:
- Small sample (n=40) for complex latent structure
- May overfit with too many latent states
- Identifiability concerns (latent + dispersion + trend)

**Falsification test**:
- Latent states should be interpretable (low/medium/high intensity)
- WAIC penalty should not explode (overfitting check)
- Leave-one-out should not fail (identifiable predictions)

**I will abandon this if**: Latent states are non-identifiable or WAIC penalty > benefit

---

## Model Class 1: Piecewise Negative Binomial with Unknown Changepoint

### Scientific Motivation

Many real processes undergo **discrete structural changes**:
- Policy interventions (new regulation at specific time)
- Technology adoption (market penetration threshold)
- Ecological regime shifts (population crossing critical density)

The EDA evidence (7.8× mean increase, dispersion changes) suggests a **step-function** change, not gradual.

### Full Bayesian Specification

#### Likelihood
```
For observation i at time t_i:

C_i ~ NegativeBinomial2(mu_i, phi_regime[i])

log(mu_i) = beta0 + beta1_before × year_i × I(i < tau)
                  + beta1_after  × year_i × I(i >= tau)
                  + beta2_before × year_i^2 × I(i < tau)
                  + beta2_after  × year_i^2 × I(i >= tau)

regime[i] = 1  if i < tau, else 2

phi_regime[1] = phi_before   (dispersion before changepoint)
phi_regime[2] = phi_after    (dispersion after changepoint)
```

**Key features**:
- **Unknown changepoint** τ ∈ {10, ..., 30} (exclude extremes)
- **Regime-specific slopes** (growth rate changes)
- **Regime-specific dispersion** (matches EDA finding)
- **Continuity NOT enforced** (allow discontinuous jump)

#### Priors

**Changepoint location** (discrete uniform over plausible range):
```
tau ~ DiscreteUniform(10, 30)
```
*Rationale*: Exclude first/last 10 observations (need data to estimate regimes). Weakly informative: any midpoint equally likely a priori.

**Intercept**:
```
beta0 ~ Normal(4.5, 1.0)
```
*Rationale*: On log-scale, exp(4.5) ≈ 90, near observed mean. SD=1 allows factor of ~7× variation (exp(±2) range).

**Linear slopes** (regime-specific):
```
beta1_before ~ Normal(0, 1.5)
beta1_after  ~ Normal(0, 1.5)
```
*Rationale*: Centered at 0 (no trend a priori). SD=1.5 allows strong trends (exp(1.5) ≈ 4.5× per unit) but pulls toward parsimony. Regime-specific allows acceleration.

**Quadratic terms**:
```
beta2_before ~ Normal(0, 0.5)
beta2_after  ~ Normal(0, 0.5)
```
*Rationale*: Smaller prior SD (0.5 vs 1.5) because quadratic can create extreme extrapolation. Weakly regularizing.

**Dispersion parameters** (regime-specific):
```
phi_before ~ Gamma(2, 0.1)   # E[phi] = 20, mode near 10
phi_after  ~ Gamma(2, 0.1)   # Allow different overdispersion
```
*Rationale*: NegBin2 parameterization: var = mu + mu²/phi. Smaller phi = more overdispersion. Gamma(2, 0.1) has mean=20, allowing wide range (5-50+). **Crucially allows phi_before ≠ phi_after** to match EDA evidence.

**Alternative**: Could use hierarchical prior φ ~ Gamma(a, b) with hyperprior on (a, b), but with only 2 regimes, probably overkill.

#### Generated Quantities (for Model Comparison)

```stan
generated quantities {
  vector[N] log_lik;  // Pointwise log-likelihood for LOO
  vector[N] y_rep;    // Posterior predictive samples

  for (i in 1:N) {
    int regime = i < tau ? 1 : 2;
    real mu = exp(beta0 + beta1[regime] * year[i] + beta2[regime] * year_sq[i]);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu, phi[regime]);
    y_rep[i] = neg_binomial_2_rng(mu, phi[regime]);
  }

  // Also compute expected count at changepoint for interpretability
  real mu_before_tau = exp(beta0 + beta1_before * year[tau-1] + beta2_before * year_sq[tau-1]);
  real mu_after_tau  = exp(beta0 + beta1_after  * year[tau]   + beta2_after  * year_sq[tau]);
  real jump_size = mu_after_tau / mu_before_tau;  // Multiplicative jump
}
```

### Falsification Criteria

**I will abandon this model if**:

1. **Changepoint posterior is diffuse**
   - Test: Posterior SD(tau) > 5 observations (no information gained)
   - Interpretation: Data don't support discrete change

2. **Changepoint at boundary**
   - Test: P(tau < 15) > 0.95 or P(tau > 25) > 0.95
   - Interpretation: Model trying to degenerate to single regime

3. **Dispersion parameters not different**
   - Test: P(|phi_after - phi_before| < 2) > 0.80
   - Interpretation: No evidence for regime-specific dispersion

4. **LOO strongly favors polynomial**
   - Test: ΔLOO(polynomial - changepoint) > 10 ELPD
   - Interpretation: Smooth trend is simpler and better

5. **Posterior predictive check fails**
   - Test: Observed data outside 95% credible envelope for >20% of points
   - Interpretation: Model fundamentally misspecified

### Expected Posterior Distributions (if model correct)

- **tau**: Should concentrate around observation 20-24 (where visual "elbow" appears)
- **beta1_before < beta1_after**: Acceleration in linear term
- **phi_before > phi_after**: More overdispersion in later period (EDA shows opposite in early, but overall later has more variance)
- **jump_size**: Should be ~2-4× (discontinuous jump at changepoint)

### Computational Plan

**Implementation**: Stan via CmdStanPy

**Key Stan features needed**:
- Discrete parameter (tau) handled via marginalization or discrete sampling
- Use `target +=` to increment log-probability over changepoint values
- Or: Vectorize over all possible tau values, then sample using `categorical`

**Stan code structure**:
```stan
parameters {
  real<lower=3, upper=5> beta0;  // Constrain to reasonable log-scale
  real beta1_before;
  real beta1_after;
  real beta2_before;
  real beta2_after;
  real<lower=0> phi_before;
  real<lower=0> phi_after;
}

transformed parameters {
  vector[N] log_mu[K];  // K = tau_max - tau_min + 1 possible changepoints
  vector[K] lp;         // Log-probability for each changepoint

  for (k in 1:K) {
    int tau_k = tau_min + k - 1;
    for (i in 1:N) {
      if (i < tau_k) {
        log_mu[k][i] = beta0 + beta1_before * year[i] + beta2_before * year_sq[i];
      } else {
        log_mu[k][i] = beta0 + beta1_after * year[i] + beta2_after * year_sq[i];
      }
    }

    // Compute log-probability for this changepoint
    lp[k] = 0;
    for (i in 1:N) {
      int regime = i < tau_k ? 1 : 2;
      lp[k] += neg_binomial_2_lpmf(C[i] | exp(log_mu[k][i]), phi[regime]);
    }
  }
}

model {
  // Priors
  beta0 ~ normal(4.5, 1.0);
  beta1_before ~ normal(0, 1.5);
  beta1_after ~ normal(0, 1.5);
  beta2_before ~ normal(0, 0.5);
  beta2_after ~ normal(0, 0.5);
  phi_before ~ gamma(2, 0.1);
  phi_after ~ gamma(2, 0.1);

  // Marginalize over tau (discrete uniform prior implicit)
  target += log_sum_exp(lp);
}

generated quantities {
  int<lower=tau_min, upper=tau_max> tau;
  tau = tau_min + categorical_rng(softmax(lp)) - 1;

  // ... (log_lik and y_rep as above, using sampled tau)
}
```

**Sampling settings**:
- 4 chains, 2000 iterations (1000 warmup)
- `adapt_delta = 0.95` (changepoint models can have difficult geometry)
- Monitor: Rhat < 1.01, ESS > 400, no divergences

**Computational complexity**: O(N × K) per iteration, where K ≈ 20 changepoint candidates. Manageable.

**Identifiability concerns**:
- Quadratic + changepoint: Risk of overfitting with n=40
- Mitigation: Regularizing priors on beta2, compare to simpler version (no quadratic)
- Monitor: Posterior correlation matrix (if beta2 and tau highly correlated, problem)

### Stress Test: Try to Break the Model

**Adversarial data simulation**:
1. Generate data from **smooth cubic** (no changepoint)
2. Fit this model → Should NOT find strong changepoint evidence
3. If posterior tau is concentrated, model is "hallucinating" structure

**Extreme case**:
1. Fit model to **first 25 observations only**
2. Posterior tau should be diffuse (not enough data for second regime)
3. If concentrated, model is overconfident

### Alternative Variants to Consider

**Variant 1A: Linear-only (no quadratic)**
- Remove beta2 terms
- Test if quadratic necessary or just capturing changepoint curvature
- Faster, fewer parameters

**Variant 1B: Shared dispersion**
- phi_before = phi_after = phi
- Test if regime-specific dispersion justified
- Nested model (easier comparison)

**Variant 1C: Multiple changepoints**
- Allow tau1, tau2 (three regimes)
- Only if single changepoint inadequate
- Risk: severe overfitting with n=40

---

## Model Class 2: Smoothly Varying Trend (Cubic Spline) with Time-Varying Dispersion

### Scientific Motivation

Many growth processes are **smooth but nonlinear**:
- Logistic growth (S-curves in adoption, epidemics)
- Gompertz curves (tumor growth, bacterial cultures)
- Polynomial approximations to complex dynamics

The EDA shows excellent quadratic fit (R²=0.964), suggesting **smooth acceleration** might suffice. The apparent "regime shift" could be human pattern-seeking on limited data.

### Why This Might Be Right (vs. Changepoint)

1. **Occam's Razor**: Fewer assumptions than discrete breakpoint
2. **Biological plausibility**: Many processes smooth at high temporal resolution
3. **Polynomial parsimony**: Quadratic is 2 parameters, changepoint is 5+ parameters
4. **Measurement**: With annual data, smooth monthly changes look discrete

### Full Bayesian Specification

#### Likelihood
```
C_i ~ NegativeBinomial2(mu_i, phi(t_i))

log(mu_i) = f(year_i)   // Flexible smooth function

f(year) = sum_{k=1}^K beta_k × B_k(year)   // B-spline basis

phi(t) = exp(gamma0 + gamma1 × year)   // Time-varying dispersion (log-linear)
```

**Key features**:
- **B-spline basis**: Piecewise cubic polynomials, C² continuous (smooth!)
- **K knots**: Place at quantiles (e.g., K=5 knots at 20%, 40%, 60%, 80%)
- **Time-varying dispersion**: Allows gradual change in variance structure
- **No discontinuities**: Everything smooth by construction

#### Why Splines vs. Simple Polynomial?

**Cubic spline advantages**:
- Local control (knot placement at suspected inflection points)
- Prevents wild extrapolation (polynomial tail behavior)
- More flexible than global polynomial
- Standard in GAMs (proven approach)

**Polynomial alternative** (simpler):
```
log(mu_i) = beta0 + beta1 × year + beta2 × year² + beta3 × year³
```
- Fewer parameters (4 vs. 5-8 for splines)
- Global influence (can be bad for extrapolation)
- Try this first as baseline

#### Priors

**For Polynomial Version**:
```
beta0 ~ Normal(4.5, 1.0)     # Intercept (log-scale)
beta1 ~ Normal(0, 1.5)       # Linear term
beta2 ~ Normal(0, 0.5)       # Quadratic (smaller SD, prevent extremes)
beta3 ~ Normal(0, 0.1)       # Cubic (heavily regularized)
```
*Rationale*: Hierarchical shrinkage: higher-order terms more constrained. Cubic rarely needed, so strong prior toward zero.

**For Spline Version**:
```
beta ~ Normal(0, tau)         # Spline coefficients
tau ~ Exponential(1)          # Global shrinkage (automatic relevance determination)
```
*Rationale*: Horseshoe-like prior allows some coefficients to be large (important knots) while shrinking others. Exponential(1) on tau creates sparsity.

**Dispersion function**:
```
gamma0 ~ Normal(3.0, 1.0)    # Baseline log-dispersion: exp(3) = 20
gamma1 ~ Normal(0, 0.5)      # Change in dispersion over time
```
*Rationale*: Allow dispersion to increase or decrease with time. gamma1=0 is constant dispersion (null hypothesis). SD=0.5 allows factor of ~3× change over time range.

**Alternative**: Non-parametric dispersion using GP (see Model 3)

#### Generated Quantities
```stan
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N] phi_t;  // Time-varying dispersion for each observation

  for (i in 1:N) {
    real mu = exp(f[i]);  // f computed in transformed parameters
    phi_t[i] = exp(gamma0 + gamma1 * year[i]);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu, phi_t[i]);
    y_rep[i] = neg_binomial_2_rng(mu, phi_t[i]);
  }

  // Compute effective degrees of freedom (for complexity penalty)
  real edf = sum(diagonal(Omega));  // Omega = smoother matrix (computed in Stan)
}
```

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior predictive shows regime structure**
   - Test: Plot residuals colored by period → systematic pattern
   - Interpretation: Smooth trend missing discrete structure

2. **Dispersion time-trend inadequate**
   - Test: Residual variance still changes abruptly across periods
   - Interpretation: Need regime-specific dispersion, not smooth change

3. **LOO much worse than changepoint model**
   - Test: ΔLOO(changepoint - spline) > 10 ELPD
   - Interpretation: Data prefer discrete change

4. **Cubic coefficient extreme**
   - Test: |beta3| > 1.0 (on standardized scale)
   - Interpretation: Overfitting, need higher-order or different form

5. **Time-varying phi not justified**
   - Test: Credible interval for gamma1 includes zero comfortably
   - Interpretation: Constant dispersion simpler

### Expected Posterior Distributions (if model correct)

- **beta1 > 0**: Positive linear trend
- **beta2 > 0**: Positive curvature (acceleration)
- **beta3 ≈ 0**: Cubic probably unnecessary (should be small)
- **gamma1 > 0**: Dispersion increases over time (matches EDA: later variance larger)
- **Posterior predictive**: Smooth acceleration, no "elbow"

### Computational Plan

**Implementation**: Stan via CmdStanPy

**Polynomial version** (try first):
```stan
parameters {
  real beta0;
  real beta1;
  real beta2;
  real beta3;
  real gamma0;
  real gamma1;
}

transformed parameters {
  vector[N] log_mu;
  vector[N] phi;

  for (i in 1:N) {
    log_mu[i] = beta0 + beta1*year[i] + beta2*year_sq[i] + beta3*year_cub[i];
    phi[i] = exp(gamma0 + gamma1*year[i]);
  }
}

model {
  // Priors
  beta0 ~ normal(4.5, 1.0);
  beta1 ~ normal(0, 1.5);
  beta2 ~ normal(0, 0.5);
  beta3 ~ normal(0, 0.1);
  gamma0 ~ normal(3.0, 1.0);
  gamma1 ~ normal(0, 0.5);

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // ... (as above)
}
```

**Spline version** (if polynomial inadequate):
- Use `mgcv` or `patsy` in Python to generate B-spline basis matrix
- Pass as data to Stan
- More complex but more flexible

**Sampling settings**:
- 4 chains, 2000 iterations
- `adapt_delta = 0.90` (should be easier than changepoint)
- Monitor divergences (time-varying phi can create funnels)

**Computational complexity**: O(N) per iteration (simple!), much faster than changepoint

**Identifiability concerns**:
- High correlation between beta1, beta2, beta3 (multicollinearity)
- Mitigation: Orthogonal polynomials (use `poly()` transformation)
- Or: Regularizing priors (already included)

### Stress Test

**Adversarial simulation**:
1. Generate data with **discrete jump** (true changepoint)
2. Fit smooth model → Should show poor fit at transition
3. Posterior predictive check should fail (observed outside envelope)

**Extreme extrapolation**:
1. Fit to first 30 observations
2. Predict last 10
3. If cubic goes wild (negative counts), model unstable

### Alternative Variants

**Variant 2A: Quadratic only**
- Remove beta3 (simpler)
- Test if cubic necessary
- Less overfitting risk

**Variant 2B: Constant dispersion**
- gamma1 = 0 (fixed)
- Test if time-varying dispersion justified
- Fewer parameters

**Variant 2C: Log-quadratic dispersion**
- phi(t) = exp(gamma0 + gamma1×year + gamma2×year²)
- Allow non-monotonic dispersion change
- More flexible, more parameters

---

## Model Class 3: Hierarchical Temporal Segmentation (Latent Regime Model)

### Scientific Motivation

What if the truth is **neither discrete jump nor smooth curve**, but something in between?

**Latent regime model**: Observable counts driven by unobserved "state" that evolves stochastically:
- State = "low growth regime" vs. "high growth regime"
- Transitions between states are **probabilistic**, not deterministic
- Dispersion depends on current state
- Observations within same state are more similar (hierarchical clustering)

**Real-world examples**:
- Economic cycles (recession/expansion states)
- Disease surveillance (endemic/epidemic states)
- Population dynamics (low/high density states)

This model **hedges between hypotheses 1 and 2**: allows regime structure but with soft transitions.

### Why This Might Be Right

1. **Heterogeneous dispersion**: EDA shows 0.68 → 13.11 → 7.23 (three distinct patterns)
2. **Autocorrelation**: High ACF even after detrending suggests latent structure
3. **Occasional jumps**: Large count changes (see EDA Δ plots) suggest state transitions
4. **Uncertainty**: With n=40, discrete changepoint may be too rigid

### Full Bayesian Specification

#### Likelihood (Hierarchical Structure)
```
Level 1 (Observations):
C_i ~ NegativeBinomial2(mu_i, phi[z_i])

log(mu_i) = alpha[z_i] + beta[z_i] × year_i

Level 2 (Latent States):
z_i ~ Categorical(theta_i)

theta_i = softmax(gamma0 + gamma1 × year_i)  // State probability evolves with time

Level 3 (Hyperpriors):
alpha[k] ~ Normal(mu_alpha, sigma_alpha)  // Partial pooling across states
beta[k]  ~ Normal(mu_beta, sigma_beta)
phi[k]   ~ Gamma(a_phi, b_phi)
```

**Key features**:
- **K = 3 latent states** (low/medium/high growth regimes)
- **State-specific parameters**: intercept, slope, dispersion
- **Soft transitions**: State probability changes smoothly with time
- **Hierarchical priors**: States share information (partial pooling)

#### State Transition Model

**Time-varying state probabilities**:
```
logit(P(z_i = k | year_i)) = gamma[k,0] + gamma[k,1] × year_i
```

This creates a **multinomial logistic regression** for state membership:
- Early years: High P(state=1), low P(state=3)
- Late years: Low P(state=1), high P(state=3)
- Middle years: Mixed probabilities

**Alternative**: Hidden Markov Model (HMM) with Markov transitions:
```
P(z_i | z_{i-1}) = Π[z_{i-1}, z_i]   // Transition matrix
```
- More realistic for truly sequential states
- Harder to estimate with n=40
- Try this if time-varying logistic inadequate

#### Priors

**Hyperparameters** (state-level means):
```
mu_alpha ~ Normal(4.0, 1.0)   # Average intercept across states
mu_beta  ~ Normal(1.0, 0.5)   # Average slope (positive growth expected)
```

**Hierarchical variance** (how much states differ):
```
sigma_alpha ~ Exponential(1)  # Small = states similar, large = states very different
sigma_beta  ~ Exponential(2)  # Expect slopes more similar than intercepts
```

**State-specific parameters**:
```
alpha[k] ~ Normal(mu_alpha, sigma_alpha)  for k = 1, 2, 3
beta[k]  ~ Normal(mu_beta, sigma_beta)    for k = 1, 2, 3
phi[k]   ~ Gamma(2, 0.1)                   # As before, allows different dispersion
```

**State transition parameters**:
```
gamma[k,0] ~ Normal(0, 2)     # Baseline state preference (weakly informative)
gamma[k,1] ~ Normal(0, 1)     # How state probability changes with time
```
*Rationale*: Centered at 0 (no preference a priori). Large SD (2 for intercept) allows strong state separation if data support it.

**Constraints for identifiability**:
```
gamma[1,0] = 0  # Reference category (state 1)
gamma[1,1] = 0
```
This identifies the model (multinomial logit needs reference).

#### Generated Quantities
```stan
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  simplex[K] state_prob[N];  // Posterior state probabilities for each observation
  int<lower=1,upper=K> z_rep[N];  // Most likely state (MAP)

  for (i in 1:N) {
    // Compute state probabilities
    vector[K] logit_prob;
    for (k in 1:K) {
      logit_prob[k] = gamma[k,0] + gamma[k,1] * year[i];
    }
    state_prob[i] = softmax(logit_prob);

    // Marginalize over states for log_lik (mixture)
    vector[K] lp_state;
    for (k in 1:K) {
      real mu = exp(alpha[k] + beta[k] * year[i]);
      lp_state[k] = log(state_prob[i][k]) + neg_binomial_2_lpmf(C[i] | mu, phi[k]);
    }
    log_lik[i] = log_sum_exp(lp_state);

    // Sample from mixture for posterior predictive
    int z = categorical_rng(state_prob[i]);
    real mu = exp(alpha[z] + beta[z] * year[i]);
    y_rep[i] = neg_binomial_2_rng(mu, phi[z]);
    z_rep[i] = z;
  }
}
```

### Falsification Criteria

**I will abandon this model if**:

1. **States not separable**
   - Test: Posterior state probabilities always ~uniform (1/3, 1/3, 1/3)
   - Interpretation: No evidence for latent structure, use simpler model

2. **WAIC penalty explodes**
   - Test: p_WAIC > 20 (effective parameters >> actual parameters)
   - Interpretation: Overfitting, model too flexible

3. **LOO has many high Pareto-k**
   - Test: >25% of observations have k > 0.7
   - Interpretation: Influential points, poor generalization

4. **States temporally scrambled**
   - Test: Most likely state sequence jumps randomly (1→3→1→2→1→...)
   - Interpretation: Model not capturing temporal structure, just noise

5. **Hierarchical shrinkage extreme**
   - Test: sigma_alpha, sigma_beta → 0 (states identical)
   - Interpretation: No heterogeneity, use single-state model

### Expected Posterior Distributions (if model correct)

- **State sequence**: 1→1→1→...→2→2→...→3→3→... (mostly monotonic)
- **alpha[1] < alpha[2] < alpha[3]**: States ordered by intensity
- **beta[1] ≤ beta[2] ≤ beta[3]**: Possibly increasing growth rates
- **phi pattern**: Will reveal which state(s) have high dispersion
- **State probabilities**: Sharp transitions (P ≈ 1 for dominant state)

### Computational Plan

**Implementation**: Stan via CmdStanPy (challenging!)

**Stan code challenges**:
- Discrete latent states (z_i) require **marginalization**
- Use `log_sum_exp` trick to integrate out states
- Computationally expensive: O(N × K²) per iteration

**Stan code sketch**:
```stan
parameters {
  real mu_alpha;
  real mu_beta;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  vector[K] alpha;
  vector[K] beta;
  vector<lower=0>[K] phi;
  matrix[K-1, 2] gamma_raw;  // Excluding reference category
}

transformed parameters {
  matrix[K, 2] gamma;
  gamma[1] = [0, 0]';  // Reference category
  gamma[2:K] = gamma_raw;
}

model {
  // Hyperpriors
  mu_alpha ~ normal(4.0, 1.0);
  mu_beta ~ normal(1.0, 0.5);
  sigma_alpha ~ exponential(1);
  sigma_beta ~ exponential(2);

  // State-specific priors
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);
  phi ~ gamma(2, 0.1);

  // Transition parameters
  to_vector(gamma_raw) ~ normal(0, 1);

  // Likelihood (marginalized over states)
  for (i in 1:N) {
    vector[K] lp_state;
    vector[K] logit_prob;
    for (k in 1:K) {
      logit_prob[k] = gamma[k,1] + gamma[k,2] * year[i];
    }
    vector[K] state_prob = softmax(logit_prob);

    for (k in 1:K) {
      real mu = exp(alpha[k] + beta[k] * year[i]);
      lp_state[k] = log(state_prob[k]) + neg_binomial_2_lpmf(C[i] | mu, phi[k]);
    }
    target += log_sum_exp(lp_state);
  }
}

// ... generated quantities as above
```

**Sampling settings**:
- 4 chains, 3000 iterations (1500 warmup) - need longer for convergence
- `adapt_delta = 0.99` (mixture models have difficult geometry)
- `max_treedepth = 12` (may need deeper trees)
- **Expect**: Some divergences likely, may need reparameterization

**Computational complexity**:
- O(N × K²) per iteration, with K=3 and N=40: ~480 operations
- Marginalization makes this ~10× slower than simple models
- Total runtime: ~30-60 minutes

**Identifiability concerns** (serious!):
- **Label switching**: States can swap labels (alpha[1] ↔ alpha[2])
- Mitigation: Order constraints (alpha[1] < alpha[2] < alpha[3])
- Or: Post-hoc label switching correction
- **State collapse**: Model may use only 1-2 of 3 states
- Monitor: Posterior P(state=k) for each k (should be substantial for all)

### Stress Test

**Adversarial simulation**:
1. Generate data from **simple polynomial** (no states)
2. Fit this model → Should collapse to K=1 (all states identical)
3. If states still separate, model is overfitting noise

**State recovery test**:
1. Simulate data with **known states** (e.g., state 1 for i<15, state 2 for i≥15)
2. Fit model → Should recover true state sequence (after resolving label switching)
3. If not, model is misspecified or non-identifiable

### Alternative Variants

**Variant 3A: Two states only**
- K = 2 instead of 3
- Simpler, less overfitting risk
- Test if three states justified

**Variant 3B: Hidden Markov Model**
- Replace time-varying logistic with Markov transitions
- P(z_i | z_{i-1}, Π)
- More realistic temporal dependence
- Harder to estimate

**Variant 3C: Add quadratic terms**
- log(mu_i) = alpha[z_i] + beta1[z_i]×year + beta2[z_i]×year²
- Allow nonlinear trends within states
- Even more parameters (dangerous with n=40)

---

## Model Comparison Strategy

### Phase 1: Fit All Three Model Classes

**Order of fitting** (simplest to most complex):
1. Model 2A: Polynomial (quadratic, constant dispersion)
2. Model 2B: Polynomial with time-varying dispersion
3. Model 1A: Piecewise linear (no quadratic)
4. Model 1B: Piecewise quadratic
5. Model 3A: Two-state hierarchical
6. Model 3B: Three-state hierarchical

**Stopping rule**: If simple polynomial (2A) has excellent fit and passes all diagnostics, **STOP**. Don't fit complex models unnecessarily.

### Phase 2: Posterior Predictive Checks

For each model, generate:
1. **Posterior predictive distribution** (y_rep)
2. **Test statistics**:
   - T1: Mean by period (Early, Middle, Late) → Check regime structure
   - T2: Variance by period → Check dispersion heterogeneity
   - T3: Maximum count → Check tail behavior
   - T4: Autocorrelation lag-1 → Check temporal structure
3. **Bayesian p-value**: P(T(y_rep) > T(y_obs))

**Decision rule**:
- If p-value < 0.05 or > 0.95 for any test statistic → Model fails PPC
- Visual check: Observed data should be "typical" of replicated data

### Phase 3: LOO Cross-Validation

Compute LOO-CV for all models:
```python
import arviz as az
loo1 = az.loo(trace1, pointwise=True)
loo2 = az.loo(trace2, pointwise=True)
comparison = az.compare({'model1': loo1, 'model2': loo2})
```

**Interpretation**:
- ΔLOO > 10: Strong evidence for better model
- ΔLOO 4-10: Moderate evidence
- ΔLOO < 4: Models similar (prefer simpler)
- **Crucial**: Check Pareto-k values (k > 0.7 → PSIS unreliable)

**Decision rule**:
- If complex model LOO ≈ simple model LOO → Use simple model (parsimony)
- If complex model has many high Pareto-k → Suspicious (overfitting?)

### Phase 4: Sensitivity Analysis

**Prior sensitivity**:
1. Refit best model with more/less informative priors
2. Check if posterior conclusions change
3. If highly sensitive, results not robust → Report uncertainty

**Subset sensitivity**:
1. Fit to first 30 observations, predict last 10
2. Fit to last 30 observations only
3. Results should be qualitatively consistent

### Evaluation Criteria Summary

| Criterion | Weight | Metric |
|-----------|--------|--------|
| Out-of-sample prediction | 40% | LOO-ELPD (higher better) |
| Posterior predictive fit | 30% | p-values, visual checks |
| Parsimony | 15% | Effective parameters (fewer better) |
| Interpretability | 10% | Parameter interpretability |
| Computational stability | 5% | Divergences, Rhat, ESS |

**Final model selection**:
- Best LOO ± SE must be clearly best (not within 1 SE)
- Must pass posterior predictive checks
- Must have stable computation (Rhat < 1.01, ESS > 400)
- Prefer simplest adequate model

---

## Critical Decision Points & Pivot Triggers

### Checkpoint 1: After Polynomial Model (Model 2A)

**Question**: Does smooth polynomial suffice?

**Tests**:
- R² > 0.95 on log-scale?
- Residuals show no regime patterns?
- Dispersion constant across time?

**If YES** → Done! Use polynomial, don't fit complex models.
**If NO** → Proceed to changepoint/hierarchical models.

### Checkpoint 2: After Changepoint Model (Model 1)

**Question**: Is discrete changepoint supported?

**Tests**:
- Posterior P(18 < tau < 26) > 0.80? (concentrated)
- LOO improves over polynomial by >4 ELPD?
- Posterior predictive shows sharp transition?

**If YES** → Changepoint model is strong candidate, but still fit hierarchical for comparison.
**If NO** → Changepoint not justified, pivot to pure hierarchical or GP.

### Checkpoint 3: After Hierarchical Model (Model 3)

**Question**: Is additional complexity justified?

**Tests**:
- States clearly separable (not uniform posteriors)?
- LOO improves over changepoint by >4 ELPD?
- WAIC penalty reasonable (< 15)?

**If YES** → Hierarchical model may be best (report with changepoint for comparison).
**If NO** → Use simpler model (polynomial or changepoint).

### Red Flags That Trigger Abandonment

**Abandon current model class if**:

1. **Computational failure**:
   - >50% divergent transitions after tuning
   - Rhat > 1.05 for key parameters after 5000 iterations
   - ESS < 100 despite long chains
   - **Interpretation**: Model geometry fundamentally wrong

2. **Prior-posterior conflict**:
   - Posteriors pile up at prior boundaries
   - Example: phi → 0 or phi → 1000
   - **Interpretation**: Wrong parameterization or likelihood

3. **Posterior predictive disaster**:
   - Generated data looks nothing like observed
   - Example: Predicting negative counts, or all zeros
   - **Interpretation**: Model misspecified at basic level

4. **LOO degeneracy**:
   - >50% of observations have Pareto-k > 0.7
   - **Interpretation**: Influential points, poor generalization

### Major Pivot Options

**If all three model classes fail**:

1. **Gaussian Process trend** (non-parametric):
   - log(mu_i) = GP(year_i | lengthscale, amplitude)
   - Maximally flexible, but requires strong priors

2. **State-space model** (dynamic latent):
   - log(mu_i) = theta_i + epsilon_i
   - theta_i ~ Normal(theta_{i-1}, sigma_process)
   - Structural time series approach

3. **Reconsider likelihood family**:
   - Maybe not Negative Binomial?
   - Try: Poisson-Lognormal, Conway-Maxwell, Generalized Poisson
   - Or: Continuous approximation (Gamma, Lognormal) if discrete not critical

4. **Data quality investigation**:
   - Check for measurement error, reporting changes
   - Consult domain experts
   - Maybe model misspecification is data problem, not model problem

---

## Warning Signs & Diagnostics

### During Fitting

**Monitor in real-time**:
- Divergent transitions → Increase adapt_delta or reparameterize
- Low ESS for key parameters → Longer chains or different sampler
- High Rhat → Check initialization, look for bimodality

### After Fitting

**Required diagnostics**:
1. **Trace plots**: Should look like "fuzzy caterpillars" (good mixing)
2. **Pairs plots**: Check for strong correlations (collinearity)
3. **Prior-posterior overlap**: Posterior should differ from prior (learning!)
4. **Posterior predictive**: y_rep should cover y_obs
5. **Residuals**: Should be uncorrelated, homoscedastic

**Specific to these models**:
- **Changepoint**: Plot posterior P(tau = t) for all t
- **Polynomial**: Check for extreme extrapolation (plot beyond data range)
- **Hierarchical**: Check state assignment stability (no label switching)

### Red Flag Patterns

**Pattern 1: Banana-shaped posteriors**
- Indicates non-identifiable parameters
- Example: beta2 and beta3 highly correlated
- Fix: Orthogonal polynomials or stronger priors

**Pattern 2: Funnel geometry**
- Common with hierarchical models (sigma ↓ → theta constrained)
- Fix: Non-centered parameterization

**Pattern 3: Multimodal posterior**
- Example: tau could be 20 or 25 (two equally good changepoints)
- This is **information**, not failure! Report both modes.

---

## Implementation Priorities

### Week 1: Baseline Models
1. Fit polynomial (quadratic and cubic)
2. Implement posterior predictive checks
3. Compute LOO for all variants
4. **Decision**: If quadratic sufficient, stop here

### Week 2: Changepoint Models (if needed)
1. Implement Stan marginalization over tau
2. Fit linear and quadratic piecewise models
3. Compare to polynomial via LOO
4. **Decision**: Is discrete changepoint justified?

### Week 3: Hierarchical Models (if needed)
1. Start with two-state model (simpler)
2. Check identifiability and state recovery
3. If successful, try three-state
4. **Decision**: Final model selection

### Week 4: Validation & Reporting
1. Prior sensitivity analysis
2. Out-of-sample predictions
3. Posterior predictive checks (comprehensive)
4. Final model comparison table
5. Write-up with honest assessment

---

## Expected Challenges & Mitigation

### Challenge 1: Small Sample Size (n=40)

**Problem**: Overfitting risk, especially for hierarchical model

**Mitigation**:
- Strong regularizing priors
- Prefer simpler models (Occam's Razor)
- Cross-validation (LOO) to detect overfitting
- Don't use all 3 models unless clearly needed

### Challenge 2: Changepoint Identifiability

**Problem**: True changepoint might be "soft" (gradual), hard to pin down

**Mitigation**:
- Plot posterior P(tau) carefully
- If diffuse, report uncertainty honestly
- Consider transition model (logistic, not step function)

### Challenge 3: Hierarchical Label Switching

**Problem**: States can swap identities across MCMC iterations

**Mitigation**:
- Order constraints: alpha[1] < alpha[2] < alpha[3]
- Post-processing: Relabel samples consistently
- Report aggregate quantities (mixture, not state-specific)

### Challenge 4: Computational Cost

**Problem**: Hierarchical model may take hours to fit

**Mitigation**:
- Start with simpler models (don't fit all variants)
- Use compiled Stan (CmdStanPy, not PyStan)
- Parallel chains (use all CPU cores)
- Consider approximations (Variational Bayes) for initial exploration

### Challenge 5: Prior Sensitivity

**Problem**: With n=40, priors may influence posteriors substantially

**Mitigation**:
- Use weakly informative priors (not flat, not dogmatic)
- Prior predictive checks (do priors generate plausible data?)
- Sensitivity analysis (refit with different priors)
- Report honestly if conclusions depend on priors

---

## Success Criteria

### Model Success (Individual)
- ✓ Rhat < 1.01 for all parameters
- ✓ ESS > 400 for key parameters
- ✓ < 1% divergent transitions
- ✓ Posterior predictive p-values ∈ [0.05, 0.95]
- ✓ No extreme parameter values (within prior range)

### Comparison Success (Portfolio)
- ✓ LOO computed for all models (Pareto-k < 0.7 for >90%)
- ✓ Clear ranking by LOO-ELPD
- ✓ Best model not within 1 SE of second-best (decisive)
- ✓ Best model passes all diagnostics
- ✓ Results robust to prior perturbations

### Scientific Success (Truth-Seeking)
- ✓ Honest assessment of model limitations
- ✓ Uncertainty quantified (not just point estimates)
- ✓ Alternative explanations considered
- ✓ Know when to stop (don't overfit)
- ✓ Clear documentation of decisions and pivots

---

## Alternative Hypotheses & Escape Routes

### If All Models Fail Badly...

**Hypothesis A: Data Quality Issue**
- Check for: Measurement changes, reporting errors, missing context
- Test: Ask for data provenance, compare to external sources
- Escape: Consult domain expert, revise data or question

**Hypothesis B: Wrong Likelihood Family**
- Check: Maybe not count data? (Measurement process continuous, rounded?)
- Test: Fit Gamma or Lognormal instead of NegBin
- Escape: Continuous approximation (valid for counts > 20)

**Hypothesis C: Missing Covariates**
- Check: Maybe trend driven by unobserved variable?
- Test: Residuals have structure that covariates might explain
- Escape: Request additional data (if available)

**Hypothesis D: Fundamentally Unpredictable**
- Check: Maybe process is chaotic, not stochastic?
- Test: Lyapunov exponents, phase-space reconstruction
- Escape: Admit defeat, report high uncertainty

---

## Final Recommendations

### Start Here:
1. **Model 2A** (polynomial, quadratic, constant dispersion)
   - Simplest, fastest, likely adequate
   - If R² > 0.95 and diagnostics pass → STOP, you're done

### If Polynomial Inadequate:
2. **Model 1A** (piecewise linear changepoint)
   - More complex, but addresses regime shift directly
   - If posterior tau concentrated → Strong evidence for discrete change

### If Changepoint Uncertain:
3. **Model 3A** (two-state hierarchical)
   - Hedges between smooth and discrete
   - Most flexible, but most complex

### Don't Fit Unless Necessary:
- Three-state hierarchical (only if two-state clearly insufficient)
- Cubic polynomial (quadratic usually enough)
- Splines (polynomial simpler, likely adequate)

### Prioritize:
- **Truth-seeking** over task completion
- **Parsimony** over flexibility
- **Honest uncertainty** over false precision
- **Computational stability** over theoretical elegance

---

## Conclusion

This experiment plan proposes **three competing Bayesian model classes** for count time series with structural change:

1. **Piecewise Negative Binomial**: Discrete regime shift at unknown changepoint
2. **Smooth Polynomial**: Continuous nonlinear trend with time-varying dispersion
3. **Hierarchical Latent States**: Soft regime membership with partial pooling

Each model has:
- ✓ Full Bayesian specification with priors
- ✓ Stan implementation plan
- ✓ Explicit falsification criteria
- ✓ Expected posteriors and diagnostics
- ✓ Computational considerations

**Critical mindset**: I expect to **abandon** most of these models. The goal is to discover which (if any) survive contact with data. Success = finding truth, not confirming hypotheses.

**Next steps**:
1. Implement Model 2A (simplest) first
2. Check diagnostics rigorously
3. Only proceed to complex models if necessary
4. Report honestly when models fail
5. Pivot quickly when evidence contradicts assumptions

**Stopping rule**: If simple polynomial works well, STOP. Don't fit complex models just because they're interesting.

---

**Files to Create**:
- `/workspace/experiments/designer_3/model_1_changepoint.stan`
- `/workspace/experiments/designer_3/model_2_polynomial.stan`
- `/workspace/experiments/designer_3/model_3_hierarchical.stan`
- `/workspace/experiments/designer_3/fit_models.py` (CmdStanPy implementation)
- `/workspace/experiments/designer_3/diagnostics.py` (PPC, LOO, comparison)

**Timeline**: 3-4 weeks (1 week per model class + validation)

**Budget**: ~40 hours (including computational time, diagnostics, writing)
