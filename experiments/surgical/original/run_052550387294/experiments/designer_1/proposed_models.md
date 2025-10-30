# Bayesian Model Proposals for Binomial Overdispersion
## Designer 1: Variance Modeling Focus

**Date**: 2025-10-30
**Context**: 12 binomial trials, φ = 3.51, strong overdispersion (p < 0.001)
**Design Philosophy**: Focus on models that explicitly parameterize and explain variance inflation

---

## Executive Summary

I propose **three fundamentally different model classes** to explain the observed overdispersion:

1. **Beta-Binomial Model**: Continuous probability variation via conjugate prior
2. **Dirichlet-Process Mixture**: Non-parametric clustering with unknown number of groups
3. **Logistic-Normal Hierarchical Model**: Gaussian variation on logit scale

**Key Insight**: These models make DIFFERENT predictions about the data-generating process:
- Beta-Binomial assumes smooth continuous variation
- DP Mixture assumes discrete probability regimes (unknown number)
- Logistic-Normal assumes multiplicative effects on odds

**Critical Question**: Is overdispersion due to continuous variation, discrete subgroups, or scale-dependent effects?

---

## Model Class 1: Beta-Binomial (Conjugate Approach)

### Model Specification

```
Likelihood:
  r_i | θ_i, n_i ~ Binomial(n_i, θ_i)
  θ_i | α, β ~ Beta(α, β)

Priors:
  α ~ Gamma(2, 0.1)     # E[α] = 20, wide spread
  β ~ Gamma(2, 0.005)   # E[β] = 400, implies E[θ] ≈ 0.05
```

**Marginalized Form** (more numerically stable):
```
r_i | α, β, n_i ~ BetaBinomial(n_i, α, β)

where:
  E[r_i/n_i] = α/(α+β)
  Var[r_i/n_i] = (α·β)/[(α+β)²·(α+β+1)] · (1 + n_i)/(n_i)
```

**Overdispersion Parameterization**:
```
Alternative specification:
  μ = α/(α+β)          # Mean probability
  φ = α + β            # Concentration parameter

Priors:
  μ ~ Beta(2, 25)      # Centers near 0.074
  φ ~ Gamma(2, 2)      # E[φ] = 1, allows φ << 12 for overdispersion

Then:
  α = μ·φ
  β = (1-μ)·φ
```

### Theoretical Justification

**Data-Generating Process**:
- Each trial samples from a population with its own latent success probability θ_i
- These probabilities are themselves drawn from a Beta distribution
- Represents "exchangeable heterogeneity" - trials differ but are fundamentally similar

**Why This Addresses Overdispersion**:
- Beta distribution allows probabilities to vary continuously
- When φ is small (φ < n_i), variance inflation occurs
- Expected dispersion: Var[r_i] = n_i·μ·(1-μ)·[1 + (n_i-1)·ρ] where ρ = 1/(φ+1)
- With φ ≈ 3, this produces ρ ≈ 0.25, matching observed φ = 3.5

**Assumptions**:
1. **Exchangeability**: All trials drawn from same Beta distribution
2. **Continuous variation**: No discrete probability regimes
3. **Independence**: Trials are independent conditional on hyperparameters
4. **Symmetry**: Beta shape symmetric with respect to covariates

**When This Model is CORRECT**:
- Trials sample from heterogeneous populations with continuous variation
- No hidden clustering structure
- Variation driven by unobserved continuous factors

### Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Posterior predictive check fails**:
   - Simulated datasets show significantly less variance than observed
   - p-value for χ² statistic < 0.05 or > 0.95
   - Funnel plot violations persist in posterior samples

2. **Bimodal posterior for μ**:
   - Indicates mixing of distinct groups, not continuous variation
   - Would suggest mixture model is more appropriate

3. **Extreme concentration parameter**:
   - φ posterior < 0.5: Model struggling to capture dispersion
   - φ posterior > 50: Nearly constant probability (contradicts overdispersion)

4. **Poor LOO-CV performance**:
   - Multiple observations with Pareto k > 0.7 (influential outliers)
   - LOO worse than simpler baseline by ΔELPD > 5

5. **Systematic residual patterns**:
   - Residuals correlate with sample size (n_i)
   - Residuals show temporal structure
   - Clear visual evidence of discrete clusters in posterior θ_i

### Implementation Considerations

**Stan Implementation**:
```stan
data {
  int<lower=1> N;           // Number of trials
  array[N] int<lower=0> n;  // Sample sizes
  array[N] int<lower=0> r;  // Success counts
}

parameters {
  real<lower=0,upper=1> mu;     // Mean probability
  real<lower=0> kappa_minus_2;  // φ - 2, ensures φ > 2
}

transformed parameters {
  real<lower=2> phi = kappa_minus_2 + 2;
  real<lower=0> alpha = mu * phi;
  real<lower=0> beta = (1 - mu) * phi;
}

model {
  // Priors
  mu ~ beta(2, 25);
  kappa_minus_2 ~ gamma(2, 2);

  // Likelihood
  r ~ beta_binomial(n, alpha, beta);
}

generated quantities {
  array[N] int r_rep;  // Posterior predictive
  for (i in 1:N) {
    r_rep[i] = beta_binomial_rng(n[i], alpha, beta);
  }
}
```

**Computational Challenges**:
1. **Label switching**: Not applicable (no discrete components)
2. **Prior sensitivity**: α, β scale jointly - use φ parameterization
3. **Numerical stability**: beta_binomial uses log-space internally (Stan handles this)
4. **Weak identification**: With only 12 trials, φ posterior may be wide

**Expected Diagnostics**:
- Rhat < 1.01 for all parameters (should converge easily)
- ESS > 400 per chain (simple 2-parameter model)
- No divergences expected (convex posterior)

**Prior Sensitivity Areas**:
- **mu prior**: Try Beta(1,13), Beta(2,25), Beta(5,65) to check robustness
- **phi prior**: Try Gamma(2,1), Gamma(2,2), Gamma(3,3) to assess dispersion sensitivity
- **Critical test**: If conclusions change dramatically with prior, model may be poorly identified

---

## Model Class 2: Dirichlet Process Mixture (Non-parametric Clustering)

### Model Specification

```
Likelihood:
  r_i | θ_i, n_i ~ Binomial(n_i, θ_i)

Mixture Model:
  θ_i ~ Σ_{k=1}^K π_k · Beta(α_k, β_k)

  where K is UNKNOWN and learned from data

Dirichlet Process Prior:
  G ~ DP(γ, G_0)
  θ_i ~ G

Base Distribution:
  G_0 = Beta(α_0, β_0)

Hyperpriors:
  α_0 ~ Gamma(2, 20)    # E[θ_0] ≈ 0.1
  β_0 ~ Gamma(25, 250)  # E[θ_0] ≈ 0.1
  γ ~ Gamma(1, 1)       # Concentration (controls K)
```

**Practical Implementation** (Stick-Breaking):
```
Stick-breaking construction:
  v_k ~ Beta(1, γ) for k = 1, ..., K_max
  π_k = v_k · Π_{j<k} (1 - v_j)

  θ_k ~ Beta(α_0, β_0)
  z_i ~ Categorical(π)
  r_i ~ Binomial(n_i, θ_{z_i})

Where K_max = 10 (truncation, sufficient for 12 obs)
```

### Theoretical Justification

**Data-Generating Process**:
- Trials belong to UNKNOWN number of discrete probability regimes
- Each regime has its own success probability drawn from Beta(α_0, β_0)
- Dirichlet Process automatically infers number of active groups
- Similar to finite mixture but K is data-driven

**Why This Addresses Overdispersion**:
- EDA suggests 2-3 distinct groups (tercile analysis)
- Mixture of binomials is overdispersed relative to single binomial
- Allows for sharp probability differences between groups
- More flexible than assuming K=2 or K=3 a priori

**Assumptions**:
1. **Discrete clustering**: Trials cluster into distinct probability groups
2. **Exchangeability within clusters**: Trials in same cluster are similar
3. **Sparse clustering**: Most trials belong to a few dominant clusters (DP prior encourages this)
4. **No temporal structure**: Group membership is random with respect to trial_id

**When This Model is CORRECT**:
- Trials conducted under fundamentally different conditions
- "Batch effects" or regime switches
- Discrete experimental conditions not recorded in data
- EDA tercile/median-split patterns reflect real clusters

### Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Posterior collapses to K=1**:
   - All trials assigned to single cluster with posterior probability > 0.8
   - Would indicate continuous variation, not discrete groups
   - Beta-Binomial would be simpler and preferred

2. **Excessive clusters (K > 6)**:
   - Each trial gets its own cluster
   - Overfitting indicator
   - Model not borrowing strength appropriately
   - Would abandon for simpler hierarchical model

3. **No cluster separation**:
   - Posterior cluster probabilities θ_k overlap completely
   - 95% credible intervals overlap for all clusters
   - Cannot distinguish clusters - suggests continuous variation

4. **Unstable assignments**:
   - High posterior uncertainty in cluster membership z_i
   - Trials flip between clusters across MCMC iterations
   - Indicates clusters are not well-defined

5. **Poor predictive performance**:
   - LOO-CV worse than Beta-Binomial by ΔELPD > 3
   - Posterior predictive checks show poor calibration
   - Model complexity not justified by fit

6. **Implausible cluster structure**:
   - Clusters correlate with sample size n_i (would indicate size-dependent artifact)
   - Single large trial (trial 4) forms its own cluster (outlier-driven clustering)

### Implementation Considerations

**Stan Implementation** (Truncated DP):
```stan
data {
  int<lower=1> N;           // Number of trials
  array[N] int<lower=0> n;  // Sample sizes
  array[N] int<lower=0> r;  // Success counts
  int<lower=1> K_max;       // Max clusters (e.g., 10)
}

parameters {
  vector<lower=0,upper=1>[K_max] v;      // Stick-breaking weights
  vector<lower=0,upper=1>[K_max] theta;  // Cluster probabilities
  real<lower=0> alpha_0;                  // Base distribution shape
  real<lower=0> beta_0;                   // Base distribution shape
  real<lower=0> gamma;                    // DP concentration
}

transformed parameters {
  simplex[K_max] pi;  // Mixture weights
  pi[1] = v[1];
  for (k in 2:K_max) {
    pi[k] = v[k] * prod(1 - v[1:(k-1)]);
  }
}

model {
  // Priors
  alpha_0 ~ gamma(2, 20);
  beta_0 ~ gamma(25, 250);
  gamma ~ gamma(1, 1);
  v ~ beta(1, gamma);

  // Cluster probabilities
  theta ~ beta(alpha_0, beta_0);

  // Likelihood (marginalized over cluster assignments)
  for (i in 1:N) {
    vector[K_max] lps;
    for (k in 1:K_max) {
      lps[k] = log(pi[k]) + binomial_lpmf(r[i] | n[i], theta[k]);
    }
    target += log_sum_exp(lps);
  }
}

generated quantities {
  // Inferred number of active clusters
  int K_active = 0;
  for (k in 1:K_max) {
    if (pi[k] > 0.05) K_active += 1;  // Threshold for "active"
  }

  // Cluster assignments (MAP)
  array[N] int z;
  for (i in 1:N) {
    vector[K_max] lps;
    for (k in 1:K_max) {
      lps[k] = log(pi[k]) + binomial_lpmf(r[i] | n[i], theta[k]);
    }
    z[i] = categorical_logit_rng(lps);
  }
}
```

**Computational Challenges**:
1. **Label switching**: Major issue - clusters can swap labels across iterations
   - Solution: Post-process with relabeling algorithm (e.g., Stephen's algorithm)
   - Or use ordered constraint: theta[k] < theta[k+1]

2. **Multimodality**: Posterior may have multiple modes with different K
   - Need multiple chains with dispersed inits
   - Check for converged Rhat across chains

3. **Slow mixing**: Cluster assignments can have high autocorrelation
   - May need 10,000+ iterations per chain
   - Monitor ESS for pi and theta parameters

4. **Weak identification**: With N=12, hard to learn many clusters
   - Expect wide posterior on K_active
   - Strong influence from γ prior

**Expected Diagnostics**:
- Rhat may be > 1.01 due to label switching (this is OK if post-processed)
- ESS may be low (<100) for individual theta[k] if K_active is uncertain
- Potential divergences if clusters are poorly separated

**Prior Sensitivity Areas**:
- **γ (concentration)**: Crucially affects K_active
  - Try Gamma(0.5, 0.5), Gamma(1,1), Gamma(2,2)
  - Low γ → fewer clusters; high γ → more clusters
- **Base distribution (α_0, β_0)**: Affects cluster separation
  - Strong priors → clusters similar; weak priors → more separation
- **K_max**: Try 5, 10, 15 to ensure truncation doesn't affect results

---

## Model Class 3: Logistic-Normal Hierarchical Model

### Model Specification

```
Likelihood:
  r_i | p_i, n_i ~ Binomial(n_i, p_i)

Logit Transform:
  logit(p_i) = η_i

Hierarchical Prior:
  η_i ~ Normal(μ_η, σ_η²)

Hyperpriors:
  μ_η ~ Normal(-2.5, 1.5²)   # Implies median p ≈ 0.076
  σ_η ~ HalfNormal(1)         # Controls between-trial variation
```

**Relation to Probability**:
```
p_i = logistic(η_i) = 1/(1 + exp(-η_i))

E[p_i] ≈ logistic(μ_η) when σ_η is small
Var[p_i] increases with σ_η
```

**Overdispersion Mechanism**:
```
Var[r_i/n_i] = E[Var[r_i/n_i | p_i]] + Var[E[r_i/n_i | p_i]]
             = E[p_i·(1-p_i)/n_i] + Var[p_i]

Overdispersion factor ≈ 1 + n_i·Var[p_i]/(E[p_i]·(1-E[p_i]))
```

### Theoretical Justification

**Data-Generating Process**:
- Each trial has latent log-odds η_i
- Log-odds vary normally across trials
- Represents multiplicative effects on odds scale
- Natural for modeling probabilities with covariates

**Why This Addresses Overdispersion**:
- Normal variation on logit scale → overdispersion on probability scale
- Log-odds scale is more natural for many processes (e.g., multiplicative risk factors)
- Variation is scale-dependent: more variation when p ≈ 0.5, less when p ≈ 0 or 1
- Observed data has low p ≈ 0.074, so logit scale may be more appropriate than logistic

**Assumptions**:
1. **Normality on logit scale**: Log-odds are Gaussian (central limit theorem for multiplicative effects)
2. **Homoscedasticity**: Variance σ_η² is constant across trials
3. **No correlation**: η_i are independent (conditional on hyperparameters)
4. **Scale symmetry**: Effects are symmetric on log-odds scale

**When This Model is CORRECT**:
- Probabilities vary due to multiplicative factors
- Underlying process involves combining multiple small effects
- Common in epidemiology, ecology (e.g., logistic growth with random effects)
- True when effects are additive on odds scale

**Key Difference from Beta-Binomial**:
- Beta-Binomial: Uniform variation across probability range
- Logistic-Normal: Scale-dependent variation (less extreme probabilities)
- LN more conservative for extreme values

### Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Non-normality of log-odds posterior**:
   - Q-Q plot of posterior means η_i shows heavy tails or skewness
   - Suggests alternative distribution (e.g., t-distribution) needed
   - Or discrete groups (mixture model) more appropriate

2. **Residuals show heteroscedasticity**:
   - Variance of (logit(r_i/n_i) - η_i) correlates with n_i
   - Would indicate sample-size-dependent variation
   - Violation of constant σ_η assumption

3. **Extreme σ_η posterior**:
   - σ_η posterior mass near 0 → No between-trial variation (contradicts data)
   - σ_η posterior > 3 → Implausibly large variation (probabilities near 0 and 1)

4. **Poor fit to extreme observations**:
   - Trial 1 (r=0) or trial 8 (r=31, p=0.144) poorly predicted
   - Model cannot accommodate extreme probabilities
   - Log-odds scale may be inappropriate

5. **Worse predictive performance than Beta-Binomial**:
   - LOO-CV ΔELPD < -3 compared to Beta-Binomial
   - Posterior predictive checks fail (e.g., χ² p-value < 0.05)

6. **Prior-posterior conflict**:
   - Strong disagreement between prior and likelihood
   - Indicates prior misspecification or model inadequacy
   - Would try robustifying with t-distribution

### Implementation Considerations

**Stan Implementation**:
```stan
data {
  int<lower=1> N;           // Number of trials
  array[N] int<lower=0> n;  // Sample sizes
  array[N] int<lower=0> r;  // Success counts
}

parameters {
  vector[N] eta;              // Log-odds for each trial
  real mu_eta;                // Mean log-odds
  real<lower=0> sigma_eta;    // SD of log-odds
}

transformed parameters {
  vector<lower=0,upper=1>[N] p = inv_logit(eta);
}

model {
  // Priors
  mu_eta ~ normal(-2.5, 1.5);
  sigma_eta ~ normal(0, 1);  // Half-normal (sigma_eta > 0)

  // Hierarchical prior
  eta ~ normal(mu_eta, sigma_eta);

  // Likelihood
  r ~ binomial_logit(n, eta);  // More numerically stable
}

generated quantities {
  // Posterior predictive
  array[N] int r_rep;
  for (i in 1:N) {
    r_rep[i] = binomial_rng(n[i], p[i]);
  }

  // Derived quantities
  real mu_p = inv_logit(mu_eta);  // Mean probability
  real overdispersion = 1 + mean(n) * variance(p) / (mu_p * (1 - mu_p));
}
```

**Computational Challenges**:
1. **Non-conjugacy**: No closed-form posterior, requires MCMC
2. **Divergences possible**: When σ_η is near 0 (funnel geometry)
   - Solution: Non-centered parameterization (see below)
3. **Trial 1 (r=0)**: Log-odds tends to -∞
   - Stan handles this but posterior for η_1 will be left-truncated
   - May need informative prior to stabilize

**Non-Centered Parameterization** (if divergences occur):
```stan
parameters {
  vector[N] eta_raw;          // Standard normal
  real mu_eta;
  real<lower=0> sigma_eta;
}

transformed parameters {
  vector[N] eta = mu_eta + sigma_eta * eta_raw;
  vector<lower=0,upper=1>[N] p = inv_logit(eta);
}

model {
  mu_eta ~ normal(-2.5, 1.5);
  sigma_eta ~ normal(0, 1);
  eta_raw ~ std_normal();
  r ~ binomial_logit(n, eta);
}
```

**Expected Diagnostics**:
- Rhat < 1.01 (should converge well)
- ESS > 400 for mu_eta, sigma_eta
- ESS may be lower (>100) for individual eta[i]
- Possible divergences if centered parameterization used

**Prior Sensitivity Areas**:
- **mu_eta prior**:
  - Normal(-2.5, 1.5) → p ≈ 0.076 ± 0.15
  - Try Normal(-2.5, 1), Normal(-2.5, 2) to check robustness
  - If posterior far from prior, model learning effectively

- **sigma_eta prior**:
  - HalfNormal(1) is moderately informative
  - Try HalfNormal(0.5), HalfNormal(2), Exponential(1)
  - Critical for overdispersion - if sensitive, may be poorly identified

---

## Model Comparison Strategy

### Competing Hypotheses

These three models represent FUNDAMENTALLY DIFFERENT hypotheses:

| Model | Hypothesis | Prediction |
|-------|------------|------------|
| **Beta-Binomial** | Continuous smooth variation in probabilities | θ_i spread continuously, no clustering |
| **DP Mixture** | Discrete probability regimes (K unknown) | θ_i cluster into 2-4 groups |
| **Logistic-Normal** | Gaussian variation on log-odds scale | η_i normally distributed, scale-dependent overdispersion |

### Critical Comparisons

**Beta-Binomial vs. DP Mixture**:
- **Distinguishing evidence**: Posterior separation of trial probabilities
  - BB: Posterior θ_i will overlap smoothly
  - DPM: Posterior θ_k will separate into distinct clusters
- **Test**: Hierarchical clustering of posterior mean θ_i
  - If clear gap → DPM favored
  - If continuous → BB favored

**Beta-Binomial vs. Logistic-Normal**:
- **Distinguishing evidence**: Behavior at extreme probabilities
  - BB: Symmetric prior, treats p=0.01 and p=0.99 similarly
  - LN: Asymmetric, less variation at extremes
- **Test**: Posterior for trial 1 (r=0) and trial 8 (p=0.144)
  - If extreme values well-predicted → Check which model
  - If LN gives tighter posteriors at extremes → LN favored

**DP Mixture vs. Logistic-Normal**:
- **Distinguishing evidence**: Continuity vs. discreteness
  - DPM: Sharp cluster boundaries
  - LN: Smooth gradation
- **Test**: Tercile analysis of posterior θ_i
  - If clear gaps between terciles → DPM
  - If overlapping distributions → LN

### Decision Tree

```
1. Fit all three models
   ↓
2. Check convergence (Rhat, ESS, divergences)
   - If any fail → Debug before proceeding
   ↓
3. Posterior predictive checks
   - Simulate 1000 datasets from each model
   - Calculate χ², overdispersion factor
   - Compare to observed data
   - ELIMINATE models with p-value < 0.05 or > 0.95
   ↓
4. LOO-CV comparison
   - Calculate LOO for each model
   - ELIMINATE if ΔELPD < -5 from best
   - Check Pareto k values
   ↓
5. Substantive interpretation
   - For DP: How many clusters? Interpretable?
   - For BB: Is φ in plausible range?
   - For LN: Is σ_η reasonable?
   ↓
6. Sensitivity analysis
   - Vary priors (weak, moderate, strong)
   - If conclusions change → Model poorly identified
   ↓
7. FINAL DECISION
   - If multiple models pass all tests → Model averaging
   - If one clearly best → Adopt as primary
   - If all fail → PIVOT to alternative approach
```

### Red Flags for MAJOR PIVOT

**Abandon ALL three models if**:

1. **None fit the data**:
   - All three fail posterior predictive checks
   - All have LOO issues (multiple k > 0.7)
   - Systematic residual patterns across all models

2. **Evidence of temporal dependence**:
   - Autocorrelation in residuals
   - Would require time-series model (e.g., state-space)

3. **Evidence of sample-size dependence**:
   - Residuals correlate with n_i
   - Would require covariate model

4. **Zero-inflation**:
   - Trial 1 (r=0) is outlier in all models
   - Multiple other trials with r=0 poorly fit
   - Would require zero-inflated model

5. **Heavy-tailed departures**:
   - Multiple extreme outliers
   - Would require robust model (t-distribution)

**Alternative model classes to consider**:
- **State-space model** (if temporal structure)
- **Beta regression with covariates** (if size-dependent)
- **Zero-inflated Beta-Binomial** (if excess zeros)
- **Robust Beta-Binomial** (if heavy tails)

---

## Stress Tests

To break these models, I will conduct **adversarial analyses**:

### Stress Test 1: Outlier Robustness
- **Fit all models excluding trial 1 (r=0)**
- **Expected**: Models should give similar results
- **FAIL if**: Conclusions change dramatically (e.g., K_active changes, φ shifts by >50%)

### Stress Test 2: Large-Sample Influence
- **Fit all models excluding trial 4 (n=810)**
- **Expected**: Trial 4 should not dominate inference
- **FAIL if**: Posterior means shift by >30%, model suddenly can't fit remaining data

### Stress Test 3: Prior Robustness
- **Fit with priors 10x more diffuse and 10x more concentrated**
- **Expected**: Posterior should be data-dominated
- **FAIL if**: Posterior changes substantially (indicates weak identification)

### Stress Test 4: Predictive Coherence
- **LOO-CV: Predict each trial using model fit to other 11**
- **Expected**: Predictions roughly calibrated
- **FAIL if**: Multiple trials have k > 0.7 or very low p_loo

### Stress Test 5: Simulation-Based Calibration (SBC)
- **Simulate 100 datasets from prior predictive**
- **Fit model to each, check rank statistics**
- **Expected**: Uniform ranks
- **FAIL if**: Ranks are non-uniform (model misspecified or computational issues)

---

## Computational Plan

### Stan Implementation Priority
1. **Start with Beta-Binomial** (simplest, fastest)
   - 2 parameters, conjugate structure
   - Should converge in <1 minute
   - Establish baseline performance

2. **Then Logistic-Normal** (moderate complexity)
   - N+2 parameters, non-conjugate
   - Test centered vs. non-centered parameterization
   - ~2-5 minutes

3. **Finally DP Mixture** (most complex)
   - ~2*K_max parameters + marginalization
   - May need long runs (10K+ iterations)
   - ~10-30 minutes
   - Post-process for label switching

### Sampling Strategy
- **Chains**: 4 (for Rhat calculation)
- **Warmup**: 2000 (conservative)
- **Sampling**: 2000 per chain (8000 total)
- **Thinning**: None initially (diagnose first)
- **Adapt delta**: 0.95 (to reduce divergences)

### Parallel Implementation
- Run all three models in parallel (independent analyses)
- Compare results only after each converges
- Avoid "peeking" at one model while fitting others

---

## Expected Outcomes and Interpretation

### Scenario A: Beta-Binomial Wins
**If**: LOO best, posterior predictive checks pass, φ ≈ 2-5
**Interpretation**: Overdispersion due to continuous variation
**Conclusion**: Trials sample from heterogeneous but smoothly varying probabilities
**Action**: Use for inference, report posterior for α, β

### Scenario B: DP Mixture Wins
**If**: LOO best, K_active = 2-3, clear cluster separation
**Interpretation**: Discrete probability regimes exist
**Conclusion**: Trials belong to distinct groups (batch effects, conditions)
**Action**: Investigate cluster membership, look for external explanations

### Scenario C: Logistic-Normal Wins
**If**: LOO best, σ_η ≈ 0.5-1.5, η_i normally distributed
**Interpretation**: Multiplicative variation on odds scale
**Conclusion**: Effects combine additively on log-odds
**Action**: Use for inference, report on log-odds scale

### Scenario D: All Models Similar
**If**: LOO within ±2, all pass predictive checks
**Interpretation**: Models make similar predictions despite different structures
**Conclusion**: Cannot distinguish with N=12
**Action**: Model averaging or choose simplest (Beta-Binomial)

### Scenario E: All Models Fail
**If**: All fail predictive checks or LOO diagnostics
**Interpretation**: Fundamental model misspecification
**Conclusion**: Need alternative approach (see Red Flags above)
**Action**: PIVOT to different model class, investigate domain knowledge

---

## Success Criteria

**This modeling effort is SUCCESSFUL if**:

1. At least one model passes all diagnostic checks
2. Posterior predictive simulations match observed χ² distribution
3. LOO-CV shows reasonable predictive performance (no k > 0.7)
4. Results are robust to prior specification
5. Model provides interpretable story about overdispersion source

**This modeling effort FAILS if**:

1. No model adequately captures overdispersion
2. Computational issues prevent convergence
3. Results highly sensitive to priors or outliers
4. Posterior predictive checks consistently fail
5. Cannot distinguish between fundamentally different hypotheses

**In case of FAILURE**: Document thoroughly and pivot to alternative approaches. Failure is informative - it tells us these model classes are inadequate for this data-generating process.

---

## Timeline and Checkpoints

### Checkpoint 1: Initial Fits (1-2 hours)
- Fit all three models with default settings
- Check convergence diagnostics
- **GO/NO-GO**: If none converge, debug before proceeding

### Checkpoint 2: Model Diagnostics (2-3 hours)
- Posterior predictive checks
- LOO-CV calculation
- Initial model comparison
- **GO/NO-GO**: If all fail basic checks, PIVOT to alternatives

### Checkpoint 3: Sensitivity Analysis (2-3 hours)
- Vary priors
- Outlier exclusion
- Prior-posterior comparison
- **GO/NO-GO**: If results unstable, question identifiability

### Checkpoint 4: Final Decision (1 hour)
- Synthesize all evidence
- Make model recommendation
- Document uncertainties
- **DELIVERABLE**: Final model choice or pivot decision

**Total estimated time**: 6-9 hours

---

## Limitations and Caveats

### Sample Size (N=12)
- **Small sample**: Hard to distinguish complex models
- **Wide posteriors expected**: Especially for hyperparameters
- **Limited power**: May not detect subtle patterns
- **Implication**: Favor simpler models (Occam's Razor)

### No Covariates
- **Cannot explain variation**: Only describe it
- **Missing confounders possible**: Overdispersion may be due to unmeasured variables
- **Implication**: Predictions are descriptive, not causal

### Trial 1 Influence
- **Zero successes**: r=0 out of n=47
- **Unusual but plausible**: 2.7% probability under pooled model
- **Potential outlier**: May disproportionately affect inference
- **Implication**: Sensitivity analysis essential

### Domain Knowledge Absent
- **No context**: Don't know what's being measured
- **Cannot assess plausibility**: Of resulting probabilities or clusters
- **No external validation**: Can't check if clusters make sense
- **Implication**: Statistical fit is necessary but not sufficient

---

## Falsification Summary

To emphasize the critical thinking required, here's a summary of what would make me **completely abandon** each model:

| Model | Primary Falsification Criterion |
|-------|--------------------------------|
| **Beta-Binomial** | Bimodal posterior for θ_i, indicating discrete groups not continuous variation |
| **DP Mixture** | Posterior K_active = 1 or K_active > 6, indicating either no clustering or severe overfitting |
| **Logistic-Normal** | Non-normal posterior for η_i (Q-Q plot shows heavy departure), indicating wrong distributional assumption |

**Universal Falsification** (all three models):
- Systematic residual patterns with sample size or trial order
- Posterior predictive p-values < 0.05 for χ² statistic
- Multiple LOO k > 0.7 diagnostics
- Prior-posterior conflict (hyperparameters hit prior boundaries)

---

## Final Notes

**Philosophy**: These models are HYPOTHESES to be tested, not truths to be confirmed. I expect at least one will fail, and possibly all three. That's the scientific process.

**Commitment**: I will report negative results as enthusiastically as positive ones. A model that fails informatively is more valuable than one that fits poorly but is presented as successful.

**Pivot readiness**: If evidence emerges that all three models are inadequate, I will immediately propose alternative approaches rather than forcing a choice among bad options.

**Collaboration**: These proposals are designed to be synthesized with other designers' approaches. I expect healthy disagreement and look forward to model comparisons across design philosophies.

---

**End of Proposal**

**Files**: `/workspace/experiments/designer_1/proposed_models.md`
**Next Steps**: Implement models in Stan, run diagnostics, compare to other designers' proposals
