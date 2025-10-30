# Bayesian Model Proposals: Designer 3
## Alternative Approaches for Binomial Overdispersion

**Designer**: Model Designer 3 (Alternative/Robust Approaches)
**Date**: 2025-10-30
**Focus**: Mixture models, robust specifications, and assumption-challenging frameworks

---

## Executive Summary

Given strong overdispersion (φ = 3.51) and evidence for 2-3 distinct probability groups, I propose three **fundamentally different** model classes that challenge standard assumptions:

1. **Finite Mixture Model**: Discrete latent classes with different success probabilities
2. **Robust Contamination Model**: Clean binomial process with outlier-generating mechanism
3. **Latent Beta-Binomial with Outlier Detection**: Continuous variation + explicit outlier modeling

**Key Philosophy**: Standard Beta-Binomial assumes smooth continuous variation. But what if the DGP has **discrete regimes** or **contamination**? These models explicitly test those alternatives.

**Critical Decision Point**: If mixture components don't separate cleanly (posterior overlap > 80%), abandon discrete mixtures entirely and pivot to continuous models.

---

## Model 1: Finite Mixture of Binomials

### 1.1 Model Specification

**Core Idea**: Trials come from K distinct populations, each with its own success probability.

#### Mathematical Formulation

```
# Likelihood
For each trial i:
  z_i ~ Categorical(π) where π = (π_1, ..., π_K)  # Latent class membership
  r_i | z_i, p ~ Binomial(n_i, p_{z_i})           # Conditional on class

# Priors
π ~ Dirichlet(α_1, ..., α_K)                       # Mixing proportions
p_k ~ Beta(a_k, b_k) for k = 1, ..., K             # Class-specific probabilities

# Ordering constraint (for identifiability)
p_1 < p_2 < ... < p_K
```

#### Specific Parameterization (K=3)

Based on tercile analysis showing groups at p ≈ 0.039, 0.067, 0.115:

```
# Prior on mixing proportions (weakly informative)
π ~ Dirichlet(2, 2, 2)  # Slight preference for equal mixing

# Prior on class probabilities (ordered)
p_1 ~ Beta(2, 48)  # E[p_1] ≈ 0.04, concentrated in [0.01, 0.10]
p_2 ~ Beta(3, 42)  # E[p_2] ≈ 0.067, concentrated in [0.02, 0.15]
p_3 ~ Beta(5, 40)  # E[p_3] ≈ 0.11, concentrated in [0.04, 0.20]

# With ordering constraint: p_1 < p_2 < p_3
```

#### Stan Implementation Strategy

```stan
data {
  int<lower=1> N;          // Number of trials (12)
  int<lower=1> K;          // Number of components (2 or 3)
  int<lower=0> r[N];       // Success counts
  int<lower=1> n[N];       // Trial sizes
}

parameters {
  simplex[K] pi;           // Mixing proportions
  ordered[K] p;            // Ordered probabilities (identifiability)
}

model {
  // Priors
  pi ~ dirichlet(rep_vector(2, K));

  // Priors on ordered probabilities
  p[1] ~ beta(2, 48);
  p[2] ~ beta(3, 42);
  p[3] ~ beta(5, 40);

  // Likelihood (marginal over latent classes)
  for (i in 1:N) {
    vector[K] log_pi_k;
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) + binomial_lpmf(r[i] | n[i], p[k]);
    }
    target += log_sum_exp(log_pi_k);
  }
}

generated quantities {
  // Posterior classification probabilities
  matrix[N, K] gamma;  // P(z_i = k | data)

  for (i in 1:N) {
    vector[K] log_pi_k;
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) + binomial_lpmf(r[i] | n[i], p[k]);
    }
    gamma[i] = softmax(log_pi_k)';
  }
}
```

### 1.2 Theoretical Justification

**Why This Addresses Overdispersion**:
- Heterogeneity comes from **discrete population structure**
- Variance = E[Var(r|z)] + Var[E(r|z)]
- Second term captures between-group variation
- Natural if trials conducted under distinct experimental conditions

**Data-Generating Process**:
- Trials sample from K distinct sub-populations
- Each sub-population has homogeneous success rate
- Examples: Different labs, time periods, treatment conditions
- Trial 1 (p=0) might be from low-probability regime
- Trials 2, 8, 11 (p > 0.11) from high-probability regime

**Assumptions and Validity**:
1. **Discrete regimes exist**: Tercile analysis (p < 0.05) supports this
2. **Within-group homogeneity**: Each component is true binomial (testable)
3. **Known K**: Must compare K=2 vs K=3 (model selection critical)
4. **No temporal structure**: Validated by EDA (runs test p = 0.23)

### 1.3 Falsification Criteria

**I will abandon this model if:**

1. **Poor component separation** (PRIMARY CRITERION)
   - If posterior P(z_i = k | data) < 0.6 for all trials
   - Component probabilities overlap heavily (credible intervals > 80% overlap)
   - Indicates continuous variation, not discrete groups

2. **Within-component overdispersion**
   - Posterior predictive checks show overdispersion WITHIN clusters
   - Chi-square test within each component: φ > 2.0
   - Means mixture doesn't capture all heterogeneity

3. **Unstable cluster assignments**
   - Trials switch components across MCMC samples
   - High posterior uncertainty in γ (entropy > 1.0 for K=3)
   - Label switching not resolved by ordering constraint

4. **K-sensitivity**
   - Model fit improves monotonically with K (no elbow)
   - WAIC prefers K > 5 (overfitting)
   - Suggests continuous variation, not discrete

5. **Poor predictive performance**
   - LOO-CV worse than Beta-Binomial baseline
   - Posterior predictive p-value < 0.05 for χ² test
   - Model fails to generate realistic data

**Diagnostic Checks**:
```
1. Component separation: mean(max(γ_i)) > 0.7
2. Within-component dispersion: φ_k < 1.5 for all k
3. Assignment stability: Cramer's V across posterior samples > 0.8
4. Model comparison: ΔWAIC < -2 compared to Beta-Binomial
5. Posterior predictive: p-value ∈ [0.2, 0.8] for test statistics
```

### 1.4 Implementation Considerations

**Computational Challenges**:

1. **Label switching**: Ordered constraint helps but may not fully resolve
   - Monitor: Check p[k] ordering holds across all samples
   - Solution: Post-process to align components if needed

2. **Local modes**: Multiple posterior modes with different cluster assignments
   - Strategy: Run 4 chains with dispersed inits
   - Diagnostic: Gelman-Rubin R̂ < 1.01 for all parameters

3. **Marginal likelihood**: Complex to compute for model comparison
   - Use WAIC/LOO instead of Bayes factors
   - Bridge sampling if needed for sensitivity

**Parameter Identifiability**:

1. **Ordering constraint ensures identifiability**: p_1 < p_2 < p_3
2. **But K is not identified**: Must fit K=2, K=3 separately
3. **Mixing proportions weakly identified** with N=12:
   - Wide posteriors expected
   - Don't over-interpret π estimates

**Prior Sensitivity**:

1. **Critical sensitivity**: Priors on p_k
   - Test: Uniform Beta(1,1) vs informative Beta(a_k, b_k)
   - If posteriors differ substantially, data is weak

2. **Moderate sensitivity**: Dirichlet concentration
   - Test: α = 1 (uniform) vs α = 2 (weak preference for balance)

3. **Low sensitivity**: Ordering constraint direction
   - Should not affect marginal likelihood

**Prior Predictive Checks**:
```python
# Simulate from prior to ensure reasonable support
for _ in range(1000):
    pi = dirichlet([2, 2, 2])
    p = sorted([beta(2, 48), beta(3, 42), beta(5, 40)])
    assert 0.01 < p[0] < 0.06  # Low group
    assert 0.04 < p[1] < 0.10  # Medium group
    assert 0.08 < p[2] < 0.15  # High group
```

---

## Model 2: Robust Contamination Model

### 2.1 Model Specification

**Core Idea**: Most trials follow Beta-Binomial, but some are contaminated by outlier process.

#### Mathematical Formulation

```
# Mixture of clean and contaminated processes
For each trial i:
  δ_i ~ Bernoulli(λ)                    # Contamination indicator

  If δ_i = 0 (clean):
    θ_i ~ Beta(α, β)                    # Draw from population distribution
    r_i | θ_i ~ Binomial(n_i, θ_i)      # Standard binomial

  If δ_i = 1 (contaminated):
    θ_i ~ Uniform(0, 1)                 # Outlier has arbitrary probability
    r_i | θ_i ~ Binomial(n_i, θ_i)      # Still binomial, but weird θ

# Priors
λ ~ Beta(1, 9)                          # Expect ~10% contamination
α, β ~ Gamma(2, 0.5)                    # Weakly informative on Beta shape
```

#### Specific Parameterization

```
# Contamination rate (conservative)
λ ~ Beta(1, 9)  # E[λ] = 0.1, 95% CI: [0.003, 0.35]

# Clean process hyperparameters
μ = α/(α+β)      # Population mean probability
κ = α + β        # Concentration (inverse variance)

# Hierarchical prior on (μ, κ)
μ ~ Beta(2, 25)  # Centers at pooled 0.074
κ ~ Gamma(2, 0.1)  # Allows φ ∈ [2, 10]

# Implied α, β
α = μ * κ
β = (1 - μ) * κ
```

#### Stan Implementation Strategy

```stan
data {
  int<lower=1> N;
  int<lower=0> r[N];
  int<lower=1> n[N];
}

parameters {
  real<lower=0, upper=1> lambda;      // Contamination probability
  real<lower=0, upper=1> mu;          // Clean process mean
  real<lower=0> kappa;                // Clean process concentration
  vector<lower=0, upper=1>[N] theta;  // Trial-specific probabilities
}

transformed parameters {
  real<lower=0> alpha = mu * kappa;
  real<lower=0> beta = (1 - mu) * kappa;
}

model {
  // Priors
  lambda ~ beta(1, 9);
  mu ~ beta(2, 25);
  kappa ~ gamma(2, 0.1);

  // Likelihood (mixture)
  for (i in 1:N) {
    target += log_mix(
      lambda,
      uniform_lpdf(theta[i] | 0, 1) + binomial_lpmf(r[i] | n[i], theta[i]),  // Contaminated
      beta_lpdf(theta[i] | alpha, beta) + binomial_lpmf(r[i] | n[i], theta[i]) // Clean
    );
  }
}

generated quantities {
  vector[N] prob_outlier;  // Posterior probability of contamination

  for (i in 1:N) {
    real lp_contam = log(lambda) + uniform_lpdf(theta[i] | 0, 1) +
                     binomial_lpmf(r[i] | n[i], theta[i]);
    real lp_clean = log(1 - lambda) + beta_lpdf(theta[i] | alpha, beta) +
                    binomial_lpmf(r[i] | n[i], theta[i]);
    prob_outlier[i] = exp(lp_contam - log_sum_exp(lp_contam, lp_clean));
  }
}
```

### 2.2 Theoretical Justification

**Why This Addresses Overdispersion**:
- Separates **structural variation** (Beta-Binomial) from **contamination**
- Outliers don't inflate population variance estimates
- Robust to extreme observations (trials 1, 8)

**Data-Generating Process**:
- Most trials: Legitimate sampling from heterogeneous population
- Few trials: Measurement error, protocol violation, different definition of success
- Trial 1 (0/47): Possible complete failure of process
- Trial 8 (31/215): Possible measurement artifact

**Assumptions and Validity**:
1. **Outliers exist but are rare**: EDA shows 2-3 extreme observations
2. **Contamination is random**: No systematic pattern (validated)
3. **Clean process is Beta-Binomial**: Standard assumption
4. **Outliers are uninformative**: Uniform(0,1) is agnostic

### 2.3 Falsification Criteria

**I will abandon this model if:**

1. **No clear outliers identified** (PRIMARY)
   - All prob_outlier[i] < 0.5
   - Lambda posterior concentrated near 0 (95% CI: [0, 0.05])
   - Indicates no contamination, just heterogeneity

2. **Too many "outliers"**
   - prob_outlier > 0.5 for more than 4 trials (33%)
   - Lambda posterior > 0.3
   - Means "contamination" is the main pattern (model inversion)

3. **Outliers not the expected ones**
   - Trials 1, 8 not flagged (prob_outlier < 0.3)
   - But other "normal" trials flagged
   - Indicates model misspecification

4. **Clean process still overdispersed**
   - Among trials with prob_outlier < 0.3, still φ > 2.0
   - Posterior predictive checks fail on non-outlier subset
   - Contamination doesn't explain all excess variation

5. **Worse predictive performance**
   - LOO worse than Beta-Binomial (ΔLOO > 2)
   - Overfitting to presumed outliers

**Diagnostic Checks**:
```
1. Outlier coherence: trials 1 OR 8 has prob_outlier > 0.6
2. Contamination rate: 0.05 < E[λ] < 0.25
3. Clean subset dispersion: φ_clean < 1.5
4. Predictive performance: LOO comparable to alternatives
5. Prior-posterior check: λ posterior differs from prior
```

### 2.4 Implementation Considerations

**Computational Challenges**:

1. **Bimodal posteriors**: Trials near decision boundary
   - Symptom: Low effective sample size for prob_outlier
   - Solution: Longer chains, careful initialization

2. **Weak identification of λ**: Only 12 observations
   - Prior matters substantially
   - Run prior sensitivity: Beta(1,9) vs Beta(1,19) vs Beta(2,18)

3. **Degeneracy**: If no outliers, λ → 0 boundary
   - Check: Posterior mass at λ < 0.01
   - Interpretation: Model reduces to Beta-Binomial

**Parameter Identifiability**:

1. **Trade-off between λ and κ**:
   - High contamination + high concentration OR
   - Low contamination + low concentration
   - Both explain overdispersion differently
   - Monitor joint posterior of (λ, κ)

2. **Individual θ_i weakly identified**:
   - Only n_i observations per trial
   - For small n_i, posterior ≈ prior
   - Expected and acceptable

**Prior Sensitivity Analysis**:

Test three prior scenarios:
```
# Conservative (prefer no outliers)
λ ~ Beta(1, 19)  # E[λ] = 0.05

# Moderate (baseline)
λ ~ Beta(1, 9)   # E[λ] = 0.10

# Liberal (expect outliers)
λ ~ Beta(1, 4)   # E[λ] = 0.20
```

If conclusions change qualitatively, data is insufficient to distinguish contamination from heterogeneity.

---

## Model 3: Latent Beta-Binomial with Structured Outlier Detection

### 3.1 Model Specification

**Core Idea**: Continuous Beta-Binomial variation PLUS explicit model for trials that don't fit.

#### Mathematical Formulation

```
# Two-stage hierarchical model
For each trial i:

  # Stage 1: Population model
  θ_i ~ Beta(α, β)                      # Base heterogeneity
  r_i^* | θ_i ~ Binomial(n_i, θ_i)      # Expected count

  # Stage 2: Outlier mechanism
  ω_i ~ Bernouial(ψ(θ_i, r_i^*))        # Outlier flag (data-dependent)

  If ω_i = 0:
    r_i = r_i^*                         # Observed matches expected

  If ω_i = 1:
    r_i ~ Binomial(n_i, θ_i^*)          # Draw from alternative process
    where θ_i^* ~ Beta(1, 1)            # Outlier probability

# Outlier probability function
ψ(θ, r^*) = logistic(η_0 + η_1 * |r - r^*|/√(n*θ*(1-θ)))

# Priors
α, β ~ Gamma(2, 0.5)                    # Population heterogeneity
η_0 ~ Normal(-2, 1)                     # Baseline outlier log-odds
η_1 ~ Normal(0, 1)                      # Sensitivity to deviations
```

#### Specific Parameterization

This is a **semi-supervised** approach: let data tell us what "outlier" means.

```
# Population Beta-Binomial (as baseline)
μ ~ Beta(2, 25)        # E[μ] ≈ 0.074
φ ~ Gamma(2, 0.5)      # E[φ] ≈ 4, allows overdispersion

α = μ * (1/φ - 1)
β = (1 - μ) * (1/φ - 1)

# Outlier detection mechanism
η_0 ~ Normal(-2, 1)    # E[P(outlier | no deviation)] ≈ 0.12
η_1 ~ Normal(1, 0.5)   # Deviations increase outlier probability
```

#### PyMC Implementation Strategy

Using PyMC for more flexible modeling:

```python
import pymc as pm
import numpy as np

with pm.Model() as outlier_detection_model:
    # Hyperpriors
    mu = pm.Beta('mu', alpha=2, beta=25)
    phi = pm.Gamma('phi', alpha=2, beta=0.5)

    alpha = pm.Deterministic('alpha', mu / phi)
    beta = pm.Deterministic('beta', (1 - mu) / phi)

    # Outlier detection parameters
    eta_0 = pm.Normal('eta_0', mu=-2, sigma=1)
    eta_1 = pm.Normal('eta_1', mu=1, sigma=0.5)

    # Trial-specific parameters
    theta_base = pm.Beta('theta_base', alpha=alpha, beta=beta, shape=N)

    # Expected counts under base model
    r_expected = n * theta_base

    # Standardized residuals (if observed r available)
    std_resid = (r_obs - r_expected) / pm.math.sqrt(n * theta_base * (1 - theta_base))

    # Outlier probability (increases with deviation)
    psi = pm.Deterministic('psi', pm.math.invlogit(eta_0 + eta_1 * pm.math.abs(std_resid)))

    # Outlier indicators (latent)
    omega = pm.Bernoulli('omega', p=psi, shape=N)

    # Outlier-specific probabilities
    theta_outlier = pm.Beta('theta_outlier', alpha=1, beta=1, shape=N)

    # Mixing: use base or outlier theta
    theta_eff = pm.Deterministic('theta_eff',
                                  omega * theta_outlier + (1 - omega) * theta_base)

    # Likelihood
    r = pm.Binomial('r', n=n, p=theta_eff, observed=r_obs)

    # Posterior inference
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
```

### 3.2 Theoretical Justification

**Why This Addresses Overdispersion**:
- **Primary source**: Beta-Binomial captures population heterogeneity
- **Secondary source**: Outliers contribute additional variance
- **Data-driven**: Model learns outlier definition from residuals
- **Flexible**: Can handle both continuous variation AND discrete outliers

**Data-Generating Process**:
- Most trials: Natural variation in success probability
- Some trials: Additional shock/perturbation to the process
- Outlier probability increases with poor fit to base model
- More principled than ad-hoc threshold

**Assumptions and Validity**:
1. **Hierarchical structure**: Population + individual trial variation
2. **Outliers are detectable**: Manifest as large residuals
3. **Outlier probability depends on deviation**: Seems reasonable
4. **Base model is Beta-Binomial**: Could be wrong (testable)

### 3.3 Falsification Criteria

**I will abandon this model if:**

1. **Outlier mechanism is unnecessary** (PRIMARY)
   - All posterior P(ω_i = 1) < 0.2
   - η_1 ≈ 0 (deviation doesn't predict outlier status)
   - Model reduces to Beta-Binomial (simpler model preferred)

2. **Outlier mechanism is too complex**
   - Computational issues: divergences, low ESS
   - Can't achieve R̂ < 1.01 even with strong priors
   - Overparameterized for n=12 observations

3. **Circular reasoning detected**
   - Outliers defined by residuals, but θ depends on outlier status
   - Feedback loop leads to unstable inference
   - Different random seeds give qualitatively different results

4. **Poor model fit**
   - Posterior predictive checks still fail
   - WAIC substantially worse than simpler alternatives
   - Model is too flexible (overfits)

5. **Uninterpretable outlier pattern**
   - ω doesn't flag expected trials (1, 8)
   - Flags random trials with no clear pattern
   - Suggests noise mining, not signal detection

**Diagnostic Checks**:
```
1. Model complexity justified: WAIC prefers this over Beta-Binomial
2. Outlier detection coherent: Trials 1 or 8 have E[ω] > 0.5
3. Computational health: R̂ < 1.01, ESS > 400, divergences < 1%
4. Parameter sensitivity: η_1 credible interval excludes 0
5. Posterior predictive: Generates data similar to observed
```

### 3.4 Implementation Considerations

**Computational Challenges**:

1. **Discrete latent variables (ω)**:
   - PyMC handles via marginalization or enumeration
   - For N=12, can enumerate 2^12 = 4096 states (feasible)
   - Alternative: NUTS with continuous relaxation

2. **Multimodality**:
   - Different outlier configurations have high posterior mass
   - Need multiple chains with dispersed initializations
   - Check: All chains converge to same marginal posteriors

3. **High-dimensional parameter space**:
   - N theta values + N omega values + hyperparameters
   - Requires strong sampling (target_accept = 0.95)
   - Expect slower sampling than simpler models

**Parameter Identifiability**:

1. **Trade-off between base and outlier**:
   - Large φ (low heterogeneity) + many outliers OR
   - Small φ (high heterogeneity) + few outliers
   - Both explain data similarly
   - Monitor: Joint posterior of (φ, sum(ω))

2. **Individual θ_outlier poorly identified**:
   - Only identified for trials with ω ≈ 1
   - For ω ≈ 0, θ_outlier is effectively prior
   - Don't interpret θ_outlier for non-outliers

3. **η parameters**:
   - Identified only if clear outlier/non-outlier separation
   - If all ω ≈ 0.5, η posterior ≈ prior
   - Check posterior differs substantially from prior

**Prior Sensitivity**:

Critical priors to test:
```
# Conservative (few outliers)
η_0 ~ Normal(-3, 0.5)  # Lower baseline

# Moderate (baseline)
η_0 ~ Normal(-2, 1)

# Liberal (many outliers)
η_0 ~ Normal(-1, 1)    # Higher baseline
```

Also test: Uniform(0,1) vs Beta(1,1) for θ_outlier

---

## Comparative Model Summary

| Feature | Finite Mixture | Robust Contamination | Structured Outlier |
|---------|---------------|---------------------|-------------------|
| **Philosophy** | Discrete regimes | Clean + noise | Continuous + anomaly |
| **Heterogeneity** | K-component mixture | Beta-Binomial + outliers | Beta-Binomial + data-driven |
| **Outlier handling** | Assigns to component | Flags as contaminated | Probabilistic detection |
| **Identifiability** | Moderate (ordering) | Good (clear structure) | Challenging (trade-offs) |
| **Complexity** | Medium | Medium | High |
| **Interpretability** | High (groups) | High (outliers) | Moderate (mechanism) |
| **Computational** | Stable | Stable | Challenging |
| **Prior sensitivity** | Moderate | High (λ) | High (η) |

---

## Unified Falsification Framework

### Global Red Flags (Abandon ALL Models)

1. **Data quality issue discovered**
   - Error in data transcription
   - Measurement units inconsistent
   - Trials not actually independent

2. **Fundamental distributional failure**
   - Binomial likelihood itself is wrong
   - Need multinomial or count model instead
   - Beta-Binomial baseline also fails badly

3. **Structural dependence detected**
   - Temporal autocorrelation emerges in posterior predictive
   - Sample size correlates with parameters in posterior
   - Violated exchangeability assumption

### Decision Tree for Model Selection

```
START: Fit all three models + Beta-Binomial baseline

STEP 1: Check computational health
  - If ANY model has R̂ > 1.01 or divergences > 5%:
    → Fix parameterization or abandon model

STEP 2: Compare to Beta-Binomial baseline
  - Compute ΔWAIC for each alternative model
  - If ALL models have ΔWAIC > -2:
    → Use Beta-Binomial (simpler is better)
    → STOP

STEP 3: Check model-specific falsification criteria
  - Finite Mixture: Component separation metric
  - Robust Contamination: Outlier identification coherence
  - Structured Outlier: Mechanism necessity
  - Eliminate models that fail criteria

STEP 4: Among remaining models
  - Prefer simplest with ΔWAIC < -2
  - If tie, use posterior predictive performance
  - If still tied, use interpretability

STEP 5: Sensitivity analysis on winner
  - Vary priors
  - Jackknife trials
  - If conclusions fragile → report uncertainty
```

---

## Stress Tests and Robustness Checks

### Stress Test 1: Trial Deletion

**Purpose**: Check if results driven by specific observations

```
For each model:
  - Refit excluding trial 1 (zero successes)
  - Refit excluding trial 8 (highest proportion)
  - Refit excluding trial 4 (largest sample)

Compare:
  - Parameter posterior means shift < 20%
  - Model selection decision unchanged
  - Outlier/component assignments stable for remaining trials
```

**Abandon model if**: Conclusions qualitatively change when removing one trial

### Stress Test 2: Prior Perturbation

**Purpose**: Ensure conclusions data-driven, not prior-driven

```
For each model:
  - Fit with weakly informative priors (baseline)
  - Fit with very weak priors (e.g., Uniform)
  - Fit with stronger priors (more concentrated)

Compare:
  - KL divergence between posteriors < 0.5
  - Predictive performance similar (ΔLOO < 2)
  - Key parameters shift < 30%
```

**Abandon model if**: Prior choice substantially changes model preference

### Stress Test 3: Alternative Likelihood

**Purpose**: Check if Binomial likelihood is appropriate

```
For best model:
  - Refit with Beta-Binomial likelihood (continuous θ marginalized)
  - Refit with Quasibinomial (explicit overdispersion parameter)
  - Compare fit
```

**Abandon Binomial entirely if**: Alternative likelihoods dramatically better

### Stress Test 4: Posterior Predictive Extremes

**Purpose**: Can model generate data as extreme as observed?

```
For each model:
  - Generate 1000 posterior predictive datasets
  - Compute: min(p), max(p), φ, number with r=0
  - Compare to observed values

Check:
  - Observed min in middle 95% of predictive
  - Observed max in middle 95% of predictive
  - Observed φ in middle 95% of predictive
```

**Abandon model if**: Consistently fails to generate extremes (too conservative)

---

## Implementation Roadmap

### Phase 1: Initial Fitting (Priority 1)

1. **Implement Finite Mixture (K=2, K=3)** in Stan
   - Fit both models
   - Check convergence, label switching
   - Compute component separation metrics
   - Generate posterior predictive datasets

2. **Implement Robust Contamination** in Stan
   - Fit with baseline priors
   - Check outlier identification
   - Examine clean subset dispersion

3. **Implement Structured Outlier** in PyMC
   - May be slow, budget extra time
   - Focus on computational stability
   - Monitor divergences carefully

4. **Fit Beta-Binomial baseline** (for comparison)

### Phase 2: Diagnostics (Priority 1)

1. **Computational health**: R̂, ESS, divergences
2. **Prior-posterior comparison**: Did we learn?
3. **Posterior predictive checks**: Do models generate realistic data?
4. **Model comparison**: WAIC, LOO-CV

### Phase 3: Falsification (Priority 2)

1. **Apply model-specific criteria** from sections 1.3, 2.3, 3.3
2. **Eliminate failed models**
3. **Document reasons for rejection**

### Phase 4: Stress Tests (Priority 2)

1. **Trial deletion** (jackknife)
2. **Prior sensitivity**
3. **Alternative likelihoods**
4. **Posterior predictive extremes**

### Phase 5: Selection (Priority 3)

1. **Apply decision tree**
2. **Choose best model(s)**
3. **If multiple viable, report ensemble or uncertainty**

---

## Expected Outcomes and Pivots

### Scenario 1: Finite Mixture Wins

**Evidence**:
- Clear component separation (mean max(γ) > 0.75)
- K=3 preferred by WAIC
- Components align with tercile analysis
- Interpretable group structure

**Interpretation**:
- Trials truly from discrete populations
- Report group-specific success probabilities
- Investigate: Why do groups differ? (domain knowledge)

**Next steps**:
- Assign trials to components
- Report π, p_k with credible intervals
- Covariate analysis if metadata available

### Scenario 2: Robust Contamination Wins

**Evidence**:
- 2-3 trials clearly flagged as outliers (prob > 0.7)
- Includes trial 1 and/or trial 8
- Clean subset has φ < 1.5
- Better LOO than finite mixture

**Interpretation**:
- Legitimate heterogeneity + some contamination
- Outliers should be investigated/excluded
- Population inference based on clean trials

**Next steps**:
- Report clean population parameters (μ, κ)
- Flag outlier trials for follow-up
- Sensitivity: Include vs exclude outliers

### Scenario 3: All Alternative Models Fail

**Evidence**:
- All ΔWAIC > -2 vs Beta-Binomial
- Falsification criteria trigger for all
- Computational issues
- No clear outliers or components

**Interpretation**:
- Overdispersion is smooth/continuous
- No discrete structure or contamination
- Beta-Binomial is sufficient

**PIVOT**:
- **Accept Beta-Binomial as best model**
- Focus on: Prior specification, hierarchical extensions
- Consider: Covariates, non-parametric alternatives

### Scenario 4: Multiple Models Viable

**Evidence**:
- Similar WAIC (within ±2)
- All pass falsification tests
- Different interpretations

**Interpretation**:
- Data cannot distinguish mechanisms
- Fundamental uncertainty about DGP

**Report**:
- **Model ensemble** or **model averaging**
- Sensitivity analysis across models
- Emphasize robust conclusions (hold across models)
- Flag fragile conclusions (model-dependent)

---

## Alternative Approaches if All Fail

### Backup Plan 1: Non-Parametric Beta-Binomial

If Beta parametrization is too restrictive:

```
θ_i ~ G where G ~ DirichletProcess(α, G_0)
G_0 = Beta(a, b)
r_i ~ Binomial(n_i, θ_i)
```

**Advantages**: Learns number of components from data
**Disadvantages**: Complex, may overfit with N=12

### Backup Plan 2: Hierarchical Logistic-Normal

If Beta distribution wrong shape:

```
logit(θ_i) ~ Normal(μ, σ²)
r_i ~ Binomial(n_i, θ_i)
```

**Advantages**: More flexible tails
**Disadvantages**: Less interpretable

### Backup Plan 3: Explicit Covariate Search

If unmodeled structure suspected:

```
logit(θ_i) = β_0 + β_1 * f(n_i) + β_2 * g(trial_id)
```

Even if EDA shows no effect, model-based analysis might reveal subtle patterns

---

## Documentation Requirements

For each fitted model, report:

1. **Model specification**: Full mathematical notation
2. **Prior choices**: Justification and sensitivity
3. **Computational diagnostics**: R̂, ESS, divergences, runtime
4. **Posterior summaries**: All parameters with 95% CI
5. **Model fit**: WAIC, LOO, posterior predictive p-values
6. **Falsification results**: Which criteria checked, pass/fail
7. **Stress test results**: Robustness to perturbations
8. **Interpretation**: What does model say about DGP?

---

## Critical Success Factors

This modeling exercise succeeds if:

1. **We identify clear failure modes**: Know why models DON'T work
2. **We document decision process**: Transparent, reproducible
3. **We acknowledge uncertainty**: No false precision
4. **We challenge assumptions**: Don't accept defaults blindly
5. **We prioritize learning over completion**: Finding truth > finishing tasks

**Remember**:
- The "best" model might be "none of these"
- Negative results are valuable results
- Computational failure often indicates conceptual failure
- Simple models that work beat complex models that don't

---

## Final Notes

These three models represent fundamentally different perspectives:

1. **Finite Mixture**: "The world has distinct states"
2. **Robust Contamination**: "Most data is good, some is corrupt"
3. **Structured Outlier**: "Learn what 'outlier' means from data"

They will likely give **different answers**. That's the point. We're testing competing hypotheses, not confirming a predetermined story.

**If the models agree**: Great, robust conclusion.
**If the models disagree**: Even better, we've learned the data is ambiguous.

The worst outcome is picking one model arbitrarily and over-interpreting it. The best outcome is understanding which aspects of the DGP we CAN and CANNOT identify with 12 observations.

---

**Files Created**:
- `/workspace/experiments/designer_3/proposed_models.md` (this document)

**Next Steps**:
- Implement models in Stan/PyMC
- Run diagnostics and falsification tests
- Synthesize with other designers' proposals
- Select best model(s) or report irreducible uncertainty
