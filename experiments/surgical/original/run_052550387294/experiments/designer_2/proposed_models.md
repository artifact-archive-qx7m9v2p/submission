# Bayesian Model Proposals for Overdispersed Binomial Data
## Model Designer 2: Hierarchical & Group-Based Focus

**Date**: 2025-10-30
**Designer**: Model Designer 2
**Focus**: Hierarchical structures and group-based heterogeneity models

---

## Executive Summary

I propose **three distinct Bayesian model classes** to explain the observed overdispersion (φ = 3.51):

1. **Hierarchical Beta-Binomial** - Continuous probability heterogeneity with partial pooling
2. **Finite Mixture Model** - Discrete probability regimes (2-3 groups)
3. **Non-Centered Hierarchical Logit** - Trial-specific effects with efficient parameterization

**Key philosophical stance**: I will prioritize models that can FAIL CLEARLY. Each model makes different assumptions about the heterogeneity structure, and I've designed falsification criteria that would force me to abandon each approach.

**Critical Warning**: With only 12 observations, we face severe identifiability challenges. I may need to abandon ALL these models if evidence suggests the data-generating process is more complex than we can reliably identify.

---

## Model 1: Hierarchical Beta-Binomial Model

### 1.1 Model Specification

**Likelihood**:
```
r_i | θ_i, n_i ~ Binomial(n_i, θ_i)    for i = 1, ..., 12
θ_i | μ, κ ~ Beta(μκ, (1-μ)κ)
```

**Hyperpriors**:
```
μ ~ Beta(2, 25)           # pooled probability, E[μ] ≈ 0.074
κ ~ Gamma(2, 0.1)         # concentration parameter
```

**Parameterization Notes**:
- μ = mean success probability across trials
- κ = concentration (larger κ = less heterogeneity)
- Var(θ) = μ(1-μ)/(κ+1)
- For observed φ = 3.5, we expect κ ∈ [2, 10]

**Mathematical Details**:
```
Alpha = μκ
Beta = (1-μ)κ

E[θ_i] = μ
Var[θ_i] = μ(1-μ)/(κ+1)

Overdispersion factor:
φ = 1 + 1/(κ+1)

For φ = 3.5:
1 + 1/(κ+1) = 3.5
κ ≈ -0.6  ← PROBLEM! κ must be positive!
```

**CRITICAL REALIZATION**: The observed overdispersion φ = 3.51 is **incompatible** with the standard Beta-Binomial overdispersion formula φ = 1 + 1/(κ+1), which has maximum φ = 1 as κ→0.

**This means**:
1. The overdispersion here is NOT the Beta-Binomial intraclass correlation
2. The χ² statistic φ = χ²/df measures something different
3. This model CAN still work, but I'm measuring different overdispersion concepts

**Corrected Interpretation**:
- The Beta-Binomial naturally induces overdispersion
- Small κ → high heterogeneity in θ_i
- The χ² overdispersion factor and Beta-Binomial ICC are different quantities
- I will use posterior predictive checks to see if the model captures the observed variance

### 1.2 Implementation Strategy

**Stan Code Structure**:
```stan
data {
  int<lower=1> N;           // 12 trials
  int<lower=0> n[N];        // sample sizes
  int<lower=0> r[N];        // success counts
}

parameters {
  real<lower=0, upper=1> mu;           // pooled probability
  real<lower=0> kappa;                 // concentration
  vector<lower=0, upper=1>[N] theta;   // trial-specific probabilities
}

model {
  // Hyperpriors
  mu ~ beta(2, 25);
  kappa ~ gamma(2, 0.1);

  // Hierarchical structure
  theta ~ beta(mu * kappa, (1 - mu) * kappa);

  // Likelihood
  r ~ binomial(n, theta);
}

generated quantities {
  vector[N] r_rep;
  real overdispersion_check;

  // Posterior predictive
  for (i in 1:N) {
    r_rep[i] = binomial_rng(n[i], theta[i]);
  }

  // Check if we reproduce observed variance
  overdispersion_check = variance(r_rep ./ to_vector(n)) /
                         variance(to_vector(r) ./ to_vector(n));
}
```

**PyMC Implementation**:
```python
import pymc as pm

with pm.Model() as beta_binomial_model:
    # Hyperpriors
    mu = pm.Beta('mu', alpha=2, beta=25)
    kappa = pm.Gamma('kappa', alpha=2, beta=0.1)

    # Trial-specific probabilities
    theta = pm.Beta('theta', alpha=mu*kappa, beta=(1-mu)*kappa, shape=12)

    # Likelihood
    r_obs = pm.Binomial('r_obs', n=n, p=theta, observed=r)

    # Posterior predictive
    r_pred = pm.Binomial('r_pred', n=n, p=theta, shape=12)
```

### 1.3 Theoretical Justification

**Data-Generating Process**:
This model assumes trials are **exchangeable draws** from a population where:
- Each trial has its own true probability θ_i
- These probabilities vary continuously according to a Beta distribution
- The Beta distribution is parameterized by population-level parameters μ and κ

**Physical Interpretation**:
- Trials sample from subtly different populations
- Differences are continuous, not discrete
- Natural shrinkage: extreme observations pulled toward population mean
- Uncertainty propagates from trial level → population level

**Why This Addresses Overdispersion**:
1. **Extra variance layer**: θ_i varies, inducing variance beyond binomial
2. **Partial pooling**: Borrows strength across trials for stable estimates
3. **Natural regularization**: Extreme trials (like trial 1, 8) shrunk toward μ
4. **Explicit uncertainty**: Posterior for κ quantifies heterogeneity magnitude

**Assumptions & Validity**:

✓ **Valid**:
- Trials are independent (supported by runs test p=0.23)
- Exchangeability (no sample size effect p>0.7, no temporal trend p>0.2)
- Continuous heterogeneity is plausible

? **Questionable**:
- Beta distribution for θ_i (why not another continuous distribution?)
- Symmetric variation around μ (could have asymmetric heterogeneity)
- No covariates (are we missing important structure?)

✗ **Potentially Invalid**:
- If true structure is discrete groups, continuous Beta may be misspecified
- If heterogeneity is driven by unmeasured covariates, this is inefficient

### 1.4 Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Prior-posterior conflict on κ**:
   - If posterior κ pushes hard against prior boundary (κ → 0 or κ → ∞)
   - Suggests Beta distribution is wrong shape for heterogeneity
   - **Diagnostic**: Check κ posterior vs prior, look for 95% CI near boundaries

2. **Posterior predictive failure**:
   - If p-value for observed variance < 0.05 or > 0.95
   - Model cannot reproduce the overdispersion pattern
   - **Diagnostic**: Calculate P(variance(r_rep) > variance(r_obs))

3. **Systematic residual patterns**:
   - If standardized residuals correlate with sample size, trial ID, or fitted values
   - Suggests missing structure
   - **Diagnostic**: Plot residuals vs predictors, test for correlation

4. **Bimodal θ_i posterior**:
   - If posterior for θ_i shows clear bimodality/multimodality
   - Suggests discrete groups, not continuous variation
   - **Diagnostic**: Examine θ_i marginal posteriors, compute bimodality coefficient

5. **LOO-CV warnings**:
   - If >2 observations have Pareto-k > 0.7
   - Indicates influential points or model misspecification
   - **Diagnostic**: Run LOO-CV, examine high-influence trials

6. **Shrinkage failure for extreme trials**:
   - If θ_1 (trial 1, 0/47) doesn't shrink toward μ
   - If θ_8 (trial 8, 31/215) doesn't shrink toward μ
   - Suggests model doesn't capture structure
   - **Diagnostic**: Compare posterior θ_i to MLE p̂_i, check shrinkage magnitude

**Decision Rule**: If ≥3 of these criteria trigger, ABANDON this model and pivot to Model 2 or 3.

### 1.5 Implementation Considerations

**Computational Challenges**:

1. **Divergent transitions** (likely):
   - Beta distribution with small α, β can have funnel geometry
   - **Solution**: Try non-centered parameterization:
     ```
     θ_raw ~ Beta(μκ, (1-μ)κ)
     θ = logit^{-1}(logit(μ) + σ * logit(θ_raw))
     ```
   - Or use log(κ) parameterization

2. **κ near zero**:
   - If true heterogeneity is extreme, κ → 0 creates sampling difficulties
   - **Solution**: Put informative lower bound κ > 0.1, or switch to Model 3

3. **Label switching** (not applicable here):
   - Not a mixture model, so no label switching issues

**Identifiability Concerns**:

1. **μ and κ trade-off**:
   - With 12 observations, hard to distinguish between:
     - High μ, low κ (high mean, high variance)
     - Low μ, moderate κ (low mean, moderate variance)
   - **Check**: Examine μ-κ joint posterior for strong correlation

2. **Trial 1 (0/47) influence**:
   - Zero successes forces θ_1 near 0
   - May pull μ down and inflate heterogeneity estimate
   - **Mitigation**: Run sensitivity analysis excluding trial 1

3. **Trial 4 (810 observations) dominance**:
   - Largest sample may dominate μ estimate
   - **Check**: Compare posterior with/without trial 4

**Prior Sensitivity**:

Test these alternative priors:

1. **Weak prior**: μ ~ Beta(1, 1), κ ~ Gamma(1, 0.01)
2. **Strong prior**: μ ~ Beta(3, 37) [tighter around 0.074], κ ~ Gamma(3, 0.3)
3. **Different κ prior**: κ ~ Exponential(0.2), or κ ~ HalfNormal(5)

**Expected Results**:
- μ posterior centered near 0.074 but wider than binomial MLE
- κ posterior likely in range [1, 20]
- θ_i posteriors: trial 1 near 0.02, trial 8 near 0.12, others between

---

## Model 2: Finite Mixture Model (2-3 Components)

### 2.1 Model Specification

**Two-Component Version**:
```
Component assignment: z_i ~ Categorical(π)  where π = (π_1, π_2), Σπ_k = 1
Component probabilities: p_k ~ Beta(a_k, b_k)  for k = 1, 2
Likelihood: r_i | z_i=k, n_i ~ Binomial(n_i, p_k)
```

**Priors**:
```
π ~ Dirichlet(α_1, α_2)     where α = (2, 2) [weakly informative]
p_1 ~ Beta(2, 40)           # Low probability group, E[p_1] ≈ 0.05
p_2 ~ Beta(4, 30)           # High probability group, E[p_2] ≈ 0.12
```

**Three-Component Version** (if 2-component inadequate):
```
π ~ Dirichlet(1, 1, 1)
p_1 ~ Beta(2, 50)           # Low: E[p] ≈ 0.04
p_2 ~ Beta(3, 40)           # Medium: E[p] ≈ 0.07
p_3 ~ Beta(5, 40)           # High: E[p] ≈ 0.11
```

**Identifiability Constraints**:
```
Ordered constraint: p_1 < p_2 [< p_3]
This prevents label switching during MCMC
```

### 2.2 Implementation Strategy

**Stan Code Structure**:
```stan
data {
  int<lower=1> N;
  int<lower=0> n[N];
  int<lower=0> r[N];
  int<lower=2> K;  // number of components (2 or 3)
}

parameters {
  simplex[K] pi;                           // mixing proportions
  ordered[K] p_logit;                      // ordered logit(p_k) for identifiability
}

transformed parameters {
  vector<lower=0, upper=1>[K] p;
  p = inv_logit(p_logit);
}

model {
  vector[K] log_pi = log(pi);

  // Priors
  pi ~ dirichlet(rep_vector(2.0, K));

  // Component probabilities (on logit scale for better sampling)
  p_logit[1] ~ normal(logit(0.05), 1);
  p_logit[2] ~ normal(logit(0.12), 1);

  // Likelihood (marginalizing over component assignments)
  for (i in 1:N) {
    vector[K] log_lik;
    for (k in 1:K) {
      log_lik[k] = log_pi[k] + binomial_lpmf(r[i] | n[i], p[k]);
    }
    target += log_sum_exp(log_lik);
  }
}

generated quantities {
  // Component assignments (posterior probabilities)
  matrix[N, K] gamma;  // responsibility matrix
  vector[N] z_pred;    // most likely component

  for (i in 1:N) {
    vector[K] log_lik;
    for (k in 1:K) {
      log_lik[k] = log(pi[k]) + binomial_lpmf(r[i] | n[i], p[k]);
    }
    gamma[i] = softmax(log_lik)';
    z_pred[i] = categorical_rng(gamma[i]');
  }
}
```

**PyMC Implementation**:
```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as mixture_model:
    # Number of components
    K = 2

    # Mixing proportions
    pi = pm.Dirichlet('pi', a=np.array([2., 2.]))

    # Component probabilities (ordered for identifiability)
    p_ordered = pm.Beta('p_ordered', alpha=[2, 4], beta=[40, 30], shape=K)

    # Ensure ordering (this is tricky in PyMC, may need custom distribution)
    # Alternative: use transformations

    # Likelihood - marginalizing over component assignments
    def logp(r, n, pi, p):
        # For each observation, compute mixture likelihood
        logp_components = pm.math.stack([
            pm.logp(pm.Binomial.dist(n=n, p=p[k]), r) + pm.math.log(pi[k])
            for k in range(K)
        ], axis=1)
        return pm.logsumexp(logp_components, axis=1)

    # Use pm.Potential for custom likelihood
    pm.Potential('mixture_logp',
                 logp(r_obs, n_obs, pi, p_ordered).sum())
```

### 2.3 Theoretical Justification

**Data-Generating Process**:
- Trials come from **K distinct populations** with different success rates
- Each trial is randomly assigned to one population
- Assignment probabilities π reflect population prevalence
- Within each population, success probability is constant

**Physical Interpretation**:
Could represent:
1. **Different experimental conditions** (unknown to analyst)
2. **Batch effects** (2-3 different batches)
3. **Population stratification** (mixing subpopulations)
4. **Regime changes** (e.g., before/after process change)

**Why This Addresses Overdispersion**:
1. **Discrete heterogeneity**: p_k values differ, creating extra-binomial variance
2. **Natural clustering**: EDA showed median-split groups differ (p=0.012)
3. **Tercile analysis**: Low (p≈0.04), Medium (p≈0.07), High (p≈0.11) groups
4. **Explicit group structure**: Can identify which trials belong to which regime

**Evidence Supporting This Model**:
- Median split t-test: p = 0.012 (significant difference)
- Visual examination suggests possible 2-3 clusters
- Tercile analysis shows gradient: 0.039 → 0.067 → 0.115
- Not inconsistent with continuous variation, but discrete may be simpler

**Assumptions & Validity**:

✓ **Valid**:
- Trials are independent
- Each trial belongs to exactly one component
- Component membership is latent (unobserved)

? **Questionable**:
- Why exactly 2 (or 3) components? Could be 4, 5, or continuous
- Are components well-separated or overlapping?
- Is ordering stable or could p_1 ≈ p_2?

✗ **Potentially Invalid**:
- If true heterogeneity is continuous, discrete mixture is misspecified
- With 12 observations, may overfit (too many parameters for K=3)
- If components overlap heavily, identifiability problems

### 2.4 Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **Component collapse**:
   - If π_k < 0.1 for any component (one component nearly empty)
   - Suggests fewer components needed
   - **Diagnostic**: Examine π posterior, check if any component has <1-2 trials

2. **Component overlap**:
   - If 95% CI for p_1 and p_2 overlap heavily (>50% overlap)
   - Components not well-separated, continuous model better
   - **Diagnostic**: Plot posterior p_k densities, compute overlap area

3. **Uncertain assignments**:
   - If >50% of trials have maximum γ_ik < 0.7 (no clear component membership)
   - Suggests mixture structure doesn't match data
   - **Diagnostic**: Examine responsibility matrix γ, count ambiguous assignments

4. **Model selection favors K=1**:
   - If WAIC/LOO strongly favors simple binomial over mixture
   - Overdispersion not explained by discrete groups
   - **Diagnostic**: Compare WAIC for K=1,2,3

5. **Posterior predictive failure**:
   - If mixture cannot reproduce observed variance or outliers
   - **Diagnostic**: Generate r_rep, compare to r_obs distribution

6. **Label switching despite constraints**:
   - If MCMC shows bimodal posteriors for p_k (swapping)
   - Identifiability problem
   - **Diagnostic**: Trace plots for p_k, check for mode switching

**Decision Rule**: If ≥2 of these trigger, switch to Model 1 (continuous) or Model 3 (trial-specific).

### 2.5 Implementation Considerations

**Computational Challenges**:

1. **Marginalization over z_i**:
   - Must integrate out discrete component assignments
   - Can be slow for large K or N
   - **Solution**: Use log_sum_exp trick for numerical stability

2. **Label switching**:
   - MCMC may swap component labels
   - **Solution**: Enforce ordering constraint p_1 < p_2 < ... < p_K
   - Or post-process to align labels

3. **Local modes**:
   - Likelihood may have multiple modes
   - Different initializations may find different solutions
   - **Solution**: Run multiple chains with different starting values

4. **Choosing K**:
   - No clear answer for 2 vs 3 components
   - **Solution**: Fit both, compare via LOO-CV/WAIC

**Identifiability Concerns**:

1. **K too large**:
   - With 12 observations, K=3 means only 4 trials per component on average
   - May lead to overfitting or non-identifiability
   - **Mitigation**: Start with K=2, only increase if clearly needed

2. **Empty components**:
   - Some components may attract zero trials
   - **Check**: Examine π posterior, expected number of trials per component

3. **π and p trade-offs**:
   - Can fit data with either:
     - Two balanced components with moderate separation
     - Two unbalanced components with large separation
   - **Check**: Joint posterior for (π, p_1, p_2)

**Prior Sensitivity**:

Test these alternatives:

1. **Uninformative π**: Dirichlet(1, 1, ..., 1)
2. **Informative p_k**: Based on tercile analysis
   - p_1 ~ Beta(4, 96) centered at 0.04
   - p_2 ~ Beta(7, 93) centered at 0.07
   - p_3 ~ Beta(11, 89) centered at 0.11
3. **Wider p_k**: Beta(1, 10) for all components

**Expected Results**:
- K=2: π ≈ (0.5, 0.5), p_1 ≈ 0.05, p_2 ≈ 0.11
- High-probability component: trials 2, 8, 10, 11
- Low-probability component: trials 1, 4, 5, 7
- Ambiguous: trials 3, 6, 9, 12

---

## Model 3: Non-Centered Hierarchical Logit Model

### 3.1 Model Specification

**Parameterization**:
```
Trial-level effects (non-centered):
  η_i ~ Normal(0, σ)                    # standardized effects

Logit-scale probability:
  logit(θ_i) = μ_logit + η_i           # additive on logit scale

Population parameters:
  μ_logit ~ Normal(logit(0.074), 1.5)  # pooled logit-probability
  σ ~ HalfNormal(2)                     # heterogeneity scale

Likelihood:
  r_i | θ_i, n_i ~ Binomial(n_i, θ_i)
```

**Equivalent Centered Parameterization** (for comparison):
```
logit(θ_i) ~ Normal(μ_logit, σ)
r_i ~ Binomial(n_i, θ_i)
```

**Why Non-Centered?**
- Better geometry for HMC when σ is small
- Separates location (μ_logit) from scale (σ) effects
- Typically faster convergence and fewer divergences

### 3.2 Implementation Strategy

**Stan Code Structure**:
```stan
data {
  int<lower=1> N;
  int<lower=0> n[N];
  int<lower=0> r[N];
}

parameters {
  real mu_logit;              // population mean on logit scale
  real<lower=0> sigma;        // heterogeneity standard deviation
  vector[N] eta;              // standardized trial effects
}

transformed parameters {
  vector[N] logit_theta;
  vector<lower=0, upper=1>[N] theta;

  logit_theta = mu_logit + sigma * eta;
  theta = inv_logit(logit_theta);
}

model {
  // Priors
  mu_logit ~ normal(-2.5, 1.5);  // logit(0.074) ≈ -2.52
  sigma ~ normal(0, 2);           // half-normal via truncation
  eta ~ std_normal();             // standard normal

  // Likelihood
  r ~ binomial(n, theta);
}

generated quantities {
  real mu_prob = inv_logit(mu_logit);  // back to probability scale
  vector[N] r_rep;

  for (i in 1:N) {
    r_rep[i] = binomial_rng(n[i], theta[i]);
  }

  // Compute effective overdispersion
  real var_theta = variance(theta);
  real expected_var = mu_prob * (1 - mu_prob);  // binomial variance
}
```

**PyMC Implementation**:
```python
import pymc as pm

with pm.Model() as logit_hierarchical:
    # Hyperpriors
    mu_logit = pm.Normal('mu_logit', mu=-2.5, sigma=1.5)
    sigma = pm.HalfNormal('sigma', sigma=2)

    # Non-centered parameterization
    eta = pm.Normal('eta', mu=0, sigma=1, shape=12)
    logit_theta = pm.Deterministic('logit_theta', mu_logit + sigma * eta)
    theta = pm.Deterministic('theta', pm.math.invlogit(logit_theta))

    # Likelihood
    r_obs = pm.Binomial('r_obs', n=n, p=theta, observed=r)

    # Population probability
    mu_prob = pm.Deterministic('mu_prob', pm.math.invlogit(mu_logit))
```

### 3.3 Theoretical Justification

**Data-Generating Process**:
- Each trial has its own latent effect η_i on the logit scale
- Effects are normally distributed around population mean μ_logit
- Scale σ controls degree of heterogeneity
- Logit link ensures probabilities stay in (0,1)

**Why Logit Scale?**
1. **Natural for binomial data**: Canonical link function
2. **Unbounded variation**: η_i can range over (-∞, ∞)
3. **Symmetric on logit scale**: Easier to model with Normal distribution
4. **Multiplicative on odds scale**: Interpretable as odds ratios

**Physical Interpretation**:
- Each trial has log-odds: log(θ_i/(1-θ_i)) = μ_logit + η_i
- Deviations η_i represent trial-specific factors (unmeasured covariates)
- σ measures "strength" of trial-to-trial variation
- Normal distribution on logit scale → skewed distribution on probability scale

**Why This Addresses Overdispersion**:
1. **Additive effects on logit scale**: Each trial gets its own adjustment
2. **Flexible heterogeneity**: σ controls spread, data determines magnitude
3. **Natural shrinkage**: Small-sample trials shrunk more toward μ_logit
4. **Efficient parameterization**: Often better computational properties than Model 1

**Advantages Over Model 1**:
- Better HMC geometry (non-centered parameterization)
- More natural link function for binomial data
- Easier to extend with covariates: logit(θ_i) = μ_logit + X_i*β + σ*η_i
- Often more stable posterior sampling

**Advantages Over Model 2**:
- Fewer parameters (no component assignments)
- Continuous variation (more flexible if groups aren't sharp)
- No label switching issues

**Assumptions & Validity**:

✓ **Valid**:
- Trials independent
- Logit scale is natural for binomial
- Normal distribution on logit scale is flexible

? **Questionable**:
- Why Normal distribution for η_i? Could be Student-t (heavier tails)
- Is variation symmetric on logit scale?
- Do we need trial-specific parameters with only 12 trials?

✗ **Potentially Invalid**:
- If heterogeneity has specific structure (groups, covariates), this misses it
- If variation is better modeled on probability scale (Beta), this is inefficient

### 3.4 Falsification Criteria

**I WILL ABANDON THIS MODEL IF**:

1. **σ posterior at boundary**:
   - If 95% CI for σ includes 0 or is very large (>5)
   - σ ≈ 0: No heterogeneity, use simple binomial
   - σ >> 1: Model likely misspecified
   - **Diagnostic**: Examine σ posterior, check boundary behavior

2. **Non-normality of η_i**:
   - If posterior η_i values show heavy tails or multimodality
   - Normal assumption violated
   - **Diagnostic**: Q-Q plot of posterior mean η_i vs theoretical N(0,1)

3. **Computational pathologies**:
   - If >1% divergent transitions persist despite tuning
   - If non-centered parameterization doesn't help
   - Suggests fundamental geometry problem
   - **Diagnostic**: Check divergence rate, examine trace plots

4. **Posterior predictive failure**:
   - Cannot reproduce observed variance or outliers
   - **Diagnostic**: Compare variance(r_rep) to variance(r_obs)

5. **Asymmetric residuals**:
   - If residuals show systematic skewness
   - Suggests logit scale inappropriate
   - **Diagnostic**: Test skewness of (r_i/n_i - θ_i), expect ≈0

6. **Extreme shrinkage**:
   - If all θ_i posteriors collapse to μ_prob (over-shrinkage)
   - Model too regularized, not capturing heterogeneity
   - **Diagnostic**: Compare posterior SD(θ_i) to observed SD(r_i/n_i)

**Decision Rule**: If ≥3 criteria trigger, pivot to Model 1 or Model 2.

### 3.5 Implementation Considerations

**Computational Challenges**:

1. **Centered vs non-centered**:
   - Non-centered better when σ is small (high pooling)
   - Centered better when σ is large (weak pooling)
   - With σ uncertain, non-centered is safer default
   - **Solution**: Could implement both and switch based on σ posterior

2. **Funnel geometry** (less severe than Model 1):
   - When σ → 0, η_i becomes non-identified
   - Non-centered parameterization mitigates this
   - **Check**: Examine σ vs η_i joint posteriors

3. **Prior sensitivity**:
   - Results may depend on σ prior
   - **Test**: HalfNormal(1), HalfNormal(2), Exponential(0.5)

**Identifiability Concerns**:

1. **μ_logit and η_i sum-to-zero constraint**:
   - Technically, could add constant to μ_logit and subtract from all η_i
   - Not a problem in practice because η_i ~ N(0, σ) centers them
   - **Check**: Ensure E[η_i] ≈ 0 in posterior

2. **Trial 1 (0/47) influence**:
   - Zero successes implies θ_1 near 0, hence η_1 << 0
   - May inflate σ estimate
   - **Mitigation**: Sensitivity analysis without trial 1

3. **Small sample regularization**:
   - With 12 trials, estimating 14 parameters (μ_logit, σ, 12×η_i)
   - Hierarchical structure provides regularization
   - **Check**: Effective sample size for all parameters

**Prior Sensitivity**:

Test these alternatives:

1. **Weak priors**:
   - mu_logit ~ Normal(0, 5)
   - sigma ~ HalfNormal(5)

2. **Strong priors**:
   - mu_logit ~ Normal(-2.5, 0.5)  # tighter around logit(0.074)
   - sigma ~ HalfNormal(1)         # expects less heterogeneity

3. **Heavy-tailed alternative**:
   - Replace eta ~ Normal(0, 1) with eta ~ StudentT(nu=4, 0, 1)
   - Allows for outlier trials

**Expected Results**:
- mu_logit ≈ -2.5 (probability ≈ 0.074)
- sigma ≈ 0.5-1.5 (moderate heterogeneity)
- η_1 << 0 (trial 1, zero successes)
- η_8 >> 0 (trial 8, highest proportion)
- Shrinkage stronger for small-n trials

---

## Model Comparison Strategy

### 4.1 How to Choose Between Models

I will fit **all three models** and compare using:

1. **LOO-CV (Leave-One-Out Cross-Validation)**:
   - Primary criterion for predictive performance
   - Accounts for model complexity
   - Prefer model with lowest LOO (highest ELPD)
   - Check for Pareto-k warnings (influence diagnostics)

2. **WAIC (Widely Applicable Information Criterion)**:
   - Secondary criterion
   - Should be consistent with LOO
   - If LOO and WAIC disagree, examine why

3. **Posterior Predictive Checks**:
   - Can the model reproduce:
     - Observed variance in proportions?
     - Overdispersion factor φ ≈ 3.5?
     - Outlier trials (1, 8)?
     - Distribution of r_i?
   - Compute p-values for test statistics

4. **Prior-Posterior Overlap**:
   - How much does data update priors?
   - If minimal, data is weak (concerning)
   - If massive, prior may be misspecified

5. **Interpretability**:
   - Model 2 (mixture) is most interpretable if groups are real
   - Models 1 and 3 harder to interpret but more flexible
   - Scientific context matters

### 4.2 Decision Tree

```
START: Fit all three models
  ↓
CHECK: LOO-CV comparison
  ↓
├─ If Model 2 (mixture) wins by >4 ELPD:
│   ├─ Check: Are components well-separated? (p_1 vs p_2 95% CIs)
│   ├─ Check: Are assignments certain? (γ_ik > 0.7 for most trials)
│   ├─ If YES: Accept Model 2, interpret groups
│   └─ If NO: Despite LOO, model may be overfit → use Model 1 or 3
│
├─ If Model 1 (Beta-Binomial) wins:
│   ├─ Check: Posterior predictive adequate? (reproduce overdispersion)
│   ├─ Check: κ well-identified? (not at boundary)
│   ├─ If YES: Accept Model 1
│   └─ If NO: Switch to Model 3
│
├─ If Model 3 (Logit Hierarchical) wins:
│   ├─ Check: σ well-identified? (not at 0 or boundary)
│   ├─ Check: η_i approximately normal?
│   ├─ If YES: Accept Model 3
│   └─ If NO: Return to Model 1
│
└─ If models tie (ELPD within 2):
    ├─ Use MODEL AVERAGING (stack predictions)
    └─ Report all models as plausible
```

### 4.3 Stress Tests

**For ALL models, I will**:

1. **Remove trial 1** (0/47):
   - Does conclusion change?
   - Is this trial driving results?

2. **Remove trial 8** (31/215, highest proportion):
   - Symmetric test to trial 1
   - Check robustness to high outlier

3. **Remove trial 4** (810 observations):
   - Largest sample, may dominate
   - Check influence on pooled estimate

4. **Simulate new data**:
   - Generate r_new from posterior predictive
   - Refit models to r_new
   - Do parameter estimates match original posteriors?
   - If not: model misspecified or data is weak

5. **Prior sensitivity**:
   - For each model, try 3 prior specifications
   - If conclusions flip: data is weak, need more information
   - If robust: confidence in inference

### 4.4 Red Flags for ALL Models

**STOP AND RECONSIDER EVERYTHING IF**:

1. **All models fail posterior predictive checks**:
   - None can reproduce observed variance
   - Suggests data-generating process is more complex than assumed
   - **Action**: Consider covariates, non-binomial likelihood, or structured models

2. **LOO gives warnings for >25% of observations**:
   - High Pareto-k values indicate poor model fit
   - **Action**: Examine influential points, consider robust alternatives

3. **Posteriors are prior-dominated**:
   - Data too weak to update beliefs
   - With 12 observations, this is possible
   - **Action**: Report high uncertainty, collect more data

4. **Models give contradictory substantive conclusions**:
   - E.g., Model 1 says μ ≈ 0.05, Model 3 says μ ≈ 0.10
   - **Action**: Models may all be misspecified, need to rethink

5. **Computational failures persist**:
   - Divergences, low ESS, non-convergence across all models
   - **Action**: Fundamental issue with model class or data

---

## Alternative Models (If All Fail)

If Models 1-3 all fail falsification criteria, I will consider:

### Plan B: Beta-Binomial with Robust Likelihood

```
θ_i ~ Beta(μκ, (1-μ)κ)
r_i | θ_i ~ RobustBinomial(n_i, θ_i, ν)

where RobustBinomial uses Student-t approximation or mixture:
  r_i ~ (1-w)*Binomial(n_i, θ_i) + w*Uniform(0, n_i)
  w ~ Beta(1, 99)  # 1% contamination
```

**Rationale**: Allows for outliers that don't fit binomial model.

### Plan C: Overdispersed Binomial (Quasi-Likelihood)

```
E[r_i] = n_i * θ_i
Var[r_i] = φ * n_i * θ_i * (1 - θ_i)

where φ is estimated dispersion parameter
```

**Rationale**: Phenomenological model, doesn't assume specific heterogeneity structure.

### Plan D: Non-Parametric Dirichlet Process Mixture

```
G ~ DirichletProcess(α, G_0)
θ_i ~ G
r_i ~ Binomial(n_i, θ_i)

where G_0 = Beta(a, b) is base distribution
```

**Rationale**: Lets data determine number of components.

**WARNING**: Plan D is risky with 12 observations, likely to overfit.

---

## Implementation Timeline

### Phase 1: Initial Fitting (Day 1)

1. Implement all three models in Stan
2. Run 4 chains, 2000 iterations each (1000 warmup)
3. Check convergence: R̂ < 1.01, ESS > 400
4. Save posteriors for all parameters

### Phase 2: Diagnostics (Day 1-2)

1. Compute LOO-CV for each model
2. Run posterior predictive checks
3. Generate diagnostic plots:
   - Trace plots
   - Posterior distributions
   - Prior-posterior comparison
   - Residual plots

### Phase 3: Comparison (Day 2)

1. Compare LOO-CV scores
2. Check falsification criteria for each model
3. Identify winning model or plan model averaging

### Phase 4: Sensitivity (Day 2-3)

1. Refit with alternative priors
2. Refit excluding influential trials (1, 4, 8)
3. Check robustness of conclusions

### Phase 5: Reporting (Day 3)

1. Document final model choice (or averaging strategy)
2. Report parameter estimates with uncertainty
3. Visualize results
4. State limitations and caveats

---

## Critical Self-Assessment

### What Could Go Wrong?

1. **All models may be wrong**:
   - True data-generating process could be more complex
   - 12 observations may be too few to distinguish models
   - Overdispersion could be from unmeasured covariates

2. **Identifiability may fail**:
   - With 12 trials, difficult to estimate hyperparameters precisely
   - Models may fit data but with high posterior uncertainty
   - Prior choice may drive conclusions

3. **I may be overconfident**:
   - Bayesian models give precise posteriors even with weak data
   - Wide posteriors = "I don't know", not "here's the answer"
   - Must report uncertainty honestly

### Philosophical Stance

**I commit to**:

1. **Falsification over confirmation**:
   - I will actively try to break each model
   - Passing tests doesn't prove model is correct
   - Failing tests proves model is wrong (actionable)

2. **Uncertainty quantification**:
   - Wide posteriors are honest answers
   - Narrow posteriors may reflect prior, not data
   - Will report when data is too weak for strong conclusions

3. **Model averaging if needed**:
   - If models tie, I will average predictions
   - This is success (learning from data), not failure

4. **Knowing when to stop**:
   - If all models fail and alternatives are speculative
   - Better to say "need more data" than force a model
   - 12 observations may simply be insufficient

---

## Expected Outcomes

### Best Case Scenario

- One model clearly wins on LOO-CV (ΔELPD > 4)
- Passes all posterior predictive checks
- Parameter estimates stable across sensitivity analyses
- Scientific interpretation is clear
- **Action**: Report this model as best explanation

### Likely Scenario

- Models 1 and 3 perform similarly (ΔELPD < 2)
- Model 2 slightly worse (more parameters, overfitting risk)
- All models reproduce overdispersion but with wide uncertainty
- **Action**: Model averaging, report range of plausible parameters

### Worst Case Scenario

- All models fail posterior predictive checks
- LOO warnings for >3 observations
- Parameter estimates sensitive to priors
- Contradictory conclusions across models
- **Action**: Report failure, recommend more data or covariate collection

---

## Summary Table

| Model | Pros | Cons | Parameters | When to Use |
|-------|------|------|------------|-------------|
| **1: Beta-Binomial** | Standard, continuous variation, interpretable μ and κ | May have funnel geometry, κ hard to interpret | μ, κ, {θ_i} (14 total) | Default choice, natural overdispersion model |
| **2: Finite Mixture** | Interpretable groups, matches tercile analysis | May overfit, label switching, needs K choice | π, {p_k}, {z_i} (~6-9) | If groups are scientifically meaningful |
| **3: Logit Hierarchical** | Best HMC geometry, natural link, easy to extend | Less interpretable, normal assumption | μ_logit, σ, {η_i} (14 total) | If computational issues with Model 1 |

---

## Files to Create

1. `/workspace/experiments/designer_2/stan_models/model1_beta_binomial.stan`
2. `/workspace/experiments/designer_2/stan_models/model2_mixture.stan`
3. `/workspace/experiments/designer_2/stan_models/model3_logit_hierarchical.stan`
4. `/workspace/experiments/designer_2/pymc_models/model1_beta_binomial.py`
5. `/workspace/experiments/designer_2/pymc_models/model2_mixture.py`
6. `/workspace/experiments/designer_2/pymc_models/model3_logit_hierarchical.py`
7. `/workspace/experiments/designer_2/diagnostics/posterior_predictive_checks.py`
8. `/workspace/experiments/designer_2/diagnostics/model_comparison.py`

---

## Final Note

These are **PROPOSALS**, not final models. The true test is empirical performance. I expect:

- At least one model will partially fail
- Parameter uncertainty will be substantial (n=12 is small)
- Model averaging may be necessary
- Sensitivity analyses will reveal brittleness

**I will adapt as data demands, not force a predetermined model.**

**Success = finding truth, not completing a checklist.**

---

**End of Model Proposals**

Designer 2, ready for synthesis with other designers.
