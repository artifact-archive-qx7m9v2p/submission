# Robust and Alternative Bayesian Models for Binomial Data with Outliers
## Model Designer 3: Focus on Robustness and Alternative Perspectives

**Date:** 2025-10-30
**Dataset:** 12 groups, severe overdispersion (φ ≈ 3.5-5.1), 42% outlier rate
**Challenge:** Group 8 extreme outlier (z=3.94), Group 1 zero successes, wide heterogeneity

---

## Executive Summary

The EDA reveals an **outlier-dominated dataset** where 5 of 12 groups fall outside expected ranges. Standard hierarchical models (Normal or Beta priors) may be overly influenced by these outliers. I propose **three robust alternatives** that explicitly model outlier mechanisms:

1. **Student-t Hierarchical Model**: Heavy-tailed group effects (robust to outliers)
2. **Horseshoe Prior Model**: Sparse effects (most groups near mean, few truly deviate)
3. **Mixture Model**: Latent subgroups (normal vs outlier groups)

### Key Philosophy: Robustness vs Efficiency Trade-off

- **Standard Normal priors**: Efficient when data is clean, but vulnerable to outliers
- **Robust alternatives**: Protect inference when outliers are present, at cost of wider intervals
- **Critical question**: Are these "outliers" true signal or measurement artifacts?

---

## Model Comparison Table

| Model | Outlier Handling | Complexity | When to Use | When to Abandon |
|-------|------------------|------------|-------------|-----------------|
| **Student-t Hierarchical** | Heavy tails accommodate outliers without contaminating population mean | Medium (1 extra param: ν) | When outliers are plausible but you want robust population inference | If posterior ν → ∞ (data is actually Normal) |
| **Horseshoe Prior** | Sparse shrinkage: most groups shrink aggressively, outliers shrink less | Medium-High (12 local scales) | When most groups truly identical, few genuinely different | If all groups have large λ (no sparsity) |
| **Mixture Model** | Explicitly models two subpopulations (normal + outlier) | High (3 extra params) | When discrete clusters suspected | If mixing probability → 0 or 1 (no mixture) |

---

## Critical Design Philosophy: Planning for Failure

### What Makes These Models "Robust"?

1. **Outlier Resistance**: Population mean μ not overly influenced by Group 8
2. **Automatic Outlier Detection**: Models identify which groups are outliers via posterior
3. **Adaptive Shrinkage**: Outliers shrink less than under Normal priors
4. **Falsifiable**: Clear posterior signals when model is wrong

### Red Flags That Would Reject ALL Models

1. **Non-binomial data generation**: Posterior predictive checks show systematic misfit
2. **Temporal instability**: If we had time data showing rates changing within groups
3. **Extreme computational pathologies**: Divergences, non-convergence despite reparameterization
4. **Prior-data conflict**: Posterior concentrates at edge of prior support
5. **Negative results from all models**: If robust and non-robust models all fail validation

### Decision Points for Major Pivots

- **After Model 1 (Student-t)**: If ν < 5, consider discrete mixture instead
- **After Model 2 (Horseshoe)**: If λ_i all similar (no sparsity), abandon sparse modeling
- **After Model 3 (Mixture)**: If π ≈ 0.5, data may have fundamental bimodality requiring covariate explanation
- **After all three**: If none validate, consider:
  - Beta regression on rates (continuous outcome)
  - Negative binomial (different overdispersion mechanism)
  - Data quality issues requiring investigation

---

## Model 1: Student-t Hierarchical Model

### Conceptual Motivation

**Problem with Normal priors**: Normal distribution has exponentially decaying tails. Group 8 (z=3.94) has probability density ~10^-4 under Normal. This creates tension:
- Either population σ inflates to accommodate Group 8 (contaminating inference)
- Or Group 8 shrinks excessively toward mean (ignoring real signal)

**Student-t solution**: Polynomial (not exponential) tails. If Group 8 is a genuine outlier, it gets probability ~10^-3 (1000× more plausible), reducing shrinkage and contamination.

### Mathematical Specification

```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)                    for i = 1, ..., 12

Level 2 (Group effects with heavy tails):
  logit(p_i) = μ + α_i
  α_i ~ Student-t(ν, 0, σ)                    [Heavy-tailed random effects]

Level 3 (Hyperpriors):
  μ ~ Normal(-2.5, 1)                         [Population mean on logit scale]
                                               [logit^-1(-2.5) ≈ 0.076]
  σ ~ Half-Cauchy(0, 1)                       [Between-group SD]
  ν ~ Gamma(2, 0.1)                           [Degrees of freedom: ν > 1]
                                               [Mode at ν ≈ 10, heavy prior mass on ν < 30]

Implied marginal:
  logit(p_i) ~ Student-t(ν, μ, σ)
```

### Prior Justification

1. **μ ~ Normal(-2.5, 1)**:
   - Centers on observed pooled rate (7.6%)
   - SD of 1 on logit scale → 95% CI: [1%, 35%]
   - Weakly informative, avoids extreme rates

2. **σ ~ Half-Cauchy(0, 1)**:
   - Standard weakly informative prior for hierarchical SD
   - Heavy tail allows large between-group variation if data supports it
   - Median ≈ 0.45, 95% quantile ≈ 3.3

3. **ν ~ Gamma(2, 0.1)**:
   - Mode at 10, median at 14
   - Prior belief: some tail heaviness likely given EDA outliers
   - 95% interval: [2.4, 54]
   - If ν → ∞, recovers Normal model (data doesn't need robustness)
   - If ν < 5, very heavy tails (strong outlier presence)

### When This Model Succeeds vs Fails

**I will abandon this model if:**
1. Posterior ν > 50 with narrow CI → Data is actually Normal, robustness unnecessary
2. Posterior ν < 2 with divergences → Need discrete mixture instead
3. Posterior predictive checks fail (can't reproduce observed patterns)
4. LOO-CV shows worse predictive performance than Normal hierarchical
5. Prior-posterior conflict on ν (posterior at edge of prior support)

**Evidence for success:**
- Posterior ν ≈ 5-15 (moderate heavy tails)
- Group 8's α_8 has wider posterior than under Normal prior
- Population μ posterior narrower than under Normal (less contamination)
- No divergences, good sampling diagnostics

### Expected Posterior Behavior

- **If outliers are real signal**:
  - Posterior ν ≈ 5-10 (moderate heavy tails needed)
  - Group 8: E[α_8] ≈ +1.5, less shrinkage than Normal model
  - Group 1: E[α_1] ≈ -1.2, moderate shrinkage toward 1-2%
  - Population σ: E[σ] ≈ 0.8-1.0 (robust to outliers)

- **If outliers are artifacts**:
  - Posterior ν → large (data doesn't need heavy tails)
  - Model reduces to Normal hierarchical
  - BIC penalizes unnecessary ν parameter

### Stan Implementation

```stan
// student_t_hierarchical.stan
data {
  int<lower=1> N;                  // Number of groups
  array[N] int<lower=0> n_trials;  // Trials per group
  array[N] int<lower=0> r;         // Successes per group
}

parameters {
  real mu;                         // Population mean (logit scale)
  real<lower=0> sigma;             // Between-group SD
  real<lower=1> nu;                // Degrees of freedom (>1 for finite variance)
  vector[N] alpha_raw;             // Non-centered random effects
}

transformed parameters {
  vector[N] alpha;
  vector[N] logit_p;

  // Non-centered parameterization for efficiency
  alpha = sigma * alpha_raw;
  logit_p = mu + alpha;
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  sigma ~ cauchy(0, 1);
  nu ~ gamma(2, 0.1);

  // Student-t random effects (non-centered)
  alpha_raw ~ student_t(nu, 0, 1);

  // Likelihood
  r ~ binomial_logit(n_trials, logit_p);
}

generated quantities {
  // Posterior predictive checks
  array[N] int r_rep;
  vector[N] p = inv_logit(logit_p);
  real mean_p = mean(p);

  // Posterior predictive samples
  for (i in 1:N) {
    r_rep[i] = binomial_rng(n_trials[i], p[i]);
  }

  // Compute overdispersion parameter from posterior
  real var_p = variance(p);
  real expected_var_p = mean_p * (1 - mean_p) / mean(to_vector(n_trials));
  real phi_posterior = var_p / expected_var_p;

  // Flag outlier groups (|alpha| > 2)
  array[N] int is_outlier;
  for (i in 1:N) {
    is_outlier[i] = (fabs(alpha[i]) > 2.0) ? 1 : 0;
  }
}
```

### Computational Considerations

**Challenges:**
- Student-t can have geometry issues for small ν (< 3)
- Need non-centered parameterization for efficiency
- ν is often hard to estimate precisely (wide posterior)

**Sampling strategy:**
- Use 4 chains, 2000 iterations (1000 warmup)
- Target: Rhat < 1.01, ESS > 400
- If divergences occur: increase adapt_delta to 0.95
- If ν posterior truncated at 1: switch to mixture model

**Diagnostics:**
- Check ν posterior not at boundary (ν = 1 or ν → ∞)
- Pairs plot of (μ, σ, ν) for degeneracies
- Posterior predictive: does model reproduce outlier frequency?

### Comparison to Standard Hierarchical Normal

**When Student-t outperforms Normal:**
- True data has outliers not explained by covariates
- Want robust population mean (μ) estimates
- Group 8 shouldn't contaminate inference about typical groups

**Trade-offs:**
- Student-t gives wider credible intervals (conservative)
- One additional parameter (ν) to estimate
- Harder to interpret ν (Normal has no analogue)

---

## Model 2: Horseshoe Prior Model (Sparse Hierarchical Effects)

### Conceptual Motivation

**Alternative hypothesis**: Most groups are essentially identical (α_i ≈ 0), but a **sparse subset** truly differs. This is fundamentally different from continuous variation (Normal/Student-t).

**Horseshoe intuition**:
- Groups close to population mean → aggressive shrinkage (α_i → 0)
- True outliers → minimal shrinkage (retain large |α_i|)
- Automatic selection: data determines which groups are "special"

**When is this the right model?**
- If Groups 2, 8, 11 are genuinely different entities (different populations)
- If most groups (7 out of 12) are truly interchangeable
- If we believe sparse effects but don't know which groups a priori

### Mathematical Specification

```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)                    for i = 1, ..., 12

Level 2 (Group effects with horseshoe prior):
  logit(p_i) = μ + α_i
  α_i ~ Normal(0, τ · λ_i)                    [Local-global shrinkage]

  λ_i ~ Cauchy^+(0, 1)                        [Local shrinkage (per group)]
  τ ~ Cauchy^+(0, τ_0)                        [Global shrinkage]

  τ_0 = (p_0 / (N - p_0)) * (σ_y / sqrt(N))  [Expected sparsity]
        where p_0 ≈ 3 (expect ~3 outlier groups)
              σ_y ≈ 1 (typical effect size on logit scale)

Level 3 (Hyperprior):
  μ ~ Normal(-2.5, 1)                         [Population mean]
```

### Prior Justification

**Horseshoe prior philosophy**:
- The horseshoe is a **continuous spike-and-slab** prior
- Small effects get heavily shrunk to zero (λ_i → 0)
- Large effects are barely shrunk (λ_i → large)
- No need to specify number of outliers a priori

**Why better than Normal for sparse effects?**

Normal prior: E[α_i | data] ∝ (1 - shrinkage) × MLE[α_i]
- Shrinkage is **constant** across groups
- All groups shrink by same factor (~85.6% per EDA)

Horseshoe prior: E[α_i | data] involves adaptive shrinkage
- Groups near zero: shrinkage → 100%
- Groups far from zero: shrinkage → 0%
- **Each group gets custom shrinkage factor**

**τ_0 calculation** (prior expectation of sparsity):
- Based on EDA: expect ~3 of 12 groups to be outliers
- p_0 = 3, N = 12
- τ_0 = (3 / 9) * (1 / sqrt(12)) ≈ 0.096
- This is a **weakly informative** prior favoring sparsity

### When This Model Succeeds vs Fails

**I will abandon this model if:**
1. All λ_i similar (no sparsity) → continuous variation model better
2. Posterior τ >> τ_0 → many non-zero effects, violates sparsity assumption
3. Posterior α_i bimodal without clear separation → need mixture model
4. Computational issues: slow mixing on λ_i parameters
5. Worse LOO-CV than Normal hierarchical → sparsity assumption wrong

**Evidence for success:**
- λ_i clearly separates into "small" (≈0.1) and "large" (≈1.5) groups
- Posterior τ ≈ τ_0 (sparsity prior was well-calibrated)
- 3-5 groups have P(|α_i| > 0.5) > 0.95 (clearly non-zero)
- 7-9 groups have P(|α_i| < 0.1) > 0.95 (clearly zero)
- Better out-of-sample prediction than non-sparse models

### Expected Posterior Behavior

**If sparsity is real**:
- Group 8: λ_8 large (≈1.0-2.0), α_8 ≈ +1.5 (minimal shrinkage)
- Group 1: λ_1 moderate (≈0.3-0.5), α_1 ≈ -0.8
- Groups 2, 11: λ large, positive α
- Groups 3, 4, 5, 6, 7, 9, 12: λ small (< 0.2), α ≈ 0 (shrunk heavily)
- Global τ ≈ 0.1-0.2

**If sparsity is wrong**:
- All λ_i ≈ 0.5-1.0 (no separation)
- τ large (global scale inflates)
- Model reduces to standard hierarchical Normal

### Stan Implementation

```stan
// horseshoe_hierarchical.stan
data {
  int<lower=1> N;                  // Number of groups
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r;
  real<lower=0> tau0;              // Expected sparsity level
}

parameters {
  real mu;                         // Population mean
  vector[N] alpha_raw;             // Raw random effects

  // Horseshoe prior components
  vector<lower=0>[N] lambda;       // Local shrinkage (per group)
  real<lower=0> tau;               // Global shrinkage
}

transformed parameters {
  vector[N] alpha;
  vector[N] logit_p;

  // Horseshoe shrinkage
  alpha = alpha_raw .* lambda * tau;
  logit_p = mu + alpha;
}

model {
  // Hyperpriors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, tau0);           // Global shrinkage scale

  // Horseshoe prior
  lambda ~ cauchy(0, 1);           // Local shrinkage scales
  alpha_raw ~ normal(0, 1);        // Standardized effects

  // Likelihood
  r ~ binomial_logit(n_trials, logit_p);
}

generated quantities {
  // Posterior predictive checks
  array[N] int r_rep;
  vector[N] p = inv_logit(logit_p);

  // Effective shrinkage for each group
  vector[N] kappa;  // kappa_i ≈ 1 means no shrinkage, ≈ 0 means full shrinkage
  for (i in 1:N) {
    kappa[i] = lambda[i]^2 / (lambda[i]^2 + tau^2);
  }

  // Count number of "active" groups (large effects)
  int n_active = sum(to_array_1d(lambda) > 0.5);

  // Posterior predictive samples
  for (i in 1:N) {
    r_rep[i] = binomial_rng(n_trials[i], p[i]);
  }

  // Identify sparse groups (λ < 0.2 indicates heavy shrinkage)
  array[N] int is_sparse;
  for (i in 1:N) {
    is_sparse[i] = (lambda[i] < 0.2) ? 1 : 0;
  }
}
```

### Computational Considerations

**Challenges:**
- Cauchy priors can have heavy tails → slow mixing
- N λ_i parameters can be correlated with τ
- Horseshoe models often require more iterations

**Sampling strategy:**
- Use 4 chains, 3000 iterations (1500 warmup)
- Monitor λ_i effective sample sizes carefully
- If ESS(λ_i) < 100 for any i: increase iterations or use regularized horseshoe

**Alternative: Regularized Horseshoe**
If sampling is difficult, use regularized version:
```
λ_i ~ Cauchy^+(0, 1)
c^2 ~ Inverse-Gamma(ν/2, ν·s^2/2)  // slab regularization
λ_i_tilde = c * λ_i / sqrt(c^2 + λ_i^2)
```
This prevents λ_i → ∞ and improves sampling.

### Comparison to Standard Models

**Horseshoe vs Normal hierarchical**:
- Normal: all groups shrink by ~85%
- Horseshoe: some shrink >95%, others <50%
- Normal: symmetric treatment of all groups
- Horseshoe: automatic outlier identification

**When to prefer Horseshoe**:
- Strong prior belief that most groups are similar
- Want automatic selection of outlier groups
- Care about prediction (sparsity often improves out-of-sample fit)

**When to avoid Horseshoe**:
- All groups genuinely different (no sparsity)
- Need simple interpretation (Horseshoe harder to explain)
- Computational budget limited (slower sampling)

---

## Model 3: Finite Mixture Model (Latent Subgroups)

### Conceptual Motivation

**Radical alternative hypothesis**: The outliers aren't just tail events of a single distribution - they're **a different population entirely**.

**Mixture interpretation**:
- **Component 1 (Normal groups)**: μ_1 ≈ -2.8, σ_1 ≈ 0.3 (tight cluster)
  - Groups: 1, 3, 4, 5, 6, 7, 9, 12
- **Component 2 (Outlier groups)**: μ_2 ≈ -1.8, σ_2 ≈ 0.5 (high-rate cluster)
  - Groups: 2, 8, 11, possibly 10

**When is this the right model?**
- If there's an unmeasured binary covariate (e.g., treatment A vs B)
- If outliers represent fundamentally different process
- If continuous models fail to capture bimodality

### Mathematical Specification

```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)                    for i = 1, ..., 12

Level 2 (Mixture of group effects):
  logit(p_i) = μ_z[i] + α_i

  α_i | z_i ~ Normal(0, σ_z[i])               [Cluster-specific variance]
  z_i ~ Categorical(π)                        [Latent cluster membership]
                                               [z_i ∈ {1, 2}]

Level 3 (Cluster-specific parameters):
  μ_1 ~ Normal(-3.0, 0.5)                     [Low-rate cluster mean]
  μ_2 ~ Normal(-2.0, 0.5)                     [High-rate cluster mean]

  σ_1 ~ Half-Normal(0, 0.3)                   [Low-rate cluster SD]
  σ_2 ~ Half-Normal(0, 0.5)                   [High-rate cluster SD]

  π ~ Dirichlet([1, 1])                       [Mixing proportions]
      Equivalent to π ~ Beta(1, 1) = Uniform(0, 1)

Constraint: μ_1 < μ_2 (for identifiability)
```

### Prior Justification

**Why two components?**
- EDA suggests possible bimodality (5 outliers, 7 normal)
- More than 2 components unidentifiable with N=12
- If data doesn't need mixture, π will concentrate at 0 or 1

**Cluster mean priors**:
- μ_1 ~ Normal(-3.0, 0.5): centers on ~5% success rate (lower cluster)
- μ_2 ~ Normal(-2.0, 0.5): centers on ~12% success rate (upper cluster)
- Separation: E[μ_2 - μ_1] = 1.0 on logit scale (substantial difference)

**Cluster variance priors**:
- σ_1 ~ Half-Normal(0, 0.3): expect less variation within normal cluster
- σ_2 ~ Half-Normal(0, 0.5): allow more variation within outlier cluster
- If data doesn't support this, posteriors will overlap

**Mixing proportion prior**:
- π ~ Uniform(0, 1): completely agnostic about cluster sizes
- Posterior π will be data-driven
- If π → 0 or 1, mixture is unnecessary

### When This Model Succeeds vs Fails

**I will abandon this model if:**
1. Posterior π ≈ 0 or π ≈ 1 → no mixture needed, use single component
2. Posterior μ_1 ≈ μ_2 → clusters not separated, use Normal hierarchical
3. Label switching in MCMC (chains swap cluster labels) → identifiability issues
4. Posterior z_i probabilities all ≈ 0.5 → can't assign groups to clusters
5. Worse LOO-CV than simpler models → mixture overfits

**Evidence for success:**
- Posterior π ≈ 0.3-0.7 (meaningful mixture)
- Clear separation: P(μ_2 > μ_1 + 0.5) > 0.95
- Group assignments clear: max(P(z_i = k)) > 0.8 for all i
- Groups 2, 8, 11 assigned to cluster 2 with high probability
- Groups 1, 3, 4, 5, 6, 7, 9, 12 assigned to cluster 1
- Better posterior predictive fit than single-component models

### Expected Posterior Behavior

**If mixture is real**:
- Cluster 1 (normal): μ_1 ≈ -2.8 (6.5% rate), σ_1 ≈ 0.2
  - Members: 8-9 groups with low/moderate rates
- Cluster 2 (outlier): μ_2 ≈ -1.9 (13% rate), σ_2 ≈ 0.3
  - Members: 3-4 groups with high rates (Groups 2, 8, 11)
- Mixing proportion: π ≈ 0.7 (70% normal, 30% outlier)
- Group 1 (zero successes): likely in cluster 1 but extreme within-cluster

**If mixture is wrong**:
- Label switching (chains can't agree on cluster identities)
- Posterior π very uncertain (95% CI covers [0.1, 0.9])
- μ_1 and μ_2 posteriors heavily overlapping
- Group assignments ambiguous (probabilities ≈ 0.5)

### Stan Implementation

```stan
// mixture_hierarchical.stan
data {
  int<lower=1> N;                  // Number of groups
  int<lower=2> K;                  // Number of mixture components (K=2)
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r;
}

parameters {
  // Cluster-specific parameters
  ordered[K] mu;                   // Ordered for identifiability (mu[1] < mu[2])
  vector<lower=0>[K] sigma;        // Cluster-specific SDs

  // Mixing proportion
  simplex[K] pi;                   // Sums to 1

  // Group-specific parameters
  vector[N] alpha_raw;             // Non-centered random effects
}

transformed parameters {
  vector[N] logit_p;

  // For each group, compute mixture density
  // This is computed in model block for efficiency
}

model {
  // Priors on cluster parameters
  mu[1] ~ normal(-3.0, 0.5);       // Low-rate cluster
  mu[2] ~ normal(-2.0, 0.5);       // High-rate cluster

  sigma[1] ~ normal(0, 0.3);       // Tight cluster
  sigma[2] ~ normal(0, 0.5);       // Looser cluster

  pi ~ dirichlet(rep_vector(1.0, K));  // Uniform mixing

  // Marginalize over latent cluster assignments
  for (i in 1:N) {
    vector[K] log_pi_k;

    for (k in 1:K) {
      // Log probability of group i in cluster k
      log_pi_k[k] = log(pi[k]) +
                    normal_lpdf(alpha_raw[i] | 0, 1) +
                    binomial_logit_lpmf(r[i] | n_trials[i],
                                        mu[k] + sigma[k] * alpha_raw[i]);
    }

    // Marginalize over clusters (log-sum-exp trick)
    target += log_sum_exp(log_pi_k);
  }
}

generated quantities {
  // Posterior predictive checks
  array[N] int r_rep;
  vector[N] p;

  // Posterior cluster assignments
  array[N] simplex[K] prob_cluster;  // P(z_i = k | data)
  array[N] int<lower=1, upper=K> cluster_assignment;

  for (i in 1:N) {
    vector[K] log_pi_k;

    // Compute posterior cluster probabilities
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) +
                    normal_lpdf(alpha_raw[i] | 0, 1) +
                    binomial_logit_lpmf(r[i] | n_trials[i],
                                        mu[k] + sigma[k] * alpha_raw[i]);
    }

    prob_cluster[i] = softmax(log_pi_k);

    // Hard assignment (most probable cluster)
    cluster_assignment[i] = (prob_cluster[i][2] > 0.5) ? 2 : 1;

    // Posterior predictive: sample from mixture
    int z_sim = categorical_rng(pi);
    real logit_p_sim = mu[z_sim] + sigma[z_sim] * alpha_raw[i];
    p[i] = inv_logit(logit_p_sim);
    r_rep[i] = binomial_rng(n_trials[i], p[i]);
  }

  // Cluster size counts
  int n_cluster1 = sum(cluster_assignment == 1);
  int n_cluster2 = sum(cluster_assignment == 2);

  // Cluster separation metric
  real cluster_separation = mu[2] - mu[1];
}
```

### Computational Considerations

**Challenges:**
- **Label switching**: Chains may swap cluster labels (cluster 1 ↔ 2)
- **Local modes**: Mixture models often have multiple posterior modes
- **Slow mixing**: Cluster assignments can be sticky

**Solutions:**
1. **Ordered constraint**: Force μ[1] < μ[2] for identifiability
2. **Informative initialization**: Start chains with plausible cluster assignments
3. **Longer warmup**: Use 2000 warmup iterations
4. **Post-processing**: Check for label switching across chains

**Sampling strategy:**
- 4 chains, 4000 iterations (2000 warmup)
- init: Initialize μ[1] = -3, μ[2] = -2
- If Rhat > 1.05 on μ or π: likely label switching, need post-processing

**Diagnostics:**
- Trace plots of μ[1] and μ[2]: should not cross
- Pairs plot of (μ[1], μ[2], π): check for label switching
- Cluster assignments: should be stable across iterations

### Comparison to Continuous Models

**Mixture vs Normal/Student-t hierarchical**:
- Normal/Student-t: **Continuous distribution of effects**
  - All groups on spectrum
  - Outliers are tail events
- Mixture: **Discrete subpopulations**
  - Groups belong to latent clusters
  - Outliers are different category

**When to prefer Mixture**:
- Strong theoretical reason to expect subgroups
- Continuous models fail posterior predictive checks
- Want to identify which groups are "different"
- Bimodal distribution of observed rates

**When to avoid Mixture**:
- N too small (< 20) for reliable cluster identification
- No theoretical justification for discrete groups
- Computational difficulties (label switching)
- Simpler models fit equally well

---

## Model Selection Strategy

### Falsification Criteria Summary

Each model has **clear rejection criteria** built in:

| Model | Reject if... | Posterior Signal |
|-------|-------------|------------------|
| Student-t | ν > 50 | Data doesn't need heavy tails |
| Student-t | ν < 2 with divergences | Tails too heavy, need mixture |
| Horseshoe | All λ_i ≈ 0.5-1.0 | No sparsity detected |
| Horseshoe | τ >> τ_0 | Too many non-zero effects |
| Mixture | π → 0 or π → 1 | No mixture needed |
| Mixture | μ_1 ≈ μ_2 | Clusters not separated |

### Sequential Model Building Plan

**Phase 1: Fit all three models**
1. Student-t hierarchical (baseline robust model)
2. Horseshoe prior (sparse alternative)
3. Mixture model (discrete subgroups)

**Phase 2: Posterior diagnostics**
- Check sampling diagnostics (Rhat, ESS, divergences)
- Examine key parameters (ν, λ_i, π)
- Apply falsification criteria

**Phase 3: Model comparison**
- LOO-CV (leave-one-out cross-validation)
- PSIS-LOO Pareto k diagnostics (identify influential points)
- Posterior predictive checks
- AIC/BIC for parsimony

**Phase 4: Sensitivity analysis**
- Refit best model excluding Group 8 (extreme outlier)
- Refit with different priors (prior sensitivity)
- Check posterior-prior overlap (prior informativeness)

### Decision Tree

```
START
  |
  |--> Fit Student-t hierarchical
        |
        |--> If ν > 50: Use Normal hierarchical instead (no robustness needed)
        |--> If ν ∈ [5, 30]: Student-t is appropriate
        |--> If ν < 5: Consider mixture model
  |
  |--> Fit Horseshoe
        |
        |--> If 3-5 groups have λ > 0.5: Sparsity detected
        |--> If all λ ≈ 0.3-0.7: No sparsity, use Student-t
  |
  |--> Fit Mixture
        |
        |--> If π ∈ [0.2, 0.8] AND μ_2 - μ_1 > 0.5: Mixture justified
        |--> If π → 0 or 1: No mixture, use Student-t
  |
  |--> Compare LOO-CV
        |
        |--> Best LOO: Primary model
        |--> ΔLOO < 2: Models equivalent (choose simplest)
        |--> ΔLOO > 10: Clear winner
  |
  |--> Posterior predictive checks
        |
        |--> Can model reproduce:
              - Overdispersion (φ ≈ 3.5)?
              - Outlier frequency (5/12)?
              - Zero counts (Group 1)?
        |--> If NO: All models fail, reconsider data generation process
  |
END: Select best model OR declare all models inadequate
```

---

## Expected Outcomes and Interpretations

### Scenario 1: Student-t Wins

**Interpretation**: Outliers are real but continuous
- Group 8 is extreme but plausible under heavy-tailed distribution
- No evidence for discrete subgroups
- Robustness needed but not sparsity

**Implications**:
- Population mean estimates robust to outliers
- Group 8 shrinks less than under Normal prior
- Continuous spectrum of group effects

### Scenario 2: Horseshoe Wins

**Interpretation**: Sparse effects (most groups similar, few different)
- 8-9 groups effectively identical (α ≈ 0)
- 3-4 groups genuinely different (Groups 2, 8, 11)
- Sparsity improves prediction

**Implications**:
- Most groups can be pooled together
- Outliers are not errors, but truly different
- Resource allocation: focus on outlier groups

### Scenario 3: Mixture Wins

**Interpretation**: Discrete subpopulations
- Two distinct populations in the data
- Suggests unmeasured binary covariate
- Groups 2, 8, 11 fundamentally different from others

**Implications**:
- Need to investigate what distinguishes clusters
- Different interventions for different clusters
- Combining clusters would bias inference

### Scenario 4: All Models Fail

**What would cause this?**
- Posterior predictive checks systematically fail
- Large LOO Pareto k values (> 0.7) for multiple points
- Residual patterns not captured by any model

**Next steps**:
- Check for temporal trends (if data has time structure)
- Consider negative binomial (different overdispersion mechanism)
- Investigate data quality (measurement error, data entry issues)
- Look for covariates (group-level predictors)

---

## Stress Tests and Validation

### Built-in Stress Tests

**Test 1: Exclude Group 8** (extreme outlier)
- Refit best model without Group 8
- Compare population parameters (μ, σ) to full-data fit
- **Pass criterion**: Robust model should have similar μ, non-robust model's μ should shift

**Test 2: Simulate from fitted model**
- Generate 100 fake datasets from posterior predictive
- Refit model to each fake dataset
- **Pass criterion**: Can recover known parameters (simulation-based calibration)

**Test 3: Prior-posterior overlap**
- Compute KL divergence between prior and posterior
- **Pass criterion**: Some parameters should be data-informed (low overlap)
- **Fail criterion**: Posterior = prior (model not learning)

**Test 4: Cross-validation**
- Leave each group out, predict its success rate
- **Pass criterion**: 90% of groups within 50% predictive interval
- **Fail criterion**: Systematic overprediction or underprediction

### Posterior Predictive Checks

**Key quantities to reproduce**:
1. **Overdispersion**: φ_observed ≈ 3.5-5.1
   - P(φ_rep ∈ [3.0, 6.0]) > 0.5
2. **Outlier frequency**: 5 of 12 groups outside 95% limits
   - P(n_outliers_rep ∈ [3, 7]) > 0.5
3. **Zero count frequency**: 1 group with zero successes
   - P(n_zeros_rep ≥ 1) > 0.1
4. **Range of rates**: 0% to 14.4%
   - P(max(rate_rep) > 0.12) > 0.5

If model fails any of these, it's not capturing key data features.

---

## Implementation Notes and Warnings

### Stan vs PyMC

**Recommendation**: Use Stan (via CmdStanPy)
- Student-t: Both Stan and PyMC handle well
- Horseshoe: Stan has better samplers for heavy-tailed priors
- Mixture: Stan's log_sum_exp more numerically stable

**PyMC alternative**: If using PyMC, be aware:
- Horseshoe requires custom implementation or use `pm.AsymmetricLaplace` approximation
- Mixture models need `pm.Mixture` class (syntactically simpler than Stan)
- Student-t is `pm.StudentT` (straightforward)

### Common Pitfalls

**Pitfall 1: Non-centered parameterization**
- **Problem**: Centered parameterization (α ~ Student-t(ν, μ, σ)) can have funnel geometry
- **Solution**: Use α = μ + σ × α_raw, where α_raw ~ Student-t(ν, 0, 1)

**Pitfall 2: Horseshoe sampling**
- **Problem**: Cauchy tails can cause slow mixing
- **Solution**: Use regularized horseshoe or increase iterations

**Pitfall 3: Mixture label switching**
- **Problem**: Chains swap cluster labels, Rhat falsely high
- **Solution**: Use ordered constraint or post-process to align labels

**Pitfall 4: Overconfidence in outlier detection**
- **Problem**: With N=12, any one group is 8.3% of data
- **Solution**: Don't over-interpret individual group classifications

### Computational Budget

**Time estimates** (4 chains, adequate iterations):
- Student-t: ~2-5 minutes
- Horseshoe: ~5-10 minutes
- Mixture: ~10-20 minutes (marginalizing is expensive)

**If time is limited**:
- Start with Student-t (fastest, most likely to work)
- Only fit Horseshoe/Mixture if Student-t fails validation

---

## Critical Reflection: What Could Go Wrong?

### Meta-Failure Modes

**Failure Mode 1: All three models agree too much**
- **Signal**: Posterior predictions nearly identical
- **Interpretation**: Robustness features are unnecessary, data fits standard Normal model
- **Action**: Report that standard hierarchical model sufficient

**Failure Mode 2: All three models disagree wildly**
- **Signal**: Different μ estimates, different outlier classifications
- **Interpretation**: Data insufficient to distinguish hypotheses
- **Action**: Collect more groups or acknowledge fundamental uncertainty

**Failure Mode 3: Best model by LOO has worst interpretability**
- **Signal**: Mixture wins LOO-CV but π ≈ 0.5 (no clear clusters)
- **Interpretation**: Overfitting vs interpretability trade-off
- **Action**: Use simpler model, report sensitivity

### Philosophical Concerns

**Are these outliers or errors?**
- We assume outliers are real signal, but Group 8 (z=3.94) is suspicious
- Robust models reduce impact of potential errors
- But they can't fix data quality issues

**Is 12 groups enough?**
- Mixture models typically need N > 20 for stable cluster identification
- Horseshoe needs large N to distinguish sparse from dense
- With N=12, be cautious about strong claims

**Are we overthinking this?**
- Maybe beta-binomial (from standard EDA) is sufficient
- Robust models add complexity for modest gains
- Occam's razor: prefer simpler model unless clear evidence otherwise

---

## Summary and Recommendations

### Model Portfolio

I propose **three complementary robust models**:

1. **Student-t Hierarchical**: Continuous heavy-tailed effects
   - **Use when**: Outliers are plausible extreme values
   - **Advantage**: Robustness without assuming sparsity or clusters
   - **Risk**: If ν → ∞, complexity wasted

2. **Horseshoe Prior**: Sparse effects with automatic selection
   - **Use when**: Most groups similar, few truly different
   - **Advantage**: Better prediction via sparsity
   - **Risk**: If no sparsity, harder to interpret than Student-t

3. **Mixture Model**: Discrete latent subgroups
   - **Use when**: Evidence for bimodality or theoretical reason for clusters
   - **Advantage**: Identifies which groups are fundamentally different
   - **Risk**: Label switching, local modes, requires N > 20 ideally

### Final Recommendation

**Primary model**: Student-t hierarchical
- Most robust to outliers
- Simplest of the three
- Good balance of flexibility and interpretability

**Sensitivity analysis**: Horseshoe
- If Student-t validation is marginal
- If we care about prediction more than inference
- If computational budget allows

**Exploratory only**: Mixture
- N=12 is small for reliable clustering
- Only if Student-t and Horseshoe both fail
- High risk of overinterpretation

### Success Criteria

This modeling effort succeeds if:
1. At least one model passes all validation checks
2. We can quantify uncertainty in outlier classification
3. We can robustly estimate population mean despite outliers
4. We document what evidence would falsify our conclusions

This effort fails if:
1. All models fail posterior predictive checks
2. Computational difficulties prevent inference
3. Models are too complex to interpret
4. We can't distinguish between competing hypotheses

**Remember**: Finding that robust models are unnecessary (i.e., Normal hierarchical is sufficient) is a **success**, not a failure. It means the data is cleaner than suspected.

---

## File Locations

All model implementations saved to:
- `/workspace/experiments/designer_3/student_t_hierarchical.stan`
- `/workspace/experiments/designer_3/horseshoe_hierarchical.stan`
- `/workspace/experiments/designer_3/mixture_hierarchical.stan`

This document: `/workspace/experiments/designer_3/proposed_models.md`

---

**End of Model Design Document**

*Remember: The goal is truth, not task completion. Be ready to reject all models if evidence demands it.*
