# Alternative Bayesian Model Specifications
## Model Designer 2: Mixture Models and Robust Approaches

**Date:** 2025-10-30
**Designer:** Model Designer 2 (Alternative Approaches)
**Data:** 12 groups, binomial trials with strong heterogeneity

---

## Executive Summary

Based on the EDA findings (ICC = 0.42, 3 distinct clusters, 2 extreme outliers), I propose three alternative Bayesian model classes that differ fundamentally from standard hierarchical models:

1. **Finite Mixture Model (K=3):** Explicitly models the observed cluster structure
2. **Robust Beta-Binomial with Student-t:** Handles outliers via heavy-tailed distributions
3. **Dirichlet Process Mixture:** Non-parametric clustering without pre-specifying K

**Critical insight:** The EDA reveals three distinct subpopulations (mean rates: 3.1%, 6.5%, 13.2%). Standard hierarchical models assume a unimodal distribution of group effects. These alternatives explicitly model multimodality and test whether cluster structure is real or artifact.

---

## Model 1: Finite Mixture Model (K=3 Components)

### 1.1 Model Name & Description

**Finite Gaussian Mixture on Logit Scale (FMM-3)**

This model explicitly represents the three clusters identified in EDA as distinct mixture components. Each group belongs probabilistically to one of K=3 latent subpopulations, each with its own success rate distribution.

**How it differs from standard hierarchy:**
- Standard hierarchy: Single normal distribution for all group effects
- This model: Mixture of K normal distributions (multimodal population)
- Allows for discrete heterogeneity (qualitatively different group types)

### 1.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)                    # observed successes
logit(p_j) = theta_j                         # log-odds transformation
theta_j ~ sum_{k=1}^K pi_k * Normal(mu_k, tau_k)  # mixture of normals
```

**Priors:**
```
# Mixture weights (probabilities of cluster membership)
pi ~ Dirichlet(alpha)                        # alpha = [2, 2, 2] (weakly informative)

# Cluster-specific parameters
mu_k ~ Normal(-2.6, 1.5)                     # cluster means (logit scale)
tau_k ~ Normal(0, 0.5)                       # cluster SDs (half-normal)

# Constraints for identifiability
ordered[K] mu_k                              # enforce mu_1 < mu_2 < mu_3
```

**Prior Justification:**

- **Dirichlet(2,2,2):** Weakly favors equal cluster sizes but allows imbalance (observed: 1, 8, 3 groups)
- **mu_k ~ N(-2.6, 1.5):** Centers on observed pooled rate (0.07 = logit^-1(-2.6)), allows wide separation between clusters
- **tau_k ~ N+(0, 0.5):** Allows within-cluster variation, less restrictive than between-cluster variation
- **Ordered constraint:** Necessary for identifiability (prevents label switching)

**Full Generative Model:**
```
For each group j = 1, ..., 12:
  1. Draw cluster assignment z_j ~ Categorical(pi)
  2. Draw group effect theta_j ~ Normal(mu_{z_j}, tau_{z_j})
  3. Compute success probability p_j = logit^-1(theta_j)
  4. Generate data r_j ~ Binomial(n_j, p_j)
```

### 1.3 Stan Implementation Sketch

```stan
data {
  int<lower=1> J;              // number of groups (12)
  int<lower=1> K;              // number of mixture components (3)
  array[J] int<lower=0> n;     // trials per group
  array[J] int<lower=0> r;     // successes per group
}

parameters {
  simplex[K] pi;               // mixture weights
  ordered[K] mu;               // cluster means (ordered for identifiability)
  vector<lower=0>[K] tau;      // cluster SDs
  vector[J] theta;             // group-level logit success rates
}

model {
  // Priors
  pi ~ dirichlet(rep_vector(2.0, K));
  mu ~ normal(-2.6, 1.5);
  tau ~ normal(0, 0.5);

  // Mixture likelihood for group effects
  for (j in 1:J) {
    vector[K] log_pi_k = log(pi);
    for (k in 1:K) {
      log_pi_k[k] += normal_lpdf(theta[j] | mu[k], tau[k]);
    }
    target += log_sum_exp(log_pi_k);
  }

  // Data likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior cluster probabilities
  matrix[J, K] cluster_prob;

  // Posterior predictive
  array[J] int r_rep;

  for (j in 1:J) {
    // Log-likelihood
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);

    // Cluster assignment probabilities
    vector[K] log_pi_k;
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) + normal_lpdf(theta[j] | mu[k], tau[k]);
    }
    cluster_prob[j] = softmax(log_pi_k)';

    // Posterior predictive
    r_rep[j] = binomial_rng(n[j], inv_logit(theta[j]));
  }
}
```

**Data Input Specification (Python/PyMC):**
```python
import pymc as pm
import numpy as np

with pm.Model() as fmm_model:
    # Data
    n_trials = pm.ConstantData('n', observed_n)
    r_success = pm.ConstantData('r', observed_r)

    # Priors
    pi = pm.Dirichlet('pi', a=np.array([2., 2., 2.]))  # mixture weights

    # Cluster parameters (ordered)
    mu_ordered = pm.Normal('mu_ordered', mu=-2.6, sigma=1.5, shape=3,
                           transform=pm.distributions.transforms.Ordered())
    tau = pm.HalfNormal('tau', sigma=0.5, shape=3)

    # Mixture model for group effects
    theta = pm.NormalMixture('theta', w=pi, mu=mu_ordered, sigma=tau, shape=12)

    # Likelihood
    r = pm.Binomial('r', n=n_trials, logit_p=theta, observed=r_success)
```

### 1.4 Theoretical Justification

**Why this alternative approach is valuable:**

1. **Directly tests cluster hypothesis:** EDA found 3 clusters; this model explicitly represents them
2. **Discrete heterogeneity:** Some phenomena have qualitatively different group types (e.g., high-performing vs struggling centers)
3. **More interpretable:** Cluster membership probabilities are actionable (identify which groups need intervention)
4. **Better predictive performance:** If clusters are real, mixture model will predict new groups better

**What patterns it captures that hierarchical models miss:**

- **Multimodality:** Standard hierarchy assumes unimodal distribution of group effects; this allows multiple modes
- **Sparse regions:** Gaps between clusters (no groups with rates 9-11%)
- **Within-cluster homogeneity:** Groups within same cluster may be more similar than hierarchy assumes
- **Cluster-specific shrinkage:** Small groups shrink toward their cluster mean, not global mean

**Link to EDA findings:**

| EDA Finding | How FMM Captures It |
|-------------|---------------------|
| 3 distinct clusters (K=3 optimal) | K=3 mixture components |
| Cluster means: 3.1%, 6.5%, 13.2% | mu_1 < mu_2 < mu_3 (ordered) |
| Cluster sizes: 1, 8, 3 groups | pi ~ Dirichlet (inferred from data) |
| No overlap in 95% CIs | Large separation in mu_k |
| Groups 4, 8 extreme outliers | Belong to extreme clusters (k=1 or k=3) |

### 1.5 Falsification Criteria

**I will abandon this model if:**

1. **Cluster assignments are ambiguous:**
   - Most groups have cluster probabilities near 1/3 (no clear assignment)
   - Posterior entropy of cluster assignments > 0.8 (on scale 0-1)
   - **Diagnostic:** `mean(max(cluster_prob, axis=1)) < 0.6`

2. **Clusters collapse:**
   - Posterior intervals for mu_k overlap substantially (mu_1 ≈ mu_2 ≈ mu_3)
   - Effective number of clusters < 2 (calculated via entropy)
   - **Diagnostic:** `abs(mu[k+1] - mu[k]) < 2 * sqrt(tau[k]^2 + tau[k+1]^2)` for any k

3. **Worse predictive performance than hierarchy:**
   - LOO-ELPD significantly worse than standard hierarchical model (delta > 5)
   - Higher RMSE on leave-one-out predictions
   - **Diagnostic:** `loo_compare(fmm, hierarchy)` shows FMM worse by > 1 SE

4. **Computational failure:**
   - MCMC fails to converge (Rhat > 1.01) after 4000 iterations
   - Excessive divergences (> 1% of samples)
   - Label switching detected in trace plots (mu_k swap positions)

5. **Overfitting evidence:**
   - WAIC/LOO heavily penalizes model complexity (p_eff > 15)
   - Posterior predictive checks show overdispersion (data outside 95% envelope)

**Expected differences from standard hierarchy:**

| Metric | Standard Hierarchy | FMM-3 | Test |
|--------|-------------------|-------|------|
| Shrinkage | Toward global mean | Toward cluster mean | Group 10 (very low) shrinks less in FMM |
| Predictive variance | Unimodal | Multimodal | Posterior predictive for new group bimodal |
| Group 8 (outlier) | Partial shrinkage | Minimal shrinkage (cluster 3) | theta_8 closer to raw rate in FMM |
| Cluster membership | N/A | High certainty (>0.8) | Most groups clearly assigned |

**Specific predictions:**
- Group 10 (very low, 3.1%) should have >80% probability in cluster 1
- Groups 1, 2, 8 (high rates) should have >70% probability in cluster 3
- Groups in cluster 2 (majority) should show less shrinkage than in standard hierarchy

### 1.6 Computational Considerations

**Expected challenges:**

1. **Label switching:** Mixture models suffer from permutation invariance
   - **Solution:** Ordered constraint on mu_k (implemented above)
   - **Monitoring:** Check trace plots for consistent ordering

2. **Local modes:** MCMC may get stuck in suboptimal configurations
   - **Solution:** Multiple chains (4+) with dispersed initializations
   - **Solution:** Longer warmup (2000 iterations)

3. **Slower convergence:** More parameters than standard hierarchy
   - Standard: 2 hyperparameters (mu, tau) + 12 group effects = 14 params
   - FMM-3: 2 mixture weights + 6 cluster params + 12 group effects = 20 params
   - **Expect:** 2x longer sampling time

4. **Sensitivity to K:** Results depend on number of components
   - **Mitigation:** Compare K=2, K=3, K=4 models via LOO-CV
   - **Robustness check:** Dirichlet process model (doesn't fix K)

**Comparison to hierarchical model complexity:**

| Aspect | Hierarchical | FMM-3 | Ratio |
|--------|-------------|-------|-------|
| Parameters | 14 | 20 | 1.43x |
| Effective parameters (p_eff) | ~8 | ~12 | 1.5x |
| Sampling time (est.) | 10s | 20s | 2x |
| Convergence difficulty | Easy | Moderate | - |
| Risk of divergences | Low (<0.1%) | Moderate (1-2%) | 10-20x |

**Overall assessment:** More complex but feasible. Main risk is label switching (mitigated by ordered constraint). Worth the cost if cluster structure is real.

---

## Model 2: Robust Beta-Binomial with Student-t Hierarchy

### 2.1 Model Name & Description

**Student-t Hierarchical Beta-Binomial (Robust-HBB)**

This model uses a beta-binomial conjugate structure but replaces the normal distribution for group effects with a Student-t distribution. Heavy tails allow extreme outliers (Groups 4, 8) to exert less influence on hyperparameter estimates.

**How it differs from standard hierarchy:**
- Standard hierarchy: Normal(mu, tau) distribution for group effects
- This model: Student-t(nu, mu, tau) distribution (heavier tails, robust to outliers)
- Beta-binomial conjugacy simplifies computation
- Infers degrees of freedom (nu) from data

### 2.2 Mathematical Specification

**Likelihood:**
```
r_j ~ BetaBinomial(n_j, alpha_j, beta_j)     # overdispersed binomial
p_j = alpha_j / (alpha_j + beta_j)           # mean success rate
kappa_j = alpha_j + beta_j                   # concentration parameter
```

**Reparameterization (mean-concentration):**
```
p_j ~ Student-t(nu, mu_p, tau_p)             # robust hierarchy on success rates
kappa ~ Gamma(a_kappa, b_kappa)              # overdispersion parameter
alpha_j = p_j * kappa
beta_j = (1 - p_j) * kappa
```

**Priors:**
```
# Population-level parameters
mu_p ~ Beta(2, 28)                           # prior mean success rate ≈ 0.07
tau_p ~ HalfCauchy(0, 0.05)                  # between-group SD
nu ~ Gamma(2, 0.1)                           # degrees of freedom (heavy tails)

# Overdispersion parameter
kappa ~ Gamma(2, 0.1)                        # concentration (higher = less overdispersion)

# Constraint
nu >= 1                                      # ensure finite variance
```

**Prior Justification:**

- **mu_p ~ Beta(2, 28):** Centers on 0.07 (pooled rate), allows range [0.01, 0.20]
- **tau_p ~ HalfCauchy(0.05):** Heavy-tailed prior for scale (allows large between-group variation)
- **nu ~ Gamma(2, 0.1):** Allows nu = 1 (Cauchy, very heavy) to nu > 30 (nearly normal); nu ≈ 4-10 expected
- **kappa ~ Gamma(2, 0.1):** Overdispersion; larger kappa = less binomial noise

**Full Generative Model:**
```
For each group j = 1, ..., 12:
  1. Draw success rate p_j ~ Student-t(nu, mu_p, tau_p)  [truncated to (0,1)]
  2. Compute alpha_j = p_j * kappa, beta_j = (1 - p_j) * kappa
  3. Generate data r_j ~ BetaBinomial(n_j, alpha_j, beta_j)
```

### 2.3 Stan Implementation Sketch

```stan
data {
  int<lower=1> J;              // number of groups
  array[J] int<lower=0> n;     // trials per group
  array[J] int<lower=0> r;     // successes per group
}

parameters {
  real<lower=0, upper=1> mu_p;        // population mean success rate
  real<lower=0> tau_p;                // population SD
  real<lower=1> nu;                   // degrees of freedom
  real<lower=0> kappa;                // concentration parameter
  vector<lower=0, upper=1>[J] p;      // group-level success rates
}

transformed parameters {
  vector<lower=0>[J] alpha;
  vector<lower=0>[J] beta;

  for (j in 1:J) {
    alpha[j] = p[j] * kappa;
    beta[j] = (1 - p[j]) * kappa;
  }
}

model {
  // Priors
  mu_p ~ beta(2, 28);
  tau_p ~ cauchy(0, 0.05);
  nu ~ gamma(2, 0.1);
  kappa ~ gamma(2, 0.1);

  // Hierarchical model with Student-t
  p ~ student_t(nu, mu_p, tau_p);  // Stan auto-truncates to (0,1) given constraints

  // Likelihood
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha[j], beta[j]);
  }
}

generated quantities {
  vector[J] log_lik;
  array[J] int r_rep;
  real nu_effective;  // effective df after truncation

  for (j in 1:J) {
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha[j], beta[j]);
    r_rep[j] = beta_binomial_rng(n[j], alpha[j], beta[j]);
  }

  // Diagnose heaviness of tails
  nu_effective = nu;  // if nu < 5, heavy tails; nu > 30, nearly normal
}
```

**PyMC Implementation:**
```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as robust_hbb:
    # Data
    n_trials = pm.ConstantData('n', observed_n)
    r_success = pm.ConstantData('r', observed_r)

    # Hyperpriors
    mu_p = pm.Beta('mu_p', alpha=2, beta=28)
    tau_p = pm.HalfCauchy('tau_p', beta=0.05)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    kappa = pm.Gamma('kappa', alpha=2, beta=0.1)

    # Group-level success rates (Student-t hierarchy)
    p = pm.TruncatedNormal('p', mu=mu_p, sigma=tau_p,
                           lower=0, upper=1, shape=12)
    # Note: PyMC doesn't have TruncatedStudentT, so we'd need custom distribution
    # or use Bound(pm.StudentT(...), lower=0, upper=1)

    # Beta-binomial likelihood
    alpha = pm.Deterministic('alpha', p * kappa)
    beta_param = pm.Deterministic('beta', (1 - p) * kappa)

    r = pm.BetaBinomial('r', alpha=alpha, beta=beta_param,
                        n=n_trials, observed=r_success)
```

### 2.4 Theoretical Justification

**Why this alternative approach is valuable:**

1. **Robustness to outliers:** Student-t tails prevent extreme groups from dominating hyperparameter estimates
2. **Learns tail heaviness:** nu inferred from data (adaptive robustness)
3. **Beta-binomial overdispersion:** Accounts for extra-binomial variation beyond group heterogeneity
4. **Computational advantages:** Conjugate beta-binomial is well-studied, stable

**What patterns it captures that hierarchical models miss:**

- **Outlier influence:** Groups 4 and 8 get partial downweighting if nu is small
- **Extra-binomial noise:** Beta-binomial allows trial-level variation beyond pure binomial
- **Asymmetric shrinkage:** Heavy tails mean extreme groups shrink less than in normal hierarchy
- **Adaptive pooling:** Amount of pooling depends on tail heaviness (learned from data)

**Link to EDA findings:**

| EDA Finding | How Robust-HBB Captures It |
|-------------|----------------------------|
| 2 extreme outliers (Groups 4, 8) | Heavy tails (nu < 10) downweight influence |
| Variance ratio = 2.78 | kappa parameter captures overdispersion |
| High ICC (0.42) | tau_p reflects between-group variation |
| Large sample outliers (n=810, n=215) | Less shrinkage due to heavy tails |
| Heterogeneity (χ² p < 0.001) | Student-t allows more extreme deviations |

### 2.5 Falsification Criteria

**I will abandon this model if:**

1. **Tails collapse to normal:**
   - Posterior nu > 30 (indistinguishable from normal)
   - 95% CI for nu excludes nu < 10
   - **Diagnostic:** `quantile(nu, 0.95) > 30`
   - **Interpretation:** Heavy tails not needed; standard hierarchy sufficient

2. **No overdispersion detected:**
   - Posterior kappa very large (> 1000)
   - Beta-binomial reduces to binomial
   - **Diagnostic:** `quantile(kappa, 0.05) > 500`
   - **Interpretation:** Binomial model adequate, beta-binomial unnecessary

3. **Worse fit than standard hierarchy:**
   - LOO-ELPD significantly worse (delta > 3)
   - Posterior predictive checks fail (data outside 95% envelope)
   - **Diagnostic:** `loo_compare(robust_hbb, hierarchy)` shows hierarchy wins by > 1 SE

4. **Prior-data conflict:**
   - Posterior concentrates far from prior (mu_p posterior >> 0.15 or << 0.03)
   - Suggests prior misspecified
   - **Diagnostic:** `abs(mean(mu_p_posterior) - 0.07) > 0.05`

5. **Computational instability:**
   - Divergences > 2% despite tuning
   - Truncated Student-t causes boundary issues
   - **Diagnostic:** Monitor divergence warnings in Stan

**Expected differences from standard hierarchy:**

| Metric | Standard Hierarchy | Robust-HBB | Test |
|--------|-------------------|------------|------|
| Outlier shrinkage | Moderate (50%) | Less (30%) | Groups 4, 8 closer to raw rates |
| Hyperparameter SE | Smaller | Larger | tau_p uncertainty higher |
| nu (df parameter) | N/A (fixed at ∞) | 4-15 (estimated) | Posterior concentrated < 30 |
| Overdispersion | None | kappa ≈ 20-100 | Beta-binomial variance > binomial |
| LOO-ELPD | Baseline | Similar or better | Delta < 2 |

**Specific predictions:**
- Group 4 (largest outlier, n=810) should have less shrinkage than in normal hierarchy
- Posterior nu should concentrate between 4-15 (moderately heavy tails)
- kappa should be finite (not > 1000), indicating real overdispersion

### 2.6 Computational Considerations

**Expected challenges:**

1. **Truncated Student-t complexity:**
   - Truncating to (0,1) complicates sampling
   - May cause boundary issues if p_j near 0 or 1
   - **Solution:** Use logit transformation (non-centered parameterization)

2. **Beta-binomial log-likelihood:**
   - More expensive than binomial (requires beta function evaluations)
   - **Impact:** ~1.5x slower than standard hierarchy

3. **nu parameter inference:**
   - Degrees of freedom parameter often has slow mixing
   - May require tighter priors or longer chains
   - **Solution:** 4 chains, 2000 warmup + 2000 sampling

4. **Correlation between mu_p and tau_p:**
   - Common in hierarchical models
   - **Solution:** Non-centered parameterization if needed

**Comparison to hierarchical model complexity:**

| Aspect | Hierarchical | Robust-HBB | Ratio |
|--------|-------------|------------|-------|
| Parameters | 14 | 16 | 1.14x |
| Effective parameters (p_eff) | ~8 | ~10 | 1.25x |
| Sampling time (est.) | 10s | 15s | 1.5x |
| Convergence difficulty | Easy | Moderate | - |
| Risk of divergences | Low | Moderate (1%) | 5-10x |

**Overall assessment:** Moderate complexity increase. Main challenge is truncated Student-t. If nu collapses to > 30, model reduces to standard hierarchy (natural simplification).

---

## Model 3: Dirichlet Process Mixture (Non-parametric)

### 3.1 Model Name & Description

**Dirichlet Process Mixture of Beta-Binomials (DP-MBB)**

This model uses a Dirichlet process prior to allow an infinite number of potential clusters. The data determine the effective number of clusters (no need to pre-specify K). Each cluster has its own success rate distribution.

**How it differs from standard hierarchy:**
- Standard hierarchy: All groups from single population
- Finite mixture (Model 1): Groups from K=3 fixed populations
- This model: Groups from unknown number of populations (K inferred from data)
- Non-parametric: Adapts complexity to data

### 3.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)                     # observed successes
p_j ~ DP(alpha, G0)                          # Dirichlet process
```

**Dirichlet Process Specification:**
```
G ~ DP(alpha, G0)                            # random distribution
p_j ~ G                                      # draw success rate from G

# Base distribution G0 (continuous)
G0 = Beta(a0, b0)                            # centered on prior mean

# Concentration parameter
alpha ~ Gamma(2, 2)                          # controls number of clusters
```

**Stick-breaking Representation (for inference):**
```
# Cluster parameters
p_k* ~ Beta(a0, b0)                          # cluster-specific success rates

# Stick-breaking weights
v_k ~ Beta(1, alpha)                         # stick-breaking proportions
pi_k = v_k * prod_{l<k}(1 - v_l)            # cluster weights

# Cluster assignments
z_j ~ Categorical(pi)                        # latent group assignments
p_j = p_{z_j}*                              # success rate from assigned cluster
```

**Priors:**
```
# Base distribution parameters
a0 = 2, b0 = 28                              # E[p] = 2/30 ≈ 0.067

# Concentration parameter
alpha ~ Gamma(2, 2)                          # E[alpha] = 1, allows 1-5 clusters

# Truncation (for computational tractability)
K_max = 10                                   # maximum number of clusters
```

**Prior Justification:**

- **Beta(2, 28) base:** Centers on pooled rate (0.067), moderate prior sample size (30)
- **Gamma(2, 2) for alpha:** Weak prior on number of clusters; alpha ≈ 1-3 expected (2-5 clusters)
- **K_max = 10:** Computational truncation; with J=12 groups, unlikely to exceed this
- **Non-parametric:** Model complexity adapts to data (parsimony via automatic cluster pruning)

**Full Generative Model:**
```
For each cluster k = 1, ..., K_max:
  1. Draw stick-breaking weight v_k ~ Beta(1, alpha)
  2. Compute cluster probability pi_k = v_k * prod_{l<k}(1 - v_l)
  3. Draw cluster success rate p_k* ~ Beta(2, 28)

For each group j = 1, ..., 12:
  1. Draw cluster assignment z_j ~ Categorical(pi)
  2. Set success rate p_j = p_{z_j}*
  3. Generate data r_j ~ Binomial(n_j, p_j)
```

### 3.3 Stan Implementation Sketch

```stan
data {
  int<lower=1> J;              // number of groups
  int<lower=1> K_max;          // truncation level (e.g., 10)
  array[J] int<lower=0> n;     // trials per group
  array[J] int<lower=0> r;     // successes per group

  // Base distribution parameters
  real<lower=0> a0;            // Beta(a0, b0) base
  real<lower=0> b0;
}

parameters {
  real<lower=0> alpha;         // DP concentration parameter
  vector<lower=0, upper=1>[K_max] v;  // stick-breaking proportions
  vector<lower=0, upper=1>[K_max] p_star;  // cluster-specific success rates
  simplex[K_max] z[J];         // soft cluster assignments (alternative: discrete)
}

transformed parameters {
  simplex[K_max] pi;           // cluster weights
  vector<lower=0, upper=1>[J] p;  // group success rates

  // Stick-breaking construction
  pi[1] = v[1];
  for (k in 2:K_max) {
    pi[k] = v[k] * prod(1 - v[1:(k-1)]);
  }

  // Compute success rates from assignments
  for (j in 1:J) {
    p[j] = dot_product(z[j], p_star);
  }
}

model {
  // Priors
  alpha ~ gamma(2, 2);
  v ~ beta(1, alpha);
  p_star ~ beta(a0, b0);

  // Cluster assignments
  for (j in 1:J) {
    z[j] ~ dirichlet(pi);
  }

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  vector[J] log_lik;
  array[J] int r_rep;
  int K_effective;             // number of active clusters
  array[K_max] int cluster_size;  // groups per cluster

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // Count effective clusters (weight > 0.05)
  K_effective = 0;
  for (k in 1:K_max) {
    if (pi[k] > 0.05) K_effective += 1;
    cluster_size[k] = 0;
    for (j in 1:J) {
      if (z[j][k] > 0.5) cluster_size[k] += 1;
    }
  }
}
```

**PyMC Implementation (using stick-breaking):**
```python
import pymc as pm
import numpy as np

K_max = 10  # truncation

with pm.Model() as dp_mixture:
    # Data
    n_trials = pm.ConstantData('n', observed_n)
    r_success = pm.ConstantData('r', observed_r)

    # DP concentration
    alpha = pm.Gamma('alpha', alpha=2, beta=2)

    # Stick-breaking weights
    v = pm.Beta('v', alpha=1, beta=alpha, shape=K_max)

    # Compute cluster probabilities
    def stick_breaking(v):
        pi = np.empty(K_max)
        remaining = 1.0
        for k in range(K_max - 1):
            pi[k] = v[k] * remaining
            remaining *= (1 - v[k])
        pi[K_max - 1] = remaining
        return pi

    pi = pm.Deterministic('pi', stick_breaking(v))

    # Cluster-specific success rates
    p_star = pm.Beta('p_star', alpha=2, beta=28, shape=K_max)

    # Cluster assignments and success rates
    z = pm.Categorical('z', p=pi, shape=12)
    p = pm.Deterministic('p', p_star[z])

    # Likelihood
    r = pm.Binomial('r', n=n_trials, p=p, observed=r_success)
```

**Note:** Stan code above uses soft assignments (simplex); standard DP uses discrete z. For MCMC, may need specialized sampler or marginalization.

### 3.4 Theoretical Justification

**Why this alternative approach is valuable:**

1. **Model uncertainty about K:** EDA suggests K=3, but this could be artifact; DP doesn't commit
2. **Automatic complexity control:** Model self-regularizes (unused clusters pruned)
3. **Rich gets richer:** Chinese Restaurant Process prior favors merging similar groups
4. **Principled non-parametric inference:** Bayesian framework without parametric assumptions

**What patterns it captures that hierarchical models miss:**

- **Unknown number of subpopulations:** Doesn't assume unimodal or K=3
- **Cluster discovery:** Data determine number and composition of clusters
- **Flexible mixing:** Groups can be "between" clusters (soft assignments)
- **Hierarchical clustering:** Implicit similarity metric via DP

**Link to EDA findings:**

| EDA Finding | How DP-MBB Captures It |
|-------------|------------------------|
| K=3 clusters identified | Posterior K_effective ≈ 3 (if true) |
| Uncertainty about cluster boundaries | Soft assignments, posterior uncertainty in z |
| Possibility of 2 or 4 clusters | Model can infer K=2 or K=4 if better fit |
| No prior commitment to structure | Non-parametric approach |
| Small J (12 groups) | Prior on alpha prevents overfitting |

### 3.5 Falsification Criteria

**I will abandon this model if:**

1. **Single cluster dominates:**
   - K_effective = 1 (posterior collapses to single cluster)
   - One cluster has > 90% of groups
   - **Diagnostic:** `max(cluster_size) >= 11`
   - **Interpretation:** No clustering; standard hierarchy sufficient

2. **Extreme fragmentation:**
   - K_effective > 6 (overfitting, each group its own cluster)
   - Most clusters have single group
   - **Diagnostic:** `sum(cluster_size == 1) > 6`
   - **Interpretation:** No pooling benefit; overfitting

3. **Unstable cluster assignments:**
   - High posterior uncertainty in z (most groups have max probability < 0.6)
   - Cluster membership changes drastically across MCMC samples
   - **Diagnostic:** `mean(max(prob(z), axis=clusters)) < 0.5`

4. **Worse than finite mixture (K=3):**
   - LOO-ELPD significantly worse than FMM-3 (delta > 3)
   - Additional complexity not justified
   - **Diagnostic:** `loo_compare(dp_mbb, fmm3)` shows DP worse

5. **Computational intractability:**
   - MCMC doesn't converge after 8000 iterations
   - Label switching across K_max clusters prevents inference
   - **Diagnostic:** Rhat > 1.02 for cluster assignments

**Expected differences from finite mixture (K=3):**

| Metric | FMM-3 | DP-MBB | Test |
|--------|-------|--------|------|
| Number of clusters | Fixed K=3 | Inferred K_eff | Posterior K_effective distribution |
| Cluster assignments | Hard (ordered) | Soft (probabilistic) | Entropy of z |
| Model complexity | p_eff ≈ 12 | p_eff ≈ 10-15 | DP may be simpler if K_eff < 3 |
| LOO-ELPD | Baseline | Similar or worse | DP penalized for flexibility |
| Interpretability | High (3 groups) | Moderate (uncertain K) | - |

**Specific predictions:**
- Posterior K_effective concentrates on 2-4 (likely 3)
- Cluster assignments mostly clear (max prob > 0.7 for 8+ groups)
- Similar fit to FMM-3 but with uncertainty about K

### 3.6 Computational Considerations

**Expected challenges:**

1. **Discrete cluster assignments:**
   - Categorical z is discrete → requires specialized samplers
   - **Solution:** Marginalize over z (collapsed Gibbs) or use soft assignments
   - Stan doesn't support discrete parameters well → need marginalization

2. **Label switching (worse than FMM):**
   - K_max=10 clusters can permute → multimodal posterior
   - **Solution:** Post-processing to align labels across samples
   - Not as bad as FMM if most clusters empty

3. **Truncation level sensitivity:**
   - K_max too small: biased results
   - K_max too large: slow convergence, wasted computation
   - **Solution:** Check sensitivity to K_max = 5, 10, 15

4. **Stick-breaking construction:**
   - Sequential dependence in v_k → slower mixing
   - **Solution:** Centered parameterization, longer chains

5. **Small J (12 groups):**
   - Limited data to infer K clusters + alpha + cluster parameters
   - **Risk:** Overfitting or underfitting
   - **Solution:** Informative prior on alpha to regularize

**Comparison to other models:**

| Aspect | Hierarchical | FMM-3 | DP-MBB |
|--------|-------------|-------|--------|
| Parameters | 14 | 20 | 25-35 (depends on K_max) |
| Effective parameters | ~8 | ~12 | ~10-15 |
| Sampling time (est.) | 10s | 20s | 40s |
| Convergence difficulty | Easy | Moderate | Hard |
| Risk of divergences | Low | Moderate | High (5%+) |
| Interpretability | High | High | Moderate |

**Overall assessment:** Most complex model. Main value is testing whether K=3 is robust or artifact. If K_effective posterior concentrates on 3, validates FMM-3. If K_effective = 1, supports standard hierarchy. If K_effective ≠ 3, reveals EDA limitation.

**Practical recommendation:** Run DP-MBB as robustness check AFTER fitting FMM-3. If results similar (K_eff ≈ 3), increases confidence. If different, reveals model sensitivity.

---

## Model Comparison and Selection Strategy

### 4.1 Decision Tree

```
START
  |
  ├─ Fit all 3 models + standard hierarchy baseline
  |
  ├─ Check computational diagnostics
  |   ├─ Any model fails (Rhat > 1.01, divergences > 2%)? → REJECT
  |   └─ All pass → Continue
  |
  ├─ Compare LOO-CV
  |   ├─ Clear winner (delta > 5)? → SELECT + validate
  |   ├─ Similar performance (delta < 3)? → Check theory
  |   └─ Uncertain? → Posterior predictive checks
  |
  ├─ Theoretical validation
  |   ├─ FMM-3: Check K_effective = 3 and clear assignments
  |   ├─ Robust-HBB: Check nu < 30 and kappa finite
  |   ├─ DP-MBB: Check K_effective ∈ [2, 5]
  |   └─ Any theory failure? → REJECT
  |
  ├─ Robustness checks
  |   ├─ Prior sensitivity (3 prior specifications)
  |   ├─ Posterior predictive checks (calibration)
  |   └─ Leave-one-out predictions (accuracy)
  |
  └─ DECISION
      ├─ Single best model → REPORT + interpret
      ├─ Model averaging → ENSEMBLE (weighted by LOO)
      └─ All fail → RECONSIDER (new model class)
```

### 4.2 Expected Outcomes

**Scenario 1: Mixture model wins**
- FMM-3 has best LOO-ELPD (delta > 3)
- K_effective ≈ 3 in DP-MBB (validates K=3)
- Clear cluster assignments (prob > 0.7)
- **Conclusion:** Discrete heterogeneity is real; 3 subpopulations

**Scenario 2: Robust model wins**
- Robust-HBB has best LOO-ELPD
- nu ∈ [4, 15] (moderately heavy tails)
- Outliers less influential than in normal hierarchy
- **Conclusion:** Continuous heterogeneity with extreme outliers

**Scenario 3: Standard hierarchy wins**
- Baseline outperforms all alternatives
- FMM clusters collapse (mu_k similar)
- Robust-HBB has nu > 30 (normal tails)
- DP-MBB has K_effective = 1
- **Conclusion:** EDA clusters are sampling artifacts; simple hierarchy sufficient

**Scenario 4: Model uncertainty**
- Multiple models similar (LOO delta < 2)
- **Resolution:** Model averaging (BMA weights by stacking)
- **Interpretation:** Cluster structure is weak/ambiguous

### 4.3 Red Flags Triggering Strategy Pivot

**STOP and reconsider everything if:**

1. **All models fail posterior predictive checks**
   - Data systematically outside 95% prediction envelope
   - **Implication:** Missing key feature (covariate? non-exchangeability?)

2. **Prior-data conflict across all priors**
   - All sensitivity analyses push parameters to prior boundaries
   - **Implication:** Prior misspecification or wrong likelihood

3. **Extreme parameter values**
   - kappa > 10,000 (no overdispersion)
   - nu → 1 (Cauchy tails, pathological)
   - All groups assigned to single cluster
   - **Implication:** Model class inappropriate

4. **Computational failure across all models**
   - All models have Rhat > 1.01 or divergences > 5%
   - **Implication:** Fundamental misspecification or data issue

5. **Inconsistent results across data subsets**
   - Leave-one-out analyses give drastically different cluster assignments
   - **Implication:** Overfitting or unstable inference

**Alternative approaches if all fail:**
- Covariates (sample size, temporal order)
- Observation-level models (account for trial heterogeneity)
- Non-binomial likelihoods (e.g., beta-binomial at trial level)
- Suspect data quality (re-examine for errors)

### 4.4 Success Criteria

**I will consider this design successful if:**

1. **At least one model converges and fits well** (Rhat < 1.01, no divergences)
2. **Clear differentiation between models** (LOO delta > 2 for top vs. others)
3. **Results consistent with EDA** (e.g., if FMM-3 wins, clusters match EDA)
4. **Interpretable findings** (can explain why model class is appropriate)
5. **Robust to prior specification** (qualitative conclusions unchanged)

**Crucially:** Success does NOT require all models to work. Finding that only standard hierarchy works (all alternatives fail falsification) is a successful outcome—it means EDA patterns were artifacts and simple model is best.

---

## Summary Table: Model Comparison

| Feature | FMM-3 | Robust-HBB | DP-MBB | Standard Hierarchy |
|---------|-------|------------|--------|-------------------|
| **Model Class** | Finite mixture | Robust hierarchy | Non-parametric | Normal hierarchy |
| **Clusters** | K=3 (fixed) | None (continuous) | K=? (inferred) | None |
| **Tail behavior** | Normal | Student-t | Normal | Normal |
| **Parameters** | 20 | 16 | 25-35 | 14 |
| **p_eff (est.)** | 12 | 10 | 10-15 | 8 |
| **Complexity** | Moderate | Moderate | High | Low |
| **Interpretability** | High | Moderate | Moderate | High |
| **Computation time** | 2x | 1.5x | 4x | 1x |
| **Main strength** | Captures clusters | Robust to outliers | Flexible K | Simple, stable |
| **Main weakness** | Assumes K=3 | May overfit tails | Hard to converge | May underfit |
| **Falsification test** | Clusters collapse | nu > 30 | K_eff = 1 or > 6 | LOO worse than others |
| **Best if...** | Discrete groups | Heavy outliers | Uncertain K | No real heterogeneity |

---

## Implementation Plan

### Phase 1: Sequential Fitting (Fail-Fast)
1. **Start with Robust-HBB** (simplest alternative, most likely to work)
   - If fails → standard hierarchy likely best
   - If succeeds → proceed to FMM-3

2. **Fit FMM-3** (tests cluster hypothesis directly)
   - If clear clusters (prob > 0.7) → likely winner
   - If ambiguous → DP-MBB for robustness

3. **Fit DP-MBB** (most complex, only if needed)
   - Only run if FMM-3 shows promise but K uncertain
   - Validate K=3 assumption

### Phase 2: Comparison
- Compute LOO-CV for all converged models
- Compare ELPD (delta > 3 = clear winner)
- Posterior predictive checks
- Prior sensitivity (3 specifications each)

### Phase 3: Validation
- Leave-one-out predictions (RMSE, calibration)
- Cluster stability (if applicable)
- Parameter interpretation (align with domain knowledge)

### Phase 4: Reporting
- Winner + runner-up (if delta < 5)
- Model averaging weights (stacking)
- Falsification results (what made us reject alternatives)
- Limitations and caveats

---

## Files to Generate

All files in `/workspace/experiments/designer_2/`:

1. **proposed_models.md** (this document)
2. **model_fmm3.stan** - Finite mixture Stan code
3. **model_robust_hbb.stan** - Robust beta-binomial Stan code
4. **model_dp_mbb.stan** - Dirichlet process Stan code
5. **fit_alternatives.py** - Python script to fit all models
6. **compare_models.py** - LOO comparison and diagnostics
7. **falsification_tests.py** - Automated checks for each model
8. **results_summary.md** - Findings after fitting (to be generated later)

---

## Key Takeaways

1. **All three models are scientifically motivated** by EDA findings (clusters, outliers, heterogeneity)
2. **Each has clear falsification criteria** - we know what evidence would make us reject it
3. **Computational cost increases** with flexibility (HBB: 1.5x, FMM: 2x, DP: 4x)
4. **Success = finding the right model class**, not forcing all to work
5. **Be ready to abandon all** if standard hierarchy outperforms (means EDA patterns were noise)

**Final recommendation:** Start with Robust-HBB (good risk/reward), then FMM-3 if clusters are real. DP-MBB is for robustness checking, not primary analysis.
