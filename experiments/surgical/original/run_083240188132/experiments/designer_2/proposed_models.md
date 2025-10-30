# Alternative Bayesian Model Specifications (Designer 2)

## Executive Summary

This document proposes three alternative Bayesian model classes that offer fundamentally different perspectives on the binomial outcome data. Each model is designed to capture different structural assumptions about the data generation process that standard beta-binomial or random effects models might miss.

**Key Data Features to Address**:
- Strong heterogeneity (ICC = 0.66)
- Substantial overdispersion (φ = 3.5-5.1)
- Possible bimodal distribution (typical ~7%, elevated ~12-14%)
- Three high outliers (Groups 2, 8, 11)
- One zero-event group (Group 1)

**Proposed Alternative Models**:
1. **Finite Mixture Model**: Captures latent subpopulations with distinct risk profiles
2. **Robust Hierarchical Model with Student-t**: Handles outliers without contaminating population estimates
3. **Dirichlet Process Mixture**: Nonparametric approach that learns the number of latent clusters

---

## Model 1: Finite Mixture Model (Bimodal Risk Structure)

### 1.1 Theoretical Justification

**Core hypothesis**: The groups come from 2-3 distinct latent subpopulations with different baseline risk levels, rather than being continuous draws from a single hierarchical distribution.

**Evidence from EDA**:
- Distribution appears bimodal: ~6 groups around 6-7%, ~3 groups around 11-14%
- Three outliers cluster together (Groups 2, 8, 11: 12.2%, 14.4%, 11.3%)
- Remaining groups more homogeneous (typical cluster: 3.8-8.2%)
- If truly bimodal, forcing a unimodal hierarchical prior may miss this structure

**When this perspective is valuable**:
- If groups represent fundamentally different populations (e.g., different treatment protocols, patient risk groups, geographical regions)
- If there's a latent categorical variable that wasn't measured
- If mixture components correspond to actionable subgroups

### 1.2 Model Specification

**Likelihood**:
```
r_i | n_i, theta_i ~ Binomial(n_i, theta_i)
```

**Mixture structure** (2-component for identifiability):
```
theta_i ~ w * Beta(alpha_1, beta_1) + (1-w) * Beta(alpha_2, beta_2)

Equivalently:
z_i ~ Categorical(p=[w, 1-w])  # Component assignment
theta_i | z_i=k ~ Beta(alpha_k, beta_k)
```

**Component ordering constraint** (for identifiability):
```
mu_1 < mu_2  where mu_k = alpha_k / (alpha_k + beta_k)
```

**Priors**:
```
# Mixture weight (weakly informative toward equal components)
w ~ Beta(2, 2)

# Component means (informative based on EDA: low-risk ~6%, high-risk ~12%)
mu_1 ~ Beta(6, 94)  # Centers at 0.06, SD ~0.024
mu_2 ~ Beta(12, 88)  # Centers at 0.12, SD ~0.033

# Component concentration (controls within-component variance)
kappa_1 ~ Gamma(2, 0.1)  # Weakly informative, mean=20
kappa_2 ~ Gamma(2, 0.1)

# Transform to alpha, beta parameterization
alpha_k = mu_k * kappa_k
beta_k = (1 - mu_k) * kappa_k
```

**Rationale for priors**:
- `w ~ Beta(2, 2)`: Weakly informative, allows data to determine mixture proportion
- `mu_k ~ Beta()`: Centers on observed bimodal peaks, but allows substantial flexibility
- `kappa_k ~ Gamma(2, 0.1)`: Weakly informative on concentration (inverse variance)
- Prior predictive will span 0-20% range with most mass in observed region

### 1.3 How This Addresses Key Challenges

**Heterogeneity**: Explicitly models two distinct subpopulations rather than continuous variation
**Overdispersion**: Mixture variance includes both within-component and between-component variance
**Outliers**: High-risk component naturally accommodates Groups 2, 8, 11 without treating them as outliers
**Zero events**: Group 1 assigned to low-risk component with beta prior regularization
**Bimodality**: Directly models the observed bimodal pattern

### 1.4 Unique Advantages

1. **Interpretable subpopulations**: Component membership has clear interpretation (low-risk vs high-risk groups)
2. **Actionable insights**: If mixture components correspond to modifiable factors, this identifies intervention targets
3. **Better prediction**: For new groups, predicts not just a single distribution but which component they likely belong to
4. **Natural handling of outliers**: High-rate groups aren't "outliers" but members of a distinct subpopulation
5. **Testable hypothesis**: Can formally compare mixture vs. unimodal hierarchical model

### 1.5 Prior Specification Strategy

**Prior predictive simulation** (before fitting):
1. Draw 10,000 samples from prior
2. Simulate datasets with n=12 groups
3. Verify:
   - Generates bimodal distributions plausibly
   - Component means span observed range (0-20%)
   - Mixture weights range from 0.1-0.9 (not locked to 0.5)
   - Individual group rates stay in 0-20% range

**Sensitivity analysis plan**:
- Vary `kappa` priors (more/less concentrated components)
- Vary `mu` priors (less informative: `Beta(1,1)` for each)
- Try symmetric vs. asymmetric weight priors

### 1.6 Implementation Plan (PyMC)

```python
import pymc as pm
import numpy as np

def build_mixture_model(n, r):
    """
    Two-component finite mixture model for binomial outcomes.

    Parameters
    ----------
    n : array (n_groups,)
        Sample sizes per group
    r : array (n_groups,)
        Event counts per group
    """
    n_groups = len(n)

    with pm.Model() as mixture_model:
        # Mixture weight
        w = pm.Beta('w', alpha=2, beta=2)

        # Component means (ordered: mu_1 < mu_2)
        mu_raw = pm.Beta('mu_raw', alpha=[6, 12], beta=[94, 88], shape=2)
        mu = pm.Deterministic('mu', pm.math.sort(mu_raw))  # Enforce ordering

        # Component concentrations
        kappa = pm.Gamma('kappa', alpha=2, beta=0.1, shape=2)

        # Beta distribution parameters
        alpha = pm.Deterministic('alpha', mu * kappa)
        beta_param = pm.Deterministic('beta_param', (1 - mu) * kappa)

        # Component assignments (latent variable)
        z = pm.Categorical('z', p=pm.math.stack([w, 1-w]), shape=n_groups)

        # Group-level probabilities (depends on component assignment)
        theta = pm.Beta('theta',
                       alpha=alpha[z],
                       beta=beta_param[z],
                       shape=n_groups)

        # Likelihood
        y = pm.Binomial('y', n=n, p=theta, observed=r)

        # Derived quantities
        component_probs = pm.Deterministic('component_probs',
                                          pm.math.stack([w, 1-w]))

        # Posterior predictive for new group
        z_new = pm.Categorical('z_new', p=component_probs)
        theta_new = pm.Beta('theta_new',
                           alpha=alpha[z_new],
                           beta=beta_param[z_new])
        r_new = pm.Binomial('r_new', n=100, p=theta_new)

    return mixture_model

# Sampling strategy
def fit_mixture_model(model):
    """Fit mixture model with appropriate MCMC settings."""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,  # Higher for mixture models
            return_inferencedata=True,
            random_seed=42
        )
    return trace
```

**Computational considerations**:
- Mixture models have multimodal posteriors (label switching)
- Solution: Order constraint on means (`mu_1 < mu_2`)
- Alternative: Post-hoc relabeling based on means
- May need longer warmup (tune=2000) and higher target_accept
- Check for mode switching in traces

### 1.7 Falsification Criteria

**I will abandon this model if**:

1. **Posterior component weights are extreme** (w < 0.1 or w > 0.9)
   - Interpretation: One component is nearly empty, suggesting unimodal structure is sufficient
   - Action: Revert to standard hierarchical beta-binomial

2. **Component means are too close** (|mu_2 - mu_1| < 0.03)
   - Interpretation: Components are not meaningfully distinct
   - Action: Single-component model is more parsimonious

3. **Poor LOO performance** compared to unimodal hierarchical model
   - Interpretation: Extra complexity not justified by predictive improvement
   - Action: Use simpler model

4. **Component assignments lack coherence**
   - Interpretation: Groups flip between components across posterior samples (weak identification)
   - Action: Model is overfitting, use simpler structure

5. **Prior-posterior conflict** on component weights
   - If posterior sharply contradicts prior (e.g., prior allows 0.1-0.9, posterior is 0.99)
   - Action: Either use stronger prior or admit unimodal structure

**Red flags**:
- Divergent transitions (indicates difficult geometry)
- Low effective sample size on `z` (component assignments)
- Trace plots showing mode switching even with ordering constraint

### 1.8 Comparison to Standard Hierarchical Approaches

**When to prefer Finite Mixture**:
- Evidence of distinct subpopulations (bimodal distribution)
- Scientific interest in identifying group membership
- Predictive interest: which component does a new group belong to?
- Within-component homogeneity > between-component heterogeneity

**When to prefer Standard Hierarchical Beta-Binomial**:
- Continuous heterogeneity (unimodal)
- No scientific basis for discrete subpopulations
- LOO favors simpler model
- Mixture fails falsification criteria above

**Key distinction**: Mixture assumes **discrete** latent structure; standard hierarchical assumes **continuous** variation.

---

## Model 2: Robust Hierarchical Model with Student-t

### 2.1 Theoretical Justification

**Core hypothesis**: The population distribution of risk rates has heavier tails than a beta distribution can capture, and outliers (Groups 2, 8, 11) represent draws from these heavy tails rather than being fundamentally different or contaminated.

**Evidence from EDA**:
- Three groups are 2-4 standard deviations above mean
- Standard beta-binomial may over-shrink these legitimate high-risk groups
- Overdispersion (φ = 5.1) suggests more extreme variation than beta can easily accommodate
- If outliers are "real" (not errors), we want a model that accommodates them without contamination

**When this perspective is valuable**:
- If extreme groups are legitimate (not errors or different populations)
- If we want to estimate population mean robustly without contamination
- If we believe true risk distribution has fat tails (more extreme groups than beta suggests)
- If we want less aggressive shrinkage for outliers

### 2.2 Model Specification

**Likelihood**:
```
r_i | n_i, theta_i ~ Binomial(n_i, theta_i)
```

**Hierarchical structure** (logit scale for robustness):
```
logit(theta_i) = mu + tau * eta_i

eta_i ~ Student-t(nu, 0, 1)  # Heavy-tailed random effects

Equivalently:
logit(theta_i) ~ Student-t(nu, mu, tau)
```

**Priors**:
```
# Population mean (logit scale)
mu ~ Normal(logit(0.07), 1)  # Centers at pooled estimate

# Population scale (between-group variation)
tau ~ HalfCauchy(0, 0.5)  # Weakly informative, allows large heterogeneity

# Degrees of freedom (controls tail heaviness)
nu ~ Gamma(2, 0.1)  # Weakly informative, mean=20
# nu=1: Cauchy (very heavy tails)
# nu=30: Nearly normal
# nu=5-10: Moderately heavy tails (likely range)
```

**Why Student-t on logit scale?**:
- Logit scale: Ensures 0 < theta < 1 constraint automatically
- Student-t: Heavy tails accommodate outliers naturally
- As nu → ∞, approaches standard normal random effects
- Data determines optimal tail weight via nu

### 2.3 How This Addresses Key Challenges

**Heterogeneity**: `tau` parameter captures between-group variation robustly
**Overdispersion**: Heavy tails accommodate extra-binomial variation
**Outliers**: Groups 2, 8, 11 receive less shrinkage; their high rates influence population estimates less
**Zero events**: Group 1 still shrunk toward population mean, but less aggressively if it's a genuine low outlier
**Robustness**: Extreme groups don't contaminate population-level estimates

### 2.4 Unique Advantages

1. **Robust population estimates**: Outliers have less influence on mu estimate
2. **Adaptive shrinkage**: Amount of shrinkage depends on tail weight (nu parameter)
3. **Falsifiable**: If data supports nu > 30, reverts to normal random effects (standard approach)
4. **Principled outlier handling**: Doesn't treat outliers as "bad" but as draws from heavy-tailed distribution
5. **Less aggressive for extremes**: High-risk groups retain more of their identity
6. **Interpretable**: If nu is small (5-10), this tells us the population is genuinely heavy-tailed

### 2.5 Prior Specification Strategy

**Prior predictive simulation**:
1. Draw 10,000 samples from prior
2. Simulate datasets with n=12 groups
3. Verify:
   - `nu` prior allows range from near-Cauchy (nu=2) to near-normal (nu=30)
   - Population mean stays in reasonable range (1-15%)
   - `tau` prior generates heterogeneity comparable to observed (ICC ~ 0.5-0.8)
   - Occasional extreme groups (>15%) are possible but not dominant

**Sensitivity analysis plan**:
- Try different priors on nu: `Exponential(0.05)`, `Uniform(2, 50)`
- More informative prior on mu: `Normal(logit(0.07), 0.5)` (tighter)
- Compare to normal random effects (fix nu=30)

### 2.6 Implementation Plan (PyMC)

```python
import pymc as pm
import numpy as np

def build_robust_hierarchical_model(n, r):
    """
    Robust hierarchical model with Student-t random effects.

    Parameters
    ----------
    n : array (n_groups,)
        Sample sizes per group
    r : array (n_groups,)
        Event counts per group
    """
    n_groups = len(n)

    with pm.Model() as robust_model:
        # Population parameters (on logit scale)
        mu = pm.Normal('mu', mu=pm.math.log(0.07/(1-0.07)), sigma=1)
        tau = pm.HalfCauchy('tau', beta=0.5)

        # Degrees of freedom (tail weight)
        nu = pm.Gamma('nu', alpha=2, beta=0.1)

        # Group-level random effects (heavy-tailed)
        eta = pm.StudentT('eta', nu=nu, mu=0, sigma=1, shape=n_groups)

        # Group-level logits
        logit_theta = pm.Deterministic('logit_theta', mu + tau * eta)

        # Transform to probability scale
        theta = pm.Deterministic('theta', pm.math.invlogit(logit_theta))

        # Likelihood
        y = pm.Binomial('y', n=n, p=theta, observed=r)

        # Derived quantities
        # Population mean on probability scale
        pop_mean = pm.Deterministic('pop_mean', pm.math.invlogit(mu))

        # Between-group SD on probability scale (approximation)
        # Using delta method: SD(p) ≈ SD(logit(p)) * p * (1-p)
        pop_sd_approx = pm.Deterministic('pop_sd_approx',
                                         tau * pop_mean * (1 - pop_mean))

        # Posterior predictive for new group
        eta_new = pm.StudentT('eta_new', nu=nu, mu=0, sigma=1)
        logit_theta_new = pm.Deterministic('logit_theta_new', mu + tau * eta_new)
        theta_new = pm.Deterministic('theta_new', pm.math.invlogit(logit_theta_new))
        r_new = pm.Binomial('r_new', n=100, p=theta_new)

        # Shrinkage factors (for diagnostics)
        # Smaller for outliers in heavy-tailed model
        shrinkage = pm.Deterministic('shrinkage',
                                     1 / (1 + tau**2 / (1 + 1/nu)))

    return robust_model

# Sampling strategy
def fit_robust_model(model):
    """Fit robust model with appropriate MCMC settings."""
    with model:
        trace = pm.sample(
            draws=2000,
            tune=1500,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42
        )
    return trace
```

**Computational considerations**:
- Student-t can have slower mixing than normal
- May need higher target_accept (0.95)
- `nu` parameter can be tricky to estimate (often needs more samples)
- Check that nu is identifiable (posterior not equal to prior)

### 2.7 Falsification Criteria

**I will abandon this model if**:

1. **Posterior nu > 30 with high confidence** (95% CI entirely above 30)
   - Interpretation: Normal random effects are sufficient; heavy tails not needed
   - Action: Use standard hierarchical model with normal random effects

2. **Posterior nu < 2 with high confidence**
   - Interpretation: Extremely heavy tails suggest model misspecification (perhaps mixture is better)
   - Action: Consider mixture model or investigate outliers as separate population

3. **Prior-posterior conflict on tau or mu**
   - If model strongly contradicts prior (e.g., tau posterior is extreme)
   - Action: Rethink prior or model structure

4. **No improvement over normal random effects**
   - LOO comparison shows no meaningful difference
   - Action: Use simpler normal model (Occam's razor)

5. **Divergent transitions persist** despite tuning
   - Indicates difficult geometry, possible misspecification
   - Action: Reparameterize or reconsider model structure

**Red flags**:
- `nu` posterior equals prior (non-identifiability)
- Outliers still shrunk nearly as much as in normal model (model not doing its job)
- Extreme Rhat or low ESS on nu parameter

### 2.8 Comparison to Standard Hierarchical Approaches

**When to prefer Robust Student-t Model**:
- Evidence of heavy tails (nu < 10)
- Outliers are believed to be legitimate (not errors or distinct populations)
- Want population mean robust to extremes
- Prior belief that risk distribution is fat-tailed

**When to prefer Standard Normal Random Effects**:
- nu posterior suggests near-normality (nu > 30)
- Outliers are few and clearly distinct (mixture might be better)
- Computational efficiency is critical
- No strong prior belief in heavy tails

**Key distinction**: Robust model assumes **continuous heavy-tailed** distribution; normal model assumes light tails with aggressive shrinkage.

---

## Model 3: Dirichlet Process Mixture (Nonparametric)

### 3.1 Theoretical Justification

**Core hypothesis**: We don't know how many latent subpopulations exist (could be 1, 2, 3, or more), and we want the data to determine this rather than pre-specifying it.

**Evidence from EDA**:
- Unclear if distribution is unimodal, bimodal, or more complex
- Three outliers might be: (a) one cluster, (b) three separate anomalies, (c) part of continuous distribution
- Finite mixture requires choosing K (number of components) a priori
- Dirichlet Process (DP) learns K from data

**When this perspective is valuable**:
- When we genuinely don't know the clustering structure
- When we want to avoid committing to K components
- When we want the most flexible model to discover latent structure
- When interpretability of clusters is secondary to predictive performance

### 3.2 Model Specification

**Likelihood**:
```
r_i | n_i, theta_i ~ Binomial(n_i, theta_i)
```

**Dirichlet Process Mixture structure**:
```
theta_i ~ sum_{k=1}^{infinity} pi_k * Beta(alpha_k, beta_k)

where:
- pi ~ Stick-Breaking(alpha_dp)  # Infinite mixture weights
- alpha_k, beta_k ~ base distribution H_0
```

**Stick-breaking representation** (computational):
```
# DP concentration parameter
alpha_dp ~ Gamma(2, 2)  # Weakly informative, mean=1

# Base distribution for component parameters
# (mean, concentration) parameterization
mu_k ~ Beta(7, 93)  # Weakly informative around pooled mean
kappa_k ~ Gamma(2, 0.1)

alpha_k = mu_k * kappa_k
beta_k = (1 - mu_k) * kappa_k

# Stick-breaking weights (infinite, truncated in practice)
v_k ~ Beta(1, alpha_dp) for k=1, 2, ..., K_max
pi_k = v_k * prod_{j<k}(1 - v_j)
```

**Truncation**: In practice, truncate at K_max = 8-10 components (sufficient for n=12 groups).

**Priors**:
```
# DP concentration (controls expected number of clusters)
alpha_dp ~ Gamma(2, 2)
# alpha_dp=0.1: few clusters (1-2)
# alpha_dp=1: moderate (2-4 clusters)
# alpha_dp=10: many clusters (5+ clusters)

# Base distribution parameters
mu_0 ~ Beta(7, 93)  # Centers at 0.07
kappa_0 ~ Gamma(2, 0.1)  # Weakly informative

# Each component draws from base
mu_k ~ Beta(7, 93) for k=1..K_max
kappa_k ~ Gamma(2, 0.1) for k=1..K_max
```

### 3.3 How This Addresses Key Challenges

**Heterogeneity**: Discovers latent cluster structure automatically
**Overdispersion**: Mixture variance includes within- and between-component variance
**Outliers**: Can form their own clusters if distinct enough; otherwise merged with similar groups
**Zero events**: Assigned to appropriate cluster with beta prior regularization
**Model selection**: Learns number of clusters from data (no need to compare K=1 vs K=2 vs K=3)

### 3.4 Unique Advantages

1. **No pre-commitment to K**: Data determines number of meaningful clusters
2. **Maximum flexibility**: Can represent unimodal, bimodal, or more complex distributions
3. **Automatic model selection**: Effective number of clusters is a posterior quantity
4. **Uncertainty in K**: Posterior includes uncertainty about number of clusters
5. **Theoretically principled**: Well-developed Bayesian nonparametric framework
6. **Handles ambiguous cases**: If unclear whether outliers form a cluster, posterior reflects this uncertainty

### 3.5 Prior Specification Strategy

**Prior predictive simulation**:
1. Draw 10,000 samples from prior
2. For each draw, determine effective K using DP stick-breaking
3. Verify:
   - Prior on K is diffuse but reasonable (1-8 clusters plausible)
   - Component means span 0-20%
   - alpha_dp prior generates reasonable cluster propensities
   - Occasional samples have many clusters, most have 2-4

**Sensitivity analysis plan**:
- Vary alpha_dp prior: `Gamma(1, 1)`, `Gamma(5, 2)` (different propensities for clustering)
- Vary base distribution: More/less informative on mu_0
- Different truncation levels: K_max = 6, 10, 15

### 3.6 Implementation Plan (PyMC)

```python
import pymc as pm
import numpy as np

def build_dp_mixture_model(n, r, K_max=10):
    """
    Dirichlet Process mixture model for binomial outcomes.

    Parameters
    ----------
    n : array (n_groups,)
        Sample sizes per group
    r : array (n_groups,)
        Event counts per group
    K_max : int
        Truncation level for stick-breaking (default 10)
    """
    n_groups = len(n)

    with pm.Model() as dp_model:
        # DP concentration parameter
        alpha_dp = pm.Gamma('alpha_dp', alpha=2, beta=2)

        # Base distribution parameters
        mu_0 = pm.Beta('mu_0', alpha=7, beta=93)
        kappa_0 = pm.Gamma('kappa_0', alpha=2, beta=0.1)

        # Component-specific parameters (draw from base distribution)
        mu_k = pm.Beta('mu_k', alpha=7, beta=93, shape=K_max)
        kappa_k = pm.Gamma('kappa_k', alpha=2, beta=0.1, shape=K_max)

        # Beta parameters for each component
        alpha_k = pm.Deterministic('alpha_k', mu_k * kappa_k)
        beta_k = pm.Deterministic('beta_k', (1 - mu_k) * kappa_k)

        # Stick-breaking weights
        v = pm.Beta('v', alpha=1, beta=alpha_dp, shape=K_max)

        # Convert to mixture weights via stick-breaking
        # pi_k = v_k * prod_{j<k}(1 - v_j)
        # Use log-space for numerical stability
        log_w = pm.math.log(v)
        log_1minus_w = pm.math.log(1 - v)
        log_cum_prod = pm.math.concatenate([[0], pm.math.cumsum(log_1minus_w[:-1])])
        log_pi = log_w + log_cum_prod
        pi = pm.Deterministic('pi', pm.math.exp(log_pi))

        # Component assignments
        z = pm.Categorical('z', p=pi, shape=n_groups)

        # Group-level probabilities
        theta = pm.Beta('theta',
                       alpha=alpha_k[z],
                       beta=beta_k[z],
                       shape=n_groups)

        # Likelihood
        y = pm.Binomial('y', n=n, p=theta, observed=r)

        # Derived quantities
        # Effective number of clusters (clusters with >5% probability mass)
        K_eff = pm.Deterministic('K_eff', pm.math.sum(pi > 0.05))

        # Posterior predictive for new group
        z_new = pm.Categorical('z_new', p=pi)
        theta_new = pm.Beta('theta_new',
                           alpha=alpha_k[z_new],
                           beta=beta_k[z_new])
        r_new = pm.Binomial('r_new', n=100, p=theta_new)

    return dp_model

# Sampling strategy
def fit_dp_model(model):
    """Fit DP model with appropriate MCMC settings."""
    with model:
        # DP mixtures are challenging; need careful tuning
        trace = pm.sample(
            draws=3000,  # More samples for nonparametrics
            tune=2000,
            chains=4,
            target_accept=0.95,
            init='adapt_diag',  # Better initialization
            return_inferencedata=True,
            random_seed=42
        )
    return trace

# Post-processing: Analyze effective K
def analyze_effective_clusters(trace):
    """
    Determine effective number of clusters from DP posterior.
    """
    # Extract mixture weights
    pi_samples = trace.posterior['pi'].values  # (chains, draws, K_max)

    # For each posterior sample, count components with >5% mass
    K_eff_samples = np.sum(pi_samples > 0.05, axis=-1)

    # Posterior distribution of K
    K_eff_mean = K_eff_samples.mean()
    K_eff_mode = stats.mode(K_eff_samples.flatten())[0]

    return {
        'mean_clusters': K_eff_mean,
        'modal_clusters': K_eff_mode,
        'K_distribution': K_eff_samples.flatten()
    }
```

**Computational considerations**:
- DP mixtures are computationally expensive (K_max components, label switching)
- May need longer chains (draws=3000+)
- Truncation level K_max should be validated (ensure largest component < 5% mass)
- Label switching can occur; may need post-hoc relabeling
- Check that alpha_dp posterior is well-identified

### 3.7 Falsification Criteria

**I will abandon this model if**:

1. **Posterior effective K = 1 with high confidence** (95% of samples have K_eff=1)
   - Interpretation: No meaningful clustering; unimodal hierarchical model sufficient
   - Action: Revert to standard beta-binomial hierarchical

2. **Posterior K is uniformly distributed** (no clear peak)
   - Interpretation: Model cannot determine cluster structure; data insufficient
   - Action: Use more structured model (finite mixture with K=2 or standard hierarchical)

3. **Extreme computational difficulties**
   - Persistent divergences, very low ESS (<100), non-convergence
   - Action: DP is too complex for this dataset; use simpler model

4. **Clusters are not interpretable or actionable**
   - Posterior clusters don't correspond to any meaningful grouping
   - Action: Use simpler model; added complexity not justified

5. **LOO strongly prefers simpler model**
   - DP performs worse than finite mixture (K=2) or standard hierarchical
   - Action: Occam's razor; use simpler model

**Red flags**:
- K_eff posterior equals prior (non-identifiability)
- alpha_dp posterior has not moved from prior
- Label switching still present after relabeling attempts
- Excessive computational time (>1 hour for 12 groups suggests misspecification)

### 3.8 Comparison to Standard Hierarchical Approaches

**When to prefer DP Mixture**:
- Genuinely uncertain about number of clusters
- Want maximum flexibility to discover structure
- Computational resources available (can afford longer MCMC)
- Scientific interest in discovering latent structure without pre-commitment

**When to prefer Finite Mixture (K=2)**:
- Strong prior belief in specific number of clusters
- Computational efficiency important
- Interpretability of specific clusters is critical
- Smaller datasets (DP needs more data to identify)

**When to prefer Standard Hierarchical**:
- Evidence suggests unimodal distribution
- LOO prefers simpler model
- Clustering is not scientifically meaningful
- Computational simplicity is priority

**Key distinction**: DP is **nonparametric** (learns cluster count); finite mixture is **semi-parametric** (fixed K); standard hierarchical is **parametric** (single population).

---

## Comparative Summary

### Model Comparison Matrix

| Aspect | Finite Mixture (K=2) | Robust (Student-t) | DP Mixture |
|--------|---------------------|-------------------|------------|
| **Structural assumption** | Discrete subpopulations | Heavy-tailed continuous | Unknown clusters |
| **Number of parameters** | ~8-10 | ~15 | ~3K_max + 15 |
| **Complexity** | Moderate | Low-Moderate | High |
| **Interpretability** | High (cluster membership) | Moderate (tail weight) | Moderate (cluster count) |
| **Computational cost** | Moderate | Low | High |
| **When best** | Evidence of bimodality | Outliers are legitimate | Uncertain about structure |
| **Handling outliers** | Separate cluster | Less shrinkage | Flexible clustering |
| **Prior commitment** | Must choose K | Assumes single population | No commitment to K |
| **Falsifiability** | Test K=1 vs K=2 | Test nu>30 | Test K_eff=1 |

### Decision Tree for Model Selection

```
START: What do we believe about the data structure?

1. Do we think there are discrete subpopulations?
   YES → Go to 2
   NO → Go to 3

2. Do we know how many subpopulations?
   YES, K=2 → Use Finite Mixture (Model 1)
   NO → Use DP Mixture (Model 3)

3. Do we think outliers are legitimate draws from population?
   YES, from heavy tails → Use Robust Student-t (Model 2)
   NO, continuous variation → Use Standard Hierarchical Beta-Binomial (baseline)

4. Check all models with LOO:
   - If LOO clearly prefers one, use it
   - If LOO is similar, report all and explain differences
   - If all LOO worse than standard hierarchical, use standard hierarchical
```

### Implementation Complexity Assessment

**Easiest to Hardest**:
1. **Robust Student-t**: Direct extension of standard hierarchical; well-behaved MCMC
2. **Finite Mixture (K=2)**: Requires ordering constraint; some label switching risk
3. **DP Mixture**: High computational cost; requires careful tuning; post-processing needed

**Recommended implementation order**:
1. Start with Robust Student-t (quick, provides baseline for heavy-tails perspective)
2. Implement Finite Mixture if nu < 10 suggests distinct groups
3. Implement DP only if finite mixture results ambiguous or K uncertain

### Red Flags Across All Models

**Computational red flags**:
- Divergent transitions > 1% (even after tuning)
- Rhat > 1.01 on key parameters
- ESS < 400 per chain (< 100 is critical)
- Extreme autocorrelation in traces

**Statistical red flags**:
- Prior-posterior conflict (posterior very different from prior without good reason)
- Posterior equals prior (non-identifiability)
- Extreme parameter values (e.g., w=0.99, nu=1.5, K_eff > 8 for n=12 groups)
- Poor posterior predictive checks (model can't reproduce key features)

**Scientific red flags**:
- Model conclusions don't make domain sense
- Clusters/parameters not actionable or interpretable
- Results highly sensitive to arbitrary prior choices
- LOO worse than simpler baseline

---

## Falsification Mindset: Global Stopping Rules

### When to Abandon ALL Alternative Models

**I will revert to standard hierarchical beta-binomial if**:

1. **All three alternatives fail falsification criteria**
   - Each model rejected based on its specific criteria
   - Action: Standard hierarchical model is most appropriate

2. **All alternatives show worse LOO than baseline**
   - None provide predictive improvement
   - Action: Extra complexity not justified

3. **Computational difficulties across all models**
   - All models have divergences, poor convergence
   - Interpretation: Data may not support these complex structures
   - Action: Use simpler model that converges well

4. **Prior predictive checks reveal misspecification**
   - None of the priors generate data resembling observations
   - Action: Rethink fundamental assumptions

### When to Pivot Entirely (New Model Class)

**I will consider completely different approaches if**:

1. **Temporal or spatial structure discovered**
   - If groups have ordering that matters
   - Action: Consider GP, time-series, or spatial models

2. **Covariates become available**
   - If group-level predictors emerge
   - Action: Regression-based hierarchical model

3. **Data collection issues revealed**
   - If binomial assumption violated (non-independence within groups)
   - Action: Consider beta-binomial likelihood or more complex correlation structures

4. **Extreme parameter estimates suggest wrong likelihood**
   - If models consistently struggle
   - Action: Consider zero-inflated, hurdle, or other likelihood families

---

## Prior Predictive Check Protocol

### For All Models (Before Fitting)

1. **Sample from prior**: Draw 10,000 parameter sets
2. **Simulate data**: For each, generate dataset with n=12 groups, n_i and r_i
3. **Check coverage**:
   - Do simulated proportions span 0-20%?
   - Are most in 5-10% range (observed pooled mean)?
   - Are extreme groups (>15%) rare but possible?
4. **Check key statistics**:
   - Simulated ICC: Should span 0.3-0.9
   - Simulated overdispersion: Should include φ=3-5
   - Simulated outlier rate: 0-30% of groups >2 SD from mean
5. **Visualize**:
   - Overlay 100 prior predictive datasets on actual data
   - If prior generates implausible datasets, adjust priors

**Failure modes**:
- Prior too informative: All simulated data looks similar
- Prior too diffuse: Simulated data includes absurd values (50% event rates)
- Prior inconsistent: Simulated statistics don't match observed ranges

---

## Posterior Predictive Check Protocol

### For All Models (After Fitting)

1. **Generate posterior predictive datasets**: 1,000 draws from posterior predictive
2. **Check distributional features**:
   - Do PPD datasets reproduce observed histogram shape?
   - Do PPD ICC values include observed ICC=0.66?
   - Do PPD outlier counts include observed 3 outliers?
3. **Check group-specific features**:
   - For each group, does observed r_i fall within 95% predictive interval?
   - Are calibration errors uniform across groups?
4. **Check aggregate features**:
   - Does PPD reproduce total events (208 out of 2,814)?
   - Does PPD reproduce pooled proportion (7.39%)?
5. **Visualize**:
   - Plot observed vs. PPD quantiles (calibration plot)
   - Overlay observed data on PPD samples

**Failure modes**:
- Model underfits: PPD is less variable than observed data
- Model overfits: PPD is more variable than observed data
- Systematic bias: Observed statistics consistently in tails of PPD
- Poor calibration: Many groups outside 95% predictive intervals

---

## Model Comparison Strategy

### Step 1: Individual Model Assessment
- Fit each model independently
- Check convergence (Rhat, ESS, divergences)
- Check posterior predictive fit
- Apply model-specific falsification criteria
- Decision: ACCEPT, REVISE, or REJECT each model

### Step 2: Accepted Models Comparison
For all ACCEPTED models:
1. **LOO Cross-Validation**:
   - Compute ELPD_LOO for each model
   - Compare with LOO-IC differences and standard errors
   - If ΔLOO < 2*SE, models are indistinguishable
2. **Calibration**:
   - Compute LOO-PIT (probability integral transform)
   - Check uniformity (Kolmogorov-Smirnov test)
3. **Absolute Performance**:
   - RMSE for group-level proportions
   - Coverage of 95% posterior intervals
4. **Interpretability**:
   - Which model provides most actionable insights?
   - Which model aligns with domain knowledge?

### Step 3: Model Selection or Averaging
- **If one model clearly best** (ΔLOO > 4): Report this model
- **If 2+ models similar** (ΔLOO < 2*SE):
  - Report all
  - Consider Bayesian model averaging
  - Explain what each model reveals
- **If all models poor**: Return to model design phase

---

## Scientific Plausibility Checks

### Before Accepting Any Model

**Ask**:
1. **Do parameter estimates make domain sense?**
   - Is population mean ~7% plausible?
   - Are between-group differences explainable?
   - Are outlier groups systematically different in ways we can verify?

2. **Are clusters/components interpretable?**
   - For mixture models: Do components correspond to known subgroups?
   - For robust models: Does tail weight suggest genuine heavy-tailed process?
   - For DP: Do discovered clusters have scientific meaning?

3. **Are predictions reasonable?**
   - For a new group with n=100, what's predicted r?
   - Are prediction intervals too wide/narrow?
   - Do predictions match expert intuition?

4. **Can we validate?**
   - If we split data, do results replicate?
   - If we add new groups, does model prediction hold?
   - Do conclusions hold under jackknife resampling?

---

## Final Recommendations

### Pragmatic Implementation Plan

**Phase 1: Fit Robust Student-t Model (Model 2)**
- Rationale: Lowest complexity, fastest to fit, addresses outlier concern
- If nu > 30: Revert to standard hierarchical
- If nu < 10: Proceed to Phase 2

**Phase 2: Fit Finite Mixture (Model 1)**
- Rationale: Evidence of bimodality, interpretable clusters
- Compare to Robust model via LOO
- If mixture clearly better: Report mixture model
- If similar: Report both perspectives

**Phase 3: Consider DP Mixture (Model 3) Only If:**
- Uncertain whether K=1, 2, or 3 clusters
- Computational resources available
- Scientific interest in flexible clustering

**Phase 4: Compare to Standard Hierarchical Baseline**
- Always compare to standard beta-binomial or normal random effects
- Only report complex model if meaningfully better (ΔLOO > 4)

### Deliverable Structure

For each model that passes Phase 1-2:
1. **Model specification document** (mathematical + code)
2. **Prior predictive check report**
3. **Posterior summary** (convergence, parameter estimates)
4. **Posterior predictive check report**
5. **Falsification assessment** (did model pass/fail criteria?)
6. **LOO comparison** (if multiple models accepted)
7. **Scientific interpretation** (what does this model tell us?)

---

## Conclusion

These three alternative models offer genuinely different perspectives:
- **Finite Mixture**: Discrete subpopulations with distinct risk profiles
- **Robust Student-t**: Heavy-tailed continuous distribution with downweighted outliers
- **DP Mixture**: Nonparametric discovery of latent cluster structure

Each model can be falsified based on clear criteria, and each has specific conditions under which it should be preferred over standard hierarchical approaches. The implementation plan prioritizes models by computational feasibility and scientific interpretability.

**Key principle**: These are alternatives to explore, not a predetermined plan to complete. If all fail falsification, reverting to standard hierarchical beta-binomial is the correct scientific outcome.

---

**Document prepared by**: Model Designer 2
**Date**: 2025-10-30
**Next step**: Implement models sequentially, starting with Robust Student-t (Model 2) as lowest-complexity alternative
