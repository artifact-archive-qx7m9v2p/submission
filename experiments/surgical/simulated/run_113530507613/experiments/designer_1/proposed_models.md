# Hierarchical Bayesian Model Proposals
**Designer:** Model Designer 1 (Hierarchical/Multilevel Specialist)
**Date:** 2025-10-30
**Context:** 12 groups with binomial data, strong heterogeneity (ICC=0.42), two outliers, three clusters

---

## Executive Summary

Based on EDA findings showing strong heterogeneity (variance ratio = 2.78, χ² p < 0.001), I propose **three hierarchical Bayesian models** that differ fundamentally in their assumptions about the error structure and robustness properties:

1. **Standard Hierarchical Logit-Normal** (baseline, most common)
2. **Robust Student-t Hierarchy** (handles outliers via heavy tails)
3. **Hierarchical Beta-Binomial** (overdispersion via conjugate structure)

**Critical principle:** Each model makes different assumptions about how groups vary. I will abandon models based on falsification criteria, not just goodness of fit.

---

## Model 1: Standard Hierarchical Logit-Normal

### 1.1 Model Name & Description
**Standard Hierarchical Logistic Regression with Non-Centered Parameterization**

This is the canonical model for grouped binomial data with partial pooling. Groups share a common population distribution of success rates on the logit scale, with normally distributed random effects.

### 1.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)           for j = 1, ..., 12
p_j = logit^(-1)(theta_j)
```

**Hierarchical structure (non-centered parameterization):**
```
theta_j = mu + tau * theta_raw_j
theta_raw_j ~ Normal(0, 1)
```

**Hyperpriors:**
```
mu ~ Normal(-2.6, 1.0)             # population mean (logit scale)
tau ~ Normal(0, 0.5)               # between-group SD (half-normal via truncation)
```

**Equivalent centered parameterization (for comparison):**
```
theta_j ~ Normal(mu, tau)
```

**Full generative model:**
1. Draw population mean: mu ~ Normal(-2.6, 1.0)
2. Draw between-group SD: tau ~ Half-Normal(0, 0.5)
3. For each group j:
   - Draw raw effect: theta_raw_j ~ Normal(0, 1)
   - Compute group effect: theta_j = mu + tau * theta_raw_j
   - Compute success probability: p_j = logit^(-1)(theta_j)
   - Generate data: r_j ~ Binomial(n_j, p_j)

### 1.3 Stan Implementation

```stan
data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=1> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
}

parameters {
  real mu;                           // population mean (logit scale)
  real<lower=0> tau;                 // between-group SD
  vector[J] theta_raw;               // raw group effects (non-centered)
}

transformed parameters {
  vector[J] theta;                   // group effects (logit scale)
  vector[J] p;                       // success probabilities

  theta = mu + tau * theta_raw;     // non-centered parameterization
  p = inv_logit(theta);
}

model {
  // Hyperpriors
  mu ~ normal(-2.6, 1.0);
  tau ~ normal(0, 0.5);              // half-normal via <lower=0> constraint

  // Group effects
  theta_raw ~ std_normal();          // standard normal

  // Likelihood
  r ~ binomial_logit(n, theta);      // more numerically stable
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior predictive samples
  array[J] int<lower=0> r_rep;

  // Derived quantities
  real pooled_p = inv_logit(mu);
  real tau_probability = inv_logit(mu + tau) - inv_logit(mu - tau);

  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }
}
```

**PyMC Implementation Sketch:**
```python
import pymc as pm
import numpy as np

with pm.Model() as hierarchical_logit:
    # Data
    n = pm.Data('n', n_trials)
    r = pm.Data('r', r_successes)
    J = len(n_trials)

    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.6, sigma=1.0)
    tau = pm.HalfNormal('tau', sigma=0.5)

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Likelihood
    likelihood = pm.Binomial('r_obs', n=n, p=p, observed=r)

    # Posterior predictive
    r_rep = pm.Binomial('r_rep', n=n, p=p, shape=J)
```

### 1.4 Theoretical Justification

**Why this model is appropriate:**
1. **Partial pooling:** Naturally balances group-specific information with population information
2. **Logit link:** Unbounded parameter space on real line, bounded probabilities [0,1]
3. **Normal hierarchy:** Maximum entropy distribution given mean and variance constraints
4. **Non-centered parameterization:** Resolves funnel geometry when tau is small (expected here: tau ~ 0.02-0.05)

**Patterns it captures:**
- Between-group heterogeneity (via tau)
- Differential shrinkage based on sample size (small n_j → more shrinkage)
- Population-level structure (via mu)
- EDA finding: ICC = 0.42 suggests substantial between-group variance

**Link to EDA findings:**
- Addresses heterogeneity (χ² p < 0.001, variance ratio = 2.78)
- Implements partial pooling (strongly recommended in EDA)
- Prior centers on observed region (mu ~ -2.6 ≈ logit(0.07))
- Handles sample size imbalance (n ranges 47-810)

### 1.5 Falsification Criteria

**I will abandon this model if:**

1. **Posterior-predictive failure:**
   - If p-values from χ² test statistic < 0.05 or > 0.95
   - If more than 20% of groups fall outside 95% posterior predictive intervals
   - If model systematically under/over-predicts outliers (Groups 4, 8)

2. **Normality assumption violated:**
   - If posterior distributions of theta_j are strongly skewed (|skewness| > 1.5)
   - If Q-Q plot of theta_j posteriors shows severe departure from normality
   - If two outliers (Groups 4, 8) have posterior probabilities < 0.01 under the model

3. **Prior-posterior conflict:**
   - If posterior tau is pinned at boundary (suggests misspecification)
   - If posterior mu shifts > 2 SD from prior mean (suggests prior-data conflict)
   - If effective sample size for mu or tau < 100 (suggests non-identifiability)

4. **Computational pathologies:**
   - If > 1% divergent transitions (after reparameterization attempts)
   - If Rhat > 1.01 for any parameter
   - If ESS/iteration < 0.001 (indicates severe inefficiency)

5. **Predictive inadequacy:**
   - If LOO-CV Pareto k > 0.7 for > 2 groups (influential observations)
   - If ELPD significantly worse than simpler/alternative models (ΔELPD > 5)

**Specific predictions this model makes:**
- Posterior theta_j will be approximately Normal(mu, tau)
- Shrinkage will be inversely proportional to sqrt(n_j)
- Group 4 (n=810) should shrink ~27%, Group 1 (n=47) should shrink ~75%
- Posterior predictive p-value for χ² should be between 0.05-0.95
- Between-group SD tau should be in range [0.01, 0.10] on logit scale

**Expected posterior patterns:**
- Unimodal, symmetric posteriors for all parameters
- mu: posterior mean ~ -2.7 to -2.5, SD ~ 0.3
- tau: posterior mean ~ 0.02-0.05, right-skewed
- theta_j: will shrink toward mu, with varying amounts based on n_j
- Groups 4 and 8 will have non-overlapping 95% credible intervals

### 1.6 Computational Considerations

**Expected challenges:**

1. **Funnel geometry:**
   - When tau → 0, posterior becomes funnel-shaped (narrow at top, wide at bottom)
   - Non-centered parameterization mitigates this
   - May still see divergences if tau posterior has substantial mass near 0

2. **Small number of groups (J=12):**
   - Hyperparameter (tau) uncertainty will be substantial
   - May see slower mixing for tau
   - ESS for tau may be lower than for theta_j

3. **Extreme sample size imbalance:**
   - Group 4 (n=810) dominates: 29% of total data
   - May pull population mean toward its value
   - Could create bimodal exploration if Group 4 is truly different

**Mitigation strategies:**

1. **Use non-centered parameterization** (already implemented)
2. **Increase adapt_delta to 0.95** if divergences appear (default 0.8)
3. **Increase max_treedepth to 12** if hitting maximum (default 10)
4. **Run longer chains:** 4 chains × 2000 iterations (1000 warmup, 1000 sampling)
5. **Initialize near data:** Start mu near log(pooled_rate), tau near 0.05
6. **Monitor Rhat, ESS, divergences** after every run

**Approximate runtime:**
- Stan: ~30-60 seconds for 4000 post-warmup samples (4 chains × 1000)
- PyMC: ~1-2 minutes with NUTS sampler (more complex gradient computations)
- Bottleneck: Evaluating binomial_logit likelihood 12J times per iteration

**Expected diagnostics:**
- Rhat: should be < 1.01 for all parameters
- ESS (bulk): > 400 per chain for theta_j, > 200 for tau
- ESS (tail): > 400 per chain (important for credible intervals)
- Divergences: 0 with non-centered parameterization (target < 10 if any)
- Energy: E-BFMI > 0.3 (indicates good Hamiltonian exploration)

---

## Model 2: Robust Student-t Hierarchy

### 2.1 Model Name & Description
**Hierarchical Logistic Regression with Student-t Group Effects**

This model replaces the Normal distribution for group effects with a Student-t distribution, which has heavier tails. This provides robustness to outliers (Groups 4, 8) while maintaining partial pooling. If outliers are truly extreme, they will be downweighted rather than pulling the population mean.

### 2.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)
p_j = logit^(-1)(theta_j)
```

**Hierarchical structure:**
```
theta_j ~ StudentT(nu, mu, tau)    # heavy-tailed hierarchy
```

**Hyperpriors:**
```
mu ~ Normal(-2.6, 1.0)             # population mean
tau ~ Normal(0, 0.5)               # scale parameter (half-normal)
nu ~ Gamma(2, 0.1)                 # degrees of freedom (higher = more robust)
```

**Parameterization details:**
- nu < 3: very heavy tails (extreme outlier robustness)
- nu = 3-10: moderate robustness
- nu > 30: approximately Normal
- EDA suggests nu ~ 5-10 (two outliers out of 12)

**Full generative model:**
1. Draw population mean: mu ~ Normal(-2.6, 1.0)
2. Draw scale: tau ~ Half-Normal(0, 0.5)
3. Draw degrees of freedom: nu ~ Gamma(2, 0.1)
4. For each group j:
   - Draw group effect: theta_j ~ StudentT(nu, mu, tau)
   - Compute success probability: p_j = logit^(-1)(theta_j)
   - Generate data: r_j ~ Binomial(n_j, p_j)

### 2.3 Stan Implementation

```stan
data {
  int<lower=1> J;
  array[J] int<lower=1> n;
  array[J] int<lower=0> r;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=1> nu;                 // degrees of freedom
  vector[J] theta;                  // centered parameterization (more natural for Student-t)
}

transformed parameters {
  vector[J] p = inv_logit(theta);
}

model {
  // Hyperpriors
  mu ~ normal(-2.6, 1.0);
  tau ~ normal(0, 0.5);
  nu ~ gamma(2, 0.1);               // mean=20, SD=14 (weakly informative)

  // Group effects (Student-t hierarchy)
  theta ~ student_t(nu, mu, tau);

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] log_lik;
  array[J] int<lower=0> r_rep;

  // Diagnostics
  real pooled_p = inv_logit(mu);
  real tail_weight = 1.0 / nu;      // higher = heavier tails

  // Outlier detection: probability under Normal vs Student-t
  vector[J] z_score = (theta - mu) / tau;
  vector[J] normal_density;
  vector[J] t_density;

  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);

    normal_density[j] = exp(normal_lpdf(z_score[j] | 0, 1));
    t_density[j] = exp(student_t_lpdf(z_score[j] | nu, 0, 1));
  }
}
```

**PyMC Implementation Sketch:**
```python
with pm.Model() as robust_hierarchy:
    # Data
    n = pm.Data('n', n_trials)
    r = pm.Data('r', r_successes)
    J = len(n_trials)

    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.6, sigma=1.0)
    tau = pm.HalfNormal('tau', sigma=0.5)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)  # degrees of freedom

    # Student-t hierarchy (centered)
    theta = pm.StudentT('theta', nu=nu, mu=mu, sigma=tau, shape=J)
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Likelihood
    likelihood = pm.Binomial('r_obs', n=n, p=p, observed=r)
```

### 2.4 Theoretical Justification

**Why this model is appropriate:**
1. **Outlier robustness:** Heavy tails allow extreme observations without distorting population mean
2. **Adaptive robustness:** Data determines nu (degree of robustness needed)
3. **Nests Normal model:** As nu → ∞, converges to Model 1
4. **Downweighting mechanism:** Extreme groups (4, 8) contribute less to mu estimate

**Patterns it captures:**
- Same as Model 1, plus:
- **Differential influence:** Outliers shrink more aggressively toward mu
- **Heterogeneous precision:** Effective variance for each group differs
- **Cluster structure indirectly:** Heavy tails can accommodate multiple modes

**Link to EDA findings:**
- Directly addresses outliers (Groups 4, 8 with z-scores > 3)
- EDA shows 25% of groups outside 95% CI (suggests need for robust model)
- Two extreme outliers with large n → cannot be dismissed as noise
- Variance ratio = 2.78 could be inflated by outliers

### 2.5 Falsification Criteria

**I will abandon this model if:**

1. **No evidence for robustness:**
   - If posterior nu > 30 (effectively Normal) with 95% CI entirely above 20
   - If LOO-CV shows no improvement over Model 1 (ΔELPD < 1)
   - If outliers (Groups 4, 8) have similar posterior distributions to Model 1

2. **Over-robustness:**
   - If posterior nu < 3 (too heavy-tailed, suggests wrong likelihood)
   - If all groups shrink equally regardless of sample size
   - If posterior predictive checks show worse fit than Normal model

3. **Computational failure:**
   - If posterior for nu is non-identifiable (uniform/bimodal)
   - If centered parameterization causes divergences (try non-centered for Student-t)
   - If Rhat > 1.01 or ESS < 200 for nu

4. **Mechanistic implausibility:**
   - If domain knowledge suggests outliers are real population differences (not errors)
   - If cross-validation shows overfitting (worse out-of-sample prediction)

**Specific predictions this model makes:**
- Posterior nu will be between 3-20 (moderate robustness)
- Groups 4 and 8 will have larger posterior uncertainty than in Model 1
- Population mean mu will be closer to median than mean of observed rates
- Shrinkage for outliers will be > 1.5× that of Model 1
- Predictive intervals will be wider for extreme groups

**Expected posterior patterns:**
- nu posterior: Right-skewed, mode ~ 5-10
- mu posterior: Less influenced by Groups 4 and 8 than Model 1
- theta_j for outliers: More shrinkage, wider credible intervals
- Comparison metric: Smaller LOOIC than Model 1 by 2-5 points

### 2.6 Computational Considerations

**Expected challenges:**

1. **Non-identifiability of nu:**
   - With J=12, difficult to precisely estimate tail behavior
   - nu posterior will have high uncertainty
   - May need informative prior to stabilize

2. **Geometry issues:**
   - Centered parameterization may cause divergences
   - Student-t doesn't have convenient non-centered form
   - May need to increase adapt_delta to 0.98

3. **Slower mixing:**
   - Student-t likelihood has more complex gradients
   - May need longer chains (2000+ post-warmup samples)

**Mitigation strategies:**
1. **Informative prior on nu:** Gamma(2, 0.1) centers mass at reasonable values
2. **Increase adapt_delta to 0.98** (more conservative step sizes)
3. **Monitor nu separately:** Check ESS, Rhat, trace plots
4. **Initialize nu = 10:** Start in reasonable region
5. **Consider alternative:** Laplace hierarchy if Student-t fails (double exponential)

**Approximate runtime:**
- Stan: ~60-120 seconds (2× Model 1 due to Student-t complexity)
- PyMC: ~2-4 minutes
- Bottleneck: Evaluating Student-t gradients

**Expected diagnostics:**
- Rhat < 1.01 for all parameters (nu may be slower)
- ESS for nu: target > 200 (acceptable given difficulty)
- Divergences: < 1% with adapt_delta = 0.98
- Trace plots for nu: should show stationary exploration

---

## Model 3: Hierarchical Beta-Binomial

### 3.1 Model Name & Description
**Hierarchical Beta-Binomial with Logit-Normal Hyperpriors**

This model explicitly models overdispersion via a Beta-Binomial likelihood rather than via logit-normal random effects. Each group has a Beta-distributed success probability, with hierarchical structure on Beta parameters. This is a fundamentally different approach: overdispersion is in the likelihood, not the random effects.

### 3.2 Mathematical Specification

**Likelihood:**
```
r_j ~ BetaBinomial(n_j, alpha_j, beta_j)
```

**Hierarchical structure (via mean-concentration parameterization):**
```
alpha_j = mu_p * phi
beta_j = (1 - mu_p) * phi

where:
  mu_p = inv_logit(mu)              # population mean probability
  phi ~ Gamma(2, 0.01)              # concentration (inverse overdispersion)
```

**Hyperpriors:**
```
mu ~ Normal(-2.6, 1.0)              # population mean (logit scale)
phi ~ Gamma(2, 0.01)                # concentration parameter
```

**Alternative parameterization (conjugate structure):**
```
p_j ~ Beta(alpha_0, beta_0)         # group-specific probabilities
r_j ~ Binomial(n_j, p_j)

with:
  alpha_0 = mu_p * phi
  beta_0 = (1 - mu_p) * phi
```

**Full generative model:**
1. Draw population mean: mu ~ Normal(-2.6, 1.0)
2. Compute population probability: mu_p = inv_logit(mu)
3. Draw concentration: phi ~ Gamma(2, 0.01)
4. For each group j:
   - Draw success probability: p_j ~ Beta(mu_p * phi, (1-mu_p) * phi)
   - Generate data: r_j ~ Binomial(n_j, p_j)

**Relationship between phi and tau:**
```
Var(logit(p_j)) ≈ 1 / (mu_p * (1 - mu_p) * phi)
tau^2 ≈ 1 / (mu_p * (1 - mu_p) * phi)

For mu_p = 0.07:
  tau = 0.05 → phi ≈ 3000
  tau = 0.02 → phi ≈ 20000
```

### 3.3 Stan Implementation

```stan
data {
  int<lower=1> J;
  array[J] int<lower=1> n;
  array[J] int<lower=0> r;
}

parameters {
  real mu;                          // population mean (logit scale)
  real<lower=0> phi;                // concentration parameter
  vector<lower=0, upper=1>[J] p;    // group success probabilities
}

transformed parameters {
  real<lower=0, upper=1> mu_p = inv_logit(mu);
  real<lower=0> alpha = mu_p * phi;
  real<lower=0> beta = (1 - mu_p) * phi;
}

model {
  // Hyperpriors
  mu ~ normal(-2.6, 1.0);
  phi ~ gamma(2, 0.01);             // mean=200, SD=141

  // Group probabilities
  p ~ beta(alpha, beta);

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  vector[J] log_lik;
  array[J] int<lower=0> r_rep;

  // Derived quantities
  real tau_approx = sqrt(1.0 / (mu_p * (1 - mu_p) * phi));
  real overdispersion = 1.0 + 1.0 / phi;  // ratio of variance to binomial

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]) +
                 beta_lpdf(p[j] | alpha, beta);  // marginal likelihood
    r_rep[j] = binomial_rng(n[j], p[j]);
  }
}
```

**Alternative: Beta-Binomial directly:**
```stan
model {
  mu ~ normal(-2.6, 1.0);
  phi ~ gamma(2, 0.01);

  // Beta-binomial likelihood (marginalizing over p_j)
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}
```

**PyMC Implementation Sketch:**
```python
with pm.Model() as beta_binomial_hierarchy:
    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.6, sigma=1.0)
    phi = pm.Gamma('phi', alpha=2, beta=0.01)

    # Population parameters
    mu_p = pm.Deterministic('mu_p', pm.math.invlogit(mu))
    alpha = pm.Deterministic('alpha', mu_p * phi)
    beta = pm.Deterministic('beta', (1 - mu_p) * phi)

    # Group probabilities
    p = pm.Beta('p', alpha=alpha, beta=beta, shape=J)

    # Likelihood
    likelihood = pm.Binomial('r_obs', n=n, p=p, observed=r)
```

### 3.4 Theoretical Justification

**Why this model is appropriate:**
1. **Natural overdispersion model:** Beta-Binomial is canonical for overdispersed binomial data
2. **Conjugate structure:** Beta prior + Binomial likelihood = Beta posterior
3. **Interpretable parameters:** phi directly measures concentration (high phi = less variability)
4. **Alternative mechanism:** Overdispersion from probability distribution, not group effects

**Patterns it captures:**
- Between-group heterogeneity via Beta distribution of p_j
- Overdispersion beyond binomial (variance ratio = 2.78)
- Partial pooling via shared alpha, beta parameters
- Same shrinkage properties as Model 1, but different parameterization

**Link to EDA findings:**
- Addresses overdispersion (variance ratio = 2.78)
- ICC = 0.42 corresponds to phi ~ 100-500
- Implements partial pooling (EDA recommendation)
- Beta distribution can be skewed (matches observed asymmetry)

**Key difference from Models 1-2:**
- Models 1-2: Overdispersion via random effects on logit scale
- Model 3: Overdispersion via Beta distribution on probability scale
- Both produce similar marginal distributions, but different conditional structure

### 3.5 Falsification Criteria

**I will abandon this model if:**

1. **Posterior predictive failure:**
   - If χ² test statistic p-value < 0.05 or > 0.95 (same as Model 1)
   - If model fits worse than Model 1 on held-out data (LOO-CV)
   - If posterior predictive distribution too narrow (doesn't capture heterogeneity)

2. **Mechanistic implausibility:**
   - If domain knowledge suggests logit-normal is more natural (e.g., linear predictors)
   - If we want to add covariates (Beta-Binomial harder to extend)
   - If interpretation requires logit scale (e.g., odds ratios)

3. **Parameter boundary issues:**
   - If posterior for phi is pinned at 0 (infinite overdispersion, wrong model)
   - If posterior for p_j hits boundaries 0 or 1 (suggests zero-inflation)
   - If posterior for phi > 10000 (effectively no overdispersion, use binomial)

4. **Computational difficulties:**
   - If sampling p_j causes divergences (31-parameter space)
   - If correlation between p_j and hyperparameters causes slow mixing
   - If ESS < 200 for any p_j

5. **No advantage over simpler models:**
   - If LOOIC within 2 points of Model 1 (not worth extra complexity)
   - If posterior tau_approx ≈ posterior tau from Model 1 (equivalent models)

**Specific predictions this model makes:**
- Posterior phi will be between 50-500
- Implied tau_approx will be ~ 0.02-0.05 (matching EDA)
- p_j posteriors will be constrained to [0, 1] (no logit transformation needed)
- Overdispersion parameter will be ~ 1.002-1.02 (variance 0.2-2% larger than binomial)

**Expected posterior patterns:**
- mu: similar to Models 1-2
- phi: right-skewed, mode ~ 100-300
- p_j: Beta-distributed around mu_p, with spread determined by phi
- Comparison: Should give similar point estimates to Model 1, but different uncertainty structure

### 3.6 Computational Considerations

**Expected challenges:**

1. **High-dimensional parameter space:**
   - 12 group probabilities p_j + 2 hyperparameters = 14 parameters
   - May have posterior correlations between p_j and phi
   - Geometry may be complex (Beta constraints)

2. **Boundary issues:**
   - p_j constrained to [0, 1]
   - If data extreme (r=0 or r=n), posterior may hit boundary
   - May need to add small constant (e.g., r+0.5, n+1) for stability

3. **Non-centered reparameterization less obvious:**
   - Beta distribution doesn't have simple non-centered form
   - May need manual transformation if divergences occur

**Mitigation strategies:**
1. **Use beta_binomial directly:** Marginalizes over p_j, reduces dimensions to 2 parameters
2. **Informative prior on phi:** Gamma(2, 0.01) prevents extreme values
3. **Monitor p_j separately:** Check for boundary issues, ESS
4. **Initialize phi ~ 200:** Start in reasonable region based on EDA
5. **Consider alternative parameterization:** Logit-normal on p_j (hybrid with Model 1)

**Approximate runtime:**
- Stan (with p_j): ~60-90 seconds (14 parameters)
- Stan (beta_binomial marginalized): ~20-40 seconds (2 parameters, much faster!)
- PyMC: ~1-2 minutes
- Recommendation: Use marginalized version for efficiency

**Expected diagnostics:**
- Rhat < 1.01 for all parameters
- ESS > 400 for mu, > 200 for phi
- ESS > 400 for p_j (if sampled explicitly)
- Divergences: should be 0 (simpler geometry than Models 1-2)
- Trace plots: should show good mixing for all parameters

**Key advantage:**
- **Marginalized version is fastest and simplest!** Only 2 parameters instead of 14.
- If computational efficiency matters and model fits well, this is preferred.

---

## Comparative Analysis

### Model Comparison Framework

| Aspect | Model 1: Logit-Normal | Model 2: Student-t | Model 3: Beta-Binomial |
|--------|----------------------|-------------------|----------------------|
| **Complexity** | Moderate (13 params) | High (14 params) | Low (2 params marginalized) |
| **Robustness** | Moderate (Normal tails) | High (heavy tails) | Moderate (Beta tails) |
| **Interpretability** | High (standard) | Moderate (nu parameter) | High (conjugate) |
| **Computational** | Fast (~30-60s) | Slow (~60-120s) | Very fast (~20-40s) |
| **Extensibility** | Excellent (add covariates) | Good | Limited (Beta constraints) |
| **Theoretical** | Maximum entropy | Robust statistics | Conjugate Bayesian |

### Decision Rules

**Choose Model 1 (Logit-Normal) if:**
- No strong evidence of outliers beyond Normal tails
- Want to add covariates later (e.g., group characteristics)
- Interpretation on logit/odds-ratio scale is natural
- Standard model for field/publication

**Choose Model 2 (Student-t) if:**
- Outliers (Groups 4, 8) distort Normal model
- LOO-CV shows improvement over Model 1 (ΔELPD > 2)
- Domain knowledge suggests some groups are anomalous
- Robustness is priority over interpretation

**Choose Model 3 (Beta-Binomial) if:**
- Want simplest, fastest model (marginalized version)
- Interpretation on probability scale is natural
- No covariates planned
- Overdispersion is primary concern, not outlier detection

### Red Flags for Model Class Switch

**Abandon ALL hierarchical models if:**
1. **Posterior predictive p-value < 0.01:** Fundamental model misspecification
2. **Cluster structure dominates:** If mixture model fits far better (ΔELPD > 10)
3. **No partial pooling:** If posterior tau → 0 and all theta_j identical (use complete pooling)
4. **Complete separation:** If all groups identical after shrinkage (use no pooling)
5. **Non-exchangeability:** If groups have known structure (e.g., temporal ordering)

**Switch to mixture model if:**
- LOO-CV shows 3-component mixture far superior (ΔELPD > 10)
- Posterior predictive checks show systematic misfit
- Visual inspection shows clear 3-cluster structure not captured by single hierarchy

**Switch to GP/time-series if:**
- Sequential patterns emerge (group_id has meaning)
- Autocorrelation detected in residuals
- Temporal covariate available

---

## Stress Tests

Each model will be subjected to:

1. **Prior sensitivity analysis:**
   - Run with vague priors: mu ~ Normal(0, 10), tau ~ Half-Cauchy(0, 5)
   - Run with informative priors: mu ~ Normal(-2.6, 0.5), tau ~ Half-Normal(0, 0.02)
   - Compare posteriors: if ΔELPD > 5, prior dominates (problem!)

2. **Leave-one-out robustness:**
   - Refit excluding Group 4 (largest, low rate)
   - Refit excluding Group 8 (large, high rate)
   - If mu changes by > 0.5 logit units, model too sensitive

3. **Posterior predictive checks:**
   - χ² test statistic (heterogeneity)
   - Maximum success rate (captures Group 8?)
   - Minimum success rate (captures Group 4?)
   - Variance of success rates (captures ICC?)

4. **Cross-validation:**
   - LOO-CV with Pareto k diagnostic
   - If k > 0.7 for any group, influential observation (investigate)
   - Compare ELPD across models

5. **Simulation-based calibration (SBC):**
   - Generate synthetic data from prior predictive
   - Fit model to synthetic data
   - Check if posteriors cover true parameters at nominal rates
   - If not, model misspecified or computational issues

---

## Implementation Plan

### Phase 1: Prior Predictive Checks (Day 1)
- Sample from prior predictive distribution for each model
- Verify priors generate plausible datasets
- Check that observed data is not extreme under prior

### Phase 2: Model Fitting (Day 1-2)
- Fit all three models to full dataset
- Monitor diagnostics (Rhat, ESS, divergences)
- Generate posterior samples

### Phase 3: Posterior Checking (Day 2)
- Posterior predictive checks (graphical and numerical)
- LOO-CV comparison
- Parameter interpretation

### Phase 4: Sensitivity Analysis (Day 2-3)
- Prior sensitivity (3 prior specifications × 3 models = 9 runs)
- Outlier sensitivity (leave-one-out)
- Robustness checks

### Phase 5: Model Selection (Day 3)
- Compare LOOIC across all models
- Weight by predictive performance
- Identify best model(s) for inference

### Phase 6: SBC Validation (Day 3-4)
- Generate 100-200 synthetic datasets from prior predictive
- Fit model to each, check coverage
- Validate computational pipeline

---

## Expected Outcomes

### Scenario 1: Models converge (most likely)
- All three models give similar point estimates (within 10%)
- LOOIC within 2-3 points of each other
- Conclusion: Heterogeneity well-characterized, choose simplest (Model 3 marginalized)

### Scenario 2: Student-t wins (if outliers dominate)
- Model 2 LOOIC 5+ points better than Model 1
- Posterior nu < 10 (moderate heavy tails)
- Conclusion: Outliers require robust model, use Model 2

### Scenario 3: All models fail (fundamental misspecification)
- Posterior predictive p-values extreme (< 0.01)
- LOO-CV shows poor predictions (many k > 0.7)
- Conclusion: Need different model class (mixture, GP, covariate adjustment)

### Scenario 4: Cluster structure dominates
- Visual inspection shows clear 3 clusters
- Single hierarchy doesn't capture structure
- Conclusion: Abandon hierarchical models, propose mixture model

---

## Falsification Summary

**I will abandon hierarchical models entirely if:**
1. All three models show posterior predictive p-value < 0.05
2. LOOIC for all models is worse than no-pooling by > 10 points
3. Visual inspection + domain knowledge suggests fixed clusters, not continuous hierarchy
4. Computational issues persist across all models (suggests fundamental problem)

**I will switch to mixture models if:**
1. Posterior predictive checks consistently fail for heterogeneity
2. EDA cluster structure (K=3) is not captured by single hierarchy
3. Outliers cannot be accommodated even with Student-t

**I will switch to simpler models if:**
1. Posterior tau → 0 with high precision (no between-group variance)
2. All three hierarchical models give identical LOOIC to complete pooling
3. Computational cost not justified by predictive improvement

---

## Conclusion

I propose three hierarchical Bayesian models that make different assumptions about how groups vary:

1. **Logit-Normal** (standard, interpretable, extensible)
2. **Student-t** (robust, handles outliers, more complex)
3. **Beta-Binomial** (simple, fast, conjugate)

Each has clear falsification criteria. I expect Models 1 and 3 to perform similarly, with Model 2 potentially superior if outliers dominate. If all fail, I will pivot to mixture models or other model classes.

**Key principle:** I seek truth, not completion. If evidence suggests abandoning all three models, I will do so immediately and propose alternatives.

---

**Files to generate in implementation:**
- `/workspace/experiments/designer_1/model_1_logit_normal.stan`
- `/workspace/experiments/designer_1/model_2_student_t.stan`
- `/workspace/experiments/designer_1/model_3_beta_binomial.stan`
- `/workspace/experiments/designer_1/prior_predictive_checks.py`
- `/workspace/experiments/designer_1/model_fitting.py`
- `/workspace/experiments/designer_1/model_comparison.py`

**End of Model Proposal Document**
