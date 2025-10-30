# Bayesian Model Design: Beta-Binomial and Mixture Approaches
## Designer 1 - Model Specifications

**Analysis Date:** 2025-10-30
**Focus:** Beta-binomial models and mixture alternatives
**Dataset:** 12 groups, severe overdispersion (φ=3.5-5.1), ICC=0.73

---

## Executive Summary: Competing Hypotheses

Based on EDA findings, I propose three fundamentally different model classes to explain the observed overdispersion:

1. **Model A: Homogeneous Beta-Binomial** - All groups share common hyperparameters (continuous variation)
2. **Model B: Reparameterized Beta-Binomial** - More interpretable mean-concentration parameterization
3. **Model C: Two-Component Mixture** - Discrete subpopulations (low-rate vs high-rate groups)

**Critical stance:** I expect Model A or B to win, but Model C serves as a falsification test. If Model C fits substantially better, it means the continuous variation assumption is wrong and we need to rethink the entire approach.

---

## Model Comparison Overview

| Feature | Model A | Model B | Model C |
|---------|---------|---------|---------|
| **Class** | Beta-binomial (α,β) | Beta-binomial (μ,κ) | Mixture of 2 beta-binomials |
| **Parameters** | 2 hyperparameters | 2 hyperparameters | 5 parameters |
| **Complexity** | Low | Low | Medium |
| **Interpretability** | Moderate | High | High (if justified) |
| **Handles zeros** | Yes | Yes | Yes |
| **EDA support** | Strong | Strong | Weak (no clusters) |
| **Prior sensitivity** | Moderate | Low | High |
| **Falsification risk** | If mixture fits better | If mixture fits better | If groups overlap completely |

---

## Model A: Homogeneous Beta-Binomial (Baseline)

### Mathematical Specification

```
Level 1 (Data likelihood):
  r_i | p_i, n_i ~ Binomial(n_i, p_i)    for i = 1, ..., 12

Level 2 (Group-level variation):
  p_i | α, β ~ Beta(α, β)                 [Common hyperparameters]

Hyperpriors:
  α ~ Gamma(2, 0.5)                       [Shape > 0, mean = 4]
  β ~ Gamma(2, 0.1)                       [Shape > 0, mean = 20]
```

**Implied prior for mean success rate:**
- E[p] = α/(α+β) ≈ 4/24 = 0.167 (slightly higher than observed 0.076)
- This allows data to shrink mean downward if needed

**Implied prior for overdispersion:**
- Var(p) = αβ/[(α+β)²(α+β+1)] ≈ 0.003 (moderate variation)
- Concentration κ = α+β has prior mean ≈ 24 (moderate to strong concentration)

### Rationale from EDA Findings

**Why this model might be CORRECT:**
- EDA shows continuous distribution of group rates (Shapiro-Wilk p=0.496)
- No clear clustering in caterpillar plots
- 84.8% of group pairs have overlapping confidence intervals
- Beta-binomial had best AIC (47.69) in preliminary analysis
- Handles Group 1 zero count naturally (no log-odds singularity)

**Why this model might FAIL (falsification criteria):**
1. **If posterior shows bimodality** in p_i distribution → suggests mixture instead
2. **If residuals are systematically structured** (e.g., small-n groups differ) → need covariates
3. **If posterior α and β are extreme** (α or β < 0.5 or > 100) → wrong parameterization
4. **If Groups 1 and 8 get nearly identical shrinkage** → insufficient flexibility
5. **If posterior predictive checks fail** to reproduce observed variance structure

**I will abandon this model if:**
- Mixture model (Model C) has ΔAIC > 10 or Bayes Factor > 100
- Posterior predictive p-value for variance < 0.01 or > 0.99
- Leave-one-out cross-validation shows systematic misprediction (Pareto k > 0.7 for >3 groups)

### Prior Justification

**Choice of Gamma(2, 0.5) for α:**
- Shape=2 prevents extreme values near zero
- Rate=0.5 gives mean=4, SD=2.8
- Allows range roughly [0.5, 15] (95% prior mass)
- Weakly informative: compatible with many success rates

**Choice of Gamma(2, 0.1) for β:**
- Mean=20 slightly favors low success rates (matching pooled rate 7.6%)
- SD=14 allows substantial uncertainty
- Allows range roughly [2, 60] (95% prior mass)
- Will let data dominate for mean, but provides regularization

**Why not flat priors?**
- Improper priors can lead to improper posteriors in hierarchical models
- Flat on (α,β) is NOT flat on induced mean or variance
- Weakly informative priors aid MCMC convergence
- These priors allow wide range of success rates [0.01, 0.50] with high probability

### Prior Predictive Distribution

**Expected range of predictions under prior:**
- Success rate mean: 50% prior mass on [0.05, 0.30]
- Success rate variance: 50% prior mass on [0.001, 0.010]
- Allows for zero counts: P(r=0 | n=47) ≈ 0.05-0.15 under prior
- Allows for Group 8 rate: P(r>30 | n=215) ≈ 0.01-0.10 under prior

**Prior predictive check:** Generate 1000 datasets from prior → should see:
- Mean success rates ranging [0.0, 0.5]
- Occasional zero-count groups (realistic)
- φ values spanning [1, 10] (covers observed φ=3.5)

### Stan Implementation

```stan
data {
  int<lower=1> N;                  // Number of groups
  array[N] int<lower=0> n_trials;  // Trials per group
  array[N] int<lower=0> r_success; // Successes per group
}

parameters {
  real<lower=0> alpha;             // Beta shape parameter 1
  real<lower=0> beta;              // Beta shape parameter 2
}

model {
  // Hyperpriors
  alpha ~ gamma(2, 0.5);
  beta ~ gamma(2, 0.1);

  // Likelihood (vectorized)
  r_success ~ beta_binomial(n_trials, alpha, beta);
}

generated quantities {
  // Derived quantities of interest
  real mean_p = alpha / (alpha + beta);
  real kappa = alpha + beta;
  real var_p = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1));

  // Posterior predictive for group-level rates
  array[N] real<lower=0, upper=1> p_rep;
  for (i in 1:N) {
    p_rep[i] = beta_rng(alpha, beta);
  }

  // Posterior predictive for data
  array[N] int r_rep;
  for (i in 1:N) {
    real p_i = beta_rng(alpha, beta);
    r_rep[i] = binomial_rng(n_trials[i], p_i);
  }

  // Overdispersion parameter
  real phi = 1 + var_p / (mean_p * (1 - mean_p));
}
```

### Expected Posterior Behavior if Correct

**Convergence diagnostics:**
- Rhat < 1.01 (should converge easily - only 2 parameters)
- ESS > 1000 (good mixing expected)
- No divergent transitions (model is simple)

**Posterior distributions:**
- α posterior: Expect E[α] ≈ 2-6 (low shape → right-skewed beta)
- β posterior: Expect E[β] ≈ 20-50 (high shape → concentrates near zero)
- Mean success rate: E[p] ≈ 0.06-0.09 (close to observed 0.076)
- Overdispersion: E[φ] ≈ 3.0-4.0 (matching observed φ=3.5)

**Shrinkage patterns:**
- Group 1 (0/47): Should shrink to ≈2-4% (strong shrinkage from 0%)
- Group 8 (31/215): Should shrink to ≈11-13% (moderate shrinkage from 14.4%)
- Group 4 (46/810): Minimal shrinkage (large sample)

**Posterior predictive checks:**
- Should reproduce observed variance in success rates
- Should predict zero counts occasionally (Group 1 not impossible)
- Should not systematically over/under-predict any group

### Computational Considerations

**Sampling challenges:**
- Low - this is a simple model
- Beta-binomial is conjugate-like, good geometry
- Only 2 parameters → fast sampling

**Potential issues:**
- If α or β → 0, can get extreme shrinkage (numerical instability)
- If both α,β → ∞, approaches point mass (reduced identifiability)
- Solution: Priors prevent these extremes

**Expected runtime:**
- 4 chains × 2000 iterations ≈ 30-60 seconds
- No warmup issues expected

---

## Model B: Reparameterized Beta-Binomial (Mean-Concentration)

### Mathematical Specification

```
Level 1 (Data likelihood):
  r_i | p_i, n_i ~ Binomial(n_i, p_i)

Level 2 (Group-level variation):
  p_i | μ, κ ~ Beta(μκ, (1-μ)κ)           [Reparameterization: α=μκ, β=(1-μ)κ]

Hyperpriors:
  μ ~ Beta(2, 18)                          [Mean success rate, prior mean=0.1]
  κ ~ Gamma(2, 0.1)                        [Concentration, prior mean=20]
```

**Interpretation:**
- μ ∈ [0,1]: Population mean success rate
- κ > 0: Concentration (higher κ → less between-group variation)
- When κ→∞, all groups have same rate μ (no overdispersion)
- When κ→0, groups can have any rate (extreme overdispersion)

### Rationale from EDA Findings

**Why this model might be CORRECT:**
- Same likelihood as Model A, just reparameterized
- EDA gives clear estimate of mean (7.6%) → easier to specify prior
- Concentration parameter directly interpretable (relates to ICC)
- Separation of location (μ) and scale (κ) aids interpretation

**Why this model might FAIL:**
- Same failures as Model A (mixture better, etc.)
- Additional risk: **μ and κ can be correlated** in posterior → slower mixing
- If prior on μ is too tight, might not capture true mean

**I will abandon this model if:**
- Same criteria as Model A (same model, different parameterization)
- Additionally: If posterior correlation(μ, κ) > 0.95 → identifiability issues

### Prior Justification

**Choice of Beta(2, 18) for μ:**
- Prior mean = 2/(2+18) = 0.1 (close to observed 0.076)
- Prior SD = 0.061
- 95% credible interval: [0.01, 0.25]
- Weakly informative: allows data to shift mean as needed
- Prevents extreme values near 0 or 1

**Choice of Gamma(2, 0.1) for κ:**
- Prior mean = 20
- Prior SD = 14
- Allows κ ∈ [5, 60] with high probability
- Low κ (5-10): High overdispersion, groups very different
- High κ (50-100): Low overdispersion, groups similar
- EDA suggests moderate κ needed to match φ=3.5

**Relationship to ICC:**
- ICC ≈ 1/(1+κ) for beta-binomial
- Observed ICC = 0.73 → implies κ ≈ 0.37
- **WARNING:** This is VERY low! Prior mean of 20 may be too high
- Strategy: Wide prior allows data to find true κ, but we should check sensitivity

### Prior Predictive Distribution

**Expected range:**
- Same as Model A (same model)
- But now easier to verify: μ ~ Beta(2,18) directly gives success rate distribution

**Prior predictive check:**
- Mean success rate: 50% prior mass on [0.05, 0.15]
- Between-group SD: 50% prior mass on [0.02, 0.08]
- This should cover observed mean=0.076, SD=0.041

### Stan Implementation

```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r_success;
}

parameters {
  real<lower=0, upper=1> mu;       // Mean success rate
  real<lower=0> kappa;             // Concentration parameter
}

transformed parameters {
  real<lower=0> alpha = mu * kappa;
  real<lower=0> beta = (1 - mu) * kappa;
}

model {
  // Hyperpriors
  mu ~ beta(2, 18);
  kappa ~ gamma(2, 0.1);

  // Likelihood
  r_success ~ beta_binomial(n_trials, alpha, beta);
}

generated quantities {
  real var_p = (mu * (1 - mu)) / (kappa + 1);
  real phi = 1 + 1 / kappa;  // Approximate overdispersion
  real icc = 1 / (1 + kappa); // Approximate ICC

  // Posterior predictive
  array[N] real<lower=0, upper=1> p_rep;
  array[N] int r_rep;
  for (i in 1:N) {
    p_rep[i] = beta_rng(alpha, beta);
    r_rep[i] = binomial_rng(n_trials[i], p_rep[i]);
  }
}
```

### Expected Posterior Behavior

**If model is correct:**
- μ posterior: E[μ] ≈ 0.07-0.08 (near observed 0.076)
- κ posterior: E[κ] ≈ 0.3-5 (much lower than prior mean if ICC=0.73 is real!)
- **RED FLAG ALERT:** If κ posterior << prior, it means groups are MUCH more variable than prior assumed
- This could indicate: (1) True high heterogeneity, or (2) Mixture model needed

**Correlation structure:**
- Expect posterior corr(μ, κ) ≈ -0.3 to 0.3 (weakly correlated)
- If |corr| > 0.7, consider non-centered parameterization

### Computational Considerations

**Potential issues:**
- μ-κ parameterization can have worse geometry than α-β
- If κ is very small, numerical issues possible
- May need longer warmup than Model A

**Mitigation:**
- Use non-centered parameterization if needed
- Increase adapt_delta to 0.95 if divergences occur

---

## Model C: Two-Component Mixture (Adversarial Falsification Test)

### Mathematical Specification

```
Level 1 (Data likelihood):
  r_i | p_i, n_i ~ Binomial(n_i, p_i)

Level 2 (Mixture of subpopulations):
  p_i ~ π × Beta(α_1, β_1) + (1-π) × Beta(α_2, β_2)

  where:
    - Subpop 1: "Low-rate groups" with mean μ_1
    - Subpop 2: "High-rate groups" with mean μ_2
    - π: Mixing proportion

Hyperpriors:
  π ~ Beta(2, 2)                  [Uniform-ish on [0,1], slightly concentrated]
  μ_1 ~ Beta(3, 50)              [Low rate prior, mean ≈ 0.057]
  μ_2 ~ Beta(5, 20)              [High rate prior, mean ≈ 0.20]
  κ_1 ~ Gamma(2, 0.1)            [Concentration for low-rate cluster]
  κ_2 ~ Gamma(2, 0.1)            [Concentration for high-rate cluster]
```

### Rationale: Why Consider This Model?

**EDA evidence AGAINST this model:**
- Shapiro-Wilk test p=0.496: Success rates consistent with normal distribution (continuous)
- 84.8% of group pairs have overlapping CIs (no clear separation)
- Caterpillar plots show continuous spread, not discrete clusters
- No bimodality visible in histograms

**Then why include it?**
This is a **falsification test**. I want to see if:
1. Data strongly prefer continuous variation (Models A/B) over mixture
2. Or if my interpretation of EDA was wrong and clustering is real

**I will be SURPRISED if this model wins.** If it does, it means:
- My EDA interpretation was wrong
- There IS meaningful clustering (e.g., "normal" vs "exceptional" groups)
- Need to rethink entire approach

### Falsification Criteria

**I will abandon this model if:**
- ΔAIC vs Model B > 5 (mixture not justified)
- Posterior π or (1-π) < 0.1 (one component vanishes → use simpler model)
- Posterior μ_1 and μ_2 overlap substantially (credible intervals >80% overlap)
- Mixture components don't align with any interpretable grouping

**I will SERIOUSLY CONSIDER this model if:**
- ΔAIC vs Model B < -10 (strong evidence for mixture)
- Bayes factor > 100 favoring mixture
- Posterior clearly separates groups into 2 clusters
- Components have interpretable meaning (e.g., Groups 1,5 in low cluster; Groups 2,8,11 in high cluster)

### Prior Justification

**Mixing proportion π ~ Beta(2,2):**
- Slightly favors balanced mixtures (mean=0.5)
- But allows wide range [0.1, 0.9]
- Prevents degenerate solutions with π≈0 or π≈1

**Component means:**
- μ_1 ~ Beta(3,50): Mean≈0.06, concentrated on low rates
- μ_2 ~ Beta(5,20): Mean≈0.20, allows high rates
- **Constraint:** Need E[μ_2] > E[μ_1] for identifiability
- Should use ordered constraint in Stan

**Component concentrations:**
- Same prior for both: Gamma(2, 0.1)
- Allows components to have different within-cluster variation
- If κ_1 ≈ κ_2 in posterior → unnecessary complexity

### Stan Implementation

```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r_success;
}

parameters {
  real<lower=0, upper=1> pi;              // Mixing proportion
  ordered[2] mu;                           // Ordered means (μ_1 < μ_2 for identifiability)
  vector<lower=0>[2] kappa;                // Concentrations
}

transformed parameters {
  vector[2] alpha;
  vector[2] beta;

  for (k in 1:2) {
    alpha[k] = mu[k] * kappa[k];
    beta[k] = (1 - mu[k]) * kappa[k];
  }
}

model {
  // Priors
  pi ~ beta(2, 2);
  mu[1] ~ beta(3, 50);   // Low-rate component
  mu[2] ~ beta(5, 20);   // High-rate component
  kappa ~ gamma(2, 0.1);

  // Mixture likelihood
  for (i in 1:N) {
    target += log_mix(pi,
                      beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[1], beta[1]),
                      beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[2], beta[2]));
  }
}

generated quantities {
  // Posterior probability of each group belonging to each component
  array[N] simplex[2] component_prob;

  for (i in 1:N) {
    real lp1 = beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[1], beta[1]);
    real lp2 = beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[2], beta[2]);

    component_prob[i][1] = pi * exp(lp1) / (pi * exp(lp1) + (1-pi) * exp(lp2));
    component_prob[i][2] = 1 - component_prob[i][1];
  }

  // Posterior predictive
  array[N] int r_rep;
  for (i in 1:N) {
    int component = bernoulli_rng(pi) + 1;  // Sample component
    real p_i = beta_rng(alpha[component], beta[component]);
    r_rep[i] = binomial_rng(n_trials[i], p_i);
  }
}
```

### Expected Posterior Behavior

**If mixture model is CORRECT (I doubt it):**
- π ∈ [0.3, 0.7] with clear posterior mode
- μ_1 ≈ 0.04-0.06, μ_2 ≈ 0.11-0.15 (clear separation)
- Component assignments consistent:
  - Low cluster: Groups 1, 3, 4, 5, 6, 7, 9, 12
  - High cluster: Groups 2, 8, 11
- ΔAIC < -10 vs Model B

**If mixture model is WRONG (expected):**
- Posterior π → 0 or π → 1 (one component dominates)
- μ_1 and μ_2 posteriors overlap substantially
- Component assignments uncertain (all groups have prob ≈ π for each component)
- ΔAIC > 0 vs Model B (penalized for extra parameters)

### Computational Considerations

**Known challenges with mixture models:**
- Label switching: Components can flip between chains
- Multimodality: Posterior may have multiple modes
- Slow mixing: Difficult to explore full posterior

**Mitigation strategies:**
- Use `ordered[2] mu` constraint to prevent label switching
- Run 8 chains instead of 4
- Increase iterations to 4000
- Check Rhat carefully for each component parameter
- Use initialization near prior mode to help convergence

**Expected runtime:**
- 8 chains × 4000 iterations ≈ 3-5 minutes
- May see some divergences if components are poorly separated

---

## Model Comparison Strategy

### Quantitative Metrics

**Information Criteria:**
- Compute WAIC and LOO-CV for all three models
- Prefer model with lowest WAIC/LOO
- ΔAIC > 10 = strong evidence against higher-AIC model

**Bayes Factors:**
- Use bridge sampling to compute marginal likelihoods
- BF > 100 = decisive evidence
- BF 10-100 = strong evidence
- BF 3-10 = moderate evidence

**Posterior predictive checks:**
- Does model reproduce observed variance?
- Does model reproduce observed zero counts?
- Does model reproduce outlier behavior (Group 8)?

### Qualitative Assessment

**Scientific plausibility:**
- Is there a domain reason for subpopulations? (No info in dataset)
- Are parameter estimates reasonable?
- Do shrinkage patterns make sense?

**Computational health:**
- Rhat < 1.01 for all parameters
- ESS > 400 (preferably > 1000)
- No divergent transitions
- Pareto k < 0.7 for all LOO observations

### Decision Criteria

**Choose Model A/B over Model C if:**
- ΔAIC(C vs A/B) > 5
- Model C has identifiability issues (overlapping components)
- Model C offers no interpretable advantage

**Choose Model C over A/B if:**
- ΔAIC(C vs A/B) < -10
- Posterior clearly shows two distinct subpopulations
- Can explain WHY groups cluster (even post-hoc)

**Choose between A and B:**
- Doesn't matter! Same model, different parameterization
- Prefer B if μ interpretation is important
- Prefer A if computational issues arise with B

---

## Red Flags and Pivot Points

### When to Abandon Beta-Binomial Entirely

**Evidence that would make me switch to completely different model class:**

1. **Temporal structure discovered**
   - If external info reveals trials are ordered in time
   - → Switch to state-space model or GP

2. **Covariate information emerges**
   - If group-level predictors become available
   - → Switch to hierarchical GLM with covariates

3. **Non-binomial likelihood needed**
   - If overdispersion is even worse than beta-binomial can handle
   - → Consider beta-negative binomial or fully nonparametric

4. **Zero-inflation beyond Group 1**
   - If posterior predictive checks show systematic under-prediction of zeros
   - → Consider zero-inflated beta-binomial

5. **Spatial structure revealed**
   - If groups have geographic structure
   - → Add spatial correlation (GP or CAR prior)

### Critical Decision Points

**After fitting all three models:**
- **Decision 1:** If all models fail posterior predictive checks → NEW model class needed
- **Decision 2:** If mixture wins but π≈0.5 and assignment uncertain → Need more data or abandon mixture
- **Decision 3:** If κ posterior << 1 → Groups TOO heterogeneous, consider discrete effects instead

**Stopping rules:**
- If 3+ models all fail same posterior predictive check → Problem is with likelihood, not hierarchy
- If shrinkage makes all groups nearly identical → Data too sparse for hierarchical model
- If Group 8 still extreme outlier after shrinkage → Investigate data quality issue

---

## Prior Sensitivity Analysis Plan

### Parameters to Test

**Model A/B:**
- Vary prior on α (or μ): Gamma(1,0.5), Gamma(2,0.5), Gamma(5,0.5)
- Vary prior on β (or κ): Gamma(2,0.05), Gamma(2,0.1), Gamma(2,0.5)
- Check: Do posterior conclusions change?

**Model C:**
- Vary prior on π: Beta(1,1), Beta(2,2), Beta(5,5)
- Check: Does component assignment change?

### Sensitivity Metrics

**Robust conclusions:**
- Mean success rate should be stable within ±0.01 across priors
- Overdispersion φ should be stable within ±0.5

**Prior-dependent conclusions (warning sign):**
- If Group 1 posterior changes from 2% to 6% based on prior → Need more data
- If Model C component probabilities flip → Mixture not identified

---

## Expected Deliverables After Model Fitting

### For Each Model:

1. **Convergence diagnostics table** (Rhat, ESS, divergences)
2. **Posterior summary table** (mean, SD, 95% CI for all parameters)
3. **Prior vs posterior plot** (did data update beliefs?)
4. **Posterior predictive check plots**:
   - Observed vs predicted success rates
   - Observed vs predicted variance
   - Calibration plot (rank histograms)
5. **Shrinkage plot** (MLE vs posterior mean for each group)
6. **Leave-one-out cross-validation** (PSIS-LOO with Pareto k diagnostics)

### Model Comparison:

1. **AIC/WAIC/LOO table** with standard errors
2. **Bayes factor table** (if computable)
3. **Overlay of posterior predictions** from all models
4. **Recommendation** with strength of evidence

---

## Implementation Notes

### Python/PyStan Workflow

```python
import cmdstanpy
import numpy as np
import pandas as pd
import arviz as az

# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Prepare for Stan
stan_data = {
    'N': len(data),
    'n_trials': data['n_trials'].values,
    'r_success': data['r_successes'].values
}

# Compile and fit Model A
model_a = cmdstanpy.CmdStanModel(stan_file='model_a_beta_binomial.stan')
fit_a = model_a.sample(data=stan_data, chains=4, iter_sampling=2000,
                        adapt_delta=0.95, max_treedepth=12)

# Convert to ArviZ for analysis
idata_a = az.from_cmdstanpy(fit_a)

# Check diagnostics
print(az.summary(idata_a, var_names=['alpha', 'beta']))
print(f"Divergences: {fit_a.divergences}")

# Posterior predictive checks
az.plot_ppc(idata_a, num_pp_samples=100)

# LOO-CV
loo_a = az.loo(idata_a)
print(loo_a)
```

### PyMC Alternative

```python
import pymc as pm
import arviz as az

with pm.Model() as model_a:
    # Hyperpriors
    alpha = pm.Gamma('alpha', alpha=2, beta=0.5)
    beta = pm.Gamma('beta', alpha=2, beta=0.1)

    # Likelihood
    r = pm.BetaBinomial('r', alpha=alpha, beta=beta,
                        n=n_trials, observed=r_success)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4,
                      target_accept=0.95, return_inferencedata=True)

    # Posterior predictive
    ppc = pm.sample_posterior_predictive(trace)
```

---

## Conclusion: My Predictions

**Most likely outcome:**
- **Model B wins** (reparameterized beta-binomial) due to better interpretability
- Model A is essentially tied (same likelihood)
- Model C loses decisively (ΔAIC > 10)

**If I'm wrong:**
- Model C winning would mean **discrete subpopulations exist** → Need to explain why
- Could indicate need for covariate-based model or domain knowledge we're missing

**Critical unknowns:**
- True value of κ (concentration): Is ICC=0.73 real or EDA artifact?
- Is Group 8 genuinely different or measurement error?
- Are there latent covariates we don't observe?

**Next steps after fitting:**
- If all models fit well: Choose simplest (Model B)
- If all models fit poorly: Rethink likelihood structure
- If mixture wins: Investigate what distinguishes clusters

---

## File Locations

**Stan code:** `/workspace/experiments/designer_1/stan_models/`
- `model_a_beta_binomial.stan`
- `model_b_reparameterized.stan`
- `model_c_mixture.stan`

**Python scripts:** `/workspace/experiments/designer_1/scripts/`
- `fit_models.py` (main fitting script)
- `prior_predictive.py` (prior checks)
- `posterior_analysis.py` (diagnostics and plots)
- `model_comparison.py` (LOO-CV and WAIC)

**Outputs:** `/workspace/experiments/designer_1/results/`
- `convergence_diagnostics.csv`
- `posterior_summaries.csv`
- `model_comparison_table.csv`
- `plots/` (all diagnostic plots)

---

**Designer 1 Sign-off**

I expect Model B to win, but I've designed Model C as an adversarial test. If the data prefer discrete clustering, I'll abandon continuous variation models entirely and propose GROUP-SPECIFIC parameters instead. The goal is finding truth, not defending beta-binomial models.
