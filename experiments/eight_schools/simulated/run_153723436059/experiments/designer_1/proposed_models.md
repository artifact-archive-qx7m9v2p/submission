# Bayesian Model Proposals for Eight Schools Dataset
## Designer 1 - Independent Modeling Strategy

**Date**: 2025-10-29
**Context**: Eight Schools hierarchical modeling problem
**EDA Source**: `/workspace/eda/eda_report.md`

---

## Executive Summary

I propose **three distinct model classes** that embody fundamentally different hypotheses about the data generation process:

1. **Standard Hierarchical Model** (Partial Pooling) - Assumes exchangeable school effects with unknown between-school variation
2. **Near-Complete Pooling Model** (Skeptical of Heterogeneity) - Assumes effects are essentially identical with regularization toward homogeneity
3. **Mixture Model** (Latent Subgroups) - Assumes schools belong to distinct unobserved clusters

**Critical Finding from EDA**: The variance paradox (observed variance < expected) and extremely low I² = 1.6% suggests this dataset may challenge standard hierarchical modeling assumptions. Each model makes different bets on what this means.

---

## Philosophy: Competing Hypotheses & Falsification

### The Core Tension

The EDA reveals a statistical anomaly: observed between-school variance (124) is **75% of expected sampling variance** (166). This is backwards from typical hierarchical settings where we expect additional variation beyond sampling error.

**Three competing explanations:**
1. **Random fluctuation** - With n=8, could just be sampling variability (standard hierarchical model)
2. **True homogeneity** - Schools genuinely have nearly identical effects (near-complete pooling)
3. **Hidden structure** - Apparent homogeneity masks distinct subgroups that cancel out (mixture model)

### Truth-Seeking Mindset

**My commitment**: If evidence falsifies a model, I will abandon it immediately. Success is discovering which model class the data actually support, not defending initial choices.

**Red flags to watch for**:
- Prior-posterior conflict (model fighting the data)
- Extreme parameter values (tau near boundary, mixture weights near 0/1)
- Poor posterior predictive performance despite good in-sample fit
- Computational pathologies (divergences, low ESS, high R-hat)

---

## Model 1: Standard Hierarchical Model (Partial Pooling)

### Mathematical Specification

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i)   for i = 1, ..., 8
```
where y_i is observed effect, sigma_i is known standard error

**School-level model**:
```
theta_i ~ Normal(mu, tau)
```
where theta_i is true effect for school i

**Hyperpriors**:
```
mu ~ Normal(0, 50)           # Weakly informative on overall mean
tau ~ HalfCauchy(0, 25)      # Gelman's recommendation for hierarchical SD
```

### Stan Implementation Sketch

```stan
data {
  int<lower=0> J;              // number of schools = 8
  vector[J] y;                 // observed effects
  vector<lower=0>[J] sigma;    // known SEs
}
parameters {
  real mu;                     // population mean
  real<lower=0> tau;           // between-school SD
  vector[J] theta;             // school-specific effects
}
model {
  // Hyperpriors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 25);

  // School level
  theta ~ normal(mu, tau);

  // Likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
  // Posterior predictive for new school
  real theta_new = normal_rng(mu, tau);

  // Shrinkage factors
  vector[J] shrinkage;
  for (j in 1:J) {
    shrinkage[j] = 1 - tau^2 / (tau^2 + sigma[j]^2);
  }
}
```

### Theoretical Rationale

**Why this model class?**
- **Exchangeability**: Treats schools as random sample from population of schools
- **Adaptive pooling**: Data determines optimal shrinkage via posterior of tau
- **Standard framework**: This IS the canonical Eight Schools model (Rubin 1981, Gelman & Hill 2007)
- **Conservative**: Makes minimal structural assumptions beyond normality and exchangeability

**What it assumes:**
1. School effects are exchangeable (no ordering, no covariates)
2. Between-school variation is additive and Gaussian
3. Effects come from single normal population
4. tau is constant across schools (homoscedastic)

### Prior Justification

**For mu ~ Normal(0, 50)**:
- Centered at 0 (no prior directional belief)
- SD = 50 allows for effects in range [-100, 100] with high probability
- Given observed range of -5 to 26, this is weakly informative
- Will not dominate likelihood with n=8 schools

**For tau ~ HalfCauchy(0, 25)**:
- **Gelman (2006) recommendation** for hierarchical variance parameters
- Heavy tails: allows for large tau if heterogeneity exists
- Regularizes toward 0: shrinks tau when data suggest homogeneity
- Scale = 25 is motivated by plausible effect sizes
- Performs well with small number of groups (J=8)
- Avoids funnel geometry better than half-normal

**Alternative priors to test in sensitivity analysis**:
- `tau ~ HalfNormal(0, 25)` - Less heavy-tailed, stronger regularization
- `tau ~ HalfStudent_t(3, 0, 25)` - Intermediate
- `tau ~ Exponential(1/10)` - Stronger shrinkage toward 0
- `tau ~ Uniform(0, 100)` - Non-informative but risks impropriety

### Expected Behavior

**Given EDA findings (I² = 1.6%, variance ratio = 0.75)**:

1. **Posterior of tau should be small**:
   - Likely mode near 0-5
   - High posterior probability that tau < 10
   - May have long right tail (uncertainty about heterogeneity with n=8)

2. **Substantial shrinkage expected**:
   - School 4 (effect=25.7) shrinks down toward mu
   - School 5 (effect=-4.9) shrinks up toward mu
   - Schools with larger sigma shrink more (School 8 most, sigma=18)
   - Shrinkage factor formula: 1 - tau²/(tau² + sigma_i²)

3. **Posterior of mu likely in range [8, 15]**:
   - Between unweighted mean (12.5) and weighted mean (10.0)
   - Influenced by all schools with inverse-variance weighting
   - School 5 pulls down due to high precision

4. **Narrow credible intervals for theta_i**:
   - Posterior intervals narrower than y_i ± 2*sigma_i
   - Reflects information borrowing across schools
   - Quantifies shrinkage uncertainty

### Falsification Criteria: I Will Abandon This Model If...

**1. Tau posterior concentrates at boundary (tau ≈ 0)**:
- **What it means**: Complete pooling is strongly preferred
- **Action**: Switch to Model 2 (near-complete pooling)
- **Threshold**: If P(tau < 1 | data) > 0.9

**2. Posterior predictive checks fail systematically**:
- **What to check**:
  - Distribution of replicated y_rep doesn't match observed y
  - Systematic bias in certain schools (especially School 5)
  - Replicated data shows more heterogeneity than observed
- **Action**: Investigate mixture model (Model 3) or non-exchangeability

**3. Prior-posterior conflict for tau**:
- **What it means**: Prior and likelihood give contradictory information
- **Evidence**: Posterior narrower than prior (information from prior, not data)
- **Action**: Question whether tau is identifiable with n=8 and high sigma_i

**4. School 5 behaves as true outlier**:
- **What to check**: Pareto-k diagnostic from LOO-CV > 0.7 for School 5
- **What it means**: School 5 not well-predicted by leaving it out
- **Action**: Consider removing School 5 or robust model with heavier tails

**5. Computational pathologies**:
- **Divergent transitions** > 1% (even after increasing adapt_delta)
- **Low ESS** for tau (< 100 effective samples)
- **High R-hat** > 1.01
- **What it means**: Geometry of posterior is pathological (funnel, multimodality)
- **Action**: Reparameterize (non-centered) or reconsider model structure

### Stress Tests

**Test 1: Leave-One-Out Sensitivity**
- **Purpose**: Check if any single school drives inference
- **Method**: Refit model excluding each school sequentially
- **Expected**: Posterior of mu should be robust, tau may be sensitive with n=8
- **Failure**: If excluding School 5 changes mu by > 5 or tau doubles/halves

**Test 2: Prior Sensitivity**
- **Purpose**: Assess influence of tau prior on posterior
- **Method**: Refit with tau ~ Exponential(0.1), HalfNormal(0, 10), Uniform(0, 50)
- **Expected**: Posterior should be similar if data are informative
- **Failure**: If posterior of tau varies by factor > 2 across reasonable priors

**Test 3: Posterior Predictive for New School**
- **Purpose**: Test out-of-sample generalization
- **Method**: Draw theta_new ~ N(mu, tau), compare to observed distribution
- **Expected**: theta_new distribution should cover observed effects
- **Failure**: If observed effects systematically more/less variable than predicted

**Test 4: Shrinkage Consistency**
- **Purpose**: Check if shrinkage pattern matches uncertainty structure
- **Method**: Compare shrinkage factors to sigma_i
- **Expected**: Schools with larger sigma shrink more
- **Failure**: If shrinkage uncorrelated with sigma (would suggest model misspecification)

---

## Model 2: Near-Complete Pooling Model (Homogeneity with Noise)

### Mathematical Specification

**Core hypothesis**: Schools have essentially identical true effects; observed variation is pure sampling noise.

**Likelihood**:
```
y_i ~ Normal(mu, sigma_i)   for i = 1, ..., 8
```

**Hyperprior**:
```
mu ~ Normal(10, 15)         # Informed by observed data range
```

**Alternative formulation with minimal heterogeneity**:
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
mu ~ Normal(10, 15)
tau ~ Exponential(0.5)      # Strong regularization toward tau ≈ 0
```

### Stan Implementation Sketch

```stan
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;                      // common mean
  real<lower=0> tau;            // minimal residual variation
  vector[J] theta_raw;          // non-centered parameterization
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  // Strong regularization toward homogeneity
  mu ~ normal(10, 15);
  tau ~ exponential(0.5);       // Mode at 0, E[tau] = 2
  theta_raw ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
  // Check if complete pooling justified
  real prob_homogeneous = tau < 5 ? 1 : 0;
}
```

### Theoretical Rationale

**Why this model class?**
- **Takes EDA findings seriously**: I² = 1.6% suggests near-zero heterogeneity
- **Occam's Razor**: Simplest explanation for variance paradox is true homogeneity
- **Statistical evidence**: Chi-square test p = 0.42 fails to reject H₀: all effects equal
- **Meta-analytic convention**: I² < 25% often treated as "low heterogeneity" justifying fixed-effect models

**What it assumes:**
1. Schools differ only due to sampling variability
2. Any residual tau is negligible (< 5 on scale of effects)
3. Treatment effect is context-independent
4. Observed heterogeneity is random noise

**Philosophical stance**:
- **Skeptical of heterogeneity**: Given small n=8 and high individual uncertainty, apparent differences may be illusory
- **Conservative inference**: Pooling maximizes precision when justified
- **Falsifiable**: Can test if data truly support this strong assumption

### Prior Justification

**For mu ~ Normal(10, 15)**:
- Centered at 10 (near weighted mean from EDA)
- SD = 15 covers observed range [-5, 26] within ±2 SD
- More informative than Model 1 (reflects belief in homogeneity)
- Still allows data to shift mu considerably

**For tau ~ Exponential(0.5)**:
- **Strong regularization** toward 0 (mode at 0, mean = 2)
- Implies belief that P(tau > 5) is small a priori
- Contrasts with half-Cauchy: much less heavy-tailed
- Forces model to demonstrate heterogeneity to overcome prior
- **Falsifiable**: If data show strong heterogeneity, posterior will conflict with prior

**Non-centered parameterization**:
- theta_raw ~ N(0,1) with theta = mu + tau * theta_raw
- Avoids funnel geometry when tau near 0
- Improves sampling efficiency
- Essential for this model given expected small tau

### Expected Behavior

1. **Posterior of tau concentrated near 0**:
   - Mode likely < 2
   - 95% CI likely contained in [0, 8]
   - Exponential prior + low-heterogeneity data reinforce each other

2. **Strong shrinkage toward common mean**:
   - All theta_i pulled strongly toward mu
   - School 4 shrinks from 25.7 toward ~12
   - School 5 shrinks from -4.9 toward ~12
   - Minimal differentiation between schools

3. **Precision gain from pooling**:
   - Posterior SD of mu approximately 1/sqrt(sum(1/sigma_i²)) ≈ 3.2
   - Much narrower than any individual school's SE
   - All school posteriors roughly equally narrow

4. **Posterior of mu near weighted mean**:
   - Weighted mean = 10.0 from EDA
   - Prior centered at 10
   - Expected posterior mode around 10-11

### Falsification Criteria: I Will Abandon This Model If...

**1. Posterior of tau escapes from prior (prior-posterior conflict)**:
- **Threshold**: If posterior mode of tau > 10
- **What it means**: Data contain more heterogeneity than exponential prior allows
- **Action**: Switch back to Model 1 with diffuse tau prior

**2. Posterior predictive replications too homogeneous**:
- **Test**: If observed y variance > 95th percentile of replicated y variance
- **What it means**: Model forces too much similarity, doesn't capture observed spread
- **Action**: Abandon homogeneity hypothesis

**3. LOO-CV shows poor predictive performance**:
- **Threshold**: If LOO-ELPD worse than Model 1 by > 5 (on log scale)
- **What it means**: Pooling destroys predictive accuracy
- **Action**: Return to partial pooling

**4. School 5 systematically mispredicted**:
- **Test**: Posterior predictive p-value for School 5 < 0.05
- **What it means**: School 5 genuinely different, not just noise
- **Action**: Consider mixture model or remove outlier

**5. Posterior intervals fail to cover true effects**:
- **Test**: If we later get "true" effects (e.g., larger study), check coverage
- **What it means**: Overconfidence from excessive pooling
- **Action**: Recognize homogeneity assumption was wrong

### Stress Tests

**Test 1: Compare Complete Pooling vs. This Model**
- **Purpose**: Is minimal tau adding anything?
- **Method**: Fit pure fixed-effect model (tau = 0 exactly)
- **Expected**: Should give nearly identical inferences
- **Failure**: If posteriors differ substantially, model is hedging inappropriately

**Test 2: School 5 Exclusion**
- **Purpose**: Is apparent homogeneity driven by one outlier canceling others?
- **Method**: Refit excluding School 5 (negative outlier)
- **Expected**: Homogeneity should strengthen (I² even lower)
- **Failure**: If tau increases after removing School 5, suggests hidden heterogeneity

**Test 3: Informative Prior Robustness**
- **Purpose**: Check sensitivity to prior on mu (more informative than Model 1)
- **Method**: Refit with mu ~ Normal(0, 50) and compare
- **Expected**: Posterior should shift but not dramatically
- **Failure**: If posterior mode of mu changes by > 3, prior too influential

**Test 4: Predictive Calibration**
- **Purpose**: Does model correctly quantify uncertainty?
- **Method**: Check if 68% posterior predictive intervals contain y_i for ~68% of schools
- **Expected**: Should be well-calibrated
- **Failure**: If systematically under-covers, model too confident

---

## Model 3: Mixture Model (Latent Subgroups)

### Mathematical Specification

**Core hypothesis**: Schools belong to K=2 unobserved subpopulations with different mean effects. Low observed heterogeneity is artifact of groups canceling out.

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i)   for i = 1, ..., 8
```

**Mixture structure**:
```
theta_i ~ sum_{k=1}^{K} pi_k * Normal(mu_k, tau_k)
```
where pi_k are mixing proportions, sum to 1

**For K=2 (two groups)**:
```
z_i ~ Categorical(pi)            # Latent group membership
theta_i | z_i=k ~ Normal(mu_k, tau_k)
pi ~ Dirichlet(alpha)            # Mixing proportions
```

**Hyperpriors**:
```
mu_k ~ Normal(0, 50)             for k = 1, 2
tau_k ~ HalfCauchy(0, 15)        for k = 1, 2  (within-group SD)
pi ~ Dirichlet([2, 2])           # Weakly informative on proportions
```

### Stan Implementation Sketch

```stan
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
  int<lower=1> K;                  // number of components (try K=2)
}
parameters {
  simplex[K] pi;                   // mixing proportions
  ordered[K] mu;                   // ordered to avoid label switching
  vector<lower=0>[K] tau;          // within-component SDs
  vector[J] theta;
}
model {
  // Priors
  pi ~ dirichlet(rep_vector(2, K));
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 15);

  // Mixture likelihood for theta
  for (j in 1:J) {
    vector[K] log_pi = log(pi);
    for (k in 1:K) {
      log_pi[k] += normal_lpdf(theta[j] | mu[k], tau[k]);
    }
    target += log_sum_exp(log_pi);
  }

  // Observation likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
  // Posterior cluster probabilities
  matrix[J, K] gamma;              // responsibility matrix
  for (j in 1:J) {
    vector[K] log_pi = log(pi);
    for (k in 1:K) {
      log_pi[k] += normal_lpdf(theta[j] | mu[k], tau[k]);
    }
    gamma[j] = softmax(log_pi)';
  }
}
```

### Theoretical Rationale

**Why this model class?**
- **Alternative explanation for variance paradox**: Low observed heterogeneity because two groups with opposite effects cancel out
- **Explains School 5**: Perhaps School 5 represents different subpopulation (negative vs. positive effects)
- **Accounts for potential confounders**: If schools differ on unmeasured characteristics, mixture captures latent structure
- **More flexible**: Can represent unimodal (if pi_1 ≈ 1) or multimodal distributions

**What it assumes:**
1. Discrete latent structure (not continuous heterogeneity)
2. Within-group effects are exchangeable
3. Groups are identifiable from data (despite n=8)
4. No measured predictors of group membership

**Philosophical stance**:
- **Adversarial to homogeneity**: Questions whether apparent similarity is real
- **Heterogeneity skepticism**: Maybe standard hierarchical model misses structure
- **Pattern detection**: Allows for School 5 to be genuinely different type

### Prior Justification

**For mu_k ~ Normal(0, 50) with ordered constraint**:
- Same diffuse prior as Model 1
- Ordered constraint (mu_1 < mu_2) prevents label switching
- Allows for widely separated groups or near-identical groups

**For tau_k ~ HalfCauchy(0, 15)**:
- Within-group variation expected smaller than overall (hence scale=15 vs 25)
- Still allows for substantial within-group heterogeneity if present
- Heavy tails avoid forcing tight clusters

**For pi ~ Dirichlet([2, 2])**:
- Weakly informative: slight preference for balanced groups
- Prior mean of each pi_k = 0.5 for K=2
- Allows for imbalanced groups (e.g., 7-1 split)
- Dirichlet(1,1) would be uniform, but alpha=2 prevents extreme weights

**Label switching solution**:
- Constraint mu_1 < mu_2 identifies components
- Alternative: post-hoc relabeling based on posterior mode
- With n=8, label switching may still occur in MCMC

### Expected Behavior

**Scenario A: Mixture is spurious (homogeneity is real)**:
1. Posterior of pi collapses: pi_1 ≈ 1 or pi_2 ≈ 1
2. One component contains all schools
3. Other component becomes ghost mode (high uncertainty, no mass)
4. Model reduces to single-component (equivalent to Model 1 or 2)

**Scenario B: Mixture is real (two subpopulations)**:
1. Balanced mixing proportions: pi_1 and pi_2 both > 0.2
2. Separated means: |mu_1 - mu_2| > 10
3. School 5 assigned to different component than Schools 1-4
4. Responsibility matrix gamma shows clear assignments

**Most likely outcome given EDA**:
- **Scenario A is expected**: EDA shows no evidence of multimodality
- But worth testing explicitly to rule out hidden structure
- If Scenario A occurs, model fails gracefully (becomes single-component)

### Falsification Criteria: I Will Abandon This Model If...

**1. Posterior mixing proportions collapse to single component**:
- **Threshold**: If max(pi) > 0.85
- **What it means**: Data do not support mixture structure
- **Action**: Abandon mixture model, return to simpler Model 1 or 2

**2. Components not separable (mu_1 ≈ mu_2)**:
- **Threshold**: If |mu_1 - mu_2| < 5 in posterior
- **What it means**: Groups not meaningfully different
- **Action**: Mixture adds complexity without benefit

**3. High uncertainty in cluster assignments (diffuse gamma)**:
- **Threshold**: If all gamma[j,k] in [0.3, 0.7] (no clear assignments)
- **What it means**: Model cannot decide which schools belong where
- **Action**: Identifiability failure, abandon mixture

**4. Mixture worse than simpler models (model comparison)**:
- **Threshold**: If WAIC or LOO-CV of mixture > Model 1 + 5
- **What it means**: Complexity penalty outweighs fit improvement
- **Action**: Accept simpler model

**5. Computational issues (label switching, non-convergence)**:
- **Evidence**: R-hat > 1.1 for pi or mu despite long chains
- **What it means**: Model too complex for data, multimodality in posterior
- **Action**: Cannot reliably fit this model, abandon

### Stress Tests

**Test 1: K=1 vs K=2 vs K=3**
- **Purpose**: How many components do data support?
- **Method**: Fit with K=1 (no mixture), K=2, K=3 and compare WAIC/LOO
- **Expected**: K=1 should win (simplest and adequate)
- **Failure**: If K>1 substantially better, suggests hidden structure

**Test 2: Posterior Predictive for Bimodality**
- **Purpose**: Does mixture predict bimodal distribution we don't observe?
- **Method**: Generate y_rep and check if more bimodal than observed
- **Expected**: Should not create spurious modes
- **Failure**: If replicated data systematically more bimodal, model overfitting

**Test 3: Prior Sensitivity for pi**
- **Purpose**: Are conclusions sensitive to prior on mixing proportions?
- **Method**: Refit with Dirichlet([0.5, 0.5]) and Dirichlet([5, 5])
- **Expected**: If data are uninformative, posterior tracks prior
- **Failure**: If posterior highly sensitive, mixture is data artifact

**Test 4: School 5 Assignment**
- **Purpose**: Is School 5 naturally assigned to different component?
- **Method**: Check posterior gamma[5, k] - should be decisive if real
- **Expected**: If mixture real, School 5 in different group than Schools 1-4
- **Failure**: If School 5 assignment uncertain, mixture not explaining its outlier status

---

## Model Comparison Strategy

### Falsification Cascade

**Phase 1: Fit all three models independently**
- Ensure convergence for each (R-hat < 1.01, ESS > 400)
- Check for computational pathologies

**Phase 2: Posterior predictive checks**
- For each model, generate replicated datasets y_rep
- Visual comparison: Do replicates resemble observed data?
- Test statistics: variance, min, max, mean
- P-values: P(T(y_rep) > T(y_obs))

**Phase 3: Model comparison metrics**
- Compute WAIC (Watanabe-Akaike Information Criterion)
- Compute LOO-CV (Leave-One-Out Cross-Validation with Pareto-k diagnostic)
- Compare: WAIC differences > 5 considered meaningful
- Check: Any schools with Pareto-k > 0.7 (influential outliers)

**Phase 4: Sensitivity analysis**
- For best model(s), vary priors and check robustness
- Exclude School 5 and refit
- Check if conclusions stable

### Decision Rules

**If Model 1 (Standard Hierarchical) wins**:
- Proceed with partial pooling as planned
- Report shrinkage estimates and tau posterior
- This is expected outcome and standard practice

**If Model 2 (Near-Complete Pooling) wins**:
- Accept that heterogeneity is negligible
- Report pooled estimate with narrow CI
- Acknowledge EDA findings support this strongly

**If Model 3 (Mixture) wins**:
- **Major re-evaluation needed**
- Investigate why standard model failed
- Explore School 5 and other schools for unmeasured differences
- Consider this a surprising discovery requiring explanation

**If models are inconclusive (close comparison)**:
- Use Bayesian Model Averaging
- Report results under model uncertainty
- Acknowledge limitation of n=8 for discrimination

### Red Flags Across All Models

**1. All models fail posterior predictive checks**:
- **What it means**: Normality assumption wrong, or fundamental model class mismatch
- **Action**: Consider robust models (t-distributed errors) or non-parametric approaches

**2. Extreme parameter values across models**:
- **Examples**: mu > 50 or < -50, tau > 100
- **What it means**: Data contain extreme information not captured by priors
- **Action**: Re-examine data quality, check for data entry errors

**3. School 5 always problematic**:
- **Evidence**: High Pareto-k across all models, poor predictive performance
- **What it means**: School 5 genuinely different mechanism
- **Action**: Investigate School 5 methodology, consider excluding or modeling separately

**4. Wide posterior intervals despite pooling**:
- **Evidence**: Posterior SD for mu > 10
- **What it means**: Data too noisy to learn anything even with pooling
- **Action**: Accept high uncertainty, report weak conclusions

---

## Stopping Rules & Pivot Points

### When to Stop and Reconsider Everything

**Checkpoint 1: After initial fits**
- **If**: Convergence failures across multiple models
- **Then**: Question whether MCMC is appropriate, consider variational inference or simpler methods

**Checkpoint 2: After posterior predictive checks**
- **If**: All models fail systematic checks
- **Then**: Return to EDA, question normality assumption, consider data issues

**Checkpoint 3: After model comparison**
- **If**: Models give contradictory substantive conclusions
- **Then**: Report model uncertainty, do not force single answer

**Checkpoint 4: After sensitivity analysis**
- **If**: Conclusions highly sensitive to School 5 inclusion
- **Then**: Report two analyses (with/without School 5), investigate School 5 deeply

### Alternative Approaches If Initial Models Fail

**Backup Plan A: Robust Hierarchical Model**
- Replace Normal with Student-t distribution (heavier tails)
- Allow for outliers without mixture structure
- Estimate degrees of freedom from data

**Backup Plan B: Non-parametric Dirichlet Process**
- Use infinite mixture model
- Let data determine number of components
- More flexible but requires larger n (may not work with n=8)

**Backup Plan C: Empirical Bayes**
- Estimate tau by method of moments or REML
- Plug into shrinkage formula
- Faster, less principled, but may be adequate given data limitations

**Backup Plan D: Simple Weighted Average**
- If all hierarchical approaches fail or overfit
- Report inverse-variance weighted mean = 10.02
- Acknowledge we cannot reliably estimate between-school variation

---

## Summary Table: Model Comparison

| Aspect | Model 1: Standard | Model 2: Near-Complete | Model 3: Mixture |
|--------|------------------|----------------------|-----------------|
| **Core hypothesis** | Random heterogeneity | Homogeneity + noise | Latent subgroups |
| **tau prior** | HalfCauchy(0, 25) | Exponential(0.5) | HalfCauchy(0, 15) per group |
| **Expected tau posterior** | Mode: 3-8 | Mode: 0-2 | Separate tau_k |
| **Expected shrinkage** | Moderate | Strong | Within-group |
| **Parameters** | 10 (mu, tau, theta_i) | 10 (mu, tau, theta_i) | 13 (mu_k, tau_k, pi, theta_i, z_i) |
| **Complexity** | Medium | Low | High |
| **Identifiability** | Good | Excellent | Questionable with n=8 |
| **Computational** | Standard | Easy (non-centered) | Challenging (label switching) |
| **EDA support** | Moderate | Strong (I²=1.6%) | Weak (no multimodality) |
| **Falsifiable by** | tau → 0 or ∞ | tau >> 0 | Single component |
| **Best for** | Standard inference | Precise estimation | Discovery of structure |

---

## Implementation Plan

### Phase 1: Model Fitting (Week 1)
1. Implement all three Stan models
2. Run MCMC with default settings (warmup=1000, iter=2000, chains=4)
3. Check convergence diagnostics
4. If divergences, increase adapt_delta to 0.99
5. If needed, implement non-centered parameterization

### Phase 2: Posterior Analysis (Week 1-2)
1. Extract and visualize posteriors for key parameters
2. Compute shrinkage factors for each model
3. Generate posterior predictive distributions
4. Create forest plots comparing observed vs. posterior means

### Phase 3: Model Checking (Week 2)
1. Posterior predictive checks with graphical displays
2. Compute WAIC and LOO-CV for all models
3. Check Pareto-k diagnostics
4. Visual comparison of replicated vs. observed data

### Phase 4: Sensitivity Analysis (Week 2-3)
1. Refit Model 1 with alternative tau priors
2. Exclude School 5 and refit all models
3. Compare to complete pooling baseline
4. Check prior-posterior plots for conflict

### Phase 5: Reporting (Week 3)
1. Document which model(s) are supported
2. Report falsification checks: did any model fail?
3. Provide substantive interpretation
4. Quantify uncertainty and limitations
5. Recommend next steps or additional data needs

---

## Anticipated Challenges

### Challenge 1: Small Sample Size (n=8)
- **Issue**: Hard to estimate tau with only 8 groups
- **Evidence**: Wide posterior for tau expected
- **Mitigation**: Priors on tau more influential than usual
- **Acceptance**: High uncertainty is honest answer

### Challenge 2: High Individual Uncertainty
- **Issue**: Large sigma_i means weak information per school
- **Evidence**: Mean sigma = 12.5, comparable to effect SD = 11.2
- **Mitigation**: Pooling is crucial precisely because of this
- **Acceptance**: Some questions may not have precise answers

### Challenge 3: School 5 Outlier
- **Issue**: Only negative effect, may distort inferences
- **Evidence**: Z-score = -1.56, high precision (sigma=9)
- **Mitigation**: Check sensitivity to inclusion/exclusion
- **Acceptance**: May need to report conditional on School 5 treatment

### Challenge 4: Variance Paradox
- **Issue**: Observed variance < expected is backwards from usual hierarchical setting
- **Evidence**: Variance ratio = 0.75
- **Mitigation**: This supports near-complete pooling, not a problem for analysis
- **Acceptance**: Unusual but interpretable pattern

### Challenge 5: Model Comparison with Small n
- **Issue**: WAIC and LOO may be unreliable with n=8
- **Evidence**: Effective parameters comparable to sample size
- **Mitigation**: Use multiple comparison metrics, visual checks
- **Acceptance**: Model selection may be inconclusive

---

## Success Criteria

**This modeling effort succeeds if**:

1. **We discover the truth**, even if it contradicts initial hypotheses
2. **We identify which model class is falsified** by the data
3. **We quantify uncertainty honestly**, including model uncertainty
4. **We provide actionable recommendations** for interpretation
5. **We document what we learned** about the data generation process

**This modeling effort FAILS if**:

1. We report a "best model" that fails posterior predictive checks
2. We ignore evidence against our chosen model
3. We claim certainty where data provide only weak evidence
4. We fit complex models without checking if they improve over simple baselines
5. We treat model selection as the goal rather than understanding the phenomenon

---

## Conclusion

I propose three distinct model classes that make **testable, falsifiable predictions**:

1. **Model 1 (Standard Hierarchical)**: Bet that partial pooling with data-driven tau is optimal
2. **Model 2 (Near-Complete Pooling)**: Bet that I²=1.6% means effects are essentially identical
3. **Model 3 (Mixture)**: Bet that apparent homogeneity masks hidden structure

**My prediction**: Model 1 or Model 2 will be supported; Model 3 will collapse to single component. The key question is whether posterior of tau in Model 1 is small enough to justify Model 2's stronger assumptions.

**My commitment**: If evidence falsifies my predictions, I will report that as success, not failure. The goal is truth, not confirmation.

**Key decision point**: After fitting all three models, if they give contradictory conclusions, I will NOT force a single answer. I will report model uncertainty and the range of plausible inferences.

---

## Files and Code

**Model implementations**: Will be provided in `/workspace/experiments/designer_1/models/`
- `model1_standard_hierarchical.stan`
- `model2_near_complete_pooling.stan`
- `model3_mixture.stan`

**Analysis scripts**: Will be provided in `/workspace/experiments/designer_1/code/`
- `fit_models.py` - Fit all models and check convergence
- `posterior_analysis.py` - Analyze posteriors and compute summaries
- `model_comparison.py` - WAIC, LOO, and posterior predictive checks
- `sensitivity_analysis.py` - Prior sensitivity and robustness checks

**Outputs**: Results will be written to `/workspace/experiments/designer_1/results/`
- `convergence_diagnostics.txt`
- `posterior_summaries.csv`
- `model_comparison.csv`
- `posterior_predictive_plots.png`
- `final_report.md`

---

**Designer**: 1
**Date**: 2025-10-29
**Status**: Awaiting implementation phase
