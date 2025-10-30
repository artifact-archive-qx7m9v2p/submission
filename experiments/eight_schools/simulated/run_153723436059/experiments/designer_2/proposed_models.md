# Bayesian Model Proposals for Eight Schools Dataset
## Designer 2 - Independent Analysis

**Date**: 2025-10-29
**Dataset**: Eight Schools (n=8 schools, known sigma_i)
**Key EDA Finding**: Very low heterogeneity (I²=1.6%), variance paradox (observed < expected)

---

## Executive Summary

I propose three **distinct** Bayesian model classes that make fundamentally different assumptions about the data generation process. The EDA suggests low heterogeneity, but I'm deliberately proposing models that can **falsify this conclusion** if it's wrong.

**Critical insight**: The variance paradox (observed variance 75% of expected) could mean:
1. True homogeneity (Model 1)
2. Shrinkage artifact from correlated sampling (Model 2)
3. Measurement issue with reported sigma_i (Model 3)

Each model represents a different interpretation of the same evidence.

---

## Model 1: Adaptive Hierarchical Normal Model (Standard Approach)

### Mathematical Specification

```
Likelihood:    y_i ~ Normal(theta_i, sigma_i)     [sigma_i known]
School level:  theta_i ~ Normal(mu, tau)
Hyperpriors:   mu ~ Normal(0, 50)                 [weakly informative]
               tau ~ half-Cauchy(0, 10)           [regularized at 0]
```

### Stan Implementation Outline

```stan
data {
  int<lower=1> J;              // 8 schools
  vector[J] y;                 // observed effects
  vector<lower=0>[J] sigma;    // known SEs
}
parameters {
  real mu;                     // population mean effect
  real<lower=0> tau;           // between-school SD
  vector[J] theta_raw;         // non-centered parameterization
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 10);
  theta_raw ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] y_rep;
  real theta_new;

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
  }
  theta_new = normal_rng(mu, tau);  // predictive for new school
}
```

### Theoretical Rationale

**Why this model for this problem?**

1. **Matches the data structure**: Two-level hierarchy (schools nested in population) with known within-school uncertainty
2. **Data-adaptive pooling**: The posterior of tau determines shrinkage strength - if tau→0, model converges to complete pooling; if tau is large, minimal pooling
3. **Non-centered parameterization**: Essential for low-tau cases (likely here given I²=1.6%) to avoid divergent transitions
4. **Standard reference**: This is the canonical model for this exact scenario, allowing comparison with literature

**What causal story does this assume?**

- Schools are exchangeable samples from a common superpopulation
- True effects theta_i vary around population mean mu with SD tau
- Observed effects y_i are noisy measurements of theta_i
- No systematic differences between schools beyond random variation

### Prior Justification

**mu ~ Normal(0, 50)**:
- Centered at 0 (no prior belief in positive or negative effect)
- SD=50 allows for effects up to ±100 at 2σ, which covers the observed range (-5 to 26)
- With n=8 schools, data will dominate this prior
- More conservative than half-Cauchy(0, 25) recommended in BDA3, deliberately avoiding overconfidence

**tau ~ half-Cauchy(0, 10)**:
- Scale=10 instead of standard 25, reflecting belief that heterogeneity is likely low
- Still allows tau up to 30+ if data demand it (heavy tails)
- Regularizes toward 0 when evidence is weak
- Performs well with small number of groups (n=8)
- **Alternative**: half-Normal(0, 10) if stronger regularization desired

**Why not uniform or inverse-gamma?**
- Uniform(0, ∞) is improper and can cause computational issues
- Inverse-Gamma struggles when true tau is near 0 (as suspected here)
- half-Cauchy is modern best practice (Gelman 2006)

### Expected Behavior If Model Is Correct

**Posterior predictions**:
1. **tau posterior**: Expect posterior mode near 0-5, with substantial probability mass at very low values
2. **Strong shrinkage**: School effects theta_i will shrink heavily toward mu
   - School 4 (y=25.7) shrinks down to ~12-15
   - School 5 (y=-4.9) shrinks up to ~8-12
   - School 8 (sigma=18) experiences most shrinkage
3. **Narrow credible intervals**: Posterior intervals for theta_i will be much narrower than y_i ± 2*sigma_i due to borrowing strength
4. **mu posterior**: Centered around 8-12 (between unweighted mean 12.5 and weighted mean 10.0)
5. **Posterior predictive**: Generated y_rep should match observed y in distribution

**Convergence**: Should converge easily with non-centered parameterization. R-hat < 1.01, ESS > 400 per chain.

### Falsification Criteria: I Will Abandon This Model If...

**1. Posterior-prior conflict on tau**:
   - If posterior of tau is pushed far from 0 (e.g., posterior mode > 15) despite regularizing prior
   - **Interpretation**: Data strongly demand heterogeneity that prior fights against
   - **Action**: Switch to Model 2 (mixture) or investigate outliers

**2. Posterior predictive checks fail systematically**:
   - If replicated data y_rep consistently underestimate variance
   - If test statistics (e.g., max-min range) from y_rep don't cover observed
   - **Interpretation**: Normal model inadequate, or tau prior too restrictive
   - **Action**: Consider t-distribution (robustness) or Model 3 (sigma uncertainty)

**3. Strong divergent transitions or non-convergence**:
   - If >1% divergent transitions after tuning
   - If R-hat > 1.05 for key parameters
   - **Interpretation**: Model geometry problematic, likely tau near boundary
   - **Action**: Try centered parameterization, or extremely informative tau prior

**4. School 5 remains extreme outlier**:
   - If School 5's posterior theta_5 is >3 SD away from mu
   - If excluding School 5 drastically changes inference
   - **Interpretation**: School 5 is not exchangeable with others
   - **Action**: Model 2 (mixture) or separate model for School 5

**5. LOO-CV indicates poor predictive performance**:
   - If Pareto-k values > 0.7 for multiple schools
   - If ELPD_loo is worse than complete pooling baseline
   - **Interpretation**: Model not capturing individual school behavior
   - **Action**: Reconsider pooling assumptions

**6. Extreme shrinkage produces implausible results**:
   - If all theta_i collapse to within ±2 of mu despite sigma_i being known
   - If shrinkage factors exceed 80% for precise schools (5, 2)
   - **Interpretation**: Over-pooling destroying genuine information
   - **Action**: Consider fixed-effects model or investigate why tau→0

### What Would Make Me Switch Model Classes Entirely?

- **Evidence of discrete subgroups**: If two schools cluster far from others
- **Non-normal residuals**: If (y_i - theta_i)/sigma_i show skewness or heavy tails
- **Sigma_i are suspect**: If posterior predictive suggests sigma_i are too small
- **Prior sensitivity is extreme**: If conclusions flip with minor prior changes

---

## Model 2: Two-Component Mixture Model (Subgroup Hypothesis)

### Mathematical Specification

```
Likelihood:     y_i ~ Normal(theta_i, sigma_i)
Mixture:        theta_i ~ pi * Normal(mu_1, tau_1) + (1-pi) * Normal(mu_2, tau_2)
Hyperpriors:    mu_1 ~ Normal(0, 50)
                mu_2 ~ Normal(0, 50)
                tau_1 ~ half-Normal(0, 10)
                tau_2 ~ half-Normal(0, 10)
                pi ~ Beta(2, 2)              [weakly informative, symmetric]
```

**Alternative formulation** (discrete mixture):
```
Likelihood:     y_i ~ Normal(theta_i, sigma_i)
School level:   theta_i ~ Normal(mu[z_i], tau[z_i])
Cluster:        z_i ~ Categorical(pi)        [z_i ∈ {1, 2}]
Hyperpriors:    mu[k] ~ Normal(0, 50)        for k=1,2
                tau[k] ~ half-Normal(0, 10)
                pi ~ Dirichlet([2, 2])       [symmetric prior]
```

### PyMC Implementation Outline

```python
import pymc as pm

with pm.Model() as mixture_model:
    # Hyperpriors for two components
    mu_1 = pm.Normal('mu_1', mu=0, sigma=50)
    mu_2 = pm.Normal('mu_2', mu=0, sigma=50)
    tau_1 = pm.HalfNormal('tau_1', sigma=10)
    tau_2 = pm.HalfNormal('tau_2', sigma=10)

    # Mixture weight
    pi = pm.Beta('pi', alpha=2, beta=2)

    # Mixture likelihood for theta_i
    theta = pm.Mixture('theta',
                       w=pm.math.stack([pi, 1-pi]),
                       comp_dists=pm.Normal.dist(mu=[mu_1, mu_2],
                                                  sigma=[tau_1, tau_2]),
                       shape=8)

    # Observed data
    y = pm.Normal('y', mu=theta, sigma=sigma_data, observed=y_data)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=theta, sigma=sigma_data, shape=8)

    # Sample
    trace = pm.sample(2000, tune=2000, target_accept=0.95)
```

### Theoretical Rationale

**Why consider a mixture model?**

1. **School 5 is qualitatively different**: Only negative effect, high precision, pulls weighted mean down
2. **Runs test was significant (p=0.022)**: Suggests non-random clustering
3. **Bimodality hypothesis**: Perhaps two types of implementations (high fidelity vs. low fidelity)
4. **Explains variance paradox**: Within-cluster homogeneity could produce observed < expected variance

**What causal story does this assume?**

- Schools belong to one of two latent subgroups (e.g., urban vs. rural, strict vs. flexible implementation)
- Within each subgroup, schools are exchangeable with means mu_1, mu_2
- Subgroup membership is unknown but can be inferred
- Variance paradox arises because within-cluster variance is low, even if between-cluster variance exists

**When would this be scientifically plausible?**

- If schools differ in implementation quality/fidelity
- If treatment works differently for different populations
- If there's an unobserved moderator variable

### Prior Justification

**mu_1, mu_2 ~ Normal(0, 50)**:
- Identical priors enforce exchangeability a priori
- Posterior will separate them if data demand it
- No assumption about which component is "high" or "low"

**tau_1, tau_2 ~ half-Normal(0, 10)**:
- half-Normal instead of half-Cauchy for stronger regularization (avoid degenerate clusters)
- Expect within-cluster variation to be small
- If tau_k → 0, cluster k is homogeneous

**pi ~ Beta(2, 2)**:
- Symmetric prior centered at 0.5
- Weakly informative: allows unbalanced clusters but doesn't force them
- More informative than Beta(1,1) (uniform), which can lead to label switching
- Expected cluster sizes: 4-4, but allows 1-7 if data demand

**Alternative**: Dirichlet process prior for unknown number of clusters, but with n=8 this is likely overkill.

### Expected Behavior If Model Is Correct

**If two distinct clusters exist**:

1. **Posterior of pi**: Should be clearly away from 0.5 (e.g., 0.3 or 0.7)
2. **Posterior separation**: mu_1 and mu_2 should have non-overlapping 95% credible intervals
3. **Cluster assignment**: Posterior probabilities P(z_i = k | data) should be near 0 or 1 for most schools
4. **Candidate clusters**:
   - **Cluster 1** (high effects): Schools 1, 2, 3, 4 (all positive, moderate to large)
   - **Cluster 2** (low/negative effects): Schools 5, 6, 7, 8 (small or negative)
5. **Model comparison**: WAIC or LOO should favor mixture over single-component if subgroups are real

**If clusters do NOT exist**:

1. **Label switching**: MCMC will alternate between mu_1 > mu_2 and mu_2 > mu_1
2. **Collapsed clusters**: mu_1 ≈ mu_2 in posterior
3. **Posterior of pi**: Remains near 0.5 with high uncertainty
4. **Model comparison**: Single-component model (Model 1) will have better WAIC/LOO

### Falsification Criteria: I Will Abandon This Model If...

**1. Clusters collapse (mu_1 ≈ mu_2)**:
   - If posterior distributions of mu_1 and mu_2 overlap >80%
   - If |E[mu_1 - mu_2 | data]| < 5
   - **Interpretation**: No evidence for subgroups
   - **Action**: Revert to Model 1 (single hierarchy)

**2. Severe label switching**:
   - If trace plots show constant flipping between mu_1 > mu_2 and mu_2 > mu_1
   - If posterior summaries are nonsensical due to multimodality
   - **Interpretation**: Model identification problem, clusters not real
   - **Action**: Use ordered constraint (mu_1 < mu_2) or abandon mixture

**3. Poor predictive performance**:
   - If WAIC or LOO is worse than Model 1 by >5 units
   - If Pareto-k diagnostic suggests overfitting
   - **Interpretation**: Added complexity not justified by data
   - **Action**: Occam's razor - use simpler Model 1

**4. Cluster assignments are ambiguous**:
   - If most schools have P(z_i = 1 | data) near 0.5 (uncertain assignment)
   - If cluster membership changes drastically with minor prior tweaks
   - **Interpretation**: Data don't support discrete clustering
   - **Action**: Consider continuous covariate model instead of discrete mixture

**5. Biologically/practically implausible clusters**:
   - If clusters don't correspond to any known school characteristics
   - If clusters split in scientifically nonsensical ways (e.g., mix high and low sigma)
   - **Interpretation**: Spurious pattern, not real subgroups
   - **Action**: Revert to Model 1

**6. Computational failure**:
   - If sampler gets stuck, divergent transitions, or non-convergence despite tuning
   - If sampling takes >10x longer than Model 1 without clear benefit
   - **Interpretation**: Model too complex for data
   - **Action**: Simplify or abandon

### What Would Make Me Switch Model Classes Entirely?

- **All clusters collapse to mu_1 = mu_2 = 0**: Switch to complete pooling (no hierarchy)
- **Evidence that sigma_i are misspecified**: Switch to Model 3
- **Non-normal residuals**: Consider t-distribution or robust model
- **External information**: If we learn school characteristics, switch to regression model

---

## Model 3: Uncertainty in Sigma Model (Measurement Error Hypothesis)

### Mathematical Specification

```
Likelihood:     y_i ~ Normal(theta_i, sigma_true_i)     [sigma_true unknown!]
Sigma model:    sigma_true_i ~ half-Normal(sigma_i, epsilon)
School level:   theta_i ~ Normal(mu, tau)
Hyperpriors:    mu ~ Normal(0, 50)
                tau ~ half-Cauchy(0, 10)
                epsilon ~ half-Normal(0, 5)  [meta-uncertainty in reported SEs]
```

**Alternative formulation** (multiplicative error):
```
Likelihood:     y_i ~ Normal(theta_i, sigma_i * lambda_i)
Lambda:         lambda_i ~ Lognormal(0, epsilon)       [lambda_i > 0]
School level:   theta_i ~ Normal(mu, tau)
Hyperpriors:    mu ~ Normal(0, 50)
                tau ~ half-Cauchy(0, 10)
                epsilon ~ half-Normal(0, 0.2)   [lambda_i ≈ 1 a priori]
```

### Stan Implementation Outline

```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma_reported;  // "known" but potentially uncertain
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
  vector<lower=0>[J] lambda;          // multiplicative error on sigma
  real<lower=0> epsilon;              // meta-uncertainty
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
  vector[J] sigma_true = sigma_reported .* lambda;  // element-wise multiply
}
model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 10);
  theta_raw ~ std_normal();
  epsilon ~ normal(0, 0.2);
  lambda ~ lognormal(0, epsilon);  // lambda centered at 1

  // Likelihood with uncertain sigma
  y ~ normal(theta, sigma_true);
}
generated quantities {
  vector[J] y_rep;
  real mean_lambda = exp(epsilon^2 / 2);  // E[Lognormal(0, epsilon)]

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma_true[j]);
  }
}
```

### Theoretical Rationale

**Why consider sigma_i as uncertain?**

1. **Variance paradox**: Observed variance < expected could mean reported sigma_i are overestimates
2. **Unknown sample sizes**: We don't know n_i for each school - sigma_i might have estimation error
3. **Meta-analytic context**: In real meta-analyses, reported SEs often have errors or are approximations
4. **Explains strong shrinkage**: If true sigmas are smaller, less shrinkage is needed

**What causal story does this assume?**

- Original studies reported sigma_i, but these are estimates, not truth
- True sampling uncertainty sigma_true_i differs from reported sigma_i
- If sigma_i are overestimated, observed variance paradox is explained
- Schools are exchangeable once we account for measurement error in reported SEs

**When would this be scientifically plausible?**

- If studies used different SE estimation methods (delta method, bootstrap, etc.)
- If some studies had complex designs where simple SE formula is approximate
- If there's rounding or reporting error in published SEs
- If original papers had typos or used different uncertainty measures (SE vs. SD)

### Prior Justification

**epsilon ~ half-Normal(0, 0.2)**:
- Controls variation in lambda_i around 1
- epsilon = 0.2 implies lambda_i typically within [0.82, 1.22] (±1σ in log scale)
- Allows up to ±50% error in reported sigma_i at 2σ
- **Rationale**: Reported SEs in meta-analyses are usually approximately correct, not wildly wrong

**lambda_i ~ Lognormal(0, epsilon)**:
- Centered at 1: E[lambda_i] ≈ 1 (no systematic bias)
- Ensures lambda_i > 0 (SEs must be positive)
- Symmetric on log scale: equal chance of over/underestimation
- As epsilon→0, model converges to Model 1 (sigma_i known exactly)

**Why multiplicative rather than additive error?**
- SEs are inherently positive, multiplicative error respects this
- Relative error (±20%) more plausible than absolute error (±5 units) which depends on scale
- Lognormal is standard for positive-valued error terms

### Expected Behavior If Model Is Correct

**If reported sigma_i are overestimates (lambda_i < 1 on average)**:

1. **Posterior of epsilon**: Should be clearly away from 0 (e.g., posterior mode 0.1-0.3)
2. **Posterior of lambda_i**:
   - Schools with large residuals (|y_i - mu|) should have lambda_i > 1 (underestimated sigma)
   - Schools with small residuals should have lambda_i < 1 (overestimated sigma)
3. **Variance resolution**: Posterior predictive variance should match observed better than Model 1
4. **Less shrinkage**: theta_i should be closer to y_i than in Model 1, because true sigmas are smaller
5. **Model comparison**: WAIC/LOO should favor this model if sigma uncertainty is real

**If reported sigma_i are accurate (lambda_i ≈ 1)**:

1. **Posterior of epsilon**: Concentrated near 0
2. **Posterior of lambda_i**: Tight around 1 for all schools
3. **Model converges to Model 1**: Inference on mu, tau, theta_i nearly identical
4. **Model comparison**: Model 1 will be favored (simpler, no cost for extra parameters)

### Falsification Criteria: I Will Abandon This Model If...

**1. Posterior of epsilon collapses to 0**:
   - If posterior 95% CI for epsilon is [0, 0.05]
   - If all lambda_i posteriors tightly concentrated at 1
   - **Interpretation**: No evidence for sigma uncertainty, reported SEs are accurate
   - **Action**: Revert to Model 1

**2. Lambda_i posteriors are implausible**:
   - If lambda_i > 2 or < 0.5 for multiple schools (implies 2x error in reported SE)
   - If lambda pattern correlates with sigma_i (suggests model misspecification)
   - **Interpretation**: Model is fitting noise or structural problem exists
   - **Action**: Reconsider prior on epsilon, or abandon model

**3. No improvement in posterior predictive checks**:
   - If variance of y_rep still doesn't match observed variance
   - If adding sigma uncertainty doesn't resolve variance paradox
   - **Interpretation**: Problem is not in sigma_i, but in model structure
   - **Action**: Consider Model 2 (mixture) or non-normal likelihood

**4. Computational issues**:
   - If >5% divergent transitions due to complex geometry
   - If sampling efficiency is very low (ESS < 100 per chain)
   - **Interpretation**: Model is overparameterized for n=8
   - **Action**: Fix epsilon at a value or revert to Model 1

**5. Model comparison clearly favors simpler model**:
   - If WAIC/LOO is worse than Model 1 by >10 units
   - If effective number of parameters suggests overfitting
   - **Interpretation**: Complexity not justified
   - **Action**: Use Model 1

**6. Prior sensitivity is extreme**:
   - If changing epsilon prior from half-Normal(0, 0.2) to half-Normal(0, 0.3) completely changes conclusions
   - **Interpretation**: Data too weak to inform this parameter
   - **Action**: Fix sigma_i as known (Model 1)

### What Would Make Me Switch Model Classes Entirely?

- **Evidence sigma_i are systematically biased (not just noisy)**: Add bias parameter to sigma model
- **Correlation between lambda_i and school characteristics**: Switch to regression on sigma
- **Non-normal residuals even with sigma adjustment**: Robust likelihood (t-distribution)
- **Complete failure to resolve variance paradox**: Question the entire modeling framework

---

## Model Comparison Strategy

### Computational Approach

**For all models**:
1. Run 4 chains, 2000 warmup + 2000 sampling iterations
2. Target accept rate 0.95 for Model 1, 0.99 for Models 2-3 (more complex)
3. Check convergence: R-hat < 1.01, ESS > 400 per chain
4. Save posterior samples for all parameters
5. Generate posterior predictive samples (y_rep)

**Model-specific tuning**:
- Model 1: Non-centered parameterization essential
- Model 2: Monitor for label switching, may need ordering constraint
- Model 3: May need tighter priors on epsilon if non-identified

### Model Selection Criteria

**1. Leave-One-Out Cross-Validation (LOO-CV)**:
```
Compute ELPD_loo for each model
Select model with highest ELPD_loo
If difference < 4 units, models are equivalent
```

**2. Widely Applicable Information Criterion (WAIC)**:
```
Lower WAIC = better predictive performance
Penalizes complexity via effective number of parameters
```

**3. Pareto-k diagnostic**:
```
k < 0.5: reliable LOO estimate
k > 0.7: influential observation, refit needed
```

**4. Posterior predictive checks**:
```
Test statistics:
- Variance of effects: Var(y_rep) vs Var(y)
- Range: max(y_rep) - min(y_rep) vs observed
- Number of "significant" effects: sum(|y_rep/sigma| > 2)
- Homogeneity test: χ² statistic from y_rep
```

**5. Prior sensitivity**:
```
For each model, vary key priors by 2x:
- Scale of tau prior: 10 vs 20
- Scale of mu prior: 50 vs 100
- Model 2: Beta(2,2) vs Beta(1,1) on pi
- Model 3: half-Normal(0, 0.2) vs 0.3 on epsilon

If conclusions flip, model is weakly identified
```

### Decision Rules

**Model 1 is best if**:
- Lowest LOO/WAIC
- Posterior predictive checks pass
- No evidence for subgroups (Model 2 collapses)
- No evidence for sigma uncertainty (Model 3 collapses)
- **Action**: Report Model 1 as final, conduct sensitivity analysis

**Model 2 is best if**:
- LOO/WAIC substantially better than Model 1 (>4 units)
- Clear cluster separation (mu_1 and mu_2 distinct)
- Cluster assignments have high posterior probability
- Scientifically interpretable clusters
- **Action**: Investigate what distinguishes clusters, report mixture model

**Model 3 is best if**:
- LOO/WAIC better than Model 1
- Posterior predictive checks show improved variance matching
- lambda_i posteriors clearly away from 1
- Plausible pattern in which sigmas are mis-estimated
- **Action**: Report corrected sigmas, caveat about measurement error

**No model is adequate if**:
- All models fail posterior predictive checks on same test statistic
- High Pareto-k values (>0.7) for multiple schools across all models
- Extreme prior sensitivity in all models
- **Action**: Consider fundamentally different model class (robust t-likelihood, non-exchangeable models, covariate models)

---

## Stress Tests and Red Flags

### Stress Test 1: Exclude School 5 (Negative Outlier)

**Procedure**: Refit all three models with n=7 schools (drop School 5)

**Predictions**:
- Model 1: tau should decrease, mu should increase (School 5 was pulling both down)
- Model 2: If School 5 was defining a cluster, mixture should collapse to single component
- Model 3: lambda_5 should be far from 1 if School 5's sigma_5=9 is misspecified

**Red flag**: If excluding School 5 drastically changes inference on remaining 7 schools, model is not robust. Consider:
- School 5 is not exchangeable (needs separate model)
- Influential point diagnostic suggests poor model fit
- Need robust likelihood (t-distribution)

### Stress Test 2: Exclude School 8 (Highest Uncertainty)

**Procedure**: Refit all models with n=7 schools (drop School 8, sigma=18)

**Predictions**:
- Model 1: Should have minimal impact (School 8 is low precision, contributes little)
- Model 2: No impact on cluster structure
- Model 3: lambda_8 should test if sigma_8=18 is accurate

**Red flag**: If excluding School 8 improves model fit substantially, sigma_8 may be misspecified or School 8 is qualitatively different.

### Stress Test 3: Complete Pooling Baseline

**Procedure**: Fit trivial model theta_i = mu for all i, with mu ~ Normal(0, 50)

**Purpose**: Establish lower bound on model performance

**Predictions**:
- LOO/WAIC should be worse than all hierarchical models
- If complete pooling is competitive, tau ≈ 0 in Model 1

**Red flag**: If complete pooling has better LOO/WAIC than hierarchical models, either:
- Heterogeneity is truly zero (rare)
- Hierarchical models are overparameterized
- Priors on tau are too restrictive

### Stress Test 4: Posterior Predictive Permutation Test

**Procedure**:
1. Generate 1000 datasets from posterior predictive of each model
2. For each dataset, compute I² statistic
3. Compare distribution of I²_rep to observed I²=1.6%

**Predictions**:
- Good model: observed I² within central 95% of I²_rep distribution
- Model 1: If tau→0, I²_rep should be concentrated near 0-5%

**Red flag**: If observed I²=1.6% is in tail of I²_rep distribution (p < 0.05), model is not generating data like what we observed.

### Stress Test 5: Prior Predictive Checks

**Procedure**: Sample from prior (before seeing data), generate datasets

**Purpose**: Ensure priors are not accidentally informative

**Check**:
- Do prior samples include effects ranging from -50 to +50? (mu prior)
- Do prior samples include tau from 0 to 50? (tau prior)
- Are prior predictive effects plausible given context?

**Red flag**: If prior generates only implausible datasets (e.g., all effects >100), priors are too diffuse or mis-specified.

---

## Red Flags That Trigger Major Strategy Pivots

### Red Flag 1: All Models Show Posterior-Prior Conflict

**Symptom**: Posteriors pushed against prior boundaries across all models

**Interpretation**: Fundamental model class issue, not just prior choice

**Actions**:
1. Consider non-normal likelihood (Student-t with estimated df)
2. Consider non-exchangeable models (fixed effects)
3. Question whether sigma_i are on correct scale (maybe reported as SD not SE?)

### Red Flag 2: Posterior Predictive Variance Consistently Wrong

**Symptom**: All models generate y_rep with variance 2x or 0.5x observed variance

**Interpretation**: Structural problem with likelihood or data generation process

**Actions**:
1. Model 3 should have addressed this - if it didn't, sigma_i may be fundamentally wrong
2. Consider that y_i might not be approximately normal (check original data sources)
3. Consider extra-binomial variation model (if underlying data are proportions)

### Red Flag 3: Computational Failure Across All Models

**Symptom**: Divergent transitions, non-convergence, or extreme sampling inefficiency in all models

**Interpretation**: Geometry of posterior is pathological, likely due to weak identification

**Actions**:
1. Add informative priors based on domain knowledge (e.g., tau < 20 is certain)
2. Reparameterize: consider modeling theta_i directly without hierarchy
3. Use Variational Inference as sanity check (faster, may reveal issues)

### Red Flag 4: Extreme Prior Sensitivity

**Symptom**: Doubling prior scale changes posterior means by >50%

**Interpretation**: Data are too weak to overcome prior, n=8 is insufficient for these models

**Actions**:
1. Acknowledge fundamental uncertainty - report full posterior distributions, not just point estimates
2. Consider this an "inference with small data" problem - be very cautious
3. Use skeptical priors (e.g., tau ~ half-Normal(0, 5)) and report sensitivity

### Red Flag 5: LOO Pareto-k > 0.7 for Multiple Schools

**Symptom**: Leave-one-out diagnostic suggests overfitting or influential points

**Interpretation**: Model is not capturing individual school behavior well

**Actions**:
1. Consider school-specific features (if data available) - move to regression model
2. Acknowledge heterogeneity beyond simple hierarchy
3. Report individual school posteriors with large uncertainty

---

## Alternative Models to Pivot To (If Needed)

### Escape Route 1: Student-t Likelihood (Robust Model)

**When**: If posterior predictive checks show model can't capture outliers

```
Likelihood:    y_i ~ Student_t(nu, theta_i, sigma_i)
Hyperprior:    nu ~ Gamma(2, 0.1)   [df parameter, heavy tails if nu < 10]
```

**Purpose**: Downweight outliers automatically without excluding data

### Escape Route 2: Beta-Binomial Meta-Regression

**When**: If we learn y_i are derived from binary outcomes (success rates)

```
Latent:        y_i ~ Beta(alpha_i, beta_i)
               alpha_i = p_i * n_i, beta_i = (1-p_i) * n_i
Regression:    logit(p_i) = theta_i
```

### Escape Route 3: Fixed Effects Model (No Pooling)

**When**: If all hierarchical models fail and schools appear truly independent

```
Each theta_i ~ Normal(0, 50) independently
No connection between schools
```

**Warning**: This abandons the entire hierarchical framework - only use as last resort.

### Escape Route 4: Empirical Bayes (Quick Approximation)

**When**: If full Bayes is too computationally expensive or unstable

**Procedure**:
1. Estimate tau via method of moments or REML
2. Plug tau_hat into shrinkage formula
3. Compute shrunk estimates theta_i

**Caveat**: Does not propagate uncertainty in tau, underestimates posterior variance.

---

## Decision Points and Stopping Rules

### Decision Point 1: After Model 1 Fitting

**Questions**:
1. Does Model 1 converge cleanly? (R-hat < 1.01, no divergences)
2. Does posterior predictive check pass? (variance, range, normality)
3. Is tau posterior clearly away from 0 or near 0?

**Decisions**:
- If tau ≈ 0 and PPCs pass: Model 1 likely sufficient, fit Models 2-3 for comparison but expect collapse
- If tau > 10 and PPCs fail: Proceed immediately to Model 2 (mixture) and Model 3 (sigma uncertainty)
- If computational issues: Debug parameterization before proceeding

### Decision Point 2: After Fitting All Three Models

**Questions**:
1. Do models agree on mu within ±2 units?
2. Does model comparison clearly favor one model (ΔLOO > 4)?
3. Do all models pass same posterior predictive checks?

**Decisions**:
- If models agree and Model 1 is simplest: Report Model 1 as primary, others as sensitivity
- If Model 2 is clearly best: Investigate cluster interpretation, report mixture model
- If Model 3 is best: Report sigma uncertainty, caveat about measurement error
- If no model is clearly best: Report model averaging or acknowledge model uncertainty

### Decision Point 3: After Stress Tests

**Questions**:
1. Does excluding School 5 change conclusions?
2. Does complete pooling baseline perform competitively?
3. Do posterior predictive permutation tests pass?

**Decisions**:
- If School 5 is driving results: Consider sensitivity analysis or separate model for School 5
- If complete pooling is competitive: Strong evidence for homogeneity, report complete pooling with caveat
- If stress tests fail: Reconsider entire modeling approach

### Stopping Rule 1: Convergence Failure

**When to stop**: If after 3 reparameterizations and prior adjustments, models still don't converge

**Action**: Report that data are insufficient for Bayesian hierarchical modeling, use simpler methods (empirical Bayes, frequentist REML)

### Stopping Rule 2: All Models Fail Same PPC

**When to stop**: If variance, range, or normality check fails for all models consistently

**Action**: Problem is with fundamental assumptions (normal likelihood, exchangeability). Consider non-normal models or fixed effects.

### Stopping Rule 3: Extreme Prior Sensitivity Persists

**When to stop**: If after trying 5 different prior specifications, conclusions vary wildly

**Action**: Acknowledge that n=8 is too small for robust inference. Report full range of posterior estimates under different priors, emphasize uncertainty.

---

## Summary of Model Proposals

| Aspect | Model 1: Hierarchical | Model 2: Mixture | Model 3: Sigma Uncertainty |
|--------|----------------------|------------------|---------------------------|
| **Assumption** | Schools exchangeable | Two latent subgroups | Reported sigma_i uncertain |
| **Parameters** | mu, tau, theta_i (10 total) | mu_1, mu_2, tau_1, tau_2, pi, z_i (14 total) | mu, tau, theta_i, lambda_i, epsilon (19 total) |
| **Complexity** | Simple | Moderate | High |
| **Expected if correct** | tau ≈ 0, strong shrinkage | Cluster separation | lambda_i away from 1 |
| **Falsification** | tau > 15, PPC fail | Clusters collapse | epsilon → 0 |
| **Best if** | Homogeneity is real | Subgroups exist | Measurement error in sigma |
| **Pivot if** | Outliers dominate | Label switching | Overparameterization |

### Recommended Workflow

1. **Start with Model 1** (standard hierarchical)
   - Most likely to be adequate given EDA
   - If it passes all checks, done

2. **Fit Model 2** (mixture) if:
   - Model 1 PPCs fail on multimodality tests
   - School 5 remains extreme outlier
   - Scientific reason to suspect subgroups

3. **Fit Model 3** (sigma uncertainty) if:
   - Variance paradox persists in Model 1 posterior predictive
   - Domain knowledge suggests reported SEs may be approximate
   - Model 1 shrinkage seems excessive

4. **Compare via LOO/WAIC**
   - If ΔLOO < 4: Models are equivalent, report simplest (Model 1)
   - If ΔLOO > 4: Report best model, use others for sensitivity

5. **Conduct stress tests** on best model

6. **Report final model** with:
   - Posterior distributions for all parameters
   - Posterior predictive checks showing model adequacy
   - Sensitivity analysis for key priors
   - Clear statement of assumptions and limitations

---

## Expected Outcomes and Scientific Interpretation

### If Model 1 is adequate (most likely given EDA):

**Finding**: tau posterior concentrated near 0-5, mu ≈ 10-12

**Interpretation**: Treatment effects are homogeneous across schools, apparent variation is due to sampling error

**Scientific implication**: Treatment works similarly regardless of school context, one overall effect estimate is justified

**Recommendation**: Report mu as best estimate of treatment effect, theta_i as school-specific estimates incorporating pooling

### If Model 2 is best (mixture):

**Finding**: Two clusters, e.g., mu_1 ≈ 20, mu_2 ≈ 5, with clear separation

**Interpretation**: Schools fall into two categories with different effect sizes

**Scientific implication**: Treatment works well in some contexts (high-fidelity implementation?) but less well in others

**Recommendation**: Investigate what distinguishes clusters - implementation quality? Student population? Resources?

### If Model 3 is best (sigma uncertainty):

**Finding**: lambda_i vary systematically, epsilon ≈ 0.2, implies ±20% error in reported SEs

**Interpretation**: Published standard errors were approximate or estimated with error

**Scientific implication**: Effect estimates y_i are noisier than originally thought, more uncertainty in conclusions

**Recommendation**: Report adjusted sigmas, caveat that original study SEs may not be reliable

---

## Conclusion

I have proposed three **fundamentally different** Bayesian model classes, each representing a distinct hypothesis about the data generation process:

1. **Model 1** assumes exchangeability and homogeneity (standard)
2. **Model 2** assumes latent subgroups (mixture)
3. **Model 3** assumes measurement error in reported uncertainties

Each model has:
- Clear mathematical specification
- Justifiable priors
- Explicit falsification criteria
- Expected behaviors if correct
- Decision rules for abandoning

**Key philosophical point**: I expect Model 1 to be adequate given the EDA findings (I²=1.6%), but I've designed Models 2 and 3 to test whether this conclusion could be wrong. If it's wrong, I have explicit escape routes.

**Falsification mindset**: Success means discovering which model is WRONG early, not completing all three. If Model 1 passes all checks, I will not force Models 2-3 to be used. If all models fail, I will pivot to alternative model classes (t-likelihood, fixed effects, etc.).

**Final output**: The model that best explains the data, with transparent reporting of how I reached that conclusion and what alternative hypotheses were tested and rejected.

---

## File Locations

- This proposal: `/workspace/experiments/designer_2/proposed_models.md`
- EDA report: `/workspace/eda/eda_report.md`
- Data: `/workspace/data/data.csv`

**Next steps**: Implement these models in Stan/PyMC, fit to data, compare via LOO/WAIC, conduct posterior predictive checks, report results.
