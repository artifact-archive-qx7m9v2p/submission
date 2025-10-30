# Model Design Proposal: Designer #2
## Fixed-Effects and Complete Pooling Perspective

**Designer**: Model Designer #2
**Date**: 2025-10-28
**Focus**: Fixed-effects models, complete pooling strategies
**Design Philosophy**: When is complete pooling justified? When should we abandon it?

---

## Executive Summary

Based on the EDA findings (I²=0%, Q p=0.696, complete CI overlap), I propose **three distinct Bayesian model classes** that take seriously the possibility of a **common underlying effect**. However, I approach this with deep skepticism: **I²=0% in a small sample (J=8) is weak evidence for homogeneity**, and the "low heterogeneity paradox" revealed in EDA suggests large measurement errors may be masking true variation.

**Core tension**: The data support complete pooling statistically, but the scientific plausibility is uncertain. All three models must be ready to fail.

**Proposed Model Classes**:
1. **Bayesian Fixed-Effect Meta-Analysis** (classical complete pooling)
2. **Precision-Stratified Fixed-Effect Model** (partial pooling by precision groups)
3. **Fixed-Effect Model with Measurement Error Uncertainty** (known SEs with epistemic uncertainty)

---

## Design Philosophy: The Falsificationist Stance

### What Would Make Me Abandon Fixed-Effects Models Entirely?

**Immediate red flags** (any one triggers model class change):
1. **Posterior predictive failure**: Model systematically underpredicts variation in observed effects
2. **Leave-one-out instability**: Pooled estimate changes >5 units when removing single studies
3. **Precision-dependent residuals**: Systematic patterns in residuals vs. SE (heteroscedasticity beyond measurement error)
4. **Prior-data conflict**: Strong conflict (p < 0.01) suggesting model misspecification
5. **Subgroup divergence**: If we later obtain covariates showing clear effect moderation

**Moderate warnings** (combination triggers reconsideration):
- LOO-CV strongly favors random-effects model (Δelpd > 10)
- Study 1 (y=28, SE=15) has standardized posterior residual > 3
- Posterior probability of homogeneity P(τ=0 | data) < 0.3 in competing random-effects model
- Meta-regression (if covariates emerge) explains substantial variance

### When Is Complete Pooling Scientifically Justified?

**Supporting evidence** (all present in this dataset):
- I²=0% with non-significant Q test
- Complete overlap of all study confidence intervals
- No correlation between precision and effect size
- No obvious study-level heterogeneity sources

**Against complete pooling** (also present):
- Small sample (J=8) provides weak evidence for homogeneity
- Effect range (-3 to 28) spans 31 units
- Unweighted mean (8.75) > weighted mean (7.69) suggests heterogeneity signal
- Simulation shows same variation → I²=63% with 50% smaller SEs

**My stance**: Fixed-effects models are **conditionally appropriate** as a starting point but must be vigorously tested against alternatives.

---

## Model 1: Bayesian Fixed-Effect Meta-Analysis

### 1.1 Model Name and Class
**Bayesian Fixed-Effect Meta-Analysis with Known Measurement Errors**

**Model class**: Complete pooling, inverse-variance weighted estimation

**Key assumption**: All studies estimate the **same common true effect** (θ_i = μ for all i)

### 1.2 Mathematical Specification

**Likelihood**:
```
y_i ~ Normal(mu, sigma_i)    for i = 1, ..., 8
```

Where:
- `y_i`: Observed effect size for study i (data)
- `mu`: Common true effect (parameter to estimate)
- `sigma_i`: Known measurement error for study i (data, treated as fixed)

**Prior on mu**:
```
mu ~ Normal(0, 15)
```

**Complete model**:
```
# Likelihood
for (i in 1:8) {
  y[i] ~ normal(mu, sigma[i]);
}

# Prior
mu ~ normal(0, 15);
```

### 1.3 Prior Justification

**mu ~ Normal(0, 15)**:

**Rationale**:
1. **Empirically calibrated**: Observed effects range -3 to 28, SD=10.44
2. **Contains observed range**: 95% prior interval [-30, 30] covers observed range with room
3. **Weakly informative**: SD=15 is ~1.5× observed effect SD, allowing data dominance
4. **Reference scale**: Based on Gelman et al. (2008) recommendation for standardized effects
5. **Conservative center**: Centered at 0 (no effect) requires data to move us

**Why not broader (e.g., Normal(0, 50))?**
- With J=8, extremely vague priors can lead to poor geometry in posterior
- SE range [9, 18] means measurement precision is moderate; we need some regularization
- Weakly informative > uninformative for computational stability

**Why not narrower (e.g., Normal(0, 5))?**
- Would conflict with observed data (y=28 outside 5× prior SD)
- Would impose strong prior belief in small effects without scientific justification
- Dataset lacks context to justify strong priors

**Alternative priors to test in sensitivity analysis**:
- `Normal(0, 10)`: More informative, pulls toward null
- `Normal(0, 25)`: More diffuse, closer to reference analysis
- `Student_t(3, 0, 15)`: Heavier tails, robust to outliers in prior space
- `Normal(7.69, 15)`: Centered at frequentist estimate (data-dependent, for comparison only)

### 1.4 What This Model Captures

**Accounts for**:
1. **Known measurement heterogeneity**: Different σ_i weights studies appropriately
2. **Inverse-variance weighting**: Study 5 (σ=9) gets 4× weight of Study 8 (σ=18)
3. **Common effect hypothesis**: Assumes all variation is measurement error
4. **Uncertainty quantification**: Full posterior P(μ | data) vs. point estimate

**Does NOT account for**:
1. **True effect heterogeneity**: Assumes τ=0 (no between-study variation)
2. **Outliers**: No robust likelihood (assumes Normal errors)
3. **Publication bias**: No selection model
4. **Covariate effects**: No moderators

**Key insight**: This model is **maximally simple** and **maximally powerful** (if correct). It pools all information into a single parameter. If the homogeneity assumption is wrong, this model will fail visibly in posterior predictive checks.

### 1.5 Falsification Criteria

**This model is WRONG if any of the following hold**:

**Criterion 1: Posterior Predictive Failure**
- **Test**: Generate y_rep ~ Normal(mu_post, σ_i) for each posterior draw
- **Failure**: If observed y_i fall outside 95% posterior predictive interval for >1 study
- **Specific threshold**: Study 1 (y=28) should have p_post < 0.975 if model is correct
- **Action if failed**: Immediate switch to hierarchical model with τ > 0

**Criterion 2: Leave-One-Out Sensitivity**
- **Test**: Refit model 8 times, each time dropping one study
- **Failure**: If μ_(-i) changes by >4 units (50% of SE of pooled estimate)
- **Specific threshold**: Removing Study 1 should not change μ by >4
- **Action if failed**: Investigate robust fixed-effect model (Student-t likelihood)

**Criterion 3: Residual Patterns**
- **Test**: Standardized residuals r_i = (y_i - μ_post) / σ_i
- **Failure**: If |r_i| > 2 for any study, or if r_i correlates with σ_i (|ρ| > 0.5)
- **Action if failed**: Switch to random-effects model (Model 1 is too restrictive)

**Criterion 4: LOO-CV Comparison**
- **Test**: Compare LOO-elpd to random-effects model
- **Failure**: If Δelpd > 5 favoring random-effects (>2 SE difference)
- **Action if failed**: Adopt random-effects as primary model

**Criterion 5: Prior-Data Conflict**
- **Test**: Prior predictive P-value (Pr(y_rep more extreme than y_obs | prior))
- **Failure**: If p < 0.01 (strong conflict) suggesting model structural inadequacy
- **Action if failed**: Reconsider model structure entirely

**Criterion 6: Parsimony Failure**
- **Test**: WAIC comparison with random-effects model
- **Failure**: If random-effects has better WAIC despite extra parameter (Δ > 4)
- **Action if failed**: Fixed-effect model's simplicity does not justify its restrictions

### 1.6 Implementation Notes

**Platform**: Stan (via CmdStanPy) or PyMC

**Stan code**:
```stan
data {
  int<lower=1> J;              // number of studies
  vector[J] y;                 // observed effects
  vector<lower=0>[J] sigma;    // known SEs
}

parameters {
  real mu;                     // common effect
}

model {
  // Prior
  mu ~ normal(0, 15);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive
  vector[J] y_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    y_rep[j] = normal_rng(mu, sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | mu, sigma[j]);
  }
}
```

**PyMC code**:
```python
import pymc as pm
import numpy as np

with pm.Model() as fixed_effect_model:
    # Prior
    mu = pm.Normal('mu', mu=0, sigma=15)

    # Likelihood (sigma is data, not parameter)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=mu, sigma=sigma, shape=J)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)
```

**Sampling strategy**:
- 4 chains × 2000 iterations = 8000 posterior draws
- 1000 warmup iterations (tune)
- Target: ESS > 400 per parameter, R-hat < 1.01
- Expected: ESS > 4000 (simple model, single parameter)

**Computational considerations**:
- **Extremely fast**: Single parameter, no hierarchical structure
- **No convergence issues expected**: Posterior is nearly Normal
- **No identifiability concerns**: mu is directly identified by data
- **Memory efficient**: Minimal storage requirements

### 1.7 Expected Challenges

**Challenge 1: Model Too Simple**
- **Issue**: Fixed-effect model may be overly restrictive
- **Symptom**: Wide posterior predictive intervals but poor individual study coverage
- **Diagnostic**: Posterior predictive p-values < 0.05 for multiple studies
- **Solution**: Move to Model 2 (precision-stratified) or Designer #1's random-effects

**Challenge 2: Influential Study Dominance**
- **Issue**: Study 1 (y=28, high leverage) may overly influence μ
- **Symptom**: LOO diagnostics show high Pareto-k (>0.7) for Study 1
- **Diagnostic**: PSIS-LOO warnings
- **Solution**: Robust likelihood (Student-t) or outlier investigation

**Challenge 3: Weak Pooled Inference**
- **Issue**: Even with complete pooling, precision may be limited (J=8, large SEs)
- **Symptom**: Wide posterior credible interval for μ (e.g., [-5, 20])
- **Diagnostic**: Posterior SD > 5
- **Solution**: Expected and unavoidable; report honestly

**Challenge 4: Borderline Significance**
- **Issue**: Frequentist estimate p≈0.05; posterior may span zero
- **Symptom**: P(μ > 0) ≈ 0.5-0.6 (inconclusive)
- **Diagnostic**: Credible interval includes zero
- **Solution**: Report probability scale results, avoid dichotomous conclusions

**Challenge 5: Philosophical Objection**
- **Issue**: Fixed-effect model assumes perfect homogeneity, scientifically implausible
- **Symptom**: Scientific community rejects model on principle
- **Diagnostic**: N/A (external to statistical analysis)
- **Solution**: Present as boundary case; random-effects as primary if needed

---

## Model 2: Precision-Stratified Fixed-Effect Model

### 2.1 Model Name and Class
**Bayesian Fixed-Effect Meta-Analysis with Precision-Group Specific Effects**

**Model class**: Partial pooling by precision strata

**Key assumption**: Studies cluster into **precision groups** that may have different true effects, but within-group effects are homogeneous

### 2.2 Mathematical Specification

**Motivation**: EDA revealed bimodal SE distribution:
- High-precision group: σ ∈ [9, 11] (Studies 2, 4, 5, 6, 7) - 5 studies
- Low-precision group: σ ∈ [15, 18] (Studies 1, 3, 8) - 3 studies

**Research question**: Do high-precision studies estimate a different effect than low-precision studies?

**Likelihood**:
```
y_i ~ Normal(mu_g[i], sigma_i)    for i = 1, ..., 8
```

Where:
- `g[i] ∈ {1, 2}`: Group indicator (1=high-precision, 2=low-precision)
- `mu_g[i]`: Effect for the group containing study i

**Priors**:
```
mu_1 ~ Normal(0, 15)    # High-precision group effect
mu_2 ~ Normal(0, 15)    # Low-precision group effect
```

**Group assignment** (deterministic from data):
- Studies 2, 4, 5, 6, 7 → Group 1 (high-precision, σ ≤ 11)
- Studies 1, 3, 8 → Group 2 (low-precision, σ ≥ 15)

**Complete model**:
```
# Data
group[1] = 2, group[2] = 1, group[3] = 2, group[4] = 1,
group[5] = 1, group[6] = 1, group[7] = 1, group[8] = 2

# Likelihood
for (i in 1:8) {
  y[i] ~ normal(mu_group[group[i]], sigma[i]);
}

# Priors
mu_group[1] ~ normal(0, 15);  # High-precision effect
mu_group[2] ~ normal(0, 15);  # Low-precision effect
```

### 2.3 Prior Justification

**mu_g ~ Normal(0, 15) for both groups**:

**Rationale**:
1. **No prior information** distinguishes precision groups
2. **Identical priors**: Avoids bias toward either group
3. **Exchangeable**: Both groups get same prior precision
4. **Data-driven differentiation**: Let data determine if μ_1 ≠ μ_2

**Alternative: Hierarchical prior on group means**:
```
mu_g ~ Normal(mu_overall, tau_group)
mu_overall ~ Normal(0, 15)
tau_group ~ Half-Normal(0, 10)
```
This would partially pool group effects toward overall mean. **Trade-off**: More parameters (3 vs 2), better if groups truly differ moderately.

**Why this prior scale?**
- Same reasoning as Model 1: empirically calibrated to data scale
- Each group should be able to span observed effect range independently

### 2.4 What This Model Captures

**Accounts for**:
1. **Precision-stratified effects**: Allows high/low precision studies to differ
2. **Small-study effects**: If present, would manifest as μ_1 ≠ μ_2
3. **Known measurement structure**: Different σ_i within groups
4. **Multiple effect hypotheses**: Tests whether precision correlates with effect size

**Does NOT account for**:
1. **Within-group heterogeneity**: Assumes τ=0 within each group
2. **Continuous precision effects**: Treats precision categorically, not as continuous moderator
3. **Publication bias mechanisms**: No selection model
4. **Other study characteristics**: Only uses precision

**Key insight**: This model **relaxes complete pooling** slightly by allowing two effects. It tests a specific hypothesis: "Do large SE studies estimate different effects than small SE studies?" This is a **small-study effect** test in Bayesian framework.

### 2.5 Falsification Criteria

**This model is WRONG if any of the following hold**:

**Criterion 1: Group Effects Are Equivalent**
- **Test**: Posterior P(|μ_1 - μ_2| > 3 | data) - is difference substantial?
- **Failure**: If P(|μ_1 - μ_2| < 2 | data) > 0.95 (groups essentially identical)
- **Action if failed**: Revert to Model 1 (no benefit from stratification)
- **Success criterion**: Groups should differ by >5 units to justify added complexity

**Criterion 2: Within-Group Heterogeneity**
- **Test**: Posterior predictive checks within each group separately
- **Failure**: If studies within a group show systematic deviations from group mean
- **Specific**: Study 1 (Group 2, y=28) should match Group 2 mean reasonably
- **Action if failed**: Need random effects within groups (hierarchical structure)

**Criterion 3: Model Comparison Favors Simpler Model**
- **Test**: LOO-CV and WAIC comparison with Model 1
- **Failure**: If Model 1 has lower WAIC (simpler model preferred)
- **Threshold**: Need Δelpd > 3 to justify extra parameter
- **Action if failed**: Accept Model 1, stratification unnecessary

**Criterion 4: Continuous Precision Effects**
- **Test**: Plot posterior group means vs. group-average precision
- **Failure**: If residuals within groups correlate with σ_i (continuous, not categorical)
- **Action if failed**: Consider meta-regression with σ_i as continuous predictor

**Criterion 5: Arbitrary Threshold Sensitivity**
- **Test**: Re-run with different threshold (e.g., σ < 12 vs σ ≥ 12)
- **Failure**: If substantive conclusions change dramatically with cutpoint choice
- **Action if failed**: Continuous modeling (meta-regression) required

**Criterion 6: Group Imbalance**
- **Issue**: Group 2 has only 3 studies (Study 1, 3, 8)
- **Failure**: If μ_2 posterior is dominated by Study 1 (largest effect, moderate SE)
- **Diagnostic**: μ_2 ≈ Study 1's effect, ignoring Studies 3 and 8
- **Action if failed**: Too sensitive to single study; revert to Model 1 or random-effects

### 2.6 Implementation Notes

**Platform**: Stan or PyMC

**Stan code**:
```stan
data {
  int<lower=1> J;                      // number of studies (8)
  vector[J] y;                         // observed effects
  vector<lower=0>[J] sigma;            // known SEs
  int<lower=1, upper=2> group[J];      // group indicator
}

parameters {
  vector[2] mu_group;                  // group-specific effects
}

model {
  // Priors
  mu_group ~ normal(0, 15);

  // Likelihood
  for (j in 1:J) {
    y[j] ~ normal(mu_group[group[j]], sigma[j]);
  }
}

generated quantities {
  // Derived quantities
  real mu_diff = mu_group[1] - mu_group[2];      // difference between groups
  real mu_avg = (mu_group[1] + mu_group[2]) / 2; // average effect

  // Posterior predictive
  vector[J] y_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    y_rep[j] = normal_rng(mu_group[group[j]], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | mu_group[group[j]], sigma[j]);
  }
}
```

**PyMC code**:
```python
import pymc as pm

# Prepare group indicators
group_idx = np.array([1, 0, 1, 0, 0, 0, 0, 1])  # 0=high-prec, 1=low-prec

with pm.Model() as stratified_model:
    # Priors
    mu_group = pm.Normal('mu_group', mu=0, sigma=15, shape=2)

    # Likelihood
    mu_study = mu_group[group_idx]
    y_obs = pm.Normal('y_obs', mu=mu_study, sigma=sigma, observed=y)

    # Derived quantities
    mu_diff = pm.Deterministic('mu_diff', mu_group[0] - mu_group[1])

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)
```

**Group assignment**:
```
Study | SE  | Group | Group Name
------|-----|-------|-------------
  1   | 15  |   2   | Low-precision
  2   | 10  |   1   | High-precision
  3   | 16  |   2   | Low-precision
  4   | 11  |   1   | High-precision
  5   |  9  |   1   | High-precision
  6   | 11  |   1   | High-precision
  7   | 10  |   1   | High-precision
  8   | 18  |   2   | Low-precision
```

**Sampling strategy**:
- Same as Model 1: 4 chains × 2000 iterations
- Expected ESS: >2000 per parameter (2 parameters, simple)
- Expected R-hat: <1.01 (no convergence issues)

**Computational considerations**:
- **Still simple**: Only 2 parameters
- **No hierarchical complications**: Groups fixed, not random
- **Fast**: Similar speed to Model 1
- **Post-processing**: Compute group difference distribution

### 2.7 Expected Challenges

**Challenge 1: Group Imbalance**
- **Issue**: Group 1 (5 studies) vs Group 2 (3 studies) imbalanced
- **Impact**: μ_2 estimate less precise, wider credible interval
- **Diagnostic**: SD(μ_2) > 1.5 × SD(μ_1)
- **Solution**: Expected; report uncertainty honestly

**Challenge 2: Within-Group Heterogeneity**
- **Issue**: Study 1 (y=28) may not match Studies 3 (-3) and 8 (12) in Group 2
- **Impact**: Poor within-group fit for low-precision studies
- **Diagnostic**: Posterior predictive p-value < 0.05 for Study 1
- **Solution**: Random effects within groups or abandon stratification

**Challenge 3: No Group Difference**
- **Issue**: μ_1 ≈ μ_2, groups not actually different
- **Impact**: Model complexity not justified
- **Diagnostic**: P(|μ_1 - μ_2| > 2) < 0.6 (weak evidence for difference)
- **Solution**: Revert to Model 1 via LOO comparison

**Challenge 4: Arbitrary Categorization**
- **Issue**: Cutpoint at σ=12 is data-driven, not principled
- **Impact**: Results may be sensitive to cutpoint choice
- **Diagnostic**: Different cutpoint (e.g., σ=13) gives different μ_diff
- **Solution**: Sensitivity analysis across cutpoints; consider continuous model

**Challenge 5: Interpretation Complexity**
- **Issue**: Two effects harder to communicate than one overall effect
- **Impact**: Stakeholders may misinterpret as subgroup analysis
- **Diagnostic**: N/A (communication issue)
- **Solution**: Clear framing as precision-stratification sensitivity analysis

---

## Model 3: Fixed-Effect Model with Measurement Error Uncertainty

### 3.1 Model Name and Class
**Bayesian Fixed-Effect Meta-Analysis with Uncertain Measurement Errors**

**Model class**: Complete pooling with hierarchical uncertainty on known SEs

**Key assumption**: Reported standard errors σ_i are **estimates**, not exact known values. We have uncertainty about measurement precision itself.

### 3.2 Mathematical Specification

**Motivation**: EDA treats σ_i as "known" but in reality, these are estimated from individual studies. Each has uncertainty. Ignoring this creates false precision.

**Standard model** (Model 1):
```
y_i ~ Normal(mu, sigma_i)    # sigma_i treated as fixed data
```

**This model**:
```
y_i ~ Normal(mu, sigma_i_true)          # true unknown measurement error
sigma_i_true ~ Gamma(alpha_i, beta_i)   # uncertainty about true SE
```

Where `alpha_i, beta_i` parameterize measurement error uncertainty based on degrees of freedom (if available).

**Practical implementation** (if df unavailable, as in this dataset):

**Option A**: Add small hierarchical variance to measurement errors
```
y_i ~ Normal(mu, sqrt(sigma_i^2 + epsilon_i^2))
epsilon_i ~ Half-Normal(0, sigma_epsilon)
sigma_epsilon ~ Half-Normal(0, 2)
```

**Option B**: Scale-inflated measurement errors
```
y_i ~ Normal(mu, sigma_i * lambda)
lambda ~ LogNormal(0, 0.2)    # Multiplicative uncertainty factor
```

**I will implement Option B** for simplicity and interpretability.

**Complete model**:
```
# Likelihood
for (i in 1:8) {
  y[i] ~ normal(mu, sigma[i] * lambda);
}

# Priors
mu ~ normal(0, 15);
lambda ~ lognormal(0, 0.2);    # E[lambda]=1, SD ≈ 0.2
```

Where `lambda > 1` indicates SEs are underestimated, `lambda < 1` indicates overestimated.

### 3.3 Prior Justification

**mu ~ Normal(0, 15)**:
- Same as Models 1 and 2 (consistency across model suite)

**lambda ~ LogNormal(0, 0.2)**:

**Rationale**:
1. **Centered at 1**: Prior median is 1 (no systematic bias in reported SEs)
2. **Allows moderate uncertainty**: 95% prior interval [0.67, 1.49]
3. **Symmetric in log space**: Equal prior probability of over/underestimation
4. **Conservative**: SD=0.2 in log space allows ±20% typical variation
5. **Regularization**: Prevents extreme inflation/deflation of measurement errors

**Why LogNormal instead of Normal?**
- `lambda` must be positive (multiplicative scale factor)
- LogNormal is standard for scale parameters
- Interpretable: log(lambda) = 0 means exact measurement errors

**Sensitivity analysis alternatives**:
- `LogNormal(0, 0.1)`: Tighter (assume SEs quite accurate)
- `LogNormal(0, 0.5)`: More diffuse (substantial SE uncertainty)
- `LogNormal(-0.1, 0.2)`: Prior median < 1 (assume SEs overestimated)

**When is this prior inappropriate?**
- If we have actual degrees of freedom for each study, should use proper likelihood
- If SEs are from simulation (not estimation), lambda should be fixed at 1

### 3.4 What This Model Captures

**Accounts for**:
1. **Epistemic uncertainty**: Reported SEs may be inaccurate estimates
2. **Common effect hypothesis**: Still assumes θ_i = μ
3. **Systematic SE bias**: If all SEs underestimated, lambda > 1 corrects
4. **Measurement calibration**: Learns appropriate SE scaling from data

**Does NOT account for**:
1. **Study-specific SE biases**: lambda is common across studies
2. **True effect heterogeneity**: Still assumes τ=0
3. **Heavy-tailed errors**: Likelihood remains Normal
4. **Non-random measurement error**: Assumes errors unbiased, just uncertain magnitude

**Key insight**: This model asks, "Are the reported standard errors correctly calibrated?" It's a **meta-measurement** model that questions the precision of our precision estimates.

**When is this model important?**
- Small studies where SE estimates based on small n
- Unclear SE calculation methods
- Suspected underestimation of uncertainty
- Conservative inference desired

**Relationship to other approaches**:
- Similar to **inflate variance** correction in frequentist meta-analysis
- Related to **measurement error models** in econometrics
- Bridges to **robust meta-analysis** (but maintains Normal likelihood)

### 3.5 Falsification Criteria

**This model is WRONG if any of the following hold**:

**Criterion 1: Lambda Near 1 (No SE Uncertainty)**
- **Test**: Posterior P(0.9 < lambda < 1.1 | data) - is SE scaling trivial?
- **Failure**: If P > 0.95, measurement errors are well-calibrated, no adjustment needed
- **Action if failed**: Revert to Model 1 (simpler, equivalent)
- **Success threshold**: P(lambda > 1.2 or lambda < 0.8) > 0.5 for model to be useful

**Criterion 2: Implausible Lambda Values**
- **Test**: Posterior mean of lambda
- **Failure**: If E[lambda | data] > 2 or < 0.5 (SEs off by factor of 2)
- **Interpretation**: Suggests measurement errors grossly miscalibrated OR model misspecified
- **Action if failed**: Investigate individual study SE calculations; may indicate heterogeneity, not SE error

**Criterion 3: Model Comparison Favors Simple Model**
- **Test**: LOO-CV comparison with Model 1
- **Failure**: If Model 1 has lower LOO-elpd (simpler model better)
- **Threshold**: Need Δelpd > 2 to justify extra parameter
- **Action if failed**: SE uncertainty doesn't improve predictions; use Model 1

**Criterion 4: Posterior Predictive Still Fails**
- **Test**: Generate y_rep with adjusted SEs, check coverage
- **Failure**: If observed y still systematically outside 95% posterior predictive intervals
- **Interpretation**: Problem is not SE miscalibration but true heterogeneity (τ > 0)
- **Action if failed**: Abandon fixed-effect models entirely; adopt random-effects

**Criterion 5: Lambda Confounded with Heterogeneity**
- **Test**: Fit competing model with tau > 0 and lambda = 1
- **Failure**: If random-effects model with tau ~ Half-Cauchy(0,5) fits better
- **Interpretation**: Lambda absorbing true effect variation, not SE uncertainty
- **Action if failed**: Use random-effects model; this model conflates sources of variance

**Criterion 6: Study-Specific SE Miscalibration**
- **Test**: Compute study-specific standardized residuals with adjusted SEs
- **Failure**: If lambda adjustment doesn't eliminate residual patterns
- **Action if failed**: Need study-specific lambda_i (more complex) or heterogeneity model

### 3.6 Implementation Notes

**Platform**: Stan or PyMC

**Stan code**:
```stan
data {
  int<lower=1> J;              // number of studies
  vector[J] y;                 // observed effects
  vector<lower=0>[J] sigma;    // reported SEs
}

parameters {
  real mu;                     // common effect
  real<lower=0> lambda;        // SE scale factor
}

model {
  // Priors
  mu ~ normal(0, 15);
  lambda ~ lognormal(0, 0.2);

  // Likelihood with scaled SEs
  y ~ normal(mu, sigma * lambda);
}

generated quantities {
  // Adjusted SEs
  vector[J] sigma_adjusted = sigma * lambda;

  // Posterior predictive
  vector[J] y_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    y_rep[j] = normal_rng(mu, sigma[j] * lambda);
    log_lik[j] = normal_lpdf(y[j] | mu, sigma[j] * lambda);
  }

  // Diagnostics
  real total_variance_explained = variance(rep_vector(mu, J));
  real total_variance_observed = variance(y);
  real variance_ratio = total_variance_explained / total_variance_observed;
}
```

**PyMC code**:
```python
import pymc as pm

with pm.Model() as uncertain_se_model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=15)
    lambda_scale = pm.LogNormal('lambda', mu=0, sigma=0.2)

    # Adjusted SEs
    sigma_adjusted = sigma * lambda_scale

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_adjusted, observed=y)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=mu, sigma=sigma_adjusted, shape=J)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)
```

**Sampling strategy**:
- 4 chains × 2000 iterations (8000 draws)
- 1000 warmup iterations
- Target: ESS > 400, R-hat < 1.01
- Expected: Slight correlation between mu and lambda (both affect scale)

**Computational considerations**:
- **Slightly slower than Model 1**: 2 parameters instead of 1
- **Potential correlation**: mu and lambda posteriors may be correlated
- **No convergence issues expected**: Still simple model
- **Identifiability**: lambda identified if variation in σ_i across studies

### 3.7 Expected Challenges

**Challenge 1: Lambda Not Identified**
- **Issue**: With only 8 studies, hard to distinguish SE miscalibration from chance
- **Symptom**: Lambda posterior ≈ prior (data don't update belief)
- **Diagnostic**: Prior-posterior overlap > 90%
- **Solution**: Accept that data insufficient to calibrate SEs; use Model 1

**Challenge 2: Lambda-Mu Correlation**
- **Issue**: Increasing lambda (wider SEs) pulls mu toward prior mean
- **Symptom**: Posterior correlation between mu and lambda
- **Diagnostic**: |cor(mu, lambda)| > 0.5
- **Solution**: Expected for this model; center inference on marginal posteriors

**Challenge 3: Confounding with Heterogeneity**
- **Issue**: Lambda absorbs true effect variation instead of SE uncertainty
- **Symptom**: lambda > 1.5 and random-effects model has tau ≈ 0
- **Diagnostic**: Models explain variance differently
- **Solution**: Fit model with both tau and lambda; see which is identified

**Challenge 4: Overparameterization**
- **Issue**: Adding lambda may not improve fit enough to justify complexity
- **Symptom**: LOO-CV worse than Model 1 despite extra flexibility
- **Diagnostic**: Δelpd < 0 (worse), or wide PSIS-LOO intervals
- **Solution**: Use Model 1 as primary; report this as sensitivity analysis

**Challenge 5: Interpretation Difficulty**
- **Issue**: Lambda as "measurement error miscalibration" hard to explain
- **Symptom**: Stakeholders misinterpret as data quality issue
- **Diagnostic**: N/A (communication issue)
- **Solution**: Frame as "conservative adjustment" or "robustness check"

**Challenge 6: Prior Sensitivity**
- **Issue**: Lambda posterior sensitive to prior SD (0.2 vs 0.5)
- **Symptom**: Conclusions change with prior choice
- **Diagnostic**: Compare multiple lambda priors
- **Solution**: Report sensitivity; acknowledge weak identifiability

---

## Cross-Model Comparison Strategy

### Model Selection Criteria

**Primary criterion**: LOO-CV (Leave-One-Out Cross-Validation)
- Use Pareto-smoothed importance sampling (PSIS-LOO)
- Compare expected log pointwise predictive density (elpd)
- Model with highest elpd preferred
- Significant difference: Δelpd > 2 SE

**Secondary criteria**:
- **WAIC**: Watanabe-Akaike Information Criterion (alternative to LOO)
- **Posterior predictive checks**: Which model best matches observed data patterns?
- **Parsimony**: Prefer simpler model if Δelpd < 2

**Tertiary considerations**:
- **Scientific plausibility**: Is model assumption reasonable?
- **Robustness**: How sensitive to prior specifications?
- **Communication**: Which model is easiest to explain and justify?

### Expected Outcomes

**Scenario 1: Model 1 Wins (Most Likely)**
- **Evidence**: Simple fixed-effect model has best LOO, reasonable PPC
- **Interpretation**: Complete pooling justified, I²=0% reflects reality
- **Recommendation**: Use Model 1 as primary, report Models 2-3 as sensitivity

**Scenario 2: Model 2 Wins**
- **Evidence**: Precision-stratified model has Δelpd > 3 vs Model 1
- **Interpretation**: Small-study effects present; high/low precision studies differ
- **Recommendation**: Use Model 2 as primary; investigate why groups differ

**Scenario 3: Model 3 Wins**
- **Evidence**: SE uncertainty model has best LOO, lambda posterior away from 1
- **Interpretation**: Reported SEs miscalibrated; need conservative adjustment
- **Recommendation**: Use Model 3 as primary; question original SE calculations

**Scenario 4: None Win (Competing Models Lose to Random-Effects)**
- **Evidence**: All three fixed-effect models fail posterior predictive checks
- **Interpretation**: True heterogeneity exists (τ > 0); fixed-effect assumption wrong
- **Recommendation**: **Abandon all three models**; adopt Designer #1's random-effects models
- **Red flag**: This is the most important outcome scientifically

### Decision Tree

```
1. Fit all three models + Designer #1's random-effects model
2. Compute LOO-CV for all models
3. Check convergence (R-hat < 1.01, ESS > 400)

IF any model fails convergence:
  - Investigate parameterization
  - Increase iterations
  - If persistent, flag model as unreliable

IF all models converge:
  - Compare LOO-elpd values

  IF random-effects has Δelpd > 5 vs best fixed-effect:
    --> **ABANDON FIXED-EFFECTS MODELS**
    --> Use random-effects as primary
    --> Report fixed-effects as boundary cases

  ELSE IF Model 1 has best or equivalent LOO (Δelpd < 2):
    --> Use Model 1 as primary (simplest)
    --> Report Models 2-3 as sensitivity analyses

  ELSE IF Model 2 has Δelpd > 3 vs Model 1:
    --> Use Model 2 as primary (precision-stratification matters)
    --> Investigate why groups differ

  ELSE IF Model 3 has Δelpd > 3 vs Model 1:
    --> Use Model 3 as primary (SE miscalibration matters)
    --> Investigate original study SE calculations

4. Perform posterior predictive checks for top model

  IF PPC fails (p < 0.05 for multiple studies):
    --> **FLAG MODEL INADEQUACY**
    --> Reconsider model class entirely

  ELSE:
    --> Proceed with inference
```

---

## Implementation Timeline and Checkpoints

### Phase 1: Model Implementation (Estimated: 2-3 hours)

**Tasks**:
1. Implement Model 1 in Stan and PyMC
2. Implement Model 2 in Stan and PyMC
3. Implement Model 3 in Stan and PyMC
4. Verify all models compile and run on test data

**Checkpoint 1**: Do all models compile and sample?
- **Success**: Continue to Phase 2
- **Failure**: Debug parameterization, simplify models

### Phase 2: Fitting and Diagnostics (Estimated: 1-2 hours)

**Tasks**:
1. Fit all models on real data
2. Check convergence diagnostics (R-hat, ESS, trace plots)
3. Check PSIS-LOO diagnostics (Pareto-k values)
4. Compute posterior predictive distributions

**Checkpoint 2**: Do all models converge?
- **Success**: Continue to Phase 3
- **Failure**: Increase iterations, reparameterize, or flag model issues

### Phase 3: Model Comparison (Estimated: 1 hour)

**Tasks**:
1. Compute LOO-CV for all models (including Designer #1's)
2. Generate LOO comparison table
3. Perform posterior predictive checks
4. Create visual comparisons (forest plots, residual plots)

**Checkpoint 3**: Is there a clearly best model?
- **Yes**: Proceed with that model to Phase 4
- **No (Δelpd < 2 for top models)**: Report model uncertainty, use averaging or ensemble
- **Fixed-effects fail**: Abandon this design, use random-effects

### Phase 4: Sensitivity Analysis (Estimated: 1-2 hours)

**Tasks**:
1. Prior sensitivity for top model(s)
2. Leave-one-out study sensitivity
3. Influence diagnostics
4. Robustness checks

**Checkpoint 4**: Is inference robust?
- **Success**: Proceed to reporting
- **Failure**: Flag sensitivity, widen uncertainty, or reconsider model

### Phase 5: Reporting and Interpretation (Estimated: 1 hour)

**Tasks**:
1. Generate final visualizations
2. Write inference summary
3. Compute key probability statements
4. Document limitations and assumptions

**Total estimated time**: 6-9 hours of computation and analysis

---

## Red Flags and Escape Routes

### Critical Red Flags (STOP and Reconsider)

**Red Flag 1: Posterior Predictive Failure**
- **Sign**: Multiple studies (>2) fall outside 95% posterior predictive intervals
- **Interpretation**: Fixed-effect assumption fundamentally wrong
- **Escape route**: Adopt random-effects model from Designer #1
- **No patching**: Don't try to fix with robust likelihoods or ad-hoc adjustments

**Red Flag 2: LOO Strongly Favors Random-Effects**
- **Sign**: Random-effects Δelpd > 10 vs best fixed-effect model
- **Interpretation**: True heterogeneity exists; complete pooling inappropriate
- **Escape route**: Use random-effects as primary, report fixed-effects as boundary
- **Lesson**: I²=0% was misleading due to small sample/large SEs

**Red Flag 3: Study 1 Dominates or Distorts**
- **Sign**: Removing Study 1 changes pooled effect by >50%
- **Interpretation**: Single study driving inference; not true pooling
- **Escape route**: Robust methods, outlier investigation, or random-effects
- **Action**: Do not simply remove Study 1 without justification

**Red Flag 4: Prior-Data Conflict**
- **Sign**: Prior predictive p-value < 0.01
- **Interpretation**: Model structure incompatible with data
- **Escape route**: Reconsider likelihood, check for coding errors, or change model class
- **Not a prior issue**: Don't solve by changing priors; likely structural problem

**Red Flag 5: Implausible Parameter Estimates**
- **Sign**: lambda > 3 or < 0.3 in Model 3; |mu| > 50
- **Interpretation**: Model absorbing artifacts or misspecified
- **Escape route**: Check data, check code, reconsider model assumptions
- **May indicate**: Coding error, data issue, or fundamental model failure

### Moderate Warnings (Proceed with Caution)

**Warning 1: Wide Posterior Credible Intervals**
- **Sign**: 95% CI for mu spans 20+ units
- **Interpretation**: High uncertainty (expected with J=8, large SEs)
- **Action**: Report honestly, avoid overconfident conclusions
- **Not a failure**: This is legitimate uncertainty

**Warning 2: Model Comparison Inconclusive**
- **Sign**: Top 2-3 models within 2 elpd of each other
- **Interpretation**: Data insufficient to distinguish models
- **Action**: Model averaging, report multiple models, or use simplest
- **Not a failure**: Honest uncertainty quantification

**Warning 3: Sensitivity to Priors**
- **Sign**: Posterior changes moderately with prior specifications
- **Interpretation**: Data not overwhelming prior (expected with small sample)
- **Action**: Report sensitivity, use weakly informative priors, communicate uncertainty
- **Acceptable**: If conclusions qualitatively similar across reasonable priors

**Warning 4: Low ESS for Some Parameters**
- **Sign**: ESS > 100 but < 400 for some parameters
- **Interpretation**: Inefficient sampling (but not fatal)
- **Action**: Increase iterations, check for high correlation, monitor convergence
- **Acceptable**: If R-hat < 1.01 and posterior well-explored

---

## Success Criteria

### Minimum Success (Model Suite Viable)

1. **All models converge**: R-hat < 1.01, ESS > 400
2. **At least one model fits**: PPC not rejected (p > 0.05)
3. **Clear comparison**: LOO-CV distinguishes models or shows equivalence
4. **Stable inference**: Leave-one-out doesn't change conclusions drastically
5. **Interpretable results**: Can state P(μ > 0 | data) with confidence

### Moderate Success (Preferred Outcome)

1. All minimum criteria met
2. **Best model clearly identified**: Δelpd > 3 from runner-up
3. **Robust to sensitivity**: Prior changes don't flip conclusions
4. **Scientifically plausible**: Model assumptions align with domain knowledge
5. **Communicable**: Can explain model to non-Bayesian audience

### Exceptional Success (Gold Standard)

1. All moderate criteria met
2. **Posterior predictive excellent**: Observed data well within predicted range
3. **Model comparison decisive**: Δelpd > 5, clear winner
4. **Prior-posterior learning**: Data substantially update prior beliefs
5. **External validation**: If new studies added, model predictions hold
6. **Scientific insight**: Model reveals substantive patterns (e.g., precision effects in Model 2)

### Honest Failure (Also a Success)

1. **Fixed-effects models fail**: PPC rejected, LOO poor
2. **Random-effects strongly preferred**: Clear evidence for heterogeneity
3. **Conclusion**: I²=0% was misleading; τ > 0 is reality
4. **Action**: Adopt Designer #1's models, document why fixed-effects failed
5. **Contribution**: Ruled out complete pooling, narrowed model space

**This is not a failure of the modeling process**—it's scientific progress. Discovering your model is wrong is success if you discover it early and pivot appropriately.

---

## Key Differences from Other Designers

**Designer #1** (Hierarchical/Random-Effects):
- **Focus**: Between-study heterogeneity (τ > 0)
- **Philosophy**: Assume studies may differ, let data determine extent
- **Strength**: Flexible, conservative, accounts for heterogeneity
- **Weakness**: May overparameterize if τ truly zero

**Designer #2** (This Design - Fixed-Effects):
- **Focus**: Complete pooling (τ = 0) or precision-stratified effects
- **Philosophy**: Take I²=0% seriously, test homogeneity assumption rigorously
- **Strength**: Maximally powerful if assumption holds, simpler models
- **Weakness**: Strong assumption; will fail if heterogeneity exists

**Designer #3** (Expected: Robust/Non-parametric):
- **Focus**: Likely robustness to outliers, non-Normal errors
- **Philosophy**: Protect against model misspecification
- **Strength**: Robust to heavy tails, outliers
- **Weakness**: More complex, may sacrifice power if Normal is correct

**Complementarity**: These three perspectives bracket the truth. One will likely be clearly best; others serve as sensitivity analyses.

---

## Summary: Three Models, One Philosophy

All three models share the **fixed-effect philosophy** but test different facets:

1. **Model 1**: Classical complete pooling (simplest, strongest assumption)
2. **Model 2**: Precision-stratified effects (partial pooling, tests small-study effects)
3. **Model 3**: Complete pooling with SE uncertainty (conservative adjustment)

**Unifying question**: Can we justify θ_i = μ for all i (possibly with adjustments)?

**Unifying falsification**: If posterior predictive checks fail across all three, **abandon fixed-effects entirely**.

**Implementation priority**:
1. Fit Model 1 first (fastest, baseline)
2. Fit Model 2 second (most interesting alternative)
3. Fit Model 3 third (sensitivity/robustness check)
4. Compare all to Designer #1's random-effects models

**Expected outcome**: Model 1 or random-effects will dominate, but comparison is essential to justify choice.

---

## References and Theoretical Grounding

**Fixed-effect meta-analysis**:
- Borenstein et al. (2010). *A basic introduction to fixed-effect and random-effects models for meta-analysis*
- Hedges & Vevea (1998). Fixed- and random-effects models in meta-analysis

**Bayesian meta-analysis**:
- Gelman et al. (2013). *Bayesian Data Analysis* (3rd ed.), Chapter 5
- Sutton & Abrams (2001). Bayesian methods in meta-analysis and evidence synthesis

**Prior specification**:
- Gelman (2006). Prior distributions for variance parameters in hierarchical models
- Chung et al. (2013). A nondegenerate penalized likelihood estimator for variance parameters

**Model comparison**:
- Vehtari et al. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC
- Piironen & Vehtari (2017). Comparison of Bayesian predictive methods

**Measurement error**:
- Carroll et al. (2006). *Measurement Error in Nonlinear Models*
- Gustafson (2003). *Measurement Error and Misclassification in Statistics and Epidemiology*

**Meta-analysis standards**:
- Cochrane Handbook for Systematic Reviews (2023)
- PRISMA guidelines for meta-analysis reporting

---

**End of Proposal: Model Designer #2**

**Files to be created upon implementation**:
- `/workspace/experiments/designer_2/model_1_fixed_effect.stan`
- `/workspace/experiments/designer_2/model_1_fixed_effect.py`
- `/workspace/experiments/designer_2/model_2_precision_stratified.stan`
- `/workspace/experiments/designer_2/model_2_precision_stratified.py`
- `/workspace/experiments/designer_2/model_3_uncertain_se.stan`
- `/workspace/experiments/designer_2/model_3_uncertain_se.py`
- `/workspace/experiments/designer_2/model_comparison_report.md`

**Next step**: Await synthesis with Designers #1 and #3, then implement models.
