# Bayesian Model Design for Hierarchical Measurement Error Dataset
## Model Designer 1

**Date:** 2025-10-28
**Dataset:** 8 observations with known measurement error (sigma)
**Key Challenge:** Measurement error (sigma ≈ 12.5) comparable to observed variation (SD ≈ 11.1)

---

## Executive Summary

I propose THREE competing model classes that make fundamentally different assumptions about the data generation process. Each model can FAIL in specific ways, and I explicitly define what evidence would force me to abandon each approach.

**Critical Insight from EDA:** Between-group variance = 0, SNR ≈ 1, complete pooling supported (p=0.42)

**My Strategy:** Start with Model 1 (complete pooling) as the EDA-supported baseline. Models 2 and 3 are designed to CHALLENGE this conclusion and test whether hierarchical structure or outlier effects exist despite null EDA findings.

**Success Criterion:** Finding a model that fails is success - it means we've learned something about the true data generation process.

---

# Model 1: Complete Pooling (EDA-Supported Baseline)

## Problem Formulation

**Hypothesis:** All groups share a single common true mean. Observed variation is entirely due to known measurement error.

**Data Generation Process:**
```
True value (shared across all groups): theta
Observed value for group i: y_i ~ Normal(theta, sigma_i)
```

This is the SIMPLEST hypothesis consistent with EDA findings (chi-square test p=0.42, between-group variance = 0).

---

## Mathematical Specification

### Likelihood
```
y_i ~ Normal(mu, sigma_i)    for i = 1, ..., 8

where:
  y_i     = observed value for group i
  mu      = population mean (shared parameter)
  sigma_i = known measurement error for group i
```

### Prior Distribution
```
mu ~ Normal(10, 20)

Justification:
  - Center at 10: EDA weighted mean = 10.02
  - SD = 20: Weakly informative, allows [-30, 50] range
  - Lets data dominate but prevents extreme values
```

### Posterior (analytical)
With known measurement error, the posterior is conjugate:
```
mu | y ~ Normal(mu_post, sigma_post^2)

where:
  precision_i = 1/sigma_i^2
  precision_prior = 1/20^2

  precision_post = precision_prior + sum(precision_i)

  mu_post = (precision_prior * 10 + sum(precision_i * y_i)) / precision_post

  sigma_post = 1/sqrt(precision_post)
```

**Expected Result:** mu_post ≈ 10, sigma_post ≈ 4

---

## Theoretical Justification

### Why This Model Makes Sense

1. **EDA Support:**
   - Chi-square homogeneity test: p = 0.42 (fail to reject H0: groups homogeneous)
   - Between-group variance = 0 (observed var < expected measurement var)
   - Leave-one-out analysis: no outliers detected

2. **Measurement Model:**
   - Properly accounts for heteroscedastic known measurement error
   - Weights observations by precision (1/sigma_i^2)
   - Group 4 (smallest sigma=9) gets highest weight despite negative value

3. **Maximum Information Sharing:**
   - With only 8 observations and SNR ≈ 1, pooling maximizes precision
   - Effective sample size ≈ 5.5 (accounting for heterogeneous errors)

4. **Scientific Plausibility:**
   - Common in measurement settings: all measuring same quantity
   - Analogous to meta-analysis with single true effect
   - Parsimonious: fewest parameters

### Why This Model Might FAIL

1. **Hidden Group Structure:**
   - EDA may have insufficient power to detect small between-group differences
   - With n=8 and high measurement error, could miss tau up to ~5 units

2. **Non-Normal Errors:**
   - Assumes Gaussian measurement error
   - Heavy tails or contamination would violate assumptions

3. **Systematic Bias:**
   - Model assumes measurement errors are unbiased
   - If some groups have systematic bias, pooling is inappropriate

4. **Temporal/Spatial Structure:**
   - Model assumes exchangeability
   - If group order matters (time, space), structure is ignored

---

## Falsification Criteria

### I WILL ABANDON THIS MODEL IF:

1. **Poor Posterior Predictive Performance (PRIMARY CRITERION)**
   - Test: Posterior predictive p-value < 0.05 OR > 0.95
   - Metric: Chi-square statistic comparing observed vs predicted dispersion
   - Implementation: Compute T_obs = sum((y_i - mu_post)^2 / sigma_i^2), compare to replicated data
   - **Red Flag:** If T_obs consistently outside [2.5%, 97.5%] of T_rep distribution
   - **Action if failed:** Move to Model 2 (hierarchical)

2. **Extreme Residuals**
   - Test: Any standardized residual |r_i| > 3 where r_i = (y_i - mu_post) / sigma_i
   - Current status: Max |r_i| = 1.86 for Group 4 (acceptable)
   - **Red Flag:** If any |r_i| > 3 after fitting
   - **Action if failed:** Investigate outliers, consider Model 3 (robust)

3. **Prior-Data Conflict**
   - Test: Prior predictive vs observed data
   - Metric: P(y_obs | prior) - should not be in extreme tails
   - **Red Flag:** If observed data have prior predictive probability < 1%
   - **Action if failed:** Reconsider prior centering or move to different model class

4. **Leave-One-Out Cross-Validation Failures**
   - Test: PSIS-LOO with all Pareto k < 0.7
   - **Red Flag:** Any Pareto k > 0.7 indicates influential observation
   - **Action if failed:** Investigate influential points, consider robust model

5. **Systematic Pattern in Residuals**
   - Test: Residuals vs fitted, residuals vs group order
   - **Red Flag:** Systematic trend (e.g., residuals correlate with group ID)
   - **Action if failed:** Indicates missing structure, add predictors or hierarchical effects

### Secondary Warning Signs

- Posterior 95% CI excludes too many observed values (>1 out of 8)
- Predictive intervals fail to cover future observations
- Effective sample size (ESS) < 100 per chain (computational issues suggest misspecification)

---

## Computational Considerations

### Expected MCMC Behavior

**Favorable Properties:**
- Single parameter (mu) should converge quickly
- Conjugate structure (could solve analytically)
- No divergent transitions expected
- No funnel geometry issues

**Expected Convergence:**
- R-hat < 1.01 with 1000 iterations
- ESS > 1000 per chain
- Trace plots: well-mixing random walk

### Stan Implementation
```stan
data {
  int<lower=1> N;           // number of observations
  vector[N] y;              // observed values
  vector<lower=0>[N] sigma; // known measurement errors
}

parameters {
  real mu;                  // population mean
}

model {
  // Prior
  mu ~ normal(10, 20);

  // Likelihood with known measurement error
  y ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive
  vector[N] y_rep;
  vector[N] log_lik;
  real chi_sq_obs = 0;
  real chi_sq_rep = 0;

  for (i in 1:N) {
    y_rep[i] = normal_rng(mu, sigma[i]);
    log_lik[i] = normal_lpdf(y[i] | mu, sigma[i]);
    chi_sq_obs += square((y[i] - mu) / sigma[i]);
    chi_sq_rep += square((y_rep[i] - mu) / sigma[i]);
  }
}
```

### Potential Issues
- **None expected** - this is a textbook simple model
- If convergence issues arise, indicates data-model mismatch (good diagnostic!)

---

## Stress Tests

### Test 1: Exclude High-SNR Groups
- **Design:** Fit model using only groups 4-7 (SNR < 1)
- **Purpose:** Test if conclusion relies only on precise measurements
- **Expected:** Wider posterior but similar mean
- **Abandon if:** Mean shifts by >10 units (suggests high-SNR groups driving result)

### Test 2: Prior Sensitivity
- **Design:** Fit with mu ~ N(0, 100) (vague prior)
- **Purpose:** Test prior influence
- **Expected:** Posterior shifts slightly toward 0 but remains positive
- **Abandon if:** Posterior changes drastically (insufficient data, prior matters too much)

### Test 3: Leave-One-Out
- **Design:** Refit excluding each group sequentially
- **Purpose:** Test influence of individual observations
- **Expected:** Minimal change in posterior mean (<2 units)
- **Abandon if:** Any single group changes conclusion (model too fragile)

---

# Model 2: Hierarchical Partial Pooling (Falsification Model)

## Problem Formulation

**Hypothesis:** Groups have distinct true means drawn from a common population distribution. This CONTRADICTS the EDA finding that between-group variance = 0.

**Purpose:** This model is designed to FAIL if EDA conclusions are correct, but would succeed if EDA missed subtle group structure.

**Data Generation Process:**
```
Population mean: mu
Between-group variation: tau
True mean for group i: theta_i ~ Normal(mu, tau)
Observed value: y_i ~ Normal(theta_i, sigma_i)
```

---

## Mathematical Specification

### Likelihood
```
y_i ~ Normal(theta_i, sigma_i)    for i = 1, ..., 8

where:
  theta_i = group-specific true mean
  sigma_i = known measurement error
```

### Hierarchical Structure
```
theta_i ~ Normal(mu, tau)         for i = 1, ..., 8

where:
  mu  = population mean (hyperparameter)
  tau = between-group standard deviation
```

### Prior Distributions
```
mu  ~ Normal(10, 20)              # Population mean
tau ~ Half-Cauchy(0, 5)           # Between-group SD

Justification:
  - mu prior: Same as Model 1 for comparability
  - tau prior: Half-Cauchy is standard (Gelman 2006)
  - Scale=5: Conservative given observed SD ≈ 11
```

### Parameter Interpretation
- **mu:** Average across all groups (what we'd measure with infinite precision)
- **tau:** How much groups truly differ (scientific quantity of interest)
- **theta_i:** True value for group i (what we'd observe with sigma_i = 0)

---

## Theoretical Justification

### Why This Model Makes Sense

1. **Standard Approach:**
   - Canonical hierarchical model for grouped data
   - Lets data determine degree of pooling
   - Reduces to Model 1 if tau → 0, to no pooling if tau → ∞

2. **Adaptive Shrinkage:**
   - Low-precision groups (large sigma_i) shrink more toward mu
   - High-precision groups (small sigma_i) retain more individual information
   - Optimal compromise between pooling and independence

3. **Meta-Analytic Framework:**
   - Equivalent to random-effects meta-analysis
   - Each group is a "study" with known standard error
   - Established methodology with well-understood properties

4. **Conservative Approach:**
   - More flexible than complete pooling
   - Protects against Type I error (falsely claiming homogeneity)

### Why This Model Might FAIL

1. **Insufficient Data for tau Estimation:**
   - With n=8 groups and high measurement error, tau is poorly identified
   - Posterior for tau may be dominated by prior
   - Could estimate tau > 0 even when true tau = 0 (false positive)

2. **Computational Difficulties:**
   - When tau is near 0, creates "funnel" geometry
   - Centered parameterization has divergent transitions
   - Non-centered parameterization may be necessary

3. **Overfit to Noise:**
   - May attribute measurement error to "group effects"
   - theta_i estimates may track y_i too closely

4. **Model More Complex Than Needed:**
   - If EDA is correct (tau = 0), this model wastes degrees of freedom
   - Wider credible intervals without gain in fit

---

## Falsification Criteria

### I WILL ABANDON THIS MODEL IF:

1. **tau Posterior Concentrates at Zero (PRIMARY CRITERION)**
   - Test: Posterior median tau < 1 AND 95% CI includes 0
   - Interpretation: Data support complete pooling (Model 1 is sufficient)
   - **Threshold:** If P(tau < 2 | data) > 0.95, revert to Model 1
   - **Action:** Accept that groups are homogeneous, use simpler model

2. **Computational Pathologies**
   - Test: Any divergent transitions with centered parameterization
   - Test: ESS(tau) < 100 even with non-centered parameterization
   - **Red Flag:** Poor sampling indicates model-data mismatch
   - **Action:** tau is unidentified; data insufficient for hierarchical structure

3. **No Improvement Over Model 1**
   - Test: WAIC or LOO-CV comparison
   - Metric: Delta-ELPD < 2 (no meaningful difference)
   - **Threshold:** If Model 1 has equal or better predictive performance
   - **Action:** Parsimony favors simpler model (Model 1)

4. **Theta_i Posteriors Indistinguishable**
   - Test: All pairwise 95% CIs for theta_i overlap
   - Test: max(theta_i) - min(theta_i) < 5 in posterior median
   - **Interpretation:** Groups not meaningfully different
   - **Action:** Model 1 is adequate

5. **Prior-Data Conflict on tau**
   - Test: Prior-to-posterior shrinkage ratio for tau
   - **Red Flag:** If posterior is narrower than prior but concentrated at boundary (tau=0)
   - **Interpretation:** Data actively pushing tau toward zero
   - **Action:** Strong evidence against between-group variation

---

## Computational Considerations

### Expected MCMC Behavior

**Potential Issues:**
- **Funnel geometry:** When tau is small, theta_i are tightly constrained
- **Divergent transitions:** Centered parameterization struggles near tau=0
- **Poor mixing:** tau may have low ESS

**Expected Result Given EDA:**
- Posterior tau will concentrate near 0
- Wide uncertainty in tau (poorly identified)
- theta_i will shrink heavily toward mu
- Effective model reduction to complete pooling

### Stan Implementation Strategy

**Non-Centered Parameterization (REQUIRED):**
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[N] theta_raw;  // non-centered
}

transformed parameters {
  vector[N] theta = mu + tau * theta_raw;
}

model {
  mu ~ normal(10, 20);
  tau ~ cauchy(0, 5);
  theta_raw ~ std_normal();  // implies theta ~ normal(mu, tau)

  y ~ normal(theta, sigma);
}
```

**Why Non-Centered:**
- When tau is small, theta_raw remains well-identified
- Avoids funnel geometry
- Critical for models with poorly-identified variance parameters

### Diagnostic Strategy
1. Check divergent transitions (should be 0)
2. Check ESS for tau (need >100)
3. Pairs plot: look for funnel in (tau, theta_i) space
4. Compare centered vs non-centered parameterizations

---

## Stress Tests

### Test 1: Informative Prior on tau
- **Design:** tau ~ Half-Normal(0, 2) (shrink toward 0)
- **Purpose:** Test if posterior tau relies on flat prior
- **Expected:** tau posterior should be similar
- **Abandon if:** tau estimate changes drastically (prior-driven, not data-driven)

### Test 2: Exclude Group 4
- **Design:** Fit without the negative observation
- **Purpose:** Test if negative value creates false heterogeneity
- **Expected:** tau should decrease (less apparent variation)
- **Abandon if:** tau increases (suggests Group 4 was masking true heterogeneity)

### Test 3: No-Pooling Comparison
- **Design:** Fit model with theta_i ~ Normal(10, 100) (independent)
- **Purpose:** Show benefit of hierarchical structure
- **Expected:** Hierarchical model should have narrower theta_i posteriors
- **Abandon if:** No shrinkage occurs (hierarchical structure not helping)

---

# Model 3: Robust Model with t-Distribution (Outlier-Detection Model)

## Problem Formulation

**Hypothesis:** Group 4 (y = -4.88) may represent an outlier or contaminated measurement. A t-distributed likelihood can robustly estimate mu while downweighting extreme observations.

**Purpose:** Test whether the negative observation is statistically inconsistent with the rest, even though leave-one-out analysis (p=0.063) and EDA suggest it's not.

**Data Generation Process:**
```
True mean (shared): mu
Observed value with heavy-tailed errors: y_i ~ Student_t(nu, mu, sigma_i)

where nu controls tail heaviness (nu=1: Cauchy, nu→∞: Normal)
```

---

## Mathematical Specification

### Likelihood
```
y_i ~ Student_t(nu, mu, sigma_i)    for i = 1, ..., 8

where:
  nu      = degrees of freedom (heaviness of tails)
  mu      = location parameter (population mean)
  sigma_i = scale parameter (known measurement error)
```

### Prior Distributions
```
mu  ~ Normal(10, 20)                # Population mean (same as Model 1)
nu  ~ Gamma(2, 0.1)                 # Degrees of freedom

Justification for nu prior:
  - Gamma(2, 0.1) has mean=20, SD=14
  - Allows light tails (nu>30 ≈ normal) and heavy tails (nu<5)
  - Weakly informative: lets data determine tail behavior
  - Mode at nu≈10: moderate robustness
```

### Parameter Interpretation
- **mu:** Robust location estimate (less influenced by outliers than normal)
- **nu:** Degree of "outlierness" in data
  - nu > 30: Data consistent with normal (Model 1 adequate)
  - nu < 10: Evidence for heavy tails (outliers present)
  - nu < 5: Strong evidence for contamination

---

## Theoretical Justification

### Why This Model Makes Sense

1. **Robust to Outliers:**
   - t-distribution has heavier tails than normal
   - Automatically downweights extreme observations
   - Standard approach in robust Bayesian analysis

2. **Group 4 is Suspicious:**
   - Only negative observation (y = -4.88)
   - Leave-one-out p = 0.063 (borderline)
   - If truly from different process, t-distribution would detect it

3. **Data-Driven Robustness:**
   - If nu → ∞, reduces to Model 1 (normal likelihood)
   - If nu → 1, becomes Cauchy (very robust)
   - Lets data determine appropriate level of robustness

4. **Conservative Inference:**
   - Protects against influential outliers
   - Credible intervals may be wider but more trustworthy

### Why This Model Might FAIL

1. **No Evidence for Outliers in EDA:**
   - Leave-one-out analysis: all groups consistent (|z| < 2.5)
   - Group 4 only 1.86 SD from mean (not extreme)
   - EDA found no outliers by any criterion

2. **Overparameterization:**
   - Adding nu parameter may not improve fit
   - Could overfit to noise
   - With n=8, estimating nu from data is challenging

3. **t-Distribution Assumes Symmetric Tails:**
   - Model treats positive and negative extremes equally
   - If Group 4 is systematically different (not random outlier), model fails

4. **Computational Complexity:**
   - t-distribution is more complex than normal
   - May have convergence issues with small sample size
   - nu may be poorly identified

---

## Falsification Criteria

### I WILL ABANDON THIS MODEL IF:

1. **nu Posterior Indicates Normal Distribution (PRIMARY CRITERION)**
   - Test: Posterior median nu > 30
   - Interpretation: Data consistent with normal, no outliers detected
   - **Threshold:** If P(nu > 30 | data) > 0.75, revert to Model 1
   - **Action:** Accept that t-distribution is unnecessary, use normal likelihood

2. **No Improvement Over Model 1**
   - Test: LOO comparison with Model 1
   - Metric: Delta-ELPD < 2 (no meaningful difference)
   - **Threshold:** If Model 1 has equal or better predictive performance
   - **Action:** Parsimony favors simpler model

3. **mu Estimate Similar to Model 1**
   - Test: |mu_robust - mu_model1| < 2
   - Test: 95% CIs overlap substantially
   - **Interpretation:** Outlier robustness not affecting inference
   - **Action:** Model 1 is adequate

4. **Posterior Predictive Shows Overdispersion**
   - Test: Model predicts more variation than observed
   - Metric: Posterior predictive p-value for variance
   - **Red Flag:** Heavy tails predict extreme values not seen in data
   - **Action:** Model is too permissive, doesn't match data generation process

5. **nu is Poorly Identified**
   - Test: Posterior SD(nu) > 20
   - Test: ESS(nu) < 100
   - **Interpretation:** Insufficient data to estimate tail behavior
   - **Action:** Revert to Model 1 with fixed nu=∞ (normal)

---

## Computational Considerations

### Expected MCMC Behavior

**Potential Issues:**
- **nu may be poorly identified:** Posterior may track prior
- **Slower mixing:** t-distribution more complex than normal
- **Multimodality:** Possible if outlier detection is ambiguous

**Expected Result Given EDA:**
- nu posterior likely > 30 (data consistent with normal)
- mu estimate similar to Model 1
- Wider credible intervals
- No evidence for outlier detection

### Stan Implementation
```stan
data {
  int<lower=1> N;
  vector[N] y;
  vector<lower=0>[N] sigma;
}

parameters {
  real mu;
  real<lower=1> nu;  // constrain to > 1 for finite variance
}

model {
  mu ~ normal(10, 20);
  nu ~ gamma(2, 0.1);

  y ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    y_rep[i] = mu + sigma[i] * student_t_rng(nu, 0, 1);
    log_lik[i] = student_t_lpdf(y[i] | nu, mu, sigma[i]);
  }
}
```

### Diagnostic Strategy
1. Check posterior for nu: is it constrained by data or prior?
2. Compare LOO pointwise contributions: does Group 4 have large influence?
3. Posterior predictive checks: does model predict right amount of variation?
4. Prior-posterior overlap: is posterior learning from data?

---

## Stress Tests

### Test 1: Exclude Group 4
- **Design:** Fit without Group 4 (the suspected outlier)
- **Purpose:** Test if outlier detection relies on single point
- **Expected:** nu should increase (fewer outliers without Group 4)
- **Abandon if:** nu decreases (other outliers present, Group 4 not the issue)

### Test 2: Fixed nu
- **Design:** Fit models with nu fixed at 5, 10, 20, 30
- **Purpose:** Test sensitivity to nu prior
- **Expected:** Similar mu estimates across all
- **Abandon if:** Estimates vary wildly (model unstable)

### Test 3: Simulate Outliers
- **Design:** Add synthetic extreme outliers to data, refit
- **Purpose:** Test if model can detect actual outliers
- **Expected:** nu should decrease, outliers downweighted
- **Abandon if:** Model doesn't detect obvious outliers (robustness mechanism failed)

---

# Model Comparison Strategy

## Decision Tree

```
START: Fit Model 1 (Complete Pooling)
│
├─ Posterior predictive checks PASS?
│  ├─ YES: Check LOO-CV
│  │  ├─ All Pareto k < 0.7? → ACCEPT Model 1 ✓
│  │  └─ Any Pareto k > 0.7? → Try Model 3 (Robust)
│  └─ NO: Systematic residual pattern?
│     ├─ YES: Fit Model 2 (Hierarchical)
│     └─ NO: Investigate data quality issues
│
├─ If Model 2 fitted:
│  ├─ Posterior median tau < 1? → REVERT to Model 1 ✓
│  ├─ tau > 5? → ACCEPT Model 2, investigate group structure
│  └─ 1 < tau < 5? → Uncertain region, use Model 2 conservatively
│
└─ If Model 3 fitted:
   ├─ Posterior median nu > 30? → REVERT to Model 1 ✓
   ├─ nu < 10? → ACCEPT Model 3, investigate outliers
   └─ Divergent transitions? → Computational issues, suspect model-data mismatch
```

## Formal Comparison Metrics

### 1. Posterior Predictive Checks (Primary)
**For each model, compute:**
```
T_obs = sum((y_i - mu_post)^2 / sigma_i^2)
T_rep = sum((y_rep,i - mu_post)^2 / sigma_i^2)  # from posterior predictive

p-value = P(T_rep >= T_obs | data)
```
**Criterion:** 0.05 < p-value < 0.95 (model captures data structure)

### 2. Leave-One-Out Cross-Validation
**Use PSIS-LOO with Pareto diagnostics:**
- ELPD_LOO: Expected log pointwise predictive density
- Pareto k: Shape parameter for importance sampling
  - k < 0.5: Excellent
  - 0.5 < k < 0.7: Good
  - k > 0.7: Poor (influential observation)

**Comparison Rule:**
- Compute Delta-ELPD between models
- SE(Delta-ELPD) from pointwise differences
- If |Delta-ELPD| < 2*SE, models are equivalent → Choose simpler

### 3. Prior-to-Posterior Learning
**For each parameter, compute:**
```
Prior SD
Posterior SD
Shrinkage = 1 - (Posterior SD / Prior SD)
```
**Criterion:** Shrinkage > 0.5 indicates data are informative

### 4. Effective Sample Size Ratios
**For hierarchical model specifically:**
```
ESS_bulk(mu) > 400 per chain
ESS_bulk(tau) > 100 per chain  # Often lower
ESS_tail(all params) > 400 per chain

R-hat < 1.01 for all parameters
```

---

## What Evidence Would Make Me Reconsider Everything?

### Major Red Flags (Would Trigger Complete Rethink)

1. **All Three Models Fail Posterior Predictive Checks**
   - Implication: Data generation process is fundamentally different than assumed
   - Action: Investigate measurement model, check for:
     - Correlations between groups
     - Systematic patterns in sigma
     - Non-Gaussian error structure (e.g., bimodal, truncated)
     - Time/spatial autocorrelation

2. **Computational Difficulties Across All Models**
   - Implication: Data-model mismatch at fundamental level
   - Action:
     - Check data quality (typos, unit errors)
     - Examine raw measurement process
     - Consider discrete/count data models if y is actually discrete

3. **Group 4 Has Extreme Pareto k (>1.0) in All Models**
   - Implication: This observation is qualitatively different
   - Action:
     - Investigate measurement process for Group 4
     - Consider excluding (if measurement error)
     - Fit mixture model (if genuinely from different population)

4. **Posteriors Track Priors Too Closely**
   - Implication: Insufficient data to overcome prior
   - Action:
     - Report high uncertainty
     - Recommend collecting more data
     - Consider fully Bayesian decision analysis

5. **Posterior Predictive Distributions Too Wide**
   - Implication: Models predict more variation than observed
   - Action:
     - Check if sigma values are correct (might be inflated)
     - Consider constrained parameter space
     - Investigate informative priors from domain knowledge

---

## Alternative Model Classes (Escape Routes)

If all three proposed models fail, pivot to:

### Escape Route 1: Non-Hierarchical Structure
**Assumption:** Groups differ but not hierarchically
```
y_i ~ Normal(alpha + beta * group_i, sigma_i)
```
**When:** If groups show linear trend rather than random variation

### Escape Route 2: Mixture Model
**Assumption:** Two subpopulations (e.g., Groups 0-3 vs 4-7)
```
y_i ~ Normal(mu_1, sigma_i) with probability p
y_i ~ Normal(mu_2, sigma_i) with probability 1-p
```
**When:** If posterior predictive shows bimodality

### Escape Route 3: Measurement Model Misspecification
**Assumption:** sigma values are uncertain or biased
```
y_i ~ Normal(mu, sigma_i * kappa_i)
kappa_i ~ LogNormal(0, 0.5)  # Multiplicative error on sigma
```
**When:** If residual patterns suggest sigma is misspecified

### Escape Route 4: Correlated Errors
**Assumption:** Groups are not independent
```
y ~ MVNormal(mu * 1, Sigma)
Sigma_ij = sigma_i^2 if i=j, rho*sigma_i*sigma_j if i≠j
```
**When:** If groups are spatially/temporally related

### Escape Route 5: Data Quality Issues
**Action:** Stop modeling, investigate data collection
**When:** Computational issues persist, multiple diagnostics fail

---

# Summary of Falsification Philosophy

## My Modeling Principles

1. **Start Simple:** Model 1 is my baseline. It's EDA-supported and parsimonious.

2. **Challenge Aggressively:** Models 2 and 3 are designed to FALSIFY Model 1. If they fail (as expected), it strengthens confidence in Model 1.

3. **Explicit Failure Criteria:** Each model has clear metrics that would make me abandon it.

4. **Computational Issues Are Informative:** Divergent transitions and poor mixing aren't just nuisances - they often indicate model misspecification.

5. **Be Ready to Pivot:** If all three models fail, I've outlined five escape routes.

6. **Truth Over Completion:** Success is finding the right model, not completing a plan.

---

## Expected Outcomes (Predictions)

Based on EDA findings, I predict:

**Model 1 (Complete Pooling):**
- Will PASS all diagnostics ✓
- Posterior: mu ≈ 10 ± 4
- No computational issues
- Best LOO-CV score
- **Likely winner**

**Model 2 (Hierarchical):**
- Will estimate tau ≈ 0-2
- Computational challenges (funnel)
- Will effectively reduce to Model 1
- Similar LOO-CV to Model 1
- **Will fail falsification test** (tau too small → revert to Model 1)

**Model 3 (Robust):**
- Will estimate nu > 30 (approximately normal)
- mu estimate similar to Model 1
- No improvement in LOO-CV
- **Will fail falsification test** (nu too large → revert to Model 1)

## What Would Surprise Me

If these predictions are WRONG, it means:
- EDA was misleading (possible with n=8)
- Measurement model assumptions are incorrect
- There's subtle structure we missed

**That would be valuable information - it would mean we learned something unexpected about the data generation process.**

---

## Implementation Checklist

For each model:
- [ ] Write Stan code
- [ ] Specify priors explicitly
- [ ] Run MCMC (4 chains, 2000 iterations)
- [ ] Check diagnostics (R-hat, ESS, divergences)
- [ ] Posterior predictive checks
- [ ] LOO-CV with Pareto k diagnostics
- [ ] Evaluate falsification criteria
- [ ] Document decision (accept/reject/pivot)

---

## Final Statement

**I will judge success not by fitting all three models, but by:**
1. Discovering which model best represents the true data generation process
2. Being able to definitively reject models that don't fit
3. Having clear criteria to make these decisions
4. Being ready to abandon all three if evidence demands it

**The goal is truth, not task completion.**

---

## References

- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515-534.
- Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- McElreath, R. (2020). *Statistical Rethinking*, 2nd ed. CRC Press.
- Gabry, J., et al. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A*, 182(2), 389-402.

---

**Document Location:** `/workspace/experiments/designer_1/proposed_models.md`
**Next Steps:** Implement models in Stan/PyMC and evaluate against falsification criteria
