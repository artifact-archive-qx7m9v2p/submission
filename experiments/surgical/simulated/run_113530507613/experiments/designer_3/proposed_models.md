# Bayesian Regression Models with Covariates and Structured Effects

**Designer:** Model Designer 3 (Regression/Covariate Specialist)
**Date:** 2025-10-30
**Focus:** Incorporating covariates and regression structures to explain heterogeneity

---

## Executive Summary

The EDA reveals strong heterogeneity (ICC = 0.42) with success rates varying 4.5-fold across 12 groups. While simple correlation analysis found no significant linear relationship between sample size and success rate (r = -0.34, p = 0.278), this does NOT mean covariates are uninformative. I propose three regression-based models that:

1. **Explicitly model sample size effects** (non-linear, precision weighting)
2. **Test sequential/spatial structure** (group ordering, smooth trends)
3. **Combine hierarchical random effects with fixed covariates** (hybrid approach)

**Critical Philosophy:** The absence of significant correlation in EDA does NOT justify ignoring covariates. Formal Bayesian regression allows us to:
- Quantify evidence FOR/AGAINST covariate effects with uncertainty
- Detect non-linear relationships missed by Pearson correlation
- Properly account for precision differences (small vs large groups)
- Compare models rigorously via LOO-CV

**Falsification Mindset:** Each model includes specific criteria that would make me abandon it and pivot to alternative approaches.

---

## Model 1: Hierarchical Logistic Regression with Sample Size Covariate

### 1.1 Model Name & Rationale

**Name:** `hier_logistic_size_covariate`

**Core Hypothesis:** Sample size (n_trials) may have a non-linear or indirect effect on success rates that linear correlation missed. Larger studies might be conducted in different contexts (e.g., more rigorous protocols, different populations).

**Why This Matters:**
- Group 4 (n=810, largest) has unusually low rate (0.042)
- EDA shows weak negative correlation (r = -0.34), but p = 0.278 (underpowered)
- With J=12, we lack power for significance, but effect size might be scientifically meaningful
- Bayesian approach quantifies uncertainty rather than binary "significant/not"

### 1.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = alpha_j
alpha_j ~ Normal(mu_j, tau)
```

**Regression Structure (Linear Predictor):**
```
mu_j = beta_0 + beta_1 * log(n_j / n_mean)
```

**Priors:**
```
beta_0 ~ Normal(-2.6, 1.0)           # Intercept (grand mean on logit scale)
beta_1 ~ Normal(0, 0.5)              # Slope for log(sample size), weakly informative
tau ~ Normal^+(0, 0.5)               # Between-group SD (residual heterogeneity)
```

**Full Model Specification:**
- **Intercept (beta_0):** Population-level baseline success rate (logit scale)
- **Slope (beta_1):** Effect of (log) sample size on success rate
- **Random effects (alpha_j):** Group-specific deviations from regression line
- **Hyperparameter (tau):** Residual heterogeneity NOT explained by sample size

**Key Design Choices:**
1. **Log-transform n_trials:** Stabilizes variance, assumes diminishing returns with scale
2. **Center by mean:** n_mean = 234.5 (improves sampling, reduces correlation with intercept)
3. **Weakly informative prior on beta_1:** N(0, 0.5) allows ±1 SD on logit scale (substantial)
4. **tau prior:** Half-normal(0, 0.5) allows substantial residual heterogeneity

### 1.3 Stan Implementation

```stan
data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
  vector[J] log_n_centered;          // log(n) centered by mean
}

parameters {
  real beta_0;                       // intercept
  real beta_1;                       // slope for log(sample size)
  real<lower=0> tau;                 // residual between-group SD
  vector[J] alpha_raw;               // non-centered parameterization
}

transformed parameters {
  vector[J] alpha;                   // group effects
  vector[J] mu;                      // regression predictor
  vector[J] p;                       // success probabilities

  // Regression structure
  mu = beta_0 + beta_1 * log_n_centered;

  // Non-centered parameterization for better sampling
  alpha = mu + tau * alpha_raw;

  // Inverse logit
  p = inv_logit(alpha);
}

model {
  // Priors
  beta_0 ~ normal(-2.6, 1.0);
  beta_1 ~ normal(0, 0.5);
  tau ~ normal(0, 0.5);
  alpha_raw ~ std_normal();

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  // Log-likelihood for LOO
  vector[J] log_lik;

  // Posterior predictive
  array[J] int r_rep;

  // R-squared (variance explained by covariate)
  real<lower=0, upper=1> R2;

  // Effect size: how much does doubling sample size change success rate?
  real effect_double_n;

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // R-squared: proportion of variance explained by regression vs random effects
  {
    real var_fitted = variance(mu);
    real var_total = var_fitted + tau^2;
    R2 = var_fitted / var_total;
  }

  // Effect of doubling sample size on probability scale
  effect_double_n = inv_logit(beta_0 + beta_1 * log(2)) - inv_logit(beta_0);
}
```

**Data Preprocessing (Python):**
```python
import numpy as np
log_n_centered = np.log(data['n_trials']) - np.log(data['n_trials'].mean())
stan_data = {
    'J': len(data),
    'n': data['n_trials'].values,
    'r': data['r_successes'].values,
    'log_n_centered': log_n_centered
}
```

### 1.4 Theoretical Justification

**Why Sample Size as Covariate?**

1. **Study Design Confounding:** Larger studies may differ systematically:
   - More resources → better protocols → potentially different populations
   - Smaller studies → pilot/exploratory → different inclusion criteria

2. **Publication Bias Analogy:** In meta-analysis, small studies often show inflated effects
   - Here, small studies might have inflated success rates (though data shows opposite)

3. **Non-linear Effects:** Log-transform captures:
   - Diminishing information returns
   - Multiplicative rather than additive effects
   - Better alignment with information theory

4. **Precision Weighting:** The model implicitly weights groups by precision:
   - Large groups have narrow likelihood → strong data signal
   - Small groups shrink more toward regression line
   - This is CORRECT behavior (unlike treating correlation as unweighted)

**What This Captures vs. Random Effects Only:**
- **Random effects only:** Treats all heterogeneity as "unexplained noise"
- **This model:** Partitions heterogeneity into:
  - Systematic component (sample size effect)
  - Residual random variation

**Key Difference:** Even if beta_1 ≈ 0, we gain information by EXPLICITLY TESTING this hypothesis.

### 1.5 Falsification Criteria

**I will abandon this model if:**

1. **Posterior for beta_1 overwhelmingly includes zero:**
   - 95% CI for beta_1 includes zero AND is narrow (e.g., [-0.1, 0.1])
   - R² < 0.05 (covariate explains < 5% of variance)
   - **Evidence threshold:** LOO comparison shows no improvement over random effects only

2. **Residual heterogeneity unchanged:**
   - tau in covariate model ≈ tau in random-effects-only model
   - No reduction in between-group variance
   - **Implication:** Covariate adds complexity without explanatory power

3. **Poor posterior predictive performance:**
   - Posterior predictive p-value < 0.05 or > 0.95
   - Systematic deviations in PP checks
   - **Suggests:** Wrong functional form or missing other covariates

4. **Computational pathologies:**
   - Divergent transitions (> 1% of samples)
   - Rhat > 1.01
   - Low ESS (< 400) despite non-centered parameterization
   - **Indicates:** Model misspecification, not just sampling issues

5. **Unrealistic effect sizes:**
   - |beta_1| > 1.0 (implausibly large on logit scale)
   - Effect of doubling sample size > 0.05 (5 percentage points)
   - **Suggests:** Overfitting or spurious correlation

**Expected Results Under Null:**
- beta_1 ~ N(0, 0.3) [posterior narrower than prior]
- R² ~ Uniform(0, 0.15) [low but uncertain]
- tau_covariate ≈ tau_random_only

**Decision Rule:**
If ALL of conditions 1-2 are met, conclude: "Sample size is not a meaningful covariate. Prefer simpler random effects model."

### 1.6 Computational Considerations

**Multicollinearity:**
- With only one covariate, multicollinearity is not an issue
- If extending to multiple covariates, check VIF < 5

**Centering/Scaling Strategy:**
- **Log-transform:** Handles wide range (47 to 810, 17-fold)
- **Mean-centering:** Reduces correlation between beta_0 and beta_1
- **Do NOT standardize:** We want interpretable units (log trials)

**Prior for beta_1:**
- N(0, 0.5) on logit scale is weakly informative
- Allows substantial effects but penalizes extreme values
- Sensitivity analysis: Try N(0, 0.25) and N(0, 1.0)

**Non-centered Parameterization:**
- Essential when tau is small relative to beta_1
- Improves sampling efficiency
- Standard practice for hierarchical models

**Sampling Strategy:**
- 4 chains, 2000 iterations (1000 warmup)
- Target: Rhat < 1.01, ESS > 400
- Expect ~30-60 seconds on modern CPU

---

## Model 2: Hierarchical Regression with Non-linear (Quadratic) Group Effect

### 2.1 Model Name & Rationale

**Name:** `hier_logistic_quadratic_group`

**Core Hypothesis:** The sequential group_id (1-12) might reflect underlying structure:
- **Temporal:** If groups are time periods, trends over time
- **Spatial:** If groups are sites, geographic gradients
- **Ordinal:** If groups have natural ordering (severity, dose, etc.)

**Why Quadratic (Not Linear)?**
- EDA found no linear trend (p = 0.69)
- But clusters suggest non-monotonic pattern:
  - Groups 1, 2 (high rates) at beginning
  - Groups 4-7 (low rates) in middle
  - Group 8 (high rate) at end
- Quadratic captures U-shape or inverted-U

**Critical Caveat:** If group_id is truly arbitrary (e.g., random labeling), this model should find beta_1 ≈ beta_2 ≈ 0. That's a FEATURE, not a bug—we're testing a hypothesis, not assuming it's true.

### 2.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = alpha_j
alpha_j ~ Normal(mu_j, tau)
```

**Regression Structure (Quadratic Predictor):**
```
mu_j = beta_0 + beta_1 * group_id_scaled_j + beta_2 * group_id_scaled_j^2
```

**Scaling:**
```
group_id_scaled = (group_id - 6.5) / 3.5  # Scale to [-1, 1] range
```

**Priors:**
```
beta_0 ~ Normal(-2.6, 1.0)           # Intercept (grand mean)
beta_1 ~ Normal(0, 0.5)              # Linear term (trend)
beta_2 ~ Normal(0, 0.5)              # Quadratic term (curvature)
tau ~ Normal^+(0, 0.5)               # Residual heterogeneity
```

**Interpretation:**
- **beta_1 = 0, beta_2 = 0:** No sequential structure (group_id arbitrary)
- **beta_1 ≠ 0, beta_2 = 0:** Linear trend (monotonic increase/decrease)
- **beta_1 = 0, beta_2 ≠ 0:** U-shape or inverted-U (symmetric)
- **beta_1 ≠ 0, beta_2 ≠ 0:** Asymmetric curve

### 2.3 Stan Implementation

```stan
data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
  vector[J] group_scaled;            // group_id scaled to [-1, 1]
  vector[J] group_scaled_sq;         // group_scaled^2
}

parameters {
  real beta_0;                       // intercept
  real beta_1;                       // linear term
  real beta_2;                       // quadratic term
  real<lower=0> tau;                 // residual between-group SD
  vector[J] alpha_raw;               // non-centered parameterization
}

transformed parameters {
  vector[J] alpha;                   // group effects
  vector[J] mu;                      // regression predictor
  vector[J] p;                       // success probabilities

  // Quadratic regression structure
  mu = beta_0 + beta_1 * group_scaled + beta_2 * group_scaled_sq;

  // Non-centered parameterization
  alpha = mu + tau * alpha_raw;

  // Inverse logit
  p = inv_logit(alpha);
}

model {
  // Priors
  beta_0 ~ normal(-2.6, 1.0);
  beta_1 ~ normal(0, 0.5);
  beta_2 ~ normal(0, 0.5);
  tau ~ normal(0, 0.5);
  alpha_raw ~ std_normal();

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  // Log-likelihood for LOO
  vector[J] log_lik;

  // Posterior predictive
  array[J] int r_rep;

  // R-squared (variance explained by polynomial)
  real<lower=0, upper=1> R2;

  // Peak/trough location (if quadratic is significant)
  real peak_location;  // in scaled units [-1, 1]

  // Test for curvature
  real curvature_significant;  // = 1 if 95% CI for beta_2 excludes zero

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // R-squared
  {
    real var_fitted = variance(mu);
    real var_total = var_fitted + tau^2;
    R2 = var_fitted / var_total;
  }

  // Peak of parabola (only meaningful if beta_2 != 0)
  peak_location = -beta_1 / (2 * beta_2);

  // Indicator for significant curvature
  curvature_significant = (beta_2 > 0) ? 1.0 : 0.0;  // Simplified
}
```

**Data Preprocessing (Python):**
```python
group_scaled = (data['group_id'] - 6.5) / 3.5  # Scale to [-1, 1]
group_scaled_sq = group_scaled ** 2

stan_data = {
    'J': len(data),
    'n': data['n_trials'].values,
    'r': data['r_successes'].values,
    'group_scaled': group_scaled.values,
    'group_scaled_sq': group_scaled_sq.values
}
```

### 2.4 Theoretical Justification

**Why Sequential/Ordinal Structure?**

1. **Temporal Processes:** If groups are time periods:
   - Learning effects (early groups different from late)
   - Environmental changes (seasons, policies)
   - Intervention refinement (protocols improve over time)

2. **Spatial Gradients:** If groups are locations:
   - Geographic clines (latitude, altitude)
   - Proximity to resource centers
   - Cultural/demographic gradients

3. **Dose-Response:** If groups represent ordered categories:
   - Treatment intensities
   - Risk strata
   - Disease severity stages

4. **Artifact Detection:** If group_id is truly arbitrary:
   - Model should find beta_1 ≈ beta_2 ≈ 0
   - This TESTS the exchangeability assumption
   - If significant, we've learned something important

**What Quadratic Captures:**
- **Linear term (beta_1):** Monotonic trend
- **Quadratic term (beta_2):** Curvature (acceleration/deceleration)
- **Together:** Can approximate many smooth functions locally

**Advantages Over Splines:**
- Only 2 parameters (vs. 3+ for splines)
- Interpretable coefficients
- Adequate for J=12 (splines would overfit)

**What This Captures vs. Model 1:**
- Model 1: Sample size (study characteristic)
- Model 2: Sequential position (potential temporal/spatial structure)
- These are INDEPENDENT hypotheses (could test both simultaneously)

### 2.5 Falsification Criteria

**I will abandon this model if:**

1. **Both coefficients include zero:**
   - 95% CI for beta_1 includes zero AND 95% CI for beta_2 includes zero
   - R² < 0.05
   - **Conclusion:** No sequential structure, group_id is arbitrary

2. **LOO-CV favors simpler model:**
   - elpd_diff < 0 (worse than random effects only)
   - SE(elpd_diff) > |elpd_diff| (no decisive evidence)
   - **Implication:** Added complexity not justified

3. **Unrealistic curvature:**
   - Peak/trough outside observed range (beyond groups 1-12)
   - Predicted reversal that doesn't match data
   - **Suggests:** Overfitting to noise

4. **Collinearity issues:**
   - Despite orthogonalization, posterior correlation(beta_1, beta_2) > 0.7
   - Wide posteriors for both parameters
   - **Solution:** Try single linear term or GAM

5. **Posterior predictive failure:**
   - Systematic deviations in PP checks
   - Model captures spurious pattern that doesn't generalize

**Expected Results Under Null:**
- beta_1 ~ N(0, 0.3)
- beta_2 ~ N(0, 0.3)
- R² ~ Uniform(0, 0.15)

**Decision Rule:**
If condition 1 AND 2 are met, conclude: "No evidence for sequential structure. Group ordering is arbitrary or uninformative."

### 2.6 Computational Considerations

**Multicollinearity:**
- Quadratic terms naturally correlated with linear terms
- Mitigation: Scale to [-1, 1] (orthogonalizes somewhat)
- Check: posterior correlation(beta_1, beta_2) < 0.5 desired

**Scaling Strategy:**
- **Linear:** (group_id - 6.5) / 3.5 centers and scales
- **Quadratic:** Use scaled version, not raw
- **Why [-1, 1]?** Numerical stability, interpretable coefficients

**Prior Specification:**
- Same N(0, 0.5) for both beta_1 and beta_2
- Symmetric (no preference for linear vs. quadratic)
- Allows substantial curvature but penalizes overfitting

**Alternative Parameterization:**
- Could use orthogonal polynomials (Legendre)
- Would eliminate correlation between beta_1 and beta_2
- Trade-off: Less intuitive interpretation

**Sampling Strategy:**
- 4 chains, 2000 iterations
- Monitor correlation between beta_1 and beta_2
- If poor mixing, switch to orthogonal polynomials

---

## Model 3: Hierarchical Regression with Random Slopes (Varying Effects)

### 3.1 Model Name & Rationale

**Name:** `hier_logistic_random_slopes`

**Core Hypothesis:** The effect of sample size might VARY across subpopulations. This extends Model 1 by allowing each group's response to sample size to differ.

**Why Random Slopes?**

1. **Cluster Heterogeneity:** EDA identified 3 clusters with different baseline rates
   - High-rate cluster might have different size-response than low-rate cluster

2. **Measurement Precision:** Random slopes model heteroskedasticity
   - Some groups more "stable" (flat slope)
   - Others more "variable" (steep slope)

3. **Scientific Realism:** One-size-fits-all slopes rarely hold in biology
   - Treatment effects vary across subpopulations
   - Environmental responses differ by genotype

4. **Robust to Misspecification:** If slopes don't actually vary, model reduces to Model 1
   - Posterior for sigma_slope will concentrate near zero
   - This is "self-correcting"—model complexity penalized if not needed

**Key Insight:** This tests whether the relationship between n_trials and success rate is HOMOGENEOUS across groups or HETEROGENEOUS.

### 3.2 Mathematical Specification

**Likelihood:**
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = alpha_j + gamma_j * log(n_j / n_mean)
```

**Hierarchical Structure:**
```
alpha_j ~ Normal(beta_0, tau_alpha)          # Random intercepts
gamma_j ~ Normal(beta_1, tau_gamma)          # Random slopes
```

**Priors:**
```
beta_0 ~ Normal(-2.6, 1.0)                   # Population mean intercept
beta_1 ~ Normal(0, 0.5)                      # Population mean slope
tau_alpha ~ Normal^+(0, 0.5)                 # SD of intercepts
tau_gamma ~ Normal^+(0, 0.3)                 # SD of slopes (more conservative)
```

**Correlation:**
```
Optionally: (alpha_j, gamma_j) ~ MVN(mu, Sigma)
where Sigma has correlation rho
```

**Interpretation:**
- **beta_0:** Average success rate across groups (logit scale)
- **beta_1:** Average effect of sample size
- **tau_alpha:** Between-group variation in baseline rates
- **tau_gamma:** Between-group variation in size-response
- **rho:** Correlation between intercepts and slopes (if included)

### 3.3 Stan Implementation

```stan
data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
  vector[J] log_n_centered;          // log(n) centered by mean
}

parameters {
  real beta_0;                       // population mean intercept
  real beta_1;                       // population mean slope
  real<lower=0> tau_alpha;           // SD of intercepts
  real<lower=0> tau_gamma;           // SD of slopes
  vector[J] alpha_raw;               // non-centered intercepts
  vector[J] gamma_raw;               // non-centered slopes
  real<lower=-1, upper=1> rho;       // correlation between intercepts and slopes
}

transformed parameters {
  vector[J] alpha;                   // group intercepts
  vector[J] gamma;                   // group slopes
  vector[J] mu;                      // linear predictor
  vector[J] p;                       // success probabilities

  // Non-centered parameterization with correlation
  // Using Cholesky decomposition for efficiency
  {
    matrix[2, 2] L_Sigma;            // Cholesky factor of correlation matrix
    matrix[J, 2] z;                  // Standard normal draws
    matrix[J, 2] effects;            // Correlated effects

    // Build correlation matrix and decompose
    L_Sigma[1, 1] = 1.0;
    L_Sigma[2, 1] = rho;
    L_Sigma[2, 2] = sqrt(1 - rho^2);
    L_Sigma[1, 2] = 0.0;

    // Stack raw parameters
    z[, 1] = alpha_raw;
    z[, 2] = gamma_raw;

    // Transform with correlation and scaling
    effects = z * L_Sigma';

    alpha = beta_0 + tau_alpha * effects[, 1];
    gamma = beta_1 + tau_gamma * effects[, 2];
  }

  // Linear predictor with varying slopes
  for (j in 1:J) {
    mu[j] = alpha[j] + gamma[j] * log_n_centered[j];
  }

  // Inverse logit
  p = inv_logit(mu);
}

model {
  // Priors
  beta_0 ~ normal(-2.6, 1.0);
  beta_1 ~ normal(0, 0.5);
  tau_alpha ~ normal(0, 0.5);
  tau_gamma ~ normal(0, 0.3);        // More conservative for slopes
  rho ~ uniform(-1, 1);              // Uninformative on correlation
  alpha_raw ~ std_normal();
  gamma_raw ~ std_normal();

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  // Log-likelihood for LOO
  vector[J] log_lik;

  // Posterior predictive
  array[J] int r_rep;

  // Variance partitioning
  real var_explained_intercepts;
  real var_explained_slopes;
  real total_var;
  real prop_var_slopes;              // Proportion due to varying slopes

  // Test for slope variation
  real slopes_vary;                  // = 1 if 95% CI for tau_gamma excludes zero

  // Extreme slope detection
  real max_abs_slope;

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // Variance components
  var_explained_intercepts = tau_alpha^2;
  var_explained_slopes = tau_gamma^2 * variance(log_n_centered);
  total_var = var_explained_intercepts + var_explained_slopes;
  prop_var_slopes = var_explained_slopes / total_var;

  // Indicator for varying slopes
  slopes_vary = (tau_gamma > 0.1) ? 1.0 : 0.0;

  // Detect extreme slope heterogeneity
  max_abs_slope = max(fabs(gamma));
}
```

**Simpler Version (No Correlation):**
If rho causes sampling issues, fix rho = 0:
```stan
alpha = beta_0 + tau_alpha * alpha_raw;
gamma = beta_1 + tau_gamma * gamma_raw;
```

### 3.4 Theoretical Justification

**Why Random Slopes?**

1. **Individual Differences:** In meta-analysis, treatment effects vary across studies
   - Similarly, size-response might vary across contexts

2. **Heteroskedasticity:** Random slopes model unequal variance
   - Some groups "tighter" (small tau_gamma[j])
   - Others "looser" (large tau_gamma[j])

3. **Interaction Effects:** Random slopes capture group × covariate interactions
   - Without pre-specifying which groups differ
   - Data-driven discovery of heterogeneity

4. **Generalization:** More realistic than fixed slopes
   - Acknowledges parameter uncertainty
   - Better predictive performance for new groups

**What This Captures vs. Model 1:**
- **Model 1:** One slope for all (beta_1 fixed across groups)
- **Model 3:** Each group has its own slope (gamma_j varies)
- **Trade-off:** More parameters, but more flexible

**Correlation Structure:**
- **Negative rho:** High-baseline groups have weaker size effects
- **Positive rho:** High-baseline groups have stronger size effects
- **Zero rho:** Independent variation (simpler, often adequate)

**Why This Might Fail:**
- With J=12, estimating tau_gamma is ambitious
- May not have power to detect slope variation
- Could overfit if variation is small

### 3.5 Falsification Criteria

**I will abandon this model if:**

1. **No evidence for slope variation:**
   - 95% CI for tau_gamma includes zero (or < 0.1)
   - prop_var_slopes < 0.05 (slopes explain < 5% of variance)
   - **Conclusion:** Slopes don't meaningfully vary, use Model 1 (simpler)

2. **LOO-CV penalizes complexity:**
   - elpd_diff < 0 compared to Model 1 (fixed slopes)
   - p_loo warns about overfitting
   - **Implication:** Extra parameters not justified by improved fit

3. **Extreme slope estimates:**
   - max_abs_slope > 2.0 (implausibly large)
   - Some gamma_j have opposite signs from beta_1
   - **Suggests:** Overfitting, estimating noise

4. **Posterior for tau_gamma equals prior:**
   - Posterior ≈ half-normal(0, 0.3) (prior not updated)
   - **Indicates:** Data uninformative about slope variation

5. **Computational failures:**
   - Divergent transitions persist despite reparameterization
   - rho has extreme values (±0.99)
   - **Solution:** Fix rho=0 or use Model 1

**Expected Results Under Null:**
- tau_gamma ~ half-normal(0, 0.2)
- prop_var_slopes < 0.1
- max_abs_slope < 1.0

**Decision Rule:**
If condition 1 AND 2 are met, conclude: "Slopes do not meaningfully vary across groups. Prefer Model 1 (fixed slopes) for parsimony."

### 3.6 Computational Considerations

**Multicollinearity:**
- Intercepts and slopes naturally correlated
- Centering covariate helps but doesn't eliminate
- rho parameter explicitly models this correlation

**Scaling Strategy:**
- Log-transform and center (as in Model 1)
- Do NOT standardize (loses interpretability)
- Centering crucial for separating alpha and gamma

**Prior for tau_gamma:**
- More conservative than tau_alpha (N(0, 0.3) vs. N(0, 0.5))
- Rationale: Harder to estimate with J=12
- Regularization prevents overfitting

**Non-centered Parameterization:**
- Essential for both alpha and gamma
- Cholesky decomposition for correlated effects
- If sampling issues, fix rho=0 (simplifies)

**Sampling Strategy:**
- 4 chains, 2000 iterations (may need more)
- Monitor Rhat for tau_gamma and rho (often problematic)
- Expect 2-5 minutes (more complex than Models 1-2)

**Fallback Strategy:**
- If Model 3 fails, try Model 1 (simpler)
- If correlation causes issues, fix rho=0
- If tau_gamma not identified, use fixed slopes

---

## Model Comparison Strategy

### How to Choose Among Models

**Sequential Testing Approach:**

1. **Fit all three models** independently (no peeking at results to choose)

2. **LOO-CV comparison:**
   - Compare elpd_loo (expected log predictive density)
   - ΔLOO > 4: Substantial evidence
   - ΔLOO < 2: Negligible evidence
   - Check p_loo for overfitting warnings

3. **Posterior predictive checks:**
   - All models should pass basic PPC (p-value in [0.05, 0.95])
   - Check specific discrepancies (e.g., capture outliers?)

4. **Coefficient interpretation:**
   - **Model 1:** Is beta_1 credibly non-zero?
   - **Model 2:** Are beta_1 or beta_2 credibly non-zero?
   - **Model 3:** Is tau_gamma credibly > 0.1?

5. **Parsimony principle:**
   - If ΔLOO < 2, prefer simpler model
   - Complexity must be justified by improved fit

**Decision Tree:**

```
Start
  |
  +--> Fit all 3 models + baseline (random effects only)
  |
  +--> LOO comparison
         |
         +--> If Model 1 wins (ΔLOO > 4):
         |      Sample size effect present
         |      Report beta_1 and R²
         |
         +--> If Model 2 wins (ΔLOO > 4):
         |      Sequential structure detected
         |      Report beta_1, beta_2, peak location
         |
         +--> If Model 3 wins (ΔLOO > 4):
         |      Varying slopes justified
         |      Report tau_gamma and slope heterogeneity
         |
         +--> If baseline wins (all ΔLOO < 2):
               No covariates improve fit
               Use simple random effects model
               Report ICC and shrinkage only
```

### Cross-Model Diagnostics

**All models must satisfy:**

1. **MCMC convergence:**
   - Rhat < 1.01 for all parameters
   - ESS > 400 for all parameters
   - No divergent transitions (< 1%)

2. **Posterior predictive checks:**
   - p-value in [0.05, 0.95] for test statistics
   - Visually: PP intervals contain observed data
   - Capture heterogeneity (variance, outliers)

3. **LOO diagnostics:**
   - All Pareto-k < 0.7 (no influential observations)
   - p_loo ≈ number of parameters (no overfitting)

4. **Prior sensitivity:**
   - Results stable under prior perturbations
   - Posterior not dominated by prior

**Red Flags Across All Models:**

- **Prior-posterior conflict:** Prior and likelihood "fight"
- **Posterior at boundaries:** tau=0, rho=±1, extreme betas
- **Poor predictive performance:** High RMSE, miscalibration
- **Inconsistent with EDA:** Contradicts known patterns

---

## Integration with Other Designers

### Expected Overlaps and Unique Contributions

**Designer 1 (Random Effects):**
- **Overlap:** Both use hierarchical structure
- **My contribution:** Explicitly test covariates, not just random noise

**Designer 2 (Mixture/Robust):**
- **Overlap:** Both address heterogeneity
- **My contribution:** Continuous covariates vs. discrete clusters

**Complementarity:**
- If my models fail (covariates uninformative), supports Designer 1
- If my Model 2 succeeds (sequential pattern), complements Designer 2's clusters
- If Model 3 succeeds (varying slopes), suggests mixture model might also work

### Model Averaging and Stacking

**If no single model dominates:**

1. **Bayesian Model Averaging (BMA):**
   - Weight models by LOO-IC
   - Combine predictions proportionally

2. **Stacking:**
   - Optimal linear combination maximizing predictive performance
   - Often outperforms BMA

3. **Ensemble:**
   - Report all plausible models
   - Quantify model uncertainty

**Don't force a winner if ΔLOO < 2!** Multiple models may be equally plausible.

---

## Falsification Summary: When to Pivot Completely

### Conditions to Abandon ALL Regression Models

**I will reject the entire covariate-based approach if:**

1. **No model improves over baseline:**
   - All ΔLOO < 2 compared to random effects only
   - All R² < 0.05
   - **Conclusion:** Covariates (n_trials, group_id) are uninformative

2. **Computational failures across all models:**
   - Persistent divergences despite reparameterization
   - Poor mixing for all coefficient posteriors
   - **Suggests:** Data-model mismatch, wrong likelihood

3. **Posterior predictive failures:**
   - All models show systematic deviations
   - Cannot capture outliers (Groups 4, 8)
   - **Indicates:** Need different model class (mixture, robust)

4. **Unrealistic parameter estimates:**
   - All models produce extreme coefficients
   - Predictions outside plausible range
   - **Suggests:** Overfitting or misspecified priors

### Alternative Approaches if Regression Fails

**Pivot to:**

1. **Pure random effects** (Designer 1's approach)
   - If heterogeneity truly unexplained by covariates
   - Focus on shrinkage and ICC

2. **Mixture models** (Designer 2's approach)
   - If 3-cluster structure is the real story
   - Discrete subpopulations, not continuous covariates

3. **Non-linear models:**
   - Gaussian processes (flexible smooth functions)
   - Splines (more flexible than quadratic)
   - Requires justification given J=12

4. **Robust models:**
   - Student-t likelihoods
   - Beta-binomial overdispersion
   - If outliers cannot be captured by regression

**Decision Point:** After fitting all models, if NONE achieve ΔLOO > 2, document failure and recommend alternative approaches.

---

## Computational Implementation Plan

### Software and Dependencies

**Primary:** Stan (via CmdStanPy)
- Robust MCMC (NUTS sampler)
- Excellent diagnostics
- LOO-CV built-in (loo package)

**Alternative:** PyMC
- Similar capabilities
- Easier Python integration
- Potentially slower for complex models

**Required Packages:**
```python
import cmdstanpy
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Workflow

1. **Data preprocessing:**
   - Load data
   - Create covariates (log_n_centered, group_scaled)
   - Validate (check ranges, no NAs)

2. **Prior predictive checks:**
   - Sample from prior
   - Verify predictions in reasonable range
   - Adjust priors if needed

3. **Model fitting:**
   - Compile Stan models (once)
   - Fit 4 chains × 2000 iterations
   - Save samples for analysis

4. **Diagnostics:**
   - Check Rhat, ESS, divergences
   - LOO-CV (loo package)
   - Posterior predictive checks

5. **Comparison:**
   - LOO comparison table
   - Plot predictions vs. data
   - Report coefficients with uncertainty

6. **Sensitivity analysis:**
   - Vary priors (weak vs. strong)
   - Check robustness

### Expected Runtime

- Model 1: ~30 seconds
- Model 2: ~30 seconds
- Model 3: 2-5 minutes (more complex)
- Total (including diagnostics): ~30 minutes

### Diagnostic Thresholds

| Metric | Acceptable | Marginal | Unacceptable |
|--------|-----------|----------|--------------|
| Rhat | < 1.01 | 1.01-1.05 | > 1.05 |
| ESS | > 400 | 200-400 | < 200 |
| Divergences | 0% | < 1% | > 1% |
| Pareto-k | < 0.7 | 0.7-1.0 | > 1.0 |

---

## Expected Outcomes and Interpretation

### Scenario 1: Model 1 Wins (Sample Size Effect)

**If beta_1 credibly non-zero (95% CI excludes 0):**

- **Interpretation:** Sample size systematically related to success rate
- **Effect size:** Report inv_logit(beta_0 + beta_1) - inv_logit(beta_0)
- **Scientific implication:** Larger studies differ systematically (confounding?)
- **Next steps:** Investigate WHY (study design, population differences)

**Example result:**
```
beta_1 = -0.3 [95% CI: -0.5, -0.1]
Interpretation: Doubling sample size associated with 2 percentage point
                decrease in success rate
R² = 0.25 (sample size explains 25% of between-group variance)
```

### Scenario 2: Model 2 Wins (Sequential Structure)

**If beta_2 credibly non-zero (95% CI excludes 0):**

- **Interpretation:** Non-monotonic trend across groups
- **Pattern:** U-shape (beta_2 > 0) or inverted-U (beta_2 < 0)
- **Scientific implication:** Group ordering is NOT arbitrary
- **Next steps:** Investigate context (temporal? spatial? other?)

**Example result:**
```
beta_2 = 0.4 [95% CI: 0.1, 0.7]
Interpretation: Success rates higher at extremes (groups 1-2, 11-12),
                lower in middle (groups 5-8)
Peak location: Group 9 (scaled = 0.6)
```

### Scenario 3: Model 3 Wins (Varying Slopes)

**If tau_gamma credibly > 0.1:**

- **Interpretation:** Size-response varies across groups
- **Heterogeneity:** Some groups sensitive to sample size, others not
- **Scientific implication:** Interaction between group and sample size
- **Next steps:** Identify which groups have extreme slopes

**Example result:**
```
tau_gamma = 0.2 [95% CI: 0.1, 0.4]
Interpretation: Substantial variation in how groups respond to sample size
prop_var_slopes = 0.18 (18% of variance due to varying slopes)
```

### Scenario 4: Baseline Wins (No Covariates Help)

**If all ΔLOO < 2:**

- **Interpretation:** Covariates (n_trials, group_id) uninformative
- **Heterogeneity:** Pure random variation across groups
- **Scientific implication:** True differences not explained by measured covariates
- **Next steps:** Focus on random effects model, investigate unmeasured factors

**Example result:**
```
ΔLOO (Model 1 vs Baseline) = 1.2 ± 2.5 (negligible)
ΔLOO (Model 2 vs Baseline) = -0.5 ± 1.8 (negligible)
ΔLOO (Model 3 vs Baseline) = 0.8 ± 3.1 (negligible)
Conclusion: No evidence for covariate effects
```

---

## Final Recommendations

### Primary Model Priority

1. **Model 1** (Sample size covariate): HIGHEST PRIORITY
   - Most theoretically justified
   - Simplest extension of baseline
   - Directly tests EDA finding (r = -0.34)

2. **Model 2** (Quadratic group): MEDIUM PRIORITY
   - Exploratory (tests exchangeability)
   - May reveal hidden structure
   - Less interpretable if group_id arbitrary

3. **Model 3** (Random slopes): LOWEST PRIORITY (if time permits)
   - Most complex
   - Ambitious given J=12
   - Best for sensitivity analysis

### Synthesis Strategy

**After fitting all models:**

1. Create LOO comparison table
2. Report coefficients for best model(s)
3. Visualize predictions vs. data
4. Discuss scientific implications
5. Acknowledge uncertainty (if ΔLOO < 2)

### Communication Plan

**Report should include:**

1. **Methods:** Model specifications, priors, diagnostics
2. **Results:** Coefficients, R², LOO comparison
3. **Visualizations:**
   - Observed vs. predicted
   - Coefficient posteriors
   - LOO comparison (with SE bars)
4. **Interpretation:** Scientific meaning, limitations
5. **Falsification:** What evidence would change conclusion

### Success Criteria

**This analysis succeeds if:**

- All models converge (Rhat < 1.01)
- At least one model improves over baseline (ΔLOO > 2), OR
- Clear evidence that covariates are uninformative (all ΔLOO < 2 with narrow SE)
- Results interpretable and scientifically plausible
- Limitations acknowledged

**This analysis FAILS if:**

- Models don't converge
- Results contradict EDA without explanation
- Parameter estimates implausible
- Cannot distinguish models (all similar, wide SE)

---

## Conclusion

I have designed three Bayesian regression models that systematically test whether available covariates (sample size, group ordering) explain the observed heterogeneity. Each model:

1. Has clear theoretical justification
2. Includes falsification criteria
3. Can be compared via LOO-CV
4. Acknowledges when it might fail

**Key Philosophy:** I'm not trying to prove covariates matter. I'm rigorously TESTING whether they matter. If they don't (ΔLOO < 2), that's a valid and important finding—it tells us to focus on other explanations.

**Critical Insight:** The EDA found r = -0.34, p = 0.278 (not significant). But:
- With J=12, power is low (Type II error likely)
- Bayesian approach quantifies effect size and uncertainty
- Non-linear relationships might exist (quadratic, varying slopes)
- Formal model comparison (LOO) more powerful than correlation test

**Next Steps:**
1. Implement models in Stan
2. Run prior predictive checks
3. Fit to data with MCMC
4. Compare via LOO-CV
5. Report best model(s) with appropriate caveats

**Falsification Commitment:** If all models fail to improve over baseline random effects, I will recommend abandoning the covariate approach and focusing on Designer 1's random effects or Designer 2's mixture models.

---

**Files to create:**
- `/workspace/experiments/designer_3/model1_size_covariate.stan`
- `/workspace/experiments/designer_3/model2_quadratic_group.stan`
- `/workspace/experiments/designer_3/model3_random_slopes.stan`
- `/workspace/experiments/designer_3/fit_models.py` (implementation script)
- `/workspace/experiments/designer_3/comparison_report.md` (after fitting)
