# Bayesian Modeling Strategy: Designer 1
## Standard Hierarchical Approaches and Comparisons

**Date**: 2025-10-30
**Designer**: Model Designer #1
**Focus**: Standard hierarchical approaches with rigorous falsification testing

---

## Executive Summary

Based on EDA findings showing strong overdispersion (φ=3.59, ICC=0.56), I propose **three competing model classes** plus two baselines:

1. **Hierarchical Logit-Normal** (PRIMARY) - Standard approach, expected to perform best
2. **Beta-Binomial** (ALTERNATIVE) - Simpler conjugate model, may fail if heterogeneity is too complex
3. **Hierarchical Robust** (ROBUSTNESS CHECK) - Student-t hyperprior to handle outliers
4. **Pooled Model** (BASELINE) - Expected to fail, documents inadequacy
5. **Unpooled Model** (BASELINE) - Expected to overfit, documents need for pooling

**Critical Mindset**: I expect Model 1 to succeed, but I will actively try to break it. If extreme groups (2, 4, 8) cause systematic misprediction or if τ posterior conflicts strongly with prior, I will abandon the standard approach for mixture models or structured alternatives.

---

## Problem Formulation

### Observed Data
- **12 groups**: Exchangeable, no covariates
- **n**: [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360] trials per group
- **r**: [6, 19, 8, 34, 12, 13, 9, 30, 16, 3, 19, 27] successes per group
- **Empirical rates**: 3.1% to 14.0% (4.5-fold range)
- **Pooled rate**: 6.97% (but rejected by χ²=39.47, p<0.0001)

### Competing Hypotheses

**H1 (Hierarchical Logit-Normal)**: Groups are drawn from a common population with success rates varying log-normally on the logit scale. Implies:
- Symmetric heterogeneity around population mean
- Extreme groups are tail events, not fundamentally different
- Single variance parameter τ describes all variation

**H2 (Beta-Binomial)**: Groups share a common Beta distribution describing success probability variability. Implies:
- Conjugacy simplifies inference
- Overdispersion captured by Beta concentration parameters
- May be inadequate if logit-scale normality assumption is violated

**H3 (Hierarchical Robust)**: Groups vary around population mean but with heavier tails (Student-t). Implies:
- Outliers (groups 2, 4, 8) are more common than normal distribution suggests
- Need protection against extreme influence
- Extra degrees-of-freedom parameter ν

### What Would Make Me Abandon Each Hypothesis?

**Abandon H1 if**:
- Posterior predictive p-values < 0.01 for overdispersion test
- Pareto-k > 0.7 for multiple groups (LOO failure)
- Extreme groups systematically mispredicted (|z-score| > 3 in posterior predictive)
- τ posterior concentrates at extreme values (>2 on logit scale)
- Divergent transitions persist despite reparameterization

**Abandon H2 if**:
- LOO dramatically worse than H1 (ΔLOO > 10)
- Cannot capture observed variance ratio (posterior predictive φ << 3.59)
- Prior-posterior conflict on concentration parameters

**Abandon H3 if**:
- ν posterior concentrates near 30 (degenerates to normal, H1 is simpler)
- No improvement over H1 in LOO or posterior predictive checks
- Computational issues worse than H1

### Red Flags Triggering Major Strategy Pivot

**Switch to mixture models if**:
- Bimodal posterior for any group-level θ
- Systematic separation of groups 2, 8 vs. 4, 10 in posterior
- Evidence of discrete subpopulations (e.g., k-means on posterior means shows clusters)

**Switch to structured models if**:
- Correlation emerges between group ID and posterior residuals
- Temporal/spatial structure discovered in data generation process
- Groups 4 and 8 (largest) systematically different from others

**Switch to nonparametric models if**:
- All parametric models fail posterior predictive checks
- Evidence of non-exchangeability despite EDA suggesting otherwise
- τ posterior extremely diffuse (uniform over wide range)

---

## Model 1: Hierarchical Logit-Normal (PRIMARY CANDIDATE)

### Mathematical Specification

**Likelihood**:
```
r_j ~ Binomial(n_j, p_j)  for j = 1, ..., 12
```

**Link function** (logit scale):
```
logit(p_j) = θ_j
θ_j ~ Normal(μ, τ²)  [hyperprior on group-level parameters]
```

**Hyperpriors**:
```
μ ~ Normal(-2.5, 1²)        [population mean on logit scale]
τ ~ Half-Cauchy(0, 1)       [between-group SD on logit scale]
```

**Non-centered parameterization** (for computational stability):
```
θ_j = μ + τ · z_j
z_j ~ Normal(0, 1)  [standardized group effects]
```

### Stan Implementation

```stan
data {
  int<lower=1> J;              // Number of groups (12)
  array[J] int<lower=0> n;     // Trials per group
  array[J] int<lower=0> r;     // Successes per group
}

parameters {
  real mu;                      // Population mean (logit scale)
  real<lower=0> tau;            // Between-group SD (logit scale)
  vector[J] z;                  // Standardized group effects (non-centered)
}

transformed parameters {
  vector[J] theta = mu + tau * z;  // Group-level logit rates
}

model {
  // Hyperpriors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);           // Half-Cauchy via constraint

  // Group effects (non-centered)
  z ~ std_normal();

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);              // Success probabilities
  array[J] int<lower=0> r_rep;                 // Posterior predictive replicates
  vector[J] log_lik;                           // Pointwise log-likelihood for LOO

  // Posterior predictive replicates
  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }

  // Summary statistics for posterior predictive checks
  real pooled_rate_rep = sum(r_rep) / sum(n);
  real chi_sq_rep;  // Compute in post-processing
}
```

### Prior Justification

**μ ~ Normal(-2.5, 1)**:
- **Back-transforms to**: logit^(-1)(-2.5 ± 2) ≈ [4%, 27%] (95% interval)
- **Centered at**: 7.6%, close to observed pooled rate 6.97%
- **Rationale**: Weakly informative, allows data to dominate while preventing extreme values
- **Domain knowledge**: Success rates in this range are plausible (not near 0% or 100%)

**τ ~ Half-Cauchy(0, 1)**:
- **Scale=1** on logit scale corresponds to substantial heterogeneity
- **Prior implies**: 50% of groups within ±0.67 of μ (on logit scale) ≈ 1.5-2× on probability scale
- **Heavy tails**: Allows for extreme τ if data demand it, but regularizes toward reasonable values
- **Standard choice**: Gelman (2006) recommendation for hierarchical SD priors
- **EDA evidence**: Observed τ ≈ 0.36 is well within prior support

**Non-centered parameterization**:
- **Purpose**: Decorrelates μ, τ, θ in posterior when τ is small or n varies widely
- **When essential**: Groups with extreme sample sizes (n=47 vs. n=810) benefit from this
- **Alternative**: Centered parameterization (θ ~ Normal(μ, τ)) may work but risks divergences

### Strengths

1. **Theoretically well-justified**: Logit transformation ensures p ∈ (0,1), symmetry on transformed scale is plausible
2. **Handles heterogeneity naturally**: τ parameter directly models between-group variation
3. **Borrows strength appropriately**: Small-sample groups (1, 10) shrink toward μ; large groups (4, 12) shrink less
4. **Computational advantages**: Well-studied, efficient samplers in Stan/PyMC
5. **Interpretable parameters**: μ (population mean), τ (heterogeneity), θ_j (group-specific rates)
6. **Matches EDA prior recommendations**: Half-Cauchy(0,1) for τ was explicitly suggested

### Weaknesses

1. **Assumes symmetric tails**: Normal distribution may underestimate probability of extreme groups (2, 4, 8)
2. **Single variance parameter**: Cannot model different heterogeneity for high vs. low success rates
3. **Sensitive to outliers**: Extreme groups may inflate τ, over-shrinking moderate groups
4. **May underfit**: If true DGP has subpopulations or structure, single τ is inadequate
5. **Logit unboundedness**: For groups near 0% or 100%, logit(p) → ±∞ can cause numerical issues
6. **No mechanism for robustness**: One bad group affects entire hierarchy

### Falsification Criteria

#### 1. Prior Predictive Checks (BEFORE fitting)

**Test**: Simulate from prior, check if it's realistic.

```python
# PyMC pseudocode
with pm.Model() as prior_model:
    mu = pm.Normal('mu', -2.5, 1)
    tau = pm.HalfCauchy('tau', 1)
    z = pm.Normal('z', 0, 1, shape=12)
    theta = pm.Deterministic('theta', mu + tau * z)
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    prior_samples = pm.sample_prior_predictive(samples=1000)
```

**Success criteria**:
- ✅ 95% of prior predictive rates fall in [1%, 30%] (observed: [3%, 14%])
- ✅ Prior predictive τ concentrates mass in [0, 2] (observed: ~0.36)
- ✅ Prior predictive dispersion φ spans observed 3.59 (allows both under- and over-dispersion)

**Failure triggers abandonment if**:
- ❌ Prior predictive rates regularly exceed 50% or fall below 0.1%
- ❌ Prior cannot generate observed overdispersion (all φ < 2)
- ❌ Prior so diffuse that inference will be dominated by regularization, not data

#### 2. Simulation-Based Calibration (SBC)

**Test**: Fit model to data generated from prior, check if posteriors recover true parameters.

**Protocol**:
1. Draw (μ_true, τ_true, z_true) from priors
2. Simulate r_sim ~ Binomial(n, logit^(-1)(μ + τ·z))
3. Fit model to (n, r_sim)
4. Check if μ_true, τ_true in posterior credible intervals
5. Repeat 100 times

**Success criteria**:
- ✅ 90-95 of 100 simulations have μ_true in 95% CI
- ✅ 90-95 of 100 simulations have τ_true in 95% CI
- ✅ Rank statistics uniformly distributed (Kolmogorov-Smirnov test p > 0.05)

**Failure triggers investigation if**:
- ❌ Coverage < 85% (model systematically biased)
- ❌ Coverage > 98% (posteriors too wide, prior dominance)
- ❌ Rank statistics show clear patterns (identifiability issues)

#### 3. Posterior Predictive Checks (AFTER fitting)

**Test A: Overdispersion**

Compute posterior predictive distribution of dispersion parameter φ:
```
φ = Var(r_j/n_j) / Var_binomial(pooled_rate)
φ_obs = 3.59
```

**Success criteria**:
- ✅ φ_obs within 95% posterior predictive interval for φ
- ✅ Posterior predictive p-value ∈ [0.05, 0.95]
- ✅ Bayesian p-value ≈ 0.5 (model not systematically over- or under-predicting variance)

**Failure triggers abandonment if**:
- ❌ φ_obs outside 99% posterior predictive interval
- ❌ p-value < 0.01 or > 0.99 (extreme misfit)

**Test B: Extreme Group Prediction**

For each outlier group (2, 4, 8):
```
z_j = (r_j - E[r_rep_j]) / SD[r_rep_j]  [posterior predictive z-score]
```

**Success criteria**:
- ✅ |z_j| < 3 for all groups (within 99.7% of posterior predictive distribution)
- ✅ No systematic bias (mean of z_j ≈ 0)
- ✅ Outliers not consistently under- or over-predicted

**Failure triggers abandonment if**:
- ❌ |z_4| > 3 (group 4 with n=810 is systematically mispredicted)
- ❌ Groups 2 and 8 both z > 3 (high outliers not captured by model)
- ❌ More than 2 groups with |z| > 2.5

**Test C: Shrinkage Validation**

Compare posterior mean θ_j to MLE θ_j^MLE:
```
shrinkage_j = |θ_j - θ_j^MLE| / |μ - θ_j^MLE|  [proportion shrunk toward population mean]
```

**Success criteria** (from EDA expectations):
- ✅ Groups 1, 10 (small n): shrinkage 60-72%
- ✅ Groups 4, 12 (large n): shrinkage 19-30%
- ✅ Monotonic relationship: larger n → less shrinkage
- ✅ Group 4 (n=810) shrinks least of all groups

**Failure triggers concern if**:
- ❌ Large-sample groups shrink >50% (over-regularization)
- ❌ Small-sample groups shrink <40% (under-regularization)
- ❌ Non-monotonic relationship (model not respecting precision)

#### 4. LOO Cross-Validation

**Success criteria**:
- ✅ All Pareto-k < 0.7 (reliable LOO estimates)
- ✅ ΔLOO(hierarchical vs. pooled) > 10 (strong evidence against pooled)
- ✅ ΔLOO(hierarchical vs. unpooled) > 0 (parsimony favors hierarchical)
- ✅ Effective number of parameters p_loo ≈ 3-8 (between 2 for pooled, 12 for unpooled)

**Failure triggers abandonment if**:
- ❌ Pareto-k > 0.7 for >2 groups (LOO unreliable, influential observations)
- ❌ Pareto-k > 1.0 for any group (refit with K-fold required)
- ❌ ΔLOO(hierarchical vs. unpooled) < -4 (unpooled preferred, need different model)

#### 5. Convergence Diagnostics

**Success criteria**:
- ✅ R̂ < 1.01 for all parameters
- ✅ Effective sample size (ESS) > 400 for all parameters
- ✅ No divergent transitions (<1% after warmup)
- ✅ E-BFMI > 0.2 (no energy exploration issues)

**Failure triggers investigation if**:
- ❌ Divergent transitions >1% (try centered parameterization or stronger priors)
- ❌ R̂ > 1.05 for τ (poor mixing, may need longer chains)
- ❌ ESS < 100 for any parameter (insufficient posterior exploration)

### Expected Computational Challenges

1. **Funnel geometry**: When τ → 0, posterior concentrates in narrow funnel. Non-centered helps but may not eliminate divergences.
   - **Solution**: Monitor divergences; if >5%, try centered parameterization or tighter τ prior (e.g., Half-Normal(0, 0.5))

2. **Extreme sample size range**: Group 10 (n=97, r=3) vs. Group 4 (n=810, r=34) have vastly different precision.
   - **Solution**: Non-centered parameterization essential. Verify θ_4 and θ_10 have similar ESS.

3. **Label switching**: Not an issue (groups are labeled by data, not posterior).

4. **Slow mixing for τ**: Hierarchical SD often mixes slowly.
   - **Solution**: Run longer chains (4000+ iterations) or use informative prior if EDA strongly suggests τ ≈ 0.36.

5. **Outlier influence**: Groups 2, 4, 8 may pull τ to extreme values.
   - **Diagnostic**: Check τ posterior; if concentrates at prior tail (>1.5), outliers dominating. Consider Model 3 (robust).

### Decision Rules for Model 1

**Proceed to reporting if**:
- ✅ All falsification checks pass
- ✅ LOO strongly favors hierarchical over baselines
- ✅ Posterior predictive checks show good fit

**Investigate further if**:
- ⚠️ 1-2 falsification checks marginal but not failing
- ⚠️ Computational issues manageable with tuning

**Abandon Model 1 and try Model 2 or 3 if**:
- ❌ Multiple falsification checks fail
- ❌ Extreme groups systematically mispredicted
- ❌ τ posterior conflicts with prior or concentrates at extreme values

---

## Model 2: Beta-Binomial (SIMPLER ALTERNATIVE)

### Mathematical Specification

**Likelihood**:
```
r_j ~ BetaBinomial(n_j, α, β)  for j = 1, ..., 12
```

**Equivalent formulation**:
```
p_j ~ Beta(α, β)  [conjugate prior for each group's success rate]
r_j | p_j ~ Binomial(n_j, p_j)
```

**Parameterization** (mean-concentration):
```
μ_p = α / (α + β)         [population mean success rate]
κ = α + β                  [concentration parameter, inverse variance]

α = μ_p · κ
β = (1 - μ_p) · κ
```

**Priors**:
```
μ_p ~ Beta(5, 50)          [prior mean success rate ≈ 9%]
κ ~ Gamma(2, 0.1)          [weakly informative, allows overdispersion]
```

### Stan Implementation

```stan
data {
  int<lower=1> J;              // Number of groups (12)
  array[J] int<lower=0> n;     // Trials per group
  array[J] int<lower=0> r;     // Successes per group
}

parameters {
  real<lower=0, upper=1> mu_p;  // Population mean success rate
  real<lower=0> kappa;          // Concentration (inverse variance)
}

transformed parameters {
  real<lower=0> alpha = mu_p * kappa;
  real<lower=0> beta = (1 - mu_p) * kappa;
}

model {
  // Priors
  mu_p ~ beta(5, 50);
  kappa ~ gamma(2, 0.1);

  // Likelihood
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}

generated quantities {
  array[J] int<lower=0> r_rep;
  vector[J] log_lik;
  real phi_rep;  // Overdispersion parameter

  for (j in 1:J) {
    r_rep[j] = beta_binomial_rng(n[j], alpha, beta);
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha, beta);
  }

  // Compute overdispersion in post-processing
}
```

### Prior Justification

**μ_p ~ Beta(5, 50)**:
- **Mean**: 5/(5+50) = 9.1%, close to pooled rate 7.0%
- **95% interval**: [2%, 20%] (from quantile calculations)
- **Rationale**: Matches EDA recommendation, weakly informative
- **Shape**: Slightly right-skewed, consistent with bounded success rates

**κ ~ Gamma(2, 0.1)**:
- **Mean**: 2/0.1 = 20
- **Interpretation**: Effective "sample size" of prior information
- **Implies**: α ≈ 1.8, β ≈ 18.2 for μ_p = 0.09
- **Allows overdispersion**: Small κ → high variance; large κ → low variance
- **Weakly informative**: Wide prior allowing data to dominate

### Strengths

1. **Conjugacy**: Beta is conjugate prior for binomial, simplifies computation
2. **Natural overdispersion**: Beta-binomial directly models extra-binomial variation
3. **Bounded support**: Success rates automatically constrained to [0, 1]
4. **Interpretable parameters**: μ_p (average rate), κ (concentration/homogeneity)
5. **Computational efficiency**: Often faster than hierarchical logit-normal
6. **Robust to small counts**: No logit transformation issues near boundaries

### Weaknesses

1. **Population-level only**: Does not provide group-specific posterior estimates θ_j
2. **Less flexible**: Single concentration parameter may inadequately capture heterogeneity patterns
3. **No shrinkage estimates**: Cannot directly compare MLE vs. Bayes estimates per group
4. **Harder to extend**: Adding covariates or structure more complex than in logit models
5. **May underfit**: If groups have complex heterogeneity (e.g., bimodal), single Beta inadequate
6. **Overdispersion limits**: Beta-binomial has bounded overdispersion; may not capture φ=3.59 well

### Falsification Criteria

#### Prior Predictive Checks

**Success criteria**:
- ✅ Prior predictive rates in [1%, 30%]
- ✅ Prior allows for φ ≈ 3.59

**Failure if**:
- ❌ Prior predictive φ cannot exceed 2.0 (Beta-binomial constraint violated)

#### Posterior Predictive Checks

**Test A: Overdispersion**
- ✅ φ_obs = 3.59 within posterior predictive 95% interval
- ❌ **Failure if**: φ_pred consistently < 3.0 (Beta-binomial insufficiently flexible)

**Test B: Group-Level Fit**
- ✅ No systematic misprediction of extreme groups
- ❌ **Failure if**: Groups 2, 4, 8 consistently outside posterior predictive intervals

#### LOO Comparison

**Success criteria**:
- ✅ ΔLOO(Beta-binomial vs. pooled) > 10
- ⚠️ **Acceptable if**: ΔLOO(Hierarchical logit vs. Beta-binomial) < 4 (models comparable)
- ❌ **Failure if**: ΔLOO(Hierarchical logit vs. Beta-binomial) > 10 (Beta-binomial substantially worse)

### Decision Rules for Model 2

**Model 2 SUCCEEDS if**:
- ✅ Overdispersion captured adequately (φ_obs within posterior predictive)
- ✅ LOO comparable to or better than Model 1
- ✅ Computational advantages (faster, no divergences)

**Model 2 FAILS if**:
- ❌ Cannot capture observed overdispersion (φ_pred << 3.59)
- ❌ LOO substantially worse than Model 1 (ΔLOO > 10)
- ❌ Posterior predictive checks show systematic misfit

**Use Model 2 instead of Model 1 if**:
- ✅ Both models fit equally well (ΔLOO < 4)
- ✅ Model 2 computationally simpler and faster
- ✅ Group-specific estimates not scientifically required

---

## Model 3: Hierarchical with Robust Hyperprior (ROBUSTNESS CHECK)

### Mathematical Specification

**Likelihood**:
```
r_j ~ Binomial(n_j, p_j)  for j = 1, ..., 12
logit(p_j) = θ_j
```

**Robust hyperprior** (Student-t instead of Normal):
```
θ_j ~ StudentT(ν, μ, τ²)  [heavier tails than Normal]
```

**Hyperpriors**:
```
μ ~ Normal(-2.5, 1)
τ ~ Half-Cauchy(0, 1)
ν ~ Gamma(2, 0.1)          [degrees of freedom, controls tail heaviness]
```

**Interpretation of ν**:
- ν = 1: Cauchy (very heavy tails)
- ν = 3-5: Heavy tails (robust to outliers)
- ν > 30: Approximates Normal (Model 1)

### Stan Implementation

```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=1> nu;             // Degrees of freedom (lower=1 prevents Cauchy issues)
  vector[J] z;                  // Non-centered (requires custom transformation)
}

transformed parameters {
  vector[J] theta;
  for (j in 1:J) {
    theta[j] = mu + tau * z[j]; // Approximate; exact Student-t NCP complex
  }
}

model {
  // Hyperpriors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  nu ~ gamma(2, 0.1);

  // Robust hyperprior (Student-t)
  theta ~ student_t(nu, mu, tau);  // Centered parameterization

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);
  array[J] int<lower=0> r_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }
}
```

**Note**: Non-centered parameterization for Student-t is complex (requires custom transformation). Implementation may use centered parameterization and accept potential divergences, or approximate with scaled mixture of normals.

### Prior Justification

**ν ~ Gamma(2, 0.1)**:
- **Mean**: 20 (moderate tail heaviness)
- **Allows**: ν ∈ [3, 50+] (heavy tails to near-normal)
- **Rationale**: If outliers are genuine, ν will concentrate <10; if normal sufficient, ν → large
- **Data-driven**: Posterior of ν informs whether robust model needed

### Strengths

1. **Robustness to outliers**: Heavy tails reduce influence of extreme groups (2, 4, 8)
2. **Adaptive**: ν posterior tells us if outliers are structural or noise
3. **Protects hierarchical mean**: Extreme groups less likely to distort μ estimate
4. **Diagnostic value**: If ν → large, Model 1 is sufficient (falsifies need for robustness)

### Weaknesses

1. **Computational complexity**: Student-t hyperprior harder to sample, especially non-centered
2. **Identifiability**: ν, τ correlated in posterior (both control spread)
3. **Overfitting risk**: Extra parameter ν may overfit if outliers are genuine signal
4. **Interpretation**: Harder to explain "degrees of freedom" to stakeholders than "variance"
5. **May be unnecessary**: EDA shows only 3 outliers; normal hyperprior may suffice

### Falsification Criteria

#### Posterior Analysis

**Model 3 JUSTIFIED if**:
- ✅ ν posterior concentrates below 10 (heavy tails needed)
- ✅ LOO improvement over Model 1: ΔLOO(M3 vs. M1) > 4
- ✅ Posterior predictive checks for extreme groups improve

**Model 3 UNNECESSARY if**:
- ❌ ν posterior concentrates above 30 (degenerates to normal, use Model 1)
- ❌ ΔLOO(M3 vs. M1) < 2 (no improvement for extra complexity)
- ❌ μ, τ posteriors nearly identical to Model 1 (robust model adds nothing)

#### Convergence Diagnostics

**Success criteria**:
- ✅ R̂ < 1.01 for all parameters (especially ν)
- ✅ No divergent transitions >1%

**Failure if**:
- ❌ Divergent transitions >5% (model too complex for data)
- ❌ ν posterior multimodal or extremely diffuse (identifiability issues)

### Decision Rules for Model 3

**Use Model 3 if**:
- ✅ ν posterior clearly below 10
- ✅ LOO improvement over Model 1
- ✅ Scientific interest in outlier robustness

**Use Model 1 instead if**:
- ✅ ν posterior above 20 (robust model unnecessary)
- ✅ No LOO improvement
- ✅ Computational issues persist

---

## Baseline Models (NOT PRIMARY, for comparison only)

### Model 4: Pooled (Expected to Fail)

**Purpose**: Document inadequacy of assuming single success rate.

**Specification**:
```
r_j ~ Binomial(n_j, p)  [same p for all groups]
p ~ Beta(5, 50)
```

**Expected outcome**: χ² test already rejects (p<0.0001). Bayesian posterior predictive should show:
- ❌ φ_pred ≈ 1 (no overdispersion)
- ❌ Extreme groups grossly mispredicted
- ❌ LOO much worse than hierarchical

**Value**: Establishes baseline; quantifies improvement from hierarchical structure.

### Model 5: Unpooled (Expected to Overfit)

**Purpose**: Document need for partial pooling.

**Specification**:
```
r_j ~ Binomial(n_j, p_j)  [independent p_j for each group]
p_j ~ Beta(5, 50)         [no hierarchical structure]
```

**Expected outcome**:
- ✅ Fits data perfectly (saturated model)
- ❌ Overfits small-sample groups (1, 10)
- ❌ LOO worse than hierarchical (no parsimony)
- ❌ Posterior predictive intervals too wide for small-sample groups

**Value**: Demonstrates that borrowing strength improves inference.

---

## Model Comparison Strategy

### Stage 1: Prior Predictive Checks (Before Fitting)

For each model, simulate from prior and check:
1. **Plausibility**: Do prior predictive rates span [1%, 30%]?
2. **Flexibility**: Can prior generate φ ≈ 3.59?
3. **Informativeness**: Is prior sufficiently weak to allow data dominance?

**Decision**: Proceed to fitting only if prior predictive checks pass.

### Stage 2: Posterior Inference

Fit all models with:
- **Chains**: 4 independent chains
- **Iterations**: 2000 warmup + 2000 sampling = 4000 total per chain
- **Thinning**: None (modern MCMC doesn't require thinning)
- **Diagnostics**: Monitor R̂, ESS, divergences, E-BFMI

**Decision**: Flag models with convergence issues; investigate and refit if necessary.

### Stage 3: Posterior Predictive Checks

For each model:
1. **Overdispersion test**: Is φ_obs within posterior predictive 95% interval?
2. **Extreme group test**: Are groups 2, 4, 8 within posterior predictive intervals?
3. **Shrinkage test**: Does shrinkage follow expected pattern (small n → more shrinkage)?

**Decision**: Models failing any test are candidates for rejection.

### Stage 4: LOO Cross-Validation

Compute LOO for all models:
1. **Pareto-k diagnostic**: Flag groups with k > 0.7
2. **Pairwise comparison**: Compute ΔLOO and SE for all pairs
3. **Ranking**: Order models by LOO (higher = better predictive accuracy)

**Decision rules**:
- **Clear winner**: ΔLOO > 4 (±2 SE) → Prefer higher-LOO model
- **Tie**: |ΔLOO| < 4 → Prefer simpler model (fewer parameters)
- **LOO invalid**: Any Pareto-k > 1.0 → Use K-fold CV instead

### Stage 5: Final Model Selection

**Primary decision matrix**:

| Condition | Action |
|-----------|--------|
| Model 1 passes all checks, LOO best | **Report Model 1** |
| Model 1 fails PPC, Model 3 succeeds | **Report Model 3** (outliers need robustness) |
| Model 1 and 2 tie in LOO, both pass | **Report Model 2** (simpler) |
| All models fail PPC | **Reject all, propose mixture/structured models** |
| LOO invalid (k > 1.0) | **Use K-fold CV, reconsider model class** |

**Secondary considerations**:
- **Scientific interpretability**: Prefer models with clear parameter meanings
- **Computational feasibility**: Avoid models requiring excessive tuning
- **Stakeholder communication**: Simpler models easier to explain

---

## Simulation-Based Calibration (SBC) Protocol

**Purpose**: Validate that inference algorithm recovers true parameters from known data-generating process.

**Protocol**:
1. **Generate synthetic data**:
   ```
   For i = 1 to 100:
     - Draw (μ_true, τ_true) from priors
     - Draw z_j ~ Normal(0, 1) for j = 1, ..., 12
     - Compute θ_j = μ_true + τ_true * z_j
     - Simulate r_j ~ Binomial(n_j, logit^(-1)(θ_j))
   ```

2. **Fit model to synthetic data**:
   ```
   For each synthetic dataset:
     - Run MCMC with same settings as real data analysis
     - Extract posterior samples for μ, τ
   ```

3. **Check calibration**:
   ```
   For each parameter:
     - Compute rank of true value within posterior samples
     - Plot histogram of ranks (should be uniform)
     - Run Kolmogorov-Smirnov test for uniformity
   ```

**Success criteria**:
- ✅ Rank histograms visually uniform (no U-shape or ∩-shape)
- ✅ K-S test p > 0.05 for all parameters
- ✅ 90-95% of true values in 95% posterior intervals

**Failure triggers investigation if**:
- ❌ U-shaped rank histogram (underdispersed posteriors)
- ❌ ∩-shaped rank histogram (overdispersed posteriors)
- ❌ Coverage <85% or >98% (systematic bias or prior dominance)

---

## Computational Implementation Plan

### Software Stack

**Primary**: PyStan 3.x or PyMC 5.x
- **Rationale**: Mature, well-tested, strong community support
- **Stan advantages**: Efficient HMC, excellent diagnostics, fast for hierarchical models
- **PyMC advantages**: Pythonic API, easier prototyping, good visualization tools

**Supplementary**: ArviZ for diagnostics and visualization
- **Rationale**: Unified interface for posterior analysis across Stan/PyMC

### Workflow

1. **Data preparation**:
   ```python
   import numpy as np
   import pandas as pd

   data = pd.read_csv('data/binomial_data.csv')
   stan_data = {
       'J': len(data),
       'n': data['n'].values,
       'r': data['r'].values
   }
   ```

2. **Model fitting** (Stan example):
   ```python
   import stan

   model = stan.build(model_code, data=stan_data)
   fit = model.sample(num_chains=4, num_samples=2000, num_warmup=2000)
   ```

3. **Diagnostics**:
   ```python
   import arviz as az

   idata = az.from_pystan(posterior=fit)
   print(az.summary(idata, var_names=['mu', 'tau']))
   az.plot_trace(idata, var_names=['mu', 'tau'])
   ```

4. **Posterior predictive checks**:
   ```python
   r_rep = fit['r_rep']  # Posterior predictive replicates
   phi_obs = observed_dispersion(data)
   phi_rep = [compute_dispersion(r_rep[i]) for i in range(len(r_rep))]
   p_value = np.mean(phi_rep >= phi_obs)
   ```

5. **LOO comparison**:
   ```python
   loo_model1 = az.loo(idata_model1, pointwise=True)
   loo_model2 = az.loo(idata_model2, pointwise=True)
   comparison = az.compare({'Model 1': loo_model1, 'Model 2': loo_model2})
   ```

### Expected Runtime

- **Model 1 (Hierarchical Logit-Normal)**: ~30-60 seconds
- **Model 2 (Beta-Binomial)**: ~10-20 seconds
- **Model 3 (Robust)**: ~60-180 seconds
- **SBC (100 iterations)**: ~1-2 hours per model

---

## Red Flags and Escape Routes

### Red Flag 1: Posterior-Prior Conflict

**Signal**: τ posterior concentrates at extreme tail of prior (>1.5 on logit scale).

**Interpretation**: Data demand more heterogeneity than prior allows, or outliers inflating τ.

**Action**:
1. Check prior predictive: Does prior allow observed heterogeneity?
2. Fit Model 3 (robust) to see if outliers driving conflict
3. If conflict persists, consider:
   - Mixture model (two subpopulations)
   - Check for data errors in extreme groups

### Red Flag 2: Extreme Groups Systematically Mispredicted

**Signal**: Groups 2, 4, 8 all have posterior predictive |z| > 3.

**Interpretation**: Single-variance hierarchical model inadequate; groups may be genuinely different.

**Action**:
1. Fit Model 3 to see if robust model helps
2. If not, propose:
   - **Mixture model**: Two latent classes (high-rate vs. low-rate groups)
   - **Structured model**: Covariate explaining group 4 (largest sample) difference

### Red Flag 3: Computational Pathologies

**Signal**: >5% divergent transitions despite tuning, or ESS < 100 for key parameters.

**Interpretation**: Model geometry incompatible with HMC, or model misspecified.

**Action**:
1. Try centered vs. non-centered parameterization
2. Tighter prior on τ (e.g., Half-Normal(0, 0.5))
3. If issues persist, consider:
   - Simpler model (Beta-binomial)
   - Variational inference as diagnostic (if VI succeeds but HMC fails, suspect misspecification)

### Red Flag 4: LOO Unreliable

**Signal**: Pareto-k > 0.7 for >2 groups.

**Interpretation**: Influential observations; model may not generalize well.

**Action**:
1. Report which groups are influential
2. Use K-fold CV (K=12, leave-one-group-out) for model comparison
3. Investigate influential groups: Are they data errors? Genuinely different?

### Red Flag 5: All Models Fail

**Signal**: Every proposed model fails posterior predictive checks.

**Interpretation**: Data generation process fundamentally different from assumptions.

**Action**: **STOP AND RECONSIDER**. Propose entirely new model classes:
- **Mixture models**: Latent classes of groups
- **Dirichlet process**: Nonparametric number of clusters
- **Temporal/spatial models**: If ordering is actually meaningful despite EDA
- **Zero-inflation**: If success mechanism differs from count mechanism
- **Consult domain expert**: May reveal scientific structure missed in EDA

---

## Iteration Strategy

### If Model 1 Succeeds
1. ✅ Report Model 1 as primary result
2. ✅ Report Models 2, 3 as sensitivity analyses
3. ✅ Document why hierarchical logit-normal is appropriate

### If Model 1 Fails but Model 3 Succeeds
1. ✅ Report Model 3 as primary result
2. ✅ Explain why outliers require robust approach
3. ✅ Report posterior of ν to quantify tail heaviness
4. ⚠️ Investigate: Are extreme groups data errors or genuine?

### If Model 2 Succeeds but Model 1 Fails
1. ⚠️ Investigate: Why does Beta-binomial succeed where logit-normal fails?
2. ⚠️ Check: Is overdispersion within Beta-binomial limits?
3. ✅ If Beta-binomial genuinely better, report as primary
4. ⚠️ Consider: Expand to hierarchical Beta-binomial (group-specific α_j, β_j)

### If All Proposed Models Fail
1. ❌ Reject current model classes
2. 🔄 Return to EDA: Look for missed patterns
   - Recheck exchangeability assumption
   - Investigate groups 2, 4, 8 in detail
   - Look for temporal or collection batch effects
3. 🔄 Propose new model classes:
   - **Mixture model**: K=2 latent classes
   - **Beta-binomial with group-level structure**
   - **Dirichlet process mixture**
4. 🔄 Consult domain expert for scientific guidance

### If Computational Issues Persist
1. Try alternative parameterizations (centered vs. non-centered)
2. Try tighter priors (e.g., Half-Normal instead of Half-Cauchy for τ)
3. Try alternative samplers:
   - PyMC: Try NUTS tuning parameters
   - Stan: Try `adapt_delta=0.95` for divergences
4. If HMC fails entirely:
   - Try variational inference (ADVI) as diagnostic
   - Consider simpler model (Beta-binomial)
   - Question model specification fundamentally

---

## Success Criteria for Model Design Phase

**This design succeeds if**:
1. ✅ At least one model passes all falsification checks
2. ✅ Model comparison clearly identifies best model
3. ✅ Posterior estimates are scientifically interpretable
4. ✅ Computational implementation is feasible and robust

**This design fails if**:
1. ❌ All models rejected by posterior predictive checks
2. ❌ Model comparison inconclusive (all models equally bad)
3. ❌ Computational issues prevent reliable inference
4. ❌ Posterior estimates scientifically implausible

**If design fails**: Iterate with new model classes, potentially involving:
- Mixture models for subpopulations
- Structured models with discovered covariates
- Nonparametric approaches
- Consultation with domain experts

---

## Summary of Proposed Models

| Model | Class | Parameters | Complexity | Expected Performance | Falsification Risk |
|-------|-------|------------|------------|---------------------|-------------------|
| **Model 1** | Hierarchical Logit-Normal | μ, τ, θ_j (14 total) | Moderate | **Best** (matches EDA) | Low (standard approach) |
| **Model 2** | Beta-Binomial | μ_p, κ (2 total) | Low | Good (if overdispersion captured) | Moderate (may underfit) |
| **Model 3** | Hierarchical Robust | μ, τ, ν, θ_j (15 total) | High | Good (if outliers problematic) | Moderate (may be unnecessary) |
| **Model 4** | Pooled | p (1 total) | Very Low | **Poor** (already rejected) | High (expected to fail) |
| **Model 5** | Unpooled | p_j (12 total) | Moderate | Mediocre (overfits) | High (no regularization) |

**Primary recommendation**: Model 1, with Models 2-3 as sensitivity checks and Models 4-5 as baselines.

**Pivot plan**: If Model 1 fails, immediately investigate Model 3 (robustness). If both fail, propose mixture models.

---

## Key Deliverables from Model Design

1. ✅ **Mathematical specifications** for 5 models
2. ✅ **Stan/PyMC pseudocode** for implementation
3. ✅ **Prior justifications** with EDA alignment
4. ✅ **Falsification criteria** for each model
5. ✅ **Computational challenges** anticipated
6. ✅ **Decision rules** for model selection
7. ✅ **Iteration strategy** if models fail
8. ✅ **Red flags** triggering major pivots

**Next Steps**: Implement models in Stan/PyMC, run inference, execute falsification checks, and report results.

---

## Appendix: Mathematical Relationships

### Overdispersion Parameter φ

For binomial data with pooled rate p̄:
```
Var_binomial(r_j / n_j) = p̄(1 - p̄) / n̄  [expected under homogeneity]
Var_observed(r_j / n_j) = empirical variance

φ = Var_observed / Var_binomial
```

φ = 1: No overdispersion (binomial sufficient)
φ > 1: Overdispersion (heterogeneity present)
Observed: φ = 3.59 → strong overdispersion

### Intraclass Correlation (ICC)

```
ICC = τ² / (τ² + σ²_within)

For binomial on logit scale:
σ²_within ≈ π²/3 (logistic distribution variance)

ICC = 0.56 → τ² ≈ 1.1 → τ ≈ 1.05 on logit scale
```

But EDA estimates τ ≈ 0.36. Discrepancy may be due to:
- Approximation error (logistic vs. binomial variance)
- Extreme groups inflating ICC estimate
- Sample size weighting effects

**Resolution**: Let posterior inference determine τ; use EDA estimate as prior check.

### Shrinkage Formula

For group j with MLE θ̂_j and posterior mean θ̃_j:
```
shrinkage_j = (θ̂_j - θ̃_j) / (θ̂_j - μ)

Approximately:
shrinkage_j ≈ τ² / (τ² + σ²_j)

where σ²_j = 1 / (n_j · p̂_j · (1 - p̂_j)) on logit scale
```

Larger n_j → smaller σ²_j → less shrinkage (group estimate more precise).

---

**End of Model Design Document**

**Files Generated**:
- `/workspace/experiments/designer_1/proposed_models.md` (this document)

**Next Phase**: Implement models in Stan/PyMC and execute inference pipeline.
