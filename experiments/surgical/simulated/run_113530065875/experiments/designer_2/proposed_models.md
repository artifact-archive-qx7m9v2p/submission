# Alternative Bayesian Modeling Strategies - Designer #2
## Focus: Robust Approaches and Alternative Parameterizations

**Date**: 2025-10-30
**Designer**: Model Designer #2
**EDA Source**: `/workspace/eda/eda_report.md`

---

## Executive Summary

Based on EDA findings, I propose three alternative model classes that address specific data challenges:

1. **Robust Hierarchical with Student-t Hyperpriors** - Handles outlier groups (2, 4, 8) without excessive shrinkage
2. **Beta-Binomial Direct Parameterization** - Computationally simpler, avoids logit transformation issues
3. **Finite Mixture Model (2-Component)** - Explicitly models potential subpopulations

**Key Data Challenges These Address:**
- Group 4 dominance (29% of data, low outlier)
- Three extreme groups (2, 4, 8) with |z| > 2.5
- Wide dispersion (φ=3.59, 4.5-fold rate range)
- Small sample groups (n=47, 97) requiring different shrinkage

---

## Critical Thinking Framework

### What Could Go Wrong?
- **Outlier accommodation might be spurious** - Groups 2, 4, 8 might be genuine, not aberrant
- **Mixture model could be overfit** - We only have 12 groups; 2-component model has many parameters
- **Beta-binomial might underperform** - Logit scale is natural for hierarchical modeling
- **Group 4's dominance might bias everything** - 29% of data could anchor all estimates

### Decision Points for Model Class Changes
1. **If robust model doesn't improve LOO**: Outliers are genuine, not statistical aberrations
2. **If mixture model has degenerate components**: No subpopulation structure exists
3. **If beta-binomial's priors are too constraining**: Logit scale offers more flexibility
4. **If all models struggle with Group 4**: May need leave-one-group-out sensitivity analysis

### When to Abandon This Entire Approach
- **Prior-posterior conflict across all models** - Data fundamentally inconsistent with hierarchical structure
- **All Pareto k values > 0.7** - Suggests misspecification, not just outliers
- **Posterior predictive checks systematically fail** - Even accommodating outliers doesn't help
- **Computational issues persist across parameterizations** - Indicates deeper model inadequacy

---

## Model 1: Robust Hierarchical with Student-t Hyperpriors

### Motivation
Standard hierarchical assumes θ_i ~ Normal(μ, τ²), which downweights outliers aggressively. Groups 2, 4, 8 are 2.8-4.0 standard deviations from pooled rate. **If these are genuine biological variation** (not measurement errors), Student-t hyperpriors allow heavier tails while still pooling information.

### Mathematical Specification

```
Data:
  J = 12 groups
  n_i = trials in group i (47 to 810)
  r_i = successes in group i

Parameters:
  μ ∈ ℝ               # Population mean (logit scale)
  τ > 0               # Between-group scale (logit scale)
  ν ∈ (2, 30)         # Degrees of freedom for Student-t
  θ_i ∈ ℝ, i=1..J     # Group-level logit rates

Hierarchical Structure:
  θ_i ~ Student-t(ν, μ, τ)    # Heavy-tailed hyperprior
  r_i ~ Binomial(n_i, logit⁻¹(θ_i))

Priors:
  μ ~ Normal(-2.5, 1)          # Weakly informative
  τ ~ Half-Cauchy(0, 1)        # Standard for scale
  ν ~ Gamma(2, 0.1)            # Prior on degrees of freedom
                                # Mean ≈ 20, allows ν ∈ [4, 50]
```

### Stan Implementation

```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=2, upper=30> nu;      // Degrees of freedom
  vector[J] theta;
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  nu ~ gamma(2, 0.1);

  // Robust hierarchical structure
  theta ~ student_t(nu, mu, tau);  // Heavy tails accommodate outliers

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);
  vector[J] log_lik;
  vector[J] r_rep;                  // Posterior predictive draws

  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // Outlier diagnostics
  vector[J] z_score = (theta - mu) / tau;
}
```

### Prior Justification

1. **ν ~ Gamma(2, 0.1)**:
   - Mean ≈ 20 (moderately heavy tails)
   - SD ≈ 14 (allows data to determine tail behavior)
   - Lower bound of 2 ensures finite variance
   - Upper bound of 30 prevents effective Normal (ν → ∞)

2. **Why not just Normal?**:
   - With 3 outliers out of 12 groups (25%), Normal hyperprior forces choice: either shrink outliers heavily (lose signal) or increase τ (lose pooling)
   - Student-t offers "gentle shrinkage" - pulls outliers toward mean but not as aggressively

### Strengths vs Standard Hierarchical

1. **Accommodates genuine outliers**: Groups 2, 4, 8 won't be over-shrunk
2. **Adaptive tail behavior**: Data determines ν (heavy vs light tails)
3. **Maintains pooling**: Still borrows strength across groups
4. **Better LOO expected**: Outliers won't have high Pareto k values

### Weaknesses

1. **Extra parameter**: ν adds complexity, may not be identifiable with J=12
2. **Computational cost**: Student-t can be slower than Normal
3. **Might be overkill**: If outliers are just sampling noise, adds unnecessary complexity
4. **Interpretation harder**: What does ν=8 vs ν=20 mean scientifically?

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior for ν concentrates near 30**: Data doesn't support heavy tails; use Normal
2. **LOO is worse than standard hierarchical**: Complexity doesn't improve prediction
3. **Pareto k values still high for Groups 2, 4, 8**: Model still misspecifies these groups
4. **Posterior predictive checks fail**: Outlier accommodation didn't help fit
5. **θ estimates nearly identical to Normal version**: Robustness adds no value

**Success criteria:**
- ν posterior clearly < 20 (indicates heavy tails needed)
- LOO improvement > 4 (ΔELPD > 2×SE)
- Pareto k values < 0.5 for all groups
- Group 4's θ estimate less shrunk than in Normal model

### Expected Computational Challenges

1. **Divergent transitions**: Student-t can have complex geometry, especially for small ν
2. **Slow mixing**: θ and (μ, τ, ν) correlated; may need non-centered parameterization
3. **Initialization**: Bad inits could get stuck in tails
4. **ν identifiability**: With J=12, may need stronger prior or fix ν

**Mitigation strategies:**
- Use `adapt_delta=0.95`
- Try non-centered parameterization: `theta = mu + tau * theta_raw * sqrt(nu / (nu - 2))`
- Initialize ν at 10 (not at boundary)
- If ν posterior is flat, fix ν ∈ {5, 10, 15, 20} and compare LOO

---

## Model 2: Beta-Binomial Direct Parameterization

### Motivation
Hierarchical logit-normal requires transformations (logit/inv_logit) that can cause numerical issues. **Beta-binomial works directly on probability scale**, using conjugate structure for efficiency. With strong overdispersion (φ=3.59), beta-binomial's extra dispersion parameter might be more natural than logit-normal's τ.

### Mathematical Specification

```
Data:
  J = 12 groups
  n_i = trials in group i
  r_i = successes in group i

Parameters:
  α > 0, β > 0        # Beta distribution shape parameters
  θ_i ∈ (0,1), i=1..J # Group-level success probabilities

Hierarchical Structure:
  θ_i ~ Beta(α, β)              # Natural conjugate prior
  r_i ~ Binomial(n_i, θ_i)

Priors:
  α ~ Gamma(2, 0.2)             # Mean = 10, SD = 7.07
  β ~ Gamma(2, 0.02)            # Mean = 100, SD = 70.7

  # This induces:
  # E[θ] = α/(α+β) ≈ 0.09 (9%, close to observed 7%)
  # Var[θ] = αβ/[(α+β)²(α+β+1)] ≈ 0.0008
  # SD[θ] ≈ 0.028 (2.8%), allows for observed range [3%, 14%]
```

### Stan Implementation

```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
  vector<lower=0, upper=1>[J] theta;
}

model {
  // Priors on Beta shape parameters
  alpha ~ gamma(2, 0.2);        // Expect α ≈ 10
  beta ~ gamma(2, 0.02);        // Expect β ≈ 100

  // Hierarchical structure
  theta ~ beta(alpha, beta);    // Groups drawn from common Beta

  // Likelihood
  r ~ binomial(n, theta);
}

generated quantities {
  real mu_pop = alpha / (alpha + beta);           // Population mean
  real sigma_pop = sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)));
  real kappa = alpha + beta;                      // "Concentration" parameter

  vector[J] log_lik;
  vector[J] r_rep;

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], theta[j]);
  }

  // Overdispersion metric
  real expected_var = mu_pop * (1 - mu_pop);
  real actual_var = variance(theta);
  real phi = actual_var / expected_var;           // Should be > 1
}
```

### Alternative: Beta-Binomial Marginalized

For computational efficiency, can marginalize out θ_i:

```stan
model {
  alpha ~ gamma(2, 0.2);
  beta ~ gamma(2, 0.02);

  // Marginalized likelihood (no θ parameters)
  for (j in 1:J) {
    target += beta_binomial_lpmf(r[j] | n[j], alpha, beta);
  }
}
```

This is faster but loses group-level θ estimates.

### Prior Justification

1. **Gamma priors on (α, β)**:
   - Avoid boundary (α, β > 0)
   - Induced mean: α/(α+β) ≈ 0.09, close to pooled 7%
   - Allows wide range of dispersions
   - Weakly informative (high SD relative to mean)

2. **Why this parameterization?**:
   - EDA suggests rates in [3%, 14%] → Beta(10, 100) covers this
   - Concentration κ = α + β relates to effective sample size
   - Low κ → high variance (accommodates φ=3.59)

3. **Prior predictive check**:
   - Draw (α, β) from priors
   - Draw θ ~ Beta(α, β)
   - Should cover observed rate range

### Strengths vs Standard Hierarchical

1. **No transformation required**: Works on probability scale directly
2. **Conjugate structure**: Potential for analytical insights
3. **Natural dispersion parameter**: κ = α + β directly controls overdispersion
4. **Interpretability**: α, β have intuitive meaning (pseudo-counts)
5. **Computational speed**: Marginalized version is very fast

### Weaknesses

1. **Less flexible than logit-normal**: Beta distribution symmetric on logit scale only for special cases
2. **Boundary issues**: θ_i ∈ (0,1) forces strict positivity (can't model θ=0 or θ=1)
3. **Asymmetry**: If true distribution is skewed, Beta might not capture it well
4. **Group 4 with r=34, n=810**: θ ≈ 0.042 is far in Beta tail, might cause issues

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior for θ_4 (Group 4) is at boundary**: Beta can't handle low rates
2. **LOO is worse than logit-normal by >4**: Transformation flexibility matters
3. **Posterior predictive checks fail for extreme groups**: Beta's bounded support is too constraining
4. **Prior-posterior conflict**: Data wants α, β outside plausible range
5. **Dispersion estimate φ̂ << 3.59**: Model can't capture observed overdispersion

**Success criteria:**
- LOO within 2 of logit-normal (equivalent performance)
- Posterior predictive captures observed variance
- Faster computation (>2× speed)
- Clear interpretation of α, β in domain terms

### Expected Computational Challenges

1. **Boundary issues**: θ_i near 0 or 1 can cause sampling problems
2. **Correlation**: α and β are negatively correlated, can slow mixing
3. **Prior sensitivity**: Small changes in Gamma priors can shift α, β significantly
4. **Marginalized version loses information**: Can't examine group-level θ posteriors

**Mitigation strategies:**
- Use non-centered or "sum-to-constraint" parameterization
- Reparameterize: μ = α/(α+β), κ = α+β for better geometry
- Strong priors if identifiability issues arise
- Always compare marginalized vs full version

---

## Model 3: Finite Mixture Model (2-Component)

### Motivation
EDA identified three outlier groups (2, 4, 8) with distinct behavior. **What if there are genuinely two subpopulations of groups?** For example:
- **High-rate subpopulation**: Groups 1, 2, 8, 9, 11, 12 (rates 7-14%)
- **Low-rate subpopulation**: Groups 3, 4, 5, 6, 7, 10 (rates 3-7%)

This is **highly speculative** with J=12, but if true, would explain extreme heterogeneity better than single hierarchy.

### Mathematical Specification

```
Data:
  J = 12 groups
  n_i = trials in group i
  r_i = successes in group i

Parameters:
  π ∈ (0, 1)           # Mixture weight (proportion in component 1)
  μ₁, μ₂ ∈ ℝ           # Component means (logit scale), μ₁ < μ₂
  τ₁, τ₂ > 0           # Component scales (logit scale)
  z_i ∈ {1, 2}         # Component assignment for group i
  θ_i ∈ ℝ              # Group-level logit rate

Hierarchical Structure:
  z_i ~ Categorical({1-π, π})

  θ_i | z_i=1 ~ Normal(μ₁, τ₁²)     # Low-rate component
  θ_i | z_i=2 ~ Normal(μ₂, τ₂²)     # High-rate component

  r_i ~ Binomial(n_i, logit⁻¹(θ_i))

Priors:
  π ~ Beta(2, 2)                     # Symmetric, allows skewed mixture
  μ₁ ~ Normal(-3, 0.5)               # Low-rate: ~5%
  μ₂ ~ Normal(-2, 0.5)               # High-rate: ~12%
  τ₁ ~ Half-Cauchy(0, 0.5)           # Less within-component variance
  τ₂ ~ Half-Cauchy(0, 0.5)

Constraint: μ₁ < μ₂ for identifiability
```

### Stan Implementation

```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  real<lower=0, upper=1> pi;          // Mixture weight
  ordered[2] mu;                       // Ordered for identifiability
  vector<lower=0>[2] tau;
  vector[J] theta;
  simplex[2] lambda[J];                // Mixture probabilities per group
}

model {
  // Priors
  pi ~ beta(2, 2);
  mu[1] ~ normal(-3, 0.5);             // Low component
  mu[2] ~ normal(-2, 0.5);             // High component
  tau ~ cauchy(0, 0.5);

  // Mixture likelihood
  for (j in 1:J) {
    vector[2] lp;
    lp[1] = log(1 - pi) + normal_lpdf(theta[j] | mu[1], tau[1]);
    lp[2] = log(pi) + normal_lpdf(theta[j] | mu[2], tau[2]);
    target += log_sum_exp(lp);

    // Binomial likelihood
    r[j] ~ binomial_logit(n[j], theta[j]);
  }
}

generated quantities {
  vector[J] p = inv_logit(theta);
  vector[J] log_lik;
  int<lower=1, upper=2> z[J];          // MAP component assignment

  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);

    // Posterior component assignment
    real lp1 = log(1 - pi) + normal_lpdf(theta[j] | mu[1], tau[1]);
    real lp2 = log(pi) + normal_lpdf(theta[j] | mu[2], tau[2]);
    z[j] = (lp2 > lp1) ? 2 : 1;
  }

  // Component-level summaries
  real p_comp1 = inv_logit(mu[1]);
  real p_comp2 = inv_logit(mu[2]);
}
```

### Prior Justification

1. **π ~ Beta(2, 2)**:
   - Symmetric, favors balanced mixtures slightly
   - Allows for extreme imbalance if data supports it
   - Mean = 0.5, SD = 0.22

2. **Ordered μ constraint**:
   - Prevents label switching (component 1 is always low-rate)
   - Essential for MCMC convergence

3. **Narrow priors on μ₁, μ₂**:
   - Strong prior: components should be separated
   - μ₁ ≈ -3 → 4.7%, μ₂ ≈ -2 → 12%
   - Reflects EDA evidence of bimodality

4. **τ ~ Half-Cauchy(0, 0.5)**:
   - Within-component variance should be smaller than overall τ ≈ 0.36
   - If τ₁, τ₂ → τ, mixture collapses to single component

### Strengths vs Standard Hierarchical

1. **Explicitly models subpopulations**: If Groups 2, 4, 8 are from different processes
2. **Less shrinkage within components**: Outliers only compared to their subpopulation
3. **Testable hypothesis**: Can compare 1-component vs 2-component via LOO
4. **Scientific insight**: Discovering subpopulations is valuable, not just fitting

### Weaknesses

1. **Highly overparameterized**: 6 parameters (π, μ₁, μ₂, τ₁, τ₂, plus J assignments) for 12 groups
2. **Identifiability issues**: With J=12, may not have power to detect 2 components
3. **Might be spurious**: Apparent bimodality could be sampling noise
4. **Group 4's influence**: 29% of data in one group might dominate one component
5. **Computational nightmare**: Label switching, multimodality, slow convergence

### Falsification Criteria

**I will abandon this model if:**

1. **Components collapse**: Posterior μ₁ ≈ μ₂ or τ₁, τ₂ >> overall τ
2. **Highly uncertain assignments**: Mixture probabilities λ_j ≈ (0.5, 0.5) for most groups
3. **No LOO improvement**: ΔLOO < 2 compared to standard hierarchical
4. **One component is nearly empty**: π̂ < 0.1 or π̂ > 0.9 with wide uncertainty
5. **Computational failure**: Divergent transitions, R̂ > 1.05, label switching

**Success criteria:**
- Clear bimodal posterior for μ
- At least 3 groups assigned to each component with >80% probability
- LOO improvement > 4 over standard hierarchical
- No computational pathologies (R̂ < 1.01, no divergences)

### Expected Computational Challenges

1. **Label switching**: Components can swap during sampling → need post-hoc relabeling
2. **Multimodality**: Posterior has 2^J modes (all possible assignments)
3. **Slow mixing**: Discrete z_i hard to sample
4. **Divergent transitions**: Complex geometry near component boundaries
5. **Initialization critical**: Bad inits → stuck in local mode

**Mitigation strategies:**
- Use ordered constraint on μ
- Marginalize out z_i (use mixture likelihood, not explicit assignments)
- High adapt_delta (0.95 or 0.99)
- Multiple chains with overdispersed inits
- Consider variational inference for initial exploration
- **If this fails, it's a strong signal to abandon mixtures entirely**

---

## Model Comparison Strategy

### Computational Plan

1. **Fit all models independently** (4 chains, 2000 iterations each)
2. **Diagnose convergence**: R̂, ESS, divergences, tree depth
3. **Compare LOO**: All models on same data, use SE for significance
4. **Posterior predictive checks**: Visual and quantitative

### LOO Comparison

```
Expected LOO hierarchy (best to worst):
1. Mixture (if subpopulations exist) OR Robust (if outliers genuine)
2. Standard hierarchical (baseline)
3. Beta-binomial (if transformations don't matter)
4. Robust/Mixture (whichever fails)

Decision rules:
- ΔLOO > 4: Strong preference
- ΔLOO ∈ (2, 4): Moderate preference
- ΔLOO < 2: Models equivalent, use simplicity

Pareto k diagnostics:
- k < 0.5: No issues
- k ∈ (0.5, 0.7): Moderate concern, check predictive
- k > 0.7: Model misspecification for that group
```

### Posterior Predictive Checks

**Quantitative metrics** (for all models):
1. **Dispersion ratio**: E[φ̂] should be ≈ 3.59
2. **Coverage**: 95% CI should cover 11-12 groups
3. **Extreme groups**: P(r_rep ≥ r | r, n) for Groups 2, 4, 8
4. **Between-group variance**: Posterior SD should match observed SD ≈ 3.39%

**Visual checks**:
1. **Overlay plot**: Observed vs predicted success counts
2. **Residual plot**: Standardized residuals vs n_i
3. **Shrinkage plot**: MLE vs posterior mean by sample size
4. **QQ plot**: Empirical vs predicted quantiles

### Prior Predictive Checks

**Before fitting**, simulate from each prior:

```
For each model:
1. Draw (μ, τ, ...) from priors
2. Draw θ_i from hierarchical structure
3. Draw r_i ~ Binomial(n_i, logit⁻¹(θ_i))
4. Check:
   - Success rates in [0%, 20%] (reasonable)
   - Overdispersion φ ∈ [1, 10] (allows observed 3.59)
   - No extreme values (r > n, r < 0)

If prior predictive is too narrow → priors too informative
If prior predictive is too wide → priors too vague
```

### Sensitivity Analysis

**Critical for Group 4** (29% of data):

1. **Leave-one-group-out**: Fit each model with Group 4 removed
2. **Compare posteriors**: μ̂ with vs without Group 4
3. **If large shift**: Group 4 is overly influential, consider downweighting

**Prior sensitivity**:
- Vary prior SDs by factor of 2
- If posteriors shift substantially, priors are too informative

---

## Simulation-Based Calibration (SBC)

### Purpose
Validate that our samplers can recover known parameters. **Critical for mixture model** (complex geometry).

### Protocol

For each model:

1. **Generate synthetic datasets**:
   ```
   For i = 1 to 100:
     - Draw (μ, τ, ...) from priors
     - Draw θ_j from hierarchical model
     - Draw r_j ~ Binomial(n_j, logit⁻¹(θ_j))  # Use actual n_j
     - Fit model to synthetic (r_j, n_j)
     - Check: Does posterior contain true parameters?
   ```

2. **Diagnostic plots**:
   - **Rank histograms**: Should be uniform for each parameter
   - **Coverage**: 95% CI should contain truth ~95% of time
   - **Shrinkage**: Posterior SD < prior SD (confirms learning)

3. **Failure modes**:
   - Biased ranks → sampler has bias
   - Poor coverage → uncertainty underestimated
   - Flat ranks → sampler doesn't explore full posterior

**If SBC fails**: Don't trust results! Fix sampler or change model.

---

## When to Prefer Each Model

### Robust Hierarchical (Student-t)
**Choose this if:**
- Pareto k > 0.5 for Groups 2, 4, or 8 in standard hierarchical
- Posterior predictive checks show under-prediction of variance
- Scientific domain suggests outliers are meaningful (not errors)
- ν posterior is clearly < 20 (heavy tails needed)

**Avoid this if:**
- ν posterior concentrates near 30 (Normal is sufficient)
- Computational issues persist (divergences, low ESS)
- No LOO improvement over standard hierarchical

### Beta-Binomial
**Choose this if:**
- Computational speed is critical
- Interpretability on probability scale is important
- LOO is competitive with logit-normal (within 2)
- Domain experts prefer α, β parameterization

**Avoid this if:**
- Boundary issues arise (θ near 0 or 1)
- LOO is worse by >4
- Need flexibility of logit scale

### Finite Mixture
**Choose this if:**
- LOO improvement > 4 over standard hierarchical
- Component assignments are clear (>80% probability)
- Scientific hypothesis supports subpopulations
- μ₁ and μ₂ posteriors are well-separated

**Avoid this if:**
- Components collapse (μ₁ ≈ μ₂)
- High computational cost with no gain
- Assignments are uncertain (can't interpret)
- Evidence for subpopulations is weak (post-hoc storytelling)

---

## Red Flags and Escape Routes

### Major Red Flags (Stop and Reconsider Everything)

1. **All models have high Pareto k** (>0.7 for multiple groups)
   - **Interpretation**: Fundamental misspecification, not just outliers
   - **Escape route**: Check for temporal structure, batch effects, covariates
   - **Alternative**: Non-exchangeable models, regression structure

2. **Prior-posterior conflict across all models**
   - **Interpretation**: Data is incompatible with hierarchical structure
   - **Escape route**: Fit unpooled model, examine group-specific issues
   - **Alternative**: Fixed effects with regularization

3. **Computational failure in all approaches**
   - **Interpretation**: Model class is fundamentally wrong
   - **Escape route**: Simplify dramatically (pooled + overdispersion)
   - **Alternative**: Frequentist random effects, GEE

4. **Group 4 causes 80%+ shift in μ**
   - **Interpretation**: One group shouldn't dominate population estimate
   - **Escape route**: Downweight Group 4, fit separately
   - **Alternative**: Meta-analysis framework with group-level covariates

### Model-Specific Red Flags

**Robust Hierarchical:**
- ν posterior is bimodal → Multiple failure modes, model unstable
- ν̂ = 2 (boundary) → Priors too constraining
- ν̂ > 25 with narrow CI → Heavy tails not needed

**Beta-Binomial:**
- Posterior α, β > 1000 → Effectively no pooling
- Posterior α, β < 0.1 → Model at boundary
- θ̂_4 at boundary → Beta can't handle low rates

**Mixture:**
- π̂ ≈ 0.5 with all λ_j ≈ (0.5, 0.5) → Spurious mixture
- Label switching despite ordering → Computational artifact
- One component has 1-2 groups only → Overfitting

---

## Alternative Approaches If All Models Fail

### 1. Non-Hierarchical Alternatives
- **Pooled + overdispersion correction**: Quasibinomial, variance inflation
- **Unpooled**: Treat groups as fixed effects (no borrowing)
- **Bootstrap**: Empirical distribution of group rates

### 2. Different Hierarchical Classes
- **Continuous mixtures**: Dirichlet process prior (infinite mixtures)
- **Random slopes**: Add group-level covariates (if any emerge)
- **State-space**: If groups have temporal/spatial order

### 3. Model Averaging
- **Ensemble**: Weighted combination of models via stacking
- **BMA**: Bayesian model averaging with LOO weights

### 4. Domain Reassessment
- **Check data quality**: Are Groups 2, 4, 8 measurement errors?
- **Look for covariates**: Group characteristics that explain heterogeneity
- **Consult experts**: Is 4.5-fold range reasonable?

---

## Implementation Checklist

### Before Fitting
- [ ] Implement all three models in Stan
- [ ] Write prior predictive check code
- [ ] Set up LOO comparison framework
- [ ] Define posterior predictive metrics
- [ ] Prepare SBC infrastructure

### During Fitting
- [ ] Check convergence diagnostics (R̂, ESS)
- [ ] Monitor divergent transitions
- [ ] Track computational time
- [ ] Save all chains (don't discard warmup)

### After Fitting
- [ ] Compute LOO for all models
- [ ] Run posterior predictive checks
- [ ] Visualize shrinkage patterns
- [ ] Examine Pareto k values
- [ ] Assess sensitivity to Group 4

### Decision Making
- [ ] Compare ΔLOO with SE
- [ ] Check for red flags (see above)
- [ ] Validate chosen model with SBC
- [ ] Document rationale for final selection
- [ ] Plan sensitivity analyses

---

## Expected Timeline and Milestones

### Phase 1: Model Implementation (1 day)
- Code all three Stan models
- Test on simulated data
- Verify prior predictive checks

### Phase 2: Model Fitting (1-2 days)
- Fit standard hierarchical (baseline)
- Fit robust hierarchical
- Fit beta-binomial (fast)
- Fit mixture (slow, may fail)

### Phase 3: Model Comparison (1 day)
- LOO comparison
- Posterior predictive checks
- Convergence diagnostics
- Sensitivity analyses

### Phase 4: Validation (1 day)
- SBC for chosen model
- Leave-one-group-out checks
- Prior sensitivity
- Final decision

### Decision Points
- **End of Phase 2**: If all models fail to converge, pivot to simpler approaches
- **End of Phase 3**: If LOO differences are negligible, choose simplest model
- **During Phase 4**: If SBC fails, debug or change model class

---

## Summary of Falsification Criteria

| Model | Abandon If | Success If |
|-------|-----------|-----------|
| **Robust** | ν̂ > 25, no LOO gain, k > 0.7 | ν̂ < 15, ΔLOO > 4, k < 0.5 |
| **Beta-binomial** | Boundary issues, ΔLOO < -4 | ΔLOO ≈ 0, faster compute |
| **Mixture** | Components collapse, π̂ uncertain | Clear assignments, ΔLOO > 4 |
| **All** | High k everywhere, prior-post conflict | Good convergence, predictive checks pass |

---

## Conclusion

These three alternative models address specific challenges:
- **Robust**: Handles outliers without over-shrinking
- **Beta-binomial**: Computationally efficient, direct parameterization
- **Mixture**: Explicitly models subpopulations (high risk, high reward)

**Most likely outcome**: Robust hierarchical will improve over standard, mixture will fail due to small J. Beta-binomial will be competitive but not clearly better.

**Unexpected outcome that would change everything**: If Group 4 alone drives all heterogeneity, we may need to treat it separately (not hierarchically).

**Next steps**: Implement models, fit in parallel, compare LOO, be ready to pivot.

---

## File Locations

**This document**: `/workspace/experiments/designer_2/proposed_models.md`

**Stan models to create**:
- `/workspace/experiments/designer_2/model_robust_hierarchical.stan`
- `/workspace/experiments/designer_2/model_beta_binomial.stan`
- `/workspace/experiments/designer_2/model_mixture.stan`

**Analysis scripts to create**:
- `/workspace/experiments/designer_2/fit_models.py`
- `/workspace/experiments/designer_2/compare_models.py`
- `/workspace/experiments/designer_2/validate_models.py`

**EDA reference**: `/workspace/eda/eda_report.md`
