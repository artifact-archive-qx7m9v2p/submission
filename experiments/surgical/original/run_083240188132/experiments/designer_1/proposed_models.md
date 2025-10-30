# Proposed Bayesian Models for Overdispersed Binomial Data
## Designer 1: Robust Hierarchical Approaches

**Date**: 2025-10-30
**Data Context**: 12 groups, n=47-810, r=0-46, pooled rate 7.39%
**Key Challenges**: Strong overdispersion (φ=3.5-5.1), high heterogeneity (ICC=0.66), zero-event group, three outliers

---

## Executive Summary

I propose **three robust, well-established Bayesian model classes** for this overdispersed binomial data:

1. **Beta-Binomial Hierarchical Model** (PRIMARY RECOMMENDATION)
   - Direct modeling of overdispersion through beta-distributed success probabilities
   - Natural conjugate structure for binomial data
   - Handles zero-events and outliers through automatic shrinkage

2. **Random Effects Logistic Regression (GLMM)**
   - Standard hierarchical approach on logit scale
   - Gaussian random effects for between-group variation
   - Widely used and well-understood

3. **Hierarchical Binomial with t-distributed Effects** (ROBUST VARIANT)
   - Heavy-tailed alternative for handling outliers
   - More robust to extreme groups than Gaussian random effects
   - Adaptive shrinkage based on outlierness

All models use partial pooling to handle the zero-event group and extreme heterogeneity. Priority order: Model 1 > Model 2 > Model 3 (use Model 3 only if Models 1-2 show poor fit to outliers).

---

## Model 1: Beta-Binomial Hierarchical Model (PRIMARY)

### 1.1 Full Model Specification

**Likelihood:**
```
r_i | p_i, n_i ~ Binomial(n_i, p_i)           for i = 1,...,12
p_i | α, β ~ Beta(α, β)                        (group-level success probability)
```

**Hyperpriors:**
```
α ~ Gamma(2, 0.1)                              shape parameter (weakly informative)
β ~ Gamma(2, 0.1)                              shape parameter (weakly informative)
```

**Derived quantities:**
```
μ = α / (α + β)                                population mean success rate
κ = α + β                                      concentration (inverse variance)
σ² = μ(1-μ) / (κ + 1)                         population variance of p_i
φ = 1 + 1/κ                                    overdispersion factor
```

**Reparameterization (alternative, more interpretable):**
```
r_i | p_i, n_i ~ Binomial(n_i, p_i)
p_i | μ, κ ~ Beta(μκ, (1-μ)κ)
μ ~ Beta(2, 18)                                prior mean centered near 0.1 (10%)
κ ~ Gamma(2, 0.1)                              prior on concentration
```

### 1.2 Theoretical Justification

**Why Beta-Binomial?**
- **Conjugate structure**: Beta is the conjugate prior for binomial likelihood, providing mathematical elegance and computational stability
- **Natural overdispersion**: Beta-binomial is the marginal distribution when p_i varies according to a Beta distribution, exactly matching our conceptual model
- **Compound distribution**: Explicitly models two sources of variation:
  1. Binomial sampling variation (within groups)
  2. Beta variation in underlying rates (between groups)
- **Established theory**: Beta-binomial is the canonical model for overdispersed binomial data in biostatistics, ecology, and epidemiology

**Mathematical Foundation**:
The beta-binomial arises naturally when:
- True success probability varies across groups: p_i ~ Beta(α, β)
- Observations are binomial given p_i: r_i | p_i ~ Binomial(n_i, p_i)
- Marginalizing over p_i yields: r_i ~ BetaBinomial(n_i, α, β)

This compound distribution has variance: Var(r_i) = n_i μ (1-μ) [1 + (n_i-1)ρ] where ρ = 1/(α+β+1) is the intraclass correlation, directly matching our observed ICC=0.66.

### 1.3 How It Addresses Key Challenges

**Overdispersion (φ=3.5-5.1)**:
- Concentration parameter κ = α + β directly controls overdispersion
- Lower κ → higher variance → stronger overdispersion
- Expected κ ≈ 1/0.66 - 1 ≈ 0.5 given ICC=0.66
- Model will naturally estimate κ from data to match observed φ

**Group Heterogeneity (ICC=0.66)**:
- Beta distribution allows p_i to vary substantially across groups
- ICC = 1/(κ+1), so ICC=0.66 implies κ ≈ 0.5
- Model directly estimates between-group variance through κ
- Partial pooling: groups shrink toward μ proportional to their precision

**Zero-Event Group (Group 1: 0/47)**:
- Beta prior prevents p_1 = 0 exactly (Bayesian regularization)
- Posterior for p_1 will be Beta(α + 0, β + 47) = Beta(α, β + 47)
- Heavy shrinkage toward population mean μ = α/(α+β)
- Expected posterior mean ≈ α/(α+β+47) ≈ 2/(2+0.1*20+47) ≈ 0.04 (4%)
- Uncertainty properly quantified (wide credible interval)

**Outliers (Groups 2, 8, 11)**:
- Hierarchical shrinkage reduces extreme estimates toward μ
- Amount of shrinkage inversely proportional to sample size and consistency
- Group 8 (31/215, 14.4%) with large n will shrink less if genuinely different
- β hyperparameter adapts to accommodate high-rate groups
- Posterior predictive checks will reveal if outliers are poorly fit

### 1.4 Prior Specification Strategy

**Prior Choice Philosophy**: Weakly informative, allowing data to dominate while preventing pathological values.

**Hyperprior on α, β: Gamma(2, 0.1)**
- **Shape = 2**: Mild regularization away from 0, avoiding infinite variance
- **Rate = 0.1**: Expected value E[α] = E[β] = 20
- **Implied prior on μ**: α/(α+β) ~ centered near 0.5 with wide dispersion
- **Implied prior on κ**: κ = α + β ~ Gamma-like, allowing wide range (0.1 to 100+)
- **Justification**:
  - Allows κ << 1 (strong overdispersion) or κ >> 1 (weak overdispersion)
  - Does not impose strong belief about mean rate
  - Computational: keeps α, β > 0 (required for Beta distribution)

**Alternative Prior (Reparameterized)**: μ ~ Beta(2, 18), κ ~ Gamma(2, 0.1)
- **Beta(2, 18) on μ**: Centers prior mean near 0.1 (10%), matching pooled estimate
  - E[μ] = 2/(2+18) = 0.1
  - SD[μ] ≈ 0.06, so 95% prior interval ≈ (0.02, 0.25)
  - Weakly informative: consistent with rates 2-25%
- **Gamma(2, 0.1) on κ**: Same as above, allows data to determine concentration
- **Justification**:
  - Incorporates weak domain knowledge (rates typically 5-15% in similar contexts)
  - More efficient sampling (μ directly interpretable)
  - Recommended if prior information available

**Prior Predictive Implications**:
- Under Gamma(2, 0.1) priors: simulated datasets will show wide variation in group rates (0-50%) and varying overdispersion
- Under Beta(2,18) on μ: simulated rates centered near 10% but still allowing 2-25%
- Prior predictive check MUST show: generated data covers observed range (0-15%)

**Sensitivity Analysis**: Fit model with alternative priors:
- μ ~ Beta(1, 1) [uniform], κ ~ Gamma(1, 0.1) [more vague]
- μ ~ Beta(5, 45) [stronger prior at 10%], κ ~ Gamma(2, 0.5) [tighter]
- Compare posteriors: if substantially different, priors may be influencing results

### 1.5 PyMC Implementation Plan

```python
import pymc as pm
import numpy as np

# Data
groups = np.arange(1, 13)
n = np.array([47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360])
r = np.array([0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24])
n_groups = len(groups)

# Model 1: Beta-Binomial Hierarchical (Standard Parameterization)
with pm.Model() as model_bb_standard:
    # Hyperpriors
    alpha = pm.Gamma("alpha", alpha=2, beta=0.1)
    beta = pm.Gamma("beta", alpha=2, beta=0.1)

    # Group-level probabilities
    p = pm.Beta("p", alpha=alpha, beta=beta, shape=n_groups)

    # Likelihood
    obs = pm.Binomial("obs", n=n, p=p, observed=r)

    # Derived quantities
    mu = pm.Deterministic("mu", alpha / (alpha + beta))
    kappa = pm.Deterministic("kappa", alpha + beta)
    sigma_sq = pm.Deterministic("sigma_sq", mu * (1 - mu) / (kappa + 1))
    phi = pm.Deterministic("phi", 1 + 1/kappa)

    # Posterior predictive
    pp_check = pm.Binomial("pp_check", n=n, p=p, shape=n_groups)


# Model 1 Alternative: Beta-Binomial (Mean-Concentration Parameterization)
with pm.Model() as model_bb_reparam:
    # Hyperpriors (more interpretable)
    mu = pm.Beta("mu", alpha=2, beta=18)  # prior mean centered at 0.1
    kappa = pm.Gamma("kappa", alpha=2, beta=0.1)  # concentration

    # Reparameterize to alpha, beta
    alpha_param = mu * kappa
    beta_param = (1 - mu) * kappa

    # Group-level probabilities
    p = pm.Beta("p", alpha=alpha_param, beta=beta_param, shape=n_groups)

    # Likelihood
    obs = pm.Binomial("obs", n=n, p=p, observed=r)

    # Derived quantities
    sigma_sq = pm.Deterministic("sigma_sq", mu * (1 - mu) / (kappa + 1))
    phi = pm.Deterministic("phi", 1 + 1/kappa)
    icc = pm.Deterministic("icc", 1 / (kappa + 1))

    # Posterior predictive
    pp_check = pm.Binomial("pp_check", n=n, p=p, shape=n_groups)


# Sampling strategy
with model_bb_reparam:
    # Prior predictive check
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)

    # MCMC sampling
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.95,  # higher for complex posteriors
        random_seed=42,
        return_inferencedata=True
    )

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

    # Convergence diagnostics
    # Check: Rhat < 1.01, ESS > 400 per chain, no divergences
```

**Implementation Notes**:
- Use reparameterized version (mu, kappa) for better sampling efficiency
- Beta distribution is well-behaved in PyMC, typically samples well
- May need higher target_accept (0.95) if seeing divergences
- Concentration parameter κ may have slow mixing if poorly constrained

### 1.6 Falsification Criteria (I will abandon this model if...)

**Statistical Red Flags**:

1. **Prior-Posterior Conflict**:
   - If posterior mean of μ falls outside prior 95% credible interval AND prior was already weak
   - Suggests fundamental model misspecification
   - Action: Reconsider likelihood structure

2. **Extreme Posterior Concentration**:
   - If posterior κ → 0 (< 0.01) or κ → ∞ (> 1000)
   - κ → 0: Data show even MORE variation than beta-binomial can accommodate
   - κ → ∞: Data are actually consistent with simple binomial (no overdispersion)
   - Either case suggests beta-binomial is wrong model class

3. **Poor Posterior Predictive Fit**:
   - If observed r_i falls outside 95% posterior predictive interval for 3+ groups (25%)
   - Particularly bad if outliers (Groups 2, 8, 11) are systematically underestimated
   - Systematic bias in posterior predictive checks (e.g., all predictions too low)
   - Suggests beta distribution inadequate for between-group variation

4. **Zero-Event Group Pathology**:
   - If posterior P(p_1 < 0.01) > 0.95 (model believes Group 1 truly has near-zero rate)
   - Conflicts with hierarchical assumption that groups are exchangeable
   - May indicate Group 1 is fundamentally different population

5. **Outlier Accommodation Failure**:
   - If posterior P(p_8 > 0.20) > 0.95 (Group 8 rate wildly different from others)
   - If κ is so small that all groups are essentially independent (no pooling benefit)
   - Suggests mixture model may be more appropriate

6. **Computational Failures**:
   - Persistent divergences (>1% of samples) despite tuning
   - Rhat > 1.05 for any parameter
   - Effective sample size < 100 per chain after 4000 samples
   - Often indicates fundamental geometry problem in posterior

**Domain/Scientific Red Flags**:

7. **Biologically Implausible Posteriors**:
   - If credible intervals for any p_i include values outside (0.001, 0.5)
   - Event rates >50% or <0.1% would contradict domain knowledge
   - Would trigger re-examination of data and model

8. **Overdispersion Factor Mismatch**:
   - If posterior φ differs dramatically from EDA estimate (3.5-5.1)
   - E.g., posterior median φ < 2.0 or φ > 10
   - Suggests model not capturing observed variance structure

**Decision Rules**:
- **REJECT if**: Criteria 1, 2, 3, or 6 occur
- **REVISE if**: Criteria 4, 5, or 8 occur (try alternative priors or mixture model)
- **INVESTIGATE if**: Criterion 7 occurs (check data and domain knowledge)

### 1.7 Expected Computational Challenges

**Likely Issues**:

1. **Concentration Parameter Mixing**:
   - κ (or α, β jointly) can be difficult to estimate with only 12 groups
   - Posterior may have heavy tails or multimodality
   - **Mitigation**:
     - Use reparameterized version (mu, kappa)
     - Increase target_accept to 0.95 or 0.99
     - Check trace plots carefully for poor mixing

2. **Zero-Event Group**:
   - Group 1 (0/47) may cause extreme posterior concentration for p_1
   - Could lead to boundary behavior in Beta distribution
   - **Mitigation**:
     - Prior prevents exact p_1 = 0
     - Monitor p_1 trace for proper exploration
     - Consider adding small continuity correction if numerical issues arise

3. **Correlation in Posterior**:
   - α and β are likely highly negatively correlated
   - μ and κ reparameterization reduces but doesn't eliminate correlation
   - **Mitigation**:
     - Use non-centered parameterization if standard approach struggles
     - Accept that ESS may be lower for hyperparameters
     - Focus on derived quantities (mu, phi) which may sample better

4. **Divergences**:
   - Possible in regions where α or β approach 0
   - More likely with only 12 groups (weak information about hyperparameters)
   - **Mitigation**:
     - Increase adapt_delta (target_accept) to 0.95+
     - Try tighter priors if divergences persist
     - Check if divergences concentrate in specific parameter regions

5. **Computation Time**:
   - 12 Beta draws per iteration is fast
   - Total runtime: ~2-5 minutes for 2000 draws × 4 chains
   - **Not a significant concern for this model**

**Sampling Strategy**:
- Start with 1000 tune, 1000 draws, 2 chains for quick diagnostics
- If Rhat < 1.01 and no divergences: proceed with full run (2000 tune, 2000 draws, 4 chains)
- If issues: increase target_accept incrementally (0.90 → 0.95 → 0.99)
- Last resort: reparameterize using non-centered approach

**Success Metrics**:
- Rhat < 1.01 for all parameters
- ESS > 400 per chain (1600 total for 4 chains)
- Divergences < 1% of post-warmup samples
- Visual trace plots show good mixing (no trends, good coverage)

---

## Model 2: Random Effects Logistic Regression (GLMM)

### 2.1 Full Model Specification

**Likelihood (on logit scale):**
```
r_i | θ_i, n_i ~ Binomial(n_i, logit^(-1)(θ_i))     for i = 1,...,12
θ_i | μ, τ ~ Normal(μ, τ²)                           (group-level log-odds)
```

**Hyperpriors:**
```
μ ~ Normal(logit(0.075), 1²)                         population mean log-odds
τ ~ HalfNormal(1)                                     between-group SD (log-odds scale)
```

**Derived quantities:**
```
p_i = logit^(-1)(θ_i) = 1 / (1 + exp(-θ_i))         group-level probabilities
p_mean = logit^(-1)(μ)                                population mean probability
σ²_logit = τ²                                         between-group variance (log-odds)
```

**Alternative Prior Specification:**
```
μ ~ Normal(-2.5, 1.5²)                               vague prior centered at ~8%
τ ~ Exponential(1)                                    exponential prior on SD
```

### 2.2 Theoretical Justification

**Why Random Effects Logistic?**
- **Standard approach**: Most widely used model for grouped binomial data
- **Familiar parameterization**: Log-odds scale is well-understood in regression contexts
- **Gaussian random effects**: Assumes log-odds of group probabilities are normally distributed
- **Extensibility**: Easy to add covariates (fixed effects) if group-level predictors available
- **Computational**: Well-established in software (PyMC, Stan, lme4, etc.)

**Mathematical Foundation**:
The logistic GLMM assumes:
- Within-group: r_i | p_i ~ Binomial(n_i, p_i)
- Between-group: logit(p_i) ~ Normal(μ, τ²)
- This induces overdispersion because p_i varies across groups

On the log-odds scale:
- θ_i = logit(p_i) = log(p_i / (1 - p_i))
- Gaussian distribution on θ_i is unbounded (-∞, +∞)
- Inverse logit transformation maps to (0, 1) for probabilities

**Relationship to Overdispersion**:
The overdispersion factor φ depends on both τ² and the position on the probability scale. For moderate probabilities (5-15%), τ ≈ 0.5 to 1.0 on the log-odds scale typically produces φ ≈ 2-5.

### 2.3 How It Addresses Key Challenges

**Overdispersion (φ=3.5-5.1)**:
- Between-group variance τ² directly induces extra-binomial variation
- Expected τ ≈ 0.7-1.0 to achieve φ = 3.5-5.1 at p ≈ 0.07
- Model will estimate τ from data, allowing posterior φ to match observed
- Posterior predictive checks will validate if estimated τ produces correct overdispersion

**Group Heterogeneity (ICC=0.66)**:
- τ² captures all between-group variation on log-odds scale
- Large τ (>0.5) allows groups to differ substantially
- ICC on probability scale depends on τ and mean level
- Partial pooling: groups shrink toward μ proportional to (τ² + σ²_i)^(-1)

**Zero-Event Group (Group 1: 0/47)**:
- Gaussian prior on θ_1 prevents -∞ (equivalent to p_1 = 0)
- Posterior for θ_1 will be strongly negative but finite
- Hierarchical shrinkage pulls θ_1 toward population mean μ
- Expected posterior: θ_1 ≈ μ - 1.5 to μ - 2 (depending on τ)
- Transformed to probability: p_1 ≈ 2-4% (not 0%)

**Outliers (Groups 2, 8, 11)**:
- High-rate groups have large positive θ_i (> μ + τ)
- Gaussian random effects provide shrinkage toward μ
- Shrinkage less extreme than if τ were small
- Group 8 (14.4%) requires θ_8 ≈ μ + 1.5τ to μ + 2τ
- If τ ≈ 0.8, this is plausible under Normal(μ, 0.8²)
- **Potential issue**: Gaussian tails may be too light if outliers are extreme

### 2.4 Prior Specification Strategy

**Prior Choice Philosophy**: Weakly informative, regularizing extreme log-odds while allowing data to dominate.

**Hyperprior on μ: Normal(logit(0.075), 1²)**
- **Center**: logit(0.075) ≈ -2.52 (7.5% probability, near pooled estimate)
- **Scale**: SD = 1 on log-odds scale
  - Implies 95% prior interval: μ ∈ (-4.5, -0.5)
  - Corresponds to probabilities: (1%, 38%)
- **Justification**:
  - Centers on pooled estimate but allows wide range
  - Prevents pathological values (< 0.1% or > 50%)
  - Weakly informative: data will dominate with 12 groups

**Alternative**: Normal(-2.5, 1.5²) [more vague, centered at ~8%]
- Wider SD allows more uncertainty
- 95% prior: (-5.5, 0.5) → probabilities (0.4%, 62%)
- Use if want to be more agnostic about mean level

**Hyperprior on τ: HalfNormal(1)**
- **Scale**: SD = 1 on log-odds scale
  - E[τ] ≈ 0.80 (expected between-group SD)
  - 95% prior interval: (0, 2.0)
- **Justification**:
  - Allows wide range of heterogeneity (τ = 0.1 to 2.0)
  - Weakly regularizes away from extreme τ → ∞
  - τ = 1 implies groups can differ by ~2 log-odds units (e.g., 5% vs 20%)
- **Alternative**: Exponential(1)
  - E[τ] = 1.0, more heavy-tailed
  - Allows larger τ more easily (useful if heterogeneity very strong)

**Prior Predictive Implications**:
- Under these priors: simulated group rates will range 1-30% with most 3-15%
- SD of simulated rates ≈ 3-5%, matching observed CV ≈ 0.52
- Prior predictive check MUST show: coverage of observed range (0-15%)

**Sensitivity Analysis**:
- Fit with τ ~ Exponential(0.5) [stronger regularization]
- Fit with τ ~ Exponential(1) [weaker regularization]
- Compare posteriors for μ, τ, and group-specific p_i

### 2.5 PyMC Implementation Plan

```python
import pymc as pm
import numpy as np
from scipy.special import logit, expit

# Data
groups = np.arange(1, 13)
n = np.array([47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360])
r = np.array([0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24])
n_groups = len(groups)

# Model 2: Random Effects Logistic Regression
with pm.Model() as model_logistic_re:
    # Hyperpriors
    mu_logit = pm.Normal("mu_logit", mu=logit(0.075), sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Group-level random effects (non-centered parameterization)
    theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic("theta", mu_logit + tau * theta_raw)

    # Transform to probability scale
    p = pm.Deterministic("p", pm.math.invlogit(theta))

    # Likelihood
    obs = pm.Binomial("obs", n=n, p=p, observed=r)

    # Derived quantities
    p_mean = pm.Deterministic("p_mean", pm.math.invlogit(mu_logit))
    sigma_sq_logit = pm.Deterministic("sigma_sq_logit", tau**2)

    # Approximate overdispersion factor (at mean probability)
    # phi ≈ 1 + n_mean * p_mean * (1 - p_mean) * tau^2
    n_mean = n.mean()
    phi_approx = pm.Deterministic(
        "phi_approx",
        1 + n_mean * p_mean * (1 - p_mean) * tau**2
    )

    # Posterior predictive
    pp_check = pm.Binomial("pp_check", n=n, p=p, shape=n_groups)


# Sampling strategy
with model_logistic_re:
    # Prior predictive check
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)

    # MCMC sampling (non-centered parameterization should sample well)
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.90,  # standard for logistic models
        random_seed=42,
        return_inferencedata=True
    )

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

    # Convergence diagnostics
    # Check: Rhat < 1.01, ESS > 400 per chain, no divergences
```

**Implementation Notes**:
- **Non-centered parameterization** (theta_raw) crucial for efficient sampling
- Standard centered parameterization (theta ~ Normal(mu, tau)) often has sampling issues
- Non-centered separates location (mu) from scale (tau), reducing posterior correlation
- This is well-established best practice for hierarchical models

**Non-Centered Explanation**:
- Centered: θ_i ~ Normal(μ, τ) — direct sampling, but μ and τ highly correlated
- Non-centered: θ_i = μ + τ * z_i where z_i ~ Normal(0, 1)
  - Samples z_i (standardized effects) independently of μ, τ
  - Reduces posterior correlation, improves mixing
  - Essential when data provide weak information about τ (as with 12 groups)

### 2.6 Falsification Criteria (I will abandon this model if...)

**Statistical Red Flags**:

1. **Extreme Between-Group Variance**:
   - If posterior τ > 2.0 (log-odds SD > 2)
   - Implies groups differ by 4+ log-odds units (e.g., 1% vs 50%)
   - Gaussian random effects may be inappropriate; suggests mixture model
   - Action: Switch to heavy-tailed random effects (Model 3)

2. **Poor Fit to Outliers**:
   - If Groups 2, 8, 11 consistently fall outside 95% posterior predictive intervals
   - Gaussian tails too light to accommodate extreme groups
   - Posterior predictive p-values < 0.025 for all three outliers
   - Action: Use Student-t random effects or mixture model

3. **Zero-Event Group Pathology**:
   - If posterior P(p_1 < 0.005) > 0.95 (believes Group 1 has <0.5% rate)
   - If θ_1 posterior extends to -5 or below (p < 0.7%)
   - Suggests Group 1 is fundamentally different, not just low by chance
   - Action: Investigate Group 1 data quality, consider excluding

4. **Overdispersion Mismatch**:
   - If posterior predictive φ < 2.0 or φ > 10 (far from observed 3.5-5.1)
   - Model not capturing observed variance structure
   - Particularly problematic if PPCs show systematic underdispersion
   - Action: Reconsider likelihood or variance structure

5. **Computational Failures**:
   - Persistent divergences (>1% samples) despite non-centered parameterization
   - Rhat > 1.05 for θ_i or τ
   - Often indicates model misspecification rather than just tuning issue
   - Action: Check data, simplify model, or switch model class

6. **Systematic Bias in Predictions**:
   - If posterior predictive consistently underestimates high-rate groups
   - Or overestimates low-rate groups
   - Suggests systematic lack of fit, not just noise
   - Action: Add group-level covariates or use more flexible model

**Domain Red Flags**:

7. **Implausible Posteriors**:
   - If any p_i credible interval includes <0.1% or >30%
   - Would contradict domain knowledge about event rates
   - Action: Investigate data, check model, consider informative priors

8. **Shrinkage Pathology**:
   - If high-rate outliers shrink toward mean so much that their posteriors overlap typical groups
   - Loss of genuine signal, over-pooling
   - Suggests τ underestimated or model inappropriate
   - Action: Check τ posterior, consider less informative prior

**Decision Rules**:
- **REJECT if**: Criteria 1, 2, 4, or 5 occur
- **REVISE if**: Criteria 3 or 6 occur (adjust priors, add covariates)
- **SWITCH TO MODEL 3 if**: Criterion 2 occurs (outlier accommodation issue)

### 2.7 Expected Computational Challenges

**Likely Issues**:

1. **Funnel Geometry** (if using centered parameterization):
   - μ and τ posteriors correlated, especially when τ near 0
   - Creates "funnel" in posterior: narrow when τ small, wide when τ large
   - **Mitigation**: Use non-centered parameterization (already implemented)
   - Non-centered essentially eliminates this issue

2. **Zero-Event Group**:
   - Group 1 may pull θ_1 → -∞ (but regularized by prior)
   - Can cause numerical instability in logit transformation
   - **Mitigation**:
     - Non-centered parameterization helps
     - Monitor θ_1 trace for extreme values
     - If θ_1 < -5 frequently, may need stronger prior

3. **Weak Information on τ**:
   - With only 12 groups, τ posterior may be wide and skewed
   - Possible multimodality if some groups very different from others
   - **Mitigation**:
     - Accept wider credible intervals on τ
     - Focus on group-specific p_i which will be better estimated
     - Consider sensitivity analysis with different τ priors

4. **Computational Efficiency**:
   - Non-centered parameterization samples very efficiently
   - Expected runtime: ~2-5 minutes for 2000 draws × 4 chains
   - Logistic models generally sample well in PyMC
   - **Not a significant concern with non-centered approach**

5. **Divergences** (unlikely with non-centered):
   - May occur if τ → 0 or if θ_i very extreme
   - Non-centered parameterization makes divergences rare
   - If occur: increase target_accept to 0.95
   - **Expected divergence rate: <0.1% with non-centered**

**Sampling Strategy**:
- Non-centered parameterization should "just work" with default settings
- Start with 1000 tune, 1000 draws, 2 chains for quick check
- If diagnostics good: proceed with full run (2000 tune, 2000 draws, 4 chains)
- If issues (rare): increase target_accept incrementally

**Success Metrics**:
- Rhat < 1.01 for all parameters
- ESS > 400 per chain
- Divergences < 0.1% (expect zero with non-centered)
- Smooth trace plots, good mixing

**Advantages Over Model 1**:
- More standard parameterization (log-odds familiar)
- Better computational properties with non-centered approach
- Easier to extend with covariates
- More widely understood in applied research

---

## Model 3: Hierarchical Binomial with t-Distributed Random Effects (ROBUST)

### 3.1 Full Model Specification

**Likelihood (on logit scale):**
```
r_i | θ_i, n_i ~ Binomial(n_i, logit^(-1)(θ_i))     for i = 1,...,12
θ_i | μ, τ, ν ~ StudentT(ν, μ, τ²)                   (heavy-tailed random effects)
```

**Hyperpriors:**
```
μ ~ Normal(logit(0.075), 1²)                         population mean log-odds
τ ~ HalfNormal(1)                                     between-group SD (log-odds scale)
ν ~ Gamma(2, 0.1)                                     degrees of freedom (controls tail weight)
```

**Derived quantities:**
```
p_i = logit^(-1)(θ_i)                                group-level probabilities
p_mean = logit^(-1)(μ)                                population mean probability
σ²_logit = τ² * ν/(ν-2)  for ν > 2                  between-group variance (log-odds)
```

**Constraint**: ν constrained to ν > 2 for finite variance (alternative: ν > 4 for finite kurtosis).

### 3.2 Theoretical Justification

**Why Heavy-Tailed Random Effects?**
- **Robustness**: Student-t distribution has heavier tails than Gaussian
- **Outlier accommodation**: Extreme groups (2, 8, 11) less influential on hyperparameters
- **Adaptive shrinkage**: Outliers shrink less, typical groups shrink more
- **Flexibility**: ν parameter controls tail weight
  - ν = ∞: reduces to Gaussian (Model 2)
  - ν = 5-10: moderately heavy tails
  - ν = 2-4: very heavy tails
- **Theoretical**: Student-t is a scale mixture of Gaussians, robust to outliers

**Mathematical Foundation**:
Student-t distribution can be written as:
- θ_i | μ, τ, ν, λ_i ~ Normal(μ, τ²/λ_i)
- λ_i ~ Gamma(ν/2, ν/2)

This means each group has its own variance: τ²/λ_i. Outlier groups get larger variance (smaller λ_i), reducing their shrinkage toward μ.

**When to Use This Model**:
- If Model 2 (Gaussian random effects) shows poor fit to outliers
- If posterior predictive checks reveal systematic underfitting of extreme groups
- If EDA suggests groups come from multiple subpopulations
- **Use as backup, not primary model** (unless Model 2 fails)

### 3.3 How It Addresses Key Challenges

**Overdispersion (φ=3.5-5.1)**:
- Heavy tails provide additional variance beyond Gaussian
- Effective τ varies by group based on outlierness
- Expected to accommodate observed φ naturally
- May fit better than Gaussian if overdispersion driven by outliers

**Group Heterogeneity (ICC=0.66)**:
- Student-t allows groups to differ substantially
- Variance: Var(θ) = τ² * ν/(ν-2) for ν > 2
- If ν ≈ 5: variance is 1.67 × τ² (67% more than Gaussian)
- Heavy tails allow extreme deviations without affecting μ, τ estimates

**Zero-Event Group (Group 1: 0/47)**:
- Treated similarly to Model 2
- Hierarchical prior prevents θ_1 → -∞
- Student-t provides slightly heavier tail, allowing more extreme θ_1 if needed
- Expected posterior similar to Model 2

**Outliers (Groups 2, 8, 11)** [KEY ADVANTAGE]:
- **This is where Model 3 shines**
- Heavy tails naturally accommodate extreme groups
- Groups 2, 8, 11 will have larger individual variances (smaller λ_i)
- Less shrinkage toward mean compared to Gaussian
- Typical groups (not outliers) unaffected, still shrink normally
- μ and τ estimates less influenced by extreme groups

**Adaptive Shrinkage Mechanism**:
- Typical group: λ_i ≈ 1 → shrinkage similar to Gaussian
- Outlier group: λ_i < 1 → less shrinkage, larger individual variance
- Automatically detects which groups are outliers via λ_i posterior

### 3.4 Prior Specification Strategy

**Prior Choice Philosophy**: Weakly informative, allowing heavy tails while preventing pathological values.

**Hyperpriors on μ, τ**: Same as Model 2
- μ ~ Normal(logit(0.075), 1²)
- τ ~ HalfNormal(1)
- Justification: Same reasoning as Model 2

**Hyperprior on ν: Gamma(2, 0.1)** [KEY PARAMETER]
- **Shape = 2, Rate = 0.1**: E[ν] = 20
- **Prior mass**:
  - P(ν < 4) ≈ 0.05 (very heavy tails, rare)
  - P(4 < ν < 30) ≈ 0.70 (moderate to light tails)
  - P(ν > 30) ≈ 0.25 (nearly Gaussian)
- **Justification**:
  - Allows data to determine tail weight
  - Weakly informative: doesn't impose strong belief about ν
  - Prior centered at ν = 20 (slightly heavy tails) but allows ν = 5 or ν = 50
- **Constraint**: Constrain ν > 2 in code to ensure finite variance

**Alternative Priors for ν**:
- ν ~ Exponential(1/10) [E[ν] = 10, more heavy-tailed]
- ν ~ Gamma(2, 0.5) [E[ν] = 4, very heavy tails]
- Use if expect strong outliers or if Model 2 badly fails

**Prior Predictive Implications**:
- Student-t with ν = 20 is similar to Gaussian but with slightly heavier tails
- Will generate occasional extreme groups (outliers) more readily than Gaussian
- Should cover observed range including Group 8 (14.4%)

**Key Insight**: If posterior ν >> 30, data suggest Gaussian random effects adequate (Model 2 sufficient). If posterior ν < 10, heavy tails are needed (Model 3 justified).

### 3.5 PyMC Implementation Plan

```python
import pymc as pm
import numpy as np
from scipy.special import logit, expit

# Data
groups = np.arange(1, 13)
n = np.array([47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360])
r = np.array([0, 18, 8, 46, 8, 13, 9, 31, 14, 8, 29, 24])
n_groups = len(groups)

# Model 3: Random Effects Logistic with Student-t
with pm.Model() as model_logistic_t:
    # Hyperpriors
    mu_logit = pm.Normal("mu_logit", mu=logit(0.075), sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Degrees of freedom (constrained > 2 for finite variance)
    nu_raw = pm.Gamma("nu_raw", alpha=2, beta=0.1)
    nu = pm.Deterministic("nu", nu_raw + 2)  # ensures nu > 2

    # Group-level random effects (Student-t distribution)
    # Note: PyMC StudentT uses nu, mu, sigma parameterization
    theta = pm.StudentT("theta", nu=nu, mu=mu_logit, sigma=tau, shape=n_groups)

    # Transform to probability scale
    p = pm.Deterministic("p", pm.math.invlogit(theta))

    # Likelihood
    obs = pm.Binomial("obs", n=n, p=p, observed=r)

    # Derived quantities
    p_mean = pm.Deterministic("p_mean", pm.math.invlogit(mu_logit))

    # Variance (only defined for nu > 2)
    sigma_sq_logit = pm.Deterministic(
        "sigma_sq_logit",
        tau**2 * nu / (nu - 2)
    )

    # Outlier detection: estimate scale parameters lambda_i
    # (for interpretation, not essential for model)
    # lambda_i posterior can be computed post-hoc from theta

    # Posterior predictive
    pp_check = pm.Binomial("pp_check", n=n, p=p, shape=n_groups)


# Alternative: Non-centered parameterization for Student-t
with pm.Model() as model_logistic_t_nc:
    # Hyperpriors
    mu_logit = pm.Normal("mu_logit", mu=logit(0.075), sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=1.0)
    nu_raw = pm.Gamma("nu_raw", alpha=2, beta=0.1)
    nu = pm.Deterministic("nu", nu_raw + 2)

    # Non-centered: theta_i = mu + tau * z_i, where z_i ~ StudentT(nu, 0, 1)
    theta_raw = pm.StudentT("theta_raw", nu=nu, mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic("theta", mu_logit + tau * theta_raw)

    # Rest same as above
    p = pm.Deterministic("p", pm.math.invlogit(theta))
    obs = pm.Binomial("obs", n=n, p=p, observed=r)
    p_mean = pm.Deterministic("p_mean", pm.math.invlogit(mu_logit))
    sigma_sq_logit = pm.Deterministic("sigma_sq_logit", tau**2 * nu / (nu - 2))
    pp_check = pm.Binomial("pp_check", n=n, p=p, shape=n_groups)


# Sampling strategy
with model_logistic_t_nc:
    # Prior predictive check
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)

    # MCMC sampling
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.95,  # higher for Student-t (more complex geometry)
        random_seed=42,
        return_inferencedata=True
    )

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)
```

**Implementation Notes**:
- Student-t random effects can be tricky to sample
- Non-centered parameterization highly recommended
- ν parameter may have slow mixing (only 12 groups to inform it)
- May need higher target_accept (0.95) to avoid divergences
- Computation time: ~3-10 minutes (slower than Models 1-2)

### 3.6 Falsification Criteria (I will abandon this model if...)

**Statistical Red Flags**:

1. **ν Posterior Strongly Favors Gaussian**:
   - If posterior P(ν > 30) > 0.95
   - Data don't support heavy tails
   - Model 2 (Gaussian) is simpler and adequate
   - Action: Use Model 2 as final model

2. **ν Posterior Hits Constraint**:
   - If posterior mass concentrated at ν ≈ 2 (constraint boundary)
   - Suggests infinite-variance distribution (Cauchy-like)
   - Data are even MORE extreme than Student-t can handle
   - Action: Consider mixture model or investigate data quality

3. **Poor Fit Despite Heavy Tails**:
   - If outliers still outside 95% posterior predictive intervals
   - Student-t tails not heavy enough
   - Suggests discrete subpopulations, not continuous variation
   - Action: Use mixture model (two components)

4. **Computational Failures**:
   - Persistent divergences (>2%) despite high target_accept
   - Rhat > 1.05 for ν or θ_i
   - Student-t geometry difficult for MCMC
   - Action: Simplify to Model 2, or use different software (Stan)

5. **Implausible Shrinkage**:
   - If typical groups shrink too little (minimal pooling benefit)
   - Heavy tails causing underfitting of typical groups
   - Loss of partial pooling advantages
   - Action: Use Model 2 with outlier sensitivity analysis instead

6. **Overdispersion Mismatch**:
   - If posterior φ still doesn't match observed (3.5-5.1)
   - Suggests fundamental model misspecification beyond tail weight
   - Action: Reconsider likelihood structure (beta-binomial?)

**Decision Rules**:
- **USE MODEL 2 INSTEAD if**: Criterion 1 occurs (ν > 30)
- **REJECT if**: Criteria 2, 4, 5, or 6 occur
- **SWITCH TO MIXTURE MODEL if**: Criterion 3 occurs

### 3.7 Expected Computational Challenges

**Likely Issues**:

1. **ν Parameter Difficult to Estimate**:
   - Only 12 groups, weak information about tail weight
   - ν posterior may be wide, skewed, or multimodal
   - **Mitigation**:
     - Use informative prior if strong belief about outliers
     - Accept wide credible intervals
     - Focus on group-specific p_i, not ν itself
     - Posterior ν is useful for model selection (ν > 30 → use Model 2)

2. **Sampling Challenges**:
   - Student-t has more complex geometry than Gaussian
   - Tails can cause sampler to get stuck
   - Non-centered helps but doesn't eliminate issues
   - **Mitigation**:
     - Increase target_accept to 0.95 or 0.99
     - May need more tuning steps (3000-4000)
     - Check trace plots carefully for exploration issues

3. **Divergences**:
   - More likely than Models 1-2 due to complex geometry
   - May concentrate near extreme θ_i values or low ν
   - **Mitigation**:
     - Non-centered parameterization reduces divergences
     - Increase target_accept incrementally
     - If persistent (>2%): may indicate model inappropriate

4. **Computation Time**:
   - Slower than Models 1-2 due to Student-t evaluations
   - Expected: ~5-10 minutes for 2000 draws × 4 chains
   - **Not prohibitive but noticeably slower**

5. **Convergence**:
   - May need more samples for adequate ESS (especially for ν)
   - θ_i typically converge well, but ν may lag
   - **Mitigation**:
     - Run longer if needed (3000-4000 draws)
     - Focus on θ_i and p_i which are usually well-estimated
     - Wide ν posterior is acceptable if predictions good

**When to Use This Model**:
- **NOT as primary model** — start with Models 1 or 2
- **Use if**: Model 2 shows systematic poor fit to outliers (Groups 2, 8, 11)
- **Use if**: Posterior predictive checks reveal outlier accommodation issues
- **Don't use if**: Model 2 fits well (ν > 30 posterior suggests unnecessary)

**Success Metrics**:
- Rhat < 1.01 for θ_i (may allow Rhat < 1.05 for ν if wide posterior)
- ESS > 400 per chain for θ_i
- ESS > 200 per chain for ν (acceptable if lower)
- Divergences < 2%
- Visual checks: ν trace should explore range, not stuck

---

## Comparative Summary

### Model Comparison Table

| Feature | Model 1: Beta-Binomial | Model 2: Logistic GLMM | Model 3: Robust Logistic |
|---------|------------------------|------------------------|---------------------------|
| **Likelihood** | Binomial | Binomial | Binomial |
| **Between-group variation** | Beta(α, β) | Normal(μ, τ²) | StudentT(ν, μ, τ²) |
| **Parameterization** | Probability scale | Log-odds scale | Log-odds scale |
| **Overdispersion handling** | Direct (Beta) | Indirect (random effects) | Indirect (heavy-tailed RE) |
| **Outlier robustness** | Moderate | Moderate | High |
| **Zero-event handling** | Excellent | Good | Good |
| **Interpretability** | High (probabilities) | High (log-odds) | Moderate (ν less intuitive) |
| **Computational cost** | Low (~2-5 min) | Low (~2-5 min) | Moderate (~5-10 min) |
| **Sampling difficulty** | Easy | Very easy (non-centered) | Moderate |
| **Extensibility** | Limited | High (add covariates) | High (add covariates) |
| **When to use** | Primary choice | Alternative primary | Backup if Model 2 fails |

### Strengths and Weaknesses

**Model 1: Beta-Binomial Hierarchical**

Strengths:
- Direct modeling of overdispersion (most natural for overdispersed binomial)
- Conjugate structure (mathematical elegance)
- Excellent handling of zero-events (Beta prior)
- Parameters directly interpretable (α, β → mean and variance)
- Fast computation, stable sampling
- Theoretically grounded (canonical model for this data type)

Weaknesses:
- Limited extensibility (hard to add covariates)
- May not accommodate extreme outliers as well as Student-t
- Concentration parameter κ can be difficult to estimate with few groups
- Less familiar to some applied researchers than logistic regression
- Assumes all variation explained by Beta distribution (may be too restrictive)

**Model 2: Random Effects Logistic Regression**

Strengths:
- Standard, widely understood approach
- Easy to extend with group-level covariates
- Familiar log-odds parameterization
- Excellent computational properties (non-centered)
- Well-established in applied research
- Easy to interpret random effects variance τ²

Weaknesses:
- Gaussian random effects may not accommodate extreme outliers well
- Overdispersion is indirect consequence of random effects, not explicit
- May underfit high-rate groups (Groups 2, 8, 11) if tails too light
- Requires careful parameterization (non-centered essential)
- Overdispersion factor φ depends on position on probability scale

**Model 3: Robust Logistic with Student-t**

Strengths:
- Robust to outliers (heavy tails)
- Adaptive shrinkage (outliers shrink less)
- Automatically detects outlier groups via λ_i
- Includes Model 2 as special case (ν → ∞)
- Excellent for datasets with heterogeneous subgroups

Weaknesses:
- More complex, harder to sample
- ν parameter difficult to estimate with 12 groups
- Slower computation (~2× Model 2)
- May be unnecessary if Model 2 fits well
- Less familiar to applied researchers
- Can overfit if ν poorly constrained

### Recommended Priority Order

**1. Model 1: Beta-Binomial Hierarchical (PRIMARY)**
- Best theoretical fit for overdispersed binomial data
- Direct modeling of key data features
- Fast, stable, interpretable
- **Implement first, use as baseline**

**2. Model 2: Random Effects Logistic Regression (CLOSE ALTERNATIVE)**
- Standard approach, well-understood
- Excellent computational properties
- Easy to extend if needed
- **Implement second for comparison**
- If Model 1 and Model 2 give similar results: report both, slight preference for Model 1

**3. Model 3: Robust Logistic (CONDITIONAL)**
- **Only implement if**:
  - Model 2 shows poor fit to outliers (Groups 2, 8, 11)
  - Posterior predictive checks reveal systematic outlier accommodation failure
  - Posterior for Model 2 suggests heavy tails needed
- Not needed if Models 1-2 fit well

### Model Selection Strategy

**Phase 1**: Fit Models 1 and 2 in parallel
- Both should converge successfully
- Compare posterior predictive checks
- Compare LOO-CV if both adequate

**Phase 2**: Assess outlier fit
- If both models fit outliers well: **choose Model 1** (more natural for data type)
- If Model 2 underfits outliers but Model 1 okay: **choose Model 1**
- If both underfit outliers: **fit Model 3**

**Phase 3**: Final selection
- Primary criterion: Posterior predictive fit (qualitative and quantitative)
- Secondary: LOO-CV (predictive accuracy)
- Tertiary: Interpretability and extensibility for downstream use

**Expected outcome**: Model 1 or Model 2 will be adequate. Both should give similar substantive conclusions. Model 3 likely unnecessary unless outliers very extreme.

---

## Prior Predictive Checks (Essential Before Fitting)

For each model, generate prior predictive datasets and verify:

### Checks to Perform

1. **Range Check**:
   - Do simulated r_i cover observed range (0 to 46)?
   - Do simulated proportions cover 0% to 15%?
   - If not: priors may be too informative

2. **Zero-Event Check**:
   - Are zero-event groups (0/n) generated occasionally?
   - Expected frequency for n=47, p≈0.07: ~3% of simulations
   - If never: prior may be too informative against zeros

3. **Outlier Check**:
   - Are extreme groups (>12%) generated occasionally?
   - Should see some simulations with 2-3 "outlier" groups
   - If never: model may not accommodate observed outliers

4. **Overdispersion Check**:
   - Compute φ for each simulated dataset
   - Does distribution of φ include observed φ=3.5-5.1?
   - If always φ < 2: prior implies too little overdispersion

5. **Heterogeneity Check**:
   - Compute ICC for each simulated dataset
   - Does distribution of ICC include observed ICC=0.66?
   - Should see wide range but cover 0.5-0.8

### Prior Adjustment Criteria

- **Priors too vague**: If >10% simulations have implausible values (e.g., rates >50%)
- **Priors too tight**: If <5% simulations cover observed range
- **Priors just right**: Wide range including observed data but excluding pathological cases

**Action**: Adjust priors if checks fail, re-run prior predictive, iterate until reasonable.

---

## Posterior Predictive Checks (Essential After Fitting)

For each fitted model, assess:

### Quantitative Checks

1. **Coverage**:
   - % of observed r_i within 95% posterior predictive interval
   - Expected: ~95%, acceptable: 80-100%
   - If <75%: model systematically underfits

2. **Outlier Fit**:
   - Are Groups 2, 8, 11 within 95% intervals?
   - If not: model cannot accommodate outliers (try Model 3)

3. **Zero-Event Fit**:
   - Is Group 1 (0/47) within 95% interval?
   - Posterior predictive P(r_1 = 0) should be ~2-5%
   - If posterior P(r_1 = 0) < 0.1%: model thinks zeros too rare

4. **Dispersion Check**:
   - Posterior predictive distribution of φ
   - Does posterior φ match observed φ=3.5-5.1?
   - If systematically lower: model underdispersed

### Qualitative Checks

5. **PPC Plots**:
   - Overlay posterior predictive draws on observed data
   - Visual check: do simulations "look like" observed data?
   - Check for systematic patterns (e.g., always underpredict high groups)

6. **Residual Analysis**:
   - Compute Bayesian residuals: (r_i - E[r_i|data]) / SD[r_i|data]
   - Should be roughly Normal(0, 1)
   - Check for patterns vs. n_i, group ID, fitted values

### Falsification Triggers

- **REJECT** if: <70% coverage, outliers systematically misfit, φ far from observed
- **REVISE** if: 70-85% coverage, some systematic patterns
- **ACCEPT** if: >85% coverage, no systematic patterns, φ matches observed

---

## Implementation Workflow

### Step-by-Step Plan

**1. Prior Predictive Checks** (before fitting):
- Generate 1000 prior predictive datasets for each model
- Verify range, overdispersion, outlier generation reasonable
- Adjust priors if necessary, iterate

**2. Model Fitting**:
- Fit Models 1 and 2 in parallel
- Check convergence diagnostics (Rhat, ESS, divergences)
- Troubleshoot if sampling issues arise

**3. Posterior Predictive Checks**:
- Generate 1000 posterior predictive datasets
- Compute coverage, dispersion, residuals
- Visual checks (PPC plots)

**4. Model Critique**:
- Apply falsification criteria for each model
- Decision: ACCEPT / REVISE / REJECT
- If both models adequate: proceed to comparison

**5. Model Comparison** (if both accepted):
- LOO cross-validation (elpd_loo)
- Compare predictions for specific groups
- Assess interpretability and extensibility

**6. Sensitivity Analysis**:
- Fit with alternative priors
- Assess influence of outlier groups (exclude, check posteriors)
- Assess influence of Group 4 (high leverage)

**7. Final Model Selection**:
- Choose best model based on:
  - Posterior predictive fit (primary)
  - LOO-CV (secondary)
  - Interpretability (tertiary)
- Document decision and rationale

**8. Model 3 (Conditional)**:
- Only if Models 1-2 fail outlier accommodation
- Fit Model 3, assess ν posterior
- If ν > 30: revert to Model 2
- If ν < 30: use Model 3

---

## Expected Outcomes and Decisions

### Scenario 1: Models 1 and 2 Both Adequate (Most Likely)
- Both pass posterior predictive checks
- Similar LOO-CV scores (within 2 SE)
- Similar posteriors for p_i
- **Decision**: Report both, slight preference for **Model 1** (more natural for binomial overdispersion)
- **Reasoning**: Beta-binomial directly models overdispersion; logistic GLMM is indirect

### Scenario 2: Model 1 Better Than Model 2
- Model 1 better outlier fit or lower LOO-CV
- Model 2 shows systematic underfit
- **Decision**: Use **Model 1** as final model
- **Reasoning**: Data favor direct overdispersion modeling

### Scenario 3: Model 2 Better Than Model 1
- Model 2 better predictive performance
- Model 1 shows poor fit (rare but possible)
- **Decision**: Use **Model 2** as final model
- **Reasoning**: Data favor logistic random effects structure

### Scenario 4: Both Models Underfit Outliers
- Groups 2, 8, 11 consistently outside 95% intervals
- Coverage <75%
- **Decision**: Fit **Model 3** with heavy-tailed random effects
- **Reasoning**: Gaussian/Beta tails too light for observed heterogeneity

### Scenario 5: All Models Fail
- No model passes posterior predictive checks
- Systematic lack of fit
- **Decision**: Consider mixture model, investigate data quality, consult domain experts
- **Reasoning**: Data may have structure not captured by single-component hierarchical models

### Most Likely Outcome
**Model 1 (Beta-Binomial) will be adequate and selected as final model.**
- Direct modeling of overdispersion
- Zero-event and outlier handling should work well
- Fast, stable, interpretable

**Model 2 will provide useful comparison and similar conclusions.**

**Model 3 likely unnecessary unless outliers very extreme.**

---

## Summary

I propose three robust, well-established Bayesian models for this overdispersed binomial data:

1. **Beta-Binomial Hierarchical** (PRIMARY): Direct overdispersion modeling, conjugate structure, excellent zero-event handling
2. **Logistic GLMM** (ALTERNATIVE): Standard approach, log-odds scale, excellent computational properties
3. **Robust Logistic** (BACKUP): Heavy-tailed random effects for outlier accommodation if needed

All models use partial pooling, weakly informative priors, and PyMC for inference. Priority order: Model 1 > Model 2 > Model 3 (conditional).

**Implementation plan**: Fit Models 1-2 first, assess with posterior predictive checks, compare if both adequate, fit Model 3 only if needed.

**Expected outcome**: Model 1 or Model 2 will be adequate, provide similar substantive conclusions, and handle the zero-event group and outliers appropriately through hierarchical shrinkage.

**Key design principles**: Falsification criteria specified, computational challenges anticipated, sensitivity analyses planned, clear decision rules for model selection.

Ready for implementation.
