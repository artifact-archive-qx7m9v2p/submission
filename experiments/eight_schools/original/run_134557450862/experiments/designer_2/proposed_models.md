# Bayesian Model Design: Pooling Strategy Focus
## Designer 2 - Independent Analysis

**Date:** 2025-10-28
**Focus Area:** Pooling strategies and model parsimony
**Dataset:** Eight Schools meta-analysis (n=8)

---

## Executive Summary

The EDA provides exceptionally strong evidence for complete homogeneity (Q p=0.696, I²=0%, tau²=0). However, my role as a modeler is to **question whether this apparent homogeneity is real or an artifact of low statistical power (n=8)**.

I propose three Bayesian model classes representing the full pooling spectrum, each with explicit falsification criteria. The goal is not to validate the EDA conclusions, but to **stress-test them** and understand what assumptions each pooling strategy makes.

**Key insight:** With only 8 observations and large measurement errors (sigma 9-18), we have limited power to detect moderate heterogeneity. Complete pooling is supported by the data, but we must rigorously test whether this is truth or power limitation.

---

## Model Class 1: Complete Pooling (Fixed Effect Model)

### Mathematical Specification

```
# Likelihood
y_i ~ Normal(mu, sigma_i)    for i = 1,...,8

# Prior
mu ~ Normal(0, 25)
```

**Parameter count:** 1 (mu only)

### Assumptions and Rationale

**Core assumption:** All schools share EXACTLY the same true treatment effect. The 31-point range in observed effects (from -3 to 28) is entirely due to measurement error.

**Why this might be true:**
- All schools received the same intervention protocol
- Schools are from similar educational contexts
- EDA shows variance ratio < 1 (observed variance less than expected under noise)
- No school is a statistical outlier (all within ±2 SD)

**Prior justification:**
- `Normal(0, 25)`: Weakly informative, allows effects from -50 to +50 (2 SD)
- Centered at zero (neutral expectation about intervention)
- SD=25 is generous given observed range [-3, 28]
- Data dominate: pooled SE=4.07, prior SE=25, so likelihood weight ~36x prior weight

### Expected Posterior Behavior

**If model is correct:**
- Posterior: mu ~ Normal(7.7, 4.1) (matching weighted mean from EDA)
- Posterior predictive: new schools should scatter around 7.7 with SD ~ sqrt(4.1² + 12.5²) ≈ 13.2
- All 8 schools within 95% posterior predictive intervals
- Effective sample size: n_eff ~ 8 (full information from all schools)

**If model is wrong:**
- Posterior predictive checks will show systematic misfit
- Residual patterns (e.g., low-precision schools consistently over/underestimated)
- Coverage: observed effects outside 95% credible intervals

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior predictive p-value < 0.05** for test statistic: max(|y_i - mu|/sigma_i)
   - Current max z-score is 1.35 (School 1)
   - If model is correct, seeing z > 1.35 should happen ~35% of the time
   - If actual data is extreme (p < 0.05), evidence against complete pooling

2. **More than 2 schools fall outside 95% posterior predictive intervals**
   - Expected: 0.4 schools (8 × 0.05) should fall outside
   - 1-2 schools outside: sampling variation
   - 3+ schools outside: systematic underfitting

3. **Leave-one-out cross-validation shows poor calibration**
   - LOO-PIT (probability integral transform) should be uniform
   - If LOO-PIT shows U-shape: underdispersed (model too confident)
   - If LOO-PIT shows inverse-U: overdispersed (but unlikely here)

4. **Systematic residual patterns by precision**
   - Plot (y_i - E[mu]) vs 1/sigma_i
   - If high-precision schools consistently deviate in same direction: heterogeneity signal
   - Current data: Schools 5 (low) and 7 (high) are high-precision with opposite deviations - good sign

5. **Prior-posterior conflict detected**
   - If posterior pushes against prior boundaries or shows multimodality
   - Not expected here (prior is weak), but would indicate model misspecification

### Computational Implementation (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as complete_pooling:
    # Prior
    mu = pm.Normal('mu', mu=0, sigma=25)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Posterior sampling
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

    # Posterior predictive checks
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    # LOO cross-validation
    loo = pm.loo(trace)
```

**Expected runtime:** < 30 seconds (trivial model)

### Model Strengths

1. **Maximum parsimony:** Occam's razor - simplest explanation
2. **Strong data support:** EDA heterogeneity tests all non-significant
3. **Maximum precision:** Pools all information, narrowest credible intervals
4. **Easy interpretation:** Single effect estimate for policy decisions
5. **Robust to outliers:** Precision weighting downweights noisy observations

### Model Weaknesses

1. **No uncertainty about heterogeneity:** Assumes tau=0 with certainty
2. **Ignores exchangeability structure:** Schools as independent observations, not sample from population
3. **Cannot answer "how much do schools vary?"** - assumes no variation
4. **May underestimate uncertainty:** If true tau > 0 but small, intervals too narrow
5. **Philosophically extreme:** Requires belief that ALL schools identical

### Decision Rule

**Use complete pooling as primary model IF:**
- All 5 falsification criteria pass
- AND posterior predictive p-values > 0.10 for multiple test statistics
- AND no evidence from other model classes (hierarchical, no pooling) suggests tau > 5

**Abandon IF:**
- Any falsification criterion fails with strong evidence (p < 0.01)
- OR hierarchical model shows tau posterior concentrated above 5

---

## Model Class 2: No Pooling (Independent Effects Model)

### Mathematical Specification

```
# Likelihood
y_i ~ Normal(theta_i, sigma_i)    for i = 1,...,8

# Priors (independent)
theta_i ~ Normal(0, 25)          for i = 1,...,8
```

**Parameter count:** 8 (theta_1, ..., theta_8)

### Assumptions and Rationale

**Core assumption:** Each school's true effect is COMPLETELY INDEPENDENT. No information sharing. Knowing School 1's effect tells us nothing about School 2.

**Why this might be true:**
- Schools implemented intervention differently (fidelity variation)
- Student populations fundamentally different (demographics, baseline achievement)
- Contextual factors (teachers, resources, culture) overwhelm intervention effect
- Measurement error is smaller than we think (sigma values underestimated)

**Why this is probably WRONG:**
- EDA shows observed variance LESS than expected under pure noise
- If effects were truly independent AND heterogeneous, we'd expect variance ratio > 1
- Variance ratio = 0.66 suggests even sampling error overexplains variation
- Schools labeled "A, B, C..." suggests they're exchangeable (common sampling frame)

**Prior justification:**
- Same `Normal(0, 25)` as complete pooling for comparability
- Each school gets independent prior (no hierarchical structure)
- Data per school: 1 observation with sigma 9-18
- Prior-to-data weight: roughly equal (weak data, weak prior)

### Expected Posterior Behavior

**If model is correct:**
- Posterior: theta_i ~ Normal(y_i, sigma_i) (data slightly regularized by prior)
- School 1: theta_1 ~ Normal(26, 14.5) (barely shrunk from observed 28)
- School 5: theta_5 ~ Normal(-0.8, 8.9) (barely shrunk from observed -1)
- Posterior intervals VERY WIDE (no information borrowing)
- Posterior SD for theta_i ≈ 0.96 × sigma_i (minimal prior influence)

**If model is wrong (likely):**
- Intervals unnecessarily wide (ignoring similarity across schools)
- Poor out-of-sample prediction (each school fit in isolation)
- LOO cross-validation penalizes complexity (8 parameters for 8 observations)

### Falsification Criteria

**I will abandon this model if:**

1. **LOO cross-validation shows worse predictive performance than complete pooling**
   - Compare ELPD (expected log pointwise predictive density)
   - If complete pooling ELPD > no pooling ELPD by > 2 (ELPD standard errors), strong evidence against no pooling
   - Expected: complete pooling wins decisively given EDA

2. **Posterior predictive distribution for NEW school (school 9) is excessively wide**
   - No pooling prediction: draw random theta_9 ~ Normal(0, 25), then y_9 ~ Normal(theta_9, 13)
   - Complete pooling prediction: y_9 ~ Normal(7.7, sqrt(4.1² + 13²))
   - If no pooling 95% CI is > 2x wider than complete pooling AND data don't require it: abandon

3. **Shrinkage analysis shows posteriors barely differ from observed values**
   - If theta_i posteriors are just y_i ± sigma_i with minimal regularization
   - Suggests model isn't learning from structure in data
   - **Current expectation:** Posteriors will be ~96% determined by individual y_i

4. **Model comparison (WAIC, LOO) heavily penalizes no pooling**
   - Effective parameter count p_eff ≈ 8 (one per school)
   - Complete pooling: p_eff ≈ 1
   - If penalty for complexity outweighs fit improvement: abandon

5. **Philosophical inconsistency:** Cannot justify why schools are exchangeable for sampling but not for analysis
   - If we believe schools are "similar enough" to study together, we should pool information
   - Exchangeability assumption undermines no pooling justification

### Computational Implementation (PyMC)

```python
with pm.Model() as no_pooling:
    # Priors (independent)
    theta = pm.Normal('theta', mu=0, sigma=25, shape=8)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Posterior sampling
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

    # Posterior predictive
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    # LOO for model comparison
    loo = pm.loo(trace)
```

**Expected runtime:** < 1 minute (8 independent parameters)

### Model Strengths

1. **Maximum flexibility:** No constraints on how effects can vary
2. **No pooling assumptions:** Cannot be biased by false homogeneity
3. **Individual school inference:** Each school gets own parameter
4. **Conservative:** Wide intervals acknowledge uncertainty
5. **Baseline for comparison:** Useful reference point for hierarchical model

### Model Weaknesses

1. **Poor precision:** Doesn't leverage similarity across schools
2. **Overfitting risk:** 8 parameters for 8 observations (perfect fit guaranteed)
3. **Poor prediction:** Cannot generalize to new schools (must draw from flat prior)
4. **Ignores EDA evidence:** Data show homogeneity, model assumes heterogeneity
5. **Inefficient:** Throws away information from similar schools
6. **Contradicts variance decomposition:** EDA variance ratio < 1 incompatible with independent effects

### Decision Rule

**Use no pooling as primary model IF:**
- LOO/WAIC decisively favors it over complete/partial pooling (very unlikely)
- AND posterior predictive checks show complete pooling systematically fails
- AND there's domain knowledge suggesting schools are fundamentally different

**Expected outcome:** No pooling will serve as **lower bound on model performance**. Any reasonable model should beat it given the EDA evidence. If no pooling wins, something is deeply wrong with the data or other models.

**Abandon IF (expected):**
- LOO favors complete or partial pooling (almost certain)
- Posterior intervals uninformatively wide
- Cannot make useful predictions for new schools

---

## Model Class 3: Partial Pooling with Prior Sensitivity (Hierarchical Variants)

### Mathematical Specification (Base Version)

```
# Likelihood
y_i ~ Normal(theta_i, sigma_i)    for i = 1,...,8

# School effects (partial pooling)
theta_i ~ Normal(mu, tau)         for i = 1,...,8

# Hyperpriors
mu ~ Normal(0, 25)
tau ~ Half-Cauchy(0, scale_tau)
```

**Parameter count:** 10 (mu, tau, theta_1, ..., theta_8)

### The Critical Question: What Prior on tau?

This is the **most important modeling decision**. The tau prior controls the degree of pooling:
- tau → 0: collapses to complete pooling
- tau → ∞: collapses to no pooling
- tau ~ 5-10: moderate heterogeneity

**The problem:** With n=8 and large sigma_i, the data have LIMITED power to inform tau. The prior matters greatly.

I propose testing **THREE tau priors** to assess sensitivity:

#### Variant 3a: Weakly Informative (Gelman Recommendation)

```
tau ~ Half-Cauchy(0, 5)
```

**Rationale:**
- Gelman (2006) recommendation for hierarchical models
- Median ≈ 3.5, but heavy right tail allows tau up to 50+
- Designed to be "weakly informative" - doesn't force tau near 0
- Scale=5 chosen based on typical effect size scale (~10 units)

**Expected posterior:**
- Given EDA (tau² = 0), expect posterior concentrated near 0
- But Cauchy tail allows learning if data support larger tau
- Likely posterior: tau ~ Half-Cauchy truncated below 10, mode near 0-2

**Shrinkage factor:** B_i = tau² / (tau² + sigma_i²)
- If tau = 0: B_i = 0 (full shrinkage to mu, complete pooling)
- If tau = 5, sigma = 12: B_i = 0.15 (85% shrinkage)
- If tau = 15, sigma = 12: B_i = 0.61 (39% shrinkage)

#### Variant 3b: Skeptical Prior (Conservative About Heterogeneity)

```
tau ~ Half-Normal(0, 3)
```

**Rationale:**
- More skeptical about heterogeneity than Half-Cauchy
- Half-Normal concentrates mass near 0 (95% of mass below 6)
- Appropriate if we believe EDA homogeneity findings
- Less heavy-tailed: won't allow extreme tau unless data strongly support it

**Expected posterior:**
- Even more concentrated near 0 than Half-Cauchy
- Posterior mode likely tau < 1
- Shrinkage: 90-95% toward mu for most schools

**Interpretation:** "I believe schools are similar until proven otherwise"

#### Variant 3c: Permissive Prior (Generous About Heterogeneity)

```
tau ~ Uniform(0, 50)
```

**Rationale:**
- Maximally uninformative on [0, 50] range
- Doesn't favor any value of tau (flat prior)
- Let data speak without prior influence
- Tests whether data alone can identify tau

**Expected posterior:**
- Data will push toward low tau (given EDA)
- But posterior will have heavier right tail than Half-Cauchy
- If posterior is flat/diffuse: data cannot identify tau (fundamental problem)

**Warning:** Uniform priors can be problematic for hierarchical models (improper posteriors if k small). Need to verify posterior is proper.

### Falsification Criteria (All Variants)

**I will abandon hierarchical approach if:**

1. **Posterior for tau is completely uninformative (reflects prior)**
   - Plot prior vs posterior for tau
   - If KL divergence < 0.1 nats: data added negligible information
   - Interpretation: n=8 insufficient to learn about tau, pooling decision is prior-driven

2. **Posterior mode for tau at boundary (tau ≈ 0) with tight concentration**
   - If posterior: tau | y ~ Exponential-like decay from 0
   - AND 95% CI for tau is [0, 3]
   - **Interpretation:** Model reduces to complete pooling, use simpler Model 1

3. **Strong prior sensitivity: posteriors differ substantially across Variants 3a/b/c**
   - If tau posterior medians differ by > 5 across priors
   - AND predictions for new school differ by > 10 units (effect scale)
   - **Interpretation:** Data cannot resolve pooling degree, answer is prior-dependent

4. **Computational pathologies:**
   - Divergent transitions (> 1% of samples)
   - R-hat > 1.01 for tau
   - Effective sample size < 100 for tau
   - **Interpretation:** Model geometry problematic, often indicates misspecification or tau at boundary

5. **Hierarchical model doesn't outperform simpler models in LOO**
   - If LOO(complete pooling) ≈ LOO(hierarchical) within 2 SE
   - AND effective parameter count p_eff ≈ 1 (not using flexibility)
   - **Interpretation:** Paying complexity cost for no predictive gain, use complete pooling

### Expected Posterior Behavior

**Given EDA evidence for homogeneity:**

**Variant 3a (Half-Cauchy):**
- mu: Normal(7.7, 4.2)
- tau: mode ≈ 0-2, median ≈ 3, 95% CI [0, 8]
- theta_i: strongly shrunk toward mu (60-90% shrinkage)
- Effective parameter count: p_eff ≈ 2-3 (between 1 and 8)

**Variant 3b (Half-Normal):**
- mu: Normal(7.7, 4.1)
- tau: mode ≈ 0, median ≈ 1, 95% CI [0, 4]
- theta_i: very strongly shrunk toward mu (80-95% shrinkage)
- Effective parameter count: p_eff ≈ 1.5 (close to complete pooling)

**Variant 3c (Uniform):**
- mu: Normal(7.7, 4.3)
- tau: mode ≈ 0, but long tail, 95% CI [0, 15]
- theta_i: moderate to strong shrinkage (40-80% shrinkage)
- Effective parameter count: p_eff ≈ 3-4
- **Risk:** Posterior may be poorly identified if data too weak

### Computational Implementation (PyMC)

```python
# Variant 3a: Half-Cauchy (Gelman)
with pm.Model() as partial_pooling_cauchy:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sigma=25)
    tau = pm.HalfCauchy('tau', beta=5)

    # Non-centered parameterization (for better sampling)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=8)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Sampling
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, random_seed=42)

    # Diagnostics
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    loo = pm.loo(trace)

# Variant 3b: Half-Normal
with pm.Model() as partial_pooling_normal:
    mu = pm.Normal('mu', mu=0, sigma=25)
    tau = pm.HalfNormal('tau', sigma=3)

    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=8)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, random_seed=42)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    loo = pm.loo(trace)

# Variant 3c: Uniform
with pm.Model() as partial_pooling_uniform:
    mu = pm.Normal('mu', mu=0, sigma=25)
    tau = pm.Uniform('tau', lower=0, upper=50)

    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=8)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, random_seed=42)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    loo = pm.loo(trace)
```

**Key implementation notes:**
- **Non-centered parameterization** essential for tau near 0 (avoids funnel geometry)
- `target_accept=0.95` for better exploration of tau near boundary
- Expect 2-5 minutes per variant

### Model Strengths

1. **Philosophically appropriate:** Schools are exchangeable but not identical
2. **Automatic adaptation:** Data determine degree of pooling via tau posterior
3. **Handles boundary gracefully:** Bayesian approach allows tau ≈ 0 without issues
4. **Conservative:** Accounts for uncertainty about heterogeneity
5. **Rich inference:** Can estimate both common effect (mu) and variation (tau)
6. **Generalizable:** Predictions for new schools use learned tau

### Model Weaknesses

1. **Complex:** 10 parameters for 8 observations (potential overfitting)
2. **Weak identification:** With n=8, tau posterior may be prior-dominated
3. **Computational:** Non-centered parameterization, boundary issues, longer runtime
4. **Interpretation:** Must explain hierarchical structure to non-statisticians
5. **May reduce to complete pooling:** If tau ≈ 0, added complexity for no gain

### Decision Rule

**Use hierarchical model as primary IF:**
- Posterior for tau shows learning from data (prior vs posterior differ substantively)
- AND tau posterior is similar across reasonable priors (Variants 3a/b agree on tau < 5)
- AND LOO doesn't decisively favor complete pooling (within 3 ELPD)
- AND computational diagnostics are clean (R-hat < 1.01, no divergences)

**Abandon hierarchical model IF:**
- tau posterior is prior-driven (KL divergence < 0.1)
- OR tau concentrates at 0 with 95% CI [0, 2] AND complete pooling has same LOO
- OR strong prior sensitivity (variants give contradictory conclusions)
- OR computational issues persist despite reparameterization

**Expected outcome:** Hierarchical model will likely show tau ≈ 0-3, effectively reducing to complete pooling. This is NOT a failure - it's Bayesian learning confirming the EDA. The value is in QUANTIFYING uncertainty about tau.

---

## Comparative Analysis: Expected Model Rankings

### Predictive Performance (LOO-CV)

**Expected ranking (best to worst):**

1. **Complete Pooling (Model 1)** - ELPD ≈ -33
   - Simple, data strongly support it
   - Minimal effective parameters (p_eff ≈ 1)
   - Should win by 2-4 ELPD over hierarchical

2. **Partial Pooling - Variant 3b (Half-Normal)** - ELPD ≈ -34
   - Will effectively reduce to complete pooling (tau ≈ 0)
   - Pays small complexity penalty (p_eff ≈ 1.5-2)
   - Slightly worse than complete pooling but within SE

3. **Partial Pooling - Variant 3a (Half-Cauchy)** - ELPD ≈ -35
   - More dispersed tau prior, larger complexity penalty
   - p_eff ≈ 2-3
   - Still close to complete pooling

4. **Partial Pooling - Variant 3c (Uniform)** - ELPD ≈ -36
   - Largest tau prior uncertainty, largest penalty
   - p_eff ≈ 3-4
   - May suffer from poor identification

5. **No Pooling (Model 2)** - ELPD ≈ -40
   - Large complexity penalty (p_eff = 8)
   - Poor predictions (doesn't use similarity)
   - Expected to lose decisively

**Interpretation:** If ranking differs substantially from this, it indicates:
- Heterogeneity present (tau > 0) → hierarchical models rise
- Complete pooling failure → no pooling rises
- Data too weak → all models similar LOO (concerning)

### Posterior Predictive Accuracy

**Test statistic:** For new school 9, predict y_9 and compare to holdout observation

**Expected 95% predictive intervals:**

| Model | Prediction Mean | Prediction SD | 95% Interval Width |
|-------|----------------|---------------|-------------------|
| Complete Pooling | 7.7 | 13.2 | 52 units |
| Partial (3b, Half-Normal) | 7.7 | 13.5 | 53 units |
| Partial (3a, Half-Cauchy) | 7.7 | 14.0 | 55 units |
| Partial (3c, Uniform) | 7.7 | 15.0 | 59 units |
| No Pooling | 0.0 | 28.0 | 110 units |

**Key insight:** Complete pooling and tight hierarchical priors give similar predictions. No pooling prediction reverts to prior (no learning across schools), massively wider interval.

### Computational Efficiency

| Model | Expected Runtime | Sampling Issues | Interpretation |
|-------|-----------------|-----------------|----------------|
| Complete Pooling | < 30 sec | None | Trivial |
| No Pooling | < 1 min | None | Simple |
| Partial (3b, Half-Normal) | 2-3 min | Possible boundary warnings | tau near 0 |
| Partial (3a, Half-Cauchy) | 3-5 min | Possible boundary warnings | Heavy tail |
| Partial (3c, Uniform) | 3-5 min | Possible non-identifiability | Weak prior |

**Decision:** If hierarchical variants take > 10 min or have persistent divergences, consider it a red flag for model misspecification.

---

## Stress Tests and Red Flags

### Stress Test 1: Posterior Predictive Check for Extreme Values

**Test:** Simulate 1000 datasets from posterior predictive distribution. For each:
- Compute max(|y_i - mu|/sigma_i) [maximum standardized residual]
- Compare to observed value (1.35 from School 1)

**Expected outcome:**
- Complete pooling: p-value ≈ 0.30-0.40 (observed max is typical)
- Partial pooling: p-value ≈ 0.35-0.45 (slightly less extreme due to tau)
- No pooling: p-value ≈ 0.50 (everything is typical in no pooling)

**Red flag:** p-value < 0.05 → observed data are more extreme than model predicts → model underestimates heterogeneity

### Stress Test 2: Leave-One-Out Subgroup Analysis

**Test:** Systematically leave out each school, refit models, check if:
1. Posterior for mu changes by > 3 units (more than pooled SE)
2. Posterior for tau changes substantially (e.g., doubles or halves)
3. Predictions for left-out school are systematically biased

**Expected outcome:**
- Complete pooling: mu varies by ±2 units (matching EDA influence analysis)
- Partial pooling: tau remains near 0 for all leave-one-out fits
- No pooling: each school unaffected by others (by design)

**Red flag:**
- If removing School 1 (extreme value) causes tau to drop from 5 to 1 → School 1 driving heterogeneity estimate
- If removing School 5 or 7 (high precision) radically changes mu → overfitting to precise observations

### Stress Test 3: Prior-Posterior Overlap (Information Criterion)

**Test:** For each model, compute KL divergence from prior to posterior:
- KL(posterior || prior) measures how much data changed beliefs
- Large KL: data informative
- Small KL: posterior ≈ prior (data not informative)

**Expected KL divergences:**

| Parameter | Model | Expected KL (nats) | Interpretation |
|-----------|-------|-------------------|----------------|
| mu | All | 5-10 | Data highly informative (8 obs) |
| tau | Variant 3a | 1-3 | Modest learning |
| tau | Variant 3b | 0.5-2 | Limited learning (tight prior) |
| tau | Variant 3c | 0.2-1 | Minimal learning (weak prior) |
| theta_i | No pool | 0.5-1 per school | Limited learning per school |

**Red flag:** If KL(tau) < 0.1 for Variant 3c (uniform prior), data are TOO WEAK to identify tau. Hierarchical model is overparameterized. **Use complete pooling instead.**

### Stress Test 4: Synthetic Data Recovery

**Test:** Simulate data with KNOWN tau:
- tau = 0 (complete pooling truth)
- tau = 5 (moderate heterogeneity)
- tau = 15 (strong heterogeneity)

For each, fit all models and check:
1. Do models correctly identify true tau?
2. What minimum tau is detectable with n=8, sigma ~ 9-18?

**Expected outcomes:**
- tau = 0: all models should recover this (favor complete pooling)
- tau = 5: hierarchical models should detect with wide CI [2, 10]
- tau = 15: even partial pooling should clearly detect (hierarchical models dominate)

**Red flag:** If models cannot reliably distinguish tau = 0 from tau = 5 with n=8 and observed sigma → **fundamental power limitation**. Any claims about tau in [0, 5] range are speculative.

**Action:** If power analysis shows tau < 5 undetectable, REPORT THIS. Don't overinterpret posterior.

---

## Decision Framework: Pivot Points

### Checkpoint 1: After Fitting All Models (2 hours into analysis)

**Questions to answer:**

1. Do hierarchical variants (3a/b/c) give similar tau posteriors?
   - **YES (all show tau < 3):** Low prior sensitivity, proceed with Variant 3a
   - **NO (tau posteriors differ by > 5):** High prior sensitivity, flag as uncertain

2. Does LOO decisively favor one model (> 4 ELPD difference)?
   - **Complete pooling wins:** Use it, hierarchical is overparameterized
   - **Hierarchical wins:** Evidence for tau > 0, focus on hierarchical
   - **Tie (within 3 ELPD):** Report both, discuss uncertainty

3. Are computational diagnostics clean?
   - **YES:** Proceed with interpretation
   - **NO:** Investigate boundary issues, reparameterization

### Checkpoint 2: After Posterior Predictive Checks (3 hours in)

**Questions to answer:**

1. Do models pass Stress Tests 1-2 (PPC, LOO calibration)?
   - **All pass:** Models are adequate, choose based on LOO/philosophy
   - **Complete pooling fails, hierarchical passes:** Evidence for heterogeneity
   - **All fail:** Something wrong with data or likelihood (investigate)

2. What is the effective parameter count (p_eff) for hierarchical models?
   - **p_eff ≈ 1-2:** Data favor low tau, use complete pooling for simplicity
   - **p_eff ≈ 4-6:** Data use hierarchical flexibility, keep it
   - **p_eff ≈ 8:** Model not shrinking (tau large or infinite), investigate

### Checkpoint 3: Power Analysis (4 hours in)

**Critical question:** With n=8 and sigma ~ 9-18, what tau is DETECTABLE?

**Simulation:**
- Generate 1000 datasets with varying tau (0, 2, 5, 10, 15)
- Fit hierarchical model to each
- Check: at what true tau does 95% CI for tau exclude 0?

**Expected result:** tau < 5 is hard to distinguish from 0 with high confidence.

**Implication:** If observed tau posterior is [0, 7], we've learned "tau is small" but cannot say "tau = 0 vs tau = 3" with confidence.

**Action:** Report uncertainty honestly. Don't claim "no heterogeneity" - claim "heterogeneity, if present, is small (tau < 5)."

### Final Decision: Model Selection

**Scenario A: Complete Pooling Dominates**
- LOO(complete) > LOO(hierarchical) by > 2 ELPD
- AND hierarchical tau posterior: 95% CI [0, 3]
- AND p_eff for hierarchical ≈ 1.5

**Decision:** **Use Complete Pooling (Model 1)** as primary model. Report hierarchical as sensitivity check. Conclusion: "Strong evidence for homogeneity, all schools share common effect ~7.7."

---

**Scenario B: Hierarchical Slightly Better**
- LOO(hierarchical) ≈ LOO(complete) within 2 ELPD
- AND tau posterior: 95% CI [0, 6], median ~ 2-3
- AND p_eff ≈ 2-3

**Decision:** **Use Hierarchical (Model 3a, Half-Cauchy)** as primary. Report complete pooling as limit case. Conclusion: "Modest evidence for heterogeneity, tau likely small (0-5), schools share substantial similarity."

---

**Scenario C: Strong Prior Sensitivity**
- Variants 3a/b/c give very different tau posteriors (medians differ by > 5)
- AND LOO similar across all models
- AND Stress Test 3 shows KL(tau) < 0.5

**Decision:** **Report uncertainty**. Cannot resolve degree of pooling with available data. Present complete pooling (lower bound) and no pooling (upper bound) as bracketing uncertainty. Conclusion: "Data insufficient to determine heterogeneity; schools are similar (tau < 10) but exact degree unknown."

---

**Scenario D: Hierarchical Strongly Favored**
- LOO(hierarchical) > LOO(complete) by > 4 ELPD
- AND tau posterior: 95% CI [5, 20], median ~ 10
- AND posterior predictive checks show complete pooling fails

**Decision:** **Use Hierarchical (Model 3a)**, investigate why EDA missed heterogeneity. Possibilities:
- Q-test underpowered (n=8)
- Outlier effects (School 1) masked by precision weighting
- Non-normal distribution of true effects
**Action:** Flag discrepancy between EDA and Bayesian analysis, investigate further.

---

## Summary: Expected Outcomes and Honest Uncertainty

### What I Expect to Find

**Most likely outcome:** Complete pooling (Model 1) or very tight hierarchical model (tau ≈ 0-2) will win. The EDA evidence for homogeneity is strong.

**Predicted tau posterior (Variant 3a):**
- Mode: 0-1
- Median: 2-3
- 95% CI: [0, 7]
- Interpretation: "Heterogeneity, if present, is small"

**Predicted model ranking (LOO):**
1. Complete pooling (ELPD ≈ -33)
2. Hierarchical - Half-Normal (ELPD ≈ -34)
3. Hierarchical - Half-Cauchy (ELPD ≈ -35)
4. Hierarchical - Uniform (ELPD ≈ -36)
5. No pooling (ELPD ≈ -40)

**Effective shrinkage:** 70-90% toward pooled mean for most schools.

### What Would Surprise Me (and Trigger Reassessment)

1. **tau posterior with substantial mass above 10**
   - Would contradict EDA heterogeneity tests
   - Investigate: influential observations, outlier effects, model misspecification

2. **No pooling outperforms complete pooling in LOO**
   - Would suggest independence assumption is correct
   - Investigate: are schools actually from different populations?

3. **Strong prior sensitivity (Variants 3a/b/c give contradictory answers)**
   - Would mean data cannot resolve pooling degree
   - Acknowledge fundamental limitation, report range

4. **Posterior predictive checks fail for complete pooling**
   - Would indicate systematic model misfit
   - Investigate: non-normal effects, outliers, measurement error misspecification

5. **Computational issues persist despite non-centered parameterization**
   - Often signature of model misspecification (wrong likelihood, wrong structure)
   - Investigate: are sigma_i really known? Is normal likelihood appropriate?

### Honest Limitations

**What these models CANNOT tell us:**

1. **Causal interpretation:** Models estimate effects, not causation. Need study design info.
2. **External validity:** Results apply to these 8 schools, generalization unknown.
3. **Measurement model:** Assume sigma_i are known exactly. If estimated, add uncertainty.
4. **Tau resolution:** With n=8, cannot distinguish tau=0 from tau=5 with high confidence.
5. **Why effects vary (if they do):** Models quantify heterogeneity, not explain it.

**What I will NOT claim:**

- "Schools have exactly the same effect" (tau=0 is a boundary, not a point)
- "We have proven homogeneity" (absence of evidence ≠ evidence of absence)
- "The hierarchical model is the truth" (all models are wrong, some useful)

**What I WILL claim:**

- "Data are consistent with substantial pooling across schools"
- "If heterogeneity exists, it is small (tau < 5-7)"
- "Measurement uncertainty dominates true variation"
- "Complete pooling is parsimonious and well-supported"

---

## Implementation Plan (Recommended Order)

### Phase 1: Fit Core Models (2 hours)

1. **Complete Pooling (Model 1)** - 30 min
   - Fit, posterior checks, LOO
   - Establish baseline

2. **No Pooling (Model 2)** - 30 min
   - Fit, compute shrinkage (none), LOO
   - Establish upper bound

3. **Hierarchical - Half-Cauchy (Variant 3a)** - 1 hour
   - Non-centered parameterization
   - Diagnostic checks (R-hat, divergences, ESS)
   - Posterior for tau, mu, theta_i
   - LOO

### Phase 2: Sensitivity Analysis (1.5 hours)

4. **Hierarchical - Half-Normal (Variant 3b)** - 45 min
   - Fit, compare tau posterior to 3a
   - Check prior sensitivity

5. **Hierarchical - Uniform (Variant 3c)** - 45 min
   - Fit, check for identification issues
   - Compare to 3a/3b

### Phase 3: Model Comparison and Diagnostics (2 hours)

6. **LOO comparison** - 30 min
   - Compare all 5 models
   - Effective parameter counts
   - ELPD differences

7. **Posterior Predictive Checks** - 1 hour
   - Stress Test 1: extreme values
   - Coverage checks
   - Visual diagnostics (shrinkage plots)

8. **Prior-Posterior Analysis** - 30 min
   - KL divergences (Stress Test 3)
   - Prior vs posterior plots for tau
   - Identify weak learning

### Phase 4: Power Analysis and Reporting (1.5 hours)

9. **Synthetic Data Simulations** - 1 hour
   - Stress Test 4: recovery of known tau
   - Determine minimum detectable tau

10. **Final Report** - 30 min
    - Model selection decision
    - Uncertainty quantification
    - Limitations and caveats

**Total estimated time:** 7 hours (can parallelize some fits)

---

## File Outputs (for Reproducibility)

All code and results will be saved to `/workspace/experiments/designer_2/`:

1. **`models.py`** - PyMC model definitions for all 5 models
2. **`fit_models.py`** - Script to fit all models and save traces
3. **`diagnostics.py`** - Posterior predictive checks, LOO, stress tests
4. **`traces/`** - Saved posterior samples (*.nc files)
5. **`results_summary.md`** - Model comparison table, decisions, interpretation
6. **`figures/`** - Diagnostic plots
   - `shrinkage_plot.png` - theta_i posteriors vs observed y_i
   - `tau_posteriors.png` - Compare tau across Variants 3a/b/c
   - `loo_comparison.png` - ELPD comparison across models
   - `ppc_max_z.png` - Posterior predictive check for max standardized residual
   - `prior_posterior_tau.png` - Prior vs posterior for tau (Variants 3a/b/c)

---

## Conclusion: Falsification Mindset

**My goal is NOT to confirm the EDA findings.** My goal is to **test whether apparent homogeneity survives rigorous Bayesian analysis**.

**Success = rejecting my models if evidence demands it.**

If complete pooling fails stress tests → I abandon it and use hierarchical.
If hierarchical shows strong prior sensitivity → I report uncertainty honestly.
If no model fits well → I investigate data quality and model assumptions.

**The data may be consistent with homogeneity, but I will not assume it.** I will test it, quantify uncertainty, and report limitations transparently.

**Red line:** I will not claim "schools have identical effects" unless:
1. Hierarchical tau posterior has 95% CI entirely below 3
2. Complete pooling passes all posterior predictive checks
3. LOO favors complete pooling by > 2 ELPD
4. Power analysis shows we could detect tau=5 if it existed

Otherwise, I report "effects are similar (substantial pooling appropriate) but exact equality unverified."

---

**End of Proposed Models Document**

**Next Steps:** Await approval to implement, or iterate based on feedback from main agent/other designers.
