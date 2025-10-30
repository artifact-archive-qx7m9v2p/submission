# Bayesian Model Design Proposal: Designer 1
## Classical Hierarchical Modeling and Pooling Strategies

**Designer:** Model Designer 1
**Focus:** Classical hierarchical meta-analysis models
**Date:** 2025-10-28
**Dataset:** J=8 studies, very low heterogeneity (I²=2.9%), pooled effect 11.27

---

## Executive Summary

Based on the EDA findings revealing very low heterogeneity (I²=2.9%), I propose THREE fundamentally different model classes representing distinct data generation processes:

1. **Complete Pooling (Common Effect) Model** - assumes a single true effect
2. **Partial Pooling (Hierarchical Random Effects) Model** - allows study-specific effects with shrinkage
3. **Weakly Informative Skeptical Prior Model** - challenges the observed large positive effect

These are NOT parameter variations - they represent fundamentally different assumptions about the underlying data generation process. Each has clear falsification criteria and is designed to potentially fail.

---

## Critical Assumptions and Falsification Philosophy

### My Core Hypothesis
The EDA suggests minimal heterogeneity, but with J=8 studies, this could be:
- **Hypothesis A:** Genuine homogeneity - all studies measure same parameter
- **Hypothesis B:** Insufficient power - heterogeneity exists but undetectable
- **Hypothesis C:** Artifact of influential studies - Study 4 dominates (33% influence)

### What Would Make Me Abandon Everything?
1. **Posterior-prior conflict:** If posterior fights the prior across all reasonable prior specifications
2. **Extreme shrinkage instability:** If removing one study changes shrinkage factors by >50%
3. **Divergent transitions in Stan:** Often indicates model misspecification, not just tuning
4. **Negative pooled effect:** EDA shows positive effect, but if posterior is robustly negative, rethink
5. **Study 5 doesn't shrink:** If the negative study resists pooling, heterogeneity is underestimated

### Decision Points for Major Pivots
- **If tau > 10:** Heterogeneity is NOT low - switch to meta-regression or mixture models
- **If LOO shows Study 4 or 5 as outliers (Pareto-k > 0.7):** Consider robust likelihoods (t-distribution)
- **If posterior mu CI excludes EDA point estimate:** Data doesn't support hierarchical structure
- **If complete and partial pooling differ by >5 units:** Random effects structure is wrong

---

## Model Class 1: Complete Pooling (Common Effect)

### Mathematical Specification

**Likelihood:**
```
y_i ~ Normal(mu, sigma_i)    for i = 1, ..., 8
```

**Priors:**
```
mu ~ Normal(0, 50)
```

**Key Feature:** No between-study variance parameter. All studies measure the same true effect with different sampling errors.

### Theoretical Justification

**From meta-analysis literature:**
- Hedges & Olkin (1985): Fixed effect model appropriate when Q-test fails to reject homogeneity
- Cochran's Q = 7.21, p = 0.407 - cannot reject H0 of homogeneity
- I² = 2.9% means 97.1% of variation is sampling error
- AIC selects this model over random effects (63.85 vs 65.82)

**Why this might be TRUE:**
- Studies may be from same lab, same population, same protocol (unknown from data)
- Very low I² suggests genuine homogeneity, not just low power
- Large sigma_i (9-18) explains observed variation without needing tau

**Why this might be WRONG:**
- J=8 gives low power to detect heterogeneity even if present
- Assumes measurement processes identical across studies (unrealistic)
- Study 5 has negative effect (-4.88) - unusual if truly homogeneous
- Ignores potential unmeasured moderators

### Expected Behavior Given EDA

**Predictions:**
- Posterior mu ≈ 11.27 (precision-weighted mean from EDA)
- 95% CI should be narrower than random effects: approximately [3.5, 19.0]
- No shrinkage variation - all studies shrink identically toward mu
- Effective sample size = J = 8 (no degrees of freedom lost)

**Comparison to EDA:**
- EDA inverse-variance weighted mean = 11.27
- Model should recover this almost exactly with weak prior
- Posterior SD(mu) ≈ sqrt(1/sum(1/sigma_i²)) ≈ 4.0

### Falsification Criteria

**I will abandon this model if:**

1. **Prior-data conflict:** If posterior strongly conflicts with prior across multiple prior choices, suggests missing model structure
   - Test: Fit with mu ~ N(0, 10), N(0, 50), N(0, 100)
   - Red flag: If posteriors diverge substantially

2. **Residual patterns:** If standardized residuals show systematic patterns
   - Test: Plot (y_i - mu_post) / sigma_i
   - Red flag: Non-random patterns, outliers beyond ±2 SD

3. **Poor LOO-CV:** If LOO predictive performance is poor
   - Test: Compute LOO-ELPD and compare to random effects
   - Red flag: ΔLOO > 4 favoring random effects (Vehtari et al. 2017)

4. **Study 5 residual extreme:** If Study 5 has |z-score| > 2.5
   - Current z = (y_5 - 11.27) / 9 = -1.79
   - Red flag: Posterior z-score exceeds threshold

5. **Posterior predictive check fails:** If observed data looks extreme under posterior
   - Test: Plot y_rep vs y_obs, compute p-values
   - Red flag: Observed data in tail of predictive distribution

**Decision rule:** Abandon if 2+ criteria met. Switch to Model 2 (random effects).

### Computational Considerations (Stan)

**Implementation:**
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
}
model {
  mu ~ normal(0, 50);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  for (j in 1:J) {
    y_rep[j] = normal_rng(mu, sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | mu, sigma[j]);
  }
}
```

**Diagnostics to monitor:**
- Rhat < 1.01 for mu (should converge easily)
- ESS > 400 per chain (high efficiency expected)
- No divergent transitions (none expected - simple model)
- Trace plots should show good mixing

**Expected computational cost:** Fast, ~100ms per 1000 iterations

**Potential issues:**
- None expected - this is a simple model
- If convergence issues arise, indicates data or implementation problem

---

## Model Class 2: Partial Pooling (Hierarchical Random Effects)

### Mathematical Specification

**Likelihood:**
```
y_i ~ Normal(theta_i, sigma_i)    for i = 1, ..., 8
```

**Study-level effects:**
```
theta_i ~ Normal(mu, tau)         for i = 1, ..., 8
```

**Priors:**
```
mu ~ Normal(0, 50)
tau ~ Half-Normal(0, 10)
```

**Key Feature:** Study-specific effects theta_i are partially pooled toward global mean mu with strength determined by tau.

### Theoretical Justification

**From meta-analysis literature:**
- DerSimonian & Laird (1986): Random effects model standard for meta-analysis
- Gelman & Hill (2007): Partial pooling optimal for bias-variance tradeoff
- Higgins & Thompson (2002): Random effects more conservative with small J
- Betancourt (2018): Hierarchical models naturally regularize via geometry

**Why this might be TRUE:**
- Studies likely have different populations, protocols, or contexts
- Conservative approach given uncertainty about heterogeneity with J=8
- Small tau (EDA estimate = 2.02) still allows for study variation
- Prediction intervals require random effects structure

**Why this might be WRONG:**
- EDA shows I² = 2.9% - tau may be zero or negligible
- With small tau, collapses to Model 1 (complete pooling)
- May waste degrees of freedom estimating tau when unnecessary
- Tau estimation unreliable with J=8 (Röver et al. 2015)

### Expected Behavior Given EDA

**Predictions:**
- Posterior mu ≈ 11.27 (similar to Model 1)
- Posterior tau ≈ 2.02 (DerSimonian-Laird estimate from EDA)
- 95% CI on mu slightly wider than Model 1: approximately [2.5, 20.0]
- Study-specific theta_i should show strong shrinkage toward mu
- Shrinkage factors: (sigma_i² / (sigma_i² + tau²)), expect >0.95 given small tau

**Shrinkage expectations:**
- Study 5 (y = -4.88, sigma = 9): theta_5 ≈ 10.5 (massive shrinkage)
- Study 4 (y = 25.73, sigma = 11): theta_4 ≈ 11.7 (strong shrinkage)
- Average shrinkage ~96% toward mu (matches EDA)

**Comparison to Model 1:**
- If tau → 0, models converge
- Expect tau posterior to include zero in credible interval
- Model 2 should have slightly worse LOO (more parameters) unless heterogeneity real

### Falsification Criteria

**I will abandon this model if:**

1. **Tau hits prior boundary:** If posterior tau concentrates at upper limit
   - Test: Check if P(tau > 9 | data) > 0.1
   - Red flag: Prior is constraining, heterogeneity higher than assumed
   - Action: Switch to Heavy-Tailed model or meta-regression

2. **Shrinkage breaks down:** If shrinkage factors vary wildly between studies
   - Test: Compute (theta_i - mu) / (y_i - mu) for each study
   - Expected: ~0.05 (95% shrinkage) with low variance
   - Red flag: Varies by >0.3 across studies
   - Interpretation: tau estimation unstable or model misspecified

3. **Influential study dominance:** If removing Study 4 changes mu by >8 units
   - EDA shows 33% influence (-3.74 change)
   - Test: LOO Pareto-k > 0.7 for Study 4
   - Red flag: Model not robust to single study
   - Action: Investigate robust likelihoods (Model 3 variant)

4. **Prior-posterior overlap minimal:** If posterior on tau far from prior
   - Test: Compute KL divergence or overlap coefficient
   - Red flag: Data strongly contradicts weakly informative prior
   - Interpretation: Either prior wrong or model misspecified

5. **Computational pathologies:** Divergent transitions persist after tuning
   - Test: Run with adapt_delta = 0.99
   - Red flag: >1% divergences remain
   - Interpretation: Funnel geometry in tau (common with small J)
   - Action: Reparameterize using non-centered parameterization

**Decision rule:** Abandon if 2+ criteria met. Consider robust variants or different model class.

### Computational Considerations (Stan)

**Implementation (Centered Parameterization):**
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta;
}
model {
  mu ~ normal(0, 50);
  tau ~ normal(0, 10);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real tau_sq = tau^2;
  vector[J] shrinkage;
  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    shrinkage[j] = (tau^2) / (tau^2 + sigma[j]^2);
  }
}
```

**Implementation (Non-Centered Parameterization - PREFERRED):**
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;  // Standard normal
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 50);
  tau ~ normal(0, 10);
  theta_raw ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
```

**Why non-centered preferred:**
- With tau likely small (≈2), funnel geometry is severe
- Centered parameterization will have divergences
- Non-centered decorrelates mu, tau, theta_raw
- Betancourt (2018) shows this is essential for small tau

**Diagnostics to monitor:**
- Rhat < 1.01 for all parameters
- ESS > 400 for mu, tau (may be lower for theta_i)
- Divergent transitions = 0 (increase adapt_delta if needed)
- Trace plots: tau should explore low values without sticking at zero
- Pairs plot: check for funnel in (tau, theta_i) space

**Expected computational cost:**
- Moderate, ~500ms per 1000 iterations
- May need adapt_delta = 0.95-0.99
- 4 chains × 2000 iterations should suffice

**Potential issues:**
1. **Funnel geometry:** tau near zero creates difficult geometry
   - Solution: Non-centered parameterization
2. **Tau boundary:** May hit zero (model reduces to Model 1)
   - Not a problem - indicates complete pooling appropriate
3. **ESS for theta_i low:** Common in hierarchical models
   - Not a problem if ESS(mu) and ESS(tau) adequate

---

## Model Class 3: Weakly Informative Skeptical Prior Model

### Mathematical Specification

**Likelihood:**
```
y_i ~ Normal(theta_i, sigma_i)    for i = 1, ..., 8
theta_i ~ Normal(mu, tau)         for i = 1, ..., 8
```

**Priors (DIFFERENT from Model 2):**
```
mu ~ Normal(0, 15)                # Skeptical: 95% CI ≈ [-30, 30]
tau ~ Half-Normal(0, 5)           # Skeptical: expects low heterogeneity
```

**Key Feature:** Priors express skepticism about large positive effects and high heterogeneity, forcing data to prove itself.

### Theoretical Justification

**From Bayesian decision theory:**
- Spiegelhalter et al. (2004): Skeptical priors protect against overconfidence
- Gelman et al. (2017): Prior predictive checks should cover plausible range
- Held & Ott (2018): Classical p-values overstate evidence; Bayesian skepticism corrects

**Why this is scientifically motivated:**
- EDA shows mu ≈ 11, but with J=8, could be sampling fluctuation
- Prior predictive: mu ~ N(0, 15) implies 95% of effects in [-30, 30]
- Observed effect (11.27) is in 62nd percentile of prior - not extreme
- If effect is real, data will overwhelm prior; if not, prior provides protection

**Philosophical stance:**
- This model embodies "show me" attitude - data must convince skeptic
- Particularly important given Study 4 influence (33% shift)
- Tests whether positive effect is robust or artifact

**Why this might be WRONG:**
- May be too skeptical - if effect is real, narrows posterior unnecessarily
- Prior on tau tighter than Model 2 - may constrain if heterogeneity exists
- Could be considered "double-dipping" if prior informed by this dataset

### Expected Behavior Given EDA

**Predictions:**
- Posterior mu should be pulled toward zero compared to Model 2
- Expected posterior mu ≈ 9-10 (shrunk from 11.27)
- Posterior width similar to Model 2, but centered lower
- If data is strong, posterior will overwhelm prior
- If data is weak, prior will provide regularization

**Prior-posterior comparison:**
- Prior: mu ~ N(0, 15), 95% CI = [-29.4, 29.4]
- Expected posterior: mu ~ N(9.5, 4.5), 95% CI ≈ [0.7, 18.3]
- Prior weight ≈ 15% (prior precision / total precision)

**Contrast with Model 2:**
- Model 2 prior: mu ~ N(0, 50), prior precision = 1/2500
- Model 3 prior: mu ~ N(0, 15), prior precision = 1/225
- Model 3 prior is 11× stronger
- If posteriors differ by <1 unit, data dominates; if >3 units, prior matters

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior-prior conflict severe:** If posterior mean far outside prior 95% CI
   - Test: Is posterior mu > 29.4 (prior 97.5% quantile)?
   - Red flag: Prior-data conflict suggests model misspecification
   - Interpretation: Prior too skeptical OR outlier process (not normal)

2. **Prior dominates data:** If posterior barely updates from prior
   - Test: Compare prior and posterior KL divergence
   - Expected: KL(post || prior) > 2 nats (substantial update)
   - Red flag: KL < 0.5 nats (prior dominates)
   - Interpretation: Data too weak or model wrong

3. **Worse predictive than Model 2:** If LOO-ELPD substantially worse
   - Test: ΔLOO between Model 3 and Model 2
   - Red flag: Model 3 worse by >4 (Vehtari et al. 2017)
   - Interpretation: Prior mismatch hurts prediction

4. **Individual study posteriors bizarre:** If skeptical prior creates strange shrinkage
   - Test: Check if Study 5 posterior goes positive (y = -4.88)
   - Expected: Should stay negative or near zero
   - Red flag: If theta_5 posterior mean > 5
   - Interpretation: Prior forcing unrealistic pooling

5. **Sensitivity to prior scale:** If changing prior SD from 15 to 10 or 20 changes mu by >3 units
   - Test: Refit with mu ~ N(0, 10) and mu ~ N(0, 20)
   - Red flag: Results highly sensitive
   - Interpretation: Data insufficient to overcome prior

**Decision rule:** Abandon if 2+ criteria met. Accept that effect is robustly positive and skepticism unwarranted.

### Computational Considerations (Stan)

**Implementation (Non-Centered):**
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  // Skeptical priors
  mu ~ normal(0, 15);      // More informative than Model 2
  tau ~ normal(0, 5);      // Tighter than Model 2
  theta_raw ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real prior_mu_sample = normal_rng(0, 15);
  real prior_tau_sample = fabs(normal_rng(0, 5));
  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
```

**Key implementation detail:**
- Include prior samples in generated quantities for direct prior-posterior comparison
- Use same non-centered parameterization as Model 2

**Diagnostics (same as Model 2):**
- Monitor convergence (Rhat, ESS)
- Check divergences (expect similar to Model 2)
- Visual inspection of prior vs posterior

**Additional diagnostic:**
- **Prior-posterior overlap:** Compute overlap coefficient
  - Overlap < 0.1 suggests strong updating (good)
  - Overlap > 0.5 suggests prior dominates (bad)

**Expected computational cost:** Same as Model 2, ~500ms per 1000 iterations

---

## Stress Tests and Red Flags

### Stress Test 1: Study 4 Removal
**Hypothesis:** If Study 4 (most influential, 33% impact) drives results, model is fragile

**Test:**
1. Fit all 3 models with full data (J=8)
2. Refit all 3 models excluding Study 4 (J=7)
3. Compare posterior mu:
   - Full data: mu_post ≈ 11.27
   - Without Study 4: expect mu_post ≈ 7.5 (based on EDA LOO)

**Red flags:**
- If Model 1 changes by >30% → Heterogeneity underestimated
- If Model 2 tau increases substantially → Study 4 masking heterogeneity
- If Model 3 posterior barely changes → Prior dominating, not learning from data

**Decision:** If mu changes by >5 units, consider:
- Robust likelihoods (Student-t errors)
- Investigation of why Study 4 is influential
- Potential for unmeasured moderator

### Stress Test 2: Prior Sensitivity Analysis
**Hypothesis:** Results should be robust to reasonable prior choices

**Test matrix:**

| Prior Setup | mu prior | tau prior |
|------------|----------|-----------|
| Weak | N(0, 100) | Half-N(0, 20) |
| Moderate (Model 2) | N(0, 50) | Half-N(0, 10) |
| Skeptical (Model 3) | N(0, 15) | Half-N(0, 5) |
| Very skeptical | N(0, 10) | Half-N(0, 3) |

**Red flags:**
- If posterior mu varies by >20% across reasonable priors → Data too weak
- If tau posterior highly sensitive → J=8 insufficient for heterogeneity estimation
- If very skeptical prior conflicts with data → Effect may be real and large

**Decision:** If high sensitivity, report range and emphasize uncertainty

### Stress Test 3: Posterior Predictive Checks
**Hypothesis:** Model should generate data resembling observed data

**Test:**
1. Draw 1000 datasets from posterior predictive: y_rep ~ posterior
2. Compute test statistics on y_rep: mean, SD, min, max, range
3. Compare to observed data test statistics
4. Compute Bayesian p-values

**Red flags:**
- If observed mean or SD in tail (p < 0.05 or p > 0.95) → Model misspecified
- If observed range much larger than predicted → Underestimating heterogeneity
- If Study 5 always flagged as extreme in y_rep → Potential outlier

**Decision:** If multiple test statistics have extreme p-values, consider:
- Robust likelihoods (t-distribution)
- Mixture models (some studies from different population)
- Meta-regression with covariates

### Stress Test 4: LOO Cross-Validation
**Hypothesis:** Model should predict held-out studies well

**Test:**
1. Compute LOO-ELPD for each model
2. Check Pareto-k diagnostics for each study
3. Compare models using ΔLOO and SE

**Red flags:**
- Pareto-k > 0.7 for any study → That study is influential/outlier
- Large ΔLOO between complete and partial pooling → Heterogeneity matters
- LOO SE very large → High uncertainty, more data needed

**Expected results:**
- Model 1 (complete pooling): ELPD ≈ -31
- Model 2 (partial pooling): ELPD ≈ -31 (similar, slightly worse due to extra parameter)
- Model 3 (skeptical): ELPD ≈ -31 to -32 (may be slightly worse if prior hurts prediction)

**Decision:** Use LOO to select final model. If ΔLOO < 4, models are equivalent.

---

## Model Comparison and Selection Strategy

### Comparison Framework

**Three models, three philosophies:**
1. **Model 1 (Complete Pooling):** Parsimony - assume simplest explanation
2. **Model 2 (Partial Pooling):** Conservatism - allow for uncertainty about heterogeneity
3. **Model 3 (Skeptical Prior):** Skepticism - make data prove large effect

**Not just parameter variations** - these represent different scientific stances:
- Model 1: "Studies are measuring same thing"
- Model 2: "Studies may differ, but we don't know how much"
- Model 3: "Large positive effects are rare, be skeptical"

### Selection Criteria

**Primary criterion: LOO-CV**
- Use LOO-ELPD as primary model selection metric
- ΔLOO > 4 is meaningful difference (Vehtari et al. 2017)
- Report LOO standard errors for uncertainty

**Secondary criteria:**
1. **Posterior predictive checks:** Model should generate realistic data
2. **Prior sensitivity:** Results should be robust to reasonable priors
3. **Computational diagnostics:** No divergences, good Rhat/ESS
4. **Scientific interpretability:** Parameters should make sense

**Tertiary criteria:**
1. **Influence analysis:** Results robust to single study removal
2. **Residual diagnostics:** No systematic patterns
3. **Domain knowledge:** Consistent with meta-analysis theory

### Expected Selection Outcome

**My prediction (falsifiable!):**

**Most likely outcome (70% confidence):**
- LOO selects Model 1 or 2 (complete or partial pooling)
- ΔLOO between them < 4 (models equivalent)
- Tau posterior in Model 2 concentrates near zero
- **Conclusion:** Complete pooling adequate, heterogeneity negligible

**Alternative outcome (25% confidence):**
- LOO equivalent across all models
- Model 3 posterior barely differs from Model 2
- **Conclusion:** Data overwhelms all reasonable priors, effect is robust

**Surprising outcome (5% confidence):**
- LOO strongly favors Model 2 over Model 1
- Tau posterior well away from zero
- **Conclusion:** EDA misled us, heterogeneity is real but hard to see with J=8

**What would shock me:**
- Model 3 posterior mu < 5 (skeptical prior dominates)
- Any model with Pareto-k > 0.7 for multiple studies
- Posterior tau > 10 in any model
- Negative posterior mu in any model

### Reporting Plan

**If models agree (expected):**
- Report all three model results in table
- Emphasize robustness across specifications
- Use Model 2 (partial pooling) for inference (conservative)
- Report Model 3 as sensitivity analysis

**If models disagree (surprising):**
- Report all results with explanation
- Investigate reasons for disagreement
- Consider additional models (robust likelihoods, meta-regression)
- Emphasize uncertainty and need for more data

**Always report:**
- Complete results table with mu, tau, CI for each model
- LOO comparison with standard errors
- Prior sensitivity analysis
- Influence analysis (Study 4 removal)
- Posterior predictive check results

---

## Prioritization and Implementation Order

### Recommended Implementation Order

**Stage 1: Baseline Models (Priority: CRITICAL)**

1. **Model 1 (Complete Pooling)** - Fit first
   - Simplest model, establishes baseline
   - Fast convergence, easy diagnostics
   - If this fails, something fundamentally wrong
   - **Time estimate:** 2 hours (coding + diagnostics)

2. **Model 2 (Partial Pooling, non-centered)** - Fit second
   - Standard model, most important for inference
   - May need tuning for adapt_delta
   - **Time estimate:** 3 hours (coding + convergence tuning)

**Stage 2: Sensitivity Analysis (Priority: HIGH)**

3. **Model 3 (Skeptical Prior)** - Fit third
   - Tests robustness to prior choice
   - Same code as Model 2, just different priors
   - **Time estimate:** 1 hour (easy adaptation)

4. **LOO Cross-Validation** - All models
   - Compute LOO-ELPD and Pareto-k diagnostics
   - **Time estimate:** 1 hour

**Stage 3: Stress Tests (Priority: MEDIUM)**

5. **Study 4 Removal** - Refit Models 1-3
   - Tests influence and robustness
   - **Time estimate:** 2 hours

6. **Prior Sensitivity Grid** - Vary priors systematically
   - 4 prior setups × 1 model (Model 2)
   - **Time estimate:** 2 hours

7. **Posterior Predictive Checks** - All models
   - Generate replicated datasets, compute test statistics
   - **Time estimate:** 2 hours

**Stage 4: Advanced Diagnostics (Priority: LOW)**

8. **Residual Analysis** - Visual diagnostics
9. **Shrinkage Visualization** - Compare observed to posterior theta
10. **Prior-Posterior Plots** - Especially for Model 3

**Total time estimate:** 15-20 hours for complete analysis

### Minimum Viable Analysis

If time constrained, absolute minimum is:
1. Model 2 (partial pooling) - fitted and diagnosed
2. LOO cross-validation
3. Basic posterior predictive checks
4. Study 4 removal sensitivity

**Time estimate:** 6-8 hours

### Stopping Rules

**Stop and reconsider if:**
1. Any model has persistent divergences after tuning
2. Pareto-k > 0.7 for multiple studies in LOO
3. Posterior predictive checks fail badly (p < 0.01 or p > 0.99)
4. Tau posterior hits upper prior limit in Model 2
5. Removing Study 4 changes results by >50%

**If any stopping rule triggered:**
- Do NOT proceed with planned analysis
- Investigate root cause
- Consider alternative model classes:
  - Robust likelihoods (Student-t)
  - Mixture models
  - Meta-regression (if covariates available)
  - Non-parametric approaches

---

## Expected Challenges and Pitfalls

### Challenge 1: Funnel Geometry with Small Tau

**Problem:** When tau ≈ 0, hierarchical model has funnel-shaped posterior
- Centered parameterization fails (divergences)
- Non-centered parameterization essential
- Even with non-centered, may need high adapt_delta

**Solution:**
- Use non-centered parameterization from start
- Set adapt_delta = 0.95 initially
- Increase to 0.99 if divergences persist
- Monitor ESS for tau - may be low even with good parameterization

**How to recognize:**
- Divergences in centered parameterization
- Pairs plot shows funnel: wide when tau large, narrow when tau small
- Trace plots show tau exploring near zero

### Challenge 2: Boundary Effects for Tau

**Problem:** True tau may be zero or near-zero
- Half-Normal prior prevents negative values
- Posterior may pile up at tau = 0
- Standard diagnostics may flag this as problem (falsely)

**Solution:**
- This is NOT a problem if scientifically plausible
- Check if tau posterior is consistent with EDA (tau_DL = 2.02)
- If tau → 0, model is telling you complete pooling appropriate
- Compare Model 1 and Model 2 LOO to confirm

**How to recognize:**
- Tau posterior mode at or near zero
- Posterior mean tau < 1
- Model 2 LOO similar to Model 1 LOO

### Challenge 3: Influential Study Masking

**Problem:** Study 4 has 33% influence on pooled estimate
- May be masking heterogeneity
- May be driving apparent homogeneity
- Removal changes estimate substantially

**Solution:**
- Always report with and without Study 4
- Check LOO Pareto-k for Study 4
- If k > 0.7, consider robust likelihood
- Investigate why Study 4 is influential (external data needed)

**How to recognize:**
- Large leave-one-out change (>30%)
- High Pareto-k diagnostic
- Posterior theta_4 far from mu

### Challenge 4: Prior-Data Balance in Model 3

**Problem:** Skeptical prior may dominate or conflict with data
- If prior too strong, learns slowly
- If prior contradicted, may indicate misspecification
- Boundary between "appropriately skeptical" and "inappropriately constraining" unclear

**Solution:**
- Report prior-posterior overlap
- Show prior-posterior plots
- Compare to Model 2 (weak prior) results
- If posteriors differ substantially, data may be weak

**How to recognize:**
- Posterior barely updates from prior (KL divergence low)
- Or: Posterior far outside prior 95% CI (prior-data conflict)
- Large difference between Model 2 and Model 3 posteriors

### Challenge 5: Low Effective Sample Size with J=8

**Problem:** Only 8 studies limits what we can learn
- Heterogeneity parameters hard to estimate
- Results sensitive to individual studies
- Posterior credible intervals wide

**Solution:**
- Acknowledge limitation explicitly
- Report uncertainty prominently
- Don't overinterpret small differences
- Consider results exploratory, not definitive

**How to recognize:**
- Wide credible intervals on tau
- High sensitivity to prior choice
- Large influence measures for individual studies

---

## Connection to Other Designers

I am working independently, but recognize others may propose:
- **More complex models:** Robust likelihoods, non-parametric approaches
- **Different model classes:** Meta-regression, state-space, GPs
- **Alternative priors:** Informative priors from external data

**My models are baseline classics** - should be fitted regardless of what others propose:
- Model 1: Simplest possible (common effect)
- Model 2: Standard approach (random effects)
- Model 3: Sensitivity analysis (skeptical)

**If other designers find:**
- My models inadequate (falsification criteria met) → Good! We learned something
- Robust/GP models fit better → Suggests outliers or non-normality I missed
- Meta-regression explains variation → Covariates matter, I couldn't test

**Success is NOT** all models fitting well
**Success IS** learning which models fail and why

---

## Summary and Key Takeaways

### Three Models, Three Questions

1. **Model 1 asks:** Is one true effect enough?
2. **Model 2 asks:** How much do studies really differ?
3. **Model 3 asks:** Is the large positive effect robust to skepticism?

### Falsification Mindset

Each model has clear criteria for rejection:
- **Model 1:** Poor residuals, LOO favors Model 2, Study 5 extreme
- **Model 2:** Tau hits boundary, divergences persist, Study 4 dominates
- **Model 3:** Prior conflicts with data, or data too weak to update prior

### Expected Outcome (Falsifiable)

**My prediction:** Models 1 and 2 will be equivalent (tau ≈ 0), Model 3 will show data overwhelms skeptical prior, all models agree mu ≈ 10-12.

**What would change my mind:**
- Tau posterior well away from zero
- Study 4 or 5 flagged by LOO (Pareto-k > 0.7)
- Posterior predictive checks fail
- Substantial disagreement across models

### Implementation Priority

1. **Must do:** Models 1 and 2, LOO comparison
2. **Should do:** Model 3, Study 4 removal, posterior predictive checks
3. **Nice to have:** Full prior sensitivity grid, extensive diagnostics

### Success Metrics

**Success is NOT:**
- All models converging without issues
- Results matching EDA exactly
- Narrow credible intervals

**Success IS:**
- Understanding which models fail and why
- Quantifying uncertainty honestly
- Finding evidence that forces model revision
- Learning about the data generation process

---

## References

- Betancourt, M. (2018). A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. Controlled Clinical Trials, 7(3), 177-188.
- Gelman, A., & Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
- Gelman, A., et al. (2017). The Prior Can Often Only Be Understood in the Context of the Likelihood. Entropy, 19(10), 555.
- Hedges, L. V., & Olkin, I. (1985). Statistical Methods for Meta-Analysis. Academic Press.
- Held, L., & Ott, M. (2018). On p-Values and Bayes Factors. Annual Review of Statistics and Its Application, 5, 393-419.
- Higgins, J. P., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. Statistics in Medicine, 21(11), 1539-1558.
- Röver, C., et al. (2015). Bayesian random-effects meta-analysis using the bayesmeta R package. arXiv:1711.08683.
- Spiegelhalter, D. J., et al. (2004). Bayesian measures of model complexity and fit. Journal of the Royal Statistical Society: Series B, 64(4), 583-639.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing, 27(5), 1413-1432.

---

**Document prepared by:** Model Designer 1
**Focus area:** Classical hierarchical modeling and pooling strategies
**Models proposed:** 3 distinct classes (complete pooling, partial pooling, skeptical prior)
**Falsification criteria:** Specified for each model
**Implementation time:** 15-20 hours full analysis, 6-8 hours minimum viable
**Key principle:** Success = discovering model failures early, not confirming preconceptions

---

**File location:** `/workspace/experiments/designer_1/proposed_models.md`
