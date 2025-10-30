# Bayesian Model Proposals for Eight Schools Dataset
## Designer 3 - Independent Model Design

**Date**: 2025-10-29
**Designer**: Designer 3
**Output Directory**: `/workspace/experiments/designer_3/`

---

## Executive Summary

Based on the EDA findings (I² = 1.6%, variance ratio = 0.75, high individual uncertainty), I propose three distinct Bayesian model classes that embody fundamentally different assumptions about the data generation process:

1. **Near-Complete Pooling Hierarchical Model** - Assumes effects are highly similar with minimal heterogeneity
2. **Flexible Horseshoe Hierarchical Model** - Assumes most schools are similar, but allows for a few true outliers
3. **Measurement Error Robust Model** - Questions whether the reported sigmas are truly known, allowing for sigma misspecification

**Critical stance**: The EDA suggests complete pooling might suffice (I² = 1.6%), but I'm suspicious of taking this at face value with only 8 schools. The variance paradox (observed < expected) is unusual and could indicate: (a) true homogeneity, (b) measurement issues, or (c) small-sample artifacts. My models explore these competing explanations.

---

## Model 1: Near-Complete Pooling Hierarchical Model (BASELINE)

### Mathematical Specification

```
Likelihood:    y_i ~ Normal(theta_i, sigma_i)     [sigma_i known]
School level:  theta_i ~ Normal(mu, tau)
Hyperpriors:   mu ~ Normal(0, 50)
               tau ~ HalfNormal(0, 5)             [INFORMATIVE - expects small tau]
```

### Theoretical Rationale

**Why this model for this problem?**

The EDA provides overwhelming evidence for near-homogeneity:
- I² = 1.6% (extremely low heterogeneity)
- Chi-square test p = 0.417 (fail to reject homogeneity)
- Variance ratio = 0.75 (observed variation LESS than expected from sampling alone)

This model differs from the canonical "weakly informative" approach by using a **HalfNormal(0, 5)** prior on tau instead of HalfCauchy(0, 25). This prior:
- Concentrates mass near zero (median ≈ 3.4)
- Allows tau up to ~15 at 99th percentile
- Embeds the EDA finding that heterogeneity appears minimal
- Still permits heterogeneity if strongly supported by data

**Key assumption**: Schools are implementing essentially the same intervention with similar populations, and apparent differences are mostly sampling noise.

### Prior Justification

**mu ~ Normal(0, 50)**:
- Centers on zero (no prior bias toward positive/negative effects)
- SD = 50 allows for effects ranging from -100 to +100 at 95% prior mass
- Observed effects range from -5 to 26, so prior is weakly informative
- Based on typical educational intervention effect sizes (often |d| < 1, or ~10-20 points on many scales)

**tau ~ HalfNormal(0, 5)**:
- Median prior τ ≈ 3.4
- 95% prior mass: τ ∈ (0, 10)
- Justification: Observed between-school SD ≈ 11, but this is inflated by sampling error. True τ likely much smaller.
- The EDA variance ratio of 0.75 suggests τ might be near zero
- HalfNormal has lighter tail than HalfCauchy - less mass on large τ values
- This is an **informed prior** based on EDA, not a generic weakly informative prior

### Stan Implementation

```stan
data {
  int<lower=0> J;                // number of schools = 8
  vector[J] y;                   // observed effects
  vector<lower=0>[J] sigma;      // known standard errors
}

parameters {
  real mu;                       // population mean effect
  real<lower=0> tau;             // between-school SD
  vector[J] theta_raw;           // non-centered parameterization
}

transformed parameters {
  vector[J] theta;
  theta = mu + tau * theta_raw;  // non-centered: avoids funnel geometry
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ normal(0, 5);            // half-normal via constraint
  theta_raw ~ std_normal();      // standard normal

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Posterior predictive for new school
  real theta_new = normal_rng(mu, tau);
  real y_new = normal_rng(theta_new, mean(sigma)); // using average sigma

  // Shrinkage factors
  vector[J] shrinkage;
  for (j in 1:J) {
    shrinkage[j] = 1 - pow(tau, 2) / (pow(tau, 2) + pow(sigma[j], 2));
  }

  // Log-likelihood for LOO-CV
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
```

**Key implementation details**:
- **Non-centered parameterization**: Essential for avoiding funnel geometry when tau is small
- **Generated quantities**: Compute shrinkage factors and predictive checks
- **LOO-CV preparation**: Log-likelihood for model comparison

### Expected Behavior

**If model is correct**:
1. Posterior τ will be small (E[τ|data] < 5, possibly near zero)
2. Strong shrinkage toward mu for all schools:
   - School 4 (effect = 25.7) shrinks substantially toward mu
   - School 5 (effect = -4.9) shrinks toward positive mu
   - Schools with larger sigma_i shrink more (School 8)
3. Posterior intervals for theta_i will overlap heavily
4. mu will be around 8-12 (weighted toward precise schools)
5. Posterior predictive checks should show good fit
6. Low LOO-CV uncertainty (stable predictions)

**Computational expectations**:
- Should converge easily with NUTS sampler
- Non-centered parameterization prevents funnel issues
- 4 chains × 2000 iterations should suffice
- Expect R-hat < 1.01 for all parameters

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior τ is large and well-identified**: If posterior mean τ > 10 with narrow credible interval, this contradicts the prior assumption of near-homogeneity
   - Action: Switch to Model 2 (Horseshoe) which better handles heterogeneity

2. **Prior-posterior conflict on τ**: If posterior pushes against the prior upper tail (strong evidence for τ > 15), the informative prior is being rejected by data
   - Action: Re-run with HalfCauchy(0, 25) to see if results change substantially

3. **Poor posterior predictive checks**: If replicated datasets systematically differ from observed (e.g., underestimate variance, show different patterns)
   - Specifically: If P(SD(y_rep) < SD(y_obs)) > 0.95, model is too constrained
   - Action: Investigate Model 3 (sigma misspecification)

4. **Leave-one-out failures**: If LOO Pareto-k > 0.7 for multiple schools, indicating influential outliers that model can't accommodate
   - Especially School 5 (negative effect) or School 4 (largest effect)
   - Action: Switch to robust model or horseshoe

5. **Shrinkage feels scientifically implausible**: If domain experts insist schools truly differ (e.g., different populations, interventions), strong pooling is inappropriate
   - Action: Reconsider model structure entirely, possibly incorporate covariates

6. **Comparison to complete pooling shows no benefit**: If LOO-CV shows hierarchical model performs no better than simple complete pooling, the extra complexity isn't justified
   - This would actually SUPPORT the EDA finding of near-complete homogeneity
   - Action: Report complete pooling as adequate

### Stress Tests

**Test 1: Sensitivity to tau prior**
- Re-fit with HalfCauchy(0, 25) [standard weakly informative]
- Re-fit with HalfNormal(0, 10) [even less informative]
- If posterior τ changes substantially, prior is dominating likelihood
- **Decision rule**: If posterior τ changes by >50% across priors, increase prior uncertainty

**Test 2: Leave-one-out School 5**
- Fit model excluding School 5 (the negative outlier)
- Check if posterior τ decreases (suggesting School 5 drives heterogeneity)
- Check if mu increases (School 5 pulls mean down)
- **Decision rule**: If excluding School 5 changes mu by >3 points or τ by >2 points, School 5 is highly influential and may deserve special modeling

**Test 3: Leave-one-out School 4**
- Fit model excluding School 4 (largest effect)
- Check if posterior τ decreases
- **Decision rule**: Similar to School 5 test

**Test 4: Posterior predictive p-value**
- T(y) = SD(y) [test statistic]
- Compute p_B = P(SD(y_rep) < SD(y_obs) | data)
- **Decision rule**: If p_B > 0.95 or p_B < 0.05, model misfit

---

## Model 2: Flexible Horseshoe Hierarchical Model (SPARSE HETEROGENEITY)

### Mathematical Specification

```
Likelihood:    y_i ~ Normal(theta_i, sigma_i)
School level:  theta_i ~ Normal(mu, lambda_i * tau)    [school-specific variance]
               lambda_i ~ HalfCauchy(0, 1)              [local shrinkage]
Hyperpriors:   mu ~ Normal(0, 50)
               tau ~ HalfCauchy(0, 25)                  [global shrinkage]
```

### Theoretical Rationale

**Alternative hypothesis**: What if most schools are similar (justifying strong pooling), but 1-2 schools are genuinely different outliers?

The horseshoe prior allows for **sparse heterogeneity**:
- Most lambda_i will be near zero (strong shrinkage for most schools)
- A few lambda_i can be large (minimal shrinkage for outliers)
- The model adaptively decides which schools are outliers

**Key difference from Model 1**: Model 1 applies same shrinkage to all schools (proportional to sigma_i). Model 2 allows school-specific shrinkage independent of measurement error.

**When would this be true?**
- School 5 used different intervention protocol (hence negative effect)
- School 4 had different population (hence large positive effect)
- Remaining 6 schools are homogeneous

**Why consider this despite low I²?**
- I² averages over all schools - might miss sparse outliers
- With n=8, power to detect 1-2 outliers is low
- Horseshoe naturally shrinks to complete pooling if no outliers exist

### Prior Justification

**lambda_i ~ HalfCauchy(0, 1)**:
- Standard horseshoe specification
- Heavy tails allow individual lambda_i to be arbitrarily large
- But most mass near zero, so most schools will shrink strongly
- Scale parameter = 1 is conventional (pairs with tau for overall scale)

**tau ~ HalfCauchy(0, 25)**:
- Global shrinkage parameter
- Sets overall scale of heterogeneity
- Heavier tail than Model 1 - allows for more heterogeneity
- Scale = 25 is standard for educational data (Gelman 2006)

**Effective prior on theta_i**:
- theta_i - mu ~ HorseShoE(0, tau)
- Heavy tails, sharp peak at zero
- Different from Normal: more mass at zero AND in tails
- Ideal for "most similar, few different" scenario

### Stan Implementation

```stan
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  vector<lower=0>[J] lambda;     // local shrinkage parameters
  vector[J] theta_raw;
}

transformed parameters {
  vector[J] theta;
  for (j in 1:J) {
    theta[j] = mu + tau * lambda[j] * theta_raw[j];
  }
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 25);
  lambda ~ cauchy(0, 1);
  theta_raw ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Identify "outlier" schools (lambda > 1)
  vector[J] is_outlier;
  for (j in 1:J) {
    is_outlier[j] = lambda[j] > 1.0 ? 1.0 : 0.0;
  }

  // Effective shrinkage
  vector[J] shrinkage;
  for (j in 1:J) {
    shrinkage[j] = pow(lambda[j] * tau, 2) / (pow(lambda[j] * tau, 2) + pow(sigma[j], 2));
  }

  // PPCs
  real theta_new = normal_rng(mu, tau * cauchy_rng(0, 1));
  real y_new = normal_rng(theta_new, mean(sigma));

  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
```

**Implementation notes**:
- Non-centered parameterization still used
- lambda_i are school-specific scale parameters
- Generated quantities identify which schools are "outliers" (lambda > 1)

### Expected Behavior

**If model is correct (sparse heterogeneity)**:
1. Most lambda_i will be small (< 0.5), indicating most schools pool strongly
2. 1-2 lambda_i will be large (> 1), indicating true outliers
   - Candidate outliers: School 5 (negative) and School 4 (largest)
3. Posterior tau might be moderate (not as small as Model 1)
4. Outlier schools will have wider posterior intervals (less shrinkage)
5. Non-outlier schools will shrink nearly to complete pooling

**If sparse heterogeneity is WRONG (truly homogeneous)**:
1. All lambda_i will be small and similar
2. Model will behave like Model 1 (horseshoe shrinks to normal)
3. LOO-CV will likely penalize extra complexity
4. This is evidence FOR Model 1, not against Model 2

**Computational expectations**:
- More challenging than Model 1 due to heavy-tailed priors
- May need more iterations (4000-5000)
- Watch for divergences (may need adapt_delta = 0.95)
- Cauchy priors can cause slow mixing

### Falsification Criteria

**I will abandon this model if**:

1. **All lambda_i are essentially identical**: If posterior lambda_i all fall in narrow range (e.g., all in [0.3, 0.6]), no evidence for sparse heterogeneity
   - Action: Model 1 is simpler and should be preferred

2. **Many schools identified as outliers**: If 5+ schools have lambda > 1, this is NOT sparse heterogeneity, it's just heterogeneity
   - Action: Standard hierarchical model with HalfCauchy prior on tau (not horseshoe)

3. **No improvement in LOO-CV over Model 1**: If ELPD difference is negligible (< 1) and SE overlaps zero, extra complexity isn't justified
   - Action: Prefer simpler Model 1

4. **Computational failures**: If chains don't converge after tuning (R-hat > 1.05), model is too complex for this data
   - Action: Abandon horseshoe, stick with Model 1

5. **Lambda posteriors are prior-dominated**: If posteriors look like priors (flat tail), data aren't informative about local shrinkage
   - Action: Model 1 is more appropriate

6. **Scientifically implausible outlier identification**: If model identifies School 8 (highest sigma) as outlier, this might just be reflecting measurement uncertainty, not true difference
   - Action: Model 1 or Model 3

### Stress Tests

**Test 1: Which schools are outliers?**
- Examine posterior P(lambda_i > 1 | data) for each school
- Expected candidates: Schools 4 and 5
- **Decision rule**: If no school has P(lambda > 1) > 0.5, no evidence for sparse heterogeneity

**Test 2: Sensitivity to lambda prior scale**
- Re-fit with lambda ~ HalfCauchy(0, 0.5) [more shrinkage]
- Re-fit with lambda ~ HalfCauchy(0, 2) [less shrinkage]
- **Decision rule**: If outlier identification changes, results are prior-sensitive (bad sign)

**Test 3: Compare to standard hierarchical**
- Fit standard model with tau ~ HalfCauchy(0, 25), no horseshoe
- Compare LOO-CV
- **Decision rule**: If horseshoe ELPD < standard hierarchical, complexity isn't helping

**Test 4: Prior predictive check**
- Generate datasets from prior
- Check if prior generates reasonable heterogeneity patterns
- **Decision rule**: If prior generates mostly extreme heterogeneity (not sparse), recalibrate tau prior scale

---

## Model 3: Measurement Error Robust Model (SIGMA MISSPECIFICATION)

### Mathematical Specification

```
Likelihood:    y_i ~ Normal(theta_i, sigma_i_true)
Meas. error:   sigma_i_true = sigma_i * psi_i        [unknown inflation factors]
School level:  theta_i ~ Normal(mu, tau)
Error model:   psi_i ~ LogNormal(0, omega)           [multiplicative errors]
Hyperpriors:   mu ~ Normal(0, 50)
               tau ~ HalfNormal(0, 5)
               omega ~ HalfNormal(0, 0.3)           [small misspecification]
```

### Theoretical Rationale

**Adversarial question**: What if the variance paradox (observed variance < expected) isn't evidence of homogeneity, but evidence that the reported sigma_i are **wrong**?

Specifically:
- What if some sigmas are underestimated? (psi_i > 1)
- What if some sigmas are overestimated? (psi_i < 1)
- This would create spurious patterns in the data

**Why might sigmas be wrong?**
1. **Study quality variation**: Some schools had better-designed studies (more reliable sigma)
2. **Sample size misreporting**: Sigma depends on n; if n was misreported, sigma is wrong
3. **Within-school heterogeneity**: Reported sigma assumes homogeneous participants, but some schools might have had more variable populations
4. **Different estimation methods**: Schools might have used different formulas or software

**Evidence from EDA**:
- Variance ratio = 0.75 is unusual - typically see ratio ≥ 1
- School 8 has sigma = 18 (2x School 5's sigma = 9) - why such difference?
- If School 8's sigma is overestimated by 30%, true sigma ≈ 12.6, making it more comparable

**Key insight**: In traditional analysis, sigma_i are treated as **known**. But they're estimates from studies with finite samples. Why not model our uncertainty about them?

### Prior Justification

**psi_i ~ LogNormal(0, omega)**:
- LogNormal ensures psi_i > 0 (can't have negative SD)
- Mean of LogNormal(0, omega) ≈ 1 for small omega (unbiased by default)
- psi_i < 1: sigma_i was overestimated (true uncertainty smaller)
- psi_i > 1: sigma_i was underestimated (true uncertainty larger)
- LogNormal is standard for multiplicative errors

**omega ~ HalfNormal(0, 0.3)**:
- omega controls how wrong sigmas can be
- omega = 0: sigmas are exactly correct (reduces to Model 1)
- omega = 0.3: 95% of psi_i fall in [0.5, 2.0] approximately
  - i.e., sigmas can be wrong by factor of 2 in either direction
- This is weakly informative: allows misspecification but doesn't expect large errors
- Based on: typical SE estimation errors in meta-analyses are 10-30%

**Why this parameterization?**
- School-specific psi_i (not global) because different studies might have different quality
- But shared omega pools information about overall misspecification level

### Stan Implementation

```stan
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma_reported;   // reported SEs (now treated as uncertain)
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
  vector<lower=0>[J] psi;              // inflation factors
  real<lower=0> omega;                 // misspecification scale
}

transformed parameters {
  vector[J] theta;
  vector[J] sigma_true;

  theta = mu + tau * theta_raw;

  for (j in 1:J) {
    sigma_true[j] = sigma_reported[j] * psi[j];
  }
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ normal(0, 5);
  theta_raw ~ std_normal();

  omega ~ normal(0, 0.3);
  psi ~ lognormal(0, omega);           // school-specific misspecification

  // Likelihood with corrected sigmas
  y ~ normal(theta, sigma_true);
}

generated quantities {
  // How wrong were the reported sigmas?
  vector[J] sigma_correction_factor = psi;
  real mean_correction = mean(psi);

  // Did correcting sigmas affect inferences?
  real tau_to_sigma_ratio = tau / mean(sigma_true);

  // Posterior predictive (using corrected uncertainties)
  real theta_new = normal_rng(mu, tau);
  real y_new = normal_rng(theta_new, mean(sigma_true));

  // Which schools had worst misspecification?
  vector[J] misspec_magnitude;
  for (j in 1:J) {
    misspec_magnitude[j] = fabs(log(psi[j]));  // distance from 1 on log scale
  }

  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma_true[j]);
  }
}
```

**Implementation notes**:
- sigma_reported is now just data, not fixed truth
- sigma_true is inferred from data
- psi = 1 means reported sigma was correct
- Can compare sigma_true to sigma_reported to assess misspecification

### Expected Behavior

**If sigmas are misspecified**:
1. Posterior omega will be clearly > 0 (narrow credible interval away from zero)
2. Some psi_i will be clearly ≠ 1
3. School 8 (highest reported sigma) might have psi < 1 (overestimated)
4. Correcting sigmas might increase variance ratio toward 1
5. LOO-CV will improve vs Model 1 (better predictive performance)
6. Posterior tau might increase (once we correct for inflated sigmas, heterogeneity becomes more apparent)

**If sigmas are correct**:
1. Posterior omega will be small, possibly near zero
2. All psi_i posteriors will include 1 with high probability
3. sigma_true ≈ sigma_reported for all schools
4. Model reduces to Model 1 (extra parameters don't help)
5. LOO-CV will penalize extra complexity (no benefit)

**Computational expectations**:
- More complex than Model 1 (J + 2 extra parameters)
- LogNormal can be tricky; may need reparameterization
- Might need more iterations to explore psi space
- Watch for correlations between tau and omega

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior omega near zero**: If 95% credible interval for omega includes zero or posterior mean < 0.1, no evidence for misspecification
   - Action: Model 1 is correct, sigmas are fine

2. **All psi_i include 1**: If all 95% credible intervals for psi_i include 1.0, we can't distinguish corrected from reported sigmas
   - Action: Model 1 is adequate

3. **No improvement in LOO-CV**: If ELPD(Model 3) ≈ ELPD(Model 1), complexity isn't justified
   - Action: Prefer simpler Model 1

4. **Corrections are implausibly large**: If posterior suggests psi > 3 or psi < 0.3 for multiple schools, model is compensating for poor theta model, not sigma misspecification
   - E.g., if School 5's psi = 10, model is trying to "explain away" the negative effect by inflating its uncertainty
   - Action: This is model misuse, abandon

5. **Pattern doesn't match data generation**: If we know sigmas were computed carefully from large samples, misspecification is implausible
   - Action: Don't use this model if domain knowledge says sigmas are reliable

6. **Posterior omega is prior-dominated**: If posterior omega ≈ prior omega (no learning), data aren't informative
   - Action: Model 1

### Stress Tests

**Test 1: Prior sensitivity on omega**
- Re-fit with omega ~ HalfNormal(0, 0.5) [more misspecification allowed]
- Re-fit with omega ~ HalfNormal(0, 0.15) [less misspecification]
- **Decision rule**: If conclusions change (e.g., omega clearly > 0 vs ≈ 0), results are prior-driven

**Test 2: Which schools have misspecified sigmas?**
- Examine P(psi_i > 1.2 or psi_i < 0.8 | data) for each school
- **Decision rule**: If all P < 0.5, no clear misspecification detected

**Test 3: Does correcting sigmas resolve variance paradox?**
- Compute variance ratio with sigma_true instead of sigma_reported
- **Decision rule**: If ratio moves closer to 1.0, misspecification explains the paradox

**Test 4: Compare to fixed-sigma model**
- Compute posterior predictive p-value for variance in both models
- **Decision rule**: If Model 3 has better calibration (p-value closer to 0.5), it's a better model

---

## Model Comparison Strategy

### Primary Metrics

1. **LOO-CV (Leave-One-Out Cross-Validation)**:
   - Compute ELPD (Expected Log Predictive Density) for each model
   - Compare via elpd_diff and SE
   - **Decision rule**: Prefer model with higher ELPD if difference > 2*SE

2. **Pareto-k diagnostics**:
   - k < 0.5: Good
   - 0.5 < k < 0.7: OK
   - k > 0.7: Problematic (influential observation)
   - **Decision rule**: If model has multiple k > 0.7, it's misspecified

3. **Posterior predictive checks**:
   - Test statistic: SD(y), variance ratio, min(y), max(y)
   - Compute Bayesian p-value
   - **Decision rule**: p-value in [0.05, 0.95] indicates good calibration

4. **WAIC (Widely Applicable Information Criterion)**:
   - Alternative to LOO-CV
   - Should agree with LOO-CV if models are reasonable
   - **Decision rule**: If LOO and WAIC disagree, investigate

### Secondary Considerations

5. **Parameter interpretation**:
   - Are posteriors scientifically plausible?
   - Are effect sizes reasonable for educational interventions?
   - Do school rankings make sense?

6. **Prior sensitivity**:
   - If conclusions change drastically with different priors, be skeptical
   - Report sensitivity analyses

7. **Computational efficiency**:
   - All else equal, prefer simpler model
   - If Model 1 performs as well as Model 2/3, use Model 1

### Expected Model Rankings

**Most likely scenario** (based on EDA):
1. **Model 1 wins**: Low heterogeneity is real, not artifact
   - LOO-CV best or tied
   - Posterior tau small
   - Good posterior predictive checks

**Alternative scenario 1** (sparse outliers):
2. **Model 2 wins**: School 5 (and maybe 4) are true outliers
   - LOO-CV better than Model 1 (especially for Schools 4, 5)
   - Lambda clearly > 1 for 1-2 schools
   - Posterior tau moderate

**Alternative scenario 2** (measurement issues):
3. **Model 3 wins**: Reported sigmas are unreliable
   - LOO-CV better than Model 1
   - Posterior omega > 0 with narrow CI
   - Corrected sigmas resolve variance paradox

**Null scenario** (all models equivalent):
- All three models give similar predictions
- LOO-CV differences < 1 standard error
- **Interpretation**: Data are too sparse to distinguish hypotheses
- **Action**: Report Model 1 (simplest), note that data are consistent with multiple explanations

---

## Critical Decision Points

### Stage 1: After Model 1 Fit (Baseline)

**Questions to answer**:
1. Is posterior tau near zero (< 3)?
2. Do posterior predictive checks pass (p ∈ [0.05, 0.95])?
3. Are there LOO-CV flags (Pareto k > 0.7)?

**Decision**:
- If YES to Q1-2, NO to Q3: **Model 1 is adequate, STOP**
- If NO to Q1: tau is substantial, proceed to **Stage 2A** (Model 2)
- If YES to Q3: influential outliers, proceed to **Stage 2A** (Model 2)
- If NO to Q2 (especially variance paradox not resolved): proceed to **Stage 2B** (Model 3)

### Stage 2A: After Model 2 Fit (Horseshoe)

**Questions**:
1. Are 1-2 schools clearly identified as outliers (lambda > 1)?
2. Is LOO-CV better than Model 1?
3. Do identified outliers make scientific sense?

**Decision**:
- If YES to all: **Model 2 wins**, report horseshoe results
- If NO to Q1: no sparse heterogeneity, **Model 1 wins**
- If NO to Q2: extra complexity not justified, **Model 1 wins**
- If NO to Q3: model is data-fitting, not discovering truth, **be suspicious**

### Stage 2B: After Model 3 Fit (Sigma misspecification)

**Questions**:
1. Is posterior omega clearly > 0?
2. Do corrected sigmas resolve variance paradox?
3. Is LOO-CV better than Model 1?
4. Are corrections scientifically plausible?

**Decision**:
- If YES to all: **Model 3 wins**, sigmas were misspecified
- If NO to Q1 or Q2: no evidence for misspecification, **Model 1 wins**
- If NO to Q3: complexity not helping, **Model 1 wins**
- If NO to Q4: model is compensating for other issues, **rethink everything**

### Stage 3: Final Model Selection

**Compare all models**:
- Compute ELPD differences
- Check if differences > 2*SE
- Review posterior predictive checks for all

**Final decision rule**:
1. If Model 1 LOO-CV within 1 SE of others: **prefer Model 1 (simplicity)**
2. If Model 2 or 3 clearly better (difference > 2 SE): **prefer best model**
3. If all models fail posterior predictive checks: **ABANDON ALL, RECONSIDER**

---

## Alternative Approaches if All Models Fail

If all three models show:
- Poor posterior predictive checks
- High Pareto-k values
- Inconsistent results
- Scientifically implausible inferences

**Then consider**:

### Plan B: Non-exchangeable Model
- Maybe schools aren't exchangeable (e.g., Schools 1-4 are urban, 5-8 are rural)
- Add structure: theta_i ~ Normal(mu_group[i], tau)
- Requires external information about grouping

### Plan C: Robust Likelihood
- Replace Normal with Student-t for y_i
- Allows for outliers in effect estimates (not just variance)
- Only if normality tests fail (they didn't in EDA)

### Plan D: Mixture Model
- theta_i ~ p * Normal(mu1, tau1) + (1-p) * Normal(mu2, tau2)
- Two populations of schools
- Requires strong evidence of bimodality (not present in EDA)

### Plan E: Regression on Covariates
- If sigma_i is predictive: theta_i ~ Normal(mu + beta * sigma_i, tau)
- Current EDA shows weak correlation (r = 0.43, p = 0.29)
- Only pursue if relationship strengthens with different model

### Plan F: Reconsider Data Quality
- Contact original investigators
- Check for data entry errors
- Verify school identities and study protocols
- This is non-statistical but often most productive

---

## Red Flags and Escape Routes

### Red Flag 1: Posterior Appears Prior-Dominated

**Symptoms**:
- Posteriors closely match priors
- High posterior uncertainty despite data
- Minimal learning

**Diagnosis**: Data too weak to overcome prior

**Action**:
- Try more diffuse priors
- But accept that with n=8, some questions may be unanswerable
- Report high posterior uncertainty honestly

### Red Flag 2: Computational Issues

**Symptoms**:
- Divergent transitions
- Low effective sample size (ESS < 100)
- R-hat > 1.05
- Chains don't mix

**Diagnosis**: Model geometry issues or misspecification

**Action**:
- Increase adapt_delta to 0.95 or 0.99
- Try non-centered parameterization (already used)
- Simplify model (remove complexity)
- If nothing works: model is too complex for this data

### Red Flag 3: Extreme Parameter Values

**Symptoms**:
- tau > 50 (implausibly large between-school variation)
- theta_i > 100 or < -100 (implausible effect sizes)
- psi > 5 (sigmas wrong by factor of 5+)

**Diagnosis**: Model compensating for misspecification

**Action**:
- Check data for errors
- Add stronger priors (weakly informative → informative)
- Reconsider likelihood structure

### Red Flag 4: All Schools Shrink to Same Value

**Symptoms**:
- Posterior theta_i essentially identical
- Very small tau (< 0.1)
- Model effectively implements complete pooling

**Diagnosis**: This might actually be CORRECT given EDA

**Action**:
- Accept that complete pooling may be appropriate
- Report that hierarchical model discovered near-homogeneity
- This is success, not failure

### Red Flag 5: LOO-CV Says All Models Equal

**Symptoms**:
- ELPD differences < 0.5
- Standard errors large relative to differences
- Pareto-k values similar across models

**Diagnosis**: Data too sparse to distinguish hypotheses

**Action**:
- Report simplest model (Model 1)
- Acknowledge uncertainty about data generating process
- State clearly: "Multiple models are consistent with the data"
- This is honest science

---

## Success Criteria

**This modeling exercise will be successful if**:

1. **We find a well-fitting model**: Posterior predictive checks pass, LOO-CV stable
2. **Inferences are scientifically plausible**: Effect sizes reasonable, shrinkage makes sense
3. **We understand the data generation process**: Can articulate why schools differ (or don't)
4. **Computational convergence**: R-hat < 1.01, ESS > 400, no divergences
5. **Prior sensitivity acceptable**: Conclusions robust to reasonable prior choices
6. **We know when to stop**: Either found good model, or determined data insufficient

**Success does NOT require**:
- Finding large heterogeneity (homogeneity is a valid finding)
- Complex model winning (simplicity is valuable)
- Rejecting null hypothesis (science isn't about p-values)
- Certainty (acknowledging uncertainty is scientific maturity)

**The goal is TRUTH, not completing a checklist.**

---

## Implementation Timeline

### Phase 1: Fit Model 1 (Baseline)
- **Time**: 30 minutes
- Implement Stan code
- Run 4 chains, 2000 iterations each
- Check diagnostics
- Compute LOO-CV
- Posterior predictive checks
- **Output**: Baseline results, decision on Stage 2

### Phase 2: Conditional Fits
- **Model 2** (if Stage 1 indicates heterogeneity): 1 hour
  - More complex, may need tuning
- **Model 3** (if Stage 1 shows variance issues): 1 hour
  - Check for sigma misspecification
- **Output**: Comparison of 2-3 models

### Phase 3: Model Comparison
- **Time**: 30 minutes
- Compute all pairwise LOO-CV comparisons
- Sensitivity analyses
- Final posterior predictive checks
- **Output**: Model selection decision

### Phase 4: Stress Tests
- **Time**: 1 hour
- Leave-one-out schools
- Prior sensitivity
- Robustness checks
- **Output**: Confidence in selected model

### Total Time: 3-4 hours

---

## Conclusion

I've proposed three distinct model classes that embody competing explanations for the Eight Schools data:

1. **Model 1**: The variance paradox reflects true homogeneity → Near-complete pooling is appropriate
2. **Model 2**: The low average heterogeneity hides 1-2 genuine outliers → Sparse shrinkage is appropriate
3. **Model 3**: The variance paradox reflects sigma misspecification → Measurement error correction is needed

**My prior belief** (based on EDA): Model 1 will likely win, but I'm designing the analysis to **falsify this belief**. If Models 2 or 3 show better predictive performance or resolve modeling issues, I will abandon Model 1.

**Critical mindset**: I expect most schools to pool strongly. If they don't, that's evidence I was wrong about the data generation process, and I should update my beliefs.

**Stopping rule**: I'll stop when:
1. One model clearly wins (ELPD > 2 SE better), OR
2. All models are equivalent (accept uncertainty), OR
3. All models fail (reconsider fundamentally)

**This design prioritizes discovering truth over confirming hypotheses.**

---

## Files and Code

All Stan implementations are included above. To run:

1. Load data: `/workspace/data/data.csv`
2. Compile each model
3. Run MCMC sampling
4. Compare via LOO-CV
5. Report results

PyMC implementations would follow similar structure with equivalent priors.

**Output Directory**: `/workspace/experiments/designer_3/`

---

## References

- Gelman (2006): "Prior distributions for variance parameters in hierarchical models"
- Carvalho et al. (2010): "The horseshoe estimator for sparse signals"
- Rubin (1981): Original Eight Schools paper
- Vehtari et al. (2017): "Practical Bayesian model evaluation using LOO-CV"
- Betancourt (2017): "Hierarchical modeling" (for computational considerations)

**End of Designer 3 Model Proposals**
