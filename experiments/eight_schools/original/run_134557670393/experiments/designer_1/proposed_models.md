# Bayesian Model Proposals: Hierarchical/Multilevel Perspective
## Model Designer #1

**Date**: 2025-10-28
**Focus**: Hierarchical structures, partial pooling, shrinkage strategies
**Dataset**: Meta-analysis with J=8 studies, I²=0%, borderline pooled effect

---

## Executive Summary

I propose **three fundamentally different hierarchical model classes** that make competing assumptions about the data generation process:

1. **Adaptive Hierarchical Model** - Lets the data decide between fixed and random effects
2. **Robust Hierarchical Model** - Assumes heavy-tailed effects to handle potential outliers
3. **Informative Heterogeneity Model** - Incorporates external evidence about between-study variation

**Critical insight**: The I²=0% finding is highly suspicious given the 31-point range in effects. This could mean:
- (a) True homogeneity obscured by large measurement errors [Model 1 favors this]
- (b) Outlier-driven variance inflation [Model 2 addresses this]
- (c) Insufficient power due to J=8 [Model 3 addresses this]

**My falsification mindset**: All three models assume exchangeability of studies. If leave-one-out analysis shows dramatic instability, or if any study has extreme posterior influence (Cook's D-like metric > 1.0), I will REJECT all hierarchical models and pivot to mixture models or outlier detection frameworks.

---

## Model 1: Adaptive Hierarchical Meta-Analysis (Standard Approach)

### 1.1 Model Class
**Bayesian Random-Effects Meta-Analysis** with adaptive shrinkage via half-Cauchy prior on heterogeneity.

### 1.2 Mathematical Specification

**Likelihood**:
```
y_i | theta_i, sigma_i ~ Normal(theta_i, sigma_i^2)   for i = 1, ..., 8
```
Where:
- `y_i` = observed effect size (data)
- `sigma_i` = known standard error (data, NOT estimated)
- `theta_i` = true underlying effect for study i

**Hierarchical Structure**:
```
theta_i | mu, tau ~ Normal(mu, tau^2)
```
Where:
- `mu` = population mean effect (estimand of primary interest)
- `tau` = between-study standard deviation (heterogeneity)

**Priors**:
```
mu ~ Normal(0, 50)                    # Weakly informative
tau ~ Half-Cauchy(0, 5)               # Standard meta-analysis prior
```

**Full Joint Distribution**:
```
p(mu, tau, theta | y, sigma) ∝
  p(mu) * p(tau) *
  ∏[p(y_i | theta_i, sigma_i) * p(theta_i | mu, tau)]
```

### 1.3 Prior Justification

**mu ~ Normal(0, 50)**:
- **Rationale**: Weakly informative, allows observed range (-3 to 28) with low prior influence
- **Reference**: Gelman et al. (2013) "Bayesian Data Analysis" recommend SD = 2-3x expected effect range
- **Calculation**: Observed range = 31, so SD=50 gives 95% prior coverage of [-98, 98]
- **Falsifiable**: If posterior mean for mu is near ±50, prior is constraining (BAD)

**tau ~ Half-Cauchy(0, 5)**:
- **Rationale**: Standard meta-analysis prior (Gelman, 2006; Polson & Scott, 2012)
- **Properties**: Heavy tails allow large tau if data demands; mass near zero if homogeneous
- **Justification**: With J=8 and sigma_i ≈ 12, typical tau in similar meta-analyses is 0-10
- **Reference**: Spiegelhalter et al. (2004) for half-Cauchy in hierarchical models
- **Alternative considered**: Half-Normal(0, 5) - more mass near zero, less flexible

**sigma_i = data (KNOWN)**:
- **Critical**: We treat measurement errors as KNOWN, not estimated
- **Justification**: Standard in meta-analysis when studies report standard errors
- **Assumption**: Reported SEs are accurate (not inflated/deflated by publication bias)

### 1.4 What This Model Captures

**Partial pooling mechanism**:
- When tau ≈ 0: Strong shrinkage toward mu (near-complete pooling, fixed-effect-like)
- When tau > sigma_i: Weak shrinkage (study estimates stay near y_i, independent-like)
- Adaptive: Data determines shrinkage strength via tau posterior

**Patterns captured**:
1. **Measurement error structure**: Precise studies (small sigma_i) get higher weight
2. **Exchangeability**: All studies assumed drawn from same distribution
3. **Uncertainty propagation**: Full posterior for all parameters
4. **Shrinkage**: Study-specific effects theta_i pulled toward pooled mean mu

**Nested models as special cases**:
- tau → 0: Fixed-effect meta-analysis
- tau → ∞: No pooling (independent effects)
- Intermediate tau: Partial pooling (typical outcome)

### 1.5 Falsification Criteria: When to REJECT This Model

**I will abandon Model 1 if**:

1. **Posterior-prior conflict** (Bayesian p-value):
   - If P(tau > 10 | data) > 0.5 AND prior had P(tau > 10) < 0.05
   - **Interpretation**: Data fighting the prior strongly = prior misspecification
   - **Action**: Refit with more diffuse tau prior or switch to Model 3

2. **Extreme shrinkage asymmetry**:
   - If any theta_i posterior mean differs from y_i by > 3*sigma_i
   - **Interpretation**: Model is "correcting" a study excessively = misspecified hierarchy
   - **Action**: Investigate that study as outlier, switch to Model 2

3. **Posterior predictive failure**:
   - If observed y values fall outside 95% posterior predictive intervals for > 1 study
   - **Test**: y_rep ~ Normal(theta_i, sigma_i), compare to observed y
   - **Action**: Likelihood misspecified, switch to Model 2 (robust)

4. **Leave-one-out instability**:
   - If removing any single study changes posterior mean for mu by > 5 units
   - **Calculation**: max_i |E[mu | data_{-i}] - E[mu | data]| > 5
   - **Interpretation**: Model too sensitive to individual studies = hierarchical structure inappropriate
   - **Action**: Abandon hierarchical framework, consider mixture models

5. **Computational pathology**:
   - R-hat > 1.05 after 10K iterations with warmup
   - Divergences > 1% of samples
   - **Interpretation**: Posterior geometry is problematic = model misspecified
   - **Action**: Reparameterize (non-centered if tau near zero) or switch model class

6. **Unidentifiability warning**:
   - If posterior for tau is essentially uniform from 0 to upper prior support
   - **Interpretation**: Data provides NO information about heterogeneity = J too small
   - **Action**: Switch to Model 3 (informative prior) or fixed-effect model

### 1.6 Implementation Notes

**Software**: Stan (CmdStanPy) preferred over PyMC for better handling of hierarchical models with small J

**Parameterization**:
```stan
// Centered parameterization (default)
parameters {
  real mu;
  real<lower=0> tau;
  vector[8] theta;
}
model {
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);  // sigma is data
}
```

**Potential issue - Funnel geometry**: If tau posterior concentrates near zero, may need non-centered parameterization:

```stan
// Non-centered parameterization (if divergences occur)
parameters {
  real mu;
  real<lower=0> tau;
  vector[8] theta_raw;  // Standard normal
}
transformed parameters {
  vector[8] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma);
}
```

**Sampling settings**:
- Chains: 4
- Iterations: 4000 (2000 warmup, 2000 sampling)
- Target acceptance: 0.95 (conservative for small sample)
- Max tree depth: 12

**Diagnostics to monitor**:
- R-hat < 1.01 for all parameters
- ESS > 400 for mu, tau (primary parameters)
- ESS > 200 for theta_i (study-specific parameters)
- Divergences = 0 (strict requirement)
- Energy diagnostic: E-BFMI > 0.3

### 1.7 Expected Challenges

1. **Weak identification of tau**:
   - With J=8, tau posterior will be wide and potentially improper-like
   - **Evidence**: Literature suggests need J≥20 for reliable tau estimation
   - **Mitigation**: Use informative prior (Model 3) or accept wide posterior

2. **Near-zero tau posterior**:
   - Given I²=0%, tau may concentrate at zero, creating funnel geometry
   - **Symptom**: Divergent transitions in centered parameterization
   - **Solution**: Switch to non-centered parameterization

3. **Study 1 dominance**:
   - y_1 = 28 is 3.3 SD above mean, may have outsized influence
   - **Check**: Leave-one-out cross-validation (LOO-CV via ArviZ)
   - **Threshold**: Pareto-k diagnostic > 0.7 indicates problematic influence

4. **Posterior probability near 0.5**:
   - With borderline frequentist p=0.05, expect P(mu > 0 | data) ≈ 0.51-0.53
   - **Not a failure**: Just reflects genuine uncertainty
   - **Report**: Full posterior, don't dichotomize

5. **Prior sensitivity**:
   - Small sample means priors matter more than usual
   - **Mitigation**: Sensitivity analysis with Half-Cauchy(0, 2.5) and (0, 10)

---

## Model 2: Robust Hierarchical Meta-Analysis (Heavy-Tailed)

### 2.1 Model Class
**Robust Bayesian Meta-Analysis** using Student-t distributions to accommodate outliers and heavy-tailed effects.

### 2.2 Mathematical Specification

**Likelihood** (Robust):
```
y_i | theta_i, sigma_i ~ Normal(theta_i, sigma_i^2)   # Measurement errors still normal
```

**Hierarchical Structure** (Heavy-tailed):
```
theta_i | mu, tau, nu ~ Student-t(nu, mu, tau^2)
```
Where:
- `nu` = degrees of freedom (controls tail heaviness)
- `nu → ∞`: Approaches normal (Model 1)
- `nu = 1`: Cauchy distribution (very heavy tails)
- `nu ≈ 3-7`: Typical robust choice

**Priors**:
```
mu ~ Normal(0, 50)                    # Same as Model 1
tau ~ Half-Cauchy(0, 5)               # Same heterogeneity prior
nu ~ Gamma(2, 0.1)                    # Regularizing prior on tail heaviness
```

**Gamma(2, 0.1) justification**:
- Mean = 20, SD = 14.1
- Allows nu from ~5 (robust) to ~50 (near-normal)
- Prevents nu → ∞ (which would just recover Model 1)
- Reference: Juarez & Steel (2010) on robust meta-analysis

### 2.3 Prior Justification

**Why Student-t hierarchy?**:
- **Motivation**: Study 1 (y=28) is 2.7 SD above median, may be outlier
- **Problem with normal**: Outliers inflate tau, shrink all other studies excessively
- **Solution**: Student-t downweights outliers via heavy tails
- **Reference**: Sutton & Abrams (2001) on robust meta-analysis

**nu ~ Gamma(2, 0.1)**:
- **Rationale**: Weakly informative, favors moderate robustness (nu ≈ 10-30)
- **Avoids**: nu < 2 (infinite variance) or nu > 100 (practically normal)
- **Alternative**: nu ~ Gamma(1, 0.1) for heavier tails (if Model 2 fails)

### 2.4 What This Model Captures

**Additional patterns beyond Model 1**:
1. **Outlier accommodation**: Extreme studies get downweighted automatically
2. **Robust shrinkage**: theta_i estimates less influenced by extreme studies
3. **Tail-weight learning**: nu posterior tells us if heavy tails needed

**Interpretation of nu posterior**:
- If nu > 30: Data doesn't need robust model, Model 1 sufficient
- If nu ≈ 3-10: Moderate tail heaviness detected, robustness useful
- If nu ≈ 1-2: Extreme outliers, may need mixture model

### 2.5 Falsification Criteria: When to REJECT This Model

**I will abandon Model 2 if**:

1. **Nu posterior is uninformative**:
   - If posterior for nu spans entire prior range (3 to 50+)
   - **Interpretation**: Data provides no information about tail heaviness = unnecessary complexity
   - **Action**: Revert to Model 1 (simpler)

2. **Nu posterior concentrates at upper limit**:
   - If P(nu > 50 | data) > 0.8
   - **Interpretation**: Data doesn't need heavy tails = Model 1 sufficient
   - **Action**: Abandon Model 2, report Model 1 results

3. **Model doesn't improve fit**:
   - If LOO-CV (ELPD) difference between Model 2 and Model 1 is < 2 (on SE scale)
   - **Standard**: ELPD difference < 4 is "not worth mentioning" (Vehtari et al., 2017)
   - **Action**: Model 2 adds complexity without benefit, use Model 1

4. **Shrinkage becomes too aggressive**:
   - If all theta_i posteriors have SD < 0.1 * sigma_i (over-shrinkage)
   - **Interpretation**: Heavy tails causing near-complete pooling = model malfunction
   - **Action**: Prior on nu too restrictive, refit or abandon

5. **Computational intractability**:
   - Student-t can cause sampling difficulties with small J
   - If ESS < 100 for nu after 10K iterations = REJECT
   - **Action**: Model too complex for data, use Model 1

### 2.6 Implementation Notes

**Software**: Stan (Student-t well-supported)

**Stan code**:
```stan
parameters {
  real mu;
  real<lower=0> tau;
  real<lower=1> nu;  // DOF for Student-t
  vector[8] theta;
}
model {
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  nu ~ gamma(2, 0.1);
  theta ~ student_t(nu, mu, tau);  // Heavy-tailed hierarchy
  y ~ normal(theta, sigma);
}
```

**Potential issue**: Student-t may require more iterations for convergence
- Increase to 6000 iterations (3000 warmup)
- Increase adapt_delta to 0.98

**Diagnostics**: Same as Model 1, plus:
- Check nu posterior for degeneracy at boundaries
- Compare theta_i posteriors to Model 1 (should be less shrunk)

### 2.7 Expected Challenges

1. **Over-parameterization**:
   - With J=8, estimating mu, tau, AND nu may be too ambitious
   - **Check**: If nu posterior equals prior, model is over-parameterized

2. **Weak sensitivity to nu**:
   - Given large sigma_i (9-18), measurement error may dominate
   - Heavy tails in hierarchy may not matter much
   - **Check via simulation**: Generate data with nu=5 vs nu=50, fit both

3. **Longer sampling time**:
   - Student-t likelihood is computationally heavier than normal
   - Expect 2-3x longer run time than Model 1

---

## Model 3: Informative Heterogeneity Meta-Analysis (External Evidence)

### 3.1 Model Class
**Bayesian Meta-Analysis with Informative Prior on Heterogeneity**, incorporating external evidence about between-study variation from similar meta-analyses.

### 3.2 Mathematical Specification

**Likelihood and hierarchy**: Same as Model 1
```
y_i | theta_i, sigma_i ~ Normal(theta_i, sigma_i^2)
theta_i | mu, tau ~ Normal(mu, tau^2)
```

**Priors** (Key difference is tau):
```
mu ~ Normal(0, 50)                    # Same as Model 1
tau ~ Half-Normal(0, 3)               # INFORMATIVE prior (key change)
```

**Rationale for Half-Normal(0, 3)**:
- **Empirical basis**: Turner et al. (2015) analyzed 19,000+ meta-analyses
- **Findings**: Median heterogeneity SD across medical/social science ≈ 0.2 on log-scale
- **Translation**: On natural scale, typical tau is 0-10 for continuous outcomes
- **Prior choice**: Half-Normal(0, 3) gives 95% prior mass on tau ∈ [0, 6]
- **Interpretation**: Incorporates field knowledge that most meta-analyses have moderate heterogeneity

### 3.3 Prior Justification

**Why informative prior on tau?**:
- **Problem**: With J=8, tau is weakly identified (wide posterior under vague prior)
- **Solution**: Borrow strength from external meta-analyses
- **Reference**: Rhodes et al. (2015) "Predictive distributions for between-study heterogeneity"
- **Empirical basis**:
  - Cochrane Database Review (2015): Median I² = 45% across medical meta-analyses
  - Translates to tau ≈ 0.3-5 for typical effect sizes

**Half-Normal(0, 3) specification**:
- **Prior median**: 2.0 (50% probability tau < 2)
- **Prior 95% interval**: [0, 6]
- **Prior mean**: 2.4
- **Rationale**: Conservative middle ground between vague and highly informative

**Sensitivity**: Will test with Half-Normal(0, 5) (more diffuse) and Half-Normal(0, 1.5) (more informative)

**Alternative considered**: Log-Normal(-1, 0.5) - used in some meta-analysis packages, but less interpretable

### 3.4 What This Model Captures

**Advantages over Model 1**:
1. **Reduced posterior uncertainty**: Narrower tau posterior via external information
2. **Borrowing strength**: Incorporates field knowledge about typical heterogeneity
3. **Realistic shrinkage**: More plausible theta_i estimates in small samples

**Trade-off**:
- **Risk**: If this dataset has atypical heterogeneity, informative prior will bias results
- **Check**: Prior-posterior conflict diagnostic

### 3.5 Falsification Criteria: When to REJECT This Model

**I will abandon Model 3 if**:

1. **Prior-data conflict** (KEY CRITERION):
   - If prior predictive p-value < 0.05 OR > 0.95
   - **Test**: P(Q > Q_obs | prior) where Q is Cochran's Q statistic
   - **Interpretation**: Prior is incompatible with observed heterogeneity
   - **Action**: REJECT informative prior, use Model 1

2. **Posterior for tau hits prior boundary**:
   - If P(tau > 5 | data) > 0.2 with Half-Normal(0, 3) prior
   - **Interpretation**: Data wants larger tau than prior allows
   - **Action**: Prior is constraining inference = BAD, refit with vague prior

3. **Prior dominates likelihood**:
   - If posterior SD for tau < 0.8 * prior SD
   - **Calculation**: Prior SD = 3/√(1-2/π) ≈ 3.7, so posterior SD should be < 3.0
   - **Interpretation**: Data hasn't updated prior much = prior too strong OR tau truly near zero
   - **Action**: Check against Model 1; if posteriors very different, REJECT Model 3

4. **External evidence is irrelevant**:
   - If literature review reveals this meta-analysis domain typically has I² > 75%
   - **Interpretation**: Informative prior based on wrong external evidence
   - **Action**: Re-specify prior or abandon

5. **Model comparison failure**:
   - If LOO-CV strongly disfavors Model 3 vs Model 1 (ELPD difference > 4)
   - **Interpretation**: Informative prior hurts predictive performance
   - **Action**: External evidence misleading, use Model 1

### 3.6 Implementation Notes

**Software**: Stan (same as Model 1, just change prior)

**Stan code**:
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[8] theta;
}
model {
  mu ~ normal(0, 50);
  tau ~ normal(0, 3);  // INFORMATIVE (with truncation to positive)
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
}
```

**Prior predictive check** (ESSENTIAL):
```stan
generated quantities {
  real tau_prior = abs(normal_rng(0, 3));
  real mu_prior = normal_rng(0, 50);
  array[8] real y_prior;
  for (i in 1:8) {
    real theta_prior = normal_rng(mu_prior, tau_prior);
    y_prior[i] = normal_rng(theta_prior, sigma[i]);
  }
  real Q_prior = /* compute Q statistic from y_prior */;
}
```

**Critical diagnostic**: Compare Q_prior distribution to observed Q = 4.707
- If Q_obs is in tail of prior predictive (< 5th or > 95th percentile), REJECT prior

### 3.7 Expected Challenges

1. **Prior-data tension**:
   - Observed I²=0% is VERY low compared to typical meta-analyses
   - Informative prior expecting tau > 0 may conflict with data
   - **Prediction**: Prior-posterior conflict likely, may need to REJECT Model 3

2. **False precision**:
   - Informative prior will narrow posteriors artificially if wrong
   - **Check**: Compare posterior intervals to Model 1; if much narrower, investigate

3. **External evidence validity**:
   - Literature on heterogeneity priors is from medical/social science
   - This dataset domain is unknown - may not apply
   - **Mitigation**: Sensitivity analysis with range of prior SDs

---

## Model Comparison and Selection Strategy

### Decision Framework

**Phase 1: Individual Model Assessment**
1. Fit all three models independently
2. Check convergence and diagnostics for each
3. Apply falsification criteria (as specified above)
4. REJECT any model that fails its criteria

**Phase 2: Comparative Evaluation** (if multiple models pass Phase 1)

Use multiple criteria:

| Criterion | Weight | Interpretation |
|-----------|--------|----------------|
| LOO-CV (ELPD) | HIGH | Predictive performance |
| Prior predictive check | HIGH | Prior-data compatibility |
| Posterior predictive check | MEDIUM | Fit quality |
| Scientific plausibility | MEDIUM | Domain knowledge |
| Computational efficiency | LOW | Practical considerations |

**LOO-CV decision rules**:
- ELPD difference < 4: Models equivalent, prefer simpler (Model 1)
- ELPD difference 4-10: Moderate evidence for better model
- ELPD difference > 10: Strong evidence for better model

**Phase 3: Sensitivity Analysis**

For selected model(s):
1. Leave-one-out jackknife (8 fits, each dropping one study)
2. Prior sensitivity (vary prior SDs by ±50%)
3. Likelihood robustness (if Model 1/3 selected, try Student-t)

### Expected Outcome Prediction

**My best guess** (before seeing results):

1. **Model 1 will be primary**: Standard hierarchical model most appropriate
   - I²=0% suggests tau ≈ 0, which Model 1 can capture
   - Falsification criteria unlikely to trigger

2. **Model 2 may be unnecessary**:
   - Study 1 (y=28) is not a statistical outlier given SE=15
   - Heavy tails may not improve fit
   - Prediction: nu posterior will concentrate near 30+, indicating Model 1 sufficient

3. **Model 3 likely to fail**:
   - Observed I²=0% conflicts with typical meta-analysis heterogeneity
   - Informative prior expecting tau > 0 may fight the data
   - Prediction: Prior-data conflict diagnostic will trigger rejection

**However**: I could be completely wrong. If Study 1 is truly problematic, Model 2 may win. If external evidence strongly supports tau ≈ 2-3, Model 3 may win. **Data trumps prediction.**

---

## Stress Tests and Adversarial Checks

### Stress Test 1: Parameter Recovery Simulation

**Goal**: Check if models can recover known parameters

**Design**:
1. Simulate data with known mu=10, tau=0 (fixed-effect, matching I²=0%)
2. Simulate data with known mu=10, tau=5 (random-effects)
3. Fit all three models to both datasets
4. Check: Do posteriors cover true values?

**Failure mode**: If Model 1 can't recover tau=0, or if Model 3 prior biases mu away from truth, REJECT.

### Stress Test 2: Break-Point Analysis

**Goal**: Find dataset modifications that would reject each model

**Tests**:
1. **For Model 1**: Multiply Study 1 effect by 2 (y=56) - should trigger outlier falsification
2. **For Model 2**: Replace all effects with identical values - nu should go to infinity
3. **For Model 3**: Add external studies with large heterogeneity - should create prior-data conflict

### Stress Test 3: Synthetic Heterogeneity

**Goal**: Determine if models can detect heterogeneity if present

**Design**:
1. Generate y_new with same sigma_i but with added tau=5 heterogeneity
2. Fit models to y_new
3. Check: Does tau posterior move away from zero?

**Expected**: Model 1 should detect tau>0; Model 3 may be more sensitive due to informative prior

---

## Red Flags and Escape Routes

### Red Flags That Would Trigger Fundamental Reconsideration

1. **All three models fail falsification criteria**
   - **Interpretation**: Hierarchical framework is wrong
   - **Escape route**: Switch to mixture models (two subpopulations) or meta-regression with latent covariates

2. **Leave-one-out shows Study 1 dominates everything**
   - **Interpretation**: Dataset is too fragile for pooling
   - **Escape route**: Report Study 1 separately, meta-analyze remaining 7

3. **Posterior predictive checks fail for multiple studies**
   - **Interpretation**: Likelihood misspecified (not just Normal)
   - **Escape route**: Non-parametric Bayesian meta-analysis (Dirichlet process)

4. **tau posterior is bimodal**
   - **Interpretation**: Two distinct study populations
   - **Escape route**: Finite mixture of normals (2-component mixture)

5. **Prior sensitivity analysis shows qualitative conclusion changes**
   - **Interpretation**: Data too weak to support inference
   - **Escape route**: Report only posterior for mu, acknowledge tau is unidentified

### Pivot Points: When to Switch Model Classes Entirely

**Abandon hierarchical framework if**:
- Exchangeability assumption violated (study characteristics matter)
- Outliers dominate (need mixture or non-parametric)
- Sample too small even for simplest model (J<5 effective)

**Pivot to**:
- **Mixture models**: If bimodality or clear subgroups
- **Fixed-effect only**: If tau persistently near zero and simpler model sufficient
- **Bayesian model averaging**: If model uncertainty dominates parameter uncertainty

---

## Summary: Core Hypotheses and Decision Points

### Competing Hypotheses

**H1 (Model 1)**: Studies are exchangeable draws from Normal(mu, tau) with tau ≈ 0
- **Evidence for**: I²=0%, no outliers, overlapping CIs
- **Evidence against**: 31-point effect range seems large for homogeneity
- **Decisive test**: If tau posterior 95% CI excludes 0, reject H1

**H2 (Model 2)**: Studies are exchangeable but with heavy-tailed distribution due to outliers
- **Evidence for**: Study 1 is 2.7 SD above median
- **Evidence against**: Study 1 not statistical outlier given large SE
- **Decisive test**: If nu > 30, reject H2 (heavy tails unnecessary)

**H3 (Model 3)**: Studies have moderate heterogeneity (tau ≈ 2-3) typical of field
- **Evidence for**: External meta-analytic literature
- **Evidence against**: Observed I²=0% conflicts with typical heterogeneity
- **Decisive test**: Prior predictive p-value < 0.05, reject H3

### Decision Points for Major Pivots

1. **After initial fits**: Check convergence
   - If all models fail: Problem is computational, not conceptual
   - Action: Simplify priors or reparameterize

2. **After diagnostics**: Check falsification criteria
   - If 2+ models fail: Hierarchical framework wrong
   - Action: Switch to mixture or non-parametric

3. **After LOO-CV**: Compare predictive performance
   - If all models similar: Sample too small for model selection
   - Action: Report all three, use Bayesian model averaging

4. **After sensitivity analysis**: Check robustness
   - If conclusions flip with minor prior changes: Data too weak
   - Action: Report uncertainty, don't make strong claims

### Success Criteria

**Scientific success** (primary):
- Honest quantification of uncertainty about mu and tau
- Clear statement of what data CAN and CANNOT tell us
- Identification of influential studies or limitations

**Technical success** (secondary):
- R-hat < 1.01, ESS > 400
- No divergences, good posterior geometry
- Posterior predictive checks pass

**Philosophical success** (meta):
- Willingness to reject all models if warranted
- Clear documentation of assumptions and failure modes
- Transparency about what would change conclusions

---

## Implementation Priorities

### Must Do (Critical)

1. **Prior predictive checks**: Before fitting, simulate from priors and check plausibility
2. **Falsification criteria**: Hard-code all checks listed above, run automatically
3. **Leave-one-out CV**: Essential for influence analysis with J=8
4. **Convergence diagnostics**: Zero tolerance for divergences or R-hat > 1.01

### Should Do (Important)

1. **Non-centered parameterization**: For Model 1 if divergences occur
2. **Prior sensitivity**: At least 3 variants per model
3. **Posterior predictive checks**: Visual + quantitative (p-values)
4. **Parameter recovery simulation**: Validate before trusting real results

### Nice to Have (If Time)

1. **Bayesian model averaging**: Weight all three models by LOO
2. **Meta-regression**: If study characteristics become available
3. **Publication bias modeling**: Selection models (Copas, p-curve)

---

## Final Meta-Note: On Being Wrong

**I expect to be wrong about**:
- Which model performs best (could be any of the three, or none)
- Whether heterogeneity is truly zero (I²=0% might be artifact)
- Whether Study 1 is influential (might not matter given large SE)
- Whether informative priors help or hurt (depends on external evidence quality)

**I will know I was wrong if**:
- Data strongly rejects my prior assumptions (conflict diagnostics)
- Leave-one-out analysis shows fragility I didn't anticipate
- Posterior predictive checks fail systematically
- Computational issues indicate model misspecification

**I will consider it a success if**:
- We discover early that hierarchical models are inappropriate
- We identify that tau is unidentifiable with J=8 and report that honestly
- We find that conclusions are sensitive to assumptions and state that clearly
- We pivot to simpler or different models based on evidence

**The goal is truth, not task completion.** If all three models fail, that's valuable information about the limits of what this dataset can tell us.

---

**End of Proposal**

**Files**: `/workspace/experiments/designer_1/proposed_models.md`

**Next steps**:
1. Await proposals from other designers (if parallel)
2. Synthesize competing proposals
3. Implement models in Stan/PyMC
4. Execute falsification checks
5. Be prepared to abandon everything if evidence demands it
