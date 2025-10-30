# Model Designer 3: Critical Adversarial Proposals
## Challenging the "Complete Pooling" Consensus

**Designer**: Model Designer 3 (Adversarial/Edge Case Specialist)
**Date**: 2025-10-28
**Philosophy**: Assume the EDA is misleading. Design models to BREAK the consensus.

---

## Executive Summary: Why the EDA Might Be Wrong

The EDA confidently concludes:
- Between-group variance = 0 (complete pooling recommended)
- Population mean ≈ 10 (positive and significant)
- Measurement error dominates (SNR ≈ 1)

**My job: Find models that could reveal these conclusions are artifacts.**

### Critical Questions the EDA Didn't Answer
1. What if measurement error is UNDERESTIMATED? (sigma values may be wrong)
2. What if there's a latent subgroup structure? (2 clusters, not 8 homogeneous groups)
3. What if the "no between-group variance" is an artifact of tiny sample size?
4. What if one extreme observation (Group 4: y = -4.88) is masking heterogeneity?
5. What if the measurement process itself is informative? (sigma correlates with true value)

---

## Model Suite: Three Competing Hypotheses

I propose three fundamentally different model classes, each designed to challenge the EDA's conclusions.

---

# Model 1: Misspecified Measurement Error Model
## "What if the reported sigmas are wrong?"

### Hypothesis
The known measurement errors (sigma) may be systematically biased or underestimated. Perhaps:
- Measurement devices report nominal precision, not actual precision
- Sigma values are rounded or truncated
- True measurement error has additional unknown components

### Model Specification

```
# Measurement model with inflation factor
y_i ~ Normal(theta_i, sigma_i * lambda)

# Group means (test between-group variation)
theta_i ~ Normal(mu, tau)

# Population parameters
mu ~ Normal(0, 30)                    # Vague, centered at zero
tau ~ Half-Cauchy(0, 10)              # Allow substantial variation

# KEY: Measurement error inflation
lambda ~ Uniform(0.5, 3.0)            # Test if sigma is wrong by factor of 0.5-3x
```

### Key Innovation
**The inflation factor lambda** tests whether reported measurement errors are correct:
- lambda ≈ 1.0: Reported errors are accurate (EDA is right)
- lambda > 1.5: Measurement errors are UNDERESTIMATED (tau might be non-zero)
- lambda < 0.8: Measurement errors are OVERESTIMATED (even more complete pooling)

### Why This Model Could Succeed
1. **Directly tests the measurement error assumption** - EDA took sigma as ground truth
2. **Could explain away "zero variance"** - If true error is 2x larger, between-group variance might appear
3. **Falsifiable** - If posterior for lambda concentrates near 1.0, measurement errors are valid

### Falsification Criteria (Abandon if:)
- **Primary**: Posterior for lambda has 95% CI entirely within [0.9, 1.1] → measurement errors are accurate
- **Secondary**: Posterior predictive checks show model UNDERPERFORMS complete pooling
- **Tertiary**: Computational issues (MCMC doesn't converge after 10,000 iterations)

### Expected Results if EDA is Right
- lambda posterior: centered at 1.0 (0.8, 1.2)
- tau posterior: still near 0
- Effective reduction to complete pooling

### Expected Results if EDA is Wrong
- lambda posterior: shifted to 1.5-2.5
- tau posterior: non-negligible (>5)
- Group means show more separation

### Computational Considerations
**Challenges:**
- Identifiability issue: lambda and tau are confounded
- Need strong priors to separate inflation from true heterogeneity
- May require non-centered parameterization

**MCMC Strategy:**
```stan
parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0.5, upper=3.0> lambda;
  vector[8] theta_raw;  // Non-centered
}
transformed parameters {
  vector[8] theta = mu + tau * theta_raw;
}
model {
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma .* lambda);
  ...
}
```

**Diagnostics to watch:**
- High correlation between lambda and tau (>0.8) indicates identifiability problems
- Divergent transitions suggest model is too flexible
- Effective sample size < 100 for lambda suggests weak identification

### Stress Test
**Synthetic data generation:** Create data where lambda = 2.0 (errors are 2x larger). Can model recover this?

---

# Model 2: Latent Mixture Model
## "What if there are hidden subgroups?"

### Hypothesis
The EDA assumes all 8 groups come from the same population. But what if there are 2-3 latent clusters?

**Evidence from EDA:**
- Groups 0-3: High SNR, positive values (15-26)
- Groups 4-7: Low SNR, mixed/low values (-5 to 8.5)

This could indicate:
- Different measurement regimes (high-quality vs low-quality)
- Different true processes (positive vs zero-centered)
- Survivorship bias (only large true values produce high SNR)

### Model Specification

```
# Mixture model with K=2 components
y_i ~ Normal(theta_i, sigma_i)

# Group means come from one of K clusters
theta_i ~ Normal(mu[z_i], tau[z_i])

# Cluster assignment (latent)
z_i ~ Categorical(pi)

# Cluster-specific parameters
mu[1] ~ Normal(0, 10)      # Cluster 1: maybe zero-centered
mu[2] ~ Normal(20, 10)     # Cluster 2: maybe positive
tau[1] ~ Half-Cauchy(0, 5)
tau[2] ~ Half-Cauchy(0, 5)

# Mixing proportions
pi ~ Dirichlet([1, 1])     # Uniform prior on mixing
```

### Key Innovation
**Allows for heterogeneity the EDA couldn't see** because it assumed homogeneity. Tests whether:
- High-SNR groups are systematically different from low-SNR groups
- There's a "low-quality measurement" cluster vs "high-quality" cluster
- Negative observation (Group 4) belongs to different cluster

### Why This Model Could Succeed
1. **Explains the bimodal pattern** - EDA noted possible bimodality but dismissed it
2. **Accounts for SNR divide** - 4 good groups vs 4 poor groups might be real clustering
3. **More scientifically plausible** - Often real data has subpopulations

### Falsification Criteria (Abandon if:)
- **Primary**: Posterior for pi concentrates at [0.5, 0.5] with wide uncertainty → no clear clustering
- **Secondary**: Cluster assignments are unstable (group assignments flip across MCMC samples)
- **Tertiary**: LOO-CV/WAIC prefers complete pooling by >4 units
- **Quaternary**: Posterior predictive checks show NO improvement over complete pooling

### Expected Results if EDA is Right
- Mixture proportions: pi ≈ [0.5, 0.5] with high uncertainty
- Cluster means collapse: mu[1] ≈ mu[2] ≈ 10
- Model effectively reduces to complete pooling
- WAIC penalizes complexity

### Expected Results if EDA is Wrong
- Clear cluster separation: pi ≈ [0.5, 0.5] with LOW uncertainty
- Distinct cluster means: e.g., mu[1] ≈ 5, mu[2] ≈ 20
- Groups 0-3 assigned to one cluster, 4-7 to another
- WAIC prefers mixture model

### Computational Considerations
**Challenges:**
- Label switching: Clusters can swap identities across iterations
- Small sample size (n=8): Hard to infer 2 means, 2 variances, 1 mixing proportion (5 parameters!)
- May overfit

**MCMC Strategy:**
```stan
// Use ordered means to prevent label switching
parameters {
  ordered[2] mu;  // Force mu[1] < mu[2]
  ...
}
```

**Or post-hoc label switching correction** using largest cluster assignment.

**Diagnostics:**
- R-hat > 1.01 for cluster parameters indicates label switching
- Effective sample size << 1000 for z_i suggests uncertainty in assignments
- Trace plots for mu showing bimodal distribution = label switching

### Stress Test
**Posterior predictive check**:
- Generate fake data from complete pooling (H0)
- Fit mixture model to fake data
- Does it spuriously detect clusters? (Type I error rate)

**Critical**: If mixture model finds clusters in data generated from complete pooling, the model is too flexible and should be abandoned.

---

# Model 3: Informative Measurement Error Process
## "What if sigma is correlated with the true value?"

### Hypothesis
The EDA tested correlation between y (observed) and sigma, finding r=0.43 (p=0.39). But this is underpowered.

**What if measurement error is actually LARGER for larger true values?**
- Harder to measure large quantities precisely
- Measurement scale nonlinearities
- Detector saturation effects

This would violate the "known heteroscedastic error" assumption entirely.

### Model Specification

```
# Functional relationship between measurement error and true value
y_i ~ Normal(theta_i, sigma_i * exp(alpha * theta_i))

# Group means
theta_i ~ Normal(mu, tau)

# Population parameters
mu ~ Normal(10, 20)
tau ~ Half-Cauchy(0, 10)

# KEY: Does measurement error scale with true value?
alpha ~ Normal(0, 0.1)   # Small effect, but testable
                         # alpha > 0: error increases with theta
                         # alpha < 0: error decreases with theta
                         # alpha ≈ 0: error independent (EDA is right)
```

### Alternative: Linear Scaling
```
y_i ~ Normal(theta_i, sqrt(sigma_i^2 + (beta * theta_i)^2))
beta ~ Normal(0, 0.5)
```

### Key Innovation
**Tests a fundamental assumption**: that measurement error is independent of the thing being measured.

### Why This Model Could Succeed
1. **Directly tests an untested assumption** - EDA correlated y with sigma, not theta with sigma
2. **Common in real measurement systems** - Many detectors have proportional error
3. **Could explain the SNR pattern** - High values have higher sigma in our data (Groups 0-3)

### Falsification Criteria (Abandon if:)
- **Primary**: 95% CI for alpha includes 0 with high probability (>80% posterior mass in [-0.05, 0.05])
- **Secondary**: Model fit (WAIC) is WORSE than constant-error model
- **Tertiary**: Posterior predictive checks show systematic bias
- **Quaternary**: Non-identifiability (posterior is just the prior)

### Expected Results if EDA is Right
- alpha posterior: centered at 0 (-0.02, 0.02)
- tau posterior: still near 0
- No improvement over complete pooling

### Expected Results if EDA is Wrong
- alpha posterior: clearly positive (e.g., 0.05 to 0.15)
- Measurement error model changes substantially
- Better predictions for new data

### Computational Considerations
**Challenges:**
- Highly nonlinear likelihood
- Potential for alpha and tau confounding
- May require informative priors on alpha to avoid extreme values

**MCMC Strategy:**
```stan
// Bound alpha to prevent numerical issues
parameters {
  real<lower=-0.5, upper=0.5> alpha;
  ...
}
```

**Diagnostics:**
- Divergent transitions indicate numerical instability in exp(alpha * theta)
- Check that sigma_effective = sigma * exp(alpha * theta) stays positive and reasonable
- Posterior predictive: Does model generate realistic sigma patterns?

### Stress Test
**Simulate data with known alpha = 0.1**, then fit model. Can it recover the true alpha?

---

# Model Comparison Strategy

## Falsification Hierarchy

Test models in this order:

### Stage 1: Test Measurement Error Assumptions (Model 1 & 3)
**Question**: Are the reported measurement errors correct and independent?

**If lambda ≈ 1.0 AND alpha ≈ 0:**
→ Measurement errors are well-specified
→ Proceed to Stage 2

**If lambda ≠ 1.0 OR alpha ≠ 0:**
→ STOP. Re-evaluate the entire measurement model
→ Cannot trust any downstream inferences
→ Need to understand measurement process better

### Stage 2: Test Population Structure (Model 2)
**Question**: Is there latent heterogeneity?

**If mixture model reduces to complete pooling:**
→ EDA conclusion is confirmed
→ Accept complete pooling

**If mixture model finds clear clusters:**
→ EDA was wrong (misled by assuming homogeneity)
→ Investigate subgroup structure

### Stage 3: Final Model Selection
Use all evidence to select best model:
- LOO-CV / WAIC for predictive accuracy
- Posterior predictive checks for model adequacy
- Scientific plausibility
- Parsimony (simplest model that fits)

---

# Computational Implementation Plan

## Software Stack
- **Primary**: Stan (via CmdStanPy or PyStan)
- **Alternative**: PyMC for prototyping (faster iteration)
- **Baselines**: Closed-form solutions where possible (complete pooling)

## MCMC Settings

### Model 1 (Inflation factor)
```python
# Stan configuration
chains = 4
iter_warmup = 2000  # Extra warmup due to identifiability
iter_sampling = 2000
adapt_delta = 0.95  # High to avoid divergences
max_treedepth = 12
```

### Model 2 (Mixture)
```python
# Stan configuration
chains = 8  # More chains to detect label switching
iter_warmup = 3000  # Mixture models need more warmup
iter_sampling = 2000
adapt_delta = 0.90
```

### Model 3 (Functional error)
```python
# Stan configuration
chains = 4
iter_warmup = 1500
iter_sampling = 2000
adapt_delta = 0.95  # Nonlinear likelihood
max_treedepth = 12
```

## Convergence Diagnostics

For ALL models, check:
1. **R-hat < 1.01** for all parameters
2. **ESS_bulk > 400** and **ESS_tail > 400**
3. **No divergent transitions** (if >1% divergences, increase adapt_delta)
4. **Trace plots**: Good mixing, no trends
5. **Pairs plots**: Check for funnel geometries or strong correlations

## Model Checking Protocol

### 1. Prior Predictive Checks
Before fitting, simulate from prior:
- Do simulated y values span reasonable range?
- Does model allow impossible values?
- Are priors too restrictive or too vague?

### 2. Posterior Predictive Checks
After fitting, simulate from posterior:
- Can model reproduce observed variance?
- Does model capture the SNR pattern?
- Are extreme observations (Group 4) covered by predictive distribution?

### 3. Leave-One-Out Cross-Validation
- Compute LOO for each model
- Check Pareto k diagnostics (all k < 0.7?)
- Compare ELPD differences (>4 = meaningful)

---

# Red Flags and Pivot Criteria

## When to Abandon Model 1 (Inflation)

**Red Flag 1**: Lambda posterior = prior
- **Diagnosis**: Model is non-identifiable
- **Action**: Try stronger prior or simplify model

**Red Flag 2**: Posterior correlation(lambda, tau) > 0.9
- **Diagnosis**: Cannot separate error inflation from true heterogeneity
- **Action**: Fix lambda = 1.0 and proceed to standard hierarchical model

**Red Flag 3**: Posterior predictive shows systematic bias
- **Diagnosis**: Model is misspecified in a different way
- **Action**: Try additive error component instead of multiplicative

## When to Abandon Model 2 (Mixture)

**Red Flag 1**: Label switching (R-hat > 1.05)
- **Diagnosis**: Non-identifiable clusters
- **Action**: Try ordered constraint or K-means initialization

**Red Flag 2**: Cluster assignments unstable
- **Diagnosis**: Data don't support clustering
- **Action**: Abandon mixture, accept complete pooling

**Red Flag 3**: WAIC prefers complete pooling by >10 units
- **Diagnosis**: Mixture is overfitting
- **Action**: Accept complete pooling

## When to Abandon Model 3 (Functional Error)

**Red Flag 1**: Alpha posterior = prior
- **Diagnosis**: No signal for functional relationship
- **Action**: Accept independent error assumption

**Red Flag 2**: Numerical instability (divergences)
- **Diagnosis**: Exponential error model is problematic
- **Action**: Try linear scaling instead

**Red Flag 3**: Worse predictive accuracy than constant error
- **Diagnosis**: Added complexity hurts
- **Action**: Accept simpler model

---

# Expected Outcomes and Interpretations

## Scenario A: EDA is Completely Right
**Evidence:**
- Model 1: lambda ≈ 1.0 (CrI: 0.9-1.1)
- Model 2: No clear clusters, mu[1] ≈ mu[2]
- Model 3: alpha ≈ 0 (CrI: -0.02, 0.02)
- All models have tau ≈ 0

**Interpretation**: Measurement errors are well-specified, groups are homogeneous, complete pooling is correct.

**Recommendation**: Use complete pooling model, report that adversarial models found no evidence against it.

## Scenario B: Measurement Errors are Wrong
**Evidence:**
- Model 1: lambda ≈ 2.0 (CrI: 1.5-2.5)
- Model 1: tau becomes non-negligible once errors are inflated

**Interpretation**: Reported sigmas underestimate true measurement error. Between-group variance might exist but is masked.

**Recommendation**: STOP. Investigate measurement process. Cannot trust current data without understanding error structure.

## Scenario C: Latent Clusters Exist
**Evidence:**
- Model 2: Clear cluster separation (mu[1] ≈ 5, mu[2] ≈ 20)
- Model 2: Stable cluster assignments
- Model 2: WAIC prefers mixture by >4 units

**Interpretation**: EDA missed subgroup structure by assuming homogeneity. Two distinct populations.

**Recommendation**: Investigate WHY clusters exist. Is it measurement regime? Different processes? Revisit data collection.

## Scenario D: Measurement Error Scales with True Value
**Evidence:**
- Model 3: alpha clearly positive (CrI: 0.05-0.15)
- Better predictive accuracy than constant error

**Interpretation**: Measurement system has proportional error component. Standard model is misspecified.

**Recommendation**: Revise measurement model to account for heteroscedasticity. May need generalized error structure.

---

# Why These Models Challenge the EDA

## EDA Assumption 1: "Measurement errors are known"
**Challenged by**: Model 1 (Inflation factor)
**Why**: Known errors might be systematically biased. Testing this assumption is critical.

## EDA Assumption 2: "Between-group variance is zero"
**Challenged by**: Model 2 (Mixture)
**Why**: Sample size n=8 is tiny. Could have latent structure EDA couldn't detect.

## EDA Assumption 3: "Measurement error is independent of true value"
**Challenged by**: Model 3 (Functional error)
**Why**: EDA tested correlation with OBSERVED values, not TRUE values. Important distinction.

## EDA Assumption 4: "All groups are from same population"
**Challenged by**: Model 2 (Mixture)
**Why**: Bimodal pattern + SNR divide suggests possible clustering. Worth testing explicitly.

## EDA Assumption 5: "Complete pooling is obviously correct"
**Challenged by**: All models
**Why**: With n=8, many models could explain the data. Need to test alternatives, not assume one is right.

---

# Critical Success Criteria

## This model design succeeds if:
1. **At least one model finds evidence against complete pooling** → EDA was overconfident
2. **All models converge and produce reliable inference** → Methods are sound
3. **Model comparison clearly distinguishes alternatives** → We learn which hypothesis is best
4. **Falsification criteria are met** → We know WHEN to abandon each model

## This model design fails if:
1. **All models confirm complete pooling** → No new information, but confirms EDA
2. **Computational issues prevent inference** → Models are too complex for this data
3. **Models give contradictory results** → Cannot distinguish alternatives
4. **We cannot falsify any model** → Models are not scientifically useful

---

# Philosophical Stance

## Adversarial Modeling Principles

1. **Assume the EDA is wrong** - Start with skepticism, not confirmation
2. **Design models to BREAK the consensus** - If they fail to break it, consensus is stronger
3. **Test hidden assumptions** - EDA made many implicit assumptions
4. **Embrace complexity** - Simple models are elegant, but might miss truth
5. **Falsification over confirmation** - Know when to abandon each model

## Why This Approach Matters

With n=8 observations:
- Statistical power is LOW
- Many models can explain the data
- EDA's confidence might be misplaced
- Need explicit tests of alternatives

**If all adversarial models fail to break complete pooling, we have MUCH stronger evidence for it than EDA alone provided.**

**If any adversarial model succeeds, we've discovered something EDA missed.**

---

# Implementation Roadmap

## Week 1: Model Development
- Day 1-2: Implement Model 1 (Inflation) in Stan
- Day 3-4: Implement Model 2 (Mixture) in Stan
- Day 5: Implement Model 3 (Functional error) in Stan

## Week 2: Fitting and Diagnostics
- Day 1-2: Fit all models, check convergence
- Day 3-4: Posterior predictive checks
- Day 5: Model comparison (LOO/WAIC)

## Week 3: Stress Tests
- Day 1-2: Synthetic data validation
- Day 3-4: Sensitivity analyses
- Day 5: Finalize conclusions

---

# Summary Table: Models at a Glance

| Model | Key Parameter | Falsification Threshold | Computational Challenge | Scientific Question |
|-------|--------------|------------------------|------------------------|-------------------|
| **Model 1**: Inflation | lambda | 95% CrI ⊂ [0.9, 1.1] | Identifiability with tau | Are reported errors accurate? |
| **Model 2**: Mixture | pi, mu[1], mu[2] | Unstable assignments OR WAIC penalty >4 | Label switching | Are there hidden subgroups? |
| **Model 3**: Functional Error | alpha | 95% CrI ⊂ [-0.05, 0.05] | Nonlinear likelihood | Does error scale with true value? |
| **Baseline**: Complete Pooling | mu | N/A | None | Is the EDA right? |

---

# Final Recommendation

**Start with Model 1** (Inflation factor) because:
1. It tests the most fundamental assumption (measurement error validity)
2. If lambda ≠ 1, all other models are built on flawed foundation
3. Relatively simple to implement and diagnose

**Then Model 3** (Functional error) because:
4. Also tests measurement model
5. Can be fit independently of Model 1
6. Provides complementary information

**Finally Model 2** (Mixture) because:
7. Most complex (label switching, multiple parameters)
8. Only interpretable if measurement models are valid
9. Could still find subgroups if Models 1 & 3 fail

**Throughout**: Compare everything to complete pooling baseline. The adversarial models must demonstrate clear improvement to be taken seriously.

---

# Deliverables

1. **Stan code** for all three models (with detailed comments)
2. **Convergence diagnostics** for each fit
3. **Posterior summaries** with 95% credible intervals
4. **Model comparison table** (LOO, WAIC, posterior predictive accuracy)
5. **Falsification report** - which models were abandoned and why
6. **Final recommendation** - best model for this data

---

**End of Adversarial Model Design**

**Remember**: Our goal is not to confirm complete pooling. Our goal is to TRY to break it. If we fail, complete pooling is even stronger. If we succeed, we've learned something important the EDA missed.
