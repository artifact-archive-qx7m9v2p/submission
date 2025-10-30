# Practical Bayesian Modeling Strategy for Overdispersed Binomial Data
## Designer #3: Practical Considerations and Model Selection

**Date**: 2025-10-30
**Perspective**: Skeptical Practitioner - "Will it actually work?"
**Focus**: Computational efficiency, interpretability, robustness, predictive performance

---

## Executive Summary

**Bottom Line**: Start with the simplest model that respects the data structure, validate ruthlessly, and only add complexity if you can demonstrate clear benefits.

**Recommended Strategy**:
1. **Start here**: Hierarchical binomial (non-centered) - best balance of simplicity and correctness
2. **Validate against**: Beta-binomial (marginal model) - simpler, often good enough
3. **Escalate only if needed**: Robust hierarchical (Student-t) - if outliers break standard model

**Decision Rule**: Choose the model with acceptable diagnostics AND lowest complexity. "Good enough" beats "theoretically optimal but won't converge."

---

## The Practical Reality Check

### What We Know from EDA
- 12 groups, 2,814 trials, 196 successes (6.97% overall)
- Strong overdispersion (φ=3.59, p<0.0001) - **this is NOT noise**
- Wide rate variation (3.1% to 14.0%) - **4.5-fold difference**
- Group 4 dominates (810/2814 = 29% of data)
- Three outlier groups (2, 4, 8)
- Groups are exchangeable (no covariates)

### What This Means Practically
- ❌ **Can't use pooled model** - empirically falsified
- ❌ **Can't use unpooled model** - small groups too unstable
- ✅ **Must use partial pooling** - only question is HOW
- ⚠️ **Outliers will matter** - Groups 2, 4, 8 will drive results
- ⚠️ **Computation matters** - 12 groups is small, but we need this to work

---

## Model Proposals (Ranked by Practicality)

# Model 1: Hierarchical Binomial (Non-Centered) ⭐ RECOMMENDED START

## Model Specification

### Mathematical Notation
```
Level 1 (Data):
  rⱼ ~ Binomial(nⱼ, pⱼ)    for j = 1,...,12

Level 2 (Groups):
  logit(pⱼ) = μ + τ·ηⱼ
  ηⱼ ~ Normal(0, 1)         [Non-centered parameterization]

Level 3 (Hyperpriors):
  μ ~ Normal(-2.5, 1)        [Weakly informative, implies ~8% success rate]
  τ ~ Half-Cauchy(0, 1)      [Standard weakly informative for group SD]
```

### Stan Code (Production-Ready)
```stan
data {
  int<lower=1> J;              // Number of groups (12)
  array[J] int<lower=0> n;     // Trials per group
  array[J] int<lower=0> r;     // Successes per group
}

parameters {
  real mu;                      // Population mean (logit scale)
  real<lower=0> tau;            // Between-group SD (logit scale)
  vector[J] eta;                // Non-centered group effects
}

transformed parameters {
  vector[J] theta = mu + tau * eta;  // Group-level logit rates
  vector[J] p = inv_logit(theta);    // Success probabilities
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);           // Half-Cauchy via lower=0 constraint
  eta ~ std_normal();           // Standard normal for non-centered

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Posterior predictive checks
  array[J] int r_rep;
  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // LOO pointwise log-likelihood
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }

  // Derived quantities
  real pooled_p = inv_logit(mu);  // Population mean rate
  real tau_p = tau;                // Between-group SD (for monitoring)
}
```

### PyMC Code (Alternative)
```python
import pymc as pm
import numpy as np

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.5, sigma=1)
    tau = pm.HalfCauchy('tau', beta=1)

    # Non-centered parameterization
    eta = pm.Normal('eta', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * eta)
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Likelihood
    r_obs = pm.Binomial('r_obs', n=n, p=p, observed=r)

    # Posterior predictive for checks
    r_pred = pm.Binomial('r_pred', n=n, p=p)
```

## Practical Advantages

1. **Computational Efficiency** ⚡
   - Non-centered parameterization: 2-10x faster convergence than centered
   - Only 14 parameters (μ, τ, 12 ηⱼ) - fits in <1 minute on laptop
   - Well-conditioned: τ and ηⱼ are a posteriori independent

2. **Interpretability** 📊
   - "Each group has its own rate, but we learn from all groups"
   - μ (population mean) is what most people care about
   - τ (variation between groups) quantifies heterogeneity
   - Shrinkage is automatic and intuitive

3. **Robustness** 🛡️
   - Handles small samples well (Groups 1, 10 with n<100)
   - Automatically down-weights extreme observations
   - No manual outlier removal needed
   - Half-Cauchy prior regularizes τ away from extreme values

4. **Predictive Performance** 🎯
   - Partial pooling optimal for prediction
   - LOO cross-validation directly evaluable
   - Natural prediction for new (unseen) group: draw from Normal(μ, τ)

5. **Standard Model** 📚
   - Textbook example (Gelman et al. BDA3, McElreath Statistical Rethinking)
   - Most practitioners will understand it
   - Extensive debugging resources available

## Practical Disadvantages

1. **Prior Sensitivity** ⚠️
   - τ prior matters with only 12 groups
   - μ prior can influence small groups
   - Need to check prior predictive distributions

2. **Outlier Vulnerability** ⚠️
   - Groups 2, 4, 8 may pull μ and τ
   - Normal distribution has light tails
   - May underestimate uncertainty if true distribution is heavy-tailed

3. **Assumption Rigidity** ⚠️
   - Assumes groups are exchangeable (true in our case)
   - Assumes symmetric heterogeneity (may not be true)
   - Logit scale assumes symmetry in uncertainty

4. **Intermediate Complexity** ⚠️
   - More complex than beta-binomial
   - Non-centered parameterization confusing for beginners
   - Harder to explain than "just pool everything"

## Computational Profile

**Expected Performance** (4 chains, 2000 iterations, 1000 warmup):
- **Sampling time**: 30-60 seconds on modern laptop (M1/i7)
- **Convergence**: Rhat < 1.01 expected for all parameters
- **ESS**: >400 expected for μ, τ (bulk), >200 for tails
- **Memory**: <50 MB
- **Pareto k (LOO)**: <0.5 expected for most groups, possibly 0.5-0.7 for outliers

**Known Issues**:
- **Divergences**: <1% expected with default adapt_delta=0.8
  - If divergences occur: increase to 0.95 (2x slower, but reliable)
- **Max treedepth**: Rarely hit with 12 groups
- **ESS too low**: Possible for τ if groups are very similar (τ near 0)
  - Solution: Run longer or use tighter prior

**Scaling**:
- Tested on problems with 100+ groups
- Complexity: O(J) where J = number of groups
- Bottleneck: Matrix operations (minimal with J=12)

## Interpretability

### For Statisticians
- "Standard random effects logistic regression"
- "Partial pooling on logit scale"
- "Empirical Bayes estimator in fully Bayesian framework"

### For Non-Statisticians
**Simple Explanation**:
> "Each group has its own success rate, but we assume all groups are related. Small groups borrow information from large groups. The model automatically decides how much to trust each group's data versus the overall pattern."

**Key Outputs to Report**:
1. **μ (population mean)**: "The typical success rate across all groups"
2. **τ (between-group SD)**: "How much groups differ from each other"
3. **Shrinkage plot**: "How much we adjusted each group toward the average"
4. **Predictive interval for new group**: "What we'd expect for a future group"

### Visualization Strategy
1. **Caterpillar plot**: Raw vs posterior rates for each group
2. **Shrinkage arrows**: Show pooling in action
3. **Posterior predictive**: Overlay observed vs predicted distributions
4. **LOO-PIT**: Calibration plot (advanced audiences)

## Falsification Criteria

### Red Flags: ABANDON this model if...

1. **Computational**
   - ❌ Divergent transitions >5% (even with adapt_delta=0.99)
   - ❌ Rhat >1.05 for any parameter after 4000 iterations
   - ❌ ESS <100 for μ or τ
   - **Action**: Switch to Model 2 (Beta-binomial) or Model 3 (Robust)

2. **Prior-Posterior Conflict**
   - ❌ Posterior τ consistently hits upper bound of prior support
   - ❌ Posterior μ >2 SD away from prior mean
   - **Action**: Re-examine priors or consider model misspecification

3. **Predictive Failure**
   - ❌ Posterior predictive p-value <0.01 or >0.99
   - ❌ Systematic bias in posterior predictions (all too high or too low)
   - ❌ Multiple groups with Pareto k >0.7
   - **Action**: Switch to Model 3 (Robust hierarchical)

4. **Shrinkage Pathology**
   - ❌ Large groups (n>500) shrink >40%
   - ❌ Small groups (n<50) shrink <10%
   - ❌ Extreme groups (2, 4, 8) shrink more than moderate groups
   - **Action**: Check for data errors or model misspecification

5. **Parameter Estimates**
   - ❌ τ >2 (implies 95% of groups span 50-fold odds ratios - unlikely)
   - ❌ Any group p >0.5 (implies >50% success rate - implausible)
   - ❌ Effective number of parameters >J (overfitting)
   - **Action**: Revisit priors or consider structural problem

### Green Flags: ACCEPT this model if...

1. **Computational** ✅
   - All Rhat <1.01
   - ESS >400 for key parameters
   - Divergences <1%
   - Sampling time <2 minutes

2. **Predictive Performance** ✅
   - LOO outperforms pooled and unpooled by ΔLOO >10
   - All Pareto k <0.7
   - Posterior predictive checks pass (p-value 0.05-0.95)
   - Coverage: ~95% of groups within 95% credible intervals

3. **Scientific Plausibility** ✅
   - τ in range [0.2, 1.5] (moderate heterogeneity)
   - μ implies pooled rate 4-12% (consistent with data)
   - Shrinkage follows expected pattern (small→large)
   - Outliers identified but not excluded

4. **Practical Utility** ✅
   - Results stable across different random seeds
   - Predictions reasonable for new groups
   - Stakeholders understand the outputs

---

# Model 2: Beta-Binomial (Marginal Model) 🔧 PRAGMATIC ALTERNATIVE

## Model Specification

### Mathematical Notation
```
Level 1 (Marginal):
  rⱼ ~ BetaBinomial(nⱼ, α, β)

Parameterization:
  α = μ·φ
  β = (1-μ)·φ

Hyperpriors:
  μ ~ Beta(5, 50)            [Implies ~9% mean rate]
  φ ~ Gamma(1, 0.1)          [Weakly informative concentration]
```

**Key Difference**: No group-level parameters! We only estimate population-level α and β.

### Stan Code
```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}

parameters {
  real<lower=0, upper=1> mu;      // Mean success rate
  real<lower=0> phi;               // Concentration parameter
}

transformed parameters {
  real alpha = mu * phi;
  real beta = (1 - mu) * phi;
}

model {
  // Priors
  mu ~ beta(5, 50);                // Weakly informative
  phi ~ gamma(1, 0.1);             // Allows overdispersion

  // Likelihood
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}

generated quantities {
  array[J] int r_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    r_rep[j] = beta_binomial_rng(n[j], alpha, beta);
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha, beta);
  }

  // Implied variance ratio
  real var_ratio = 1 + (1 / (phi + 1));  // Overdispersion factor
}
```

### PyMC Code
```python
with pm.Model() as beta_binomial_model:
    # Priors
    mu = pm.Beta('mu', alpha=5, beta=50)
    phi = pm.Gamma('phi', alpha=1, beta=0.1)

    # Beta-binomial parameters
    alpha = pm.Deterministic('alpha', mu * phi)
    beta = pm.Deterministic('beta', (1 - mu) * phi)

    # Likelihood
    r_obs = pm.BetaBinomial('r_obs', n=n, alpha=alpha, beta=beta, observed=r)
```

## Practical Advantages

1. **Ultimate Simplicity** 🏆
   - Only 2 parameters (μ, φ) vs 14 for hierarchical
   - 5-10x faster sampling
   - No parameterization tricks needed
   - Fits in <10 seconds

2. **Robust to Small Samples** 💪
   - Doesn't try to estimate individual group rates
   - No shrinkage artifacts
   - Groups 1 and 10 (small n) are not problematic

3. **Minimal Prior Sensitivity** 🎯
   - Only 2 priors to tune
   - μ prior is on natural probability scale (easier to specify)
   - φ prior matters less than τ prior in hierarchical model

4. **Easy Interpretation** 📖
   - "Groups are draws from a common distribution"
   - μ = "average success rate"
   - φ = "how consistent groups are" (higher φ = more similar)
   - No need to explain shrinkage or partial pooling

5. **Often Good Enough** ✅
   - For prediction of population mean: often indistinguishable from hierarchical
   - For quantifying heterogeneity: φ captures it directly
   - For stakeholder communication: simpler is better

## Practical Disadvantages

1. **No Group-Level Inference** ❌
   - Can't estimate individual group success rates
   - Can't answer "Which group is best?"
   - Can't predict specific group's future performance
   - **This is the dealbreaker for many applications**

2. **Weaker Predictive Power** 📉
   - Treats all groups identically
   - Ignores sample size differences in predictions
   - LOO will penalize this (uses same params for all groups)

3. **Coarser Uncertainty** 📏
   - Only captures population uncertainty
   - Doesn't decompose into within-group vs between-group
   - Can't identify which groups are outliers

4. **Limited Diagnostics** 🔍
   - Can't check group-specific posterior predictive fit
   - Harder to detect localized model failures
   - Averaging over groups can hide problems

## Computational Profile

**Expected Performance**:
- **Sampling time**: 5-15 seconds
- **Convergence**: Rhat <1.01 virtually guaranteed
- **ESS**: >1000 typical
- **Memory**: <10 MB
- **Pareto k**: All <0.5 (marginal models are LOO-stable)

**Known Issues**:
- **Boundary behavior**: μ near 0 or 1 can slow convergence
  - Our data (3-14%) is far from boundaries, so not a concern
- **φ scale**: Gamma prior can put too much mass near 0
  - Use Gamma(2, 0.1) if φ keeps hitting lower bound

## Interpretability

**For Everyone**:
> "We model all groups as random samples from the same underlying distribution. We estimate the average success rate and how variable groups are around that average."

**Key Limitation to Communicate**:
> "We can't say which specific group is best - only describe the population pattern."

## Falsification Criteria

### Red Flags: ABANDON if...
- ❌ Posterior predictive checks fail (variance not captured)
- ❌ φ posterior hits prior boundary (φ >100 or φ <0.1)
- ❌ LOO much worse than hierarchical (ΔLOO >20)
- ❌ Stakeholders need group-level estimates (not a model failure, but wrong tool)

### Green Flags: ACCEPT if...
- ✅ Only care about population mean and variance
- ✅ Want to avoid partial pooling complexity
- ✅ Need fast inference (real-time, simulation studies)
- ✅ Posterior predictive variance matches observed (φ ≈ 3.6)

### When to Use This Instead of Hierarchical
**Use beta-binomial if**:
- Main question is "What's the average success rate across groups?"
- Don't need to compare specific groups
- Want simplest defensible model
- Computational speed is critical
- Audience prefers simpler models

**Use hierarchical if**:
- Need group-specific estimates
- Want to identify best/worst groups
- Planning future experiments and want group predictions
- Have follow-up questions like "Which groups should we investigate?"

---

# Model 3: Robust Hierarchical (Student-t) 🔥 ESCALATION MODEL

## Model Specification

### Mathematical Notation
```
Level 1 (Data):
  rⱼ ~ Binomial(nⱼ, pⱼ)

Level 2 (Groups - HEAVY TAILS):
  logit(pⱼ) = μ + τ·ηⱼ
  ηⱼ ~ StudentT(ν, 0, 1)        [ν controls tail heaviness]

Level 3 (Hyperpriors):
  μ ~ Normal(-2.5, 1)
  τ ~ Half-Cauchy(0, 1)
  ν ~ Gamma(2, 0.1)              [Degrees of freedom, ν<5 = heavy tails]
```

**Key Difference**: Student-t distribution for group effects allows outliers without disrupting population estimates.

### Stan Code
```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0> nu;               // Degrees of freedom
  vector[J] eta;                  // Still non-centered, but now t-distributed
}

transformed parameters {
  vector[J] theta = mu + tau * eta;
  vector[J] p = inv_logit(theta);
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  nu ~ gamma(2, 0.1);             // Learns tail heaviness

  // Heavy-tailed group effects (KEY DIFFERENCE)
  eta ~ student_t(nu, 0, 1);

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Same as Model 1, plus:
  real nu_post = nu;              // Monitor tail parameter

  array[J] int r_rep;
  vector[J] log_lik;
  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }
}
```

## Practical Advantages

1. **Outlier Robustness** 🛡️
   - Groups 2, 4, 8 won't pull μ and τ as much
   - Automatically downweights extreme observations
   - ν parameter learns from data how heavy tails should be

2. **Flexible Heterogeneity** 📊
   - Can model both "normal" groups and outliers simultaneously
   - Doesn't force all groups into Gaussian mold
   - Better for real-world data (which is never perfectly Normal)

3. **Self-Diagnostic** 🔍
   - If ν → ∞: Data says "Normal is fine" (use Model 1)
   - If ν <5: Data says "Heavy tails needed" (validates Model 3)
   - Posterior ν tells you if upgrade was necessary

4. **Same Interpretation** 📖
   - Still hierarchical, still partial pooling
   - Just "more robust to extreme groups"
   - Non-statisticians won't notice the difference

## Practical Disadvantages

1. **Computational Cost** ⏱️
   - 2-5x slower than Model 1
   - More divergences possible (Student-t has tricky gradients)
   - Needs adapt_delta=0.95 almost always
   - 1-3 minutes instead of 30 seconds

2. **More Complex** 🧩
   - One more parameter (ν) to monitor
   - Harder to explain to non-experts
   - "Why not just use Normal?" is a fair question

3. **Prior Sensitivity** ⚠️
   - ν prior matters a lot
   - Too tight: Forces near-Normal behavior
   - Too loose: Allows unrealistic heavy tails
   - Gamma(2, 0.1) is reasonable default but not universal

4. **Uncertain Benefit** 🤔
   - With only 12 groups, ν is poorly identified
   - May not actually improve predictions
   - LOO may not distinguish from Model 1
   - **Use only if Model 1 demonstrably fails**

## Computational Profile

**Expected Performance**:
- **Sampling time**: 1-3 minutes
- **Convergence**: Rhat <1.01 (may need longer warmup)
- **ESS**: >300 for most parameters, ν may be lower (100-200)
- **Divergences**: 1-5% typical (acceptable with adapt_delta=0.95)
- **Pareto k**: May improve for outliers (2, 4, 8) vs Model 1

**Known Issues**:
- **ν near boundary**: If ν <1 or ν >30, prior may be problematic
- **Treedepth**: More likely to hit max_treedepth than Model 1

## Interpretability

**For Statisticians**:
> "Hierarchical model with robust Student-t hyperprior to accommodate outliers."

**For Others**:
> "Same as standard model, but less sensitive to unusual groups."

## Falsification Criteria

### Red Flags: ABANDON if...
- ❌ ν posterior is entirely >10 (no evidence for heavy tails → use Model 1)
- ❌ Divergences >10% despite tuning
- ❌ LOO worse than Model 1 (complexity not justified)
- ❌ Sampling time >5 minutes (unacceptable for 12 groups)

### Green Flags: ESCALATE to this model if...
- ✅ Model 1 has multiple Pareto k >0.7
- ✅ Model 1 posterior predictive checks fail for outlier groups
- ✅ Posterior ν <5 (evidence for heavy tails)
- ✅ LOO improves over Model 1 by >5
- ✅ Stakeholders concerned about outlier influence

### When to Use This
**Only escalate to Student-t if**:
1. Model 1 shows specific failures (LOO, PPC, Pareto k)
2. Outliers are known to be problematic in domain
3. Need to defend against "what about those extreme groups?" criticism
4. Prior work suggests heavy-tailed heterogeneity

**Don't use if**:
- Model 1 works fine (Occam's razor)
- Computational time is critical
- Only 12 groups (too few to identify ν reliably)

---

## Model Comparison Strategy

### Phase 1: Quick Triage (20 minutes)

**Fit all three models** with default settings:
- Model 1 (Hierarchical): 4 chains, 2000 iter
- Model 2 (Beta-binomial): 4 chains, 2000 iter
- Model 3 (Robust): 4 chains, 2000 iter (adapt_delta=0.95)

**Immediate Checks**:
1. Do all models converge? (Rhat <1.01)
2. Any divergences in Models 1 or 3?
3. ESS acceptable for key parameters?

**Decision**:
- If Model 1 fails diagnostics → Try Model 3
- If Model 3 also fails → Revisit data quality
- If Model 2 is only success → Check why hierarchical fails

### Phase 2: Predictive Comparison (30 minutes)

**Compute LOO for all converged models**:
```r
loo1 <- loo(fit1)
loo2 <- loo(fit2)
loo3 <- loo(fit3)
loo_compare(loo1, loo2, loo3)
```

**Interpretation**:
- **ΔLOO <4**: Models effectively equivalent, choose simpler
- **ΔLOO 4-10**: Weak preference for better model
- **ΔLOO >10**: Strong preference for better model

**Pareto k diagnostic**:
- All k <0.5: Model is well-specified
- k=0.5-0.7: Model is OK, but watch these groups
- k >0.7: Model is misspecified for this group

**Expected Results** (based on EDA):
- Model 1 vs Model 2: Hierarchical should win (ΔLOO ≈ 10-20)
- Model 1 vs Model 3: Likely equivalent (ΔLOO <4) unless outliers are extreme
- Pareto k for Groups 2, 4, 8: May be 0.5-0.7 in Model 1, <0.5 in Model 3

### Phase 3: Posterior Predictive Checks (20 minutes)

**For best LOO model**, check:

1. **Graphical PPC**:
   ```python
   # Overlay observed vs replicated data
   plot_ppc(observed=r, replicated=r_rep)
   ```
   - Should see overlap between blue bars (observed) and gray bands (predicted)

2. **Test Statistics**:
   - Variance ratio: Does model capture φ ≈ 3.6?
   - Min/max: Does model predict range 3.1%-14.0%?
   - Chi-square: Does model generate similar overdispersion?

3. **Group-Specific**:
   - Do outliers (Groups 2, 4, 8) have reasonable predictions?
   - Are small groups (1, 10) over-shrunk or under-shrunk?

**Pass Criteria**:
- Observed data within 95% posterior predictive interval
- Test statistics within central 90% of posterior predictive distribution
- No systematic bias (e.g., all predictions too high)

### Phase 4: Prior Sensitivity (30 minutes - ONLY IF AMBIGUOUS)

**Only do this if**:
- Models give substantively different conclusions
- LOO is inconclusive (ΔLOO ≈ 4-6)
- Stakeholders concerned about prior influence

**Procedure**:
1. Refit best model with alternative priors:
   - μ: Normal(-2.5, 0.5) and Normal(-2.5, 2)
   - τ: Half-Normal(0, 1) and Half-Cauchy(0, 0.5)

2. Compare posterior means and 95% intervals for:
   - μ (population mean)
   - τ (between-group SD)
   - Individual group rates pⱼ

**Acceptable Sensitivity**:
- Parameters change <10%
- Conclusions don't flip (e.g., Group 8 still highest rate)

**Problematic Sensitivity**:
- μ changes >20%
- Rank order of groups changes
- Some groups shrink toward prior mean instead of data

---

## Decision Tree: Which Model to Use?

```
START
  │
  ├─ Do you need GROUP-SPECIFIC estimates?
  │   │
  │   NO → Model 2 (Beta-binomial)
  │   │     ├─ Fast, simple, good for population inference
  │   │     └─ Accept if: φ ≈ 3.6, PPC passes, stakeholders satisfied
  │   │
  │   YES → Continue
  │       │
  │       ├─ Fit Model 1 (Hierarchical)
  │       │   │
  │       │   ├─ Convergence OK? (Rhat <1.01, ESS >400, Divergences <1%)
  │       │   │   │
  │       │   │   NO → Troubleshoot:
  │       │   │       ├─ Try adapt_delta=0.95
  │       │   │       ├─ Check data for errors
  │       │   │       └─ If still fails → Try Model 3
  │       │   │   │
  │       │   │   YES → Check LOO/PPC
  │       │   │       │
  │       │   │       ├─ All Pareto k <0.7?
  │       │   │       │   │
  │       │   │       │   NO → Try Model 3 (Robust)
  │       │   │       │   │
  │       │   │       │   YES → Check PPC
  │       │   │       │       │
  │       │   │       │       ├─ PPC passes?
  │       │   │       │       │   │
  │       │   │       │       │   YES → ✅ ACCEPT Model 1
  │       │   │       │       │   │      Report results, DONE
  │       │   │       │       │   │
  │       │   │       │       │   NO → Try Model 3
  │       │   │       │       │       │
  │       │   │       │       │       ├─ Model 3 PPC better?
  │       │   │       │       │       │   │
  │       │   │       │       │       │   YES → ✅ ACCEPT Model 3
  │       │   │       │       │       │   │
  │       │   │       │       │       │   NO → 🚩 MODEL MISSPECIFICATION
  │       │   │       │       │       │        ├─ Check data quality
  │       │   │       │       │       │        ├─ Consider covariates
  │       │   │       │       │       │        └─ Consult domain expert
```

---

## Red Flags: When to Stop and Reconsider

### Computational Red Flags 🚨
1. **Persistent divergences** (>5% after tuning)
   - **Meaning**: Posterior geometry is pathological
   - **Action**: Reparameterize or reconsider model class

2. **Rhat >1.05** after 4000 iterations
   - **Meaning**: Chains haven't converged
   - **Action**: Longer warmup, check for multimodality, simplify model

3. **ESS <100** for key parameters
   - **Meaning**: Effective sample too small for inference
   - **Action**: Run longer or reparameterize

4. **Sampling time >10 minutes** for 12 groups
   - **Meaning**: Something is very wrong
   - **Action**: Check model specification, try simpler model

### Statistical Red Flags 🚩
1. **Multiple Pareto k >0.7**
   - **Meaning**: Model systematically misspecified for some groups
   - **Action**: Try robust model (Student-t) or check for outliers

2. **Posterior predictive p-value** <0.01 or >0.99
   - **Meaning**: Model doesn't capture key data features
   - **Action**: Check test statistic choice, try richer model

3. **Prior-posterior conflict**
   - **Meaning**: Data strongly disagrees with prior
   - **Action**: Use weaker prior OR reconsider if data is trustworthy

4. **τ posterior at boundary** (near 0 or very large)
   - **Meaning**: Hierarchical structure may not exist OR prior too strong
   - **Action**: Check if pooled/unpooled is actually better

### Scientific Red Flags ⚠️
1. **Implausible parameter values**
   - μ implies <1% or >30% success rate
   - τ >2 (implies wild heterogeneity)
   - Individual pⱼ outside [0.01, 0.30]
   - **Action**: Check data, priors, or model assumptions

2. **Shrinkage pathology**
   - Large groups shrink more than small groups
   - Extreme groups shrink less than moderate groups
   - **Action**: Model misspecification, not prior issue

3. **Unstable conclusions**
   - Results change drastically across runs
   - Prior sensitivity >20% for key parameters
   - **Action**: More data needed OR model too complex

### Data Red Flags 📊
1. **Influential observations**
   - Removing one group changes μ by >15%
   - **Action**: Investigate that group (data error? true outlier?)

2. **Subset inconsistency**
   - Results differ for Groups 1-6 vs 7-12
   - **Action**: Exchangeability assumption violated, need covariates

3. **Zero or perfect success rates**
   - Any group with r=0 or r=n
   - **Action**: May need continuity correction or beta prior

---

## Green Flags: When Iteration Likely Helps

### Worth Exploring 🟢
1. **Pareto k = 0.5-0.7** for a few groups
   - **Action**: Try robust model, may get ΔLOO ≈ 5-10

2. **ESS = 100-200** for τ
   - **Action**: Longer run will stabilize, not a model problem

3. **Divergences 1-3%** in hierarchical model
   - **Action**: adapt_delta=0.95 will likely fix

4. **Posterior predictive check** marginal (p=0.05 or 0.95)
   - **Action**: Check different test statistics, may be false alarm

### Not Worth It 🛑
1. **All diagnostics perfect**, LOO clearly favors one model
   - **Action**: STOP ITERATING, accept the model, report results

2. **ΔLOO <2** between models
   - **Action**: Choose simpler model, move on

3. **Results substantively identical** across models
   - **Action**: Use simplest model for communication

4. **Stakeholders satisfied** with current model
   - **Action**: Perfect is enemy of good, ship it

---

## Resource Estimates

### Time Budget (One Analyst, Modern Laptop)

**Model 1 (Hierarchical)** - Total: 1.5 hours
- Coding/setup: 30 min
- Sampling: 1 min
- Diagnostics: 20 min
- Visualization: 30 min
- Interpretation: 10 min

**Model 2 (Beta-binomial)** - Total: 1 hour
- Coding/setup: 20 min
- Sampling: 10 sec
- Diagnostics: 15 min
- Visualization: 20 min
- Interpretation: 5 min

**Model 3 (Robust)** - Total: 2 hours
- Coding/setup: 40 min (more complex)
- Sampling: 2-3 min
- Diagnostics: 30 min (check ν)
- Visualization: 40 min
- Interpretation: 15 min

**Full Comparison** - Total: 3 hours
- Fit all three: 5 min
- LOO comparison: 10 min
- PPC for best model: 30 min
- Prior sensitivity: 30 min
- Write-up: 1.5 hours

### Computational Resources

**Minimum Viable**:
- Laptop: 4-core i5 or M1
- RAM: 8 GB
- Storage: 1 GB (for samples)
- Software: Stan 2.30+ or PyMC 5.0+

**Recommended**:
- Desktop: 8-core i7 or M2
- RAM: 16 GB
- Storage: 5 GB
- Software: Latest stable versions
- Parallel: 4 chains on separate cores

**Not Needed**:
- GPU (12 groups is tiny)
- HPC cluster
- Distributed computing
- More than 16 GB RAM

### Scaling Expectations

**If you had MORE groups**:
- J=50: Model 1 takes ~5 min, Model 3 takes ~10 min
- J=100: Model 1 takes ~15 min, Model 3 takes ~30 min
- J>200: Consider approximate methods (INLA, variational Bayes)

**If you had FEWER groups**:
- J=5: Hierarchical model poorly identified, may need stronger priors
- J<5: Don't use hierarchical, just use unpooled or informative priors

---

## Implementation Checklist

### Before Fitting Any Model ✓
- [ ] Read data, confirm J=12, N=2814, r=196
- [ ] Check for missing values (should be none)
- [ ] Verify r[j] ≤ n[j] for all j
- [ ] Compute observed overdispersion (should get φ ≈ 3.6)
- [ ] Identify outlier groups (2, 4, 8)

### Model 1 (Hierarchical) ✓
- [ ] Use non-centered parameterization
- [ ] Prior: μ ~ Normal(-2.5, 1)
- [ ] Prior: τ ~ Half-Cauchy(0, 1)
- [ ] Prior: η ~ Normal(0, 1)
- [ ] Likelihood: binomial_logit
- [ ] 4 chains, 2000 iterations, 1000 warmup
- [ ] Check Rhat <1.01 for all parameters
- [ ] Check ESS >400 for μ, τ
- [ ] Check divergences <1%
- [ ] Compute LOO, check Pareto k
- [ ] Posterior predictive checks (variance, range)
- [ ] Visualize shrinkage (raw vs posterior rates)

### Model 2 (Beta-binomial) ✓
- [ ] Prior: μ ~ Beta(5, 50)
- [ ] Prior: φ ~ Gamma(1, 0.1)
- [ ] Likelihood: beta_binomial
- [ ] 4 chains, 2000 iterations, 1000 warmup
- [ ] Should converge easily (Rhat <1.01)
- [ ] Compute LOO
- [ ] Check implied variance ratio vs observed (3.6)
- [ ] Posterior predictive checks

### Model 3 (Robust) - Only if Needed ✓
- [ ] Same as Model 1, plus:
- [ ] Prior: ν ~ Gamma(2, 0.1)
- [ ] Replace Normal with Student-t for η
- [ ] adapt_delta=0.95
- [ ] Monitor ν posterior
- [ ] Check if ν <5 (heavy tails needed)
- [ ] Compare LOO to Model 1

### Comparison ✓
- [ ] LOO for all fitted models
- [ ] loo_compare() to rank models
- [ ] Check ΔLOO >10 for "clear winner"
- [ ] PPC for best model
- [ ] Prior sensitivity if needed
- [ ] Document decision rationale

### Reporting ✓
- [ ] Model specification (equations + code)
- [ ] Convergence diagnostics (Rhat, ESS, divergences)
- [ ] LOO comparison table
- [ ] Posterior summaries (μ, τ, individual pⱼ)
- [ ] Shrinkage visualization
- [ ] Posterior predictive checks
- [ ] Conclusions: which groups differ, by how much

---

## Final Recommendations

### For This Dataset Specifically

**Base Case** (90% confidence):
1. **Start with Model 1** (Hierarchical non-centered)
2. **It will probably work** (converge, pass diagnostics)
3. **Report those results** unless clear problems emerge

**Expected Findings**:
- μ ≈ -2.5 to -2.3 (implies ~7-9% pooled rate)
- τ ≈ 0.3 to 0.6 (moderate heterogeneity)
- Groups 1, 10 (small n) shrink 50-70%
- Group 4 (large n) shrinks 15-25%
- Groups 2, 8 (high rates) shrink toward μ
- LOO outperforms pooled by >20
- All Pareto k <0.7

**If Model 1 Fails**:
1. First check: adapt_delta=0.95
2. Still fails: Check data quality
3. Data OK: Try Model 3 (Robust)
4. Model 3 also fails: Consult domain expert

**If Stakeholders Only Care About Population Mean**:
- Use Model 2 (Beta-binomial)
- Faster, simpler, good enough
- Report φ as heterogeneity measure

### General Advice

1. **Start simple**: Model 1 is the right default for this problem
2. **Validate ruthlessly**: Don't trust convergence without diagnostics
3. **Compare models**: Fit 2-3, let LOO decide
4. **Embrace "good enough"**: Perfect model doesn't exist
5. **Document everything**: Future you will thank present you
6. **Communicate clearly**: Stakeholders don't care about MCMC

### The Bayesian Workflow

```
1. Prior predictive checks (20 min)
   └─ Do priors generate plausible data?

2. Fit model (1-5 min)
   └─ Wait for chains to finish

3. Convergence diagnostics (10 min)
   └─ Rhat, ESS, divergences, pairs plots

4. Posterior predictive checks (30 min)
   └─ Does model capture key data features?

5. Model comparison (20 min)
   └─ LOO, ΔLOO, Pareto k

6. Sensitivity analysis (30 min - if needed)
   └─ Robustness to prior choices

7. Interpretation (1 hour)
   └─ What do parameters mean scientifically?

8. Communication (2 hours)
   └─ Plots, tables, narrative for stakeholders
```

**Total Time**: 4-6 hours for complete analysis

**Stopping Rule**:
- If Step 4 passes → DONE, report results
- If Step 4 fails → Loop back to Step 1 with modified model
- If looping >3 times → Fundamental problem, reconsider approach

---

## Conclusion

**For this dataset**:
- **Model 1 (Hierarchical)** is the correct choice
- **Model 2 (Beta-binomial)** is acceptable if only population inference needed
- **Model 3 (Robust)** is probably overkill but worth checking

**Practical reality**:
- Model 1 will work fine
- Spend time on interpretation, not model tweaking
- Ship results, don't chase perfect diagnostics

**Success criteria**:
- Rhat <1.01 ✓
- ESS >400 ✓
- PPC passes ✓
- Stakeholders understand results ✓

**Remember**: The goal is inference, not sampling. A "good enough" model that answers the scientific question beats a "perfect" model that no one understands.

---

**Files Created**:
- `/workspace/experiments/designer_3/proposed_models.md` (this document)

**Next Steps**:
1. Implement Model 1 in Stan/PyMC
2. Run diagnostics
3. If successful, report results
4. If not, escalate to Model 3 or consult other designers
