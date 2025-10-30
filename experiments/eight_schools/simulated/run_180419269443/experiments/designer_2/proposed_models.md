# Robust Bayesian Model Proposals for Meta-Analysis
## Designer 2: Focus on Distributional Robustness and Sensitivity Analysis

**Date:** 2025-10-28
**Designer:** Model Designer 2 (Independent)
**Dataset:** J=8 studies, meta-analysis with known standard errors
**EDA Summary:** I²=2.9%, pooled effect=11.27, no outliers detected, Study 4 influential

---

## Executive Summary

While EDA suggests low heterogeneity and normality assumptions are reasonable, this proposal takes an **adversarial stance**: What if the apparent homogeneity is an artifact of small sample size? What if future outliers emerge? What if our distributional assumptions are wrong?

I propose **three distinct model classes** that differ fundamentally in their assumptions about data generation:

1. **Heavy-Tailed Hierarchical Model (t-distribution)** - Robustness to outliers
2. **Mixture Model** - Heterogeneity in heterogeneity (subpopulations)
3. **Non-Parametric Dirichlet Process** - Minimal distributional assumptions

Each model has explicit **falsification criteria** and **escape routes** if evidence contradicts assumptions.

---

## Philosophy: Planning for Failure

### Critical Mindset

**The EDA might be misleading because:**
- J=8 is too small to reliably detect outliers (power issue)
- Study 5's negative effect could signal a hidden subpopulation
- Study 4's high influence (33% change if removed) suggests fragility
- I²=2.9% could be underestimated due to small sample size
- Future studies might reveal heterogeneity not apparent now

**My commitment:** I will **abandon these models entirely** if they exhibit:
- Prior-posterior conflict (model fighting data)
- Extreme computational issues (often signals misspecification)
- Systematically poor predictive performance
- Inconsistent shrinkage patterns across validation folds

**Success metric:** Finding a model that truly explains the data, even if it means rejecting all proposals and starting over.

---

## Model Class 1: Heavy-Tailed Hierarchical (t-Distribution)

### Mathematical Specification

```
Likelihood:
  y_i ~ Student_t(nu, theta_i, sigma_i)    for i = 1, ..., 8

Hierarchical structure:
  theta_i ~ Normal(mu, tau)                 # Study-specific effects

Priors:
  mu ~ Normal(0, 50)                        # Weakly informative on mean
  tau ~ Half-Cauchy(0, 5)                   # Heavy-tailed prior on heterogeneity
  nu ~ Gamma(2, 0.1)                        # Prior on degrees of freedom
                                            # (mean=20, allows 3-100+ range)
```

### Why Student-t Instead of Normal?

**Theoretical justification:**
- Student-t with low degrees of freedom (nu<10) has heavier tails
- Downweights extreme observations automatically (robust to outliers)
- Nests normal distribution as nu → infinity
- Allows data to inform tail behavior via nu estimation

**When to prefer this model:**
- If future studies reveal outliers (even though EDA shows none now)
- If Study 5's negative effect is more extreme than sampling variation
- If we're uncertain about data quality in original studies
- If domain knowledge suggests occasional extreme effects

**Expected behavior given EDA:**
- Should estimate nu > 30 (near-normal) given current low heterogeneity
- If nu < 10, suggests data favor heavy tails despite low I²
- Posterior for mu should be similar to normal model
- Posterior for tau might be slightly larger (accounting for tails)

### Falsification Criteria

**I will abandon this model if:**

1. **Nu estimates unrealistically low (nu < 3)**
   - Signal: Model over-explaining tail behavior
   - Action: Check for data errors or coding bugs

2. **Posterior predictive checks fail systematically**
   - Test: Generate y_rep from posterior, compare to observed y
   - Threshold: If more than 2/8 studies outside 95% prediction interval

3. **Extreme prior-posterior conflict**
   - Signal: Posterior concentrates far from prior mode despite weak prior
   - Example: tau posterior peaks at boundary (0 or infinity)

4. **Computational issues persist**
   - If after reparameterization, MCMC still shows:
     - R-hat > 1.05 after 10,000 iterations
     - Effective sample size < 100
     - Divergent transitions > 5%

5. **Worse predictive performance than normal model**
   - Metric: Leave-one-out cross-validation (LOO-CV)
   - Threshold: ELPD difference > 2 SE worse than normal model

### Stress Tests

**Designed to break the model:**

1. **Extreme value injection:**
   - Add hypothetical Study 9 with y=100, sigma=10
   - Check if nu adapts or model breaks

2. **Study 5 sensitivity:**
   - Refit excluding Study 5
   - If nu changes by > 50%, model is unstable

3. **Prior sensitivity:**
   - Vary nu prior: Gamma(2, 0.5), Exponential(0.1), Uniform(1, 100)
   - If posteriors differ substantially, data are uninformative

### Computational Strategy (Stan)

```stan
data {
  int<lower=1> J;           // Number of studies
  vector[J] y;              // Observed effects
  vector<lower=0>[J] sigma; // Known standard errors
}

parameters {
  real mu;                  // Population mean
  real<lower=0> tau;        // Between-study SD
  vector[J] theta_raw;      // Non-centered parameterization
  real<lower=1> nu;         // Degrees of freedom
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  nu ~ gamma(2, 0.1);
  theta_raw ~ normal(0, 1);

  // Likelihood (t-distribution)
  y ~ student_t(nu, theta, sigma);
}

generated quantities {
  vector[J] log_lik;
  vector[J] y_rep;
  real mu_pred;

  for (i in 1:J) {
    log_lik[i] = student_t_lpdf(y[i] | nu, theta[i], sigma[i]);
    y_rep[i] = student_t_rng(nu, theta[i], sigma[i]);
  }
  mu_pred = normal_rng(mu, tau);
}
```

**Key implementation notes:**
- Use **non-centered parameterization** for theta (avoids funnel geometry)
- nu prior encourages moderate values but allows wide range
- Generate log_lik for LOO-CV comparison
- Generate y_rep for posterior predictive checks

### Red Flags and Pivot Points

**Decision point 1: After initial fit**
- If nu > 50: Data don't need heavy tails → pivot to normal model
- If nu < 5: Extreme tails suggested → investigate data quality

**Decision point 2: After LOO-CV**
- If LOO-CV worse than normal: Heavy tails not improving prediction
- If LOO-CV substantially better (ΔELPD > 4): Validate with simulation

**Escape route:**
- If model fails all tests → Try mixture model (Model 2)
- If computational issues persist → Try empirical Bayes with robust errors

---

## Model Class 2: Mixture Model (Heterogeneous Heterogeneity)

### Mathematical Specification

```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)

Mixture hierarchical structure:
  theta_i ~ pi * Normal(mu_1, tau_1) + (1-pi) * Normal(mu_2, tau_2)

Priors:
  mu_1 ~ Normal(0, 50)                      # Mean of cluster 1
  mu_2 ~ Normal(0, 50)                      # Mean of cluster 2
  tau_1 ~ Half-Normal(0, 5)                 # SD of cluster 1
  tau_2 ~ Half-Normal(0, 10)                # SD of cluster 2 (allow larger)
  pi ~ Beta(2, 2)                           # Mixing proportion (favor 50/50)
```

### Why Mixture Model?

**Theoretical justification:**
- Allows for **hidden subpopulations** (e.g., different study designs, populations)
- Study 5's negative effect might indicate separate cluster
- More flexible than single tau (heterogeneity in heterogeneity)
- Can detect bimodality not apparent with J=8

**When to prefer this model:**
- If studies cluster into high/low effect groups
- If Study 5 truly represents different data-generation process
- If domain knowledge suggests subgroups (e.g., RCT vs observational)
- If sensitivity analyses show bifurcation in estimates

**Expected behavior given EDA:**
- With I²=2.9%, expect **model to collapse** to single cluster (pi → 0 or 1)
- This is good! Means data don't support mixture
- If mixture persists, suggests hidden structure in apparent homogeneity
- Should see mu_1 ≈ mu_2 if truly homogeneous

### Falsification Criteria

**I will abandon this model if:**

1. **Label switching persists despite constraints**
   - Signal: MCMC chains randomly swap cluster labels
   - Action: Add ordering constraint (mu_1 < mu_2), if still fails → abandon

2. **Mixing proportion unidentifiable**
   - Test: Posterior for pi covers full [0,1] range uniformly
   - Means: Data provide no information about clusters

3. **Both clusters collapse to single point**
   - If tau_1 ≈ tau_2 ≈ 0 and mu_1 ≈ mu_2
   - Means: Should use common-effect model instead

4. **Cluster assignments nonsensical**
   - Example: Studies 1-4 in cluster 1, studies 5-8 in cluster 2
   - But no covariate or characteristic distinguishes them

5. **Worse fit than single-cluster hierarchical**
   - Metric: WAIC or LOO-CV
   - Threshold: More than 2 SE worse (mixture complexity not justified)

### Stress Tests

**Designed to break the model:**

1. **Identifiability test:**
   - Simulate data from single cluster
   - Fit mixture model, check if pi → 0 or 1
   - If not, model has fundamental identification issue

2. **Study 5 focus:**
   - Force Study 5 into separate cluster (informative prior)
   - Does model support this or fight it?

3. **Prior sensitivity on pi:**
   - Try Beta(1,1) (uniform), Beta(10,1) (favor cluster 1)
   - If posteriors wildly different, data are uninformative

### Computational Strategy (PyMC)

```python
import pymc as pm
import numpy as np

with pm.Model() as mixture_model:
    # Priors
    mu_1 = pm.Normal('mu_1', mu=0, sigma=50)
    mu_2 = pm.Normal('mu_2', mu=0, sigma=50)
    tau_1 = pm.HalfNormal('tau_1', sigma=5)
    tau_2 = pm.HalfNormal('tau_2', sigma=10)
    pi = pm.Beta('pi', alpha=2, beta=2)

    # Ordering constraint to avoid label switching
    mu = pm.Deterministic('mu', pm.math.stack([mu_1, mu_2]))
    tau = pm.Deterministic('tau', pm.math.stack([tau_1, tau_2]))

    # Mixture distribution for study effects
    components = [
        pm.Normal.dist(mu=mu[0], sigma=tau[0]),
        pm.Normal.dist(mu=mu[1], sigma=tau[1])
    ]
    w = pm.Deterministic('w', pm.math.stack([pi, 1-pi]))
    theta = pm.Mixture('theta', w=w, comp_dists=components, shape=J)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Sample
    trace = pm.sample(2000, tune=2000, target_accept=0.95)
```

**Key implementation notes:**
- **Ordering constraint** (mu_1 < mu_2) prevents label switching
- Use `target_accept=0.95` for challenging geometry
- Monitor convergence carefully (mixture models can be tricky)
- Consider alternative: **Dirichlet Process** for arbitrary clusters

### Red Flags and Pivot Points

**Decision point 1: After convergence diagnostics**
- If R-hat > 1.05 after extensive tuning → Model too complex for data
- If effective sample size < 50 → Posterior poorly identified

**Decision point 2: After posterior analysis**
- If pi posterior: [0.45, 0.55] and mu_1 ≈ mu_2 → Collapse to single cluster
- If pi posterior: [0.01, 0.05] or [0.95, 0.99] → Essentially single cluster
- If pi posterior bimodal or uniform → Data uninformative

**Decision point 3: After predictive validation**
- If posterior predictive check shows poor calibration → Model misspecified
- If LOO-CV worse than hierarchical normal → Complexity not warranted

**Escape route:**
- If mixture doesn't identify → Use hierarchical normal (simpler)
- If evidence of >2 clusters → Try Dirichlet Process (Model 3)
- If label switching unsolvable → Use different parameterization or abandon

---

## Model Class 3: Dirichlet Process Mixture (Non-Parametric)

### Mathematical Specification

```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)

Dirichlet Process prior:
  theta_i ~ G
  G ~ DP(alpha, G_0)

Base distribution:
  G_0 = Normal(mu_0, tau_0)

Hyperpriors:
  mu_0 ~ Normal(0, 50)
  tau_0 ~ Half-Normal(0, 10)
  alpha ~ Gamma(1, 1)                       # Concentration parameter
```

### Why Dirichlet Process?

**Theoretical justification:**
- **Makes minimal assumptions** about number of clusters
- Lets data determine complexity (cluster count not fixed)
- Natural Bayesian generalization of mixture models
- Appropriate when we're fundamentally uncertain about heterogeneity structure

**When to prefer this model:**
- If mixture model suggests >2 clusters but we're uncertain
- If we want to avoid specifying cluster count a priori
- If domain knowledge provides no guidance on subgroups
- If sensitivity to model choice is high and we want robustness

**Expected behavior given EDA:**
- With I²=2.9%, expect **most studies assigned to single cluster**
- Concentration parameter alpha should be small (favors fewer clusters)
- If alpha large, suggests data favor multiple clusters despite low I²
- Number of effective clusters should be close to 1-2

### Falsification Criteria

**I will abandon this model if:**

1. **Model assigns each study to separate cluster**
   - Signal: No pooling happening, equivalent to no-pooling model
   - Threshold: More than 5 distinct clusters with J=8

2. **Concentration parameter hits boundary**
   - If alpha → 0: Reduce to single cluster (use hierarchical normal)
   - If alpha → infinity: Reduce to no pooling (use fixed effects)

3. **Computational failure despite proper implementation**
   - DP models are challenging; if multiple samplers fail → too complex

4. **Predictions indistinguishable from hierarchical normal**
   - Test: Compare posterior predictive distributions
   - If essentially identical, complexity not justified

5. **Cluster assignments change dramatically across MCMC samples**
   - Signal: Posterior not well-defined or poorly identified

### Stress Tests

**Designed to break the model:**

1. **Concentration sensitivity:**
   - Vary alpha prior: Gamma(0.1, 0.1), Gamma(5, 1), Fixed at 1
   - Check if cluster count changes dramatically

2. **Base distribution sensitivity:**
   - Vary tau_0 prior: Half-Normal(0, 5) vs Half-Normal(0, 20)
   - If results change substantially, data uninformative

3. **Initialization test:**
   - Run 3 chains with different random seeds
   - Check if they agree on cluster count and assignments

### Computational Strategy (PyMC with Stick-Breaking)

```python
import pymc as pm
import numpy as np

def stick_breaking(beta):
    """Stick-breaking construction of DP weights"""
    portion_remaining = pm.math.concatenate([[1], pm.math.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

with pm.Model() as dp_model:
    # Maximum number of clusters (truncation)
    K = 8  # Upper bound (J studies)

    # Base distribution hyperparameters
    mu_0 = pm.Normal('mu_0', mu=0, sigma=50)
    tau_0 = pm.HalfNormal('tau_0', sigma=10)

    # Concentration parameter
    alpha = pm.Gamma('alpha', alpha=1, beta=1)

    # Stick-breaking construction
    beta = pm.Beta('beta', alpha=1, beta=alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    # Cluster means from base distribution
    mu_k = pm.Normal('mu_k', mu=mu_0, sigma=tau_0, shape=K)

    # Mixture for each study
    theta = pm.Mixture('theta', w=w, comp_dists=pm.Normal.dist(mu=mu_k, sigma=1e-6), shape=J)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Sample with careful tuning
    trace = pm.sample(2000, tune=3000, target_accept=0.99)
```

**Key implementation notes:**
- Use **stick-breaking representation** (more stable than Chinese Restaurant Process)
- Truncate at K=8 (finite approximation to infinite DP)
- May need long warm-up and high target_accept
- Alternative: Use specialized DP sampler in NumPyro

### Red Flags and Pivot Points

**Decision point 1: After checking effective cluster count**
- If posterior estimate of K=1 always → Use hierarchical normal
- If K uniformly distributed over [1, 8] → Model unidentified

**Decision point 2: After comparing to simpler models**
- If LOO-CV no better than hierarchical normal → Not worth complexity
- If LOO-CV substantially worse → Model too flexible (overfitting)

**Decision point 3: After interpretation**
- If cluster assignments make no scientific sense → Spurious patterns
- If same studies cluster across different initializations → Trust pattern

**Escape route:**
- If DP too complex → Return to finite mixture (Model 2)
- If finite mixture fails → Return to hierarchical normal with t-errors (Model 1)
- If all models fail → Reconsider data quality and problem formulation

---

## Model Comparison and Selection Strategy

### Prioritization Recommendation

**Order of implementation:**

1. **Start with Model 1 (Heavy-tailed hierarchical)** ⭐ RECOMMENDED FIRST
   - Rationale: Conservative extension of standard model
   - Low risk: If nu > 30, recovers normal model
   - Computational: Most stable, well-tested
   - Interpretability: Easy to explain (robust version of standard)

2. **Then try Model 2 (Mixture) if Model 1 shows issues**
   - Rationale: Next level of complexity
   - Conditional on: If Study 5 shows as potential outlier in diagnostics
   - Risk: Identifiability issues with J=8
   - Only pursue if clear evidence of subgroups

3. **Reserve Model 3 (Dirichlet Process) for deep uncertainty**
   - Rationale: Most flexible, least assumptions
   - Use when: Fundamental uncertainty about structure
   - Risk: May be too complex for J=8
   - Primarily for sensitivity analysis

### Quantitative Selection Criteria

**Model comparison metrics (use all three):**

1. **Leave-One-Out Cross-Validation (LOO-CV)**
   - Compute ELPD (expected log pointwise predictive density)
   - Prefer model with highest ELPD
   - If ΔELPD < 2 SE: Models equivalent, choose simpler

2. **Posterior Predictive Checks**
   - Generate y_rep from posterior
   - Test statistics: mean, SD, min, max, range
   - Threshold: p-value in [0.05, 0.95] for each test

3. **Parameter Interpretability**
   - Check for extreme values (e.g., tau > 100)
   - Check for boundary issues (e.g., tau ≈ 0)
   - Check for prior-posterior conflict

### Decision Tree

```
START: Fit Model 1 (t-distribution)
  |
  ├─ If nu > 50 → Normal distribution adequate
  |    └─ Compare to standard hierarchical normal (Designer 1's model)
  |
  ├─ If 10 < nu < 50 → Mild heavy tails
  |    └─ Report as primary model
  |
  ├─ If nu < 10 → Strong heavy tails
  |    ├─ Check data quality
  |    └─ Try Model 2 (Mixture) to see if subgroups explain
  |
  └─ If computational issues → Try simpler normal model

If Model 2 pursued:
  |
  ├─ If pi ≈ 0 or 1 → Single cluster (return to Model 1)
  |
  ├─ If 0.1 < pi < 0.9 AND mu_1 ≠ mu_2 → Genuine mixture
  |    └─ Investigate which studies in which cluster
  |
  └─ If clusters > 2 suggested → Try Model 3 (DP)

If Model 3 pursued:
  |
  ├─ If K_eff = 1 → Return to hierarchical normal
  |
  ├─ If K_eff = 2 → Validate with Model 2
  |
  └─ If K_eff > 2 → Question data assumptions
```

---

## Comparison to Classical Normal Models

### Standard Hierarchical Normal (Designer 1's likely model)

```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
mu ~ Normal(0, 50)
tau ~ Half-Normal(0, 10)
```

**My models differ by:**

1. **Model 1 vs Standard:**
   - Adds nu parameter for tail behavior
   - Cost: +1 parameter, +10% computational time
   - Benefit: Robustness to outliers, data-driven tail assessment
   - When equivalent: If nu > 50, models give same inference

2. **Model 2 vs Standard:**
   - Adds subpopulation structure (pi, mu_2, tau_2)
   - Cost: +3 parameters, +50% computational time, identifiability risk
   - Benefit: Can detect hidden heterogeneity
   - When equivalent: If pi → 0 or 1, reduces to standard

3. **Model 3 vs Standard:**
   - Adds infinite flexibility in cluster count
   - Cost: +alpha parameter, +100% computational time, complexity
   - Benefit: Minimal assumptions about structure
   - When equivalent: If K_eff = 1, reduces to standard

### Expected Findings (Given EDA)

**My predictions:**

- **Model 1:** Will likely estimate nu ≈ 40-60 (near-normal)
  - If so, recommend standard normal model for parsimony
  - If nu < 20, this is important finding worth reporting

- **Model 2:** Will likely collapse to single cluster (pi ≈ 0 or 1)
  - If so, validates homogeneity assumption
  - If genuine mixture found, this is major discovery

- **Model 3:** Will likely favor K_eff ≈ 1-2 clusters
  - If so, confirms simple structure
  - If K_eff > 3, questions EDA conclusions

**What would surprise me (and change everything):**

1. **Model 1 estimates nu < 10**
   - Would suggest: Data have heavier tails than apparent
   - Action: Investigate Study 5 and 4 more carefully
   - Pivot: Consider measurement error models

2. **Model 2 finds genuine 50/50 mixture**
   - Would suggest: Hidden subpopulation not visible with I² statistic
   - Action: Seek covariates to explain clusters
   - Pivot: Meta-regression if covariates available

3. **Model 3 consistently estimates K_eff > 3**
   - Would suggest: Each study from different population
   - Action: Question pooling assumption entirely
   - Pivot: Consider no-pooling or report study-level estimates only

---

## Sensitivity Analyses (Across All Models)

### Required for Each Model

1. **Prior sensitivity**
   - Vary all hyperpriors by factor of 2-3
   - Check if posteriors change by > 10%
   - Report range of estimates

2. **Influence analysis**
   - Remove Study 4 (most influential from EDA)
   - Remove Study 5 (only negative effect)
   - Check if model class choice changes

3. **Computational validation**
   - Run 4 chains from different initializations
   - Check R-hat < 1.01 for all parameters
   - Check ESS > 400 for all parameters

### Cross-Model Validation

**Posterior predictive checks (same for all models):**

```python
# Generate predictions for new study
for each posterior sample:
    theta_new ~ model-specific distribution
    y_new ~ Normal(theta_new, sigma_new=11)  # median observed SE

# Check if observed studies fall in prediction interval
for i in 1 to 8:
    p_value[i] = mean(y[i] < y_rep[,i])

# Good calibration: p_values roughly uniform on [0,1]
```

**Expect to see:**
- All models should have similar predictive performance
- If one model much better: Strong evidence for that structure
- If all similar: Choose simplest (parsimony)

---

## Red Flags That Would Abort Everything

### Global Falsification Criteria (All Models)

**I will stop and reconsider the entire approach if:**

1. **All models show same pathology**
   - Example: All estimate tau → 0 (no between-study variation)
   - Conclusion: Data may be fabricated or severely rounded

2. **Computational issues across all implementations**
   - Example: Even normal model shows divergences
   - Conclusion: Data may have errors or need transformation

3. **Posterior predictive checks fail for all models**
   - Example: All models predict narrower range than observed
   - Conclusion: Fundamental model misspecification

4. **Sensitivity analyses show extreme instability**
   - Example: Removing any study changes estimate by >50%
   - Conclusion: Too few studies for reliable inference

5. **Models give qualitatively different conclusions**
   - Example: Model 1 says effect=15, Model 2 says effect=5
   - Conclusion: Data insufficient to distinguish hypotheses

### Alternative Approaches if All Models Fail

**Plan B options:**

1. **Fixed-effect model only**
   - If tau consistently → 0 across all models
   - Report: "Data suggest no between-study heterogeneity"

2. **No pooling**
   - If all pooling models show poor fit
   - Report study-level estimates with caution

3. **Empirical Bayes**
   - If full Bayes shows computational issues
   - Use REML for tau, plug in to Bayesian framework

4. **Frequentist meta-analysis**
   - If Bayesian models too unstable
   - Use DerSimonian-Laird with profile likelihood CI

5. **Narrative synthesis**
   - If all statistical models fail
   - Qualitative description of studies

---

## Implementation Roadmap

### Phase 1: Baseline (Week 1)

- Implement Model 1 (t-distribution) in Stan
- Run 4 chains × 2000 iterations
- Check convergence diagnostics
- Compute LOO-CV
- Run posterior predictive checks

**Go/No-Go decision:**
- Go to Phase 2 if: Converged, nu interpretable, LOO reasonable
- Abort if: Divergences >1%, R-hat >1.05, nonsensical posteriors

### Phase 2: Exploration (Week 2)

- If Phase 1 successful: Implement Model 2 (Mixture)
- Compare LOO-CV to Model 1
- Investigate cluster assignments
- Run sensitivity analyses

**Go/No-Go decision:**
- Go to Phase 3 if: Evidence of >1 cluster
- Skip to Phase 4 if: Single cluster confirmed

### Phase 3: Advanced (Week 3, conditional)

- Only if mixture model shows promise
- Implement Model 3 (Dirichlet Process)
- Compare across all three models
- Investigate cluster structure

**Go/No-Go decision:**
- Continue if: DP provides insights
- Abandon if: Too complex without benefit

### Phase 4: Synthesis (Week 4)

- Compare best model to Designer 1's model
- Conduct cross-validation
- Write final recommendations
- Prepare visualizations

---

## Expected Deliverables

### For Each Model Class

1. **Stan/PyMC code** (fully commented)
2. **Convergence diagnostics** (R-hat, ESS, trace plots)
3. **Posterior summaries** (tables with quantiles)
4. **LOO-CV results** (with SE)
5. **Posterior predictive checks** (graphical + numerical)
6. **Sensitivity analyses** (prior, influence)

### Comparative Analysis

1. **Model comparison table** (LOO, WAIC, parameters)
2. **Forest plots** (comparing posteriors across models)
3. **Decision justification** (why chosen model is best)
4. **Failure modes documented** (what would make us switch)

### Final Recommendation

Based on current EDA, **I predict:**
- Model 1 will be sufficient (nu ≈ 30-50)
- Model 2 will collapse to single cluster
- Model 3 will not be needed

**But I'm prepared to be wrong.** If data show:
- nu < 10 → Heavy tails are real, report Model 1
- Genuine mixture → Report Model 2 as primary
- Complex heterogeneity → Report Model 3 or abandon pooling

---

## Success Criteria

### Scientific Success

**The project succeeds if:**
- We find a model that genuinely explains the data
- We understand why other models fail
- We can predict future studies reasonably well
- We quantify uncertainty honestly

**The project fails if:**
- We fit models without checking assumptions
- We ignore evidence contradicting our approach
- We report point estimates without prediction intervals
- We claim certainty where none exists

### Statistical Success

**Good outcomes:**
- Posterior predictive p-values in [0.05, 0.95]
- LOO-CV pareto-k < 0.7 for all studies
- Effective sample size > 400 for all parameters
- Convergence R-hat < 1.01

**Warning signs:**
- Any pareto-k > 0.7 (influential point)
- Posterior predictive checks with p < 0.01 or p > 0.99
- Wide credible intervals on tau (poorly identified)

---

## Conclusion: Adversarial Mindset

This proposal assumes **the EDA might be wrong**:
- I²=2.9% could be underestimated (small sample)
- Normality could be local approximation (not global truth)
- Study 4's influence suggests fragility
- Future data might reveal structure not apparent now

**My commitment:** I will **actively try to break these models** and switch approaches if evidence demands it. Success = finding truth, not confirming assumptions.

**Falsification is the goal.** Each model has explicit criteria for abandonment. If all models fail, I will report that honestly and recommend alternative approaches.

**The best outcome:** Discovering early that our assumptions are wrong and pivoting quickly to better models.

---

**Files to be created:**
- `/workspace/experiments/designer_2/model_1_t_distribution.stan`
- `/workspace/experiments/designer_2/model_2_mixture.py`
- `/workspace/experiments/designer_2/model_3_dirichlet_process.py`
- `/workspace/experiments/designer_2/analysis_script.py`
- `/workspace/experiments/designer_2/validation_tests.py`

**Next steps:** Await approval to implement, or iterate on design based on feedback.
