# Designer 3: Robust Model Specifications
## Eight Schools Meta-Analysis - Robustness-Focused Approach

**Designer:** Designer 3 (Independent - Robust Models Focus)
**Date:** 2025-10-28
**Design Philosophy:** Adversarial testing of robustness assumptions

---

## Executive Summary

**Critical Question:** Do we need robust models at all for this dataset?

The EDA shows NO outliers (all |z| < 2), NO heterogeneity (Q p=0.696), and perfect consistency with normal sampling variability. This suggests **robustness may be a solution looking for a problem**.

However, I propose three model classes to test this hypothesis empirically:
1. **Student-t models** (heavy-tailed alternatives)
2. **Mixture models** (latent outlier detection)
3. **Prior sensitivity analysis** (robustness to prior specification)

**Key Insight:** If robust models don't differ from normal models, that's EVIDENCE that normality assumptions are appropriate. If they do differ, we learn something important about model misspecification.

---

## Design Philosophy: When Robustness Backfires

### The Robustness Paradox

Adding robustness when it's not needed:
- **Wastes computational resources**
- **Dilutes information** from well-behaved observations
- **Reduces power** to detect true effects
- **Complicates interpretation** unnecessarily
- **May mask genuine model misspecification** by accommodating it

### Falsification-First Approach

For EACH model class, I define:
1. **What would make me abandon it** (falsification criteria)
2. **What would make me prefer it** (evidence thresholds)
3. **What would make me reconsider everything** (fundamental red flags)

### Expected Outcome

**Hypothesis:** Robust models will converge to standard normal models for this dataset, because:
- No outliers detected in EDA
- All observations within expected range
- Measurement error is known and accounted for
- No evidence of contamination or misspecification

**If I'm wrong:** That would be scientifically interesting and suggest EDA missed something.

---

## Model Class 1: Student-t Hierarchical Models

### Mathematical Specification

**Model 1A: Student-t Data Distribution**
```
Likelihood:
  y_i ~ StudentT(nu, theta_i, sigma_i)     [Heavy-tailed observations]

Hierarchy:
  theta_i ~ Normal(mu, tau)                 [Normal random effects]

Priors:
  mu ~ Normal(0, 20)                        [Weakly informative]
  tau ~ Half-Cauchy(0, 5)                   [Gelman recommendation]
  nu ~ Gamma(2, 0.1)                        [Degrees of freedom, mean=20]
```

**Model 1B: Student-t Random Effects**
```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)            [Normal observations]

Hierarchy:
  theta_i ~ StudentT(nu_re, mu, tau)        [Heavy-tailed effects]

Priors:
  mu ~ Normal(0, 20)
  tau ~ Half-Cauchy(0, 5)
  nu_re ~ Gamma(2, 0.1)
```

**Model 1C: Double Student-t (Full Robustness)**
```
Likelihood:
  y_i ~ StudentT(nu_y, theta_i, sigma_i)    [Heavy-tailed observations]

Hierarchy:
  theta_i ~ StudentT(nu_theta, mu, tau)     [Heavy-tailed effects]

Priors:
  mu ~ Normal(0, 20)
  tau ~ Half-Cauchy(0, 5)
  nu_y ~ Gamma(2, 0.1)
  nu_theta ~ Gamma(2, 0.1)
```

### Rationale

**Why consider Student-t at all?**
- EDA says no outliers, BUT School 1 (y=28, SE=15) has z=1.35
- With n=8, we have LOW POWER to detect tail heaviness
- Student-t nests Normal as nu → ∞, so we can test this
- If nu_posterior > 30, data favor normality

**Why three variants?**
- Model 1A: Robustness to **measurement outliers** (anomalous y_i)
- Model 1B: Robustness to **effect heterogeneity** (extreme theta_i)
- Model 1C: Robustness to **both** (maximally defensive)

### Prior Justifications

**nu ~ Gamma(2, 0.1):**
- Mean = 20, SD = 14
- Allows nu as low as 4 (very heavy tails) but centered at normal-ish range
- If data favor nu > 30, tails are effectively normal
- Juárez & Steel (2010) recommendation for model comparison

**Alternative: nu ~ Uniform(1, 100)**
- More diffuse but creates computational issues near boundaries
- Not recommended unless Gamma prior causes prior-data conflict

### Expected Behavior Given EDA

**Prediction:**
- nu_posterior will concentrate at upper end (20-40+)
- Effective sample size for nu will be LOW (parameter not identified)
- Point estimates for mu, tau, theta_i will be nearly identical to normal model
- Credible intervals may be slightly wider (cost of robustness)

**If this happens:** Evidence that normal assumptions are appropriate

**If this doesn't happen (nu_posterior < 10):**
- SERIOUS red flag - suggests EDA missed something
- Investigate: Which observations driving this?
- Reconsider: Is measurement model itself misspecified?

### Falsification Criteria

**I will abandon Student-t models if:**

1. **Posterior nu > 30 with high certainty** (95% CI > 20)
   - Evidence: Data clearly favor normality
   - Action: Revert to normal models, conclude robustness unnecessary

2. **Computational difficulties** (divergences, poor mixing)
   - Evidence: Model is overparameterized for n=8
   - Action: Focus on simpler models

3. **LOO-CV/WAIC shows no improvement** over normal model
   - Evidence: Extra complexity not justified
   - Action: Parsimony principle favors normal model

4. **nu highly correlated with tau** in posterior
   - Evidence: Non-identifiability, parameters trading off
   - Action: Model is not learning separate information

**I will prefer Student-t models if:**

1. **Posterior nu < 10 with confidence** (95% CI < 15)
   - Evidence: True heavy tails beyond normal
   - Action: Investigate source of tail heaviness

2. **LOO-CV improvement > 3 (substantial)**
   - Evidence: Better predictive performance
   - Action: Robustness is capturing real data features

3. **Posterior predictive checks show better tail coverage**
   - Evidence: Normal model underestimates tail probabilities
   - Action: Student-t provides better uncertainty quantification

### Stress Tests

**Test 1: Posterior Predictive Check**
- Generate 1000 datasets from fitted model
- Check: Does observed y_1 = 28 fall in expected range?
- If Student-t generates more extreme values → appropriate
- If Student-t identical to Normal → unnecessary

**Test 2: Leave-One-Out School 1**
- Fit both Normal and Student-t without School 1
- Predict y_1 from both models
- If Student-t prediction is closer → it handles extremes better
- If predictions identical → no benefit

**Test 3: Extreme Value Sensitivity**
- Simulate y_1 = 50 (more extreme than observed)
- Refit both models
- If Student-t is less sensitive → good robustness
- If both break equally → model misspecification is deeper

### Implementation Notes (PyMC)

```python
import pymc as pm

with pm.Model() as model_1a:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # Random effects (non-centered)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=8)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood (Student-t)
    y_obs = pm.StudentT('y_obs', nu=nu, mu=theta, sigma=sigma_data,
                         observed=y_data)

    # Sampling
    trace = pm.sample(2000, tune=2000, target_accept=0.95)
```

**Computational considerations:**
- Non-centered parameterization critical for low-n hierarchical models
- High target_accept (0.95) to avoid divergences with nu near boundary
- Monitor nu convergence carefully (may be slow)
- Consider thinning if autocorrelation is high

---

## Model Class 2: Mixture Models (Latent Outlier Detection)

### Mathematical Specification

**Model 2A: Outlier Indicator Model**
```
Likelihood:
  y_i ~ (1 - p_i) * Normal(theta_i, sigma_i) + p_i * Normal(theta_i, k * sigma_i)

where:
  p_i ~ Bernoulli(pi)                       [Is school i an outlier?]
  k > 1                                      [Inflation factor for outliers]

Hierarchy:
  theta_i ~ Normal(mu, tau)

Priors:
  mu ~ Normal(0, 20)
  tau ~ Half-Cauchy(0, 5)
  pi ~ Beta(1, 9)                           [Prior: 10% outliers]
  k ~ Gamma(4, 1)                           [Prior: mean k=4]
```

**Model 2B: Latent Class Model (Two Subgroups)**
```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)

Hierarchy:
  theta_i ~ z_i * Normal(mu_1, tau_1) + (1 - z_i) * Normal(mu_2, tau_2)

where:
  z_i ~ Bernoulli(pi)                       [Group membership]

Priors:
  mu_1, mu_2 ~ Normal(0, 20)
  tau_1, tau_2 ~ Half-Cauchy(0, 5)
  pi ~ Beta(2, 2)                           [Symmetric prior]

Constraint: mu_1 < mu_2                     [Identifiability]
```

**Model 2C: Dirichlet Process Mixture (Fully Nonparametric)**
```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)

Hierarchy:
  theta_i ~ DP(alpha, G_0)

where:
  G_0 = Normal(mu_0, tau_0)                 [Base distribution]
  alpha ~ Gamma(1, 1)                       [Concentration parameter]

Priors:
  mu_0 ~ Normal(0, 20)
  tau_0 ~ Half-Cauchy(0, 10)
```

### Rationale

**Why consider mixture models?**

CRITICAL COUNTERARGUMENT: EDA shows Q-test p=0.696, I²=0%, NO evidence of subgroups.

**But:**
- EDA tests for continuous heterogeneity, not discrete clusters
- Mann-Whitney test showed p=0.029 for median split (suggestive)
- Gap ratio = 2.5 hints at possible bimodality
- With n=8, we can't rule out 2 latent groups

**Scientific question:** Are there qualitatively different school types?
- High-performing schools (1, 7, 8, 2): y > 7
- Low-performing schools (5, 6, 3, 4): y ≤ 7

**Bayesian advantage:** Model can learn there's only ONE group (posterior collapses)

### Prior Justifications

**pi ~ Beta(1, 9) for outlier model:**
- Prior expectation: 10% outlier rate
- Skeptical prior - requires data to support outliers
- If posterior pi → 0, evidence against outliers

**pi ~ Beta(2, 2) for mixture model:**
- Symmetric, weakly informative
- Allows anywhere from 1:9 to 9:1 split
- Centers at 50:50 split (no prior bias)

**k ~ Gamma(4, 1) for inflation factor:**
- Mean = 4 (outliers have 4x variance)
- Allows k from 1.5 (mild) to 10+ (extreme)
- If posterior k → 1, no inflation needed

**alpha ~ Gamma(1, 1) for DP:**
- Mean = 1 (prior expectation: ~2 clusters for n=8)
- Allows alpha → 0 (few clusters) or alpha → 5+ (many clusters)
- Let data determine complexity

### Expected Behavior Given EDA

**Prediction for Model 2A (Outlier indicators):**
- Posterior p_i → 0 for all schools (no outliers detected)
- Posterior k → 1 (no inflation needed)
- Posterior pi → 0 (base rate goes to zero)
- Effective sample size for p_i will be LOW (not identified)

**Prediction for Model 2B (Two groups):**
- Posterior mu_1 ≈ mu_2 ≈ 7.7 (groups collapse)
- Posterior tau_1, tau_2 both → 0 (no within-group variance)
- Posterior pi → 0.5 ± 0.5 (can't identify split)
- High posterior correlation between mu_1 and mu_2

**Prediction for Model 2C (Dirichlet process):**
- Posterior will favor K=1 cluster (all schools in one group)
- Posterior alpha will be LOW (data don't support complexity)
- Effective cluster assignments will be degenerate

**If predictions hold:** Evidence that mixture structure is unnecessary

**If predictions fail (multiple distinct clusters):**
- RECONSIDER EVERYTHING - EDA badly misleading
- Investigate: What defines cluster membership?
- Domain knowledge: Are there known school characteristics?

### Falsification Criteria

**I will abandon mixture models if:**

1. **Posterior collapses to one component** (K=1 with probability > 0.9)
   - Evidence: Data don't support multiple clusters
   - Action: Revert to single hierarchical model

2. **Label switching / non-identifiability**
   - Evidence: Cannot distinguish components
   - Action: Model too complex for n=8

3. **WAIC/LOO substantially worse** than simple model (>5 difference)
   - Evidence: Overfitting penalty exceeds any benefit
   - Action: Occam's razor - use simpler model

4. **Posterior predictive checks show no improvement**
   - Evidence: Mixture not capturing real data features
   - Action: Added complexity provides no value

5. **Assigned "outliers" are arbitrary** (posterior p_i ≈ 0.5 for all)
   - Evidence: Model can't decide what's an outlier
   - Action: Signal is too weak for this approach

**I will prefer mixture models if:**

1. **Clear bimodality in posterior theta_i distribution**
   - Evidence: Two distinct effect populations
   - Action: Investigate what distinguishes groups

2. **LOO improvement > 3 AND interpretable clusters**
   - Evidence: Model finds meaningful structure
   - Action: Report cluster characteristics, validate externally

3. **Outlier probabilities are decisive** (most p_i < 0.1 or > 0.9)
   - Evidence: Clear outlier/non-outlier distinction
   - Action: Investigate identified outliers scientifically

### Stress Tests

**Test 1: Prior Predictive Check**
- Sample from prior before seeing data
- Does prior generate realistic datasets?
- If prior allows y_i = 1000, it's too diffuse
- Tune priors to be weakly informative but sensible

**Test 2: Synthetic Data with Known Structure**
- Generate data with NO clusters (tau=0)
- Fit mixture model
- Does it correctly infer K=1?
- If model hallucinates clusters → not robust

**Test 3: Synthetic Data with Two Clusters**
- Generate data with mu_1=0, mu_2=15, clear separation
- Fit mixture model
- Does it recover true K=2 and parameters?
- If it can't find obvious clusters → broken

**Test 4: Permutation Test**
- Randomly permute observed y_i
- Refit mixture model
- Does it still find "clusters"?
- If yes → clusters are artifacts

### Implementation Notes (PyMC)

```python
import pymc as pm

with pm.Model() as model_2a:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfCauchy('tau', beta=5)
    pi = pm.Beta('pi', alpha=1, beta=9)  # Outlier base rate
    k = pm.Gamma('k', alpha=4, beta=1)   # Inflation factor

    # Random effects
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=8)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Outlier indicators
    p = pm.Bernoulli('p', p=pi, shape=8)  # Per-school outlier flag

    # Mixture likelihood
    sigma_mix = sigma_data * (1 + p * (k - 1))  # Inflated if outlier
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma_mix,
                      observed=y_data)

    # Sampling
    trace = pm.sample(2000, tune=2000, target_accept=0.95)
```

**Computational warnings:**
- Discrete latent variables (p_i, z_i) require specialized samplers
- Use pm.Marginalized for efficient inference (marginalizes discrete variables)
- Label switching is a MAJOR issue - use ordered constraints or post-process
- Convergence diagnostics may be misleading with multimodality

**For Dirichlet Process:**
- PyMC doesn't have native DP - use stick-breaking approximation
- Truncate at K_max = 5 (unlikely to need more with n=8)
- Very slow sampling - consider variational inference

---

## Model Class 3: Prior Sensitivity Analysis

### Specification Framework

Rather than a single model, this is a SYSTEMATIC EXPLORATION of how conclusions change with prior choices.

**Base Model:**
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
mu ~ ???
tau ~ ???
```

**Vary systematically:**
1. Prior on mu: Information level
2. Prior on tau: Regularization strength
3. Prior on likelihood: Distributional assumptions

### Specification Grid

**Dimension 1: Prior on mu (Grand Mean)**

| Prior | Specification | Information Level | Rationale |
|-------|---------------|-------------------|-----------|
| Flat | Uniform(-1000, 1000) | Non-informative | Improper but works numerically |
| Vague | Normal(0, 100) | Very weak | Negligible prior influence |
| Weak | Normal(0, 20) | Weak | Standard choice (EDA recommendation) |
| Moderate | Normal(5, 10) | Moderate | Centered near observed mean |
| Strong | Normal(8, 5) | Strong | Tight around observed mean |
| Very Strong | Normal(8, 2) | Very strong | Nearly fixes mu |

**Expected:** Posteriors converge for Vague → Moderate (data dominate). If Strong/Very Strong differ substantially, data-prior conflict.

**Dimension 2: Prior on tau (Between-School SD)**

| Prior | Specification | Regularization | Rationale |
|-------|---------------|----------------|-----------|
| Uniform | Uniform(0, 50) | None | Flat on bounded support |
| Half-Normal-Weak | Half-Normal(0, 20) | Weak | Concentrates near 0, diffuse |
| Half-Cauchy-5 | Half-Cauchy(0, 5) | Moderate | Gelman recommendation |
| Half-Cauchy-2 | Half-Cauchy(0, 2) | Strong | More skeptical of heterogeneity |
| Exponential | Exponential(1) | Very strong | Mean=1, strongly favors low tau |
| Half-Normal-Strong | Half-Normal(0, 3) | Very strong | Tight concentration near 0 |

**Expected:** Given EDA tau_estimate=0, all priors should yield similar posteriors near 0. If Uniform prior yields tau_posterior >> 0, suggests prior was constraining (unlikely).

**Dimension 3: Observational Model**

| Model | Specification | Purpose |
|-------|---------------|---------|
| Normal-Fixed | y_i ~ Normal(theta_i, sigma_i) | Baseline (sigma_i known) |
| Normal-Uncertain | y_i ~ Normal(theta_i, sigma_i * delta_i), delta_i ~ LogNormal(0, 0.2) | SE uncertainty |
| Student-t | y_i ~ StudentT(nu, theta_i, sigma_i), nu ~ Gamma(2, 0.1) | Heavy tails |
| Skew-Normal | y_i ~ SkewNormal(alpha, theta_i, sigma_i), alpha ~ Normal(0, 2) | Asymmetry |

**Expected:** Normal-Fixed is correct per problem statement. Others should not improve fit.

### Analysis Protocol

**Step 1: Fit all combinations** (6 mu priors × 6 tau priors × 1 base model = 36 models)

**Step 2: Extract key quantities:**
- Posterior mean and SD for mu
- Posterior mean and SD for tau
- Posterior mean for theta_1 (most extreme school)
- LOO-CV score

**Step 3: Sensitivity metrics:**
```
Sensitivity(parameter, prior_dimension) =
    max(posterior_mean) - min(posterior_mean) across prior choices

Relative_Sensitivity = Sensitivity / posterior_SD
```

**Step 4: Decision-relevant quantities:**
- P(tau < 3): Probability of low heterogeneity
- P(theta_1 > 15): Probability School 1 truly exceptional
- P(effect_new_school > 10): Prediction for new school

### Expected Behavior Given EDA

**Prediction:**
- LOW sensitivity to mu prior (data are informative, pooled SE=4.07)
- LOW sensitivity to tau prior (data strongly favor tau≈0)
- All posteriors converge to similar estimates:
  - mu_posterior ≈ 7.5-8.0 ± 4
  - tau_posterior ≈ 0-3 (concentrated near 0)
  - theta_1_posterior ≈ 7.5-9.0 (strong shrinkage from y_1=28)

**What would indicate PROBLEM:**
- High sensitivity (Relative_Sensitivity > 1): Prior dominates data
- Divergent conclusions: Some priors say tau=0, others say tau=10
- Prior-data conflict: Posterior pushed away from prior AND away from data

### Falsification Criteria

**I will conclude priors are robust if:**

1. **Relative sensitivity < 0.5 for mu and tau**
   - Evidence: Data dominate prior
   - Action: Recommend weakly informative prior from grid

2. **All priors yield P(tau < 5) > 0.8**
   - Evidence: Consistent conclusion about low heterogeneity
   - Action: Conclusion is robust to prior choice

3. **Decision-relevant quantities vary by < 20%**
   - Evidence: Practical conclusions unchanged
   - Action: Report range, acknowledge uncertainty

**I will conclude priors are NOT robust if:**

1. **Relative sensitivity > 1.0 for key parameters**
   - Evidence: Prior matters more than data
   - Action: COLLECT MORE DATA or use more informative prior

2. **Different priors give opposite conclusions**
   - Evidence: Data are too weak to learn
   - Action: Report full sensitivity analysis, acknowledge uncertainty

3. **Strong prior-data conflict** (posterior far from both)
   - Evidence: Model misspecification
   - Action: Reconsider model structure entirely

### Stress Tests

**Test 1: Prior Predictive Checks**
- For each prior, generate 1000 synthetic datasets
- Check: Are observed data typical under prior?
- If prior generates y_i = 1000 routinely → too vague
- If prior never generates y_i > 20 → too informative

**Test 2: Synthetic Data with Known Truth**
- Generate data with known mu=8, tau=0
- Fit with all priors
- Check: Do all recover truth within credible intervals?
- If some priors systematically biased → problematic

**Test 3: Data Splitting**
- Hold out School 1 (most extreme)
- Fit with different priors on remaining 7 schools
- Predict y_1
- Check: Do predictions vary substantially by prior?
- If yes → prior matters for extrapolation

**Test 4: Effective Sample Size**
- Compute effective sample size for prior:
```
n_eff = (posterior_sd / prior_sd)^2
```
- If n_eff << 8, prior is dominating
- If n_eff ≈ 8, data dominating
- Goal: n_eff ≈ 8 for all reasonable priors

### Implementation Notes (PyMC)

```python
import pymc as pm
import itertools

# Define prior grids
mu_priors = {
    'flat': pm.Uniform('mu', -1000, 1000),
    'vague': pm.Normal('mu', mu=0, sigma=100),
    'weak': pm.Normal('mu', mu=0, sigma=20),
    'moderate': pm.Normal('mu', mu=5, sigma=10),
    'strong': pm.Normal('mu', mu=8, sigma=5),
    'very_strong': pm.Normal('mu', mu=8, sigma=2),
}

tau_priors = {
    'uniform': pm.Uniform('tau', 0, 50),
    'half_normal_weak': pm.HalfNormal('tau', sigma=20),
    'half_cauchy_5': pm.HalfCauchy('tau', beta=5),
    'half_cauchy_2': pm.HalfCauchy('tau', beta=2),
    'exponential': pm.Exponential('tau', lam=1),
    'half_normal_strong': pm.HalfNormal('tau', sigma=3),
}

results = {}

for mu_name, tau_name in itertools.product(mu_priors.keys(), tau_priors.keys()):
    with pm.Model() as model:
        # Priors (from grid)
        mu = mu_priors[mu_name]
        tau = tau_priors[tau_name]

        # Hierarchy (non-centered)
        theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=8)
        theta = pm.Deterministic('theta', mu + tau * theta_raw)

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma_data,
                          observed=y_data)

        # Inference
        trace = pm.sample(1000, tune=1000)
        loo = pm.loo(trace, model)

        # Store results
        results[(mu_name, tau_name)] = {
            'mu_mean': trace['mu'].mean(),
            'mu_sd': trace['mu'].std(),
            'tau_mean': trace['tau'].mean(),
            'tau_sd': trace['tau'].std(),
            'loo': loo.loo,
        }

# Analyze sensitivity
# ... (compute metrics from results dictionary)
```

**Practical considerations:**
- 36 models × 2000 iterations = 72,000 samples (manageable)
- Parallelize across prior combinations
- Use same random seed for fair comparison
- Cache results for sensitivity analysis

---

## Critical Red Flags: When to Abandon Everything

These are "circuit breakers" that would make me stop and fundamentally reconsider:

### Red Flag 1: Prior-Posterior Conflict in Multiple Models

**What it looks like:**
- Posterior is far from prior AND far from data
- High divergences across model classes
- Posterior predictive checks fail badly

**What it means:**
- Fundamental model misspecification
- Not just wrong distribution - wrong structure

**What to do:**
- Question the hierarchical assumption itself
- Consider: Are schools truly exchangeable?
- Check: Is measurement model correct? (sigma_i truly known?)
- Escalate: May need domain expert input

### Red Flag 2: Extreme Parameter Values

**What it looks like:**
- tau_posterior → 50 (huge between-school variance)
- nu_posterior → 1 (extremely heavy tails)
- k_posterior → 100 (massive outlier inflation)

**What it means:**
- Model fighting the data structure
- Trying to accommodate something it can't explain

**What to do:**
- Examine which observations driving this
- Check for data errors (typos, wrong units)
- Consider: Are sigma_i values correct?
- Reconsider: Is this really a meta-analysis problem?

### Red Flag 3: Computational Breakdown

**What it looks like:**
- >10% divergent transitions (even with high target_accept)
- Rhat > 1.1 after 10,000 iterations
- Effective sample size < 50

**What it means:**
- Geometry of posterior is pathological
- Model is too complex or misspecified

**What to do:**
- Simplify model drastically
- Check parameterization (try centered vs non-centered)
- If simple models also break → data problem, not model problem

### Red Flag 4: Inconsistent Results Across Methods

**What it looks like:**
- Normal model: mu=8, tau=0
- Student-t model: mu=15, tau=10
- Mixture model: Two groups at mu=-5 and mu=25
- ALL with good diagnostics

**What it means:**
- Models are capturing different aspects
- Each model is a different lens, seeing different patterns
- No single "truth" - model class matters enormously

**What to do:**
- Report ALL models, acknowledge uncertainty
- Don't pick "best" by WAIC - fundamental uncertainty
- Use model averaging if predictions needed
- Consider: Collect more data to discriminate

### Red Flag 5: Posterior Predictive Checks Fail Universally

**What it looks like:**
- All models predict y_i ∈ [0, 15]
- But observed y_1 = 28
- Test statistic: p_value < 0.01 for all models

**What it means:**
- No model in our class can explain the data
- Need fundamentally different approach

**What to do:**
- Question data generation process
- Could y_i and sigma_i be on different scales?
- Could there be additional covariates needed?
- Could the problem setup be wrong?

---

## Decision Framework: When Does Robustness Matter?

### Conditions Favoring Robust Models

Robust models are USEFUL when:

1. **Suspected contamination:** Some observations may be from different process
2. **Unknown unknowns:** Don't trust normality but can't specify alternative
3. **Conservative predictions:** Heavy tails give wider, safer intervals
4. **Regulatory context:** Need to protect against worst-case scenarios
5. **Sequential analysis:** Early data suggest tail behavior

### Conditions Favoring Simple Models

Simple (normal) models are BETTER when:

1. **Strong theory:** Normality follows from Central Limit Theorem
2. **Known measurement model:** sigma_i are known (as here)
3. **No empirical evidence:** EDA shows no outliers or non-normality
4. **Small sample:** Robustness adds parameters we can't estimate with n=8
5. **Interpretability matters:** Stakeholders understand normal models

### The Eight Schools Context

**For this specific dataset:**

Arguments AGAINST robustness:
- Q-test p=0.696 (no heterogeneity)
- No outliers (all |z| < 2)
- sigma_i are KNOWN (not estimated)
- Measurement model is Normal by design (test scores, CLT applies)
- n=8 is too small to estimate extra robustness parameters

Arguments FOR robustness:
- Low power to detect issues with n=8
- School 1 looks extreme (y=28 vs pooled 7.7)
- Robustness is cheap to check with modern computation
- If robust models agree → reassuring
- May reveal subtle misspecification

**My assessment:** Robustness is probably unnecessary here, but WORTH CHECKING as a sensitivity analysis.

---

## Synthesis: Recommended Analysis Strategy

### Phase 1: Baseline (Normal Models)

1. Fit standard hierarchical model (as EDA recommended)
2. Check diagnostics thoroughly
3. Establish baseline LOO-CV score
4. Save posteriors for comparison

### Phase 2: Robustness Checks (My Models)

1. Fit Student-t model (1A)
   - Check posterior nu
   - Compare to baseline
   - If nu > 30 → stop, robustness not needed

2. Fit outlier indicator model (2A)
   - Check posterior p_i and pi
   - If all p_i < 0.2 → stop, no outliers

3. Prior sensitivity (full grid)
   - Compute sensitivity metrics
   - If Relative_Sensitivity < 0.5 → robust to priors

### Phase 3: Interpretation

**Scenario A: All robust models agree with baseline**
- Conclusion: Normal assumptions justified
- Report: "Robustness checks confirm primary analysis"
- Action: Use simple model, report robustness as supplement

**Scenario B: Robust models differ from baseline**
- Conclusion: Normal assumptions questionable
- Report: "Models sensitive to distributional assumptions"
- Action: Report range, use model averaging, collect more data

**Scenario C: Red flags encountered**
- Conclusion: Fundamental model misspecification
- Report: "Data inconsistent with standard meta-analysis models"
- Action: Consult domain experts, reconsider problem setup

### Phase 4: Reporting

**What to include:**
- Primary analysis: Normal hierarchical model
- Robustness checks: Student-t and prior sensitivity
- Key comparison: nu_posterior, sensitivity metrics, LOO-CV differences
- Decision: Which model for inference? Report range if sensitive.

**What NOT to include:**
- Mixture models (unless they show clear improvement)
- Dirichlet process (too complex for n=8)
- Every prior combination (summarize with sensitivity plots)

---

## Expected Computational Requirements

| Model Class | Models | Iterations | Time (est.) | Difficulty |
|-------------|--------|------------|-------------|------------|
| Normal baseline | 1 | 2,000 | 1 min | Easy |
| Student-t (3 variants) | 3 | 2,000 each | 5 min | Medium |
| Mixture (outlier) | 1 | 4,000 | 10 min | Hard |
| Mixture (latent class) | 1 | 8,000 | 20 min | Very hard |
| Prior sensitivity (grid) | 36 | 1,000 each | 20 min | Easy |
| **Total** | **42** | **~60,000** | **~1 hour** | **Manageable** |

**Parallelization:** Can fit all 42 models in parallel → ~10-15 minutes wall time

**Bottlenecks:**
- Mixture models (label switching, discrete parameters)
- Student-t near boundary (nu → 1 is tricky)
- Prior sensitivity (many models, but each is fast)

---

## Comparison with Other Designers (Anticipated)

I expect the other two designers to focus on:
- **Designer 1:** Likely exploring hierarchical structures (varying tau priors, non-centered parameterizations)
- **Designer 2:** Likely exploring alternative model classes (GP, state-space, non-hierarchical)

**My unique contribution:**
- Systematic robustness to distributional assumptions
- Quantitative sensitivity analysis framework
- Clear falsification criteria for robustness
- Critical perspective on whether robustness is even needed

**Potential overlap:**
- All three may propose Student-t models
- All three may do prior sensitivity
- All three should reach similar conclusions (given strong EDA evidence)

**Desired outcome:**
- If all three designers converge → strong evidence
- If we diverge → reveals fundamental uncertainty
- Either outcome is scientifically valuable

---

## Falsification Summary (TL;DR)

### I will abandon Student-t models if:
- Posterior nu > 30 (clearly favor normality)
- No improvement in LOO-CV
- Computational issues

### I will abandon mixture models if:
- Posterior favors K=1 cluster
- Label switching prevents inference
- LOO-CV worse than simple model

### I will report high prior sensitivity if:
- Relative_Sensitivity > 1.0
- Different priors give contradictory conclusions

### I will abandon ALL models if:
- Prior-posterior conflict across all classes
- Extreme parameter values (tau > 50, nu < 3)
- Computational breakdown in simple models
- Posterior predictive checks fail universally

---

## Final Reflection: The Robustness Paradox

**My core tension:** The EDA is so clear that I'm designing models I expect to FAIL.

**Why this is good:**
- If they fail (converge to normal) → validates EDA
- If they succeed (find real deviations) → EDA missed something
- Either way, we learn

**The real risk:** Over-interpreting small differences between models when n=8.

**My commitment:** I will be HONEST about uncertainty and will NOT declare a robust model "better" just because it's more complex. Parsimony is a virtue.

**Success metric:** Not whether robust models are selected, but whether robustness analysis provides insight into model assumptions and sensitivity.

---

## Files and Deliverables

This document: `/workspace/experiments/designer_3/proposed_models.md`

Expected outputs (after modeling):
- `/workspace/experiments/designer_3/model_1a_student_t_data.py`
- `/workspace/experiments/designer_3/model_1b_student_t_effects.py`
- `/workspace/experiments/designer_3/model_2a_outlier_indicators.py`
- `/workspace/experiments/designer_3/prior_sensitivity_results.csv`
- `/workspace/experiments/designer_3/sensitivity_plots.png`
- `/workspace/experiments/designer_3/comparison_with_baseline.md`

---

**END OF DESIGN DOCUMENT**

*Remember: The goal is not to find robustness problems - it's to test whether robustness is needed. Negative results (no need for robustness) are just as valuable as positive results (robustness matters).*
