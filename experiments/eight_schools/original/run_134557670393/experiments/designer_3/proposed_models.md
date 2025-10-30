# Robust and Alternative Bayesian Model Proposals
## Model Designer #3: Robustness-Focused Perspective

**Designer**: Model Designer #3 (Robust Methods & Alternative Formulations)
**Date**: 2025-10-28
**Context**: Meta-analysis with J=8 studies, I²=0% but potential heterogeneity paradox
**Focus**: Robustness to outliers, model misspecification, and distributional assumptions

---

## Executive Summary

This document proposes **3 distinct Bayesian model classes** from a robustness perspective:

1. **Robust Meta-Analysis with Student-t Likelihood** - Heavy-tailed distributions to handle potential outliers
2. **Finite Mixture Meta-Analysis** - Explicit subgroup modeling for observed clustering
3. **Uncertainty-Inflated Meta-Analysis** - Accounts for potential underestimation of reported standard errors

Each model addresses specific concerns raised by the EDA:
- Study 1 (y=28) is highly influential despite not being a statistical outlier
- I²=0% may be artifact of low power ("heterogeneity paradox")
- Potential clustering into high-effect and low-effect groups
- Reported standard errors assumed known but may themselves be uncertain

**Critical Philosophy**: These models are designed to **fail informatively**. Each has explicit falsification criteria that would tell us to abandon it.

---

## Model 1: Robust Meta-Analysis with Student-t Likelihood

### 1.1 Model Name and Class
**Name**: Robust Student-t Meta-Analysis (RSTMA)
**Class**: Hierarchical Bayesian meta-analysis with heavy-tailed likelihood
**Key Feature**: Robustness to outliers and distributional misspecification

### 1.2 Mathematical Specification

**Likelihood** (heavy-tailed):
```
y_i ~ Student-t(nu, theta_i, sigma_i)    for i = 1, ..., 8
```
Where:
- `y_i` = observed effect size for study i
- `theta_i` = true underlying effect for study i
- `sigma_i` = known standard error (FIXED, from data)
- `nu` = degrees of freedom (controls tail heaviness)

**Hierarchical Structure**:
```
theta_i ~ Normal(mu, tau)                 for i = 1, ..., 8
```
Where:
- `mu` = overall population mean effect
- `tau` = between-study heterogeneity (SD)

**Priors**:
```
mu ~ Normal(0, 25)                        # Weakly informative on effect
tau ~ Half-Cauchy(0, 5)                   # Standard for meta-analysis (Gelman et al. 2006)
nu ~ Gamma(2, 0.1)                        # Weakly informative, allows nu ∈ [1, ∞)
```

**Alternative parameterization for nu** (more interpretable):
```
nu ~ Shifted-Exponential(1) + 1           # Support on (1, ∞), mode near 2-3
```

### 1.3 Prior Justification

**mu ~ Normal(0, 25)**:
- Centered at zero (no prior belief about direction)
- SD = 25 covers observed range (-3 to 28) with ~68% prior mass in [-25, 25]
- Less diffuse than EDA recommendation (SD=50) - moderately informative
- Rationale: Effects outside [-50, 50] would be clinically implausible in most contexts

**tau ~ Half-Cauchy(0, 5)**:
- Standard recommendation for meta-analysis (Gelman, 2006)
- Heavy right tail allows large heterogeneity if supported by data
- Mode at zero (consistent with I²=0% finding)
- Scale=5 is conservative for small sample sizes
- Reference: Gelman A. "Prior distributions for variance parameters in hierarchical models." Bayesian Analysis 1(3):515-534, 2006.

**nu ~ Gamma(2, 0.1)**:
- Shape=2, rate=0.1 gives mean=20, SD=14.1
- Allows anything from nu≈1 (very heavy tails) to nu>30 (near-Normal)
- Mode at nu≈10 (moderately heavy tails)
- Reference: Juarez MA, Steel MF. "Model-based clustering of non-Gaussian panel data based on skew-t distributions." Journal of Business & Economic Statistics 28(1):52-66, 2010.
- Justification: We expect some tail heaviness (Study 1 influence) but not extreme

**Why Student-t?**:
- Normal likelihood assumes no outliers - violated if Study 1 is more extreme than expected
- Student-t is a mixture of Normals with different variances - adaptively downweights extreme observations
- As nu→∞, Student-t→Normal (model nests the standard approach)
- Low nu (2-5) = heavy tails, robust to outliers
- High nu (>30) = essentially Normal, data says "no robustification needed"

### 1.4 What This Model Captures

**Primary patterns addressed**:
1. **Potential outliers**: Study 1 (y=28, z=1.87) is influential but not statistical outlier
   - Student-t automatically downweights if too extreme relative to other studies
   - Magnitude of downweighting learned from data via nu parameter

2. **Distributional uncertainty**: We assumed y_i ~ Normal but never tested this rigorously
   - Student-t relaxes normality assumption
   - If nu posterior is low (2-10), suggests tail heaviness was important
   - If nu posterior is high (>30), suggests Normal was adequate

3. **Robust heterogeneity estimation**: tau estimation less sensitive to extreme values
   - In Normal models, outliers inflate tau estimates
   - Student-t separates "true heterogeneity" from "outlier-induced variation"

4. **Conservative inference**: Wider credible intervals in presence of anomalies
   - Automatically increases uncertainty when data doesn't fit clean Normal story

**What it does NOT capture**:
- Subgroup structure (assumes single distribution)
- Uncertainty in sigma_i (still treats them as known)
- Non-independence between studies (e.g., nested in labs)

### 1.5 Falsification Criteria

**I will abandon this model if:**

1. **nu posterior concentrates strongly at upper bound** (nu > 50):
   - **Evidence**: Posterior P(nu > 50 | data) > 0.8
   - **Interpretation**: Student-t converges to Normal; added complexity unnecessary
   - **Action**: Revert to Normal likelihood (simpler model)
   - **Why this matters**: Parsimony principle - don't use complex model if simple one fits

2. **Posterior predictive checks fail systematically**:
   - **Evidence**: Observed y values fall outside 95% posterior predictive intervals for >2 studies (25%)
   - **Interpretation**: Even heavy tails can't accommodate the data structure
   - **Action**: Consider mixture model (Model 2) or structural misspecification

3. **Prior-posterior conflict on nu**:
   - **Evidence**: Posterior mean for nu < 1.5 (extreme tails) with tight CIs
   - **Interpretation**: Single Student-t insufficient; may need mixture or discrete outlier model
   - **Action**: Switch to contamination model (Normal + discrete outlier component)

4. **LOO diagnostics show poor fit**:
   - **Evidence**: Pareto k > 0.7 for >2 studies (indicates poor leave-one-out predictions)
   - **Interpretation**: Model misspecified, not just outlier issue
   - **Action**: Investigate mixture model or structural changes

5. **Unreasonable shrinkage patterns**:
   - **Evidence**: Study 1 estimate shrinks <10% toward pooled mean (insufficient robustification)
   - **OR**: Study 1 estimate shrinks >90% (over-robustification, losing information)
   - **Interpretation**: nu parameter not calibrating outlier detection appropriately
   - **Action**: Consider fixed-nu versions or different robust families (Laplace, logistic)

6. **Tau posterior unchanged from prior**:
   - **Evidence**: Posterior for tau visually identical to prior (KL divergence < 0.1)
   - **Interpretation**: Data too weak to inform heterogeneity; J=8 insufficient
   - **Action**: Use fixed-effect model or informative tau prior from literature

**Red flags during fitting**:
- Divergent transitions concentrated in nu parameter → reparameterization needed
- Effective sample size for nu < 100 → poor mixing, may need stronger prior
- R-hat for any parameter > 1.01 → non-convergence, longer chains or reparameterization

### 1.6 Implementation Notes

**Platform**: Stan (via CmdStanPy) - preferred for complex Student-t parameterizations

**Stan code structure**:
```stan
data {
  int<lower=1> J;              // Number of studies
  vector[J] y;                 // Observed effects
  vector<lower=0>[J] sigma;    // Known standard errors
}

parameters {
  real mu;                     // Overall mean
  real<lower=0> tau;           // Between-study SD
  vector[J] theta_raw;         // Non-centered parameterization
  real<lower=1> nu;            // Degrees of freedom
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;  // Study-specific effects
}

model {
  // Priors
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  nu ~ gamma(2, 0.1);
  theta_raw ~ normal(0, 1);    // Non-centered parameterization

  // Likelihood (heavy-tailed)
  y ~ student_t(nu, theta, sigma);
}

generated quantities {
  vector[J] y_rep;             // Posterior predictive
  vector[J] log_lik;           // For LOO-CV

  for (j in 1:J) {
    y_rep[j] = student_t_rng(nu, theta[j], sigma[j]);
    log_lik[j] = student_t_lpdf(y[j] | nu, theta[j], sigma[j]);
  }
}
```

**Key implementation considerations**:

1. **Non-centered parameterization**: Essential for small J (avoids divergent transitions)
   - Instead of `theta ~ normal(mu, tau)`, use `theta = mu + tau * theta_raw` where `theta_raw ~ normal(0,1)`
   - Critical for efficient sampling when tau near zero

2. **nu parameterization**: May need log(nu-1) transform for better sampling
   - If nu sampling is inefficient, use `nu_raw ~ normal(0, 1)` and `nu = exp(nu_raw) + 1`

3. **Initialization**:
   - Initialize nu between 5-10 (avoid extreme starting values)
   - Initialize theta near observed y values
   - Initialize mu at median(y)

4. **Sampling settings**:
   - Chains: 4
   - Warmup: 2000 (longer for complex models)
   - Sampling: 2000 per chain
   - Target acceptance: 0.95 (higher for complex geometries)

5. **Computational expectations**:
   - Runtime: ~2-5 minutes for this model (8 studies, simple structure)
   - May see divergent transitions if tau→0 and using centered parameterization
   - nu parameter may show slower mixing (ESS ~400-800)

**Alternative in PyMC**:
```python
with pm.Model() as rstma:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=25)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # Hierarchical structure (non-centered)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y_obs = pm.StudentT('y_obs', nu=nu, mu=theta, sigma=sigma, observed=y)
```

### 1.7 Expected Challenges

**1. Identifiability between nu and tau**:
- **Issue**: Heavy tails (low nu) can mimic heterogeneity (high tau)
- **Evidence**: Strong negative correlation between nu and tau posteriors (r < -0.7)
- **Mitigation**: With J=8, may not be resolvable; report joint posterior
- **Decision rule**: If correlation < -0.8, report uncertainty about source of variation

**2. Weak information for nu with J=8**:
- **Issue**: Need many observations to estimate tail behavior reliably
- **Evidence**: Wide nu posterior (95% CI spans >20 df units)
- **Mitigation**: Prior becomes influential; sensitivity analysis essential
- **Decision rule**: If posterior/prior overlap >80%, data insufficient for nu

**3. Computational geometry**:
- **Issue**: Student-t likelihood creates challenging posterior geometry
- **Evidence**: Divergent transitions, low ESS for nu
- **Mitigation**: Non-centered parameterization, longer warmup, higher adapt_delta
- **Decision rule**: If divergences persist after adapt_delta=0.99, reparameterize

**4. Interpretation ambiguity**:
- **Issue**: Low nu could mean "outliers" OR "wrong model structure"
- **Evidence**: nu < 5 with poor predictive performance
- **Mitigation**: Posterior predictive checks, compare to mixture model
- **Decision rule**: If PPC fails despite low nu, structural issue not outliers

**5. Prior sensitivity for nu**:
- **Issue**: J=8 provides weak data; nu prior matters
- **Evidence**: Change in prior changes posterior median by >5 df
- **Mitigation**: Fit with multiple nu priors (Gamma(2,0.1), Gamma(4,0.2), Exponential(0.1)+1)
- **Decision rule**: If posterior median varies >30% across reasonable priors, report sensitivity

**6. Small sample bias**:
- **Issue**: Student-t ML estimates biased in small samples
- **Evidence**: Posterior mean(nu) systematically lower than simulation truth
- **Mitigation**: Use Bayesian framework (already doing); compare to simulations
- **Decision rule**: Run parameter recovery study before trusting nu estimates

---

## Model 2: Finite Mixture Meta-Analysis

### 2.1 Model Name and Class
**Name**: Two-Component Mixture Meta-Analysis (TMMA)
**Class**: Finite mixture model with latent group allocation
**Key Feature**: Explicitly models potential subgroup structure in effects

### 2.2 Mathematical Specification

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i)           for i = 1, ..., 8
```

**Latent Group Structure**:
```
theta_i ~ pi * Normal(mu_1, tau_1) + (1-pi) * Normal(mu_2, tau_2)
```
Equivalently (indicator variable formulation):
```
z_i ~ Bernoulli(pi)                      # Latent group indicator (0 or 1)
theta_i | z_i=0 ~ Normal(mu_1, tau_1)    # "Low effect" group
theta_i | z_i=1 ~ Normal(mu_2, tau_2)    # "High effect" group
```

**Constraint for identifiability**:
```
mu_1 < mu_2                              # Order constraint (low group < high group)
```

**Priors**:
```
# Group means (ordered)
mu_1 ~ Normal(0, 20)
mu_2 ~ Normal(10, 20)                    # Encourage separation
constraint: mu_1 < mu_2

# Within-group heterogeneity
tau_1 ~ Half-Cauchy(0, 5)
tau_2 ~ Half-Cauchy(0, 5)

# Mixture proportion
pi ~ Beta(2, 2)                          # Slight preference for balanced groups
```

### 2.3 Prior Justification

**Why mixture model?**
- EDA clustering analysis suggested potential subgroups (p=0.009)
- Visual inspection shows possible high-effect group (Studies 1, 7, 8) vs low-effect group (Studies 2-6)
- I²=0% globally but may hide within-group homogeneity and between-group heterogeneity
- Small-study effects or publication bias could create artificial groups

**mu_1 ~ Normal(0, 20), mu_2 ~ Normal(10, 20) with mu_1 < mu_2**:
- Ordered to ensure identifiability (avoid label switching)
- mu_1 centered at 0, mu_2 at 10 (encourages ~10-point separation)
- SD=20 is weakly informative (allows observed range)
- Reference: Frühwirth-Schnatter S. "Finite Mixture and Markov Switching Models." Springer, 2006.

**tau_1, tau_2 ~ Half-Cauchy(0, 5)**:
- Same prior for both groups (no assumption one is more variable)
- Allows tau_k→0 (homogeneous within group) or tau_k>0 (heterogeneous within group)
- Scale=5 standard for meta-analysis

**pi ~ Beta(2, 2)**:
- Symmetric prior, mild concentration near pi=0.5
- Mean = 0.5, SD = 0.22
- Allows anything from pi≈0.05 to pi≈0.95
- More informative alternatives: Beta(1, 1) = Uniform (less informative), Beta(5, 5) (stronger pull to balanced)
- Reference: Gelman et al. "Bayesian Data Analysis" (3rd ed), p.41

**Why this prior for pi?**
- We don't know if groups exist or their sizes
- Beta(2,2) is skeptical of extreme mixing proportions (pi<0.1 or >0.9)
- With J=8, very small groups (pi<0.15 → 1-2 studies) unlikely to be identifiable

### 2.4 What This Model Captures

**Primary patterns addressed**:

1. **Subgroup structure**: Clustering analysis in EDA suggested groups (p=0.009)
   - Potential "high-effect" studies: 1 (y=28), 7 (y=18), 8 (y=12)
   - Potential "low-effect" studies: 2-6 (y=-3 to 8)
   - Mixture model explicitly tests if this clustering is real

2. **Heterogeneity paradox resolution**: I²=0% may hide structure
   - Global heterogeneity (τ) conflates within-group and between-group variation
   - If groups exist: within-group τ_k may be ≈0, but μ_1 ≠ μ_2 (explains spread)
   - Could reconcile "no heterogeneity" with "wide range of effects"

3. **Influential study explanation**: Study 1 may belong to different population
   - Instead of "outlier", Study 1 is representative of high-effect group
   - Principled way to handle influence without arbitrary exclusion

4. **Publication bias mechanism**: If bias exists, could create bimodality
   - Published studies from two distributions (null effects suppressed, large effects published)
   - Mixture detects this structure

**What it does NOT capture**:
- Continuous moderators (if groups defined by covariate, need meta-regression)
- More than 2 groups (could extend to K-component mixture)
- Temporal trends (if groups = early vs late studies)
- Robust to outliers (uses Normal likelihood; could combine with Student-t)

### 2.5 Falsification Criteria

**I will abandon this model if:**

1. **Posterior assigns all studies to one group** (degenerate solution):
   - **Evidence**: Posterior P(pi < 0.1 or pi > 0.9) > 0.8
   - **Interpretation**: Data doesn't support two groups; single-distribution model sufficient
   - **Action**: Revert to standard hierarchical model (normal or Student-t)
   - **Why this matters**: Mixture adds complexity only if groups are meaningful

2. **Group means not separated** (mu_1 ≈ mu_2):
   - **Evidence**: Posterior P(|mu_2 - mu_1| < 5) > 0.7
   - **OR**: 95% credible intervals for mu_1 and mu_2 overlap >70%
   - **Interpretation**: Two groups not distinguishable; mixture unnecessary
   - **Action**: Fit simpler single-population model

3. **High within-group heterogeneity** (tau_k large):
   - **Evidence**: Posterior median(tau_1) > 10 OR median(tau_2) > 10
   - **Interpretation**: Groups not internally homogeneous; mixture doesn't explain structure
   - **Action**: Try continuous moderator (meta-regression) or normal hierarchical model

4. **Group assignments highly uncertain** (label switching or diffuse posteriors):
   - **Evidence**: For each study, P(z_i=1 | data) ∈ [0.3, 0.7] for >4 studies (>50%)
   - **Interpretation**: Cannot reliably assign studies to groups; mixture unidentified
   - **Action**: J=8 insufficient for mixture; use simpler model

5. **Poor model fit despite two components**:
   - **Evidence**: LOO-CV worse than single-population model (elpd difference < -2 SE)
   - **Interpretation**: Mixture structure not improving predictions; overfitting
   - **Action**: Use simpler model with better predictive performance

6. **Posterior predictive checks fail**:
   - **Evidence**: Mixture-predicted bimodality not seen in data
   - **OR**: Observed effects in tails of both component distributions
   - **Interpretation**: Two-component Normal mixture wrong structure
   - **Action**: Try non-parametric mixture, Student-t, or structural change

**Red flags during fitting**:
- Label switching (mu_1 and mu_2 swap mid-chain) → ordering constraint needed
- Divergent transitions → reparameterize mixture (use stick-breaking or non-centered)
- Multimodal posterior → multiple local optima; run many chains from different inits

### 2.6 Implementation Notes

**Platform**: Stan (preferred) or PyMC
**Challenge**: Finite mixtures are computationally demanding in HMC

**Stan code structure** (marginalized version for efficiency):
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  ordered[2] mu;               // Ordered means (mu[1] < mu[2])
  real<lower=0> tau_1;
  real<lower=0> tau_2;
  real<lower=0,upper=1> pi;    // Mixing proportion
  vector[J] theta_raw_1;       // Non-centered for group 1
  vector[J] theta_raw_2;       // Non-centered for group 2
}

transformed parameters {
  vector[J] theta_1 = mu[1] + tau_1 * theta_raw_1;
  vector[J] theta_2 = mu[2] + tau_2 * theta_raw_2;
}

model {
  // Priors
  mu[1] ~ normal(0, 20);
  mu[2] ~ normal(10, 20);
  tau_1 ~ cauchy(0, 5);
  tau_2 ~ cauchy(0, 5);
  pi ~ beta(2, 2);
  theta_raw_1 ~ normal(0, 1);
  theta_raw_2 ~ normal(0, 1);

  // Marginalized mixture likelihood
  for (j in 1:J) {
    target += log_mix(pi,
                      normal_lpdf(y[j] | theta_1[j], sigma[j]),
                      normal_lpdf(y[j] | theta_2[j], sigma[j]));
  }
}

generated quantities {
  vector[J] z_prob;            // Posterior group assignment probability
  vector[J] log_lik;

  for (j in 1:J) {
    // Posterior probability of group 2
    real lp1 = log(1-pi) + normal_lpdf(y[j] | theta_1[j], sigma[j]);
    real lp2 = log(pi) + normal_lpdf(y[j] | theta_2[j], sigma[j]);
    z_prob[j] = exp(lp2 - log_sum_exp(lp1, lp2));

    // Log-likelihood for LOO
    log_lik[j] = log_mix(pi,
                         normal_lpdf(y[j] | theta_1[j], sigma[j]),
                         normal_lpdf(y[j] | theta_2[j], sigma[j]));
  }
}
```

**Key implementation considerations**:

1. **Marginalization**: Integrate out discrete z_i indicators (more efficient than discrete sampling)
   - Use `log_mix()` function in Stan
   - Compute posterior P(z_i=1|data) in generated quantities

2. **Ordering constraint**: Use `ordered[2] mu` to avoid label switching
   - Forces mu[1] < mu[2]
   - Still need to check if constraint binds (suggests groups not separated)

3. **Initialization**:
   - K-means or hierarchical clustering on y to initialize group assignments
   - Initialize mu[1] = mean(low group), mu[2] = mean(high group)
   - Avoid initializing near prior mean (can trap sampler)

4. **Sampling settings**:
   - Chains: 4
   - Warmup: 3000 (mixtures need longer warmup)
   - Sampling: 2000 per chain
   - Target acceptance: 0.95
   - May need max_treedepth = 12 or 13

5. **Computational expectations**:
   - Runtime: ~5-15 minutes (mixtures are slower)
   - May see divergent transitions if groups not well-separated
   - ESS typically lower for mixture models (aim for ESS > 200)

**Alternative in PyMC** (using mixture distribution):
```python
with pm.Model() as tmma:
    # Priors
    mu = pm.Normal('mu', mu=[0, 10], sigma=20, shape=2)
    tau = pm.HalfCauchy('tau', beta=5, shape=2)
    pi = pm.Beta('pi', alpha=2, beta=2)

    # Ensure ordering (post-hoc transformation)
    mu_ordered = pm.Deterministic('mu_ordered', pm.math.sort(mu))

    # Mixture components
    components = [
        pm.Normal.dist(mu=mu_ordered[0], sigma=tau[0]),
        pm.Normal.dist(mu=mu_ordered[1], sigma=tau[1])
    ]

    # Mixture for theta
    theta = pm.Mixture('theta', w=[1-pi, pi], comp_dists=components, shape=J)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)
```

### 2.7 Expected Challenges

**1. Identifiability with J=8**:
- **Issue**: Estimating 6 parameters (mu_1, mu_2, tau_1, tau_2, pi) + 8 theta_i with only 8 observations
- **Evidence**: Wide posteriors, prior sensitivity
- **Mitigation**: Strong ordering constraint, informative priors
- **Decision rule**: If any parameter posterior overlaps prior >70%, model too complex

**2. Label switching** (even with ordering):
- **Issue**: MCMC chains switch between labeling conventions
- **Evidence**: Trace plots show alternating patterns for mu[1] and mu[2]
- **Mitigation**: Post-hoc relabeling, ordered parameter declaration
- **Decision rule**: If label switching detected, re-run with stronger constraint

**3. Multimodal posterior**:
- **Issue**: Multiple equally plausible group configurations
- **Evidence**: Different chains converge to different posteriors (R-hat > 1.1)
- **Mitigation**: Run 8-10 chains from diverse initializations
- **Decision rule**: If R-hat > 1.05 after convergence, report multimodality

**4. Overfitting risk**:
- **Issue**: 2-component mixture always fits better in-sample than 1-component
- **Evidence**: LOO-CV penalizes mixture heavily (high p_loo)
- **Mitigation**: Use LOO-CV not raw log-likelihood for model comparison
- **Decision rule**: Trust LOO over in-sample fit; if LOO worse, don't use mixture

**5. Computational inefficiency**:
- **Issue**: HMC struggles with discrete-like posteriors (z_i near 0 or 1)
- **Evidence**: Low ESS (<100), divergent transitions
- **Mitigation**: Marginalize out z_i, use non-centered parameterization
- **Decision rule**: If ESS < 100 after long sampling, model not feasible

**6. Interpretation challenges**:
- **Issue**: Even if groups detected, unclear what they represent
- **Evidence**: No covariates to explain group membership
- **Mitigation**: Post-hoc exploration - do groups differ in sigma_i, study ID, etc?
- **Decision rule**: If groups detected but unexplainable, treat as nuisance structure

**7. Sensitivity to prior on pi**:
- **Issue**: With J=8, pi weakly identified
- **Evidence**: Posterior(pi) similar across Beta(1,1), Beta(2,2), Beta(5,5) priors
- **Mitigation**: Sensitivity analysis with multiple pi priors
- **Decision rule**: If posterior P(pi > 0.5) changes by >0.3 across reasonable priors, inconclusive

---

## Model 3: Uncertainty-Inflated Meta-Analysis

### 3.1 Model Name and Class
**Name**: Hierarchical Bayesian Meta-Analysis with Uncertainty Inflation (UIMA)
**Class**: Meta-analysis with stochastic measurement error
**Key Feature**: Accounts for potential underestimation of reported standard errors

### 3.2 Mathematical Specification

**Core Insight**: Standard meta-analysis assumes sigma_i are KNOWN exactly. But:
- These are themselves estimates from primary studies
- May be systematically underestimated (optimistic SE estimates common in literature)
- Accounting for this uncertainty provides robustness

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i * lambda)   for i = 1, ..., 8
```
Where:
- `lambda >= 1` = inflation factor (multiplier on reported SEs)
- `lambda = 1` → trust reported SEs exactly (standard model)
- `lambda > 1` → reported SEs underestimated by factor lambda

**Hierarchical Structure**:
```
theta_i ~ Normal(mu, tau)                 for i = 1, ..., 8
```

**Priors**:
```
mu ~ Normal(0, 25)                        # Overall mean effect
tau ~ Half-Cauchy(0, 5)                   # Between-study SD
lambda ~ Log-Normal(0, 0.5)               # Inflation factor (median=1, allows >1)
```

**Alternative formulation** (more flexible per-study inflation):
```
y_i ~ Normal(theta_i, sigma_i * lambda_i)
lambda_i ~ Log-Normal(mu_lambda, sigma_lambda)
mu_lambda ~ Normal(0, 0.3)                # Centered at log(1)=0
sigma_lambda ~ Half-Normal(0, 0.5)        # Variation in inflation across studies
```

### 3.3 Prior Justification

**Why uncertainty inflation?**
- Meta-analysis typically treats sigma_i as known, but they're estimates
- Particularly relevant when primary studies have small samples or questionable methodology
- EDA shows sigma_i range 9-18; if these are ML estimates, they have ~10-20% uncertainty themselves
- Inflating SEs provides robustness to SE misspecification

**lambda ~ Log-Normal(0, 0.5)**:
- Log-Normal ensures lambda > 0
- Median = 1 (neutral: trust reported SEs by default)
- Mean ≈ 1.13 (slight inflation on average)
- 95% prior interval: [0.37, 2.72] (allows 0.4x to 3x reported SE)
- SD = 0.5 on log scale is moderately informative
- Reference: Turner RM et al. "Predictive distributions for between-study heterogeneity and simple methods for their application in Bayesian meta-analysis." Statistics in Medicine 34(6):984-998, 2015.

**Justification for lambda**:
- Primary studies may report SE_i but not degrees of freedom
- If study i had small sample (n<30), SE_i estimate uncertain
- Systematic underestimation documented in psychology/medicine literature
- Reference: Simmons JP et al. "False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant." Psychological Science 22(11):1359-1366, 2011.

**When is this model appropriate?**
1. No access to primary study raw data (can't verify SE calculations)
2. Heterogeneous study designs (some may have poor SE estimation)
3. Concern about publication bias via SE manipulation
4. As robustness check even if SEs believed accurate

**mu ~ Normal(0, 25)**:
- Weakly informative on effect size
- Consistent with other models for comparability

**tau ~ Half-Cauchy(0, 5)**:
- Standard meta-analysis prior
- Note: tau and lambda partially trade off (both explain variation)

### 3.4 What This Model Captures

**Primary patterns addressed**:

1. **Measurement error uncertainty**: sigma_i are estimates, not truth
   - In standard model, treating sigma_i as known underestimates total uncertainty
   - This model adds uncertainty layer: y_i ~ Normal(theta_i, sigma_i) but sigma_i uncertain
   - Equivalent to saying "reported SEs may be too small"

2. **Robustness to SE misspecification**:
   - If some studies underreported SEs (e.g., didn't account for clustering), lambda compensates
   - Prevents overconfident conclusions based on potentially optimistic SEs

3. **Conservative inference**:
   - By allowing lambda > 1, model widens credible intervals appropriately
   - Particularly important for borderline significance (pooled p≈0.05 in EDA)
   - If lambda posterior > 1, suggests reported SEs were indeed too small

4. **Heterogeneity vs measurement error trade-off**:
   - Standard model: all variation beyond sigma_i attributed to tau (heterogeneity)
   - This model: variation can be explained by lambda (SE underestimation) OR tau
   - Helps separate "studies differ" from "measurement more uncertain than reported"

**Interpretation of lambda posterior**:

- **lambda ≈ 1**: Reported SEs adequate, standard model sufficient
- **lambda = 1.2-1.5**: Mild underestimation (10-50% inflation needed)
- **lambda > 1.5**: Substantial underestimation, serious concerns about reported SEs
- **lambda < 1**: Over-reported SEs (rare; would be surprising)

**What it does NOT capture**:
- Outliers (uses Normal likelihood)
- Subgroup structure
- Publication bias (unless manifested as SE manipulation)
- Study-specific SE issues (global lambda assumes all studies equally affected)

### 3.5 Falsification Criteria

**I will abandon this model if:**

1. **lambda posterior concentrates at 1** (no inflation needed):
   - **Evidence**: Posterior P(0.95 < lambda < 1.05) > 0.7
   - **Interpretation**: Reported SEs are adequate; added complexity unnecessary
   - **Action**: Revert to standard model with fixed sigma_i
   - **Why this matters**: Parsimony - don't model uncertainty that isn't there

2. **lambda posterior concentrates at upper prior bound** (extreme inflation):
   - **Evidence**: Posterior median(lambda) > 2.5 with 95% CI excluding 2.0
   - **Interpretation**: Model trying to inflate SEs excessively; likely structural misspecification
   - **Action**: Investigate outliers (Student-t) or mixture models
   - **Why this matters**: lambda > 2 means "reported SEs 50% too small" - implausible for all studies

3. **Negative correlation between lambda and tau**:
   - **Evidence**: Posterior corr(lambda, tau) < -0.8
   - **Interpretation**: Model parameters trading off (identifiability issue)
   - **Action**: Cannot distinguish heterogeneity from SE inflation; model not identified
   - **Decision**: Report joint posterior and acknowledge non-identifiability

4. **No improvement in predictive performance**:
   - **Evidence**: LOO-CV worse than standard model (elpd difference < -1 SE)
   - **Interpretation**: SE inflation not supported by predictive checks
   - **Action**: Use standard fixed-sigma model

5. **Prior-posterior overlap for lambda**:
   - **Evidence**: KL divergence between prior and posterior < 0.1
   - **Interpretation**: Data provides no information about lambda; prior dominates
   - **Action**: J=8 insufficient to estimate lambda; fix lambda=1 or use informative prior

6. **Posterior predictive checks fail**:
   - **Evidence**: Observed y values outside 95% posterior predictive intervals for >25% of studies
   - **Interpretation**: Even with SE inflation, model doesn't fit
   - **Action**: Structural issue, not just SE problem; try mixture or Student-t

**Red flags during fitting**:
- Wide lambda posterior (95% CI spans [0.5, 3]) → weakly identified
- Divergent transitions → may need to reparameterize (log-lambda scale)
- lambda and tau highly correlated → non-identifiability

### 3.6 Implementation Notes

**Platform**: Stan (preferred) or PyMC

**Stan code structure**:
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;    // Reported SEs
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0> lambda;        // SE inflation factor
  vector[J] theta_raw;
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
  vector[J] sigma_inflated = sigma * lambda;  // Inflated SEs
}

model {
  // Priors
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  lambda ~ lognormal(0, 0.5);  // Median=1, allows >1
  theta_raw ~ normal(0, 1);

  // Likelihood with inflated SEs
  y ~ normal(theta, sigma_inflated);
}

generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real inflation_percent = (lambda - 1) * 100;  // Percent inflation

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma_inflated[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma_inflated[j]);
  }
}
```

**Alternative: Per-study inflation** (more flexible but harder to identify):
```stan
parameters {
  real mu;
  real<lower=0> tau;
  real mu_lambda;              // Mean log-inflation
  real<lower=0> sigma_lambda;  // SD of log-inflation across studies
  vector[J] lambda_raw;        // Raw inflation per study
  vector[J] theta_raw;
}

transformed parameters {
  vector[J] lambda = exp(mu_lambda + sigma_lambda * lambda_raw);  // Study-specific
  vector[J] theta = mu + tau * theta_raw;
  vector[J] sigma_inflated = sigma .* lambda;  // Element-wise multiplication
}

model {
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  mu_lambda ~ normal(0, 0.3);       // Prior median lambda = exp(0) = 1
  sigma_lambda ~ normal(0, 0.5);
  lambda_raw ~ normal(0, 1);
  theta_raw ~ normal(0, 1);

  y ~ normal(theta, sigma_inflated);
}
```

**Key implementation considerations**:

1. **Non-centered parameterization**: Essential for both theta and lambda_i
   - Use theta_raw ~ Normal(0,1) and theta = mu + tau*theta_raw
   - Use lambda_raw ~ Normal(0,1) and lambda = exp(mu_lambda + sigma_lambda*lambda_raw)

2. **Parameter scaling**:
   - lambda on log scale for computational stability
   - Report exp(lambda) - 1 as "percent inflation" for interpretability

3. **Identifiability check**:
   - Monitor correlation between lambda and tau
   - If |corr| > 0.8, report joint uncertainty
   - Consider fixing one if both not identifiable

4. **Initialization**:
   - lambda = 1 (neutral starting point)
   - theta near observed y
   - mu at median(y)

5. **Sampling settings**:
   - Chains: 4
   - Warmup: 2000
   - Sampling: 2000 per chain
   - Target acceptance: 0.90

6. **Computational expectations**:
   - Runtime: ~2-4 minutes (similar to standard model)
   - Should converge smoothly if properly parameterized
   - Watch for funnel geometry if tau→0 and lambda not centered

**PyMC implementation**:
```python
with pm.Model() as uima:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=25)
    tau = pm.HalfCauchy('tau', beta=5)
    lambda_log = pm.Normal('lambda_log', mu=0, sigma=0.5)
    lambda_param = pm.Deterministic('lambda', pm.math.exp(lambda_log))

    # Hierarchical structure
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Inflated SEs
    sigma_inflated = sigma * lambda_param

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma_inflated, observed=y)
```

### 3.7 Expected Challenges

**1. Identifiability between lambda and tau**:
- **Issue**: Both parameters explain variability beyond sampling error
- **Evidence**: Posterior correlation |r| > 0.7
- **Mitigation**: With J=8, may not separate; consider fixing one
- **Decision rule**: If corr > 0.8, report "cannot distinguish SE underestimation from heterogeneity"

**2. Weak data for lambda estimation**:
- **Issue**: Need large J to estimate global inflation factor reliably
- **Evidence**: Wide lambda posterior (95% CI includes 1 by wide margin)
- **Mitigation**: Use as robustness check, not primary inference
- **Decision rule**: If posterior width > 1.5 on lambda scale, data insufficient

**3. Prior sensitivity**:
- **Issue**: With J=8, prior on lambda matters
- **Evidence**: Posterior mean(lambda) changes >20% across reasonable priors
- **Mitigation**: Sensitivity analysis with Log-Normal(0, 0.3), Log-Normal(0, 0.5), Log-Normal(0, 1)
- **Decision rule**: If inference about mu changes qualitatively across lambda priors, inconclusive

**4. Interpretation ambiguity**:
- **Issue**: lambda > 1 could mean "SEs underreported" OR "extra variation from other sources"
- **Evidence**: Cannot distinguish without external validation
- **Mitigation**: Treat as robustness parameter, not causal inference
- **Decision rule**: Report as "effective SE inflation for robust inference"

**5. Extreme inflation suggests wrong model**:
- **Issue**: lambda > 2 implies 100% SE underestimation (implausible for all studies)
- **Evidence**: Posterior median(lambda) > 2
- **Mitigation**: Probably outliers or mixture, not SE problem
- **Decision rule**: If lambda > 2, switch to Student-t or mixture model

**6. Lambda < 1 (over-reported SEs)**:
- **Issue**: Surprising finding, suggests data over-inflated
- **Evidence**: Posterior P(lambda < 1) > 0.7
- **Mitigation**: Check for data errors, simulation, or influential studies
- **Decision rule**: If lambda < 0.8, investigate data quality issues

**7. Computational non-issues**:
- **Good news**: This model usually converges well (simpler than mixture)
- **Monitor**: ESS for lambda (aim for >400)
- **Watch**: Funnel if tau→0 (use non-centered parameterization)

---

## Cross-Model Considerations

### 4.1 Model Comparison Strategy

**Within-model validation**:
1. Prior predictive checks (do priors generate reasonable data?)
2. Posterior predictive checks (do posteriors match observed data?)
3. Convergence diagnostics (R-hat, ESS, divergences)
4. Parameter recovery simulation (if fit to synthetic data, recover truth?)

**Between-model comparison**:
1. **LOO-CV** (via ArviZ): Primary metric for predictive performance
   - Expected elpd differences (elpd_diff)
   - Standard errors of differences (SE_diff)
   - Pareto k diagnostic (k > 0.7 indicates influential points)

2. **WAIC**: Alternative information criterion (less stable than LOO)

3. **Bayes factors** (optional): If testing specific hypotheses
   - E.g., BF for "tau = 0" (fixed-effect) vs "tau > 0" (random-effect)
   - Via bridge sampling or Savage-Dickey ratio

**Decision rules**:
- If |elpd_diff| < 2*SE_diff: Models equivalent, choose simpler
- If elpd_diff > 2*SE_diff: Better model preferred (unless substantive concerns)
- If Pareto k > 0.7 for >2 studies: LOO unreliable, use K-fold CV

**Expected findings** (speculative):
- **Model 1 (Student-t)** may fit best if Study 1 is truly anomalous
- **Model 2 (Mixture)** may fit best if clustering is real
- **Model 3 (Uncertainty Inflation)** likely equivalent to standard model (lambda≈1) unless SEs systematically wrong
- Standard Normal hierarchical may be competitive if I²=0% is accurate

### 4.2 Sensitivity Analyses (Cross-Model)

**Essential sensitivities** to run for ALL models:

1. **Prior sensitivity on tau**:
   - Half-Cauchy(0, 2.5) [tight]
   - Half-Cauchy(0, 5) [standard]
   - Half-Cauchy(0, 10) [wide]
   - Check if posterior for mu, tau, or model conclusions change

2. **Prior sensitivity on mu**:
   - Normal(0, 10) [informative]
   - Normal(0, 25) [weakly informative]
   - Normal(0, 100) [vague]
   - Less critical unless borderline significance sensitive to prior

3. **Leave-one-out sensitivity**:
   - Fit each model 8 times, each time removing one study
   - Check if conclusions robust to Study 1 removal (most influential)
   - Check if conclusions robust to Study 5 removal (second influential)

4. **Fixed vs random effect**:
   - Compare models with tau fixed at 0 (fixed-effect) vs estimated (random-effect)
   - Use LOO to determine if tau > 0 supported

**Model-specific sensitivities**:

- **Model 1**: Prior on nu (Gamma(2,0.1) vs Gamma(4,0.2) vs Exponential(0.05)+1)
- **Model 2**: Prior on pi (Beta(1,1) vs Beta(2,2) vs Beta(5,5))
- **Model 3**: Prior on lambda (Log-Normal(0,0.3) vs Log-Normal(0,0.5) vs Log-Normal(0,1))

### 4.3 Expected Outcomes and Decision Tree

**Scenario 1: All models converge to similar mu**
- **Evidence**: mu posteriors overlap >80% across all 3 models
- **Interpretation**: Effect estimate robust to modeling choices
- **Action**: Report pooled estimate with caveat about borderline significance
- **Choose model**: Simplest (likely Student-t or standard hierarchical)

**Scenario 2: Student-t finds nu << 30**
- **Evidence**: Posterior median(nu) < 10, P(nu < 30) > 0.9
- **Interpretation**: Heavy tails important; Study 1 or others genuinely anomalous
- **Action**: Report Student-t results as primary; check if clustering also present
- **Choose model**: Student-t, possibly with mixture exploration

**Scenario 3: Mixture finds clear groups**
- **Evidence**: |mu_2 - mu_1| > 10, P(0.2 < pi < 0.8) > 0.8, clear group assignments
- **Interpretation**: Subgroup structure real; I²=0% hides between-group heterogeneity
- **Action**: Report mixture results; investigate what defines groups
- **Choose model**: Mixture (but acknowledge lack of covariate explanation)

**Scenario 4: Uncertainty inflation finds lambda >> 1**
- **Evidence**: Posterior median(lambda) > 1.5, P(lambda > 1.2) > 0.9
- **Interpretation**: Reported SEs systematically underestimated
- **Action**: Report inflated uncertainty; raise data quality concerns
- **Choose model**: Uncertainty-inflated (but investigate why SEs wrong)

**Scenario 5: No model fits well**
- **Evidence**: All models fail posterior predictive checks, LOO diagnostic issues
- **Interpretation**: Structural misspecification beyond these 3 model classes
- **Action**: Pivot to alternative model class (state-space, GP, non-parametric)
- **Choose model**: None of these; document failure and propose alternatives

**Scenario 6: All models have wide posteriors, conclusions unclear**
- **Evidence**: 95% CIs for mu include zero in all models, tau poorly estimated
- **Interpretation**: J=8 insufficient for confident inference
- **Action**: Report high uncertainty, recommend more studies before conclusions
- **Choose model**: Standard hierarchical (simplest) with strong uncertainty statements

---

## Implementation Timeline and Computational Budget

### 5.1 Proposed Implementation Order

**Phase 1: Standard baseline** (for comparison)
- Fit standard Normal hierarchical model
- Time: 30 min (coding + running + diagnostics)
- Purpose: Benchmark against which to compare robust models

**Phase 2: Model 1 (Student-t)**
- Implement and fit Student-t model
- Time: 1-2 hours (more complex likelihood)
- Priority: HIGH (most likely to be useful)

**Phase 3: Model 3 (Uncertainty Inflation)**
- Implement and fit uncertainty inflation model
- Time: 1 hour (similar structure to standard)
- Priority: MEDIUM (easy to implement, may not add much)

**Phase 4: Model 2 (Mixture)**
- Implement and fit mixture model
- Time: 2-3 hours (most complex, may have convergence issues)
- Priority: MEDIUM (interesting but may not be identifiable with J=8)

**Phase 5: Cross-model comparison and sensitivity**
- LOO-CV comparison
- Leave-one-out sensitivity for all models
- Prior sensitivity for best-performing model(s)
- Time: 2-3 hours

**Total estimated time**: 6-10 hours (depending on convergence issues)

### 5.2 Computational Resources

**Per model fit**:
- 4 chains × 4000 iterations (2000 warmup, 2000 sampling)
- Standard/Uncertainty models: ~2-5 minutes each
- Student-t model: ~5-10 minutes
- Mixture model: ~10-20 minutes
- Total per model: ~20-30 minutes including diagnostics

**Sensitivity analyses**:
- 3 prior variants × 3 models = 9 fits (~2-3 hours)
- 8 leave-one-out fits × 3 models = 24 fits (~4-8 hours if run serially)
- Can parallelize: ~1-2 hours wall time on 4+ core machine

**Storage**:
- Stan output per fit: ~50-100 MB
- Total for all models + sensitivities: ~2-5 GB
- Visualizations: ~50-100 MB

**Recommended hardware**:
- CPU: 4+ cores (for parallel chains)
- RAM: 8+ GB
- Disk: 10 GB free space
- Runtime: All analyses feasible in 1 day

### 5.3 Stopping Rules and Pivot Points

**Stop and pivot to simpler model if**:
1. Divergent transitions persist after reparameterization (>100 divergences after tuning)
2. R-hat > 1.05 for any parameter after extended sampling (10k iterations)
3. ESS < 50 for key parameters after extended sampling
4. Extreme parameter values (e.g., nu < 1.1, lambda > 5, tau > 50)
5. Posterior predictive checks show catastrophic failure (all observations in tails)

**Stop and pivot to more complex model if**:
1. Simpler model fails all posterior predictive checks
2. LOO Pareto k > 0.7 for >50% of observations
3. Obvious structure not captured (e.g., clear clustering but using non-mixture model)

**Declare success if**:
1. All chains converge (R-hat < 1.01, ESS > 400)
2. Posterior predictive checks reasonable (>80% observations within 95% intervals)
3. LOO Pareto k < 0.7 for >75% observations
4. Posteriors differ meaningfully from priors (learning from data)
5. Conclusions robust to reasonable sensitivity analyses

**Declare failure if**:
1. No model converges after extensive tuning
2. All models fail posterior predictive checks
3. Extreme sensitivity to priors (conclusions flip)
4. Identifiability issues across all models (parameter posteriors = priors)

---

## Key References

**Robust meta-analysis**:
- Baker R, Jackson D. "A new approach to outliers in meta-analysis." Health Care Management Science 11:121-131, 2008.
- Lee KJ, Thompson SG. "Flexible parametric models for random-effects distributions." Statistics in Medicine 27:418-434, 2008.

**Mixture models**:
- Frühwirth-Schnatter S. "Finite Mixture and Markov Switching Models." Springer, 2006.
- Beath KJ. "Infant mortality and economic growth: a flexible multivariate model." Health Economics 21:1165-1178, 2012.

**Measurement error**:
- Turner RM et al. "Predictive distributions for between-study heterogeneity and simple methods for their application in Bayesian meta-analysis." Statistics in Medicine 34:984-998, 2015.
- Riley RD et al. "Meta-analysis of individual participant data: rationale, conduct, and reporting." BMJ 340:c221, 2010.

**General Bayesian meta-analysis**:
- Gelman A. "Prior distributions for variance parameters in hierarchical models." Bayesian Analysis 1:515-534, 2006.
- Higgins JPT et al. "Quantifying heterogeneity in a meta-analysis." Statistics in Medicine 21:1539-1558, 2002.
- Röver C. "Bayesian random-effects meta-analysis using the bayesmeta R package." Journal of Statistical Software 93:1-51, 2020.

**Stan implementation**:
- Betancourt M. "Diagnosing biased inference with divergences." Stan Case Studies, 2017.
- Stan Development Team. "Stan User's Guide, v2.32." https://mc-stan.org/docs/, 2023.

---

## Summary: Three Models, Three Perspectives

| Aspect | Model 1: Student-t | Model 2: Mixture | Model 3: Inflation |
|--------|-------------------|------------------|-------------------|
| **Core idea** | Heavy-tailed likelihood | Two latent groups | Uncertain SEs |
| **What it handles** | Outliers, tail heaviness | Subgroup structure | SE underestimation |
| **Key parameter** | nu (degrees of freedom) | pi (mixing proportion) | lambda (inflation) |
| **Identifiability** | Moderate (nu vs tau) | Low (6 params, J=8) | Moderate (lambda vs tau) |
| **Complexity** | Medium | High | Low |
| **Computational** | ~5-10 min | ~10-20 min | ~2-5 min |
| **When to use** | Study 1 influential | Clear clustering | SE quality doubts |
| **When to abandon** | nu > 50 | Groups collapse | lambda ≈ 1 |
| **Fail condition** | PPC fails despite nu | Can't assign studies | No improvement |

**Recommended priority**: Model 1 > Model 3 > Model 2
**Reason**: Student-t most robust and identifiable; Mixture most complex and may not converge with J=8

---

## Appendix: Diagnostic Checklist

For each fitted model, verify:

**Convergence**:
- [ ] R-hat < 1.01 for all parameters
- [ ] ESS_bulk > 400 for mu, tau, and model-specific params
- [ ] ESS_tail > 400 for mu, tau
- [ ] No divergent transitions (or < 10 after tuning)
- [ ] No max treedepth warnings (or < 5% of iterations)
- [ ] Trace plots show good mixing (no trends, stationarity)

**Validation**:
- [ ] Prior predictive check: Priors generate reasonable data
- [ ] Posterior predictive check: P-values for test statistics ∈ [0.1, 0.9]
- [ ] Observed data within 95% posterior predictive intervals for >80% of studies
- [ ] Parameter recovery: If fit to synthetic data, recover within 95% CIs

**Model-specific**:
- [ ] **Student-t**: nu posterior differs from prior, 1 < nu < 100, no divergences
- [ ] **Mixture**: No label switching, groups separated (|mu_2-mu_1| > 5), clear assignments
- [ ] **Inflation**: lambda posterior differs from prior, 0.5 < lambda < 3

**Comparison**:
- [ ] LOO Pareto k < 0.7 for >75% of studies
- [ ] LOO elpd SE reported for model comparisons
- [ ] WAIC computed as alternative check

**Reporting**:
- [ ] Posterior summaries (mean, median, 95% CI) for mu, tau
- [ ] Posterior probability statements: P(mu > 0 | data), P(tau > 0 | data)
- [ ] Forest plot with posterior shrinkage
- [ ] Sensitivity to leave-one-out (especially Study 1)
- [ ] Sensitivity to prior choices (at least tau prior)
- [ ] Visualizations: trace plots, posterior densities, PPC plots

---

**End of Model Design Document**
**Next step**: Implement models in Stan/PyMC, run MCMC, evaluate against falsification criteria
**Key principle**: Abandon model quickly if falsification criteria met; pivot to alternatives
**Success metric**: Find model that genuinely explains data, not just completes analysis plan
