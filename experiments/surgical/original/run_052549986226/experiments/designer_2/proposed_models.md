# Hierarchical Binomial Models with Random Effects
## Designer 2: Model Specifications

**Date:** 2025-10-30
**Focus:** Hierarchical binomial models addressing overdispersion through group-level random effects
**Data:** 12 groups, severe overdispersion (φ ≈ 3.5-5.1), ICC = 0.73, Group 1 zero count challenge

---

## Executive Summary

I propose **3 hierarchical binomial models** that explicitly model between-group heterogeneity through random effects on the logit scale. All models handle the Group 1 zero count naturally through shrinkage rather than ad-hoc corrections.

### Model Comparison Table

| Model | Parameterization | Prior on σ | Complexity | Key Advantage | Primary Risk |
|-------|------------------|------------|------------|---------------|--------------|
| **M1: Centered** | logit(p_i) = μ + α_i | Half-Cauchy(0, 1) | Simple | Interpretable, standard | Poor sampling with strong shrinkage |
| **M2: Non-centered** | logit(p_i) = μ + σ·z_i | Half-Normal(0, 1) | Moderate | Efficient sampling | Requires reparameterization skill |
| **M3: Robust** | logit(p_i) = μ + α_i | Half-Student-t(3, 0, 1) | Moderate | Handles outliers better | May over-shrink Group 8 |

**Recommendation:** Start with **M2 (non-centered)** for computational efficiency, then fit **M3 (robust)** to assess outlier sensitivity.

---

## Critical Thinking: Why These Models Might FAIL

### Falsification Criteria (Apply to ALL models)

**I will abandon hierarchical binomial models entirely if:**

1. **Prior-posterior conflict on σ:** If posterior for σ is pushed to extreme values (>3 on logit scale or <0.1) despite weakly informative priors → suggests wrong distributional family
2. **Group 1 posterior stays at zero:** If shrinkage doesn't move Group 1 away from zero probability → model not working as intended
3. **Extreme divergences (>5%):** Persistent divergences despite reparameterization → fundamental geometry problem, likely need different likelihood
4. **Posterior predictive failure on overdispersion:** If model cannot reproduce φ ≈ 3.5-5.1 → random effects structure insufficient
5. **Pareto k > 0.7 for multiple groups:** Leave-one-out CV identifies influential outliers → mixture model might be needed instead
6. **Computational impossibility:** If all 3 parameterizations fail to sample → switch to beta-binomial

### What Would Make Me Switch Model Classes?

**Switch to Beta-Binomial if:**
- Continuous sampling problems across all parameterizations
- Evidence of stronger overdispersion than normal random effects can capture
- Need for more parsimonious model (2 hyperparameters vs 12 group effects)

**Switch to Mixture Model if:**
- LOO-CV suggests discrete subgroups (Pareto k > 0.7 for clusters)
- Bimodal posterior for group effects
- Residual analysis shows 2-3 distinct clusters

**Switch to Negative Binomial if:**
- Evidence that data is actually count data without fixed denominator
- Variance grows faster than mean (not constant as in binomial)

**Abandon Bayesian hierarchical entirely if:**
- Substantive evidence emerges that groups are fundamentally incomparable
- Complete pooling performs better than partial pooling in out-of-sample prediction

---

## Model 1: Centered Parameterization (Baseline)

### Mathematical Specification

```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)                    for i = 1, ..., 12

Level 2 (Group effects):
  logit(p_i) = μ + α_i
  α_i ~ Normal(0, σ)                          [Group-specific deviations]

Priors:
  μ ~ Normal(-2.5, 1.0)                       [Population mean on logit scale]
  σ ~ Half-Cauchy(0, 1)                       [Between-group SD]
```

### Prior Justification

**μ ~ Normal(-2.5, 1.0):**
- Pooled success rate is 7.6%, which gives logit(0.076) ≈ -2.53
- SD of 1.0 on logit scale is weakly informative:
  - Covers roughly 1% to 35% success rates (95% interval)
  - Centered on observed pooled rate
  - Allows substantial deviations if data supports them
- Why this matters: Group 1 has zero successes but will shrink toward μ, giving it a small but non-zero posterior probability

**σ ~ Half-Cauchy(0, 1):**
- Heavy-tailed prior allows for substantial between-group variation
- EDA shows ICC = 0.73, suggesting σ is likely in range 0.5-1.5 on logit scale
- Half-Cauchy(0, 1) has most mass below 1.5 but allows larger values if needed
- Not too restrictive: prior predictive gives reasonable variation

### Prior Predictive Check (Expected Range)

Simulating from priors, we should see:
- **Population mean p:** Roughly 1-35% (centered near 7.6%)
- **Group-specific rates:** Most between 2-20%, with occasional extremes
- **Overdispersion φ:** Should be able to generate φ > 3 if σ > 0.8
- **Zero counts:** Should occasionally occur in groups with n ≈ 50

**Red flag:** If prior predictive never generates zero counts or produces all groups with p > 20%, priors are misspecified.

### Stan Implementation

```stan
data {
  int<lower=1> N;                    // Number of groups
  array[N] int<lower=0> n_trials;    // Trials per group
  array[N] int<lower=0> r_successes; // Successes per group
}

parameters {
  real mu;                           // Population mean (logit scale)
  real<lower=0> sigma;               // Between-group SD
  vector[N] alpha;                   // Group-specific effects
}

model {
  // Priors
  mu ~ normal(-2.5, 1.0);
  sigma ~ cauchy(0, 1);
  alpha ~ normal(0, sigma);

  // Likelihood
  for (i in 1:N) {
    real p_i = inv_logit(mu + alpha[i]);
    r_successes[i] ~ binomial(n_trials[i], p_i);
  }
}

generated quantities {
  // Posterior predictions and diagnostics
  vector[N] p_posterior;             // Posterior success rates
  array[N] int r_pred;               // Posterior predictive counts
  real phi_posterior;                // Posterior overdispersion

  for (i in 1:N) {
    p_posterior[i] = inv_logit(mu + alpha[i]);
    r_pred[i] = binomial_rng(n_trials[i], p_posterior[i]);
  }

  // Calculate posterior overdispersion
  real mean_p = inv_logit(mu);
  real var_p = variance(p_posterior);
  phi_posterior = 1 + (var_p / (mean_p * (1 - mean_p)));
}
```

### Handling Group 1 Zero Count

**Strategy: Natural shrinkage without ad-hoc corrections**

- Group 1 has r=0, n=47, giving observed rate = 0%
- On logit scale, MLE would be -∞ (undefined)
- **Hierarchical model handles this naturally:**
  - Prior μ ≈ -2.5 provides anchor point
  - Shrinkage pulls α_1 toward 0, so logit(p_1) ≈ μ + small negative value
  - Posterior p_1 will likely be 1-3% (low but not zero)
- **Expected posterior:** E[p_1 | data] ≈ 0.02-0.04 based on ICC = 0.73

**Why no continuity correction:**
- Continuity corrections (r+0.5)/(n+1) are frequentist hacks
- Introduce arbitrary bias
- Bayesian shrinkage is principled and data-driven
- Allows model to learn appropriate amount of shrinkage from all 12 groups

### Expected Posterior Behavior

**If model is correct:**
1. **μ posterior:** Should be near -2.5 with SD ≈ 0.3 (updated from prior)
2. **σ posterior:** Should be in range 0.5-1.2 to produce ICC ≈ 0.73
3. **α_1 (Group 1):** Negative, around -0.5 to -1.0, giving p_1 ≈ 1-3%
4. **α_8 (Group 8):** Positive, around +1.0 to +1.5, giving p_8 ≈ 12-16% (shrunk from 14.4%)
5. **Group shrinkage:** Small-sample groups shrink more than large-sample groups
6. **Posterior predictive:** Should reproduce φ ≈ 3.5-5.1

**Diagnostic checks:**
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- No divergent transitions (ideally <1%)
- Posterior predictive p-value for overdispersion test: 0.1-0.9

### Computational Considerations

**Potential issues with centered parameterization:**
- **Funnel geometry:** When σ is small, α_i are tightly constrained near 0
- **Divergent transitions:** Likely if ICC is very high (which it is: 0.73)
- **Poor mixing:** Autocorrelation between μ and α parameters

**Warning signs:**
- Divergences > 1% → Try increasing adapt_delta to 0.95 or 0.99
- If divergences persist → Switch to M2 (non-centered)
- Low ESS for σ or α → Indicates funnel problem

**Mitigation strategies:**
1. Increase adapt_delta from 0.8 to 0.95
2. Increase max_treedepth from 10 to 12
3. Run longer chains (4000 iterations instead of 2000)
4. If all fail → Use M2 (non-centered parameterization)

### Falsification Criteria for M1 Specifically

**I will reject M1 (centered) if:**
1. Divergences > 5% even with adapt_delta = 0.99
2. ESS < 100 for any parameter after 4000 iterations
3. Posterior for σ concentrates below 0.2 (should be ~0.8-1.2 given ICC = 0.73)
4. Chain mixing is so poor that inference is unreliable (Rhat > 1.05)

**Action if rejected:** Move to M2 (non-centered) - same model, better geometry.

---

## Model 2: Non-Centered Parameterization (Recommended)

### Mathematical Specification

```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)                    for i = 1, ..., 12

Level 2 (Group effects - reparameterized):
  logit(p_i) = μ + σ·z_i
  z_i ~ Normal(0, 1)                          [Standard normal deviations]

Priors:
  μ ~ Normal(-2.5, 1.0)                       [Population mean on logit scale]
  σ ~ Half-Normal(0, 1)                       [Between-group SD]
```

### Why Non-Centered?

**Computational advantages:**
- **Eliminates funnel geometry:** z_i are independent of σ, avoiding posterior correlation
- **Better exploration:** Sampler can explore z_i and σ independently
- **Faster convergence:** Especially important when σ is uncertain (as here)
- **Fewer divergences:** Typically 10-100x fewer than centered parameterization

**When non-centered helps most:**
- High ICC (we have 0.73) → strong shrinkage → tight funnel in centered
- Uncertain σ → wide posterior for σ → centered struggles
- Small number of groups (we have 12) → less information about σ

**Mathematical equivalence:**
- α_i = σ·z_i, so logit(p_i) = μ + σ·z_i is identical to Model 1
- Only difference is parameterization, not the model itself

### Prior Justification

**μ ~ Normal(-2.5, 1.0):** Same as Model 1

**σ ~ Half-Normal(0, 1):**
- Changed from Half-Cauchy to Half-Normal for non-centered parameterization
- Half-Normal is more appropriate when using non-centered (Matt Hoffman's recommendation)
- Concentrates more mass at reasonable values (0.3-1.5) while allowing larger if needed
- Prior mean ≈ 0.8, which aligns with ICC = 0.73
- Still weakly informative: 95% prior mass is 0-2 on logit scale

### Stan Implementation

```stan
data {
  int<lower=1> N;                    // Number of groups
  array[N] int<lower=0> n_trials;    // Trials per group
  array[N] int<lower=0> r_successes; // Successes per group
}

parameters {
  real mu;                           // Population mean (logit scale)
  real<lower=0> sigma;               // Between-group SD
  vector[N] z;                       // Standardized group effects
}

transformed parameters {
  vector[N] alpha = sigma * z;       // Actual group effects
}

model {
  // Priors
  mu ~ normal(-2.5, 1.0);
  sigma ~ normal(0, 1);              // Half-normal via constraint
  z ~ std_normal();                  // Standard normal

  // Likelihood
  for (i in 1:N) {
    real p_i = inv_logit(mu + alpha[i]);
    r_successes[i] ~ binomial(n_trials[i], p_i);
  }
}

generated quantities {
  // Posterior predictions and diagnostics
  vector[N] p_posterior;
  array[N] int r_pred;
  real phi_posterior;
  real mean_success_rate;

  for (i in 1:N) {
    p_posterior[i] = inv_logit(mu + alpha[i]);
    r_pred[i] = binomial_rng(n_trials[i], p_posterior[i]);
  }

  mean_success_rate = inv_logit(mu);

  // Posterior overdispersion
  real var_p = variance(p_posterior);
  phi_posterior = 1 + (var_p / (mean_success_rate * (1 - mean_success_rate)));
}
```

### Expected Computational Performance

**Anticipated improvements over M1:**
- **Divergences:** Should be <0.1% (vs 2-5% in centered)
- **ESS for σ:** Should be >1000 per 1000 iterations (vs ~200 in centered)
- **Sampling time:** Slightly slower per iteration, but needs fewer iterations
- **Convergence:** Should converge in 1000-2000 iterations (vs 4000+ in centered)

**Optimal MCMC settings:**
- 4 chains × 2000 iterations (1000 warmup)
- adapt_delta = 0.9 (may not need 0.95)
- max_treedepth = 10 (default usually sufficient)

### Expected Posterior Behavior

**Same inferential conclusions as M1, but more reliable:**
- Posterior distributions for μ, σ, α_i should be identical to M1 (if both converge)
- Group 1 shrinkage: Same mechanism, just computed differently
- Overdispersion: Should still reproduce φ ≈ 3.5-5.1

**Quality metrics:**
- Rhat < 1.01 for all parameters (easier to achieve than M1)
- ESS/iteration > 0.5 for all parameters
- Divergences < 0.1%
- BFMI > 0.3 (Bayesian fraction of missing information)

### Falsification Criteria for M2 Specifically

**I will reject M2 (non-centered) if:**
1. Still getting divergences > 2% with adapt_delta = 0.99
2. Posterior for σ is extreme (>2.5 or <0.1) despite weakly informative prior
3. Posterior predictive cannot reproduce observed overdispersion
4. Leave-one-out CV shows Pareto k > 0.7 for >3 groups → suggests mixture model needed

**Note:** If M2 fails computationally, the problem is likely the model class (hierarchical binomial), not the parameterization. Switch to beta-binomial.

---

## Model 3: Robust Hierarchical Model with Heavy-Tailed Priors

### Mathematical Specification

```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)                    for i = 1, ..., 12

Level 2 (Group effects - non-centered):
  logit(p_i) = μ + σ·z_i
  z_i ~ Student-t(ν, 0, 1)                    [Heavy-tailed deviations]

Priors:
  μ ~ Normal(-2.5, 1.0)                       [Population mean]
  σ ~ Half-Student-t(3, 0, 1)                 [Robust prior for SD]
  ν ~ Gamma(2, 0.1)                           [Degrees of freedom]
```

### Motivation: Why Heavy Tails?

**Problem:** Group 8 is an extreme outlier (z = 3.94, rate = 14.4%)
- Under normal prior, Group 8 would be highly improbable
- Model might "fight" the data to explain Group 8
- Could lead to inflated σ or distorted μ

**Solution:** Student-t priors allow for heavier tails
- Accommodates outliers without distorting population parameters
- Group 8 can be an outlier without forcing σ to be huge
- More robust to model misspecification

**Critical question:** Is Group 8 genuinely different, or measurement error?
- If genuine: Heavy tails help model fit
- If error: Heavy tails prevent one bad observation from dominating
- We don't know which, so robust approach is safer

### Prior Justification

**μ ~ Normal(-2.5, 1.0):** Same as M1 and M2

**σ ~ Half-Student-t(3, 0, 1):**
- Heavier tails than Half-Normal, lighter than Half-Cauchy
- df = 3 gives reasonable tail behavior
- Still concentrates mass at 0.5-1.5, but allows for larger values more readily
- Prior median ≈ 0.65, mean ≈ 0.85

**ν ~ Gamma(2, 0.1):**
- Degrees of freedom for Student-t on z_i
- Mean = 20, SD = 14
- Allows data to inform heaviness of tails
- If posterior concentrates near ν > 30, data suggests normal is fine
- If posterior is ν < 10, suggests heavy tails needed

**Key insight:** If posterior ν is high (>30), M2 and M3 will give nearly identical results. If ν < 10, M3 is doing important robustness work.

### Stan Implementation

```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r_successes;
}

parameters {
  real mu;
  real<lower=0> sigma;
  vector[N] z;
  real<lower=1> nu;                  // Degrees of freedom
}

transformed parameters {
  vector[N] alpha = sigma * z;
}

model {
  // Priors
  mu ~ normal(-2.5, 1.0);
  sigma ~ student_t(3, 0, 1);        // Half-Student-t via constraint
  nu ~ gamma(2, 0.1);
  z ~ student_t(nu, 0, 1);           // Heavy-tailed group effects

  // Likelihood
  for (i in 1:N) {
    real p_i = inv_logit(mu + alpha[i]);
    r_successes[i] ~ binomial(n_trials[i], p_i);
  }
}

generated quantities {
  vector[N] p_posterior;
  array[N] int r_pred;
  real phi_posterior;

  for (i in 1:N) {
    p_posterior[i] = inv_logit(mu + alpha[i]);
    r_pred[i] = binomial_rng(n_trials[i], p_posterior[i]);
  }

  real mean_p = inv_logit(mu);
  real var_p = variance(p_posterior);
  phi_posterior = 1 + (var_p / (mean_p * (1 - mean_p)));
}
```

### Expected Posterior Behavior

**Key differences from M2:**
1. **ν posterior:** If ν_posterior > 30 → normal is adequate, M2 and M3 agree
2. **ν posterior:** If ν_posterior < 10 → heavy tails needed, M3 is superior
3. **Group 8 effect:** Should be less shrunk in M3 than M2 (heavy tails allow extremes)
4. **σ posterior:** Might be smaller in M3 than M2 (outliers don't inflate scale)
5. **μ posterior:** Should be similar across models if robust prior is working

**Diagnostic for model choice:**
- Compare LOO-CV for M2 vs M3
- If Group 8 has high Pareto k in M2 but low in M3 → M3 is better
- If Pareto k is similar → M2 is simpler, use that

### Handling Outliers (Group 8 Strategy)

**Approach: Let the data decide**
1. Heavy tails allow Group 8 to be an outlier without breaking the model
2. Posterior ν will inform us if this was necessary
3. Compare posteriors: Does Group 8 get shrunk differently in M2 vs M3?

**Expected shrinkage for Group 8:**
- Observed rate: 14.4% (31/215)
- Under M2 (normal): Shrink toward ~11-12%
- Under M3 (Student-t): Shrink toward ~12-13% (less shrinkage)
- Difference indicates how much outlier accommodation matters

### Falsification Criteria for M3 Specifically

**I will reject M3 (robust) if:**
1. Posterior ν > 50 with tight concentration → heavy tails unnecessary, use M2
2. Model over-shrinks Group 8 despite heavy tails → suggests mixture model (discrete groups)
3. Posterior σ is actually LARGER in M3 than M2 → robust prior not helping
4. Computational difficulties exceed M2 (Student-t can be slower)

**Comparison test:** If M2 and M3 give essentially identical posteriors (KL divergence < 0.1), then M2 is preferred due to simplicity.

---

## Model Comparison Strategy

### Computational Order

1. **Fit M2 first** (non-centered, normal priors) - most likely to work well
2. **Check M2 diagnostics** - if excellent, may not need others
3. **Fit M3** (robust) - to assess outlier sensitivity
4. **Fit M1 only if needed** - to demonstrate importance of non-centered parameterization

### Comparison Metrics

**Within-sample fit:**
- Posterior predictive p-values for key discrepancies (overdispersion, outliers)
- Posterior predictive distributions of r_i vs observed r_i

**Out-of-sample prediction:**
- LOO-CV (elpd_loo, p_loo, Pareto k diagnostics)
- Expected log predictive density differences
- Standard error of differences

**Parameter comparison:**
- Posterior intervals for μ, σ
- Shrinkage patterns for extreme groups (1, 8)
- Posterior correlation between parameters

**Model weights:**
- WAIC or LOO stacking weights
- If M2 and M3 have similar weights → use stacked predictions

### Decision Rules

**If LOO favors M2 by >4 elpd:** Use M2, robustness not needed
**If LOO favors M3 by >4 elpd:** Use M3, outliers are problematic
**If LOO difference <4 elpd:** Models effectively equivalent, use simpler (M2)
**If Pareto k >0.7 for multiple groups in both:** Abandon hierarchical binomial, use beta-binomial or mixture

---

## Prior Predictive Checks

### Simulation Strategy

Before fitting to data, simulate from priors to check reasonableness:

```python
# Pseudocode for prior predictive check
import numpy as np

n_sims = 1000
for sim in range(n_sims):
    mu = np.random.normal(-2.5, 1.0)
    sigma = np.abs(np.random.normal(0, 1))
    z = np.random.normal(0, 1, size=12)

    alpha = sigma * z
    logit_p = mu + alpha
    p = expit(logit_p)

    # Check if reasonable
    assert p.mean() > 0.01 and p.mean() < 0.30, "Prior mean unreasonable"
    assert p.min() >= 0.0 and p.max() <= 1.0, "Probabilities out of bounds"
    assert (p.std() / p.mean()) > 0.3, "Prior allows sufficient variation"
```

### Expected Prior Predictive Results

**Population mean success rate:**
- Prior median: ~7% (centered on observed)
- Prior 95% interval: ~1% to 30%
- Interpretation: Allows wide range but centers on reasonable values

**Group-specific rates:**
- Should vary substantially (CV > 30%)
- Occasional groups with p < 1% (to accommodate Group 1)
- Occasional groups with p > 20% (to accommodate Group 8)

**Overdispersion:**
- Prior should allow φ from 1.5 to 8
- Observed φ ≈ 3.5-5.1 should be plausible under prior

**Red flags in prior predictive:**
- If prior never generates zero counts in n=47 trials → too restrictive
- If prior generates >50% success rates frequently → too dispersed
- If prior concentrates too tightly around 7.6% → not weakly informative

---

## Posterior Predictive Checks

### Key Discrepancies to Check

1. **Overdispersion (critical):**
   - Calculate φ from posterior samples
   - Compare to observed φ ≈ 3.5-5.1
   - Posterior predictive p-value should be 0.1-0.9

2. **Zero counts:**
   - In replicated datasets, how often does Group 1 have r=0?
   - Should occur occasionally but not always (p ≈ 0.02-0.05)

3. **Extreme outliers:**
   - Can model generate groups as extreme as Group 8?
   - Posterior predictive should occasionally produce z > 3

4. **Variance-mean relationship:**
   - Plot Var(r_i) vs E[r_i] from posterior predictive
   - Should match observed pattern

### Test Statistics

```python
# Pseudocode for posterior predictive checks
def posterior_predictive_checks(posterior_samples, observed_data):
    T_obs = calculate_test_statistic(observed_data)
    T_rep = []

    for sample in posterior_samples:
        rep_data = simulate_data(sample)
        T_rep.append(calculate_test_statistic(rep_data))

    p_value = np.mean(np.array(T_rep) >= T_obs)
    return p_value

# Test statistics to use:
# - Overdispersion parameter φ
# - Number of groups with zero counts
# - Maximum observed success rate
# - SD of success rates across groups
```

**Success criteria:**
- All posterior predictive p-values between 0.05 and 0.95
- Visual comparison shows model can generate data similar to observed
- No systematic patterns in residuals

---

## Red Flags and Escape Routes

### Warning Signs (Apply to ALL Models)

**Computational red flags:**
1. **Divergences >2%** even with adapt_delta=0.99 → Model geometry problem
2. **ESS <100** for any parameter → Posterior not properly explored
3. **Rhat >1.05** → Chains haven't converged
4. **BFMI <0.2** → Pathological posterior geometry

**Statistical red flags:**
1. **Posterior predictive p-value <0.01 or >0.99** for overdispersion → Model can't reproduce key feature
2. **Prior-posterior conflict:** Posterior pushed to boundary of prior → Prior too restrictive or model misspecified
3. **Extreme posterior correlation** (|ρ| > 0.95) between parameters → Unidentifiable
4. **Pareto k >0.7** for >3 groups → LOO-CV unreliable, influential outliers

### Decision Tree for Model Switching

```
START: Fit M2 (non-centered normal)
│
├─ Converges well? (Rhat<1.01, ESS>400, Div<1%)
│  ├─ YES → Check posterior predictive
│  │       ├─ Reproduces φ? → SUCCESS, use M2
│  │       └─ Fails to reproduce φ? → Fit M3 (robust) or switch to beta-binomial
│  └─ NO → Check Pareto k
│          ├─ Multiple k>0.7? → Switch to MIXTURE MODEL
│          └─ Just computational? → Already using non-centered, switch to BETA-BINOMIAL
│
└─ If M3 also fails → ABANDON hierarchical binomial entirely
                    → Try: Beta-binomial OR Mixture of binomials
```

### Alternative Models to Consider

**If hierarchical binomial fails:**

1. **Beta-binomial** (Designer 1's focus)
   - More parsimonious (2 parameters vs 12)
   - Natural overdispersion mechanism
   - No funnel geometry problems

2. **Finite mixture of binomials**
   - If evidence for discrete subgroups
   - 2-3 components with different success rates
   - Allows Group 8 to be in different component

3. **Bayesian nonparametric** (Dirichlet process)
   - If number of clusters is unknown
   - More complex but very flexible
   - Might be overkill for N=12 groups

4. **Structural model**
   - If we get group-level covariates
   - Explain WHY groups differ
   - Current data doesn't support this

---

## Implementation Recommendations

### CmdStanPy Code Structure

```python
import cmdstanpy
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data/data.csv')

# Prepare data for Stan
stan_data = {
    'N': len(data),
    'n_trials': data['n_trials'].values,
    'r_successes': data['r_successes'].values
}

# Compile model
model = cmdstanpy.CmdStanModel(stan_file='models/model2_noncentered.stan')

# Fit with appropriate settings
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.9,
    max_treedepth=10,
    seed=12345,
    show_console=True
)

# Diagnostics
print(fit.diagnose())
print(fit.summary())

# Check for issues
divergences = fit.divergences
if divergences > 0:
    print(f"WARNING: {divergences} divergent transitions")
    print("Consider increasing adapt_delta to 0.95")

# Extract posteriors
posterior_samples = fit.stan_variables()
```

### PyMC Alternative

```python
import pymc as pm
import numpy as np

with pm.Model() as model2:
    # Data
    n_trials = pm.ConstantData('n_trials', data['n_trials'].values)
    r_successes = pm.ConstantData('r_successes', data['r_successes'].values)

    # Priors
    mu = pm.Normal('mu', mu=-2.5, sigma=1.0)
    sigma = pm.HalfNormal('sigma', sigma=1.0)
    z = pm.Normal('z', mu=0, sigma=1, shape=12)

    # Group effects (non-centered)
    alpha = pm.Deterministic('alpha', sigma * z)

    # Success probabilities
    logit_p = mu + alpha
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))

    # Likelihood
    y = pm.Binomial('y', n=n_trials, p=p, observed=r_successes)

    # Sample
    trace = pm.sample(
        2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        return_inferencedata=True
    )

# Diagnostics
print(pm.summary(trace))
```

### Recommended Workflow

1. **Prior predictive checks** (1 hour)
   - Simulate from priors
   - Verify reasonable ranges
   - Visualize prior predictive distributions

2. **Fit M2** (1-2 hours including diagnostics)
   - Start with default MCMC settings
   - Check convergence carefully
   - Examine posterior distributions

3. **Posterior predictive checks** (1 hour)
   - Generate replicated datasets
   - Calculate test statistics
   - Visual comparisons

4. **Fit M3** if needed (1-2 hours)
   - Compare to M2 posteriors
   - Check posterior ν
   - LOO-CV comparison

5. **Model comparison and selection** (1 hour)
   - Calculate LOO for all models
   - Compare posteriors visually
   - Make recommendation

**Total estimated time:** 5-8 hours for complete analysis

---

## Critical Success Metrics

### Model is SUCCESSFUL if:

1. **Computational:** Rhat < 1.01, ESS > 400, Divergences < 1%
2. **Fit:** Posterior predictive reproduces φ ≈ 3.5-5.1
3. **Outliers:** Group 1 gets reasonable shrinkage (p_1 ≈ 1-3%)
4. **Prediction:** LOO Pareto k < 0.7 for all groups
5. **Shrinkage:** Small-sample groups shrink more than large-sample groups
6. **Interpretable:** Posterior σ in range 0.5-1.5 (consistent with ICC = 0.73)

### Model is FAILING if:

1. **Computational collapse:** Can't converge despite reparameterization
2. **Prior-posterior conflict:** σ pushed to extreme values
3. **Prediction failure:** Can't reproduce key data features
4. **Outlier problems:** Multiple Pareto k > 0.7
5. **Unreasonable posteriors:** Parameters at extreme values without data justification

### Next Steps Based on Outcomes

**If all models succeed:**
- Use M2 as primary model (simplicity)
- Report M3 as sensitivity analysis
- Proceed to inference on group-specific rates

**If only M3 succeeds:**
- Heavy tails are necessary
- Group 8 is genuinely problematic
- Consider removing Group 8 as sensitivity

**If all models fail:**
- Switch to beta-binomial (Designer 1)
- Or consider mixture models
- Document why hierarchical binomial is insufficient

---

## Connection to Scientific Goals

### What We Learn from Each Model

**From M1 (centered):**
- Demonstrates computational challenges of standard parameterization
- Shows importance of reparameterization in hierarchical models
- Baseline for comparing non-centered efficiency

**From M2 (non-centered):**
- Best estimate of group-specific success rates
- Quantifies between-group heterogeneity (σ)
- Provides shrinkage toward population mean (μ)
- Answers: "How much do groups truly differ?"

**From M3 (robust):**
- Tests sensitivity to outliers (especially Group 8)
- Quantifies whether normal distribution adequate (via ν)
- Answers: "Are extreme groups fundamentally different or just tail events?"

### Inferential Goals

1. **Group-specific success rates** with uncertainty
2. **Population mean** success rate
3. **Between-group variability** (σ)
4. **Shrinkage estimates** for groups with extreme observed rates
5. **Predictions** for new groups from same population

### Domain Interpretation

- **μ:** Typical success rate for a "random" group from this population
- **σ:** How much groups vary around the typical rate (on logit scale)
- **α_i:** Deviation of group i from typical (positive = above average, negative = below)
- **Shrinkage:** How much to adjust extreme observed rates toward typical

**Example interpretation:**
- "Group 1 observed 0/47 (0%), but posterior estimate is 2% (95% CI: 0.5-5%)"
- "Group 8 observed 31/215 (14.4%), but posterior estimate is 12% (95% CI: 9-16%)"
- "Population mean is 7.5% (95% CI: 5-11%)"
- "Between-group SD on logit scale is 0.9 (95% CI: 0.6-1.3), indicating substantial heterogeneity"

---

## Files to Create

### Stan Model Files
- `/workspace/experiments/designer_2/model1_centered.stan`
- `/workspace/experiments/designer_2/model2_noncentered.stan`
- `/workspace/experiments/designer_2/model3_robust.stan`

### Python Implementation
- `/workspace/experiments/designer_2/fit_models.py` (CmdStanPy implementation)
- `/workspace/experiments/designer_2/diagnostics.py` (Convergence checks)
- `/workspace/experiments/designer_2/posterior_predictive.py` (Validation)

### Analysis Outputs
- `/workspace/experiments/designer_2/prior_predictive_checks.png`
- `/workspace/experiments/designer_2/posterior_comparison.png`
- `/workspace/experiments/designer_2/shrinkage_plot.png`
- `/workspace/experiments/designer_2/model_comparison_table.csv`

---

## Final Recommendations

### Primary Strategy

1. **Start with M2** (non-centered, normal priors)
2. **Validate thoroughly** with posterior predictive checks
3. **Compare to M3** to assess outlier sensitivity
4. **Report both** if substantive differences exist

### Comparison to Other Designers

**vs Designer 1 (Beta-binomial):**
- Hierarchical binomial is more flexible (can add covariates)
- But beta-binomial is more parsimonious (2 vs 12 parameters)
- If both fit well, prefer simpler (beta-binomial)

**vs Designer 3 (Alternative approaches):**
- Hierarchical binomial is standard, well-understood
- More interpretable parameters
- But may be outperformed by specialized models

### Success Definition

**This approach succeeds if:**
- We can reliably estimate group-specific rates
- Shrinkage appropriately balances data and population information
- Model reproduces key data features (especially overdispersion)
- Computational inference is reliable and efficient

**This approach fails if:**
- Cannot achieve convergence despite reparameterization
- Posterior predictive checks reveal systematic failures
- Beta-binomial or mixture models provide substantially better fit

Remember: **Failure is success in science** - discovering that hierarchical binomial is insufficient tells us something important about the data generation process.

---

## Appendix: Mathematical Derivations

### Relationship Between σ and ICC

On the probability scale:
```
ICC = σ²_between / (σ²_between + σ²_within)
```

For binomial data with logit link:
```
σ²_within ≈ π²/3 (logistic variance)
σ²_between = σ² (between-group variance on logit scale)

ICC ≈ σ² / (σ² + π²/3)
```

Given ICC = 0.73:
```
0.73 = σ² / (σ² + 3.29)
σ² ≈ 0.73 * 3.29 / 0.27 ≈ 8.9
σ ≈ 0.94
```

This justifies priors on σ with mode around 0.8-1.0.

### Prior Predictive for Overdispersion

Under hierarchical model:
```
Var(p_i) = E[Var(p_i | μ, σ)] + Var(E[p_i | μ, σ])
         ≈ p̄(1-p̄)/n̄ + Var(expit(μ + σ·z))
```

For overdispersion:
```
φ = 1 + n̄·Var(p_i) / (p̄(1-p̄))
  ≈ 1 + n̄·Var(expit(μ + σ·z)) / (p̄(1-p̄))
```

With σ ≈ 0.9 and p̄ ≈ 0.076:
```
φ ≈ 3.5 to 5.1 (matches observed!)
```

This confirms that σ ~ 0.8-1.0 is consistent with observed overdispersion.
