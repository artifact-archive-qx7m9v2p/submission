# Alternative Bayesian Models: Robustness & Novel Perspectives

**Designer**: Alternative Perspectives Specialist (Designer 3)
**Date**: 2025-10-29
**Philosophy**: Challenge standard approaches, explore overlooked model classes, prioritize robustness

---

## Executive Summary

The EDA consensus recommends Negative Binomial GLM for overdispersed count data (Var/Mean ≈ 70). While sound, this approach makes specific distributional assumptions that may be unnecessarily restrictive. I propose three **alternative Bayesian model classes** that:

1. **Model 1**: **Gamma-Poisson Mixture with Hierarchical Structure** - Decomposes overdispersion into random effects
2. **Model 2**: **Student-t Regression on Log-Transformed Counts** - Robust to extreme values and asymmetry
3. **Model 3**: **Conway-Maxwell-Poisson (COM-Poisson) Regression** - Flexible dispersion beyond NegBin assumptions

Each model offers a fundamentally different perspective on the data generation process and may reveal insights the standard approach misses.

---

## Critical Context from EDA

**What we know**:
- n = 40 observations, counts range [21, 269]
- Severe overdispersion: Var/Mean ≈ 70
- Strong exponential growth: log(μ) ≈ 4.36 + 0.85×year
- Time-varying dispersion (heteroscedasticity)
- No zero-inflation, no outliers, no autocorrelation

**What the standard approach assumes**:
- Negative Binomial is the "correct" distribution
- Log-link is the "natural" choice
- Dispersion is homogeneous (or follows a simple pattern)

**Why we should question this**:
- NegBin assumes a specific variance-mean relationship (V = μ + μ²/φ)
- Observed heteroscedasticity suggests more complex variance structure
- With only 40 observations, we have limited power to distinguish distributions
- Alternative parameterizations might be more scientifically interpretable

---

## Model 1: Hierarchical Gamma-Poisson (Random Effects Decomposition)

### Core Idea

Instead of treating overdispersion as a fixed distributional property, model it as **unobserved heterogeneity** using a hierarchical structure. Each observation has a latent "intensity parameter" that follows a Gamma distribution.

### Mathematical Specification

```
# Level 1: Observation-level Poisson
C[i] ~ Poisson(λ[i])

# Level 2: Random intensity with Gamma prior
λ[i] ~ Gamma(α[i], β)
  where α[i] = μ[i] × β  (shape parameter scaled by mean)
        β ~ Gamma(2, 0.1)  (common rate parameter)

# Level 3: Mean structure
log(μ[i]) = β₀ + β₁ × year[i]

# Priors
β₀ ~ Normal(4.3, 1.0)
β₁ ~ Normal(0.85, 0.3)
```

**Marginal distribution**: This is mathematically equivalent to Negative Binomial, BUT:
- The hierarchical formulation makes latent heterogeneity explicit
- We can examine individual λ[i] values for patterns
- Can extend to time-varying β or group-specific effects

### Why This Might Be Better

1. **Interpretability**: Separates "systematic trend" (μ) from "random variation" (λ)
2. **Extensibility**: Easy to add covariates affecting dispersion
3. **Diagnostics**: Can check if λ[i] values correlate with time (evidence of heteroscedasticity)
4. **Prediction**: Uncertainty naturally propagates through hierarchy

### Expected Parameter Values

Based on NegBin φ ≈ 1.5:
- **β₀**: 4.3 (intercept on log-scale)
- **β₁**: 0.85 (growth rate)
- **β**: 1.5 (inverse dispersion; higher β = less overdispersion)

Relationship: NegBin φ ≈ β in Gamma-Poisson parameterization

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior λ[i] values show no interpretable pattern**: If the random effects are just noise with no structure, the hierarchy adds complexity without insight
2. **β posterior is extremely diffuse**: Suggests data doesn't constrain dispersion well
3. **Posterior predictive checks fail**: If simulated Var/Mean ratio is far from 70
4. **LOO-CV is worse than standard NegBin by >4 ELPD**: Complexity penalty isn't justified
5. **Computational issues**: If sampling is unstable (Rhat > 1.01, low ESS), indicates misspecification

### Stress Test

**Test**: Fit model and examine posterior correlation between λ[i] and year[i]. If correlation is strong (|r| > 0.3), this suggests dispersion varies with time, and we should move to Model 3 with time-varying dispersion.

### Potential Advantages Over Standard NegBin

- **Explicit heterogeneity**: λ[i] values reveal which observations are "overdispersed outliers"
- **Model checking**: Can identify specific observations with extreme random effects
- **Scientific interpretation**: "Each observation has its own intensity drawn from a population distribution"
- **Flexible extensions**: Can add covariates on β or use nested hierarchies

### Implementation Notes (Stan/CmdStanPy)

```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> phi;  // dispersion parameter
  vector<lower=0>[N] lambda;  // latent intensities
}
model {
  vector[N] mu;

  // Priors
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.3);
  phi ~ gamma(2, 0.1);

  // Mean structure
  mu = exp(beta_0 + beta_1 * year);

  // Hierarchical structure
  lambda ~ gamma(mu * phi, phi);  // shape = mu*phi, rate = phi

  // Likelihood
  C ~ poisson(lambda);
}
generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (i in 1:N) {
    log_lik[i] = poisson_lpmf(C[i] | lambda[i]);
    C_rep[i] = poisson_rng(lambda[i]);
  }
}
```

**Warning**: This formulation may have sampling difficulties because λ[i] are continuous auxiliary parameters. If NUTS struggles, switch to marginalized NegBin (which integrates out λ analytically).

---

## Model 2: Robust Student-t Regression on Log-Transformed Counts

### Core Idea

**Radical alternative**: Instead of modeling counts directly, transform to continuous domain and use robust regression. The Student-t likelihood is **less sensitive to extreme values** and **heavy tails** than Normal/Lognormal.

This challenges the assumption that we *must* use count distributions. For counts ≥ 20, log-transformation often works well.

### Mathematical Specification

```
# Transform counts to continuous
y[i] = log(C[i] + 0.5)  # Add 0.5 to avoid log(0), though no zeros present

# Student-t likelihood (robust to outliers)
y[i] ~ StudentT(ν, μ[i], σ)

# Mean structure
μ[i] = β₀ + β₁ × year[i]

# Priors
β₀ ~ Normal(4.3, 1.0)     # log(mean count) ≈ 4.3
β₁ ~ Normal(0.85, 0.3)    # growth rate
σ ~ Exponential(1)        # residual scale
ν ~ Gamma(2, 0.1)         # degrees of freedom (controls tail heaviness)
```

**Key parameter**: ν (nu)
- ν → ∞: Approaches Normal distribution
- ν = 4: Heavy tails (robust to outliers)
- ν = 1: Cauchy distribution (extremely heavy tails)

### Why This Might Be Better

1. **Robustness**: Student-t downweights extreme observations automatically
2. **Simplicity**: Linear regression on log-scale is easier to interpret than GLM
3. **Heteroscedasticity**: Residual variance is modeled directly (can extend to σ[i])
4. **No distributional assumptions**: Doesn't assume counts follow NegBin specifically
5. **Flexible tails**: Data determines tail behavior via ν parameter

### Expected Parameter Values

Based on log(C) having mean ≈ 4.3:
- **β₀**: 4.3 (log-scale intercept)
- **β₁**: 0.85 (growth rate, matches exponential interpretation)
- **σ**: ~0.5-1.0 (residual SD on log-scale; EDA shows high variability)
- **ν**: 4-10 (if data has moderate outliers, ν will be small; if Normal-like, ν > 30)

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior ν > 30 with high certainty**: Data is Normal-like, no need for Student-t robustness
2. **Residual patterns**: If residuals show strong heteroscedasticity that isn't captured by constant σ
3. **Back-transformed predictions fail**: If exp(μ[i]) doesn't match observed counts well
4. **LOO-CV significantly worse**: If count-based models (NegBin) outperform by >6 ELPD
5. **Posterior predictive Var/Mean ≠ 70**: If model can't reproduce overdispersion structure

### Stress Test

**Test 1**: Posterior predictive check on **original count scale**:
- Simulate C_rep from posterior: C_rep[i] = round(exp(y_rep[i]))
- Compare distribution of C_rep to observed C
- Check if Var(C_rep)/Mean(C_rep) ≈ 70

**Test 2**: Cross-validation on **held-out counts** (not log-counts):
- If predictive RMSE on count scale is much worse than NegBin, abandon approach

### Potential Advantages Over Standard NegBin

- **Automatic outlier handling**: No need to worry if a few counts are "too extreme"
- **Direct interpretability**: Coefficients are exactly the exponential growth rate
- **Diagnostic simplicity**: Residual plots on log-scale are familiar to everyone
- **No link function complications**: Linear model on transformed scale
- **Variance modeling**: Can easily extend to σ[i] = exp(γ₀ + γ₁×year) for heteroscedasticity

### Implementation Notes (Stan/CmdStanPy)

```stan
data {
  int<lower=0> N;
  vector[N] y;  // log(C + 0.5)
  vector[N] year;
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0> sigma;
  real<lower=1> nu;  // degrees of freedom
}
model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.3);
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);

  // Likelihood
  y ~ student_t(nu, beta_0 + beta_1 * year, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  array[N] int C_rep;

  for (i in 1:N) {
    log_lik[i] = student_t_lpdf(y[i] | nu, beta_0 + beta_1 * year[i], sigma);
    y_rep[i] = student_t_rng(nu, beta_0 + beta_1 * year[i], sigma);
    C_rep[i] = poisson_rng(exp(y_rep[i]));  // back-transform to count scale
  }
}
```

**Critical**: Must validate back-transformed predictions match count distribution!

---

## Model 3: Conway-Maxwell-Poisson (COM-Poisson) Regression

### Core Idea

The Negative Binomial assumes a **specific form** of overdispersion (variance-mean relationship V = μ + μ²/φ). But what if this relationship is wrong?

The **COM-Poisson** distribution has a dispersion parameter that:
- Can model **overdispersion** (like NegBin)
- Can model **underdispersion** (unlike NegBin)
- Can model **equidispersion** (reduces to Poisson)
- **Doesn't impose a specific variance-mean functional form**

It's a more flexible generalization of Poisson than Negative Binomial.

### Mathematical Specification

```
# COM-Poisson likelihood
C[i] ~ COMPoisson(λ[i], ν)
  where P(C = c) ∝ λ^c / (c!)^ν

  ν = 1: Poisson (equidispersion)
  ν < 1: Overdispersion
  ν > 1: Underdispersion

# Mean structure
log(λ[i]) = β₀ + β₁ × year[i]

# Priors
β₀ ~ Normal(4.3, 1.0)
β₁ ~ Normal(0.85, 0.3)
ν ~ Gamma(2, 2)  # Centered near 1, allows exploration of dispersion
```

**Key insight**: λ in COM-Poisson is NOT the mean (unlike Poisson). The mean depends on both λ and ν in a complex way. This makes interpretation harder but dispersion modeling more flexible.

### Why This Might Be Better

1. **Flexible dispersion**: Doesn't assume NegBin's quadratic variance-mean relationship
2. **Data-driven dispersion**: ν is estimated, letting data choose the dispersion type
3. **Nests Poisson**: If ν ≈ 1, we recover Poisson (model simplifies automatically)
4. **Theoretical foundation**: Derived from queueing theory and count processes with state-dependent rates
5. **Alternative perspective**: Challenges the "NegBin is always right for overdispersion" dogma

### Expected Parameter Values

Based on severe overdispersion (Var/Mean ≈ 70):
- **β₀**: ~4.3 (though interpretation differs from NegBin)
- **β₁**: ~0.85 (growth rate)
- **ν**: **< 0.5** (strong overdispersion; smaller ν = more overdispersion)

**Warning**: Parameter interpretation is non-trivial! λ is a "rate-scale" parameter, not the mean.

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior ν ≈ 1 with high certainty**: Data is Poisson, not overdispersed (contradicts EDA)
2. **Posterior ν exactly matches NegBin-implied value**: If COM-Poisson reduces to NegBin, use NegBin (simpler)
3. **Computational failure**: COM-Poisson normalizing constant is expensive to compute; if sampling fails, abandon
4. **LOO-CV no better than NegBin**: If ΔELPD ≈ 0, prefer simpler NegBin
5. **Posterior predictive Var/Mean << 70**: Model fails to capture overdispersion

### Stress Test

**Test**: Compare implied variance-mean relationship to NegBin:
1. Simulate data from fitted COM-Poisson model
2. Compute Var vs. Mean across different μ values
3. Plot against NegBin's V = μ + μ²/φ curve
4. **If curves are nearly identical**, COM-Poisson adds complexity without new insight

### Potential Advantages Over Standard NegBin

- **No parametric assumptions on variance**: NegBin imposes V = μ + μ²/φ; COM-Poisson is more flexible
- **Dispersion test**: ν gives a direct measure of deviation from Poisson
- **Theoretical richness**: Connects to count process theory beyond simple mixture models
- **Robustness check**: If COM-Poisson and NegBin give similar results, increases confidence in NegBin

### Implementation Notes (Stan/CmdStanPy)

**Major challenge**: COM-Poisson normalizing constant is computationally expensive!

```stan
functions {
  // Approximate COM-Poisson log PMF (truncated at max_count)
  real com_poisson_lpmf(int c, real lambda, real nu, int max_count) {
    real log_Z;  // log normalizing constant
    real log_prob;
    vector[max_count + 1] terms;

    // Compute normalizing constant (sum over 0 to max_count)
    for (k in 0:max_count) {
      terms[k + 1] = k * log(lambda) - nu * lgamma(k + 1);
    }
    log_Z = log_sum_exp(terms);

    // Compute log probability
    log_prob = c * log(lambda) - nu * lgamma(c + 1) - log_Z;

    return log_prob;
  }
}
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
  int<lower=1> max_count;  // truncation for normalizing constant
}
parameters {
  real beta_0;
  real beta_1;
  real<lower=0.01> nu;  // dispersion parameter
}
model {
  vector[N] lambda;

  // Priors
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.3);
  nu ~ gamma(2, 2);  // centered near 1

  // Mean structure
  lambda = exp(beta_0 + beta_1 * year);

  // Likelihood
  for (i in 1:N) {
    target += com_poisson_lpmf(C[i] | lambda[i], nu, max_count);
  }
}
generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;

  for (i in 1:N) {
    real lambda_i = exp(beta_0 + beta_1 * year[i]);
    log_lik[i] = com_poisson_lpmf(C[i] | lambda_i, nu, max_count);
    // C_rep requires custom RNG (not built into Stan)
    C_rep[i] = -1;  // placeholder
  }
}
```

**Major caveat**: Stan doesn't have built-in COM-Poisson, so we must:
1. Implement custom log PMF (computationally expensive)
2. Truncate normalizing constant (introduce approximation error)
3. May require long sampling time or custom RNG

**Fallback**: If Stan implementation is too slow, use PyMC (which might have COM-Poisson support) or abandon this model.

---

## Model Comparison Strategy

### Fitting Protocol

1. **Fit all three models** using Stan/CmdStanPy with:
   - 4 chains, 2000 iterations (1000 warmup)
   - `adapt_delta = 0.95` (for difficult posteriors)
   - `max_treedepth = 12` (allow complex geometry)

2. **Convergence diagnostics**:
   - Rhat < 1.01 for all parameters
   - ESS_bulk > 400, ESS_tail > 400
   - No divergent transitions
   - Visual trace plots for mixing

3. **If any model fails**: Document failure mode (it tells us something!) and compare remaining models

### Model Comparison Criteria

| Criterion | Method | Decision Rule |
|-----------|--------|---------------|
| **Predictive accuracy** | LOO-CV (ELPD) | Prefer model with highest ELPD |
| **Parsimony** | LOO-CV SE | If ΔELPD < 2×SE, prefer simpler model |
| **Calibration** | PPC: Var/Mean ratio | Must recover ≈ 70 |
| **Interpretability** | Domain knowledge | Can we explain results scientifically? |
| **Robustness** | Prior sensitivity | Do conclusions change with reasonable prior variants? |

### Posterior Predictive Checks (All Models)

Must verify:
1. **Variance-to-mean ratio**: Mean(C_rep)/Var(C_rep) ≈ 70
2. **Distributional match**: QQ-plot, kernel density overlay
3. **Time-varying dispersion**: Does Var[C|year] match observed pattern?
4. **Extreme value coverage**: Do 95% intervals cover extreme counts?

### Decision Tree

```
Start with all 3 models
    |
    v
Fit & check convergence
    |
    +-- Any model fails? --> Document failure mode
    |
    v
Compute LOO-CV for all working models
    |
    v
Compare ELPDs:
    |
    +-- Clear winner (ΔELPD > 4)? --> Use that model
    |
    +-- Tied (ΔELPD < 2)? --> Prefer simpler model
    |
    v
Posterior predictive checks
    |
    +-- All fail Var/Mean check? --> **RED FLAG**: Reconsider model classes entirely
    |
    +-- One passes, others fail? --> Use the passing model
    |
    +-- All pass? --> Compare interpretability & robustness
    |
    v
Final model selection + sensitivity analysis
```

---

## Falsification: When to Abandon Everything

### Red Flags That Indicate Fundamental Misspecification

**I will reconsider all model classes if**:

1. **All three models produce similar LOO-CV**: Suggests distributional choice doesn't matter much, focus on structural form (quadratic, regime shift) instead

2. **All models fail Var/Mean posterior predictive check**: None can reproduce overdispersion → Need to model dispersion as time-varying explicitly

3. **Parameter estimates wildly inconsistent**: If β₁ varies by >50% across models → Data doesn't strongly constrain growth rate, need more informative priors

4. **Posterior-prior overlap is high**: Data isn't informative → Need stronger priors or different model structure

5. **Computational failure across multiple models**: Suggests data has pathologies (multimodality, extreme values) requiring robust preprocessing

### Escape Routes

If standard approaches fail:

**Alternative 1**: **Bayesian Quantile Regression**
- Model conditional quantiles instead of means
- Robust to distributional assumptions
- Can capture heteroscedasticity naturally

**Alternative 2**: **Gaussian Process on Log-Counts**
- Non-parametric trend (no need to choose linear/quadratic)
- Automatic smoothness estimation
- Handles heteroscedasticity via lengthscale parameter

**Alternative 3**: **Hierarchical State-Space Model**
- Treats counts as noisy observations of latent state
- Explicitly models time-varying process
- Can incorporate regime shifts endogenously

**Alternative 4**: **Re-examine data quality**
- Check for measurement error
- Look for hidden covariates (e.g., seasonality in standardized "year")
- Question if growth is truly monotonic

---

## Expected Outcomes & Interpretation

### Scenario 1: Model 2 (Student-t) Wins

**Interpretation**: Overdispersion is primarily driven by **heavy tails and extreme observations**, not true count-specific dispersion. The log-scale is the natural scale for this data.

**Implication**: Growth process is multiplicative/exponential with occasional large deviations.

**Next steps**: Examine ν parameter - does it suggest outliers? Check which observations have large residuals.

### Scenario 2: Model 1 (Hierarchical Gamma-Poisson) Wins

**Interpretation**: Overdispersion arises from **unobserved heterogeneity** - each observation has a latent "rate" drawn from a population.

**Implication**: There's meaningful variation beyond the systematic trend. Examine λ[i] posteriors for patterns.

**Next steps**: Test if λ[i] correlates with time → Move to time-varying dispersion model.

### Scenario 3: Model 3 (COM-Poisson) Wins

**Interpretation**: Variance-mean relationship is **more complex** than NegBin's quadratic form. Data needs flexible dispersion.

**Implication**: NegBin's parametric assumptions are too restrictive.

**Next steps**: Plot fitted variance-mean curve vs. NegBin to see difference.

### Scenario 4: All Models Perform Similarly

**Interpretation**: With n=40, we have **limited power to distinguish** distributional families. Distributional choice matters less than structural form (linear vs. quadratic).

**Implication**: Focus on functional form (Model 1 vs. 2 from EDA) rather than likelihood family.

**Next steps**: Do sensitivity analysis - if conclusions hold across all three distributional assumptions, we have robust findings.

---

## Prior Sensitivity Analysis

For each model, test two prior variants:

### Variant A: Weakly Informative (Default)
- β₀ ~ Normal(4.3, 1.0)
- β₁ ~ Normal(0.85, 0.3)
- Dispersion ~ Weakly informative

### Variant B: Vague (Sensitivity Check)
- β₀ ~ Normal(0, 10)
- β₁ ~ Normal(0, 5)
- Dispersion ~ Vague

**Test**: Do posterior means change by >10%? If yes, data is not strongly informative → Report uncertainty honestly.

---

## Implementation Priority

Given computational constraints and model complexity:

**Priority 1**: Model 2 (Student-t)
- Simplest to implement
- Fast sampling
- Establishes baseline for log-scale modeling

**Priority 2**: Model 1 (Hierarchical Gamma-Poisson)
- Moderate complexity
- Stan has efficient Gamma/Poisson samplers
- Directly comparable to standard NegBin

**Priority 3**: Model 3 (COM-Poisson)
- Most complex (custom likelihood)
- Potentially slow sampling
- Only fit if Models 1-2 show issues OR if time permits

---

## Summary Table

| Model | Likelihood | Key Parameter | Strength | Risk |
|-------|-----------|---------------|----------|------|
| **1. Hierarchical Gamma-Poisson** | Poisson(λ[i]), λ ~ Gamma | β (dispersion) | Interpretable heterogeneity | May reduce to NegBin |
| **2. Student-t Regression** | Student-t(log(C)) | ν (tail heaviness) | Robust, simple | Ignores count structure |
| **3. COM-Poisson** | COMPoisson(λ, ν) | ν (dispersion type) | Flexible dispersion | Computational cost |

All three offer **fundamentally different perspectives** on overdispersion:
- Model 1: Random heterogeneity
- Model 2: Heavy-tailed residuals
- Model 3: Flexible variance-mean relationship

---

## Success Criteria

This experiment is successful if:

1. **At least 2 models converge** with Rhat < 1.01
2. **LOO-CV distinguishes models** (SE of ΔELPD < |ΔELPD|)
3. **Winner reproduces Var/Mean ≈ 70** in posterior predictive checks
4. **Results are interpretable** in domain context
5. **Sensitivity to priors is low** (robust conclusions)

This experiment **fails** (but provides learning) if:

1. **All models produce similar predictions**: → Distributional choice doesn't matter
2. **None reproduce overdispersion**: → Need time-varying dispersion models
3. **Computational failures**: → Need to revisit parameterization
4. **High prior sensitivity**: → Data is not informative enough

---

## Files to Generate

After fitting, create:

1. **`model_comparison.csv`**: LOO-CV results, ΔELPD, SE
2. **`posterior_summary.csv`**: Parameter estimates for all models
3. **`ppc_results.csv`**: Posterior predictive check statistics
4. **`convergence_diagnostics.txt`**: Rhat, ESS for all parameters
5. **`visualizations/`**:
   - `loo_comparison.png`: LOO-CV comparison plot
   - `ppc_var_mean.png`: Posterior predictive Var/Mean ratio
   - `parameter_comparison.png`: Forest plot of β₀, β₁ across models
   - `residual_diagnostics.png`: For Model 2 (Student-t)
   - `random_effects.png`: For Model 1 (λ[i] posteriors)

---

## Final Thoughts: Why These Models?

The EDA provides strong evidence for Negative Binomial GLM. **I'm not disputing that**.

But science progresses by challenging consensus:

- **Model 1** asks: "Is overdispersion really homogeneous, or is there structure in the heterogeneity?"
- **Model 2** asks: "Do we need count distributions at all, or is log-Normal with robust tails enough?"
- **Model 3** asks: "Is NegBin's variance-mean relationship the only way to handle overdispersion?"

If all three models lead to the same conclusions as standard NegBin, **that's a success** - we've validated the standard approach from multiple angles.

If one provides better fit or new insights, **that's a discovery**.

Either way, we learn something.

---

**Next Steps**: Implement models in priority order, starting with Model 2 (fastest/simplest). Report back with convergence diagnostics and LOO-CV comparison.
