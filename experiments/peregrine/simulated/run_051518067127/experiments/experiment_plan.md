# Bayesian Model Experiment Plan
## Synthesis of Three Parallel Designer Proposals

**Date**: 2025-10-30
**Project**: Count Time Series Bayesian Modeling
**Data**: 40 observations, exponential growth, severe overdispersion, high autocorrelation

---

## Overview

Three independent model designers proposed a total of **9 distinct model classes**. After synthesis and removing duplicates, this plan prioritizes **5 core experiments** based on theoretical justification, computational feasibility, and ability to address the key data challenges:

1. **Exponential growth** (2.37× per year)
2. **Severe overdispersion** (Var/Mean = 70.43)
3. **Strong autocorrelation** (ACF lag-1 = 0.971)
4. **Regime shifts** (7.8× increase early to late)

---

## Experiment Prioritization Strategy

**Start simple, add complexity only when justified by diagnostics**

### Priority Tiers

**Tier 1 (MUST ATTEMPT)**: Core models addressing primary features
- Experiment 1: Negative Binomial GLM with Quadratic Trend
- Experiment 2: AR(1) Log-Normal with Regime-Switching

**Tier 2 (ATTEMPT IF TIER 1 INADEQUATE)**: Enhanced temporal/structural models
- Experiment 3: Changepoint Negative Binomial
- Experiment 4: Gaussian Process on Log-Scale

**Tier 3 (ONLY IF NEEDED)**: Complex hierarchical structures
- Experiment 5: Hierarchical NB with Time-Varying Dispersion

**Minimum Attempt Policy**: Must attempt at least Experiments 1 and 2 unless Experiment 1 fails pre-fit validation.

---

## Experiment 1: Negative Binomial GLM with Quadratic Trend

**Source**: Designer 1, Model 1
**Priority**: 1 (baseline, MUST attempt)
**Complexity**: Low (fast to fit, ~30 sec)

### Model Specification

```
Likelihood:
  C_t ~ NegativeBinomial2(mu_t, phi)

Link function:
  log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2

Parameters:
  beta_0 ~ Normal(4.5, 1.0)     # Intercept on log scale
  beta_1 ~ Normal(0.9, 0.5)     # Linear growth
  beta_2 ~ Normal(0, 0.3)       # Quadratic term
  phi ~ Gamma(2, 0.1)           # Dispersion parameter
```

### Why This First

1. **Occam's Razor**: Simplest model addressing core features (overdispersion + nonlinear growth)
2. **Computational stability**: Standard GLM, well-tested
3. **Interpretability**: Clear parameter meanings
4. **Baseline**: Establishes minimum acceptable performance

### Falsification Criteria

**I will abandon this if**:
- Residual ACF lag-1 > 0.5 (autocorrelation not captured)
- Posterior predictive checks fail (systematic bias)
- R-hat > 1.01 or divergent transitions
- LOO Pareto-k > 0.7 for >10% of observations

### Expected Outcomes

**Most likely**: Adequate fit for mean trend, but fails residual diagnostics due to autocorrelation
**If succeeds**: May be sufficient (don't overcomplicate!)
**If fails**: Pivot to Experiment 2 (adds AR structure)

### Implementation Details

- **Software**: Stan via CmdStanPy
- **Sampling**: 4 chains, 2000 iterations (1000 warmup)
- **Diagnostics**: R-hat, ESS, divergences, prior/posterior predictive checks
- **Log-likelihood**: Required for LOO-CV comparison
- **Expected runtime**: 30-60 seconds

---

## Experiment 2: AR(1) Log-Normal with Regime-Switching

**Source**: Designer 2, Model 1
**Priority**: 1 (MUST attempt)
**Complexity**: Medium

### Model Specification

```
Likelihood:
  C[t] ~ LogNormal(mu[t], sigma_regime[regime[t]])

Mean structure:
  mu[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2 + phi * epsilon[t-1]

Autoregressive error:
  epsilon[t] = log(C[t]) - (alpha + beta_1 * year[t] + beta_2 * year[t]^2)
  epsilon[1] ~ Normal(0, sigma_regime[1] / sqrt(1 - phi^2))

Regimes (known from EDA):
  Early: t=1-14, Middle: t=15-27, Late: t=28-40

Parameters:
  alpha ~ Normal(4.3, 0.5)
  beta_1 ~ Normal(0.86, 0.2)
  beta_2 ~ Normal(0, 0.3)
  phi ~ Uniform(-0.95, 0.95)          # AR coefficient
  sigma_regime[1:3] ~ HalfNormal(0, 1)  # Regime-specific variance
```

### Why This Model

1. **Addresses autocorrelation**: AR(1) structure explicitly models temporal dependence
2. **Addresses regime heterogeneity**: Different variances by period
3. **Leverages log-scale success**: EDA showed R²=0.937 on log scale
4. **Balances complexity**: 7 parameters, interpretable

### Falsification Criteria

**I will abandon this if**:
- Residual ACF lag-1 > 0.3 after fitting (AR(1) insufficient)
- All sigma_regime posteriors overlap >80% (no regime effect)
- phi posterior centered near 0 (no autocorrelation benefit)
- Back-transformed predictions systematically biased
- Worse LOO than Experiment 1

### Expected Outcomes

**Most likely**: Best overall performance (addresses all key features)
**Expected phi**: 0.6-0.8 (high positive autocorrelation)
**Expected sigma ordering**: Middle > Late > Early (based on EDA dispersion patterns)

### Implementation Details

- **Software**: Stan via CmdStanPy
- **Sampling**: 4 chains, 2000 iterations
- **Special considerations**: AR(1) requires careful initialization for stationarity
- **Back-transformation**: Use exp(mu + sigma²/2) for mean predictions
- **Expected runtime**: 2-3 minutes

---

## Experiment 3: Changepoint Negative Binomial

**Source**: Designer 3, Model 1
**Priority**: 2 (attempt if Exps 1-2 inadequate)
**Complexity**: Medium-High

### Model Specification

```
Likelihood:
  C_t ~ NegativeBinomial2(mu_t, phi_regime[r_t])

Changepoint structure:
  tau ~ DiscreteUniform(10, 30)  # Unknown changepoint time
  regime[t] = 1 if t < tau, else 2

Regime-specific trends:
  log(mu_t) = beta_0[regime[t]] + beta_1[regime[t]] * year_t

Parameters:
  beta_0[1:2] ~ Normal(3, 1)
  beta_1[1:2] ~ Normal(0.8, 0.5)
  phi[1:2] ~ Gamma(2, 0.1)
```

### Why This Model

1. **Tests discrete regime hypothesis**: 7.8× increase suggests structural break
2. **Data-driven changepoint**: Marginalizes uncertainty over tau
3. **Regime-specific everything**: Growth rates AND dispersion can differ

### Falsification Criteria

**I will abandon this if**:
- Changepoint posterior is uniform/diffuse (no information)
- Changepoint at boundary (t<12 or t>38)
- LOO worse than polynomial model by >4 ELPD
- High Pareto-k values (poor pointwise fit)

### Expected Outcomes

**Expected changepoint**: Around t=20-24 (visual inspection suggests)
**If succeeds**: Strong evidence for discrete structural change
**If fails**: Suggests smooth transition, not discrete shift

### Implementation Details

- **Software**: Stan via CmdStanPy
- **Computational challenge**: Marginalizing discrete changepoint parameter
- **Strategy**: Use Stan's mixture modeling or discrete parameter marginalization
- **Expected runtime**: 5-10 minutes

---

## Experiment 4: Gaussian Process on Log-Scale

**Source**: Designer 2, Model 2
**Priority**: 2 (if patterns more complex than parametric)
**Complexity**: High

### Model Specification

```
Likelihood:
  log(C[t]) ~ Normal(f[t], sigma)

Gaussian Process:
  f ~ GP(mean_trend, K)

Mean function:
  mean_trend[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2

Kernel:
  K = Matern32(eta, rho)

Parameters:
  alpha ~ Normal(4.3, 0.5)
  beta_1 ~ Normal(0.86, 0.2)
  beta_2 ~ Normal(0, 0.3)
  eta ~ HalfNormal(0, 1)        # Marginal SD
  rho ~ InvGamma(5, 5)          # Length scale
  sigma ~ HalfNormal(0, 0.5)    # Observation noise
```

### Why This Model

1. **Nonparametric flexibility**: Discovers arbitrary temporal patterns
2. **Handles regime shifts**: Short length-scales allow abrupt changes
3. **Minimal assumptions**: Doesn't assume AR(1), polynomial, etc.

### Falsification Criteria

**I will abandon this if**:
- Length-scale rho → ∞ (no temporal correlation) or → 0 (white noise)
- Marginal SD eta ≈ 0 (GP adds nothing)
- Poor mixing (ESS < 100) - overparameterization
- LOO worse than AR(1) model by >4 ELPD

### Expected Outcomes

**Most likely**: Discovers smooth exponential trend, doesn't improve over simpler models
**Computational cost**: O(n³) = O(64,000) operations, manageable for n=40
**If succeeds**: Evidence for complex temporal patterns beyond AR(1)/polynomial

### Implementation Details

- **Software**: Stan via CmdStanPy (PyMC alternative if Stan issues)
- **Computational**: Use Cholesky decomposition for efficiency
- **Non-centered parameterization**: Essential for convergence
- **Expected runtime**: 10-20 minutes

---

## Experiment 5: Hierarchical NB with Time-Varying Dispersion

**Source**: Designer 1, Model 2
**Priority**: 3 (only if simpler models fail)
**Complexity**: High

### Model Specification

```
Likelihood:
  C_t ~ NegativeBinomial2(mu_t, phi[period[t]])

Period structure:
  period[1:14] = 1 (early)
  period[15:27] = 2 (middle)
  period[28:40] = 3 (late)

Hierarchical dispersion:
  phi[p] ~ Gamma(alpha_phi, beta_phi) for p in 1:3
  alpha_phi ~ Gamma(2, 1)
  beta_phi ~ Gamma(2, 1)

Fixed trend:
  log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2

Parameters:
  beta_0 ~ Normal(4.5, 1.0)
  beta_1 ~ Normal(0.9, 0.5)
  beta_2 ~ Normal(0, 0.3)
```

### Why This Model

1. **Tests dispersion heterogeneity**: EDA showed var/mean ratios: 0.68, 13.11, 7.23
2. **Partial pooling**: Borrows strength across periods
3. **Explicit hypothesis**: Is overdispersion time-varying?

### Falsification Criteria

**I will abandon this if**:
- All phi posteriors overlap >80% (homogeneous dispersion)
- Hierarchical structure doesn't improve LOO vs fixed phi
- Convergence issues (R-hat > 1.05)

### Expected Outcomes

**Expected phi ordering**: Likely phi[1] > phi[2] ≈ phi[3] (early period less dispersed)
**If fails**: Suggests dispersion differences are noise, not signal

### Implementation Details

- **Software**: Stan via CmdStanPy
- **Expected runtime**: 3-5 minutes
- **Priority**: LOW - only if regime/AR models inadequate

---

## Model Comparison Framework

### Convergence Requirements (All Models)

- **R-hat < 1.01** for all parameters
- **ESS > 400** (bulk and tail)
- **No divergences** (or <1% if unavoidable)
- **Prior predictive**: Covers data range [21, 269]
- **Posterior predictive**: Mean absolute error < 30 counts

### Model Selection Criteria

**Primary**: LOO-CV (Leave-One-Out Cross-Validation via ArviZ)
- Models ranked by ELPD (Expected Log Pointwise Predictive Density)
- Use SE of difference for pairwise comparisons
- |ΔELPD| < 2×SE → models indistinguishable (prefer simpler)

**Secondary**:
- Posterior predictive checks (visual and quantitative)
- Residual autocorrelation (should be near 0)
- Pareto-k diagnostics (< 0.7 for reliable LOO)
- Scientific interpretability

**Tertiary**:
- Computational efficiency
- Parameter identifiability
- Prior sensitivity

### Stopping Rules

**Stop early if**:
1. Experiment 1 or 2 achieves:
   - R-hat < 1.01, ESS > 400
   - Residual ACF lag-1 < 0.3
   - Posterior predictive MAE < 20
   - No systematic bias in PPCs
   - Pareto-k < 0.7 for >90% observations

**Continue to next experiment if**:
1. Convergence fails (R-hat > 1.01)
2. Residual diagnostics poor (ACF > 0.5)
3. Posterior predictive checks fail
4. High Pareto-k values (> 0.7 for >10% obs)

**Major pivot if**:
1. All Tier 1 models fail convergence
2. All models show systematic bias
3. Stan numerical issues persist across models
   - Document errors in experiments/*/diagnostics/
   - Consider PyMC as fallback PPL

---

## Implementation Workflow

### For Each Experiment

**Phase A: Prior Predictive Check** (prior-predictive-checker agent)
- Generate data from priors
- Verify prior predictive covers data range
- Identify prior-data conflicts
- **FAIL → Skip model, document reason**
- **PASS → Proceed to Phase B**

**Phase B: Simulation-Based Validation** (simulation-based-validator agent)
- Generate data from known parameters
- Fit model, check parameter recovery
- Assess identifiability
- **FAIL → Skip model, document reason**
- **PASS → Proceed to Phase C**

**Phase C: Posterior Inference** (model-fitter agent)
- Fit to real data
- Diagnose convergence
- **Save log_likelihood for LOO**
- **FAIL → Try refinement OR skip**
- **PASS → Proceed to Phase D**

**Phase D: Posterior Predictive Check** (posterior-predictive-checker agent)
- Generate predictions from posterior
- Compare to observed data
- Assess residual patterns
- **Document fit quality (continue regardless)**

**Phase E: Model Critique** (model-critique agent)
- Comprehensive assessment
- Decision: ACCEPT / REVISE / REJECT
- If REVISE → model-refiner creates new experiment

### Minimum Attempts

Per workflow requirements:
- **Must attempt**: Experiments 1 and 2
- **Unless**: Experiment 1 fails pre-fit validation (Phases A or B)
- **Document**: Any deviations in log.md

---

## Falsification Summary

### Global Falsification Criteria (Any Model)

**Computational red flags**:
- R-hat > 1.05 after extended sampling
- Persistent divergent transitions (>5%)
- ESS < 100 for any parameter

**Statistical red flags**:
- Prior-posterior conflict (posterior at prior boundary)
- Posterior predictive p-value < 0.01 or > 0.99
- Residual autocorrelation > 0.6

**Practical red flags**:
- Mean absolute error > 50 counts (>20% of range)
- Predictions outside [0, 400] range
- Negative count predictions (for continuous models)

### Model-Specific Criteria

See individual experiment sections above for each model's unique falsification criteria.

---

## Expected Timeline

**Week 1**:
- Day 1-2: Experiment 1 (NB Quadratic) - all phases
- Day 3-5: Experiment 2 (AR Log-Normal) - all phases

**Week 2** (only if needed):
- Day 1-3: Experiment 3 (Changepoint) OR Experiment 4 (GP)
- Day 4-5: Model comparison, adequacy assessment

**Most likely outcome**: Experiment 2 is adequate, Week 2 not needed

---

## Pivot Strategies

### If All Count Likelihood Models Fail
1. Try Student-t on log-scale (robust to outliers)
2. Conway-Maxwell-Poisson (flexible mean-variance)
3. Beta-binomial with large N (approximate count)

### If All Log-Transform Models Fail
1. Return to count scale with better overdispersion model
2. Consider measurement error models
3. Zero-inflated models (if zeros appear)

### If All Temporal Models Fail
1. State-space models (structural time series)
2. Dynamic linear models
3. GARCH-type conditional heteroskedasticity

### If Everything Fails
1. Review data quality and collection process
2. Consider piecewise deterministic trends
3. Consult domain expert on mechanistic models
4. **Stay within Bayesian paradigm** - non-Bayesian methods not acceptable per requirements

---

## Success Criteria

**Adequate Bayesian Model** has:
1. **Convergence**: R-hat < 1.01, ESS > 400, no divergences
2. **Calibration**: 90% posterior intervals contain ~90% of held-out data
3. **Fit quality**: MAE < 25 counts, residual ACF < 0.3
4. **LOO reliability**: Pareto-k < 0.7 for >90% observations
5. **Posterior predictive**: No systematic bias in PPCs
6. **Interpretability**: Parameters have clear scientific meaning
7. **Falsification**: Model survives stress tests

**Not required**:
- Perfect predictions (this is science, not engineering)
- Single "best" model (multiple adequate models acceptable)
- R² > 0.99 (goodness-of-fit balanced with complexity)

---

## Deliverables

For each experiment attempted:
- `experiments/experiment_N/metadata.md` - Model specification
- `experiments/experiment_N/prior_predictive_check/` - Phase A outputs
- `experiments/experiment_N/simulation_based_validation/` - Phase B outputs
- `experiments/experiment_N/posterior_inference/` - Phase C outputs (with log_likelihood)
- `experiments/experiment_N/posterior_predictive_check/` - Phase D outputs
- `experiments/experiment_N/model_critique/` - Phase E decision

After all experiments:
- `experiments/model_comparison/comparison_report.md` - LOO-CV comparison (if 2+ ACCEPT)
- `experiments/adequacy_assessment.md` - Final adequacy decision
- `final_report/report.md` - Comprehensive final report

---

## Summary

This experiment plan synthesizes three independent designer perspectives into a coherent, falsification-driven workflow. The plan:

1. **Starts simple** (Experiment 1: NB Quadratic)
2. **Adds complexity only when justified** (Experiment 2: AR + regimes)
3. **Has clear stopping rules** (don't overfit!)
4. **Documents all failures** (science learns from what doesn't work)
5. **Maintains Bayesian rigor** (priors, PPL, LOO throughout)

**Core philosophy**: Finding adequate model > finding perfect model. Stop when good enough, iterate when necessary, pivot when models fail, stay Bayesian always.
