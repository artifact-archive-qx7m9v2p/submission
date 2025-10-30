# Experiment 2 Refined: Negative Binomial AR(1) Model with Constrained Priors

**Experiment ID**: experiment_2_refined
**Date Created**: 2025-10-29
**Status**: Prior Predictive Check Pending
**Parent Experiment**: experiment_2 (failed prior predictive check)

---

## Model Specification

### Statistical Model

**Likelihood**:
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
```

**Temporal Structure**:
```
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t
ν_t ~ Normal(0, σ)
```

**Initialization**:
```
ε₀ ~ Normal(0, σ/√(1-ρ²))    [Stationary initialization]
```

### Prior Specifications (REFINED VERSION 2)

**Regression Parameters**:
```
β₀ ~ Normal(4.69, 1.0)
β₁ ~ TruncatedNormal(1.0, 0.5, lower=-0.5, upper=2.0)
```

**Dispersion (informed by Experiment 1)**:
```
φ ~ Normal(35, 15), constrained to φ > 0
```

**Temporal Correlation**:
```
ρ ~ Beta(20, 2)       [E[ρ] = 0.909]
σ ~ Exponential(5)    [E[σ] = 0.20]
```

---

## Changes from Original Experiment 2

### Summary of Refinements

| Parameter | Original | Refined | Rationale |
|-----------|----------|---------|-----------|
| β₀ | Normal(4.69, 1.0) | **No change** | Aligned with data, working well |
| β₁ | Normal(1.0, 0.5) | **TruncatedNormal(1.0, 0.5, -0.5, 2.0)** | Constrain extreme growth rates |
| φ | Gamma(2, 0.1) | **Normal(35, 15), φ > 0** | Inform from Experiment 1 posterior |
| ρ | Beta(20, 2) | **No change** | Appropriately encodes high correlation |
| σ | Exponential(2) | **Exponential(5)** | Tighter innovation scale |

### Detailed Change Justifications

#### 1. β₁: Truncated to [-0.5, 2.0]

**Problem Addressed**: Original prior allowed extreme growth rates (β₁ up to 2.5+), creating tail explosions through exponential link.

**Change**:
- Lower bound: -0.5 → exp(-0.5×1.67) ≈ 0.44 (56% decline over study period)
- Upper bound: 2.0 → exp(2.0×1.67) ≈ 25× (2400% growth over study period)
- Compare to observed: 21→269 is 12.8× growth

**Scientific Justification**:
- Growth rates beyond 25× over 40 years are implausible for this domain
- Observed data shows ~13× growth, well within truncated range
- Still allows substantial uncertainty (25× vs 13× observed)
- **Experiment 1 posterior**: β₁ = 0.87 ± 0.04 (well within new bounds)

**Expected Impact**: Eliminates extreme trajectories (>10,000 counts) from tail combinations

#### 2. φ: Informed by Experiment 1

**Problem Addressed**: Wide Gamma(2, 0.1) prior allowed very small φ values (<5), creating high variance that amplified tail explosions.

**Change**:
- From: Gamma(2, 0.1) with mean=20, SD=14.1, range≈[0.2, 70]
- To: Normal(35, 15) with mean=35, SD=15, truncated at 0
- **Source**: Experiment 1 posterior was φ = 35.6 ± 10.8

**Scientific Justification**:
- φ is a data-generating parameter unlikely to change with AR(1) addition
- Experiment 1 successfully fit the marginal count distribution
- Using Exp1 posterior as prior is valid information transfer
- Still allows ±43% uncertainty (mean ± 1.96×SD = [5.6, 64.4])

**Trade-offs**:
- More informative (less "objective")
- BUT: Prevents numerical instability from small φ
- AND: φ posterior should be similar regardless of temporal correlation model

**Expected Impact**: Stabilizes variance structure, reduces extreme outliers

#### 3. σ: Tighter Innovation Scale

**Problem Addressed**: Exponential(2) with E[σ]=0.5 allowed heavy tail (95% quantile ≈ 1.5), creating large AR process innovations.

**Change**:
- From: Exponential(2) → E[σ] = 0.50, SD = 0.50
- To: Exponential(5) → E[σ] = 0.20, SD = 0.20

**Scientific Justification**:
- With ρ ≈ 0.91, stationary SD of ε is σ/√(1-ρ²) ≈ 2.4×σ
- Original: E[SD(ε)] ≈ 1.2, allows up to ~3.5
- Refined: E[SD(ε)] ≈ 0.48, allows up to ~1.4
- AR innovations should be small deviations from trend, not large shocks
- **Stability**: Smaller σ prevents runaway AR trajectories

**Expected Impact**:
- Constrains ε_t range to approximately [-3, +3] instead of [-10, +10]
- On exp scale: Reduces log-rate η volatility
- Primary mechanism for controlling tail explosions

#### 4. β₀ and ρ: Unchanged

**β₀ ~ Normal(4.69, 1.0)**:
- Already appropriate (exp(4.69) ≈ 109, near observed mean)
- Experiment 1 posterior: β₀ = 4.35 ± 0.04 (well within prior)
- No modification needed

**ρ ~ Beta(20, 2)**:
- Strongly motivated by EDA ACF(1) = 0.971
- Prior predictive check showed appropriate sampling
- AR(1) structure validated (despite short series challenges)
- Maintain theoretical motivation

---

## Expected Prior Predictive Behavior

### Predicted Improvements

Based on root cause analysis from original failure:

1. **Extreme Outliers** (was: 3.22% > 10,000)
   - **Target**: < 1% of counts > 10,000
   - **Mechanism**: Truncated β₁ + tighter σ prevent joint extremes
   - **Expected**: ~0.5% > 10,000

2. **Maximum Counts** (was: 674 million max)
   - **Target**: Max count < 100,000
   - **Mechanism**: All three refinements constrain upper tail
   - **Expected**: Max ≈ 10,000-50,000

3. **99th Percentile** (was: 143,745)
   - **Target**: < 5,000
   - **Mechanism**: Reduced extreme parameter combinations
   - **Expected**: ~3,000-5,000

4. **AR(1) Process Range** (was: ε ∈ [-10, 10])
   - **Target**: ε mostly in [-3, +3]
   - **Mechanism**: σ ~ Exp(5) instead of Exp(2)
   - **Expected**: 95% of ε values in [-2, +2]

### What Should Still Work

1. **Median Coverage**: Should remain good (original was 112, observed range 21-269)
2. **ACF Structure**: ρ distribution unchanged, temporal correlation preserved
3. **Growth Trend**: Truncated β₁ still allows -56% to +2400% over 40 years
4. **Dispersion**: φ centered at Exp1 value maintains appropriate variance structure

### Falsification Criteria

If refined priors still fail:
- **Counts > 10,000**: Still > 1% → Problem is structural, not just priors
- **AR validation**: Correlation(ρ, ACF) still < 0.5 → N=40 insufficient for AR(1)
- **Numerical instability**: NaN values → Fundamental model issue

Then consider:
- Simplifying to independent errors (drop AR component)
- Alternative temporal structures (random walk, changepoint)
- Different likelihood family

---

## Prior Predictive Check Plan

### Validation Steps

1. **Sample 500 prior draws** (same as original)
2. **Generate AR(1) time series** with refined priors
3. **Apply same diagnostic checks**:
   - Parameter distribution validation
   - Temporal correlation structure
   - Count distribution plausibility
   - Growth pattern realism
   - AR process behavior

### Success Criteria

Must pass all of:
- [ ] < 1% of counts exceed 10,000
- [ ] < 5% of counts exceed 5,000
- [ ] Maximum count < 100,000 per series
- [ ] Mean realized ACF(1) within [0.7, 0.95]
- [ ] Correlation(ρ, ACF) > 0.3 (relaxed, given N=40)
- [ ] Growth rates: 95% within [-100%, +1000%]
- [ ] No numerical instabilities (NaN, Inf)

### Comparison to Original

Will generate comparative metrics:
- Original vs Refined count percentiles
- Original vs Refined maximum counts
- Original vs Refined AR process range
- Demonstrate constraint effectiveness

---

## Relationship to Experiment 1

### Information Transfer

**What we use from Experiment 1**:
1. **φ posterior** (35.6 ± 10.8) → Prior mean for φ
2. **β₁ range** (0.87 ± 0.04) → Justifies truncation bounds
3. **Model structure** → Confirms NB likelihood appropriate

**What we don't assume**:
1. **Exact β values** → Allow data to inform with AR(1) structure
2. **Residual correlation** → AR(1) explicitly models this
3. **No temporal structure** → That's what we're testing

### Scientific Validity

Using Experiment 1 φ posterior as prior is valid because:
- φ governs marginal count dispersion (not temporal structure)
- Adding AR(1) affects ε_t dynamics, not count variance given μ_t
- This is **hierarchical model building**, not circular reasoning
- Alternative: Could keep uninformative φ prior, but computational instability risk

---

## Implementation Notes

### PyMC Implementation

```python
import pymc as pm
import numpy as np

with pm.Model() as model:
    # Priors - REFINED VERSION
    beta_0 = pm.Normal('beta_0', mu=4.69, sigma=1.0)
    beta_1 = pm.TruncatedNormal('beta_1', mu=1.0, sigma=0.5,
                                 lower=-0.5, upper=2.0)
    phi = pm.TruncatedNormal('phi', mu=35, sigma=15, lower=0)
    rho = pm.Beta('rho', alpha=20, beta=2)
    sigma = pm.Exponential('sigma', lam=5)

    # AR(1) process
    epsilon = pm.AR('epsilon', rho=[rho], sigma=sigma,
                    shape=N, init_dist=pm.Normal.dist(0, sigma/pm.math.sqrt(1-rho**2)))

    # Linear predictor with AR(1) errors
    eta = beta_0 + beta_1 * year + epsilon
    mu = pm.math.exp(eta)

    # Likelihood
    C_obs = pm.NegativeBinomial('C_obs', mu=mu, alpha=phi, observed=C)
```

### Stan Implementation (if available)

```stan
data {
  int<lower=1> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real<lower=-0.5, upper=2.0> beta_1;  // Truncated
  real<lower=0> phi;
  real<lower=0, upper=1> rho;
  real<lower=0> sigma;
  vector[N] epsilon_raw;
}

transformed parameters {
  vector[N] epsilon;
  vector[N] eta;
  vector[N] mu;

  // AR(1) process (non-centered)
  epsilon[1] = epsilon_raw[1] * sigma / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + sigma * epsilon_raw[t];
  }

  eta = beta_0 + beta_1 * year + epsilon;
  mu = exp(eta);
}

model {
  // Priors - REFINED
  beta_0 ~ normal(4.69, 1.0);
  beta_1 ~ normal(1.0, 0.5);  // Truncation handled in parameters block
  phi ~ normal(35, 15);
  rho ~ beta(20, 2);
  sigma ~ exponential(5);
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}
```

---

## Risk Assessment

### Computational Risks

1. **MCMC Challenges**:
   - AR(1) with high ρ can cause slow mixing
   - **Mitigation**: Non-centered parameterization, high target_accept
   - **Fallback**: Centered parameterization if divergences occur

2. **Numerical Stability**:
   - Truncated normals can cause boundary issues
   - **Mitigation**: Reasonable truncation bounds
   - **Monitor**: Check for samples at boundaries

3. **Initialization**:
   - AR process needs careful initialization
   - **Mitigation**: Use stationary distribution for ε₀
   - **Fallback**: Simple ε₀ ~ Normal(0, σ) if issues

### Scientific Risks

1. **Over-informativeness**:
   - Using Exp1 φ might overconstraint
   - **Check**: Compare posterior to prior (should move if data demands)
   - **Alternative**: Revert to Gamma(2, 0.1) with tighter σ

2. **Truncation Bias**:
   - β₁ truncation might distort inference
   - **Check**: Posterior should be well away from bounds
   - **Exp1 evidence**: β₁ = 0.87, far from [-0.5, 2.0] bounds

3. **Model Inadequacy**:
   - AR(1) might not be right temporal structure
   - **Evidence**: Will come from posterior predictive checks
   - **Alternative**: Random walk, state-space model

---

## Success Metrics

### Prior Predictive Check
- [ ] All 7 diagnostic checks pass
- [ ] Extremes reduced by >90% vs original
- [ ] Median behavior unchanged

### Model Fitting (if PPC passes)
- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] < 1% divergent transitions
- [ ] ρ posterior separates from prior (data informative)

### Model Validation
- [ ] Posterior predictive checks pass
- [ ] LOO-CV better than Experiment 1
- [ ] Residual autocorrelation reduced
- [ ] No influential observations

---

## Files and Organization

**Location**: `/workspace/experiments/experiment_2_refined/`

**Structure**:
```
experiment_2_refined/
├── metadata.md                          [This file]
├── refinement_rationale.md              [Detailed justification]
├── prior_predictive_check/
│   ├── code/
│   │   └── prior_predictive_check.py   [Standalone validation]
│   ├── plots/                           [6 diagnostic plots]
│   └── findings.md                      [Results and decision]
├── posterior_inference/                 [After PPC passes]
│   ├── code/
│   ├── diagnostics/
│   └── plots/
└── posterior_predictive_check/          [After fitting]
```

---

## References to Original Failure

**Original Experiment**: `/workspace/experiments/experiment_2/`

**Failure Documentation**: `/workspace/experiments/experiment_2/prior_predictive_check/findings.md`

**Key Failure Metrics**:
- 3.22% of counts > 10,000 (threshold: < 1%)
- Maximum count: 674,970,346
- Mean maximum per series: 2,038,561
- 99th percentile: 143,745

**Root Cause**: Multiplicative explosion from wide priors (β₀, β₁, σ) through exponential link

**Resolution Strategy**: Targeted constraints on growth (β₁), dispersion (φ), and innovations (σ)

---

## Next Steps

1. **Run Prior Predictive Check**: Execute `/workspace/experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py`
2. **Evaluate Results**: Compare to original, verify all criteria met
3. **Decision**:
   - **If PASS**: Proceed to model fitting with PyMC
   - **If FAIL**: Document issues, consider further refinement or model simplification
4. **Iteration Log**: Update `/workspace/experiments/iteration_log.md` with results

---

**Created**: 2025-10-29
**Last Modified**: 2025-10-29
**Status**: Ready for Prior Predictive Check
