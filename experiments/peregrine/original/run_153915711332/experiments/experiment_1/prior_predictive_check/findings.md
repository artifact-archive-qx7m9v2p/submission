# Prior Predictive Check: Negative Binomial State-Space Model

**Experiment:** Experiment 1
**Model:** Negative Binomial State-Space with Random Walk Drift
**Date:** 2025-10-29
**Status:** FAIL - Priors Too Diffuse

---

## Executive Summary

The prior predictive check reveals that the current prior specification generates data that is **too variable and often implausibly extreme**. While the priors successfully cover the observed data range, they also assign substantial probability mass to scientifically unrealistic scenarios (counts exceeding 10,000, growth factors beyond 100x). The priors need tightening to be properly "weakly informative" rather than effectively uninformative.

**Decision: FAIL** - Priors require adjustment before proceeding to simulation validation.

---

## Visual Diagnostics Summary

All plots are located in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`

1. **parameter_prior_marginals.png** - Marginal distributions of delta, sigma_eta, and phi priors
2. **prior_predictive_trajectories.png** - 50 random prior predictive count trajectories vs observed data
3. **prior_predictive_coverage.png** - Prior predictive coverage diagnostics (means, maxima, growth factors, time-specific coverage)
4. **computational_red_flags.png** - Distribution of extreme values and parameter regions that generate them
5. **latent_state_prior.png** - Prior predictive latent state (eta) trajectories and initial state distribution
6. **joint_prior_diagnostics.png** - Joint parameter behavior and predictive space relationships

---

## Key Findings

### 1. Parameter Prior Behavior

#### Delta (Drift Parameter)
The drift prior `delta ~ Normal(0.05, 0.02)` performs **well**:
- **Prior mean:** 0.049 (target: 0.05)
- **95% CI:** [0.010, 0.085]
- **Assessment:** Appropriately centered and covers plausible growth rates
- **Evidence:** `parameter_prior_marginals.png` shows good concentration around target

#### Sigma_eta (Innovation SD)
The innovation prior `sigma_eta ~ Exponential(10)` is **too diffuse**:
- **Prior median:** 0.071 (expected: 0.05-0.10)
- **95% CI:** [0.003, 0.373]
- **Problem:** Upper tail extends to 0.83, allowing extremely volatile trajectories
- **Impact:** Generates unrealistic random walks with wild fluctuations
- **Evidence:** `parameter_prior_marginals.png` shows heavy right tail; `computational_red_flags.png` panel D shows extreme counts cluster at high sigma_eta values

#### Phi (Dispersion Parameter)
The dispersion prior `phi ~ Exponential(0.1)` is **too diffuse**:
- **Prior median:** 7.0 (expected: 10-20)
- **95% CI:** [0.31, 39.8]
- **Problem:** Wide range from near-Poisson (high phi) to extreme overdispersion (low phi)
- **Impact:** Allows both implausibly tight and implausibly dispersed count distributions
- **Evidence:** `parameter_prior_marginals.png` shows long right tail to 60+

### 2. Prior Predictive Coverage

#### Central Tendency
The prior predictive **fails to appropriately regularize** around observed magnitudes:
- **Prior mean of means:** 418.8 (observed: 109.5)
- **95% CI of means:** [15.9, 2196.0]
- **Problem:** Median prior predictive is 4x higher than observed mean
- **Evidence:** `prior_predictive_coverage.png` panel A shows observed data at extreme left tail

#### Maximum Values
The prior generates **implausibly extreme maxima**:
- **Prior median max:** 550.5 (observed: 272)
- **95% CI:** [44, 11,610]
- **Extreme max:** 175,837 (647x observed!)
- **Problem:** 0.4% of all prior predictive counts exceed 10,000
- **Evidence:** `prior_predictive_coverage.png` panel B; `computational_red_flags.png` panels A-B show long right tails

#### Growth Dynamics
The prior on **growth factors is reasonable but still too dispersed**:
- **Prior median growth:** 6.6x (observed: 8.45x)
- **95% CI:** [0.79x, 59.2x]
- **Problem:** Allows implausible declines and explosive growth (1.6% exceed 100x)
- **Evidence:** `prior_predictive_coverage.png` panel C shows observed within range but extreme tail is problematic

### 3. Latent State Behavior

The latent state trajectories reveal the **compounding effect of diffuse priors**:
- **Initial state:** Well-specified - prior mean log(50)=3.91 is close to observed log(29)=3.37
- **Trajectory evolution:** 95% CI widens dramatically over time, reaching [2.3, 8.5] by t=40
- **Problem:** Wide sigma_eta prior allows cumulative divergence → exp(8.5) ≈ 5000 by end of series
- **Evidence:** `latent_state_prior.png` panel A shows observed trajectory near bottom of prior envelope

### 4. Joint Prior Diagnostics

The **joint parameter relationships** reveal problematic interactions:
- **Delta vs sigma_eta (Panel A):** Independent as expected (no correlation)
- **Sigma_eta vs extreme counts:** High sigma_eta values (>0.2) strongly associated with extreme predictions
- **Prior predictive space (Panel F):** Observed data (mean=109, max=272) falls in lower-left corner; prior mass spreads far beyond to (mean=5000, max=50,000+)
- **Evidence:** `joint_prior_diagnostics.png` shows observed data as outlier in prior predictive space

### 5. Computational Red Flags

#### Extreme Value Frequency
- **Counts > 10,000:** 159 / 40,000 (0.40%) - While low, this is still concerning
- **Growth > 100x:** 16 / 1,000 (1.6%) - Clearly implausible
- **Near-zero phi:** 0% - Good, no numerical issues
- **Evidence:** `computational_red_flags.png` panel D shows extreme counts occur throughout parameter space but concentrate at high sigma_eta

#### Parameter Space Pathologies
The extreme values are primarily driven by:
1. **High sigma_eta** (>0.2): Allows random walk to drift far from mean
2. **Low phi** (<2): Creates extreme overdispersion in observation model
3. **Interaction effect:** Large sigma_eta + low phi = explosive counts

---

## Specific Issues Requiring Adjustment

### Issue 1: Sigma_eta Prior Too Diffuse
**Problem:** Exponential(10) has mean=0.1 and allows values up to 0.83
**Impact:** Latent state can compound to implausible magnitudes over 40 time steps
**Mechanism:** Each step adds Normal(delta, sigma_eta) noise; large sigma_eta → random walk explodes

**Evidence:**
- `computational_red_flags.png` panel D: Extreme counts cluster at sigma_eta > 0.2
- `joint_prior_diagnostics.png` panel E: High sigma_eta → high growth factors
- `latent_state_prior.png` panel A: Prior 95% CI spans 6 log-units by end

### Issue 2: Phi Prior Too Diffuse
**Problem:** Exponential(0.1) has mean=10 but allows values from 0.02 to 60
**Impact:** Low phi values create extreme overdispersion, amplifying latent state variability
**Mechanism:** NegBin variance = μ + μ²/φ; low φ → variance explodes for large μ

**Evidence:**
- `computational_red_flags.png` panel D: Some extreme counts at low phi (though mainly driven by sigma_eta)
- `parameter_prior_marginals.png`: Phi distribution has mode around 4 but long tail to 60

### Issue 3: Lack of Regularization on Predictive Scale
**Problem:** Priors don't effectively constrain the **joint** predictive distribution
**Impact:** While marginal priors seem reasonable, their interaction generates extremes
**Mechanism:** Compounding over time: exp(Σ innovations) can explode even with modest innovations

**Evidence:**
- `prior_predictive_trajectories.png`: 95% CI balloons from ~50 at t=1 to ~7000 at t=40
- `prior_predictive_coverage.png` panel D: Violin plots show extreme right skew by t=39

---

## Recommended Prior Adjustments

### Recommended Prior Specification v2

```
# Original (FAILED)
δ ~ Normal(0.05, 0.02)      # OK - Keep
σ_η ~ Exponential(10)       # TOO DIFFUSE
φ ~ Exponential(0.1)        # TOO DIFFUSE
η_1 ~ Normal(log(50), 1)    # OK - Keep

# Revised (RECOMMENDED)
δ ~ Normal(0.05, 0.02)      # KEEP - Appropriate
σ_η ~ Exponential(20)       # TIGHTEN - Mean 0.05, 95% to ~0.15
φ ~ Exponential(0.05)       # TIGHTEN - Mean 20, concentrate 5-50
η_1 ~ Normal(log(50), 1)    # KEEP - Appropriate
```

### Justification for Changes

#### Sigma_eta: Exponential(10) → Exponential(20)
- **Current:** Mean=0.1, median=0.07, 95th percentile=0.30
- **Revised:** Mean=0.05, median=0.035, 95th percentile=0.15
- **Rationale:**
  - EDA suggests innovations should be small (ACF=0.989 implies smooth evolution)
  - Over 40 steps, even σ_η=0.1 compounds to large cumulative changes
  - Tighter prior keeps cumulative log-scale change within [-1, 5] range
- **Impact:** Will dramatically reduce extreme trajectories while still covering observed growth

#### Phi: Exponential(0.1) → Exponential(0.05)
- **Current:** Mean=10, median=7, 95th percentile=30
- **Revised:** Mean=20, median=14, 95th percentile=60
- **Rationale:**
  - Expected phi ≈ 10-20 based on EDA (after accounting for temporal structure)
  - Higher phi = less overdispersion = more regularization
  - Current prior has mode at 4, which is too overdispersed
- **Impact:** Will reduce extreme count variance while maintaining flexibility for overdispersion

### Alternative: Hierarchical Prior on Cumulative Change

If the adjusted Exponential priors still allow extremes, consider explicitly regularizing the **cumulative change**:

```stan
// Constrain total log-scale change
real<lower=-1, upper=5> total_change;
delta = total_change / (N - 1);  // Force average drift to respect total bound

// Or: Informative prior on endpoint
eta[N] ~ normal(log(200), 0.5);  // Endpoint must be plausible
```

This directly addresses the compounding issue by constraining the **result** rather than just the increments.

---

## Re-run Checklist

After implementing revised priors, verify:

1. **No extreme counts:** <0.1% of prior predictive counts exceed 1000
2. **Plausible growth:** <0.5% of growth factors exceed 50x
3. **Appropriate coverage:** Observed mean falls within prior predictive IQR
4. **Concentrated trajectories:** Prior 95% CI at t=40 should be roughly [50, 2000]
5. **Parameter distributions:**
   - sigma_eta: 95% CI approximately [0.005, 0.15]
   - phi: 95% CI approximately [2, 60]

---

## Technical Implementation

### Stan Model Code

The following Stan code was used for this prior predictive check (saved in `/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_model.stan`):

```stan
// Prior Predictive Model for Negative Binomial State-Space
// This version samples ONLY from priors (no data likelihood)

data {
  int<lower=1> N;  // Number of time points
}

generated quantities {
  // Sample from priors
  real delta = normal_rng(0.05, 0.02);
  real<lower=0> sigma_eta = exponential_rng(10);
  real<lower=0> phi = exponential_rng(0.1);

  // Sample initial state from prior
  real eta_1 = normal_rng(log(50), 1);

  // Generate latent state trajectory
  vector[N] eta;
  eta[1] = eta_1;

  for (t in 2:N) {
    eta[t] = normal_rng(eta[t-1] + delta, sigma_eta);
  }

  // Generate prior predictive counts
  array[N] int<lower=0> C_prior;
  for (t in 1:N) {
    C_prior[t] = neg_binomial_2_log_rng(eta[t], phi);
  }

  // Summary statistics for diagnostics
  real prior_mean_count = mean(to_vector(C_prior));
  real prior_max_count = max(C_prior);
  real prior_min_count = min(C_prior);
  real prior_growth_factor = exp(eta[N] - eta[1]);
  real total_log_change = eta[N] - eta[1];
}
```

**Note:** This model was implemented and run in pure NumPy (due to CmdStan installation issues) but the logic is identical.

### Revised Model for Re-check

```stan
generated quantities {
  // REVISED priors (changes marked with //)
  real delta = normal_rng(0.05, 0.02);
  real<lower=0> sigma_eta = exponential_rng(20);  // Changed: 10 → 20
  real<lower=0> phi = exponential_rng(0.05);      // Changed: 0.1 → 0.05

  // Rest of model identical...
}
```

---

## Computational Details

- **Samples:** 1000 prior predictive draws
- **Time points:** N = 40
- **Random seed:** 42 (for reproducibility)
- **Implementation:** NumPy (scipy.stats for negative binomial)
- **Runtime:** ~10 seconds

### Files Generated

All files are in `/workspace/experiments/experiment_1/prior_predictive_check/`:

**Code:**
- `code/run_prior_predictive_numpy.py` - Main sampling script
- `code/visualize_prior_predictive.py` - Plotting script
- `code/prior_predictive_model.stan` - Stan model (reference)
- `code/prior_samples.npz` - Saved samples (numpy binary)
- `code/prior_predictive_summary.json` - Summary statistics

**Plots:**
- `plots/parameter_prior_marginals.png`
- `plots/prior_predictive_trajectories.png`
- `plots/prior_predictive_coverage.png`
- `plots/computational_red_flags.png`
- `plots/latent_state_prior.png`
- `plots/joint_prior_diagnostics.png`

---

## Next Steps

1. **Update priors** in metadata.md to revised specification
2. **Re-run this prior predictive check** with adjusted priors
3. **Verify PASS criteria:**
   - Extreme counts <0.1%
   - Growth factors <50x for 99.5% of draws
   - Observed data within prior predictive IQR
4. **Only after PASS:** Proceed to simulation-based calibration

---

## Conclusion

The current prior specification is **too permissive** and fails to encode our domain knowledge that counts should remain within [0, ~1000] range and growth should be substantial but not explosive. The priors are technically "proper" and cover the observed data, but they are **not weakly informative** in the sense of regularizing toward plausible values.

The key insight is that **priors that seem reasonable marginally can compound to extreme joint distributions** in dynamic models. A random walk with σ_η ~ Exp(10) might seem fine for one step, but over 40 steps it accumulates to wild trajectories.

The recommended tighter priors on sigma_eta and phi will:
1. Maintain coverage of plausible parameter space
2. Dramatically reduce extreme tail behavior
3. Better encode domain knowledge about smooth growth
4. Improve computational efficiency by avoiding extreme regions

**Decision: FAIL** - Do not proceed to simulation validation until priors are revised and re-checked.
