# Prior Predictive Check: Hierarchical Logit Model (Experiment 2)

**Date**: 2025-10-30
**Model**: Hierarchical binomial with logit-normal random effects (non-centered)
**Status**: ✅ **PASS**

---

## Executive Summary

The Hierarchical Logit Model with non-centered parameterization passes all prior predictive checks. The priors generate scientifically plausible data that covers the observed range without being overly restrictive. Key findings:

- **Total successes**: Observed value (208/2814) falls at 38th percentile of prior predictive (reasonable)
- **Extreme trials**: Both extreme cases (Trial 1: 0/47 and Trial 8: 31/215) are well-covered by priors
- **Parameter ranges**: μ_prob ∈ [0.01, 0.38] at 95% level (plausible for this domain)
- **Heterogeneity**: σ prior allows appropriate range of trial-to-trial variation
- **Numerical stability**: No computational issues with logistic transformation
- **Sensitivity**: Results robust to alternative prior specifications

**Decision**: Proceed to simulation-based validation and model fitting.

---

## Visual Diagnostics Summary

All visualizations are located in `/workspace/experiments/experiment_2/prior_predictive_check/plots/`

### Primary Diagnostic Plots

1. **`parameter_plausibility.png`** - Assesses whether prior parameter distributions generate reasonable values
   - μ_logit and μ_prob distributions on both scales
   - σ (scale parameter) distribution
   - Interpretation guide for heterogeneity implications

2. **`prior_predictive_coverage.png`** - Tests if priors cover observed data adequately
   - Total successes distribution vs observed
   - Trial proportion distributions vs observed
   - Calibration Q-Q plot
   - Coverage by credible intervals

3. **`extreme_values_diagnostic.png`** - Deep dive into extreme observed trials
   - Trial 1 (0/47): count and probability distributions
   - Trial 8 (31/215): count and probability distributions
   - Quantifies prior probability of extreme events

4. **`heterogeneity_diagnostic.png`** - Evaluates σ implications for overdispersion
   - σ vs probability range and SD
   - Distribution of θ by σ quartile
   - Example datasets for different σ values

5. **`trial_by_trial_comparison.png`** - Complete trial-level view
   - All 12 trials with prior predictive distributions
   - Observed value placement for each trial
   - Percentile rankings

6. **`logit_scale_behavior.png`** - Validates logistic transformation behavior
   - Distribution of log-odds
   - Logistic transformation trajectories
   - Joint prior: μ_logit vs σ
   - Distribution of θ ranges by trial

7. **`sensitivity_analysis.png`** - Tests robustness to alternative priors
   - Comparison of 5 prior specifications
   - Coverage of observed data under alternatives
   - Calibration across all priors

---

## Model Specification

### Likelihood
```
r_i ~ Binomial(n_i, θ_i)
logit(θ_i) = μ_logit + σ·η_i
η_i ~ Normal(0, 1)
```

### Priors
```
μ_logit ~ Normal(-2.53, 1)    # logit(0.074) ≈ -2.53
σ ~ HalfNormal(0, 1)          # Truncated at 0
η_i ~ Normal(0, 1)            # Standard normal, i=1,...,12
```

### Transformation
```
θ_i = logistic(μ_logit + σ·η_i) = 1/(1 + exp(-(μ_logit + σ·η_i)))
```

**Non-centered parameterization**: Separates hierarchy (σ) from trial effects (η_i) for better HMC geometry.

---

## Data Summary

- **Trials**: 12
- **Total subjects**: 2814
- **Total successes**: 208 (7.39%)
- **Sample sizes**: [47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360]
- **Proportion range**: [0.000, 0.144]
- **Extreme cases**:
  - Trial 1: 0/47 (0.0%)
  - Trial 8: 31/215 (14.4%)

---

## Prior Predictive Results

### Simulation Configuration
- **Prior draws**: 2000
- **Random seed**: 42
- **Implementation**: Pure Python with NumPy/SciPy

### Parameter Distributions

#### Population Mean (μ_logit → μ_prob)

**Logit scale**:
- Mean: -2.49
- SD: 0.99
- Range: [-5.77, 1.32]

**Probability scale** (`parameter_plausibility.png` top-right):
- Mean: 0.106
- SD: 0.095
- **95% interval: [0.012, 0.384]**
- Observed pooled proportion: 0.074 (well within range)

**Interpretation**: Prior centers appropriately on observed pooled proportion but allows substantial uncertainty. The 95% credible interval [1.2%, 38.4%] is scientifically reasonable for this domain.

#### Scale Parameter (σ)

**Distribution** (`parameter_plausibility.png` bottom-left):
- Mean: 0.806
- SD: 0.600
- **95% interval: [0.038, 2.19]**
- Range: [0.000, 3.93]

**Interpretation guide** (`parameter_plausibility.png` bottom-right):
- σ = 0.5: Minimal heterogeneity (θ range ≈ 0.04-0.12 for μ=-2.53)
- σ = 1.0: Moderate heterogeneity (θ range ≈ 0.02-0.28)
- σ = 1.5: High heterogeneity (θ range ≈ 0.01-0.42)
- σ = 2.0: Very high heterogeneity (θ range ≈ 0.00-0.55)

The prior appropriately allows for both minimal and substantial trial-to-trial variation.

#### Trial Probabilities (θ_i)

**Distribution** (`logit_scale_behavior.png` bottom-right):
- Mean: 0.129
- SD: 0.150
- Range: [0.000011, 0.996]
- **Proportion < 0.01**: 6.1% (allows rare events like Trial 1)
- **Proportion > 0.30**: 11.0% (allows high success rates if warranted)

**Key insight**: Priors generate appropriate diversity - most θ values fall in plausible range [0.05, 0.15], but extremes near 0 and 1 are possible when data demands.

### Coverage of Observed Data

#### Total Successes

**Prior predictive** (`prior_predictive_coverage.png` top-left):
- Mean: 363.5
- SD: 300.6
- Range: [9, 2238]
- **95% interval: [38, 1158]**

**Observed**: 208 successes
- **Percentile: 38.4%** ✅

**Assessment**: Observed value is below prior mean but well within credible range. Not extreme or implausible.

#### Trial-Level Coverage

**50% credible intervals** (`prior_predictive_coverage.png` bottom-right):
- Covered: 5/12 trials (42%)
- Expected: ~6/12 (50%)

**95% credible intervals**:
- Covered: 12/12 trials (100%) ✅
- Expected: ~11/12 (95%)

**Calibration** (`prior_predictive_coverage.png` bottom-left):
- Q-Q plot shows reasonable alignment with diagonal
- No systematic bias (points scatter around perfect calibration line)

**Trial-by-trial percentiles** (`trial_by_trial_comparison.png`):
- Range: 7th to 79th percentile
- Median: 49th percentile
- No trials < 5th or > 95th percentile ✅

### Extreme Values Analysis

#### Trial 1: 0/47 (extreme low)

**Prior predictive counts** (`extreme_values_diagnostic.png` top-left):
- Simulated zeros: 279/2000 (14.0%)
- Mean: 6.16
- SD: 7.25
- Range: [0, 47]
- **Observed at 7th percentile** ✅

**Prior predictive probabilities** (`extreme_values_diagnostic.png` top-right):
- θ₁ mean: 0.132
- **P(θ₁ < 0.01) = 6.1%** (allows very low probabilities)
- Range: [0.0001, 0.50]

**Assessment**: Observing 0/47 is unlikely but scientifically plausible under the prior. The model appropriately allows for rare events without forcing them.

#### Trial 8: 31/215 (extreme high)

**Prior predictive counts** (`extreme_values_diagnostic.png` bottom-left):
- Simulated ≥ 31: 576/2000 (28.8%)
- Mean: 27.80
- SD: 33.35
- Range: [0, 206]
- **Observed at 74th percentile** ✅

**Prior predictive probabilities** (`extreme_values_diagnostic.png` bottom-right):
- θ₈ mean: 0.129
- Range: [0.0001, 0.87]
- **Observed (0.144) at 55th percentile** ✅

**Assessment**: Trial 8's higher success rate is well-covered by the prior. Not an outlier.

### Heterogeneity Assessment

**σ vs probability range** (`heterogeneity_diagnostic.png` top-left):
- Positive relationship between σ and θ range (as expected)
- Observed range (0.144) is within prior predictive distribution

**σ vs probability SD** (`heterogeneity_diagnostic.png` top-right):
- Observed SD of proportions (0.038) is covered by prior
- Prior allows both minimal and substantial overdispersion

**Distribution by σ quartile** (`heterogeneity_diagnostic.png` bottom-left):
- Low σ (Q1): Narrow θ distributions centered near μ_prob
- High σ (Q4): Wide θ distributions with heavy tails
- Appropriate range of heterogeneity

**Example datasets** (`heterogeneity_diagnostic.png` bottom-right):
- Shows how different σ values create different data patterns
- Observed data pattern is consistent with σ ≈ 0.5-1.5

### Numerical Stability

**Logistic transformation** (`logit_scale_behavior.png`):
- No NaN or Inf values in θ
- Theta values at boundaries: 0 instances < 1e-10 or > 1-1e-10 ✅
- Smooth transformation from logit to probability scale
- Joint prior shows independence of μ_logit and σ (correlation: -0.003)

**Assessment**: No computational red flags. Model is numerically stable.

---

## Sensitivity Analysis

Tested 5 alternative prior specifications (`sensitivity_analysis.png`):

1. **Baseline**: N(-2.53, 1), HalfN(0, 1) - Original
2. **Wider μ**: N(-2.53, 2), HalfN(0, 1) - More uncertain about population mean
3. **More dispersed σ**: N(-2.53, 1), HalfN(0, 2) - More heterogeneity
4. **Both wider**: N(-2.53, 2), HalfN(0, 2) - Less informative overall
5. **Tighter μ**: N(-2.53, 0.5), HalfN(0, 1) - More confident about mean

### Key Findings

#### Coverage is robust
- All 5 specifications cover observed total successes in 95% intervals
- Observed percentile ranges from 30% to 43% across specifications
- All cover extreme trials (Trial 1 and Trial 8)

#### Parameter ranges

| Prior | μ_prob 95% interval | σ 95% interval | Total r mean (SD) |
|-------|---------------------|----------------|-------------------|
| Baseline | [0.011, 0.334] | [0.027, 2.270] | 346 (287) |
| Wider μ | [0.002, 0.793] | [0.029, 2.159] | 536 (605) |
| More dispersed σ | [0.012, 0.377] | [0.059, 4.427] | 464 (382) |
| Both wider | [0.002, 0.769] | [0.079, 4.344] | 579 (583) |
| Tighter μ | [0.030, 0.171] | [0.027, 2.282] | 301 (187) |

**Observations**:
- Wider μ prior increases prior predictive variance but doesn't exclude observed data
- More dispersed σ prior allows more heterogeneity (as intended)
- Tighter μ prior reduces variance but still covers observed data
- **Baseline prior strikes good balance**: informative but not restrictive

#### Calibration across priors

All prior specifications show reasonable calibration:
- Trial percentiles scatter around 50% for most trials
- No systematic over/under-coverage
- Extreme trials (1, 8) handled appropriately by all specifications

### Recommendation

**Use baseline prior**: N(-2.53, 1) for μ_logit, HalfN(0, 1) for σ

**Rationale**:
- Incorporates domain knowledge (centers on pooled proportion)
- Allows sufficient uncertainty for data to dominate
- Not overly restrictive or overly vague
- Results are robust to reasonable alternatives

---

## Scientific Plausibility Assessment

### Domain Constraints ✅

- **No negative counts**: All simulated r_i ≥ 0
- **No counts exceeding sample size**: All r_i ≤ n_i
- **Probabilities in [0,1]**: All θ_i ∈ [0, 1] (no boundary issues)

### Scale Reasonableness ✅

- **Population mean**: 95% interval [1.2%, 38.4%] is plausible for psychology/cognitive studies
- **Trial probabilities**: Most fall in [5%, 15%] range, which matches observed data pattern
- **Heterogeneity**: σ allows both minimal variation and substantial trial differences

### Structural Soundness ✅

- **Logit scale appropriate**: Natural for modeling binary outcomes with multiplicative effects
- **Non-centered parameterization**: Mathematically equivalent to centered but better for HMC
- **Hierarchy structure**: Allows partial pooling (trials inform each other)
- **No prior-likelihood conflict**: Priors and likelihood work together smoothly

### Computational Viability ✅

- **No numerical overflow**: Logistic transformation stable across all prior draws
- **No extreme parameter values**: No σ > 5 or |μ_logit| > 10
- **Appropriate prior mass**: 95% of prior mass in scientifically plausible region

---

## Comparison to Beta-Binomial (Experiment 1)

| Aspect | Beta-Binomial | Hierarchical Logit |
|--------|---------------|-------------------|
| **Mixing scale** | Probability (θ) | Log-odds (logit θ) |
| **Interpretation** | Direct probability variation | Multiplicative effects |
| **Extreme values** | Can struggle with 0/n | Naturally handles via log-odds |
| **Prior predictive mean** | ~298 successes | ~364 successes |
| **Prior predictive SD** | ~192 | ~301 |
| **Coverage of observed** | 40th %ile | 38th %ile |

**Key differences**:
- Hierarchical Logit has wider prior predictive (more uncertainty)
- Both cover observed data well
- Logit scale may be more natural for extreme trials (0/47)

**Next step**: Fit both models and compare posterior predictive performance and LOO-CV.

---

## Potential Issues Identified

### None Critical

1. **Prior predictive mean > observed**: Mean total successes (364) exceeds observed (208)
   - **Not a problem**: Prior should be uncertain; posterior will update
   - Observed is at reasonable percentile (38%)

2. **Wide prior predictive intervals**: 95% CI for total [38, 1158] is wide
   - **By design**: Priors should be regularizing but not overly restrictive
   - Data will shrink posterior substantially

3. **Trial 1 (0/47) at 7th percentile**: Slightly low but not extreme
   - **Acceptable**: Within expected range [2.5%, 97.5%]
   - Model allows rare events without forcing them

### Recommendations for Fitting

1. **Use adapt_delta = 0.95**: Hierarchical models can have complex geometry
2. **Monitor divergences**: Non-centered helps, but watch for any issues
3. **Check η_i posteriors**: Should remain approximately normal
4. **Assess shrinkage**: Compare trial-specific θ_i to pooled estimate

---

## Decision Criteria Review

### PASS Criteria (All Met) ✅

- [x] Generated data respects domain constraints (binary outcomes, counts ≤ n)
- [x] Range covers plausible values without being absurd ([1%, 40%] is reasonable)
- [x] No numerical/computational warnings (no NaN, Inf, or boundary issues)
- [x] 95% intervals cover observed data (12/12 trials covered)
- [x] Extreme values within plausible range (both Trial 1 and 8 covered)
- [x] Prior allows reasonable heterogeneity (σ ∈ [0, 4] covers range)
- [x] Transformation behaves well (logistic smooth and stable)

### FAIL Criteria (None Met) ✅

- [ ] Consistent domain violations - **NO violations**
- [ ] Numerical instabilities - **NO instabilities**
- [ ] Prior-likelihood conflict - **NO conflict**
- [ ] Observed data extreme outlier (p < 0.01 or > 0.99) - **NO, observed at 38th %ile**
- [ ] Prior too restrictive - **NO, allows appropriate range**
- [ ] Forces impossible values - **NO, all values plausible**

---

## Conclusion

**PASS**: The Hierarchical Logit Model with non-centered parameterization demonstrates excellent prior specification. The priors:

1. Generate scientifically plausible datasets
2. Cover all observed data without being overly vague
3. Allow appropriate heterogeneity (both low and high σ)
4. Handle extreme trials (0/47 and 31/215) naturally
5. Show no numerical or computational issues
6. Are robust to reasonable alternative specifications

The model is ready for:
- Simulation-based calibration (to validate inference)
- Fitting to observed data
- Posterior predictive checking
- Comparison with Beta-Binomial model (Experiment 1)

**No revisions needed.** Proceed to next validation step.

---

## Files Generated

### Code
- `/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_simulation.py`
- `/workspace/experiments/experiment_2/prior_predictive_check/code/visualizations.py`
- `/workspace/experiments/experiment_2/prior_predictive_check/code/sensitivity_analysis.py`

### Data
- `/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_samples.npz`
- `/workspace/experiments/experiment_2/prior_predictive_check/code/sensitivity_summary.csv`

### Plots
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/parameter_plausibility.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_coverage.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/extreme_values_diagnostic.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/heterogeneity_diagnostic.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/trial_by_trial_comparison.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/logit_scale_behavior.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/sensitivity_analysis.png`

### Report
- `/workspace/experiments/experiment_2/prior_predictive_check/findings.md` (this document)

---

**Validator**: Claude (Bayesian Model Validation Specialist)
**Date**: 2025-10-30
**Next Step**: Simulation-based calibration
