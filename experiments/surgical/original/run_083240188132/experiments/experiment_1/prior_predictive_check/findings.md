# Prior Predictive Check: Beta-Binomial Hierarchical Model (Experiment 1)

**Date**: 2025-10-30
**Model**: Beta-Binomial Hierarchical Model
**Status**: FAIL - Prior Revision Required

---

## Executive Summary

The prior predictive check reveals a **critical issue with the concentration parameter κ**: the current prior Gamma(2, 0.1) is **too concentrated around high values**, resulting in an overdispersion parameter φ = 1 + 1/κ that is far too small (mean = 1.10, 95% CI: [1.02, 1.49]). This fails to accommodate the strong overdispersion observed in the data (empirically φ ≈ 3.5-5.1).

**Decision**: **FAIL** - Priors must be revised before proceeding to model fitting.

---

## Visual Diagnostics Summary

All visualizations are located in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`:

1. **`hyperparameter_priors.png`** - Shows distributions of μ, κ, φ, and their relationships
2. **`group_proportion_priors.png`** - Displays prior predictive coverage of group-level proportions
3. **`prior_predictive_counts.png`** - Shows prior predictive distributions for each group's count
4. **`diagnostic_dashboard.png`** - Comprehensive overview of all diagnostics

---

## Model Specification

### Likelihood
```
r_i | p_i, n_i ~ Binomial(n_i, p_i)  for i = 1, ..., 12 groups
```

### Hierarchical Structure
```
p_i | μ, κ ~ Beta(α, β)  where α = μκ, β = (1-μ)κ
```

### Priors (Current - Under Review)
```
μ ~ Beta(2, 18)          # E[μ] = 0.10, centers on population mean
κ ~ Gamma(2, 0.1)        # E[κ] = 20, controls between-group variation
```

### Data Context
- **12 groups** with varying sample sizes (n = 47 to 810)
- **Observed counts** (r = 0 to 46)
- **Observed proportions**: 0% to 14.4%
- **Pooled rate**: 7.39%
- **Strong overdispersion**: Empirically φ ≈ 3.5-5.1, ICC = 0.66

---

## Diagnostic Results

### 1. Population Mean (μ)

**Assessment**: ✅ **PASS**

The prior Beta(2, 18) on μ performs well:
- **Prior mean**: 0.102 (theoretical E[μ] = 0.10)
- **Prior 95% CI**: [0.015, 0.251]
- **Observed pooled rate**: 0.074

**Visual Evidence** (`hyperparameter_priors.png`, top-left panel):
The observed pooled rate (green dashed line) falls well within the prior 95% credible interval. The prior is appropriately centered and provides reasonable uncertainty.

**Interpretation**: The μ prior is weakly informative, centered near the observed pooled rate but allowing for substantial variation. This is appropriate for a hierarchical model where we expect some uncertainty about the population mean.

---

### 2. Concentration Parameter (κ) and Overdispersion (φ)

**Assessment**: ❌ **FAIL - Critical Issue**

The prior Gamma(2, 0.1) on κ generates concentration values that are **too high**, resulting in overdispersion parameters that are **far too low**:

**κ Prior Performance**:
- **Prior mean**: 19.7 (theoretical E[κ] = 20)
- **Prior 95% CI**: [2.04, 53.76]

**φ Prior Performance**:
- **Prior mean**: 1.10
- **Prior 95% CI**: [1.02, 1.49]
- **Expected empirical range**: 2-10 (based on observed data)

**Visual Evidence** (`hyperparameter_priors.png`, bottom-left panel):
The overdispersion parameter distribution shows almost all mass concentrated below φ = 1.5, with the expected range of 2-10 (shown in green shading) barely covered by the prior at all.

**Interpretation**: The relationship φ = 1 + 1/κ means:
- High κ → Low φ → Little overdispersion
- Low κ → High φ → Strong overdispersion

The current prior places most mass on high κ values (mean ≈ 20), which implies φ ≈ 1.05. This represents **nearly binomial behavior** with minimal extra-binomial variation. However, the observed data shows **strong overdispersion** (ICC = 0.66), requiring φ in the range 2-10.

**Why This Matters**: When κ is too high:
- The hierarchical structure provides minimal pooling
- Group-level proportions p_i become too tightly clustered around μ
- The model cannot capture the substantial between-group variation observed in the data

---

### 3. Group-Level Proportions (p_i)

**Assessment**: ✅ **PASS (Marginal)**

**Visual Evidence** (`group_proportion_priors.png`):
- **Top-left panel**: The prior predictive distribution of all p_i samples shows reasonable spread, with observed proportions (red lines) well within the bulk of the prior mass
- **Top-right panel**: Box plots for each group show the prior predictive distributions generally encompass the observed values (red diamonds)
- **Bottom-left panel**: Coverage check shows all 12 groups have observed counts within the 95% prior predictive intervals (all bars are green)

**Findings**:
- **Prior predictive range**: [0.000, 1.000] (full support)
- **Prior predictive 95% CI**: [0.000, 0.372]
- **Observed range**: [0.000, 0.144]
- **Implausible high values** (p_i > 0.5): 0.68% of samples

**Interpretation**: Despite the concentration parameter issue, the marginal distribution of p_i is acceptable because:
1. The μ prior allows values from near-zero to 0.25
2. When κ is occasionally small (in the tail), p_i spreads appropriately
3. No systematic violations of plausibility (very few samples > 0.5)

However, this marginal adequacy masks the joint problem with κ.

---

### 4. Prior Predictive Counts (r_i)

**Assessment**: ✅ **PASS**

**Visual Evidence** (`prior_predictive_counts.png`):
Across all 12 groups, the observed counts (red dashed lines) fall within the prior predictive distributions. Key observations:

- **Group 1** (n=47, r=0): Prior median ≈ 3, 95% CI [0, 20], observed = 0 ✓
- **Group 2** (n=148, r=18): Prior median ≈ 10, 95% CI [0, 40], observed = 18 ✓
- **Group 4** (n=810, r=46): Prior median ≈ 63, 95% CI [0, 290], observed = 46 ✓
- **Group 8** (n=215, r=31): Prior median ≈ 15, 95% CI [0, 85], observed = 31 ✓

**Coverage**: 12/12 groups (100%) have observed counts within the prior predictive 95% credible intervals.

**Interpretation**: The prior predictive successfully covers all observed data. The wide intervals reflect appropriate prior uncertainty, though this is somewhat inflated by the occasional low-κ samples that create extreme variability.

---

### 5. Prior-Data Conflict

**Assessment**: ✅ **PASS**

**Visual Evidence** (`group_proportion_priors.png`, bottom panels):

- **Bottom-left panel (Coverage Check)**: Shows the percentile of each observed count within its prior predictive distribution. All 12 groups fall within the 2.5-97.5% range (green bars).
- **Bottom-right panel (QQ Plot)**: Shows reasonable calibration, with empirical quantiles tracking the theoretical uniform distribution reasonably well, all within the 95% tolerance band.

**Findings**:
- **Extreme tails**: 0/12 groups (0%) in tails < 2.5% or > 97.5%
- **Percentile distribution**: All observed counts fall within central 95% of prior predictions

**Interpretation**: There is **no evidence of prior-data conflict**. The priors generate data that is compatible with what we observe. This is reassuring and indicates the model structure is fundamentally sound.

---

### 6. Computational Diagnostics

**Assessment**: ✅ **PASS**

**Findings**:
- **Extreme α** (> 1000): 0.0% of samples
- **Extreme β** (> 1000): 0.0% of samples
- **Extreme κ** (> 1000): 0.0% of samples

**Visual Evidence** (`hyperparameter_priors.png`, bottom-right panel):
The scatter plot of α vs β shows no extreme values that would cause numerical instabilities in Beta distribution sampling.

**Interpretation**: No computational red flags. The parameterization α = μκ and β = (1-μ)κ produces values that are well-behaved for Beta distribution calculations.

---

## Key Visual Evidence

### Most Important Diagnostic Plots

1. **Overdispersion Failure** (`hyperparameter_priors.png`, bottom-left):
   - Shows φ distribution concentrated at 1.02-1.49
   - Expected range 2-10 barely covered
   - This is the smoking gun for the FAIL decision

2. **Overall Coverage** (`diagnostic_dashboard.png`, middle row):
   - Violin plots show all observed proportions fall within prior predictive distributions
   - Demonstrates model structure is sound, just needs parameter adjustment

3. **Group-Level Success** (`prior_predictive_counts.png`):
   - All 12 individual group plots show observed counts within prior predictions
   - Confirms no systematic misspecification

---

## Decision Criteria

| Criterion | Status | Result |
|-----------|--------|--------|
| **μ_coverage** | ✅ PASS | Observed pooled rate within prior 95% CI |
| **phi_range** | ❌ FAIL | Prior φ 95% CI [1.02, 1.49] does not cover expected range 2-10 |
| **p_plausible** | ✅ PASS | < 5% of samples have implausibly high p_i > 0.5 |
| **coverage_adequate** | ✅ PASS | 12/12 groups (100%) within prior predictive 95% CI |
| **no_extreme_params** | ✅ PASS | No extreme α, β, or κ values causing computational issues |
| **no_severe_conflict** | ✅ PASS | 0/12 groups in extreme tails |

**Overall**: 5/6 criteria passed, but the failed criterion (phi_range) is **critical** for model performance.

---

## Final Decision: FAIL

### Why This Matters

While the prior predictive check shows good marginal coverage and no computational problems, the **fundamental issue with κ is critical**:

1. **Statistical**: The prior cannot accommodate the observed overdispersion (φ ≈ 3.5-5.1 vs prior φ ≈ 1.1)
2. **Scientific**: The ICC = 0.66 indicates strong between-group variation that the model cannot capture
3. **Inferential**: A prior that is inconsistent with known data properties will lead to:
   - Posterior concentrated in low-probability prior regions
   - Poor convergence in MCMC sampling
   - Unreliable uncertainty quantification

### The Core Problem

The relationship φ = 1 + 1/κ means:
- To get φ = 2 (minimum for observed overdispersion), we need κ = 1
- To get φ = 5 (middle of observed range), we need κ = 0.25
- Current prior has E[κ] = 20, implying φ ≈ 1.05

**The prior is off by an order of magnitude.**

---

## Recommendations

### Immediate Action: Revise κ Prior

**Current**: κ ~ Gamma(2, 0.1) with E[κ] = 20

**Recommended**: κ ~ Gamma(1.5, 0.5) with E[κ] = 3

This revised prior:
- Centers κ around 3, giving φ ≈ 1.33
- Has mode at κ ≈ 1, giving φ = 2
- Allows substantial mass for κ < 1, enabling φ > 2
- Still puts reasonable prior mass on κ up to 10-15

**Alternative options**:
1. **More dispersed**: κ ~ Gamma(1, 0.3) with E[κ] = 3.33, but more spread
2. **Half-Normal**: κ ~ HalfNormal(σ = 2) centers around κ = 1.6
3. **Log-Normal**: log(κ) ~ Normal(-1, 1) gives median κ ≈ 0.37, E[φ] ≈ 3.7

### Keep μ Prior Unchanged

The prior μ ~ Beta(2, 18) is working well:
- Appropriate center
- Reasonable uncertainty
- Good coverage of observed pooled rate

**No changes needed for μ.**

### Next Steps

1. **Revise κ prior** to Gamma(1.5, 0.5) or similar
2. **Re-run this prior predictive check** with revised prior
3. **Verify** φ distribution now covers the 2-10 range
4. **Proceed to model fitting** only after PASS decision

---

## Technical Details

### Prior Predictive Sampling Method

- **Software**: NumPy/SciPy for direct sampling
- **Sample size**: 1,000 prior predictive draws
- **Procedure**:
  1. Sample μ ~ Beta(2, 18)
  2. Sample κ ~ Gamma(2, 0.1)
  3. Compute α = μκ, β = (1-μ)κ
  4. Sample p_i ~ Beta(α, β) for each of 12 groups
  5. Sample r_i ~ Binomial(n_i, p_i)
- **Random seed**: 42 (reproducible)

### Files Generated

**Code**:
- `/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_check.py`

**Plots**:
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/hyperparameter_priors.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/group_proportion_priors.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_predictive_counts.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/diagnostic_dashboard.png`

**Summary**:
- `/workspace/experiments/experiment_1/prior_predictive_check/summary.txt`

---

## References

**Observed Data**: `/workspace/data/data.csv`
- 12 groups, n = 47 to 810, r = 0 to 46
- Pooled rate = 7.39%
- Strong overdispersion: φ = 3.5-5.1, ICC = 0.66

**Model Specification**: Beta-Binomial hierarchical model with:
- Population mean μ
- Concentration κ controlling between-group variation
- Overdispersion parameter φ = 1 + 1/κ

---

## Conclusion

The prior predictive check successfully identified a critical misspecification in the κ prior before any computational resources were spent on model fitting. The κ prior Gamma(2, 0.1) is **too concentrated on high values**, resulting in insufficient overdispersion to match the observed data structure.

**Action Required**: Revise κ prior to allow lower values (higher φ) and re-run this prior predictive check.

**DO NOT PROCEED TO MODEL FITTING** until this issue is resolved and a PASS decision is obtained.

---

*Generated*: 2025-10-30
*Analyst*: Bayesian Model Validator (Prior Predictive Check Specialist)
*Status*: GO/NO-GO Decision = **NO-GO**
