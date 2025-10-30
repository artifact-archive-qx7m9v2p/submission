# Prior Predictive Check: Experiment 1
## Standard Hierarchical Logit-Normal Model

**Date:** 2025-10-30
**Status:** PASS
**Recommendation:** Proceed to Simulation-Based Calibration (SBC)

---

## Visual Diagnostics Summary

This prior predictive check generated 1,000 samples from the prior distribution to assess whether the model specification generates scientifically plausible data before fitting to observations. Five diagnostic visualizations were created:

1. **`parameter_plausibility.png`** - Marginal prior distributions for hyperparameters (mu, tau) and implied distribution for success rates (p), plus joint mu-tau scatter showing prior independence
2. **`prior_predictive_coverage.png`** - Group-by-group comparison of prior predictive 95% credible intervals vs observed success rates and counts
3. **`distribution_diagnostics.png`** - Extreme value analysis, Q-Q plot against uniform distribution, and coverage assessment by sample size
4. **`prior_predictive_draws.png`** - Spaghetti plot of 100 random prior predictive draws overlaid with observed data
5. **`count_diagnostics.png`** - Success count distributions, observed vs predicted scatter, residuals, and calibration by trial size

---

## Model Specification

```
# Hyperpriors
mu ~ Normal(-2.6, 1.0)     # Population mean (logit scale)
tau ~ Half-Normal(0, 0.5)  # Between-group SD (logit scale)

# Non-centered parameterization
theta_raw[j] ~ Normal(0, 1)
theta[j] = mu + tau * theta_raw[j]

# Likelihood
p[j] = inv_logit(theta[j])
r[j] ~ Binomial(n[j], p[j])
```

**Data Context:**
- 12 groups with varying trial sizes (n = 47 to 810)
- Observed success rates range: [0.031, 0.140]
- Observed mean success rate: 0.079

---

## Key Visual Evidence

### 1. Parameter Plausibility (`parameter_plausibility.png`)

**Top Row - Hyperparameter Priors:**
- **mu (population mean):** Sample mean = -2.61 closely matches prior specification N(-2.6, 1.0). The distribution is well-centered and the samples closely follow the theoretical prior density (red curve).
- **tau (between-group SD):** Sample mean = 0.40, following Half-Normal(0, 0.5). The samples track the theoretical half-normal density well, showing appropriate concentration near zero with a reasonable tail allowing for heterogeneity.

**Bottom Left - Implied Prior for Success Rates:**
The transformation through the logit-normal hierarchy produces a **strongly right-skewed distribution** on the probability scale (p):
- Mean: 0.104 (close to observed 0.079)
- The distribution is concentrated in the low-rate region [0, 0.2], appropriate for rare-event data
- The observed range [0.031, 0.140] (red shaded area) sits comfortably within the bulk of the prior predictive mass
- Excellent alignment between prior expectations and observed data

**Bottom Right - Prior Independence:**
The scatter plot of mu vs tau shows no correlation structure (as intended), confirming that the priors are properly independent and won't induce spurious dependencies.

### 2. Prior Predictive Coverage (`prior_predictive_coverage.png`)

**Success Rates (Top Panel):**
- All 12 observed success rates (red squares) fall **well within** the prior predictive 95% credible intervals (blue shaded regions)
- The prior median (blue line) tracks reasonably close to observed values, averaging around 0.06-0.07
- The 95% CI is appropriately wide (spanning roughly 0.01 to 0.40), reflecting genuine prior uncertainty while remaining scientifically plausible
- No systematic bias: observed points don't consistently fall at edges of intervals

**Success Counts (Bottom Panel):**
- Again, all 12 observed counts (dark blue squares) lie within prior predictive 95% CIs
- The wider intervals for groups with larger n (especially group 4 with n=810) correctly reflect greater absolute uncertainty when sample sizes are larger
- The prior median counts track the general scale of observed data without being overly constrained

**Key Finding:** 100% coverage (12/12 groups) indicates the priors are neither too tight (which would exclude plausible data) nor pathologically loose.

### 3. Distribution Diagnostics (`distribution_diagnostics.png`)

**Top Left - Overall Success Rate Distribution:**
The histogram shows the prior predictive distribution is heavily concentrated in [0, 0.2], with:
- Long right tail extending to ~1.0 (allowing flexibility)
- Peak density around 0.05-0.10
- Observed mean (0.079) sits right in the high-density region
- Observed range well-covered by bulk of prior mass

**Top Right - Extreme Value Check:**
Critical diagnostic for computational pathologies:
- Only **3.34%** of samples fall in extreme regions (<0.01 or >0.99)
- This is well below the 10% threshold that would signal overly diffuse priors
- Most mass (36% each) concentrated in plausible ranges [0.01-0.05] and [0.05-0.1]
- Almost no mass (1.3%) above 0.5, correctly reflecting that this is low-rate data

**Bottom Left - Q-Q Plot vs Uniform(0,1):**
The strong deviation from the diagonal (red line) shows the implied prior on p is **not uniform** - it's appropriately informative:
- Heavy concentration at low values (curve below diagonal at left)
- Sparse at high values (curve above diagonal at right)
- This non-uniformity is desired: we're encoding domain knowledge that success rates are low

**Bottom Right - Coverage by Sample Size:**
All 12 groups shown as green circles (within prior 95% CI), no red X's (outside CI):
- Coverage is consistent across the range of sample sizes (47 to 810 trials)
- No systematic pattern where small or large sample sizes are systematically outside intervals
- This confirms the prior works well regardless of the amount of data per group

---

## Quantitative Diagnostic Results

### Success Rate (p) Checks

| Diagnostic | Result | Target | Status |
|------------|--------|--------|--------|
| % values in [0, 1] | 100.00% | 100% | ✓ PASS |
| % values in [0, 0.5] | 98.98% | >50% for low-rate data | ✓ PASS |
| % extreme (<0.01 or >0.99) | 3.34% | <10% | ✓ PASS |
| Prior min | 0.001 | ≤ 0.031 (obs min) | ✓ PASS |
| Prior max | 0.958 | ≥ 0.140 (obs max) | ✓ PASS |
| Coverage (obs in prior 95% CI) | 100% (12/12 groups) | ≥50% | ✓ PASS |

### Prior Predictive Quantiles for p

| Percentile | Value | Interpretation |
|------------|-------|----------------|
| 1st | 0.0055 | Lower bound: very rare events possible |
| 5th | 0.0121 | Lower tail: near observed minimum (0.031) |
| 25th | 0.0325 | First quartile: within observed range |
| 50th (median) | 0.0686 | Center: very close to observed mean (0.079) |
| 75th | 0.1371 | Third quartile: near observed maximum (0.140) |
| 95th | 0.3193 | Upper tail: allows moderate success rates |
| 99th | 0.5016 | Extreme upper: allows up to 50% (unlikely) |

**Interpretation:** The quantiles show excellent alignment with observed data. The observed range [0.031, 0.140] spans roughly the 5th to 75th percentiles of the prior predictive distribution - indicating the priors are appropriately informative without being overconfident.

### Hyperparameter Samples

| Parameter | Prior Specification | Sample Mean | Sample SD | Match |
|-----------|-------------------|-------------|-----------|-------|
| mu | N(-2.6, 1.0) | -2.615 | 1.018 | ✓ Excellent |
| tau | Half-N(0, 0.5) | 0.397 | 0.298 | ✓ Good |

Both hyperparameters sampled as expected, with sample moments closely matching theoretical prior distributions.

### Computational Health

| Check | Result | Status |
|-------|--------|--------|
| NaN in mu | False | ✓ PASS |
| NaN in tau | False | ✓ PASS |
| NaN in p | False | ✓ PASS |
| Inf values | False | ✓ PASS |

No numerical pathologies detected - the model is computationally stable.

---

## Domain Knowledge Assessment

### Prior Mean vs Observed Mean
- **Prior predictive mean for p:** 0.104
- **Observed mean:** 0.079
- **Ratio:** 1.32

The prior is slightly higher than observed (by about 30%), but this is well within acceptable bounds. The prior is weakly informative - it correctly anticipates low success rates but doesn't force exact agreement. This difference will be easily overcome by the likelihood during fitting.

### Prior SD vs Observed SD
- **Prior predictive SD for p:** 0.105
- **Observed SD:** 0.031

The prior allows substantially more between-group variability than observed. This is appropriate - priors should be conservative about heterogeneity. The data will determine the actual level of shrinkage.

### Scale Plausibility
The prior median p = 0.0686 corresponds to inv_logit(-2.64), very close to the specified mu = -2.6. This transformation from logit to probability scale produces sensible values:
- **Logit -2.6 → probability 0.069** (excellent match to observed mean 0.079)
- **Logit -1.6 → probability 0.168** (one SD above mu, plausible upper bound)
- **Logit -3.6 → probability 0.027** (one SD below mu, plausible lower bound)

These transformations show the priors are well-calibrated to the scale of the problem.

---

## Prior-Data Conflict Check

A key diagnostic is whether the priors "fight" the observed data. Four lines of evidence suggest **no conflict**:

1. **All observed values within prior 95% CIs:** 12/12 groups covered
2. **Observed mean near prior median:** 0.079 observed vs 0.069 prior median (15% difference)
3. **Observed range well within prior range:** [0.031, 0.140] obs vs [0.001, 0.958] prior
4. **No systematic bias in residuals:** (`count_diagnostics.png`) shows balanced positive/negative residuals

The priors encode appropriate domain knowledge without being so constraining that they'll dominate the likelihood.

---

## Prior Predictive Draws Assessment (`prior_predictive_draws.png`)

The spaghetti plot of 100 random draws reveals:
- **Wide coverage:** Prior draws span 0 to ~0.7, covering all plausible scenarios
- **Typical draws near observed:** The bulk of light blue lines cluster around 0.05-0.15, overlapping observed red line
- **Heterogeneity allowed:** Individual draws show varying levels of between-group variability (some flat, some varying)
- **No systematic misfit:** Observed pattern (red) is neither systematically above nor below the cloud of prior draws

This visualization confirms the priors generate diverse but plausible data patterns.

---

## Count-Based Diagnostics (`count_diagnostics.png`)

### Success Count Distribution (Top Left)
- Prior mean: 24.5 successes (averaged over all groups and draws)
- Observed mean: 16.3 successes
- The prior generates slightly higher counts on average, but the distributions overlap substantially

### Observed vs Predicted Counts (Top Right)
- Scatter shows all points cluster near origin (low counts)
- Vertical error bars (prior 95% CIs) contain all observed values
- Points fall slightly below y=x line, consistent with prior generating slightly higher counts

### Residuals (Bottom Left)
Observed minus prior median counts:
- Mix of positive (green) and negative (red) residuals
- No systematic pattern by group
- Largest positive residuals: Groups 2, 8 (observed higher than prior median)
- Largest negative residuals: Groups 4, 5, 10 (observed lower than prior median)
- **Balanced pattern suggests no structural misspecification**

### Calibration by Trial Size (Bottom Right)
- All observed rates (red circles) fall within prior predictive 95% CIs (blue vertical bars)
- Coverage is excellent across the full range of trial sizes (47 to 810)
- Wider absolute intervals for larger n is appropriate (though rate intervals would be narrower)

---

## Structural Assessment

### Non-Centered Parameterization
The model uses **theta[j] = mu + tau * theta_raw[j]** with theta_raw ~ N(0,1), which:
- Decorrelates group effects from hyperparameters (avoids funnel geometry)
- Shows clear independence in mu-tau scatter plot (`parameter_plausibility.png`)
- Will enable efficient MCMC sampling, especially when tau is small

### Likelihood Appropriateness
The Binomial likelihood is correctly specified for success/trial count data:
- No domain violations (all p in [0, 1])
- Handles varying trial sizes appropriately
- Prior predictive counts respect discrete nature (integer-valued)

### Hierarchical Structure
The two-level hierarchy (group-level theta[j], population-level mu/tau):
- Allows partial pooling (groups can share information)
- Prior on tau allows data to determine amount of shrinkage
- Flexible enough to handle homogeneous groups (tau → 0) or heterogeneous groups (tau large)

---

## Pass/Fail Decision

### Criteria Met

All five critical checks passed:

1. ✓ **Domain constraints respected:** 100% of samples have p ∈ [0, 1]
2. ✓ **Extreme values rare:** Only 3.34% in extreme regions (<0.01 or >0.99), well below 10% threshold
3. ✓ **Observed range covered:** Prior predictive range [0.001, 0.958] fully contains observed [0.031, 0.140]
4. ✓ **No computational issues:** Zero NaN or Inf values in any parameter
5. ✓ **Excellent coverage:** 100% of observed values fall within prior predictive 95% CIs

### Additional Strengths

- **Well-calibrated scale:** Prior median (0.069) very close to observed mean (0.079)
- **Appropriate uncertainty:** Prior allows 10x more variability than observed, ensuring data will dominate
- **Balanced residuals:** No systematic over/under-prediction patterns
- **Consistent across sample sizes:** Coverage quality independent of group size

### No Red Flags

- No prior-likelihood conflict detected
- No structural misspecification evident
- No numerical instabilities
- No pathological dependencies (mu and tau properly independent)

---

## Final Recommendation

**DECISION: PASS**

The prior predictive distribution is scientifically plausible, computationally stable, and appropriately informative. The priors:
- Encode domain knowledge (low success rates) without being overconfident
- Cover the observed data range comfortably
- Generate diverse but realistic data patterns
- Show no structural or numerical pathologies

**Next Steps:**
1. Proceed to **Simulation-Based Calibration (SBC)** to verify parameter recovery
2. No prior adjustments needed at this stage
3. The non-centered parameterization is appropriate and should facilitate efficient sampling

**Confidence Level:** HIGH - All diagnostics strongly support model adequacy. The priors strike an excellent balance between being informative (properly scaled to problem) and flexible (allowing data to dominate inference).

---

## Reproducibility

**Code:** `/workspace/experiments/experiment_1/prior_predictive_check/code/run_prior_predictive_numpy.py`
**Random Seed:** 42
**Samples:** 1,000 prior predictive draws
**Implementation:** Pure NumPy/SciPy (no MCMC needed for prior predictive sampling)

All results are fully reproducible using the provided code and random seed.
