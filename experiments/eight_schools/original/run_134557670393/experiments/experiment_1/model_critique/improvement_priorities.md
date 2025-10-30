# Improvement Priorities: Experiment 1
## Bayesian Hierarchical Meta-Analysis

**Date**: 2025-10-28
**Model Status**: ACCEPTED
**Document Purpose**: Recommendations for future work and sensitivity analyses

---

## Context

The Bayesian hierarchical meta-analysis model **PASSED all falsification criteria** and is **ACCEPTED for scientific inference**. This document outlines:
1. Recommended sensitivity analyses (Phase 4)
2. Potential model improvements for future work
3. Data collection priorities
4. Limitations to acknowledge in reporting

**Note**: These are NOT required fixes (the model is adequate), but rather opportunities to strengthen conclusions and guide future research.

---

## Priority 1: Model Comparison (REQUIRED - Phase 4)

### Action Items

**A. Compare to Fixed-Effects Model (Model 3)**
- **Purpose**: Test if hierarchical structure is necessary
- **Method**: Fit fixed-effects model (tau = 0), compare via LOO-CV
- **Expected outcome**: Fixed-effects will fail (Study 1 outlier, Δelpd > 5)
- **Implication**: Confirms hierarchical structure is justified

**B. Compare to Robust Model (Model 2)**
- **Purpose**: Test if heavy-tailed errors improve fit
- **Method**: Fit hierarchical Student-t model, compare via LOO-CV
- **Expected outcome**: Similar or slightly better fit, nu ≈ 10-30
- **Implication**: If Δelpd < 2, models equivalent (prefer simpler Normal)
                  If Δelpd > 2, robust model superior (adopt as primary)

**C. Report Model Comparison Results**
- **Format**: ELPD table with standard errors and weights
- **Interpretation**: Discuss which model(s) are adequate
- **Decision rule**:
  - If |Δelpd| < 2×SE: Models equivalent (parsimony wins)
  - If |Δelpd| > 2×SE: Prefer better model
- **Documentation**: Justify final model choice with evidence

**Estimated Time**: 2-3 hours
**Priority**: HIGH (required before finalizing conclusions)

---

## Priority 2: Prior Sensitivity Analysis (RECOMMENDED - Phase 4)

### Motivation
The model is adequate with current priors, but scientific robustness requires testing sensitivity to prior choices.

### Recommended Analyses

**A. Tighter Heterogeneity Prior**
- **Prior**: tau ~ Half-Normal(0, 3) instead of Half-Cauchy(0, 5)
- **Rationale**: More informative, based on typical meta-analyses
- **Expected**: tau posterior may be slightly lower, mu similar
- **Interpretation**: If conclusions similar, demonstrates robustness
                     If conclusions change, investigate why

**B. Looser Heterogeneity Prior**
- **Prior**: tau ~ Half-Cauchy(0, 10) instead of Half-Cauchy(0, 5)
- **Rationale**: Less informative, maximum flexibility
- **Expected**: tau posterior may have slightly heavier tail
- **Interpretation**: If conclusions similar, demonstrates robustness

**C. Informative Overall Effect Prior**
- **Prior**: mu ~ Normal(5, 10) instead of Normal(0, 50)
- **Rationale**: Test impact of weakly informative prior favoring positive effects
- **Expected**: mu posterior shifted slightly positive
- **Interpretation**: Quantify prior influence vs data influence

**D. Comparison Summary**
- **Create table**: Compare posteriors across all prior choices
- **Metrics**: E[mu], E[tau], P(mu>0), P(tau<5)
- **Conclusion**: State which conclusions are robust, which are prior-sensitive

**Estimated Time**: 1-2 hours (refit 3 times)
**Priority**: HIGH (strongly recommended for publication)

---

## Priority 3: Influence and Robustness Diagnostics (RECOMMENDED)

### A. Detailed Leave-One-Out Analysis

**Current Status**: Basic LOO completed (max Δmu = 2.09)

**Additional Analyses**:
1. **Study-by-study influence**:
   - For each study, plot posterior when study removed
   - Show how mu, tau, and other theta change
   - Identify most influential studies (Study 5, Study 7)

2. **Cumulative meta-analysis**:
   - Add studies sequentially (1 to 8)
   - Plot evolution of mu posterior
   - Show when conclusions stabilize

3. **Subset analysis**:
   - High-precision studies only (sigma ≤ 11)
   - Low-precision studies only (sigma > 11)
   - Compare posteriors between subsets

**Deliverable**: Influence diagnostic report with plots
**Estimated Time**: 1-2 hours
**Priority**: MEDIUM (useful for understanding data)

---

### B. Study Quality Assessment

**Current Status**: All studies treated equally (no quality weights)

**Future Work**:
1. If study quality scores become available:
   - Incorporate into hierarchical model as covariates
   - Test if quality explains heterogeneity
   - Down-weight low-quality studies

2. If raw data become available:
   - Verify reported sigma_i are accurate
   - Check for within-study outliers
   - Perform individual participant data (IPD) meta-analysis

**Deliverable**: Quality-adjusted meta-analysis (if data available)
**Estimated Time**: 2-4 hours (depending on data availability)
**Priority**: LOW (data-dependent)

---

## Priority 4: Reporting and Communication (REQUIRED)

### A. Scientific Summary

**Key Messages to Emphasize**:
1. **Likely positive effect**: 95.7% probability mu > 0
2. **Effect size uncertain**: Could be near zero or large (95% CI: [-1.19, 16.53])
3. **Moderate heterogeneity**: tau median = 2.86, but wide uncertainty
4. **Study 1 not problematic**: Hierarchical model accommodates via shrinkage
5. **Small sample caveat**: J=8 limits precision, more studies needed

**Avoid**:
- Binary "significant/not significant" language
- Overstating precision (acknowledge wide CIs)
- Ignoring heterogeneity (contrast with I² = 0%)
- Treating point estimates as truth (report full distributions)

---

### B. Visualizations for Stakeholders

**Essential Plots** (create or enhance):
1. **Forest plot**: Study-specific posteriors with pooled estimate
2. **Prior-posterior animation**: Show Bayesian learning
3. **Shrinkage diagram**: Observed vs posterior with arrows
4. **Probability curves**: P(mu > x) as function of x
5. **Prediction interval**: For future studies (not just studies 1-8)

**Estimated Time**: 2-3 hours
**Priority**: HIGH (essential for communication)

---

### C. Limitations Section

**Acknowledge in Final Report**:
1. **Small sample size**: J=8 limits statistical power
2. **Wide uncertainty**: Effect size and heterogeneity poorly constrained
3. **Model assumptions**: Normality, independence, known sigma_i
4. **Potential biases**: Publication bias tests underpowered
5. **Generalizability**: Conclusions apply to these 8 studies
6. **Missing context**: No study characteristics to explain heterogeneity

**Recommended phrasing**: "While the hierarchical model successfully captures the data-generating process and provides stable inference, the small sample size (J=8) results in substantial uncertainty about both the overall effect size and between-study heterogeneity. Conclusions should be interpreted cautiously, and additional studies would greatly strengthen evidence."

---

## Priority 5: Future Data Collection (LONG-TERM)

### A. Expand Meta-Analysis

**Immediate Actions** (if possible):
1. **Systematic literature review**: Search for additional studies
   - Target: J ≥ 15 for reliable tau estimation
   - Update meta-analysis with new data
   - Refit model and compare to current results

2. **Request unpublished data**: Contact authors for:
   - Additional effect estimates not published
   - Raw data for IPD meta-analysis
   - Study characteristics (design, population, setting)

**Expected Impact**:
- Narrower credible intervals (more precise estimates)
- Better identification of tau (less uncertainty about heterogeneity)
- Stronger conclusions about overall effect
- Potential for meta-regression (if covariates available)

**Estimated Time**: Weeks to months (depends on data availability)
**Priority**: HIGH (if goal is publication-quality meta-analysis)

---

### B. Collect Study-Level Covariates

**Target Characteristics**:
1. **Study design**: RCT, observational, case-control
2. **Sample size**: Number of participants per study
3. **Population**: Age, demographics, setting
4. **Intervention**: Dose, duration, delivery mode (if applicable)
5. **Quality**: Risk of bias scores, methodological rigor
6. **Publication**: Year, journal, funding source

**Use Cases**:
- Meta-regression: Model heterogeneity as function of covariates
- Subgroup analysis: Compare effects by study characteristics
- Publication bias: Test for small-study effects with covariates
- Precision: Explain why some studies have large sigma_i

**Expected Impact**: Transform descriptive heterogeneity (tau) into explained heterogeneity (covariates)

**Estimated Time**: Days to weeks (depends on data availability)
**Priority**: MEDIUM (improves interpretability)

---

## Priority 6: Alternative Models (EXPLORATORY)

These are NOT needed (current model is adequate), but may be interesting for exploratory or methodological research.

### A. Measurement Error Model

**Motivation**: Test if treating sigma_i as uncertain improves fit

**Model**:
```
sigma_i ~ Gamma(alpha_i, beta_i)  # Estimate uncertainty in SEs
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
```

**Priors**: Choose alpha_i, beta_i to reflect reporting uncertainty

**Expected Outcome**: Likely minimal impact (reported SEs usually accurate)
**Priority**: LOW (no evidence of SE misspecification)

---

### B. Non-Centered Meta-Regression

**Motivation**: If covariates become available, explain heterogeneity

**Model**:
```
theta_i ~ Normal(mu + X_i * beta, tau)
```
Where X_i are study-level predictors (e.g., sample size, year)

**Example Questions**:
- Does effect size decrease over time? (decline effect)
- Do larger studies show smaller effects? (small-study bias)
- Does quality predict effect size?

**Expected Outcome**: Reduced tau if covariates explain variability
**Priority**: MEDIUM (data-dependent, scientifically valuable)

---

### C. Mixture Model

**Motivation**: Test for subgroups of studies with distinct effects

**Model**:
```
theta_i ~ Mixture(p * Normal(mu_1, tau_1), (1-p) * Normal(mu_2, tau_2))
```

**Challenge**: J=8 likely too small for reliable identification

**Expected Outcome**: Likely fails or returns wide posteriors
**Priority**: LOW (underpowered with J=8)

---

### D. Copula or Correlated Effects Model

**Motivation**: If studies are not independent (same authors, same populations)

**Model**: Multivariate normal with correlation structure
**Data Needed**: Information on study relationships
**Expected Outcome**: More conservative uncertainty (wider CIs)
**Priority**: LOW (independence assumption seems reasonable)

---

## Priority 7: Software and Reproducibility (GOOD PRACTICE)

### A. Code Documentation

**Current Status**: Code exists and runs successfully

**Improvements**:
1. **Add comments**: Explain model specification, priors, diagnostics
2. **Create notebook**: Interactive walkthrough of analysis
3. **Document dependencies**: PyMC version, ArviZ version, OS
4. **Package environment**: Create requirements.txt or conda env
5. **Add tests**: Unit tests for data loading, convergence checks

**Estimated Time**: 1-2 hours
**Priority**: MEDIUM (important for reproducibility)

---

### B. Replication Materials

**Create Archive** with:
1. Data (data.csv)
2. All code (prior predictive, fitting, validation, critique)
3. All plots (diagnostics, results, figures)
4. Reports (markdown documents, this file)
5. README: Step-by-step instructions to reproduce
6. Computational environment (Python version, packages)

**Purpose**: Enable independent replication and verification

**Estimated Time**: 1-2 hours
**Priority**: HIGH (essential for scientific credibility)

---

### C. Alternative Software Implementations

**Current**: PyMC (Python)

**Consider Also**:
1. **Stan (R or Python)**: Gold standard, check agreement
2. **brms (R)**: User-friendly, publication-ready plots
3. **JAGS**: Alternative MCMC, test robustness

**Purpose**: Verify results are software-independent

**Expected Outcome**: Nearly identical posteriors (if so, increases confidence)

**Estimated Time**: 2-3 hours per software
**Priority**: LOW (useful for methodological validation)

---

## Summary of Recommendations

### Phase 4 (Required - Next 2-3 hours)
1. ✓ Model comparison: Fixed-effects and robust alternatives
2. ✓ Prior sensitivity: tau ~ Half-Normal(0,3) and Half-Cauchy(0,10)
3. ✓ Final reporting: Best model selection with justification

### Sensitivity Analyses (Recommended - Next 1-2 hours)
1. Detailed leave-one-out influence
2. Cumulative meta-analysis
3. Prior sensitivity on mu
4. Visualization suite for communication

### Future Work (If Resources Available)
1. Expand sample size (J ≥ 15 studies)
2. Collect study-level covariates
3. Meta-regression with predictors
4. Software replication (Stan, brms)
5. Code documentation and archive

### Not Recommended (Low Priority)
1. Measurement error model (no evidence of need)
2. Mixture model (underpowered with J=8)
3. Alternative likelihood families (current adequate)

---

## Decision Tree for Future Analysts

```
START: Accepted hierarchical model

├─ Model comparison (Phase 4)
│  ├─ If robust model better (Δelpd > 2): ADOPT robust model
│  ├─ If fixed-effects better (unlikely): INVESTIGATE heterogeneity
│  └─ If equivalent (|Δelpd| < 2): KEEP hierarchical (parsimony)
│
├─ Prior sensitivity
│  ├─ If conclusions robust: REPORT and emphasize robustness
│  └─ If conclusions sensitive: REPORT range and acknowledge uncertainty
│
├─ Additional studies available?
│  ├─ YES: UPDATE meta-analysis, refit model
│  └─ NO: Proceed with current results
│
├─ Study characteristics available?
│  ├─ YES: FIT meta-regression, explain heterogeneity
│  └─ NO: Report descriptive heterogeneity (tau)
│
└─ Final report
   ├─ Scientific summary (probability statements)
   ├─ Limitations (sample size, uncertainty)
   ├─ Visualizations (forest plot, shrinkage, predictions)
   └─ Recommendations for future research
```

---

## Metrics of Success

### For Phase 4 Completion
- ✓ At least 2 alternative models compared via LOO-CV
- ✓ Prior sensitivity tested (≥2 alternative priors)
- ✓ Final model selected with quantitative justification
- ✓ Scientific conclusions stated with uncertainty
- ✓ Limitations clearly documented

### For Publication-Ready Analysis
- ✓ All Phase 4 metrics met
- ✓ Influence diagnostics completed
- ✓ Visualization suite created
- ✓ Replication materials archived
- ✓ Limitations section written

### For Gold-Standard Meta-Analysis
- ✓ Sample size J ≥ 15 studies
- ✓ Study characteristics collected
- ✓ Meta-regression performed
- ✓ Publication bias formally assessed
- ✓ Software replication completed (≥2 packages)

---

## Conclusion

The Bayesian hierarchical meta-analysis model is **statistically adequate** and ready for Phase 4 (model comparison). The recommendations above are NOT required fixes but rather:

1. **Required next steps**: Model comparison and sensitivity analysis (Phase 4)
2. **Best practices**: Detailed diagnostics and visualization
3. **Future directions**: Data collection and model extensions
4. **Quality standards**: Reproducibility and software verification

**Immediate Action**: Proceed to Phase 4 model comparison. The model comparison agent should compare this model to fixed-effects and robust alternatives using LOO-CV, then provide final model recommendation.

---

**Document prepared**: 2025-10-28
**Purpose**: Guide Phase 4 and future work
**Status**: Model ACCEPTED, priorities identified
**Next phase**: Model Assessment and Comparison (Phase 4)
