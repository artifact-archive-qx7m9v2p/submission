# EDA Process Log: Count Distribution Analysis

**Analyst:** Analyst 2
**Focus Area:** Count Distribution & Statistical Properties
**Date:** 2025-10-29

---

## Analysis Workflow

### Round 1: Initial Exploration & Distribution Assessment

**Objective:** Understand basic distributional properties and test Poisson assumption

**Steps Taken:**
1. Loaded data and computed descriptive statistics
2. Calculated distributional moments (mean, variance, skewness, kurtosis)
3. Computed variance-to-mean ratio (key metric for count data)
4. Performed chi-square dispersion test
5. Examined temporal patterns in counts
6. Checked for zero-inflation
7. Performed outlier detection using multiple methods

**Key Findings:**
- Variance/Mean ratio = 70.43 (extreme overdispersion)
- Chi-square test p < 0.001 (reject Poisson)
- No zeros observed (no zero-inflation)
- No statistical outliers detected
- Strong temporal trend (counts increase from ~30 to ~250)

**Hypothesis Generated:**
- H1: Poisson is inappropriate (LIKELY REJECTED)
- H2: Negative Binomial will fit well (TO TEST)
- H3: Dispersion may vary over time (TO TEST)

**Visualizations Created:**
- `01_basic_distribution.png`: Histogram, boxplot, Q-Q plot, empirical CDF

**Next Questions:**
1. How well does Negative Binomial fit?
2. Is dispersion constant or time-varying?
3. Are there alternative distributions to consider?

---

### Round 2: Formal Distribution Fitting & Model Comparison

**Objective:** Test competing distributional hypotheses and compare model fit

**Steps Taken:**
1. Fitted Poisson distribution and conducted goodness-of-fit tests
2. Fitted Negative Binomial using both Method of Moments and MLE
3. Conducted Kolmogorov-Smirnov tests for both distributions
4. Performed Likelihood Ratio test comparing NB vs Poisson
5. Calculated AIC and BIC for model comparison
6. Examined mean-variance relationship using sliding windows
7. Tested for temporal variation in dispersion (Levene's test)

**Key Findings:**
- Poisson: KS p < 0.001, AIC = 2954 (REJECTED)
- Negative Binomial: KS p = 0.261, AIC = 456 (ACCEPTED)
- MLE parameters: r = 1.549, p = 0.014
- Delta AIC = 2498 (very strong evidence for NB)
- Levene's test p = 0.010 (significant time-varying dispersion)
- Mean-variance shows clear quadratic relationship

**Hypothesis Results:**
- H1: Poisson is inappropriate ✓ CONFIRMED
- H2: Negative Binomial fits well ✓ CONFIRMED
- H3: Dispersion varies over time ✓ CONFIRMED

**Visualizations Created:**
- `02_poisson_vs_negbinom.png`: Distribution comparison, PMF overlays
- `03_mean_variance_relationship.png`: Rolling statistics, mean-variance scatter
- `04_dispersion_analysis.png`: Temporal dispersion patterns, residuals

**Next Questions:**
1. Are continuous approximations viable?
2. Do extreme values drive overdispersion?
3. How do residuals look for the fitted models?

---

### Round 3: Diagnostic Analysis & Robustness Checks

**Objective:** Validate model fit quality and test robustness of conclusions

**Steps Taken:**
1. Created Q-Q plots for both Poisson and Negative Binomial
2. Created P-P plots for CDF comparison
3. Calculated and examined Pearson residuals
4. Generated rootograms (specialized count diagnostic)
5. Performed influence diagnostics
6. Tested alternative distributions (Log-Normal, Gamma)
7. Conducted robustness check by removing extreme values
8. Examined temporal stability of dispersion parameter

**Key Findings:**
- NB Q-Q plot shows excellent fit; Poisson shows severe deviation
- NB residuals approximately standard normal; Poisson residuals biased
- Rootogram confirms NB fit is much better than Poisson
- No influential observations or outliers detected
- Log-Normal and Gamma provide similar fit to NB (AIC within 2 points)
- Overdispersion persists even after removing top 5% (Var/Mean = 66.48)
- Dispersion parameter r varies from 1.0 to 69.1 across time windows

**Robustness Confirmed:**
- Overdispersion is not driven by outliers
- Multiple distributions (NB, Log-Normal, Gamma) agree on fit quality
- Conclusions hold under various diagnostic frameworks

**Visualizations Created:**
- `05_qq_plots.png`: Quantile-quantile diagnostics
- `06_pp_plots.png`: Probability-probability diagnostics
- `07_residual_analysis.png`: Residual patterns and distributions
- `08_rootograms.png`: Hanging rootograms for count data
- `09_influence_diagnostics.png`: Outlier and leverage diagnostics

**Additional Insights:**
- Continuous approximations (Log-Normal, Gamma) are viable for some contexts
- Time-varying dispersion may warrant more complex model specifications
- Early periods show near-Poisson behavior; later periods show high dispersion

---

## Competing Hypotheses Tested

### 1. Distribution Family
- **H0:** Data follows Poisson distribution
  - **Result:** REJECTED (p < 0.001, Delta AIC = 2498)
- **H1:** Data follows Negative Binomial distribution
  - **Result:** ACCEPTED (p = 0.261, excellent diagnostics)
- **H2:** Data follows Log-Normal distribution (continuous approximation)
  - **Result:** VIABLE (p = 0.232, best AIC but loses discrete structure)
- **H3:** Data follows Gamma distribution (continuous approximation)
  - **Result:** VIABLE (p = 0.234, similar to NB fit)

### 2. Overdispersion Source
- **H0:** Overdispersion is artifact of outliers
  - **Result:** REJECTED (persists after outlier removal)
- **H1:** Overdispersion is fundamental property of process
  - **Result:** ACCEPTED (robust across analyses)

### 3. Dispersion Structure
- **H0:** Dispersion is constant over time
  - **Result:** REJECTED (Levene p = 0.010, r varies 1.0-69.1)
- **H1:** Dispersion varies systematically with time/mean
  - **Result:** ACCEPTED (clear temporal patterns observed)

### 4. Zero-Inflation
- **H0:** Data exhibits zero-inflation
  - **Result:** REJECTED (0 zeros observed, expectation ~0.001)
- **H1:** Count process has consistently positive baseline
  - **Result:** ACCEPTED (all counts ≥ 21)

---

## Unexpected Findings

1. **Severity of overdispersion:** Variance/Mean = 70 is much higher than typical
   - Expected: Mild to moderate overdispersion (ratio 2-10)
   - Observed: Extreme overdispersion (ratio > 70)
   - Implication: Very low r parameter (r = 1.5)

2. **Temporal variation in dispersion:** Early periods near-Poisson, later periods overdispersed
   - Expected: Constant dispersion parameter
   - Observed: r varies by factor of 69 across time
   - Implication: May need time-varying dispersion in refined models

3. **No outliers despite large range:** Max/Min = 12.8 but no statistical outliers
   - Expected: Some extreme values given large range
   - Observed: All values consistent with NB distribution
   - Implication: Entire range reflects natural process variability

4. **Continuous distributions fit equally well:** Log-Normal AIC actually best
   - Expected: Discrete models to clearly dominate
   - Observed: Log-Normal, Gamma, NB all within 2 AIC points
   - Implication: Large counts allow accurate continuous approximation

---

## Methodological Decisions

### Why Multiple Outlier Detection Methods?
- Different methods sensitive to different anomaly types
- Triangulation increases confidence in conclusions
- Robust outlier methods (modified Z-score) less affected by skewness

### Why Test Continuous Distributions?
- Large counts (mean = 109) make continuous approximation viable
- Some modeling contexts may prefer continuous likelihoods
- Provides additional validation of discrete model conclusions

### Why Examine Temporal Variation?
- Trend in mean suggests potential trend in dispersion
- Heteroskedasticity affects inference
- Time-varying parameters may improve model fit

### Why Multiple Goodness-of-Fit Tests?
- Each test has different power against different alternatives
- KS test: sensitive to overall distribution shape
- Chi-square: sensitive to local deviations
- Likelihood ratio: directly compares nested models
- Convergent evidence strengthens conclusions

---

## Tentative vs. Robust Findings

### ROBUST (High Confidence)
1. Poisson is inappropriate (multiple tests, huge AIC difference)
2. Negative Binomial provides good fit (passes all diagnostics)
3. Severe overdispersion exists (multiple measures, robust to outliers)
4. No zero-inflation (definitional - 0 zeros observed)
5. No statistical outliers (multiple methods agree)

### TENTATIVE (Moderate Confidence)
1. Time-varying dispersion structure (significant but limited time points)
2. Specific value of r parameter (small sample, wide uncertainty)
3. Continuous approximations equally good (may be sample-specific)

### REQUIRES FURTHER INVESTIGATION
1. Temporal autocorrelation (not examined in this analysis)
2. Causal factors driving dispersion changes (no covariates available)
3. Forecasting performance (no out-of-sample validation)
4. Structural breaks or change points (informal observation only)

---

## Data Quality Flags

### PASSED (No Issues)
- ✓ Completeness (no missing values)
- ✓ Range validity (all positive integers)
- ✓ Temporal consistency (monotonic time index)
- ✓ Outlier assessment (no anomalous values)

### NOTED (For Awareness)
- ⚠ Heteroskedasticity (variance increases over time)
- ⚠ Small sample size (n=40 limits complex modeling)
- ⚠ Single time series (no replication or external validation)

### NOT ASSESSED (Beyond Scope)
- ? Temporal autocorrelation
- ? Seasonal patterns
- ? Covariate relationships
- ? Measurement error

---

## Modeling Recommendations Summary

### PRIMARY RECOMMENDATION
**Use Negative Binomial likelihood with r ≈ 1.5**

**Justification:**
- Very strong statistical evidence (Delta AIC = 2498)
- Passes all goodness-of-fit tests
- Well-behaved residuals
- Respects discrete nature of count data
- Established Bayesian implementations available

### SECONDARY OPTIONS

**Option A: Time-Varying Dispersion NB**
- Use if modeling temporal trend
- Allow r to vary with time or mean
- More complex but may improve fit
- Supported by Levene test (p = 0.010)

**Option B: Log-Normal (Continuous Approximation)**
- Use if discrete structure not critical
- Best AIC (453.99) among all models
- Simpler computation in some frameworks
- Loses count-specific interpretation

**Option C: Gamma (Continuous Approximation)**
- Middle ground between NB and Log-Normal
- Natural for positive continuous data
- Related to NB through Gamma-Poisson mixture
- Similar fit quality to Log-Normal

### DO NOT USE
**Poisson Distribution**
- Completely inappropriate
- Catastrophic lack of fit
- Will severely mislead inference

---

## Files Generated

### Code
- `/workspace/eda/analyst_2/code/01_initial_exploration.py`
- `/workspace/eda/analyst_2/code/02_distribution_visualizations.py`
- `/workspace/eda/analyst_2/code/03_hypothesis_testing.py`
- `/workspace/eda/analyst_2/code/04_diagnostic_plots.py`
- `/workspace/eda/analyst_2/code/05_extreme_value_analysis.py`
- `/workspace/eda/analyst_2/code/06_summary_visualization.py`

### Visualizations
- `/workspace/eda/analyst_2/visualizations/01_basic_distribution.png`
- `/workspace/eda/analyst_2/visualizations/02_poisson_vs_negbinom.png`
- `/workspace/eda/analyst_2/visualizations/03_mean_variance_relationship.png`
- `/workspace/eda/analyst_2/visualizations/04_dispersion_analysis.png`
- `/workspace/eda/analyst_2/visualizations/05_qq_plots.png`
- `/workspace/eda/analyst_2/visualizations/06_pp_plots.png`
- `/workspace/eda/analyst_2/visualizations/07_residual_analysis.png`
- `/workspace/eda/analyst_2/visualizations/08_rootograms.png`
- `/workspace/eda/analyst_2/visualizations/09_influence_diagnostics.png`
- `/workspace/eda/analyst_2/visualizations/10_comprehensive_summary.png`

### Reports
- `/workspace/eda/analyst_2/findings.md` (comprehensive findings report)
- `/workspace/eda/analyst_2/eda_log.md` (this file)

---

## Lessons Learned

1. **Overdispersion magnitude matters:** Not just presence/absence, but degree affects parameter estimates and model choice

2. **Multiple diagnostics essential:** Single test can mislead; convergent evidence provides confidence

3. **Temporal patterns reveal complexity:** What appears uniform may have important time-varying structure

4. **Continuous approximations viable for large counts:** Discrete models theoretically correct but continuous may be practical

5. **Outlier detection requires multiple methods:** Robust statistics important for skewed distributions

6. **Visual diagnostics complement formal tests:** Q-Q plots, rootograms provide intuitive understanding

---

**Analysis Duration:** Approximately 3 rounds of iterative exploration
**Total Visualizations:** 10 multi-panel figures
**Total Hypothesis Tests:** 12 formal statistical tests
**Confidence in Recommendation:** Very High

**Bottom Line:** Use Negative Binomial with r ≈ 1.5 for modeling this count data.
