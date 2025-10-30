# Key Figures Guide
## Bayesian Meta-Analysis Visual Summary

**Report**: Bayesian Hierarchical Meta-Analysis of Eight Studies
**Date**: October 28, 2025

---

## Figure Index

This document provides a guide to all essential figures in the final report, with captions explaining what each visualization demonstrates and what to notice.

---

## Figure 1: Forest Plot with Study Estimates and Pooled Effect

**File**: `/workspace/final_report/figures/fig1_forest_plot.png`

**Source**: EDA Phase (Analyst #1)

**What it shows**:
- Individual effect estimates (y_i) for each of 8 studies with 95% confidence intervals
- Pooled effect estimate from classical fixed-effects meta-analysis (diamond)
- Study precision reflected in CI width (wider = less precise)
- All studies with CIs crossing zero (none individually significant)

**What to notice**:
- Wide, overlapping confidence intervals across all studies
- Study 1 (y=28) has highest effect but also large uncertainty (SE=15)
- Study 3 (y=-3) only negative effect
- Pooled estimate (7.69) just barely excludes zero (95% CI: -0.30 to 15.67)
- Classical analysis shows borderline significance despite no individual study significant

**Key insight**: Meta-analysis reveals signal that individual studies are too noisy to detect alone.

---

## Figure 2: Posterior Distributions for mu and tau

**File**: `/workspace/final_report/figures/fig2_posterior_distributions.png`

**Source**: Posterior Inference Phase

**What it shows**:
- Posterior distribution for population mean effect (mu)
- Posterior distribution for between-study heterogeneity (tau)
- Density plots showing full shape of uncertainty
- Credible intervals marked

**What to notice**:
- **mu distribution**: Centered around 7.75, right-skewed, substantial spread
  - 95% CI: -1.19 to 16.53 (includes zero but mass concentrated on positive side)
  - Mode slightly below mean due to right skew
- **tau distribution**: Right-skewed with mode near 2, long upper tail
  - 95% CI: 0.14 to 11.32 (includes zero but probability mass suggests tau > 0)
  - Median (1.98) lower than mean (2.86) due to skewness

**Key insight**: Full posterior distributions show uncertainty shape beyond simple intervals. Right skewness indicates long tails of possibility.

---

## Figure 3: Prior vs Posterior Comparison for tau

**File**: `/workspace/final_report/figures/fig3_prior_posterior_tau.png`

**Source**: Model Critique Phase

**What it shows**:
- Prior distribution for tau: Half-Cauchy(0, 5)
- Posterior distribution for tau (after seeing data)
- Overlap showing how data updated beliefs

**What to notice**:
- Prior (red/orange): Heavy-tailed, allows wide range of tau values
- Posterior (blue): Concentrated around tau=2-3, upper tail less heavy than prior
- Data shifted probability mass from higher values toward moderate heterogeneity
- Posterior mode near 2, consistent with moderate heterogeneity
- Prior still visible in posterior tail (some prior influence remains)

**Key insight**: Data learned appropriate heterogeneity scale, shifting from diffuse prior to concentrated posterior while retaining flexibility. No prior-posterior conflict (data and prior compatible).

---

## Figure 4: Posterior Predictive Check Summary

**File**: `/workspace/final_report/figures/fig4_posterior_predictive.png`

**Source**: Posterior Predictive Check Phase

**What it shows**:
- Observed effect sizes (y_i) for each study (points)
- 50% and 90% posterior predictive intervals (shaded regions)
- How well model-generated data matches observed data

**What to notice**:
- All 8 studies fall within 95% posterior predictive intervals (excellent fit)
- Most studies near center of predictive distribution
- Study 1 (y=28) in upper tail but within bounds (p=0.244)
- Study 3 (y=-3) in lower tail but within bounds
- No posterior predictive outliers (0/8)

**Key insight**: Model captures data-generating process well. Zero outliers passes falsification criterion (reject if >1 outlier).

---

## Figure 5: LOO Pareto k Diagnostics

**File**: `/workspace/final_report/figures/fig5_loo_diagnostics.png`

**Source**: Model Critique Phase

**What it shows**:
- Pareto k values for each study (measuring LOO reliability)
- Threshold lines for diagnostic interpretation
- Color-coded by reliability category

**What to notice**:
- All 8 studies: k < 0.7 (reliable LOO)
- 6 studies: k < 0.5 (excellent reliability)
- 2 studies (4,5): 0.5 < k < 0.7 (good reliability)
- No studies in problematic range (k > 0.7)
- Studies 4 and 5 slightly more influential but still fine

**Key insight**: Leave-one-out cross-validation approximations are highly trustworthy. No need for more expensive K-fold CV or moment matching.

**Threshold interpretation**:
- k < 0.5: Excellent (green)
- 0.5 ≤ k < 0.7: Good (yellow)
- 0.7 ≤ k < 1.0: Bad (orange)
- k ≥ 1.0: Very bad (red)

---

## Figure 6: LOO-PIT Calibration Plot

**File**: `/workspace/final_report/figures/fig6_calibration.png`

**Source**: Model Assessment Phase

**What it shows**:
- Left: Histogram of LOO probability integral transform values
- Right: Q-Q plot comparing LOO-PIT to uniform distribution
- Uniformity test results (KS statistic, p-value)

**What to notice**:
- Histogram approximately flat (uniform distribution)
- Q-Q plot points follow diagonal line closely
- KS test p=0.975 (strong evidence for uniformity, p>0.05)
- No systematic deviations or patterns

**Key insight**: Model's predictive distributions are well-calibrated globally. Neither systematically overconfident nor underconfident. LOO-PIT uniformity indicates proper uncertainty quantification.

**What uniformity means**: If model is well-calibrated, the probability that y_i falls below its predicted value should be uniform on [0,1]. Deviations suggest mis-calibration.

---

## Figure 7: Shrinkage Plot (Hierarchical Partial Pooling)

**File**: `/workspace/final_report/figures/fig7_shrinkage.png`

**Source**: Posterior Inference Phase

**What it shows**:
- Observed effects (y_i) vs posterior mean study effects (theta_i)
- Arrows showing direction and magnitude of shrinkage
- Population mean (mu=7.75) as reference line

**What to notice**:
- **Study 1 (y=28)**: Massive shrinkage downward (-18.75) to theta=9.25
- **Study 3 (y=-3)**: Large shrinkage upward (+9.98) to theta=6.98
- **Study 7 (y=18)**: Moderate shrinkage downward (-8.91) to theta=9.09
- Extreme observations shrink most toward population mean
- Studies near mean (2,4,6,8) shrink minimally

**Key insight**: Hierarchical model automatically moderates extreme observations through partial pooling. This "borrowing strength" improves predictions by balancing individual study data with population information.

**Why shrinkage helps**: Extreme values are partially due to noise. Shrinking toward the mean reduces overfitting to noise while retaining genuine signals.

---

## Additional Figures (Supplementary)

The following figures are available in `/workspace/experiments/experiment_1/` for technical audiences:

### Convergence Diagnostics
- **Trace plots**: MCMC chains over iterations (check mixing)
- **Rank plots**: Uniform histogram indicates good mixing
- **Autocorrelation**: Decay to near-zero within ~10 lags
- **Energy diagnostic**: Bayesian Fraction of Missing Information

### Posterior Structure
- **Pair plot (mu, tau)**: Joint distribution showing correlation
- **Study-specific posteriors**: Individual theta_i distributions
- **Probability statements**: Visual summary of P(mu>0), P(tau>0), etc.

### Model Critique
- **LOO influence plot**: Delta mu when each study removed
- **Shrinkage diagnostics**: Magnitude vs study precision
- **Residual plots**: Standardized residuals vs fitted values

---

## Visual Summary Box (for Executive Summary)

**The 3 Most Important Figures**:

1. **Figure 2 (Posterior Distributions)**: Shows primary results - mu≈8 with 95.7% probability positive, tau≈3 with 81% probability of heterogeneity

2. **Figure 4 (Posterior Predictive)**: Demonstrates model adequacy - all studies within predictive intervals (0 outliers)

3. **Figure 6 (Calibration)**: Confirms model reliability - well-calibrated predictions (LOO-PIT p=0.975)

**For rapid assessment**: These three figures convey the main findings (mu, tau estimates), model validation (good fit), and predictive reliability (calibration).

---

## Guide to Reading Bayesian Figures

### Credible Intervals
- **50% CI**: Central half of probability mass (median ± ~0.67 SD)
- **90% CI**: 90% of probability mass (5th to 95th percentile)
- **95% CI**: 95% of probability mass (2.5th to 97.5th percentile)

**Interpretation**: "There is a 95% probability the parameter lies in this interval given the data" (direct probability statement).

### Posterior Distributions
- **Height**: Probability density (higher = more probable)
- **Width**: Uncertainty (wider = less certain)
- **Shape**: Skewness, modality, tail behavior
- **Mode**: Most probable value (peak)
- **Mean**: Expected value (average)
- **Median**: 50th percentile (middle value)

**For skewed distributions**: Median often preferred over mean (less influenced by tails).

### Cross-Validation Plots
- **LOO**: Leave-one-out (assess prediction when study removed)
- **Pareto k**: Reliability of LOO approximation
- **LOO-PIT**: Probability integral transform (calibration check)

**Interpretation**: Good LOO diagnostics indicate model generalizes well beyond training data.

---

## Figure Quality Standards

All figures meet publication standards:
- **Resolution**: High-resolution PNG (300 DPI equivalent)
- **Fonts**: Readable at publication size
- **Colors**: Colorblind-friendly palettes where possible
- **Labels**: Clear axis labels, legends, titles
- **Captions**: Comprehensive explanations in report

**Reproducibility**: All figures generated by code in `/workspace/experiments/` with random seed 12345.

---

## Figure Usage Guidelines

**For Publications**:
- Include Figures 1, 2, 4, 6 in main text (core results and validation)
- Figures 3, 5, 7 suitable for supplementary materials
- All figures have detailed captions in main report

**For Presentations**:
- Figure 2 (posteriors) for main findings slide
- Figure 4 (PPC) for model validation slide
- Figure 7 (shrinkage) for explaining hierarchical modeling

**For Stakeholder Reports**:
- Figure 1 (forest plot) for data overview
- Figure 2 (posteriors) for results summary
- Simplified probability statement graphics

---

**Guide Prepared**: October 28, 2025
**Total Figures**: 7 primary + additional supplementary
**Purpose**: Facilitate interpretation and communication of Bayesian meta-analysis results
