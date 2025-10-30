# Posterior Predictive Check Findings
## Experiment 1: Hierarchical Normal Model

**Date:** 2025-10-28
**Model:** Hierarchical Normal Model with known within-study variance
**Data:** 8 studies, observed effects y_i ~ N(theta_i, sigma_i²)
**Posterior Samples:** 20,000 (4 chains × 5,000 draws)

---

## Executive Summary

**OVERALL ASSESSMENT: GOOD FIT**

The hierarchical normal model demonstrates excellent posterior predictive performance. All nine test statistics show good calibration (p-values in [0.05, 0.95]), and 7 of 8 studies exhibit good study-specific fit. The model successfully captures:
- Central tendency (pooled mean effect)
- Between-study variation (dispersion)
- Extreme values (range, min, max)
- Study-specific effects with appropriate uncertainty

One study (Study 8) shows marginal fit (p=0.949), but this is not substantively concerning and represents expected sampling variability with 8 studies.

**Recommendation:** Proceed with confidence. The model is well-calibrated and suitable for inference.

---

## Plots Generated

Visual evidence for all findings is provided in the following diagnostic plots:

| Plot File | Aspect Tested | Key Finding |
|-----------|--------------|-------------|
| `study_level_ppc.png` | Study-specific predictive distributions | All observed values fall within 95% predictive intervals; no systematic misfit |
| `test_statistics_checks.png` | Global test statistics (mean, SD, range, etc.) | All 9 test statistics well-calibrated (p ∈ [0.29, 0.85]) |
| `predictive_intervals.png` | Predictive vs posterior intervals | Observed data within predictive intervals; appropriate uncertainty quantification |
| `standardized_residuals.png` | Residual patterns | No systematic patterns; all residuals within ±2 SD of predicted |
| `qq_plot_calibration.png` | Distributional calibration | Residuals follow standard normal distribution; good model calibration |
| `pooled_statistics.png` | Central tendency, dispersion, extremes | Model captures mean (p=0.35), SD (p=0.71), and range (p=0.76) |
| `observed_vs_replicated.png` | Overall data generation | Observed data indistinguishable from replicated datasets |
| `study_pvalues.png` | Study-specific fit quality | 7/8 studies good fit, 1/8 marginal (Study 8, p=0.949) |

---

## Test Statistics: Bayesian p-values

We compute test statistics T(y) on observed data and compare to the distribution of T(y_rep) from 20,000 posterior predictive replications. Bayesian p-value = P(T_rep ≥ T_obs | data).

**Interpretation:**
- Good fit: 0.05 ≤ p ≤ 0.95 (observed value consistent with model)
- Marginal fit: 0.025 ≤ p < 0.05 or 0.95 < p ≤ 0.975
- Poor fit: p < 0.025 or p > 0.975

### Global Test Statistics

| Statistic | T_obs | T_rep (mean) | T_rep (SD) | p-value | Assessment |
|-----------|-------|--------------|------------|---------|------------|
| **Mean** | 12.50 | 10.02 | 6.26 | 0.345 | Good |
| **SD** | 11.15 | 13.65 | 4.13 | 0.707 | Good |
| **Min** | -4.88 | -10.02 | 10.08 | 0.322 | Good |
| **Max** | 26.08 | 30.64 | 10.89 | 0.638 | Good |
| **Range** | 30.96 | 40.66 | 13.18 | 0.758 | Good |
| **n_negative** | 1.00 | 1.91 | 1.35 | 0.850 | Good |
| **Q25** | 5.35 | 2.17 | 6.91 | 0.325 | Good |
| **Q75** | 21.45 | 17.68 | 7.22 | 0.292 | Good |
| **IQR** | 16.10 | 15.51 | 6.39 | 0.414 | Good |

**Summary:** 9/9 test statistics show good fit. No evidence of systematic misfit.

**Visual Evidence:** All test statistics shown in `test_statistics_checks.png` have observed values near the center of the posterior predictive distribution, confirming good calibration across multiple data features.

---

## Study-Specific Diagnostics

We assess whether each observed y_i is consistent with its predictive distribution by computing standardized residuals: z_i = (y_i - theta_i) / sigma_i.

| Study | y_obs | theta (mean) | sigma | z-score | p-value | Assessment |
|-------|-------|--------------|-------|---------|---------|------------|
| 1 | 20.02 | 11.26 | 15 | 0.58 | 0.555 | Good |
| 2 | 15.30 | 11.04 | 10 | 0.43 | 0.669 | Good |
| 3 | 26.08 | 11.88 | 16 | 0.89 | 0.384 | Good |
| 4 | 25.73 | 13.17 | 11 | 1.14 | 0.258 | Good |
| 5 | -4.88 | 5.85 | 9 | -1.19 | 0.234 | Good |
| 6 | 6.08 | 8.96 | 11 | -0.26 | 0.794 | Good |
| 7 | 3.17 | 8.16 | 10 | -0.50 | 0.622 | Good |
| 8 | 8.55 | 9.70 | 18 | -0.06 | 0.949 | Marginal |

**Summary:** 7/8 studies show good fit, 1/8 marginal (Study 8).

**Key Findings:**

1. **Study 5** (y = -4.88, only negative effect): Despite being an outlier relative to other studies, it shows good fit (p=0.234). The model successfully accommodates this negative effect through appropriate posterior uncertainty in theta_5. This is evident in `study_level_ppc.png` (panel 5), where the observed value falls well within the predictive distribution.

2. **Studies 3-4** (largest effects, y ~ 26): Both show good fit (p=0.384, 0.258). The model generates appropriate uncertainty to capture these larger effects, as shown in `predictive_intervals.png`.

3. **Study 8** (marginal fit, p=0.949): The observed value (y=8.55) is very close to the posterior mean (theta=9.70), resulting in a z-score near zero (z=-0.06). The marginal p-value reflects that the observed residual is unusually small (close to predicted mean), not that the model fails to accommodate the data. This is visible in `standardized_residuals.png`, where Study 8's residual is closest to zero. This represents benign sampling variability, not model inadequacy.

**Visual Evidence:** `study_level_ppc.png` shows all 8 studies with observed values (red dashed lines) falling within the posterior predictive distributions (blue histograms). The 95% predictive intervals (light blue shading) consistently contain the observed data.

---

## Calibration Assessment

### Q-Q Plot Analysis

The Q-Q plot (`qq_plot_calibration.png`) compares the distribution of standardized residuals to a standard normal distribution.

**Findings:**
- Points fall close to the identity line (black dashed), indicating good calibration
- All residuals within approximate 95% confidence bands (gray shading)
- No systematic deviations suggesting model misspecification
- Small deviations expected with n=8 studies

**Interpretation:** The model's predictive distribution is well-calibrated. Residuals follow the expected normal distribution, confirming that the hierarchical structure appropriately models study-level variation.

---

## Visual Diagnosis Summary

| Aspect Tested | Plot File | Finding | Implication |
|---------------|-----------|---------|-------------|
| Study-level predictive distributions | `study_level_ppc.png` | All observed within 95% intervals | Excellent study-specific calibration |
| Global test statistics | `test_statistics_checks.png` | All p-values in [0.29, 0.85] | No systematic global misfit |
| Predictive intervals | `predictive_intervals.png` | Observed data within predictive bands | Appropriate uncertainty quantification |
| Residual patterns | `standardized_residuals.png` | No systematic patterns, all |z| < 2 | No detectable model deficiencies |
| Distributional calibration | `qq_plot_calibration.png` | Residuals follow N(0,1) | Well-calibrated likelihood |
| Central tendency | `pooled_statistics.png` (panel A) | Observed mean (12.5) within predictive dist | Captures pooled effect |
| Dispersion | `pooled_statistics.png` (panel B) | Observed SD (11.2) within predictive dist | Captures between-study variation |
| Extremes | `pooled_statistics.png` (panel C) | Observed range (31) within predictive dist | Handles extreme values appropriately |
| Overall data structure | `observed_vs_replicated.png` | Observed indistinguishable from replicates | Model generates realistic data |
| Study-specific fit quality | `study_pvalues.png` | 7/8 good (green), 1/8 marginal (orange) | Minimal study-specific misfit |

---

## Systematic Patterns: None Detected

**Checked for:**
- Systematic over/under-prediction by study → Not found (residuals scattered around zero in `standardized_residuals.png`)
- Heteroskedasticity (variance related to effect size) → Not found (residuals uniform across studies)
- Outlier studies with extreme p-values → Not found (all p-values > 0.23)
- Non-normality of residuals → Not found (Q-Q plot shows good fit to normal)
- Failure to capture extreme values → Not found (observed min/max within predictive distributions)

**Conclusion:** The model does not exhibit any systematic deficiencies that would invalidate inference.

---

## Interpretation: What Good Fit Reveals

The excellent posterior predictive performance indicates:

1. **Appropriate likelihood specification:** The normal likelihood with known variance accurately describes within-study variation. Observed data are consistent with Normal(theta_i, sigma_i²).

2. **Appropriate prior-to-posterior updating:** The hierarchical structure successfully borrows strength across studies while allowing study-specific effects. Strong shrinkage (70-88%, from posterior inference) is validated by the model's ability to generate data matching observed patterns.

3. **Well-calibrated uncertainty:** Predictive intervals appropriately balance precision and coverage. The model neither over-confidently predicts (too narrow intervals) nor is excessively uncertain (too wide intervals).

4. **No evidence of model misspecification:** Absence of systematic residual patterns, good calibration across multiple test statistics, and consistency across all studies suggest the hierarchical normal model is an appropriate representation of the data-generating process.

5. **Study 5's negative effect is accommodated:** Despite being the only negative effect, Study 5 shows good fit (p=0.234). The model's hierarchical structure allows theta_5 to differ from the pooled mean (mu) while maintaining partial pooling. This demonstrates the model's flexibility.

6. **Study 8's marginal fit is benign:** The marginal p-value (0.949) for Study 8 arises because the observed value is very close to the predicted mean (z=-0.06). This is not a failure of the model but rather an instance where the data happened to fall near the center of the predictive distribution. With 8 studies, observing one p-value near 0.05 or 0.95 is consistent with sampling variability (expected ~1 out of 10 under the null).

---

## Comparison to LOO Diagnostics

**Consistency with LOO results:**
- LOO indicated excellent predictive performance (all Pareto k < 0.7, max k = 0.647)
- PPC confirms this: model generates data consistent with observed features
- Both diagnostics converge on "good fit" conclusion

**Why both matter:**
- LOO: Assesses out-of-sample predictive accuracy (leave-one-out)
- PPC: Assesses whether model captures important data features (global and study-specific)

Both passing strengthens confidence that the model is well-specified for this data.

---

## Limitations and Caveats

1. **Small sample size (n=8):** With only 8 studies, we have limited power to detect subtle model misspecification. The good fit observed is consistent with the model being correct, but does not definitively rule out all possible alternatives.

2. **Single negative effect:** Study 5 is the only negative effect. While the model accommodates it well (p=0.234), we have limited information about the model's behavior with predominantly negative effects or balanced positive/negative studies.

3. **Known variance assumption:** This PPC validates the model conditional on the assumption that sigma_i are known exactly. If the within-study variances are actually uncertain or mis-estimated, the good fit observed here could be misleading.

4. **Test statistics chosen:** We examined 9 test statistics covering central tendency, dispersion, and extremes. Other features (e.g., autocorrelation if data were sequential, spatial patterns) were not assessed but are not relevant for this cross-sectional meta-analysis.

5. **Study 8's marginal fit:** While we interpret this as benign, if additional studies showed similar marginal p-values near 1.0, it might suggest the model is slightly over-predicting variation (making predictions too close to observed data). With a single instance out of 8, this is not concerning.

---

## Recommendations

**Proceed with confidence.** The hierarchical normal model is well-calibrated and appropriate for inference. Specifically:

1. **Use posterior estimates for scientific conclusions:** The model's excellent fit validates using posterior means, credible intervals, and derived quantities (e.g., I², predictive distributions for new studies) for substantive interpretation.

2. **Predictive intervals are reliable:** When predicting effects in new studies, use the posterior predictive distribution. The PPC demonstrates these intervals have good coverage properties.

3. **No model refinement needed:** The absence of systematic misfit means we do not need to consider alternative specifications (e.g., heavier-tailed likelihoods, heterogeneous tau, covariate adjustments) unless motivated by domain knowledge rather than fit diagnostics.

4. **Compare to alternative models cautiously:** If comparing to other models (e.g., fixed effects, robust hierarchical models), this PPC establishes a high bar. Alternative models should demonstrate comparable or better fit to be preferred.

5. **Sensitivity analysis optional but not urgent:** Given the excellent fit, sensitivity to priors or likelihood assumptions is less critical. If conducted, sensitivity analysis is more about understanding robustness than addressing fit concerns.

---

## Technical Details

**Posterior Predictive Sampling:**
- For each of 20,000 posterior samples (mu, tau, theta_1, ..., theta_8):
  - Generated y_rep_i ~ Normal(theta_i, sigma_i) for i = 1, ..., 8
- This produces 20,000 replicated datasets of 8 studies each

**Test Statistics Computation:**
- Observed: T_obs = T(y_obs)
- Replicated: T_rep = {T(y_rep^(1)), ..., T(y_rep^(20000))}
- Bayesian p-value: P(T_rep ≥ T_obs) = (1/20000) * sum(T_rep ≥ T_obs)

**Study-Specific p-values:**
- Computed P(|z_rep| ≥ |z_obs|) where z = (y - theta) / sigma
- Tests whether observed standardized residual is unusually large in magnitude

**Software:**
- ArviZ InferenceData from Stan/PyMC fit (confirmed hierarchical model)
- Posterior samples: 4 chains × 5,000 draws = 20,000 samples
- All computations use full posterior (not thinned or subsampled)

---

## Conclusion

The hierarchical normal model passes all posterior predictive checks with flying colors. The model:
- Captures central tendency (p=0.345)
- Captures dispersion (p=0.707)
- Captures extremes (p=0.758 for range)
- Fits individual studies well (7/8 good, 1/8 marginal)
- Shows no systematic residual patterns
- Is well-calibrated (Q-Q plot, uniform coverage)

**This level of fit quality, combined with convergence diagnostics (all R-hat=1.00) and LOO diagnostics (all k<0.7), provides strong evidence that the hierarchical normal model is appropriate for these data.**

**Final Recommendation:** Proceed to model critique and interpret posterior inferences. The model is fit for purpose.

---

## Files Generated

**Code:**
- `/workspace/experiments/experiment_1/posterior_predictive_check/code/generate_ppc.py` - PPC generation and test statistics
- `/workspace/experiments/experiment_1/posterior_predictive_check/code/create_visualizations.py` - Diagnostic plots
- `/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npz` - Numerical results

**Plots:**
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/study_level_ppc.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/test_statistics_checks.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/predictive_intervals.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/standardized_residuals.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/qq_plot_calibration.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/pooled_statistics.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/observed_vs_replicated.png`
- `/workspace/experiments/experiment_1/posterior_predictive_check/plots/study_pvalues.png`

**Documentation:**
- `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md` - This document
