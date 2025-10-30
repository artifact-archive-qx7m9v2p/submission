# Model Comparison Matrix: Designer 3 Robust Models

## Quick Reference: When to Use Each Model

| Model Class | Best For | Evidence Required | Computational Cost | Expected Outcome |
|-------------|----------|-------------------|-------------------|------------------|
| **Normal Baseline** | Default analysis | None (from EDA) | Low | Baseline comparison |
| **Student-t Data (1A)** | Suspected measurement outliers | nu < 10 | Medium | nu > 30 (not needed) |
| **Student-t Effects (1B)** | Extreme true effects | nu < 10 | Medium | nu > 30 (not needed) |
| **Student-t Double (1C)** | Maximum robustness | Failures in 1A & 1B | High | Overparameterized |
| **Outlier Indicators (2A)** | Known potential outliers | p_i > 0.8 for some schools | High | All p_i < 0.2 |
| **Latent Classes (2B)** | Theoretical subgroups | Clear bimodality | Very High | Groups collapse |
| **Dirichlet Process (2C)** | Unknown clusters | Model comparison | Extreme | K=1 cluster |
| **Prior Sensitivity** | Robustness to priors | Always useful | Medium (parallel) | Low sensitivity |

## Decision Tree

```
START: Do we need robust models?
│
├─> EDA shows outliers (|z| > 2)?
│   ├─> YES → Try Student-t (1A) or Outliers (2A)
│   └─> NO → [Our case] Skip to prior sensitivity
│
├─> EDA shows heterogeneity (I² > 25%)?
│   ├─> YES → Consider mixture models (2B)
│   └─> NO → [Our case] Standard hierarchical sufficient
│
├─> Domain knowledge suggests subgroups?
│   ├─> YES → Try latent class (2B) with covariates
│   └─> NO → [Our case] No theoretical basis
│
└─> Uncertain about prior choices?
    ├─> YES → [Our case] ALWAYS do prior sensitivity (3)
    └─> NO → Report with single prior, but risky
```

## Expected Results Matrix

| Model | mu_posterior | tau_posterior | LOO-CV (vs baseline) | Conclusion |
|-------|--------------|---------------|----------------------|------------|
| Normal | 7.7 ± 4.1 | ~0-3 | Baseline | Standard |
| Student-t (1A) | 7.7 ± 4.2 | ~0-3, nu ~25 | +0 to +2 | Converges to normal |
| Student-t (1B) | 7.7 ± 4.1 | ~0-3, nu ~25 | +0 to +2 | Converges to normal |
| Outlier (2A) | 7.7 ± 4.1 | ~0-3, pi ~0 | -2 to +1 | No outliers detected |
| Latent Class (2B) | mu1≈mu2≈7.7 | tau1,tau2 ~0-3 | -5 to 0 | Single group |
| Prior Weak | 7.5-8.0 ± 4.0 | ~0-3 | +0 ± 1 | Robust |
| Prior Strong | 7.8 ± 3.8 | ~0-2 | +0 ± 1 | Slight prior influence |

**Key:** LOO-CV differences > 3 are substantial, 1-3 are modest, < 1 are negligible.

## Parameter Interpretation Guide

### Student-t Degrees of Freedom (nu)

| nu_posterior | Interpretation | Action |
|--------------|----------------|--------|
| < 5 | Extremely heavy tails, strong outliers | Investigate cause |
| 5-10 | Heavy tails, mild outliers | Consider robust model |
| 10-30 | Moderate tails, borderline | Check sensitivity |
| > 30 | Effectively normal | Use normal model |
| > 100 | Data strongly prefer normal | Robustness unnecessary |

### Outlier Probability (p_i)

| p_i | Interpretation | Action |
|-----|----------------|--------|
| < 0.1 | Very likely NOT an outlier | Standard analysis |
| 0.1-0.3 | Uncertain | Sensitivity analysis |
| 0.3-0.7 | Ambiguous | Cannot decide - report range |
| 0.7-0.9 | Likely outlier | Investigate, consider removal |
| > 0.9 | Very likely outlier | Investigate, validate |

### Outlier Inflation Factor (k)

| k_posterior | Interpretation | Action |
|-------------|----------------|--------|
| < 1.5 | Minimal inflation | Robustness not needed |
| 1.5-3 | Modest inflation | Borderline - check sensitivity |
| 3-10 | Substantial inflation | True outlier effect |
| > 10 | Extreme inflation | Data error or fundamentally different |

### Mixture Weights (pi for latent classes)

| pi_posterior | Interpretation | Action |
|--------------|----------------|--------|
| < 0.1 or > 0.9 | One dominant group | Essentially single group |
| 0.1-0.3 or 0.7-0.9 | Unbalanced groups | Possible, investigate |
| 0.3-0.7 | Balanced groups | Check if interpretable |
| Uniform [0,1] | Cannot identify | Mixture model inappropriate |

## Prior Sensitivity Thresholds

### Sensitivity Metric
```
Relative_Sensitivity = (max - min of posterior means) / posterior_SD
```

| Relative_Sensitivity | Interpretation | Action |
|----------------------|----------------|--------|
| < 0.3 | Very low - negligible prior influence | Any reasonable prior OK |
| 0.3-0.5 | Low - data dominate | Use weakly informative prior |
| 0.5-1.0 | Moderate - prior matters | Report sensitivity, choose carefully |
| 1.0-2.0 | High - prior competitive with data | Use informative prior or more data |
| > 2.0 | Very high - prior dominates | Need more data or accept uncertainty |

### Expected for Eight Schools

| Parameter | Expected Rel_Sens | Reasoning |
|-----------|-------------------|-----------|
| mu | 0.2-0.4 | Data are informative (pooled SE = 4.07) |
| tau | 0.4-0.8 | At boundary (0), priors may matter |
| theta_1 | 0.3-0.6 | Strong shrinkage, but extreme observation |
| theta_5 | 0.2-0.4 | Already near pooled mean, less sensitive |

## Computational Diagnostics

### What to Monitor

| Diagnostic | Threshold | Interpretation | Fix |
|------------|-----------|----------------|-----|
| R-hat | < 1.01 | Convergence check | More iterations |
| ESS bulk | > 400 | Posterior exploration | Longer chains |
| ESS tail | > 400 | Tail behavior | Better parameterization |
| Divergences | < 1% | Geometry issues | Higher target_accept |
| Tree depth | < max | Complexity issues | Reparameterize |
| BFMI | > 0.3 | Energy issues | Non-centered param |

### Expected Issues

| Model | Likely Issue | Reason | Solution |
|-------|--------------|--------|----------|
| Student-t | Low ESS for nu | Boundary at 1, weak identification | More iterations, informative prior |
| Outlier (2A) | Label switching | Discrete variables | Marginalization |
| Latent Class (2B) | Non-convergence | Multimodality, small n | Ordered constraints |
| DP (2C) | Very slow | Stick-breaking approximation | Variational inference |
| Strong prior on tau | Divergences | Prior-data conflict | Weaker prior |

## Reporting Template

### If Robust Models Converge to Normal

> "We conducted robustness checks using Student-t distributions (Model 1A) and outlier detection models (Model 2A). The posterior degrees of freedom for the Student-t model was nu = 28 [15, 45], indicating data are consistent with normal distributions. The outlier model assigned low probabilities (p_i < 0.15) to all schools being outliers. LOO-CV scores differed by less than 2 across models, providing no evidence against normal assumptions. We conclude that standard normal hierarchical models are appropriate for these data."

### If Robust Models Differ from Normal

> "Robustness checks revealed sensitivity to distributional assumptions. The Student-t model yielded posterior degrees of freedom nu = 6 [3, 12], suggesting heavier tails than normal. This model assigned 30% lower weight to School 1 (y=28) and resulted in a pooled effect estimate of 6.2 [1.8, 10.5] compared to 7.7 [-0.3, 15.7] under normal assumptions. LOO-CV favored the Student-t model by 4.2 points. We recommend the robust model for inference, with sensitivity analyses reported in Supplement."

### If High Prior Sensitivity

> "Prior sensitivity analysis revealed moderate sensitivity to tau prior specification (Relative_Sensitivity = 0.7). Weakly informative priors (Half-Cauchy(0, 5)) yielded tau_posterior = 2.1 [0.3, 6.8], while strongly regularizing priors (Exponential(1)) yielded tau_posterior = 0.8 [0.1, 2.4]. However, conclusions about the pooled effect mu were robust (Relative_Sensitivity = 0.3), with all priors yielding mu ∈ [7.2, 8.1]. We recommend Half-Cauchy(0, 5) as a reasonable default, and provide sensitivity analyses for alternative specifications."

## Integration with Other Designers

### Expected Overlap
- All designers likely propose hierarchical models with varying tau priors
- Student-t may be proposed by multiple designers
- Prior sensitivity should be common across designs

### Unique Contributions (Designer 3)
- Systematic outlier detection framework
- Quantitative sensitivity metrics (Relative_Sensitivity)
- Decision thresholds for model selection
- Critical perspective on whether robustness is needed

### Synthesis Strategy
If multiple designers propose similar models:
1. Compare specific prior choices
2. Pool computational resources
3. Focus detailed analysis on areas of disagreement
4. Use agreement as validation

If designers propose very different models:
1. Conduct careful model comparison (LOO-CV, posterior predictive)
2. Identify which data features drive differences
3. Report range of plausible models
4. Use ensemble/averaging if predictions needed

## References and Justifications

### Student-t Models
- Juárez, M. A., & Steel, M. F. (2010). Model-based clustering of non-Gaussian panel data based on skew-t distributions. Journal of Business & Economic Statistics, 28(1), 52-66.
- Geweke, J. (1993). Bayesian treatment of the independent Student-t linear model. Journal of Applied Econometrics, 8(S1), S19-S40.

### Mixture Models
- Marin, J. M., Mengersen, K., & Robert, C. P. (2005). Bayesian modelling and inference on mixtures of distributions. Handbook of Statistics, 25, 459-507.
- Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). Chapter 22: Mixture models.

### Prior Sensitivity
- Depaoli, S., & Van de Schoot, R. (2017). Improving transparency and replication in Bayesian statistics: The WAMBS-checklist. Psychological Methods, 22(2), 240.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models (comment on article by Browne and Draper). Bayesian Analysis, 1(3), 515-534.

### Outlier Detection
- Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). Section 6.8: Mixture models for outliers.

---

**Last Updated:** 2025-10-28
**Designer:** Designer 3 (Robust Models Specialist)
**Status:** Design Complete - Ready for Implementation
