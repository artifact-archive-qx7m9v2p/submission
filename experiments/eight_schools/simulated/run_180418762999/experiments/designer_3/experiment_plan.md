# Experiment Plan: Designer 3 (Adversarial Testing)
## Testing Whether Complete Pooling is Too Simple

**Date**: 2025-10-28
**Designer**: Model Designer 3
**Philosophy**: Adversarial modeling - try to BREAK the EDA's consensus

---

## Executive Summary

The EDA strongly recommends **complete pooling** (all groups share one mean). This plan designs three models specifically to challenge that conclusion.

**If all adversarial models fail → complete pooling is even stronger**
**If any adversarial model succeeds → we found something the EDA missed**

---

## Problem Formulation: Multiple Competing Hypotheses

### Hypothesis H0 (EDA Consensus): Complete Homogeneity
- All groups share the same true mean (mu)
- Reported measurement errors (sigma) are accurate
- Between-group variance = 0
- No latent structure

**Evidence FOR**: Chi-square test p=0.42, tau^2=0, extensive overlap in confidence intervals

**Evidence AGAINST**: Only n=8, high measurement error, untested assumptions

### Hypothesis H1: Measurement Error Misspecification
- Reported sigma values are systematically wrong (underestimated or overestimated)
- True measurement error = sigma * lambda where lambda ≠ 1
- Once corrected, between-group variance might emerge

**Rationale**: Measurement error is "known" but could be nominal rather than actual precision

**Test**: Model 1 (Inflation factor)

### Hypothesis H2: Latent Clustering
- There are K=2 hidden subgroups with different means
- EDA couldn't detect them because it assumed homogeneity
- High-SNR groups vs low-SNR groups might be different populations

**Rationale**:
- Clear SNR divide: Groups 0-3 (SNR>1) vs Groups 4-7 (SNR<1)
- Possible bimodal distribution noted in EDA
- Small sample size makes clustering hard to detect

**Test**: Model 2 (Mixture model)

### Hypothesis H3: Functional Heteroscedasticity
- Measurement error is not independent of true value
- Larger true values have larger measurement errors (or vice versa)
- Standard model with "known" errors is misspecified

**Rationale**:
- EDA found weak positive correlation (r=0.43, p=0.39)
- Underpowered test with n=8
- Correlation test used OBSERVED y, not TRUE theta

**Test**: Model 3 (Functional error model)

---

## Model Classes to Explore

### Model Class 1: Error-Inflated Hierarchical Model

**Mathematical Specification**:
```
y_i ~ Normal(theta_i, sigma_i * lambda)
theta_i ~ Normal(mu, tau)
mu ~ Normal(0, 30)
tau ~ Half-Cauchy(0, 10)
lambda ~ Uniform(0.5, 3.0)  # KEY PARAMETER
```

**Variants**:
- Variant 1A: Multiplicative inflation (above)
- Variant 1B: Additive inflation (sigma_eff = sigma + delta)
- Variant 1C: Group-specific inflation (lambda_i for each group)

**Primary variant**: 1A (simplest, most identifiable)

**Scientific Question**: Are the reported measurement errors accurate?

**I will abandon this if**:
1. Posterior for lambda has 95% CrI ⊆ [0.9, 1.1] (errors are accurate)
2. High correlation between lambda and tau (>0.9) indicates non-identifiability
3. Posterior predictive checks are systematically worse than baseline
4. Stress test shows model cannot recover known lambda values

**Red flags**:
- Divergent transitions >2% despite high adapt_delta
- Lambda posterior equals prior (no learning)
- Effective sample size for lambda < 100

**Escape route**: If lambda is non-identifiable, try fixing it at specific values (0.8, 1.0, 1.5, 2.0) and compare model fit

---

### Model Class 2: Finite Mixture Model

**Mathematical Specification**:
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu[z_i], tau[z_i])
z_i ~ Categorical(pi)
mu ~ Ordered([Normal(0,10), Normal(20,10)])
tau ~ Half-Cauchy(0, 5)  [vector of length K]
pi ~ Dirichlet([1, 1])
```

**Variants**:
- Variant 2A: K=2 clusters (ordered means to prevent label switching)
- Variant 2B: K=3 clusters (if K=2 shows evidence of mixture)
- Variant 2C: Dirichlet Process mixture (K infinite)

**Primary variant**: 2A (K=2, most parsimonious)

**Scientific Question**: Is there hidden subgroup structure?

**I will abandon this if**:
1. Cluster means collapse: |mu[2] - mu[1]| < 5 with 95% CrI including 0
2. Mixing proportions are highly uncertain: SD(pi) > 0.3
3. Cluster assignments are unstable (>20% of groups flip assignments)
4. LOO/WAIC penalizes mixture by >10 units compared to complete pooling
5. Posterior predictive shows NO improvement in capturing variance

**Red flags**:
- Label switching (R-hat > 1.05 for cluster parameters)
- Separation metric < 1.0 (clusters not distinct)
- Pareto k > 0.7 for any observations (overfitting)

**Escape route**: If mixture is too complex, try simpler partition based on observed SNR (high vs low) as a fixed effect

---

### Model Class 3: Heteroscedastic Error Model

**Mathematical Specification**:
```
y_i ~ Normal(theta_i, sigma_i * exp(alpha * theta_i))
theta_i ~ Normal(mu, tau)
mu ~ Normal(10, 20)
tau ~ Half-Cauchy(0, 10)
alpha ~ Normal(0, 0.1)  # KEY PARAMETER
```

**Variants**:
- Variant 3A: Exponential scaling (above)
- Variant 3B: Linear scaling: sigma_eff = sqrt(sigma^2 + (beta*theta)^2)
- Variant 3C: Power law: sigma_eff = sigma * |theta|^gamma

**Primary variant**: 3A (exponential, most stable numerically)

**Scientific Question**: Does measurement error scale with true value?

**I will abandon this if**:
1. Posterior for alpha has 95% CrI ⊆ [-0.05, 0.05] (no functional relationship)
2. WAIC is worse than constant-error model by >2 units
3. Numerical instability (divergences, effective sample size < 100)
4. Posterior predictive generates implausible sigma values (e.g., >100 or <0.1)

**Red flags**:
- Alpha posterior equals prior (non-identifiable)
- Extreme correlation between alpha and tau (>0.9)
- Sigma_effective has huge range (>10x variation)

**Escape route**: If exponential model is unstable, try simpler linear scaling (Variant 3B)

---

## Falsification Criteria Summary

### Decision Tree

```
START: Fit all three adversarial models + baseline

├─ STAGE 1: Test Measurement Assumptions
│  ├─ Model 1: Is lambda ≈ 1?
│  │  ├─ YES (95% CrI ⊆ [0.9, 1.1]) → Measurement errors are accurate
│  │  └─ NO → STOP. Investigate measurement process.
│  │
│  └─ Model 3: Is alpha ≈ 0?
│     ├─ YES (95% CrI ⊆ [-0.05, 0.05]) → Errors independent of true value
│     └─ NO → STOP. Revise error model.
│
├─ STAGE 2: Test Population Structure (if Stage 1 passes)
│  └─ Model 2: Are there clusters?
│     ├─ NO (means collapse OR unstable assignments OR WAIC penalty) → Homogeneous
│     └─ YES (clear separation AND stable assignments AND better WAIC) → Heterogeneous
│
└─ FINAL DECISION
   ├─ All adversarial models falsified → Complete pooling CONFIRMED
   ├─ Measurement model issues found → REVISE measurement assumptions
   └─ Clustering found → REVISE homogeneity assumption
```

### Specific Thresholds

| Model | Parameter | Falsification Criterion | Interpretation |
|-------|-----------|------------------------|----------------|
| Model 1 | lambda | Pr(0.9 < lambda < 1.1) > 0.80 | Errors are accurate |
| Model 1 | Cor(lambda, tau) | r > 0.90 | Non-identifiable |
| Model 2 | Separation | \|mu[2]-mu[1]\|/sqrt(tau^2) < 1.5 | No clusters |
| Model 2 | WAIC diff | WAIC_mixture - WAIC_baseline > 4 | Overfitting |
| Model 3 | alpha | Pr(-0.05 < alpha < 0.05) > 0.80 | No functional error |
| Model 3 | WAIC diff | WAIC_functional - WAIC_baseline > 2 | Worse fit |

---

## Alternative Approaches if Models Fail

### If Model 1 is non-identifiable (lambda ⊗ tau)
**Alternative**: Profile likelihood approach
- Fix lambda at grid of values: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
- For each lambda, fit hierarchical model
- Compare marginal likelihoods
- Identify which lambda maximizes evidence

### If Model 2 has label switching
**Alternative**: Post-hoc relabeling
- At each MCMC iteration, order clusters by mean: mu[1] < mu[2]
- Or use largest cluster (max pi) as "reference"
- Or implement ordered constraint in Stan: `ordered[K] mu`

### If Model 3 has numerical issues
**Alternative**: Discretize functional form
- Partition observations into "low theta" vs "high theta"
- Allow different error scaling for each partition
- Simpler than continuous function

---

## Decision Points for Major Strategy Pivots

### Decision Point 1: After fitting Model 1 and Model 3
**Checkpoint**: Are measurement errors well-specified?

**If NO (lambda ≠ 1 OR alpha ≠ 0)**:
- **PIVOT**: Stop all hierarchical modeling
- **New strategy**: Investigate measurement process
- **Actions**:
  - Contact data source to understand measurement procedures
  - Examine whether sigma values are nominal vs actual
  - Consider re-analysis with corrected errors

**If YES**: Proceed to Model 2

### Decision Point 2: After fitting all models
**Checkpoint**: Does any adversarial model beat complete pooling?

**If NO (all falsified)**:
- **Accept**: Complete pooling is correct
- **Report**: EDA conclusion is strongly supported
- **Communicate**: Even adversarial models confirm homogeneity

**If YES (at least one succeeds)**:
- **Reject**: Complete pooling is too simple
- **Investigate**: Why did EDA miss this?
- **Report**: Model that beat baseline

### Decision Point 3: If multiple models succeed
**Checkpoint**: Which model is most scientifically plausible?

**Criteria**:
1. Predictive accuracy (LOO/WAIC)
2. Posterior predictive checks
3. Scientific interpretability
4. Parsimony

**Priority order**:
1. Model 1 or 3 (measurement issues) → most fundamental
2. Model 2 (clustering) → substantive scientific finding

---

## Warning Signs to Document

### Prior-Posterior Conflict
**Sign**: Prior and posterior are very different (KL divergence > 10)

**Possible causes**:
- Prior is misspecified (too narrow)
- Data are highly informative (good!)
- Model is fighting the data (bad!)

**Action**:
- Check prior predictive: Does it allow observed data?
- If not, revise prior
- If yes, this is legitimate learning

### Computational Difficulties

**Sign**: Divergent transitions, low ESS, high R-hat

**Possible causes**:
- Pathological posterior geometry (funnels, multi-modality)
- Non-identified parameters
- Misspecified model

**Actions**:
1. Increase adapt_delta to 0.95, 0.99
2. Try non-centered parameterization
3. Check pairs plots for correlations
4. If persistent, model is likely misspecified → abandon

### Extreme Parameter Values

**Sign**: lambda > 2.5, alpha > 0.3, tau > 20

**Interpretation**: Model is compensating for something

**Actions**:
- Check if these values are scientifically plausible
- Examine posterior predictive: Does model generate realistic data?
- Consider whether model is overfitting

### Poor Predictive Performance Despite Good Fit

**Sign**: Low in-sample error, but poor LOO (high Pareto k)

**Interpretation**: Overfitting

**Action**: Penalize model complexity or use simpler model

### Inconsistent Results Across Data Subsets

**Sign**: Removing one observation drastically changes inference

**Interpretation**: Inference is unstable, model is fragile

**Action**:
- Identify influential observations
- Examine why they're influential
- Consider robust alternatives (t-distribution)

---

## Stress Tests to Break Models

### Test 1: Null Data (Complete Pooling is True)
**Procedure**:
1. Generate synthetic data from complete pooling: y ~ N(10, sigma)
2. Fit all adversarial models
3. Check Type I error: Do they spuriously detect complexity?

**Success criterion**: Models should NOT find lambda ≠ 1, alpha ≠ 0, or clusters

**If test fails**: Model is too flexible, prone to overfitting → increase regularization

### Test 2: Known Departure (e.g., lambda = 1.5)
**Procedure**:
1. Generate synthetic data with known lambda = 1.5
2. Fit Model 1
3. Check: Does posterior recover lambda ≈ 1.5?

**Success criterion**: 95% CrI should cover 1.5

**If test fails**: Model is biased or non-identifiable → fix model

### Test 3: Extreme Observation
**Procedure**:
1. Fit models to real data
2. Generate replicated data from posterior predictive
3. Add extreme observation (e.g., y = 50 or y = -30)
4. Refit models

**Success criterion**: Inference should be robust (or model should flag as outlier)

**If test fails**: Model is too sensitive to outliers → consider robust alternatives

---

## Implementation Schedule

### Week 1: Model Development and Fitting
**Days 1-2**:
- Implement all Stan models
- Run prior predictive checks
- Ensure models compile without errors

**Days 3-4**:
- Fit baseline (complete pooling)
- Fit Model 1 (Inflation)
- Check convergence diagnostics

**Days 5-7**:
- Fit Model 2 (Mixture)
- Fit Model 3 (Functional error)
- Document any computational issues

### Week 2: Diagnostics and Comparison
**Days 1-2**:
- Posterior predictive checks for all models
- LOO/WAIC computation
- Falsification decisions

**Days 3-4**:
- Model comparison
- Identify best model(s)
- Sensitivity analyses (prior robustness)

**Day 5**:
- Decision Point 2: Does any adversarial model succeed?
- Document findings

### Week 3: Stress Tests and Validation
**Days 1-2**:
- Stress Test 1 (null data)
- Stress Test 2 (known parameters)

**Days 3-4**:
- Stress Test 3 (extreme observations)
- Robustness checks

**Day 5**:
- Final report
- Recommendations

---

## Expected Outcomes by Scenario

### Scenario A: EDA is Correct (Most Likely)
**Evidence**:
- lambda ≈ 1.0 (errors are accurate)
- alpha ≈ 0 (errors independent of theta)
- No clear clusters (mixture collapses)
- All adversarial models have worse WAIC than baseline

**Outcome**: Complete pooling is confirmed

**Report**:
"Adversarial modeling approach tested three alternative hypotheses (measurement error misspecification, latent clustering, functional heteroscedasticity). All adversarial models were falsified, providing STRONG support for the EDA conclusion that complete pooling is appropriate. The reported measurement errors are accurate, groups are homogeneous, and there is no evidence of latent structure."

### Scenario B: Measurement Errors are Wrong
**Evidence**:
- lambda ≈ 1.8 (95% CrI: [1.5, 2.2])
- Once errors are inflated, tau becomes non-negligible (tau ≈ 8)

**Outcome**: STOP. Revise measurement model.

**Report**:
"Model 1 found evidence that reported measurement errors are UNDERESTIMATED by approximately 80%. When corrected, between-group variance emerges (tau ≈ 8), contradicting the EDA conclusion. RECOMMENDATION: Investigate measurement procedures before proceeding with inference. Current data cannot be trusted without understanding error structure."

### Scenario C: Hidden Clusters Exist
**Evidence**:
- Clear cluster separation: mu[1] ≈ 6, mu[2] ≈ 22
- Stable assignments: Groups 0-3 → Cluster 2, Groups 4-7 → Cluster 1
- WAIC prefers mixture by 8 units

**Outcome**: Reject homogeneity, accept clustering

**Report**:
"Model 2 found evidence for K=2 latent clusters with distinct means (6 vs 22). Cluster membership aligns with SNR divide (high-SNR groups vs low-SNR groups), suggesting different measurement regimes or true processes. EDA assumption of homogeneity was incorrect. RECOMMENDATION: Investigate why clusters exist and model them explicitly."

### Scenario D: Functional Heteroscedasticity
**Evidence**:
- alpha ≈ 0.12 (95% CrI: [0.06, 0.18])
- Measurement error increases with true value
- Better predictive accuracy than constant-error model

**Outcome**: Revise error model

**Report**:
"Model 3 found evidence that measurement error SCALES with true value (alpha ≈ 0.12). Larger theta have larger measurement errors, violating the 'known heteroscedastic error' assumption. RECOMMENDATION: Use generalized error model that accounts for proportional error component."

---

## Computational Considerations

### MCMC Settings (Conservative)
```
Baseline: 4 chains, 1000 warmup, 2000 sampling
Model 1:  4 chains, 2000 warmup, 2000 sampling, adapt_delta=0.95
Model 2:  8 chains, 3000 warmup, 2000 sampling, adapt_delta=0.90
Model 3:  4 chains, 1500 warmup, 2000 sampling, adapt_delta=0.95
```

### Expected Runtime (on modern CPU)
- Baseline: ~30 seconds
- Model 1: ~2 minutes
- Model 2: ~5 minutes (8 chains, label switching)
- Model 3: ~2 minutes

Total: ~10 minutes for all models

### Disk Space Requirements
- Stan models: ~50 KB each
- MCMC samples: ~10 MB each
- Diagnostics/plots: ~5 MB
Total: ~50 MB

### Software Dependencies
- Python 3.8+
- CmdStanPy or PyStan
- ArviZ (for diagnostics)
- NumPy, Pandas, Matplotlib, Seaborn

---

## Deliverables

### Code
1. `model1_inflation.stan` - Inflation factor model
2. `model2_mixture.stan` - Mixture model
3. `model3_functional.stan` - Functional error model
4. `baseline_complete_pooling.stan` - Baseline
5. `fit_all_models.py` - Python script to fit all models
6. `model_comparison.py` - LOO/WAIC comparison
7. `posterior_checks.py` - Posterior predictive checks
8. `stress_tests.py` - Validation via synthetic data

### Reports
1. `convergence_diagnostics.md` - R-hat, ESS, divergences for all models
2. `model_comparison_table.md` - LOO, WAIC, parameter estimates
3. `falsification_decisions.md` - Which models were abandoned and why
4. `final_recommendation.md` - Best model for this data

### Visualizations
1. `trace_plots_*.png` - MCMC diagnostics
2. `posterior_distributions_*.png` - Parameter posteriors
3. `ppc_*.png` - Posterior predictive checks
4. `model_comparison.png` - WAIC comparison plot
5. `pairs_*.png` - Pairwise parameter correlations

---

## Success Metrics

**This experiment plan succeeds if**:

1. **Scientific Learning**: We learn which hypotheses are supported by data
2. **Computational Success**: All models converge (R-hat < 1.01, ESS > 400)
3. **Clear Decision**: We can definitively falsify or support each model
4. **Robustness**: Stress tests validate our conclusions
5. **Actionable Outcome**: Clear recommendation for best model

**This experiment plan fails if**:

1. **Computational Failure**: Models don't converge despite tuning
2. **Non-identifiability**: Posteriors equal priors
3. **Contradictory Results**: Models give inconsistent conclusions
4. **No Falsification**: Cannot distinguish any models
5. **Overfitting**: All complex models show signs of overfitting

---

## Final Philosophy: Embracing Failure

**Remember**:

- If all adversarial models are falsified → **This is SUCCESS**
  - We tried hard to break complete pooling and failed
  - EDA conclusion is now much stronger

- If an adversarial model succeeds → **This is also SUCCESS**
  - We discovered something the EDA missed
  - Science advances by finding the truth

- If we can't compute reliable inference → **This teaches us**
  - The data are insufficient for these questions
  - Need more data or simpler questions

**The only true failure is not trying to break our assumptions.**

---

**End of Experiment Plan - Designer 3**

**Next Steps**: Implement Stan models and begin fitting. Report back with convergence diagnostics and model comparison results.
