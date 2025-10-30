# Model Designer #2: Alternative Parameterizations and Robust Approaches

**Focus Area**: Robust hierarchical models, alternative parameterizations, and mixture models

**Date**: 2025-10-30

---

## Overview

This designer proposes three alternative Bayesian model classes that differ from the standard hierarchical logit-normal approach. Each model addresses specific data challenges identified in the EDA:

1. **Robust Hierarchical** - Student-t hyperpriors for outlier groups
2. **Beta-Binomial** - Direct probability scale parameterization
3. **Finite Mixture** - Two-component subpopulation model

---

## Key Data Challenges Addressed

Based on `/workspace/eda/eda_report.md`:

- **Strong overdispersion**: φ = 3.59 (3.6× expected variance)
- **Three outlier groups**: Groups 2, 4, 8 (|z| > 2.5)
- **Group 4 dominance**: 29% of total data (n=810), low rate (4.2%)
- **Wide rate range**: 3.1% to 14.0% (4.5-fold difference)
- **Small sample groups**: Groups 1 (n=47), 10 (n=97)

---

## Files in This Directory

### Documentation
- `proposed_models.md` - **Main document** with detailed model specifications, falsification criteria, and comparison strategy
- `README.md` - This file (overview and quick reference)

### Stan Models
- `model_robust_hierarchical.stan` - Student-t hyperpriors (handles outliers)
- `model_beta_binomial.stan` - Beta-binomial on probability scale (full)
- `model_beta_binomial_marginalized.stan` - Beta-binomial (marginalized for speed)
- `model_mixture.stan` - 2-component finite mixture model

### Analysis Scripts
- `fit_and_compare_models.py` - Main script to fit all models and compare via LOO

### Results (created after running)
- `results/` - Directory containing:
  - Convergence diagnostics
  - LOO comparison tables
  - Posterior summaries
  - Stan fit objects

---

## Quick Start

### Prerequisites

```bash
pip install cmdstanpy arviz pandas numpy matplotlib seaborn
```

### Run All Models

```bash
cd /workspace/experiments/designer_2
python fit_and_compare_models.py
```

This will:
1. Compile all Stan models
2. Fit each model (4 chains, 2000 iterations)
3. Check convergence (R-hat, ESS, divergences)
4. Compute LOO for model comparison
5. Run posterior predictive checks
6. Save results to `results/` directory

---

## Model Summaries

### Model 1: Robust Hierarchical (Student-t)

**Key Idea**: Replace Normal(μ, τ) hyperprior with Student-t(ν, μ, τ) to accommodate outliers

**Parameters**:
- `mu` - Population mean (logit scale)
- `tau` - Between-group scale
- `nu` - Degrees of freedom (2 to 30, controls tail heaviness)
- `theta[j]` - Group-level logit rates

**Advantages**:
- Gentler shrinkage for outlier groups (2, 4, 8)
- Data determines tail behavior via `nu`
- Still pools information across groups

**Disadvantages**:
- Extra parameter `nu` may not be identifiable with J=12
- Computational cost (Student-t slower than Normal)
- Harder to interpret scientifically

**When to use**:
- Pareto k > 0.5 for outlier groups in standard model
- Posterior ν < 20 (indicates heavy tails needed)
- LOO improvement > 4 over standard hierarchical

**File**: `model_robust_hierarchical.stan`

---

### Model 2: Beta-Binomial

**Key Idea**: Work directly on probability scale using Beta(α, β) hyperprior

**Parameters**:
- `alpha`, `beta` - Beta distribution shape parameters
- `theta[j]` - Group-level success probabilities (0 to 1)

**Derived quantities**:
- `mu_pop = α/(α+β)` - Population mean
- `kappa = α+β` - Concentration (inverse dispersion)

**Advantages**:
- No logit transformation needed
- Conjugate structure (analytical insights)
- Natural dispersion parameter
- Interpretable (α, β as pseudo-counts)

**Disadvantages**:
- Less flexible than logit-normal
- Boundary issues if θ near 0 or 1
- Beta distribution may not capture asymmetry

**Two versions**:
1. **Full**: Sample θ[j] explicitly (`model_beta_binomial.stan`)
2. **Marginalized**: Integrate out θ[j] for speed (`model_beta_binomial_marginalized.stan`)

**When to use**:
- Computational speed critical
- Probability scale interpretation preferred
- LOO competitive with logit-normal (within 2)

**Files**:
- `model_beta_binomial.stan`
- `model_beta_binomial_marginalized.stan`

---

### Model 3: Finite Mixture (2-Component)

**Key Idea**: Model two subpopulations of groups (low-rate vs high-rate)

**Parameters**:
- `pi` - Mixture weight (proportion in high-rate component)
- `mu[1]`, `mu[2]` - Component means (ordered: μ₁ < μ₂)
- `tau[1]`, `tau[2]` - Component scales
- `z[j]` - Component assignment for group j

**Advantages**:
- Explicitly models subpopulations
- Less shrinkage within components
- Testable hypothesis (1 vs 2 components)

**Disadvantages**:
- Highly overparameterized for J=12
- Identifiability issues
- Computational challenges (label switching, multimodality)
- May be spurious (overfitting)

**When to use**:
- Clear component separation (μ₁ << μ₂)
- Assignment certainty > 80% for most groups
- LOO improvement > 4 over standard hierarchical
- Scientific hypothesis supports subpopulations

**RED FLAGS** (abandon model if):
- Components collapse (μ₁ ≈ μ₂)
- Uncertain assignments (all ~50%)
- One component nearly empty (π < 0.1 or > 0.9)
- Computational failure (divergences, R-hat > 1.05)

**File**: `model_mixture.stan`

---

## Decision Framework

### LOO Comparison Rules

| ΔLOO | Interpretation | Action |
|------|----------------|--------|
| < 2×SE | Models equivalent | Choose simplest |
| 2-4 | Weak preference | Consider complexity trade-off |
| > 4 | Strong preference | Choose better model |

### Pareto k Diagnostics

| k value | Interpretation | Action |
|---------|----------------|--------|
| < 0.5 | Good | No issues |
| 0.5-0.7 | Moderate | Check predictive checks |
| > 0.7 | Bad | Model misspecification |

### Convergence Criteria

**Must have ALL**:
- R-hat < 1.05 for all parameters
- ESS_bulk > 400 for key parameters
- ESS_tail > 400 for key parameters
- Divergent transitions = 0 (or very few)
- Tree depth warnings = 0

---

## Expected Outcomes

### Most Likely Result
- **Robust hierarchical** improves over standard (better LOO, lower Pareto k)
- **Beta-binomial** competitive but not clearly better
- **Mixture** fails due to small J (components collapse or unclear assignments)

### Model Rankings (Predicted)
1. Robust hierarchical (best for handling outliers)
2. Standard hierarchical (solid baseline)
3. Beta-binomial (fast alternative)
4. Mixture (too complex for J=12)

### Red Flags That Would Change Everything

1. **All models have high Pareto k** (> 0.7 for multiple groups)
   - Indicates fundamental misspecification
   - Consider non-hierarchical approaches

2. **Group 4 causes 80%+ shift in μ**
   - One group shouldn't dominate population estimate
   - May need to downweight or fit separately

3. **Computational failure across all models**
   - Model class is wrong
   - Consider simpler approaches (pooled + overdispersion)

4. **Prior-posterior conflict everywhere**
   - Data incompatible with hierarchical structure
   - Check for covariates, batch effects, temporal structure

---

## Key Falsification Criteria

### Robust Hierarchical
**Abandon if**:
- ν posterior concentrates near 30 (Normal sufficient)
- No LOO improvement over standard
- Pareto k still > 0.7 for outliers

**Success if**:
- ν < 15 (heavy tails needed)
- ΔLOO > 4
- All Pareto k < 0.5

### Beta-Binomial
**Abandon if**:
- Boundary issues (θ at 0 or 1)
- LOO worse by > 4
- Can't capture dispersion (φ << 3.59)

**Success if**:
- LOO within 2 of logit-normal
- Faster computation (> 2× speed)
- Clear parameter interpretation

### Mixture
**Abandon if**:
- Components collapse (μ₁ ≈ μ₂)
- Assignments uncertain (λ ≈ 0.5 everywhere)
- No LOO improvement
- Computational failure

**Success if**:
- Clear separation (|μ₂ - μ₁| > 1)
- At least 3 groups per component with > 80% probability
- ΔLOO > 4
- Clean convergence (R-hat < 1.01)

---

## Posterior Predictive Checks

### Quantitative Metrics
1. **Dispersion ratio**: φ ≈ 3.59 (should be in posterior interval)
2. **Coverage**: 95% CI covers 11-12 groups (not just 95%)
3. **Extreme groups**: P(r_rep ≥ r | data) for Groups 2, 4, 8
4. **Variance**: Posterior SD of rates ≈ 3.39% (observed)

### Visual Checks
1. **Overlay plot**: Observed vs predicted counts
2. **Residual plot**: Standardized residuals vs sample size
3. **Shrinkage plot**: MLE vs posterior mean
4. **QQ plot**: Empirical vs predicted quantiles

---

## Sensitivity Analyses

### Group 4 Influence
- Fit each model **with and without Group 4**
- Compare μ estimates
- If large shift (> 20%), Group 4 is overly influential

### Prior Sensitivity
- Vary prior SDs by factor of 2
- If posteriors shift substantially, priors too informative
- Re-evaluate prior choices

### Small Sample Sensitivity
- Check shrinkage for Groups 1, 10 (smallest samples)
- Should shrink more than large-sample groups
- If not, pooling mechanism broken

---

## Next Steps After Fitting

### If Models Converge Well
1. Compare LOO values (with SE)
2. Check Pareto k diagnostics
3. Run posterior predictive checks
4. Choose best model (or ensemble)
5. Run sensitivity analyses
6. Validate with simulation-based calibration

### If Models Fail to Converge
1. Check diagnostics (R-hat, ESS, divergences)
2. Try higher adapt_delta (0.99)
3. Try non-centered parameterization
4. Simplify model (reduce parameters)
5. If still failing, abandon model class

### If LOO Differences Are Small (< 2)
1. Choose simplest model (Occam's razor)
2. Document that alternatives were explored
3. Use model averaging if appropriate
4. Report all models in supplementary materials

### If All Models Fail Red Flags
1. Re-examine EDA for missed patterns
2. Check for data quality issues
3. Consider non-hierarchical approaches
4. Consult with domain experts
5. Document lessons learned

---

## References

**EDA Report**: `/workspace/eda/eda_report.md`

**Key Findings**:
- Overdispersion: φ = 3.59, ICC = 0.56
- Outliers: Groups 2, 4, 8
- Sample sizes: 47 to 810 (16-fold range)
- Success rates: 3.1% to 14.0% (4.5-fold range)

**Main Design Document**: `proposed_models.md`

**Stan User's Guide**:
- Hierarchical models: Chapter 9
- Student-t distribution: Chapter 12.3
- Mixture models: Chapter 12
- Beta-binomial: Chapter 10.5

---

## Contact

**Designer**: Model Designer #2
**Focus**: Alternative parameterizations and robust approaches
**Date**: 2025-10-30
**Status**: Awaiting implementation and comparison

---

## Summary

This design explores three alternatives to standard hierarchical logit-normal:

1. **Robust** - Better outlier handling
2. **Beta-binomial** - Computational efficiency
3. **Mixture** - Explicit subpopulations (high risk)

**Expected winner**: Robust hierarchical, with beta-binomial as fast alternative

**Most likely to fail**: Mixture (too few groups for reliable component separation)

**Next phase**: Fit all models, compare LOO, run diagnostics, choose best or pivot
