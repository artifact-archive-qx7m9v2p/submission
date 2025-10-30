# Designer #2: Quick Reference Summary

**Date**: 2025-10-30
**Focus**: Alternative Parameterizations & Robust Approaches

---

## Three Proposed Models

### 1. Robust Hierarchical (Student-t Hyperpriors)
```
θ_i ~ Student-t(ν, μ, τ)  # ν ∈ [2, 30]
r_i ~ Binomial(n_i, logit⁻¹(θ_i))
```
- **Purpose**: Handle outliers (Groups 2, 4, 8) without over-shrinking
- **Key parameter**: ν (degrees of freedom) - data determines tail heaviness
- **Success**: ν < 15, ΔLOO > 4, all Pareto k < 0.5
- **Failure**: ν > 25 (Normal sufficient), no LOO gain

### 2. Beta-Binomial (Direct Probability Scale)
```
θ_i ~ Beta(α, β)
r_i ~ Binomial(n_i, θ_i)  # No transformation!
```
- **Purpose**: Computational efficiency, direct interpretation
- **Key parameter**: κ = α + β (concentration, inverse dispersion)
- **Success**: LOO within 2 of logit-normal, fast computation
- **Failure**: Boundary issues, LOO worse by > 4

### 3. Finite Mixture (2-Component)
```
z_i ~ Categorical(π)
θ_i | z_i=k ~ Normal(μ_k, τ_k)  # k ∈ {1, 2}
r_i ~ Binomial(n_i, logit⁻¹(θ_i))
```
- **Purpose**: Explicitly model low-rate vs high-rate subpopulations
- **Key parameter**: π (mixture weight), μ₁ vs μ₂ (component means)
- **Success**: Clear separation, assignments > 80% certain, ΔLOO > 4
- **Failure**: Components collapse, uncertain assignments, computational issues

---

## Why These Models?

### Data Challenges from EDA
- **Overdispersion**: φ = 3.59 (3.6× expected)
- **Outliers**: Groups 2 (12.8%), 4 (4.2%), 8 (14.0%)
- **Dominance**: Group 4 = 29% of data (n=810)
- **Range**: 4.5-fold difference in success rates (3.1% to 14.0%)

### Standard Hierarchical Limitations
- **Normal hyperprior**: Shrinks outliers too aggressively
- **Logit transformation**: Can be numerically unstable
- **Single hierarchy**: Assumes one homogeneous population

---

## Critical Decision Points

### When to Stop and Reconsider Everything

1. **All models have Pareto k > 0.7** for multiple groups
   - Signal: Fundamental misspecification
   - Action: Abandon hierarchical structure entirely

2. **Group 4 causes 80%+ shift in μ**
   - Signal: One group dominating population estimate
   - Action: Downweight or fit separately

3. **Computational failure across all approaches**
   - Signal: Model class is wrong
   - Action: Simplify to pooled + overdispersion

4. **Prior-posterior conflict everywhere**
   - Signal: Data incompatible with hierarchical structure
   - Action: Check for covariates, non-exchangeability

---

## Model Selection Strategy

### Step 1: Convergence Check
- R-hat < 1.05 for all parameters
- ESS_bulk > 400
- Divergent transitions = 0 (or very few)

### Step 2: LOO Comparison
| ΔLOO | Decision |
|------|----------|
| < 2×SE | Equivalent, choose simplest |
| 2-4 | Weak preference |
| > 4 | Strong preference |

### Step 3: Pareto k Diagnostics
| k | Interpretation |
|---|----------------|
| < 0.5 | Good |
| 0.5-0.7 | Moderate concern |
| > 0.7 | Model misspecification |

### Step 4: Posterior Predictive Checks
- Dispersion φ ≈ 3.59?
- 95% CI coverage ≈ 95%?
- Outliers well-predicted?

---

## Expected Outcome

### Most Likely Ranking
1. **Robust hierarchical** - Best for outliers
2. **Standard hierarchical** - Solid baseline
3. **Beta-binomial** - Fast alternative
4. **Mixture** - Fails (too few groups)

### Alternative Scenario
If Group 4's dominance is problematic:
- Consider downweighting
- Fit with/without Group 4
- Use leave-one-group-out CV

---

## Files Generated

### Documentation
- `proposed_models.md` - Detailed specifications (28 KB)
- `README.md` - Overview and quick start (11 KB)
- `DESIGN_SUMMARY.md` - This file

### Stan Models
- `model_robust_hierarchical.stan`
- `model_beta_binomial.stan`
- `model_beta_binomial_marginalized.stan`
- `model_mixture.stan`

### Analysis
- `fit_and_compare_models.py` - Run all models

---

## Quick Commands

```bash
# Navigate to directory
cd /workspace/experiments/designer_2

# Install dependencies
pip install cmdstanpy arviz pandas numpy matplotlib seaborn

# Fit all models
python fit_and_compare_models.py

# Results will be in:
# results/convergence_diagnostics.csv
# results/loo_comparison.csv
# results/posterior_summary_*.csv
```

---

## Key Falsification Criteria

### Robust Hierarchical
- ✓ Success: ν < 15, ΔLOO > 4, k < 0.5
- ✗ Failure: ν > 25, no LOO gain

### Beta-Binomial
- ✓ Success: LOO ≈ logit-normal, fast
- ✗ Failure: Boundary issues, LOO << logit-normal

### Mixture
- ✓ Success: |μ₂ - μ₁| > 1, assignments clear, ΔLOO > 4
- ✗ Failure: μ₁ ≈ μ₂, assignments ~50%, divergences

---

## What Makes This Design Different?

### Standard Approach (Designer #1, typical)
- Hierarchical logit-normal
- Normal hyperpriors
- Non-centered parameterization
- Focus on efficiency

### This Design (Designer #2)
- **Heavy-tailed hyperpriors** (Student-t)
- **Alternative scales** (probability vs logit)
- **Explicit subpopulations** (mixture)
- Focus on robustness and interpretability

---

## Philosophy

> "The goal is finding truth, not completing tasks."

- **Plan for failure** - Each model has explicit abandonment criteria
- **Think adversarially** - What would break these models?
- **Be ready to pivot** - Switching model classes = learning, not failure
- **Question EDA** - Apparent patterns might be artifacts

---

## Bottom Line

**Most conservative bet**: Robust hierarchical will outperform standard

**Most ambitious bet**: Mixture will reveal subpopulations (or fail spectacularly)

**Most practical bet**: Beta-binomial will be fast and good enough

**Most likely reality**: Robust hierarchical wins, mixture fails, beta-binomial competitive

**Next phase**: Implement, fit, compare, iterate based on evidence
