# Quick Reference: Model Comparison Table

## Three Robust Models for Outlier-Heavy Binomial Data

| Aspect | Student-t Hierarchical | Horseshoe Prior | Mixture Model |
|--------|------------------------|-----------------|---------------|
| **Core Idea** | Heavy-tailed continuous variation | Sparse effects (most groups identical) | Discrete latent subgroups |
| **Random Effects** | α_i ~ Student-t(ν, 0, σ) | α_i ~ Normal(0, τ·λ_i) | α_i ~ Normal(0, σ_z[i]) where z_i ∈ {1,2} |
| **Key Parameters** | ν (degrees of freedom) | λ_i (local shrinkage), τ (global) | π (mixing), μ_1, μ_2 (cluster means) |
| **Complexity** | Medium (1 extra param: ν) | Medium-High (N local scales) | High (marginalizing clusters) |
| **Runtime** | ~2-5 min | ~5-10 min | ~10-20 min |
| | | | |
| **Outlier Handling** | | | |
| Group 8 (z=3.94) | Less shrinkage than Normal | λ_8 large, minimal shrinkage | Assigned to high-rate cluster |
| Group 1 (0/47) | Moderate shrinkage | May shrink heavily to zero | Likely in low-rate cluster |
| Population mean (μ) | Robust to contamination | Robust via sparse selection | Separate means per cluster |
| | | | |
| **When to Use** | | | |
| Best scenario | Outliers are plausible tail events | Most groups similar, few different | Discrete clusters suspected |
| Avoid if | Data is actually Normal (ν→∞) | No sparsity (all λ similar) | No mixture (π→0 or 1) |
| | | | |
| **Falsification Criteria** | | | |
| Abandon if... | ν > 50 (unnecessary complexity) | All λ_i ≈ 0.5 (no sparsity) | π → 0 or 1 (no mixture) |
| | ν < 2 (need mixture) | τ >> τ_0 (too many effects) | μ_1 ≈ μ_2 (no separation) |
| | Posterior predictive fails | Computational issues | Label switching |
| | | | |
| **Success Signals** | | | |
| Good fit | ν ∈ [5, 30] | 3-5 groups with λ > 0.5 | π ∈ [0.2, 0.8] |
| | Group 8 wider posterior | Clear λ bimodality | μ_2 - μ_1 > 0.5 |
| | μ more precise | Better LOO-CV | Clear cluster assignments |
| | | | |
| **Interpretation** | | | |
| Philosophy | Continuous spectrum | Sparse signal detection | Discrete taxonomy |
| Group effects | All groups on continuum | Most ≈0, few ≠0 | Groups belong to clusters |
| Outliers | Extreme but plausible | Truly different | Different population |
| | | | |
| **Advantages** | | | |
| 1. | Robust population inference | Automatic outlier selection | Identifies subpopulations |
| 2. | Well-studied, stable | Better prediction via sparsity | Explicit cluster structure |
| 3. | Simple interpretation | Shrinks non-outliers aggressively | Can estimate cluster sizes |
| | | | |
| **Disadvantages** | | | |
| 1. | One-size-fits-all tails | Harder to interpret λ_i | N=12 is small for clustering |
| 2. | Can't identify which outliers | Slower sampling (Cauchy) | Label switching issues |
| 3. | May still overfit | Assumes sparsity a priori | Computational complexity |
| | | | |
| **Recommended For** | | | |
| Primary analysis | ✓ Yes (most robust) | Sensitivity analysis | Exploratory only |
| Domain experts | ✓ Easy to explain | Moderate difficulty | Hard to explain (latent z) |
| Publications | ✓ Standard approach | Novel, requires justification | Requires strong justification |
| | | | |
| **Model Selection Advice** | | | |
| If LOO-CV wins | Use it | Check sparsity real | Check π and separation |
| If equivalent | Prefer this (simplest) | Only if sparsity clear | Avoid (most complex) |
| If worse | Still useful baseline | Abandon | Abandon |

## Decision Tree

```
START: Read EDA findings (5 of 12 outliers, φ ≈ 3.5)
  |
  +--> Q1: Do you believe outliers are measurement errors?
  |      YES --> Investigate data quality first
  |      NO  --> Continue to modeling
  |
  +--> Q2: Do you need robust population inference?
  |      YES --> Fit Student-t hierarchical (PRIMARY)
  |      NO  --> Use standard Normal hierarchical
  |
  +--> Q3: Do you believe most groups are identical?
  |      YES --> Fit Horseshoe (TEST SPARSITY)
  |      NO  --> Skip to Q4
  |
  +--> Q4: Do you have theoretical reason for discrete clusters?
  |      YES --> Fit Mixture (EXPLORATORY)
  |      NO  --> Stop after Student-t and Horseshoe
  |
  +--> Q5: Compare models via LOO-CV
  |      ΔLOO < 2  --> Models equivalent, choose simplest (Student-t)
  |      ΔLOO > 10 --> Clear winner, use best model
  |
  +--> Q6: Check falsification criteria
  |      Student-t: ν ∈ [5, 30]? ✓ Good
  |      Horseshoe: Sparsity detected? ✓ Good
  |      Mixture: π ∈ [0.2, 0.8] AND sep > 0.5? ✓ Good
  |
  +--> Q7: Posterior predictive checks
  |      All models fail? --> Reconsider model class (negative binomial?)
  |      At least one passes? --> Use that model
  |
END: Report best model with limitations
```

## Expected Posterior Values (Our Dataset)

Based on EDA findings (Groups 1-12, mean rate 7.6%, Group 8 at 14.4%):

| Parameter | Student-t | Horseshoe | Mixture |
|-----------|-----------|-----------|---------|
| **Population** | | | |
| μ (logit) | -2.6 ± 0.3 | -2.7 ± 0.2 | μ_1 ≈ -2.8, μ_2 ≈ -1.9 |
| μ (prob) | 0.069 | 0.063 | Cluster 1: 0.058, Cluster 2: 0.130 |
| σ | 0.9 ± 0.3 | τ ≈ 0.15 | σ_1 ≈ 0.2, σ_2 ≈ 0.3 |
| **Special** | | | |
| ν | 8-15 (moderate tails) | - | - |
| n_active | - | 3-5 groups | - |
| π | - | - | [0.7, 0.3] (70% normal, 30% outlier) |
| **Group 8** | | | |
| α_8 | +1.2 to +1.8 | +1.5 to +2.0 | z_8 = 2 (outlier cluster) |
| p_8 | 0.12-0.15 | 0.13-0.15 | 0.13-0.15 |
| Shrinkage | Moderate (60%) | Low (30%) | Cluster-level only |
| **Group 1** | | | |
| α_1 | -1.0 to -1.5 | -0.5 to -1.0 | z_1 = 1 (normal cluster) |
| p_1 | 0.01-0.03 | 0.01-0.04 | 0.01-0.02 |
| Shrinkage | High (90%) | Very high (95%) | Cluster-level only |

## Computational Requirements

### Minimum System Requirements
- Python 3.8+
- 4 GB RAM (8 GB recommended)
- CmdStan installed
- ~500 MB disk space for samples

### Package Versions (tested)
```
cmdstanpy >= 1.2.0
arviz >= 0.18.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
scipy >= 1.11.0
```

### Installation
```bash
pip install cmdstanpy arviz numpy pandas matplotlib seaborn scipy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## Quick Start

### Fit All Models
```bash
cd /workspace/experiments/designer_3
python fit_robust_models.py --model all
```

### View Results
```bash
# Model comparison
cat results/model_comparison.csv

# Group estimates (Student-t)
cat results/student_t_group_estimates.csv

# Diagnostic plots
ls results/student_t_diagnostics/
```

## Common Issues and Solutions

| Issue | Model | Solution |
|-------|-------|----------|
| Divergences | Student-t | Increase adapt_delta to 0.99 |
| | Horseshoe | Use regularized horseshoe |
| | Mixture | Increase warmup to 3000 |
| Rhat > 1.01 | Any | Increase iterations |
| | Mixture | Check for label switching |
| Low ESS (< 100) | Horseshoe | λ_i often slow, increase iterations |
| | Mixture | Marginalizing expensive, expected |
| ν at boundary (=1) | Student-t | Switch to mixture model |
| ν → ∞ (>100) | Student-t | Use Normal hierarchical |
| All λ similar | Horseshoe | No sparsity, use Student-t |
| π → 0 or 1 | Mixture | No mixture, use Student-t |
| Label switching | Mixture | Check ordered constraint, post-process |

## Interpretation Guide

### Student-t: Degrees of Freedom (ν)

- **ν = 3**: Very heavy tails (like Cauchy)
- **ν = 5-10**: Moderate heavy tails (typical for outlier data)
- **ν = 10-30**: Mild heavy tails (slight robustness)
- **ν > 50**: Nearly Normal (robustness unnecessary)
- **ν → ∞**: Equivalent to Normal distribution

### Horseshoe: Local Shrinkage (λ_i)

- **λ < 0.1**: Very heavy shrinkage (α_i → 0)
- **λ ≈ 0.2-0.5**: Moderate shrinkage
- **λ > 0.5**: "Active" group (minimal shrinkage)
- **λ > 1.0**: Strong signal (clear outlier)

**Effective shrinkage:** κ_i = λ_i² / (λ_i² + τ²)
- κ ≈ 0: Full shrinkage to zero
- κ ≈ 1: No shrinkage

### Mixture: Cluster Probabilities

- **P(z_i = k) > 0.8**: Clear assignment to cluster k
- **P(z_i = k) ≈ 0.5**: Ambiguous (between clusters)
- **P(z_i = k) < 0.2**: Clearly NOT in cluster k

**Mixing proportion (π):**
- π ≈ 0.1-0.9: Meaningful mixture
- π < 0.1 or > 0.9: Likely no mixture

## References

### Papers
- Gelman (2006): Prior distributions for variance parameters
- Carvalho et al. (2010): The horseshoe estimator
- Piironen & Vehtari (2017): Regularized horseshoe
- Vehtari et al. (2017): LOO-CV for Bayesian models

### Books
- Gelman et al. (2013): Bayesian Data Analysis (3rd ed.)
- McElreath (2020): Statistical Rethinking
- Kruschke (2014): Doing Bayesian Data Analysis

### Software Documentation
- Stan User's Guide: https://mc-stan.org/docs/
- ArviZ: https://arviz-devs.github.io/
- CmdStanPy: https://cmdstanpy.readthedocs.io/

---

**Created:** 2025-10-30
**Designer:** Model Designer 3 (Robust Models Specialist)
**Dataset:** 12 groups, 5 outliers, φ ≈ 3.5-5.1
