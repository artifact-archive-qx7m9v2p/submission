# Robust and Alternative Bayesian Models (Designer 3)

## Overview

This directory contains **three robust Bayesian models** designed to handle outlier-heavy binomial data. The dataset has 5 of 12 groups identified as outliers, with severe overdispersion (φ ≈ 3.5-5.1).

## Models Implemented

### 1. Student-t Hierarchical Model
**File:** `student_t_hierarchical.stan`

- **Key feature:** Heavy-tailed group effects (Student-t instead of Normal)
- **Robustness:** Population mean not contaminated by outliers
- **Falsification:** If posterior ν > 50, data doesn't need heavy tails
- **Use when:** Outliers are plausible extreme values

### 2. Horseshoe Prior Model
**File:** `horseshoe_hierarchical.stan`

- **Key feature:** Sparse effects with local-global shrinkage
- **Robustness:** Most groups shrink heavily, outliers shrink minimally
- **Falsification:** If all λ_i similar (no sparsity), use continuous model
- **Use when:** Most groups similar, few truly different

### 3. Mixture Model
**File:** `mixture_hierarchical.stan`

- **Key feature:** Latent subgroups (normal vs outlier clusters)
- **Robustness:** Explicitly models two distinct populations
- **Falsification:** If π → 0 or 1, no mixture needed
- **Use when:** Evidence for discrete clusters

## Files in This Directory

```
experiments/designer_3/
├── README.md                          # This file
├── proposed_models.md                 # Detailed model specifications and rationale
├── student_t_hierarchical.stan        # Student-t model (Stan code)
├── horseshoe_hierarchical.stan        # Horseshoe model (Stan code)
├── mixture_hierarchical.stan          # Mixture model (Stan code)
├── fit_robust_models.py               # Python script to fit all models
└── results/                           # Output directory (created on run)
    ├── model_comparison.csv
    ├── student_t_group_estimates.csv
    ├── horseshoe_group_estimates.csv
    ├── mixture_group_estimates.csv
    └── *_diagnostics/                 # Diagnostic plots per model
```

## Installation

Requires:
- Python 3.8+
- CmdStanPy (for Stan interface)
- ArviZ (for diagnostics)
- Standard scientific stack (numpy, pandas, matplotlib, seaborn, scipy)

```bash
pip install cmdstanpy arviz numpy pandas matplotlib seaborn scipy
```

## Usage

### Fit Individual Models

```bash
# Fit Student-t hierarchical model
python fit_robust_models.py --model student_t

# Fit Horseshoe prior model
python fit_robust_models.py --model horseshoe

# Fit Mixture model
python fit_robust_models.py --model mixture
```

### Fit All Models and Compare

```bash
python fit_robust_models.py --model all
```

This will:
1. Fit all three models
2. Print comprehensive summaries
3. Compute LOO-CV for model comparison
4. Generate diagnostic plots
5. Save results to `results/`

## Interpreting Results

### Student-t Model Output

Key parameter: **ν (degrees of freedom)**

- **ν ∈ [5, 30]**: Moderate heavy tails, Student-t appropriate
- **ν > 50**: Data doesn't need heavy tails, use Normal hierarchical
- **ν < 5**: Very heavy tails, consider mixture model instead

**Group outliers:** Check `is_outlier` (1 if |α| > 2)

### Horseshoe Model Output

Key parameters: **λ (local shrinkage)**, **τ (global shrinkage)**

- **λ_i > 0.5**: Group i is "active" (not shrunk)
- **λ_i < 0.2**: Group i heavily shrunk toward mean
- **n_active ≈ 3-5**: Sparsity detected (expected)
- **n_active ≈ 8-10**: No sparsity, use continuous model

### Mixture Model Output

Key parameters: **π (mixing proportion)**, **μ (cluster means)**

- **π ≈ 0.3-0.7**: Meaningful mixture
- **π → 0 or 1**: No mixture needed
- **μ_2 - μ_1 > 0.5**: Clear cluster separation
- **Cluster assignments:** Check `prob_cluster` for each group

### Model Comparison (LOO-CV)

- **ΔLOO < 2**: Models equivalent (choose simplest)
- **ΔLOO ∈ [2, 10]**: Some evidence for best model
- **ΔLOO > 10**: Strong evidence for best model

**Pareto k diagnostics:**
- **k < 0.5**: Good (LOO reliable)
- **k > 0.7**: Bad (influential point, LOO unreliable)

## Expected Outcomes

### Scenario 1: Student-t Wins
**Interpretation:** Outliers are real but continuous
- No discrete subgroups
- Robustness needed but not sparsity
- Population mean robust to Group 8

### Scenario 2: Horseshoe Wins
**Interpretation:** Sparse effects (most groups identical)
- 8-9 groups pooled (α ≈ 0)
- 3-4 groups genuinely different (Groups 2, 8, 11)
- Better prediction via sparsity

### Scenario 3: Mixture Wins
**Interpretation:** Discrete subpopulations
- Two distinct clusters detected
- Suggests unmeasured binary covariate
- Groups 2, 8, 11 in outlier cluster

### Scenario 4: All Models Fail
**Next steps:**
- Check posterior predictive (can't reproduce patterns)
- High Pareto k (> 0.7) for multiple groups
- Consider negative binomial or data quality issues

## Computational Notes

### Runtime

- **Student-t:** ~2-5 minutes
- **Horseshoe:** ~5-10 minutes (Cauchy priors slower)
- **Mixture:** ~10-20 minutes (marginalizing expensive)

### Sampling Issues

**If divergences occur:**
1. Increase `adapt_delta` (already at 0.95)
2. Check parameter trace plots for funnels
3. Student-t: If ν < 3, may need mixture model
4. Horseshoe: If many divergences, use regularized horseshoe

**If Rhat > 1.01:**
1. Mixture: Check for label switching (trace plots)
2. Increase iterations (already generous)
3. Check for multimodality

**If ESS < 400:**
1. Increase `iter_sampling`
2. Check for high autocorrelation in chains
3. Horseshoe: λ_i often slow mixing (expected)

## Validation Checks

### Posterior Predictive Checks

Model should reproduce:
1. **Overdispersion:** φ ≈ 3.5-5.1
2. **Outlier frequency:** 5/12 groups outside 95% limits
3. **Zero counts:** Group 1 (0/47)
4. **Range of rates:** 0% to 14.4%

### Falsification Criteria

Each model has **built-in rejection criteria** (see `proposed_models.md`):

- Student-t: Posterior ν diagnostic
- Horseshoe: Sparsity check (λ_i distribution)
- Mixture: Mixing proportion and cluster separation

## Stress Tests

### Test 1: Exclude Group 8 (extreme outlier)

Manually remove Group 8 from `data.csv` and refit:

```python
df = pd.read_csv("/workspace/data/data.csv")
df_no_group8 = df[df['group'] != 8]
df_no_group8.to_csv("/workspace/data/data_no_group8.csv", index=False)
```

Then modify `fit_robust_models.py` to use `data_no_group8.csv`.

**Expected:** Robust model's μ should be similar to full data.

### Test 2: Simulate from Fitted Model

Use `r_rep` (posterior predictive samples) to:
1. Generate 100 fake datasets
2. Refit model to each
3. Check parameter recovery (simulation-based calibration)

## References

### Student-t Hierarchical Models
- Gelman (2006): "Prior distributions for variance parameters in hierarchical models"
- Juarez & Steel (2010): "Model-based clustering of non-Gaussian panel data"

### Horseshoe Prior
- Carvalho et al. (2010): "The horseshoe estimator for sparse signals"
- Piironen & Vehtari (2017): "Sparsity information and regularization in the horseshoe"

### Mixture Models
- Richardson & Green (1997): "On Bayesian analysis of mixtures with unknown number of components"
- Frühwirth-Schnatter (2006): "Finite Mixture and Markov Switching Models"

## Contact and Support

This is Model Designer 3 (Robust Models focus).

For issues:
1. Check `proposed_models.md` for detailed specifications
2. Verify Stan installation: `python -c "import cmdstanpy; cmdstanpy.CmdStanModel()"`
3. Check sampling diagnostics in output

---

**Remember:** A good robust model should tell you when robustness is unnecessary (ν → ∞, no sparsity, no mixture). Don't force complexity if data doesn't support it.
