# Designer 3: Hierarchical/Compositional Models

## Overview

This directory contains **Model Designer 3's** proposals focusing on **hierarchical and compositional decomposition** of the Y-x relationship.

## Philosophy

Rather than focusing solely on functional form, I decompose the data generation process into structured levels:
1. **Population-level trend** (systematic saturation)
2. **Group-level effects** (replicate structure) or **spatial correlation** (smooth deviations)
3. **Observation-level noise** (measurement error)

## Files in This Directory

### Documentation
- `experiment_plan.md` - Executive summary and implementation roadmap (12 KB)
- `proposed_models.md` - Comprehensive model specifications with theory and falsification criteria (36 KB)
- `README.md` - This file

### Stan Model Implementations
- `model_1_additive.stan` - Additive Decomposition (Trend + GP + Noise)
- `model_2_hierarchical.stan` - Hierarchical Replicate Model (Random effects for replicate groups)
- `model_3_variance.stan` - Compositional Variance Model (Location-scale heteroscedasticity)

## Three Proposed Model Classes

### Priority 1: Model 2 - Hierarchical Replicate Model
**Rationale**: 21/27 observations are replicates - this structure should be explicitly modeled

**Structure**:
```
Y_ij ~ Normal(μ_ij, σ_within)
μ_ij = α + β·log(x_j) + u_j
u_j ~ Normal(0, σ_between)
```

**Key Parameter**: ICC = σ²_between / (σ²_between + σ²_within)
- ICC > 0.1 suggests meaningful hierarchy
- ICC ≈ 0 suggests replicates are pure noise

**Falsification**: Abandon if ICC ≈ 0 or σ_between >> σ_within

---

### Priority 2: Model 3 - Compositional Variance Model
**Rationale**: EDA shows 4.6:1 variance ratio (low vs high x), but n=27 may lack power to detect

**Structure**:
```
Y_i ~ Normal(μ_i, σ_i)
μ_i = α + β·log(x_i)
log(σ_i) = γ_0 + γ_1·log(x_i)
```

**Key Parameter**: γ_1 (variance trend)
- γ_1 < 0 indicates decreasing variance with x
- γ_1 = 0 indicates constant variance

**Falsification**: Abandon if γ_1 credible interval includes 0 AND LOO doesn't improve

---

### Priority 3: Model 1 - Additive Decomposition Model
**Rationale**: Separates parametric trend from smooth deviations, useful for gap interpolation

**Structure**:
```
Y_i ~ Normal(μ_i, σ_noise)
μ_i = α + β·log(x_i) + f_GP(x_i)
f_GP ~ GaussianProcess(0, k_SE)
```

**Key Parameter**: η (GP amplitude)
- η << β suggests trend dominates (good)
- η >> β suggests trend misspecified (bad)

**Falsification**: Abandon if η ≈ 0 (no structured deviations)

---

## Quick Start: Fitting the Models

### Prerequisites
```bash
pip install cmdstanpy numpy pandas arviz matplotlib seaborn
```

### Data Preparation for Model 2 (Hierarchical)

Model 2 requires grouping structure. Here's how to prepare the data:

```python
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('/workspace/data.csv')
x = data['x'].values
Y = data['Y'].values

# Create group structure for hierarchical model
df = pd.DataFrame({'x': x, 'Y': Y})
df['group_id'] = pd.factorize(df['x'])[0] + 1  # 1-indexed for Stan

# Group-level information
group_info = df.groupby('group_id').agg({
    'x': 'first',
    'Y': 'count'
}).rename(columns={'Y': 'n_replicates'})

group_info['has_replicates'] = (group_info['n_replicates'] > 1).astype(int)

# Prepare Stan data
stan_data_model2 = {
    'N': len(df),
    'J': len(group_info),
    'Y': df['Y'].values,
    'group_id': df['group_id'].values,
    'x_group': group_info['x'].values,
    'has_replicates': group_info['has_replicates'].values
}
```

### Fitting Models with CmdStanPy

```python
from cmdstanpy import CmdStanModel

# Model 1: Additive Decomposition
model1 = CmdStanModel(stan_file='/workspace/experiments/designer_3/model_1_additive.stan')
fit1 = model1.sample(
    data={'N': len(x), 'x': x, 'Y': Y},
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,  # Higher for GP
    seed=12345
)

# Model 2: Hierarchical Replicate
model2 = CmdStanModel(stan_file='/workspace/experiments/designer_3/model_2_hierarchical.stan')
fit2 = model2.sample(
    data=stan_data_model2,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,  # Higher for hierarchical
    seed=12345
)

# Model 3: Compositional Variance
model3 = CmdStanModel(stan_file='/workspace/experiments/designer_3/model_3_variance.stan')
fit3 = model3.sample(
    data={'N': len(x), 'x': x, 'Y': Y},
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=12345
)
```

### Model Comparison

```python
import arviz as az

# Convert to arviz InferenceData
idata1 = az.from_cmdstanpy(fit1)
idata2 = az.from_cmdstanpy(fit2)
idata3 = az.from_cmdstanpy(fit3)

# Compute LOO-CV
loo1 = az.loo(idata1, pointwise=True)
loo2 = az.loo(idata2, pointwise=True)
loo3 = az.loo(idata3, pointwise=True)

# Compare models
comparison = az.compare({
    'Model 1 (Additive)': idata1,
    'Model 2 (Hierarchical)': idata2,
    'Model 3 (Variance)': idata3
}, ic='loo')

print(comparison)
# Look for dloo > 4 for meaningful differences
```

### Key Diagnostics to Check

```python
# Convergence diagnostics
print("Model 1 Summary:")
print(fit1.diagnose())
print(az.summary(idata1, var_names=['alpha', 'beta', 'eta', 'sigma_noise']))

print("\nModel 2 Summary:")
print(fit2.diagnose())
print(az.summary(idata2, var_names=['alpha', 'beta', 'sigma_between', 'sigma_within', 'ICC']))

print("\nModel 3 Summary:")
print(fit3.diagnose())
print(az.summary(idata3, var_names=['alpha', 'beta', 'gamma_0', 'gamma_1']))

# Posterior predictive checks
az.plot_ppc(idata1, num_pp_samples=100)
az.plot_ppc(idata2, num_pp_samples=100)
az.plot_ppc(idata3, num_pp_samples=100)
```

## Key Falsification Checks

### For Model 2 (Hierarchical):
```python
# Check if hierarchy is needed
icc_samples = idata2.posterior['ICC'].values.flatten()
print(f"ICC: {np.median(icc_samples):.3f} [{np.percentile(icc_samples, 2.5):.3f}, {np.percentile(icc_samples, 97.5):.3f}]")

if np.percentile(icc_samples, 97.5) < 0.05:
    print("FALSIFIED: ICC ≈ 0, hierarchy not needed")
```

### For Model 3 (Variance):
```python
# Check if heteroscedasticity is real
gamma1_samples = idata3.posterior['gamma_1'].values.flatten()
ci_gamma1 = np.percentile(gamma1_samples, [2.5, 97.5])

print(f"γ₁: {np.median(gamma1_samples):.3f} [{ci_gamma1[0]:.3f}, {ci_gamma1[1]:.3f}]")

if ci_gamma1[0] < 0 < ci_gamma1[1]:
    print("WARNING: γ₁ credible interval includes 0, heteroscedasticity not clearly detected")
```

### For Model 1 (Additive):
```python
# Check if GP component is meaningful
eta_samples = idata1.posterior['eta'].values.flatten()
beta_samples = idata1.posterior['beta'].values.flatten()

print(f"GP amplitude (η): {np.median(eta_samples):.3f}")
print(f"Trend slope (β): {np.median(beta_samples):.3f}")
print(f"Ratio η/β: {np.median(eta_samples)/np.median(beta_samples):.3f}")

if np.median(eta_samples) < 0.01:
    print("FALSIFIED: GP component negligible, use parametric model only")
```

## Expected Outcomes

### Most Likely
**Model 2 wins** with small but non-zero ICC (0.05-0.15), indicating replicates have some structure but mostly measurement noise.

### Alternative
**Model 3 preferred** but γ₁ credible interval includes 0 - heteroscedasticity exists but n=27 insufficient to detect decisively.

### Surprising
**Model 1 shows large GP component** - indicates logarithmic trend is misspecified, need to investigate what GP captures.

## Contact

For questions about model specifications or implementation:
- See `proposed_models.md` for full mathematical details
- See `experiment_plan.md` for strategic overview
- Stan models are fully documented with inline comments

## Version Info

- **Date Created**: 2025-10-28
- **Stan Version**: 2.33+ (uses gp_exp_quad_cov function)
- **Python Version**: 3.8+
- **Key Dependencies**: cmdstanpy, arviz, numpy, pandas
