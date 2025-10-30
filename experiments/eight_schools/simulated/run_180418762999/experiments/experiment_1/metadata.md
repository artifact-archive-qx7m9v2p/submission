# Experiment 1: Complete Pooling Model

**Model Class**: Complete Pooling with Known Measurement Error
**Date**: 2025-10-28
**Status**: In Development
**Priority**: HIGH (Baseline)

---

## Model Specification

### Mathematical Formulation

**Likelihood**:
```
y_i ~ Normal(mu, sigma_i)    for i = 1, ..., 8
```

where:
- `y_i`: observed value for group i
- `mu`: population mean (shared parameter to estimate)
- `sigma_i`: known measurement error for group i (from data)

**Prior Distribution**:
```
mu ~ Normal(10, 20)
```

Justification:
- Center at 10: EDA weighted mean = 10.02
- SD = 20: Weakly informative, allows range [-30, 50]
- Lets data dominate while preventing extreme values

---

## Theoretical Justification

### Why This Model
1. **EDA Support**:
   - Chi-square homogeneity test: p = 0.42 (groups are homogeneous)
   - Between-group variance = 0 (observed < expected from measurement error)
   - Signal-to-noise ratio ≈ 1 (measurement error dominates)

2. **Parsimony**:
   - Simplest model consistent with data
   - Single parameter to estimate
   - Maximum information sharing across all 8 observations

3. **Statistical Principle**:
   - Properly accounts for heteroscedastic known measurement error
   - Weights observations by precision (1/sigma_i²)
   - Analogous to meta-analysis with single true effect

---

## Falsification Criteria

### Primary Rejection Criteria
Will REJECT this model if:

1. **LOO-CV diagnostic**: Any observation has Pareto k > 0.7
   - Indicates influential observation
   - Suggests model inadequate for that observation

2. **Posterior Predictive Check**: Systematic misfit
   - Observed variance consistently outside 95% predictive interval
   - Systematic residual patterns

3. **Prior-Posterior Conflict**:
   - Substantial conflict between prior and likelihood
   - May indicate model misspecification

### Secondary Checks
- Convergence: R-hat < 1.01 for all parameters (should be trivial for 1-parameter model)
- Effective Sample Size: ESS > 1000 (expect > 4000 with 4 chains × 2000 draws)
- Computational: No divergences expected (if present, indicates misspecification)

---

## Expected Outcomes

### If EDA is Correct (Expected)
- **Posterior**: mu ≈ 10 ± 4
- **LOO-CV**: Best predictive performance among all models
- **PPC**: Good fit, no systematic deviations
- **Diagnostics**: Perfect (R-hat ≈ 1.00, ESS > 4000)
- **Decision**: **ACCEPT**

### If EDA is Wrong (Unexpected)
- **Posterior**: Wide, multimodal, or contradicts prior
- **LOO-CV**: Poor predictive performance (Pareto k > 0.7)
- **PPC**: Systematic misfit (e.g., underestimates variance)
- **Decision**: **REJECT** → Try Model 3 (Measurement Error Inflation)

---

## Implementation Details

### PPL: PyMC

**Model Code Structure**:
```python
import pymc as pm
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('data/data.csv')
y_obs = df['y'].values
sigma_obs = df['sigma'].values

with pm.Model() as model:
    # Prior
    mu = pm.Normal('mu', mu=10, sigma=20)

    # Likelihood (known measurement error)
    y = pm.Normal('y', mu=mu, sigma=sigma_obs, observed=y_obs)

    # Sample
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.90,
        return_inferencedata=True
    )

    # Compute log-likelihood for LOO-CV
    pm.compute_log_likelihood(trace)
```

---

## Validation Pipeline Status

- [ ] Prior Predictive Check
- [ ] Simulation-Based Validation (SBC)
- [ ] Posterior Inference (Fit Model)
- [ ] Posterior Predictive Check
- [ ] Model Critique → ACCEPT/REJECT Decision

---

## Notes

This is the **baseline model** that all other models will be compared against. Given the strong EDA support for complete pooling, we expect this model to:
1. Pass all validation checks
2. Have best LOO-CV score
3. Be accepted as adequate

If this model fails, it indicates either:
- Fundamental misunderstanding of the data
- Measurement error assumptions are wrong → Try Model 3
- Data quality issues
