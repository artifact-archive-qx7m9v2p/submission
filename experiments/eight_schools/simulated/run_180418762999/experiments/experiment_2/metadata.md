# Experiment 2: Hierarchical Partial Pooling Model

**Model Class**: Hierarchical Partial Pooling with Known Measurement Error
**Date**: 2025-10-28
**Status**: In Development
**Priority**: HIGH (Required for minimum attempt policy)

---

## Model Specification

### Mathematical Formulation

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i)    for i = 1, ..., 8
```

**Group-Level Model** (Partial Pooling):
```
theta_i ~ Normal(mu, tau)    for i = 1, ..., 8
```

**Hyperpriors**:
```
mu ~ Normal(10, 20)           # Population mean
tau ~ Half-Normal(0, 10)      # Between-group SD (regularizing prior)
```

**Non-Centered Parameterization** (to avoid funnel geometry):
```
theta_i = mu + tau * theta_raw_i
theta_raw_i ~ Normal(0, 1)
```

where:
- `y_i`: observed value for group i
- `theta_i`: true (latent) value for group i
- `mu`: population mean (hyperparameter)
- `tau`: between-group standard deviation (hyperparameter)
- `sigma_i`: known measurement error for group i (from data)

---

## Theoretical Justification

### Why This Model

1. **Tests Hierarchical Structure**:
   - Allows groups to differ while sharing information
   - tau quantifies between-group heterogeneity
   - If tau → 0, reduces to complete pooling (Model 1)
   - If tau → ∞, approaches no pooling

2. **Standard Approach**:
   - Classic hierarchical/multilevel model
   - Widely used for meta-analysis
   - Allows data to inform degree of pooling

3. **Challenges EDA Conclusion**:
   - EDA found tau² = 0 (complete pooling)
   - This model formally tests if tau > 0
   - Low power with n=8, so explicit test needed

4. **Non-Centered Parameterization**:
   - Prevents funnel geometry when tau near 0
   - Better convergence than centered parameterization
   - Standard practice for hierarchical models

---

## Falsification Criteria

### Primary Rejection Criteria
Will REJECT this model if:

1. **tau Near Zero** (reduces to Model 1):
   - Posterior 95% CI for tau entirely below 1.0
   - Indicates complete pooling is adequate
   - No benefit from hierarchical structure

2. **Divergences > 5%**:
   - Even with non-centered parameterization
   - Indicates funnel geometry or misspecification
   - Common when tau near boundary (0)

3. **LOO-CV Worse Than Model 1**:
   - ΔELPD < -2×SE (Model 2 significantly worse)
   - Additional complexity not justified
   - Overfitting to noise

4. **Convergence Failure**:
   - R-hat > 1.01 for any parameter
   - ESS < 100 for tau (poorly identified)
   - Cannot reliably estimate parameters

### Secondary Checks
- **Funnel geometry**: Even with non-centered, may indicate tau ≈ 0
- **Poor mixing**: tau bouncing between 0 and larger values
- **Wide posterior for tau**: Unidentifiable from data

---

## Expected Outcomes

### If EDA is Correct (Most Likely)
- **Posterior**: tau ≈ 0-2 (95% CI: [0, 5])
- **Implication**: Between-group variance negligible
- **LOO-CV**: Similar or worse than Model 1 (more parameters, no improvement)
- **Divergences**: Possible (funnel when tau near 0)
- **Decision**: **REJECT** → Model 1 preferred (parsimony)

### If EDA is Wrong (Unlikely)
- **Posterior**: tau > 5 (clearly positive)
- **Implication**: Groups genuinely differ
- **LOO-CV**: Better than Model 1 (ΔELPD > 2×SE)
- **Convergence**: Good (tau away from boundary)
- **Decision**: **ACCEPT** → Hierarchical structure needed

---

## Implementation Details

### PPL: PyMC

**Model Code Structure** (Non-Centered):
```python
import pymc as pm
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('data/data.csv')
y_obs = df['y'].values
sigma_obs = df['sigma'].values
n_groups = len(y_obs)

with pm.Model() as model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=10, sigma=20)
    tau = pm.HalfNormal('tau', sigma=10)  # Regularizing

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood (known measurement error)
    y = pm.Normal('y', mu=theta, sigma=sigma_obs, observed=y_obs)

    # Sample with higher target_accept for hierarchical model
    trace = pm.sample(
        draws=2000,
        tune=2000,        # More tuning for hierarchical
        chains=4,
        target_accept=0.95,  # Higher for complex geometry
        return_inferencedata=True
    )

    # Compute log-likelihood for LOO-CV
    pm.compute_log_likelihood(trace)
```

**Key Differences from Model 1**:
- 10 parameters instead of 1 (mu, tau, theta[1:8])
- Non-centered to avoid funnel
- Higher target_accept (0.95 vs 0.90)
- More tuning iterations (2000 vs 1000)

---

## Validation Pipeline Status

- [ ] Prior Predictive Check
- [ ] Simulation-Based Validation (SBC)
- [ ] Posterior Inference (Fit Model)
- [ ] Posterior Predictive Check
- [ ] Model Critique → ACCEPT/REJECT Decision

---

## Comparison to Model 1

| Aspect | Model 1 (Complete) | Model 2 (Hierarchical) |
|--------|-------------------|----------------------|
| Parameters | 1 (mu) | 10 (mu, tau, theta[1:8]) |
| Pooling | Complete | Partial (data-informed) |
| Complexity | Simple | More complex |
| Assumes | Groups identical | Groups may differ |
| EDA Support | Strong (p=0.42) | Weak (tau²=0) |
| Expected Result | ACCEPT | REJECT (tau≈0) |

---

## Designer Consensus

All 3 designers recommended hierarchical partial pooling:
- **Designer 1**: Standard hierarchical with Half-Cauchy prior
- **Designer 2**: Regularizing Half-Normal prior (chosen)
- **Designer 3**: Test for hidden heterogeneity

**Consensus**: This is the primary alternative to complete pooling

---

## Notes

**Purpose**: Test whether EDA conclusion (complete pooling) holds under formal Bayesian analysis.

**Expected Outcome**: Model will estimate tau ≈ 0, effectively reducing to Model 1. This will STRENGTHEN confidence in complete pooling by showing:
1. Formal hierarchical analysis agrees with EDA
2. Data do not support between-group heterogeneity
3. Additional complexity not warranted

**If tau > 0**: Would indicate EDA missed heterogeneity (power limitation with n=8)

**Computational Challenge**: Funnel geometry expected when tau ≈ 0, even with non-centered parameterization. Divergences < 5% acceptable if tau posterior clearly near 0.
