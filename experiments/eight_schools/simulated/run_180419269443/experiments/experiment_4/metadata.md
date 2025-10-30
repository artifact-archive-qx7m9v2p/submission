# Experiment 4: Skeptical-Enthusiastic Prior Ensemble

**Model Class:** Prior sensitivity testing via opposing priors
**Priority:** MEDIUM (mandatory for J=8 small sample)
**Status:** In progress

## Rationale

With only J=8 studies, prior choices can substantially affect inference. This experiment tests robustness by fitting models with opposing prior beliefs:

- **Model 4a (Skeptical):** Priors favor small/null effects
- **Model 4b (Enthusiastic):** Priors favor large positive effects

If models converge despite opposing priors, inference is robust.
If models diverge, data are insufficient to overcome prior influence.

## Model Specifications

### Model 4a: Skeptical Priors

```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)
  theta_i ~ Normal(mu, tau)

Priors (null-favoring):
  mu ~ Normal(0, 10)          # Skeptical of large effects
  tau ~ Half-Normal(0, 5)     # Expects low heterogeneity
```

### Model 4b: Enthusiastic Priors

```
Likelihood:
  y_i ~ Normal(theta_i, sigma_i)
  theta_i ~ Normal(mu, tau)

Priors (optimistic):
  mu ~ Normal(15, 15)         # Expects large positive effect
  tau ~ Half-Cauchy(0, 10)    # Allows high heterogeneity
```

## Falsification Criteria

Abandon if:
1. Models converge trivially (|mu_4a - mu_4b| < 1) → Data overwhelms priors, prior choice irrelevant
2. Models diverge absurdly (|mu_4a - mu_4b| > 20) → Data insufficient
3. Computational failure in either model
4. Stacking weights extreme (w < 0.01) → One prior inappropriate

## Expected Results

**Moderate agreement (|mu_diff| ≈ 3-7):**
- Skeptical: mu ≈ 7-9 (pulled from 0 toward data)
- Enthusiastic: mu ≈ 11-13 (pulled from 15 toward data)
- Ensemble via stacking: mu ≈ 9-11

## Decision Rules

- If |mu_diff| < 5: **Robust inference** - Proceed with confidence
- If 5 < |mu_diff| < 10: **Report range** - Acknowledge uncertainty
- If |mu_diff| > 10: **Data insufficient** - Report honestly

## Comparison Context

- Experiment 1 (weakly informative): mu = 9.87 ± 4.89
- Experiment 2 (complete pooling): mu = 10.04 ± 4.05
- These should bracket the skeptical-enthusiastic range
