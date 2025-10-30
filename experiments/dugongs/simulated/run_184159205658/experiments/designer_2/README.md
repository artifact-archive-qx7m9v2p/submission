# Designer 2 - Bayesian Model Proposals

## Quick Reference

This directory contains independent model proposals from Designer 2 for analyzing the Y-x relationship with N=27 observations showing saturation/plateau behavior.

## File Structure

```
designer_2/
├── README.md                          # This file
├── proposed_models.md                 # Complete model proposals (MAIN DOCUMENT)
└── stan_models/
    ├── model1_michaelis_menten.stan  # MM saturation model
    ├── model2_powerlaw.stan          # Power-law with exponent 0<c<1
    └── model3_exponential.stan       # Exponential approach to asymptote
```

## Three Proposed Models

### 1. Michaelis-Menten Saturation (PRIMARY)
- **Form:** Y = Y_max * x / (K + x)
- **Parameters:** Y_max (asymptote), K (half-saturation), sigma
- **Best for:** Biological/enzymatic processes, clear asymptote
- **Rank:** 1st choice

### 2. Power-Law with Saturation (ALTERNATIVE 1)
- **Form:** Y = a + b * x^c, where 0 < c < 1
- **Parameters:** a (intercept), b (scale), c (exponent), sigma
- **Best for:** When uncertain if true asymptote exists
- **Rank:** 2nd choice

### 3. Exponential Saturation (ALTERNATIVE 2)
- **Form:** Y = Y_max - (Y_max - Y_0) * exp(-r * x)
- **Parameters:** Y_max (asymptote), Y_0 (initial), r (rate), sigma
- **Best for:** Physical equilibration processes
- **Rank:** 3rd choice

## Key Design Principles

1. **Falsification-focused:** Each model has explicit failure criteria
2. **Mechanistic:** Models represent different data-generating hypotheses
3. **Honest uncertainty:** Wide intervals expected with N=27 and sparse high-x data
4. **Adaptive:** Clear decision points for switching model classes

## Critical Decision Points

| If this happens... | Then do this... |
|-------------------|-----------------|
| All models converge, similar LOO-CV | Choose Model 1 (simplest interpretation) |
| Y_max posterior < max(Y_observed) | Abandon asymptotic models → use logarithmic |
| Posterior K > 20 with wide CI | Saturation not identifiable → use simpler model |
| All models fail posterior predictive | Try GP, segmented, or hierarchical models |

## Quick Start

1. Read **proposed_models.md** for full rationale and specifications
2. Fit Model 1 (Michaelis-Menten) first
3. Check convergence (R-hat < 1.01, no divergences)
4. Run posterior predictive checks
5. If adequate, done. If not, try Model 2 or pivot per decision tree.

## Success Metrics

- R-hat < 1.01 for all parameters
- ESS > 400 (bulk and tail)
- No divergent transitions
- Posterior predictive checks pass (no residual patterns)
- LOO-CV Pareto-k < 0.7 for all observations
- Parameters scientifically plausible

## When to Abandon This Approach

- All three models show persistent MCMC problems
- All fail posterior predictive checks
- LOO-CV worse than simple logarithmic baseline
- Prior-posterior overlap >80% (data uninformative)

See **proposed_models.md Section: Decision Points and Pivoting Strategy** for full details.

## Implementation Notes

All models use:
- **Likelihood:** Normal(mu, sigma) - justified by EDA residual normality
- **Priors:** Weakly informative, centered on EDA estimates
- **Sampling:** Stan with HMC (4 chains, 2000 iterations, 1000 warmup)
- **Diagnostics:** R-hat, ESS, divergences, posterior predictive checks, LOO-CV

## Contact Information

**Designer:** Bayesian Modeling Strategist Agent (Designer 2)
**Date:** 2025-10-27
**Data Source:** /workspace/data/data.csv
**EDA Source:** /workspace/eda/eda_report.md
