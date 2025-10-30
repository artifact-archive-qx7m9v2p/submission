# Designer 1: Parametric Modeling Approach

## Overview

This directory contains the parametric modeling proposal for the Y vs x relationship (N=27 observations). The approach focuses on **explicit functional forms** with interpretable parameters, representing three competing scientific hypotheses about the data-generating mechanism.

## Core Philosophy

**Falsificationist approach**: Each model represents a testable hypothesis. Success means discovering which hypothesis best explains the data, not completing all planned models. We plan for failure explicitly.

## Three Competing Hypotheses

1. **Logarithmic Growth** (H1): Y grows without bound, following Weber-Fechner law
   - Common in: Psychophysics, information theory, learning curves
   - Prediction: Y continues increasing slowly forever

2. **Asymptotic Saturation** (H2): Y approaches finite upper limit (Michaelis-Menten)
   - Common in: Enzyme kinetics, resource-limited growth, dose-response
   - Prediction: Y plateaus at Y_max

3. **Polynomial Trajectory** (H3): Y follows quadratic curve (local approximation)
   - Common in: Empirical curve fitting, polynomial approximation
   - Prediction: Valid only within observed range

These hypotheses are **mutually exclusive** in their limiting behavior (x→∞).

## Files in This Directory

### Documentation
- **`proposed_models.md`** - Full detailed proposal (15,000+ words)
  - Scientific justification for each model
  - Complete prior specifications
  - Falsification criteria
  - Computational strategy
  - Sensitivity analyses
  - Red flags and stopping rules

- **`model_summary.md`** - Quick reference guide
  - One-page summary of three models
  - Key falsification criteria
  - Expected outcomes
  - Priority order

- **`README.md`** - This file

### Stan Model Implementations
- **`stan_models/logarithmic_model.stan`** - Model 1: Y = α + β·log(x)
- **`stan_models/michaelis_menten_model.stan`** - Model 2: Y = Y_max - (Y_max - Y_min)·K/(K + x)
- **`stan_models/quadratic_model.stan`** - Model 3: Y = α + β₁·x + β₂·x²

All models include:
- Weakly informative priors based on EDA
- Generated quantities for posterior predictive checks
- Log-likelihood for LOO-CV computation

## Key Decisions

### Model Priority
1. **Logarithmic** (fit first) - Most likely winner, balances fit and parsimony
2. **Michaelis-Menten** (fit second) - Tests key scientific question (asymptote?)
3. **Quadratic** (fit third) - Baseline for comparison, extrapolation concerns

### Comparison Strategy
- **Primary metric**: LOO-CV (leave-one-out cross-validation)
- **Threshold**: ΔLOO > 4 = strong evidence, ΔLOO < 2 = models equivalent
- **Secondary**: Posterior predictive checks, scientific plausibility

### Falsification Criteria

Each model has explicit failure conditions:

| Model | Abandon If... |
|-------|--------------|
| Logarithmic | LOO worse by >4, systematic residuals at high x |
| Michaelis-Menten | Y_max unbounded, K > 25 (no saturation yet) |
| Quadratic | Vertex at x < 31.5 (downturn in data range) |

### Red Flags (All Models)
- High Pareto k (>0.7) for multiple points → Influential observations
- All models fail PPC → Need robust likelihood or heteroscedastic variance
- Gap region deviation → Consider piecewise models
- Prior-posterior conflict → Wrong model class

## Expected Outcome

**Most likely**: Logarithmic model wins
- Best balance of fit, parsimony, and plausibility
- LOO-ELPD ≈ -3 to +3
- Posterior: α≈1.75±0.1, β≈0.27±0.05, σ≈0.12±0.02

**Alternative**: Michaelis-Menten competitive but Y_max weakly identified
- Similar LOO, wider uncertainty on asymptote
- Data insufficient to conclusively distinguish bounded vs unbounded growth

**Baseline**: Quadratic fits best empirically but vertex concerns
- LOO-ELPD ≈ 0 to +2
- Use for interpolation only, not extrapolation

## Next Steps

### Implementation Workflow
1. **Prior predictive checks** - Verify priors are reasonable
2. **MCMC fitting** - Stan with 4 chains, 2000 iterations
3. **Diagnostics** - Rhat, ESS, divergences, energy plots
4. **Posterior predictive checks** - Does model reproduce data?
5. **LOO comparison** - Which model predicts best?
6. **Sensitivity analysis** - Refit without x=31.5, wider priors

### Success Criteria
- **Not**: Completing all three models
- **But**: Finding the model that truly explains the data-generating process
- Willing to abandon plan if evidence suggests it

### Stopping Rules
- **Declare success**: One model clearly superior OR models equivalent (use averaging)
- **Escalate**: All models fail diagnostics OR high Pareto k across models
- **Pivot**: If parametric models inadequate, recommend GP/splines

## Critical Insights from EDA

### Strong Evidence
- Spearman ρ=0.78 (p<0.001): Strong monotonic relationship
- Clear saturation pattern: Diminishing returns at high x
- Approximately constant variance (BP test p=0.546)
- Log R²=0.829, Quadratic R²=0.862 (only 3% difference)

### Key Uncertainties
- **Data gap**: x∈[23, 29] has no observations (6.5-unit gap)
- **Influential point**: x=31.5 (Cook's D=0.84, 4% of data)
- **Limited high-x data**: Only 7 observations for x>15
- **Functional form**: Log vs quadratic vs asymptotic not clearly distinguished

### Critical Questions
1. True asymptote or slow unbounded growth?
2. Constant or decreasing variance?
3. Behavior in gap region?

## Contact

**Designer**: Parametric Modeling Specialist (Model Designer 1)
**Perspective**: Explicit functional forms with interpretable parameters
**Philosophy**: Falsificationist - plan for failure, embrace uncertainty

**Status**: Proposal complete, awaiting implementation approval

---

## Quick Start

To understand this proposal:
1. Read `model_summary.md` for quick overview (5 minutes)
2. Read `proposed_models.md` sections 1-3 for model details (20 minutes)
3. Examine Stan models in `stan_models/` for implementation (10 minutes)

Total: ~35 minutes for comprehensive understanding

---

**Last Updated**: 2025-10-28
