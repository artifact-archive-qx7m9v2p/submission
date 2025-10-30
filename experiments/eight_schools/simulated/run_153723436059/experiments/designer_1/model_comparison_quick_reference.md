# Quick Reference: Three Competing Model Classes

## Core Question: Why is observed variance LESS than expected?

### Model 1: Standard Hierarchical (Partial Pooling)
**Hypothesis**: It's just random fluctuation with n=8
**Tau prior**: HalfCauchy(0, 25) - diffuse, data-driven
**Prediction**: tau posterior mode around 3-8
**Abandon if**: tau ≈ 0 (go to Model 2) OR tau >> 15 (reconsider)

### Model 2: Near-Complete Pooling
**Hypothesis**: Effects are genuinely identical (I² = 1.6% is real)
**Tau prior**: Exponential(0.5) - strong regularization toward 0
**Prediction**: tau posterior mode < 2, strong shrinkage
**Abandon if**: tau posterior mode > 10 (prior-posterior conflict)

### Model 3: Mixture (Latent Subgroups)
**Hypothesis**: Two groups with opposite effects cancel out
**Structure**: K=2 components with separate mu_k, tau_k
**Prediction**: School 5 in different component than others
**Abandon if**: Single component (max pi > 0.85)

## Decision Cascade

```
Fit all three models
    |
    v
Check convergence (R-hat, ESS, divergences)
    |
    v
Posterior predictive checks
    |
    v
Model comparison (WAIC, LOO-CV)
    |
    v
    /          |          \
Model 1    Model 2    Model 3
wins       wins       wins
    |          |          |
Partial    Complete   Hidden
pooling    pooling    structure
    |          |          |
Standard   Simple     Investigate
inference  estimate   subgroups
```

## Falsification Matrix

| Evidence | Falsifies | Action |
|----------|-----------|--------|
| tau → 0 | Model 1 | Switch to Model 2 |
| tau >> 0 with poor PPC | Model 2 | Stick with Model 1 |
| Single component | Model 3 | Abandon mixture |
| Poor PPC for all | All models | Question normality |
| School 5 always outlier | Exchangeability | Exclude or robust model |

## Expected Outcome (My Bet)

**Most likely**: Model 1 and Model 2 give similar results
- tau posterior will be small (< 5)
- Near-complete pooling justified by data
- Model 2 slightly simpler, Model 1 more conservative

**Least likely**: Model 3 discovers meaningful structure
- Would require clear bimodality in posterior
- EDA shows no evidence for this
- But worth testing to rule out

## Key Parameters to Watch

- **tau in Model 1**: If mode < 3, nearly complete pooling
- **mu across models**: Should be similar (around 10-12)
- **pi in Model 3**: If max(pi) > 0.8, mixture collapses
- **Shrinkage in Model 1**: Schools with larger sigma should shrink more

## Success = Discovery

- Finding Model 2 is best = SUCCESS (homogeneity confirmed)
- Finding Model 1 is best = SUCCESS (partial pooling works)
- Finding Model 3 is best = SURPRISING SUCCESS (hidden structure)
- Finding all fail PPC = SUCCESS (learned normality wrong)

The goal is NOT to confirm Model 1. The goal is to learn which model class the data actually support.
