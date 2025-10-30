# Experiment 2: Complete Pooling (Common Effect) Model

**Model Class:** Complete pooling - assumes single true effect across all studies
**Priority:** HIGH (mandatory per experiment plan)
**Status:** In progress

## Model Specification

**Likelihood:**
```
y_i ~ Normal(mu, sigma_i)    for i = 1,...,8
```

**Prior:**
```
mu ~ Normal(0, 50)
```

**Key Feature:** No between-study variance parameter (tau). All studies measure the same true effect with different sampling errors.

## Theoretical Justification

From experiment plan and Designer 1:
- AIC-preferred in EDA (63.85 vs 65.82 for random effects)
- I²=2.9% suggests homogeneity
- Tests null hypothesis: "no between-study heterogeneity"
- Very fast (~30 sec), provides comparison benchmark

## Falsification Criteria

Abandon if:
1. LOO strongly prefers Experiment 1 (ΔELPD > 4)
2. Posterior predictive checks show systematic under-dispersion
3. Residuals show patterns (e.g., Study 5 consistently extreme)
4. Prior sensitivity analysis reveals instability

## Expected Results

- mu ≈ 11.27 ± 3.8 (narrower CI than Experiment 1)
- Similar to EDA pooled estimate
- If |mu_Exp2 - mu_Exp1| < 2 and CIs overlap strongly, pooling is adequate

## Comparison to Experiment 1

This model is nested within Experiment 1 (tau → 0).
If data strongly support tau > 0, Experiment 1 will be preferred by LOO.
If data consistent with tau ≈ 0, this simpler model may be adequate by parsimony.
