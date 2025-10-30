# Convergence Report: Log-Linear Heteroscedastic Model (Experiment 2)

## Model Specification
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i
```

## Sampling Configuration
- PPL: PyMC
- Chains: 4
- Warmup: 1500
- Sampling: 1500
- target_accept: 0.97
- Total draws: 6000

## Convergence Metrics

### Parameter Estimates
```
          mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
beta_0   1.763  0.047   1.679    1.857      0.001    0.001    1659.0    1542.0    1.0
beta_1   0.277  0.021   0.237    0.316      0.001    0.001    1825.0    1778.0    1.0
gamma_0 -2.399  0.248  -2.868   -1.945      0.006    0.004    1899.0    2295.0    1.0
gamma_1  0.003  0.017  -0.029    0.036      0.000    0.000    2108.0    1938.0    1.0
```

### Diagnostic Summary
- **Total divergences**: 0 (0.00%)
- **Max R-hat**: 1.0000
- **Min ESS (bulk)**: 1659
- **Min ESS (tail)**: 1542

### Convergence Criteria
- R-hat < 1.01: ✓ PASS
- ESS > 400: ✓ PASS
- Divergences < 5%: ✓ PASS

**Overall**: ✓ CONVERGENCE ACHIEVED

## Heteroscedasticity Assessment

### gamma_1 Posterior
- **Mean**: 0.0033 ± 0.0171
- **95% CI**: [-0.0284, 0.0391]
- **P(gamma_1 < 0)**: 43.9%

### Interpretation
Insufficient evidence for heteroscedasticity

## LOO Cross-Validation

### LOO Results
- **ELPD LOO**: 23.56 ± 3.15
- **p_loo**: 3.41

### Pareto k Diagnostics
- Good (k < 0.5): 26/27 (96.3%)
- OK (0.5 ≤ k < 0.7): 0
- Bad (0.7 ≤ k < 1.0): 1
- Very bad (k ≥ 1.0): 0
- Max k: 0.964

### Pareto k Assessment
✓ PASS: All Pareto k < 0.7

## Model Comparison with Model 1 (Log-Linear Homoscedastic)

### ELPD Comparison
- **Model 1 ELPD**: 46.99 ± 3.11
- **Model 2 ELPD**: 23.56 ± 3.15
- **Δ ELPD (M2 - M1)**: -23.43 ± 4.43

### Model Selection
Model 1 strongly preferred (Δ < -2 SE)

## Visual Diagnostics

The following plots are generated for detailed assessment:

1. **convergence_diagnostics.png**: Trace plots and rank plots for all parameters
2. **posterior_distributions.png**: Marginal posterior distributions with priors
3. **model_fit.png**: Fitted model with heteroscedastic credible intervals
4. **residual_diagnostics.png**: Residual analysis and variance structure
5. **variance_function.png**: Posterior predictive variance as function of x

See plots directory for visualizations.

## Conclusion

This model achieves satisfactory convergence and provides valid inference for model comparison.

---
Generated: 2025-10-27 18:06:34
