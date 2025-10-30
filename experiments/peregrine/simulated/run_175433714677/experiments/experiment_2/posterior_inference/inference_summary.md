
# Model 2 Inference Summary: Quadratic Negative Binomial

## Decision: **REJECT**

## Model Specification
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0, 0.5)
  φ ~ Exponential(0.667)
```

## Convergence
- Maximum R̂: 1.0000 (target: < 1.01) ✓
- Minimum ESS (bulk): 8710 (target: > 400) ✓
- Minimum ESS (tail): 7783 (target: > 400) ✓
- Divergent transitions: 0
- Status: CONVERGED (excellent)

## Posterior Estimates

### Key Parameter: β₂ (Quadratic Term)
- Mean: 0.0594
- 95% CI: [-0.0505, 0.1731]
- Excludes 0: False ✗
- Meaningful (|β₂| > 0.1): False ✗
- **Significance: NOT SIGNIFICANT**

### All Parameters
```
         mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
beta0   4.300  0.073   4.160    4.434      0.001    0.001    8851.0    7930.0    1.0
beta1   0.862  0.050   0.763    0.951      0.000    0.000   10148.0    8217.0    1.0
beta2   0.059  0.057  -0.048    0.166      0.001    0.001    8710.0    7783.0    1.0
phi    13.579  3.368   7.722   20.058      0.032    0.023   10290.0    8304.0    1.0
```

## Model Comparison (LOO-CV)

### ELPD Scores
- Model 1 (Linear): -174.61 ± 4.80
- Model 2 (Quadratic): -175.06 ± 5.21
- **ΔELPD (Model 2 - Model 1): -0.45 ± 7.09**

### Interpretation
**Models equivalent** (Model 1 slightly favored)

### Detailed Comparison
```
                   rank    elpd_loo     p_loo  elpd_diff     weight   se  dse  warning  scale
Model_1_Linear        0 -174.610621  1.514178   0.000000   0.561827  4.80  0.00    False    log
Model_2_Quadratic     1 -175.056026  2.124954   0.445405   0.438173  5.21  0.93    False    log
```

## Posterior Predictive Checks

### 1. Residual Curvature
- Quadratic coefficient in residuals: **-11.99**
- Target: |coef| < 1.0
- Status: ✗ **FAIL** (WORSENED from -5.22 in Model 1!)

### 2. Early vs Late Period Fit
- MAE (early period): 6.10
- MAE (late period): 27.80
- MAE ratio (late/early): **4.56**
- Target: < 2.0
- Status: ✗ **FAIL** (WORSE than 4.17 in Model 1)

### 3. Var/Mean Recovery
- Observed Var/Mean: 68.67
- Posterior predictive Var/Mean: 96.91 [56.98, 158.82]
- Target: overlaps [50, 90]
- Status: ✓ PASS

### 4. Coverage Calibration
- 80% interval coverage: 100.0%
- 95% interval coverage: 100.0%
- Target: 80-95%
- Status: ✗ **FAIL** (too conservative/wide)

## Visual Diagnostics

All diagnostic plots confirm the quantitative findings:

1. **`convergence_trace.png`**: Trace plots show excellent mixing with stable chains
2. **`convergence_rank.png`**: Rank plots confirm uniform mixing across chains
3. **`posterior_distributions.png`**: β₂ posterior clearly overlaps 0, showing NO significant quadratic effect
4. **`residual_diagnostics.png`**: Residuals STILL SHOW STRONG CURVATURE PATTERN (coef=-11.99, WORSE than Model 1's -5.22)
5. **`posterior_predictive_checks.png`**: Model shows poor fit, especially in late period with MAE ratio=4.56
6. **`model_comparison.png`**: Model 2 provides NO improvement - models are equivalent with Model 1 slightly favored

## Criteria Summary

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Convergence | R̂ < 1.01, ESS > 400 | R̂=1.000, ESS=8710 | ✓ |
| β₂ significance | CI excludes 0, \|β₂\| > 0.1 | [-0.051, 0.173] | ✗ |
| LOO improvement | ΔELPD > 4 | -0.4 | ✗ |
| Residual curvature | \|coef\| < 1.0 | 12.0 | ✗ |
| Early/late fit | ratio < 2.0 | 4.56 | ✗ |
| Var/Mean recovery | in [50, 90] | [57.0, 158.8] | ✓ |
| Coverage | 80-95% | 80%=100%, 95%=100% | ✗ |

## Final Decision: **REJECT**

### Justification for REJECTION

Model 2 (Quadratic Negative Binomial) **FAILS to improve over Model 1**:

#### 1. β₂ is NOT significant
The quadratic term has a 95% CI of [-0.051, 0.173], which clearly includes 0. The posterior mean of 0.059 is far too small to have meaningful impact (threshold: |β₂| > 0.1). **The data do NOT support a quadratic trend.**

#### 2. NO LOO-CV improvement
ΔELPD = -0.45 ± 7.09 indicates Model 2 provides NO predictive advantage. In fact, Model 1 (Linear) is slightly favored. Adding the quadratic term merely adds complexity without improving fit.

#### 3. Residual curvature WORSENED
The quadratic coefficient in residuals is **-11.99**, compared to -5.22 in Model 1. This is **MORE THAN DOUBLE the misfit!** The quadratic form in the model is clearly misspecified - it's not capturing the true nonlinearity in the data.

#### 4. Late-period fit DEGRADED
MAE ratio = 4.56, compared to 4.17 in Model 1. The fit actually got **WORSE**, not better. This suggests the issue is not polynomial curvature but something more fundamental (possibly exponential growth, changepoint, or missing covariates).

#### 5. Overly conservative intervals
100% coverage on both 80% and 95% intervals indicates the model is too uncertain - it's not learning the pattern effectively.

### What Went Wrong?

The EDA predicted **ΔELPD > 10**, but we observed **ΔELPD ≈ 0**. This dramatic failure reveals:

#### Wrong functional form
The data's curvature is NOT well-captured by a simple quadratic. The persistent and worsened residual curvature (-11.99) suggests:
- **Exponential growth** (log-linear in log-space)
- **Changepoint structure** (different regimes)
- **Time-varying growth rate**
- **Missing covariates** driving the pattern

#### Prior-data conflict
The weak posterior for β₂ (essentially unchanged from prior) suggests the likelihood doesn't strongly favor any quadratic term, despite EDA suggesting otherwise.

#### Overfitting signature
Model 2's higher p_loo (2.12) vs Model 1 (1.51) indicates effective parameters increased without improving fit - classic overfitting.

### Recommendation

**REJECT Model 2** - the quadratic term provides no benefit and may be harmful.

### Next Steps

1. **Try exponential model (Model 3)**: The accelerating growth suggests exponential rather than polynomial:
   - `log(μ[i]) = β₀ + exp(β₁ + β₂ × year[i])` or similar
   - This would allow growth rate itself to grow

2. **Investigate changepoint models**: The 4.6× MAE ratio suggests different regimes:
   - Pre/post changepoint with different growth rates
   - Structural break model

3. **Consider external factors**: What changed in the data-generating process?
   - Policy changes?
   - Measurement changes?
   - Population growth?

4. **Re-examine EDA assumptions**: Why did EDA predict quadratic would help but Bayesian inference disagrees?
   - EDA used point estimates (MLE/OLS)
   - Bayesian approach properly accounts for uncertainty
   - May need more sophisticated residual analysis

### Key Lesson

This is a textbook example of **"the sampler is telling you about your model"**. Despite:
- Excellent convergence (R̂=1.00, ESS>8000)
- No divergences
- Proper prior specification

The model still **FAILS** because the **functional form is wrong**. No amount of sampling tuning would fix this. The data simply don't exhibit simple quadratic curvature - they need a different model structure entirely.

### Comparison to Model 1

| Metric | Model 1 (Linear) | Model 2 (Quadratic) | Verdict |
|--------|------------------|---------------------|---------|
| ELPD | -174.61 | -175.06 | Model 1 better |
| p_loo | 1.51 | 2.12 | Model 1 simpler |
| Residual curvature | -5.22 | -11.99 | Model 1 better |
| MAE ratio | 4.17 | 4.56 | Model 1 better |
| Parameters | 3 | 4 | Model 1 simpler |

**Model 1 (Linear) remains the preferred model** - simpler and performs better or equivalently on all metrics.
