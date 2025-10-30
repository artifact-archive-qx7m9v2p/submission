# Designer 1: Baseline Bayesian Count Models

## Overview

This directory contains **baseline Bayesian GLM models** for overdispersed count data with temporal trends. Models are intentionally simple to isolate mean structure and overdispersion from temporal correlation.

---

## Quick Start

### Recommended Reading Order

1. **START HERE**: `SUMMARY.md` - 5-minute overview
2. **FULL DETAILS**: `proposed_models.md` - Complete specifications (30 min)
3. **IMPLEMENTATION**: `models/*.stan` - Stan code ready to compile

---

## Three Models at a Glance

| Model | Type | Parameters | Runtime | Use When |
|-------|------|------------|---------|----------|
| **Model 1** | NegBin Linear | 3 | 30-60s | **Default baseline** |
| **Model 2** | NegBin Quadratic | 4 | 45-90s | Test vs Model 1 |
| **Model 3** | Gamma-Poisson | N+3 | 2-5 min | Convergence backup |

**Recommendation**: Start with Model 1, compare to Model 2 via LOO.

---

## Key Files

### Documentation
- `README.md` - This file (quickstart)
- `SUMMARY.md` - Executive summary with decision trees
- `proposed_models.md` - Full mathematical specifications

### Stan Models
- `models/model_1_negbin_linear.stan` - Exponential growth baseline
- `models/model_2_negbin_quadratic.stan` - Polynomial trend test
- `models/model_3_gamma_poisson.stan` - Hierarchical alternative

### To Be Created (Implementation Phase)
- `scripts/prior_predictive_check.py`
- `scripts/fit_models.py`
- `scripts/diagnostics.py`
- `scripts/compare_models.py`
- `results/` - Fitted models and summaries
- `figures/` - Diagnostic plots

---

## What These Models Do

### Primary Goals
1. **Handle overdispersion** (Var/Mean = 70.43)
   - Via Negative Binomial distribution
2. **Model growth trend** (R² = 0.937)
   - Via log link with linear/quadratic predictor
3. **Provide baseline** for temporal models
   - Clean comparison for AR/GP models
4. **Test functional form**
   - Linear vs quadratic on log scale

### What They DON'T Do (By Design)
- Model temporal autocorrelation (ACF = 0.971) ← Designer 2's job
- Allow structural breaks ← Designer 3's job
- Use non-parametric smooths ← Designer 3's job
- Time-varying dispersion ← Future extensions

**Note**: High residual autocorrelation is EXPECTED and informative, not a failure.

---

## Model Specifications

### Model 1: Negative Binomial (Linear)

**Likelihood**:
```
C_i ~ NegativeBinomial(mu_i, phi)
log(mu_i) = beta_0 + beta_1 * year_i
```

**Priors**:
```
beta_0 ~ Normal(4.69, 1.0)    # log(mean count)
beta_1 ~ Normal(1.0, 0.5)     # growth rate
phi ~ Gamma(2, 0.1)           # overdispersion
```

**Interpretation**:
- `beta_0`: Expected log count at year=0 (center)
- `beta_1`: Percent change per SD of year = exp(beta_1) - 1
- `phi`: Larger phi → less overdispersion

**When to Reject**:
- LOO >> Model 2 (ΔELPD >4)
- Quadratic pattern in residuals
- PPCs fail systematically

---

### Model 2: Negative Binomial (Quadratic)

**Likelihood**:
```
C_i ~ NegativeBinomial(mu_i, phi)
log(mu_i) = beta_0 + beta_1 * year_i + beta_2 * year_i²
```

**Priors**: Same as Model 1, plus:
```
beta_2 ~ Normal(0, 0.5)       # quadratic term
```

**Interpretation**:
- `beta_2 > 0`: Accelerating growth
- `beta_2 < 0`: Decelerating growth
- `beta_2 ≈ 0`: Linear sufficient (use Model 1)

**When to Reject**:
- `beta_2` credible interval includes zero
- Pareto k >0.7 for >20% observations
- LOO not better than Model 1 (ΔELPD <2)

---

### Model 3: Gamma-Poisson (Hierarchical)

**Likelihood**:
```
C_i | lambda_i ~ Poisson(lambda_i)
lambda_i ~ Gamma(alpha, alpha / mu_i)
log(mu_i) = beta_0 + beta_1 * year_i
```

**Note**: Marginally equivalent to Model 1 but hierarchical.

**When to Use**: Only if Model 1 has convergence issues.

---

## Expected Results

### Likely Outcome
- **Model 1 or 2 converges** quickly (R-hat <1.01)
- **Good mean fit** (captures exponential growth)
- **Overdispersion handled** (phi ≈ 10-40)
- **Residual ACF high** (0.6-0.9) ← EXPECTED
- **Conclusion**: Baseline successful, pass to Designer 2

### Parameter Estimates (Predicted)
- `beta_0` ≈ 4.5 to 4.9 (near prior)
- `beta_1` ≈ 0.7 to 1.2 (strong positive growth)
- `beta_2` ≈ -0.5 to +0.5 (if Model 2)
- `phi` ≈ 10 to 40 (moderate overdispersion)

---

## Decision Guide

### Choose Model 1 If:
- Simplicity preferred (3 vs 4 parameters)
- Model 2 doesn't improve LOO (ΔELPD <2)
- Model 2 beta_2 ≈ 0 (not practically significant)

### Choose Model 2 If:
- Clear improvement (ΔELPD >4)
- beta_2 posterior clearly non-zero
- Model 1 residuals show curvature

### Move to Designer 2 If:
- Residual ACF >0.8 (temporal correlation needs modeling)
- Heteroscedastic residuals over time
- Good fit but correlated errors

### Revisit Framework If:
- All models fail PPCs (p <0.01)
- Extreme phi values (>100 or <5)
- None converge after tuning

---

## Implementation Steps

### Phase 1: Prior Predictive (1 hour)
1. Sample from priors (no data)
2. Generate 100 datasets
3. Check: plausible range [1, 1000]
4. Adjust priors if needed

### Phase 2: Fit Models (2 hours)
1. Compile Stan models
2. Run 4 chains × 2000 iterations
3. Check convergence (R-hat, ESS)
4. Extract posteriors

### Phase 3: Diagnostics (1 hour)
1. Compute LOO-CV
2. Posterior predictive checks
3. Residual diagnostics
4. Parameter interpretations

### Phase 4: Compare & Decide (30 min)
1. Compare Models 1 vs 2
2. Assess residual patterns
3. Recommend best baseline
4. Document limitations

---

## Critical Assumptions

### Explicit Assumptions
1. Counts are Negative Binomial (not Poisson)
2. Overdispersion constant over time (phi fixed)
3. Growth is smooth (polynomial on log scale)
4. Observations independent conditional on trend

### Known Violations (By Design)
1. **Temporal independence** ← ACF = 0.971 violates this
   - **Justification**: Isolate overdispersion first
2. **Constant variance** ← Heteroscedastic per EDA
   - **Justification**: Start simple, test separately
3. **No structural breaks** ← Possible changepoint at year=-0.21
   - **Justification**: Smooth models are baseline

**These violations are intentional** - baseline models establish what patterns remain after mean and dispersion are modeled.

---

## Falsification Mindset

### We Will REJECT Entire Framework If:
- LOO worse than naive mean model
- PPCs show <80% coverage
- Posterior for phi conflicts with EDA (phi >100)
- Computational failure across all parameterizations

### We Will SUCCEED If:
- At least one model converges and fits mean trend
- Overdispersion parameter sensible (phi 5-50)
- Clear recommendation for Model 1 vs 2
- Quantified residual patterns for Designer 2

**Remember**: Finding what doesn't work is progress, not failure.

---

## Connection to EDA

| EDA Finding | Baseline Model Response | Future Extension |
|-------------|------------------------|------------------|
| Var/Mean = 70 | NegBin distribution ✓ | Time-varying phi |
| R² = 0.937 | Log-linear growth ✓ | GP smoothing |
| ACF = 0.971 | **Not addressed** | AR(1), state-space |
| Quadratic R² = 0.964 | Model 2 tests this ✓ | Splines, GP |
| Changepoint? | **Not addressed** | Segmented models |
| Heteroscedastic | **Not addressed** | Time-varying phi |

---

## Software Requirements

```python
# Core dependencies
import cmdstanpy      # >= 1.2.0
import arviz          # >= 0.17.0
import numpy          # >= 1.24.0
import pandas         # >= 2.0.0
import matplotlib     # >= 3.7.0
import scipy          # >= 1.11.0

# Install CmdStan
python -m cmdstanpy.install_cmdstan --version 2.33.1
```

**Platform**: Linux/macOS/Windows (Stan is cross-platform)
**Hardware**: Any modern laptop (4 cores, 4GB RAM)
**Time**: <10 minutes for all 3 models (N=40)

---

## Troubleshooting

### Problem: Divergent transitions
**Solution**: Increase `adapt_delta` to 0.95 or 0.99

### Problem: Low ESS (<100)
**Solution**: Run more iterations (4000 instead of 2000)

### Problem: R-hat >1.01
**Solution**: Check trace plots, ensure chains mixing

### Problem: Model 3 very slow
**Solution**: Skip Model 3, use only Models 1-2

### Problem: phi posterior at boundary
**Solution**: Check prior specification, may need wider Gamma prior

### Problem: Extreme predictions (mu >10,000)
**Solution**: Priors too diffuse, tighten beta_1 prior

---

## Output Structure

```
experiments/designer_1/
├── README.md              ← You are here
├── SUMMARY.md             ← Executive summary
├── proposed_models.md     ← Full specifications
├── models/
│   ├── model_1_negbin_linear.stan
│   ├── model_2_negbin_quadratic.stan
│   └── model_3_gamma_poisson.stan
├── scripts/               ← To be created
│   ├── prior_predictive_check.py
│   ├── fit_models.py
│   ├── diagnostics.py
│   └── compare_models.py
├── results/               ← Generated by scripts
│   ├── model_1_fit.pkl
│   ├── model_2_fit.pkl
│   ├── loo_comparison.csv
│   └── posterior_summaries.csv
└── figures/               ← Generated by scripts
    ├── prior_predictive/
    ├── posterior_predictive/
    ├── diagnostics/
    └── comparison/
```

---

## FAQ

**Q: Why not model autocorrelation here?**
A: Intentional design - isolate overdispersion from correlation for diagnostic clarity. Designer 2 adds temporal structure.

**Q: Why Model 3 if it's equivalent to Model 1?**
A: Robustness check. Sometimes alternative parameterizations sample better. But skip if time-limited.

**Q: What if Model 2 is only slightly better?**
A: Use LOO standard error. If ΔELPD <2×SE, choose simpler Model 1.

**Q: What if all models fail?**
A: Revisit likelihood family. Consider zero-inflation, mixture models, or different distribution.

**Q: How do I know priors are reasonable?**
A: Run prior predictive checks. Prior should generate plausible data (counts 1-1000) without crazy outliers.

**Q: Can I modify the priors?**
A: Yes! Provided priors are weakly informative starting points. Adjust based on domain knowledge or sensitivity analysis.

---

## Next Steps

1. **Read**: `SUMMARY.md` for decision trees
2. **Study**: `proposed_models.md` for full math
3. **Implement**: Prior predictive checks first
4. **Fit**: Start with Model 1, add Model 2 if needed
5. **Diagnose**: LOO, PPCs, residual ACF
6. **Report**: Best model with limitations
7. **Handoff**: Residual patterns to Designer 2

---

## Citation

If using these models, acknowledge:
- **Stan**: Carpenter et al. (2017) - Stan: A probabilistic programming language
- **LOO**: Vehtari et al. (2017) - Practical Bayesian model evaluation
- **Negative Binomial**: Hilbe (2011) - Negative Binomial Regression

---

**Status**: COMPLETE - Ready for implementation
**Estimated time to results**: 3-4 hours from scratch
**Confidence**: High (well-tested model class)

For questions or issues, refer to full documentation in `proposed_models.md`.
