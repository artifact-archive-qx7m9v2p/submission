# Quick Reference: Model Designer 1 Proposals

## Three Distinct Model Classes

### Model 1: Complete Pooling (Common Effect)
**Mathematical form:** `y_i ~ Normal(mu, sigma_i)`
- **Assumption:** All studies measure identical true effect
- **Key parameter:** mu (single pooled effect)
- **When to abandon:** Study 5 residual |z| > 2.5, ΔLOO > 4 favoring Model 2, poor posterior predictive
- **Expected:** mu ≈ 11.27, CI [3.5, 19.0]

### Model 2: Partial Pooling (Hierarchical Random Effects)
**Mathematical form:** `y_i ~ Normal(theta_i, sigma_i)`, `theta_i ~ Normal(mu, tau)`
- **Assumption:** Studies have different true effects, partially pooled
- **Key parameters:** mu (mean effect), tau (between-study SD), theta_i (study effects)
- **When to abandon:** Tau hits prior boundary, divergences persist, Study 4 removal changes mu > 8 units
- **Expected:** mu ≈ 11.27, tau ≈ 2.02, strong shrinkage (>95%)

### Model 3: Skeptical Prior (Hierarchical with Tight Priors)
**Mathematical form:** Same as Model 2, different priors
- **Assumption:** Large positive effects are rare, make data prove it
- **Priors:** `mu ~ Normal(0, 15)`, `tau ~ Half-Normal(0, 5)` (tighter than Model 2)
- **When to abandon:** Posterior far outside prior 95% CI, or data too weak to update
- **Expected:** mu ≈ 9-10 (shrunk from 11.27), demonstrates robustness

## Key Differences (Why These Are Not Just Variations)

| Aspect | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| **Philosophy** | Parsimony | Conservatism | Skepticism |
| **Heterogeneity** | None (tau = 0 fixed) | Estimated from data | Estimated, but skeptical prior |
| **Shrinkage** | Complete, uniform | Partial, data-driven | Partial, prior-influenced |
| **Parameters** | 1 (mu) | 2 + J (mu, tau, theta) | 2 + J (same structure) |
| **Use case** | High homogeneity | Standard meta-analysis | Sensitivity analysis |

## Critical Red Flags for Major Pivot

1. **Tau > 10 in Model 2:** Heterogeneity much higher than EDA suggests → Need meta-regression or mixture models
2. **Pareto-k > 0.7 for multiple studies:** Outliers present → Need robust likelihoods (Student-t)
3. **Study 4 removal changes mu by >8 units:** Single-study dominance → Investigate robust approaches
4. **Divergent transitions persist (>1%) after adapt_delta=0.99:** Geometry issues → Reparameterize or different model class
5. **All models give negative posterior mu:** Contradicts EDA → Fundamental rethink needed

## Implementation Checklist

### Phase 1: Core Models (6-8 hours minimum)
- [ ] Model 1: Complete pooling fitted and diagnosed
- [ ] Model 2: Partial pooling (non-centered) fitted and diagnosed
- [ ] LOO cross-validation for both models
- [ ] Basic posterior predictive checks

### Phase 2: Sensitivity (4-6 hours)
- [ ] Model 3: Skeptical prior fitted
- [ ] Study 4 removal for all models
- [ ] Prior sensitivity analysis (4 prior setups)

### Phase 3: Advanced Diagnostics (4-6 hours)
- [ ] Full posterior predictive checks with test statistics
- [ ] Residual analysis and visualization
- [ ] Shrinkage factor comparisons
- [ ] Prior-posterior overlap analysis

## Expected Results (Falsifiable Predictions)

**Most likely (70%):** Models 1 and 2 equivalent (ΔLOO < 4), tau ≈ 0, complete pooling adequate

**Alternative (25%):** All models agree, data overwhelms all priors, robust positive effect

**Surprising (5%):** LOO strongly favors Model 2, tau well away from zero, heterogeneity real

**Would shock me:** Negative posterior mu, Pareto-k > 0.7 for multiple studies, tau > 10

## Stopping Rules

**STOP and reconsider if:**
- Persistent divergences after tuning
- Multiple Pareto-k > 0.7
- Posterior predictive p < 0.01 or p > 0.99
- Study 4 removal changes results >50%
- Tau hits upper prior limit

**Then consider:** Robust likelihoods, mixture models, meta-regression, non-parametric approaches

## Success Definition

**NOT success:** All models converging, matching EDA, narrow CIs

**SUCCESS:** Understanding failures, quantifying uncertainty honestly, finding evidence for model revision

---

**Quick start:** Fit Model 2 first with non-centered parameterization, check diagnostics, compare to Model 1 via LOO.
