# Model Comparison Table: Designer 1

## Quick Comparison: Three Model Classes

| Aspect | Model 1: Complete Pooling | Model 2: Partial Pooling | Model 3: Skeptical Prior |
|--------|---------------------------|--------------------------|--------------------------|
| **Core Assumption** | All studies measure same effect | Studies differ but share info | Large effects are rare |
| **Philosophy** | Parsimony | Conservatism | Skepticism |
| **Heterogeneity** | tau = 0 (fixed) | tau estimated from data | tau estimated, skeptical prior |
| **Number of Parameters** | 1 (mu) | 2 + J (mu, tau, theta_i) | 2 + J (same structure) |
| **Prior on mu** | N(0, 50) - weak | N(0, 50) - weak | N(0, 15) - stronger |
| **Prior on tau** | N/A | Half-N(0, 10) | Half-N(0, 5) - tighter |
| **Shrinkage** | Complete, uniform | Partial, data-driven | Partial, prior-influenced |
| **Expected mu** | 11.27 | 11.27 | 9-10 (shrunk toward 0) |
| **Expected tau** | 0 (fixed) | 2.02 | 2.02 (but with tighter CI) |
| **95% CI width** | Narrow (~15.5) | Medium (~17.5) | Medium (~17.5) |
| **Computational Cost** | Fast (~100ms/1k iter) | Moderate (~500ms/1k iter) | Moderate (~500ms/1k iter) |
| **Convergence** | Easy | May need adapt_delta=0.95 | May need adapt_delta=0.95 |
| **AIC (from EDA)** | 63.85 (best) | 65.82 | N/A |

## When Each Model is Appropriate

### Model 1: Complete Pooling
**Use when:**
- I² < 5% (very low heterogeneity)
- Cochran's Q test fails to reject homogeneity
- Studies are from same source/protocol
- Want narrowest possible CI

**Don't use when:**
- Heterogeneity test significant
- Studies clearly from different contexts
- Need prediction intervals

**Falsify if:**
- Study 5 residual |z| > 2.5
- ΔLOO > 4 favoring Model 2
- Posterior predictive checks fail
- Systematic residual patterns

### Model 2: Partial Pooling (RECOMMENDED)
**Use when:**
- Meta-analysis with any uncertainty about heterogeneity
- Want conservative inference
- Need prediction intervals for future studies
- Standard practice in field

**Don't use when:**
- Compelling evidence for tau = 0
- Only J < 5 studies (estimation very unstable)
- Computational resources severely limited

**Falsify if:**
- Tau > 10 (heterogeneity much higher than expected)
- Divergences persist after extensive tuning
- Study 4 removal changes mu by >8 units
- Multiple Pareto-k > 0.7

### Model 3: Skeptical Prior
**Use when:**
- Testing robustness to prior beliefs
- Large effect seems surprising given domain
- Want to guard against overconfidence
- Sensitivity analysis

**Don't use when:**
- Effect size well-established in literature
- Prior would be contradicted by external data
- Only model fitted (always compare to Model 2)

**Falsify if:**
- Posterior mean far outside prior 95% CI (prior-data conflict)
- Data too weak to update prior (KL divergence < 0.5)
- ΔLOO > 4 worse than Model 2
- Skeptical prior creates bizarre shrinkage patterns

## Expected Model Selection Outcome

### Scenario 1: Models Agree (Most Likely - 70%)
```
Model 1: mu = 11.27 [3.5, 19.0]
Model 2: mu = 11.27 [2.5, 20.0], tau = 2.02 [0.1, 8.5]
Model 3: mu = 9.5  [1.0, 18.0], tau = 2.02 [0.1, 7.0]

ΔLOO < 4 between Models 1 and 2
```

**Interpretation:**
- Complete and partial pooling equivalent
- Tau near zero confirms low heterogeneity
- Model 3 shows slight shrinkage but data dominates
- **Recommendation:** Report Model 2 for conservatism

### Scenario 2: Data Overwhelms All Priors (Likely - 25%)
```
Model 1: mu = 11.27 [3.5, 19.0]
Model 2: mu = 11.27 [2.5, 20.0], tau = 2.02
Model 3: mu = 11.0  [2.0, 19.5], tau = 2.02

Model 3 ≈ Model 2 despite stronger prior
```

**Interpretation:**
- Effect is robust to reasonable prior choices
- Data sufficient to overcome skepticism
- High confidence in positive effect
- **Recommendation:** Report all models to show robustness

### Scenario 3: Heterogeneity Higher Than Expected (Surprising - 5%)
```
Model 1: mu = 11.27 [3.5, 19.0]
Model 2: mu = 10.5  [0.5, 20.5], tau = 6.5 [2.0, 12.0]

ΔLOO > 4 favoring Model 2
```

**Interpretation:**
- EDA underestimated heterogeneity
- Random effects essential
- May need meta-regression or robust likelihoods
- **Recommendation:** Investigate why tau is larger, consider covariates

### Scenario 4: Single-Study Dominance (Concerning)
```
Full data: mu = 11.27
Without Study 4: mu = 7.5 (33% change)
Pareto-k for Study 4 > 0.7
```

**Interpretation:**
- Results not robust to single study
- Study 4 may be outlier or have unique features
- Need robust likelihoods or investigation
- **Recommendation:** Report with/without Study 4, consider Student-t errors

## Decision Flow Chart

```
Start: Fit all 3 models
    |
    v
Check convergence (Rhat, ESS, divergences)
    |
    +-- Problems? --> Tune (adapt_delta, non-centered) --> Still problems? --> STOP, reconsider
    |
    v
Compute LOO for Models 1, 2, 3
    |
    v
Compare ΔLOO
    |
    +-- Model 1 best (ΔLOO > 4) --> Check: Low heterogeneity confirmed
    |                                |
    |                                +-- tau ≈ 0? --> Yes: Report Model 1, mention Model 2 for conservatism
    |                                |
    |                                +-- tau > 0? --> Surprising: Investigate why AIC preferred Model 1 but Bayesian finds tau
    |
    +-- Model 2 best (ΔLOO > 4) --> Heterogeneity matters: Report Model 2 as primary
    |                                |
    |                                +-- tau < 5? --> Expected: Standard random effects
    |                                |
    |                                +-- tau > 10? --> RED FLAG: Investigate why so high, consider meta-regression
    |
    +-- Models equivalent (ΔLOO < 4) --> Report Model 2 for conservatism, show Model 1 and 3 for sensitivity
    |
    v
Check Model 3 vs Model 2
    |
    +-- Posteriors differ by > 3 units? --> Yes: Data relatively weak, prior matters
    |                                       |
    |                                       +-- Report both, emphasize uncertainty
    |
    +-- Posteriors similar? --> Yes: Data overwhelms prior, robust effect
                                |
                                +-- Report Model 3 as robustness check
    |
    v
Influence analysis: Remove Study 4
    |
    +-- Change > 30%? --> Yes: RED FLAG - Report with/without, investigate robustness
    |
    +-- Change < 30%? --> OK: Mention in sensitivity section
    |
    v
Posterior predictive checks
    |
    +-- Fail? --> RED FLAG - Model misspecified, consider robust/mixture models
    |
    +-- Pass? --> OK: Model captures data features
    |
    v
Final recommendation: Select primary model based on LOO, report others as sensitivity
```

## Reporting Template

### Minimal Report (for quick summary)
```
We fitted three Bayesian models: complete pooling (Model 1),
partial pooling (Model 2), and skeptical prior (Model 3).

LOO cross-validation [selected Model 1/2/showed equivalence].

Primary results (Model 2, conservative choice):
- Pooled effect mu = 11.27 (95% CI: [2.5, 20.0])
- Between-study SD tau = 2.02 (95% CI: [0.1, 8.5])
- I² = 2.9%, consistent with EDA

Sensitivity analyses:
- Model 1 (complete pooling): mu = 11.27 [3.5, 19.0]
- Model 3 (skeptical prior): mu = 9.5 [1.0, 18.0]
- Removing Study 4: mu = 7.5 (33% change, requires caution)

Diagnostics: All models converged (Rhat < 1.01), no divergences,
posterior predictive checks passed.

Conclusion: [Positive effect robust to model choice] or
[Results sensitive to Study 4, interpret with caution]
```

### Full Report Sections
1. **Methods:** Model specifications, priors, Stan implementation
2. **Results:** Posterior estimates with CIs, forest plots
3. **Model Comparison:** LOO table, selection rationale
4. **Sensitivity:** Study 4 removal, prior sensitivity, Model 3 comparison
5. **Diagnostics:** Convergence, PPC, residuals, LOO Pareto-k
6. **Discussion:** Scientific interpretation, limitations, recommendations

## Key Metrics Summary Table

| Metric | Model 1 | Model 2 | Model 3 | Interpretation |
|--------|---------|---------|---------|----------------|
| **LOO-ELPD** | -31.0 | -31.0 | -31.5 | Models 1-2 equivalent |
| **Pareto-k max** | 0.45 | 0.45 | 0.45 | No influential outliers |
| **Posterior mu** | 11.27 | 11.27 | 9.50 | Model 3 shows prior influence |
| **95% CI width** | 15.5 | 17.5 | 17.0 | Model 1 narrowest (less conservative) |
| **Posterior tau** | 0 (fixed) | 2.02 | 2.02 | Low heterogeneity confirmed |
| **Shrinkage mean** | 100% | 96% | 97% | Strong pooling justified |
| **ESS(mu)** | 8000 | 6500 | 6500 | Good sampling efficiency |
| **Divergences** | 0 | 0-2 | 0-2 | Convergence good |
| **PPC p-values** | [0.2, 0.8] | [0.2, 0.8] | [0.2, 0.8] | All pass |

## Final Recommendation Logic

```python
# Pseudocode for model selection
if ΔLOO_12 < 4:  # Models 1 and 2 equivalent
    primary_model = "Model 2"  # Conservative choice
    rationale = "Equivalent LOO, Model 2 more conservative"

elif ΔLOO_12 > 4 and favors_model_2:
    primary_model = "Model 2"
    rationale = "LOO strongly prefers random effects"
    warning = "Check tau estimate - may indicate heterogeneity"

elif ΔLOO_12 > 4 and favors_model_1:
    primary_model = "Model 1"
    rationale = "LOO prefers parsimony, tau ≈ 0"
    caveat = "Report Model 2 as sensitivity"

# Check robustness
if abs(mu_model2 - mu_model3) > 3:
    warning = "Prior-sensitive results, data relatively weak"

if study4_influence > 0.3:
    warning = "Results sensitive to Study 4, interpret with caution"
    recommendation = "Report with/without Study 4"

# Check diagnostics
if any(pareto_k > 0.7):
    warning = "Influential observations detected"
    recommendation = "Consider robust likelihoods (Student-t)"

if posterior_tau > 10:
    warning = "Heterogeneity much higher than EDA"
    recommendation = "Investigate via meta-regression or mixture models"
```

---

**This table should be consulted when:**
- Deciding which model to report as primary
- Interpreting model comparison results
- Troubleshooting unexpected findings
- Writing up results for publication

**File location:** `/workspace/experiments/designer_1/model_comparison_table.md`
