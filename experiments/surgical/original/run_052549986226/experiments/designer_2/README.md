# Designer 2: Hierarchical Binomial Models with Random Effects

## Overview

This directory contains Bayesian hierarchical binomial models that address severe overdispersion (φ ≈ 3.5-5.1) through group-level random effects on the logit scale. All models naturally handle the Group 1 zero count through shrinkage.

## Model Specifications

### Model 1: Centered Parameterization (Baseline)
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ + α_i
α_i ~ Normal(0, σ)

Priors:
  μ ~ Normal(-2.5, 1.0)      # Centered on logit(7.6%)
  σ ~ Half-Cauchy(0, 1)      # Between-group SD
```

**Status:** Standard parameterization, may have computational issues (funnel geometry)
**Use case:** Demonstrating importance of non-centered parameterization

### Model 2: Non-Centered Parameterization (RECOMMENDED)
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ + σ·z_i
z_i ~ Normal(0, 1)

Priors:
  μ ~ Normal(-2.5, 1.0)
  σ ~ Half-Normal(0, 1)
```

**Status:** Computationally efficient, eliminates funnel geometry
**Use case:** Primary model for inference

### Model 3: Robust with Student-t Priors
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ + σ·z_i
z_i ~ Student-t(ν, 0, 1)

Priors:
  μ ~ Normal(-2.5, 1.0)
  σ ~ Half-Student-t(3, 0, 1)
  ν ~ Gamma(2, 0.1)          # Degrees of freedom
```

**Status:** Heavy-tailed to handle outliers (especially Group 8)
**Use case:** Sensitivity analysis for outlier robustness

## Key Design Decisions

### 1. Why Logit Link?
- Natural for probabilities bounded in [0, 1]
- Group effects plausibly normal on logit scale
- Standard in binomial regression
- Alternative (probit) would give similar results

### 2. Why Non-Centered Parameterization?
- High ICC (0.73) → strong shrinkage → funnel geometry in centered
- Non-centered eliminates correlation between σ and α_i
- 10-100x reduction in divergences expected
- Same model, just reparameterized for better sampling

### 3. How is Group 1 (0/47) Handled?
- **No ad-hoc corrections** (no continuity correction)
- Hierarchical shrinkage provides natural regularization
- Expected posterior: p_1 ≈ 1-3% (low but not zero)
- Uncertainty appropriately inflated due to small n

### 4. Why Robust Model (M3)?
- Group 8 is extreme outlier (z = 3.94, rate = 14.4%)
- Heavy tails allow outliers without distorting population parameters
- Posterior ν informs whether robustness was necessary:
  - ν > 30: Normal adequate, use M2
  - ν < 10: Heavy tails important, use M3

## Prior Justification

All priors are weakly informative, based on EDA findings:

**μ ~ Normal(-2.5, 1.0):**
- Centered on logit(7.6%) = -2.53 (observed pooled rate)
- SD = 1.0 allows 1-35% success rates (95% interval)
- Allows substantial deviation if data supports it

**σ ~ Half-Normal(0, 1) or Half-Cauchy(0, 1):**
- Expected σ ≈ 0.9 based on ICC = 0.73:
  ```
  ICC = σ² / (σ² + π²/3)
  0.73 = σ² / (σ² + 3.29)
  → σ ≈ 0.94
  ```
- Prior has mode around 0.6-0.8, allows up to 2.0
- Consistent with observed overdispersion φ ≈ 3.5-5.1

## Expected Posterior Behavior

### If Models are Correct:

**Population parameters:**
- μ ≈ -2.5 (posterior SD ≈ 0.3)
- σ ≈ 0.8-1.2 (to produce ICC ≈ 0.73)
- Population success rate ≈ 7% (90% CI: 5-11%)

**Group shrinkage:**
- Group 1 (0/47): Shrink from 0% to ~2%
- Group 8 (31/215): Shrink from 14.4% to ~11-12%
- Small-sample groups shrink more than large-sample

**Overdispersion:**
- Posterior φ ≈ 3.5-5.1 (matching observed)
- If model can't reproduce this → model inadequate

**Diagnostics:**
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- Divergences < 1% (especially for M2)

## Falsification Criteria

### Abandon These Models If:

**Computational failure:**
- M2 (non-centered) still has >2% divergences with adapt_delta=0.99
- ESS < 100 for any parameter after 4000 iterations
- All three parameterizations fail to converge

**Statistical failure:**
- Cannot reproduce φ ≈ 3.5-5.1 in posterior predictive
- Prior-posterior conflict (σ pushed to extreme values)
- Multiple Pareto k > 0.7 in LOO-CV (suggests mixture model)
- Group 1 posterior stays at exactly zero (shrinkage not working)

**Alternative needed:**
- If all fail → Switch to beta-binomial (simpler, 2 parameters)
- If discrete subgroups evident → Mixture model
- If structural model justified → Add group-level covariates

## Files in This Directory

### Model Specifications
- `model1_centered.stan` - Centered parameterization
- `model2_noncentered.stan` - Non-centered (RECOMMENDED)
- `model3_robust.stan` - Robust Student-t priors

### Implementation
- `fit_models.py` - Main fitting script (CmdStanPy)
- `model_comparison.py` - Compare models, create visualizations
- `proposed_models.md` - Full model design document (this proposal)
- `README.md` - This file

### Outputs (after running)
- `results/` - Posterior samples, summaries, diagnostics
- `visualizations/` - Comparison plots
- `stan_output/` - Raw Stan output files

## Usage Instructions

### 1. Fit Models

```bash
# Fit all models (recommended order: M2, M3, M1)
python fit_models.py --model all

# Or fit individually
python fit_models.py --model 2  # Non-centered only
python fit_models.py --model 3  # Robust only

# With custom settings
python fit_models.py --model 2 --chains 4 --iter 2000 --adapt_delta 0.95
```

### 2. Compare Models

```bash
python model_comparison.py
```

This creates:
- Population parameter comparison plots
- Group-specific posterior comparison
- Shrinkage analysis
- Overdispersion check
- Group 1 zero count handling
- ν posterior (heavy tail assessment)
- Diagnostic comparison table
- Summary report

### 3. Review Results

**Check diagnostics first:**
- `results/*_diagnostics.json` - Convergence metrics
- Look for: Rhat < 1.01, ESS > 400, Divergences < 1%

**Examine posteriors:**
- `results/*_summary.csv` - Posterior summaries
- `visualizations/*.png` - Visual comparisons

**Read summary:**
- `visualizations/comparison_report.txt` - Comprehensive summary

## Comparison to Other Designers

### vs Designer 1 (Beta-Binomial)
**Hierarchical Binomial Advantages:**
- More flexible (can add group covariates)
- Explicit group effects (interpretable α_i)
- Natural for modeling group heterogeneity

**Beta-Binomial Advantages:**
- More parsimonious (2 parameters vs 12)
- No funnel geometry issues
- Analytically conjugate in some cases

**Decision rule:** If both fit well, prefer simpler (beta-binomial) unless group-specific effects are scientifically interesting.

### vs Designer 3 (Alternative Approaches)
**Hierarchical Binomial Advantages:**
- Standard, well-understood framework
- Extensive literature and diagnostics
- Natural interpretation

**Alternative Approaches Advantages:**
- May handle specific data features better
- Could be more robust to specific violations
- Might offer computational advantages

**Decision rule:** Use LOO-CV to compare predictive performance.

## Computational Requirements

**Time estimates:**
- Model compilation: ~30 seconds per model
- Sampling (4 chains × 2000 iterations):
  - M1 (centered): 5-10 minutes (may need more iterations)
  - M2 (non-centered): 2-5 minutes
  - M3 (robust): 3-7 minutes (Student-t is slower)
- Comparison/visualization: 2-3 minutes

**Total workflow: ~20-30 minutes for complete analysis**

**Memory requirements:**
- Minimal (<1 GB for this dataset)
- Scales with number of groups × iterations

## Scientific Interpretation

### What Each Parameter Means

**μ (population mean):**
- Typical success rate on logit scale
- Converts to probability: p = 1/(1 + exp(-μ))
- Interpretation: "Expected rate for a random group from this population"

**σ (between-group SD):**
- Variability of groups around population mean
- On logit scale, so not directly interpretable as probability
- Larger σ → more heterogeneous groups
- Related to ICC: proportion of variance between groups

**α_i (group effects):**
- Deviation of group i from population mean
- Positive → above average, negative → below average
- Subject to shrinkage (more for small-sample groups)

**Example interpretation:**
> "The population mean success rate is estimated at 7.5% (95% CI: 5.3-10.2%). Groups vary substantially around this mean, with a between-group SD of 0.9 on the logit scale, corresponding to an ICC of 71%. Group 1, despite observing zero successes in 47 trials, is estimated to have a true success rate of 2.1% (95% CI: 0.5-5.8%), reflecting shrinkage toward the population mean. Group 8's observed rate of 14.4% is estimated at 11.8% (95% CI: 8.9-15.3%) after shrinkage."

## Red Flags and Warning Signs

### During Fitting

⚠️ **If you see:**
- Many divergences (>1%) → Try increasing adapt_delta to 0.95
- Low ESS (<400) → Run longer chains
- High Rhat (>1.01) → Check convergence, may need more iterations
- Max treedepth warnings → Increase max_treedepth to 12

⚠️ **If M2 fails to converge:**
- Problem is likely the model class, not parameterization
- Consider switching to beta-binomial
- Or investigate mixture models

### During Diagnostics

⚠️ **If posterior predictive fails:**
- Cannot reproduce φ ≈ 3.5-5.1 → Model inadequate
- Zero counts never occur → Shrinkage too strong
- Many Pareto k > 0.7 → Influential outliers, try mixture

⚠️ **If results seem unreasonable:**
- Group 1 posterior at exactly 0% → Shrinkage not working
- σ posterior at extreme values (>2 or <0.1) → Misspecification
- All groups shrunk to same value → Complete pooling, not partial

## Success Metrics

### Model is SUCCESSFUL if:

✓ **Computational:**
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- Divergences < 1%

✓ **Statistical:**
- Reproduces observed φ ≈ 3.5-5.1
- Posterior predictive checks pass
- LOO Pareto k < 0.7 for all groups

✓ **Scientific:**
- Group 1 gets reasonable shrinkage (not stuck at 0%)
- Small-sample groups shrink more than large-sample
- σ consistent with ICC ≈ 0.73

### Model is FAILING if:

✗ **Any of:**
- Cannot converge despite all efforts
- Posterior predictive systematically fails
- Results scientifically implausible
- Better alternatives exist (via LOO-CV)

## Next Steps After This Analysis

### If Models Succeed:

1. **Compare to other designers** using LOO-CV
2. **Generate predictions** for new groups
3. **Report results** with uncertainty
4. **Sensitivity analyses:**
   - Remove Group 8 (outlier)
   - Remove Group 1 (zero count)
   - Different prior specifications

### If Models Fail:

1. **Try beta-binomial** (Designer 1)
2. **Consider mixture models** (if discrete subgroups)
3. **Investigate alternative links** (probit, cloglog)
4. **Document why failure occurred** (important learning!)

## References

**Stan Documentation:**
- Non-centered parameterization: https://mc-stan.org/docs/stan-users-guide/reparameterization.html
- Hierarchical models: https://mc-stan.org/docs/stan-users-guide/hierarchical-models.html

**Theoretical Background:**
- Gelman & Hill (2007): "Data Analysis Using Regression and Multilevel/Hierarchical Models"
- McElreath (2020): "Statistical Rethinking" (Chapter 13: Multi-level models)

**Overdispersion in Binomial Data:**
- Williams (1982): "Extra-binomial variation in logistic linear models"
- Agresti (2013): "Categorical Data Analysis" (Section 7.1.4)

## Contact and Support

This design is part of a parallel modeling exercise. Compare results across all designers and use LOO-CV for final model selection.

**Remember:** Finding that hierarchical binomial is insufficient is a success, not a failure. It tells us something important about the data generation process.
