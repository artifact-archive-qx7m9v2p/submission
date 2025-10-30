# Bayesian Model Experiment Plan
## Binomial Trial Dataset - Hierarchical Modeling Strategy

**Date:** 2024
**Data:** 12 groups with binomial trials (severe overdispersion φ ≈ 3.5-5.1, ICC = 0.73)

---

## Executive Summary

Three independent model designers proposed **9 total models** addressing the severe overdispersion detected in EDA. After synthesis and removing duplicates, this plan prioritizes **4 core models** for sequential evaluation, with clear decision points for when to stop or pivot.

### Core Finding from Design Phase
All three designers converged on the need for hierarchical structure due to:
- Severe overdispersion (φ ≈ 3.5-5.1, p < 0.0001)
- High ICC (0.73) indicating 73% variance between groups
- Group 1 zero count (0/47) requiring shrinkage
- Group 8 extreme outlier (z=3.94) requiring robust handling

### Recommended Model Priority
1. **Beta-binomial (reparameterized)** - Best preliminary AIC, handles zero counts naturally
2. **Hierarchical binomial (non-centered)** - Standard approach, highly flexible
3. **Student-t hierarchical** - Robust to outliers
4. **Horseshoe hierarchical** - Tests sparsity hypothesis (exploratory)

---

## Model Synthesis from Designers

### Designer 1: Beta-Binomial Approaches
**Proposed 3 models:**
- Model A: Standard beta-binomial (α, β)
- Model B: Reparameterized beta-binomial (μ, κ) ⭐ **RECOMMENDED by designer**
- Model C: Two-component mixture (falsification test)

**Key insights:**
- Beta-binomial had best AIC (47.69) in EDA preliminary analysis
- No log-odds singularity for Group 1 zero count
- Models A and B mathematically equivalent, B more interpretable
- Designer expects mixture to LOSE (no clusters in EDA)

### Designer 2: Hierarchical Binomial with Random Effects
**Proposed 3 models:**
- Model 1: Centered parameterization (baseline)
- Model 2: Non-centered parameterization ⭐ **RECOMMENDED by designer**
- Model 3: Robust Student-t priors

**Key insights:**
- Non-centered eliminates funnel geometry (10-100× fewer divergences expected)
- Natural shrinkage handles Group 1 without ad-hoc corrections
- Expected σ ≈ 0.8-1.2 on logit scale (consistent with ICC = 0.73)
- Model 3 overlaps with Designer 3's primary recommendation

### Designer 3: Robust and Alternative Models
**Proposed 3 models:**
- Student-t hierarchical ⭐ **PRIMARY recommendation**
- Horseshoe hierarchical (sparse effects)
- Mixture hierarchical (discrete clusters)

**Key insights:**
- Student-t provides heavy tails for 42% outlier rate
- If ν → ∞, automatically reduces to Normal (parsimony!)
- Horseshoe tests if only 3-5 groups truly differ
- Designer cautions N=12 too small for reliable mixture clustering

---

## Synthesized Experiment Plan

### Model Selection Rationale

After removing duplicates and synthesizing recommendations:

1. **Start with beta-binomial** (Designer 1's Model B)
   - Best preliminary AIC
   - Simplest (2 parameters)
   - Natural handling of zero counts
   - If successful, provides strong baseline

2. **Follow with hierarchical binomial non-centered** (Designer 2's Model 2)
   - More flexible than beta-binomial
   - Can extend to covariates
   - Standard hierarchical framework
   - Direct comparison with beta-binomial

3. **Test robustness with Student-t** (Designer 3's primary / Designer 2's Model 3)
   - Addresses 42% outlier rate explicitly
   - Posterior ν tells if robustness needed
   - Sensitivity analysis for outliers

4. **Exploratory: Horseshoe** (Designer 3's second)
   - Only if first 3 models show limitations
   - Tests sparsity hypothesis (most groups identical)
   - May win prediction if only 3-5 groups genuinely differ

**NOT prioritized:**
- Centered parameterization (Designer 2's Model 1): Known computational issues
- Mixture models: EDA shows no discrete clusters, N=12 too small
- Standard beta-binomial (Designer 1's Model A): Same as Model B, less interpretable

---

## Experiment 1: Beta-Binomial (Reparameterized)

### Model Specification
```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)  for i = 1, ..., 12

Level 2 (Group probabilities):
  p_i ~ Beta(a, b)

Reparameterization:
  a = μ * κ
  b = (1 - μ) * κ
  where μ = population mean, κ = concentration

Priors:
  μ ~ Beta(2, 18)         # Centered on 0.1, weak prior
  κ ~ Gamma(2, 0.1)       # Mean = 20, allows wide range

Generated quantities:
  φ = 1 + 1/κ             # Overdispersion parameter
  log_lik[i]              # For LOO-CV
  y_rep[i]                # Posterior predictive samples
```

### Prior Justification
- **μ ~ Beta(2, 18):** EDA pooled rate = 7.6%, prior centered on 10% with 95% interval [0.02, 0.28]
- **κ ~ Gamma(2, 0.1):** Wide prior (SD = 20), allows κ → 0 for extreme heterogeneity or κ → ∞ for homogeneity
- Implied ICC from prior: E[ICC] ≈ 0.5 (weakly informative given EDA ICC = 0.73)

### Expected Posterior (if model correct)
- μ ≈ 0.07-0.08 (near observed 7.6%)
- κ ≈ 0.3-5 (low concentration, high heterogeneity)
- φ ≈ 3.0-4.0 (matching observed overdispersion)
- Group 1: Shrink from 0% to 2-4%
- Group 8: Shrink from 14.4% to 11-13%

### Falsification Criteria - ABANDON if:
1. **Cannot reproduce overdispersion:** Posterior predictive φ_rep < 2.0 (observed ≈ 3.5)
2. **Prior-posterior conflict:** Posterior κ entirely outside prior support
3. **Poor predictive performance:** LOO Pareto k > 0.7 for >3 groups
4. **Fails posterior predictive checks:** p-value for variance < 0.01 or > 0.99
5. **Bimodality in group rates:** Posterior shows discrete clusters (contradicts EDA)

### Success Criteria - ACCEPT if:
- Rhat < 1.01, ESS > 400 for μ, κ
- Posterior φ ≈ 3.0-4.5 (reproduces overdispersion)
- LOO Pareto k < 0.7 for all groups
- Posterior predictive checks pass (0.05 < p < 0.95 for key statistics)
- Shrinkage patterns reasonable (Group 1 → 1-4%, Group 8 moderate shrinkage)

### Computational Plan
- **Software:** Stan via CmdStanPy (beta_binomial_lpmf available)
- **Sampling:** 4 chains, 2000 iterations (1000 warmup), adapt_delta = 0.95
- **Expected runtime:** 2-5 minutes
- **Diagnostics:** Trace plots, pairs plot for μ and κ, posterior predictive plots

### Implementation Files
- Stan model: `experiments/designer_1/stan_models/model_b_reparameterized.stan`
- Fitting script: `experiments/designer_1/scripts/fit_models.py`

---

## Experiment 2: Hierarchical Binomial (Non-Centered)

### Model Specification
```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)
  logit(p_i) = μ + σ * z_i

Level 2 (Non-centered parameterization):
  z_i ~ Normal(0, 1)  for i = 1, ..., 12

Priors:
  μ ~ Normal(-2.5, 1.0)     # logit(0.076) ≈ -2.5
  σ ~ Half-Normal(0, 1.0)   # Between-group SD

Derived quantities:
  α_i = σ * z_i             # Group-specific effects (centered form)
  φ = Var(p_i) / Var_binomial  # Overdispersion check
  log_lik[i]                # For LOO-CV
  y_rep[i]                  # Posterior predictive samples
```

### Prior Justification
- **μ ~ Normal(-2.5, 1.0):** Centered on logit(7.6%), 95% interval covers [1%, 35%] on probability scale
- **σ ~ Half-Normal(0, 1.0):** Weakly informative, allows σ → 0 (no heterogeneity) or σ → 2+ (high heterogeneity)
- EDA ICC = 0.73 suggests σ ≈ 0.8-1.2, which is well within prior support

### Expected Posterior (if model correct)
- μ ≈ -2.6 to -2.4 (6.5-8.5% on probability scale)
- σ ≈ 0.8-1.2 (consistent with ICC = 0.73)
- Group 1: α_1 ≈ -1.5, p_1 ≈ 0.02 (shrink from 0% to 2%)
- Group 8: α_8 ≈ +0.8, p_8 ≈ 0.12 (shrink from 14.4% to 12%)
- Average shrinkage ≈ 85% toward mean (consistent with EDA estimate)

### Falsification Criteria - ABANDON if:
1. **Divergences persist:** >5% divergences despite non-centered parameterization
2. **Cannot reproduce overdispersion:** Posterior φ_rep < 2.0
3. **Poor LOO performance:** ΔLOO vs beta-binomial > 10 (much worse)
4. **Posterior σ → 0:** No between-group variance (contradicts EDA)
5. **Group 1 posterior includes zero:** Shrinkage fails (should be ~1-3%)

### Success Criteria - ACCEPT if:
- Rhat < 1.01, ESS > 400 for μ, σ, all z_i
- Divergences < 1%
- Posterior σ ≈ 0.7-1.5 (consistent with ICC)
- LOO Pareto k < 0.7 for all groups
- Posterior predictive φ ≈ 3.0-4.5
- Competitive with or better than beta-binomial (|ΔLOO| < 4)

### Computational Plan
- **Software:** Stan via CmdStanPy
- **Sampling:** 4 chains, 2000 iterations (1000 warmup), adapt_delta = 0.95
- **Expected runtime:** 3-7 minutes (more parameters than beta-binomial)
- **Diagnostics:** Energy plots (check HMC geometry), pairs plot for μ and σ

### Implementation Files
- Stan model: `experiments/designer_2/model2_noncentered.stan`
- Fitting script: `experiments/designer_2/fit_models.py`

---

## Experiment 3: Student-t Hierarchical (Robust)

### Model Specification
```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)
  logit(p_i) = μ + σ * z_i

Level 2 (Robust random effects):
  z_i ~ Student-t(ν, 0, 1)  for i = 1, ..., 12

Priors:
  μ ~ Normal(-2.5, 1.0)
  σ ~ Half-Student-t(3, 0, 1)  # Robust prior for scale
  ν ~ Gamma(2, 0.1)            # Degrees of freedom (mean = 20)

Generated quantities:
  α_i = σ * z_i
  robustness_gain = compare tail probabilities vs Normal
  log_lik[i]
  y_rep[i]
```

### Prior Justification
- **μ, σ:** Same as Experiment 2
- **ν ~ Gamma(2, 0.1):** Mean = 20 (moderate tails), allows ν → ∞ (reduce to Normal) or ν → 3 (heavy tails)
- If ν → ∞, model tells us robustness is unnecessary (parsimony!)

### Expected Posterior (if robustness needed)
- μ ≈ -2.6 (similar to Normal)
- σ ≈ 0.9-1.3 (slightly higher than Normal due to outlier accommodation)
- **ν ≈ 8-15:** Moderate heavy tails (key indicator of robustness need)
- Group 8: Less shrinkage than Normal hierarchical (α_8 ≈ +1.0 vs +0.8)

### Falsification Criteria - ABANDON if:
1. **ν → ∞ (>50):** Robustness unnecessary, use Normal hierarchical instead
2. **No improvement in LOO:** Similar or worse than Normal despite extra parameter
3. **Posterior φ still mismatched:** Can't reproduce overdispersion even with robust model

### Success Criteria - ACCEPT if:
- Rhat < 1.01, ESS > 400
- **ν ≈ 5-20:** Clearly different from Normal (ν = ∞)
- LOO improvement over Normal ≥ 2*SE (if ν < 30)
- Group 8 less shrunk than in Normal model
- Posterior predictive checks better than Normal for outliers

### Computational Plan
- **Software:** Stan via CmdStanPy
- **Sampling:** 4 chains, 2000 iterations, adapt_delta = 0.95
- **Expected runtime:** 5-10 minutes (more complex posterior)
- **Diagnostics:** Posterior ν is key - if ν > 50, robustness not needed

### Implementation Files
- Stan model: `experiments/designer_3/student_t_hierarchical.stan` or `experiments/designer_2/model3_robust.stan`
- Fitting script: Available from both designers

---

## Experiment 4: Horseshoe Hierarchical (Exploratory)

### Model Specification
```
Level 1 (Likelihood):
  r_i ~ Binomial(n_i, p_i)
  logit(p_i) = μ + α_i

Level 2 (Horseshoe prior):
  α_i ~ Normal(0, τ * λ_i)  # Local-global shrinkage
  λ_i ~ Half-Cauchy(0, 1)   # Local shrinkage parameters
  τ ~ Half-Cauchy(0, 1)     # Global shrinkage parameter

Priors:
  μ ~ Normal(-2.5, 1.0)

Generated quantities:
  n_active = sum(λ_i > 0.5)  # Number of "active" groups
  log_lik[i]
  y_rep[i]
```

### Prior Justification
- **Horseshoe:** Provides adaptive shrinkage - outliers shrink less, typical groups shrink more
- Tests hypothesis that only 3-5 groups genuinely differ from mean
- EDA identified 5 outliers (Groups 1, 2, 5, 8, 11) - horseshoe should detect these

### Expected Posterior (if sparsity present)
- μ ≈ -2.6
- **n_active ≈ 3-5:** Only a few groups truly different
- Groups 2, 8, 11: Large λ (minimal shrinkage)
- Other groups: Small λ (heavy shrinkage toward mean)
- May win prediction if sparsity is real

### Falsification Criteria - ABANDON if:
1. **All λ ≈ 0.5:** No sparsity detected (all groups equally different)
2. **Poor LOO:** Worse than simpler models
3. **Unstable sampling:** High Rhat or low ESS due to horseshoe geometry

### Success Criteria - ACCEPT if:
- Clear bimodality in λ posterior (some large, some small)
- n_active ≈ 3-5 (consistent with outlier analysis)
- LOO improvement over hierarchical Normal ≥ 2*SE
- Better predictive performance

### Computational Plan
- **Software:** Stan via CmdStanPy
- **Sampling:** 4 chains, 2000 iterations, adapt_delta = 0.95
- **Expected runtime:** 7-12 minutes
- **Warning:** Only attempt if first 3 models show limitations
- **Risk:** N=12 may be too small for reliable sparsity detection

### Implementation Files
- Stan model: `experiments/designer_3/horseshoe_hierarchical.stan`
- Fitting script: `experiments/designer_3/fit_robust_models.py`

---

## Model Comparison Strategy

### Sequential Evaluation Plan

**Phase 1: Core Models (Required)**
1. Fit Experiment 1 (Beta-binomial)
2. Fit Experiment 2 (Hierarchical binomial non-centered)
3. Compare via LOO-CV

**Decision Point 1:**
- If |ΔLOO| < 2*SE: Both models equivalent, prefer simpler (beta-binomial)
- If one clearly better (ΔLOO > 4*SE): Use best model, proceed to validation
- If both fail validation: Fit Experiment 3 (Student-t)

**Phase 2: Robustness Check (Conditional)**
3. Fit Experiment 3 (Student-t) if:
   - Experiments 1 or 2 passed validation BUT
   - Posterior predictive p-values borderline (0.01-0.05 or 0.95-0.99)
   - Group 8 influential (Pareto k > 0.5)
   - Want sensitivity analysis

**Decision Point 2:**
- If Student-t shows ν > 50: Robustness unnecessary, use Normal
- If Student-t shows ν < 30 AND better LOO: Use Student-t
- If no improvement: Use best from Phase 1

**Phase 3: Exploratory (Optional)**
4. Fit Experiment 4 (Horseshoe) only if:
   - Interested in sparsity hypothesis
   - Previous models suggest only few groups differ
   - Have time/computational resources

### Comparison Metrics

**Primary: LOO-CV (Pareto-smoothed importance sampling)**
- Compute elpd_loo and SE for each model
- ΔLOO > 4 (2*SE) considered meaningful difference
- Prefer parsimony when models tied: fewer parameters wins

**Secondary: Posterior Predictive Checks**
- Does model reproduce φ ≈ 3.5-5.1?
- Does model reproduce range of success rates?
- Does model handle zero counts and outliers?

**Tertiary: Interpretability**
- Can we explain posterior to domain experts?
- Are parameters interpretable?
- Is the model extendable to new data?

### Comparison Table Template

After fitting models, populate this table:

| Model | Parameters | ELPD_LOO ± SE | ΔLOO ± SE | Pareto k > 0.7 | φ_rep (95% CI) | ν posterior (if applicable) |
|-------|-----------|---------------|-----------|----------------|----------------|---------------------------|
| Beta-binomial | 2 | — | — | — | — | — |
| Hierarchical (NC) | 14 | — | — | — | — | — |
| Student-t | 15 | — | — | — | — | — |
| Horseshoe | 26 | — | — | — | — | — |

---

## Minimum Attempt Policy

**Must attempt at least:**
- Experiment 1 (Beta-binomial)
- Experiment 2 (Hierarchical binomial non-centered)

**Exceptions:**
- If Experiment 1 fails pre-fit validation (prior predictive check), document and proceed to Experiment 2
- If both fail pre-fit validation, document and attempt Experiment 3

**After completing required experiments:**
- Proceed to Phase 4 (Model Assessment) regardless of number attempted
- Document reasons if fewer than 2 models attempted

---

## Global Decision Rules

### When to Stop Iteration
- ✅ One model passes all validation criteria
- ✅ Multiple models pass, clear LOO winner (ΔLOO > 4)
- ✅ Two models tied (|ΔLOO| < 2*SE), choose simpler
- ⚠️ All attempted models fail → Document limitations, try simpler Bayesian approach

### When to Pivot Model Classes
- All hierarchical models show ν → ∞ AND poor fit → Try non-hierarchical
- Consistent bimodality in posteriors → Consider mixture models seriously
- Temporal patterns discovered → Consider state-space models
- Covariates become available → Extend to hierarchical GLM

### When to Question Data
- All models fail to converge despite reparameterizations
- Group 8 consistently dominates LOO Pareto k (k > 0.7)
- Posterior predictive checks systematically fail
- Prior-posterior conflict across all reasonable priors

**Action:** Verify Group 8 data accuracy, consider excluding if measurement error suspected

---

## Validation Pipeline for Each Model

Every model must pass through this pipeline:

### Stage 1: Prior Predictive Check
**Question:** Do priors generate reasonable data?
- Sample from prior: μ, κ (or σ), generate y_rep
- Check: Do 95% of y_rep fall in [0, n_i]? (should be ~95%)
- Check: Does prior φ_rep span observed φ ≈ 3.5?
- **Fail condition:** Prior generates impossible values (y > n) >10% of time
- **Action if fail:** Adjust priors, document change

### Stage 2: Simulation-Based Validation
**Question:** Can model recover known parameters?
- Simulate data from known μ, κ (or σ)
- Fit model to simulated data
- Check: Are true parameters in 95% credible intervals?
- **Fail condition:** Coverage < 80% across multiple simulations
- **Action if fail:** Model misspecified or implementation error, investigate code

### Stage 3: Posterior Inference (Fitting)
**Question:** Does sampler converge?
- Fit to real data
- Check: Rhat < 1.01, ESS > 400, divergences < 1%
- **Fail condition:** Persistent convergence issues despite tuning
- **Action if fail:** Try reparameterization, increase adapt_delta, or skip model

### Stage 4: Posterior Predictive Check
**Question:** Does model reproduce observed patterns?
- Generate y_rep from posterior
- Check: φ_rep ≈ 3.0-4.5 (observed ≈ 3.5)
- Check: Range of success rates similar to observed
- Check: Can model occasionally generate zero counts?
- **Fail condition:** p-value < 0.01 or > 0.99 for key statistics
- **Action if fail:** Model inadequate, document and try next model

### Stage 5: LOO Cross-Validation
**Question:** How well does model predict?
- Compute LOO-CV
- Check: All Pareto k < 0.7
- Compare ΔLOO across models
- **Fail condition:** Multiple Pareto k > 0.7 (influential points)
- **Action if fail:** Model overly sensitive to outliers, consider robust version

---

## Success Criteria for Adequate Solution

The modeling effort reaches adequacy when:

1. **At least one model passes all validation stages**
2. **Posterior reproduces key data features** (overdispersion, range, outliers)
3. **Predictions are reasonable** (LOO Pareto k < 0.7 for all groups)
4. **Parameters are interpretable** (can explain to domain experts)
5. **Uncertainty is quantified** (credible intervals for all group rates)
6. **Handles special cases** (Group 1 zero count, Group 8 outlier)

**Quality bar:** Model need not be perfect, but must be:
- Scientifically defensible
- Computationally reliable
- Practically useful
- Honestly uncertain where appropriate

---

## Timeline and Resource Estimates

### Computational Resources
- **Per model fitting:** 2-10 minutes
- **Total expected runtime:** 30-60 minutes for core experiments
- **Disk space:** ~100 MB for Stan outputs
- **Memory:** 2-4 GB for Stan compilation and sampling

### Human Time
- **Prior predictive checks:** 30 minutes total
- **Model fitting and diagnostics:** 2 hours
- **Posterior predictive checks:** 1 hour
- **Model comparison and reporting:** 1 hour
- **Total:** ~4-5 hours for complete workflow

---

## Falsification Philosophy

This experiment plan is designed with falsification at its core:

1. **Every model has clear rejection criteria** before seeing results
2. **Success ≠ making any particular model work**
3. **Success = finding model that genuinely explains the data**
4. **Finding that a simpler model suffices** (e.g., ν → ∞) is **success**, not failure
5. **Reporting that all models fail** is honest science, not failure

### Key Falsification Tests

**Test 1: Is hierarchical structure necessary?**
- Falsify with: Posterior σ → 0 or κ → ∞
- Outcome: Use pooled model instead

**Test 2: Is robustness necessary?**
- Falsify with: Student-t posterior ν > 50
- Outcome: Use Normal hierarchical instead

**Test 3: Is sparsity present?**
- Falsify with: Horseshoe all λ ≈ 0.5
- Outcome: Continuous heterogeneity confirmed

**Test 4: Are mixture models needed?**
- Falsify with: No bimodality in group rate posteriors
- Outcome: Continuous variation confirmed (consistent with EDA)

---

## Contingency Plans

### If all models fail convergence:
1. Check Stan installation and version
2. Verify data format (r_i ≤ n_i for all i)
3. Try PyMC as alternative PPL
4. Consult Stan forums with minimal reproducible example

### If all models fail validation:
1. Re-examine data quality (especially Group 8)
2. Try simpler Bayesian models (pooled, unpooled)
3. Consider non-Bayesian methods for baseline comparison
4. Report honestly: "Data may require different model class"

### If computational resources exceeded:
1. Reduce iterations (minimum 1000 post-warmup)
2. Fit only Experiments 1 and 2 (skip 3 and 4)
3. Use variational inference as approximation (with caveats)

---

## Reporting Plan

After model evaluation, the final report will include:

1. **Model comparison table** (LOO, parameters, diagnostics)
2. **Selected model specification** with full mathematical detail
3. **Posterior summaries** (μ, σ or κ, group-specific rates)
4. **Visualizations:**
   - Caterpillar plot of group posteriors with 95% CIs
   - Posterior vs prior comparison
   - Posterior predictive check plots
   - Shrinkage plot (observed → posterior)
5. **Interpretation** of findings (which groups differ? by how much?)
6. **Limitations** (sample size, assumptions, data quality concerns)
7. **Recommendations** for next steps (collect more data? covariates needed?)

---

## Conclusion

This experiment plan provides:
- ✅ Clear model priorities (beta-binomial, hierarchical, robust)
- ✅ Explicit falsification criteria for each model
- ✅ Sequential decision points to avoid wasted effort
- ✅ Minimum attempt policy (2 models required)
- ✅ Contingency plans for failures
- ✅ Realistic timeline and resource estimates

**Next step:** Begin Phase 3 (Model Development) with Experiment 1 (Beta-binomial), following the validation pipeline for each model sequentially.

---

## Appendix: Designer Proposals

- **Designer 1 (Beta-binomial):** `experiments/designer_1/proposed_models.md`
- **Designer 2 (Hierarchical binomial):** `experiments/designer_2/proposed_models.md`
- **Designer 3 (Robust models):** `experiments/designer_3/proposed_models.md`

All Stan code and fitting scripts available in respective designer directories.
