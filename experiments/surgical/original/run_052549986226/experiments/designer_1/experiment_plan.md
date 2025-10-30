# Experiment Plan: Beta-Binomial and Mixture Models
## Designer 1 - Model Specifications

---

## Executive Summary

I propose **three Bayesian models** to explain the observed overdispersion in binomial trial data:

1. **Model A: Standard Beta-Binomial** (α, β parameterization)
2. **Model B: Reparameterized Beta-Binomial** (μ, κ parameterization)
3. **Model C: Two-Component Mixture** (adversarial falsification test)

**My prediction:** Model B wins due to interpretability. Model C loses because EDA shows no clustering. **But if I'm wrong and mixture wins**, I will abandon continuous variation models entirely.

---

## Problem Formulation: Competing Hypotheses

### Hypothesis 1: Continuous Variation (Models A & B)
**Claim:** Groups are drawn from a continuous distribution of success rates (beta distribution).

**Evidence FOR:**
- Success rates approximately normal (Shapiro-Wilk p=0.496)
- 84.8% of group pairs have overlapping CIs
- No visible clustering in caterpillar plots
- Beta-binomial had best AIC in preliminary analysis

**Evidence AGAINST:**
- If mixture model has ΔAIC < -10
- If posterior shows bimodal distribution of p_i
- If component separation is clear and interpretable

**I will abandon this hypothesis if:** Model C has Bayes Factor > 100 or ΔAIC < -10 vs Model B.

---

### Hypothesis 2: Discrete Subpopulations (Model C)
**Claim:** Groups belong to 2 distinct clusters (e.g., "low-rate" vs "high-rate").

**Evidence FOR:**
- Group 8 is extreme outlier (z=3.94)
- Could be "normal" groups (1-7, 9-10, 12) vs "exceptional" (2, 8, 11)

**Evidence AGAINST (stronger):**
- EDA shows continuous spread, not clusters
- No clear gap in success rate distribution
- Groups transition smoothly across range

**I will abandon this hypothesis if:**
- Posterior π → 0 or π → 1 (one component vanishes)
- Component means overlap substantially (>80% credible interval overlap)
- ΔAIC(C vs B) > 5

---

## Model Specifications

### Model A: Homogeneous Beta-Binomial

**Full Bayesian specification:**
```
Data likelihood:
  r_i | p_i, n_i ~ Binomial(n_i, p_i)    i = 1, ..., 12

Group-level variation:
  p_i | α, β ~ Beta(α, β)                [Common hyperparameters]

Hyperpriors:
  α ~ Gamma(2, 0.5)                      [E[α] = 4, allows [0.5, 15]]
  β ~ Gamma(2, 0.1)                      [E[β] = 20, allows [2, 60]]
```

**Why these priors?**
- Gamma(2, 0.5) for α: Prevents extreme concentration near zero, mean=4
- Gamma(2, 0.1) for β: Mean=20 favors low success rates (matching observed 7.6%)
- Weakly informative: Allow wide range of success rates [0.01, 0.50]
- Proper priors aid MCMC convergence

**Prior predictive range:**
- Mean success rate: 50% mass on [0.05, 0.30]
- Overdispersion φ: 50% mass on [1, 10] (covers observed φ=3.5)
- P(r=0 | n=47) ≈ 0.05-0.15 (zero counts plausible)

**Expected posterior behavior:**
- E[α] ≈ 2-6 (low shape → right-skewed beta)
- E[β] ≈ 20-50 (high shape → concentrates near zero)
- E[mean_p] ≈ 0.07-0.08 (close to 0.076)
- E[φ] ≈ 3.0-4.0 (matching observed)

**Falsification criteria - I will abandon if:**
1. Posterior shows bimodality in p_i distribution
2. Residuals systematically structured by sample size
3. α or β posteriors are extreme (< 0.5 or > 100)
4. Posterior predictive p-value for variance < 0.01 or > 0.99
5. LOO Pareto k > 0.7 for more than 3 groups
6. Model C has ΔAIC < -10

**Computational considerations:**
- Low complexity: Only 2 parameters
- Expected runtime: 30-60 seconds for 4 chains × 2000 iterations
- No divergences expected (simple geometry)
- Rhat < 1.01, ESS > 1000 expected

**Stan implementation:** `/workspace/experiments/designer_1/stan_models/model_a_beta_binomial.stan`

---

### Model B: Reparameterized Beta-Binomial

**Full Bayesian specification:**
```
Data likelihood:
  r_i | p_i, n_i ~ Binomial(n_i, p_i)

Group-level variation:
  p_i | μ, κ ~ Beta(μκ, (1-μ)κ)         [α = μκ, β = (1-μ)κ]

Hyperpriors:
  μ ~ Beta(2, 18)                        [Mean success rate, E[μ] = 0.1]
  κ ~ Gamma(2, 0.1)                      [Concentration, E[κ] = 20]
```

**Why these priors?**
- Beta(2, 18) for μ: Prior mean=0.1, close to observed 0.076, 95% CI=[0.01, 0.25]
- Gamma(2, 0.1) for κ: Same as Model A concentration (α+β)
- **WARNING:** If ICC=0.73 is real, κ should be ~0.37, but prior mean is 20
  - Strategy: Wide prior allows data to find true κ
  - Will check if posterior << prior (evidence for high heterogeneity)

**Interpretability advantage:**
- μ: Directly interpretable as population mean success rate
- κ: Higher κ → less between-group variation
- ICC ≈ 1/(1+κ): Directly links to intraclass correlation

**Expected posterior behavior:**
- E[μ] ≈ 0.07-0.08
- E[κ] ≈ 0.3-5 (MUCH lower than prior if ICC=0.73 is real!)
- **RED FLAG:** If κ posterior << prior mean, groups are more variable than assumed
- Posterior corr(μ, κ) expected ≈ -0.3 to 0.3 (if |corr| > 0.7, consider reparameterization)

**Falsification criteria:**
- Same as Model A (same likelihood, different parameterization)
- Additional: If posterior corr(μ, κ) > 0.95 → identifiability issues

**Computational considerations:**
- μ-κ geometry can be worse than α-β
- May need non-centered parameterization if divergences occur
- Increase adapt_delta to 0.95 preventatively

**Stan implementation:** `/workspace/experiments/designer_1/stan_models/model_b_reparameterized.stan`

---

### Model C: Two-Component Mixture (Falsification Test)

**Full Bayesian specification:**
```
Data likelihood:
  r_i | p_i, n_i ~ Binomial(n_i, p_i)

Mixture of beta distributions:
  p_i ~ π × Beta(α_1, β_1) + (1-π) × Beta(α_2, β_2)

Reparameterized:
  α_k = μ_k × κ_k,  β_k = (1-μ_k) × κ_k    for k = 1, 2

Hyperpriors:
  π ~ Beta(2, 2)                          [Mixing proportion, E[π] = 0.5]
  μ_1 ~ Beta(3, 50)                       [Low-rate mean, E[μ_1] ≈ 0.057]
  μ_2 ~ Beta(5, 20)                       [High-rate mean, E[μ_2] ≈ 0.20]
  κ_k ~ Gamma(2, 0.1)                     [Concentrations, same prior]

Constraint: μ_1 < μ_2 (ordered for identifiability)
```

**Why consider this model?**
This is an **adversarial test**. EDA evidence is AGAINST this model:
- No visible clustering
- Continuous distribution of rates
- 84.8% group overlap

**But I include it to:**
1. Test if my EDA interpretation was wrong
2. Quantify evidence for continuous vs discrete variation
3. Have a clear falsification criterion for beta-binomial

**I expect this model to LOSE.** If it wins, it means:
- Discrete subpopulations genuinely exist
- Need to explain what distinguishes clusters
- Should abandon continuous variation models

**Expected posterior if mixture is WRONG:**
- π → 0 or π → 1 (one component dominates)
- μ_1 and μ_2 overlap substantially
- Component assignments uncertain (all groups have prob ≈ π)
- ΔAIC(C vs B) > 5

**Expected posterior if mixture is CORRECT (surprising):**
- π ∈ [0.3, 0.7] with clear mode
- μ_1 ≈ 0.04-0.06, μ_2 ≈ 0.11-0.15 (clear separation)
- Component assignments:
  - Low: Groups 1, 3, 4, 5, 6, 7, 9, 12
  - High: Groups 2, 8, 11
- ΔAIC < -10

**Falsification criteria - I will abandon if:**
1. ΔAIC(C vs B) > 5 (mixture not justified)
2. Posterior π < 0.1 or π > 0.9 (one component vanishes)
3. Component means overlap >80% of credible intervals
4. No interpretable pattern in component membership

**Computational considerations:**
- **Known challenges:** Label switching, multimodality, slow mixing
- **Mitigations:**
  - Use `ordered[2] mu` constraint
  - 8 chains instead of 4
  - 3000 iterations instead of 2000
  - Check Rhat carefully
- Expected runtime: 3-5 minutes
- May see some divergences if components poorly separated

**Stan implementation:** `/workspace/experiments/designer_1/stan_models/model_c_mixture.stan`

---

## Model Comparison Strategy

### Quantitative Metrics

1. **LOO-CV (Leave-One-Out Cross-Validation):**
   - Lower LOO is better
   - ΔLOO > 10: Strong evidence against higher-LOO model
   - Check Pareto k < 0.7 for all observations

2. **WAIC (Widely Applicable Information Criterion):**
   - Similar to LOO, alternative metric
   - Cross-validate results

3. **Bayes Factors (if computable):**
   - BF > 100: Decisive evidence
   - BF 10-100: Strong evidence
   - BF 3-10: Moderate evidence

### Qualitative Assessment

4. **Posterior Predictive Checks:**
   - Does model reproduce observed variance?
   - Does model predict zero counts occasionally (Group 1)?
   - Does model handle outliers appropriately (Group 8)?
   - Calibration: rank histograms uniform?

5. **Scientific Plausibility:**
   - Are parameter estimates reasonable?
   - Do shrinkage patterns make sense?
   - Is there domain justification (if mixture wins)?

6. **Computational Health:**
   - Rhat < 1.01 for all parameters
   - ESS > 400 (preferably > 1000)
   - No divergent transitions
   - No problematic Pareto k values

### Decision Rules

**Choose Model A/B over C if:**
- ΔAIC(C vs A/B) > 5
- Mixture has identifiability issues
- No interpretable clustering

**Choose Model C over A/B if:**
- ΔAIC(C vs A/B) < -10
- Clear component separation
- Can explain WHY groups cluster

**Choose between A and B:**
- Doesn't matter (same model!)
- Prefer B for interpretability
- Prefer A if computational issues

---

## Red Flags and Major Pivot Points

### Evidence That Would Make Me Switch Model Classes Entirely

**1. Temporal structure discovered**
- If trials are ordered in time
- **→ Pivot to:** State-space model or GP with temporal correlation

**2. Covariate information emerges**
- If group-level predictors become available
- **→ Pivot to:** Hierarchical GLM with covariates

**3. Overdispersion beyond beta-binomial**
- If posterior predictive checks systematically under-predict variance
- **→ Pivot to:** Zero-inflated beta-binomial or beta-negative binomial

**4. Zero-inflation beyond Group 1**
- If multiple groups have zeros and model under-predicts
- **→ Pivot to:** Zero-inflated mixture model

**5. Spatial correlation revealed**
- If groups have geographic structure
- **→ Pivot to:** Spatial model with GP or CAR prior

**6. Non-binomial likelihood needed**
- If binomial assumption fails (e.g., trials not independent)
- **→ Pivot to:** Different likelihood (Poisson, negative binomial, etc.)

### Critical Decision Points

**Decision Point 1: After Fitting All Models**
- If all models fail same posterior predictive check → Problem is LIKELIHOOD, not hierarchy
- Action: Reconsider binomial assumption

**Decision Point 2: If Mixture Wins**
- If π ≈ 0.5 but assignments uncertain → Need more data or abandon mixture
- If π clear but no interpretation → False positive, stick with continuous

**Decision Point 3: If κ Posterior << 1**
- Groups TOO heterogeneous for beta-binomial
- Action: Consider group-specific parameters without partial pooling

**Decision Point 4: If Group 8 Still Extreme After Shrinkage**
- Posterior p_8 > 0.12 after shrinkage → Genuine outlier or data issue
- Action: Verify data quality, consider robust likelihood

### Stopping Rules

**When to stop and reconsider everything:**
1. 3+ models all fail same posterior predictive check
2. Shrinkage makes all groups nearly identical (data too sparse)
3. Cannot achieve convergence despite reparameterizations
4. Domain expert contradicts model implications

---

## Prior Sensitivity Analysis Plan

### Parameters to Test

**Model A/B:**
- **α (or μ) prior:** Try Gamma(1, 0.5), Gamma(2, 0.5), Gamma(5, 0.5)
- **β (or κ) prior:** Try Gamma(2, 0.05), Gamma(2, 0.1), Gamma(2, 0.5)

**Model C:**
- **π prior:** Try Beta(1, 1), Beta(2, 2), Beta(5, 5)
- Check: Does component assignment change?

### Robustness Criteria

**Conclusions should be stable:**
- Mean success rate: ±0.01 across priors
- Overdispersion φ: ±0.5 across priors
- Model ranking: Same model wins under all reasonable priors

**Warning signs:**
- Group 1 posterior changes from 2% to 6% → Need more data
- Mixture component probabilities flip → Not identified
- Conclusions reverse with different priors → Data too weak

---

## Implementation Plan

### Phase 1: Prior Predictive Checks (Verify Priors)
**Script:** `scripts/prior_predictive.py`

**Checks:**
1. Do priors allow observed mean ≈ 0.076?
2. Do priors allow observed φ ≈ 3.5?
3. Are prior predictions reasonable?
4. P(0.05 < mean_p < 0.10) > 0.2 under prior?

**Decision:** If priors too restrictive, widen them.

---

### Phase 2: Fit Models
**Script:** `scripts/fit_models.py`

**Workflow:**
```bash
# Fit all models
python scripts/fit_models.py --model all

# Or fit individually
python scripts/fit_models.py --model a
python scripts/fit_models.py --model b
python scripts/fit_models.py --model c
```

**Diagnostics to check:**
- Rhat < 1.01 ✓
- ESS > 400 ✓
- No divergences ✓
- Pareto k < 0.7 ✓

---

### Phase 3: Model Comparison
**Script:** `scripts/model_comparison.py` (to be created)

**Metrics:**
1. LOO-CV comparison
2. Posterior predictive checks
3. Shrinkage plots
4. Parameter summaries

**Decision:** Choose model with best LOO and passes diagnostic checks.

---

### Phase 4: Sensitivity Analysis
**Script:** `scripts/sensitivity_analysis.py` (to be created)

**Tests:**
1. Vary priors, refit
2. Exclude Group 8, refit
3. Use different random seeds
4. Check robustness

**Decision:** If conclusions stable → proceed. If unstable → report uncertainty.

---

## Expected Deliverables

### For Each Model:

1. **Convergence diagnostics table**
   - Rhat, ESS, divergences for all parameters

2. **Posterior summary table**
   - Mean, SD, 95% CI for hyperparameters
   - Group-specific shrinkage estimates

3. **Diagnostic plots**
   - Trace plots (check mixing)
   - Prior vs posterior (did data update?)
   - Pairs plots (check correlations)

4. **Posterior predictive checks**
   - Observed vs predicted success rates
   - Variance reproduction
   - Calibration plots

5. **Shrinkage visualization**
   - MLE vs posterior mean for each group
   - Shows Group 1 and 8 shrinkage

6. **LOO-CV results**
   - Pareto k diagnostics
   - Predictive accuracy

### Model Comparison Report:

1. **LOO/WAIC comparison table**
2. **Model recommendation** with strength of evidence
3. **Falsification assessment:** Did any models fail critical tests?
4. **Sensitivity results:** Are conclusions robust?

---

## File Structure

```
/workspace/experiments/designer_1/
├── README.md                          # This file
├── experiment_plan.md                 # Detailed plan
├── proposed_models.md                 # Full model specifications
├── stan_models/
│   ├── model_a_beta_binomial.stan
│   ├── model_b_reparameterized.stan
│   └── model_c_mixture.stan
├── scripts/
│   ├── fit_models.py                  # Main fitting script
│   ├── prior_predictive.py            # Prior checks
│   ├── posterior_analysis.py          # (To create)
│   └── model_comparison.py            # (To create)
└── results/
    ├── model_a/                       # Model A outputs
    ├── model_b/                       # Model B outputs
    ├── model_c/                       # Model C outputs
    ├── prior_predictive_plots/        # Prior visualizations
    └── comparison/                    # Model comparison results
```

---

## Timeline and Computational Budget

**Estimated runtime:**
- Prior predictive checks: ~2 minutes
- Model A fitting: ~1 minute
- Model B fitting: ~1 minute
- Model C fitting: ~3-5 minutes
- Posterior analysis: ~5 minutes
- **Total: ~15-20 minutes**

**Computational resources:**
- CPU-only (no GPU needed)
- ~8GB RAM sufficient
- Standard laptop adequate

---

## Success Criteria

**This experiment succeeds if:**
1. ✅ All models converge (Rhat < 1.01)
2. ✅ One model clearly preferred by LOO-CV
3. ✅ Chosen model passes posterior predictive checks
4. ✅ Conclusions robust to prior changes
5. ✅ Can answer: "Is variation continuous or discrete?"

**This experiment succeeds EVEN IF:**
- Mixture model wins (falsification is success!)
- All models fail (tells us to rethink likelihood)
- Results contradict EDA (data update beliefs)

**The goal is TRUTH, not defending beta-binomial.**

---

## Key Predictions

**My predictions (to be verified):**
1. Model B wins on interpretability (same likelihood as A)
2. Model C loses decisively (ΔAIC > 10)
3. Posterior κ << 20 (groups more variable than prior)
4. Group 1 shrinks to ≈ 2-4%
5. Group 8 shrinks to ≈ 11-13%
6. Overdispersion φ ≈ 3.0-3.5

**If I'm wrong about:**
- Mixture winning → Need to explain clustering
- All models failing → Rethink entire approach
- No shrinkage → Data too sparse for hierarchical model

---

## Final Notes

This experiment plan is designed with **intellectual honesty** as the core principle:
- Competing hypotheses stated upfront
- Falsification criteria explicit
- Escape routes planned
- Ready to abandon approach if data demand it

**Success = Finding the right model**, not completing a predetermined plan.

---

**Ready to proceed:** All Stan models written, scripts prepared, priors justified.

**Next step:** Run prior predictive checks to verify priors are sensible.
