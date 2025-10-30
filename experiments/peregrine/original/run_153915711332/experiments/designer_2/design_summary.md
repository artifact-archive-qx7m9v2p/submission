# Model Design Summary - Designer 2
## Quick Reference for Implementation

**Date:** 2025-10-29
**Full Details:** See `/workspace/experiments/designer_2/proposed_models.md`

---

## Three Competing Hypotheses

### 1. State-Space Model (PRIORITY 1 - ⭐⭐⭐)
**Hypothesis:** Massive autocorrelation (0.989) indicates latent random walk with drift, not cross-sectional GLM.

**Key Idea:** Decompose variance into:
- Systematic drift (δ ≈ 0.05-0.07 per period in log-space)
- Small random innovations (σ²_η ≈ 0.05-0.1)
- Observation overdispersion (φ ≈ 10-20, much less than naive 70)

**Abandon if:**
- Innovation variance σ²_η ≈ total variance (state decomposition pointless)
- Residual ACF > 0.5 after accounting for state evolution
- One-step-ahead predictions are poor (coverage < 75%)

### 2. Changepoint Model (PRIORITY 2 - ⭐⭐)
**Hypothesis:** CUSUM minimum at year ≈ 0.3 reflects discrete regime shift (4.5x mean jump).

**Key Idea:** Piecewise linear growth with different slopes and levels before/after breakpoint.

**Abandon if:**
- Changepoint location τ has diffuse posterior (SD > 0.5)
- Level shift β_2 and slope change β_3 both include zero
- Smooth transition model fits better (ΔLOO > 10)

### 3. Gaussian Process Model (PRIORITY 2 - ⭐⭐)
**Hypothesis:** Neither parametric form is correct; growth is smooth but complex.

**Key Idea:** Flexible nonparametric function with automatic smoothness selection via lengthscale ℓ.

**Abandon if:**
- Lengthscale suggests nearly linear function (ℓ > 2.0)
- Simple parametric model fits equivalently (ΔLOO < 5)
- GP shows discontinuities (suggests changepoint better)

---

## Critical Design Decisions

### All Models Use Negative Binomial
**Justification:** Extreme overdispersion (Var/Mean = 68) rules out Poisson.

**Parameterization:** NegativeBinomial(μ, φ) where:
- E[C] = μ
- Var[C] = μ + μ²/φ
- Smaller φ → more overdispersion

**Expected φ values:**
- State-Space: 10-20 (less than naive because temporal correlation explains variance)
- Changepoint: 10-25
- GP: 8-20

### All Models Must Include LOO-CV
**Implementation:** Every Stan model must have `generated quantities` block with:
```stan
generated quantities {
  vector[N] log_lik;      // For LOO computation
  array[N] int C_rep;     // For posterior predictive checks

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | mu[t], phi);
    C_rep[t] = neg_binomial_2_log_rng(mu[t], phi);
  }
}
```

### Non-Centered Parameterizations Essential
**Why:** High autocorrelation creates posterior correlations that slow MCMC mixing.

**How:**
- State-Space: `η_t = η_{t-1} + drift + σ_η * z_t` where `z_t ~ N(0,1)`
- GP: `f = mean_function + chol(K) * z` where `z ~ N(0,I)`

---

## Falsification Strategy

### Primary Decision Rule: LOO-ELPD
- **Strong preference:** ΔLOO > 10 (model clearly better)
- **Equivalent:** ΔLOO < 4 (within standard error)
- **If all equivalent:** Use Bayesian Model Averaging, report uncertainty

### Secondary Validation (ALL must pass):
1. **Residual ACF(1) < 0.3** - Temporal structure captured
2. **80% prediction interval coverage: 75-85%** - Calibrated uncertainty
3. **R-hat < 1.01, ESS > 400** - Converged chains
4. **Posterior predictive checks p-values: 0.05-0.95** - Model captures data features

### Red Flags (STOP if any occur):
- φ → 0: Overdispersion was illusory (contradicts EDA)
- φ → ∞: Distribution family wrong
- Prior-posterior overlap > 80%: Data not informative
- All models have ACF > 0.6: None capture temporal structure

---

## Implementation Roadmap

### Phase 1: Basic Fitting (Day 1)
1. Implement all three Stan models (skeletons provided in main document)
2. Fit with default settings (2000 warmup + 2000 sampling, 4 chains)
3. Check convergence (R-hat, ESS, traceplots)
4. Compute LOO-ELPD for all models

**Expected time:** 3-5 hours

### Phase 2: Diagnostics (Day 1-2)
1. Residual ACF analysis
2. One-step-ahead predictions
3. Posterior predictive checks (max, min, mean, variance, ACF of replicates)
4. Prior sensitivity analysis (weak vs. strong priors)

**Expected time:** 2-4 hours

### Phase 3: Model Variants (Day 2)
**Only if needed based on Phase 2 results:**

- State-Space: Try Variant 1a (no time effect), 1b (stochastic drift)
- Changepoint: Try Variant 2a (smooth transition), 2b (variance changepoint)
- GP: Try Variant 3a (zero mean), 3b (additive parametric + GP)

**Expected time:** 2-3 hours per variant

### Phase 4: Stress Tests (Day 2-3)
1. **Extrapolation:** Fit on year < 1.0, predict year ≥ 1.0
2. **Jackknife:** Remove observations near changepoint, refit
3. **Simulation:** Generate data under known mechanisms, test recovery

**Expected time:** 3-4 hours

### Phase 5: Reporting (Day 3)
1. Model comparison table with LOO-ELPD, diagnostics
2. Parameter posterior summaries (mean, SD, 95% CI)
3. Visualizations (data + fitted trends, residuals, predictions)
4. Scientific interpretation of winning model
5. Limitations and uncertainty quantification

**Expected time:** 2-3 hours

---

## What Success Looks Like

### Best Case Scenario
**State-Space model wins with:**
- LOO-ELPD ≈ -150, ΔLOO = 12 over others
- δ ≈ 0.06 [0.04, 0.08], σ_η ≈ 0.08 [0.05, 0.12], φ ≈ 15 [10, 22]
- Residual ACF(1) = 0.18
- One-step-ahead coverage: 82% (80% interval), 94% (95% interval)
- All R-hat < 1.005, all ESS > 1000

**Interpretation:** Data generated by smooth random walk with positive drift (~6% per period). Most apparent "overdispersion" is actually temporal correlation. Process is predictable in short term but uncertain in long term.

### Acceptable Scenario
**Models within LOO-ELPD 4 of each other:**
- All show good diagnostics (ACF < 0.3, coverage 75-85%)
- Different mechanistic interpretations but similar predictions
- Report: "Data consistent with multiple mechanisms, use BMA for prediction"

**Interpretation:** n=40 insufficient to distinguish state-space from changepoint from smooth trend. Emphasize predictive performance over mechanism.

### Failure Scenario (Pivot Required)
**All models show:**
- ACF > 0.6 in residuals
- Coverage < 70%
- LOO-ELPD < -180

**Interpretation:** Our model classes fundamentally wrong. Possible causes:
- Count distribution is not Negative Binomial (try Conway-Maxwell-Poisson)
- Temporal structure is more complex (ARMA, not just AR)
- Unobserved covariates driving dynamics

**Action:** Return to EDA, reconsider everything, possibly admit defeat.

---

## Key Insights for Modeler

### Why These Models?
The EDA shows two competing patterns:
1. **Extreme smoothness:** ACF(1) = 0.989 suggests C_t ≈ C_{t-1}
2. **Possible discontinuity:** CUSUM suggests break at year ≈ 0.3

Models 1 and 2 test these directly. Model 3 is the "hedge" - if both are wrong, GP should reveal the truth.

### Expected Computational Challenges
1. **State-Space:** Posterior correlations among η_t values slow mixing
   - **Fix:** Non-centered parameterization, adapt_delta = 0.95
2. **Changepoint:** Discrete indicator function can cause divergences
   - **Fix:** adapt_delta = 0.95, max_treedepth = 12
3. **GP:** Matrix inversions can be numerically unstable
   - **Fix:** Add jitter (1e-9) to diagonal, use Cholesky decomposition

### What I'm Most Uncertain About
1. **Is φ really ~15, or ~5, or ~30?** Overdispersion is confounded with temporal structure.
2. **Is the changepoint real or illusory?** Could be smooth acceleration appearing discrete.
3. **Will n=40 be enough?** GP with 2-3 hyperparameters may be data-hungry.

### What Would Genuinely Surprise Me
1. **If φ < 5:** Would suggest overdispersion is mostly from misspecified trend
2. **If σ²_η > 0.5:** Would suggest state evolution is very noisy (more than I expect)
3. **If all models fail diagnostics:** Would indicate data generation process is stranger than I thought

---

## Deliverables Checklist

### Code Files
- [ ] `model_1_state_space.stan`
- [ ] `model_2_changepoint.stan`
- [ ] `model_3_gaussian_process.stan`
- [ ] `fit_models.py` (or `.R`) - Stan interface script
- [ ] `diagnostics.py` - Residual checks, LOO, PPCs

### Output Files
- [ ] `loo_comparison.csv` - Model comparison table
- [ ] `posterior_summaries.csv` - Parameter estimates
- [ ] `model_1_diagnostics.png` - Traceplots, ACF, posterior predictive
- [ ] `model_2_diagnostics.png`
- [ ] `model_3_diagnostics.png`
- [ ] `predictions.png` - Data + fitted trends + intervals

### Report
- [ ] `modeling_report.md` - Full analysis with interpretation
- [ ] Executive summary: Which model won and why
- [ ] Parameter interpretation (scientific, not just statistical)
- [ ] Limitations and caveats
- [ ] Recommendations for future work

---

## Questions for Main Agent

1. **Is there domain context for this data?** (e.g., biology, economics, engineering)
   - Would inform prior selection and interpretation
   - Might suggest mechanistic models beyond these three

2. **What is the primary goal?** Prediction vs. mechanistic understanding
   - If prediction: Focus on LOO-ELPD, may use BMA
   - If mechanism: Need stronger evidence to prefer one model

3. **Are there other covariates available?**
   - Could explain some of the temporal correlation
   - Would change model structure substantially

4. **What happens after this analysis?**
   - If this is exploratory: Can be more aggressive with model selection
   - If this informs decisions: Need to be very conservative

---

## Final Note: Epistemic Humility

These models represent my best hypotheses given the EDA, but I fully expect:
- At least one model to fail diagnostics
- The truth to be more complex than any single model captures
- Surprises that challenge my assumptions

**I will prioritize:**
1. Honest uncertainty quantification over point estimates
2. Diagnostic failures over task completion
3. Admitting "I don't know" over forcing a conclusion

The goal is learning what's true about this data, not checking boxes.

---

**Prepared by:** Designer 2 (Temporal Structure Specialist)
**Full Documentation:** `/workspace/experiments/designer_2/proposed_models.md`
**Ready for Implementation:** Yes
