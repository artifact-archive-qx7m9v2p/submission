# Hierarchical & Random Effects Model Proposals
**Designer 1 - Independent Analysis**
**Date:** 2025-10-28
**Focus:** Hierarchical structures, centering strategies, prior sensitivity

---

## Executive Summary

The EDA reveals a critical challenge for hierarchical modeling: **strong evidence of homogeneity** (tau^2=0, I^2=0%, Q p=0.696) despite appropriate hierarchical structure. This creates a boundary estimation problem where the between-school variance may be at or near zero. I propose three competing hierarchical model classes that differ fundamentally in:

1. **Parameterization** (centered vs. non-centered vs. adaptive)
2. **Prior philosophy** (weakly informative vs. skeptical vs. adaptive)
3. **Failure modes** (geometry issues vs. boundary problems vs. overfitting)

**Key Insight:** Models that work well when tau is large may fail catastrophically when tau ≈ 0. I explicitly design for this challenging regime.

---

## Model Class 1: Non-Centered Hierarchical (Standard Approach)

### Mathematical Specification

```
# Data model
y_i ~ Normal(theta_i, sigma_i)    where sigma_i are KNOWN

# Hierarchical structure (non-centered parameterization)
theta_i = mu + tau * eta_i
eta_i ~ Normal(0, 1)              [standardized random effects]

# Priors
mu ~ Normal(0, 20)                [weakly informative on grand mean]
tau ~ Half-Cauchy(0, 10)          [Gelman (2006) recommendation]
```

### Why This Parameterization?

**Non-centered is standard for hierarchical models** because:
- Decorrelates mu and tau in the posterior (better sampling geometry)
- Prevents funnel-shaped posterior when tau is small
- Allows efficient exploration when tau ≈ 0

**Critical for this dataset:** With tau expected near zero, centered parameterization would create severe correlation between mu and theta_i, making sampling difficult.

### Prior Justifications

**Half-Cauchy(0, 10) for tau:**
- Heavy tails allow large tau if data support it
- Scale=10 is reasonable given observed effect range (-3 to 28, span=31)
- Does NOT force tau toward zero (unlike half-normal)
- EDA shows observed SD=10.4, so scale=10 accommodates this

**Normal(0, 20) for mu:**
- Weakly informative: SD=20 covers [-40, 40] at 2 SD
- Observed range [-3, 28] well within prior support
- Centered at 0 (no directional bias)
- Data are informative (pooled SE=4.07), so prior has modest influence

### Expected Behavior Given EDA

**Posterior predictions:**
- **mu:** Concentrate around 7.7 (pooled estimate), SD ≈ 4-5
- **tau:** Concentrated near 0, but with substantial right tail due to Cauchy
  - Median likely 2-5
  - 95% CI likely [0, 15] or wider
  - Heavy uncertainty in small-sample regime
- **theta_i:** Strong shrinkage (70-90%) toward mu
  - School 1 (y=28): shrink from 28 → ~10-12
  - School 5 (y=-1): shrink from -1 → ~5-7
  - High-precision schools (5,7) shrink less than low-precision schools (1,3,8)

**Shrinkage pattern:**
```
Shrinkage_i = 1 - [tau^2 / (tau^2 + sigma_i^2)]
```
With tau ≈ 3 and sigma_i ≈ 9-18, expect 60-90% shrinkage.

### Falsification Criteria

**I will ABANDON this model if:**

1. **Computational failure:**
   - Divergent transitions > 1% despite tuning
   - R-hat > 1.01 for any parameter
   - ESS < 100 for tau (indicates sampling problems)
   - **Why this matters:** Non-centered should be robust; failure indicates model misspecification

2. **Prior-posterior conflict for tau:**
   - Posterior median tau > 15 (exceeds observed SD=10.4)
   - Posterior entirely concentrated at tau=0 (model degenerate)
   - **Why this matters:** If tau >> observed SD, data don't support hierarchical structure; if tau ≡ 0, reduce to simpler model

3. **Shrinkage inconsistencies:**
   - High-precision schools (sigma=9-10) shrink MORE than low-precision schools (sigma=15-18)
   - **Why this matters:** Violates fundamental shrinkage logic; indicates model failure

4. **Posterior predictive failure:**
   - Less than 90% of observed y_i fall within their posterior predictive 95% intervals
   - **Why this matters:** Model should easily capture observed data given large sigma_i

5. **Extreme parameter values:**
   - Any theta_i posterior mean outside [-10, 20] (beyond plausible range)
   - **Why this matters:** Schools should shrink toward pooled mean, not away from it

### Stress Test: "Break the Shrinkage"

**Test design:** Compute posterior for School 1 (y=28, sigma=15) and School 5 (y=-1, sigma=9).
- **Expected:** School 1 shrinks more (lower precision)
- **Failure mode:** If School 5 shrinks more, model is broken

**Diagnostic:** Plot shrinkage vs. precision. Should be monotonic increasing.

### Model Weaknesses

1. **Half-Cauchy may be too dispersed:** With n=8, limited information to constrain tau. Heavy tails might allow implausibly large tau.

2. **No protection against boundary:** If true tau=0, model wastes parameters on theta_i that reduce to mu.

3. **Ignores measurement error heterogeneity:** Treats sigma_i as fixed. If there's uncertainty in sigma_i (not stated), model underestimates uncertainty.

4. **Assumes exchangeability:** If schools have meaningful structure (e.g., public vs. private), pooling may be inappropriate.

---

## Model Class 2: Skeptical Hierarchical (Boundary-Aware)

### Mathematical Specification

```
# Data model
y_i ~ Normal(theta_i, sigma_i)

# Hierarchical structure (non-centered)
theta_i = mu + tau * eta_i
eta_i ~ Normal(0, 1)

# Priors
mu ~ Normal(8, 5)                 [informative, based on pooled estimate]
tau ~ Half-Normal(0, 5)           [skeptical of large heterogeneity]
```

### Why This Approach?

**Skeptical prior philosophy:** EDA provides STRONG evidence for homogeneity (Q p=0.696, I^2=0%). This model takes that evidence seriously:

- **Half-Normal(0, 5) for tau:** More mass near zero than Half-Cauchy
  - P(tau < 5) ≈ 68% (vs. 50% for Cauchy)
  - Lighter tails: less support for extreme tau
  - **Reflects EDA:** Observed variance ratio = 0.66 suggests tau ≈ 0

- **Normal(8, 5) for mu:** Moderately informative
  - Centered on pooled estimate (7.69 ≈ 8)
  - SD=5 is tighter than Model 1 (SD=20), but still allows [-2, 18] at 2SD
  - **Reflects EDA:** Pooled estimate is 7.69 ± 4.07

### Key Difference from Model 1

**Model 1 (agnostic):** "I don't know if there's heterogeneity; let the data decide with minimal prior constraint"

**Model 2 (skeptical):** "Strong frequentist evidence suggests homogeneity; Bayesian prior should reflect this while remaining open to evidence"

### Expected Behavior

**Posterior predictions:**
- **mu:** Tighter around 7.7, SD ≈ 3-4 (prior pulls slightly)
- **tau:** More concentrated near 0
  - Median likely 1-3 (lower than Model 1)
  - 95% CI likely [0, 8] (narrower than Model 1)
  - Less posterior uncertainty
- **theta_i:** STRONGER shrinkage (80-95%) toward mu
  - School 1: shrink from 28 → ~8-10 (more than Model 1)
  - School 5: shrink from -1 → ~6-8 (more than Model 1)

### Falsification Criteria

**I will ABANDON this model if:**

1. **Prior dominates data:**
   - Posterior mean for mu differs from prior mean by < 1 SD
   - **Test:** Compute KL(posterior || prior) for mu. If KL < 0.1, prior too strong.
   - **Why this matters:** Prior should guide, not dictate

2. **Tau posterior truncated by prior:**
   - Posterior 95% CI for tau hits prior upper tail (>10)
   - Indicates prior is constraining data-supported values
   - **Why this matters:** Data should be able to overwhelm skeptical prior if heterogeneity exists

3. **Predictive failure for extreme schools:**
   - School 1 (y=28) or School 3 (y=-3) outside posterior predictive 95% interval
   - **Why this matters:** Over-shrinkage makes model unable to accommodate observed variation

4. **Model comparison failure:**
   - LOO-CV / WAIC substantially worse than Model 1 (diff > 5)
   - **Why this matters:** Skeptical prior should improve fit if EDA evidence is correct, or at least not harm it

5. **Posterior width narrower than sampling uncertainty:**
   - If posterior SD(theta_i) < 4 (less than pooled SE)
   - **Why this matters:** Cannot be more certain than pooled estimate given sampling error

### Stress Test: "Prior Sensitivity"

**Test design:** Compare posteriors under:
- Half-Normal(0, 5) [proposed]
- Half-Normal(0, 2) [very skeptical]
- Half-Normal(0, 10) [less skeptical]

**Expected:** Posterior should be relatively stable (data informative).
**Failure mode:** If posterior changes dramatically, prior is dominating (bad sign).

### Model Weaknesses

1. **Prior too strong:** With n=8, skeptical prior might overwhelm data, especially for tau.

2. **Asymmetry:** Half-Normal has lighter right tail than Half-Cauchy. If true heterogeneity exists (contra EDA), model will underestimate.

3. **Informal on mu:** Prior centered at 8 is data-driven (from EDA). Philosophically questionable to use data to set prior, then evaluate model on same data.

4. **Not robust to EDA error:** If Q test is misleading (low power), skeptical prior compounds the error.

---

## Model Class 3: Adaptive Non-Centered (Geometry Optimizer)

### Mathematical Specification

```
# Data model
y_i ~ Normal(theta_i, sigma_i)

# Adaptive parameterization
theta_i = mu + tau * eta_i                    if tau > threshold
        = mu + sigma_pooled * eta_i * (tau/threshold)   if tau ≤ threshold

eta_i ~ Normal(0, 1)

# Priors
mu ~ Normal(0, 20)
tau ~ Half-Cauchy(0, 10)

# Adaptive parameter
threshold = 3  [switch parameterization when tau < 3]
sigma_pooled = 4  [approximate pooled SE from EDA]
```

### Why This Approach?

**Problem:** When tau → 0, even non-centered parameterization can struggle:
- eta_i parameters become weakly identified
- Posterior explores irrelevant regions of parameter space
- Computation wastes time on near-zero tau * eta_i terms

**Solution:** Adaptively reparameterize based on tau magnitude:
- **When tau is large (>3):** Standard non-centered (optimal geometry)
- **When tau is small (<3):** Scale by sigma_pooled to maintain geometric properties

**Analogy:** Automatic differentiation adapts step size; this adapts parameterization.

### Implementation in PyMC

```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfCauchy('tau', beta=10)
    eta = pm.Normal('eta', mu=0, sigma=1, shape=8)

    # Adaptive parameterization
    threshold = 3.0
    sigma_pooled = 4.0

    # Smooth transition using sigmoid
    weight = pm.math.sigmoid((tau - threshold) * 2)  # smooth switch

    # Adaptive theta
    theta_std = weight * tau + (1 - weight) * sigma_pooled * (tau / threshold)
    theta = pm.Deterministic('theta', mu + theta_std * eta)

    # Likelihood
    y = pm.Normal('y', mu=theta, sigma=sigma_obs, observed=y_obs)
```

### Expected Behavior

**When tau posterior is near 0 (<3):**
- eta_i parameters remain well-identified (scaled by sigma_pooled ≈ 4)
- Avoids geometric pathologies
- Faster sampling, better ESS

**When tau posterior is large (>3):**
- Reverts to standard non-centered
- No loss of fidelity

**Posterior predictions:**
- Similar to Model 1 for mu and theta_i
- Potentially tighter posterior for tau (better sampling efficiency)
- Computational advantage: 2-3x higher ESS for tau

### Falsification Criteria

**I will ABANDON this model if:**

1. **No computational advantage:**
   - ESS for tau not significantly higher than Model 1 (< 10% improvement)
   - Sampling time not reduced (< 10% faster)
   - **Why this matters:** Complexity only justified if it improves inference

2. **Discontinuity artifacts:**
   - Bimodal posterior for tau around threshold=3
   - Sharp changes in theta_i posteriors at tau=3
   - **Why this matters:** Smooth transition should prevent discontinuities

3. **Threshold sensitivity:**
   - Results differ by >10% when threshold changes from 2 to 4
   - **Why this matters:** Model should be robust to hyperparameter choice

4. **Same failures as Model 1:**
   - Divergent transitions, R-hat > 1.01, etc.
   - **Why this matters:** If no better than Model 1, use simpler model

5. **Misleading posterior:**
   - Artificially tight posterior for tau due to reparameterization bias
   - **Test:** Compare to Model 1 posterior; should be similar or wider, not narrower
   - **Why this matters:** Cannot manufacture certainty through reparameterization

### Stress Test: "Threshold Grid Search"

**Test design:** Run model with threshold ∈ {1, 2, 3, 5, 10}.
- **Expected:** Results stable for threshold ∈ [2, 5]
- **Failure mode:** Large changes indicate threshold sensitivity (model unreliable)

**Diagnostic:** Plot tau posterior density for each threshold. Should overlay closely.

### Model Weaknesses

1. **Complexity:** Added hyperparameters (threshold, sigma_pooled) increase model complexity without strong theoretical justification.

2. **Arbitrary threshold:** Choosing threshold=3 is somewhat arbitrary (based on EDA pooled SE ≈ 4, but why 3 not 2 or 5?).

3. **Untested approach:** Less standard than Models 1-2; higher risk of unexpected behavior.

4. **Potential bias:** Reparameterization might inadvertently bias posterior (need careful validation).

5. **Overkill for simple problem:** With n=8 and simple structure, standard methods should suffice.

---

## Cross-Model Comparisons

### Decision Points for Model Selection

**Stage 1: Computational Feasibility (before looking at posteriors)**
- If Model 1 has divergences or ESS < 100 for tau → Try Model 3
- If Model 3 has no advantage over Model 1 → Drop Model 3, keep Model 1
- If Model 2 has prior-data conflict (KL < 0.1) → Drop Model 2

**Stage 2: Posterior Plausibility (after examining posteriors)**
- If any model has theta_i outside [-10, 20] → Reject that model
- If tau posterior median > 10 in any model → Question hierarchical structure
- If Models 1 & 2 disagree on tau by >2x → Prior sensitivity is critical issue

**Stage 3: Predictive Performance (after PPC)**
- If any model fails PPC (coverage < 90%) → Reject that model
- If Model 2 LOO >> Model 1 → Skeptical prior is appropriate
- If Model 1 LOO >> Model 2 → Skeptical prior too strong

### Red Flags Triggering Strategy Pivot

**Major red flags that would make me reconsider hierarchical modeling entirely:**

1. **Posterior predictive failure across all models:**
   - If ALL three models fail to capture observed data adequately
   - **Pivot to:** Robust models (t-distributed errors), mixture models, or examine data quality

2. **Extreme shrinkage inconsistencies:**
   - If high-precision schools consistently shrink more than low-precision schools in all models
   - **Pivot to:** No-pooling models, or models with school-specific structure

3. **Tau posterior firmly at boundary (tau < 0.1) in all models:**
   - **Pivot to:** Complete pooling model (fixed effects), abandon hierarchical structure

4. **Computational failure in all three models:**
   - Divergences, low ESS, high R-hat despite extensive tuning
   - **Pivot to:** Alternative frameworks (maximum likelihood, empirical Bayes, frequentist meta-analysis)

5. **Strong outliers emerge in residual analysis:**
   - If PPC reveals School 1 or School 3 are true outliers (not captured by hierarchical variation)
   - **Pivot to:** Outlier-robust models, or exclude outliers and refit

6. **Between-model variance exceeds within-model variance:**
   - If Models 1-3 give wildly different answers (e.g., tau ∈ [1,3] vs [8,15])
   - **Indicates:** Model uncertainty dominates; need different approach or accept uncertainty

### Alternative Models if All Three Fail

**If hierarchical models prove inadequate:**

**Option A: Complete Pooling (Fixed Effect)**
```
y_i ~ Normal(mu, sigma_i)
mu ~ Normal(0, 20)
```
Simplest model; justified if tau ≈ 0 confirmed.

**Option B: Robust Hierarchical (t-distributed)**
```
y_i ~ StudentT(nu, theta_i, sigma_i)
theta_i = mu + tau * eta_i
nu ~ Gamma(2, 0.1)  [degrees of freedom]
```
Protects against outliers (School 1?).

**Option C: Regression on Precision**
```
theta_i = mu + beta * (1/sigma_i^2)
```
Tests if high-precision schools differ systematically.

**Option D: Non-exchangeable Structure**
```
theta_i = mu + X_i * beta  [school covariates]
```
If schools have meaningful grouping structure.

---

## Implementation Plan

### Computational Considerations

**PyMC Implementation Notes:**

1. **Sampling strategy:**
   - 4 chains, 2000 iterations each, 1000 warmup
   - Target accept_prob = 0.95 (conservative for potential funnels)
   - NUTS sampler (default for PyMC)

2. **Priors in PyMC syntax:**
   ```python
   # Model 1
   mu = pm.Normal('mu', mu=0, sigma=20)
   tau = pm.HalfCauchy('tau', beta=10)

   # Model 2
   mu = pm.Normal('mu', mu=8, sigma=5)
   tau = pm.HalfNormal('tau', sigma=5)

   # Model 3
   mu = pm.Normal('mu', mu=0, sigma=20)
   tau = pm.HalfCauchy('tau', beta=10)
   # (plus adaptive logic shown earlier)
   ```

3. **Non-centered parameterization:**
   ```python
   eta = pm.Normal('eta', mu=0, sigma=1, shape=8)
   theta = pm.Deterministic('theta', mu + tau * eta)
   ```

4. **Expected sampling time:**
   - Model 1: ~30-60 seconds (standard hierarchical)
   - Model 2: ~30-60 seconds (similar complexity)
   - Model 3: ~45-90 seconds (adaptive logic adds overhead)

5. **Diagnostics to monitor:**
   - Trace plots for mu, tau, theta (check mixing)
   - Pair plots for (mu, tau) - check for funnel
   - ESS for all parameters (target > 400)
   - R-hat for all parameters (target < 1.01)
   - Divergence count (target 0)
   - Energy plots (check HMC geometry)

### Validation Protocol

**For each model, execute in order:**

1. **Prior predictive checks:**
   - Sample from priors only (no data)
   - Check: Are simulated y_i in plausible range [-50, 50]?
   - Ensures priors are not inadvertently degenerate

2. **Fit model to data:**
   - Run sampling as specified above
   - Monitor for warnings (divergences, ESS, R-hat)

3. **Convergence diagnostics:**
   - Visual: Trace plots, pair plots
   - Numerical: R-hat < 1.01, ESS > 400 for all parameters

4. **Posterior summaries:**
   - Mean, median, SD, 95% HDI for mu, tau, theta_i
   - Shrinkage plot: theta_i posterior vs observed y_i

5. **Posterior predictive checks:**
   - Generate 1000 replicate datasets from posterior
   - Check: What % of observed y_i fall within 95% predictive interval?
   - Plot: Observed y_i vs posterior predictive distribution

6. **Model comparison:**
   - Compute LOO-CV for all models
   - Check: Are differences substantial (>5)?
   - Pareto-k diagnostic: Are any observations problematic?

7. **Sensitivity analysis:**
   - Re-run with different prior choices (grid search)
   - Check: How stable are posteriors?

---

## Summary: Model Selection Strategy

### Phase 1: Initial Fitting (All 3 Models in Parallel)

Fit all three models and immediately check computational diagnostics:
- Model 1: Baseline (should work)
- Model 2: Skeptical (should work, potentially tighter tau)
- Model 3: Adaptive (might fail or show no advantage)

**Decision:** Drop any model with severe computational issues (divergences > 2%, ESS < 100).

### Phase 2: Posterior Examination

Compare posteriors across surviving models:
- **If Models 1 & 2 agree (tau medians within 50%):** Prior choice not critical → Proceed with both
- **If Models 1 & 2 disagree (tau medians differ > 2x):** Prior sensitivity critical → Report both, discuss implications
- **If Model 3 shows no advantage:** Drop Model 3, focus on 1 & 2

### Phase 3: Predictive Validation

Perform PPC for surviving models:
- **If all pass (>90% coverage):** Model class is appropriate → Proceed to model comparison
- **If all fail (<80% coverage):** Hierarchical structure inadequate → Pivot to alternatives (robust, complete pooling, etc.)

### Phase 4: Final Recommendation

Based on LOO-CV and scientific interpretability:
- **If Model 2 LOO better:** Recommend skeptical prior (aligns with EDA evidence)
- **If Model 1 LOO better:** Recommend standard prior (data override EDA intuition)
- **If LOO similar:** Recommend both as sensitivity analysis

**Stopping rule:** If no hierarchical model performs adequately, switch to complete pooling fixed effect model.

---

## Expected Results Given EDA

**Most likely outcome:**
- All three models will converge successfully (simple problem, well-behaved data)
- Posteriors for mu will be nearly identical: ~N(7.7, 4.0)
- Posteriors for tau will differ:
  - Model 1: Median ≈ 3-5, 95% CI [0, 12]
  - Model 2: Median ≈ 2-3, 95% CI [0, 8]
  - Model 3: Similar to Model 1, but tighter (higher ESS)
- Strong shrinkage (70-90%) in all models
- All models will pass PPC easily (large sigma_i makes this easy)
- LOO-CV will be similar across models (differences < 3)

**Conclusion:** Models will likely agree substantively, differing mainly in tau posterior width. This validates hierarchical framework while confirming EDA finding of minimal heterogeneity.

---

## Final Notes: Philosophical Stance

**I am prepared to reject hierarchical modeling entirely if:**
- Evidence suggests tau ≡ 0 (switch to complete pooling)
- Evidence suggests strong outliers (switch to robust models)
- Evidence suggests systematic structure (switch to regression models)

**Success is not fitting all three models—it's finding the truth:**
- If truth is "complete homogeneity," hierarchical models should gracefully reduce to that
- If truth is "measurement error dominates," models should reflect massive uncertainty
- If truth is "one school is different," models should struggle (revealing inadequacy)

**I will trust the data, not the model class.**

---

## Files and Code Locations

**This proposal:** `/workspace/experiments/designer_1/proposed_models.md`

**Expected implementation outputs:**
- `/workspace/experiments/designer_1/model1_noncentered.py` [PyMC code]
- `/workspace/experiments/designer_1/model2_skeptical.py` [PyMC code]
- `/workspace/experiments/designer_1/model3_adaptive.py` [PyMC code]
- `/workspace/experiments/designer_1/model_comparison.md` [results summary]
- `/workspace/experiments/designer_1/diagnostics/` [trace plots, PPCs, etc.]

**Data locations:**
- Input: `/workspace/data/data.csv`
- EDA report: `/workspace/eda/eda_report.md`

---

**END OF PROPOSAL**

*Designer 1 (Hierarchical/Random Effects Specialist)*
*Independent analysis completed: 2025-10-28*
