# Improvement Priorities: Path Forward After Experiment 3

**Date:** 2025-10-29
**Context:** Experiment 3 (Latent AR(1) NB) REJECTED for architectural inadequacy
**Status:** Need to decide on next modeling direction

---

## Executive Summary

Experiment 3 conclusively demonstrated that **latent-scale temporal structures do not work** for these data. After two failed attempts (Exp 1: no temporal structure, Exp 3: latent temporal structure), we face a critical decision:

1. **One more attempt:** Try observation-level conditional AR (fundamentally different architecture)
2. **Accept baseline:** Declare Experiment 1 adequate with limitations
3. **Pivot completely:** Try different mean function or external predictors

**Recommendation:** Option 1 (one final attempt) THEN Option 2 (accept baseline if needed).

**Stopping rule:** Maximum one additional experiment. Evidence of diminishing returns.

---

## Priority 1: Decide Whether to Continue Temporal Modeling

### The Core Question

**Is resolving residual ACF(1)=0.686 necessary for the scientific objectives?**

**If YES → Attempt Priority 2 (Observation-level AR)**
**If NO → Skip to Priority 4 (Accept Experiment 1)**

### Decision Criteria

**Continue temporal modeling IF:**
- [ ] Temporal forecasting is critical (need to predict C_{t+1} from C_t)
- [ ] Mechanistic understanding of dynamics required
- [ ] Autocorrelation interpretation is core scientific goal
- [ ] Publication requires addressing temporal dependence
- [ ] Regulatory/stakeholder requirements demand it

**Accept simpler model IF:**
- [ ] Only mean trend estimation needed (is there acceleration?)
- [ ] Conservative uncertainty acceptable (100% coverage okay)
- [ ] Point predictions adequate (R²=0.883 sufficient)
- [ ] Time/resource constraints tight
- [ ] Diminishing returns not worth continued effort

### Recommendation

**Try ONE more architectural variant (observation-level AR) before accepting baseline.**

**Rationale:**
- Exp 3 tested latent-scale structure (failed)
- Haven't tested observation-scale structure yet
- Fundamentally different approach with better theoretical justification
- If this fails, we've exhaustively tested temporal approaches
- Stopping rule: No more than 1 additional experiment

---

## Priority 2: IF CONTINUING - Observation-Level Conditional AR (Experiment 4)

### Model Specification

**Proposed: Conditional Autoregressive Negative Binomial**

```stan
// Observation model with direct count-on-count dependence
for (t in 2:N) {
  C[t] ~ neg_binomial_2_log(log_mu[t], phi);
  log_mu[t] = beta_0 + beta_1*year[t] + beta_2*year_sq[t] + gamma*log(C[t-1] + 1);
}

// First observation (no lag available)
C[1] ~ neg_binomial_2_log(log_mu_1, phi);
log_mu_1 = beta_0 + beta_1*year[1] + beta_2*year_sq[1];

// Priors
beta_0 ~ normal(4.7, 0.3);
beta_1 ~ normal(0.8, 0.2);
beta_2 ~ normal(0.3, 0.1);
gamma ~ normal(0, 0.5);  // AR coefficient on log(C_{t-1})
phi ~ gamma(2, 0.5);
```

### Why This Might Work (Where Exp 3 Failed)

**Theoretical advantages:**
1. **Direct observation-level dependence:** C_t explicitly conditions on C_{t-1}
2. **Correct scale:** Correlation on count-scale, not log-latent-scale
3. **Natural interpretation:** γ measures how previous count affects current intensity
4. **No nonlinear barrier:** Doesn't rely on exp() transformation preserving structure

**Key difference from Experiment 3:**
- **Exp 3:** C_t ⊥ C_{t-1} | α_t (conditional independence given latent state)
- **Exp 4:** C_t → C_{t-1} (direct dependence)

**This is fundamentally different architecture, not incremental tuning.**

### Success Criteria (ALL Must Be Met)

**Primary (Required):**
1. **Residual ACF(1) < 0.3** (currently 0.690, target is < 0.3)
   - This is THE metric. If not met, model fails.
   - Check: Compute residuals as C_t - E[C_t | C_{t-1}, year], compute ACF

2. **Coverage 90-98% at 95% level** (currently 100%)
   - Tighter than current over-conservative intervals
   - Check: Empirical coverage of posterior predictive intervals

**Secondary (Desired):**
3. **ΔLOO vs Exp 1 > 2×SE** (currently 4.85 ± 7.47 for Exp 3)
   - Clear predictive improvement, not marginal
   - Check: LOO-CV comparison with decisive difference

4. **Point predictions no worse than Exp 1** (Exp 3 worsened R²)
   - R² ≥ 0.883 (Exp 1 baseline)
   - Residual SD ≤ 34.44 (Exp 1 baseline)
   - Check: Posterior mean predictions vs observed

5. **No systematic temporal patterns** (currently U-shaped waves)
   - Residuals vs time should be flat
   - No smooth trends visible
   - Check: Visual inspection + runs test

**Stopping criteria:**
- If criteria 1-2 met: SUCCESS, use this model
- If criterion 1 not met: STOP temporal modeling, accept Exp 1
- If criteria 1-2 met but 3-5 fail: Evaluate cost-benefit

### Implementation Details

**Practical considerations:**

1. **Initial condition problem:**
   - C[1] has no previous observation
   - Options:
     - Use trend-only for first observation (implemented above)
     - Use stationary distribution
     - Condition on pre-observation period (if data available)

2. **Log-transform of lag:**
   - log(C_{t-1} + 1) handles C_{t-1}=0 case
   - +1 offset prevents log(0)
   - Could also use sqrt(C_{t-1}) or C_{t-1}^0.5 as alternatives

3. **Interpretation of γ:**
   - γ > 0: Positive dependence (high counts follow high counts)
   - γ < 0: Negative dependence (oscillation)
   - γ = 0: No conditional dependence (reduces to Exp 1)

4. **Prior on γ:**
   - Normal(0, 0.5) weakly informative
   - Allows both positive/negative correlation
   - Could use Normal(0.5, 0.3) if expecting positive correlation

**Computational considerations:**

- Expect similar convergence to Exp 1 (simpler than Exp 3)
- No latent variables (faster than Exp 3)
- Slightly longer than Exp 1 due to lag dependency
- Estimate: ~15 minutes for full sampling

### Validation Plan

**If model converges (R̂ < 1.05, ESS > 400):**

1. **Residual ACF analysis** (PRIMARY CHECK)
   - Compute: r_t = C_t - E[C_t | C_{t-1}, year_t]
   - Plot: ACF(r_t) for lags 1-15
   - Test: Is ACF(1) < 0.3? (YES = pass, NO = fail)

2. **Posterior predictive checks**
   - Coverage at 50%, 80%, 95% levels
   - Test statistics (13 summary statistics)
   - Residuals vs time plot (should be flat)
   - Trajectory comparison (replication should look like observed)

3. **Model comparison**
   - LOO-CV vs Experiment 1
   - Check: ΔELPD > 2×SE for decisive improvement
   - Stacking weights

4. **Parameter interpretation**
   - Is γ significantly different from zero?
   - Does γ ≈ 0.7 (matching residual ACF from Exp 1)?
   - Are β₀, β₁, β₂ similar to Exp 1/Exp 3?

**Timeline:** 3-4 days (1 day model implementation, 1 day sampling, 1-2 days diagnostics)

### Expected Outcomes

**Best case:** ACF(1) drops to < 0.3
- Use Experiment 4 for final analysis
- Publish with temporal structure correctly modeled
- γ parameter provides scientific insight

**Likely case:** ACF(1) improves but remains 0.4-0.6
- Marginal improvement, not decisive
- Cost-benefit unclear
- Need to evaluate if worth complexity

**Worst case:** ACF(1) unchanged at ~0.69
- Observation-level AR also insufficient
- Three different architectures all failed
- Strong evidence to accept Exp 1 baseline

---

## Priority 3: Alternative Approaches (If Exp 4 Fails)

If observation-level AR (Priority 2) doesn't achieve ACF(1) < 0.3, consider these alternatives **before** accepting baseline:

### Option A: Different Mean Function

**Hypothesis:** Residual ACF is artifact of mean function misspecification, not true temporal correlation.

**Evidence supporting:**
- U-shaped residual pattern vs fitted values (suggests wrong functional form)
- Large negative residuals at end of series (late-period bias)
- Both Exp 1 and Exp 3 have same residual patterns (common mean structure)

**Try: Exponential Growth Model**

```stan
C[t] ~ neg_binomial_2_log(log_mu[t], phi);
log_mu[t] = beta_0 + beta_1 * exp(beta_2 * year[t]);
```

**Rationale:**
- Data show accelerating growth (quadratic currently captures this)
- True growth may be exponential (common in many domains)
- Better mean function might eliminate apparent autocorrelation

**Success criteria:**
- Residual ACF(1) < 0.3
- R² > 0.883 (better than Exp 1)
- No U-shaped residual pattern

**Cost:** ~3 days (similar to Exp 4)

**Stopping rule:** If still fails ACF test, proceed to Priority 4 (accept baseline)

### Option B: Spline/Nonparametric Trend

**Hypothesis:** Parametric trend (quadratic/exponential) too restrictive.

**Try: Generalized Additive Model**

```python
# Using mgcv or pymc-experimental
C[t] ~ NegBinomial(mu[t], phi)
log(mu[t]) = s(year[t])  # smooth function of year
```

**Rationale:**
- Flexible mean function adapts to data
- No forced parametric form
- Might capture subtle non-monotonic trends

**Cost:** ~5 days (more complex implementation)

**Caution:** Higher risk of overfitting with only 40 observations

### Option C: External Predictors

**Hypothesis:** Temporal correlation due to unmeasured time-varying covariates.

**If additional data available:**
- Economic indicators (GDP, market trends)
- Environmental variables (temperature, events)
- Policy changes (regulations, interventions)

**Model:**
```stan
log_mu[t] = beta_0 + beta_1*year[t] + beta_2*year_sq[t] + beta_3*X[t]
```

**Success criteria:**
- X[t] significantly associated with outcome
- Residual ACF(1) < 0.5 (partial improvement acceptable)
- Improved LOO-CV

**Limitation:** Requires data we may not have

---

## Priority 4: ACCEPT Experiment 1 as Adequate Baseline (Fallback)

If temporal modeling attempts fail (Exp 3 failed, Exp 4 fails, alternatives fail), **accept Experiment 1** with documented limitations.

### Justification for Acceptance

**Experiment 1 (Quadratic NB) is adequate IF:**

1. **Primary scientific question is answerable:**
   - "Is growth accelerating?" → YES (β₂ = 0.10 [0.01, 0.19])
   - "What is the trend?" → YES (R² = 0.883)
   - "What is overall magnitude?" → YES (mean well-captured)

2. **Predictions are reasonable:**
   - Mean predictions highly correlated with observed (R² = 0.883)
   - Conservative uncertainty (100% coverage protects against surprises)
   - No systematic bias in point estimates

3. **Limitations are documented:**
   - Residual ACF(1) = 0.686 is known and reported
   - Temporal dependence acknowledged but unresolved
   - Coverage excessive (100%) noted as conservative
   - Not suitable for temporal forecasting (stated clearly)

4. **Complexity not justified:**
   - Three attempts at temporal structures failed (Exp 1, 3, 4)
   - Diminishing returns evident
   - Simpler model preferred when complex models don't improve fit

### What to Document

**In methods section:**
```
"We fitted a Negative Binomial regression model with quadratic time trend
to capture overall growth patterns. Posterior predictive checks revealed
residual temporal autocorrelation (ACF(1)=0.686), indicating observations
are not fully independent given time. However, attempts to model this
correlation using latent AR(1) structures (not shown) did not improve
predictive performance. We therefore present results from the simpler
quadratic model, noting that temporal dependence may affect uncertainty
estimates."
```

**In results section:**
```
"The model captures overall acceleration in counts (β₂ = 0.10, 95% CI [0.01, 0.19]),
with point predictions strongly correlated with observed data (R² = 0.883).
Prediction intervals are conservative (100% coverage of 95% intervals),
likely reflecting unmodeled temporal correlation."
```

**In limitations section:**
```
"The model treats observations as independent given time, but residual
autocorrelation (ACF(1)=0.686) indicates temporal persistence not fully
captured. This may affect short-term forecasting accuracy. The model is
most appropriate for estimating long-term trends rather than predicting
specific future values."
```

### When to Use Experiment 1

**Appropriate uses:**
- Estimating overall trend direction and magnitude
- Testing hypothesis about acceleration
- Comparing trends across groups/conditions
- Illustrative predictions with wide uncertainty
- Exploratory data analysis

**Inappropriate uses:**
- Temporal forecasting (predicting next observation)
- Precise uncertainty quantification
- Claiming observations are independent
- Mechanistic modeling of temporal dynamics
- Regulatory decisions requiring exact predictions

### Comparison Table for Decision-Making

| Use Case | Exp 1 Adequate? | Reason |
|----------|-----------------|--------|
| Is growth accelerating? | ✓ YES | β₂ well-estimated |
| Forecast next month | ✗ NO | Ignores temporal dependence |
| Compare to other series | ✓ YES | Trend estimation robust |
| Understand dynamics | ✗ NO | Doesn't model temporal process |
| Risk assessment | ~ MAYBE | Conservative intervals protective but imprecise |
| Publication (descriptive) | ✓ YES | With documented limitations |
| Publication (causal) | ? DEPENDS | On whether temporal confounding matters |

---

## Priority 5: Meta-Analysis of What We Learned

### Summary of Three Experiments

| Experiment | Structure | Outcome | Lesson |
|------------|-----------|---------|--------|
| Exp 1 | Quadratic trend, no temporal | Adequate mean fit, ACF=0.686 | Baseline captures trend but not dynamics |
| Exp 3 | Quadratic + latent AR(1) | Perfect convergence, ACF=0.690 | Latent temporal structures don't work |
| Exp 4 | Quadratic + conditional AR | TBD | Will test observation-level structures |

### What Worked

**Negative Binomial distribution:**
- All experiments show overdispersion well-handled
- φ parameter consistently estimated ~14-20
- No evidence of zero-inflation or distribution family issues

**Quadratic trend:**
- β₀, β₁, β₂ similar across Exp 1 and Exp 3 (robust)
- Captures acceleration (β₂ > 0)
- R² ≈ 0.86-0.88 across experiments

**Computational methods:**
- PyMC implementation successful
- Convergence excellent in all experiments
- NUTS sampling efficient
- Diagnostics comprehensive

### What Didn't Work

**Latent-scale temporal structures:**
- AR(1) on log-scale doesn't translate to count-scale
- Nonlinear transformation breaks correlation propagation
- σ_η too small to matter (0.09)

**Excessive complexity without benefit:**
- 46 parameters (Exp 3) vs 4 parameters (Exp 1) = 11x increase
- 2.5x computation time
- LOO improvement < 1 SE (weak evidence)
- Zero residual ACF improvement

### What We Still Don't Know

**Source of temporal autocorrelation:**
- Is it true autoregressive dynamics? (C_t depends on C_{t-1})
- Is it mean function misspecification? (Wrong functional form)
- Is it unmeasured covariates? (External time-varying factors)
- Is it measurement artifact? (Overlapping observation windows)

**Whether it's resolvable:**
- Multiple architectures failed
- May be fundamental limitation
- May require data we don't have
- May not matter for scientific questions

---

## Recommended Action Plan

### Short-Term (Next 2 Weeks)

**Week 1: Experiment 4 (Conditional AR)**
- Day 1-2: Implement observation-level AR model
- Day 3-4: Run sampling and convergence diagnostics
- Day 5: Compute residual ACF (PRIMARY CHECK)

**Week 2: Decision**
- If ACF(1) < 0.3: Full PPC suite, proceed with Exp 4
- If ACF(1) > 0.3: Accept Experiment 1, write up with limitations
- Document decision and rationale

### Medium-Term (Next Month)

**If Experiment 4 succeeds:**
- Complete full validation
- Compare to Exp 1 and Exp 3
- Write methods section
- Prepare for publication

**If Experiment 4 fails:**
- Formal acceptance of Experiment 1
- Document limitations thoroughly
- Focus analysis on trend estimation (not forecasting)
- Consider sensitivity analyses

### Long-Term (Beyond This Project)

**Future work that could address limitations:**
- Collect additional time points (longer series)
- Gather external predictors (covariates)
- Try changepoint models (non-stationary)
- Gaussian process models (different smoothness assumption)
- Consult domain expert on temporal mechanisms

**Meta-lessons for future projects:**
- Start with simplest model (don't over-engineer)
- Test multiple architectures early (latent vs observation-scale)
- Have clear stopping criteria (avoid endless tuning)
- Accept imperfection when justified (diminishing returns)

---

## Decision Tree Summary

```
START: Experiment 3 REJECTED
│
├─ Q: Is temporal modeling critical for scientific goals?
│  │
│  ├─ YES → Priority 2: Try Experiment 4 (Obs-level AR)
│  │  │
│  │  ├─ ACF(1) < 0.3? → SUCCESS, use Exp 4
│  │  │
│  │  └─ ACF(1) > 0.3? → Priority 3: Try alternatives
│  │     │
│  │     ├─ Exponential mean function
│  │     ├─ Spline/GAM
│  │     └─ If all fail → Priority 4: Accept Exp 1
│  │
│  └─ NO → Priority 4: Accept Experiment 1 immediately
│
END: Use Exp 1 or Exp 4, document limitations
```

---

## Final Recommendation

**PRIORITY ORDER:**

1. **Try Experiment 4 (observation-level conditional AR)** - 1 week
   - Fundamentally different from Exp 3 (observation-scale)
   - Clear success criteria (ACF(1) < 0.3)
   - If succeeds: Use it. If fails: Stop temporal modeling.

2. **If Exp 4 fails, accept Experiment 1** - immediate
   - Simpler is better when complex models don't help
   - Document limitations transparently
   - Focus on trend estimation, not forecasting

3. **Do NOT pursue:**
   - More latent temporal structures (ruled out by Exp 3)
   - Endless tuning of existing models (diminishing returns)
   - Complex models without clear benefit

**STOPPING RULE:** Maximum 1 additional experiment (Exp 4). If that fails, accept Exp 1.

**TIMELINE:** Decision within 2 weeks.

---

**Document created:** 2025-10-29
**Status:** Ready for implementation
**Next action:** Decide on Experiment 4 vs accept Experiment 1
