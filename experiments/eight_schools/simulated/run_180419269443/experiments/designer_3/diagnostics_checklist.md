# Model Diagnostics Checklist
## Designer 3 - Bayesian Meta-Analysis Quality Control

**Purpose:** Systematic checklist to assess model adequacy before accepting results.
**Principle:** Each model must PASS all critical checks or be rejected/modified.

---

## Phase 1: Pre-Fitting Validation (MUST COMPLETE BEFORE SAMPLING)

### 1.1 Prior Predictive Checks

**Goal:** Ensure priors are sensible before seeing posterior

**Procedure:**
1. Sample from priors only (no data): `mu_sim ~ prior, tau_sim ~ prior`
2. Generate synthetic datasets: `y_sim ~ Normal(theta_sim, sigma_observed)`
3. Compute statistics on synthetic data: mean, SD, range, I²

**Checks:**
- [ ] Prior predictive mean(y_sim) includes observed mean (11.27)
- [ ] Prior predictive range includes observed range [-4.88, 26.08]
- [ ] Prior predictive I² distribution includes observed I² (2.9%)
- [ ] 95% of prior predictive datasets are "reasonable" (not extreme)

**Pass criteria:**
- Observed data should be in middle 50% of prior predictive distribution
- If observed data is in extreme 5% tails, prior is TOO INFORMATIVE

**Fail actions:**
- If FAIL: Widen priors (increase SD by 50%)
- If STILL FAIL: Reconsider model class entirely

---

### 1.2 Simulation-Based Calibration (SBC)

**Goal:** Verify that MCMC sampler can recover true parameters

**Procedure:**
1. Draw true parameters from prior: `mu_true ~ prior, tau_true ~ prior`
2. Simulate dataset: `y_sim ~ model(mu_true, tau_true)`
3. Fit model to y_sim, check if posterior contains mu_true, tau_true
4. Repeat 100 times

**Checks:**
- [ ] Rank statistics are uniform (SBC histogram is flat)
- [ ] Coverage: 95% credible intervals contain true value 95% of time
- [ ] No bias: posterior mean ≈ true value on average

**Pass criteria:**
- SBC p-value > 0.05 (uniform rank histogram)
- Coverage within [93%, 97%] for 100 simulations

**Fail actions:**
- If FAIL: Check Stan code for errors
- If STILL FAIL: Model is misspecified or sampler is broken

---

## Phase 2: MCMC Diagnostics (CHECK IMMEDIATELY AFTER SAMPLING)

### 2.1 Convergence Diagnostics

**Goal:** Ensure MCMC has converged to true posterior

**Checks:**
- [ ] R-hat < 1.01 for all parameters (mu, tau, theta[1:J])
- [ ] R-hat < 1.05 for all generated quantities (I², theta_new)
- [ ] Trace plots show "hairy caterpillar" (no trends, no sticking)
- [ ] No stuck chains (all chains explore same space)

**Pass criteria:**
- ALL parameters have R-hat < 1.01
- Visual inspection of traces shows good mixing

**Fail actions:**
- If FAIL: Increase warmup (1000 → 2000 iterations)
- If STILL FAIL: Increase adapt_delta (0.8 → 0.95 → 0.99)
- If STILL FAIL: Reparameterize (centered → non-centered or vice versa)

---

### 2.2 Effective Sample Size (ESS)

**Goal:** Ensure enough independent samples for inference

**Checks:**
- [ ] ESS_bulk > 1000 for mu, tau
- [ ] ESS_tail > 1000 for mu, tau (for 95% CI estimation)
- [ ] ESS > 400 per chain (4 chains = 1600 total minimum)
- [ ] No parameters with ESS < 100 (critical failure)

**Pass criteria:**
- ESS_bulk > 1000 AND ESS_tail > 1000 for key parameters

**Fail actions:**
- If FAIL: Run more iterations (2000 → 5000)
- If ESS very low (<100): Increase adapt_delta, check for funnel geometry

---

### 2.3 Divergent Transitions

**Goal:** Detect regions of posterior that sampler cannot explore accurately

**Checks:**
- [ ] Zero divergent transitions (ideal)
- [ ] <1% divergent transitions (acceptable if localized)
- [ ] Pairs plot: divergences NOT concentrated in funnel (tau→0, mu varies widely)

**Pass criteria:**
- Zero divergences (strict)
- OR <10 divergences AND not concentrated in funnel region

**Fail actions:**
- If FAIL: Increase adapt_delta (0.8 → 0.95 → 0.99)
- If STILL FAIL: Use non-centered parameterization
- If STILL FAIL: Model may be misspecified (e.g., tau near boundary)

---

### 2.4 MCMC Standard Error (MCSE)

**Goal:** Ensure Monte Carlo error is small relative to posterior SD

**Checks:**
- [ ] MCSE(mu) < 0.1 * SD(mu)
- [ ] MCSE(tau) < 0.1 * SD(tau)
- [ ] MCSE for all theta[j] < 0.1 * SD(theta[j])

**Pass criteria:**
- MCSE < 0.1 * posterior SD for all parameters

**Fail actions:**
- If FAIL: Run more iterations (MCSE scales as 1/sqrt(ESS))

---

## Phase 3: Model Adequacy (CHECK AFTER CONVERGENCE CONFIRMED)

### 3.1 Posterior Predictive Checks (PPC)

**Goal:** Verify model can reproduce observed data patterns

**Procedure:**
1. Draw posterior samples: `mu_s, tau_s, theta_s ~ posterior`
2. Simulate replicate datasets: `y_rep_s ~ Normal(theta_s, sigma_obs)`
3. Compute test statistics: T(y_rep) and T(y_obs)
4. Bayesian p-value: P(T(y_rep) > T(y_obs))

**Test statistics to check:**
- [ ] Mean: P(mean(y_rep) > mean(y_obs)) ∈ [0.05, 0.95]
- [ ] SD: P(SD(y_rep) > SD(y_obs)) ∈ [0.05, 0.95]
- [ ] Min: P(min(y_rep) < min(y_obs)) ∈ [0.05, 0.95]
- [ ] Max: P(max(y_rep) > max(y_obs)) ∈ [0.05, 0.95]
- [ ] Range: P(range(y_rep) > range(y_obs)) ∈ [0.05, 0.95]

**Visual checks:**
- [ ] Histogram of y_rep overlaps with y_obs
- [ ] Q-Q plot of y_rep vs y_obs is linear
- [ ] LOO-PIT histogram is roughly uniform

**Pass criteria:**
- At least 4 of 5 test statistics have p ∈ [0.05, 0.95]
- Visual checks show good overlap

**Fail actions:**
- If mean/SD fail: Model misspecified (wrong location/scale)
- If min/max fail: Tails too light (consider t-distribution)
- If multiple failures: Reject model, try alternative class

---

### 3.2 Leave-One-Out Cross-Validation (LOO-CV)

**Goal:** Assess predictive performance and identify influential observations

**Procedure:**
1. Compute Pareto k for each study using PSIS-LOO
2. Calculate elpd_loo (expected log predictive density)
3. Check for highly influential studies

**Checks:**
- [ ] All Pareto k < 0.5 (all studies well-fit)
- [ ] At most 1 study with 0.5 < k < 0.7 (moderate influence)
- [ ] No studies with k > 0.7 (high influence / outliers)
- [ ] elpd_loo SE is reasonable (not huge uncertainty)

**Pareto k interpretation:**
| k value | Interpretation | Action |
|---------|----------------|--------|
| k < 0.5 | Good | None |
| 0.5 < k < 0.7 | Moderate influence | Investigate study |
| 0.7 < k < 1.0 | High influence | Refit without study |
| k > 1.0 | Very high influence | Posterior dominated by study |

**Expected results (based on EDA):**
- Study 4: Pareto k ~ 0.5-0.7 (EDA showed 33% influence)
- Study 5: Pareto k ~ 0.4-0.6 (EDA showed 23% influence)
- Others: Pareto k < 0.5

**Fail actions:**
- If k > 0.7 for Study 4 or 5: Refit without that study
- If k > 0.7 for >2 studies: Model may be misspecified
- If k > 1.0 for any study: Critical failure, investigate immediately

---

### 3.3 Prior-Posterior Comparison

**Goal:** Ensure data updated prior (learned from data)

**Procedure:**
1. Plot prior density and posterior density on same axis
2. Compute overlap: integral[min(prior, posterior)] dθ
3. Compute prior-posterior contraction: SD(posterior) / SD(prior)

**Checks:**
- [ ] Overlap(mu) ∈ [0.1, 0.8] (learned, but not completely surprised)
- [ ] Overlap(tau) ∈ [0.1, 0.8]
- [ ] Contraction(mu) < 0.9 (posterior is tighter than prior)
- [ ] Contraction(tau) < 0.9

**Interpretation:**
- Overlap ~ 1: No learning (data too weak OR prior perfect)
- Overlap ~ 0: Severe prior-data conflict (prior was very wrong)
- Overlap ~ 0.3-0.5: Healthy learning

**Pass criteria:**
- Overlap ∈ [0.1, 0.8] AND Contraction < 0.9 for both mu and tau

**Fail actions:**
- If Overlap > 0.9: Data too weak, consider abandoning pooling
- If Overlap < 0.05: Prior severely misspecified, refit with wider prior
- If Contraction > 0.9: Prior dominated inference, refit with weaker prior

---

### 3.4 Shrinkage Diagnostics

**Goal:** Verify hierarchical structure is working as expected

**Procedure:**
1. Compute shrinkage factor for each study:
   ```
   B_j = 1 / (1 + sigma_j^2 / tau^2)
   ```
   Expected shrinkage toward pooled mean.

2. Compare observed shrinkage to theoretical:
   ```
   theta_j_posterior ≈ B_j * y_j + (1 - B_j) * mu
   ```

**Checks:**
- [ ] Studies with small sigma shrink less (B_j small)
- [ ] Studies with large sigma shrink more (B_j large)
- [ ] Shrinkage pattern matches EDA findings (>95% shrinkage expected)
- [ ] No "anti-shrinkage" (posterior theta_j farther from mu than y_j)

**Expected results (EDA: I² = 2.9%, strong shrinkage):**
- Study 5 (sigma=9): Shrinkage ~0.98 (98% toward pooled mean)
- Study 8 (sigma=18): Shrinkage ~0.99 (99% toward pooled mean)
- All studies: Shrinkage > 0.95

**Pass criteria:**
- Shrinkage factors match theoretical formula within ±0.1
- No anti-shrinkage observed

**Fail actions:**
- If shrinkage < expected: tau is overestimated
- If anti-shrinkage: Model misspecified or convergence issue

---

### 3.5 Influence Analysis (Leave-One-Out Refitting)

**Goal:** Quantify impact of each study on posterior inference

**Procedure:**
1. Refit model J times, each time excluding one study
2. Compute change in posterior mean:
   ```
   influence_j = |mu_full - mu_(-j)| / SD(mu_full)
   ```
3. Compare to EDA findings

**Checks:**
- [ ] Study 4 influence ~ 33% (matches EDA: -33.2% when removed)
- [ ] Study 5 influence ~ 23% (matches EDA: +23.0% when removed)
- [ ] No study has influence > 50% (no single study dominates)
- [ ] Influence pattern correlates with precision weights

**Expected results:**
| Study | EDA Change | Expected Bayesian Influence |
|-------|------------|----------------------------|
| 4 | -33.2% | ~0.3-0.4 SD |
| 5 | +23.0% | ~0.2-0.3 SD |
| Others | <10% | <0.1 SD |

**Pass criteria:**
- Bayesian influence within 2x of EDA findings
- No influence > 1.0 SD (very extreme)

**Fail actions:**
- If influence >> EDA: Posterior too concentrated
- If influence << EDA: Posterior too dispersed
- If influence > 1 SD: Report sensitivity prominently

---

## Phase 4: Model-Specific Diagnostics

### 4.1 Model 1 Specific (Weakly Informative)

**Checks:**
- [ ] Posterior tau ∈ [1, 10] (not at boundary, not extreme)
- [ ] Posterior I² ∈ [0%, 20%] (consistent with EDA 2.9%)
- [ ] Posterior mu ∈ [5, 20] (consistent with EDA 11.27)
- [ ] No funnel geometry in pairs(mu, tau)

**Fail criteria:**
- Posterior tau > 15: Heterogeneity severely underestimated by EDA
- Posterior I² > 30%: Prior on tau is too restrictive
- Funnel geometry: Switch to non-centered parameterization

---

### 4.2 Model 2 Specific (Conflict Detection)

**Checks:**
- [ ] Posterior pi_conflict < 0.3 (most studies not conflicts)
- [ ] At most 2 studies have P(z_j=1) > 0.5
- [ ] Inflation factor posterior is data-informed (not equal to prior)
- [ ] Mixture components resolved (not stuck at 50-50 weights)

**Expected results:**
- Study 5 might be flagged: P(z_5=1) ~ 0.4-0.6
- pi_conflict posterior ~ 0.1-0.2
- Inflation factor ~ 2-4

**Fail criteria:**
- pi_conflict > 0.5: Too many conflicts, model overfitting
- All z_j = 0: Conflict mechanism unused, unnecessary complexity
- Mixture weights unchanged from prior: No learning

---

### 4.3 Model 3 Specific (Ensemble)

**Checks:**
- [ ] Skeptical and Enthusiastic models both converge (R-hat < 1.01)
- [ ] |mu_skep - mu_enth| quantified with uncertainty
- [ ] Stacking weights have reasonable CI (not [0,1] ± 0.5)
- [ ] Agreement metric computed for thresholds [3, 5, 10]

**Expected results (if data are informative):**
- |mu_skep - mu_enth| < 5 (models agree)
- Stacking weights ~ [0.5, 0.5] ± [0.2, 0.2]
- Agreement at threshold=5: P(agree) > 0.8

**Expected results (if data are weak):**
- |mu_skep - mu_enth| > 10 (models diverge)
- Stacking weights ~ [0.8, 0.2] or [0.2, 0.8] (one dominates)
- Agreement at threshold=5: P(agree) < 0.3

**Fail criteria:**
- Both models identical (priors too weak)
- Stacking weights unstable (changes >0.3 with single study removal)
- Agreement metric is threshold-dependent (uninformative)

---

## Phase 5: Final Validation (BEFORE REPORTING RESULTS)

### 5.1 Stress Tests

**Test 1: Extreme prior sensitivity**
- Refit Model 1 with mu ~ N(0, 1000) [nearly flat]
- Posterior mu should change by < 20%

**Test 2: Outlier injection**
- Add synthetic outlier: y_9 = 100, sigma_9 = 10
- Pareto k_9 should be > 0.7 (detected as outlier)

**Test 3: Data doubling**
- Duplicate all studies (J=16)
- Posterior SD(mu) should decrease by ~0.71x
- Posterior tau should remain similar

**Test 4: Heterogeneity injection**
- Simulate data with tau=10 (not tau=2)
- Model should recover tau ~ 8-12

**Pass criteria:**
- Test 1: Robust to prior (< 20% change)
- Test 2: Detects outliers
- Test 3: Uncertainty scales correctly
- Test 4: Recovers true heterogeneity

---

### 5.2 Comparison to EDA

**Checks:**
- [ ] Posterior mu ~ 11 ± 4 (EDA: 11.27, CI: 3.29-19.25)
- [ ] Posterior tau ~ 2 ± 2 (EDA: 2.02)
- [ ] Posterior I² ~ 5% ± 5% (EDA: 2.9%)
- [ ] Shrinkage matches EDA (>95% for all studies)

**Pass criteria:**
- Bayesian estimates within EDA 95% CI
- Direction of effects matches (all positive except Study 5)

**Fail actions:**
- If Bayesian estimate outside EDA CI: Investigate prior influence
- If direction differs: Critical failure, investigate data/code

---

### 5.3 Scientific Plausibility

**Checks:**
- [ ] Posterior mu is scientifically plausible (not extreme)
- [ ] Posterior I² is reasonable for this field (0-20% is typical)
- [ ] Prediction interval width is sensible (not wider than data range)
- [ ] Individual theta_j are all plausible

**Domain knowledge checks:**
- Effect size mu ~ 11: Is this large/small/reasonable?
- Heterogeneity I² ~ 3%: Consistent with similar meta-analyses?
- Study 5 negative effect: Can this be explained by moderators?

**Pass criteria:**
- Results do not contradict domain knowledge
- Magnitude of effects is interpretable

---

## Summary: GO / NO-GO Decision

**GREEN LIGHT (report results):**
- All Phase 1-2 diagnostics PASS
- At least 4/5 Phase 3 diagnostics PASS
- Model-specific checks PASS
- Stress tests PASS

**YELLOW LIGHT (report with caveats):**
- Phase 1-2 PASS, but 1-2 Phase 3 checks FAIL
- Model-specific checks mostly PASS
- Stress tests reveal sensitivity (report)

**RED LIGHT (reject model):**
- Any Phase 2 diagnostic FAILS (non-convergence, divergences)
- >2 Phase 3 diagnostics FAIL
- Stress tests reveal severe fragility
- Results contradict domain knowledge

---

## Checklist Completion Log

**Model 1:**
- [ ] Phase 1 complete: _______ (date)
- [ ] Phase 2 complete: _______ (date)
- [ ] Phase 3 complete: _______ (date)
- [ ] Phase 4 complete: _______ (date)
- [ ] Phase 5 complete: _______ (date)
- [ ] Final decision: GREEN / YELLOW / RED

**Model 2:**
- [ ] Phase 1 complete: _______ (date)
- [ ] Phase 2 complete: _______ (date)
- [ ] Phase 3 complete: _______ (date)
- [ ] Phase 4 complete: _______ (date)
- [ ] Phase 5 complete: _______ (date)
- [ ] Final decision: GREEN / YELLOW / RED

**Model 3:**
- [ ] Phase 1 complete: _______ (date)
- [ ] Phase 2 complete: _______ (date)
- [ ] Phase 3 complete: _______ (date)
- [ ] Phase 4 complete: _______ (date)
- [ ] Phase 5 complete: _______ (date)
- [ ] Final decision: GREEN / YELLOW / RED

---

**Notes:**
- Do NOT skip phases - each builds on previous
- If any RED LIGHT: stop, investigate, do not proceed
- Document all failures and actions taken
- When in doubt, err on side of caution (reject model rather than accept bad fit)
