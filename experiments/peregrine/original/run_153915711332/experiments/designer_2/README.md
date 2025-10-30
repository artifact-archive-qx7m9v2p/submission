# Designer 2: Temporal Structure & Trend Models

**Focus Area:** Time series structures, autocorrelation, trend specification
**Date:** 2025-10-29
**Data:** `/workspace/data/data_designer_2.csv` (n=40)

---

## Quick Start

### Files in This Directory

1. **`proposed_models.md`** (35KB) - MAIN DOCUMENT
   - Full model specifications with Stan code skeletons
   - Detailed falsification criteria
   - Stress tests and pivot strategies
   - Complete rationale and priors

2. **`design_summary.md`** (10KB) - QUICK REFERENCE
   - Executive summary of three model classes
   - Implementation roadmap (5 phases)
   - Success/failure criteria
   - Key insights and uncertainties

3. **`README.md`** (this file) - INDEX

---

## Three Proposed Models

| # | Model Class | Core Mechanism | Why It Might Be Right | Why It Might Fail | Priority |
|---|-------------|----------------|----------------------|-------------------|----------|
| **1** | **State-Space** | Latent random walk + drift + NegBin observation | ACF(1)=0.989 suggests autoregressive process, not cross-sectional trend | If innovation variance = observation variance (no benefit from latent structure) | ⭐⭐⭐ |
| **2** | **Changepoint** | Piecewise linear with break at t≈0.3 | CUSUM shows clear minimum, 4.5x mean jump, t-test p<0.0001 | If smooth transition fits better, or changepoint location is diffuse | ⭐⭐ |
| **3** | **Gaussian Process** | Nonparametric smooth function via GP prior | Neither parametric form may be correct, GP can represent any smooth trend | If simple parametric model fits equivalently (Occam's razor) | ⭐⭐ |

---

## Key Design Philosophy

### Competing Hypotheses Approach
These three models represent **fundamentally different data generation processes**:
- **Model 1:** Temporal evolution dominates (time series mindset)
- **Model 2:** Discrete structural break (intervention/shock mindset)
- **Model 3:** Smooth but complex growth (agnostic/flexible mindset)

### Falsification Mindset
Each model has explicit criteria for abandonment:
- **State-Space:** Abandon if residual ACF > 0.5 (temporal structure not captured)
- **Changepoint:** Abandon if τ posterior is uniform (no preferred location)
- **GP:** Abandon if lengthscale → ∞ (equivalent to linear, overparameterized)

### Decision Rules
1. **If ΔLOO > 10:** Strong preference for better model
2. **If ΔLOO < 4:** Models equivalent, use Bayesian Model Averaging
3. **If all fail diagnostics (ACF > 0.6, coverage < 70%):** PIVOT to new model classes

---

## Critical EDA Insights (from `/workspace/eda/eda_report.md`)

**What We Know:**
- Extreme overdispersion: Var/Mean = 67.99
- Massive autocorrelation: ACF(1) = 0.989
- Strong growth: 8.45× increase (29 → 245)
- Lag-1 regression: R² = 0.977, slope ≈ 1.011 (near random walk)
- Possible changepoint: CUSUM minimum at year = 0.30

**What This Implies:**
- Poisson models will FAIL (overdispersion too extreme)
- Independent observations assumption VIOLATED (ACF = 0.989)
- Linear trend INADEQUATE (residuals show U-shape)
- Time series methods likely more appropriate than cross-sectional GLM

---

## Implementation Checklist

### Phase 1: Basic Fitting
- [ ] Code three Stan models (skeletons in `proposed_models.md` Appendix)
- [ ] Fit all with 4 chains × 4000 iterations (2000 warmup)
- [ ] Check convergence: R-hat < 1.01, ESS > 400
- [ ] Compute LOO-ELPD for model comparison

### Phase 2: Diagnostics
- [ ] Residual ACF analysis (target: ACF(1) < 0.3)
- [ ] One-step-ahead predictions (target: 80% coverage = 75-85%)
- [ ] Posterior predictive checks (max, mean, var, ACF of replicates)
- [ ] Prior sensitivity (weak vs. strong priors)

### Phase 3: Model Selection
- [ ] Compare LOO-ELPD (decision rule: ΔLOO > 10 = clear winner)
- [ ] Check all diagnostics pass for "winner"
- [ ] If ambiguous: Bayesian Model Averaging
- [ ] If all fail: Document failure, pivot strategy

### Phase 4: Stress Tests
- [ ] Extrapolation: Fit on year < 1.0, predict year ≥ 1.0
- [ ] Jackknife: Remove changepoint region, refit
- [ ] Simulation: Generate under known mechanisms, test recovery

### Phase 5: Reporting
- [ ] Model comparison table
- [ ] Parameter posteriors with interpretation
- [ ] Visualizations (data + fits + predictions)
- [ ] Limitations and uncertainties

---

## Expected Outcomes

### My Prediction (60% confidence)
**State-Space model will win because:**
- The near-perfect lag-1 autocorrelation (R² = 0.977) is the dominant signal
- Much of the "overdispersion" is actually temporal correlation
- A random walk with drift naturally produces exponential-looking growth

**Expected parameters:**
- Drift δ ≈ 0.06 [0.04, 0.08] per period in log-space
- Innovation SD σ_η ≈ 0.08 [0.05, 0.12]
- Dispersion φ ≈ 15 [10, 22] (much less than naive 68)
- LOO-ELPD ≈ -150

### Alternative Scenario (25% confidence)
**Models are equivalent (ΔLOO < 4):**
- All capture the data reasonably well
- n=40 insufficient to distinguish mechanisms
- Use Bayesian Model Averaging for predictions
- Report: "Multiple mechanisms consistent with data"

### Failure Scenario (15% confidence)
**All models fail (ACF > 0.6, coverage < 70%):**
- Our model classes are fundamentally wrong
- Possible causes: Wrong distribution family, more complex temporal structure, unobserved covariates
- Action: Return to EDA, consider ARMA models, Conway-Maxwell-Poisson, or admit defeat

---

## Key Parameters to Watch

### State-Space Model
- **δ (drift):** Should be positive (0.03-0.08), consistent growth
- **σ²_η (innovation variance):** Should be small (< 0.1) if drift captures trend
- **φ (dispersion):** Expect 10-20, much less than naive estimate

### Changepoint Model
- **τ (location):** Should be near 0.3 with SD < 0.3 if real
- **β_2 (level shift):** Should exclude 0 if true regime change
- **β_3 (slope change):** Tests if growth rate differs pre/post

### Gaussian Process Model
- **ℓ (lengthscale):** Key interpretability parameter
  - ℓ < 0.2: Very wiggly, possible overfitting
  - ℓ = 0.3-1.0: Appropriate smoothness
  - ℓ > 2.0: Nearly linear, GP is overkill
- **α (marginal SD):** Deviation from mean trend

---

## Red Flags (Stop and Diagnose)

**Computational:**
- R-hat > 1.05 after 10k iterations
- Divergent transitions > 5% even with adapt_delta=0.99
- ESS < 100 for key parameters

**Statistical:**
- φ → 0 (contradicts overdispersion in EDA)
- φ → ∞ (distribution family wrong)
- All models LOO-ELPD < -200 (barely better than random)
- Prior-posterior overlap > 80% (data not informative)

**Substantive:**
- All models show ACF > 0.6 (none capture temporal structure)
- Predictions wildly inconsistent with data range
- Posterior predictive p-values < 0.01 or > 0.99

---

## Stan Code Templates

**Location:** See Appendix in `proposed_models.md`

All three models include:
- Non-centered parameterizations (for MCMC efficiency)
- `generated quantities` block with:
  - `log_lik` vector for LOO-CV
  - `C_rep` array for posterior predictive checks
- Appropriate priors based on EDA findings

**Estimated runtime:** 2-5 minutes per model on standard laptop

---

## Contact/Questions

This design is complete and ready for implementation. Key decision points:

1. **If State-Space wins cleanly (ΔLOO > 10):**
   - Report drift and innovation parameters
   - Emphasize time series nature of data
   - One-step-ahead predictions should be excellent

2. **If Changepoint wins:**
   - Report changepoint location and regime parameters
   - Investigate domain context (what happened at t≈0.3?)
   - Implications for future: could another break occur?

3. **If GP wins:**
   - Extract functional form from posterior
   - Try to fit parametric model post-hoc
   - Emphasize smooth but complex growth

4. **If ambiguous or all fail:**
   - Use Bayesian Model Averaging
   - Report uncertainty honestly
   - Recommend additional data or alternative approaches

---

**Prepared by:** Designer 2 (Temporal Structure Specialist)
**Full Specifications:** `/workspace/experiments/designer_2/proposed_models.md`
**Quick Reference:** `/workspace/experiments/designer_2/design_summary.md`
**Status:** READY FOR IMPLEMENTATION
