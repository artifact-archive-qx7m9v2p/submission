# Executive Summary - Designer 2 Model Proposals
## Bayesian Time Series Models for Count Data with Extreme Autocorrelation

**Designer:** Model Designer 2 (Temporal Structure & Trend Specification)
**Date:** 2025-10-29
**Status:** Ready for Implementation

---

## The Challenge

**Data:** 40 observations of time series count data showing:
- **Extreme overdispersion:** Variance/Mean = 67.99 (Poisson assumption violated by 68×)
- **Massive autocorrelation:** ACF(1) = 0.989, R²(lag-1) = 0.977
- **Strong growth:** 8.45× increase from 29 to 245
- **Possible regime change:** CUSUM minimum at year ≈ 0.3

**The Core Question:** Is this growth driven by (1) smooth temporal evolution, (2) discrete structural break, or (3) complex nonparametric dynamics?

---

## Three Proposed Bayesian Models

### Model 1: Dynamic State-Space (PRIORITY: ⭐⭐⭐)

**Hypothesis:** The near-perfect autocorrelation (0.989) indicates an autoregressive process - data evolve over time as random walk with drift, not independent observations.

**Structure:**
```
C_t ~ NegativeBinomial(exp(η_t), φ)
η_t = η_{t-1} + δ + β·year_t + ε_t
ε_t ~ Normal(0, σ²_η)
```

**Key Parameters:**
- δ (drift): Expected 0.04-0.08 (4-8% growth per period in log-space)
- σ_η (innovation SD): Expected 0.05-0.12 (small fluctuations around trend)
- φ (dispersion): Expected 10-20 (much less than naive 68 after accounting for temporal correlation)

**Why It Will Win (60% confidence):**
The lag-1 R² of 0.977 means C_t ≈ C_{t-1}, classic autoregressive behavior. This model explicitly captures that structure.

**Abandon If:**
- Innovation variance σ²_η ≈ total variance (no benefit from state decomposition)
- Residual ACF(1) > 0.5 (temporal structure not captured)
- One-step-ahead prediction coverage < 75%

---

### Model 2: Changepoint Regime Shift (PRIORITY: ⭐⭐)

**Hypothesis:** CUSUM analysis shows clear minimum at year ≈ 0.3 with 4.5× mean jump, suggesting discrete structural break.

**Structure:**
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = β_0 + β_1·year + β_2·I(year≥τ) + β_3·(year-τ)·I(year≥τ)
```

**Key Parameters:**
- τ (changepoint): Expected 0.2-0.4 (if well-identified, SD < 0.3)
- β_2 (level shift): Expected 0.4-1.2 (significant discontinuity)
- β_3 (slope change): Expected 0.2-0.8 (different growth rates pre/post)

**Why It Might Win (25% confidence):**
The CUSUM pattern is visually striking, and t-test is highly significant (p < 0.0001). If there was a real intervention or threshold effect, this will capture it.

**Abandon If:**
- τ posterior is uniform (no preferred location)
- β_2 and β_3 both include zero (no regime shift)
- Smooth models fit better (ΔLOO > 10)

---

### Model 3: Gaussian Process Nonparametric (PRIORITY: ⭐⭐)

**Hypothesis:** Neither parametric form is correct - growth is smooth but complex, better represented by flexible function.

**Structure:**
```
C_t ~ NegativeBinomial(exp(f_t), φ)
f ~ GP(β_0 + β_1·year, k(year, year'))
k(year_i, year_j) = α² · exp(-(year_i - year_j)² / (2ℓ²))
```

**Key Parameters:**
- ℓ (lengthscale): Expected 0.3-1.0 (if too small → overfitting, too large → linear)
- α (marginal variance): Expected 0.3-0.8 (deviations from linear trend)
- Function shape: Will reveal whether growth is truly smooth or has discontinuities

**Why It Might Win (15% confidence):**
If both parametric assumptions (random walk, changepoint) are wrong, GP should reveal the truth by fitting better without strong assumptions.

**Abandon If:**
- Lengthscale → ∞ (equivalent to linear, overparameterized)
- Parametric model fits equivalently (Occam's razor favors simpler)
- GP shows discontinuities (suggests true changepoint)

---

## Decision Framework

### Model Selection Process

1. **Fit all three models** in Stan (4 chains × 4000 iterations each)
2. **Check convergence:** R-hat < 1.01, ESS > 400
3. **Compute LOO-ELPD** for model comparison
4. **Apply decision rules:**
   - **ΔLOO > 10:** Clear winner → Validate with diagnostics → Report
   - **ΔLOO < 4:** Equivalent → Use Bayesian Model Averaging
   - **All fail diagnostics (ACF > 0.6, coverage < 70%):** PIVOT to alternative model classes

### Validation Criteria (ALL must pass for winner)

- ✓ **Residual ACF(1) < 0.3** (temporal structure captured)
- ✓ **80% prediction interval coverage: 75-85%** (calibrated uncertainty)
- ✓ **Posterior predictive checks p-values: 0.05-0.95** (captures data features)
- ✓ **Parameters in plausible ranges** (see expected values above)

---

## Expected Outcome (My Prediction)

### Most Likely: Model 1 Wins (60% confidence)

**Expected Results:**
- LOO-ELPD ≈ -150, ΔLOO ≈ 12 over Model 2, ΔLOO ≈ 8 over Model 3
- Drift δ ≈ 0.061 [0.042, 0.078]
- Innovation SD σ_η ≈ 0.076 [0.050, 0.110]
- Dispersion φ ≈ 14.3 [10.2, 21.7]
- Residual ACF(1) ≈ 0.18
- One-step-ahead coverage ≈ 82%

**Scientific Interpretation:**
"The data are best explained by a **random walk with positive drift**. The system grows at approximately **6% per period** in log-space (exponential growth), with small random fluctuations (σ_η ≈ 0.08) around this trend. After accounting for temporal correlation, residual overdispersion is **moderate (φ ≈ 14)**, much less than the unconditional estimate (68). This suggests **most apparent overdispersion is actually temporal correlation**, not intrinsic count variability. Short-term predictions (1-3 steps ahead) should be highly accurate; long-term predictions are exponential with growing uncertainty."

---

## Implementation Resources

### Complete Documentation Suite (100KB, 2224 lines)

1. **`README.md`** (8KB, 224 lines) - START HERE
   - Quick index and model overview
   - 3-minute read

2. **`IMPLEMENTATION_GUIDE.md`** (17KB, 598 lines) - PRACTICAL STEPS
   - Step-by-step instructions with code
   - Troubleshooting guide
   - Expected timeline: 5-7 hours

3. **`design_summary.md`** (10KB, 278 lines) - QUICK REFERENCE
   - Executive summary of three models
   - Falsification criteria
   - Decision trees

4. **`model_comparison_matrix.md`** (13KB, 247 lines) - SIDE-BY-SIDE
   - Detailed comparison tables
   - Expected outcomes for each model
   - Stress test matrix

5. **`proposed_models.md`** (35KB, 877 lines) - FULL SPECIFICATION
   - Complete mathematical specifications
   - Stan code skeletons
   - Detailed priors and rationale
   - Comprehensive falsification strategy

---

## Key Innovations in This Design

### 1. Competing Hypotheses Approach
Rather than proposing a single "best" model, I explicitly test three fundamentally different data generation mechanisms. This forces us to think about **what kind of process could have produced this data**, not just "what model fits well."

### 2. Falsification-First Mindset
Each model has explicit criteria for abandonment. Success is defined as **discovering which models are wrong**, not just fitting data. This prevents confirmation bias and forces honest uncertainty quantification.

### 3. Temporal Structure as Primary Signal
Unlike standard GLM approaches that treat time as just another covariate, these models recognize that **ACF(1) = 0.989 is the dominant signal**. The data are fundamentally a time series, not cross-sectional observations with a trend.

### 4. Overdispersion Decomposition
The naive Var/Mean = 68 conflates temporal correlation with intrinsic overdispersion. The state-space model decomposes this into:
- Systematic temporal evolution (captured by drift)
- Random innovations (σ_η)
- Observation-level overdispersion (φ)

This reveals that **much of the "overdispersion" is actually autocorrelation**.

### 5. Explicit Escape Routes
The design includes:
- **Model variants** if primary versions fail diagnostics
- **Decision trees** for when to pivot model classes
- **Red flags** that trigger stopping rules
- **Stress tests** designed to break assumptions

---

## What Success Looks Like

### Best Case: Clear Winner with Good Diagnostics
- One model has LOO-ELPD > 10 better than others
- All diagnostics pass (ACF < 0.3, coverage 75-85%, R-hat < 1.01)
- Parameters are in expected ranges and scientifically interpretable
- Posterior predictive checks show good fit
- **Action:** Report winner, interpret parameters, make predictions

### Acceptable: Models Equivalent
- All models within LOO-ELPD 4 of each other
- All pass diagnostics reasonably well
- Different mechanistic interpretations but similar predictions
- **Action:** Use Bayesian Model Averaging, emphasize predictive performance over mechanism, report honest uncertainty

### Failure: All Models Poor
- All show residual ACF > 0.6
- Coverage < 70% or LOO-ELPD < -180
- Systematic posterior predictive check failures
- **Action:** This is STILL success if we recognize it! Pivot to alternative model classes, document what doesn't work, recommend different approaches

---

## Critical Insights for Implementation

### What I'm Most Confident About
1. **Negative Binomial is appropriate** (extreme overdispersion rules out Poisson)
2. **Temporal structure must be modeled** (ACF = 0.989 cannot be ignored)
3. **State-Space will likely win** (autoregressive signal is strongest)

### What I'm Uncertain About
1. **Exact value of φ** (confounded with temporal correlation)
2. **Whether changepoint is real or illusory** (could be smooth acceleration)
3. **Whether n=40 is sufficient** (for GP, may be data-hungry)

### What Would Genuinely Surprise Me
1. **If φ < 5:** Would suggest almost all overdispersion is temporal (more extreme than I expect)
2. **If σ²_η > 0.5:** Would suggest state is very noisy (contradicts smooth ACF structure)
3. **If Model 2 wins cleanly:** Would require strong evidence of discrete break

---

## Timeline and Deliverables

### Implementation Phase (5-7 hours)
- **Phase 1:** Implement Stan models (2-3 hours)
- **Phase 2:** Model comparison (1 hour)
- **Phase 3:** Diagnostics (1 hour)
- **Phase 4:** Interpretation (30 min)
- **Phase 5:** Visualization (1 hour)

### Expected Outputs
- **Code:** Three `.stan` files, fitting scripts, diagnostic scripts
- **Results:** LOO comparison table, posterior summaries, diagnostic plots
- **Report:** Comprehensive modeling report with scientific interpretation

---

## Final Thoughts

This design embodies three core principles:

1. **Truth over completion:** If all models fail, that's a valuable discovery
2. **Falsification over confirmation:** Each model has explicit failure criteria
3. **Uncertainty over certainty:** Multiple plausible mechanisms → honest reporting

The goal is not to "finish the modeling task" but to **learn what actually generated this data**. Success means correctly identifying which model class is appropriate, even if that means rejecting our initial proposals.

**The models are ready to implement. The falsification criteria are clear. Let's find out what's true.**

---

**Prepared by:** Model Designer 2 (Temporal Structure Specialist)
**Full Documentation:** `/workspace/experiments/designer_2/`
**Primary Document:** `proposed_models.md` (35KB, complete specifications)
**Implementation Guide:** `IMPLEMENTATION_GUIDE.md` (17KB, step-by-step)
**Status:** READY FOR IMPLEMENTATION

**Contact:** See documents for detailed specifications, troubleshooting, and decision frameworks.
