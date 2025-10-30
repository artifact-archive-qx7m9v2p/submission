# Model Improvement Priorities: Experiment 1

**Model**: Fixed Changepoint Negative Binomial Regression (Simplified)
**Status**: ACCEPTED with documented limitations
**Date**: 2025-10-29

---

## Overview

While Experiment 1 is **adequate for its primary purpose** (testing structural break hypothesis), several improvements would enhance the model for broader applications. This document prioritizes refinements by **impact**, **feasibility**, and **necessity**.

---

## Priority 1: Add AR(1) Autocorrelation Structure

**Priority Level**: HIGH
**Status**: CRITICAL for publication-quality analysis
**Effort**: MODERATE (60-90 minutes)
**Impact**: HIGH (resolves primary limitation)

### Problem

**Current**: Residual ACF(1) = 0.519 (exceeds 0.5 threshold)
**Issue**: Model treats observations as conditionally independent, ignoring temporal dependencies
**Consequence**: Parameter uncertainty understated, prediction intervals too wide, falsification criterion #2 violated

### Proposed Solution

Implement full AR(1) error structure:

```
ε_t ~ Normal(ρ × ε_{t-1}, σ_ε)  for t > 1
ε_1 ~ Normal(0, σ_ε / √(1 - ρ²))  # stationary initialization

log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > 17) × (year_t - year_17) + ε_t
```

**Priors**:
- ρ ~ Beta(12, 1) → E[ρ] = 0.923 (informed by EDA ACF(1) = 0.944)
- σ_ε ~ Exponential(2) → E[σ_ε] = 0.5

### Implementation Path

**Option A: CmdStan (Recommended)**
1. Install CmdStan with system build tools
   ```bash
   pip install cmdstanpy
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```
2. Use existing Stan model at `/workspace/experiments/experiment_1/simulation_based_validation/code/model.stan`
3. Fit with `cmdstanpy.CmdStanModel()`
4. Time: ~60 minutes (including installation)

**Option B: PyMC with Workaround**
1. Use manual loop construction (avoid `pm.scan`)
2. Pre-compute AR(1) structure in transformed data
3. Less elegant but functional
4. Time: ~90 minutes (requires custom implementation)

### Expected Improvements

| Metric | Current | Expected with AR(1) |
|--------|---------|---------------------|
| Residual ACF(1) | 0.519 | < 0.3 |
| PPC ACF p-value | 0.000 | > 0.05 |
| 90% Coverage | 100% (too wide) | ~90% (calibrated) |
| Falsification criteria | 5/6 pass | 6/6 pass |

### Validation Plan

After AR(1) implementation:
1. **Convergence**: Check R̂, ESS, divergences
2. **Residual ACF**: Should be < 0.3 for all lags
3. **PPC**: ACF(1) test should pass (p ∈ [0.05, 0.95])
4. **LOO comparison**: elpd_loo should improve vs. simplified model
5. **Parameter changes**: β₂ should remain clearly positive (conclusion robust)

### Why This Matters

- **Falsification criterion**: Currently violates residual ACF threshold
- **Scientific integrity**: Temporal dependencies are real, should be modeled
- **Publication quality**: Reviewers will expect AR structure for time series
- **Uncertainty calibration**: Current intervals likely anti-conservative
- **Forecasting**: Essential if model used for prediction (currently inappropriate)

---

## Priority 2: Sensitivity Analysis on Changepoint Location

**Priority Level**: MEDIUM
**Status**: RECOMMENDED for robustness
**Effort**: LOW (30 minutes)
**Impact**: MODERATE (quantifies changepoint uncertainty)

### Problem

**Current**: Changepoint τ = 17 specified from EDA, not estimated
**Issue**: Uncertainty in changepoint timing not propagated to inferences
**Consequence**: If true changepoint at t=16 or t=18, conclusions might shift

### Proposed Solution

Fit models with alternative changepoint locations:
- τ ∈ {15, 16, 17, 18, 19}
- Compare LOO-CV across specifications
- Test if β₂ remains clearly positive across alternatives

### Implementation Path

1. **Copy and modify model code** for each τ
2. **Fit 5 models** (1 per changepoint location)
3. **Extract LOO and β₂ estimates** for each
4. **Compare**:
   - Which τ has best LOO (lowest elpd_loo)?
   - Does β₂ exclude zero for all reasonable τ?
   - How sensitive are estimates to ±1 observation shift?

### Expected Results

Based on EDA (strong evidence for τ=17):
- τ=17 likely has best LOO
- τ=16, τ=18 probably similar, slightly worse
- τ=15, τ=19 likely noticeably worse
- β₂ should remain clearly positive across all

### Why This Matters

- **Robustness**: Demonstrates conclusion not artifact of τ choice
- **Transparency**: Acknowledges τ was chosen, not estimated
- **Scientific rigor**: Tests sensitivity to key assumption
- **Reviewers**: Will likely ask "why t=17?"

### Effort-Benefit Analysis

**Effort**: LOW
- Just re-run existing code with different τ
- 5 fits × ~6 minutes = 30 minutes total

**Benefit**: MODERATE
- Strengthens robustness claim
- Quantifies changepoint uncertainty
- Addresses obvious reviewer question
- Low cost, decent value

---

## Priority 3: Investigate Dispersion Parameterization

**Priority Level**: LOW
**Status**: OPTIONAL clarification
**Effort**: LOW (15-20 minutes)
**Impact**: LOW (technical cleanup, no substantive change)

### Problem

**Current**: PyMC α = 5.41 vs. EDA α ≈ 0.61
**Issue**: Parameterization confusion (PyMC α vs. traditional φ = 1/α)
**Consequence**: Unclear what dispersion parameter means, variance predictions wrong

### Proposed Solution

**Clarify parameterization**:
1. PyMC: `NegativeBinomial(mu, alpha)` → variance = μ + μ²/α
2. Traditional: φ parameterization → variance = μ + φμ²
3. Relationship: φ = 1/α

**Verify**:
- If α = 5.41, then φ = 1/5.41 ≈ 0.185
- EDA estimated φ ≈ 0.61 (higher dispersion)
- Model fitted lower dispersion than EDA expected

### Investigation Path

1. **Simulate data** with known dispersion
2. **Fit PyMC model** to simulated data
3. **Check**: Does recovered α match true φ via φ = 1/α?
4. **Conclusion**: Document correct interpretation

### Why This Matters (or Doesn't)

**Matters**:
- Technical correctness
- Interpretation of dispersion parameter
- Variance/mean ratio predictions

**Doesn't affect**:
- Mean structure (β₀, β₁, β₂)
- Structural break conclusion
- Regime change inference

**Verdict**: Low priority, but easy fix for cleaner documentation

---

## Priority 4: Prior Sensitivity Analysis

**Priority Level**: LOW
**Status**: OPTIONAL validation
**Effort**: LOW (20-30 minutes)
**Impact**: LOW (priors already weakly informative)

### Problem

**Current**: Single prior specification used
**Issue**: Haven't tested if conclusions robust to prior choice
**Concern**: What if β₂ > 0 only because of informative prior?

### Proposed Solution

Re-fit model with alternative priors for β₂:
1. **Current**: β₂ ~ Normal(0.85, 0.5)
2. **Weakly informative**: β₂ ~ Normal(0, 1)
3. **Skeptical**: β₂ ~ Normal(0, 0.3) [favors no change]
4. **Uniform**: β₂ ~ Uniform(-2, 2)

### Expected Results

Based on strong data signal (posterior mean 0.556, far from prior mean 0.85):
- All priors should yield β₂ > 0
- Posteriors should be similar across priors
- Confirms finding is data-driven, not prior-driven

### Why This Matters (or Doesn't)

**Good to check**:
- Scientific best practice
- Demonstrates objectivity
- Rules out prior dependence

**Not urgent because**:
- Posterior already shows substantial learning from prior
- β₂ posterior (0.556) differs from prior mean (0.85)
- Prior-posterior plots show clear data update
- LOO shows low effective parameters (p_loo = 0.98)

**Verdict**: Low priority, but quick and scientifically responsible

---

## Priority 5: Implement Unknown Changepoint Model

**Priority Level**: VERY LOW
**Status**: FUTURE WORK (research extension)
**Effort**: HIGH (2-3 hours)
**Impact**: HIGH for methodology, LOW for current conclusions

### Problem

**Current**: τ = 17 fixed based on EDA
**Limitation**: Cannot estimate changepoint or quantify its uncertainty
**Consequence**: Assumption that changepoint known exactly

### Proposed Solution

**Option A: Discrete Unknown Changepoint**
```
τ ~ Categorical(p_1, ..., p_N)
log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > τ) × (year_t - year_τ)
```

**Option B: Smooth Transition Function**
```
w_t = 1 / (1 + exp(-k × (t - τ)))  # logistic transition
log(μ_t) = β_0 + β_1 × year_t + β_2 × w_t × year_t
```

**Option C: Multiple Changepoints**
```
τ_1, τ_2, ... ~ Prior
log(μ_t) = β_0 + Σ β_j × I(t > τ_j) × (year_t - year_τj)
```

### Why Not Now

**Reasons to defer**:
- Complex implementation (non-standard MCMC, label switching)
- EDA provided strong evidence for specific τ=17
- Sensitivity analysis (Priority 2) addresses this more simply
- Current model adequate for present research question
- This is research-level extension, not validation step

**When to pursue**:
- If reviewer requests unknown changepoint estimation
- If extending analysis to multiple datasets with unknown breaks
- If research question shifts to changepoint detection methodology
- As follow-up publication on changepoint methods

**Verdict**: Interesting methodologically, not necessary for current work

---

## Improvements NOT Recommended

### 1. Switch to Poisson Likelihood

**Reason to avoid**: EDA definitively ruled out Poisson (ΔAIC = +2417)
**Verdict**: Do not pursue

### 2. Remove Changepoint (Polynomial Only)

**Reason to avoid**: EDA showed two-regime model 80% better than cubic polynomial
**Verdict**: Would be regression, not improvement

### 3. Different Link Function (e.g., Identity)

**Reason to avoid**:
- Log link appropriate for count data
- Exponential growth pattern in data
- Identity link could produce negative predictions

**Verdict**: Log link is correct choice

### 4. Homogeneous Variance Assumption

**Reason to avoid**: Data show clear heteroscedasticity, Negative Binomial needed
**Verdict**: Current approach is appropriate

---

## Summary Recommendation Matrix

| Improvement | Priority | Effort | Impact | Recommended Timing |
|-------------|----------|--------|--------|-------------------|
| **AR(1) structure** | HIGH | Moderate | High | Before publication |
| **Changepoint sensitivity** | MEDIUM | Low | Moderate | Soon (30 min task) |
| **Dispersion clarification** | LOW | Low | Low | Optional, easy win |
| **Prior sensitivity** | LOW | Low | Low | Optional, good practice |
| **Unknown changepoint** | VERY LOW | High | High* | Future research |

*High impact for methodology, low for current conclusions

---

## Implementation Timeline

### Immediate (Before Experiment 2)
- None required; current model adequate for comparison

### Short-Term (After Experiment 2, Before Decision)
- **Changepoint sensitivity** (30 min) - easy robustness check

### Medium-Term (Before Publication)
- **AR(1) implementation** (60-90 min) - essential for publication quality
- **Prior sensitivity** (30 min) - good scientific practice
- **Dispersion clarification** (20 min) - clean documentation

### Long-Term (Future Research)
- **Unknown changepoint model** (2-3 hours) - methodological extension
- **Multiple changepoints** (research project) - if data support

---

## Resource Requirements

### For AR(1) Implementation (Priority 1)

**Software**:
- CmdStan (requires C++ compiler, make)
- OR PyMC with custom AR(1) code

**Time**:
- Installation: 30 minutes
- Implementation: 30 minutes
- Validation: 30 minutes
- Total: ~90 minutes

**Skills**:
- Stan programming (model already written)
- OR PyMC custom distributions
- MCMC diagnostics interpretation

**Output**:
- Updated posterior inference
- Improved residual diagnostics
- Better calibrated uncertainties

### For Other Priorities

**Changepoint sensitivity**: Just computing resources (5 model fits)
**Dispersion investigation**: Minimal (simulation + documentation)
**Prior sensitivity**: Computing resources (3-4 model fits)
**Unknown changepoint**: Advanced MCMC expertise, research time

---

## Success Metrics

### Priority 1 (AR(1)) Success Criteria

Model should achieve:
- ✓ Residual ACF(1) < 0.3
- ✓ PPC ACF test: p-value ∈ [0.05, 0.95]
- ✓ 90% coverage ≈ 90% (±5%)
- ✓ All falsification criteria pass (6/6)
- ✓ LOO improvement over simplified model
- ✓ β₂ still clearly positive (conclusion robust)

### Priority 2 (Sensitivity) Success Criteria

Analysis should show:
- ✓ τ=17 has best or near-best LOO
- ✓ β₂ > 0 for all reasonable τ (15-19)
- ✓ Estimates stable within ±1 observation
- ✓ Conclusion documented as robust

### Priorities 3-4 Success Criteria

Documentation should clarify:
- ✓ Dispersion parameterization correct and explained
- ✓ Prior sensitivity demonstrated
- ✓ Conclusions shown to be data-driven

---

## Decision Impact

**If Priority 1 (AR(1)) completed**:
- Model moves from "ACCEPT with conditions" to "ACCEPT unconditionally"
- Publication-ready quality achieved
- All falsification criteria pass
- No remaining critical limitations

**If Priority 2 (Sensitivity) completed**:
- Robustness claim strengthened
- Reviewer concerns preemptively addressed
- Scientific rigor demonstrated

**If Priorities 3-4 completed**:
- Technical documentation cleaner
- Methodological transparency enhanced
- Minor uncertainties resolved

**If Priority 5 pursued**:
- Major research contribution
- Methodological advancement
- Separate publication potential

---

## Conclusion

**Most critical**: AR(1) implementation (Priority 1)
**Quick wins**: Changepoint sensitivity (Priority 2)
**Optional**: Dispersion clarification, prior sensitivity (Priorities 3-4)
**Future**: Unknown changepoint estimation (Priority 5)

**For current analysis**: Model adequate as-is, improvements enhance but don't fundamentally change conclusions

**For publication**: Priority 1 (AR(1)) strongly recommended, Priority 2 (sensitivity) adds value

**For workflow**: Proceed to Experiment 2 with current model, implement improvements after model selection

---

**Document prepared**: 2025-10-29
**Status**: Improvement roadmap defined, priorities clear
