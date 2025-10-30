# Model Designer 2: Temporal Correlation Structures

**Designer Focus**: Bayesian models with explicit temporal dependence
**Key Innovation**: Handling ACF = 0.971 through different correlation mechanisms
**Date**: 2025-10-29

---

## Executive Summary

This designer proposes **three Bayesian models** that explicitly address the extremely high temporal autocorrelation (ACF = 0.971) observed in the time series data. The central scientific question is:

> **Is the high ACF genuine stochastic dependence, or merely an artifact of a smooth deterministic trend?**

Each model represents a different hypothesis about the temporal structure:

1. **NB-AR1**: Stationary autoregressive process (mean-reverting)
2. **NB-GP**: Gaussian process (flexible smooth trend)
3. **NB-RW**: Random walk (non-stationary, integrated process)

---

## Document Overview

### 1. `proposed_models.md` (PRIMARY DOCUMENT)
**Purpose**: Complete theoretical and practical specifications for all three models

**Contents**:
- Theoretical justifications for each model
- Full mathematical specifications
- Prior specifications with detailed rationale
- Falsification criteria (when to reject each model)
- Stan implementation notes
- Computational cost estimates

**Key sections**:
- Model 1: NB-AR1 (pages 1-15)
- Model 2: NB-GP (pages 16-28)
- Model 3: NB-RW (pages 29-38)
- Model comparison framework
- Critical decision points
- Stress tests

### 2. `stan_implementation_guide.md` (TECHNICAL REFERENCE)
**Purpose**: Complete Stan code with troubleshooting

**Contents**:
- Full Stan code for all three models
- Python interface code (cmdstanpy)
- Posterior predictive check examples
- Troubleshooting guide (divergences, low ESS, etc.)
- Computational benchmarks
- Success checklist

**Use this for**: Actual implementation and debugging

### 3. `model_selection_framework.md` (DECISION TOOL)
**Purpose**: Systematic approach to choosing between models

**Contents**:
- Decision tree for model selection
- Quantitative criteria (LOO, Pareto-k, ACF)
- Qualitative criteria (smoothness, interpretability)
- Special tests for each model class
- Model averaging strategy
- Reporting templates for different scenarios

**Use this for**: Final model selection and reporting

### 4. `README.md` (THIS FILE)
**Purpose**: Navigation and quick reference

---

## Quick Start

### For Model Implementers

1. **Read**: `proposed_models.md` sections on your assigned model
2. **Code**: Use Stan code from `stan_implementation_guide.md`
3. **Debug**: Refer to troubleshooting section if issues arise
4. **Validate**: Run all checks in the success checklist

### For Model Evaluators

1. **Fit**: All three models + baseline (independent errors)
2. **Compare**: Use criteria in `model_selection_framework.md`
3. **Decide**: Follow decision tree
4. **Report**: Use appropriate template based on outcome

---

## Key Design Decisions

### 1. Why These Three Models?

**AR(1)**:
- Standard approach in time series literature
- Assumes stationarity (ρ < 1)
- Most likely to be identifiable with n=40

**GP**:
- Tests if high ACF is just smooth trend
- No parametric assumptions
- Most flexible but most complex

**RW**:
- Natural limit as ρ → 1
- Explicit non-stationarity
- Simplest of the three to implement

**Not included**: Higher-order AR(p), ARMA, structural breaks
- Reason: n=40 is too small to reliably estimate more complex structures
- These can be explored if initial models fail

### 2. Prior Specifications

All priors are **weakly informative** with justification from EDA:

**Intercept (β₀)**: Normal(log(109.4), 1)
- Centered at observed mean
- Allows ±2.7× deviation on count scale

**Growth (β₁)**: Normal(1.0, 0.5)
- Positive growth expected
- Allows wide range [0, 2] on standardized scale

**Dispersion (α)**: Gamma(2, 0.1)
- Favors overdispersion (E[α] = 20)
- Consistent with Var/Mean = 70

**AR correlation (ρ)**: Beta(20, 2)
- Informed by ACF = 0.971
- Centers at 0.91, allows [0.7, 0.99]
- **Key decision**: Strong prior, but allows data to override

**GP lengthscale (ℓ)**: InverseGamma(5, 5)
- Centers at ~1.25 (long correlation)
- Allows [0.5, 5] range

**RW innovation (σ_ω)**: Exponential(10)
- Small steps expected (E = 0.1)
- Allows larger if data requires

### 3. Falsification Criteria

Each model has explicit **rejection criteria**:

**Reject AR(1) if**:
- ρ posterior at boundary (> 0.99)
- Residual ACF remains high
- Computational pathologies (divergences, low ESS)

**Reject GP if**:
- Lengthscale → ∞ (constant function)
- No improvement over polynomial trend
- Numerical instability (Cholesky failures)

**Reject RW if**:
- σ_ω → 0 (deterministic trend)
- First differences show autocorrelation (not a RW)
- Variance growth inconsistent with RW behavior

**Reject ALL temporal models if**:
- Baseline (independent errors) is within ΔELPD < 2
- Posterior predictive checks fail systematically
- Computational problems persist across all models

---

## Critical Insights

### The Identifiability Problem

With ACF = 0.971, it's **extremely difficult** to separate:
- Smooth deterministic trend
- High stochastic correlation

**Implication**: We may conclude "data cannot distinguish structures" - **this is a valid scientific conclusion**, not a failure.

### The n=40 Challenge

Sample size of 40 is:
- **Sufficient** for basic parameter estimation
- **Marginal** for complex correlation structures
- **Insufficient** for structural breaks, regime switching, etc.

**Implication**: Simpler models (AR1, RW) are more likely to be identifiable than complex models (GP with many hyperparameters).

### The Trend vs Correlation Trade-off

**If we specify trend flexibly** (quadratic, cubic):
- Less need for correlation structure
- Risk: overfitting with too many trend terms

**If we specify trend simply** (linear):
- More need for correlation to explain deviations
- Risk: confounding trend with correlation

**Solution**: Compare models with different trend specifications

---

## Expected Scenarios and Recommendations

### Scenario A: AR(1) Wins
**Posterior**: ρ = 0.94 [0.91, 0.97], ΔELPD = 12 vs baseline

**Interpretation**: Genuine stationary stochastic dependence

**Recommendation**: Report AR(1) as primary model, discuss implications for uncertainty quantification

### Scenario B: GP Wins
**Posterior**: ℓ = 0.8 [0.5, 1.5], ΔELPD = 15 vs baseline

**Interpretation**: High ACF reflects flexible smooth trend, not stochastic correlation

**Recommendation**: Report GP, but also fit quadratic trend model to see if simpler parametric form suffices

### Scenario C: RW Wins
**Posterior**: σ_ω = 0.15 [0.10, 0.22], ΔELPD = 10 vs baseline

**Interpretation**: Process is non-stationary (unit root)

**Recommendation**: Report RW, acknowledge challenges for long-term forecasting, consider stationarity tests

### Scenario D: Models Indistinguishable
**ELPD differences**: All within 2 SE

**Interpretation**: Data (n=40) cannot distinguish temporal structures

**Recommendation**: Use model averaging with stacking weights, emphasize structural uncertainty

### Scenario E: Temporal Structure Unnecessary
**ELPD**: Baseline within 2 of best temporal model

**Interpretation**: High ACF is purely trend artifact

**Recommendation**: Use simpler model with independent errors, discuss why ACF was misleading

---

## Red Flags and Contingencies

### Red Flag 1: All Models Have Poor Diagnostics
**Symptoms**: Divergences, Rhat > 1.05, ESS < 100

**Action**:
1. Try reparameterizations (in `stan_implementation_guide.md`)
2. If persists: Models are too complex for data
3. Fallback: Fit simpler models (e.g., AR1 with weak prior)

### Red Flag 2: Posterior Predictive Checks Fail
**Symptoms**: Systematic deviations in ACF, time series plots

**Action**:
1. Check if structural break is present (EDA noted year = -0.21)
2. Consider segmented models
3. May need to revisit model class entirely

### Red Flag 3: Parameters at Boundaries
**Symptoms**: ρ → 1, ℓ → ∞, σ_ω → 0

**Action**:
1. Model is collapsing to simpler form
2. Fit that simpler form explicitly
3. Use model comparison to confirm

### Red Flag 4: Out-of-Sample Predictions Terrible
**Symptoms**: Hold-out predictions far worse than in-sample

**Action**:
1. Models are overfitting
2. Use stronger regularization (tighter priors)
3. Consider model averaging for robustness

---

## Next Steps After Model Fitting

### 1. Convergence Diagnostics (CRITICAL)
- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 (bulk and tail)
- [ ] No divergent transitions (< 0.1%)
- [ ] Trace plots show mixing

**If any fail**: DO NOT PROCEED until resolved

### 2. Posterior Predictive Checks
- [ ] Time series overlay (observed vs replicated)
- [ ] ACF comparison (observed vs posterior predictive)
- [ ] Residual diagnostics (should be white noise)
- [ ] Rootogram or histogram comparison

**If any fail**: Model is misspecified

### 3. Model Comparison
- [ ] PSIS-LOO for all models
- [ ] Pareto-k diagnostics (< 0.7)
- [ ] Compare ELPD with SE bands
- [ ] Apply decision tree from `model_selection_framework.md`

### 4. Sensitivity Analysis
- [ ] Refit with skeptical priors (weaker correlation)
- [ ] Refit with informed priors (stronger correlation)
- [ ] Check posterior stability

### 5. Reporting
- [ ] Choose template from `model_selection_framework.md`
- [ ] Report selected model with justification
- [ ] Acknowledge limitations (n=40, structural uncertainty)
- [ ] Provide model-averaged predictions if appropriate

---

## Connection to Other Designers

### Designer 1 (Hierarchical Models)
**Potential overlap**: If Designer 1 proposes group-level effects

**Resolution strategy**:
- Fit both: hierarchical + temporal correlation
- Compare: Are group effects OR temporal correlation sufficient?
- Test: Does hierarchical structure explain temporal dependence?

**Likely outcome**: Probably won't need both (n=40 is small)

### Designer 3 (Alternative Specifications)
**Potential overlap**: If Designer 3 proposes measurement error models

**Resolution strategy**:
- Temporal correlation vs observation error: Similar in some ways
- Can combine: Latent AR(1) + observation error
- Test: Identifiability with n=40 is questionable

**Likely outcome**: Choose one or the other, not both

---

## Key References

### Methodological
- **Zeger & Qaqish (1988)**: Markov regression models for time series - Original AR(1) for counts
- **Durbin & Koopman (2012)**: Time Series Analysis by State Space Methods - RW and state-space models
- **Rasmussen & Williams (2006)**: Gaussian Processes for Machine Learning - GP theory and practice

### Bayesian Workflow
- **Vehtari et al. (2017)**: Practical Bayesian model evaluation using LOO-CV - PSIS-LOO methodology
- **Gelman et al. (2020)**: Bayesian Workflow - Prior/posterior predictive checks

### Ecological Applications
- **Morris & Doak (2002)**: Quantitative Conservation Biology - AR(1) in population dynamics
- **Dennis et al. (2006)**: Estimating density dependence - Stochastic vs deterministic trends

---

## Summary: Design Philosophy

This designer's approach is characterized by:

1. **Multiple competing hypotheses**: AR(1) vs GP vs RW represent fundamentally different assumptions
2. **Explicit falsification**: Each model has clear rejection criteria
3. **Honest uncertainty**: If models are indistinguishable, say so
4. **Practical focus**: All models implementable with Stan/PyMC
5. **Scientific skepticism**: High ACF might be artifact, not real

**Core principle**: The goal is to **discover the truth about temporal structure**, not to force a complex model onto the data.

---

## File Locations

```
/workspace/experiments/designer_2/
├── README.md                          # This file (navigation)
├── proposed_models.md                 # Main theoretical document
├── stan_implementation_guide.md      # Technical implementation
└── model_selection_framework.md      # Decision framework
```

**Primary document**: `proposed_models.md` (start here)

**For implementation**: `stan_implementation_guide.md`

**For final decision**: `model_selection_framework.md`

---

## Contact / Questions

**Designer**: Model Designer 2 (Temporal Correlation Specialist)

**Key strength**: Explicit modeling of temporal dependence structures

**Key limitation**: n=40 may be too small to reliably distinguish complex correlation forms

**Design complete**: 2025-10-29

**Ready for**: Implementation and model fitting phase

---

**End of README**
