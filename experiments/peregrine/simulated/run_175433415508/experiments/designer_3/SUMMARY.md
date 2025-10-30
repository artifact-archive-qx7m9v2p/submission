# DESIGNER 3 SUMMARY - Non-Linear and Hierarchical Models

**Role**: Complexity and Structural Change Specialist
**Date**: 2025-10-29
**Status**: ✓ Complete and Ready for Implementation

---

## Quick Reference

| Model | Key Innovation | Extra Parameters | LOO Threshold | Runtime | Likely Outcome |
|-------|---------------|------------------|---------------|---------|----------------|
| **Baseline** | Linear + AR(1) | 0 (reference) | N/A | 2 min | **60% chance WINNER** |
| **Model 1** | Quadratic term | +1 (β₂) | >4 SE | 2-4 min | 30% chance |
| **Model 2** | Changepoint | +2 (τ, β₂) | >6 SE | 4-8 min | 8% chance |
| **Model 3** | Gaussian Process | +42 (ℓ, α, f) | >10 SE | 10-20 min | 2% chance |

---

## Three Models Proposed

### 1. Quadratic Negative Binomial + AR(1)
```
log(μ_t) = β₀ + β₁·year_t + β₂·year_t² + ε_t
```

**Why**: EDA shows quadratic R²=0.964 vs exponential R²=0.937 (2.7% better)

**Skepticism**: With n=40, this may be overfitting. Must prove value via LOO.

**Reject if**: β₂ ≈ 0 OR ΔLOO < 4 SE vs linear

**File**: `model_1_quadratic.stan`

---

### 2. Bayesian Changepoint + AR(1)
```
log(μ_t) = β₀ + β₁·year_t + β₂·I(year_t > τ)·(year_t - τ) + ε_t
τ ~ Uniform(-1.5, 1.5)
```

**Why**: EDA detected possible break at year=-0.21, growth rate changes 9.6-fold

**Skepticism**: Single-analyst finding, may be spurious. Estimating τ with n=40 is hard.

**Reject if**: τ posterior uniform OR β₂ ≈ 0 OR ΔLOO < 6 SE

**File**: `model_2_changepoint.stan`

---

### 3. Gaussian Process Negative Binomial
```
log(μ_t) = β₀ + β₁·year_t + f(year_t)
f ~ GP(0, K), K = α²·exp(-dist²/(2ℓ²))
```

**Why**: Stress test for parametric assumptions, let data determine shape

**Skepticism**: n=40 is too small for GP. Likely computational failure or overfitting.

**Reject if**: ℓ→0 OR ℓ→∞ OR Cholesky fails OR ΔLOO < 10 SE

**File**: `model_3_gp.stan`

---

## Falsification-First Philosophy

### I Will Declare SUCCESS If:

1. **Linear model wins** (60% expected)
   - All complex models fail LOO comparison
   - Conclusion: EDA non-linearity was noise
   - **This is GOOD - we avoided overfitting**

2. **One complex model clearly wins** (35% expected)
   - ΔLOO exceeds threshold by large margin
   - Passes all diagnostics and stress tests
   - **This is GOOD - we found genuine structure**

3. **All models fail computationally** (5% expected)
   - Non-convergence despite reparameterization
   - Conclusion: Data too noisy for these approaches
   - **This is GOOD - we learned what doesn't work**

### I Will Declare FAILURE If:

1. **Defend complexity without evidence**
   - Report complex model despite ΔLOO < threshold
   - Ignore falsification criteria
   - **I commit to NOT doing this**

2. **Ignore computational warnings**
   - Proceed despite divergences, poor ESS
   - Rationalize away red flags
   - **I commit to abandoning models that fail diagnostics**

---

## Sequential Testing Logic

```
START
  ↓
FIT LINEAR BASELINE (Negative Binomial + AR(1))
  ↓
  Document: LOO, RMSE, posterior predictive checks
  ↓
FIT MODEL 1: QUADRATIC
  ↓
  Check: Is β₂ credible interval excluding zero?
  ├─ NO → STOP, use linear baseline ✓
  └─ YES → Continue
  ↓
  Check: Is ΔLOO > 4 SE vs baseline?
  ├─ NO → STOP, use linear baseline ✓
  └─ YES → Continue
  ↓
FIT MODEL 2: CHANGEPOINT (if still interested)
  ↓
  Check: Is τ posterior concentrated (not uniform)?
  ├─ NO → REJECT Model 2
  └─ YES → Continue
  ↓
  Check: Is β₂ credible interval excluding zero?
  ├─ NO → REJECT Model 2
  └─ YES → Continue
  ↓
  Check: Is ΔLOO > 6 SE vs baseline?
  ├─ NO → Use best model so far
  └─ YES → Model 2 is candidate
  ↓
  Check: Do Models 1-2 show systematic PPC failures?
  ├─ NO → DONE, select best model ✓
  └─ YES → Continue to stress test
  ↓
FIT MODEL 3: GAUSSIAN PROCESS (stress test only)
  ↓
  Check: Did fitting succeed without Cholesky failures?
  ├─ NO → ABANDON GP, use best parametric ✓
  └─ YES → Continue
  ↓
  Check: Is length scale ℓ in reasonable range [0.3, 3]?
  ├─ NO → GP inappropriate, use best parametric ✓
  └─ YES → Continue
  ↓
  Check: Is ΔLOO > 10 SE vs best parametric?
  ├─ NO → Use best parametric ✓
  └─ YES → GP wins (surprising!)
  ↓
DONE: Report best model with full justification
```

---

## Evidence Required for Each Model

### Linear Baseline WINS if:
- All complex models have ΔLOO < threshold
- Simplest model explains data adequately
- **Default position unless proven otherwise**

### Quadratic WINS if:
- β₂ posterior: 90% CI excludes zero
- ΔLOO > 4 SE vs linear
- No divergences, R-hat < 1.01
- Posterior predictive checks pass
- Growth rate acceleration interpretable

### Changepoint WINS if:
- τ posterior concentrated (e.g., 90% CI width < 1.0)
- β₂ posterior: 90% CI excludes zero
- ΔLOO > 6 SE vs linear
- ESS(τ) > 100
- Regime shift scientifically plausible

### Gaussian Process WINS if:
- Length scale ℓ in [0.3, 3]
- ΔLOO > 10 SE vs best parametric
- No Cholesky failures
- Smooth realization, not overfitting noise
- **High bar - unlikely with n=40**

---

## Red Flags by Model

### Model 1 (Quadratic) - ABANDON IF:
- ⚠ Correlation(β₁, β₂) > 0.9 → Non-identifiable
- ⚠ Divergences > 1% despite adapt_delta=0.99
- ⚠ β₂ interval spans zero widely
- ⚠ Posterior predictive: systematic curvature mismatch
- ⚠ Extrapolation produces absurd predictions

### Model 2 (Changepoint) - ABANDON IF:
- ⚠ τ posterior is uniform → No information
- ⚠ τ posterior has >2 modes → Model confused
- ⚠ P(τ at boundary) > 0.3 → Spurious edge detection
- ⚠ ESS(τ) < 50 → Cannot estimate reliably
- ⚠ β₂ interval spans zero

### Model 3 (GP) - ABANDON IF:
- ⚠ Cholesky decomposition fails → Ill-conditioned
- ⚠ Length scale ℓ < 0.3 → White noise
- ⚠ Length scale ℓ > 3 → Collapses to linear
- ⚠ Divergences > 5%
- ⚠ f(year) has >5 inflection points → Overfitting

---

## Key Files and Their Purpose

| File | Purpose | Audience |
|------|---------|----------|
| `experiment_plan.md` | Executive summary | **START HERE** - Synthesis agent |
| `proposed_models.md` | Mathematical details | Implementation team |
| `README.md` | Integration guide | Cross-designer coordination |
| `SUMMARY.md` | Quick reference | **THIS FILE** - Fast overview |
| `model_1_quadratic.stan` | Polynomial implementation | Stan sampler |
| `model_2_changepoint.stan` | Regime shift implementation | Stan sampler |
| `model_3_gp.stan` | Non-parametric implementation | Stan sampler |

---

## Expected Timeline

**Day 1 Morning (2 hours)**:
- Prior predictive checks for all models
- Adjust priors if needed
- Verify Stan programs compile

**Day 1 Afternoon (3 hours)**:
- Fit linear baseline (30 min)
- Fit Model 1 quadratic (30 min)
- **DECISION POINT**: Continue or stop?
- If continue: Fit Model 2 changepoint (1 hour)

**Day 2 (4 hours)**:
- **DECISION POINT**: Is Model 3 justified?
- If yes: Fit Model 3 GP (2 hours)
- Diagnostics for all models (2 hours)

**Day 3 (4 hours)**:
- LOO comparison
- Posterior predictive checks
- Stress tests
- Final report and visualizations

**Total**: 13 hours over 3 days

---

## Integration with Other Designers

### Designer 1 (Baseline Models)
**Relationship**: Designer 3 extends Designer 1's linear baseline

**Comparison**:
- Both propose: Linear Negative Binomial + AR(1)
- Designer 3 adds: Quadratic, changepoint, GP variants
- **Strategy**: Use Designer 1's baseline as reference for all

### Designer 2 (Time Series)
**Relationship**: Complementary focuses

**Comparison**:
- Designer 2: Temporal correlation structures (AR, ARIMA, state-space)
- Designer 3: Non-linear mean functions (polynomial, changepoint, GP)
- **Potential overlap**: Both use AR(1), could combine approaches

### Synthesis Priority
1. Establish baseline (Designer 1's linear model)
2. Test temporal correlation (Designer 2's AR models)
3. Test non-linearity (Designer 3's quadratic/changepoint)
4. Consider combinations if both improve fit

---

## Most Likely Outcome

**Prediction (60% confidence)**:
> "Linear Negative Binomial with AR(1) is the best model. All complex extensions (quadratic, changepoint, GP) failed to justify their added complexity via LOO cross-validation. The EDA finding of quadratic superiority (R²=0.964 vs 0.937) was an artifact of in-sample overfitting with n=40. This is a successful outcome - we avoided overcomplicating the model and found the truth."

**Why this is success**:
- We proposed plausible alternatives based on EDA evidence
- We tested them rigorously with falsification criteria
- We rejected them honestly when data didn't support complexity
- We learned the data are simpler than initially suggested
- We avoided overfitting and produced reliable predictions

**Communication strategy**:
- Lead with: "Rigorous testing revealed linear model is sufficient"
- Not: "Complex models failed" (negative framing)
- Emphasize: Scientific process worked as designed

---

## Computational Environment

**Requirements**:
- Stan 2.33+ or CmdStanPy 1.2+
- Python 3.9+ with numpy, pandas, matplotlib, arviz
- 4GB RAM for Models 1-2, 8GB for Model 3
- 4 CPU cores for parallel chains
- ~30 minutes total runtime

**Recommended configuration**:
```python
import cmdstanpy

model = cmdstanpy.CmdStanModel(stan_file='model_1_quadratic.stan')
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.9,
    max_treedepth=10,
    show_progress=True
)
```

**If divergences appear**:
```python
# Increase adapt_delta sequentially
adapt_delta_values = [0.9, 0.95, 0.99]
for delta in adapt_delta_values:
    fit = model.sample(..., adapt_delta=delta)
    if fit.num_divergences() == 0:
        break
# If still diverging at 0.99 → Model inappropriate
```

---

## Final Checklist

### Before Implementation
- [x] Stan programs written and documented
- [x] Priors justified based on EDA
- [x] Falsification criteria specified
- [x] LOO thresholds defined
- [x] Decision logic documented

### During Implementation
- [ ] Prior predictive checks completed
- [ ] Linear baseline fitted
- [ ] Model 1 fitted (if justified)
- [ ] Model 2 fitted (if justified)
- [ ] Model 3 fitted (if justified)

### After Fitting
- [ ] Convergence diagnostics pass
- [ ] LOO computed for all models
- [ ] Posterior predictive checks completed
- [ ] Stress tests performed
- [ ] Best model selected with justification

### For Reporting
- [ ] Parameter estimates with 90% CI
- [ ] ΔLOO values with SE
- [ ] Interpretation in plain language
- [ ] Limitations documented
- [ ] Falsification evidence presented

---

## Contact Information

**File Location**: `/workspace/experiments/designer_3/`

**Key Contact Points**:
1. Synthesis Agent: Start with `experiment_plan.md`
2. Implementation Team: Use Stan files + `proposed_models.md`
3. Domain Experts: Refer to plain-language sections
4. Quick Questions: This `SUMMARY.md` file

**Philosophy Statement**:
> "I am optimizing for scientific truth, not task completion. If all complex models fail, that's success - we learned the data are simpler than we thought. Abandon complexity unless data demand it."

---

**Designer**: Model Designer 3
**Status**: ✓ Complete and Ready
**Last Updated**: 2025-10-29
**Confidence**: High - all specifications rigorous and falsifiable
