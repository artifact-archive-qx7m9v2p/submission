# Model Comparison Matrix - Designer 2

## At-a-Glance Comparison

| Feature | Model 1: Michaelis-Menten | Model 2: Power-Law | Model 3: Exponential |
|---------|---------------------------|-------------------|---------------------|
| **Functional Form** | Y_max·x/(K+x) | a + b·x^c | Y_max - Δ·exp(-r·x) |
| **Has Asymptote?** | YES (Y_max) | NO | YES (Y_max) |
| **# Parameters** | 3 (Y_max, K, σ) | 4 (a, b, c, σ) | 4 (Y_max, Δ, r, σ) |
| **Interpretability** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Good |
| **MCMC Difficulty** | ⭐⭐⭐ Moderate | ⭐⭐ Easy | ⭐⭐⭐ Moderate |
| **Extrapolation** | Bounded (safe) | Unbounded (risky) | Bounded (safe) |
| **Recommended For** | Default choice | Uncertain asymptote | Physical equilibration |
| **Priority Rank** | 1st | 2nd | 3rd |

## Parameter Interpretations

### Model 1 (Michaelis-Menten)
- **Y_max:** Maximum possible response (asymptote as x → ∞)
  - *Expected range:* [2.5, 3.0]
  - *Prior:* Normal(2.6, 0.3)
- **K:** Half-saturation constant (x where Y = Y_max/2)
  - *Expected range:* [0.5, 10]
  - *Prior:* Lognormal(log(5), 1)
- **σ:** Residual standard deviation
  - *Expected range:* [0.15, 0.25]
  - *Prior:* HalfNormal(0.25)

**Key insight:** K tells you characteristic scale of saturation. Small K → rapid saturation. Large K → slow saturation.

### Model 2 (Power-Law)
- **a:** Intercept (Y when x=0)
  - *Expected range:* [1.4, 2.0]
  - *Prior:* Normal(1.7, 0.5)
- **b:** Scaling coefficient
  - *Expected range:* [0.2, 1.0]
  - *Prior:* Normal(0.5, 0.5) truncated at 0
- **c:** Power exponent (0 < c < 1 for diminishing returns)
  - *Expected range:* [0.2, 0.7]
  - *Prior:* Beta(2, 2)
  - *Interpretation:* c→0 is log-like, c→1 is linear
- **σ:** Residual standard deviation
  - *Expected range:* [0.15, 0.25]
  - *Prior:* HalfNormal(0.25)

**Key insight:** Posterior c tells you if relationship is strongly diminishing (c<0.3), moderately (0.3<c<0.7), or weakly (c>0.7).

### Model 3 (Exponential)
- **Y_max:** Asymptotic maximum (as x → ∞)
  - *Expected range:* [2.5, 3.0]
  - *Prior:* Normal(2.6, 0.3)
- **Δ (delta):** Total change from x=0 to x=∞ (= Y_max - Y_0)
  - *Expected range:* [0.6, 1.2]
  - *Prior:* Normal(0.9, 0.3)
- **r:** Rate constant (larger r → faster approach to asymptote)
  - *Expected range:* [0.05, 1.0]
  - *Prior:* Exponential(0.5)
- **σ:** Residual standard deviation
  - *Expected range:* [0.15, 0.25]
  - *Prior:* HalfNormal(0.25)

**Key insight:** At x=1/r, 63% of total change is complete. At x=3/r, 95% complete.

## Strengths and Weaknesses

### Model 1: Michaelis-Menten
**Strengths:**
- Most interpretable parameters (Y_max, K have direct meaning)
- Widely used in biology, pharmacology, economics
- Bounded predictions (can't exceed Y_max)
- Two-parameter mean function (parsimonious)

**Weaknesses:**
- K and Y_max can be correlated (funnel geometry)
- May have divergences if data doesn't truly saturate
- Requires reparameterization (log_K) for good sampling
- Assumes specific saturation form (may not fit all data)

**When it fails:**
- Posterior Y_max < max(Y_observed) → asymptote below data
- Posterior K > 20 → saturation beyond observed range
- Persistent divergences → geometry incompatible with data

### Model 2: Power-Law
**Strengths:**
- Very flexible (nests log and linear as special cases)
- Easy to sample (no funnel geometry)
- Can test whether saturation exists via posterior c
- No assumption about asymptote

**Weaknesses:**
- Four parameters (more complex than Model 1)
- Unbounded growth (Y → ∞ as x → ∞)
- Less mechanistically interpretable
- c may be weakly identified with N=27

**When it fails:**
- Posterior c near 0 or 1 → reduce to simpler model
- Extrapolation predicts Y > 3.5 → implausible
- Posterior c spans [0.1, 0.9] → completely uncertain

### Model 3: Exponential
**Strengths:**
- Mechanistically motivated (equilibration process)
- Bounded predictions
- Rate constant r has clear interpretation
- Can derive half-saturation point for comparison with MM

**Weaknesses:**
- Four parameters (more complex than Model 1)
- r and Δ can be correlated
- Less commonly used for cross-sectional data
- Very similar to MM in practice (likely similar fits)

**When it fails:**
- Posterior r near 0 → no saturation
- Posterior Δ near 0 → no change with x
- Half-saturation > 50 → saturation too slow to identify

## Expected Posterior Distributions (Predictions)

Based on EDA findings (R²≈0.82-0.83 for nonlinear fits, Y∈[1.71, 2.63], x∈[1.0, 31.5]):

### Model 1 (Michaelis-Menten)
```
Y_max: 2.62 (95% CI: [2.50, 2.75])
K:     3.5  (95% CI: [1.2, 8.5])   [wide due to sparse high-x data]
σ:     0.19 (95% CI: [0.15, 0.24])
```

**Rationale:** EDA fitted MM gave Y_max=2.59, K=0.64. With Bayesian priors, K will shift toward prior mean (5) but data should pull toward lower values. Wide CI for K expected.

### Model 2 (Power-Law)
```
a: 1.73 (95% CI: [1.55, 1.90])
b: 0.45 (95% CI: [0.30, 0.65])
c: 0.35 (95% CI: [0.20, 0.55])  [moderately log-like]
σ: 0.19 (95% CI: [0.15, 0.24])
```

**Rationale:** c between log (c=0) and sqrt (c=0.5) expected. Data shows strong diminishing returns, so c should be well below 0.7.

### Model 3 (Exponential)
```
Y_max: 2.62 (95% CI: [2.50, 2.75])
Δ:     0.88 (95% CI: [0.70, 1.10])
r:     0.20 (95% CI: [0.08, 0.45])   [wide due to limited x range]
σ:     0.19 (95% CI: [0.15, 0.24])
```

**Rationale:** Y_max similar to Model 1. Δ ≈ Y_max - Y(x=1) ≈ 2.6 - 1.86 = 0.74, but uncertainty about Y_0. Rate r likely <0.5 given gradual saturation.

## Model Selection Strategy

### Stage 1: Fit all three models

**Compare via LOO-CV:**
```
If ΔELPD(Model 1 vs Model 2) > 5:  Strong preference
If ΔELPD(Model 1 vs Model 2) < 2:  Essentially equivalent
```

**Decision rules:**
1. If Models 1 and 3 are equivalent (ΔELPD < 2) → Choose Model 1 (simpler interpretation, fewer parameters)
2. If Model 2 substantially better (ΔELPD > 5) → Use Model 2, but check if posterior c→0 or c→1
3. If all three equivalent → Choose Model 1 (default, most interpretable)

### Stage 2: Posterior predictive checks

**For chosen model:**
- Plot residuals vs x (should be patternless)
- Overlay Y_rep draws on observed data (should envelope data)
- Check test statistics: mean, SD, min, max of Y_rep vs Y_obs

**If checks fail:**
- Try robust likelihood (Student-t instead of Normal)
- Try heteroscedastic variance (σ as function of x)
- Consider GP or spline if functional form too restrictive

### Stage 3: Prior sensitivity

**Refit chosen model with:**
1. Vague priors (SD × 10)
2. Informative priors (SD ÷ 3)

**If posteriors change substantially:**
- Report sensitivity
- Acknowledge prior influence
- Consider collecting more data

**If posteriors stable:**
- Data is informative
- Prior choice robust
- Proceed with confidence

## Falsification Checklist

Before accepting any model, check these red flags:

**Model 1 (MM):**
- [ ] Y_max posterior > max(Y_observed)? ✓ Required
- [ ] K posterior credibly different from 0? ✓ Required
- [ ] Divergences < 1%? ✓ Required
- [ ] LOO Pareto-k < 0.7 for all points? ✓ Required
- [ ] Residuals show no pattern? ✓ Required

**Model 2 (Power-Law):**
- [ ] c posterior excludes 0 and 1? ✓ Desired (if not, simplify)
- [ ] b posterior > 0? ✓ Required
- [ ] Extrapolation to x=100 gives Y < 5? ✓ Plausibility check
- [ ] R-hat < 1.01? ✓ Required
- [ ] Residuals show no pattern? ✓ Required

**Model 3 (Exp):**
- [ ] Y_max posterior > max(Y_observed)? ✓ Required
- [ ] r posterior > 0.01? ✓ Required (else no saturation)
- [ ] Δ posterior > 0? ✓ Required
- [ ] Half-saturation x < 50? ✓ Plausibility check
- [ ] Residuals show no pattern? ✓ Required

**If any required check fails → Model is falsified, try alternatives.**

## Computational Expectations

### Runtime (Stan on modern laptop, 4 chains × 2000 iterations)

- Model 1: ~3 minutes
- Model 2: ~2 minutes
- Model 3: ~3 minutes

**Total time for all three: <10 minutes**

With N=27, Stan will be very fast. Iteration time is negligible.

### MCMC Diagnostics to Monitor

| Diagnostic | Threshold | Interpretation |
|-----------|-----------|----------------|
| R-hat | < 1.01 | Chains converged |
| ESS_bulk | > 400 | Sufficient samples for mean/SD |
| ESS_tail | > 400 | Sufficient samples for quantiles |
| Divergences | 0 | No geometry problems |
| E-BFMI | > 0.3 | Energy transitions smooth |

**If any diagnostic fails:**
1. Increase adapt_delta (0.90 → 0.95 → 0.99)
2. Increase iterations (2000 → 4000)
3. Reparameterize (centered → non-centered, or vice versa)
4. If still failing, model geometry incompatible with data

## Decision Tree Flowchart

```
START
  ↓
Fit Model 1 (Michaelis-Menten)
  ↓
Converged? (R-hat < 1.01, no divergences)
  ├─ NO → Try exponential (Model 3)
  │        ↓
  │      Converged?
  │        ├─ NO → Use power-law (Model 2) [usually easy to fit]
  │        └─ YES → Compare Model 1 vs 3 via LOO
  │                  ↓
  │                Choose better (or Model 1 if tied)
  └─ YES → Check posterior predictive
           ↓
         Passes? (no residual patterns)
           ├─ NO → Try Model 2 or 3
           └─ YES → Check LOO-CV vs Model 2
                    ↓
                  Model 1 better or equivalent?
                    ├─ YES → ACCEPT Model 1 ✓
                    └─ NO → Model 2 better by >5 ELPD
                            ↓
                          Check posterior c
                            ├─ c near 0 → Use log model (simpler)
                            ├─ c near 1 → Use linear (simpler)
                            └─ c in (0.2, 0.8) → ACCEPT Model 2 ✓
```

## Quick Interpretation Guide

### If Model 1 is chosen:

**Report:**
- "The relationship between Y and x follows a Michaelis-Menten saturation pattern."
- "The estimated asymptote is Y_max = [value] (95% CI: [lower, upper])."
- "Half-saturation occurs at x = K = [value] (95% CI: [lower, upper])."
- "For x > [3×K], Y is predicted to remain near Y_max."

### If Model 2 is chosen:

**Report:**
- "The relationship follows a power-law: Y = a + b·x^c."
- "The exponent c = [value] (95% CI: [lower, upper]) indicates [strong/moderate/weak] diminishing returns."
- "This model does not impose a hard asymptote; Y continues to increase (slowly) as x increases."

### If Model 3 is chosen:

**Report:**
- "The relationship shows exponential approach to asymptote Y_max = [value]."
- "The rate constant r = [value] implies 63% of saturation is complete by x = [1/r]."
- "This pattern is consistent with [equilibration/diffusion/learning] processes."

## Common Pitfalls to Avoid

1. **Overconfident extrapolation:** With only 3 points x>20, predictions for x>30 should have very wide intervals.
2. **Ignoring prior influence:** With N=27, priors matter. Always do sensitivity analysis.
3. **Forcing saturation:** If data doesn't clearly saturate, Model 2 (power-law) may be more honest.
4. **Ignoring computational warnings:** Divergences are not cosmetic - they indicate real problems.
5. **Cherry-picking based on R²:** Use LOO-CV, not in-sample fit. R² rewards complexity.

## Final Checklist Before Reporting

- [ ] Fit at least 2 models (Model 1 + one alternative)
- [ ] All R-hat < 1.01
- [ ] All ESS > 400
- [ ] No divergences (or <1% with high adapt_delta)
- [ ] Posterior predictive checks performed and passed
- [ ] LOO-CV computed for model comparison
- [ ] Prior sensitivity analysis done
- [ ] Parameters scientifically plausible
- [ ] Uncertainty acknowledged (especially x>20)
- [ ] Model limitations discussed

---

*For complete details, see `/workspace/experiments/designer_2/proposed_models.md`*
