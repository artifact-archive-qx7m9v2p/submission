# Prior Revision Guide (If Needed)

**Current Status**: Priors APPROVED - no revision needed

This document provides specific guidance on how to revise priors IF post-fitting diagnostics reveal issues. Use this as a reference if posterior predictive checks fail.

---

## When to Revise Priors

Revise priors ONLY if:

1. **Posterior predictive checks fail** (e.g., still generating many zeros despite no observed zeros)
2. **Sampling diagnostics show problems** (e.g., divergent transitions, poor mixing)
3. **Posterior concentrates in implausible regions** (e.g., negative growth when data shows strong positive growth)

Do NOT revise based solely on prior predictive check "failures" if they reflect genuine uncertainty.

---

## Issue 1: Zero Inflation (Minor Concern)

### Current Situation
- Prior predictive: 4.8 zeros per dataset (12%)
- Observed: 0 zeros
- Status: Acceptable as prior uncertainty

### If This Becomes a Problem After Fitting

**Symptom**: Posterior predictive still generates significant zeros despite observing none.

**Diagnosis**: Model structure may be inappropriate (consider zero-truncated NegBin) OR priors too diffuse.

**Solution Options** (in order of preference):

#### Option A: Tighten β₀ (Intercept) Prior
```
Current: β₀ ~ Normal(4.3, 1.0)
Revised: β₀ ~ Normal(4.3, 0.5)  # Reduces SD from 1.0 to 0.5
```
**Effect**: Reduces probability of very small mean counts that generate zeros.
**Trade-off**: Less uncertainty about baseline count level.

#### Option B: Restrict β₁ to Positive Growth
```
Current: β₁ ~ Normal(0.85, 0.5)
Revised: β₁ ~ Normal(0.85, 0.5) truncated at 0  # Only positive growth
```
**Effect**: Prevents decline scenarios that can lead to low counts and zeros.
**Trade-off**: Assumes growth is certain (may be too strong).

#### Option C: Increase Minimum φ
```
Current: φ ~ Exponential(0.667)  # Can be arbitrarily small
Revised: φ ~ Exponential(0.5) + 0.3  # Shifted to ensure φ ≥ 0.3
```
**Effect**: Reduces extreme overdispersion that can produce zeros.
**Trade-off**: Less flexibility for highly overdispersed data.

**Recommended First Step**: Option A (tighten β₀), as it's least restrictive.

---

## Issue 2: Extreme Value Tail (Currently Acceptable)

### Current Situation
- 2.6% of prior draws exceed 10,000
- Observed maximum: 269
- Status: Acceptable (rare extremes reflect uncertainty)

### If This Becomes a Problem After Fitting

**Symptom**: Posterior predictions regularly exceed 10,000 or show poor calibration in upper tail.

**Diagnosis**: β₁ prior too diffuse, allowing explosive growth scenarios.

**Solution Options**:

#### Option A: Tighten β₁ (Growth Rate) Prior
```
Current: β₁ ~ Normal(0.85, 0.5)
Revised: β₁ ~ Normal(0.85, 0.3)  # Reduces SD from 0.5 to 0.3
```
**Effect**: Constrains growth to [0.25, 1.45] with 95% probability (vs [−0.15, 1.85] currently).
**Trade-off**: Less uncertainty about growth rate.

#### Option B: Use Upper-Bounded Prior on β₁
```
Current: β₁ ~ Normal(0.85, 0.5)
Revised: β₁ ~ Normal(0.85, 0.4) truncated above 2.0
```
**Effect**: Prevents extremely high growth rates (exp(2.0) = 7.4x per year).
**Trade-off**: Assumes explosive growth is implausible.

**Recommended First Step**: Option A (tighten β₁ to 0.3 SD).

---

## Issue 3: Var/Mean Mismatch (Currently Borderline)

### Current Situation
- 45% of prior draws in "plausible range" [20, 200]
- Observed Var/Mean: 68.7
- Status: Borderline but scientifically reasonable

### If This Becomes a Problem After Fitting

**Symptom**: Posterior φ concentrates far from observed Var/Mean pattern, or posterior predictive Var/Mean doesn't match observed.

**Diagnosis**: φ prior may need more informative specification.

**Solution Options**:

#### Option A: Use Gamma Prior on φ (More Concentrated)
```
Current: φ ~ Exponential(0.667)  # Mean = 1.5, SD = 1.5
Revised: φ ~ Gamma(shape=4, rate=2)  # Mean = 2.0, SD = 1.0
```
**Effect**: Concentrates φ near 2.0 with less tail weight.
**Trade-off**: More restrictive about overdispersion level.

#### Option B: Use Half-Normal Prior on φ
```
Current: φ ~ Exponential(0.667)
Revised: φ ~ HalfNormal(scale=2.0)  # Symmetric around 0, peaks at 1.6
```
**Effect**: Different tail behavior, less weight on very large φ.
**Trade-off**: Different distributional family may affect MCMC.

#### Option C: Informative Normal Prior (Truncated)
```
Current: φ ~ Exponential(0.667)
Revised: φ ~ Normal(1.5, 0.7) truncated at 0
```
**Effect**: Strongly concentrates φ near expected value from EDA.
**Trade-off**: Most restrictive option, use only if strong domain knowledge.

**Recommended First Step**: Option A (Gamma prior), as it maintains flexibility while reducing extreme tail.

---

## Combined Revision Strategy

If MULTIPLE issues arise, revise priors systematically:

### Stage 1: Minimal Revision (Address Most Critical Issue)
- Identify which single parameter is causing the most problems
- Make smallest adjustment to that parameter's prior
- Re-run prior predictive check
- Re-fit model

### Stage 2: Moderate Revision (If Stage 1 Insufficient)
- Tighten both β₀ and β₁ moderately:
  ```
  β₀ ~ Normal(4.3, 0.7)  # From 1.0 to 0.7
  β₁ ~ Normal(0.85, 0.4)  # From 0.5 to 0.4
  φ ~ Gamma(4, 2)  # From Exponential(0.667)
  ```

### Stage 3: Strong Revision (Last Resort)
- Use tight, informative priors based on EDA:
  ```
  β₀ ~ Normal(4.3, 0.3)  # Very tight
  β₁ ~ Normal(0.85, 0.2)  # Very tight
  φ ~ Normal(1.5, 0.5) truncated at 0  # Informative
  ```
- **Warning**: Only use if you have strong domain knowledge or extensive data to justify.

---

## Decision Tree for Prior Revision

```
START: Posterior predictive check reveals issues
  │
  ├─ Issue: Posterior generates zeros?
  │   └─ YES → Tighten β₀ to Normal(4.3, 0.5)
  │
  ├─ Issue: Extreme predictions (>10,000)?
  │   └─ YES → Tighten β₁ to Normal(0.85, 0.3)
  │
  ├─ Issue: Var/Mean mismatch?
  │   └─ YES → Change φ to Gamma(4, 2)
  │
  ├─ Issue: Multiple problems?
  │   └─ YES → Follow staged revision strategy
  │
  └─ Issue: Sampling problems (divergences)?
      └─ Consider reparameterization (see below)
```

---

## Alternative Parameterizations

If sampling issues persist despite prior revision, consider model reparameterization:

### Current Parameterization
```
log(μ[i]) = β₀ + β₁ × year[i]
```

### Alternative 1: Centered Parameterization
```
log(μ[i]) = α + β₁ × year[i]
where α = β₀ + β₁ × mean(year)
```
**Benefit**: Reduces correlation between intercept and slope.

### Alternative 2: Non-Centered Parameterization (for φ)
```
φ = exp(log_phi)
log_phi ~ Normal(log(1.5), 0.5)
```
**Benefit**: Can improve sampling for dispersion parameter.

### Alternative 3: Mean-Scaled Parameterization
```
log(μ[i]) = log(μ_ref) + β₁ × (year[i] - year_ref)
μ_ref ~ LogNormal(log(70), 0.3)
```
**Benefit**: Directly specify prior on interpretable scale.

---

## Validation After Revision

After any prior revision, ALWAYS:

1. **Re-run prior predictive check** with new priors
2. **Compare to original**: Document what changed
3. **Re-fit model** with new priors
4. **Posterior predictive check**: Verify issues resolved
5. **Sensitivity analysis**: Ensure posterior not overly sensitive to prior choice

---

## Documentation Requirements

If you revise priors, document in `experiments/experiment_1/metadata.md`:

```markdown
## Prior Revision History

### Version 1 (Original)
- β₀ ~ Normal(4.3, 1.0)
- β₁ ~ Normal(0.85, 0.5)
- φ ~ Exponential(0.667)
- Status: APPROVED but minor zero inflation

### Version 2 (Revised: [Date])
- β₀ ~ Normal(4.3, 0.5)  # CHANGED: Reduced SD to address zero inflation
- β₁ ~ Normal(0.85, 0.5)  # UNCHANGED
- φ ~ Exponential(0.667)  # UNCHANGED
- Reason: Posterior predictive checks showed [specific issue]
- Result: [Outcome of revision]
```

---

## Contact Points for Further Guidance

If prior revision doesn't resolve issues, consider:

1. **Model structure**: Maybe NegBin isn't appropriate (try Zero-Truncated NegBin, Poisson, etc.)
2. **Data quality**: Check for outliers, measurement errors
3. **Covariate specification**: Maybe linear trend insufficient (try quadratic, splines)
4. **Likelihood specification**: Maybe overdispersion structure wrong

**Remember**: Priors are only one component. If multiple rounds of prior revision don't help, the problem likely lies elsewhere in the model specification.

---

## Current Recommendation

**DO NOT REVISE PRIORS YET**

The current priors are well-specified. Proceed to model fitting and assess posterior behavior before making any changes. This guide is for future reference only.
