# Improvement Priorities: Experiment 1
## What to Test Next and Why

**Date**: 2025-10-28
**Model Status**: ACCEPTED as baseline
**Priority**: Compare against alternatives per minimum attempt policy

---

## Overview

The logarithmic model with Normal likelihood passed all validation checks and is scientifically adequate. However, per the minimum attempt policy and pre-registered experiment plan, we must compare against at least 2 alternative models before making final conclusions.

This document prioritizes:
1. **Alternative models to fit** (comparison targets)
2. **Sensitivity analyses to conduct** (robustness checks)
3. **Criteria for changing our decision** (what would make us prefer alternatives)

---

## Section 1: Alternative Models to Compare

### Priority 1: Model 2 - Student-t Likelihood (HIGH)

**Model Structure**:
```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = β₀ + β₁·log(x_i)

Priors:
  β₀ ~ Normal(2.3, 0.3)
  β₁ ~ Normal(0.29, 0.15)
  σ ~ Exponential(10)
  ν ~ Gamma(2, 0.1)  # degrees of freedom, mean=20 (moderately heavy tails)
```

**Rationale**:
1. **Minor Q-Q tail deviation**: Current model shows slight departure from normality in extreme tails
2. **Robustness test**: Student-t likelihood is more robust to outliers
3. **Pre-registered**: Specified in experiment plan as first alternative
4. **Low cost**: Same mean structure, only likelihood changes
5. **x=31.5 observation**: Potentially outlying (Y=2.57 vs expected ~2.68), Student-t may accommodate better

**Expected Outcome**: **ΔLOO < 4** (Normal adequate)
- Q-Q deviation is minor, not severe
- All Pareto k < 0.5 suggests no strong outliers
- Residuals approximately normal (no extreme violations)
- Conservative prediction: Student-t will match Normal, not substantially improve

**Decision Criteria**:
| ΔLOO Range | Interpretation | Action |
|------------|----------------|--------|
| < 2 | Equivalent models | Accept Normal (simpler) |
| 2 to 4 | Marginal preference | Accept Normal (parsimony) OR Student-t (robustness) - context decides |
| > 4 | Strong evidence | Accept Student-t (heavy tails matter) |

**What We'll Learn**:
- Whether tail robustness improves predictive performance
- If x=31.5 is truly influential (though Pareto k=0.32 says no)
- Whether residual non-normality is consequential (probably not)

**Implementation Notes**:
- Use same priors for β₀, β₁, σ as Model 1 (fair comparison)
- Add ν ~ Gamma(2, 0.1) for degrees of freedom (mean ≈ 20, moderately heavy tails)
- Check that ν posterior is away from boundaries (ν→∞ equivalent to Normal, ν<5 very heavy tails)
- If ν posterior is >100: tails not needed, Normal sufficient
- If ν posterior is <10: heavy tails important, Student-t preferred

---

### Priority 2: Model 3 - Piecewise Linear (LOG SPACE) (HIGH)

**Model Structure**:
```
Y_i ~ Normal(μ_i, σ)

μ_i = {
  β₀₁ + β₁·log(x_i)           if x_i < τ (Regime 1)
  β₀₂ + β₂·log(x_i)           if x_i ≥ τ (Regime 2)
}

Constrain: Continuous at breakpoint
  β₀₂ = β₀₁ + (β₁ - β₂)·log(τ)

Priors:
  β₀₁ ~ Normal(2.3, 0.3)
  β₁ ~ Normal(0.29, 0.15)
  β₂ ~ Normal(0.29, 0.15)
  τ ~ Uniform(3, 20)  # breakpoint constrained to middle range
  σ ~ Exponential(10)
```

**Rationale**:
1. **Two-regime hypothesis**: EDA mentioned potential changepoint
2. **Sharp vs smooth saturation**: Tests whether transition is abrupt or gradual
3. **Biological plausibility**: Many processes have threshold effects
4. **Pre-registered**: Specified in experiment plan as key alternative
5. **Scientific insight**: If breakpoint exists, could be mechanistically meaningful

**Expected Outcome**: **ΔLOO < 0** (Logarithmic model preferred)
- Residuals show no clustering or regime structure
- No visible breakpoint in fitted curve
- Piecewise adds 2 parameters (β₂, τ) without clear justification
- Likely to overfit (p_loo > 5 expected)

**Decision Criteria**:
| Scenario | ΔLOO | Breakpoint | Action |
|----------|------|------------|--------|
| No improvement | < 0 | N/A | Accept Log (parsimony) |
| Marginal improvement | 0 to 4 | Arbitrary | Accept Log (simpler) |
| Strong improvement, meaningful τ | > 4 | Scientifically interpretable | Accept Piecewise (regime change real) |
| Strong improvement, arbitrary τ | > 4 | Not interpretable | Accept Log (overfitting suspected) |

**What We'll Learn**:
- Whether sharp changepoint exists or saturation is smooth
- If there's a scientifically meaningful threshold (e.g., τ ≈ 10 where pattern changes)
- Whether residuals cluster by regime (evidence of model misspecification)
- If added complexity is justified by predictive improvement

**Implementation Notes**:
- Constrain τ to middle range [3, 20] to avoid boundary issues
- Enforce continuity at τ (no jumps in mean function)
- Check τ posterior is away from boundaries (if τ→3 or τ→20, breakpoint not well-defined)
- Compare slopes β₁ vs β₂: if similar, no regime change detected
- If β₁ ≈ β₂ AND ΔLOO < 0: strong evidence for single regime (logarithmic model correct)

---

### Priority 3: Model 4 - Gaussian Process (MEDIUM, OPTIONAL)

**Model Structure**:
```
Y ~ Normal(f(x), σ)
f(x) ~ GP(m(x), k(x, x'))

Mean function: m(x) = β₀ + β₁·log(x)
Kernel: k(x, x') = α²·exp(-ρ·(x - x')²)  # Squared exponential

Priors:
  β₀, β₁ ~ as before (mean function priors)
  α ~ Exponential(1)  # signal variance
  ρ ~ InvGamma(5, 5)  # length scale
  σ ~ Exponential(10)  # noise variance
```

**Rationale**:
1. **Maximum flexibility**: No parametric assumptions beyond log mean function
2. **Detect subtle patterns**: Can capture wiggles missed by log model
3. **Nonparametric robustness**: Tests if parametric log form is sufficient
4. **Pre-registered**: Specified as flexible alternative
5. **Benchmark**: Provides upper bound on achievable predictive performance

**Expected Outcome**: **ΔLOO < 0** (Overfitting likely)
- Current model has clean residuals (no systematic patterns to capture)
- n=27 is small for flexible GP
- Likely to memorize noise rather than learn signal
- May show high p_loo (>6) indicating overfitting
- Predictive uncertainty may be inflated

**Decision Criteria**:
| Scenario | ΔLOO | Pattern | Action |
|----------|------|---------|--------|
| No improvement | < 0 | N/A | Accept Log (parsimony) |
| Marginal improvement | 0 to 4 | Wiggly/uninterpretable | Accept Log (overfitting) |
| Strong improvement | > 4 | Smooth, scientifically meaningful | Consider GP (unexpected!) |
| Strong improvement | > 4 | Wiggly, complex | Accept Log (noise-fitting) |

**What We'll Learn**:
- Whether logarithmic function is flexible enough
- If there are subtle nonlinear patterns we missed
- Upper bound on predictive performance (GP should not do worse)
- Whether simplicity/interpretability tradeoff favors parametric model

**Implementation Notes**:
- Use log(x) as input to GP (not raw x) to leverage prior knowledge
- Monitor p_loo: if >6, overfitting suspected
- Check posterior predictive: if uncertainty bands much wider than Model 1, overfitting confirmed
- Examine f(x) draws: if very wiggly between observations, overfitting
- If GP matches log model (ΔLOO ≈ 0): strong evidence log form is correct

**Priority Justification**: MEDIUM because:
- Models 2-3 more directly address concerns (tails, regimes)
- GP is computationally expensive
- n=27 may be too small for reliable GP fitting
- Can skip if Models 2-3 confirm Model 1

---

## Section 2: Sensitivity Analyses to Conduct

### Sensitivity 1: Prior Robustness (HIGH PRIORITY)

**Test**: Refit Model 1 with less informative priors

**Alternative Prior Specification**:
```
β₀ ~ Normal(2.3, 0.6)      # 2× wider (SD 0.6 vs 0.3)
β₁ ~ Normal(0.29, 0.30)    # 2× wider (SD 0.3 vs 0.15)
σ ~ Exponential(5)         # 2× wider (mean 0.2 vs 0.1)
```

**Rationale**:
- Current priors are informed by EDA (weakly informative)
- Should verify conclusions robust to reasonable prior variation
- Standard practice in Bayesian analysis
- Reviewers will ask about sensitivity

**Expected Outcome**: **Posteriors change < 10%**
- Data strongly informed (precision increased 7-8×)
- Wider priors should yield similar posteriors
- May see slightly wider HDIs (5-10% increase) but similar means

**Decision Criteria**:
| Change in Posterior | Interpretation | Action |
|---------------------|----------------|--------|
| Means shift < 10%, HDIs widen < 20% | Robust | Report original results, note robustness |
| Means shift 10-20%, HDIs widen 20-50% | Moderate sensitivity | Report both, discuss sensitivity |
| Means shift > 20%, HDIs widen > 50% | High sensitivity | Use less informative priors, report sensitivity |

**What We'll Learn**:
- Whether conclusions depend on specific prior choice
- How much prior vs data drives inference
- If informative priors are justified by robustness

**Implementation**:
- Fit with wide priors
- Compare posteriors using plots (overlay distributions)
- Compute relative change: |mean_new - mean_old| / SD_old
- If change < 0.5 SD: robust (good)
- If change > 1 SD: sensitive (concerning)

---

### Sensitivity 2: LOO Stability Check (ALREADY COMPLETE)

**Test**: Check Pareto k diagnostics for LOO reliability

**Result**: ✓ PASSED
- All Pareto k < 0.5 (max = 0.32)
- LOO estimates are fully reliable
- No need for K-fold CV or other alternatives

**Conclusion**: No further action needed. LOO is trustworthy.

---

### Sensitivity 3: Influential Observations (ALREADY COMPLETE)

**Test**: Identify high-leverage or influential observations

**Result**: ✓ NO ISSUES
- Pareto k diagnostics show no influential points
- x=31.5 (Y=2.57) has largest residual but k=0.32 (acceptable)
- No standardized residuals > 2.5

**Conclusion**: No need for outlier removal or robust regression beyond testing Student-t.

---

### Sensitivity 4: Functional Form Exploration (IN PROGRESS)

**Test**: Compare against alternative functional forms (piecewise, GP, etc.)

**Status**: This IS the model comparison (Experiments 2-4)

**Conclusion**: Being addressed by planned experiments.

---

## Section 3: What Would Change Our Decision?

### Scenario 1: Student-t Substantially Improves (ΔLOO > 4)

**Implication**: Heavy tails matter, Normal likelihood inadequate

**New Decision**: REVISE to use Student-t likelihood
- Keep logarithmic mean function (μ = β₀ + β₁·log(x))
- Switch likelihood from Normal to Student-t
- Report ν posterior (effective degrees of freedom)
- Interpret: "Data show heavier tails than Normal distribution"

**Action Items**:
1. Adopt Student-t model as primary
2. Re-run all diagnostics (convergence, PPC, LOO)
3. Report: "Student-t likelihood preferred (ΔLOO = X.X ± SE)"
4. Discuss: Implications of heavy tails for prediction intervals

**What This Tells Us**:
- Outlier-robustness matters in this domain
- Extreme observations more common than Normal predicts
- Prediction intervals should be wider (conservative)

---

### Scenario 2: Piecewise Improves with Interpretable Breakpoint (ΔLOO > 4)

**Implication**: Sharp regime change exists, smooth log curve oversimplifies

**New Decision**: REVISE to use piecewise model
- Report breakpoint location τ and 95% HDI
- Interpret slopes β₁ (Regime 1) and β₂ (Regime 2)
- Check if breakpoint scientifically meaningful (e.g., dose threshold)

**Action Items**:
1. Adopt piecewise model as primary
2. Investigate mechanism: Why does relationship change at τ?
3. Report: "Two-regime structure detected at x ≈ τ (95% HDI: [τ_low, τ_high])"
4. Discuss: Scientific implications of regime change

**What This Tells Us**:
- System has threshold or critical point
- Different processes operate below/above threshold
- Extrapolation requires regime-specific predictions

**Caveat**: If τ posterior is vague (wide HDI) or uninterpretable (no mechanistic reason for that value), prefer logarithmic model even if ΔLOO > 4 (simpler, more interpretable).

---

### Scenario 3: Multiple Models Substantially Outperform (ΔLOO > 6)

**Implication**: Logarithmic form fundamentally wrong

**New Decision**: REJECT parametric log model, adopt flexible alternative (likely GP)

**Action Items**:
1. Investigate why log model fails (systematic residual patterns?)
2. Fit additional exploratory models (power law, exponential, splines)
3. Consider non-saturation mechanisms
4. Report: "Parametric log model inadequate, nonparametric approach required"

**What This Tells Us**:
- Relationship is more complex than logarithmic saturation
- Mechanistic theory needs revision
- May need more data to characterize complex relationship

**Likelihood**: VERY LOW (<5%). Current model passes all checks convincingly.

---

### Scenario 4: Prior Sensitivity Detected (>20% change in means)

**Implication**: Inferences depend on specific prior choice

**New Decision**: REPORT SENSITIVITY prominently

**Action Items**:
1. Refit with multiple reasonable priors (sensitivity grid)
2. Report range of conclusions across priors
3. Consider using least informative prior that converges
4. Discuss: "Results somewhat sensitive to prior choice; range of plausible values is [X, Y]"

**What This Tells Us**:
- Data alone insufficient to pin down parameters precisely
- Prior knowledge plays meaningful role
- Should be transparent about epistemic uncertainty

**Likelihood**: LOW (~10%). Data strongly informed posteriors (7-8× precision gain).

---

## Section 4: What Will NOT Change Decision

### Small Improvements (ΔLOO < 2)

**Why**: Within noise. Model choice arbitrary at this level.

**Action**: Accept logarithmic model (parsimony). Report alternatives as "equivalent performance" in supplement.

---

### Marginal PPC Improvements

**Why**: Current model already passes all 10 test statistics. Tiny improvements (p-values moving from 0.4 to 0.5) are not meaningful.

**Action**: Accept logarithmic model. Note that alternatives also pass PPC, but don't add value.

---

### Slightly Better R² (e.g., 0.89 → 0.91)

**Why**: In-sample fit can improve through overfitting. LOO is better metric.

**Action**: Ignore R² differences < 0.05. Focus on ΔLOO.

---

### Preference for Flexibility/Complexity

**Why**: Simpler models preferred unless complexity clearly justified (parsimony principle).

**Action**: Accept logarithmic model unless ΔLOO > 4 AND scientific insight gained.

---

### Aesthetic Preferences

**Why**: Model selection is evidence-based, not preference-based.

**Action**: Follow decision criteria (ΔLOO, scientific interpretability), not subjective appeal.

---

## Section 5: Decision Framework Summary

### Accept Logarithmic Model (Model 1) if:

1. **All alternatives show ΔLOO < 4** (no substantial improvement)
   - Even if some improve, improvement is within noise or not meaningful
   - Parsimony strongly favors simpler model

2. **Alternatives improve but lose interpretability**
   - E.g., GP improves ΔLOO = 5 but shows wiggly uninterpretable patterns
   - Logarithmic form provides clear scientific interpretation

3. **Alternatives show signs of overfitting**
   - High p_loo (>6)
   - Very wide predictive intervals
   - Unstable posterior (high R-hat, low ESS)

4. **Prior sensitivity is low**
   - Posteriors stable across reasonable prior variations
   - Conclusions robust

### Switch to Alternative Model if:

1. **ΔLOO > 4** (substantial evidence)
2. **AND** alternative passes validation checks (convergence, PPC)
3. **AND** provides additional scientific insight (e.g., interpretable breakpoint)
4. **AND** interpretability not sacrificed (unless ΔLOO >> 6)

### Report Multiple Models if:

1. **Close comparison** (ΔLOO ∈ [2, 4])
2. **Different scientific interpretations** (e.g., smooth vs sharp transition)
3. **High prior sensitivity** (conclusions depend on prior choice)

---

## Section 6: Expected Timeline

### Week 1: Alternative Model Fitting
- **Day 1-2**: Fit Model 2 (Student-t)
  - Run MCMC (expect ~30min)
  - Compute diagnostics, LOO
  - Compare to Model 1

- **Day 3-4**: Fit Model 3 (Piecewise)
  - Run MCMC (expect ~1hr, more complex)
  - Compute diagnostics, LOO
  - Interpret breakpoint if present

- **Day 5**: Sensitivity Analysis
  - Refit Model 1 with wide priors
  - Compare posteriors
  - Document robustness

### Week 2: Model Comparison and Decision
- **Day 1**: Compute ΔLOO for all models
  - Create comparison table
  - Check ΔLOO > 2×SE threshold

- **Day 2**: Final Model Selection
  - Apply decision criteria
  - Select best model or report multiple

- **Day 3**: (Optional) Fit Model 4 (GP) if needed
  - Only if Models 2-3 show unexpected patterns
  - Or if stakeholders require maximum flexibility

### Week 3: Documentation
- **Final Report**: Synthesis of all models
- **Supplement**: Full details on alternatives
- **Presentation**: Key findings and recommendations

---

## Section 7: Success Criteria

### We Will Be Satisfied If:

1. **Minimum 2 alternatives tested** (Models 2-3, per policy)
2. **Clear decision reached** (ACCEPT Model 1 OR switch to alternative)
3. **Decision criteria followed** (ΔLOO threshold, scientific interpretability)
4. **Sensitivity assessed** (priors, functional form)
5. **Conclusions robust** (don't depend on arbitrary choices)

### Red Flags That Would Concern Us:

1. **All models fail validation** (convergence issues, PPC failures)
   - Suggests data quality problems or fundamental misspecification

2. **ΔLOO >> 10 favoring complex model** (e.g., GP destroys log model)
   - Suggests we missed major feature of data
   - Would require mechanistic re-thinking

3. **High prior sensitivity across all models** (>50% changes)
   - Suggests n=27 insufficient for precise inference
   - Would need more data

4. **Contradictory conclusions** (Model 2 says X, Model 3 says opposite)
   - Suggests model uncertainty is high
   - Would need to report range of possibilities

**Likelihood of Red Flags**: LOW. Model 1 passed all checks convincingly.

---

## Section 8: Stakeholder Communication

### If Logarithmic Model Remains Preferred (Expected):

**Message**: "We tested multiple alternatives (robust likelihood, breakpoint model) and confirmed the logarithmic model is adequate. It provides the best balance of fit quality, interpretability, and parsimony. Conclusions are robust to reasonable variations in assumptions."

### If Alternative Model Wins:

**Message**: "Initial logarithmic model was adequate but [Student-t/Piecewise] improved predictive performance meaningfully (ΔLOO = X ± SE). This indicates [heavy tails matter / regime change exists]. We recommend using the [alternative] model for final inference."

### If Multiple Models Equivalent:

**Message**: "Multiple models provide similar predictive performance (ΔLOO < 4). We report results from the logarithmic model (simplest, most interpretable) with sensitivity analysis showing conclusions robust to alternative specifications."

---

## Conclusion

**Primary Goal**: Compare Models 1-2-3, apply decision criteria, select best model.

**Expected Outcome**: Model 1 (logarithmic with Normal likelihood) will remain preferred.
- ΔLOO < 4 for all alternatives
- Minor issues (Q-Q tails) will not translate to improved prediction
- Parsimony strongly favors simpler model

**Contingency Plans**: If alternatives improve:
- Student-t (ΔLOO > 4): Adopt robust likelihood
- Piecewise (ΔLOO > 4 + interpretable τ): Adopt regime-change model
- GP (ΔLOO > 4 + smooth pattern): Unlikely, but would adopt if clear

**Bottom Line**: We're confident in Model 1, but science requires testing alternatives. We expect to confirm Model 1 as final choice, but we're open to evidence favoring alternatives.

---

**Status**: Ready to begin model comparison
**Next Action**: Fit Experiment 2 (Student-t likelihood)
**Timeline**: Complete comparison within 1-2 weeks
**Decision**: Apply criteria objectively, follow evidence

**Prepared by**: Model Criticism Specialist
**Date**: 2025-10-28
