# Flexible & Nonparametric Bayesian Models: Designer 2

**Focus Area**: Splines, Gaussian Processes, and Smoothing Methods
**Designer**: Flexible/Nonparametric Modeling Specialist
**Date**: 2025-10-28
**Status**: Design Complete, Ready for Implementation

---

## Overview

This directory contains a complete Bayesian modeling strategy for the Y~x relationship using **flexible, nonparametric approaches**. The design emphasizes:

1. **Falsification over confirmation** - Each model has explicit abandonment criteria
2. **Computational honesty** - Diagnostics are treated as warnings, not nuisances
3. **Scientific humility** - Simple models may be right; complexity must prove itself
4. **Rapid pivoting** - Clear decision points to avoid wasted effort

---

## Contents

### 1. Main Proposal: `proposed_models.md` (31 KB, 870 lines)

**THE CORE DOCUMENT** - Read this first for complete model specifications.

Includes:
- Three competing model classes (GP, P-splines, Adaptive GP)
- Full mathematical specifications with priors
- Falsification criteria for each model
- Expected strengths and weaknesses
- Decision points and pivot strategies
- Escape routes if models fail

**Key sections**:
- Lines 1-100: Executive summary and competing hypotheses
- Lines 100-300: Model 1 (Gaussian Process)
- Lines 300-500: Model 2 (Penalized B-Splines)
- Lines 500-700: Model 3 (Adaptive GP with changepoint)
- Lines 700-870: Comparison strategy and outcomes

---

### 2. Quick Reference: `model_summary.md` (6 KB, 173 lines)

**START HERE** if you need a quick overview before diving into details.

Contents:
- One-paragraph summary of each model
- Critical decision points (4 checkpoints)
- Falsification mindset overview
- Success metrics
- Timeline and workflow

**Use this for**:
- Quick refresher during implementation
- Explaining models to collaborators
- Decision-making at checkpoints

---

### 3. Decision Tree: `decision_tree.md` (13 KB, 280 lines)

**FLOWCHART** for navigating the modeling process.

Structure:
```
START
├─ Fit Model 1 (GP)
│  ├─ Converges? → LOO? → PPC? → Success!
│  └─ Fails? → Try Model 2 or pivot
├─ Checkpoints (4 decision points)
└─ Final deliverable
```

**Use this when**:
- Making real-time decisions during fitting
- Stuck and unsure what to do next
- Need to decide whether to abandon an approach
- Communicating workflow to team

---

### 4. Implementation Guide: `implementation_guide.md` (23 KB, 682 lines)

**COMPLETE RUNNABLE CODE** for all three models in PyMC.

Contents:
- Data preparation and setup
- Full PyMC implementations (copy-paste ready)
- Diagnostic checks with thresholds
- Falsification checks (automated)
- Model comparison code
- Outlier sensitivity analysis
- Troubleshooting guide

**Code blocks**:
- Lines 10-60: Setup and data prep
- Lines 60-180: Model 1 (GP) - full implementation
- Lines 180-280: Model 2 (P-splines) - full implementation
- Lines 280-400: Model 3 (Adaptive GP) - full implementation
- Lines 400-500: Model comparison and PPC
- Lines 500-600: Sensitivity analysis
- Lines 600-682: Troubleshooting

**All code is tested syntax** (not pseudocode) and ready to run.

---

### 5. This File: `README.md`

Navigation and overview of the entire design.

---

## Three Proposed Models

### Model 1: Gaussian Process with Matérn 3/2 Kernel ⭐ RECOMMENDED START

**Philosophy**: Smooth saturation, no true regime change

**Mathematical Form**:
```
Y ~ Normal(f(x), σ)
f(x) ~ GP(β₀ + β₁*log(x), Matérn₃/₂(ℓ, α))
```

**Why this model**:
- Most principled (minimal ad hoc choices)
- Once-differentiable (realistic for real processes)
- Can discover structure EDA missed
- Provides uncertainty quantification naturally

**I will abandon if**:
- Divergences > 5%
- LOO worse than logarithmic baseline
- Length scale ℓ at extreme values (< 0.5 or > 20)
- Wild oscillations in posterior mean

**Expected outcome**: Beats baseline by ΔLOO ≈ 2-3, validates EDA's smooth saturation story

---

### Model 2: Penalized B-Splines with 6 Interior Knots

**Philosophy**: Locally complex, can handle sharp transitions

**Mathematical Form**:
```
Y ~ Normal(μ, σ)
μ = Σⱼ βⱼ * Bⱼ(x)
Δβⱼ ~ Normal(0, τ)  (random walk penalty)
```

**Why this model**:
- Fast to fit (2-3 minutes vs 5-10 for GP)
- Interpretable basis functions
- Less parametric than GP kernel choice
- Can capture sharp regime changes

**I will abandon if**:
- Coefficients oscillate wildly (>50% sign changes)
- Smoothness parameter τ → 0 (infinite penalty)
- LOO worse than Model 1
- Poor extrapolation behavior

**Expected outcome**: Comparable to GP, possibly better if regime change is sharp

---

### Model 3: Adaptive GP with Regime-Specific Smoothness ⚠️ ADVANCED

**Philosophy**: True regime change at unknown τ with different dynamics

**Mathematical Form**:
```
For x ≤ τ:  f(x) ~ GP(m₁(x), k₁(x,x'))
For x > τ:  f(x) ~ GP(m₂(x), k₂(x,x'))
τ ~ Uniform(4, 12)
ℓ₁ ≠ ℓ₂  (regime-specific length scales)
```

**Why this model**:
- Directly tests EDA's two-regime hypothesis
- Estimates changepoint location with uncertainty
- Allows different smoothness per regime

**I will abandon if**:
- Divergences > 10% (computational failure)
- τ posterior is uniform (no changepoint detected)
- ℓ₁ ≈ ℓ₂ (no regime-specific smoothness)
- LOO worse than Models 1-2 (complexity unjustified)

**Expected outcome**: Only fit if Models 1-2 suggest regime change; may confirm/reject at τ ≈ 7

---

## Critical Success Criteria

### Computational (Must Pass)
- ✓ R-hat < 1.01 for all parameters
- ✓ ESS_bulk > 400, ESS_tail > 400
- ✓ Divergences < 1% (ideally 0%)
- ✓ E-BFMI > 0.3

### Predictive (Primary Metric)
- ✓ LOO-CV lower than baseline (logarithmic model)
- ✓ ΔLOO > 2 considered meaningful difference
- ✓ k-hat < 0.7 for all observations (no unstable points)

### Scientific (Validation)
- ✓ Posterior predictions are monotonic increasing
- ✓ Predictions asymptote at high x
- ✓ Variance matches observed data
- ✓ Robust to x=31.5 outlier removal

---

## Four Critical Checkpoints

### Checkpoint 1: Convergence (Day 1 Morning)
**Question**: Do models converge?
- Fit Models 1 & 2 in parallel
- Check R-hat, ESS, divergences
- **Go/No-Go**: If both fail, abandon flexible approaches

### Checkpoint 2: LOO Comparison (Day 1 Afternoon)
**Question**: Do flexible models beat parametric baseline?
- Compare with logarithmic model (Designer 1)
- **Go/No-Go**: If ΔLOO < -2, accept simple model

### Checkpoint 3: Posterior Predictive Checks (Day 2 Morning)
**Question**: Do models capture data features?
- Monotonicity, asymptotic behavior, variance
- **Go/No-Go**: If systematic failures, reconsider likelihood

### Checkpoint 4: Scientific Story (Day 2 Afternoon)
**Question**: Can we interpret results?
- Clear winner or model averaging
- Outlier sensitivity check
- **Deliverable**: Final recommendation with evidence

---

## Implementation Workflow

### Phase 1: Model 1 (GP) [Priority 1]
1. Fit GP with Matérn 3/2 kernel
2. Check convergence diagnostics
3. Compute LOO-CV
4. Compare to baseline
5. **Decision**: If successful, proceed. If fails, try Model 2

**Time**: 2-3 hours (including diagnostics)

### Phase 2: Model 2 (P-splines) [Priority 2]
1. Fit penalized B-splines
2. Check for coefficient oscillations
3. Compute LOO-CV
4. Compare to Model 1
5. **Decision**: Choose better model or proceed with both

**Time**: 1-2 hours

### Phase 3: Model 3 (Adaptive GP) [Priority 3, OPTIONAL]
1. Only fit if Models 1-2 suggest regime change
2. Expect computational challenges
3. Check changepoint posterior concentration
4. Compare length scales ℓ₁ vs ℓ₂
5. **Decision**: Keep if clearly superior (ΔLOO > 4)

**Time**: 3-4 hours (challenging model)

### Phase 4: Validation [Required]
1. Posterior predictive checks for winner(s)
2. Outlier sensitivity (refit without x=31.5)
3. Final comparison table
4. Write report with honest assessment

**Time**: 2-3 hours

**Total Estimated Time**: 2-3 days (analyst time), 2-3 hours (compute time)

---

## Key Predictions (Falsifiable!)

I predict:
1. **Model 1 (GP) will beat logarithmic baseline by ΔLOO ≈ 2-3**
   - If wrong: Logarithmic is truly optimal, flexibility overfits

2. **Length scale ℓ ≈ 5-10 (on original scale)**
   - If ℓ << 5: Data have sharp local features
   - If ℓ >> 10: Data are nearly linear, GP unnecessary

3. **Model 2 will activate ≥8 of 10 basis functions**
   - If wrong: Fewer active functions → simpler model sufficient

4. **Model 3 (if fit) will identify τ ∈ [6, 8]**
   - If wrong: Either no changepoint exists or it's elsewhere

5. **All models will struggle with x=31.5 outlier**
   - If wrong: Not actually an outlier, or models too flexible

---

## Escape Routes (If Things Go Wrong)

### If all flexible models overfit:
→ Accept parametric approach (Designer 1)
→ Write "Why Flexibility Failed with n=27"
→ Recommend minimum sample size for GP/splines

### If computational issues persist:
→ Try variational inference (faster, approximate)
→ Simplify models (fewer knots, fixed changepoint)
→ Use Designer 1's parametric models

### If outlier dominates:
→ Switch to Student-t likelihood
→ Consider robust regression
→ Flag in report as data quality issue

### If models disagree fundamentally:
→ Use Bayesian model averaging
→ Report uncertainty honestly
→ Recommend collecting more data

### If sample size is truly limiting:
→ Document that n=27 is insufficient
→ Use strongly informative priors
→ Accept wider uncertainty intervals

---

## What Success Looks Like

### Minimal Success ✓
- At least one model converges cleanly
- LOO-CV computed successfully
- Clear recommendation (even if "use simple model")
- Honest documentation of what worked/failed

### Target Success ✓✓
- Multiple models converge
- One clear winner (ΔLOO > 4)
- Posterior predictive checks pass
- Robust to outlier
- Interpretable scientific story

### Exceptional Success ✓✓✓
- Discover structure EDA missed (validated by data)
- Definitively confirm/reject regime change hypothesis
- Models agree on fundamentals despite different approaches
- Actionable predictions with well-calibrated uncertainty
- Honest account of what I got wrong

---

## What Failure Looks Like (And That's OK!)

### "Good Failure" 😊
- Flexible models don't improve on simple logarithmic
- Document why complexity doesn't help
- Recommend parametric approach
- Learn that n=27 is too small for GP/splines

**This is success in disguise** - we learned the truth!

### "Bad Failure" 😞
- All models fail convergence despite tuning
- Cannot compute LOO-CV (unstable)
- Models produce implausible predictions
- No clear recommendation possible

**Even this is informative** - tells us about data limitations

---

## Philosophy: Embracing Uncertainty

This design embodies:

**Bayesian Epistemology**
- Models are tentative hypotheses, not truth
- Posterior uncertainty is information, not failure
- Model comparison reveals what data can/cannot tell us

**Falsificationism (Popper)**
- Focus on how models can fail, not how they succeed
- Computational diagnostics are tests, not nuisances
- Divergences mean "this model may be wrong," not "run longer"

**Scientific Humility**
- Simple models are often right (Occam's Razor)
- Complex models must earn their keep (ΔLOO > 2)
- n=27 limits what we can learn (acknowledge it!)

**Pragmatism**
- Perfect is the enemy of good
- Document what worked and what didn't
- Honest negative results are valuable

---

## Getting Started

### If you're implementing:
1. **Read**: `model_summary.md` (5 min overview)
2. **Implement**: `implementation_guide.md` (copy-paste code)
3. **Navigate**: `decision_tree.md` (when stuck)
4. **Reference**: `proposed_models.md` (deep details)

### If you're reviewing:
1. **Read**: This README + `model_summary.md`
2. **Check**: Mathematical specifications in `proposed_models.md`
3. **Evaluate**: Falsification criteria and decision points

### If you're deciding what to do:
1. **Start**: `decision_tree.md` at the top
2. **Follow**: Branches based on what happens
3. **Pivot**: When checkpoints say to pivot
4. **Report**: Honestly, whatever the outcome

---

## Dependencies

### Required Software
- Python 3.9+
- PyMC 5.x (Bayesian PPL)
- ArviZ (diagnostics and visualization)
- NumPy, Pandas (data manipulation)
- Matplotlib, Seaborn (plotting)

### Optional
- Patsy (B-spline basis generation)
- SciPy (optimization, special functions)

### Installation
```bash
pip install pymc arviz numpy pandas matplotlib seaborn patsy scipy
```

---

## File Locations

- **Main proposal**: `/workspace/experiments/designer_2/proposed_models.md`
- **Summary**: `/workspace/experiments/designer_2/model_summary.md`
- **Decision tree**: `/workspace/experiments/designer_2/decision_tree.md`
- **Implementation**: `/workspace/experiments/designer_2/implementation_guide.md`
- **This README**: `/workspace/experiments/designer_2/README.md`

**Data**: `/workspace/data/data.csv` (n=27)
**EDA Report**: `/workspace/eda/eda_report.md`

---

## Questions?

**Q: Which model should I start with?**
A: Model 1 (GP with Matérn 3/2). It's most principled and likely to succeed.

**Q: What if Model 1 fails?**
A: Follow the decision tree. Usually: try Model 2, then consider parametric.

**Q: Should I fit all three models?**
A: No! Only fit Model 3 if Models 1-2 suggest regime change is real.

**Q: What if LOO says all models are similar?**
A: Choose most interpretable (likely Model 2 or parametric). Or use model averaging.

**Q: What defines success?**
A: A clear, honest recommendation backed by evidence. Even "use simple model" is success!

**Q: How much should I tune if divergences occur?**
A: Try target_accept up to 0.99. If still >5% divergences, abandon the model.

**Q: What if I'm stuck?**
A: Consult the decision tree. If still stuck, document where and why, then pivot.

---

## Contact

**Designer**: Flexible/Nonparametric Modeling Specialist
**Design Date**: 2025-10-28
**Design Status**: Complete and ready for implementation

---

**Bottom Line**: These models are designed to fail informatively. Success means learning the truth, whether that's "flexibility helps" or "simple models win." Let the data decide!
