# Designer 3: Adversarial Bayesian Model Design

**Role**: Critical edge-case specialist - challenge the EDA consensus
**Philosophy**: Assume the EDA is wrong, design models to break the "complete pooling" recommendation

---

## Quick Start

**Main deliverable**: `/workspace/experiments/designer_3/proposed_models.md`

This contains three adversarial model designs that challenge the EDA's strong conclusion that complete pooling is appropriate.

---

## File Guide

### Core Documents

1. **`proposed_models.md`** - MAIN DOCUMENT
   - Three adversarial model classes with full specifications
   - Falsification criteria for each model
   - Why each model might succeed where EDA failed
   - Scientific justification and computational considerations

2. **`experiment_plan.md`** - IMPLEMENTATION ROADMAP
   - Detailed experimental design
   - Decision tree for model selection
   - Stress tests and validation procedures
   - Timeline and deliverables

3. **`stan_model_sketches.md`** - CODE TEMPLATES
   - Ready-to-use Stan code for all models
   - Python code for model fitting and comparison
   - Posterior predictive check implementations
   - Stress test validation code

4. **`README.md`** - THIS FILE
   - Overview and navigation guide

---

## Three Adversarial Models

### Model 1: Measurement Error Inflation
**Hypothesis**: Reported sigma values are systematically wrong

**Key parameter**: `lambda` (inflation factor)
- lambda ≈ 1.0 → errors are accurate (EDA is right)
- lambda > 1.5 → errors are underestimated
- lambda < 0.8 → errors are overestimated

**Why it matters**: If measurement errors are wrong, between-group variance might exist but be masked.

**Falsification**: Abandon if 95% CrI for lambda ⊆ [0.9, 1.1]

---

### Model 2: Latent Mixture (K=2 clusters)
**Hypothesis**: Hidden subgroup structure exists

**Key parameters**: `mu[1]`, `mu[2]` (cluster means), `pi` (mixing)

**Why it matters**:
- EDA noted SNR divide (Groups 0-3 high, Groups 4-7 low)
- Possible bimodal distribution
- Small sample (n=8) makes clustering hard to detect

**Falsification**: Abandon if clusters collapse (mu[1] ≈ mu[2]) OR WAIC penalty >4

---

### Model 3: Functional Heteroscedasticity
**Hypothesis**: Measurement error scales with true value

**Key parameter**: `alpha` (scaling exponent)
- alpha ≈ 0 → error independent of value (EDA is right)
- alpha > 0 → error increases with true value
- alpha < 0 → error decreases with true value

**Why it matters**: EDA tested correlation with OBSERVED y, not TRUE theta. Important distinction.

**Falsification**: Abandon if 95% CrI for alpha ⊆ [-0.05, 0.05]

---

## Critical Design Principles

### 1. Falsification Mindset
Every model has clear criteria for when to abandon it. Success is learning which models DON'T work.

### 2. Stress Testing
All models will be validated on synthetic data where ground truth is known.

### 3. Computational Realism
Models must converge (R-hat < 1.01, ESS > 400) or be abandoned.

### 4. Scientific Plausibility
Models test real scientific questions, not just mathematical flexibility.

---

## Expected Outcomes

### Most Likely: EDA is Correct
- All adversarial models falsified
- lambda ≈ 1.0, alpha ≈ 0, no clusters
- Complete pooling confirmed (even stronger evidence)

### Alternative: EDA Missed Something
- At least one adversarial model succeeds
- Evidence for:
  - Wrong measurement errors (lambda ≠ 1)
  - Hidden clusters (mixture preferred)
  - Functional error (alpha ≠ 0)

---

## How This Differs from Other Designers

**Designer 1 & 2**: Likely propose standard hierarchical models, maybe with different priors or parameterizations

**Designer 3 (this)**:
- Explicitly challenges the EDA conclusion
- Tests ASSUMPTIONS, not just model variants
- Designed to find failures in the consensus
- If adversarial models fail, consensus is much stronger

---

## Implementation Priority

**Run in this order:**

1. **Baseline** (complete pooling) - EDA recommendation
2. **Model 1** (Inflation) - Test measurement error validity
3. **Model 3** (Functional) - Test error independence
4. **Model 2** (Mixture) - Test homogeneity (only if Models 1 & 3 pass)

**Rationale**: If measurement model is wrong (Models 1 or 3), cannot trust any hierarchical inference (Model 2).

---

## Key Questions Answered

### Q: What if all models confirm complete pooling?
**A**: This is SUCCESS! We tried hard to break it and failed. Much stronger evidence than EDA alone.

### Q: What if measurement errors are wrong?
**A**: STOP everything. Investigate measurement process. Cannot trust current analysis.

### Q: What if we find clusters?
**A**: EDA was wrong. Investigate WHY clusters exist. Revise scientific understanding.

### Q: What if models don't converge?
**A**: Try non-centered parameterization, increase adapt_delta. If still fails, model is too complex for n=8.

---

## Computational Requirements

**Time**: ~10 minutes total for all models (on modern CPU)

**Software**:
- Stan (CmdStanPy or PyStan)
- Python 3.8+ with ArviZ, NumPy, Pandas

**Disk**: ~50 MB for all outputs

---

## Red Flags to Watch

### Model 1 (Inflation)
- Correlation(lambda, tau) > 0.9 → non-identifiable
- Divergent transitions >2%
- Lambda posterior = prior

### Model 2 (Mixture)
- R-hat > 1.05 → label switching
- Unstable cluster assignments
- Pareto k > 0.7 → overfitting

### Model 3 (Functional)
- Numerical instability (exp blows up)
- Alpha posterior = prior
- Extreme sigma_effective values

---

## Success Criteria

**Scientific**:
- Clear conclusion about which models are supported
- Understanding of why models succeed/fail

**Computational**:
- All models converge or fail gracefully
- Diagnostics are clean (R-hat, ESS)

**Decision**:
- Can definitively choose best model
- Clear falsification decisions

---

## Philosophy: Adversarial Modeling

From `proposed_models.md`:

> "Our goal is not to confirm complete pooling. Our goal is to TRY to break it. If we fail, complete pooling is even stronger. If we succeed, we've learned something important the EDA missed."

**Key insight**: The best way to test a hypothesis is to try as hard as possible to falsify it. If it survives, we have much more confidence.

---

## Contact and Questions

This is an independent design from Designer 3. Compare with Designer 1 and Designer 2 proposals to see different perspectives on the same problem.

**Main questions this design addresses**:
1. Are the measurement errors actually accurate?
2. Is there hidden structure the EDA couldn't detect?
3. Does error scale with the quantity being measured?

**If all answers are "no"**: Complete pooling is strongly confirmed
**If any answer is "yes"**: EDA missed something critical

---

## Next Steps

1. Review `proposed_models.md` for full scientific justification
2. Review `experiment_plan.md` for implementation details
3. Review `stan_model_sketches.md` for code templates
4. Implement models in order: Baseline → Model 1 → Model 3 → Model 2
5. Report convergence diagnostics and model comparison

---

**Designer 3 Deliverables Summary**:
- 3 adversarial model classes (9 total variants)
- Full Stan implementations
- Falsification criteria for each
- Stress test validation procedures
- Decision tree for model selection
- Expected outcomes by scenario

**Total pages**: ~35 pages of detailed specifications

---

**END OF DESIGNER 3 DOCUMENTATION**

*Remember: Finding that a model is wrong is just as valuable as finding one that's right. Science advances through falsification.*
