# Executive Summary: Smooth Nonlinear Models
## Designer 2 Deliverable

**Author**: Model Designer 2 (Smooth Nonlinear Specialist)
**Date**: 2025-10-29
**Status**: Ready for Implementation

---

## The Question

Can the apparent structural break at observation 17 be explained by **smooth, continuous acceleration** rather than a **discrete regime change**?

---

## Three Proposed Models

### 1. Polynomial Regression (Baseline)
- **Approach**: Quadratic/cubic trend with AR(1) errors
- **Strength**: Simple, fast, interpretable
- **Weakness**: Limited flexibility, likely underfit
- **Expected**: Will fail (ΔLOO ≈ -20 vs changepoint)

### 2. Gaussian Process Regression (Most Flexible)
- **Approach**: GP with squared exponential kernel, mean trend, AR(1) errors
- **Strength**: Maximum flexibility, principled uncertainty
- **Weakness**: O(N³) computation, may overfit
- **Expected**: Best smooth model, but likely still fails (ΔLOO ≈ -10 vs changepoint)

### 3. Penalized B-Spline Regression (Balanced)
- **Approach**: Cubic B-splines with smoothing penalty, AR(1) errors
- **Strength**: Local flexibility, efficient computation
- **Weakness**: Knot sensitivity, may show artifacts
- **Expected**: Middle ground (ΔLOO ≈ -15 vs changepoint)

---

## All Models Include

- **Negative Binomial likelihood** (variance/mean = 67.99)
- **Log link function** (exponential growth)
- **AR(1) autocorrelation** (ACF(1) = 0.944)
- **EDA-informed priors** (from `/workspace/eda/eda_report.md`)
- **Stan or PyMC implementation** (full PPL required)

---

## Critical Insight

**These models are designed to FAIL if a discrete break exists.**

This is not a bug—it's a feature. Good scientific models should fail on wrong data-generating processes.

---

## Decision Criteria

| LOO-ELPD Difference | Interpretation | Action |
|---------------------|----------------|--------|
| ΔLOO > -10 | Smooth sufficient | Use smooth model |
| ΔLOO -10 to -20 | Borderline | Check diagnostics |
| ΔLOO < -20 | Discrete break real | Use changepoint model |

Additional red flags:
- GP lengthscale < 0.2 (trying to capture discontinuity)
- First derivative discontinuous at observation 17
- Residual ACF(1) > 0.5 (autocorrelation not captured)
- Prior-posterior conflict (parameters at boundaries)

---

## Expected Outcome (Honest Prediction)

**Most likely (75% confidence)**: All smooth models FAIL

**Evidence**:
- EDA shows 730% growth rate increase at observation 17
- Four independent structural break tests converge
- Discrete regime change is most parsimonious
- Smooth models will show lengthscale shrinkage (GP), derivative discontinuity (all), poor LOO

**Conclusion**: Discrete break is real, use Designer 1's changepoint models

**Scientific value**: Systematic falsification confirms discrete break is robust finding, not artifact.

---

## Implementation Roadmap

1. **Fit Polynomial** (2-3 hours) → Fast baseline
2. **Fit GP** (4-6 hours) → Maximum flexibility test
3. **Fit Spline** (3-4 hours) → Semi-parametric alternative
4. **Model Comparison** (2-3 hours) → LOO, residuals, derivatives
5. **Final Decision** (1-2 hours) → Compare to Designer 1's models

**Total**: ~14 hours

---

## Deliverable Files

### Core Documentation
1. **`proposed_models.md`** (28KB) - Full mathematical specifications
   - Complete model definitions
   - Prior justifications from EDA
   - Falsification criteria for each model
   - Implementation code (Stan + PyMC)

2. **`falsification_protocol.md`** (10KB) - Systematic testing framework
   - 7 falsification tests
   - Decision flowcharts
   - Stopping rules
   - Expected failure modes

3. **`implementation_guide.md`** (21KB) - Step-by-step code
   - Complete working examples
   - Diagnostic procedures
   - Comparison methods
   - Expected timeline

4. **`predictions.md`** (14KB) - Falsifiable predictions
   - Before-fitting predictions
   - Parameter value forecasts
   - LOO-ELPD expectations
   - What would change our mind

### Supporting Files
5. **`model_summary.md`** (2KB) - Quick reference
6. **`model_architecture.txt`** (7KB) - Visual diagrams
7. **`README.md`** (3KB) - Overview
8. **`EXECUTIVE_SUMMARY.md`** - This file

---

## Key Innovations

### 1. Falsification-First Design
Every model includes explicit criteria for rejection:
- "I will abandon this model if..."
- Not just "what would make it work" but "what would prove it wrong"

### 2. Honest Predictions
Pre-specified predictions documented before fitting:
- Expected LOO values
- Expected parameter ranges
- Expected failure modes
- Prevents post-hoc rationalization

### 3. Adversarial Testing
Designed tests that SHOULD make models fail:
- Synthetic data with true discrete break
- Leave-future-out CV (extrapolation stress test)
- First derivative discontinuity check

### 4. Decision Framework
Clear rules for when to abandon entire approach:
- ΔLOO < -20: Switch to changepoint models
- All tests fail: Discrete break confirmed
- Not "which smooth model" but "smooth vs discrete"

---

## Comparison to Designer 1

| Aspect | Designer 1 (Changepoint) | Designer 2 (Smooth) |
|--------|-------------------------|---------------------|
| **Assumption** | Discrete regime change | Continuous acceleration |
| **Models** | Fixed/Random changepoint | Polynomial/GP/Spline |
| **Flexibility** | Two regimes | Smooth function |
| **Expected Winner** | YES (75% confidence) | NO (25% confidence) |
| **Scientific Role** | Primary hypothesis | Falsification test |

**Complementarity**: Designer 1 proposes what likely works. Designer 2 tests if simpler alternatives can be ruled out.

---

## What Makes This Different?

Most modeling exercises:
1. Propose model
2. Fit model
3. Show it works
4. Conclude success

This modeling exercise:
1. Propose model
2. **Document how it should fail**
3. **Design tests to break it**
4. Fit model
5. **Apply falsification tests**
6. **Accept or reject based on evidence**
7. Conclude based on reality, not confirmation bias

---

## Success Metrics

**Traditional view**: Model fits well → Success

**Our view**:
- If smooth models work: Success (found simpler explanation)
- If smooth models fail AS PREDICTED: Success (confirmed discrete break)
- If smooth models fail UNEXPECTEDLY: Most interesting (learn something new)

**Failure** would be:
- Models fail but we can't diagnose why
- Results are borderline and inconclusive
- We rationalize unexpected results post-hoc

---

## Next Steps for Implementer

1. **Read** `proposed_models.md` for mathematical details
2. **Follow** `implementation_guide.md` for step-by-step code
3. **Apply** `falsification_protocol.md` for testing
4. **Compare** to Designer 1's changepoint models
5. **Decide** based on evidence, not preferences
6. **Document** what didn't work (as important as what did)

---

## Final Thoughts

**Philosophy**: The goal is finding truth, not completing a task list.

**Expectation**: Smooth models will likely fail. That's okay—it means discrete break is robust.

**Hope**: Whatever happens, evidence is decisive (not borderline).

**Commitment**: Will accept discrete break if evidence demands it. Won't force smooth models to work.

**Scientific Value**: Even if models fail, systematic falsification has value:
- Rules out smooth alternatives
- Confirms discrete break isn't artifact
- Demonstrates what evidence supports changepoint models
- Shows intellectual honesty in model comparison

---

## Contact Information

**Designer**: Model Designer 2 (Smooth Nonlinear Specialist)
**Focus**: Gaussian Processes, Splines, Polynomial Regression
**Philosophy**: Falsification-first, honest predictions, adversarial testing
**Location**: `/workspace/experiments/designer_2/`
**Status**: Complete, ready for implementation

---

## File Locations (Absolute Paths)

All files in: `/workspace/experiments/designer_2/`

1. `/workspace/experiments/designer_2/proposed_models.md`
2. `/workspace/experiments/designer_2/falsification_protocol.md`
3. `/workspace/experiments/designer_2/implementation_guide.md`
4. `/workspace/experiments/designer_2/predictions.md`
5. `/workspace/experiments/designer_2/model_summary.md`
6. `/workspace/experiments/designer_2/model_architecture.txt`
7. `/workspace/experiments/designer_2/README.md`
8. `/workspace/experiments/designer_2/EXECUTIVE_SUMMARY.md`

---

**Ready for implementation. Good luck, and may the best model win (even if it's not mine).**
