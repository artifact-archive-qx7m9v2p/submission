# Conceptual Framework: Three Competing Explanations

## Designer 3 - Bayesian Model Design Philosophy

### The Central Mystery: The Variance Paradox

**Observed**: Between-school variance (124) < Expected from sampling error alone (166)
**Ratio**: 0.75 (observed is only 75% of expected)

**Question**: Why is observed variation LESS than we'd expect from random noise?

---

## Three Distinct Explanations

### Explanation 1: TRUE HOMOGENEITY
**Model 1: Near-Complete Pooling Hierarchical**

```
The Hypothesis:
Schools really are implementing the same effective intervention.
Apparent differences are just sampling noise.
The variance paradox is EVIDENCE of similarity, not artifact.

Visual Mental Model:
        True Effects (theta_i)
        ────────────────────────
        ████████████████████████  (all clustered near mu)

        Observed Effects (y_i)
        ─────────────────────────────────────────
        ████  ████    ████    ████   (spread by sampling error)
```

**Key Parameters**:
- tau: Small (< 5) - minimal between-school variation
- mu: Population average (around 8-12 based on data)
- theta_i: All shrink strongly toward mu

**What we'd observe if TRUE**:
- Posterior tau concentrated near 0
- Heavy shrinkage for all schools
- School 4 (26) shrinks down to ~10-12
- School 5 (-5) shrinks up to ~10-12
- Good posterior predictive checks
- LOO-CV stable (no influential schools)

**Falsification signature**:
- Posterior tau > 10 with narrow CI
- Poor PPC on variance
- High Pareto-k for multiple schools

---

### Explanation 2: SPARSE HETEROGENEITY
**Model 2: Flexible Horseshoe Hierarchical**

```
The Hypothesis:
MOST schools are similar (explaining low average heterogeneity),
BUT 1-2 schools are genuinely different outliers.
The variance paradox is dominated by the similar majority,
hiding the different minority.

Visual Mental Model:
        True Effects (theta_i)
        ────────────────────────────────────
        ████████████████████   ●        ●    (6 similar, 2 outliers)
                               ↑        ↑
                            School 5  School 4
                            (negative) (large)

        Horseshoe Detection:
        Schools 1,2,3,6,7,8: lambda_i ≈ 0.2  (strong shrinkage)
        School 5:            lambda_i ≈ 2.0  (minimal shrinkage)
        School 4:            lambda_i ≈ 1.5  (minimal shrinkage)
```

**Key Parameters**:
- tau: Moderate (5-15) - overall scale
- lambda_i: Most small (<0.5), 1-2 large (>1)
- mu: Average of majority cluster
- theta_i: Differential shrinkage

**What we'd observe if TRUE**:
- Clear bimodal pattern in lambda posteriors
- School 5 and/or 4 have lambda > 1
- Remaining schools have lambda < 0.5
- Better LOO-CV for outlier schools
- Moderate posterior tau
- Outliers retain wide posteriors

**Falsification signature**:
- All lambda_i in narrow range (0.3-0.6)
- No schools clearly identified as outliers
- No improvement over Model 1 in LOO-CV

**Scientific interpretation if TRUE**:
- Schools 4 and 5 implemented different protocols
- Or had different populations
- Or had methodological differences
- Requires follow-up investigation

---

### Explanation 3: MEASUREMENT ERROR
**Model 3: Sigma Misspecification Robust**

```
The Hypothesis:
The variance paradox is an ARTIFACT of incorrect sigma_i values.
Some schools over-reported uncertainty, some under-reported.
Once we correct for measurement error, heterogeneity becomes apparent.

Visual Mental Model:
        Reported Sigmas (sigma_i)
        School 8: sigma = 18  ← TOO HIGH (psi < 1)
        School 5: sigma = 9   ← maybe OK
        School 1: sigma = 15  ← TOO HIGH (psi < 1)
        ...

        After Correction (sigma_i * psi_i):
        School 8: sigma_true = 18 * 0.7 = 12.6
        School 5: sigma_true = 9 * 1.0 = 9.0
        School 1: sigma_true = 15 * 0.8 = 12.0

        Result: Variance ratio closer to 1.0
```

**Key Parameters**:
- omega: Scale of misspecification (hopefully > 0)
- psi_i: Correction factors (hope some ≠ 1)
- tau: True heterogeneity (after correction)
- sigma_true_i: Corrected uncertainties

**What we'd observe if TRUE**:
- Posterior omega clearly > 0 (e.g., 0.2-0.4)
- Some psi_i posteriors exclude 1.0
- School 8 (largest sigma) might have psi < 1
- Corrected variance ratio → 1.0
- Better LOO-CV than Model 1
- tau might increase after correction

**Falsification signature**:
- Posterior omega near 0
- All psi_i posteriors include 1.0
- Corrections implausibly large (psi > 3)
- No improvement in predictive performance

**Scientific interpretation if TRUE**:
- Study quality varied across schools
- Sample sizes misreported
- Within-school heterogeneity underestimated
- Need to contact original investigators

---

## Why These Three?

### Model 1: The EDA Default
- EDA overwhelmingly suggests this
- I² = 1.6% is extremely low
- Chi-square test p = 0.42
- Variance ratio = 0.75
- **Most likely to be correct**

### Model 2: The "What if outliers?" Alternative
- Addresses concern that I² averages over all schools
- School 5 is only negative value
- School 4 is only significant value
- Horseshoe is ideal for sparse signals
- **Less likely, but important to check**

### Model 3: The "Question the data" Alternative
- Addresses concern about taking sigmas as gospel
- Sigmas are estimates, not knowns
- Variance paradox is unusual - could be artifact
- Educational studies vary in quality
- **Least likely, but would be important discovery**

---

## The Falsificationist Approach

### Traditional approach (BAD):
1. Propose model based on EDA
2. Fit model
3. Report results
4. Declare success

**Problem**: Confirmation bias. We see what we expect.

### My approach (GOOD):
1. Propose THREE competing models
2. State clearly what would falsify each
3. Fit all models (or conditionally based on Stage 1)
4. Actively look for evidence AGAINST each model
5. Select based on falsification resistance

**Key insight**: Science advances by elimination, not confirmation.

---

## Decision Framework

### Stage 1: Fit Model 1
**Questions**:
- Is tau small? → Homogeneity supported
- Do PPCs pass? → Model adequate
- Any LOO flags? → Potential issues

**Decisions**:
- If all good: STOP, report Model 1
- If tau large: Proceed to Model 2
- If variance paradox unresolved: Proceed to Model 3

### Stage 2A: Fit Model 2 (if needed)
**Questions**:
- Are outliers identified? → Sparse heterogeneity supported
- Does LOO improve? → Complexity justified
- Do outliers make sense? → Not just overfitting

**Decisions**:
- If yes to all: Model 2 wins
- If no outliers found: Model 1 wins
- If results ambiguous: Report both with caveats

### Stage 2B: Fit Model 3 (if needed)
**Questions**:
- Is omega > 0? → Misspecification supported
- Do corrections help? → Not just noise
- Are corrections plausible? → Not compensating for other issues

**Decisions**:
- If yes to all: Model 3 wins (important finding!)
- If omega ≈ 0: Model 1 wins
- If corrections implausible: Something else wrong

### Stage 3: Final Comparison
**Compare all fitted models**:
- LOO-CV ELPD differences
- Posterior predictive checks
- Scientific plausibility
- Computational stability

**Select**:
- Model with best evidence
- Or acknowledge uncertainty if tied
- Or abandon all if all fail

---

## What Each Model Teaches Us

### If Model 1 wins:
**Learning**: Pooling is powerful. With high uncertainty and low heterogeneity, borrowing strength across studies dramatically improves inference. The Eight Schools are essentially interchangeable.

**Implication**: Future studies should use hierarchical modeling. Individual school estimates are unreliable.

### If Model 2 wins:
**Learning**: Most schools are interchangeable, but outliers exist. Horseshoe prior elegantly handles this. Schools 4/5 need investigation.

**Implication**: Clustering and outlier detection are valuable. Follow up on identified outliers to understand why they differ.

### If Model 3 wins:
**Learning**: Never trust reported standard errors without scrutiny. Measurement error propagates. Meta-analyses should account for SE uncertainty.

**Implication**: Contact original investigators. Develop methods for SE reliability assessment.

### If all models fail:
**Learning**: Our understanding of the data generation process is fundamentally wrong. Need to reconsider:
- Are schools exchangeable?
- Is normality appropriate?
- Are there unreported covariates?
- Is the data itself correct?

**Implication**: More data collection or qualitative investigation needed.

---

## Philosophical Stance

### On Model Complexity:
"All models are wrong, but some are useful" - George Box

- Model 1: Wrong but likely useful
- Model 2: More wrong (more complex), possibly more useful
- Model 3: Even more wrong, possibly most useful if sigma issues real

### On Uncertainty:
"It is better to be roughly right than precisely wrong" - John Maynard Keynes

- With n=8, some questions may be unanswerable
- High posterior uncertainty is OK
- Don't force certainty where none exists

### On Falsification:
"No amount of observations can prove a theory, but one observation can disprove it" - Karl Popper

- Can't prove Model 1 is correct
- CAN show it's better than alternatives
- CAN show when it fails
- Science advances by elimination

### On Simplicity:
"Entities should not be multiplied without necessity" - Occam's Razor

- Prefer Model 1 if performance is equivalent
- Complexity needs to be JUSTIFIED by evidence
- Simplicity is a virtue, not a vice

---

## Expected Outcome

**My prediction** (before seeing posterior):
- Model 1 will fit well (tau < 5, good PPCs)
- Model 2 will show no clear outliers (all lambda similar)
- Model 3 will show omega ≈ 0 (no misspecification)
- Conclusion: Homogeneity is real, pool strongly

**How I'll know I'm wrong**:
- Model 1 has posterior tau > 10
- Model 2 clearly identifies Schools 4 & 5 as outliers
- Model 3 shows omega > 0.3 with corrections that make sense
- LOO-CV strongly prefers Model 2 or 3

**I am ready to be wrong.** That's the point of this exercise.

---

## Meta-Commentary: Why This Approach?

### Problem with typical analyses:
1. Researcher has favorite model
2. Fits that model
3. Finds supporting evidence (confirmation bias)
4. Reports success
5. May miss important issues

### This approach instead:
1. Propose multiple competing models
2. State falsification criteria upfront
3. Actively seek evidence against each
4. Select based on survival, not confirmation
5. Acknowledge when data insufficient

**Result**: More trustworthy science. If Model 1 survives this gauntlet, we can be confident. If it doesn't, we learned something important.

---

## Files in This Framework

- **This document**: Conceptual overview
- **proposed_models.md**: Technical specifications (911 lines)
- **model_comparison_table.md**: Quick reference table
- **README.md**: Executive summary

All in: `/workspace/experiments/designer_3/`

---

## Final Thought

**The goal is not to confirm that schools are similar (even though EDA suggests this).**

**The goal is to discover what the data actually tell us, by trying hard to prove ourselves wrong.**

If we still believe homogeneity after that process, we've earned it.

---

**Designer 3, signing off.**
