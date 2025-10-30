# Designer 3: Alternative Perspectives & Robustness

**Design Philosophy**: Challenge standard approaches, explore overlooked model classes, prioritize robustness and falsification.

---

## Overview

The EDA consensus recommends **Negative Binomial GLM** for this overdispersed count data (Var/Mean ≈ 70). This is a sound choice, but may be unnecessarily restrictive. I propose **three alternative Bayesian model classes** that offer fundamentally different perspectives on the data generation process.

---

## Three Alternative Models

### Model 1: Hierarchical Gamma-Poisson
**Perspective**: Overdispersion arises from **unobserved heterogeneity**

- Each observation has a latent "intensity" drawn from a Gamma distribution
- Mathematically equivalent to NegBin, but hierarchy makes heterogeneity explicit
- Can reveal patterns in random effects (e.g., correlation with time)
- **Advantage**: Separates systematic trend from random variation

### Model 2: Student-t Regression on Log-Counts
**Perspective**: Count structure is less important than **heavy-tailed residuals**

- Transforms counts to log-scale and uses robust Student-t regression
- Downweights extreme observations automatically
- Simpler interpretation (linear model on log-scale)
- **Advantage**: Robust to extreme values, direct coefficient interpretation

### Model 3: Conway-Maxwell-Poisson (COM-Poisson)
**Perspective**: NegBin's variance-mean relationship may be **too restrictive**

- Flexible dispersion parameter that nests Poisson as special case
- Doesn't impose quadratic variance-mean relationship
- Data-driven dispersion estimation
- **Advantage**: Tests whether NegBin assumptions are necessary

---

## Key Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| **`proposed_models.md`** | Complete model specifications, priors, falsification criteria | Modeler (main reference) |
| **`implementation_guide.md`** | Stan code, Python fitting scripts, diagnostics | Implementer (copy-paste ready) |
| **`falsification_plan.md`** | Explicit rejection criteria, stress tests, escape routes | Critical analyst |
| **`README.md`** | Overview and navigation | Everyone (start here) |

---

## Quick Start

### For Implementers

1. **Read**: `implementation_guide.md` for complete Stan code and Python scripts
2. **Priority order**:
   - Start with **Model 2** (Student-t): Fastest, simplest
   - Then **Model 1** (Hierarchical): Moderate complexity
   - Finally **Model 3** (COM-Poisson): Most complex, optional
3. **Expected time**: ~1 hour for all three models

### For Critical Reviewers

1. **Read**: `falsification_plan.md` for explicit rejection criteria
2. **Check**: Each model has clear "I will abandon this if..." statements
3. **Verify**: Stress tests are implemented and results documented

### For Decision Makers

1. **Read**: This README and executive summary in `proposed_models.md`
2. **Key question**: Do these alternatives improve on standard NegBin?
3. **If yes**: Use the winner
4. **If no**: Standard NegBin is validated from multiple perspectives

---

## Expected Outcomes

### Scenario A: One Alternative Wins (ΔELPD > 4)

**Interpretation**: Standard approach was suboptimal

**Action**: Use the winning model, report why it's better

**Example**: "Student-t regression outperforms NegBin because data has heavy-tailed residuals that NegBin's parametric form doesn't capture"

### Scenario B: All Models Perform Similarly (ΔELPD < 2)

**Interpretation**: Distributional choice doesn't matter with n=40

**Action**: Use simplest model (standard NegBin), report robustness

**Example**: "Results are robust to distributional assumptions—all three alternatives give consistent growth rate estimates (β₁ ≈ 0.85)"

### Scenario C: All Alternatives Fail

**Interpretation**: Standard NegBin is well-justified

**Action**: Use standard NegBin, cite validation from alternatives

**Example**: "We tested hierarchical, robust, and flexible dispersion models; all converge to standard NegBin, validating the EDA recommendation"

### Scenario D: All Models Fail Var/Mean Check

**Interpretation**: Homogeneous dispersion assumption is wrong

**Action**: Pivot to time-varying dispersion models

**Example**: "All models fail to recover Var/Mean ≈ 70, suggesting dispersion varies with time. Next: fit heteroscedastic NegBin with log(φ[i]) = γ₀ + γ₁×year[i]"

---

## Critical Success Criteria

**This experiment succeeds if**:

1. **Scientific learning**: We understand why one approach is better/worse
2. **Robust conclusions**: Key findings (growth rate) are consistent across models
3. **Honest uncertainty**: We report where data is ambiguous
4. **Actionable insights**: Decision-makers know which model to use

**This experiment FAILS if**:

1. **Blind fitting**: We fit models without understanding failures
2. **Cherry-picking**: We report only the "best" model without context
3. **False confidence**: We claim precision when data is ambiguous
4. **Ignoring failures**: We don't document what went wrong

---

## Model Comparison Strategy

### Step 1: Fit All Models (Priority Order)

```bash
# Model 2: Student-t (2 min)
python fit_model_2.py

# Model 1: Hierarchical (5 min)
python fit_model_1.py

# Model 3: COM-Poisson (20 min, optional)
python fit_model_3.py
```

### Step 2: Check Convergence

- Rhat < 1.01 for all parameters
- ESS > 400 (bulk and tail)
- No divergent transitions
- Reasonable runtimes

### Step 3: Compare LOO-CV

```python
import arviz as az

# Compare models
comparison = az.compare({
    'Student-t': idata_2,
    'Hierarchical': idata_1,
    'COM-Poisson': idata_3  # if fitted
})

print(comparison)
```

**Decision rule**:
- ΔELPD > 4: Clear winner
- ΔELPD < 2: Models are tied (use simplest)
- Between 2-4: Report uncertainty

### Step 4: Posterior Predictive Checks

**Must verify**:
1. Var/Mean ratio ≈ 70
2. Distribution of C_rep matches C
3. No systematic residual patterns
4. 95% intervals have appropriate coverage

### Step 5: Report Findings

Create `summary_report.md` with:
- Model comparison table (LOO-CV)
- Parameter estimates (all models)
- Posterior predictive check results
- Interpretation and recommendation
- Limitations and caveats

---

## What Makes These Models "Alternative"?

| Aspect | Standard NegBin | Alternative Approaches |
|--------|-----------------|------------------------|
| **Likelihood** | Negative Binomial | Student-t (log-scale), COM-Poisson |
| **Scale** | Count scale | Log-scale (Model 2) |
| **Dispersion** | Fixed parameter φ | Random effects (M1), flexible ν (M3) |
| **Robustness** | Parametric assumptions | Heavy tails (M2), flexible V-M relationship (M3) |
| **Interpretability** | GLM coefficients | Direct growth rate (M2), latent heterogeneity (M1) |

---

## Why These Models Might Fail (And That's OK)

### Model 1: Hierarchical Gamma-Poisson

**Might fail because**:
- Random effects show no structure (just noise)
- Computational cost outweighs benefits
- Reduces to standard NegBin anyway

**If it fails, we learn**: Overdispersion is homogeneous, no unobserved heterogeneity

### Model 2: Student-t Regression

**Might fail because**:
- Count discreteness matters (can't treat as continuous)
- Back-transformation doesn't preserve distribution
- ν > 40 (data is Normal-like, not heavy-tailed)

**If it fails, we learn**: Count structure is important, need proper count models

### Model 3: COM-Poisson

**Might fail because**:
- Computational cost is prohibitive
- Reduces to NegBin (same variance-mean relationship)
- ν ≈ 1 (data is actually Poisson, contradicting EDA)

**If it fails, we learn**: NegBin's parametric form is appropriate

---

## Red Flags: When to Stop and Reconsider

### Computational Red Flags

- Divergent transitions >1% after tuning
- Rhat > 1.01 after increasing warmup
- Runtime >1 hour for any model
- ESS < 100 for any parameter

**Action**: Document failure mode, investigate pathology, consider simpler models

### Scientific Red Flags

- All models fail Var/Mean check (all << 70 or >> 70)
- Parameter estimates differ by >50% across models
- High prior-posterior overlap (data not informative)
- LOO-CV SE > |ΔELPD| (no clear winner)

**Action**: Pivot strategy (e.g., time-varying dispersion), report honest uncertainty

---

## Links to Key Sections

### In `proposed_models.md`:

- **Model 1 specification**: Lines 47-130
- **Model 2 specification**: Lines 132-218
- **Model 3 specification**: Lines 220-312
- **Model comparison strategy**: Lines 314-370
- **Falsification criteria**: Lines 372-420
- **Expected outcomes**: Lines 422-470

### In `implementation_guide.md`:

- **Model 1 Stan code**: Lines 23-95
- **Model 2 Stan code**: Lines 143-210
- **Model 3 Stan code**: Lines 258-330
- **Troubleshooting guide**: Lines 450-510

### In `falsification_plan.md`:

- **Model 1 rejection criteria**: Lines 25-85
- **Model 2 rejection criteria**: Lines 87-145
- **Model 3 rejection criteria**: Lines 147-200
- **Master decision tree**: Lines 320-380

---

## Contact Points for Questions

### "Which model should I fit first?"

→ **Model 2 (Student-t)**: Fastest, simplest, establishes baseline

### "What if all models give similar LOO-CV?"

→ **Good news!** Results are robust. Use simplest (standard NegBin) and report robustness

### "What if my favorite model doesn't converge?"

→ **Follow falsification plan**: Document failure, investigate cause, try escape routes

### "How do I know if I'm done?"

→ **Checklist**:
- [ ] At least 2 models converged (Rhat < 1.01)
- [ ] LOO-CV comparison completed
- [ ] Winner identified (or tie reported)
- [ ] Posterior predictive checks run
- [ ] Var/Mean ≈ 70 validated (or failure documented)
- [ ] Summary report written

---

## Philosophy: Falsification Over Confirmation

**Traditional approach**:
1. Propose model
2. Fit model
3. If it works, report success
4. If it fails, try another

**This approach**:
1. Propose multiple models
2. Define explicit failure criteria
3. Actively try to break each model
4. Report both successes AND failures
5. Learn from what doesn't work

**Why?**
- Confirmation bias is easy; falsification is hard
- Failures teach us about the data
- Robust findings emerge from multiple perspectives
- Honest uncertainty builds trust

---

## Final Thoughts

The EDA recommends Negative Binomial GLM. **I'm not disputing that.** It's a solid choice.

But science progresses by asking:
- **Could there be a better way?**
- **What assumptions are we making?**
- **How would we know if we're wrong?**

These three alternative models ask:
1. Is overdispersion really homogeneous? (Model 1)
2. Is count structure really essential? (Model 2)
3. Is NegBin's variance-mean relationship really correct? (Model 3)

**If all three alternatives fail**: NegBin is validated from multiple angles (success!)

**If one alternative wins**: We've discovered a better model (success!)

**If all models fail Var/Mean check**: We've learned dispersion is time-varying (success!)

**Either way, we learn something.**

---

## Next Steps

1. **Implementer**: Start with `implementation_guide.md`, fit Model 2 first
2. **Reviewer**: Read `falsification_plan.md`, verify stress tests are run
3. **Decision-maker**: Wait for `summary_report.md`, review model comparison
4. **Skeptic**: Challenge everything in this README—that's the point!

---

**Generated**: 2025-10-29
**Designer**: Alternative Perspectives Specialist (Designer 3)
**Status**: Ready for implementation
**Estimated time**: 1 hour (all models) or 40 minutes (Models 1-2 only)
