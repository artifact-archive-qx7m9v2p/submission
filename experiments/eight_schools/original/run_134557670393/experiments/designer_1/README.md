# Model Designer #1: Hierarchical/Multilevel Models

**Designer Focus**: Hierarchical structures, partial pooling, shrinkage strategies
**Date**: 2025-10-28
**Dataset**: Meta-analysis (J=8 studies, I²=0%, borderline pooled effect)

---

## Quick Navigation

### For Quick Overview
→ **Start here**: `model_summary.md` (7 KB)
- 3 models in table format
- Decision tree for model selection
- Falsification checklist
- Implementation priorities

### For Full Details
→ **Complete proposal**: `proposed_models.md` (28 KB)
- Full mathematical specifications
- Detailed prior justifications
- Comprehensive falsification criteria
- Stress tests and adversarial checks
- Implementation code and diagnostics

### For Model Comparison
→ **Side-by-side reference**: `model_comparison_table.md` (11 KB)
- Mathematical formulations compared
- Prior predictive distributions
- Shrinkage behavior analysis
- When to prefer each model
- Code templates

---

## Three Proposed Model Classes

### Model 1: Adaptive Hierarchical Meta-Analysis (STANDARD)
**Core idea**: Let the data decide between fixed and random effects via Half-Cauchy prior on heterogeneity (τ).

**Key features**:
- Normal hierarchy: θ_i ~ Normal(μ, τ²)
- Vague prior: τ ~ Half-Cauchy(0, 5)
- Adaptive shrinkage based on data

**When to use**: Default choice, most meta-analyses

**Falsify if**: Posterior predictive fails OR leave-one-out instability > 5 units

---

### Model 2: Robust Hierarchical Meta-Analysis (OUTLIER-RESISTANT)
**Core idea**: Use Student-t hierarchy to accommodate heavy-tailed effects and downweight outliers.

**Key features**:
- Heavy-tailed hierarchy: θ_i ~ Student-t(ν, μ, τ²)
- ν estimated from data: ν ~ Gamma(2, 0.1)
- Robust to Study 1 (y=28, potentially influential)

**When to use**: If Study 1 is problematic OR domain suggests outliers

**Falsify if**: ν posterior > 50 OR no LOO-CV improvement over Model 1

---

### Model 3: Informative Heterogeneity Meta-Analysis (EXTERNAL EVIDENCE)
**Core idea**: Incorporate external evidence about typical meta-analytic heterogeneity to stabilize inference with J=8.

**Key features**:
- Same hierarchy as Model 1
- Informative prior: τ ~ Half-Normal(0, 3)
- Based on literature (Turner et al. 2015: 19,000+ meta-analyses)

**When to use**: If external evidence reliable AND J too small for τ estimation

**Falsify if**: Prior-data conflict (p < 0.05) OR I²=0% conflicts with prior

**Expected**: LIKELY TO FAIL for this dataset (I²=0% unusual)

---

## Critical Philosophy: Falsification First

### What Makes This Design Different

**Traditional approach**:
1. Pick a model
2. Fit it
3. Report results
4. Done

**Our approach**:
1. Propose multiple competing models
2. **Define failure criteria BEFORE fitting**
3. Fit models
4. **Apply falsification checks**
5. **REJECT models that fail** (this is success!)
6. Report only models that survive

### Pre-Commitment to Rejection

**I WILL REJECT these models if**:
- All 3 fail falsification → Hierarchical framework is wrong for this data
- Leave-one-out shows Study 1 dominates → Dataset too fragile for pooling
- Posterior predictive fails systematically → Likelihood misspecified
- Prior sensitivity flips conclusions → Data too weak for inference

**Alternative approaches if hierarchical models fail**:
- Mixture models (2-component: high-effect vs low-effect studies)
- Fixed-effect only (if τ truly zero)
- Non-parametric Bayesian (Dirichlet process)
- Report "cannot estimate reliably" (honest answer)

---

## Key Insights from EDA

### What We Know
1. **I²=0%** but effect range is 31 points (-3 to 28)
2. **Large measurement errors** (σ: 9-18, mean 12.5)
3. **Borderline pooled effect** (7.69, 95% CI: -0.30 to 15.67)
4. **No individual study significant** (all p > 0.06)
5. **Study 1 potentially influential** (y=28, highest)

### The "Low Heterogeneity Paradox"
**Apparent contradiction**: How can effects range 31 points but I²=0%?

**Answer**: Large within-study variance (mean σ²=156) overwhelms between-study variance (var(y)=109).

**Implication**: I²=0% may reflect **lack of power** to detect heterogeneity, not true homogeneity.

**Modeling consequence**:
- Model 1 (vague prior on τ) will let τ → 0 if data suggests
- Model 3 (informative prior expecting τ > 0) may conflict with data
- This is a **testable disagreement** between models

---

## Decision Framework

### Phase 1: Fit and Diagnose (All 3 Models)
```
For each model:
  1. Run prior predictive check
  2. Fit with HMC (Stan/PyMC)
  3. Check convergence (R-hat, ESS, divergences)
  4. Apply model-specific falsification criteria
  5. Mark as PASS or FAIL
```

### Phase 2: Compare Survivors
```
If 0 models pass:
  → Hierarchical framework wrong
  → PIVOT to mixture/non-parametric

If 1 model passes:
  → Report that model
  → Sensitivity analysis

If 2+ models pass:
  → LOO-CV comparison
  → If ELPD diff < 4: Models equivalent, use simplest
  → If ELPD diff > 4: Prefer better-performing model
  → Consider Bayesian model averaging
```

### Phase 3: Sensitivity Analysis
```
For selected model(s):
  1. Leave-one-out (8 fits, drop each study)
  2. Prior sensitivity (vary prior SDs by ±50%)
  3. Likelihood robustness (Normal vs Student-t)

If conclusions change qualitatively:
  → Data too weak for strong inference
  → Report uncertainty, don't over-claim
```

---

## Implementation Plan

### Software Stack
- **Primary**: Stan (via CmdStanPy)
  - Better handling of hierarchical models
  - Superior diagnostics
  - Non-centered parameterization for funnel geometry

- **Alternative**: PyMC
  - More Pythonic
  - Good visualization tools (ArviZ)
  - May struggle with funnel if τ → 0

### Computational Strategy

**Parameterization decision**:
- Start with **centered** (standard): θ ~ Normal(μ, τ)
- If divergences occur: Switch to **non-centered**: θ = μ + τ·θ_raw

**Sampling settings**:
- Chains: 4
- Iterations: 4000 (2000 warmup, 2000 sampling)
- Target acceptance: 0.95 (conservative for small J)
- Max tree depth: 12

**Expected issues**:
1. Funnel geometry if τ → 0 (Model 1, 3) → Use non-centered
2. Slow sampling with Student-t (Model 2) → Increase iterations
3. Wide posteriors for τ with J=8 → Accept uncertainty, don't force precision

---

## What Success Looks Like

### Scientific Success (Primary Goal)
✓ Honest quantification: "We estimate μ = 7.5 ± 5, P(μ>0) = 0.52"
✓ Clear limitations: "With J=8, τ is weakly identified"
✓ Influential studies noted: "Study 1 drives 30% of posterior uncertainty"
✓ Falsification applied: "We rejected Model 3 due to prior-data conflict"

### Technical Success (Secondary Goal)
✓ Convergence: R-hat < 1.01, ESS > 400, zero divergences
✓ Validation: Parameter recovery simulation works
✓ Fit: Posterior predictive checks pass
✓ Comparison: LOO-CV provides clear model ranking

### Philosophical Success (Meta Goal)
✓ Transparency: All assumptions explicit
✓ Falsifiability: Rejection criteria stated before fitting
✓ Humility: Willing to say "data insufficient"
✓ Adaptability: Ready to pivot if models fail

---

## Files in This Directory

| File | Size | Purpose |
|------|------|---------|
| `README.md` | 7 KB | This file - navigation and overview |
| `model_summary.md` | 7 KB | Quick reference, decision tree, checklist |
| `proposed_models.md` | 28 KB | Full proposal with detailed specifications |
| `model_comparison_table.md` | 11 KB | Side-by-side model comparison, code templates |

**Total documentation**: ~53 KB, ~1800 lines

---

## Connection to Broader Project

### Input
- **EDA report**: `/workspace/eda/eda_report.md`
  - Three parallel analysts explored data
  - Key finding: I²=0% may be artifact of low power

### Output (This Directory)
- **Model proposals**: Three hierarchical model classes
- **Falsification criteria**: Pre-specified rejection rules
- **Implementation guidance**: Stan code, diagnostics, sensitivity analyses

### Next Steps
1. **Synthesis**: Main agent combines this with other designers' proposals
2. **Implementation**: Build Stan/PyMC models
3. **Execution**: Fit models, apply falsification checks
4. **Reporting**: Document results, survived models, rejected models, lessons learned

---

## Contact Points with Other Designers

### If Designer #2 proposes different model classes:
- Compare via LOO-CV (cross-validation)
- If their models strongly outperform: Hierarchical framework may be wrong
- If similar performance: Model class choice less critical than assumed

### If Designer #3 proposes overlapping models:
- Synthesize best elements from both proposals
- Compare prior choices (may differ)
- Run both as sensitivity analysis

### Model disagreement = valuable information:
- If hierarchical models and (say) mixture models both fit well: Data has multiple explanations
- If one class dominates: Strong evidence about data structure
- If all classes fail: Dataset has properties we didn't anticipate

---

## FAQ

**Q: Why three models instead of one?**
A: Multiple competing hypotheses force us to think about what distinguishes them. If all three fail, we learn hierarchical framework is wrong. If all three pass, we learn model choice doesn't matter much (good news!).

**Q: Why such detailed falsification criteria?**
A: Easy to fool yourself with Bayesian models - they always "fit". Pre-commitment to rejection criteria prevents confirmation bias.

**Q: What if I²=0% is real (true homogeneity)?**
A: Model 1 will capture this (τ → 0 is allowed). Model 3 may struggle (prior expects τ > 0). This disagreement is informative.

**Q: Why not just use frequentist meta-analysis?**
A: With J=8, frequentist methods struggle (τ² estimator can be unstable, inference assumes asymptotic normality). Bayesian methods handle small samples better and quantify uncertainty fully.

**Q: What if all models are rejected?**
A: **Good!** We learned something: the hierarchical framework doesn't fit this data. Pivot to mixture models, non-parametric methods, or report that reliable pooling isn't possible with this dataset.

---

## Key Takeaways

1. **Three hierarchical model classes** proposed: Adaptive (standard), Robust (outliers), Informative (external evidence)

2. **Falsification-first approach**: Pre-specified criteria for rejecting each model

3. **Expected outcome**: Model 1 likely wins, Model 3 likely fails, Model 2 depends on Study 1 influence

4. **Success ≠ completing the plan**: Success = discovering truth, even if that means rejecting all models

5. **Small sample (J=8) is challenging**: Expect wide posteriors, weak identification of τ, high sensitivity to priors and influential studies

6. **Hierarchical framework may be wrong**: If all models fail, that's valuable information about data structure

---

**Philosophy**: "The goal is not to defend a model, but to discover how it fails."

**Next**: Implementation phase - let's find out which (if any) of these models survive contact with the data.
