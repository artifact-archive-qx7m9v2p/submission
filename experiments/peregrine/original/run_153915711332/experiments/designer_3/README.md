# Designer 3: Structural Hypotheses and Model Complexity

**Focus Area:** Changepoint/regime-switching, hierarchical structures, flexible functional forms (GPs, splines), hypothesis testing

**Designer:** Model Designer 3
**Date:** 2025-10-29

---

## Quick Navigation

### Main Documents

1. **[proposed_models.md](proposed_models.md)** - MAIN DELIVERABLE
   - Full probabilistic specifications for 3 model classes
   - Detailed rationale, priors, falsification criteria
   - Scientific hypotheses being tested
   - 39KB, comprehensive technical document

2. **[model_summary.md](model_summary.md)** - QUICK REFERENCE
   - One-page summary of three models
   - Expected parameter ranges
   - Red flags and decision points
   - 2.4KB, perfect for quick lookup

3. **[implementation_plan.md](implementation_plan.md)** - EXECUTION GUIDE
   - Phase-by-phase implementation roadmap
   - Code templates and diagnostics checklists
   - Contingency plans and timelines
   - 12KB, practical implementation guide

---

## Three Proposed Models

### Model 1: Hierarchical Changepoint (PRIORITY 1)
**Scientific Story:** System underwent discrete regime shift at year ≈ 0.3

**Key Innovation:** Bayesian estimation of changepoint location with regime-specific parameters

**Why This Might Win:**
- Direct test of strongest EDA finding (CUSUM shows clear break)
- Most interpretable (before/after intervention)
- 4.5× mean increase suggests real structural change

**Why This Might Fail:**
- "Changepoint" could be artifact of smooth acceleration
- May generate unrealistic discontinuities in predictions

---

### Model 2: Gaussian Process (PRIORITY 2)
**Scientific Story:** Smooth acceleration without discrete breaks

**Key Innovation:** Nonparametric flexible function with Negative Binomial likelihood

**Why This Might Win:**
- EDA shows polynomials fit well (R² = 0.96 for quadratic)
- No need to assume specific functional form
- Naturally handles autocorrelation through kernel

**Why This Might Fail:**
- May overfit with N=40
- Computational challenges in Stan
- If true changepoint exists, GP will struggle

---

### Model 3: Latent State-Space (PRIORITY 3)
**Scientific Story:** Unobserved latent process evolving smoothly, observations are noisy

**Key Innovation:** Separates signal (state evolution) from noise (observation error)

**Why This Might Win:**
- ACF = 0.989 suggests strong latent process
- Natural for non-stationary I(1) data
- Best handles extreme autocorrelation

**Why This Might Fail:**
- Most complex, identification issues
- Latent state might just be smoothed observed data
- Over-parameterized for N=40

---

## Key Design Principles

1. **Falsification mindset:** Each model includes explicit "abandon if..." criteria
2. **Multiple competing hypotheses:** Test fundamentally different data generation stories
3. **Plan for failure:** Detailed contingency plans if models don't converge
4. **Scientific interpretability:** Models test specific hypotheses, not just statistical fits

---

## Critical EDA Findings Driving Design

- **Extreme overdispersion:** Variance/Mean = 67.99 → All models use Negative Binomial
- **Massive autocorrelation:** ACF(1) = 0.989 → Model 3 focuses on this
- **Changepoint evidence:** CUSUM minimum at year = 0.3 → Model 1 tests this
- **Smooth polynomial fits:** R² = 0.96 for quadratic → Model 2 explores this
- **I(1) non-stationary:** First diff reduces variance 98% → Informs state-space design

---

## Expected Outcomes

| Scenario | Model 1 | Model 2 | Model 3 | Action |
|----------|---------|---------|---------|--------|
| Changepoint is real | Win | Lose | Lose | Use Model 1 |
| Growth is smooth | Lose | Win | Maybe | Use Model 2 |
| Strong latent structure | Maybe | Maybe | Win | Use Model 3 |
| All models equivalent | Tie | Tie | Tie | Choose simplest (Model 1) |
| All models fail | Fail | Fail | Fail | Pivot to frequentist or simpler models |

---

## Files to be Generated

### Stan Models
- `models/model_1_changepoint.stan` - Hierarchical changepoint model
- `models/model_2_gp.stan` - Gaussian process model
- `models/model_3_statespace.stan` - Latent state-space model

### Python Scripts
- `fit_models.py` - Main fitting script for all three models
- `diagnostics.py` - Convergence checks (Rhat, ESS, divergences)
- `compare_models.py` - LOO-CV, posterior predictive checks

### Results
- `results/loo_comparison.csv` - Model comparison via LOO-CV
- `results/posterior_summaries.csv` - Parameter estimates
- `results/ppc_results.txt` - Posterior predictive check outcomes
- `results/visualizations/` - Plots for all models

---

## Implementation Timeline

| Task | Time | Priority |
|------|------|----------|
| Model 1 implementation | 3h | HIGH |
| Model 2 implementation | 4h | HIGH |
| Model 3 implementation | 5h | MEDIUM |
| Model comparison | 3h | HIGH |
| Sensitivity analysis | 2h | MEDIUM |
| **Total** | **17-20h** | - |

---

## Decision Criteria

### Statistical Fit (50% weight)
- LOO-ELPD (primary metric)
- WAIC (secondary)
- Posterior predictive checks

### Computational Health (25% weight)
- Rhat < 1.01
- ESS > 100
- Divergences < 1%

### Scientific Interpretability (25% weight)
- Does model tell coherent story?
- Are parameters meaningful?
- Can results guide future research?

---

## Red Flags

**STOP and reconsider if:**
1. All models fail to converge after tuning
2. All models fail posterior predictive checks
3. Dispersion parameter phi > 100 or < 1 in all models
4. LOO completely fails (Pareto k > 0.7 for many points)

**Action:** Pivot to simpler approaches (parametric GLM, frequentist methods)

---

## What Success Looks Like

1. At least one model converges cleanly (Rhat < 1.01)
2. LOO-CV provides clear ranking OR shows equivalence
3. At least one model is clearly falsified
4. Winning model passes posterior predictive checks
5. Clear scientific story emerges

---

## What Failure Looks Like (and that's OK!)

1. None of the three model classes fit
2. Computational issues across all models
3. All models generate unrealistic predictions

**This is valuable scientific information!** It tells us:
- Data is more complex than anticipated
- N=40 is insufficient for these model classes
- Need different approach (simpler models, more data, domain constraints)

---

## Contact and Questions

This is a complete, self-contained model design ready for implementation. The three models represent fundamentally different scientific hypotheses about the data-generating process.

**Philosophy:** Success = discovering the truth, even if it means rejecting all proposed models.

---

## Quick Start

1. Read `model_summary.md` for overview (2 minutes)
2. Read `proposed_models.md` for full specifications (30 minutes)
3. Follow `implementation_plan.md` for step-by-step execution (17-20 hours)

**All models are fully specified with:**
- Complete Stan code templates
- Prior justifications
- Falsification criteria
- Expected computational challenges
- Contingency plans

Ready to implement!
