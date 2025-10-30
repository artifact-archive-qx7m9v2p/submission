# Model Designer 3: Structural Change & Nonlinear Patterns
## Experiment Plan for Count Time Series with Regime Shifts

---

## Overview

This directory contains a complete Bayesian modeling strategy for count time series data exhibiting:
- **Exponential growth** (2.37× per year)
- **Severe overdispersion** (variance 70× mean)
- **Regime shifts** (7.8× increase early→late, heterogeneous dispersion)
- **High autocorrelation** (ACF lag-1 = 0.971)

**Design Philosophy**: Propose multiple competing model classes, each with explicit falsification criteria. Success = discovering which models the data reject, not confirming preconceptions.

---

## File Structure

```
/workspace/experiments/designer_3/
│
├── README.md                        # This file (start here!)
├── proposed_models.md               # Detailed model specifications (MAIN DOCUMENT)
├── model_decision_tree.md           # Decision flowchart & stopping rules
├── stan_templates.md                # Ready-to-use Stan code
│
└── (To be created during implementation):
    ├── model_2a_polynomial.stan
    ├── model_2b_poly_varying_phi.stan
    ├── model_1a_changepoint.stan
    ├── model_1b_changepoint_quad.stan
    ├── model_3a_two_state.stan
    ├── fit_models.py
    ├── diagnostics.py
    └── results/
        ├── model_comparison.csv
        ├── posterior_summaries/
        ├── diagnostic_plots/
        └── final_report.md
```

---

## Quick Start

### Step 1: Read the Documents (30 minutes)

1. **Start here**: `proposed_models.md` - Full model specifications with priors, falsification criteria, and scientific motivation
2. **Then**: `model_decision_tree.md` - When to stop, when to pivot, how to compare
3. **Finally**: `stan_templates.md` - Copy-paste ready Stan code and Python scripts

### Step 2: Fit Baseline Model (2 hours)

```bash
cd /workspace/experiments/designer_3

# Copy Stan template
cp stan_templates.md model_2a_polynomial.stan
# (Extract the Stan code from the markdown)

# Copy Python fitting script
cp stan_templates.md fit_models.py
# (Extract the Python code)

# Fit Model 2A (simplest)
python fit_models.py
```

**Checkpoint**: Does Model 2A pass diagnostics AND have good fit?
- **YES** → Stop here! Don't overfit. Write report.
- **NO** → Proceed to Step 3.

### Step 3: Fit Complex Models (if needed, 1-2 days)

```python
# Only if Model 2A inadequate:
# - Fit Model 1A (changepoint) if regime shift suspected
# - Fit Model 3A (hierarchical) if changepoint uncertain
# - Compare via LOO-CV
```

### Step 4: Final Report (4 hours)

See template in `model_decision_tree.md` → Communication Protocol section.

---

## Three Model Classes Proposed

### Model 1: Piecewise Changepoint (Discrete Regime Shift)

**Hypothesis**: Process changed abruptly at unknown time τ

**Key features**:
- Unknown changepoint τ ~ DiscreteUniform(10, 30)
- Regime-specific slopes, dispersion
- Tests: Discrete structural break (policy change? environmental shift?)

**I will abandon this if**:
- Posterior for τ is diffuse (no concentration)
- LOO worse than polynomial by >4 ELPD
- Changepoint at boundary (degenerate)

**Files**: `model_1a_changepoint.stan`, `model_1b_changepoint_quad.stan`

---

### Model 2: Smooth Polynomial (Continuous Nonlinearity)

**Hypothesis**: Growth rate accelerates smoothly (no discrete change)

**Key features**:
- Quadratic or cubic trend
- Time-varying dispersion: phi(t) = exp(γ₀ + γ₁×year)
- Occam's Razor: Simplest explanation

**I will abandon this if**:
- Residuals show clear regime structure
- Posterior predictive fails (observed outside envelope)
- Dispersion heterogeneity not captured by smooth trend

**Files**: `model_2a_polynomial.stan`, `model_2b_poly_varying_phi.stan`

---

### Model 3: Hierarchical Latent States (Soft Regimes)

**Hypothesis**: Latent "growth state" evolves, observations cluster within states

**Key features**:
- K=2 or 3 latent states
- State-specific parameters (intercept, slope, dispersion)
- Soft transitions: P(state | time) via logistic regression
- Hierarchical priors (partial pooling across states)

**I will abandon this if**:
- State posteriors uniform (no separation)
- WAIC penalty explodes (overfitting)
- LOO Pareto-k > 0.7 for >25% observations
- States temporally scrambled (not interpretable)

**Files**: `model_3a_two_state.stan`, (optional: `model_3b_three_state.stan`)

---

## Critical Decision Points

### Checkpoint 1: After Polynomial (Model 2A)

**Question**: Is smooth trend sufficient?

**Tests**:
- [ ] R² > 0.95?
- [ ] Residuals show no patterns?
- [ ] Posterior predictive checks pass?

**If YES**: **STOP**. Use polynomial. Don't fit complex models.
**If NO**: Proceed to changepoint or hierarchical.

---

### Checkpoint 2: After Changepoint (Model 1)

**Question**: Is discrete changepoint justified?

**Tests**:
- [ ] Posterior P(18 < τ < 26) > 0.80? (concentrated)
- [ ] LOO improves by >4 ELPD over polynomial?
- [ ] Posterior predictive shows sharp transition?

**If YES**: Changepoint model strong candidate.
**If NO**: Changepoint not supported, try hierarchical.

---

### Checkpoint 3: After Hierarchical (Model 3)

**Question**: Does complexity help?

**Tests**:
- [ ] States clearly separable?
- [ ] LOO improves by >4 ELPD?
- [ ] WAIC penalty reasonable (<15)?

**If YES**: Hierarchical may be best (report with simpler for comparison).
**If NO**: Use simpler model (polynomial or changepoint).

---

## Falsification Philosophy

### Core Principle

**We are NOT trying to confirm these models. We are trying to break them.**

Each model includes:
1. Explicit falsification criteria ("I will abandon this if...")
2. Stress tests (adversarial simulations)
3. Alternative hypotheses (what else could explain the pattern?)
4. Escape routes (what to try if this fails)

### Red Flags That Trigger Abandonment

**Computational**:
- >50% divergences after tuning
- Rhat > 1.05 for key parameters
- ESS < 100 despite long chains

**Statistical**:
- Posteriors at prior boundaries
- Posterior predictive disaster (generated data ≠ observed)
- LOO degeneracy (>50% high Pareto-k)

**Scientific**:
- Parameters make no scientific sense
- Model contradicts domain knowledge
- Results fragile to prior perturbations

---

## Model Comparison Strategy

### Metrics (in priority order)

1. **LOO-ELPD** (40%): Out-of-sample predictive accuracy
   - ΔLOO > 10: Strong evidence
   - ΔLOO 4-10: Moderate evidence
   - ΔLOO < 4: Models similar (prefer simpler)

2. **Posterior Predictive Checks** (30%): Does model capture key features?
   - Test statistics: Mean, variance, max, autocorrelation by period
   - Bayesian p-values should be ∈ [0.05, 0.95]

3. **Parsimony** (15%): Effective degrees of freedom
   - Prefer simplest adequate model
   - WAIC penalty indicates overfitting

4. **Interpretability** (10%): Can parameters be explained scientifically?
   - Posterior intervals should be interpretable
   - Parameters should align with domain knowledge

5. **Computational Stability** (5%): Diagnostics
   - Must pass Rhat, ESS, divergence checks
   - Unstable computation often indicates misspecification

### Decision Rule

**Final model must**:
- Best LOO by >4 ELPD (or within 1 SE and simpler)
- Pass all posterior predictive checks
- Pass all computational diagnostics (Rhat < 1.01, ESS > 400, <1% divergences)
- Have interpretable parameters
- Be robust to prior perturbations

If multiple models tie, **prefer the simplest**.

---

## Warning Signs & Diagnostics

### Green Flags (Model Likely Good)
- ✓ Rhat < 1.01, ESS > 400, <1% divergences
- ✓ Trace plots look like "fuzzy caterpillars"
- ✓ Posterior differs meaningfully from prior
- ✓ Posterior predictive data looks like observed data
- ✓ LOO has low Pareto-k values (<0.7)
- ✓ Parameters scientifically plausible

### Yellow Flags (Investigate)
- ⚠ ESS 100-400 (marginal)
- ⚠ 1-5% divergences (increase adapt_delta)
- ⚠ Some high Pareto-k (check those observations)
- ⚠ Posterior predictive p-value = 0.10 or 0.90 (borderline)
- ⚠ Prior-posterior overlap moderate (weak learning)

### Red Flags (Model Likely Wrong)
- ⚠️ Rhat > 1.05, ESS < 100, >5% divergences
- ⚠️ Multimodal posteriors (unless scientifically meaningful)
- ⚠️ Posteriors at prior boundaries
- ⚠️ Posterior predictive p-value < 0.05 or > 0.95
- ⚠️ >25% observations with Pareto-k > 0.7
- ⚠️ Parameters scientifically implausible

---

## Expected Outcomes

### Most Likely (70%)
- **Polynomial (Model 2A/2B) is adequate**
- Smooth acceleration captures main pattern
- Time-varying dispersion may be needed
- R² ≈ 0.96, clean residuals

### Alternative 1 (20%)
- **Clear changepoint around observation 20-22**
- Posterior τ concentrated (SD ≈ 2-3)
- Regime-specific parameters differ substantially
- Evidence for discrete structural break

### Alternative 2 (8%)
- **Hierarchical model with soft transitions**
- Two states interpretable (low/high growth)
- Gradual regime change, not discrete
- More complex but better fit

### Unexpected (2%)
- **All models fail**
- Need to reconsider data quality or likelihood
- Pivot to alternative approach (GP, state-space, different family)

---

## Implementation Timeline

### Week 1: Baseline (Required)
- [ ] Day 1-2: Implement Model 2A (polynomial), run diagnostics
- [ ] Day 3: Posterior predictive checks, LOO-CV
- [ ] Day 4: Prior sensitivity analysis
- [ ] Day 5: Decision: Stop or continue?

### Week 2: Changepoint (If Needed)
- [ ] Day 1-2: Implement Model 1A (piecewise linear)
- [ ] Day 3: Debug convergence issues (expect some!)
- [ ] Day 4: Compare to polynomial via LOO
- [ ] Day 5: Decision: Changepoint justified?

### Week 3: Hierarchical (If Needed)
- [ ] Day 1-3: Implement Model 3A (two-state)
- [ ] Day 4: Check identifiability, state recovery
- [ ] Day 5: Final model comparison

### Week 4: Finalization
- [ ] Day 1: Comprehensive diagnostics on best model
- [ ] Day 2: Out-of-sample validation
- [ ] Day 3: Sensitivity analyses
- [ ] Day 4-5: Write final report

**Total**: 3-4 weeks, ~30-40 hours

---

## Key Reminders

1. **Simplicity first**: Start with Model 2A. Stop if adequate.
2. **Diagnostics always**: Never interpret poor-quality samples.
3. **Falsification mindset**: Try to break models, not confirm them.
4. **Honest uncertainty**: Report limitations and alternatives.
5. **Scientific plausibility**: Parameters must make sense.
6. **Parsimony**: Prefer simplest adequate model.
7. **Computational stability**: Unstable = likely misspecified.
8. **Know when to stop**: Don't fit all models just because.

---

## Questions to Ask Continuously

1. **Does this model make scientific sense?**
2. **What would make me reject this model?**
3. **Am I overfitting?**
4. **Are parameters identifiable?**
5. **Does posterior predictive data look like real data?**
6. **Is the added complexity justified?**
7. **What am I assuming, and is it reasonable?**
8. **What alternative explanations exist?**

---

## Contact & Collaboration

**Designer**: Model Designer 3 (Structural Change Specialist)
**Focus**: Regime shifts, changepoints, nonlinear trends, hierarchical structure
**Output Directory**: `/workspace/experiments/designer_3/`

**Coordination with other designers**:
- Designer 1 (Baseline GLM): Compare simple vs complex
- Designer 2 (Time Series): Compare regime vs autoregressive structure
- Main Agent: Report checkpoints, seek guidance on pivots

---

## Final Thoughts

This is **not a recipe to follow blindly**. It's a **framework for discovery**.

The models proposed here are **hypotheses to test**, not answers to confirm. If data reject them, that's success—we learned something.

The goal is **finding truth**, not completing tasks. If the simplest model works, stop there. If all models fail, admit it and pivot.

**Science is iterative**. Be ready to:
- Abandon models that don't work
- Revise assumptions based on evidence
- Switch model classes entirely if needed
- Report honestly when uncertain

**Good luck, and may your posteriors be well-behaved!**

---

## Appendix: Quick Reference

### Model Selection at a Glance

| Scenario | Use This Model | Why |
|----------|---------------|-----|
| Smooth trend, no regimes | Model 2A (polynomial) | Simplest, likely adequate |
| Clear "elbow" in data | Model 1A (changepoint) | Tests discrete break |
| Heterogeneous dispersion | Model 2B (varying phi) | Time-varying variance |
| Uncertain about regimes | Model 3A (hierarchical) | Soft transitions |
| All models fail | Pivot to GP or state-space | Non-parametric fallback |

### Diagnostic Thresholds

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Rhat | < 1.01 | 1.01-1.05 | > 1.05 |
| ESS | > 400 | 100-400 | < 100 |
| Divergences | < 1% | 1-5% | > 5% |
| Pareto-k > 0.7 | < 10% | 10-25% | > 25% |
| PPC p-value | 0.05-0.95 | 0.02-0.98 | < 0.02 or > 0.98 |

### LOO Comparison

| ΔLOO | Interpretation | Action |
|------|---------------|--------|
| < 4 | Models similar | Use simpler |
| 4-10 | Moderate evidence | Consider complexity trade-off |
| > 10 | Strong evidence | Use better model (if diagnostics pass) |

---

**Document Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Ready for implementation
