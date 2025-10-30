# Model Designer 2: Complete Index

**Total Documentation**: 3,261 lines across 5 documents
**Total Size**: 108 KB
**Focus**: Bayesian models with temporal correlation structures
**Date**: 2025-10-29

---

## Quick Navigation

| Document | Lines | Purpose | Start Here If... |
|----------|-------|---------|-----------------|
| **SUMMARY.md** | 418 | Visual overview, tables | You want quick reference |
| **README.md** | 403 | Navigation guide | You're new to this work |
| **proposed_models.md** | 1,132 | Full specifications | You need theory + math |
| **stan_implementation_guide.md** | 762 | Code + debugging | You're implementing |
| **model_selection_framework.md** | 546 | Decision rules | You're choosing models |

---

## Reading Paths

### Path 1: Quick Overview (15 minutes)
1. Read: SUMMARY.md (visual overview)
2. Skim: README.md (executive summary)
3. Look at: Three model diagrams in SUMMARY.md

**Goal**: Understand the three model classes and why they exist

---

### Path 2: Theory Deep Dive (60 minutes)
1. Read: README.md (context)
2. Read: proposed_models.md, Model 1 (NB-AR1)
3. Read: proposed_models.md, Model 2 (NB-GP)
4. Read: proposed_models.md, Model 3 (NB-RW)
5. Review: Falsification criteria for each

**Goal**: Understand mathematical specifications and theoretical justifications

---

### Path 3: Implementation Focus (45 minutes)
1. Skim: proposed_models.md (get model equations)
2. Read: stan_implementation_guide.md (full Stan code)
3. Study: Python interface code
4. Review: Troubleshooting guide

**Goal**: Be ready to write and debug Stan code

---

### Path 4: Model Selection (30 minutes)
1. Read: model_selection_framework.md (decision tree)
2. Study: Quantitative criteria (LOO, Pareto-k)
3. Review: Reporting templates
4. Check: Red flags section

**Goal**: Be ready to choose between fitted models

---

## Document Contents

### SUMMARY.md (418 lines)

**Sections**:
- Visual overview diagram
- Three models at a glance (compact)
- Key differences table
- Decision flowchart
- Parameter interpretation tables
- Posterior predictive checks checklist
- Prior sensitivity tests
- Expected outcomes
- Success criteria

**Best for**: Quick reference, tables, visual diagrams

**Key tables**:
- Model comparison (line 67)
- Parameter interpretation (lines 160-190)
- Computational benchmarks (line 242)

---

### README.md (403 lines)

**Sections**:
1. Executive Summary
2. Document Overview
3. Quick Start
4. Key Design Decisions
5. Critical Insights
6. Expected Scenarios
7. Red Flags and Contingencies
8. Next Steps
9. Connection to Other Designers
10. Summary: Design Philosophy

**Best for**: First-time readers, navigation, understanding design choices

**Key sections**:
- "Why These Three Models?" (line 50)
- "The Identifiability Problem" (line 140)
- "Expected Scenarios" (line 180)

---

### proposed_models.md (1,132 lines)

**Sections**:

**Introduction** (lines 1-85):
- Executive summary
- Model overview table
- Critical assumptions
- Falsification strategy

**Model 1: NB-AR1** (lines 87-380):
- Theoretical justification
- Mathematical specification
- Prior specifications (detailed)
- Temporal correlation details
- Prior predictive distribution
- Falsification criteria
- Stan implementation notes
- Computational cost

**Model 2: NB-GP** (lines 382-680):
- Theoretical justification
- Mathematical specification (GP kernel)
- Prior specifications (lengthscale focus)
- Connection to ACF
- Gaussian correlation structure
- Falsification criteria
- Stan implementation notes (Cholesky)
- Computational cost

**Model 3: NB-RW** (lines 682-920):
- Theoretical justification
- State-space formulation
- Prior specifications (innovation variance)
- Non-stationary correlation
- Comparison to AR(1) and GP
- Falsification criteria
- Stan implementation notes
- Computational cost

**Model Comparison Framework** (lines 922-1040):
- Quantitative criteria
- Qualitative assessment
- Comparing correlation structures

**Critical Decision Points** (lines 1042-1080):
- Three key decision checkpoints
- Evidence to evaluate at each
- Decision rules

**Stress Tests** (lines 1082-1132):
- Prior sensitivity
- Structural break sensitivity
- Forecast performance
- Residual diagnostics

**Best for**: Complete theoretical understanding, mathematical details

**Key formulas**:
- AR(1) specification (line 120)
- GP kernel (line 450)
- RW state equation (line 710)

---

### stan_implementation_guide.md (762 lines)

**Sections**:

**Overview** (lines 1-20)

**Model 1: NB-AR1** (lines 22-190):
- Complete Stan code (60 lines)
- Key implementation details
- Boundary issues and solutions
- Diagnostics to monitor
- Expected parameter ranges

**Model 2: NB-GP** (lines 192-360):
- Complete Stan code (70 lines)
- Covariance function implementation
- Cholesky decomposition
- Numerical stability solutions
- Expected parameter ranges

**Model 3: NB-RW** (lines 362-480):
- Complete Stan code (55 lines)
- Alternative parameterization
- Non-stationarity handling
- Expected parameter ranges

**Python Interface** (lines 482-580):
- Data preparation
- cmdstanpy code for each model
- Extract results
- LOO comparison

**Posterior Predictive Checks** (lines 582-680):
- ACF check code
- Time series plot code
- Residual ACF check code

**Troubleshooting Guide** (lines 682-762):
- Divergent transitions
- Low ESS
- Cholesky failures
- Posterior = Prior
- Unrealistic predictions

**Best for**: Copy-paste code, debugging specific issues

**Key code blocks**:
- AR(1) Stan model (line 42)
- GP Stan model (line 212)
- RW Stan model (line 372)
- Python fitting (lines 500-550)

---

### model_selection_framework.md (546 lines)

**Sections**:

**The Central Question** (lines 1-10):
- Core scientific question

**Decision Tree** (lines 12-60):
- Full decision flowchart
- Branching logic
- Terminal recommendations

**Baseline Model** (lines 62-95):
- Stan code for independent errors
- Residual ACF diagnostic

**Quantitative Criteria** (lines 97-210):
1. PSIS-LOO-CV
2. Pareto-k diagnostics
3. Posterior predictive ACF
4. Effective parameters

**Qualitative Criteria** (lines 212-280):
5. Smoothness of predictions
6. Parameter interpretability
7. Prior-posterior conflict

**Special Tests** (lines 282-370):
- Test A: Is AR(1) appropriate? (PACF)
- Test B: Is GP appropriate? (vs quadratic)
- Test C: Is RW appropriate? (unit root test)

**Model Averaging** (lines 372-420):
- Stacking weights
- Pseudo-BMA weights
- Implementation code

**Reporting Framework** (lines 422-490):
- Scenario 1: Clear winner
- Scenario 2: Indistinguishable
- Scenario 3: Unnecessary
- Scenario 4: All inadequate

**Red Flags** (lines 492-546):
- Flag 1: Computational pathologies
- Flag 2: Unrealistic posteriors
- Flag 3: Prior-posterior overlap
- Flag 4: Poor predictive performance

**Best for**: Making final decisions, writing reports

**Key algorithms**:
- Decision tree (line 15)
- Model selection function (line 520)

---

## Key Concepts Reference

### The Identifiability Problem
**Discussed in**: README.md (line 140), proposed_models.md (line 30)

**Core issue**: With ACF = 0.971, very hard to separate smooth trend from high correlation

**Solution**: Compare to baseline, check if ΔELPD > 5

---

### The n=40 Challenge
**Discussed in**: README.md (line 155), proposed_models.md (line 55)

**Core issue**: Small sample limits complexity we can estimate

**Solution**: Prefer simpler models (AR1, RW) over complex (GP)

---

### Prior Specifications
**Detailed in**: proposed_models.md (lines 180-220, 480-520, 740-770)

**Key priors**:
- ρ ~ Beta(20, 2): E = 0.91 (informed by ACF)
- ℓ ~ InvGamma(5, 5): E = 1.25 (long correlation)
- σ_ω ~ Exponential(10): E = 0.1 (small innovations)

**Rationale**: Weakly informative, allow data to override

---

### Falsification Criteria
**Detailed in**: proposed_models.md (lines 300-350, 600-650, 850-900)

**Philosophy**: Each model has explicit rejection rules

**Example**:
- Reject AR(1) if ρ > 0.99 (boundary issue)
- Reject GP if ℓ > 10 (constant function)
- Reject RW if σ_ω < 0.05 (deterministic)

---

### Model Comparison
**Framework in**: model_selection_framework.md (lines 97-210)

**Primary metric**: PSIS-LOO ELPD
**Secondary**: Pareto-k, posterior predictive ACF
**Tertiary**: Parameter interpretability

**Decision rule**: ΔELPD > 5 → clear winner

---

## Common Questions

### Q1: Which model should I implement first?
**A**: Start with AR(1) - most standard, most likely to work

**See**: README.md "Implementation Priorities" (line 320)

---

### Q2: How do I know if temporal correlation is real?
**A**: Compare to baseline (independent errors). If ΔELPD < 2, it's trend artifact

**See**: model_selection_framework.md "Baseline Model" (line 62)

---

### Q3: What if all models have convergence issues?
**A**: Try reparameterizations, then consider models too complex for n=40

**See**: stan_implementation_guide.md "Troubleshooting" (line 682)

---

### Q4: What if models are indistinguishable?
**A**: Use model averaging with stacking weights. This is a valid conclusion!

**See**: model_selection_framework.md "Model Averaging" (line 372)

---

### Q5: How do I interpret a posterior ρ = 0.97?
**A**: Very high correlation, near unit root. Consider fitting RW model as well.

**See**: SUMMARY.md "Parameter Interpretation" (line 165)

---

## Mathematical Notation

### Common Symbols

| Symbol | Meaning | Context |
|--------|---------|---------|
| C_t | Count at time t | Response variable |
| μ_t | Expected count at time t | Mean parameter |
| η_t | Log-mean at time t | Linear predictor |
| α | NB dispersion | Overdispersion parameter |
| ρ | Correlation | AR(1) coefficient |
| ℓ | Lengthscale | GP correlation range |
| σ_ω | Innovation SD | Random walk step size |
| β₀, β₁ | Intercept, slope | Trend parameters |

### Model-Specific

**AR(1)**:
- ε_t: Deviation from trend (AR process)
- σ_η: Innovation standard deviation
- Corr(ε_t, ε_{t+k}) = ρ^k

**GP**:
- f(t): Latent GP function
- K(t,t'): Covariance matrix
- η: GP amplitude (marginal SD)

**RW**:
- ξ_t: Cumulative random walk
- ω_t: Innovation at time t
- Var(ξ_t) = t·σ_ω²

---

## Stan Code Reference

### File Paths (Once Implemented)

```
/workspace/models/
├── nb_ar1.stan          # From stan_implementation_guide.md, line 42
├── nb_gp.stan           # From stan_implementation_guide.md, line 212
├── nb_rw.stan           # From stan_implementation_guide.md, line 372
└── nb_baseline.stan     # From model_selection_framework.md, line 75
```

### Python Scripts

```
/workspace/scripts/
├── fit_ar1.py           # From stan_implementation_guide.md, line 500
├── fit_gp.py            # From stan_implementation_guide.md, line 520
├── fit_rw.py            # From stan_implementation_guide.md, line 540
└── compare_models.py    # From stan_implementation_guide.md, line 560
```

---

## Visualization Reference

### Figures to Generate

1. **Time series overlay**: Observed vs posterior predictive
   - Code: stan_implementation_guide.md, line 620

2. **ACF comparison**: Observed vs model-based ACF
   - Code: stan_implementation_guide.md, line 600

3. **Residual diagnostics**: ACF of residuals
   - Code: stan_implementation_guide.md, line 650

4. **Parameter posteriors**: Density plots of ρ, ℓ, σ_ω
   - Use: `az.plot_posterior(idata, var_names=[...])`

5. **Model comparison**: ELPD with error bars
   - Use: `az.plot_compare(comparison)`

---

## Success Checklist

From SUMMARY.md (lines 300-330):

**Convergence**:
- [ ] Rhat < 1.01
- [ ] ESS > 400
- [ ] < 0.1% divergences
- [ ] Good mixing

**Fit Quality**:
- [ ] ACF captured
- [ ] Residual ACF < 0.2
- [ ] 90% observations in PP interval
- [ ] Pareto-k < 0.7

**Model Comparison**:
- [ ] PSIS-LOO computed
- [ ] SE reasonable
- [ ] Clear ranking or averaging justified
- [ ] Decision documented

---

## Timeline Estimate

### Phase 1: Implementation (Day 1)
- Write Stan code: 2 hours
- Fit AR(1) model: 30 min
- Debug if needed: 1-2 hours
- **Total**: 4-5 hours

### Phase 2: Comparison (Day 2)
- Fit GP and RW: 1 hour
- Fit baseline: 30 min
- Compute LOO: 30 min
- **Total**: 2 hours

### Phase 3: Analysis (Day 3)
- Posterior predictive checks: 2 hours
- Model comparison: 1 hour
- Sensitivity analysis: 1 hour
- **Total**: 4 hours

### Phase 4: Reporting (Day 4)
- Generate figures: 1 hour
- Write results: 2 hours
- **Total**: 3 hours

**Grand Total**: 13-14 hours of work

---

## Critical Warnings

### 1. Do Not Proceed Without Convergence
If Rhat > 1.05, **STOP** and debug. Results are meaningless.

### 2. n=40 is Small
Do not expect to distinguish subtle model differences. Be ready to conclude "indistinguishable."

### 3. High ACF May Be Artifact
Do not assume temporal correlation is real. Always compare to baseline.

### 4. Parameter Boundaries Are Red Flags
If ρ → 1, ℓ → ∞, or σ_ω → 0, model is telling you something. Listen.

### 5. Computational Issues = Model Issues
Divergences, low ESS, Cholesky failures often indicate misspecification, not just numerical problems.

---

## Contact Information

**Designer**: Model Designer 2
**Specialization**: Temporal correlation structures
**Approach**: Bayesian state-space and time series models
**Implementation**: Stan/CmdStanPy

**Documents created**: 5
**Total lines**: 3,261
**Total documentation**: 108 KB

**Status**: Complete and ready for implementation

**Date**: 2025-10-29

---

## Final Notes

This documentation represents a **complete Bayesian modeling strategy** for temporal correlation in count time series data. The three models (AR1, GP, RW) span the range of plausible correlation structures from stationary to non-stationary.

**Key philosophy**: Truth-seeking over task-completion. If models fail or are indistinguishable, say so. That's science.

**Remember**: With ACF = 0.971 and n=40, identifiability is challenging. Be prepared for uncertainty in final conclusions.

Good luck!

---

**END OF INDEX**
