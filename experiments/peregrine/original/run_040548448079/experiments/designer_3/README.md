# Designer 3: Time-Series and Dynamic Models

**Focus**: Bayesian models where temporal dynamics and autocorrelation are PRIMARY features, not afterthoughts.

**Date**: 2025-10-29
**Data**: 40 time-ordered counts with extreme overdispersion (variance/mean=67.99), strong autocorrelation (ACF(1)=0.944), and structural break at observation 17

---

## What's in This Directory

### Core Documents

1. **`proposed_models.md`** (MAIN DOCUMENT - 11,000 words)
   - Three competing Bayesian model classes
   - Complete mathematical specifications
   - Prior recommendations from EDA
   - Falsification criteria for each model
   - Implementation details and computational considerations
   - Model comparison strategy
   - Red flags and pivot criteria

2. **`implementation_priority.md`** (QUICK START GUIDE)
   - Priority order for fitting models
   - Falsification test checklist
   - Red flag triggers
   - Decision tree
   - Computational tips
   - Timeline estimate (3 weeks)

3. **`model_comparison_summary.md`** (VISUAL GUIDE)
   - Side-by-side model comparison
   - Conceptual diagrams
   - Strengths/weaknesses
   - Expected parameter values
   - Posterior predictive check guidance

4. **`README.md`** (THIS FILE)
   - Directory overview
   - Quick reference

---

## Three Proposed Models

### Model 1: Dynamic Linear Model (State-Space) - PRIMARY RECOMMENDATION

**Core Idea**: Log-rate is a latent dynamic process with time-varying velocity. Regime change is a discrete shift in drift.

**Equation**:
```
C_t ~ NegativeBinomial(exp(η_t), α)
η_t = η_{t-1} + δ_t + ν_t
δ_t = φ × δ_{t-1} + I(t>τ) × Δδ + ω_t
```

**Key Parameters**: φ (velocity persistence), τ (changepoint), Δδ (regime shift magnitude)

**Why Primary**: Best theoretical alignment with EDA findings (strong autocorrelation + discrete regime change)

**Runtime**: 15-30 minutes

---

### Model 2: Negative Binomial AR (Observation-Driven) - BASELINE

**Core Idea**: Today's count depends directly on yesterday's count, plus a time-varying mean with regime shift.

**Equation**:
```
C_t ~ NegativeBinomial(μ_t, α_t)
log(μ_t) = β_0 + β_1×year + β_2×year² + γ×log(C_{t-1}+1) + I(t>τ)×β_3×(year-year_τ)
```

**Key Parameters**: γ (AR coefficient), β_3 (post-break slope change)

**Why Baseline**: Simplest, fastest, easy to interpret. Good sanity check.

**Runtime**: 10-20 minutes

---

### Model 3: Gaussian Process (Parameter-Driven) - VALIDATOR

**Core Idea**: Log-rate is a smooth unknown function of time. Let GP discover the pattern (including regime change).

**Equation**:
```
C_t ~ NegativeBinomial(exp(β_0 + f(year_t)), α)
f ~ GP(0, K)
K(s,t) = σ_f² × exp(-ρ × (s-t)²) + σ_n² × δ_{st}
```

**Key Parameters**: length_scale (smoothness), σ_f (signal strength)

**Why Validator**: Tests whether discrete changepoint is necessary or if smooth transition fits better. Maximum flexibility.

**Runtime**: 20-40 minutes (slowest)

---

## Quick Start (When Ready to Implement)

### Step 1: Read the Documents (1 hour)
```bash
cd /workspace/experiments/designer_3
cat proposed_models.md        # Full specifications
cat implementation_priority.md  # How to proceed
cat model_comparison_summary.md # Visual comparisons
```

### Step 2: Create Directory Structure
```bash
mkdir -p models scripts results figures
```

### Step 3: Implement Model 1 (Priority)
```bash
# Create Stan file
vim models/model_1_dlm.stan

# Create fitting script
vim scripts/fit_model_1.py

# Run
python scripts/fit_model_1.py
```

### Step 4: Diagnostics
```bash
# Check convergence
# - R-hat < 1.01 for all parameters
# - ESS > 400
# - Divergences < 1%

# Run falsification tests
python scripts/falsification_tests.py --model 1
```

### Step 5: Compare Models
```bash
# After fitting all three
python scripts/compare_models.py
```

---

## Key Design Principles

1. **Falsification Mindset**: Each model has explicit criteria for rejection. If 2+ criteria fail, abandon that model.

2. **Autocorrelation is Primary**: Not an afterthought or nuisance parameter. All models treat temporal dependency as central.

3. **Multiple Hypotheses**: Three fundamentally different mechanisms for data generation. Let data adjudicate.

4. **Ready to Pivot**: If all three fail, we propose alternative model classes (ARMA, HMM, etc.). Failure is learning.

5. **Scientific Truth > Task Completion**: Goal is finding the right model, not completing the plan.

---

## Falsification Criteria (Critical)

Each model can be **rejected** if:

### Model 1 (DLM)
- Velocity AR is unnecessary (φ ≈ 0)
- Observation noise dominates (σ_η >> σ_δ)
- Changepoint is diffuse (SD(τ) > 5)
- Residual ACF(1) > 0.5

### Model 2 (NB-AR)
- AR term is unnecessary (γ ≈ 0)
- Residual ACF(1) > 0.5
- Identifiability problem (|corr(γ, β_2)| > 0.7)

### Model 3 (GP)
- Length-scale collapses (l < 0.3) or explodes (l > 5)
- Nugget dominates (σ_n/σ_f > 0.5)
- GP trajectory is smooth (no visible regime change)
- Residual ACF(1) > 0.5

**If 2+ criteria fail for a model, DO NOT proceed with that model.**

---

## Expected Outcomes (Predictions)

Based on EDA evidence:

1. **Model 1 will likely pass all tests**: φ ≈ 0.8, Δδ ≈ 0.75, τ concentrates at 17
2. **Model 3 may over-smooth the break**: GP trajectory will be gradual, not sharp
3. **Model 2 may fail ACF test**: AR(1) term insufficient to explain ACF(1)=0.944

**LOO Ranking Prediction**: Model 1 > Model 3 > Model 2 (but Model 1 vs 3 may be close)

**If predictions are wrong**: That's valuable information! It means our understanding of the data generation process needs revision.

---

## Red Flags (When to Stop and Reconsider)

### Stop ALL work if:

1. **All models fail autocorrelation test** (residual ACF(1) > 0.5 for all)
   - Autocorrelation is more complex than AR(1)/GP/DLM
   - Pivot to ARMA(p,q) or Hidden Markov Models

2. **All models fail structural break test** (< 60% of posterior predictive datasets show break)
   - Regime change may be smooth (trust GP) or illusory (outliers)
   - Pivot to pure GP or investigate data quality

3. **All models underestimate variance** (pp variance/mean < 40 vs observed 67.99)
   - Negative Binomial is inadequate
   - Pivot to generalized distributions or mixture models

4. **Computational failure across all models** (> 10% divergences despite tuning)
   - Problem geometry is pathological
   - Pivot to variational inference or simplify to GLM

---

## Timeline

**Week 1**: Implementation & Initial Fitting
- Days 1-2: Model 1 (DLM)
- Days 3-4: Model 3 (GP)
- Day 5: Model 2 (NB-AR)

**Week 2**: Falsification & Comparison
- Days 1-2: Run all falsification tests
- Day 3: LOO comparison
- Days 4-5: Sensitivity analysis

**Week 3**: Validation & Reporting
- Day 1: Out-of-sample prediction
- Day 2: Posterior inference
- Days 3-5: Write-up and documentation

**Total**: 15 working days

---

## Contact & Help

**If stuck on**:
- Stan syntax: See `proposed_models.md` for complete code templates
- Divergences: See `implementation_priority.md` computational tips section
- All models fail: See `proposed_models.md` section "Red Flags Requiring Model Class Pivot"
- Interpretation: See `model_comparison_summary.md` for parameter interpretation

**Remember**: The goal is finding truth, not completing tasks. A model that fails falsification tests early has taught us something valuable.

---

## File Locations

All files in: `/workspace/experiments/designer_3/`

- `proposed_models.md` - Main document (11K words)
- `implementation_priority.md` - Quick start guide
- `model_comparison_summary.md` - Visual comparison
- `README.md` - This file

**Future directories** (create when implementing):
- `models/` - Stan code files
- `scripts/` - Python fitting scripts
- `results/` - Posterior samples, diagnostics
- `figures/` - Plots and visualizations

---

## Key References (In Documents)

- **Stan code templates**: See `proposed_models.md` sections for each model
- **Prior justifications**: See `proposed_models.md` "Prior Recommendations" for each model
- **Falsification tests**: See `implementation_priority.md` "Falsification Test Checklist"
- **Decision tree**: See `implementation_priority.md` or `model_comparison_summary.md`
- **Expected posteriors**: See `model_comparison_summary.md` "Expected Parameter Values"

---

## Philosophy

This modeling strategy embodies a **Bayesian falsification approach**:

1. Propose multiple competing hypotheses (3 model classes)
2. Define explicit criteria for rejecting each
3. Let data adjudicate among survivors
4. Be ready to abandon all if evidence demands it
5. Pivot quickly when models fail

**Success is not "getting a model to work"**. Success is discovering which model (if any) genuinely explains the data generation process.

**Model failure is learning**. If all three models fail, we've learned that the data generation process is more complex than AR(1)/GP/DLM dynamics, and we propose next steps (ARMA, HMM, etc.).

---

**Next Action**: Read `proposed_models.md` fully, then proceed with implementation following `implementation_priority.md`.

**Critical Reminder**: Each model has falsification criteria. Use them. If a model fails, move on. Don't try to "save" a failed model.

---

**End of README**
