# Bayesian Model Design: Designer 2
## Flexible & Adaptive Approaches

**Designer**: Bayesian Model Designer 2
**Focus**: Piecewise, spline, and hierarchical models for regime shift detection
**Date**: 2025-10-27

---

## Overview

This directory contains a complete Bayesian modeling strategy for the Y~x relationship, emphasizing **flexible and adaptive** approaches that can handle both sharp and smooth transitions.

**Core Philosophy**: Design models to be falsifiable, not just fit-able. Success is discovering truth, not completing tasks.

---

## Proposed Models

### Model 1: Bayesian Change-Point Regression
- **Type**: Piecewise linear with inferred breakpoint
- **Strengths**: Explicit regime shift, interpretable parameters
- **Risk**: May impose artificial breakpoint if transition is smooth
- **Key Parameter**: τ (breakpoint location)
- **Falsification**: Reject if SD(τ) > 5

### Model 2: Hierarchical B-Spline Regression
- **Type**: Smooth nonlinear with adaptive basis functions
- **Strengths**: Data-driven flexibility, fast convergence
- **Risk**: May oversmooth real regime shift
- **Key Parameter**: τ (smoothness control)
- **Falsification**: Reject if wild oscillations or poor extrapolation

### Model 3: Mixture-of-Experts with Gating Network
- **Type**: Soft mixture of linear and constant experts
- **Strengths**: Learns transition sharpness from data
- **Risk**: Overparameterization, identifiability issues
- **Key Parameter**: τ_eff (effective breakpoint), γ₁ (sharpness)
- **Falsification**: Reject if SD(τ_eff) > 10 or gating uninformative

---

## Directory Structure

```
designer_2/
├── README.md                    # This file
├── proposed_models.md           # Detailed model specifications
├── validation_plan.md           # Validation & falsification strategy
├── implementation_code.py       # Python implementation (Stan)
└── models/
    ├── model1_changepoint.stan  # Change-point model
    ├── model2_spline.stan       # B-spline model
    └── model3_mixture.stan      # Mixture model
```

---

## Key Files

### 1. Proposed Models (`proposed_models.md`)
**Contents**:
- Detailed specifications for all 3 models (likelihood, priors, equations)
- Theoretical justifications and scientific interpretations
- Explicit falsification criteria for each model
- Expected performance metrics
- Implementation notes for Stan
- Red flags and decision points
- Alternative escape routes if models fail

**Length**: ~8000 words, comprehensive

### 2. Stan Model Files (`models/*.stan`)
**Contents**:
- Production-ready Stan code for each model
- Informative priors based on EDA findings
- Posterior predictive samples for validation
- Log-likelihood for LOO-CV

**Models**:
- `model1_changepoint.stan`: Piecewise linear with continuous transition
- `model2_spline.stan`: B-spline with hierarchical shrinkage
- `model3_mixture.stan`: Gating network with two experts

### 3. Implementation Code (`implementation_code.py`)
**Contents**:
- Complete Python workflow using cmdstanpy
- Data loading and B-spline basis generation
- Model fitting with proper Stan settings
- Convergence diagnostics (R-hat, ESS, divergences)
- Posterior predictive checks (coverage, R², RMSE)
- LOO-CV with Pareto-k diagnostics
- Comprehensive visualization (fits, residuals, comparisons)

**Usage**:
```bash
# Fit all models
python implementation_code.py --data_path /path/to/data.csv --model all

# Fit specific model
python implementation_code.py --data_path /path/to/data.csv --model changepoint
```

### 4. Validation Plan (`validation_plan.md`)
**Contents**:
- 4-stage validation pipeline (prior, MCMC, posterior, LOO)
- Model-specific falsification criteria with diagnostic plots
- Stress tests (synthetic recovery, LOO sensitivity, prior sensitivity)
- Decision framework for model selection
- Red flags triggering major pivots
- Timeline and success metrics

**Length**: ~6000 words, systematic

---

## Quick Start

### Step 1: Read EDA Report
```bash
# Review comprehensive EDA findings
cat /workspace/eda/eda_report.md
```

**Key Findings**:
- N=27 observations, x ∈ [1, 31.5], Y ∈ [1.71, 2.63]
- Strong nonlinear saturation pattern (R²_nonlinear = 0.89 vs R²_linear = 0.52)
- Evidence for regime shift around x=9-10
- Correlation drops from 0.94 (x<10) to -0.03 (x≥10)

### Step 2: Review Proposed Models
```bash
# Read detailed model specifications
cat /workspace/experiments/designer_2/proposed_models.md
```

**Focus On**:
- Section "Model Specification" for each model (likelihood, priors)
- Section "Falsification Criteria" for rejection rules
- Section "Red Flags & Decision Points" for when to pivot

### Step 3: Fit Models
```bash
# Ensure data is available
ls /workspace/data/

# Run implementation
cd /workspace/experiments/designer_2
python implementation_code.py --data_path /workspace/data/data.csv --model all
```

**Expected Runtime**: 2-5 minutes for all three models (depending on hardware)

### Step 4: Validate & Compare
```bash
# Check convergence diagnostics in output
# Review posterior predictive checks
# Compare LOO-ELPD across models
# Apply falsification criteria from validation_plan.md
```

**Decision Rule**:
- If ΔLOO-ELPD > 2×SE: Clear winner
- If ΔLOO-ELPD < 2×SE: Models tied → prefer simpler/interpretable
- Check falsification criteria before finalizing

---

## Expected Outcomes

### Performance Targets

| Model | R² | LOO-ELPD | Convergence | Runtime |
|-------|-----|----------|-------------|---------|
| Change-Point | 0.88-0.92 | High | Moderate | 30-60s |
| B-Spline | 0.85-0.90 | Moderate-High | Excellent | <30s |
| Mixture | 0.87-0.91 | High | Moderate | 60-120s |

### Personal Predictions

**Most Likely Winner**: Change-Point (Model 1)
- EDA shows strong evidence for regime shift
- Piecewise OLS achieved best fit (R²=0.904)
- Should excel in interpretability

**Dark Horse**: Mixture (Model 3)
- If transition is gradual (not sharp), mixture will capture it
- Gating network provides best of both worlds
- May outperform if SD(τ) in Model 1 is large

**Safe Bet**: B-Spline (Model 2)
- Fastest convergence (linear in parameters)
- Will definitely fit reasonably well
- Lacks interpretability but robust

---

## Falsification Mindset

### Critical Questions

**Before fitting**:
- What would make me reject each model?
- What evidence would surprise me?
- What alternative models should I pivot to?

**After fitting**:
- Does the model capture the saturation pattern?
- Are posteriors well-constrained or vague?
- Do residuals show systematic patterns?
- Is the model sensitive to prior choice or single observations?

**During comparison**:
- Are performance differences meaningful (>2×SE)?
- Does the "best" model pass falsification checks?
- Would I trust this model for scientific inference?

### Rejection Criteria Summary

| Model | Primary Rejection Criterion | Alternative Action |
|-------|----------------------------|-------------------|
| Change-Point | SD(τ) > 5 | Switch to B-Spline |
| B-Spline | Wild oscillations or R² < 0.80 | Switch to parametric (exponential) |
| Mixture | SD(τ_eff) > 10 or uninformative gating | Simplify to single expert |
| ALL FAIL | LOO-R² < 0.75 for all | Pivot to GP or transformations |

---

## Integration with Other Designers

This is **Designer 2** in a parallel design framework. Other designers may propose:

- **Designer 1**: Mechanistic models (exponential, Michaelis-Menten, logistic)
- **Designer 3**: Robust models (Student-t, heteroscedastic, transformations)

**Synthesis Strategy**:
- Compare LOO-ELPD across all designers' models
- Select best-performing models from each class
- Report ensemble if models are statistically tied
- Prefer interpretable models when performance is equivalent

---

## Design Principles Applied

### 1. Falsification First
- Every model has explicit rejection criteria
- Falsification checks built into validation pipeline
- Model failure is informative, not a setback

### 2. Adaptive Flexibility
- Models span sharp (piecewise) to smooth (spline) transitions
- Mixture model learns transition type from data
- Ready to pivot if initial models fail

### 3. Computational Pragmatism
- All models implement able in Stan (production-ready)
- B-spline uses pre-computed basis (fast)
- Change-point uses smooth approximation to avoid discrete sampling

### 4. Scientific Interpretability
- Change-point: τ has clear meaning (saturation threshold)
- Mixture: γ₁ quantifies transition sharpness
- All parameters have domain interpretations

### 5. Honest Uncertainty
- Report 95% credible intervals for all parameters
- Quantify breakpoint uncertainty (SD(τ))
- Use LOO-CV for out-of-sample predictions
- Flag extrapolation limitations

---

## Limitations & Caveats

### Known Limitations

1. **Small Sample Size** (N=27)
   - Limits model complexity (especially Model 3)
   - Wide credible intervals expected
   - Extrapolation uncertainty high

2. **Limited High-x Data** (only 3 points with x>20)
   - Plateau level poorly constrained
   - Extrapolation beyond x=31.5 unreliable
   - Informative priors critical

3. **Single Predictor** (x only)
   - May miss important covariates
   - Residual variance may include unmeasured factors
   - Saturation might be multi-causal

4. **Homoscedastic Assumption**
   - Models assume constant variance
   - If violated, may need heteroscedastic extensions
   - Check posterior predictive checks for evidence

### What Could Go Wrong

**Scenario 1: All models fail to converge**
- Action: Simplify to quadratic polynomial or power law
- Indicates nonlinear models are too complex for data

**Scenario 2: Models fit well but posteriors are vague**
- Action: Report high uncertainty, suggest more data
- Data insufficient to distinguish model classes

**Scenario 3: Best model fails falsification checks**
- Action: Pivot to GP or transform-and-fit approach
- Parametric assumptions may be too restrictive

**Scenario 4: Results highly sensitive to priors**
- Action: Report ensemble or use model averaging
- Data weak relative to prior information

---

## Success Definition

**Minimum Success**: At least one model converges and achieves R² > 0.85

**Typical Success**: Models 1 & 2 converge, clear winner emerges from LOO comparison

**Excellent Success**: All models converge, winner passes all falsification checks with high confidence

**Learning Success**: Discover why models fail and identify correct path forward (e.g., GP, transforms)

**The goal is finding truth, not completing tasks.**

---

## Contact & Questions

**Designer**: Bayesian Model Designer 2
**Expertise**: Flexible models, regime detection, computational Bayesian methods
**Philosophy**: Truth over task completion, falsification over confirmation

For questions about:
- Model specifications → See `proposed_models.md`
- Validation strategy → See `validation_plan.md`
- Implementation → See `implementation_code.py` comments
- Stan code → See `models/*.stan` files

---

## Version History

- **v1.0** (2025-10-27): Initial design with 3 model classes
  - Change-point regression
  - Hierarchical B-spline
  - Mixture-of-experts
  - Complete Stan implementations
  - Comprehensive validation plan

---

**Files in This Directory**:

```
/workspace/experiments/designer_2/proposed_models.md
/workspace/experiments/designer_2/validation_plan.md
/workspace/experiments/designer_2/implementation_code.py
/workspace/experiments/designer_2/models/model1_changepoint.stan
/workspace/experiments/designer_2/models/model2_spline.stan
/workspace/experiments/designer_2/models/model3_mixture.stan
/workspace/experiments/designer_2/README.md
```

All paths are absolute. All models are production-ready.

---

**END OF DESIGNER 2 DOCUMENTATION**
