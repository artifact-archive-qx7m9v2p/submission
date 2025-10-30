# DESIGNER 3: Complete File Index

**Designer**: Model Designer 3 - Non-Linear and Hierarchical Bayesian Models
**Date**: 2025-10-29
**Status**: ✓ Complete - 2,519 lines of documentation and code

---

## Start Here

### For Synthesis Agent (Coordinators)
1. **FIRST**: Read `SUMMARY.md` (389 lines) - Quick 5-minute overview
2. **THEN**: Read `experiment_plan.md` (425 lines) - Executive summary with decision logic
3. **FINALLY**: Consult `proposed_models.md` (1,044 lines) - Full mathematical specifications

### For Implementation Team (Modelers)
1. **FIRST**: Read `README.md` (322 lines) - Integration guide
2. **THEN**: Use Stan files directly (339 lines total):
   - `model_1_quadratic.stan` (87 lines)
   - `model_2_changepoint.stan` (120 lines)
   - `model_3_gp.stan` (132 lines)
3. **REFERENCE**: `proposed_models.md` for mathematical details

### For Quick Reference
- **THIS FILE** (`INDEX.md`) - Navigation and file manifest
- `SUMMARY.md` - One-page overview with tables and decision trees

---

## File Descriptions

### Documentation (4 files, 2,180 lines)

#### 1. `SUMMARY.md` (389 lines)
**Purpose**: Quick reference with tables and decision trees
**Audience**: Anyone needing fast overview
**Key Content**:
- 3 models at-a-glance table
- Sequential testing logic flowchart
- Red flags by model
- Expected timeline

**Read this if**: You have 5 minutes and need the essentials

---

#### 2. `experiment_plan.md` (425 lines)
**Purpose**: Executive summary for synthesis agent
**Audience**: Project coordinators, synthesis agent
**Key Content**:
- Model specifications (compact form)
- Sequential testing strategy
- Falsification criteria
- Communication templates

**Read this if**: You need to understand the overall strategy

---

#### 3. `proposed_models.md` (1,044 lines)
**Purpose**: Comprehensive mathematical specifications
**Audience**: Statisticians, implementation team
**Key Content**:
- Full mathematical derivations
- Prior justifications with predictive checks
- Detailed falsification criteria
- Stan implementation notes
- Computational cost analysis
- Stress test specifications
- Escape routes for model failures

**Read this if**: You're implementing the models or need deep understanding

**Sections**:
1. Model 1: Quadratic NB + AR(1) (detailed spec)
2. Model 2: Changepoint NB + AR(1) (detailed spec)
3. Model 3: Gaussian Process NB (detailed spec)
4. Model comparison strategy
5. Prior predictive checks
6. Implementation roadmap
7. Escape routes
8. Appendices with math details

---

#### 4. `README.md` (322 lines)
**Purpose**: Integration and coordination guide
**Audience**: Cross-designer synthesis, project management
**Key Content**:
- Models overview
- Decision logic
- Integration with Designers 1-2
- Computational specifications
- Quality assurance checklist
- Common issues and solutions

**Read this if**: You need to integrate with other designers' work

---

### Stan Programs (3 files, 339 lines)

#### 5. `model_1_quadratic.stan` (87 lines)
**Model**: Quadratic Negative Binomial with AR(1)
**Parameters**: β₀, β₁, β₂, φ, ρ, σ (+ 40 AR errors)
**Key Feature**: Polynomial growth term (β₂·year²)
**Expected Runtime**: 2-4 minutes
**Complexity**: Medium

**Sections**:
- Data block (3 parameters)
- Transformed data (pre-compute year²)
- Parameters (6 core + N innovations)
- Transformed parameters (AR(1) structure + mean function)
- Model (priors + likelihood)
- Generated quantities (log_lik, predictions, derived quantities)

**Special features**:
- Non-centered AR(1) parameterization
- Stationary initialization
- Derived growth rates at start/center/end

---

#### 6. `model_2_changepoint.stan` (120 lines)
**Model**: Bayesian Changepoint with AR(1)
**Parameters**: β₀, β₁, β₂, τ, φ, ρ, σ (+ 40 AR errors)
**Key Feature**: Unknown changepoint location τ
**Expected Runtime**: 4-8 minutes
**Complexity**: High (discrete changepoint)

**Sections**:
- Data block (same as Model 1)
- Parameters (adds τ with bounds)
- Transformed parameters (piecewise linear mean function)
- Model (uniform prior on τ)
- Generated quantities (regime slopes, changepoint index)

**Special features**:
- Continuous junction at τ
- Derived pre/post slopes
- Rate change calculation
- Changepoint index determination

---

#### 7. `model_3_gp.stan` (132 lines)
**Model**: Gaussian Process Negative Binomial
**Parameters**: β₀, β₁, α, ℓ, φ (+ N latent GP values)
**Key Feature**: Non-parametric mean via GP
**Expected Runtime**: 10-20 minutes
**Complexity**: Very high (dense covariance matrix)

**Sections**:
- Data block (same)
- Transformed data (pre-compute distance matrix)
- Parameters (GP hyperparameters + innovations)
- Transformed parameters (Cholesky decomposition + GP realization)
- Model (InvGamma prior on length scale)
- Generated quantities (inflection point counting, decomposition)

**Special features**:
- Squared exponential kernel
- Numerical jitter for stability
- Non-centered GP parameterization
- Inflection point detection

---

### This File

#### 8. `INDEX.md` (this document)
**Purpose**: Navigation and file manifest
**Audience**: Anyone exploring the directory
**Key Content**:
- File descriptions
- Reading order recommendations
- Quick access guide

---

## Total Statistics

| Type | Files | Lines | Size |
|------|-------|-------|------|
| Documentation | 4 | 2,180 | 67 KB |
| Stan Programs | 3 | 339 | 9.8 KB |
| Navigation | 1 | - | 4 KB |
| **TOTAL** | **8** | **2,519** | **~81 KB** |

---

## Models Quick Reference

| ID | Name | File | Parameters | Runtime |
|----|------|------|------------|---------|
| Baseline | Linear NB + AR(1) | (Designer 1) | 5 | 2 min |
| D3M1 | Quadratic NB + AR(1) | model_1_quadratic.stan | 6 | 2-4 min |
| D3M2 | Changepoint NB + AR(1) | model_2_changepoint.stan | 7 | 4-8 min |
| D3M3 | GP NB | model_3_gp.stan | 5 + N latent | 10-20 min |

---

## Decision Tree (Visual)

```
                     START
                       |
              FIT LINEAR BASELINE
                       |
           +-----------+-----------+
           |                       |
    ΔLOO < threshold?         β₂ ≈ 0?
           |                       |
          YES → STOP              YES → STOP
           |                       |
           NO                      NO
           |                       |
           +-------+-------+-------+
                   |
              FIT MODEL 2
                   |
           +-------+-------+
           |               |
      τ uniform?      ΔLOO < 6 SE?
           |               |
          YES → STOP      YES → STOP
           |               |
           NO              NO
           |               |
      Systematic PPC failures?
           |
           +-------+-------+
           |               |
          YES             NO
           |               |
      FIT MODEL 3      SELECT BEST
           |
    Computational OK?
           |
           +-------+-------+
           |               |
          NO              YES
           |               |
    USE BEST MODEL    ΔLOO > 10 SE?
                           |
                    +------+------+
                    |             |
                   YES           NO
                    |             |
              SELECT GP    USE PARAMETRIC
```

---

## Reading Paths by Role

### Path 1: Synthesis Agent (30 minutes)
1. `SUMMARY.md` (5 min) - Get oriented
2. `experiment_plan.md` (15 min) - Understand strategy
3. Skim `proposed_models.md` (10 min) - Check mathematical rigor
4. **Decision**: Approve for implementation

### Path 2: Statistical Reviewer (60 minutes)
1. `README.md` (10 min) - Context
2. `proposed_models.md` (30 min) - Full specifications
3. `model_1_quadratic.stan` (5 min) - Verify implementation
4. `model_2_changepoint.stan` (5 min) - Check changepoint logic
5. `model_3_gp.stan` (5 min) - Review GP structure
6. `experiment_plan.md` (5 min) - Confirm falsification criteria
7. **Decision**: Approve, request changes, or reject

### Path 3: Implementation Team (90 minutes)
1. `SUMMARY.md` (5 min) - Overview
2. `README.md` (10 min) - Integration plan
3. All Stan files (20 min) - Understand code
4. `proposed_models.md` (40 min) - Deep dive on math
5. Write fitting scripts (15 min)
6. **Outcome**: Ready to run models

### Path 4: Domain Expert (20 minutes)
1. `SUMMARY.md` (5 min) - What are we doing?
2. `experiment_plan.md` sections:
   - "Proposed Models" (3 min) - Plain language
   - "Falsification Philosophy" (3 min) - Why trust results?
   - "Communication Plan" (3 min) - What you'll receive
3. `README.md` "Integration" section (3 min) - How models relate
4. `proposed_models.md` "Theoretical Justification" (3 min) - Why these models?
5. **Outcome**: Understand scientific approach

---

## Key Takeaways by File

### SUMMARY.md
> "Linear model will probably win (60% chance). All complex models must prove their value via strict falsification criteria. If they fail, that's success—we avoided overfitting."

### experiment_plan.md
> "Sequential testing with early stopping. Model 1 must beat baseline by ΔLOO > 4 SE. Model 2 must beat baseline by >6 SE. Model 3 must beat best by >10 SE. High bars are intentional—complexity is guilty until proven innocent."

### proposed_models.md
> "Three competing hypotheses: (1) Polynomial acceleration, (2) Discrete regime shift, (3) Non-parametric smooth function. Each has explicit falsification criteria. Each has escape routes. Each has stress tests. Success is finding truth, not defending complexity."

### README.md
> "Designer 3 extends Designer 1's baseline with non-linear structures. Complements Designer 2's temporal focus. Integration via shared baseline and LOO comparison. Most likely outcome: linear model wins, which is good science."

### Stan programs
> "Production-ready implementations with non-centered parameterizations, derived quantities for interpretability, and log_lik for LOO. Model 1 is straightforward, Model 2 has discrete changepoint challenges, Model 3 may be computationally intractable with n=40."

---

## Version History

**v1.0 (2025-10-29)**:
- Initial complete specification
- 3 models proposed with full Stan implementations
- 2,519 lines of documentation and code
- Falsification criteria for all models
- Sequential testing strategy
- Integration plan with other designers

---

## Quick Access by Question

**"Which model should I start with?"**
→ Linear baseline (from Designer 1), then Model 1 if justified

**"What makes you abandon a model?"**
→ See "Falsification Criteria" in each model section

**"How do I know if complexity is justified?"**
→ ΔLOO must exceed (# extra parameters × 2 SE), minimum

**"What if all models fail?"**
→ Report linear baseline as best—that's success!

**"How long will this take?"**
→ 13 hours over 3 days (see SUMMARY.md timeline)

**"What could go wrong?"**
→ See "Red Flags" in SUMMARY.md or "Common Issues" in README.md

**"How does this relate to other designers?"**
→ See "Integration with Other Designers" in README.md

**"What's the math behind Model X?"**
→ See relevant section in proposed_models.md

**"How do I implement Model X?"**
→ Use corresponding .stan file + fitting notes in proposed_models.md

**"Why these specific priors?"**
→ See "Priors" and "Prior Predictive Distribution" in proposed_models.md

---

## Contact and Navigation

**All files in**: `/workspace/experiments/designer_3/`

**For questions about**:
- Overall strategy → `experiment_plan.md`
- Mathematical details → `proposed_models.md`
- Implementation → Stan files + `README.md`
- Quick facts → `SUMMARY.md`
- File navigation → This file (`INDEX.md`)

---

**Designer**: Model Designer 3
**Philosophy**: Complexity must prove itself or be abandoned
**Status**: ✓ Complete and ready for rigorous testing
**Last Updated**: 2025-10-29
