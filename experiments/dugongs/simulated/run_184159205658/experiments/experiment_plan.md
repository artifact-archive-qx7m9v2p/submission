# Bayesian Modeling Experiment Plan
## Synthesis of Three Independent Model Design Proposals

**Date:** 2024-10-27
**Dataset:** N=27 observations, Y vs x relationship
**Goal:** Build adequate Bayesian models for Y-x relationship with proper validation

---

## Overview

Three independent model designers proposed a total of **9 distinct model classes**. After removing duplicates and synthesizing recommendations, I have identified **5 unique model classes** to evaluate, ranked by expected adequacy and theoretical justification.

### Design Philosophy

1. **Falsification-first:** Each model has explicit failure criteria
2. **Parsimony preferred:** Start simple, add complexity only if justified
3. **Bayesian throughout:** All models use proper priors and MCMC inference
4. **Minimum attempt policy:** Evaluate at least first 2 models unless Model 1 fails pre-fit validation

---

## Model Classes to Evaluate (Ranked)

### Model 1: Logarithmic Regression (PRIMARY - Expected Success: 80%)

**Source:** Designer 1 (Primary), Designer 2 (mentioned as alternative)

**Complete Specification:**
```
Likelihood:
  Y_i ~ Normal(μ_i, σ)
  μ_i = β₀ + β₁ · log(x_i)

Priors:
  β₀ ~ Normal(1.73, 0.5)     # Intercept, centered at EDA estimate
  β₁ ~ Normal(0.28, 0.15)    # Slope, weakly positive
  σ ~ Exponential(5)         # Residual SD, mean=0.2
```

**Theoretical Justification:**
- Captures diminishing returns naturally (concave function)
- Elasticity interpretation: 1% change in x → (β₁/100)% change in Y
- Strong EDA support (R²=0.83)
- Only 2 parameters (most parsimonious)
- Linear in parameters (easy MCMC sampling)

**Success Criteria:**
- R-hat < 1.01, ESS > 400 for all parameters
- No divergent transitions
- β₁ > 0 (positive relationship)
- Residuals show no systematic patterns
- Posterior predictive checks: >90% coverage at 95% CI
- LOO-CV ELPD better than linear baseline

**Failure Criteria (ABANDON if):**
- Systematic residual patterns vs x or fitted values
- β₁ posterior includes 0 or negative values
- Posterior predictive check shows <85% coverage
- LOO-CV: >20% of observations have Pareto k > 0.7

**Implementation:** Stan (preferred) or PyMC
**Expected Runtime:** <5 minutes

---

### Model 2: Michaelis-Menten Saturation (ALTERNATIVE 1 - Expected Success: 60%)

**Source:** Designer 2 (Primary)

**Complete Specification:**
```
Likelihood:
  Y_i ~ Normal(μ_i, σ)
  μ_i = Y_max · x_i / (K + x_i)

Priors:
  Y_max ~ Normal(2.6, 0.3)      # Maximum response (asymptote)
  log_K ~ Normal(log(5), 1)     # Log half-saturation (reparameterized)
  σ ~ HalfNormal(0.25)          # Residual SD

Reparameterization:
  K = exp(log_K)  # Transform for better sampling
```

**Theoretical Justification:**
- Explicitly models saturation with bounded predictions
- Clear mechanistic interpretation (common in biology, pharmacology)
- Y_max = maximum achievable response
- K = x value where Y reaches half of maximum
- Graceful extrapolation (Y ≤ Y_max for all x)

**Success Criteria:**
- R-hat < 1.01, ESS > 400, no divergences
- Posterior Y_max > max(Y_observed) = 2.63
- Posterior K in plausible range [0.5, 20]
- Better LOO-CV than logarithmic model (ΔELPD > 2×SE)
- Residuals normal, no patterns

**Failure Criteria (ABANDON if):**
- Posterior Y_max < 2.63 (impossible - data shows higher values)
- Posterior K > 20 (beyond data range, unidentifiable)
- Persistent divergent transitions (>5% after tuning)
- No LOO-CV improvement over Model 1
- Prior-posterior conflict (KL divergence indicates data-prior mismatch)

**Implementation:** Stan (required for reparameterization)
**Expected Runtime:** 5-10 minutes

---

### Model 3: Quadratic Polynomial (ALTERNATIVE 2 - Expected Success: 70%)

**Source:** Designer 1 (mentioned as power law), Designer 3 (polynomial family)
**Note:** Simplified from power law for computational ease

**Complete Specification:**
```
Likelihood:
  Y_i ~ Normal(μ_i, σ)
  μ_i = β₀ + β₁·x_i + β₂·x_i²

Priors:
  β₀ ~ Normal(1.8, 0.5)          # Intercept
  β₁ ~ Normal(0.15, 0.1)         # Linear coefficient
  β₂ ~ Normal(-0.002, 0.002)     # Quadratic coefficient (negative for concavity)
  σ ~ HalfNormal(0.25)           # Residual SD
```

**Theoretical Justification:**
- EDA found quadratic model has best R² (0.86)
- Flexible enough to capture curvature
- Still interpretable (vertex, curvature, etc.)
- Linear in parameters (easy sampling)

**Success Criteria:**
- R-hat < 1.01, ESS > 400
- β₂ < 0 (concave, not convex)
- Better LOO-CV than logarithmic (ΔELPD > 2×SE)
- Reasonable extrapolation (no runaway predictions for x>35)
- Residuals normal, no patterns

**Failure Criteria (ABANDON if):**
- β₂ > 0 (wrong curvature - would predict U-shape)
- Extrapolation predicts Y < 0 or Y > 5 for x ∈ [35, 50]
- No improvement over Model 1 in LOO-CV
- Systematic residual patterns

**Implementation:** Stan or PyMC
**Expected Runtime:** <5 minutes

---

### Model 4: B-Spline with Shrinkage (FLEXIBLE - Expected Success: 40%)

**Source:** Designer 3 (Primary)

**Complete Specification:**
```
Likelihood:
  Y_i ~ Normal(μ_i, σ)
  μ_i = β₀ + Σ_{k=1}^9 β_k · B_k(x_i)

Priors:
  β₀ ~ Normal(2.3, 0.5)                    # Global intercept
  β_k ~ Normal(0, τ_k · τ_global)          # Spline coefficients
  τ_k ~ HalfCauchy(0, 1)                   # Local shrinkage
  τ_global ~ HalfCauchy(0, 0.2)            # Global shrinkage
  σ ~ HalfNormal(0.3)                      # Residual SD

Basis:
  9 cubic B-spline basis functions
  5 internal knots at quantiles [0.2, 0.4, 0.6, 0.8, 0.95]
```

**Theoretical Justification:**
- Flexible local adaptation without global constraints
- Hierarchical shrinkage prevents overfitting
- Can discover if relationship is truly simple or complex
- Better uncertainty quantification in sparse regions

**Success Criteria:**
- Convergence (R-hat < 1.01, ESS > 400)
- Effective degrees of freedom: 3-6 (indicates regularization working)
- LOO-CV improvement over Model 1 (ΔELPD > 3×SE, higher bar for complexity)
- No oscillations between knots in posterior mean
- Residuals normal, improved over simpler models

**Failure Criteria (ABANDON if):**
- No shrinkage (all τ_k similar and large)
- Overfitting: LOO-CV worse than Model 1 or 3
- Effective DF > 10 (too flexible for N=27)
- Computational failure (divergences, low ESS)
- Oscillations in posterior predictive (Runge phenomenon)

**Implementation:** PyMC (preferred, good spline support)
**Expected Runtime:** 10-15 minutes
**Note:** Only fit if Models 1-3 show systematic inadequacies

---

### Model 5: Gaussian Process (FLEXIBLE - Expected Success: 30%)

**Source:** Designer 3 (Alternative 1)

**Complete Specification:**
```
Likelihood:
  Y_i ~ Normal(f(x_i), σ)
  f ~ GP(β₀, k(x, x'))
  k(x, x') = η² · exp(-(x - x')² / (2ℓ²))

Priors:
  β₀ ~ Normal(2.3, 0.3)        # Mean function
  η² ~ HalfNormal(0.5)         # Signal variance
  ℓ ~ InverseGamma(5, 10)      # Length scale
  σ ~ HalfNormal(0.2)          # Noise SD
```

**Theoretical Justification:**
- Nonparametric: no assumed functional form
- Optimal uncertainty quantification
- Learns smoothness from data via length scale
- Principled extrapolation (reverts to prior mean)

**Success Criteria:**
- Convergence (may require >4000 iterations)
- Length scale ℓ ∈ [1, 15] (interpretable smoothness)
- Substantial LOO-CV improvement over all parametric models (ΔELPD > 5×SE)
- Posterior predictive shows excellent fit
- Credible intervals narrower in dense regions, wider in sparse regions

**Failure Criteria (ABANDON if):**
- Length scale collapse (ℓ < 0.5) or escape (ℓ > 50)
- Computational failure (O(N³) scaling problematic even at N=27)
- No improvement over simpler models
- Overfitting: LOO-CV shows Pareto k > 0.7 for >30% of points

**Implementation:** PyMC (preferred for GP)
**Expected Runtime:** 15-30 minutes
**Note:** Only fit if all parametric models inadequate

---

## Experiment Workflow

### Phase 1: Core Models (Required)

**Experiment 1:** Fit Model 1 (Logarithmic)
- Complete validation pipeline
- If ADEQUATE → proceed to Phase 4 (assessment)
- If INADEQUATE → continue to Experiment 2

**Experiment 2:** Fit Model 2 (Michaelis-Menten)
- Complete validation pipeline
- Compare to Model 1 via LOO-CV
- If either model ADEQUATE → proceed to Phase 4
- If both INADEQUATE → continue to Experiment 3

### Phase 2: Extended Evaluation (If Needed)

**Experiment 3:** Fit Model 3 (Quadratic)
- Three-way comparison via LOO-CV
- If any model ADEQUATE → proceed to Phase 4
- If all inadequate → consider Phase 3

### Phase 3: Flexible Models (If All Parametric Fail)

**Experiment 4:** Fit Model 4 (B-Spline) OR Model 5 (GP)
- Choose based on computational constraints
- If convergence issues with GP, use B-Spline
- Final validation and assessment

### Phase 4: Model Assessment & Comparison (Always Run)

After Phase 3 completes, assess all ACCEPTED models:
- Single model: LOO diagnostics, calibration, absolute metrics
- Multiple models: LOO comparison, parsimony rule application
- Document in `experiments/model_assessment/` or `experiments/model_comparison/`

### Phase 5: Adequacy Assessment

Evaluate overall modeling effort:
- ADEQUATE → proceed to reporting
- CONTINUE → refine best candidates
- STOP → document limitations

---

## Validation Pipeline (All Models)

Each model must pass through:

1. **Prior Predictive Check** (pre-fit)
   - Simulate from priors only
   - Check predictions are plausible
   - FAIL → adjust priors, retry once

2. **Simulation-Based Validation** (pre-fit)
   - Generate fake data from model
   - Fit model, check parameter recovery
   - FAIL → model not identifiable, skip

3. **Model Fitting** (MCMC)
   - Convergence: R-hat < 1.01, ESS > 400
   - No divergent transitions (or <1%)
   - FAIL → try tuning, then skip

4. **Posterior Predictive Check** (post-fit)
   - Visual: overlay data with posterior predictive draws
   - Quantitative: coverage, calibration
   - Document quality (not automatic fail)

5. **Model Critique** (decision point)
   - Assess: ACCEPT / REVISE / REJECT
   - If borderline, use parallel critics

---

## Decision Rules

### When to Accept a Model
- Passes all convergence diagnostics
- Residuals show no systematic patterns
- Posterior predictive coverage >90%
- Parameters scientifically plausible
- If multiple models adequate: choose simplest or best LOO-CV

### When to Refine a Model
- Fixable issues identified (e.g., missing covariate)
- Convergence issues that may respond to reparameterization
- Prior-posterior conflict suggesting prior adjustment

### When to Reject a Model
- Fundamental misspecification (wrong likelihood, wrong functional form)
- Multiple refinement attempts fail
- Better alternative available

### Parsimony Rule (Model Comparison)
If two models have LOO-CV difference |ΔELPD| < 2×SE:
- Choose simpler model (fewer parameters)
- Prefer interpretable over flexible

---

## Stopping Criteria

**Stop iterating and report if:**
1. Any model meets ADEQUATE criteria
2. Diminishing returns: improvements < 2×SE
3. 5 model classes attempted with no success
4. Computational limits reached

**If no adequate model found:**
- Consider data limitations (N=27, sparse x>20)
- Report best available model with caveats
- Recommend data collection strategies

---

## Expected Outcomes

**Most Likely (70%):** Model 1 (Logarithmic) is adequate
- Simplest model that captures diminishing returns
- Strong EDA support
- Easy to interpret and communicate

**Alternative (20%):** Model 2 (Michaelis-Menten) or Model 3 (Quadratic) preferred
- LOO-CV shows meaningful improvement
- Theoretical justification for saturation
- Still interpretable

**Unlikely (10%):** Flexible models (4-5) needed
- Systematic inadequacies in parametric forms
- Evidence of more complex relationship
- Higher uncertainty in conclusions

---

## Implementation Notes

### Software Stack
- **Primary:** Stan via CmdStanPy (better for complex models)
- **Alternative:** PyMC (better for GP, splines)
- **Diagnostics:** ArviZ for all models
- **Visualization:** Matplotlib/Seaborn

### Required Outputs (All Models)
- Stan/PyMC code in `experiments/experiment_N/code/`
- MCMC diagnostics plots
- Posterior predictive check plots
- LOO-CV results with log_likelihood saved
- Summary statistics (R-hat, ESS, parameter estimates)
- Decision document (ACCEPT/REVISE/REJECT)

### Log-Likelihood Requirement
All models must save pointwise log-likelihood for LOO-CV:
- **Stan:** Add `vector[N] log_lik` in generated quantities
- **PyMC:** Use `pm.Deterministic` or ensure log_likelihood group in InferenceData
- **ArviZ:** Verify with `az.loo(idata)` works without error

---

## Summary

**Primary Strategy:** Fit logarithmic model (Model 1), validate thoroughly, stop if adequate.

**Backup Plan:** If Model 1 inadequate, evaluate Michaelis-Menten (Model 2) and Quadratic (Model 3).

**Last Resort:** If all parametric models fail, use B-Spline (Model 4) or GP (Model 5).

**Philosophy:** "The simplest model that passes all checks is the best model."

**Minimum Attempts:** At least Models 1 and 2 unless Model 1 fails pre-fit validation.

---

## File Outputs

- **This plan:** `/workspace/experiments/experiment_plan.md`
- **Designer proposals:** `/workspace/experiments/designer_{1,2,3}/proposed_models.md`
- **Experiments:** `/workspace/experiments/experiment_{1,2,3,...}/`
- **Assessment:** `/workspace/experiments/model_assessment/` or `model_comparison/`
- **Adequacy:** `/workspace/experiments/adequacy_assessment.md`

---

**Next Step:** Begin Phase 1, Experiment 1 (Logarithmic Regression)
