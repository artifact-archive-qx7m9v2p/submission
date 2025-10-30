# Unified Experiment Plan: Bayesian Models for Y vs x Relationship

**Date**: 2025-01-XX
**Dataset**: N=27, x ∈ [1, 31.5], Y ∈ [1.71, 2.63]
**Status**: Synthesized from 3 parallel model designers

---

## Executive Summary

After reviewing proposals from three independent model designers with different perspectives (parametric, flexible/robust, hierarchical/compositional), I have synthesized a unified experiment plan that covers the model space systematically. The plan prioritizes models by: (1) Theoretical justification, (2) Parsimony, (3) EDA alignment, (4) Robustness.

**Total Models Proposed**: 9 distinct classes
**Selected for Implementation**: 5 priority models
**Minimum Attempts**: 2 (as per policy)

---

## Model Space Coverage

### Designer 1 (Parametric) - 3 models:
- Logarithmic (unbounded growth)
- Michaelis-Menten (true asymptote)
- Quadratic (polynomial approximation)

### Designer 2 (Flexible/Robust) - 3 models:
- Gaussian Process (non-parametric)
- Robust Regression with Student-t errors
- Penalized B-Splines

### Designer 3 (Hierarchical/Compositional) - 3 models:
- Hierarchical Replicate Model
- Compositional Variance Model (heteroscedastic)
- Additive Decomposition (Trend + GP)

---

## Synthesis: Removing Duplicates & Identifying Priorities

### Overlapping Concepts:
1. **GP appears twice**: Designer 2 (standalone) & Designer 3 (additive component)
   - **Resolution**: Implement Designer 2's standalone GP first; only add Designer 3's additive version if GP alone shows promise

2. **Logarithmic trend**: Appears in Designer 1 and implicitly in Designer 2 (robust) and Designer 3 (hierarchical)
   - **Resolution**: Designer 1's logarithmic is the baseline; Designer 2/3 are extensions

3. **Variance modeling**: Designer 2 (robust-t) and Designer 3 (heteroscedastic) address similar concerns
   - **Resolution**: Try robust-t first (simpler); escalate to heteroscedastic if needed

### Unique Contributions:
- **Designer 1**: Specific functional forms (MM, quadratic) with clear hypotheses
- **Designer 2**: Robustness focus (Student-t, B-splines)
- **Designer 3**: Replicate structure modeling (hierarchical decomposition)

---

## Prioritized Experiment Plan

Following the Minimum Attempt Policy, I will attempt at least 2 models. Here's the prioritized order:

### **Experiment 1: Logarithmic Regression** (HIGHEST PRIORITY)
**Source**: Designer 1, Model 1
**Rationale**:
- Best EDA fit (R²=0.829)
- Simplest model (2 parameters: α, β)
- Strong theoretical justification (Weber-Fechner law)
- Parsimony principle favors this

**Model Specification**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = α + β·log(x_i)

Priors:
α ~ Normal(1.75, 0.5)    # Intercept (from EDA)
β ~ Normal(0.27, 0.15)   # Log-slope (positive)
σ ~ HalfNormal(0.2)      # Residual SD
```

**Falsification Criteria**:
- Abandon if systematic residual pattern at high x (PPC p-value < 0.05)
- Abandon if LOO worse than next model by >4
- Abandon if Pareto k > 0.7 for >5 observations

**Expected Outcome**: Most likely winner
**Implementation**: Stan (CmdStanPy)

---

### **Experiment 2: Hierarchical Replicate Model** (HIGH PRIORITY)
**Source**: Designer 3, Model 2
**Rationale**:
- Addresses key data structure: 21/27 observations are replicates
- Tests whether between-group variance is meaningful
- Orthogonal to Experiment 1 (different modeling question)

**Model Specification**:
```
Y_ij ~ Normal(μ_j + δ_j, σ_within)
μ_j = α + β·log(x_j)
δ_j ~ Normal(0, σ_between)

Priors:
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
σ_within ~ HalfNormal(0.15)
σ_between ~ HalfNormal(0.1)
```

**Falsification Criteria**:
- Abandon if ICC ≈ 0 (no hierarchy needed)
- Abandon if σ_between >> σ_within (replicates aren't true replicates)
- Abandon if LOO doesn't improve over Experiment 1

**Expected Outcome**: Will quantify replicate structure; may not improve prediction
**Implementation**: Stan (CmdStanPy)

---

### **Experiment 3: Robust Regression (Student-t)** (MEDIUM PRIORITY)
**Source**: Designer 2, Model 2
**Rationale**:
- Protects against influential point at x=31.5
- Tests whether outlier robustness is needed
- Simple extension of Experiment 1 (adds 1 parameter: ν)

**Model Specification**:
```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = α + β·log(x_i)

Priors:
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
ν ~ Gamma(2, 0.1)    # Allows data to choose robustness
σ ~ HalfNormal(0.2)
```

**Falsification Criteria**:
- Abandon if ν_posterior > 50 (no outliers detected)
- Abandon if LOO worse than Gaussian by >2

**Expected Outcome**: Will assess outlier sensitivity; may not be necessary
**Implementation**: Stan (CmdStanPy)

---

### **Experiment 4: Michaelis-Menten (Asymptotic)** (MEDIUM PRIORITY)
**Source**: Designer 1, Model 2
**Rationale**:
- Tests key scientific hypothesis: Is there a true asymptote?
- Different from logarithmic (bounded vs unbounded)
- 3 interpretable parameters (Y_max, Y_min, K)

**Model Specification**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = Y_max - (Y_max - Y_min)·K/(K + x_i)

Priors:
Y_max ~ Normal(2.7, 0.3)     # Upper asymptote
Y_min ~ Normal(1.8, 0.2)     # Lower asymptote
K ~ Normal(5, 3)             # Half-saturation constant
σ ~ HalfNormal(0.15)
```

**Falsification Criteria**:
- Abandon if Y_max posterior unbounded (SD > 1.0)
- Abandon if K > 25 (no saturation observed in data range)
- Abandon if LOO worse than logarithmic by >4

**Expected Outcome**: Likely weak identifiability of Y_max; may not distinguish from log
**Implementation**: Stan (CmdStanPy) with careful reparameterization

---

### **Experiment 5: Gaussian Process Regression** (LOWER PRIORITY)
**Source**: Designer 2, Model 1
**Rationale**:
- Non-parametric benchmark (no functional form assumed)
- Tests whether parametric models are adequate
- Provides uncertainty-aware interpolation in gap x∈[23,29]

**Model Specification**:
```
f ~ GP(μ_0, K(x, x'))
K(x, x') = α²·Matérn3/2(x, x'; ℓ)
Y_i ~ Normal(f(x_i), σ)

Priors:
μ_0 ~ Normal(2.3, 0.5)       # Mean function
α² ~ HalfNormal(0.3)         # Signal variance
ℓ ~ InverseGamma(5, 10)      # Lengthscale (mode ≈ 2-3)
σ ~ HalfNormal(0.15)         # Noise
```

**Falsification Criteria**:
- Abandon if ℓ_posterior >> 30 (data too simple for GP)
- Abandon if LOO worse than logarithmic by >3
- Abandon if GP predictions are constant (no structure learned)

**Expected Outcome**: Likely overkill for this data; serves as upper bound on complexity
**Implementation**: Stan (CmdStanPy) - may have computational challenges

---

## Models NOT Selected (With Justification)

### Quadratic Model (Designer 1, Model 3)
**Reason**: Good empirical fit but dangerous for extrapolation; vertex concerns
**Decision**: Use as sensitivity check if needed, not primary model

### Penalized B-Splines (Designer 2, Model 3)
**Reason**: Similar flexibility to GP but less interpretable; redundant with Experiment 5
**Decision**: Backup if GP fails computationally

### Compositional Variance Model (Designer 3, Model 3)
**Reason**: EDA shows 4.6:1 variance ratio, but not statistically significant with n=27
**Decision**: Add heteroscedasticity only if Experiments 1-3 show poor calibration

### Additive Decomposition (Designer 3, Model 1)
**Reason**: Most complex model; only justified if GP alone shows clear deviations
**Decision**: Escalate to this only if Experiment 5 reveals structured residuals

---

## Implementation Strategy

### Phase 3: Model Development Loop

**Order of Execution**:
1. **Experiment 1** (Logarithmic) - REQUIRED
2. **Experiment 2** (Hierarchical) - REQUIRED (satisfies Minimum Attempt Policy)
3. **Experiment 3** (Robust-t) - Conditional on Experiments 1-2 results
4. **Experiment 4** (Michaelis-Menten) - Conditional on Experiment 1 results
5. **Experiment 5** (Gaussian Process) - If parametric models inadequate

**Validation Pipeline** (for each experiment):
```
Prior Predictive Check
    ↓ PASS
Simulation-Based Validation
    ↓ PASS
Model Fitting (with log_likelihood)
    ↓ PASS
Posterior Predictive Check
    ↓
Model Critique → [ACCEPT/REVISE/REJECT]
```

**Stopping Conditions**:
- Stop early if Experiment 1 clearly adequate (LOO-RMSE < 0.12, good calibration)
- Stop if 2 ACCEPTed models found and ΔLOO < 2 (equivalent models)
- Stop if computational budget exhausted (unlikely with n=27)

---

## Decision Rules

### When One Model Wins Decisively (ΔLOO > 4):
- Accept as primary model
- Document why others failed
- Proceed to Phase 4 (Assessment)

### When Models Equivalent (ΔLOO < 2):
- Apply parsimony rule: choose simpler model
- Consider Bayesian model averaging
- Report structural uncertainty

### When All Models Fail:
- Check computational issues (priors, parameterization)
- Escalate to backup models (quadratic, splines)
- Consider non-Bayesian preprocessing to diagnose issues

---

## Sensitivity Analyses (Required for All ACCEPTed Models)

1. **Influential Point**: Remove x=31.5, refit
2. **Prior Sensitivity**: Scale all priors by 2×, check parameter shifts
3. **Gap Assessment**: Fit on x≤20, predict x∈[20,31.5]
4. **Replicate Variance**: Check within-group consistency
5. **Synthetic Recovery**: Generate data from fitted model, attempt recovery

---

## Success Criteria

### Computational Success:
- Rhat < 1.01 for all parameters
- ESS > 400 (10% of 4000 post-warmup samples)
- Divergences < 1% of total iterations
- Reasonable runtime (< 5 min per model)

### Statistical Success:
- At least 1 model with LOO-RMSE < 0.15
- Posterior predictive coverage ∈ [0.90, 0.97]
- Pareto k < 0.7 for >90% of observations
- Parameters scientifically interpretable

### Scientific Success:
- Understand functional form of relationship (bounded vs unbounded)
- Quantify uncertainty in gap region x∈[23,29]
- Assess whether replicate structure matters
- Provide actionable recommendations

---

## Red Flags Requiring Plan Adjustment

### Global Red Flags (affect all models):
1. **Influential point dominates**: All models highly sensitive to x=31.5 → Need more high-x data
2. **Gap predictions diverge**: All models disagree wildly in x∈[23,29] → Cannot interpolate safely
3. **Convergence failures**: Stan won't sample → Reparameterization or PyMC fallback
4. **All models fail PPC**: Systematic misfit → Need different model class entirely

### Model-Specific Red Flags:
- **Logarithmic**: Systematic residuals at high x → Try MM or quadratic
- **Hierarchical**: σ_between >> σ_within → Replicates aren't true replicates (need metadata)
- **Robust-t**: ν_posterior very low (< 5) → True outliers present, investigate
- **MM**: Y_max unbounded → Data insufficient for asymptote estimation
- **GP**: Lengthscale very large → Data too simple, use parametric model

---

## Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| Setup | Environment, data prep | 15 min |
| Exp 1 | Log model (all stages) | 1.5 hr |
| Exp 2 | Hierarchical model | 1.5 hr |
| Assessment | LOO comparison, decide next | 30 min |
| Exp 3 | Robust-t (if needed) | 1 hr |
| Exp 4 | MM (if needed) | 1 hr |
| Exp 5 | GP (if needed) | 1.5 hr |
| **Total** | **Minimum (Exp 1-2 only)** | **~4 hrs** |
| **Total** | **Complete (all 5)** | **~7 hrs** |

---

## Key Design Principles (Synthesized from All Designers)

1. **Falsificationism**: Success = finding what's wrong early
2. **Parsimony**: Prefer simpler models unless complexity justified by ΔLOO > 2-3
3. **Robustness**: Small n=27 requires protection against assumptions
4. **Structure**: Explicitly model data features (replicates, variance)
5. **Honesty**: Admit uncertainty, especially in gap region
6. **Iteration**: Plan for refinement, not one-shot success

---

## Output Structure (for each experiment)

```
experiments/
└── experiment_N/
    ├── metadata.md                    # Model specification & rationale
    ├── prior_predictive_check/
    │   ├── code/
    │   ├── plots/
    │   └── findings.md
    ├── simulation_based_validation/
    │   ├── code/
    │   ├── plots/
    │   └── recovery_metrics.md
    ├── posterior_inference/
    │   ├── code/
    │   ├── diagnostics/
    │   │   └── posterior_inference.netcdf  # ArviZ InferenceData with log_lik
    │   ├── plots/
    │   └── inference_summary.md
    ├── posterior_predictive_check/
    │   ├── code/
    │   ├── plots/
    │   └── ppc_findings.md
    └── model_critique/
        ├── critique_summary.md
        ├── decision.md                # ACCEPT/REVISE/REJECT
        └── improvement_priorities.md
```

---

## Coordination with Existing Work

**EDA Findings** (from Phase 1):
- Logarithmic best fit (R²=0.829, RMSE=0.115)
- Quadratic slightly better (R²=0.862) but overfit risk
- Variance approximately constant (weak hetero evidence)
- Influential point: x=31.5 (Cook's D=0.84)
- Gap: x∈[23,29] sparse

**This Plan's Response**:
- Experiment 1 directly tests EDA's logarithmic recommendation
- Experiment 2 tests replicate structure (not explored in EDA)
- Experiment 3 addresses influential point concern
- Experiment 4 tests bounded vs unbounded hypothesis
- Experiment 5 provides non-parametric upper bound

---

## Minimum Attempt Commitment

Per Minimum Attempt Policy:
- **Will attempt**: Experiments 1 & 2 (unless Exp 1 fails pre-fit validation)
- **May attempt**: Experiments 3-5 (conditional on 1-2 results)
- **Documentation**: If <2 attempted, reason recorded in iteration_log.md

---

## Final Notes

This synthesized plan:
- Covers parametric, flexible, and hierarchical model classes
- Prioritizes simplicity while acknowledging complexity may be needed
- Has clear decision points and falsification criteria
- Balances thoroughness with efficiency
- Commits to honest reporting of all results

**Ready to proceed to Phase 3: Model Development Loop**

---

**Plan Status**: FINALIZED
**Next Action**: Begin Experiment 1 (Logarithmic Regression)
**Last Updated**: 2025-01-XX
