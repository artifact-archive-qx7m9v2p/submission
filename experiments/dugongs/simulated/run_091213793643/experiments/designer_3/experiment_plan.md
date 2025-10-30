# Experiment Plan: Hierarchical/Compositional Modeling Strategy
## Designer 3 - Structured Decomposition Perspective

**Date**: 2025-10-28
**Objective**: Design Bayesian models that decompose the Y-x relationship into interpretable hierarchical components
**Data**: N=27, x∈[1,31.5], Y∈[1.71,2.63], 6 x-values with replicates

---

## Core Philosophy: Decompose, Test, Falsify

From a hierarchical/compositional perspective, I view the data generation process as **multi-level**:
- Systematic trend (population level)
- Structured deviations (group or spatial level)
- Observation-level noise (measurement level)

My strategy: **Build models that explicitly separate these levels**, then test whether each level is necessary.

---

## Proposed Model Classes

### Model 1: Additive Decomposition Model
**Structure**: Y = Parametric Trend + Gaussian Process + Noise

**Key Features**:
- Separates interpretable saturation (log trend) from smooth deviations (GP)
- GP provides uncertainty-aware interpolation in data gap x∈[23,29]
- Three-level variance decomposition

**Falsification**: Abandon if GP amplitude η ≈ 0 (no structured deviations exist)

**Priority**: LOWEST (most complex, weakest prior evidence)

---

### Model 2: Hierarchical Replicate Model
**Structure**: Y = Population Trend + Group Effects + Measurement Error

**Key Features**:
- Explicitly models 21/27 replicate observations
- Decomposes variance into between-group (σ_between) and within-group (σ_within)
- Partial pooling: groups share information via hierarchical structure

**Falsification**: Abandon if ICC ≈ 0 (no hierarchy needed) OR σ_between >> σ_within (replicates aren't true replicates)

**Priority**: HIGHEST (directly addresses data structure)

---

### Model 3: Compositional Variance Model
**Structure**: Y ~ Normal(μ(x), σ(x)) with both mean and variance as functions of x

**Key Features**:
- Location-scale model: explicitly models heteroscedasticity
- log(σ) = γ_0 + γ_1·log(x) captures observed 4.6:1 variance ratio
- Tests whether variance structure is real or artifact of small n

**Falsification**: Abandon if γ_1 credible interval includes 0 AND LOO doesn't improve

**Priority**: MEDIUM (EDA evidence exists but not statistically significant)

---

## Falsification Framework

### Global Red Flags (Abandon All Models)
1. **Computational failure**: Stan won't converge after tuning
2. **Nonsensical predictions**: All models predict outside reasonable range
3. **Severe miscalibration**: Coverage < 0.80 or > 0.99 for all models
4. **No improvement over baseline**: All models worse than mean(Y)

### Model-Specific Falsification
- **Model 1**: GP component negligible (η posterior ≈ 0) → Use simpler parametric model
- **Model 2**: No between-group variance (σ_between ≈ 0) → Replicates are pure noise
- **Model 3**: Variance trend includes 0 (γ_1 CI crosses 0) → Constant variance sufficient

---

## Decision Points and Pivot Plans

### Decision Point 1: After Initial Fits
**Check**: Do models converge and make sense?
- **If YES**: Proceed to model comparison
- **If NO**: Diagnose issue (priors? parameterization? model misspecification?)

### Decision Point 2: Model Comparison
**Check**: Is there a clear winner (ΔLOO > 4)?
- **If YES**: Select best model, proceed to validation
- **If NO**: Multiple models similar → report uncertainty, test hybrid models

### Decision Point 3: Replicate Structure Test
**Check**: Can model predict held-out replicates (>90% coverage)?
- **If NO**: Replicates contain hidden structure (batches? time?) → **PIVOT** to batch-aware model

### Decision Point 4: Gap Interpolation
**Check**: Are predictions in x∈[23,29] reasonable?
- **If NO**: Large uncertainty or implausible values → **PIVOT** to more flexible model or acknowledge limitation

### Decision Point 5: Influential Point Sensitivity
**Check**: Do conclusions change without x=31.5?
- **If YES**: Parameter shifts > 20% → **STOP** and recommend more high-x data collection

---

## Pivot Strategies

### Pivot 1: If Replicate Test Fails
**Symptom**: None of my models capture replicate variability adequately

**New Direction**: Add batch/experimental condition effects
```
Y_ij ~ Normal(μ_ij, σ)
μ_ij = f(x_j) + batch_effect[b_j]
```

**Rationale**: Replicates may represent unobserved grouping

---

### Pivot 2: If Variance Models All Fail
**Symptom**: Neither constant nor log-linear variance fits pattern

**New Direction**: Try robust Student-t likelihood
```
Y_i ~ StudentT(ν, μ_i, σ_i)
```

**Rationale**: Heavy tails may explain variance structure better than heteroscedasticity

---

### Pivot 3: If Trend Misspecified
**Symptom**: Large GP component or systematic residuals

**New Direction**: Try alternative saturating functions
```
μ = θ_max · x / (θ_half + x)  [Michaelis-Menten]
```

**Rationale**: Logarithmic trend may not capture true functional form

---

## Stress Tests

### Test 1: Prior-Posterior Conflict
**Method**: Compare prior predictive vs posterior predictive distributions
**Red flag**: KS statistic > 0.5 (model fighting the data)
**Action**: Respecify priors OR reconsider model class

### Test 2: Posterior Predictive Calibration
**Method**: Check if 95% intervals contain ~95% of observations
**Red flag**: Coverage < 0.85 or > 0.99
**Action**: Model is miscalibrated → adjust variance structure

### Test 3: Replicate Prediction Cross-Validation
**Method**: Hold out one replicate per group, predict it
**Red flag**: Success rate < 85%
**Action**: Model doesn't capture replicate variability → need hierarchical structure

---

## Success Criteria

### Computational Success
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- No divergent transitions (or < 1%)
- Reasonable runtime (< 5 minutes per model)

### Statistical Success
- At least one model has LOO-RMSE < 0.15
- Posterior predictive coverage ∈ [0.93, 0.97]
- Parameter estimates scientifically plausible
- Model comparison clearly distinguishes approaches

### Scientific Success
- Understand which structural features are necessary (hierarchy? variance?)
- Quantify uncertainty in gap region x∈[23,29]
- Provide actionable recommendations for future data collection
- Honest reporting: clearly state what n=27 CANNOT identify

---

## Implementation Roadmap

### Phase 1: Parallel Model Fitting (Hours 0-2)
1. Implement three Stan models
2. Fit each with 4 chains, 2000 iterations
3. Check convergence diagnostics
4. Generate posterior predictive samples

**Deliverable**: Three fitted models with diagnostics

---

### Phase 2: Model Comparison (Hours 2-4)
1. Compute LOO-CV for each model
2. Run three stress tests
3. Perform cross-model diagnostics
4. Identify best model(s)

**Deliverable**: Model comparison table, diagnostic results

---

### Phase 3: Model Selection (Hours 4-6)
1. Select final model based on evidence
2. If needed, implement hybrid model
3. Extended sampling for final model
4. Comprehensive posterior checks

**Deliverable**: Final model specification with justification

---

### Phase 4: Validation (Hours 6-8)
1. Sensitivity analysis (exclude x=31.5)
2. Prior sensitivity checks
3. Gap interpolation assessment
4. Recommendations for future work

**Deliverable**: Complete validation report

---

## Stopping Rules

### Stop and Declare Success When:
1. Model achieves all computational success criteria
2. Posterior predictive checks validate model
3. Parameter posteriors are interpretable
4. Scientific questions answered (even if answer is "data insufficient")

### Stop and Pivot When:
1. All models fail convergence
2. All models fail replicate test
3. All models show major prior-posterior conflict
4. Clear evidence of model misspecification

### Stop and Recommend New Data When:
1. High sensitivity to influential point (x=31.5)
2. Gap uncertainty too large for reliable interpolation
3. Cannot distinguish between competing models (all ΔLOO < 2)
4. Variance structure hints at heteroscedasticity but n=27 insufficient to confirm

---

## Expected Outcomes

### Most Likely Scenario
**Winner**: Model 2 (Hierarchical Replicate)

**Reasoning**: Replicates are most salient data feature, hierarchical structure provides natural variance decomposition

**Expected Results**:
- ICC ∈ [0.02, 0.20] (small but meaningful between-group variance)
- α ∈ [1.6, 1.9], β ∈ [0.20, 0.34]
- LOO-RMSE ≈ 0.13

---

### Alternative Scenario
**Winner**: Model 3 (Compositional Variance) selected but γ_1 includes 0

**Reasoning**: Heteroscedasticity exists but n=27 too small to detect decisively

**Expected Results**:
- γ_1 ∈ [-0.4, 0.1] (negative mode, but wide CI)
- Better prediction intervals despite weak statistical evidence
- Recommendation: collect more data to confirm

---

### Surprising Scenario
**Winner**: Model 1 with large GP component

**Interpretation**: Logarithmic trend is WRONG, structured deviations exist

**Action**: Examine GP predictions to identify what's being captured, pivot to alternative parametric form

---

## What Would Make Me Reconsider Everything

1. **σ_between >> σ_within in Model 2**: Replicates aren't true replicates → need metadata about experimental conditions
2. **Large GP amplitude in Model 1**: Parametric trend fundamentally wrong → try alternative functional forms
3. **All models equivalent (ΔLOO < 2)**: Data insufficient to distinguish → report honest uncertainty
4. **High influential point sensitivity**: Single point drives conclusions → need more high-x data before confident inference

---

## Key Contributions of This Perspective

### Unique Aspects
1. **Explicit variance decomposition**: Hierarchical structure separates sources of variation
2. **Replicate structure modeling**: 21/27 observations leveraged, not treated as independent
3. **Compositional thinking**: Models built by adding/testing components, not monolithic

### Complementary to Other Designers
- Other designers may focus on functional form (parametric vs non-parametric)
- I focus on **structure**: what are the levels of variation and how do they interact?
- Provides orthogonal perspective: even if we agree on mean function, variance structure matters

### Value Even If Models Fail
- If ICC ≈ 0: We learn replicates are pure noise (important negative result)
- If γ_1 includes 0: We quantify that n=27 insufficient to detect heteroscedasticity (informs future experiments)
- If all complex models fail: We learn simple model is sufficient (parsimony wins)

---

## Reproducibility and Transparency

### Code Availability
All Stan and Python code provided in `/workspace/experiments/designer_3/`

### Random Seeds
All sampling uses fixed seeds for reproducibility

### Reporting Commitment
- Will report ALL models, not just winner
- Will clearly state limitations and uncertainties
- Will document all pivot decisions with justification
- Will provide honest assessment even if results are negative

---

## Summary

I propose **three hierarchical/compositional models** that decompose the Y-x relationship into interpretable components. My **highest priority** is Model 2 (Hierarchical Replicate) which directly addresses the replicate structure in the data. Each model has **clear falsification criteria**, and I've defined **pivot plans** if initial approaches fail.

**Key insight**: With 21/27 observations being replicates, the variance structure (between-group vs within-group) is potentially as important as the mean function. My models explicitly test whether this structure matters.

**Success metric**: Not just "model fits well" but "I understand WHICH structural features are necessary and which add unnecessary complexity". Even negative results (e.g., ICC ≈ 0) are scientifically valuable.

**File locations**:
- Full proposal: `/workspace/experiments/designer_3/proposed_models.md`
- This summary: `/workspace/experiments/designer_3/experiment_plan.md`

---

**END OF EXPERIMENT PLAN**
