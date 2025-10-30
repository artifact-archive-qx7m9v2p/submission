# Bayesian Modeling Experiment Plan

**Project**: Y vs x relationship modeling
**Data**: N=27 observations
**Date**: 2025-10-27

---

## Synthesis of Designer Proposals

Two independent model designers have proposed **6 total model classes**, with strong convergence on logarithmic transformations and variance modeling. After removing duplicates and prioritizing by theoretical justification and feasibility, the final experiment plan includes **4 distinct model classes**.

### Designer Convergence Analysis

**Strong Agreement:**
- Both designers prioritized log transformation of x (Designer 1: Models 1-2, Designer 2: Model 2)
- Both recognized heteroscedasticity as critical (Designer 1: Model 3, Designer 2: Models 1-2)
- Both proposed robustness to influential point 26 (Designer 1: Model 2, Designer 2: Model 3)
- Both emphasized small sample constraints (n=27 → max 3-4 parameters)

**Complementary Perspectives:**
- Designer 1: Focus on log-scale transformations (cleaner residuals)
- Designer 2: Focus on original-scale interpretation (explicit variance modeling)
- Designer 1: Student-t on log scale
- Designer 2: Student-t on original scale, polynomial alternatives

### Selected Model Classes (Prioritized)

After synthesis, we will test **4 model classes** in order:

| Priority | Model Name | Parameters | Designer | Rationale |
|----------|-----------|------------|----------|-----------|
| **1** | Log-Log Linear | 3 | Designer 1 | Best EDA fit, simplest, strong theoretical basis |
| **2** | Log-Linear Heteroscedastic | 4 | Designer 2 | Combines log(x) + variance modeling |
| **3** | Robust Log-Log (Student-t) | 4 | Designer 1 | Robustness to point 26 |
| **4** | Quadratic Heteroscedastic | 5 | Designer 2 | Original-scale alternative |

**Rationale for prioritization:**
1. Model 1 is PRIMARY based on EDA convergence (R² = 0.90)
2. Model 2 adds variance modeling while keeping log(x) transformation
3. Model 3 adds robustness if point 26 problematic
4. Model 4 provides original-scale comparison if transformations fail

**Dropped models:**
- Designer 2's Robust Polynomial (redundant with Model 3, weaker theoretical basis)

---

## Experiment 1: Log-Log Linear Model (PRIMARY)

### Model Specification

**Type**: Bayesian linear regression on log-log scale
**Parameters**: 3 (alpha, beta, sigma)

**Mathematical Form:**
```
log(Y_i) ~ Normal(mu_i, sigma)
mu_i = alpha + beta * log(x_i)

Priors:
  alpha ~ Normal(0.6, 0.3)      # Log-scale intercept
  beta ~ Normal(0.13, 0.1)      # Power law exponent
  sigma ~ Half-Normal(0.1)      # Log-scale residual SD
```

**Interpretation:**
- Back-transforms to power law: Y = exp(alpha) × x^beta
- Expected: Y ≈ 1.82 × x^0.13 (diminishing returns)

### Stan Implementation

**File to create**: `experiments/experiment_1/model.stan`

Key features:
- Log transformation in `transformed data`
- Simple linear predictor on log-log scale
- `log_lik` vector for LOO cross-validation
- Posterior predictive samples in original scale

### Prior Justification

**Alpha**: EDA shows log(Y) ≈ 0.6, prior allows [0, 1.2] → Y-intercept [1.0, 3.3]
**Beta**: EDA power law exponent ≈ 0.126, prior allows [0, 0.33] (positive relationship)
**Sigma**: EDA residual SD ≈ 0.05, prior allows up to 0.2 (weakly informative)

### Validation Pipeline

1. **Prior Predictive Check**
   - Generate data from priors
   - Check coverage of observed range
   - Success: Generated Y ∈ [0.5, 5] encompasses observed [1.77, 2.72]

2. **Simulation-Based Validation**
   - Simulate data with known alpha=0.6, beta=0.13, sigma=0.05
   - Fit model, check parameter recovery
   - Success: 95% credible intervals cover true values

3. **Model Fitting**
   - 4 chains, 2000 iterations (1000 warmup)
   - Target: R̂ < 1.01, ESS > 400
   - Success: All diagnostics pass

4. **Posterior Predictive Check**
   - Compare observed vs simulated data
   - Check residual patterns in log scale
   - Success: No systematic patterns, good coverage

### Falsification Criteria (will REJECT if)

1. **Beta posterior contradicts EDA**:
   - Posterior mean < 0 or > 0.3
   - 95% CI doesn't contain 0.13

2. **Poor LOO diagnostics**:
   - Pareto k > 0.7 for multiple points
   - LOO-RMSE > 0.12

3. **Convergence failure**:
   - R̂ > 1.01 after tuning
   - ESS < 200

4. **Systematic residuals**:
   - U-shaped or other clear pattern
   - Shapiro-Wilk p < 0.05 on residuals

5. **Back-transformation bias**:
   - Median(exp(log_Y_pred)) systematically differs from observed Y

**If rejected**: Proceed to Experiment 2 (add variance modeling)

### Expected Performance

- **In-sample R²**: > 0.88 (close to EDA R² = 0.90)
- **LOO-RMSE**: 0.09-0.11 (on log scale)
- **LOO-ELPD**: Baseline for comparison
- **Pareto k**: < 0.5 for most points, possibly 0.5-0.7 for point 26

---

## Experiment 2: Log-Linear Heteroscedastic Model

### Model Specification

**Type**: Bayesian regression with log(x) transformation and heteroscedastic variance
**Parameters**: 4 (beta_0, beta_1, gamma_0, gamma_1)

**Mathematical Form:**
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i

Priors:
  beta_0 ~ Normal(1.8, 0.5)
  beta_1 ~ Normal(0.3, 0.2)
  gamma_0 ~ Normal(-2, 1)
  gamma_1 ~ Normal(-0.05, 0.05)
```

**Interpretation:**
- Works in original Y scale (more interpretable)
- Models variance decreasing with x (matches EDA finding of 7.5× decrease)

### Prior Justification

**Beta_0**: Y at x=1 is ≈1.77, prior allows [0.8, 2.8]
**Beta_1**: Rate per log-unit x ≈ 0.3, prior allows [0, 0.7]
**Gamma_0**: Log-variance in low-x ≈ -2.78, prior allows [-4, 0]
**Gamma_1**: Variance decreases with x, prior allows [-0.15, 0.05]

### Validation Pipeline

Same 4-step process as Experiment 1, but:
- Check heteroscedasticity in posterior predictive
- Verify gamma_1 < 0 (variance decreases)
- Ensure 4 parameters don't overfit (n=27)

### Falsification Criteria (will REJECT if)

1. **Gamma_1 posterior includes zero**:
   - 95% CI contains 0 → No heteroscedasticity, revert to simpler model

2. **Overfitting evident**:
   - LOO-ELPD worse than Model 1 by > 2 SE
   - Extreme posterior uncertainty (SE > 0.5 for beta parameters)

3. **Convergence issues**:
   - R̂ > 1.01 (too complex for n=27)
   - Divergent transitions > 5%

4. **LOO diagnostics**:
   - Pareto k > 0.7 (variance modeling not helping)

**If rejected**: Model 1 is adequate, or try Model 3 (robustness)

### Expected Performance

- **In-sample R²**: > 0.88 (similar to Model 1)
- **LOO-RMSE**: 0.08-0.10 (potentially better if heteroscedasticity strong)
- **ΔLOO vs Model 1**: +2 to +8 (variance modeling helps, but costs 1 parameter)
- **Pareto k**: Improved over Model 1 if variance modeling appropriate

### Comparison with Model 1

**Prefer Model 2 if**:
- ΔLOO > 4 (strong evidence, > 2 SE)
- Residual variance patterns in Model 1
- Gamma_1 clearly negative (95% CI < 0)

**Prefer Model 1 if**:
- ΔLOO < 2 (parsimony rule)
- Convergence issues in Model 2
- Variance modeling adds complexity without benefit

---

## Experiment 3: Robust Log-Log Model (Student-t)

### Model Specification

**Type**: Bayesian regression with Student-t likelihood (outlier-robust)
**Parameters**: 4 (alpha, beta, sigma, nu)

**Mathematical Form:**
```
log(Y_i) ~ Student_t(nu, mu_i, sigma)
mu_i = alpha + beta * log(x_i)

Priors:
  alpha ~ Normal(0.6, 0.3)
  beta ~ Normal(0.13, 0.1)
  sigma ~ Half-Normal(0.1)
  nu ~ Gamma(2, 0.1)          # Degrees of freedom
```

**Interpretation:**
- Same structure as Model 1, but heavy-tailed likelihood
- Automatically downweights influential points
- Nu ≈ 30+ → nearly Normal; Nu ≈ 5-10 → moderate robustness needed

### Prior Justification

Same as Model 1 for alpha, beta, sigma
**Nu**: Gamma(2, 0.1) has mode at 10, allows range [2, 50]
- If nu > 30: Student-t ≈ Normal, revert to Model 1
- If nu < 10: Robustness needed, validates this model

### Validation Pipeline

Same as Model 1, plus:
- Check nu posterior location
- Verify point 26 gets downweighted if nu < 20
- Ensure robustness doesn't hide model misspecification

### Falsification Criteria (will REJECT if)

1. **Nu posterior > 30**:
   - Student-t unnecessary, Model 1 is adequate
   - Extra complexity not justified

2. **Nu posterior < 3**:
   - Extreme heavy tails indicate model misspecification
   - Need different model structure, not just robustness

3. **LOO worse than Model 1**:
   - ΔLOO < -2 → Robustness hurts predictive performance
   - Model 1 is better

4. **Same Pareto k issues**:
   - Student-t doesn't improve LOO diagnostics
   - Problem is structural, not outliers

**If rejected**: Accept Model 1 (if adequate) or try Model 4

### Expected Performance

- **In-sample R²**: Similar to Model 1 (0.88-0.90)
- **LOO-RMSE**: 0.09-0.11
- **Nu posterior**: 8-15 (moderate robustness) if point 26 is problematic
- **Pareto k**: Should improve over Model 1, especially for point 26

### Comparison with Model 1

**Prefer Model 3 if**:
- Point 26 has Pareto k > 0.7 in Model 1
- Nu posterior < 20 (robustness needed)
- ΔLOO ≥ 0 (at least as good)

**Prefer Model 1 if**:
- Nu > 30 (robustness unnecessary)
- ΔLOO < -2 (simpler is better)
- Model 1 Pareto k < 0.7 for all points

---

## Experiment 4: Quadratic Heteroscedastic Model

### Model Specification

**Type**: Polynomial regression with heteroscedastic variance (original scale)
**Parameters**: 5 (beta_0, beta_1, beta_2, gamma_0, gamma_1)

**Mathematical Form:**
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * x_i + beta_2 * x_i^2
log(sigma_i) = gamma_0 + gamma_1 * x_i

Priors:
  beta_0 ~ Normal(1.8, 0.5)
  beta_1 ~ Normal(0.05, 0.05), beta_1 > 0
  beta_2 ~ Normal(-0.001, 0.002), beta_2 < 0
  gamma_0 ~ Normal(-2.5, 1)
  gamma_1 ~ Normal(-0.05, 0.03), gamma_1 < 0
```

**Interpretation:**
- Quadratic curvature captures diminishing returns
- Original scale (no transformation)
- Variance decreases with x

### Prior Justification

**Beta_0**: Y(x≈1) ≈ 1.77, prior allows [0.8, 2.8]
**Beta_1**: Initial slope > 0, prior allows [0, 0.15]
**Beta_2**: Curvature < 0, prior allows [-0.005, 0]
**Gamma_0, Gamma_1**: Same reasoning as Model 2

### Validation Pipeline

Same 4-step process, with attention to:
- Polynomial extrapolation beyond x=32 (dangerous!)
- Overfitting risk with 5 parameters and n=27
- Constraint satisfaction (beta_1 > 0, beta_2 < 0)

### Falsification Criteria (will REJECT if)

1. **Overfitting evident**:
   - LOO-ELPD worse than Models 1-2 by > 4 SE
   - 5 parameters too many for n=27

2. **Convergence failure**:
   - R̂ > 1.01 (too complex)
   - Divergent transitions

3. **Worse fit than logarithmic**:
   - R² < 0.85
   - LOO-RMSE > 0.13

4. **Implausible extrapolation**:
   - Polynomial predicts Y < 0 or Y > 5 for x ∈ [1, 35]

**If rejected**: Accept best of Models 1-3

### Expected Performance

- **In-sample R²**: 0.85-0.87 (lower than log models per EDA)
- **LOO-RMSE**: 0.10-0.13
- **ΔLOO vs Model 1**: -5 to +2 (likely worse due to overfitting)
- **Pareto k**: May improve with variance modeling

### Comparison with Models 1-3

**Prefer Model 4 if**:
- Original-scale interpretation critical
- Comparable LOO-ELPD (within 4 SE of best)
- Stakeholders reject transformations

**Prefer Models 1-3 if**:
- Better LOO-ELPD (ΔLOO > 4)
- Overfitting evident in Model 4
- Transformation acceptable

---

## Model Comparison Strategy

### Individual Model Assessment (After Each Experiment)

For each model, run `model-critique` agent to assess:
1. Prior predictive check results
2. Simulation-based validation results
3. Convergence diagnostics (R̂, ESS, divergences)
4. LOO diagnostics (ELPD, Pareto k)
5. Posterior predictive check results
6. Residual patterns

**Decision for each model**: ACCEPT, REVISE, or REJECT

- **ACCEPT**: All validation passed, meets success criteria
- **REVISE**: Fixable issues identified (adjust priors, reparameterize, etc.)
- **REJECT**: Fundamental issues, proceed to next model

### Multi-Model Comparison (After All Experiments)

Run `model-assessment-analyst` to compare all ACCEPTED models:

**Comparison Metrics:**
1. **LOO-ELPD comparison** via `az.compare`
   - Primary metric for model selection
   - Use parsimony rule: prefer simpler if ΔLOO < 2 SE

2. **Pareto k diagnostics**
   - Check if any model has k > 0.7
   - Compare k values across models for point 26

3. **Predictive accuracy**
   - LOO-RMSE (smaller is better)
   - Coverage of prediction intervals

4. **Interpretability**
   - Original scale vs log scale
   - Parameter interpretability

### Selection Criteria

**Primary**: LOO-ELPD with parsimony rule
- Prefer simpler model if ΔLOO < 2 SE
- 5-parameter Model 4 needs ΔLOO > 4 to justify complexity

**Secondary**: Scientific considerations
- Interpretability (original vs log scale)
- Extrapolation safety (polynomial dangerous)
- Robustness to influential points

**Stopping Rule**: If best model has Pareto k > 0.7 for multiple points, consider:
1. More flexible model (e.g., splines, Gaussian process)
2. Mixture model (if subpopulations evident)
3. Data quality investigation

---

## Minimum Attempt Policy

Per workflow guidelines:
- **Minimum**: Attempt Experiments 1 and 2
- **Unless**: Experiment 1 fails prior predictive or simulation validation (fundamental issue)

**Rationale**:
- Experiment 1 is PRIMARY (strongest EDA support)
- Experiment 2 tests if variance modeling improves fit
- Together, these cover the core hypothesis (logarithmic relationship)

**If both fail**:
- Investigate data quality
- Consider more flexible Bayesian approaches (GP, splines)
- Document failures in `experiments/iteration_log.md`

---

## Success Criteria (Overall)

At least one model should achieve:
- **R² > 0.85** (significantly better than linear R² = 0.68)
- **LOO-RMSE < 0.12**
- **Pareto k < 0.7** for > 90% of observations
- **R̂ < 1.01, ESS > 400** for all parameters
- **Posterior predictive checks pass** (no systematic patterns)
- **Parameter posteriors sensible** (match EDA expectations)

If no model meets these criteria:
- Document why in `experiments/adequacy_assessment.md`
- Consider if data quality/quantity is limiting factor
- Propose next steps (more data, different approach, accept limitations)

---

## Timeline and Execution

### Execution Order

1. **Experiment 1** (Log-Log Linear) - PRIMARY
   - Full validation pipeline
   - If ACCEPT and adequate → Consider stopping
   - If ACCEPT but improvable → Continue to Experiment 2
   - If REJECT → Continue to Experiment 2 anyway (minimum 2 attempts)

2. **Experiment 2** (Log-Linear Heteroscedastic)
   - Compare with Experiment 1
   - If substantially better → ACCEPT as best
   - If similar → Prefer Experiment 1 (simpler)

3. **Experiment 3** (Robust Student-t) - OPTIONAL
   - Only if point 26 problematic in Experiments 1-2
   - Compare robustness benefits vs complexity cost

4. **Experiment 4** (Quadratic) - OPTIONAL
   - Only if transformation-based models fail
   - Original-scale alternative

### Parallel Critique Policy

- **Simple pass/fail**: Single critique agent
- **Borderline/complex tradeoffs**: 2 parallel critique agents + synthesis

**Expect borderline for**:
- Model 2 vs Model 1 (variance modeling trade-off)
- Model 3 vs Model 1 (robustness vs simplicity)

---

## Documentation Plan

### For Each Experiment

**Directory structure**: `experiments/experiment_N/`

**Required files**:
1. `metadata.md` - Model specification, priors, expectations
2. `prior_predictive_check/findings.md` - Prior validation results
3. `simulation_based_validation/recovery_metrics.md` - SBC results
4. `posterior_inference/inference_summary.md` - Fit summary and diagnostics
5. `posterior_predictive_check/ppc_findings.md` - Model validation results
6. `model_critique/decision.md` - ACCEPT/REVISE/REJECT with reasoning

### Experiment Log

**File**: `experiments/iteration_log.md`

**Contents**:
- Each experiment attempt
- Decisions (ACCEPT/REVISE/REJECT)
- Refinements made
- Comparison results
- Stopping decisions

### Final Assessment

**File**: `experiments/adequacy_assessment.md`

**Contents**:
- Best model(s) identified
- Strengths and limitations
- Adequacy determination (ADEQUATE/CONTINUE/STOP)
- Recommendations for next steps

---

## Summary

**Total model classes to test**: 4
**Minimum to attempt**: 2 (Experiments 1-2)
**Expected best**: Experiment 1 or 2 (log-based models)
**Key decision point**: Variance modeling worth the complexity?
**Risk point**: Point 26 influence (may need robustness)

**Final deliverable**: One or more ACCEPTED Bayesian models with full diagnostics, comparison, and adequacy assessment, followed by comprehensive final report.

---

**Next Step**: Begin Experiment 1 (Log-Log Linear Model) with prior predictive check.
