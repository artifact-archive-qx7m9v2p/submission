# Falsification Criteria: Model Designer 2

## Philosophy: Plan for Failure

**Core Principle**: Each model should have clear criteria for rejection. Success is discovering the model is wrong early, not defending it.

---

## Model 1: Gaussian Process

### I will abandon this model if:

1. **Lengthscale extreme** (ℓ_post > 30 or ℓ_post < 0.5)
   - Too large → Data is nearly linear/simple parametric sufficient
   - Too small → Overfitting noise between points

2. **Signal-to-noise unreasonable** (α²/σ² < 0.1 or > 100)
   - Low ratio → Most variation is noise
   - High ratio → Model overfitting

3. **Posterior predictive failures**:
   - Predictions outside [1.5, 3.0] frequently
   - Non-monotonic function
   - Can't capture replicate variability

4. **Computational red flags**:
   - Rhat > 1.05 despite tuning
   - Divergences > 5%
   - Runtime > 1 hour

5. **Worse predictions than log model** (ΔELPD_LOO > 3 in favor of log)
   - Complexity penalized without benefit
   - Parsimony wins

6. **Prior-posterior conflict**:
   - Posterior far from prior despite informative prior
   - Suggests model fighting the data

---

## Model 2: Robust Student-t

### I will abandon this model if:

1. **Degrees of freedom high** (ν_post > 50 with 95% CI not including 30)
   - Data strongly prefers Gaussian
   - Extra parameter wasted

2. **Degrees of freedom extreme** (ν_post < 3)
   - Extreme outliers suggest data error
   - Or functional form completely wrong

3. **Worse than Gaussian** (ΔELPD_LOO > 2 in favor of Gaussian)
   - Robustness doesn't help
   - Unnecessary complexity

4. **Posterior predictive unrealistic**:
   - Heavy tails generate Y outside [1, 3]
   - Distribution much wider than observed

5. **Residual patterns persist**:
   - Systematic structure remains
   - Robustness doesn't fix functional form

6. **No identifiable outliers**:
   - All standardized residuals similar
   - Robustness not needed

---

## Model 3: Penalized B-Splines

### I will abandon this model if:

1. **Regularization extreme** (τ_post < 0.1)
   - Strong penalty → nearly linear
   - Simple parametric sufficient

2. **No regularization needed** (τ_post > 5)
   - No penalty applied
   - Risk of overfitting

3. **Coefficients erratic**:
   - Non-smooth pattern in β_k
   - Suggests overfitting or poor knot placement

4. **Worse than parametric** (ΔELPD_LOO > 3 in favor of log)
   - Flexibility penalized
   - Simpler model preferred

5. **Computational instability**:
   - High correlation in β coefficients
   - ESS < 100
   - Poor mixing

6. **Visual overfitting**:
   - Function too wiggly
   - Worse residual patterns than log model

7. **Knot artifacts**:
   - Visible discontinuities at knot locations
   - Boundary effects

---

## Global Falsification Criteria (All Models)

### Evidence that ALL models are wrong:

1. **Systematic residual patterns** in all three models
   - Action: Consider changepoint, mixture, or transformation

2. **All have poor LOO diagnostics** (many Pareto-k > 0.7)
   - Action: Data quality issue or fundamental misspecification

3. **All posterior predictive checks fail**:
   - Action: Rethink data generation process completely

4. **Results driven by single point** (x=31.5 sensitivity test)
   - Action: Report with/without, consider data error

5. **Gap predictions wildly inconsistent** across models
   - Action: High uncertainty region, need new data

6. **Priors dominate posteriors** in all models
   - Action: Data insufficient or uninformative

---

## Stress Tests: Designed to Break Models

### Test 1: Remove Influential Point
**Action**: Refit without x=31.5 (Cook's D=0.84)
**Success**: Conclusions unchanged
**Failure**: Parameters shift dramatically → Point is driving results

### Test 2: Extrapolation Challenge
**Action**: Fit on x≤20, predict x>20
**Success**: Predictions match observed within 2σ
**Failure**: Predictions wildly off → Extrapolation unreliable

### Test 3: Replicate Variance Test
**Action**: Check if posterior predictive variance matches observed at 6 replicate x values
**Success**: Observed variance within 95% credible interval
**Failure**: Model underestimates or overestimates variance → Misspecified

### Test 4: Prior Sensitivity
**Action**: Refit with priors 2x wider
**Success**: Posterior conclusions unchanged
**Failure**: Results shift → Priors too informative or data weak

### Test 5: Synthetic Data Recovery
**Action**: Simulate from known log model, fit all three models
**Success**: True model recovered/detected
**Failure**: Models can't distinguish truth → Poor power with n=27

### Test 6: Cross-Validation Stability
**Action**: 5-fold CV, check prediction consistency
**Success**: Predictions stable across folds
**Failure**: High variance in predictions → Overfitting

---

## Decision Points: When to Pivot

### Pivot Point 1: After Initial Fits
**Trigger**: All Rhat > 1.05 or divergences > 10%
**Action**: Reparameterize, tighten priors, or reconsider model class

### Pivot Point 2: After LOO Comparison
**Trigger**: All ΔELPD < 2 (models tied)
**Action**: Choose simplest, report model uncertainty

### Pivot Point 3: After Posterior Predictive Checks
**Trigger**: All models fail PPC
**Action**: Try backup models (heteroscedastic, changepoint, quantile)

### Pivot Point 4: After Sensitivity Analysis
**Trigger**: x=31.5 drives all conclusions
**Action**: Report two analyses (with/without), warn about data dependence

### Pivot Point 5: After Stress Tests
**Trigger**: Models break under reasonable perturbations
**Action**: Reduce scope (don't extrapolate), increase uncertainty, collect more data

---

## Stopping Rules: When We're Done

### Success Criteria (Stop and Report):
1. One model clearly best (ΔELPD > 3) AND
2. Passes all diagnostic checks AND
3. Robust to sensitivity tests AND
4. Posterior predictive checks pass AND
5. Parameters interpretable and plausible

### Failure Criteria (Stop and Rethink):
1. No model passes diagnostics after tuning OR
2. All models fail posterior predictive checks OR
3. Results entirely driven by one point OR
4. Computational intractability (>6 hours per model) OR
5. Prior-posterior conflict in all models

### Uncertainty Criteria (Stop and Report Ambiguity):
1. Models tied (ΔELPD < 2) AND
2. All pass diagnostics BUT
3. Predictions differ substantially in gap region
→ Report model averaging or multiple scenarios

---

## Red Flags That Trigger Model Class Change

| Red Flag | Interpretation | Alternative Model |
|----------|---------------|-------------------|
| Residual patterns persist | Functional form wrong | Try changepoint, mixture |
| Heteroscedasticity extreme | Variance model needed | Add σ(x) function |
| Outliers not downweighted | Need heavier tails | Try quantile regression |
| Function non-monotonic | Regime changes | Try changepoint or piecewise |
| Gap uncertainty extreme | Need local model | Try local regression |
| All models overfit | n=27 too small | Report parametric with large CI |

---

## Communicating Failure Positively

### If GP Fails:
"The Gaussian Process model's lengthscale posterior suggests the data is well-described by a simpler parametric form. This validates the logarithmic model from EDA."

### If Robust-t Fails:
"The degrees of freedom posterior ν > 50 indicates no outliers are present. Standard Gaussian errors are appropriate, and the influential point x=31.5 is not aberrant."

### If Spline Fails:
"The regularization parameter τ → 0 indicates strong smoothing is optimal. This suggests the data does not require local flexibility beyond a simple parametric curve."

### If All Fail:
"None of the proposed flexible models substantially improve upon the logarithmic baseline, suggesting either: (1) the logarithmic form is adequate, or (2) with n=27, we lack power to detect deviations. We recommend the simplest model (log) with appropriately wide credible intervals."

---

## Final Note: Embracing Model Failure

**Model failure is scientific progress.**

- If GP simplifies to parametric → We learned the function is smooth
- If robust model finds no outliers → We learned errors are Gaussian
- If spline over-smooths → We learned flexibility unnecessary
- If all models fail → We learned we need different data or approach

**The goal is not to defend a model, but to discover the truth about the data.**

---

**End of Falsification Criteria Document**
