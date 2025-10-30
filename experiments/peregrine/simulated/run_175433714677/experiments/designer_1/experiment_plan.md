# Experiment Plan: Parsimonious Bayesian Approach

**Designer**: Agent 1 (Parsimony Track)

**Philosophy**: Simplicity, Interpretability, Computational Efficiency

**Date**: 2025-10-29

---

## Executive Summary

I propose a **parsimonious modeling strategy** that starts with the simplest adequate model and adds complexity only when evidence strongly demands it. Based on EDA findings showing severe overdispersion (Var/Mean ≈ 70), I will test 2 competing Negative Binomial models:

1. **Log-Linear** (3 parameters): Constant exponential growth
2. **Quadratic** (4 parameters): Accelerating growth

**Critical commitment**: I will **abandon** quadratic models if β₂'s 95% CI includes zero with |β₂| < 0.1. I will **abandon** linear models if LOO-CV shows ΔELPD > 4 favoring quadratic.

**Success criterion**: Find the simplest model that adequately explains the data, not the most complex one that slightly improves fit.

---

## Problem Formulation

### Research Question
What is the simplest Bayesian model that adequately captures:
1. **Overdispersion** (Var/Mean ≈ 70)
2. **Growth trend** (linear vs. accelerating)
3. **Uncertainty** (prediction intervals)

### Competing Hypotheses

**H1: Exponential Growth (Linear on log-scale)**
- **Process**: Single exponential growth mechanism
- **Prediction**: log(μ) = β₀ + β₁·year
- **Complexity**: 3 parameters
- **Interpretation**: Constant growth rate β₁ (exp(β₁) = 2.34 → 134% annual growth)

**H2: Accelerating Growth (Quadratic on log-scale)**
- **Process**: Compound growth or positive feedback
- **Prediction**: log(μ) = β₀ + β₁·year + β₂·year²
- **Complexity**: 4 parameters
- **Interpretation**: Time-varying growth rate = β₁ + 2β₂·year

### Falsification Framework

**I will abandon H1 (Linear) if**:
1. LOO-CV: ΔELPD > 4 compared to H2
2. Posterior predictive residuals show systematic U-shaped curvature
3. Mean absolute error in last 10 observations > 2× first 10 observations
4. Variance-to-mean ratio outside [50, 90] in posterior predictive checks

**I will abandon H2 (Quadratic) if**:
1. β₂ is not significant: 95% CI includes 0 AND |β₂| < 0.1
2. LOO-CV: ΔELPD < 2 compared to H1 (not worth added complexity)
3. Overfitting: >10% of observations have Pareto-k > 0.7
4. Computational issues: >0.5% divergent transitions

**I will reconsider EVERYTHING if**:
1. Prior-posterior conflict: Posteriors >3 SD from EDA estimates
2. Computational pathology: >5% divergences after tuning, ESS < 100
3. Extreme parameters: |β₀| > 8, |β₁| > 2, φ < 0.3 or φ > 10
4. Prediction failure: Posterior predictive variance < 20 or > 200 (EDA: 70)

---

## Model Specifications

### Model 1: Log-Linear Negative Binomial (BASELINE)

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)    for i = 1,...,40
log(μ[i]) = β₀ + β₁ × year[i]
```

**Priors** (weakly informative, centered at EDA estimates):
```
β₀ ~ Normal(4.3, 1.0)     # EDA: log(mean) at year=0 ≈ 4.3
β₁ ~ Normal(0.85, 0.5)    # EDA: growth rate ≈ 0.85
φ  ~ Exponential(0.667)   # EDA: φ ≈ 1.5, E[Exp(0.667)] = 1.5
```

**Theoretical Justification**:
- Simplest model: Only 3 parameters
- Interpretability: β₁ = constant growth rate
- EDA support: Linear fit R² = 0.92
- Scientific plausibility: Early-stage exponential growth (e.g., technology adoption, population growth)

**Expected Posteriors** (from EDA):
- β₀: 4.3 ± 0.15
- β₁: 0.85 ± 0.10 (exp(0.85) = 2.34 → 134% annual growth)
- φ: 1.5 ± 0.5

**Computational Properties**:
- Expected ESS: >1000 (simple geometry)
- Expected divergences: 0 (well-identified)
- Expected runtime: <10 seconds (4 chains × 1000 iterations)

---

### Model 2: Quadratic Negative Binomial

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)      # Same as Model 1
β₁ ~ Normal(0.85, 0.5)     # Same as Model 1
β₂ ~ Normal(0, 0.3)        # Centered at 0 (no prior bias)
φ  ~ Exponential(0.667)    # Same as Model 1
```

**Theoretical Justification**:
- EDA evidence: R² = 0.96 with quadratic, Chow test p < 0.000001
- Growth acceleration: 9.6× increase from early to late period
- Scientific plausibility: Network effects, compound growth, positive feedback
- Parsimony trade-off: Only +1 parameter vs. Model 1

**Expected Posteriors**:
- β₀: 4.3 ± 0.15
- β₁: 0.85 ± 0.10
- β₂: 0.3 ± 0.15 (positive = acceleration)
- φ: 1.5 ± 0.5

**Computational Properties**:
- Expected ESS: >500 (potential β₁-β₂ correlation)
- Expected divergences: <0.5% (mild multicollinearity)
- Expected runtime: <15 seconds

---

## Experimental Workflow

### Phase 1: Model 1 (ALWAYS)

**Step 1.1: Fit Model**
```python
# Stan sampling with 4 chains, 1000 warmup, 1000 sampling
fit1 = model1.sample(data=stan_data, chains=4, iter_warmup=1000,
                     iter_sampling=1000, adapt_delta=0.8)
```

**Step 1.2: Convergence Diagnostics**
- [ ] R̂ < 1.01 for all parameters
- [ ] ESS_bulk > 400, ESS_tail > 400
- [ ] Divergences < 0.5%
- [ ] No max treedepth hits
- [ ] BFMI > 0.3

**Step 1.3: Posterior Checks**
- Compare posteriors to EDA estimates (should be within 2 SD)
- Check for extreme values (reject if |β₀| > 8, |β₁| > 2, φ < 0.3 or > 10)

**Step 1.4: Posterior Predictive Checks**
- [ ] Var/Mean ratio: Observed ≈ 70, Predicted 95% CI should include 70
- [ ] Calibration: 50% PI contains ~50% obs, 90% PI contains ~90% obs
- [ ] Extreme values: Can model generate counts up to observed max (269)?
- [ ] Residuals: No systematic patterns vs. fitted or year

**Step 1.5: LOO-CV**
- Compute LOO-ELPD and standard error
- Check Pareto-k diagnostics (flag if >0.7)

**Decision Point 1**:
- If all checks pass: **Model 1 is adequate** → Consider stopping
- If residuals show curvature OR variance calibration fails: → Proceed to Phase 2
- If computational issues: → Reconsider model class

---

### Phase 2: Model 2 (CONDITIONAL)

**Trigger Criteria** (fit Model 2 only if ONE of these is true):
1. Model 1 residuals show systematic curvature (U-shape or inverted-U)
2. Model 1 fails variance calibration (Var/Mean outside [50, 90])
3. Model 1 shows poor late-period predictions (MAE_late > 2 × MAE_early)

**Step 2.1: Fit Model**
```python
# Increase adapt_delta if Model 1 showed any divergences
fit2 = model2.sample(data=stan_data, chains=4, iter_warmup=1000,
                     iter_sampling=1000, adapt_delta=0.9)
```

**Step 2.2: Convergence Diagnostics**
- Same criteria as Model 1
- Additionally check: β₁-β₂ correlation (<0.8 preferred)

**Step 2.3: Parameter Significance**
- **Critical**: Is β₂ significantly different from 0?
- Compute 95% credible interval for β₂
- If 95% CI includes 0 AND |β₂| < 0.1 → **Abandon Model 2**

**Step 2.4: Posterior Predictive Checks**
- Same checks as Model 1
- Compare to Model 1 performance

**Step 2.5: Model Comparison (LOO-CV)**
```python
comp = az.compare({'Model_1': idata1, 'Model_2': idata2})
```

**Decision Rules**:
- **ΔELPD < 2**: Models equivalent → **Choose Model 1** (parsimony)
- **2 < ΔELPD < 4**: Weak evidence → Consider interpretability, choose simpler if equal value
- **ΔELPD > 4**: Strong evidence → **Choose Model 2**

**Decision Point 2**:
- If Model 2 significantly better: **Accept Model 2**
- If models equivalent: **Accept Model 1** (parsimony)
- If Model 2 fails diagnostics: **Accept Model 1** (robustness)

---

### Phase 3: Stress Tests (FINAL MODEL)

Apply to whichever model was selected:

**Test 1: Extrapolation Stability**
- Generate predictions for year ∈ [-2, 2] (beyond data range)
- Check for explosive growth (>10,000 counts)
- Verify uncertainty increases appropriately

**Test 2: Variance Recovery**
- Compute Var/Mean ratio for each posterior draw
- Verify 95% CI includes observed ratio (70)
- Plot distribution vs. observed

**Test 3: Tail Behavior**
- Check if max(y_rep) across all draws includes observed max (269)
- Assess if model systematically under-predicts high values

**Test 4: Residual Randomness**
- Plot residuals vs. fitted (should be random cloud)
- Plot residuals vs. year (should be random)
- Runs test for randomness (p > 0.05)

**Test 5: Prior Sensitivity**
- Re-fit with priors doubled in width
- Check if posteriors change substantially (>20% shift)
- If sensitive: Report sensitivity, potentially use wider priors

---

## Red Flags & Decision Points

### STOP and Reconsider Everything If:

**Red Flag 1: Prior-Posterior Collapse**
- Posteriors are essentially identical to priors
- **Diagnosis**: Model not identified by data
- **Action**: Check Stan code, verify data is loaded correctly

**Red Flag 2: Prior-Posterior Conflict**
- Posteriors >3 SD from priors (and priors were data-informed)
- **Diagnosis**: Model misspecification or incorrect priors
- **Action**: Re-examine EDA, check for data errors

**Red Flag 3: Computational Pathology**
- >5% divergences after adapt_delta=0.99
- ESS < 100 for any parameter
- R̂ > 1.05
- **Diagnosis**: Model geometry issues, possibly misspecified
- **Action**: Consider reparameterization or different model class

**Red Flag 4: Extreme Parameter Values**
- |β₀| > 8 (implies mean counts <0.3 or >3000 at year=0)
- |β₁| > 2 (implies >600% or <14% annual growth)
- φ < 0.3 or φ > 10 (extreme overdispersion)
- **Diagnosis**: Model misspecification or data issues
- **Action**: Check for outliers, measurement errors, or try different likelihood

**Red Flag 5: Prediction Failure**
- Posterior predictive variance < 20 or > 200 (observed: 70)
- >20% of observations outside 90% prediction intervals
- **Diagnosis**: Model not capturing data structure
- **Action**: Consider time-varying dispersion, mixture models, or other approaches

---

## Alternative Approaches (If All Models Fail)

### Escape Route 1: Heteroscedastic Model
**Trigger**: Variance calibration fails consistently

**Model**:
```
C[i] ~ NegativeBinomial(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i]
log(φ[i]) = γ₀ + γ₁ × year[i]
```

**Justification**: EDA shows Var/Mean varies 20× across periods

### Escape Route 2: Changepoint Model
**Trigger**: Residuals show clear regime shift

**Model**:
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × I(year[i] > τ) + β₃ × (year[i] - τ) × I(year[i] > τ)
```

**Justification**: Chow test significant at year ≈ -0.21

### Escape Route 3: Student-t Observation Model
**Trigger**: Posterior predictive checks show model can't generate extreme values

**Model**: Replace NegBin likelihood with hierarchical Student-t

### Escape Route 4: Gaussian Process
**Trigger**: Systematic residual patterns that can't be captured by polynomial

**Model**: GP with squared exponential kernel (non-PPL approach, only as last resort)

### Escape Route 5: Data Re-examination
**Trigger**: All models fail basic checks

**Action**:
- Check for recording errors
- Look for batch effects or temporal clusters
- Search for hidden covariates (day of week, seasonal, etc.)
- Consult domain expert

---

## Model Comparison Criteria

### Primary: LOO-CV (Leave-One-Out Cross-Validation)

**Metric**: ELPD (Expected Log Pointwise Predictive Density)

**Decision Rules**:
| ΔELPD | SE ratio | Decision |
|-------|----------|----------|
| <2 | Any | Models equivalent → Choose simpler |
| 2-4 | <2 | Weak evidence → Consider interpretability |
| 2-4 | >2 | Moderate evidence → Favor better model |
| >4 | Any | Strong evidence → Choose better model |

**Diagnostic**: Pareto-k values
- k < 0.5: Good
- 0.5 < k < 0.7: Okay
- k > 0.7: Problematic (observation is influential)

**Action if many k > 0.7**: Investigate influential observations, consider refit with K-fold CV

### Secondary: Interpretability

**Questions**:
1. Can we explain parameters to domain experts?
2. Do parameters align with scientific theory?
3. Can we use model for intervention planning?

**Preference**: Simpler model with clear interpretation >> Complex model with slightly better fit

### Tertiary: Computational Efficiency

**Considerations**:
1. Warmup time (faster = better for production)
2. ESS per second (efficiency of sampling)
3. Robustness to hyperparameter tuning

**Preference**: Model that "just works" >> Model requiring extensive tuning

---

## Success Criteria

### Model is SUCCESSFUL if:
1. **Converges**: R̂ < 1.01, ESS > 400, <0.5% divergences
2. **Calibrates**: 50% PI ≈ 50% coverage, 90% PI ≈ 90% coverage
3. **Recovers variance**: Posterior predictive Var/Mean = 60-80 (observed: 70)
4. **Interpretable**: Parameters align with EDA (within 2 SD)
5. **Robust**: Predictions stable under slight data perturbations

### Model is ADEQUATE if:
- All success criteria met EXCEPT may need complexity for calibration
- Decision: Use Bayesian Occam's Razor (prefer simpler if ΔELPD < 2)

### Model FAILS if:
- Any convergence diagnostic fails
- Posterior predictive checks show systematic bias
- Parameters are extreme or scientifically implausible

---

## Stopping Rules

### Stop Exploring When:

**Success**:
- A model meets all success criteria
- No obvious failures in stress tests

**Diminishing Returns**:
- ΔELPD < 1 between last two models tested
- Computational effort exceeds interpretability gain

**Exhausted Options**:
- All proposed models tested (Models 1-2, possibly heteroscedastic)
- No model meets success criteria → Escalate

**Resource Limits**:
- Computation time exceeds available resources
- Time better spent on data collection or domain consultation

### Escalate When:

1. **All models fail** → Consult domain expert, review data quality
2. **Models disagree** → May indicate fundamental uncertainty, report range
3. **Computational issues persist** → Consider simpler model class or non-Bayesian baseline

---

## Implementation Details

### Stan Parameterization

**Likelihood function**: `neg_binomial_2_log`
- **Parameter 1**: `mu` (on log-scale)
- **Parameter 2**: `phi` (overdispersion, larger = less overdispersion)
- **Variance**: Var(Y) = μ + μ²/φ

**Why this parameterization?**
- Directly models log(μ), matching GLM framework
- More stable numerically than alternative parameterizations
- Matches EDA analysis (φ ≈ 1.5)

### Sampling Configuration

**Initial run**:
```
chains: 4
iter_warmup: 1000
iter_sampling: 1000
adapt_delta: 0.8
max_treedepth: 10
seed: 12345
```

**If divergences occur** (>0.5%):
```
adapt_delta: 0.95  (more conservative step size)
iter_warmup: 2000  (longer adaptation)
```

**If still divergences** (>1%):
```
adapt_delta: 0.99
iter_warmup: 5000
# Consider reparameterization or model change
```

### Generated Quantities

**Required**:
1. `log_lik[i]`: Pointwise log-likelihood for LOO-CV
2. `y_rep[i]`: Posterior predictive samples for PPC

**Optional but helpful**:
3. `mu[i]`: Expected counts on natural scale (for interpretation)
4. `growth_rate`: exp(β₁) for Model 1, or time-varying for Model 2

---

## Timeline & Resources

### Computational Resources
- **CPU**: Standard laptop (4 cores) sufficient
- **Memory**: <2 GB required
- **Storage**: <100 MB for all outputs

### Expected Timeline

| Phase | Task | Time |
|-------|------|------|
| Setup | Data loading, model compilation | 5 min |
| Phase 1 | Fit Model 1, diagnostics, PPC | 15 min |
| Phase 2 | Fit Model 2, comparison | 15 min |
| Phase 3 | Stress tests, reporting | 15 min |
| **Total** | | **~50 min** |

### Outputs

**Files**:
1. `model1_linear.stan` (provided)
2. `model2_quadratic.stan` (provided)
3. `fit_models.py` (provided)
4. `results/model_1_linear_idata.nc` (InferenceData object)
5. `results/model_2_quadratic_idata.nc` (if fitted)
6. `results/model_1_linear_ppc.png` (posterior predictive check plots)
7. `results/model_2_quadratic_ppc.png` (if fitted)
8. `results/model_comparison_report.txt` (summary)

---

## Integration with Other Designers

This is **Designer 1: Parsimony Track**.

### Collaboration Strategy

**If other designers propose complex models**:
1. Use these models as **baselines** for comparison
2. Ask: "Does your model improve ELPD by >2 compared to these?"
3. Ask: "Is your model more interpretable?"
4. Ask: "Does your model reveal insights we're missing?"

**If all designers converge**:
- Strong evidence for that model class
- Report consensus

**If designers diverge**:
- Report model uncertainty
- Use stacking or model averaging
- May indicate fundamental ambiguity in data

### Communication

**What I'll provide to other designers**:
1. Baseline LOO-ELPD values (for comparison)
2. Posterior predictive Var/Mean ratios (for calibration target)
3. Convergence diagnostics (computational benchmarks)

**What I need from other designers**:
1. LOO-ELPD for their models (for comparison)
2. Explanation of added complexity (justify >4 parameters)
3. Insights not captured by simple models

---

## Philosophy & Mindset

### Parsimony ≠ Simplicity for Its Own Sake

**Parsimony means**:
- Fewer parameters → Easier to interpret, faster to fit, less prone to overfitting
- But NOT at the cost of predictive adequacy

**When to add complexity**:
- Strong evidence (ΔELPD > 4)
- Scientific justification (captures known mechanism)
- Reveals new insights (not just better fit)

### Falsification > Confirmation

**Bad approach**: "Which model fits best?"
**Good approach**: "Which models can I reject?"

**Mindset**:
- Assume models are wrong until proven adequate
- Look for failures, not successes
- Plan escape routes before fitting

### Model Selection is Learning

**Successful outcomes**:
1. **Model 1 adequate**: Exponential growth is sufficient → Simple story
2. **Model 2 better**: Acceleration is real → More complex story
3. **Both fail**: Data structure more complex than anticipated → Learn and pivot

**Unsuccessful outcome**:
- Forcing a model to work when diagnostics scream "STOP"

---

## Reporting Plan

### Final Report Structure

1. **Executive Summary**
   - Which model was selected and why
   - Key parameters and interpretation
   - Uncertainty quantification

2. **Model Comparison**
   - LOO-CV results table
   - Decision rationale (parsimony vs. fit trade-off)

3. **Diagnostics**
   - Convergence: R̂, ESS, divergences
   - Posterior predictive checks: Calibration, variance recovery
   - Stress tests: Extrapolation, residuals

4. **Parameter Interpretation**
   - Posterior means, medians, 95% CIs
   - Comparison to EDA estimates
   - Scientific interpretation (growth rates, etc.)

5. **Limitations**
   - What the model doesn't capture
   - Assumptions that may not hold
   - Uncertainty that remains

6. **Recommendations**
   - Can this model be used for decision-making?
   - What additional data would help?
   - What questions remain unanswered?

### Visualizations

1. **Observed vs. Predicted** (with 90% PI)
2. **Residuals vs. Fitted**
3. **Posterior Predictive Distribution** (histogram overlay)
4. **Variance-Mean Recovery** (posterior vs. observed)
5. **Parameter Posteriors** (forest plot)
6. **LOO Comparison** (if multiple models)

---

## Key Takeaways

1. **Start simple**: Model 1 (3 parameters) is the baseline
2. **Add complexity judiciously**: Model 2 only if evidence supports it
3. **Falsify, don't confirm**: Each model has explicit rejection criteria
4. **Computation is diagnostic**: Divergences, ESS indicate problems
5. **Parsimony is a feature**: Simpler models are more interpretable and robust

**Final commitment**: I will report the simplest model that adequately explains the data, even if a more complex model has slightly better fit. ΔELPD < 2 → choose simple.

---

## References

- **EDA Report**: `/workspace/eda/eda_report.md`
- **Model Specifications**: `/workspace/experiments/designer_1/model_specifications.md`
- **Stan Manual**: https://mc-stan.org/docs/
- **ArviZ Documentation**: https://arviz-devs.github.io/arviz/

---

**End of Experiment Plan**
