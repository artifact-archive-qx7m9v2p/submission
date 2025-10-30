# Improvement Priorities: From Experiment 1 to Phase 2

**Date:** 2025-10-29
**Current Model:** Negative Binomial Quadratic (Experiment 1)
**Status:** REJECTED - Temporal structure required
**Next Phase:** Phase 2 Temporal Models

---

## Priority 1: Add Temporal Correlation Structure (CRITICAL)

### Problem
- **Residual ACF(1) = 0.686** (threshold: 0.5)
- **Observed data ACF(1) = 0.944** (nearly perfect autocorrelation)
- Independence assumption fundamentally violated
- 89% of variance at time t predictable from time t-1

### Solution: Experiment 3 - AR(1) Negative Binomial

**Recommended model structure:**
```python
# Temporal extension of current model
μ[t] = exp(β₀ + β₁·year[t] + β₂·year²[t] + α[t])
α[t] ~ Normal(ρ·α[t-1], σ_α)  # AR(1) random effect
α[1] ~ Normal(0, σ_α/sqrt(1-ρ²))  # Stationary initialization
C[t] ~ NegBinomial(μ[t], φ)
```

**New parameters:**
- `ρ`: Autocorrelation coefficient (expect ≈ 0.7 based on residual ACF)
- `σ_α`: Innovation standard deviation for AR process
- `α[1:T]`: Temporal random effects (T=40)

**Priors:**
```python
ρ ~ Beta(8, 2)  # Favors ρ ∈ [0.5, 0.9], allows strong correlation
σ_α ~ HalfNormal(0.5)  # Innovation variance
```

**Why this will work:**
- AR(1) directly models lag-1 correlation = 0.686
- Preserves all successful elements (NegBin, quadratic trend)
- Adds only 2 parameters + 40 random effects
- Well-established model class

**Expected improvements:**
- Residual ACF(1) drops from 0.686 to <0.3
- Coverage normalizes from 100% to 90-98%
- ACF(1) test statistic p-value moves to healthy range
- ELPD improves by 10-20+ points
- Temporal wave pattern disappears from residuals

**Success criteria:**
1. Residual ACF(1) < 0.3 for all lags
2. Coverage ∈ [90%, 98%]
3. No extreme p-values for ACF test
4. LOO improvement > 10 points
5. Convergence maintained (R̂ < 1.01)

**If this doesn't work:**
- Check ACF(2), ACF(3) - may need AR(2) or AR(3)
- Consider MA component (ARMA models)
- Try state-space formulation (Experiment 4)

---

## Priority 2: Improve Coverage Calibration (IMPORTANT)

### Problem
- **95% coverage = 100%** (excessive, target: 90-98%)
- All observations fall within prediction intervals
- Model is too conservative, overestimates uncertainty
- Likely consequence of missing temporal structure

### Why This Matters
- Overly wide intervals reduce statistical power
- Hides poor mean predictions behind uncertainty
- Suggests model is uncertain about wrong things
- Indicates improper uncertainty quantification

### Expected Resolution
**Adding temporal structure should improve coverage naturally:**

1. **Temporal model reduces unexplained variance:**
   - Current model treats all variation as "noise"
   - AR model explains some variation via temporal correlation
   - Reduces residual variance → tighter intervals

2. **More accurate uncertainty estimates:**
   - Current intervals too wide because they ignore temporal structure
   - Temporal model properly partitions variance
   - Intervals reflect true predictive uncertainty

3. **Better calibration:**
   - Should achieve 90-98% coverage
   - Intervals appropriately sized
   - Some observations outside intervals (as expected)

**Validation:**
- After fitting AR model, recompute coverage rates
- Should see 2-10% of observations outside 95% intervals
- Check uniform distribution of LOO-PIT values

**If coverage still excessive after AR model:**
- May indicate overdispersion overestimated
- Try tighter prior on φ
- Consider alternative observation distributions

---

## Priority 3: Address Extreme Value Under-Generation (MODERATE)

### Problem
- **Observed maximum (272) at 99.4th percentile** (p = 0.994)
- Model rarely generates values as extreme as observed
- Cannot reproduce highest counts
- Risk assessment for extreme events unreliable

### Why This May Resolve with Temporal Model
**Extreme values likely cluster in time:**

1. **Temporal correlation creates persistence:**
   - High counts followed by high counts
   - Creates local "runs" of extreme values
   - Temporal model can generate these patterns

2. **Current model treats extremes as independent:**
   - Probability of hitting 272 is small at any single time point
   - But with temporal correlation, extremes occur together
   - AR model may naturally generate higher maximums

3. **Indirect evidence:**
   - Late time points (35-40) have largest residuals
   - Suggests extremes occur in connected sequences
   - AR structure should capture this clustering

**Validation after AR model:**
- Check if maximum test statistic p-value improves
- Should move from p=0.994 closer to healthy range [0.1, 0.9]
- Plot temporal trajectory of replications vs observed

**If problem persists:**
- May need heavier-tailed distribution (though NegBin is already heavy)
- Consider time-varying dispersion: φ[t] = f(μ[t])
- Check if mean function undershoots at late times (curvature issue)

---

## Priority 4: Refine Functional Form (LOW)

### Problem
- **U-shaped residual pattern vs fitted values**
- Some curvature misspecification remains
- R² = 0.883 is good but not perfect
- Largest residuals at late time points

### Potential Improvements
**This is low priority because:**
- Temporal correlation is much larger issue (ACF=0.686)
- R² = 0.883 already strong
- May resolve when temporal structure added

**If still needed after temporal model:**

**Option A: Exponential trend (simpler)**
```python
log(μ[t]) = β₀ + β₁·year[t]  # Linear in log-space = exponential growth
```
- Simpler (1 fewer parameter)
- May fit late acceleration better
- Test as Experiment 2 for comparison

**Option B: Piecewise trend**
```python
log(μ[t]) = β₀ + β₁·year[t] + β₂·max(0, year[t] - c)
```
- Allows changepoint at time c
- Based on EDA finding of 6× rate increase
- More complex but mechanistically interpretable

**Option C: Spline/Gaussian Process**
- Highly flexible
- May overfit with n=40
- Only if simpler approaches fail

**Recommendation:**
- Wait until after AR model fitted
- Check if residual pattern vs fitted improves
- Only revisit if systematic pattern remains

---

## Priority 5: Validate Distribution Shape (LOW)

### Problem
- **Skewness: p = 0.999** (observed less skewed than model predicts)
- **Kurtosis: p = 1.000** (observed flatter than model predicts)
- Marginal distribution shape slightly mismatched

### Why This is Low Priority
1. **Minor compared to temporal issues:**
   - Skewness/kurtosis are second-order moments
   - Temporal correlation is first-order structural issue

2. **May be artifact of temporal correlation:**
   - Independence assumption creates different marginal distribution
   - Temporal model may naturally fix shape

3. **Mean and variance fit well:**
   - Mean: p = 0.668
   - Variance: p = 0.910
   - First two moments correct, higher moments less critical

**If problem persists after temporal model:**

**Option A: Alternative observation distributions**
- Try zero-inflated negative binomial (if zeros problematic)
- Try generalized Poisson
- Try Conway-Maxwell-Poisson (flexible dispersion)

**Option B: Transformation**
- Model on square-root scale instead of log
- May alter shape properties
- Less interpretable

**Recommendation:**
- Ignore for now
- Reassess after Phase 2 models fitted
- Only pursue if critical for scientific conclusions

---

## Implementation Strategy

### Step 1: Fit Experiment 3 (AR Negative Binomial)

**Timeline:** Immediate next step

**Actions:**
1. Write Stan/PyMC model with AR(1) structure
2. Specify priors for ρ and σ_α
3. Run prior predictive check (ensure stationarity)
4. Fit to real data
5. Check convergence (expect some correlation in α parameters)

**Challenges:**
- AR models have more parameters → slower sampling
- May need longer chains or more thinning
- Initial state prior needs care for stationarity

**Resources needed:**
- Stan model: ~50 lines (extension of current)
- Sampling time: ~10-30 minutes (vs 2.5 min for Experiment 1)
- Memory: ~2-3× current (store α[t] trajectories)

### Step 2: Validate AR Model

**Timeline:** Immediately after Experiment 3 fitting

**Diagnostics to run:**
1. **Convergence:** R̂, ESS (may be lower due to AR structure)
2. **Posterior predictive check:** Full PPC suite
3. **Residual diagnostics:** Check ACF(1) < 0.3
4. **Coverage analysis:** Should be 90-98%
5. **LOO comparison:** vs Experiment 1 baseline

**Success indicators:**
- Residual ACF dramatically reduced
- Coverage normalized
- ELPD substantially improved
- No new systematic patterns

**Failure indicators:**
- Residual ACF still > 0.5 → need higher-order AR
- Convergence issues → reparameterize
- No ELPD improvement → temporal structure not helpful (unlikely)

### Step 3: Model Comparison and Selection

**Timeline:** After fitting Experiments 2-4

**Comparisons:**
1. **Experiment 1 vs 3:** Benefit of AR structure
2. **Experiment 2 vs 3:** Quadratic vs exponential + AR
3. **Experiment 3 vs 4:** AR vs state-space
4. **Best model vs all others:** Final selection

**Metrics:**
- LOO-ELPD (primary)
- Residual ACF (must be <0.3)
- Coverage calibration
- Parsimony (parameters vs fit)
- Interpretability

### Step 4: Final Inference

**Timeline:** After model selection

**Report:**
1. Selected model and justification
2. Parameter estimates with uncertainty
3. Scientific interpretation
4. Predictions with intervals
5. Model limitations

---

## Alternative Path: If AR(1) Insufficient

### Scenario: AR(1) reduces ACF(1) but not enough

**If residual ACF(1) after AR(1) is 0.3-0.5:**

**Diagnosis:** Need higher-order temporal structure

**Option A: AR(2) model**
```python
α[t] ~ Normal(ρ₁·α[t-1] + ρ₂·α[t-2], σ_α)
```
- Captures lag-2 correlation (currently ACF(2) = 0.423)
- Only 1 additional parameter
- Well-established extension

**Option B: ARMA(1,1) model**
```python
α[t] = ρ·α[t-1] + ε[t] + θ·ε[t-1]
ε[t] ~ Normal(0, σ_α)
```
- Moving average component
- Can model different ACF decay patterns
- More flexible than pure AR

**Option C: State-space model (Experiment 4)**
```python
# Latent state evolution
z[t] ~ Normal(z[t-1] + δ, σ_state)  # Random walk with drift
μ[t] = exp(β₀ + β₁·year[t] + β₂·year²[t] + z[t])
```
- Separate observation from state evolution
- Natural for smooth trends
- More parameters but interpretable

### Scenario: AR(1) doesn't converge

**If R̂ > 1.01 or ESS < 100:**

**Diagnosis:** Parameterization or prior issues

**Solutions:**
1. **Centered vs non-centered parameterization:**
   - Try: `α[t] = ρ·α[t-1] + σ_α·ε[t]` where `ε[t] ~ Normal(0,1)`
   - Can improve sampling efficiency

2. **Reparameterize correlation:**
   - Use logit(ρ) instead of ρ directly
   - Ensures ρ ∈ (-1, 1)

3. **Stronger priors:**
   - If ρ wanders, use more informative prior
   - Based on residual ACF = 0.686 estimate

4. **Increase adapt_delta:**
   - Try 0.99 instead of 0.95
   - Reduces divergences at cost of speed

---

## Success Metrics for Phase 2

### Must Achieve (Required)
1. ✓ Residual ACF(1) < 0.3
2. ✓ Coverage ∈ [85%, 98%]
3. ✓ R̂ < 1.01 for all parameters
4. ✓ ESS > 400 for main parameters

### Should Achieve (Expected)
5. ✓ ELPD improvement > 10 vs Experiment 1
6. ✓ No extreme p-values for ACF test
7. ✓ Residual patterns resolved
8. ✓ Maximum value p-value in [0.1, 0.9]

### Nice to Have (Desirable)
9. ○ Skewness/kurtosis p-values improve
10. ○ Mechanistic interpretation of correlation parameter
11. ○ One-step-ahead forecast accuracy improves
12. ○ Sampling efficiency remains acceptable

---

## What NOT to Do

### Don't Waste Time On:

**1. Adding polynomial terms to Experiment 1**
- Won't fix temporal correlation
- Makes model more complex without addressing root cause
- Temporal structure is priority, not functional form

**2. Trying many observation distributions**
- Negative Binomial is appropriate for overdispersion
- Temporal correlation is the issue, not marginal distribution
- Only revisit if Phase 2 models still show issues

**3. Overfitting with complex temporal models**
- Start simple: AR(1) first
- Only increase complexity if needed
- n=40 is small, avoid overparameterization

**4. Ignoring convergence issues**
- If temporal model doesn't converge, diagnose why
- Don't just run longer chains without understanding problem
- Reparameterization often better than more iterations

**5. Accepting mediocre fit**
- If AR(1) only reduces ACF(1) to 0.4, keep trying
- The trigger was >0.5, but goal is <0.3
- "Better than Experiment 1" is not enough

---

## Resource Planning

### Computational Resources

**Experiment 3 (AR model):**
- **Expected runtime:** 15-30 minutes
- **Memory:** ~2 GB
- **Chains:** 4 (parallel)
- **Samples:** 2000 per chain (may need 3000)

**Experiment 4 (State-space):**
- **Expected runtime:** 30-60 minutes
- **Memory:** ~3 GB
- **May need:** More sophisticated sampling (ADVI initialization)

### Human Resources

**Analyst time:**
- Model specification: 2 hours
- Prior predictive check: 1 hour
- Model fitting: 0.5 hours (mostly waiting)
- Diagnostics: 2 hours
- Comparison: 2 hours
- **Total: ~8 hours for Experiment 3**

### Decision Timeline

**Aggressive path:** 2 days
- Day 1: Fit Experiment 3, run diagnostics
- Day 2: If needed, fit Experiment 4 and compare

**Realistic path:** 3-5 days
- Allows for troubleshooting
- Thorough diagnostics
- Multiple model comparisons
- Sensitivity analyses

---

## Summary: The Path Forward

### Immediate Next Step
**Fit Experiment 3: AR(1) Negative Binomial Model**

This addresses the critical failure (temporal correlation) while preserving all successful elements from Experiment 1.

### Expected Outcome
Residual ACF(1) drops below 0.3, coverage normalizes, and model becomes adequate for scientific inference.

### Backup Plan
If AR(1) insufficient, try Experiment 4 (state-space) or higher-order AR models.

### Success Criteria
Model passes all checks with residual ACF < 0.3 and proper coverage calibration.

### Timeline
2-5 days to Phase 2 completion and final model selection.

---

**Document Date:** 2025-10-29
**Status:** Ready to proceed to Experiment 3
**Priority:** HIGH - Temporal correlation is critical issue
**Next Action:** Specify AR(1) model and run prior predictive check
