# Improvement Priorities for Experiment 2
## AR(1) Log-Normal with Regime-Switching

**Date**: 2025-10-30
**Current Model Status**: CONDITIONAL ACCEPT
**Revision Needed**: YES - AR(2) recommended for publication-quality analysis

---

## Overview

The AR(1) model has been conditionally accepted as substantially better than Experiment 1, but residual ACF=0.549 indicates higher-order temporal structure remains. This document outlines prioritized improvements for Experiment 3.

**Key Insight**: The model's limitation is **specific and addressable** - it captures lag-1 dependence but misses lag-2 effects. We're not starting over; we're building on success.

---

## Priority 1: AR(2) Structure (HIGHEST PRIORITY)

**Objective**: Capture higher-order temporal dependence to reduce residual ACF below 0.3

### Model Specification

**Current AR(1)**:
```
mu[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2 + phi * epsilon[t-1]
epsilon[t] = log(C[t]) - (alpha + beta_1 * year[t] + beta_2 * year[t]^2)
```

**Proposed AR(2)**:
```
mu[t] = alpha + beta_1 * year[t] + phi_1 * epsilon[t-1] + phi_2 * epsilon[t-2]
epsilon[t] = log(C[t]) - (alpha + beta_1 * year[t])
```

**Key Changes**:
1. Add phi_2 parameter for lag-2 dependence
2. Drop beta_2 (quadratic term weakly identified)
3. Rename phi to phi_1 for clarity

### Parameter Priors

**phi_1 (lag-1 coefficient)**:
```
phi_1_raw ~ Beta(20, 2)
phi_1 = 0.95 * phi_1_raw
```
- Median: ~0.87 (same as current AR(1))
- Range: (0, 0.95)
- Rationale: Reuse successful prior from Experiment 2

**phi_2 (lag-2 coefficient)**:
```
phi_2_raw ~ Beta(5, 5)
phi_2 = 0.5 * (phi_2_raw - 0.5)  # Scale to (-0.25, 0.25)
```
- Median: ~0.0 (weakly informative, centered at 0)
- Range: (-0.25, 0.25)
- Rationale: Exploratory prior allowing positive or negative lag-2 effects

**Stationarity Constraint**:
```
# In PyMC, add constraint:
pm.Potential("stationarity", pm.math.switch(phi_1 + phi_2 < 1, 0, -np.inf))
```
- Ensures AR(2) process is stationary
- Prevents explosive behavior

**Other Priors** (unchanged from Experiment 2):
```
alpha ~ Normal(4.3, 0.5)
beta_1 ~ Normal(0.86, 0.15)
sigma_regime[1:3] ~ HalfNormal(0, 0.5)
```

### AR(2) Initialization

**For stationarity**, epsilon[0] and epsilon[1] must be drawn from the stationary distribution.

**Stationary variance for AR(2)**:
```
var_stationary = sigma^2 / (1 - phi_1^2 - phi_2^2 - 2*phi_1*phi_2/(1-phi_2))
```

**Implementation**:
```python
# Initialize epsilon[0] and epsilon[1] from stationary distribution
sigma_init_sq = sigma_regime[0]**2 / (1 - phi_1**2 - phi_2**2 - 2*phi_1*phi_2/(1-phi_2))
sigma_init = pm.math.sqrt(sigma_init_sq)

epsilon[0] ~ Normal(0, sigma_init)
epsilon[1] ~ Normal(phi_1 * epsilon[0], sigma_regime[1])

# Then AR(2) for t >= 2
for t in range(2, T):
    mu_trend[t] = alpha + beta_1 * year[t]
    mu[t] = mu_trend[t] + phi_1 * epsilon[t-1] + phi_2 * epsilon[t-2]
    log_C[t] ~ Normal(mu[t], sigma_regime[regime[t]])
    epsilon[t] = log_C[t] - mu_trend[t]
```

### Expected Improvements

**Residual ACF**:
- Current: 0.549
- Expected: < 0.3 (target threshold)
- Optimistic: < 0.2 (excellent)

**Fit Metrics**:
- MAE: Marginal improvement (13.99 → ~13.5)
- RMSE: Marginal improvement (20.12 → ~19.5)
- Bayesian R²: Slight improvement (0.952 → ~0.955)

**Posterior Predictive Checks**:
- ACF test: Should remain PASS (p > 0.05)
- All test statistics: Should remain PASS
- Residual independence: Should now PASS

**LOO-CV**:
- ΔELPD vs AR(1): +2 to +5 (moderate improvement)
- ΔELPD vs Exp 1: > +15 (substantial)

### Implementation Steps

**1. Specify Model in PyMC**:
- Create `/workspace/experiments/experiment_3/model_ar2.py`
- Implement AR(2) structure with stationarity constraint
- Use vectorized operations for efficiency

**2. Prior Predictive Check**:
- Generate 1,000 prior draws
- Verify ACF distribution still covers observed (~0.975)
- Check stationarity constraint is satisfied
- Target: >15% draws in plausible range [10, 500]

**3. Simulation-Based Validation**:
- Generate synthetic data with known AR(2) parameters
- Fit model and verify parameter recovery
- Check: Can we recover both phi_1 and phi_2?
- Watch for: Identifiability issues with N=40

**4. Posterior Inference**:
- Fit to real data
- Check convergence (R-hat < 1.01, ESS > 400)
- Compute residual ACF (target: < 0.3)
- Compare to AR(1) via LOO-CV

**5. Posterior Predictive Check**:
- Generate 1,000 replications
- Test autocorrelation (should PASS)
- Test residual independence (should now PASS)
- All test statistics should remain PASS

### Risks and Mitigation

**Risk 1: Identifiability with N=40**
- With 6 parameters (alpha, beta_1, phi_1, phi_2, sigma_1, sigma_2, sigma_3) and N=40, may have identification issues
- **Mitigation**: Use informative priors, check posterior correlation matrix, consider simplifying (drop regime structure if needed)

**Risk 2: Stationarity Constraint Complicates Sampling**
- Hard constraint may cause sampling difficulties
- **Mitigation**: Start with soft penalty, then add hard constraint if needed
- Alternative: Reparameterize in terms of partial autocorrelations (guarantees stationarity)

**Risk 3: AR(2) May Not Be Sufficient**
- Residual ACF may remain > 0.3 even with AR(2)
- **Mitigation**: Have AR(3) and state-space models as fallback plans
- Acceptance criterion: If AR(2) reduces ACF by >50%, consider success even if > 0.3

**Risk 4: Overfitting with Small Sample**
- Adding parameters may overfit to noise
- **Mitigation**: LOO-CV will detect overfitting, use conservative priors, compare to AR(1)

### Success Criteria

**Minimum Acceptable**:
- Residual ACF < 0.4 (27% improvement over AR(1))
- No convergence issues (R-hat < 1.01)
- LOO-ELPD improvement > 2 over AR(1)

**Target**:
- Residual ACF < 0.3 (45% improvement)
- All PPC test statistics PASS
- LOO-ELPD improvement > 5 over AR(1)

**Excellent**:
- Residual ACF < 0.2 (64% improvement)
- Residual diagnostics show no patterns
- LOO-ELPD improvement > 10 over AR(1)

**Decision Rules**:
- If minimum met: ACCEPT with caveats
- If target met: ACCEPT as final model
- If excellent met: ACCEPT enthusiastically, publish

---

## Priority 2: Simplify Trend Structure (MEDIUM PRIORITY)

**Objective**: Reduce overparameterization by dropping weakly identified quadratic term

### Current Finding

**beta_2 = 0.015 ± 0.125**:
- 95% credible interval includes 0: [-0.21, 0.26]
- Posterior mass centered very close to 0
- Adds complexity without clear benefit

### Proposed Change

**From quadratic**:
```
mu_trend[t] = alpha + beta_1 * year[t] + beta_2 * year[t]^2
```

**To linear**:
```
mu_trend[t] = alpha + beta_1 * year[t]
```

### Justification

1. **Parsimony**: Reduces parameter count from 7 to 6
2. **Identifiability**: beta_1 and phi_1 are partially confounded; removing beta_2 helps
3. **EDA alignment**: EDA showed linear trend on log-scale (R²=0.937)
4. **Interpretation**: Simpler to communicate (constant growth rate on log-scale)

### Expected Impact

- **Minimal fit loss**: beta_2 ≈ 0 means little predictive contribution
- **Improved identifiability**: Fewer parameters to estimate
- **Faster convergence**: Simpler model, less posterior correlation
- **Clearer interpretation**: Single growth rate parameter

### Implementation

**Already included in Priority 1 (AR(2))** - the proposed AR(2) specification drops beta_2.

**If keeping AR(1)** (not recommended), could update to:
```
mu[t] = alpha + beta_1 * year[t] + phi * epsilon[t-1]
```

### Validation

- Compare LOO-CV: AR(1) linear vs AR(1) quadratic
- Expected: Minimal difference (ΔELPD < 2)
- If quadratic better: Retain beta_2 despite weak identification

---

## Priority 3: Regime-Dependent AR Coefficients (ALTERNATIVE TO AR(2))

**Objective**: Allow autocorrelation to vary by regime instead of adding lag-2

### Rationale

**Current finding**: sigma_1 > sigma_2 > sigma_3 (variance decreases over time)

**Hypothesis**: Autocorrelation may also vary by regime:
- Early period: High variance, low autocorrelation (more volatile)
- Middle period: Moderate variance, high autocorrelation (persistent)
- Late period: Low variance, moderate autocorrelation (stable)

**Advantage over AR(2)**: Same parameter count (3 phi vs 1 phi + 1 phi_2)

### Model Specification

```
phi[t] = phi_regime[regime[t]]
mu[t] = alpha + beta_1 * year[t] + phi[t] * epsilon[t-1]

phi_regime[1] ~ Beta(15, 3) scaled to (0, 0.95)  # Early
phi_regime[2] ~ Beta(20, 2) scaled to (0, 0.95)  # Middle (strongest)
phi_regime[3] ~ Beta(15, 3) scaled to (0, 0.95)  # Late
```

### Expected Benefits

- Captures regime-specific dynamics
- Same parsimony as AR(1) (6 total parameters)
- May explain residual ACF if due to regime transitions
- More interpretable than AR(2) for some audiences

### Expected Limitations

- May not reduce residual ACF as much as AR(2)
- Assumes AR(1) sufficient within regimes
- More complex than homogeneous AR(2)

### Decision Criterion

**When to use this instead of AR(2)**:
- If AR(2) fails (residual ACF still > 0.3)
- If AR(2) has identifiability issues
- If regime transitions are scientifically meaningful

**Otherwise**: AR(2) is simpler and more standard

---

## Priority 4: State-Space Model (FALLBACK IF AR(2) FAILS)

**Objective**: Allow time-varying growth rates and flexible correlation

### Rationale

**If AR(2) and regime-dependent AR both fail**:
- Data may have non-stationary dynamics
- Growth rate may evolve over time
- Need more flexible temporal structure

### Model Specification

**Local Level + Local Trend**:
```
# State equations
mu[1] ~ Normal(4.3, 0.5)
growth[1] ~ Normal(0.86, 0.15)

mu[t] = mu[t-1] + growth[t-1] + nu[t]
growth[t] = phi * growth[t-1] + omega[t]

# Observation equation
log(C[t]) ~ Normal(mu[t], sigma_regime[regime[t]])

# Process noise
nu[t] ~ Normal(0, sigma_level)
omega[t] ~ Normal(0, sigma_growth)
```

### Priors

```
phi ~ Beta(20, 2) scaled to (0, 0.95)      # AR on growth rate
sigma_level ~ HalfNormal(0, 0.2)           # Level innovation
sigma_growth ~ HalfNormal(0, 0.1)          # Growth innovation
sigma_regime[1:3] ~ HalfNormal(0, 0.5)     # Observation noise
```

### Expected Benefits

- Captures time-varying growth rates
- Separates process and observation noise
- Can endogenously detect regime changes
- Very flexible temporal structure

### Expected Costs

- 3 additional parameters (phi, sigma_level, sigma_growth)
- More complex to implement and interpret
- May overfit with N=40
- Longer runtime (~5-10 minutes)

### Decision Criterion

**Only use if**:
- AR(2) residual ACF still > 0.3
- Regime-dependent AR insufficient
- Evidence of time-varying dynamics in residuals
- N ≥ 50 for adequate power (may need to wait for more data)

---

## Priority 5: Alternative Likelihoods (LOW PRIORITY)

**Objective**: Address occasional extreme predictions from log-normal

### Current Issue

**Replicated maxima**: 389 ± 207 (observed: 269)
- Bayesian p-value = 0.524 (acceptable)
- Log-normal has heavier tails than data
- Not a critical problem (p > 0.5) but worth monitoring

### Option A: Student-t Errors

**Replace**:
```
log(C[t]) ~ Normal(mu[t], sigma_regime[regime[t]])
```

**With**:
```
log(C[t]) ~ StudentT(nu, mu[t], sigma_regime[regime[t]])
nu ~ Gamma(2, 0.1)  # Degrees of freedom
```

**Benefit**: Robust to outliers, lighter tails than log-normal if nu > 2

**Cost**: One additional parameter, more complex sampling

### Option B: Truncated Normal

**Replace**:
```
log(C[t]) ~ Normal(mu[t], sigma_regime[regime[t]])
```

**With**:
```
log(C[t]) ~ TruncatedNormal(mu[t], sigma_regime[regime[t]], lower=log(1), upper=log(1000))
```

**Benefit**: Caps extreme values, matches data range

**Cost**: Introduces bias if truncation is strong, less principled

### Decision Criterion

**Only pursue if**:
- AR(2) still generates p < 0.1 on maximum value test
- Extreme predictions cause practical problems
- Alternative likelihoods improve LOO-CV

**Otherwise**: Log-normal is adequate (p=0.524 acceptable)

---

## Implementation Timeline

### Phase 1: AR(2) Core Implementation (Highest Priority)
**Timeline**: 3-4 hours

1. Specify AR(2) model in PyMC (1 hour)
2. Prior predictive check (30 min)
3. Simulation-based validation (1 hour)
4. Posterior inference on real data (30 min)
5. Posterior predictive check (30 min)
6. Model comparison via LOO-CV (30 min)

**Deliverables**:
- `/workspace/experiments/experiment_3/model_ar2.py`
- Full validation suite (prior/SBC/posterior/PPC)
- LOO comparison to AR(1) and Exp 1
- Decision: ACCEPT, REVISE, or REJECT AR(2)

### Phase 2: Simplification (If AR(2) Succeeds)
**Timeline**: 30 min

1. Confirm beta_2 still ≈ 0 in AR(2) model
2. If so, document that linear trend is sufficient
3. Update final model specification

**Deliverable**: Simplified AR(2) specification for publication

### Phase 3: Alternative Approaches (If AR(2) Fails)
**Timeline**: 4-6 hours per alternative

**If residual ACF still > 0.3 after AR(2)**:

**Option A**: Regime-dependent AR (4 hours)
- Implement and validate as per Priority 3
- Compare to homogeneous AR(2)

**Option B**: State-space model (6 hours)
- Implement local level + local trend
- May require extended sampling time

**Decision criterion**: Try one alternative, then assess whether to continue or accept limitations

### Phase 4: Documentation and Comparison (Final)
**Timeline**: 2 hours

1. Comprehensive comparison table (Exp 1 vs Exp 2 vs Exp 3)
2. Final model critique for accepted model
3. Adequacy assessment (Phase 5 of workflow)
4. Recommendations for future work

---

## Success Metrics for Each Priority

### Priority 1 (AR(2)): Success Defined As

**Minimum**:
- Residual ACF < 0.4 (vs current 0.549)
- R-hat < 1.01, ESS > 400
- LOO-ELPD better than AR(1) by >2

**Target**:
- Residual ACF < 0.3 (meets falsification criterion)
- All PPC test statistics pass
- LOO-ELPD better than AR(1) by >5

**Excellent**:
- Residual ACF < 0.2
- Residual plots show no patterns
- LOO-ELPD better than AR(1) by >10

### Priority 2 (Simplify): Success Defined As

- LOO-ELPD difference < 2 (minimal loss from dropping beta_2)
- Clearer interpretation of growth rate
- Faster convergence (ESS increases)

### Priority 3 (Regime-AR): Success Defined As

- Residual ACF < 0.3
- Regime-specific phi interpretable
- LOO-ELPD competitive with AR(2)

### Priority 4 (State-Space): Success Defined As

- Residual ACF < 0.2
- Time-varying growth rates meaningful
- LOO-ELPD substantially better than AR(2) (>10)

---

## Recommended Path Forward

**Step 1**: Implement Priority 1 (AR(2)) as Experiment 3
- Most likely to succeed
- Direct response to identified limitation
- Standard approach in time series

**Step 2**: If AR(2) succeeds (residual ACF < 0.3)
- Apply Priority 2 (simplify by dropping beta_2)
- ACCEPT final model
- Document as publication-quality

**Step 3**: If AR(2) partially succeeds (ACF 0.3-0.4)
- CONDITIONALLY ACCEPT with caveats
- Document that data may have AR(3) or long-memory
- Recommend future work with larger N

**Step 4**: If AR(2) fails (ACF > 0.4)
- Try Priority 3 (regime-dependent AR)
- If that fails, consider Priority 4 (state-space)
- If all fail, accept that data complexity exceeds modeling capacity with N=40

**Step 5**: Complete workflow
- Finalize model comparison (Exp 1 vs 2 vs 3)
- Adequacy assessment (Phase 5)
- Final recommendations

---

## Not Recommended

**Do NOT**:

1. **Increase prior informativeness** beyond current levels
   - Risk of prior-data conflict
   - Current priors are well-justified
   - Posterior should dominate with N=40

2. **Add more regimes** (4+ periods)
   - Already at limit with 3 regimes and N=40
   - Each regime has ~13 observations
   - Identifiability would suffer

3. **Switch to completely different model class** (e.g., GP) before trying AR(2)
   - AR(2) is simpler and more standard
   - GP may overfit with N=40
   - Save Tier 2 models for if AR extensions fail

4. **Abandon log-scale** and return to count scale
   - Log-scale working well (R²=0.952)
   - Count scale failed in Exp 1
   - No evidence log-scale is problematic

5. **Combine all extensions** (AR(2) + regime-phi + state-space)
   - Too many parameters for N=40
   - Overparameterization certain
   - Violates parsimony principle

---

## Summary: Clear Path Forward

**The diagnosis is clear**: AR(1) captures lag-1 dependence but misses lag-2 structure.

**The treatment is obvious**: Add lag-2 term (AR(2)).

**The prognosis is good**: Expected residual ACF < 0.3, achieving target.

**The timeline is short**: 3-4 hours to implement and validate.

**The decision is straightforward**: If AR(2) works, accept it. If not, try alternatives.

This is exactly how iterative model building should work: each model reveals what the next model should address.

---

**Document prepared by**: Model Criticism Specialist
**Date**: 2025-10-30
**Status**: Ready for implementation as Experiment 3
**Recommended next action**: Begin AR(2) implementation immediately
