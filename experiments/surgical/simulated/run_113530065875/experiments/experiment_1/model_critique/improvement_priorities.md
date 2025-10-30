# Improvement Priorities for Experiment 1: Hierarchical Binomial Model

**Date**: 2025-10-30
**Model**: Hierarchical Binomial (Logit-Normal, Non-Centered)
**Status**: CONDITIONAL ACCEPT (see decision.md)
**Purpose**: Recommended actions to address identified limitations

---

## Decision Context

The model is **CONDITIONALLY ACCEPTED** for research use. These improvement priorities are **optional recommendations** for strengthening the analysis, not required fixes. The model is already adequate for its primary purpose (estimating group-level success rates with uncertainty).

**If revision were pursued**, these priorities would guide the improvement process. However, given the CONDITIONAL ACCEPT decision, these are suggested enhancements rather than mandatory changes.

---

## Priority 1: Perform Leave-Out-Group Sensitivity Analysis

**Issue Addressed**: Model sensitivity to extreme groups (Groups 4 and 8 have Pareto k > 1.0)

**Proposed Action**:
Refit the model twice:
1. Excluding Group 4 (n=810, lowest rate 4.2%, k=1.01)
2. Excluding Group 8 (n=215, highest rate 14.0%, k=1.06)

Compare results:
- τ estimates across three fits (full, -Group4, -Group8)
- μ estimates
- Remaining group-level estimates
- Qualitative conclusions about heterogeneity

**Implementation**:
```python
# Fit 1: Exclude Group 4
data_no_4 = data[data['group'] != 4]
fit_no_4 = fit_hierarchical_binomial(data_no_4)

# Fit 2: Exclude Group 8
data_no_8 = data[data['group'] != 8]
fit_no_8 = fit_hierarchical_binomial(data_no_8)

# Compare
compare_tau = {
    'Full': tau_full,
    'Without Group 4': tau_no_4,
    'Without Group 8': tau_no_8
}
```

**Expected Outcomes**:
- **If τ changes <25%**: Strengthens claim of robustness, validates current model
- **If τ changes 25-50%**: Moderate sensitivity, report range of estimates
- **If τ changes >50%**: High sensitivity, should report uncertainty more prominently

**Effort**: Low
- Time: ~5 minutes (2 model fits at ~2.5 minutes each)
- Complexity: Minimal (subset data and refit)
- Code: Already written, just filter data

**Benefit**: High
- Quantifies impact of most influential observations
- Provides concrete evidence for robustness claims
- Addresses reviewer concerns about high Pareto k
- Strengthens publication readiness

**Recommendation**: **HIGHLY RECOMMENDED**

**Why not required**: Current model already passes core diagnostics; this analysis is confirmatory rather than corrective.

---

## Priority 2: Fit Beta-Binomial Model (Experiment 3)

**Issue Addressed**:
- LOO diagnostics fail (10/12 groups k > 0.7)
- Due diligence requires testing multiple models
- Potential within-group overdispersion unmodeled

**Proposed Action**:
Implement hierarchical Beta-binomial model:
```
alpha_j ~ Gamma(shape, rate)  # or other prior
beta_j ~ Gamma(shape, rate)
p_j = alpha_j / (alpha_j + beta_j)
r_j ~ BetaBinomial(n_j, alpha_j, beta_j)
```

Or equivalently with overdispersion parameter:
```
phi_j ~ Gamma(shape, rate)  # overdispersion per group
r_j ~ BetaBinomial(n_j, p_j, phi_j)
```

Compare to current model using:
1. **Posterior predictive checks**: Which model better reproduces data features?
2. **WAIC**: Quantitative comparison (not LOO due to k issues)
3. **Pareto k diagnostics**: Does Beta-binomial improve LOO reliability?
4. **Scientific interpretability**: Is additional complexity justified?

**Expected Outcomes**:

**Scenario A: Beta-binomial also has high k values**
- Confirms issue is data structure (J=12, extreme groups), not model misspecification
- Validates current Hierarchical Binomial as adequate
- Proceed with current model (ACCEPT decision reinforced)

**Scenario B: Beta-binomial improves k values**
- Suggests within-group overdispersion was present but undetected
- Beta-binomial may be more appropriate
- Decision: Prefer Beta-binomial if LOO improves substantially (e.g., <5 groups k>0.7)

**Scenario C: Beta-binomial worse fit**
- Posterior predictive checks fail or WAIC worse
- Over-parameterization without benefit
- Confirms current model is appropriate (ACCEPT decision reinforced)

**Effort**: Moderate
- Time: Full model development cycle (~2-3 hours)
  - Prior predictive check: ~45 min
  - Model fitting: ~2 min
  - Posterior predictive check: ~30 min
  - Comparison: ~30 min
- Complexity: Moderate (new likelihood, new priors)
- Code: Requires writing new model specification

**Benefit**: Moderate to High
- Tests alternative hypothesis about data generating process
- Standard workflow step (due diligence)
- If Beta-binomial improves LOO, provides superior model
- If not, validates current model choice

**Recommendation**: **RECOMMENDED** (part of standard workflow)

**Why not Priority 1**:
- More effort than sensitivity analysis
- Current model already adequate
- Outcome uncertain (may not improve LOO)
- Priority 1 directly addresses known issue

**Implementation Note**: This is Experiment 3 in the workflow plan. Should be pursued as part of comprehensive model comparison regardless of current model adequacy.

---

## Priority 3: Compute WAIC for Model Comparison

**Issue Addressed**: LOO unreliable for model comparison (high Pareto k values)

**Proposed Action**:
Compute WAIC (Widely Applicable Information Criterion) as alternative to LOO:

```python
import arviz as az

# Compute WAIC
waic_result = az.waic(idata, pointwise=True)

# Extract
waic_value = waic_result.waic
waic_se = waic_result.waic_se
p_waic = waic_result.p_waic  # effective number of parameters

# Compare to alternatives (if fitted)
comparison = az.compare({
    'Hierarchical Binomial': idata_exp1,
    'Beta-Binomial': idata_exp3
}, ic='waic')
```

**Why WAIC instead of LOO?**
- Does not require leave-one-out posterior approximation
- More stable with influential observations
- Widely accepted in Bayesian literature
- Can still be unreliable if model is misspecified, but less sensitive than LOO

**Expected Outcomes**:
- Provides quantitative comparison metric between models
- p_waic indicates effective model complexity
- WAIC differences guide model selection (prefer lower WAIC)
- SE of difference indicates if models are distinguishable

**Interpretation Guidelines**:
- ΔWAIC > 4: Substantial evidence for preferred model
- 2 < ΔWAIC < 4: Moderate evidence
- ΔWAIC < 2: Models effectively equivalent

**Effort**: Very Low
- Time: <1 minute per model
- Complexity: Single function call
- Code: Already available in ArviZ

**Benefit**: Moderate
- Provides quantitative comparison when LOO fails
- Widely accepted alternative to LOO
- Easy to implement and report

**Recommendation**: **RECOMMENDED** if comparing multiple models

**Caveat**: WAIC can still be unreliable if model is severely misspecified. Should be used alongside posterior predictive checks, not as sole comparison metric.

---

## Priority 4: Prior Sensitivity Analysis

**Issue Addressed**:
- Prior predictive check showed 6.88% extreme values (p > 0.8)
- Half-Cauchy prior on τ has heavy tails
- Verify posterior is not sensitive to prior choice

**Proposed Action**:
Refit model with alternative prior on τ:

**Original prior**: τ ~ Half-Cauchy(0, 1)
**Alternative prior**: τ ~ Half-Normal(0, 1)

**Why Half-Normal?**
- Lighter tails than Half-Cauchy
- Still weakly informative
- Standard alternative for hierarchical SD
- Reduces extreme prior values

Compare posteriors:
```python
# Compute posterior differences
tau_diff = tau_halfcauchy - tau_halfnormal
mu_diff = mu_halfcauchy - mu_halfnormal

# Check if posteriors overlap
overlap = compute_overlap(post_halfcauchy, post_halfnormal)

# Visual comparison
plot_prior_sensitivity(posterior_halfcauchy, posterior_halfnormal)
```

**Expected Outcomes**:

**Scenario A: Posteriors very similar**
- Confirms prior choice doesn't matter (data dominates)
- Validates current prior selection
- No action needed

**Scenario B: Posteriors moderately different**
- Prior has some influence on posterior
- Report sensitivity: "Results are somewhat sensitive to prior choice on τ"
- Consider using Half-Normal if it reduces Pareto k values

**Scenario C: Posteriors substantially different**
- Prior-data conflict or weak identifiability
- Concerning for inference
- May need to rethink model structure

**Effort**: Low
- Time: ~3 minutes (one additional model fit)
- Complexity: Minimal (change one line of prior specification)
- Code: Trivial modification

**Benefit**: Low to Moderate
- Validates prior choice (if similar)
- Addresses prior predictive check concern
- Standard sensitivity check for publication
- May improve LOO (if Half-Normal reduces k values)

**Recommendation**: **OPTIONAL but good practice**

**Why Priority 4**:
- Lower impact than Priorities 1-3
- Prior predictive issue was minor
- Posterior likely dominated by data (n=2814)
- More of a check-the-box exercise than addressing real concern

---

## Priority 5: K-Fold Cross-Validation

**Issue Addressed**:
- LOO unreliable (high Pareto k)
- Need trustworthy out-of-sample predictive accuracy estimate
- K-fold more stable than LOO for hierarchical models

**Proposed Action**:
Implement K-fold cross-validation:

```python
from sklearn.model_selection import KFold

# Define folds
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=42)

elpd_list = []

for train_idx, test_idx in kf.split(data):
    # Split data
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    # Fit on training data
    fit_train = fit_hierarchical_binomial(train_data)

    # Compute log predictive density on test data
    elpd_test = compute_elpd(fit_train, test_data)
    elpd_list.append(elpd_test)

# Aggregate
elpd_kfold = sum(elpd_list)
se_elpd_kfold = compute_se(elpd_list)
```

**Why K-fold instead of LOO?**
- Leaves out larger chunks (K groups instead of 1)
- Less sensitive to individual influential observations
- More stable for hierarchical models
- Provides genuine out-of-sample validation

**Recommended K values**:
- K = 10: Standard choice, balances stability and computation
- K = 5: Faster, still reasonable
- K = 12 (leave-one-group-out): Most relevant but unstable with influential groups

**Expected Outcomes**:
- Trustworthy estimate of out-of-sample predictive accuracy
- Can compare to alternative models using K-fold
- May show different model ranking than LOO (if LOO is unreliable)

**Effort**: High
- Time: ~K × 2 minutes (e.g., K=10 → 20 minutes)
- Complexity: Moderate (requires refitting K times)
- Code: More complex than single fit

**Benefit**: Moderate
- Provides reliable predictive accuracy estimate
- Gold standard for model comparison
- Addresses LOO limitation directly
- Publication-quality cross-validation

**Recommendation**: **OPTIONAL** (computationally expensive, diminishing returns)

**Why Priority 5**:
- Most computationally expensive
- Current model already adequate for inference
- Predictive accuracy not primary goal
- WAIC provides faster alternative
- Only pursue if:
  - Predictive accuracy is critical research goal
  - Multiple models need rigorous comparison
  - Computational budget allows

**Special Note**: With J=12 groups, leave-one-group-out CV may still show instability due to influential groups. Consider K=5 or K=10 with stratified sampling to ensure each fold has mix of sample sizes.

---

## Priority 6: Collect More Data (Long-term)

**Issue Addressed**:
- Small J (12 groups) contributes to LOO instability
- Group 4 dominates (29% of data)
- Improved inferences with more groups

**Proposed Action**:
For future studies, collect data on:
- **More groups**: Target J ≥ 20-30
- **Balanced sample sizes**: Reduce dominance of single groups
- **Group metadata**: Covariates explaining differences (if available)

**Why more groups help**:
- Each group less influential (1/30 vs 1/12)
- Hierarchical variance (τ) better identified
- LOO more stable
- Predictions more reliable

**Effort**: N/A (future study design)

**Benefit**: High (for future work)

**Recommendation**: **Consider for future studies**

---

## Recommended Action Plan

### Minimum Due Diligence (2-3 hours)
1. **Sensitivity Analysis** (Priority 1): 5 minutes
2. **Beta-Binomial Model** (Priority 2): 2-3 hours
3. **WAIC Comparison** (Priority 3): 1 minute

**Why**: These three actions provide comprehensive assessment of model robustness and adequacy. Priority 1 quantifies sensitivity, Priority 2 tests alternative hypothesis, Priority 3 enables quantitative comparison.

### Standard Publication Package (3-4 hours)
Add to above:
4. **Prior Sensitivity** (Priority 4): 3 minutes

**Why**: Adds standard sensitivity check expected in publications. Demonstrates thoroughness.

### Comprehensive Validation (4-5 hours)
Add to above:
5. **K-fold CV** (Priority 5): 20 minutes

**Why**: Provides gold-standard predictive accuracy estimate. Only if predictive performance is critical to research question.

### Time Budget Recommendations

**If you have 30 minutes**:
- Priority 1: Sensitivity Analysis (5 min)
- Priority 3: WAIC (1 min)
- Priority 4: Prior Sensitivity (3 min)
- Write up results (21 min)

**If you have 3 hours**:
- Priority 1: Sensitivity Analysis (5 min)
- Priority 2: Beta-Binomial Model (2.5 hours)
- Priority 3: WAIC Comparison (1 min)
- Priority 4: Prior Sensitivity (3 min)
- Write up results (20 min)

**If you have 5+ hours**:
- All priorities 1-5
- Comprehensive documentation
- Create publication-ready figures

---

## Implementation Status

**Current Status**: Model CONDITIONALLY ACCEPTED without improvements

**Required for publication**:
- None (model already adequate)
- Must document LOO limitations (see decision.md)

**Recommended for strong publication**:
- Priority 1 (Sensitivity Analysis): High impact, low effort
- Priority 2 (Beta-Binomial): Standard workflow requirement
- Priority 3 (WAIC): Enables model comparison

**Optional for comprehensive validation**:
- Priority 4 (Prior Sensitivity): Good practice
- Priority 5 (K-fold CV): Gold standard if needed

---

## Decision Points

### If Sensitivity Analysis (Priority 1) shows:
- **τ robust (<25% change)**: → Proceed with current model, publish with confidence
- **τ moderately sensitive (25-50%)**: → Report range, proceed with caution
- **τ highly sensitive (>50%)**: → Consider REVISE decision, investigate further

### If Beta-Binomial (Priority 2) shows:
- **Similar or worse LOO**: → Current model validated, proceed
- **Better LOO**: → Consider switching to Beta-binomial
- **Better PP checks**: → Prefer Beta-binomial
- **Worse PP checks**: → Prefer current model

### If WAIC (Priority 3) shows:
- **Current model preferred (ΔWAIC < -2)**: → Validated choice
- **Models equivalent (|ΔWAIC| < 2)**: → Current model adequate
- **Alternative preferred (ΔWAIC > 2)**: → Consider alternative if interpretable

---

## Summary

The Hierarchical Binomial model is **already adequate** for research use (CONDITIONAL ACCEPT). These improvement priorities are **enhancements**, not fixes:

1. **Priority 1**: Quantify sensitivity to extreme groups (**HIGHLY RECOMMENDED**, 5 min)
2. **Priority 2**: Try Beta-binomial alternative (**RECOMMENDED**, 2-3 hours)
3. **Priority 3**: Use WAIC for comparison (**RECOMMENDED**, 1 min)
4. **Priority 4**: Check prior sensitivity (**OPTIONAL**, 3 min)
5. **Priority 5**: K-fold cross-validation (**OPTIONAL**, 20 min)

**Recommended minimum**: Priorities 1-3 (total ~3 hours)

**Expected outcome**: Either (a) confirm current model is best choice, or (b) identify superior alternative (Beta-binomial). Either way, analysis is strengthened.

**Key insight**: The model's weaknesses are known and documented. These priorities help determine if the weaknesses are consequential (require switching models) or acceptable (proceed with current model).

---

**Document Status**: Ready for implementation
**Estimated Total Time**: 3-5 hours for comprehensive validation
**Required**: None (model already conditionally accepted)
**Recommended**: Priorities 1-3
