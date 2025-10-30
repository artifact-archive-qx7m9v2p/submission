# Simulation-Based Calibration: Recovery Metrics

**Model**: Beta-Binomial with conjugate priors
**Date**: 2025-10-30
**Method**: MAP estimation + Laplace approximation
**SBC Iterations**: 149/150 successful (99.3% success rate)

---

## Visual Assessment

The following diagnostic plots provide visual evidence for parameter recovery quality:

1. **`sbc_rank_histograms.png`**: Tests rank uniformity (primary SBC diagnostic)
2. **`coverage_calibration.png`**: Visualizes credible interval coverage rates
3. **`parameter_recovery_scatter.png`**: Shows bias and accuracy of point estimates
4. **`posterior_contraction.png`**: Demonstrates learning from data vs prior
5. **`parameter_space_identifiability.png`**: Maps explored parameter space
6. **`sbc_comprehensive_summary.png`**: Combined overview of all diagnostics

---

## Parameter Recovery Summary

### μ (Mean Success Probability)

**As illustrated in `parameter_recovery_scatter.png` (top left)**, μ recovery shows excellent performance:

- **Coverage Rate**: 0.966 (target: 0.950) ✓
  - Slightly conservative (over-coverage by 1.6%)
  - As shown in `coverage_calibration.png` (top left), green points indicate true values within 95% CI

- **Bias**: 0.0129 (target: ~0) ✓
  - Small positive bias, but negligible relative to scale
  - `parameter_recovery_scatter.png` (bottom left) shows errors scattered around zero

- **RMSE**: 0.0318 ✓
  - Good accuracy given prior SD = 0.0495

- **Posterior Contraction**: 0.645 ✓
  - As illustrated in `posterior_contraction.png` (left panel), posterior SD systematically smaller than prior SD
  - Indicates strong learning from 12 trials

- **Rank Statistics**: χ² = 71.27, p < 0.001 ⚠️
  - **Critical Finding**: As shown in `sbc_rank_histograms.png` (left panel), ranks are **NOT uniformly distributed**
  - Histogram shows excess mass around rank 750, with deficit in tails
  - Suggests **mild miscalibration** despite good coverage

**Visual Evidence**: The μ recovery scatter plot shows tight clustering around the identity line with symmetric scatter, confirming accurate point estimates.

---

### φ (Concentration Parameter)

**As illustrated in `parameter_recovery_scatter.png` (top right)**, φ recovery shows **SEVERE PROBLEMS**:

- **Coverage Rate**: 0.456 (target: 0.950) ✗ **CRITICAL FAILURE**
  - Only 45.6% of true values fall within 95% credible intervals
  - As shown in `coverage_calibration.png` (top right), **most points are RED** (outside CI)
  - True φ values consistently fall **above** the 95% upper bound

- **Bias**: -2.185 (target: ~0) ✗ **CRITICAL FAILURE**
  - Large negative bias: posterior systematically **underestimates** φ by ~2.2 units
  - `parameter_recovery_scatter.png` (bottom right) shows all errors negative and large magnitude
  - Pattern worsens for larger true φ values

- **RMSE**: 3.156 ✗ **CRITICAL FAILURE**
  - Extremely high error relative to prior SD = 0.707
  - RMSE > 4× prior SD indicates **severe recovery failure**

- **Posterior Contraction**: 0.908 ⚠️
  - As shown in `posterior_contraction.png` (right panel), posterior SD ≈ prior SD
  - **Minimal learning**: data provides little information about φ
  - Wide spread across true φ values suggests **weak identifiability**

- **Rank Statistics**: χ² = 1104.29, p < 0.001 ✗ **CRITICAL FAILURE**
  - **Extremely non-uniform ranks** as shown in `sbc_rank_histograms.png` (right panel)
  - Almost all ranks fall in narrow range [1000-1500], with virtually zero density elsewhere
  - This is **catastrophic miscalibration** indicating fundamental model problems

**Visual Evidence**: The φ recovery scatter plot shows a characteristic "shrinkage pattern" where posterior means are compressed toward small values (0-3) even when true φ is large (5-15).

---

## Critical Visual Findings

### 1. Rank Histogram Patterns (`sbc_rank_histograms.png`)

- **μ**: Roughly bell-shaped with central mode, suggesting over-concentration of posterior mass around true value
- **φ**: **Extremely concentrated spike** between ranks 1000-1500, indicating posterior is nearly identical across different true φ values

### 2. Coverage Failure Pattern (`coverage_calibration.png`)

- **μ**: Green points dominate (good coverage), credible intervals properly bracket true values
- **φ**: **Systematic under-coverage** - true φ consistently exceeds upper 95% bound, especially for φ > 5
  - Pattern suggests **credible intervals are too narrow** and **shifted too low**

### 3. Systematic Bias (`parameter_recovery_scatter.png`)

- **μ**: Points cluster tightly around y=x line (perfect recovery)
- **φ**: Points consistently **below** y=x line with increasing deviation at larger φ
  - This "shrinkage to prior" pattern indicates **weak identifiability** with N=12 trials

### 4. Posterior Contraction Patterns (`posterior_contraction.png`)

- **μ**: Clear contraction (posterior SD < prior SD), indicating strong learning
- **φ**: Minimal contraction (posterior SD ≈ prior SD), indicating **data provides little information**
  - Posterior uncertainty dominated by prior, not likelihood

### 5. Parameter Space Coverage (`parameter_space_identifiability.png`)

- Prior contours show most probability mass for φ < 4
- SBC successfully explored φ up to 15, revealing recovery failures in low-prior-density regions

---

## Convergence and Computational Performance

- **Success Rate**: 99.3% (149/150 iterations)
  - 1 optimization failure (iteration 17)
  - Excellent computational stability

- **Convergence Rate**: 100%
  - All successful fits achieved convergence (by optimization criteria)
  - However, **convergence ≠ correct calibration** (φ shows this clearly)

---

## Quantitative Calibration Metrics

### Coverage Diagnostics

| Parameter | Coverage | Target | Status | Deviation |
|-----------|----------|--------|--------|-----------|
| μ         | 0.966    | 0.950  | PASS   | +1.6%     |
| φ         | 0.456    | 0.950  | **FAIL** | **-49.4%** |

### Bias Diagnostics

| Parameter | Bias    | RMSE  | Prior SD | RMSE/Prior SD | Status |
|-----------|---------|-------|----------|---------------|--------|
| μ         | 0.0129  | 0.032 | 0.0495   | 0.64          | PASS   |
| φ         | -2.185  | 3.156 | 0.7071   | **4.46**      | **FAIL** |

### Rank Uniformity Tests

| Parameter | χ² Statistic | p-value | Interpretation | Status |
|-----------|--------------|---------|----------------|--------|
| μ         | 71.27        | <0.001  | Non-uniform ranks | WARN |
| φ         | 1104.29      | <0.001  | **Catastrophic** | **FAIL** |

**Note**: For well-calibrated SBC, rank statistics should follow χ²(19) distribution with expected value ≈19. The φ value of 1104 is **58× larger** than expected.

---

## Z-Score Analysis

Z-scores measure: (posterior_mean - true_value) / posterior_SD

Should follow N(0,1) if model is well-calibrated.

### μ Z-scores
- Mean: Small deviation from 0
- SD: Close to 1
- Distribution roughly normal
- **Conclusion**: Reasonable calibration

### φ Z-scores
- **Strongly negative** (large negative mean)
- **Left-skewed distribution**
- Indicates systematic underestimation with overly narrow uncertainties
- **Conclusion**: Poor calibration

---

## Identifiability Assessment

### μ Identifiability: **STRONG** ✓
- Posterior contracts significantly from prior (ratio = 0.645)
- Accurate recovery across full prior range [0, 0.25]
- 12 trials provide sufficient information

### φ Identifiability: **WEAK** ✗
- Minimal posterior contraction (ratio = 0.908)
- Poor recovery, especially for φ > 5
- 12 trials provide **insufficient information** to distinguish φ values
- **Practical non-identifiability** in regions with low prior mass

**Key Insight**: The model can learn about μ (mean success rate) but struggles to learn about φ (overdispersion) with only 12 trials. This is a **data limitation**, not necessarily a model misspecification.

---

## Pass/Fail Decision

### OVERALL: **CONDITIONAL FAIL**

**μ Parameter**: **PASS** ✓
- Excellent coverage (96.6%)
- Negligible bias (0.013)
- Strong identifiability
- Minor rank non-uniformity acceptable

**φ Parameter**: **FAIL** ✗
- Catastrophic coverage failure (45.6% vs 95% target)
- Large systematic bias (-2.185)
- Extremely non-uniform ranks (χ² = 1104)
- Weak identifiability with N=12 trials
- Credible intervals systematically too narrow and too low

---

## Implications for Real Data Analysis

### ⚠️ Critical Warning

**The model's failure to recover φ in simulation means it cannot be trusted to estimate φ from real data.**

### Specific Concerns

1. **Underestimation of Overdispersion**
   - Real data φ estimates will likely be biased downward
   - May conclude "moderate overdispersion" when true overdispersion is high

2. **Over-confident Uncertainty Intervals**
   - 95% credible intervals for φ will be too narrow
   - Real φ likely outside reported intervals

3. **μ Estimates May Still Be Useful**
   - μ recovery is good, so mean success rate estimates should be reliable
   - However, μ and φ are correlated, so φ miscalibration may affect μ

4. **Decision Risk**
   - Any decisions based on φ estimates (e.g., assessing overdispersion strength) are **high risk**
   - Cannot trust the model to distinguish φ=2 from φ=8

---

## Diagnostic Failure Mode

The specific pattern of φ failure suggests:

1. **Laplace approximation inadequacy**: Normal approximation may be poor for φ posterior
2. **Weak likelihood information**: 12 trials insufficient to constrain φ
3. **Prior-likelihood conflict**: Data weakly informs φ, posterior dominated by prior
4. **Practical non-identifiability**: φ effects subtle in data, hard to distinguish from binomial variability

---

## Recommendations

### Short-term (Required before using this model)

1. **Do NOT proceed with real data fitting** until φ recovery improves
2. **Re-run SBC with full MCMC** (not Laplace approximation) to test if approximation is the issue
3. **Investigate φ posterior geometry** - may have heavy tails, multimodality, or other non-normal features

### Medium-term (Model refinement)

4. **Consider reparameterization**: Log(φ) or other transformations for better posterior geometry
5. **Informative φ prior**: Use stronger prior informed by domain knowledge if available
6. **Hierarchical extension**: If real data has replicates, hierarchical structure may improve φ identifiability

### Long-term (Alternative approaches)

7. **Simpler model**: If φ not critical, use pooled binomial (ignores overdispersion)
8. **More data**: Collect additional trials if possible (need N >> 12 for reliable φ estimation)
9. **Alternative models**: Beta-Binomial may not be identifiable with this data structure

---

## Conclusion

**The Beta-Binomial model successfully recovers μ but catastrophically fails to recover φ with N=12 trials.**

The visual evidence in all diagnostic plots consistently shows:
- μ: Tight recovery around true values (good calibration)
- φ: Systematic underestimation and over-confident intervals (severe miscalibration)

**This is not a subtle issue** - with 45.6% coverage vs 95% target, the model is fundamentally unreliable for φ inference. The rank histogram for φ (showing almost all ranks in a narrow spike) is a "textbook example" of catastrophic SBC failure.

**Status**: Model validation **FAILED** for φ parameter. Cannot proceed to real data analysis without addressing this fundamental issue.
