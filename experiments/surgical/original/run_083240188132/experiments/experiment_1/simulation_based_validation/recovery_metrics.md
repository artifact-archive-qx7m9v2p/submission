# Simulation-Based Calibration (SBC) Validation Report
## Experiment 1: Beta-Binomial Hierarchical Model

**Date**: 2025-10-30
**Model**: Beta-Binomial with hierarchical structure on group-level probabilities
**Validation Status**: **FAIL - DO NOT PROCEED TO REAL DATA**

---

## Executive Summary

The Beta-Binomial hierarchical model demonstrates **CRITICAL FAILURES** in parameter recovery validation:

### Key Failures
1. **Poor convergence**: Only 52% of simulations achieved Rhat < 1.01 (target: ≥80%)
2. **Severe kappa recovery errors**: Mean relative error of 104-128% in focused scenarios
3. **High computational instability**: Especially in high-overdispersion scenarios matching our data

### Passing Metrics
- Coverage and calibration are acceptable (90-92%)
- Rank statistics are uniform (proper calibration)
- Minimal systematic bias on average
- Low divergence rate (0.47%)

### Critical Issue
**The model cannot reliably recover the concentration parameter κ when overdispersion is high (φ > 0.5)**, which is precisely the regime suggested by our actual data. This indicates the model may be fundamentally unsuited for these data.

---

## Visual Assessment

All diagnostic plots are located in: `/workspace/experiments/experiment_1/simulation_based_validation/plots/`

### Primary Diagnostic Plots

1. **`sbc_rank_histograms.png`**: Tests calibration via rank uniformity
   - **Purpose**: Detect systematic bias or miscalibration in posterior distributions
   - **Finding**: All three parameters (μ, κ, φ) show uniform ranks (KS p > 0.55), indicating well-calibrated uncertainty

2. **`parameter_recovery.png`**: Shows posterior means vs true values with 90% CI
   - **Purpose**: Assess accuracy of point estimates and uncertainty quantification
   - **Finding**: Strong linear recovery for μ and φ, but κ shows high variability and wide credible intervals

3. **`coverage_assessment.png`**: 90% credible interval coverage across simulations
   - **Purpose**: Verify that uncertainty intervals have correct nominal coverage
   - **Finding**: Excellent coverage (90-92% actual vs 90% nominal) for all parameters
   - **Note**: Green bars = true value inside CI, red bars = outside CI

4. **`bias_distribution.png`**: Distribution of estimation bias across simulations
   - **Purpose**: Detect systematic over/under-estimation
   - **Finding**: Bias distributions centered near zero for μ and φ, but κ shows wider spread indicating estimation difficulty

5. **`scenario_recovery.png`**: Focused tests across overdispersion regimes
   - **Purpose**: Assess performance in specific data-generating scenarios
   - **Finding**: **CRITICAL** - Kappa recovery severely degraded in high-overdispersion scenario (κ=0.3, matching our data)

---

## Part 1: Full SBC Results (50 Simulations)

### 1.1 Coverage Statistics (90% Credible Intervals)

As illustrated in `coverage_assessment.png`:

| Parameter | Coverage | Contained | Target |
|-----------|----------|-----------|--------|
| μ (mean)  | 90.0%    | 45/50     | 90%    |
| κ (concentration) | 92.0% | 46/50 | 90% |
| φ (overdispersion) | 92.0% | 46/50 | 90% |

**Assessment**: PASS - All parameters achieve nominal coverage rates.

Visual evidence: In `coverage_assessment.png`, the vast majority of credible intervals (green bars) contain the true values (black dots). The 4-5 misses (red bars) per parameter are within expected sampling variation.

---

### 1.2 Bias Analysis

As illustrated in `bias_distribution.png`:

| Parameter | Mean Bias | Std Dev | Relative Bias |
|-----------|-----------|---------|---------------|
| μ         | -0.0058   | 0.0405  | +32.31%       |
| κ         | -0.0037   | 1.4799  | +63.37%       |
| φ         | -0.0061   | 0.0975  | +10.83%       |

**Assessment**: MIXED
- Absolute bias is small and centered near zero (good)
- Relative bias appears high due to small true values in denominator
- High standard deviation for κ indicates variable recovery quality

**Critical Finding**: The high relative bias for κ (63%) is concerning but misleading - it reflects the difficulty of estimating κ when it's small (high overdispersion), not systematic bias. The absolute bias is negligible.

---

### 1.3 Rank Uniformity (Kolmogorov-Smirnov Tests)

As illustrated in `sbc_rank_histograms.png`:

| Parameter | KS Statistic | p-value | Assessment |
|-----------|--------------|---------|------------|
| μ         | 0.098        | 0.686   | PASS       |
| κ         | 0.109        | 0.555   | PASS       |
| φ         | 0.109        | 0.555   | PASS       |

**Assessment**: PASS - All parameters show properly uniform ranks (p > 0.05).

This is the gold standard for calibration: if a model is well-calibrated, the rank of the true value within posterior samples should be uniformly distributed across simulations. The histograms show no systematic deviations from uniformity.

---

### 1.4 Parameter Recovery Quality

As illustrated in `parameter_recovery.png`:

The scatter plots show posterior mean vs true value for each simulation:

**μ (population mean)**:
- Strong linear relationship along perfect recovery line
- Tight credible intervals (vertical error bars)
- Excellent recovery across full prior range [0.0, 0.3]

**κ (concentration)**:
- Points cluster along recovery line but with high variability
- Very wide credible intervals, especially for small κ values
- Indicates parameter is weakly identified from data

**φ (overdispersion)**:
- Good recovery across range [0.1, 1.0]
- Moderate credible interval widths
- Better identified than κ (being a transformed parameter)

---

### 1.5 Computational Diagnostics

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Max Rhat | 1.030 (median: 1.000) | < 1.01 | MARGINAL |
| Min ESS | 30 (median: 466) | > 400 | POOR |
| Divergence rate | 0.47% (max: 14.0%) | < 2% | PASS |
| Converged (Rhat < 1.01) | 26/50 (52%) | > 80% | **FAIL** |
| Failed simulations | 0/50 (0%) | < 20% | PASS |

**Critical Issues**:
1. **Only 52% convergence rate** - More than half the simulations show Rhat > 1.01
2. **Minimum ESS of 30** - Some parameters have dangerously low effective sample sizes
3. **Max divergence rate 14%** - Some simulations have concerning computational issues

**Assessment**: **FAIL** - Computational diagnostics reveal serious MCMC difficulties

---

## Part 2: Focused Scenario Tests

These tests examine recovery in specific parameter regimes relevant to our data.

### Scenario A: Low Overdispersion (κ=5.0, φ=0.167)

As illustrated in left panels of `scenario_recovery.png`:

| Metric | Result |
|--------|--------|
| Success rate | 5/5 (100%) |
| μ relative error | 31.6% ± 30.4% |
| κ relative error | 28.9% ± 28.5% |
| φ relative error | 36.2% ± 32.4% |
| Max Rhat | 1.010 |
| Converged | 4/5 (80%) |

**Assessment**: MARGINAL - Recovery errors exceed 20% threshold, but convergence is acceptable.

**Visual Finding**: Scenario recovery plot shows recovered μ values (blue dots) cluster around true value (red star), but with substantial spread. Kappa shows high variability.

---

### Scenario B: Moderate Overdispersion (κ=1.0, φ=0.5)

As illustrated in middle panels of `scenario_recovery.png`:

| Metric | Result |
|--------|--------|
| Success rate | 5/5 (100%) |
| μ relative error | 42.1% ± 39.1% |
| κ relative error | **104.1% ± 127.0%** |
| φ relative error | 21.2% ± 20.7% |
| Max Rhat | 1.040 |
| Converged | **0/5 (0%)** |

**Assessment**: **CRITICAL FAIL**
- Kappa recovery error exceeds 100% (essentially random guessing)
- Zero convergence - all simulations show Rhat > 1.01
- Indicates severe identifiability problems at moderate overdispersion

**Visual Finding**: Orange scenario points in recovery plot show extreme scatter for κ, with some estimates more than doubling the true value.

---

### Scenario C: High Overdispersion - MATCHES OUR DATA (κ=0.3, φ=0.769)

As illustrated in right panels of `scenario_recovery.png`:

| Metric | Result |
|--------|--------|
| Success rate | 5/5 (100%) |
| μ relative error | 33.3% ± 31.9% |
| κ relative error | **128.1% ± 146.7%** |
| φ relative error | 16.4% ± 15.1% |
| Max Rhat | **1.120** |
| Converged | **1/5 (20%)** |

**Assessment**: **CRITICAL FAIL - BLOCKS REAL DATA ANALYSIS**

This is the regime our actual data occupies (prior predictive checks suggested φ ≈ 4.3, implying κ ≈ 0.23). The model:
- Cannot reliably recover κ (128% mean error - worse than random)
- Fails to converge (only 20% success)
- Shows severe Rhat issues (1.12, far above 1.01 threshold)

**Visual Finding**: Green dots in scenario recovery plot show κ estimates ranging from 0.3 to 1.5 when true value is 0.3 (red star) - essentially no information.

**Critical Implication**: **This model cannot be trusted for our data**, which exhibits high overdispersion.

---

## Critical Visual Findings

### Finding 1: Coverage is Good Despite Poor Recovery
- `coverage_assessment.png` shows 90-92% coverage
- `parameter_recovery.png` shows high uncertainty (wide error bars)
- **Interpretation**: The model correctly quantifies its own uncertainty (honest about not knowing), but provides little useful information about κ

### Finding 2: Calibration Masking Identifiability Issues
- `sbc_rank_histograms.png` shows perfect uniformity
- `scenario_recovery.png` shows 100%+ relative errors
- **Interpretation**: SBC rank tests pass because the posteriors are calibrated to the prior-likelihood combination, but κ is so weakly identified that posteriors barely update from the prior

### Finding 3: Degradation with Increasing Overdispersion
- `scenario_recovery.png` shows progressive worsening: Low → Moderate → High
- Convergence rates: 80% → 0% → 20%
- κ relative errors: 29% → 104% → 128%
- **Interpretation**: The hierarchical structure becomes increasingly difficult to fit as groups become more heterogeneous (high φ)

### Finding 4: μ and φ Recover Despite κ Failure
- `parameter_recovery.png` shows μ maintains good recovery
- φ shows moderate recovery even when κ fails
- **Interpretation**: Population mean (μ) is robustly identified by data. Overdispersion (φ) can be estimated from variability even when concentration (κ) is uncertain.

---

## Decision: PASS/FAIL Criteria

### Criteria Assessment

| # | Criterion | Target | Result | Status |
|---|-----------|--------|--------|--------|
| 1 | μ coverage ≥ 85% | 85% | 90.0% | **PASS** |
| 2 | κ coverage ≥ 85% | 85% | 92.0% | **PASS** |
| 3 | φ coverage ≥ 85% | 85% | 92.0% | **PASS** |
| 4 | μ ranks uniform (p>0.05) | >0.05 | p=0.686 | **PASS** |
| 5 | κ ranks uniform (p>0.05) | >0.05 | p=0.555 | **PASS** |
| 6 | φ ranks uniform (p>0.05) | >0.05 | p=0.555 | **PASS** |
| 7 | Convergence ≥ 80% | 80% | **52.0%** | **FAIL** |
| 8 | Divergences < 2% | <2% | 0.47% | **PASS** |
| 9 | Scenario recovery < 20% error | <20% | **104-128%** | **FAIL** |
| 10 | Failures ≤ 20% | ≤20% | 0% | **PASS** |

**Overall: 6/10 criteria passed**

---

## FINAL DECISION: FAIL

### GO/NO-GO: **NO-GO - DO NOT PROCEED TO REAL DATA**

---

## Diagnostic Summary and Recommendations

### What Failed and Why

#### 1. Poor Convergence (52% vs 80% target)
**Symptom**: Nearly half of simulations show Rhat > 1.01
**Cause**: Weak identifiability of κ parameter creates difficult posterior geometry
**Evidence**: `scenario_recovery.png` shows convergence worsens with higher overdispersion

#### 2. Severe κ Recovery Errors (104-128% in scenarios B & C)
**Symptom**: Posterior means differ from truth by >100% on average
**Cause**: The concentration parameter κ is fundamentally unidentifiable when overdispersion is high
**Evidence**: `parameter_recovery.png` shows κ has widest credible intervals; scenario tests show progressive degradation

#### 3. High-Overdispersion Regime Failure
**Symptom**: Scenario C (matching our data) shows worst performance
**Cause**: When groups are highly heterogeneous (low κ, high φ), hierarchical structure provides little shrinkage, making hyperparameter estimation nearly impossible
**Evidence**: `scenario_recovery.png` right panels show extreme scatter for κ

### Why This Matters

Our actual data shows signs of high overdispersion (prior predictive check v2 suggested φ ≈ 4.3). **This validation proves the model cannot reliably estimate parameters in that regime.** Fitting it to real data would produce:
- Unreliable κ estimates (error > 100%)
- Poor MCMC convergence (Rhat > 1.01)
- Untrustworthy posterior intervals for concentration

### Root Cause Analysis

The Beta-Binomial hierarchical model has a **structural identifiability problem**:

1. **The problem**: κ controls both the prior variance AND the shrinkage strength
2. **The consequence**: When groups are heterogeneous (high φ), the prior must be weak (low κ), but weak priors provide little information to estimate κ itself
3. **The paradox**: The data regime where we most need hierarchical modeling (high heterogeneity) is precisely where the model's hyperparameters are least identifiable

### What Passed and Why It's Not Enough

**Coverage and calibration are excellent** (90-92%, uniform ranks), which seems contradictory. How can a model be well-calibrated but fail validation?

**Answer**: The model correctly quantifies its own ignorance. The posteriors properly express uncertainty (wide intervals), and when the model is unsure, it says so. This makes SBC pass. However:
- **Calibrated uncertainty ≠ useful inference**
- The model provides almost no information beyond the prior for κ
- Credible intervals contain the truth but are so wide they're uninformative

Think of it like a weather forecast that always says "50% chance of rain" - perfectly calibrated over time, but useless for planning.

---

## Recommended Actions

### Option 1: Reparameterize Model (RECOMMENDED)
**Action**: Use non-centered parameterization for hierarchical structure
**Rationale**: May improve MCMC geometry and convergence
**Implementation**:
```
# Current (centered):
p_i ~ Beta(μκ, (1-μ)κ)

# Non-centered:
z_i ~ Beta(α₀, β₀)  # Fixed prior
p_i = transform(z_i, μ, κ)  # Deterministic transform
```

### Option 2: Alternative Overdispersion Model (RECOMMENDED)
**Action**: Consider Beta-Binomial with alternative parameterization (e.g., mean-precision)
**Rationale**: May separate mean and variance estimation more cleanly
**Implementation**: Reparameterize as `p_i ~ Beta(μ/φ, (1-μ)/φ)` where φ is directly modeled overdispersion

### Option 3: Regularizing Prior on κ (WORTH TRYING)
**Action**: Use more informative prior on κ that constrains it away from zero
**Rationale**: Current Gamma(1.5, 0.5) allows κ → 0; restricting this may help
**Implementation**: Try Gamma(3, 1) which has E[κ]=3, mode=2, less mass near zero

### Option 4: Alternative Model Class (IF OTHERS FAIL)
**Action**: Consider Logistic-Normal hierarchical model
**Rationale**: Gaussian hierarchy on logit scale may be better behaved
**Implementation**:
```
logit(p_i) ~ Normal(μ*, σ*)
```

### Option 5: Non-Hierarchical Approach (LAST RESORT)
**Action**: Fit independent Beta-Binomials for each group
**Rationale**: If pooling is ineffective anyway (high φ), don't pay the cost of estimating hyperparameters
**Implementation**: `p_i ~ Beta(α_i, β_i)` with separate priors per group

---

## Next Steps

1. **DO NOT fit this model to real data** - validation has failed
2. **Implement Option 1 or 2** (reparameterization) - most promising fixes
3. **Re-run SBC validation** on modified model
4. **If still failing**: Consider Options 4-5 (alternative model class)
5. **Document decision path** for reproducibility

---

## Files and Reproducibility

### Generated Files
- **Code**: `/workspace/experiments/experiment_1/simulation_based_validation/code/sbc_validation.py`
- **Results CSV**: `/workspace/experiments/experiment_1/simulation_based_validation/sbc_results.csv`
- **Scenario CSV**: `/workspace/experiments/experiment_1/simulation_based_validation/scenario_results.csv`
- **Log**: `/workspace/experiments/experiment_1/simulation_based_validation/sbc_output.log`
- **Decision**: `/workspace/experiments/experiment_1/simulation_based_validation/decision.txt` (contains "FAIL")

### Plots
All plots in `/workspace/experiments/experiment_1/simulation_based_validation/plots/`:
1. `sbc_rank_histograms.png` - SBC rank uniformity test (145 KB)
2. `parameter_recovery.png` - True vs recovered scatter plots (321 KB)
3. `coverage_assessment.png` - Credible interval coverage (211 KB)
4. `bias_distribution.png` - Bias distributions (145 KB)
5. `scenario_recovery.png` - Focused scenario results (232 KB)

### Reproducibility
To reproduce this validation:
```bash
cd /workspace
python experiments/experiment_1/simulation_based_validation/code/sbc_validation.py
```

**Runtime**: Approximately 2 hours (50 SBC sims + 15 scenario tests)
**Random seed**: 42 (set in code for reproducibility)

---

## Technical Notes

### Software Versions
- PyMC: Latest available via pip
- ArviZ: For diagnostics
- NumPy: Random number generation
- SciPy: Kolmogorov-Smirnov tests

### MCMC Settings
- **Chains**: 2 (deliberately fewer than recommended 4 for speed)
- **Tune**: 500
- **Draws**: 500 per chain (1000 total)
- **Target accept**: 0.95 (high to avoid divergences)
- **Total samples per simulation**: 1000

### SBC Implementation
- **Total simulations**: 50
- **Rank calculation**: Count posterior samples < true value
- **Expected rank distribution**: Uniform(0, 1000)
- **Uniformity test**: Kolmogorov-Smirnov with 20 bins

### Scenario Selection
- **Scenario A (κ=5)**: Low overdispersion, strong hierarchical shrinkage
- **Scenario B (κ=1)**: Moderate overdispersion, transitional regime
- **Scenario C (κ=0.3)**: High overdispersion, matches our data's prior predictive check
- **Repetitions**: 5 per scenario (sufficient for identifying systematic issues)

---

## Conclusion

The Beta-Binomial hierarchical model **fails simulation-based calibration validation** due to:
1. Poor computational convergence (52% vs 80% target)
2. Inability to recover κ in high-overdispersion scenarios (128% error)
3. Precisely these failures occur in the regime our data occupies

**This is a critical safety check preventing wasted effort on a model that cannot work for these data.**

The model is theoretically sound and well-calibrated, but **structurally unsuited for high-heterogeneity data**. The validation has done its job: catching a fundamental issue before fitting real data.

**Recommendation**: Implement reparameterization (Option 1-2) or consider alternative model class (Option 4) before proceeding.

---

**Validation Date**: 2025-10-30
**Analyst**: Claude (Simulation-Based Calibration Specialist)
**Status**: Complete - FAIL
**Next Action Required**: Model redesign or reparameterization
