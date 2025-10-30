# Experiment 2 Refined: NB-AR(1) Model with Constrained Priors

**Status**: Ready for Prior Predictive Check
**Date**: 2025-10-29
**Parent**: experiment_2 (failed prior predictive check)

---

## Quick Start

**Execute Prior Predictive Check**:
```bash
python /workspace/experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py
```

**Expected Runtime**: ~30-60 seconds
**Outputs**: 6 diagnostic plots + console summary

---

## What This Experiment Does

This is a **refined version** of Experiment 2 that fixes prior predictive check failures through targeted parameter constraints.

**Original Issue**: 3.22% of simulated counts exceeded 10,000 (vs observed max: 269)
**Root Cause**: Wide priors + exponential link → tail explosions
**Solution**: Constrain growth (β₁), stabilize variance (φ), tighten innovations (σ)

---

## Model Specification

### Statistical Model (Unchanged)

```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = η_t
η_t = β₀ + β₁×year_t + ε_t
ε_t = ρ×ε_{t-1} + ν_t
ν_t ~ Normal(0, σ)
```

### Prior Refinements (Changed)

| Parameter | Original | Refined | Change |
|-----------|----------|---------|--------|
| β₀ | Normal(4.69, 1.0) | Normal(4.69, 1.0) | **None** |
| β₁ | Normal(1.0, 0.5) | **TruncatedNormal(1.0, 0.5, -0.5, 2.0)** | Truncate |
| φ | Gamma(2, 0.1) | **Normal(35, 15), φ>0** | Inform from Exp1 |
| ρ | Beta(20, 2) | Beta(20, 2) | **None** |
| σ | Exponential(2) | **Exponential(5)** | Tighten |

---

## Key Files

### Documentation
- **`metadata.md`** - Complete model specification and prior justifications
- **`refinement_rationale.md`** - Detailed explanation of why each prior was changed
- **`README.md`** - This file

### Code
- **`prior_predictive_check/code/prior_predictive_check.py`** - Validation script (ready to run)

### Outputs (After Running PPC)
- **`prior_predictive_check/plots/`** - 6 diagnostic visualizations
- **`prior_predictive_check/findings.md`** - Decision summary (auto-generated)

---

## Refinement Summary

### What Changed and Why

#### 1. β₁: Truncated Growth Rate

**Original**: β₁ ~ Normal(1.0, 0.5) → Unbounded, allows β₁ > 2.5
**Refined**: β₁ ~ TruncatedNormal(1.0, 0.5, -0.5, 2.0)

**Why**:
- Prevents extreme growth rates (>25× over 40 years)
- Observed growth: 13× (21→269)
- Truncation bound: 25× (conservative)
- Experiment 1 posterior: β₁=0.87, far from bounds

**Impact**: Eliminates runaway trajectories in upper tail

#### 2. φ: Informed Dispersion

**Original**: φ ~ Gamma(2, 0.1) → Mean=20, allows φ<5
**Refined**: φ ~ Normal(35, 15), φ>0

**Why**:
- Experiment 1 posterior: φ=35.6±10.8
- φ is data property (count dispersion), unchanged by AR(1)
- Valid information transfer
- Prevents small φ (high variance) amplification

**Impact**: Stabilizes variance structure, reduces extreme draws

#### 3. σ: Tighter Innovations

**Original**: σ ~ Exponential(2) → E[σ]=0.5
**Refined**: σ ~ Exponential(5) → E[σ]=0.2

**Why**:
- AR innovations should be small deviations, not shocks
- Stationary SD: 0.48 vs 1.2 original
- Prevents extreme ε accumulation
- Still allows data to inform

**Impact**: Constrains AR process range to ~[-3,+3] vs [-10,+10]

### What Stayed the Same

**β₀ ~ Normal(4.69, 1.0)**: Already appropriate, exp(4.69)≈109
**ρ ~ Beta(20, 2)**: Theoretically motivated by EDA ACF=0.971

---

## Expected Improvements

### Quantitative Targets

| Metric | Original | Target | Expected |
|--------|----------|--------|----------|
| % counts > 10,000 | 3.22% | <1% | ~0.3% |
| Maximum count | 674 million | <100,000 | ~30,000 |
| 99th percentile | 143,745 | <5,000 | ~3,500 |
| Median count | 112 | ~110 | ~110 |

**Overall**: >90% reduction in extremes while preserving median

### Qualitative Goals

- No numerical instabilities (NaN, Inf)
- AR process range controlled
- Growth patterns realistic
- Temporal correlation structure preserved

---

## Success Criteria

### Prior Predictive Check Must Pass

- [ ] <1% of counts exceed 10,000
- [ ] <5% of counts exceed 5,000
- [ ] Maximum count < 100,000 per simulation
- [ ] Mean ACF(1) within [0.3, 0.99]
- [ ] No numerical errors
- [ ] Median behavior preserved (~100-200)

### If Successful → Next Steps

1. **Model Fitting**: PyMC implementation
   - 4 chains × 2000 iterations
   - target_accept=0.95
   - Monitor convergence closely

2. **Posterior Diagnostics**:
   - R-hat < 1.01 all parameters
   - ESS > 400 all parameters
   - <1% divergences
   - Posterior separates from prior

3. **Model Validation**:
   - Posterior predictive checks
   - LOO-CV vs Experiment 1
   - Residual ACF reduction

### If Failed → Alternative Paths

**Still >1% extremes**:
- Further constrain: β₁∈[-0.3,1.5], σ~Exp(10)
- OR: Simplify temporal structure
- OR: Alternative likelihood family

**Numerical issues**:
- Reparameterize (non-centered AR)
- Different software (Stan if available)

**AR validation poor**:
- N=40 may be insufficient
- Consider simpler correlation structure
- Accept Experiment 1 as final

---

## Relationship to Other Experiments

### From Experiment 1

**Information Used**:
- φ posterior (35.6±10.8) → prior center
- β₁ range (0.87±0.04) → truncation justification
- Residual ACF (0.511) → motivation for AR(1)

**Information Not Used**:
- Exact β values (let data inform with AR structure)
- Point estimates (use posterior uncertainty)

### To Experiment 2 (Original)

**Differences**:
- Same model structure
- Refined priors only
- Addresses PPC failures

**Improvements**:
- Expected >90% reduction in extremes
- Computational stability
- Scientific plausibility

---

## Running the Prior Predictive Check

### Command

```bash
cd /workspace
python experiments/experiment_2_refined/prior_predictive_check/code/prior_predictive_check.py
```

### What It Does

1. **Samples 500 parameter draws** from refined priors
2. **Generates AR(1) time series** for each draw
3. **Simulates count data** via negative binomial
4. **Computes diagnostics**:
   - Parameter distribution validation
   - Temporal correlation structure
   - Count distribution plausibility
   - Growth pattern realism
   - AR process behavior
5. **Creates 6 diagnostic plots**
6. **Outputs decision summary**

### Expected Output

**Console**:
- Parameter summaries
- Count distribution statistics
- Comparison to original Experiment 2
- Pass/fail for 7 checks
- Overall decision

**Plots** (saved to `prior_predictive_check/plots/`):
1. `prior_parameter_distributions.png` - Shows refined vs original
2. `temporal_correlation_diagnostics.png` - AR structure validation
3. `prior_predictive_trajectories.png` - Time series behavior
4. `prior_acf_structure.png` - Autocorrelation patterns
5. `prior_predictive_coverage.png` - Range and plausibility
6. `decision_summary.png` - One-page overview

### Interpreting Results

**PASS**: All checks green
- Action: Proceed to model fitting
- Document success in iteration_log.md

**CONDITIONAL PASS**: 6/7 checks, AR validation marginal
- Action: Proceed with caution
- N=40 limitation acknowledged
- Monitor in posterior diagnostics

**FAIL**: Critical checks failed
- Action: Diagnose and iterate
- Options: Further refinement vs simplification
- Document in findings.md

---

## File Organization

```
experiment_2_refined/
├── README.md                          # This file
├── metadata.md                        # Complete specification
├── refinement_rationale.md            # Detailed justification
│
├── prior_predictive_check/
│   ├── code/
│   │   └── prior_predictive_check.py # Validation script
│   ├── plots/                         # Diagnostic plots (after run)
│   └── findings.md                    # Results (after run)
│
├── posterior_inference/               # Created after PPC passes
│   ├── code/
│   ├── diagnostics/
│   └── plots/
│
└── posterior_predictive_check/        # Created after fitting
    ├── code/
    └── plots/
```

---

## Technical Notes

### Implementation Details

**PyMC Code** (after PPC passes):
```python
with pm.Model() as model:
    beta_0 = pm.Normal('beta_0', mu=4.69, sigma=1.0)
    beta_1 = pm.TruncatedNormal('beta_1', mu=1.0, sigma=0.5,
                                 lower=-0.5, upper=2.0)
    phi = pm.TruncatedNormal('phi', mu=35, sigma=15, lower=0)
    rho = pm.Beta('rho', alpha=20, beta=2)
    sigma = pm.Exponential('sigma', lam=5)

    epsilon = pm.AR('epsilon', rho=[rho], sigma=sigma, shape=N,
                    init_dist=pm.Normal.dist(0, sigma/pm.math.sqrt(1-rho**2)))

    eta = beta_0 + beta_1 * year + epsilon
    mu = pm.math.exp(eta)

    C_obs = pm.NegativeBinomial('C_obs', mu=mu, alpha=phi, observed=C)
```

### Potential Issues

**Computational**:
- High ρ (near 1) can cause slow mixing
- Truncated distributions may hit boundaries
- AR initialization sensitive

**Mitigations**:
- Non-centered parameterization if needed
- High target_accept (0.95)
- Monitor trace plots for boundary behavior

**Scientific**:
- Priors more informative than original
- Truncation might constrain inference
- φ assumption (unchanged by AR) might be wrong

**Checks**:
- Posterior vs prior separation
- No pileup at truncation bounds
- Sensitivity analysis on φ prior

---

## References

**Related Experiments**:
- `/workspace/experiments/experiment_1/` - Baseline NB linear model
- `/workspace/experiments/experiment_2/` - Original AR(1) (failed PPC)

**Key Documents**:
- `/workspace/experiments/experiment_2/prior_predictive_check/findings.md` - Original failure analysis
- `/workspace/experiments/iteration_log.md` - Complete iteration history

**Data**:
- `/workspace/data/data.csv` - Time series count data (N=40)

---

## Contact & Attribution

**Created**: 2025-10-29
**Model Refinement Agent**: Systematic prior improvement based on diagnostic feedback
**Workflow**: Bayesian workflow with prior predictive checks

**Guiding Principles**:
1. One change at a time (but can bundle related changes)
2. Target root causes, not symptoms
3. Maintain scientific plausibility
4. Know when to stop iterating
5. Document everything

---

**Status**: Ready for Validation
**Next Step**: Run prior_predictive_check.py
**Expected Outcome**: PASS with >90% improvement in tail behavior
