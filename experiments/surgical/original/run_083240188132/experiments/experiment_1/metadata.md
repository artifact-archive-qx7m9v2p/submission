# Experiment 1: Beta-Binomial Hierarchical Model

## Model Specification

**Model Class**: Beta-Binomial Hierarchical (Overdispersed Binomial with Continuous Random Effects)

**Likelihood**:
```
r_i | p_i, n_i ~ Binomial(n_i, p_i)  for i = 1, ..., 12 groups
```

**Hierarchical Structure**:
```
p_i | μ, κ ~ Beta(α, β)  where α = μκ, β = (1-μ)κ
```

**Priors** (REVISED - Version 2, CONDITIONAL PASS):
```
μ ~ Beta(2, 18)          # E[μ] = 0.1, centered on pooled 7.4%
κ ~ Gamma(1.5, 0.5)      # E[κ] = 3, allows overdispersion φ ≈ 2-6
```

**Original priors** (Version 1, REJECTED):
- κ ~ Gamma(2, 0.1) → E[κ] = 20, φ ≈ 1.05 (too low overdispersion)

## Parameters of Interest

- **μ** (scalar): Population mean proportion, expected ~0.074
- **κ** (scalar): Concentration parameter, controls overdispersion
  - Lower κ → more overdispersion
  - φ = 1 + 1/κ (overdispersion factor)
  - Expected κ ~ 0.2-0.5 → φ ~ 3-6
- **p_i** (12 values): Group-specific proportions
  - Expected range: 0.03 to 0.15
  - Group 1: Should shrink from 0% toward μ
  - Groups 2,8,11: Should remain elevated but with some shrinkage

## Theoretical Justification

1. **Overdispersion modeling**: Beta-binomial is the canonical model for overdispersed binomial data
2. **Conjugate structure**: Beta is conjugate prior for binomial, leading to stable computation
3. **Direct parameterization**: φ = 1 + 1/κ directly links to observed overdispersion
4. **EDA alignment**: EDA showed φ = 3.5-5.1, ICC = 0.66 → κ should be 0.2-0.4

## How This Model Addresses Data Challenges

1. **Overdispersion (φ=3.5-5.1)**: Directly modeled via κ parameter
2. **Heterogeneity (ICC=0.66)**: Beta distribution allows wide variation in p_i
3. **Zero-event group (Group 1)**: Beta prior prevents p=0, shrinks toward μ
4. **Outliers (Groups 2,8,11)**: Beta allows extreme values with moderate shrinkage

## Falsification Criteria

Will REJECT this model if:
1. **Boundary behavior**: κ → 0 or κ → ∞ in posterior
2. **Poor coverage**: < 70% of groups within posterior predictive 95% CI
3. **φ mismatch**: Posterior φ doesn't overlap observed [3.5, 5.1]
4. **Computational failure**: Divergences > 2%, Rhat > 1.01, ESS < 400
5. **Implausible posteriors**: Any p_i > 0.3 or μ > 0.2

## Expected Outcomes

- κ posterior: 0.2-0.5 (median ~0.35)
- φ posterior: 3.0-6.0 (median ~4.0)
- μ posterior: 0.06-0.09 (median ~0.074)
- Group 1 posterior: p₁ ~ 0.03-0.05 (shrunk from 0%)
- Runtime: 2-5 minutes (4 chains × 1000 samples)

## Implementation Details

**Software**: PyMC 5.x
**Sampler**: NUTS (No-U-Turn Sampler)
**Chains**: 4
**Samples per chain**: 1000 (after 1000 tune)
**Total posterior samples**: 4000

## Status

- [FAILED] Prior predictive check v1 (κ ~ Gamma(2, 0.1)) - 2025-10-30
- [CONDITIONAL PASS] Prior predictive check v2 (κ ~ Gamma(1.5, 0.5)) - 2025-10-30
- [**FAILED**] Simulation-based validation - 2025-10-30
- [SKIPPED] Model fitting - validation failed
- [SKIPPED] Posterior predictive check - validation failed
- [REJECTED] Model critique - **EXPERIMENT 1 REJECTED**

## FINAL DECISION: REJECTED

**Date**: 2025-10-30
**Reason**: Simulation-based calibration revealed structural identifiability issues

**Key Problems**:
1. Poor convergence (52% vs 80% target)
2. Catastrophic κ recovery in high overdispersion scenarios (128% error)
3. Our data regime (φ≈4.3) is exactly where model fails
4. Root cause: κ parameter controls both prior variance and shrinkage, creating identifiability paradox

**What worked**: Coverage, calibration, bias all excellent - model is well-calibrated but provides little information about κ

**Lesson**: This is exactly why SBC validation exists - caught broken model before fitting real data

## Prior Predictive Check Results (v2)

**Date**: 2025-10-30
**Decision**: CONDITIONAL PASS

**Key Findings**:
- Prior φ range: [1.05, 95.70], 90% interval: [1.13, 3.92]
- Observed φ range [3.5, 5.1] is COVERED (v1 failed this)
- Prior predictive: 82.4% of simulations have between-group variability ≥ observed
- No computational red flags
- Prior is weakly informative (only 2.7% mass in exact observed φ range)

**Why Conditional**: Prior is somewhat diffuse, but this is acceptable for weakly informative approach. Data (n=2814) will dominate likelihood.

**Next Step**: Proceed to simulation-based validation

## Revision History

**v1** (Initial): κ ~ Gamma(2, 0.1)
- Prior predictive check FAILED
- Reason: κ too high (E[κ]=20), φ too low (≈1.05) to capture observed overdispersion (φ≈3.5-5.1)

**v2** (Revised): κ ~ Gamma(1.5, 0.5)
- Prior predictive check CONDITIONAL PASS
- Allows φ range 1.05-96, 90% interval [1.13, 3.92]
- Covers observed φ range [3.5, 5.1]
- Weakly informative, lets data drive inference
