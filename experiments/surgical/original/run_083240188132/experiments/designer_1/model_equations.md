# Model Equations: Side-by-Side Comparison
## Designer 1: Complete Mathematical Specifications

---

## Model 1: Beta-Binomial Hierarchical Model

### Likelihood Layer
```
r_i | p_i, n_i ~ Binomial(n_i, p_i)        for i = 1,...,12
```

### Group-Level Layer
```
p_i | μ, κ ~ Beta(μκ, (1-μ)κ)
```

### Hyperprior Layer
```
μ ~ Beta(2, 18)                             E[μ] = 0.10
κ ~ Gamma(2, 0.1)                           E[κ] = 20
```

### Derived Quantities
```
σ² = μ(1-μ) / (κ + 1)                      Population variance of p_i
φ = 1 + 1/κ                                 Overdispersion factor
ICC = 1 / (κ + 1)                          Intraclass correlation
```

### Parameter Interpretation
- **μ**: Population mean success rate (probability scale)
- **κ**: Concentration parameter
  - Low κ (< 1) → high variance → strong overdispersion
  - High κ (> 10) → low variance → weak overdispersion
- **p_i**: Group-specific success probability
- **α = μκ**, **β = (1-μ)κ**: Beta distribution shape parameters

### Why This Parameterization?
- μ directly interpretable as population mean rate
- κ controls concentration/dispersion
- More intuitive than raw (α, β)
- Better computational properties

---

## Model 2: Random Effects Logistic Regression

### Likelihood Layer
```
r_i | θ_i, n_i ~ Binomial(n_i, logit^(-1)(θ_i))    for i = 1,...,12
```

### Group-Level Layer (Non-Centered)
```
θ_i = μ + τ · z_i
z_i ~ Normal(0, 1)
```

### Hyperprior Layer
```
μ ~ Normal(logit(0.075), 1²)                μ = log(p/(1-p)) for p ≈ 0.075
τ ~ HalfNormal(1)                            Between-group SD on logit scale
```

### Derived Quantities
```
p_i = logit^(-1)(θ_i) = 1 / (1 + exp(-θ_i))    Group probability
p_mean = logit^(-1)(μ)                           Population mean probability
σ²_logit = τ²                                    Between-group variance (logit scale)
```

### Parameter Interpretation
- **μ**: Population mean on log-odds scale
  - μ = -2.5 → p ≈ 7.6%
  - μ = -2.0 → p ≈ 12%
- **τ**: Between-group standard deviation (logit scale)
  - τ = 0.5 → moderate heterogeneity
  - τ = 1.0 → substantial heterogeneity
  - τ = 2.0 → extreme heterogeneity
- **θ_i**: Group-specific log-odds
- **z_i**: Standardized random effect (non-centered)

### Why Non-Centered?
- Separates location (μ) from scale (τ)
- Reduces posterior correlation
- Essential for efficient MCMC sampling
- Standard practice for hierarchical models

---

## Model 3: Robust Logistic with Student-t Random Effects

### Likelihood Layer
```
r_i | θ_i, n_i ~ Binomial(n_i, logit^(-1)(θ_i))    for i = 1,...,12
```

### Group-Level Layer (Non-Centered)
```
θ_i = μ + τ · w_i
w_i ~ StudentT(ν, 0, 1)
```

### Hyperprior Layer
```
μ ~ Normal(logit(0.075), 1²)                μ = log-odds of population mean
τ ~ HalfNormal(1)                            Between-group SD (logit scale)
ν ~ Gamma(2, 0.1)                           Degrees of freedom (constrained > 2)
```

### Derived Quantities
```
p_i = logit^(-1)(θ_i)                       Group probability
p_mean = logit^(-1)(μ)                      Population mean probability
σ²_logit = τ² · ν/(ν-2)  for ν > 2        Between-group variance (logit scale)
```

### Parameter Interpretation
- **μ, τ**: Same as Model 2
- **ν**: Degrees of freedom (controls tail weight)
  - ν → ∞: Approaches Gaussian (Model 2)
  - ν ≈ 30: Nearly Gaussian
  - ν ≈ 10: Moderately heavy tails
  - ν ≈ 5: Heavy tails
  - ν ≈ 2: Very heavy tails (near Cauchy)
- **w_i**: Standardized heavy-tailed random effect

### Heavy-Tail Mechanism
Student-t as scale mixture of Gaussians:
```
w_i | λ_i ~ Normal(0, 1/λ_i)
λ_i ~ Gamma(ν/2, ν/2)
```
- Typical groups: λ_i ≈ 1 → Normal(0, 1)
- Outlier groups: λ_i < 1 → Larger variance

---

## Side-by-Side Comparison Table

| Component | Model 1: Beta-Binomial | Model 2: Logistic GLMM | Model 3: Robust Logistic |
|-----------|------------------------|------------------------|---------------------------|
| **Likelihood** | r ~ Binomial(n, p) | r ~ Binomial(n, logit⁻¹(θ)) | r ~ Binomial(n, logit⁻¹(θ)) |
| **Group-level** | p ~ Beta(μκ, (1-μ)κ) | θ = μ + τz, z ~ N(0,1) | θ = μ + τw, w ~ t(ν,0,1) |
| **Scale** | Probability [0,1] | Log-odds [-∞,∞] | Log-odds [-∞,∞] |
| **Hyperparameters** | μ, κ | μ, τ | μ, τ, ν |
| **Overdispersion** | Via κ (direct) | Via τ (indirect) | Via τ, ν (indirect) |
| **Tail behavior** | Beta (bounded) | Gaussian | Student-t (heavy) |
| **Outlier robustness** | Moderate | Moderate | High |
| **Parameters** | 2 hyperparams | 2 hyperparams | 3 hyperparams |

---

## Prior Choices Comparison

### Model 1 Priors
```
μ ~ Beta(2, 18)          Center: 0.10, SD: 0.06, Range: [0.02, 0.25] (95%)
κ ~ Gamma(2, 0.1)        Center: 20, SD: 14, Range: [2, 60] (95%)
```
**Philosophy**: Weakly informative, center on pooled estimate (10%), allow wide overdispersion range

### Model 2 Priors
```
μ ~ Normal(-2.5, 1²)     Center: logit(0.075), SD: 1, Range: [-4.5, -0.5] (95%)
                         → Probability range: [1%, 38%]
τ ~ HalfNormal(1)        Center: 0.8, SD: 0.6, Range: [0, 2] (95%)
```
**Philosophy**: Weakly regularizing, center on log-odds of pooled rate, allow substantial heterogeneity

### Model 3 Priors
```
μ ~ Normal(-2.5, 1²)     Same as Model 2
τ ~ HalfNormal(1)        Same as Model 2
ν ~ Gamma(2, 0.1)        Center: 20, SD: 14, Range: [2, 60] (95%)
```
**Philosophy**: Same as Model 2 for μ, τ; ν prior allows data to determine tail weight

---

## Relationship Between Parameters

### Model 1: ICC and κ
```
ICC = 1 / (κ + 1)

Given ICC = 0.66:
  κ = (1 - ICC) / ICC = 0.34 / 0.66 ≈ 0.51

Overdispersion φ:
  φ = 1 + 1/κ = 1 + 1/0.51 ≈ 2.96
```

### Model 2: τ and Overdispersion
```
On probability scale (approximate, for p ≈ 0.07):
  φ ≈ 1 + n̄ · p̄ · (1-p̄) · τ²

Given φ ≈ 4, n̄ = 235, p̄ = 0.074:
  4 ≈ 1 + 235 × 0.074 × 0.926 × τ²
  3 ≈ 16.2 × τ²
  τ ≈ √(3/16.2) ≈ 0.43

More generally, expect τ ≈ 0.5-1.0 for strong heterogeneity
```

### Model 3: ν and Tail Weight
```
Kurtosis = 6/(ν-4)  for ν > 4

ν = 5  → Kurtosis = 6  (very leptokurtic)
ν = 10 → Kurtosis = 1  (moderately leptokurtic)
ν = 30 → Kurtosis = 0.23  (nearly mesokurtic, close to Gaussian)
ν → ∞ → Kurtosis = 0  (Gaussian, mesokurtic)
```

---

## Expected Posterior Ranges

Based on EDA (φ=3.5-5.1, ICC=0.66, pooled rate=7.39%):

### Model 1
```
μ:     [0.06, 0.09]    (6-9% population mean)
κ:     [0.3, 1.0]      (implies ICC ≈ 0.5-0.77, φ ≈ 2-4)
φ:     [3.0, 5.0]      (matches observed overdispersion)
ICC:   [0.5, 0.77]     (overlaps observed 0.66)
p_i:   [0.01, 0.13]    (group-level rates)
```

### Model 2
```
μ:     [-2.8, -2.4]    (logit scale, → p ≈ 6-9%)
τ:     [0.5, 1.2]      (moderate to strong heterogeneity)
θ_i:   [-4.5, -1.5]    (group log-odds, → p ≈ 1-18%)
p_i:   [0.01, 0.13]    (group-level rates, similar to Model 1)
```

### Model 3
```
μ:     [-2.8, -2.4]    (same as Model 2)
τ:     [0.5, 1.2]      (same as Model 2)
ν:     [5, 40]         (data will determine if heavy tails needed)
θ_i:   [-4.5, -1.5]    (may allow more extreme values if ν low)
p_i:   [0.01, 0.13]    (similar to Models 1-2)
```

### Key Group Predictions

**Group 1 (0/47, zero events)**:
- Model 1: p_1 ≈ 0.03-0.05 (3-5%)
- Model 2: p_1 ≈ 0.02-0.05 (2-5%)
- Model 3: p_1 ≈ 0.02-0.06 (2-6%, wider due to heavy tails)

**Group 8 (31/215, 14.4%, highest outlier)**:
- Model 1: p_8 ≈ 0.11-0.14 (11-14%, moderate shrinkage)
- Model 2: p_8 ≈ 0.10-0.14 (10-14%, moderate shrinkage)
- Model 3: p_8 ≈ 0.11-0.15 (11-15%, less shrinkage due to heavy tails)

**Group 4 (46/810, 5.7%, largest sample)**:
- All models: p_4 ≈ 0.05-0.07 (5-7%, minimal shrinkage due to large n)

---

## Computational Complexity

### Model 1
```
Parameters per iteration:
  - μ, κ (2 hyperparameters)
  - p_1, ..., p_12 (12 group probabilities)
  - Total: 14 parameters

Sampling difficulty: EASY
  - Beta distribution well-behaved
  - May have slow mixing for κ with only 12 groups
  - Expected runtime: 2-5 minutes
```

### Model 2
```
Parameters per iteration:
  - μ, τ (2 hyperparameters)
  - z_1, ..., z_12 (12 standardized effects, non-centered)
  - θ_1, ..., θ_12 (12 log-odds, deterministic)
  - p_1, ..., p_12 (12 probabilities, deterministic)
  - Total: 26 (14 sampled, 12 deterministic)

Sampling difficulty: VERY EASY (with non-centered)
  - Non-centered parameterization eliminates funnel
  - Logistic models sample efficiently
  - Expected runtime: 2-5 minutes
```

### Model 3
```
Parameters per iteration:
  - μ, τ, ν (3 hyperparameters)
  - w_1, ..., w_12 (12 heavy-tailed standardized effects)
  - θ_1, ..., θ_12 (12 log-odds, deterministic)
  - p_1, ..., p_12 (12 probabilities, deterministic)
  - Total: 27 (15 sampled, 12 deterministic)

Sampling difficulty: MODERATE
  - Student-t more complex geometry than Gaussian
  - ν parameter difficult to estimate with 12 groups
  - May need higher target_accept (0.95)
  - Expected runtime: 5-10 minutes
```

---

## Prior Predictive Distributions

What do priors imply before seeing data?

### Model 1
```
Prior on μ: Beta(2, 18)
  → 95% prior interval: [0.02, 0.25]
  → Most mass 0.05-0.20

Prior on κ: Gamma(2, 0.1)
  → 95% prior interval: [2, 60]
  → E[φ] = E[1 + 1/κ] ≈ 1.05 (but wide range 1.02-1.5)

Prior predictive datasets:
  → Group rates: wide range 0-40%, covering observed
  → Occasional zero-event groups: ~5-10% of simulations
  → Occasional high-rate outliers (>15%): ~5-10% of simulations
```

### Model 2
```
Prior on μ: Normal(-2.5, 1²)
  → 95% interval on log-odds: [-4.5, -0.5]
  → 95% interval on probability: [1%, 38%]

Prior on τ: HalfNormal(1)
  → 95% interval: [0, 2.0]
  → E[τ] ≈ 0.8

Prior predictive datasets:
  → Group rates: wide range 0-40%, covering observed
  → Between-group SD: typically 3-8% on probability scale
  → Occasional extreme groups: ~5% of simulations
```

### Model 3
```
Same as Model 2, plus:

Prior on ν: Gamma(2, 0.1)
  → 95% interval: [2, 60]
  → E[ν] = 20 (moderately heavy tails)
  → Prior allows ν = 5 (very heavy) to ν = 50 (nearly Gaussian)

Prior predictive datasets:
  → Similar to Model 2 but with occasional more extreme outliers
  → More variability in between-group heterogeneity
```

---

## Falsification Thresholds (Numeric)

### Model 1: REJECT if
```
κ < 0.01                                 (extreme overdispersion)
κ > 1000                                 (no overdispersion)
P(3.5 < φ < 5.1 | data) < 0.05         (φ doesn't overlap observed)
Coverage < 70%                           (systematic misfit)
Divergences > 1% of samples             (computational failure)
max(Rhat) > 1.05                        (convergence failure)
```

### Model 2: REJECT if
```
τ > 2.0                                  (extreme heterogeneity)
Coverage < 70%                           (systematic misfit)
Outliers outside 95% PI: 3/3            (all outliers misfit)
Divergences > 1% of samples             (computational failure)
max(Rhat) > 1.05                        (convergence failure)
```

### Model 3: REJECT if
```
P(ν > 30 | data) > 0.95                 (Model 2 adequate)
P(ν < 4 | data) > 0.90                  (infinite kurtosis region)
Coverage < 70%                           (systematic misfit even with heavy tails)
Divergences > 2% of samples             (computational failure)
max(Rhat) > 1.05                        (convergence failure)
```

---

## Summary

### Model 1: Beta-Binomial
- **Best for**: Direct overdispersion modeling, natural for binomial data
- **Key parameters**: μ (mean), κ (concentration)
- **Expected posterior**: κ ≈ 0.5, φ ≈ 4, ICC ≈ 0.67

### Model 2: Logistic GLMM
- **Best for**: Standard approach, extensible, familiar
- **Key parameters**: μ (log-odds mean), τ (between-group SD)
- **Expected posterior**: τ ≈ 0.7-1.0

### Model 3: Robust Logistic
- **Best for**: Accommodating extreme outliers
- **Key parameters**: μ, τ (same as Model 2), ν (degrees of freedom)
- **Expected posterior**: ν ≈ 10-20 if needed; ν > 30 suggests Model 2 adequate

**All models should give similar group-level estimates (p_i) if data fit well.**

**Expected final choice: Model 1** (most natural for overdispersed binomial data)

---

## Quick Reference: Model Selection

```
IF Model 1 & Model 2 both adequate:
  → Choose Model 1 (more direct for binomial overdispersion)

IF only Model 1 adequate:
  → Use Model 1

IF only Model 2 adequate:
  → Use Model 2

IF both inadequate (especially outlier fit):
  → Try Model 3
  → IF ν > 30: revert to Model 2
  → IF ν < 30: use Model 3

IF all fail:
  → Consider mixture model (two subpopulations)
  → Investigate data quality
  → Consult domain experts
```

---

**End of Model Equations Document**
