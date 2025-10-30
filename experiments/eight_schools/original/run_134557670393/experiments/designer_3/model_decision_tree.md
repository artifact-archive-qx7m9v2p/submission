# Model Decision Tree: Quick Visual Guide
## Model Designer #3 - Robust & Alternative Models

```
                          START: Standard Normal Hierarchical Model
                          (y_i ~ Normal(theta_i, sigma_i))
                                        |
                                        |
                          Run EDA → Key findings:
                          - I²=0% but wide range (-3 to 28)
                          - Study 1 influential (y=28)
                          - Potential clustering (p=0.009)
                          - Borderline significance (p≈0.05)
                                        |
                                        v
                          ================================
                          WHICH CONCERN IS PRIMARY?
                          ================================
                                        |
            +---------------------------+---------------------------+
            |                           |                           |
            v                           v                           v
    [OUTLIER/TAILS]            [SUBGROUP STRUCTURE]        [SE UNCERTAINTY]
            |                           |                           |
            v                           v                           v
    ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
    │ MODEL 1:          │      │ MODEL 2:          │      │ MODEL 3:          │
    │ STUDENT-T         │      │ MIXTURE           │      │ INFLATION         │
    │                   │      │                   │      │                   │
    │ y ~ Student-t(nu) │      │ θ ~ π*N(μ₂,τ₂) + │      │ y ~ N(θ, σ*λ)    │
    │ nu ~ Gamma(2,0.1) │      │     (1-π)*N(μ₁,τ₁)│      │ λ ~ LogNormal(0,.5)│
    │                   │      │ π ~ Beta(2,2)     │      │                   │
    │ Priority: HIGH    │      │ Priority: MEDIUM  │      │ Priority: MEDIUM  │
    └───────────────────┘      └───────────────────┘      └───────────────────┘
            |                           |                           |
            v                           v                           v
    ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
    │ FIT & CHECK       │      │ FIT & CHECK       │      │ FIT & CHECK       │
    │ - Converged?      │      │ - Converged?      │      │ - Converged?      │
    │ - PPC pass?       │      │ - Groups separate?│      │ - λ differs from 1?│
    │ - nu reasonable?  │      │ - Clear assignments│      │ - Improves LOO?   │
    └───────────────────┘      └───────────────────┘      └───────────────────┘
            |                           |                           |
            v                           v                           v
    ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
    │ FALSIFICATION     │      │ FALSIFICATION     │      │ FALSIFICATION     │
    │                   │      │                   │      │                   │
    │ ✗ nu > 50?        │      │ ✗ π extreme?      │      │ ✗ λ ≈ 1?         │
    │   → Use Normal    │      │   → Single pop    │      │   → Fix σ        │
    │                   │      │                   │      │                   │
    │ ✗ nu < 1.5?       │      │ ✗ |μ₂-μ₁| < 5?   │      │ ✗ λ > 2.5?       │
    │   → Contamination │      │   → Single pop    │      │   → Try Student-t│
    │                   │      │                   │      │                   │
    │ ✗ PPC fail?       │      │ ✗ Uncertain z_i?  │      │ ✗ LOO worse?     │
    │   → Try Mixture   │      │   → Too complex   │      │   → Use standard │
    │                   │      │                   │      │                   │
    │ ✗ LOO Pareto k?   │      │ ✗ LOO worse?      │      │ ✗ λ-τ corr>0.8?  │
    │   → Misspecified  │      │   → Overfitting   │      │   → Non-ID       │
    └───────────────────┘      └───────────────────┘      └───────────────────┘
            |                           |                           |
            +---------------------------+---------------------------+
                                        |
                                        v
                          ================================
                          MODEL COMPARISON (LOO-CV)
                          ================================
                                        |
                          Compare all models that passed
                          falsification checks
                                        |
                                        v
                    ┌───────────────────────────────────────┐
                    │ Decision Rules:                       │
                    │                                       │
                    │ 1. |elpd_diff| < 2*SE?               │
                    │    → Models equivalent, use SIMPLEST │
                    │                                       │
                    │ 2. elpd_diff > 2*SE?                 │
                    │    → Use BEST performing model       │
                    │                                       │
                    │ 3. All fail PPC?                     │
                    │    → PIVOT to different model class  │
                    │                                       │
                    │ 4. All show wide posteriors?         │
                    │    → J=8 insufficient, report as is  │
                    └───────────────────────────────────────┘
                                        |
                                        v
                          ================================
                          SENSITIVITY ANALYSES
                          ================================
                                        |
                          For best model(s):
                          - Leave-one-out (Study 1!)
                          - Prior sensitivity (tau)
                          - Robustness checks
                                        |
                                        v
                          ================================
                          FINAL REPORT
                          ================================
                                        |
                          - Posterior estimates
                          - Probability statements
                          - Uncertainty quantification
                          - Model limitations
                          - Recommendations
```

---

## Detailed Decision Points

### 1. Model Selection (Initial)

**Choose Student-t if:**
- Concerned about Study 1 influence
- Want distributional robustness
- Moderate complexity acceptable
- **RECOMMENDED FIRST CHOICE**

**Choose Mixture if:**
- Strong evidence of clustering from EDA
- Suspect distinct subpopulations
- Willing to deal with complexity
- Have time for longer fitting

**Choose Inflation if:**
- Doubt quality of reported SEs
- Want conservative uncertainty
- Need quick implementation
- As robustness check

**Fit all three if:**
- Have 6-10 hours available
- Want comprehensive comparison
- Uncertainty about which is best

---

### 2. Convergence Thresholds

```
MUST HAVE (all models):
├── R-hat < 1.01 for ALL parameters
├── ESS_bulk > 400 for key params (mu, tau, model-specific)
├── ESS_tail > 400 for mu, tau
├── Divergences < 10 after tuning
└── Max treedepth warnings < 5% iterations

NICE TO HAVE:
├── ESS > 1000 (better precision)
├── Zero divergences (perfect geometry)
└── All chains agree (visual check)

IF NOT MET:
├── Try: Longer warmup (3000-5000)
├── Try: Higher adapt_delta (0.95-0.99)
├── Try: Reparameterization
└── If persistent: ABANDON model
```

---

### 3. Falsification Decision Table

| Model | Check | Threshold | Action if Triggered | Priority |
|-------|-------|-----------|---------------------|----------|
| **Student-t** | nu > 50 | P(nu>50) > 0.8 | Use Normal | HIGH |
| **Student-t** | nu < 1.5 | Median < 1.5 | Contamination model | HIGH |
| **Student-t** | PPC fail | <80% coverage | Try Mixture | HIGH |
| **Student-t** | Pareto k | >0.7 for >2 studies | Misspecified | MEDIUM |
| **Mixture** | π extreme | P(π<0.1 or >0.9) > 0.8 | Single population | HIGH |
| **Mixture** | Not separated | P(\|μ₂-μ₁\|<5) > 0.7 | Single population | HIGH |
| **Mixture** | Uncertain z | >50% uncertain | Too complex for J=8 | HIGH |
| **Mixture** | LOO worse | elpd_diff < -2*SE | Overfitting | MEDIUM |
| **Inflation** | λ ≈ 1 | P(0.95<λ<1.05) > 0.7 | Fix σ | HIGH |
| **Inflation** | λ extreme | Median(λ) > 2.5 | Try Student-t | HIGH |
| **Inflation** | High corr | \|corr(λ,τ)\| > 0.8 | Non-identifiable | MEDIUM |
| **Inflation** | LOO worse | elpd_diff < -1*SE | Use standard | MEDIUM |

---

### 4. LOO Comparison Decision Flow

```
Step 1: Compute LOO for all fitted models
        └── Check Pareto k diagnostic
            ├── If k > 0.7 for >25%: LOO unreliable, use K-fold CV
            └── If k < 0.7 for >75%: Proceed

Step 2: Rank models by elpd_loo
        └── Best model = highest elpd

Step 3: Compare best to second-best
        ├── elpd_diff = elpd_best - elpd_second
        ├── SE_diff = standard error of difference
        └── Compute ratio: elpd_diff / SE_diff

Step 4: Decide
        ├── |ratio| < 2: Models equivalent
        │   └── Choose SIMPLER model
        │       Order: Standard < Inflation < Student-t < Mixture
        │
        └── |ratio| > 2: Best model clearly superior
            └── Choose BEST model
                (unless substantive concerns override)

Step 5: Validate choice
        ├── Check PPC for chosen model
        ├── Run leave-one-out sensitivity
        ├── Check prior sensitivity
        └── If all pass: Report this model
```

---

### 5. Posterior Interpretation Guide

```
For parameter θ with posterior samples θ₁, θ₂, ..., θₙ:

POINT ESTIMATE:
├── Median: median(θ)           [preferred for skewed]
├── Mean: mean(θ)               [preferred for symmetric]
└── Mode: density peak          [rarely used]

UNCERTAINTY:
├── 95% Credible Interval: [Q₀.₀₂₅, Q₀.₉₇₅]
├── 90% CI: [Q₀.₀₅, Q₀.₉₅]     [if less conservative]
└── SD: sd(θ)                   [average uncertainty]

PROBABILITY STATEMENTS:
├── P(θ > 0 | data) = mean(θ > 0)
├── P(θ > c | data) = mean(θ > c)  for any threshold c
└── P(θ₁ > θ₂ | data) = mean(θ₁ > θ₂)  for comparisons

INTERPRETATION THRESHOLDS (for μ):
├── P(μ > 0) > 0.975: "Strong evidence for positive effect"
├── P(μ > 0) > 0.95:  "Moderate evidence for positive effect"
├── P(μ > 0) > 0.80:  "Weak evidence for positive effect"
├── P(μ > 0) > 0.50:  "More likely positive than negative"
└── P(μ > 0) < 0.50:  "More likely negative than positive"

HETEROGENEITY (for τ):
├── P(τ > 0) ≈ 1 always (continuous prior)
├── Median(τ) < 1: "Low heterogeneity"
├── Median(τ) ∈ [1,5]: "Moderate heterogeneity"
├── Median(τ) > 5: "High heterogeneity"
└── Compare to effect size: τ/|μ| ratio
```

---

### 6. Red Flags and Warnings

```
🚩 STOP IMMEDIATELY IF:
├── R-hat > 1.05 after extended sampling
├── Negative ESS (indicates severe problems)
├── Posterior = Prior (no learning)
├── Parameters at boundary (tau=0, nu=1, etc.)
└── Extreme values (mu > 100, tau > 50 for this data)

⚠️  INVESTIGATE IF:
├── R-hat between 1.01-1.05
├── ESS < 200 for any parameter
├── Divergences > 10 but < 50
├── High correlation between parameters (|r| > 0.9)
└── Wide posteriors (95% CI spans >100 units)

✓ GOOD SIGNS:
├── R-hat < 1.01
├── ESS > 400 (ideally > 1000)
├── Zero divergences
├── Posterior differs meaningfully from prior
└── Results roughly consistent with EDA
```

---

### 7. Time-Limited Decision Path

**If you have only 2-3 hours:**
```
1. Fit Model 0 (Standard)                [30 min]
2. Fit Model 1 (Student-t)               [1 hour]
3. Compare via LOO                       [15 min]
4. Run PPC for better model              [15 min]
5. Quick sensitivity check (Study 1 LOO) [30 min]
6. Report                                [30 min]
TOTAL: ~3 hours
```

**If you have 4-6 hours:**
```
Add to above:
1. Fit Model 3 (Inflation)               [1 hour]
2. Three-way LOO comparison              [15 min]
3. Prior sensitivity (tau)               [1 hour]
4. Enhanced report                       [30 min]
TOTAL: ~5.5 hours
```

**If you have 8-10 hours:**
```
Add to above:
1. Fit Model 2 (Mixture)                 [2 hours]
2. Four-way LOO comparison               [30 min]
3. Full sensitivity suite                [2 hours]
4. Comprehensive report with visuals     [1 hour]
TOTAL: ~10 hours
```

---

### 8. What to Report (Minimum)

```
MUST REPORT:
├── Model selected and why
├── Posterior for μ: median + 95% CI
├── Posterior for τ: median + 95% CI
├── P(μ > 0 | data)
├── Convergence diagnostics (R-hat, ESS)
├── PPC results (coverage %)
└── LOO comparison table

SHOULD REPORT:
├── Model-specific parameters (nu, λ, π)
├── Leave-one-out sensitivity (Study 1)
├── Prior sensitivity (tau)
├── Study-specific shrinkage estimates
├── Forest plot with posteriors
└── Trace plots for key parameters

NICE TO REPORT:
├── Prior predictive check
├── Full posterior distributions (plots)
├── Correlation between parameters
├── Sensitivity to all priors
└── Comparison to frequentist estimates
```

---

## Quick Reference: Which Model for Which Problem?

| Problem/Pattern | Primary Model | Alternative | Rationale |
|----------------|---------------|-------------|-----------|
| Study 1 influential | Student-t | Inflation | Robust downweighting |
| I²=0% paradox | Standard | Student-t | May be true homogeneity |
| Clear clustering | Mixture | Student-t | Explicit subgroups |
| Borderline sig | Inflation | Student-t | Conservative CIs |
| Small J (=8) | Standard | Student-t | Fewer parameters |
| Suspect SEs | Inflation | Student-t | Direct SE modeling |
| Unknown dist | Student-t | Standard | Flexible tails |
| Need speed | Standard | Inflation | Fastest to fit |
| Need robustness | Student-t | Mixture | Multiple safeguards |
| Exploratory | All three | Compare LOO | Learn from data |

---

## Emergency Troubleshooting

```
PROBLEM: Model won't converge after 1 hour of trying
SOLUTION: Abandon complex model, use simpler one
          J=8 may be too small for complexity

PROBLEM: All models fail PPC
SOLUTION: Data has structure none of these capture
          Consider: state-space, GP, or report EDA only

PROBLEM: Results change drastically with prior
SOLUTION: Data too weak, report high uncertainty
          Don't force a conclusion from insufficient data

PROBLEM: LOO comparison shows all equivalent
SOLUTION: Use simplest model (Standard)
          Complexity not justified if predictive performance equal

PROBLEM: Inference contradicts EDA
SOLUTION: Check data input, investigate discrepancy
          Either modeling or EDA has error

PROBLEM: Can't decide between models
SOLUTION: Report results from 2-3 plausible models
          Model uncertainty is real, acknowledge it
```

---

**Remember**: The goal is **finding truth**, not **completing tasks**

- If models don't converge → say so
- If data insufficient → say so
- If conclusions uncertain → say so
- If assumptions violated → say so

Honest reporting of limitations is success, not failure.

