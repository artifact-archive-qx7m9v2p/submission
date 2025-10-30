# Visual Model Comparison: Designer 2 Approaches

This document provides intuitive visual/conceptual comparisons of the proposed models.

---

## Model Architecture Comparison

```
MODEL 1: Quadratic + Time-Varying Dispersion
═══════════════════════════════════════════════

Data: C[i] (counts at time points)
         ↓
  ┌──────────────────┐
  │ Mean Function    │
  │ log(μ) = β₀ +    │
  │   β₁×year +      │  ← Captures ACCELERATION
  │   β₂×year²       │
  └──────────────────┘
         ↓
  ┌──────────────────┐
  │ Dispersion Fn    │
  │ log(φ) = γ₀ +    │  ← Captures HETEROSCEDASTICITY
  │   γ₁×year        │
  └──────────────────┘
         ↓
  ┌──────────────────┐
  │ Negative         │
  │ Binomial(μ, φ)   │  ← Captures OVERDISPERSION
  └──────────────────┘
         ↓
    Likelihood

Complexity: 5 parameters
Flexibility: Moderate (smooth curves)
Strength: Captures known EDA patterns
Weakness: Polynomial extrapolation risk
```

```
MODEL 2: Piecewise Regime Shift
═══════════════════════════════════

Data: C[i] at time points
         ↓
    [Changepoint τ = -0.21]
         ↓
  Early Regime  │  Late Regime
  (year < τ)    │  (year > τ)
                │
  β₀ + β₁×year  │  (β₀+β₂) + (β₁+β₃)×year
                │
  ──────────────┼──────────────  ← DISCONTINUOUS JUMP
                │
         ↓
  ┌──────────────────┐
  │ Regime-specific  │
  │ Dispersion       │  ← Two dispersion levels
  │ φ₁ or φ₂         │
  └──────────────────┘
         ↓
  Negative Binomial(μ, φ)
         ↓
    Likelihood

Complexity: 5 parameters
Flexibility: Moderate (two linear pieces)
Strength: Interpretable regime change
Weakness: Artificial discontinuity if shift is smooth
```

```
MODEL 3: Hierarchical B-Spline
═══════════════════════════════════

Data: C[i] at time points
         ↓
  ┌──────────────────┐
  │ B-Spline Basis   │
  │ B₁, B₂, ..., B₆  │  ← 6 knots for flexibility
  └──────────────────┘
         ↓
  ┌──────────────────┐
  │ Hierarchical     │
  │ Coefficients     │  ← Adaptive shrinkage
  │ β ~ N(0, σ_β)    │     prevents overfitting
  └──────────────────┘
         ↓
  ┌──────────────────┐
  │ Flexible Mean    │
  │ log(μ) = Σ β_k   │  ← Can capture ANY smooth shape
  │         × B_k    │
  └──────────────────┘
         ↓
  [Similar spline for dispersion]
         ↓
  Negative Binomial(μ, φ)
         ↓
    Likelihood

Complexity: 10+ parameters (regularized)
Flexibility: Maximum (local adaptation)
Strength: No functional form assumptions
Weakness: Computational cost, harder interpretation
```

---

## Visual Intuition: What Each Model Captures

```
OBSERVED DATA PATTERN (from EDA):
═════════════════════════════════

Count
  |
300|                                    * *
  |                                  *   *
250|                               *
  |                           * *
200|                        *
  |                    * *
150|                 *
  |            * *
100|         *
  |    * *
 50|  *
  |*
  +─────────────────────────────────────> Year
 -1.5                0               +1.5

Key features:
1. Curvature (not straight line)
2. Increasing variance (spread gets wider)
3. Possible "kink" around year = -0.21
```

```
MODEL 1 FIT (Quadratic):
════════════════════════

Count
  |                                    ╱
300|                                 ╱
  |                               ╱ ← Smooth curve
250|                            ╱
  |                         ╱
200|                      ╱
  |                   ╱
150|                ╱
  |             ╱
100|          ╱
  |       ╱
 50|    ╱
  | ╱
  +─────────────────────────────────────> Year

Captures:
✓ Smooth acceleration
✓ Time-varying variance (via φ[i])
✗ Sharp regime change (if present)
```

```
MODEL 2 FIT (Piecewise):
════════════════════════

Count
  |                    ╱
300|                 ╱ ← Steep slope
  |               ╱
250|            ╱
  |          ╱
200|       ╱
  |     ╱
150|   ╱
  | ╱──┘ ← JUMP at τ=-0.21
100|╱
  |╱ ← Gentle slope
 50|
  |
  +─────────────────────────────────────> Year
     -0.21 = changepoint

Captures:
✓ Discrete regime shift
✓ Different slopes pre/post
✗ Smooth transitions
```

```
MODEL 3 FIT (Spline):
═════════════════════

Count
  |                                 ╱⌢
300|                              ╱    ← Can wiggle
  |                            ╱⌣
250|                         ╱
  |                      ╱⌢
200|                   ╱
  |                ╱⌣
150|             ╱
  |          ╱⌢
100|       ╱
  |    ╱⌣
 50| ╱
  |╱
  +─────────────────────────────────────> Year

Captures:
✓ Any smooth shape
✓ Local features
✗ May overfit
```

---

## Variance Structure Comparison

```
TIME-VARYING DISPERSION (Models 1-2):
═════════════════════════════════════

Early Period         Middle Period        Late Period
(year < -0.5)        (-0.5 < year < 0.5)  (year > 0.5)

Var/Mean ≈ 0.58      Var/Mean ≈ 11.85     Var/Mean ≈ 4.4

  *                      * *                   *  *
  *                    *     *                 *    *
  *                  *         *              *      *
  *                *             *           *        *
  * *            *                 *        *          *

φ ≈ 2.5              φ ≈ 1.0               φ ≈ 1.8

Model 1: log(φ) = γ₀ + γ₁×year (smooth change)
Model 2: log(φ) = γ₀ + γ₁×I(year>τ) (step change)
Model 3: log(φ) = Σ γ_j × B_j(year) (flexible)
```

```
CONSTANT DISPERSION (Designer 1 likely proposes):
═════════════════════════════════════════════════

All time periods: φ ≈ 1.5 (constant)

  * *                  * *                   * *
  *  *                *   *                 *   *
  *   *              *     *               *     *
  *    *            *       *             *       *
  *     *          *         *           *         *

Problem: Cannot capture 20x variance change!
→ Miscalibrated prediction intervals
→ Poor uncertainty quantification
```

---

## Decision Tree: Which Model to Use?

```
                        ┌─────────────────┐
                        │ Start: Fit      │
                        │ Model 1         │
                        │ (Quadratic +    │
                        │  time-vary φ)   │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Convergence OK?         │
                    │ (R-hat < 1.01)          │
                    └────────┬────────────────┘
                         Yes │       No
                             │        │
                             │        └──> Simplify to
                             │             constant φ
                             │
                    ┌────────┴────────────┐
                    │ β₂ significant?     │
                    │ (CI excludes 0)     │
                    └────────┬────────────┘
                         Yes │       No
                             │        │
                             │        └──> Revert to
                             │             log-linear
                             │
                    ┌────────┴────────────┐
                    │ γ₁ significant?     │
                    │ (CI excludes 0)     │
                    └────────┬────────────┘
                         Yes │       No
                             │        │
                             │        └──> Use constant φ
                             │
                    ┌────────┴────────────┐
                    │ Posterior pred      │
                    │ checks pass?        │
                    └────────┬────────────┘
                         Yes │       No
                             │        │
                             │        ├──> Systematic pattern
                             │        │    near τ = -0.21?
                             │        │         │
                             │        │         └──> Try Model 2
                             │        │              (Piecewise)
                             │        │
                             │        └──> Other patterns?
                             │                  │
                             │                  └──> Try Model 3
                             │                       (Spline)
                             │
                    ┌────────┴────────────┐
                    │ LOO-CV comparison   │
                    │ with Designer 1     │
                    └────────┬────────────┘
                             │
                    ┌────────┴────────────┐
                    │ ΔELPD > 2×SE?       │
                    └────────┬────────────┘
                         Yes │       No
                             │        │
                      Accept │        └──> Models tied,
                      Model 1│             prefer simpler
                             │
                    ┌────────┴────────────┐
                    │ SUCCESS: Model 1    │
                    │ captures complexity │
                    └─────────────────────┘
```

---

## Parameter Interpretation Guide

### Model 1: Quadratic + Time-Varying Dispersion

```
Parameter  │ Interpretation                        │ Expected Value
═══════════╪═══════════════════════════════════════╪═══════════════
β₀         │ Log-count at year=0 (center)          │ 4.3 ± 0.3
           │ → Count ≈ exp(4.3) ≈ 73                │
───────────┼───────────────────────────────────────┼───────────────
β₁         │ Linear growth rate                    │ 0.85 ± 0.2
           │ → 135% increase per year               │
───────────┼───────────────────────────────────────┼───────────────
β₂         │ Acceleration term                     │ 0.3 ± 0.15
           │ → Growth rate increases over time      │
           │ → CRITICAL: Test if ≠ 0                │
───────────┼───────────────────────────────────────┼───────────────
γ₀         │ Baseline log-dispersion               │ 0.4 ± 0.3
           │ → φ ≈ exp(0.4) ≈ 1.5 at year=0         │
───────────┼───────────────────────────────────────┼───────────────
γ₁         │ Dispersion change over time           │ -0.2 ± 0.3
           │ → Captures heteroscedasticity          │
           │ → CRITICAL: Test if ≠ 0                │
```

**Key inference**: If both β₂ and γ₁ are significant → Model 1 justified
                 If either ≈ 0 → Simplify accordingly

---

### Model 2: Piecewise Regime Shift

```
Parameter  │ Interpretation                        │ Expected Value
═══════════╪═══════════════════════════════════════╪═══════════════
β₀         │ Log-count at year=0, pre-regime       │ 4.0 ± 0.3
───────────┼───────────────────────────────────────┼───────────────
β₁         │ Growth rate in early regime           │ 0.3 ± 0.2
           │ → Slow initial growth                  │
───────────┼───────────────────────────────────────┼───────────────
β₂         │ Level shift at changepoint            │ 0.5 ± 0.3
           │ → Jump in counts at τ=-0.21            │
───────────┼───────────────────────────────────────┼───────────────
β₃         │ Additional slope post-regime          │ 2.5 ± 0.5
           │ → Late growth = β₁ + β₃ ≈ 2.8          │
           │ → Acceleration = (β₁+β₃)/β₁ ≈ 9.3x     │
───────────┼───────────────────────────────────────┼───────────────
γ₀, γ₁     │ Regime-specific dispersion            │ Similar to M1
```

**Key inference**: β₃ >> 0 confirms 9.6x acceleration
                  Scientific meaning of τ=-0.21 determines model validity

---

## Computational Complexity Comparison

```
SAMPLING TIME ESTIMATES (4 chains × 2000 iter):
═══════════════════════════════════════════════

Model 1 (Quadratic):        ~2-3 minutes
├─ Parameters: 5
├─ Geometry: Simple (quadratic surface)
└─ Typical divergences: < 1%

Model 2 (Piecewise):        ~3-5 minutes
├─ Parameters: 5
├─ Geometry: Discontinuity at changepoint
└─ Typical divergences: 1-3% (at τ)

Model 3 (Spline):           ~10-20 minutes
├─ Parameters: 10+
├─ Geometry: High-dimensional
├─ Requires: adapt_delta=0.99
└─ Typical divergences: 2-5%

Log-Linear (Designer 1):    ~1-2 minutes
├─ Parameters: 3
├─ Geometry: Simple
└─ Typical divergences: < 0.5%

Tradeoff: Complexity ↔ Computational Cost ↔ Model Fit
```

---

## Stress Test Scenarios

```
TEST 1: Holdout Prediction (Last 20% of data)
═════════════════════════════════════════════

Train on:  year ∈ [-1.67, +1.00]  (n=32)
Test on:   year ∈ [+1.00, +1.67]  (n=8)

Expected:
  Model 1: Good (polynomial continues smoothly)
  Model 2: Good (stays in regime 2)
  Model 3: Risky (spline extrapolation uncertain)

Failure signal: RMSE > 30 or MAE > 20
```

```
TEST 2: Leave-One-Out Extremes
═══════════════════════════════

Iteratively remove:
  - Highest 5 observations (C > 250)
  - Lowest 5 observations (C < 30)

Expected:
  Parameter estimates stable within 20%

Failure signal:
  β₂ or γ₁ change > 50% → Model driven by outliers
```

```
TEST 3: Variance Reproduction
══════════════════════════════

Split data: Early (n=13) | Middle (n=14) | Late (n=13)

Observed Var/Mean:  0.58  |  11.85  |  4.4
Model 1 posterior:  0.4-0.8|  8-15  | 3-6  ✓
Constant φ:         ~3.5   |  ~3.5  | ~3.5  ✗

Success: Within 50% of observed in each period
Failure: Cannot reproduce U-shaped pattern
```

---

## Summary: Model Selection Rubric

```
Choose Model 1 (Quadratic) if:
✓ β₂ significantly positive
✓ γ₁ significantly non-zero
✓ Smooth acceleration evident
✓ No sharp regime change visible

Choose Model 2 (Piecewise) if:
✓ Clear "kink" at τ = -0.21
✓ Scientific reason for regime shift
✓ Model 1 shows residual clustering at τ

Choose Model 3 (Spline) if:
✓ Models 1-2 show systematic failures
✓ Complex local patterns evident
✓ Computational resources available

Simplify to Log-Linear if:
✓ β₂ ≈ 0 (no acceleration)
✓ γ₁ ≈ 0 (constant dispersion)
✓ LOO-CV strongly favors simpler model

REALITY CHECK:
If NONE of these fit well → Reconsider data-generating process entirely
```

---

## Final Prediction

**My bet**: Model 1 (Quadratic + time-varying φ) will win because:

1. EDA evidence is overwhelming (R² = 0.96, Levene's p < 0.01)
2. Visual inspection clearly shows curvature
3. Statistical tests highly significant
4. Computational feasibility is good

**Alternative scenarios**:
- If regime shift is scientifically meaningful → Model 2 wins
- If β₂ ≈ 0 → Designer 1's log-linear wins
- If computational issues → Simplify to constant φ

**The experiment will decide** via LOO-CV comparison.

---

**Files in this directory**:
- `/workspace/experiments/designer_2/proposed_models.md` - Detailed specifications
- `/workspace/experiments/designer_2/stan_model_templates.md` - Stan code
- `/workspace/experiments/designer_2/design_philosophy.md` - Rationale
- `/workspace/experiments/designer_2/model_comparison_visual.md` - This file
- `/workspace/experiments/designer_2/README.md` - Navigation guide
