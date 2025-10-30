# Visual Guide: Model Designer 3

## Model Architecture Diagrams

### Model 1: Hierarchical Changepoint
```
DATA GENERATION STORY:
====================

Timeline:  |-------- Regime 1 --------|τ|-------- Regime 2 --------|
           year = -1.67               0.3                      1.67

Parameters:
- Regime 1: β₀₁, β₁₁, φ₁
- Regime 2: β₀₂, β₁₂, φ₂
- Changepoint: τ

For each observation t:
  If year[t] < τ:
    log(μₑ) = β₀₁ + β₁₁ × yearₑ
    Cₜ ~ NegBin(μₜ, φ₁)
  Else:
    log(μₑ) = β₀₂ + β₁₂ × (yearₑ - τ)
    Cₜ ~ NegBin(μₜ, φ₂)

GRAPHICAL MODEL:
┌─────────────────────────────────────────────────┐
│ Hyperpriors (Hierarchical Structure)           │
│   μ_beta ~ N(1.0, 1.0)                         │
│   σ_beta ~ HalfN(1.0)                          │
└──────────────┬──────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Regime Parameters    │
    │  β₁₁ ~ N(μ_beta, σ)  │
    │  β₁₂ ~ N(μ_beta, σ)  │
    │  β₀₁ ~ N(3.5, 1)     │
    │  β₀₂ ~ N(5.0, 1)     │
    │  φ₁, φ₂ ~ Gamma(2,.1)│
    │  τ ~ N(0.3, 0.5)     │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Mean Function        │
    │  μₜ = f(yearₜ, τ)    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Observations         │
    │  C₁, C₂, ..., C₄₀    │
    └──────────────────────┘
```

**Key Feature:** Hierarchical structure on growth rates borrows strength across regimes

---

### Model 2: Gaussian Process
```
DATA GENERATION STORY:
====================

True function: f(year) = [smooth, unknown function]
We observe: Cₜ ~ NegBin(exp(f(yearₜ)), φ)

Structure:
  f(year) = Trend(year) + GP_deviation(year)
  Trend = β₀ + β₁×year + β₂×year²
  GP ~ N(0, K)   where K[i,j] = α² exp(-ρ²(yearᵢ - yearⱼ)²)

GRAPHICAL MODEL:
┌─────────────────────────────────────────────────┐
│ GP Hyperparameters                              │
│   α ~ HalfNormal(1)      [amplitude]           │
│   ρ ~ InvGamma(5, 5)     [length scale]        │
│   β₀, β₁, β₂ ~ Normal    [trend coefficients]  │
│   φ ~ Gamma(2, 0.1)      [dispersion]          │
└──────────────┬──────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Covariance Matrix    │
    │  K = α²·exp(-ρ²·D²)  │
    │  D = distance matrix │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Latent Function      │
    │  f ~ MVN(trend, K)   │
    │  f = [f₁,...,f₄₀]    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Observations         │
    │  Cₜ ~ NegBin(eᶠᵗ, φ) │
    └──────────────────────┘

SPATIAL CORRELATION STRUCTURE:

  Correlation
      1.0 │ ████████▓▓▓▒▒▒░░░
          │        ╲
      0.5 │         ╲____
          │              ╲___
      0.0 │__________________╲________
          └─────────────────────────────
          0    0.5   1.0   1.5   2.0  Distance

Length scale ρ controls how quickly correlation decays
```

**Key Feature:** Non-parametric - data chooses the shape of f(year)

---

### Model 3: Latent State-Space
```
DATA GENERATION STORY:
====================

Hidden State Evolution (smooth):
  θ₁ ~ N(μ₀, σ₀)                    [initial state]
  θₜ = θₜ₋₁ + γ + wₜ,  wₜ ~ N(0, σ²ᵩ) [random walk + drift]

Observations (noisy):
  Cₜ ~ NegBin(exp(θₜ), φ)

GRAPHICAL MODEL:
┌─────────────────────────────────────────────────┐
│ State Evolution Parameters                      │
│   μ₀ ~ N(3.5, 1.0)       [initial state mean]  │
│   σ₀ ~ HalfN(0.5)        [initial uncertainty] │
│   γ ~ N(0.15, 0.05)      [drift/growth rate]   │
│   σ_w ~ HalfN(0.2)       [innovation SD]       │
│   φ ~ Gamma(2, 0.1)      [observation noise]   │
└──────────────┬──────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │ Latent States (Markov Chain)        │
    │  θ₁ → θ₂ → θ₃ → ... → θ₄₀           │
    │   │     │     │           │          │
    │   ↓     ↓     ↓           ↓          │
    │   C₁    C₂    C₃    ...   C₄₀        │
    └──────────────────────────────────────┘

TEMPORAL STRUCTURE:

  Latent θₜ:  ────────────────────────  (smooth)

  Observed Cₜ: ●──●──●──●──●──●──●──●   (noisy)

  Separation of:
  - Process variance (σ²ᵩ): how much θ changes
  - Observation variance (φ): how noisy C is given θ
```

**Key Feature:** Explicitly separates signal (latent state) from noise (observations)

---

## Conceptual Comparison

```
                    FLEXIBILITY vs INTERPRETABILITY

  High │                          ╔═══════╗
  Flex │                          ║ GP(2) ║
       │                          ╚═══════╝
       │
       │        ╔═══════════════╗
  Med  │        ║ State-Space(3)║
       │        ╚═══════════════╝
       │
       │ ╔═════════════╗
  Low  │ ║ Changepoint ║
       │ ║    (1)      ║
       └─╚═════════════╝──────────────────────────
         High                            Low
              INTERPRETABILITY

Model 1: Simple story, easy to explain, rigid structure
Model 2: Flexible shape, harder to explain, data-driven
Model 3: Sophisticated mechanism, hardest to explain, many parameters
```

---

## Hypothesis Testing Framework

```
EDA FINDINGS → THREE COMPETING EXPLANATIONS
════════════════════════════════════════════

Finding 1: CUSUM shows pattern at year ≈ 0.3
├─► H1 (Model 1): Discrete changepoint EXISTS
│   Test: Is posterior for τ concentrated?
│   Evidence: If SD(τ) < 0.3, changepoint is real
│
├─► H2 (Model 2): It's just inflection point of smooth curve
│   Test: Does GP posterior show discontinuity?
│   Evidence: If f is smooth everywhere, no changepoint
│
└─► H3 (Model 3): Latent state accelerates at that time
    Test: Does state drift γ change over time?
    Evidence: If drift is constant, changepoint not needed

─────────────────────────────────────────────────

Finding 2: ACF(1) = 0.989 (extreme autocorrelation)
├─► H1 (Model 1): Artifact of smooth trend
│   Test: Do residuals still show autocorrelation?
│   Evidence: If ACF(residuals) ≈ 0, it was just trend
│
├─► H2 (Model 2): Explained by GP kernel correlation
│   Test: Does estimated length scale match ACF?
│   Evidence: If ρ ≈ 1, kernel explains correlation
│
└─► H3 (Model 3): True sequential dependence in latent state
    Test: Is σ_w small (smooth evolution)?
    Evidence: If σ_w < 0.2, state evolves smoothly

─────────────────────────────────────────────────

Finding 3: Var/Mean = 67.99 (extreme overdispersion)
├─► H1 (Model 1): Changes across regimes
│   Test: Is φ₁ ≠ φ₂?
│   Evidence: If CIs don't overlap, dispersion changes
│
├─► H2 (Model 2): Constant across time
│   Test: Is single φ sufficient?
│   Evidence: If yes, overdispersion is stable
│
└─► H3 (Model 3): Observation noise separate from state variance
    Test: Are σ_w and φ both identified?
    Evidence: If posteriors well-separated, distinction is real
```

---

## Decision Flow

```
START: Fit all three models in parallel
│
├─────────────────────────────────────────┐
│                                         │
▼                                         ▼
MODEL 1                              MODEL 2
Changepoint                          Gaussian Process
│                                         │
├─► Converges?                           ├─► Converges?
│   └─► Check Rhat, ESS                  │   └─► Check Rhat, ESS
│                                         │
├─► τ identified?                        ├─► ρ identified?
│   └─► SD(τ) < 0.5?                     │   └─► Posterior ≠ Prior?
│                                         │
└─► Regimes different?                   └─► GP adds structure?
    └─► β₁₂ > β₁₁?                           └─► f ≠ trend?

                MODEL 3
                State-Space
                │
                ├─► Converges?
                │   └─► Check Rhat, ESS
                │
                ├─► σ_w identified?
                │   └─► Posterior ≠ Prior?
                │
                └─► Latent state informative?
                    └─► corr(θ, log C) < 0.99?

─────────────────────────────────────────────────

COMPARISON PHASE:
│
├─► Compute LOO-CV for all converged models
│   └─► Identify best by ELPD
│
├─► Posterior Predictive Checks (all models)
│   ├─► Does replicated data match observed mean?
│   ├─► Does replicated data match observed variance?
│   ├─► Does replicated ACF match observed (0.989)?
│   └─► Does replicated growth match observed (8.45×)?
│
└─► Falsification Tests (model-specific)
    ├─► Model 1: Check for unrealistic discontinuities
    ├─► Model 2: Check for spurious jumps in posterior
    └─► Model 3: Check if latent state is trivial

─────────────────────────────────────────────────

DECISION:
│
├─► ΔELPD > 10 AND passes all tests?
│   └─► YES: Clear winner, use that model
│   └─► NO: Continue
│
├─► All models equivalent (ΔELPD < 4)?
│   └─► YES: Choose simplest (Model 1)
│
└─► All models fail?
    └─► Pivot to simpler approaches
        ├─► Parametric NB GLM (quadratic)
        ├─► Frequentist GEE-AR(1)
        └─► Accept N=40 is too small
```

---

## Expected Posterior Patterns

### Model 1: If Changepoint is Real
```
Posterior for τ:

  Density
    │     ╱█╲
    │    ╱███╲
    │   ╱█████╲
    │  ╱███████╲
    │ ╱█████████╲___
    └─────────────────── year
       -0.5  0.3  1.0

Concentrated at 0.3 ± 0.2
SD < 0.3 indicates strong identification
```

### Model 2: If Growth is Smooth
```
Posterior for f(year):

  log(μ)
    6 │           ╱╱╱
      │        ╱╱╱
    5 │     ╱╱╱
      │   ╱╱╱
    4 │ ╱╱╱
      │╱
    3 │─────────────── year
     -1.67      0      1.67

Smooth curve, no jumps
GP confidence bands widen at extremes
```

### Model 3: If Latent State is Meaningful
```
Latent θ vs Observed log(C):

  log(C)
    6 │  ●        ●
      │       ●     ● ●
    5 │    ●  ──●──
      │  ●  ╱      ●
    4 │ ● ╱   ●
      │  ╱ ●
    3 │╱───────────── year

  ● = observed log(C)
  ─ = latent state θ (smooth)

θ should be smoother than log(C)
```

---

## Practical Implementation Tips

### Starting Point (Day 1, Hour 1)

1. **Load data and verify:**
```python
data = pd.read_csv('/workspace/data/data_designer_3.csv')
assert len(data) == 40
assert data['C'].min() == 19
assert data['C'].max() == 272
```

2. **Prior predictive check (critical!):**
```python
# Sample from priors BEFORE looking at data
# Generate C_sim from prior
# Check: Does prior cover reasonable range?
```

3. **Fit simplest model first (Model 1):**
```python
# Get something working, then iterate
# Don't try to be perfect on first attempt
```

### Debugging Convergence Issues

```
Problem: Rhat > 1.01
├─► Solution 1: Increase warmup (try 2000)
├─► Solution 2: Increase adapt_delta (0.95 → 0.99)
├─► Solution 3: Use non-centered parameterization
└─► Solution 4: Tighten priors (make more informative)

Problem: Divergent transitions
├─► Solution 1: Increase adapt_delta to 0.99
├─► Solution 2: Increase max_treedepth to 12 or 15
├─► Solution 3: Reparameterize (check for funnels)
└─► Solution 4: Simplify model (fewer parameters)

Problem: Low ESS (< 100)
├─► Solution 1: Run longer (more iterations)
├─► Solution 2: Check for posterior correlations
├─► Solution 3: Use QR decomposition for regression
└─► Solution 4: Non-centered parameterization
```

---

## Success Checklist (Copy and Use!)

### Model 1: Changepoint
- [ ] Model compiles without errors
- [ ] Prior predictive check: Simulated data covers observed range
- [ ] Rhat < 1.01 for tau, all betas, all phis
- [ ] ESS > 400 for tau
- [ ] Divergent transitions < 1%
- [ ] Trace plots show good mixing
- [ ] Posterior for tau: SD < 0.5
- [ ] beta1_2 > beta1_1 (95% CI doesn't overlap)
- [ ] phi in range [5, 50]
- [ ] LOO: No Pareto k > 0.7
- [ ] PPC: Mean, variance, ACF all p > 0.05
- [ ] Visualization: Clear regime shift at estimated tau
- [ ] Residuals: No remaining patterns

### Model 2: Gaussian Process
- [ ] Model compiles without errors
- [ ] Prior predictive check: GP covers plausible functions
- [ ] Rhat < 1.01 for alpha, rho, phi, all betas
- [ ] ESS > 200 for alpha and rho
- [ ] Divergent transitions < 1%
- [ ] Posterior for rho: Different from prior
- [ ] rho in range [0.5, 2.0]
- [ ] alpha in range [0.3, 1.5]
- [ ] phi in range [5, 50]
- [ ] Posterior for f(year): Smooth, no jumps
- [ ] GP adds structure beyond trend
- [ ] LOO: No Pareto k > 0.7
- [ ] PPC: Mean, variance, ACF all p > 0.05

### Model 3: State-Space
- [ ] Model compiles without errors
- [ ] Prior predictive check: Reasonable state evolution
- [ ] Rhat < 1.01 for gamma, sigma_w, phi
- [ ] Rhat < 1.02 for all theta_t (relaxed for latent states)
- [ ] ESS > 100 for gamma, sigma_w, phi
- [ ] Divergent transitions < 2%
- [ ] sigma_w < 0.5 (smooth evolution)
- [ ] sigma_w posterior different from prior (identified)
- [ ] Latent theta smoother than observed log(C)
- [ ] Correlation(theta, log C) < 0.95
- [ ] phi in range [5, 50]
- [ ] LOO: No Pareto k > 0.7
- [ ] PPC: Mean, variance, ACF all p > 0.05
- [ ] One-step predictions accurate

---

**This visual guide should be used alongside the main technical documents for intuitive understanding of model structures and workflows.**
