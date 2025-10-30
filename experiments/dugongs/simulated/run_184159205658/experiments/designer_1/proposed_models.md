# Bayesian Model Design Strategy: Designer 1
## Parsimonious & Interpretable Approach

**Date:** 2025-10-27
**Designer:** Designer 1 (Parsimony Focus)
**Dataset:** N=27, Y vs x, strong nonlinear relationship
**EDA Summary:** Diminishing returns pattern, R²(log)=0.83, normal residuals, homoscedastic

---

## Executive Summary

I propose **three parsimonious Bayesian models** that balance interpretability with predictive performance. All models use Normal likelihood (EDA confirms normal residuals, homoscedasticity). My strategy prioritizes:

1. **Two-parameter models over complex ones** (N=27 is modest)
2. **Clear mechanistic interpretation** (can we explain β to stakeholders?)
3. **Graceful extrapolation** (sparse data for x>20)
4. **Computational simplicity** (linear-in-parameters preferred)

**Ranked recommendation:**
1. **Logarithmic** (PRIMARY) - Best parsimony-performance tradeoff
2. **Power Law** (ALTERNATIVE) - More flexible, still interpretable
3. **Asymptotic (Michaelis-Menten)** (THEORETICAL) - If saturation expected

---

## Philosophy: Falsification-First Mindset

**Critical principle:** Each model below includes explicit **failure criteria**. I am NOT trying to confirm these models - I am trying to BREAK them. Success means finding where they fail quickly, then pivoting.

**Red flags for ALL models:**
- Prior-posterior conflict (data fighting priors)
- Systematic residual patterns after fitting
- Posterior predictive checks show poor calibration
- LOO-CV identifies >20% of points as influential
- Extreme parameter values (|z-score| > 3 from prior)
- Divergent transitions in HMC (model misspecification)

**Global escape hatch:** If ALL three models fail posterior predictive checks, I will:
1. Question the Normal likelihood assumption (try Student-t with ν estimated)
2. Consider heteroscedastic variance models (σ depends on x or μ)
3. Explore mixture models (two distinct regimes?)
4. Investigate Gaussian Process regression (non-parametric alternative)

---

# Model 1: Logarithmic Regression (PRIMARY)

## Model Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = β₀ + β₁ · log(x_i)
```

### Priors

**Intercept (β₀):**
```
β₀ ~ Normal(1.73, 0.5)
```
- **Justification:** EDA log-fit gives β₀ ≈ 1.73. Prior centered there with SD=0.5 allows Y-intercept to range roughly [0.7, 2.7] (95% CI), covering observed Y range [1.71, 2.63].
- **Rationale:** Weakly informative - restricts to plausible region but lets data dominate.

**Slope (β₁):**
```
β₁ ~ Normal(0.28, 0.15)
```
- **Justification:** EDA log-fit gives β₁ ≈ 0.28. Prior SD=0.15 gives 95% CI [−0.02, 0.58].
- **Rationale:** Weakly positive - allows near-zero (no effect) but favors positive (monotonic increase). If β₁ < 0, model is fundamentally wrong (decreasing Y).

**Residual standard deviation (σ):**
```
σ ~ Exponential(5)
```
- **Justification:** Exponential(5) has mean=0.2, matching EDA residual SD ≈ 0.19. Mode at 0 encourages parsimony.
- **Rationale:** Weakly informative, proper prior. 95% quantile ≈ 0.6, well above observed σ.
- **Alternative:** HalfNormal(0.3) if you prefer symmetric prior.

### Functional Form

**Mean structure:**
```
μ(x) = β₀ + β₁ · log(x)
```

**Interpretation:**
- **Elasticity:** A 1% increase in x → (β₁/100)% increase in Y
- **Diminishing returns:** d²Y/dx² = -β₁/x² < 0 (concave)
- **At x=1:** μ(1) = β₀ (log(1)=0)
- **Extrapolation:** log(x) grows unboundedly - model predicts Y → ∞ as x → ∞

**Domain assumptions:**
- x > 0 (log undefined at x≤0, but our data x ∈ [1, 31.5])
- No natural upper bound on Y (may be unrealistic if Y is bounded quantity)

### Variance Modeling

**Structure:** Constant variance (homoscedastic)
```
Var(Y|x) = σ² (constant)
```

**Justification:**
- EDA Breusch-Pagan test p=0.546 (fail to reject homoscedasticity)
- EDA Levene test p=0.370 (variance equality across segments)
- Sample variance by segment: σ(low)=0.13, σ(mid)=0.10, σ(high)=0.19
  - Differences not statistically significant
  - High-x variance larger but only 3 obs, high sampling variability

**When to switch:** If posterior predictive checks show residual variance increases with x, refit with:
```
σ_i = σ₀ · exp(γ · x_i)  # Exponential heteroscedasticity
```

### Implementation Strategy

**Framework:** PyMC (preferred) or Stan

**PyMC code outline:**
```python
import pymc as pm
import numpy as np

with pm.Model() as log_model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.73, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.28, sigma=0.15)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Mean structure
    mu = beta_0 + beta_1 * np.log(x_obs)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_obs)

    # Posterior sampling
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
```

**Stan alternative:** Linear-in-parameters → efficient HMC sampling. No reparameterization needed.

**Computational concerns:**
- **None expected** - simple linear regression in transformed space
- Expect R-hat < 1.01, ESS > 1000 per chain
- If divergences occur → problem with data or likelihood, not model structure

---

## Success Criteria

**This model is ADEQUATE if:**

1. **Convergence:** All R-hat < 1.01, ESS > 400 per parameter
2. **Residual diagnostics:**
   - No systematic patterns in (y_obs - y_pred) vs x
   - Residuals approximately normal (Q-Q plot)
   - No remaining autocorrelation (Durbin-Watson ≈ 2)
3. **Posterior predictive checks:**
   - Observed Y within 95% prediction interval
   - ppc_stat(mean): observed mean ∈ 95% of replicated means
   - ppc_stat(sd): observed SD ∈ 95% of replicated SDs
4. **Parameter credibility:**
   - β₁ > 0 (positive relationship, 95% CI excludes 0)
   - σ ∈ [0.1, 0.3] (reasonable residual SD)
5. **Predictive accuracy:**
   - LOO-CV RMSE < 0.25 (better than marginal SD of Y = 0.28)
   - No observations flagged as highly influential (Pareto-k < 0.7)
6. **Out-of-sample:**
   - If using train/test split: test RMSE within 20% of train RMSE

---

## Failure Criteria (When to ABANDON)

**I will REJECT this model if:**

1. **Prior-posterior conflict:**
   - Posterior mean for β₁ < 0 (would imply decreasing relationship)
   - Posterior 95% CI for β₁ includes 0 with high probability (>40%)
   - σ posterior mean > 0.4 (implying R² < 0.3, worse than EDA)

2. **Systematic residual patterns:**
   - Clear quadratic or higher-order pattern in residual plot
   - Residual SD increases substantially with x (heteroscedasticity)
   - Residual autocorrelation persists (DW < 1.2 or > 2.8)

3. **Poor posterior predictive performance:**
   - >20% of observed Y fall outside 95% posterior prediction intervals
   - Observed Y systematically at edge of prediction intervals (calibration failure)
   - ppc_stat checks fail for >2 test statistics

4. **Influential observations:**
   - >5 observations (>18%) with LOO Pareto-k > 0.7
   - Model fit changes drastically when single observation removed
   - High-x points (x>20) have disproportionate influence

5. **Computational red flags:**
   - Divergent transitions (even after tuning) → model misspecified
   - R-hat > 1.05 → chains exploring different modes
   - Bimodal posteriors → model unidentified or multiple solutions

6. **Mechanistic implausibility:**
   - Model predicts Y > 3.5 for x=50 (unbounded growth may be unrealistic)
   - Extrapolation beyond x=31.5 shows absurd behavior

**If any of these occur, PIVOT to:**
- Model 2 (Power Law) - more flexible functional form
- Model 3 (Asymptotic) - if unbounded growth is the issue
- Gaussian Process - if residual patterns are complex

---

## Expected Challenges

1. **Extrapolation uncertainty:**
   - Only 3 observations x>20
   - Log(x) continues growing - may overpredict at high x
   - **Mitigation:** Wide credible intervals, explicit warnings

2. **Model simplicity:**
   - Two parameters may be insufficient
   - Could miss curvature not captured by log
   - **Mitigation:** Posterior predictive checks will reveal

3. **Asymptotic behavior:**
   - Real Y may plateau, log(x) does not
   - **Detection:** Check if residuals for x>20 are systematically negative
   - **Response:** Switch to Model 3 (asymptotic)

---

## Theoretical Justification

**Why logarithmic?**

1. **Diminishing returns:** Common in economics (utility), psychology (learning), biology (dose-response at low doses)
2. **Constant elasticity:** % change in Y proportional to % change in x
3. **EDA support:** R²=0.83, best simple functional form
4. **Occam's razor:** Fewest parameters that adequately fit data
5. **Interpretability:** β₁ has clear meaning (elasticity coefficient)

**Domain assumptions:**
- Process has no hard ceiling (Y can grow indefinitely)
- Marginal effect decreases but never reaches zero
- No inflection point (monotonic curvature)

**When this fails mechanistically:**
- If Y has biological/physical upper limit → use asymptotic model
- If process has inflection point → use logistic or Gompertz
- If marginal returns increase then decrease → use quadratic

---

## Model Checking Plan

**Step 1: Convergence diagnostics**
- Plot trace plots for all parameters
- Check R-hat, ESS
- Examine pairplot for parameter correlations

**Step 2: Residual analysis**
- Plot residuals vs fitted values
- Plot residuals vs x
- Q-Q plot for normality
- Test for autocorrelation

**Step 3: Posterior predictive checks**
- Generate 1000 datasets from posterior predictive
- Compare observed vs replicated: mean, SD, min, max, quantiles
- Visual: overlay replicated Y vs x curves on data

**Step 4: LOO-CV**
- Compute LOO-ELPD
- Check Pareto-k diagnostics
- Identify influential observations

**Step 5: Parameter interpretation**
- Extract posterior summaries
- Compare to priors (prior-posterior plot)
- Check if estimates make domain sense

---

# Model 2: Power Law Regression (ALTERNATIVE)

## Model Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = β₀ + β₁ · x_i^β₂
```

### Priors

**Intercept (β₀):**
```
β₀ ~ Normal(1.8, 0.5)
```
- **Justification:** Lower than log model since power term will be smaller at x=1
- Near observed Y minimum ≈ 1.71

**Scale (β₁):**
```
β₁ ~ Normal(0.5, 0.3)
```
- **Justification:** Controls magnitude of power term contribution
- Must be positive for monotonic increase (though prior allows negative)
- Wide prior due to uncertainty in power function scaling

**Exponent (β₂):**
```
β₂ ~ Normal(0.3, 0.2)
```
- **Justification:** Exponent < 1 gives diminishing returns (concave)
- Prior centered at 0.3 (analogous to log which is limit as β₂→0)
- 95% CI ≈ [−0.1, 0.7] allows exploration
- **Key constraint:** Need β₂ < 1 for concavity

**Residual standard deviation (σ):**
```
σ ~ Exponential(5)
```
- Same as Model 1

### Functional Form

**Mean structure:**
```
μ(x) = β₀ + β₁ · x^β₂
```

**Interpretation:**
- **At x=1:** μ(1) = β₀ + β₁
- **Curvature:** Controlled by β₂
  - β₂ < 1: diminishing returns (concave)
  - β₂ = 1: linear
  - β₂ > 1: accelerating returns (convex)
- **Marginal effect:** dμ/dx = β₁ · β₂ · x^(β₂-1)
  - Decreases with x if β₂ < 1

**Advantages over log:**
- More flexible (3 parameters vs 2)
- Can handle x near 0 better (if needed)
- Nests linear model (β₂=1)

**Disadvantages:**
- One more parameter (overfitting risk with N=27)
- Less interpretable (what is x^0.3?)
- Extrapolation still unbounded

### Variance Modeling

Same as Model 1: constant variance σ²

### Implementation Strategy

**Framework:** PyMC (Stan would work but PyMC preferred for flexibility)

**PyMC code outline:**
```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as power_model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.8, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.5, sigma=0.3)
    beta_2 = pm.Normal('beta_2', mu=0.3, sigma=0.2)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Mean structure
    mu = beta_0 + beta_1 * pt.power(x_obs, beta_2)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_obs)

    # Sampling (may need tuning)
    trace = pm.sample(2000, tune=2000, chains=4, target_accept=0.99)
```

**Computational concerns:**
- **Nonlinear in β₂** → may have geometry challenges
- Possible parameter correlation (β₁ and β₂ trade off)
- May need longer tuning (tune=2000) and higher target_accept
- Watch for divergences

**Reparameterization if needed:**
If sampling struggles, try centered parameterization or non-centered for β₁, β₂.

---

## Success Criteria

**This model is ADEQUATE if:**

1. All criteria from Model 1 (convergence, residuals, PPC, etc.)
2. **Additional:**
   - β₂ ∈ (0, 1) with >80% posterior probability (concave function)
   - LOO-CV improvement over Model 1: ΔELPD > 3 (substantial)
   - Parameter correlation |ρ(β₁,β₂)| < 0.85 (identifiable)

---

## Failure Criteria (When to ABANDON)

**I will REJECT this model if:**

1. **All Model 1 failure criteria apply**
2. **Additional red flags:**
   - Posterior for β₂ includes 1 with high probability (→ linear, simpler model better)
   - Extreme parameter correlation |ρ(β₁,β₂)| > 0.95 (unidentifiable)
   - Divergent transitions persist even with target_accept=0.99
   - LOO-CV shows NO improvement over Model 1 (overfitting)
   - WAIC or LOO favors simpler Model 1 (parsimony penalty)

3. **Computational failure:**
   - Cannot achieve R-hat < 1.05 even with extended sampling
   - Posterior has multiple modes (different parameterizations fit equally)
   - Sampling time >10x Model 1 without accuracy gain

**If these occur, REVERT to Model 1 (log) or try Model 3 (asymptotic)**

---

## Expected Challenges

1. **Parameter identification:**
   - β₁ and β₂ trade off (scale-shape confounding)
   - May need informative priors or constraints
   - **Mitigation:** Check parameter correlation matrix

2. **Computational cost:**
   - Nonlinear model → slower sampling
   - May need 4000+ effective samples for stable estimates
   - **Mitigation:** Parallel chains, longer runs

3. **Overfitting:**
   - N=27 with 3 parameters → concern
   - **Detection:** LOO-CV Pareto-k diagnostics
   - **Mitigation:** Compare WAIC to Model 1

4. **Interpretation:**
   - Hard to explain β₂=0.31 to stakeholders
   - **Mitigation:** Focus on predicted curve shape, not parameters

---

## Theoretical Justification

**Why power law?**

1. **Empirical ubiquity:** Power laws common in nature (scaling laws, allometry, psychophysics)
2. **Flexibility:** Includes linear (β₂=1), log-like (β₂≈0), square-root (β₂=0.5) as special cases
3. **Scale invariance:** Proportional changes preserved (useful if units arbitrary)
4. **Biological plausibility:** Many dose-response relationships follow power laws

**Domain assumptions:**
- No upper bound on Y (like log model)
- Smooth, monotonic relationship
- Curvature constant on log-log scale

**When this fails mechanistically:**
- If relationship has asymptote → Model 3
- If relationship has inflection → logistic/Gompertz
- If simple log is adequate → Occam's razor favors Model 1

---

## Model Checking Plan

Same 5-step plan as Model 1, plus:

**Step 6: Compare to Model 1**
- LOO-CV comparison: compute ΔELPD and SE
- Check if added complexity justified
- Visualize both models' posterior predictions side-by-side

---

# Model 3: Asymptotic Regression (Michaelis-Menten) (THEORETICAL)

## Model Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = Y_min + (Y_max - Y_min) · x_i / (K + x_i)
```

**Alternative parameterization:**
```
μ_i = Y_max · x_i / (K + x_i)  # If Y_min = 0 assumed
```

**I'll use full 3-parameter form:**
```
μ_i = Y_min + Y_range · x_i / (K + x_i)
where Y_range = Y_max - Y_min
```

### Priors

**Minimum response (Y_min):**
```
Y_min ~ Normal(1.7, 0.3)
```
- **Justification:** Observed min(Y) ≈ 1.71
- Prior centered there, SD=0.3 allows [1.1, 2.3] (95% CI)
- Represents Y as x → 0

**Response range (Y_range):**
```
Y_range ~ Normal(0.9, 0.3)
```
- **Justification:** Observed range ≈ 2.63 - 1.71 = 0.92
- Positive prior (could use LogNormal to enforce Y_range > 0)
- Y_max = Y_min + Y_range ≈ 2.6

**Half-saturation constant (K):**
```
K ~ LogNormal(log(5), 1)
```
- **Justification:** K is x-value where μ = (Y_min + Y_max)/2
- Log-scale prior due to positive constraint and large uncertainty
- Mean ≈ 5, median = e^log(5) = 5
- 95% CI ≈ [0.7, 37] (very wide, data-driven)
- **Interpretation:** Smaller K → faster approach to asymptote

**Residual standard deviation (σ):**
```
σ ~ Exponential(5)
```
- Same as previous models

### Functional Form

**Mean structure:**
```
μ(x) = Y_min + Y_range · x / (K + x)
```

**Key properties:**
- **As x → 0:** μ → Y_min
- **As x → ∞:** μ → Y_min + Y_range = Y_max (asymptote)
- **At x = K:** μ = Y_min + Y_range/2 (half-max)
- **Monotonic increase:** dμ/dx = Y_range · K / (K + x)² > 0
- **Concave:** d²μ/dx² < 0 (diminishing returns)

**Interpretation:**
- Y_min: baseline response (x near 0)
- Y_max: maximum achievable response
- K: x-value for half-maximal effect (smaller = faster saturation)

**Biological meaning (if applicable):**
- Enzyme kinetics: substrate concentration (x) vs reaction rate (Y)
- Learning curves: practice amount (x) vs skill (Y)
- Dose-response: drug dose (x) vs effect (Y)

### Variance Modeling

Same as Models 1 and 2: constant variance σ²

### Implementation Strategy

**Framework:** PyMC preferred (Stan also works well)

**PyMC code outline:**
```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as asymptotic_model:
    # Priors
    Y_min = pm.Normal('Y_min', mu=1.7, sigma=0.3)
    Y_range = pm.Normal('Y_range', mu=0.9, sigma=0.3)
    K = pm.LogNormal('K', mu=np.log(5), sigma=1.0)
    sigma = pm.Exponential('sigma', lam=5.0)

    # Derived quantity
    Y_max = pm.Deterministic('Y_max', Y_min + Y_range)

    # Mean structure (Michaelis-Menten)
    mu = Y_min + Y_range * x_obs / (K + x_obs)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_obs)

    # Sampling (may need tuning)
    trace = pm.sample(2000, tune=2000, chains=4, target_accept=0.99)
```

**Computational concerns:**
- **Highly nonlinear** → most challenging of three models
- Parameter correlation: Y_min, Y_range, K can trade off
- K on log-scale → may need careful initialization
- **Reparameterization option:** Use non-centered for K if needed

**Initialization strategy:**
- Initialize K near EDA asymptotic fit value (K ≈ 0.64, but prior broader)
- May need multiple chains to ensure global mode found

---

## Success Criteria

**This model is ADEQUATE if:**

1. All criteria from Model 1 (convergence, residuals, PPC, etc.)
2. **Additional:**
   - Y_max ∈ [2.4, 2.8] with >80% probability (matches data range)
   - K posterior not dominated by prior (data informative)
   - Asymptotic behavior matches high-x data (x>20: Y plateaus)
   - LOO-CV comparable to or better than Models 1 and 2

3. **Mechanistic plausibility:**
   - Y_max > max(Y_obs) with high probability (asymptote not yet reached)
   - OR Y_max ≈ max(Y_obs) (asymptote near data ceiling)
   - K > 0 (always satisfied by LogNormal prior)

---

## Failure Criteria (When to ABANDON)

**I will REJECT this model if:**

1. **All Model 1 failure criteria apply**
2. **Additional red flags:**
   - K posterior extremely wide (data not informative about saturation)
   - Y_max posterior unbounded (effectively no asymptote)
   - Y_range posterior includes 0 (model reduces to constant)
   - Parameter correlation matrix near-singular (unidentifiable)

3. **Computational failure:**
   - Persistent divergent transitions (>5% even with tuning)
   - Rhat > 1.05 for K or Y_range
   - Bimodal posterior (multiple local optima)
   - Sampling fails to converge after 10,000 iterations

4. **Model comparison:**
   - LOO-CV substantially worse than simpler models (ΔELPD < -10)
   - Pareto-k warnings for >30% of observations
   - WAIC strongly penalizes complexity

**If these occur:**
- Simpler is better → revert to Model 1 (log) or Model 2 (power)
- Or try non-parametric: Gaussian Process

---

## Expected Challenges

1. **Parameter identification:**
   - N=27 may be insufficient to estimate 3 nonlinear parameters reliably
   - K (half-saturation) hardest to pin down
   - **Mitigation:** Informative priors, sensitivity analysis

2. **Computational difficulty:**
   - Most complex model → longest sampling time
   - Nonlinear parameter interactions
   - **Mitigation:** Extended tuning, reparameterization

3. **Data sparsity at extremes:**
   - Only 3 observations x>20 → asymptote poorly constrained
   - **Detection:** Y_max posterior very wide
   - **Mitigation:** May need more data or stronger priors

4. **Mechanistic assumption:**
   - Assumes saturation exists (may not be true)
   - **Detection:** Y_max unbounded or K → ∞
   - **Response:** Switch to model without asymptote

---

## Theoretical Justification

**Why asymptotic (Michaelis-Menten)?**

1. **Biological ubiquity:** Enzyme kinetics, receptor binding, learning curves
2. **Bounded growth:** Realistic if Y has physical/biological ceiling
3. **Clear interpretation:** Y_max = maximum capacity, K = efficiency
4. **EDA support:** R²=0.82, reasonable fit
5. **Extrapolation:** Graceful (Y bounded as x → ∞)

**Domain assumptions:**
- Y has upper limit (saturation)
- Process reaches equilibrium at high x
- No overshoot or oscillation

**When this fails mechanistically:**
- If Y grows without bound → Models 1 or 2 better
- If saturation not yet approached in data → unidentifiable
- If relationship has inflection → logistic

---

## Model Checking Plan

Same 5-step plan as Models 1 and 2, plus:

**Step 6: Asymptotic behavior check**
- Plot posterior draws of μ(x) for x ∈ [0, 100]
- Verify asymptote stabilizes
- Check if Y_max posterior informed by data vs prior

**Step 7: Compare all three models**
- LOO-CV comparison table
- WAIC comparison
- Posterior predictive overlay (all models on same plot)
- Select based on parsimony, fit, and interpretability

---

# Model Prioritization and Strategy

## Ranked Recommendation

### 1. Logarithmic (PRIMARY) - Fit First

**Rationale:**
- **Parsimony:** Only 2 parameters (β₀, β₁), lowest overfitting risk
- **Interpretability:** Clear elasticity interpretation
- **EDA support:** R²=0.83, strong performance
- **Computational ease:** Linear model, fast sampling
- **Benchmark:** Establishes baseline for comparison

**Expected outcome:** Adequate fit, will serve as reference model even if others explored

**When to stop here:** If posterior predictive checks pass and LOO-CV shows strong performance

---

### 2. Power Law (ALTERNATIVE) - Explore if Needed

**Rationale:**
- **Flexibility:** 3 parameters, more expressive than log
- **Generality:** Nests linear, approaches log as β₂ → 0
- **Exploration:** Worth trying if Model 1 shows systematic residual patterns

**When to fit:**
- Model 1 posterior predictive checks show systematic bias
- Residual plot shows pattern not captured by log
- Scientific interest in exact functional form

**When to skip:**
- Model 1 adequate (Occam's razor)
- N=27 too small for reliable 3-parameter nonlinear estimation

---

### 3. Asymptotic (THEORETICAL) - Only if Mechanism Demands

**Rationale:**
- **Mechanistic:** If domain knowledge suggests saturation
- **Extrapolation:** Bounded predictions for x > 31.5
- **Interpretation:** Y_max has physical meaning

**When to fit:**
- Model 1 or 2 predict implausible unbounded Y
- High-x observations (x>20) show clear plateau
- Scientific theory predicts saturation

**When to skip:**
- No evidence of asymptote in data (all models similar fit)
- Computational cost not justified
- Data insufficient to identify K reliably

---

## Decision Tree

```
START: Fit Model 1 (Logarithmic)
    |
    V
Check convergence, residuals, PPC
    |
    +-- PASS --> Report Model 1, check LOO-CV
    |              |
    |              V
    |           LOO-CV good? --> DONE (use Model 1)
    |              |
    |              NO (Pareto-k warnings)
    |              |
    |              V
    |           Investigate influential points
    |
    +-- FAIL --> Systematic residual pattern?
                    |
                    YES --> Fit Model 2 (Power Law)
                    |          |
                    |          V
                    |       Compare LOO-CV: Model 1 vs Model 2
                    |          |
                    |          ΔELPD > 3? --> Use Model 2
                    |          |
                    |          NO --> Stick with Model 1 (simpler)
                    |
                    NO --> Check assumptions (normality, variance)
                              |
                              V
                           Heteroscedasticity? --> Refit with σ(x)
                              |
                              Outliers? --> Robust likelihood (Student-t)
                              |
                              Still failing? --> Fit Model 3 or GP
```

---

## Global Stopping Rules

**I will stop model development and declare SUCCESS when:**
1. Current model passes all convergence diagnostics (R-hat, ESS, trace plots)
2. Posterior predictive checks show good calibration (>90% coverage)
3. LOO-CV shows predictive accuracy better than marginal (RMSE < 0.25)
4. Residuals show no systematic patterns
5. Parameter estimates make scientific sense

**I will stop model development and declare FAILURE when:**
1. All three models show severe prior-posterior conflict
2. All three models fail posterior predictive checks
3. Computational issues persist across all models (divergences, non-convergence)
4. Data quality issues discovered (e.g., measurement error, confounding)

**In case of global failure, PIVOT to:**
- Gaussian Process regression (non-parametric, flexible)
- Robust likelihood (Student-t with estimated df)
- Hierarchical variance model (heteroscedasticity)
- Mixture models (two distinct regimes in data?)
- **Last resort:** Question the data collection process

---

# Implementation Plan

## Phase 1: Model 1 (Days 1-2)

1. **Setup:**
   - Load data, exploratory plots
   - Implement Model 1 in PyMC
   - Run prior predictive checks

2. **Fitting:**
   - Sample posterior (4 chains × 2000 draws)
   - Check convergence diagnostics
   - Examine trace plots

3. **Validation:**
   - Residual analysis
   - Posterior predictive checks
   - LOO-CV computation

4. **Decision:**
   - If adequate → Report and STOP
   - If inadequate → Proceed to Phase 2

## Phase 2: Model Comparison (Days 3-4)

5. **Fit Model 2:**
   - Implement power law
   - Sample posterior (longer tuning expected)
   - Check convergence

6. **Compare Models 1 & 2:**
   - LOO-CV ΔELPD and SE
   - WAIC comparison
   - Posterior predictive overlay

7. **Decision:**
   - If Model 2 substantially better → Use it
   - If comparable → Use Model 1 (simpler)
   - If both inadequate → Proceed to Phase 3

## Phase 3: Asymptotic Model (Days 5-6, if needed)

8. **Fit Model 3:**
   - Implement Michaelis-Menten
   - Extended sampling (may need 4000+ iterations)
   - Check convergence carefully

9. **Three-way comparison:**
   - LOO-CV across all models
   - Scientific interpretability
   - Extrapolation behavior

10. **Final selection:**
    - Choose based on parsimony, fit, and domain knowledge

## Phase 4: Reporting (Day 7)

11. **Documentation:**
    - Report chosen model with full diagnostics
    - Posterior summaries and credible intervals
    - Predictive plots with uncertainty
    - Limitations and caveats (especially x>20)

---

# Stress Tests and Red Flags

## Stress Test 1: Extrapolation Check

**Test:** Predict Y for x ∈ [40, 60] (far beyond data)

**Expected outcomes:**
- Model 1 (log): Wide credible intervals, unbounded growth
- Model 2 (power): Wide credible intervals, unbounded growth
- Model 3 (asymptotic): Narrow CI near Y_max (bounded)

**Red flag:** If intervals absurdly wide (>5× data range) or predictions implausible

**Response:** Explicitly document uncertainty, recommend data collection at high x

---

## Stress Test 2: Leave-High-X-Out CV

**Test:** Hold out all x>20 (N=3), fit on x≤20 (N=24), predict held-out

**Purpose:** Assess extrapolation quality

**Expected outcome:** Poor prediction (high RMSE, wide intervals)

**Red flag:** If model predicts held-out points with high confidence (overconfident)

**Response:** Downweight importance of high-x fit, focus on x≤20 region

---

## Stress Test 3: Prior Sensitivity Analysis

**Test:** Refit Model 1 with:
- Vague priors: β₀ ~ Normal(0, 10), β₁ ~ Normal(0, 10), σ ~ Exponential(0.1)
- Informative priors: β₀ ~ Normal(1.73, 0.2), β₁ ~ Normal(0.28, 0.05), σ ~ Exponential(10)

**Expected outcome:** Posteriors similar across prior choices (data-dominated)

**Red flag:** Posterior dramatically changes with prior (data not informative)

**Response:** Report sensitivity, acknowledge prior influence, collect more data

---

## Stress Test 4: Posterior Predictive Surprise

**Test:** Generate 1000 replicated datasets, compute:
- P(max(Y_rep) > max(Y_obs))
- P(min(Y_rep) < min(Y_obs))
- P(range(Y_rep) > range(Y_obs))

**Expected outcome:** All probabilities ≈ 0.5 (observed data typical)

**Red flag:** Probabilities < 0.05 or > 0.95 (model surprised by data)

**Response:** Investigate which feature of data is unusual, refit or question data

---

## Red Flag Summary

**Computational:**
- Divergent transitions (model misspecification)
- R-hat > 1.05 (non-convergence)
- ESS < 400 (inefficient sampling)
- Bimodal posteriors (unidentified)

**Statistical:**
- Systematic residual patterns (wrong functional form)
- Failed posterior predictive checks (poor calibration)
- LOO Pareto-k > 0.7 for >20% points (influential obs)
- Prior-posterior conflict (data fighting prior)

**Scientific:**
- Parameter estimates nonsensical (e.g., negative slope)
- Extrapolation absurd (e.g., Y=10 at x=50)
- Predictions inconsistent with domain knowledge

**Any of these triggers model revision or pivot**

---

# Summary: Designer 1 Philosophy

**Core principles:**
1. **Start simple** (2-parameter log model)
2. **Add complexity only if justified** (3-parameter models tentative)
3. **Falsify aggressively** (try to break models, not confirm)
4. **Interpret clearly** (stakeholders must understand)
5. **Acknowledge uncertainty** (especially x>20 extrapolation)
6. **Pivot quickly** (don't force failing models)

**Success metric:** Find simplest model that passes all checks, not most complex model that fits

**Expected deliverable:** Logarithmic model (Model 1) with comprehensive diagnostics, potentially supplemented by Model 2 comparison if needed

---

*End of Model Design Document - Designer 1*
