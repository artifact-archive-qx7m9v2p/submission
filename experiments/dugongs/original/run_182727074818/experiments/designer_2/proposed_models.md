# Non-Linear Mechanistic Bayesian Models: Designer #2

**Designer Focus:** Non-linear, mechanistic, and saturation models
**Date:** 2025-10-27
**Dataset:** 27 observations, Y vs x, strong diminishing returns pattern

---

## Executive Summary

Based on EDA findings showing strong non-linear diminishing returns (R² improvement from 0.68 to 0.89 with log transform), potential change point at x≈7, and bounded Y range [1.77, 2.72], I propose **three mechanistic model classes** that explicitly capture saturation, asymptotic behavior, and potential phase transitions.

**Key Design Philosophy:**
- Focus on mechanistically interpretable functional forms
- Explicitly model asymptotic/saturation behavior
- Consider biological/physical plausibility
- Plan for computational challenges with non-linear models
- Design falsification criteria for each model

**Proposed Models (Priority Order):**
1. **Michaelis-Menten Saturation Model** - Enzyme kinetics inspired, explicit asymptote
2. **Sigmoid Growth Model (Logistic/Hill)** - S-curve with inflection point
3. **Broken-Stick with Smooth Transition** - Piecewise linear with Gaussian smoothing

---

## Model 1: Michaelis-Menten Saturation Model (PRIMARY)

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(μ_i, σ²)

μ_i = Y_min + (Y_max - Y_min) * (x_i / (K + x_i))

Parameters:
- Y_min: baseline response at x→0
- Y_max: asymptotic maximum response as x→∞
- K: half-saturation constant (x value where μ = (Y_max + Y_min)/2)
- σ: residual standard deviation
```

**Alternative Parameterization (for better geometry):**
```
μ_i = Y_min + Δ * (x_i / (K + x_i))

where Δ = Y_max - Y_min (dynamic range)
```

### Prior Distributions

**Informative priors based on data range:**

```stan
// Baseline level (observed minimum)
Y_min ~ Normal(1.8, 0.2)  // Data min = 1.77, tight prior

// Dynamic range (Y_max - Y_min)
Δ ~ Normal(1.0, 0.5)  // Observed range ≈ 0.95, allow wider
// Implies Y_max ~ 2.8 with uncertainty

// Half-saturation constant
K ~ Gamma(5, 0.5)  // Mean=10, SD≈4.5, mode≈8
// Concentrated around observed "bend" in data

// Residual SD
σ ~ HalfCauchy(0, 0.2)  // Expect small residuals given good fit
```

**Prior Justification:**
- **Y_min**: Well-constrained by low-x data (x=1.0 gives Y=1.8)
- **Δ**: Conservative estimate allowing Y_max ∈ [2.3, 3.3] (95% interval)
- **K**: Gamma prior ensures K>0, mode near x=8 where curvature changes
- **σ**: Weakly informative, residual SD should be < 0.5 given R²=0.89

### Mechanistic Interpretation

**Physical/Biological Meaning:**
- **Y_min**: Intrinsic baseline activity/response independent of x
- **Y_max**: Maximum achievable response (system capacity/saturation limit)
- **K**: Sensitivity parameter - low K means rapid saturation, high K means gradual
- **Michaelis-Menten form**: Classic enzyme kinetics, resource saturation, binding equilibria

**Possible Real-World Processes:**
1. **Enzyme kinetics**: Reaction rate vs substrate concentration
2. **Drug response**: Effect vs dose (receptor saturation)
3. **Learning curves**: Performance vs practice time
4. **Resource utilization**: Output vs resource availability
5. **Adsorption isotherms**: Amount bound vs concentration

**Why This Functional Form?**
- Data show rapid increase at low x, then plateau
- Y appears bounded above (max observed = 2.72)
- Smooth monotonic increase (no S-curve evidence)
- Two-parameter flexibility (K and Y_max) captures shape

### Falsification Criteria

**I will abandon this model if:**

1. **Posterior of Y_max is unbounded**: If Y_max posterior extends beyond [2.5, 3.5], suggests no true asymptote exists
   - **Evidence needed**: 95% credible interval for Y_max > 1.5 units wide
   - **Action**: Switch to power-law or logarithmic model (no asymptote)

2. **K posterior at boundary**: If K→0 or K→∞ (>50), model is degenerate
   - **Evidence needed**: Posterior mode at K<1 or K>50
   - **Action**: Model is unidentifiable, switch to logarithmic

3. **Systematic residual patterns**: If residuals show U-shape or trend
   - **Evidence needed**: Runs test p<0.05 or visual pattern in residuals vs x
   - **Action**: Add quadratic term or switch to segmented model

4. **Poor fit compared to log model**: If WAIC worse by >10 than simple log model
   - **Evidence needed**: ΔWAIC > 10 (strong evidence against)
   - **Action**: Accept that saturation interpretation is wrong, use log model

5. **Prior-posterior conflict**: If posterior fights strongly against prior
   - **Evidence needed**: Posterior mean >3 SDs from prior mean for any parameter
   - **Action**: Re-examine data or consider model misspecification

### Computational Considerations

**Challenges:**
- **Non-linear**: Requires HMC/NUTS, no conjugacy
- **Parameter correlation**: K and Y_max typically correlated
- **Identification**: With n=27, Y_max may be weakly identified if true asymptote is high
- **Initialization**: Need reasonable starting values to avoid local modes

**Mitigation Strategies:**
1. **Reparameterization**: Use Δ = Y_max - Y_min instead of Y_max directly
2. **Non-centered parameterization**: Consider μ_i = Y_min + Δ/(1 + K/x_i)
3. **Informative priors**: Essential for stabilizing fit with small sample
4. **Initial values**:
   - Y_min ≈ 1.8 (observed min)
   - Y_max ≈ 2.8 (slightly above observed max)
   - K ≈ 10 (midpoint of x range)
   - σ ≈ 0.2 (residual SD estimate)

**Expected Sampling Performance:**
- Target: 4 chains × 2000 iterations (1000 warmup)
- Expected: Rhat < 1.01 for all parameters
- Potential issue: ESS for Y_max may be low (<400) if data don't extend to asymptote
- Divergent transitions: Expect 0-5% if priors are informative

**Stan-Specific Tips:**
```stan
// Better geometry with this formulation:
real mu = Y_min + Delta * x[i] / (K + x[i]);

// Monitor for funnel geometry between Delta and K
// May need: target += -0.5 * square(K - 10) / 16;  // soft constraint
```

### Model Stress Tests

**Designed to Break the Model:**

1. **High-x extrapolation test**: Predict at x=100, x=1000
   - If predictions unreasonable, asymptote is wrong

2. **Leave-out high-x data**: Fit without x>20 observations
   - Compare Y_max posteriors - should be similar if model correct

3. **Synthetic data test**: Generate from log model, fit MM model
   - Should detect that asymptote doesn't exist

4. **Prior sensitivity**: Fit with Y_max ~ Normal(10, 5) (absurdly high)
   - Posterior should pull back strongly if model correct

### Expected Performance

**If Model is Correct:**
- R² ≈ 0.83-0.88 (comparable to log model)
- Y_max posterior concentrated in [2.6, 3.0]
- K posterior mode in [7, 15]
- Residuals random, no pattern
- WAIC within 5 units of log model

**If Model is Wrong:**
- Y_max posterior diffuse or at boundary
- Systematic residual curvature
- Much worse WAIC than log model
- Computational difficulties (divergences, low ESS)

---

## Model 2: Sigmoid Growth Model (Logistic/Hill Function)

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(μ_i, σ²)

// Three-parameter logistic
μ_i = Y_min + (Y_max - Y_min) / (1 + (K/x_i)^n)

// Or Hill function form:
μ_i = Y_min + (Y_max - Y_min) * x_i^n / (K^n + x_i^n)

Parameters:
- Y_min: baseline (as before)
- Y_max: asymptote (as before)
- K: midpoint (x where μ = (Y_max + Y_min)/2)
- n: Hill coefficient (cooperativity/steepness)
- σ: residual SD
```

**Why Different from Michaelis-Menten?**
- MM is special case with n=1 (hyperbolic)
- Hill with n>1 gives S-shaped curve (steeper transition)
- Hill with n<1 gives more gradual curve than MM

### Prior Distributions

```stan
Y_min ~ Normal(1.8, 0.2)  // Same as MM model
Y_max ~ Normal(2.8, 0.3)  // Slightly looser (less confident)
K ~ Gamma(5, 0.5)  // Mean=10
n ~ Gamma(2, 1)  // Mean=2, allows n ∈ [0.5, 5] mostly
σ ~ HalfCauchy(0, 0.2)
```

**Prior Justification for n:**
- n=1 recovers Michaelis-Menten
- n>1 indicates cooperative binding / threshold effect
- n<1 indicates anti-cooperative / gradual saturation
- EDA doesn't show obvious S-curve, so prior centered on n=2 but allows wide range

### Mechanistic Interpretation

**Physical Meaning of n:**
- **n=1**: Simple binding equilibrium (Michaelis-Menten)
- **n>1**: Positive cooperativity - binding facilitates more binding (e.g., hemoglobin)
- **n<1**: Negative cooperativity - binding inhibits further binding
- **n≈2-4**: Typical for cooperative systems

**When to Use This Model:**
1. If change point at x=7 reflects **rapid phase transition** rather than gradual bend
2. If underlying process has **threshold behavior** (switch-like response)
3. If data show **S-curve** (inflection point) - though EDA doesn't strongly suggest this

**Advantage over MM:**
- More flexible shape (extra parameter n)
- Can capture sharper transitions
- Common in pharmacology, systems biology

**Disadvantage:**
- One more parameter (less parsimonious)
- n may be poorly identified with n=27 observations
- More complex geometry for sampler

### Falsification Criteria

**I will abandon this model if:**

1. **n posterior includes 1 with high mass**: If 95% CI for n ⊃ [0.8, 1.2], no evidence for Hill behavior
   - **Action**: Switch to simpler MM model (n=1 fixed)

2. **n posterior is extreme**: If n<0.3 or n>10, model is fitting noise
   - **Evidence**: Posterior mode at boundary
   - **Action**: Model misspecified, try segmented model instead

3. **No improvement over MM**: If WAIC vs MM model shows ΔWAIC < 2 (equivalent)
   - **Action**: Use MM by parsimony principle

4. **Computational failure**: If >10% divergent transitions or Rhat>1.05
   - **Evidence**: Sampler struggling with geometry
   - **Action**: Model too complex for data, simplify to MM

5. **Inflection point in wrong place**: If x_inflection (where d²μ/dx²=0) is outside [5,15]
   - **Evidence**: Calculate x_inf = K*(n-1)^(1/n), check if in [5,15]
   - **Action**: Data don't support S-curve, use hyperbolic model

### Computational Considerations

**Additional Challenges Beyond MM:**
- **Parameter explosion**: 5 parameters for n=27 observations (ratio ~5.4)
- **Correlation**: K and n are strongly correlated
- **Identification**: n requires data both below and above inflection point
- **Funnel geometry**: Likely between Y_max, K, and n

**Mitigation:**
1. **Very informative priors**: Essential for n with small sample
2. **Reparameterization**: Use median_x = K instead of K directly
3. **Incremental approach**:
   - First fit MM (n=1 fixed)
   - Then fit Hill, compare posteriors
   - If n posterior overlaps 1, stick with MM

**Expected Sampling:**
- Longer warmup needed (1500 instead of 1000)
- Lower ESS expected for n (<200 acceptable)
- May need tighter priors if divergences occur

**Stan Implementation Note:**
```stan
// Numerically stable formulation:
real log_ratio = n * (log(x[i]) - log(K));
real mu = Y_min + Delta / (1 + exp(-log_ratio));

// This avoids overflow for large n or extreme x/K ratios
```

### Model Stress Tests

1. **Fix n=1, compare to MM**: Should be identical if implemented correctly
2. **Fix n=0.5, n=2, n=4**: Compare fits to see which n is best supported
3. **Simulate from log model**: Fit Hill model, check if n→1 (should detect no cooperativity)
4. **Inflate prior on n**: Use n ~ Gamma(2, 0.1) (mean=20), see if posterior ignores it

### Expected Performance

**If Sigmoid is Correct:**
- n posterior concentrated away from 1 (e.g., [1.5, 3.5])
- Steeper transition around K than MM model
- Slightly better WAIC than MM (ΔWAIC > 3)
- Residuals random

**If Sigmoid is Wrong:**
- n posterior includes 1
- Indistinguishable from MM
- Computational issues
- No WAIC improvement

**Most Likely Outcome:**
- n≈1-2 (weak cooperativity)
- No strong evidence over MM
- **Decision**: Use MM for parsimony

---

## Model 3: Broken-Stick with Smooth Transition (SEGMENTED)

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(μ_i, σ²)

// Continuous piecewise linear with smooth transition
μ_i = α + β₁*x_i + (β₂ - β₁)*S(x_i, τ, γ)

where S(x, τ, γ) is a smooth transition function:

// Option 1: Logistic smoother
S(x, τ, γ) = (x - τ) / (1 + exp(-γ*(x - τ)))

// Option 2: Gaussian CDF smoother (easier to interpret)
S(x, τ, γ) = (x - τ) * Φ((x - τ)/γ)

Parameters:
- α: intercept
- β₁: slope for x < τ (steep phase)
- β₂: slope for x > τ (flat phase)
- τ: change point location
- γ: transition smoothness (SD of transition zone)
- σ: residual SD
```

**Simplification for Identifiability:**
```
// Reduce to 5 parameters by assuming continuity
μ_i = α + β₁*min(x_i, τ) + β₂*max(0, x_i - τ) * S((x_i - τ)/γ)

// Or simple piecewise linear with enforced continuity:
μ_i = α + β₁*x_i                           if x_i ≤ τ
μ_i = α + β₁*τ + β₂*(x_i - τ)             if x_i > τ
```

**I recommend the simpler piecewise linear (discontinuous slope) to start.**

### Prior Distributions

```stan
// Intercept
α ~ Normal(1.8, 0.3)  // Near Y_min

// Slope before change point (steep)
β₁ ~ Normal(0.15, 0.1)  // Positive, bounded away from 0
// Reasoning: (2.5 - 1.8) / 7 ≈ 0.10

// Slope after change point (flat)
β₂ ~ Normal(0.02, 0.05)  // Near zero but can be positive
// Reasoning: (2.7 - 2.5) / 25 ≈ 0.01

// Change point location
τ ~ Uniform(5, 12)  // Constrained around x=7 from EDA
// Or τ ~ Normal(7, 2) T[1, 20]  // Truncated normal

// Residual SD
σ ~ HalfCauchy(0, 0.2)
```

**Prior Justification:**
- **β₁ > β₂**: Enforce that slope decreases at τ (essential for diminishing returns)
- **τ ∈ [5,12]**: EDA found best at x=7, but allow uncertainty
- **β₂ near 0**: Plateau phase has minimal slope
- Tight priors needed because change points are weakly identified with n=27

### Mechanistic Interpretation

**Physical Meaning:**
- **τ**: Critical threshold where system behavior changes
- **β₁**: Initial sensitivity (high responsiveness)
- **β₂**: Post-transition sensitivity (reduced responsiveness)
- **α + β₁*τ**: Response level at transition point

**Possible Mechanisms:**
1. **Resource exhaustion**: System saturates after consuming available resource at x=τ
2. **Kinetic regime change**: Fast kinetics to slow kinetics transition
3. **Capacity constraint**: System hits physical limit at τ
4. **Two-process model**: Fast process dominates x<τ, slow process dominates x>τ
5. **Experimental artifact**: Change in measurement method or conditions

**Why This Model?**
- EDA found 66% RSS improvement with breakpoint at x=7 (strongest signal!)
- Captures sharp transition better than smooth curves
- Interpretable if τ has mechanistic meaning
- Simple within each regime (linear)

### Falsification Criteria

**I will abandon this model if:**

1. **τ posterior is uniform**: If posterior ≈ prior (uninformative update)
   - **Evidence**: KL divergence(posterior || prior) < 0.1
   - **Action**: No change point exists, use smooth model (log or MM)

2. **τ at data boundaries**: If τ<3 or τ>25 (outside main data range)
   - **Evidence**: Posterior mode at edges
   - **Action**: Change point is artifact, use global model

3. **β₁ ≈ β₂**: If slopes are indistinguishable (95% CI overlap heavily)
   - **Evidence**: P(β₁ > β₂) < 0.7
   - **Action**: No regime change, use single-slope model

4. **Worse than smooth models**: If WAIC worse than log or MM by >5
   - **Evidence**: ΔWAIC > 5 vs best smooth model
   - **Action**: Change point is spurious, smooth transition is better

5. **Data artifacts around τ**: If observations near τ are outliers/influential
   - **Evidence**: Cook's distance high for obs near τ
   - **Action**: Breakpoint is driven by few points, not robust

6. **Posterior predictive failure**: If PPC shows discontinuity that data lack
   - **Evidence**: Replicate datasets show visible "kink" that real data don't
   - **Action**: Over-interpreting noise

### Computational Considerations

**Challenges:**
- **τ is discrete-ish**: Creates rough posterior landscape
- **Label switching**: For some data, multiple τ give similar fit
- **Identification**: With only ~13 points on each side of τ=7, weak evidence
- **Correlation**: τ and β₁, β₂ are correlated

**Mitigation:**
1. **Tight prior on τ**: Use τ ~ Normal(7, 2) T[5, 12] to stabilize
2. **Constraints**: Enforce β₁ > β₂ via:
   ```stan
   β₁ ~ Normal(0.15, 0.1) T[0, ];  // Positive
   Δβ ~ Normal(0.13, 0.1) T[0, ];  // Positive difference
   β₂ = β₁ - Δβ;  // Ensures β₂ < β₁
   ```
3. **Initialization**: Start at τ=7 (MLE from EDA)
4. **Adaptation**: May need longer warmup (1500 iterations)

**Expected Sampling:**
- Rhat < 1.01 achievable with tight priors
- ESS for τ may be low (200-400) due to discrete-like posterior
- Trace plots for τ may show sticky behavior
- 0-5% divergences expected

**Stan Note:**
```stan
// Vectorized implementation for speed
vector[N] mu;
for (i in 1:N) {
  if (x[i] <= tau) {
    mu[i] = alpha + beta1 * x[i];
  } else {
    mu[i] = alpha + beta1 * tau + beta2 * (x[i] - tau);
  }
}
```

### Model Stress Tests

1. **Grid search validation**: Compute marginal likelihood for τ ∈ [5, 12]
   - Should be concentrated, not flat

2. **Leave-τ-region-out**: Remove observations x ∈ [6, 8], refit
   - τ posterior should be wider but still recover ~7

3. **Simulate from continuous model**: Generate from log model, fit broken-stick
   - Should NOT find spurious change point (τ posterior ≈ prior)

4. **Bootstrap**: Resample data 100 times, refit
   - Check stability of τ estimates (should be in [5, 10] mostly)

### Expected Performance

**If Change Point is Real:**
- τ posterior concentrated in [6, 8]
- β₁ > β₂ with high confidence (P(β₁ > β₂) > 0.95)
- Large improvement over global log model (ΔWAIC > 5)
- Residuals random within each regime

**If Change Point is Artifact:**
- τ posterior diffuse or at boundaries
- β₁ ≈ β₂ (no slope difference)
- No WAIC improvement
- Model fits essentially reduce to linear with extra parameters

**Most Likely Outcome:**
- Moderate evidence for change point
- τ ∈ [6, 9] but with uncertainty
- **Decision**: Compare to MM model via WAIC
  - If ΔWAIC < 3, prefer MM (mechanistic + smooth)
  - If ΔWAIC > 5, change point is real and important

---

## Model Comparison Strategy

### Sequential Fitting Approach

**Stage 1: Fit All Three Models Independently**
1. Michaelis-Menten (MM)
2. Hill (sigmoid)
3. Broken-stick (BS)

**Stage 2: Initial Triage**
- Check computational diagnostics (Rhat, ESS, divergences)
- Eliminate models with sampling failures
- Compute WAIC and LOO-CV for each

**Stage 3: Scientific Comparison**
- Plot posterior predictive distributions
- Examine parameter interpretability
- Check residual patterns
- Assess prior sensitivity

**Stage 4: Decision**
- If ΔWAIC < 3: Models equivalent, choose by interpretability
- If ΔWAIC ∈ [3, 10]: Weak preference for better model, report both
- If ΔWAIC > 10: Strong preference, use best model

### Comparison Metrics

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| **WAIC** | Out-of-sample prediction | ΔWAIC > 3 meaningful |
| **LOO-CV** | Cross-validation (robust) | ΔLOO > 3 meaningful |
| **Pareto k** | Outlier detection | k > 0.7 problematic |
| **R²** | In-sample fit | For intuition only |
| **Posterior predictive p-value** | Absolute fit | p ∈ [0.2, 0.8] good |
| **Residual plots** | Misspecification | Visual check |

### Expected Ranking (My Predictions)

**Most Likely Scenario:**
1. **Michaelis-Menten** wins by WAIC
   - Reason: Right balance of fit and parsimony
   - Best mechanistic interpretation
   - n=27 sufficient for 4 parameters

2. **Hill function** equivalent to MM
   - Reason: n posterior includes 1, no cooperativity
   - Penalized for extra parameter
   - Decision: Use MM by Occam's razor

3. **Broken-stick** competitive but uncertain
   - Reason: n=27 is small for change point estimation
   - τ posterior may be diffuse
   - Decision: Interesting but not conclusive

**Alternative Scenario (if change point is real):**
1. **Broken-stick** wins
   - Strong τ posterior concentration
   - Large WAIC improvement (ΔWAIC > 8)
   - Mechanistic interpretation available (domain knowledge)

2. **MM** second
   - Good global fit but misses sharp transition
   - Smooth curve can't capture regime change

3. **Hill** third
   - Can't capture sharp enough transition
   - S-curve wrong shape for data

### Red Flags Across All Models

**Stop and Reconsider Everything If:**

1. **All models fail diagnostics**: Rhat > 1.05 for any parameter in all models
   - **Implication**: Data are pathological or priors are wrong
   - **Action**: Re-examine data for errors, try robust likelihood (Student-t)

2. **All models show systematic residual patterns**: U-shape or trend in all
   - **Implication**: Missing crucial feature (e.g., heteroscedasticity, outlier)
   - **Action**: Check for influential points, try hierarchical model for replicates

3. **Posteriors strongly conflict with priors**: All posteriors >3 SD from prior mean
   - **Implication**: Priors are wrong or data are inconsistent with assumptions
   - **Action**: Revise priors based on data, check for data errors

4. **Models all equivalent by WAIC**: ΔWAIC < 2 for all pairwise comparisons
   - **Implication**: Data are insufficient to distinguish mechanisms
   - **Action**: Report model uncertainty, use Bayesian model averaging

5. **Predictions fail smell test**: All models predict Y<0 or Y>10 for reasonable x
   - **Implication**: Fundamental misspecification
   - **Action**: Return to EDA, consider transformations of Y

### Decision Points for Major Pivots

**When to Abandon Non-Linear Approach:**

1. **If simple log model dominates**: ΔWAIC(log vs MM) > 10
   - **Decision**: Asymptote doesn't exist, use log or power-law

2. **If heteroscedasticity emerges**: Residual variance clearly increases with x
   - **Decision**: Add variance model, σ_i = σ * x_i^η

3. **If outliers dominate**: Pareto k > 0.7 for multiple observations
   - **Decision**: Switch to Student-t likelihood or robust regression

4. **If replicates show structure**: Large between-replicate variance
   - **Decision**: Hierarchical model with random effects for replicate groups

### Alternative Models (Escape Routes)

**If All Three Models Fail:**

1. **Generalized Additive Model (GAM)**:
   - Y ~ Normal(f(x), σ²) where f is a spline
   - Use mgcv or brms::s(x)
   - Advantage: Non-parametric, flexible
   - Disadvantage: Less interpretable, requires more data

2. **Gaussian Process Regression**:
   - Y ~ GP(m(x), k(x, x'))
   - Use squared exponential or Matern kernel
   - Advantage: Fully Bayesian uncertainty, smooth
   - Disadvantage: Computationally expensive for n>50

3. **Power Law Model**:
   - Y ~ Normal(α * x^β, σ²)
   - Simpler than MM, no asymptote
   - Use if data don't saturate

4. **Piecewise Polynomial (Spline)**:
   - Y ~ Normal(Σ β_j B_j(x), σ²) where B_j are basis functions
   - Middle ground between parametric and GAM
   - Use mgcv::gam() with basis="bs"

5. **Heteroscedastic Model**:
   - Y ~ Normal(μ(x), σ(x)²)
   - σ(x) = σ_0 * exp(γ * x) or σ(x) = σ_0 * x^η
   - Use if variance structure matters

---

## Implementation Roadmap

### Phase 1: Single Model Implementation (2 hours)

**Start with Michaelis-Menten:**
1. Write Stan model (30 min)
2. Simulate fake data from known parameters (30 min)
3. Fit simulated data, check recovery (30 min)
4. Fit real data (15 min)
5. Diagnostics and PPCs (15 min)

**Deliverables:**
- `mm_model.stan`
- `mm_fit_diagnostics.md`
- `mm_posterior_plots.png`

### Phase 2: Model Comparison (3 hours)

1. Fit Hill model (1 hour)
2. Fit Broken-stick model (1 hour)
3. Compute WAIC and LOO-CV (30 min)
4. Generate comparison table and plots (30 min)

**Deliverables:**
- `hill_model.stan`, `bs_model.stan`
- `model_comparison_table.csv`
- `model_comparison_plots.png`

### Phase 3: Sensitivity Analysis (2 hours)

1. Prior sensitivity: Re-fit with 2× wider priors (1 hour)
2. Outlier sensitivity: Re-fit without x=31.5 (30 min)
3. Data sensitivity: Re-fit with 10-fold CV (30 min)

**Deliverables:**
- `sensitivity_analysis.md`
- `sensitivity_plots.png`

### Phase 4: Final Report (1 hour)

**Deliverables:**
- `modeling_results.md` with:
  - Model selection justification
  - Parameter estimates and interpretations
  - Predictions with uncertainty
  - Limitations and recommendations

**Total Time: ~8 hours**

---

## Key Insights and Limitations

### What These Models Can Tell Us

1. **Is there an asymptote?** MM and Hill directly estimate Y_max
2. **Where is the transition?** K (in MM/Hill) or τ (in BS) locates change
3. **How sharp is the transition?** n (in Hill) or slope difference (in BS) quantifies
4. **What are plausible predictions?** Posterior predictive distribution gives full uncertainty

### What These Models Cannot Tell Us

1. **Causal mechanism**: Correlation ≠ causation
2. **Extrapolation far beyond data**: All models speculative for x>50
3. **Optimality**: Models describe data, don't determine "best" x value
4. **Generalization**: Results specific to this dataset, may not transfer

### Critical Assumptions

1. **Independence**: Observations are independent (replicates don't violate)
2. **Functional form**: One of three models is approximately correct
3. **Homoscedasticity**: Variance constant (EDA supports)
4. **No covariates**: x is the only predictor (may be confounded)
5. **Measurement error in Y only**: x is measured without error

### Sample Size Limitations (n=27)

**What We Can Estimate:**
- 4 parameters reliably (MM model)
- Change point location ±2 units (BS model)
- Asymptote ±0.3 units (MM model)

**What We Cannot Estimate:**
- Interactions or complex hierarchical structures
- Multiple change points
- Fine details of transition shape (n in Hill model)

### Recommendations for Future Data Collection

1. **More observations at x>25**: Better characterize asymptote
2. **More replicates at key x values**: Separate measurement from process error
3. **Observations around x=7**: Confirm or refute change point
4. **Low x values (x<1)**: Better estimate baseline Y_min
5. **Independent validation set**: Test predictions prospectively

---

## Conclusion

I propose three mechanistically interpretable Bayesian models that explicitly address the diminishing returns pattern in the data:

1. **Michaelis-Menten** (primary): Enzyme kinetics, explicit asymptote, 4 parameters
2. **Hill function** (secondary): Sigmoid/cooperative, 5 parameters, more flexible
3. **Broken-stick** (exploratory): Sharp transition at x=7, 5 parameters, regime change

**Key Principles:**
- All models have clear falsification criteria
- Computational challenges anticipated and mitigated
- Multiple stress tests designed to break models
- Decision rules specified for model selection
- Escape routes identified if all models fail

**Expected Outcome:**
- MM model likely wins by WAIC (best balance of fit and parsimony)
- Y_max ≈ 2.7-2.9, K ≈ 8-12 (posterior estimates)
- Change point evidence weak with n=27 (need more data)
- Simple log model may be equivalent (ΔWAIC < 3)

**Success Criteria:**
- Not completing all three models, but finding the model that best explains the data
- Willing to abandon all three if evidence demands it
- Honest reporting of uncertainty and limitations

**Next Steps:**
Implement MM model first, assess fit, then decide whether Hill or BS models are worth the added complexity.

---

**File Locations:**
- This document: `/workspace/experiments/designer_2/proposed_models.md`
- Stan models: `/workspace/experiments/designer_2/*.stan` (to be created)
- Analysis code: `/workspace/experiments/designer_2/fit_models.py` (to be created)
- Results: `/workspace/experiments/designer_2/results/` (to be created)
