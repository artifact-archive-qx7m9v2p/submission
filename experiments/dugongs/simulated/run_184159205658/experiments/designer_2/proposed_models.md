# Bayesian Model Design Proposals - Designer 2
## Independent Assessment of Y-x Relationship

**Date:** 2025-10-27
**Designer:** Agent 2 - Bayesian Modeling Strategist
**Data:** N=27 observations, Y vs x relationship with saturation/plateau pattern
**EDA Source:** `/workspace/eda/eda_report.md`

---

## Executive Summary

Based on EDA findings, I propose **three theoretically distinct Bayesian model classes** to capture the saturation relationship between Y and x. Each model represents a fundamentally different hypothesis about the data-generating process:

1. **Michaelis-Menten Saturation Model** (PRIMARY) - Biological/enzymatic saturation with clear asymptote
2. **Power-Law with Saturation** (ALTERNATIVE 1) - Gradual diminishing returns without hard asymptote
3. **Exponential Saturation Model** (ALTERNATIVE 2) - Exponential approach to ceiling

These models differ from simple logarithmic/polynomial fits by explicitly modeling **mechanistic saturation processes** with interpretable parameters. Each includes falsification criteria and computational considerations.

**Key Design Philosophy:**
- Focus on mechanistic interpretability over pure fit
- Design models that can FAIL in measurable ways
- Prioritize extrapolation behavior (sparse data x>20)
- All models use Stan for robust HMC sampling

---

## Critical Assessment: EDA Findings and Potential Pitfalls

### What the EDA Got Right
- Normal residuals, constant variance (good for Gaussian likelihood)
- Clear saturation pattern (Y plateaus ~2.5-2.6)
- Sparse high-x data (only 3 points x>20)
- Strong signal (r=0.72, R²=0.82-0.86 for nonlinear fits)

### What Could Mislead Us
1. **Plateau may be artifact** - Only 8 points in "high x" region (x>15). Could be sampling noise rather than true asymptote.
2. **Logarithmic fit (R²=0.83) assumes unbounded growth** - Data shows Y∈[1.7, 2.6], suggesting possible ceiling. Log model would predict Y→∞ as x→∞, which may be implausible.
3. **Variance homogeneity uncertain at extremes** - Only 3 points x>20. Cannot reliably test if variance increases at high x.
4. **Possible autocorrelation (DW=0.663)** - If data has temporal/spatial structure, independence assumption violated.

### Competing Hypotheses About Data Generation
1. **True asymptotic saturation** (e.g., enzymatic reaction reaching max capacity)
2. **Logarithmic diminishing returns** (no hard ceiling, just slowing growth)
3. **Plateau is coincidence** (true relationship continues increasing beyond observed range)

**My stance:** Models 1-3 test hypothesis 1 (asymptotic saturation) with different functional forms. I will explicitly state what evidence would force me to switch to hypothesis 2 (logarithmic) or 3 (higher-order polynomial).

---

## Model 1: Michaelis-Menten Saturation Model (PRIMARY RECOMMENDATION)

### Theoretical Motivation
The Michaelis-Menten equation describes enzyme kinetics, receptor-ligand binding, learning curves, and dose-response relationships. It has two interpretable parameters:
- **Y_max**: Maximum achievable response (asymptote)
- **K**: Half-saturation constant (x value where Y = Y_max/2)

**Why this form?**
- Data shows Y increasing rapidly at low x, then flattening near 2.5-2.6
- At x=0.64 (EDA-reported K estimate), we'd expect Y ≈ 1.29 (half of max 2.59)
- Observed Y at x=1.0 is 1.86, at x=31.5 is 2.52 - consistent with saturation

**Domain applicability:** Any process with limiting resource/capacity (nutrient uptake, market saturation, learning).

### Complete Model Specification

#### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = Y_max * x_i / (K + x_i)
```

#### Priors

**Y_max (asymptote):**
```
Y_max ~ Normal(2.6, 0.3)
```
- **Justification:** Maximum observed Y is 2.63. Prior centered at 2.6 with SD=0.3 allows Y_max ∈ [2.0, 3.2] at 95% coverage. Weakly informative - data can easily override if true max is higher.
- **Constraint:** Y_max > max(Y_observed) should hold. If posterior concentrates below 2.63, model is falsified.

**K (half-saturation constant):**
```
K ~ Normal(5, 5) truncated at K > 0
```
- **Justification:** EDA found K≈0.64, but this is suspiciously low (Y already near saturation at x=1). Prior centered at 5 (middle of x range) with wide SD allows data to pull toward 0.64 if justified. Positive constraint required by model definition.
- **Alternative parameterization:** Use log(K) to improve sampling (below).

**σ (residual SD):**
```
σ ~ HalfNormal(0.25)
```
- **Justification:** EDA residual SD ≈ 0.19. HalfNormal(0.25) gives prior median ≈ 0.17, 95th percentile ≈ 0.50. Conservative - allows larger error than observed.

#### Reparameterization for MCMC Efficiency

**Problem:** K and Y_max are highly correlated in Michaelis-Menten model. HMC can struggle with funnel geometries.

**Solution:** Centered parameterization with log(K):
```stan
parameters {
  real<lower=0> Y_max;
  real log_K;      // K = exp(log_K)
  real<lower=0> sigma;
}

transformed parameters {
  real K = exp(log_K);
}

model {
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = Y_max * x[i] / (K + x[i]);
  }

  // Priors
  Y_max ~ normal(2.6, 0.3);
  log_K ~ normal(log(5), 1);  // Implies K ~ lognormal(log(5), 1)
  sigma ~ normal(0, 0.25);

  // Likelihood
  Y ~ normal(mu, sigma);
}
```

**Note:** log_K ~ Normal(log(5), 1) implies prior median K ≈ 5, 95% interval [0.7, 37], allowing EDA estimate K=0.64 while regularizing.

### Implementation Strategy

**Framework:** Stan (preferred for HMC) or PyMC (more accessible)

**Stan advantages:**
- Superior HMC tuning (NUTS algorithm)
- Better handling of complex geometries
- Comprehensive diagnostics (divergences, E-BFMI)

**Sampling parameters:**
- 4 chains, 2000 iterations each (1000 warmup)
- `adapt_delta = 0.95` (conservative to avoid divergences)
- `max_treedepth = 12` (allow deeper trajectories)

**Diagnostics to monitor:**
- R-hat < 1.01 for all parameters
- ESS_bulk > 400 (N*4*0.5/4 = 400 is baseline)
- ESS_tail > 400 (tail behavior critical for credible intervals)
- No divergent transitions (indicates geometry problems)
- Energy diagnostic (E-BFMI > 0.3)

### Success Criteria

The model is **adequate** if:
1. **R-hat < 1.01** for all parameters (convergence)
2. **Posterior predictive checks pass:**
   - Replicated data Y_rep matches observed Y in distribution
   - No systematic residual patterns (plot residuals vs x, vs fitted)
3. **Parameter plausibility:**
   - Y_max ∈ [2.5, 3.0] (consistent with observed max and plateau)
   - K ∈ [0.5, 10] (reasonable half-saturation point)
   - σ ∈ [0.15, 0.25] (consistent with observed residual SD)
4. **Predictive accuracy:**
   - LOO-CV ELPD better than linear model by >5 (meaningful improvement)
   - Pareto-k diagnostics all < 0.7 (no highly influential points)
5. **Extrapolation behavior reasonable:**
   - Posterior predictive intervals for x=50 show Y approaching Y_max (not diverging)

### Failure Criteria (When to ABANDON This Model)

I will **abandon Michaelis-Menten** if:

1. **Persistent divergent transitions** (>1% of samples) despite:
   - Increasing adapt_delta to 0.99
   - Trying non-centered parameterization
   - Reparameterizing as Y_max, log_K
   - **Why this matters:** Indicates fundamental geometry problem, likely non-identifiability of K and Y_max from this data.

2. **Posterior Y_max < 2.63** (maximum observed Y):
   - **Interpretation:** Model predicts asymptote below observed data - logically impossible.
   - **Action:** Switch to unbounded model (power-law, logarithmic).

3. **Posterior K > 20 with wide uncertainty:**
   - **Interpretation:** Half-saturation far beyond observed x range. Model cannot distinguish saturation from linear/logarithmic growth.
   - **Action:** Saturation not identifiable from this data. Use simpler log or polynomial model.

4. **Prior-posterior overlap >80% for K:**
   - **Interpretation:** Data provides negligible information about K. Prior dominates.
   - **Action:** Model too complex for data. Simplify to logarithmic or quadratic.

5. **Systematic residual pattern in posterior predictive:**
   - E.g., U-shaped residuals persist in posterior mean fit
   - **Interpretation:** Functional form misspecified despite saturation shape.
   - **Action:** Try power-law or exponential saturation (Models 2-3).

6. **LOO-CV worse than logarithmic baseline:**
   - **Interpretation:** Additional complexity (K parameter) not justified by predictive performance.
   - **Action:** Adopt simpler logarithmic model.

### Expected Challenges

1. **K and Y_max correlation:** Posterior samples may show strong negative correlation (as K↑, Y_max must ↑ to maintain fit at high x). Monitor with pairs plot.

2. **Weak identification of K:** With only 27 points and sparse high-x coverage, posterior K may be wide. This is honest uncertainty, not failure.

3. **Boundary behavior for K:** If posterior concentrates near 0, model approaches linear (Y ≈ Y_max * x / K for small x). Check if K significantly different from 0.

4. **Prior sensitivity:** With N=27, priors matter. Run sensitivity analysis with vague priors (Y_max ~ Normal(2.5, 2), K ~ Normal(5, 50)) to check robustness.

### Stress Test: Designed to Break the Model

**Test:** Fit model to data with x≤15 only (N=19), then predict x∈[16, 31.5].

**Hypothesis:** If saturation is real, model should predict plateau. If apparent plateau is noise, predictions will have massive uncertainty.

**Breakage criterion:** If 95% posterior predictive interval for x=30 includes Y<2.0 or Y>3.0, saturation not reliably identified. Switch to logarithmic model.

---

## Model 2: Power-Law with Saturation (ALTERNATIVE 1)

### Theoretical Motivation

Power-law relationships appear in scaling phenomena (allometry, economics, physics). Unlike Michaelis-Menten, power-laws don't have a hard asymptote but exhibit **diminishing returns without bound**. This may be more appropriate if:
- True Y_max is unknowable (no physical constraint)
- Saturation is gradual, not sharp
- EDA "plateau" is artifact of limited x range

**Functional form:**
```
μ(x) = a + b * x^c,  where 0 < c < 1
```

**Interpretation:**
- **c < 1:** Diminishing returns (curvature)
- **c = 0.5:** Square-root relationship (common in diffusion processes)
- **c → 0:** Approaches logarithmic
- **c → 1:** Approaches linear

**Why consider this?** EDA logarithmic fit (R²=0.83) is power-law with c→0. Generalizes to test whether c significantly different from 0 (log) or 1 (linear).

### Complete Model Specification

#### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = a + b * x_i^c
```

#### Priors

**a (intercept at x=0):**
```
a ~ Normal(1.7, 0.5)
```
- **Justification:** Observed Y at x=1.0 is 1.86. Intercept should be slightly below. Prior allows a∈[0.7, 2.7] at 95%.

**b (scaling coefficient):**
```
b ~ Normal(0.5, 0.5) truncated at b > 0
```
- **Justification:** For c≈0.5, need b large enough that b*31.5^0.5 + a ≈ 2.6. Thus b ≈ (2.6-1.7)/5.6 ≈ 0.16. Prior centered higher with wide SD to be weakly informative. Positive constraint ensures Y increases with x.

**c (power exponent):**
```
c ~ Beta(2, 2)  [support on (0, 1)]
```
- **Justification:** Beta(2,2) is symmetric on [0,1] with mean=0.5, concentrating mass away from boundaries. Regularizes toward "middle ground" between log (c→0) and linear (c=1). Data will determine if c is small (log-like) or moderate.

**σ (residual SD):**
```
σ ~ HalfNormal(0.25)
```
- Same as Model 1.

#### Reparameterization

**Problem:** Priors on a, b, c may not align with data scale. Power-law can be unstable near c=0 or c=1.

**Solution:** Centered parameterization + informative priors constrained by data range:

```stan
parameters {
  real a;
  real<lower=0> b;
  real<lower=0, upper=1> c;
  real<lower=0> sigma;
}

model {
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = a + b * pow(x[i], c);
  }

  // Priors
  a ~ normal(1.7, 0.5);
  b ~ normal(0.5, 0.5);
  c ~ beta(2, 2);
  sigma ~ normal(0, 0.25);

  // Likelihood
  Y ~ normal(mu, sigma);
}
```

**Note:** `pow(x[i], c)` is numerically stable in Stan for c∈(0,1).

### Implementation Strategy

**Framework:** Stan (power functions well-optimized)

**Sampling parameters:**
- 4 chains, 2000 iterations (1000 warmup)
- `adapt_delta = 0.90` (less stringent than MM model, simpler geometry)

**Potential issues:**
- Posterior c near 0 or 1 may indicate boundary behavior - interpret as "effectively logarithmic" or "effectively linear"
- If c posterior spans full [0,1], model too flexible for data

### Success Criteria

Model is **adequate** if:
1. **Convergence:** R-hat < 1.01, ESS > 400
2. **Posterior c credibly different from boundaries:**
   - 95% CI for c excludes 0 and 1 (e.g., c∈[0.15, 0.75])
   - If c∼0, revert to logarithmic; if c∼1, revert to linear
3. **Posterior predictive checks pass:** No residual patterns
4. **Parameters plausible:**
   - a + b*1^c ≈ 1.9 (predicted Y at x=1 matches data)
   - a + b*30^c ≈ 2.5 (predicted Y at x=30 matches plateau)
5. **LOO-CV competitive:** Within 2 ELPD of Michaelis-Menten

### Failure Criteria (When to ABANDON)

Abandon **power-law** if:

1. **Posterior c includes 0 or 1 in 50% interval:**
   - **Interpretation:** Data cannot distinguish power-law from logarithmic/linear.
   - **Action:** Use simpler model (log if c→0, linear if c→1).

2. **Predicted Y unbounded at high x:**
   - For c>0, Y→∞ as x→∞. If posterior suggests Y>3.5 for x=100, physically implausible given observed Y∈[1.7, 2.6].
   - **Action:** Switch to asymptotic model (Michaelis-Menten or exponential saturation).

3. **Posterior a < 1.0:**
   - **Interpretation:** Model predicts unreasonably low Y at low x.
   - **Action:** Try alternative functional form.

4. **Worse LOO-CV than logarithmic:**
   - **Interpretation:** Additional parameter c not useful.
   - **Action:** Simplify to Y = a + b*log(x).

5. **Divergent transitions or poor mixing:**
   - Despite tuning, if sampling pathological.
   - **Action:** Model geometry unsuitable. Try exponential saturation.

### Expected Challenges

1. **c weakly identified:** With N=27, distinguishing c=0.3 from c=0.5 may be difficult. Wide posterior is honest.

2. **Correlation among a, b, c:** Parameters jointly determine curve shape. Monitor pairs plot.

3. **Extrapolation uncertainty:** Without hard asymptote, predictions for x>40 will be speculative. Communicate wide intervals.

### Stress Test

**Test:** Fit to full data, then compute posterior predictive for x=100.

**Breakage:** If 95% interval for Y(x=100) includes Y>5.0, model is extrapolating unreasonably (unbounded growth inconsistent with plateau). Indicates true asymptote exists - switch to Model 1.

---

## Model 3: Exponential Saturation Model (ALTERNATIVE 2)

### Theoretical Motivation

Exponential saturation describes approach to equilibrium in first-order systems:
- Chemical reactions approaching equilibrium
- Temperature equilibration (Newton's cooling)
- Capacitor charging
- Learning curves (exponential approach to mastery)

**Functional form:**
```
μ(x) = Y_max - (Y_max - Y_0) * exp(-r * x)
```

**Interpretation:**
- **Y_max:** Asymptotic maximum (as x→∞)
- **Y_0:** Initial value (at x=0)
- **r:** Rate of approach to asymptote (larger r = faster saturation)

**At x=0:** μ(0) = Y_0
**As x→∞:** μ(∞) = Y_max
**At x=1/r:** μ(1/r) = Y_max - (Y_max - Y_0)/e ≈ Y_max - 0.37*(Y_max - Y_0)

**Why consider this?**
- Provides different saturation dynamics than Michaelis-Menten
- MM model: μ = Y_max * x/(K+x) → saturation from ratio
- Exponential: μ = Y_max - decay → saturation from exponential approach
- If both fit equally well, prefer simpler (MM has cleaner interpretation)

### Complete Model Specification

#### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = Y_max - (Y_max - Y_0) * exp(-r * x_i)
```

#### Priors

**Y_max (asymptote):**
```
Y_max ~ Normal(2.6, 0.3)
```
- Same as Model 1. Maximum observed Y is 2.63, prior centered just below with moderate uncertainty.

**Y_0 (initial value at x=0):**
```
Y_0 ~ Normal(1.7, 0.3)
```
- **Justification:** Observed Y at x=1.0 is 1.86. Extrapolating backward, Y(0) should be ~1.7. Prior allows Y_0 ∈ [1.1, 2.3].
- **Constraint:** Must have Y_0 < Y_max (enforced in Stan).

**r (rate constant):**
```
r ~ Exponential(0.5)
```
- **Justification:** Exponential prior with mean=2 (rate=0.5) is weakly informative. Implies median r ≈ 1.4. At r=1, saturation is 63% complete by x=1. EDA shows rapid initial rise, so r likely >0.1. Exponential prior concentrates on positive values, avoids r→0 (no saturation).

**σ (residual SD):**
```
σ ~ HalfNormal(0.25)
```

#### Reparameterization

**Problem:** Y_max, Y_0, r correlated. If Y_max↑ and Y_0↑ proportionally, similar fit. Need to break degeneracy.

**Solution:** Reparameterize as Y_max, delta = Y_max - Y_0, r:

```stan
parameters {
  real<lower=0> Y_max;
  real<lower=0> delta;  // Y_max - Y_0
  real<lower=0> r;
  real<lower=0> sigma;
}

transformed parameters {
  real Y_0 = Y_max - delta;
}

model {
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = Y_max - delta * exp(-r * x[i]);
  }

  // Priors
  Y_max ~ normal(2.6, 0.3);
  delta ~ normal(0.9, 0.3);  // Expected range: Y_max - Y_0 ≈ 2.6 - 1.7 = 0.9
  r ~ exponential(0.5);
  sigma ~ normal(0, 0.25);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  real half_saturation_x = -log(0.5) / r;  // x where 50% of delta is reached
}
```

**Advantage:** delta more interpretable (total change from x=0 to x=∞). Also computes half-saturation x as derived quantity for comparison with MM model's K.

### Implementation Strategy

**Framework:** Stan

**Sampling:**
- 4 chains, 2000 iterations (1000 warmup)
- `adapt_delta = 0.90`

**Derived quantities:**
- Half-saturation x: Point where Y reaches (Y_0 + Y_max)/2
- Compare with MM model's K for consistency check

### Success Criteria

Model is **adequate** if:
1. **Convergence:** R-hat < 1.01, ESS > 400, no divergences
2. **Parameter plausibility:**
   - Y_max ∈ [2.5, 3.0]
   - Y_0 ∈ [1.4, 2.0]
   - r ∈ [0.05, 1.0] (if r too small, no saturation; if r too large, instant saturation)
3. **Posterior predictive checks pass**
4. **Half-saturation x between 1 and 15:** Should align with visual "bend" in data
5. **LOO-CV competitive with Models 1-2**

### Failure Criteria (When to ABANDON)

Abandon **exponential saturation** if:

1. **Posterior r near 0 (r < 0.01):**
   - **Interpretation:** exp(-r*x) ≈ 1 for all x, model collapses to constant.
   - **Action:** No saturation identifiable. Use logarithmic or linear model.

2. **Posterior delta includes 0:**
   - **Interpretation:** Y_max ≈ Y_0, meaning no change with x (flat relationship).
   - **Action:** Model fundamentally wrong. Check data.

3. **Half-saturation x > 50:**
   - **Interpretation:** Saturation so slow it's beyond observed x range. Model cannot distinguish from linear growth.
   - **Action:** Use simpler model without asymptote.

4. **Worse LOO-CV than MM model by >5 ELPD:**
   - **Interpretation:** Exponential saturation form not supported.
   - **Action:** Prefer Michaelis-Menten.

5. **Systematic residual patterns:** E.g., poor fit at low x despite 3 parameters.
   - **Action:** Try power-law or polynomial.

### Expected Challenges

1. **r weakly identified:** With sparse high-x data, rate of approach to asymptote uncertain. Posterior may be wide.

2. **Comparison with MM model:** Both have 3 parameters (Y_max/delta/r vs Y_max/K/σ), similar flexibility. Prefer whichever has better predictive performance and simpler interpretation.

3. **Extrapolation:** Like MM, predicts bounded Y. Useful if true asymptote exists.

### Stress Test

**Test:** Fit to data with x≤10 only, predict x∈[10, 31.5].

**Breakage:** If posterior predictive for x=30 has 95% interval [1.5, 3.5] (spanning almost entire observed Y range), exponential form cannot extrapolate. Indicates data insufficient to identify saturation dynamics.

---

## Model Ranking and Rationale

### Primary Recommendation: Model 1 (Michaelis-Menten)

**Rank:** 1st

**Rationale:**
1. **Clearest interpretation:** Y_max is maximum response, K is half-saturation point. These have direct domain meaning.
2. **Widespread applicability:** MM form appears in biochemistry, pharmacology, ecology, economics (diminishing returns to scale).
3. **EDA support:** Fitted MM (R²=0.816) performed well with Y_max=2.59, K=0.64.
4. **Extrapolation behavior:** Bounded predictions (Y ≤ Y_max) are scientifically plausible if data represents saturating process.
5. **Prior informativeness:** Can elicit expert priors for Y_max (physical maximum) and K (characteristic scale).

**Trade-off:** Nonlinear in parameters → potential MCMC challenges. Reparameterization with log(K) mitigates this.

### Alternative 1: Model 2 (Power-Law)

**Rank:** 2nd

**Rationale:**
1. **Generality:** Nests logarithmic (c→0) and linear (c→1) as special cases. Model comparison via posterior c.
2. **Fewer assumptions:** Doesn't impose hard asymptote. Appropriate if plateau is artifact.
3. **Flexibility:** Can fit data equally well without forcing saturation.

**Trade-off:** Unbounded growth (Y→∞ as x→∞). If true process has ceiling, power-law will extrapolate poorly. Posterior c may be weakly identified.

**Use case:** If domain knowledge does NOT suggest physical maximum, or if EDA plateau is questioned.

### Alternative 2: Model 3 (Exponential Saturation)

**Rank:** 3rd

**Rationale:**
1. **Mechanistic:** Exponential approach to equilibrium is common in physical/chemical systems.
2. **Different dynamics:** Compared to MM, provides alternative hypothesis for saturation functional form.

**Trade-off:** Very similar to MM model (both 3-parameter asymptotic models). Likely to yield similar fits. If MM and exponential both fit well, prefer simpler interpretation (MM's K vs exponential's r).

**Use case:** If domain specifically suggests exponential decay to equilibrium (rare for regression contexts). More common in time-series.

**Why ranked 3rd?** For cross-sectional x-Y data, MM model is more conventional and interpretable. Exponential saturation typically used for time-series. Unless domain expertise suggests exponential dynamics, MM is preferred.

---

## Decision Points and Pivoting Strategy

### Stage 1: Initial Fitting (Models 1-3)

**Action:** Fit all three models to full data (N=27).

**Decision point after Stage 1:**

| Outcome | Action |
|---------|--------|
| All three converge, similar LOO-CV (within 2 ELPD) | Prefer Model 1 (simplest interpretation) |
| Model 1 or 3 clearly better LOO (>5 ELPD) | Adopt asymptotic model, report Y_max estimates |
| Model 2 clearly better LOO (>5 ELPD) | Adopt power-law, report posterior c to classify (log-like vs linear-like) |
| Model 1 has divergences, Model 3 converges | Use exponential saturation |
| All models have poor posterior predictive (residual patterns) | PIVOT to Stage 2 |

### Stage 2: Model Class Reconsideration

**Triggered if:** All saturation models fail posterior predictive checks OR LOO-CV worse than simple logarithmic baseline.

**Alternative model classes to consider:**

**A. Segmented/Piecewise Linear:**
```
μ(x) = β_0 + β_1*x  if x ≤ θ
μ(x) = β_0 + β_1*θ  if x > θ
```
- **Hypothesis:** True saturation is abrupt transition (changepoint) rather than smooth.
- **Implementation:** Stan with changepoint parameter θ.

**B. Gaussian Process Regression:**
```
f(x) ~ GP(0, k(x, x'))
Y_i ~ Normal(f(x_i), σ)
```
- **Hypothesis:** Relationship is complex, non-parametric approach needed.
- **Kernel:** Squared exponential or Matern.
- **Caveat:** With N=27, GP may overfit. Use informative priors on lengthscale.

**C. Hierarchical Model (if autocorrelation confirmed):**
```
Y_i ~ Normal(μ(x_i), σ)
μ(x_i) = f(x_i) + α_group[i]
α_group ~ Normal(0, τ)
```
- **Hypothesis:** Data has grouping structure (e.g., batches, time periods).
- **Requires:** Metadata about data collection order/grouping.

### Stage 3: Fundamental Questioning

**Triggered if:** All models fail, including alternatives from Stage 2.

**Questions to revisit:**
1. Is relationship actually monotonic? (check for non-monotonicity in EDA plots)
2. Are there hidden confounders? (x may proxy for multiple causal factors)
3. Is variance truly constant? (refit with heteroscedastic models)
4. Is normality assumption violated? (try Student-t likelihood)

**Escape routes:**
- Student-t likelihood: `Y_i ~ StudentT(ν, μ_i, σ)` with ν ~ Gamma(2,0.1) for robustness
- Heteroscedastic: `σ(x) = σ_0 * exp(γ*x)` or `σ(x) = σ_0 * x^γ`
- Mixture model: `Y_i ~ p*N(μ_1(x), σ_1) + (1-p)*N(μ_2(x), σ_2)` if two processes

### Red Flags Requiring Immediate Model Class Change

1. **All asymptotic models predict Y_max < max(Y_observed):**
   - **Interpretation:** Asymptote below data is impossible.
   - **Action:** Abandon saturation hypothesis. Use logarithmic or polynomial.

2. **Posterior K or r has 95% CI spanning [0.01, 100]:**
   - **Interpretation:** Saturation parameter completely unidentified.
   - **Action:** Data insufficient to estimate saturation. Use simpler model.

3. **LOO Pareto-k > 0.7 for >5 observations:**
   - **Interpretation:** Model highly sensitive to individual points, likely misspecified.
   - **Action:** Check for outliers, try robust likelihood (Student-t).

4. **Posterior predictive p-value extreme (p < 0.01 or p > 0.99) for multiple test statistics:**
   - **Interpretation:** Model fundamentally inconsistent with data.
   - **Action:** Try entirely different functional form (GP, spline, polynomial).

### Stopping Rules

**When to stop exploring models:**

1. **Achieved adequate fit:** LOO-CV stable, posterior predictive checks pass, parameters scientifically plausible.
2. **Exhausted reasonable options:** Tried parametric (MM, power, exponential), semi-parametric (splines), non-parametric (GP). If all fail, data may be too sparse/noisy for reliable inference.
3. **Discovered data issue:** E.g., measurement error, transcription mistakes, hidden confounders. Report limitation rather than force model fit.

**Maximum iterations:** Do not fit >10 distinct model classes. If no satisfactory model after 10 attempts, conclude data inadequate and recommend:
- Collect more data (especially x>20)
- Investigate data generation process
- Accept descriptive statistics only (no predictive model)

---

## Prior Sensitivity and Model Validation Plan

### Prior Sensitivity Analysis

For chosen model (likely Model 1), refit with:

1. **Vague priors:**
   - Y_max ~ Normal(2.5, 5) [very wide]
   - K ~ Normal(10, 50) [extremely wide]
   - σ ~ HalfCauchy(1) [heavy-tailed]

2. **Strong priors (if expert knowledge available):**
   - Y_max ~ Normal(2.55, 0.1) [confident about asymptote]
   - K ~ Normal(1, 0.5) [confident about saturation point]

**Compare posteriors:** If vague and weakly informative priors yield similar posteriors (KL divergence < 0.1), data is informative and prior choice robust. If posteriors differ substantially, report sensitivity.

### Cross-Validation Strategy

**LOO-CV (primary):**
- Compute ELPD_loo for all models
- Check Pareto-k diagnostics
- Report ELPD differences with standard errors

**K-fold CV (if computational budget allows):**
- 5-fold CV, stratified by x quintiles to ensure coverage in each fold
- Compute RMSE, MAE, coverage of 95% predictive intervals

**Holdout set (if data permits):**
- Randomly hold out 20% (5-6 observations)
- Fit to 80%, predict holdout
- Check calibration of predictive intervals

### Posterior Predictive Checks

**Test statistics to monitor:**
1. **Mean(Y_rep) vs mean(Y_obs):** Should overlap
2. **SD(Y_rep) vs SD(Y_obs):** Should overlap
3. **Min/Max(Y_rep) vs min/max(Y_obs):** Check for extreme value generation
4. **Correlation(Y_rep, x) vs correlation(Y_obs, x):** Check monotonicity preserved

**Graphical checks:**
1. **Overlay plot:** Y_obs vs x with 100 draws of Y_rep(x) overlaid as light lines. Should "envelope" observed data.
2. **Residual plot:** Plot residuals from posterior mean vs x. Should show no pattern.
3. **QQ plot:** Quantiles of Y_obs vs quantiles of Y_rep. Should follow diagonal.

**Decision:** If >2 of above checks fail (p-value < 0.05 or visual misfit), model inadequate.

---

## Computational Considerations

### Expected Runtimes (Stan on modern laptop)

| Model | Warmup | Sampling | Total (4 chains) |
|-------|--------|----------|------------------|
| Model 1 (MM) | ~20s | ~20s | ~3 min |
| Model 2 (Power) | ~15s | ~15s | ~2 min |
| Model 3 (Exp) | ~20s | ~20s | ~3 min |

**Note:** N=27 is very small. Stan will be fast. Main bottleneck is model iteration, not computation.

### Diagnosing MCMC Problems

**If divergences occur:**
1. Increase `adapt_delta` from 0.90 → 0.95 → 0.99
2. Check pairs plot for funnel geometry (sign of non-centered parameterization)
3. Try non-centered: instead of `K ~ Normal(5, 5)`, use `K = 5 + 5*K_raw; K_raw ~ Normal(0,1)`
4. Inspect which parameters have divergences (use `sampler_params` in Stan output)

**If ESS low (<100):**
1. Check trace plots for stickiness
2. Increase iterations (4000 total)
3. Thin samples if autocorrelation high (though thinning loses information)
4. Consider reparameterization

**If R-hat > 1.01:**
1. Run longer chains (4000 iterations)
2. Check for multimodality (plot posteriors, look for multiple peaks)
3. If multimodal, model may be non-identified

---

## Summary: Key Differences Between Models

| Aspect | Model 1 (MM) | Model 2 (Power) | Model 3 (Exp) |
|--------|-------------|----------------|---------------|
| **Asymptote** | Yes (Y_max) | No | Yes (Y_max) |
| **Parameters** | Y_max, K, σ | a, b, c, σ | Y_max, Y_0, r, σ |
| **Flexibility** | Moderate | High | Moderate |
| **Extrapolation** | Bounded | Unbounded | Bounded |
| **Interpretation** | Clear (biochem) | General | Physical equilibrium |
| **MCMC difficulty** | Moderate (K-Y_max correlation) | Low | Moderate (r-delta correlation) |
| **Use case** | Saturation expected | Uncertain about asymptote | Exponential approach |
| **Rank** | 1st | 2nd | 3rd |

---

## Final Philosophical Notes

### On Model Selection
The goal is NOT to find the "true" model (all models are wrong). The goal is to find:
1. **Adequate description** of Y-x relationship in observed range
2. **Reliable predictions** with calibrated uncertainty
3. **Interpretable parameters** for scientific communication

### On Falsification
Each model has explicit failure criteria. **Discovering a model is wrong is progress**, not failure. If all proposed models fail, we learn that:
- Data is more complex than assumed
- Sample size insufficient
- Assumptions about normality/homoscedasticity violated
- Relationship may not be smooth/monotonic

This would motivate data collection or alternative analyses (non-parametric, semi-parametric).

### On Switching Model Classes
If Model 1 (MM) fails but logarithmic succeeds:
- **Interpretation:** Asymptote not identifiable from this data. Growth continues (slowly) beyond observed x.
- **Action:** Report logarithmic model, acknowledge cannot estimate Y_max.

If all parametric models fail:
- **Interpretation:** Functional form unknown or too complex.
- **Action:** Use Gaussian Process or spline for flexible fit, sacrifice interpretability for predictive accuracy.

### On Uncertainty
With N=27 and sparse high-x coverage, **wide credible intervals are a feature, not a bug**. Honest uncertainty quantification is the strength of Bayesian approach. Resist temptation to:
- Use strong priors to narrow intervals (unless justified by domain knowledge)
- Report point estimates only (always include intervals)
- Extrapolate confidently beyond observed x range

---

## Implementation Checklist

**Before fitting:**
- [ ] Standardize x (x_std = (x - mean(x)) / sd(x)) if needed for numerical stability
- [ ] Center Y (Y_centered = Y - mean(Y)) if intercept unimportant
- [ ] Check data file matches EDA (N=27, no missing values)
- [ ] Set random seed for reproducibility

**After fitting:**
- [ ] Check R-hat < 1.01 for ALL parameters
- [ ] Check ESS_bulk > 400
- [ ] Check ESS_tail > 400
- [ ] Inspect trace plots (should look like "fuzzy caterpillars")
- [ ] Check for divergences (should be 0)
- [ ] Examine pairs plot for pathological correlations
- [ ] Run posterior predictive checks
- [ ] Compute LOO-CV with Pareto-k diagnostics
- [ ] Compare to baseline (logarithmic) model
- [ ] Plot posterior mean with 95% credible bands
- [ ] Report parameter posteriors with interpretation

**Reporting:**
- [ ] State priors used (not just "weakly informative")
- [ ] Report MCMC diagnostics (R-hat, ESS, divergences)
- [ ] Show posterior predictive check plots
- [ ] Compare models via LOO-CV table
- [ ] Discuss parameter interpretation in domain context
- [ ] Acknowledge limitations (sparse high-x data)
- [ ] Provide predictions with uncertainty for x∈[1, 40]

---

## File Paths and Resources

**All outputs will be saved to:**
- `/workspace/experiments/designer_2/`

**Model code:**
- `/workspace/experiments/designer_2/stan_models/model1_michaelis_menten.stan`
- `/workspace/experiments/designer_2/stan_models/model2_powerlaw.stan`
- `/workspace/experiments/designer_2/stan_models/model3_exponential.stan`

**Analysis scripts:**
- `/workspace/experiments/designer_2/scripts/fit_models.py` (Python driver for Stan)
- `/workspace/experiments/designer_2/scripts/diagnostics.py` (MCMC diagnostics)
- `/workspace/experiments/designer_2/scripts/comparison.py` (Model comparison via LOO)

**Results:**
- `/workspace/experiments/designer_2/results/model1_summary.txt` (Posterior summaries)
- `/workspace/experiments/designer_2/results/loo_comparison.csv` (LOO-CV results)
- `/workspace/experiments/designer_2/results/posterior_predictive_checks.png`

**Data:**
- `/workspace/data/data.csv` (N=27, columns: x, Y)

---

*End of Proposal - Designer 2*
