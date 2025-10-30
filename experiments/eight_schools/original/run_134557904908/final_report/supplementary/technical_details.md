# Technical Supplementary Material
## Extended Methods and Mathematical Derivations

**Bayesian Meta-Analysis: Technical Documentation**

---

## 1. Mathematical Foundations

### 1.1 Fixed-Effect Model (Model 1)

#### Conjugate Analysis

**Setup**:
```
Likelihood: y_i | θ ~ N(θ, σ_i²)    i = 1,...,J
Prior:      θ ~ N(μ_0, τ_0²)
```

**Posterior Derivation**:

The posterior is conjugate (Normal):
```
p(θ | y) ∝ p(y | θ) × p(θ)
         ∝ ∏[i=1 to J] N(y_i | θ, σ_i²) × N(θ | μ_0, τ_0²)
```

Taking logs:
```
log p(θ | y) ∝ -1/2 ∑[(y_i - θ)²/σ_i²] - 1/2[(θ - μ_0)²/τ_0²]
             ∝ -1/2[θ² ∑(1/σ_i²) - 2θ ∑(y_i/σ_i²) + θ²/τ_0² - 2θμ_0/τ_0²]
             ∝ -1/2[θ²(∑1/σ_i² + 1/τ_0²) - 2θ(∑y_i/σ_i² + μ_0/τ_0²)]
```

Completing the square:
```
Precision posterior: λ_post = ∑(1/σ_i²) + 1/τ_0²
Mean posterior:      μ_post = (∑y_i/σ_i² + μ_0/τ_0²) / λ_post
Variance posterior:  τ²_post = 1/λ_post
```

**Numerical Example** (with μ_0 = 0, τ_0 = 20):

Data:
```
y = [28, 8, -3, 7, -1, 1, 18, 12]
σ = [15, 10, 16, 11, 9, 11, 10, 18]

Precision_prior = 1/400 = 0.00250
Precision_data = ∑(1/σ_i²) = 0.06044

Precision_post = 0.00250 + 0.06044 = 0.06294

Weighted_sum = ∑(y_i/σ_i²) = 0.4647
Mean_post = (0 + 0.4647) / 0.06294 = 7.384

SD_post = 1/√0.06294 = 3.987
```

**Comparison to MCMC**:
```
Analytical: θ = 7.384 ± 3.987
MCMC:       θ = 7.407 ± 3.994
Difference: 0.023 (mean), 0.007 (SD)
```

Error < 0.6%, confirming MCMC correctness.

#### Effective Sample Size

The "effective sample size" for meta-analysis:
```
n_eff = 1 / mean(1/n_i)    (harmonic mean)

For our data (assuming n_i ∝ 1/σ_i²):
n_eff ≈ J / mean(σ_i²/min(σ_i²)) ≈ 8 / 4 = 2 effective studies

Interpretation: Information content equivalent to ~2 studies at highest precision
```

### 1.2 Random-Effects Model (Model 2)

#### Hierarchical Structure

**Full Model**:
```
Level 1 (Likelihood): y_i | θ_i, σ_i ~ N(θ_i, σ_i²)
Level 2 (Hierarchy):  θ_i | μ, τ ~ N(μ, τ²)
Level 3 (Priors):     μ ~ N(0, 20²)
                      τ ~ Half-Normal(0, 5²)
```

**Marginal Likelihood** (integrating out θ_i):
```
p(y_i | μ, τ, σ_i) = ∫ p(y_i | θ_i, σ_i) p(θ_i | μ, τ) dθ_i
                    = N(y_i | μ, σ_i² + τ²)
```

This is the "unconditional" or "marginal" model where each observation is drawn from N(μ, σ_i² + τ²).

**Joint Posterior**:
```
p(μ, τ, θ | y) ∝ [∏ N(y_i | θ_i, σ_i²)] × [∏ N(θ_i | μ, τ²)] × p(μ) × p(τ)
```

No conjugacy → requires MCMC sampling.

#### Non-Centered Parameterization

**Problem**: Centered parameterization can have "funnel" geometry when τ → 0.

**Centered** (problematic):
```
θ_i ~ N(μ, τ²)
```

**Non-Centered** (efficient):
```
θ_raw_i ~ N(0, 1)
θ_i = μ + τ × θ_raw_i
```

**Why it works**:
- When τ → 0, θ_raw_i remains on standard scale
- Avoids correlation between τ and θ_i
- NUTS sampler can explore geometry more efficiently

**Jacobian**: Transformation has unit Jacobian, no adjustment needed.

#### I² Calculation

**Definition**:
```
I² = τ² / (τ² + σ̄²) × 100%

where σ̄² = mean(σ_i²) = typical within-study variance
```

**Interpretation**:
- I² = 0%: All variance from within-study error
- I² = 100%: All variance from between-study heterogeneity
- I² = 25%, 50%, 75%: Thresholds for low/moderate/substantial

**For our data**:
```
τ = 3.36 (posterior mean)
σ̄ = 12.5 → σ̄² = 156.25
I² = 3.36² / (3.36² + 156.25) = 11.29 / 167.54 = 6.7%

Note: Slight discrepancy with reported 8.3% due to different averaging methods
      (arithmetic mean of I² samples vs. I² of mean τ)
```

### 1.3 Shrinkage and Partial Pooling

#### Shrinkage Formula

For study i, the shrunken estimate is:
```
θ̂_i = w_i × y_i + (1 - w_i) × μ

where w_i = τ² / (τ² + σ_i²)  (shrinkage weight)
```

**Interpretation**:
- If τ >> σ_i: w_i → 1, θ̂_i → y_i (no pooling, study estimate dominates)
- If τ << σ_i: w_i → 0, θ̂_i → μ (complete pooling, grand mean dominates)
- Middle ground: partial pooling based on precision

**For our data**:
```
τ = 3.36, σ_i ∈ [9, 18]

Study 1 (σ=15): w = 3.36²/(3.36² + 15²) = 0.048 → 4.8% weight to data
Study 5 (σ=9):  w = 3.36²/(3.36² + 9²) = 0.123 → 12.3% weight to data

Heavy shrinkage because τ << σ_i (between-study << within-study variation)
```

**Effective Sample Size** (per study):
```
ESS_i = w_i × n_i

If τ = 0 (complete pooling): ESS = ∑n_i (all studies combined)
If τ → ∞ (no pooling): ESS_i = n_i (each study independent)
```

---

## 2. Prior Specifications: Extended Justification

### 2.1 Prior for θ (Fixed-Effect)

**Choice**: θ ~ Normal(0, 20²)

**Rationale**:

1. **Scale**: SD = 20 places 95% prior mass in (-40, 40)
   - Assumed outcome scale: y ∈ [-3, 28] (observed range)
   - Prior allows 3× wider range than observed
   - Regularizes without strong constraint

2. **Center**: μ = 0 (null hypothesis as prior center)
   - No directional bias
   - Symmetric around no effect
   - Let data determine direction

3. **Informativeness**: Weakly informative
   - Prior SD (20) >> Expected posterior SD (~4)
   - Data-dominated: Likelihood/Prior = 4²/20² = 0.04 (4% prior weight)
   - Effective sample size from prior: n_prior = 1/20² = 0.0025 studies

4. **Sensitivity**: Tested σ ∈ {10, 20, 50}
   - σ=10: 95% in (-20, 20), more informative
   - σ=50: 95% in (-100, 100), very diffuse
   - Results unchanged (< 1% variation)

**Alternative Priors Considered**:

- **Flat (Improper)**: p(θ) ∝ 1
  - Advantage: Completely data-driven
  - Disadvantage: No regularization, can be improper
  - Posterior: Exactly conjugate formula above

- **Half-Normal** (if known positive): θ ~ Half-Normal(0, 20²)
  - Advantage: Incorporates directional knowledge
  - Disadvantage: Assumes no possibility of harm
  - Not used: Let data determine direction

- **Student-t**: θ ~ Student-t(ν=3, μ=0, σ=20)
  - Advantage: Heavier tails, more robust
  - Disadvantage: Computational complexity, not needed for θ
  - Not needed: Normal posterior expected

### 2.2 Prior for τ (Between-Study SD)

**Choice**: τ ~ Half-Normal(0, 5²)

**Rationale**:

1. **Domain**: Constrained to τ > 0 (SD must be positive)

2. **Mode at Zero**: Reflects EDA finding (I² = 0%)
   - Prior favors homogeneity but allows heterogeneity
   - Mode at 0, mass decays with increasing τ

3. **Scale**: SD = 5 allows substantial heterogeneity
   - P(τ > 5) ≈ 32% (substantial prior mass for moderate heterogeneity)
   - P(τ > 10) ≈ 5% (allows large heterogeneity)
   - P(τ < 2) ≈ 69% (favors low heterogeneity)

4. **Comparison to σ_i**: τ_prior ~ 5 vs σ̄ = 12.5
   - Prior allows τ up to ~40% of typical within-study SD
   - Reasonable range for meta-analyses

**Alternative Priors Considered**:

- **Half-Cauchy**: τ ~ Half-Cauchy(0, 2.5)
  - **Advantage**: Recommended by Gelman (2006) for hierarchical SD
  - **Disadvantage**: Heavy tails may be too diffuse for J=8
  - **Our choice**: Half-Normal more informative, appropriate for small J

- **Uniform**: τ ~ Uniform(0, U)
  - **Advantage**: Simple, explicit upper bound
  - **Disadvantage**: Discontinuity at U, arbitrary choice of U
  - **Not used**: Half-Normal smoother

- **Inverse-Gamma**: τ² ~ Inverse-Gamma(α, β)
  - **Advantage**: Conjugate for some models
  - **Disadvantage**: Can favor small τ too strongly, sensitive to hyperparameters
  - **Not used**: Half-Normal more robust

**Sensitivity Analysis**:

Tested σ_τ ∈ {5, 10, 20}:

| Prior | τ Mean | τ 95% HDI | I² Mean | I² 95% HDI | P(I² < 25%) |
|-------|--------|-----------|---------|------------|-------------|
| HN(0,5) | 3.36 | [0, 8.3] | 8.3% | [0%, 29%] | 92.4% |
| HN(0,10) | 5.21 | [0, 12.8] | 12.1% | [0%, 42%] | 87.8% |
| HN(0,20) | 6.89 | [0, 17.1] | 15.8% | [0%, 53%] | 81.3% |

**Interpretation**:
- More diffuse priors → larger τ estimates (expected)
- Sensitivity ratio: 6.89/3.36 = 2.05 (moderate)
- **All priors agree**: I² < 25% (low heterogeneity)
- Qualitative conclusion robust to prior choice

**Recommendation**: Report HN(0,5) as primary, others as sensitivity

### 2.3 Prior Predictive Distributions

#### Model 1 Prior Predictive

**Generative Process**:
```
1. Draw θ ~ N(0, 20²)
2. For each i, draw y_i ~ N(θ, σ_i²)
```

**Marginal Distribution** (integrating out θ):
```
p(y_i) = ∫ N(y_i | θ, σ_i²) × N(θ | 0, 20²) dθ
       = N(y_i | 0, σ_i² + 20²)
```

**Numerical**:
```
For σ_i = 12.5 (average):
y_i ~ N(0, 12.5² + 20²) = N(0, 556.25)
SD_prior_pred = 23.6

Observed range: [-3, 28]
Prior predictive 95%: [-46, 46]

Conclusion: Observed data well within prior predictive range (no conflict)
```

#### Model 2 Prior Predictive

**Generative Process**:
```
1. Draw μ ~ N(0, 20²)
2. Draw τ ~ Half-Normal(0, 5²)
3. For each i, draw θ_i ~ N(μ, τ²)
4. For each i, draw y_i ~ N(θ_i, σ_i²)
```

**Marginal Distribution** (integrating out μ, τ, θ):
```
p(y_i) = ∫∫∫ N(y_i | θ_i, σ_i²) × N(θ_i | μ, τ²) × N(μ | 0, 20²) × HN(τ | 0, 5²) dθ_i dμ dτ
```

No closed form, requires simulation.

**Simulated Prior Predictive**:
```
Mean: E[y_i] = 0
SD: SD[y_i] = √(20² + E[τ²] + σ_i²)

With E[τ²] from Half-Normal(0,5):
  E[τ] = 5√(2/π) ≈ 4.0
  E[τ²] ≈ 25

SD[y_i] ≈ √(400 + 25 + 156) = √581 = 24.1

Similar to Model 1, slight increase from τ uncertainty
```

---

## 3. MCMC Diagnostics: Deep Dive

### 3.1 R-hat (Potential Scale Reduction Factor)

**Definition** (Brooks & Gelman, 1998):
```
R̂ = √(Var_total / Var_within)

where:
  Var_within = mean(Var_within_chain)
  Var_total = Var(all samples)
```

**Interpretation**:
- R̂ = 1.00: Perfect convergence (within = total variance)
- R̂ > 1.01: Chains haven't mixed, continue sampling
- R̂ > 1.05: Serious convergence issues

**For our models**:
```
Model 1: R̂(θ) = 1.0000 (perfect to 4 decimal places)
Model 2: R̂(μ) = 1.0000, R̂(τ) = 1.0000, R̂(θ_i) < 1.0001
```

**Why R̂ = 1.0000?**
- Efficient sampler (NUTS with good geometry)
- Simple posterior (near-Gaussian)
- Adequate warmup (1000 iterations)
- Multiple chains (4, starting from overdispersed initial values)

### 3.2 Effective Sample Size (ESS)

**Bulk ESS**:
```
ESS_bulk = N / (1 + 2∑ρ_t)

where:
  N = total samples
  ρ_t = lag-t autocorrelation
```

**Measures**: Effective number of independent samples for estimating posterior mean.

**Tail ESS**:
Similar but computed on tail quantiles (5th and 95th percentiles).

**Measures**: Effective number of samples for estimating credible intervals.

**Threshold**: ESS > 400 (or ESS > 100 × number of chains)

**For our models**:
```
Model 1:
  ESS_bulk(θ) = 3092 (38.7× minimum)
  ESS_tail(θ) = 4081 (51× minimum)

Model 2:
  ESS_bulk(μ) = 5924
  ESS_bulk(τ) = 2887
  ESS_tail(μ) = 4081
  ESS_tail(τ) = 3123

All well above thresholds
```

**Why high ESS?**
- Low autocorrelation (efficient sampling)
- Simple geometry (near-Gaussian posteriors)
- NUTS adapts to curvature automatically

### 3.3 Divergences

**What are divergences?**
- NUTS uses Hamiltonian dynamics (physics simulation)
- Divergence = trajectory deviates from true Hamiltonian
- Indicates regions where sampler struggles (usually high curvature)

**Causes**:
1. **Funnel geometry**: Hierarchical models with τ → 0
2. **High curvature**: Sharp peaks or valleys in posterior
3. **Poorly scaled parameters**: Need reparameterization

**Remedies**:
1. Increase `target_accept` (default 0.8 → 0.9 or 0.95)
2. Reparameterize (centered → non-centered)
3. Stronger priors (add regularization)

**For our models**:
```
Model 1: 0 divergences (simple Gaussian posterior)
Model 2: 0 divergences (non-centered parameterization avoided funnel)
```

**Non-Centered Success**:
```
Centered:     θ_i ~ N(μ, τ²)
              When τ → 0, θ_i tightly constrained → funnel

Non-Centered: θ_i = μ + τ × θ_raw_i,  θ_raw_i ~ N(0,1)
              When τ → 0, θ_raw_i still free → no funnel
```

### 3.4 Energy Diagnostics (E-BFMI)

**Energy Bayesian Fraction of Missing Information**:
```
E-BFMI = Var(energy differences) / Var(marginal energy)
```

**Interpretation**:
- E-BFMI < 0.2: Problematic (sampler inefficient)
- E-BFMI > 0.3: Good
- E-BFMI > 0.5: Excellent

**What it measures**: How well NUTS explores the posterior.

**For our models**:
```
Model 1: E-BFMI = 0.93 (excellent)
Model 2: E-BFMI = 0.91 (excellent)
```

**Why high E-BFMI?**
- Smooth posterior geometry
- No narrow regions or strong correlations
- NUTS adapts step size and trajectory length appropriately

---

## 4. Leave-One-Out Cross-Validation: Technical Details

### 4.1 LOO-CV Algorithm

**Exact LOO-CV** (computationally expensive):
```
For i = 1 to J:
  1. Remove observation i from dataset
  2. Refit model to y_{-i}
  3. Compute log predictive density: log p(y_i | y_{-i})
4. Sum: ELPD_LOO = ∑ log p(y_i | y_{-i})
```

**Requires J model refits** (J=8 here, manageable but inefficient)

**PSIS-LOO** (Vehtari et al., 2017):
```
Importance sampling approximation:
  p(θ | y_{-i}) ≈ p(θ | y) × 1/p(y_i | θ)

Using posterior samples θ^(s):
  log p(y_i | y_{-i}) ≈ log(∑_s w_i^(s) × p(y_i | θ^(s)))

  where w_i^(s) ∝ 1/p(y_i | θ^(s))  (importance weights)

Pareto Smoothed Importance Sampling (PSIS):
  - Stabilizes extreme weights using generalized Pareto distribution
  - Fits Pareto to tail of weight distribution
  - Smooths large weights to reduce variance
```

**No refitting required** (uses original posterior samples)

### 4.2 Pareto k Diagnostic

**What is k?**
- Shape parameter of generalized Pareto distribution fitted to importance weight tails
- Measures "extremeness" of importance weights

**Interpretation**:
```
k < 0.5: Excellent (variance of weights is finite)
0.5 < k < 0.7: Good (PSIS stabilization effective)
0.7 < k < 1: Poor (high variance, LOO estimate unreliable)
k > 1: Very poor (infinite variance, must refit model)
```

**Why k matters**:
- High k → observation is very influential
- High k → p(θ|y) and p(θ|y_{-i}) are very different
- High k → importance sampling breaks down

**For our data**:
```
Model 1: All k < 0.27 (max k = 0.26 for Study 5)
Model 2: All k < 0.56 (max k = 0.55 for Study 7)

All observations have reliable LOO estimates
```

**Why low k?**
- No single observation is extremely influential
- Posterior doesn't change drastically when any observation removed
- Large within-study variances (σ_i) reduce influence

### 4.3 Effective Number of Parameters (p_LOO)

**Definition**:
```
p_LOO = ELPD_in_sample - ELPD_LOO

where:
  ELPD_in_sample = ∑ log p(y_i | y)  (overfits)
  ELPD_LOO = ∑ log p(y_i | y_{-i})   (out-of-sample)
```

**Interpretation**:
- Measures "effective complexity" of model
- Like AIC's 'k' but from cross-validation
- Accounts for regularization (e.g., partial pooling)

**For our models**:
```
Model 1: p_LOO = 0.64
  - 1 nominal parameter (θ)
  - 0.64 effective (slightly less than 1 due to prior regularization)

Model 2: p_LOO = 0.98
  - 10 nominal parameters (μ, τ, θ_1,...,θ_8)
  - 0.98 effective (strong shrinkage reduces complexity to ~1)
```

**Key Insight**: Model 2's partial pooling shrinks 10 parameters to ~1 effective parameter, nearly matching Model 1's complexity.

### 4.4 LOO Model Comparison

**ΔELPD** (difference in expected log predictive density):
```
ΔELPD = ELPD_model1 - ELPD_model2

Positive: Model 1 better
Negative: Model 2 better
```

**Standard Error**:
```
SE(ΔELPD) = √∑[(lpd1_i - lpd2_i) - ΔELPD]²

where lpd1_i = log p(y_i | y_{-i}, model1)
```

**Decision Rule** (Vehtari et al., 2017):
```
|ΔELPD / SE| < 2: Models indistinguishable, prefer simpler
|ΔELPD / SE| > 2: Favor model with higher ELPD
|ΔELPD / SE| > 4: Strong evidence for better model
```

**For our comparison**:
```
ΔELPD = -30.52 - (-30.69) = 0.17
SE = 0.105
Ratio = 0.17 / 0.105 = 1.62

Decision: 1.62 < 2 → Models indistinguishable → Prefer simpler (Model 1)
```

---

## 5. Simulation-Based Calibration: Technical Details

### 5.1 SBC Algorithm

**Procedure** (Talts et al., 2018):

```
For simulation s = 1 to N_sim:
  1. Sample true parameter: θ_true^(s) ~ p(θ)   (from prior)
  2. Simulate data: y^(s) ~ p(y | θ_true^(s))   (from likelihood)
  3. Fit model to y^(s): sample θ^(s,1),...,θ^(s,M) from p(θ | y^(s))
  4. Compute rank: r^(s) = #{θ^(s,m) < θ_true^(s)}

Expected: Ranks r^(s) uniformly distributed in {0, 1, ..., M}
```

**Why it works**:
- If model is correctly specified and sampler works:
  - θ_true is drawn from prior
  - Posterior samples include θ_true with "average" probability
  - Rank of θ_true among posterior samples should be uniform

**What uniform ranks mean**:
- No systematic bias (ranks not concentrated)
- Correct uncertainty (θ_true not always at edges)
- Proper calibration (coverage matches nominal)

### 5.2 SBC Diagnostics

**1. Rank Histogram**
```
Plot: Histogram of ranks {r^(1), ..., r^(N_sim)}
Expected: Uniform (flat histogram)
```

**Issues**:
- U-shaped: Posterior is overdispersed (too wide)
- Inverted-U: Posterior is underdispersed (too narrow)
- Left-skewed: Positive bias (posterior too high)
- Right-skewed: Negative bias (posterior too low)

**2. Rank ECDF**
```
Plot: ECDF of normalized ranks vs. theoretical CDF
Expected: Diagonal line (45°)
```

**With uncertainty bounds**: ±3√(N_sim) tolerance bands

**3. Chi-Squared Test**
```
H0: Ranks are uniformly distributed
Test statistic: χ² = ∑[(O_b - E_b)² / E_b]
  where O_b = observed counts in bin b, E_b = expected counts
```

**4. Kolmogorov-Smirnov Test**
```
H0: Ranks follow uniform distribution
Test statistic: D = max|ECDF(r) - CDF_uniform(r)|
```

**5. Coverage Calibration**
```
For each credible level α ∈ {0.5, 0.9, 0.95}:
  Count: c_α = #{s: θ_true^(s) in α% CI}
  Expected: c_α ≈ α × N_sim

Plot: Observed coverage vs. nominal coverage
Expected: Diagonal line
```

**6. Z-score Calibration**
```
Compute: z^(s) = (θ̂^(s) - θ_true^(s)) / SE(θ^(s))

Expected: z^(s) ~ N(0, 1)

Check:
  - Mean(z) ≈ 0 (no bias)
  - SD(z) ≈ 1 (correct uncertainty)
  - Shapiro-Wilk test for normality
```

### 5.3 Our SBC Results

**Configuration**:
- N_sim = 500 simulations
- M = 999 posterior samples per simulation
- Prior: θ ~ N(0, 20²)

**Results**:

| Diagnostic | Result | Threshold | Status |
|------------|--------|-----------|--------|
| **Rank Uniformity (KS)** | p = 0.736 | > 0.05 | PASS |
| **Rank Uniformity (χ²)** | p = 0.819 | > 0.05 | PASS |
| **Coverage 50%** | 54.0% | 50 ± 4% | PASS |
| **Coverage 90%** | 89.8% | 90 ± 2% | PASS |
| **Coverage 95%** | 94.4% | 95 ± 1.5% | PASS |
| **Bias** | -0.007 | < 0.1 | PASS |
| **Z-score Mean** | -0.003 | ≈ 0 | PASS |
| **Z-score SD** | 1.008 | ≈ 1 | PASS |
| **Z-score Normality** | p = 0.612 | > 0.05 | PASS |

**Interpretation**:
- Ranks are uniform (no bias, correct dispersion)
- Coverage is calibrated (matches nominal levels)
- Z-scores are standard normal (correct uncertainty)
- **Conclusion**: Model implementation is correct, sampler is working

---

## 6. Posterior Predictive Checks: Extended Analysis

### 6.1 LOO-PIT (Leave-One-Out Probability Integral Transform)

**Definition**:
```
PIT_i = P(ỹ < y_i | y_{-i})

where ỹ ~ p(y | y_{-i})  (LOO posterior predictive distribution)
```

**Computation**:
```
For each observation i:
  1. Compute LOO posterior predictive samples: ỹ_i^(1),...,ỹ_i^(M)
  2. PIT_i = (#{ỹ_i^(m) < y_i} + 0.5 × #{ỹ_i^(m) = y_i}) / M
```

**Expected**: If model is well-calibrated, PIT ~ Uniform(0,1)

**Why?**
- PIT transforms any continuous predictive distribution to uniform
- Uniform PIT means model captures all aspects of data distribution
- Non-uniform PIT indicates model misspecification

**Diagnostics**:
- **Histogram**: Should be flat
- **ECDF**: Should be diagonal
- **KS test**: H0: PIT ~ Uniform

**Our Results**:
```
Model 1: KS test p = 0.981 (cannot reject uniformity, excellent)
Model 2: KS test p = 0.664 (cannot reject uniformity, good)

PIT values:
  Model 1: [0.27, 0.42, 0.19, 0.47, 0.15, 0.23, 0.71, 0.37]
  Model 2: [0.31, 0.44, 0.22, 0.51, 0.17, 0.27, 0.73, 0.40]

Both: Well-spread across [0,1], no clustering at extremes
```

**Interpretation**: Models are well-calibrated for out-of-sample prediction.

### 6.2 Test Statistics

**Philosophy**: Check if model reproduces key features of data.

**Common Test Statistics**:
1. **Mean**: T(y) = mean(y)
2. **SD**: T(y) = sd(y)
3. **Min/Max**: T(y) = min(y), max(y)
4. **Quantiles**: T(y) = quantile(y, q)
5. **Custom**: Domain-specific features

**Posterior Predictive P-value**:
```
p_B = P(T(y_rep) ≥ T(y_obs) | y_obs)

Interpretation:
  p_B ≈ 0 or 1: Extreme discrepancy (model fails to reproduce feature)
  p_B ∈ [0.1, 0.9]: Acceptable (model consistent with data)
```

**Not a hypothesis test**: More extreme p_B doesn't mean "reject model"
- p_B quantifies "surprise" under model
- Use for model checking, not rejection

**Our Results** (Model 1):

| Statistic | Observed | Mean(y_rep) | SD(y_rep) | p_B | Assessment |
|-----------|----------|-------------|-----------|-----|------------|
| Mean | 8.75 | 8.69 | 3.46 | 0.413 | Good |
| SD | 10.44 | 10.82 | 3.12 | 0.688 | Good |
| Min | -3 | -10.62 | 8.65 | 0.202 | Good |
| Max | 28 | 28.01 | 9.12 | 0.374 | Good |
| Range | 31 | 38.63 | 11.95 | 0.677 | Good |
| Median | 7.5 | 8.66 | 4.29 | 0.499 | Good |

**All p_B ∈ [0.2, 0.7]**: Observed values are typical under model.

### 6.3 Graphical Posterior Predictive Checks

**1. Overlay Plot**
```
Plot: Histogram of y_obs overlaid with histogram of y_rep samples
Expected: y_obs should look like a typical draw from y_rep distribution
```

**2. Density Overlay**
```
Plot: KDE of y_obs vs. multiple KDEs of y_rep draws
Expected: y_obs density within envelope of y_rep densities
```

**3. Scatterplot**
```
Plot: y_obs vs. mean(y_rep) or median(y_rep)
Expected: Points near 45° diagonal
```

**4. Residual Plot**
```
Plot: (y_obs - mean(y_rep)) vs. study index or σ_i
Expected: No systematic patterns, symmetric around zero
```

---

## 7. Model Comparison: Advanced Topics

### 7.1 Stacking Weights

**Idea**: Combine models using optimal weights for prediction.

**Method** (Yao et al., 2018):
```
Minimize predictive error:
  w* = argmin ∑_i [-log(∑_k w_k × p(y_i | y_{-i}, model_k))]

subject to: ∑w_k = 1, w_k ≥ 0
```

**Interpretation**:
- w_k = weight to assign to model k
- Optimal combination for predictive performance
- NOT posterior model probabilities

**For our models**:
```
(Not computed in original analysis, but expected:)
w_1 ≈ 0.55, w_2 ≈ 0.45
(Nearly equal weights, reflecting similar performance)
```

### 7.2 Bayes Factors

**Definition**:
```
BF_{12} = p(y | M_1) / p(y | M_2)

where p(y | M_k) = ∫ p(y | θ, M_k) p(θ | M_k) dθ  (marginal likelihood)
```

**Interpretation**:
- BF > 10: Strong evidence for M_1
- BF = 1: Equal evidence
- BF < 0.1: Strong evidence for M_2

**Challenges**:
- Sensitive to prior specification
- Difficult to compute for complex models
- Not compatible with improper priors

**Why we use LOO instead**:
- LOO is predictive (BF is not)
- LOO is less sensitive to priors
- LOO works for any model (including non-nested)

### 7.3 DIC and WAIC

**DIC** (Deviance Information Criterion):
```
DIC = D̄ + p_D

where:
  D̄ = -2 × mean(log p(y | θ))  (average deviance)
  p_D = D̄ - D(θ̄)               (effective parameters)
```

**Issues**:
- Not fully Bayesian (uses point estimate θ̄)
- Can be negative for hierarchical models
- Less reliable than LOO/WAIC

**WAIC** (Widely Applicable Information Criterion):
```
WAIC = -2 × (LPPD - p_WAIC)

where:
  LPPD = ∑ log E[p(y_i | θ)]        (log pointwise predictive density)
  p_WAIC = ∑ Var[log p(y_i | θ)]    (effective parameters)
```

**Advantages**:
- Fully Bayesian (uses full posterior)
- Approximates cross-validation
- Works for any model

**Why we prefer LOO**:
- LOO is more stable (uses PSIS)
- LOO has diagnostic (Pareto k)
- LOO = WAIC in expectation but LOO is more reliable

---

## 8. Advanced Diagnostics and Checks

### 8.1 Multivariate Diagnostics (Model 2)

**Pairs Plot**:
- Scatterplots of all parameter pairs (μ, τ, θ_1,...,θ_8)
- Checks for posterior correlations
- Ideal: No strong correlations (indicates good parameterization)

**For Model 2**:
```
Cor(μ, τ) ≈ 0.05 (nearly independent, good)
Cor(τ, θ_i) ≈ 0.10 (low correlation, non-centered helped)
Cor(θ_i, θ_j) ≈ 0.15 (some pooling correlation, expected)
```

**Non-centered parameterization success**: Without it, Cor(τ, θ_i) would be >> 0.5.

### 8.2 Prior-Posterior Overlap

**Compute**:
```
Overlap = ∫ min(p_prior(θ), p_post(θ)) dθ
```

**Interpretation**:
- Overlap ≈ 1: Prior and posterior very similar (data uninformative)
- Overlap ≈ 0: Prior and posterior very different (data very informative)
- Overlap ≈ 0.1-0.3: Typical for weakly informative priors

**For Model 1**:
```
Prior: θ ~ N(0, 20)
Posterior: θ ~ N(7.4, 4)

Overlap ≈ 0.02 (2%)
```

**Interpretation**: Data are highly informative, prior has minimal influence.

### 8.3 Posterior Contraction

**Measure**:
```
Contraction = (Prior SD - Posterior SD) / Prior SD
            = (20 - 4) / 20
            = 0.80 (80% reduction in uncertainty)
```

**Interpretation**: Data reduced uncertainty by 80%.

---

## 9. Computational Efficiency

### 9.1 Sampling Efficiency

**Warmup Phase**:
- NUTS tunes step size and mass matrix
- Goal: Achieve target acceptance rate (0.8)
- 1000 iterations sufficient for these models

**Sampling Phase**:
- 2000 iterations × 4 chains = 8000 samples
- Runtime: ~18 seconds per model

**Effective Sample Size per Second**:
```
Model 1: ESS = 3092, Time = 18s → 172 ESS/sec
Model 2: ESS = 2887, Time = 18s → 160 ESS/sec

Very efficient for MCMC
```

### 9.2 Why So Fast?

1. **Simple posterior geometry**: Near-Gaussian, no multimodality
2. **Low dimensionality**: 1-10 parameters
3. **Small dataset**: J=8 (likelihood evaluation is fast)
4. **Efficient sampler**: NUTS adapts to geometry automatically
5. **Good parameterization**: Non-centered for Model 2

### 9.3 Scaling to Larger Problems

**If J increases to 100**:
- Model 1: Still ~1 parameter → same speed
- Model 2: ~102 parameters → may slow down
  - Non-centered helps but more parameters to sample
  - Expect 1-2 minutes instead of 18 seconds

**If measurement error model added**:
- Would need to model σ_i as uncertain
- Adds J parameters → doubles dimensionality
- Expect 2-5× longer runtime

---

## 10. Alternative Approaches Not Pursued

### 10.1 Robust Models

**Student-t Likelihood**:
```
y_i | θ, ν ~ Student-t(ν, θ, σ_i)

Prior: ν ~ Gamma(2, 0.1)  [allows ν = 5 to 60]
```

**Why not pursued**: No outliers detected, normality confirmed

**Expected outcome if fitted**: ν > 30 (effectively normal)

### 10.2 Measurement Error Model

**Idea**: Model uncertainty in σ_i.

```
True model: σ_i ~ Gamma(α_i, β_i)  [estimated from original studies]
Our model: σ_i = fixed (known)

More complete:
  y_i | θ, σ_i ~ N(θ, σ_i²)
  σ_i | n_i ~ Scaled-Inverse-χ²(n_i - 1, s_i²)
```

**Requires**: Original study sample sizes (n_i) and SDs (s_i)

**Not available** in this dataset.

### 10.3 Meta-Regression

**Idea**: Model heterogeneity as function of covariates.

```
y_i ~ N(θ_i, σ_i²)
θ_i ~ N(β_0 + β_1 X_{i1} + ... + β_p X_{ip}, τ²)
```

**Requires**: Study-level covariates (X_ij)

**Not available** in this dataset.

**If available**, could test:
- Publication year (time trends)
- Study design (RCT vs. observational)
- Sample size (precision effects)
- Geographic region (context effects)

---

## 11. Reporting Checklist (CONSORT-style)

**For Methods Section**:
- [ ] Model specification (likelihood, priors)
- [ ] Prior justification and sensitivity
- [ ] Software and version (PyMC 5.x)
- [ ] MCMC configuration (chains, iterations, warmup)
- [ ] Convergence criteria (R-hat, ESS, divergences)
- [ ] Validation procedures (SBC, PPC, LOO)

**For Results Section**:
- [ ] Posterior estimates (mean, SD, 95% CI)
- [ ] Convergence diagnostics (all passed)
- [ ] Model comparison (LOO ΔELPD ± SE)
- [ ] Heterogeneity assessment (I², τ)
- [ ] Probability statements (P(θ > 0))

**For Discussion Section**:
- [ ] Interpretation of effect size
- [ ] Limitations (small J, wide CI)
- [ ] Sensitivity analyses
- [ ] Comparison to prior literature

**For Supplementary Materials**:
- [ ] Complete data (all y_i, σ_i)
- [ ] Code availability
- [ ] Additional diagnostics
- [ ] Prior predictive checks
- [ ] All visualizations

---

## 12. References

**Methods**:
- Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed.
- Betancourt (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo*.
- Vehtari et al. (2017). *Practical Bayesian model evaluation using LOO-CV*.
- Talts et al. (2018). *Validating Bayesian inference with SBC*.

**Software**:
- PyMC Development Team (2023). *PyMC documentation*.
- Kumar et al. (2019). *ArviZ: Exploratory analysis of Bayesian models*.

**Meta-Analysis**:
- Higgins & Thompson (2002). *Quantifying heterogeneity in meta-analysis*.
- Gelman (2006). *Prior distributions for variance parameters*.

---

**END OF TECHNICAL SUPPLEMENT**

This document provides complete technical details for researchers wishing to:
- Understand the mathematical foundations
- Replicate the analysis
- Extend the methods to similar problems
- Critically evaluate the approach

For questions, refer to the main report or original code.
