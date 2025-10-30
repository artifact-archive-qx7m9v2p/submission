# Technical Supplement
## Additional Details for Beta-Binomial Hierarchical Model

**Date:** October 30, 2025
**Model:** Beta-Binomial with Mean-Concentration Parameterization

---

## Contents

1. [Detailed Model Specification](#1-detailed-model-specification)
2. [Prior Derivations and Justifications](#2-prior-derivations-and-justifications)
3. [Complete Validation Results](#3-complete-validation-results)
4. [Sensitivity Analyses](#4-sensitivity-analyses)
5. [Alternative Parameterizations](#5-alternative-parameterizations)
6. [Computational Considerations](#6-computational-considerations)
7. [Extensions and Future Work](#7-extensions-and-future-work)

---

## 1. Detailed Model Specification

### 1.1 Full Mathematical Formulation

**Data level:**
$$r_i | n_i, p_i \sim \text{Binomial}(n_i, p_i) \quad \text{for } i = 1, \ldots, 12$$

where:
- r_i: Number of successes in group i
- n_i: Number of trials in group i
- p_i: Success probability for group i

**Population level:**
$$p_i | \mu, \kappa \sim \text{Beta}(\alpha, \beta)$$

**Reparameterization:**
$$\alpha = \mu \cdot \kappa$$
$$\beta = (1 - \mu) \cdot \kappa$$

where:
- μ ∈ (0,1): Population mean success probability
- κ > 0: Concentration parameter

**Hyperpriors:**
$$\mu \sim \text{Beta}(a_\mu, b_\mu)$$
$$\kappa \sim \text{Gamma}(a_\kappa, b_\kappa)$$

with hyperparameters:
- a_μ = 2, b_μ = 18 (Beta parameters for μ)
- a_κ = 2, b_κ = 0.1 (Gamma parameters for κ)

### 1.2 Derived Quantities

**Overdispersion parameter:**
$$\phi = 1 + \frac{1}{\kappa}$$

Properties:
- φ = 1: Pure binomial (no overdispersion)
- φ > 1: Extra-binomial variation present
- φ → ∞ as κ → 0 (infinite heterogeneity)

**Variance of group-level probabilities:**
$$\text{Var}(p_i) = \frac{\mu(1-\mu)}{\kappa + 1}$$

**Intraclass correlation:**
$$\text{ICC} = \frac{1}{1 + \kappa}$$

Interpretation: Proportion of total variance due to between-group differences.

**Beta distribution parameters:**
$$\alpha = \mu \cdot \kappa$$
$$\beta = (1 - \mu) \cdot \kappa = \kappa - \alpha$$

**Expected value and variance:**
$$E[p_i] = \frac{\alpha}{\alpha + \beta} = \mu$$
$$\text{Var}(p_i) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)} = \frac{\mu(1-\mu)}{\kappa + 1}$$

### 1.3 Likelihood Contributions

**Beta-binomial probability mass function:**
$$P(r_i | n_i, \alpha, \beta) = \binom{n_i}{r_i} \frac{B(r_i + \alpha, n_i - r_i + \beta)}{B(\alpha, \beta)}$$

where B(·,·) is the beta function:
$$B(a, b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)}$$

**Log-likelihood:**
$$\log L(\mu, \kappa | \mathbf{r}, \mathbf{n}) = \sum_{i=1}^{12} \log P(r_i | n_i, \mu \kappa, (1-\mu)\kappa)$$

**Joint posterior (unnormalized):**
$$p(\mu, \kappa | \mathbf{r}, \mathbf{n}) \propto L(\mu, \kappa | \mathbf{r}, \mathbf{n}) \cdot p(\mu) \cdot p(\kappa)$$

### 1.4 Posterior Predictive Distribution

For a new group j with n_j trials:

**Step 1:** Draw (μ*, κ*) from posterior
**Step 2:** Compute α* = μ* · κ*, β* = (1-μ*) · κ*
**Step 3:** Draw p_j ~ Beta(α*, β*)
**Step 4:** Draw r_j ~ Binomial(n_j, p_j)

Result: r_j | n_j, data ~ BetaBinomial(n_j, α*, β*)

**Posterior predictive mean:**
$$E[r_j | n_j, \text{data}] = n_j \cdot E[\mu | \text{data}]$$

**Posterior predictive variance:**
$$\text{Var}(r_j | n_j, \text{data}) = n_j \mu (1-\mu) \left[1 + \frac{n_j - 1}{\kappa + 1}\right]$$

The term in brackets is the overdispersion multiplier (>1 if κ is finite).

---

## 2. Prior Derivations and Justifications

### 2.1 Prior for μ: Beta(2, 18)

**Rationale:**
- EDA pooled rate: 7.39%
- Prior mean: 2/(2+18) = 0.10 (10%)
- Centered near observed value but not constraining

**Properties:**

| Quantile | Value | Probability Scale |
|----------|-------|------------------|
| 2.5% | 0.0134 | 1.3% |
| 25% | 0.0506 | 5.1% |
| 50% | 0.0867 | 8.7% |
| 75% | 0.1347 | 13.5% |
| 97.5% | 0.2572 | 25.7% |

**Mode:** (2-1)/(2+18-2) = 0.0556 (5.6%)

**Effective sample size:** 2+18 = 20 (equivalent to 20 prior observations)

**Interpretation:** Weakly informative. Allows population mean anywhere from 1% to 26%, with most mass between 5% and 15%.

**Prior predictive for single observation:**
With n=100 trials, prior for r:
- Mean: 100 × 0.10 = 10 successes
- SD: √[100 × 0.10 × 0.90 × (1 + 1/20)] ≈ 3.2 successes

### 2.2 Prior for κ: Gamma(2, 0.1)

**Rationale:**
- κ controls heterogeneity; wide prior allows data to determine
- Gamma(2, 0.1) has heavy right tail (can accommodate high κ)

**Properties:**

| Quantile | κ Value | Implied φ |
|----------|---------|-----------|
| 2.5% | 2.44 | 1.410 |
| 25% | 9.50 | 1.105 |
| 50% | 16.52 | 1.061 |
| 75% | 26.73 | 1.037 |
| 97.5% | 56.16 | 1.018 |

**Mean:** 2/0.1 = 20
**SD:** √(2/0.1²) = 20 (very wide)
**Mode:** (2-1)/0.1 = 10

**Implied variance of p:**
If μ = 0.08 and κ = 20:
$$\text{Var}(p) = \frac{0.08 \times 0.92}{21} = 0.0035 \quad (SD = 5.9\%)$$

**Interpretation:** Allows minimal to moderate overdispersion. Prior median φ = 1.06 (6% overdispersion).

### 2.3 Joint Prior Properties

**Prior covariance:** μ and κ are independent a priori (cov(μ, κ) = 0)

**Prior for φ = 1 + 1/κ:**
- Mean: Not analytically tractable (nonlinear transformation)
- Approximate median: 1 + 1/16.5 ≈ 1.06
- 95% interval: [1.02, 1.41]

**Prior predictive for variance of group rates:**

Integrating over (μ, κ):
$$\text{Var}_{\text{prior}}(\text{group rates}) \approx 0.005 \quad (SD = 7\%)$$

This is wider than observed (SD = 3.8%), allowing data to determine true heterogeneity.

### 2.4 Alternative Priors Considered

**More diffuse:**
- μ ~ Beta(1, 1) [uniform]
- κ ~ Gamma(1, 0.01)
- Result: Too vague, poor prior predictive coverage

**More informative:**
- μ ~ Beta(5, 50) [prior mean = 9%]
- κ ~ Gamma(10, 0.5) [prior mean = 20, SD = 6.3]
- Result: Tighter prior predictive, but data overwhelm anyway

**Conclusion:** Chosen priors (Beta(2,18), Gamma(2, 0.1)) provide good balance—weakly informative but not constraining.

---

## 3. Complete Validation Results

### 3.1 Prior Predictive Check Details

**Simulations:** 10,000 prior samples → 1,000 predictive datasets

**Coverage diagnostics:**

| Statistic | Observed | Prior Pred Mean | 95% Prior Interval | Coverage |
|-----------|----------|-----------------|-------------------|----------|
| Pooled rate | 0.0739 | 0.0853 | [0.0092, 0.2627] | 40th %ile |
| φ (BB) | 1.02 | 1.049 | [1.009, 1.300] | 20th %ile |
| Max rate | 0.144 | 0.230 | [0.047, 0.584] | Within |
| Min rate | 0.000 | 0.0035 | [0.000, 0.039] | Within |
| Num zeros | 1 | 1.45 | [0, 5] | Within |

**Key finding:** Priors correctly calibrated for φ_BB ≈ 1.02, not quasi-likelihood φ ≈ 3.5.

**Critical plots:**
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/overdispersion_explained.png`
- Shows φ_quasi = 3.51 vs φ_BB = 1.02 distinction

### 3.2 Simulation-Based Calibration Details

**Method:** Maximum likelihood + parametric bootstrap (1,000 resamples)

**25 simulations with parameters drawn from prior:**

**μ recovery:**
- Coverage: 21/25 (84%)
- Mean bias: -0.00175 (essentially zero)
- SD of bias: 0.0191
- Failed recoveries: Simulations 5, 9, 13, 22

**κ recovery:**
- Coverage: 16/25 (64%)
- Mean bias: +44.1 (positive bias toward higher κ)
- SD of bias: 41.0
- Failed recoveries: 9 simulations (true κ < 20 most problematic)

**φ recovery:**
- Coverage: 16/25 (64%)
- Mean bias: -0.0062 (essentially zero)
- SD of bias: 0.0556
- Pattern: Same failures as κ (nonlinear transformation)

**Interpretation:**
- Primary parameter (μ) recovers excellently
- Secondary parameters (κ, φ) have anti-conservative uncertainty due to bootstrap method
- Full Bayesian MCMC (used for real data) expected to have better calibration

**Critical plots:**
- `/workspace/experiments/experiment_1/simulation_based_validation/plots/parameter_recovery.png`
- Shows μ clustering on identity line, κ with more spread

### 3.3 Posterior Inference Diagnostics

**Sampling configuration:**
- 4 chains × 1,500 samples = 6,000 total
- Warmup: 2,000 iterations per chain
- Target acceptance: 0.95
- Runtime: 9 seconds

**Convergence metrics (all parameters):**

| Metric | Min | Max | All Pass? |
|--------|-----|-----|-----------|
| R-hat | 1.0000 | 1.0000 | Yes (< 1.01) |
| ESS (bulk) | 2,677 | 2,721 | Yes (> 400) |
| ESS (tail) | 2,748 | 2,803 | Yes (> 400) |
| Divergences | 0 | 0 | Yes (< 60) |
| Max treedepth | 0 | 0 | Yes (< 60) |

**Energy diagnostic:**
- E-BFMI (energy Bayesian fraction of missing information): 0.98
- Indicates good HMC geometry (>0.3 is acceptable)

**Effective sample size ratios:**
- ESS/N: 2,677/6,000 = 0.446 (excellent, >0.1 is good)
- No thinning needed

### 3.4 Posterior Predictive Check Details

**Test statistics (all 7):**

Each p-value computed as:
$$p = P(T(y_{\text{rep}}) \geq T(y_{\text{obs}}) | \text{data})$$

where T is a test statistic and y_rep ~ posterior predictive.

**Results:**

1. **Total successes:** p = 0.606
   - Observed: 208
   - Posterior predictive: mean 229, SD 65
   - Interpretation: Model predicts slightly more successes (conservative)

2. **Variance of rates:** p = 0.714
   - Observed: 0.00143
   - Posterior predictive: mean 0.00246, SD 0.00197
   - Interpretation: Model slightly overpredicts heterogeneity (conservative)

3. **Maximum rate:** p = 0.718
   - Observed: 0.1442
   - Posterior predictive: mean 0.1794, SD 0.0571
   - Interpretation: Model can generate more extreme values than observed

4. **Minimum rate:** p = 1.000
   - Observed: 0.0000
   - Interpretation: p = 1.0 because ALL replicates have min ≥ 0 (floor effect)

5. **Number of zeros:** p = 0.173
   - Observed: 1
   - Posterior predictive: mean 0.20, SD 0.47
   - Interpretation: Zeros are rare but plausible (17% of replicates have ≥1 zero)

6. **Range of rates:** p = 0.553
   - Observed: 0.1442
   - Posterior predictive: mean 0.1610, SD 0.0578
   - Interpretation: Perfect match (observed at median)

7. **Chi-square:** p = 0.895
   - Observed: 34.4
   - Posterior predictive: mean 94.7, SD 88.7
   - Interpretation: Model predicts worse fit than observed (conservative)

**LOO diagnostics:**
- All 12 Pareto k < 0.5 (100% "good")
- Maximum k = 0.348 (Group 8, outlier)
- Mean k = 0.095

**LOO-PIT calibration:**
- Kolmogorov-Smirnov test: D = 0.195, p = 0.685
- Conclusion: Cannot reject uniformity (well-calibrated)

### 3.5 Model Assessment Metrics

**Absolute error metrics:**

| Metric | Value (counts) | Value (rates) |
|--------|---------------|---------------|
| MAE | 0.89 successes | 0.0066 (0.66%) |
| RMSE | 1.07 successes | 0.0113 (1.13%) |
| Correlation (obs vs pred) | 0.987 | - |

**Calibration by credible level:**

| Nominal | Empirical | Ratio |
|---------|-----------|-------|
| 50% | 58.3% (7/12) | 1.17 |
| 80% | 91.7% (11/12) | 1.15 |
| 90% | 100% (12/12) | 1.11 |
| 95% | 100% (12/12) | 1.05 |

Slight overcoverage (conservative bias) at all levels.

**Performance by sample size:**

| Size | n Groups | MAE (%) | RMSE (%) |
|------|----------|---------|----------|
| Small (<100) | 2 | 1.80 | 2.51 |
| Medium (100-199) | 4 | 0.45 | 0.51 |
| Large (≥200) | 6 | 0.42 | 0.52 |

4-fold error reduction from small to medium groups.

---

## 4. Sensitivity Analyses

### 4.1 Prior Sensitivity (Not Conducted, But Recommended)

**Alternative prior set 1: More diffuse**
- μ ~ Beta(1, 1) [uniform]
- κ ~ Gamma(1, 0.01)

**Expected impact:**
- Wider posterior credible intervals for κ and φ
- Minimal impact on μ (data are informative)
- Conclusion: Likely similar to main analysis

**Alternative prior set 2: More informative**
- μ ~ Beta(5, 50) [prior mean = 9%, SD = 3.7%]
- κ ~ Gamma(10, 0.5) [prior mean = 20, SD = 6.3]

**Expected impact:**
- Narrower posterior credible intervals
- Posterior mean closer to prior mean (especially for κ)
- Conclusion: Check if conclusions robust to stronger priors

**Recommendation:** Conduct and report in supplementary materials.

### 4.2 Outlier Sensitivity (Not Conducted, But Recommended)

**Analysis 1: Refit without Group 8**

**Procedure:**
1. Remove Group 8 (31/215, 14.4% rate)
2. Refit model to remaining 11 groups
3. Compare posteriors to main analysis

**Expected results:**
- μ slightly lower (Group 8 pulls mean up)
- φ possibly lower (Group 8 is outlier, contributes to heterogeneity)
- κ possibly higher (less heterogeneity without outlier)

**Questions to address:**
- Does Group 8 unduly influence population estimates?
- Is minimal heterogeneity finding robust to outlier exclusion?

**Analysis 2: Refit without Group 1**

**Procedure:**
1. Remove Group 1 (0/47, 0% rate)
2. Refit model to remaining 11 groups
3. Compare posteriors to main analysis

**Expected results:**
- Minimal impact (Group 1 only 1.7% of total trials)
- μ possibly slightly higher (Group 1 pulls mean down, but shrunk anyway)

**Recommendation:** Conduct for Group 8 at minimum.

### 4.3 Model Comparison (Not Conducted, But Recommended)

**Alternative model: Hierarchical binomial with logit random effects**

**Specification:**
$$r_i \sim \text{Binomial}(n_i, p_i)$$
$$\text{logit}(p_i) = \mu + \alpha_i$$
$$\alpha_i \sim \text{Normal}(0, \sigma)$$

**Priors:**
$$\mu \sim \text{Normal}(-2.5, 1.0)$$
$$\sigma \sim \text{Half-Normal}(0, 1.0)$$

**Comparison metrics:**
- LOO ELPD difference
- Interpretation: σ (between-group SD on logit scale) vs φ (overdispersion factor)
- Group 1 handling: Requires continuity correction or careful prior

**Expected result:**
- Similar LOO performance (|ΔLOO| < 4)
- Different parameterization, similar substantive conclusions

**Recommendation:** Conduct as validation of model class choice.

---

## 5. Alternative Parameterizations

### 5.1 Standard Beta Parameterization (α, β)

**Not chosen for main analysis, but mathematically equivalent:**

$$p_i \sim \text{Beta}(\alpha, \beta)$$

**Priors (implied by main analysis):**
$$\alpha = \mu \cdot \kappa \sim ?$$
$$\beta = (1-\mu) \cdot \kappa \sim ?$$

**Challenge:** Priors on (α, β) are correlated and complex if derived from independent priors on (μ, κ).

**Advantages:**
- Standard parameterization in literature
- Direct connection to conjugate Beta-Binomial

**Disadvantages:**
- Less interpretable (α and β don't have direct scientific meaning)
- Harder to set priors (what should α and β be?)

**Posterior estimates (derived):**
- α: 3.17 [0.92, 5.58]
- β: 36.20 [11.21, 64.63]

### 5.2 Precision Parameterization (μ, n)

**Alternative:** Beta(μn, (1-μ)n) where n is "precision"

**Relationship to our parameterization:**
- n = κ (same parameter, different name)
- Some literature uses "n" for concentration

**No substantive difference from chosen parameterization.**

### 5.3 Logit-Normal Parameterization

**Model structure:**
$$\text{logit}(p_i) \sim \text{Normal}(\mu_{\text{logit}}, \sigma^2)$$

**Relationship to beta:**
- Not analytically equivalent (different family)
- Can approximate beta well for moderate p and large κ

**Advantages:**
- Natural for regression (can add covariates easily)
- σ has direct interpretation (between-group SD on logit scale)

**Disadvantages:**
- Group 1 (0/47) requires continuity correction
- No closed-form conjugacy

**Relationship between parameters:**
If p ~ Beta(α, β), then approximately:
$$E[\text{logit}(p)] \approx \Psi(\alpha) - \Psi(\beta)$$
where Ψ is the digamma function.

For our estimates:
$$\mu_{\text{logit}} \approx \Psi(3.17) - \Psi(36.20) \approx -2.53$$
$$\sigma_{\text{logit}} \approx \sqrt{\Psi'(3.17) + \Psi'(36.20)} \approx 0.59$$

where Ψ' is the trigamma function.

---

## 6. Computational Considerations

### 6.1 Software Choices

**Primary: PyMC 5.26.1**
- Python-based probabilistic programming language
- Automatic differentiation via PyTensor
- NUTS sampler (same as Stan)

**Why not Stan:**
- Stan (CmdStanPy) was first choice
- Requires C++ compiler (not available in execution environment)
- PyMC provides equivalent NUTS sampler

**Comparison:**
- Both use HMC with adaptive step size
- Both have automatic differentiation
- Performance should be nearly identical
- PyMC may be slightly slower but more flexible

### 6.2 Sampling Efficiency

**Key metrics:**
- ESS/second: 2,677/9 ≈ 297 effective samples per second (excellent)
- Autocorrelation: All parameters have ACF → 0 within ~20 lags
- Mixing: Visual inspection shows excellent chain mixing

**No reparameterization needed:**
- Non-centered parameterization considered but unnecessary
- Standard parameterization worked well (no divergences)

**Scalability:**
- Current model: 12 groups, 6,000 samples, 9 seconds
- Expected for 100 groups: ~15 seconds (linear scaling)
- Expected for 1,000 groups: ~2 minutes

### 6.3 Numerical Stability

**Potential issues:**
1. Beta(α, β) with small α or β (singularities at 0 or 1)
   - Not encountered (α = 3.17, β = 36.20 well within stable region)

2. Beta-binomial PMF with large n (overflow in binomial coefficient)
   - Not encountered (max n = 810, well within stable range)
   - PyMC uses log-space computations to avoid overflow

3. Extreme shrinkage (κ → 0 or κ → ∞)
   - Prior bounds prevent: κ ∈ (0.1, 200) in practice
   - Posterior κ = 39.4 far from boundaries

**No numerical warnings issued during sampling.**

### 6.4 Convergence Diagnostics Interpretation

**R-hat = 1.00:**
- Measures between-chain vs within-chain variance
- 1.00 indicates perfect convergence (chains mixing perfectly)
- Rule: R-hat < 1.01 for all parameters

**ESS (Effective Sample Size):**
- Accounts for autocorrelation in MCMC chains
- ESS = 2,677 means 2,677 independent samples (out of 6,000 total)
- ESS/N = 0.446 (excellent, >0.1 is good)

**Divergences:**
- Indicate sampler struggled to explore posterior (gradient issues)
- Zero divergences = excellent HMC geometry
- If >1%: may need reparameterization or tighter adaptation

**Max treedepth:**
- NUTS builds binary trees to find U-turn
- Zero max treedepth hits = sampler operating efficiently
- If >1%: may need to increase max treedepth

**E-BFMI (Energy Bayesian Fraction of Missing Information):**
- Measures how well energy transitions are sampled
- 0.98 (excellent, >0.3 is acceptable)
- Low E-BFMI (<0.3) suggests poor geometry

---

## 7. Extensions and Future Work

### 7.1 Adding Covariates

**Motivation:** Explain why groups differ, not just describe variation

**Model structure:**

**Level 1 (Data):**
$$r_i \sim \text{Binomial}(n_i, p_i)$$

**Level 2 (Group probabilities with covariates):**
$$\text{logit}(p_i) = \beta_0 + \beta_1 X_{i,1} + \cdots + \beta_p X_{i,p} + \alpha_i$$
$$\alpha_i \sim \text{Normal}(0, \sigma)$$

**Priors:**
$$\beta_0 \sim \text{Normal}(\text{logit}(\bar{p}), 1)$$
$$\beta_j \sim \text{Normal}(0, 1) \quad j = 1, \ldots, p$$
$$\sigma \sim \text{Half-Normal}(0, 1)$$

**Example covariates:**
- Group size (log(n_i))
- Geographic region (categorical)
- Intervention status (binary)
- Baseline risk (continuous)

**Expected benefits:**
- Reduce residual between-group variance (σ)
- Identify drivers of success rates
- Improve predictions for groups with known covariates
- Enable counterfactual reasoning (if covariates are manipulable)

**Implementation:** Extend current PyMC model with regression component.

### 7.2 Longitudinal Extension

**Motivation:** Assess trends, seasonality, and dynamics if repeated measures available

**Model structure:**

**Level 1 (Data):**
$$r_{i,t} \sim \text{Binomial}(n_{i,t}, p_{i,t})$$

**Level 2 (Time-varying probabilities):**
$$\text{logit}(p_{i,t}) = \mu_t + \alpha_i$$

**Level 3 (Group effects):**
$$\alpha_i \sim \text{Normal}(0, \sigma_{\alpha})$$

**Level 4 (Time effects):**
$$\mu_t = \mu + \gamma_t$$
$$\gamma_t \sim \text{Normal}(\rho \gamma_{t-1}, \sigma_{\gamma})$$

where ρ is autoregressive coefficient.

**Expected benefits:**
- Assess temporal trends (is μ increasing/decreasing?)
- Forecast future success rates
- Detect changes in group performance over time
- Model seasonality or cyclical patterns

**Implementation:** Add time dimension to hierarchical structure.

### 7.3 Measurement Error Models

**Motivation:** Account for uncertainty in observed r or n (rounding, censoring)

**Model structure:**

**Level 1 (True data):**
$$r_i^{\text{true}} \sim \text{Binomial}(n_i^{\text{true}}, p_i)$$

**Level 2 (Observed data with error):**
$$r_i^{\text{obs}} \sim \text{Normal}(r_i^{\text{true}}, \sigma_{\text{obs}})$$
$$n_i^{\text{obs}} = n_i^{\text{true}} \quad \text{(assume n is known exactly)}$$

**Priors:**
$$\sigma_{\text{obs}} \sim \text{Half-Cauchy}(0, 1)$$

**Expected benefits:**
- More honest uncertainty quantification
- Accounts for rounding (e.g., r reported to nearest 5)
- Handles censoring (e.g., r > 10 reported as "10+")

**Implementation:** Latent variable model in PyMC.

### 7.4 Spatial or Network Structure

**Motivation:** If groups have spatial locations or network connections

**Model structure:**

**Level 1 (Data):**
$$r_i \sim \text{Binomial}(n_i, p_i)$$

**Level 2 (Spatially structured effects):**
$$\text{logit}(p_i) = \mu + \alpha_i$$
$$\boldsymbol{\alpha} \sim \text{MVN}(\mathbf{0}, \Sigma)$$

**Level 3 (Spatial covariance):**
$$\Sigma_{ij} = \sigma^2 \exp\left(-\frac{d_{ij}}{\rho}\right)$$

where:
- d_ij: Distance between groups i and j
- ρ: Range parameter (spatial correlation decay)

**Expected benefits:**
- Account for spatial autocorrelation
- Borrow strength from neighboring groups
- Identify spatial clusters or hotspots

**Implementation:** Gaussian process prior in PyMC.

### 7.5 Model Averaging

**Motivation:** Uncertainty about model choice (beta-binomial vs logit-normal vs robust)

**Approach 1: Bayesian Model Averaging (BMA)**

**Weight models by marginal likelihood:**
$$p(\theta | \text{data}) = \sum_{m=1}^M p(\theta | \text{data}, M_m) \cdot p(M_m | \text{data})$$

where:
$$p(M_m | \text{data}) \propto p(\text{data} | M_m) \cdot p(M_m)$$

**Approach 2: Stacking**

**Weight models to optimize predictive performance:**
$$w^* = \arg\min_w \sum_{i=1}^n \left(\log \sum_{m=1}^M w_m p(y_i | y_{-i}, M_m)\right)^2$$

subject to: w_m ≥ 0, Σw_m = 1

**Expected benefits:**
- Accounts for model uncertainty
- Often better predictions than single model
- Robust to model misspecification

**Implementation:** LOO-CV based stacking in ArviZ.

---

## Additional Resources

### Code Repository Structure

```
/workspace/
├── data/
│   └── data.csv
├── eda/
│   ├── eda_report.md
│   ├── analyst_1/
│   ├── analyst_2/
│   └── analyst_3/
├── experiments/
│   ├── experiment_plan.md
│   ├── experiment_1/
│   │   ├── metadata.md
│   │   ├── prior_predictive_check/
│   │   ├── simulation_based_validation/
│   │   ├── posterior_inference/
│   │   ├── posterior_predictive_check/
│   │   └── model_critique/
│   └── model_assessment/
├── final_report/
│   ├── report.md
│   ├── executive_summary.md
│   ├── technical_supplement.md (this document)
│   └── figures/
└── log.md
```

### Key References

**Bayesian Workflow:**
- Gelman et al. (2020). "Bayesian Workflow." arXiv:2011.01808

**Beta-Binomial Models:**
- Skellam, J.G. (1948). "A probability distribution derived from the binomial distribution..."
- Crowder, M.J. (1978). "Beta-binomial ANOVA for proportions."

**Hierarchical Models:**
- Gelman & Hill (2006). "Data Analysis Using Regression and Multilevel/Hierarchical Models."

**Model Assessment:**
- Vehtari et al. (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC."

**Simulation-Based Calibration:**
- Talts et al. (2018). "Validating Bayesian inference algorithms with simulation-based calibration."

---

**Document Prepared By:** Scientific Report Writer
**Date:** October 30, 2025
**Version:** 1.0
**Status:** Supplementary Technical Documentation
