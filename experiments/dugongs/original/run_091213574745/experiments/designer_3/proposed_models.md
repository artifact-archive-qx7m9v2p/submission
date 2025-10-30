# Robust & Alternative Bayesian Models for Y vs x Relationship
## Designer 3: Robust Modeling & Mixture Approaches

**Date**: 2025-10-28
**Focus Area**: Robust methods, hierarchical structures, mixture models, heavy-tailed likelihoods
**Dataset**: 27 observations with 1 influential outlier at x=31.5

---

## Executive Summary

This design focuses on **robust modeling approaches** that explicitly handle the influential outlier (Cook's D = 1.51) and consider alternative data generation processes. Rather than treating the outlier as measurement error, I propose models that:

1. **Automatically downweight outliers** via heavy-tailed likelihoods
2. **Model heterogeneity explicitly** through mixture distributions
3. **Consider multiple data-generating regimes** simultaneously
4. **Provide uncertainty quantification** about which observations are anomalous

### Philosophy: Falsification First

**I will abandon these models if:**
- The outlier at x=31.5 is actually the most informative point (e.g., reveals a downturn)
- Simple normal likelihood models perform as well (outlier isn't actually problematic)
- The two-regime structure is a sampling artifact, not a real phenomenon
- Models show prior-data conflict or extreme parameter estimates
- Cross-validation shows worse predictive performance than simpler models

---

## Model Priority Ranking

1. **Student-t Logarithmic Model** (HIGHEST PRIORITY)
   - Robust automatic outlier downweighting
   - Simplest robust approach
   - Expected to perform best

2. **Mixture of Two Regimes** (MEDIUM PRIORITY)
   - Tests whether two-regime structure is real heterogeneity
   - Explicitly models the sharp transition at x~7
   - More complex but mechanistically interpretable

3. **Hierarchical Variance Model** (LOWER PRIORITY)
   - Models heteroscedasticity we might be missing
   - Tests whether variance structure is more complex
   - Backup if first two approaches fail

---

## Model 1: Robust Student-t Logarithmic Model

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(nu, mu_i, sigma)
mu_i = beta_0 + beta_1 * log(x_i)

Priors:
beta_0 ~ Normal(2.3, 0.5)          # Intercept: centered at observed mean Y
beta_1 ~ Normal(0.29, 0.15)        # Slope: centered at OLS estimate, wider uncertainty
sigma ~ Exponential(lambda=10)     # Scale: 1/10 = 0.1, allowing RMSE ~0.087
nu ~ Gamma(2, 0.1)                 # Degrees of freedom: mean=20, allows 3-100 range
```

### Theoretical Justification

**Why Student-t likelihood?**
- The t-distribution has heavier tails than Normal, automatically downweighting outliers
- When nu -> infinity, converges to Normal (so we can test if robustness is needed)
- When nu is small (3-7), very robust to outliers
- The posterior on nu tells us HOW heavy-tailed the data actually are

**Why logarithmic mean function?**
- EDA shows R^2 = 0.897 for log model
- Theoretically justified for saturation/dose-response phenomena
- Parsimonious (2 parameters for mean function)
- Smooth, no discontinuities

### Prior Rationale

**beta_0 ~ Normal(2.3, 0.5)**:
- Centered at observed mean Y = 2.33
- SD = 0.5 allows range [1.3, 3.3] at 2 sigma, covering full observed range [1.77, 2.72]
- Weakly informative: data will dominate with n=27

**beta_1 ~ Normal(0.29, 0.15)**:
- Centered at OLS estimate 0.290 from EDA
- SD = 0.15 gives 95% interval [0.0, 0.59], ensuring positive relationship but allowing uncertainty
- Regularizes to prevent overfitting with small sample

**sigma ~ Exponential(10)**:
- Mean = 0.1, matching observed RMSE ~0.087
- Allows sigma in [0.03, 0.25] at 95%, reasonable range
- Proper prior (integrated to 1) prevents improper posteriors

**nu ~ Gamma(2, 0.1)**:
- Mean = 20, SD = 14.1
- Allows nu from ~3 (very robust) to ~100 (almost Normal)
- Mode at nu = 10 (moderately robust)
- This prior is KEY: it lets the data decide how much robustness is needed

### How It Handles Outliers and Small Sample

**Outlier handling**:
- Observation at x=31.5, Y=2.57 (standardized residual = -2.31 in normal model)
- Student-t automatically reduces its influence when nu is estimated to be small
- Unlike manual outlier removal, this is principled and quantifies uncertainty

**Small sample (n=27)**:
- Informative priors regularize estimates
- Student-t is especially helpful with small samples (more uncertainty about tails)
- Posterior predictive intervals will be wider, appropriately reflecting uncertainty

### Expected Strengths

1. **Automatic outlier detection**: Posterior on nu tells us if outliers are present
2. **No manual data dropping**: Principled probabilistic approach
3. **Simple and interpretable**: Only adds 1 parameter (nu) vs Normal model
4. **Robust predictions**: Outliers don't distort predictions at other x values
5. **Computational feasibility**: Well-supported in Stan/PyMC

### Expected Weaknesses

1. **Doesn't capture regime shift**: Smooth log curve may miss sharp transition at x=7
2. **Assumes symmetric tails**: Student-t has symmetric tails; if outliers are one-sided, might overcompensate
3. **May not need robustness**: If nu_posterior >> 30, Normal would have sufficed
4. **Still susceptible to leverage**: Extreme x values (like 31.5) still have high leverage even with robust likelihood

### Falsification Criteria

**I will abandon this model if:**

1. **nu posterior > 30 with high confidence**: Suggests Normal likelihood is adequate, robustness unnecessary
2. **Residual patterns remain**: If residuals vs x show systematic patterns, mean function is wrong
3. **Poor predictive performance**: If LOO-CV is worse than simpler models
4. **Prior-posterior conflict**: If posterior pushes hard against priors, misspecification likely
5. **Observation at x=31.5 is downweighted too much**: Visual inspection might show it's actually consistent with a downturn we should model
6. **Computational issues**: Divergences, poor mixing, R-hat > 1.01 (signals misspecification)

### Stress Test

**Designed to break this model:**
- Fit model, then examine case where x=31.5 is NOT an outlier but the START of a downturn
- Add synthetic data at x=35, 40, 45 with Y declining to 2.3, 2.0, 1.7
- Does the model adapt or fail catastrophically?
- If it fails, need change-point or non-monotonic model

---

## Model 2: Mixture of Two Gaussian Process Regimes

### Mathematical Specification

```
Likelihood (mixture model):
For observation i:
  regime_i ~ Categorical(pi)     # Which regime?

  If regime_i == 1 (Growth):
    Y_i ~ Normal(mu1_i, sigma1)
    mu1_i = alpha1 + beta1 * x_i

  If regime_i == 2 (Plateau):
    Y_i ~ Normal(mu2_i, sigma2)
    mu2_i = alpha2 + beta2 * x_i

Regime Assignment (probabilistic):
log(p(regime_i = 1) / p(regime_i = 2)) = gamma_0 + gamma_1 * x_i
# Logistic regression for regime membership

Priors:
# Growth regime (steep)
alpha1 ~ Normal(1.7, 0.3)          # Intercept for low-x regime
beta1 ~ Normal(0.11, 0.05)         # Steep slope, centered at EDA estimate
sigma1 ~ Exponential(10)           # Variance in growth regime

# Plateau regime (flat)
alpha2 ~ Normal(2.2, 0.3)          # Intercept for high-x regime
beta2 ~ Normal(0.02, 0.02)         # Shallow slope, centered at EDA estimate
sigma2 ~ Exponential(10)           # Variance in plateau regime

# Regime assignment parameters
gamma_0 ~ Normal(0, 2)             # Baseline regime probability
gamma_1 ~ Normal(-1, 0.5)          # Negative: higher x -> more likely plateau
```

### Theoretical Justification

**Why mixture model?**
- EDA shows F-test significance (p < 0.0001) for two regimes with changepoint at x=7
- Traditional changepoint models assume SHARP transition
- Mixture model allows SOFT transition: observations near x=7 could be in either regime
- Tests whether heterogeneity is real or sampling artifact

**Why logistic regime assignment?**
- Probability of being in growth regime decreases smoothly with x
- No arbitrary cutpoint; model learns transition region from data
- Observations can have uncertainty about which regime they belong to
- More realistic than hard threshold at x=7

**Why different variances (sigma1 vs sigma2)?**
- Growth regime might have more variance (dynamic process)
- Plateau regime might have less variance (settled equilibrium)
- Tests whether variance is constant (EDA found no heteroscedasticity, but maybe missed subtle structure)

### Prior Rationale

**Regime-specific slopes**:
- beta1 ~ N(0.11, 0.05): Growth regime, centered at EDA piecewise estimate, 95% in [0.01, 0.21]
- beta2 ~ N(0.02, 0.02): Plateau regime, centered at EDA estimate, 95% in [-0.02, 0.06], allows flat or slight increase
- Prior SD chosen to allow overlap but favor distinct regimes

**Regime assignment**:
- gamma_1 ~ N(-1, 0.5): Negative mean means higher x favors plateau
- At x=7: log-odds = gamma_0 - 7*gamma_1 (should be near 0 for 50/50 split)
- Prior allows smooth transition from x=5 to x=10

### How It Handles Outliers and Small Sample

**Outlier handling**:
- x=31.5, Y=2.57 might be ambiguous: which regime?
- If Y=2.57 is high for plateau regime (expected ~2.3), model can assign it to growth regime (where higher Y is normal)
- Effectively "explains away" outlier as regime mixing rather than noise
- More interpretable than just downweighting

**Small sample (n=27)**:
- Mixture models need larger samples ideally, but we have strong regime signal (F=22.4)
- Informative priors prevent overfitting
- Only 9 observations in growth regime - priors crucial here
- Posterior on gamma_0, gamma_1 will have substantial uncertainty - appropriate

### Expected Strengths

1. **Mechanistically interpretable**: Two regimes have scientific meaning
2. **Tests EDA finding**: Is changepoint real or artifact?
3. **Soft transitions**: More realistic than hard cutoff
4. **Explains heterogeneity**: Outliers might be misclassified regime members
5. **Rich inference**: Posterior on regime membership for each observation

### Expected Weaknesses

1. **Complex**: 7 parameters vs 2-3 for simple models
2. **Identifiability**: With n=27, might struggle to estimate all parameters
3. **Overfitting risk**: Might fit noise in regime assignment
4. **May not be needed**: If simple log model fits well, Occam's razor argues against
5. **Computational**: Mixture models can have multi-modal posteriors, challenging MCMC

### Falsification Criteria

**I will abandon this model if:**

1. **Regime probabilities are diffuse**: If all observations are 40-60% in each regime, model isn't learning
2. **Estimated slopes are similar**: If beta1 ≈ beta2 posteriorly, no real regime difference
3. **Worse predictive performance**: If LOO-CV << simpler models, overfitting
4. **Variances are identical**: If sigma1 ≈ sigma2, no heterogeneity, mixture unjustified
5. **Transition is NOT smooth**: If regime assignment changes abruptly, hard changepoint model better
6. **Identifiability problems**: If posterior shows strong correlations, model is underidentified

### Stress Test

**Designed to break this model:**
- Simulate data from TRUE model: smooth log curve with no regimes
- Add random noise to create apparent "regimes" by chance
- Does model correctly identify NO regime structure? Or does it overfit?
- Good model should collapse to single regime (gamma_1 ≈ 0, beta1 ≈ beta2)

---

## Model 3: Hierarchical Variance Model with Spatial Structure

### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)

Hierarchical Variance (observation-level):
log(sigma_i) = eta + zeta * (x_i - x_mean) + epsilon_i
epsilon_i ~ Normal(0, tau)

Priors:
beta_0 ~ Normal(2.3, 0.5)          # Mean function (same as Model 1)
beta_1 ~ Normal(0.29, 0.15)
eta ~ Normal(-2.3, 0.5)            # log(sigma) baseline: exp(-2.3) ≈ 0.1
zeta ~ Normal(0, 0.2)              # Variance trend with x
tau ~ Exponential(2)               # Between-observation variance in sigma
```

### Theoretical Justification

**Why hierarchical variance?**
- EDA found NO heteroscedasticity, but with n=27, power is low
- Outlier at x=31.5 might indicate variance increases at high x (sparse region)
- Allows each observation to have its own variance, pooled via hierarchy
- Tests whether constant variance assumption is too strong

**Why log(sigma_i) ~ linear in x?**
- Log-link ensures sigma_i > 0
- Linear in x tests monotonic variance trend
- zeta > 0: variance increases with x (funnel shape)
- zeta < 0: variance decreases with x (reverse funnel)
- zeta ≈ 0: constant variance (collapse to Model 1)

**Why observation-level noise (epsilon_i)?**
- Even after accounting for trend, observations might have idiosyncratic variance
- tau quantifies "how much more complex is variance than a simple trend?"
- If tau ≈ 0, variance is deterministic function of x
- If tau is large, variance is highly variable even at similar x

### Prior Rationale

**eta ~ Normal(-2.3, 0.5)**:
- exp(-2.3) = 0.1, matching observed RMSE
- SD = 0.5 gives exp(-2.3 ± 1) = [0.037, 0.27], reasonable range

**zeta ~ Normal(0, 0.2)**:
- Centered at 0 (no trend by default)
- SD = 0.2 allows zeta in [-0.4, 0.4] at 95%
- For x ranging [1, 31.5], this allows sigma to vary by factor of 2-3
- Regularizes to prevent extreme heteroscedasticity

**tau ~ Exponential(2)**:
- Mean = 0.5, SD = 0.5
- If tau << 0.5, variance structure is simple
- If tau > 1, variance is highly variable (suggests model misspecification)

### How It Handles Outliers and Small Sample

**Outlier handling**:
- x=31.5 is in sparse region; model can assign it HIGH variance
- Instead of downweighting (like Student-t), this says "we expect more variability here"
- If posterior sigma_31.5 >> sigma_typical, outlier is "explained" by location-specific uncertainty

**Small sample (n=27)**:
- With 27 observations estimating 27 sigmas, risk of overfitting
- Hierarchy pools information: observations at similar x share variance info
- tau controls pooling strength: larger tau = more pooling
- Priors regularize to prevent extreme estimates

### Expected Strengths

1. **Tests assumption**: Rigorously evaluates constant variance
2. **Flexible**: Can detect subtle heteroscedasticity EDA missed
3. **Explains outliers spatially**: High variance at x=31.5 rather than "bad data"
4. **Uncertainty quantification**: Posterior on sigma_i for each observation
5. **Adaptive**: If no heteroscedasticity, collapses to constant variance

### Expected Weaknesses

1. **Complexity**: 27+ parameters for variance structure
2. **Overfitting risk**: With n=27, might fit noise
3. **Probably unnecessary**: EDA found no heteroscedasticity
4. **Computational cost**: More parameters, slower MCMC
5. **Interpretation challenge**: 27 sigmas hard to summarize/interpret
6. **May not help outlier**: If outlier is error, not high-variance region, this doesn't help

### Falsification Criteria

**I will abandon this model if:**

1. **zeta posterior includes 0 with high probability**: No variance trend, constant variance sufficient
2. **tau is tiny**: If tau ≈ 0, no observation-level variance variation, hierarchy collapses
3. **Worse predictive performance**: LOO-CV worse than constant variance models
4. **All sigma_i are similar**: If posterior sigma_i ≈ 0.1 for all i, hierarchy unnecessary
5. **Computational issues**: If >5% divergences, model too complex for data
6. **Prior-posterior conflict on tau**: If posterior pushes tau to extremes, misspecification

### Stress Test

**Designed to break this model:**
- Fit to data where TRUE variance is constant (simulate from Model 1)
- Does model correctly infer zeta ≈ 0 and tau ≈ 0?
- Or does it hallucinate heteroscedasticity due to small sample variability?
- Good model should collapse to constant variance when appropriate

---

## Model Comparison Strategy

### Phase 1: Individual Model Diagnostics (MANDATORY)

For EACH model, check:

1. **MCMC Convergence**:
   - R-hat < 1.01 for ALL parameters
   - ESS_bulk > 400 per chain
   - ESS_tail > 400 per chain
   - Zero divergent transitions (or <1% if complex model)
   - No max treedepth warnings

2. **Prior-Posterior Overlap**:
   - Plot prior vs posterior for each parameter
   - If overlap is minimal, check for prior-data conflict
   - If posterior is at edge of prior support, prior may be too restrictive

3. **Posterior Predictive Checks**:
   - **PPC 1**: Histogram of Y_rep vs Y_obs
   - **PPC 2**: Scatter of Y_rep vs x, overlay Y_obs
   - **PPC 3**: Distribution of min(Y_rep), max(Y_rep) vs observed
   - **PPC 4**: Check if x=31.5 observation is within posterior predictive distribution

4. **Residual Diagnostics**:
   - Plot posterior mean residuals vs x (should be patternless)
   - Plot posterior mean residuals vs fitted (should be patternless)
   - Check for autocorrelation (DW statistic on posterior mean residuals)

### Phase 2: Cross-Model Comparison

1. **Information Criteria**:
   - Compute LOO-CV (Leave-One-Out Cross-Validation) for each model
   - Compute WAIC (Widely Applicable Information Criterion)
   - Report ELPD (Expected Log Predictive Density) ± SE
   - Check for Pareto-k warnings (k > 0.7 indicates influential observations)

2. **Model Weights**:
   - Use LOO stacking weights to combine models
   - If one model has weight > 0.8, it clearly dominates
   - If weights are diffuse, models are similar, consider averaging

3. **Predictive Performance**:
   - Root Mean Squared Error on held-out predictions
   - Coverage of 95% posterior predictive intervals
   - Sharpness of predictions (narrow intervals preferred if well-calibrated)

### Phase 3: Scientific Interpretation

1. **Parameter Recovery**:
   - Do estimates make scientific sense?
   - Are magnitudes reasonable given domain knowledge?

2. **Uncertainty Quantification**:
   - Are posterior intervals wide or narrow?
   - Do intervals include scientifically plausible values?

3. **Outlier Analysis**:
   - Model 1: What is posterior nu? (If nu > 30, outlier not problematic)
   - Model 2: Which regime is x=31.5 assigned to?
   - Model 3: What is sigma_i for x=31.5 vs typical observations?

### Decision Rules

**If Model 1 (Student-t Log) wins**:
- Check nu posterior: if nu > 30, refit with Normal likelihood (simpler)
- Report as primary model
- Use for predictions and inference

**If Model 2 (Mixture) wins**:
- Examine regime assignments: which observations in which regime?
- Report transition point (where p(regime 1) = 0.5)
- Compare to EDA changepoint at x=7
- Consider implications for data collection: sample more in transition region?

**If Model 3 (Hierarchical Variance) wins**:
- Plot estimated sigma_i vs x
- Interpret: is variance increasing, decreasing, or complex?
- Consider whether heteroscedasticity invalidates previous analyses

**If models are similar (ELPD within 2 SE)**:
- Use Bayesian Model Averaging (BMA) with stacking weights
- Report predictions as weighted average
- Acknowledge model uncertainty in interpretation

### Red Flags (STOP and Reconsider)

1. **All models fail diagnostics**: Mean function may be fundamentally wrong
   - **Action**: Consider non-monotonic models (quadratic, spline)

2. **All models show residual patterns**: Systematic bias remains
   - **Action**: Explore interaction terms, non-parametric regression

3. **Extreme posterior estimates**: Parameters at boundaries of plausible range
   - **Action**: Check for data errors, consider alternative formulations

4. **Poor out-of-sample prediction**: Models overfit training data
   - **Action**: Stronger regularization, simpler models, more data

5. **High Pareto-k for x=31.5 in ALL models**: Observation is fundamentally incompatible
   - **Action**: Sensitivity analysis excluding x=31.5, investigate measurement

---

## Alternative Models (If Primary Models Fail)

### Backup 1: Gaussian Process Regression

If all parametric models show poor fit:

```
Y ~ MVNormal(0, K + sigma^2 * I)
K = RBF kernel with lengthscale l and variance sigma_f^2
```

- Non-parametric, very flexible
- No assumption about functional form
- Good for small samples with complex relationships
- Downside: harder to interpret, computational cost

### Backup 2: Non-Monotonic Polynomial

If we suspect downturn after plateau:

```
Y ~ Normal(mu, sigma)
mu = beta_0 + beta_1*x + beta_2*x^2 + beta_3*x^3
```

- Quadratic or cubic can capture turning points
- EDA dismissed this but maybe outlier is signal of downturn?
- Downside: poor extrapolation, overfitting risk

### Backup 3: Changepoint with Student-t

Combine robustness AND regime structure:

```
Y ~ StudentT(nu, mu, sigma)
mu = {alpha1 + beta1*x,  if x < tau
     {alpha2 + beta2*x,  if x >= tau
```

- Best of Model 1 and Model 2
- More complex (5+ parameters)
- Only if both robustness and regimes are necessary

---

## Success Criteria Summary

### Minimum Requirements (ALL must pass):

1. **R-hat < 1.01** for all parameters
2. **ESS > 400** bulk and tail
3. **Divergences < 1%** of samples
4. **Posterior predictive p-value** in [0.05, 0.95] for key test statistics
5. **Residuals show no pattern** when plotted vs x or fitted values
6. **LOO-CV Pareto-k < 0.7** for >90% of observations

### Model Selection Criteria:

1. **Primary**: LOO-CV ELPD (higher is better)
2. **Secondary**: Posterior predictive coverage (should be ~95% for 95% intervals)
3. **Tertiary**: Simplicity (fewer parameters preferred if performance similar)
4. **Qualitative**: Scientific interpretability and plausibility

### Reporting Requirements:

1. **Posterior summaries**: Mean, SD, 95% CI for all parameters
2. **Convergence diagnostics**: R-hat, ESS, divergences
3. **Model comparison**: LOO-CV table with SE, Pareto-k diagnostics
4. **Posterior predictive plots**: Y_rep vs Y_obs, by x
5. **Residual plots**: Patterns, outliers, heteroscedasticity
6. **Uncertainty quantification**: Credible intervals on predictions at x = 1, 5, 10, 15, 20, 30

---

## Expected Outcomes and Pivots

### Most Likely Scenario: Model 1 Wins

**Expectation**: Student-t logarithmic model will perform best

**Reasoning**:
- Simplest robust model
- Log transformation is well-justified
- EDA already showed R^2 = 0.897 for log model
- Outlier will be automatically handled

**Posterior predictions**:
- nu will be between 5-15 (moderately robust)
- beta_1 will be ~0.25-0.35 (similar to OLS 0.29)
- sigma will be ~0.08-0.10
- Observation at x=31.5 will have larger residual but less influence

**If this happens**: Report Model 1 as primary, mention that Student-t was necessary (cite nu posterior)

### Alternative Scenario 1: Normal Likelihood Sufficient

**If**: Model 1 finds nu > 30 (posterior mean)

**Interpretation**: Outlier is not actually problematic, Normal likelihood adequate

**Action**:
- Refit with Normal likelihood for simplicity
- Report that robustness was tested but unnecessary
- Focus on log transformation as key insight

### Alternative Scenario 2: Mixture Model Wins

**If**: Model 2 has LOO-CV >> Model 1

**Interpretation**: Two-regime structure is REAL heterogeneity, not smooth transition

**Action**:
- Examine regime assignments
- Compare transition zone to EDA changepoint
- Consider mechanistic interpretation (what causes regime shift?)
- Recommend collecting more data near transition

**Red flag**: If transition is very abrupt (all x < 6 in regime 1, all x > 8 in regime 2), use hard changepoint instead

### Alternative Scenario 3: Heteroscedasticity Detected

**If**: Model 3 finds zeta significantly different from 0

**Interpretation**: Variance IS changing with x, EDA missed it

**Action**:
- Plot sigma_i vs x to visualize trend
- Consider weighted regression for simpler analyses
- Report that constant variance assumption violated
- Check if this changes substantive conclusions

**Red flag**: If tau is large, variance is chaotic, suggests fundamental model misspecification

### Worst Case Scenario: All Models Fail

**Signs**:
- Poor convergence (R-hat > 1.01, divergences)
- Residual patterns remain
- LOO-CV all similar and poor
- Pareto-k warnings for many observations

**Interpretation**: Parametric models are wrong, need more flexible approach

**Action**:
1. Fit Gaussian Process (Backup 1)
2. Try non-monotonic polynomial (Backup 2)
3. Visual inspection for patterns we missed
4. Consider data quality issues
5. Consult with domain expert about plausible mechanisms

**Pivot**: Move to non-parametric or semi-parametric models

---

## Computational Implementation Plan

### Software & Tools

**Primary**: Stan (via CmdStanPy or PyStan)
- Best MCMC diagnostics
- Excellent for complex models
- Fast HMC sampler

**Alternative**: PyMC (if Stan struggles)
- More Pythonic interface
- Good for prototyping
- Slightly slower but more flexible

### Sampling Strategy

**For Model 1 (Student-t Log)**:
- 4 chains × 2000 iterations (1000 warmup)
- Should converge quickly (<30 seconds)
- Adaptation tuning: adapt_delta = 0.95

**For Model 2 (Mixture)**:
- 4 chains × 4000 iterations (2000 warmup)
- Expect slower convergence due to mixture
- May need adapt_delta = 0.99
- Check for label switching (regime 1 ↔ regime 2)

**For Model 3 (Hierarchical Variance)**:
- 4 chains × 3000 iterations (1500 warmup)
- More parameters, needs more samples
- Reparameterize if divergences occur: use non-centered parameterization

### Computational Checks

1. **Divergences**: If >1%, increase adapt_delta by 0.01 increments
2. **Max treedepth**: If warnings, increase max_treedepth to 15
3. **Reparameterization**: If tau or nu have poor ESS, try non-centered
4. **Initialization**: Use OLS estimates to initialize chains

---

## Key Questions These Models Answer

### Scientific Questions

1. **Is the outlier at x=31.5 measurement error or real?**
   - Model 1: If heavily downweighted (large residual, nu small), likely error
   - Model 2: If assigned to unexpected regime, possibly real but unusual
   - Model 3: If sigma_31.5 is huge, it's an uncertain region

2. **Is the two-regime structure real or sampling artifact?**
   - Model 2 directly tests this
   - If Model 2 >> Model 1, regimes are real
   - If similar, smooth log curve is adequate

3. **How uncertain are predictions at high x (>20)?**
   - All models will show wider intervals at x > 20 (sparse data)
   - Model 3 might show increasing uncertainty explicitly
   - Critical for extrapolation guidance

4. **What is the asymptotic behavior of Y?**
   - Model 1: Logarithmic growth continues indefinitely (unrealistic for x → infinity)
   - Model 2: Plateau regime suggests Y ≈ 2.5-2.7 asymptote
   - Need to check model predictions at x = 50, 100 (extrapolation)

### Statistical Questions

1. **Is robustness necessary?**
   - nu posterior in Model 1 answers this
   - nu < 10: yes, very necessary
   - nu > 30: no, Normal sufficient

2. **Is variance constant?**
   - Model 3 directly tests
   - If zeta ≈ 0 and tau ≈ 0, yes
   - Otherwise, need to account for heteroscedasticity

3. **How much model uncertainty exists?**
   - If LOO-CV weights are diffuse, substantial uncertainty
   - If one model dominates, structure is clear
   - Report model weights regardless

---

## Final Thoughts: What Would Make Me Pivot Completely?

### Abandon ALL proposed models if:

1. **Observation x=31.5 is the most predictive**: If removing it makes models worse, it's not an outlier but crucial information about downturn

2. **Non-monotonic pattern emerges**: If visual inspection or residuals suggest Y decreases after x=20, need polynomial or spline

3. **Prior-data conflict across all models**: If all posteriors fight priors, priors are wrong OR model class is wrong

4. **Computational failure**: If all models have divergences/convergence issues, parameterization is fundamentally flawed

5. **Cross-validation is poor**: If LOO-CV log score is very negative, models don't predict well even in-sample

6. **Domain expert contradicts**: If expert says "logarithmic/asymptotic doesn't make sense for this phenomenon", start over with mechanistic model

### Success Looks Like:

- At least ONE model passes all diagnostics
- LOO-CV clearly ranks models
- Posterior predictive checks show good fit
- Outlier is handled principled way (not dropped, but understood)
- Uncertainty is quantified appropriately
- Results are scientifically interpretable

### Failure Looks Like:

- All models fail diagnostics
- LOO-CV shows all models are equally bad
- Residual patterns persist
- Posteriors are bizarre or extreme
- Results don't make scientific sense

**If we fail**: Don't force it. Admit parametric models are insufficient. Move to non-parametric methods (GP, splines) or collect more data.

---

## Implementation Priority

### Week 1: Model 1 (Student-t Log)
- Highest priority, most likely to succeed
- Quick to implement and run
- Establishes baseline for robustness

### Week 2: Model Comparison
- Fit all three models
- Run LOO-CV comparisons
- Generate posterior predictive checks

### Week 3: Sensitivity & Reporting
- Sensitivity analysis excluding x=31.5
- Finalize model choice
- Generate predictions and visualizations
- Write up results

**Total estimated time**: 2-3 weeks for thorough analysis

---

## Files to Generate

1. **`model1_student_t_log.stan`** - Stan code for Model 1
2. **`model2_mixture_regimes.stan`** - Stan code for Model 2
3. **`model3_hierarchical_variance.stan`** - Stan code for Model 3
4. **`fit_models.py`** - Python script to fit all models
5. **`diagnostics.py`** - Convergence and posterior checks
6. **`model_comparison.py`** - LOO-CV and WAIC comparison
7. **`visualization.py`** - Posterior predictive plots
8. **`results_summary.md`** - Final report with model choice and interpretation

---

**END OF DESIGN DOCUMENT**

---

## Appendix: Prior Sensitivity Analysis Plan

For the chosen model, we MUST check prior sensitivity:

1. **Refit with wider priors** (2× SD on all priors)
   - If results change substantially, priors too influential
   - If results similar, inference is robust

2. **Refit with narrower priors** (0.5× SD on all priors)
   - Tests if data alone can overcome strong priors
   - If posteriors stay similar, data is informative

3. **Refit with different prior families**
   - e.g., Uniform instead of Normal for slopes
   - Tests structural prior assumptions

**Criterion**: Posterior means should not change by >20% under reasonable prior perturbations. If they do, sample size is too small for reliable inference, or prior choice is crucial (report sensitivity prominently).

---

**Document prepared by Designer 3: Robust & Alternative Modeling Specialist**
**Ready for implementation and iteration based on empirical results**
