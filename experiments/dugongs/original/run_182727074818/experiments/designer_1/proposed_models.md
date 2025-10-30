# Parametric Bayesian Regression Models: Designer #1

**Date:** 2025-10-27
**Designer Focus:** Parametric regression (linear, polynomial, logarithmic, power law)
**Data Context:** 27 observations, Y ∈ [1.77, 2.72], x ∈ [1.0, 31.5]

---

## Executive Summary

I propose **3 competing parametric model classes** that represent fundamentally different hypotheses about the data-generating process:

1. **Logarithmic Model** (Primary) - Diminishing returns with unbounded growth
2. **Rational Function Model** (Secondary) - Bounded asymptotic behavior
3. **Piecewise Linear Model** (Tertiary) - Discrete regime change hypothesis

**Critical insight:** While EDA suggests logarithmic is best, I'm deliberately challenging this by exploring models that could falsify it. The EDA's 66% RSS improvement from a change point is a major red flag that smooth parametric forms might be misspecified.

**My confidence levels:**
- 60% one of these models is adequate
- 30% we'll need a non-parametric approach (e.g., GP)
- 10% there's something fundamentally wrong with the data/assumptions

---

## Core Philosophy: Falsification-First Design

### What Would Make Me Abandon All These Models?

1. **Prior-data conflict:** If all priors are overwhelmed by data (posterior completely determined by likelihood), suggests we're missing key structure
2. **Systematic residual patterns:** If all models show clear residual structure, parametric form is wrong
3. **Extreme parameter instability:** Small data perturbations causing huge posterior shifts
4. **Poor replicate prediction:** If models can't predict replicates at same x value
5. **Computational pathologies:** Divergent transitions, non-convergence despite reparameterization

### Red Flags for Model Class Switch

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| Residuals correlate with x bins | Missing non-linear structure | Try GP/spline |
| Different fits on data subsets | Heterogeneity not captured | Try hierarchical model |
| Posterior predictive checks fail systematically | Wrong likelihood family | Try Student-t, mixture |
| Change point posterior is bimodal | Data supports multiple regimes | Try mixture model |
| All models equally bad (WAIC) | Wrong model class entirely | Pivot to non-parametric |

---

## Model Class 1: Logarithmic Model (Primary)

### Hypothesis
The relationship follows **logarithmic growth** where increases in x have diminishing marginal effects on Y. This assumes:
- No true asymptote (Y continues growing, but slowly)
- Smooth, continuous relationship
- The "change point" at x=7 is an artifact of logarithmic curvature

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = alpha + beta * log(x_i + c)
```

**Parameters:** alpha (intercept), beta (log-slope), c (translation constant), sigma (error SD)

**Priors:**
```stan
// Intercept: Y values centered around 2.3
alpha ~ normal(2.3, 0.5);

// Slope: Empirically beta ≈ (2.7-1.8)/log(31.5+1) ≈ 0.27
// But allow for substantial uncertainty
beta ~ normal(0.3, 0.3);  // Enforces positivity weakly

// Translation constant: Usually 0 or 1, but let's learn it
c ~ gamma(2, 2);  // Mean=1, SD=0.71, concentrates on [0.1, 3]

// Error SD: Should be < SD(Y)=0.27
sigma ~ exponential(1/0.15);  // Mean=0.15, allows up to ~0.5
```

**Why These Priors?**
- **alpha:** Centered on data mean, but allows ±1 SD coverage of full Y range
- **beta:** Positive slope expected, prior mean matches OLS estimate
- **c:** Gamma prior prevents negative values (log undefined), concentrates on sensible range
- **sigma:** Exponential (not half-Cauchy) to avoid heavy tails - this is a specific choice to see if data fights it

### Why This Model Might Be Right

1. **Empirical support:** R²=0.888 in EDA (best simple form)
2. **Parsimony:** Only 4 parameters for 27 observations (conservative)
3. **Interpretability:** "Doubling x increases Y by beta*log(2) ≈ 0.21 units"
4. **Extrapolation:** Doesn't explode like polynomials
5. **Theoretical plausibility:** Common in learning curves, dose-response, economic returns

### Why This Model WILL Fail (Falsification Criteria)

**I will abandon this model if:**

1. **Residuals show clear structure:** If plotting residuals vs x shows U-shape or systematic pattern
   - *Evidence threshold:* Runs test p < 0.05, or visual inspection shows obvious trend

2. **Change point is real:** If segmented model improves WAIC by >6 points
   - *Implication:* The x=7 breakpoint is not just logarithmic curvature

3. **Replicate prediction fails:** If model can't predict held-out replicates within 80% credible intervals
   - *Evidence:* Coverage < 60% on replicated x values

4. **Posterior strongly prefers c→0:** If posterior mean c < 0.1
   - *Implication:* log(x) is better than log(x+c), but then x=1 is problematic

5. **Extreme extrapolation:** If predicting at x=50 gives Y>3.5
   - *Implication:* Model doesn't capture saturation, asymptotic form needed

### Stress Tests

1. **Outlier sensitivity:** Refit dropping x=31.5 observation
   - If beta changes by >30%, model is unstable

2. **Prior sensitivity:** Double prior SDs
   - If posteriors change substantially, data is weak for parameter identification

3. **Transformation sensitivity:** Compare log(x+1) vs log(x+0.1) vs log(x+c) with fixed c
   - If results drastically different, model is not robust

### Computational Considerations

**Expected performance:**
- Fast sampling (linear in parameters)
- Potential issue: c and alpha/beta correlation if c→0
- Solution: Use non-centered parameterization or fix c=1 if learning c is problematic

**Diagnostics to monitor:**
- Rhat < 1.01 for all parameters
- ESS > 400 for each parameter
- No divergent transitions
- Trace plots show good mixing (no sticky regions)

**Implementation note:**
If c causes sampling issues, I will **immediately** try:
1. Fix c=1 (standard choice)
2. Reparameterize: mu_i = alpha + beta * (log(x_i + c) - log(c))
3. If still problematic, this suggests log transform is fundamentally wrong

---

## Model Class 2: Rational Function Model (Secondary)

### Hypothesis
The relationship exhibits **bounded saturation** following a rational function (Michaelis-Menten type). This assumes:
- True asymptote exists (Y → Y_max as x → ∞)
- Smooth transition to plateau
- The "change point" is really the saturation inflection point

### Mathematical Specification

**Likelihood:**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = Y_min + (Y_max - Y_min) * x_i^h / (K^h + x_i^h)
```

**Parameters:** Y_min (baseline), Y_max (asymptote), K (half-saturation), h (Hill coefficient), sigma

**Priors:**
```stan
// Baseline: Observed minimum is 1.77
Y_min ~ normal(1.8, 0.2);  // Tight prior, we see the low end

// Asymptote: Must be > max observed (2.72)
Y_max ~ normal(2.9, 0.3);  // Mean above data, but uncertain

// Half-saturation: Visual inspection suggests K ~ 5-10
K ~ gamma(3, 0.3);  // Mean=10, SD=5.77, allows [2, 25]

// Hill coefficient: Usually 1-2 in biology
h ~ gamma(4, 2);  // Mean=2, SD=1, concentrates on [0.5, 4]

// Error SD
sigma ~ exponential(1/0.15);
```

**Why These Priors?**
- **Y_min:** Tight because we have data at low x
- **Y_max:** Weakly informative but enforces Y_max > observed max
- **K:** Concentrates around visual "elbow" location
- **h:** Hill coefficient >1 allows sharper transitions, =1 is standard Michaelis-Menten

### Why This Model Might Be Right

1. **Mechanistic plausibility:** Common in saturation processes (enzyme kinetics, carrying capacity)
2. **Bounded predictions:** Y_max parameter prevents unrealistic extrapolation
3. **Flexibility:** Hill coefficient h allows sharp (h>1) or gradual (h<1) saturation
4. **EDA hint:** Asymptotic model had R²=0.834, only slightly worse than log
5. **Visual evidence:** Data appears to be leveling off at high x

### Why This Model WILL Fail (Falsification Criteria)

**I will abandon this model if:**

1. **Y_max is not identified:** If posterior for Y_max has huge variance or is truncated by prior
   - *Evidence:* 95% CI for Y_max includes >3.5 (unreasonably high)
   - *Implication:* Data doesn't extend far enough to see asymptote

2. **K and h are strongly correlated:** If posterior correlation |ρ(K,h)| > 0.9
   - *Implication:* Parameters are fundamentally non-identifiable

3. **Computational failure:** Divergent transitions persist despite reparameterization
   - *Evidence:* >1% divergences after trying centered/non-centered forms
   - *Implication:* Model geometry is pathological

4. **Worse fit than logarithmic:** If WAIC is >6 points worse than log model
   - *Implication:* Added complexity of saturation doesn't help

5. **h posterior concentrates at boundary:** If posterior mean h < 0.6 or h > 3.5
   - *Implication:* Hill form is inappropriate, try simpler (h=1) or different functional form

### Stress Tests

1. **Prior on Y_max:** Try Y_max ~ normal(2.8, 0.5) vs normal(3.5, 0.5)
   - If posteriors drastically different, Y_max is not data-identified

2. **Fix h=1:** Compare to model with h fixed at 1 (standard Michaelis-Menten)
   - If WAIC similar, h is unnecessary complexity

3. **Identifiability check:** Simulate from prior predictive
   - If prior predictive is diffuse/unrealistic, priors are too weak

### Computational Considerations

**Expected performance:**
- **Slow sampling:** Non-linear optimization in NUTS sampler
- **High parameter correlation:** K, h, Y_max will be correlated
- **Potential divergences:** Likelihood surface may have "funnel" geometry

**Reparameterization strategies if needed:**

1. **Non-centered for Y_max:**
   ```stan
   Y_max_raw ~ normal(0, 1);
   Y_max = 2.9 + 0.3 * Y_max_raw;
   ```

2. **Reparameterize K:**
   Use log(K) as parameter to avoid boundary issues:
   ```stan
   log_K ~ normal(log(10), 0.5);
   K = exp(log_K);
   ```

3. **If correlation is extreme:**
   Fix h=1 and estimate Y_min, Y_max, K only (reduce to 4 parameters)

**Decision rule:**
If >5% divergent transitions after reparameterization attempts, **this model is rejected**. Divergences with HMC almost always indicate fundamental misspecification, not just tuning issues.

---

## Model Class 3: Piecewise Linear Model (Tertiary)

### Hypothesis
There is a **discrete regime change** around x=7 representing fundamentally different processes. This assumes:
- Two distinct linear relationships
- Sharp (but possibly smooth) transition
- The 66% RSS improvement from EDA change point is real, not artifact

### Mathematical Specification

**Variant A: Discontinuous (Simpler)**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = alpha_1 + beta_1 * x_i                    if x_i <= tau
mu_i = alpha_2 + beta_2 * x_i                    if x_i > tau
```

**Variant B: Continuous (Preferred)**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = alpha + beta_1 * x_i                      if x_i <= tau
mu_i = alpha + beta_1 * tau + beta_2 * (x_i - tau)  if x_i > tau
```

**I will implement Variant B** (continuous) unless data strongly suggests discontinuity.

**Parameters:** alpha (intercept), beta_1 (slope segment 1), beta_2 (slope segment 2), tau (change point), sigma

**Priors:**
```stan
// Intercept: Y at low x
alpha ~ normal(1.9, 0.3);

// Slope 1: Steeper initial relationship
beta_1 ~ normal(0.15, 0.1);  // Positive, moderate slope

// Slope 2: Flatter post-transition
beta_2 ~ normal(0.03, 0.05);  // Positive but smaller

// Change point: EDA suggests x≈7
tau ~ normal(7, 2);  // Allow [3, 11] roughly (95% CI)

// Error SD
sigma ~ exponential(1/0.15);
```

**Why These Priors?**
- **alpha:** Anchors Y at low x where we have data
- **beta_1 > beta_2:** Prior expectation is steeper-then-flatter (EDA finding)
- **tau:** Centered on EDA's suggested change point, but allows substantial variation
- **Continuity constraint:** Prevents discontinuous jumps which are rarely plausible physically

### Why This Model Might Be Right

1. **Strong empirical support:** EDA found 66% RSS improvement with breakpoint
2. **Interpretability:** Two distinct regimes may reflect real process change
3. **Domain plausibility:** Many systems have phase transitions, threshold effects
4. **Simplicity in each regime:** Linear segments are maximally simple
5. **Diagnostic power:** If this fits well, smooth models are definitely wrong

### Why This Model WILL Fail (Falsification Criteria)

**I will abandon this model if:**

1. **Change point is not identified:** If posterior for tau has SD > 3 or is uniform across prior range
   - *Evidence:* 95% CI for tau spans >50% of data range
   - *Implication:* No clear breakpoint exists

2. **Slopes are not different:** If posterior distributions for beta_1 and beta_2 overlap >80%
   - *Evidence:* P(beta_1 > beta_2) < 0.75
   - *Implication:* Single slope is adequate

3. **Residuals still show pattern:** If residuals within each segment show non-linearity
   - *Evidence:* Runs test p < 0.05 within each segment
   - *Implication:* Even within segments, relationship is not linear

4. **Worse than logarithmic:** If WAIC is >4 points worse than log model
   - *Implication:* Added complexity of change point doesn't justify itself

5. **Change point at data boundary:** If posterior mean tau < 2 or tau > 25
   - *Implication:* Model is trying to avoid having two regimes, wants single slope

### Stress Tests

1. **Prior on tau:** Try tau ~ Uniform(3, 15) to remove informative prior
   - If posterior drastically changes, tau is not data-identified

2. **Allow beta_2 < 0:** Remove positivity constraint on beta_2
   - Tests if second segment could actually be flat/decreasing

3. **Discontinuous variant:** Fit Variant A (allow jump at tau)
   - If jump is substantial, continuity assumption is wrong

4. **Multiple change points:** Try model with 2 change points
   - If much better fit, single change point is oversimplified

### Computational Considerations

**Expected performance:**
- **Moderate sampling difficulty:** Change point tau is discrete-like parameter
- **Potential multimodality:** Likelihood may have multiple local modes for different tau values
- **Label switching:** If beta_1 ≈ beta_2, tau becomes non-identified

**Diagnostics to monitor:**
- **Trace plot for tau:** Should not show "jumping" between distinct values (multimodality)
- **Joint posterior (tau, beta_1, beta_2):** Check for ridges or non-convexities
- **ESS for tau:** Often lowest ESS in model, may need more iterations

**Implementation strategies:**

1. **Initialization:** Start chains at different tau values (e.g., 5, 7, 9) to check for multimodality

2. **If tau is poorly identified:**
   - Plot likelihood profile across tau values
   - Consider fixing tau at MLE value and doing sensitivity analysis

3. **If label switching occurs:**
   - Add constraint: beta_1 > beta_2 + epsilon
   - Or use ordered vector: [beta_2, beta_1] ~ ordered[2](...)

**Decision rule:**
If tau posterior is flat or bimodal, **this model is rejected** - suggests no real change point or multiple possible change points.

---

## Model Comparison Strategy

### Sequential Testing Protocol

**Phase 1: Fit all three models independently**
- Use same priors specification as above
- Run 4 chains, 2000 iterations (1000 warmup) each
- Check convergence (Rhat, ESS, trace plots)
- Record WAIC, LOO-CV for each model

**Phase 2: Initial model comparison**
```
Decision rules:
- If WAIC difference < 2: Models are equivalent
- If WAIC difference 2-6: Weak preference
- If WAIC difference > 6: Strong preference
```

**Phase 3: Posterior predictive checks** (for all models)
1. Replicate data distribution (compare Y_rep to Y)
2. Residual normality (QQ plots)
3. Residual independence (runs test, ACF)
4. Coverage of replicates at same x (hold-out test)

**Phase 4: Falsification checks**
- Apply specific falsification criteria for each model (see above)
- Any model that fails its criteria is **immediately dropped**

**Phase 5: Sensitivity analysis** (for surviving models)
- Prior sensitivity (widen/narrow priors by factor of 2)
- Outlier sensitivity (drop x=31.5)
- Data subset sensitivity (random 80% subsamples)

### Comparison Metrics

| Metric | Interpretation | Decision Weight |
|--------|----------------|-----------------|
| WAIC | Out-of-sample prediction | Primary (40%) |
| LOO-CV | Cross-validation | Primary (40%) |
| Posterior predictive p-value | Goodness of fit | Secondary (10%) |
| Residual patterns | Mis-specification | Secondary (10%) |

**Why this weighting?**
- WAIC/LOO are most objective (less dependent on arbitrary choices)
- Posterior predictive checks can be misleading with small samples
- Residual patterns are important but subjective

### What If All Models Fail?

**Escape routes if all parametric models are inadequate:**

1. **Try non-parametric approach:**
   - Gaussian Process regression
   - Bayesian splines (B-splines with random effects)
   - Bayesian Additive Regression Trees (BART)

2. **Try different likelihood:**
   - Student-t (if outliers are issue)
   - Beta regression (if Y is actually bounded in [0,1] after transformation)
   - Mixture of normals (if subpopulations exist)

3. **Try different error structure:**
   - Heteroscedastic errors: sigma_i = sigma * x_i^gamma
   - Autocorrelated errors: AR(1) structure if x is ordered by time

4. **Reconsider data:**
   - Are there covariates missing?
   - Is x the right predictor? (maybe should be x^2, 1/x, etc.)
   - Are there measurement errors in x? (errors-in-variables model)

### Decision Points for Major Pivots

**Checkpoint 1: After Phase 2 (Initial Comparison)**
- If all WAIC within 2 points → **Stop, do model averaging**
- If clear winner (ΔWAIC > 6) → **Focus on that model, do sensitivity**
- If all models terrible (absolute fit metrics bad) → **Pivot to non-parametric**

**Checkpoint 2: After Phase 4 (Falsification)**
- If all models fail falsification → **Pivot to different model class entirely**
- If 2+ models survive → **Continue with both, consider averaging**
- If 1 model survives → **That's our model, but remain skeptical**

**Checkpoint 3: After Phase 5 (Sensitivity)**
- If winner is sensitive to priors → **Data is weak, report high uncertainty**
- If winner is sensitive to outliers → **Consider robust likelihood (Student-t)**
- If winner is sensitive to subsets → **Heterogeneity exists, consider mixture/hierarchical**

---

## Alternative Models (If Initial Set Fails)

### Backup Plan 1: Power Law Model

**If logarithmic fails but relationship is still smooth:**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = alpha + beta * x_i^gamma

Priors:
alpha ~ normal(1.8, 0.3)
beta ~ normal(0.5, 0.5)
gamma ~ normal(0.5, 0.3)  // <1 for diminishing returns
sigma ~ exponential(1/0.15)
```

**Advantage:** More flexible than log, still smooth
**Disadvantage:** Computational difficulty, parameter correlation

### Backup Plan 2: Quadratic with Plateau

**If data shows saturation but rational function fails:**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = min(Y_max, alpha + beta_1*x_i + beta_2*x_i^2)

Priors:
Y_max ~ normal(2.9, 0.3)
alpha ~ normal(1.8, 0.3)
beta_1 ~ normal(0.2, 0.2)
beta_2 ~ normal(-0.005, 0.01)  // Negative for concave
sigma ~ exponential(1/0.15)
```

**Advantage:** Quadratic captures curvature, ceiling prevents extrapolation disaster
**Disadvantage:** Discontinuous derivative at plateau, not mechanistic

### Backup Plan 3: Smooth Transition Regression

**If piecewise model works but discontinuity is implausible:**
```
Y_i ~ Normal(mu_i, sigma^2)
mu_i = alpha + beta_1*x_i + (beta_2 - beta_1)*x_i * Phi((x_i - tau)/gamma)

Where Phi is standard normal CDF, gamma controls smoothness
```

**Advantage:** Smooth transition between regimes, interpretable
**Disadvantage:** Additional parameter (gamma), computational complexity

---

## Prior Justification: Philosophy and Specifics

### General Prior Philosophy

**I am using "weakly informative" priors, but specifically:**

1. **Regularizing toward plausibility:** Priors gently constrain parameters to scientifically reasonable ranges
2. **Calibrated to data scale:** Prior SDs are proportional to observed data variation
3. **Allowing surprise:** Prior mass extends to values that would "surprise" us but aren't impossible
4. **Computational pragmatism:** Priors help sampler avoid pathological regions

**Key principle:** If posterior is just the prior, **the model has failed** (data didn't teach us anything).

### Why Not Flat Priors?

Flat priors are **bad practice** for multiple reasons:
1. Not invariant under transformation (flat in α is not flat in exp(α))
2. Often improper (don't integrate to 1)
3. Allow pathological parameter values that waste sampler time
4. Provide no regularization with small sample (n=27)

### Why Not More Informative Priors?

I could use tighter priors based on:
- OLS estimates from EDA
- Domain knowledge (if available)
- Historical data (if available)

**But I'm deliberately avoiding this** for two reasons:
1. Want to see if data alone can identify parameters (falsification test)
2. Tight priors mask model inadequacy (good fit doesn't mean good model)

### Prior Sensitivity: Built-In Test

My **sensitivity analysis will double/halve all prior SDs**:
- If results barely change → Good (data dominates)
- If results change moderately → Acceptable (prior-data compromise)
- If results change drastically → **Red flag** (data is weak, model may be wrong)

---

## Expected Outcomes and Confidence Intervals

### My Predictions

**Most Likely Scenario (50% probability):**
- Logarithmic model wins by ΔWAIC = 3-5 points
- Rational function second (fits well but overparameterized)
- Piecewise model third (change point not strongly identified)
- Winner explains ~85-90% of variance
- Residuals are mostly random with minor patterns

**Optimistic Scenario (25% probability):**
- One model clearly superior (ΔWAIC > 8)
- Falsification tests all pass
- Posterior predictive checks show excellent fit
- Prior sensitivity shows data dominance
- We found the right model!

**Pessimistic Scenario (25% probability):**
- All models have similar WAIC (within 3 points)
- All show some residual patterns
- Posteriors are sensitive to priors
- No clear winner → Need different approach

### What Success Looks Like

**Good outcome:**
- Clear model winner with ΔWAIC > 4
- Winner passes all falsification tests
- Residuals appear random
- Posteriors stable under sensitivity tests
- **But:** I remain skeptical and check carefully

**Great outcome:**
- Model predicts held-out replicates accurately
- Extrapolation to x=50 is reasonable (Y ∈ [2.7, 3.2])
- Posterior interpretations make scientific sense
- Model suggests experiments to refine understanding

**Bad outcome:**
- No model is adequate (all fail falsification)
- Computational issues persist despite reparameterization
- Residuals show clear patterns
- **Action:** Pivot to non-parametric approach immediately

---

## Implementation Roadmap

### Week 1: Model Implementation
- Day 1-2: Code all three models in Stan
- Day 3: Debug and test on simulated data
- Day 4-5: Fit to real data, check convergence

### Week 2: Diagnostics and Comparison
- Day 1-2: Posterior predictive checks, residual analysis
- Day 3: WAIC/LOO comparison
- Day 4-5: Falsification tests for each model

### Week 3: Sensitivity and Refinement
- Day 1-2: Prior sensitivity analysis
- Day 3: Outlier sensitivity (drop x=31.5)
- Day 4-5: Data subset sensitivity

### Week 4: Reporting and Decisions
- Day 1-2: Synthesize results, make model recommendation
- Day 3-4: If needed, implement backup models
- Day 5: Final report with decision

**Stopping Rule:** If after Week 2 all models clearly fail, **stop and pivot to non-parametric** (don't waste time on sensitivity analysis of bad models).

---

## Computational Specifications

### Software Stack
- **Stan** (via PyStan or CmdStanPy)
- Python 3.9+ for pre/post-processing
- ArviZ for diagnostics and visualization

### Sampling Settings
```
chains: 4
iterations: 2000 (1000 warmup + 1000 sampling)
adapt_delta: 0.95 (increase if divergences occur)
max_treedepth: 12 (increase if saturating)
```

### Convergence Criteria
- Rhat < 1.01 for all parameters
- ESS_bulk > 400 per chain (1600 total)
- ESS_tail > 400 per chain
- No divergent transitions (<1% acceptable if localized)
- Max treedepth warnings <5% of iterations

### Hardware Requirements
- Expected runtime: 1-5 minutes per model on modern CPU
- RAM: <2GB (small dataset)
- No GPU needed

---

## Communication Plan

### Outputs

1. **Model comparison table** with WAIC, LOO, diagnostics
2. **Posterior summaries** for all parameters (mean, median, 95% CI)
3. **Trace plots** for convergence assessment
4. **Posterior predictive plots** comparing Y_rep to Y
5. **Residual diagnostic plots** (4-panel)
6. **Sensitivity analysis results** (tables and plots)

### Key Visualizations

1. **Fit comparison plot:** All three models overlaid on data
2. **Residual comparison:** Side-by-side residual plots
3. **Posterior predictive distributions:** Showing uncertainty
4. **Parameter posterior plots:** Marginal distributions for key parameters
5. **WAIC comparison:** Bar chart with standard errors

### Report Structure

1. Executive Summary (1 page)
2. Model Specifications (3 pages, one per model)
3. Results (5 pages with tables/figures)
4. Falsification Analysis (2 pages)
5. Sensitivity Analysis (2 pages)
6. Discussion and Recommendations (2 pages)
7. Appendix: Stan code, full diagnostics (10 pages)

---

## Reflection: What Could I Be Missing?

### Blind Spots in My Approach

1. **Parametric assumption:** All my models assume simple functional forms
   - Maybe relationship is genuinely irregular (needs GP/spline)

2. **Independence assumption:** I'm treating observations as independent
   - Maybe there's temporal/spatial structure if x is ordered

3. **Single predictor:** Maybe Y depends on other variables not in data
   - If so, all models will show unexplained variance

4. **Measurement error:** I'm assuming x is measured without error
   - If x has error, all models will be biased

5. **Homoscedasticity:** All models assume constant variance
   - EDA said this is OK, but replicate variance varies

### How I'll Detect These Issues

1. **Parametric inadequacy:** Residual patterns, poor predictive checks
2. **Dependence:** ACF of residuals, runs test
3. **Missing variables:** Consistent underprediction in regions
4. **Measurement error:** Replicates at same x show more variance than model predicts
5. **Heteroscedasticity:** Scale-location plot shows trend (though EDA said no)

### Contingency Plans

If I discover these issues during modeling:
- **Non-parametric needed:** Implement GP as Model 4
- **Dependence:** Add AR(1) errors if x is ordered
- **Missing variables:** Report limitation, suggest collecting covariates
- **Measurement error:** Try errors-in-variables model (complex, likely not feasible with n=27)
- **Heteroscedasticity:** Try sigma_i = sigma * f(x_i) with simple f

---

## Final Thoughts: Success Criteria

### Scientific Success
- Understand the Y-x relationship deeply
- Identify limitations clearly
- Suggest follow-up experiments
- **Not necessarily finding "the right model"**

### Methodological Success
- Demonstrate rigorous Bayesian workflow
- Show proper diagnostics and sensitivity analysis
- Falsify models that don't work
- Be honest about uncertainty

### Practical Success
- Provide reliable predictions within data range [1, 31.5]
- Quantify uncertainty appropriately
- Give decision-makers usable information
- **Not overstate confidence in extrapolation**

**Core principle:** I'd rather report "We tried 3 models and none are fully adequate, here's why" than "Model X is great!" when it's actually not. **Truth over completion.**

---

## Appendix A: Stan Code Skeletons

### Model 1: Logarithmic
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> c;
  real<lower=0> sigma;
}
model {
  vector[N] mu;

  // Priors
  alpha ~ normal(2.3, 0.5);
  beta ~ normal(0.3, 0.3);
  c ~ gamma(2, 2);
  sigma ~ exponential(1/0.15);

  // Likelihood
  mu = alpha + beta * log(x + c);
  Y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha + beta * log(x[n] + c);
    Y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma);
  }
}
```

### Model 2: Rational Function
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}
parameters {
  real<lower=0> Y_min;
  real<lower=Y_min> Y_max;
  real<lower=0> K;
  real<lower=0> h;
  real<lower=0> sigma;
}
model {
  vector[N] mu;

  // Priors
  Y_min ~ normal(1.8, 0.2);
  Y_max ~ normal(2.9, 0.3);
  K ~ gamma(3, 0.3);
  h ~ gamma(4, 2);
  sigma ~ exponential(1/0.15);

  // Likelihood
  for (n in 1:N) {
    real x_h = pow(x[n], h);
    real K_h = pow(K, h);
    mu[n] = Y_min + (Y_max - Y_min) * x_h / (K_h + x_h);
  }
  Y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real x_h = pow(x[n], h);
    real K_h = pow(K, h);
    real mu_n = Y_min + (Y_max - Y_min) * x_h / (K_h + x_h);
    Y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma);
  }
}
```

### Model 3: Piecewise Linear (Continuous)
```stan
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}
parameters {
  real alpha;
  real beta_1;
  real beta_2;
  real<lower=min(x), upper=max(x)> tau;
  real<lower=0> sigma;
}
model {
  vector[N] mu;

  // Priors
  alpha ~ normal(1.9, 0.3);
  beta_1 ~ normal(0.15, 0.1);
  beta_2 ~ normal(0.03, 0.05);
  tau ~ normal(7, 2);
  sigma ~ exponential(1/0.15);

  // Likelihood
  for (n in 1:N) {
    if (x[n] <= tau) {
      mu[n] = alpha + beta_1 * x[n];
    } else {
      mu[n] = alpha + beta_1 * tau + beta_2 * (x[n] - tau);
    }
  }
  Y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n;
    if (x[n] <= tau) {
      mu_n = alpha + beta_1 * x[n];
    } else {
      mu_n = alpha + beta_1 * tau + beta_2 * (x[n] - tau);
    }
    Y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(Y[n] | mu_n, sigma);
  }
}
```

---

## Appendix B: Falsification Checklist

### Before Declaring Success

For each model, I will verify:

- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 for all parameters
- [ ] Trace plots show good mixing (no trends, no stickiness)
- [ ] No divergent transitions (or <1% and localized)
- [ ] Pairs plot shows no extreme correlations (|ρ| < 0.95)
- [ ] Posterior predictive checks: Y_rep resembles Y distribution
- [ ] Residual QQ plot: Approximately straight line
- [ ] Residuals vs x: No obvious pattern (random scatter)
- [ ] Residuals vs fitted: No obvious pattern
- [ ] Runs test on residuals: p > 0.05
- [ ] Replicate coverage: >60% of replicates in 80% CI
- [ ] Prior sensitivity: Posteriors stable when prior SD doubled
- [ ] Outlier sensitivity: Results stable when x=31.5 dropped
- [ ] Extrapolation check: Prediction at x=50 is reasonable (Y < 3.5)
- [ ] Parameter interpretations are scientifically plausible
- [ ] Model-specific falsification criteria passed (see each model section)

**If any critical check fails (especially model-specific falsification criteria), the model is rejected regardless of WAIC.**

---

**End of Proposal**

**Files this document references:**
- Input: `/workspace/data/data.csv`
- Input: `/workspace/eda/eda_report.md`
- Output: `/workspace/experiments/designer_1/proposed_models.md` (this file)
- Future outputs: `/workspace/experiments/designer_1/results/` (to be created during implementation)
