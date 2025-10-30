# Model Proposals - Designer 1
## Classical Meta-Analysis and Measurement Error Models

**Date**: 2025-10-28
**Designer**: Model Designer #1
**Focus Area**: Classical meta-analysis and measurement error models
**Dataset**: 8 observations with known measurement uncertainties

---

## Executive Summary

Based on the EDA findings showing strong homogeneity (I² = 0%, Q p = 0.696), I propose three Bayesian model classes that span different complexity levels and robustness assumptions. All models explicitly account for known measurement uncertainties and use proper Bayesian inference.

**Key EDA Insights Driving Design**:
- No detected heterogeneity (τ² = 0)
- No publication bias or outliers
- Clean, normally-distributed data
- Pooled estimate: θ = 7.686 ± 4.072

**Design Philosophy**: Start with the simplest plausible model (fixed-effect), then add complexity only to test robustness and validate the homogeneity finding.

---

## Model 1: Fixed-Effect Normal Model (BASELINE)

### Model Class
Classical fixed-effect meta-analysis with Gaussian measurement error model.

### Theoretical Justification

This is the canonical model for meta-analysis when all studies estimate the same underlying parameter. The EDA provides overwhelming evidence for this scenario:

1. **Homogeneity**: Cochran's Q test (p = 0.696) shows no evidence against the null hypothesis that all studies share a common effect
2. **I² = 0%**: All observed variation is explained by within-study sampling error
3. **Theoretical parsimony**: Occam's razor favors the simplest model consistent with data
4. **Statistical efficiency**: When homogeneity holds, fixed-effect models are maximally efficient

The model treats the observed outcomes as noisy measurements of a single true parameter θ, with heterogeneous but known measurement precision across studies.

### Mathematical Specification

**Likelihood**:
```
y_i | θ, σ_i ~ Normal(θ, σ_i²)   for i = 1, ..., 8
```

Where:
- y_i: Observed outcome from study i
- θ: True common effect (parameter of interest)
- σ_i: Known standard error for study i [15, 10, 16, 11, 9, 11, 10, 18]

**Prior**:
```
θ ~ Normal(0, 20²)
```

**Key Assumptions**:
1. All studies estimate the same parameter (no between-study heterogeneity)
2. Measurement errors are normally distributed
3. Measurement uncertainties σ_i are known and correctly specified
4. Studies are independent conditional on θ
5. No systematic biases (e.g., publication bias)

**Likelihood Structure**:
The log-likelihood is:
```
log p(y | θ) = -1/2 * Σ[(y_i - θ)² / σ_i² + log(2π σ_i²)]
```

This gives precision-weighted estimation where studies with smaller σ_i contribute more information.

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as fixed_effect_model:
    # Prior on common effect
    theta = pm.Normal('theta', mu=0, sigma=20)

    # Likelihood with known heterogeneous variances
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=4,
                      return_inferencedata=True,
                      target_accept=0.95)
```

**Computational Notes**:
- This is a conjugate model: posterior is analytically Normal
- MCMC will converge immediately (single scalar parameter)
- Expected ESS > 3000, R-hat = 1.000
- Can validate MCMC against closed-form solution:
  ```
  Precision_post = 1/20² + Σ(1/σ_i²) = 0.0025 + 0.06044 = 0.06294
  σ_post = 3.988
  μ_post = (0 * 0.0025 + Σ(y_i/σ_i²)) / 0.06294 ≈ 7.686
  ```

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior predictive check fails badly**:
   - If y_rep (replicated data) shows systematically different patterns than observed y
   - Specifically: If test statistic p-values are extreme (< 0.01 or > 0.99)
   - Red flag: If observed y falls outside 95% predictive envelope

2. **Evidence of heterogeneity emerges**:
   - If standardized residuals show patterns (e.g., several > 2 SD)
   - If leave-one-out cross-validation reveals one or more studies as severe outliers (pareto-k > 0.7)
   - If posterior predictive p-values for individual studies are extreme

3. **Prior-data conflict detected**:
   - If posterior is multimodal (suggests model misspecification)
   - If MCMC diagnostics show pathological behavior despite simple structure

4. **Sensitivity analysis shows instability**:
   - If minor prior changes cause >50% shift in posterior mean
   - Suggests data is too weak to overcome prior, indicating model may be inappropriate

5. **Domain expert review**:
   - If subject-matter experts identify scientific implausibility in the assumptions
   - Example: If θ must be positive but posterior includes substantial negative mass

**Stress Test**:
- Simulate data under heterogeneous effects (τ = 5) and verify this model produces poor LOO-CV scores
- Confirms the model is actually sensitive to violations of homogeneity

### Expected Posterior Characteristics

**If model is correct, expect**:

1. **Posterior distribution**:
   - θ_post ~ Normal(μ ≈ 7.7, σ ≈ 4.0)
   - 95% CrI: approximately [-0.3, 15.7]
   - Posterior very close to weighted least squares estimate

2. **MCMC diagnostics**:
   - R-hat < 1.01 (should be 1.00 given simplicity)
   - ESS > 3000 (near-independent samples)
   - No divergences
   - Trace plots show white noise

3. **Posterior predictive checks**:
   - Replicated data y_rep has mean ≈ 8.75, SD ≈ 10.4
   - Test statistics (mean, median, max, min) fall within predictive distribution
   - Bayesian p-values ≈ 0.5 for all test statistics

4. **LOO-CV**:
   - All Pareto-k < 0.5 (no influential observations)
   - LOO predictive density reasonable for all observations
   - ELPD (expected log pointwise predictive density) ≈ -30

5. **Parameter recovery**:
   - Posterior mean very close to weighted mean 7.686
   - Posterior SD very close to analytical SE 4.072
   - 95% CrI coverage should be nominal in simulation studies

**Validation**:
- Compare MCMC posterior to analytical solution (should match within Monte Carlo error)
- If they differ substantially, suggests computational problems

---

## Model 2: Bayesian Random-Effects Model (HETEROGENEITY TEST)

### Model Class
Hierarchical Bayesian model allowing for between-study heterogeneity (DerSimonian-Laird generalization).

### Theoretical Justification

While the EDA shows no heterogeneity, with only J=8 studies, power to detect moderate τ is limited. A Bayesian hierarchical model provides:

1. **Formal heterogeneity quantification**: Posterior distribution for τ quantifies uncertainty about between-study variance
2. **Robustness**: If heterogeneity exists but is small, this model will capture it
3. **Model comparison**: Comparing to Model 1 via Bayes factors or LOO tests the homogeneity hypothesis
4. **Conservative inference**: If τ > 0, credible intervals appropriately wider

**Why this might be wrong**:
- The data show I² = 0%, suggesting no heterogeneity
- Random effects models have poor power with J=8
- Adding τ parameter may cause identification issues
- Prior on τ becomes very influential with weak data

### Mathematical Specification

**Likelihood (Hierarchical)**:
```
y_i | θ_i, σ_i ~ Normal(θ_i, σ_i²)
θ_i | μ, τ ~ Normal(μ, τ²)
```

Where:
- y_i: Observed outcome from study i
- θ_i: Study-specific true effect
- μ: Population mean effect (parameter of interest)
- τ: Between-study standard deviation (heterogeneity parameter)
- σ_i: Known within-study standard error

**Priors**:
```
μ ~ Normal(0, 20²)
τ ~ Half-Cauchy(0, 5)
```

**Alternative prior on τ** (for sensitivity):
```
τ ~ Half-Normal(0, 5²)
```

**Key Assumptions**:
1. Study-specific effects θ_i are exchangeable draws from N(μ, τ²)
2. Within-study errors remain normal with known variance
3. Between-study heterogeneity (if any) is normally distributed
4. Studies are independent
5. J=8 is sufficient to estimate τ (questionable!)

**Prior Rationale**:
- Half-Cauchy(0, 5) is weakly informative: allows substantial heterogeneity but regularizes extreme values
- This is the prior recommended by Gelman (2006) for hierarchical variance parameters
- Half-Normal alternative is less heavy-tailed, pulls τ toward zero more strongly

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = len(y)

with pm.Model() as random_effects_model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfCauchy('tau', beta=5)

    # Study-specific effects (non-centered parameterization)
    # More efficient: θ_i = μ + τ * z_i, where z_i ~ N(0, 1)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=4,
                      return_inferencedata=True,
                      target_accept=0.95)
```

**Computational Notes**:
- Use **non-centered parameterization** to avoid funnel geometry (Neal's funnel)
- This is critical when τ is near zero (as expected here)
- Centered parameterization (θ_i ~ N(μ, τ²)) will have divergences when τ → 0
- Expect slower mixing than Model 1 due to additional parameters
- Target ESS > 400 for all parameters

**Potential Issues**:
- If τ posterior concentrates near zero, model reduces to Model 1
- Posterior for τ may be sensitive to prior choice with J=8
- θ_i estimates will shrink toward μ (partial pooling)

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior for τ is entirely prior-driven**:
   - If changing Half-Cauchy(5) to Half-Cauchy(2) or Half-Normal(5) dramatically changes τ posterior
   - Indicates J=8 provides insufficient information to learn about heterogeneity
   - Red flag: τ posterior looks like the prior

2. **Computational pathologies**:
   - Many divergences (> 1%) even with non-centered parameterization
   - R-hat > 1.05 for any parameter
   - Funnel-shaped posterior in (τ, μ) space
   - These suggest fundamental identification problems

3. **LOO-CV favors simpler model by >4 units**:
   - If ΔELPD(Model 2 - Model 1) < -4 and se(ΔELPD) < 2
   - Suggests random effects structure is overfitting
   - Model 1 (fixed effect) is clearly preferred

4. **Posterior predictive checks identical to Model 1**:
   - If y_rep from both models are indistinguishable
   - Suggests added complexity provides no predictive benefit

5. **τ posterior includes zero in 99% credible interval**:
   - If P(τ < 1) ≈ 1, there's no meaningful heterogeneity
   - In this case, Model 1 is more appropriate by parsimony

**Stress Test**:
- Simulate homogeneous data (τ = 0) and verify that τ posterior concentrates near zero
- Confirms the model can "learn" homogeneity

### Expected Posterior Characteristics

**If τ ≈ 0 (homogeneity, as EDA suggests)**:

1. **Posterior for τ**:
   - Mode near 0, median < 2
   - 95% CrI: [0, τ_upper] where τ_upper ≈ 5-10
   - Right-skewed distribution
   - Most posterior mass at small values

2. **Posterior for μ**:
   - μ_post ~ Normal(7.7, 4.0)
   - Nearly identical to θ_post from Model 1
   - 95% CrI: [-0.3, 15.7]

3. **Study-specific effects θ_i**:
   - Shrink toward μ (partial pooling)
   - Shrinkage factor ≈ 1 when τ → 0 (no shrinkage, equals y_i ± σ_i)
   - All θ_i posteriors overlap substantially

4. **Model comparison**:
   - LOO-CV: ELPD(Model 2) ≈ ELPD(Model 1) ± 2 (equivalent performance)
   - Bayes factor BF(M1/M2) ≈ 1-3 (weak preference for simpler model)
   - WAIC: similar or slightly worse for Model 2 (penalty for extra parameters)

5. **MCMC diagnostics**:
   - R-hat < 1.01 for all parameters
   - ESS(μ) > 1000, ESS(τ) > 400 (τ may mix slower)
   - Possible trace plot autocorrelation for τ
   - No divergences with non-centered parameterization

**If τ > 0 (unexpected heterogeneity)**:
- τ posterior would shift right, with mode > 3
- θ_i would show clear separation beyond what σ_i predicts
- LOO-CV would favor Model 2
- This would contradict EDA, triggering investigation of data quality

---

## Model 3: Robust Fixed-Effect Model (OUTLIER PROTECTION)

### Model Class
Fixed-effect meta-analysis with Student-t likelihood for robustness to outliers and heavy tails.

### Theoretical Justification

Although the EDA shows no outliers, with J=8 observations, even one anomalous study could distort inference. The robust model provides:

1. **Heavy-tailed errors**: Student-t allows occasional large deviations without compromising overall fit
2. **Automatic downweighting**: Outliers receive less weight than in Gaussian model
3. **Data-driven robustness**: The degrees-of-freedom parameter ν is estimated, allowing data to choose between normal (ν > 30) and heavy-tailed (ν < 5)
4. **Defensive modeling**: Protects against undetected outliers or model misspecification

**Why this might be wrong**:
- EDA shows no outliers (all observations within ±2 SD)
- Normal Q-Q plots look good
- Adding ν parameter increases complexity without apparent need
- May provide minimal improvement over Model 1

### Mathematical Specification

**Likelihood**:
```
y_i | θ, σ_i, ν ~ Student_t(ν, θ, σ_i²)
```

Where:
- y_i, θ, σ_i as before
- ν: Degrees of freedom (controls tail heaviness)
- Student-t with ν > 30 ≈ Normal
- Student-t with ν < 5 has very heavy tails

**Priors**:
```
θ ~ Normal(0, 20²)
ν ~ Gamma(2, 0.1)
```

**Alternative prior on ν** (for sensitivity):
```
ν ~ Exponential(1/29)  # Prior mean = 29, weakly favors normality
```

**Key Assumptions**:
1. All studies estimate the same θ (like Model 1)
2. Measurement errors have heavier tails than normal (if ν < 30)
3. All studies share the same ν (homogeneous tail behavior)
4. σ_i still scales the uncertainty correctly

**Prior Rationale for ν**:
- Gamma(2, 0.1): mean = 20, allows ν ∈ [5, 50] with high probability
- Weak preference for moderate ν
- Regularizes away from ν < 3 (which would indicate severe outliers)
- If ν_post > 30, data suggest normal is adequate

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as robust_fixed_effect_model:
    # Priors
    theta = pm.Normal('theta', mu=0, sigma=20)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # Likelihood with Student-t errors
    y_obs = pm.StudentT('y_obs', nu=nu, mu=theta, sigma=sigma, observed=y)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=4,
                      return_inferencedata=True,
                      target_accept=0.95)
```

**Computational Notes**:
- Student-t likelihood is non-conjugate: requires MCMC
- May have slightly heavier tails than normal posterior
- ν can be slow to mix if poorly identified
- Consider reparameterization for ν if mixing is poor
- Expected ESS(θ) > 1000, ESS(ν) > 400

**Potential Issues**:
- ν may be poorly identified with J=8
- Prior on ν becomes influential
- If ν_post > 30, model is essentially Model 1 with extra complexity

### Falsification Criteria

**I will abandon this model if**:

1. **ν posterior is entirely prior-driven**:
   - If ν posterior closely resembles Gamma(2, 0.1) prior
   - Indicates data provide no information about tail behavior
   - Red flag: Posterior nearly unchanged when prior varies

2. **ν posterior strongly favors normality**:
   - If P(ν > 30 | y) > 0.9
   - Suggests Student-t provides no benefit over Gaussian
   - Model 1 preferred by parsimony

3. **LOO-CV worse than Model 1**:
   - If ΔELPD(Model 3 - Model 1) < -2
   - Extra parameter hurts predictive performance
   - Occam's razor favors simpler model

4. **Posterior for θ virtually identical to Model 1**:
   - If |E[θ | y]_Model3 - E[θ | y]_Model1| < 0.1
   - If 95% CrI overlap > 99%
   - Suggests robustness provides no practical difference

5. **Computational instability**:
   - Divergences, poor R-hat, or extreme ESS issues
   - May indicate model is over-parameterized for data

**Stress Test**:
- Add one severe outlier (e.g., y_9 = 50, σ_9 = 10) and verify:
  - Model 3 downweights it (ν decreases)
  - Model 1 is distorted by it
  - This confirms robustness mechanism works

### Expected Posterior Characteristics

**If data are Gaussian (as EDA suggests)**:

1. **Posterior for ν**:
   - Median ν ≈ 15-30 (moderately high)
   - 95% CrI: [5, 60] (wide, poorly identified)
   - Right-skewed distribution
   - Overlaps region where Student-t ≈ Normal

2. **Posterior for θ**:
   - θ_post ~ approximately Normal(7.7, 4.1)
   - Mean similar to Model 1 (within ±0.5)
   - SD slightly larger (≈5-10% increase) due to heavier tails
   - 95% CrI: approximately [-0.5, 16.0] (slightly wider)

3. **Posterior predictive checks**:
   - y_rep very similar to Model 1
   - Possibly slightly heavier tails
   - Test statistics indistinguishable from Model 1

4. **Model comparison**:
   - LOO-CV: ELPD(Model 3) ≈ ELPD(Model 1) ± 1 (equivalent)
   - WAIC: slightly worse due to extra parameter penalty
   - Bayes factor weakly favors Model 1 by parsimony

5. **MCMC diagnostics**:
   - R-hat < 1.01 for θ, possibly 1.01-1.02 for ν
   - ESS(θ) > 1000, ESS(ν) > 400 (ν may be slower)
   - No divergences
   - Trace plot for ν may show more autocorrelation than θ

**If outliers present** (not expected):
- ν posterior would shift left (mode < 10)
- θ posterior would differ substantially from Model 1
- LOO-CV would favor Model 3
- Individual observation weights would show downweighting

**Interpretation if ν > 30**:
- Data confirm Gaussian assumption is adequate
- Model 1 is preferred for simplicity
- Robustness analysis complete: normal model validated

---

## Priority Ranking

### Rank 1: Model 1 (Fixed-Effect Normal) - PRIMARY MODEL

**Justification**:
- EDA provides overwhelming evidence for homogeneity
- Simplest model consistent with data (Occam's razor)
- Maximally efficient when assumptions hold (and they appear to)
- Computationally trivial, analytically tractable
- Serves as baseline for all comparisons

**Expected outcome**: This model will fit well, pass all diagnostics, and provide the tightest credible intervals.

**Action**: Fit first, use as reference standard.

### Rank 2: Model 2 (Random Effects) - HYPOTHESIS TEST

**Justification**:
- Formally tests the homogeneity assumption
- Provides conservative inference if heterogeneity exists
- LOO/WAIC comparison quantifies evidence for/against τ > 0
- Model averaging possible if τ posterior is uncertain

**Expected outcome**: τ posterior will concentrate near zero, validating Model 1. If not, this is a critical finding requiring investigation.

**Action**: Fit second to test whether added complexity is justified. If τ ≈ 0, confirm Model 1. If τ > 0, investigate why EDA missed heterogeneity.

### Rank 3: Model 3 (Robust Fixed-Effect) - SENSITIVITY ANALYSIS

**Justification**:
- Defensive modeling for high-stakes inference
- Tests sensitivity to normality assumption
- Provides robustness bounds on θ estimates
- If ν > 30, validates normal assumption

**Expected outcome**: ν posterior will favor moderate-to-high values, indicating normality is adequate. θ posterior will closely match Model 1.

**Action**: Fit third as sensitivity check. If results are nearly identical to Model 1, conclude normal model is robust. If different, investigate why EDA Q-Q plots looked normal.

---

## Model Comparison Strategy

### Workflow

1. **Fit all three models** using PyMC with identical MCMC settings
2. **Check MCMC diagnostics** for all models (R-hat, ESS, divergences)
3. **Posterior predictive checks** for each model (visual and quantitative)
4. **LOO cross-validation** to compare predictive performance
5. **Sensitivity analysis** on priors (especially τ and ν priors)
6. **Parameter comparison** across models (θ estimates)

### Decision Rules

**If Models 1, 2, 3 all agree** (expected scenario):
- All give θ ≈ 7.7 ± 4.0
- Model 2: τ ≈ 0
- Model 3: ν > 20
- LOO-CV equivalent across models
- **Conclusion**: Fixed-effect normal model (Model 1) is validated. Report Model 1 results with statement that conclusions are robust to model choice.

**If Model 2 shows τ > 0** (unexpected):
- Re-examine EDA for missed heterogeneity
- Check if τ posterior is prior-driven
- Investigate study characteristics for heterogeneity sources
- **Conclusion**: Random effects model (Model 2) may be more appropriate. Revise modeling strategy.

**If Model 3 shows ν < 10** (unexpected):
- Investigate which observations are being downweighted
- Check for outliers missed by EDA
- Re-examine data quality
- **Conclusion**: Robust model (Model 3) reveals issues. Further data investigation needed.

**If models strongly disagree** (θ estimates differ by > 2 SD):
- Red flag for model misspecification
- Check for computational errors
- Re-examine data generating assumptions
- May need entirely different model class

### Model Averaging

If Model 2 and Model 1 have similar LOO-CV (within 2 ELPD), consider Bayesian model averaging:

```
θ_avg = w_1 * θ_Model1 + w_2 * θ_Model2
w_i ∝ exp(ELPD_i)
```

This accounts for model uncertainty in the final inference.

---

## Implementation Checklist

For each model:

**Pre-Sampling**:
- [ ] Specify model in PyMC
- [ ] Set informative variable names
- [ ] Check prior predictive distribution (simulate y_prior ~ p(y | θ_prior))
- [ ] Verify prior predictive is plausible (rules out extreme values)

**Sampling**:
- [ ] Run 4 chains, 2000 iterations, 1000 warmup
- [ ] Set target_accept = 0.95 for complex models
- [ ] Monitor convergence during sampling

**Post-Sampling Diagnostics**:
- [ ] Check R-hat < 1.01 for all parameters
- [ ] Check ESS > 400 (preferably > 1000) for all parameters
- [ ] Check for divergences (should be 0 or < 0.1%)
- [ ] Examine trace plots (should look like white noise)
- [ ] Check energy diagnostics (E-BFMI > 0.2)

**Posterior Checks**:
- [ ] Plot posterior distributions
- [ ] Compute posterior means, medians, 95% CrI
- [ ] Generate posterior predictive samples (y_rep)
- [ ] Compare y_rep to observed y (graphically and quantitatively)
- [ ] Compute Bayesian p-values for test statistics

**Model Comparison**:
- [ ] Compute LOO-CV for each model
- [ ] Check Pareto-k diagnostics (all < 0.7)
- [ ] Compare ELPD differences with standard errors
- [ ] Compute WAIC as alternative IC

**Sensitivity Analysis**:
- [ ] Vary priors (e.g., θ ~ N(0, 10²) vs N(0, 50²))
- [ ] Check robustness of conclusions
- [ ] Document any prior sensitivity

**Reporting**:
- [ ] Create forest plot with posterior distributions
- [ ] Table of parameter estimates (mean, SD, 95% CrI)
- [ ] Table of model comparison (LOO, WAIC)
- [ ] Statement of model assumptions and limitations
- [ ] Posterior predictive check figures

---

## Red Flags and Escape Routes

### Warning Signs That Current Model Class Is Wrong

**Red Flag #1: Prior-Posterior Conflict**
- Posterior has much tighter interval than data alone suggest
- Indicates prior is dominating or model is misspecified
- **Escape route**: Try weakly informative or flat prior, or switch to different model class

**Red Flag #2: Poor Posterior Predictive Checks**
- Test statistics systematically differ from y_rep
- Observed y falls outside 95% predictive envelope
- **Escape route**: Consider non-normal errors, mixture models, or contamination models

**Red Flag #3: LOO-CV Failures**
- Multiple observations with Pareto-k > 0.7
- Indicates model is highly sensitive to individual observations
- **Escape route**: Robust models (Model 3) or mixture models

**Red Flag #4: Computational Pathologies**
- Persistent divergences despite tuning
- Very low E-BFMI (< 0.2)
- Often indicates fundamental model misspecification
- **Escape route**: Try reparameterization, or question whether Bayesian model is appropriate

**Red Flag #5: Inconsistent with EDA**
- If Model 2 gives τ >> 0 despite I² = 0% in EDA
- Suggests either EDA method issue or model issue
- **Escape route**: Investigate data more carefully, may need measurement error model with uncertain σ_i

### Alternative Model Classes to Consider

**If homogeneity assumption fails**:
- Meta-regression with study-level covariates
- Mixture models (two subpopulations)
- Time-varying effects models

**If normality assumption fails**:
- Skew-normal or skew-t distributions
- Non-parametric models (Dirichlet process)
- Transformation of y (log, Box-Cox)

**If σ_i are uncertain**:
- Hierarchical model for σ_i
- Measurement error model with σ_i ~ prior
- This is beyond classical meta-analysis, would require different designer

**If independence assumption fails**:
- Spatial/temporal correlation models
- Network meta-analysis
- Multivariate meta-analysis

---

## Stopping Rules

### When to Stop and Declare Success

**Stop if**:
1. All three models pass diagnostics
2. Posterior predictive checks look good for all models
3. Models agree on θ ≈ 7.7 ± 4
4. LOO-CV shows no clear winner (within 2 ELPD)
5. Sensitivity analysis confirms robustness

**Conclusion**: Fixed-effect normal model is adequate, report Model 1 with robustness statements.

### When to Stop and Pivot Strategy

**Stop and pivot if**:
1. All models fail posterior predictive checks similarly
2. Computational issues persist across models
3. Estimates are implausibly sensitive to prior choice
4. Domain expert identifies fundamental issue with meta-analysis framing

**Action**: Consult with other designers, consider entirely different model classes.

### When to Investigate Further

**Investigate if**:
1. Model 2 shows τ > 0 (contradicts EDA)
2. Model 3 shows ν < 10 (suggests outliers)
3. Models disagree substantially (> 1 SD difference in θ)
4. LOO-CV strongly favors one model but diagnostics are unclear

**Action**: Deep dive into data, rerun EDA, check for errors, consider additional models.

---

## Expected Timeline

**Phase 1: Model Implementation** (1-2 hours)
- Code all three models in PyMC
- Check syntax, run prior predictive checks
- Test on synthetic data

**Phase 2: MCMC Sampling** (30 minutes)
- Fit all models (each takes ~5 min with J=8)
- Monitor convergence
- Troubleshoot any issues

**Phase 3: Diagnostics** (1 hour)
- Check all MCMC diagnostics
- Run posterior predictive checks
- Compute LOO-CV

**Phase 4: Comparison and Reporting** (1-2 hours)
- Compare models formally
- Create visualizations
- Write summary of findings

**Total**: Approximately 3-5 hours for complete analysis.

---

## Scientific Plausibility Constraints

**Domain-Agnostic Checks**:
- Is θ posterior in a scientifically plausible range?
- Are credible intervals too wide to be useful?
- Does the uncertainty quantification match our intuition about J=8 studies?

**If Domain is Known** (e.g., medical treatment effect):
- Must θ > 0? (Use truncated prior if yes)
- Are there known bounds (e.g., proportions, counts)?
- Is the scale of θ consistent with outcome units?
- Do results align with previous meta-analyses?

**Red Flag**: If domain expert says "θ should be positive" but posterior puts 40% mass on negative values, either:
- Prior was not sufficiently informative
- Model is missing constraints
- Data genuinely contradict expert belief (interesting!)

---

## Summary

I propose three classical Bayesian meta-analysis models, ranked by priority:

1. **Fixed-Effect Normal** (Primary): Simplest model consistent with EDA, likely best
2. **Random Effects** (Hypothesis test): Tests homogeneity, expected to collapse to Model 1
3. **Robust Fixed-Effect** (Sensitivity): Tests normality, expected to validate Model 1

All models are fully Bayesian, use known measurement uncertainties, and can be implemented in PyMC. Falsification criteria are clearly stated for each model, and I expect that Model 1 will be validated by Models 2 and 3.

**Critical mindset**: I am prepared to abandon the fixed-effect framework if:
- Model 2 reveals genuine heterogeneity (τ > 0)
- Model 3 reveals outliers or non-normality (ν < 10)
- Posterior predictive checks fail for all models
- LOO-CV suggests poor out-of-sample prediction

**Next steps**: Implement models, run MCMC, compare results, and iterate if necessary. If all models agree, report Model 1 with confidence. If they disagree, investigate why and consider alternative model classes.

---

**End of Model Proposals**
