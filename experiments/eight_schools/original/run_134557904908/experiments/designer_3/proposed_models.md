# Model Proposals - Designer 3: Hierarchical & Structured Effects

**Designer**: Model Designer #3 (Hierarchical/Structured Specialist)
**Date**: 2025-10-28
**Dataset**: Meta-analysis with J=8 observations
**EDA Key Findings**: Homogeneous effects (I²=0%, Q p=0.696), θ_pooled=7.686±4.072

---

## Design Philosophy

Despite strong evidence for homogeneity in the EDA, I propose hierarchical models for three critical reasons:

1. **Small Sample Paradox**: With J=8, we have low power to detect moderate heterogeneity. The absence of detected heterogeneity doesn't prove homogeneity exists.

2. **Scientific Humility**: True homogeneity across independent studies is theoretically suspect. Even "identical" interventions vary in implementation, populations, and contexts.

3. **Partial Pooling Benefits**: Hierarchical models adaptively shrink toward complete pooling when data suggest it, providing automatic regularization without forcing a binary choice.

4. **Robustness to Misspecification**: If true heterogeneity exists but is masked by large σ_i, hierarchical models will discover it. If homogeneity is real, they'll converge to fixed effects naturally.

**Key Question**: How much structure can we justify with J=8 observations and large uncertainties?

---

## Model 1: Adaptive Hierarchical Normal Model (AHNM)

### Model Class
Bayesian hierarchical meta-analysis with adaptive partial pooling

### Theoretical Justification

**Why this model makes sense**:
- EDA suggests homogeneity, but this is observationally equivalent to "heterogeneity smaller than measurement noise"
- Hierarchical structure allows data to choose the degree of pooling
- With large σ_i (mean=12.5), we can't distinguish τ=0 from τ=5
- Prior on τ must be carefully calibrated to avoid over-regularization
- Natural extension of fixed effects that degrades gracefully if heterogeneity exists

**Scientific rationale**:
- Studies differ in populations, implementation, contexts → expect some variability
- Large within-study noise (σ_i) may mask between-study variability (τ)
- Partial pooling provides automatic bias-variance tradeoff

### Mathematical Specification

**Likelihood**:
```
y_i | θ_i, σ_i ~ Normal(θ_i, σ_i²)    for i = 1,...,8
θ_i | μ, τ ~ Normal(μ, τ²)              (study-specific effects)
```

**Priors**:
```
μ ~ Normal(0, 20²)                     (population mean)
τ ~ Half-Normal(0, 5²)                 (between-study SD)
```

**Alternative prior on τ** (more conservative):
```
τ ~ Half-Cauchy(0, 2.5)                (weakly informative, robust)
```

**Key Parameters**:
- μ: Population mean effect (what all studies are estimating on average)
- τ: Between-study standard deviation (heterogeneity scale)
- θ_i: Study-specific true effects (latent parameters)

**Implied Marginal**:
```
y_i | μ, τ ~ Normal(μ, σ_i² + τ²)
Total variance = within-study + between-study
```

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = len(y_obs)

with pm.Model() as hierarchical_model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)

    # Study-specific effects (non-centered parameterization)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y = pm.Normal('y', mu=theta, sigma=sigma, observed=y_obs)

    # Derived quantities
    I_squared = pm.Deterministic('I_squared',
                                 100 * tau**2 / (tau**2 + pm.math.mean(sigma**2)))

    # Sample
    trace = pm.sample(2000, tune=2000, chains=4,
                     target_accept=0.95, random_seed=42)
```

**Technical details**:
- **Non-centered parameterization**: Critical for avoiding divergences when τ→0
- **Target_accept=0.95**: Needed because posterior may be funnel-shaped
- **I² as derived quantity**: Directly comparable to frequentist I²
- **Convergence diagnostics**: Check R_hat < 1.01, ESS > 400 for all parameters

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior for τ concentrates at exactly zero with no uncertainty**
   - Evidence: Posterior median τ < 0.01 AND 95% CrI = [0, 0.5]
   - Interpretation: Data strongly prefer fixed effects, hierarchical structure is superfluous
   - Action: Switch to simpler fixed effect model

2. **Severe funnel geometry and divergent transitions**
   - Evidence: >5% divergences even with target_accept=0.99
   - Interpretation: Model is poorly identified, posterior geometry is pathological
   - Action: Either (a) use fixed effects, or (b) stronger regularizing prior on τ

3. **Posterior predictive checks fail systematically**
   - Evidence: Observed data fall outside 95% posterior predictive intervals for >3 observations
   - Interpretation: Normal likelihood inadequate, need robust alternative
   - Action: Switch to Student-t likelihood (Model 3)

4. **Prior-posterior conflict for τ**
   - Evidence: Posterior for τ pushes hard against prior upper tail (e.g., posterior mode at τ=10+)
   - Interpretation: Prior is too restrictive, data want more heterogeneity than model allows
   - Action: Relax prior or investigate systematic differences between studies

5. **Study-specific effects θ_i show no shrinkage**
   - Evidence: Posterior θ_i essentially equal raw study estimates (no pooling)
   - Interpretation: Studies are too different to pool; heterogeneity too large
   - Action: Investigate meta-regression with covariates or structured heterogeneity

### Expected Posterior Characteristics

**If model is appropriate** (based on EDA findings):

1. **τ posterior**:
   - Median: 0-3 (small but non-zero)
   - 95% CrI: [0, 8] (wide uncertainty due to J=8)
   - Mass concentrated near zero but allowing for moderate heterogeneity

2. **μ posterior**:
   - Mean: 7-8 (similar to fixed effect estimate)
   - SD: 4-5 (slightly wider than fixed effect due to τ uncertainty)
   - 95% CrI: [-1, 16]

3. **θ_i posteriors**:
   - Shrunken toward μ, especially for low-precision studies
   - High-precision studies (small σ_i) shrink less
   - Posterior SD(θ_i) < raw SE for most studies

4. **I² posterior**:
   - Median: 0-20% (low heterogeneity)
   - 95% CrI: [0, 60%] (wide uncertainty)

5. **Shrinkage factors**:
   - Studies with large σ_i shrink more toward μ
   - Studies with small σ_i retain closer to raw estimates
   - Average shrinkage: 30-50%

6. **Convergence**:
   - R_hat < 1.01 for all parameters
   - ESS > 400 (may be lower for τ, acceptable if >200)
   - Trace plots show stationarity

7. **Model comparison**:
   - LOO-CV or WAIC similar to fixed effect model (±1-2)
   - No strong preference, suggesting data are consistent with both

**Validation checks**:
- Posterior predictive p-values for mean, SD near 0.5
- No systematic deviations in residuals
- Leave-one-out predictions reasonably accurate

---

## Model 2: Measurement Error Model with Uncertainty in σ_i

### Model Class
Hierarchical measurement error model accounting for uncertainty in reported standard errors

### Theoretical Justification

**Critical assumption often ignored**: Standard errors σ_i are themselves estimates, not known constants!

**Why this matters**:
- Each σ_i is computed from within-study data → has estimation uncertainty
- With moderate within-study sample sizes, σ_i could vary by ±20%
- Treating σ_i as fixed understates total uncertainty
- Can lead to spurious homogeneity if we're overconfident in precision estimates

**When this model is appropriate**:
- Studies report SE or confidence intervals (not raw variance)
- Within-study sample sizes are moderate (n < 100)
- We have access to within-study sample sizes or degrees of freedom
- Meta-analyst wants conservative inference accounting for all uncertainties

**Limitation**: We don't have within-study sample sizes in this dataset, so we'll use a pragmatic approximation.

### Mathematical Specification

**Generative model**:
```
True precision: λ_i ~ Gamma(a_i, b_i)           (latent precision)
Observed SE: σ_i ~ InverseGamma(df_i/2, df_i/(2λ_i))  (reported, uncertain)
Study effect: θ_i ~ Normal(μ, τ²)                (latent true effects)
Observation: y_i ~ Normal(θ_i, 1/λ_i)            (data)
```

**Simplified version** (when df unknown):
```
# Stage 1: True precisions (latent)
log(λ_i) ~ Normal(log(1/σ_i²), 0.2²)           (uncertainty around reported precision)

# Stage 2: Study effects
θ_i ~ Normal(μ, τ²)

# Stage 3: Observations
y_i ~ Normal(θ_i, 1/λ_i)

# Priors
μ ~ Normal(0, 20²)
τ ~ Half-Normal(0, 5²)
```

**Rationale for log-normal on λ_i**:
- Precision is positive → log-scale natural
- SD of 0.2 on log-scale → λ_i can vary by factor of ~1.5
- Centered at reported precision but allows deviations
- Conservative: accounts for σ_i estimation uncertainty

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma_reported = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = len(y_obs)

# Approximate uncertainty in SE (typically SE of SE ≈ SE/sqrt(2n))
# Without n, use conservative estimate: CV ≈ 0.2
se_uncertainty = 0.2  # SD on log-precision scale

with pm.Model() as measurement_error_model:
    # Priors on population parameters
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)

    # Latent true precisions (accounting for SE estimation error)
    # Log-normal centered at reported precision
    log_precision_true = pm.Normal('log_precision_true',
                                   mu=np.log(1/sigma_reported**2),
                                   sigma=se_uncertainty,
                                   shape=J)
    precision_true = pm.Deterministic('precision_true', pm.math.exp(log_precision_true))
    sigma_true = pm.Deterministic('sigma_true', 1/pm.math.sqrt(precision_true))

    # Study-specific effects (non-centered)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood using true (latent) precisions
    y = pm.Normal('y', mu=theta, sigma=sigma_true, observed=y_obs)

    # Derived quantities
    I_squared = pm.Deterministic('I_squared',
                                 100 * tau**2 / (tau**2 + pm.math.mean(sigma_true**2)))

    # Sample
    trace = pm.sample(2000, tune=2000, chains=4,
                     target_accept=0.95, random_seed=42)
```

**Key implementation choices**:
- **Log-normal for precision**: Ensures positivity, natural for scale parameters
- **Non-centered parameterization**: Still needed for θ_i
- **Derived σ_true**: Shows how latent SEs differ from reported
- **Conservative uncertainty**: Can adjust se_uncertainty based on domain knowledge

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior σ_true essentially identical to reported σ_i**
   - Evidence: For all i, posterior σ_true_i within ±2% of σ_reported_i
   - Interpretation: SE estimation uncertainty is negligible; standard model sufficient
   - Action: Revert to Model 1 (simpler)

2. **Posterior for σ_true wildly different from reported values**
   - Evidence: Multiple studies show σ_true > 2×σ_reported or < 0.5×σ_reported
   - Interpretation: Our uncertainty model is miscalibrated or reported SEs are wrong
   - Action: Investigate data quality; contact study authors

3. **No material change in μ or τ posteriors**
   - Evidence: Posterior means and SDs for μ, τ differ by <5% from standard model
   - Interpretation: Added complexity buys nothing
   - Action: Report standard model for parsimony

4. **Computational issues**
   - Evidence: >5% divergences, poor mixing, ESS < 100
   - Interpretation: Model too complex for data; latent precision not identifiable
   - Action: Simplify to standard hierarchical model

5. **Prior on se_uncertainty dominates**
   - Evidence: Posterior for log_precision_true essentially reproduces prior
   - Interpretation: Data uninformative about true precisions; model unidentified
   - Action: Fix σ_i at reported values (standard approach)

### Expected Posterior Characteristics

**If measurement error is meaningful**:

1. **σ_true posteriors**:
   - Medians: Within ±10-20% of reported σ_i
   - 95% CrI: Wider than point estimates, reflecting uncertainty
   - Studies with extreme σ_i show larger adjustments (regression to mean)

2. **μ and τ posteriors**:
   - Similar to Model 1 but with wider credible intervals
   - Accounting for SE uncertainty adds ~5-15% to posterior SD
   - Point estimates remain near 7-8 for μ

3. **Effective sample size**:
   - Slightly lower ESS than Model 1 (more parameters, higher correlation)
   - Still acceptable if ESS > 200-400

4. **Model comparison**:
   - WAIC/LOO likely penalizes extra complexity
   - May prefer simpler model unless SE uncertainty is substantial
   - Useful as sensitivity analysis even if not "best" model

**If measurement error is negligible** (expected for this dataset):

1. **σ_true ≈ σ_reported**: Posteriors tightly centered at reported values
2. **Results nearly identical to Model 1**
3. **Model comparison favors simpler model**

**Value proposition**: Even if posteriors are similar, this model quantifies sensitivity to SE specification uncertainty—valuable for robustness claims.

---

## Model 3: Robust Hierarchical Model (Student-t Likelihood)

### Model Class
Hierarchical meta-analysis with heavy-tailed likelihood for outlier robustness

### Theoretical Justification

**Why heavy tails matter**:
- EDA shows no outliers, but with J=8, a single future observation could be extreme
- Normal likelihood is fragile: one outlier can dramatically shift estimates
- Student-t is a mixture of normals with different variances → automatic downweighting
- Degrees of freedom ν controls tail heaviness: ν→∞ gives normal, ν<5 is very heavy

**When to use**:
- When skeptical of perfect normality (always, in practice)
- For robustness to future data that might contain outliers
- When stakes are high and you want defensive modeling
- When you can't verify all studies used identical measurement protocols

**Trade-off**:
- Added complexity (1 extra parameter ν)
- Slightly wider intervals even with clean data
- More conservative inference (feature, not bug)

### Mathematical Specification

**Likelihood**:
```
y_i | θ_i, σ_i, ν ~ Student_t(ν, θ_i, σ_i²)    (robust likelihood)
θ_i | μ, τ ~ Normal(μ, τ²)                      (hierarchical structure)
```

**Priors**:
```
μ ~ Normal(0, 20²)                              (population mean)
τ ~ Half-Normal(0, 5²)                          (between-study SD)
ν ~ Gamma(2, 0.1)                               (degrees of freedom)
```

**Prior on ν rationale**:
- Gamma(2, 0.1) has mean=20, SD=14
- Allows ν < 5 (very heavy tails) if data demand it
- Has substantial mass at ν > 30 (near-normal) as null hypothesis
- Data-driven: will choose tail heaviness from observed deviations

**Alternative prior** (more conservative):
```
ν ~ Gamma(2, 0.2)    # Mean=10, allows heavier tails more easily
```

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = len(y_obs)

with pm.Model() as robust_hierarchical_model:
    # Priors on population parameters
    mu = pm.Normal('mu', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)  # Heavy-tail robustness

    # Study-specific effects (non-centered)
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Robust likelihood (Student-t)
    y = pm.StudentT('y', nu=nu, mu=theta, sigma=sigma, observed=y_obs)

    # Derived quantities
    I_squared = pm.Deterministic('I_squared',
                                 100 * tau**2 / (tau**2 + pm.math.mean(sigma**2)))

    # Effective normality: is ν > 30?
    effectively_normal = pm.Deterministic('effectively_normal', nu > 30)

    # Sample
    trace = pm.sample(2000, tune=2000, chains=4,
                     target_accept=0.95, random_seed=42)
```

**Technical considerations**:
- **Student-t in PyMC**: Parameterized with ν, μ, σ (not precision)
- **Check posterior for ν**: If ν > 30, normal adequate; if ν < 10, real outliers
- **Derived indicator**: `effectively_normal` is interpretable summary
- **Computational cost**: Slightly slower than normal likelihood, but manageable

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior for ν concentrates at ν > 50**
   - Evidence: Posterior median ν > 50, 95% CrI doesn't include values < 20
   - Interpretation: Data are effectively normal; Student-t adds no value
   - Action: Use simpler normal likelihood (Model 1)

2. **Posterior for ν pushes toward ν < 2**
   - Evidence: Posterior mode ν < 2 (undefined variance!)
   - Interpretation: Extreme outliers present OR model badly misspecified
   - Action: Investigate data quality; consider mixture model or discrete outlier detection

3. **Posteriors for μ, τ essentially identical to normal model**
   - Evidence: <1% difference in posterior means and SDs compared to Model 1
   - Interpretation: Robustness doesn't matter for these data
   - Action: Report normal model for simplicity and interpretability

4. **Poor model fit despite heavy tails**
   - Evidence: Posterior predictive checks still fail (p-values <0.05 or >0.95)
   - Interpretation: Problem isn't outliers; fundamental model misspecification
   - Action: Reconsider entire modeling framework (non-normal base distribution, systematic biases)

5. **ν posterior matches prior exactly**
   - Evidence: Posterior ≈ Gamma(2, 0.1) with no data updating
   - Interpretation: ν is unidentified; J=8 insufficient to learn tail behavior
   - Action: Fix ν at reasonable value (e.g., ν=4) or revert to normal

### Expected Posterior Characteristics

**Given clean data from EDA**:

1. **ν posterior**:
   - Median: 15-30 (moderately heavy tails, but near-normal)
   - 95% CrI: [5, 60] (wide uncertainty with J=8)
   - Interpretation: Data consistent with normality but allowing safety margin

2. **μ posterior**:
   - Mean: 7-8 (similar to other models)
   - SD: 4.5-5.5 (slightly wider than normal likelihood)
   - 95% CrI: [-2, 17] (conservative)

3. **τ posterior**:
   - Similar to Model 1, possibly slightly larger
   - Student-t can attribute heterogeneity to either τ or heavy tails
   - More uncertainty than normal model

4. **θ_i posteriors**:
   - Less extreme shrinkage than normal model
   - Outlier studies (if any) shrunk less aggressively
   - "Robust" means tolerating more variation before calling it an outlier

5. **Model comparison**:
   - WAIC/LOO likely similar to Model 1 (within 1-2 units)
   - May slightly favor normal if data truly clean
   - Value is in robustness, not better fit

6. **Convergence**:
   - May need target_accept=0.95 or 0.99
   - ESS for ν typically lower than other parameters (acceptable if >100)
   - Watch for divergences if ν poorly identified

**Interpretation strategy**:
- If ν > 30: "Data consistent with normality"
- If 10 < ν < 30: "Modest tail heaviness; robust estimates provided"
- If ν < 10: "Evidence for outliers or non-normality; investigate further"

---

## Priority Ranking

### 1st Priority: Model 1 (Adaptive Hierarchical Normal Model)

**Justification**:
- Best balance of flexibility and parsimony
- Directly addresses the key scientific question: fixed vs random effects?
- EDA suggests homogeneity, but hierarchical structure tests this formally
- Non-centered parameterization ensures computational stability
- Posterior for τ will tell us if heterogeneity exists
- Standard approach in meta-analysis; well-understood and widely accepted

**Expected outcome**: τ posterior near zero, confirming EDA findings, but with quantified uncertainty.

**Why start here**: If this model works well, it may be sufficient. Other models are refinements/robustness checks.

---

### 2nd Priority: Model 3 (Robust Hierarchical Model)

**Justification**:
- Minimal added complexity (one parameter ν)
- Provides robustness to potential outliers without needing to identify them
- With J=8, any future observation could be influential → robustness valuable
- ν posterior is interpretable: directly answers "are tails heavier than normal?"
- If ν > 30, we've validated normality assumption; if not, we've protected against it

**Expected outcome**: ν ≈ 20-30, confirming normality is adequate but providing safety margin.

**Why second**: Complements Model 1 by testing distributional assumptions. Easy to compare directly.

---

### 3rd Priority: Model 2 (Measurement Error Model)

**Justification**:
- Theoretically sound (σ_i ARE estimates), but may be overkill for this problem
- Without within-study sample sizes, we're making assumptions about SE uncertainty
- Expected to give similar results to Model 1, making it mainly a sensitivity analysis
- Most complex computationally and conceptually
- Value is in demonstrating robustness, not likely to change conclusions

**Expected outcome**: Posteriors similar to Model 1 with slightly wider intervals.

**Why third**: Technically interesting but likely unnecessary. Only pursue if reviewers question SE assumptions or if we obtain within-study sample sizes.

---

## Synthesis: Recommended Analysis Strategy

### Phase 1: Core Analysis
1. **Fit Model 1** (Hierarchical Normal)
   - Check convergence (R_hat, ESS, trace plots)
   - Examine τ posterior: Is heterogeneity detected?
   - Compare to fixed effect estimate from EDA
   - Posterior predictive checks: Does model reproduce data?

### Phase 2: Robustness
2. **Fit Model 3** (Robust Hierarchical)
   - Compare posteriors for μ, τ to Model 1
   - Examine ν posterior: Is normality adequate?
   - Check if any studies are downweighted
   - Assess sensitivity: Do conclusions change?

### Phase 3: Sensitivity (if time permits)
3. **Fit Model 2** (Measurement Error)
   - Compare to Model 1 and 3
   - How much do σ_true differ from reported?
   - Does accounting for SE uncertainty matter?

### Phase 4: Model Comparison
4. **Formal comparison**
   - Compute WAIC or LOO-CV for all models
   - Compare posterior predictive performance
   - Assess computational efficiency
   - Consider parsimony vs fit trade-off

### Phase 5: Reporting
5. **Synthesis**
   - Report Model 1 as primary analysis
   - Report Model 3 as robustness check
   - Report Model 2 if results differ materially
   - State clearly when models agree (builds confidence)
   - State clearly when they disagree (identifies sensitivity)

---

## Red Flags That Would Trigger Major Pivots

### Evidence Against ALL Hierarchical Approaches

1. **Extreme prior sensitivity**: If conclusions flip with modest prior changes, data are too weak for these models. Need more data or simpler fixed effect model.

2. **Systematic posterior predictive failures**: If all models fail to reproduce basic data features (mean, variance, extremes), we're missing something fundamental. Consider:
   - Measurement model is wrong (e.g., y_i measured on different scales)
   - Dependence among studies (violates independence assumption)
   - Systematic bias not captured by random effects

3. **Computational breakdown across all models**: Persistent divergences, poor mixing, funnel geometry in all variants suggests problem isn't solvable with current data.

4. **τ posterior completely dominated by prior**: If data don't update prior on τ, we can't learn about heterogeneity with J=8. Fix τ=0 (fixed effect) or report high uncertainty.

### Evidence FOR Different Model Class Entirely

1. **Bimodal posteriors for μ or θ_i**: Suggests mixture model with distinct subgroups, not smooth hierarchical structure.

2. **Strong patterns in residuals**: If residuals correlate with study characteristics (even if we don't have covariates), suggests meta-regression needed.

3. **Heavy-tailed model demands ν < 3**: Indicates extreme outliers or fundamentally non-normal process. Consider:
   - Contaminated normal (explicit outlier model)
   - Skew-normal or other non-symmetric distributions
   - Discrete mixture model

4. **Evidence of dependence**: If studies cluster (e.g., by research group, geography), need correlated effects model.

---

## Key Assumptions and Potential Violations

### Assumptions Shared by All Three Models

1. **Independence**: Studies are independent conditional on parameters
   - **Violation risk**: Studies from same lab/group might be correlated
   - **Diagnostic**: Cluster studies by author; check if within-cluster correlation exists
   - **Solution**: Nested hierarchy (study within lab)

2. **Exchangeability**: Study-specific effects θ_i are exchangeable (no ordering)
   - **Violation risk**: Studies conducted over time might show temporal trends
   - **Diagnostic**: Plot θ_i posterior means vs study ID; look for trends
   - **Solution**: Meta-regression with time covariate

3. **Measurement model**: y_i ~ f(θ_i, σ_i) with known σ_i
   - **Violation risk**: σ_i misreported, correlated measurement errors
   - **Diagnostic**: Check if reported CIs match y_i ± 1.96σ_i
   - **Solution**: Model 2 addresses this partially

4. **No publication bias**: All studies reported regardless of results
   - **Violation risk**: Small studies with null results unpublished
   - **Diagnostic**: EDA found no evidence, but low power with J=8
   - **Solution**: Selection models (beyond scope but should acknowledge limitation)

### Model-Specific Assumptions

**Model 1** (Normal hierarchy):
- Study effects θ_i ~ Normal(μ, τ²): Symmetry, no heavy tails
- **Diagnostic**: Check posterior predictive distributions of min(y), max(y)

**Model 2** (Measurement error):
- Precision uncertainty is log-normal with known CV
- **Assumption**: CV ≈ 0.2 (reasonable but arbitrary without sample sizes)

**Model 3** (Student-t):
- All studies share same ν (tail heaviness homogeneous)
- **Alternative**: Study-specific ν_i (but likely unidentified with J=8)

---

## Success Criteria for This Modeling Effort

### What "Success" Looks Like

1. **Convergence**: All models converge with R_hat < 1.01, ESS > 400
2. **Consistency**: Models 1 and 3 give similar μ posteriors (within ±10%)
3. **Coherence**: Posterior for τ consistent with EDA finding of I²=0%
4. **Robustness**: Conclusions insensitive to reasonable prior choices
5. **Validation**: Posterior predictive checks pass for all models
6. **Interpretability**: Results can be clearly communicated to domain experts

### What "Failure" Looks Like

1. **Incoherence**: Models disagree wildly about μ or τ
2. **Non-convergence**: Persistent computational issues across models
3. **Prior-data conflict**: Posterior fighting against prior or data
4. **Predictive failure**: Models can't reproduce basic data features
5. **Non-identification**: Posteriors match priors (data uninformative)

### What Would Make Me Reject Hierarchical Modeling Entirely

1. **All hierarchical models fail diagnostics while simple fixed effect works perfectly**: Data are truly homogeneous, hierarchical structure adds noise not signal.

2. **Between-study heterogeneity τ is unidentifiable**: With J=8, we may not have enough data to estimate both μ and τ reliably. Better to fix τ=0 or report it as unknown.

3. **Domain experts provide strong prior that true effects MUST be identical**: If scientific theory demands homogeneity, forcing random effects is inappropriate.

---

## Final Thoughts: Philosophy of This Design

I'm proposing hierarchical models despite EDA evidence for homogeneity because:

1. **Absence of evidence ≠ evidence of absence**: J=8 gives low power to detect τ < 5
2. **Hierarchical models are conservative**: They automatically pool when appropriate
3. **Partial pooling is principled**: Provides optimal bias-variance tradeoff
4. **Scientific plausibility**: Perfect homogeneity across studies is theoretically unlikely

But I'm also proposing **clear falsification criteria** because:

1. **Data may genuinely be homogeneous**: If so, fixed effects is correct and simpler
2. **Overparameterization is wasteful**: Don't fit complexity the data can't support
3. **Computational issues signal problems**: Divergences often mean model is wrong

**Core principle**: Let the data tell us whether hierarchical structure is needed. If τ→0, we've learned something. If τ>0, we've captured it appropriately. Either way, we win.

**Expected result**: Model 1 will show τ near zero, confirming homogeneity, but with proper uncertainty quantification. Models 2 and 3 will provide robustness checks. All will agree on μ ≈ 7-8 with wide intervals.

**What would surprise me**: If τ posterior has substantial mass at τ > 5, suggesting heterogeneity masked by large σ_i. This would justify the hierarchical approach and invalidate fixed effects model from EDA.

---

**End of Model Proposals - Designer 3**
