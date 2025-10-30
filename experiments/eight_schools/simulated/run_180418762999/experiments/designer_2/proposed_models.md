# Bayesian Model Proposals for Hierarchical Measurement Error Dataset
## Designer 2 - Independent Analysis

**Date**: 2025-10-28
**Dataset**: 8 observations with known measurement error
**Approach**: Critical, falsification-focused model design

---

## Executive Summary

The EDA suggests complete pooling with homogeneous groups (tau^2 = 0, p = 0.42). However, I propose three model classes that challenge different aspects of this conclusion:

1. **Model A**: Complete pooling baseline (what EDA recommends)
2. **Model B**: Weakly informative hierarchical model (tests whether we can detect ANY group structure)
3. **Model C**: Measurement error misspecification model (challenges the assumption that sigma values are correct)

**Critical Stance**: The EDA may be misleading. With n=8 and SNR~1, we have very limited power to detect group structure. The "zero between-group variance" finding could be an artifact of small sample size combined with large measurement errors. I will treat each model as a hypothesis to be falsified.

---

## Model A: Complete Pooling with Known Measurement Error

### Mathematical Specification

**Likelihood**:
```
y_i ~ Normal(mu, sigma_i)    for i = 1,...,8
```
where sigma_i are the known measurement standard deviations.

**Prior**:
```
mu ~ Normal(0, 30)
```

### Design Rationale

**Why this prior, not N(10, 20)?**
- The EDA-suggested prior N(10, 20) is too informative given the evidence
- The weighted mean of 10.02 has SE of 4.07, so the data are not that strong
- A more skeptical prior centered at 0 with wider SD=30 better represents genuine prior ignorance
- This tests whether the data truly support mu > 0 or if the EDA conclusion is prior-dependent

**Why this model might be RIGHT**:
- Chi-square homogeneity test p=0.42
- Between-group variance decomposition gives tau^2 = 0
- Simple and parsimonious

**Why this model might be WRONG**:
1. **Assumption violation**: The sigma values might be underestimated (laboratories often underestimate uncertainty)
2. **Hidden heterogeneity**: With n=8, we lack power to detect moderate group differences (tau ~ 5-10)
3. **Oversimplification**: Assumes all observations come from identical process, which may not be scientifically plausible
4. **One extreme value**: Group 4 (y = -4.88) is the only negative observation - could indicate a subgroup

### Falsification Criteria

**I will abandon this model if**:
1. Posterior predictive checks show systematic misfit (e.g., observed variance consistently outside 95% predictive interval)
2. Leave-one-out cross-validation gives poor predictive performance for specific groups (suggests heterogeneity)
3. Posterior for mu has substantial mass on both sides of zero (suggests model is not learning from data)
4. Prior-posterior conflict: prior and likelihood in strong disagreement (indicator of model misspecification)

**Quantitative red flags**:
- LOO-CV: any Pareto k > 0.7 (influential observation suggesting model inadequacy)
- Posterior predictive p-value < 0.05 for variance test (underfitting the spread)
- Effective sample size < 100 for mu (sampling difficulties suggest misspecification)

### Expected Computational Challenges

**Easy sampling**:
- Single parameter (mu) with Gaussian posterior
- No divergences expected
- Should converge in < 1000 iterations

**Diagnostics to check**:
- Trace plots should show good mixing
- R-hat < 1.01 easily achievable
- ESS > 1000 with 4 chains x 2000 iterations

**If I see computational problems**: This would suggest the model is misspecified (Gaussian model should be trivial to sample).

---

## Model B: Weakly Regularized Hierarchical Model

### Mathematical Specification

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i)    for i = 1,...,8
```

**Group-level model**:
```
theta_i ~ Normal(mu, tau)         for i = 1,...,8
```

**Priors**:
```
mu ~ Normal(0, 30)
tau ~ Normal_plus(0, 10)    # Half-normal, weakly regularizing
```

### Design Rationale

**Why half-normal instead of half-Cauchy?**
- Standard recommendation is half-Cauchy for hierarchical models
- BUT: Half-Cauchy has very heavy tails, allowing tau >> observed variation
- With n=8 groups, we have minimal information about tau
- Half-normal with SD=10 is weakly regularizing but prevents extreme values
- **This is a deliberate choice to test sensitivity to prior on tau**

**Alternative parameterization consideration**:
- I considered non-centered parameterization: theta_i = mu + tau * theta_raw_i
- With tau likely near 0, non-centered would be better for sampling
- However, I'll start with centered to see if we get the funnel geometry that would necessitate non-centered
- **If I see divergences, I will switch to non-centered**

**Why this model might be RIGHT**:
1. **More realistic**: Groups likely have some heterogeneity, even if small
2. **Adaptive**: Lets data determine pooling strength
3. **Scientific plausibility**: Different groups probably have different true means
4. **Conservative**: Accounts for model uncertainty

**Why this model might be WRONG**:
1. **Overfitting**: With n=8, we may not have enough data to estimate tau
2. **Sampling difficulties**: Tau near boundary (0) can cause problems
3. **False heterogeneity**: May find spurious group structure in noise
4. **Measurement model**: Assumes sigma_i are known exactly, which may not be true

### Falsification Criteria

**I will abandon this model if**:
1. **Funnel geometry with divergences**: tau posterior concentrates at 0 with divergent transitions (suggests model is fighting the data)
2. **Posterior for tau entirely at boundary**: If tau has 95% of mass below 0.1, model reduces to complete pooling and hierarchical structure is unjustified
3. **Extreme shrinkage**: If all theta_i shrink to within ±2 of mu, we're not learning anything about group structure
4. **Poor predictive performance**: LOO-CV worse than complete pooling (complexity penalty not justified)

**Quantitative red flags**:
- More than 5% divergent transitions (sampling pathology)
- R-hat > 1.05 for tau (convergence failure)
- LOO-CV: ELPD difference from Model A with SE interval overlapping zero (no improvement)
- Posterior median for tau < 1 (essentially no between-group variation)

**Decision point**: If tau < 1, I will compare LOO-CV between this and Model A. If Model A is better (simpler), I abandon hierarchical model.

### Expected Computational Challenges

**Potential difficulties**:
1. **Neal's funnel**: When tau is small, theta_i are tightly constrained, creating funnel geometry
   - Symptom: divergent transitions when tau near 0
   - Solution: switch to non-centered parameterization

2. **Boundary effects**: Half-normal prior on tau can create sampling challenges at tau=0
   - May need higher adapt_delta (0.95 or 0.99)
   - May need more warmup iterations

3. **Low effective sample size**: tau is hardest parameter to estimate
   - Expect ESS(tau) much lower than ESS(mu)
   - Will need longer chains if ESS < 100

**Diagnostics plan**:
- Check for divergences first (most critical)
- If divergences > 1%, switch to non-centered immediately
- Check trace plots for tau (should not stick at boundary)
- Energy diagnostic for HMC quality

---

## Model C: Inflated Measurement Error Model

### Mathematical Specification

**Likelihood** (key difference):
```
y_i ~ Normal(mu, sqrt(sigma_i^2 + tau_meas^2))
```
where:
- sigma_i are the reported measurement errors
- tau_meas is the unobserved additional measurement error (systematic underestimation)

**Prior**:
```
mu ~ Normal(0, 30)
tau_meas ~ Normal_plus(0, 5)    # Additional unaccounted-for measurement error
```

### Design Rationale

**Critical hypothesis**: The reported sigma values may underestimate true measurement uncertainty.

**Why this matters**:
1. **Common problem**: Measurement error estimates are often too optimistic
2. **Resolves discrepancy**: Observed variance (124) < expected measurement variance (166) could mean sigma values are wrong
3. **One-sided error**: Laboratories typically underestimate, not overestimate, uncertainty
4. **Alternative explanation**: Instead of "perfect homogeneity," we have "imperfect measurement error estimates"

**Why this model might be RIGHT**:
1. **Realistic**: Measurement uncertainties are rarely known exactly
2. **Explains negative variance**: If sigma_i are underestimates, observed variance < sum(sigma_i^2) makes sense
3. **Better uncertainty**: Wider credible intervals that better reflect our ignorance
4. **Mechanistic**: Addresses actual data generation process

**Why this model might be WRONG**:
1. **Identifiability**: Hard to separate tau_meas from true between-group variation
2. **Ad hoc**: Adding parameters to make model fit better (overfitting)
3. **No evidence**: If sigma values are from calibrated instruments, they may be trustworthy
4. **Computational**: Additional parameter with limited information

### Falsification Criteria

**I will abandon this model if**:
1. **tau_meas posterior at zero**: If 95% of posterior mass is below 2, then reported errors are adequate
2. **No improvement in fit**: If LOO-CV is worse than Model A, added complexity isn't justified
3. **Extreme values**: If tau_meas > 20, we're explaining all variation with measurement error (not plausible)
4. **Non-identifiability**: If posterior for (mu, tau_meas) shows strong ridge structure (can't separate parameters)

**Quantitative red flags**:
- Posterior mean for tau_meas < 3 (below practical significance threshold)
- Correlation between mu and tau_meas > 0.7 (identifiability issue)
- LOO-CV difference < 0.5 ELPD (no meaningful improvement)
- Posterior SD for mu increases > 2x compared to Model A (losing all precision)

**Key test**: Compare posterior predictive variance to observed variance. If model predicts much wider spread than observed, tau_meas is too large.

### Expected Computational Challenges

**Moderate difficulty**:
1. **Parameter correlation**: mu and tau_meas will be correlated
   - Larger tau_meas implies wider distribution, affects inference about mu
   - May slow mixing

2. **Boundary**: tau_meas ~ Half-normal can stick at boundary if not supported
   - Similar issues to Model B
   - May need higher adapt_delta

3. **Effective sample size**: Both parameters harder to estimate than Model A
   - Expect ESS(mu) < ESS(mu in Model A)
   - Expect ESS(tau_meas) to be low

**Diagnostic strategy**:
- Pairs plot of (mu, tau_meas) to check correlation
- If correlation > 0.7, identifiability is questionable
- Compare posterior SDs across models (should not inflate wildly)

---

## Model Comparison Strategy

### Primary Comparison Metric: LOO-CV

**Why LOO-CV, not WAIC?**
- LOO-CV is more robust with small sample sizes
- Provides Pareto k diagnostic for influential observations
- Can identify model misspecification through high k values

**Comparison plan**:
1. Fit all three models with same data
2. Compute LOO-CV for each (elpd_loo, SE)
3. Compare using elpd_diff with SE
4. Check Pareto k for all observations

**Decision rules**:
- If Model A has best LOO-CV: Use complete pooling (simplest model wins)
- If Model B improves LOO-CV by > 2 SE: Hierarchical structure is justified
- If Model C improves LOO-CV by > 2 SE: Measurement errors are misspecified

**What if all models are similar?**
- Choose simplest (Model A) by parsimony
- Report uncertainty about model choice
- This would indicate data are insufficient to distinguish hypotheses

### Secondary Metrics

**Posterior predictive checks**:
- Can model reproduce observed variance in y?
- Can model reproduce min/max values?
- Graphical: posterior predictive distribution vs observed data

**Prior sensitivity**:
- Refit Model A with N(10, 20) prior to compare with my N(0, 30)
- If posteriors differ substantially, data are not strong enough
- This is a critical check given n=8

**Computational diagnostics as model checks**:
- Divergences suggest model fighting the data
- Poor ESS suggests sampling difficulties (often indicates misspecification)
- Prior-posterior conflict tests

---

## Alternative Models I Considered But Rejected

### Rejected Model 1: Robust Errors (t-distributed)

**Specification**:
```
y_i ~ Student_t(nu, mu, sqrt(sigma_i^2 + se^2))
mu ~ Normal(0, 30)
nu ~ Gamma(2, 0.1)  # degrees of freedom
se ~ Normal_plus(0, 10)
```

**Why rejected**:
- EDA shows no outliers (all |z| < 2.5)
- Normality tests pass (Shapiro-Wilk p = 0.67)
- Adding t-distribution is unnecessary complexity
- Would have identifiability issues with nu, se, and mu

**When I would use it**:
- If Model A posterior predictive checks fail for extreme values
- If LOO-CV shows high Pareto k for specific observations
- If one observation is much more influential than others

### Rejected Model 2: Mixture Model

**Specification**:
```
y_i ~ Normal(mu_1, sigma_i) with probability p
y_i ~ Normal(mu_2, sigma_i) with probability (1-p)
```

**Why rejected**:
- EDA gap analysis found no strong clustering (ratio 1.82 < 2.5)
- With n=8, cannot reliably estimate two subgroups
- Only one negative observation (Group 4), insufficient for mixture
- Weak scientific justification for two latent groups

**When I would use it**:
- If Group 4 shows high Pareto k consistently (influential outlier)
- If domain knowledge suggests two distinct processes
- If we had n > 20 and clearer bimodality

### Rejected Model 3: Structured Priors Based on SNR

**Specification**:
```
theta_i ~ Normal(mu, tau * f(SNR_i))
```
where f(SNR_i) gives larger variance to low-SNR groups.

**Why rejected**:
- Circular reasoning: using data (y/sigma) to set prior variance
- Overly complex for exploratory analysis
- No strong theoretical justification
- Better handled through measurement error model

**When I would use it**:
- In confirmatory analysis with external SNR information
- If we had prior knowledge that certain groups are more variable
- Multi-stage modeling with separate calibration data

---

## Stress Tests and Robustness Checks

### Stress Test 1: Leave-One-Out at Model Level

**Procedure**:
1. Fit Model A leaving out each observation in turn (8 fits)
2. Check stability of posterior for mu
3. Identify influential observations

**Success criteria**:
- Posterior mean for mu should shift < 5 units when dropping any observation
- No single observation should change posterior SD by > 50%

**Failure interpretation**:
- If Group 4 (negative value) is highly influential: Suggests heterogeneity, reconsider Model B
- If any group changes conclusions: Model is too sensitive, need more robust approach

### Stress Test 2: Prior Sensitivity for tau (Model B)

**Procedure**:
Refit Model B with three different priors on tau:
1. Half-Normal(0, 10) - my proposed prior
2. Half-Cauchy(0, 5) - standard recommendation
3. Exponential(0.2) - more concentrated at zero

**Success criteria**:
- Posterior median for tau should agree within factor of 2 across priors
- Model comparison results (LOO-CV) should not change

**Failure interpretation**:
- If posterior strongly prior-dependent: Insufficient data, cannot estimate tau reliably
- Should revert to Model A or gather more data

### Stress Test 3: Posterior Predictive Variance Check

**Procedure**:
For each model:
1. Draw samples from posterior predictive distribution
2. Compute variance of predicted y values
3. Compare to observed variance (124.27)

**Success criteria**:
- Observed variance should fall within 50-95% credible interval of predictive variance
- Should not systematically under- or over-predict spread

**Failure interpretation**:
- Systematic underprediction: Model too constrained (wrong prior, missing variance component)
- Systematic overprediction: Model too flexible (overfitting noise)

---

## Decision Points and Escape Routes

### Decision Point 1: After Fitting Model A

**Check these first**:
1. Posterior predictive p-value for variance
2. LOO-CV Pareto k diagnostics
3. Trace plots and ESS

**Possible outcomes**:

| Outcome | Interpretation | Next Action |
|---------|---------------|-------------|
| All diagnostics good, predictive checks pass | Model A adequate | Proceed to sensitivity analysis |
| Pareto k > 0.7 for one observation | Influential outlier | Fit Model B to check if hierarchical helps |
| Posterior predictive variance too small | Missing variance component | Fit Model C (inflated errors) |
| Poor convergence (ESS < 100) | Model misspecification | Reconsider likelihood (t-distribution?) |

### Decision Point 2: After Fitting Model B

**Key comparison**: Model B vs Model A LOO-CV

**Scenarios**:

1. **Model B wins by > 2 SE**:
   - Check tau posterior (should be > 2 for meaningful heterogeneity)
   - Check for divergences (sampling quality)
   - If both good: Hierarchical structure is real, use Model B

2. **Model A wins or tie (within 2 SE)**:
   - Complete pooling is adequate
   - Abandon hierarchical model
   - Complexity penalty outweighs any benefit

3. **Model B has sampling problems (divergences > 5%)**:
   - Try non-centered parameterization
   - If still problems: Model fighting the data, abandon

**Escape route**: If Model B is inconclusive (sampling issues AND similar LOO-CV), skip to Model C.

### Decision Point 3: After Fitting Model C

**Key question**: Are reported sigma values adequate?

**Evidence synthesis**:

| Evidence | Interpretation |
|----------|---------------|
| tau_meas posterior mass > 5 | Measurement errors underestimated |
| tau_meas posterior mass < 2 | Reported errors adequate |
| LOO-CV improvement > 2 SE | Error inflation model preferred |
| Identifiability issues (corr > 0.7) | Cannot distinguish tau_meas from other sources |

**Escape route**: If Model C has identifiability issues, report this as fundamental limitation. Data insufficient to distinguish measurement error misspecification from true homogeneity.

### Major Strategy Pivot Triggers

**I will reconsider the entire modeling approach if**:

1. **All models fail posterior predictive checks**:
   - Suggests fundamental misspecification
   - Might need: mixture model, t-distribution, or different likelihood entirely

2. **Strong prior-posterior conflict across models**:
   - Data are extremely weak (worse than anticipated)
   - Might need: informative priors from domain knowledge, or more data

3. **Extreme parameter values in any model**:
   - mu > 50 or mu < -30: Suggests outliers or data errors
   - tau > 30: Implausible heterogeneity
   - Need to revisit data quality

4. **Systematic computational failures**:
   - Even simple Model A won't converge
   - Indicates data incompatible with assumed model
   - Might need non-parametric approach

---

## Summary of Falsification Framework

### Model A: Complete Pooling
- **Abandon if**: LOO Pareto k > 0.7, posterior predictive variance check fails
- **Red flag threshold**: Any ESS < 100 (should be trivial to sample)
- **Alternative if abandoned**: Model B (heterogeneity) or Model C (error misspecification)

### Model B: Hierarchical
- **Abandon if**: Divergences > 5%, tau posterior median < 1, LOO-CV worse than Model A
- **Red flag threshold**: R-hat > 1.05 for any parameter
- **Alternative if abandoned**: Model A (simplify) or try non-centered (if divergences)

### Model C: Inflated Errors
- **Abandon if**: tau_meas < 2, identifiability issues (corr > 0.7), LOO-CV worse than Model A
- **Red flag threshold**: tau_meas posterior SD > 10 (completely uncertain)
- **Alternative if abandoned**: Model A (reported errors are adequate)

### Overall Strategy
- **Success criterion**: At least one model passes all diagnostics and predictive checks
- **Failure criterion**: All models rejected by their own falsification criteria
- **Stopping rule**: If all fail, report fundamental model inadequacy and need for different approach

---

## Implementation Notes

### Stan vs PyMC Choice

**Recommendation**: Stan for all models

**Rationale**:
- HMC with NUTS is gold standard for these models
- Better divergence diagnostics (critical for Model B)
- More stable for boundary problems (tau near 0)
- Easier to implement non-centered parameterization if needed

**PyMC alternative**:
- Could use for Model A (simple Gaussian)
- Model B might have sampling issues in PyMC (less mature HMC)
- Reserve PyMC for comparison if Stan results are surprising

### Computational Budget

**Estimated time per model**:
- Model A: 1-2 minutes (4 chains, 2000 iterations)
- Model B: 5-10 minutes (may need longer warmup, higher adapt_delta)
- Model C: 5-10 minutes (similar to Model B)

**Total**: ~30 minutes for all models plus diagnostics

**Parallelization**: Run 4 chains in parallel for each model

### Prior Predictive Checks

**Before fitting**, simulate from priors to check sensibility:

Model A:
- mu ~ N(0, 30) should give predictive distribution with most mass in [-90, 90]
- Check: does this cover plausible values?

Model B:
- Simulate mu, tau, theta_i, y_i from priors
- Check: does predictive distribution cover observed data?
- Ensure priors don't exclude observed range

Model C:
- Check implied total variance: sigma_i^2 + tau_meas^2
- Ensure prior on tau_meas doesn't make total variance implausibly large

---

## What I Would Do With More Data

If we could collect more observations:

**High priority** (would change everything):
1. **Increase n to 20-30**: Would enable reliable tau estimation in Model B
2. **Replicate measurements**: Multiple observations per group would validate sigma_i
3. **Calibration data**: External validation of measurement errors

**Medium priority** (nice to have):
1. **Stratified by SNR**: Ensure balance of high and low quality measurements
2. **Covariates**: Group-level predictors to explain heterogeneity
3. **Temporal data**: Repeated measures over time

**What I'd do differently with n > 20**:
- Fit more complex hierarchical structures
- Consider mixture models seriously
- Estimate sigma_i from data rather than treat as known
- Implement cross-validation more rigorously

---

## Final Philosophical Note

**On Complete Pooling**:
The EDA strongly recommends complete pooling. I am deliberately NOT starting from that assumption. With n=8 and SNR~1, we have very weak evidence for OR against group heterogeneity. The fact that between-group variance estimates at zero could be:
1. True homogeneity (EDA conclusion)
2. Insufficient power to detect heterogeneity (my concern)
3. Measurement error misspecification (Model C hypothesis)

**My approach**: Fit models that represent these competing hypotheses and let LOO-CV and posterior predictive checks arbitrate. If complete pooling wins, great - we have confirmed it against alternatives. If not, we've learned something the EDA missed.

**On Falsification**:
I have explicitly stated conditions under which I would abandon each model. This is not a weakness - it's intellectual honesty. Models that survive their own falsification tests are stronger than models that are accepted by default.

**On Uncertainty**:
With n=8, we should expect high model uncertainty. If no clear winner emerges, that IS the finding - we need more data. Reporting "unable to distinguish between complete pooling and modest heterogeneity" is more honest than forcing a conclusion.

---

## File Locations

All Stan/PyMC code will be saved to:
- `/workspace/experiments/designer_2/model_a_complete_pooling.stan`
- `/workspace/experiments/designer_2/model_b_hierarchical.stan`
- `/workspace/experiments/designer_2/model_c_inflated_errors.stan`

Results and diagnostics:
- `/workspace/experiments/designer_2/model_comparison_results.csv`
- `/workspace/experiments/designer_2/diagnostic_report.md`

Visualizations:
- `/workspace/experiments/designer_2/posterior_plots.png`
- `/workspace/experiments/designer_2/model_comparison.png`
- `/workspace/experiments/designer_2/predictive_checks.png`

---

**End of Model Proposals**
