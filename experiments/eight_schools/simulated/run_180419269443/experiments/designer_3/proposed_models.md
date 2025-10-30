# Bayesian Model Proposals: Prior Specification and Model Adequacy
## Designer 3 - Independent Analysis

**Date:** 2025-10-28
**Analyst:** Model Designer 3
**Focus:** Prior specification strategies and model adequacy checking
**Dataset:** J=8 studies, I²=2.9%, tau=2.02, pooled effect=11.27

---

## Executive Summary

This proposal presents **three distinct Bayesian model classes** that differ fundamentally in their prior specification philosophy, not just parameter choices. Each model embodies a different epistemological stance toward small-sample meta-analysis:

1. **Weakly Informative Hierarchical Model** - Standard approach with carefully chosen priors
2. **Prior-Data Conflict Detection Model** - Explicitly models prior-data disagreement
3. **Skeptical-Enthusiastic Ensemble Model** - Combines opposing prior beliefs to test robustness

**Key insight:** With J=8 studies, prior choices dramatically affect inference. The EDA suggests low heterogeneity (I²=2.9%), but this could be:
- True homogeneity (pooling maximally beneficial)
- Underestimation due to small sample (need conservative priors)
- Artifact of one influential study (Study 4 affects estimate by 33%)

**Critical principle:** I will abandon these models if prior-posterior conflicts suggest fundamental misspecification, not just poor prior choice.

---

## Model Class 1: Weakly Informative Hierarchical Model (BASELINE)

### Mathematical Specification

**Likelihood:**
```
y_i ~ Normal(theta_i, sigma_i)  for i = 1,...,8
theta_i ~ Normal(mu, tau)
```

**Priors:**
```
mu ~ Normal(0, 25)              # Weakly informative on mean effect
tau ~ Half-Normal(0, 10)         # Weakly informative on heterogeneity
```

**Derived quantities:**
```
I² = tau² / (tau² + sigma_pooled²)  where sigma_pooled² = mean(sigma_i²)
theta_new ~ Normal(mu, tau)           # Predictive distribution for new study
```

### Theoretical Justification

**Why this prior on mu?**
- Normal(0, 25) places 95% mass on [-50, 50]
- Observed range is [-4.88, 26.08], so prior is appropriately vague
- Centered at zero represents scientific neutrality (no prior expectation)
- SD = 25 chosen to be ~2x the observed SD (11.15), weakly informative but not flat

**Why Half-Normal for tau?**
- Half-Normal(0, 10) is standard in meta-analysis literature (Gelman, 2006)
- Median ~6.7, mode at 0, allowing for both low and moderate heterogeneity
- 95% of mass below tau=19.6, ruling out extreme heterogeneity
- Given observed median sigma=11, this allows tau up to ~1.8x typical SE
- More conservative than Half-Cauchy (heavier tails), appropriate for J=8

**Alternative tau priors considered:**
- Half-Cauchy(0, 5): Heavier tails, allows extreme heterogeneity (may be too permissive)
- Inverse-Gamma(1, 1): Traditional but poor behavior near zero
- Uniform(0, 20): Improper inference for small tau (not recommended)

### Expected Behavior Given EDA

**If EDA is correct (I²=2.9%):**
- Posterior tau will concentrate near 2 (matching DL estimate)
- Posterior mu will be close to 11.27 with tight CI
- Strong shrinkage of individual theta_i toward mu
- Posterior I² will be ~3-5% (slightly higher than frequentist due to uncertainty)

**Prior-posterior comparison:**
- Prior on tau has mode at 0, data pulls it to ~2
- Prior on mu is centered at 0, data pulls it to ~11
- Both priors are "surprised" by data but not in conflict
- Prior predictive: mu ~ N(0, 26.9), should include observed pooled estimate

**Computational expectations:**
- Should converge quickly (<1000 iterations)
- No divergences expected (simple model)
- Effective sample size should be high (>4000 per chain)
- R-hat < 1.01 for all parameters

### Falsification Criteria

**I will abandon this model if:**

1. **Severe prior-posterior conflict on tau:**
   - Posterior mode of tau > 15 (suggesting I² underestimated due to small J)
   - Posterior 95% CI for tau includes upper bound of prior (indicating prior too restrictive)
   - Prior predictive p-value < 0.01 for observed heterogeneity

2. **Evidence of non-normal study effects:**
   - Posterior predictive checks show systematic misfit (e.g., LOO-PIT plots non-uniform)
   - Pareto k > 0.7 for multiple studies (indicating outliers not handled by normal model)
   - Residual patterns suggest skewness or heavy tails

3. **Influential observations dominate:**
   - Study 4 removal changes posterior mu by >50% (suggests fragile inference)
   - Single study has posterior weight > 40% (contradicts partial pooling logic)
   - LOO cross-validation identifies >2 highly influential studies

4. **Prior on mu is overwhelmed:**
   - Posterior mode of mu > 30 (outside plausible effect range)
   - Data suggests different location entirely (e.g., bimodal posterior)
   - Mean(theta_i) diverges substantially from mu (hierarchical structure inappropriate)

5. **Computational pathologies:**
   - Persistent divergences even with tight adaptation
   - Effective sample size < 100 for tau (suggests funnel of hell)
   - R-hat > 1.1 persists after 10,000 iterations

**Red flags (not immediate failure, but reconsider):**
- Study 5 (negative effect) has posterior theta_i still negative with high probability
- Posterior I² > 25% (suggests EDA severely underestimated heterogeneity)
- Prediction interval width > 50 (suggests model overfitting)

### Stan Implementation Considerations

**Parameterization:**
- Use non-centered parameterization if ESS(tau) < 1000:
  ```stan
  theta_i = mu + tau * theta_raw_i
  theta_raw_i ~ Normal(0, 1)
  ```
- Avoids funnel geometry when tau is small

**Priors in Stan:**
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[8] theta_raw;  // non-centered
}
transformed parameters {
  vector[8] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 25);
  tau ~ normal(0, 10);  // half-normal by constraint
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma);  // sigma is data
}
generated quantities {
  real I_squared = tau^2 / (tau^2 + mean(sigma^2));
  real theta_new = normal_rng(mu, tau);
}
```

**Diagnostics to monitor:**
- Trace plots for mu, tau (should be hairy caterpillars)
- Pairs plot for mu-tau correlation (expect negative correlation)
- Posterior predictive distributions vs observed data
- LOO-CV with Pareto k diagnostics

---

## Model Class 2: Prior-Data Conflict Detection Model (DIAGNOSTIC)

### Mathematical Specification

**Likelihood:** (same as Model 1)
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu, tau)
```

**Priors with mixture components:**
```
mu ~ 0.5 * Normal(0, 25) + 0.5 * Normal(11, 8)    # Mixture: skeptical + optimistic
tau ~ 0.7 * Half-Normal(0, 5) + 0.3 * Half-Cauchy(0, 10)  # Mixture: tight + heavy-tailed
```

**Key addition - Conflict detection:**
```
pi_conflict ~ Beta(1, 1)  # Probability of prior-data conflict
z_i ~ Bernoulli(pi_conflict)  # Indicator: is study i in conflict?

# Modified likelihood with conflict mechanism:
y_i ~ (1 - z_i) * Normal(theta_i, sigma_i) + z_i * Normal(theta_i, sigma_i * inflaction_factor)
inflation_factor ~ LogNormal(log(3), 0.5)  # SE inflation if conflict detected
```

### Theoretical Justification

**Why mixture priors?**
- Single "weakly informative" prior imposes hidden assumptions
- Mixture explicitly represents epistemic uncertainty about prior location
- Component 1 (skeptical): centered at null, SD=25 (wide but null-favoring)
- Component 2 (optimistic): centered at observed estimate, SD=8 (data-aligned)
- Mixture weights (50-50) represent genuine uncertainty pre-data

**Why detect conflicts explicitly?**
- With J=8, a single discrepant study can severely bias estimate
- Study 5 (only negative effect) might be from different population
- Rather than treating as outlier, model it as potential conflict with prior assumptions
- Inflation factor increases effective SE for conflicted studies, downweighting them

**Philosophical stance:**
- Standard Bayesian updating assumes prior and likelihood share reality
- With small samples, this assumption may fail
- Model explicitly tests: "Does this study belong to the same process?"
- If many studies flagged (high pi_conflict), suggests prior misspecification

### Expected Behavior Given EDA

**If EDA is correct (homogeneous studies):**
- Posterior pi_conflict will be low (~0.1)
- Few or no z_i = 1 (no studies flagged as conflicts)
- Mixture components will merge (posterior dominated by one component)
- Results will converge to Model 1

**If Study 5 is genuinely different:**
- z_5 = 1 with high probability (>0.8)
- Posterior tau will be smaller (Study 5 excluded from heterogeneity estimate)
- Posterior mu will shift toward 13-14 (positive studies dominate)
- Inflation factor will be large (3-5x)

**Prior predictive implications:**
- Prior on mu is bimodal (mixture), so prior predictive checks must account for this
- Prior predictive p-value for observed mean should be higher than Model 1
- More flexible, but also more complex to interpret

### Falsification Criteria

**I will abandon this model if:**

1. **Conflict mechanism is unused:**
   - Posterior pi_conflict < 0.05 with tight CI (suggests unnecessary complexity)
   - No studies flagged: all z_i = 0 with probability > 0.95
   - Model reduces to Model 1 but with worse LOO-CV (penalty for extra parameters)

2. **Conflict mechanism dominates:**
   - Posterior pi_conflict > 0.5 (suggests most studies are "outliers" - absurd)
   - More than 3 studies flagged (z_i = 1) (indicates model misspecification, not outliers)
   - Inflation factors become extreme (>10x), suggesting overfitting

3. **Mixture components don't resolve:**
   - Posterior on mu remains bimodal (data unable to choose between skeptical/optimistic)
   - Mixture weights in posterior are 50-50 (no information gain)
   - Suggests data is too weak to overcome prior uncertainty

4. **Computational issues:**
   - Label switching between mixture components (MCMC can't distinguish)
   - Non-convergence due to multimodality
   - Excessive autocorrelation (ESS < 50)

5. **Inference is incoherent:**
   - Studies with larger sigma have higher probability of being flagged (mechanism is backwards)
   - Conflict detection is random (no pattern with study characteristics)
   - Posterior predictive checks worse than Model 1 despite added flexibility

**Red flags:**
- Only Study 5 flagged (might just be using mechanism to handle one outlier inefficiently)
- Mixture prior on mu gives narrower posterior than simple prior (overfitting)

### PyMC Implementation Considerations

**Why PyMC for this model?**
- Mixture priors easier to specify than in Stan
- Discrete parameters (z_i) handled more naturally
- Can use pm.Mixture() directly

**PyMC pseudocode:**
```python
with pm.Model() as conflict_model:
    # Mixture prior on mu
    w_mu = pm.Dirichlet('w_mu', a=[1, 1])
    mu_components = pm.Normal.dist([0, 11], [25, 8])
    mu = pm.Mixture('mu', w=w_mu, comp_dists=mu_components)

    # Mixture prior on tau
    w_tau = pm.Dirichlet('w_tau', a=[7, 3])  # favor tight component
    tau_comp1 = pm.HalfNormal.dist(sigma=5)
    tau_comp2 = pm.HalfCauchy.dist(beta=10)
    tau = pm.Mixture('tau', w=w_tau, comp_dists=[tau_comp1, tau_comp2])

    # Conflict detection
    pi_conflict = pm.Beta('pi_conflict', alpha=1, beta=1)
    z = pm.Bernoulli('z', p=pi_conflict, shape=8)

    # Inflation factor for conflicts
    inflation = pm.LogNormal('inflation', mu=np.log(3), sigma=0.5)

    # Likelihood with conflict-adjusted SE
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=8)
    sigma_adj = sigma * (1 + (inflation - 1) * z)  # inflated SE if z=1
    y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma_adj, observed=y)
```

**Diagnostics:**
- Posterior predictive: simulate new datasets, check if they match observed patterns
- Examine z_i posteriors: which studies are flagged and why?
- Compare LOO-CV to Model 1: does complexity buy better predictions?

---

## Model Class 3: Skeptical-Enthusiastic Ensemble Model (ROBUSTNESS)

### Mathematical Specification

**Two parallel models run simultaneously:**

**Model 3a: Skeptical (shrink toward null)**
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu_skep, tau_skep)
mu_skep ~ Normal(0, 10)         # Strong prior: no effect
tau_skep ~ Half-Normal(0, 5)    # Expect low heterogeneity
```

**Model 3b: Enthusiastic (allow large effects)**
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(mu_enth, tau_enth)
mu_enth ~ Normal(15, 15)        # Prior: moderate-large positive effect
tau_enth ~ Half-Cauchy(0, 10)   # Allow higher heterogeneity
```

**Model 3c: Ensemble (mixture of epistemic states)**
```
# Posterior model averaging with stacking weights
omega ~ Dirichlet(alpha=[1, 1])  # Model weights learned from data
mu_ensemble = omega[0] * mu_skep + omega[1] * mu_enth
theta_ensemble_i = omega[0] * theta_skep_i + omega[1] * theta_enth_i

# Quantify agreement
agreement = 1{|mu_skep - mu_enth| < 5}  # Do models agree within 5 units?
```

### Theoretical Justification

**Why two opposing models?**
- With J=8, prior choice critically affects inference
- Rather than pretending priors don't matter, make disagreement explicit
- Skeptical model represents null-hypothesis stance (common in medicine)
- Enthusiastic model represents optimistic stance (common in early-stage research)
- If models converge to similar posteriors, inference is robust to priors
- If models diverge, data are insufficient to overcome prior beliefs

**Why ensemble them?**
- Stacking weights (Yao et al., 2018) are learned from cross-validation
- Better than equal weights (50-50 averaging)
- Better than Bayes factors (sensitive to prior scales)
- Provides optimal predictive combination

**Philosophical stance:**
- Acknowledges that prior choice encodes subjective beliefs
- Tests robustness by deliberately choosing opposing priors
- If models agree, result is trustworthy regardless of prior
- If models disagree, honest reporting of uncertainty

### Expected Behavior Given EDA

**If data are strong (dominate priors):**
- Both models converge: |mu_skep - mu_enth| < 3
- Posterior agreement = 1 with high probability
- Ensemble weights ~[0.5, 0.5] (both models equally good)
- Final mu_ensemble ~ 11 (close to EDA estimate)

**If data are weak (insufficient to overcome priors):**
- Models diverge: mu_skep ~ 5, mu_enth ~ 15
- Posterior agreement = 0
- Ensemble weights favor better-calibrated model (via LOO-CV)
- Final mu_ensemble has high posterior uncertainty

**If Study 4 is influential (33% effect on estimate):**
- Skeptical model less affected (strong prior shrinks estimate)
- Enthusiastic model more affected (weak prior allows larger shifts)
- Divergence in leave-one-out analyses reveals fragility

### Falsification Criteria

**I will abandon this ensemble approach if:**

1. **Models converge trivially:**
   - Both posteriors identical within MCMC noise (priors too similar in practice)
   - Posterior agreement = 1 but only because priors were too weak
   - Ensemble doesn't help: stacking weights are [0.5, 0.5] with no predictive gain

2. **Models diverge absurdly:**
   - |mu_skep - mu_enth| > 20 (posteriors don't overlap at all)
   - Stacking weights are [0, 1] or [1, 0] (one model completely rejected)
   - Suggests one prior was pathological, not genuinely skeptical/enthusiastic

3. **Ensemble is worse than either component:**
   - LOO-CV of ensemble < min(LOO_skep, LOO_enth) (averaging hurts)
   - Posterior predictive checks show ensemble misfits data systematically
   - Suggests models are capturing different aspects, not genuine uncertainty

4. **Stacking weights are unstable:**
   - Weights change dramatically with single study removal (fragile)
   - Weights have high posterior uncertainty (can't decide between models)
   - Cross-validation folds give contradictory weights

5. **Agreement metric is uninformative:**
   - Agreement threshold (5 units) is arbitrary and results depend on choice
   - Models "agree" but both are wrong (posterior predictive checks fail)
   - Agreement doesn't correlate with actual predictive performance

**Red flags:**
- Skeptical model has posterior mu > 15 (prior completely overwhelmed)
- Enthusiastic model has posterior tau < 2 (heterogeneity prior ignored)
- Ensemble has higher posterior variance than either component (instability)

### Stan Implementation Considerations

**Approach: Fit two models separately, then stack**

**Model 3a (Skeptical):**
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 10);      // Skeptical prior
  tau ~ normal(0, 5);      // Expect low heterogeneity
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] log_lik;
  for (i in 1:J) log_lik[i] = normal_lpdf(y[i] | theta[i], sigma[i]);
}
```

**Model 3b (Enthusiastic):**
```stan
// Same structure, different priors
model {
  mu ~ normal(15, 15);     // Enthusiastic prior
  tau ~ cauchy(0, 10);     // Allow higher heterogeneity
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma);
}
```

**Post-processing (Python):**
```python
# Extract log-likelihoods
log_lik_skep = fit_skep.extract()['log_lik']
log_lik_enth = fit_enth.extract()['log_lik']

# Compute stacking weights using LOO
import arviz as az
weights = az.stacking_weights([log_lik_skep, log_lik_enth])

# Ensemble posterior
mu_skep = fit_skep.extract()['mu']
mu_enth = fit_enth.extract()['mu']
mu_ensemble = weights[0] * mu_skep + weights[1] * mu_enth

# Agreement metric
agreement = np.mean(np.abs(mu_skep - mu_enth) < 5)
```

**Diagnostics:**
- Plot posterior distributions of mu_skep and mu_enth side-by-side
- Show stacking weights with bootstrapped CI
- Compute agreement metric for different thresholds (3, 5, 10 units)
- LOO-CV for each model and ensemble

---

## Prioritization and Recommendation

### Fitting Order

**Phase 1: Baseline (MUST FIT FIRST)**
- **Model 1: Weakly Informative Hierarchical**
- Rationale: Standard approach, fast to fit, establishes baseline
- Computational cost: ~1 minute on CPU
- Decision point: If Model 1 passes all diagnostics, proceed to Phase 2

**Phase 2: Robustness Check (FIT SECOND)**
- **Model 3: Skeptical-Enthusiastic Ensemble**
- Rationale: Tests prior sensitivity without added model complexity
- Computational cost: ~2 minutes (two models in parallel)
- Decision point: If models agree, inference is robust → DONE
- Decision point: If models diverge, proceed to Phase 3

**Phase 3: Diagnostic Deep Dive (FIT ONLY IF NEEDED)**
- **Model 2: Prior-Data Conflict Detection**
- Rationale: Most complex, fit only if Phase 1-2 reveal issues
- Computational cost: ~5 minutes (PyMC slower, discrete parameters)
- Decision point: If conflict mechanism identifies specific studies, refit Model 1 without them

### Recommended Primary Analysis

**Primary model: Model 1 (Weakly Informative Hierarchical)**

**Justification:**
- EDA shows low heterogeneity (I²=2.9%) - simple model appropriate
- J=8 is small, weakly informative priors prevent overfitting
- Standard approach, interpretable, fast
- If diagnostics pass, no need for complexity

**Sensitivity analysis: Model 3 (Ensemble)**

**Justification:**
- Tests robustness to prior choice explicitly
- If models agree (|mu_skep - mu_enth| < 5), inference is trustworthy
- If models disagree, data are insufficient → report uncertainty honestly

**Exploratory analysis: Model 2 (Conflict Detection)**

**Justification:**
- Fit only if Model 1 shows poor LOO-CV for some studies
- Or if Study 4/5 influence analyses suggest outliers
- Most complex, so reserve for situations where simpler models fail

### Expected Findings and Pivots

**Scenario A: EDA is correct (most likely)**
- Model 1 fits well, no diagnostics fail
- Model 3 shows convergence (agreement ~ 1)
- Model 2 unnecessary
- **Conclusion:** Pooled effect ~ 11, tau ~ 2, strong evidence for homogeneity

**Scenario B: Heterogeneity underestimated (possible with J=8)**
- Model 1 shows posterior tau > 10, wide posterior
- Model 3 shows divergence (skeptical/enthusiastic disagree)
- Model 2 flags no specific studies (overall high pi_conflict)
- **Pivot:** Reconsider model class entirely → investigate heavy-tailed models (t-distributions)

**Scenario C: Study 4 or 5 is genuine outlier (plausible)**
- Model 1 shows Pareto k > 0.7 for these studies
- Model 3 diverges only in leave-one-out analyses
- Model 2 flags specific study (z_4 = 1 or z_5 = 1)
- **Pivot:** Report robust estimate excluding outlier, investigate study-level moderators

**Scenario D: Data insufficient to overcome priors (concerning but possible)**
- Model 1 posterior dominated by prior (wide posterior, high uncertainty)
- Model 3 shows strong divergence (stacking weights [0, 1] or [1, 0])
- Model 2 shows high pi_conflict across all studies
- **Pivot:** Abandon pooling, report individual study estimates with warning

---

## Prior Predictive Checks (Pre-Data Validation)

### Model 1 Prior Predictive

**Simulate from priors:**
```
mu ~ Normal(0, 25)
tau ~ Half-Normal(0, 10)
For each of J=8 studies:
  theta_i ~ Normal(mu, tau)
  y_i_sim ~ Normal(theta_i, sigma_i)  # Use observed sigma
```

**Checks:**
1. What is range of simulated pooled estimates?
   - Should include observed 11.27
   - If not, prior is too restrictive
2. What is range of simulated I²?
   - Should include observed 2.9%
   - If not, prior on tau is inappropriate
3. What is range of simulated effect sizes?
   - Should include observed range [-4.88, 26.08]
   - If not, prior is too informative

**Expected results:**
- Prior predictive mean(y_sim) has wide range: ~N(0, 27)
- 95% of prior predictive means should be [-54, 54]
- Observed mean 11.27 should be within this range (CHECK PASSES)
- Prior predictive I² has median ~25%, includes 2.9% (CHECK PASSES)

### Model 2 Prior Predictive

**More complex due to mixtures:**
```
Sample component from mixture
mu ~ component
tau ~ component
pi_conflict ~ Beta(1, 1)
For each study:
  z_i ~ Bernoulli(pi_conflict)
  If z_i = 1: inflate sigma_i by factor ~ LogNormal(log(3), 0.5)
  theta_i ~ Normal(mu, tau)
  y_i_sim ~ Normal(theta_i, adjusted_sigma_i)
```

**Checks:**
1. Does mixture prior on mu make sense?
   - Should be bimodal with modes at 0 and 11
   - Observed 11.27 should be near one mode (CHECK PASSES)
2. What fraction of simulated datasets have conflicts?
   - With Beta(1,1), expect ~50% of datasets to have at least one conflict
   - Too high? Suggests prior is too permissive
3. Do conflicted studies have reasonable SEs?
   - Inflated SE should be ~3x original, not extreme

### Model 3 Prior Predictive

**Separate for each model:**

**Skeptical:**
- Prior predictive mean(y_sim) ~ N(0, 11)
- 95% range: [-22, 22]
- Observed 11.27 at upper tail (prior is skeptical, as intended)

**Enthusiastic:**
- Prior predictive mean(y_sim) ~ N(15, 18)
- 95% range: [-21, 51]
- Observed 11.27 well within range

**Ensemble:**
- Mixture of above two
- Should include observed data comfortably

---

## Computational Considerations

### Stan vs PyMC Choice

**Use Stan for:**
- Model 1 (simple, fast, excellent diagnostics)
- Model 3 (two simple models, parallel fitting)

**Use PyMC for:**
- Model 2 (mixture priors easier, discrete parameters)

### Expected Runtimes (4 chains, 2000 iterations each)

| Model | Platform | CPU Time | GPU? | Divergences Expected |
|-------|----------|----------|------|---------------------|
| Model 1 | Stan | ~1 min | No | 0 |
| Model 2 | PyMC | ~5 min | Optional | <10 (discrete parameters tricky) |
| Model 3a | Stan | ~45 sec | No | 0 |
| Model 3b | Stan | ~45 sec | No | 0 |

### Convergence Criteria

**All models must satisfy:**
- R-hat < 1.01 for all parameters
- ESS_bulk > 1000 for mu, tau
- ESS_tail > 1000 for mu, tau
- No divergent transitions (or <1% if unavoidable)
- MCMC-SE < 0.1 * posterior SD

**If convergence fails:**
1. Increase warmup iterations (1000 → 2000)
2. Tighten adaptation (adapt_delta = 0.95 → 0.99)
3. Reparameterize (centered → non-centered)
4. Check prior: is it improper? multimodal?
5. Consider model misspecification (not just sampling issue)

---

## Model Adequacy Checks (Post-Fitting)

### 1. Posterior Predictive Checks

**For each model, simulate new datasets:**
```
For each posterior sample s:
  theta_s ~ posterior sample
  For each study i:
    y_rep_s_i ~ Normal(theta_s_i, sigma_i)
```

**Compare y_rep to observed y:**
- Overlay histograms (should overlap)
- Q-Q plot (should be linear)
- Test statistics: mean, SD, min, max, range
- Bayesian p-value: P(T(y_rep) > T(y_obs) | data)

**Pass if:** Bayesian p-values in [0.05, 0.95] for key statistics

### 2. Leave-One-Out Cross-Validation (LOO-CV)

**Compute for each model:**
```
For each study i:
  Reweight posterior to approximate p(theta | y_{-i})
  Compute log p(y_i | y_{-i})
Sum log-likelihoods to get elpd_loo (expected log pointwise predictive density)
```

**Diagnostics:**
- Pareto k < 0.5: study is well-fit
- 0.5 < k < 0.7: moderate influence
- k > 0.7: study is highly influential or outlier

**Compare models:**
- Higher elpd_loo is better (more predictive)
- Difference > 2*SE is significant

### 3. Prior-Posterior Overlap

**For each parameter:**
```
overlap = integral[ min(prior(θ), posterior(θ)) ] dθ
```

**Interpret:**
- Overlap ~ 1: posterior same as prior (NO LEARNING)
- Overlap ~ 0.5: moderate learning
- Overlap ~ 0: posterior very different (strong learning OR prior-data conflict)

**Red flag if:**
- Overlap > 0.9 for mu (data too weak)
- Overlap < 0.01 for tau (prior severely misspecified)

### 4. Shrinkage Diagnostics

**Compute for each study:**
```
shrinkage_i = 1 - (posterior_SD(theta_i) / prior_SD(theta_i))
```

**Expected patterns:**
- High precision studies (small sigma_i): less shrinkage (~0.2)
- Low precision studies (large sigma_i): more shrinkage (~0.5)
- If tau is small: all studies shrink strongly (>0.7)

**Check:**
- Does shrinkage match EDA prediction (>95% shrinkage)?
- Are outliers (Study 5) shrunk appropriately?
- Is shrinkage pattern consistent with hierarchical model assumptions?

### 5. Influence Analysis

**For each study i, refit model with study i removed:**
```
mu_(-i) = posterior mean of mu without study i
influence_i = |mu_full - mu_(-i)| / SD(mu_full)
```

**Compare to EDA findings:**
- Study 4 should have influence ~ 33% (EDA found -33.2% change)
- Study 5 should have influence ~ 23% (EDA found +23.0% change)
- If Bayesian influence differs dramatically, investigate why

**Red flag if:**
- Any influence > 50% (single study dominates)
- Influence pattern inconsistent with precision weights

---

## Stress Tests (Adversarial Validation)

### Test 1: Extreme Prior Sensitivity

**Refit Model 1 with:**
- Prior A: mu ~ N(0, 1000) [nearly flat]
- Prior B: mu ~ N(0, 5) [very tight]
- Prior C: tau ~ Uniform(0, 50) [flat on heterogeneity]

**Pass if:** Posterior mu changes by < 20% across priors

**Fail if:** Posterior mu changes by > 50% (data insufficient)

### Test 2: Outlier Injection

**Create synthetic dataset:**
- Keep studies 1-7
- Replace study 8 with extreme outlier: y_8 = 100, sigma_8 = 10

**Refit all models**

**Expected:**
- Model 1: Pareto k > 0.7 for study 8, posterior mu shifts toward 100
- Model 2: z_8 = 1 with high probability (flags outlier)
- Model 3: Models diverge (enthusiastic follows outlier, skeptical resists)

**Pass if:** Models detect outlier appropriately

**Fail if:** Outlier undetected or all models blindly follow it

### Test 3: Data Doubling

**Create synthetic dataset:**
- Duplicate each study: J = 16
- Keep same effects and SEs

**Refit Model 1**

**Expected:**
- Posterior SD(mu) should decrease by ~1/sqrt(2) ~ 0.71x
- Posterior tau should remain similar (between-study variance unchanged)
- ESS should double

**Pass if:** Uncertainty decreases as expected

**Fail if:** Posterior tau also decreases (model conflating within/between variance)

### Test 4: Heterogeneity Injection

**Create synthetic dataset:**
- Keep same mean and SEs
- Add random effects: theta_i ~ N(11, 10) [much higher tau]

**Refit Model 1**

**Expected:**
- Posterior tau should increase to ~8-12
- Posterior I² should increase to ~40-60%
- Shrinkage should decrease

**Pass if:** Model detects increased heterogeneity

**Fail if:** Posterior tau remains small (prior constraining inference too much)

---

## Decision Tree: When to Pivot

```
START: Fit Model 1
  |
  v
[Diagnostics Pass?]
  | YES --> Fit Model 3 (robustness check)
  |           |
  |           v
  |         [Models Agree?]
  |           | YES --> DONE: Report Model 1, note robustness
  |           | NO  --> [Study-specific conflict?]
  |                      | YES --> Fit Model 2, identify outlier
  |                      | NO  --> PIVOT: Data insufficient, report high uncertainty
  |
  | NO --> [Computational issue?]
        | YES --> Reparameterize, increase adapt_delta
        |         | Still fails? --> PIVOT: Try t-distribution (heavy tails)
        |
        | NO --> [Prior-posterior conflict?]
              | YES --> [Conflict on tau?]
              |         | YES --> PIVOT: Use Half-Cauchy or heavier tail prior
              |         | NO  --> [Conflict on mu?]
              |                   | YES --> Check prior predictive, re-examine data
              |
              | NO --> [Influential studies?]
                    | YES --> [>2 studies with Pareto k > 0.7?]
                    |         | YES --> PIVOT: Consider non-exchangeable model
                    |         | NO  --> Fit Model 2, handle specific outliers
                    |
                    | NO --> UNKNOWN FAILURE: Stop and investigate
```

---

## Expected Deliverables

After fitting all models, I will produce:

1. **Comparison table:**
   - Posterior mean, median, 95% CI for mu, tau
   - LOO-CV statistics
   - Pareto k diagnostics
   - Prior-posterior overlap

2. **Visualization suite:**
   - Forest plots with posterior estimates
   - Prior-posterior comparison plots
   - Shrinkage plots (observed → posterior)
   - LOO-PIT diagnostic plots
   - Trace plots and pairs plots

3. **Sensitivity analysis:**
   - Model 3 agreement metric
   - Influence analysis (leave-one-out)
   - Prior sensitivity (vary SD by 50%)

4. **Model adequacy report:**
   - All posterior predictive checks
   - Diagnostic flags raised
   - Recommendation for primary analysis
   - Caveats and limitations

---

## Why These Models Might FAIL (Falsification Mindset)

### Model 1 (Weakly Informative) Will Fail If:

1. **The EDA's low I² is an artifact:**
   - With J=8, I² has huge sampling variance
   - True heterogeneity might be 20-30%, but estimated at 3%
   - Weakly informative prior on tau might not be weak enough
   - Evidence: Posterior tau hits upper bound of prior, poor LOO-CV

2. **Study 4 is not just influential, but from different population:**
   - EDA shows 33% change when removed - extreme
   - If Study 4 is truly different, pooling is inappropriate
   - Model 1 will fit Study 4 poorly (high Pareto k)
   - Evidence: Residual for Study 4 is extreme, posterior predictive miss

3. **Normal distribution is wrong:**
   - Effects might have heavier tails (common in meta-analysis)
   - One extreme study could exist but be unlikely under normal model
   - Model 1 forces normal assumption
   - Evidence: Q-Q plot deviates in tails, posterior predictive fails

### Model 2 (Conflict Detection) Will Fail If:

1. **Conflict mechanism is too flexible:**
   - With 8 studies and many parameters (pi_conflict, z_i, inflation), overfitting likely
   - Model might "explain" noise by flagging random studies
   - Evidence: z_i posterior is non-informative, stacking weights favor Model 1

2. **Discrete parameters cause mixing problems:**
   - PyMC may struggle with discrete z_i, leading to poor convergence
   - Label switching between mixture components
   - Evidence: R-hat > 1.05, ESS < 100, divergences

3. **Inflation factor is poorly identified:**
   - Only 8 studies, if only 1-2 are flagged, inflation factor posterior = prior
   - No information to learn how much to inflate
   - Evidence: Posterior inflation factor is very wide or matches prior exactly

### Model 3 (Ensemble) Will Fail If:

1. **Priors are not actually different enough:**
   - Skeptical N(0, 10) and Enthusiastic N(15, 15) might converge to same posterior
   - Difference in priors washed out by data
   - Evidence: Models agree trivially, stacking weights [0.5, 0.5] with no information

2. **Stacking is unstable:**
   - With LOO-CV on only 8 studies, stacking weights have high variance
   - Might change dramatically with single study removal
   - Evidence: Bootstrap CI on weights includes [0, 1] for both models

3. **Agreement metric is arbitrary:**
   - Threshold of 5 units is chosen ad-hoc
   - Results might change with threshold 3 or 10
   - Binary agreement doesn't capture partial disagreement
   - Evidence: Agreement is threshold-dependent, uninformative

---

## What Would Make Me Reconsider Everything?

### Abandon All Three Models If:

1. **Prior predictive checks fail catastrophically:**
   - None of 1000 prior predictive datasets look remotely like observed data
   - Suggests fundamental model class is wrong (not just prior choice)
   - Pivot to: Non-parametric models, non-exchangeable models

2. **Posterior predictive checks fail for all models:**
   - All three models show systematic misfit (e.g., can't capture variance structure)
   - Bayesian p-values < 0.01 for all key statistics
   - Pivot to: Heavy-tailed models (t-distribution), mixture models

3. **Study 4 drives everything:**
   - Removing Study 4 changes all posterior means by >100%
   - Study 4 has Pareto k > 1.0 in all models
   - Pivot to: Abandon meta-analysis, report "insufficient data for pooling"

4. **Heterogeneity is genuinely large, not small:**
   - All models converge to posterior I² > 50%
   - EDA's 3% was severe underestimate due to small J
   - Pivot to: Meta-regression (find moderators), or report "effects too heterogeneous to pool"

5. **Data are fabricated or erroneous:**
   - Extreme coincidences (e.g., all effects multiples of same number)
   - Impossibly small SEs given sample sizes (if known)
   - Pivot to: Data quality investigation before modeling

---

## Summary: Three Distinct Epistemic Stances

| Aspect | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| **Philosophy** | Standard Bayesian | Conflict-aware | Adversarial validation |
| **Prior stance** | Weakly informative, single choice | Mixture, multiple possibilities | Opposing extremes |
| **Complexity** | Simple (2 params) | Complex (4+ params) | Medium (2 models) |
| **Computational** | Fast, Stan | Slow, PyMC | Medium, Stan (parallel) |
| **Diagnostic focus** | LOO-CV, posterior predictive | Conflict detection | Prior sensitivity |
| **When it fails** | Heterogeneity underestimated | Overfitting, non-convergence | Stacking instability |
| **Best for** | Standard reporting | Outlier detection | Robustness checks |
| **Worst case** | Overconfident pooling | Explains noise | Uninformative agreement |

---

## Final Recommendation

**Primary Analysis:** Model 1 with extensive diagnostics
**Mandatory Sensitivity:** Model 3 to test robustness
**Conditional Analysis:** Model 2 only if outliers detected

**Honest uncertainty quantification:**
- Report posterior + 95% CI for mu, tau
- Report I² with uncertainty (not just point estimate)
- Report prediction interval for future study (much wider than CI)
- Emphasize sensitivity to Study 4 in all reporting

**Key message:** With J=8, prior choices matter. These models test whether conclusions are robust to reasonable prior variation. If models agree, inference is trustworthy. If not, data are insufficient to overcome prior uncertainty, and we must report this honestly.

---

**Files to generate:**
- `/workspace/experiments/designer_3/model_1_spec.stan`
- `/workspace/experiments/designer_3/model_2_spec.py` (PyMC)
- `/workspace/experiments/designer_3/model_3a_spec.stan` (Skeptical)
- `/workspace/experiments/designer_3/model_3b_spec.stan` (Enthusiastic)
- `/workspace/experiments/designer_3/diagnostics_checklist.md`
- `/workspace/experiments/designer_3/prior_predictive_checks.py`

**Next steps:** Await approval, then implement models in specified order (1 → 3 → 2 if needed).
