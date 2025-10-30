# Model Proposals - Designer 2
## Focus: Robust Models and Distributional Alternatives

**Date**: 2025-10-28
**Designer**: Model Designer #2
**Dataset**: 8 observations with measurement error structure
**EDA Summary**: Homogeneous effects (I² = 0%), pooled θ = 7.686 ± 4.072, no outliers detected

---

## Design Philosophy

While the EDA suggests a clean, homogeneous dataset, I adopt a **skeptical, defensive modeling stance**:

1. **Small sample vulnerability**: With only 8 observations, a single undetected outlier or misspecified uncertainty could badly mislead inference
2. **Normality is convenient, not universal**: The EDA normality tests have low power with n=8
3. **Known sigma assumption**: We treat sigma as "known" but these are estimates from primary studies
4. **Tail risk**: Extreme events in the tails matter for decision-making even if rare

My models prioritize **robustness over efficiency** - I'll sacrifice some precision to protect against misspecification.

---

## Model 1: Heavy-Tailed Robust Model (Student-t Likelihood)

### Theoretical Justification

**Core Idea**: Replace Gaussian likelihood with Student-t to automatically downweight observations that deviate from the central tendency.

**Why This Makes Sense**:
- Student-t has heavier tails than normal, providing natural robustness to outliers
- As ν → ∞, converges to normal model (data can choose)
- With n=8, a single bad observation could dominate Gaussian inference
- Even if EDA shows no outliers NOW, measurement error estimates (sigma) might be wrong for some studies
- Protects against "one bad apple" scenario

**When This Would Fail**:
- If most observations are actually outliers (model assumes most data is good)
- If true DGP has bounded support (t has infinite tails)
- If we need exact match to normal theory for regulatory reasons

### Mathematical Specification

**Likelihood**:
```
y_i | theta, nu, sigma_i ~ Student_t(nu, theta, sigma_i^2)  for i = 1,...,8

where:
  - nu: degrees of freedom (controls tail heaviness)
  - theta: location parameter (pooled effect)
  - sigma_i: scale parameter (known measurement SD)
```

**Priors**:
```
theta ~ Normal(0, 20^2)          # Weakly informative, allows wide range
nu ~ Gamma(2, 0.1)                # Mean = 20, SD = 14.14
                                  # Allows nu in [5, 60] with high probability
                                  # nu < 5: very heavy tails
                                  # nu > 30: approximately normal
```

**Alternative nu Prior** (more conservative):
```
nu ~ Gamma(2, 0.5)                # Mean = 4, forces heavier tails
                                  # Use if expecting outliers a priori
```

**Key Assumptions**:
1. Observations are exchangeable conditional on theta
2. Tail behavior is symmetric (not skewed)
3. Scale parameters sigma_i are correctly specified on average
4. A single shape parameter nu applies to all observations

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as robust_t_model:
    # Priors
    theta = pm.Normal('theta', mu=0, sigma=20)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # Likelihood (Student-t with known scale)
    y_obs = pm.StudentT('y_obs', nu=nu, mu=theta, sigma=sigma, observed=y)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)
```

**Computational Notes**:
- Student-t is standard in PyMC, sampling should be efficient
- May need higher target_accept (0.95) if nu explores low values (< 5)
- Check for divergences - indicates geometry problems
- ν posterior may be wide if data doesn't constrain it

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior for nu concentrates at upper bound** (nu > 50): Indicates normal model is sufficient, extra complexity unjustified
2. **Effective sample size (ESS) < 100 for nu**: Parameter is unidentified, prior is driving inference
3. **Posterior predictive checks fail systematically**: Model generates data inconsistent with observed patterns
4. **WAIC/LOO significantly worse than normal model**: Overfitting penalty dominates any robustness gain
5. **Prior-posterior overlap > 80% for nu**: Data not informative about tail behavior

**Red Flags During Fitting**:
- Divergent transitions (> 1% of samples)
- R-hat > 1.01 for any parameter
- Bimodal posterior for theta (suggests model instability)
- Extreme weight placed on single observation

### Expected Posterior Characteristics

**If Model is Valid**:

1. **theta posterior**:
   - Mean: 7.5 to 8.5 (similar to normal model but slightly pulled toward median)
   - SD: 4.0 to 5.0 (slightly wider than normal due to heavier tails)
   - Shape: Symmetric, unimodal

2. **nu posterior**:
   - If data truly normal: Mean > 30, wide uncertainty
   - If outliers present: Mean in [5, 15], tighter posterior
   - If misspecification severe: Mean < 5

3. **Effective weights**:
   - All observations receive weight > 0.5 if no outliers
   - Outliers (if any) automatically downweighted

4. **Posterior predictive**:
   - Mean of y_rep matches mean of y
   - SD of y_rep slightly larger than y (heavier tails)
   - No systematic deviations in PP checks

**Diagnostic Plots**:
- Plot nu posterior: if peaked at boundary, model unnecessary
- Plot effective weights: identifies downweighted observations
- PP check for tail behavior: Q-Q plot of y_rep vs y

---

## Model 2: Contaminated Normal (Mixture Model)

### Theoretical Justification

**Core Idea**: Assume most observations come from the "good" normal distribution, but allow a small proportion to come from a contamination distribution with inflated variance.

**Why This Makes Sense**:
- Explicit model for "most studies are good, some are bad"
- More interpretable than Student-t (can identify which observations are contaminated)
- Allows different contamination mechanisms (inflated uncertainty vs shifted mean)
- Captures the idea that measurement error estimates might be wrong for a subset of studies
- Mixture models can capture multimodality if present

**When This Would Fail**:
- If contamination is gradual, not discrete (Student-t would be better)
- If most observations are contaminated (majority, not minority)
- If contamination structure is more complex than 2-component mixture
- If we need stable point estimates (mixture models can have label switching)

### Mathematical Specification

**Likelihood** (Variance Inflation Version):
```
For each observation i:
  z_i ~ Bernoulli(pi)              # Contamination indicator

  If z_i = 0 (good observation):
    y_i | theta, sigma_i ~ Normal(theta, sigma_i^2)

  If z_i = 1 (contaminated observation):
    y_i | theta, sigma_i ~ Normal(theta, (lambda * sigma_i)^2)

where:
  - pi: contamination probability
  - lambda: variance inflation factor (lambda > 1)
  - theta: pooled effect (same for both components)
```

**Alternative: Location-Shift Contamination**:
```
  If z_i = 1:
    y_i | theta, delta, sigma_i ~ Normal(theta + delta, sigma_i^2)

  where delta is a systematic bias
```

**Priors**:
```
theta ~ Normal(0, 20^2)           # Pooled effect
pi ~ Beta(1, 4)                   # Mean = 0.2 (expect 20% contamination)
                                  # Weakly informative, allows [0, 0.5]
lambda ~ Gamma(2, 0.5)            # Mean = 4, inflation factor
                                  # Allows inflated variance 2x to 10x

Alternative (for location-shift):
delta ~ Normal(0, 15^2)           # Systematic bias if contaminated
```

**Key Assumptions**:
1. Contamination affects minority of observations (pi < 0.5 prior)
2. Contamination mechanism is homogeneous (same lambda for all)
3. Good observations have correctly specified sigma_i
4. Contamination is independent across observations

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as mixture_model:
    # Priors
    theta = pm.Normal('theta', mu=0, sigma=20)
    pi = pm.Beta('pi', alpha=1, beta=4)  # Contamination probability
    lambda_inflate = pm.Gamma('lambda', alpha=2, beta=0.5)  # Variance inflation

    # Mixture components
    sigma_good = sigma
    sigma_bad = lambda_inflate * sigma

    # Likelihood as mixture
    y_obs = pm.NormalMixture('y_obs',
                              w=[1-pi, pi],
                              mu=theta,
                              sigma=pm.math.stack([sigma_good, sigma_bad]).T,
                              observed=y)

    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95)

    # Post-processing: compute posterior probability of contamination
    # (need to do this manually via log-likelihood evaluation)
```

**Computational Notes**:
- Mixture models can have label switching (not an issue here, components differentiated by lambda)
- May have multimodal posterior if data ambiguous about which component
- Computing posterior contamination probabilities requires extra work
- Need longer warmup (1000+) to explore mixture structure

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior for pi concentrates at 0** (< 0.05): No contamination, simpler normal model sufficient
2. **Posterior for pi > 0.6**: Majority contaminated, model assumption violated
3. **Lambda posterior includes 1.0 in 95% CrI**: No evidence of variance inflation
4. **All observations have contamination probability near 0.5**: Model can't distinguish components
5. **LOO worse than normal model by > 5 ELPD**: Overfitting, unnecessary complexity
6. **Label switching in MCMC traces**: Model poorly identified

**Red Flags**:
- Bimodal theta posterior (suggests unstable mixture)
- Extremely wide lambda posterior (parameter unidentified)
- Divergences or low ESS (mixture geometry problems)

### Expected Posterior Characteristics

**If Model is Valid and Contamination Present**:

1. **theta posterior**:
   - Mean: 7.5 to 8.5 (robust to contaminated observations)
   - SD: 3.5 to 4.5 (tighter than Student-t if contamination clear)
   - Should match normal model closely if good observations dominate

2. **pi posterior**:
   - Mean: 0.05 to 0.30 (1-2 out of 8 observations contaminated)
   - If mean < 0.05: no contamination, model unnecessary
   - If mean > 0.4: major data quality issues

3. **lambda posterior**:
   - Mean: 2 to 6 (contaminated observations have 4x to 36x inflated variance)
   - Lower bound > 1.5 (clear evidence of inflation)
   - Upper bound < 10 (not completely wild)

4. **Contamination probabilities** (per observation):
   - Most observations: P(z_i = 1) < 0.1 (clearly good)
   - 1-2 observations: P(z_i = 1) > 0.5 (likely contaminated)
   - Should identify y=28 or y=-3 if any flagged

**Diagnostic Plots**:
- Bar plot of P(contaminated) for each observation
- Posterior predictive checks separately for each component
- Lambda posterior: if overlaps 1, no inflation detected

---

## Model 3: Measurement Error Uncertainty Model (Hierarchical)

### Theoretical Justification

**Core Idea**: Treat the reported sigma_i as **estimates** rather than known constants, with their own uncertainty structure.

**Why This Makes Sense**:
- In real meta-analyses, sigma_i are estimated from primary studies
- Ignoring uncertainty in sigma inflates false confidence
- Small primary studies have unreliable variance estimates
- Some studies may systematically over/underestimate their precision
- Hierarchical structure learns global variance pattern

**When This Would Fail**:
- If sigma_i are truly known (physical measurement devices with known precision)
- If we don't have information about primary study sample sizes
- If variance heterogeneity is extreme (no useful global pattern)
- With only 8 observations, hierarchical variance structure may be poorly identified

### Mathematical Specification

**Likelihood and Hierarchical Structure**:
```
# Level 1: Observations
y_i | theta, tau_i ~ Normal(theta, tau_i^2)   for i = 1,...,8

# Level 2: True uncertainties (not directly observed)
tau_i ~ Half-Normal(sigma_i, psi^2)           # Centered at reported sigma_i
                                              # But allow deviation

# Global pattern
psi ~ Half-Cauchy(0, 5)                       # Global uncertainty in sigmas

where:
  - tau_i: true (latent) standard deviations
  - sigma_i: reported (observed) standard deviations
  - psi: heterogeneity in measurement error estimates
```

**Alternative Specification** (if sample sizes n_i available):
```
# Model reported variance as chi-square
s_i^2 ~ (sigma_i^2 / n_i) * ChiSquare(n_i - 1)

where s_i = reported SE, sigma_i = true SD
```

**Priors**:
```
theta ~ Normal(0, 20^2)           # Pooled effect
psi ~ Half-Cauchy(0, 5)           # Uncertainty in sigma estimates
tau_i ~ Half-Normal(sigma_i, psi^2)  # True SDs centered at reported

# Constraint: keep tau_i within reasonable bounds
tau_i ~ Half-Normal(sigma_i, psi^2) * I(0.5*sigma_i < tau_i < 2*sigma_i)
```

**Key Assumptions**:
1. Reported sigma_i are unbiased on average
2. Errors in sigma_i are independent across studies
3. Uncertainty in sigma follows common pattern (psi applies to all)
4. Deviations are symmetric around reported values

### Implementation Notes (PyMC)

```python
import pymc as pm
import numpy as np

# Data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma_reported = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as uncertainty_model:
    # Priors
    theta = pm.Normal('theta', mu=0, sigma=20)
    psi = pm.HalfCauchy('psi', beta=5)  # Global uncertainty in sigmas

    # True (latent) standard deviations
    tau = pm.TruncatedNormal('tau',
                             mu=sigma_reported,
                             sigma=psi,
                             lower=0.5*sigma_reported,
                             upper=2.0*sigma_reported,
                             shape=8)

    # Likelihood with uncertain sigmas
    y_obs = pm.Normal('y_obs', mu=theta, sigma=tau, observed=y)

    # Sample
    trace = pm.sample(2000, tune=1500, chains=4, target_accept=0.95)
```

**Computational Notes**:
- Truncation bounds prevent extreme tau values
- May have complex posterior geometry (8 latent tau parameters)
- Likely need longer warmup (1500+ iterations)
- Check ESS for all tau_i (hierarchical models often have low ESS)
- Centered vs non-centered parameterization may matter

**Alternative (Non-Centered) Parameterization**:
```python
# Often more efficient for hierarchical models
tau_offset = pm.Normal('tau_offset', mu=0, sigma=1, shape=8)
tau = pm.Deterministic('tau', sigma_reported + psi * tau_offset)
# Add truncation as needed
```

### Falsification Criteria

**I will abandon this model if**:

1. **Posterior for psi concentrates near 0** (< 1): sigma_i effectively known, uncertainty negligible
2. **All tau_i posteriors centered on sigma_i**: No learning about true variances
3. **ESS < 50 for tau parameters**: Model poorly identified, MCMC struggling
4. **Posterior for theta much wider than normal model**: Propagating unnecessary uncertainty
5. **LOO/WAIC much worse than fixed-sigma models**: Complexity not justified
6. **Shrinkage factor < 0.1**: No meaningful hierarchical pooling

**Red Flags**:
- Divergences (hierarchical geometry problems)
- Strong correlations between theta and tau_i (confounding)
- Posterior for psi very wide (unidentified)
- All tau_i hit truncation bounds

### Expected Posterior Characteristics

**If Model is Valid**:

1. **theta posterior**:
   - Mean: 7.0 to 9.0 (may shift slightly as tau_i adjust)
   - SD: 4.5 to 5.5 (wider than fixed-sigma due to propagated uncertainty)
   - Should be robust to misspecification in individual sigma_i

2. **psi posterior**:
   - Mean: 1 to 5 (moderate uncertainty in sigma estimates)
   - If mean < 1: sigma_i are accurate, model unnecessary
   - If mean > 5: large uncertainty, maybe data quality issues

3. **tau_i posteriors**:
   - Centered near sigma_i but with some shrinkage toward global pattern
   - Width proportional to psi
   - Outlier sigma_i (if any) should be adjusted most
   - Shrinkage toward group mean

4. **Effective sample size**:
   - theta: ESS > 400
   - psi: ESS > 200 (often harder to sample)
   - tau_i: ESS > 100 each (acceptable for hierarchical)

**Diagnostic Plots**:
- tau_i vs sigma_i scatter: how much do true SDs differ from reported?
- Shrinkage plot: (tau_i - sigma_i) vs sigma_i
- psi posterior: if narrow and away from 0, model is learning
- Posterior for theta: compare width to fixed-sigma models

---

## Priority Ranking

### Rank 1: Heavy-Tailed Robust Model (Student-t)

**Justification**:
- **Best risk-benefit ratio**: One extra parameter (nu) for substantial robustness gain
- **Interpretable**: nu directly measures tail heaviness
- **Computationally stable**: Standard model in PyMC, no special tricks needed
- **Defensible**: Even if data looks normal, small samples warrant caution
- **Automatic robustness**: Downweights outliers without manual intervention
- **Nests normal model**: If nu > 30, effectively recovers normal inference

**Expected Outcome**:
- If data truly clean: nu posterior > 30, similar inference to normal
- If misspecification present: nu < 15, automatically robust
- Either way, we learn something about tail behavior

**Use Case**: This should be the **default robust model** for this dataset.

---

### Rank 2: Contaminated Normal (Mixture)

**Justification**:
- **Explicit outlier detection**: Can identify which observations are problematic
- **Interpretable components**: Separates "good" from "bad" observations
- **More flexible than t**: Can capture asymmetric contamination
- **Diagnostic value**: Even if not best-fitting, tells us about data quality

**Caveats**:
- More complex (3 parameters: theta, pi, lambda)
- Computational challenges (mixtures can be slow)
- May overfit with n=8
- Requires posterior probability calculation for interpretation

**Expected Outcome**:
- If no contamination: pi near 0, model reduces to normal
- If 1-2 outliers: pi = 0.1-0.25, identifies specific observations
- Useful for **sensitivity analysis** and **outlier identification**

**Use Case**: Fit this as a **diagnostic tool** to identify problematic observations.

---

### Rank 3: Measurement Error Uncertainty Model

**Justification**:
- **Philosophically correct**: Sigma values are estimates, not known
- **Propagates uncertainty**: More honest about total uncertainty
- **Realistic modeling**: Matches actual meta-analysis data structure

**Caveats**:
- Most complex (9 parameters: theta, psi, 8x tau_i)
- May be poorly identified with n=8
- Computational challenges (hierarchical structure)
- Requires additional information (primary study n_i) for best performance
- Without sample sizes, must rely on global pattern with only 8 points

**Expected Outcome**:
- Likely finds psi small (reported sigmas fairly accurate)
- Wider credible intervals for theta (propagated uncertainty)
- May struggle with identification

**Use Case**: Fit this if we have **additional information** (primary study sample sizes) or if sigma accuracy is a major concern. Otherwise, **skip in favor of simpler robust models**.

---

## Model Comparison Strategy

### Recommended Workflow:

1. **Fit all three models** + standard normal baseline (4 models total)

2. **Convergence diagnostics** for each:
   - R-hat < 1.01 for all parameters
   - ESS > 400 for theta, > 200 for other parameters
   - No divergences (< 1%)
   - Visual trace inspection

3. **Model comparison**:
   - LOO-CV (expected log predictive density)
   - WAIC (Watanabe-Akaike Information Criterion)
   - Posterior predictive checks (visual)
   - Parameter interpretability

4. **Sensitivity checks**:
   - Prior sensitivity (vary prior widths by 2x)
   - Influence analysis (leave-one-out for theta)
   - Extreme prior checks (vague vs informative)

5. **Decision rules**:
   - If LOO within 2 ELPD: choose simplest model (normal or Student-t)
   - If LOO difference > 5: trust the winner
   - If convergence issues: simplify or reparameterize
   - If priors dominate posteriors (prior-post overlap > 80%): need more data

### Expected Result:

My prediction: **Student-t model will win or tie with normal** (both within 2 ELPD of each other). The mixture and hierarchical models will likely show the data doesn't justify the added complexity, but will provide useful diagnostics.

**If Student-t wins**: Use it for inference
**If normal wins**: Report Student-t as robustness check
**If mixture wins**: We've discovered data quality issues, investigate further

---

## Falsification Summary Table

| Model | Abandon If... | Red Flag |
|-------|--------------|----------|
| **Student-t** | nu > 50 consistently | Divergences, bimodal theta |
| **Mixture** | pi < 0.05 or pi > 0.6 | Label switching, lambda includes 1 |
| **Hierarchical** | psi < 1.0 | ESS < 50 for tau_i, divergences |

---

## Critical Self-Assessment

### What Could Go Wrong with My Approach:

1. **Over-engineering**: With n=8, these models may be too complex
   - **Mitigation**: Compare to simple normal baseline, use LOO-CV

2. **False positives**: Finding "robustness" where none needed
   - **Mitigation**: Check if robust model posteriors match normal

3. **Computational failure**: Hierarchical model may not converge
   - **Mitigation**: Have simpler fallbacks ready, use non-centered parameterization

4. **Misplaced focus**: Worrying about tails when location uncertainty dominates
   - **Mitigation**: Focus on theta posterior width, not just shape

5. **Prior sensitivity**: With small n, priors matter a lot
   - **Mitigation**: Test multiple prior specifications

### Escape Routes:

- If all robust models fail: Revert to normal model, acknowledge limitation
- If computational issues: Use simpler approximations (ADVI, maximum a posteriori)
- If data truly problematic: Recommend collecting more observations before inference

---

## Implementation Priority

**Must Implement**:
1. Normal model (baseline)
2. Student-t model (primary robust model)

**Should Implement** (if time):
3. Mixture model (diagnostic value)

**Optional**:
4. Hierarchical model (only if other models show sigma misspecification)

---

## Final Notes

**Key Insight**: With n=8, the main source of uncertainty is **small sample size**, not distributional misspecification. Robust models provide insurance against worst-case scenarios but may not dramatically change inference if EDA is correct.

**Success Criteria**:
- A good outcome is **discovering these models are unnecessary** (robust posteriors match normal)
- A great outcome is **identifying subtle issues** the EDA missed
- A failure is **adding complexity without learning anything**

**Decision Point**: After fitting Models 1-2, if they match the normal model closely, **stop and declare victory**. Don't fit Model 3 unless there's evidence of sigma misspecification.

---

**Output Location**: `/workspace/experiments/designer_2/proposed_models.md`
**Status**: Ready for implementation phase
**Dependencies**: EDA findings (confirmed), PyMC environment (required)
