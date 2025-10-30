# Experiment Plan: Bayesian Meta-Analysis Model Development
## Synthesized from Three Independent Model Designers

**Date**: 2025-10-28
**Dataset**: Meta-analysis with J=8 studies, I²=0%, borderline pooled effect
**EDA Report**: `/workspace/eda/eda_report.md`

---

## Executive Summary

Three independent model designers proposed 9 total model classes from complementary perspectives:
- **Designer #1** (Hierarchical): 3 models focusing on partial pooling and shrinkage
- **Designer #2** (Fixed-effects): 3 models focusing on complete pooling
- **Designer #3** (Robust): 3 models focusing on robustness to outliers/misspecification

After removing duplicates and prioritizing by theoretical justification, this plan identifies **4 distinct model classes** to implement, ordered by priority:

1. **Bayesian Hierarchical Meta-Analysis** (Standard) - Primary model
2. **Robust Hierarchical with Student-t** - Robustness check
3. **Bayesian Fixed-Effect Meta-Analysis** - Simplicity check
4. **Precision-Stratified Model** - Exploratory (if time permits)

**Minimum Attempt Policy**: We will attempt at least Models 1 and 2 unless Model 1 fails pre-fit validation.

---

## Model Selection Rationale

### Models Selected for Implementation

**Model 1** (Priority: HIGH - REQUIRED):
- Combines Designer #1's "Adaptive Hierarchical" with Designer #3's heterogeneity framework
- Most flexible: nests both fixed-effect (τ→0) and random-effects (τ>0)
- Addresses "heterogeneity paradox" via data-driven shrinkage
- Recommended by all three designers in synthesis

**Model 2** (Priority: HIGH - REQUIRED):
- Designer #3's "Robust Student-t Meta-Analysis"
- Addresses Study 1 influence via heavy-tailed likelihood
- Critical robustness check given borderline significance
- Learns tail behavior from data (nu parameter)

**Model 3** (Priority: MEDIUM):
- Designer #2's "Fixed-Effect Meta-Analysis"
- Takes I²=0% at face value
- Simplicity benchmark: If this passes all checks, more complex models unnecessary
- Maximum power if homogeneity assumption correct

**Model 4** (Priority: LOW - Optional):
- Designer #2's "Precision-Stratified Model"
- Tests for small-study effects without full random-effects
- Exploratory: implement only if first 3 models show issues

### Models Deferred (Not Duplicates, But Lower Priority)

**Designer #1's "Informative Heterogeneity Model"**:
- Reason: Likely to fail given I²=0% conflicts with typical meta-analysis heterogeneity
- Designer #1 predicted 60% failure probability
- Value: Would test if external evidence applies, but low expected utility

**Designer #2's "Measurement Error Uncertainty Model"**:
- Reason: No evidence in EDA that sigma_i are miscalibrated
- Value: Conservative robustness check, but Designer #3's Student-t addresses same concern differently

**Designer #3's "Finite Mixture Model"**:
- Reason: J=8 likely too small for reliable mixture identification
- Designer #3 noted this may not identify
- Value: Would address clustering (p=0.009 from EDA), but needs larger sample

---

## Model 1: Bayesian Hierarchical Meta-Analysis (Standard)

### Model Class
**Bayesian Random-Effects Meta-Analysis** with adaptive shrinkage

### Mathematical Specification

**Likelihood**:
```
y_i | theta_i, sigma_i ~ Normal(theta_i, sigma_i^2)   for i = 1, ..., 8
```

**Hierarchical Structure**:
```
theta_i | mu, tau ~ Normal(mu, tau^2)
```

**Priors**:
```
mu ~ Normal(0, 50)           # Weakly informative on overall effect
tau ~ Half-Cauchy(0, 5)      # Standard meta-analysis prior (Gelman 2006)
```

**Where**:
- `y_i` = observed effect size (data)
- `sigma_i` = known standard error (data, NOT estimated)
- `theta_i` = true underlying effect for study i
- `mu` = population mean effect (primary estimand)
- `tau` = between-study standard deviation

### Prior Justification

**mu ~ Normal(0, 50)**:
- Weakly informative: 95% prior covers [-98, 98], well beyond observed range [-3, 28]
- Centered at zero (no prior belief about direction)
- Allows data to dominate inference
- Reference: Gelman et al. (2013) "Bayesian Data Analysis"

**tau ~ Half-Cauchy(0, 5)**:
- Standard recommendation for meta-analysis heterogeneity (Gelman 2006, Polson & Scott 2012)
- Heavy tails allow large tau if data supports it
- Mode at zero consistent with I²=0% finding
- Scale=5 is 0.5× mean within-study SE (12.5), conservative for small samples
- Reference: Gelman A. "Prior distributions for variance parameters in hierarchical models." Bayesian Analysis 1(3):515-534, 2006.

**sigma_i = data (KNOWN)**:
- Standard meta-analysis assumption: reported SEs are accurate
- Not estimated, treated as fixed
- Sensitivity: Model 2 (Designer #2's measurement error model) tests this if needed

### What This Model Captures

1. **Partial pooling**: Studies borrow strength based on precision and heterogeneity
2. **Adaptive shrinkage**:
   - If tau≈0: Strong shrinkage (fixed-effect-like)
   - If tau>sigma_i: Weak shrinkage (studies stay near observed y_i)
3. **Measurement error structure**: Precise studies weighted higher
4. **Uncertainty propagation**: Full posterior for all parameters
5. **Nested models**: Contains fixed-effect (tau→0) and no-pooling (tau→∞) as special cases

### Falsification Criteria (ACCEPT/REVISE/REJECT)

**REJECT if any of these hold**:

1. **Posterior predictive failure**:
   - Test: Generate y_rep ~ Normal(theta_i, sigma_i) from posterior
   - Reject if: >1 observed y_i falls outside 95% posterior predictive interval
   - Action: Switch to Model 2 (robust Student-t)

2. **Leave-one-out instability**:
   - Test: Remove each study, compute E[mu | data_{-i}]
   - Reject if: max_i |E[mu | data_{-i}] - E[mu | data]| > 5 units
   - Action: Hierarchical structure inappropriate, consider mixture models

3. **Convergence failure**:
   - Test: R-hat, ESS, divergences after 10K iterations
   - Reject if: R-hat > 1.05 OR ESS < 400 OR divergences > 1%
   - Action: Try non-centered parameterization, then switch models

4. **Extreme shrinkage asymmetry**:
   - Test: Compare theta_i posterior mean to y_i
   - Reject if: Any |E[theta_i] - y_i| > 3*sigma_i
   - Action: Investigate as outlier, switch to Model 2

**REVISE if**:
- Prior-posterior conflict: P(tau > 10 | data) > 0.5 with prior P(tau > 10) < 0.05
  → Action: Refit with more diffuse prior (tau ~ Half-Cauchy(0, 10))
- Unidentifiability: tau posterior essentially uniform
  → Action: Add informative prior from literature

**ACCEPT if**:
- All falsification checks pass
- Convergence achieved (R-hat < 1.01, ESS > 400, no divergences)
- Posterior predictive check shows reasonable fit
- Leave-one-out shows stability (all Δmu < 5)

### Implementation Notes

**Software**: Stan via CmdStanPy (preferred for hierarchical models)

**Parameterization**: Start with centered, switch to non-centered if tau posterior near zero causes divergences

**Sampling**: 4 chains, 2000 iterations each (1000 warmup), adapt_delta=0.95

**Stan code** (centered parameterization):
```stan
data {
  int<lower=1> J;          // Number of studies
  vector[J] y;             // Observed effects
  vector<lower=0>[J] sigma; // Known SEs
}
parameters {
  real mu;                 // Overall mean
  real<lower=0> tau;       // Between-study SD
  vector[J] theta;         // Study-specific effects
}
model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);

  // Hierarchical structure
  theta ~ normal(mu, tau);

  // Likelihood
  y ~ normal(theta, sigma);
}
generated quantities {
  vector[J] log_lik;       // For LOO
  vector[J] y_rep;         // For PPC

  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    y_rep[j] = normal_rng(theta[j], sigma[j]);
  }
}
```

### Expected Challenges

1. **Funnel geometry**: If tau→0, centered parameterization may have divergences → use non-centered
2. **tau weakly identified**: With J=8, tau posterior may be wide → expected, not problematic
3. **Study 1 influence**: May dominate posterior → check via leave-one-out
4. **I²=0% interpretation**: tau posterior may concentrate near zero → this is informative, not failure

---

## Model 2: Robust Hierarchical Meta-Analysis with Student-t

### Model Class
**Hierarchical Bayesian Meta-Analysis with Heavy-Tailed Likelihood**

### Mathematical Specification

**Likelihood** (robust):
```
y_i | theta_i, sigma_i, nu ~ Student-t(nu, theta_i, sigma_i)   for i = 1, ..., 8
```

**Hierarchical Structure**:
```
theta_i | mu, tau ~ Normal(mu, tau^2)
```

**Priors**:
```
mu ~ Normal(0, 25)           # Weakly informative
tau ~ Half-Cauchy(0, 5)      # Standard meta-analysis prior
nu ~ Gamma(2, 0.1)           # Degrees of freedom for tail heaviness
```

**Where**:
- `nu` = degrees of freedom (controls tail heaviness)
  - Low nu (2-5): Heavy tails, robust to outliers
  - High nu (>30): Near-Normal, no robustification needed
  - nu→∞: Model collapses to Model 1 (Normal likelihood)

### Prior Justification

**mu ~ Normal(0, 25)**:
- Moderately informative: 95% prior covers [-50, 50]
- Tighter than Model 1 (SD=50) for computational stability with extra parameter (nu)
- Still covers observed range [-3, 28] comfortably

**tau ~ Half-Cauchy(0, 5)**: Same as Model 1

**nu ~ Gamma(2, 0.1)**:
- Mean = 20, SD = 14.1, Mode ≈ 10
- Allows full range from nu≈1 (very heavy tails) to nu>30 (near-Normal)
- Weakly informative: lets data determine tail behavior
- Reference: Juarez & Steel (2010) "Model-based clustering based on skew-t distributions"

### What This Model Captures

1. **Outlier robustness**: Automatically downweights extreme observations (Study 1)
2. **Data-driven robustification**: nu learned from data
   - If nu<10: Heavy tails important (outliers present)
   - If nu>30: Normal adequate (no outliers)
3. **Robust heterogeneity estimation**: tau less sensitive to extremes
4. **Conservative inference**: Wider intervals when data shows anomalies

### Falsification Criteria

**REJECT if**:

1. **nu posterior concentrates at upper bound** (nu > 50):
   - Evidence: P(nu > 50 | data) > 0.8
   - Interpretation: Student-t converges to Normal; unnecessary complexity
   - Action: Revert to Model 1 (simpler)

2. **Posterior predictive failure**:
   - Test: Generate y_rep from Student-t posterior predictive
   - Reject if: >2 studies (25%) outside 95% intervals
   - Interpretation: Even heavy tails can't fit data
   - Action: Consider mixture model

3. **LOO diagnostics poor**:
   - Reject if: Pareto k > 0.7 for >2 studies
   - Interpretation: Model misspecified structurally
   - Action: Re-examine model class

**ACCEPT if**:
- All checks pass
- Convergence achieved
- nu posterior is informative (not at boundaries)
- Provides better fit than Model 1 OR similar fit with reasonable nu

### Implementation Notes

**Software**: Stan via CmdStanPy

**Sampling**: 4 chains, 2000 iterations, adapt_delta=0.95

**Stan code** (key sections):
```stan
parameters {
  real mu;
  real<lower=0> tau;
  real<lower=1> nu;          // Degrees of freedom
  vector[J] theta;
}
model {
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  nu ~ gamma(2, 0.1);
  theta ~ normal(mu, tau);
  y ~ student_t(nu, theta, sigma);  // Robust likelihood
}
```

### Expected Challenges

1. **nu identification**: May be weakly identified with J=8 → wide posterior OK
2. **Computation**: Student-t slightly slower than Normal → expect +20% runtime
3. **Interpretation**: If nu≈10-30, unclear if robustification was "necessary" → compare to Model 1 via LOO

---

## Model 3: Bayesian Fixed-Effect Meta-Analysis

### Model Class
**Complete Pooling Model** - All studies estimate same common effect

### Mathematical Specification

**Likelihood**:
```
y_i | mu, sigma_i ~ Normal(mu, sigma_i)   for i = 1, ..., 8
```

**Prior**:
```
mu ~ Normal(0, 15)
```

**Key assumption**: theta_i = mu for all i (no heterogeneity, tau=0)

### Prior Justification

**mu ~ Normal(0, 15)**:
- Weakly informative: 95% prior [-30, 30] covers observed range
- SD=15 is 1.5× observed effect SD (10.44)
- Tighter than Model 1 for computational stability (fewer parameters)

### What This Model Captures

1. **Common effect hypothesis**: Assumes I²=0% is real (not artifact)
2. **Maximum power**: Pools all information into single parameter
3. **Inverse-variance weighting**: Automatic via sigma_i
4. **Simplicity benchmark**: If this passes, Models 1-2 may be overparameterized

### Falsification Criteria

**REJECT if**:

1. **Posterior predictive failure**:
   - Reject if: >1 study outside 95% posterior predictive interval
   - Interpretation: Homogeneity assumption violated
   - Action: Switch to Model 1 (hierarchical)

2. **Leave-one-out instability**:
   - Reject if: Any E[mu | data_{-i}] differs from E[mu | data] by >5 units
   - Interpretation: Model too sensitive (heterogeneity present)
   - Action: Switch to Model 1

3. **LOO comparison**:
   - Reject if: Model 1 has Δelpd > 5 (strongly better)
   - Interpretation: Data support heterogeneity despite I²=0%
   - Action: Adopt Model 1

**ACCEPT if**:
- All checks pass
- LOO comparison shows Model 1 not substantially better (Δelpd < 2)
- Posterior predictive fit is reasonable
- Parsimony principle: Simpler model preferred if fit is adequate

### Implementation Notes

**Software**: Stan via CmdStanPy

**Sampling**: 4 chains, 2000 iterations (fast: 1 parameter)

**Stan code**:
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
}
model {
  mu ~ normal(0, 15);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[J] log_lik;
  vector[J] y_rep;
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | mu, sigma[j]);
    y_rep[j] = normal_rng(mu, sigma[j]);
  }
}
```

### Expected Challenges

1. **Likely to fail**: EDA showed 31-point range, clustering (p=0.009)
2. **Useful failure**: If rejected, confirms hierarchical structure necessary
3. **Fast to test**: Simplest model, quick to implement and diagnose

---

## Model 4: Precision-Stratified Fixed-Effect Model (Optional)

### Model Class
**Fixed-Effect with Subgroup Structure** - Middle ground between Models 1 and 3

### Mathematical Specification

**Likelihood**:
```
y_i | mu_g[i], sigma_i ~ Normal(mu_g[i], sigma_i)
```

**Where**:
- Group 1 (high precision): sigma_i ≤ 11 (Studies 2, 4, 5, 6, 7)
- Group 2 (low precision): sigma_i > 11 (Studies 1, 3, 8)
- mu_g[i] = effect for study i's precision group

**Priors**:
```
mu_1 ~ Normal(0, 15)   # High-precision group effect
mu_2 ~ Normal(0, 15)   # Low-precision group effect
```

### What This Model Captures

1. **Small-study effects**: Tests if precision correlates with effect
2. **Partial pooling**: Within groups but not across
3. **Middle ground**: Between complete pooling (Model 3) and full hierarchical (Model 1)

### Falsification Criteria

**REJECT if**:

1. **Groups not different**:
   - Test: P(|mu_1 - mu_2| < 5 | data) > 0.8
   - Interpretation: Stratification unnecessary
   - Action: Use Model 3 (simpler)

2. **LOO comparison**:
   - Reject if: Model 1 has Δelpd > 5
   - Interpretation: Full hierarchical better captures structure
   - Action: Use Model 1

**ACCEPT if**:
- Groups differ meaningfully (|mu_1 - mu_2| > 5)
- LOO competitive with Models 1 and 3
- Provides interpretable explanation of heterogeneity

### Implementation Notes

**Priority**: LOW - Implement only if Models 1-3 show issues
**Expected outcome**: Likely to be rejected (EDA found no precision-effect correlation)

---

## Iteration Strategy

### Phase 3: Model Development Loop

For each model in order of priority:

```
1. Prior Predictive Check
   ├─ PASS → Continue
   └─ FAIL → Document, skip to next model

2. Simulation-Based Validation
   ├─ PASS → Continue
   └─ FAIL → Document, skip to next model

3. Model Fitting (save log_lik in InferenceData)
   ├─ PASS (converged) → Continue
   ├─ FAIL (no convergence) → Try refinement once
   └─ Still FAIL → Document, skip to next model

4. Posterior Predictive Check
   └─ Continue regardless (document fit quality)

5. Model Critique
   ├─ ACCEPT → Add to successful models
   ├─ REVISE → model-refiner → new experiment → loop
   └─ REJECT → Document, next model

Minimum Attempt: Models 1 and 2 (unless Model 1 fails prior/simulation)
```

### Comparison Criteria

**If multiple models ACCEPT**:
- Use LOO-CV via ArviZ (`az.loo`, `az.compare`)
- Δelpd < 2×SE: Models equivalent, prefer simpler (parsimony)
- Δelpd > 2×SE: Prefer better-fitting model
- Report all accepted models if differences are small

**If all models REJECT**:
- Document failure modes
- Consider deferred models (mixture, measurement error uncertainty)
- Possibly: Data insufficient for reliable inference

---

## Success Criteria (Phase 5: Adequacy Assessment)

**Adequate solution achieved if**:

1. **At least one model passes all Phase 3 checks** (ACCEPT)
2. **Convergence diagnostics excellent**:
   - R-hat < 1.01 for all parameters
   - ESS > 400 for all parameters
   - No divergences (or <0.1%)
3. **Posterior predictive checks reasonable**:
   - Most studies within 95% intervals
   - No systematic patterns in residuals
4. **Stable inference**:
   - Leave-one-out: All Δmu < 5
   - Prior sensitivity: Conclusions robust to reasonable prior variations
5. **Interpretable results**:
   - Clear probability statements (e.g., P(mu > 0 | data))
   - Full uncertainty quantification
   - Effect size estimates with credible intervals

**Continue iteration if**:
- Models show fixable issues (wrong priors, bad parameterization)
- Clear improvement path exists
- Computational issues solvable

**Stop and report limitations if**:
- All reasonable models fail
- Data quality issues discovered
- Sample size (J=8) insufficient for reliable inference
- Computational limits reached

---

## Expected Timeline

**Model 1** (Hierarchical):
- Prior predictive: 1 hour
- Simulation validation: 1 hour
- Fitting: 30 min
- Posterior predictive: 30 min
- Critique: 1 hour
- **Total: ~4 hours**

**Model 2** (Robust Student-t):
- Prior predictive: 30 min (modify Model 1)
- Simulation validation: 1 hour
- Fitting: 45 min (slower)
- Posterior predictive: 30 min
- Critique: 1 hour
- **Total: ~3.5 hours**

**Model 3** (Fixed-effect):
- Prior predictive: 30 min
- Simulation validation: 30 min
- Fitting: 15 min (fast)
- Posterior predictive: 20 min
- Critique: 30 min
- **Total: ~2 hours**

**Phase 4** (Assessment/Comparison): 1-2 hours

**Grand total**: ~10-12 hours for complete workflow

---

## Key Design Decisions (Synthesis Rationale)

### Why These 4 Models?

1. **Theoretical coverage**: Span hierarchical (1-2), fixed-effect (3), and subgroup (4) approaches
2. **Complementary assumptions**: Different views on I²=0% interpretation
3. **Robustness progression**: Normal → Student-t → Fixed captures key sensitivities
4. **Implementable**: All have complete Stan code, clear falsification criteria

### Why Not the Other 5 Models?

**Designer #1's Models 2-3**:
- Model 2 (Robust Hierarchical): Redundant with Designer #3's Student-t (same concept)
- Model 3 (Informative Prior): Likely to fail (I²=0% conflicts with typical heterogeneity)

**Designer #2's Model 3**:
- Measurement Error Uncertainty: No evidence of SE miscalibration in EDA
- Student-t addresses same robustness concern more directly

**Designer #3's Models 2-3**:
- Mixture Model: J=8 too small for reliable identification per designer
- Worth revisiting if Models 1-4 all fail

### Core Philosophy

All three designers emphasized **falsificationism**:
- Every model has explicit rejection criteria
- Success = finding what's wrong, not forcing a fit
- Abandoning all models is a valid scientific outcome
- Honest uncertainty quantification > task completion

This plan operationalizes that philosophy with pre-specified decision rules at every stage.

---

## Summary

**Primary models**: 1 (Hierarchical), 2 (Robust), 3 (Fixed-effect)
**Minimum attempts**: Models 1-2 per policy
**Expected outcome**: Model 1 or 2 likely to succeed; Model 3 useful benchmark
**Adequate solution**: At least one model passes all checks with good convergence
**Timeline**: 10-12 hours total

**Next step**: Begin Phase 3 with Model 1 (Prior Predictive Check)

---

**Document prepared**: 2025-10-28
**Designers synthesized**: 3 (Designer #1, #2, #3)
**Total models proposed**: 9
**Models selected for implementation**: 4
**Ready to proceed**: Yes
