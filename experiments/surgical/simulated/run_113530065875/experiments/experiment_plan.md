# Bayesian Modeling Experiment Plan

**Date**: 2024
**Data**: Binomial data, 12 groups, overdispersion φ=3.59, ICC=0.56

---

## Overview

Three parallel designers independently proposed Bayesian model classes. This document synthesizes their proposals, removes duplicates, and prioritizes models for sequential implementation.

**Convergent Recommendations** (all 3 designers agree):
1. **Hierarchical binomial** (logit-normal or Student-t) is the primary candidate
2. **Beta-binomial** is a strong simpler alternative
3. **Pooled/unpooled** are baselines for comparison (expected to fail/overfit)
4. **Non-centered parameterization** should be default for hierarchical models

**Divergent Proposals**:
- Designer 2 suggested **finite mixture** (2 subpopulations) - HIGH RISK, likely to fail with J=12
- All designers agree mixture is exploratory and should only be attempted if standard hierarchical fails

---

## Experiment Queue (Prioritized)

### **Experiment 1: Hierarchical Binomial (Non-Centered Logit-Normal)** ⭐ PRIMARY
**Status**: REQUIRED (attempt first)
**Confidence**: 90% this will work (Designer 3 estimate)
**Expected outcome**: ACCEPT

**Model Specification**:
```stan
data {
  int<lower=1> J;              // Number of groups (12)
  int<lower=0> n[J];           // Trials per group
  int<lower=0> r[J];           // Successes per group
}

parameters {
  real mu;                      // Population mean (logit scale)
  real<lower=0> tau;            // Between-group SD
  vector[J] theta_raw;          // Non-centered
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}

model {
  mu ~ normal(-2.5, 1);         // Weakly informative
  tau ~ cauchy(0, 1);           // Half-Cauchy (lower=0 in parameters)
  theta_raw ~ normal(0, 1);     // Non-centered
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }
}
```

**Priors Justification** (from EDA):
- `mu ~ N(-2.5, 1)`: Centers at ~7-8% success rate, weak regularization
- `tau ~ Half-Cauchy(0, 1)`: Standard weakly informative for hierarchical SD
- Non-centered: Better geometry when J small and tau uncertain

**Falsification Criteria**:
1. **Prior predictive**: Allow rates [3%, 14%] and φ ≈ 3.6
2. **SBC**: 90-95% coverage, uniform rank statistics (100 simulations)
3. **Posterior predictive**:
   - φ_obs in 95% posterior predictive interval
   - Groups 2, 4, 8 have |z| < 3 in posterior predictive
   - Shrinkage: small-n (60-72%), large-n (19-30%)
4. **LOO**: Pareto k < 0.7 for all groups, ΔLOO > 10 vs pooled
5. **Convergence**: R̂ < 1.01, ESS > 400, divergences < 1%

**Decision Paths**:
- ✅ **All checks pass** → ACCEPT, proceed to Phase 4 (Assessment)
- ⚠️ **Convergence issues** → Try centered parameterization or increase adapt_delta
- ⚠️ **Outliers mispredicted** → Try Experiment 2 (Robust Student-t)
- ❌ **Fundamental failure** → Document and try Experiment 3 (Beta-binomial)

**Expected Results**:
- μ ≈ -2.4 (pooled rate ~8%)
- τ ≈ 0.4 (moderate heterogeneity)
- Sampling time: <1 minute
- All diagnostics pass

---

### **Experiment 2: Hierarchical Robust (Student-t Hyperprior)** 🛡️ ROBUSTNESS CHECK
**Status**: CONDITIONAL (only if Experiment 1 has outlier issues)
**Confidence**: 70% this will improve over Experiment 1 if outliers problematic
**Expected outcome**: ACCEPT if ν < 10, otherwise equivalent to Experiment 1

**Model Specification**:
```stan
// Same as Experiment 1, but replace:
// theta ~ normal(mu, tau);
// with:
// theta ~ student_t(nu, mu, tau);

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=2> nu;             // Degrees of freedom (lower=2 for identifiability)
  vector[J] theta_raw;
}

model {
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  nu ~ gamma(2, 0.1);           // Weakly informative, mass at 2-30

  theta ~ student_t(nu, mu, tau);  // Robust hyperprior
  r ~ binomial_logit(n, theta);
}
```

**When to Use**:
- Experiment 1 systematically mispredicts groups 2, 4, 8 (|z| > 3)
- Posterior predictive checks show tail misfit
- Groups 2, 4, 8 are genuine outliers, not just sampling variation

**Falsification Criteria**:
- **Robust superior**: ν < 10 in posterior (heavier tails needed)
- **LOO improvement**: ΔLOO > 2 vs Experiment 1
- **Outlier handling**: Groups 2, 4, 8 now have |z| < 2 in posterior predictive
- **Abandon if**: ν > 25 (Normal sufficient, use Experiment 1), no LOO gain

**Expected Results**:
- ν ≈ 5-15 (heavier tails than normal)
- Otherwise similar to Experiment 1
- Sampling time: 2-3× slower than Experiment 1

---

### **Experiment 3: Beta-Binomial** 🔄 SIMPLER ALTERNATIVE
**Status**: CONDITIONAL (attempt if hierarchical has computational issues OR for comparison)
**Confidence**: 60% this will be competitive with Experiment 1
**Expected outcome**: GOOD (may tie with Experiment 1 on LOO)

**Model Specification**:
```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  real<lower=0, upper=1> mu_p;  // Mean success rate
  real<lower=0> kappa;          // Concentration
}

transformed parameters {
  real<lower=0> alpha = mu_p * kappa;
  real<lower=0> beta = (1 - mu_p) * kappa;
}

model {
  mu_p ~ beta(5, 50);           // Weakly informative, centered at ~0.09
  kappa ~ gamma(2, 0.1);        // Allows wide range of overdispersion

  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}

generated quantities {
  vector[J] log_lik;
  real phi;                     // Overdispersion parameter

  phi = 1 / (kappa + 1);        // φ = variance inflation factor

  for (j in 1:J) {
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha, beta);
  }
}
```

**When to Use**:
- Experiment 1 has computational issues (divergences, long sampling)
- Only population-level inference needed (not group-specific)
- Want simpler, more interpretable model

**Falsification Criteria**:
- **Captures overdispersion**: φ ≈ 3.6 in posterior
- **LOO competitive**: |ΔLOO| < 4 vs Experiment 1
- **Boundary issues**: mu_p not at 0 or 1 (computational failure)
- **Abandon if**: φ << 3.6 (cannot capture heterogeneity), LOO worse by > 10

**Advantages**:
- 2× faster sampling than hierarchical
- Direct probability scale (easier interpretation)
- No logit transformation
- Natural conjugate structure

**Disadvantages**:
- No group-specific estimates
- Cannot assess shrinkage patterns
- May underfit if heterogeneity complex

**Expected Results**:
- mu_p ≈ 0.07 (7% pooled rate)
- phi ≈ 3.0-4.0 (captures overdispersion)
- kappa ≈ 10-30
- Sampling time: 10-20 seconds

---

### **Experiment 4: Pooled (Baseline)** 📊 COMPARISON ONLY
**Status**: REQUIRED (for LOO comparison baseline)
**Confidence**: 0% this is adequate (already rejected by EDA χ²=39.47, p<0.0001)
**Expected outcome**: REJECT (but needed for ΔLOO quantification)

**Model Specification**:
```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  real<lower=0, upper=1> p;    // Single success rate
}

model {
  p ~ beta(5, 50);
  r ~ binomial(n, p);
}

generated quantities {
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p);
  }
}
```

**Purpose**: Document inadequacy, quantify ΔLOO improvement of hierarchical models

**Expected Failure Modes**:
- Posterior predictive checks fail (φ observed >> φ predicted)
- LOO worse than Experiment 1 by ΔLOO > 10
- Systematically mispredicts all extreme groups

**Do NOT iterate**: This model is already rejected. Fit once for comparison only.

---

### **Experiment 5: Unpooled (Baseline)** 📊 COMPARISON ONLY
**Status**: OPTIONAL (for LOO comparison if time permits)
**Confidence**: 10% this is adequate (expected to overfit)
**Expected outcome**: REJECT (overfitting, poor LOO despite good fit)

**Model Specification**:
```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  vector<lower=0, upper=1>[J] p;  // Independent rate per group
}

model {
  p ~ beta(5, 50);
  for (j in 1:J) {
    r[j] ~ binomial(n[j], p[j]);
  }
}

generated quantities {
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
  }
}
```

**Purpose**: Demonstrate need for partial pooling, show hierarchical regularization benefit

**Expected Failure Modes**:
- Good in-sample fit but poor LOO (overfitting)
- Wide posteriors for small-sample groups (1, 10)
- No information sharing, cannot generalize to new groups

**Do NOT iterate**: Fit once for comparison only, then discard.

---

### **Experiment 6: Finite Mixture (2 Subpopulations)** 🧪 EXPLORATORY
**Status**: OPTIONAL (only if all standard models fail)
**Confidence**: 10% this will work (J=12 likely too small for mixture)
**Expected outcome**: FAIL (component collapse or computational issues)

**Model Specification**:
```stan
data {
  int<lower=1> J;
  int<lower=0> n[J];
  int<lower=0> r[J];
}

parameters {
  real<lower=0, upper=1> pi;      // Mixing proportion
  real mu1;                        // Low-rate subpopulation mean
  real mu2;                        // High-rate subpopulation mean
  real<lower=0> tau1;
  real<lower=0> tau2;
  simplex[2] lambda[J];           // Group membership probabilities
}

model {
  pi ~ beta(2, 2);
  mu1 ~ normal(-3, 1);            // Prior: low-rate component
  mu2 ~ normal(-2, 1);            // Prior: high-rate component
  tau1 ~ cauchy(0, 0.5);
  tau2 ~ cauchy(0, 0.5);

  for (j in 1:J) {
    vector[2] lp;
    lp[1] = log(pi) + normal_lpdf(logit(r[j] / n[j]) | mu1, tau1);
    lp[2] = log(1 - pi) + normal_lpdf(logit(r[j] / n[j]) | mu2, tau2);
    target += log_sum_exp(lp);
  }
}
```

**When to Use** (all must be true):
- Experiments 1-3 all fail posterior predictive checks
- Strong evidence for bimodal distribution of group rates
- Clear separation between "low-rate" (Groups 4, 5, 6, 7, 10) and "high-rate" (Groups 1, 2, 8) groups

**Falsification Criteria**:
- **Components distinct**: |mu1 - mu2| > 0.5 on logit scale
- **Clear membership**: Most groups have lambda[j,k] > 0.8 for one component
- **LOO improvement**: ΔLOO > 4 vs Experiment 2
- **Abandon if**: Components collapse (|mu1 - mu2| < 0.2), uncertain membership (all lambda ≈ 0.5), computational failure

**Expected Failure Modes** (highly likely):
- Component collapse (mu1 ≈ mu2)
- Label switching issues
- Uncertain assignments (cannot distinguish subpopulations)
- Long sampling time (> 10 minutes)

**WARNING**: This is a high-risk exploratory model. Only attempt if standard hierarchical models demonstrate fundamental inadequacy.

---

## Implementation Order (Minimum Attempt Policy)

### **Required Experiments** (must attempt at least these):
1. **Experiment 1** (Hierarchical logit-normal) - PRIMARY CANDIDATE
2. **Experiment 3** (Beta-binomial) OR **Experiment 2** (Robust) - ALTERNATIVE

**Rationale**: Per workflow guidelines, attempt at least the first two models unless Model 1 fails pre-fit validation (prior predictive or SBC).

### **Conditional Experiments**:
- **Experiment 2**: Only if Experiment 1 has outlier issues
- **Experiment 4**: Required for LOO comparison baseline
- **Experiment 5**: Optional, for completeness
- **Experiment 6**: Only if Experiments 1-3 all fail

### **Stopping Rule**:
- If Experiment 1 passes all falsification checks → ACCEPT, proceed to Phase 4
- If Experiment 1 + refinement pass → ACCEPT
- If 2+ models pass → Proceed to Phase 4 with all passing models
- If all models fail → Phase 5 (Adequacy Assessment) to determine next steps

---

## Model Comparison Strategy (Phase 4)

**Stage 1: Individual Assessment**
- Each model evaluated independently against falsification criteria
- Models classified as ACCEPT, REVISE, or REJECT

**Stage 2: LOO Comparison** (if 2+ models ACCEPT)
```python
import arviz as az

# Compare all accepted models
comparison = az.compare({
    'hierarchical': idata_exp1,
    'beta_binomial': idata_exp3,
    'robust': idata_exp2
})

# Decision rules:
# - ΔLOO > 4: Clear winner
# - |ΔLOO| < 4: Equivalent, choose simpler (beta-binomial > robust > hierarchical in complexity)
# - Pareto k > 0.7: Investigate influential observations
```

**Stage 3: Parsimony Rule**
- If |ΔLOO| < 2×SE between models → Choose simpler model
- Simplicity ranking: Beta-binomial < Hierarchical logit-normal < Robust Student-t < Mixture

**Stage 4: Scientific Criteria**
- Interpretability (can we explain to non-statisticians?)
- Generalizability (can we predict new groups?)
- Computational efficiency (production use feasible?)

---

## Red Flags (Trigger Strategy Pivot)

### **Computational Red Flags**:
- **Divergences > 5%** after tuning → Try centered parameterization or Beta-binomial
- **Sampling time > 10 minutes** → Switch to Beta-binomial
- **R̂ > 1.05** persistently → Fundamental sampling issue, try different model class

### **Statistical Red Flags**:
- **Multiple Pareto k > 0.7** → Influential observations, investigate Group 4 dominance
- **Posterior-prior conflict** (tau > 1.5) → Check for data errors or consider mixture
- **All models fail PPC** → Hierarchical structure wrong, need different approach

### **Scientific Red Flags**:
- **Implausible parameters** (any p > 0.3, tau > 2) → Check data quality
- **Shrinkage patterns wrong** (large-n groups shrink more than small-n) → Model misspecification

**Pivot Actions**:
1. **If computational issues persist** → Accept Beta-binomial (simpler but adequate)
2. **If all hierarchical fail** → Consider non-parametric (Dirichlet process) or structured models
3. **If data quality issues** → STOP, report limitations, recommend data collection improvements

---

## Green Flags (Success Indicators)

### **Experiment 1 Success** (expect this outcome):
- ✅ All convergence diagnostics pass (R̂ < 1.01, ESS > 400, divergences < 1%)
- ✅ LOO clearly favors hierarchical (ΔLOO > 10 vs pooled, Pareto k < 0.7)
- ✅ PPC captures observed overdispersion (φ ≈ 3.6)
- ✅ Shrinkage follows expected pattern (small-n → more, large-n → less)
- ✅ Parameters interpretable (mu ≈ -2.4, tau ≈ 0.4)

→ **ACCEPT** and proceed to Phase 4 (Assessment) → Phase 6 (Reporting)

### **Multiple Models Succeed**:
- Compare via LOO, apply parsimony rule
- Report all adequate models with comparison
- Recommend simplest adequate model

---

## Timeline Estimates

| Experiment | Prior Pred | SBC | Fit | Post Pred | LOO | Total |
|------------|-----------|-----|-----|-----------|-----|-------|
| Exp 1 | 10 min | 1 hr | 1 min | 15 min | 2 min | **1.5 hr** |
| Exp 2 | 10 min | 1 hr | 3 min | 15 min | 2 min | **1.5 hr** |
| Exp 3 | 10 min | 30 min | 20 sec | 10 min | 1 min | **0.9 hr** |
| Exp 4 | 5 min | Skip | 10 sec | 5 min | 1 min | **0.2 hr** |
| Exp 5 | 5 min | Skip | 10 sec | 5 min | 1 min | **0.2 hr** |
| Exp 6 | 15 min | 2 hr | 10 min | 20 min | 5 min | **3.7 hr** |

**Expected Total** (Experiments 1, 3, 4): ~2.6 hours
**Maximum Total** (all experiments): ~8.0 hours

---

## Success Criteria for Phase 2 Completion

Phase 2 (Model Design) is complete when:
- [X] Multiple designers proposed models independently
- [X] Proposals synthesized and duplicates removed
- [X] Models prioritized by theoretical justification and practical considerations
- [X] Falsification criteria defined for each model
- [X] Implementation order established with stopping rules

**Status**: ✅ Phase 2 COMPLETE

**Next Step**: Proceed to Phase 3 (Model Development) starting with Experiment 1 (Hierarchical Binomial)

---

## References to Detailed Specifications

- **Designer 1 (Standard)**: `/workspace/experiments/designer_1/proposed_models.md` (37KB, 5 models)
- **Designer 2 (Alternatives)**: `/workspace/experiments/designer_2/proposed_models.md` (28KB, 4 Stan files)
- **Designer 3 (Practical)**: `/workspace/experiments/designer_3/` (124KB package, decision trees)

---

## Philosophy

This plan follows the Bayesian modeling guidelines:
- **Falsification-first**: Each model has explicit rejection criteria
- **Multiple hypotheses**: Not just variations of one approach
- **Iteration strategy**: Clear decision paths for refinement
- **Stopping rules**: Know when to accept, iterate, or pivot
- **Minimum attempts**: At least 2 models unless pre-fit validation fails

**Key Principle**: "Success is finding truth, not completing a predetermined plan."
