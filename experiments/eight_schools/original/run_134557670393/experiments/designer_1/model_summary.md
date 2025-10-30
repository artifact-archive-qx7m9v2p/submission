# Quick Reference: Hierarchical Models for Meta-Analysis

**Designer**: Model Designer #1 (Hierarchical/Multilevel Focus)
**Date**: 2025-10-28
**Dataset**: J=8 studies, I²=0%, borderline effect (p≈0.05)

---

## Three Proposed Model Classes

| Model | Core Assumption | Key Parameter | When to REJECT |
|-------|----------------|---------------|----------------|
| **Model 1: Adaptive Hierarchical** | Studies are exchangeable, Normal hierarchy | tau ~ Half-Cauchy(0,5) | If posterior predictive fails OR leave-one-out instability > 5 units |
| **Model 2: Robust Hierarchical** | Heavy-tailed effects (outliers) | nu ~ Gamma(2,0.1) | If nu posterior > 50 OR no improvement over Model 1 |
| **Model 3: Informative Heterogeneity** | Typical meta-analytic heterogeneity | tau ~ Half-Normal(0,3) | If prior-data conflict (p < 0.05) OR tau hits prior boundary |

---

## Critical Decision Tree

```
START: Fit all 3 models
│
├─ Check Convergence (R-hat, divergences)
│  ├─ All fail → Computational issue, reparameterize
│  └─ Pass → Continue
│
├─ Apply Falsification Criteria (see below)
│  ├─ 2+ models fail → Hierarchical framework wrong
│  │                   → PIVOT to mixture/non-parametric
│  └─ 1+ pass → Continue
│
├─ LOO-CV Comparison
│  ├─ ELPD diff < 4 → Models equivalent, use simplest (Model 1)
│  ├─ ELPD diff 4-10 → Moderate preference
│  └─ ELPD diff > 10 → Strong preference
│
└─ Sensitivity Analysis
   ├─ Conclusion robust → Report selected model
   └─ Conclusion flips → Data too weak, report uncertainty
```

---

## Falsification Triggers (REJECT Model If...)

### Model 1: Adaptive Hierarchical
- [ ] Any theta_i shrinks > 3*sigma_i from y_i
- [ ] Posterior predictive fails (y outside 95% PPI for >1 study)
- [ ] Leave-one-out changes mu by > 5 units
- [ ] Divergences > 1% OR R-hat > 1.05
- [ ] Tau posterior is uniform (unidentified)

### Model 2: Robust Hierarchical
- [ ] Nu posterior > 50 (heavy tails unnecessary)
- [ ] Nu posterior spans entire prior (uninformative)
- [ ] LOO improvement over Model 1 < 2 ELPD
- [ ] ESS for nu < 100 after 10K iterations

### Model 3: Informative Heterogeneity
- [ ] Prior predictive p-value < 0.05 or > 0.95
- [ ] P(tau > 5 | data) > 0.2 with Half-Normal(0,3) prior
- [ ] Posterior SD(tau) < 0.8 * Prior SD(tau)
- [ ] LOO-CV worse than Model 1 by > 4 ELPD

---

## Expected Outcomes (Predictions)

**Most likely**: Model 1 wins
- I²=0% suggests tau ≈ 0, which standard hierarchical can capture
- Study 1 not a true outlier given SE=15

**Possible surprise**: Model 2 wins
- If Study 1 is more problematic than statistics suggest
- If heavy tails improve predictive performance

**Unlikely**: Model 3 wins
- I²=0% conflicts with typical meta-analytic heterogeneity
- Informative prior may fight data

**HOWEVER**: These are predictions, not conclusions. Data decides.

---

## Red Flags for Complete Pivot

If ANY of these occur, abandon hierarchical framework entirely:

1. **All 3 models fail falsification** → Hierarchical assumption wrong
2. **Study 1 dominates leave-one-out** (Δmu > 10) → Dataset too fragile
3. **Posterior predictive fails for 3+ studies** → Likelihood misspecified
4. **Tau posterior is bimodal** → Mixture of populations
5. **Prior sensitivity changes conclusions qualitatively** → Data too weak

**Pivot options**: Mixture models, fixed-effect only, Bayesian model averaging, or report limitations and stop modeling.

---

## Implementation Checklist

### Before Fitting
- [ ] Prior predictive checks (simulate from priors)
- [ ] Parameter recovery simulation (known tau=0 and tau=5)
- [ ] Stan code reviewed (check parameterization)

### During Fitting
- [ ] Monitor convergence (R-hat, ESS, divergences)
- [ ] Use non-centered if funnel geometry detected
- [ ] Increase adapt_delta to 0.95 if needed

### After Fitting
- [ ] Apply all falsification criteria (automated)
- [ ] Posterior predictive checks (visual + quantitative)
- [ ] Leave-one-out CV (all 8 studies)
- [ ] LOO-CV comparison (if multiple models pass)

### Sensitivity Analysis
- [ ] Prior sensitivity (±50% on prior SDs)
- [ ] Likelihood robustness (Normal vs Student-t)
- [ ] Influential study removal (Studies 1 and 5)

---

## Key Implementation Details

### Model 1: Stan Code (Centered)
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[8] theta;
}
model {
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);  // sigma is data
}
```

**If divergences occur, switch to non-centered**:
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[8] theta_raw;
}
transformed parameters {
  vector[8] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 5);
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma);
}
```

### Model 2: Additional Parameter
```stan
real<lower=1> nu;  // DOF for Student-t
...
nu ~ gamma(2, 0.1);
theta ~ student_t(nu, mu, tau);
```

### Model 3: Change Prior Only
```stan
tau ~ normal(0, 3);  // Half-Normal via truncation at 0
```

---

## Success Metrics

**Scientific (Primary)**:
- Honest uncertainty quantification
- Clear limits of what data can/cannot tell us
- Influential studies identified

**Technical (Secondary)**:
- R-hat < 1.01, ESS > 400
- Zero divergences
- Posterior predictive p-value ∈ [0.05, 0.95]

**Philosophical (Meta)**:
- Willingness to reject all models
- Transparency about assumptions
- Clear falsification criteria

---

## Contact Points with Other Designers

**If Designer #2 proposes**:
- **GP models**: Compare LOO-CV; if GP wins, hierarchical framework may be wrong for this data
- **State-space**: Unlikely relevant (no temporal structure in meta-analysis)
- **Mixture models**: If they propose 2-component mixture, this is my "escape route" if hierarchical fails

**If Designer #3 proposes**:
- **Measurement error models**: Relevant if they question sigma_i accuracy
- **Selection models**: Publication bias modeling, could nest within hierarchical
- **Non-parametric**: If they propose Dirichlet process, compare to my parametric models

**Synthesis strategy**: Run all models, compare via LOO-CV, report best performing. If model classes differ substantially (e.g., hierarchical vs mixture), this tells us something important about data structure.

---

## File Locations

- **Full proposal**: `/workspace/experiments/designer_1/proposed_models.md`
- **This summary**: `/workspace/experiments/designer_1/model_summary.md`
- **Stan implementations**: To be created in modeling phase
- **Results**: Will be in `/workspace/experiments/designer_1/results/`

---

**Philosophy**: "All models are wrong, but some are useful. Our job is to discover how ours are wrong before claiming they're useful."
