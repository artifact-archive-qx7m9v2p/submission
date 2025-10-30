# Executive Summary: Practical Modeling Strategy
## Designer #3 - Quick Reference Guide

**Date**: 2025-10-30
**Dataset**: 12 groups, 2,814 trials, 196 successes, strong overdispersion (φ=3.59)

---

## Bottom Line Recommendation

### START HERE: Hierarchical Binomial (Non-Centered)
- **Why**: Best balance of simplicity, interpretability, and correctness
- **Time**: 1-2 hours total including diagnostics
- **Confidence**: 90% this will work perfectly
- **Expected result**: All diagnostics pass, LOO validates, stakeholders understand

---

## Three-Model Strategy

| Model | Use When | Time | Complexity | Confidence |
|-------|----------|------|------------|------------|
| **1. Hierarchical** | Need group estimates (usual case) | 1.5 hr | Medium | ⭐⭐⭐⭐⭐ |
| **2. Beta-binomial** | Only need population mean | 1 hr | Low | ⭐⭐⭐ |
| **3. Robust (t-dist)** | Model 1 fails diagnostics | 2 hr | High | ⭐⭐ |

---

## Decision Flow (30-Second Version)

```
Do you need estimates for individual groups?
  ├─ NO  → Use Model 2 (Beta-binomial)
  └─ YES → Use Model 1 (Hierarchical)
            ├─ Diagnostics pass? → DONE ✓
            └─ Diagnostics fail? → Try Model 3
```

---

## Model 1: Hierarchical (RECOMMENDED)

### Stan Code (Copy-Paste Ready)
```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;
}
transformed parameters {
  vector[J] theta = mu + tau * eta;
  vector[J] p = inv_logit(theta);
}
model {
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  eta ~ std_normal();
  r ~ binomial_logit(n, theta);
}
generated quantities {
  array[J] int r_rep;
  vector[J] log_lik;
  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }
}
```

### Expected Results
- Sampling time: <1 minute
- Rhat: <1.01 for all parameters
- ESS: >400 for μ, τ
- Divergences: <1%
- Pareto k: All <0.7

### Pass/Fail Criteria
**PASS if**:
- ✅ All diagnostics in expected range
- ✅ Posterior predictive variance ≈ 3.6× binomial
- ✅ Shrinkage logical (small groups more than large)

**FAIL if**:
- ❌ Divergences >5%
- ❌ Multiple Pareto k >0.7
- ❌ Posterior predictive misses observed variance

---

## Model 2: Beta-Binomial (SIMPLE ALTERNATIVE)

### When to Use
- Only care about: "What's the average success rate?"
- Don't need: "Which group is best?"
- Want: Simplest defensible model

### Stan Code
```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}
parameters {
  real<lower=0, upper=1> mu;
  real<lower=0> phi;
}
transformed parameters {
  real alpha = mu * phi;
  real beta = (1 - mu) * phi;
}
model {
  mu ~ beta(5, 50);
  phi ~ gamma(1, 0.1);
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}
```

### Pros/Cons
**Pros**:
- 10x faster
- 2 parameters vs 14
- Easy to explain

**Cons**:
- No group-level inference
- Weaker predictions
- Can't identify outliers

---

## Model 3: Robust Hierarchical (ESCALATION ONLY)

### When to Use
**Only escalate if**:
- Model 1 has Pareto k >0.7 for multiple groups
- Posterior predictive checks fail
- Stakeholders worried about outlier influence

### Key Change
Replace `eta ~ std_normal()` with:
```stan
real<lower=0> nu;
...
nu ~ gamma(2, 0.1);
eta ~ student_t(nu, 0, 1);
```

### Interpretation
- If posterior ν >10: No evidence for heavy tails, use Model 1
- If posterior ν <5: Heavy tails justified, keep Model 3

---

## Practical Checklist

### Before Starting
- [ ] Data loaded: 12 groups, 2814 trials, 196 successes
- [ ] Confirm overdispersion: χ² test p<0.001
- [ ] Identify outliers: Groups 2, 4, 8

### Fitting Model 1
- [ ] Use non-centered parameterization (critical!)
- [ ] Priors: μ ~ N(-2.5, 1), τ ~ Half-Cauchy(0,1)
- [ ] Run: 4 chains, 2000 iterations, 1000 warmup
- [ ] Should finish in <1 minute

### Diagnostics
- [ ] Rhat <1.01 for all parameters
- [ ] ESS >400 for μ, τ
- [ ] Divergences <1%
- [ ] Pairs plot: No funnel or banana shapes

### Validation
- [ ] Compute LOO, check Pareto k <0.7
- [ ] Posterior predictive: variance ratio ≈ 3.6
- [ ] Shrinkage plot: Small groups shrink more
- [ ] Compare to unpooled/pooled: Should outperform

### If Problems Arise
- [ ] Divergences 1-5%? Try adapt_delta=0.95
- [ ] Pareto k 0.5-0.7? Try Model 3
- [ ] τ near zero? Check if pooling justified
- [ ] Implausible parameters? Check data quality

---

## Expected Parameter Estimates

Based on EDA, Model 1 should yield:

| Parameter | Expected Range | Interpretation |
|-----------|----------------|----------------|
| μ | -2.5 to -2.2 | Pooled rate 7-9% |
| τ | 0.3 to 0.6 | Moderate heterogeneity |
| p[1] | 8-11% | Group 1 (was 12.8%, shrinks 30%) |
| p[4] | 4-5% | Group 4 (was 4.2%, shrinks 15%) |
| p[8] | 11-13% | Group 8 (was 14.0%, shrinks 20%) |
| p[10] | 5-7% | Group 10 (was 3.1%, shrinks 70%) |

---

## Red Flags: Stop and Reassess

**Computational Issues**:
- Sampling takes >5 minutes
- Divergences persist despite adapt_delta=0.99
- Rhat >1.05 after 4000 iterations

**Statistical Issues**:
- Multiple Pareto k >0.7
- Posterior predictive p-value <0.01 or >0.99
- τ >2 or any p[j] >0.3

**Practical Issues**:
- Results change >20% with different priors
- Stakeholders can't understand outputs
- Conclusions inconsistent across runs

---

## Green Flags: You're Done!

**Stop Iterating if**:
- All diagnostics pass
- ΔLOO clearly favors one model (>10)
- Posterior predictive checks pass
- Results stable across runs
- Stakeholders understand findings

**Remember**: "Good enough" is good enough!

---

## Time Budget

**Realistic Timeline** (one analyst, first time):
- Setup + data prep: 30 min
- Model 1 fitting: 1 min
- Diagnostics: 20 min
- Visualization: 30 min
- Model 2 fitting (for comparison): 1 min
- LOO comparison: 10 min
- Write-up: 1.5 hours

**Total**: 3-4 hours for complete analysis

**If Experienced**:
- Can be done in 1-2 hours

---

## Communication Strategy

### For Statisticians
> "Hierarchical binomial model with non-centered parameterization and weakly informative priors. Partial pooling on logit scale with between-group SD estimated from data."

### For Domain Experts
> "Each group has its own success rate, but we learn from all groups together. Small groups borrow strength from large groups. The model automatically decides how much to trust each group's data."

### For Executives
> "We found success rates vary 3-14% across groups. We can't explain why, but we can estimate each group's true rate accounting for sample size differences."

### Key Outputs
1. **Table**: Group rates with 95% credible intervals
2. **Plot**: Shrinkage arrows showing raw vs adjusted rates
3. **Number**: Population mean ± uncertainty
4. **Prediction**: Expected rate for new group

---

## FAQs

**Q: Why non-centered parameterization?**
A: 2-10x faster convergence. Always use it for hierarchical models.

**Q: Why not just use frequentist mixed model?**
A: Bayesian handles small groups better, gives direct probability statements, easy to add complexity later.

**Q: Do I need to try all three models?**
A: No. Start with Model 1. Only try others if it fails or you want validation.

**Q: What if LOO says models are tied (ΔLOO<4)?**
A: Choose simpler model. Occam's razor applies.

**Q: How do I know if priors are too strong?**
A: Refit with 2x wider priors. If estimates change <10%, you're fine.

**Q: What's the most common mistake?**
A: Centered parameterization. Always use non-centered for hierarchical models.

**Q: When should I give up?**
A: If Model 3 fails diagnostics after tuning, consult domain expert. May be data quality issue.

---

## Summary

**For this dataset**:
- Use Model 1 (Hierarchical)
- It will work
- Spend time on interpretation, not model tweaking

**Success = Answering the scientific question**, not perfect diagnostics.

---

**Full Details**: See `/workspace/experiments/designer_3/proposed_models.md`
