# Falsification Criteria: Quick Reference
## Model Designer #3 - Robust & Alternative Models

**Purpose**: Fast reference for when to ABANDON each model during analysis
**Philosophy**: Models designed to fail informatively - abandoning is success, not failure

---

## Model 1: Robust Student-t Meta-Analysis

### Abandon if:

1. **nu > 50** (P > 0.8)
   - Meaning: Student-t → Normal, complexity not needed
   - Action: Use standard Normal hierarchical model

2. **PPC fails** (>25% studies outside 95% intervals)
   - Meaning: Heavy tails insufficient, structural problem
   - Action: Try Mixture Model

3. **nu < 1.5** (extreme tails with tight CI)
   - Meaning: Single Student-t inadequate
   - Action: Contamination model or mixture

4. **LOO Pareto k > 0.7** for >2 studies
   - Meaning: Model misspecified
   - Action: Investigate mixture or alternative structure

5. **Poor shrinkage** (Study 1 shrinks <10% or >90%)
   - Meaning: nu not calibrating robustness properly
   - Action: Try alternative robust families

### Red flags during fitting:
- Divergences on nu → reparameterize
- ESS(nu) < 100 → stronger prior needed
- R-hat > 1.01 → non-convergence

---

## Model 2: Finite Mixture Meta-Analysis

### Abandon if:

1. **Degenerate groups** (pi < 0.1 or > 0.9, P > 0.8)
   - Meaning: Data doesn't support two groups
   - Action: Single-population model

2. **Groups not separated** (|mu_2 - mu_1| < 5, P > 0.7)
   - Meaning: Two groups indistinguishable
   - Action: Simpler single-population model

3. **High within-group heterogeneity** (tau_k > 10)
   - Meaning: Groups not homogeneous internally
   - Action: Try meta-regression or standard model

4. **Uncertain assignments** (P(z_i=1) ∈ [0.3,0.7] for >50% studies)
   - Meaning: Can't assign studies to groups reliably
   - Action: J=8 insufficient, use simpler model

5. **Poor LOO** (worse than single-population by >2 SE)
   - Meaning: Overfitting, not improving predictions
   - Action: Use simpler model

6. **PPC fails** (bimodality predicted but not observed)
   - Meaning: Mixture structure wrong
   - Action: Try non-parametric or Student-t

### Red flags during fitting:
- Label switching → stronger ordering constraint
- Multimodal posterior (R-hat > 1.1) → run more chains
- ESS < 100 → marginalize or reparameterize

---

## Model 3: Uncertainty-Inflated Meta-Analysis

### Abandon if:

1. **lambda ≈ 1** (P(0.95 < lambda < 1.05) > 0.7)
   - Meaning: Reported SEs adequate
   - Action: Standard model with fixed sigma_i

2. **Extreme inflation** (lambda > 2.5)
   - Meaning: Model inflating SEs excessively, structural problem
   - Action: Try Student-t or mixture

3. **Strong lambda-tau correlation** (r < -0.8)
   - Meaning: Can't distinguish SE underestimation from heterogeneity
   - Action: Report joint uncertainty, acknowledge non-identifiability

4. **No LOO improvement** (worse than standard by >1 SE)
   - Meaning: SE inflation not helping predictions
   - Action: Standard fixed-sigma model

5. **Prior dominates** (KL divergence < 0.1)
   - Meaning: Data uninformative about lambda
   - Action: Fix lambda=1 or use informative prior

6. **PPC fails** (>25% studies outside intervals even with inflation)
   - Meaning: Not just SE problem, structural issue
   - Action: Try mixture or Student-t

### Red flags during fitting:
- Wide lambda posterior (CI: [0.5, 3]) → weakly identified
- lambda < 0.8 → check data quality
- Divergences → log-scale reparameterization

---

## Cross-Model Decision Tree

```
START
  |
  ├─ All models converge to similar mu?
  |    YES → Report consensus, use simplest model
  |    NO  → Continue to diagnostic...
  |
  ├─ Student-t: nu < 30?
  |    YES → Heavy tails matter, use Student-t
  |    NO  → Not needed
  |
  ├─ Mixture: Clear groups (|mu_2-mu_1| > 10, 0.2 < pi < 0.8)?
  |    YES → Subgroup structure real, use Mixture
  |    NO  → Not supported
  |
  ├─ Inflation: lambda > 1.5?
  |    YES → SEs underestimated, use Inflation
  |    NO  → SEs adequate
  |
  └─ No model fits well (PPC fails)?
       YES → PIVOT to alternative model class
       NO  → Choose best by LOO-CV
```

---

## Global Stopping Rules

### Stop all analyses if:

1. **Computational failure**: No model converges after extensive tuning
   - Action: Declare modeling infeasible, report EDA only

2. **Universal PPC failure**: All models fail posterior predictive checks
   - Action: Fundamental model misspecification, need new model class

3. **Extreme prior sensitivity**: Conclusions flip across reasonable priors
   - Action: Data too weak (J=8), report high uncertainty

4. **Non-identifiability**: All models show parameter posteriors ≈ priors
   - Action: J=8 insufficient for these models, use simpler approach

### Declare success if:

1. At least one model converges well (R-hat < 1.01, ESS > 400)
2. PPC reasonable (>80% observations within 95% intervals)
3. LOO reliable (Pareto k < 0.7 for >75% observations)
4. Conclusions robust to reasonable sensitivities

---

## Priority if Time/Compute Limited

**First priority**: Model 1 (Student-t)
- Most robust, moderate complexity
- Most likely to handle Study 1 influence
- Good balance of flexibility and identifiability

**Second priority**: Model 3 (Uncertainty Inflation)
- Simple to implement
- Quick to run (~2-5 min)
- Good robustness check even if lambda ≈ 1

**Third priority**: Model 2 (Mixture)
- Most complex, slowest (~10-20 min)
- May not be identifiable with J=8
- Only if clear evidence of clustering persists

**Always run**: Standard Normal hierarchical (benchmark)

---

## Quick Diagnostic Checklist

Before trusting ANY model:

- [ ] R-hat < 1.01? (convergence)
- [ ] ESS > 400? (sufficient samples)
- [ ] Divergences < 10? (geometry okay)
- [ ] PPC passes? (>80% in intervals)
- [ ] LOO Pareto k < 0.7? (predictive validity)
- [ ] Posterior differs from prior? (learning from data)
- [ ] Robust to Study 1 removal? (not over-influenced)
- [ ] Robust to reasonable prior changes? (not prior-driven)

If ANY fail → don't trust results, investigate or abandon model

---

## Remember

**Good modeling discovers it was wrong and pivots quickly**

Success = Finding model that genuinely explains data
Failure = Completing analysis with wrong model

**When in doubt, use simpler model**
**When conflicted, report uncertainty honestly**
**When stuck, pivot to alternative approach**

