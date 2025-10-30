# Quick Reference: Alternative Bayesian Models (Designer 2)

## Three Alternative Model Classes

### Model 1: Finite Mixture (K=2 Components)
**Core idea**: Groups come from 2 distinct subpopulations (low-risk ~6%, high-risk ~12%)

**Key parameters**:
- `w`: Mixture weight (proportion in component 1)
- `mu_1, mu_2`: Component means (ordered: mu_1 < mu_2)
- `kappa_1, kappa_2`: Component concentrations

**Falsification**: Abandon if w < 0.1 or > 0.9, or |mu_2 - mu_1| < 0.03

**Best when**: Evidence of bimodality, interest in subgroup membership

---

### Model 2: Robust Hierarchical (Student-t Random Effects)
**Core idea**: Population distribution has heavy tails; outliers are legitimate draws

**Key parameters**:
- `mu`: Population mean (logit scale)
- `tau`: Between-group scale
- `nu`: Degrees of freedom (tail weight)

**Falsification**: Abandon if nu > 30 (normal sufficient) or nu < 2 (too extreme)

**Best when**: Outliers are real, want robust population estimates, nu < 10 suggests heavy tails

---

### Model 3: Dirichlet Process Mixture (Nonparametric)
**Core idea**: Unknown number of clusters; let data determine K

**Key parameters**:
- `alpha_dp`: DP concentration (controls cluster propensity)
- `K_eff`: Effective number of clusters (derived)
- Component-specific means and concentrations

**Falsification**: Abandon if K_eff = 1 consistently, or extreme computational issues

**Best when**: Genuinely uncertain about cluster count, have computational resources

---

## Implementation Priority

1. **Start with Model 2 (Robust Student-t)**: Fastest, easiest, direct test of heavy tails
2. **Then Model 1 (Finite Mixture)**: If evidence supports discrete subpopulations
3. **Finally Model 3 (DP)**: Only if cluster count truly uncertain and resources allow

---

## Key Falsification Criteria (All Models)

**Global stopping rules** - Revert to standard hierarchical if:
- All alternatives fail their specific criteria
- All show worse LOO than baseline
- All have computational difficulties
- Prior predictives are implausible

**Model-specific red flags**:
- Divergent transitions > 1%
- ESS < 400 per chain
- Prior-posterior conflict
- Parameters take extreme values
- LOO worse than simpler baseline

---

## Comparison to Standard Hierarchical

| Feature | Standard Beta-Binomial | Model 1 (Mixture) | Model 2 (Robust) | Model 3 (DP) |
|---------|----------------------|------------------|-----------------|-------------|
| Structure | Continuous, unimodal | Discrete, K=2 | Continuous, heavy-tailed | Nonparametric, flexible K |
| Outlier handling | Aggressive shrinkage | Separate cluster | Less shrinkage | Flexible assignment |
| Complexity | Low | Moderate | Low-Moderate | High |
| Comp. cost | Low | Moderate | Low | High |
| When best | Unimodal, continuous | Bimodal evidence | Heavy tails | Unknown K |

---

## Complete Specifications

See `/workspace/experiments/designer_2/proposed_models.md` for:
- Full mathematical specifications
- PyMC implementation code
- Prior justification and sensitivity analysis
- Detailed falsification criteria
- Posterior predictive check protocols
- Model comparison strategies

---

**Quick Start**: Implement Model 2 (Robust Student-t) first as it's the most tractable alternative that addresses the key concern (outliers) with minimal added complexity.
