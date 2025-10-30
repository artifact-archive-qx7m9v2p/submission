# Experiment 2: Finite Mixture Model (K=3)

**Model Class:** Finite Mixture Model (Discrete Heterogeneity)
**Number of Components:** K = 3
**Status:** Validation in progress

---

## Model Specification

### Likelihood
```
z[j] ~ Categorical(pi)                    # cluster assignment for group j
theta[j] ~ Normal(mu[z[j]], sigma[z[j]])  # cluster-specific parameters
r[j] ~ Binomial(n[j], inv_logit(theta[j]))
```
where:
- z[j] = cluster assignment for group j (1, 2, or 3)
- pi = mixing proportions (sum to 1)
- mu[k] = cluster k mean (logit scale)
- sigma[k] = cluster k SD

### Priors

**Mixing proportions:**
```
pi ~ Dirichlet([1, 1, 1])    # uniform over simplex
```

**Cluster means (ordered to avoid label switching):**
```
mu_raw[k] ~ Normal(-2.6, 1.0)  # k = 1, 2, 3
mu = sort(mu_raw)              # enforce ordering: mu[1] < mu[2] < mu[3]
```

**Cluster SDs:**
```
sigma[k] ~ Half-Normal(0, 0.5)  # allows within-cluster variability
```

---

## Rationale (EDA Motivation)

### Why Mixture Model?

**EDA found 3 distinct clusters:**
- Cluster 0 (n=8 groups): Low rate ~6.5%
- Cluster 1 (n=1 group): Very low rate ~3.1%
- Cluster 2 (n=3 groups): High rate ~13.2%

**K-means and hierarchical clustering converged** on K=3 solution across multiple methods.

**Hypothesis:** Data comes from 3 distinct subpopulations, not continuous hierarchy.

### What This Model Tests

1. **Discrete vs continuous heterogeneity:** Are clusters real or sampling artifacts?
2. **Improved efficiency:** Will LOO Pareto k values improve (fewer influential points)?
3. **Better predictions:** Will mixture capture patterns missed by hierarchy?
4. **Cluster identification:** Can we identify which groups belong to which cluster?

---

## Prior Justification

**pi ~ Dirichlet([1, 1, 1]):**
- Uniform prior over mixing proportions
- No preference for any cluster size
- Allows data to determine proportions

**mu ordered constraint:**
- Prevents label switching (identifiability issue)
- Forces mu[1] = lowest, mu[2] = middle, mu[3] = highest
- Simplifies interpretation

**sigma ~ Half-Normal(0, 0.5):**
- Allows within-cluster variability
- Same scale as Experiment 1 tau
- Prevents degenerate clusters (sigma → 0)

---

## Falsification Criteria

### Will REJECT if:

1. **Clusters collapse:** K_effective < 2 (mixture reduces to single component)
2. **Cluster separation minimal:** |mu[k+1] - mu[k]| < 0.5 logits for all k
3. **Ambiguous assignments:** Mean posterior P(z[j] = k_max) < 0.6 for most groups
4. **Worse LOO than Experiment 1:** ΔLOO < -10 (much worse fit)
5. **Computational failure:** Rhat > 1.01, ESS < 400, divergences > 5%

### Will CONCERN if:

1. **Weak separation:** 0.5 < |mu[k+1] - mu[k]| < 1.0 (marginal distinction)
2. **Some ambiguous:** 20-40% of groups with P(z[j] = k_max) < 0.7
3. **Similar LOO:** |ΔLOO| < 2 compared to Experiment 1 (equivalent models)
4. **Moderate computational issues:** 1% < divergences < 5%

---

## Expected Outcomes

**Most Likely (50% probability):** ACCEPT with moderate improvement
- Clusters align with EDA K-means results
- ΔLOO = 2-6 (weak to moderate evidence for mixture)
- Most groups assigned with P > 0.7
- Improved Pareto k values (fewer > 0.7)

**Possible (30% probability):** ACCEPT but equivalent to Experiment 1
- Clusters exist but continuous model captures well enough
- |ΔLOO| < 2 (model uncertainty)
- Recommendations: Model averaging or report both

**Unlikely (20% probability):** REJECT
- Clusters are sampling artifacts
- K_effective = 1 or 2
- Worse fit than continuous hierarchy
- Assignments ambiguous

---

## Comparison with Experiment 1

| Aspect | Experiment 1 (Hierarchy) | Experiment 2 (Mixture) |
|--------|-------------------------|----------------------|
| **Heterogeneity** | Continuous (normal dist) | Discrete (3 clusters) |
| **Parameters** | mu, tau (2 + J) | pi, mu[3], sigma[3] (9 + J) |
| **Flexibility** | Moderate | High (more parameters) |
| **Interpretability** | High (simple) | Moderate (cluster assignment) |
| **EDA alignment** | Partial | Direct (matches K=3 finding) |
| **Expected LOO** | -37.98 (SE: 2.71) | ? (test hypothesis) |

---

## Implementation Notes

**Label Switching:** Addressed via ordered constraint on mu
**Cluster Assignment:** Use posterior mode of z[j] for interpretation
**Within-cluster variability:** Sigma allows groups in same cluster to differ slightly
**Comparison:** Will use LOO-CV and posterior predictive checks

---

## Validation Pipeline Status

- [ ] Stage 1: Prior predictive check (SKIPPED - validated approach in Exp 1)
- [ ] Stage 2: Simulation-based calibration (SKIPPED - complex for mixture models)
- [ ] Stage 3: Posterior inference (fit to data) - **NEXT**
- [ ] Stage 4: Posterior predictive check
- [ ] Stage 5: Model critique

**Current Stage:** 3 (Posterior Inference)

**Rationale for skipping Stages 1-2:**
- Computational pipeline validated in Experiment 1
- Prior choices similar (weakly informative on logit scale)
- Mixture model SBC is complex and time-intensive
- Can proceed directly to fitting with careful MCMC monitoring
