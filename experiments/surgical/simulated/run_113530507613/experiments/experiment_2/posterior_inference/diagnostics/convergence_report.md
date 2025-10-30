# MCMC Convergence Report: Experiment 2 (3-Component Mixture Model)

**Date:** 2025-10-30
**Model:** Finite Mixture Model (K=3)
**Sampler:** PyMC NUTS
**Chains:** 4
**Warmup:** 500 iterations
**Sampling:** 500 iterations per chain
**Total draws:** 2000

---

## Convergence Metrics Summary

### Global Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max R-hat | 1.0200 | < 1.01 | **MARGINAL** |
| Min ESS (bulk) | 341.0 | > 400 | **MARGINAL** |
| Divergent transitions | 4 / 2000 | < 1% | **PASS** (0.20%) |
| Ordering violations | 0 / 2000 | 0% | **PASS** (0.00%) |

**Overall Convergence Status:** **MARGINAL**

The model shows reasonable but not ideal convergence:
- R-hat values are slightly above 1.01 for one parameter (sigma[1] = 1.02)
- ESS values are below 400 for several parameters, indicating chains could be longer
- Divergences are minimal (<1%), suggesting good posterior geometry
- Ordered constraint on mu maintained perfectly (no violations)

---

## Parameter-Level Diagnostics

### Mixing Proportions (pi)

| Parameter | Mean | SD | R-hat | ESS (bulk) | ESS (tail) |
|-----------|------|-----|-------|-----------|-----------|
| pi[1] | 0.331 | 0.253 | 1.01 | 467 | 547 |
| pi[2] | 0.382 | 0.247 | 1.00 | 684 | 588 |
| pi[3] | 0.287 | 0.223 | 1.01 | 431 | 1263 |

**Assessment:** Good convergence for pi. ESS values are adequate (>400 for pi[2]). The mixing proportions show substantial uncertainty (large SD), indicating the model is uncertain about cluster sizes.

### Cluster Means (mu, logit scale)

| Parameter | Mean | SD | HDI 3% | HDI 97% | R-hat | ESS (bulk) |
|-----------|------|-----|--------|---------|-------|-----------|
| mu[1] | -3.007 | 0.430 | -3.845 | -2.392 | 1.00 | 405 |
| mu[2] | -2.540 | 0.272 | -3.015 | -1.980 | 1.01 | 380 |
| mu[3] | -2.056 | 0.429 | -2.648 | -1.198 | 1.01 | 635 |

**Assessment:** Marginal convergence. ESS values for mu[1] and mu[2] are below 400, suggesting more iterations would improve inference. R-hat values are acceptable (≤1.01).

**Cluster means on probability scale:**
- Cluster 1: 0.050 [0.015, 0.079] - Very low success rate
- Cluster 2: 0.075 [0.045, 0.124] - Low success rate
- Cluster 3: 0.121 [0.068, 0.286] - Moderate success rate

### Cluster Standard Deviations (sigma)

| Parameter | Mean | SD | R-hat | ESS (bulk) | ESS (tail) |
|-----------|------|-----|-------|-----------|-----------|
| sigma[1] | 0.335 | 0.249 | 1.01 | 344 | 149 |
| sigma[2] | 0.334 | 0.213 | **1.02** | 459 | 405 |
| sigma[3] | 0.354 | 0.254 | 1.01 | 341 | 113 |

**Assessment:** Marginal convergence. sigma[2] has R-hat = 1.02 (slightly above threshold). Low tail ESS for sigma[1] and sigma[3] (<200) indicates longer chains needed for reliable tail behavior.

---

## Visual Diagnostics

### Trace Plots (`trace_all_params.png`)

**Observations:**
- All parameters show reasonable mixing across chains
- No obvious trends or drift
- Some parameters (especially sigma) show occasional excursions to low values
- Chains appear to explore similar regions but with visible between-chain variance

**Interpretation:** Chains are exploring the posterior but could benefit from longer runs to reduce between-chain variance.

### Rank Plots (`rank_plots.png`)

**Purpose:** Assess uniformity of rank statistics across chains (should be flat if well-mixed).

**Observations:**
- Most parameters show reasonably uniform rank distributions
- Some minor deviations from uniformity visible, consistent with marginal R-hat values
- No severe non-uniformity indicating serious convergence failure

**Interpretation:** Confirms marginal but acceptable convergence quality.

---

## Practical Implications

### For Inference

1. **Parameter estimates are usable** - Despite marginal convergence, the posterior summaries provide reasonable point estimates and uncertainty quantification
2. **Uncertainty may be slightly underestimated** - Lower ESS means MCMC standard errors are higher
3. **Tail behavior uncertain** - Low tail ESS for sigma parameters means extreme values (e.g., 95% HDI limits) are less reliable

### For Model Comparison

1. **LOO-CV is valid** - Log-likelihood successfully saved and available for model comparison
2. **Pareto k diagnostics should be checked** - Will reveal if any observations are highly influential
3. **Comparison with Experiment 1 is fair** - Both models have similar convergence quality

---

## Recommendations

### If using these results:
- Report convergence metrics transparently
- Note that results are from a "quick probe" with 500 draws
- Acknowledge uncertainty in tail behavior for sigma

### If extending this analysis:
- Increase to 1500-2000 draws per chain
- Increase target_accept to 0.95 or 0.99 to reduce remaining divergences
- Consider non-centered parameterization for sigma if tail ESS remains low

---

## Conclusion

**Convergence Status:** MARGINAL (usable with caveats)

The model has achieved reasonable but not ideal convergence. The main issues are:
1. R-hat slightly above 1.01 for one parameter
2. ESS below 400 for several parameters
3. Low tail ESS for sigma parameters

These issues suggest the chains could benefit from longer runs, but the current results are adequate for:
- Comparing model fit to Experiment 1 (via LOO)
- Identifying cluster structure
- Drawing preliminary conclusions about cluster parameters

For publication-quality inference, extending to 1500-2000 draws per chain is recommended.
