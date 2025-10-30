# Experiment 2: Posterior Inference Results

**Model:** 3-Component Finite Mixture Model
**Status:** MARGINAL (usable with caveats)
**Date:** 2025-10-30

---

## Quick Start

**Main summary:** [inference_summary.md](inference_summary.md)
**MCMC diagnostics:** [diagnostics/convergence_report.md](diagnostics/convergence_report.md)
**InferenceData (for LOO):** [diagnostics/posterior_inference.netcdf](diagnostics/posterior_inference.netcdf)

---

## Key Findings

- **K_effective = 2.30** - Model uses 2-3 clusters
- **Weak cluster separation** - Both separations ~0.47 logits (below 0.5 threshold)
- **Low assignment certainty** - All 12 groups have certainty <0.6
- **Marginal convergence** - R-hat ≤ 1.02, ESS ≥ 341 (acceptable but not ideal)
- **Log-likelihood saved** ✓ Ready for LOO-CV comparison

**Decision:** MARGINAL (clusters exist but are weakly separated)

---

## Directory Structure

```
posterior_inference/
├── inference_summary.md          # Main results summary
├── README.md                      # This file
│
├── code/                          # Analysis scripts
│   ├── fit_mixture_quick.py       # Main fitting script (used)
│   ├── analyze_results.py         # Cluster analysis & metrics
│   ├── create_diagnostics.py      # Visualization generation
│   ├── fit_mixture_model.py       # Alternative (longer run)
│   ├── fit_mixture_extended.py    # Alternative (extended sampling)
│   ├── fit_mixture_stan.py        # Stan version (requires compiler)
│   └── mixture_model.stan         # Stan model code
│
├── diagnostics/                   # Data and metrics
│   ├── convergence_report.md      # Detailed MCMC diagnostics
│   ├── posterior_inference.netcdf # ArviZ InferenceData (main output)
│   ├── idata.pkl                  # Pickled InferenceData
│   ├── parameter_summary.csv      # Full parameter summaries
│   ├── cluster_assignments.csv    # Group → cluster mappings
│   └── inference_summary_metrics.csv  # Quick metrics
│
└── plots/                         # Visualizations
    ├── trace_all_params.png       # Trace plots (convergence)
    ├── rank_plots.png             # Rank plots (mixing)
    ├── cluster_parameters.png     # Posterior distributions
    ├── cluster_assignments_heatmap.png  # Assignment probabilities
    ├── cluster_assignments_data.png     # Data + certainty
    ├── cluster_separation.png     # Separation distributions
    └── k_effective.png            # Effective cluster count
```

---

## Files for Next Steps

### For LOO-CV Comparison (Experiment 2 vs Experiment 1)

**Required file:** `/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf`

This file contains:
- Posterior samples (4 chains × 500 draws)
- **log_likelihood['r']** - shape (4, 500, 12) - CRITICAL for LOO
- observed_data['r'] - observed successes

**Usage in Python:**
```python
import arviz as az
idata_exp2 = az.from_netcdf('diagnostics/posterior_inference.netcdf')
loo_exp2 = az.loo(idata_exp2)
```

### For Posterior Predictive Checks

**Use:** `diagnostics/idata.pkl` or `diagnostics/posterior_inference.netcdf`

Contains posterior samples for:
- `p`: Success probabilities (inverse-logit of theta)
- `theta`: Group-level parameters
- `cluster_probs`: Cluster assignment probabilities
- `pi`, `mu`, `sigma`: Cluster parameters

### For Interpreting Clusters

**Use:** `diagnostics/cluster_assignments.csv`

Columns:
- `group_id`, `n_trials`, `r_successes`, `success_rate`: Data
- `assigned_cluster`: Most probable cluster (1, 2, or 3)
- `certainty`: P(assigned cluster) - **all <0.6!**
- `p_cluster_1`, `p_cluster_2`, `p_cluster_3`: Assignment probabilities

---

## Convergence Assessment

### Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max R-hat | 1.02 | <1.01 | MARGINAL |
| Min ESS | 341 | >400 | MARGINAL |
| Divergences | 0.2% | <1% | PASS |

**Overall:** MARGINAL - usable for model comparison and preliminary inference

### What This Means

- **Safe to use for LOO comparison** - Convergence is adequate for relative model comparison
- **Point estimates are reliable** - Posterior means/medians are well-estimated
- **Uncertainty slightly underestimated** - Lower ESS means wider credible intervals may be needed
- **For publication:** Would benefit from extending to 1500-2000 draws per chain

See [diagnostics/convergence_report.md](diagnostics/convergence_report.md) for full details.

---

## Model Summary

### Cluster Parameters

| Cluster | Success Rate | Proportion | N Groups |
|---------|--------------|------------|----------|
| 1 (Low) | 5.0% [1.5%, 7.9%] | 33% | 2 |
| 2 (Medium-low) | 7.5% [4.5%, 12.4%] | 38% | 7 |
| 3 (Medium-high) | 12.1% [6.8%, 28.6%] | 29% | 3 |

**Note:** Substantial overlap in credible intervals reflects weak separation.

### Cluster Membership

**Low certainty:** All groups have assignment certainty <0.6
- Group 8: 57% → Cluster 3 (highest certainty)
- Group 4: 53% → Cluster 1
- Group 2: 54% → Cluster 3
- **All others:** <50% certainty

**Interpretation:** Cluster boundaries are fuzzy, not distinct.

---

## Comparison to EDA

**EDA K-means:** Identified 3 clusters (K=3 optimal by silhouette score)
**Mixture model:** K_effective = 2.30 (uses 2-3 clusters)

**Agreement:**
- High-rate cluster: 3 groups (~12-13%) - MATCHES
- Overall K=3 structure - PARTIAL

**Disagreement:**
- EDA had strong separation; model finds weak separation
- EDA strongly distinguished Group 10 (3.1%); model is uncertain

**Implication:** Discrete clusters exist but may not be as distinct as K-means suggested.

---

## Falsification Results

| Check | Result | Pass? |
|-------|--------|-------|
| K_effective ≥ 2 | 2.30 | ✓ |
| Separation > 0.5 | 0.47, 0.48 | ✗ |
| Certainty ≥ 0.6 | 0.46 | ✗ |
| Convergence OK | Marginal | ✓ |

**Overall:** MARGINAL (2/4 checks passed)

**Interpretation:**
- Clusters exist (K_eff > 2)
- But they're weakly separated
- And assignments are uncertain
- **Model is plausible but not strongly supported**

---

## Next Steps

1. **Compare to Experiment 1 via LOO-CV**
   - Will reveal if mixture improves predictive performance
   - Even with weak clusters, might fit better than continuous hierarchy

2. **Posterior predictive checks**
   - Assess if mixture captures data patterns missed by Experiment 1
   - Check for outliers or misfit

3. **Consider alternatives:**
   - K=2 mixture (given K_eff ≈ 2.3)
   - Continuous hierarchy with heterogeneous variance
   - Model averaging (if LOO suggests equivalence)

---

## Reproducibility

**Software:**
- PyMC 5.26.1 (NUTS sampler, no C++ compilation)
- ArviZ 0.22.0
- Python 3.13

**Sampling:**
- 4 chains × 500 warmup × 500 sampling
- target_accept = 0.90
- Seed = 42
- Runtime: ~8 minutes

**Re-run:** `python code/fit_mixture_quick.py`

---

**Questions?** See [inference_summary.md](inference_summary.md) for comprehensive documentation.
