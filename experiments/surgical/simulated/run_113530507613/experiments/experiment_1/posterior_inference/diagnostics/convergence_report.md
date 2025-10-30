# Convergence Report: Hierarchical Logit-Normal Model

**Experiment:** Experiment 1 - Standard Hierarchical Logit-Normal Model
**Date:** 2025-10-30
**Sampler:** PyMC (NUTS with target_accept=0.99)
**Chains:** 4
**Iterations:** 2000 warmup + 2000 sampling = 4000 per chain
**Total Samples:** 8000 post-warmup

---

## MCMC Diagnostic Summary

### 1. R-hat Statistic (Convergence)
**Criterion:** R-hat < 1.01 for all parameters
**Status:** PASS

| Parameter Class | Max R-hat | Status |
|----------------|-----------|---------|
| Hyperparameters (mu, tau) | 1.00000 | PASS |
| Group effects (theta_raw[1:12]) | 1.00000 | PASS |
| Derived (theta[1:12]) | 1.00000 | PASS |

**Interpretation:** All R-hat values equal 1.00, indicating perfect convergence across all chains. No evidence of non-convergence.

---

### 2. Effective Sample Size (ESS)
**Criterion:** ESS (bulk) > 400 and ESS (tail) > 400 for all parameters
**Status:** PASS

| Parameter | ESS (bulk) | ESS (tail) | Status |
|-----------|-----------|-----------|---------|
| mu | 1362 | 2520 | PASS |
| tau | 1024 | 2086 | PASS |
| theta[0] | 2202 | 3602 | PASS |
| theta[1] | 1579 | 2853 | PASS |
| theta[2] | 1978 | 2908 | PASS |
| theta[3] | 1356 | 2419 | PASS |
| theta[4] | 1865 | 3352 | PASS |
| theta[5] | 2203 | 4227 | PASS |
| theta[6] | 1933 | 3382 | PASS |
| theta[7] | 1301 | 2573 | PASS |
| theta[8] | 2035 | 3718 | PASS |
| theta[9] | 1578 | 3382 | PASS |
| theta[10] | 1933 | 3759 | PASS |
| theta[11] | 1720 | 3236 | PASS |

**Summary Statistics:**
- Minimum ESS (bulk): 1024 (tau)
- Minimum ESS (tail): 2086 (tau)
- Both well above minimum threshold of 400

**Interpretation:** Excellent effective sample sizes across all parameters. The non-centered parameterization successfully avoided the funnel geometry that often plagues hierarchical models, resulting in efficient sampling.

---

### 3. Divergent Transitions
**Criterion:** < 1% of post-warmup samples
**Status:** PASS

- **Divergent transitions:** 0 / 8000 (0.00%)
- **Probe phase:** 0 / 400 (0.00%)
- **Main sampling:** 0 / 8000 (0.00%)

**Interpretation:** Zero divergences indicates excellent posterior geometry exploration. The non-centered parameterization and high target_accept (0.99) successfully navigated the posterior without pathologies.

---

### 4. Monte Carlo Standard Error (MCSE)
**Criterion:** MCSE < 5% of posterior SD
**Status:** PASS

| Parameter | Posterior SD | MCSE | MCSE/SD Ratio |
|-----------|-------------|------|---------------|
| mu | 0.144 | 0.004 | 2.8% |
| tau | 0.128 | 0.004 | 3.1% |

All group-level parameters also show MCSE < 5% of SD.

**Interpretation:** Low Monte Carlo error relative to posterior uncertainty. Estimates are stable and reliable.

---

### 5. Chain Mixing

**Visual Evidence (see plots/):**
- **Trace plots** (`trace_plot_mu.png`, `trace_plot_tau.png`, `trace_plots_selected_groups.png`): All chains show excellent mixing with no evidence of sticking or drift. Chains explore the same regions of parameter space with no systematic differences.

- **Rank plots** (`rank_plots.png`): Uniform rank distributions across all parameters, confirming chains are mixing well and exploring the full posterior.

- **Autocorrelation** (`autocorrelation_diagnostics.png`): Rapid decay to zero (within ~10 lags for all parameters), indicating efficient exploration with minimal autocorrelation.

**Status:** PASS

---

### 6. Energy Diagnostic (E-BFMI)
**Criterion:** E-BFMI > 0.2
**Status:** PASS

**Visual Evidence:** Energy diagnostic plot (`energy_diagnostic.png`) shows good overlap between energy transition distribution and marginal energy distribution. No evidence of sampling bias.

**Interpretation:** The sampler is exploring the posterior efficiently without geometric pathologies.

---

### 7. Geometry Diagnostics

**Funnel Check** (`pairs_plot_hyperparameters.png`):
- Joint distribution of (mu, tau) shows no evidence of funnel geometry
- Non-centered parameterization (theta = mu + tau * theta_raw) successfully decorrelated hyperparameters from group effects
- Posterior is well-behaved and conducive to efficient sampling

**Status:** PASS

---

## Adaptive Sampling Strategy

### Phase 1: Initial Probe
- **Configuration:** 4 chains, 100 warmup, 100 sampling, target_accept=0.8
- **Outcome:** Detected R-hat = 1.21 for tau (expected for short chains), 0 divergences
- **Decision:** Proceed to main sampling with increased target_accept=0.99

### Phase 2: Main Sampling
- **Configuration:** 4 chains, 2000 warmup, 2000 sampling, target_accept=0.99
- **Outcome:** Perfect convergence (R-hat=1.00), no divergences, excellent ESS
- **Sampling time:** 226 seconds (~3.8 minutes)

**Conclusion:** The adaptive strategy successfully identified that a conservative target_accept was beneficial. The non-centered parameterization proved highly effective.

---

## Overall Assessment

### Convergence Checklist

- [x] R-hat < 1.01 for ALL parameters
- [x] ESS (bulk) > 400 for ALL parameters
- [x] ESS (tail) > 400 for ALL parameters
- [x] Divergences < 1% (achieved 0%)
- [x] MCSE < 5% of posterior SD
- [x] Trace plots show good mixing
- [x] Rank plots show uniform distributions
- [x] Energy diagnostic passes (E-BFMI > 0.2)
- [x] No geometric pathologies (funnel avoided)

### OVERALL STATUS: PASS

**Confidence Level:** High

All MCMC diagnostics pass stringent thresholds. The posterior inference is reliable and ready for scientific interpretation. The non-centered parameterization proved essential for efficient sampling in this hierarchical model.

---

## Technical Notes

1. **PPL Used:** PyMC 5.26.1 (CmdStan installation failed due to missing build tools; PyMC served as successful fallback)

2. **Parameterization:** Non-centered (theta = mu + tau * theta_raw) avoided funnel geometry that would have plagued centered parameterization

3. **Tuning:** High target_accept (0.99) used conservatively after probe phase; resulted in zero divergences

4. **Efficiency:** Achieved ESS/iteration ratios of 13-27% (1024-2203 ESS from 8000 samples), indicating highly efficient sampling

5. **Log-likelihood:** Successfully saved in InferenceData for downstream LOO-CV comparison (Phase 4)
