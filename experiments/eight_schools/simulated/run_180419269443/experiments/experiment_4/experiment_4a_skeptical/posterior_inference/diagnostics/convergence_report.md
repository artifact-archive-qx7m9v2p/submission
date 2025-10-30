# Convergence Report: Model 4a (Skeptical Priors)

## Sampling Configuration
- Sampler: Gibbs (non-centered parameterization)
- Chains: 4
- Iterations: 10000 (5000 warmup, thin=2)
- Total samples: 10000

## Convergence Metrics
- Max R-hat: 1.6000 ✗
- Min ESS_bulk: 7 ✗
- Min ESS_tail: 16 ✗

## Posterior Estimates
**mu (population mean):**
- Posterior: 8.58 ± 3.80
- 95% CI: [1.05, 16.12]
- Prior: N(0, 10) [skeptical]
- Shift: 8.58 units from prior mean

**tau (population SD):**
- Posterior: 0.13 ± 0.64
- 95% CI: [0.00, 1.30]
- Prior: Half-Normal(0, 5)

## Visual Diagnostics
See plots/ directory for diagnostic plots.
