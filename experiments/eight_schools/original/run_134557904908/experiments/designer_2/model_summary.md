# Quick Reference: Designer 2 Models

## Three Robust Bayesian Models for Meta-Analysis

### Model 1: Student-t Robust (PRIORITY 1)
- **Class**: Heavy-tailed likelihood
- **Parameters**: theta (location), nu (degrees of freedom)
- **Priors**: theta ~ N(0, 20²), nu ~ Gamma(2, 0.1)
- **Advantage**: Automatic outlier robustness, 1 extra parameter
- **Reject if**: nu > 50 (normal is sufficient)

### Model 2: Contaminated Normal Mixture (PRIORITY 2)
- **Class**: Mixture model
- **Parameters**: theta, pi (contamination prob), lambda (variance inflation)
- **Priors**: theta ~ N(0, 20²), pi ~ Beta(1,4), lambda ~ Gamma(2, 0.5)
- **Advantage**: Identifies specific contaminated observations
- **Reject if**: pi < 0.05 or pi > 0.6

### Model 3: Hierarchical Uncertainty (PRIORITY 3)
- **Class**: Hierarchical model for measurement error
- **Parameters**: theta, psi (global sigma uncertainty), tau_i (true SDs)
- **Priors**: theta ~ N(0, 20²), psi ~ HC(0, 5), tau_i ~ TruncNormal(sigma_i, psi²)
- **Advantage**: Accounts for uncertainty in reported sigmas
- **Reject if**: psi < 1.0 (sigmas effectively known)

## Implementation Order
1. Normal baseline (for comparison)
2. Student-t (primary robust model)
3. Mixture (if Student-t shows issues)
4. Hierarchical (only if sigma misspecification suspected)

## Key Metrics
- LOO-CV for model comparison (ELPD difference)
- Posterior predictive checks (visual)
- ESS > 400 for theta, > 200 for auxiliary parameters
- R-hat < 1.01, divergences < 1%

## Decision Rules
- LOO within 2 ELPD → choose simpler model
- LOO difference > 5 → trust winner
- All models similar → normal model sufficient
- Robust models differ from normal → investigate why
