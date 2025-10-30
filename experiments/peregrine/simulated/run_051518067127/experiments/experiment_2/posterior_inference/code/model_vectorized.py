"""
Vectorized AR(1) Log-Normal Model with Regime-Switching

This implementation uses a fully vectorized approach to avoid PyTensor compilation issues.
Key differences from sequential implementation:
- Single vectorized likelihood instead of 40 separate observation nodes
- Handles AR structure using tensor operations
- Works around regime-switching by computing likelihood terms separately
"""

import pymc as pm
import numpy as np
import pytensor.tensor as pt


def build_model_vectorized(data, regime_idx):
    """
    Build AR(1) Log-Normal model with vectorized likelihood

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'year' and 'C' columns
    regime_idx : np.ndarray
        0-indexed regime indicators (0, 1, 2 for regimes 1, 2, 3)

    Returns
    -------
    model : pm.Model
        PyMC model with vectorized likelihood
    """

    year = data['year'].values
    log_C = np.log(data['C'].values)
    n_obs = len(log_C)
    n_regimes = 3

    with pm.Model() as model:
        # --- Priors ---
        alpha = pm.Normal('alpha', mu=4.3, sigma=0.5)
        beta_1 = pm.Normal('beta_1', mu=0.86, sigma=0.15)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)

        # AR(1) coefficient
        phi_raw = pm.Beta('phi_raw', alpha=20, beta=2)
        phi = pm.Deterministic('phi', 0.95 * phi_raw)

        # Regime-specific standard deviations
        sigma_regime = pm.HalfNormal('sigma_regime', sigma=0.5, shape=n_regimes)

        # --- Compute trend (without AR component) ---
        mu_trend = alpha + beta_1 * year + beta_2 * year**2

        # --- AR(1) Structure (Vectorized) ---

        # Initialize epsilon[0] from stationary distribution
        sigma_init = sigma_regime[regime_idx[0]] / pm.math.sqrt(1 - phi**2)
        epsilon_0 = pm.Normal('epsilon_0', mu=0, sigma=sigma_init)

        # For likelihood computation, we need:
        # - mu[0] = mu_trend[0] + phi * epsilon_0
        # - mu[t] = mu_trend[t] + phi * epsilon[t-1] for t > 0
        #   where epsilon[t-1] = log_C[t-1] - mu_trend[t-1]

        # Build the full mu vector using scan or custom operation
        # We'll use a simpler approach: compute contributions separately

        # Epsilon values (past residuals) - these are deterministic given data
        epsilon_realized = log_C - mu_trend

        # Build mu vector:
        # mu[0] uses epsilon_0 (latent)
        # mu[t] uses epsilon[t-1] = log_C[t-1] - mu_trend[t-1] (observed)

        mu_0 = mu_trend[0] + phi * epsilon_0

        # For t >= 1, use the realized epsilons from previous time steps
        mu_rest = mu_trend[1:] + phi * epsilon_realized[:-1]

        # Combine into full mu vector
        mu_full = pt.concatenate([[mu_0], mu_rest])

        # Select regime-specific sigmas for each observation
        sigma_full = sigma_regime[regime_idx]

        # --- Vectorized Likelihood ---
        # Split into first observation and rest to handle different structure

        # First observation (depends on epsilon_0)
        pm.Normal('obs_0', mu=mu_0, sigma=sigma_full[0], observed=log_C[0])

        # Remaining observations (depend on previous observed values)
        pm.Normal('obs_rest', mu=mu_rest, sigma=sigma_full[1:], observed=log_C[1:])

        # --- Compute log-likelihood for each observation (for LOO-CV) ---
        # We need pointwise log-likelihood, so compute separately for each obs

        log_lik_0 = pm.logp(pm.Normal.dist(mu=mu_0, sigma=sigma_full[0]), log_C[0])
        log_lik_rest = pm.logp(pm.Normal.dist(mu=mu_rest, sigma=sigma_full[1:]), log_C[1:])

        # Combine into single vector
        log_likelihood = pm.Deterministic('log_likelihood',
                                          pt.concatenate([[log_lik_0], log_lik_rest]))

        # Store fitted values for diagnostics
        pm.Deterministic('mu_full', mu_full)
        pm.Deterministic('mu_trend', mu_trend)

    return model
