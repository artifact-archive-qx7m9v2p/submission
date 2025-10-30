"""
PyMC Model for AR(1) Log-Normal with Regime-Switching

This model implements:
- AR(1) error structure with sequential dependencies
- Regime-specific variances (3 regimes)
- Log-normal likelihood with quadratic trend
- Updated priors from v2 (Beta for phi_raw, tightened sigma)
"""

import pymc as pm
import numpy as np
import pandas as pd


def build_model(data, regime_idx):
    """
    Build AR(1) Log-Normal model with regime-switching

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'year' and 'C' columns
    regime_idx : np.ndarray
        0-indexed regime indicators (0, 1, 2 for regimes 1, 2, 3)
        Shape: (n_obs,)

    Returns
    -------
    model : pm.Model
        PyMC model with all priors and likelihood
    """

    # Prepare data
    year = data['year'].values
    log_C = np.log(data['C'].values)
    n_obs = len(log_C)
    n_regimes = 3

    with pm.Model() as model:
        # --- Priors ---

        # Trend parameters
        alpha = pm.Normal('alpha', mu=4.3, sigma=0.5)
        beta_1 = pm.Normal('beta_1', mu=0.86, sigma=0.15)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)

        # AR(1) coefficient (updated prior: Beta scaled to (0, 0.95))
        phi_raw = pm.Beta('phi_raw', alpha=20, beta=2)
        phi = pm.Deterministic('phi', 0.95 * phi_raw)

        # Regime-specific standard deviations (tightened)
        sigma_regime = pm.HalfNormal('sigma_regime', sigma=0.5, shape=n_regimes)

        # --- Compute trend (without AR component) ---
        mu_trend = alpha + beta_1 * year + beta_2 * year**2

        # --- AR(1) Structure ---
        # We'll build the likelihood sequentially

        # Initialize epsilon[0] (for t=1) from stationary distribution
        sigma_init = sigma_regime[regime_idx[0]] / pm.math.sqrt(1 - phi**2)
        epsilon_0 = pm.Normal('epsilon_0', mu=0, sigma=sigma_init)

        # Store all epsilon values for sequential computation
        epsilon = [epsilon_0]

        # First observation likelihood
        mu_0 = mu_trend[0] + phi * epsilon_0
        sigma_0 = sigma_regime[regime_idx[0]]
        pm.Normal('obs_0', mu=mu_0, sigma=sigma_0, observed=log_C[0])

        # Compute first epsilon (actual residual from observed data)
        epsilon_1_realized = log_C[0] - mu_trend[0]

        # For t=2 to n_obs (sequential dependencies)
        epsilon_prev = epsilon_1_realized

        for t in range(1, n_obs):
            # Mean at time t includes AR component from previous residual
            mu_t = mu_trend[t] + phi * epsilon_prev
            sigma_t = sigma_regime[regime_idx[t]]

            # Observation likelihood
            pm.Normal(f'obs_{t}', mu=mu_t, sigma=sigma_t, observed=log_C[t])

            # Update epsilon for next iteration
            epsilon_prev = log_C[t] - mu_trend[t]

        # --- Compute log-likelihood for LOO-CV ---
        # We need to recompute the full log-likelihood in vectorized form

        # For posterior predictive checks, we need the full mu vector
        # First element
        mu_full = pm.math.concatenate([
            [mu_trend[0] + phi * epsilon_0],
            mu_trend[1:] + phi * (log_C[:-1] - mu_trend[:-1])
        ])

        # Vectorized log-likelihood
        sigma_full = sigma_regime[regime_idx]
        log_lik = pm.Deterministic(
            'log_likelihood',
            pm.logp(pm.Normal.dist(mu=mu_full, sigma=sigma_full), log_C)
        )

        # Store fitted values for diagnostics
        pm.Deterministic('mu_full', mu_full)
        pm.Deterministic('mu_trend', mu_trend)

    return model


def build_model_alternative(data, regime_idx):
    """
    Alternative implementation using pm.AR for better sampling

    This version uses a more standard AR formulation that may sample better.
    Falls back to this if sequential version has convergence issues.
    """

    year = data['year'].values
    log_C = np.log(data['C'].values)
    n_obs = len(log_C)
    n_regimes = 3

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=4.3, sigma=0.5)
        beta_1 = pm.Normal('beta_1', mu=0.86, sigma=0.15)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)

        phi_raw = pm.Beta('phi_raw', alpha=20, beta=2)
        phi = pm.Deterministic('phi', 0.95 * phi_raw)

        sigma_regime = pm.HalfNormal('sigma_regime', sigma=0.5, shape=n_regimes)

        # Trend component
        mu_trend = alpha + beta_1 * year + beta_2 * year**2

        # Detrend observations
        y_detrended = log_C - mu_trend

        # AR(1) model on detrended data with regime-specific variance
        # Note: This is approximate as AR doesn't natively support regime switching
        # Use weighted variance approach

        # For simplicity in AR, use mean sigma (suboptimal but allows AR class)
        sigma_mean = pm.math.mean(sigma_regime)

        # AR(1) likelihood
        # First observation from stationary distribution
        epsilon_0 = pm.Normal('epsilon_0',
                              mu=0,
                              sigma=sigma_mean / pm.math.sqrt(1 - phi**2))

        # Subsequent observations
        for t in range(1, n_obs):
            epsilon_t = pm.Normal(f'epsilon_{t}',
                                  mu=phi * (y_detrended[t-1] if t==1 else log_C[t-1] - mu_trend[t-1]),
                                  sigma=sigma_regime[regime_idx[t]])
            pm.Deterministic(f'y_pred_{t}', mu_trend[t] + epsilon_t)

        # Observations
        for t in range(n_obs):
            if t == 0:
                mu_t = mu_trend[0] + epsilon_0
            else:
                mu_t = mu_trend[t] + phi * (log_C[t-1] - mu_trend[t-1])

            pm.Normal(f'obs_{t}',
                     mu=mu_t,
                     sigma=sigma_regime[regime_idx[t]],
                     observed=log_C[t])

    return model


if __name__ == '__main__':
    # Test model builds correctly
    data = pd.DataFrame({
        'year': np.linspace(-1.67, 1.67, 40),
        'C': np.random.lognormal(4, 0.5, 40)
    })

    # Create regime indices (0-indexed)
    regime_idx = np.concatenate([
        np.zeros(14, dtype=int),
        np.ones(13, dtype=int),
        np.full(13, 2, dtype=int)
    ])

    model = build_model(data, regime_idx)
    print("Model built successfully!")
    print(model.basic_RVs)
