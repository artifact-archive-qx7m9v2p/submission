"""
Latent AR(1) Negative Binomial Model (Experiment 3)

Model addresses residual autocorrelation from Experiment 1 by adding AR(1) errors.

Observation model:
    C_t ~ NegativeBinomial(μ_t, φ)
    log(μ_t) = α_t

Latent AR(1) state process:
    α_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
    ε_t = ρ·ε_{t-1} + η_t
    η_t ~ Normal(0, σ_η)
    ε_1 ~ Normal(0, σ_η/√(1-ρ²))

Non-centered parameterization for better sampling:
    ε_raw[t] ~ Normal(0, 1)
    ε[t] = ρ·ε[t-1] + σ_η·ε_raw[t]
"""

import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor import scan


def build_model(year, C):
    """
    Build the Latent AR(1) Negative Binomial model.

    Parameters
    ----------
    year : array-like
        Standardized year values
    C : array-like
        Case count observations

    Returns
    -------
    model : pm.Model
        PyMC model object
    """
    N = len(year)

    with pm.Model() as model:
        # Data
        year_data = pm.Data('year', year)
        C_obs = pm.Data('C', C)

        # Priors for trend parameters
        beta_0 = pm.Normal('beta_0', mu=4.7, sigma=0.3)
        beta_1 = pm.Normal('beta_1', mu=0.8, sigma=0.2)
        beta_2 = pm.Normal('beta_2', mu=0.3, sigma=0.1)

        # Priors for AR(1) parameters
        # ρ ~ Beta(12, 3) gives mean=0.8, concentrated around high autocorrelation
        rho = pm.Beta('rho', alpha=12, beta=3)

        # σ_η ~ HalfNormal(0, 0.5)
        sigma_eta = pm.HalfNormal('sigma_eta', sigma=0.5)

        # Prior for dispersion
        phi = pm.Gamma('phi', alpha=2, beta=0.5)

        # Quadratic trend component
        trend = beta_0 + beta_1 * year_data + beta_2 * year_data**2

        # AR(1) errors with non-centered parameterization using scan
        # Initial error with stationary distribution
        epsilon_raw_0 = pm.Normal('epsilon_raw_0', mu=0, sigma=1)
        sigma_0 = sigma_eta / pt.sqrt(1 - rho**2)
        epsilon_0 = sigma_0 * epsilon_raw_0

        # AR(1) innovations (non-centered)
        epsilon_raw = pm.Normal('epsilon_raw', mu=0, sigma=1, shape=N-1)

        # Use scan to build AR(1) process
        def ar1_step(epsilon_raw_t, epsilon_tm1, rho, sigma_eta):
            """One step of AR(1): epsilon_t = rho * epsilon_tm1 + sigma_eta * epsilon_raw_t"""
            return rho * epsilon_tm1 + sigma_eta * epsilon_raw_t

        epsilon_rest, _ = scan(
            fn=ar1_step,
            sequences=[epsilon_raw],
            outputs_info=[epsilon_0],
            non_sequences=[rho, sigma_eta],
            strict=True
        )

        # Combine initial and subsequent errors
        epsilon = pt.concatenate([[epsilon_0], epsilon_rest])

        # Latent state: α_t = trend + ε_t
        alpha = pm.Deterministic('alpha', trend + epsilon)

        # Expected count on natural scale
        mu = pm.Deterministic('mu', pt.exp(alpha))

        # Likelihood: NegativeBinomial(μ, φ)
        # PyMC parameterization: NegativeBinomial(mu, alpha)
        # where alpha is the dispersion parameter (our φ)
        obs = pm.NegativeBinomial('obs', mu=mu, alpha=phi, observed=C_obs)

    return model


def summary_info():
    """Return model summary information"""
    return {
        'name': 'Latent AR(1) Negative Binomial',
        'parameters': ['beta_0', 'beta_1', 'beta_2', 'rho', 'sigma_eta', 'phi'],
        'latent_variables': ['alpha', 'epsilon'],
        'parameterization': 'non-centered',
        'purpose': 'Model residual autocorrelation from Experiment 1'
    }
