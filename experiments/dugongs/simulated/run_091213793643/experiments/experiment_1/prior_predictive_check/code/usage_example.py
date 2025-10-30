"""
Example: How to use logarithmic_model.stan for different purposes

The Stan model supports both prior predictive sampling and full inference
through the 'prior_only' flag.
"""

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# Load data
data = pd.read_csv('/workspace/data/data.csv')
x_obs = data['x'].values
Y_obs = data['Y'].values
N = len(data)

# Compile model (once)
model = CmdStanModel(stan_file='/workspace/experiments/experiment_1/prior_predictive_check/code/logarithmic_model.stan')

# ==============================================================================
# USE CASE 1: Prior Predictive Check (what we just did)
# ==============================================================================
"""
Sample from priors WITHOUT conditioning on data.
Use prior_only=1 to skip likelihood evaluation.
"""
prior_data = {
    'N': N,
    'x': x_obs,
    'Y': Y_obs,  # Not used when prior_only=1, but required by Stan data block
    'prior_only': 1
}

prior_fit = model.sample(
    data=prior_data,
    chains=1,
    iter_sampling=1000,
    iter_warmup=0,
    fixed_param=True,  # Use fixed_param sampler for prior sampling
    show_progress=True
)

# Extract prior samples
alpha_prior = prior_fit.stan_variable('alpha')
beta_prior = prior_fit.stan_variable('beta')
sigma_prior = prior_fit.stan_variable('sigma')
Y_rep_prior = prior_fit.stan_variable('Y_rep')  # Prior predictive datasets

# ==============================================================================
# USE CASE 2: Full Bayesian Inference (next stage)
# ==============================================================================
"""
Fit model to observed data to get posterior distributions.
Use prior_only=0 to evaluate the likelihood.
"""
inference_data = {
    'N': N,
    'x': x_obs,
    'Y': Y_obs,
    'prior_only': 0  # EVALUATE LIKELIHOOD
}

posterior_fit = model.sample(
    data=inference_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    show_progress=True
)

# Extract posterior samples
alpha_post = posterior_fit.stan_variable('alpha')
beta_post = posterior_fit.stan_variable('beta')
sigma_post = posterior_fit.stan_variable('sigma')
Y_rep_post = posterior_fit.stan_variable('Y_rep')  # Posterior predictive
log_lik = posterior_fit.stan_variable('log_lik')  # For LOO-CV

# Check convergence
print(posterior_fit.diagnose())
print(posterior_fit.summary())

# ==============================================================================
# USE CASE 3: Simulation-Based Calibration (next step)
# ==============================================================================
"""
Generate synthetic data from known parameters, fit model, check if we
recover the true parameters.
"""
# Generate synthetic data
np.random.seed(123)
alpha_true = 1.8
beta_true = 0.25
sigma_true = 0.12

mu_true = alpha_true + beta_true * np.log(x_obs)
Y_synthetic = mu_true + np.random.normal(0, sigma_true, N)

# Fit to synthetic data
sbc_data = {
    'N': N,
    'x': x_obs,
    'Y': Y_synthetic,
    'prior_only': 0
}

sbc_fit = model.sample(data=sbc_data, chains=4, iter_sampling=1000)

# Check recovery
alpha_recovered = sbc_fit.stan_variable('alpha')
print(f"True α: {alpha_true:.3f}, Recovered: {alpha_recovered.mean():.3f} ± {alpha_recovered.std():.3f}")
print(f"True β: {beta_true:.3f}, Recovered: {sbc_fit.stan_variable('beta').mean():.3f}")
print(f"True σ: {sigma_true:.3f}, Recovered: {sbc_fit.stan_variable('sigma').mean():.3f}")

# ==============================================================================
# KEY POINTS
# ==============================================================================
"""
1. The same Stan model handles all use cases via the 'prior_only' flag
2. prior_only=1: Prior predictive (skips likelihood, use fixed_param sampler)
3. prior_only=0: Full inference (evaluates likelihood, use HMC/NUTS)
4. log_lik in generated quantities enables LOO-CV via ArviZ
5. Y_rep enables posterior predictive checks
"""
