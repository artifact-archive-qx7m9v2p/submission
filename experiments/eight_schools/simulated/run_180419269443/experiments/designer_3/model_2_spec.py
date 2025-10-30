"""
Model 2: Prior-Data Conflict Detection Model
Designer 3 - Bayesian Meta-Analysis
PyMC implementation with mixture priors and conflict detection mechanism
"""

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

def build_conflict_detection_model(data_path='/workspace/data/data.csv'):
    """
    Build PyMC model with:
    - Mixture priors on mu (skeptical + optimistic)
    - Mixture priors on tau (tight + heavy-tailed)
    - Explicit conflict detection via Bernoulli indicators
    - SE inflation for conflicted studies

    Parameters
    ----------
    data_path : str
        Path to CSV with columns: study, y, sigma

    Returns
    -------
    model : pm.Model
        PyMC model ready for sampling
    data : pd.DataFrame
        Observed data
    """
    # Load data
    data = pd.read_csv(data_path)
    J = len(data)
    y_obs = data['y'].values
    sigma_obs = data['sigma'].values

    with pm.Model() as model:
        # ============================================================
        # MIXTURE PRIOR ON MU (mean effect)
        # Component 1: Skeptical (centered at null)
        # Component 2: Optimistic (centered at observed pooled estimate)
        # ============================================================
        w_mu = pm.Dirichlet('w_mu', a=np.array([1.0, 1.0]))  # Equal prior weights

        mu_skeptical = pm.Normal('mu_skeptical', mu=0, sigma=25)
        mu_optimistic = pm.Normal('mu_optimistic', mu=11, sigma=8)

        mu = pm.Mixture('mu', w=w_mu, comp_dists=[
            pm.Normal.dist(mu=mu_skeptical, sigma=1e-6),  # Point mass approximation
            pm.Normal.dist(mu=mu_optimistic, sigma=1e-6)
        ])

        # Simpler alternative: direct mixture
        # mu_components = pm.Normal('mu_components', mu=[0, 11], sigma=[25, 8], shape=2)
        # mu = pm.Mixture('mu', w=w_mu, comp_dists=mu_components)

        # ============================================================
        # MIXTURE PRIOR ON TAU (between-study SD)
        # Component 1: Tight (expect low heterogeneity) - 70% weight
        # Component 2: Heavy-tailed (allow high heterogeneity) - 30% weight
        # ============================================================
        w_tau = pm.Dirichlet('w_tau', a=np.array([7.0, 3.0]))  # Favor tight component

        tau_tight = pm.HalfNormal('tau_tight', sigma=5)
        tau_heavy = pm.HalfCauchy('tau_heavy', beta=10)

        tau = pm.Mixture('tau', w=w_tau, comp_dists=[
            pm.HalfNormal.dist(sigma=tau_tight),
            pm.HalfCauchy.dist(beta=tau_heavy)
        ])

        # ============================================================
        # CONFLICT DETECTION MECHANISM
        # ============================================================
        # Probability that a study is in conflict with prior assumptions
        pi_conflict = pm.Beta('pi_conflict', alpha=1, beta=1)  # Uniform [0, 1]

        # Binary indicators: is study j in conflict?
        z = pm.Bernoulli('z', p=pi_conflict, shape=J)

        # SE inflation factor for conflicted studies
        # LogNormal(log(3), 0.5) has median=3, mean~3.4
        inflation_factor = pm.LogNormal('inflation_factor', mu=np.log(3), sigma=0.5)

        # ============================================================
        # HIERARCHICAL STRUCTURE
        # ============================================================
        # Study-specific effects
        theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)

        # Adjusted standard errors: inflate if study is flagged
        # sigma_adj[j] = sigma[j] if z[j]=0, else sigma[j] * inflation_factor
        sigma_adj = sigma_obs * (1 + (inflation_factor - 1) * z)

        # ============================================================
        # LIKELIHOOD
        # ============================================================
        y_obs_node = pm.Normal('y_obs', mu=theta, sigma=sigma_adj, observed=y_obs)

        # ============================================================
        # DERIVED QUANTITIES
        # ============================================================
        # I-squared
        sigma_pooled_sq = pm.Deterministic('sigma_pooled_sq', pm.math.mean(sigma_obs**2))
        I_squared = pm.Deterministic('I_squared', tau**2 / (tau**2 + sigma_pooled_sq))

        # Number of studies flagged as conflicts
        n_conflicts = pm.Deterministic('n_conflicts', pm.math.sum(z))

        # Predictive distribution for new study (not conflicted)
        theta_new = pm.Normal('theta_new', mu=mu, sigma=tau)

    return model, data


def sample_model(model, draws=2000, tune=2000, chains=4, target_accept=0.95):
    """
    Sample from the conflict detection model

    Parameters
    ----------
    model : pm.Model
        PyMC model
    draws : int
        Number of posterior samples per chain
    tune : int
        Number of tuning samples
    chains : int
        Number of MCMC chains
    target_accept : float
        Target acceptance rate for NUTS

    Returns
    -------
    trace : az.InferenceData
        Posterior samples and diagnostics
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True}  # For LOO-CV
        )

    return trace


def diagnose_model(trace, model, data):
    """
    Run diagnostic checks on fitted model

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples
    model : pm.Model
        PyMC model
    data : pd.DataFrame
        Observed data

    Returns
    -------
    diagnostics : dict
        Dictionary of diagnostic results
    """
    diagnostics = {}

    # Convergence diagnostics
    diagnostics['rhat'] = az.rhat(trace)
    diagnostics['ess_bulk'] = az.ess(trace, method='bulk')
    diagnostics['ess_tail'] = az.ess(trace, method='tail')
    diagnostics['mcse'] = az.mcse(trace)

    # Model comparison
    diagnostics['loo'] = az.loo(trace, pointwise=True)
    diagnostics['waic'] = az.waic(trace, pointwise=True)

    # Posterior predictive checks
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    diagnostics['ppc'] = ppc

    # Summary statistics
    diagnostics['summary'] = az.summary(
        trace,
        var_names=['mu', 'tau', 'pi_conflict', 'n_conflicts', 'I_squared',
                   'inflation_factor', 'z', 'theta'],
        hdi_prob=0.95
    )

    return diagnostics


def identify_conflicts(trace, data, threshold=0.5):
    """
    Identify which studies are flagged as conflicts

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples
    data : pd.DataFrame
        Observed data
    threshold : float
        Posterior probability threshold for flagging

    Returns
    -------
    conflicts : pd.DataFrame
        Studies with P(z_j = 1) > threshold
    """
    z_posterior = trace.posterior['z'].values  # Shape: (chains, draws, J)
    z_mean = z_posterior.mean(axis=(0, 1))  # Posterior mean for each study

    conflicts = pd.DataFrame({
        'study': data['study'].values,
        'y': data['y'].values,
        'sigma': data['sigma'].values,
        'P_conflict': z_mean,
        'is_conflict': z_mean > threshold
    })

    return conflicts.sort_values('P_conflict', ascending=False)


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == '__main__':
    # Build model
    model, data = build_conflict_detection_model()

    # Print model structure
    print("Model structure:")
    print(model)

    # Sample (uncomment to run)
    # trace = sample_model(model)

    # Diagnostics (uncomment to run)
    # diagnostics = diagnose_model(trace, model, data)
    # print("\nModel diagnostics:")
    # print(diagnostics['summary'])

    # Identify conflicts (uncomment to run)
    # conflicts = identify_conflicts(trace, data)
    # print("\nStudies flagged as conflicts:")
    # print(conflicts[conflicts['is_conflict']])


"""
NOTES:
------
1. Mixture priors in PyMC can be tricky - may need to use pm.NormalMixture instead
2. Discrete parameters (z) may cause mixing issues - consider using continuous approximation
3. Inflation factor may be poorly identified if few studies are flagged
4. Computational cost ~5-10 minutes for 8000 total samples

ALTERNATIVES IF MODEL FAILS:
-----------------------------
1. Replace discrete z with continuous "conflict weight": w_j ~ Beta(1,1)
   - sigma_adj = sigma * (1 + w * (inflation - 1))
   - Smoother, better mixing

2. Simplify mixture priors to single components
   - Just use optimistic prior: mu ~ N(11, 8)
   - Compare to skeptical: mu ~ N(0, 25)
   - Reduces complexity

3. Use marginalized likelihood (integrate out z analytically)
   - Faster, but less interpretable
"""
