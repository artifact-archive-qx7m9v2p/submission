"""
Model 2: Mixture Model for Heterogeneous Heterogeneity
Allows for hidden subpopulations in meta-analysis

Designer: Model Designer 2
Date: 2025-10-28
Dataset: J=8 meta-analysis studies

Key features:
- Two-component normal mixture
- Allows different means and variances per cluster
- Mixing proportion estimated from data
- Ordering constraint to prevent label switching
"""

import pymc as pm
import numpy as np
import arviz as az
import pandas as pd


def fit_mixture_model(y, sigma, n_chains=4, n_samples=2000, n_tune=2000):
    """
    Fit two-component mixture model to meta-analysis data.

    Parameters
    ----------
    y : array-like
        Observed effect sizes (J studies)
    sigma : array-like
        Known standard errors (J studies)
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Number of post-warmup samples per chain
    n_tune : int
        Number of warmup samples

    Returns
    -------
    trace : arviz.InferenceData
        Posterior samples
    model : pymc.Model
        PyMC model object
    """
    J = len(y)

    with pm.Model() as mixture_model:
        # Priors on cluster means
        mu_1 = pm.Normal('mu_1', mu=0, sigma=50)
        mu_2 = pm.Normal('mu_2', mu=0, sigma=50)

        # Priors on cluster SDs (allow cluster 2 to be more variable)
        tau_1 = pm.HalfNormal('tau_1', sigma=5)
        tau_2 = pm.HalfNormal('tau_2', sigma=10)

        # Mixing proportion (Beta(2,2) favors 50/50 but allows learning)
        pi = pm.Beta('pi', alpha=2, beta=2)

        # Ordering constraint to prevent label switching
        # Ensure mu_1 < mu_2
        mu_ordered = pm.Deterministic(
            'mu_ordered',
            pm.math.stack([pm.math.minimum(mu_1, mu_2),
                          pm.math.maximum(mu_1, mu_2)])
        )
        tau_ordered = pm.Deterministic(
            'tau_ordered',
            pm.math.stack([tau_1, tau_2])
        )

        # Study-specific effects from mixture
        # Each study belongs to one of two clusters
        cluster = pm.Categorical('cluster', p=[pi, 1-pi], shape=J)

        # Study effects conditional on cluster assignment
        theta = pm.Normal(
            'theta',
            mu=mu_ordered[cluster],
            sigma=tau_ordered[cluster],
            shape=J
        )

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

        # Posterior predictive sampling
        y_pred = pm.Normal('y_pred', mu=theta, sigma=sigma, shape=J)

        # Sample posterior
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=0.95,
            return_inferencedata=True
        )

    return trace, mixture_model


def check_mixture_collapse(trace):
    """
    Check if mixture has collapsed to single cluster.

    A mixture has collapsed if:
    1. pi ≈ 0 or pi ≈ 1 (one cluster dominates)
    2. mu_1 ≈ mu_2 (cluster means are similar)
    3. All studies assigned to same cluster

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from mixture model

    Returns
    -------
    dict
        Diagnostics about mixture collapse
    """
    pi_mean = trace.posterior['pi'].mean().item()
    pi_ci = az.hdi(trace.posterior['pi'], hdi_prob=0.95)

    mu_1_mean = trace.posterior['mu_1'].mean().item()
    mu_2_mean = trace.posterior['mu_2'].mean().item()
    mu_diff = abs(mu_2_mean - mu_1_mean)

    # Check cluster assignments
    cluster_probs = trace.posterior['cluster'].mean(dim=['chain', 'draw'])

    diagnostics = {
        'pi_mean': pi_mean,
        'pi_ci': (pi_ci[0].item(), pi_ci[1].item()),
        'collapsed_to_cluster_1': pi_mean < 0.1,
        'collapsed_to_cluster_2': pi_mean > 0.9,
        'means_similar': mu_diff < 5,
        'mu_1_mean': mu_1_mean,
        'mu_2_mean': mu_2_mean,
        'mu_difference': mu_diff,
        'cluster_assignments': cluster_probs.values
    }

    return diagnostics


def compute_mixture_loo(trace, model):
    """
    Compute Leave-One-Out Cross-Validation for mixture model.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples
    model : pymc.Model
        PyMC model object

    Returns
    -------
    loo : arviz.ELPDData
        LOO-CV results
    """
    loo = az.loo(trace, pointwise=True)
    return loo


def posterior_predictive_checks(trace, y_observed):
    """
    Conduct posterior predictive checks.

    Test statistics:
    1. Mean of y_rep vs observed
    2. SD of y_rep vs observed
    3. Min/Max of y_rep vs observed
    4. Proportion of studies outside prediction interval

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples
    y_observed : array-like
        Observed effect sizes

    Returns
    -------
    dict
        PPC results with p-values
    """
    y_rep = trace.posterior_predictive['y_pred'].values.reshape(-1, len(y_observed))

    # Test statistics
    ppc_results = {
        'mean': {
            'observed': np.mean(y_observed),
            'predicted_mean': np.mean(y_rep.mean(axis=1)),
            'p_value': np.mean(y_rep.mean(axis=1) >= np.mean(y_observed))
        },
        'sd': {
            'observed': np.std(y_observed),
            'predicted_mean': np.mean(y_rep.std(axis=1)),
            'p_value': np.mean(y_rep.std(axis=1) >= np.std(y_observed))
        },
        'min': {
            'observed': np.min(y_observed),
            'predicted_mean': np.mean(y_rep.min(axis=1)),
            'p_value': np.mean(y_rep.min(axis=1) <= np.min(y_observed))
        },
        'max': {
            'observed': np.max(y_observed),
            'predicted_mean': np.mean(y_rep.max(axis=1)),
            'p_value': np.mean(y_rep.max(axis=1) >= np.max(y_observed))
        }
    }

    return ppc_results


def main():
    """
    Example usage with meta-analysis data.
    """
    # Load data
    data = pd.read_csv('/workspace/data/data.csv')
    y = data['y'].values
    sigma = data['sigma'].values

    print("Fitting Mixture Model (2 components)...")
    print("=" * 60)

    # Fit model
    trace, model = fit_mixture_model(y, sigma)

    # Convergence diagnostics
    print("\nConvergence Diagnostics:")
    print(az.summary(trace, var_names=['mu_1', 'mu_2', 'tau_1', 'tau_2', 'pi']))

    # Check for mixture collapse
    print("\nMixture Collapse Diagnostics:")
    collapse_diag = check_mixture_collapse(trace)
    print(f"Mixing proportion (pi): {collapse_diag['pi_mean']:.3f}")
    print(f"95% CI: [{collapse_diag['pi_ci'][0]:.3f}, {collapse_diag['pi_ci'][1]:.3f}]")
    print(f"Cluster 1 mean: {collapse_diag['mu_1_mean']:.2f}")
    print(f"Cluster 2 mean: {collapse_diag['mu_2_mean']:.2f}")
    print(f"Mean difference: {collapse_diag['mu_difference']:.2f}")

    if collapse_diag['collapsed_to_cluster_1']:
        print("\n⚠️ WARNING: Mixture collapsed to single cluster (pi ≈ 0)")
        print("   Recommendation: Use hierarchical normal model instead")
    elif collapse_diag['collapsed_to_cluster_2']:
        print("\n⚠️ WARNING: Mixture collapsed to single cluster (pi ≈ 1)")
        print("   Recommendation: Use hierarchical normal model instead")
    elif collapse_diag['means_similar']:
        print("\n⚠️ WARNING: Cluster means are similar")
        print("   Recommendation: Data may not support mixture structure")

    # LOO-CV
    print("\nLeave-One-Out Cross-Validation:")
    loo = compute_mixture_loo(trace, model)
    print(f"ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

    # Posterior predictive checks
    print("\nPosterior Predictive Checks:")
    ppc = posterior_predictive_checks(trace, y)
    for stat_name, stat_values in ppc.items():
        print(f"{stat_name}: p-value = {stat_values['p_value']:.3f}")
        if stat_values['p_value'] < 0.05 or stat_values['p_value'] > 0.95:
            print(f"  ⚠️ WARNING: Poor calibration for {stat_name}")

    print("\nModel fitting complete!")


if __name__ == '__main__':
    main()
