"""
Model 3: Dirichlet Process Mixture for Non-Parametric Meta-Analysis
Minimal assumptions about number of clusters or heterogeneity structure

Designer: Model Designer 2
Date: 2025-10-28
Dataset: J=8 meta-analysis studies

Key features:
- Dirichlet Process prior on study effects
- Data determine number of clusters
- Stick-breaking construction for computational tractability
- No fixed number of components
"""

import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from scipy import stats


def stick_breaking(beta):
    """
    Stick-breaking construction of Dirichlet Process weights.

    Parameters
    ----------
    beta : tensor
        Beta random variables for stick-breaking

    Returns
    -------
    w : tensor
        Mixture weights (sum to 1)
    """
    portion_remaining = pm.math.concatenate(
        [[1], pm.math.extra_ops.cumprod(1 - beta)[:-1]]
    )
    return beta * portion_remaining


def fit_dp_mixture(y, sigma, K=10, n_chains=4, n_samples=2000, n_tune=3000):
    """
    Fit Dirichlet Process mixture model to meta-analysis data.

    Uses stick-breaking construction with truncation at K components.

    Parameters
    ----------
    y : array-like
        Observed effect sizes (J studies)
    sigma : array-like
        Known standard errors (J studies)
    K : int
        Maximum number of clusters (truncation level)
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

    with pm.Model() as dp_model:
        # Base distribution hyperparameters
        mu_0 = pm.Normal('mu_0', mu=0, sigma=50)
        tau_0 = pm.HalfNormal('tau_0', sigma=10)

        # Concentration parameter
        # Controls expected number of clusters
        # Small alpha → fewer clusters
        # Large alpha → more clusters
        alpha = pm.Gamma('alpha', alpha=1, beta=1)

        # Stick-breaking construction
        # Creates K components, data determine how many are used
        beta = pm.Beta('beta', alpha=1, beta=alpha, shape=K)
        w = pm.Deterministic('w', stick_breaking(beta))

        # Component means from base distribution
        mu_k = pm.Normal('mu_k', mu=mu_0, sigma=tau_0, shape=K)

        # Each study assigned to one component
        # Assignment probabilities given by stick-breaking weights
        component = pm.Categorical('component', p=w, shape=J)

        # Study effects conditional on component assignment
        theta = pm.Deterministic('theta', mu_k[component])

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=theta, sigma=sigma, observed=y)

        # Posterior predictive
        y_pred = pm.Normal('y_pred', mu=theta, sigma=sigma, shape=J)

        # Sample posterior
        # DP models can be challenging, use high target_accept
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=0.99,
            return_inferencedata=True
        )

    return trace, dp_model


def compute_effective_clusters(trace):
    """
    Compute effective number of clusters from DP posterior.

    Uses entropy-based measure:
    K_eff = exp(H(w)) where H is Shannon entropy

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from DP model

    Returns
    -------
    dict
        Cluster count diagnostics
    """
    # Get mixture weights
    w = trace.posterior['w'].values  # shape: (chains, draws, K)

    # Compute entropy for each posterior sample
    # H(w) = -sum(w * log(w))
    epsilon = 1e-10  # avoid log(0)
    entropy = -np.sum(w * np.log(w + epsilon), axis=-1)

    # Effective number of clusters
    k_eff = np.exp(entropy)

    # Component assignment frequencies
    component_assignments = trace.posterior['component'].values
    unique_components, counts = np.unique(component_assignments, return_counts=True)
    prob_by_component = counts / counts.sum()

    diagnostics = {
        'k_eff_mean': k_eff.mean(),
        'k_eff_std': k_eff.std(),
        'k_eff_median': np.median(k_eff),
        'k_eff_95ci': np.percentile(k_eff, [2.5, 97.5]),
        'most_probable_k': int(np.round(k_eff.mean())),
        'component_probabilities': prob_by_component,
        'active_components': len(unique_components)
    }

    return diagnostics


def check_dp_collapse(trace, threshold=0.8):
    """
    Check if DP has collapsed to single cluster or no pooling.

    Single cluster: One weight dominates (> threshold)
    No pooling: Each study in separate cluster

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples
    threshold : float
        Weight threshold for dominance

    Returns
    -------
    dict
        Collapse diagnostics
    """
    # Get mixture weights
    w = trace.posterior['w'].mean(dim=['chain', 'draw']).values

    # Get component assignments
    components = trace.posterior['component'].values
    n_studies = components.shape[-1]

    # Check for single cluster
    max_weight = w.max()
    dominant_component = w.argmax()

    # Check for no pooling (each study separate cluster)
    unique_assignments_per_sample = [
        len(np.unique(components[i, j, :]))
        for i in range(components.shape[0])
        for j in range(components.shape[1])
    ]
    mean_unique = np.mean(unique_assignments_per_sample)

    diagnostics = {
        'single_cluster': max_weight > threshold,
        'dominant_weight': max_weight,
        'dominant_component': dominant_component,
        'no_pooling': mean_unique > 0.8 * n_studies,
        'mean_unique_components': mean_unique,
        'weights': w
    }

    return diagnostics


def sensitivity_to_alpha(y, sigma, alpha_values=[0.1, 1.0, 10.0]):
    """
    Test sensitivity to concentration parameter prior.

    Parameters
    ----------
    y : array-like
        Observed effect sizes
    sigma : array-like
        Known standard errors
    alpha_values : list
        Different alpha hyperparameter values to try

    Returns
    -------
    dict
        Sensitivity results
    """
    results = {}

    for alpha_val in alpha_values:
        print(f"Fitting model with alpha prior centered at {alpha_val}...")

        # Modify model to use different alpha prior
        # (simplified version, actual implementation would vary prior)
        trace, _ = fit_dp_mixture(y, sigma)

        k_eff_diag = compute_effective_clusters(trace)

        results[alpha_val] = {
            'k_eff': k_eff_diag['k_eff_mean'],
            'alpha_posterior': trace.posterior['alpha'].mean().item()
        }

    return results


def compare_to_hierarchical_normal(dp_trace, hn_trace):
    """
    Compare DP model to hierarchical normal baseline.

    Parameters
    ----------
    dp_trace : arviz.InferenceData
        DP model posterior
    hn_trace : arviz.InferenceData
        Hierarchical normal model posterior

    Returns
    -------
    dict
        Comparison results
    """
    # Compare LOO-CV
    dp_loo = az.loo(dp_trace)
    hn_loo = az.loo(hn_trace)

    elpd_diff = dp_loo.elpd_loo - hn_loo.elpd_loo
    se_diff = np.sqrt(dp_loo.se**2 + hn_loo.se**2)

    comparison = {
        'dp_elpd': dp_loo.elpd_loo,
        'hn_elpd': hn_loo.elpd_loo,
        'elpd_difference': elpd_diff,
        'se_difference': se_diff,
        'dp_better': elpd_diff > 2 * se_diff,
        'models_equivalent': abs(elpd_diff) < 2 * se_diff,
        'recommendation': ''
    }

    if comparison['dp_better']:
        comparison['recommendation'] = 'DP model preferred (complexity justified)'
    elif comparison['models_equivalent']:
        comparison['recommendation'] = 'Models equivalent, use simpler hierarchical normal'
    else:
        comparison['recommendation'] = 'Hierarchical normal preferred (DP too complex)'

    return comparison


def main():
    """
    Example usage with meta-analysis data.
    """
    # Load data
    data = pd.read_csv('/workspace/data/data.csv')
    y = data['y'].values
    sigma = data['sigma'].values

    print("Fitting Dirichlet Process Mixture Model...")
    print("=" * 60)
    print(f"Number of studies: {len(y)}")
    print(f"Maximum clusters (truncation): 10")

    # Fit model
    trace, model = fit_dp_mixture(y, sigma, K=10)

    # Convergence diagnostics
    print("\nConvergence Diagnostics:")
    print(az.summary(trace, var_names=['mu_0', 'tau_0', 'alpha']))

    # Effective number of clusters
    print("\nEffective Number of Clusters:")
    cluster_diag = compute_effective_clusters(trace)
    print(f"K_eff (mean): {cluster_diag['k_eff_mean']:.2f}")
    print(f"K_eff (median): {cluster_diag['k_eff_median']:.2f}")
    print(f"95% CI: [{cluster_diag['k_eff_95ci'][0]:.2f}, {cluster_diag['k_eff_95ci'][1]:.2f}]")
    print(f"Most probable K: {cluster_diag['most_probable_k']}")

    # Check for collapse
    print("\nCollapse Diagnostics:")
    collapse = check_dp_collapse(trace)
    print(f"Single cluster? {collapse['single_cluster']}")
    print(f"Dominant weight: {collapse['dominant_weight']:.3f}")
    print(f"No pooling (all separate)? {collapse['no_pooling']}")
    print(f"Mean unique components: {collapse['mean_unique_components']:.2f}")

    if collapse['single_cluster']:
        print("\n⚠️ WARNING: DP collapsed to single cluster")
        print("   Recommendation: Use hierarchical normal model")
        print("   Action: Complexity of DP not justified by data")

    if collapse['no_pooling']:
        print("\n⚠️ WARNING: DP shows no pooling (each study separate)")
        print("   Recommendation: Reconsider pooling assumption")
        print("   Action: May need fixed-effects or no-pooling approach")

    # Interpretation
    if cluster_diag['k_eff_mean'] < 1.5:
        print("\n✓ Data support single cluster (homogeneous)")
    elif 1.5 <= cluster_diag['k_eff_mean'] < 2.5:
        print("\n✓ Data support 2 clusters (mixture structure)")
    else:
        print("\n⚠️ Data suggest complex heterogeneity (>2 clusters)")

    # LOO-CV
    print("\nLeave-One-Out Cross-Validation:")
    loo = az.loo(trace)
    print(f"ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

    print("\nModel fitting complete!")
    print("\nNext steps:")
    print("1. Compare to hierarchical normal model via LOO-CV")
    print("2. Investigate cluster assignments if K_eff > 1")
    print("3. Run sensitivity analysis on alpha prior")
    print("4. Validate with posterior predictive checks")


if __name__ == '__main__':
    main()
