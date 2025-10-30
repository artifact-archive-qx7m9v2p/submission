"""
Implementation Templates for Alternative Bayesian Models (Designer 2)

Ready-to-use PyMC code templates for three alternative model classes:
1. Finite Mixture (K=2 components)
2. Robust Hierarchical (Student-t random effects)
3. Dirichlet Process Mixture (nonparametric)

Usage:
    import pymc as pm
    import numpy as np
    from implementation_templates import build_mixture_model, fit_model

    # Load data
    n = data['n'].values
    r = data['r'].values

    # Build and fit model
    model = build_mixture_model(n, r)
    trace = fit_model(model, model_name='mixture')
"""

import pymc as pm
import numpy as np
import arviz as az


# ============================================================================
# MODEL 1: FINITE MIXTURE (K=2 COMPONENTS)
# ============================================================================

def build_mixture_model(n, r):
    """
    Two-component finite mixture model for binomial outcomes.

    Core hypothesis: Groups come from two distinct subpopulations
    with different baseline risk levels (low-risk ~6%, high-risk ~12%).

    Parameters
    ----------
    n : array-like, shape (n_groups,)
        Sample sizes per group
    r : array-like, shape (n_groups,)
        Event counts per group

    Returns
    -------
    model : pm.Model
        PyMC model object

    Notes
    -----
    - Mixture weights: w ~ Beta(2, 2)
    - Component means: mu_k ~ Beta() with ordering constraint
    - Component concentrations: kappa_k ~ Gamma(2, 0.1)
    - Falsification: Abandon if w < 0.1 or w > 0.9
    """
    n = np.asarray(n)
    r = np.asarray(r)
    n_groups = len(n)

    with pm.Model() as mixture_model:
        # Mixture weight (symmetric prior)
        w = pm.Beta('w', alpha=2, beta=2)

        # Component means (ordered: mu_1 < mu_2)
        # Informative priors based on EDA: low-risk ~6%, high-risk ~12%
        mu_raw = pm.Beta('mu_raw',
                        alpha=[6, 12],
                        beta=[94, 88],
                        shape=2)
        mu = pm.Deterministic('mu', pm.math.sort(mu_raw))

        # Component concentrations (inverse variance)
        kappa = pm.Gamma('kappa', alpha=2, beta=0.1, shape=2)

        # Beta distribution parameters
        alpha = pm.Deterministic('alpha', mu * kappa)
        beta_param = pm.Deterministic('beta_param', (1 - mu) * kappa)

        # Component assignments (latent variable)
        z = pm.Categorical('z',
                          p=pm.math.stack([w, 1-w]),
                          shape=n_groups)

        # Group-level probabilities
        theta = pm.Beta('theta',
                       alpha=alpha[z],
                       beta=beta_param[z],
                       shape=n_groups)

        # Likelihood
        y = pm.Binomial('y', n=n, p=theta, observed=r)

        # Derived quantities
        component_probs = pm.Deterministic('component_probs',
                                          pm.math.stack([w, 1-w]))

        # Component separation (for falsification)
        mu_diff = pm.Deterministic('mu_diff', mu[1] - mu[0])

        # Posterior predictive for new group (n=100)
        z_new = pm.Categorical('z_new', p=component_probs)
        theta_new = pm.Beta('theta_new',
                           alpha=alpha[z_new],
                           beta=beta_param[z_new])
        r_new = pm.Binomial('r_new', n=100, p=theta_new)

    return mixture_model


# ============================================================================
# MODEL 2: ROBUST HIERARCHICAL (STUDENT-T RANDOM EFFECTS)
# ============================================================================

def build_robust_hierarchical_model(n, r):
    """
    Robust hierarchical model with Student-t random effects.

    Core hypothesis: Population distribution has heavy tails;
    outliers are legitimate draws from tails rather than
    separate subpopulation.

    Parameters
    ----------
    n : array-like, shape (n_groups,)
        Sample sizes per group
    r : array-like, shape (n_groups,)
        Event counts per group

    Returns
    -------
    model : pm.Model
        PyMC model object

    Notes
    -----
    - Population mean: mu ~ Normal(logit(0.07), 1)
    - Population scale: tau ~ HalfCauchy(0, 0.5)
    - Tail weight: nu ~ Gamma(2, 0.1)
    - Falsification: Abandon if nu > 30 (normal sufficient)
    - Logit scale ensures 0 < theta < 1
    """
    n = np.asarray(n)
    r = np.asarray(r)
    n_groups = len(n)

    # Pooled estimate for centering
    pooled_rate = r.sum() / n.sum()

    with pm.Model() as robust_model:
        # Population parameters (on logit scale)
        mu = pm.Normal('mu',
                      mu=pm.math.log(pooled_rate / (1 - pooled_rate)),
                      sigma=1)
        tau = pm.HalfCauchy('tau', beta=0.5)

        # Degrees of freedom (tail weight)
        nu = pm.Gamma('nu', alpha=2, beta=0.1)

        # Group-level random effects (heavy-tailed)
        eta = pm.StudentT('eta', nu=nu, mu=0, sigma=1, shape=n_groups)

        # Group-level logits
        logit_theta = pm.Deterministic('logit_theta', mu + tau * eta)

        # Transform to probability scale
        theta = pm.Deterministic('theta', pm.math.invlogit(logit_theta))

        # Likelihood
        y = pm.Binomial('y', n=n, p=theta, observed=r)

        # Derived quantities
        # Population mean on probability scale
        pop_mean = pm.Deterministic('pop_mean', pm.math.invlogit(mu))

        # Between-group SD (approximation via delta method)
        pop_sd_approx = pm.Deterministic('pop_sd_approx',
                                         tau * pop_mean * (1 - pop_mean))

        # Shrinkage factor (for diagnostics)
        # Smaller shrinkage for heavy tails
        shrinkage = pm.Deterministic('shrinkage',
                                     1 / (1 + tau**2 * (1 + 1/nu)))

        # Posterior predictive for new group (n=100)
        eta_new = pm.StudentT('eta_new', nu=nu, mu=0, sigma=1)
        logit_theta_new = mu + tau * eta_new
        theta_new = pm.Deterministic('theta_new',
                                    pm.math.invlogit(logit_theta_new))
        r_new = pm.Binomial('r_new', n=100, p=theta_new)

    return robust_model


# ============================================================================
# MODEL 3: DIRICHLET PROCESS MIXTURE (NONPARAMETRIC)
# ============================================================================

def build_dp_mixture_model(n, r, K_max=10):
    """
    Dirichlet Process mixture model for binomial outcomes.

    Core hypothesis: Unknown number of latent subpopulations;
    let data determine the effective number of clusters.

    Parameters
    ----------
    n : array-like, shape (n_groups,)
        Sample sizes per group
    r : array-like, shape (n_groups,)
        Event counts per group
    K_max : int, default=10
        Truncation level for stick-breaking approximation

    Returns
    -------
    model : pm.Model
        PyMC model object

    Notes
    -----
    - DP concentration: alpha_dp ~ Gamma(2, 2)
    - Base distribution: Beta(7, 93) for means
    - Stick-breaking representation truncated at K_max
    - Falsification: Abandon if K_eff = 1 consistently
    - Most computationally expensive of the three models
    """
    n = np.asarray(n)
    r = np.asarray(r)
    n_groups = len(n)

    with pm.Model() as dp_model:
        # DP concentration parameter
        alpha_dp = pm.Gamma('alpha_dp', alpha=2, beta=2)

        # Base distribution parameters
        mu_0 = pm.Beta('mu_0', alpha=7, beta=93)
        kappa_0 = pm.Gamma('kappa_0', alpha=2, beta=0.1)

        # Component-specific parameters (draw from base distribution)
        mu_k = pm.Beta('mu_k', alpha=7, beta=93, shape=K_max)
        kappa_k = pm.Gamma('kappa_k', alpha=2, beta=0.1, shape=K_max)

        # Beta parameters for each component
        alpha_k = pm.Deterministic('alpha_k', mu_k * kappa_k)
        beta_k = pm.Deterministic('beta_k', (1 - mu_k) * kappa_k)

        # Stick-breaking weights
        v = pm.Beta('v', alpha=1, beta=alpha_dp, shape=K_max)

        # Convert to mixture weights via stick-breaking
        # pi_k = v_k * prod_{j<k}(1 - v_j)
        # Use log-space for numerical stability
        def stick_breaking(v):
            """Convert stick-breaking variables to mixture weights."""
            log_w = pm.math.log(v)
            log_1minus_w = pm.math.log(1 - v)
            # Cumulative product of (1 - v_j) for j < k
            log_cum_prod = pm.math.concatenate(
                [[0], pm.math.cumsum(log_1minus_w[:-1])]
            )
            log_pi = log_w + log_cum_prod
            return pm.math.exp(log_pi)

        pi = pm.Deterministic('pi', stick_breaking(v))

        # Component assignments
        z = pm.Categorical('z', p=pi, shape=n_groups)

        # Group-level probabilities
        theta = pm.Beta('theta',
                       alpha=alpha_k[z],
                       beta=beta_k[z],
                       shape=n_groups)

        # Likelihood
        y = pm.Binomial('y', n=n, p=theta, observed=r)

        # Derived quantities
        # Effective number of clusters (components with >5% mass)
        K_eff = pm.Deterministic('K_eff', pm.math.sum(pi > 0.05))

        # Posterior predictive for new group (n=100)
        z_new = pm.Categorical('z_new', p=pi)
        theta_new = pm.Beta('theta_new',
                           alpha=alpha_k[z_new],
                           beta=beta_k[z_new])
        r_new = pm.Binomial('r_new', n=100, p=theta_new)

    return dp_model


# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def fit_model(model, model_name='model', draws=2000, tune=1500,
              chains=4, target_accept=0.95, **kwargs):
    """
    Fit a PyMC model with appropriate MCMC settings.

    Parameters
    ----------
    model : pm.Model
        PyMC model to fit
    model_name : str
        Name for tracking/logging
    draws : int
        Number of posterior samples per chain
    tune : int
        Number of tuning/warmup samples
    chains : int
        Number of MCMC chains
    target_accept : float
        Target acceptance rate for NUTS
    **kwargs : dict
        Additional arguments to pm.sample()

    Returns
    -------
    trace : arviz.InferenceData
        Posterior samples and diagnostics

    Notes
    -----
    - Uses NUTS sampler (default in PyMC)
    - Returns InferenceData object (ArviZ format)
    - Automatically checks convergence diagnostics
    """
    print(f"Fitting {model_name}...")
    print(f"Settings: {draws} draws, {tune} tuning, {chains} chains")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42,
            **kwargs
        )

    # Check convergence
    print("\nConvergence diagnostics:")
    print(f"Max Rhat: {az.rhat(trace).max().to_array().values.max():.4f}")
    print(f"Min ESS: {az.ess(trace).min().to_array().values.min():.0f}")

    # Check for divergences
    divergences = trace.sample_stats.diverging.sum().item()
    if divergences > 0:
        print(f"WARNING: {divergences} divergent transitions detected!")

    return trace


def fit_mixture_model(model, **kwargs):
    """Fit finite mixture model with appropriate settings."""
    return fit_model(
        model,
        model_name='Finite Mixture (K=2)',
        draws=2000,
        tune=2000,  # Longer warmup for mixtures
        target_accept=0.95,
        **kwargs
    )


def fit_robust_model(model, **kwargs):
    """Fit robust hierarchical model with appropriate settings."""
    return fit_model(
        model,
        model_name='Robust Hierarchical (Student-t)',
        draws=2000,
        tune=1500,
        target_accept=0.95,
        **kwargs
    )


def fit_dp_model(model, **kwargs):
    """Fit Dirichlet Process model with appropriate settings."""
    return fit_model(
        model,
        model_name='Dirichlet Process Mixture',
        draws=3000,  # More samples for nonparametrics
        tune=2000,
        target_accept=0.95,
        init='adapt_diag',  # Better initialization
        **kwargs
    )


# ============================================================================
# PRIOR PREDICTIVE CHECK
# ============================================================================

def prior_predictive_check(model, n_samples=1000):
    """
    Generate prior predictive samples for validation.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    n_samples : int
        Number of prior predictive samples

    Returns
    -------
    prior : arviz.InferenceData
        Prior predictive samples

    Notes
    -----
    Use this to verify priors generate plausible data before fitting.
    """
    print("Generating prior predictive samples...")
    with model:
        prior = pm.sample_prior_predictive(samples=n_samples,
                                          random_seed=42)
    return prior


# ============================================================================
# POSTERIOR PREDICTIVE CHECK
# ============================================================================

def posterior_predictive_check(model, trace, n_samples=1000):
    """
    Generate posterior predictive samples for model checking.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    trace : arviz.InferenceData
        Posterior samples from pm.sample()
    n_samples : int
        Number of posterior predictive samples

    Returns
    -------
    trace_with_ppc : arviz.InferenceData
        Original trace with posterior_predictive group added

    Notes
    -----
    Use this to check if model can reproduce observed data patterns.
    """
    print("Generating posterior predictive samples...")
    with model:
        pm.sample_posterior_predictive(
            trace,
            var_names=['y', 'r_new'],
            extend_inferencedata=True,
            random_seed=42
        )
    return trace


# ============================================================================
# FALSIFICATION CHECKS
# ============================================================================

def check_mixture_falsification(trace):
    """
    Check falsification criteria for finite mixture model.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples

    Returns
    -------
    dict
        Falsification results

    Falsification criteria:
    1. w < 0.1 or w > 0.9 (one component nearly empty)
    2. mu_diff < 0.03 (components not distinct)
    """
    w_mean = trace.posterior['w'].mean().item()
    mu_diff_mean = trace.posterior['mu_diff'].mean().item()
    mu_diff_lower = trace.posterior['mu_diff'].quantile(0.025).item()

    results = {
        'mixture_weight_mean': w_mean,
        'component_separation_mean': mu_diff_mean,
        'component_separation_lower_CI': mu_diff_lower,
        'extreme_weight': (w_mean < 0.1) or (w_mean > 0.9),
        'insufficient_separation': mu_diff_lower < 0.03,
        'falsified': False
    }

    if results['extreme_weight']:
        results['falsified'] = True
        results['reason'] = f"Extreme mixture weight: w={w_mean:.3f}"
    elif results['insufficient_separation']:
        results['falsified'] = True
        results['reason'] = f"Components not distinct: mu_diff={mu_diff_mean:.3f}"

    return results


def check_robust_falsification(trace):
    """
    Check falsification criteria for robust hierarchical model.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples

    Returns
    -------
    dict
        Falsification results

    Falsification criteria:
    1. nu > 30 (normal sufficient, heavy tails not needed)
    2. nu < 2 (extreme heavy tails suggest misspecification)
    """
    nu_mean = trace.posterior['nu'].mean().item()
    nu_lower = trace.posterior['nu'].quantile(0.025).item()
    nu_upper = trace.posterior['nu'].quantile(0.975).item()

    results = {
        'nu_mean': nu_mean,
        'nu_CI': (nu_lower, nu_upper),
        'too_heavy': nu_upper < 2,
        'too_light': nu_lower > 30,
        'falsified': False
    }

    if results['too_light']:
        results['falsified'] = True
        results['reason'] = f"Normal sufficient: nu > 30 (mean={nu_mean:.1f})"
    elif results['too_heavy']:
        results['falsified'] = True
        results['reason'] = f"Extreme tails: nu < 2 (mean={nu_mean:.1f})"

    return results


def check_dp_falsification(trace):
    """
    Check falsification criteria for DP mixture model.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples

    Returns
    -------
    dict
        Falsification results

    Falsification criteria:
    1. K_eff = 1 consistently (no clustering detected)
    2. K_eff posterior uniform (cannot determine structure)
    """
    K_eff_samples = trace.posterior['K_eff'].values.flatten()
    K_eff_mean = K_eff_samples.mean()
    K_eff_mode = np.bincount(K_eff_samples.astype(int)).argmax()
    frac_K1 = (K_eff_samples == 1).mean()

    results = {
        'K_eff_mean': K_eff_mean,
        'K_eff_mode': K_eff_mode,
        'fraction_K1': frac_K1,
        'no_clustering': frac_K1 > 0.8,
        'falsified': False
    }

    if results['no_clustering']:
        results['falsified'] = True
        results['reason'] = f"No clustering: K=1 in {frac_K1:.1%} of samples"

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example workflow for fitting alternative models.
    """
    import pandas as pd

    # Load data
    data = pd.read_csv('/workspace/data/data.csv')
    n = data['n'].values
    r = data['r'].values

    print("Alternative Bayesian Models for Binomial Outcome Data")
    print("=" * 60)
    print(f"Data: {len(n)} groups, N={n.sum()}, total events={r.sum()}")
    print(f"Pooled rate: {r.sum()/n.sum():.4f}")
    print()

    # Model 1: Finite Mixture
    print("Building Finite Mixture Model (K=2)...")
    mixture_model = build_mixture_model(n, r)
    print(mixture_model)
    print()

    # Model 2: Robust Hierarchical
    print("Building Robust Hierarchical Model (Student-t)...")
    robust_model = build_robust_hierarchical_model(n, r)
    print(robust_model)
    print()

    # Model 3: DP Mixture
    print("Building Dirichlet Process Mixture Model...")
    dp_model = build_dp_mixture_model(n, r, K_max=8)
    print(dp_model)
    print()

    # Prior predictive checks (optional but recommended)
    print("Recommended: Run prior predictive checks before fitting:")
    print("  prior = prior_predictive_check(model)")
    print()

    # Fitting example
    print("To fit models:")
    print("  trace_mixture = fit_mixture_model(mixture_model)")
    print("  trace_robust = fit_robust_model(robust_model)")
    print("  trace_dp = fit_dp_model(dp_model)")
    print()

    # Falsification example
    print("After fitting, check falsification criteria:")
    print("  check_mixture_falsification(trace_mixture)")
    print("  check_robust_falsification(trace_robust)")
    print("  check_dp_falsification(trace_dp)")
