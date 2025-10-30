"""
Bayesian Model Implementation - Designer 2
Flexible & Adaptive Models (Piecewise, Spline, Mixture)

This script implements the three proposed model classes:
1. Change-Point Regression (piecewise linear)
2. Hierarchical B-Spline Regression (smooth nonlinear)
3. Mixture-of-Experts (soft regime transition)

Dependencies:
- cmdstanpy (for Stan interface)
- numpy, pandas (data handling)
- scipy (for B-spline basis generation)
- arviz (diagnostics and visualization)
- matplotlib (plotting)

Usage:
    python implementation_code.py --data_path /path/to/data.csv --model all
"""

import os
import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.stats import norm
import argparse


# ============================================================================
# Configuration
# ============================================================================

MODELS_DIR = "/workspace/experiments/designer_2/models"
RESULTS_DIR = "/workspace/experiments/designer_2/results"
FIGURES_DIR = "/workspace/experiments/designer_2/figures"

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Stan sampling parameters
CHAINS = 4
ITER_WARMUP = 2000
ITER_SAMPLING = 2000
ADAPT_DELTA = 0.9
MAX_TREEDEPTH = 12
SEED = 42


# ============================================================================
# Utility Functions
# ============================================================================

def generate_bspline_basis(x, n_knots=3, degree=3):
    """
    Generate B-spline basis matrix for given x values.

    Args:
        x: Array of predictor values
        n_knots: Number of interior knots (default 3)
        degree: Spline degree (default 3 for cubic)

    Returns:
        B: Basis matrix (N x K) where K = n_knots + degree + 1
        knots: Interior knot locations
    """
    # Place interior knots at quantiles
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
    interior_knots = np.quantile(x, quantiles)

    # Full knot sequence with boundary knots repeated (degree+1) times
    knots = np.concatenate([
        [x.min()] * (degree + 1),
        interior_knots,
        [x.max()] * (degree + 1)
    ])

    # Number of basis functions
    n_basis = len(knots) - degree - 1

    # Evaluate each basis function at x
    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        # Create B-spline for i-th basis function
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        bspline = BSpline(knots, coef, degree)
        B[:, i] = bspline(x)

    return B, interior_knots


def load_data(data_path):
    """Load and validate dataset."""
    data = pd.read_csv(data_path)

    # Validate required columns
    if 'x' not in data.columns or 'Y' not in data.columns:
        raise ValueError("Data must contain 'x' and 'Y' columns")

    # Sort by x for easier visualization
    data = data.sort_values('x').reset_index(drop=True)

    print(f"Loaded {len(data)} observations")
    print(f"x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
    print(f"Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")

    return data


def check_convergence(fit, model_name):
    """
    Check MCMC convergence diagnostics.

    Returns True if convergence is acceptable, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Convergence Diagnostics: {model_name}")
    print(f"{'='*60}")

    # Get summary
    summary = fit.summary()

    # Check R-hat
    rhat_max = summary['R_hat'].max()
    rhat_issues = (summary['R_hat'] > 1.01).sum()

    print(f"Max R-hat: {rhat_max:.4f}")
    print(f"Parameters with R-hat > 1.01: {rhat_issues}")

    # Check effective sample size
    ess_bulk_min = summary['N_Eff'].min()
    ess_issues = (summary['N_Eff'] < 400).sum()

    print(f"Min ESS: {ess_bulk_min:.0f}")
    print(f"Parameters with ESS < 400: {ess_issues}")

    # Check divergences
    divergences = fit.divergences
    print(f"Divergent transitions: {divergences}")

    # Overall assessment
    converged = (rhat_max < 1.01) and (ess_bulk_min > 400) and (divergences < 100)

    if converged:
        print("✓ Convergence PASSED")
    else:
        print("✗ Convergence FAILED - consider tuning")

    return converged


def posterior_predictive_check(fit, Y_obs, model_name):
    """
    Perform posterior predictive checks.

    Returns dictionary with PPC metrics.
    """
    print(f"\n{'='*60}")
    print(f"Posterior Predictive Checks: {model_name}")
    print(f"{'='*60}")

    # Extract posterior predictive samples
    Y_rep = fit.stan_variable('Y_rep')  # Shape: (n_draws, N)

    # Compute coverage
    Y_lower = np.percentile(Y_rep, 2.5, axis=0)
    Y_upper = np.percentile(Y_rep, 97.5, axis=0)
    coverage = np.mean((Y_obs >= Y_lower) & (Y_obs <= Y_upper))

    print(f"95% Posterior Predictive Coverage: {coverage:.2%}")
    print(f"Target: ~95%")

    # Compute posterior predictive R²
    Y_pred_mean = Y_rep.mean(axis=0)
    ss_res = np.sum((Y_obs - Y_pred_mean) ** 2)
    ss_tot = np.sum((Y_obs - Y_obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Posterior Predictive R²: {r2:.3f}")

    # Compute RMSE
    rmse = np.sqrt(np.mean((Y_obs - Y_pred_mean) ** 2))
    print(f"Posterior Predictive RMSE: {rmse:.3f}")

    return {
        'coverage': coverage,
        'r2': r2,
        'rmse': rmse,
        'Y_pred_mean': Y_pred_mean,
        'Y_lower': Y_lower,
        'Y_upper': Y_upper
    }


def compute_loo(fit, model_name):
    """
    Compute LOO-CV using PSIS.

    Returns LOO object and diagnostics.
    """
    print(f"\n{'='*60}")
    print(f"LOO Cross-Validation: {model_name}")
    print(f"{'='*60}")

    # Extract log-likelihood
    log_lik = fit.stan_variable('log_lik')  # Shape: (n_draws, N)

    # Convert to xarray format for ArviZ
    idata = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood={'Y': log_lik}
    )

    # Compute LOO
    loo = az.loo(idata, pointwise=True)

    print(f"LOO-ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
    print(f"LOO-IC: {loo.loo:.2f}")

    # Check Pareto-k diagnostics
    pareto_k = loo.pareto_k.values
    k_high = np.sum(pareto_k > 0.7)
    k_moderate = np.sum((pareto_k > 0.5) & (pareto_k <= 0.7))

    print(f"Pareto-k > 0.7 (problematic): {k_high}")
    print(f"Pareto-k ∈ (0.5, 0.7] (concerning): {k_moderate}")

    if k_high > 0:
        print("⚠ Warning: High Pareto-k indicates influential observations")

    return loo


# ============================================================================
# Model 1: Change-Point Regression
# ============================================================================

def fit_changepoint_model(x, Y):
    """
    Fit Bayesian change-point (piecewise linear) model.
    """
    print("\n" + "="*60)
    print("MODEL 1: Change-Point Regression")
    print("="*60)

    # Prepare data
    stan_data = {
        'N': len(x),
        'x': x,
        'Y': Y
    }

    # Compile model
    model_path = os.path.join(MODELS_DIR, "model1_changepoint.stan")
    model = cmdstanpy.CmdStanModel(stan_file=model_path)

    # Sample
    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        iter_warmup=ITER_WARMUP,
        iter_sampling=ITER_SAMPLING,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        seed=SEED,
        show_progress=True
    )

    # Diagnostics
    check_convergence(fit, "Change-Point")
    ppc = posterior_predictive_check(fit, Y, "Change-Point")
    loo = compute_loo(fit, "Change-Point")

    # Extract key parameters
    tau_samples = fit.stan_variable('tau')
    print(f"\nBreakpoint (tau) posterior:")
    print(f"  Median: {np.median(tau_samples):.2f}")
    print(f"  SD: {np.std(tau_samples):.2f}")
    print(f"  95% CI: [{np.percentile(tau_samples, 2.5):.2f}, "
          f"{np.percentile(tau_samples, 97.5):.2f}]")

    # Falsification check
    if np.std(tau_samples) > 5:
        print("\n⚠ FALSIFICATION WARNING: Breakpoint uncertainty SD > 5")
        print("   → Data may not support distinct regime shift")
        print("   → Consider switching to smooth model (Model 2)")

    return fit, ppc, loo


# ============================================================================
# Model 2: B-Spline Regression
# ============================================================================

def fit_spline_model(x, Y, n_knots=3):
    """
    Fit hierarchical B-spline regression model.
    """
    print("\n" + "="*60)
    print("MODEL 2: B-Spline Regression")
    print("="*60)

    # Generate B-spline basis
    B, interior_knots = generate_bspline_basis(x, n_knots=n_knots, degree=3)
    K = B.shape[1]

    print(f"Using {K} B-spline basis functions")
    print(f"Interior knots at: {interior_knots}")

    # Prepare data
    stan_data = {
        'N': len(x),
        'K': K,
        'Y': Y,
        'B': B
    }

    # Compile model
    model_path = os.path.join(MODELS_DIR, "model2_spline.stan")
    model = cmdstanpy.CmdStanModel(stan_file=model_path)

    # Sample
    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        iter_warmup=ITER_WARMUP,
        iter_sampling=ITER_SAMPLING,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        seed=SEED,
        show_progress=True
    )

    # Diagnostics
    check_convergence(fit, "B-Spline")
    ppc = posterior_predictive_check(fit, Y, "B-Spline")
    loo = compute_loo(fit, "B-Spline")

    # Extract smoothness parameter
    tau_samples = fit.stan_variable('tau')
    print(f"\nSmoothness parameter (tau) posterior:")
    print(f"  Median: {np.median(tau_samples):.3f}")
    print(f"  95% CI: [{np.percentile(tau_samples, 2.5):.3f}, "
          f"{np.percentile(tau_samples, 97.5):.3f}]")

    return fit, ppc, loo


# ============================================================================
# Model 3: Mixture-of-Experts
# ============================================================================

def fit_mixture_model(x, Y):
    """
    Fit mixture-of-experts model with gating network.
    """
    print("\n" + "="*60)
    print("MODEL 3: Mixture-of-Experts")
    print("="*60)

    # Prepare data
    stan_data = {
        'N': len(x),
        'x': x,
        'Y': Y
    }

    # Compile model
    model_path = os.path.join(MODELS_DIR, "model3_mixture.stan")
    model = cmdstanpy.CmdStanModel(stan_file=model_path)

    # Sample (may need higher adapt_delta)
    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        iter_warmup=ITER_WARMUP,
        iter_sampling=ITER_SAMPLING,
        adapt_delta=0.95,  # Higher for nonlinear model
        max_treedepth=MAX_TREEDEPTH,
        seed=SEED,
        show_progress=True
    )

    # Diagnostics
    check_convergence(fit, "Mixture")
    ppc = posterior_predictive_check(fit, Y, "Mixture")
    loo = compute_loo(fit, "Mixture")

    # Extract key parameters
    tau_eff_samples = fit.stan_variable('tau_eff')
    gamma1_samples = fit.stan_variable('gamma1')

    print(f"\nEffective breakpoint (tau_eff) posterior:")
    print(f"  Median: {np.median(tau_eff_samples):.2f}")
    print(f"  SD: {np.std(tau_eff_samples):.2f}")
    print(f"  95% CI: [{np.percentile(tau_eff_samples, 2.5):.2f}, "
          f"{np.percentile(tau_eff_samples, 97.5):.2f}]")

    print(f"\nTransition sharpness (gamma1) posterior:")
    print(f"  Median: {np.median(gamma1_samples):.3f}")
    print(f"  Interpretation: More negative → sharper transition")

    # Falsification check
    if np.std(tau_eff_samples) > 10:
        print("\n⚠ FALSIFICATION WARNING: Effective breakpoint SD > 10")
        print("   → Transition point unconstrained by data")
        print("   → Consider simpler parametric model")

    return fit, ppc, loo


# ============================================================================
# Visualization
# ============================================================================

def plot_model_comparison(data, fits_dict, ppcs_dict):
    """
    Create comprehensive comparison plot for all models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x_obs = data['x'].values
    Y_obs = data['Y'].values

    # Generate smooth prediction grid
    x_pred = np.linspace(x_obs.min(), x_obs.max(), 200)

    for idx, (name, fit) in enumerate(fits_dict.items()):
        ax = axes.flat[idx]

        # Plot observed data
        ax.scatter(x_obs, Y_obs, c='black', s=50, alpha=0.6, label='Observed')

        # Plot posterior predictive mean and credible bands
        ppc = ppcs_dict[name]
        ax.plot(x_obs, ppc['Y_pred_mean'], 'b-', linewidth=2, label='Posterior Mean')
        ax.fill_between(x_obs, ppc['Y_lower'], ppc['Y_upper'],
                        alpha=0.3, color='blue', label='95% CI')

        # Add R² to title
        ax.set_title(f"{name}\nR² = {ppc['r2']:.3f}", fontsize=12, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    # Residual plots
    for idx, (name, fit) in enumerate(fits_dict.items()):
        ax = axes.flat[idx + 3]

        ppc = ppcs_dict[name]
        residuals = Y_obs - ppc['Y_pred_mean']

        ax.scatter(ppc['Y_pred_mean'], residuals, c='black', s=50, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"{name} - Residuals", fontsize=12)
        ax.set_xlabel('Predicted Y', fontsize=10)
        ax.set_ylabel('Residual', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {FIGURES_DIR}/model_comparison.png")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fit Bayesian models')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV data file (must have x and Y columns)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'changepoint', 'spline', 'mixture'],
                       help='Which model(s) to fit')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data = load_data(args.data_path)
    x = data['x'].values
    Y = data['Y'].values

    # Fit models
    fits = {}
    ppcs = {}
    loos = {}

    if args.model in ['all', 'changepoint']:
        fit, ppc, loo = fit_changepoint_model(x, Y)
        fits['Change-Point'] = fit
        ppcs['Change-Point'] = ppc
        loos['Change-Point'] = loo

    if args.model in ['all', 'spline']:
        fit, ppc, loo = fit_spline_model(x, Y, n_knots=3)
        fits['B-Spline'] = fit
        ppcs['B-Spline'] = ppc
        loos['B-Spline'] = loo

    if args.model in ['all', 'mixture']:
        fit, ppc, loo = fit_mixture_model(x, Y)
        fits['Mixture'] = fit
        ppcs['Mixture'] = ppc
        loos['Mixture'] = loo

    # Model comparison
    if len(fits) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Compare LOO-ELPD
        print("\nLOO-ELPD Comparison:")
        loo_values = {name: loo.elpd_loo for name, loo in loos.items()}
        loo_ses = {name: loo.se for name, loo in loos.items()}

        best_model = max(loo_values, key=loo_values.get)

        for name in loo_values:
            delta = loo_values[name] - loo_values[best_model]
            se_diff = np.sqrt(loo_ses[name]**2 + loo_ses[best_model]**2)

            if name == best_model:
                print(f"  {name}: {loo_values[name]:.2f} ± {loo_ses[name]:.2f} (BEST)")
            else:
                print(f"  {name}: {loo_values[name]:.2f} ± {loo_ses[name]:.2f} "
                      f"(Δ = {delta:.2f} ± {se_diff:.2f})")

        # Visualize
        plot_model_comparison(data, fits, ppcs)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
