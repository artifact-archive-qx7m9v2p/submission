"""
Detailed Diagnostics for AR(1) Identifiability Issues

This script investigates the beta_1 recovery failure:
1. Test multiple starting points
2. Examine parameter correlation structure
3. Profile likelihood to assess identifiability
4. Determine if this is fundamental model issue or optimization artifact
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = Path('/workspace/experiments/experiment_2/simulation_based_validation')
PLOT_DIR = OUTPUT_DIR / 'plots'

def neg_log_likelihood(params, year, log_C, regime_idx):
    """Negative log-likelihood for AR(1) model"""
    alpha, beta_1, beta_2, phi, sigma_1, sigma_2, sigma_3 = params
    sigma_regime = np.array([sigma_1, sigma_2, sigma_3])

    if np.abs(phi) >= 0.95 or np.any(sigma_regime <= 0):
        return 1e10

    n_obs = len(log_C)
    mu_trend = alpha + beta_1 * year + beta_2 * year**2

    ll = 0.0

    # First observation
    sigma_init = sigma_regime[regime_idx[0]] / np.sqrt(1 - phi**2)
    epsilon_0 = log_C[0] - mu_trend[0]
    ll += norm.logpdf(epsilon_0, loc=0, scale=sigma_init)

    # Subsequent observations
    epsilon_prev = log_C[0] - mu_trend[0]
    for t in range(1, n_obs):
        mu_t = mu_trend[t] + phi * epsilon_prev
        sigma_t = sigma_regime[regime_idx[t]]
        ll += norm.logpdf(log_C[t], loc=mu_t, scale=sigma_t)
        epsilon_prev = log_C[t] - mu_trend[t]

    return -ll


def test_multiple_inits(data, regime_idx, true_params, n_inits=10):
    """Test optimization from multiple starting points"""

    year = data['year'].values
    log_C = np.log(data['C'].values)

    results = []

    print("\nTesting optimization from multiple starting points...")
    print("="*70)

    # True params
    true_init = np.array([
        true_params['alpha'],
        true_params['beta_1'],
        true_params['beta_2'],
        true_params['phi'],
        true_params['sigma_regime'][0],
        true_params['sigma_regime'][1],
        true_params['sigma_regime'][2]
    ])

    # Test 1: Start from true parameters (with noise)
    for i in range(n_inits):
        print(f"\nInit {i+1}/{n_inits}:")

        if i == 0:
            # Start very close to truth
            init = true_init + np.random.normal(0, 0.01, size=7)
            init[4:] = np.abs(init[4:])
            print("  Starting near true parameters")
        elif i == 1:
            # Start from data-driven estimates
            # Simple linear fit on log scale
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(year, log_C)
            init = np.array([intercept, slope, 0.0, 0.5, 0.3, 0.3, 0.3])
            print("  Starting from OLS estimates (no AR)")
        elif i == 2:
            # Start with high phi (strong AR)
            init = np.array([4.0, 0.5, 0.0, 0.9, 0.3, 0.3, 0.3])
            print("  Starting with high phi (0.9)")
        elif i == 3:
            # Start with low phi (weak AR)
            init = np.array([4.0, 0.8, 0.0, 0.3, 0.3, 0.3, 0.3])
            print("  Starting with low phi (0.3)")
        else:
            # Random perturbations
            init = true_init + np.random.normal(0, 0.2, size=7)
            init[3] = np.clip(init[3], 0.1, 0.9)  # Keep phi reasonable
            init[4:] = np.abs(init[4:])
            print(f"  Starting from random perturbation")

        print(f"  Init: alpha={init[0]:.3f}, beta_1={init[1]:.3f}, phi={init[3]:.3f}")

        result = minimize(
            neg_log_likelihood,
            init,
            args=(year, log_C, regime_idx),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'disp': False}
        )

        if result.success:
            print(f"  SUCCESS: NLL={result.fun:.2f}")
            print(f"  Result: alpha={result.x[0]:.3f}, beta_1={result.x[1]:.3f}, phi={result.x[3]:.3f}")
        else:
            print(f"  FAILED: {result.message}")

        results.append({
            'init_id': i,
            'success': result.success,
            'nll': result.fun if result.success else np.inf,
            'alpha': result.x[0],
            'beta_1': result.x[1],
            'beta_2': result.x[2],
            'phi': result.x[3],
            'sigma_1': result.x[4],
            'sigma_2': result.x[5],
            'sigma_3': result.x[6]
        })

    results_df = pd.DataFrame(results)

    # Find best result
    best_idx = results_df['nll'].idxmin()
    best = results_df.iloc[best_idx]

    print("\n" + "="*70)
    print("BEST RESULT:")
    print(f"  Init #{best['init_id']}, NLL={best['nll']:.2f}")
    print(f"  alpha={best['alpha']:.3f} (true={true_params['alpha']:.3f})")
    print(f"  beta_1={best['beta_1']:.3f} (true={true_params['beta_1']:.3f})")
    print(f"  phi={best['phi']:.3f} (true={true_params['phi']:.3f})")
    print(f"  beta_1 error: {np.abs(best['beta_1']-true_params['beta_1'])/true_params['beta_1']*100:.1f}%")
    print("="*70)

    return results_df, best


def analyze_parameter_correlation(data, regime_idx, mle_params):
    """
    Analyze correlation structure between phi and beta_1
    via profile likelihood
    """

    year = data['year'].values
    log_C = np.log(data['C'].values)

    # Grid over phi and beta_1
    phi_grid = np.linspace(0.3, 0.95, 20)
    beta_1_grid = np.linspace(0.2, 1.2, 20)

    nll_surface = np.zeros((len(phi_grid), len(beta_1_grid)))

    print("\nComputing likelihood surface for phi vs beta_1...")

    for i, phi in enumerate(phi_grid):
        for j, beta_1 in enumerate(beta_1_grid):
            # Fix phi and beta_1, optimize over others
            def nll_fixed(other_params):
                alpha, beta_2, sigma_1, sigma_2, sigma_3 = other_params
                params = np.array([alpha, beta_1, beta_2, phi, sigma_1, sigma_2, sigma_3])
                return neg_log_likelihood(params, year, log_C, regime_idx)

            init_other = np.array([
                mle_params['alpha'],
                mle_params['beta_2'],
                mle_params['sigma_regime'][0],
                mle_params['sigma_regime'][1],
                mle_params['sigma_regime'][2]
            ])

            try:
                result = minimize(nll_fixed, init_other, method='Nelder-Mead',
                                options={'maxiter': 500, 'disp': False})
                nll_surface[i, j] = result.fun
            except:
                nll_surface[i, j] = np.nan

    return phi_grid, beta_1_grid, nll_surface


def plot_identifiability_diagnostics(results_df, true_params, phi_grid, beta_1_grid, nll_surface):
    """Plot identifiability diagnostics"""

    fig = plt.figure(figsize=(16, 6))

    # 1. Multiple init results
    ax1 = fig.add_subplot(131)

    successful = results_df[results_df['success']]

    ax1.scatter(successful['phi'], successful['beta_1'],
               c=successful['nll'], s=100, cmap='viridis',
               edgecolor='black', linewidth=1)
    ax1.scatter([true_params['phi']], [true_params['beta_1']],
               color='red', s=200, marker='*',
               edgecolor='black', linewidth=2,
               label='True values', zorder=10)

    ax1.set_xlabel(r'$\phi$ (AR coefficient)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'$\beta_1$ (linear trend)', fontsize=11, fontweight='bold')
    ax1.set_title('MLE Results from Different Inits\n(color = neg log-lik)',
                 fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2. Likelihood surface
    ax2 = fig.add_subplot(132)

    # Relative to best
    nll_min = np.nanmin(nll_surface)
    delta_nll = nll_surface - nll_min

    contour = ax2.contourf(beta_1_grid, phi_grid, delta_nll,
                           levels=15, cmap='viridis_r')

    # Add contour lines for chi-square thresholds
    # For 2 parameters, 95% CI is chi2(2) = 5.99, so delta_nll ~ 3
    ax2.contour(beta_1_grid, phi_grid, delta_nll,
               levels=[2, 4, 6], colors='white', linewidths=1.5,
               linestyles=['--', '-', ':'])

    ax2.scatter([true_params['beta_1']], [true_params['phi']],
               color='red', s=200, marker='*',
               edgecolor='white', linewidth=2, label='True values')

    ax2.set_xlabel(r'$\beta_1$ (linear trend)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'$\phi$ (AR coefficient)', fontsize=11, fontweight='bold')
    ax2.set_title('Profile Likelihood Surface\n(Δ NLL from minimum)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(contour, ax=ax2, label='Δ NLL')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, color='white', linewidth=0.5)

    # 3. Parameter trade-off
    ax3 = fig.add_subplot(133)

    # Show that high phi + low beta_1 can give similar fit to low phi + high beta_1
    successful_sorted = successful.sort_values('nll').head(5)

    ax3.scatter(successful_sorted['phi'], successful_sorted['beta_1'],
               s=200, c=range(len(successful_sorted)), cmap='coolwarm',
               edgecolor='black', linewidth=2)

    for idx, row in successful_sorted.iterrows():
        ax3.annotate(f"NLL={row['nll']:.1f}",
                    (row['phi'], row['beta_1']),
                    fontsize=8, ha='center')

    ax3.scatter([true_params['phi']], [true_params['beta_1']],
               color='red', s=300, marker='*',
               edgecolor='black', linewidth=2, label='True values', zorder=10)

    ax3.set_xlabel(r'$\phi$ (AR coefficient)', fontsize=11, fontweight='bold')
    ax3.set_ylabel(r'$\beta_1$ (linear trend)', fontsize=11, fontweight='bold')
    ax3.set_title('Parameter Trade-off\n(Top 5 fits by likelihood)',
                 fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'identifiability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {PLOT_DIR / 'identifiability_analysis.png'}")
    plt.close()


def main():
    """Run detailed diagnostics"""

    print("="*80)
    print("DETAILED IDENTIFIABILITY DIAGNOSTICS")
    print("="*80)

    # Load data
    data = pd.read_csv(OUTPUT_DIR / 'code' / 'synthetic_data.csv')

    regime_idx = np.concatenate([
        np.zeros(14, dtype=int),
        np.ones(13, dtype=int),
        np.full(13, 2, dtype=int)
    ])

    true_params = {
        'alpha': 4.3,
        'beta_1': 0.86,
        'beta_2': 0.05,
        'phi': 0.85,
        'sigma_regime': np.array([0.3, 0.4, 0.35])
    }

    # Test multiple inits
    results_df, best = test_multiple_inits(data, regime_idx, true_params, n_inits=10)

    # Get MLE params from best result
    mle_params = {
        'alpha': best['alpha'],
        'beta_1': best['beta_1'],
        'beta_2': best['beta_2'],
        'phi': best['phi'],
        'sigma_regime': np.array([best['sigma_1'], best['sigma_2'], best['sigma_3']])
    }

    # Analyze correlation
    phi_grid, beta_1_grid, nll_surface = analyze_parameter_correlation(data, regime_idx, mle_params)

    # Plot diagnostics
    plot_identifiability_diagnostics(results_df, true_params, phi_grid, beta_1_grid, nll_surface)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Check if identifiability is the issue
    successful = results_df[results_df['success']]
    beta_1_range = successful['beta_1'].max() - successful['beta_1'].min()
    phi_range = successful['phi'].max() - successful['phi'].min()
    nll_range = successful['nll'].max() - successful['nll'].min()

    print(f"\nAcross {len(successful)} successful optimizations:")
    print(f"  beta_1 range: {successful['beta_1'].min():.3f} to {successful['beta_1'].max():.3f} (spread={beta_1_range:.3f})")
    print(f"  phi range: {successful['phi'].min():.3f} to {successful['phi'].max():.3f} (spread={phi_range:.3f})")
    print(f"  NLL range: {successful['nll'].min():.2f} to {successful['nll'].max():.2f} (spread={nll_range:.2f})")

    if beta_1_range > 0.3 and nll_range < 2:
        print("\n⚠ WARNING: IDENTIFIABILITY ISSUE DETECTED")
        print("  - Wide range of beta_1 values give similar likelihoods")
        print("  - phi and beta_1 are confounded (AR vs trend)")
        print("  - This is a FUNDAMENTAL MODEL ISSUE, not optimization failure")
    else:
        print("\n✓ Parameters appear identifiable")
        print("  - Different inits converge to similar estimates")

    print("="*80)


if __name__ == '__main__':
    main()
