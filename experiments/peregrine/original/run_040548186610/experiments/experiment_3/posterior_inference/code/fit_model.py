#!/usr/local/bin/python3
"""
Fit Latent AR(1) Negative Binomial model using PyMC.

Adaptive sampling strategy:
1. Initial probe: 4 chains × 500 iterations (250 warmup)
2. If successful: Full sampling with 4 chains × 3000 iterations (1500 warmup)
3. Monitor divergences and adjust target_accept if needed
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import json
import warnings
from pathlib import Path

# Import model
import sys
sys.path.append('/workspace/experiments/experiment_3/posterior_inference/code')
from model import build_model, summary_info

# Paths
DATA_PATH = '/workspace/data/data.csv'
OUTPUT_DIR = Path('/workspace/experiments/experiment_3/posterior_inference')
DIAG_DIR = OUTPUT_DIR / 'diagnostics'
PLOTS_DIR = OUTPUT_DIR / 'plots'

# Suppress PyTensor compilation warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_data():
    """Load and prepare data"""
    df = pd.read_csv(DATA_PATH)
    return df['year'].values, df['C'].values.astype(int)


def run_probe_sampling(model):
    """
    Probe sampling: Quick diagnostic run to check model behavior.

    Returns
    -------
    idata : arviz.InferenceData
        Inference data from probe run
    success : bool
        Whether probe was successful
    """
    print("\n" + "="*70)
    print("PROBE SAMPLING: 4 chains × 500 iterations")
    print("="*70)

    with model:
        try:
            idata = pm.sample(
                draws=250,
                tune=250,
                chains=4,
                cores=4,
                target_accept=0.95,
                return_inferencedata=True,
                random_seed=42,
                idata_kwargs={'log_likelihood': True},
                progressbar=True
            )

            # Check for issues
            divergences = idata.sample_stats.diverging.sum().item()
            max_rhat = float(az.rhat(idata).max().to_array().max())

            print(f"\nProbe Results:")
            print(f"  Divergences: {divergences}")
            print(f"  Max R-hat: {max_rhat:.4f}")

            if divergences > 50:  # > 10% of 500 iterations
                print("  WARNING: High divergences in probe run")
                return idata, False
            elif max_rhat > 1.1:
                print("  WARNING: Poor convergence in probe run")
                return idata, False
            else:
                print("  Probe successful - proceeding to full sampling")
                return idata, True

        except Exception as e:
            print(f"  ERROR in probe sampling: {e}")
            import traceback
            traceback.print_exc()
            return None, False


def run_full_sampling(model, target_accept=0.95):
    """
    Full sampling run.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    target_accept : float
        Target acceptance probability

    Returns
    -------
    idata : arviz.InferenceData
        Inference data from full run
    """
    print("\n" + "="*70)
    print(f"FULL SAMPLING: 4 chains × 3000 iterations (target_accept={target_accept})")
    print("="*70)

    with model:
        idata = pm.sample(
            draws=1500,
            tune=1500,
            chains=4,
            cores=4,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={'log_likelihood': True},
            progressbar=True
        )

    return idata


def check_convergence(idata):
    """
    Check convergence diagnostics.

    Returns
    -------
    dict
        Convergence metrics and assessment
    """
    print("\n" + "="*70)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*70)

    # Get summary statistics
    summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'beta_2', 'rho', 'sigma_eta', 'phi'])

    # Extract key metrics
    max_rhat = summary['r_hat'].max()
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()

    # Count divergences
    divergences = int(idata.sample_stats.diverging.sum().item())
    total_draws = len(idata.posterior.chain) * len(idata.posterior.draw)
    div_pct = 100 * divergences / total_draws

    print(f"\nConvergence Metrics:")
    print(f"  Max R-hat: {max_rhat:.4f}")
    print(f"  Min ESS (bulk): {min_ess_bulk:.0f}")
    print(f"  Min ESS (tail): {min_ess_tail:.0f}")
    print(f"  Divergences: {divergences} ({div_pct:.1f}%)")

    # Assess convergence
    convergence_ok = True
    issues = []

    if max_rhat > 1.05:
        convergence_ok = False
        issues.append(f"R-hat too high: {max_rhat:.4f}")

    if min_ess_bulk < 200:
        convergence_ok = False
        issues.append(f"ESS too low: {min_ess_bulk:.0f}")

    if div_pct > 10:
        convergence_ok = False
        issues.append(f"Too many divergences: {div_pct:.1f}%")

    if convergence_ok:
        print("\n✓ CONVERGENCE CRITERIA MET")
    else:
        print("\n✗ CONVERGENCE ISSUES:")
        for issue in issues:
            print(f"  - {issue}")

    metrics = {
        'max_rhat': float(max_rhat),
        'min_ess_bulk': float(min_ess_bulk),
        'min_ess_tail': float(min_ess_tail),
        'divergences': divergences,
        'divergence_pct': float(div_pct),
        'convergence_ok': convergence_ok,
        'issues': issues
    }

    return metrics, summary


def main():
    """Main fitting routine"""
    print("\n" + "="*70)
    print("LATENT AR(1) NEGATIVE BINOMIAL MODEL - EXPERIMENT 3")
    print("="*70)

    # Load data
    print("\nLoading data...")
    year, C = load_data()
    print(f"  N = {len(year)} observations")
    print(f"  Year range: [{year.min():.2f}, {year.max():.2f}]")
    print(f"  Count range: [{C.min()}, {C.max()}]")

    # Build model
    print("\nBuilding model...")
    model = build_model(year, C)
    print(f"  Model: {summary_info()['name']}")
    print(f"  Parameters: {', '.join(summary_info()['parameters'])}")
    print(f"  Parameterization: {summary_info()['parameterization']}")

    # Probe sampling
    probe_idata, probe_success = run_probe_sampling(model)

    if not probe_success:
        print("\n⚠ Probe sampling had issues, trying full sampling with higher target_accept...")
        target_accept = 0.98
    else:
        target_accept = 0.95

    # Full sampling
    idata = run_full_sampling(model, target_accept=target_accept)

    # If still many divergences, try once more with target_accept=0.99
    divergences = int(idata.sample_stats.diverging.sum().item())
    total_draws = len(idata.posterior.chain) * len(idata.posterior.draw)
    div_pct = 100 * divergences / total_draws

    if div_pct > 10 and target_accept < 0.99:
        print(f"\n⚠ High divergences ({div_pct:.1f}%), retrying with target_accept=0.99...")
        idata = run_full_sampling(model, target_accept=0.99)

    # Check convergence
    metrics, summary = check_convergence(idata)

    # Save outputs
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    # Save InferenceData
    idata_path = DIAG_DIR / 'posterior_inference.netcdf'
    idata.to_netcdf(idata_path)
    print(f"  Saved: {idata_path}")

    # Save summary table
    summary_path = DIAG_DIR / 'summary_table.csv'
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path}")

    # Save convergence metrics
    metrics_path = DIAG_DIR / 'convergence_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    print("\n" + "="*70)
    print("FITTING COMPLETE")
    print("="*70)

    return idata, metrics


if __name__ == '__main__':
    idata, metrics = main()
