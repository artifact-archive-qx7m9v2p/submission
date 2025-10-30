"""
Fit robust logarithmic regression model using HMC sampling with PyMC.

Model: Y_i ~ StudentT(nu, mu_i, sigma) where mu_i = alpha + beta*log(x_i + c)

Note: Using PyMC as fallback since CmdStan compilation requires make/g++ which are not available.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
CODE_DIR = BASE_DIR / "code"
DIAG_DIR = BASE_DIR / "diagnostics"
PLOT_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Create directories
for dir_path in [CODE_DIR, DIAG_DIR, PLOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ROBUST LOGARITHMIC REGRESSION - HMC FITTING (PyMC)")
print("=" * 80)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
N = len(df)
x_data = df['x'].values
Y_data = df['Y'].values

print(f"  - Observations: {N}")
print(f"  - x range: [{x_data.min():.1f}, {x_data.max():.1f}]")
print(f"  - Y range: [{Y_data.min():.2f}, {Y_data.max():.2f}]")

# Save data for reproducibility
data_dict = {
    'N': int(N),
    'x': x_data.tolist(),
    'Y': Y_data.tolist()
}

with open(DIAG_DIR / 'stan_data.json', 'w') as f:
    json.dump(data_dict, f, indent=2)

# Build PyMC model
print("\n[2/6] Building PyMC model...")

with pm.Model() as model:
    # Priors (validated via PPC and SBC)
    alpha = pm.Normal('alpha', mu=2.0, sigma=0.5)
    beta = pm.Normal('beta', mu=0.3, sigma=0.2)
    c = pm.Gamma('c', alpha=2, beta=2)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.15)

    # Mean function
    mu = alpha + beta * pm.math.log(x_data + c)

    # Likelihood
    Y_obs = pm.StudentT('Y_obs', nu=nu, mu=mu, sigma=sigma, observed=Y_data)

    # Posterior predictive
    Y_pred = pm.StudentT('Y_pred', nu=nu, mu=mu, sigma=sigma, shape=N)

print("  - Model built successfully")
print(f"  - Parameters: alpha, beta, c, nu, sigma")
print(f"  - Likelihood: StudentT(nu, mu, sigma)")

# Initial sampling attempt
print("\n[3/6] HMC Sampling - Initial Run...")
print("  Strategy: 4 chains × 2000 iterations (1000 warmup)")

try:
    with model:
        # Sample with NUTS
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=42,
            target_accept=0.8,
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True}
        )

        # Sample posterior predictive
        print("\n  - Sampling posterior predictive...")
        pm.sample_posterior_predictive(
            idata,
            extend_inferencedata=True,
            random_seed=42
        )

    print("\n[4/6] Convergence Diagnostics...")

    # Get summary statistics
    summary = az.summary(
        idata,
        var_names=['alpha', 'beta', 'c', 'nu', 'sigma'],
        round_to=4
    )

    # Save summary
    summary.to_csv(DIAG_DIR / 'parameter_summary.csv')

    # Extract key metrics
    max_rhat = summary['r_hat'].max()
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()

    print(f"\n  Convergence Metrics:")
    print(f"  - Max R_hat: {max_rhat:.4f} (target: < 1.01)")
    print(f"  - Min ESS_bulk: {min_ess_bulk:.0f} (target: > 400)")
    print(f"  - Min ESS_tail: {min_ess_tail:.0f} (target: > 400)")

    # Check for divergences
    divergences = idata.sample_stats['diverging'].sum().item()
    n_samples = len(idata.posterior.chain) * len(idata.posterior.draw)
    div_pct = 100 * divergences / n_samples

    print(f"  - Divergent transitions: {divergences} ({div_pct:.2f}%)")

    # Determine if we need to resample
    needs_resampling = False
    if max_rhat > 1.01:
        print("\n  WARNING: R_hat > 1.01 detected")
        needs_resampling = True
    if min_ess_bulk < 400:
        print("\n  WARNING: ESS_bulk < 400 detected")
        needs_resampling = True
    if div_pct > 5:
        print("\n  WARNING: >5% divergent transitions")
        needs_resampling = True

    if needs_resampling:
        print("\n[4b/6] Adaptive Resampling...")
        print("  Strategy: 4 chains × 3000 iterations (1500 warmup)")
        print("  Adaptation: target_accept = 0.95")

        with model:
            idata = pm.sample(
                draws=1500,
                tune=1500,
                chains=4,
                cores=4,
                random_seed=42,
                target_accept=0.95,
                return_inferencedata=True,
                idata_kwargs={'log_likelihood': True}
            )

            # Sample posterior predictive
            pm.sample_posterior_predictive(
                idata,
                extend_inferencedata=True,
                random_seed=42
            )

        # Re-check diagnostics
        summary = az.summary(
            idata,
            var_names=['alpha', 'beta', 'c', 'nu', 'sigma'],
            round_to=4
        )
        summary.to_csv(DIAG_DIR / 'parameter_summary.csv')

        max_rhat = summary['r_hat'].max()
        min_ess_bulk = summary['ess_bulk'].min()
        min_ess_tail = summary['ess_tail'].min()
        divergences = idata.sample_stats['diverging'].sum().item()
        n_samples = len(idata.posterior.chain) * len(idata.posterior.draw)
        div_pct = 100 * divergences / n_samples

        print(f"\n  Updated Convergence Metrics:")
        print(f"  - Max R_hat: {max_rhat:.4f}")
        print(f"  - Min ESS_bulk: {min_ess_bulk:.0f}")
        print(f"  - Min ESS_tail: {min_ess_tail:.0f}")
        print(f"  - Divergent transitions: {divergences} ({div_pct:.2f}%)")

    print("\n[5/6] Saving ArviZ InferenceData...")

    # Save InferenceData (observed data already included by PyMC)
    idata_path = DIAG_DIR / 'posterior_inference.netcdf'
    idata.to_netcdf(idata_path)
    print(f"  - Saved to: {idata_path}")

    # Verify log_likelihood is present
    if 'log_likelihood' in idata.groups():
        print("  - log_likelihood group verified (ready for LOO-CV)")
        print(f"  - log_likelihood shape: {idata.log_likelihood['Y_obs'].shape}")
    else:
        print("  - WARNING: log_likelihood group missing!")

    # Verify observed_data
    if 'observed_data' in idata.groups():
        print(f"  - observed_data verified (shape: {idata.observed_data['Y_obs'].shape})")

    # Create comprehensive summary
    print("\n[6/6] Parameter Summary:")
    print("=" * 80)

    for param in ['alpha', 'beta', 'c', 'nu', 'sigma']:
        row = summary.loc[param]
        print(f"\n{param}:")
        print(f"  Mean: {row['mean']:.4f} ± {row['sd']:.4f}")
        print(f"  95% HDI: [{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")
        print(f"  R_hat: {row['r_hat']:.4f}")
        print(f"  ESS_bulk: {row['ess_bulk']:.0f}")
        print(f"  ESS_tail: {row['ess_tail']:.0f}")

    print("\n" + "=" * 80)
    print("SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Save convergence diagnostics
    with open(DIAG_DIR / 'convergence_diagnostics.txt', 'w') as f:
        f.write("HMC SAMPLING CONVERGENCE DIAGNOSTICS (PyMC)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Max R_hat: {max_rhat:.4f} (target: < 1.01)\n")
        f.write(f"Min ESS_bulk: {min_ess_bulk:.0f} (target: > 400)\n")
        f.write(f"Min ESS_tail: {min_ess_tail:.0f} (target: > 400)\n")
        f.write(f"Divergent transitions: {divergences} ({div_pct:.2f}%)\n")
        f.write(f"\nTotal posterior samples: {n_samples}\n")
        f.write(f"\nSTATUS: {'PASS' if max_rhat < 1.01 and min_ess_bulk > 400 and div_pct < 5 else 'NEEDS REVIEW'}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("PARAMETER SUMMARY:\n")
        f.write(summary.to_string())

    # Return success
    exit(0)

except Exception as e:
    print(f"\nERROR during sampling: {str(e)}")
    import traceback
    traceback.print_exc()

    # Save error details
    with open(DIAG_DIR / 'error_log.txt', 'w') as f:
        f.write(f"Error during HMC sampling (PyMC):\n")
        f.write(f"{str(e)}\n\n")
        f.write(traceback.format_exc())

    exit(1)
