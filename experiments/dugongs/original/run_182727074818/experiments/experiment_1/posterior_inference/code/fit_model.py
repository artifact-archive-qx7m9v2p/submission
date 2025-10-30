"""
Fit robust logarithmic regression model using HMC sampling with adaptive strategy.

Model: Y_i ~ StudentT(nu, mu_i, sigma) where mu_i = alpha + beta*log(x_i + c)
"""

import numpy as np
import pandas as pd
import cmdstanpy
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
print("ROBUST LOGARITHMIC REGRESSION - HMC FITTING")
print("=" * 80)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
N = len(df)
print(f"  - Observations: {N}")
print(f"  - x range: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
print(f"  - Y range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")

# Prepare Stan data
stan_data = {
    'N': N,
    'x': df['x'].values,
    'Y': df['Y'].values
}

# Save data for reproducibility
with open(DIAG_DIR / 'stan_data.json', 'w') as f:
    json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v
               for k, v in stan_data.items()}, f, indent=2)

# Compile model
print("\n[2/6] Compiling Stan model...")
model_path = CODE_DIR / "robust_log_regression.stan"
model = cmdstanpy.CmdStanModel(stan_file=str(model_path))
print(f"  - Model compiled successfully")

# Initialize parameters with informed values from EDA
init_values = {
    'alpha': 2.0,
    'beta': 0.3,
    'c': 1.0,
    'nu': 20.0,
    'sigma': 0.15
}

# Save initialization
with open(DIAG_DIR / 'init_values.json', 'w') as f:
    json.dump(init_values, f, indent=2)

print("\n[3/6] HMC Sampling - Initial Run...")
print("  Strategy: 4 chains × 2000 iterations (1000 warmup)")
print(f"  Initialization: {init_values}")

# Initial sampling attempt
try:
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=42,
        inits=init_values,
        adapt_delta=0.8,
        max_treedepth=10,
        show_console=True
    )

    # Check convergence
    print("\n[4/6] Convergence Diagnostics...")
    summary = fit.summary()

    # Save raw summary
    summary.to_csv(DIAG_DIR / 'parameter_summary.csv')

    # Check key metrics
    params_of_interest = ['alpha', 'beta', 'c', 'nu', 'sigma']
    param_summary = summary.loc[params_of_interest]

    max_rhat = param_summary['R_hat'].max()
    min_ess_bulk = param_summary['N_Eff'].min()

    print(f"\n  Convergence Metrics:")
    print(f"  - Max R_hat: {max_rhat:.4f} (target: < 1.01)")
    print(f"  - Min ESS_bulk: {min_ess_bulk:.0f} (target: > 400)")

    # Check diagnostics
    diagnostics = fit.diagnose()
    divergences = fit.num_divergences()
    max_treedepth_hits = fit.num_max_treedepth()

    print(f"  - Divergent transitions: {divergences} (target: < 5%)")
    print(f"  - Max treedepth hits: {max_treedepth_hits} (target: < 1%)")

    # Determine if we need to resample
    needs_resampling = False
    if max_rhat > 1.01:
        print("\n  WARNING: R_hat > 1.01 detected")
        needs_resampling = True
    if min_ess_bulk < 400:
        print("\n  WARNING: ESS_bulk < 400 detected")
        needs_resampling = True
    if divergences > 0.05 * 4000:  # 5% of total iterations
        print("\n  WARNING: >5% divergent transitions")
        needs_resampling = True

    if needs_resampling:
        print("\n[4b/6] Adaptive Resampling...")
        print("  Strategy: 4 chains × 3000 iterations (1500 warmup)")
        print("  Adaptation: adapt_delta = 0.95")

        fit = model.sample(
            data=stan_data,
            chains=4,
            parallel_chains=4,
            iter_warmup=1500,
            iter_sampling=1500,
            seed=42,
            inits=init_values,
            adapt_delta=0.95,
            max_treedepth=12,
            show_console=True
        )

        # Re-check diagnostics
        summary = fit.summary()
        summary.to_csv(DIAG_DIR / 'parameter_summary.csv')
        param_summary = summary.loc[params_of_interest]

        max_rhat = param_summary['R_hat'].max()
        min_ess_bulk = param_summary['N_Eff'].min()
        divergences = fit.num_divergences()

        print(f"\n  Updated Convergence Metrics:")
        print(f"  - Max R_hat: {max_rhat:.4f}")
        print(f"  - Min ESS_bulk: {min_ess_bulk:.0f}")
        print(f"  - Divergent transitions: {divergences}")

    print("\n[5/6] Creating ArviZ InferenceData...")

    # Convert to ArviZ with log_likelihood for LOO
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive={'y_rep': 'y_rep'},
        observed_data={'Y': stan_data['Y']},
        log_likelihood='log_lik',
        coords={'obs_id': np.arange(N)},
        dims={'Y': ['obs_id'], 'y_rep': ['obs_id'], 'log_lik': ['obs_id']}
    )

    # Save InferenceData
    idata_path = DIAG_DIR / 'posterior_inference.netcdf'
    idata.to_netcdf(idata_path)
    print(f"  - Saved to: {idata_path}")

    # Verify log_likelihood is present
    if 'log_likelihood' in idata.groups():
        print("  - log_likelihood group verified (ready for LOO-CV)")
    else:
        print("  - WARNING: log_likelihood group missing!")

    # Create comprehensive summary
    print("\n[6/6] Parameter Summary:")
    print("=" * 80)

    for param in params_of_interest:
        row = param_summary.loc[param]
        print(f"\n{param}:")
        print(f"  Mean: {row['Mean']:.4f} ± {row['StdDev']:.4f}")
        print(f"  95% CI: [{row['5%']:.4f}, {row['95%']:.4f}]")
        print(f"  R_hat: {row['R_hat']:.4f}")
        print(f"  ESS_bulk: {row['N_Eff']:.0f}")

    print("\n" + "=" * 80)
    print("SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Save success status
    with open(DIAG_DIR / 'convergence_diagnostics.txt', 'w') as f:
        f.write("HMC SAMPLING CONVERGENCE DIAGNOSTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Max R_hat: {max_rhat:.4f} (target: < 1.01)\n")
        f.write(f"Min ESS_bulk: {min_ess_bulk:.0f} (target: > 400)\n")
        f.write(f"Divergent transitions: {divergences}\n")
        f.write(f"Max treedepth hits: {max_treedepth_hits}\n")
        f.write(f"\nSTATUS: {'PASS' if max_rhat < 1.01 and min_ess_bulk > 400 else 'NEEDS REVIEW'}\n")

except Exception as e:
    print(f"\nERROR during sampling: {str(e)}")
    import traceback
    traceback.print_exc()

    # Save error details
    with open(DIAG_DIR / 'error_log.txt', 'w') as f:
        f.write(f"Error during HMC sampling:\n")
        f.write(f"{str(e)}\n\n")
        f.write(traceback.format_exc())

    raise
