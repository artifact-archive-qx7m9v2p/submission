"""
Fit Hierarchical Logit-Normal Model using PyMC
Adaptive HMC sampling with full diagnostics
"""

import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load data
print("="*60)
print("LOADING DATA")
print("="*60)
data_path = "/workspace/data/data.csv"
df = pd.read_csv(data_path)
print(f"\nData shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Extract data
J = len(df)
n = df['n_trials'].values
r = df['r_successes'].values

print(f"\nNumber of groups (J): {J}")
print(f"Trial counts range: [{n.min()}, {n.max()}]")
print(f"Success counts range: [{r.min()}, {r.max()}]")
print(f"Observed success rates range: [{(r/n).min():.4f}, {(r/n).max():.4f}]")

# Build PyMC model
print("\n" + "="*60)
print("BUILDING PYMC MODEL")
print("="*60)

with pm.Model() as model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.6, sigma=1.0)
    tau = pm.HalfNormal('tau', sigma=0.5)

    # Non-centered parameterization
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    # Likelihood
    y = pm.Binomial('y', n=n, logit_p=theta, observed=r)

    # Derived quantities
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Posterior predictive
    y_pred = pm.Binomial('y_pred', n=n, logit_p=theta, shape=J)

print("\n✓ Model built successfully")
print(f"  - Hyperparameters: mu, tau")
print(f"  - Group parameters: theta[{J}], p[{J}]")
print(f"  - Likelihood: Binomial with logit link")

# Initial probe - quick assessment
print("\n" + "="*60)
print("PHASE 1: INITIAL PROBE (SHORT CHAINS)")
print("="*60)
print("Purpose: Quick assessment of model behavior")
print("Strategy: 4 chains, 200 iterations (100 warmup, 100 sampling)")
print("\nStarting probe sampling...")

with model:
    try:
        probe_trace = pm.sample(
            draws=100,
            tune=100,
            chains=4,
            cores=4,
            random_seed=RANDOM_SEED,
            target_accept=0.8,  # Default
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True}
        )

        print("\n✓ Probe sampling completed!")

        # Quick diagnostics
        print("\n" + "-"*60)
        print("PROBE DIAGNOSTICS")
        print("-"*60)

        summary = az.summary(probe_trace, var_names=['mu', 'tau'])
        print("\nHyperparameter summary:")
        print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'ess_tail', 'r_hat']])

        # Check for issues
        divergences = probe_trace.sample_stats.diverging.sum().item()
        max_rhat = float(summary['r_hat'].max())
        min_ess = float(summary['ess_bulk'].min())

        print(f"\nDivergent transitions: {divergences} ({divergences/(100*4)*100:.1f}%)")
        print(f"Max R-hat: {max_rhat:.4f}")
        print(f"Min ESS (bulk): {min_ess:.1f}")

        probe_success = divergences < 20 and max_rhat < 1.05  # Lenient for probe

        if probe_success:
            print("\n✓ Probe successful - proceeding to main sampling")
        else:
            print("\n! Probe shows issues - will increase adapt_delta for main sampling")

    except Exception as e:
        print(f"\n✗ Probe failed with error: {e}")
        print("This indicates serious model issues")
        probe_success = False

# Main sampling
if probe_success:
    print("\n" + "="*60)
    print("PHASE 2: MAIN SAMPLING")
    print("="*60)
    print("Strategy: 4 chains, 3000 iterations (1000 warmup, 2000 sampling)")

    # Adjust target_accept based on probe
    if divergences > 4:  # More than 1% in probe
        target_accept = 0.95
        print(f"Using increased adapt_delta = {target_accept} due to probe divergences")
    else:
        target_accept = 0.8
        print(f"Using default adapt_delta = {target_accept}")

    print("\nStarting main sampling...")

    with model:
        try:
            trace = pm.sample(
                draws=2000,
                tune=1000,
                chains=4,
                cores=4,
                random_seed=RANDOM_SEED,
                target_accept=target_accept,
                return_inferencedata=True,
                idata_kwargs={'log_likelihood': True}
            )

            # Add posterior predictive samples
            print("\nGenerating posterior predictive samples...")
            pm.sample_posterior_predictive(
                trace,
                extend_inferencedata=True,
                random_seed=RANDOM_SEED
            )

            print("\n✓ Main sampling completed successfully!")

        except Exception as e:
            print(f"\n✗ Main sampling failed: {e}")
            print("Attempting with increased adapt_delta...")

            with model:
                trace = pm.sample(
                    draws=2000,
                    tune=2000,  # More warmup
                    chains=4,
                    cores=4,
                    random_seed=RANDOM_SEED,
                    target_accept=0.99,  # Maximum
                    return_inferencedata=True,
                    idata_kwargs={'log_likelihood': True}
                )

                pm.sample_posterior_predictive(
                    trace,
                    extend_inferencedata=True,
                    random_seed=RANDOM_SEED
                )

                print("\n✓ Retry successful with adapt_delta=0.99")
else:
    print("\n" + "="*60)
    print("ATTEMPTING MAIN SAMPLING WITH HIGH ADAPT_DELTA")
    print("="*60)

    with model:
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            cores=4,
            random_seed=RANDOM_SEED,
            target_accept=0.99,
            return_inferencedata=True,
            idata_kwargs={'log_likelihood': True}
        )

        pm.sample_posterior_predictive(
            trace,
            extend_inferencedata=True,
            random_seed=RANDOM_SEED
        )

# Save InferenceData
print("\n" + "="*60)
print("SAVING INFERENCEDATA")
print("="*60)

output_path = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
trace.to_netcdf(output_path)
print(f"✓ Saved to: {output_path}")

# Verify log_likelihood is present
print("\nVerifying log_likelihood group:")
print(trace.log_likelihood)

# Full diagnostics
print("\n" + "="*60)
print("FULL MCMC DIAGNOSTICS")
print("="*60)

summary_all = az.summary(trace, var_names=['mu', 'tau', 'theta'], filter_vars="like")
print("\nParameter summary (selected):")
print(summary_all[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'ess_tail', 'r_hat']])

# Save full summary
summary_path = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/parameter_summary.csv"
summary_all.to_csv(summary_path)
print(f"\n✓ Full summary saved to: {summary_path}")

# Convergence checks
print("\n" + "-"*60)
print("CONVERGENCE CHECKS")
print("-"*60)

divergences_main = trace.sample_stats.diverging.sum().item()
total_samples = 2000 * 4
div_pct = divergences_main / total_samples * 100

max_rhat = float(summary_all['r_hat'].max())
min_ess_bulk = float(summary_all['ess_bulk'].min())
min_ess_tail = float(summary_all['ess_tail'].min())

print(f"\nDivergent transitions: {divergences_main} / {total_samples} ({div_pct:.2f}%)")
print(f"Max R-hat: {max_rhat:.5f}")
print(f"Min ESS (bulk): {min_ess_bulk:.1f}")
print(f"Min ESS (tail): {min_ess_tail:.1f}")

# Pass/fail criteria
print("\n" + "-"*60)
print("PASS/FAIL CRITERIA")
print("-"*60)

checks = {
    'R-hat < 1.01': max_rhat < 1.01,
    'ESS (bulk) > 400': min_ess_bulk > 400,
    'ESS (tail) > 400': min_ess_tail > 400,
    'Divergences < 1%': div_pct < 1.0,
    'Log-likelihood saved': 'log_likelihood' in trace.groups()
}

for check, passed in checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {check}")

overall_pass = all(checks.values())
print("\n" + "="*60)
if overall_pass:
    print("OVERALL: PASS")
else:
    print("OVERALL: FAIL")
    if not checks['R-hat < 1.01']:
        print("  → R-hat indicates non-convergence")
    if not checks['ESS (bulk) > 400'] or not checks['ESS (tail) > 400']:
        print("  → Low ESS indicates poor sampling efficiency")
    if not checks['Divergences < 1%']:
        print("  → High divergences indicate posterior geometry issues")
print("="*60)

print("\n✓ Fitting complete! Proceed to diagnostic plots.")
