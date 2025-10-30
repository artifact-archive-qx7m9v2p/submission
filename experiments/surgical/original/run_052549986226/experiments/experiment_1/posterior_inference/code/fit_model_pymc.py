"""
Fit Beta-Binomial Model to Real Data using PyMC
Experiment 1: Posterior Inference with HMC Sampling
Fallback from CmdStanPy due to Stan compiler unavailability
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import json
from pathlib import Path

# Set random seed
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
RESULTS_DIR = OUTPUT_DIR / "results"

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data)} groups")
print(f"Total trials: {data['n_trials'].sum()}")
print(f"Total successes: {data['r_successes'].sum()}")
print(f"Pooled rate: {data['r_successes'].sum() / data['n_trials'].sum():.4f}")

# Extract data
n_trials = data['n_trials'].values
r_success = data['r_successes'].values
N = len(data)

print("\n" + "="*70)
print("BUILDING PYMC MODEL")
print("="*70)

# Build PyMC model
with pm.Model() as model:
    # Priors (matching Stan model)
    mu = pm.Beta('mu', alpha=2, beta=18)  # Population mean
    kappa = pm.Gamma('kappa', alpha=2, beta=0.1)  # Concentration parameter

    # Transformed parameters (matching Stan)
    alpha = pm.Deterministic('alpha', mu * kappa)
    beta_param = pm.Deterministic('beta', (1 - mu) * kappa)

    # Likelihood - BetaBinomial
    r_obs = pm.BetaBinomial('r_obs', alpha=alpha, beta=beta_param, n=n_trials, observed=r_success)

    # Generated quantities (matching Stan model)
    var_p = pm.Deterministic('var_p', (mu * (1 - mu)) / (kappa + 1))
    phi = pm.Deterministic('phi', 1 + 1 / kappa)
    icc = pm.Deterministic('icc', 1 / (1 + kappa))

    # Posterior predictive for group-level rates
    p_rep = pm.Beta('p_rep', alpha=alpha, beta=beta_param, shape=N)

    # Posterior predictive for data (replicate observations)
    # Draw p_i for each group, then generate r_i
    p_i_samples = pm.Beta('p_i_posterior', alpha=alpha, beta=beta_param, shape=N)
    r_rep = pm.Binomial('r_rep', n=n_trials, p=p_i_samples, shape=N)

    # Posterior mean estimates for each group (empirical Bayes)
    p_posterior_mean = pm.Deterministic('p_posterior_mean',
                                       (r_success + alpha) / (n_trials + alpha + beta_param))

    # Shrinkage factor
    shrinkage_factor = pm.Deterministic('shrinkage_factor',
                                        kappa / (kappa + n_trials))

print("Model built successfully")
print(f"Model variables: {[v.name for v in model.basic_RVs]}")

print("\n" + "="*70)
print("MODEL SPECIFICATION SUMMARY")
print("="*70)
print("\nPriors:")
print("  mu ~ Beta(2, 18)      # Population mean, prior mean = 0.1")
print("  kappa ~ Gamma(2, 0.1) # Concentration, prior mean = 20")
print("\nLikelihood:")
print("  r_i ~ BetaBinomial(n_i, alpha, beta)")
print("  where alpha = mu * kappa, beta = (1 - mu) * kappa")
print("\nGenerated Quantities:")
print("  phi = 1 + 1/kappa     # Overdispersion parameter")
print("  var_p = mu(1-mu)/(kappa+1)  # Variance of success probabilities")
print("  icc = 1/(1+kappa)     # Intraclass correlation")

print("\n" + "="*70)
print("FITTING MODEL - INITIAL PROBE (200 iterations)")
print("="*70)
print("Strategy: Short run to diagnose potential issues")
print("Chains: 4")
print("Tune: 100")
print("Draws: 100")
print("target_accept: 0.95")
print("="*70 + "\n")

# Initial probe run
with model:
    try:
        trace_probe = pm.sample(
            draws=100,
            tune=100,
            chains=4,
            cores=4,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True,
            idata_kwargs={
                'log_likelihood': True
            }
        )

        print("\n" + "="*70)
        print("PROBE RUN - SUMMARY")
        print("="*70)

        # Check probe diagnostics
        probe_summary = az.summary(trace_probe, var_names=['mu', 'kappa', 'phi'])
        print("\nKey parameters from probe:")
        print(probe_summary)

        # Check for divergences
        probe_divergences = trace_probe.sample_stats.diverging.sum().item()
        total_probe_samples = 4 * 100
        print(f"\nDivergent transitions: {probe_divergences} (out of {total_probe_samples})")
        print(f"Divergence rate: {100 * probe_divergences / total_probe_samples:.2f}%")

        # Check Rhat
        probe_max_rhat = probe_summary['r_hat'].max()
        print(f"Max R_hat: {probe_max_rhat:.4f}")

        # Check ESS
        probe_min_ess = probe_summary['ess_bulk'].min()
        print(f"Min ESS (bulk): {probe_min_ess:.0f}")

        # Assess probe results
        probe_issues = []
        if probe_divergences > 0:
            probe_issues.append(f"Divergences detected: {probe_divergences}")
        if probe_max_rhat > 1.01:
            probe_issues.append(f"High R_hat: {probe_max_rhat:.4f}")
        if probe_min_ess < 40:  # Less than 10 per chain
            probe_issues.append(f"Low ESS: {probe_min_ess:.0f}")

        if probe_issues:
            print("\n[WARNING] Probe detected issues:")
            for issue in probe_issues:
                print(f"  - {issue}")
            print("\nAdjusting sampling parameters for main run...")
            target_accept = 0.98
            tune = 1500
            draws = 1000
        else:
            print("\n[SUCCESS] Probe run successful - no issues detected")
            print("Proceeding with standard sampling parameters...")
            target_accept = 0.95
            tune = 1000
            draws = 1000

    except Exception as e:
        print(f"\n[ERROR] Probe run failed: {e}")
        print("Attempting main run with conservative parameters...")
        target_accept = 0.98
        tune = 1500
        draws = 1000

print("\n" + "="*70)
print("FITTING MODEL - MAIN RUN")
print("="*70)
print(f"Chains: 4")
print(f"Tune: {tune}")
print(f"Draws: {draws}")
print(f"target_accept: {target_accept}")
print("="*70 + "\n")

# Main sampling run
with model:
    trace = pm.sample(
        draws=draws,
        tune=tune,
        chains=4,
        cores=4,
        target_accept=target_accept,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={
            'log_likelihood': True
        }
    )

    # Sample posterior predictive
    print("\n" + "="*70)
    print("SAMPLING POSTERIOR PREDICTIVE")
    print("="*70)

    trace.extend(pm.sample_posterior_predictive(trace, random_seed=42))

print("\n" + "="*70)
print("SAMPLING COMPLETE - COMPUTING DIAGNOSTICS")
print("="*70)

# Save InferenceData
netcdf_path = DIAGNOSTICS_DIR / 'posterior_inference.netcdf'
trace.to_netcdf(netcdf_path)
print(f"\nInferenceData saved to {netcdf_path}")

# Get summary
print("\nComputing ArviZ summary...")
summary = az.summary(trace, var_names=['mu', 'kappa', 'phi', 'alpha', 'beta', 'var_p', 'icc'])
print("\nPosterior Summary (Key Parameters):")
print(summary)

# Save full summary
summary_full = az.summary(trace)
summary_full.to_csv(DIAGNOSTICS_DIR / 'arviz_summary.csv')
print(f"\nFull ArviZ summary saved to {DIAGNOSTICS_DIR / 'arviz_summary.csv'}")

# Diagnostics
print("\n" + "="*70)
print("CONVERGENCE DIAGNOSTICS")
print("="*70)

divergences = trace.sample_stats.diverging.sum().item()
total_samples = 4 * draws
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()

print(f"\nDivergent transitions: {divergences} (out of {total_samples})")
print(f"Divergence rate: {100 * divergences / total_samples:.2f}%")
print(f"Max R_hat: {max_rhat:.4f}")
print(f"Min ESS (bulk): {min_ess_bulk:.0f}")
print(f"Min ESS (tail): {min_ess_tail:.0f}")

# Check convergence criteria
print("\n" + "="*70)
print("CONVERGENCE CRITERIA CHECK")
print("="*70)

criteria = {
    'All R_hat < 1.01': max_rhat < 1.01,
    'All ESS_bulk > 400': min_ess_bulk > 400,
    'All ESS_tail > 400': min_ess_tail > 400,
    'Divergences < 1%': (divergences / total_samples) < 0.01
}

all_pass = all(criteria.values())

for criterion, passes in criteria.items():
    status = "PASS" if passes else "FAIL"
    print(f"{criterion}: {status}")

print("\n" + "="*70)
if all_pass:
    print("OVERALL: PASS - All convergence criteria met")
else:
    print("OVERALL: FAIL - Some convergence criteria not met")
print("="*70)

# Extract posterior samples for analysis
print("\n" + "="*70)
print("EXTRACTING POSTERIOR SAMPLES")
print("="*70)

# Get posterior samples
posterior = trace.posterior

# Save key posterior samples as dataframe
posterior_samples = pd.DataFrame({
    'mu': posterior['mu'].values.flatten(),
    'kappa': posterior['kappa'].values.flatten(),
    'phi': posterior['phi'].values.flatten(),
    'alpha': posterior['alpha'].values.flatten(),
    'beta': posterior['beta'].values.flatten(),
    'var_p': posterior['var_p'].values.flatten(),
    'icc': posterior['icc'].values.flatten()
})

posterior_samples.to_csv(RESULTS_DIR / 'posterior_samples_scalar.csv', index=False)
print(f"Scalar posterior samples saved to {RESULTS_DIR / 'posterior_samples_scalar.csv'}")

# Save group-level posteriors
p_posterior_mean_samples = posterior['p_posterior_mean'].values  # Shape: (chains, draws, groups)
p_posterior_mean_flat = p_posterior_mean_samples.reshape(-1, N)  # Flatten chains and draws

group_posteriors = pd.DataFrame(
    p_posterior_mean_flat,
    columns=[f'group_{i}' for i in range(1, N + 1)]
)
group_posteriors.to_csv(RESULTS_DIR / 'posterior_group_means.csv', index=False)
print(f"Group-level posteriors saved to {RESULTS_DIR / 'posterior_group_means.csv'}")

# Compute and save posterior summaries for groups
print("\n" + "="*70)
print("GROUP-LEVEL POSTERIOR SUMMARIES")
print("="*70)

group_summary = []
mu_post_mean = posterior['mu'].values.mean()

for i in range(N):
    obs_rate = r_success[i] / n_trials[i]
    post_mean = p_posterior_mean_flat[:, i].mean()
    post_sd = p_posterior_mean_flat[:, i].std()
    post_q025 = np.percentile(p_posterior_mean_flat[:, i], 2.5)
    post_q975 = np.percentile(p_posterior_mean_flat[:, i], 97.5)

    # Compute shrinkage
    if obs_rate != mu_post_mean:
        shrinkage_pct = 100 * (obs_rate - post_mean) / (obs_rate - mu_post_mean)
    else:
        shrinkage_pct = 0.0

    group_summary.append({
        'group': i + 1,
        'n_trials': n_trials[i],
        'r_successes': r_success[i],
        'observed_rate': obs_rate,
        'posterior_mean': post_mean,
        'posterior_sd': post_sd,
        'posterior_025': post_q025,
        'posterior_975': post_q975,
        'shrinkage_pct': shrinkage_pct
    })

group_summary_df = pd.DataFrame(group_summary)
group_summary_df.to_csv(RESULTS_DIR / 'group_posterior_summary.csv', index=False)

print("\nGroup-level summaries:")
print(group_summary_df.to_string(index=False))
print(f"\nSaved to {RESULTS_DIR / 'group_posterior_summary.csv'}")

# Final summary
print("\n" + "="*70)
print("FITTING COMPLETE")
print("="*70)

mu_mean = posterior['mu'].values.mean()
mu_025 = np.percentile(posterior['mu'].values, 2.5)
mu_975 = np.percentile(posterior['mu'].values, 97.5)

kappa_mean = posterior['kappa'].values.mean()
kappa_025 = np.percentile(posterior['kappa'].values, 2.5)
kappa_975 = np.percentile(posterior['kappa'].values, 97.5)

phi_mean = posterior['phi'].values.mean()
phi_025 = np.percentile(posterior['phi'].values, 2.5)
phi_975 = np.percentile(posterior['phi'].values, 97.5)

print(f"\nPosterior Mean Estimates:")
print(f"  mu: {mu_mean:.4f} (95% CI: [{mu_025:.4f}, {mu_975:.4f}])")
print(f"  kappa: {kappa_mean:.2f} (95% CI: [{kappa_025:.2f}, {kappa_975:.2f}])")
print(f"  phi: {phi_mean:.4f} (95% CI: [{phi_025:.4f}, {phi_975:.4f}])")

if all_pass:
    print("\nModel fitting SUCCESSFUL - proceed to visualization and posterior predictive checking")
else:
    print("\nModel fitting had convergence issues - review diagnostics before proceeding")

print("\nOutput files:")
print(f"  - ArviZ InferenceData: {netcdf_path}")
print(f"  - Posterior summaries: {RESULTS_DIR}")

# Create a summary report file
print("\n" + "="*70)
print("CREATING CONVERGENCE REPORT")
print("="*70)

convergence_report = f"""# Convergence Diagnostics Report
Experiment 1: Beta-Binomial Model Posterior Inference

## Software
- **PPL Used:** PyMC (fallback from CmdStanPy)
- **Reason:** Stan compiler (make) not available in environment
- **Sampler:** NUTS (No-U-Turn Sampler)

## Sampling Parameters
- **Chains:** 4
- **Tune (warmup):** {tune}
- **Draws (sampling):** {draws}
- **Target accept:** {target_accept}
- **Total samples:** {total_samples}

## Convergence Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Max R_hat | < 1.01 | {max_rhat:.4f} | {'PASS' if max_rhat < 1.01 else 'FAIL'} |
| Min ESS (bulk) | > 400 | {min_ess_bulk:.0f} | {'PASS' if min_ess_bulk > 400 else 'FAIL'} |
| Min ESS (tail) | > 400 | {min_ess_tail:.0f} | {'PASS' if min_ess_tail > 400 else 'FAIL'} |
| Divergences | < 1% | {100 * divergences / total_samples:.2f}% | {'PASS' if (divergences / total_samples) < 0.01 else 'FAIL'} |

## Overall Status: {'PASS' if all_pass else 'FAIL'}

## Posterior Parameter Estimates

### Population Parameters
- **μ (population mean):** {mu_mean:.4f} [95% CI: {mu_025:.4f}, {mu_975:.4f}]
- **κ (concentration):** {kappa_mean:.2f} [95% CI: {kappa_025:.2f}, {kappa_975:.2f}]
- **φ (overdispersion):** {phi_mean:.4f} [95% CI: {phi_025:.4f}, {phi_975:.4f}]

### Interpretation
- Population mean success rate: {100*mu_mean:.2f}%
- Overdispersion factor: {phi_mean:.3f} (φ = 1 indicates binomial, φ > 1 indicates overdispersion)
- Intraclass correlation: {posterior['icc'].values.mean():.4f}

## Files Generated
- InferenceData (NetCDF): `{netcdf_path}`
- ArviZ summary: `{DIAGNOSTICS_DIR / 'arviz_summary.csv'}`
- Posterior samples: `{RESULTS_DIR / 'posterior_samples_scalar.csv'}`
- Group summaries: `{RESULTS_DIR / 'group_posterior_summary.csv'}`
"""

with open(DIAGNOSTICS_DIR / 'convergence_report.md', 'w') as f:
    f.write(convergence_report)

print(f"Convergence report saved to {DIAGNOSTICS_DIR / 'convergence_report.md'}")
print("\nFITTING SCRIPT COMPLETE")
