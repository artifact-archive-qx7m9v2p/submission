"""
Analyze the posterior inference results from the mixture model.
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import arviz as az
import pickle
from pathlib import Path
import scipy.stats as stats

# Paths
project_root = Path("/workspace")
output_dir = project_root / "experiments" / "experiment_2" / "posterior_inference"
diag_dir = output_dir / "diagnostics"
data_path = project_root / "data" / "data.csv"

# Load data
data = pd.read_csv(data_path)
n_groups = len(data)
n_trials = data['n_trials'].values
r_successes = data['r_successes'].values

# Load trace
print("Loading posterior samples...")
with open(diag_dir / "idata.pkl", 'rb') as f:
    idata = pickle.load(f)

print(f"InferenceData groups: {list(idata.groups())}")
print(f"Posterior shape: chains={idata.posterior.dims['chain']}, draws={idata.posterior.dims['draw']}")

# Add log_likelihood if not present
if 'log_likelihood' not in idata.groups():
    print("\nAdding log_likelihood for LOO-CV...")
    p_samples = idata.posterior['p'].values

    log_lik_pointwise = np.zeros((p_samples.shape[0], p_samples.shape[1], n_groups))
    for chain_idx in range(p_samples.shape[0]):
        for draw_idx in range(p_samples.shape[1]):
            for j in range(n_groups):
                log_lik_pointwise[chain_idx, draw_idx, j] = stats.binom.logpmf(
                    r_successes[j], n_trials[j], p_samples[chain_idx, draw_idx, j]
                )

    # Add to InferenceData
    import xarray as xr
    log_lik_da = xr.DataArray(
        log_lik_pointwise,
        dims=['chain', 'draw', 'obs_id'],
        coords={
            'chain': idata.posterior.coords['chain'],
            'draw': idata.posterior.coords['draw'],
            'obs_id': range(n_groups)
        }
    )

    idata.add_groups({'log_likelihood': xr.Dataset({'r': log_lik_da})})
    print(f"  Added log_likelihood: shape {log_lik_da.shape}")

    # Save updated version
    idata.to_netcdf(diag_dir / "posterior_inference.netcdf")
    print("  Saved updated InferenceData")

# Summary statistics
print("\n" + "="*70)
print("PARAMETER SUMMARIES")
print("="*70)

summary = az.summary(idata, var_names=['pi', 'mu', 'sigma'])
print("\nMixing proportions and cluster parameters:")
print(summary)

# Cluster means on probability scale
mu_samples = idata.posterior['mu'].values.reshape(-1, 3)
p_cluster = 1 / (1 + np.exp(-mu_samples))

print("\nCluster means on probability scale:")
for k in range(3):
    print(f"  Cluster {k+1}: {p_cluster[:, k].mean():.4f} [{np.percentile(p_cluster[:, k], 2.5):.4f}, {np.percentile(p_cluster[:, k], 97.5):.4f}]")

# Cluster separation
print("\nCluster separation (logit scale):")
sep_12 = mu_samples[:, 1] - mu_samples[:, 0]
sep_23 = mu_samples[:, 2] - mu_samples[:, 1]
print(f"  mu[2] - mu[1]: {sep_12.mean():.3f} [{np.percentile(sep_12, 2.5):.3f}, {np.percentile(sep_12, 97.5):.3f}]")
print(f"  mu[3] - mu[2]: {sep_23.mean():.3f} [{np.percentile(sep_23, 2.5):.3f}, {np.percentile(sep_23, 97.5):.3f}]")

# Effective number of clusters
pi_samples = idata.posterior['pi'].values.reshape(-1, 3)
entropy = -np.sum(pi_samples * np.log(pi_samples + 1e-10), axis=1)
K_eff = np.exp(entropy)
print(f"\nEffective number of clusters: {K_eff.mean():.2f} [{np.percentile(K_eff, 2.5):.2f}, {np.percentile(K_eff, 97.5):.2f}]")

# Cluster assignments
print("\n" + "="*70)
print("CLUSTER ASSIGNMENTS")
print("="*70)

cluster_probs_samples = idata.posterior['cluster_probs'].values  # (chains, draws, groups, 3)
cluster_probs_mean = cluster_probs_samples.mean(axis=(0, 1))  # (groups, 3)

# Assign each group to most probable cluster
assignments = np.argmax(cluster_probs_mean, axis=1) + 1  # 1-indexed
certainty = np.max(cluster_probs_mean, axis=1)

assignment_df = pd.DataFrame({
    'group_id': data['group_id'],
    'n_trials': data['n_trials'],
    'r_successes': data['r_successes'],
    'success_rate': data['success_rate'],
    'assigned_cluster': assignments,
    'certainty': certainty,
    'p_cluster_1': cluster_probs_mean[:, 0],
    'p_cluster_2': cluster_probs_mean[:, 1],
    'p_cluster_3': cluster_probs_mean[:, 2]
})

print("\nCluster assignments:")
print(assignment_df[['group_id', 'success_rate', 'assigned_cluster', 'certainty']].to_string(index=False))

# Assignment certainty
print(f"\nMean assignment certainty: {certainty.mean():.3f}")
print(f"Groups with certainty < 0.6: {np.sum(certainty < 0.6)}")
print(f"Groups with certainty < 0.7: {np.sum(certainty < 0.7)}")

# Cluster sizes
print("\nCluster sizes:")
for k in range(3):
    n_k = np.sum(assignments == k+1)
    print(f"  Cluster {k+1}: {n_k} groups")

# Save assignment table
assignment_df.to_csv(diag_dir / "cluster_assignments.csv", index=False)
print(f"\nSaved cluster assignments to {diag_dir / 'cluster_assignments.csv'}")

# Convergence assessment
print("\n" + "="*70)
print("CONVERGENCE ASSESSMENT")
print("="*70)

rhat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()
divergences = idata.sample_stats.diverging.sum().item()
total_draws = idata.posterior.dims['draw'] * idata.posterior.dims['chain']

print(f"\nMax R-hat: {rhat_max:.4f} (target: < 1.01)")
print(f"Min ESS_bulk: {ess_min:.1f} (target: > 400)")
print(f"Divergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")

if rhat_max < 1.01 and ess_min > 400 and divergences < 0.01*total_draws:
    print("\n✓ CONVERGENCE: PASS")
    convergence_status = "PASS"
elif rhat_max < 1.05 and ess_min > 100:
    print("\n⚠ CONVERGENCE: MARGINAL (usable but not ideal)")
    convergence_status = "MARGINAL"
else:
    print("\n✗ CONVERGENCE: FAIL")
    convergence_status = "FAIL"

# Falsification checks
print("\n" + "="*70)
print("FALSIFICATION CHECKS")
print("="*70)

checks = {}

# 1. K_effective < 2?
checks['K_eff_gt_2'] = K_eff.mean() >= 2.0
print(f"1. K_effective >= 2? {K_eff.mean():.2f} >= 2.0 → {checks['K_eff_gt_2']}")

# 2. Cluster separation > 0.5?
checks['separation_ok'] = (sep_12.mean() > 0.5) or (sep_23.mean() > 0.5)
print(f"2. At least one separation > 0.5 logits? → {checks['separation_ok']}")
print(f"     sep[2-1] = {sep_12.mean():.3f}, sep[3-2] = {sep_23.mean():.3f}")

# 3. Mean assignment certainty > 0.6?
checks['certainty_ok'] = certainty.mean() >= 0.6
print(f"3. Mean assignment certainty >= 0.6? {certainty.mean():.3f} → {checks['certainty_ok']}")

# 4. Convergence acceptable?
checks['convergence_ok'] = convergence_status in ["PASS", "MARGINAL"]
print(f"4. Convergence acceptable? → {checks['convergence_ok']}")

# Overall decision
if all(checks.values()):
    print("\n✓ FALSIFICATION: PASS (model is plausible)")
    overall_status = "PASS"
elif convergence_status == "FAIL":
    print("\n✗ FALSIFICATION: FAIL (convergence issues)")
    overall_status = "FAIL"
else:
    print("\n⚠ FALSIFICATION: MARGINAL (some concerns)")
    overall_status = "MARGINAL"

# Save summary
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_summary = {
    'convergence_status': convergence_status,
    'falsification_status': overall_status,
    'rhat_max': rhat_max,
    'ess_min': ess_min,
    'divergences': divergences,
    'divergence_pct': 100*divergences/total_draws,
    'K_eff_mean': K_eff.mean(),
    'separation_12_mean': sep_12.mean(),
    'separation_23_mean': sep_23.mean(),
    'mean_certainty': certainty.mean(),
    'n_uncertain_06': np.sum(certainty < 0.6),
    'n_uncertain_07': np.sum(certainty < 0.7),
}

results_df = pd.DataFrame([results_summary])
results_df.to_csv(diag_dir / "inference_summary_metrics.csv", index=False)
print(f"Saved summary metrics to {diag_dir / 'inference_summary_metrics.csv'}")

print("\nAnalysis complete!")
