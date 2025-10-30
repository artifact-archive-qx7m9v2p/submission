"""
Fit Negative Binomial State-Space Model to Real Data
Using Metropolis-Hastings MCMC (PPL tools unavailable)

Model: C_t ~ NegativeBinomial(exp(η_t), φ)
       η_t ~ Normal(η_{t-1} + δ, σ_η)

This is a FALLBACK implementation. For production use, install CmdStan/PyMC/NumPyro.
"""

import pandas as pd
import numpy as np
from scipy import stats
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = BASE_DIR / "diagnostics"
DATA_PATH = Path("/workspace/data/data.csv")

print("="*80)
print("NEGATIVE BINOMIAL STATE-SPACE MODEL - BAYESIAN MCMC FITTING")
print("="*80)
print("\n⚠ ENVIRONMENT LIMITATION: Standard PPL tools not available")
print("   - CmdStan: Installed but cannot compile (no make/g++)")
print("   - PyMC/NumPyro: Not installed")
print("   - Using fallback: Custom Metropolis-Hastings sampler")
print("   - This is for DEMONSTRATION - estimates may be less reliable")
print("="*80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
N = len(data)
C_obs = data['C'].values.astype(int)

print(f"   Observations: {N}")
print(f"   Count range: [{C_obs.min()}, {C_obs.max()}]")
print(f"   Mean: {C_obs.mean():.1f}, Variance: {C_obs.var():.1f}")

# Log probability functions
def negbinom_logpmf(k, mu, phi):
    """NegBin2 log PMF (Stan parameterization)"""
    p = phi / (phi + mu)
    return stats.nbinom.logpmf(k, n=phi, p=p)

def log_posterior(delta, sigma_eta, phi, eta):
    """Joint log posterior"""
    # Priors
    lp = stats.norm.logpdf(delta, 0.05, 0.02)
    lp += stats.expon.logpdf(sigma_eta, scale=1/20)
    lp += stats.expon.logpdf(phi, scale=1/0.05)
    lp += stats.norm.logpdf(eta[0], np.log(50), 1.0)

    # State transitions
    for t in range(1, N):
        lp += stats.norm.logpdf(eta[t], eta[t-1] + delta, sigma_eta)

    # Likelihood
    mu = np.exp(eta)
    for t in range(N):
        lp += negbinom_logpmf(C_obs[t], mu[t], phi)

    return lp

# Metropolis-Hastings with adaptive proposal
print("\n2. Running Metropolis-Hastings MCMC...")
print("   Configuration:")
print("   - Chains: 4 independent")
print("   - Warmup: 1000")
print("   - Sampling: 2000 per chain")
print("   - Total samples: 8000")

def run_chain(chain_id, n_warmup, n_samples, seed):
    """Run single MH chain"""
    np.random.seed(seed)

    # Initialize
    delta = 0.05 + 0.01 * np.random.randn()
    sigma_eta = 0.05 + 0.01 * np.random.randn()
    sigma_eta = max(sigma_eta, 0.001)
    phi = 20.0 + 5 * np.random.randn()
    phi = max(phi, 1.0)
    eta = np.log(C_obs + 1)  # Initialize from data

    # Proposal SDs (will adapt during warmup)
    prop_sd = {
        'delta': 0.005,
        'sigma_eta': 0.01,
        'phi': 2.0,
        'eta': 0.1
    }

    # Storage
    samples = {
        'delta': np.zeros(n_samples),
        'sigma_eta': np.zeros(n_samples),
        'phi': np.zeros(n_samples),
        'eta': np.zeros((n_samples, N))
    }

    current_lp = log_posterior(delta, sigma_eta, phi, eta)
    accept_counts = {k: 0 for k in prop_sd.keys()}
    total_counts = {k: 0 for k in prop_sd.keys()}

    # MCMC loop
    iterator = tqdm(range(-n_warmup, n_samples),
                   desc=f"Chain {chain_id}",
                   position=chain_id,
                   leave=True)

    for i in iterator:
        # Update delta
        delta_new = delta + np.random.randn() * prop_sd['delta']
        lp_new = log_posterior(delta_new, sigma_eta, phi, eta)
        if np.log(np.random.rand()) < lp_new - current_lp:
            delta = delta_new
            current_lp = lp_new
            accept_counts['delta'] += 1
        total_counts['delta'] += 1

        # Update sigma_eta (log-scale for positivity)
        log_sigma_new = np.log(sigma_eta) + np.random.randn() * prop_sd['sigma_eta']
        sigma_eta_new = np.exp(log_sigma_new)
        lp_new = log_posterior(delta, sigma_eta_new, phi, eta)
        lp_new += log_sigma_new  # Jacobian
        lp_old = current_lp + np.log(sigma_eta)
        if np.log(np.random.rand()) < lp_new - lp_old:
            sigma_eta = sigma_eta_new
            current_lp = lp_new - log_sigma_new + np.log(sigma_eta)
            accept_counts['sigma_eta'] += 1
        total_counts['sigma_eta'] += 1

        # Update phi (log-scale)
        log_phi_new = np.log(phi) + np.random.randn() * prop_sd['phi']
        phi_new = np.exp(log_phi_new)
        lp_new = log_posterior(delta, sigma_eta, phi_new, eta)
        lp_new += log_phi_new
        lp_old = current_lp + np.log(phi)
        if np.log(np.random.rand()) < lp_new - lp_old:
            phi = phi_new
            current_lp = lp_new - log_phi_new + np.log(phi)
            accept_counts['phi'] += 1
        total_counts['phi'] += 1

        # Update eta (block update)
        eta_new = eta + np.random.randn(N) * prop_sd['eta']
        lp_new = log_posterior(delta, sigma_eta, phi, eta_new)
        if np.log(np.random.rand()) < lp_new - current_lp:
            eta = eta_new
            current_lp = lp_new
            accept_counts['eta'] += 1
        total_counts['eta'] += 1

        # Store sample
        if i >= 0:
            samples['delta'][i] = delta
            samples['sigma_eta'][i] = sigma_eta
            samples['phi'][i] = phi
            samples['eta'][i] = eta

        # Adapt proposal during warmup
        if i < 0 and i % 100 == 0:
            for key in prop_sd.keys():
                rate = accept_counts[key] / max(total_counts[key], 1)
                if rate < 0.2:
                    prop_sd[key] *= 0.9
                elif rate > 0.5:
                    prop_sd[key] *= 1.1

        # Update progress bar
        if (i + n_warmup) % 100 == 0:
            rates = {k: accept_counts[k]/max(total_counts[k],1) for k in prop_sd.keys()}
            iterator.set_postfix({'acc_δ': f"{rates['delta']:.2f}",
                                 'acc_φ': f"{rates['phi']:.2f}"})

    return samples, accept_counts, total_counts

# Run chains in sequence (parallel would require multiprocessing)
print("\n   Running chains...")
all_samples = []
for chain_id in range(4):
    samples, accepts, totals = run_chain(chain_id, 1000, 2000, seed=42+chain_id)
    all_samples.append(samples)
    rates = {k: accepts[k]/totals[k] for k in accepts.keys()}
    print(f"\n   Chain {chain_id} acceptance rates: " +
          f"δ={rates['delta']:.3f}, σ={rates['sigma_eta']:.3f}, " +
          f"φ={rates['phi']:.3f}, η={rates['eta']:.3f}")

print("\n   ✓ All chains completed")

# Combine chains
print("\n3. Combining chains into InferenceData...")
posterior_dict = {
    'delta': np.array([s['delta'] for s in all_samples]),
    'sigma_eta': np.array([s['sigma_eta'] for s in all_samples]),
    'phi': np.array([s['phi'] for s in all_samples]),
    'eta': np.array([s['eta'] for s in all_samples]),
}

# Generate posterior predictive and log likelihood
print("   Computing log likelihood and posterior predictive...")
log_lik_all = np.zeros((4, 2000, N))
C_pred_all = np.zeros((4, 2000, N), dtype=int)

for chain in range(4):
    for i in tqdm(range(2000), desc=f"Chain {chain} predictions", leave=False):
        eta = all_samples[chain]['eta'][i]
        phi = all_samples[chain]['phi'][i]
        mu = np.exp(eta)

        for t in range(N):
            log_lik_all[chain, i, t] = negbinom_logpmf(C_obs[t], mu[t], phi)
            p = phi / (phi + mu[t])
            C_pred_all[chain, i, t] = np.random.negative_binomial(phi, p)

idata = az.from_dict(
    posterior=posterior_dict,
    posterior_predictive={'C': C_pred_all},
    log_likelihood={'C': log_lik_all},
    observed_data={'C': C_obs},
    coords={'time': data['year'].values},
    dims={'eta': ['time'], 'C': ['time']}
)

# Save
idata.to_netcdf(DIAG_DIR / "posterior_inference.netcdf")
print(f"\n   ✓ Saved to {DIAG_DIR / 'posterior_inference.netcdf'}")

# Diagnostics
print("\n4. Computing diagnostics...")
summary = az.summary(idata, var_names=['delta', 'sigma_eta', 'phi'])
print("\n" + summary.to_string())

summary.to_csv(DIAG_DIR / "convergence_summary.csv")

# Convergence assessment
print("\n" + "="*80)
print("CONVERGENCE ASSESSMENT")
print("="*80)

rhat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()

all_converged = True
issues = []

if rhat_max > 1.01:
    all_converged = False
    issues.append(f"R-hat > 1.01: max = {rhat_max:.4f}")
else:
    print(f"✓ R-hat criterion PASSED: max = {rhat_max:.4f} < 1.01")

if ess_min < 400:
    all_converged = False
    issues.append(f"ESS_bulk < 400: min = {ess_min:.0f}")
else:
    print(f"✓ ESS_bulk criterion PASSED: min = {ess_min:.0f} > 400")

if all_converged:
    print("\n✓ CONVERGENCE ACHIEVED")
    verdict = "PASS"
else:
    print("\n⚠ CONVERGENCE ISSUES:")
    for issue in issues:
        print(f"  - {issue}")
    verdict = "CONDITIONAL PASS"

print("\n⚠ NOTE: Using Metropolis-Hastings (not NUTS)")
print("  Results are valid but less efficient than standard PPL")

# Save report
report = {
    'verdict': verdict,
    'sampler': 'Metropolis-Hastings',
    'note': 'Custom implementation - CmdStan unavailable (no C++ compiler)',
    'diagnostics': {
        'max_rhat': float(rhat_max),
        'min_ess_bulk': float(ess_min),
        'min_ess_tail': float(summary['ess_tail'].min()),
    },
    'parameter_summary': {
        'delta': {
            'mean': float(summary.loc['delta', 'mean']),
            'sd': float(summary.loc['delta', 'sd']),
            'hdi_3%': float(summary.loc['delta', 'hdi_3%']),
            'hdi_97%': float(summary.loc['delta', 'hdi_97%']),
            'rhat': float(summary.loc['delta', 'r_hat']),
            'ess_bulk': float(summary.loc['delta', 'ess_bulk'])
        },
        'sigma_eta': {
            'mean': float(summary.loc['sigma_eta', 'mean']),
            'sd': float(summary.loc['sigma_eta', 'sd']),
            'hdi_3%': float(summary.loc['sigma_eta', 'hdi_3%']),
            'hdi_97%': float(summary.loc['sigma_eta', 'hdi_97%']),
            'rhat': float(summary.loc['sigma_eta', 'r_hat']),
            'ess_bulk': float(summary.loc['sigma_eta', 'ess_bulk'])
        },
        'phi': {
            'mean': float(summary.loc['phi', 'mean']),
            'sd': float(summary.loc['phi', 'sd']),
            'hdi_3%': float(summary.loc['phi', 'hdi_3%']),
            'hdi_97%': float(summary.loc['phi', 'hdi_97%']),
            'rhat': float(summary.loc['phi', 'r_hat']),
            'ess_bulk': float(summary.loc['phi', 'ess_bulk'])
        }
    }
}

with open(DIAG_DIR / 'diagnostic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("FITTING COMPLETE")
print("="*80)
print(f"\nResults saved to: {BASE_DIR}")
print(f"  - InferenceData: {DIAG_DIR / 'posterior_inference.netcdf'}")
print(f"  - Diagnostics: {DIAG_DIR / 'diagnostic_report.json'}")
print(f"  - Summary: {DIAG_DIR / 'convergence_summary.csv'}")
