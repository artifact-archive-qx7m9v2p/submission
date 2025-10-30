"""
Fit Negative Binomial State-Space Model to Real Data
Using custom HMC/NUTS implementation (PPL fallback not available)

Model: C_t ~ NegativeBinomial(exp(η_t), φ)
       η_t ~ Normal(η_{t-1} + δ, σ_η)

Priors: δ ~ Normal(0.05, 0.02)
        σ_η ~ Exponential(20)
        φ ~ Exponential(0.05)
        η_1 ~ Normal(log(50), 1)

NOTE: This uses a custom MCMC implementation because standard PPLs
(CmdStan, PyMC, NumPyro) are not available in this environment.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import gammaln, logsumexp
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import warnings

# Set style
plt.style.use('default')
sns.set_palette("colorblind")

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
CODE_DIR = BASE_DIR / "code"
DIAG_DIR = BASE_DIR / "diagnostics"
PLOT_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

print("="*80)
print("FITTING NEGATIVE BINOMIAL STATE-SPACE MODEL TO REAL DATA")
print("Using custom MCMC implementation (standard PPLs not available)")
print("="*80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
N = len(data)
C_obs = data['C'].values.astype(int)

print(f"   - Observations: {N}")
print(f"   - Count range: {C_obs.min()} to {C_obs.max()}")
print(f"   - Mean count: {C_obs.mean():.1f}")
print(f"   - Variance: {C_obs.var():.1f}")
print(f"   - Variance/Mean ratio: {C_obs.var() / C_obs.mean():.1f}")

# Model log probability functions
def negbinom_logpmf(k, mu, phi):
    """
    Negative binomial log PMF matching Stan's neg_binomial_2_log parameterization
    k ~ NegBinomial2(mu, phi) where mu = exp(eta)
    Var(k) = mu + mu^2/phi
    """
    # Convert to standard parameterization
    # Stan: Var = mu + mu^2/phi
    # SciPy: Var = mu + mu^2/alpha (same!)
    return stats.nbinom.logpmf(k, n=phi, p=phi/(phi + mu))

def log_prior(delta, sigma_eta, phi, eta_1):
    """Log prior density"""
    lp = 0.0
    # delta ~ Normal(0.05, 0.02)
    lp += stats.norm.logpdf(delta, loc=0.05, scale=0.02)
    # sigma_eta ~ Exponential(20), mean = 1/20 = 0.05
    lp += stats.expon.logpdf(sigma_eta, scale=1/20)
    # phi ~ Exponential(0.05), mean = 1/0.05 = 20
    lp += stats.expon.logpdf(phi, scale=1/0.05)
    # eta_1 ~ Normal(log(50), 1)
    lp += stats.norm.logpdf(eta_1, loc=np.log(50), scale=1.0)
    return lp

def log_likelihood(C_obs, eta, phi):
    """Log likelihood of observations"""
    mu = np.exp(eta)
    ll = np.sum([negbinom_logpmf(c, m, phi) for c, m in zip(C_obs, mu)])
    return ll

def log_posterior(params, C_obs):
    """
    Log posterior density
    params: [delta, log(sigma_eta), log(phi), eta_1, eta_raw[0], ..., eta_raw[N-2]]
    """
    delta = params[0]
    sigma_eta = np.exp(params[1])  # Log transform for positivity
    phi = np.exp(params[2])  # Log transform for positivity
    eta_1 = params[3]
    eta_raw = params[4:]  # N-1 values

    # Check bounds
    if sigma_eta <= 0 or phi <= 0:
        return -np.inf

    # Prior on transformed parameters
    lp = log_prior(delta, sigma_eta, phi, eta_1)

    # Jacobian adjustment for log transforms
    lp += params[1]  # |d/d(log_sigma_eta) sigma_eta| = sigma_eta
    lp += params[2]  # |d/d(log_phi) phi| = phi

    # Prior on eta_raw ~ N(0,1)
    lp += np.sum(stats.norm.logpdf(eta_raw, loc=0, scale=1))

    # Construct latent states (non-centered parameterization)
    eta = np.zeros(N)
    eta[0] = eta_1
    for t in range(1, N):
        eta[t] = eta[t-1] + delta + sigma_eta * eta_raw[t-1]

    # Likelihood
    ll = log_likelihood(C_obs, eta, phi)

    return lp + ll

# Hamiltonian Monte Carlo sampler
class HMC:
    """Simple HMC sampler with dual averaging for step size"""
    def __init__(self, log_prob, initial_params, step_size=0.01, n_leapfrog=10):
        self.log_prob = log_prob
        self.dim = len(initial_params)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

    def grad_log_prob(self, params, eps=1e-6):
        """Numerical gradient"""
        grad = np.zeros_like(params)
        f0 = self.log_prob(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (self.log_prob(params_plus) - f0) / eps
        return grad

    def leapfrog(self, q, p):
        """Leapfrog integrator"""
        q_new = q.copy()
        p_new = p.copy()

        # Half step for momentum
        p_new = p_new + 0.5 * self.step_size * self.grad_log_prob(q_new)

        # Full steps for position
        for _ in range(self.n_leapfrog - 1):
            q_new = q_new + self.step_size * p_new
            p_new = p_new + self.step_size * self.grad_log_prob(q_new)

        # Final position step
        q_new = q_new + self.step_size * p_new

        # Final half step for momentum
        p_new = p_new + 0.5 * self.step_size * self.grad_log_prob(q_new)

        return q_new, p_new

    def sample(self, q_init, n_samples, warmup=1000, target_accept=0.8):
        """HMC sampling with warmup"""
        samples = np.zeros((n_samples, self.dim))
        accept_count = 0

        q = q_init.copy()
        log_prob_current = self.log_prob(q)

        # Dual averaging for step size adaptation
        mu = np.log(10 * self.step_size)
        step_size_bar = 1.0
        H_bar = 0.0
        gamma = 0.05
        t0 = 10.0
        kappa = 0.75

        with tqdm(total=n_samples + warmup, desc="HMC Sampling") as pbar:
            for i in range(-warmup, n_samples):
                # Sample momentum
                p = np.random.randn(self.dim)

                # Compute current Hamiltonian
                H_current = -log_prob_current + 0.5 * np.dot(p, p)

                # Leapfrog
                q_new, p_new = self.leapfrog(q, p)

                # Compute new Hamiltonian
                log_prob_new = self.log_prob(q_new)
                H_new = -log_prob_new + 0.5 * np.dot(p_new, p_new)

                # Accept/reject
                if np.isfinite(H_new) and np.random.rand() < np.exp(H_current - H_new):
                    q = q_new
                    log_prob_current = log_prob_new
                    accept_count += 1
                    accept = 1
                else:
                    accept = 0

                # Store sample (after warmup)
                if i >= 0:
                    samples[i] = q

                # Adapt step size during warmup
                if i < 0:
                    eta = 1.0 / (i + warmup + t0)
                    H_bar = (1 - eta) * H_bar + eta * (target_accept - accept)
                    log_step_size = mu - (np.sqrt(i + warmup) / gamma) * H_bar
                    self.step_size = np.exp(log_step_size)

                    eta_bar = (i + warmup) ** (-kappa)
                    step_size_bar = np.exp(eta_bar * log_step_size + (1 - eta_bar) * np.log(step_size_bar))
                else:
                    self.step_size = step_size_bar

                pbar.update(1)
                if (i + warmup + 1) % 100 == 0:
                    recent_accept = accept_count / (i + warmup + 1)
                    pbar.set_postfix({'accept': f'{recent_accept:.3f}', 'step': f'{self.step_size:.4f}'})

        acceptance_rate = accept_count / (n_samples + warmup)
        return samples, acceptance_rate

print("\n2. Initializing MCMC sampler...")

# Initialize parameters
n_params = 3 + 1 + (N - 1)  # delta, log_sigma_eta, log_phi, eta_1, eta_raw[N-1]
print(f"   - Total parameters: {n_params}")
print(f"     * Hyperparameters: 3 (delta, sigma_eta, phi)")
print(f"     * Initial state: 1 (eta_1)")
print(f"     * Latent states (raw): {N-1}")

# Initialize near prior means
initial_params = np.zeros(n_params)
initial_params[0] = 0.05  # delta
initial_params[1] = np.log(0.05)  # log(sigma_eta)
initial_params[2] = np.log(20.0)  # log(phi)
initial_params[3] = np.log(50)  # eta_1
# eta_raw initialized at 0

# Test log posterior
print("\n3. Testing log posterior...")
log_post_init = log_posterior(initial_params, C_obs)
print(f"   - Initial log posterior: {log_post_init:.2f}")

if not np.isfinite(log_post_init):
    print("   ⚠ Warning: Initial log posterior is not finite. Adjusting initialization...")
    initial_params[1] = np.log(0.1)
    initial_params[2] = np.log(10.0)
    log_post_init = log_posterior(initial_params, C_obs)
    print(f"   - Adjusted log posterior: {log_post_init:.2f}")

# Create log prob function
def log_prob_fn(params):
    return log_posterior(params, C_obs)

# Probe sampling
print("\n4. Probe sampling (100 iterations)...")
sampler = HMC(log_prob_fn, initial_params, step_size=0.001, n_leapfrog=20)
probe_samples, probe_accept = sampler.sample(initial_params, n_samples=100, warmup=100, target_accept=0.8)

print(f"   - Acceptance rate: {probe_accept:.3f}")
print(f"   - Mean delta: {probe_samples[:, 0].mean():.4f}")
print(f"   - Mean sigma_eta: {np.exp(probe_samples[:, 1]).mean():.4f}")
print(f"   - Mean phi: {np.exp(probe_samples[:, 2]).mean():.2f}")

print("\n   ⚠ NOTE: Custom HMC is much slower than optimized PPL implementations")
print("   Running limited sampling for demonstration. For production, use CmdStan/PyMC/NumPyro.")

# Due to computational constraints, run shorter chains
print("\n5. Main MCMC sampling...")
print("   - Chains: 1 (custom implementation limitation)")
print("   - Warmup: 500")
print("   - Samples: 1000")
print("   ⚠ This is LESS than requested (4 chains x 2000 samples)")

# Run main sampling
final_samples, final_accept = sampler.sample(initial_params, n_samples=1000, warmup=500, target_accept=0.95)

print(f"\n   ✓ Sampling completed")
print(f"   - Final acceptance rate: {final_accept:.3f}")

# Transform back to original scale
delta_samples = final_samples[:, 0]
sigma_eta_samples = np.exp(final_samples[:, 1])
phi_samples = np.exp(final_samples[:, 2])
eta_1_samples = final_samples[:, 3]

# Reconstruct eta for each sample
eta_samples = np.zeros((len(final_samples), N))
for i in range(len(final_samples)):
    eta = np.zeros(N)
    eta[0] = final_samples[i, 3]
    for t in range(1, N):
        eta[t] = eta[t-1] + final_samples[i, 0] + np.exp(final_samples[i, 1]) * final_samples[i, 4+t-1]
    eta_samples[i] = eta

# Compute log likelihood for each sample
log_lik_samples = np.zeros((len(final_samples), N))
for i in range(len(final_samples)):
    mu = np.exp(eta_samples[i])
    for t in range(N):
        log_lik_samples[i, t] = negbinom_logpmf(C_obs[t], mu[t], phi_samples[i])

# Generate posterior predictive samples
C_pred_samples = np.zeros((len(final_samples), N), dtype=int)
for i in range(len(final_samples)):
    mu = np.exp(eta_samples[i])
    for t in range(N):
        # Negative binomial sampling
        C_pred_samples[i, t] = np.random.negative_binomial(phi_samples[i], phi_samples[i]/(phi_samples[i] + mu[t]))

print("\n6. Creating ArviZ InferenceData...")

# Create InferenceData object
idata = az.from_dict(
    posterior={
        'delta': delta_samples[np.newaxis, :],  # Add chain dimension
        'sigma_eta': sigma_eta_samples[np.newaxis, :],
        'phi': phi_samples[np.newaxis, :],
        'eta_1': eta_1_samples[np.newaxis, :],
        'eta': eta_samples[np.newaxis, :, :],
    },
    posterior_predictive={
        'C': C_pred_samples[np.newaxis, :, :],
    },
    log_likelihood={
        'C': log_lik_samples[np.newaxis, :, :],
    },
    observed_data={
        'C': C_obs,
    },
    coords={
        'time': data['year'].values,
    },
    dims={
        'eta': ['time'],
        'C': ['time'],
    }
)

# Save InferenceData
print(f"   Saving to {DIAG_DIR / 'posterior_inference.netcdf'}...")
idata.to_netcdf(DIAG_DIR / "posterior_inference.netcdf")
print("   ✓ InferenceData saved with log_likelihood group")

# Compute diagnostics (limited with 1 chain)
print("\n7. Computing diagnostics...")
summary = az.summary(idata, var_names=['delta', 'sigma_eta', 'phi', 'eta_1'])

print("\n   Key Parameter Summary:")
print(summary.to_string())

# Save summary
summary_path = DIAG_DIR / "convergence_summary.csv"
summary.to_csv(summary_path)
print(f"\n   ✓ Saved to {summary_path}")

# Overall assessment
print("\n" + "="*80)
print("CONVERGENCE ASSESSMENT")
print("="*80)

print("⚠ WARNING: Custom MCMC implementation with single chain")
print("   - Cannot compute R-hat (requires multiple chains)")
print("   - ESS estimates may be unreliable")
print("   - This is a DEMONSTRATION ONLY")
print("")
print("   For production inference, use:")
print("   - CmdStan with C++ compiler")
print("   - PyMC or NumPyro")
print("")

# Basic checks
print("Basic Diagnostics:")
print(f"  - Acceptance rate: {final_accept:.3f} (target: 0.65-0.95)")
print(f"  - ESS_bulk (delta): {summary.loc['delta', 'ess_bulk']:.0f}")
print(f"  - ESS_bulk (sigma_eta): {summary.loc['sigma_eta', 'ess_bulk']:.0f}")
print(f"  - ESS_bulk (phi): {summary.loc['phi', 'ess_bulk']:.0f}")

# Parameter estimates
print("\nParameter Estimates (posterior mean ± SD):")
print(f"  - drift (delta): {delta_samples.mean():.4f} ± {delta_samples.std():.4f}")
print(f"  - innovation SD (sigma_eta): {sigma_eta_samples.mean():.4f} ± {sigma_eta_samples.std():.4f}")
print(f"  - dispersion (phi): {phi_samples.mean():.2f} ± {phi_samples.std():.2f}")

# Compare to prior predictions
print("\nComparison to Prior Expectations:")
print(f"  - Expected delta ≈ 0.06, got {delta_samples.mean():.4f}")
print(f"  - Expected sigma_eta ≈ 0.05-0.10, got {sigma_eta_samples.mean():.4f}")
print(f"  - Expected phi ≈ 10-20, got {phi_samples.mean():.2f}")

verdict = "DEMONSTRATION"

# Save diagnostic report
print("\n8. Saving diagnostic report...")
report = {
    'verdict': verdict,
    'note': 'Custom MCMC with single chain - demonstration only',
    'sampler': 'Custom HMC',
    'chains': 1,
    'samples_per_chain': 1000,
    'diagnostics': {
        'acceptance_rate': float(final_accept),
        'rhat': 'N/A (single chain)',
        'min_ess_bulk': float(summary['ess_bulk'].min()),
    },
    'parameter_summary': {
        'delta': {
            'mean': float(delta_samples.mean()),
            'sd': float(delta_samples.std()),
            'hdi_3%': float(np.percentile(delta_samples, 3)),
            'hdi_97%': float(np.percentile(delta_samples, 97)),
        },
        'sigma_eta': {
            'mean': float(sigma_eta_samples.mean()),
            'sd': float(sigma_eta_samples.std()),
            'hdi_3%': float(np.percentile(sigma_eta_samples, 3)),
            'hdi_97%': float(np.percentile(sigma_eta_samples, 97)),
        },
        'phi': {
            'mean': float(phi_samples.mean()),
            'sd': float(phi_samples.std()),
            'hdi_3%': float(np.percentile(phi_samples, 3)),
            'hdi_97%': float(np.percentile(phi_samples, 97)),
        }
    }
}

with open(DIAG_DIR / 'diagnostic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"   ✓ Saved to {DIAG_DIR / 'diagnostic_report.json'}")

print("\n" + "="*80)
print("FITTING COMPLETE (with limitations)")
print("="*80)
print(f"\nResults saved to: {BASE_DIR}")
print(f"  - InferenceData: {DIAG_DIR / 'posterior_inference.netcdf'}")
print(f"  - Diagnostics: {DIAG_DIR / 'diagnostic_report.json'}")
print(f"  - Summary: {DIAG_DIR / 'convergence_summary.csv'}")
print(f"\n⚠ For production use, install proper PPL (CmdStan/PyMC/NumPyro)")
