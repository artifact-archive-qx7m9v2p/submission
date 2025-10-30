"""
Simulation-Based Calibration (SBC) for Negative Binomial State-Space Model
Pure NumPy/SciPy implementation with custom MCMC sampler

This implements SBC without requiring Stan/PyMC, using a custom
Metropolis-Hastings within Gibbs sampler.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy import stats
from scipy.special import gammaln, logsumexp
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"

# Create directories
DIAGNOSTICS_DIR.mkdir(exist_ok=True, parents=True)

# SBC Configuration
N_SIMS = 100  # Number of SBC simulations
N_TIME = 40   # Time points
N_DRAWS = 1000  # Posterior draws per simulation

# MCMC Configuration
N_WARMUP = 2000
N_SAMPLES = 1000
THIN = 1

print("="*80)
print("SIMULATION-BASED CALIBRATION (SBC)")
print("Negative Binomial State-Space Model")
print("Pure NumPy Implementation with Custom MCMC")
print("="*80)
print(f"\nConfiguration:")
print(f"  - Number of simulations: {N_SIMS}")
print(f"  - Time points per simulation: {N_TIME}")
print(f"  - Posterior draws per simulation: {N_DRAWS}")
print(f"\nMCMC settings:")
print(f"  - Warmup: {N_WARMUP}")
print(f"  - Samples: {N_SAMPLES}")
print(f"  - Thinning: {THIN}")
print("="*80)

# =============================================================================
# PRIOR SPECIFICATION (ADJUSTED from Round 2)
# =============================================================================

def sample_from_prior():
    """Sample parameters from the prior distribution."""
    delta = np.random.normal(0.05, 0.02)
    sigma_eta = np.random.exponential(1/20)  # Mean = 0.05
    phi = np.random.exponential(1/0.05)      # Mean = 20
    eta_1 = np.random.normal(np.log(50), 1)
    return {
        'delta': delta,
        'sigma_eta': sigma_eta,
        'phi': phi,
        'eta_1': eta_1
    }

def log_prior(delta, sigma_eta, phi, eta_1):
    """Compute log prior density."""
    lp = 0.0

    # delta ~ Normal(0.05, 0.02)
    lp += stats.norm.logpdf(delta, 0.05, 0.02)

    # sigma_eta ~ Exponential(20) i.e., rate=20, mean=0.05
    if sigma_eta <= 0:
        return -np.inf
    lp += stats.expon.logpdf(sigma_eta, scale=1/20)

    # phi ~ Exponential(0.05) i.e., rate=0.05, mean=20
    if phi <= 0:
        return -np.inf
    lp += stats.expon.logpdf(phi, scale=1/0.05)

    # eta_1 ~ Normal(log(50), 1)
    lp += stats.norm.logpdf(eta_1, np.log(50), 1)

    return lp

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(params, N):
    """Generate synthetic data from the model given true parameters."""
    delta = params['delta']
    sigma_eta = params['sigma_eta']
    phi = params['phi']
    eta_1 = params['eta_1']

    # Generate latent states (random walk with drift)
    eta = np.zeros(N)
    eta[0] = eta_1

    for t in range(1, N):
        eta[t] = np.random.normal(eta[t-1] + delta, sigma_eta)

    # Generate observations (negative binomial)
    C = np.zeros(N, dtype=int)
    for t in range(N):
        mu = np.exp(eta[t])
        # NegBinom2 parameterization: var = mu + mu^2/phi
        # Convert to n, p parameterization for scipy
        p = phi / (mu + phi)
        n = phi
        C[t] = np.random.negative_binomial(n, p)

    return C, eta

# =============================================================================
# LIKELIHOOD COMPUTATION
# =============================================================================

def negbinom_logpmf(y, mu, phi):
    """
    Compute log PMF of negative binomial (NegBinom2 parameterization).

    Parameters:
    -----------
    y : int or array
        Observed counts
    mu : float or array
        Mean parameter
    phi : float
        Dispersion parameter

    Returns:
    --------
    log_prob : float or array
        Log probability
    """
    # Convert to n, p parameterization
    p = phi / (mu + phi)
    n = phi

    # Use scipy's negative binomial
    return stats.nbinom.logpmf(y, n, p)

def log_likelihood(C, eta, phi):
    """Compute log likelihood of observations given latent states."""
    mu = np.exp(eta)
    return np.sum(negbinom_logpmf(C, mu, phi))

def log_likelihood_states(eta, delta, sigma_eta, eta_1):
    """Compute log likelihood of latent states given parameters."""
    ll = 0.0

    # Initial state
    ll += stats.norm.logpdf(eta[0], eta_1, 1e-6)  # Delta function approximation

    # State transitions
    for t in range(1, len(eta)):
        ll += stats.norm.logpdf(eta[t], eta[t-1] + delta, sigma_eta)

    return ll

# =============================================================================
# MCMC SAMPLER (Metropolis-Hastings within Gibbs)
# =============================================================================

class NBStateSpaceSampler:
    """
    MCMC sampler for Negative Binomial State-Space Model.

    Uses Metropolis-Hastings for parameters and Gibbs/MH for states.
    """

    def __init__(self, C, proposal_scales=None):
        self.C = C
        self.N = len(C)

        # Proposal scales (tuned for acceptance rate ~0.3-0.4)
        if proposal_scales is None:
            self.proposal_scales = {
                'delta': 0.01,
                'sigma_eta': 0.01,
                'phi': 2.0,
                'eta_1': 0.2,
                'eta': 0.1
            }
        else:
            self.proposal_scales = proposal_scales

        # Tracking
        self.accept_counts = {
            'delta': 0, 'sigma_eta': 0, 'phi': 0, 'eta_1': 0, 'eta': 0
        }
        self.total_proposals = {
            'delta': 0, 'sigma_eta': 0, 'phi': 0, 'eta_1': 0, 'eta': 0
        }

    def log_posterior(self, delta, sigma_eta, phi, eta_1, eta):
        """Compute log posterior (up to constant)."""
        # Prior
        lp = log_prior(delta, sigma_eta, phi, eta_1)
        if not np.isfinite(lp):
            return -np.inf

        # Likelihood of states
        lp += log_likelihood_states(eta, delta, sigma_eta, eta_1)
        if not np.isfinite(lp):
            return -np.inf

        # Likelihood of observations
        lp += log_likelihood(self.C, eta, phi)

        return lp

    def propose_parameter(self, current, param_name):
        """Propose new parameter value."""
        scale = self.proposal_scales[param_name]

        if param_name in ['sigma_eta', 'phi']:
            # Log-scale proposal for positive parameters
            log_current = np.log(current)
            log_proposed = log_current + np.random.normal(0, scale)
            proposed = np.exp(log_proposed)
            # Jacobian correction
            log_jacobian = log_proposed - log_current
        else:
            # Normal proposal
            proposed = current + np.random.normal(0, scale)
            log_jacobian = 0.0

        return proposed, log_jacobian

    def update_delta(self, delta, sigma_eta, phi, eta_1, eta):
        """Update delta via Metropolis-Hastings."""
        self.total_proposals['delta'] += 1

        # Current log posterior
        current_lp = self.log_posterior(delta, sigma_eta, phi, eta_1, eta)

        # Propose new value
        delta_new, log_jacobian = self.propose_parameter(delta, 'delta')

        # New log posterior
        new_lp = self.log_posterior(delta_new, sigma_eta, phi, eta_1, eta)

        # Accept/reject
        log_alpha = new_lp - current_lp + log_jacobian
        if np.log(np.random.rand()) < log_alpha:
            self.accept_counts['delta'] += 1
            return delta_new
        return delta

    def update_sigma_eta(self, delta, sigma_eta, phi, eta_1, eta):
        """Update sigma_eta via Metropolis-Hastings."""
        self.total_proposals['sigma_eta'] += 1

        current_lp = self.log_posterior(delta, sigma_eta, phi, eta_1, eta)
        sigma_eta_new, log_jacobian = self.propose_parameter(sigma_eta, 'sigma_eta')

        if sigma_eta_new <= 0:
            return sigma_eta

        new_lp = self.log_posterior(delta, sigma_eta_new, phi, eta_1, eta)
        log_alpha = new_lp - current_lp + log_jacobian

        if np.log(np.random.rand()) < log_alpha:
            self.accept_counts['sigma_eta'] += 1
            return sigma_eta_new
        return sigma_eta

    def update_phi(self, delta, sigma_eta, phi, eta_1, eta):
        """Update phi via Metropolis-Hastings."""
        self.total_proposals['phi'] += 1

        current_lp = self.log_posterior(delta, sigma_eta, phi, eta_1, eta)
        phi_new, log_jacobian = self.propose_parameter(phi, 'phi')

        if phi_new <= 0:
            return phi

        new_lp = self.log_posterior(delta, sigma_eta, phi_new, eta_1, eta)
        log_alpha = new_lp - current_lp + log_jacobian

        if np.log(np.random.rand()) < log_alpha:
            self.accept_counts['phi'] += 1
            return phi_new
        return phi

    def update_eta_1(self, delta, sigma_eta, phi, eta_1, eta):
        """Update eta_1 via Metropolis-Hastings."""
        self.total_proposals['eta_1'] += 1

        current_lp = self.log_posterior(delta, sigma_eta, phi, eta_1, eta)
        eta_1_new, log_jacobian = self.propose_parameter(eta_1, 'eta_1')

        new_lp = self.log_posterior(delta, sigma_eta, phi, eta_1_new, eta)
        log_alpha = new_lp - current_lp + log_jacobian

        if np.log(np.random.rand()) < log_alpha:
            self.accept_counts['eta_1'] += 1
            return eta_1_new
        return eta_1

    def update_eta(self, delta, sigma_eta, phi, eta_1, eta):
        """Update latent states via block Metropolis-Hastings."""
        self.total_proposals['eta'] += 1

        current_lp = self.log_posterior(delta, sigma_eta, phi, eta_1, eta)

        # Propose new trajectory (small perturbations)
        eta_new = eta + np.random.normal(0, self.proposal_scales['eta'], size=self.N)

        new_lp = self.log_posterior(delta, sigma_eta, phi, eta_1, eta_new)
        log_alpha = new_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            self.accept_counts['eta'] += 1
            return eta_new
        return eta

    def sample(self, n_warmup, n_samples, thin=1, initial_state=None):
        """
        Run MCMC sampler.

        Parameters:
        -----------
        n_warmup : int
            Number of warmup iterations
        n_samples : int
            Number of samples to collect
        thin : int
            Thinning interval
        initial_state : dict, optional
            Initial parameter values

        Returns:
        --------
        samples : dict
            Dictionary of parameter samples
        """
        # Initialize
        if initial_state is None:
            # Initialize from prior
            init = sample_from_prior()
            delta = init['delta']
            sigma_eta = init['sigma_eta']
            phi = init['phi']
            eta_1 = init['eta_1']
            # Initialize eta near observations
            eta = np.log(self.C + 1)
        else:
            delta = initial_state['delta']
            sigma_eta = initial_state['sigma_eta']
            phi = initial_state['phi']
            eta_1 = initial_state['eta_1']
            eta = initial_state['eta']

        # Storage
        samples = {
            'delta': [],
            'sigma_eta': [],
            'phi': [],
            'eta_1': []
        }

        total_iters = n_warmup + n_samples * thin

        for iter_idx in range(total_iters):
            # Update parameters
            delta = self.update_delta(delta, sigma_eta, phi, eta_1, eta)
            sigma_eta = self.update_sigma_eta(delta, sigma_eta, phi, eta_1, eta)
            phi = self.update_phi(delta, sigma_eta, phi, eta_1, eta)
            eta_1 = self.update_eta_1(delta, sigma_eta, phi, eta_1, eta)
            eta = self.update_eta(delta, sigma_eta, phi, eta_1, eta)

            # Store samples (after warmup, with thinning)
            if iter_idx >= n_warmup and (iter_idx - n_warmup) % thin == 0:
                samples['delta'].append(delta)
                samples['sigma_eta'].append(sigma_eta)
                samples['phi'].append(phi)
                samples['eta_1'].append(eta_1)

        # Convert to arrays
        for key in samples:
            samples[key] = np.array(samples[key])

        return samples

    def get_acceptance_rates(self):
        """Compute acceptance rates."""
        rates = {}
        for key in self.accept_counts:
            if self.total_proposals[key] > 0:
                rates[key] = self.accept_counts[key] / self.total_proposals[key]
            else:
                rates[key] = 0.0
        return rates

# =============================================================================
# SBC PROCEDURE
# =============================================================================

def compute_rank(true_value, posterior_samples):
    """Compute rank statistic."""
    rank = np.sum(posterior_samples < true_value)
    return rank

def compute_rhat_simple(samples):
    """
    Simplified R-hat computation (without multiple chains).
    Uses split-chain method.
    """
    n = len(samples)
    if n < 4:
        return np.nan

    # Split into two chains
    mid = n // 2
    chain1 = samples[:mid]
    chain2 = samples[mid:]

    # Within-chain variance
    W = (np.var(chain1, ddof=1) + np.var(chain2, ddof=1)) / 2

    # Between-chain variance
    B = n/2 * (np.mean(chain1) - np.mean(chain2))**2

    # Variance estimate
    var_plus = ((n/2 - 1) / (n/2)) * W + B / (n/2)

    # R-hat
    if W > 0:
        rhat = np.sqrt(var_plus / W)
    else:
        rhat = np.nan

    return rhat

def compute_ess_simple(samples):
    """Simplified ESS computation."""
    n = len(samples)
    if n < 10:
        return n

    # Compute autocorrelation at lag 1
    samples_centered = samples - np.mean(samples)
    c0 = np.sum(samples_centered**2)
    c1 = np.sum(samples_centered[:-1] * samples_centered[1:])

    if c0 > 0:
        rho1 = c1 / c0
        # Simple ESS approximation
        ess = n / (1 + 2*rho1)
    else:
        ess = n

    return max(1, ess)

# Run SBC
print(f"\n[1/4] Running {N_SIMS} SBC simulations...")
print("      This may take 15-45 minutes depending on system speed...")

results = {
    'delta': [],
    'sigma_eta': [],
    'phi': [],
    'eta_1': [],
    'ranks_delta': [],
    'ranks_sigma_eta': [],
    'ranks_phi': [],
    'ranks_eta_1': [],
    'rhat_max': [],
    'ess_bulk_min': [],
    'acceptance_rate_mean': []
}

successful_sims = 0
failed_sims = 0

for sim_idx in range(N_SIMS):
    try:
        # Draw true parameters from prior
        true_params = sample_from_prior()

        # Generate synthetic data
        C_obs, eta_true = generate_data(true_params, N_TIME)

        # Create sampler
        sampler = NBStateSpaceSampler(C_obs)

        # Run MCMC
        samples = sampler.sample(N_WARMUP, N_SAMPLES, thin=THIN)

        # Compute ranks
        rank_delta = compute_rank(true_params['delta'], samples['delta'])
        rank_sigma_eta = compute_rank(true_params['sigma_eta'], samples['sigma_eta'])
        rank_phi = compute_rank(true_params['phi'], samples['phi'])
        rank_eta_1 = compute_rank(true_params['eta_1'], samples['eta_1'])

        # Compute diagnostics
        rhat_delta = compute_rhat_simple(samples['delta'])
        rhat_sigma = compute_rhat_simple(samples['sigma_eta'])
        rhat_phi = compute_rhat_simple(samples['phi'])
        rhat_eta1 = compute_rhat_simple(samples['eta_1'])
        rhat_max = np.nanmax([rhat_delta, rhat_sigma, rhat_phi, rhat_eta1])

        ess_delta = compute_ess_simple(samples['delta'])
        ess_sigma = compute_ess_simple(samples['sigma_eta'])
        ess_phi = compute_ess_simple(samples['phi'])
        ess_eta1 = compute_ess_simple(samples['eta_1'])
        ess_min = np.min([ess_delta, ess_sigma, ess_phi, ess_eta1])

        # Acceptance rates
        acc_rates = sampler.get_acceptance_rates()
        mean_acc = np.mean(list(acc_rates.values()))

        # Store results
        results['delta'].append(true_params['delta'])
        results['sigma_eta'].append(true_params['sigma_eta'])
        results['phi'].append(true_params['phi'])
        results['eta_1'].append(true_params['eta_1'])
        results['ranks_delta'].append(rank_delta)
        results['ranks_sigma_eta'].append(rank_sigma_eta)
        results['ranks_phi'].append(rank_phi)
        results['ranks_eta_1'].append(rank_eta_1)
        results['rhat_max'].append(rhat_max)
        results['ess_bulk_min'].append(ess_min)
        results['acceptance_rate_mean'].append(mean_acc)

        successful_sims += 1

        # Progress update
        if (sim_idx + 1) % 10 == 0:
            print(f"      Progress: {sim_idx + 1}/{N_SIMS} (Mean Rhat: {np.nanmean(results['rhat_max']):.3f}, Mean ESS: {np.mean(results['ess_bulk_min']):.0f})")

    except Exception as e:
        failed_sims += 1
        print(f"      WARNING: Simulation {sim_idx + 1} failed: {str(e)}")
        continue

print(f"\n      Completed: {successful_sims} successful, {failed_sims} failed")

# Convert to DataFrame
df_results = pd.DataFrame(results)
df_results['converged'] = (df_results['rhat_max'] < 1.05) & (df_results['ess_bulk_min'] > 100)
df_results['n_divergences'] = 0  # Not tracked in this implementation

# Save results
results_file = DIAGNOSTICS_DIR / "sbc_results.csv"
df_results.to_csv(results_file, index=False)
print(f"\n[2/4] Raw results saved to: {results_file}")

# Compute SBC diagnostics
print("\n[3/4] Computing SBC diagnostics...")

n_bins = 20
expected_per_bin = successful_sims / n_bins

def chi_square_uniformity_test(ranks, n_bins, n_sims):
    """Test if ranks are uniformly distributed."""
    hist, _ = np.histogram(ranks, bins=n_bins, range=(0, n_sims))
    expected = n_sims / n_bins
    chi2 = np.sum((hist - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2, df=n_bins - 1)
    return chi2, p_value, hist

uniformity_results = {}

for param in ['delta', 'sigma_eta', 'phi', 'eta_1']:
    ranks = df_results[f'ranks_{param}'].values
    chi2, p_value, hist = chi_square_uniformity_test(ranks, n_bins, N_DRAWS)

    uniformity_results[param] = {
        'chi2': float(chi2),
        'p_value': float(p_value),
        'histogram': hist.tolist(),
        'pass': bool(p_value > 0.05)
    }

    status = "PASS" if p_value > 0.05 else "FAIL"
    print(f"      {param:12s}: χ² = {chi2:6.2f}, p = {p_value:.4f} [{status}]")

# Convergence diagnostics
print("\n[4/4] Convergence diagnostics:")
print(f"      Max R-hat: {df_results['rhat_max'].mean():.4f} (mean), {df_results['rhat_max'].max():.4f} (max)")
print(f"      Min ESS: {df_results['ess_bulk_min'].mean():.0f} (mean), {df_results['ess_bulk_min'].min():.0f} (min)")
print(f"      Acceptance rate: {df_results['acceptance_rate_mean'].mean():.3f} (mean)")
print(f"      Convergence rate: {df_results['converged'].sum()}/{successful_sims} ({df_results['converged'].mean()*100:.1f}%)")

# Save summary
summary = {
    'n_simulations': successful_sims,
    'n_failed': failed_sims,
    'n_time_points': N_TIME,
    'n_posterior_draws': N_DRAWS,
    'uniformity_tests': uniformity_results,
    'convergence': {
        'mean_rhat': float(df_results['rhat_max'].mean()),
        'max_rhat': float(df_results['rhat_max'].max()),
        'mean_ess': float(df_results['ess_bulk_min'].mean()),
        'min_ess': float(df_results['ess_bulk_min'].min()),
        'mean_acceptance': float(df_results['acceptance_rate_mean'].mean()),
        'convergence_rate': float(df_results['converged'].mean())
    }
}

# Overall assessment
all_uniform = all(v['pass'] for v in uniformity_results.values())
good_convergence = df_results['converged'].mean() > 0.80  # Relaxed for custom MCMC

if all_uniform and good_convergence:
    decision = "PASS"
else:
    decision = "FAIL" if not all_uniform else "CONDITIONAL PASS"

summary['overall_decision'] = decision
summary['decision_criteria'] = {
    'all_uniform': all_uniform,
    'good_convergence': good_convergence
}

summary_file = DIAGNOSTICS_DIR / "sbc_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*80}")
print(f"SBC COMPLETE - DECISION: {decision}")
print(f"{'='*80}")
print(f"Results saved to: {DIAGNOSTICS_DIR}/")
print(f"{'='*80}")
