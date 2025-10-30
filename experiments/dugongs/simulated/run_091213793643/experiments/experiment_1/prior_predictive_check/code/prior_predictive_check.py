"""
Prior Predictive Check for Logarithmic Regression Model

Validates prior distributions by:
1. Sampling from priors without conditioning on data
2. Generating synthetic datasets from prior draws
3. Assessing scientific plausibility of predictions
4. Checking for computational red flags
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from cmdstanpy import CmdStanModel
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
WORKSPACE = Path("/workspace")
DATA_PATH = WORKSPACE / "data" / "data.csv"
EXPERIMENT_DIR = WORKSPACE / "experiments" / "experiment_1"
CODE_DIR = EXPERIMENT_DIR / "prior_predictive_check" / "code"
PLOTS_DIR = EXPERIMENT_DIR / "prior_predictive_check" / "plots"
STAN_MODEL_PATH = CODE_DIR / "logarithmic_model.stan"

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRIOR PREDICTIVE CHECK: Logarithmic Regression")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/6] Loading observed data...")
data = pd.read_csv(DATA_PATH)
N = len(data)
x_obs = data['x'].values
Y_obs = data['Y'].values

print(f"   N = {N} observations")
print(f"   x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"   Y range: [{Y_obs.min():.2f}, {Y_obs.max():.2f}]")

# ============================================================================
# Compile Stan Model
# ============================================================================
print("\n[2/6] Compiling Stan model...")
model = CmdStanModel(stan_file=str(STAN_MODEL_PATH))
print(f"   Model compiled successfully: {STAN_MODEL_PATH.name}")

# ============================================================================
# Sample from Prior Predictive Distribution
# ============================================================================
print("\n[3/6] Sampling from prior predictive distribution...")

# Prepare data for prior predictive sampling
stan_data = {
    'N': N,
    'x': x_obs,
    'Y': Y_obs,  # Not used when prior_only=1, but Stan requires it
    'prior_only': 1  # Flag to skip likelihood evaluation
}

# Sample from priors
n_prior_samples = 1000
prior_fit = model.sample(
    data=stan_data,
    chains=1,
    iter_sampling=n_prior_samples,
    iter_warmup=0,  # No warmup needed, just sampling from priors
    fixed_param=True,  # Use fixed_param algorithm for prior sampling
    show_progress=False,
    show_console=False
)

print(f"   Generated {n_prior_samples} prior predictive samples")

# ============================================================================
# Extract Prior Samples
# ============================================================================
print("\n[4/6] Extracting prior samples and predictions...")

# Extract parameter samples
alpha_prior = prior_fit.stan_variable('alpha')
beta_prior = prior_fit.stan_variable('beta')
sigma_prior = prior_fit.stan_variable('sigma')

# Extract prior predictive samples (Y_rep includes noise)
Y_rep_prior = prior_fit.stan_variable('Y_rep')  # Shape: (n_samples, N)

# Also get Y_pred (without noise) for plotting smooth functions
Y_pred_prior = prior_fit.stan_variable('Y_pred')  # Shape: (n_samples, N)

print(f"   Parameter shapes: alpha {alpha_prior.shape}, beta {beta_prior.shape}, sigma {sigma_prior.shape}")
print(f"   Prediction shape: Y_rep {Y_rep_prior.shape}")

# ============================================================================
# Compute Diagnostic Statistics
# ============================================================================
print("\n[5/6] Computing diagnostic statistics...")

# Create extended x range for smooth function plotting
x_plot = np.linspace(x_obs.min(), x_obs.max(), 100)

# Prior parameter summaries
stats = {
    'Parameter': ['alpha', 'beta', 'sigma'],
    'Mean': [alpha_prior.mean(), beta_prior.mean(), sigma_prior.mean()],
    'SD': [alpha_prior.std(), beta_prior.std(), sigma_prior.std()],
    'Q2.5': [np.percentile(alpha_prior, 2.5), np.percentile(beta_prior, 2.5), np.percentile(sigma_prior, 2.5)],
    'Median': [np.median(alpha_prior), np.median(beta_prior), np.median(sigma_prior)],
    'Q97.5': [np.percentile(alpha_prior, 97.5), np.percentile(beta_prior, 97.5), np.percentile(sigma_prior, 97.5)],
}
stats_df = pd.DataFrame(stats)
print("\n   Prior Parameter Statistics:")
print(stats_df.to_string(index=False))

# Key diagnostic questions
frac_decreasing = (beta_prior < 0).mean()
frac_y_negative = (Y_rep_prior < 0).any(axis=1).mean()  # Any Y < 0 in dataset
frac_y_large = (Y_rep_prior > 5).any(axis=1).mean()     # Any Y > 5 (2x max observed)
frac_y_extreme = (Y_rep_prior > 10).any(axis=1).mean()  # Any Y > 10 (completely unreasonable)

# Coverage: fraction of prior draws where Y_rep covers observed range
y_min_obs, y_max_obs = Y_obs.min(), Y_obs.max()
frac_covers_min = (Y_rep_prior.min(axis=1) <= y_min_obs).mean()
frac_covers_max = (Y_rep_prior.max(axis=1) >= y_max_obs).mean()
frac_covers_both = ((Y_rep_prior.min(axis=1) <= y_min_obs) &
                     (Y_rep_prior.max(axis=1) >= y_max_obs)).mean()

print("\n   Key Diagnostic Checks:")
print(f"   - Fraction with decreasing trend (beta < 0): {frac_decreasing:.1%}")
print(f"   - Fraction predicting any Y < 0: {frac_y_negative:.1%}")
print(f"   - Fraction predicting any Y > 5: {frac_y_large:.1%}")
print(f"   - Fraction predicting any Y > 10: {frac_y_extreme:.1%}")
print(f"   - Fraction covering min(Y_obs): {frac_covers_min:.1%}")
print(f"   - Fraction covering max(Y_obs): {frac_covers_max:.1%}")
print(f"   - Fraction covering full Y_obs range: {frac_covers_both:.1%}")

# ============================================================================
# Create Visualizations
# ============================================================================
print("\n[6/6] Creating visualizations...")

# ----------------------------------------------------------------------------
# Plot 1: Prior Parameter Distributions (Marginals)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Alpha
axes[0].hist(alpha_prior, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(alpha_prior.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {alpha_prior.mean():.3f}')
axes[0].axvline(1.75, color='orange', linestyle='--', linewidth=2, label='Prior center: 1.75')
axes[0].set_xlabel(r'$\alpha$ (Intercept)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(r'Prior: $\alpha \sim$ Normal(1.75, 0.5)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Beta
axes[1].hist(beta_prior, bins=50, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
axes[1].axvline(beta_prior.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {beta_prior.mean():.3f}')
axes[1].axvline(0.27, color='orange', linestyle='--', linewidth=2, label='Prior center: 0.27')
axes[1].axvline(0, color='black', linestyle=':', linewidth=2, label='Zero (flat relationship)')
axes[1].set_xlabel(r'$\beta$ (Log slope)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(r'Prior: $\beta \sim$ Normal(0.27, 0.15)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Sigma
axes[2].hist(sigma_prior, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
axes[2].axvline(sigma_prior.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sigma_prior.mean():.3f}')
axes[2].axvline(0.2, color='orange', linestyle='--', linewidth=2, label='Prior scale: 0.2')
axes[2].set_xlabel(r'$\sigma$ (Residual SD)', fontsize=12)
axes[2].set_ylabel('Density', fontsize=12)
axes[2].set_title(r'Prior: $\sigma \sim$ HalfNormal(0.2)', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot1_path = PLOTS_DIR / "parameter_marginals.png"
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   [1/4] Saved: {plot1_path.name}")

# ----------------------------------------------------------------------------
# Plot 2: Prior Predictive Functions (100 random draws)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

# Plot 100 random prior predictive functions
n_curves = 100
indices = np.random.choice(n_prior_samples, n_curves, replace=False)

for idx in indices:
    alpha_i = alpha_prior[idx]
    beta_i = beta_prior[idx]
    mu_i = alpha_i + beta_i * np.log(x_plot)
    ax.plot(x_plot, mu_i, alpha=0.15, color='steelblue', linewidth=1)

# Overlay observed data
ax.scatter(x_obs, Y_obs, color='red', s=80, alpha=0.8, edgecolors='black',
           linewidth=1.5, label='Observed data', zorder=5)

# Add reference lines
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Y = 0 (impossible)')
ax.axhline(y_max_obs, color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'max(Y_obs) = {y_max_obs:.2f}')
ax.axhline(y_min_obs, color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'min(Y_obs) = {y_min_obs:.2f}')

ax.set_xlabel('x', fontsize=14, fontweight='bold')
ax.set_ylabel('Y', fontsize=14, fontweight='bold')
ax.set_title(f'Prior Predictive Functions: {n_curves} draws from μ(x) = α + β·log(x)',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([-1, 6])  # Wider view to see impossible values

plt.tight_layout()
plot2_path = PLOTS_DIR / "prior_predictive_functions.png"
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   [2/4] Saved: {plot2_path.name}")

# ----------------------------------------------------------------------------
# Plot 3: Prior Predictive Distribution vs Observed Data
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left panel: Distribution of all prior predictions
all_Y_rep = Y_rep_prior.flatten()
axes[0].hist(all_Y_rep, bins=100, density=True, alpha=0.6, color='steelblue',
             edgecolor='black', label='Prior predictive')
axes[0].hist(Y_obs, bins=20, density=True, alpha=0.8, color='red',
             edgecolor='black', label='Observed data')
axes[0].axvline(Y_obs.mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Observed mean: {Y_obs.mean():.2f}')
axes[0].axvline(all_Y_rep.mean(), color='darkblue', linestyle='--', linewidth=2,
                label=f'Prior mean: {all_Y_rep.mean():.2f}')
axes[0].set_xlabel('Y', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=13, fontweight='bold')
axes[0].set_title('Prior Predictive vs Observed Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([-1, 6])

# Right panel: Quantile-quantile plot
axes[1].scatter(np.percentile(all_Y_rep, np.arange(1, 100)),
                np.percentile(Y_obs, np.arange(1, 100)),
                alpha=0.6, s=30, color='steelblue', edgecolors='black')
axes[1].plot([all_Y_rep.min(), all_Y_rep.max()],
             [all_Y_rep.min(), all_Y_rep.max()],
             'r--', linewidth=2, label='Perfect match')
axes[1].set_xlabel('Prior Predictive Quantiles', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Observed Data Quantiles', fontsize=13, fontweight='bold')
axes[1].set_title('Q-Q Plot: Prior vs Observed', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot3_path = PLOTS_DIR / "prior_predictive_coverage.png"
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   [3/4] Saved: {plot3_path.name}")

# ----------------------------------------------------------------------------
# Plot 4: Parameter Relationships and Diagnostic Panel
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Pairwise parameter relationships
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(alpha_prior, beta_prior, alpha=0.3, s=10, color='steelblue')
ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel(r'$\alpha$', fontsize=11, fontweight='bold')
ax1.set_ylabel(r'$\beta$', fontsize=11, fontweight='bold')
ax1.set_title(r'$\alpha$ vs $\beta$', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(alpha_prior, sigma_prior, alpha=0.3, s=10, color='forestgreen')
ax2.set_xlabel(r'$\alpha$', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'$\sigma$', fontsize=11, fontweight='bold')
ax2.set_title(r'$\alpha$ vs $\sigma$', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(beta_prior, sigma_prior, alpha=0.3, s=10, color='coral')
ax3.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel(r'$\beta$', fontsize=11, fontweight='bold')
ax3.set_ylabel(r'$\sigma$', fontsize=11, fontweight='bold')
ax3.set_title(r'$\beta$ vs $\sigma$', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Row 2: Diagnostic checks
ax4 = fig.add_subplot(gs[1, :])
n_show = 50
for i in range(n_show):
    alpha_i = alpha_prior[i]
    beta_i = beta_prior[i]
    sigma_i = sigma_prior[i]

    # Generate function
    mu_i = alpha_i + beta_i * np.log(x_plot)

    # Color by beta sign
    color = 'green' if beta_i > 0 else 'red'
    alpha_val = 0.3 if beta_i > 0 else 0.5

    ax4.plot(x_plot, mu_i, color=color, alpha=alpha_val, linewidth=1)

ax4.scatter(x_obs, Y_obs, color='black', s=60, alpha=0.9, edgecolors='white',
           linewidth=1.5, label='Observed', zorder=5)
ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax4.set_xlabel('x', fontsize=13, fontweight='bold')
ax4.set_ylabel('Y', fontsize=13, fontweight='bold')
ax4.set_title(f'Prior Functions by Sign (Green: β>0 [{(beta_prior>0).sum()}], Red: β<0 [{(beta_prior<0).sum()}])',
             fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-1, 6])

# Row 3: Range coverage diagnostics
ax5 = fig.add_subplot(gs[2, 0])
y_min_prior = Y_rep_prior.min(axis=1)
y_max_prior = Y_rep_prior.max(axis=1)
ax5.scatter(y_min_prior, y_max_prior, alpha=0.2, s=10, color='steelblue')
ax5.axhline(y_max_obs, color='red', linestyle='--', linewidth=2, label=f'max(Y_obs)={y_max_obs:.2f}')
ax5.axvline(y_min_obs, color='orange', linestyle='--', linewidth=2, label=f'min(Y_obs)={y_min_obs:.2f}')
ax5.set_xlabel('min(Y_rep)', fontsize=11, fontweight='bold')
ax5.set_ylabel('max(Y_rep)', fontsize=11, fontweight='bold')
ax5.set_title('Prior Predictive Range Coverage', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
diagnostic_data = pd.DataFrame({
    'Check': ['β < 0\n(decreasing)', 'Any Y < 0\n(impossible)',
              'Any Y > 5\n(large)', 'Any Y > 10\n(extreme)'],
    'Percentage': [frac_decreasing * 100, frac_y_negative * 100,
                   frac_y_large * 100, frac_y_extreme * 100]
})
colors = ['orange' if x < 30 else 'red' for x in diagnostic_data['Percentage']]
ax6.barh(diagnostic_data['Check'], diagnostic_data['Percentage'], color=colors,
         edgecolor='black', linewidth=1.5)
ax6.axvline(20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Warning (20%)')
ax6.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
ax6.set_title('Red Flag Diagnostics', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='x')

ax7 = fig.add_subplot(gs[2, 2])
coverage_data = pd.DataFrame({
    'Coverage': ['Covers\nmin(Y)', 'Covers\nmax(Y)', 'Covers\nboth'],
    'Percentage': [frac_covers_min * 100, frac_covers_max * 100, frac_covers_both * 100]
})
colors = ['green' if x > 80 else 'orange' for x in coverage_data['Percentage']]
ax7.barh(coverage_data['Coverage'], coverage_data['Percentage'], color=colors,
         edgecolor='black', linewidth=1.5)
ax7.axvline(80, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (80%)')
ax7.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
ax7.set_title('Data Range Coverage', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='x')
ax7.set_xlim([0, 105])

plt.suptitle('Prior Diagnostic Panel', fontsize=16, fontweight='bold', y=0.995)
plot4_path = PLOTS_DIR / "diagnostic_panel.png"
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   [4/4] Saved: {plot4_path.name}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("=" * 80)
print(f"\nAll plots saved to: {PLOTS_DIR}")
print("\nNext step: Create findings.md with interpretation and PASS/FAIL decision")
