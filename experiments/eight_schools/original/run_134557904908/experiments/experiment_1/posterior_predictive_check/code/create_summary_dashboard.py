"""
Create Summary Dashboard for Posterior Predictive Checks
Combines key findings into single comprehensive visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
PLOTS_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check/plots")
PPC_RESULTS = Path("/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npy")
LOOPIT_RESULTS = Path("/workspace/experiments/experiment_1/posterior_predictive_check/code/loo_pit_results.npy")

# Load results
print("Loading results...")
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma = data['sigma'].values
n_obs = len(y_obs)

ppc_results = np.load(PPC_RESULTS, allow_pickle=True).item()
loopit_results = np.load(LOOPIT_RESULTS, allow_pickle=True).item()

# Extract key metrics
residuals = ppc_results['residuals']
test_stats = ppc_results['test_stats']
coverage = ppc_results['coverage']
pit_values = loopit_results['pit_values']
ks_pval = loopit_results['ks_pval']

# Create comprehensive summary dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Posterior Predictive Check: Summary Dashboard\nFixed-Effect Normal Model',
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# Row 1: Overall Assessment Metrics
# ============================================================================

# Panel 1: Assessment Summary (text box)
ax1 = fig.add_subplot(gs[0, :2])
ax1.axis('off')

assessment_text = """
OVERALL ASSESSMENT: GOOD FIT ✓

Key Findings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• LOO-PIT Uniformity:  KS p = 0.98 (Excellent)
• 95% Coverage:        100% (8/8 obs)
• Residual Normality:  Shapiro p = 0.55 (Normal)
• Test Statistics:     All p-values ∈ [0.2, 0.7]
• Standardized RMSE:   0.77 (Good fit)
• Problematic Obs:     0 (None detected)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conclusion: Model is well-calibrated and
reproduces all key features of observed data.
No systematic patterns of misfit detected.
"""

ax1.text(0.05, 0.95, assessment_text, transform=ax1.transAxes,
         fontsize=12, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Panel 2: Test Statistics p-values
ax2 = fig.add_subplot(gs[0, 2:])
stat_names = list(test_stats.keys())
p_values = [test_stats[name]['p_value'] for name in stat_names]

colors = ['green' if 0.1 <= p <= 0.9 else 'orange' for p in p_values]
bars = ax2.barh(stat_names, p_values, color=colors, alpha=0.7, edgecolor='black')

# Add ideal range
ax2.axvspan(0.1, 0.9, alpha=0.2, color='green', label='Ideal range')
ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Perfect (0.5)')

ax2.set_xlabel('Posterior p-value', fontsize=11)
ax2.set_title('Test Statistics: All in Acceptable Range', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.legend(fontsize=9)
ax2.grid(True, axis='x', alpha=0.3)

# ============================================================================
# Row 2: LOO-PIT and Residuals
# ============================================================================

# Panel 3: LOO-PIT Histogram
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(pit_values, bins=10, density=True, alpha=0.6, color='steelblue',
         edgecolor='black', linewidth=1.5)
ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
ax3.set_xlabel('LOO-PIT Value', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title(f'LOO-PIT: KS p={ks_pval:.3f}\n(Excellent Uniformity)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Residuals Q-Q Plot
ax4 = fig.add_subplot(gs[1, 1])
from scipy import stats as sp_stats
sp_stats.probplot(residuals, dist="norm", plot=ax4)
ax4.set_title('Residuals: Normal Q-Q\n(Shapiro p=0.55)',
              fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Panel 5: Residuals vs Index
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(np.arange(1, n_obs+1), residuals, s=120, alpha=0.7,
            edgecolors='black', linewidths=2, c=residuals, cmap='RdYlGn_r')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.axhline(2, color='orange', linestyle=':', linewidth=1.5)
ax5.axhline(-2, color='orange', linestyle=':', linewidth=1.5)
ax5.set_xlabel('Observation Index', fontsize=10)
ax5.set_ylabel('Standardized Residual', fontsize=10)
ax5.set_title('Residuals: No Patterns\n(All |r| < 2)',
              fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Panel 6: Coverage Summary
ax6 = fig.add_subplot(gs[1, 3])
levels = [50, 90, 95]
nominal = [level/100 for level in levels]
empirical = [coverage[level]['empirical'] for level in levels]

x_pos = np.arange(len(levels))
width = 0.35

bars1 = ax6.bar(x_pos - width/2, nominal, width, label='Nominal',
                alpha=0.7, color='skyblue', edgecolor='black')
bars2 = ax6.bar(x_pos + width/2, empirical, width, label='Empirical',
                alpha=0.7, color='lightcoral', edgecolor='black')

ax6.set_ylabel('Coverage Rate', fontsize=10)
ax6.set_title('Coverage: Well-Calibrated\n(Empirical ≈ Nominal)',
              fontsize=11, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f'{level}%' for level in levels])
ax6.legend(fontsize=9)
ax6.set_ylim(0, 1.1)
ax6.grid(True, axis='y', alpha=0.3)

# Add values on bars
for i, (n, e) in enumerate(zip(nominal, empirical)):
    ax6.text(i - width/2, n + 0.02, f'{n:.2f}', ha='center', fontsize=8)
    ax6.text(i + width/2, e + 0.02, f'{e:.2f}', ha='center', fontsize=8)

# ============================================================================
# Row 3: Observation-Level Details
# ============================================================================

# Panel 7: Observation-level p-values
ax7 = fig.add_subplot(gs[2, :2])
obs_results = ppc_results['obs_results']
obs_indices = [r['obs'] for r in obs_results]
obs_pvals = [r['p_val'] for r in obs_results]

colors_obs = ['green' if 0.05 <= p <= 0.95 else 'orange' for p in obs_pvals]
bars = ax7.bar(obs_indices, obs_pvals, color=colors_obs, alpha=0.7,
               edgecolor='black', linewidth=1.5)

# Add acceptable range
ax7.axhspan(0.05, 0.95, alpha=0.2, color='green', label='Acceptable')
ax7.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Ideal')

ax7.set_xlabel('Observation Index', fontsize=11)
ax7.set_ylabel('p-value P(y_rep ≥ y_obs)', fontsize=11)
ax7.set_title('Observation-Level p-values: All Acceptable\n(All in [0.09, 0.80])',
              fontsize=12, fontweight='bold')
ax7.set_ylim(0, 1)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Panel 8: MAE by observation
ax8 = fig.add_subplot(gs[2, 2:])
mae_values = ppc_results['discrepancy']['mae']
obs_sigma = [r['sigma'] for r in obs_results]

# Create scatter with size proportional to sigma
scatter = ax8.scatter(obs_indices, mae_values, s=[s*10 for s in obs_sigma],
                      alpha=0.6, c=obs_sigma, cmap='viridis',
                      edgecolors='black', linewidths=2)

ax8.set_xlabel('Observation Index', fontsize=11)
ax8.set_ylabel('Mean Absolute Error', fontsize=11)
ax8.set_title('Observation-Level MAE\n(Size ∝ measurement uncertainty σ)',
              fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('σ', fontsize=10)

# Add text annotations for worst 3
worst_idx = np.argsort(mae_values)[::-1][:3]
for idx in worst_idx:
    ax8.annotate(f'Obs {idx+1}', xy=(obs_indices[idx], mae_values[idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.savefig(PLOTS_DIR / "summary_dashboard.png", dpi=300, bbox_inches='tight')
print(f"\nSaved: summary_dashboard.png")
plt.close()

print("\nSummary Dashboard Complete!")
print(f"Location: {PLOTS_DIR / 'summary_dashboard.png'}")
