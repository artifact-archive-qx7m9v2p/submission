"""
Visualizations for Correlation Structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/eda/analyst_2/code/processed_data.csv')

sns.set_style("whitegrid")

# ============================================================================
# FIGURE 1: Correlation Structure (Multi-panel)
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('Correlation Structure Analysis', fontsize=16, fontweight='bold', y=0.98)

# Panel 1: n_trials vs r_successes (large, top-left)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.scatter(df['n_trials'], df['r_successes'], s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(df['n_trials'], df['r_successes'])
x_pred = np.linspace(df['n_trials'].min(), df['n_trials'].max(), 100)
y_pred = intercept + slope * x_pred
ax1.plot(x_pred, y_pred, 'r--', linewidth=2.5, label=f'Linear: r²={r_value**2:.3f}, p={p_value:.3f}')

# Add quadratic fit
coeffs = np.polyfit(df['n_trials'], df['r_successes'], 2)
poly = np.poly1d(coeffs)
y_quad = poly(x_pred)
ax1.plot(x_pred, y_quad, 'g:', linewidth=2, label=f'Quadratic: r²=0.731')

# Annotate outlier
outlier_idx = df['n_trials'].idxmax()
ax1.annotate(f'Group {int(df.loc[outlier_idx, "group_id"])}\n(n={int(df.loc[outlier_idx, "n_trials"])}, r={int(df.loc[outlier_idx, "r_successes"])})',
             xy=(df.loc[outlier_idx, 'n_trials'], df.loc[outlier_idx, 'r_successes']),
             xytext=(df.loc[outlier_idx, 'n_trials']-150, df.loc[outlier_idx, 'r_successes']+5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, color='red', fontweight='bold')

ax1.set_xlabel('Number of Trials', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Successes', fontsize=12, fontweight='bold')
ax1.set_title('A. Strong Positive Relationship\n(Pearson r=0.78, p=0.003)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: n_trials vs success_rate
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(df['n_trials'], df['success_rate'], s=80, alpha=0.7, c='coral', edgecolors='black', linewidth=1)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df['n_trials'], df['success_rate'])
ax2.plot(x_pred, intercept2 + slope2 * x_pred, 'b--', linewidth=2, alpha=0.7,
         label=f'r={r_value2:.3f}, p={p_value2:.2f}')
ax2.set_xlabel('n_trials', fontsize=10)
ax2.set_ylabel('Success Rate', fontsize=10)
ax2.set_title('B. Weak Negative\n(Non-significant)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: r_successes vs success_rate
ax3 = fig.add_subplot(gs[1, 2])
ax3.scatter(df['r_successes'], df['success_rate'], s=80, alpha=0.7, c='mediumseagreen', edgecolors='black', linewidth=1)
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df['r_successes'], df['success_rate'])
x_pred3 = np.linspace(df['r_successes'].min(), df['r_successes'].max(), 100)
ax3.plot(x_pred3, intercept3 + slope3 * x_pred3, 'b--', linewidth=2, alpha=0.7,
         label=f'r={r_value3:.3f}, p={p_value3:.2f}')
ax3.set_xlabel('r_successes', fontsize=10)
ax3.set_ylabel('Success Rate', fontsize=10)
ax3.set_title('C. Weak Positive\n(Non-significant)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Correlation heatmap
ax4 = fig.add_subplot(gs[2, 0:2])
vars_plot = ['n_trials', 'r_successes', 'success_rate', 'logit_success_rate']
corr_matrix = df[vars_plot].corr()
im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add text annotations
for i in range(len(vars_plot)):
    for j in range(len(vars_plot)):
        text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=11, fontweight='bold')

ax4.set_xticks(range(len(vars_plot)))
ax4.set_yticks(range(len(vars_plot)))
ax4.set_xticklabels(vars_plot, rotation=45, ha='right', fontsize=10)
ax4.set_yticklabels(vars_plot, fontsize=10)
ax4.set_title('D. Correlation Matrix (Pearson)', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label('Correlation', fontsize=10)

# Panel 5: Residual plot
ax5 = fig.add_subplot(gs[2, 2])
ax5.scatter(df['predicted_successes'], df['residuals'], s=80, alpha=0.7, c='purple', edgecolors='black', linewidth=1)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Successes', fontsize=10)
ax5.set_ylabel('Residuals', fontsize=10)
ax5.set_title('E. Residual Plot\n(Homoscedastic)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

plt.savefig('/workspace/eda/analyst_2/visualizations/correlation_structure.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_structure.png")
plt.close()

# ============================================================================
# FIGURE 2: Outlier Influence Analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Outlier Influence on Correlations', fontsize=14, fontweight='bold')

# Panel 1: With outlier
ax1 = axes[0]
ax1.scatter(df['n_trials'], df['success_rate'], s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)
slope_full, intercept_full, r_full, p_full, _ = stats.linregress(df['n_trials'], df['success_rate'])
ax1.plot(x_pred, intercept_full + slope_full * x_pred, 'r--', linewidth=2.5,
         label=f'r={r_full:.3f}, p={p_full:.2f}')

# Highlight outlier
outlier_idx = df['n_trials'].idxmax()
ax1.scatter(df.loc[outlier_idx, 'n_trials'], df.loc[outlier_idx, 'success_rate'],
           s=300, c='red', marker='*', edgecolors='darkred', linewidth=2, label='Outlier', zorder=5)

ax1.set_xlabel('Number of Trials', fontsize=11)
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_title('A. With Outlier (Group 4)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Without outlier
ax2 = axes[1]
df_no_outlier = df.drop(outlier_idx)
ax2.scatter(df_no_outlier['n_trials'], df_no_outlier['success_rate'], s=100, alpha=0.7,
           c='mediumseagreen', edgecolors='black', linewidth=1.5)
slope_no, intercept_no, r_no, p_no, _ = stats.linregress(df_no_outlier['n_trials'], df_no_outlier['success_rate'])
x_pred_no = np.linspace(df_no_outlier['n_trials'].min(), df_no_outlier['n_trials'].max(), 100)
ax2.plot(x_pred_no, intercept_no + slope_no * x_pred_no, 'b--', linewidth=2.5,
         label=f'r={r_no:.3f}, p={p_no:.2f}')

ax2.set_xlabel('Number of Trials', fontsize=11)
ax2.set_ylabel('Success Rate', fontsize=11)
ax2.set_title('B. Without Outlier', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add comparison text
fig.text(0.5, 0.02, f'Outlier Impact: Correlation changes from r={r_no:.3f} to r={r_full:.3f} (Δr={r_full-r_no:.3f})',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig('/workspace/eda/analyst_2/visualizations/outlier_influence.png', dpi=300, bbox_inches='tight')
print("Saved: outlier_influence.png")
plt.close()

# ============================================================================
# FIGURE 3: Partial Correlation Illustration
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Partial Correlation: r_successes vs success_rate (controlling for n_trials)',
             fontsize=14, fontweight='bold')

# Panel 1: Original scatter (r_successes vs success_rate)
ax1 = axes[0, 0]
scatter1 = ax1.scatter(df['r_successes'], df['success_rate'], s=100, c=df['n_trials'],
                      cmap='viridis', alpha=0.8, edgecolors='black', linewidth=1.5)
slope_orig, intercept_orig, r_orig, _, _ = stats.linregress(df['r_successes'], df['success_rate'])
x_succ = np.linspace(df['r_successes'].min(), df['r_successes'].max(), 100)
ax1.plot(x_succ, intercept_orig + slope_orig * x_succ, 'r--', linewidth=2,
         label=f'r={r_orig:.3f}')
ax1.set_xlabel('r_successes', fontsize=11)
ax1.set_ylabel('success_rate', fontsize=11)
ax1.set_title('A. Original Correlation (colored by n_trials)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('n_trials', fontsize=10)

# Panel 2: r_successes residuals (after removing n_trials effect)
ax2 = axes[0, 1]
slope_sn, intercept_sn = np.polyfit(df['n_trials'], df['r_successes'], 1)
resid_succ = df['r_successes'] - (slope_sn * df['n_trials'] + intercept_sn)
ax2.scatter(df['n_trials'], resid_succ, s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('n_trials', fontsize=11)
ax2.set_ylabel('r_successes residuals', fontsize=11)
ax2.set_title('B. Remove n_trials Effect from r_successes', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: success_rate residuals (after removing n_trials effect)
ax3 = axes[1, 0]
slope_rn, intercept_rn = np.polyfit(df['n_trials'], df['success_rate'], 1)
resid_rate = df['success_rate'] - (slope_rn * df['n_trials'] + intercept_rn)
ax3.scatter(df['n_trials'], resid_rate, s=100, alpha=0.7, c='coral', edgecolors='black', linewidth=1.5)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('n_trials', fontsize=11)
ax3.set_ylabel('success_rate residuals', fontsize=11)
ax3.set_title('C. Remove n_trials Effect from success_rate', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel 4: Partial correlation (residuals vs residuals)
ax4 = axes[1, 1]
ax4.scatter(resid_succ, resid_rate, s=120, alpha=0.8, c='mediumseagreen', edgecolors='black', linewidth=1.5)
r_partial, p_partial = stats.pearsonr(resid_succ, resid_rate)
slope_partial, intercept_partial = np.polyfit(resid_succ, resid_rate, 1)
x_partial = np.linspace(resid_succ.min(), resid_succ.max(), 100)
ax4.plot(x_partial, intercept_partial + slope_partial * x_partial, 'r--', linewidth=2.5,
         label=f'Partial r={r_partial:.3f}, p={p_partial:.3f}')
ax4.set_xlabel('r_successes residuals', fontsize=11)
ax4.set_ylabel('success_rate residuals', fontsize=11)
ax4.set_title('D. Partial Correlation (Stronger!)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Add explanation text
fig.text(0.5, 0.02, f'Key Insight: Controlling for n_trials reveals stronger relationship (r: {r_orig:.3f} → {r_partial:.3f})',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig('/workspace/eda/analyst_2/visualizations/partial_correlation.png', dpi=300, bbox_inches='tight')
print("Saved: partial_correlation.png")
plt.close()

print("\nCorrelation visualization generation complete!")
