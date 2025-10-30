"""
Script 6: Summary Visualization
Creates a comprehensive single-figure summary of key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

# Set up paths
DATA_PATH = Path("/workspace/data/data.csv")
VIZ_DIR = Path("/workspace/eda/visualizations")

# Load data
df = pd.read_csv(DATA_PATH)

# Create comprehensive summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# Panel 1: Main scatter with recommended model
# ============================================================================
ax1 = fig.add_subplot(gs[0:2, 0:2])

# Data points
ax1.scatter(df['x'], df['Y'], s=80, alpha=0.7, color='steelblue',
           edgecolor='black', linewidth=1, label='Observed data', zorder=3)

# Logarithmic fit (recommended)
log_x = np.log(df['x'])
z_log = np.polyfit(log_x, df['Y'], 1)
x_line = np.linspace(df['x'].min(), df['x'].max(), 200)
y_log = z_log[1] + z_log[0] * np.log(x_line)
ax1.plot(x_line, y_log, 'r-', linewidth=3, label='Logarithmic fit (R²=0.83)', zorder=2)

# Confidence band (approximate)
residuals = df['Y'] - (z_log[1] + z_log[0] * np.log(df['x']))
se = residuals.std()
ax1.fill_between(x_line, y_log - 1.96*se, y_log + 1.96*se,
                alpha=0.2, color='red', label='95% CI (approx)', zorder=1)

# Mark influential point
ax1.scatter([31.5], [df[df['x']==31.5]['Y'].values[0]],
           s=200, color='orange', edgecolor='red', linewidth=2,
           marker='D', label='Influential point', zorder=4)

# Mark gap region
ax1.axvspan(22.5, 29, alpha=0.15, color='yellow', zorder=0)
ax1.text(25.75, 1.8, 'Data Gap', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax1.set_xlabel('x (Predictor)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Y (Response)', fontsize=13, fontweight='bold')
ax1.set_title('Recommended Model: Y = 1.75 + 0.27·ln(x)',
             fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# ============================================================================
# Panel 2: Residual plot
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(df['x'], residuals, alpha=0.7, s=60, color='coral',
           edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('x', fontsize=10, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax2.set_title('Residual Pattern', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.95, 'No systematic\npattern', transform=ax2.transAxes,
        ha='center', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ============================================================================
# Panel 3: Model comparison
# ============================================================================
ax3 = fig.add_subplot(gs[1, 2])
models = ['Linear', 'Sqrt', 'Asymp', 'Log', 'Quad']
r2_values = [0.518, 0.707, 0.755, 0.829, 0.862]
colors = ['lightcoral' if r2 < 0.8 else 'lightgreen' if r2 < 0.85 else 'gold'
          for r2 in r2_values]
bars = ax3.barh(models, r2_values, color=colors, edgecolor='black', linewidth=1)
ax3.axvline(x=0.829, color='red', linestyle='--', linewidth=2, label='Recommended')
ax3.set_xlabel('R² Value', fontsize=10, fontweight='bold')
ax3.set_title('Model Comparison', fontsize=11, fontweight='bold')
ax3.set_xlim([0, 1])
ax3.grid(True, alpha=0.3, axis='x')
for i, (model, r2) in enumerate(zip(models, r2_values)):
    ax3.text(r2 + 0.02, i, f'{r2:.3f}', va='center', fontsize=8)

# ============================================================================
# Panel 4: Distribution of Y
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(df['Y'], bins=12, density=True, alpha=0.7, color='forestgreen',
        edgecolor='black')
ax4.axvline(df['Y'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={df["Y"].mean():.2f}')
ax4.axvline(df['Y'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median={df["Y"].median():.2f}')
ax4.set_xlabel('Y (Response)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Density', fontsize=10, fontweight='bold')
ax4.set_title('Response Distribution', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ============================================================================
# Panel 5: Variance structure
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])
# Group variance
groups = ['Low\n(x≤7)', 'Mid\n(7<x≤13)', 'High\n(x>13)']
variances = [0.032, 0.012, 0.007]
stds = [np.sqrt(v) for v in variances]
bars = ax5.bar(groups, stds, color=['lightcoral', 'lightyellow', 'lightgreen'],
              edgecolor='black', linewidth=1)
ax5.set_ylabel('Std Dev of Y', fontsize=10, fontweight='bold')
ax5.set_title('Variance by Region', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
for i, (g, s) in enumerate(zip(groups, stds)):
    ax5.text(i, s + 0.005, f'{s:.3f}', ha='center', fontsize=8)

# ============================================================================
# Panel 6: Key statistics
# ============================================================================
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

stats_text = """
KEY FINDINGS

Data Quality:
• N = 27 observations
• No missing values
• 20 unique x values

Relationship:
• Strong positive (ρ=0.78)
• Non-linear (logarithmic)
• Saturation pattern
• R² = 0.83

Model Recommendation:
Y ~ Normal(μ, σ)
μ = α + β·log(x)

Priors (weakly informative):
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
σ ~ HalfNormal(0.2)

Sensitivity Checks:
✓ Test without x=31.5
✓ Compare vs quadratic
✓ Check heteroscedastic σ
"""

ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Overall title
fig.suptitle('EDA Summary: Y vs x Relationship (N=27)',
            fontsize=16, fontweight='bold', y=0.98)

# Save
plt.savefig(VIZ_DIR / 'eda_summary_comprehensive.png', bbox_inches='tight', dpi=300)
print("Comprehensive summary visualization saved: eda_summary_comprehensive.png")
plt.close()

# ============================================================================
# Also create a simple 1-panel summary for presentations
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Data points
ax.scatter(df['x'], df['Y'], s=100, alpha=0.7, color='steelblue',
          edgecolor='black', linewidth=1.5, label='Observed data (N=27)', zorder=3)

# Logarithmic fit
x_line = np.linspace(df['x'].min(), df['x'].max(), 200)
y_log = z_log[1] + z_log[0] * np.log(x_line)
ax.plot(x_line, y_log, 'r-', linewidth=3.5,
       label=f'Logarithmic: Y = {z_log[1]:.2f} + {z_log[0]:.2f}·ln(x)\nR² = 0.83',
       zorder=2)

# Confidence band
ax.fill_between(x_line, y_log - 1.96*se, y_log + 1.96*se,
                alpha=0.2, color='red', label='95% Confidence Band', zorder=1)

ax.set_xlabel('x (Predictor)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y (Response)', fontsize=14, fontweight='bold')
ax.set_title('Non-linear Relationship: Y vs x\nLogarithmic Model Recommended',
            fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.4)

# Add text box with key info
textstr = 'Strong positive relationship\nSaturation pattern observed\nSpearman ρ = 0.78 (p < 0.001)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
       verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'eda_summary_simple.png', bbox_inches='tight', dpi=300)
print("Simple summary visualization saved: eda_summary_simple.png")
plt.close()

print("\nSummary visualizations complete!")
