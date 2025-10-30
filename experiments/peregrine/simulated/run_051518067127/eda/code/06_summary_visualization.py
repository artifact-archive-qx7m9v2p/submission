"""
Summary Visualization: Key EDA Findings
=======================================
Goal: Create a single comprehensive figure summarizing main findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data.csv')
data['time_idx'] = np.arange(len(data))
data['period'] = pd.cut(data['time_idx'], bins=3, labels=['Early', 'Middle', 'Late'])

# Create summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Main title
fig.suptitle('Count Time Series EDA: Key Findings Summary',
             fontsize=18, fontweight='bold', y=0.98)

# 1. Main relationship plot (large, top-left spanning 2 cols)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.scatter(data['year'], data['C'], alpha=0.6, s=80, c=data['time_idx'],
           cmap='viridis', edgecolor='black', linewidth=0.5)

# Fit and plot both linear and exponential
slope, intercept, r_val, p_val, _ = stats.linregress(data['year'], data['C'])
x_line = np.linspace(data['year'].min(), data['year'].max(), 100)
y_linear = slope * x_line + intercept

# Exponential fit
log_C = np.log(data['C'])
slope_log, intercept_log, r_val_log, _, _ = stats.linregress(data['year'], log_C)
y_exp = np.exp(slope_log * x_line + intercept_log)

ax1.plot(x_line, y_linear, 'r--', linewidth=2.5, label=f'Linear: R²={r_val**2:.3f}', alpha=0.8)
ax1.plot(x_line, y_exp, 'g-', linewidth=2.5, label=f'Exponential: R²={r_val_log**2:.3f}', alpha=0.8)

ax1.set_xlabel('Standardized Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count (C)', fontsize=12, fontweight='bold')
ax1.set_title('Finding 1: Strong Exponential Growth Pattern',
             fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(ax1.collections[0], ax=ax1, pad=0.02)
cbar.set_label('Time Index', fontsize=10)

# 2. Overdispersion illustration (top-right)
ax2 = fig.add_subplot(gs[0, 2])
periods = ['Early', 'Middle', 'Late', 'Overall']
means = []
variances = []
colors_bar = ['lightblue', 'lightgreen', 'lightcoral', 'red']

for period in ['Early', 'Middle', 'Late']:
    p_data = data[data['period'] == period]['C']
    means.append(p_data.mean())
    variances.append(p_data.var(ddof=1))

means.append(data['C'].mean())
variances.append(data['C'].var(ddof=1))

x_pos = np.arange(len(periods))
ax2.scatter(means, variances, s=[200, 200, 200, 400], c=colors_bar,
           edgecolor='black', linewidth=2, alpha=0.7, zorder=3)

# Poisson reference line
mean_range = np.linspace(0, max(means) * 1.1, 100)
ax2.plot(mean_range, mean_range, 'k--', linewidth=2, label='Poisson\n(var=mean)', alpha=0.7)

for i, period in enumerate(periods):
    ax2.annotate(period, (means[i], variances[i]),
                xytext=(5, -10 if i < 3 else 10), textcoords='offset points',
                fontsize=9, fontweight='bold')

ax2.set_xlabel('Mean', fontsize=11, fontweight='bold')
ax2.set_ylabel('Variance', fontsize=11, fontweight='bold')
ax2.set_title('Finding 2: Severe Overdispersion\n(Var/Mean = 70.4)',
             fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# 3. Autocorrelation (middle-right)
ax3 = fig.add_subplot(gs[1, 2])
max_lag = 10
acf_values = []
for lag in range(max_lag + 1):
    if lag == 0:
        acf_values.append(1.0)
    else:
        acf_values.append(data['C'].autocorr(lag=lag))

lags = np.arange(len(acf_values))
ax3.bar(lags, acf_values, width=0.5, color='steelblue', alpha=0.7, edgecolor='black')
conf_level = 1.96 / np.sqrt(len(data))
ax3.axhline(conf_level, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95% CI')
ax3.axhline(-conf_level, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axhline(0, color='black', linewidth=0.5)

ax3.set_xlabel('Lag', fontsize=11, fontweight='bold')
ax3.set_ylabel('ACF', fontsize=11, fontweight='bold')
ax3.set_title('Finding 3: High Autocorrelation\n(DW = 0.47)',
             fontsize=11, fontweight='bold', pad=10)
ax3.set_xlim(-0.5, max_lag + 0.5)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# 4. Time period box plots (bottom-left)
ax4 = fig.add_subplot(gs[2, 0])
period_data_list = [
    data[data['period'] == 'Early']['C'].values,
    data[data['period'] == 'Middle']['C'].values,
    data[data['period'] == 'Late']['C'].values
]
bp = ax4.boxplot(period_data_list, labels=['Early\n(n=14)', 'Middle\n(n=13)', 'Late\n(n=13)'],
                patch_artist=True, widths=0.6)

for i, box in enumerate(bp['boxes']):
    box.set_facecolor(colors_bar[i])
    box.set_edgecolor('black')
    box.set_linewidth(1.5)

# Add means
for i, period in enumerate(['Early', 'Middle', 'Late']):
    p_data = data[data['period'] == period]['C']
    ax4.plot(i+1, p_data.mean(), 'r*', markersize=15, markeredgecolor='black',
            markeredgewidth=0.5)

ax4.set_ylabel('Count (C)', fontsize=11, fontweight='bold')
ax4.set_title('Finding 4: Regime Shifts\n(ANOVA p < 0.001)',
             fontsize=11, fontweight='bold', pad=10)
ax4.grid(alpha=0.3, axis='y')

# Add mean values as text
for i, period in enumerate(['Early', 'Middle', 'Late']):
    p_data = data[data['period'] == period]['C']
    ax4.text(i+1, ax4.get_ylim()[1] * 0.95, f'μ={p_data.mean():.1f}',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5. Distribution comparison (bottom-middle)
ax5 = fig.add_subplot(gs[2, 1])

# Observed distribution
counts, bins, _ = ax5.hist(data['C'], bins=20, density=True, alpha=0.6,
                          color='steelblue', edgecolor='black', linewidth=1,
                          label='Observed')

# Poisson overlay
mean_C = data['C'].mean()
x_range = np.arange(data['C'].min(), data['C'].max() + 1)
poisson_pmf = stats.poisson.pmf(x_range, mean_C)
ax5.plot(x_range, poisson_pmf, 'ro-', linewidth=2, markersize=5,
        label=f'Poisson(λ={mean_C:.0f})', alpha=0.7)

ax5.set_xlabel('Count (C)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Density', fontsize=11, fontweight='bold')
ax5.set_title('Finding 5: Non-Poisson\n(GoF p < 0.001)',
             fontsize=11, fontweight='bold', pad=10)
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3, axis='y')

# 6. Key statistics table (bottom-right)
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

# Create statistics text
stats_text = [
    ['Statistic', 'Value'],
    ['─' * 15, '─' * 10],
    ['Observations', '40'],
    ['Count Range', '21-269'],
    ['Mean', '109.4'],
    ['Variance', '7704.7'],
    ['Var/Mean Ratio', '70.4 ⚠'],
    ['', ''],
    ['Linear R²', '0.881'],
    ['Log-linear R²', '0.937 ✓'],
    ['Quadratic R²', '0.964 ✓✓'],
    ['', ''],
    ['ACF Lag-1', '0.971'],
    ['Durbin-Watson', '0.47 ⚠'],
    ['', ''],
    ['Growth Rate', '5.5%/step'],
    ['7.8× increase', 'Early→Late'],
]

y_pos = 0.95
for row in stats_text:
    if row[0] == '─' * 15:
        ax6.text(0.05, y_pos, '─' * 35, fontsize=9, family='monospace',
                verticalalignment='top')
    elif row[0] == '':
        y_pos -= 0.04
        continue
    else:
        text = f'{row[0]:<18} {row[1]:>10}'
        fontweight = 'bold' if row[0] == 'Statistic' else 'normal'
        fontsize = 10 if row[0] == 'Statistic' else 9
        ax6.text(0.05, y_pos, text, fontsize=fontsize, family='monospace',
                verticalalignment='top', fontweight=fontweight)
    y_pos -= 0.05

ax6.set_title('Key Statistics', fontsize=11, fontweight='bold', pad=10)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

# Add overall summary at bottom
fig.text(0.5, 0.01,
         'Summary: Exponential growth (R²=0.937) | Severe overdispersion (70×) | High autocorrelation (ACF=0.97) | Regime shifts (ANOVA p<0.001) | Use NB/log-normal, not Poisson',
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.savefig('/workspace/eda/visualizations/00_summary_findings.png', dpi=300, bbox_inches='tight')
print("Saved: 00_summary_findings.png")
print("\nSummary visualization created successfully!")
plt.close()
