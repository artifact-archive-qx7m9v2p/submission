"""
Create Experiment 1 vs Experiment 2 comparison plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
exp1_summary = pd.read_csv("/workspace/experiments/experiment_1/posterior_predictive_check/code/test_statistics_summary.csv")
exp2_summary = pd.read_csv("/workspace/experiments/experiment_2/posterior_predictive_check/code/test_statistics_summary.csv")

# Create mapping for Exp1 statistics
exp1_mapping = {
    'acf_lag1': 'ACF lag-1',
    'var_mean_ratio': 'Variance/Mean Ratio',
    'max': 'Maximum',
    'range': 'Range'
}

# Extract data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: ACF lag-1 distributions
ax = axes[0, 0]
exp1_acf_row = exp1_summary[exp1_summary['Statistic'] == 'acf_lag1'].iloc[0]
exp2_acf_row = exp2_summary[exp2_summary['Statistic'] == 'ACF lag-1'].iloc[0]

exp1_acf_dist = np.random.normal(exp1_acf_row['Replicated_Mean'], exp1_acf_row['Replicated_SD'], 1000)
exp2_acf_dist = np.random.normal(exp2_acf_row['Rep_Mean'], exp2_acf_row['Rep_SD'], 1000)

ax.hist(exp1_acf_dist, bins=40, alpha=0.5, color='orange', edgecolor='darkorange', density=True, label='Exp 1 (NB GLM)')
ax.hist(exp2_acf_dist, bins=40, alpha=0.5, color='lightblue', edgecolor='blue', density=True, label='Exp 2 (AR(1) Log-Normal)')
ax.axvline(exp2_acf_row['Observed'], color='red', linewidth=2.5, label=f'Observed: {exp2_acf_row["Observed"]:.3f}')
ax.set_xlabel('ACF Lag-1', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('A. ACF Lag-1 Distributions (Exp1 vs Exp2)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel B: Variance comparison
ax = axes[0, 1]
exp1_var_row = exp1_summary[exp1_summary['Statistic'] == 'variance'].iloc[0]
exp2_var_row = exp2_summary[exp2_summary['Statistic'] == 'Variance'].iloc[0]

exp1_var_dist = np.random.normal(exp1_var_row['Replicated_Mean'], exp1_var_row['Replicated_SD'], 1000)
exp2_var_dist = np.random.normal(exp2_var_row['Rep_Mean'], exp2_var_row['Rep_SD'], 1000)

ax.hist(exp1_var_dist, bins=40, alpha=0.5, color='orange', edgecolor='darkorange', density=True, label='Exp 1 (NB GLM)')
ax.hist(exp2_var_dist, bins=40, alpha=0.5, color='lightblue', edgecolor='blue', density=True, label='Exp 2 (AR(1) Log-Normal)')
ax.axvline(exp2_var_row['Observed'], color='red', linewidth=2.5, label=f'Observed: {exp2_var_row["Observed"]:.0f}')
ax.set_xlabel('Variance', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('B. Variance Distributions (Exp1 vs Exp2)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel C: Maximum value comparison
ax = axes[1, 0]
exp1_max_row = exp1_summary[exp1_summary['Statistic'] == 'max'].iloc[0]
exp2_max_row = exp2_summary[exp2_summary['Statistic'] == 'Maximum'].iloc[0]

exp1_max_dist = np.random.normal(exp1_max_row['Replicated_Mean'], exp1_max_row['Replicated_SD'], 1000)
exp2_max_dist = np.random.normal(exp2_max_row['Rep_Mean'], exp2_max_row['Rep_SD'], 1000)

ax.hist(exp1_max_dist, bins=40, alpha=0.5, color='orange', edgecolor='darkorange', density=True, label='Exp 1 (NB GLM)')
ax.hist(exp2_max_dist, bins=40, alpha=0.5, color='lightblue', edgecolor='blue', density=True, label='Exp 2 (AR(1) Log-Normal)')
ax.axvline(exp2_max_row['Observed'], color='red', linewidth=2.5, label=f'Observed: {exp2_max_row["Observed"]:.0f}')
ax.set_xlabel('Maximum Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('C. Maximum Value Distributions (Exp1 vs Exp2)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel D: Summary comparison table
ax = axes[1, 1]
ax.axis('off')

# Create comparison table
comparison_data = [
    ['ACF lag-1',
     f"{exp1_acf_row['Replicated_Mean']:.3f}",
     f"{exp2_acf_row['Rep_Mean']:.3f}",
     f"{exp2_acf_row['Observed']:.3f}",
     'FAIL' if exp1_acf_row['Extreme'] == '***' else 'PASS',
     exp2_acf_row['Result']],
    ['Variance/Mean',
     f"{exp1_summary[exp1_summary['Statistic'] == 'var_mean_ratio'].iloc[0]['Replicated_Mean']:.2f}",
     f"{exp2_summary[exp2_summary['Statistic'] == 'Variance/Mean Ratio'].iloc[0]['Rep_Mean']:.2f}",
     f"{exp2_summary[exp2_summary['Statistic'] == 'Variance/Mean Ratio'].iloc[0]['Observed']:.2f}",
     'PASS',
     exp2_summary[exp2_summary['Statistic'] == 'Variance/Mean Ratio'].iloc[0]['Result']],
    ['Maximum',
     f"{exp1_max_row['Replicated_Mean']:.1f}",
     f"{exp2_max_row['Rep_Mean']:.1f}",
     f"{exp2_max_row['Observed']:.1f}",
     'FAIL',
     exp2_max_row['Result']],
    ['Range',
     f"{exp1_summary[exp1_summary['Statistic'] == 'range'].iloc[0]['Replicated_Mean']:.1f}",
     f"{exp2_summary[exp2_summary['Statistic'] == 'Range'].iloc[0]['Rep_Mean']:.1f}",
     f"{exp2_summary[exp2_summary['Statistic'] == 'Range'].iloc[0]['Observed']:.1f}",
     'FAIL',
     exp2_summary[exp2_summary['Statistic'] == 'Range'].iloc[0]['Result']],
]

table = ax.table(cellText=comparison_data,
                colLabels=['Metric', 'Exp1\nMean', 'Exp2\nMean', 'Observed', 'Exp1\nResult', 'Exp2\nResult'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code results
for i in range(1, len(comparison_data) + 1):
    # Exp1 result
    if comparison_data[i-1][4] == 'PASS':
        table[(i, 4)].set_facecolor('#90EE90')
    else:
        table[(i, 4)].set_facecolor('#FFB6C6')

    # Exp2 result
    if comparison_data[i-1][5] == 'PASS':
        table[(i, 5)].set_facecolor('#90EE90')
    else:
        table[(i, 5)].set_facecolor('#FFB6C6')

ax.set_title('D. Comparison Summary (Exp1 vs Exp2)', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_predictive_check/plots/comparison_exp1_vs_exp2.png', dpi=300, bbox_inches='tight')
plt.close()

print("Comparison plot created successfully!")
