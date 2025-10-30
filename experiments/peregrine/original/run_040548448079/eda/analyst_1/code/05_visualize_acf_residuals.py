"""
Visualization: ACF/PACF and Residual Diagnostics
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)

with open('/workspace/eda/analyst_1/code/trend_models.pkl', 'rb') as f:
    model_results = pickle.load(f)

with open('/workspace/eda/analyst_1/code/acf_data.pkl', 'rb') as f:
    acf_data = pickle.load(f)

models = model_results['models']
X = model_results['X']
y = model_results['y']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# PLOT 4: ACF and PACF for Raw Data and First Differences
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

nlags = 15
confidence_interval = 1.96 / np.sqrt(len(y))

# ACF - Raw Data
ax = axes[0, 0]
acf_raw = acf_data['raw']['acf']
lags = np.arange(len(acf_raw))
ax.stem(lags, acf_raw, basefmt=' ', linefmt='steelblue', markerfmt='o')
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
ax.set_ylabel('ACF', fontsize=10, fontweight='bold')
ax.set_title('Autocorrelation Function (Raw Data)', fontsize=11, fontweight='bold')
ax.set_xlim(-0.5, nlags + 0.5)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, f'ACF(1) = {acf_raw[1]:.3f}\nHighly autocorrelated',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# PACF - Raw Data
ax = axes[0, 1]
pacf_raw = acf_data['raw']['pacf']
# Clip PACF values for better visualization
pacf_raw_clipped = np.clip(pacf_raw, -3, 3)
ax.stem(lags, pacf_raw_clipped, basefmt=' ', linefmt='darkblue', markerfmt='o')
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
ax.set_ylabel('PACF', fontsize=10, fontweight='bold')
ax.set_title('Partial Autocorrelation Function (Raw Data)', fontsize=11, fontweight='bold')
ax.set_xlim(-0.5, nlags + 0.5)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, f'PACF(1) = {pacf_raw[1]:.3f}\nAR(1) pattern',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# ACF - First Differences
ax = axes[1, 0]
acf_diff = acf_data['diff']['acf']
lags_diff = np.arange(len(acf_diff))
confidence_interval_diff = 1.96 / np.sqrt(len(y) - 1)
ax.stem(lags_diff, acf_diff, basefmt=' ', linefmt='green', markerfmt='o')
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=confidence_interval_diff, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-confidence_interval_diff, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
ax.set_ylabel('ACF', fontsize=10, fontweight='bold')
ax.set_title('ACF (First Differences)', fontsize=11, fontweight='bold')
ax.set_xlim(-0.5, nlags + 0.5)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, f'ACF(1) = {acf_diff[1]:.3f}\nNo strong autocorr.',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=9)

# PACF - First Differences
ax = axes[1, 1]
pacf_diff = acf_data['diff']['pacf']
ax.stem(lags_diff, pacf_diff, basefmt=' ', linefmt='darkgreen', markerfmt='o')
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=confidence_interval_diff, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-confidence_interval_diff, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
ax.set_ylabel('PACF', fontsize=10, fontweight='bold')
ax.set_title('PACF (First Differences)', fontsize=11, fontweight='bold')
ax.set_xlim(-0.5, nlags + 0.5)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, f'PACF(1) = {pacf_diff[1]:.3f}\nStationary after diff.',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=9)

plt.suptitle('Autocorrelation Analysis: Raw Data vs First Differences', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/04_acf_pacf_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: 04_acf_pacf_analysis.png")
plt.close()

# ============================================================================
# PLOT 5: Residual Diagnostics for Top Models
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

models_to_plot = ['Linear', 'Quadratic', 'Cubic']
colors_dict = {'Linear': '#1f77b4', 'Quadratic': '#ff7f0e', 'Cubic': '#2ca02c'}

for idx, name in enumerate(models_to_plot):
    residuals = y - models[name]['predictions']

    # Residuals vs Time
    ax = axes[idx, 0]
    ax.scatter(X, residuals, alpha=0.6, s=50, color=colors_dict[name], edgecolors='black', linewidths=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=np.std(residuals), color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=-np.std(residuals), color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Standardized Year', fontsize=9, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=9, fontweight='bold')
    ax.set_title(f'{name}: Residuals vs Time', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Residuals vs Fitted
    ax = axes[idx, 1]
    fitted = models[name]['predictions']
    ax.scatter(fitted, residuals, alpha=0.6, s=50, color=colors_dict[name], edgecolors='black', linewidths=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=np.std(residuals), color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=-np.std(residuals), color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Fitted Values', fontsize=9, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=9, fontweight='bold')
    ax.set_title(f'{name}: Residuals vs Fitted', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Q-Q Plot
    ax = axes[idx, 2]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'{name}: Q-Q Plot', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics: Model Comparison', fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_residual_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: 05_residual_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 6: Residual ACF Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

residual_acf = acf_data['residuals']['acf']
confidence_interval = 1.96 / np.sqrt(len(y))

for idx, name in enumerate(['Linear', 'Quadratic', 'Cubic', 'Exponential']):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    acf = residual_acf[name]
    lags = np.arange(len(acf))

    ax.stem(lags, acf, basefmt=' ', linefmt='steelblue', markerfmt='o')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% CI')
    ax.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Lag', fontsize=10, fontweight='bold')
    ax.set_ylabel('ACF', fontsize=10, fontweight='bold')
    ax.set_title(f'{name} Model: Residual ACF', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, nlags + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Add text annotation
    significant_lags = np.sum(np.abs(acf[1:]) > confidence_interval)
    ax.text(0.98, 0.03, f'ACF(1) = {acf[1]:.3f}\nSig. lags: {significant_lags}',
            transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

plt.suptitle('Residual Autocorrelation: Model Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/06_residual_acf_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 06_residual_acf_comparison.png")
plt.close()

print("\nAll visualizations complete.")
