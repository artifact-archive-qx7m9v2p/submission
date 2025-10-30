"""
Transformation Analysis - Testing different variable transformations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*60)
print("TRANSFORMATION ANALYSIS")
print("="*60)

# Create transformed versions
transformations = {}

# Original
transformations['Original'] = {
    'x': data['x'],
    'y': data['Y'],
    'label_x': 'x',
    'label_y': 'Y'
}

# Log transformations
transformations['Log(x)'] = {
    'x': np.log(data['x']),
    'y': data['Y'],
    'label_x': 'log(x)',
    'label_y': 'Y'
}

transformations['Log(Y)'] = {
    'x': data['x'],
    'y': np.log(data['Y']),
    'label_x': 'x',
    'label_y': 'log(Y)'
}

transformations['Log-Log'] = {
    'x': np.log(data['x']),
    'y': np.log(data['Y']),
    'label_x': 'log(x)',
    'label_y': 'log(Y)'
}

# Square root transformations
transformations['Sqrt(x)'] = {
    'x': np.sqrt(data['x']),
    'y': data['Y'],
    'label_x': 'sqrt(x)',
    'label_y': 'Y'
}

# Reciprocal
transformations['1/x'] = {
    'x': 1 / data['x'],
    'y': data['Y'],
    'label_x': '1/x',
    'label_y': 'Y'
}

# Exponential of x (if values not too large)
# transformations['Exp(-x)'] = {
#     'x': np.exp(-data['x'] / 10),  # Scale down to avoid numerical issues
#     'y': data['Y'],
#     'label_x': 'exp(-x/10)',
#     'label_y': 'Y'
# }

# Calculate metrics for each transformation
results = []

for name, trans in transformations.items():
    x_trans = trans['x'].values
    y_trans = trans['y'].values

    # Remove any inf or nan
    valid_mask = np.isfinite(x_trans) & np.isfinite(y_trans)
    x_trans = x_trans[valid_mask]
    y_trans = y_trans[valid_mask]

    if len(x_trans) > 2:
        # Correlation
        pearson_corr = np.corrcoef(x_trans, y_trans)[0, 1]
        spearman_corr = stats.spearmanr(x_trans, y_trans)[0]

        # Linear fit
        z = np.polyfit(x_trans, y_trans, 1)
        y_pred = z[0] * x_trans + z[1]
        ss_res = np.sum((y_trans - y_pred)**2)
        ss_tot = np.sum((y_trans - y_trans.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((y_trans - y_pred)**2))

        # Linearity test: compare residuals
        residuals = y_trans - y_pred
        _, p_shapiro = stats.shapiro(residuals)

        results.append({
            'Transform': name,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'R²': r2,
            'RMSE': rmse,
            'Residual_p': p_shapiro,
            'n': len(x_trans)
        })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)

print("\nTRANSFORMATION COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)

# Find best transformation
best = results_df.iloc[0]
print(f"\nBest transformation by R²: {best['Transform']}")
print(f"  R² = {best['R²']:.4f}")
print(f"  Pearson correlation = {best['Pearson']:.4f}")

# Compare to original
orig = results_df[results_df['Transform'] == 'Original'].iloc[0]
improvement = (best['R²'] - orig['R²']) / orig['R²'] * 100 if orig['R²'] > 0 else 0
print(f"\nImprovement over original: {improvement:.1f}%")

# Check linearity
linear_transforms = results_df[results_df['Pearson'].abs() > 0.9]
if len(linear_transforms) > 0:
    print(f"\nHighly linear transformations (|r| > 0.9):")
    print(linear_transforms[['Transform', 'Pearson', 'R²']].to_string(index=False))

print("\n" + "="*60)

# Visualize all transformations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Effect of Variable Transformations', fontsize=16, fontweight='bold')

transforms_to_plot = ['Original', 'Log(x)', 'Log(Y)', 'Log-Log', 'Sqrt(x)', '1/x']

for idx, name in enumerate(transforms_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    trans = transformations[name]
    x_trans = trans['x'].values
    y_trans = trans['y'].values

    # Remove inf/nan
    valid_mask = np.isfinite(x_trans) & np.isfinite(y_trans)
    x_trans = x_trans[valid_mask]
    y_trans = y_trans[valid_mask]

    # Scatter plot
    ax.scatter(x_trans, y_trans, alpha=0.6, s=100,
              edgecolors='black', linewidths=0.5, color='steelblue')

    # Fit line
    if len(x_trans) > 2:
        z = np.polyfit(x_trans, y_trans, 1)
        x_line = np.linspace(x_trans.min(), x_trans.max(), 100)
        ax.plot(x_line, z[0]*x_line + z[1], 'r--', linewidth=2, alpha=0.7)

        # Get metrics
        result = results_df[results_df['Transform'] == name].iloc[0]
        textstr = f"R² = {result['R²']:.4f}\nr = {result['Pearson']:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel(trans['label_x'], fontsize=11)
    ax.set_ylabel(trans['label_y'], fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/11_transformations.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved: 11_transformations.png")

# Compare residuals for top transformations
top_3 = results_df.head(3)['Transform'].values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Residual Analysis - Top 3 Transformations', fontsize=14, fontweight='bold')

for idx, name in enumerate(top_3):
    trans = transformations[name]
    x_trans = trans['x'].values
    y_trans = trans['y'].values

    # Remove inf/nan
    valid_mask = np.isfinite(x_trans) & np.isfinite(y_trans)
    x_trans = x_trans[valid_mask]
    y_trans = y_trans[valid_mask]

    # Fit and calculate residuals
    z = np.polyfit(x_trans, y_trans, 1)
    y_pred = z[0] * x_trans + z[1]
    residuals = y_trans - y_pred

    ax = axes[idx]
    ax.scatter(x_trans, residuals, alpha=0.6, s=100,
              edgecolors='black', linewidths=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7)

    # Add std bands
    std_res = np.std(residuals)
    ax.axhline(y=std_res, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=-std_res, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel(trans['label_x'], fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/12_transformation_residuals.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 12_transformation_residuals.png")

print("\nTransformation analysis complete.")
