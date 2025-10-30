"""
Visualize all functional form fits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

# Load model fits
model_data = np.load('/workspace/eda/analyst_2/code/model_fits.npz', allow_pickle=True)
x = model_data['x']
y = model_data['y']
x_smooth = model_data['x_smooth']

# Extract models
model_names = ['Linear', 'Quadratic', 'Cubic', 'Logarithmic',
               'Square Root', 'Power Law', 'Asymptotic']
models = {}
for name in model_names:
    if name in model_data:
        models[name] = model_data[name].item()

# Create comprehensive comparison plot
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Functional Form Exploration - All Candidate Models',
             fontsize=16, fontweight='bold')

# Plot each model
for idx, (name, metrics) in enumerate(models.items()):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black',
               linewidths=0.5, label='Data', zorder=3)

    # Model fit
    ax.plot(x_smooth, metrics['smooth'], 'r-', linewidth=2.5,
            label=f'{name} fit', zorder=2)

    # Add metrics text
    textstr = f"R² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}\nAIC = {metrics['aic']:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.savefig('/workspace/eda/analyst_2/visualizations/04_all_functional_forms.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 04_all_functional_forms.png")

# Create comparison of top 3 models
top_models = sorted(models.items(), key=lambda x: x[1]['aic'], reverse=True)[:3]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Top 3 Models by AIC', fontsize=14, fontweight='bold')

for idx, (name, metrics) in enumerate(top_models):
    ax = axes[idx]

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=100, edgecolors='black',
               linewidths=0.5, label='Data', zorder=3)

    # Model fit
    ax.plot(x_smooth, metrics['smooth'], 'r-', linewidth=3,
            label=f'{name} fit', zorder=2)

    # Residuals as vertical lines
    residuals = y - metrics['pred']
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], metrics['pred'][i]],
                'gray', alpha=0.3, linewidth=1)

    # Add metrics
    textstr = (f"R² = {metrics['r2']:.4f}\n"
               f"RMSE = {metrics['rmse']:.4f}\n"
               f"AIC = {metrics['aic']:.2f}\n"
               f"Params = {metrics['n_params']}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title(f'{name} Model', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/05_top_models_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 05_top_models_comparison.png")

# Create residual comparison for top models
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Residual Analysis - Top 3 Models', fontsize=14, fontweight='bold')

for idx, (name, metrics) in enumerate(top_models):
    residuals = y - metrics['pred']

    # Residuals vs fitted
    ax1 = axes[0, idx]
    ax1.scatter(metrics['pred'], residuals, alpha=0.6, s=80,
                edgecolors='black', linewidths=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{name}: Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)

    # Residuals vs x
    ax2 = axes[1, idx]
    ax2.scatter(x, residuals, alpha=0.6, s=80,
                edgecolors='black', linewidths=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{name}: Residuals vs x')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/06_residual_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 06_residual_comparison.png")

print("\nFunctional form visualization complete.")
