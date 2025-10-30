"""
Script 4: Testing Competing Hypotheses
Tests different functional forms and model structures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Manual metric implementations
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Set up paths
DATA_PATH = Path("/workspace/data/data.csv")
VIZ_DIR = Path("/workspace/eda/visualizations")

# Load data
df = pd.read_csv(DATA_PATH)

print("="*80)
print("TESTING COMPETING HYPOTHESES")
print("="*80)

# ============================================================================
# HYPOTHESIS 1: LINEAR RELATIONSHIP
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 1: Y = a + b*x (Linear)")
print("="*80)

# Fit linear model
z_linear = np.polyfit(df['x'], df['Y'], 1)
y_pred_linear = np.polyval(z_linear, df['x'])

# Metrics
r2_linear = r2_score(df['Y'], y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(df['Y'], y_pred_linear))
mae_linear = mean_absolute_error(df['Y'], y_pred_linear)

print(f"Model: Y = {z_linear[1]:.4f} + {z_linear[0]:.4f}*x")
print(f"R-squared: {r2_linear:.6f}")
print(f"RMSE: {rmse_linear:.6f}")
print(f"MAE: {mae_linear:.6f}")

# ============================================================================
# HYPOTHESIS 2: LOGARITHMIC RELATIONSHIP
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 2: Y = a + b*ln(x) (Logarithmic)")
print("="*80)

# Fit log model
log_x = np.log(df['x'])
z_log = np.polyfit(log_x, df['Y'], 1)
y_pred_log = z_log[1] + z_log[0] * np.log(df['x'])

# Metrics
r2_log = r2_score(df['Y'], y_pred_log)
rmse_log = np.sqrt(mean_squared_error(df['Y'], y_pred_log))
mae_log = mean_absolute_error(df['Y'], y_pred_log)

print(f"Model: Y = {z_log[1]:.4f} + {z_log[0]:.4f}*ln(x)")
print(f"R-squared: {r2_log:.6f}")
print(f"RMSE: {rmse_log:.6f}")
print(f"MAE: {mae_log:.6f}")

# ============================================================================
# HYPOTHESIS 3: ASYMPTOTIC/SATURATION RELATIONSHIP
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 3: Y = a - b/x (Asymptotic/Saturation)")
print("="*80)

# Fit inverse model: Y = a - b/x
inv_x = 1 / df['x']
z_inv = np.polyfit(inv_x, df['Y'], 1)
y_pred_inv = z_inv[1] + z_inv[0] / df['x']

# Metrics
r2_inv = r2_score(df['Y'], y_pred_inv)
rmse_inv = np.sqrt(mean_squared_error(df['Y'], y_pred_inv))
mae_inv = mean_absolute_error(df['Y'], y_pred_inv)

print(f"Model: Y = {z_inv[1]:.4f} + {z_inv[0]:.4f}/x")
print(f"R-squared: {r2_inv:.6f}")
print(f"RMSE: {rmse_inv:.6f}")
print(f"MAE: {mae_inv:.6f}")

# ============================================================================
# HYPOTHESIS 4: QUADRATIC RELATIONSHIP
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 4: Y = a + b*x + c*x^2 (Quadratic)")
print("="*80)

# Fit quadratic model
z_quad = np.polyfit(df['x'], df['Y'], 2)
y_pred_quad = np.polyval(z_quad, df['x'])

# Metrics
r2_quad = r2_score(df['Y'], y_pred_quad)
rmse_quad = np.sqrt(mean_squared_error(df['Y'], y_pred_quad))
mae_quad = mean_absolute_error(df['Y'], y_pred_quad)

print(f"Model: Y = {z_quad[2]:.4f} + {z_quad[1]:.4f}*x + {z_quad[0]:.6f}*x^2")
print(f"R-squared: {r2_quad:.6f}")
print(f"RMSE: {rmse_quad:.6f}")
print(f"MAE: {mae_quad:.6f}")

# ============================================================================
# HYPOTHESIS 5: SQUARE ROOT RELATIONSHIP
# ============================================================================

print("\n" + "="*80)
print("HYPOTHESIS 5: Y = a + b*sqrt(x) (Square Root)")
print("="*80)

# Fit square root model
sqrt_x = np.sqrt(df['x'])
z_sqrt = np.polyfit(sqrt_x, df['Y'], 1)
y_pred_sqrt = z_sqrt[1] + z_sqrt[0] * np.sqrt(df['x'])

# Metrics
r2_sqrt = r2_score(df['Y'], y_pred_sqrt)
rmse_sqrt = np.sqrt(mean_squared_error(df['Y'], y_pred_sqrt))
mae_sqrt = mean_absolute_error(df['Y'], y_pred_sqrt)

print(f"Model: Y = {z_sqrt[1]:.4f} + {z_sqrt[0]:.4f}*sqrt(x)")
print(f"R-squared: {r2_sqrt:.6f}")
print(f"RMSE: {rmse_sqrt:.6f}")
print(f"MAE: {mae_sqrt:.6f}")

# ============================================================================
# COMPARISON OF ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

models = {
    'Linear': {'r2': r2_linear, 'rmse': rmse_linear, 'mae': mae_linear, 'pred': y_pred_linear},
    'Logarithmic': {'r2': r2_log, 'rmse': rmse_log, 'mae': mae_log, 'pred': y_pred_log},
    'Asymptotic': {'r2': r2_inv, 'rmse': rmse_inv, 'mae': mae_inv, 'pred': y_pred_inv},
    'Quadratic': {'r2': r2_quad, 'rmse': rmse_quad, 'mae': mae_quad, 'pred': y_pred_quad},
    'Square Root': {'r2': r2_sqrt, 'rmse': rmse_sqrt, 'mae': mae_sqrt, 'pred': y_pred_sqrt}
}

# Create comparison table
print(f"\n{'Model':<15} {'R-squared':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 51)
for name, metrics in models.items():
    print(f"{name:<15} {metrics['r2']:<12.6f} {metrics['rmse']:<12.6f} {metrics['mae']:<12.6f}")

# Rank models
print("\nModel Rankings:")
print("\nBy R-squared (higher is better):")
sorted_r2 = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)
for i, (name, metrics) in enumerate(sorted_r2, 1):
    print(f"  {i}. {name}: {metrics['r2']:.6f}")

print("\nBy RMSE (lower is better):")
sorted_rmse = sorted(models.items(), key=lambda x: x[1]['rmse'])
for i, (name, metrics) in enumerate(sorted_rmse, 1):
    print(f"  {i}. {name}: {metrics['rmse']:.6f}")

# ============================================================================
# VISUALIZATION: ALL MODELS COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

x_line = np.linspace(df['x'].min(), df['x'].max(), 200)

# Plot each model
plot_configs = [
    ('Linear', z_linear, lambda x: np.polyval(z_linear, x), 'red'),
    ('Logarithmic', z_log, lambda x: z_log[1] + z_log[0] * np.log(x), 'green'),
    ('Asymptotic', z_inv, lambda x: z_inv[1] + z_inv[0] / x, 'purple'),
    ('Quadratic', z_quad, lambda x: np.polyval(z_quad, x), 'orange'),
    ('Square Root', z_sqrt, lambda x: z_sqrt[1] + z_sqrt[0] * np.sqrt(x), 'brown')
]

for i, (name, coeffs, func, color) in enumerate(plot_configs):
    axes[i].scatter(df['x'], df['Y'], alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5, label='Data', zorder=3)
    y_line = func(x_line)
    axes[i].plot(x_line, y_line, color=color, linewidth=2.5, label=f'{name} Fit', zorder=2)
    axes[i].set_xlabel('x (Predictor)', fontsize=11)
    axes[i].set_ylabel('Y (Response)', fontsize=11)
    axes[i].set_title(f'{name} Model\nR²={models[name]["r2"]:.4f}, RMSE={models[name]["rmse"]:.4f}',
                     fontsize=11, fontweight='bold')
    axes[i].legend(loc='best')
    axes[i].grid(True, alpha=0.3)

# All models overlay
axes[5].scatter(df['x'], df['Y'], alpha=0.7, s=70, color='steelblue', edgecolors='black',
               linewidth=0.8, label='Data', zorder=10)
colors = ['red', 'green', 'purple', 'orange', 'brown']
for (name, _, func, _), color in zip(plot_configs, colors):
    y_line = func(x_line)
    axes[5].plot(x_line, y_line, color=color, linewidth=2, label=name, alpha=0.7)
axes[5].set_xlabel('x (Predictor)', fontsize=11)
axes[5].set_ylabel('Y (Response)', fontsize=11)
axes[5].set_title('All Models Comparison', fontsize=12, fontweight='bold')
axes[5].legend(loc='best', fontsize=9)
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'hypothesis_all_models_comparison.png', bbox_inches='tight')
plt.close()

print("\nVisualization saved: hypothesis_all_models_comparison.png")

# ============================================================================
# RESIDUAL COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

residual_data = [
    ('Linear', df['Y'] - y_pred_linear, 'red'),
    ('Logarithmic', df['Y'] - y_pred_log, 'green'),
    ('Asymptotic', df['Y'] - y_pred_inv, 'purple'),
    ('Quadratic', df['Y'] - y_pred_quad, 'orange'),
    ('Square Root', df['Y'] - y_pred_sqrt, 'brown')
]

for i, (name, residuals, color) in enumerate(residual_data):
    axes[i].scatter(df['x'], residuals, alpha=0.7, s=60, color=color, edgecolors='black', linewidth=0.5)
    axes[i].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[i].set_xlabel('x (Predictor)', fontsize=11)
    axes[i].set_ylabel('Residuals', fontsize=11)
    axes[i].set_title(f'{name} Model Residuals\nStd={residuals.std():.4f}',
                     fontsize=11, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

# Residual distributions comparison
axes[5].hist([data[1] for data in residual_data], label=[data[0] for data in residual_data],
            alpha=0.5, bins=8, edgecolor='black')
axes[5].set_xlabel('Residuals', fontsize=11)
axes[5].set_ylabel('Frequency', fontsize=11)
axes[5].set_title('Residual Distributions Comparison', fontsize=12, fontweight='bold')
axes[5].legend(loc='best', fontsize=9)
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'hypothesis_residuals_comparison.png', bbox_inches='tight')
plt.close()

print("Residual comparison saved: hypothesis_residuals_comparison.png")

print("\n" + "="*80)
print("HYPOTHESIS TESTING COMPLETE")
print("="*80)
