"""
Basic Visualizations - Understanding the data structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

# Create a comprehensive overview plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Initial Data Overview - Analyst 2', fontsize=16, fontweight='bold')

# 1. Scatter plot with basic trend
ax1 = axes[0, 0]
ax1.scatter(data['x'], data['Y'], alpha=0.6, s=80, edgecolors='black', linewidths=0.5)
# Add simple linear fit
z = np.polyfit(data['x'], data['Y'], 1)
p = np.poly1d(z)
x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
ax1.plot(x_line, p(x_line), "r--", alpha=0.7, label=f'Linear fit: R²={np.corrcoef(data["x"], data["Y"])[0,1]**2:.3f}')
ax1.set_xlabel('x')
ax1.set_ylabel('Y')
ax1.set_title('Scatter Plot with Linear Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Distribution of x
ax2 = axes[0, 1]
ax2.hist(data['x'], bins=15, alpha=0.7, edgecolor='black', color='steelblue')
ax2.axvline(data['x'].mean(), color='red', linestyle='--', label=f'Mean: {data["x"].mean():.2f}')
ax2.axvline(data['x'].median(), color='green', linestyle='--', label=f'Median: {data["x"].median():.2f}')
ax2.set_xlabel('x')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of x')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Distribution of Y
ax3 = axes[0, 2]
ax3.hist(data['Y'], bins=12, alpha=0.7, edgecolor='black', color='coral')
ax3.axvline(data['Y'].mean(), color='red', linestyle='--', label=f'Mean: {data["Y"].mean():.2f}')
ax3.axvline(data['Y'].median(), color='green', linestyle='--', label=f'Median: {data["Y"].median():.2f}')
ax3.set_xlabel('Y')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Y')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Residuals from linear fit
ax4 = axes[1, 0]
linear_pred = p(data['x'])
residuals = data['Y'] - linear_pred
ax4.scatter(data['x'], residuals, alpha=0.6, s=80, edgecolors='black', linewidths=0.5)
ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax4.set_xlabel('x')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals from Linear Fit')
ax4.grid(True, alpha=0.3)

# 5. Q-Q plot for Y
ax5 = axes[1, 1]
stats.probplot(data['Y'], dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot for Y')
ax5.grid(True, alpha=0.3)

# 6. Box plots
ax6 = axes[1, 2]
box_data = [data['x'], data['Y']]
bp = ax6.boxplot(box_data, labels=['x', 'Y'], patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax6.set_ylabel('Value')
ax6.set_title('Box Plots (standardized)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_data_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 01_data_overview.png")

# Create a detailed residual analysis plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Linear Model Residual Analysis', fontsize=14, fontweight='bold')

# Residuals vs fitted
axes[0].scatter(linear_pred, residuals, alpha=0.6, s=80, edgecolors='black', linewidths=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')
axes[0].grid(True, alpha=0.3)

# Histogram of residuals
axes[1].hist(residuals, bins=10, alpha=0.7, edgecolor='black', color='lightblue')
axes[1].axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.4f}')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Q-Q plot of residuals
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot of Residuals')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/02_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 02_residual_analysis.png")

# Analyze variability at repeated x values
repeated_x = data['x'].value_counts()
repeated_x = repeated_x[repeated_x > 1].index

if len(repeated_x) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, x_val in enumerate(sorted(repeated_x)):
        subset = data[data['x'] == x_val]
        y_values = subset['Y'].values
        ax.scatter([x_val] * len(y_values), y_values, s=100, alpha=0.6,
                  edgecolors='black', linewidths=1, label=f'x={x_val}')
        # Add range bars
        if len(y_values) > 1:
            ax.plot([x_val, x_val], [y_values.min(), y_values.max()],
                   'k-', alpha=0.3, linewidth=2)

    # Add all data points for context
    ax.scatter(data['x'], data['Y'], alpha=0.2, s=50, color='gray', label='All data')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Variability at Repeated x Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('/workspace/eda/analyst_2/visualizations/03_repeated_x_variability.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: 03_repeated_x_variability.png")

    # Calculate variance at each repeated x
    print("\nVariability at repeated x values:")
    for x_val in sorted(repeated_x):
        subset = data[data['x'] == x_val]['Y']
        print(f"  x={x_val:5.1f}: Y range=[{subset.min():.4f}, {subset.max():.4f}], "
              f"std={subset.std():.4f}, n={len(subset)}")

print("\nBasic visualization phase complete.")
