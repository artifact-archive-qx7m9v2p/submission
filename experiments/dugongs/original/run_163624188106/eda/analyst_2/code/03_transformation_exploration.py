"""
Data Transformation Exploration
Focus: Explore log, sqrt, and other transformations of Y and x
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("TRANSFORMATION EXPLORATION")
print("=" * 80)

X = data['x'].values
y = data['Y'].values

# Define transformation functions
transformations = {
    'none': lambda z: z,
    'log': lambda z: np.log(z),
    'sqrt': lambda z: np.sqrt(z),
    'reciprocal': lambda z: 1/z,
    'square': lambda z: z**2,
    'log1p': lambda z: np.log1p(z)  # log(1+z), useful when values near 0
}

# Test different combinations
results = []

print("\nTesting transformation combinations...")
print("-" * 80)

for y_trans_name, y_trans in transformations.items():
    for x_trans_name, x_trans in transformations.items():
        try:
            # Apply transformations
            y_transformed = y_trans(y)
            x_transformed = x_trans(X)

            # Skip if any infinite or nan values
            if not (np.isfinite(y_transformed).all() and np.isfinite(x_transformed).all()):
                continue

            # Fit linear model
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_transformed, y_transformed)

            # Get predictions and residuals
            y_pred = intercept + slope * x_transformed
            residuals = y_transformed - y_pred

            # Calculate metrics
            r2 = r_value**2
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))

            # Normality test on residuals
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # Homoscedasticity check (correlation between |residuals| and fitted)
            hetero_corr = np.abs(stats.pearsonr(y_pred, np.abs(residuals))[0])

            results.append({
                'y_transform': y_trans_name,
                'x_transform': x_trans_name,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'shapiro_p': shapiro_p,
                'hetero_corr': hetero_corr,
                'slope': slope,
                'intercept': intercept
            })

        except Exception as e:
            pass

# Convert to DataFrame and sort by R2
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r2', ascending=False)

print("\nTop 15 transformation combinations (by R²):")
print(results_df.head(15).to_string(index=False))

print("\n\nTop 10 by normality of residuals (Shapiro-Wilk p-value):")
print(results_df.nlargest(10, 'shapiro_p')[['y_transform', 'x_transform', 'r2', 'shapiro_p', 'hetero_corr']].to_string(index=False))

print("\n\nTop 10 by homoscedasticity (low correlation):")
print(results_df.nsmallest(10, 'hetero_corr')[['y_transform', 'x_transform', 'r2', 'shapiro_p', 'hetero_corr']].to_string(index=False))

# Save results
results_df.to_csv('/workspace/eda/analyst_2/code/transformation_results.csv', index=False)
print("\n\nSaved all results to: /workspace/eda/analyst_2/code/transformation_results.csv")

# Visualize top transformations
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

# Get top 12 transformations
top_12 = results_df.head(12)

for idx, (_, row) in enumerate(top_12.iterrows()):
    y_trans = transformations[row['y_transform']]
    x_trans = transformations[row['x_transform']]

    y_transformed = y_trans(y)
    x_transformed = x_trans(X)

    # Fit model
    slope_fit, intercept_fit = row['slope'], row['intercept']
    y_pred = intercept_fit + slope_fit * x_transformed

    # Plot
    axes[idx].scatter(x_transformed, y_transformed, alpha=0.6, s=40)

    # Sort for line plot
    sort_idx = np.argsort(x_transformed)
    axes[idx].plot(x_transformed[sort_idx], y_pred[sort_idx], 'r-', linewidth=2)

    # Labels
    y_label = f"{row['y_transform']}(Y)" if row['y_transform'] != 'none' else 'Y'
    x_label = f"{row['x_transform']}(x)" if row['x_transform'] != 'none' else 'x'

    axes[idx].set_xlabel(x_label)
    axes[idx].set_ylabel(y_label)
    axes[idx].set_title(f"R²={row['r2']:.4f}, Shapiro-p={row['shapiro_p']:.3f}")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/03_transformation_fits.png', dpi=300, bbox_inches='tight')
print("\nSaved: /workspace/eda/analyst_2/visualizations/03_transformation_fits.png")
plt.close()

# Detailed residual diagnostics for top 4 transformations
print("\n" + "=" * 80)
print("DETAILED ANALYSIS OF TOP 4 TRANSFORMATIONS")
print("=" * 80)

fig, axes = plt.subplots(4, 4, figsize=(20, 16))

top_4 = results_df.head(4)

for row_idx, (_, row) in enumerate(top_4.iterrows()):
    y_trans = transformations[row['y_transform']]
    x_trans = transformations[row['x_transform']]

    y_transformed = y_trans(y)
    x_transformed = x_trans(X)

    # Fit model
    slope_fit, intercept_fit = row['slope'], row['intercept']
    y_pred = intercept_fit + slope_fit * x_transformed
    residuals = y_transformed - y_pred
    std_resid = residuals / np.std(residuals, ddof=2)

    # Column 1: Fitted vs Actual
    axes[row_idx, 0].scatter(x_transformed, y_transformed, alpha=0.6, s=40, label='Actual')
    sort_idx = np.argsort(x_transformed)
    axes[row_idx, 0].plot(x_transformed[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='Fitted')

    y_label = f"{row['y_transform']}(Y)" if row['y_transform'] != 'none' else 'Y'
    x_label = f"{row['x_transform']}(x)" if row['x_transform'] != 'none' else 'x'

    axes[row_idx, 0].set_xlabel(x_label)
    axes[row_idx, 0].set_ylabel(y_label)
    axes[row_idx, 0].set_title(f"Fit: {y_label} vs {x_label}\nR²={row['r2']:.4f}")
    axes[row_idx, 0].legend()
    axes[row_idx, 0].grid(True, alpha=0.3)

    # Column 2: Residuals vs Fitted
    axes[row_idx, 1].scatter(y_pred, residuals, alpha=0.6, s=40)
    axes[row_idx, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[row_idx, 1].set_xlabel('Fitted Values')
    axes[row_idx, 1].set_ylabel('Residuals')
    axes[row_idx, 1].set_title('Residuals vs Fitted')
    axes[row_idx, 1].grid(True, alpha=0.3)

    # Column 3: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[row_idx, 2])
    axes[row_idx, 2].set_title(f'Q-Q Plot\nShapiro p={row["shapiro_p"]:.3f}')

    # Column 4: Residual histogram
    axes[row_idx, 3].hist(residuals, bins=10, edgecolor='black', alpha=0.7, density=True)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    axes[row_idx, 3].plot(x_range, stats.norm.pdf(x_range, 0, np.std(residuals, ddof=2)),
                          'r-', linewidth=2)
    axes[row_idx, 3].set_xlabel('Residuals')
    axes[row_idx, 3].set_ylabel('Density')
    axes[row_idx, 3].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/04_top_transformations_diagnostics.png', dpi=300, bbox_inches='tight')
print("\nSaved: /workspace/eda/analyst_2/visualizations/04_top_transformations_diagnostics.png")
plt.close()

print("\n" + "=" * 80)
print("TRANSFORMATION EXPLORATION COMPLETE")
print("=" * 80)
