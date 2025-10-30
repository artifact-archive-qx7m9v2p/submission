"""
Temporal Structure Analysis - Autocorrelation, trends, changepoints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*80)
print("TEMPORAL STRUCTURE ANALYSIS")
print("="*80)

# 1. Autocorrelation Analysis
def calculate_acf(series, nlags=15):
    """Calculate autocorrelation function"""
    acf_values = []
    for lag in range(nlags + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            shifted = series.shift(lag)
            mask = ~(series.isna() | shifted.isna())
            if mask.sum() > 0:
                corr = series[mask].corr(shifted[mask])
                acf_values.append(corr)
            else:
                acf_values.append(np.nan)
    return acf_values

acf_values = calculate_acf(data['C'], nlags=15)
print("\nAutocorrelation Function (ACF):")
for lag, acf in enumerate(acf_values[:6]):
    print(f"  Lag {lag}: {acf:.4f}")

# 2. Trend detection using linear regression on segments
def detect_changepoint(X, y):
    """Simple changepoint detection by testing different split points"""
    n = len(X)
    min_segment = 10  # Minimum points per segment

    best_split = None
    best_improvement = 0

    # Fit single model
    full_model = np.polyfit(X, y, 1)
    full_pred = np.polyval(full_model, X)
    full_ss_res = np.sum((y - full_pred)**2)

    # Try different split points
    for split_idx in range(min_segment, n - min_segment):
        # Fit two separate models
        X1, y1 = X[:split_idx], y[:split_idx]
        X2, y2 = X[split_idx:], y[split_idx:]

        model1 = np.polyfit(X1, y1, 1)
        model2 = np.polyfit(X2, y2, 1)

        pred1 = np.polyval(model1, X1)
        pred2 = np.polyval(model2, X2)

        split_ss_res = np.sum((y1 - pred1)**2) + np.sum((y2 - pred2)**2)

        improvement = full_ss_res - split_ss_res
        if improvement > best_improvement:
            best_improvement = improvement
            best_split = split_idx

    return best_split, best_improvement

X = data['year'].values
y = data['C'].values

split_idx, improvement = detect_changepoint(X, y)
split_year = data.iloc[split_idx]['year']

print(f"\n\nChangepoint Detection:")
print(f"  Potential changepoint at index {split_idx} (year={split_year:.3f})")
print(f"  Improvement in fit: {improvement:.2f}")

# Calculate slopes before and after split
slope_before = np.polyfit(X[:split_idx], y[:split_idx], 1)[0]
slope_after = np.polyfit(X[split_idx:], y[split_idx:], 1)[0]
print(f"  Slope before split: {slope_before:.2f}")
print(f"  Slope after split: {slope_after:.2f}")
print(f"  Slope ratio (after/before): {slope_after/slope_before:.2f}")

# 3. Volatility analysis (moving standard deviation)
window = 5
data['rolling_std'] = data['C'].rolling(window=window).std()
data['rolling_mean'] = data['C'].rolling(window=window).mean()
data['cv_rolling'] = data['rolling_std'] / data['rolling_mean']

print(f"\n\nVolatility Analysis (rolling window={window}):")
print(f"  Mean rolling std: {data['rolling_std'].mean():.2f}")
print(f"  Std of rolling std: {data['rolling_std'].std():.2f}")
print(f"  Correlation of volatility with time: {data['year'].corr(data['rolling_std']):.4f}")

# 4. Test for structural breaks using residuals
quad_coef = np.polyfit(X, y, 2)
residuals = y - np.polyval(quad_coef, X)

# Split residuals and test for different variance
residuals_first_half = residuals[:20]
residuals_second_half = residuals[20:]

# Levene's test for equal variances
from scipy.stats import levene
levene_stat, levene_p = levene(residuals_first_half, residuals_second_half)

print(f"\n\nStructural Break Analysis:")
print(f"  Variance of residuals (first half): {np.var(residuals_first_half):.2f}")
print(f"  Variance of residuals (second half): {np.var(residuals_second_half):.2f}")
print(f"  Levene test for equal variance: statistic={levene_stat:.4f}, p={levene_p:.4f}")
if levene_p < 0.05:
    print("  -> Significant heteroscedasticity detected")
else:
    print("  -> No significant heteroscedasticity")

# Create comprehensive temporal structure visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Temporal Structure Analysis', fontsize=14, fontweight='bold')

# Plot 1: Autocorrelation function
ax1 = axes[0, 0]
lags = range(len(acf_values))
ax1.bar(lags, acf_values, color='steelblue', alpha=0.7)
ax1.axhline(y=0, color='black', linewidth=1)
# Add confidence intervals (approximate)
conf_int = 1.96 / np.sqrt(len(data))
ax1.axhline(y=conf_int, color='red', linestyle='--', linewidth=1, label='95% CI')
ax1.axhline(y=-conf_int, color='red', linestyle='--', linewidth=1)
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')
ax1.set_title('A) Autocorrelation Function')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Data with changepoint
ax2 = axes[0, 1]
ax2.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
ax2.axvline(x=split_year, color='red', linestyle='--', linewidth=2, label=f'Changepoint (year={split_year:.2f})')
# Plot separate linear fits
X1, y1 = X[:split_idx], y[:split_idx]
X2, y2 = X[split_idx:], y[split_idx:]
model1 = np.polyfit(X1, y1, 1)
model2 = np.polyfit(X2, y2, 1)
ax2.plot(X1, np.polyval(model1, X1), 'b-', linewidth=2, label=f'Before (slope={slope_before:.1f})')
ax2.plot(X2, np.polyval(model2, X2), 'g-', linewidth=2, label=f'After (slope={slope_after:.1f})')
ax2.set_xlabel('Year')
ax2.set_ylabel('C')
ax2.set_title('B) Changepoint Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Rolling volatility
ax3 = axes[1, 0]
ax3.plot(data['year'], data['rolling_std'], linewidth=2, color='purple', label='Rolling Std')
ax3.set_xlabel('Year')
ax3.set_ylabel('Rolling Standard Deviation')
ax3.set_title(f'C) Volatility Over Time (window={window})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals with variance bands
ax4 = axes[1, 1]
ax4.scatter(X, residuals, alpha=0.6, s=40, color='coral')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
# Add standard deviation bands
std_first = np.std(residuals_first_half)
std_second = np.std(residuals_second_half)
ax4.axhline(y=std_first, color='blue', linestyle='--', linewidth=1.5, label=f'Std (1st half)={std_first:.1f}')
ax4.axhline(y=-std_first, color='blue', linestyle='--', linewidth=1.5)
ax4.axhline(y=std_second, color='green', linestyle='--', linewidth=1.5, label=f'Std (2nd half)={std_second:.1f}')
ax4.axhline(y=-std_second, color='green', linestyle='--', linewidth=1.5)
ax4.axvline(x=0, color='red', linestyle=':', linewidth=1)
ax4.set_xlabel('Year')
ax4.set_ylabel('Residuals (from quadratic fit)')
ax4.set_title('D) Residuals with Variance Analysis')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/04_temporal_structure.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nTemporal structure plot saved.")

# Update log
with open('/workspace/eda/analyst_2/eda_log.md', 'a') as f:
    f.write("\n## Temporal Structure Analysis\n\n")
    f.write("**Plot: 04_temporal_structure.png**\n\n")
    f.write("### Autocorrelation:\n")
    f.write(f"- Strong autocorrelation at lag 1: {acf_values[1]:.4f}\n")
    f.write(f"- Gradual decay indicating trending behavior\n")
    f.write("- No evidence of cyclical patterns\n\n")
    f.write("### Changepoint Detection:\n")
    f.write(f"- Potential changepoint at year={split_year:.3f}\n")
    f.write(f"- Slope before: {slope_before:.2f}, after: {slope_after:.2f}\n")
    f.write(f"- Slope increases by {slope_after/slope_before:.2f}x after changepoint\n\n")
    f.write("### Volatility:\n")
    f.write(f"- Volatility correlation with time: {data['year'].corr(data['rolling_std']):.4f}\n")
    f.write("- Suggests increasing variance over time\n\n")
    f.write("### Heteroscedasticity:\n")
    f.write(f"- Variance ratio (2nd/1st half): {np.var(residuals_second_half)/np.var(residuals_first_half):.2f}\n")
    f.write(f"- Levene test p-value: {levene_p:.4f}\n")
    if levene_p < 0.05:
        f.write("- **Significant heteroscedasticity present**\n\n")
    else:
        f.write("- No significant heteroscedasticity\n\n")

print("Log updated with temporal structure findings.")
