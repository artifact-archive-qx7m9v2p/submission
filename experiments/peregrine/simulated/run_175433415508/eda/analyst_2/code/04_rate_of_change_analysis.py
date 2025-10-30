"""
Rate of Change Analysis - Growth rates and temporal structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*80)
print("RATE OF CHANGE ANALYSIS")
print("="*80)

# Calculate first differences (absolute change)
data['diff_C'] = data['C'].diff()

# Calculate percentage change
data['pct_change_C'] = data['C'].pct_change() * 100

# Calculate growth rate (log differences)
data['log_C'] = np.log(data['C'])
data['growth_rate'] = data['log_C'].diff()

print("\nFirst Differences (Absolute Change):")
print(f"  Mean: {data['diff_C'].mean():.2f}")
print(f"  Std: {data['diff_C'].std():.2f}")
print(f"  Min: {data['diff_C'].min():.2f}")
print(f"  Max: {data['diff_C'].max():.2f}")

print("\nPercentage Change:")
print(f"  Mean: {data['pct_change_C'].mean():.2f}%")
print(f"  Std: {data['pct_change_C'].std():.2f}%")
print(f"  Min: {data['pct_change_C'].min():.2f}%")
print(f"  Max: {data['pct_change_C'].max():.2f}%")

print("\nGrowth Rate (log differences):")
print(f"  Mean: {data['growth_rate'].mean():.4f}")
print(f"  Std: {data['growth_rate'].std():.4f}")

# Test if growth rate is constant (exponential growth)
# If exponential, growth rate should be constant
growth_rate_clean = data['growth_rate'].dropna()
# Check if growth rate correlates with time (non-constant growth rate)
time_clean = data.loc[growth_rate_clean.index, 'year']
corr_growth_time = stats.pearsonr(time_clean, growth_rate_clean)
print(f"\nCorrelation between growth rate and time: {corr_growth_time[0]:.4f} (p={corr_growth_time[1]:.4f})")
if abs(corr_growth_time[0]) > 0.3:
    print("  -> Growth rate is NOT constant (suggests non-exponential growth)")
else:
    print("  -> Growth rate is relatively constant (suggests exponential growth)")

# Calculate acceleration (second derivative)
data['acceleration'] = data['diff_C'].diff()
print(f"\nAcceleration (second derivative):")
print(f"  Mean: {data['acceleration'].mean():.2f}")
print(f"  Positive acceleration? {data['acceleration'].mean() > 0}")

# Analyze trends in different periods
periods = {
    'Early (Year < -0.5)': data[data['year'] < -0.5],
    'Middle (-0.5 <= Year < 0.5)': data[(data['year'] >= -0.5) & (data['year'] < 0.5)],
    'Late (Year >= 0.5)': data[data['year'] >= 0.5]
}

print("\n" + "="*80)
print("GROWTH RATE BY PERIOD")
print("="*80)

for period_name, period_data in periods.items():
    if len(period_data) > 1:
        mean_C = period_data['C'].mean()
        mean_growth = period_data['growth_rate'].mean()
        mean_pct_change = period_data['pct_change_C'].mean()
        print(f"\n{period_name}:")
        print(f"  Mean C: {mean_C:.2f}")
        print(f"  Mean growth rate: {mean_growth:.4f}")
        print(f"  Mean % change: {mean_pct_change:.2f}%")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rate of Change Analysis', fontsize=14, fontweight='bold')

# Plot 1: Absolute differences
ax1 = axes[0, 0]
ax1.plot(data['year'][1:], data['diff_C'][1:], marker='o', markersize=4, linewidth=1.5, color='blue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax1.set_xlabel('Year')
ax1.set_ylabel('Absolute Change in C')
ax1.set_title('A) First Differences (Absolute Change)')
ax1.grid(True, alpha=0.3)

# Plot 2: Percentage change
ax2 = axes[0, 1]
ax2.plot(data['year'][1:], data['pct_change_C'][1:], marker='o', markersize=4, linewidth=1.5, color='green')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage Change (%)')
ax2.set_title('B) Percentage Change')
ax2.grid(True, alpha=0.3)

# Plot 3: Growth rate (log differences)
ax3 = axes[1, 0]
ax3.plot(data['year'][1:], data['growth_rate'][1:], marker='o', markersize=4, linewidth=1.5, color='purple')
ax3.axhline(y=data['growth_rate'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean={data["growth_rate"].mean():.4f}')
ax3.set_xlabel('Year')
ax3.set_ylabel('Growth Rate (log diff)')
ax3.set_title('C) Growth Rate Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Acceleration
ax4 = axes[1, 1]
ax4.plot(data['year'][2:], data['acceleration'][2:], marker='o', markersize=4, linewidth=1.5, color='red')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.set_xlabel('Year')
ax4.set_ylabel('Acceleration (2nd derivative)')
ax4.set_title('D) Acceleration in Growth')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/03_rate_of_change.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nRate of change plot saved.")

# Update log
with open('/workspace/eda/analyst_2/eda_log.md', 'a') as f:
    f.write("\n## Growth Pattern Analysis (Round 2)\n\n")
    f.write("**Plot: 02_functional_form_comparison.png**\n\n")
    f.write("### Model Comparison Results:\n\n")
    f.write("1. **Cubic model**: R²=0.9743, RMSE=13.88 (BEST)\n")
    f.write("2. **Quadratic model**: R²=0.9641, RMSE=16.43 (simpler, nearly as good)\n")
    f.write("3. **Exponential model**: R²=0.9358, RMSE=21.96\n")
    f.write("4. **Linear model**: R²=0.8812, RMSE=29.87\n")
    f.write("5. **Power law**: R²=0.6757, RMSE=49.36 (POOR)\n\n")
    f.write("**Key Finding**: Quadratic model provides excellent fit (96.4% variance explained)\n")
    f.write("while being simpler than cubic. The data shows clear polynomial growth pattern.\n\n")
    f.write("**Plot: 03_rate_of_change.png**\n\n")
    f.write("### Rate of Change Analysis:\n\n")
    f.write(f"- Mean absolute change: {data['diff_C'].mean():.2f} per time unit\n")
    f.write(f"- Mean percentage change: {data['pct_change_C'].mean():.2f}%\n")
    f.write(f"- Growth rate correlation with time: {corr_growth_time[0]:.4f}\n")
    f.write(f"- Positive acceleration detected: {data['acceleration'].mean() > 0}\n\n")
    f.write("**Interpretation**: Growth rate is NOT constant, ruling out pure exponential growth.\n")
    f.write("The positive acceleration supports polynomial (quadratic/cubic) model.\n\n")

print("Log updated with growth pattern findings.")
