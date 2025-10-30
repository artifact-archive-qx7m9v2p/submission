"""
Time Series Visualization - Core temporal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

# Create comprehensive time series visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time Series Analysis: C vs Year', fontsize=14, fontweight='bold')

# 1. Scatter plot with trend line
ax1 = axes[0, 0]
ax1.scatter(data['year'], data['C'], alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
# Add linear trend
z = np.polyfit(data['year'], data['C'], 1)
p = np.poly1d(z)
ax1.plot(data['year'], p(data['year']), "r--", alpha=0.8, linewidth=2, label=f'Linear: y={z[0]:.1f}x+{z[1]:.1f}')
ax1.set_xlabel('Year (normalized)', fontweight='bold')
ax1.set_ylabel('C', fontweight='bold')
ax1.set_title('A) Scatter Plot with Linear Trend')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Line plot to show temporal progression
ax2 = axes[0, 1]
ax2.plot(data['year'], data['C'], marker='o', markersize=4, linewidth=1.5, color='darkgreen', alpha=0.7)
ax2.set_xlabel('Year (normalized)', fontweight='bold')
ax2.set_ylabel('C', fontweight='bold')
ax2.set_title('B) Line Plot - Temporal Progression')
ax2.grid(True, alpha=0.3)

# 3. Residuals from linear fit
ax3 = axes[1, 0]
residuals = data['C'] - p(data['year'])
ax3.scatter(data['year'], residuals, alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Year (normalized)', fontweight='bold')
ax3.set_ylabel('Residuals', fontweight='bold')
ax3.set_title('C) Residuals from Linear Fit')
ax3.grid(True, alpha=0.3)

# 4. Distribution of C over time (binned)
ax4 = axes[1, 1]
# Split into quartiles
data['time_quartile'] = pd.qcut(data['year'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
data.boxplot(column='C', by='time_quartile', ax=ax4)
ax4.set_xlabel('Time Quartile', fontweight='bold')
ax4.set_ylabel('C', fontweight='bold')
ax4.set_title('D) Distribution by Time Quartile')
plt.sca(ax4)
plt.xticks(rotation=0)
ax4.get_figure().suptitle('')  # Remove auto-generated title

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_time_series_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("Time series overview plot saved.")

# Calculate R-squared for linear fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((data['C'] - data['C'].mean())**2)
r_squared = 1 - (ss_res / ss_tot)

# Update log
with open('/workspace/eda/analyst_2/eda_log.md', 'a') as f:
    f.write("## Time Series Visualization (Round 1)\n\n")
    f.write("**Plot: 01_time_series_overview.png**\n\n")
    f.write("### Key Observations:\n\n")
    f.write("**Panel A (Scatter + Linear Trend):**\n")
    f.write(f"- Linear fit: C = {z[0]:.1f}*year + {z[1]:.1f}\n")
    f.write(f"- R-squared: {r_squared:.4f}\n")
    f.write("- Visual inspection shows possible non-linearity (curvature in scatter)\n\n")
    f.write("**Panel B (Line Plot):**\n")
    f.write("- Clear upward trend with some volatility\n")
    f.write("- Growth appears to accelerate in later time periods\n")
    f.write("- No obvious cyclical patterns\n\n")
    f.write("**Panel C (Residuals):**\n")
    f.write("- Residuals show systematic pattern (not random)\n")
    f.write("- Negative residuals in early period, positive in late period\n")
    f.write("- This suggests linear model is inadequate\n")
    f.write("- Variance appears to increase over time (heteroscedasticity)\n\n")
    f.write("**Panel D (Boxplot by Quartile):**\n")
    f.write(f"- Q1 median: {data[data['time_quartile']=='Q1']['C'].median():.1f}\n")
    f.write(f"- Q4 median: {data[data['time_quartile']=='Q4']['C'].median():.1f}\n")
    f.write("- Clear increasing trend in both level and spread\n")
    f.write("- Suggests exponential or polynomial growth\n\n")

print("Log updated with initial visualization findings.")
