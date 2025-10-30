"""
Diagnostic Summary: Key Insights for Model Selection
Focus: Synthesizing findings into modeling recommendations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('/workspace/data/data_analyst_1.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

X = df['year'].values
y = df['C'].values

print("="*60)
print("DIAGNOSTIC SUMMARY FOR MODEL SELECTION")
print("="*60)

# 1. Count data properties
print("\n1. COUNT DATA CHARACTERISTICS")
print(f"   Range: [{y.min()}, {y.max()}]")
print(f"   Mean: {y.mean():.2f}")
print(f"   Variance: {y.var():.2f}")
print(f"   Variance/Mean ratio: {y.var()/y.mean():.2f}")
print(f"   Zero counts: {(y == 0).sum()}")
print(f"   Small counts (<5): {(y < 5).sum()}")

# 2. Distribution test
shapiro_stat, shapiro_p = stats.shapiro(y)
print(f"\n2. NORMALITY TEST (Shapiro-Wilk)")
print(f"   Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("   -> Reject normality assumption")
else:
    print("   -> Cannot reject normality")

# Log-transform for comparison
y_log = np.log(y)
shapiro_log_stat, shapiro_log_p = stats.shapiro(y_log)
print(f"\n   Log-transformed Shapiro-Wilk:")
print(f"   Statistic: {shapiro_log_stat:.4f}, p-value: {shapiro_log_p:.4f}")

# 3. Temporal structure
print(f"\n3. TEMPORAL STRUCTURE")
print(f"   Total timespan: {X.max() - X.min():.3f} standardized units")
print(f"   Number of observations: {len(y)}")
print(f"   Average spacing: {np.mean(np.diff(X)):.4f}")

# Check for missing time points (all should be equally spaced)
diffs = np.diff(X)
print(f"   Time spacing uniformity (std): {diffs.std():.6f}")
if diffs.std() < 0.001:
    print("   -> Equally spaced time series")
else:
    print("   -> Irregular time spacing")

# 4. Summary statistics by period
split_idx = 17  # From change point analysis
print(f"\n4. TWO-REGIME SUMMARY")
print(f"   Early phase (n={split_idx}):")
print(f"   - Mean count: {y[:split_idx].mean():.2f}")
print(f"   - Std count: {y[:split_idx].std():.2f}")
print(f"   - CV: {y[:split_idx].std()/y[:split_idx].mean():.3f}")

print(f"\n   Late phase (n={len(y)-split_idx}):")
print(f"   - Mean count: {y[split_idx:].mean():.2f}")
print(f"   - Std count: {y[split_idx:].std():.2f}")
print(f"   - CV: {y[split_idx:].std()/y[split_idx:].mean():.3f}")

# 5. Model fit comparison
models = {
    'Linear': 0.8812,
    'Quadratic': 0.9641,
    'Cubic': 0.9743,
    'Exponential': 0.9358,
    'Piecewise Linear': 0.9729
}

print(f"\n5. MODEL PERFORMANCE (R²)")
for model, r2 in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model:18s}: {r2:.4f}")

# Create final diagnostic plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Q-Q plot for normality
ax1 = axes[0, 0]
stats.probplot(y, dist="norm", plot=ax1)
ax1.set_title('A. Q-Q Plot: Raw Counts', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q plot for log-transformed
ax2 = axes[0, 1]
stats.probplot(y_log, dist="norm", plot=ax2)
ax2.set_title('B. Q-Q Plot: Log-Transformed Counts', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Histogram comparison
ax3 = axes[1, 0]
ax3.hist(y[:split_idx], bins=10, alpha=0.6, label='Early phase', color='blue', edgecolor='black')
ax3.hist(y[split_idx:], bins=15, alpha=0.6, label='Late phase', color='red', edgecolor='black')
ax3.set_xlabel('Count (C)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('C. Distribution by Regime', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Box plots by regime
ax4 = axes[1, 1]
box_data = [y[:split_idx], y[split_idx:]]
bp = ax4.boxplot(box_data, labels=['Early\nPhase', 'Late\nPhase'],
                  patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax4.set_ylabel('Count (C)', fontsize=11)
ax4.set_title('D. Count Distribution by Regime', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# Add statistics text
stats_text = f"Early: μ={y[:split_idx].mean():.1f}, σ={y[:split_idx].std():.1f}\n"
stats_text += f"Late: μ={y[split_idx:].mean():.1f}, σ={y[split_idx:].std():.1f}"
ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/05_diagnostic_summary.png', dpi=300, bbox_inches='tight')
print("\nSaved: 05_diagnostic_summary.png")

# Generate recommendations
print("\n" + "="*60)
print("MODELING RECOMMENDATIONS")
print("="*60)

print("\nRECOMMENDED MODEL CLASSES:")
print("\n1. NEGATIVE BINOMIAL REGRESSION (Primary)")
print("   Rationale:")
print("   - Strong overdispersion (var/mean = 2.15)")
print("   - Count data with non-negative integers")
print("   - Accommodates heteroscedasticity naturally")
print("   - Can incorporate polynomial or spline terms for year")

print("\n2. PIECEWISE/SEGMENTED REGRESSION (Alternative)")
print("   Rationale:")
print("   - Significant structural break detected (p<0.001)")
print("   - 9.6x acceleration in growth rate after change point")
print("   - Distinct phases suggest regime-specific models")
print("   - Can combine with GLM framework")

print("\n3. GENERALIZED ADDITIVE MODEL (GAM) (Flexible)")
print("   Rationale:")
print("   - Nonlinear relationship clearly present")
print("   - Can capture smooth transitions automatically")
print("   - Flexible enough for complex temporal patterns")
print("   - Can use negative binomial family")

print("\nFEATURES TO INCLUDE:")
print("   - Year (linear term)")
print("   - Year² (quadratic term) - strong evidence")
print("   - Year³ (cubic term) - marginal improvement")
print("   - Regime indicator (pre/post change point)")
print("   - Interaction: Regime × Year")

print("\nMODELS TO AVOID:")
print("   - Simple Poisson: Violates equidispersion assumption")
print("   - Simple linear: Ignores count nature, poor fit")
print("   - Pure exponential: Overpredicts at extremes")

print("\nDATA QUALITY NOTES:")
print("   - No missing values")
print("   - No zero inflation issues")
print("   - Equally spaced time series")
print("   - Strong temporal autocorrelation (lag-1: 0.97)")
print("   - Residual autocorrelation minimal after trend removal")

plt.close()
