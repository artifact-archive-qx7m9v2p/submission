"""
Segmentation and Change Point Analysis
Testing for regime shifts and natural groupings in x
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
x = data['x'].values
y = data['Y'].values

print("="*60)
print("SEGMENTATION AND CHANGE POINT ANALYSIS")
print("="*60)

# Test different segmentation strategies
# 1. Quantile-based segmentation
quantiles = [0.33, 0.67]
q_values = data['x'].quantile(quantiles).values

segments_quantile = {
    'Low': data[data['x'] <= q_values[0]],
    'Medium': data[(data['x'] > q_values[0]) & (data['x'] <= q_values[1])],
    'High': data[data['x'] > q_values[1]]
}

print("\n1. QUANTILE-BASED SEGMENTATION")
print(f"Breakpoints at x = {q_values}")
for name, segment in segments_quantile.items():
    print(f"\n{name} segment:")
    print(f"  x range: [{segment['x'].min():.2f}, {segment['x'].max():.2f}]")
    print(f"  Y mean: {segment['Y'].mean():.4f} ± {segment['Y'].std():.4f}")
    print(f"  n = {len(segment)}")
    corr = segment['x'].corr(segment['Y'])
    print(f"  Correlation: {corr:.4f}")

# 2. Equal-width bins
n_bins = 3
x_range = data['x'].max() - data['x'].min()
bin_width = x_range / n_bins
bins = [data['x'].min() + i * bin_width for i in range(n_bins + 1)]

segments_equal = {}
for i in range(n_bins):
    if i < n_bins - 1:
        mask = (data['x'] >= bins[i]) & (data['x'] < bins[i+1])
    else:
        mask = (data['x'] >= bins[i]) & (data['x'] <= bins[i+1])
    segments_equal[f'Bin_{i+1}'] = data[mask]

print("\n2. EQUAL-WIDTH BINS")
print(f"Bin edges: {[f'{b:.2f}' for b in bins]}")
for name, segment in segments_equal.items():
    if len(segment) > 0:
        print(f"\n{name}:")
        print(f"  x range: [{segment['x'].min():.2f}, {segment['x'].max():.2f}]")
        print(f"  Y mean: {segment['Y'].mean():.4f} ± {segment['Y'].std():.4f}")
        print(f"  n = {len(segment)}")
        if len(segment) > 2:
            corr = segment['x'].corr(segment['Y'])
            print(f"  Correlation: {corr:.4f}")

# 3. Natural breakpoints based on x distribution
# Look for gaps in x
x_sorted = np.sort(data['x'].unique())
gaps = np.diff(x_sorted)
large_gaps_idx = np.where(gaps > np.percentile(gaps, 75))[0]

print("\n3. NATURAL BREAKPOINTS (based on x gaps)")
print(f"Large gaps in x found at indices: {large_gaps_idx}")
if len(large_gaps_idx) > 0:
    print("Gap locations:")
    for idx in large_gaps_idx:
        print(f"  Between x={x_sorted[idx]:.2f} and x={x_sorted[idx+1]:.2f} (gap={gaps[idx]:.2f})")

# 4. Test for change point using sliding window correlation
window_size = 7
if len(data) >= window_size:
    correlations = []
    window_centers = []

    data_sorted = data.sort_values('x')
    for i in range(len(data_sorted) - window_size + 1):
        window = data_sorted.iloc[i:i+window_size]
        if window['x'].std() > 0:
            corr = window['x'].corr(window['Y'])
            correlations.append(corr)
            window_centers.append(window['x'].mean())

    print("\n4. SLIDING WINDOW CORRELATION")
    print(f"Window size: {window_size}")
    print(f"Correlation range: [{min(correlations):.4f}, {max(correlations):.4f}]")
    print(f"Correlation std: {np.std(correlations):.4f}")

# 5. Test specific hypotheses about breakpoints
# Hypothesis: diminishing returns pattern (strong growth early, plateauing later)
breakpoint_candidates = [5, 10, 15, 20]

print("\n5. TESTING SPECIFIC BREAKPOINTS")
for bp in breakpoint_candidates:
    early = data[data['x'] <= bp]
    late = data[data['x'] > bp]

    if len(early) >= 3 and len(late) >= 3:
        early_corr = early['x'].corr(early['Y'])
        late_corr = late['x'].corr(late['Y'])

        # Fit linear models to each segment
        early_slope = np.polyfit(early['x'], early['Y'], 1)[0]
        late_slope = np.polyfit(late['x'], late['Y'], 1)[0]

        print(f"\nBreakpoint at x={bp}:")
        print(f"  Early (x≤{bp}): n={len(early)}, corr={early_corr:.4f}, slope={early_slope:.4f}")
        print(f"  Late (x>{bp}): n={len(late)}, corr={late_corr:.4f}, slope={late_slope:.4f}")
        print(f"  Slope ratio (late/early): {late_slope/early_slope if early_slope != 0 else 'inf':.4f}")

print("\n" + "="*60)

# Visualize segmentations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Segmentation Analysis', fontsize=16, fontweight='bold')

# 1. Quantile-based
ax1 = axes[0, 0]
colors = ['red', 'green', 'blue']
for idx, (name, segment) in enumerate(segments_quantile.items()):
    ax1.scatter(segment['x'], segment['Y'], alpha=0.7, s=100,
               label=name, color=colors[idx], edgecolors='black', linewidths=0.5)
for q in q_values:
    ax1.axvline(x=q, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('Y', fontsize=11)
ax1.set_title('Quantile-Based Segmentation', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Equal-width bins
ax2 = axes[0, 1]
colors = ['purple', 'orange', 'brown']
for idx, (name, segment) in enumerate(segments_equal.items()):
    if len(segment) > 0:
        ax2.scatter(segment['x'], segment['Y'], alpha=0.7, s=100,
                   label=name, color=colors[idx], edgecolors='black', linewidths=0.5)
for b in bins[1:-1]:
    ax2.axvline(x=b, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('Y', fontsize=11)
ax2.set_title('Equal-Width Bins', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Sliding window correlation
if len(correlations) > 0:
    ax3 = axes[1, 0]
    ax3.plot(window_centers, correlations, 'b-', linewidth=2, marker='o', markersize=6)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('x (window center)', fontsize=11)
    ax3.set_ylabel('Local Correlation', fontsize=11)
    ax3.set_title(f'Sliding Window Correlation (window={window_size})',
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

# 4. Breakpoint comparison
ax4 = axes[1, 1]
for bp in [10, 15]:
    early = data[data['x'] <= bp]
    late = data[data['x'] > bp]

    ax4.scatter(early['x'], early['Y'], alpha=0.6, s=80,
               label=f'x≤{bp}', edgecolors='black', linewidths=0.5)
    ax4.scatter(late['x'], late['Y'], alpha=0.6, s=80,
               label=f'x>{bp}', edgecolors='black', linewidths=0.5)

    # Fit lines
    if len(early) >= 2:
        z = np.polyfit(early['x'], early['Y'], 1)
        x_line = np.linspace(early['x'].min(), early['x'].max(), 50)
        ax4.plot(x_line, z[0]*x_line + z[1], '--', linewidth=2, alpha=0.7)

    if len(late) >= 2:
        z = np.polyfit(late['x'], late['Y'], 1)
        x_line = np.linspace(late['x'].min(), late['x'].max(), 50)
        ax4.plot(x_line, z[0]*x_line + z[1], '--', linewidth=2, alpha=0.7)

    ax4.axvline(x=bp, color='red', linestyle='--', alpha=0.5)
    break  # Just show one for clarity

ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('Y', fontsize=11)
ax4.set_title('Piecewise Linear Fits', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/10_segmentation_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved: 10_segmentation_analysis.png")
print("Segmentation analysis complete.")
