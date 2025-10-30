"""
Create visual summary of adequacy assessment.
Shows the complete validation journey and final decision.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Create comprehensive summary figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Color scheme
color_pass = '#2ecc71'
color_warn = '#f39c12'
color_excellent = '#27ae60'
color_reject = '#e74c3c'
color_neutral = '#95a5a6'

# ============================================================================
# Panel A: Validation Pipeline (Flowchart)
# ============================================================================
ax1 = fig.add_subplot(gs[0:2, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis('off')
ax1.set_title('Validation Pipeline - Model 1', fontsize=14, fontweight='bold', pad=10)

stages = [
    (5, 13, 'Prior Predictive\nCheck', 'PASS\n(revised)', color_pass),
    (5, 11.5, 'Simulation-Based\nCalibration', 'PASS\n100/100', color_excellent),
    (5, 10, 'Posterior\nInference', 'EXCELLENT\n(R̂=1.00)', color_excellent),
    (5, 8.5, 'Posterior\nPredictive Check', 'PASS\n(6/7)', color_pass),
    (5, 7, 'Model\nCritique', 'PASS\n(4/5)', color_pass),
    (5, 5.5, 'Model\nComparison', 'WON\n(ΔELPD=3.3)', color_excellent),
    (5, 4, 'Model\nAssessment', 'EXCELLENT\n(R²=0.89)', color_excellent),
]

for i, (x, y, label, result, color) in enumerate(stages):
    # Stage box
    box = FancyBboxPatch((x-1.5, y-0.35), 3, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='white', linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Result badge
    badge = FancyBboxPatch((x+1.8, y-0.25), 1.5, 0.5,
                          boxstyle="round,pad=0.05",
                          edgecolor=color, facecolor=color, linewidth=2, alpha=0.8)
    ax1.add_patch(badge)
    ax1.text(x+2.55, y, result, ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Arrow to next stage
    if i < len(stages) - 1:
        arrow = FancyArrowPatch((x, y-0.4), (x, stages[i+1][1]+0.4),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color=color_neutral)
        ax1.add_patch(arrow)

# Final verdict
verdict_box = FancyBboxPatch((1, 1.5), 8, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor=color_excellent, facecolor=color_excellent, 
                            linewidth=3, alpha=0.9)
ax1.add_patch(verdict_box)
ax1.text(5, 2.25, 'ADEQUATE - MODELING COMPLETE', 
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax1.text(5, 1.75, '7/7 Validation Stages Passed', 
        ha='center', va='center', fontsize=10, color='white')

# ============================================================================
# Panel B: Model Comparison
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
models = ['Model 1\n(Logarithmic)', 'Model 2\n(Change-Point)']
elpd = [23.71, 20.39]
elpd_se = [3.09, 3.35]
colors_bar = [color_excellent, color_neutral]

bars = ax2.bar(models, elpd, yerr=elpd_se, capsize=5, 
              color=colors_bar, edgecolor='black', linewidth=2, alpha=0.8)
ax2.set_ylabel('ELPD-LOO', fontsize=11, fontweight='bold')
ax2.set_title('Model Comparison (Higher = Better)', fontsize=12, fontweight='bold')
ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax2.grid(axis='y', alpha=0.3)

# Add difference annotation
ax2.annotate('', xy=(0, 23.71), xytext=(1, 20.39),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(0.5, 22, 'ΔELPD = 3.31\n(Model 1 wins)', 
        ha='center', fontsize=9, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))

# ============================================================================
# Panel C: Performance Metrics
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
metrics_names = ['R²', 'Coverage\n(90%)', 'Max\nPareto-k', 'RMSE\nImprovement']
metrics_values = [0.893, 0.963, 0.325, 0.672]
metrics_targets = [0.70, 0.90, 0.50, 0.50]
metrics_colors = [color_excellent if v >= t else color_warn 
                 for v, t in zip(metrics_values, metrics_targets)]

x_pos = np.arange(len(metrics_names))
bars = ax3.bar(x_pos, metrics_values, color=metrics_colors, 
              edgecolor='black', linewidth=2, alpha=0.8)
ax3.plot(x_pos, metrics_targets, 'ko--', linewidth=2, markersize=8, 
        label='Target', alpha=0.6)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics_names, fontsize=9)
ax3.set_ylabel('Value', fontsize=11, fontweight='bold')
ax3.set_title('Performance vs. Targets', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_ylim(0, 1.1)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, metrics_values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ============================================================================
# Panel D: Parameter Estimates (Key Scientific Parameters)
# ============================================================================
ax4 = fig.add_subplot(gs[0, 2])
params = ['α\n(intercept)', 'β\n(log-slope)', 'σ\n(scale)']
means = [1.650, 0.314, 0.093]
lower = [1.450, 0.256, 0.069]
upper = [1.801, 0.386, 0.128]
errors_low = [means[i] - lower[i] for i in range(3)]
errors_high = [upper[i] - means[i] for i in range(3)]

x_pos = np.arange(len(params))
ax4.errorbar(x_pos, means, yerr=[errors_low, errors_high], 
            fmt='o', markersize=10, capsize=8, capthick=3,
            color=color_excellent, ecolor=color_excellent, 
            linewidth=3, markeredgecolor='black', markeredgewidth=2)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(params, fontsize=10, fontweight='bold')
ax4.set_ylabel('Posterior Estimate', fontsize=11, fontweight='bold')
ax4.set_title('Key Parameter Estimates (95% CI)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add mean labels
for i, (x, y) in enumerate(zip(x_pos, means)):
    ax4.text(x+0.15, y, f'{y:.3f}', fontsize=9, fontweight='bold')

# ============================================================================
# Panel E: Convergence Diagnostics
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
diag_names = ['R-hat\n(max)', 'ESS\n(min)', 'Divergences', 'Runtime\n(min)']
diag_values = [1.0014, 1739, 0, 2]
diag_targets = [1.01, 400, 0, 10]
diag_normalized = [
    1.0 - (diag_values[0] - 1.0)/(diag_targets[0] - 1.0),  # Lower is better, invert
    min(diag_values[1] / diag_targets[1], 1.5),  # ESS ratio
    1.0,  # Divergences (0 = perfect)
    min(diag_targets[3] / diag_values[3], 1.5)  # Runtime (faster is better)
]

x_pos = np.arange(len(diag_names))
bars = ax5.bar(x_pos, diag_normalized, color=color_excellent, 
              edgecolor='black', linewidth=2, alpha=0.8)
ax5.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Target')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(diag_names, fontsize=9)
ax5.set_ylabel('Quality Score\n(1.0 = Target, >1.0 = Excellent)', fontsize=10, fontweight='bold')
ax5.set_title('Convergence Diagnostics', fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.set_ylim(0, 1.6)
ax5.grid(axis='y', alpha=0.3)

# Add actual values
actual_labels = ['1.001', '1739', '0', '2 min']
for i, (bar, label) in enumerate(zip(bars, actual_labels)):
    ax5.text(bar.get_x() + bar.get_width()/2., 0.05,
            label, ha='center', va='bottom', fontsize=8, fontweight='bold', color='white')

# ============================================================================
# Panel F: Scientific Questions Answered
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0:2])
questions = [
    'What is the relationship form?',
    'Is it linear or non-linear?',
    'What is the effect magnitude?',
    'How much uncertainty?',
    'Does it saturate?'
]
answers = [
    'YES: Logarithmic diminishing returns',
    'YES: Non-linear (log > linear by 31%)',
    'YES: β = 0.31 [0.26, 0.39]',
    'YES: Well-calibrated (96% coverage)',
    'PARTIAL: Smooth deceleration (no asymptote)'
]
confidence = [1.0, 1.0, 1.0, 1.0, 0.7]  # For coloring

y_pos = np.arange(len(questions))[::-1]  # Reverse for top-down
colors_conf = [color_excellent if c >= 0.9 else color_pass for c in confidence]

ax6.barh(y_pos, [1]*len(questions), color=colors_conf, edgecolor='black', 
        linewidth=2, alpha=0.3)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(questions, fontsize=10, fontweight='bold')
ax6.set_xlim(0, 1)
ax6.set_xticks([])
ax6.set_title('Scientific Questions Assessment', fontsize=12, fontweight='bold', pad=10)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)

# Add answers as text
for i, (y, answer, conf) in enumerate(zip(y_pos, answers, confidence)):
    badge_text = 'HIGH' if conf >= 0.9 else 'MOD'
    badge_color = color_excellent if conf >= 0.9 else color_pass
    ax6.text(0.02, y, answer, ha='left', va='center', fontsize=9, fontweight='bold')
    ax6.text(0.98, y, badge_text, ha='right', va='center', fontsize=8, 
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor=badge_color, 
                     edgecolor='black', linewidth=1.5))

# ============================================================================
# Panel G: Refinement Options Analysis
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])
refinements = ['Prior\nSensitivity', 'Outlier\nInfluence', 'Hetero-\nscedastic', 
               'Splines/GP', 'Additional\nModels']
expected_gain = [0.01, 0.00, 0.00, 0.03, -0.02]  # Negative = worse
effort_hours = [2.5, 1, 3.5, 7, 6]

x_pos = np.arange(len(refinements))
colors_ref = [color_neutral if g <= 0.02 else color_warn for g in expected_gain]

# Bubble plot: x=effort, y=gain, size=importance
for i, (eff, gain, name) in enumerate(zip(effort_hours, expected_gain, refinements)):
    color = color_reject if gain < 0 else (color_neutral if gain < 0.02 else color_warn)
    ax7.scatter(eff, gain, s=500, c=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax7.text(eff, gain, name, ha='center', va='center', fontsize=8, fontweight='bold')

ax7.axhline(0.02, color='orange', linestyle='--', linewidth=2, alpha=0.6, 
           label='Worthwhile (>2% gain)')
ax7.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax7.set_xlabel('Effort (hours)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Expected Gain (ΔELPD or ΔR²)', fontsize=11, fontweight='bold')
ax7.set_title('Refinement Cost-Benefit Analysis', fontsize=12, fontweight='bold')
ax7.legend(loc='upper right', fontsize=8)
ax7.grid(alpha=0.3)
ax7.set_xlim(-0.5, 8)
ax7.set_ylim(-0.04, 0.05)

# Add "NOT JUSTIFIED" zone
ax7.fill_between([-0.5, 8], -0.04, 0.02, color='red', alpha=0.1)
ax7.text(4, 0.01, 'NOT JUSTIFIED ZONE', ha='center', fontsize=9, 
        fontweight='bold', color='red', alpha=0.5)

# ============================================================================
# Panel H: Decision Summary Box
# ============================================================================
ax8 = fig.add_subplot(gs[3, :])
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 3)
ax8.axis('off')

# Main decision box
decision_box = FancyBboxPatch((0.5, 0.3), 9, 2.4,
                             boxstyle="round,pad=0.15",
                             edgecolor=color_excellent, facecolor='white', 
                             linewidth=4)
ax8.add_patch(decision_box)

# Title
ax8.text(5, 2.3, 'FINAL DECISION: ADEQUATE', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=color_excellent)

# Key metrics summary
summary_text = """
✓ VALIDATION: 7/7 stages passed | 100% success rate | 0 convergence failures
✓ PERFORMANCE: R² = 0.893 | 96.3% coverage | RMSE = 3.8% | All Pareto-k < 0.5
✓ COMPARISON: Model 1 beats Model 2 by ΔELPD = 3.31 (moderate preference + parsimony)
✓ SCIENTIFIC: All core questions answered | β = 0.31 [0.26, 0.39] precisely estimated
✓ REFINEMENTS: All analyzed options show <3% gain with substantial effort cost
"""

ax8.text(5, 1.2, summary_text.strip(), 
        ha='center', va='center', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, 
                 edgecolor=color_neutral, linewidth=2))

# Bottom line
ax8.text(5, 0.5, 'MODELING COMPLETE - NO FURTHER ITERATION WARRANTED', 
        ha='center', va='center', fontsize=12, fontweight='bold', 
        style='italic', color=color_excellent)

# ============================================================================
# Overall title
# ============================================================================
fig.suptitle('Bayesian Modeling Adequacy Assessment - Complete Evaluation Summary', 
            fontsize=16, fontweight='bold', y=0.98)

# Save
plt.savefig('/workspace/experiments/adequacy_summary.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Adequacy summary visualization saved to: /workspace/experiments/adequacy_summary.png")

plt.close()

# ============================================================================
# Create simplified decision flowchart
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Decision tree structure
decisions = [
    (5, 9, 'Is model adequate?', 'Assess all validation stages', None),
    (2, 7, '7/7 stages\npassed?', 'YES', color_excellent),
    (8, 7, 'Scientific\nquestions\nanswered?', 'YES\n(4 full + 1 partial)', color_excellent),
    (2, 5, 'Performance\nexcellent?', 'YES\nR²=0.89', color_excellent),
    (5, 5, 'Refinements\nworthwhile?', 'NO\n(<3% gain)', color_reject),
    (8, 5, 'Alternatives\nbetter?', 'NO\n(Model 2 worse)', color_reject),
    (5, 3, 'Continue\nmodeling?', 'NO', color_reject),
    (5, 1, 'ADEQUATE\nProceed to reporting', '', color_excellent),
]

for x, y, question, answer, color in decisions:
    if color:
        # Decision box
        box = FancyBboxPatch((x-0.8, y-0.35), 1.6, 0.7,
                            boxstyle="round,pad=0.08",
                            edgecolor='black', facecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, question, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        # Answer badge
        if answer and answer != '':
            badge = FancyBboxPatch((x-0.9, y-0.8), 1.8, 0.35,
                                  boxstyle="round,pad=0.05",
                                  edgecolor=color, facecolor=color, 
                                  linewidth=2, alpha=0.8)
            ax.add_patch(badge)
            ax.text(x, y-0.625, answer, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
    else:
        # Start node
        circle = plt.Circle((x, y), 0.5, color=color_neutral, ec='black', lw=2)
        ax.add_patch(circle)
        ax.text(x, y, question, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        ax.text(x, y-0.7, answer, ha='center', va='top',
               fontsize=8, style='italic')

# Arrows
arrows = [
    ((5, 8.5), (2, 7.4), 'Check'),
    ((5, 8.5), (8, 7.4), 'Check'),
    ((2, 6.6), (2, 5.4), 'YES'),
    ((8, 6.6), (5, 5.4), 'YES'),
    ((2, 4.6), (5, 3.4), 'NO'),
    ((8, 4.6), (5, 3.4), 'NO'),
    ((5, 2.6), (5, 1.4), 'STOP'),
]

for (x1, y1), (x2, y2), label in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color=color_neutral)
    ax.add_patch(arrow)
    # Add label
    mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
    ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
           fontweight='bold', style='italic', color=color_neutral)

# Title
ax.text(5, 9.7, 'Decision Flowchart: Is Modeling Complete?',
       ha='center', fontsize=14, fontweight='bold')

plt.savefig('/workspace/experiments/adequacy_flowchart.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Decision flowchart saved to: /workspace/experiments/adequacy_flowchart.png")

print("\n" + "="*70)
print("ADEQUACY ASSESSMENT COMPLETE")
print("="*70)
print("\nFiles created:")
print("1. /workspace/experiments/adequacy_assessment.md (comprehensive report)")
print("2. /workspace/experiments/adequacy_summary.png (visual summary)")
print("3. /workspace/experiments/adequacy_flowchart.png (decision tree)")
print("\nFINAL DECISION: ADEQUATE - Modeling Complete")
print("="*70)
