"""
Master script to run all transformation and feature engineering analyses

This script orchestrates the complete EDA workflow for Analyst 3:
1. Initial data exploration
2. Comprehensive transformation analysis
3. Polynomial and exponential model comparison
4. Visualization generation
5. Advanced diagnostic plots

Run this script to reproduce all analyses and visualizations.
"""

import subprocess
import sys
import os

# Ensure we're in the correct directory
os.chdir('/workspace/eda/analyst_3')

print("="*70)
print("ANALYST 3: FEATURE ENGINEERING & TRANSFORMATION ANALYSIS")
print("="*70)
print()

scripts = [
    ("01_initial_exploration.py", "Initial Data Exploration"),
    ("02_transformation_analysis.py", "Transformation Performance Analysis"),
    ("02b_polynomial_analysis.py", "Polynomial and Exponential Models"),
    ("03_visualization_transformations.py", "Core Visualizations"),
    ("04_advanced_visualizations.py", "Advanced Diagnostic Plots"),
]

for script, description in scripts:
    print(f"\nRunning: {description}")
    print("-" * 70)

    result = subprocess.run(
        [sys.executable, f"code/{script}"],
        cwd='/workspace/eda/analyst_3',
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✓ {description} completed successfully")
    else:
        print(f"✗ Error in {script}:")
        print(result.stderr)
        sys.exit(1)

print("\n" + "="*70)
print("ALL ANALYSES COMPLETED SUCCESSFULLY")
print("="*70)
print()
print("Generated outputs:")
print("  - Code: /workspace/eda/analyst_3/code/")
print("  - Visualizations: /workspace/eda/analyst_3/visualizations/")
print("  - Main findings: /workspace/eda/analyst_3/findings.md")
print("  - Detailed log: /workspace/eda/analyst_3/eda_log.md")
print()
print("Key recommendation: GLM with log link (Poisson/NegBin) with quadratic terms")
print("="*70)
