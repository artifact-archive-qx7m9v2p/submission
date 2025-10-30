# Decision Flowchart: Practical Model Selection
## Designer #3: Visual Decision Guide

**Purpose**: Step-by-step decision tree for model selection with clear stopping rules

---

## Main Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        START HERE                                │
│             Data: 12 groups, overdispersion confirmed            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Q1: Do you need estimates for INDIVIDUAL GROUPS?                 │
│                                                                   │
│ Examples of YES:                                                  │
│ - "Which group has the highest success rate?"                    │
│ - "Should we invest more in Group X?"                            │
│ - "Predict future performance of Group Y"                        │
│                                                                   │
│ Examples of NO:                                                   │
│ - "What's the average success rate?"                             │
│ - "How variable are the groups?"                                 │
│ - "Predict a new, unseen group"                                  │
└─────────────────┬───────────────────────┬───────────────────────┘
                  │                       │
            ┌─────┘                       └─────┐
            │ NO                                │ YES
            ▼                                   ▼
┌───────────────────────┐         ┌─────────────────────────────┐
│   MODEL 2              │         │   MODEL 1                   │
│   Beta-Binomial        │         │   Hierarchical Binomial     │
│                        │         │   (Non-Centered)            │
│ ✓ Simple (2 params)    │         │                             │
│ ✓ Fast (<10 sec)       │         │ ✓ Group-specific estimates  │
│ ✓ Easy to explain      │         │ ✓ Partial pooling           │
│ ✗ No group estimates   │         │ ✓ Automatic shrinkage       │
└───────────┬───────────┘         └─────────────┬───────────────┘
            │                                     │
            │                                     │
            ▼                                     ▼
┌───────────────────────┐         ┌─────────────────────────────┐
│ FIT MODEL 2            │         │ FIT MODEL 1                 │
│                        │         │                             │
│ 4 chains, 2000 iter    │         │ 4 chains, 2000 iter         │
│ Expected: <10 seconds  │         │ Expected: 30-60 seconds     │
└───────────┬───────────┘         └─────────────┬───────────────┘
            │                                     │
            │                                     │
            ▼                                     ▼
┌───────────────────────┐         ┌─────────────────────────────┐
│ CHECK DIAGNOSTICS      │         │ CHECK DIAGNOSTICS           │
│                        │         │                             │
│ ☑ Rhat < 1.01?         │         │ ☑ Rhat < 1.01?              │
│ ☑ ESS > 400?           │         │ ☑ ESS > 400?                │
│ ☑ φ ≈ 3.6?             │         │ ☑ Divergences < 1%?         │
└───────────┬───────────┘         └─────────────┬───────────────┘
            │                                     │
        All Pass?                             All Pass?
            │                                     │
   ┌────────┴────────┐                  ┌────────┴────────┐
   │ YES             │ NO               │ YES             │ NO
   ▼                 ▼                  ▼                 ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐
│ ✓ ACCEPT │  │ Data problem │  │ Continue │  │ Troubleshoot     │
│ MODEL 2  │  │ Investigate  │  │ to LOO   │  │                  │
└─────┬────┘  └──────────────┘  └─────┬────┘  └────────┬─────────┘
      │                                 │                │
      │                                 │                │
      │                                 ▼                ▼
      │                   ┌──────────────────────────────────────┐
      │                   │ COMPUTE LOO                          │
      │                   │                                      │
      │                   │ ☑ All Pareto k < 0.7?                │
      │                   │ ☑ PPC variance ≈ 3.6?                │
      │                   │ ☑ Shrinkage logical?                 │
      │                   └─────────────┬────────────────────────┘
      │                                 │
      │                             All Pass?
      │                                 │
      │                        ┌────────┴────────┐
      │                        │ YES             │ NO
      │                        ▼                 ▼
      │              ┌───────────────┐  ┌─────────────────────┐
      │              │ ✓ ACCEPT      │  │ Multiple k > 0.7?   │
      │              │ MODEL 1       │  │ OR PPC fails?       │
      │              └───────────────┘  └──────────┬──────────┘
      │                                             │
      │                                             │ YES
      │                                             ▼
      │                                   ┌──────────────────┐
      │                                   │ TRY MODEL 3      │
      │                                   │ Robust (Student-t)│
      │                                   └─────────┬────────┘
      │                                             │
      │                                             ▼
      │                                   ┌──────────────────┐
      │                                   │ FIT MODEL 3      │
      │                                   │                  │
      │                                   │ adapt_delta=0.95 │
      │                                   │ 2-3 minutes      │
      │                                   └─────────┬────────┘
      │                                             │
      │                                             ▼
      │                                   ┌──────────────────┐
      │                                   │ CHECK nu POST    │
      │                                   │                  │
      │                                   │ nu < 5?          │
      │                                   └────┬─────────┬───┘
      │                                        │         │
      │                                   YES  │         │ NO
      │                                        ▼         ▼
      │                              ┌──────────┐  ┌────────────┐
      │                              │ Heavy    │  │ Use Model 1│
      │                              │ tails    │  │ (simpler)  │
      │                              │ justified│  └────────────┘
      │                              └────┬─────┘
      │                                   │
      │                                   ▼
      │                              ┌──────────┐
      │                              │ LOO      │
      │                              │ better?  │
      │                              └────┬─────┘
      │                                   │
      │                          ┌────────┴────────┐
      │                          │ YES             │ NO
      │                          ▼                 ▼
      │                    ┌──────────┐     ┌───────────┐
      │                    │ ✓ ACCEPT │     │ Use Model 1│
      │                    │ MODEL 3  │     │ (simpler)  │
      │                    └──────────┘     └───────────┘
      │                          │                 │
      └──────────────────────────┴─────────────────┘
                                 │
                                 ▼
                     ┌────────────────────┐
                     │ REPORT RESULTS     │
                     └────────────────────┘
```

---

## Detailed Diagnostic Decision Trees

### Diagnostic Tree 1: Convergence Issues

```
Sampling Complete
       │
       ▼
Is Rhat > 1.01 for any parameter?
       │
       ├─ NO  → Good! Continue to ESS check
       │
       └─ YES → Which parameters?
              │
              ├─ Only tau
              │    │
              │    └─ Is tau near 0?
              │         │
              │         ├─ YES → OK, groups very similar
              │         │         Use pooled model instead
              │         │
              │         └─ NO  → Run longer
              │                  (4000 iterations)
              │
              ├─ Only individual p[j]
              │    │
              │    └─ Check that group's data
              │         Data error? Outlier?
              │
              └─ Multiple parameters
                   │
                   └─ Serious problem
                        1. Check model code
                        2. Check data quality
                        3. Try simpler model
```

### Diagnostic Tree 2: Divergent Transitions

```
Divergent transitions detected
       │
       ▼
How many? (as % of post-warmup)
       │
       ├─ <1%
       │   │
       │   └─ Acceptable, proceed with caution
       │       (May indicate minor geometry issues)
       │
       ├─ 1-5%
       │   │
       │   └─ Try increasing adapt_delta
       │       │
       │       ├─ Set to 0.90 → Still diverging?
       │       ├─ Set to 0.95 → Still diverging?
       │       └─ Set to 0.99 → Still diverging?
       │                        │
       │                        └─ YES: Model misspecification
       │                                Try robust model (Model 3)
       │
       └─ >5%
           │
           └─ Serious geometry problem
               │
               ├─ Check for extreme priors
               ├─ Verify non-centered parameterization
               └─ Try robust model (Model 3)
```

### Diagnostic Tree 3: LOO Pareto k

```
Compute LOO
       │
       ▼
Check Pareto k values
       │
       ├─ All k < 0.5
       │   │
       │   └─ Excellent! Model well-specified
       │
       ├─ Some k = 0.5-0.7
       │   │
       │   └─ Which groups?
       │       │
       │       ├─ Small-n groups (1, 10)
       │       │   │
       │       │   └─ OK, expected for small samples
       │       │
       │       └─ Outliers (2, 4, 8)
       │           │
       │           └─ Consider robust model (Model 3)
       │
       └─ Multiple k > 0.7
           │
           └─ Model misspecification
               │
               ├─ Try Model 3 (robust)
               │
               └─ If Model 3 also fails:
                   - Check data quality
                   - Consider covariates
                   - Consult domain expert
```

### Diagnostic Tree 4: Posterior Predictive Checks

```
PPC: Does model capture variance?
       │
       ▼
Observed var = 3.6 × binomial
       │
       ├─ Model var = 3.4-3.8 × binomial
       │   │
       │   └─ ✓ Pass! Heterogeneity captured
       │
       ├─ Model var = 1.0-2.0 × binomial
       │   │
       │   └─ ✗ Fail: Under-dispersed
       │       Check model specification
       │       (Are you using pooled model?)
       │
       └─ Model var > 5.0 × binomial
           │
           └─ ✗ Fail: Over-dispersed
               Check for:
               - Data errors
               - Extreme outliers
               - Model over-fitting

PPC: Does model capture range?
       │
       ▼
Observed range: 3.1% - 14.0%
       │
       ├─ Model 95% PI includes range
       │   │
       │   └─ ✓ Pass!
       │
       └─ Model 95% PI misses extremes
           │
           └─ ✗ Fail: Try robust model
               (Normal tails too light)
```

---

## Model Comparison Decision Tree

```
All models converged
       │
       ▼
Compute LOO for each
       │
       ▼
Compare ΔLOO (difference from best)
       │
       ├─ ΔLOO < 2
       │   │
       │   └─ Models effectively tied
       │       │
       │       └─ Choose SIMPLER model
       │           (Occam's razor)
       │           │
       │           ├─ Beta-binomial (2 params)
       │           ├─ Hierarchical (14 params)
       │           └─ Robust (15 params)
       │
       ├─ ΔLOO = 2-10
       │   │
       │   └─ Weak preference for better model
       │       │
       │       └─ Decision factors:
       │           - Interpretability needs
       │           - Computational constraints
       │           - Stakeholder preferences
       │           │
       │           └─ If uncertain: Choose BETTER model
       │
       └─ ΔLOO > 10
           │
           └─ Strong preference for better model
               │
               └─ Choose BETTER model
                   (Evidence is clear)
```

---

## Prior Sensitivity Decision Tree

```
Should I check prior sensitivity?
       │
       ├─ NO, if:
       │   ├─ ΔLOO > 10 (clear winner)
       │   ├─ All diagnostics perfect
       │   ├─ Using standard weakly informative priors
       │   └─ Results align with domain knowledge
       │       │
       │       └─ SKIP sensitivity analysis
       │
       └─ YES, if:
           ├─ ΔLOO ≈ 4-6 (ambiguous)
           ├─ Small sample (J < 20 groups)
           ├─ Unusual parameter estimates
           └─ Stakeholders skeptical
               │
               └─ RUN sensitivity analysis
                   │
                   ▼
Refit with alternative priors
       │
       ├─ μ: Normal(-2.5, 0.5) vs Normal(-2.5, 2)
       ├─ τ: Half-Cauchy(0, 0.5) vs Half-Cauchy(0, 2)
       └─ For beta-binomial:
           └─ φ: Gamma(2, 0.2) vs Gamma(0.5, 0.05)
               │
               ▼
Compare posterior means
       │
       ├─ Parameters change < 10%
       │   │
       │   └─ ✓ Results robust to priors
       │
       ├─ Parameters change 10-20%
       │   │
       │   └─ ⚠ Moderate sensitivity
       │       Report range of estimates
       │
       └─ Parameters change > 20%
           │
           └─ ✗ High sensitivity
               │
               ├─ Use more informative priors
               │   (based on domain knowledge)
               │
               └─ OR acknowledge uncertainty
                   (report all plausible estimates)
```

---

## Troubleshooting: Common Problems

### Problem: "Sampling is taking forever"

```
Sampling > 5 minutes for 12 groups?
       │
       ├─ Using centered parameterization?
       │   │
       │   └─ Switch to non-centered (10x speedup)
       │
       ├─ Low adapt_delta (<0.9)?
       │   │
       │   └─ OK, this slows sampling
       │       Accept it if needed for convergence
       │
       └─ Complex model (Model 3)?
           │
           └─ Expected, 2-3 min is normal
               If >10 min: Check for bugs
```

### Problem: "τ posterior is all near zero"

```
Posterior τ ≈ 0?
       │
       └─ Groups may be very similar
           │
           ├─ Check ICC from EDA
           │   │
           │   └─ EDA showed ICC=0.56
           │       So τ shouldn't be near 0
           │       │
           │       └─ Prior too strong?
           │           Weaken τ prior
           │
           └─ OR: Hierarchical model not needed
               Consider pooled model instead
```

### Problem: "Extreme shrinkage"

```
Large groups (n>500) shrinking >40%?
       │
       └─ Problem: Over-shrinking
           │
           ├─ Check τ estimate
           │   │
           │   └─ τ too small?
           │       - Weaken τ prior
           │       - Check for model misspec
           │
           └─ Check μ estimate
               │
               └─ μ far from data?
                   - Weaken μ prior
                   - Check for outliers
```

### Problem: "Results don't make sense"

```
Parameter estimates implausible?
       │
       ├─ Any p[j] > 0.30?
       │   │
       │   └─ Check raw data for that group
       │       Data entry error?
       │
       ├─ τ > 2?
       │   │
       │   └─ Extreme heterogeneity
       │       - Check for subpopulations
       │       - Consider covariates
       │
       └─ μ implies <1% or >30% rate?
           │
           └─ Prior-data conflict
               - Check data quality
               - Reconsider prior
```

---

## Stopping Rules

### STOP ITERATING when:

```
✓ All diagnostics pass
  └─ Rhat < 1.01
  └─ ESS > 400
  └─ Divergences < 1%

✓ Predictive performance acceptable
  └─ LOO Pareto k < 0.7
  └─ PPC passes (variance, range)

✓ Results stable
  └─ Same conclusions across runs
  └─ Prior sensitivity < 10%

✓ Results interpretable
  └─ Parameters scientifically plausible
  └─ Stakeholders understand outputs

→ ACCEPT MODEL AND REPORT RESULTS
```

### ESCALATE when:

```
✗ Model 1 fails specific checks
  └─ Multiple Pareto k > 0.7
  └─ PPC fails for outliers
  └─ Posterior conflicts with domain knowledge

→ TRY MODEL 3 (ROBUST)
```

### ABANDON when:

```
✗ All models fail
  └─ None converge despite tuning
  └─ All have poor predictive performance
  └─ Results contradict known facts

→ FUNDAMENTAL PROBLEM
  ├─ Revisit data quality
  ├─ Consider covariates
  ├─ Consult domain expert
  └─ Question model assumptions
```

---

## Final Decision Matrix

| Criterion | Model 1 | Model 2 | Model 3 |
|-----------|---------|---------|---------|
| **Need group estimates** | ✓ | ✗ | ✓ |
| **Computational speed** | Good (1 min) | Excellent (<10 sec) | Fair (2-3 min) |
| **Interpretability** | Good | Excellent | Fair |
| **Robustness to outliers** | Fair | N/A | Excellent |
| **Parameter count** | 14 | 2 | 15 |
| **Typical use case** | Standard | Population only | Outliers present |
| **Recommended start** | ⭐ YES | If no group estimates | Only if Model 1 fails |

---

## Quick Reference: Which Model?

**Use Model 1 (Hierarchical) if**:
- Need group-specific estimates (usual case)
- Want partial pooling benefits
- Standard hierarchical analysis

**Use Model 2 (Beta-binomial) if**:
- Only care about population mean/variance
- Want simplest defensible model
- Computational speed critical

**Use Model 3 (Robust) if**:
- Model 1 has Pareto k > 0.7
- Model 1 PPC fails for outliers
- Domain expects heavy-tailed heterogeneity

---

**This flowchart covers 95% of practical modeling decisions for this dataset.**

**Remember**: The goal is answering the scientific question, not achieving perfect diagnostics. "Good enough" is often truly good enough.
