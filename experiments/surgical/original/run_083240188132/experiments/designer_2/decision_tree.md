# Model Selection Decision Tree

## Visual Guide for Choosing Between Alternative Models

```
START: Fit Model 2 (Robust Student-t) FIRST
         │
         ├─ Posterior nu > 30?
         │    YES → STOP: Use standard hierarchical (normal random effects)
         │    NO → Continue to next check
         │
         ├─ Posterior nu in [5, 30]?
         │    YES → Model 2 ACCEPTED: Heavy tails confirmed
         │    NO → Continue to next check
         │
         └─ Posterior nu < 5?
              YES → Suggests distinct subpopulations
                    → Fit Model 1 (Finite Mixture K=2)
                    │
                    ├─ Mixture weight extreme (w < 0.1 or w > 0.9)?
                    │    YES → REJECT Model 1, use Model 2
                    │    NO → Continue
                    │
                    ├─ Components distinct (mu_diff > 0.03)?
                    │    YES → Model 1 ACCEPTED
                    │    NO → REJECT Model 1, use Model 2
                    │
                    └─ Compare Model 1 vs Model 2 via LOO
                         │
                         ├─ ΔLOO > 4 favoring Model 1?
                         │    YES → Use Model 1 (Mixture)
                         │
                         ├─ ΔLOO > 4 favoring Model 2?
                         │    YES → Use Model 2 (Robust)
                         │
                         └─ |ΔLOO| < 2*SE?
                              YES → Report BOTH models
                                    (they capture different aspects)

OPTIONAL: Fit Model 3 (DP Mixture) if:
         • Uncertain whether K=1, 2, or 3 clusters
         • Computational resources available
         • Want to let data determine K
         │
         ├─ K_eff = 1 consistently?
         │    YES → REJECT Model 3, use standard hierarchical
         │    NO → Continue
         │
         ├─ K_eff = 2 consistently?
         │    YES → Confirms Model 1 (K=2 mixture)
         │    NO → Continue
         │
         └─ K_eff > 2?
              YES → DP suggests more complex structure
                    → Investigate: Are clusters interpretable?
                    → Compare LOO to Models 1 & 2
```

---

## Falsification Quick Reference

### Model 1: Finite Mixture (K=2)
```
FALSIFIED if:
├─ w < 0.1 → "Low-risk component is empty"
├─ w > 0.9 → "High-risk component is empty"
├─ mu_diff < 0.03 → "Components not meaningfully distinct"
├─ LOO worse than unimodal → "Added complexity not justified"
└─ Component assignments unstable → "Weak identification"

ACCEPTED if:
├─ 0.2 < w < 0.8 → "Both components have mass"
├─ mu_diff > 0.05 → "Clear separation between components"
├─ LOO better than or similar to alternatives
└─ Posterior predictive checks pass
```

### Model 2: Robust Hierarchical (Student-t)
```
FALSIFIED if:
├─ nu > 30 → "Normal random effects sufficient"
├─ nu < 2 → "Extreme tails suggest misspecification"
├─ No improvement over normal RE → "Heavy tails not needed"
└─ Divergent transitions persist → "Model misspecified"

ACCEPTED if:
├─ 5 < nu < 30 → "Moderate heavy tails confirmed"
├─ LOO comparable to or better than alternatives
├─ Shrinkage less aggressive than normal RE
└─ Posterior predictive checks pass
```

### Model 3: Dirichlet Process Mixture
```
FALSIFIED if:
├─ K_eff = 1 consistently (>80%) → "No clustering detected"
├─ Extreme computational issues → "Model too complex for data"
├─ Clusters not interpretable → "Overfitting"
└─ LOO worse than finite mixture → "Unnecessary flexibility"

ACCEPTED if:
├─ K_eff posterior has clear mode (K=2, 3, or 4)
├─ LOO comparable to or better than alternatives
├─ Clusters are scientifically interpretable
└─ Computational diagnostics acceptable
```

---

## LOO Comparison Interpretation

### Comparing Two Models (A vs B)

```
ΔLOO = LOO_A - LOO_B
SE = Standard error of difference

Decision Rule:
├─ ΔLOO > 4 → Model A is clearly better
│                Use Model A
│
├─ ΔLOO < -4 → Model B is clearly better
│                Use Model B
│
├─ -2*SE < ΔLOO < 2*SE → Models indistinguishable
│                         Report both or use simpler one
│
└─ 2*SE < |ΔLOO| < 4 → Weak preference
                        Consider reporting both with caveat
```

### Comparing Multiple Models

```
Rank by ELPD_LOO (higher is better):

1. Check top model's LOO
   ├─ Pareto k > 0.7 for any point? → "Outliers or misspecification"
   └─ All Pareto k < 0.7 → "LOO reliable"

2. Compare top 2 models
   ├─ ΔLOO > 4 → Top model clearly best
   └─ ΔLOO < 2*SE → Indistinguishable

3. If indistinguishable:
   ├─ Use simpler model (Occam's razor)
   └─ Or report all with interpretation
```

---

## Posterior Predictive Check Failures

### What to Do When PPC Fails

```
PPC shows systematic misfit:

Check 1: Does model under-predict variance?
├─ YES → Try more flexible model (DP) or robust (Student-t)
└─ NO → Continue

Check 2: Does model miss specific groups?
├─ YES, outliers → Try robust model (less shrinkage)
├─ YES, zero-event group → Check prior on theta
└─ NO → Continue

Check 3: Does model miss distributional shape?
├─ YES, bimodality → Try mixture model
├─ YES, skewness → Check transformation or likelihood
└─ NO → Continue

Check 4: Does model over-predict variance?
├─ YES → Try simpler model (standard hierarchical)
└─ NO → Model may be fundamentally misspecified

If all checks fail:
→ Revisit likelihood (binomial assumption)
→ Check for within-group correlation
→ Consider completely different model class
```

---

## Computational Red Flags

### What to Do When MCMC Fails

```
Problem: Divergent Transitions

Step 1: Increase target_accept
├─ Try 0.95 (default for complex models)
├─ Try 0.99 (if still diverging)
└─ If still diverging → Model misspecification

Step 2: Check parameterization
├─ Centered vs non-centered (for hierarchical)
├─ Log-transform for positive parameters
└─ Standardize data if scale issues

Step 3: Examine divergent locations
├─ Divergences at extreme parameter values?
│   → Prior too diffuse or wrong likelihood
├─ Divergences at specific groups?
│   → Check data quality for those groups
└─ Random divergences?
    → May need longer warmup

If all fail → Model too complex for data
```

```
Problem: Low ESS (< 400 per chain)

Step 1: Check autocorrelation
├─ High autocorrelation in traces?
│   → Increase draws (3000, 4000)
│   → Consider thinning (last resort)
└─ Flat posteriors with poor mixing?
    → Non-identifiability

Step 2: Check parameterization
├─ Try non-centered parameterization
├─ Check for redundant parameters
└─ Consider stronger priors

If ESS < 100 → Serious identifiability issue
→ Simplify model or use more informative priors
```

```
Problem: Rhat > 1.01

Step 1: Visual inspection
├─ Plot traces for parameters with high Rhat
├─ Different chains at different modes?
│   → Multimodality (mixture models)
│   → Use ordering constraints
└─ Chains drifting slowly?
    → Need more warmup

Step 2: Increase warmup
├─ Try tune=2000 or 3000
└─ Check if Rhat improves

If Rhat > 1.05 → Non-convergence
→ Model misspecification or need more iterations
```

---

## Scientific Interpretation Guide

### Model 1 (Mixture): What Do Components Mean?

```
Posterior component assignments (z):

Group in Component 1 (low-risk):
└─ Typical groups with rates ~5-7%

Group in Component 2 (high-risk):
└─ Elevated groups with rates ~11-14%

Questions to ask:
1. Do components correspond to known subgroups?
   (e.g., different protocols, patient populations)

2. Are high-risk groups explainable?
   (e.g., sicker patients, different settings)

3. Can we predict component membership?
   (if covariates available)

4. Should we stratify future analyses by component?
```

### Model 2 (Robust): What Does nu Tell Us?

```
Posterior degrees of freedom (nu):

nu = 5-10:
└─ Moderate heavy tails
   → Population has more extreme variation than normal
   → Outliers are "real" but rare

nu = 10-20:
└─ Mild heavy tails
   → Slightly more variation than normal
   → Between-group differences substantial

nu > 30:
└─ Nearly normal
   → Standard hierarchical sufficient
   → No evidence of extreme variation

Interpretation:
"The population of group-level rates has heavy tails
 (nu = X), indicating [X/2] effective degrees of freedom
 worth of extra-normal variation. Outlier groups
 (2, 8, 11) receive [1-shrinkage] times less shrinkage
 than under a normal model."
```

### Model 3 (DP): What Does K_eff Mean?

```
Posterior effective clusters (K_eff):

K_eff = 1:
└─ No clustering detected
   → Use standard hierarchical

K_eff = 2:
└─ Two latent subpopulations
   → Confirms finite mixture (K=2)

K_eff = 3-4:
└─ More complex structure
   → Investigate: Are clusters interpretable?
   → May suggest more than two subgroups

K_eff > 5:
└─ Likely overfitting
   → Each group forming own cluster
   → Reduce alpha_dp or use simpler model

Interpretation:
"The data support approximately K_eff = X distinct
 subpopulations with meaningfully different risk profiles.
 These correspond to [describe clusters based on
 which groups are assigned together]."
```

---

## Final Decision Matrix

| Finding | Recommended Model | Alternative | Fallback |
|---------|------------------|-------------|----------|
| nu < 5, bimodal | Mixture (K=2) | DP Mixture | Robust |
| nu = 5-30 | Robust Student-t | Standard | Mixture |
| nu > 30 | Standard Hierarchical | N/A | N/A |
| K_eff = 1 | Standard Hierarchical | N/A | N/A |
| K_eff = 2 | Mixture (K=2) | DP (K=2) | Robust |
| K_eff > 2 | DP Mixture | Investigate | Mixture |
| LOO similar | Report all | Use simplest | Average |
| All LOO poor | Standard Hierarchical | Revisit EDA | New class |

---

**This decision tree should be consulted throughout the model fitting and assessment process to make principled decisions about model selection and falsification.**
