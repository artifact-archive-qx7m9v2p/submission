# Designer 1: Beta-Binomial Model Specifications
## Experiment Design Overview

**Designer:** Model Designer 1 (Beta-Binomial Focus)
**Date:** 2025-10-30
**Data:** 12 groups with severe overdispersion (φ=3.5-5.1, ICC=0.73)

---

## Philosophy: Adversarial Model Design

This experiment plan follows a **falsification-first** approach:
- Start with competing hypotheses (continuous vs mixture)
- Design models to FAIL under specific conditions
- Explicitly state what evidence would reject each model
- Mixture model serves as adversarial test of continuous variation assumption

**Success = Finding the right model**, not defending beta-binomial approach.

---

## Three Proposed Models

### Model A: Homogeneous Beta-Binomial (α, β)
- **Hypothesis:** Groups share common hyperparameters, continuous variation
- **Parameters:** α ~ Gamma(2, 0.5), β ~ Gamma(2, 0.1)
- **Strengths:** Simple, conjugate-like, EDA-supported
- **Falsification:** Abandon if mixture has ΔAIC < -10

### Model B: Reparameterized Beta-Binomial (μ, κ)
- **Hypothesis:** Same as A, more interpretable parameterization
- **Parameters:** μ ~ Beta(2, 18), κ ~ Gamma(2, 0.1)
- **Strengths:** Direct interpretation of mean and concentration
- **Falsification:** Same as Model A (identical likelihood)

### Model C: Two-Component Mixture
- **Hypothesis:** Discrete subpopulations (low-rate vs high-rate)
- **Parameters:** π ~ Beta(2,2), ordered component means
- **Strengths:** Tests clustering assumption
- **Falsification:** Abandon if components overlap or π→0/1

**Expected outcome:** Model B wins (interpretability), Model C loses (no clusters in EDA).

---

## Critical Decision Points

### RED FLAGS that trigger major pivots:

1. **If mixture wins decisively** (ΔAIC < -10):
   - Abandon continuous variation models
   - Investigate what defines clusters
   - Consider group-specific parameters instead

2. **If posterior κ << 1** (very low concentration):
   - Groups TOO heterogeneous for beta-binomial
   - Consider discrete group effects
   - May indicate need for covariates

3. **If all models fail posterior predictive checks**:
   - Problem is with BINOMIAL likelihood, not hierarchy
   - Consider beta-negative binomial
   - Check for zero-inflation beyond Group 1

4. **If Group 8 remains extreme outlier after shrinkage**:
   - Investigate data quality
   - Consider outlier model (robust likelihood)

5. **If shrinkage makes all groups identical**:
   - Data too sparse for hierarchical model
   - Revert to pooled estimate with caution

---

## Implementation Workflow

### Step 1: Prior Predictive Checks
```bash
cd /workspace/experiments/designer_1
python scripts/prior_predictive.py
```
**Verify:** Priors compatible with observed mean≈0.076, φ≈3.5

### Step 2: Fit All Models
```bash
python scripts/fit_models.py --model all
```
**Check:** Rhat < 1.01, ESS > 400, no divergences

### Step 3: Model Comparison
- Compare LOO-CV (lower is better)
- Check Pareto k values (< 0.7)
- Posterior predictive checks
- Choose simplest adequate model

### Step 4: Sensitivity Analysis
- Vary priors, check robustness
- Exclude Group 8, refit
- If conclusions stable → proceed

---

## Expected Results

### If Models Work as Expected:

**Model A/B posterior predictions:**
- μ ≈ 0.07-0.08 (near observed 0.076)
- κ ≈ 0.3-5 (allowing ICC≈0.73)
- φ ≈ 3.0-4.0 (matching observed)
- Group 1 shrinks to ~2-4% (from 0%)
- Group 8 shrinks to ~11-13% (from 14.4%)

**Model C posterior (if mixture is wrong):**
- π → 0 or π → 1 (one component vanishes)
- Component means overlap
- ΔAIC(C vs B) > 5
- Conclusion: No meaningful clustering

### If I'm Wrong:

**Model C wins:**
- Clear component separation (μ₁ << μ₂)
- π ≈ 0.3-0.7 (balanced)
- ΔAIC < -10
- **Action:** Abandon beta-binomial, investigate clustering

**All models fail:**
- Poor posterior predictive checks
- High Pareto k values
- **Action:** Rethink likelihood (zero-inflation? overdispersion beyond beta-binomial?)

---

## Files and Outputs

### Stan Models
- `/workspace/experiments/designer_1/stan_models/model_a_beta_binomial.stan`
- `/workspace/experiments/designer_1/stan_models/model_b_reparameterized.stan`
- `/workspace/experiments/designer_1/stan_models/model_c_mixture.stan`

### Python Scripts
- `scripts/fit_models.py` - Main fitting workflow
- `scripts/prior_predictive.py` - Prior checks
- (Additional scripts to be created: posterior_analysis.py, model_comparison.py)

### Results Directory
- `results/model_a/` - Fitted Model A outputs
- `results/model_b/` - Fitted Model B outputs
- `results/model_c/` - Fitted Model C outputs
- `results/prior_predictive_plots/` - Prior check visualizations
- `results/*_summary.csv` - Posterior summaries
- `results/*_loo.json` - LOO-CV results

---

## Key Design Principles

### 1. Falsification Over Confirmation
- Each model has explicit failure criteria
- Mixture model designed to CHALLENGE beta-binomial
- If data prefer mixture, I accept it

### 2. Prior Transparency
- All priors justified from EDA findings
- Weakly informative (not flat, not dogmatic)
- Prior predictive checks verify compatibility

### 3. Computational Honesty
- Report all diagnostics (Rhat, ESS, divergences)
- Don't hide problematic fits
- If model won't converge, it's telling us something

### 4. Uncertainty Acknowledgment
- Group 1 (0/47) creates real uncertainty
- Group 8 might be outlier or genuine
- ICC=0.73 is extreme - κ prior may be wrong

### 5. Adaptive Strategy
- Plan includes escape routes
- Ready to pivot if evidence demands
- Success = finding truth, not completing plan

---

## What Would Make Me Abandon Beta-Binomial Entirely?

1. **Temporal structure discovered** → State-space model
2. **Covariate information emerges** → Hierarchical GLM
3. **Overdispersion beyond beta-binomial** → Zero-inflation or alternative
4. **Spatial correlation** → GP or CAR prior
5. **Mixture wins decisively** → Group-specific discrete effects

---

## Contact and Collaboration

This is Designer 1's independent analysis. Compare with:
- Designer 2: (Different modeling approaches)
- Designer 3: (Different modeling approaches)

**Final model selection** should synthesize insights from all designers.

---

## Quick Start

```bash
# Navigate to experiment directory
cd /workspace/experiments/designer_1

# Run prior checks (recommended first step)
python scripts/prior_predictive.py

# Fit all models (takes ~5-10 minutes)
python scripts/fit_models.py --model all

# Or fit individual models
python scripts/fit_models.py --model a
python scripts/fit_models.py --model b
python scripts/fit_models.py --model c
```

---

## Expected Timeline

- Prior predictive checks: ~2 minutes
- Model A/B fitting: ~1 minute each
- Model C fitting: ~3-5 minutes (mixture is slower)
- Posterior analysis: ~5 minutes
- **Total: ~15-20 minutes** for complete workflow

---

**Remember:** The goal is finding a model that genuinely explains the data, not defending the beta-binomial approach. If the data tell us to pivot, we pivot.
