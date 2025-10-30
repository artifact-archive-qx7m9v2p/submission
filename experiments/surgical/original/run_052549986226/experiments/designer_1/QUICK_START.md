# Quick Start Guide - Designer 1 Models

## TL;DR

**Three models:** Beta-binomial (2 ways) + Mixture (falsification test)
**Expected winner:** Model B (reparameterized beta-binomial)
**Will abandon beta-binomial if:** Mixture has ΔAIC < -10

---

## Run Everything (5 minutes)

```bash
cd /workspace/experiments/designer_1

# 1. Check priors are sensible (~2 min)
python scripts/prior_predictive.py

# 2. Fit all models (~5-10 min)
python scripts/fit_models.py --model all

# 3. Check results
cat results/model_a_summary.csv
cat results/model_b_summary.csv
cat results/model_c_summary.csv
```

---

## The Three Models

### Model A: Beta-Binomial (α, β)
- **Priors:** α ~ Gamma(2, 0.5), β ~ Gamma(2, 0.1)
- **Parameters:** 2
- **Best for:** Simple, standard parameterization
- **Stan:** `stan_models/model_a_beta_binomial.stan`

### Model B: Beta-Binomial (μ, κ) [EXPECTED WINNER]
- **Priors:** μ ~ Beta(2, 18), κ ~ Gamma(2, 0.1)
- **Parameters:** 2 (same likelihood as A)
- **Best for:** Interpretability (μ = mean, κ = concentration)
- **Stan:** `stan_models/model_b_reparameterized.stan`

### Model C: Mixture [FALSIFICATION TEST]
- **Priors:** π ~ Beta(2,2), two component means (ordered)
- **Parameters:** 5
- **Best for:** Testing if discrete clusters exist
- **Stan:** `stan_models/model_c_mixture.stan`
- **Warning:** Expected to LOSE (no clusters in EDA)

---

## What to Look For

### Good Signs ✅
- Rhat < 1.01 for all parameters
- ESS > 400 (preferably > 1000)
- No divergent transitions
- LOO Pareto k < 0.7 for all groups
- Model B and A have nearly identical LOO (same model!)
- Model C has ΔLOO > 10 vs Model B

### Red Flags 🚩
- **If mixture wins (ΔLOO < -10):** Discrete clusters exist - rethink approach!
- **If κ << 1:** Groups TOO heterogeneous
- **If all models fail PPC:** Wrong likelihood (not binomial?)
- **If no convergence:** Model misspecification

---

## Expected Results

### Model B Posterior (if correct):
```
μ:     0.070 ± 0.015   [Mean success rate ≈ observed 0.076]
κ:     2.0  ± 1.5      [Low concentration = high heterogeneity]
φ:     3.5  ± 1.0      [Overdispersion matches observed]
```

### Group Shrinkage:
- **Group 1 (0/47):**    0.000 → 0.025 (shrink toward mean)
- **Group 8 (31/215):**  0.144 → 0.120 (moderate shrinkage)
- **Group 4 (46/810):**  0.057 → 0.058 (minimal shrinkage, large n)

### Model Comparison:
```
Model A LOO: ~-50 ± 5
Model B LOO: ~-50 ± 5  (essentially identical to A)
Model C LOO: ~-60 ± 8  (worse due to extra parameters)

ΔLOO(B vs C): ~10 → Continuous variation wins
```

---

## Decision Tree

```
1. Did all models converge? (Rhat < 1.01)
   NO → Increase iterations, check parameterization
   YES → Continue

2. Which model has best LOO?
   Model C (mixture) → SURPRISING! Investigate clustering
   Model A/B (continuous) → Expected, proceed

3. Does best model pass posterior predictive checks?
   NO → Likelihood misspecified, rethink approach
   YES → Use this model for inference

4. Are results robust to prior changes?
   NO → Report sensitivity, need more data
   YES → Final model selected!
```

---

## Falsification Checklist

### I will abandon continuous models (A/B) if:
- [ ] Model C has ΔLOO < -10
- [ ] Posterior shows clear bimodality
- [ ] Component separation > 3 SD
- [ ] Bayes Factor for mixture > 100

### I will abandon mixture model (C) if:
- [ ] π → 0 or π → 1 (one component vanishes)
- [ ] Component means overlap >80%
- [ ] ΔLOO(C vs B) > 5
- [ ] No interpretable clustering

### I will abandon ALL beta-binomial models if:
- [ ] All fail same posterior predictive check
- [ ] Overdispersion far exceeds beta-binomial capacity
- [ ] Zero-inflation beyond what beta-binomial can handle
- [ ] Temporal/spatial structure discovered

---

## Files You'll Need

### Stan Models (already created):
- `/workspace/experiments/designer_1/stan_models/model_a_beta_binomial.stan`
- `/workspace/experiments/designer_1/stan_models/model_b_reparameterized.stan`
- `/workspace/experiments/designer_1/stan_models/model_c_mixture.stan`

### Python Scripts (already created):
- `scripts/fit_models.py` - Fits all models, computes LOO
- `scripts/prior_predictive.py` - Checks priors are sensible

### Data:
- `/workspace/data/data.csv` (12 groups, n_trials, r_successes)

---

## Common Issues and Solutions

### Issue: Divergent transitions
**Solution:** Increase `adapt_delta` to 0.99

### Issue: Low ESS for κ (or kappa)
**Solution:** Use non-centered parameterization or more iterations

### Issue: Model C has convergence problems
**Solution:** Expected! Mixture models are harder. Try 8 chains, 4000 iterations.

### Issue: LOO has high Pareto k for Group 1 or 8
**Solution:** Outliers influence posterior. Check if Group 8 is data error.

### Issue: All models give same predictions
**Solution:** Data is weak, models pool heavily. Report high uncertainty.

---

## Interpretation Guide

### If Model B wins (expected):

**μ = 0.076:** Population-average success rate is 7.6%

**κ = 2.0:** Moderate concentration
- Low κ → Groups vary a lot
- High κ → Groups are similar
- Relationship: ICC ≈ 1/(1+κ) ≈ 0.33 (33% variance between groups)

**φ = 3.5:** Overdispersion parameter
- φ = 1 → Pure binomial (no overdispersion)
- φ = 3.5 → 250% more variance than binomial
- Confirms severe heterogeneity

**Shrinkage:**
- Group 1 (0/47) estimates ~2-4%, not 0%
- Group 8 (31/215) estimates ~11-13%, not 14.4%
- Large-sample groups (Group 4) barely shrink

### If Model C wins (surprising):

**π = 0.4:** 40% of groups in low-rate cluster, 60% in high-rate

**μ₁ = 0.05, μ₂ = 0.12:** Two distinct subpopulations

**Interpretation:** Groups are NOT continuously distributed!
- Need to explain: What makes low/high clusters?
- Action: Investigate group characteristics, collect covariates

---

## Next Steps After Fitting

1. **Check diagnostics** (Rhat, ESS, divergences)
2. **Compare models** (LOO-CV)
3. **Run posterior predictive checks**
4. **Sensitivity analysis** (vary priors)
5. **If Model C wins:** Investigate clustering
6. **If all fail:** Rethink likelihood

---

## Contact

This is Designer 1's independent work. For full details see:
- `experiment_plan.md` - Complete experiment plan
- `proposed_models.md` - Detailed model specifications
- `README.md` - Implementation guide

**Philosophy:** Falsification over confirmation. If data say mixture wins, I accept it.
