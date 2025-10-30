# Quick Start Guide: Regression Models with Covariates

**For the impatient data analyst who wants to fit models NOW.**

---

## 30-Second Summary

You have 12 groups with binomial data. EDA showed heterogeneity but no significant correlation with sample size. I designed 3 regression models to formally test if covariates (sample size, group ordering) explain the variation. If they don't (ΔLOO < 2), that's an important finding—use random effects only.

---

## Installation (5 minutes)

```bash
# Install Python packages
pip install cmdstanpy arviz numpy pandas matplotlib seaborn

# Install CmdStan (backend for Stan)
python -m cmdstanpy.install_cmdstan
```

---

## Running the Analysis (10 minutes)

### Step 1: Prepare Your Data

CSV file with 3 columns:
```
group_id,n_trials,r_successes
1,47,5
2,113,15
3,210,9
...
```

### Step 2: Run the Pipeline

```bash
cd /workspace/experiments/designer_3

python fit_models.py \
    --data /workspace/data/binomial_data.csv \
    --output ./results
```

### Step 3: Check Output

The script will:
1. Compile Stan models (~30 seconds)
2. Fit Model 1 (sample size covariate) (~30 seconds)
3. Fit Model 2 (quadratic group effect) (~30 seconds)
4. Ask if you want Model 3 (random slopes) - say "N" for now
5. Compare models via LOO-CV
6. Generate plots and results

**Total time: ~5 minutes**

---

## Interpreting Results

### Console Output

Look for this section:

```
MODEL COMPARISON (LOO-CV)
============================================================

         rank  elpd_loo   p_loo  elpd_diff     se
model1      0    -45.2    5.3       0.0      0.0
baseline    1    -47.8    3.2       2.6      1.8
model2      2    -48.1    6.1       2.9      2.3

ΔLOO (baseline vs model1): 2.6 ± 1.8
  → Weak evidence (2 < ΔLOO < 4)
  → Slight preference for model1
```

### Decision Rules

**If ΔLOO < 2:**
→ Covariates don't help
→ Use random effects only (Designer 1)
→ Done!

**If 2 < ΔLOO < 4:**
→ Weak evidence for covariates
→ Report with large uncertainty
→ Consider sensitivity analysis

**If ΔLOO > 4:**
→ Covariates clearly help
→ Report coefficients
→ Investigate scientific meaning

---

## What Each Model Tests

### Model 1: Sample Size Effect

**Question:** Do larger studies have different success rates?

**Check:**
- Is beta_1 credibly non-zero? (95% CI excludes 0)
- Is R² > 0.15? (sample size explains > 15% of variance)

**If YES:** Sample size is confounded with something
**If NO:** Heterogeneity not explained by sample size

### Model 2: Sequential Structure

**Question:** Is there a pattern in group ordering?

**Check:**
- Are beta_1 or beta_2 credibly non-zero?
- Does the curve match visual inspection?

**If YES:** Group ordering is NOT arbitrary (investigate why!)
**If NO:** Groups are exchangeable (ordering doesn't matter)

### Model 3: Varying Slopes (Optional)

**Question:** Does the size-response vary across groups?

**Check:**
- Is tau_gamma > 0.1? (slopes vary meaningfully)
- Does it improve LOO vs Model 1?

**If YES:** Heterogeneous responses (consider mixture model)
**If NO:** Stick with Model 1 (simpler)

---

## Output Files

```
results/
├── model1_size_covariate_predictions.png    # Observed vs predicted
├── model1_size_covariate_coefficients.png   # Posterior distributions
├── model1_size_covariate_results.json       # Detailed results
├── model2_quadratic_group_predictions.png
├── model2_quadratic_group_coefficients.png
├── model2_quadratic_group_results.json
└── model_comparison.csv                     # LOO comparison table
```

---

## Red Flags

**Stop and investigate if:**

- [ ] Rhat > 1.01 (poor convergence)
- [ ] Divergences > 1% (model misspecification)
- [ ] Pareto-k > 0.7 for many observations (influential outliers)
- [ ] Predictions wildly off from data (check plots)
- [ ] Coefficients are extreme (|beta| > 2)

**What to do:**
1. Check data (any errors?)
2. Increase adapt_delta to 0.99
3. Try tighter priors
4. Consider different model class

---

## Common Questions

### Q: Model 1 failed to converge. What now?

A: Try:
```python
# In fit_models.py, modify fit_model() call:
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_sampling=2000,        # More iterations
    iter_warmup=2000,
    adapt_delta=0.99,          # More careful sampling
    max_treedepth=12           # Allow deeper trees
)
```

### Q: All models have ΔLOO < 2. What does this mean?

A: **Covariates are uninformative.** This is a valid finding!
→ Use simple random effects model (Designer 1)
→ Focus on partial pooling, not covariate effects
→ Report that sample size and group_id don't explain heterogeneity

### Q: Can I add more covariates?

A: Yes! Modify Model 1:
```stan
// In model1_size_covariate.stan, change:
mu = beta_0 + beta_1 * log_n_centered + beta_2 * covariate2 + ...
```

But be careful with multicollinearity (check VIF < 5).

### Q: Should I always run Model 3?

A: No. Model 3 is complex and may overfit with J=12.
→ Run it only if:
  1. Models 1-2 show promise (ΔLOO > 2)
  2. You suspect heterogeneous slopes
  3. You have time for 5-minute fitting

### Q: What if results contradict EDA?

A: **This is a red flag!**
- EDA found r = -0.34 (negative), but model finds beta_1 > 0 (positive)
- Double-check data processing (centering, scaling)
- Verify model specification
- Consider Simpson's paradox (lurking variable)

---

## Minimal Example

```python
from fit_models import RegressionModelFitter

# Initialize
fitter = RegressionModelFitter(
    data_path='data.csv',
    output_dir='results'
)

# Fit Model 1 only
data1 = fitter.prepare_data_model1()
fit1 = fitter.fit_model('model1_size_covariate', data1)
loo1 = fitter.compute_loo('model1_size_covariate')

# Plot
fitter.plot_predictions('model1_size_covariate')
fitter.plot_coefficients('model1_size_covariate')

# Extract key results
results = fitter.extract_results('model1_size_covariate')
print(results['loo'])
```

---

## Expected Output

### Successful Run

```
Data validated: 12 groups
  n_trials: 47 to 810
  r_successes: 3 to 34
  Success rates: 0.004 to 0.723

Compiling model1_size_covariate...
  Compiled successfully

Fitting model1_size_covariate...
  Chains: 4, Iterations: 1000 warmup + 1000 sampling

  Diagnostics for model1_size_covariate:
    Max Rhat: 1.002 [GOOD]
    Min ESS: 1523 [GOOD]
    Divergences: 0 (0.00%) [GOOD]

Computing LOO for model1_size_covariate...
  ELPD LOO: -45.23 ± 3.12
  p_loo: 5.31

Saved prediction plot: results/model1_size_covariate_predictions.png
Saved coefficient plot: results/model1_size_covariate_coefficients.png
Saved results: results/model1_size_covariate_results.json
```

---

## When to Stop and Rethink

**Abandon regression approach if:**

1. **All models fail diagnostics** (Rhat > 1.05, persistent divergences)
   → Problem: Model class is wrong
   → Solution: Try mixture model (Designer 2) or robust model

2. **All ΔLOO < 2 with narrow SE** (e.g., ± 1.5)
   → Problem: Covariates genuinely uninformative
   → Solution: Use random effects only (Designer 1)

3. **Predictions are terrible** (large residuals, systematic bias)
   → Problem: Wrong likelihood or missing covariates
   → Solution: Check for overdispersion (beta-binomial)

4. **Coefficients are implausible** (e.g., doubling n → 50% rate change)
   → Problem: Overfitting or misspecified priors
   → Solution: Tighter priors or simpler model

---

## Next Steps After Fitting

### If Model 1 Wins:

1. **Report:** beta_1, R², effect size
2. **Interpret:** Why do large studies differ?
3. **Check:** Is this confounding? (study design, population)
4. **Sensitivity:** Try different priors

### If Model 2 Wins:

1. **Report:** beta_1, beta_2, peak location
2. **Interpret:** What does group_id represent?
3. **Check:** Is this temporal? Spatial? Ordinal?
4. **Visualize:** Plot fitted curve

### If No Model Wins:

1. **Report:** "Covariates do not explain heterogeneity"
2. **Interpret:** True random differences OR unmeasured covariates
3. **Switch:** Use Designer 1's random effects model
4. **Consider:** Mixture model (Designer 2) for clusters

---

## Final Checklist

Before reporting results:

- [ ] All Rhat < 1.01
- [ ] All ESS > 400
- [ ] Divergences < 1%
- [ ] Pareto-k < 0.7 (or justified)
- [ ] Posterior predictive checks pass
- [ ] Coefficients are plausible
- [ ] Results align with EDA (or explained)
- [ ] Interpretation is clear
- [ ] Uncertainty is quantified

---

## Help and Debugging

**Problem:** "ModuleNotFoundError: No module named 'cmdstanpy'"
→ Run: `pip install cmdstanpy`

**Problem:** "CmdStan not found"
→ Run: `python -m cmdstanpy.install_cmdstan`

**Problem:** "FileNotFoundError: model1_size_covariate.stan"
→ Check: Are .stan files in the same directory as fit_models.py?
→ Use: `--models-dir` argument to specify location

**Problem:** Model takes > 10 minutes
→ Normal for Model 3, check: CPU usage, memory
→ Try: Reduce iterations or simplify model

**Problem:** "ValueError: All Pareto k estimates above 0.7"
→ This is OK if only 1-2 groups (they're known outliers)
→ Concerning if > 50% of groups
→ Solution: May need robust likelihood (t-distribution)

---

## Contact

This quick start guide accompanies the full model documentation in `proposed_models.md`.

For detailed theory, falsification criteria, and computational notes, see:
- `proposed_models.md` - Complete model specifications
- `README.md` - Detailed usage guide

**Created by:** Model Designer 3 (Regression/Covariate Specialist)
**Date:** 2025-10-30
