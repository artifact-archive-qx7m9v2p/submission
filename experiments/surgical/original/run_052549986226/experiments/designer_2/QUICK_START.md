# Quick Start Guide: Hierarchical Binomial Models

## TL;DR - Get Running in 5 Minutes

### 1. Install Dependencies

```bash
pip install cmdstanpy pandas numpy matplotlib seaborn arviz
```

### 2. Fit the Recommended Model (M2)

```bash
cd /workspace/experiments/designer_2
python fit_models.py --model 2
```

**Expected runtime:** 2-5 minutes

### 3. Check If It Worked

Look for in the output:
- ✓ "ALL DIAGNOSTICS PASSED"
- Divergences: 0 or very low (<1%)
- Max Rhat < 1.01
- Min ESS > 400

### 4. Compare Models (Optional but Recommended)

```bash
# First fit all models
python fit_models.py --model all  # ~15 minutes

# Then compare
python model_comparison.py  # ~2 minutes
```

### 5. Review Results

Check these files:
- `visualizations/comparison_report.txt` - Summary in plain text
- `visualizations/*.png` - All comparison plots
- `results/*_summary.csv` - Detailed posterior statistics

## Key Questions Answered

### Q: Which model should I use?
**A:** Start with Model 2 (non-centered). It's computationally efficient and handles the data structure well.

### Q: What if I get divergences?
**A:** Try:
```bash
python fit_models.py --model 2 --adapt_delta 0.95
```
If still >1% divergences, something is wrong - read diagnostics carefully.

### Q: How is Group 1 (0/47) handled?
**A:** Through hierarchical shrinkage. No ad-hoc correction. Expected posterior: ~1-3%.

### Q: What's the difference between M2 and M3?
**A:** M3 uses heavy-tailed priors (Student-t) to handle outliers (especially Group 8). If M3's posterior ν > 30, M2 was sufficient.

### Q: What does success look like?
**A:**
- Reproduces φ ≈ 3.5-5.1 (overdispersion)
- Group 1 posterior is 1-3% (not 0%)
- σ ≈ 0.8-1.2 (consistent with ICC = 0.73)
- All diagnostics pass

### Q: What if it fails?
**A:**
- If computational failure → Try M2 with adapt_delta=0.99
- If statistical failure → Switch to beta-binomial (Designer 1)
- If mixture suspected → Try finite mixture models

## File Guide

**Need to modify models?** Edit `model*.stan` files
**Need to change priors?** Edit `model*.stan` files (lines ~20-30)
**Need different MCMC settings?** Use command line args:
```bash
python fit_models.py --model 2 --chains 4 --iter 3000 --adapt_delta 0.95
```

**Want to customize plots?** Edit `model_comparison.py`

## One-Liner Diagnostics Check

After fitting, run:
```bash
grep "PASS\|FAIL" results/*_diagnostics.json
```

Should see all "PASS" for successful fit.

## Expected Posteriors (Sanity Check)

After running, check `results/*_summary.csv`:

```
Parameter          Expected Range      Why?
-------------------------------------------------
mu                 -2.8 to -2.2        logit(7.6%) ≈ -2.5
sigma              0.6 to 1.2          ICC = 0.73 implies σ ≈ 0.9
p_posterior[1]     0.01 to 0.05        Group 1 shrinkage from 0%
p_posterior[8]     0.10 to 0.14        Group 8 shrinkage from 14.4%
phi_posterior      3.0 to 6.0          Match observed overdispersion
```

If any values are way outside these ranges, investigate carefully.

## Troubleshooting

**"ModuleNotFoundError: No module named 'cmdstanpy'"**
→ Run: `pip install cmdstanpy`

**"CmdStan not found"**
→ Run: `python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`

**"Divergences > 5%"**
→ Model 1 expected, use Model 2. If M2 also fails, deeper issue.

**"ESS < 100"**
→ Run longer: `python fit_models.py --model 2 --iter 4000`

**"Pareto k > 0.7 for multiple groups"**
→ Consider mixture model instead of hierarchical

**Results seem weird**
→ Check `proposed_models.md` section "Expected Posterior Behavior"

## Key Files to Read

1. **First:** `README.md` (comprehensive overview)
2. **For theory:** `proposed_models.md` (full model design)
3. **For results:** `visualizations/comparison_report.txt` (after running)

## What Makes This Approach Good?

✓ **Principled shrinkage** - No ad-hoc corrections
✓ **Handles zero counts** - Through hierarchy, not hacks
✓ **Quantifies uncertainty** - Full posterior distributions
✓ **Computationally efficient** - Non-centered parameterization
✓ **Interpretable** - Clear parameter meanings (μ, σ, α_i)
✓ **Flexible** - Can add covariates later

## What Could Go Wrong?

✗ **Beta-binomial might be simpler** - 2 params vs 12
✗ **Mixture might be better** - If discrete subgroups exist
✗ **Overdispersion might be stronger** - Than normal random effects can capture

**Solution:** Compare via LOO-CV with other designers!

## Critical Success Criteria

Before declaring victory, verify:

1. ✓ Rhat < 1.01 for ALL parameters
2. ✓ ESS > 400 for ALL parameters
3. ✓ Divergences < 1%
4. ✓ Posterior φ ≈ 3.5-5.1 (reproduces observed overdispersion)
5. ✓ Group 1 posterior is 1-3% (not stuck at 0%)
6. ✓ σ ≈ 0.8-1.2 (consistent with ICC = 0.73)

If any fail, investigate before using results!

## Time Budget

- **Minimum viable:** 5 minutes (fit M2, check diagnostics)
- **Recommended:** 30 minutes (fit all, compare, review plots)
- **Thorough:** 2 hours (sensitivity analyses, custom visualizations)

## Final Recommendation

**For this dataset:** Model 2 (non-centered) should work well. Fit it first, check diagnostics, then decide if M3 (robust) or M1 (centered) are needed for comparison.

**The goal:** Reliable inference on group-specific success rates with appropriate uncertainty quantification. Hierarchical shrinkage is the principled way to handle extreme observations (Group 1's zero, Group 8's outlier).

**Success = Finding truth, not completing tasks.** If this model doesn't work well, that's valuable information! Document why and pivot to better alternatives.
