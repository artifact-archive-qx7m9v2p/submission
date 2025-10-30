# Implementation Roadmap: Designer 3 Robust Models

## Priority Order for Implementation

### Phase 1: Essential Baseline (30 minutes)
**Purpose:** Establish what we're being robust RELATIVE TO

1. **Normal Hierarchical Model** (10 min)
   - File: `baseline_normal_model.py`
   - Priors: mu ~ Normal(0, 20), tau ~ Half-Cauchy(0, 5)
   - 2000 iterations, target_accept=0.90
   - Save: LOO-CV score, posterior samples, diagnostics

2. **Posterior Predictive Checks** (10 min)
   - File: `baseline_checks.py`
   - Generate 1000 replicated datasets
   - Test statistics: min(y), max(y), SD(y), Q-statistic
   - Visual: y_rep distributions vs observed

3. **Baseline Documentation** (10 min)
   - File: `baseline_results.md`
   - Summarize: mu, tau, theta_i posteriors
   - Note: Shrinkage pattern, LOO-CV score
   - Flag: Any diagnostic issues (divergences, R-hat, ESS)

**Decision Point 1:** If baseline has serious issues (R-hat > 1.05, divergences > 5%), STOP and reconsider model structure before adding robustness.

---

### Phase 2: Core Robustness (45 minutes)
**Purpose:** Test primary robustness hypotheses

1. **Student-t Data Model (1A)** (20 min)
   - File: `model_1a_student_t.py`
   - Key addition: nu ~ Gamma(2, 0.1)
   - 2000 iterations, target_accept=0.95 (higher for nu sampling)
   - Monitor: nu convergence (may be slow)
   - Extract: posterior nu, compare LOO-CV to baseline

2. **Outlier Indicator Model (2A)** (25 min)
   - File: `model_2a_outlier_indicators.py`
   - Key additions: p_i ~ Bernoulli(pi), pi ~ Beta(1, 9), k ~ Gamma(4, 1)
   - 3000 iterations (discrete parameters slower)
   - Use pm.Marginalized for discrete p_i if possible
   - Extract: p_i posteriors for each school, posterior pi

**Decision Point 2:**
- If nu_posterior > 30 AND all p_i < 0.2: **STOP** - robustness not needed, document and move to Phase 4
- If nu_posterior < 10 OR any p_i > 0.7: **CONTINUE** - robustness matters, proceed to Phase 3

---

### Phase 3: Extended Robustness (60 minutes, CONDITIONAL)
**Purpose:** Only if Phase 2 found evidence for robustness needs

1. **Student-t Random Effects (1B)** (20 min)
   - File: `model_1b_student_t_effects.py`
   - Tests heavy-tailed theta_i (not y_i)
   - May reveal if heterogeneity is non-normal

2. **Double Student-t (1C)** (20 min)
   - File: `model_1c_double_student_t.py`
   - Maximum robustness
   - May be overparameterized with n=8

3. **Latent Class Model (2B)** (20 min)
   - File: `model_2b_latent_classes.py`
   - Only if bimodality suspected
   - Requires careful label switching handling

**Decision Point 3:** Compare all robust models via LOO-CV. Select best if improvement > 3 points, otherwise report range.

---

### Phase 4: Prior Sensitivity (45 minutes, ALWAYS DO)
**Purpose:** Quantify robustness to prior choices

1. **Setup Prior Grid** (5 min)
   - File: `prior_grid.py`
   - Define: 6 mu priors × 6 tau priors = 36 combinations
   - Use lightweight fits (1000 iterations each)

2. **Parallel Fitting** (30 min)
   - File: `fit_prior_grid.py`
   - Loop over all 36 combinations
   - Save key posteriors: mu_mean, mu_sd, tau_mean, tau_sd, LOO
   - Parallelize across 8 cores → ~10-15 min wall time

3. **Sensitivity Analysis** (10 min)
   - File: `analyze_sensitivity.py`
   - Compute Relative_Sensitivity for mu, tau
   - Identify priors with extreme influence
   - Generate sensitivity plots

**Decision Point 4:**
- If Relative_Sensitivity < 0.5 for both mu and tau: **ROBUST** - recommend any weakly informative prior
- If Relative_Sensitivity > 1.0 for either: **SENSITIVE** - report full range, recommend specific prior with justification

---

### Phase 5: Synthesis (30 minutes)
**Purpose:** Integrate findings and make recommendations

1. **Model Comparison Table** (10 min)
   - File: `model_comparison.csv`
   - Columns: Model, mu_post, tau_post, nu_post (if applicable), LOO, LOO_se
   - Highlight best model(s) by LOO ± SE

2. **Visualization Suite** (15 min)
   - File: `comparison_plots.py`
   - Plot 1: Posterior mu across models (forest plot style)
   - Plot 2: Posterior tau across models
   - Plot 3: Prior sensitivity (heatmap of mu_post by prior choice)
   - Plot 4: Shrinkage comparison (theta_i vs y_i for each model)

3. **Final Report** (5 min)
   - File: `robustness_report.md`
   - Section 1: Executive summary (robust or not?)
   - Section 2: Key findings per model class
   - Section 3: Recommendation for primary analysis
   - Section 4: Sensitivity analyses for supplement

---

## Computational Resource Planning

### Time Estimates (Wall Clock)

| Phase | Sequential | Parallel (8 cores) | Notes |
|-------|------------|-------------------|-------|
| Phase 1 | 30 min | 20 min | Baseline is fast |
| Phase 2 | 45 min | 30 min | Can parallelize 2 models |
| Phase 3 | 60 min | 25 min | Conditional, parallelize 3 models |
| Phase 4 | 45 min | 15 min | Highly parallelizable (36 models) |
| Phase 5 | 30 min | 30 min | Mostly sequential (analysis/viz) |
| **Total** | **3.5 hr** | **2 hr** | **With parallel computing** |

### Memory Requirements

- Per model: ~500 MB (2000 samples × 8 schools × multiple parameters)
- Peak usage (Phase 4): ~8 GB (fitting 8 models simultaneously)
- Storage: ~100 MB total (saved traces + plots)

### Computing Environment

```bash
# Recommended setup
conda create -n robust_models python=3.10
conda activate robust_models
pip install pymc arviz numpy pandas matplotlib seaborn

# Hardware assumptions
# - 8 CPU cores (for parallelization)
# - 16 GB RAM (for simultaneous fits)
# - ~30 min on modern laptop
```

---

## Quick Start Commands

### Run All Analyses (Automated)

```bash
cd /workspace/experiments/designer_3
python run_all.py --parallel --cores 8
```

### Run Specific Phases

```bash
# Phase 1: Baseline
python baseline_normal_model.py
python baseline_checks.py

# Phase 2: Core robustness
python model_1a_student_t.py
python model_2a_outlier_indicators.py

# Phase 4: Prior sensitivity
python fit_prior_grid.py --parallel --cores 8
python analyze_sensitivity.py

# Phase 5: Synthesis
python comparison_plots.py
```

### Interactive Exploration

```python
import pymc as pm
import arviz as az

# Load saved traces
trace_normal = az.from_netcdf('traces/baseline_normal.nc')
trace_student_t = az.from_netcdf('traces/model_1a_student_t.nc')

# Quick comparison
az.compare({'Normal': trace_normal, 'Student-t': trace_student_t})

# Visual comparison
az.plot_forest([trace_normal, trace_student_t],
               model_names=['Normal', 'Student-t'],
               var_names=['mu', 'tau'])
```

---

## File Organization

```
/workspace/experiments/designer_3/
├── proposed_models.md              # Main design document (THIS FILE)
├── model_comparison_matrix.md       # Quick reference tables
├── implementation_roadmap.md        # This implementation guide
│
├── code/
│   ├── baseline_normal_model.py
│   ├── baseline_checks.py
│   ├── model_1a_student_t.py
│   ├── model_1b_student_t_effects.py
│   ├── model_1c_double_student_t.py
│   ├── model_2a_outlier_indicators.py
│   ├── model_2b_latent_classes.py
│   ├── prior_grid.py
│   ├── fit_prior_grid.py
│   ├── analyze_sensitivity.py
│   ├── comparison_plots.py
│   └── run_all.py
│
├── traces/                         # Saved posterior samples
│   ├── baseline_normal.nc
│   ├── model_1a_student_t.nc
│   └── ...
│
├── results/
│   ├── baseline_results.md
│   ├── model_comparison.csv
│   ├── prior_sensitivity_results.csv
│   └── robustness_report.md
│
└── figures/
    ├── baseline_posterior.png
    ├── comparison_forest.png
    ├── prior_sensitivity_heatmap.png
    └── shrinkage_comparison.png
```

---

## Critical Checkpoints and Stop Rules

### Checkpoint 1: After Baseline (Phase 1)
**Question:** Is the baseline model working properly?

**Green light (continue):**
- All R-hat < 1.01
- ESS > 400 for all parameters
- Divergences < 1%
- Posterior predictive checks look reasonable

**Yellow light (fix before continuing):**
- Some R-hat between 1.01-1.05
- ESS between 100-400
- Divergences 1-5%
- **Action:** Increase iterations, try non-centered parameterization

**Red light (stop and reconsider):**
- Any R-hat > 1.05
- ESS < 100 for any parameter
- Divergences > 5%
- Posterior predictive checks fail badly
- **Action:** Reconsider model structure, check data

### Checkpoint 2: After Core Robustness (Phase 2)
**Question:** Do we need robust models?

**Robustness NOT needed (skip Phase 3):**
- nu_posterior > 30 (Student-t → Normal)
- All p_i < 0.2 (no outliers)
- LOO difference < 2 (no improvement)
- **Action:** Document, proceed to Phase 4 (sensitivity)

**Robustness MAY BE needed (continue to Phase 3):**
- nu_posterior between 10-30
- Some p_i between 0.2-0.7
- LOO difference 2-5
- **Action:** Fit additional robust models

**Robustness CLEARLY needed (prioritize Phase 3):**
- nu_posterior < 10
- Any p_i > 0.7
- LOO improvement > 5
- **Action:** Fit all robust variants, investigate cause

### Checkpoint 3: After Prior Sensitivity (Phase 4)
**Question:** Are results robust to prior choices?

**Robust (good to go):**
- Relative_Sensitivity < 0.5
- All priors yield similar conclusions
- **Action:** Recommend weakly informative prior, report

**Moderately sensitive (acknowledge):**
- Relative_Sensitivity 0.5-1.0
- Priors matter but don't change conclusions
- **Action:** Report sensitivity, justify prior choice

**Highly sensitive (caution):**
- Relative_Sensitivity > 1.0
- Different priors → different conclusions
- **Action:** Report full range, recommend more data or informative prior

---

## Expected Outcomes by Scenario

### Scenario A: EDA is correct, robustness not needed (MOST LIKELY)

**Findings:**
- Normal model works fine (R-hat < 1.01, good PPC)
- nu_posterior = 28 [18, 42] → effectively normal
- All p_i < 0.15 → no outliers
- LOO differences < 1 → no improvement
- Relative_Sensitivity = 0.3 → low prior dependence

**Recommendation:**
- Primary analysis: Normal hierarchical model
- Robustness section: "We checked Student-t and outlier models; both converged to normal assumptions"
- Prior recommendation: Half-Cauchy(0, 5) for tau, Normal(0, 20) for mu

**Key insight:** Robustness checks VALIDATE normal assumptions (negative result is positive evidence)

### Scenario B: Mild tail heaviness, borderline robustness

**Findings:**
- Normal model works but PPC shows slight tail underestimation
- nu_posterior = 12 [6, 22] → heavier than normal but uncertain
- p_1 = 0.35 for School 1 → possibly an outlier
- LOO improvement = 2.5 ± 1.8 → marginal benefit
- Relative_Sensitivity = 0.6 → moderate prior dependence

**Recommendation:**
- Primary analysis: Student-t model (conservative)
- Sensitivity analysis: Report both Normal and Student-t
- Prior recommendation: Informative prior on nu (e.g., Gamma(4, 0.2) to center at nu=20)

**Key insight:** Model choice matters somewhat, report range to acknowledge uncertainty

### Scenario C: Clear robustness needed, EDA missed something

**Findings:**
- Normal model has issues (divergences, poor PPC)
- nu_posterior = 4 [2, 7] → very heavy tails
- p_1 = 0.92 → School 1 clearly an outlier
- LOO improvement = 8.2 ± 3.1 → substantial benefit
- Relative_Sensitivity = 1.2 → high prior dependence

**Recommendation:**
- Primary analysis: Robust model (Student-t or outlier)
- Investigation: WHY is robustness needed? Data error? Different process?
- Prior recommendation: Domain-informed prior if possible
- CAUTION: Consider removing outlier and refitting

**Key insight:** EDA was misleading (failed to detect issue), robustness models reveal it

---

## Communication Strategy

### For Technical Audience (Other Designers)

**What to emphasize:**
- Mathematical specifications (see proposed_models.md)
- Falsification criteria and decision thresholds
- Computational considerations and diagnostics
- Quantitative comparisons (LOO-CV, sensitivity metrics)

**Key tables:**
- Model comparison matrix with LOO ± SE
- Prior sensitivity results (Relative_Sensitivity)
- Parameter posteriors across models

### For Scientific Audience (Domain Experts)

**What to emphasize:**
- Are conclusions robust to distributional assumptions?
- What does "robustness" mean scientifically? (heavy tails = outliers)
- Practical implications (wider intervals, different predictions)

**Key visualizations:**
- Forest plots comparing posteriors across models
- Shrinkage plots showing impact of robustness
- Prior sensitivity plots showing range of plausible conclusions

### For Stakeholders (Decision Makers)

**What to emphasize:**
- Bottom line: Are pooled estimates reliable?
- Uncertainty quantification: How confident are we?
- Sensitivity: Do conclusions change with reasonable assumptions?

**Key message:**
- If robust: "Results hold under multiple distributional assumptions"
- If sensitive: "Conclusions depend on assumptions; we recommend [X] based on [Y evidence]"

---

## Integration with Other Designers

### Expected Common Ground
- All designers likely fit hierarchical models
- All should do some prior sensitivity
- Conclusions about mu and tau should be similar (EDA is clear)

### Unique Contribution (Designer 3)
- Systematic robustness framework
- Heavy-tailed alternatives (Student-t)
- Outlier detection models
- Quantitative sensitivity metrics

### Synthesis Approach

**If we all agree:**
- Designer 3 adds robustness validation to consensus
- Report: "Multiple independent analyses converged, with robustness checks confirming [...]"

**If we disagree:**
- Designer 3 investigates if disagreement is due to distributional assumptions
- Test: Does Student-t/outlier model reconcile different conclusions?
- Report: "Analyses differed due to [distributional assumptions / prior choices / model structure]"

**Value of disagreement:**
- Reveals genuine uncertainty or model sensitivity
- More honest than false consensus
- Guides where more data or domain knowledge would help

---

## Final Checklist Before Reporting

- [ ] All models have R-hat < 1.01
- [ ] All models have ESS > 400
- [ ] Divergences < 1% (or < 5% with explanation)
- [ ] Posterior predictive checks conducted and passed
- [ ] LOO-CV computed for all models
- [ ] Prior sensitivity analysis completed
- [ ] Key visualizations generated
- [ ] Falsification criteria checked (did any trigger?)
- [ ] Comparison with baseline documented
- [ ] Limitations acknowledged (n=8, boundary estimation, etc.)
- [ ] Code and traces saved for reproducibility
- [ ] Results files written (MD format)
- [ ] Integration with other designers considered

---

## Known Limitations and Caveats

1. **Small sample size (n=8):**
   - Low power to detect tail behavior
   - Robustness parameters (nu, k) weakly identified
   - Wide credible intervals expected

2. **Boundary estimation (tau ≈ 0):**
   - Prior on tau matters more at boundary
   - Some priors may constrain to boundary artificially
   - Report full posterior, not just point estimate

3. **Multiple testing:**
   - Testing many models increases false discovery risk
   - Use LOO-CV for formal comparison (adjusts for complexity)
   - Don't cherry-pick "best" model without justification

4. **Computational approximations:**
   - MCMC is approximate (finite samples)
   - Discrete parameters (outlier indicators) challenging
   - Some models may need variational inference

5. **Interpretation challenges:**
   - "Robust" doesn't mean "better" - means less sensitive
   - Robustness adds complexity - trade-off with interpretability
   - Negative results (robustness not needed) are valid conclusions

---

**STATUS: DESIGN COMPLETE - READY FOR IMPLEMENTATION**

**Next Steps:**
1. Implement Phase 1 (baseline)
2. Check Checkpoint 1
3. Proceed conditionally based on decision rules
4. Synthesize with other designers' findings
5. Report integrated conclusions

**Estimated Total Time:** 2-3 hours (parallelized)

**Key Philosophy:** Robustness analysis is about learning whether assumptions matter, not about always preferring robust models. Simplicity is a virtue when it's justified.

---

**Last Updated:** 2025-10-28
**Designer:** Designer 3 (Robust Models Specialist)
