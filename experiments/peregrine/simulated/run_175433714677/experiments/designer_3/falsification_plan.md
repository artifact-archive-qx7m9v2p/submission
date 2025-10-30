# Falsification Plan: How to Break These Models

**Purpose**: Document explicit criteria for rejecting models and pivoting strategies.

**Philosophy**: Good science is about being WRONG quickly and learning from failure.

---

## Critical Mindset

**These models are probably wrong in ways I can't anticipate.**

My job is to:
1. Find out HOW they're wrong
2. Find out WHEN they're wrong
3. Decide WHAT to do when they fail

**Success = discovering a model is wrong and pivoting intelligently**

---

## Model 1: Hierarchical Gamma-Poisson

### Explicit Rejection Criteria

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| **Convergence** | Rhat > 1.01 for any parameter | Reparameterize or abandon |
| **Random effects structure** | All λ[i] within 20% of μ[i] | No heterogeneity; use standard NegBin |
| **Time correlation** | \|cor(λ[i], year[i])\| > 0.4 | Indicates time-varying dispersion; pivot to Model 4 |
| **LOO-CV vs NegBin** | ΔELPD < -4 (worse than NegBin) | Hierarchy adds no value; abandon |
| **Var/Mean recovery** | Posterior median outside [50, 90] | Model misspecified; check priors or likelihood |
| **Computational cost** | >10 min runtime with divergences | Not practical; use marginalized NegBin |

### Stress Tests

**Test 1: Posterior-Prior Overlap**
```python
# If posterior and prior overlap >70%, data isn't informative
from scipy.stats import ks_2samp

phi_prior_samples = np.random.gamma(2, 1/0.1, size=10000)
phi_posterior = fit.stan_variable('phi')

ks_stat, ks_pval = ks_2samp(phi_prior_samples, phi_posterior)
print(f"Prior-Posterior overlap: KS stat = {ks_stat:.3f}")

if ks_stat < 0.3:
    print("WARNING: High prior-posterior overlap. Data is not informative!")
```

**Test 2: Random Effects Shrinkage**
```python
# Strong shrinkage suggests hierarchy is unnecessary
lambda_post = fit.stan_variable('lambda').mean(axis=0)
mu_post = fit.stan_variable('mu').mean(axis=0)

shrinkage = np.abs(lambda_post - mu_post) / mu_post
print(f"Mean shrinkage: {shrinkage.mean():.2%}")

if shrinkage.mean() < 0.05:
    print("WARNING: <5% shrinkage. Random effects are negligible!")
```

**Test 3: Heterogeneity Pattern**
```python
# If random effects show clear time trend, model is misspecified
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

lambda_post = fit.stan_variable('lambda').mean(axis=0)
corr, pval = pearsonr(data['year'], lambda_post)

plt.scatter(data['year'], lambda_post)
plt.xlabel('Year')
plt.ylabel('Posterior λ[i] (mean)')
plt.title(f'Random Effects vs. Time (r={corr:.3f}, p={pval:.3f})')
plt.savefig('experiments/designer_3/lambda_vs_time.png')

if pval < 0.05 and abs(corr) > 0.3:
    print("FALSIFICATION: Random effects correlate with time!")
    print("Action: Pivot to time-varying dispersion model")
```

### Escape Routes

**If model fails**:

1. **High shrinkage** → Use standard NegBin (hierarchy unnecessary)
2. **Time-correlated random effects** → Fit time-varying dispersion:
   ```
   log(φ[i]) = γ₀ + γ₁ × year[i]
   ```
3. **Non-convergence** → Use non-centered parameterization:
   ```stan
   parameters {
     vector[N] lambda_raw;
   }
   transformed parameters {
     vector[N] lambda = mu .* exp(lambda_raw / sqrt(phi));
   }
   model {
     lambda_raw ~ std_normal();
   }
   ```
4. **All else fails** → Standard NegBin GLM with fixed dispersion

---

## Model 2: Student-t Regression on Log-Counts

### Explicit Rejection Criteria

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| **Tail parameter** | Posterior nu > 40 | Use Normal regression instead |
| **Back-transformation** | KS test p < 0.05 for C_rep vs C | Count structure matters; use count models |
| **Var/Mean recovery** | Posterior median outside [50, 90] | Model can't handle overdispersion on count scale |
| **Residual patterns** | Heteroscedasticity evident in plots | Extend to time-varying σ |
| **LOO-CV vs NegBin** | ΔELPD < -6 | Count models are fundamentally better |

### Stress Tests

**Test 1: Normality Check**
```python
nu_post = fit.stan_variable('nu')
nu_mean = nu_post.mean()
nu_95ci = np.percentile(nu_post, [2.5, 97.5])

print(f"Posterior ν: {nu_mean:.1f} (95% CI: [{nu_95ci[0]:.1f}, {nu_95ci[1]:.1f}])")

if nu_95ci[0] > 30:
    print("FALSIFICATION: ν > 30 with high confidence")
    print("Data is approximately Normal; Student-t is overkill")
    print("Action: Switch to Normal regression or reconsider approach")
```

**Test 2: Back-Transformation Validity**
```python
from scipy.stats import ks_2samp, anderson_ksamp

C_obs = np.array(data['C'])
C_rep = fit.stan_variable('C_rep').flatten()

# KS test
ks_stat, ks_pval = ks_2samp(C_obs, C_rep)

# Anderson-Darling test (more sensitive to tails)
ad_result = anderson_ksamp([C_obs, C_rep])

print(f"Distribution match:")
print(f"  KS test: D={ks_stat:.3f}, p={ks_pval:.3f}")
print(f"  AD test: statistic={ad_result.statistic:.3f}, p={ad_result.significance_level:.3f}")

if ks_pval < 0.05:
    print("FALSIFICATION: Back-transformed predictions don't match observed distribution")
    print("Action: Log-scale modeling inappropriate; use count models")
```

**Test 3: Residual Heteroscedasticity**
```python
from scipy.stats import levene

y_rep = fit.stan_variable('y_rep')
residuals = y[:, None] - y_rep.T
residual_std = residuals.std(axis=1)

# Split by time period
early_idx = np.array(data['year']) < 0
late_idx = np.array(data['year']) >= 0

stat, pval = levene(residual_std[early_idx], residual_std[late_idx])

print(f"Residual heteroscedasticity test: F={stat:.3f}, p={pval:.3f}")

if pval < 0.05:
    print("FALSIFICATION: Residuals show heteroscedasticity")
    print("Action: Extend to σ[i] = exp(γ₀ + γ₁×year[i])")
```

**Test 4: Count Discreteness Matters**
```python
# If predicted probabilities at integer values differ substantially from
# continuous predictions, discreteness matters

y_pred = fit.stan_variable('mu').mean(axis=0)
C_pred_continuous = np.exp(y_pred)
C_pred_discrete = fit.stan_variable('C_rep').mean(axis=0)

discreteness_error = np.abs(C_pred_continuous - C_pred_discrete) / C_pred_discrete

print(f"Discretization error: {discreteness_error.mean():.1%} ± {discreteness_error.std():.1%}")

if discreteness_error.mean() > 0.15:
    print("FALSIFICATION: >15% error from treating counts as continuous")
    print("Action: Use proper count models (NegBin, Poisson, etc.)")
```

### Escape Routes

**If model fails**:

1. **ν > 40** → Use Normal regression (Student-t unnecessary)
2. **Back-transformation fails** → Use count-based models (NegBin)
3. **Heteroscedastic residuals** → Extend to time-varying variance:
   ```stan
   transformed parameters {
     vector[N] sigma_i = exp(gamma_0 + gamma_1 * year);
   }
   model {
     y ~ student_t(nu, mu, sigma_i);
   }
   ```
4. **High discreteness error** → Abandon continuous approach, use count models
5. **All else fails** → Return to standard NegBin GLM

---

## Model 3: COM-Poisson

### Explicit Rejection Criteria

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| **Convergence** | Divergences >5% or Rhat > 1.01 | Computational issues; abandon |
| **Runtime** | >60 min | Not practical; abandon |
| **Dispersion parameter** | Posterior ν ≈ 1 (within [0.9, 1.1]) | Data is Poisson; contradicts EDA |
| **Variance-mean match** | Same as NegBin (difference <5%) | COM-Poisson reduces to NegBin; no advantage |
| **LOO-CV vs NegBin** | ΔELPD < 0 | Complexity not justified |

### Stress Tests

**Test 1: Poisson Test**
```python
nu_post = fit.stan_variable('nu')
prob_poisson = np.mean((nu_post > 0.9) & (nu_post < 1.1))

print(f"Probability ν ∈ [0.9, 1.1]: {prob_poisson:.1%}")

if prob_poisson > 0.7:
    print("FALSIFICATION: Data is consistent with Poisson (ν ≈ 1)")
    print("This contradicts EDA finding of severe overdispersion!")
    print("Action: Check model implementation or data preprocessing")
```

**Test 2: NegBin Equivalence**
```python
# Simulate variance-mean relationship from COM-Poisson
# Compare to NegBin's V = μ + μ²/φ

lambda_post = fit.stan_variable('lambda').mean(axis=0)
nu_mean = fit.stan_variable('nu').mean()

# Theoretical variance-mean for COM-Poisson (approximate)
# (requires numerical computation of moments)

# Compare to NegBin
phi_negbin = 1.5  # from EDA
mu_range = np.linspace(20, 270, 50)
var_negbin = mu_range + mu_range**2 / phi_negbin

# Plot comparison
import matplotlib.pyplot as plt
plt.plot(mu_range, var_negbin, label='NegBin', linewidth=2)
# Add COM-Poisson curve (if computable)
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.legend()
plt.savefig('experiments/designer_3/variance_mean_curves.png')

# If curves nearly overlap, COM-Poisson adds no value
```

**Test 3: Computational Efficiency**
```python
import time

start_time = time.time()
# ... fit COM-Poisson ...
runtime = time.time() - start_time

print(f"Runtime: {runtime/60:.1f} minutes")

if runtime > 3600:  # 1 hour
    print("FALSIFICATION: Model is computationally impractical")
    print("Action: Abandon COM-Poisson; use NegBin or simpler alternatives")
```

### Escape Routes

**If model fails**:

1. **ν ≈ 1** → Use Poisson (but this contradicts EDA; investigate)
2. **ν matches NegBin** → Use NegBin (simpler and faster)
3. **Computational failure** → Try PyMC implementation or abandon
4. **All else fails** → Stick with standard NegBin GLM

---

## Cross-Model Comparisons: When to Abandon ALL Models

### Scenario 1: All Models Fail Var/Mean PPC

**Symptom**: None of the three models recover Var/Mean ≈ 70

**Diagnosis**: Homogeneous dispersion assumption is wrong

**Action**: Pivot to **time-varying dispersion models**
```
Model 4: Heteroscedastic NegBin
C[i] ~ NegBin(μ[i], φ[i])
log(μ[i]) = β₀ + β₁ × year[i]
log(φ[i]) = γ₀ + γ₁ × year[i]
```

### Scenario 2: All Models Give Similar LOO-CV

**Symptom**: ΔELPD < 2×SE across all models

**Diagnosis**: Distributional choice doesn't matter with n=40

**Implication**: Focus on **structural form** (linear vs. quadratic vs. piecewise) rather than likelihood family

**Action**:
- Declare "distributional choice is not identifiable"
- Focus on Model 1 vs. Model 2 from EDA report (functional form)
- Use simplest likelihood (NegBin) for that comparison

### Scenario 3: Parameter Estimates Wildly Inconsistent

**Symptom**: β₁ (growth rate) differs by >50% across models

**Example**: Model 1 gives β₁ ≈ 0.5, Model 2 gives β₁ ≈ 1.2

**Diagnosis**: Data doesn't strongly constrain growth rate

**Action**:
- Report high uncertainty honestly
- Use stronger informative priors (if domain knowledge available)
- Collect more data (if possible)
- Focus on robust predictions rather than precise parameters

### Scenario 4: All Models Have High Prior-Posterior Overlap

**Symptom**: Posteriors look similar to priors for key parameters

**Diagnosis**: n=40 is not enough to overcome prior information

**Action**:
- Use stronger priors based on domain knowledge
- Report "data is consistent with priors but doesn't strongly update beliefs"
- Consider Bayesian power analysis for future data collection

### Scenario 5: Computational Failures Across Multiple Models

**Symptom**: Divergences, non-convergence, or extreme runtimes for >1 model

**Diagnosis**: Data or model structure has pathologies

**Actions**:
1. **Check data quality**:
   - Are there hidden outliers?
   - Is standardization appropriate?
   - Are counts recorded correctly?

2. **Check model specification**:
   - Are priors reasonable?
   - Is link function appropriate?
   - Are there parameter identifiability issues?

3. **Simplify ruthlessly**:
   - Remove random effects
   - Use simpler functional forms
   - Consider non-Bayesian baseline (GLM with MLE) to check if data itself is problematic

---

## Master Decision Tree

```
Fit all 3 models
    |
    v
Check convergence
    |
    +-- >1 model fails --> Document failure pattern
    |                      Investigate shared pathology
    |                      Consider data issues
    |
    v
Compute LOO-CV
    |
    +-- All similar (ΔELPD < 2SE) --> Distributional choice doesn't matter
    |                                   Focus on functional form instead
    |
    +-- Clear winner (ΔELPD > 4SE) --> Proceed to PPC
    |
    v
Posterior Predictive Checks
    |
    +-- All fail Var/Mean --> Pivot to time-varying dispersion
    |
    +-- Winner passes --> Check parameter interpretability
    |
    v
Parameter Interpretation
    |
    +-- Inconsistent across models --> Data weakly informative
    |                                    Report high uncertainty
    |
    +-- Consistent across models --> Robust finding!
    |                                 Report confidently
    |
    v
Sensitivity Analysis
    |
    +-- High prior sensitivity --> Acknowledge prior dependence
    |                               Use informative priors if possible
    |
    +-- Low prior sensitivity --> Robust to prior choice
    |                              Data is informative
    |
    v
FINAL DECISION:
    - Document winning model
    - Report uncertainty honestly
    - Note where models agree (robust findings)
    - Note where models disagree (limitations)
```

---

## Documentation Requirements

For each model that fails, document:

1. **How it failed**: Specific criterion violated
2. **Why it matters**: Scientific interpretation of the failure
3. **What we learned**: Insights from the failure
4. **Next steps**: What to try instead

### Example Failure Report

```markdown
## Model 1 Failure Report

**Failure Mode**: Time-correlated random effects

**Evidence**:
- Correlation(λ[i], year[i]) = 0.62 (p < 0.001)
- Random effects increase monotonically with time

**Interpretation**:
The assumption of constant dispersion is violated. Overdispersion
is not random noise but has systematic time structure.

**What we learned**:
Heteroscedasticity in the EDA is not just sampling variation—it's
a real feature of the data generation process.

**Next steps**:
Fit Model 4 with time-varying dispersion:
log(φ[i]) = γ₀ + γ₁ × year[i]
```

---

## Success = Finding Truth, Not Fitting Models

**Remember**:
- A model that fails quickly and informatively is **better** than a model that "works" but misleads
- Computational failure often indicates **model misspecification**, not just technical issues
- Similar performance across models suggests **focus on the wrong level of detail**
- Inconsistent parameters mean **honest uncertainty**, not failure

**The goal is not to make these models work.**

**The goal is to learn what the data can and cannot tell us.**

---

## Red Flags Summary Table

| Red Flag | Diagnosis | Action |
|----------|-----------|--------|
| All models similar LOO-CV | Distributional choice doesn't matter | Focus on functional form |
| All fail Var/Mean PPC | Homogeneous dispersion wrong | Time-varying dispersion |
| Parameters inconsistent | Data weakly informative | Stronger priors or more data |
| High prior-posterior overlap | n=40 not enough | Informative priors or collect data |
| Multiple computational failures | Model/data pathology | Simplify or check data quality |
| Winner has Rhat > 1.01 | False winner (non-convergence) | Fix convergence or reject |
| LOO-CV SE > |ΔELPD| | No clear winner | Report model uncertainty |
| Back-transformation fails (Model 2) | Count structure matters | Use count models |
| ν ≈ 1 (Model 3) | Poisson adequate | Contradicts EDA; investigate |
| λ correlated with time (Model 1) | Heteroscedasticity | Time-varying dispersion |

---

**Final Reminder**:

**If all three alternative models fail or perform worse than standard NegBin, that's a successful experiment.**

It means the standard approach is robust and well-justified. Sometimes the simplest answer is correct.

The goal is **not** to be clever. The goal is to **find truth**.
