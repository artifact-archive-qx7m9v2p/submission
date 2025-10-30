# Falsification Checklist: Model Designer 3

**Purpose**: Quick reference for determining when to ABANDON each proposed model

---

## Model 1: Finite Mixture of Binomials

### Primary Falsification Metric: Component Separation

```python
# Compute for each trial i and component k:
gamma[i, k] = P(z_i = k | data)

# Check:
component_separation = mean([max(gamma[i, :]) for i in range(N)])

# ABANDON IF: component_separation < 0.60
```

### Secondary Checks

| Criterion | Metric | Threshold | Action if Failed |
|-----------|--------|-----------|------------------|
| Within-component dispersion | φ_k for each component | Any φ_k > 2.0 | Mixture doesn't capture all heterogeneity |
| Assignment stability | Cramer's V across MCMC | V < 0.80 | Unstable clustering |
| Model complexity | ΔWAIC vs K+1 components | Monotonic decrease | No natural K, use continuous |
| Predictive performance | ΔLOO vs Beta-Binomial | ΔLOO > 2 | Simpler model preferred |

### Diagnostic Code

```python
# Posterior predictive check WITHIN each component
for k in range(K):
    trials_in_k = [i for i in range(N) if gamma[i, k] > 0.5]
    if len(trials_in_k) > 2:
        r_k = r[trials_in_k]
        n_k = n[trials_in_k]
        p_k_hat = sum(r_k) / sum(n_k)

        # Chi-square test within component
        chi_sq = sum((r_k[i] - n_k[i]*p_k_hat)**2 / (n_k[i]*p_k_hat*(1-p_k_hat))
                     for i in range(len(trials_in_k)))
        df = len(trials_in_k) - 1
        phi_k = chi_sq / df

        print(f"Component {k}: φ = {phi_k:.2f}")
        if phi_k > 2.0:
            print(f"  WARNING: Still overdispersed!")
```

---

## Model 2: Robust Contamination Model

### Primary Falsification Metric: Outlier Coherence

```python
# Posterior probability of contamination
prob_outlier[i] = P(δ_i = 1 | data)

# Expected outliers: trials 1 and/or 8
trial_1_outlier = prob_outlier[0] > 0.6
trial_8_outlier = prob_outlier[7] > 0.6

# ABANDON IF: (NOT trial_1_outlier) AND (NOT trial_8_outlier)
```

### Secondary Checks

| Criterion | Metric | Threshold | Action if Failed |
|-----------|--------|-----------|------------------|
| Contamination rate | E[λ] | < 0.05 OR > 0.30 | Too few or too many outliers |
| Clean subset dispersion | φ on trials with prob_outlier < 0.3 | φ > 1.5 | Contamination doesn't explain overdispersion |
| Predictive performance | ΔLOO vs Beta-Binomial | ΔLOO > 2 | Overfitting to spurious outliers |
| Prior sensitivity | KL(posterior \|\| prior) for λ | KL < 0.1 | Data doesn't inform outlier rate |

### Diagnostic Code

```python
# Check clean subset
clean_trials = [i for i in range(N) if prob_outlier[i] < 0.3]
if len(clean_trials) > 5:
    r_clean = r[clean_trials]
    n_clean = n[clean_trials]
    p_clean = sum(r_clean) / sum(n_clean)

    chi_sq_clean = sum((r_clean[i] - n_clean[i]*p_clean)**2 /
                       (n_clean[i]*p_clean*(1-p_clean))
                       for i in range(len(clean_trials)))
    phi_clean = chi_sq_clean / (len(clean_trials) - 1)

    print(f"Clean subset: φ = {phi_clean:.2f}")
    if phi_clean > 1.5:
        print("  WARNING: Clean subset still overdispersed!")
```

---

## Model 3: Structured Outlier Detection

### Primary Falsification Metric: Mechanism Necessity

```python
# Posterior probability of outlier status
omega_post = [P(ω_i = 1 | data) for i in range(N)]

# Check if mechanism is used
max_omega = max(omega_post)
mean_omega = mean(omega_post)

# ABANDON IF: max_omega < 0.20
# (no trial is clearly an outlier)
```

### Secondary Checks

| Criterion | Metric | Threshold | Action if Failed |
|-----------|--------|-----------|------------------|
| Parameter informativeness | 95% CI for η_1 | Includes 0 | Deviations don't predict outlier status |
| Computational health | Divergences, R̂, ESS | Diverg > 1%, R̂ > 1.01, ESS < 400 | Model too complex |
| Seed stability | Correlation of ω across seeds | Corr < 0.7 | Multimodal/unstable inference |
| Model comparison | ΔWAIC vs Beta-Binomial | ΔWAIC > -2 | Added complexity unjustified |

### Diagnostic Code

```python
# Check if η_1 is informative
eta_1_samples = trace.posterior['eta_1'].values.flatten()
eta_1_ci = np.percentile(eta_1_samples, [2.5, 97.5])

print(f"η_1 95% CI: [{eta_1_ci[0]:.2f}, {eta_1_ci[1]:.2f}]")
if eta_1_ci[0] < 0 < eta_1_ci[1]:
    print("  WARNING: η_1 not significantly different from 0!")

# Check computational health
divergences = trace.sample_stats['diverging'].sum().item()
pct_divergent = 100 * divergences / len(trace.posterior.draw)
print(f"Divergences: {divergences} ({pct_divergent:.1f}%)")
if pct_divergent > 1.0:
    print("  WARNING: High divergence rate!")
```

---

## Universal Checks (All Models)

### Computational Health

```python
def check_convergence(trace, model_name):
    """Check if MCMC converged properly"""

    # Gelman-Rubin statistic
    rhat = az.rhat(trace)
    max_rhat = rhat.max()

    print(f"\n{model_name} Convergence:")
    print(f"  Max R̂: {max_rhat:.4f}")
    if max_rhat > 1.01:
        print("  FAIL: Poor convergence (R̂ > 1.01)")
        return False

    # Effective sample size
    ess_bulk = az.ess(trace, method='bulk')
    min_ess = ess_bulk.min()

    print(f"  Min ESS: {min_ess:.0f}")
    if min_ess < 400:
        print("  WARNING: Low effective sample size")

    # Divergences
    if 'diverging' in trace.sample_stats:
        n_divergent = trace.sample_stats['diverging'].sum().item()
        n_total = len(trace.posterior.draw) * len(trace.posterior.chain)
        pct_div = 100 * n_divergent / n_total

        print(f"  Divergences: {n_divergent} ({pct_div:.1f}%)")
        if pct_div > 1.0:
            print("  FAIL: Too many divergences")
            return False

    print("  PASS: Computational health OK")
    return True
```

### Model Comparison

```python
def compare_to_baseline(model_trace, baseline_trace, model_name):
    """Compare model to Beta-Binomial baseline"""

    # Compute WAIC
    waic_model = az.waic(model_trace)
    waic_baseline = az.waic(baseline_trace)

    delta_waic = waic_model.elpd_waic - waic_baseline.elpd_waic
    se_diff = waic_model.se_diff

    print(f"\n{model_name} vs Beta-Binomial:")
    print(f"  ΔWAIC: {delta_waic:.1f} ± {se_diff:.1f}")

    if delta_waic > -2:
        print("  REJECT: Model not better than baseline (ΔWAIC > -2)")
        return False
    elif delta_waic < -2:
        print("  PASS: Model preferred over baseline")
        return True
    else:
        print("  MARGINAL: Models essentially equivalent")
        return None
```

### Posterior Predictive Checks

```python
def posterior_predictive_overdispersion(trace, data):
    """Check if model captures overdispersion"""

    # Generate posterior predictive samples
    r_obs = data['r']
    n = data['n']
    N = len(r_obs)

    # Extract posterior predictive simulations
    r_pred = trace.posterior_predictive['r'].values  # Shape: (chains, draws, N)

    # Compute dispersion for each simulation
    phi_pred = []
    for chain in range(r_pred.shape[0]):
        for draw in range(r_pred.shape[1]):
            r_sim = r_pred[chain, draw, :]
            p_pooled = r_sim.sum() / n.sum()

            chi_sq = sum((r_sim[i] - n[i]*p_pooled)**2 /
                        (n[i]*p_pooled*(1-p_pooled) + 1e-6)
                        for i in range(N))
            phi_sim = chi_sq / (N - 1)
            phi_pred.append(phi_sim)

    # Observed dispersion
    p_obs = r_obs.sum() / n.sum()
    chi_sq_obs = sum((r_obs[i] - n[i]*p_obs)**2 /
                     (n[i]*p_obs*(1-p_obs)) for i in range(N))
    phi_obs = chi_sq_obs / (N - 1)

    # Posterior predictive p-value
    ppp = np.mean(np.array(phi_pred) >= phi_obs)

    print(f"\nPosterior Predictive Check (Dispersion):")
    print(f"  Observed φ: {phi_obs:.2f}")
    print(f"  Predicted φ: {np.mean(phi_pred):.2f} ± {np.std(phi_pred):.2f}")
    print(f"  PPP-value: {ppp:.3f}")

    if ppp < 0.05 or ppp > 0.95:
        print("  WARNING: Model doesn't capture dispersion well")
        return False
    else:
        print("  PASS: Dispersion captured")
        return True
```

---

## Decision Tree

```
START: Fit all three models + Beta-Binomial baseline

├─ Check computational health for each model
│  ├─ FAIL → REJECT model, document reason
│  └─ PASS → Continue
│
├─ Compare to baseline (WAIC)
│  ├─ ALL models ΔWAIC > -2 → USE BASELINE, STOP
│  └─ SOME models ΔWAIC < -2 → Continue with those
│
├─ Apply model-specific falsification criteria
│  │
│  ├─ Finite Mixture:
│  │  ├─ Component separation < 0.60 → REJECT
│  │  ├─ Within-component φ > 2.0 → REJECT
│  │  ├─ Assignment instability → REJECT
│  │  └─ PASS → Keep as candidate
│  │
│  ├─ Robust Contamination:
│  │  ├─ No outliers identified → REJECT
│  │  ├─ Too many outliers → REJECT
│  │  ├─ Wrong trials flagged → REJECT
│  │  ├─ Clean subset φ > 1.5 → REJECT
│  │  └─ PASS → Keep as candidate
│  │
│  └─ Structured Outlier:
│     ├─ max(ω) < 0.20 → REJECT
│     ├─ η_1 ≈ 0 → REJECT
│     ├─ Computational issues → REJECT
│     └─ PASS → Keep as candidate
│
├─ Stress tests on remaining candidates
│  ├─ Jackknife (remove trials 1, 4, 8)
│  ├─ Prior sensitivity
│  └─ Posterior predictive checks
│
└─ Final selection:
   ├─ 0 candidates → Use Beta-Binomial
   ├─ 1 candidate → Report with caveats
   └─ 2+ candidates → Model ensemble or report uncertainty
```

---

## Red Flags (Abandon Entire Approach)

These indicate fundamental problems requiring major pivot:

1. **Data quality issue discovered**
   - Measurement error
   - Transcription error
   - Wrong units

2. **Binomial likelihood inappropriate**
   - Need count model instead
   - Need categorical model instead
   - Continuous response misclassified

3. **Exchangeability violated**
   - Strong temporal autocorrelation
   - Spatial structure
   - Hierarchical clustering

4. **All models fail all checks**
   - Nothing works
   - Need completely different approach
   - Consider non-parametric methods

---

## Reporting Template

For each model tested, report:

```markdown
## [Model Name]

### Computational Diagnostics
- Max R̂: [value]
- Min ESS: [value]
- Divergences: [count] ([percentage]%)
- Runtime: [minutes]
- **Status**: [PASS/FAIL]

### Falsification Criteria
[For each criterion:]
- [Criterion name]: [metric] = [value] (threshold: [threshold])
  - **Result**: [PASS/FAIL/WARN]

### Model Comparison
- ΔWAIC vs baseline: [value] ± [SE]
- ΔLOO vs baseline: [value] ± [SE]
- **Conclusion**: [Preferred/Rejected/Equivalent]

### Stress Tests
[For each test:]
- [Test name]: [Result]

### Overall Decision
**[ACCEPT/REJECT/MARGINAL]**

Justification: [1-2 sentences]
```

---

## Quick Reference: Threshold Summary

| Model | Metric | Pass | Warn | Fail |
|-------|--------|------|------|------|
| **Finite Mixture** | Component separation | > 0.75 | 0.60-0.75 | < 0.60 |
| | Within φ_k | < 1.5 | 1.5-2.0 | > 2.0 |
| | Assignment stability (V) | > 0.85 | 0.75-0.85 | < 0.75 |
| **Contamination** | E[λ] | 0.05-0.25 | 0.25-0.30 | < 0.05 or > 0.30 |
| | Clean subset φ | < 1.3 | 1.3-1.5 | > 1.5 |
| | Outlier coherence | Trials 1 or 8 | Other trials | Random |
| **Structured** | max(ω) | > 0.50 | 0.20-0.50 | < 0.20 |
| | η_1 CI | Excludes 0 | Wide | Includes 0 |
| | Divergences | < 0.5% | 0.5-1% | > 1% |
| **All Models** | R̂ | < 1.01 | 1.01-1.05 | > 1.05 |
| | ESS | > 800 | 400-800 | < 400 |
| | ΔWAIC | < -4 | -4 to -2 | > -2 |
| | PPP-value | 0.2-0.8 | 0.05-0.2 or 0.8-0.95 | < 0.05 or > 0.95 |

---

**Remember**: Failing a check is SUCCESS if it helps us rule out wrong models. The goal is finding truth, not confirming hypotheses.
