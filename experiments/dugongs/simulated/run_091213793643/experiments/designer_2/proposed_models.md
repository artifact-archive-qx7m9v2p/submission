# Flexible & Robust Bayesian Model Design
## Designer 2: Protecting Against Misspecification

**Designer**: Model Designer 2 (Flexible/Robust Perspective)
**Date**: 2025-10-28
**Data**: N=27, x ∈ [1, 31.5], Y ∈ [1.71, 2.63]
**Philosophy**: Relax parametric assumptions, protect against outliers, and let data determine functional complexity

---

## Executive Summary

This design proposes **three distinct model classes** that protect against different types of misspecification:

1. **Gaussian Process Regression** (flexible non-parametric)
2. **Robust Regression with Student-t Errors** (outlier protection)
3. **Penalized B-Splines** (flexible parametric with automatic smoothing)

Each model addresses concerns that parametric forms (logarithmic, quadratic) might be **wrong**. The EDA shows strong evidence for non-linearity and saturation, but with only n=27 observations and one influential point (x=31.5, Cook's D=0.84), we need approaches that:
- Don't commit to a specific parametric form
- Handle potential outliers gracefully
- Quantify uncertainty about functional form itself
- Provide automatic complexity control

---

## Critical Assessment of EDA Findings

### What the EDA Got Right
- Strong non-linear relationship (Spearman ρ=0.78)
- Saturation/diminishing returns pattern
- One influential high-leverage point (x=31.5)
- Variance heterogeneity trend (4.6:1 ratio, though not significant)
- Large data gap at x ∈ [23, 29]

### What Could Be Wrong
1. **Parametric form assumption**: Log model R²=0.829 vs Quadratic R²=0.862 - small sample can't distinguish
2. **Constant variance**: Trend suggests decreasing variance, but BP test ns (power issue with n=27)
3. **Normal errors**: Residuals pass normality test, but outliers could exist
4. **Smooth function**: What if there's a regime change or changepoint?
5. **Single influential point drives conclusions about plateau**

### Why Flexible Models Matter Here
- **n=27 is small**: Parametric assumptions can mislead with few observations
- **High leverage point**: x=31.5 has Cook's D=0.84 - needs robust approach
- **Sparse high-x region**: Only 7 observations for x>15 - parametric extrapolation risky
- **Gap at x∈[23,29]**: Need honest uncertainty quantification
- **Variance trend**: Heteroscedasticity possible but underpowered - flexible model can adapt

---

## Model Class 1: Gaussian Process Regression

### Philosophy
Don't assume ANY parametric form. Let the data tell us the shape through the covariance structure. GPs provide:
- Fully non-parametric function estimation
- Automatic uncertainty quantification (wider in gaps)
- Principled extrapolation behavior
- Natural handling of sparse data

### Mathematical Specification

**Likelihood**:
```
Y_i ~ Normal(f(x_i), σ²)
f ~ GP(m(x), k(x, x'))
```

**Mean Function** (prior belief about trend):
```
m(x) = μ_0
```
Simple constant mean - let covariance capture non-linearity

**Covariance Function** (Matérn 3/2 kernel - more flexible than squared exponential):
```
k(x, x') = α² * (1 + √3*d/ℓ) * exp(-√3*d/ℓ)
where d = |x - x'|
```

**Why Matérn 3/2?**
- More flexible than SE (allows functions that aren't infinitely differentiable)
- More robust to outliers than SE
- Captures saturation naturally if lengthscale is appropriate
- Standard choice for physical processes

**Full Model**:
```
Y_i ~ Normal(f(x_i), σ²)
f ~ GP(μ_0, k(x, x'))

Priors:
μ_0 ~ Normal(2.3, 0.5)           # Prior mean near observed mean Y
α² ~ HalfNormal(0.3)              # Signal variance (marginal variance of Y ≈ 0.28²)
ℓ ~ InverseGamma(5, 10)           # Lengthscale - weakly informative
σ ~ HalfNormal(0.15)              # Noise std - slightly larger than log model residual (0.115)
```

**Prior Justification**:
- **μ_0**: Centered at observed mean(Y)=2.32, but allow wide range
- **α²**: Signal variance should capture most of variance in Y (std=0.28), so HalfNormal(0.3) gives mass to 0-0.6
- **ℓ**: Lengthscale controls smoothness. With x range [1, 31.5], span=30.5
  - Small ℓ (1-5): very wiggly, would overfit n=27
  - Large ℓ (20-50): very smooth, would underfit clear non-linearity
  - InverseGamma(5, 10) gives mode ≈ 2.5, mass on [1, 10] - reasonable smoothness
  - Prior predictive checks essential here
- **σ**: Residual noise around 0.1-0.15 based on log model

### Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;

  // For predictions
  int<lower=1> N_new;
  vector[N_new] x_new;
}

transformed data {
  real x_mean = mean(x);
  real x_sd = sd(x);
  vector[N] x_std = (x - x_mean) / x_sd;
  vector[N_new] x_new_std = (x_new - x_mean) / x_sd;
}

parameters {
  real mu_0;                    // Mean function
  real<lower=0> alpha_sq;       // Signal variance
  real<lower=0> length_scale;   // Lengthscale
  real<lower=0> sigma;          // Noise std
}

model {
  matrix[N, N] K;
  matrix[N, N] L_K;
  vector[N] mu = rep_vector(mu_0, N);

  // Priors
  mu_0 ~ normal(2.3, 0.5);
  alpha_sq ~ normal(0, 0.3);
  length_scale ~ inv_gamma(5, 10);
  sigma ~ normal(0, 0.15);

  // Covariance matrix - Matérn 3/2 on standardized x
  for (i in 1:N) {
    for (j in 1:N) {
      real d = fabs(x_std[i] - x_std[j]);
      real sqrt3_d_l = sqrt(3.0) * d / length_scale;
      K[i, j] = alpha_sq * (1 + sqrt3_d_l) * exp(-sqrt3_d_l);
      if (i == j) {
        K[i, j] = K[i, j] + sigma^2;  // Add noise to diagonal
      }
    }
  }

  L_K = cholesky_decompose(K);
  Y ~ multi_normal_cholesky(mu, L_K);
}

generated quantities {
  vector[N_new] f_new;
  vector[N_new] Y_new;
  vector[N] log_lik;

  // Posterior predictions
  {
    matrix[N, N] K;
    matrix[N_new, N_new] K_new;
    matrix[N_new, N] K_cross;
    matrix[N, N] L_K;
    vector[N] K_inv_Y;
    vector[N] mu = rep_vector(mu_0, N);

    // Compute covariances
    for (i in 1:N) {
      for (j in 1:N) {
        real d = fabs(x_std[i] - x_std[j]);
        real sqrt3_d_l = sqrt(3.0) * d / length_scale;
        K[i, j] = alpha_sq * (1 + sqrt3_d_l) * exp(-sqrt3_d_l);
        if (i == j) K[i, j] = K[i, j] + sigma^2;
      }
    }

    for (i in 1:N_new) {
      for (j in 1:N_new) {
        real d = fabs(x_new_std[i] - x_new_std[j]);
        real sqrt3_d_l = sqrt(3.0) * d / length_scale;
        K_new[i, j] = alpha_sq * (1 + sqrt3_d_l) * exp(-sqrt3_d_l);
        if (i == j) K_new[i, j] = K_new[i, j] + sigma^2;
      }
    }

    for (i in 1:N_new) {
      for (j in 1:N) {
        real d = fabs(x_new_std[i] - x_std[j]);
        real sqrt3_d_l = sqrt(3.0) * d / length_scale;
        K_cross[i, j] = alpha_sq * (1 + sqrt3_d_l) * exp(-sqrt3_d_l);
      }
    }

    L_K = cholesky_decompose(K);
    K_inv_Y = mdivide_left_tri_low(L_K, Y - mu);
    K_inv_Y = mdivide_right_tri_low(K_inv_Y', L_K)';

    f_new = rep_vector(mu_0, N_new) + K_cross * K_inv_Y;

    for (i in 1:N_new) {
      Y_new[i] = normal_rng(f_new[i], sigma);
    }
  }

  // Log likelihood for LOO
  {
    matrix[N, N] K;
    matrix[N, N] L_K;
    vector[N] mu = rep_vector(mu_0, N);

    for (i in 1:N) {
      for (j in 1:N) {
        real d = fabs(x_std[i] - x_std[j]);
        real sqrt3_d_l = sqrt(3.0) * d / length_scale;
        K[i, j] = alpha_sq * (1 + sqrt3_d_l) * exp(-sqrt3_d_l);
        if (i == j) K[i, j] = K[i, j] + sigma^2;
      }
    }

    L_K = cholesky_decompose(K);
    log_lik = multi_normal_cholesky_lpdf(Y | mu, L_K) / N * ones_vector(N);
  }
}
```

### Expected Behavior
- **Smooth interpolation** between observed points
- **Automatic adaptation** to local data density (wider CI in gap region)
- **Honest uncertainty** at x=31.5 (will have wide posterior due to high leverage)
- **Marginal variance** should match observed variance of Y
- **Lengthscale posterior** tells us about smoothness - if very large, suggests simple parametric form sufficient

### Falsification Criteria

**I will abandon this GP model if**:

1. **Posterior lengthscale ℓ_post >> 30** (much larger than x range)
   - Interpretation: Data so smooth that simple parametric form (log/linear) sufficient
   - Action: Switch to parametric model for parsimony

2. **Extreme signal-to-noise ratio**: α²/σ² < 0.1 or > 100
   - Low ratio: Most variation is noise - simpler model or data quality issue
   - High ratio: Overfitting - likely too flexible for n=27

3. **Posterior predictive checks fail**:
   - Predicted Y range wildly exceeds [1.71, 2.63]
   - Function not monotonic increasing (contradicts all evidence)
   - Predictions at replicates don't capture observed variability

4. **Computational failure**:
   - Rhat > 1.05 for key parameters
   - Divergent transitions > 5% despite tuning
   - Sampling time > 1 hour (inefficiency suggests misspecification)

5. **LOO-CV worse than simple log model by ΔELPD > 3**:
   - GP more flexible but worse predictions = overfitting
   - Parsimony principle: choose simpler model

6. **Prior-posterior conflict**:
   - Posterior concentrates far from prior (e.g., ℓ → 0 or ℓ → ∞)
   - Suggests prior misspecified or model inappropriate

### Computational Considerations

**Challenges**:
- **O(N³) complexity** for matrix inversion (but n=27 is tiny, ~20k flops)
- **Cholesky decomposition** can fail if covariance matrix ill-conditioned
- **Lengthscale** can be difficult to sample (reparameterization may help)

**Mitigation**:
- Use standardized x (mean 0, sd 1) for numerical stability
- Add small jitter to diagonal if Cholesky fails (1e-6)
- Consider log(lengthscale) parameterization
- Use informative priors on lengthscale to avoid extreme values

**Expected Runtime**: 2-5 minutes for 4000 iterations (n=27 is small)

**Stan vs PyMC**: Stan preferred here
- Stan's Cholesky decomposition more numerically stable
- Better HMC adaptation for GP hyperparameters
- PyMC GPs can be slower for custom kernels

---

## Model Class 2: Robust Regression with Student-t Errors

### Philosophy
The influential point at x=31.5 (Cook's D=0.84) and unusual replicate variability at x=15.5 suggest potential outliers. Standard Gaussian likelihood downweights outliers too harshly - one bad point can distort entire fit.

Student-t errors provide:
- **Automatic outlier detection** via heavy tails
- **Robustness** to misspecified variance
- **Degrees of freedom** as data-driven parameter (ν)

### Mathematical Specification

**Core Model**: Keep simple parametric form (log), but make errors robust

```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = α + β * log(x_i)
```

**Why Student-t?**
- ν=∞: reduces to Normal (no robustness)
- ν=1: Cauchy (extreme robustness, maybe too much)
- ν≈3-10: typical for moderate robustness
- **Let data choose ν** via prior

**Full Model**:
```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = α + β * log(x_i)

Priors:
α ~ Normal(1.75, 0.5)         # Intercept from EDA
β ~ Normal(0.27, 0.15)        # Slope from EDA, ensure positive
ν ~ Gamma(2, 0.1)             # Mean=20, allows ν ∈ [3, 50]
σ ~ HalfNormal(0.2)           # Slightly larger than normal model
```

**Prior Justification**:
- **α, β**: Same as normal log model (centered on EDA estimates)
- **ν**: Gamma(2, 0.1) gives mode ≈ 10, allows wide range
  - If ν_post > 30: data doesn't support outliers, normal sufficient
  - If ν_post < 10: outliers present, robustness needed
  - Gamma shape allows learning from data
- **σ**: Slightly larger prior than normal model (robustness trades off efficiency)

### Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;

  // For predictions
  int<lower=1> N_new;
  vector[N_new] x_new;
}

transformed data {
  vector[N] log_x = log(x);
  vector[N_new] log_x_new = log(x_new);
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
  real<lower=1> nu;  // Degrees of freedom, constrain >1 for finite variance
}

transformed parameters {
  vector[N] mu = alpha + beta * log_x;
}

model {
  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  sigma ~ normal(0, 0.2);
  nu ~ gamma(2, 0.1);  // Gamma(2, 0.1) has mean=20, sd=14.1

  // Likelihood - robust to outliers
  Y ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N_new] mu_new = alpha + beta * log_x_new;
  vector[N_new] Y_new;
  vector[N] log_lik;
  vector[N] Y_rep;

  // Posterior predictions
  for (i in 1:N_new) {
    Y_new[i] = student_t_rng(nu, mu_new[i], sigma);
  }

  // Log likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = student_t_lpdf(Y[i] | nu, mu[i], sigma);
  }

  // Posterior predictive replications
  for (i in 1:N) {
    Y_rep[i] = student_t_rng(nu, mu[i], sigma);
  }
}
```

### Expected Behavior
- **If no outliers**: ν_post → large (>30), essentially Gaussian
- **If outliers exist**: ν_post ≈ 3-10, automatic downweighting
- **x=31.5 point**: If truly influential outlier, will get lower weight
- **Replicate at x=15.5**: High variability may push ν down
- **σ posterior**: Should be similar to Gaussian model if ν large, slightly smaller if outliers present

### Falsification Criteria

**I will abandon this robust regression if**:

1. **Posterior ν > 50 with high confidence (95% CI doesn't include 30)**:
   - Interpretation: Data strongly prefers Gaussian errors
   - Action: Use standard Gaussian model for efficiency

2. **Posterior ν < 3**:
   - Interpretation: Extreme outliers, possibly data error or model misspecification
   - Action: Investigate data quality, consider different functional form

3. **LOO worse than Gaussian model by ΔELPD > 2**:
   - Robustness doesn't help predictions
   - Extra parameter (ν) penalized without benefit

4. **Posterior predictive checks show**:
   - Student-t model generates Y values outside [1, 3] frequently
   - Predicted distribution much heavier-tailed than observed
   - Model doesn't improve fit to "outlier" at x=31.5

5. **Parametric form still inadequate**:
   - Residual patterns persist (non-random)
   - Changing error distribution doesn't fix functional form issue
   - Action: Need flexible functional form, not just robust errors

6. **No identifiable outliers in posterior predictive**:
   - If all observations have similar standardized residuals
   - Robustness unnecessary

### Computational Considerations

**Advantages**:
- Simple model, fast sampling
- Student-t well-supported in Stan
- Only 1 extra parameter vs Gaussian (ν)

**Challenges**:
- ν can have multimodal posterior if data ambiguous
- Heavy tails can slow HMC adaptation
- Need to check effective sample size for ν

**Expected Runtime**: 30-60 seconds for 4000 iterations

**Stan vs PyMC**: Either works well
- Stan: slightly faster, better diagnostics
- PyMC: easier prior visualization
- **Recommendation**: Stan for consistency with GP

---

## Model Class 3: Penalized B-Spline Regression

### Philosophy
Middle ground between parametric (log/quadratic) and fully non-parametric (GP):
- Use flexible basis functions (B-splines)
- Penalize roughness to avoid overfitting
- Automatic smoothing via hierarchical prior on basis coefficients

B-splines provide:
- **Local flexibility**: Can capture different curvature in different regions
- **Interpretability**: Basis coefficients show where function changes
- **Parsimony**: Small number of knots (5-7) sufficient for n=27
- **Computational efficiency**: Linear in number of basis functions

### Mathematical Specification

**Model Structure**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = Σ_{k=1}^K β_k * B_k(x_i)

where B_k(x) are B-spline basis functions of degree 3 (cubic)
```

**Penalization** (hierarchical prior for smoothness):
```
β_k ~ Normal(0, τ²)
τ ~ HalfCauchy(0, 1)  # Regularization parameter
```

**Why This Prior?**
- Large τ: minimal penalty, very flexible (risk overfitting)
- Small τ: strong penalty, smooth function (risk underfitting)
- HalfCauchy: heavy tail allows data to choose flexibility
- Hierarchical structure: automatic complexity control

**Knot Placement**:
- Use **5 interior knots** for n=27 (conservative, ~5 obs per region)
- Place at quantiles of x: [0.17, 0.33, 0.50, 0.67, 0.83]
- With degree 3, gives K = 5 + 4 = 9 basis functions

**Full Model**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = Σ_{k=1}^9 β_k * B_k(x_i)

Priors:
β_k ~ Normal(0, τ²)  for k=1,...,9
τ ~ HalfCauchy(0, 1)
σ ~ HalfNormal(0.15)
```

**Alternative: Second-Order Penalty**
For P-spline (penalized difference):
```
β_k ~ Normal(0, τ²)  for k=1,2
β_k ~ Normal(2*β_{k-1} - β_{k-2}, τ²)  for k>2
```
This penalizes **curvature** rather than absolute size.

### Stan Implementation

```stan
functions {
  // B-spline basis functions (degree 3, cubic)
  vector build_b_spline(real x, vector knots, int deg) {
    int n_knots = rows(knots);
    int n_basis = n_knots + deg - 1;
    vector[n_basis] b;

    // Augmented knot sequence
    vector[n_knots + 2*deg] t;
    for (i in 1:deg) {
      t[i] = knots[1];
      t[n_knots + deg + i] = knots[n_knots];
    }
    for (i in 1:n_knots) {
      t[deg + i] = knots[i];
    }

    // Cox-de Boor recursion (simplified for Stan)
    // [Implementation details omitted for brevity - use splines package]
    // Return basis evaluations at x
    return b;
  }
}

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;

  int<lower=1> n_knots;
  vector[n_knots] knots;  // Interior knots
  int<lower=1> deg;        // Spline degree (typically 3)

  // For predictions
  int<lower=1> N_new;
  vector[N_new] x_new;
}

transformed data {
  int n_basis = n_knots + deg - 1;
  matrix[N, n_basis] B;
  matrix[N_new, n_basis] B_new;

  // Build design matrices
  for (i in 1:N) {
    B[i, :] = build_b_spline(x[i], knots, deg)';
  }
  for (i in 1:N_new) {
    B_new[i, :] = build_b_spline(x_new[i], knots, deg)';
  }
}

parameters {
  vector[n_basis] beta;           // Spline coefficients
  real<lower=0> tau;              // Regularization parameter
  real<lower=0> sigma;            // Residual std
}

transformed parameters {
  vector[N] mu = B * beta;
}

model {
  // Priors
  beta ~ normal(0, tau);          // Hierarchical shrinkage
  tau ~ cauchy(0, 1);             // Heavy-tailed global scale
  sigma ~ normal(0, 0.15);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N_new] mu_new = B_new * beta;
  vector[N_new] Y_new;
  vector[N] log_lik;
  vector[N] Y_rep;

  // Posterior predictions
  for (i in 1:N_new) {
    Y_new[i] = normal_rng(mu_new[i], sigma);
  }

  // Log likelihood for LOO
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }

  // Posterior predictive replications
  for (i in 1:N) {
    Y_rep[i] = normal_rng(mu[i], sigma);
  }
}
```

**Practical Note**: B-spline construction in Stan is complex. **Alternative implementation**:
- Construct B-spline basis in Python/R using `patsy`, `scipy.interpolate`, or `mgcv`
- Pass design matrix B to Stan as data
- Much simpler and faster

### Simplified Implementation (Recommended)

**Python preprocessing**:
```python
from patsy import dmatrix
import numpy as np

# Create B-spline basis (cubic, 5 knots)
B = dmatrix("bs(x, df=9, degree=3, include_intercept=True)",
            {"x": x_obs}, return_type="dataframe")
B_new = dmatrix("bs(x, df=9, degree=3, include_intercept=True)",
                {"x": x_new}, return_type="dataframe")

# Pass to Stan
stan_data = {
    'N': len(x_obs),
    'Y': Y_obs,
    'B': B.values,
    'n_basis': B.shape[1],
    'N_new': len(x_new),
    'B_new': B_new.values
}
```

**Simplified Stan model**:
```stan
data {
  int<lower=1> N;
  int<lower=1> n_basis;
  matrix[N, n_basis] B;
  vector[N] Y;

  int<lower=1> N_new;
  matrix[N_new, n_basis] B_new;
}

parameters {
  vector[n_basis] beta;
  real<lower=0> tau;
  real<lower=0> sigma;
}

model {
  beta ~ normal(0, tau);
  tau ~ cauchy(0, 1);
  sigma ~ normal(0, 0.15);

  Y ~ normal(B * beta, sigma);
}

generated quantities {
  vector[N_new] Y_new;
  vector[N] log_lik;

  for (i in 1:N_new) {
    Y_new[i] = normal_rng(dot_product(B_new[i], beta), sigma);
  }

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(Y[i] | dot_product(B[i], beta), sigma);
  }
}
```

### Expected Behavior
- **τ posterior**:
  - Small τ (< 0.5): Strong smoothing, function similar to parametric
  - Large τ (> 2): Weak smoothing, flexible function
- **Basis coefficients β**: Should show monotonic pattern (all positive or smoothly varying)
- **Visual fit**: Should closely match data, but smoother than interpolation
- **Uncertainty**: Wide in gap region [23, 29], narrow in dense regions

### Falsification Criteria

**I will abandon this spline model if**:

1. **Posterior τ → 0 (< 0.1)**:
   - Interpretation: Strong penalty → function nearly linear
   - Action: Simple parametric form (log/linear) sufficient

2. **Posterior τ → ∞ (> 5)**:
   - Interpretation: No penalty needed, but then overfitting risk
   - Action: Either simpler parametric or GP with better lengthscale

3. **Basis coefficients not smooth**:
   - Erratic pattern in β_k suggests overfitting
   - Need more regularization or fewer knots

4. **LOO worse than parametric by ΔELPD > 3**:
   - Flexibility penalized without prediction benefit
   - Parsimony favors simpler model

5. **Computational instability**:
   - High correlation between basis coefficients
   - Sampling inefficiency (ESS < 100)
   - Action: Reduce number of knots or use different basis

6. **Posterior predictive failures**:
   - Non-monotonic function (contradicts data)
   - Extreme extrapolation beyond [1, 3]
   - Poor fit to replicates

7. **Visual inspection shows worse fit than log model**:
   - If added flexibility doesn't improve residual patterns
   - If function looks "overfit" (too wiggly)

### Computational Considerations

**Advantages**:
- Fast: Linear algebra only, no covariance matrix
- Stable: Well-conditioned design matrix
- Scalable: Works for larger n (though n=27 here)

**Challenges**:
- Knot placement matters (use quantiles or fixed spacing)
- Number of knots is arbitrary choice (5-7 reasonable for n=27)
- Basis functions correlated (regularization helps)

**Expected Runtime**: 20-40 seconds for 4000 iterations

**Stan vs PyMC**: Either works
- Stan: Better for hierarchical τ
- PyMC: Easier basis construction
- **Recommendation**: Stan with Python preprocessing

---

## Model Comparison Strategy

### Primary Metrics

1. **LOO-CV (ELPD)**: Pointwise predictive accuracy
   - Difference > 3: meaningful
   - Check for high Pareto-k values (outliers/influential points)

2. **WAIC**: Alternative information criterion
   - Should agree with LOO in direction
   - Effective number of parameters (p_waic) indicates complexity

3. **Posterior Predictive Checks**:
   - Marginal: Does Y_rep distribution match Y_obs?
   - Conditional: Does model capture x-Y relationship?
   - Variance: Does model capture heteroscedasticity?
   - Replicates: Does model predict replicate variability?

4. **Prior Predictive Checks** (before fitting):
   - Do prior predictions stay in reasonable range [1, 3]?
   - Is function monotonic increasing?
   - Are predictions at replicates reasonable?

### Secondary Considerations

5. **Parsimony**: Simpler model preferred if ΔELPD < 3
6. **Interpretability**: Can parameters answer scientific questions?
7. **Computational efficiency**: Runtime, convergence diagnostics
8. **Robustness**: Sensitivity to single observations (x=31.5)
9. **Extrapolation**: Behavior beyond observed range

### Decision Rules

**If GP wins by large margin (ΔELPD > 5)**:
- Strong evidence for complex non-parametric form
- Parametric assumptions too restrictive
- Consider why: Is it specific region? Overall curvature?

**If Robust-t wins**:
- Outliers present (check ν posterior)
- Normal likelihood inadequate
- But functional form (log) may still be reasonable

**If Spline wins**:
- Moderate flexibility needed
- Parametric too simple, GP too complex
- Check τ and β patterns for insights

**If models tied (ΔELPD < 2)**:
- Use **simplest** model (parsimony)
- Likely: Robust-t or Spline (fewer assumptions than parametric)
- Report uncertainty about model choice

### Red Flags Requiring Model Class Change

**Evidence that all three models are wrong**:

1. **Systematic residual patterns persist** across all models
   - Action: Consider changepoint, mixture, or different transformation

2. **All models have poor LOO (many high Pareto-k > 0.7)**:
   - Action: Data quality issue or fundamental misspecification

3. **Posterior predictive checks fail for all models**:
   - Action: Reconsider data generation process entirely

4. **x=31.5 drives all conclusions** (sensitivity analysis shows instability):
   - Action: Report results with/without, consider data error

5. **Gap region [23, 29] predictions wildly differ** between models:
   - Action: Report high uncertainty, recommend new data collection

---

## Stress Tests and Sensitivity Analysis

### Critical Tests to Run

1. **Leave-One-Out for x=31.5**:
   - Refit all models without this point
   - Compare parameter estimates and predictions
   - If major change: Point is influential outlier

2. **Leave-Out High-x Region** (x > 20):
   - Fit on x ≤ 20 only (n=20)
   - Predict x > 20
   - Compare to observed
   - Tests extrapolation ability

3. **Replicate Consistency Check**:
   - For 6 x values with replicates, does posterior predictive variance match observed?
   - If not: Variance model inadequate

4. **Prior Sensitivity**:
   - Refit with wider priors (2x scale)
   - Check if conclusions change
   - If yes: Priors too informative

5. **Synthetic Data Recovery**:
   - Simulate from known model (e.g., log with σ=0.12)
   - Fit all three models
   - Check if true model recovered
   - Validates computational implementation

### Expected Failure Modes

**GP might fail if**:
- Lengthscale unidentifiable (posterior equals prior)
- Covariance matrix ill-conditioned (numerical errors)
- Function overfits noise in dense regions

**Robust-t might fail if**:
- No outliers exist (ν → ∞, wasted parameter)
- Functional form wrong (robustness doesn't fix shape)
- Heavy tails generate implausible predictions

**Spline might fail if**:
- Knot placement poor (artifacts at boundaries)
- Too few/many knots (under/overfitting)
- Regularization inadequate (τ not identified)

---

## Alternative Models If Initial Set Fails

### Backup Option 1: Heteroscedastic Logarithmic
If variance clearly varies with x:
```
Y_i ~ Normal(μ_i, σ_i)
μ_i = α + β * log(x_i)
log(σ_i) = γ_0 + γ_1 * log(x_i)  # Variance as function of x
```

### Backup Option 2: Quantile Regression
If interest in specific quantiles (e.g., median) rather than mean:
```
Q_τ(Y|x) = α_τ + β_τ * log(x)
```
Robust to outliers and non-constant variance.

### Backup Option 3: Gaussian Process with Input-Dependent Noise
If both function and variance are non-parametric:
```
Y_i ~ Normal(f(x_i), σ²(x_i))
f ~ GP(0, k_f(x, x'))
log(σ²) ~ GP(μ_σ, k_σ(x, x'))
```

### Backup Option 4: Changepoint Model
If evidence of regime change:
```
Y_i ~ Normal(μ_i, σ)
μ_i = α_1 + β_1*log(x_i)         if x_i < τ
      α_2 + β_2*log(x_i)         if x_i ≥ τ

τ ~ Uniform(5, 20)  # Changepoint location
```

---

## Implementation Priority and Timeline

### Phase 1: Initial Fitting (Day 1)
1. **Fit all three models** with default priors
2. **Check convergence**: Rhat, ESS, divergences
3. **Compute LOO/WAIC**
4. **Basic posterior predictive checks**

**Deliverable**: Initial ranking of models

### Phase 2: Diagnostics (Day 1-2)
5. **Prior predictive checks** (should have been first, but often forgotten)
6. **Detailed residual analysis**
7. **Sensitivity to x=31.5**
8. **Cross-validation for gap region**

**Deliverable**: Diagnostic report with red flags

### Phase 3: Model Refinement (Day 2-3)
9. **Tune problematic models** (priors, parameterization)
10. **Try backup options** if needed
11. **Final model comparison**
12. **Comprehensive posterior predictive checks**

**Deliverable**: Final model recommendation with uncertainty

### Phase 4: Interpretation (Day 3)
13. **Parameter interpretation**
14. **Scientific conclusions**
15. **Uncertainty quantification**
16. **Recommendation for future data collection**

**Deliverable**: Final report with visualizations

---

## Expected Outcomes and Interpretation

### Most Likely Scenario
**Spline or GP wins**, with ΔELPD ≈ 2-4 over simple log model
- Interpretation: Data supports moderate flexibility
- Functional form not perfectly logarithmic
- Small sample (n=27) limits strong conclusions

### High Confidence Prediction
**Robust-t ν_post > 20**
- Interpretation: No strong outliers
- x=31.5 influential but not aberrant
- Normal errors adequate

### Uncertain Outcome
**Which flexible model (GP vs Spline) fits better**
- Both should perform similarly
- Choice based on parsimony and interpretability
- Spline likely more interpretable (basis coefficients)

### Key Parameters to Report

**For GP**:
- Lengthscale ℓ (smoothness)
- Signal variance α² (total variation)
- Noise variance σ² (measurement error)

**For Robust-t**:
- ν (degree of robustness needed)
- α, β (same interpretation as Gaussian log model)
- σ (residual scale)

**For Spline**:
- τ (regularization/smoothness)
- β coefficients (local behavior)
- σ (residual scale)

---

## Summary: When Each Model is Preferred

| Model | Best If... | Worst If... |
|-------|-----------|-------------|
| **GP** | Complex non-linear, want uncertainty quantification, no parametric form clear | Simple relationship, small n makes GP unidentifiable, computational issues |
| **Robust-t** | Outliers suspected, parametric form reasonable, want robustness | No outliers (ν large), functional form wrong, heavy tails unrealistic |
| **Spline** | Moderate flexibility needed, want interpretable basis, computational efficiency | Too smooth (τ→0, use parametric), too complex (high variance), knot artifacts |

---

## Final Recommendations

### Primary Model: Penalized B-Spline
**Rationale**:
- Balance between flexibility and parsimony
- Interpretable coefficients
- Fast computation
- Handles local variations
- Well-suited for n=27

### Secondary Model: Gaussian Process
**Rationale**:
- Fully non-parametric benchmark
- Best uncertainty quantification
- Tests if parametric form adequate (via lengthscale)
- Gold standard for comparison

### Tertiary Model: Robust Student-t
**Rationale**:
- Tests outlier hypothesis
- Simple baseline for flexible models
- Checks if flexibility needed or robustness sufficient

### Decision Logic
1. **Fit all three**, compute LOO
2. **If ΔELPD < 2**: Choose simplest (likely Robust-t)
3. **If 2 < ΔELPD < 5**: Choose Spline (good balance)
4. **If ΔELPD > 5**: Choose GP (strong evidence for non-parametric)

---

## Connection to EDA Findings

### What EDA Told Us
- Logarithmic model R²=0.829 (good but not perfect)
- Variance trend (4.6:1 ratio, not significant)
- Influential point x=31.5 (Cook's D=0.84)
- Gap region [23, 29]

### How These Models Address It
- **GP**: Honest uncertainty in gap, automatic handling of influential points
- **Robust-t**: Downweights x=31.5 if outlier, protects against it
- **Spline**: Local flexibility can fit different regions differently

### What We'll Learn
- **Is log model adequate?** (Compare flexible vs parametric)
- **Are outliers present?** (Check ν posterior)
- **How uncertain are we?** (Posterior predictive intervals)
- **Where should we collect more data?** (High posterior variance regions)

---

## Computational Checklist

### Before Running
- [ ] Prior predictive checks for all models
- [ ] Verify Stan code compiles
- [ ] Test on small simulation
- [ ] Prepare visualization code

### During Running
- [ ] Monitor Rhat, ESS
- [ ] Check for divergences
- [ ] Watch for numerical errors
- [ ] Profile runtime

### After Running
- [ ] Posterior predictive checks
- [ ] LOO with Pareto-k diagnostics
- [ ] Sensitivity analysis
- [ ] Parameter interpretation

---

**End of Model Design Document**

**Next Steps**: Implement these models in Stan, run inference, compare using principled criteria, and iterate based on diagnostic failures. Remember: **The goal is finding truth, not completing a checklist**. If all models fail, pivot to alternatives.

---

## Appendix: Why These Models Over Others?

### Why Not Polynomial Regression?
- Already tested in EDA (quadratic R²=0.862)
- Risk of overfitting with high degrees
- Poor extrapolation behavior
- No automatic complexity control

### Why Not Neural Networks?
- Massive overkill for n=27
- No interpretability
- No uncertainty quantification
- Computational complexity

### Why Not Generalized Additive Models (GAMs)?
- Essentially equivalent to penalized splines (Model 3)
- Would use same basis functions
- Implementation difference, not conceptual

### Why Not Transformation Models?
- EDA tested transformations (Box-Cox, log(Y), etc.)
- No improvement found
- Flexible models more general

### Why Not Mixture Models?
- No evidence of clustering or multimodality
- Would need n >> 27 for identifiability
- Can consider if all models fail

### Why These Three?
1. **GP**: Gold standard non-parametric
2. **Robust-t**: Simplest robustness check
3. **Spline**: Best practical balance

Cover the space from simple-robust to fully flexible.
