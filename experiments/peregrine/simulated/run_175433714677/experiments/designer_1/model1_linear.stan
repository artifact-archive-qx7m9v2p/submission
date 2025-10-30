// Model 1: Log-Linear Negative Binomial
// Simplest baseline: Exponential growth with constant overdispersion
//
// Mathematical form:
//   C[i] ~ NegativeBinomial(μ[i], φ)
//   log(μ[i]) = β₀ + β₁ × year[i]
//
// Parameters: 3 (β₀, β₁, φ)
// Complexity: Low
// Interpretation: Constant exponential growth rate

data {
  int<lower=0> N;               // Number of observations (n=40)
  array[N] int<lower=0> y;      // Count outcomes (C)
  vector[N] year;               // Standardized year variable
}

parameters {
  real beta_0;                  // Intercept: log(μ) at year=0
  real beta_1;                  // Slope: growth rate on log-scale
  real<lower=0> phi;            // Dispersion parameter (larger = less overdispersion)
}

model {
  // Priors (weakly informative, centered at EDA estimates)
  beta_0 ~ normal(4.3, 1.0);    // E[log(counts)] at year=0 ≈ 4.3
  beta_1 ~ normal(0.85, 0.5);   // Expected growth rate ≈ 0.85
  phi ~ exponential(0.667);     // E[φ] = 1.5 (EDA estimate)

  // Likelihood (vectorized for efficiency)
  y ~ neg_binomial_2_log(beta_0 + beta_1 * year, phi);
}

generated quantities {
  vector[N] log_lik;            // Pointwise log-likelihood for LOO-CV
  array[N] int y_rep;           // Posterior predictive samples
  vector[N] mu;                 // Expected counts on natural scale
  real growth_rate;             // Annual growth multiplier: exp(β₁)

  // Compute derived quantities
  growth_rate = exp(beta_1);

  for (i in 1:N) {
    real log_mu_i = beta_0 + beta_1 * year[i];

    // Expected count on natural scale
    mu[i] = exp(log_mu_i);

    // Log-likelihood for LOO-CV
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | log_mu_i, phi);

    // Posterior predictive samples for PPC
    y_rep[i] = neg_binomial_2_log_rng(log_mu_i, phi);
  }
}
