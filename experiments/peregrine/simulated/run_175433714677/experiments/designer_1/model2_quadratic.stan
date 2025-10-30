// Model 2: Quadratic Negative Binomial
// Accelerating/decelerating growth with constant overdispersion
//
// Mathematical form:
//   C[i] ~ NegativeBinomial(μ[i], φ)
//   log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
//
// Parameters: 4 (β₀, β₁, β₂, φ)
// Complexity: Medium
// Interpretation: Time-varying exponential growth rate

data {
  int<lower=0> N;               // Number of observations (n=40)
  array[N] int<lower=0> y;      // Count outcomes (C)
  vector[N] year;               // Standardized year variable
  vector[N] year_sq;            // year² (precomputed for efficiency)
}

parameters {
  real beta_0;                  // Intercept: log(μ) at year=0
  real beta_1;                  // Linear coefficient
  real beta_2;                  // Quadratic coefficient (acceleration)
  real<lower=0> phi;            // Dispersion parameter
}

model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);    // Same as Model 1
  beta_1 ~ normal(0.85, 0.5);   // Same as Model 1
  beta_2 ~ normal(0, 0.3);      // Centered at 0 (no prior bias for acceleration)
  phi ~ exponential(0.667);     // E[φ] = 1.5

  // Likelihood (vectorized)
  y ~ neg_binomial_2_log(beta_0 + beta_1 * year + beta_2 * year_sq, phi);
}

generated quantities {
  vector[N] log_lik;            // Pointwise log-likelihood for LOO-CV
  array[N] int y_rep;           // Posterior predictive samples
  vector[N] mu;                 // Expected counts on natural scale
  vector[N] instantaneous_growth_rate; // d(log(μ))/d(year) = β₁ + 2×β₂×year

  for (i in 1:N) {
    real log_mu_i = beta_0 + beta_1 * year[i] + beta_2 * year_sq[i];

    // Expected count on natural scale
    mu[i] = exp(log_mu_i);

    // Instantaneous growth rate at each time point
    instantaneous_growth_rate[i] = beta_1 + 2 * beta_2 * year[i];

    // Log-likelihood for LOO-CV
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | log_mu_i, phi);

    // Posterior predictive samples
    y_rep[i] = neg_binomial_2_log_rng(log_mu_i, phi);
  }
}
