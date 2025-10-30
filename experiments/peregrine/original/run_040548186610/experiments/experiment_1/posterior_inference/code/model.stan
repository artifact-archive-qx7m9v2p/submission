// Negative Binomial Quadratic Model
// SBC-validated specification

data {
  int<lower=0> N;                    // Number of observations
  array[N] int<lower=0> C;           // Count observations
  vector[N] year;                     // Standardized year predictor
}

parameters {
  real beta_0;                        // Intercept
  real beta_1;                        // Linear coefficient
  real beta_2;                        // Quadratic coefficient
  real<lower=0> phi;                  // Dispersion parameter
}

transformed parameters {
  vector[N] mu;                       // Expected counts

  // Quadratic model on log scale
  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i] + beta_2 * year[i]^2);
  }
}

model {
  // Priors (validated via SBC)
  beta_0 ~ normal(4.7, 0.3);
  beta_1 ~ normal(0.8, 0.2);
  beta_2 ~ normal(0.3, 0.1);
  phi ~ gamma(2, 0.5);

  // Likelihood: Negative Binomial parameterization
  // Stan uses alternative parameterization: neg_binomial_2(mu, phi)
  // where phi is the "reciprocal dispersion" (larger = less dispersed)
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  // Log-likelihood for LOO-CV (required for ArviZ)
  vector[N] log_lik;

  // Posterior predictive samples
  array[N] int C_rep;

  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
  }
}
