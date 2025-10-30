// Negative Binomial GLM with Quadratic Trend
// Experiment 1: Simulation-Based Validation

data {
  int<lower=0> N;               // Number of observations
  vector[N] year;               // Standardized year values
  array[N] int<lower=0> C;      // Count data
}

parameters {
  real beta_0;                  // Intercept on log scale
  real beta_1;                  // Linear trend coefficient
  real beta_2;                  // Quadratic trend coefficient
  real<lower=0> phi;            // Dispersion parameter (reciprocal parameterization)
}

transformed parameters {
  vector[N] mu;                 // Expected count on natural scale
  vector[N] log_mu;             // Log of expected count

  // Compute log-mean
  log_mu = beta_0 + beta_1 * year + beta_2 * square(year);

  // Transform to natural scale
  mu = exp(log_mu);
}

model {
  // Priors
  beta_0 ~ normal(4.5, 1.0);
  beta_1 ~ normal(0.9, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ gamma(2, 0.1);

  // Likelihood: NegativeBinomial2(mu, phi)
  // NegativeBinomial2 parameterization: mean=mu, variance=mu + mu^2/phi
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;            // Log-likelihood for LOO-CV
  array[N] int C_rep;           // Posterior predictive draws

  // Compute log-likelihood for each observation
  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_lpmf(C[n] | mu[n], phi);
    C_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
