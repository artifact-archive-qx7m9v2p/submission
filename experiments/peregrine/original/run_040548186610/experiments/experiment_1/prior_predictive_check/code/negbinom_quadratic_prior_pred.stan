// Negative Binomial Quadratic Model - Prior Predictive Check
// Purpose: Sample from priors and generate synthetic data to validate model specification

data {
  int<lower=1> N;              // Number of observations
  vector[N] year;              // Centered year values
  int<lower=0,upper=1> prior_only;  // Flag: 1 = prior predictive, 0 = full model
}

parameters {
  real beta_0;      // Intercept
  real beta_1;      // Linear coefficient
  real beta_2;      // Quadratic coefficient
  real<lower=0> phi;  // Overdispersion parameter
}

model {
  // Priors
  beta_0 ~ normal(4.7, 0.5);
  beta_1 ~ normal(0.8, 0.3);
  beta_2 ~ normal(0.3, 0.2);
  phi ~ gamma(2, 0.5);

  // Likelihood (only if not prior_only mode)
  if (prior_only == 0) {
    // This section will be used later for full model fitting
    // For now, we're only sampling from priors
  }
}

generated quantities {
  // Generate prior predictive samples
  array[N] int y_sim;           // Simulated counts
  vector[N] mu;                  // Expected counts
  vector[N] log_mu;              // Log expected counts

  // Calculate expected counts
  for (i in 1:N) {
    log_mu[i] = beta_0 + beta_1 * year[i] + beta_2 * year[i]^2;
    mu[i] = exp(log_mu[i]);

    // Generate simulated data from negative binomial
    // Using alternative parameterization: mu and phi
    // where variance = mu + mu^2/phi
    y_sim[i] = neg_binomial_2_rng(mu[i], phi);
  }
}
