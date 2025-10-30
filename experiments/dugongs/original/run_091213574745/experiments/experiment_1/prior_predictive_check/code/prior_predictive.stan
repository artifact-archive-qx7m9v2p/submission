// Prior Predictive Model for Logarithmic Model
// Samples from priors and generates synthetic datasets

data {
  int<lower=1> N;        // Number of observations
  vector[N] x;           // Predictor values (actual x from data)
}

generated quantities {
  // Sample from priors
  real beta_0 = normal_rng(2.3, 0.3);
  real beta_1 = normal_rng(0.29, 0.15);
  real<lower=0> sigma = exponential_rng(10);

  // Generate synthetic data
  vector[N] mu;
  vector[N] y_sim;

  for (i in 1:N) {
    mu[i] = beta_0 + beta_1 * log(x[i]);
    y_sim[i] = normal_rng(mu[i], sigma);
  }

  // Summary statistics for quick checks
  real y_min = min(y_sim);
  real y_max = max(y_sim);
  real y_mean = mean(y_sim);
  real y_sd = sd(y_sim);
}
