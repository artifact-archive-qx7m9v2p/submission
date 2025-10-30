// Prior Predictive Model for Negative Binomial State-Space
// This version samples ONLY from priors (no data likelihood)

data {
  int<lower=1> N;  // Number of time points
}

generated quantities {
  // Sample from priors
  real delta = normal_rng(0.05, 0.02);
  real<lower=0> sigma_eta = exponential_rng(10);
  real<lower=0> phi = exponential_rng(0.1);

  // Sample initial state from prior
  real eta_1 = normal_rng(log(50), 1);

  // Generate latent state trajectory
  vector[N] eta;
  eta[1] = eta_1;

  for (t in 2:N) {
    eta[t] = normal_rng(eta[t-1] + delta, sigma_eta);
  }

  // Generate prior predictive counts
  array[N] int<lower=0> C_prior;
  for (t in 1:N) {
    C_prior[t] = neg_binomial_2_log_rng(eta[t], phi);
  }

  // Summary statistics for diagnostics
  real prior_mean_count = mean(to_vector(C_prior));
  real prior_max_count = max(C_prior);
  real prior_min_count = min(C_prior);

  // Growth from first to last observation
  real prior_growth_factor = exp(eta[N] - eta[1]);

  // Total change in log-scale
  real total_log_change = eta[N] - eta[1];
}
