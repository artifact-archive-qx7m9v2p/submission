// Beta-Binomial Prior Predictive Model
// Generates synthetic data from priors without conditioning on observed data

data {
  int<lower=1> N;              // Number of trials
  array[N] int<lower=0> n;     // Sample sizes for each trial
}

generated quantities {
  // Sample from priors
  real<lower=0, upper=1> mu = beta_rng(2, 25);    // Mean success probability
  real<lower=0> phi = gamma_rng(2, 2);             // Concentration parameter

  // Transform to shape parameters
  real<lower=0> alpha = mu * phi;
  real<lower=0> beta_param = (1 - mu) * phi;

  // Generate prior predictive data
  array[N] int<lower=0> r_prior;
  array[N] real<lower=0, upper=1> prop_prior;

  for (i in 1:N) {
    r_prior[i] = beta_binomial_rng(n[i], alpha, beta_param);
    prop_prior[i] = r_prior[i] * 1.0 / n[i];
  }

  // Summary statistics for checking
  int total_successes = sum(r_prior);
  real mean_proportion = mean(prop_prior);
  real sd_proportion = sd(prop_prior);

  // Compute empirical overdispersion (approximation)
  // For Beta-Binomial, theoretical variance factor is 1/(phi+1)
  real variance_inflation = sd_proportion * sd_proportion / (mu * (1 - mu));
}
