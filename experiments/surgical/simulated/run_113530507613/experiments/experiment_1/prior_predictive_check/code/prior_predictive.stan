// Prior Predictive Check for Hierarchical Logit-Normal Model
// This model samples ONLY from the priors (no data conditioning)

data {
  int<lower=1> J;                  // number of groups
  array[J] int<lower=1> n;        // number of trials per group
}

generated quantities {
  // Hyperparameters (sampled from priors)
  real mu = normal_rng(-2.6, 1.0);           // population mean (logit scale)
  real<lower=0> tau = fabs(normal_rng(0, 0.5));  // between-group SD (half-normal)

  // Group-level parameters (non-centered parameterization)
  array[J] real theta_raw;
  array[J] real theta;
  array[J] real<lower=0, upper=1> p;

  // Simulated data
  array[J] int r_sim;

  for (j in 1:J) {
    theta_raw[j] = normal_rng(0, 1);
    theta[j] = mu + tau * theta_raw[j];
    p[j] = inv_logit(theta[j]);
    r_sim[j] = binomial_rng(n[j], p[j]);
  }
}
