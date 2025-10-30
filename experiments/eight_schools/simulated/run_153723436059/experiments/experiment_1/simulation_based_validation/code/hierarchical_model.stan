// Hierarchical Model for Eight Schools (Non-centered Parameterization)
// Used for inference (fitting to data)

data {
  int<lower=0> J;              // number of schools
  vector[J] y;                 // observed treatment effects
  vector<lower=0>[J] sigma;    // known standard errors
}

parameters {
  real mu;                     // population mean
  real<lower=0> tau;           // between-school SD
  vector[J] theta_raw;         // raw school effects (standard normal)
}

transformed parameters {
  // Non-centered parameterization: theta = mu + tau * theta_raw
  vector[J] theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ cauchy(0, 25);          // Half-Cauchy via lower=0 constraint
  theta_raw ~ normal(0, 1);     // Standard normal for non-centered

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Posterior predictive
  vector[J] y_rep;
  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
  }
}
