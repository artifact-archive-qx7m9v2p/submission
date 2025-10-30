
data {
  int<lower=1> N;              // Number of groups
  vector[N] y;                 // Observed values
  vector<lower=0>[N] sigma;    // Known measurement errors
}

parameters {
  real mu;                     // Population mean
  real<lower=0> tau;           // Between-group SD
  vector[N] theta_raw;         // Non-centered group effects
}

transformed parameters {
  vector[N] theta;             // Group-specific means
  theta = mu + tau * theta_raw;
}

model {
  // Hyperpriors
  mu ~ normal(10, 20);
  tau ~ normal(0, 10);         // Half-normal via lower=0 constraint

  // Non-centered parameterization
  theta_raw ~ std_normal();

  // Likelihood (known measurement error)
  y ~ normal(theta, sigma);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[N] log_lik;

  // Posterior predictive for each observation
  vector[N] y_rep;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | theta[i], sigma[i]);
    y_rep[i] = normal_rng(theta[i], sigma[i]);
  }
}
