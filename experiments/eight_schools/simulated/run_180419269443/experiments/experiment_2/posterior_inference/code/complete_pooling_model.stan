data {
  int<lower=0> N;           // number of studies
  vector[N] y;              // observed effects
  vector<lower=0>[N] sigma; // known standard errors
}

parameters {
  real mu;                  // common effect (complete pooling)
}

model {
  // Prior
  mu ~ normal(0, 50);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;        // pointwise log-likelihood for LOO
  vector[N] y_rep;          // posterior predictive samples

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu, sigma[i]);
    y_rep[i] = normal_rng(mu, sigma[i]);
  }
}
