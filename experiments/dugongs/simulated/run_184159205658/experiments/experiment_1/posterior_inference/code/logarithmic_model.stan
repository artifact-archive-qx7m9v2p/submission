data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor variable
  vector[N] Y;              // Response variable
}

parameters {
  real beta0;               // Intercept
  real beta1;               // Log slope coefficient
  real<lower=0> sigma;      // Error standard deviation
}

transformed parameters {
  vector[N] mu;             // Mean function
  for (i in 1:N) {
    mu[i] = beta0 + beta1 * log(x[i]);
  }
}

model {
  // Priors
  beta0 ~ normal(1.73, 0.5);
  beta1 ~ normal(0.28, 0.15);
  sigma ~ exponential(5);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;        // Pointwise log-likelihood for LOO-CV
  vector[N] Y_rep;          // Posterior predictive samples

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
    Y_rep[i] = normal_rng(mu[i], sigma);
  }
}
