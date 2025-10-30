data {
  int<lower=0> N;              // Number of observations
  vector[N] x;                 // Predictor
  vector[N] y;                 // Response
}

parameters {
  real<lower=0> alpha;         // Asymptote
  real<lower=0> beta;          // Amplitude
  real<lower=0> gamma;         // Rate parameter
  real<lower=0> sigma;         // Residual SD
}

transformed parameters {
  vector[N] mu;                // Mean function
  for (i in 1:N) {
    mu[i] = alpha - beta * exp(-gamma * x[i]);
  }
}

model {
  // Priors
  alpha ~ normal(2.55, 0.1);
  beta ~ normal(0.9, 0.2);
  gamma ~ gamma(4, 20);        // E[gamma] = 4/20 = 0.2
  sigma ~ cauchy(0, 0.15);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;           // For LOO-CV
  vector[N] y_rep;             // Posterior predictive draws

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}
