// Logarithmic Model with Normal Likelihood
// Y_i ~ Normal(μ_i, σ)
// μ_i = β₀ + β₁*log(x_i)

data {
  int<lower=0> N;        // Number of observations
  vector[N] x;           // Predictor values
  vector[N] Y;           // Response values
}

parameters {
  real beta_0;           // Intercept
  real beta_1;           // Logarithmic slope
  real<lower=0> sigma;   // Residual standard deviation
}

model {
  // Priors
  beta_0 ~ normal(2.3, 0.3);
  beta_1 ~ normal(0.29, 0.15);
  sigma ~ exponential(10);

  // Likelihood
  Y ~ normal(beta_0 + beta_1 * log(x), sigma);
}

generated quantities {
  // Posterior predictive samples
  vector[N] Y_rep;
  // Log likelihood for LOO-CV
  vector[N] log_lik;

  for (i in 1:N) {
    real mu_i = beta_0 + beta_1 * log(x[i]);
    Y_rep[i] = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu_i, sigma);
  }
}
