// Logarithmic Model with Normal Likelihood
// Y_i ~ Normal(μ_i, σ)
// μ_i = β₀ + β₁*log(x_i)

data {
  int<lower=1> N;           // Number of observations
  vector[N] x;              // Predictor variable
  vector[N] Y;              // Response variable
}

parameters {
  real beta_0;              // Intercept
  real beta_1;              // Slope for log(x)
  real<lower=0> sigma;      // Residual standard deviation
}

transformed parameters {
  vector[N] mu;             // Mean function

  for (i in 1:N) {
    mu[i] = beta_0 + beta_1 * log(x[i]);
  }
}

model {
  // Priors
  beta_0 ~ normal(2.3, 0.3);
  beta_1 ~ normal(0.29, 0.15);
  sigma ~ exponential(10);

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  // Pointwise log-likelihood for LOO-CV
  vector[N] log_lik;

  // Posterior predictive samples (using same mu)
  vector[N] y_pred;

  // Replicated data for posterior predictive checks
  vector[N] y_rep;

  for (i in 1:N) {
    // Log-likelihood for each observation
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);

    // Posterior predictions (deterministic)
    y_pred[i] = mu[i];

    // Replicated data (with noise)
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}
