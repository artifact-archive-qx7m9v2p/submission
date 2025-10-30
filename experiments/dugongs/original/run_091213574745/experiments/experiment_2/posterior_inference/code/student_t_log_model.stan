// Logarithmic Model with Student-t Likelihood
// Y_i ~ StudentT(ν, μ_i, σ)
// μ_i = β₀ + β₁*log(x_i)
//
// Key: ν is truncated at 3 for numerical stability
// If ν > 30, Normal likelihood is adequate (prefer Model 1)

data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor (dose)
  vector[N] Y;              // Response (log-transformed effect)
}

transformed data {
  vector[N] log_x;
  log_x = log(x);
}

parameters {
  real beta_0;              // Intercept
  real beta_1;              // Slope for log(x)
  real<lower=0> sigma;      // Scale parameter
  real<lower=3> nu;         // Degrees of freedom (truncated at 3)
}

model {
  // Priors
  beta_0 ~ normal(2.3, 0.5);
  beta_1 ~ normal(0.29, 0.15);
  sigma ~ exponential(10);
  nu ~ gamma(2, 0.1);       // Will be truncated at 3

  // Likelihood
  vector[N] mu;
  mu = beta_0 + beta_1 * log_x;
  Y ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] log_lik;        // Log likelihood for LOO-CV
  vector[N] y_pred;         // Posterior predictive mean
  vector[N] y_rep;          // Posterior predictive draws

  {
    vector[N] mu;
    mu = beta_0 + beta_1 * log_x;

    for (n in 1:N) {
      log_lik[n] = student_t_lpdf(Y[n] | nu, mu[n], sigma);
      y_pred[n] = mu[n];
      y_rep[n] = student_t_rng(nu, mu[n], sigma);
    }
  }
}
