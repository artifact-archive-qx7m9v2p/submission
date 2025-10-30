data {
  int<lower=1> N;           // Number of observations
  vector[N] x;              // Predictor variable
  vector[N] Y;              // Response variable
}

parameters {
  real alpha;               // Intercept
  real beta;                // Slope coefficient
  real<lower=0> c;          // Shift parameter for log transform
  real<lower=2> nu;         // Degrees of freedom for Student-t
  real<lower=0> sigma;      // Scale parameter
}

model {
  // Priors (validated via PPC and SBC)
  alpha ~ normal(2.0, 0.5);
  beta ~ normal(0.3, 0.2);
  c ~ gamma(2, 2);
  nu ~ gamma(2, 0.1);
  sigma ~ normal(0, 0.15);  // with lower=0 constraint

  // Likelihood
  for (i in 1:N) {
    real mu = alpha + beta * log(x[i] + c);
    Y[i] ~ student_t(nu, mu, sigma);
  }
}

generated quantities {
  vector[N] log_lik;        // Pointwise log-likelihood for LOO-CV
  vector[N] y_rep;          // Posterior predictive samples
  vector[N] mu;             // Mean predictions

  for (i in 1:N) {
    mu[i] = alpha + beta * log(x[i] + c);
    log_lik[i] = student_t_lpdf(Y[i] | nu, mu[i], sigma);
    y_rep[i] = student_t_rng(nu, mu[i], sigma);
  }
}
