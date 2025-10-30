data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor values
  vector[N] Y;              // Response values
}

parameters {
  real alpha;               // Intercept
  real beta;                // Slope for log term
  real<lower=0> c;          // Offset for logarithm
  real<lower=1> nu;         // Degrees of freedom for Student-t
  real<lower=0> sigma;      // Scale parameter
}

transformed parameters {
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = alpha + beta * log(x[i] + c);
  }
}

model {
  // Priors (revised post prior-predictive check)
  alpha ~ normal(2.0, 0.5);
  beta ~ normal(0.3, 0.2);
  c ~ gamma(2, 2);
  nu ~ gamma(2, 0.1);
  sigma ~ normal(0, 0.15);  // half-normal due to lower=0 constraint

  // Likelihood
  Y ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] Y_rep;          // Posterior predictive samples
  vector[N] log_lik;        // Log-likelihood for each observation

  for (i in 1:N) {
    Y_rep[i] = student_t_rng(nu, mu[i], sigma);
    log_lik[i] = student_t_lpdf(Y[i] | nu, mu[i], sigma);
  }
}
