// Logarithmic Regression Model
// Y = alpha + beta * log(x) + epsilon
// Hypothesis: Unbounded slow growth (Weber-Fechner law)

data {
  int<lower=0> N;               // Number of observations
  vector[N] x;                  // Predictor (must be positive)
  vector[N] Y;                  // Response
}

transformed data {
  vector[N] log_x;
  log_x = log(x);               // Precompute log(x) once
}

parameters {
  real alpha;                   // Intercept (Y when x=1, since log(1)=0)
  real beta;                    // Log-slope (change in Y per unit log(x))
  real<lower=0> sigma;          // Residual standard deviation
}

model {
  // Priors (weakly informative, centered on EDA estimates)
  alpha ~ normal(1.75, 0.5);    // Centered at OLS estimate ~1.75
  beta ~ normal(0.27, 0.15);    // Centered at OLS estimate ~0.27, mostly positive
  sigma ~ normal(0, 0.2);       // Half-normal via constraint, scale ~0.1-0.2

  // Likelihood
  Y ~ normal(alpha + beta * log_x, sigma);
}

generated quantities {
  vector[N] mu;                 // Mean predictions at observed x
  vector[N] y_rep;              // Posterior predictive samples
  vector[N] log_lik;            // Pointwise log-likelihood for LOO-CV

  // For posterior predictive checks
  real mean_y_rep;
  real sd_y_rep;
  real min_y_rep;
  real max_y_rep;

  for (i in 1:N) {
    mu[i] = alpha + beta * log_x[i];
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }

  // Summary statistics for PPC
  mean_y_rep = mean(y_rep);
  sd_y_rep = sd(y_rep);
  min_y_rep = min(y_rep);
  max_y_rep = max(y_rep);
}
