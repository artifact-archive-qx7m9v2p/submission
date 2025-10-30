// Bayesian Change-Point Regression Model
// Designer 2 - Model 1
//
// Piecewise linear model with inferred breakpoint location
// Continuous at breakpoint but with different slopes before/after

data {
  int<lower=1> N;           // Number of observations
  vector[N] x;              // Predictor
  vector[N] Y;              // Response
}

parameters {
  real tau;                 // Breakpoint location
  real beta0;               // Intercept
  real beta1;               // Slope before breakpoint
  real beta2;               // Slope after breakpoint
  real<lower=0> sigma;      // Residual SD
}

transformed parameters {
  vector[N] mu;

  // Continuous piecewise linear function
  for (i in 1:N) {
    if (x[i] <= tau) {
      mu[i] = beta0 + beta1 * x[i];
    } else {
      mu[i] = beta0 + beta1 * tau + beta2 * (x[i] - tau);
    }
  }
}

model {
  // Priors
  tau ~ normal(9.5, 2.0);           // Breakpoint centered at EDA finding
  beta0 ~ normal(1.7, 0.2);         // Intercept near minimum Y
  beta1 ~ normal(0.08, 0.03);       // Positive slope in active regime
  beta2 ~ normal(0.0, 0.02);        // Near-zero slope in saturated regime
  sigma ~ cauchy(0, 0.15);          // Half-Cauchy via truncation

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;                  // Posterior predictive samples
  vector[N] log_lik;                // Pointwise log-likelihood for LOO

  // Generate posterior predictive samples
  for (i in 1:N) {
    Y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }
}
