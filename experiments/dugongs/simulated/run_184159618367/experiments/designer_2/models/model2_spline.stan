// Hierarchical B-Spline Regression Model
// Designer 2 - Model 2
//
// Smooth nonlinear regression using B-spline basis functions
// Hierarchical shrinkage on coefficients controls smoothness

data {
  int<lower=1> N;           // Number of observations
  int<lower=1> K;           // Number of B-spline basis functions
  vector[N] Y;              // Response
  matrix[N, K] B;           // B-spline basis matrix (pre-computed)
}

parameters {
  vector[K] beta;           // Spline coefficients
  real<lower=0> tau;        // Hierarchical SD (controls smoothness)
  real<lower=0> sigma;      // Residual SD
}

transformed parameters {
  vector[N] mu;

  // Linear combination of basis functions
  mu = B * beta;
}

model {
  // Hierarchical prior on spline coefficients
  beta ~ normal(0, tau);

  // Priors on hyperparameters
  tau ~ cauchy(0, 0.5);     // Regularization on coefficient variation
  sigma ~ cauchy(0, 0.15);  // Half-Cauchy via truncation

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;          // Posterior predictive samples
  vector[N] log_lik;        // Pointwise log-likelihood for LOO

  // Generate posterior predictive samples
  for (i in 1:N) {
    Y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }
}
