// Mixture-of-Experts Model with Gating Network
// Designer 2 - Model 3
//
// Soft mixture of linear (active) and constant (saturated) experts
// Logistic gating function learns transition smoothness from data

data {
  int<lower=1> N;           // Number of observations
  vector[N] x;              // Predictor
  vector[N] Y;              // Response
}

parameters {
  real beta0;               // Linear expert: intercept
  real<lower=0> beta1;      // Linear expert: slope (constrained positive)
  real alpha;               // Constant expert: plateau level
  real gamma0;              // Gating network: location
  real<upper=0> gamma1;     // Gating network: slope (constrained negative)
  real<lower=0> sigma;      // Residual SD
}

transformed parameters {
  vector[N] mu1;            // Linear expert predictions
  vector[N] mu2;            // Constant expert predictions
  vector[N] pi_x;           // Gating weights
  vector[N] mu;             // Mixture predictions

  // Expert predictions
  mu1 = beta0 + beta1 * x;
  mu2 = rep_vector(alpha, N);

  // Gating function (probability of linear expert)
  pi_x = inv_logit(gamma0 + gamma1 * x);

  // Weighted mixture
  mu = pi_x .* mu1 + (1 - pi_x) .* mu2;
}

model {
  // Priors for linear expert
  beta0 ~ normal(1.7, 0.2);         // Intercept
  beta1 ~ normal(0.08, 0.03);       // Slope (truncated at 0)

  // Prior for constant expert
  alpha ~ normal(2.55, 0.1);        // Plateau level

  // Priors for gating network
  gamma0 ~ normal(0, 2);            // Location (unconstrained)
  gamma1 ~ normal(-0.5, 0.3);       // Slope (truncated at 0)

  // Prior for residual SD
  sigma ~ cauchy(0, 0.15);          // Half-Cauchy via truncation

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;                  // Posterior predictive samples
  vector[N] log_lik;                // Pointwise log-likelihood for LOO
  real tau_eff;                     // Effective breakpoint (where pi=0.5)

  // Effective breakpoint
  tau_eff = -gamma0 / gamma1;

  // Generate posterior predictive samples
  for (i in 1:N) {
    Y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }
}
