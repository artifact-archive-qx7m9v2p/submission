// Model 1: Michaelis-Menten Saturation Model
// Functional form: Y = Y_max * x / (K + x)
// Parameters: Y_max (asymptote), K (half-saturation), sigma (residual SD)

data {
  int<lower=0> N;           // Number of observations
  vector[N] x;              // Predictor values
  vector[N] Y;              // Response values
}

parameters {
  real<lower=0> Y_max;      // Asymptotic maximum
  real log_K;               // log(K) for better sampling
  real<lower=0> sigma;      // Residual standard deviation
}

transformed parameters {
  real<lower=0> K = exp(log_K);  // Half-saturation constant
  vector[N] mu;                   // Mean function

  for (i in 1:N) {
    mu[i] = Y_max * x[i] / (K + x[i]);
  }
}

model {
  // Priors (weakly informative)
  Y_max ~ normal(2.6, 0.3);      // Centered at observed max
  log_K ~ normal(log(5), 1);     // Implies K ~ lognormal, median ~5
  sigma ~ normal(0, 0.25);       // Half-normal via truncation

  // Likelihood
  Y ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive samples
  vector[N] Y_rep;
  vector[N] log_lik;  // For LOO-CV

  for (i in 1:N) {
    Y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(Y[i] | mu[i], sigma);
  }

  // Derived quantities for interpretation
  real Y_at_K = Y_max / 2;  // Y value at x=K (half-saturation)
}
