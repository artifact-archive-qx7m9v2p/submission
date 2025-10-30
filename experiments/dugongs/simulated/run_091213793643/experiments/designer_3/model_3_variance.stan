// Model 3: Compositional Variance Model
// Structure: Y ~ Normal(μ(x), σ(x)) with location-scale modeling
// Designer 3 - Hierarchical/Compositional Perspective

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] Y;
}

transformed data {
  vector[N] log_x = log(x);
}

parameters {
  real alpha;
  real<lower=0> beta;
  real gamma_0;  // Log-variance intercept
  real gamma_1;  // Log-variance slope (negative = decreasing variance)
}

transformed parameters {
  vector[N] mu;
  vector<lower=0>[N] sigma;

  for (n in 1:N) {
    mu[n] = alpha + beta * log_x[n];
    sigma[n] = exp(gamma_0 + gamma_1 * log_x[n]);
  }
}

model {
  // Priors
  alpha ~ normal(1.75, 0.5);
  beta ~ normal(0.27, 0.15);
  gamma_0 ~ normal(log(0.12), 0.5);  // Log-scale prior centered at ~0.12
  gamma_1 ~ normal(0, 0.3);          // Allow increase or decrease

  // Likelihood with heteroscedastic variance
  Y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] Y_rep;
  vector[N] log_lik;
  real variance_ratio;  // Ratio of variance at x=1 vs x=max

  // Compute variance ratio to quantify heteroscedasticity
  {
    real sigma_at_min = exp(gamma_0 + gamma_1 * log(min(x)));
    real sigma_at_max = exp(gamma_0 + gamma_1 * log(max(x)));
    variance_ratio = (sigma_at_min / sigma_at_max)^2;
  }

  for (n in 1:N) {
    Y_rep[n] = normal_rng(mu[n], sigma[n]);
    log_lik[n] = normal_lpdf(Y[n] | mu[n], sigma[n]);
  }
}
