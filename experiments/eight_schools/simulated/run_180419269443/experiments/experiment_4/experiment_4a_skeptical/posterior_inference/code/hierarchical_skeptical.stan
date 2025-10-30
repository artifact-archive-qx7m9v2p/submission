// Hierarchical model with skeptical priors
// Skeptical of large effects, expects low heterogeneity

data {
  int<lower=0> J;              // number of studies
  vector[J] y;                 // estimated treatment effects
  vector<lower=0>[J] sigma;    // standard errors of effects
}

parameters {
  real mu;                     // population mean
  real<lower=0> tau;           // population SD
  vector[J] theta;             // study-specific effects
}

model {
  // Skeptical priors
  mu ~ normal(0, 10);          // Skeptical of large effects
  tau ~ normal(0, 5);          // Expects low heterogeneity (half-normal via constraint)

  // Hierarchical structure
  theta ~ normal(mu, tau);

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
