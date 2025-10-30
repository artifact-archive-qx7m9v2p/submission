// Hierarchical Normal Model with Non-Centered Parameterization
// For 8 studies meta-analysis (schools problem structure)
//
// Non-centered parameterization avoids funnel geometry when tau is small
// This is critical for computational efficiency and avoiding divergences

data {
  int<lower=0> J;              // number of studies
  vector[J] y;                 // observed effects
  vector<lower=0>[J] sigma;    // known standard errors
}

parameters {
  real mu;                     // population mean
  real<lower=0> tau;           // population SD (between-study heterogeneity)
  vector[J] theta_raw;         // non-centered study effects
}

transformed parameters {
  vector[J] theta;             // centered study effects
  theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 25);
  tau ~ normal(0, 10);         // Half-normal via constraint in parameters block
  theta_raw ~ std_normal();    // Non-centered: N(0,1)

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // For posterior predictive checks
  vector[J] y_pred;
  for (j in 1:J) {
    y_pred[j] = normal_rng(theta[j], sigma[j]);
  }
}
