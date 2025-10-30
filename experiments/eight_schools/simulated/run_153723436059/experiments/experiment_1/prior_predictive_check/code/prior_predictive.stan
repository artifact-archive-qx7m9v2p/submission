// Prior Predictive Check Model for Eight Schools
// Generates synthetic data from the prior distribution only

data {
  int<lower=0> J;              // number of schools
  vector[J] sigma;             // known standard errors
}

generated quantities {
  real mu;                     // population mean
  real<lower=0> tau;           // between-school SD
  vector[J] theta;             // school effects
  vector[J] y_sim;             // simulated observations

  // Sample from priors
  mu = normal_rng(0, 50);
  tau = fabs(cauchy_rng(0, 25));  // Half-Cauchy via absolute value

  // Generate school effects and observations
  for (j in 1:J) {
    theta[j] = normal_rng(mu, tau);
    y_sim[j] = normal_rng(theta[j], sigma[j]);
  }
}
