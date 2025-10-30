// 3-component finite mixture model
// Uses marginalized likelihood for efficient sampling

data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // trials per group
  array[J] int<lower=0> r;           // successes per group
}

parameters {
  simplex[3] pi;                     // mixing proportions
  ordered[3] mu;                     // cluster means (ordered to avoid label switching)
  vector<lower=0>[3] sigma;          // cluster SDs
  vector[J] theta;                   // group-level parameters
}

model {
  // Priors
  pi ~ dirichlet(rep_vector(1.0, 3));
  mu ~ normal(-2.6, 1.0);
  sigma ~ normal(0, 0.5);  // Half-normal via constraint

  // Marginalized mixture likelihood for theta
  for (j in 1:J) {
    vector[3] log_pi_k = log(pi);
    for (k in 1:3) {
      log_pi_k[k] += normal_lpdf(theta[j] | mu[k], sigma[k]);
    }
    target += log_sum_exp(log_pi_k);
  }

  // Data likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Log likelihood for LOO-CV
  vector[J] log_lik;

  // Cluster assignment probabilities
  matrix[J, 3] cluster_probs;

  // Posterior predictive samples
  array[J] int r_rep;

  // Compute for each group
  for (j in 1:J) {
    // Log likelihood
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);

    // Cluster probabilities
    vector[3] log_probs;
    for (k in 1:3) {
      log_probs[k] = log(pi[k]) + normal_lpdf(theta[j] | mu[k], sigma[k]);
    }
    cluster_probs[j, :] = softmax(log_probs)';

    // Posterior predictive
    r_rep[j] = binomial_rng(n[j], inv_logit(theta[j]));
  }
}
