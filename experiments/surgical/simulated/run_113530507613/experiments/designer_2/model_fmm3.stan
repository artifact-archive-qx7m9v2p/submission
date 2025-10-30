// Finite Mixture Model (K=3 Components)
// Mixture of 3 normal distributions on logit scale
// Captures the three clusters identified in EDA (low, medium, high success rates)

data {
  int<lower=1> J;              // number of groups (12)
  int<lower=1> K;              // number of mixture components (3)
  array[J] int<lower=0> n;     // trials per group
  array[J] int<lower=0> r;     // successes per group
}

parameters {
  simplex[K] pi;               // mixture weights (probabilities)
  ordered[K] mu;               // cluster means (ordered for identifiability)
  vector<lower=0>[K] tau;      // cluster standard deviations
  vector[J] theta;             // group-level logit success rates
}

model {
  // Priors
  pi ~ dirichlet(rep_vector(2.0, K));  // weakly informative, allows imbalance
  mu ~ normal(-2.6, 1.5);              // centered on pooled rate (logit scale)
  tau ~ normal(0, 0.5);                // within-cluster variation

  // Mixture likelihood for group effects
  for (j in 1:J) {
    vector[K] log_pi_k;
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) + normal_lpdf(theta[j] | mu[k], tau[k]);
    }
    target += log_sum_exp(log_pi_k);
  }

  // Data likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior cluster probabilities for each group
  matrix[J, K] cluster_prob;

  // Posterior predictive samples
  array[J] int r_rep;

  // Cluster diagnostics
  int K_effective;             // number of active clusters (pi_k > 0.05)
  vector[K] cluster_size;      // number of groups per cluster

  // Success rates on probability scale
  vector[K] p_cluster;         // cluster-level success rates
  vector[J] p_group;           // group-level success rates

  for (j in 1:J) {
    // Log-likelihood
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);

    // Cluster assignment probabilities
    vector[K] log_pi_k;
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) + normal_lpdf(theta[j] | mu[k], tau[k]);
    }
    cluster_prob[j] = softmax(log_pi_k)';

    // Posterior predictive
    r_rep[j] = binomial_rng(n[j], inv_logit(theta[j]));

    // Group success rate
    p_group[j] = inv_logit(theta[j]);
  }

  // Count effective clusters
  K_effective = 0;
  for (k in 1:K) {
    if (pi[k] > 0.05) {
      K_effective += 1;
    }
  }

  // Compute cluster sizes (based on max probability assignment)
  cluster_size = rep_vector(0, K);
  for (j in 1:J) {
    int z_j = 0;
    real max_prob = 0;
    for (k in 1:K) {
      if (cluster_prob[j, k] > max_prob) {
        max_prob = cluster_prob[j, k];
        z_j = k;
      }
    }
    cluster_size[z_j] += 1;
  }

  // Cluster-level success rates
  for (k in 1:K) {
    p_cluster[k] = inv_logit(mu[k]);
  }
}
