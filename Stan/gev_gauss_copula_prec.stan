functions {
#include gev.stan
#include copula.stan
}

data {
  int<lower = 0> n_replicate;
  int<lower = 0> n_id;
  array[n_replicate, n_id] real y;
}

parameters {
  vector<lower = 0>[n_id] mu;
  vector<lower = 0>[n_id] sigma;
  vector<lower = 0, upper = 0.5>[n_id] xi;
  real<lower = -1, upper = 1> rho;
}

model {
 matrix[n_id, n_id] Omega = AR1_precision_matrix(n_id, rho);
  for (i in 1:n_replicate) {
    vector[n_id] U;
    for (j in 1:n_id) {
      U[j] = gev_cdf(y[i, j] | mu[j], sigma[j], xi[j]);
      target += gev_lpdf(y[i, j] | mu[j], sigma[j], xi[j]);
    }
    target += normal_copula_prec_lpdf(U | Omega);
  }
}
