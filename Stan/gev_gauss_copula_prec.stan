functions {
  #include gev.stan
  #include copula.stan
}

data {
  int<lower = 0> n_replicate;
  int<lower = 0> n_id;
  int<lower = 0> n_params;
  
  array[n_id] int<lower = 0> id;
  
  array[n_replicate, n_id] real y;
  array[n_replicate, n_id] real y_test;
}

parameters {
  vector<lower = 0>[n_params] mu;
  vector<lower = 0>[n_params] sigma;
  vector<lower = -0.5, upper = 1>[n_params] xi;
  real<lower = -1, upper = 1> rho;
}


model {
  matrix[n_id, n_id] Omega = AR1_precision_matrix(n_id, rho);
  for (i in 1:n_replicate) {
    vector[n_id] U;
    for (j in 1:n_id) {
      U[j] = gev_cdf(y[i, j] | mu[id[j]], sigma[id[j]], xi[id[j]]);
      target += gev_lpdf(y[i, j] | mu[id[j]], sigma[id[j]], xi[id[j]]);
    }
    target += normal_copula_prec_lpdf(U | Omega);
  }
}

generated quantities {
  real log_lik = 0;
  {
    matrix[n_id, n_id] Omega = AR1_precision_matrix(n_id, rho);
    
    for (i in 1:n_replicate) {
      vector[n_id] U;
      for (j in 1:n_id) {
        U[j] = gev_cdf(y_test[i, j] | mu[id[j]], sigma[id[j]], xi[id[j]]);
        log_lik += gev_lpdf(y_test[i, j] | mu[id[j]], sigma[id[j]], xi[id[j]]);
      }
      log_lik += normal_copula_prec_lpdf(U | Omega);
    }
  }
  
}