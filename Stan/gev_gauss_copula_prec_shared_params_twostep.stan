functions {
  #include gev.stan
  #include copula.stan
}

data {
  int<lower = 0> n_replicate;
  int<lower = 0> n_id;
  array[n_replicate, n_id] real y;
  array[n_replicate, n_id] real y_test;

  int<lower = 0> n_nonzero_chol_Q;
  real log_det_Q;
  array[n_id] int n_values;
  array[n_nonzero_chol_Q] int index;
  vector[n_nonzero_chol_Q] value;
}

parameters {
  real<lower = 0> mu;
  real<lower = 0> sigma;
  real<lower = -0.5, upper = 1> xi;
}

model {
  for (i in 1:n_replicate) {
    for (j in 1:n_id) {
      target += gev_lpdf(y[i, j] | mu, sigma, xi);
    }
  }
}
 
generated quantities {
  real log_lik = 0;
  {
    
    for (i in 1:n_replicate) {
      vector[n_id] U;
      for (j in 1:n_id) {
        U[j] = gev_cdf(y_test[i, j] | mu, sigma, xi);
        log_lik += gev_lpdf(y_test[i, j] | mu, sigma, xi);
      }
      log_lik += normal_copula_prec_chol_lpdf(U | n_values, index, value, log_det_Q);
    }
  }
  
}


