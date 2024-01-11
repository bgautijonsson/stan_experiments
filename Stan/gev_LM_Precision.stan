data {
  int N_repetitions;
  int N_nodes;
  int N_edges;
  
  matrix[N_repetitions, N_nodes] y;
  array[N_edges] int<lower = 1, upper = N_nodes> from;
  array[N_edges] int<lower = 1, upper = N_nodes> to;
  
  real eps;
}

parameters {
  vector[N_edges] beta;
  vector<lower = 0>[N_nodes] tau;
  real<lower = 0> mu;
  real<lower = 0> sigma;
  real<lower = -0.5, upper = 1> xi;
  real<lower = -1, upper = 1> rho;
}

transformed parameters {
  vector<lower = 0>[N_nodes] sigma2 = pow(tau, -1);
  
}

model {
  matrix[N_repetitions, N_nodes] y_hat = rep_matrix(0, N_repetitions, N_nodes);
  for (i in 1:N_edges) {
    y_hat[ , to[i]] += -sigma2[to[i]] * beta[i] * y[ , from[i]];
    y_hat[ , from[i]] += -sigma2[from[i]] * beta[i] * y[ , to[i]];
  }
  
  for (i in 1:N_nodes) {
    y[ , i] ~ normal(y_hat[ , i], sqrt(sigma2[i]));
  }
}

generated quantities {
  
}
