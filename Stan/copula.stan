real normal_ar1_lpdf(vector x, real rho) {
    int N = num_elements(x);
    real out;
    real log_det = - (N - 1) * (log(1 + rho) + log(1 - rho)) / 2;
    vector[N] q;
    real scl = sqrt(1 / (1 - rho^2));
    
    q[1:(N - 1)] = scl * (x[1:(N - 1)] - rho * x[2:N]);
    q[N] = x[N];
    
    out = log_det - dot_self(q) / 2;
    
    return out;
  }
  
  real normal_copula_ar1_lpdf(vector U, real rho) {
    int N = rows(U);
    vector[N] Z = inv_Phi(U);
    return normal_ar1_lpdf(Z | rho) + dot_self(Z) / 2;
  }

real normal_copula_lpdf(vector U, matrix L) {
  int N = rows(U);
  vector[N] Z = inv_Phi(U);
  return multi_normal_cholesky_lpdf(Z | rep_vector(0, N), L) - std_normal_lpdf(Z);
}

real normal_copula_prec_lpdf(vector U, matrix L) {
  int N = rows(U);
  vector[N] Z = inv_Phi(U);
  vector[N] mu = rep_vector(0, N);
  matrix[N, N] I = diag_matrix(rep_vector(1, N));
  return multi_normal_prec_lpdf(Z | mu, L) - multi_normal_prec_lpdf(Z | mu, I);
}

matrix AR1_precision_matrix(int n_id, real rho) {
  real rho_scaler = pow(1 - rho^2, -1);
  real diag_inside = rho^2 * rho_scaler;
  real off_diag = -rho * rho_scaler;
  matrix[n_id, n_id] Omega = diag_matrix(rep_vector(rho_scaler, n_id));
  Omega[2:(n_id - 1), 2:(n_id - 1)] = add_diag(Omega[2:(n_id - 1), 2:(n_id - 1)], diag_inside);
  for (i in 1:n_id) {
    if (i > 1) {
      Omega[i, i - 1] = off_diag;
    }
    if (i < n_id) {
      Omega[i, i + 1] = off_diag;
    }
  }
  return Omega;
}

real icar_copula_lpdf(vector U, int N_edges, array[] int node1, array[] int node2) {
  int N = rows(U);
  vector[N] Z = inv_Phi(U);
  return icar_normal_lpdf(Z | N_edges, node1, node2) - std_normal_lpdf(Z);
}

real icar_normal_lpdf(vector x, int N, array[] int node1, array[] int node2) {
  return  -0.5 * (dot_self((x[node1] - x[node2]))) +
  normal_lpdf(sum(x) | 0, 0.001 * N);
}