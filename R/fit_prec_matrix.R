library(tidyverse)
library(here)
library(rstan)
library(posterior)

here("R", "data.R") |> source()

n_id <- 10
n_replicates <- 30
rho <- 0.5

d <- params |> 
  make_data(n_replicate = n_replicate, rho = rho)


y <- d |> 
  select(replicate, id, y) |> 
  pivot_wider(names_from = id, values_from = y) |> 
  select(-replicate) |> 
  as.matrix()

y_test <- d |> 
  select(replicate, id, y) |> 
  pivot_wider(names_from = id, values_from = y) |> 
  select(-replicate) |> 
  as.matrix()

stan_data <- list(
  n_replicate = n_replicate,
  n_id = nrow(params),
  
  y = y,
  y_test = y_test
)

inits <- y |> as.numeric() |> evd::fgev() |> purrr::pluck("estimate") |> as.list()


Q <- make_AR_prec_matrix_1d(n_id = n_id, rho = rho)
R <- make_AR_cor_matrix_1d(n_id = n_id, rho = rho)

X <- sample_gaussian_variables(cor_matrix = R, n_replicates = n_replicates)

from <- c()
to <- c()

for (i in 1:n_id) {
  for (j in i:n_id) {
    if (j > i) {
      if (Q[i, j] != 0) {
        from <- c(from, i)
        to <- c(to, j)
      }
    }
  }
}

n_edges <- length(from)
type <- numeric(n_id)
type[] <- 1
type[c(1, n_id)] <- 2




model <- stan_model(here("Stan", "gev_LM_Precision.stan"))

stan_data <- list(
  N_nodes = n_id,
  N_repetitions = n_replicates,
  N_edges = n_edges,
  y = y,
  from = from,
  to = to,
  type = type,
  eps = 0
)

init <- list(
  beta = rep(-0.1, n_edges),
  tau = rep(1, n_id)
)

res <- optimizing(
  model,
  data = stan_data,
  init = init
)

theta <- res$theta_tilde

beta <- theta[str_detect(colnames(theta), "beta")]

tau <- theta[str_detect(colnames(theta), "tau")]



Q_hat <- matrix(
  0,
  nrow = n_id,
  ncol = n_id
)

for (i in 1:n_id) {
  Q_hat[i, i] <- tau[i]
}

for (i in 1:n_edges) {
  Q_hat[from[i], to[i]] <- beta[i]
  Q_hat[to[i], from[i]] <- beta[i]
}


# Q_hat
sigma <- solve(Q_hat)

D <- matrix(
  0,
  nrow = n_id,
  ncol = n_id
  )

diag(D) <- sqrt(diag(sigma))

Q_fin <- D %*% Q_hat %*% D


mean(sqrt((Q[abs(Q) > 0] - Q_fin[abs(Q_fin) > 0])^2))

