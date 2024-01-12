fit_threestep_model <- function(
    n_replicate,
    n_id,
    params,
    d,
    y,
    y_test) {
  Q <- make_AR_prec_matrix_1d(n_id = n_id, rho = 0.5)

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

  time_start <- Sys.time()

  stan_data <- list(
    n_replicate = n_replicate,
    n_id = nrow(params),
    y = y,
    y_test = y_test
  )

  inits <- y |>
    as.numeric() |>
    evd::fgev() |>
    purrr::pluck("estimate") |>
    as.list()




  names(inits) <- c("mu", "sigma", "xi")

  inits$xi <- pmin(inits$xi, 1)


  inits_iid <- list(
    inits
  )

  model_iid <- cmdstan_model(here("Stan", "gev_shared_params.stan"))

  results_iid <- model_iid$optimize(
    data = stan_data,
    init = inits_iid
  )

  estimates <- results_iid$mle()


  mu <- estimates[1]
  sigma <- estimates[2]
  xi <- estimates[3]


  Z <- d |>
    select(replicate, id, y) |>
    mutate(
      U = evd::pgev(y, loc = mu, scale = sigma, shape = xi),
      Z = qnorm(U)
    ) |>
    select(replicate, id, Z) |>
    pivot_wider(names_from = id, values_from = Z) |>
    select(-replicate) |>
    as.matrix()

  stan_data_copula <- list(
    N_repetitions = n_replicate,
    N_nodes = n_id,
    N_edges = length(from),
    y = Z,
    from = from,
    to = to
  )

  model_copula <- cmdstan_model(here("Stan", "LM_precision.stan"))

  # results_copula <- model_copula$sample(
  #   data = stan_data_copula,
  #   chains = 4,
  #   parallel_chains = 4,
  #   iter_warmup = 1000,
  #   iter_sampling = 1000
  # )


  # estimates_copula <- results_copula |>
  #   summarise_draws() |>
  #   rename(value = mean)


  results_copula <- model_copula$optimize(
    data = stan_data_copula
  )
  estimates_copula <- results_copula$mle()
  estimates_copula <- tibble(
    variable = names(estimates_copula),
    value = estimates_copula
  )


  Q_offdiags <- estimates_copula |> # nolint
    filter( # nolint
      str_detect(variable, "^Q_offdiag")
    ) |>
    pull(value)

  tau <- estimates_copula |> # nolint
    filter( # nolint
      str_detect(variable, "^tau\\[")
    ) |>
    pull(value)


  Q_hat <- matrix( # nolint
    0,
    nrow = n_id,
    ncol = n_id
  )

  for (i in 1:n_id) {
    Q_hat[i, i] <- tau[i] # nolint
  }

  for (i in seq_along(from)) {
    Q_hat[from[i], to[i]] <- Q_offdiags[i] # nolint
    Q_hat[to[i], from[i]] <- Q_offdiags[i] # nolint
  }

  diag_pre <- sum(diag(Q_hat))

  while ("try-error" %in% class(try(chol(Q_hat)))) {
    Q_hat <- Q_hat + 0.01 * diag(n_id) # nolint
  }

  diag_post <- sum(diag(Q_hat))

  sigma_hat <- solve(Q_hat)

  D <- matrix( # nolint
    0,
    nrow = n_id,
    ncol = n_id
  )

  diag(D) <- sqrt(diag(sigma_hat)) # nolint

  Q_fin <- D %*% Q_hat %*% D # nolint

  chol_Q_fin <- chol(Q_fin) # nolint
  n_values <- numeric(nrow(chol_Q_fin))
  index <- c()
  value <- c()
  for (i in seq_len(nrow(chol_Q_fin))) {
    for (j in seq_len(ncol(chol_Q_fin))) {
      if (abs(chol_Q_fin[i, j]) > 0) {
        n_values[i] <- n_values[i] + 1
        index <- c(index, j)
        value <- c(value, chol_Q_fin[i, j])
      }
    }
  }

  log_det_Q <- sum(log(diag(chol_Q_fin))) # nolint



  stan_data_gev_copula <- list(
    n_replicate = n_replicate,
    n_id = n_id,
    y = y,
    y_test = y_test,
    n_nonzero_chol_Q = sum(n_values),
    n_values = n_values,
    index = index,
    value = value,
    log_det_Q = log_det_Q
  )

  model_gev_copula <- cmdstan_model( # nolint
    here("Stan", "gev_gauss_copula_prec_shared_params.stan") # nolint
  )


  results_gev_copula <- model_gev_copula$sample(
    data = stan_data_gev_copula,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    init = list(inits, inits, inits, inits)
  )

  time_stop <- Sys.time()

  total_time <- as.numeric(time_stop - time_start, units = "secs")

  results_gev_copula$draws(
    variables = c("mu", "sigma", "xi", "log_lik")
  ) |>
    summarise_draws() |> # nolint
    mutate( # nolint
      model = "three step",
      time = total_time,
      Q_diag_mult = diag_post / diag_pre
    )
}
