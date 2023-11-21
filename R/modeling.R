#' Fit a GEV model with a gaussian AR(1) copula to simulated data
#'
#' @export
fit_copula_model <- function(params, n_replicate, rho, output_dir) {
  here("R", "data.R") |> source()
  
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
  
  
  model_copula <- cmdstan_model(here("Stan", "gev_gauss_copula_prec_shared_params.stan"))
  
  results_copula <- model_copula$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    show_exceptions = F
  )
  
  out_copula <- results_copula$draws(variables = c("mu", "sigma", "xi", "rho", "log_lik")) |> 
    as_draws_df() |> 
    summarise_draws() |> 
    mutate(
      model = "copula"
    )
  

  
  model_iid <- cmdstan_model(here("Stan", "gev_shared_params.stan"))
  
  results_iid <- model_iid$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    show_exceptions = F
  )
  
  out_iid <- results_iid$draws(variables = c("mu", "sigma", "xi", "log_lik")) |> 
    as_draws_df() |> 
    summarise_draws() |> 
    mutate(
      model = "iid"
    )
  
  
  out_copula |> 
    bind_rows(
      out_iid
    ) |> 
    select(model, variable, posterior_mean = mean, rhat) |> 
    left_join(
      params |> 
        distinct(mu, sigma, xi) |> 
        pivot_longer(c(everything()), names_to = "variable", values_to = "true_value") |> 
        add_row(
          variable = "rho", true_value = rho
        ),
      by = join_by(variable)
    ) |> 
    mutate(
      n_replicate = n_replicate,
      n_id = nrow(params)
    ) |> 
    write_parquet(
      here(output_dir, "part-0.parquet")
    )
  
  
}

require(glue)
require(dplyr)
require(here)
require(tidyr)
require(cmdstanr)
require(readr)
require(posterior)
require(arrow)

i <- here("results", "Stan", "copula_data", "shared_params", "posterior") |> 
  list.files() |> 
  parse_number() |> 
  max()
i <- i + 1
# i <- 1

while (TRUE) {
  n_replicate <- sample(c(5, 10, 20, 40, 80, 160, 320), size = 1)
  n_id <- sample(c(5, 10, 20, 40, 80), size = 1)
  rho <- runif(n = 1) * rbinom(1, 1, 0.9)
  
  params <- crossing(
    mu = 6,
    sigma = 3,
    xi = 0.1
  ) |> 
    crossing(
      id = seq_len(n_id)
    )
  
  output_dir <- here("Results", "Stan", "copula_data", "shared_params", "posterior", glue("simulation={i}"))
  
  if (dir.exists(output_dir)) {
    here(output_dir, output_dir |> list.files()) |> file.remove()
  } else {
    dir.create(output_dir)
  }
  fit_copula_model(params = params, n_replicate = n_replicate, rho = rho, output_dir = output_dir)
  i <- i + 1
}
