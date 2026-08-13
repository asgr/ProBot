library(testthat)
library(ProBot)
library(torch)

# --- MDN training (minimal) ---
test_that("probotSingleEpochMDN runs without error", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochMDN(mdl, dl, opt, K)
  expect_true("loss" %in% names(metrics))
  expect_true("mae" %in% names(metrics))
  expect_true("rmse" %in% names(metrics))
  expect_true("sigma" %in% names(metrics))
  expect_true("mix" %in% names(metrics))
  expect_true(is.finite(metrics$loss))
})

test_that("probotSingleEpochMDN with lambda > 0 works", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochMDN(mdl, dl, opt, K, lambda = 0.5)
  expect_true(is.finite(metrics$loss))
})

test_that("probotTrainMDN completes and returns model and history", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  res <- probotTrainMDN(mdl, dl, opt, epochs = 3, mdn_components = K,
                         verbose = FALSE, early_stop = FALSE)
  expect_true(!is.null(res$model))
  expect_true(!is.null(res$history))
  expect_true(is.data.frame(res$history))
  expect_true("epoch" %in% names(res$history))
  expect_true("loss" %in% names(res$history))
})

# --- Point training ---
test_that("probotSingleEpochPoint runs without error", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochPoint(mdl, dl, opt)
  expect_true("loss" %in% names(metrics))
  expect_true("mae" %in% names(metrics))
  expect_true("rmse" %in% names(metrics))
  expect_true(is.finite(metrics$loss))
})

test_that("probotTrainPoint completes and returns model and history", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  res <- probotTrainPoint(mdl, dl, opt, epochs = 3, verbose = FALSE, early_stop = FALSE)
  expect_true(!is.null(res$model))
  expect_true(!is.null(res$history))
  expect_true(is.data.frame(res$history))
})

# --- Flow training ---
test_that("probotSingleEpochFlow runs without error", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  n <- 32

  # For flow: input=batch[[1]]=x (context), output=batch[[2]]=theta
  x <- matrix(rnorm(n * dim_x), n, dim_x)
  theta <- matrix(rnorm(n * dim_theta), n, dim_theta)

  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")

  metrics <- probotSingleEpochFlow(mdl, dl, opt)
  expect_true("loss" %in% names(metrics))
  expect_true(is.finite(metrics$loss))
})

test_that("probotTrainFlow completes and returns model and history", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  n <- 32

  x <- matrix(rnorm(n * dim_x), n, dim_x)
  theta <- matrix(rnorm(n * dim_theta), n, dim_theta)

  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")

  res <- probotTrainFlow(mdl, dl, opt, epochs = 3, verbose = FALSE, early_stop = FALSE)
  expect_true(!is.null(res$model))
  expect_true(!is.null(res$history))
  expect_true(is.data.frame(res$history))
})
