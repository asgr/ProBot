library(testthat)
library(ProBot)
library(torch)

test_that("probotSamplePostNF returns correct dimensions", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(dim_x)
  n_samples <- 200

  samples <- probotSamplePostNF(input, mdl, n_samples = n_samples,
                                  dim_theta = dim_theta)
  expect_equal(dim(samples), c(n_samples, dim_theta))
})

test_that("probotSamplePostNF infers dim from col_means", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(dim_x)
  col_means <- rep(0, dim_theta)
  col_sds <- rep(1, dim_theta)
  n_samples <- 200

  samples <- probotSamplePostNF(input, mdl, n_samples = n_samples,
                                  col_means = col_means, col_sds = col_sds)
  expect_equal(dim(samples), c(n_samples, dim_theta))
})

test_that("probotSamplePostNF errors without dim info", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(dim_x)

  expect_error(
    probotSamplePostNF(input, mdl, n_samples = 100),
    "Either 'dim_theta' or 'col_means'"
  )
})

test_that("probotSamplePostNF sets column names", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(dim_x)
  names <- paste0("theta_", 1:dim_theta)

  samples <- probotSamplePostNF(input, mdl, n_samples = 100,
                                  dim_theta = dim_theta, col_names = names)
  expect_equal(colnames(samples), names)
})

test_that("probotSamplePostNF accepts torch tensor input", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input_torch <- torch_randn(dim_x)  # 1D tensor
  samples <- probotSamplePostNF(input_torch, mdl, n_samples = 100,
                                  dim_theta = dim_theta)
  expect_equal(dim(samples), c(100, dim_theta))
})

test_that("probotSamplePostNF with unscaling works", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(dim_x)
  col_means <- c(10, 20, 30, 40)
  col_sds <- c(2, 3, 4, 5)

  samples <- probotSamplePostNF(input, mdl, n_samples = 100,
                                  col_means = col_means, col_sds = col_sds)
  expect_equal(dim(samples), c(100, dim_theta))
})

test_that("probotSamplePostNF with matrix input works", {
  set.seed(42)
  dim_theta <- 4; dim_x <- 2
  mdl <- probotMakeFlow(dim_theta, dim_x, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- matrix(runif(dim_x), nrow = 1, ncol = dim_x)

  samples <- probotSamplePostNF(input, mdl, n_samples = 100,
                                  dim_theta = dim_theta)
  expect_equal(dim(samples), c(100, dim_theta))
})
