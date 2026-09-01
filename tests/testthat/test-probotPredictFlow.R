library(testthat)
library(ProBot)
library(torch)

test_that("probotSamplePostNF returns correct dimensions", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(input_dim)
  n_samples <- 200

  samples <- probotSamplePostNF(input, mdl, n_samples = n_samples,
                                  output_dim = output_dim)
  expect_equal(dim(samples), c(n_samples, output_dim))
})

test_that("probotSamplePostNF infers dim from col_means", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(input_dim)
  col_means <- rep(0, output_dim)
  col_sds <- rep(1, output_dim)
  n_samples <- 200

  samples <- probotSamplePostNF(input, mdl, n_samples = n_samples,
                                  col_means = col_means, col_sds = col_sds)
  expect_equal(dim(samples), c(n_samples, output_dim))
})

test_that("probotSamplePostNF errors without dim info", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(input_dim)

  expect_error(
    probotSamplePostNF(input, mdl, n_samples = 100),
    "Either 'output_dim' or 'col_means'"
  )
})

test_that("probotSamplePostNF sets column names", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(input_dim)
  names <- paste0("theta_", 1:output_dim)

  samples <- probotSamplePostNF(input, mdl, n_samples = 100,
                                  output_dim = output_dim, col_names = names)
  expect_equal(colnames(samples), names)
})

test_that("probotSamplePostNF accepts torch tensor input", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input_torch <- torch_randn(input_dim)  # 1D tensor
  samples <- probotSamplePostNF(input_torch, mdl, n_samples = 100,
                                  output_dim = output_dim)
  expect_equal(dim(samples), c(100, output_dim))
})

test_that("probotSamplePostNF with unscaling works", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- runif(input_dim)
  col_means <- c(10, 20, 30, 40)
  col_sds <- c(2, 3, 4, 5)

  samples <- probotSamplePostNF(input, mdl, n_samples = 100,
                                  col_means = col_means, col_sds = col_sds)
  expect_equal(dim(samples), c(100, output_dim))
})

test_that("probotSamplePostNF with matrix input works", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- matrix(runif(input_dim), nrow = 1, ncol = input_dim)

  samples <- probotSamplePostNF(input, mdl, n_samples = 100,
                                  output_dim = output_dim)
  expect_equal(dim(samples), c(100, output_dim))
})

test_that("probotSamplePostNF multi-obs handles partial final chunk", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()

  input <- matrix(runif(7 * input_dim), nrow = 7)

  samples <- probotSamplePostNF(input, mdl, n_samples = 6,
                                  output_dim = output_dim,
                                  col_names = paste0("p", 1:output_dim),
                                  device = "cpu",
                                  batch_size = 3)
  expect_equal(dim(samples), c(6, output_dim, 7))
  expect_equal(dimnames(samples)[[2]], paste0("p", 1:output_dim))
  expect_true(all(is.finite(samples)))
})

test_that("probotSamplePostNF multi-obs unscaling is exact for an identity flow", {
  output_dim <- 2; input_dim <- 2
  n_samples <- 5; n_obs <- 3

  # Mock flow: inverse() is the identity map and records its z input.
  mock <- list(parameters = NULL)
  mock$eval <- function() NULL
  mock$inverse <- function(z, x) {
    mock$z_seen <<- z
    z
  }

  col_means <- c(10, 20)
  col_sds <- c(2, 3)
  input <- matrix(runif(n_obs * input_dim), nrow = n_obs)

  samples <- probotSamplePostNF(input, mock, n_samples = n_samples,
                                  col_means = col_means, col_sds = col_sds,
                                  output_dim = output_dim, device = "cpu",
                                  batch_size = n_obs)

  z_ref <- as.matrix(mock$z_seen$cpu()) # (n_obs*n_samples, output_dim)
  expected <- array(NA_real_, c(n_samples, output_dim, n_obs))
  for (o in seq_len(n_obs)) {
    for (s in seq_len(n_samples)) {
      expected[s, , o] <- z_ref[(o - 1) * n_samples + s, ] * col_sds + col_means
    }
  }

  expect_equal(as.vector(samples), as.vector(expected), tolerance = 1e-5)
})

test_that("probotSamplePostNF errors when col_sds missing with col_means", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()

  input <- runif(input_dim)

  expect_error(
    probotSamplePostNF(input, mdl, n_samples = 100, col_means = rep(0, output_dim)),
    "col_sds must be provided"
  )
})
