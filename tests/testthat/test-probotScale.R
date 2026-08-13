library(testthat)
library(ProBot)

# --- Helper: create small test data ---
make_test_data <- function(n = 50, input_dim = 3, output_dim = 2, seed = 42) {
  set.seed(seed)
  input <- matrix(rnorm(n * input_dim), n, input_dim)
  output <- matrix(rnorm(n * output_dim), n, output_dim)
  list(input = input, output = output)
}

test_that("probotScaleForward standardizes to zero mean and unit sd", {
  dat <- make_test_data()
  scaled <- probotScaleForward(dat$input)

  # Should return a matrix of same dimensions
  expect_true(is.matrix(scaled))
  expect_equal(dim(scaled), dim(dat$input))

  # col_scale attribute should be set
  expect_true(!is.null(attr(scaled, "col_scale")))
  expect_equal(length(attr(scaled, "col_scale")$col_means), ncol(dat$input))
  expect_equal(length(attr(scaled, "col_scale")$col_sds), ncol(dat$input))
})

test_that("probotScaleForward computes means and sds when not provided", {
  dat <- make_test_data()
  scaled <- probotScaleForward(dat$input)

  expect_true(!is.null(attr(scaled, "col_scale")))
  expect_equal(
    attr(scaled, "col_scale")$col_means,
    colMeans(dat$input),
    tolerance = 1e-6
  )
})

test_that("probotScaleBackward inverts forward transform", {
  dat <- make_test_data()
  col_means <- colMeans(dat$output)
  col_sds <- apply(dat$output, 2, sd)

  scaled <- probotScaleForward(dat$output, col_means, col_sds)
  unscaled <- probotScaleBackward(scaled, col_means, col_sds)

  expect_equal(as.matrix(unscaled), as.matrix(dat$output), tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("probotScaleBackward reads from attribute", {
  dat <- make_test_data()
  scaled <- probotScaleForward(dat$output)
  unscaled <- probotScaleBackward(scaled)

  expect_equal(as.matrix(unscaled), as.matrix(dat$output), tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("probotScaleBackward errors when no args and no attribute", {
  dat <- make_test_data()
  expect_error(probotScaleBackward(dat$output))
})

test_that("probotScaleBackward handles scalar col_means and col_sds", {
  dat <- make_test_data()
  nc <- ncol(dat$output)
  scaled <- probotScaleForward(dat$output, rep(0, nc), rep(1, nc))
  unscaled <- probotScaleBackward(scaled, rep(0, nc), rep(1, nc))

  expect_equal(as.matrix(unscaled), as.matrix(dat$output), tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("round-trip forward/backward is identity", {
  dat <- make_test_data()
  col_means <- colMeans(dat$input)
  col_sds <- apply(dat$input, 2, sd)

  rt <- probotScaleBackward(probotScaleForward(dat$input, col_means, col_sds), col_means, col_sds)
  expect_equal(as.matrix(rt), as.matrix(dat$input), tolerance = 1e-10, ignore_attr = TRUE)
})
