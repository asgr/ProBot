library(testthat)
library(ProBot)
library(torch)

# --- Build a tiny trained-ish MDN for prediction tests ---
setup_mdn <- function() {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  hidden <- c(16, 16)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = hidden, device = "cpu")()
  mdl$eval()

  inp <- matrix(rnorm(10 * input_dim), 10, input_dim)
  list(mdl = mdl, K = K, inp = inp, output_dim = output_dim)
}

test_that("probotPredictMDN returns a list with mu, log10_sigma, logits", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)

  expect_true("mu" %in% names(out))
  expect_true("log10_sigma" %in% names(out))
  expect_true("logits" %in% names(out))
})

test_that("probotPredictMDN output shapes are correct", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)

  n <- nrow(s$inp)
  expect_equal(dim(out$mu), c(n, s$K, s$output_dim))
  expect_equal(dim(out$log10_sigma), c(n, s$K, s$output_dim))
  expect_equal(dim(out$logits), c(n, s$K))
})

test_that("probotPredictMDN auto-detects device from model", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K, device = NULL)
  expect_true("mu" %in% names(out))
})

test_that("probotSamplePostMDN returns correct dimensions", {
  s <- setup_mdn()
  col_means <- colMeans(matrix(rnorm(100), 100, s$output_dim))
  col_sds <- apply(matrix(rnorm(100, 1, 0.5), 100, s$output_dim), 2, sd)

  n_samples <- 200
  samples <- probotSamplePostMDN(s$inp[1, ], s$mdl, s$K, n_samples = n_samples,
                                  col_means = col_means, col_sds = col_sds)

  expect_equal(dim(samples), c(n_samples, s$output_dim))
})

test_that("probotSamplePostMDN sets column names", {
  s <- setup_mdn()
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)
  names <- c("a", "b")

  samples <- probotSamplePostMDN(s$inp[1, ], s$mdl, s$K, n_samples = 100,
                                  col_means = col_means, col_sds = col_sds,
                                  col_names = names)
  expect_equal(colnames(samples), names)
})

test_that("probotSamplePostMDN accepts a vector or a 1-row matrix identically", {
  s <- setup_mdn()
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)

  set.seed(11)
  v <- probotSamplePostMDN(s$inp[2, ], s$mdl, s$K, n_samples = 300,
                           col_means = col_means, col_sds = col_sds)
  set.seed(11)
  m <- probotSamplePostMDN(matrix(s$inp[2, ], nrow = 1), s$mdl, s$K,
                           n_samples = 300,
                           col_means = col_means, col_sds = col_sds)

  expect_true(is.matrix(v))
  expect_equal(dim(v), dim(m))
  expect_equal(v, m)
})

test_that("probotSamplePostMDN keeps (n_samples x D) shape when D == 1", {
  set.seed(5)
  mdl <- probotMakeMDN(3, 1, 2, hidden_dims = c(8, 8), device = "cpu")()
  x <- matrix(rnorm(15), 5, 3)

  s <- probotSamplePostMDN(x[1, ], mdl, 2, n_samples = 100, col_names = "theta")
  expect_true(is.matrix(s))
  expect_equal(dim(s), c(100L, 1L))
  expect_equal(colnames(s), "theta")
})

test_that("probotSamplePostMDN returns an (n_samples x D x N) array for many rows", {
  s <- setup_mdn()
  n_samples <- 100
  arr <- probotSamplePostMDN(s$inp, s$mdl, s$K, n_samples = n_samples,
                             col_names = c("a", "b"))

  expect_equal(dim(arr), c(n_samples, s$output_dim, nrow(s$inp)))
  expect_equal(dimnames(arr)[[2]], c("a", "b"))
  expect_named(dimnames(arr), c("Sample", "Parameter", "Observation"))
})

test_that("probotSamplePostMDN chunking does not alter draws", {
  # With batch_size = 1 each chunk holds exactly one observation, so the
  # first observation's draws must be bit-identical to a single-observation
  # call made under the same seed. Larger chunks re-order the RNG stream (the
  # mixture draws are consumed jointly), so only moments are comparable.
  s <- setup_mdn()
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)

  set.seed(23)
  one <- probotSamplePostMDN(s$inp[1, ], s$mdl, s$K, n_samples = 100,
                             col_means = col_means, col_sds = col_sds)
  set.seed(23)
  bulk <- probotSamplePostMDN(s$inp, s$mdl, s$K, n_samples = 100,
                              col_means = col_means, col_sds = col_sds,
                              batch_size = 1)

  expect_equal(dim(bulk), c(100, s$output_dim, nrow(s$inp)))
  expect_equal(one, bulk[, , 1])
})

test_that("probotSamplePostMDN is insensitive to chunk size in distribution", {
  # Chunking must not bias the posterior: both chunkings are compared against
  # the analytic marginal moments rather than against each other, since the
  # expected values here are near zero and a relative tolerance would be
  # dominated by Monte Carlo noise.
  s <- setup_mdn()
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)
  n_samples <- 4000

  pred <- probotPredictMDN(s$inp, s$mdl, s$K, device = "cpu")
  marg <- probotMarginalPostMDN(pred, col_means = col_means, col_sds = col_sds)

  # Analytic bounds on Monte Carlo error, with headroom for the max over
  # 10 obs x 2 params cells: SE(mean) = sd/sqrt(S), SE(sd) ~ sd/sqrt(2S).
  tol_m <- 6 * marg$post_sd / sqrt(n_samples)
  tol_s <- 6 * marg$post_sd / sqrt(2 * n_samples)

  for (bs in c(1L, 10L)) {
    arr <- probotSamplePostMDN(s$inp, s$mdl, s$K, n_samples = n_samples,
                               col_means = col_means, col_sds = col_sds,
                               batch_size = bs)
    expect_equal(dim(arr), c(n_samples, s$output_dim, nrow(s$inp)))

    got_m <- t(apply(arr, c(2, 3), mean))
    got_s <- t(apply(arr, c(2, 3), sd))

    expect_true(all(abs(got_m - marg$post_mean) < tol_m))
    expect_true(all(abs(got_s - marg$post_sd) < tol_s))
  }
})

test_that("probotSamplePostMDN matches the analytic marginal moments", {
  s <- setup_mdn()
  col_means <- c(5, -2)
  col_sds <- c(2, 0.5)

  pred <- probotPredictMDN(s$inp[1, , drop = FALSE], s$mdl, s$K, device = "cpu")
  marg <- probotMarginalPostMDN(pred, col_means = col_means, col_sds = col_sds)
  draws <- probotSamplePostMDN(s$inp[1, ], s$mdl, s$K, n_samples = 1e5,
                               col_means = col_means, col_sds = col_sds)

  # Monte Carlo SE on the mean is ~ sd/sqrt(n) <= 0.011 here.
  expect_equal(colMeans(draws), as.vector(marg$post_mean), tolerance = 0.02)
  expect_equal(apply(draws, 2, sd), as.vector(marg$post_sd), tolerance = 0.02)
})

test_that("probotSamplePostMDN soft-clamps log10_sigma to [-5, 5]", {
  # A single-component MDN with log10_sigma = 8 would sample from N(0, 1e8)
  # without clamping. Training/loss clamp log10_sigma to [-5, 5], so the
  # sampler must cap sigma at 1e5 and the sample SD should land near 1e5
  # (a hundred-fold separation from the unclamped 1e8).
  # Stub module: emits fixed (mu, log10_sigma, logit) regardless of input.
  stub <- nn_module(
    initialize = function() {
      self$out <- torch_tensor(matrix(c(0, 8, 0), 1, 3))
    },
    forward = function(x) self$out$expand(c(x$size(1), -1))
  )

  set.seed(7)
  mdl <- stub()
  x <- matrix(0, 1, 1)

  n_samples <- 1e5
  samples <- probotSamplePostMDN(x, mdl, mdn_components = 1,
                                 n_samples = n_samples,
                                 col_means = 0, col_sds = 1)
  # The clamp is what this test is for: capped sigma = 1e5, not 1e8.
  expect_true(sd(samples) > 0.8e5 & sd(samples) < 1.2e5)
  # Sanity: sampling around 0, within a few Monte Carlo standard errors
  # (SE of the mean = sigma / sqrt(n) ~ 316 here).
  se <- 1e5 / sqrt(n_samples)
  expect_lt(abs(mean(samples)), 6 * se)
})

test_that("probotMarginalPostMDN returns list with post_mean and post_sd", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)

  res <- probotMarginalPostMDN(out, col_means, col_sds)
  expect_true("post_mean" %in% names(res))
  expect_true("post_sd" %in% names(res))
})

test_that("probotMarginalPostMDN output dimensions are correct", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)

  res <- probotMarginalPostMDN(out, col_means, col_sds)
  n <- nrow(s$inp)
  expect_equal(dim(res$post_mean), c(n, s$output_dim))
  expect_equal(dim(res$post_sd), c(n, s$output_dim))
})

test_that("probotMarginalPostMDN sets column names", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)
  names <- c("x1", "x2")

  res <- probotMarginalPostMDN(out, col_means, col_sds, col_names = names)
  expect_equal(colnames(res$post_mean), names)
  expect_equal(colnames(res$post_sd), names)
})

test_that("probotMarginalPostMDN soft-clamps log10_sigma to [-5, 5]", {
  # log10_sigma = 8 (sigma = 1e8) must be clamped to sigma = 1e5; the
  # analytic marginal SD is exactly that (weights = 1, mu = 0). The lower
  # bound (-5) is checked the same way.
  o_hi <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(8, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  res <- probotMarginalPostMDN(o_hi, col_means = 0, col_sds = 1)
  expect_equal(res$post_sd[1, 1], 1e5)

  o_lo <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(-8, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  res_lo <- probotMarginalPostMDN(o_lo, col_means = 0, col_sds = 1)
  expect_equal(res_lo$post_sd[1, 1], 1e-5)

  # An in-range value must pass through unchanged (sigma = 10^0.5).
  o_mid <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(0.5, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  res_mid <- probotMarginalPostMDN(o_mid, col_means = 0, col_sds = 1)
  expect_equal(res_mid$post_sd[1, 1], 10^0.5, tolerance = 1e-6)
})

# --- Point prediction tests ---
setup_point <- function() {
  set.seed(42)
  input_dim <- 3; output_dim <- 2
  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(16, 16), device = "cpu")()
  mdl$eval()
  inp <- matrix(rnorm(10 * input_dim), 10, input_dim)
  list(mdl = mdl, inp = inp, output_dim = output_dim)
}

test_that("probotPredictPoint returns a numeric matrix", {
  s <- setup_point()
  pred <- probotPredictPoint(s$inp, s$mdl)
  expect_true(is.matrix(pred))
  expect_true(is.numeric(pred))
  expect_equal(dim(pred), c(nrow(s$inp), s$output_dim))
})

test_that("probotPredictPointScaled returns un-scaled predictions", {
  s <- setup_point()
  col_means <- c(10, 20)
  col_sds <- c(2, 3)
  pred <- probotPredictPointScaled(s$inp, s$mdl, col_means, col_sds)
  expect_true(is.matrix(pred))
  expect_equal(dim(pred), c(nrow(s$inp), s$output_dim))
})
