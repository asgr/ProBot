library(testthat)
library(ProBot)
library(torch)

test_that("probotMakeMDN returns an nn_module", {
  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(16, 16), device = "cpu")()
  expect_true(inherits(mdl, "nn_module"))
})

test_that("probotMakeMDN output dimension is correct", {
  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(16, 16), device = "cpu")()
  x <- torch_tensor(matrix(0, 1, 3))
  out <- mdl(x)

  # output_dim=2, components=3 => 3*(2*2+1) = 15
  expect_equal(out$shape, c(1L, 15L))
})

test_that("probotMakeMDN handles single hidden layer", {
  mdl <- probotMakeMDN(4, 1, 2, hidden_dims = 8, device = "cpu")()
  x <- torch_tensor(matrix(0, 2, 4))
  out <- mdl(x)
  # output_dim=1, components=2 => 2*(2*1+1) = 6
  expect_equal(out$shape, c(2L, 6L))
})

test_that("probotMakeMDN with dropout creates valid module", {
  mdl <- probotMakeMDN(3, 2, 2, hidden_dims = c(16), dropout = 0.1, device = "cpu")()
  expect_equal(mdl$dropout_rate, 0.1)
})

test_that("probotMakePoint returns an nn_module", {
  mdl <- probotMakePoint(3, 2, hidden_dims = c(16, 16), device = "cpu")()
  expect_true(inherits(mdl, "nn_module"))
})

test_that("probotMakePoint output dimension is correct", {
  mdl <- probotMakePoint(3, 2, hidden_dims = c(16, 16), device = "cpu")()
  x <- torch_tensor(matrix(0, 1, 3))
  out <- mdl(x)
  expect_equal(out$shape, c(1L, 2L))
})

test_that("probotMakeFlow returns an nn_module", {
  mdl <- probotMakeFlow(4, 2, n_layers = 2, hidden_dim = 8, device = "cpu")()
  expect_true(inherits(mdl, "nn_module"))
})

test_that("probotMakeFlow forward returns z and log_det_jac", {
  mdl <- probotMakeFlow(4, 2, n_layers = 2, hidden_dim = 8, device = "cpu")()
  theta <- torch_randn(4, 4)
  x <- torch_randn(4, 2)

  out <- mdl$forward(theta, x)
  expect_true("z" %in% names(out))
  expect_true("log_det_jac" %in% names(out))
  expect_equal(out$z$shape, c(4L, 4L))
})

test_that("probotMakeFlow inverse returns theta of correct shape", {
  mdl <- probotMakeFlow(4, 2, n_layers = 2, hidden_dim = 8, device = "cpu")()
  z <- torch_randn(4, 4)
  x <- torch_randn(4, 2)

  theta <- mdl$inverse(z, x)
  expect_equal(theta$shape, c(4L, 4L))
})

test_that("probotCouplingLayer forward and inverse are consistent", {
  layer <- probotCouplingLayer(4, 2, hidden_dim = 8, device = "cpu")()
  theta <- torch_randn(2, 4)
  x <- torch_randn(2, 2)

  fwd <- layer$forward(theta, x)
  rec <- layer$inverse(fwd$z, x)

  expect_equal(
    as.numeric(rec),
    as.numeric(theta),
    tolerance = 1e-5
  )
})

test_that("probotMakeFlow forward then inverse recovers input", {
  set.seed(123)
  mdl <- probotMakeFlow(6, 3, n_layers = 3, hidden_dim = 16, device = "cpu")()
  theta <- torch_randn(2, 6)
  x <- torch_randn(2, 3)

  fwd <- mdl$forward(theta, x)
  rec <- mdl$inverse(fwd$z, x)

  expect_equal(
    as.numeric(rec),
    as.numeric(theta),
    tolerance = 1e-4
  )
})

# ---- probotNetworkSuggest tests ----

test_that("probotNetworkSuggest returns list for Point", {
  s <- probotNetworkSuggest(5, 2, type = "Point", verbose = FALSE)
  expect_type(s, "list")
  expect_named(s, c("input_dim", "output_dim", "hidden_dims", "dropout"))
  expect_equal(s$input_dim, 5L)
  expect_equal(s$output_dim, 2L)
  expect_length(s$hidden_dims, 3L)
})

test_that("probotNetworkSuggest returns list for MDN", {
  s <- probotNetworkSuggest(10, 3, type = "MDN", verbose = FALSE)
  expect_type(s, "list")
  expect_named(s, c("input_dim", "output_dim", "mdn_components", "hidden_dims", "dropout"))
  expect_gte(s$mdn_components, 3L)
  expect_lte(s$mdn_components, 20L)
})

test_that("probotNetworkSuggest returns list for Flow", {
  s <- probotNetworkSuggest(6, 4, type = "Flow", verbose = FALSE)
  expect_type(s, "list")
  expect_named(s, c("dim_x", "dim_theta", "n_layers", "hidden_dim"))
  expect_equal(s$dim_x, 6L)
  expect_equal(s$dim_theta, 4L)
})

test_that("probotNetworkSuggest type matching is case-insensitive", {
  s1 <- probotNetworkSuggest(4, 2, type = "point", verbose = FALSE)
  s2 <- probotNetworkSuggest(4, 2, type = "POINT", verbose = FALSE)
  expect_equal(s1, s2)
})

test_that("probotNetworkSuggest hidden_dims hourglass shape for Point", {
  s <- probotNetworkSuggest(5, 2, type = "Point", verbose = FALSE)
  expect_equal(s$hidden_dims[1], s$hidden_dims[3])
  expect_equal(s$hidden_dims[2], 2L * s$hidden_dims[1])
})

test_that("probotNetworkSuggest Flow n_layers clamped in [4, 16]", {
  s_small <- probotNetworkSuggest(2, 1, type = "Flow", verbose = FALSE)
  expect_gte(s_small$n_layers, 4L)

  s_large <- probotNetworkSuggest(2, 20, type = "Flow", verbose = FALSE)
  expect_lte(s_large$n_layers, 16L)
})

test_that("probotNetworkSuggest MDN mdn_components min is 3", {
  s <- probotNetworkSuggest(4, 1, type = "MDN", verbose = FALSE)
  expect_gte(s$mdn_components, 3L)
})

test_that("probotNetworkSuggest dropout 0 for small input_dim", {
  s <- probotNetworkSuggest(3, 2, type = "Point", verbose = FALSE)
  expect_equal(s$dropout, 0)
})

test_that("probotNetworkSuggest dropout > 0 for large input_dim", {
  s <- probotNetworkSuggest(15, 2, type = "Point", verbose = FALSE)
  expect_gt(s$dropout, 0)
})

test_that("probotNetworkSuggest prints output when verbose = TRUE", {
  expect_output(
    probotNetworkSuggest(5, 2, type = "MDN", verbose = TRUE),
    "probotNetworkSuggest"
  )
})

test_that("probotNetworkSuggest hidden_dims clamped to [32, 1024]", {
  # Tiny dimensions: ref = 8 * (1+1) = 16, rounds to 32
  s <- probotNetworkSuggest(1, 1, type = "Point", verbose = FALSE)
  expect_gte(s$hidden_dims[1], 32L)

  # Large dimensions: ref = 8 * (100+100) = 1600, clamps to 1024
  s2 <- probotNetworkSuggest(100, 100, type = "Point", verbose = FALSE)
  expect_lte(s2$hidden_dims[1], 1024L)
})

test_that("probotNetworkSuggest Point output plugs into probotMakePoint", {
  s <- probotNetworkSuggest(5, 2, type = "Point", verbose = FALSE)
  mdl <- do.call(probotMakePoint, c(s, list(device = "cpu")))()
  expect_true(inherits(mdl, "nn_module"))
})

test_that("probotNetworkSuggest MDN output plugs into probotMakeMDN", {
  s <- probotNetworkSuggest(5, 2, type = "MDN", verbose = FALSE)
  mdl <- do.call(probotMakeMDN, c(s, list(device = "cpu")))()
  expect_true(inherits(mdl, "nn_module"))
})

test_that("probotNetworkSuggest Flow output plugs into probotMakeFlow", {
  s <- probotNetworkSuggest(5, 2, type = "Flow", verbose = FALSE)
  mdl <- do.call(probotMakeFlow, c(s, list(device = "cpu")))()
  expect_true(inherits(mdl, "nn_module"))
})

test_that("probotNetworkSuggest errors on invalid type", {
  expect_error(probotNetworkSuggest(5, 2, type = "GAN", verbose = FALSE))
})

test_that("probotNetworkSuggest errors on non-positive dimensions", {
  expect_error(probotNetworkSuggest(0, 2, type = "Point", verbose = FALSE))
  expect_error(probotNetworkSuggest(5, 0, type = "Point", verbose = FALSE))
})
