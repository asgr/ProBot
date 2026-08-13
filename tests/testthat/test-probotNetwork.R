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
