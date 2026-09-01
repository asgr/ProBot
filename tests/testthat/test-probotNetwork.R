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

test_that(".probotRealNVPLayer forward and inverse are consistent", {
  layer <- ProBot:::.probotRealNVPLayer(4, 2, hidden_dim = 8, device = "cpu")
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

test_that("Neural Spline Flow forward and inverse are consistent", {
  set.seed(123)
  mdl <- probotMakeFlow(4, 2, n_layers = 3, hidden_dim = 16, n_bins = 8,
                        style = "nsf", device = "cpu")()
  theta <- torch_randn(3, 4)
  x <- torch_randn(3, 2)

  fwd <- mdl$forward(theta, x)
  rec <- mdl$inverse(fwd$z, x)

  expect_true(inherits(mdl, "probotFlowNSF"))
  expect_equal(fwd$z$shape, c(3L, 4L))
  expect_equal(fwd$log_det_jac$shape, c(3L, 1L))
  expect_equal(as.numeric(rec), as.numeric(theta), tolerance = 1e-4)
})

test_that("Neural Spline Flow has identity tails", {
  mdl <- probotMakeFlow(2, 1, n_layers = 1, hidden_dim = 8, style = "nsf",
                        device = "cpu")()
  theta <- torch_tensor(matrix(c(10, -10), nrow = 1))
  x <- torch_zeros(1, 1)
  out <- mdl$forward(theta, x)

  expect_equal(as.numeric(out$z), as.numeric(c(-10, 10)), tolerance = 1e-6)
  expect_equal(as.numeric(out$log_det_jac), 0, tolerance = 1e-6)
})

# ---- probotNetworkSuggest tests ----

test_that("probotNetworkSuggest returns list for Point", {
  s <- probotNetworkSuggest(5, 2, type = "Point", verbose = FALSE)
  expect_type(s, "list")
  expect_named(s, c("input_dim", "output_dim", "hidden_dims", "dropout", "n_params"))
  expect_equal(s$input_dim, 5L)
  expect_equal(s$output_dim, 2L)
  expect_length(s$hidden_dims, 3L)
})

test_that("probotNetworkSuggest returns list for MDN", {
  s <- probotNetworkSuggest(10, 3, type = "MDN", verbose = FALSE)
  expect_type(s, "list")
  expect_named(s, c("input_dim", "output_dim", "mdn_components", "hidden_dims", "dropout", "n_params"))
  expect_gte(s$mdn_components, 3L)
  expect_lte(s$mdn_components, 20L)
})

test_that("probotNetworkSuggest returns list for Flow", {
  s <- probotNetworkSuggest(6, 4, type = "Flow", verbose = FALSE)
  expect_type(s, "list")
  expect_named(s, c("dim_x", "dim_theta", "n_layers", "hidden_dim", "style", "n_params"))
  expect_equal(s$dim_x, 6L)
  expect_equal(s$dim_theta, 4L)
  expect_equal(s$style, "realnvp")
})

test_that("probotNetworkSuggest Flow defaults to RealNVP and is style-aware", {
  s_default <- probotNetworkSuggest(6, 4, type = "Flow", verbose = FALSE)
  s_realnvp  <- probotNetworkSuggest(6, 4, type = "Flow", flow_style = "realnvp",
                                     verbose = FALSE)
  expect_equal(s_default, s_realnvp)

  s_maf <- probotNetworkSuggest(6, 4, type = "Flow", flow_style = "maf",
                                  verbose = FALSE)
  expect_equal(s_maf$style, "maf")
})

test_that("probotNetworkSuggest MAF uses fewer blocks but wider conditioner", {
  # MAF pays dim_theta sequential passes per block at sample time, so the
  # heuristic keeps blocks low and shifts capacity into conditioner width.
  for (d in c(2, 4, 8, 16)) {
    cp <- probotNetworkSuggest(9, d, type = "Flow", verbose = FALSE)
    ar <- probotNetworkSuggest(9, d, type = "Flow", flow_style = "maf",
                               verbose = FALSE)
    expect_lt(ar$n_layers, cp$n_layers)
    expect_gt(ar$hidden_dim, cp$hidden_dim)
  }
})

test_that("probotNetworkSuggest flow_style is validated even for non-Flow types", {
  expect_error(probotNetworkSuggest(5, 2, type = "MDN",
                                     flow_style = "normalising", verbose = FALSE))
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

test_that("probotNetworkSuggest Flow n_layers clamped (RealNVP [4,24])", {
  s_small <- probotNetworkSuggest(2, 1, type = "Flow", verbose = FALSE)
  expect_gte(s_small$n_layers, 4L)

  s_large <- probotNetworkSuggest(2, 200, type = "Flow", verbose = FALSE)
  expect_lte(s_large$n_layers, 24L)
})

test_that("probotNetworkSuggest Flow n_layers clamped (maf [3,10])", {
  s_small <- probotNetworkSuggest(2, 1, type = "Flow",
                                   flow_style = "maf", verbose = FALSE)
  expect_gte(s_small$n_layers, 3L)
  expect_lte(s_small$n_layers, 10L)

  s_large <- probotNetworkSuggest(2, 200, type = "Flow",
                                   flow_style = "maf", verbose = FALSE)
  expect_lte(s_large$n_layers, 10L)
})

test_that("probotNetworkSuggest MAF hidden_dim has a higher floor", {
  # Tiny dims: RealNVP can drop to 32, MAF floored at 64 for stable
  # prefix conditioning.
  cp <- probotNetworkSuggest(1, 1, type = "Flow", verbose = FALSE)
  ar <- probotNetworkSuggest(1, 1, type = "Flow", flow_style = "maf",
                             verbose = FALSE)
  expect_gte(ar$hidden_dim, 64L)
  expect_gte(cp$hidden_dim, 32L)
  expect_lte(ar$hidden_dim, 512L)
})

test_that("probotNetworkSuggest Flow suggestions respect n_train", {
  s_few <- probotNetworkSuggest(9, 4, n_train = 1000, type = "Flow",
                                 flow_style = "maf", verbose = FALSE)
  s_many <- probotNetworkSuggest(9, 4, n_train = 1e6, type = "Flow",
                                 flow_style = "maf", verbose = FALSE)
  expect_lte(s_few$n_layers, s_many$n_layers)
  expect_lte(s_few$hidden_dim, s_many$hidden_dim)
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
  expect_true(inherits(mdl, "probotFlowRealNVP"))
})

test_that("probotNetworkSuggest MAF output plugs into probotMakeFlow", {
  s <- probotNetworkSuggest(5, 2, type = "Flow", flow_style = "maf",
                            verbose = FALSE)
  mdl <- do.call(probotMakeFlow, c(s, list(device = "cpu")))()
  expect_true(inherits(mdl, "probotFlowMAF"))
  expect_equal(mdl$n_blocks, s$n_layers)
})

test_that("probotNetworkSuggest NSF output plugs into probotMakeFlow", {
  s <- probotNetworkSuggest(5, 2, type = "Flow", flow_style = "nsf",
                            verbose = FALSE)
  mdl <- do.call(probotMakeFlow, c(s, list(device = "cpu")))()
  expect_true(inherits(mdl, "probotFlowNSF"))
  expect_equal(s$n_params, .count_params(mdl))
})

test_that("probotNetworkSuggest errors on invalid type", {
  expect_error(probotNetworkSuggest(5, 2, type = "GAN", verbose = FALSE))
})

test_that("probotNetworkSuggest errors on non-positive dimensions", {
  expect_error(probotNetworkSuggest(0, 2, type = "Point", verbose = FALSE))
  expect_error(probotNetworkSuggest(5, 0, type = "Point", verbose = FALSE))
})

test_that("probotNetworkSuggest rejects non-integer numeric dimensions", {
  expect_error(probotNetworkSuggest(1.9, 2, type = "Point", verbose = FALSE))
  expect_error(probotNetworkSuggest(3, 2.5, type = "MDN", verbose = FALSE))
})

test_that("probotNetworkSuggest MDN components correct for odd output_dim", {
  # output_dim=3 => ceiling(3/2) = 2, but clamped to max(3, 2) = 3
  s <- probotNetworkSuggest(5, 3, type = "MDN", verbose = FALSE)
  expect_equal(s$mdn_components, 3L)
  # output_dim=5 => ceiling(5/2) = 3, so max(3,3) = 3
  s2 <- probotNetworkSuggest(5, 5, type = "MDN", verbose = FALSE)
  expect_equal(s2$mdn_components, 3L)
  # output_dim=10 => ceiling(10/2) = 5
  s3 <- probotNetworkSuggest(5, 10, type = "MDN", verbose = FALSE)
  expect_equal(s3$mdn_components, 5L)
})

# ---- n_params tests ----

.count_params <- function(mdl) {
  sum(vapply(mdl$parameters, function(p) prod(as.integer(p$shape)), numeric(1)))
}

test_that("probotNetworkSuggest n_params matches actual Point model", {
  s <- probotNetworkSuggest(5, 2, type = "Point", verbose = FALSE)
  mdl <- do.call(probotMakePoint, c(s, list(device = "cpu")))()
  expect_equal(s$n_params, .count_params(mdl))
})

test_that("probotNetworkSuggest n_params matches actual MDN model", {
  s <- probotNetworkSuggest(7, 3, type = "MDN", verbose = FALSE)
  mdl <- do.call(probotMakeMDN, c(s, list(device = "cpu")))()
  expect_equal(s$n_params, .count_params(mdl))
})

test_that("probotNetworkSuggest n_params matches actual RealNVP Flow model", {
  s <- probotNetworkSuggest(6, 4, type = "Flow", verbose = FALSE)
  mdl <- do.call(probotMakeFlow, c(s, list(device = "cpu")))()
  expect_equal(s$n_params, .count_params(mdl))
})

test_that("probotNetworkSuggest n_params matches actual MAF Flow model", {
  s <- probotNetworkSuggest(6, 4, type = "Flow", flow_style = "maf",
                            verbose = FALSE)
  mdl <- do.call(probotMakeFlow, c(s, list(device = "cpu")))()
  # masks are buffers, not parameters, so the dense-counting heuristic is exact
  expect_equal(s$n_params, .count_params(mdl))
})

test_that("probotNetworkSuggest prints n_params when verbose = TRUE", {
  expect_output(
    probotNetworkSuggest(5, 2, type = "Point", verbose = TRUE),
    "n_params"
  )
})

test_that("probotNetworkSuggest n_params prints with thousands separators", {
  # hidden_dims clamp to 1024 for large dims -> n_params should be large
  expect_output(
    probotNetworkSuggest(100, 100, type = "Point", verbose = TRUE),
    "n_params   : ~[0-9]"
  )
})
