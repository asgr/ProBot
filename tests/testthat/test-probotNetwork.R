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
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  theta <- torch_randn(4, 4)
  x <- torch_randn(4, 2)

  out <- mdl$forward(theta, x)
  expect_true("z" %in% names(out))
  expect_true("log_det_jac" %in% names(out))
  expect_equal(out$z$shape, c(4L, 4L))
})

test_that("probotMakeFlow inverse returns theta of correct shape", {
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  z <- torch_randn(4, 4)
  x <- torch_randn(4, 2)

  theta <- mdl$inverse(z, x)
  expect_equal(theta$shape, c(4L, 4L))
})

test_that(".probotRealNVPLayer forward and inverse are consistent", {
  layer <- ProBot:::.probotRealNVPLayer(2, 4, hidden_dim = 8, device = "cpu")
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
  mdl <- probotMakeFlow(3, 6, n_layers = 3, hidden_dim = 16, device = "cpu")()
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
  mdl <- probotMakeFlow(2, 4, n_layers = 3, hidden_dim = 16, n_bins = 8,
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

test_that("NSF inverse picks the in-bin root when the quadratic coefficient is negative", {
  # Regression: the spline inverse solves a quadratic and takes the root
  # 2c / -(b + sqrt(disc)). An earlier version used -(abs(b) + sqrt(disc)),
  # which silently switches to the *other* root whenever b < 0 -- an order-1
  # error. b >= 0 for a freshly initialised conditioner, so a round-trip on a
  # default model passes either way and cannot catch it; scaling the conditioner
  # weights reaches the trained regime where some b go negative. Run in float64
  # so the assertion measures root selection rather than float32 conditioning.
  for (s in c(3, 77)) {
    torch_manual_seed(s)
    mdl <- probotFlowNSF(input_dim = 3, output_dim = 6, n_layers = 1,
                         hidden_dim = 32, n_bins = 8, tail_bound = 3,
                         device = "cpu")
    with_no_grad({ for (p in mdl$parameters) p$mul_(2) })
    mdl$to(dtype = torch_float64())
    mdl$eval()

    theta <- torch_tensor(matrix(rnorm(200 * 6) * 1.2, ncol = 6),
                          dtype = torch_float64())
    x <- torch_tensor(matrix(rnorm(200 * 3), ncol = 3), dtype = torch_float64())
    err <- as.numeric((mdl$inverse(mdl$forward(theta, x)$z, x) - theta)$abs())
    expect_lt(max(err), 1e-6)
  }
})

test_that("NSF inverse round-trips a realistic trained-conditioner model", {
  # Wider net, several layers, still float64: catches a regression that only
  # shows up once the conditioning network is expressive enough to produce
  # negative b across many bins.
  torch_manual_seed(2026)
  mdl <- probotFlowNSF(input_dim = 4, output_dim = 6, n_layers = 3,
                       hidden_dim = 64, n_bins = 8, tail_bound = 3,
                       device = "cpu")
  with_no_grad({ for (p in mdl$parameters) p$mul_(1.5) })
  mdl$to(dtype = torch_float64())
  mdl$eval()
  theta <- torch_tensor(matrix(rnorm(300 * 6) * 1.5, ncol = 6), dtype = torch_float64())
  x <- torch_tensor(matrix(rnorm(300 * 4), ncol = 4), dtype = torch_float64())
  err <- as.numeric((mdl$inverse(mdl$forward(theta, x)$z, x) - theta)$abs())
  expect_lt(max(err), 1e-6)
})

test_that("Neural Spline Flow has identity tails", {
  # input_dim = 1 (context), output_dim = 2 (parameter space)
  mdl <- probotMakeFlow(1, 2, n_layers = 1, hidden_dim = 8, style = "nsf",
                        device = "cpu")()
  theta <- torch_tensor(matrix(c(10, -10), nrow = 1))
  x <- torch_zeros(1, 1)
  out <- mdl$forward(theta, x)

  expect_equal(as.numeric(out$z), as.numeric(c(-10, 10)), tolerance = 1e-6)
  expect_equal(as.numeric(out$log_det_jac), 0, tolerance = 1e-6)
})

test_that("Neural Spline Flow backward is finite through identity tails", {
  # Regression: the spline is the identity outside [-tail_bound, tail_bound],
  # passed through torch_where(). torch_where still routes grad*0 into the
  # unselected (tail) branch, and that branch used to compute a 0/0 -> NaN, so
  # any point beyond the tail silently poisoned every downstream gradient.
  # Force tail points (|theta| > 3) and check the loss backward is NaN-free.
  set.seed(7)
  mdl <- probotMakeFlow(4, 3, n_layers = 2, hidden_dim = 16, n_bins = 8,
                        style = "nsf", device = "cpu")()
  mdl$train()

  # theta deliberately spans beyond the tail_bound = 3 in both directions
  theta <- torch_tensor(matrix(
    c(1.0, -1.0, 0.0,  5.0, -5.0, 0.0,  0.5, -3.5, 1.5), nrow = 3, byrow = TRUE
  ))
  x <- torch_randn(3, 4)

  loss <- probotLossNF(theta, x, mdl)
  expect_false(is.na(loss$item()))
  loss$backward()

  for (p in mdl$parameters) {
    if (!is.null(p$grad)) {
      expect_equal(torch_isnan(p$grad)$sum()$item(), 0)
      expect_false(is.na(p$grad$abs()$max()$item()))
    }
  }

  # Inverse (sampling) path: gradient through out-of-domain base draws must be
  # finite too. Base draws scaled up so some values land beyond tail_bound = 3.
  mdl$eval()
  z <- (torch_randn(3, 3) * 1.5)$requires_grad_(TRUE)
  theta_hat <- mdl$inverse(z, x)
  expect_equal(torch_isnan(theta_hat)$sum()$item(), 0)
  (theta_hat^2)$sum()$backward()
  expect_equal(torch_isnan(z$grad)$sum()$item(), 0)
})

# ---- rational quadratic spline: knot-level bin selection ----

# Mirrors .probotSplineCouplingLayer$spline_params(): widths and heights from a
# softmax scaled to fill [-tail_bound, tail_bound], interior knot derivatives
# from a softplus, boundary derivatives fixed at 1. A realistic parameterisation
# matters here because the bug is a float32 rounding artifact of exactly this
# softmax-then-cumsum construction.
spline_pars_realistic <- function(batch, dim, n_bins, tail_bound, seed) {
  torch_manual_seed(seed)
  raw <- torch_randn(c(batch, dim, 3 * n_bins - 1))
  widths <- 1e-3 + (2 * tail_bound - 1e-3 * n_bins) *
    nnf_softmax(raw$narrow(3, 1, n_bins), dim = 3)
  heights <- 1e-3 + (2 * tail_bound - 1e-3 * n_bins) *
    nnf_softmax(raw$narrow(3, n_bins + 1, n_bins), dim = 3)
  derivs <- 1e-3 + nnf_softplus(raw$narrow(3, 2 * n_bins + 1, n_bins - 1))
  ones <- torch_ones(c(batch, dim, 1))
  list(widths = widths, heights = heights,
       derivatives = torch_cat(list(ones, derivs, ones), dim = 3))
}

test_that("rational quadratic spline selects exactly one bin at every knot", {
  # Regression for the transient NaN loss epochs seen training NSF models. Bin
  # selection used to be the interval test (ic >= coord) & (ic < coord + span),
  # but coord[[k]] + span[[k]] and coord[[k + 1]] are reached by different
  # routes -- one addition versus one cumsum -- and disagree by up to an ulp in
  # float32. Where they overlap TWO bins match, where they gape ZERO match and
  # the old ones_like() fallback then matched all of them; either way the masked
  # SUM in select_bin() returns the sum of several bins, so theta escapes
  # [0, 1], the polynomial denominator can go negative, and log() of it is NaN.
  # Over 2.1M internal knots built the way the coupling layer builds them, 23.6%
  # have an overlapping boundary and 23.6% a gapped one, so probing every knot
  # fires reliably. Note this is float32-only: do not "tidy" the test to float64,
  # where both the old and new code pass.
  K <- 8; tb <- 3
  spline <- ProBot:::.probotRationalQuadraticSpline

  for (seed in c(11, 12, 13)) {
    p <- spline_pars_realistic(500, 4, K, tb, seed)
    # A spline knot's x position comes from the widths' cumsum and its y
    # position from the heights'. Internal knot k is the shared edge of bins k
    # and k + 1, so it carries the derivative at index k + 1.
    knot_x <- -tb + torch_cumsum(p$widths, dim = 3)
    knot_y <- -tb + torch_cumsum(p$heights, dim = 3)
    lad_expected <- log(as.array(p$derivatives)[, , 2:K])

    for (k in seq_len(K - 1)) {
      xk <- knot_x$narrow(3, k, 1)$squeeze(3)
      yk <- knot_y$narrow(3, k, 1)$squeeze(3)
      spl <- function(x, ...) spline(x, p$widths, p$heights, p$derivatives,
                                   tail_bound = tb, ...)
      tag <- sprintf("seed %d knot %d", seed, k)

      # Exactly on the knot. theta is then 0 in the bin to the knot's right, so
      # the spline must return that bin's bottom edge -- the knot's own image --
      # and its log|det| must be the knot's derivative. Both hold bit-for-bit
      # when the right single bin is selected, and break badly otherwise.
      fwd <- spl(xk)
      expect_true(all(is.finite(as.array(fwd$output))), info = tag)
      expect_true(all(is.finite(as.array(fwd$logabsdet))), info = tag)
      expect_lt(max(abs(as.array(fwd$output) - as.array(yk))), 1e-6)
      expect_lt(max(abs(as.array(fwd$logabsdet) - lad_expected[, , k])), 1e-5)

      # The inverse isolates its bins the same way, by counting bottom_edges.
      inv <- spl(yk, inverse = TRUE)
      expect_true(all(is.finite(as.array(inv))), info = tag)
      expect_lt(max(abs(as.array(inv) - as.array(xk))), 1e-6)

      # Just either side of the knot: finite, and continuous in the output.
      # log|det| is deliberately not bounded here -- it is continuous but can be
      # extremely steep near a knot when the bin's delta is large and its left
      # derivative small, so a step of 1e-4 legitimately moves it by nats.
      for (sgn in c(-1, 1)) {
        side <- spl(xk + sgn * 1e-4)
        expect_true(all(is.finite(as.array(side$output))), info = tag)
        expect_true(all(is.finite(as.array(side$logabsdet))), info = tag)
        expect_lt(max(abs(as.array(side$output) - as.array(yk))), 1e-2)
      }
    }
  }
})

test_that("rational quadratic spline is the identity at and beyond the domain edges", {
  # Guards the invariant that replaced the ones_like() fallback. Counting edges
  # can never select zero bins: coordinate[[1]] == -tail_bound <= ic because ic
  # is clamped, so the count is >= 1, and with n_bins edges it is <= n_bins. At
  # +tail_bound every edge is at or below the input, giving the count n_bins --
  # already the last valid bin, no clamp needed. Probes both edges, the points
  # just outside them, and far outside; if a future edit reintroduces a
  # reachable empty mask, the pass-through assertions below catch it.
  K <- 8; tb <- 3
  spline <- ProBot:::.probotRationalQuadraticSpline
  p <- spline_pars_realistic(200, 4, K, tb, 31)

  x <- torch_tensor(matrix(
    rep_len(c(tb, -tb, 20, -20, tb + 1e-6, -tb - 1e-6, 0, -0.5), 200 * 4),
    nrow = 200, ncol = 4))
  mx <- as.matrix(x)
  fwd <- spline(x, p$widths, p$heights, p$derivatives, tail_bound = tb)
  expect_true(all(is.finite(as.array(fwd$output))))
  expect_true(all(is.finite(as.array(fwd$logabsdet))))

  # `inside` is a strict test, so at the edges themselves the spline passes the
  # input through verbatim with zero log|det|.
  outside <- mx >= tb | mx <= -tb
  expect_gt(sum(outside), 0)
  expect_equal(as.array(fwd$output)[outside], mx[outside])
  expect_equal(as.array(fwd$logabsdet)[outside], rep(0, sum(outside)))

  inv <- spline(x, p$widths, p$heights, p$derivatives,
                inverse = TRUE, tail_bound = tb)
  expect_equal(as.array(inv)[outside], mx[outside])

  # Strictly inside the domain the transform is non-trivial but finite, and
  # forward-then-inverse recovers the input.
  round_trip <- spline(fwd$output, p$widths, p$heights, p$derivatives,
                       inverse = TRUE, tail_bound = tb)
  expect_lt(max(abs(as.array(round_trip)[!outside] - mx[!outside])), 1e-4)
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
  expect_named(s, c("input_dim", "output_dim", "n_layers", "hidden_dim", "style", "n_params"))
  expect_equal(s$input_dim, 6L)
  expect_equal(s$output_dim, 4L)
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
  # MAF pays output_dim sequential passes per block at sample time, so the
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

# ---- n_params tests ----

# Defined before its first use below: testthat evaluates the file top to
# bottom, so a helper declared after the tests that call it is not yet in
# scope when those tests run.
.count_params <- function(mdl) {
  sum(vapply(mdl$parameters, function(p) prod(as.integer(p$shape)), numeric(1)))
}

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
