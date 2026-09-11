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

# --------------------------------------------------------------------
# Memory budgeting. Peak device memory tracks (rows fed to inverse) *
# (widest internal layer), not rows * output_dim, so the old flat 2e6-row
# default asks for tens of GB on a hidden_dim = 512 flow and dies.
# --------------------------------------------------------------------

# Identity-inverse mock whose widest parameter row is `act`, so the row budget
# is controllable without allocating a real wide network.
mock_flow <- function(output_dim, act) {
  mdl <- list(parameters = list(torch_tensor(matrix(0, act, 2))))
  mdl$eval <- function() NULL
  mdl$inverse <- function(z, x) z
  # Classed as a flow so .probotIsFlow() recognises it, but left as a
  # plain list so torch's $.nn_module method cannot intercept $.
  class(mdl) <- "probotFlowRealNVP"
  mdl
}

test_that(".probotFlowActWidth reads the widest internal layer", {
  narrow <- probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 32, device = "cpu")()
  expect_equal(.probotFlowActWidth(narrow), 32L)

  wide <- probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 64, device = "cpu")()
  expect_equal(.probotFlowActWidth(wide), 64L)

  # A location head is wider than the flow beneath it and dominates the budget.
  headed <- probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 8,
                           loc_head = TRUE, loc_hidden_dims = c(100, 200),
                           device = "cpu")()
  expect_equal(.probotFlowActWidth(headed), 200L)
})

test_that(".probotFlowChunkRows keeps narrow flows at the original 2e6 rows", {
  # hidden_dim = 32 is the package default, so the common case must not shift:
  # changing the chunk size changes the torch RNG stream.
  expect_equal(.probotFlowChunkRows(32L, "cpu"), 2e6)
  expect_equal(.probotFlowChunkRows(16L, "cpu"), 2e6)

  # Wide conditioners get a row budget, and it is inversely proportional.
  cpu512 <- .probotFlowChunkRows(512L, "cpu")
  expect_lt(cpu512, 2e6)
  expect_equal(cpu512, floor(6.4e7 / 512))
  expect_lt(.probotFlowChunkRows(1024L, "cpu"), cpu512)

  # MPS holds activations on-device and measured cheaper per row.
  expect_gt(.probotFlowChunkRows(512L, "mps"), cpu512)
  # a plain string device must work as well as a torch_device
  expect_equal(.probotFlowChunkRows(512L, torch_device("cpu")), cpu512)
})

test_that("probotSamplePostNF clamps an out-of-budget batch_size with a warning", {
  output_dim <- 3
  mdl <- mock_flow(output_dim, act = 64000L)   # -> row budget of exactly 1000
  expect_equal(.probotFlowChunkRows(.probotFlowActWidth(mdl), "cpu"), 1000)

  input <- matrix(runif(10 * 2), nrow = 10)

  # 8 obs x 500 samples = 4000 rows, over the 1000-row budget: warn, clamp to 2.
  expect_warning(
    s <- probotSamplePostNF(input, mdl, n_samples = 500, output_dim = output_dim,
                            device = "cpu", batch_size = 8),
    "not memory-safe, so it is reduced to 2"
  )
  expect_equal(dim(s), c(500, output_dim, 10))
  expect_true(all(is.finite(s)))

  # Within budget: no warning at all.
  expect_silent(
    probotSamplePostNF(input, mdl, n_samples = 100, output_dim = output_dim,
                       device = "cpu", batch_size = 5)
  )
})

test_that("single-observation mode chunks n_samples and stays reproducible", {
  output_dim <- 3
  mdl <- mock_flow(output_dim, act = 64000L)   # row budget 1000
  input <- matrix(runif(2), nrow = 1)

  big <- probotSamplePostNF(input, mdl, n_samples = 2500, output_dim = output_dim,
                            device = "cpu")
  expect_equal(dim(big), c(2500, output_dim))
  expect_true(all(is.finite(big)))

  # The first budget-sized block must match an unchunked call at that size:
  # torch draws from one stream, so chunking may not shift where a chunk starts.
  set.seed(7); torch_manual_seed(123)
  one_chunk <- probotSamplePostNF(input, mdl, n_samples = 1000, output_dim = output_dim,
                                  device = "cpu")
  set.seed(7); torch_manual_seed(123)
  chunked <- probotSamplePostNF(input, mdl, n_samples = 2000, output_dim = output_dim,
                                device = "cpu")
  expect_equal(chunked[1:1000, ], one_chunk, tolerance = 0)
})

test_that(".probotChunkApply budgets flow chunks but leaves MDNs alone", {
  output_dim <- 3
  mdl <- mock_flow(output_dim, act = 64000L)   # row budget 1000
  input <- matrix(runif(6 * 2), nrow = 6)
  params <- matrix(rnorm(6 * output_dim), nrow = 6)

  # n_samples 500 -> 2 obs per chunk, so 6 obs needs 3 chunks and must not error.
  # The default path must warn nowhere: the cap is applied by the chunk loop.
  pit <- NULL
  expect_silent(pit <- probotPIT(input, model = mdl, params = params,
                                 n_samples = 500, verbose = FALSE))
  expect_equal(dim(pit), c(6, output_dim))
  expect_true(all(pit >= 0 & pit <= 1))

  # An explicit over-budget batch_size warns exactly once, not once per chunk.
  msgs <- character()
  withCallingHandlers(
    probotPIT(input, model = mdl, params = params, n_samples = 500,
              batch_size = 6, verbose = FALSE),
    warning = function(cc) {
      msgs <<- c(msgs, conditionMessage(cc)); invokeRestart("muffleWarning")
    })
  expect_length(msgs, 1)
  expect_match(msgs[1], "reduced to 2")
})


# ============================================================
# probotSigmaPostNF
# ============================================================

sigma_test_flow <- function(style = "realnvp", di = 2, D = 3, layers = 4, hid = 16) {
  probotMakeFlow(di, D, n_layers = layers, hidden_dim = hid,
                 device = "cpu", style = style)()
}

test_that("probotSigmaPostNF returns mean/sd matrices with names", {
  mdl <- sigma_test_flow()
  X <- matrix(rnorm(20 * 2), 20, 2)
  r <- probotSigmaPostNF(X, mdl, output_dim = 3,
                         col_names = c("A", "B", "C"))
  expect_named(r, c("post_mean", "post_sd"))
  expect_equal(dim(r$post_mean), c(20, 3))
  expect_equal(dim(r$post_sd), c(20, 3))
  expect_equal(colnames(r$post_sd), c("A", "B", "C"))
  expect_true(all(is.finite(unlist(r))))
  expect_true(all(r$post_sd >= 0))

  # A vector input must collapse to length-D vectors, matching row 1.
  r1 <- probotSigmaPostNF(X[1, ], mdl, output_dim = 3, col_names = c("A", "B", "C"))
  expect_equal(r1$post_sd, as.vector(r$post_sd[1, ]), tolerance = 1e-5)
  expect_equal(r1$post_mean, as.vector(r$post_mean[1, ]), tolerance = 1e-5)
})

test_that("probotSigmaPostNF sd is exactly the Jacobian row norm", {
  # Independent brute-force reference: perturb one base axis at a time with a
  # full 2-D batch, difference over 2 * eps, then take row norms.
  mdl <- sigma_test_flow()
  X <- matrix(rnorm(15 * 2), 15, 2)
  D <- 3; eps <- 1e-3
  xt <- torch_tensor(X, dtype = torch_float())
  J <- array(0, c(15, D, D))
  for (k in seq_len(D)) {
    zp <- zm <- matrix(0, 15, D)
    zp[, k] <- eps
    zm[, k] <- -eps
    up <- as.matrix(with_no_grad(mdl$inverse(torch_tensor(zp), xt)))
    lo <- as.matrix(with_no_grad(mdl$inverse(torch_tensor(zm), xt)))
    J[, , k] <- (up - lo) / (2 * eps)
  }
  ref_sd <- t(apply(J, 1, function(M) sqrt(rowSums(M^2))))

  r <- probotSigmaPostNF(X, mdl, output_dim = D, eps = eps, batch_size = 4)
  expect_equal(r$post_sd, ref_sd, tolerance = 1e-4)

  # Negative control: the COLUMN norms are a different quantity (this is the
  # dim = 2 vs dim = 3 trap), so they must not match.
  col_norms <- t(apply(J, 1, function(M) sqrt(colSums(M^2))))
  expect_false(isTRUE(all.equal(r$post_sd, col_norms, tolerance = 1e-2)))
})

test_that("probotSigmaPostNF centre matches point_estimate = TRUE", {
  for (style in c("realnvp", "maf", "nsf")) {
    mdl <- sigma_test_flow(style)
    X <- matrix(rnorm(12 * 2), 12, 2)
    r <- probotSigmaPostNF(X, mdl, output_dim = 3)
    pe <- probotSamplePostNF(X, mdl, output_dim = 3, point_estimate = TRUE)
    expect_equal(as.vector(r$post_mean), as.vector(pe),
                 tolerance = 1e-6, info = style)
  }
})

test_that("probotSigmaPostNF sd agrees with sampling for a learnable problem", {
  # Trained, in-distribution, and symmetric enough that a first-order
  # approximation should hold. RealNVP/MAF track closely; the tolerance below
  # is loose because the estimator is approximate by construction, and the
  # reference itself carries Monte-Carlo error.
  for (style in c("realnvp", "maf")) {
    set.seed(11)
    n <- 2000L
    x <- matrix(rnorm(n * 3), n, 3)
    th <- cbind(x %*% c(1, -1, .5) + .3 * rnorm(n),
                .5 * x[, 2] + .4 * rnorm(n),
                x[, 3] * .2 + .6 * rnorm(n))
    dl <- probotDataLoader(x, th, batch = 256, shuffle = TRUE, device = "cpu")
    mdl <- probotMakeFlow(3, 3, n_layers = 4, hidden_dim = 32,
                          device = "cpu", style = style)()
    probotTrainFlow(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                    epochs = 40, verbose = FALSE, early_stop = FALSE)

    te <- matrix(rnorm(60 * 3), 60, 3)
    r <- probotSigmaPostNF(te, mdl, output_dim = 3, batch_size = 30)
    S <- probotSamplePostNF(te, mdl, n_samples = 2000, output_dim = 3,
                            batch_size = 10)
    ss <- t(apply(S, c(2, 3), sd))
    ratio <- colMeans(r$post_sd / ss)
    expect_true(all(ratio > 0.8 & ratio < 1.25), info = style)
  }
})

test_that("probotSigmaPostNF unscales mean and sd consistently", {
  mdl <- sigma_test_flow()
  X <- matrix(rnorm(10 * 2), 10, 2)
  cm <- c(10, 20, 30); cs <- c(2, 3, 4)
  raw <- probotSigmaPostNF(X, mdl, output_dim = 3)
  uns <- probotSigmaPostNF(X, mdl, col_means = cm, col_sds = cs)

  expect_equal(uns$post_mean, raw$post_mean * matrix(cs, 10, 3, byrow = TRUE) +
                              matrix(cm, 10, 3, byrow = TRUE), tolerance = 1e-4)
  # sd scales by col_sds only, never by col_means.
  expect_equal(uns$post_sd, raw$post_sd * matrix(cs, 10, 3, byrow = TRUE),
               tolerance = 1e-4)

  # A scalar col_sds is recycled across output_dim, exactly as in the sampler;
  # output_dim must be given explicitly, since col_means no longer implies it.
  one <- probotSigmaPostNF(X, mdl, col_means = 5, col_sds = 2, output_dim = 3)
  expect_equal(one$post_sd, raw$post_sd * 2, tolerance = 1e-4)
  expect_equal(one$post_mean, raw$post_mean * 2 + 5, tolerance = 1e-4)
})

test_that("probotSigmaPostNF is deterministic in batch_size", {
  # The whole point of a deterministic grid: unlike the sampler, chunking
  # cannot change the answer.
  mdl <- sigma_test_flow()
  X <- matrix(rnorm(12 * 2), 12, 2)
  a <- probotSigmaPostNF(X, mdl, output_dim = 3, batch_size = 1)
  b <- probotSigmaPostNF(X, mdl, output_dim = 3, batch_size = 12)
  expect_equal(a$post_sd, b$post_sd, tolerance = 1e-4)
  expect_equal(a$post_mean, b$post_mean, tolerance = 1e-4)
})

test_that("probotSigmaPostNF clamps an over-budget batch_size with a warning", {
  output_dim <- 3
  mdl <- mock_flow(output_dim, act = 64000L)  # row budget of exactly 1000
  # n_probe = 2 * 3 + 1 = 7 rows/obs, so the cap is floor(1000 / 7) = 142.
  input <- matrix(runif(200 * 2), nrow = 200)

  expect_warning(
    r <- probotSigmaPostNF(input, mdl, output_dim = output_dim, batch_size = 200),
    "not memory-safe, so it is reduced to 142"
  )
  expect_equal(dim(r$post_sd), c(200, 3))

  # mock inverse(z, x) = z, so J is the identity: every sd is exactly 1 and
  # every centre exactly 0. Clamping must not disturb that.
  expect_equal(as.vector(r$post_sd), rep(1, 600), tolerance = 1e-5)
  expect_equal(as.vector(r$post_mean), rep(0, 600), tolerance = 1e-5)

  # Chunking is result-invariant here (deterministic grid), unlike the sampler.
  r2 <- probotSigmaPostNF(input, mdl, output_dim = output_dim, batch_size = 3)
  expect_equal(r2$post_sd, r$post_sd, tolerance = 1e-6)
})

test_that("probotSigmaPostNF errors on the same inputs as the sampler", {
  mdl <- sigma_test_flow()
  X <- matrix(rnorm(4 * 2), 4, 2)
  expect_error(probotSigmaPostNF(X, mdl), "Either 'output_dim' or 'col_means'")
  expect_error(probotSigmaPostNF(X, mdl, col_means = c(1, 2, 3)),
               "col_sds must be provided")
  expect_error(probotSigmaPostNF(X, mdl, output_dim = 3, eps = 0),
               "'eps' must be a single positive number")
  expect_error(probotSigmaPostNF(X, mdl, output_dim = 3, eps = NA_real_),
               "'eps' must be a single positive number")
  expect_error(probotSigmaPostNF(array(rnorm(8), c(2, 2, 2)), mdl, output_dim = 3),
               "must be either a vector or a matrix")
})

test_that("probotSigmaPostNF works for a location-head flow", {
  mdl <- probotMakeFlow(2, 3, n_layers = 2, hidden_dim = 8, loc_head = TRUE,
                        device = "cpu")()
  X <- matrix(rnorm(6 * 2), 6, 2)
  r <- probotSigmaPostNF(X, mdl, output_dim = 3)
  expect_equal(dim(r$post_sd), c(6, 3))
  expect_true(all(is.finite(unlist(r))))
  # A head shifts the centre by mu(x) but leaves the residual scale alone, so
  # post_mean must NOT equal probotSamplePostNF's point_estimate (= mu(x)).
  pe <- probotSamplePostNF(X, mdl, output_dim = 3, point_estimate = TRUE)
  expect_false(isTRUE(all.equal(as.vector(r$post_mean), as.vector(pe),
                                tolerance = 1e-2)))
})

test_that(".probotFlowPointEstimate grid branch validates and tiles", {
  mdl <- sigma_test_flow()
  X <- matrix(rnorm(5 * 2), 5, 2)
  xt <- torch_tensor(X, dtype = torch_float())

  expect_error(.probotFlowPointEstimate(mdl, xt, output_dim = 3, point = "grid"),
               "requires the 'z_pts'")
  expect_error(.probotFlowPointEstimate(mdl, xt, output_dim = 3, point = "grid",
                                        z_pts = matrix(0, 4, 4)),
               "must be an \\(m, output_dim\\) grid")

  out <- .probotFlowPointEstimate(mdl, xt, output_dim = 3, point = "grid",
                                  z_pts = matrix(0, 1, 3))
  expect_equal(out$shape, c(5, 1, 3))
  # z = 0 for every row, so every row must equal the centre sweep.
  centre <- .probotFlowPointEstimate(mdl, xt, output_dim = 3, point = "centre")
  expect_equal(as.matrix(out[, 1, ]), as.matrix(centre), tolerance = 1e-6)

  # A single-row grid given as a vector is promoted, not rejected.
  v <- .probotFlowPointEstimate(mdl, xt, output_dim = 3, point = "grid",
                                z_pts = c(0, 0, 0))
  expect_equal(v$shape, c(5, 1, 3))
})

test_that(".probotSigmaGrid lays out axis-major probes", {
  g <- ProBot:::.probotSigmaGrid(c(0.1, 0.2), 3)
  expect_equal(dim(g), c(6, 3))
  expect_equal(g[1, ], c(0.1, 0, 0))
  expect_equal(g[2, ], c(0.2, 0, 0))
  expect_equal(g[3, ], c(0, 0.1, 0))
  expect_equal(g[6, ], c(0, 0, 0.2))
})
