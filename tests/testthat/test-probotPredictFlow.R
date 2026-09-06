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

