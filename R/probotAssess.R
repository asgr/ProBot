probotPIT <- function(
    input,
    model,
    mdn_components,
    params,
    n_test = NULL,
    n_samples = 1e4,
    col_means,
    col_sds,
    col_names = NULL,
    batch_size = NULL,
    verbose = TRUE
){

  n_test <- .probotNTest(n_test, params)

  n_params <- ncol(params)

  pit <- matrix(
    NA_real_,
    n_test,
    n_params
  )

  if (n_test < 1L) {
    colnames(pit) <- col_names
    return(pit)
  }

  # One batched sample call per chunk: each chunk's draws are consumed and
  # released before the next is allocated, so peak memory tracks batch_size
  # rather than n_test.
  .probotChunkApply(
    input = input, n_test = n_test, model = model,
    mdn_components = mdn_components, n_samples = n_samples,
    col_means = col_means, col_sds = col_sds, col_names = col_names,
    batch_size = batch_size, verbose = verbose,
    fun = function(rows, draws) {
      for (k in seq_along(rows)) {
        i <- rows[k]
        obs <- .probotObsDraws(draws, k)
        for (j in seq_len(n_params)) {
          pit[i, j] <<- mean(obs[, j] <= params[i, j])
        }
      }
    }
  )

  colnames(pit) <- col_names

  return(pit)
}

probotTARP <- function(
    input,
    model,
    mdn_components,
    params,
    n_test = NULL,
    n_samples = 1e4,
    col_means,
    col_sds,
    col_names = NULL,
    batch_size = NULL,
    verbose = TRUE
){

  n_test <- .probotNTest(n_test, params)

  tarp <- numeric(n_test)

  if (n_test < 1L) {
    return(tarp)
  }

  .probotChunkApply(
    input = input, n_test = n_test, model = model,
    mdn_components = mdn_components, n_samples = n_samples,
    col_means = col_means, col_sds = col_sds, col_names = col_names,
    batch_size = batch_size, verbose = verbose,
    fun = function(rows, draws) {
      for (k in seq_along(rows)) {
        i <- rows[k]
        obs <- .probotObsDraws(draws, k)

        # A fresh unit direction per observation, drawn on the R side.
        direction <- rnorm(ncol(obs))
        direction <- direction / sqrt(sum(direction^2))

        sample_proj <- as.vector(obs %*% direction)
        truth_proj <- sum(params[i, ] * direction)

        tarp[i] <<- mean(sample_proj <= truth_proj)
      }
    }
  )

  return(tarp)
}

probotCRPS <- function(
    input,
    model,
    mdn_components,
    params,
    n_test = NULL,
    n_samples = 1e4,
    col_means,
    col_sds,
    col_names = NULL,
    batch_size = NULL,
    verbose = TRUE
){

  n_test <- .probotNTest(n_test, params)

  n_params <- ncol(params)

  crps <- matrix(NA_real_, n_test, n_params)

  if (n_test < 1L) {
    colnames(crps) <- col_names
    return(crps)
  }

  w <- .probotCRPSWeights(n_samples)

  .probotChunkApply(
    input = input, n_test = n_test, model = model,
    mdn_components = mdn_components, n_samples = n_samples,
    col_means = col_means, col_sds = col_sds, col_names = col_names,
    batch_size = batch_size, verbose = verbose,
    fun = function(rows, draws) {
      for (k in seq_along(rows)) {
        i <- rows[k]
        obs <- .probotObsDraws(draws, k)
        for (j in seq_len(n_params)) {
          crps[i, j] <<- .probotCRPSSample(obs[, j], params[i, j], w)
        }
      }
    }
  )

  colnames(crps) <- col_names

  return(crps)
}

# ----------------------------------------------------------------------
# Shared internals
# ----------------------------------------------------------------------

# n_test defaults to every row of params, and is capped at nrow(params).
.probotNTest <- function(n_test, params) {
  if (is.null(n_test)) {
    as.integer(nrow(params))
  } else {
    as.integer(pmin(n_test, nrow(params), na.rm = TRUE))
  }
}

# Bulk-sample the posterior for the first n_test rows of `input` in chunks of
# `batch_size` observations, calling fun(rows, draws) on each. `draws` is an
# (n_samples x n_params x length(rows)) array. Chunks are generated lazily and
# handed straight to `fun`, so peak memory is set by the chunk size rather than
# by n_test, while the forward pass stays batched.
.probotChunkApply <- function(
    input,
    n_test,
    model,
    mdn_components,
    n_samples,
    col_means,
    col_sds,
    col_names,
    batch_size,
    verbose,
    fun
){
  # A bare vector is one observation; as.matrix() would orient it as a column.
  if (is.null(dim(input))) {
    input <- matrix(input, nrow = 1L)
  } else {
    input <- as.matrix(input)
  }

  if (nrow(input) < n_test) {
    stop("input must have at least n_test rows (got ", nrow(input),
         " for n_test = ", n_test, ").")
  }

  if (is.null(batch_size)) {
    # Size the chunk so that roughly 2e6 draw rows exist at any one time.
    batch_size <- max(1L, floor(2e6 / n_samples))
  }

  batch_size <- as.integer(min(batch_size, n_test))

  for (start in seq(1L, n_test, by = batch_size)) {

    end <- min(start + batch_size - 1L, n_test)

    rows <- seq(start, end)

    draws <- probotSamplePostMDN(
      input = input[rows, , drop = FALSE],
      model = model,
      mdn_components = mdn_components,
      n_samples = n_samples,
      col_means = col_means,
      col_sds = col_sds,
      col_names = col_names,
      # The sampler chunks internally too; hand it the whole window so that
      # exactly one chunk of draws is in flight at a time.
      batch_size = length(rows),
      verbose = verbose
    )

    # A single-observation window comes back as a matrix; restore the array
    # shape the callers expect.
    if (length(dim(draws)) == 2L) {
      draws <- array(draws, dim = c(dim(draws), 1L))
    }

    fun(rows, draws)
  }

  invisible(NULL)
}

# One observation's (n_samples x n_params) draw matrix. Indexing a 3-D array
# with [, , k] drops *both* length-1 dimensions, so a 1-parameter model would
# come back as a bare vector; rebuild the matrix explicitly.
.probotObsDraws <- function(draws, k) {
  matrix(draws[, , k], nrow = dim(draws)[1], ncol = dim(draws)[2])
}

# Weights for the sorted-sample penalty term (2k - S - 1) / S^2.
# The CRPS estimator's spread term is
#   (1/(2 S^2)) * sum_i sum_j |z_i - z_j|,
# which for sorted z equals sum_k ((2k - S - 1) / S^2) * z_(k).
.probotCRPSWeights = function(n_samples) {
  (2 * seq_len(n_samples) - (n_samples + 1L)) / n_samples^2
}

# CRPS for a single parameter dimension from posterior samples.
.probotCRPSSample = function(samples, y, w) {
  mean(abs(y - samples)) - sum(w * sort(samples))
}
