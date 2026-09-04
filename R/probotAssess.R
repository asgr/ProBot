probotPIT <- function(
    input,
    model,
    mdn_components = NULL,
    params,
    n_test = NULL,
    n_samples = 1e4,
    col_means = NULL,
    col_sds = NULL,
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
    mdn_components = mdn_components, output_dim = n_params,
    n_samples = n_samples,
    col_means = col_means, col_sds = col_sds, col_names = col_names,
    batch_size = batch_size, verbose = verbose, label = "probotPIT",
    fun = function(rows, draws) {
      # Flattening the (S x D x K) array to (S x D*K) leaves parameter fastest
      # within observation, which is the order as.vector(t(params)) produces.
      mat <- matrix(draws, nrow = n_samples)
      truth <- as.vector(t(params[rows, , drop = FALSE]))
      cmp <- sweep(mat, 2L, truth, FUN = "<=")
      pit[rows, ] <<- t(matrix(colSums(cmp) / n_samples, nrow = n_params))
    }
  )

  colnames(pit) <- col_names

  return(pit)
}

probotTARP <- function(
    input,
    model,
    mdn_components = NULL,
    params,
    n_test = NULL,
    n_samples = 1e4,
    col_means = NULL,
    col_sds = NULL,
    col_names = NULL,
    batch_size = NULL,
    verbose = TRUE
){

  n_test <- .probotNTest(n_test, params)

  n_params <- ncol(params)

  tarp <- numeric(n_test)

  if (n_test < 1L) {
    return(tarp)
  }

  .probotChunkApply(
    input = input, n_test = n_test, model = model,
    mdn_components = mdn_components, output_dim = n_params,
    n_samples = n_samples,
    col_means = col_means, col_sds = col_sds, col_names = col_names,
    batch_size = batch_size, verbose = verbose, label = "probotTARP",
    fun = function(rows, draws) {
      n_obs <- length(rows)

      # A fresh unit direction per observation, drawn on the R side and held as
      # one (n_params x n_obs) matrix.
      direction <- matrix(rnorm(n_params * n_obs), nrow = n_params)
      direction <- direction / sqrt(colSums(direction^2))

      # proj[s, k] = sum_d draws[s, d, k] * direction[d, k]. Accumulated over
      # the (small) parameter axis because R's elementwise recycling cannot
      # broadcast a length-n_params vector across the sample axis directly.
      proj <- matrix(0, nrow = n_samples, ncol = n_obs)

      for (d in seq_len(n_params)) {
        proj <- proj +
          matrix(draws[, d, ], nrow = n_samples) *
          matrix(rep(direction[d, ], each = n_samples), nrow = n_samples)
      }

      truth_proj <- colSums(direction * t(params[rows, , drop = FALSE]))

      # Each observation's truth is constant down its column of `proj`, so the
      # threshold must be repeated n_samples times, not recycled blindly.
      truth_mat <- matrix(rep(truth_proj, each = n_samples),
                          nrow = n_samples, ncol = n_obs)

      tarp[rows] <<- colMeans(proj <= truth_mat)
    }
  )

  return(tarp)
}

probotCRPS <- function(
    input,
    model,
    mdn_components = NULL,
    params,
    n_test = NULL,
    n_samples = 1e4,
    col_means = NULL,
    col_sds = NULL,
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
    mdn_components = mdn_components, output_dim = n_params,
    n_samples = n_samples,
    col_means = col_means, col_sds = col_sds, col_names = col_names,
    batch_size = batch_size, verbose = verbose, label = "probotCRPS",
    fun = function(rows, draws) {
      mat <- matrix(draws, nrow = n_samples)
      truth <- as.vector(t(params[rows, , drop = FALSE]))

      mae <- colMeans(abs(sweep(mat, 2L, truth, FUN = "-")))
      penalty <- apply(mat, 2L, function(z) sum(w * sort(z)))

      crps[rows, ] <<- t(matrix(mae - penalty, nrow = n_params))
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

# Flow architectures are recognised by their nn_module class names; anything
# else is treated as an MDN. probotFlowLoc wraps one of the three base styles and
# exposes the same forward()/inverse() contract, so it is dispatched to the flow
# sampler exactly like them.
.probotFlowClasses <- c("probotFlowRealNVP", "probotFlowMAF", "probotFlowNSF",
                        "probotFlowLoc")

.probotIsFlow <- function(model) {
  inherits(model, .probotFlowClasses)
}

# Resolve the sampler that backs the assessment functions and return a closure
# with a common signature, so callers stay model-agnostic. A flow has no
# mixture head to read the parameter dimension from, so output_dim must be
# supplied; it is unused on the MDN path.
.probotPostSampler <- function(model, mdn_components, output_dim) {
  if (.probotIsFlow(model)) {
    if (is.null(output_dim)) {
      stop("'output_dim' is required when sampling from a flow model.")
    }
    function(input, n_samples, col_means, col_sds, col_names, batch_size,
             verbose) {
      probotSamplePostNF(
        input = input,
        model = model,
        n_samples = n_samples,
        col_means = col_means,
        col_sds = col_sds,
        col_names = col_names,
        output_dim = output_dim,
        batch_size = batch_size,
        verbose = verbose
      )
    }
  } else {
    if (is.null(mdn_components)) {
      stop(
        "'mdn_components' is required for MDN models (it is ignored for ",
        "normalising flow models)."
      )
    }
    function(input, n_samples, col_means, col_sds, col_names, batch_size,
             verbose) {
      probotSamplePostMDN(
        input = input,
        model = model,
        mdn_components = mdn_components,
        n_samples = n_samples,
        col_means = col_means,
        col_sds = col_sds,
        col_names = col_names,
        batch_size = batch_size,
        verbose = verbose
      )
    }
  }
}

# Bulk-sample the posterior for the first n_test rows of `input` in chunks of
# `batch_size` observations, calling fun(rows, draws) on each. `draws` is an
# (n_samples x n_params x length(rows)) array. Chunks are generated lazily and
# handed straight to `fun`, so peak memory is set by the chunk size rather than
# by n_test, while the sampling pass stays batched. The sampler is chosen from
# the class of `model`, so MDN and flow models both work. `label` names the
# calling function in the progress messages.
.probotChunkApply <- function(
    input,
    n_test,
    model,
    mdn_components,
    output_dim,
    n_samples,
    col_means,
    col_sds,
    col_names,
    batch_size,
    verbose,
    label = "probotChunkApply",
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

  sample_post <- .probotPostSampler(
    model = model, mdn_components = mdn_components, output_dim = output_dim
  )

  if (is.null(batch_size)) {
    # Size the chunk so that roughly 2e6 draw rows exist at any one time.
    batch_size <- max(1L, floor(2e6 / n_samples))
  }

  batch_size <- as.integer(min(batch_size, n_test))

  n_chunks <- as.integer(ceiling(n_test / batch_size))
  # At most ~20 progress lines however many chunks there are, but always the
  # first and last, so a long run reports that it has started.
  progress_every <- max(1L, as.integer(floor(n_chunks / 20)))

  chunk_i <- 0L

  for (start in seq(1L, n_test, by = batch_size)) {

    chunk_i <- chunk_i + 1L

    end <- min(start + batch_size - 1L, n_test)

    rows <- seq(start, end)

    if (verbose) {
      first_or_last <- chunk_i == 1L || chunk_i == n_chunks
      if (first_or_last || chunk_i %% progress_every == 0L) {
        cat(sprintf(
          "%s: chunk %d/%d (obs %d-%d of %d)\n",
          label, chunk_i, n_chunks, start, end, n_test
        ))
      }
    }

    draws <- sample_post(
      input = input[rows, , drop = FALSE],
      n_samples = n_samples,
      col_means = col_means,
      col_sds = col_sds,
      col_names = col_names,
      # The sampler chunks internally too; hand it the whole window so that
      # exactly one chunk of draws is in flight at a time. That also makes its
      # own progress reporting a constant "chunk 1/1", so it stays quiet and
      # the messages above, which know the superset, do the reporting.
      batch_size = length(rows),
      verbose = FALSE
    )

    # A single-observation window comes back as a matrix; restore the array
    # shape the callers expect.
    if (length(dim(draws)) == 2L) {
      draws <- array(draws, dim = c(dim(draws), 1L))
    }

    if (dim(draws)[2] != output_dim) {
      stop(
        "Posterior samples have ", dim(draws)[2], " dimensions but params has ",
        output_dim, ". `params` (and any col_means/col_sds) must match the ",
        "model's parameter space."
      )
    }

    fun(rows, draws)
  }

  invisible(NULL)
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
