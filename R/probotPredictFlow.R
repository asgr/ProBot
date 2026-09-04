# Deterministic point estimate for every row of `input`, computed in one pass
# without drawing any samples. `input` is an already-placed 2-D tensor and
# means_t/sds_t are the on-device scale vectors built by the caller (or NULL),
# so nothing here re-derives the device -- that is what keeps a CPU-resident
# model from being handed MPS tensors, and vice versa. See the `point_estimate`
# argument of ?probotSamplePostNF for which value is returned.
.probotFlowPointSummary <- function(input, model, output_dim, means_t, sds_t,
                                    col_names) {
  # The head, when there is one; otherwise the base distribution's mode.
  point <- if (inherits(model, "probotFlowLoc")) "loc" else "centre"

  with_no_grad({
    est <- .probotFlowPointEstimate(model, context = input,
                                    output_dim = output_dim, point = point)
    # 1-D (output_dim) vectors broadcast across rows, matching the sampler's
    # own unscaling line.
    if (!is.null(means_t)) est <- est * sds_t + means_t
  })

  m <- as.matrix(est$cpu())
  dimnames(m) <- list(NULL, col_names)
  if (nrow(m) == 1L) as.vector(m) else m
}

probotSamplePostNF <- function(input,
                               model,
                               n_samples = 5000,
                               col_means = NULL,
                               col_sds = NULL,
                               col_names = NULL,
                               output_dim = NULL,
                               device = NULL,
                               batch_size = NULL,
                               verbose = FALSE,
                               point_estimate = FALSE) {
  # ------------------------------------------------------------------
  # Determine parameter dimensionality
  # ------------------------------------------------------------------

  if (is.null(output_dim)) {
    if (!is.null(col_means)) {
      output_dim <- length(col_means)
    } else {
      stop("Either 'output_dim' or 'col_means' must be provided.")
    }
  }

  if (!is.null(col_means) && is.null(col_sds)) {
    stop("col_sds must be provided when col_means is provided.")
  }

  # ------------------------------------------------------------------
  # Device selection
  # ------------------------------------------------------------------

  if (is.null(device)) {
    if (length(model$parameters) > 0) {
      device <- model$parameters[[1]]$device
    } else {
      device <-
        if (backends_mps_is_available()) {
          torch_device("mps")
        } else {
          torch_device("cpu")
        }
    }
  }

  # ------------------------------------------------------------------
  # Convert input to tensor
  # ------------------------------------------------------------------

  if (!inherits(input, "torch_tensor")) {
    input <- torch_tensor(input, dtype = torch_float(), device = device)
  } else {
    input <- input$to(device = device)
  }

  model$eval()

  # ------------------------------------------------------------------
  # Pre-build on-device scale vectors so unscaling can happen in a
  # single fused broadcast op instead of an R-side per-column loop.
  # ------------------------------------------------------------------

  means_t <- NULL
  sds_t <- NULL

  if (!is.null(col_means)) {
    means_t <- torch_tensor(col_means, dtype = torch_float(), device = device)
    sds_t <- torch_tensor(col_sds, dtype = torch_float(), device = device)

    # Recycle length-1 vectors to output_dim (mirrors probotScaleBackward).
    if (means_t$size(1) == 1L) means_t <- means_t$expand(c(output_dim))
    if (sds_t$size(1) == 1L) sds_t <- sds_t$expand(c(output_dim))
  }

  if (input$dim() == 1L) {
    input <- input$unsqueeze(1)
  }

  # ------------------------------------------------------------------
  # POINT ESTIMATE (no sampling)
  #
  # For a residual location-head flow this returns mu(x), the head's own
  # output: deterministic, exact, and the thing the head was trained to
  # predict. For a plain flow there is no head, so the inverse of z = 0
  # (the base distribution's mode) is the closest analogue. Both cost a
  # single sweep, which is why this lives here rather than in its own
  # function. Dispatched after the device and scale tensors are resolved
  # so it cannot disagree with the model about where tensors live.
  # ------------------------------------------------------------------

  if (isTRUE(point_estimate)) {
    if (input$dim() != 2L) {
      stop("'input' must be either a vector or a matrix")
    }
    if (isTRUE(verbose)) {
      warning("point_estimate = TRUE returns one value per row, so 'verbose' ",
              "(chunk progress for sampling) is ignored.", call. = FALSE)
    }
    return(.probotFlowPointSummary(input = input, model = model,
                                   output_dim = output_dim,
                                   means_t = means_t, sds_t = sds_t,
                                   col_names = col_names))
  }

  # ------------------------------------------------------------------
  # SINGLE OBSERVATION MODE
  # ------------------------------------------------------------------

  if (input$dim() == 2L && input$size(1) == 1) {
    z_base <- torch_randn(c(n_samples, output_dim), device = device)

    context_expanded <- input$expand(c(n_samples, input$size(2)))

    with_no_grad({
      theta_t <- model$inverse(z_base, context_expanded)
    })

    if (!is.null(col_means)) {
      theta_t <- theta_t * sds_t + means_t
    }

    samples <- as.matrix(theta_t$cpu())

    colnames(samples) <- col_names

    return(samples)
  }

  # ------------------------------------------------------------------
  # MULTI-OBSERVATION MODE
  # input shape:   (N_obs, N_features)
  # returns:       array of shape (n_samples, output_dim, N_obs)
  #
  # Observations are processed in chunks of `batch_size` rows so the
  # large temporaries (z, expanded context, theta) stay bounded in
  # memory. This is what makes millions of observations feasible:
  # everything is computed on-device in bounded batches and only the
  # accumulated result is ever held in R.
  # ------------------------------------------------------------------

  if (input$dim() != 2L) {
    stop("'input' must be either a vector or a matrix")
  }

  N_obs <- input$size(1)
  N_feat <- input$size(2)

  # Auto-size the chunk so that batch_size * n_samples rows is held in
  # memory at once (~2e6 rows default). Override with batch_size for
  # machines with less/more headroom.
  if (is.null(batch_size)) {
    batch_size <- max(1L, floor(2e6 / n_samples))
  }

  batch_size <- min(batch_size, N_obs)

  n_chunks <- ceiling(N_obs / batch_size)
  progress_every <- max(1L, floor(n_chunks / 20))

  # Pre-allocate the full output once: (Sample, Parameter, Observation).
  out <- array(NA_real_, c(n_samples, output_dim, N_obs))

  chunk_i <- 0L

  for (start in seq(1L, N_obs, by = batch_size)) {
    chunk_i <- chunk_i + 1L
    end <- min(start + batch_size - 1L, N_obs)
    B <- end - start + 1L

    # Conditioning rows for this chunk: (B, N_feat); torch narrow is 1-based
    ctx_chunk <- input$narrow(1, start, B)

    # Latent samples for this chunk: (B, n_samples, output_dim)
    z_chunk <- torch_randn(c(B, n_samples, output_dim), device = device)
    z_flat <- z_chunk$reshape(c(B * n_samples, output_dim))

    # Duplicate each context row n_samples times: (B*n_samples, N_feat)
    context_flat <-
      ctx_chunk$unsqueeze(2)$expand(c(B, n_samples, N_feat))$reshape(
        c(B * n_samples, N_feat)
      )

    with_no_grad({
      theta_flat <- model$inverse(z_flat, context_flat) # (B*n_samples, output_dim)
    })

    # Unscale on-device in one fused broadcast.
    if (!is.null(col_means)) {
      theta_flat <- theta_flat * sds_t + means_t
    }

    # (B, n_samples, output_dim) -> (n_samples, output_dim, B) to match `out`.
    block <- theta_flat$reshape(c(B, n_samples, output_dim))$permute(c(2, 3, 1))

    out[, , start:end] <- as.array(block$cpu())

    if (verbose && (chunk_i %% progress_every == 0L || chunk_i == n_chunks)) {
      cat(sprintf(
        "probotSamplePostNF: chunk %d/%d (obs %d-%d)\n",
        chunk_i, n_chunks, start, end
      ))
    }
  }

  if (!is.null(col_names)) {
    dimnames(out) <- list(
      Sample = seq_len(n_samples),
      Parameter = col_names,
      Observation = seq_len(N_obs)
    )
  }

  return(out)
}
