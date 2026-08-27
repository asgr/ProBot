probotSamplePostNF <- function(input,
                               model,
                               n_samples = 5000,
                               col_means = NULL,
                               col_sds = NULL,
                               col_names = NULL,
                               dim_theta = NULL,
                               device = NULL,
                               batch_size = NULL,
                               verbose = FALSE) {
  # ------------------------------------------------------------------
  # Determine parameter dimensionality
  # ------------------------------------------------------------------

  if (!is.null(dim_theta)) {
    n_dim <- dim_theta
  } else if (!is.null(col_means)) {
    n_dim <- length(col_means)
  } else {
    stop("Either 'dim_theta' or 'col_means' must be provided.")
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

    # Recycle length-1 vectors to n_dim (mirrors probotScaleBackward).
    if (means_t$size(1) == 1L) means_t <- means_t$expand(c(n_dim))
    if (sds_t$size(1) == 1L) sds_t <- sds_t$expand(c(n_dim))
  }

  if (input$dim() == 1L) {
    input <- input$unsqueeze(1)
  }

  # ------------------------------------------------------------------
  # SINGLE OBSERVATION MODE
  # ------------------------------------------------------------------

  if (input$dim() == 2L && input$size(1) == 1) {
    z_base <- torch_randn(c(n_samples, n_dim), device = device)

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
  # returns:       array of shape (n_samples, n_dim, N_obs)
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
  out <- array(NA_real_, c(n_samples, n_dim, N_obs))

  chunk_i <- 0L

  for (start in seq(1L, N_obs, by = batch_size)) {
    chunk_i <- chunk_i + 1L
    end <- min(start + batch_size - 1L, N_obs)
    B <- end - start + 1L

    # Conditioning rows for this chunk: (B, N_feat); torch narrow is 1-based
    ctx_chunk <- input$narrow(1, start, B)

    # Latent samples for this chunk: (B, n_samples, n_dim)
    z_chunk <- torch_randn(c(B, n_samples, n_dim), device = device)
    z_flat <- z_chunk$reshape(c(B * n_samples, n_dim))

    # Duplicate each context row n_samples times: (B*n_samples, N_feat)
    context_flat <-
      ctx_chunk$unsqueeze(2)$expand(c(B, n_samples, N_feat))$reshape(
        c(B * n_samples, N_feat)
      )

    with_no_grad({
      theta_flat <- model$inverse(z_flat, context_flat) # (B*n_samples, n_dim)
    })

    # Unscale on-device in one fused broadcast.
    if (!is.null(col_means)) {
      theta_flat <- theta_flat * sds_t + means_t
    }

    # (B, n_samples, n_dim) -> (n_samples, n_dim, B) to match `out`.
    block <- theta_flat$reshape(c(B, n_samples, n_dim))$permute(c(2, 3, 1))

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
