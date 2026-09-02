probotPredictMDN <- function(input, model, mdn_components, device = NULL){

  if (is.null(device)) {
    if (length(model$parameters) > 0) {
      device <- model$parameters[[1]]$device
    } else {
      device <- if (backends_mps_is_available()) torch_device("mps") else torch_device("cpu")
    }
  }

  model$eval()

  with_no_grad({
    output <- model(
      torch_tensor(
        input,
        dtype = torch_float(),
        device = device
      )
    )
  })

  .probotUnpackMDN(output, mdn_components)
}

probotSamplePostMDN <- function(
    input,
    model,
    mdn_components,
    n_samples = 5000,
    col_means = NULL,
    col_sds = NULL,
    col_names = NULL,
    device = NULL,
    batch_size = NULL,
    verbose = FALSE
){
  # ------------------------------------------------------------------
  # Sampling mirrors probotSamplePostNF(): the caller supplies the
  # conditioning inputs (a vector or matrix), not a pre-computed
  # mdn_output. A vector or single-row matrix returns an
  # (n_samples, output_dim) matrix; a multi-row matrix returns the
  # (n_samples, output_dim, N_obs) array.
  # ------------------------------------------------------------------

  if (!is.null(col_means) && is.null(col_sds)) {
    stop("col_sds must be provided when col_means is provided.")
  }

  unscale <- !is.null(col_means)

  n_samples <- as.integer(n_samples)

  if (n_samples < 1L) {
    stop("'n_samples' must be a positive integer.")
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

  if (input$dim() == 1L) {
    input <- input$unsqueeze(1)
  }

  if (input$dim() != 2L) {
    stop("'input' must be either a vector or a matrix")
  }

  model$eval()

  N_obs <- input$size(1)

  # Auto-size the chunk so that batch_size * n_samples draw rows are held
  # in memory at once (~2e6 rows default). True memory use scales with
  # output_dim as well, since each draw is a full parameter vector; lower
  # batch_size for wide posteriors or tighter machines.
  if (is.null(batch_size)) {
    batch_size <- max(1L, floor(2e6 / n_samples))
  }

  batch_size <- as.integer(min(batch_size, N_obs))

  n_chunks <- ceiling(N_obs / batch_size)
  progress_every <- max(1L, floor(n_chunks / 20))

  out <- NULL
  output_dim <- NA_integer_

  chunk_i <- 0L

  for (start in seq(1L, N_obs, by = batch_size)) {
    chunk_i <- chunk_i + 1L
    end <- min(start + batch_size - 1L, N_obs)
    B <- end - start + 1L

    # Forward pass for this chunk only: rows are the conditioning inputs.
    pred <- probotPredictMDN(
      input = input$narrow(1, start, B),
      model = model,
      mdn_components = mdn_components,
      device = device
    )

    # Mixture parameters, brought to R once per chunk: (B, K, D) tensors.
    mu <- as.array(pred$mu$cpu())
    log10_sigma <- as.array(pred$log10_sigma$cpu())

    weights <- as.matrix(nnf_softmax(pred$logits, dim = 2)$cpu())

    # Soft clamp matches probotLossMDN()/probotMarginalPostMDN(): sigma is
    # capped at 1e5 rather than allowed to run away with an unbounded head.
    sigma <- 10^pmin(pmax(log10_sigma, -5), 5)

    D <- dim(mu)[3]

    if (is.na(output_dim)) {
      output_dim <- D

      if (unscale) {
        if (!length(col_means) %in% c(1L, D) || !length(col_sds) %in% c(1L, D)) {
          stop(
            "col_means/col_sds must have length 1 or match the model's ",
            "output dimension (", D, ")."
          )
        }
      }

      out <- array(NA_real_, c(n_samples, D, N_obs))
    } else if (D != output_dim) {
      stop("Model output dimension changed between chunks.")
    }

    # (B, D) slice per component; matrix() guard keeps shape when B == 1.
    mu_comp <- lapply(
      seq_len(mdn_components),
      function(k) matrix(mu[, k, ], nrow = B, ncol = D)
    )
    sigma_comp <- lapply(
      seq_len(mdn_components),
      function(k) matrix(sigma[, k, ], nrow = B, ncol = D)
    )

    # Component label for every (observation, draw) pair, flattened so that
    # row f of `theta` is observation ((f - 1) %/% n_samples) + 1.
    comp <- integer(B * n_samples)

    for (b in seq_len(B)) {
      rows <- seq((b - 1L) * n_samples + 1L, b * n_samples)
      comp[rows] <- sample.int(
        mdn_components,
        size = n_samples,
        replace = TRUE,
        prob = weights[b, ]
      )
    }

    theta <- matrix(NA_real_, B * n_samples, D)

    for (k in seq_len(mdn_components)) {

      idx <- which(comp == k)

      if (length(idx) == 0L)
        next

      b <- ((idx - 1L) %/% n_samples) + 1L

      z <- matrix(rnorm(length(idx) * D), nrow = length(idx), ncol = D)

      theta[idx, ] <-
        mu_comp[[k]][b, , drop = FALSE] +
        sigma_comp[[k]][b, , drop = FALSE] * z
    }

    if (unscale) {
      # probotScaleBackward() aligns stats by column; plain vector
      # arithmetic would recycle down the flattened matrix.
      theta <- probotScaleBackward(theta, col_means, col_sds)
    }

    # Flat (B*n_samples, D) -> (n_samples, D, B) for this chunk. Filling an
    # (n_samples, B, D) array reproduces the column-major row order of theta,
    # so the aperm afterwards is a pure relabeling of axes.
    out[, , start:end] <- aperm(
      array(theta, dim = c(n_samples, B, D)), c(1, 3, 2)
    )

    if (verbose && (chunk_i %% progress_every == 0L || chunk_i == n_chunks)) {
      cat(sprintf(
        "probotSamplePostMDN: chunk %d/%d (obs %d-%d)\n",
        chunk_i, n_chunks, start, end
      ))
    }
  }

  # ------------------------------------------------------------------
  # Single-observation mode: return a plain (n_samples, D) matrix
  # ------------------------------------------------------------------

  if (N_obs == 1L) {
    # Explicit matrix(): out[, , 1] would drop to a vector when output_dim == 1.
    samples <- matrix(out[, , 1], nrow = n_samples, ncol = output_dim)
    colnames(samples) <- col_names
    return(samples)
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

probotMarginalPostMDN = function(mdn_output,
                              col_means,
                              col_sds,
                              col_names = NULL){
  weights <- as.array(
    nnf_softmax(
      mdn_output$logits,
      dim = 2
    )
  )

  mu <- as.array(mdn_output$mu)

  sigma <- 10^pmin(pmax(as.array(mdn_output$log10_sigma), -5), 5)

  # ======================================================
  # Posterior mean
  # ======================================================

  N <- dim(mu)[1]
  K <- dim(mu)[2]
  D <- dim(mu)[3]

  post_mean_scaled <- matrix(0, N, D)

  for(k in 1:K){
    post_mean_scaled <- post_mean_scaled + weights[,k] * mu[,k,]
  }

  # ======================================================
  # Posterior variance
  # ======================================================

  second_moment <- matrix(0, N, D)

  for(k in 1:K){
    second_moment <- second_moment + weights[,k] * (sigma[,k,]^2 + mu[,k,]^2)
  }

  post_var_scaled <- second_moment - post_mean_scaled^2

  post_var_scaled[] <- pmax(post_var_scaled, 0)

  post_sd_scaled <- sqrt(post_var_scaled)

  # ======================================================
  # Convert back to physical units
  # ======================================================

  post_mean <- probotScaleBackward(post_mean_scaled, col_means, col_sds)
  post_sd <- probotScaleBackward(post_sd_scaled, 0, col_sds)

  colnames(post_sd) = col_names
  colnames(post_mean) = col_names

  return(list(post_mean=post_mean, post_sd=post_sd))
}
