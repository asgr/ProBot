# Width of the widest activation a forward/inverse sweep materialises per row.
# Every conditioner in every flow style is an MLP whose internal layers are
# `hidden_dim` wide, and the inverse pass touches them one layer at a time, so
# peak device memory tracks (rows in the batch) * (hidden_dim) floats -- not
# * output_dim. Estimating it from the parameters rather than metadata keeps it
# correct for a loaded checkpoint and for the probotFlowLoc wrapper, whose
# widest layer is the head, not the flow.
.probotFlowActWidth <- function(model) {
  width <- 32L
  for (p in model$parameters) {
    if (p$dim() == 2L) width <- max(width, as.integer(p$shape[1]))
  }
  width
}

# Rows the inverse pass may take at once. Two independent ceilings: the
# (rows x output_dim) draw matrix that comes back to R, and the
# (rows x hidden_dim) conditioner activations that stay on the device. The
# second is what bites on wide flows: at hidden_dim = 512 and the old 2e6-row
# default it asks for ~4 GB per layer, which does not merely slow sampling
# down, it fails outright. Measured on the 9-layer/hidden-512 prospect NSF.
# CPU: 100k rows = 4.7 GB at 10 us/row, 500k = 15.5 GB, 2e6 = fatal. MPS,
# which holds activations on-device so RSS looks flat: 50k-200k rows at
# 2.4-3.9 us/row, degrading to 21 us/row at 500k and 103 us/row at 1M. Each
# constant sits just inside its backend's measured sweet spot -- they roughly
# cancel, so the point of the budget is not crashing, not speed.
# Narrow flows still hit the original 2e6 draw-row ceiling, so their behaviour
# and their RNG stream are unchanged: 6.4e7 / 32 is exactly the old 2e6 draw
# budget, so a default-width (hidden_dim = 32) flow hits the two ceilings at
# the same point. The activation term only binds for wider conditioners --
# 125k rows at hidden_dim = 512 on CPU, 200k on MPS.
.probotFlowChunkRows <- function(act_width, device) {
  draw_rows <- 2e6
  # device may arrive as a plain string when the caller supplied one, so test
  # the rendered form rather than $type.
  act_floats <- if (grepl("mps", as.character(device))) 1.024e8 else 6.4e7
  act_rows <- floor(act_floats / max(1L, act_width))
  max(1L, min(draw_rows, act_rows))
}

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
  #
  # Same batched inverse sweep as the multi-observation path; it is a separate
  # branch only so the result can stay a plain (n_samples, output_dim) matrix.
  # Here the chunk axis is the sample axis, because n_samples on its own can
  # exceed the row budget (the assess functions default to 1e4).
  # ------------------------------------------------------------------

  act_width <- .probotFlowActWidth(model)
  row_budget <- .probotFlowChunkRows(act_width, device)

  if (input$dim() == 2L && input$size(1) == 1) {
    samples <- matrix(NA_real_, nrow = n_samples, ncol = output_dim)
    per_chunk <- min(n_samples, row_budget)

    with_no_grad({
      for (start in seq(1L, n_samples, by = per_chunk)) {
        end <- min(start + per_chunk - 1L, n_samples)
        m <- end - start + 1L

        z_base <- torch_randn(c(m, output_dim), device = device)
        theta_t <- model$inverse(
          z_base,
          input$expand(c(m, input$size(2)))
        )

        if (!is.null(col_means)) {
          theta_t <- theta_t * sds_t + means_t
        }

        samples[start:end, ] <- as.matrix(theta_t$cpu())
      }
    })

    colnames(samples) <- col_names

    return(samples)
  }

  # ------------------------------------------------------------------
  # MULTI-OBSERVATION MODE
  # input shape:   (N_obs, N_features)
  # returns:       array of shape (n_samples, output_dim, N_obs)
  #
  # Observations are processed in chunks so the (batch_size * n_samples) rows
  # handed to the inverse pass stay within row_budget. Peak memory follows
  # that number, not n_test * n_samples, and only the accumulated result is
  # ever held in R.
  # ------------------------------------------------------------------

  if (input$dim() != 2L) {
    stop("'input' must be either a vector or a matrix")
  }

  N_obs <- input$size(1)
  N_feat <- input$size(2)

  # batch_size is documented in observations. The memory budget is not
  # negotiable -- exceeding it does not slow sampling down, it exhausts RAM --
  # so an explicit batch_size above the cap is clamped. Because draws come from
  # one torch_randn() per chunk, a different chunk size means a different RNG
  # stream, so this is announced rather than done silently. Recipes whose
  # batch_size already fits are untouched and stay bit-reproducible.
  cap <- max(1L, floor(row_budget / n_samples))
  if (is.null(batch_size)) {
    batch_size <- cap
  } else if (batch_size > cap) {
    warning("batch_size = ", batch_size, " with n_samples = ", n_samples,
            " would feed ", format(batch_size * n_samples, scientific = TRUE),
            " rows to the inverse pass; for this model (widest internal layer ",
            act_width, ") that is not memory-safe, so it is reduced to ", cap,
            ". Posterior draws depend on the chunk size, so this changes the ",
            "random stream.", call. = FALSE)
    batch_size <- cap
  }
  batch_size <- as.integer(min(batch_size, N_obs))

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
