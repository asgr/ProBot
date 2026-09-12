# Score a trained model's loss on arbitrary data without touching its parameters.
#
# The single-epoch trainers bundle evaluation with gradient steps, and the
# probotLoss*() functions only accept tensors for one batch, so scoring a
# held-out split needed its own path. This is the read-only counterpart to
# probotDataLoader() + probotSingleEpoch*().
#
# Argument order follows probotDataLoader() -- context first, targets second --
# which is the opposite of the probotLoss*() functions (output_true first). The
# dim checks below exist because that inversion is the easiest way to misuse
# this: a swapped pair still type-checks as two matrices.
#
# The headline `loss` is a mean over rows, and NLL is unbounded above, so a
# single tail observation can move it by orders of magnitude while saying
# nothing about model quality. Because of that `loss` is reported alongside a
# per-row summary; see .probotLossEvalRows() for which losses decompose.
probotLossEval <- function(input,
                           output,
                           model,
                           mdn_components = NULL,
                           idx = NULL,
                           batch = 4096L,
                           loss_fn = NULL,
                           device = NULL,
                           verbose = FALSE,
                           per_row = FALSE) {

  if (!inherits(model, "nn_module")) {
    stop("'model' must be an nn_module (e.g. from probotMakeMDN, probotMakePoint ",
         "or probotMakeFlow).", call. = FALSE)
  }

  model_type <- .probotModelClassType(model)
  if (is.null(model_type)) {
    stop("Unrecognised model type; expected an MDN, Point or normalising flow ",
         "module.", call. = FALSE)
  }

  x <- .probotLossEvalAsMatrix(input, "input")
  y <- .probotLossEvalAsMatrix(output, "output")

  if (!is.null(idx)) {
    idx <- as.integer(idx)
    if (anyNA(idx) || any(idx < 1L)) {
      stop("'idx' must be positive row indices.", call. = FALSE)
    }
    if (any(idx > nrow(x)) || any(idx > nrow(y))) {
      stop("'idx' refers to rows beyond the data (max ", max(idx),
           "; input has ", nrow(x), " rows, output has ", nrow(y), ").",
           call. = FALSE)
    }
    x <- x[idx, , drop = FALSE]
    y <- y[idx, , drop = FALSE]
  }

  if (nrow(x) != nrow(y)) {
    stop("input and output must have the same number of rows (got ", nrow(x),
         " and ", nrow(y), ").", call. = FALSE)
  }
  if (nrow(x) == 0L) {
    stop("No rows to score (n = 0).", call. = FALSE)
  }

  # Only an MDN has a mixture count, but callers comparing model types pass the
  # same arguments to each, so a foreign mdn_components is ignored quietly -- as
  # .probotPostSampler() does in probotPIT/probotCRPS/probotTARP.
  if (model_type == "mdn" && is.null(mdn_components)) {
    mdn_components <- model$mdn_components
  }

  # All three model types describe p(output | input) and take the D targets, so
  # the expected target width is simply the stored output_dim. Hand-built or
  # pre-0.7.0 modules may not carry one, which is why this stays a guard rather
  # than a requirement.
  width_guard <- model$output_dim
  .probotLossEvalCheckDims(ncol(x), ncol(y), model, width_guard)

  if (model_type == "mdn" && is.null(mdn_components)) {
    stop("'mdn_components' is required for MDN models.", call. = FALSE)
  }

  default_loss <- switch(model_type,
    mdn   = probotLossMDN,
    point = nnf_mse_loss,
    flow  = probotLossNF
  )
  if (is.null(loss_fn)) loss_fn <- default_loss
  loss_name <- .probotLossEvalName(loss_fn, default_loss)

  if (is.null(device)) {
    device <- if (length(model$parameters) > 0) {
      model$parameters[[1]]$device
    } else {
      .probotChooseDevice(NULL)
    }
  } else {
    device <- torch_device(device)
  }

  batch <- as.integer(batch)
  if (length(batch) != 1L || is.na(batch) || batch < 1L) {
    stop("'batch' must be a single whole number >= 1", call. = FALSE)
  }

  x <- torch_tensor(x, dtype = torch_float(), device = device)
  y <- torch_tensor(y, dtype = torch_float(), device = device)

  # Per-row losses are only recoverable for the built-in default of each model
  # type; an arbitrary loss_fn may aggregate in a way that has no row-level
  # analogue (probotLossMSE averages the mixture mean, not a row). For a flow,
  # scoring per row needs its own forward pass, so this stays opt-in: with
  # per_row = FALSE the result is exactly the pre-existing five fields.
  rows_decomposable <- identical(loss_fn, default_loss)
  if (isTRUE(per_row) && !rows_decomposable) {
    warning("'per_row = TRUE' is only supported for a model type's default loss; ",
            "the supplied 'loss_fn' is not decomposable per row. Returning the ",
            "loss without a per-row summary.", call. = FALSE)
  }
  collect_rows <- isTRUE(per_row) && rows_decomposable

  # Eval mode, restored on the way out: scoring must not leave a model sitting
  # in eval() when the caller was still mid-training.
  was_training <- model$training
  model$eval()
  on.exit(if (was_training) model$train())

  n <- x$size(1)
  total <- 0
  n_batches <- 0L
  row_loss <- if (collect_rows) numeric(n) else NULL

  with_no_grad({
    for (s in seq(1L, n, by = batch)) {
      b <- min(batch, n - s + 1L)
      xb <- x$narrow(1L, s, b)
      yb <- y$narrow(1L, s, b)
      # Bound to NULL every iteration: only Point and MDN assign below, and the
      # row decomposition is passed this value unconditionally.
      pred <- NULL

      current_loss <- switch(model_type,
        # A flow's forward pass consumes the context itself, so both tensors go
        # to the loss function; MDN and Point score the network's raw output.
        flow = loss_fn(output_true = yb, output_pred = xb, model = model),
        {
          pred <- model(xb)
          if (model_type == "mdn") {
            loss_fn(output_true = yb, output_pred = pred,
                    mdn_components = mdn_components)
          } else {
            loss_fn(pred, yb)
          }
        }
      )

      # Row-weighted so the result is the mean per-row loss over all n rows,
      # matching what the trainers report for a full epoch. An unweighted mean
      # of batch losses would over-weight a short final batch.
      total <- total + current_loss$item() * b
      n_batches <- n_batches + 1L

      # Reuses the forward pass above for Point and MDN. The flow's z and
      # log|det J| are not returned by probotLossNF, so it scores a second pass.
      if (collect_rows) {
        row_loss[s:(s + b - 1L)] <- .probotLossEvalRows(model_type, xb, yb, model,
                                                        mdn_components, pred)
      }
    }
  })

  res <- c(
    list(
      loss = total / n,
      n = as.integer(n),
      batches = n_batches,
      model_type = model_type,
      loss_name = loss_name
    ),
    if (collect_rows) .probotLossEvalSummary(row_loss),
    if (collect_rows) list(row_loss = row_loss)
  )

  if (verbose) {
    cat(sprintf("probotLossEval: %s %s loss = %.6f over %d rows (%d batches, %s)\n",
                model_type, loss_name, res$loss, res$n, n_batches, device$type))
    if (!is.null(res$loss_median)) {
      cat(sprintf("  mean %.4f | median %.4f | p90 %.4f | max %.4f%s\n",
                  res$loss, res$loss_median, res$loss_p90, res$loss_max,
                  if (res$n_extreme > 0L)
                    sprintf("  <-- %d row(s) beyond 10 MADs of the median", res$n_extreme)
                  else ""))
    }
  }

  return(res)
}

# Per-row loss for a batch, using the same maths as each type's default loss so
# the values average to the reported scalar. `pred` is the already-computed
# network output for Point/MDN; the flow recomputes nothing because its forward
# pass happens inside the loss, so its z and log|det J| come from `model` here.
#
# The flow case is the reason this exists at all: NLL has no upper bound, and
# log p(theta|x) = -0.5*||z||^2 - D/2*log(2*pi) + log|det J| means one
# out-of-support context can push ||z|| into the thousands and contribute
# ~0.5*||z||^2 / n to a batch mean. Splitting the two terms also separates a
# genuine density failure (huge ||z||^2) from a Jacobian failure (huge |logdet|),
# which the scalar loss cannot.
.probotLossEvalRows <- function(model_type, xb, yb, model, mdn_components, pred) {
  if (model_type == "flow") {
    out <- model$forward(yb, xb)
    z2 <- (out$z^2)$sum(dim = 2)
    d <- out$z$size(2)
    # log|det J| is returned as (batch, 1); the base term is a plain vector.
    # loss = -(log_p_base + logdet), so the base normal contributes *positively*
    # here: -log_p_base = +0.5*(||z||^2 + D*log(2*pi)).
    ldj <- out$log_det_jac$squeeze(2)
    as.numeric((z2 + d * log(2 * pi)) * 0.5 - ldj)
  } else if (model_type == "mdn") {
    p <- .probotUnpackMDN(pred, mdn_components)
    # The clamped value is used for both sigma and the log-normalisation, exactly
    # as probotLossMDN does; using the raw log10_sigma there would silently
    # disagree with the reported scalar whenever the clamp binds.
    log10_sigma <- torch_clamp(p$log10_sigma, min = -5, max = 5)
    sigma <- 10^log10_sigma
    y_true <- yb$unsqueeze(2)$expand(c(yb$size(1), mdn_components, yb$size(2)))
    zz <- (y_true - p$mu) / sigma
    log_prob <- (-0.5 * (zz^2 + 2 * log10_sigma * log(10) + log(2 * pi)))$sum(dim = 3)
    log_pi <- nnf_log_softmax(p$logits, dim = 2)
    as.numeric(-torch_logsumexp(log_pi + log_prob, dim = 2))
  } else {
    # Point: MSE per row, averaged over the D target columns to match
    # nnf_mse_loss's element-wise mean.
    as.numeric(((pred - yb)^2)$mean(dim = 2))
  }
}

# Robust summary of the per-row losses. `extreme` is flagged by a MAD-scaled
# deviation rather than an absolute cutoff so the test is unit-free and works
# for every model type (Point losses are tiny and positive; flow NLLs are
# negative in the well-trained regime).
.probotLossEvalSummary <- function(row_loss) {
  med <- stats::median(row_loss)
  # 1.4826 rescales the MAD of a normal sample to its sd.
  mad <- stats::mad(row_loss, center = med, constant = 1.4826)
  spread <- if (is.finite(mad) && mad > 0) mad else NA_real_
  thresh <- if (is.finite(spread)) med + 10 * spread else Inf
  list(
    loss_median = med,
    loss_p90 = unname(stats::quantile(row_loss, 0.90)),
    loss_max = max(row_loss),
    loss_mad = spread,
    extreme_threshold = thresh,
    n_extreme = sum(row_loss > thresh)
  )
}

# Accept a matrix, data frame or tensor and hand back a plain numeric matrix.
.probotLossEvalAsMatrix <- function(m, name) {
  if (inherits(m, "torch_tensor")) {
    m <- as.matrix(m$to(device = "cpu"))
  } else {
    m <- as.matrix(m)
  }
  storage.mode(m) <- "double"
  if (anyNA(m)) {
    stop("'", name, "' contains ", sum(is.na(m)), " missing values; impute or ",
         "subset them out before scoring.", call. = FALSE)
  }
  m
}

.probotLossEvalCheckDims <- function(n_x, n_y, model, width_guard) {
  # A stored width is only a guard: constructors predating 0.7.0 and hand-built
  # modules may not carry one.
  mid <- model$input_dim
  if (is.numeric(mid) && length(mid) == 1L && n_x != mid) {
    stop("input has ", n_x, " columns but the model expects input_dim = ", mid,
         ". A bare vector is read as one column of values, so pass a matrix ",
         "(use matrix(x, nrow = 1) for a single observation). If 'input' and ",
         "'output' were passed in the wrong order, swap them.",
         call. = FALSE)
  }
  if (is.numeric(width_guard) && length(width_guard) == 1L && n_y != width_guard) {
    stop("output has ", n_y, " columns but the model expects output_dim = ",
         width_guard, ". If 'input' and 'output' were passed in the wrong order, ",
         "swap them.", call. = FALSE)
  }
  invisible(TRUE)
}

# Name the loss for the returned metadata, so a history table can say which
# metric each row holds. torch's nnf_* functions arrive as freshly generated
# closures that cannot be identified by name, so the fallback compares against
# the type's default.
.probotLossEvalName <- function(loss_fn, default_loss) {
  nm <- .probotLossEvalKnownName(loss_fn)
  if (is.null(nm) && identical(loss_fn, default_loss)) {
    nm <- .probotLossEvalKnownName(default_loss)
  }
  if (is.null(nm)) "<custom>" else nm
}

.probotLossEvalKnownName <- function(loss_fn) {
  known <- list(probotLossMDN = probotLossMDN,
                probotLossNF = probotLossNF,
                probotLossMSE = probotLossMSE,
                probotLossMAE = probotLossMAE,
                probotLossMAPE = probotLossMAPE,
                probotLossHuber = probotLossHuber,
                nnf_mse_loss = nnf_mse_loss)
  hits <- Filter(function(k) identical(known[[k]], loss_fn), names(known))
  if (length(hits)) hits[1L] else NULL
}
