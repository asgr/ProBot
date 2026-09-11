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
probotLossEval <- function(input,
                           output,
                           model,
                           mdn_components = NULL,
                           idx = NULL,
                           batch = 4096L,
                           loss_fn = NULL,
                           device = NULL,
                           verbose = FALSE) {

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

  # Eval mode, restored on the way out: scoring must not leave a model sitting
  # in eval() when the caller was still mid-training.
  was_training <- model$training
  model$eval()
  on.exit(if (was_training) model$train())

  n <- x$size(1)
  total <- 0
  n_batches <- 0L

  with_no_grad({
    for (s in seq(1L, n, by = batch)) {
      b <- min(batch, n - s + 1L)
      xb <- x$narrow(1L, s, b)
      yb <- y$narrow(1L, s, b)

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
    }
  })

  res <- list(
    loss = total / n,
    n = as.integer(n),
    batches = n_batches,
    model_type = model_type,
    loss_name = loss_name
  )

  if (verbose) {
    cat(sprintf("probotLossEval: %s %s loss = %.6f over %d rows (%d batches, %s)\n",
                model_type, loss_name, res$loss, res$n, n_batches, device$type))
  }

  return(res)
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
