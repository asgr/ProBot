# Differentiable point estimate of theta given a batch of context rows.
#
# "centre" inverts the base distribution's mode (z = 0). Cheap: one inverse
# sweep. For a symmetric conditional posterior the centre coincides with the
# mean, but for a skewed one it sits near the mode and is biased for
# squared-error loss.
#
# "mean" averages n_point_samples inverse draws, giving an unbiased estimate of
# E[theta | x] -- the Bayes-optimal point under squared error, and the closest
# flow analogue to an MDN's analytic mixture mean. Costs n_point_samples times
# as many inverse sweeps, and carries Monte-Carlo noise in both the estimate and
# its gradient.
#
# "loc" returns the head of a residual location-head flow (probotFlowLoc):
# mu(x) directly, with no inverse sweep at all and hence no MC noise and no
# Jacobian-weighting bias. Only available for that architecture, and there is
# nothing to estimate for the plain flows.
#
# Draws for all rows are stacked into a single (n_obs * m, output_dim) inverse
# call rather than looping, which is what makes the cost tolerable for the
# coupling styles. The reshape back to (n_obs, m, output_dim) must match the
# row order used to tile the context below; test that pairing before changing
# either.
.probotFlowPointEstimate <- function(model,
                                     context,
                                     output_dim,
                                     point = "centre",
                                     n_point_samples = 32) {
  point <- match.arg(point, choices = c("centre", "mean", "loc"))

  if (point == "loc") {
    if (!inherits(model, "probotFlowLoc")) {
      stop("'point = \"loc\"' requires a flow built with loc_head = TRUE; ",
           "this model has no location head.", call. = FALSE)
    }
    return(model$location(context))
  }

  if (point == "centre") {
    z <- torch_zeros(c(context$size(1), output_dim),
                     dtype = context$dtype, device = context$device)
    return(model$inverse(z, context))
  }

  n_obs <- context$size(1)
  n_ctx <- context$size(2)
  m <- as.integer(n_point_samples)

  z <- torch_randn(c(n_obs * m, output_dim),
                   dtype = context$dtype, device = context$device)

  # Each context row is repeated m times consecutively: rows
  # (b * m + 1) .. ((b + 1) * m) all belong to observation b, which is exactly
  # how z$reshape(c(n_obs, m, output_dim)) groups them.
  ctx <- context$unsqueeze(2)$expand(c(n_obs, m, n_ctx))$
    reshape(c(n_obs * m, n_ctx))

  theta <- model$inverse(z, ctx)

  theta$reshape(c(n_obs, m, output_dim))$mean(dim = 2)
}

probotSingleEpochFlow <- function(model,
                                  dataloader,
                                  optimizer,
                                  loss_fn = probotLossNF,
                                  lambda = 0,
                                  point = "centre",
                                  n_point_samples = 32) {
  if (!is.numeric(lambda) || length(lambda) != 1L || is.na(lambda) ||
      lambda < 0 || lambda > 1) {
    stop("'lambda' must be a single number in [0, 1]", call. = FALSE)
  }

  point <- match.arg(point, choices = c("centre", "mean", "loc"))

  if (lambda > 0) {
    if (!is.numeric(n_point_samples) || length(n_point_samples) != 1L ||
        is.na(n_point_samples) || n_point_samples < 1 ||
        n_point_samples != as.integer(n_point_samples)) {
      stop("'n_point_samples' must be a single whole number >= 1", call. = FALSE)
    }
  }

  model$train()
  running_loss <- 0
  running_mae <- 0
  running_rmse <- 0
  n_elems <- 0

  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()

    current_loss <- loss_fn(
      output_true = batch[[2]], # Target parameters theta (dataloader output)
      output_pred = batch[[1]], # Context inputs x (dataloader input)
      model = model
    )

    # A flow's forward pass produces latent z, not a point estimate, so the MSE
    # term has to come from the inverse map. See ?probotTrainFlow for what the
    # two estimators do and do not target.
    theta_hat <- NULL
    if (lambda > 0) {
      theta_hat <- .probotFlowPointEstimate(
        model,
        context = batch[[1]],
        output_dim = batch[[2]]$size(2),
        point = point,
        n_point_samples = n_point_samples
      )
      mse_loss <- ((batch[[2]] - theta_hat)^2)$mean()

      # Blend on tensors BEFORE backward() so both terms contribute gradients.
      current_loss <- current_loss * (1 - lambda) + mse_loss * lambda
    }

    current_loss$backward()

    # FIX: Safe, fast, in-place gradient clipping between -1 and 1
    # This correctly preserves memory references for the optimizer step
    nn_utils_clip_grad_value_(model$parameters, clip_value = 1.0)

    optimizer$step()

    running_loss <- running_loss + current_loss$item() * batch[[1]]$size(1)

    if (!is.null(theta_hat)) {
      diff <- torch_abs(batch[[2]] - theta_hat)
      running_mae <- running_mae + diff$sum()$item()
      running_rmse <- running_rmse + ((batch[[2]] - theta_hat)^2)$sum()$item()
      # MAE/RMSE are element-wise: count target elements, not rows, so the
      # reported values stay correct for multivariate targets.
      n_elems <- n_elems + batch[[2]]$numel()
    }
  })

  metrics <- list(loss = running_loss / length(dataloader$dataset))

  # Only reported when the MSE term is active, so a lambda = 0 run keeps the
  # historical history data frame schema (epoch, loss).
  if (lambda > 0) {
    metrics$mae <- running_mae / n_elems
    metrics$rmse <- sqrt(running_rmse / n_elems)
  }

  metrics
}

probotTrainFlow <- function(model,
                                  dataloader,
                                  optimizer,
                                  epochs = 100,
                                  loss_fn = probotLossNF,
                                  lambda = 0,
                                  point = "centre",
                                  n_point_samples = 32,
                                  checkpoint_dir = NULL,
                                  checkpoint_every = 10,
                                  history = NULL,
                                  verbose = TRUE,
                                  early_stop = TRUE,
                                  stop_window = 20,
                                  stop_delta = 1e-2) {
  # Resolve here too so a typo fails immediately rather than inside the loop.
  point <- match.arg(point, choices = c("centre", "mean", "loc"))

  .probotTrainLoop(
    train_fn = probotSingleEpochFlow,
    model = model,
    dataloader = dataloader,
    optimizer = optimizer,
    epochs = epochs,
    checkpoint_dir = checkpoint_dir,
    checkpoint_every = checkpoint_every,
    history = history,
    verbose = verbose,
    early_stop = early_stop,
    stop_window = stop_window,
    stop_delta = stop_delta,
    checkpoint_prefix = "flow",
    loss_fn = loss_fn,
    lambda = lambda,
    point = point,
    n_point_samples = n_point_samples
  )
}
