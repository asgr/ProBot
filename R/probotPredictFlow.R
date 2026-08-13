probotSamplePostNF <- function(
    input,
    model,
    n_samples = 5000,
    col_means = NULL,
    col_sds = NULL,
    col_names = NULL,
    dim_theta = NULL,
    device = NULL
) {
  # Infer dimensionality from arguments in priority order
  if (!is.null(dim_theta)) {
    n_dim <- dim_theta
  } else if (!is.null(col_means)) {
    n_dim <- length(col_means)
  } else {
    stop("Either 'dim_theta' or 'col_means' must be provided to determine parameter dimensionality")
  }

  # Convert input to torch tensor if needed
  if (is.null(device)) {
    if (length(model$parameters) > 0) {
      device <- model$parameters[[1]]$device
    } else {
      device <- if (backends_mps_is_available()) torch_device("mps") else torch_device("cpu")
    }
  }

  if (!inherits(input, "torch_tensor")) {
    input <- torch_tensor(input, dtype = torch_float(), device = device)
  } else {
    input <- input$to(device = device)
  }

  model$eval()

  # Draw samples from the base distribution N(0, I)
  z_base <- torch_randn(c(n_samples, n_dim), device = device)

  with_no_grad({
    # Map base samples -> posterior parameters using inverse flow
    # input must be 2D (1, C); expand it to (n_samples, C)
    if (input$dim() == 1L) {
      input <- input$unsqueeze(1L)  # (C,) -> (1, C)
    }
    if (input$size(1) != n_samples) {
      context_expanded <- input$expand(c(n_samples, input$size(2)))
    } else {
      context_expanded <- input
    }

    theta_samples_torch <- model$inverse(z_base, context_expanded)
  })

  # Convert to R matrix
  samples <- as.matrix(theta_samples_torch$to(device = "cpu"))

  # Unscale if necessary (reuses existing scaling helper)
  if (!is.null(col_means)) {
    samples <- probotScaleBackward(samples, col_means, col_sds)
  }

  colnames(samples) <- col_names

  return(samples)
}
