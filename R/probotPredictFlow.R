probotSamplePostNF <- function(input,
                               model,
                               n_samples = 5000,
                               col_means = NULL,
                               col_sds = NULL,
                               col_names = NULL,
                               dim_theta = NULL,
                               device = NULL) {
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
  # SINGLE OBSERVATION MODE
  # ------------------------------------------------------------------

  if (input$dim() == 1L) {
    input <- input$unsqueeze(1)

  }

  if (input$dim() == 2L && input$size(1) == 1) {
    z_base <- torch_randn(c(n_samples, n_dim), device = device)

    context_expanded <-
      input$expand(c(n_samples, input$size(2)))

    with_no_grad({
      theta_samples_torch <-
        model$inverse(z_base, context_expanded)

    })

    samples <-
      as.matrix(theta_samples_torch$cpu())

    if (!is.null(col_means)) {
      samples <-
        probotScaleBackward(samples, col_means, col_sds)

    }

    colnames(samples) <- col_names

    return(samples)

  }

  # ------------------------------------------------------------------
  # MULTI-OBSERVATION MODE
  # input shape:
  #   (N_obs, N_features)
  #
  # returns:
  #   (N_obs, n_samples, n_dim)
  # ------------------------------------------------------------------

  if (input$dim() != 2L) {
    stop("'input' must be either a vector or a matrix")

  }

  N_obs <- input$size(1)
  N_feat <- input$size(2)

  # ------------------------------------------------------------------
  # Sample latent space
  # ------------------------------------------------------------------

  z_base <- torch_randn(c(N_obs, n_samples, n_dim), device = device)

  z_flat <- z_base$reshape(c(N_obs * n_samples, n_dim))

  # ------------------------------------------------------------------
  # Duplicate conditioning information
  # ------------------------------------------------------------------

  context_flat <- input$unsqueeze(2)$expand(c(N_obs, n_samples, N_feat))$reshape(c(N_obs * n_samples, N_feat))

  # ------------------------------------------------------------------
  # One large inverse pass
  # ------------------------------------------------------------------

  with_no_grad({
    theta_flat <- model$inverse(z_flat, context_flat)
  })

  theta_samples <- theta_flat$reshape(c(N_obs, n_samples, n_dim))$permute(c(2, 3, 1))

  samples <- as.array(theta_samples$cpu())

  # ------------------------------------------------------------------
  # Undo scaling
  # ------------------------------------------------------------------

  if (!is.null(col_means)) {
    for (j in seq_len(n_dim)) {
      samples[,j,] <-
        samples[,j,] *
        col_sds[j] +
        col_means[j]

    }

  }

  if (!is.null(col_names)) {
    dimnames(samples) <-
      list(
        Sample = seq_len(n_samples),
        Parameter = col_names,
        Observation = seq_len(N_obs)
      )

  }

  return(samples)
}
