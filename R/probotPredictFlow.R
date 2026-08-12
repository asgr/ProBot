probotSamplePostNF <- function(
    context_x, 
    model, 
    n_samples = 5000, 
    col_means = NULL, 
    col_sds = NULL, 
    col_names = NULL,
    dim_theta = NULL
) {
  device <- context_x$device
  
  model$eval()
  
  # Infer dimensionality from arguments in priority order
  if (!is.null(dim_theta)) {
    n_dim <- dim_theta
  } else if (!is.null(col_means)) {
    n_dim <- length(col_means)
  } else {
    stop("Either 'dim_theta' or 'col_means' must be provided to determine parameter dimensionality")
  }
  
  # Draw samples from the base distribution N(0, I)
  z_base <- torch_randn(c(n_samples, n_dim), device = device)
  
  with_no_grad({
    # Map base samples -> posterior parameters using inverse flow
    # context_x must be 2D (1, C); expand it to (n_samples, C)
    if (context_x$dim() == 1L) {
      context_x <- context_x$unsqueeze(1L)  # (C,) -> (1, C)
    }
    if (context_x$size(1) != n_samples) {
      context_expanded <- context_x$expand(c(n_samples, context_x$size(2)))
    } else {
      context_expanded <- context_x
    }
    
    theta_samples_torch <- model$inverse(z_base, context_expanded)
  })
  
  # Convert to R matrix
  samples <- as.matrix(theta_samples_torch$to(device = "cpu"))
  
  # Unscale if necessary (reuses your existing scaling helper)
  if (!is.null(col_means)) {
    samples <- probotScaleBackward(samples, col_means, col_sds)
  }
  
  colnames(samples) <- col_names
  
  return(samples)
}
