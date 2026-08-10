# ============================================================================
# probotFlows.R
# Normalizing Flow extensions for ProBot (Conditional NVP / Coupling Flows)
# ============================================================================

# --- 1. Coupling Layer Definition ---

probotCouplingLayer <- nn_module(
  initialize = function(dim_theta, dim_x, hidden_dim = 32) {
    self$dim_theta <- dim_theta
    self$dim_x <- dim_x
    
    # Split parameter space in half for coupling
    self$d1 <- floor(dim_theta / 2)
    self$d2 <- dim_theta - self$d1
    
    # Input to the shift/scale network is (first_half_of_theta + context_x)
    input_dim <- self$d1 + dim_x
    
    # Network to predict affine transformation parameters for theta_2
    self$shift_scale_net <- nn_sequential(
      nn_linear(input_dim, hidden_dim),
      nn_gelu(),
      nn_linear(hidden_dim, hidden_dim),
      nn_gelu(),
      nn_linear(hidden_dim, 2 * self$d2) 
    )
    
    # Initialize indices for splitting/joining
    idx1 <- torch_arange(0, self$d1 - 1)$unsqueeze(0)$to(torch_long())
    idx2 <- torch_arange(self$d1, self$dim_theta - 1)$unsqueeze(0)$to(torch_long())
    
    # Register buffers (non-learnable parameters that move to device)
    self$register_buffer("idx1", idx1)
    self$register_buffer("idx2", idx2)
  },
  
  forward = function(theta, x) {
    # theta: (Batch, D)
    # x:     (Batch, C)
    
    device <- theta$device
    
    # Split theta into two halves
    theta_1 <- theta$gather(2, self$idx1$to(device = device))
    theta_2 <- theta$gather(2, self$idx2$to(device = device))
    
    # Concatenate context with the first half to predict transform for the second half
    combined <- torch_cat(list(theta_1, x), dim = 2)
    
    # Get shift and log_scale
    params <- self$shift_scale_net(combined) # (Batch, 2 * d2)
    
    # Unsplit into two vectors of length d2
    split_params <- torch_unsplit(params, list(self$d2, self$d2), dim = 2)
    shift <- split_params[[1]]   
    log_scale <- split_params[[2]] 
    
    # Apply affine transformation: z_2 = (theta_2 - t) * exp(s)
    z_2 <- (theta_2 - shift) * torch_exp(log_scale)
    z_1 <- theta_1
    
    # Swap halves to ensure full mixing across subsequent layers
    z <- torch_cat(list(z_2, z_1), dim = 2)
    
    # Log determinant of the Jacobian: sum(log_scale) over transformed dimensions
    log_det_jac <- log_scale$sum(dim = 2)$unsqueeze(1)
    
    list(z = z, log_det_jac = log_det_jac)
  },
  
  inverse = function(z, x) {
    # Inverse pass for sampling: map Base Distribution -> Parameters
    
    device <- z$device
    
    # Swap back to original coupling order
    z_2 <- z$gather(2, self$idx1$to(device = device))
    z_1 <- z$gather(2, self$idx2$to(device = device))
    
    # Predict transformation using the 'other' half (now z_1) and context
    combined <- torch_cat(list(z_1, x), dim = 2)
    
    params <- self$shift_scale_net(combined)
    split_params <- torch_unsplit(params, list(self$d2, self$d2), dim = 2)
    shift <- split_params[[1]]
    log_scale <- split_params[[2]]
    
    # Reverse affine transformation: theta_2 = z_2 / exp(s) + t
    theta_2 <- z_2 / torch_exp(log_scale) + shift
    theta_1 <- z_1
    
    theta <- torch_cat(list(theta_1, theta_2), dim = 2)
    return(theta)
  }
)

# --- 2. Flow Network Definition ---

probotNetworkFlow <- nn_module(
  initialize = function(dim_theta, dim_x, n_layers = 4, hidden_dim = 32) {
    self$n_layers <- n_layers
    
    # Stack multiple coupling layers
    self$coupling_layers <- list()
    for(i in seq_len(n_layers)) {
      self$coupling_layers[[i]] <- probotCouplingLayer(dim_theta, dim_x, hidden_dim)
    }
  },
  
  forward = function(theta, x) {
    # Forward pass: map Parameters -> Latent Space (for training loss)
    z <- theta
    log_det_jac <- torch_zeros(theta$size(1), device = theta$device)$unsqueeze(1)
    
    for(i in seq_len(self$n_layers)) {
      out <- self$coupling_layers[[i]]$forward(z, x)
      z <- out$z
      log_det_jac <- log_det_jac + out$log_det_jac
    }
    
    list(z = z, log_det_jac = log_det_jac)
  },
  
  inverse = function(z, x) {
    # Inverse pass: map Latent Space -> Parameters (for posterior sampling)
    theta <- z
    
    # Iterate layers in reverse order
    for(i in seq_len(self$n_layers):1) {
      theta <- self$coupling_layers[[i]]$inverse(theta, x)
    }
    
    return(theta)
  }
)

# --- 3. Loss Function for Normalizing Flows ---

probotLossNF <- function(true_theta, context_x, model, device = NULL) {
  # true_theta: (Batch, D) - The 'true' parameters from your simulation
  # context_x:  (Batch, C) - The observed data / conditioning variables
  
  if(is.null(device)) {
    device <- if(length(model$parameters) > 0) model$parameters[[1]]$device else torch_device("cpu")
  }
  
  true_theta <- true_theta$to(device = device)
  context_x  <- context_x$to(device = device)
  
  out <- model$forward(true_theta, context_x)
  z <- out$z
  log_det_jac <- out$log_det_jac
  
  # Base distribution: Standard Normal N(0, I)
  # log p_base(z) = -0.5 * sum(z^2 + log(2*pi)) across dimensions
  log_p_z <- -0.5 * (z^2 + log(2 * pi))$sum(dim = 2)$unsqueeze(1)
  
  # Total log likelihood: log p(true_theta | x) = log p_base(z) + log|det(J)|
  log_likelihood <- log_p_z + log_det_jac
  
  # Return Negative Log Likelihood (to be minimized by optimizer)
  -log_likelihood$mean()
}

# --- 4. Posterior Sampling Function for Normalizing Flows ---

probotSamplePostNF <- function(
    context_x, 
    model, 
    n_samples = 5000, 
    col_means = NULL, 
    col_sds = NULL, 
    col_names = NULL
) {
  if (is.null(context_x$device)) {
    device <- torch_device("cpu")
    context_x <- context_x$to(device = device)
  } else {
    device <- context_x$device
  }
  
  model$eval()
  
  n_dim <- length(col_means)
  
  # Draw samples from the base distribution N(0, I)
  z_base <- torch_randn(c(n_samples, n_dim), device = device)
  
  with_no_grad({
    # Map base samples -> posterior parameters using inverse flow
    # Context must be expanded to match batch size if needed
    # Assuming context_x is (1, C), repeat it to (n_samples, C)
    if(context_x$size(1) == 1 && n_samples > 1) {
      context_expanded <- context_x$expand(c(n_samples, context_x$size(2)))
    } else {
      context_expanded <- context_x
    }
    
    theta_samples_torch <- model$inverse(z_base, context_expanded)
  })
  
  # Convert to R matrix
  samples <- as.matrix(theta_samples_torch$to(device = "cpu"))
  
  # Unscale if necessary (reuses your existing scaling helper)
  if(!is.null(col_means)) {
    samples <- probotScaleBackward(samples, col_means, col_sds)
  }
  
  colnames(samples) <- col_names
  
  return(samples)
}
