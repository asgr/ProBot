# ============================================================================
# probotFlows.R
# Normalizing Flow extensions for ProBot (Conditional NVP / Coupling Flows)
# ============================================================================

# --- 1. Coupling Layer Definition ---

probotCouplingLayer <- function(dim_theta, dim_x, hidden_dim = 32, soft_clamp=3, device = NULL) {
  nn_module(
    initialize = function() {
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
      
      #Determine device: Use provided device, or auto-detect MPS / CPU fallback
      if (is.null(device)) {
        target_device <- if (backends_mps_is_available()) torch_device("mps") else torch_device("cpu")
      } else {
        target_device <- torch_device(device)
      }
      
      # 3. CRITICAL: Automatically move the entire module structure to the target hardware
      self$to(device = target_device)
      # d1 (first-half width) and d2 (second-half width) are used for narrow() slicing
    },
    
    forward = function(theta, x) {
      # theta: (Batch, D)
      # x:      (Batch, C)
      
      device <- theta$device
      
      # Split theta into two halves using narrow() — works cleanly on 2D (Batch, D) tensors
      theta_1 <- theta$narrow(2, 1, self$d1)
      theta_2 <- theta$narrow(2, self$d1 + 1, self$d2)
      
      # Concatenate context with the first half to predict transform for the second half
      combined <- torch_cat(list(theta_1, x), dim = 2)
      
      # Get shift and log_scale
      params <- self$shift_scale_net(combined) # (Batch, 2 * d2)
      
      # Split into two vectors of length d2
      split_params <- torch_chunk(params, 2, dim = 2)
      shift <- split_params[[1]]   
      log_scale <- soft_clamp*torch_tanh(split_params[[2]] / soft_clamp)  # tanh-clamp for numerical stability
      
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
      
      # Extract from swapped layout using narrow(): z_2 in [1, d2], z_1 in [d2+1, D]
      z_2 <- z$narrow(2, 1, self$d2)
      z_1 <- z$narrow(2, self$d2 + 1, self$d1)
      
      # Predict transformation using the 'other' half (z_1) and context
      combined <- torch_cat(list(z_1, x), dim = 2)
      
      params <- self$shift_scale_net(combined)
      split_params <- torch_chunk(params, 2, dim = 2)
      shift <- split_params[[1]]
      log_scale <- soft_clamp*torch_tanh(split_params[[2]]/soft_clamp)
      
      # Reverse affine transformation: theta_2 = z_2 / exp(s) + t
      theta_2 <- z_2 / torch_exp(log_scale) + shift
      theta_1 <- z_1
      
      theta <- torch_cat(list(theta_1, theta_2), dim = 2)
      return(theta)
    }
  )
}

# --- 2. Flow Network Definition ---

probotMakeFlow <-  function(dim_theta, dim_x, n_layers = 4, hidden_dim = 32, soft_clamp = 3, device = NULL) {
  nn_module(
    initialize = function() {
      self$n_layers <- n_layers
      
      # Register each coupling layer directly on self so torch tracks parameters & device placement
      for (i in seq_len(n_layers)) {
        self[[paste0("coupling_layer_", i)]] <- probotCouplingLayer(dim_theta, dim_x,
                                                  hidden_dim, soft_clamp=soft_clamp, device=device)()
      }
      
      #Determine device: Use provided device, or auto-detect MPS / CPU fallback
      if (is.null(device)) {
        target_device <- if (backends_mps_is_available()) torch_device("mps") else torch_device("cpu")
      } else {
        target_device <- torch_device(device)
      }
      
      # 3. CRITICAL: Automatically move the entire module structure to the target hardware
      self$to(device = target_device)
    },
    
    forward = function(theta, x) {
      # Forward pass: map Parameters -> Latent Space (for training loss)
      z <- theta
      log_det_jac <- torch_zeros(theta$size(1), device = theta$device)$unsqueeze(1)
      
      for (i in seq_len(self$n_layers)) {
        out <- self[[paste0("coupling_layer_", i)]]$forward(z, x)
        z <- out$z
        log_det_jac <- log_det_jac + out$log_det_jac
      }
      
      list(z = z, log_det_jac = log_det_jac)
    },
    
    inverse = function(z, x) {
      # Inverse pass: map Latent Space -> Parameters (for posterior sampling)
      theta <- z
      
      # Iterate layers in reverse order
      for (i in rev(seq_len(self$n_layers))) {
        theta <- self[[paste0("coupling_layer_", i)]]$inverse(theta, x)
      }
      
      return(theta)
    }
  )
}
