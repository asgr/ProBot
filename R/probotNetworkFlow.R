# ============================================================================
# probotFlows.R
# Normalizing Flow extensions for ProBot (Conditional NVP / Coupling Flows)
# ============================================================================

# --- 1. Coupling Layer Definition ---

.probotCouplingLayer <- nn_module(
    initialize = function(dim_theta, dim_x, hidden_dim = 32, soft_clamp=3, device = NULL) {
      self$dim_theta <- dim_theta
      self$dim_x <- dim_x

      # Split parameter space in half for coupling
      self$d1 <- floor(dim_theta / 2)
      self$d2 <- dim_theta - self$d1
      if (self$d1 < 1 || self$d2 < 1) {
        stop("'dim_theta' must be >= 2 for .probotCouplingLayer() (got ", dim_theta, ")", call. = FALSE)
      }
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
      target_device <- .probotChooseDevice(device)

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
      theta_2 <- z_2 * torch_exp(-log_scale) + shift
      theta_1 <- z_1

      theta <- torch_cat(list(theta_1, theta_2), dim = 2)
      return(theta)
    }
  )

# --- 2. Flow Network Definition ---

probotMakeFlowCouple <- nn_module(
  "probotMakeFlowCouple",
    initialize = function(dim_theta, dim_x, n_layers = 4, hidden_dim = 32, soft_clamp = 3, device = NULL) {
      self$n_layers <- n_layers

      # Register each coupling layer directly on self so torch tracks parameters & device placement
      for (i in seq_len(n_layers)) {
        self[[paste0("coupling_layer_", i)]] <- .probotCouplingLayer(dim_theta, dim_x,
                                                  hidden_dim, soft_clamp=soft_clamp, device=device)
      }

      #Determine device: Use provided device, or auto-detect MPS / CPU fallback
      target_device <- .probotChooseDevice(device)

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

.probotMaskedLinear = nn_module(
  ".probotMaskedLinear",
    initialize = function(in_features, out_features, bias = TRUE) {
      self$linear <- nn_linear(in_features, out_features, bias = bias)
      # Register mask as a buffer so it saves with the model state but isn't trained
      self$register_buffer("mask", torch_ones(out_features, in_features))
    },

    set_mask = function(mask_matrix) {
      # Expects a matrix/tensor of shape (out_features, in_features)
      self$mask$copy_(mask_matrix)
    },

    forward = function(x) {
      # Apply the mask dynamically to the weights during execution
      masked_weight <- self$linear$weight * self$mask
      nnf_linear(x, masked_weight, self$linear$bias)
    }
  )

.probotFlowAutoRegBlock = nn_module(
  ".probotFlowAutoRegBlock",
  initialize = function(dim_theta, dim_x, n_layers = 2, hidden_dim = 500,
                        soft_clamp = 3, device = NULL) {
    self$dim_theta <- dim_theta
    self$dim_x <- dim_x
    self$hidden_dim <- hidden_dim
    self$n_layers <- n_layers
    self$soft_clamp <- soft_clamp

    if (n_layers < 1) {
      stop("n_layers must be at least 1")
    }

    self$layers <- nn_module_list()
    self$layers$append(.probotMaskedLinear(dim_theta + dim_x, hidden_dim))

    if (n_layers > 1) {
      for (i in 2:n_layers) {
        self$layers$append(.probotMaskedLinear(hidden_dim, hidden_dim))
      }
    }

    self$output_layer <- .probotMaskedLinear(hidden_dim, 2 * dim_theta)

    # 1. FIX: Context features (dim_x) get degree 0 so they are always visible.
    # Target features (dim_theta) get sequential positions 1 to dim_theta.
    input_degrees <- torch_cat(list(torch_arange(1, dim_theta), torch_zeros(dim_x)), dim = 1)

    # Hidden layers get values between 1 and dim_theta (not dim_theta - 1)
    # This allows Parameter 1 to map safely across the conditioning paths
    hidden_degrees_list <- list()
    for (i in 1:n_layers) {
      hidden_degrees_list[[i]] <- torch_remainder(torch_arange(0, hidden_dim - 1), dim_theta) + 1
    }

    # 2. Build hidden connection pathways
    # MADE masking convention (Germain et al.):
    #   - First hidden layer uses strict > so hidden node k only sees theta_{1..k-1} (not theta_k itself),
    #     while x (degree 0) remains visible to all hidden nodes since k > 0 for all k >= 1.
    #   - Subsequent hidden layers and the output layer use >= so the chain propagates correctly and
    #     output_j (degree j) only reaches theta_{1..j-1} end-to-end.
    for (i in 1:n_layers) {
      current_layer <- self$layers[[i]]

      if (i == 1) {
        mask <- hidden_degrees_list[[1]]$unsqueeze(2) > input_degrees$unsqueeze(1)
      } else {
        mask <- hidden_degrees_list[[i]]$unsqueeze(2) >= hidden_degrees_list[[i-1]]$unsqueeze(1)
      }
      current_layer$set_mask(mask$to(dtype = torch_float()))
    }

    # 3. Output layer: output_j (degree j) sees hidden nodes with degree <= j.
    # Combined with the strict > in the first hidden layer this guarantees that
    # the shift and log_scale for theta_j depend only on theta_{1..j-1} and x.
    output_degrees <- torch_repeat_interleave(torch_arange(1, dim_theta), repeats = 2)
    last_hidden_degrees <- hidden_degrees_list[[n_layers]]

    output_mask <- output_degrees$unsqueeze(2) >= last_hidden_degrees$unsqueeze(1)
    self$output_layer$set_mask(output_mask$to(dtype = torch_float()))

    target_device <- if (!is.null(device)) device else (if (cuda_is_available()) "cuda" else "cpu")
    self$to(device = target_device)
  },

  forward_mlp = function(combined) {
    x <- combined
    for (i in 1:self$n_layers) {
      x <- self$layers[[i]](x) %>% nnf_gelu()
    }
    return(self$output_layer(x))
  },

  forward = function(theta, x, soft_clamp = self$soft_clamp) {
    combined <- torch_cat(list(theta, x), dim = 2)
    out <- self$forward_mlp(combined)

    out_reshaped <- out$view(c(-1, self$dim_theta, 2))
    shift <- out_reshaped$narrow(3, 1, 1)$squeeze(3)
    log_scale <- out_reshaped$narrow(3, 2, 1)$squeeze(3)

    log_scale <- soft_clamp * torch_tanh(log_scale / soft_clamp)
    z <- (theta - shift) * torch_exp(-log_scale)
    log_det_jac <- (-log_scale)$sum(dim = 2, keepdim = TRUE)

    list(z = z, log_det_jac = log_det_jac)
  },

  inverse = function(z, x, soft_clamp = self$soft_clamp) {
    theta <- torch_zeros_like(z)

    for (i in 1:self$dim_theta) {
      combined <- torch_cat(list(theta, x), dim = 2)
      out <- self$forward_mlp(combined)

      out_reshaped <- out$view(c(-1, self$dim_theta, 2))
      shift_i <- out_reshaped$narrow(2, i, 1)$narrow(3, 1, 1)$squeeze(3)$squeeze(2)
      log_scale_i <- out_reshaped$narrow(2, i, 1)$narrow(3, 2, 1)$squeeze(3)$squeeze(2)
      log_scale_i = soft_clamp * torch_tanh(log_scale_i / soft_clamp)

      z_i <- z$narrow(2, i, 1)$squeeze(2)
      theta_i <- z_i * torch_exp(log_scale_i) + shift_i

      theta$narrow(2, i, 1)$copy_(theta_i$unsqueeze(2))
    }
    return(theta)
  }
)


.probotPermutationLayer <- nn_module(
  ".probotPermutationLayer",
  initialize = function(dim_theta) {
    self$dim_theta <- dim_theta
    # Create a vector that reverses the index order: e.g., c(9, 8, 7, ..., 1)
    # Register as a buffer so it migrates to GPUs automatically
    self$register_buffer("reverse_idx", torch_arange(dim_theta, 1, step = -1, dtype = torch_long()))
  },

  forward = function(theta, x) {
    # Reverse the columns of theta
    z <- theta$index_select(2, self$reverse_idx)
    # Reversing order is an orthogonal transformation; its Jacobian determinant is always 1 (log(1) = 0)
    log_det_jac <- torch_zeros(c(theta$size(1), 1), device = theta$device)

    list(z = z, log_det_jac = log_det_jac)
  },

  inverse = function(z, x) {
    # The inverse of a simple reversal is the exact same reversal
    return(z$index_select(2, self$reverse_idx))
  }
)

probotFlowAutoReg <- nn_module(
  "probotFlowAutoReg",
  initialize = function(dim_theta, dim_x, n_blocks = 5, n_layers_per_block = 2, hidden_dim = 500,
                        soft_clamp = 3, device = NULL) {
    self$dim_theta <- dim_theta
    self$dim_x <- dim_x
    self$n_blocks <- n_blocks
    self$soft_clamp <- soft_clamp

    # Track steps sequentially
    self$blocks <- nn_module_list()

    for (i in 1:n_blocks) {
      # 1. Add the core Masked Autoregressive Flow block
      self$blocks$append(
        .probotFlowAutoRegBlock(
          dim_theta = dim_theta,
          dim_x = dim_x,
          n_layers = n_layers_per_block,
          hidden_dim = hidden_dim,
          soft_clamp = soft_clamp,
          device = device
        )
      )

      # 2. Add an alternating permutation layer between blocks (but skip after the very last block)
      if (i < n_blocks) {
        self$blocks$append(.probotPermutationLayer(dim_theta = dim_theta))
      }
    }
  },

  forward = function(theta, x) {
    # Tracks overall transformation scaling across the chain
    total_log_det_jac <- torch_zeros(c(theta$size(1), 1), device = theta$device)
    current_z <- theta

    # Process forward through every registered step
    for (i in 1:length(self$blocks)) {
      step_res <- self$blocks[[i]](current_z, x)
      current_z <- step_res$z
      total_log_det_jac <- total_log_det_jac + step_res$log_det_jac
    }

    list(z = current_z, log_det_jac = total_log_det_jac)
  },

  inverse = function(z, x) {
    current_theta <- z
    # CRITICAL: For inverse sampling, we must iterate backwards through the chain
    for (i in length(self$blocks):1) {
      current_theta <- self$blocks[[i]]$inverse(current_theta, x)
    }

    return(current_theta)
  }
)
