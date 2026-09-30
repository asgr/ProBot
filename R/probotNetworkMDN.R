probotMakeMDN <- function(input_dim, output_dim, mdn_components,
                          hidden_dims = c(128, 256, 256), activation = nnf_relu, 
                          dropout = 0, device = NULL) {
  
  nn_module(
    "mlp",
    initialize = function() {
      self$activation_fn <- activation
      self$dropout_rate  <- dropout
      # Architecture as constructed, so probotSave() can record it without being
      # told. Scalars only -- nn_linear owns the tensors -- which keeps these out
      # of state_dict(). See ?probotSave.
      self$input_dim      <- as.integer(input_dim)
      self$output_dim     <- as.integer(output_dim)
      self$mdn_components <- as.integer(mdn_components)
      self$hidden_dims    <- as.integer(hidden_dims)

      dims <- c(input_dim, hidden_dims, mdn_components * (2 * output_dim + 1))
      
      layer_list <- list()
      for (i in seq_len(length(dims) - 1)) {
        layer_list[[i]] <- nn_linear(dims[i], dims[i + 1])
      }
      
      self$layers <- nn_module_list(layer_list)
      
      #Determine device: Use provided device, or auto-detect MPS / CPU fallback
      target_device <- .probotChooseDevice(device)
      
      # 3. CRITICAL: Automatically move the entire module structure to the target hardware
      self$to(device = target_device)
    },
    
    forward = function(x) {
      n_layers <- length(self$layers)
      for (i in seq_len(n_layers - 1)) {
        x <- self$layers[[i]](x)
        x <- self$activation_fn(x)
        if (self$dropout_rate > 0) {
          x <- nnf_dropout(x, p = self$dropout_rate, training = self$training)
        }
      }
      x <- self$layers[[n_layers]](x)
      x
    }
  )
}

# Every (K, D) split consistent with an MDN head of `rows` units, as a data frame.
# The head emits K * (2D + 1) values, so the factorisation is usually ambiguous:
# rows = 15 admits (K=1,D=7), (K=3,D=2) and (K=5,D=1).
.probotMDNSplits <- function(rows) {
  rows <- as.integer(rows)
  k <- which(rows %% seq_len(rows) == 0L)
  d <- (rows / k - 1) / 2
  ok <- d >= 1 & d == as.integer(d)
  data.frame(K = k[ok], D = as.integer(d[ok]))
}

# Read the mixture count off a model rather than taking it as an argument.
#
# probotMakeMDN() stores the count it was built with, so for any in-tree model
# (including one rebuilt by probotLoad()) this is a lookup. The fallbacks exist
# so hand-built modules stay usable. `output_features` is the head width, and
# K * (2D + 1) == output_features is the only constraint linking the two
# unknowns, so the candidate splits are enumerated and the stored fields select
# one of them. K pins D and D pins K, so each selection is unique.
#
# A stored count that contradicts the model's own head width is reported rather
# than unpacked with whichever number came first: it means the object is
# internally inconsistent, and silently picking one would mis-split the head.
.probotMDNK <- function(model, output_features) {
  # An atomic `model` is almost always a leftover positional mixture count now
  # that the argument has been removed, so name that possibility.
  if (is.atomic(model) || is.null(model)) {
    stop("'model' must be the MDN module, not a ",
         if (is.null(model)) "NULL" else class(model)[1L],
         ". The mdn_components argument has been removed: the mixture count is ",
         "read from the model.", call. = FALSE)
  }

  output_features <- as.integer(output_features)
  sp <- .probotMDNSplits(output_features)
  splits <- if (nrow(sp)) {
    paste0("K=", sp$K, ", D=", sp$D, collapse = "; ")
  } else {
    "none -- no integer (K, D) pair fits"
  }

  k <- model$mdn_components
  d <- model$output_dim
  known_k <- !is.null(k) && length(k) == 1L && !is.na(k) && k >= 1
  known_d <- !is.null(d) && length(d) == 1L && !is.na(d) && d >= 1

  if (known_k && known_d) {
    if (as.integer(k) * (2L * as.integer(d) + 1L) == output_features) {
      return(as.integer(k))
    }
    stop("Model's stored mdn_components = ", k, " with output_dim = ", d,
         " implies a ", as.integer(k) * (2L * as.integer(d) + 1L),
         "-unit head, but its output has ", output_features, " units, so the ",
         "object is internally inconsistent.", call. = FALSE)
  }

  if (known_k) {
    hit <- which(sp$K == as.integer(k))
    if (length(hit) == 1L) return(sp$K[hit])
    stop("Model's stored mdn_components = ", k, " is not consistent with a ",
         output_features, "-unit MDN head (possible splits: ", splits, ").",
         call. = FALSE)
  }

  if (known_d) {
    hit <- which(sp$D == as.integer(d))
    if (length(hit) == 1L) return(sp$K[hit])
    stop("Model's stored output_dim = ", d, " is not consistent with a ",
         output_features, "-unit MDN head (possible splits: ", splits, ").",
         call. = FALSE)
  }

  if (nrow(sp) == 1L) return(sp$K)

  stop(
    "Cannot tell how many mixture components this MDN head (",
    output_features, " units) has: the model stores neither mdn_components nor ",
    "output_dim, and the head width is consistent with ", nrow(sp),
    " splits (", splits, "). Build the model with probotMakeMDN(), which stores ",
    "both.", call. = FALSE
  )
}

.probotUnpackMDN <- function(output, model){

  batch_size <- output$size(1)

  output_features <- output$size(2)

  mdn_components <- .probotMDNK(model, output_features)

  output_dim <- (output_features / mdn_components - 1) / 2

  stopifnot(output_dim == as.integer(output_dim))

  output_dim <- as.integer(output_dim)

  output <- output$view(
    c(batch_size, mdn_components, 2 * output_dim + 1)
  )

  list(
    mu = output[,,1:output_dim],
    log10_sigma =
      output[,,(output_dim + 1):(2 * output_dim)],
    logits =
      output[,,2 * output_dim + 1]
  )
}
