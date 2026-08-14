# ============================================================================
# probotNetworkSuggest.R
# Best-effort heuristics for network complexity given input/output dimensions
# and network type (Flow, MDN, Point).
# ============================================================================

probotNetworkSuggest <- function(input_dim, output_dim, type = "MDN", verbose = TRUE) {

  type <- match.arg(toupper(type), c("FLOW", "MDN", "POINT"))

  stopifnot(
    is.numeric(input_dim),  length(input_dim)  == 1L,
    input_dim  >= 1, input_dim  == as.integer(input_dim),
    is.numeric(output_dim), length(output_dim) == 1L,
    output_dim >= 1, output_dim == as.integer(output_dim)
  )

  input_dim  <- as.integer(input_dim)
  output_dim <- as.integer(output_dim)

  # --- Reference neuron count -------------------------------------------------
  # Rule: 8 * total dimension, rounded to nearest 32, clamped [32, 1024]
  n_ref <- 8L * (input_dim + output_dim)
  n_ref <- pmax(32L, pmin(1024L, .round_to_nice(n_ref)))

  # --- Build suggestion -------------------------------------------------------
  result <- switch(type,

    POINT = {
      hidden_dims <- c(n_ref, 2L * n_ref, n_ref)
      dropout     <- if (input_dim >= 10L) 0.1 else 0
      suggestion  <- list(
        input_dim   = input_dim,
        output_dim  = output_dim,
        hidden_dims = hidden_dims,
        dropout     = dropout
      )
      if (verbose) {
        .suggest_print("Point", input_dim, output_dim,
          sprintf("hidden_dims : c(%s)", paste(hidden_dims, collapse = ", ")),
          sprintf("dropout     : %.1f", dropout)
        )
      }
      suggestion
    },

    MDN = {
      hidden_dims   <- c(n_ref, 2L * n_ref, n_ref)
      mdn_components <- pmax(3L, pmin(20L, ceiling(as.numeric(output_dim) / 2)))
      dropout        <- if (input_dim >= 10L) 0.1 else 0
      suggestion <- list(
        input_dim      = input_dim,
        output_dim     = output_dim,
        mdn_components = mdn_components,
        hidden_dims    = hidden_dims,
        dropout        = dropout
      )
      if (verbose) {
        .suggest_print("MDN", input_dim, output_dim,
          sprintf("mdn_components : %d", mdn_components),
          sprintf("hidden_dims    : c(%s)", paste(hidden_dims, collapse = ", ")),
          sprintf("dropout        : %.1f", dropout)
        )
      }
      suggestion
    },

    FLOW = {
      n_layers   <- pmax(4L, pmin(16L, 2L * output_dim))
      hidden_dim <- pmax(32L, pmin(512L, n_ref))
      suggestion <- list(
        dim_x      = input_dim,
        dim_theta  = output_dim,
        n_layers   = n_layers,
        hidden_dim = hidden_dim
      )
      if (verbose) {
        .suggest_print("Flow", input_dim, output_dim,
          sprintf("n_layers   : %d", n_layers),
          sprintf("hidden_dim : %d", hidden_dim)
        )
      }
      suggestion
    }
  )

  invisible(result)
}

# --- Internal helpers ---------------------------------------------------------

# Round x to the nearest multiple of 32 (keeps values "power-of-two-friendly")
.round_to_nice <- function(x) {
  as.integer(round(x / 32) * 32L)
}

.suggest_print <- function(type_label, input_dim, output_dim, ...) {
  lines <- c(...)
  cat(sprintf(
    "probotNetworkSuggest (%s)\n  input_dim  : %d\n  output_dim : %d\n",
    type_label, input_dim, output_dim
  ))
  for (ln in lines) cat("  ", ln, "\n", sep = "")
  cat(
    "  ---\n",
    "  Note: these are heuristic starting points. Always validate with\n",
    "  training/validation loss curves before use in production.\n",
    sep = ""
  )
}
