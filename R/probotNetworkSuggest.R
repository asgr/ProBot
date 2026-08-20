# ============================================================================
# probotNetworkSuggest.R
# Best-effort heuristics for network complexity given input/output dimensions
# and network type (Flow, MDN, Point).
# ============================================================================

probotNetworkSuggest = function(input_dim,
                                 output_dim,
                                 n_train = NULL,
                                 type = "MDN",
                                 verbose = TRUE) {
  type = match.arg(toupper(type), c("FLOW", "MDN", "POINT"))

  stopifnot(
    is.numeric(input_dim),
    length(input_dim)  == 1L,
    input_dim  >= 1,
    input_dim  == as.integer(input_dim),
    is.numeric(output_dim),
    length(output_dim) == 1L,
    output_dim >= 1,
    output_dim == as.integer(output_dim)
  )

  input_dim  = as.integer(input_dim)
  output_dim = as.integer(output_dim)

  # --- Reference neuron count -------------------------------------------------
  # Rule: 8 * total dimension, rounded to nearest 32, clamped [32, 1024]
  dim_complexity = 8 * (input_dim + output_dim)

  if (is.null(n_train)) {
    n_ref = pmax(32L, pmin(1024L, .round_to_nice(dim_complexity)))
  } else{
    data_complexity = sqrt(n_train)
    n_ref = sqrt(dim_complexity * data_complexity)
    n_ref = pmax(32L, pmin(1024L, .round_to_nice(n_ref)))
  }

  # --- Build suggestion -------------------------------------------------------
  result = switch(type,

                   POINT = {
                     hidden_dims = c(n_ref, 2L * n_ref, n_ref)
                     dropout     = if (input_dim >= 10L)
                       0.1
                     else
                       0
                     suggestion  = list(
                       input_dim   = input_dim,
                       output_dim  = output_dim,
                       hidden_dims = hidden_dims,
                       dropout     = dropout
                     )
                     if (verbose) {
                       .suggest_print(
                         "Point",
                         input_dim,
                         output_dim,
                         sprintf("hidden_dims : c(%s)", paste(hidden_dims, collapse = ", ")),
                         sprintf("dropout     : %.1f", dropout)
                       )
                     }
                     suggestion
                   },

                   MDN = {
                     hidden_dims   = c(n_ref, 2L * n_ref, n_ref)

                     if (is.null(n_train)) {
                       mdn_components = pmax(3L, pmin(20L, ceiling(output_dim / 2)))
                     } else {
                       mdn_components = pmax(3L, pmin(20L, round(output_dim * log10(n_train) / 4)))
                     }

                     dropout        = if (input_dim >= 10L)
                       0.1
                     else
                       0
                     suggestion = list(
                       input_dim      = input_dim,
                       output_dim     = output_dim,
                       mdn_components = mdn_components,
                       hidden_dims    = hidden_dims,
                       dropout        = dropout
                     )
                     if (verbose) {
                       .suggest_print(
                         "MDN",
                         input_dim,
                         output_dim,
                         sprintf("mdn_components : %d", mdn_components),
                         sprintf("hidden_dims    : c(%s)", paste(hidden_dims, collapse = ", ")),
                         sprintf("dropout        : %.1f", dropout)
                       )
                     }
                     suggestion
                   },

                   FLOW = {
                     n_layers   = pmax(4L, pmin(16L, 2L + output_dim))

                     if (!is.null(n_train)) {
                       n_layers = round(n_layers *
                                           pmin(2, log10(n_train) / 4))

                     }

                     hidden_dim = pmax(32L, pmin(512L, n_ref))
                     suggestion = list(
                       dim_x      = input_dim,
                       dim_theta  = output_dim,
                       n_layers   = n_layers,
                       hidden_dim = hidden_dim
                     )
                     if (verbose) {
                       .suggest_print(
                         "Flow",
                         input_dim,
                         output_dim,
                         n_train,
                         sprintf("n_layers   : %d", n_layers),
                         sprintf("hidden_dim : %d", hidden_dim)
                       )
                     }
                     suggestion
                   })

  invisible(result)
}

# --- Internal helpers ---------------------------------------------------------

# Round x to the nearest multiple of 32 (keeps values "power-of-two-friendly")
.round_to_nice = function(x) {
  as.integer(round(x / 32) * 32L)
}

.suggest_print = function(type_label,
                           input_dim,
                           output_dim,
                           n_train = NULL,
                           ...) {
  lines = c(...)

  cat(
    sprintf(
      "probotNetworkSuggest (%s)\n  input_dim  : %d\n  output_dim : %d\n",
      type_label,
      input_dim,
      output_dim
    )
  )

  if (!is.null(n_train)) {
    cat(sprintf(
      "  n_train    : %s\n",
      format(n_train, scientific = FALSE, big.mark = ",")
    ))
  }

  for (ln in lines) {
    cat("  ", ln, "\n", sep = "")
  }

  cat(
    "  ---\n",
    "  Note: these are heuristic starting points. Always validate with\n",
    "  training/validation loss curves before use in production.\n",
    sep = ""
  )
}
