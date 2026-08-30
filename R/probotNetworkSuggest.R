# ============================================================================
# probotNetworkSuggest.R
# Best-effort heuristics for network complexity given input/output dimensions
# and network type (Flow, MDN, Point).
# ============================================================================

probotNetworkSuggest = function(input_dim,
                                 output_dim,
                                 n_train = NULL,
                                 type = "MDN",
                                 flow_style = "couple",
                                 verbose = TRUE) {
  type = match.arg(toupper(type), c("FLOW", "MDN", "POINT"))

  # flow_style is only meaningful for type = "Flow", but validated up front so
  # an invalid value fails fast regardless of type.
  flow_style = match.arg(tolower(flow_style), c("couple", "autoreg"))

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

  # n_train drives capacity scaling via log10()/sqrt(); NA would silently leak
  # NA dimensions through the whole suggestion. Treat missing as "unknown"
  # (the NULL branch), but reject non-finite or non-positive numbers.
  if (!is.null(n_train)) {
    if (length(n_train) == 1L && is.na(n_train)) {
      n_train = NULL
    } else {
      stopifnot(
        is.numeric(n_train),
        length(n_train) == 1L,
        !is.na(n_train),
        is.finite(n_train),
        n_train > 0
      )
      n_train = as.integer(n_train)
    }
  }

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
                     n_params    = .probotMLPParamCount(c(input_dim, hidden_dims, output_dim))
                     suggestion  = list(
                       input_dim   = input_dim,
                       output_dim  = output_dim,
                       hidden_dims = hidden_dims,
                       dropout     = dropout,
                       n_params    = n_params
                     )
                     if (verbose) {
                       .suggest_print(
                         "Point",
                         input_dim,
                         output_dim,
                         n_train,
                         sprintf("hidden_dims : c(%s)", paste(hidden_dims, collapse = ", ")),
                         sprintf("dropout     : %.1f", dropout),
                         .suggest_params_line(n_params)
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
                     # Final layer emits mdn_components * (2 * output_dim + 1)
                     # outputs (mu, log10_sigma, logit per component), matching
                     # probotMakeMDN().
                     n_params = .probotMLPParamCount(
                       c(input_dim, hidden_dims, mdn_components * (2 * output_dim + 1))
                     )
                     suggestion = list(
                       input_dim      = input_dim,
                       output_dim     = output_dim,
                       mdn_components = mdn_components,
                       hidden_dims    = hidden_dims,
                       dropout        = dropout,
                       n_params       = n_params
                     )
                     if (verbose) {
                       .suggest_print(
                         "MDN",
                         input_dim,
                         output_dim,
                         n_train,
                         sprintf("mdn_components : %d", mdn_components),
                         sprintf("hidden_dims    : c(%s)", paste(hidden_dims, collapse = ", ")),
                         sprintf("dropout        : %.1f", dropout),
                         .suggest_params_line(n_params)
                       )
                     }
                     suggestion
                   },

                   FLOW = {
                     # Coupling and autoregressive flows have different capacity
                     # economics (see vignettes/probot-coupling-or-autoreg.Rmd):
                     #   - Coupling transforms only half the dims per layer, so it
                     #     NEEDS depth to mix every dimension, but its inverse pass
                     #     (sampling) is a single vectorised sweep whose cost is flat
                     #     in dim_theta. => spend on n_layers, keep hidden modest.
                     #   - AutoReg conditions each dim on its full prefix within a
                     #     single block, so depth buys less, but each block multiplies
                     #     the sequential inverse cost by dim_theta. => keep blocks
                     #     low, put capacity into conditioner width instead.
                     if (flow_style == "couple") {
                       # Depth scales with dim_theta: >= ~2 layers needed per full
                       # pass over the parameter space, capped generously because
                       # deeper coupling is cheap at sample time.
                       n_layers = pmax(4L, pmin(24L, 4L + output_dim))
                       if (!is.null(n_train)) {
                         n_layers = n_layers * pmin(2, log10(n_train) / 4)
                       }
                       n_layers = pmax(4L, pmin(24L, as.integer(round(n_layers))))
                       hidden_dim = pmax(32L, pmin(512L, n_ref))
                     } else {
                       # Fewer blocks (prefix conditioning is built in); scaled
                       # conservatively since each block is dim_theta sequential
                       # masked passes when sampling.
                       n_layers = pmax(3L, pmin(10L, 3L + ceiling(output_dim / 4)))
                       if (!is.null(n_train)) {
                         n_layers = n_layers * pmin(1.5, log10(n_train) / 5)
                       }
                       n_layers = pmax(3L, pmin(10L, as.integer(round(n_layers))))
                       # Width is comparatively free at sample time -> overprovision.
                       hidden_dim = pmax(64L, pmin(512L, .round_to_nice(1.5 * n_ref)))
                     }

                     # Parameter counts mirror the architectures exactly (trainable
                     # weights + biases). MaskedLinear masks weights at run time but
                     # still registers the full nn_linear parameter tensor, so MADE
                     # blocks are counted like dense layers.
                     if (flow_style == "couple") {
                       d1 = floor(output_dim / 2)
                       d2 = output_dim - d1
                       # shift_scale_net per layer: (d1 + dim_x) -> h -> h -> 2*d2
                       per_layer = (d1 + input_dim + 1) * hidden_dim +
                         (hidden_dim + 1) * hidden_dim +
                         (hidden_dim + 1) * (2 * d2)
                       n_params = as.numeric(n_layers) * per_layer
                     } else {
                       # Each autoreg block: masked MLP (d_theta + dim_x) -> h -> h -> 2*d_theta
                       # (n_layers_per_block is fixed at 2 by probotMakeFlow());
                       # permutation layers are parameter-free.
                       per_block = (output_dim + input_dim + 1) * hidden_dim +
                         (hidden_dim + 1) * hidden_dim +
                         (hidden_dim + 1) * (2 * output_dim)
                       n_params = as.numeric(n_layers) * per_block
                     }

                     suggestion = list(
                       dim_x      = input_dim,
                       dim_theta  = output_dim,
                       n_layers   = n_layers,
                       hidden_dim = hidden_dim,
                       style      = flow_style,
                       n_params   = n_params
                     )
                     if (verbose) {
                       .suggest_print(
                         if (flow_style == "couple") "Flow (coupling)" else "Flow (autoreg)",
                         input_dim,
                         output_dim,
                         n_train,
                         if (flow_style == "couple") "style      : couple" else "style      : autoreg",
                         sprintf("n_layers   : %d%s", n_layers,
                                 if (flow_style == "couple") "" else "  (autoregressive blocks)"),
                         sprintf("hidden_dim : %d", hidden_dim),
                         .suggest_params_line(n_params)
                       )
                     }
                     suggestion
                   })

  invisible(result)
}

# --- Internal helpers ---------------------------------------------------------

# Trainable parameter count of a dense MLP given the full layer-width vector
# c(input_dim, hidden_dims..., output_dim): each nn_linear contributes
# in * out weights + out biases.
.probotMLPParamCount = function(dims) {
  sums = 0
  for (i in seq_along(dims)[-1]) {
    sums = sums + as.numeric(dims[i - 1]) * dims[i] + dims[i]
  }
  sums
}

.suggest_params_line = function(n_params) {
  sprintf("n_params   : ~%s", format(as.integer(round(n_params)), big.mark = ","))
}

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
