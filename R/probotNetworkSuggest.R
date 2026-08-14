# ============================================================================
# probotNetworkSuggest.R
# Best-effort heuristics for network complexity given input/output dimensions
# and network type (Flow, MDN, Point).
# ============================================================================

#' Suggest Network Architecture Hyperparameters
#'
#' @description
#' Provides best-effort heuristic suggestions for network complexity given the
#' number of input and output dimensions and the type of network
#' (\code{"Flow"}, \code{"MDN"}, or \code{"Point"}).  The returned list can be
#' passed directly (via \code{do.call}) to the corresponding
#' \code{probotMakeFlow}, \code{probotMakeMDN}, or \code{probotMakePoint}
#' constructor.
#'
#' The heuristics are based on common best-practice rules of thumb intended to
#' balance under- and over-fitting for typical simulation-based inference
#' problems.  The user should always verify the suggested complexity by
#' inspecting training/validation loss curves and adjusting accordingly.
#'
#' @param input_dim Integer.  Number of input features (context / observed
#'   data dimensions).  For \code{type = "Flow"} this corresponds to
#'   \code{dim_x}.
#' @param output_dim Integer.  Number of output dimensions (parameters to
#'   infer).  For \code{type = "Flow"} this corresponds to \code{dim_theta}.
#' @param type Character string; one of \code{"Flow"}, \code{"MDN"}, or
#'   \code{"Point"} (case-insensitive).
#' @param verbose Logical.  If \code{TRUE} (default) a human-readable summary
#'   of the suggestions is printed to the console.
#'
#' @details
#' ## Heuristic rules
#'
#' \strong{Shared rule — reference complexity}
#'
#' A \emph{reference neuron count} is computed as
#' \deqn{N_{\rm ref} = 8 \times (input\_dim + output\_dim).}
#' This is then rounded to a "nice" power-of-two-friendly value and clamped
#' between 32 and 1024 to avoid pathologically small or large layers.
#'
#' \strong{Point networks}
#'
#' * Three hidden layers of widths \eqn{[N_{\rm ref},\; 2 N_{\rm ref},\; N_{\rm ref}]}
#'   (hourglass-style, which provides gradual expansion then compression).
#' * No dropout is recommended for small problems; a small rate (\code{0.1})
#'   is flagged if \eqn{input\_dim \ge 10}.
#'
#' \strong{MDN networks}
#'
#' * Same hidden-layer pattern as Point.
#' * Number of mixture components: \eqn{\max(3,\; \lceil output\_dim / 2 \rceil)},
#'   clamped to 20 to prevent degenerate over-parameterisation.
#'
#' \strong{Flow networks}
#'
#' * Number of coupling layers: \eqn{\max(4,\; 2 \times output\_dim)}, clamped
#'   to 16.
#' * Hidden dimension per coupling layer: \eqn{N_{\rm ref}}, clamped between
#'   32 and 512.
#'
#' @return A named list with entries ready to be passed to the corresponding
#'   constructor:
#'
#' \describe{
#'   \item{\code{type = "Point"}}{
#'     \code{input_dim}, \code{output_dim}, \code{hidden_dims}, \code{dropout}.
#'   }
#'   \item{\code{type = "MDN"}}{
#'     \code{input_dim}, \code{output_dim}, \code{mdn_components},
#'     \code{hidden_dims}, \code{dropout}.
#'   }
#'   \item{\code{type = "Flow"}}{
#'     \code{dim_x}, \code{dim_theta}, \code{n_layers}, \code{hidden_dim}.
#'   }
#' }
#'
#' @author Aaron Robotham
#'
#' @seealso
#' \code{\link{probotMakePoint}}, \code{\link{probotMakeMDN}},
#' \code{\link{probotMakeFlow}}
#'
#' @examples
#' ## Point network suggestion
#' probotNetworkSuggest(input_dim = 5, output_dim = 2, type = "Point")
#'
#' ## MDN network suggestion (verbose = FALSE suppresses the printed summary)
#' probotNetworkSuggest(input_dim = 10, output_dim = 3, type = "MDN",
#'                      verbose = FALSE)
#'
#' ## Flow network suggestion, then pass directly to constructor
#' \dontrun{
#' suggest <- probotNetworkSuggest(input_dim = 6, output_dim = 4, type = "Flow")
#' flow <- do.call(probotMakeFlow, c(suggest, list(device = "cpu")))()
#' }
#'
#' @export
probotNetworkSuggest <- function(input_dim, output_dim, type = "MDN", verbose = TRUE) {

  type <- match.arg(toupper(type), c("FLOW", "MDN", "POINT"))

  stopifnot(
    is.numeric(input_dim),  length(input_dim)  == 1L, input_dim  >= 1,
    is.numeric(output_dim), length(output_dim) == 1L, output_dim >= 1
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
      mdn_components <- pmax(3L, pmin(20L, ceiling(output_dim / 2L)))
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
