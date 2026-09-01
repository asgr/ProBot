\name{probotNetworkSuggest}
\alias{probotNetworkSuggest}
\title{Suggest Network Architecture Hyperparameters}
\description{
Provides best-effort heuristic suggestions for network complexity given the
number of input and output dimensions and the type of network
(\code{"Flow"}, \code{"MDN"}, or \code{"Point"}).  The returned list can be
passed directly (via \code{do.call}) to the corresponding
\code{\link{probotMakeFlow}}, \code{\link{probotMakeMDN}}, or
\code{\link{probotMakePoint}} constructor.

The heuristics are based on common best-practice rules of thumb intended to
balance under- and over-fitting for typical simulation-based inference
problems.  The user should always verify the suggested complexity by
inspecting training/validation loss curves and adjusting accordingly.
}
\usage{
probotNetworkSuggest(
  input_dim,
  output_dim,
  n_train = NULL,
  type = "MDN",
  flow_style = "realnvp",
  verbose = TRUE
)
}
\arguments{
  \item{input_dim}{Integer. Number of input features (context / observed data
    dimensions). For \code{type = "Flow"} this corresponds to \code{dim_x}.}

  \item{output_dim}{Integer. Number of output dimensions (parameters to
    infer). For \code{type = "Flow"} this corresponds to \code{dim_theta}.}

  \item{n_train}{Optional integer: the number of training observations. When
    supplied, the heuristics are additionally scaled by
    \eqn{\sqrt{\mathtt{n\_train}}}: the reference neuron count is multiplied
    by this factor, the number of MDN mixture components is scaled by
    \code{round(output_dim * log10(n_train) / 4)}, and the Flow number of
    layers is scaled by \code{log10(n_train) / 4} (capped at \code{2}). If
    \code{NULL} (default), the suggestions depend only on the input/output
    dimensions.}

  \item{type}{Character string; one of \code{"Flow"}, \code{"MDN"}, or
    \code{"Point"} (case-insensitive). Defaults to \code{"MDN"}.}

  \item{flow_style}{Character string; \code{"realnvp"} (default) or
    \code{"maf"}. Only meaningful for \code{type = "Flow"}, but validated
    for all types so an invalid value fails fast.}

  \item{verbose}{Logical. If \code{TRUE} (default) a human-readable summary
    of the suggestions is printed to the console.}
}
\details{
\subsection{Heuristic rules}{

\strong{Shared rule — reference neuron count}

A \emph{reference neuron count} is computed as
\deqn{N_{\mathrm{ref}} = 8 \times (\mathtt{input\_dim} + \mathtt{output\_dim}),}
rounded to the nearest multiple of 32 and clamped between 32 and 1024 to
avoid pathologically small or large layers.

\strong{Point networks}
\itemize{
  \item Three hidden layers of widths
    \eqn{[N_{\mathrm{ref}},\; 2N_{\mathrm{ref}},\; N_{\mathrm{ref}}]}
    (hourglass / bottleneck pattern).
  \item No dropout for small problems (\code{input_dim < 10}); a small rate of
    \code{0.1} is recommended when \code{input_dim >= 10}.
}

\strong{MDN networks}
\itemize{
  \item Same hidden-layer pattern as Point networks.
  \item Number of mixture components:
    \eqn{\max\!\left(3,\,\lceil \mathtt{output\_dim}/2 \rceil\right)} when
    \code{n_train} is \code{NULL}, otherwise
    \eqn{\max\!\left(3,\,\mathtt{output\_dim} \times
    \log_{10}(\mathtt{n\_train})/4\right)}, in both cases clamped to 20 to
    prevent degenerate over-parameterisation.
  \item Dropout as per Point rule.
}

\strong{Flow networks}
\itemize{
  \item Number of layers (RealNVP layers, or autoregressive blocks):
    \eqn{\max\!\left(4,\,2 + \mathtt{output\_dim}\right)}, clamped to 16;
    when \code{n_train} is supplied it is further scaled by
    \code{log10(n_train) / 4} (capped at \code{2}).
  \item Hidden dimension per layer: \eqn{N_{\mathrm{ref}}}, clamped between
    32 and 512. Note this returns \code{hidden_dim}, which
    \code{\link{probotMakeFlow}} passes through unchanged.
}
}
}
\value{
A named list with entries ready to be passed to the corresponding constructor.
The corresponding \code{probotMake*} constructors accept (and ignore) any
extra arguments such as \code{n_params}, so the list can be passed directly
via \code{do.call}.

\describe{
  \item{\code{type = "Point"}}{
    \code{input_dim}, \code{output_dim}, \code{hidden_dims}, \code{dropout},
    \code{n_params}.
  }
  \item{\code{type = "MDN"}}{
    \code{input_dim}, \code{output_dim}, \code{mdn_components},
    \code{hidden_dims}, \code{dropout}, \code{n_params}.
  }
  \item{\code{type = "Flow"}}{
    \code{dim_x}, \code{dim_theta}, \code{n_layers}, \code{hidden_dim},
    \code{style}, \code{n_params}.
  }
}

\code{n_params} is the estimated total number of trainable parameters
(weights + biases) for the suggested architecture.

The function also prints a human-readable summary when \code{verbose = TRUE}.
The return value is invisibly returned so that it can be assigned or piped
without cluttering the console when \code{verbose = TRUE}.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakePoint}}, \code{\link{probotMakeMDN}},
\code{\link{probotMakeFlow}}
}
\examples{
## --- Point network suggestion ------------------------------------------------
probotNetworkSuggest(input_dim = 5, output_dim = 2, type = "Point")

## --- MDN network suggestion --------------------------------------------------
## verbose = FALSE suppresses the printed summary; result is still returned
s <- probotNetworkSuggest(input_dim = 10, output_dim = 3, type = "MDN",
                          verbose = FALSE)
str(s)

## --- Flow network suggestion -------------------------------------------------
probotNetworkSuggest(input_dim = 6, output_dim = 4, type = "Flow")

## --- Pass directly to network constructor ------------------------------------
\dontrun{
suggest <- probotNetworkSuggest(input_dim = 6, output_dim = 4,
                                type = "Flow", verbose = TRUE)
flow <- do.call(probotMakeFlow, c(suggest, list(device = "cpu")))()

suggest_mdn <- probotNetworkSuggest(input_dim = 10, output_dim = 3,
                                    type = "MDN", verbose = TRUE)
mdn <- do.call(probotMakeMDN, c(suggest_mdn, list(device = "cpu")))()

suggest_pt <- probotNetworkSuggest(input_dim = 5, output_dim = 2,
                                   type = "Point", verbose = TRUE)
pt <- do.call(probotMakePoint, c(suggest_pt, list(device = "cpu")))()
}
}
