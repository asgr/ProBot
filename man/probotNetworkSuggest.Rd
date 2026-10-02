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
    dimensions). For \code{type = "Flow"} this corresponds to \code{input_dim}.}

  \item{output_dim}{Integer. Number of output dimensions (parameters to
    infer). For \code{type = "Flow"} this corresponds to \code{output_dim}.}

  \item{n_train}{Optional integer: the number of training observations. When
    supplied the suggestions additionally depend on \code{n_train}: the
    reference neuron count becomes the geometric mean of the dimensionality
    term and \eqn{\sqrt{\mathtt{n\_train}}} (see \emph{Details}), the number
    of MDN mixture components becomes
    \code{round(output_dim * log10(n_train) / 4)}, and the Flow layer count is
    multiplied by \code{log10(n_train) / 4} capped at \code{2} for
    \code{flow_style = "realnvp"}, or by \code{log10(n_train) / 5} capped at
    \code{1.5} for \code{"maf"} and \code{"nsf"}. Each style then re-clamps
    the result to its own depth range. If \code{NULL} (default), the
    suggestions depend only on the input/output dimensions.}

  \item{type}{Character string; one of \code{"Flow"}, \code{"MDN"}, or
    \code{"Point"} (case-insensitive). Defaults to \code{"MDN"}.}

  \item{flow_style}{Character string; one of \code{"realnvp"} (default),
    \code{"maf"} or \code{"nsf"} (case-insensitive). Only meaningful for
    \code{type = "Flow"}, but validated for all types so an invalid value
    fails fast.}

  \item{verbose}{Logical. If \code{TRUE} (default) a human-readable summary
    of the suggestions is printed to the console.}
}
\details{
\subsection{Heuristic rules}{

\strong{Shared rule - reference neuron count}

With \code{n_train} omitted (the default), a \emph{reference neuron count} is
computed from the dimensionality term alone,
\deqn{N_{\mathrm{ref}} = 8 \times (\mathtt{input\_dim} + \mathtt{output\_dim}),}
rounded to the nearest multiple of 32 and clamped between 32 and 1024 to
avoid pathologically small or large layers.

When \code{n_train} is supplied, the reference count is instead the geometric
mean of that same dimensionality term and the data term
\eqn{\sqrt{\mathtt{n\_train}}},
\deqn{N_{\mathrm{ref}} = \sqrt{8 \times (\mathtt{input\_dim} +
  \mathtt{output\_dim}) \times \sqrt{\mathtt{n\_train}}},}
rounded and clamped to the same \eqn{[32, 1024]} range. Uncapped, width
therefore grows as \eqn{n^{1/4}} and the parameter count of a fixed-shape MLP
as \eqn{n^{1/2}}. Because the clamps are applied \emph{after} this, however,
\code{n_train} stops having any effect outside a range that depends on
\eqn{D = input\_dim + output\_dim}: the floor binds below
\eqn{n \approx (32^2 / 8D)^2} and the ceiling above
\eqn{n \approx (1024^2 / 8D)^2}, so at \eqn{D = 76} the suggestion is already
pinned at 1024 for \eqn{n} above roughly \eqn{3 \times 10^6}.

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

The three styles have different capacity economics (see also
\file{vignettes/probot-realnvp-or-maf.Rmd}), so depth and width are set
separately for each. In every case the layer count is formed from
\code{output_dim} first, optionally multiplied by the \code{n_train} factor,
and then clamped to that style's range:

\describe{
  \item{\code{"realnvp"}}{Coupling transforms only half the dimensions per
    layer, so depth is what mixes the parameter space, while the inverse pass
    at sample time is a single vectorised sweep whose cost is flat in
    \code{output_dim}. Depth is
    \eqn{\max(4,\,\min(24,\;4 + \mathtt{output\_dim}))} scaled by
    \code{min(2, log10(n_train) / 4)} and re-clamped to \eqn{[4, 24]}; width
    is \eqn{N_{\mathrm{ref}}} clamped to \eqn{[32, 512]}.}
  \item{\code{"maf"}}{MADE blocks already condition each dimension on its
    full prefix, so depth buys less and each block multiplies the sequential
    inverse cost by \code{output_dim}. Blocks are
    \eqn{\max(3,\,\min(10,\;3 + \lceil \mathtt{output\_dim}/4 \rceil))} scaled
    by \code{min(1.5, log10(n_train) / 5)} and re-clamped to \eqn{[3, 10]};
    width is over-provisioned to \eqn{1.5 \times N_{\mathrm{ref}}} clamped to
    \eqn{[64, 512]}.}
  \item{\code{"nsf"}}{Rational-quadratic splines are more expressive than
    affine couplings while keeping parallel sampling, so depth sits between
    the other two: \eqn{\max(4,\,\min(12,\;4 + \lceil \mathtt{output\_dim}/2
    \rceil))}, scaled and clamped exactly as for \code{"maf"}. Width matches
    \code{"maf"} (\eqn{1.5 \times N_{\mathrm{ref}}} in \eqn{[64, 512]}), and
    \code{n_bins} is fixed at 8.}
}

Note this returns \code{hidden_dim}, which \code{\link{probotMakeFlow}} passes
through unchanged. For \code{"maf"} the returned \code{n_layers} is read as a
\emph{block} count (the facade maps \code{n_layers} to \code{n_blocks});
\code{n_layers_per_block} stays at its default of 2.
}
}
\value{
A named list with entries ready to be passed to the corresponding constructor.
The \code{probotMake*} constructors take no \code{\dots}, so the list can be
spliced straight into them via \code{do.call}.

\describe{
  \item{\code{type = "Point"}}{
    \code{input_dim}, \code{output_dim}, \code{hidden_dims}, \code{dropout}.
  }
  \item{\code{type = "MDN"}}{
    \code{input_dim}, \code{output_dim}, \code{mdn_components},
    \code{hidden_dims}, \code{dropout}.
  }
  \item{\code{type = "Flow"}}{
    \code{input_dim}, \code{output_dim}, \code{n_layers}, \code{hidden_dim},
    \code{style} (and \code{n_bins} when \code{flow_style = "nsf"}).
  }
}

The estimated total number of trainable parameters (weights + biases) for the
suggested architecture is available as \code{attr(s, "n_params")}. It lives on
the attribute (not as a list element) precisely so the list itself can be
spliced into the constructor without an "unused argument" error.
It is also printed in the verbose summary.

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
