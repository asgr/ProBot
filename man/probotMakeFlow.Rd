\name{probotMakeFlow}
\alias{probotMakeFlow}
\title{Construct a Normalizing Flow Network Module}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a normalizing flow
for learning a conditional probability distribution. The network maps
parameters to a latent space (forward) and vice versa (inverse). Three
architectures are available: a stack of affine Real Non-Volume Preserving
(RealNVP) layers
(\code{style = "realnvp", the default}), masked autoregressive blocks with
permutation layers (\code{style = "maf"}), or rational-quadratic spline
coupling layers (\code{style = "nsf"}). See \code{details} for a
comparison and guidance on choosing between them.
}
\usage{
probotMakeFlow(input_dim, output_dim,
  n_layers = 4, hidden_dim = 32, n_blocks = NULL, n_layers_per_block = 2,
  soft_clamp = 3, device = NULL, style = "realnvp", n_bins = 8, tail_bound = 3
)
}
\arguments{
    \item{input_dim}{Dimensionality of the context/input features used to
    condition the flow.}

    \item{output_dim}{Total dimensionality of the parameter space.}

    \item{n_layers}{Number of layers to stack: RealNVP layers when
    \code{style = "realnvp"}, spline coupling layers when \code{style = "nsf"}.
    When \code{style = "maf"} and \code{n_blocks} is \code{NULL} (the default)
    this is used as the number of autoregressive blocks, so the output of
    \code{\link{probotNetworkSuggest}} can be passed straight in. Defaults to 4.}

    \item{hidden_dim}{Number of units in each hidden layer of the internal
    shift/scale network within each RealNVP layer, or of the masked MLP in
    each autoregressive block. \code{probotMakeFlow()} passes this value
    through unchanged (default 32); calling
    \code{\link{probotFlowRealNVP}} or
    \code{\link{probotFlowMAF}} directly instead uses their own
    defaults (32 and 500 respectively).}

    \item{n_blocks}{Number of autoregressive blocks when \code{style = "maf"}.
    Defaults to \code{NULL}, meaning \code{n_layers} is used; set it explicitly
    to decouple block count from \code{n_layers}. Ignored by the other styles.}

    \item{n_layers_per_block}{Number of layers per block when \code{style = "maf"}}

    \item{soft_clamp}{Upper bound on the magnitude of the log-scale, applied
    via a soft \eqn{\tanh} clamp so the scale stays bounded. Larger values
    allow wider scales; values between 1 and 5 are typical. Defaults to 3.}
    \item{n_bins}{Number of rational-quadratic spline bins when
    \code{style = "nsf"}. Defaults to 8.}
    \item{tail_bound}{Spline interval bound when \code{style = "nsf"}; the
    transform is the identity outside \code{[-tail_bound, tail_bound]}.
    Defaults to 3.}

    \item{device}{Character specifying the computing device: \code{"cpu"},
    \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), auto-detects
    MPS, falling back to CPU.}

    \item{style}{Which flow architecture to build: \code{"realnvp"} (stacked
    affine RealNVP layers, the default), \code{"maf"} (masked
    autoregressive blocks with alternating permutation layers), or
    \code{"nsf"} (rational-quadratic spline coupling layers). Determines
    which module is returned: \code{\link{probotFlowRealNVP}},
    \code{\link{probotFlowMAF}}, or \code{\link{probotFlowNSF}}.}
}
\details{
\code{probotMakeFlow()} is a facade: it returns a zero-argument constructor
that instantiates one of the three architectures. Call the result with \code{()}
to obtain the \code{\link[torch]{nn_module}}. This is the entry point used by
\code{\link{probotLoad}} to reconstruct a saved flow from its metadata.

\subsection{Flow architectures}{

All three architectures express the same object — an exact conditional density
\eqn{p(\theta \mid x)} over the parameters — but they build it in different
ways, which leads to practical differences worth understanding.

\strong{Real Non-Volume Preserving (RealNVP).} The flow is a stack of \code{n_layers}
RealNVP layers. Each layer splits \eqn{\theta} into two halves and lets one
half (together with the context \eqn{x}) predict an affine shift/scale
transformation of the \emph{other} half; the halves are then swapped. Because
each layer moves only about half the coordinates, \code{n_layers} layers are
needed before every coordinate has interacted with the others — mixing is
gradual. Both the forward and inverse passes are fully parallel over
coordinates and cost \emph{O}(\code{n_layers}) network evaluations each.

\strong{Masked Autoregressive Flows (MAF).} The flow is a stack of \code{n_blocks} masked
autoregressive blocks separated by permutation layers. Inside a block, every
coordinate's affine transformation depends on all coordinates ordered before
it (enforced by a weight mask), so the whole joint density is factorised as a
product of \code{output_dim} conditionals in a single block — far richer
per-layer mixing than RealNVP. The forward pass evaluates the masked MLP
once per block, so training is fast; the inverse (sampling) pass is
\emph{sequential}: coordinate \eqn{j} must be reconstructed before coordinate
\eqn{j + 1}, costing \code{output_dim} network evaluations per block.

\strong{Neural spline flow (NSF).} The flow replaces each affine coupling
transformation with a monotonic rational-quadratic spline. The spline
parameters are conditioned on the complementary half of the parameter vector
and \eqn{x}, so its forward and inverse passes remain parallel while modeling
nonlinear marginal transformations. It is the identity outside
\code{[-tail_bound, tail_bound]}.

\strong{Practical implications.}
\itemize{
  \item \emph{Calibration.} An autoregressive flow typically reaches lower
  training (negative-log-likelihood) loss and better posterior calibration
  than a RealNVP flow of the same depth, because each block captures more of
  the joint structure in one pass. Choose \code{style = "maf"} when
  posterior fidelity matters and \code{output_dim} is moderate.
  \item \emph{Sampling speed.} For posterior sampling the RealNVP style is
  usually much cheaper: its inverse pass is parallel, whereas the
  autoregressive inverse cost grows with both \code{output_dim} and
  \code{n_blocks}. If you draw a large number of posterior samples (e.g., for
  CRPS / PIT / TARP), the autoregressive inverse pass can dominate wall-clock
  time.
  \item \emph{Small \code{output_dim}.} A single parameter dimension is not
  supported by either flow (a lone coordinate has nothing to couple or
  autoreg over). At \code{output_dim = 2} the two architectures are close
  in expressive power; their differences matter more as \code{output_dim}
  grows.
  \item \emph{Shared setup.} Both flows train the same way via
  \code{\link{probotTrainFlow}} and \code{\link{probotLossNF}}, rely on the
  standardisation from \code{\link{probotScaleForward}}, and are reconstructed
  on load from the \code{flow_style} metadata field. Validate the choice of
  architecture with calibration diagnostics (see \code{\link{probotNetworkSuggest}})
  rather than assuming it.
}
}

The forward pass maps parameters \eqn{\theta} to latent space \eqn{z},
accumulating the log determinant of the Jacobian across all layers. The
inverse pass maps from latent space back to parameter space, applied in
reverse layer/block order.
}
\value{
A zero-argument function. Call it with \code{()} to obtain an instantiated
\code{\link[torch]{nn_module}} — the same \code{probotMakeX(...)} \code{()}
pattern used by the other ProBot model constructors.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotFlowRealNVP}},
\code{\link{probotFlowMAF}},
\code{\link{probotFlowNSF}},
\code{\link{probotLossNF}},
\code{\link{probotTrainFlow}},
\code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Create a RealNVP flow (default) with 4 layers
flow <- probotMakeFlow(input_dim = 3, output_dim = 2, device = "cpu")()

## Or a Masked Autoregressive Flow (MAF)
flow_ar <- probotMakeFlow(input_dim = 3, output_dim = 2, style = "maf", device = "cpu")()

## Or a rational-quadratic neural spline flow
flow_nsf <- probotMakeFlow(input_dim = 3, output_dim = 2, style = "nsf", device = "cpu")()

## Forward and inverse passes require torch tensors
theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

out <- flow$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
