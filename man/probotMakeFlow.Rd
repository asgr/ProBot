\name{probotMakeFlow}
\alias{probotMakeFlow}
\title{Construct a Normalizing Flow Network Module}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a normalizing flow
for learning a conditional probability distribution. The network maps
parameters to a latent space (forward) and vice versa (inverse). Two
architectures are available: a stack of affine coupling layers
(\code{style = "couple", the default}) or masked autoregressive blocks with
permutation layers (\code{style = "autoreg"}). See \code{details} for a
comparison and guidance on choosing between them.
}
\usage{
probotMakeFlow(
  dim_theta,
  dim_x,
  n_layers = 4,
  hidden_dim = 32,
  soft_clamp = 3,
  device = NULL,
  style = "couple"
)
}
\arguments{
    \item{dim_theta}{Total dimensionality of the parameter space.}

    \item{dim_x}{Dimensionality of the context/input features used to
    condition the flow.}

    \item{n_layers}{Number of coupling layers to stack, or autoregressive
    blocks when \code{style = "autoreg"}. Defaults to 4.}

    \item{hidden_dim}{Number of units in each hidden layer of the internal
    shift/scale network within each coupling layer, or of the masked MLP in
    each autoregressive block. \code{probotMakeFlow()} passes this value
    through unchanged (default 32); calling
    \code{\link{probotMakeFlowCouple}} or
    \code{\link{probotFlowAutoReg}} directly instead uses their own
    defaults (32 and 500 respectively).}

    \item{soft_clamp}{Upper bound on the magnitude of the log-scale, applied
    via a soft \eqn{\tanh} clamp so the scale stays bounded. Larger values
    allow wider scales; values between 1 and 5 are typical. Defaults to 3.}

    \item{device}{Character specifying the computing device: \code{"cpu"},
    \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), auto-detects
    MPS, falling back to CPU.}

    \item{style}{Which flow architecture to build: \code{"couple"} (stacked
    affine NVP coupling layers, the default) or \code{"autoreg"} (masked
    autoregressive blocks with alternating permutation layers). Determines
    which module is returned: \code{\link{probotMakeFlowCouple}} or
    \code{\link{probotFlowAutoReg}}.}
}
\details{
\code{probotMakeFlow()} is a facade: it returns a zero-argument constructor
that instantiates one of the two architectures. Call the result with \code{()}
to obtain the \code{\link[torch]{nn_module}}. This is the entry point used by
\code{\link{probotLoad}} to reconstruct a saved flow from its metadata.

\subsection{Coupling vs. autoregressive architecture}{

Both architectures express the same object — an exact conditional density
\eqn{p(\theta \mid x)} over the parameters — but they build it in different
ways, which leads to practical differences worth understanding.

\strong{Coupling (affine NVP).} The flow is a stack of \code{n_layers}
coupling layers. Each layer splits \eqn{\theta} into two halves and lets one
half (together with the context \eqn{x}) predict an affine shift/scale
transformation of the \emph{other} half; the halves are then swapped. Because
each layer moves only about half the coordinates, \code{n_layers} layers are
needed before every coordinate has interacted with the others — mixing is
gradual. Both the forward and inverse passes are fully parallel over
coordinates and cost \emph{O}(\code{n_layers}) network evaluations each.

\strong{Autoregressive (MAF).} The flow is a stack of \code{n_blocks} masked
autoregressive blocks separated by permutation layers. Inside a block, every
coordinate's affine transformation depends on all coordinates ordered before
it (enforced by a weight mask), so the whole joint density is factorised as a
product of \code{dim_theta} conditionals in a single block — far richer
per-layer mixing than coupling. The forward pass evaluates the masked MLP
once per block, so training is fast; the inverse (sampling) pass is
\emph{sequential}: coordinate \eqn{j} must be reconstructed before coordinate
\eqn{j + 1}, costing \code{dim_theta} network evaluations per block.

\strong{Practical implications.}
\itemize{
  \item \emph{Calibration.} An autoregressive flow typically reaches lower
  training (negative-log-likelihood) loss and better posterior calibration
  than a coupling flow of the same depth, because each block captures more of
  the joint structure in one pass. Choose \code{style = "autoreg"} when
  posterior fidelity matters and \code{dim_theta} is moderate.
  \item \emph{Sampling speed.} For posterior sampling the coupling style is
  usually much cheaper: its inverse pass is parallel, whereas the
  autoregressive inverse cost grows with both \code{dim_theta} and
  \code{n_blocks}. If you draw a large number of posterior samples (e.g., for
  CRPS / PIT / TARP), the autoregressive inverse pass can dominate wall-clock
  time.
  \item \emph{Small \code{dim_theta}.} A single parameter dimension is not
  supported by either flow (a lone coordinate has nothing to couple or
  autoregress over). At \code{dim_theta = 2} the two architectures are close
  in expressive power; their differences matter more as \code{dim_theta}
  grows.
  \item \emph{Shared setup.} Both flows train the same way via
  \code{\link{probotTrainFlow}} and \code{\link{probotLossNF}}, rely on the
  standardisation from \code{\link{probotScaleForward}}, and are reconstructed
  on load from the \code{flow_style} metadata field. Validate the choice of
  architecture with calibration diagnostics (see \code{\link{probotAssess}})
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
\code{\link{probotMakeFlowCouple}},
\code{\link{probotFlowAutoReg}},
\code{\link{probotLossNF}},
\code{\link{probotTrainFlow}},
\code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Create a coupling flow (default) with 4 layers
flow <- probotMakeFlow(dim_theta = 2, dim_x = 3, device = "cpu")()

## Or an autoregressive flow
flow_ar <- probotMakeFlow(dim_theta = 2, dim_x = 3, style = "autoreg", device = "cpu")()

## Forward and inverse passes require torch tensors
theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

out <- flow$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
