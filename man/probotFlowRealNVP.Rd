\name{probotFlowRealNVP}
\alias{probotFlowRealNVP}
\title{Construct a Real Non-Volume Preserving (RealNVP) Normalizing Flow}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a conditional
normalizing flow built from a stack of affine RealNVP layers.
This is the \code{style = "realnvp"} architecture of
\code{\link{probotMakeFlow}}. Each layer uses a half of the parameter
vector, combined with the context, to predict an affine transformation
of the other half; the halves are swapped after each layer so that all
parameters are mixed across the stack.
}
\usage{
probotFlowRealNVP(
  input_dim,
  output_dim,
  n_layers = 4,
  hidden_dim = 32,
  soft_clamp = 3,
  device = NULL
)
}
\arguments{
    \item{input_dim}{Dimensionality of the conditioning context features.}

    \item{output_dim}{Total dimensionality of the parameter space. Must be
    at least 2.}

    \item{n_layers}{Number of RealNVP layers to stack. Defaults to 4.}

    \item{hidden_dim}{Number of units in each hidden layer of the internal
    shift/scale network within each RealNVP layer. Defaults to 32.}

    \item{soft_clamp}{Upper bound on the magnitude of the log-scale. The
    network output is clamped via
    \eqn{soft\_clamp \cdot \tanh(\cdot/soft\_clamp)} so the scale stays in
    \eqn{[\exp(-soft\_clamp),\, \exp(soft\_clamp)]}, trading expressiveness
    for numerical stability. Defaults to 3; values between 1 and 5 are
    typical.}

    \item{device}{Character specifying the computing device: \code{"cpu"},
    \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), auto-detects
    MPS, falling back to CPU.}
}
\details{
The returned module exposes:
\itemize{
  \item \code{forward(theta, x)}: maps parameters \eqn{\theta} to the latent
  space \eqn{z}, returning a list with \code{z} and the accumulated
  \code{log_det_jac} (the \eqn{\log|\det J|} used by
  \code{\link{probotLossNF}}).
  \item \code{inverse(z, x)}: maps a base-distribution draw \eqn{z \sim
  N(0, I)} back to parameter space \eqn{\theta}; used by
  \code{\link{probotSamplePostNF}}. Layers are applied in reverse order.
}

The building block is the private \code{.probotRealNVPLayer()} nn_module:
it splits \eqn{\theta} into halves of widths \eqn{d_1} (the floor of
\code{output_dim}/2) and \eqn{d_2 = output_dim - d_1}, predicts a shift and a
log-scale for the second half from \code{(theta_1, x)} through a two-hidden
GeLU MLP, applies \eqn{z_2 = (\theta_2 - t)\exp(s)}, and swaps the two halves
so the next layer transforms the complementary subset. Because each layer
moves only about half the coordinates, \code{n_layers} layers are required
before every coordinate has interacted with the rest; the inverse pass is
fully parallel over coordinates and cheap to evaluate.
}
\value{
An instantiated \code{\link[torch]{nn_module}}.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeFlow}}, \code{\link{probotFlowMAF}},
\code{\link{probotLossNF}}, \code{\link{probotTrainFlow}},
\code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Build a 4-layer RealNVP flow for 2 parameters conditioned on 3 features.
## The call returns an instantiated module (no extra parentheses needed).
flow <- probotFlowRealNVP(input_dim = 3, output_dim = 2, device = "cpu")

theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

out <- flow$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
