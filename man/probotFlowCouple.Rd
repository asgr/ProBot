\name{probotFlowCouple}
\alias{probotFlowCouple}
\title{Construct a Coupling (Affine NVP) Normalizing Flow}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a conditional
normalizing flow built from a stack of affine coupling (NVP) layers.
This is the \code{style = "couple"} architecture of
\code{\link{probotMakeFlow}}. Each layer uses a half of the parameter
vector, combined with the context, to predict an affine transformation
of the other half; the halves are swapped after each layer so that all
parameters are mixed across the stack.
}
\usage{
probotFlowCouple(
  dim_theta,
  dim_x,
  n_layers = 4,
  hidden_dim = 32,
  soft_clamp = 3,
  device = NULL
)
}
\arguments{
    \item{dim_theta}{Total dimensionality of the parameter space. Must be
    at least 2.}

    \item{dim_x}{Dimensionality of the context/input features used to
    condition the flow.}

    \item{n_layers}{Number of coupling layers to stack. Defaults to 4.}

    \item{hidden_dim}{Number of units in each hidden layer of the internal
    shift/scale network within each coupling layer. Defaults to 32.}

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

The building block is the private \code{.probotCouplingLayer()} nn_module:
it splits \eqn{\theta} into halves of widths \eqn{d_1} (the floor of
\code{dim_theta}/2) and \eqn{d_2 = dim_theta - d_1}, predicts a shift and a
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
\code{\link{probotMakeFlow}}, \code{\link{probotFlowAutoReg}},
\code{\link{probotLossNF}}, \code{\link{probotTrainFlow}},
\code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Build a 4-layer coupling flow for 2 parameters conditioned on 3 features.
## The call returns an instantiated module (no extra parentheses needed).
flow <- probotFlowCouple(dim_theta = 2, dim_x = 3, device = "cpu")

theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

out <- flow$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
