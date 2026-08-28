\name{probotFlowAutoReg}
\alias{probotFlowAutoReg}
\title{Construct an Autoregressive (MAF) Normalizing Flow}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a conditional
normalizing flow built from masked autoregressive (MAF) blocks separated
by permutation layers. This is the \code{style = "autoreg"} architecture of
\code{\link{probotMakeFlow}}. Each block lets every parameter's affine
transformation depend on all parameters ordered before it, so the flow
factorises the full joint density as a product of conditional
distributions.
}
\usage{
probotFlowAutoReg(
  dim_theta,
  dim_x,
  n_blocks = 5,
  n_layers_per_block = 2,
  hidden_dim = 500,
  soft_clamp = 3,
  device = NULL
)
}
\arguments{
    \item{dim_theta}{Total dimensionality of the parameter space.}

    \item{dim_x}{Dimensionality of the context/input features used to
    condition the flow.}

    \item{n_blocks}{Number of masked autoregressive blocks to stack.
    Defaults to 5.}

    \item{n_layers_per_block}{Number of masked linear (MADE) layers inside
    each block. Defaults to 2.}

    \item{hidden_dim}{Number of units in each hidden layer of the masked
    MLP. Defaults to 500.}

    \item{soft_clamp}{Upper bound on the magnitude of the log-scale, applied
    via \eqn{soft\_clamp \cdot \tanh(\cdot/soft\_clamp)}. Defaults to 3;
    values between 1 and 5 are typical.}

    \item{device}{Character specifying the computing device: \code{"cpu"},
    \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), auto-detects
    MPS, falling back to CPU.}
}
\details{
The returned module exposes the same interface as
\code{\link{probotMakeFlowCouple}}: \code{forward(theta, x)} returns a list
with \code{z} and \code{log_det_jac}, and \code{inverse(z, x)} maps a base
draw \eqn{z \sim N(0, I)} back to parameter space.

Inside, each block is a masked autoregressive flow:
\itemize{
  \item A masked MLP (MADE) whose weights are zeroed so that the affine
  transformation of parameter \eqn{j} depends only on parameters
  \eqn{1, \dots, j-1} and the context \eqn{x}, never on itself. This
  makes the Jacobian triangular, so its determinant is simply the product of
  the scales and the \code{log_det_jac} is the sum of the clamped log-scales.
  \item A permutation layer (a column reversal) applied between blocks so
  that every parameter plays the "conditioned" role in at least one block,
  improving mixing across the stack.
}

The inverse pass reconstructs \eqn{\theta} coordinate by coordinate: because
of the autoregressive mask, the transformation for coordinate \eqn{j} is
known as soon as coordinates \eqn{1, \dots, j-1} are reconstructed, so
sampling is exact. Unlike the coupling architecture, this reconstruction is
\emph{sequential} in \code{dim_theta} and calls the masked MLP
\code{dim_theta} times per block, which is the main reason autoregressive
sampling is slower than coupling sampling (see
\code{\link{probotMakeFlow}} for a comparison).
}
\value{
An instantiated \code{\link[torch]{nn_module}}.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeFlow}}, \code{\link{probotMakeFlowCouple}},
\code{\link{probotLossNF}}, \code{\link{probotTrainFlow}},
\code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Build an autoregressive flow for 3 parameters conditioned on 2 features.
## The call returns an instantiated module (no extra parentheses needed).
flow <- probotFlowAutoReg(dim_theta = 3, dim_x = 2, device = "cpu")

theta <- torch::torch_randn(c(10, 3))
x <- torch::torch_randn(c(10, 2))

out <- flow$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
