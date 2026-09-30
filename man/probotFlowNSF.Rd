\name{probotFlowNSF}
\alias{probotFlowNSF}
\title{Construct a Neural Spline Normalizing Flow}
\description{
Construct a conditional neural spline flow using rational-quadratic spline
coupling layers. This is the \code{style = "nsf"} architecture of
\code{\link{probotMakeFlow}}.
}
\usage{
probotFlowNSF(
  input_dim, output_dim, n_layers = 4, hidden_dim = 32,
  n_bins = 8, tail_bound = 3, device = NULL
)
}
\arguments{
\item{input_dim}{Dimensionality of the conditioning features.}
\item{output_dim}{Total dimensionality of the parameter space; must be at least 2.}
\item{n_layers}{Number of spline coupling layers.}
\item{hidden_dim}{Width of each conditioner network's hidden layers.}
\item{n_bins}{Number of rational-quadratic bins per transformed coordinate.}
\item{tail_bound}{The spline is learned on \code{[-tail_bound, tail_bound]} and
is the identity outside this interval.}
\item{device}{Character specifying the computing device, or \code{NULL} for
automatic selection.}
}
\details{
Each layer transforms one half of the parameter vector using monotonic
rational-quadratic splines whose widths, heights, and derivatives are predicted
from the other half and the conditioning features. The halves are swapped after
each layer. Consequently both density evaluation and sampling are parallel over
coordinates, while the nonlinear spline transform is more flexible than affine
RealNVP.
}
\value{An instantiated \code{\link[torch]{nn_module}}.}
\seealso{\code{\link{probotMakeFlow}}, \code{\link{probotFlowRealNVP}},
\code{\link{probotFlowMAF}}, \code{\link{probotTrainFlow}}.}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  flow <- probotFlowNSF(input_dim = 3, output_dim = 2, device = "cpu")
  out <- flow$forward(torch::torch_randn(c(10, 2)),
                      torch::torch_randn(c(10, 3)))
  names(out)
}
}
