\name{probotFlowLoc}
\alias{probotFlowLoc}
\title{Wrap a Normalizing Flow in a Residual Location Head}
\description{
Compose a trained-ready flow with an explicit regression network so that the
network's output is the posterior \emph{location} and the flow models only the
residual around it. The result is an \code{\link[torch]{nn_module}} with the
same \code{forward(theta, x)} / \code{inverse(z, x)} contract as the flow
architectures in \code{\link{probotMakeFlow}}, so it trains, samples, saves,
loads and scores exactly like them.
}
\usage{
probotFlowLoc(base_flow, input_dim, output_dim,
  loc_hidden_dims = c(128, 128), device = NULL)
}
\arguments{
  \item{base_flow}{An instantiated flow module, e.g. the result of
    \code{probotMakeFlow(style = "nsf")()}.}
  \item{input_dim}{Dimensionality of the context/input features.}
  \item{output_dim}{Dimensionality of the parameter space.}
  \item{loc_hidden_dims}{Numeric vector of hidden widths for the head's MLP.
    Each hidden layer is \code{\link[torch]{nn_linear}} followed by GELU; the
    final layer emits \code{output_dim} values with no activation, so the head
    is an unbounded affine map onto the parameter space.}
  \item{device}{Character device, or \code{NULL} to auto-detect MPS, falling
    back to CPU.}
}
\details{
\code{probotFlowLoc} implements
\deqn{\theta = \mu(x) + f^{-1}(z, x), \qquad z \sim N(0, I)}
where \eqn{\mu} is the head and \eqn{f^{-1}} is the wrapped flow's inverse map.
Sampling draws \eqn{z}, applies \eqn{f^{-1}}, then adds \eqn{\mu(x)}. The
forward map is \eqn{z = f(\theta - \mu(x), x)}.

\subsection{Why the change of variables costs nothing}{
Because the shift depends on \eqn{x} and not on \eqn{\theta},
\eqn{\partial(\theta - \mu(x)) / \partial\theta} is the identity. The base
flow's reported \code{log_det_jac} is therefore already
\eqn{\log|\det \partial z / \partial\theta|} exactly: no correction term, no
new base distribution, and the negative log-likelihood computed by
\code{\link{probotLossNF}} is unchanged and remains exact. Verified against a
full autodiff Jacobian for all three styles to about \eqn{10^{-15}} nats in
double precision (\code{tests/testthat/test-probotFlowLoc.R}).
}

\subsection{What the head buys, and what it does not}{
The head adds \emph{credit assignment}, not expressiveness. Every conditioner
in \code{\link{probotFlowRealNVP}}, \code{\link{probotFlowMAF}} and
\code{\link{probotFlowNSF}} already receives \eqn{x}, so an unheaded flow can in
principle represent any conditional shift; what it lacks is a set of parameters
whose only job is that shift, and hence a gradient that lands directly on it.

That distinction matters for how the model is trained. With the point-estimate
loss blend (\code{lambda > 0} in \code{\link{probotTrainFlow}}) the MSE term
supervises \eqn{\mu(x)} through a short, dedicated path, while the NLL term
still shapes the whole density. On a plain flow the same blend has to push
through the shared conditioner, where the NLL can quietly undo what MSE
installs. \code{point = "loc"} also makes the blend cheaper and noise-free: the
target is the head's own output rather than an inverse sweep, so it avoids both
the \code{center}-estimate bias on skewed posteriors and the Monte-Carlo noise
of \code{point = "mean"}.

Under pure maximum likelihood (\code{lambda = 0}) the split between head and
flow is chosen arbitrarily by the optimiser, so the head is expected to change
very little; it should not be added on its own.
}

\subsection{Reading the point estimate}{
\code{probotSamplePostNF(..., point_estimate = TRUE)} returns \eqn{\mu(x)} for a
headed flow and the \eqn{z = 0} inverse for a plain one, in both cases without
drawing samples. \code{\link{probotPIT}}, \code{\link{probotCRPS}} and
\code{\link{probotTARP}} recognise the wrapper and dispatch to the flow sampler
automatically.
}
}
\section{Fields}{
\describe{
  \item{\code{location(x)}}{Returns \eqn{\mu(x)} as a tensor of shape
    (batch, output_dim). Differentiable, and the target used when
    \code{point = "loc"}.}
  \item{\code{base_flow}}{The wrapped flow, reachable for inspection. Its
    state-dict keys are namespaced under \code{base_flow.}, with the head under
    \code{loc_layers.}.}
}
}
\seealso{
\code{\link{probotMakeFlow}} (pass \code{loc_head = TRUE}),
\code{\link{probotTrainFlow}} (\code{point = "loc"}),
\code{\link{probotSamplePostNF}} (\code{point_estimate})
}
\examples{
\dontrun{
mdl <- probotMakeFlow(input_dim = 63, output_dim = 9, style = "nsf",
                      n_layers = 9, hidden_dim = 512,
                      loc_head = TRUE, loc_hidden_dims = c(416, 832, 416))()
res <- probotTrainFlow(mdl, dl, opt, epochs = 60,
                       lambda = 0.3, point = "loc")

mu <- probotSamplePostNF(test_input, res$model, output_dim = 9,
                         point_estimate = TRUE)
}
}
\keyword{models}
