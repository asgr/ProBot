\name{probotSigmaPostNF}
\alias{probotSigmaPostNF}
\title{Approximate Posterior Mean and SD from a Normalizing Flow}
\description{
Fast approximate posterior summaries for a trained normalizing flow, computed
by passing a small deterministic grid of points through the inverse map rather
than by drawing samples. Returns one location and one scale per parameter per
observation, in the shape of \code{\link{probotMarginalPostMDN}}, at a cost of
\eqn{2D + 1}{2D + 1} inverse rows per observation instead of \code{n_samples}.
}
\usage{
probotSigmaPostNF(
  input,
  model,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  output_dim = NULL,
  device = NULL,
  eps = 1e-3,
  batch_size = NULL,
  verbose = FALSE
)
}
\arguments{
  \item{input}{Numeric vector (single observation) or matrix (rows are
   observations), or a torch tensor. Converted and placed on the model's device
   automatically.}

  \item{model}{An instantiated normalizing flow \code{\link[torch]{nn_module}}
   (e.g., from \code{\link{probotMakeFlow}}). Set to evaluation mode
   automatically. A \code{\link{probotFlowLoc}} model works too; see
   \strong{Relation to point_estimate}.}

  \item{col_means}{Numeric vector of column means for inverse scaling
   (unstandardizing). If provided, the location and scale are returned in
   original parameter units.}

  \item{col_sds}{Numeric vector of column standard deviations for inverse
   scaling. Required whenever \code{col_means} is given.}

  \item{col_names}{Character vector of parameter names, used as column names.}

  \item{output_dim}{Integer parameter dimensionality. Only needed when
   \code{col_means} is \code{NULL}; if both are \code{NULL} the function errors.}

  \item{device}{Torch device (e.g. \code{"cpu"}, \code{"mps"}). \code{NULL}
   auto-detects from the model's parameters.}

  \item{eps}{Finite-difference step, in base-space units, used to probe the
   Jacobian. A single positive number; defaults to \code{1e-3}. Smaller is not
   better: the flow's inverse is evaluated in float32, and results were stable
   to within \eqn{4 \times 10^{-4}}{4e-4} relative over \code{eps} from
   \code{1e-4} to \code{0.1}.}

  \item{batch_size}{Number of observations processed per chunk. \code{NULL}
   chooses a size from a memory budget, exactly as
   \code{\link{probotSamplePostNF}} does, but scaled by \code{2 * output_dim + 1}
   rows rather than \code{n_samples}. Unlike the sampler, the chunk size
   \emph{cannot} change the result: the probe grid is deterministic, so any
   \code{batch_size} gives the same answer up to float32 non-associativity.
   An explicit value above the budget is reduced with a warning.}

  \item{verbose}{Logical; print chunk progress at most about twenty times.}
}
\details{
\strong{What is computed.} A flow maps base draws \eqn{z \sim N(0, I)}{z ~ N(0, I)}
to parameters by the inverse transform \eqn{\theta = f^{-1}(z, x)}{theta = f^(-1)(z, x)}.
Linearising that map at \eqn{z = 0}{z = 0} gives a Gaussian for \eqn{\theta}
with location \eqn{\theta_0 = f^{-1}(0, x)} and covariance
\deqn{\Sigma = J J^T, \quad J = \left. \frac{\partial \theta}{\partial z} \right|_{z = 0}}
so the marginal scale of parameter \eqn{j} is the norm of row \eqn{j} of the
Jacobian:
\deqn{\mathrm{sd}_j = \sqrt{\sum_k J_{jk}^2}.}
\eqn{J} is obtained by finite differences -- displace base axis \eqn{k} by
\eqn{\pm eps}, invert, and divide the difference by \eqn{2 eps}. All rows for
all observations are stacked into one batched inverse sweep.

\strong{What it is not.} This is a first-order, local Gaussian approximation of
the conditional posterior. It carries no information about skewness or
multimodality, and its location is the base-distribution \emph{mode} mapped
forward, not the posterior mean. Measured against 4000-sample marginal SDs on
RealNVP, MAF and NSF flows trained on a heteroscedastic three-parameter problem
and scored on fresh in-distribution contexts, the delta-to-sampling ratio was
0.95--1.01 for RealNVP, 1.00 for MAF, and 0.77--0.98 for NSF -- the spline
inverse's piecewise-linear derivative is the least well captured by a single
local slope. The error is not bounded in general: on a trained nine-parameter
checkpoint evaluated at random \eqn{N(0, I)}{N(0, I)} contexts (which are far
off that model's training manifold) the median ratio fell to 0.54 with a
5--95 per cent range of 0.08--2.25. Use these values for ranking, screening and
plotting error bars; for anything calibrated, use
\code{\link{probotSamplePostNF}} and score it with \code{\link{probotPIT}},
\code{\link{probotCRPS}} or \code{\link{probotTARP}}.

\strong{Why base-space axes.} A tempting shortcut is to displace base space
along parameter \eqn{j}'s own axis \eqn{e_j} and read off \eqn{\theta_j}, which
is what a "sweep \code{chi} through the flow" naturally means. That is not the
marginal scale of \eqn{\theta_j}: it traces one coordinate of a curve through
parameter space while holding the other base coordinates fixed. Coupling
inverses pass half their coordinates through untouched and MAF inverses are
triangular, so \eqn{\partial \theta_j / \partial z_j} is frequently near zero
while \eqn{\partial \theta_j / \partial z_k} for \eqn{k \ne j} is large. On the
trained nine-parameter NSF above the diagonal of \eqn{J} ranged from 0.008 to
-0.21 against off-diagonal entries above 2, and an \eqn{e_j} sweep returned
half-widths of 0.000--0.21 where sampling gave 0.7--1.2. The row-norm formula
is used instead. At \eqn{D = 1} all conventions coincide.
}
\section{Relation to point_estimate}{
The \code{post_mean} column is exactly what
\code{probotSamplePostNF(..., point_estimate = TRUE)} returns for a plain flow
-- the inverse of \code{z = 0} -- and agrees with it to float32 rounding. That
argument is not the same thing on a \code{\link{probotFlowLoc}} model, where it
returns the head \eqn{\mu(x)} directly; \code{probotSigmaPostNF} always inverts
\eqn{z = 0} through the whole model, head included, so on a headed model the
two differ by the head's offset from the centre.
}
\value{
A list with components \code{post_mean} and \code{post_sd}, each a numeric
matrix of shape \eqn{Nrow \times D}{Nrow x D}, or length-\eqn{D} numeric vectors
when \code{input} is a vector. Column names are taken from \code{col_names}
when supplied. This mirrors \code{\link{probotMarginalPostMDN}}.
}
\references{
Aaron Robotham (2026). ProBot: Torch Based Tools.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotSamplePostNF}}, \code{\link{probotMarginalPostMDN}},
\code{\link{probotMakeFlow}}, \code{\link{probotScaleBackward}},
\code{\link{probotAssess}}
}
\examples{
\dontrun{
model <- probotMakeFlow(input_dim = 5, output_dim = 3, n_layers = 6,
                        hidden_dim = 64, style = "realnvp")()
# ... train with probotTrainFlow() ...

post <- probotSigmaPostNF(
  input     = test_inputs,
  model     = model,
  col_means = col_means,
  col_sds   = col_sds,
  col_names = c("A", "B", "C")
)
head(post$post_mean)
head(post$post_sd)

# 1-sigma intervals, same shape as the MDN marginal summary
upper <- post$post_mean + post$post_sd

# Compare against the calibrated route for a handful of rows
samp <- probotSamplePostNF(test_inputs[1:20, ], model, n_samples = 4000,
                           col_means = col_means, col_sds = col_sds,
                           col_names = c("A", "B", "C"))
apply(samp, c(2, 3), sd)[, 1:6]
post$post_sd[1:20, ][1:6, ]
}
}
\keyword{models}
