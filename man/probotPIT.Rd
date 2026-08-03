\name{probotPIT}
\alias{probotPIT}
\title{Probability Integral Transform (PIT) Assessment}
\description{
Compute per-parameter PIT values for a set of test observations by comparing
true parameter values against the empirical CDF of posterior samples drawn
from a Mixture Density Network (MDN).
}
\usage{
probotPIT(
  mdn_output,
  params,
  n_samples = 1e4,
  col_means,
  col_sds,
  col_names = NULL,
  verbose = TRUE
)
}
\arguments{
  \item{mdn_output}{List returned by \code{probotPredictMDN()}, containing
    \code{logits}, \code{mu}, and \code{log10_sigma} tensors.}

  \item{params}{Numeric matrix of shape \eqn{n_\text{test} \times n_\text{params}}
    containing the true parameter values for each test observation.}

  \item{n_samples}{Number of posterior samples to draw per observation.
    Defaults to \code{1e4}.}

  \item{col_means}{Centering values used for back-transforming MDN output,
    as returned by \code{probotScaleForward()}.}

  \item{col_sds}{Scale values used for back-transforming MDN output,
    as returned by \code{probotScaleForward()}.}

  \item{col_names}{Optional character vector of parameter names to assign
    as column names of the returned matrix.}

  \item{verbose}{Logical. If \code{TRUE} (default), prints progress to the
    console every 1000 observations.}
}
\details{
For each test observation \eqn{i} and each parameter dimension \eqn{j},
the PIT value is defined as the fraction of posterior samples whose
\eqn{j}-th coordinate is less than or equal to the true value
\eqn{\theta_{ij}}:

\deqn{\text{PIT}_{ij} = \frac{1}{S} \sum_{s=1}^{S} \mathbf{1}\!\left[\tilde\theta_{ij}^{(s)} \le \theta_{ij}\right]}

where \eqn{S} = \code{n_samples}. Under a well-calibrated posterior the
PIT values for each parameter should be uniformly distributed on
\eqn{[0, 1]}.
}
\value{
A numeric matrix of shape \eqn{n_\text{test} \times n_\text{params}}
containing the PIT value for every (observation, parameter) pair. Column
names are set to \code{col_names} when provided.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotSamplePostMDN}}, \code{\link{probotPredictMDN}},
\code{\link{probotTARP}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(42)
  torch::torch_manual_seed(42)

  model <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3)()
  x     <- matrix(rnorm(20), ncol = 2)
  pred  <- probotPredictMDN(x, model, n_components = 3)

  # True parameter values (10 observations x 2 parameters)
  params <- matrix(rnorm(20), nrow = 10, ncol = 2)

  pit <- probotPIT(
    mdn_output = pred,
    params     = params,
    n_samples  = 100,
    col_means  = c(0, 0),
    col_sds    = c(1, 1),
    col_names  = c("param1", "param2"),
    verbose    = FALSE
  )

  dim(pit)
  head(pit)
}
}
