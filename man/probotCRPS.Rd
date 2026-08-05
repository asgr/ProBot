\name{probotCRPS}
\alias{probotCRPS}
\title{Continuous Ranked Probability Score (CRPS)}
\description{
Compute the CRPS for each parameter at each test observation, using
posterior samples drawn from a Mixture Density Network (MDN). Lower
values indicate better-calibrated predictive distributions.
}
\usage{
probotCRPS(
  mdn_output,
  params,
  n_test = NULL,
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

  \item{n_test}{Number of CRPS values to compute (i.e., params[1:n_test, ].
    Defaults to all rows and cannot exceed nrow(params).}

  \item{n_samples}{Number of posterior samples to draw per observation.
    Defaults to \code{1e4}.}

  \item{col_means}{Centering values used for back-transforming MDN output,
    as returned by \code{probotScaleForward()}.}

  \item{col_sds}{Scale values used for back-transforming MDN output,
    as returned by \code{probotScaleForward()}.}

  \item{col_names}{Optional character vector of parameter names to assign
    as column names of the returned matrix.}

  \item{verbose}{Logical. If \code{TRUE} (default), prints progress every
    1000 observations.}
}
\details{
For each test observation \eqn{i} and parameter dimension \eqn{j}, CRPS is
computed from \eqn{S} = \code{n_samples} posterior samples \eqn{z^{(1)},
\dots, z^{(S)}} and the true value \eqn{\theta_{ij}}:

\deqn{\text{CRPS}_{ij} = \frac{1}{S} \sum_{s=1}^{S} |\theta_{ij} - z^{(s)}| - \frac{1}{2 S^2} \sum_{s,t=1}^{S} |z^{(s)} - z^{(t)}|}

The second term is the spread penalty and is efficiently computed using the
sorted samples. Under a well-calibrated posterior with appropriate sharpness,
CRPS values should be low (with no lower bound other than zero).
}
\value{
A numeric matrix of shape \eqn{n_\text{test} \times n_\text{params}}
containing the CRPS value for every (observation, parameter) pair. Column
names are set to \code{col_names} when provided.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotSamplePostMDN}}, \code{\link{probotPredictMDN}},
\code{\link{probotPIT}}, \code{\link{probotTARP}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(42)
  torch::torch_manual_seed(42)

  model <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3)()
  x <- matrix(rnorm(20), ncol = 2)
  pred <- probotPredictMDN(x, model, mdn_components = 3)

  # True parameter values (10 observations x 2 parameters)
  params <- matrix(rnorm(20), nrow = 10, ncol = 2)

  crps <- probotCRPS(
    mdn_output = pred,
    params     = params,
    n_samples  = 100,
    col_means  = c(0, 0),
    col_sds    = c(1, 1),
    col_names  = c("param1", "param2"),
    verbose    = FALSE
   )

  dim(crps)
  head(crps)
}
}
