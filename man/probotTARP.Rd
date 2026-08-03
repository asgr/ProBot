\name{probotTARP}
\alias{probotTARP}
\title{Test of Accuracy with Random Points (TARP) Assessment}
\description{
Compute TARP coverage values for a set of test observations by projecting
both posterior samples and the true parameter vector onto a random unit
direction and measuring how often the projection of the truth falls below
the projected posterior samples.
}
\usage{
probotTARP(
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

  \item{col_names}{Optional character vector of parameter names passed
    through to \code{probotSamplePostMDN()}.}

  \item{verbose}{Logical. If \code{TRUE} (default), prints progress to the
    console every 1000 observations.}
}
\details{
For each test observation \eqn{i} a unit-length direction vector
\eqn{\mathbf{d}_i \in \mathbb{R}^{n_\text{params}}} is drawn uniformly at
random by normalising an isotropic Gaussian draw. The posterior samples and
the true parameters are then projected onto this direction:

\deqn{p^{(s)}_i = \tilde{\boldsymbol\theta}_i^{(s)\top}\mathbf{d}_i, \quad
      t_i = \boldsymbol\theta_i^\top\mathbf{d}_i}

and the TARP value is

\deqn{\text{TARP}_i = \frac{1}{S}\sum_{s=1}^{S}
      \mathbf{1}\!\left[p^{(s)}_i \le t_i\right].}

Under a well-calibrated posterior the \eqn{\text{TARP}_i} values should be
uniformly distributed on \eqn{[0, 1]}.
}
\value{
A numeric vector of length \eqn{n_\text{test}} containing the TARP
coverage value for each test observation.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotSamplePostMDN}}, \code{\link{probotPredictMDN}},
\code{\link{probotPIT}}
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

  tarp <- probotTARP(
    mdn_output = pred,
    params     = params,
    n_samples  = 100,
    col_means  = c(0, 0),
    col_sds    = c(1, 1),
    verbose    = FALSE
  )

  length(tarp)
  hist(tarp, breaks = 10, main = "TARP coverage")
}
}
