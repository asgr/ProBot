\name{probotPost}
\alias{probotSamplePostMDN}
\alias{probotMarginalPostMDN}
\title{Sample and Summarize Posteriors from MDN Outputs}
\description{
Generate posterior draws from MDN mixture outputs or compute marginal posterior
means and standard deviations.
}
\usage{
probotSamplePostMDN(
  mdn_output,
  index = 1,
  n_samples = 5000,
  col_means,
  col_sds,
  col_names = NULL
)

probotMarginalPostMDN(mdn_output, col_means, col_sds, col_names = NULL)
}
\arguments{
  \item{mdn_output}{List returned by \code{probotPredictMDN()}.}

  \item{index}{Row index to sample from in \code{mdn_output}.}

  \item{n_samples}{Number of posterior samples to generate.}

  \item{col_means}{Centering values used for output scaling.}

  \item{col_sds}{Scale values used for output scaling.}

  \item{col_names}{Optional names to assign to returned columns.}
}
\details{
\code{probotSamplePostMDN()} samples component labels from softmax weights,
then draws Gaussian samples per component and back-transforms to original
units. \code{probotMarginalPostMDN()} computes mixture-weighted posterior means
and standard deviations and back-transforms both.
}
\value{
\itemize{
  \item \code{probotSamplePostMDN()}: Numeric matrix with \code{n_samples} rows.
  \item \code{probotMarginalPostMDN()}: List with \code{post_mean} and
  \code{post_sd} matrices.
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotPredictMDN}}, \code{\link{probotScaleBackward}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(1)
  torch::torch_manual_seed(1)

  model <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2)()
  x <- matrix(rnorm(8), ncol = 2)
  pred <- probotPredictMDN(x, model, mdn_components = 2)

  draws <- probotSamplePostMDN(
    mdn_output = pred,
    index = 1,
    n_samples = 20,
    col_means = 0,
    col_sds = 1,
    col_names = "theta"
  )

  marg <- probotMarginalPostMDN(
    mdn_output = pred,
    col_means = 0,
    col_sds = 1,
    col_names = "theta"
  )

  dim(draws)
  dim(marg$post_mean)
}
}
