\name{probotPredictMDN}
\alias{probotPredictMDN}
\title{Run an MDN Model and Unpack Mixture Parameters}
\description{
Run a trained Mixture Density Network on new input data and return unpacked
mixture parameters.
}
\usage{
probotPredictMDN(input, model, mdn_components)
}
\arguments{
  \item{input}{Numeric matrix of predictors with rows as observations. In this context \option{input} needs to be scaled (e.g. with \code{\link{probotScaleForward}}).}

  \item{model}{Instantiated torch module returned by \code{probotMakeMDN(... )()}.}

  \item{mdn_components}{Number of Gaussian mixture components in \code{model}.}
}
\details{
Prediction runs in evaluation mode and without gradient tracking.
}
\value{
A list with elements:
\itemize{
  \item \code{mu}: tensor of component means.
  \item \code{log10_sigma}: tensor of component \code{log10} standard deviations.
  \item \code{logits}: tensor of unnormalized mixture logits.
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeMDN}}, \code{\link{probotSamplePostMDN}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  model <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 3)()
  x <- matrix(rnorm(12), ncol = 2)

  pred <- probotPredictMDN(x, model, mdn_components = 3)
  pred$mu$size()
}
}
