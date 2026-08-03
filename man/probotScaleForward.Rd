\name{probotScale}
\alias{probotScale}
\alias{probotScaleForward}
\alias{probotScaleBackward}
\title{Standardize and Back-Transform Matrix Columns}
\description{
Scale matrix columns to zero mean and unit variance, then invert the transform
later using stored or supplied scaling statistics.
}
\usage{
probotScaleForward(input)

probotScaleBackward(input, col_means = NULL, col_sds = NULL)
}
\arguments{
  \item{input}{Numeric matrix to scale or back-transform.}

  \item{col_means}{Optional vector (or scalar recycled across columns) of means
  used for centering.}

  \item{col_sds}{Optional vector (or scalar recycled across columns) of standard
  deviations used for scaling.}
}
\details{
\code{probotScaleForward()} stores \code{col_means} and \code{col_sds} in
\code{attr(x, "col_scale")}. \code{probotScaleBackward()} first uses explicit
arguments when provided; otherwise it reads the values from this attribute.
}
\value{
A numeric matrix with transformed values. Forward output includes a
\code{col_scale} attribute.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotSamplePostMDN}}, \code{\link{probotMarginalPostMDN}}
}
\examples{
x <- matrix(1:12, ncol = 3)

scaled <- probotScaleForward(x)
restored <- probotScaleBackward(scaled)

all.equal(unname(restored), unname(x))
}
