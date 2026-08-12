\name{probotPredictPointScaled}
\alias{probotPredictPointScaled}
\title{Predict with Point Estimate Model and Unscale Output}
\description{
Generate point predictions and unscale them back to physical units using column means and standard deviations.
}
\usage{
probotPredictPointScaled(input, model, col_means, col_sds, device = NULL)
}
\arguments{
  \item{input}{Numeric matrix of input features.}
  
  \item{model}{Trained point estimate \code{\link[torch]{nn_module}}.}
  
  \item{col_means}{Column means used for scaling during training.}
  
  \item{col_sds}{Column standard deviations used for scaling during training.}
  
  \item{device}{Device override. If NULL, inferred from model.}
}
\value{
Numeric matrix of predictions in original scale.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotPredictPoint}}, \code{\link{probotScaleBackward}}
}
