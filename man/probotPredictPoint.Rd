\name{probotPredictPoint}
\alias{probotPredictPoint}
\title{Predict with Point Estimate Model}
\description{
Generate point predictions from a trained point estimate model.
}
\usage{
probotPredictPoint(input, model, device = NULL)
}
\arguments{
  \item{input}{Numeric matrix of input features.}
  
  \item{model}{Trained point estimate \code{\link[torch]{nn_module}}.}
  
  \item{device}{Device override. If NULL, inferred from model.}
}
\value{
Numeric matrix of predictions.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakePoint}}, \code{\link{probotTrainPoint}}
}
