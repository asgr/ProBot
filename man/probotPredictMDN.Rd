\name{probotPredictMDN}
\alias{probotPredictMDN}
\title{Generate MDN Predictions from a Trained Model}
\description{
Pass input features through a trained MDN and unpack the raw network output into separate \code{logits}, \code{mu}, and \code{log10_sigma} tensors for downstream sampling or evaluation.
}
\usage{
probotPredictMDN(input, model, mdn_components, device = NULL)
}
\arguments{
  \item{input}{Numeric matrix of input features (rows = observations).}

  \item{model}{An instantiated MDN \code{\link[torch]{nn_module}} object.}

  \item{mdn_components}{Number of mixture components used when the model was trained.}

  \item{device}{Character string or \code{\link[torch]{torch_device}} object specifying
   the computing device: \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If \code{NULL}
   (default), the device is automatically inferred from \code{model}; if the model has
   no parameters, falls back to \code{"mps"} if available otherwise \code{"cpu"}.}
}
\details{
The model is set to evaluation mode (\code{model$eval}) and predictions
are computed inside a \code{\link[torch]{with_no_grad}} block. The raw network output
is unpacked by the internal \code{.probotUnpackMDN()} helper.

When \code{device = NULL} (default), the input tensor is automatically placed on the
same device as the model, eliminating device mismatch errors. Pass an explicit device
to override (e.g., to force CPU inference).

The returned tensors remain on the specified device. Convert to R matrices with
\code{as.numeric()} or \code{as.matrix()} for downstream use.
}
\value{
A list containing:
  \itemize{
    \item \code{logits}: Tensor of shape \eqn{n \times K} (mixture logits).
    \item \code{mu}: Tensor of shape \eqn{n \times K \times D} (component means).
    \item \code{log10_sigma}: Tensor of shape \eqn{n \times K \times D} (log10 component scales).
  }
  where \eqn{n} is the number of observations, \eqn{K} is \code{mdn_components},
  and \eqn{D} is \code{output_dim}.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeMDN}}, \code{\link{probotSamplePostMDN}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(42)
  torch::torch_manual_seed(42)

  model <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3)()
  x <- matrix(rnorm(20), ncol = 2)

   # Device auto-detected from model
  pred <- probotPredictMDN(x, model, mdn_components = 3)

  names(pred)
}
}
