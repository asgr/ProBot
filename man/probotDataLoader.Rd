\name{probotDataLoader}
\alias{probotDataLoader}
\title{Create a Torch DataLoader from Input and Output Matrices}
\description{
Convert numeric input/output matrices into a \code{torch} dataset and data loader
for mini-batch training.
}
\usage{
probotDataLoader(
  input,
  output,
  batch = 1024,
  shuffle = TRUE,
  train_idx = NULL,
  device = NULL
)
}
\arguments{
   \item{input}{Numeric matrix of predictors with rows as training examples.}

   \item{output}{Numeric matrix of targets with rows aligned to \code{input}.}

   \item{batch}{Mini-batch size used by the returned data loader.}

   \item{shuffle}{Logical; if \code{TRUE}, batches are shuffled each epoch.}

   \item{train_idx}{Optional integer indices selecting rows to include in the loader.}

   \item{device}{Character string specifying the computing device for tensor allocation: \code{"cpu"} (default), \code{"cuda"}, or \code{"mps"} for Apple Silicon GPUs. Tensors are created directly on this device.}
}
\details{
If \code{train_idx} is \code{NULL}, all rows are used.
}
\value{
A \code{torch::dataloader} object yielding batches with \code{$x} and \code{$y}
tensors.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotSingleEpochMDN}}, \code{\link{probotTrainMDN}}
}
\examples{
set.seed(1)

x <- matrix(rnorm(20), ncol = 2)
y <- matrix(rnorm(10), ncol = 1)

loader <- probotDataLoader(x, y, batch = 4, shuffle = FALSE)

loader$batch_size
}
