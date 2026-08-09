\name{probotIO}
\alias{probotSave}
\alias{probotLoadModel}
\alias{probotLoad}
\title{Save and Load Model Checkpoints}
\description{
Save trained models to disk with metadata, and load them back with full
reconstruction or into an explicit model skeleton.
}
\usage{
probotSave(
  model,
  optimizer = NULL,
  filename = "temp.pt",
  mdn_components = NULL,
  input_dim = NULL,
  output_dim = NULL,
  hidden_dims = NULL,
  activation_name = NULL,
  dropout = NULL,
  col_means = NULL,
  col_sds = NULL,
  training_history = NULL,
  ...
)

probotLoadModel(
  filename,
  model_skeleton = NULL,
  load_optimizer = FALSE,
  device = "cpu"
)

probotLoad(
  filename,
  load_optimizer = FALSE,
  device = "cpu"
)
}
\arguments{
    \item{model}{An nn_module object (MDN model).}

    \item{optimizer}{Optional optimizer state dict to save alongside the model.}

    \item{filename}{Path to save or load the checkpoint file.}

    \item{mdn_components}{Number of MDN mixture components; saved for later reconstruction.}

    \item{input_dim}{Dimension of the input layer; saved for later reconstruction.}

    \item{output_dim}{Dimension of the output layer; saved for later reconstruction.}

    \item{hidden_dims}{Numeric vector of hidden layer sizes; optional metadata for reconstruction.}

    \item{activation_name}{Character name of activation function (\code{"relu"},
     \code{"silu"}, or \code{"gelu"}).}

    \item{dropout}{Dropout rate used in the model; saved as metadata.}

    \item{col_means}{Centering values from \code{\link{probotScaleForward}}(); saved as metadata.}

    \item{col_sds}{Scaling values from \code{\link{probotScaleForward}}(); saved as metadata.}

    \item{training_history}{Training history data frame; saved as metadata.}

    \item{...}{Additional fields to include in the checkpoint metadata block.}

    \item{model_skeleton}{Pre-instantiated model object to load state dict into.
     If \code{NULL}, reconstruction from metadata is required (see \code{\link[=probotLoad]{probotLoad()}}).}

    \item{load_optimizer}{Logical; if \code{TRUE}, also restore the optimizer state.}

    \item{device}{Character string specifying the computing device for loading
     tensors: \code{"cpu"} (default), \code{"cuda"}, or \code{"mps"} for Apple
     Silicon GPUs. Checkpoint tensors are loaded onto this device.}
}
\details{
\code{probotSave()} stores model weights, optimizer state, and metadata in a
single \code{.pt} file. When called without required metadata
(\code{mdn_components}, \code{input_dim}, \code{output_dim}), it issues a
warning that \code{\link[=probotLoad]{probotLoad()}} will be unable to auto-reconstruct
the model.

\code{probotLoadModel()} loads a checkpoint and requires an explicit
\code{model_skeleton} for state restoration, unless the checkpoint was saved
with full metadata (in which case \code{\link[=probotLoad]{probotLoad()}} should be used).

\code{probotLoad()} automatically reconstructs the model architecture from saved
metadata and loads weights. It requires that \code{mdn_components},
\code{input_dim}, and \code{output_dim} were saved in the checkpoint. If these
are missing, use \code{\link[=probotLoadModel]{probotLoadModel()}} with an explicit model skeleton instead.
}
\value{
\code{probotSave()} returns the filename invisibly.

\code{probotLoadModel()} and \code{probotLoad()} return a list containing
\code{model}, \code{optimizer} (if requested), and \code{metadata}.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeMDN}}, \code{\link{probotTrainMDN}}
}
\examples{
\dontrun{
## Save a trained model
probotSave(
  fitted$model,
  optimizer = fitted$optimizer,
  filename = "my_model.pt",
  mdn_components = 3,
  input_dim = 2,
  output_dim = 2,
  col_means = c(0, 0),
  col_sds = c(1, 1)
)

## Load and auto-reconstruct
loaded <- probotLoad("my_model.pt")

## Load onto GPU/MPS
loaded_gpu <- probotLoad("my_model.pt", device = "mps")
}
}
