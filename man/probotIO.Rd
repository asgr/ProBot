\name{probotSave}
\alias{probotSave}
\alias{probotLoadModel}
\alias{probotLoad}
\title{Save and Reload Mixture Density Network State}
\description{
Save model (and optional optimizer) state to disk, along with metadata needed
to fully reconstruct the model. Reload from a saved checkpoint with or without
an explicit model skeleton.
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
    device = NULL
)

probotLoad(filename, load_optimizer = FALSE, device = NULL)
}
\arguments{
  \item{model}{A torch module instance to save.}

  \item{optimizer}{Optional torch optimizer whose state should also be saved.}

  \item{filename}{Path to a \code{.pt} file used for save/load.}

  \item{mdn_components}{Number of mixture components in the MDN. Saved so the
    model can be reconstructed on load.}

  \item{input_dim}{Dimensionality of input features. Saved for reconstruction.}

  \item{output_dim}{Dimensionality of output targets. Saved for reconstruction.}

  \item{hidden_dims}{Numeric vector of hidden layer sizes. Defaults to
    \code{c(128, 256, 256)} if not specified.}

  \item{activation_name}{Name of the activation function (\code{"relu"},
    \code{"silu"}, or \code{"gelu"}). Saved for reconstruction.}

  \item{dropout}{Dropout rate used in the model. Defaults to \code{0} if not
    specified.}

  \item{col_means}{Column means used to standardize outputs. Needed to reverse-
    transform predictions back to original units via
    \code{\link{probotScaleBackward}}.}

  \item{col_sds}{Column standard deviations used to standardize outputs. Needed
    to reverse-transform predictions back to original units.}

  \item{training_history}{Optional data frame or list of training history (e.g.,
    loss curves) to store alongside the model checkpoint.}

  \item{model_skeleton}{A model object with matching architecture to load into.
    Passed to \code{probotLoadModel()} when you already have a skeleton at hand.}

  \item{load_optimizer}{If \code{TRUE}, also return the saved optimizer state.}

  \item{device}{Device name for loading (e.g., \code{"cpu"} or \code{"cuda"}).
    If \code{NULL}, loads on the default device.}

  \item{\dots}{Additional named objects to store in the checkpoint file metadata.}
}
\details{
\code{probotSave()} writes a list containing \code{model_state} and a
\code{metadata} block (with version, package version, architecture hyperparameters,
and optional scaling parameters) to disk. If an optimizer is provided, its state
dict is also saved.

\code{probotLoad()} is the convenience entry point: it reads the saved file,
reconstructs the model architecture from stored metadata, and loads weights.
It requires that the checkpoint was saved with the necessary hyperparameters
(\code{mdn_components}, \code{input_dim}, \code{output_dim}).

\code{probotLoadModel()} offers more control: it loads a checkpoint without
requiring stored metadata. If you already have a \code{model_skeleton}, pass it;
otherwise the function returns an error asking for one. This is useful for
loading checkpoints saved in the older (pre-v1.0) format.

All three functions save/load in a format compatible with the torch package's
\code{torch_save()} / \code{torch_load()}. Checkpoint files include a version
tag (\code{"1.0"}) to detect incompatible formats on load.
}
\value{
  \itemize{
    \item \code{probotSave()}: The file path, returned invisibly.
    \item \code{probotLoadModel()}: A list with elements \code{model},
      \code{optimizer}, and \code{metadata}.
    \item \code{probotLoad()}: Same as \code{probotLoadModel()}, but with the
      model fully reconstructed from stored metadata.
  }
}
\author{
Aaron Robotham
}
\seealso{
  \code{\link{probotMakeMDN}}, \code{\link{probotTrainMDN}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  model <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2)()

  path <- tempfile(fileext = ".pt")

  #--- Full save with reconstruction metadata ---
  probotSave(
    model,
    filename = path,
    mdn_components = 2,
    input_dim = 2,
    output_dim = 1
  )

  #--- Load and auto-reconstruct ---
  result <- probotLoad(path)
  class(result$model)

  #--- Load into an existing skeleton ---
  restored <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2)()
  result2 <- probotLoadModel(path, model_skeleton = restored)

  file.exists(path)
}
}
