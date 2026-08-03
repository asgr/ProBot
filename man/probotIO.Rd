\name{probotSave}
\alias{probotSave}
\alias{probotLoad}
\title{Save and Reload Mixture Density Network State}
\description{
Save model (and optional optimizer) state to disk and reload model weights.
}
\usage{
probotSave(model, optimizer = NULL, filename = "temp.pt", ...)

probotLoad(filename, model_skeleton)
}
\arguments{
  \item{model}{A torch module instance to save.}

  \item{optimizer}{Optional torch optimizer whose state should also be saved.}

  \item{filename}{Path to a \code{.pt} file used for save/load.}

  \item{model_skeleton}{A model object with matching architecture to load into.}

  \item{\dots}{Additional named objects to store in the checkpoint file.}
}
\details{
\code{probotSave()} writes a list containing at least \code{model} and
\code{optimizer} states. \code{probotLoad()} loads a saved object and applies it
via \code{$load_state_dict()} to \code{model_skeleton}.
}
\value{
\itemize{
  \item \code{probotSave()}: Called for side effect; returns invisibly.
  \item \code{probotLoad()}: The updated \code{model_skeleton} with loaded weights.
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
  probotSave(model, filename = path)

  restored <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2)()
  restored <- probotLoad(path, restored)

  file.exists(path)
}
}
