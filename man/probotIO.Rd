\name{probotIO}
\alias{probotIO}
\alias{probotSave}
\alias{probotLoadModel}
\alias{probotLoad}
\title{Save and Load Model Checkpoints}
\description{
Save trained models to disk with metadata, and load them back with full
auto-reconstruction or into an explicit model skeleton. Supports MDN, Point,
and Normalizing Flow models.
}
\usage{
probotSave(
  model,
  optimizer = NULL,
  filename = "temp.pt",
  model_type = "mdn",
  mdn_components = NULL,
  input_dim = NULL,
  output_dim = NULL,
  n_layers = NULL,
  hidden_dim = NULL,
  hidden_dims = NULL,
  n_blocks = NULL,
  n_layers_per_block = NULL,
  n_bins = NULL,
  tail_bound = NULL,
  activation_name = NULL,
  dropout = NULL,
  soft_clamp = NULL,
  flow_style = NULL,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  training_history = NULL,
  extra_metadata = list()
)

probotLoadModel(
  filename,
  model_skeleton = NULL,
  load_optimizer = FALSE,
  device = NULL
)

probotLoad(
  filename,
  load_optimizer = FALSE,
  device = NULL,
  flow_style = NULL
)
}
\arguments{
    \item{model}{An nn_module object (MDN, Point, or Flow model).}

    \item{optimizer}{Optional optimizer state dict to save alongside the model.}

    \item{filename}{Path to save or load the checkpoint file.}

    \item{model_type}{Character string identifying the model type:
    \code{"mdn"}, \code{"point"}, or \code{"flow"}. Determines which metadata
    fields are required for auto-reconstruction.}

    \item{mdn_components}{Number of MDN mixture components. Required for
    \code{model_type = "mdn"}.}

    \item{input_dim}{Dimensionality of the input (conditioning) features.
    Required for Flow, MDN and Point models.}

    \item{output_dim}{Dimensionality of the output (parameter) space.
    Required for Flow, MDN and Point models.}

    \item{n_layers}{Number of RealNVP layers. Used for Flow model
    reconstruction (defaults to 4 if not saved).}

    \item{hidden_dim}{Width of the shift/scale network inside each RealNVP layer.
    Used for Flow model reconstruction (defaults to 32 if not saved).}
    \item{n_blocks}{Number of MAF (masked autoregressive) blocks for \code{style = "maf"}.}
    \item{n_layers_per_block}{Number of masked layers per autoregressive block.}
    \item{n_bins}{Number of rational-quadratic spline bins for
    \code{style = "nsf"} (defaults to 8 if not saved).}
    \item{tail_bound}{Spline interval bound for \code{style = "nsf"} (defaults
    to 3 if not saved).}

    \item{hidden_dims}{Numeric vector of hidden layer sizes. Used for MDN and
    Point reconstruction (defaults to \code{c(128, 256, 256)} if not saved).}

    \item{activation_name}{Character name of activation function (\code{"relu"},
    \code{"silu"}, or \code{"gelu"}). Used for MDN and Point reconstruction.}

    \item{dropout}{Dropout rate used in the model. Used for MDN and Point
    reconstruction (defaults to 0 if not saved).}

    \item{soft_clamp}{Soft-clamp value for Flow log_scale (defaults to 3 if
    not saved).}

    \item{flow_style}{Which flow architecture the checkpoint holds:
    \code{"realnvp"} (stacked Real Non-Volume Preserving (RealNVP) layers, the default), \code{"maf"}
    (masked autoregressive blocks with permutation layers), or \code{"nsf"}
    (rational-quadratic spline coupling layers). Only meaningful
    for \code{model_type = "flow"}. For \code{probotSave()} it is written into
    the metadata; if \code{NULL}, it is inferred from the model's class
    (\code{probotFlowMAF} instances yield \code{"maf"},
    \code{probotFlowNSF} instances yield \code{"nsf"}, and other flows yield
    \code{"realnvp"}). For \code{probotLoad()} it acts as an override that
    takes precedence over the saved value; useful for loading older flow
    checkpoints saved before \code{flow_style} was recorded.}

    \item{col_means}{Centering values from \code{\link{probotScaleForward}}(); saved as metadata.}

    \item{col_sds}{Scaling values from \code{\link{probotScaleForward}}(); saved as metadata.}

    \item{col_names}{Column names for output theta parameters we want to model; saved as metadata.}

    \item{training_history}{Training history data frame; saved as metadata.}

    \item{extra_metadata}{A named list of additional fields to include in the
    checkpoint metadata block.}

    \item{model_skeleton}{Pre-instantiated model object to load state dict into.
    If \code{NULL}, reconstruction from metadata is required (see
    \code{\link[=probotLoad]{probotLoad()}}).}

    \item{load_optimizer}{Logical; if \code{TRUE}, also restore the optimizer state.}

    \item{device}{Device string for tensor placement (\code{"cpu"}, \code{"mps"},
    etc.). Defaults to \code{NULL}, which auto-detects MPS/CPU.}
}
\details{
\code{probotSave()} stores model weights, optimizer state, and metadata in a
single \code{.pt} file. The \code{model_type} argument determines which
metadata fields are required for auto-reconstruction:

\describe{
  \item{\code{"mdn"}}{requires \code{mdn_components}, \code{input_dim}, \code{output_dim}}
  \item{\code{"point"}}{requires \code{input_dim}, \code{output_dim}}
  \item{\code{"flow"}}{requires \code{input_dim}, \code{output_dim}}
}

When required metadata is missing, \code{probotSave()} issues a warning that
\code{\link[=probotLoad]{probotLoad()}} will be unable to auto-reconstruct the
model.

\code{probotLoadModel()} loads a checkpoint and requires an explicit
\code{model_skeleton} for state restoration. Use this when you cannot (or do
not want to) auto-reconstruct from metadata.

\code{probotLoad()} automatically reconstructs the model architecture from
saved metadata and loads weights. It reads \code{model_type} from the
checkpoint to decide which constructor to call. Old checkpoints that lack
\code{model_type} default to \code{"mdn"} for backward compatibility.

For \code{model_type = "flow"}, the checkpoint's \code{flow_style} metadata
field selects the architecture (defaults to \code{"realnvp"} if the field is
absent, as in checkpoints saved before this field existed). A \code{probotSave()}
call records \code{flow_style} automatically; the \code{flow_style} argument
to \code{probotLoad()} overrides the saved value.

\code{probotLoadModel()} loads into an explicit model skeleton but does not
require \code{model_type} metadata.
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
\code{\link{probotMakeMDN}}, \code{\link{probotMakePoint}}, \code{\link{probotMakeFlow}}
}
\examples{
\dontrun{
## --- MDN model ---
probotSave(
  fitted$model,
  optimizer = fitted$optimizer,
  filename = "my_mdn.pt",
  model_type = "mdn",
  mdn_components = 3,
  input_dim = 2,
  output_dim = 2,
  col_means = c(0, 0),
  col_sds = c(1, 1)
)

loaded <- probotLoad("my_mdn.pt")

## --- Point model ---
probotSave(
  point_model,
  filename = "my_point.pt",
  model_type = "point",
  input_dim = 3,
  output_dim = 2
)

loaded_point <- probotLoad("my_point.pt", device = "mps")

## --- Flow model ---
probotSave(
  flow_model,
  filename = "my_flow.pt",
  model_type = "flow",
  input_dim = 3,
  output_dim = 4,
  n_layers = 6,
  hidden_dim = 64
)

loaded_flow <- probotLoad("my_flow.pt")

## Older MAF checkpoint saved before flow_style was recorded:
loaded_maf <- probotLoad("my_maf.pt", flow_style = "maf")
}
}
