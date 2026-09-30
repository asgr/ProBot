\name{probotIO}
\alias{probotIO}
\alias{probotSave}
\alias{probotLoadModel}
\alias{probotLoad}
\title{Save and Load Model Checkpoints}
\description{
Save trained models to disk with metadata, and load them back with full
auto-reconstruction or into an explicit model skeleton. Supports MDN, Point,
and Normalizing Flow models. The architecture metadata is read from the model
being saved, so a checkpoint cannot record an architecture other than the one it
contains.
}
\usage{
probotSave(
  model,
  optimizer = NULL,
  filename = "temp.pt",
  model_type = NULL,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  training_history = NULL,
  extra_metadata = list(),
  ...
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
    \item{model}{An nn_module object (MDN, Point, or Flow model), as returned by
    \code{\link{probotMakeMDN}}, \code{\link{probotMakePoint}} or
    \code{\link{probotMakeFlow}}.}

    \item{optimizer}{Optional optimizer to save alongside the model. Its
    \code{state_dict()} is stored.}

    \item{filename}{Path to save or load the checkpoint file.}

    \item{model_type}{For \code{probotSave()}: normally omitted, since the type is
    detected from \code{model}'s class. Pass it explicitly only for a hand-built
    \code{nn_module} that carries none of the in-tree class tags. A value that
    contradicts the object is an error, not a silent overwrite. Old checkpoints
    that lack it default to \code{"mdn"} on load; see \strong{Details}.}

    \item{col_means}{Numeric vector of centering values from
    \code{\link{probotScaleForward}}(). This is the only architecture-adjacent
    metadata a model cannot know about itself, so it is still supplied. Its
    length is checked against the model's \code{output_dim}.}

    \item{col_sds}{Numeric vector of scaling values from
    \code{\link{probotScaleForward}}(); saved as metadata.}

    \item{col_names}{Character vector of names for the modelled output
    parameters; saved as metadata.}

    \item{training_history}{Training history data frame; saved as metadata.}

    \item{extra_metadata}{A named list of additional fields to include in the
    checkpoint metadata block.}

    \item{...}{Must be empty. The arguments \code{mdn_components},
    \code{input_dim}, \code{output_dim}, \code{n_layers}, \code{hidden_dim},
    \code{hidden_dims}, \code{n_blocks}, \code{n_layers_per_block}, \code{n_bins},
    \code{tail_bound}, \code{activation_name}, \code{dropout}, \code{soft_clamp}
    and \code{flow_style} used to be accepted and now raise an error naming the
    argument, because they are inferred from \code{model}. Any other unrecognised
    argument is likewise reported rather than ignored.}

    \item{model_skeleton}{Pre-instantiated model object to load state dict into.
    If \code{NULL}, reconstruction from metadata is required (see
    \code{\link[=probotLoad]{probotLoad()}}).}

    \item{load_optimizer}{Logical; if \code{TRUE}, also restore the optimizer state.}

    \item{device}{Device string for tensor placement (\code{"cpu"}, \code{"mps"},
    etc.). Defaults to \code{NULL}, which auto-detects MPS/CPU.}

    \item{flow_style}{For \code{probotLoad()} only: an explicit override of the
    architecture inferred from the weights --- \code{"realnvp"}, \code{"maf"} or
    \code{"nsf"}. The legacy values \code{"couple"} and \code{"autoreg"} are
    mapped to \code{"realnvp"} and \code{"maf"}. Use it for checkpoints written
    before \code{flow_style} could be inferred; it is not needed for anything
    saved by a current \code{probotSave()}. An override is not reported as a
    disagreement with the weights.}
}
\details{
\code{probotSave()} stores the model's \code{state_dict()}, optional optimizer
state, and a metadata block in a single \code{.pt} file.

\subsection{What is recorded, and where from}{

Everything describing the architecture is derived from the model object passed in
\emph{and} cross-checked against its weight shapes, so the two cannot disagree in
silence:

  \itemize{
    \item \strong{From \code{state_dict()} shapes} \emph{(ground truth)}: layer
    counts and widths for all five architectures; \code{input_dim} and the hidden
    width chain for MDN and Point; \code{n_blocks} and \code{n_layers_per_block}
    for \code{"maf"}; \code{n_bins} for \code{"nsf"}, recovered from the
    conditioner's output width; and \code{flow_style}, from which key namespace is
    present.
    \item \strong{From fields the constructors store} \emph{(not visible in any
    tensor)}: \code{soft_clamp} and \code{tail_bound}, which change how the
    network computes but not its shapes; \code{input_dim} and \code{output_dim}
    for flows, where a coupling layer's conditioner sees \code{d1 + input_dim}
    features and the split is not recoverable; the \code{mdn_components} /
    \code{output_dim} pair, since an MDN head of \code{K * (2D + 1)} rows admits
    several valid \code{(K, D)} factorisations; and \code{activation} and
    \code{dropout} for MDN and Point.
    \item \strong{Supplied by the caller} \emph{(the model cannot know this)}:
    \code{col_means}, \code{col_sds}, \code{col_names}, \code{training_history}
    and \code{extra_metadata}.
  }

Where a shape-derived value contradicts a stored field, the shapes win and a
warning names the field, so an inconsistency is reported rather than written out.

Those fields used to be arguments. That is how one real checkpoint came to record
\code{hidden_dim = c(736, 1472, 736)} --- an MDN-shaped three-element vector ---
for a 19-layer, hidden-512 RealNVP: \code{probotSave()} had both \code{hidden_dim}
and \code{hidden_dims} and no \code{...}, so a mistyped argument silently
partial-matched into the wrong slot. Passing one of the retired names now errors.
}

\subsection{What cannot be inferred}{

For a model that exposes neither the stored fields above nor a resolvable shape
--- a hand-built \code{nn_module} with an ambiguous head width, for instance ---
\code{probotSave()} warns which fields are missing and notes that
\code{\link[=probotLoad]{probotLoad()}} will not be able to reconstruct it. Use
\code{probotLoadModel()} with an explicit \code{model_skeleton} for such a model.
}

\subsection{Loading}{

\code{probotLoad()} reads \code{model_type} from the checkpoint and calls the
matching constructor. Old checkpoints that lack \code{model_type} default to
\code{"mdn"} for backward compatibility.

For \code{model_type = "flow"} the architecture is read from the saved
\emph{tensors} rather than from metadata, for the same reason and with the same
effect as above: shapes are ground truth, and checkpoints written by older builds
record whatever their caller passed. \code{probotLoad()} infers \code{flow_style},
\code{n_layers} (or \code{n_blocks} and \code{n_layers_per_block} for
\code{"maf"}), \code{hidden_dim} and \code{n_bins} from the \code{state_dict}.
When saved metadata disagrees with the weights, the weights win, a warning names
the field, and the \code{metadata} list that is returned is corrected in place so
that it always describes the model you get back. Checkpoints whose metadata lacks
\code{flow_style} --- saved before that field existed, or before it was inferred
--- therefore still load into the right architecture.

\code{input_dim} and \code{output_dim} come from metadata and are required, for
the reason given above. A headed (\code{\link{probotFlowLoc}}) checkpoint needs no
argument: its \code{loc_head} and \code{loc_hidden_dims} metadata are used as
saved, and the style of the flow inside the head is inferred from the
\code{base_flow.*} key namespace.

\code{soft_clamp} and \code{tail_bound} are neither recoverable from shapes nor
adjustable at load time; they are used exactly as saved, defaulting to 3 when
absent. A current \code{probotSave()} always records them for models that use
them.

\code{probotLoadModel()} loads into an explicit model skeleton and does not
require \code{model_type} metadata.
}
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
mdl <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3)()
probotSave(
  mdl,
  optimizer = fitted$optimizer,
  filename = "my_mdn.pt",
  col_means = c(0, 0),
  col_sds = c(1, 1)
)

loaded <- probotLoad("my_mdn.pt")

## --- Point model ---
probotSave(point_model, filename = "my_point.pt")

loaded_point <- probotLoad("my_point.pt", device = "mps")

## --- Flow model: style, depth and width all recorded from the model ---
flow_model <- probotMakeFlow(input_dim = 3, output_dim = 4,
                             n_layers = 6, hidden_dim = 64)()
probotSave(flow_model, filename = "my_flow.pt")

loaded_flow <- probotLoad("my_flow.pt")

## An old MAF checkpoint whose weights predate flow_style inference:
loaded_maf <- probotLoad("my_maf.pt", flow_style = "maf")
}
}
