# Metadata schemas: lists of required field names per model type
.probotRequired <- list(
  mdn   = c("mdn_components", "input_dim", "output_dim"),
  point = c("input_dim", "output_dim"),
  flow  = c("dim_theta", "dim_x")
)

probotSave <- function(
    model,
    optimizer = NULL,
    filename = "temp.pt",
    model_type = "mdn",
    mdn_components = NULL,
    input_dim = NULL,
    output_dim = NULL,
    dim_theta = NULL,
    dim_x = NULL,
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
) {
  model_type <- match.arg(tolower(model_type), c("mdn", "point", "flow"))

  # For flow models, record which architecture the checkpoint holds so
  # probotLoad() can reconstruct the correct skeleton. Infer it from the
  # model's class when not given explicitly.
  if (model_type == "flow") {
    if (is.null(flow_style)) {
      flow_style <- if (any(grepl("MAF", class(model)))) {
        "maf"
      } else if (any(grepl("NSF", class(model)))) {
        "nsf"
      } else {
        "realnvp"
      }
    } else {
      flow_style <- match.arg(tolower(flow_style), c("realnvp", "maf", "nsf"))
    }
  } else {
    flow_style <- NULL
  }

  required <- .probotRequired[[model_type]]

  meta_values <- switch(model_type,
    mdn   = list(mdn_components, input_dim, output_dim),
    point = list(input_dim, output_dim),
    flow  = list(dim_theta, dim_x)
  )

  missing <- required[vapply(meta_values, is.null, logical(1))]
  if (length(missing) > 0) {
    warning(
      "Checkpoint metadata missing ",
      paste(missing, collapse = ", "),
      ". probotLoad() will be unable to auto-reconstruct this model; ",
      "use probotLoadModel() with an explicit model_skeleton instead.",
      call. = FALSE
    )
  }

  metadata <- c(
    list(
      version         = "1.0",
      package_version = as.character(utils::packageVersion("ProBot")),
      model_type      = model_type,
      mdn_components  = mdn_components,
      input_dim       = input_dim,
      output_dim      = output_dim,
      dim_theta       = dim_theta,
      dim_x           = dim_x,
      n_layers        = n_layers,
      hidden_dim      = hidden_dim,
      hidden_dims     = hidden_dims,
      n_blocks        = n_blocks,
      n_layers_per_block = n_layers_per_block,
      n_bins          = n_bins,
      tail_bound      = tail_bound,
      activation      = activation_name,
      dropout         = dropout,
      soft_clamp      = soft_clamp,
      flow_style      = flow_style,
      col_means       = col_means,
      col_sds         = col_sds,
      col_names       = col_names,
      training_history = training_history
    ),
    extra_metadata
  )

  torch_save(
    list(
      model_state     = model$state_dict(),
      optimizer_state = if (!is.null(optimizer)) optimizer$state_dict() else NULL,
      metadata        = metadata
    ),
    filename
  )

  invisible(filename)
}

probotLoadModel <- function(
    filename,
    model_skeleton = NULL,
    load_optimizer = FALSE,
    device = NULL
) {
  target_device <- .probotChooseDevice(device)

  checkpoint <- torch_load(filename, device = target_device)

  # Check for older format (no nested structure)
  if ("model" %in% names(checkpoint)) {
    warning("Loading from older checkpoint format (pre v1.0)")
    if (!is.null(model_skeleton)) {
      model_skeleton$load_state_dict(checkpoint$model)
      return(list(
        model     = model_skeleton,
        optimizer = if (load_optimizer) checkpoint$optimizer else NULL,
        metadata  = NULL
      ))
    }
    stop("Old format checkpoint requires an explicit model_skeleton")
  }

  if (is.null(checkpoint$metadata)) {
    warning("Checkpoint has no 'metadata' block - may be from an older version")
  }

  if (!is.null(model_skeleton)) {
    model_skeleton$load_state_dict(checkpoint$model_state)
    return(list(
      model     = model_skeleton,
      optimizer = if (load_optimizer) checkpoint$optimizer_state else NULL,
      metadata  = checkpoint$metadata
    ))
  }

  stop("No model_skeleton provided. Use probotLoad() for automatic ",
       "reconstruction from saved metadata.")
}

probotLoad <- function(filename, load_optimizer = FALSE, device = NULL, flow_style = NULL) {
  target_device <- .probotChooseDevice(device)

  checkpoint <- torch_load(filename, device = target_device)

  meta <- checkpoint$metadata
  if (is.null(meta)) {
    stop("Cannot reconstruct model: checkpoint missing 'metadata' block. ",
         "Use probotLoadModel() with an explicit model_skeleton instead.")
  }

  # Determine model type (default to "mdn" for backward compatibility)
  model_type <- if (!is.null(meta$model_type)) tolower(meta$model_type) else "mdn"
  if (!model_type %in% names(.probotRequired)) {
    model_type <- "mdn"
  }

  # flow_style is an optional override that takes precedence over the value
  # saved in metadata. It is the escape hatch for loading older flow
  # checkpoints saved before flow_style was recorded in the metadata block.
  if (!is.null(flow_style)) {
    flow_style <- match.arg(tolower(flow_style), c("realnvp", "maf", "nsf"))
  }

  required <- .probotRequired[[model_type]]
  missing_fields <- required[
    vapply(required, function(field) is.null(meta[[field]]), logical(1))
  ]
  if (length(missing_fields) > 0) {
    stop(
      "Cannot reconstruct model (type='", model_type, "'): metadata missing ",
      paste(missing_fields, collapse = ", "),
      ". Save with probotSave() specifying these parameters, or use ",
      "probotLoadModel() with an explicit model_skeleton."
    )
  }

  # Reconstruct activation function from name (MDN / Point only)
  activation <- nnf_relu
  if (!is.null(meta$activation)) {
    activation <- switch(
      meta$activation,
      relu = nnf_relu,
      silu = nnf_silu,
      gelu = nnf_gelu,
      {
        warning("Unrecognized activation '", meta$activation, "', falling back to relu")
        nnf_relu
      }
    )
  }

  model_skeleton <- switch(model_type,
    mdn = probotMakeMDN(
      input_dim      = meta$input_dim,
      output_dim     = meta$output_dim,
      mdn_components = meta$mdn_components,
      hidden_dims    = if (!is.null(meta$hidden_dims)) meta$hidden_dims else c(128, 256, 256),
      activation     = activation,
      dropout        = if (!is.null(meta$dropout)) meta$dropout else 0,
      device         = device
    )(),
    point = probotMakePoint(
      input_dim   = meta$input_dim,
      output_dim  = meta$output_dim,
      hidden_dims = if (!is.null(meta$hidden_dims)) meta$hidden_dims else c(128, 256, 256),
      activation  = activation,
      dropout     = if (!is.null(meta$dropout)) meta$dropout else 0,
      device      = device
    )(),
    flow = probotMakeFlow(
      dim_theta   = meta$dim_theta,
      dim_x       = meta$dim_x,
      n_layers    = if (!is.null(meta$n_layers)) meta$n_layers else 4,
      hidden_dim  = if (!is.null(meta$hidden_dim)) meta$hidden_dim else 32,
      n_blocks  = if (!is.null(meta$n_blocks)) meta$n_blocks else 5,
      n_layers_per_block = if (!is.null(meta$n_layers_per_block)) meta$n_layers_per_block else 2,
      n_bins = if (!is.null(meta$n_bins)) meta$n_bins else 8,
      tail_bound = if (!is.null(meta$tail_bound)) meta$tail_bound else 3,
      soft_clamp  = if (!is.null(meta$soft_clamp)) meta$soft_clamp else 3,
      # Priority: explicit override > saved metadata > "realnvp" (legacy default).
      style       = if (!is.null(flow_style)) flow_style else
                    if (!is.null(meta$flow_style)) meta$flow_style else "realnvp",
      device      = device
    )()
  )

  model_skeleton$load_state_dict(checkpoint$model_state)

  list(
    model     = model_skeleton,
    optimizer = if (load_optimizer) checkpoint$optimizer_state else NULL,
    metadata  = checkpoint$metadata
  )
}
