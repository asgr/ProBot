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
    activation_name = NULL,
    dropout = NULL,
    soft_clamp = NULL,
    col_means = NULL,
    col_sds = NULL,
    col_names = NULL,
    training_history = NULL,
    extra_metadata = list()
) {
  model_type <- match.arg(tolower(model_type), c("mdn", "point", "flow"))

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
      activation      = activation_name,
      dropout         = dropout,
      soft_clamp      = soft_clamp,
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

probotLoad <- function(filename, load_optimizer = FALSE, device = NULL) {
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
      soft_clamp  = if (!is.null(meta$soft_clamp)) meta$soft_clamp else 3,
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
