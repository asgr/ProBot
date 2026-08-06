probotSave <- function(
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
) {

  metadata <- list(
    version = "1.0",
    package_version = as.character(utils::packageVersion("ProBot")),
    mdn_components = mdn_components,
    input_dim = input_dim,
    output_dim = output_dim,
    hidden_dims = hidden_dims,
    activation = activation_name,
    dropout = dropout,
    col_means = col_means,
    col_sds = col_sds,
    training_history = training_history,
      ...
    )

  torch_save(
    list(
      model_state = model$state_dict(),
      optimizer_state = if (!is.null(optimizer)) optimizer$state_dict() else NULL,
      metadata = metadata
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

  checkpoint <- torch_load(
    filename,
    map_location = if (!is.null(device)) torch::torch_device(device) else NULL
   )

  # Check for older format (no nested structure)
  if ("model" %in% names(checkpoint)) {
    warning("Loading from older checkpoint format (pre v1.0)")
    if (!is.null(model_skeleton)) {
      model_skeleton$load_state_dict(checkpoint$model)
      return(model_skeleton)
    }
    stop("Old format checkpoint requires an explicit model_skeleton")
   }

  if (is.null(checkpoint$metadata)) {
    warning("Checkpoint has no 'metadata' block — may be from an older version")
   }

  if (!is.null(model_skeleton)) {
    model_skeleton$load_state_dict(checkpoint$model_state)
    return(list(
      model = model_skeleton,
      optimizer = if (load_optimizer) checkpoint$optimizer_state else NULL,
      metadata = checkpoint$metadata
     )
    )
   }

  stop("No model_skeleton provided. Use probotLoad() for automatic ",
        "reconstruction from saved metadata.")
}

probotLoad <- function(filename, load_optimizer = FALSE, device = NULL) {

  checkpoint <- torch_load(
    filename,
    map_location = if (!is.null(device)) torch::torch_device(device) else NULL
   )

  meta <- checkpoint$metadata
  if (is.null(meta)) {
    stop("Cannot reconstruct model: checkpoint missing 'metadata' block. ",
         "Use probotLoadModel() with an explicit model_skeleton instead.")
   }

   # Check required fields for auto-reconstruction
  required <- c("mdn_components", "input_dim", "output_dim")
  missing_fields <- required[sapply(meta, is.null)]
  if (length(missing_fields) > 0) {
    stop("Cannot reconstruct model: metadata missing ", paste(missing_fields, collapse = ", "),
         ". Save with probotSave() specifying these parameters, or use ",
         "probotLoadModel() with an explicit model_skeleton.")
   }

  # Reconstruct activation function from name
  activation <- nnf_relu
  if (!is.null(meta$activation)) {
    switch(meta$activation,
      relu = ,
        activation <- nnf_relu,
      silu = activation <- nnf_silu,
      gelu = activation <- nnf_gelu,
      warning("Unrecognized activation '", meta$activation, "', falling back to relu")
     )
   }

  # Reconstruct model architecture and load weights
  model_skeleton <- probotMakeMDN(
    input_dim = meta$input_dim,
    output_dim = meta$output_dim,
    mdn_components = meta$mdn_components,
    hidden_dims = if (!is.null(meta$hidden_dims)) meta$hidden_dims else c(128, 256, 256),
    activation = activation,
    dropout = if (!is.null(meta$dropout)) meta$dropout else 0
   )()

  model_skeleton$load_state_dict(checkpoint$model_state)

  list(
    model = model_skeleton,
    optimizer = if (load_optimizer) checkpoint$optimizer_state else NULL,
    metadata = checkpoint$metadata
   )
}
