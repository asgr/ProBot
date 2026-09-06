# Metadata schemas: lists of required field names per model type
.probotRequired <- list(
  mdn   = c("mdn_components", "input_dim", "output_dim"),
  point = c("input_dim", "output_dim"),
  flow  = c("input_dim", "output_dim")
)

# Infer a flow's architecture from its state_dict.
#
# Tensor shapes are the ground truth for depth and width; the matching metadata
# fields are whatever the caller happened to pass to probotSave(), which can be
# missing or wrong (a real checkpoint carries hidden_dim = c(736, 1472, 736),
# the *MDN/Point* three-layer shape, and n_layers = NULL for a 19-layer,
# hidden-512 RealNVP). Rebuilding from that metadata dies deep inside torch with
# "You should specify a single size argument", which names neither the file nor
# the field. Inferring instead turns that into a working load.
#
# Returns a named list holding only the fields it could determine:
# style, n_layers, hidden_dim, n_blocks, n_layers_per_block, n_bins.
.probotFlowArchFromState <- function(state_dict, output_dim = NULL) {
  nm <- names(state_dict)
  if (length(nm) == 0L) {
    return(list())
  }

  # A location head namespaces the flow's own keys under "base_flow." alongside
  # the head's own "loc_layers.*" keys. Strip the prefix from the keys that carry
  # it so every style below is read identically headed or plain; `fk` is the
  # stripped name used for pattern matching, while shapes are still looked up
  # against the original key in `nm`.
  fk <- sub("^base_flow\\.", "", nm)

  # Weight matrices are (out_features, in_features), so the first dimension is
  # the width of the layer and the second is its input.
  shape <- function(key) {
    i <- match(key, fk)
    if (is.na(i) || is.null(state_dict[[nm[[i]]]])) return(NULL)
    as.integer(state_dict[[nm[[i]]]]$shape)
  }

  # Contiguous layer indices, or NULL if they are missing/duplicated/non-monotone
  # (in which case we would not know how many layers to build). MAF registers its
  # permutation layers in the same nn_module_list as its blocks, so block indices
  # step by 2 from 0 even though the permutations hold no weights.
  dense_indices <- function(idx, start, by) {
    idx <- sort(unique(idx))
    if (length(idx) == 0L) return(NULL)
    expected <- as.integer(seq(start, start + by * (length(idx) - 1L), by = by))
    if (!identical(idx, expected)) return(NULL)
    idx
  }

  # Split key names on `pattern`, which must match every key and capture one or
  # more runs of digits. Returns a list of integer vectors, one per capture
  # group, or NULL if any key fails to match. The whole key must be anchored:
  # a sub() that consumes a trailing "." but not the remainder yields a string
  # like "1shift_scale_net.0.weight", and as.integer() silently turns that to NA.
  grab_integers <- function(keys, pattern) {
    if (!all(grepl(pattern, keys))) return(NULL)
    caps <- regmatches(keys, regexec(pattern, keys))
    # Each element of `caps` is c(whole_match, group_1, ..., group_n)
    n_groups <- length(caps[[1L]]) - 1L
    lapply(seq_len(n_groups), function(g)
      as.integer(vapply(caps, function(m) m[[g + 1L]], character(1))))
  }

  # --- RealNVP: RealNVP_layer_<i>.shift_scale_net.<0|2|4>.weight
  pat <- "^RealNVP_layer_([0-9]+)\\.shift_scale_net\\.0\\.weight$"
  keys <- grep(pat, fk, value = TRUE)
  if (length(keys) > 0L) {
    ints <- grab_integers(keys, pat)
    idx <- if (is.null(ints)) NULL else dense_indices(ints[[1L]], start = 1, by = 1)
    if (is.null(idx)) return(list(style = "realnvp"))
    return(list(
      style      = "realnvp",
      n_layers   = length(idx),
      hidden_dim = shape(keys[[1L]])[1L]
    ))
  }

  # --- NSF: spline_layer_<i>.conditioner.<0|2|4>.weight
  pat <- "^spline_layer_([0-9]+)\\.conditioner\\.0\\.weight$"
  keys <- grep(pat, fk, value = TRUE)
  if (length(keys) > 0L) {
    ints <- grab_integers(keys, pat)
    idx <- if (is.null(ints)) NULL else dense_indices(ints[[1L]], start = 1, by = 1)
    out <- list(
      style      = "nsf",
      n_layers   = if (is.null(idx)) NULL else length(idx),
      hidden_dim = shape(keys[[1L]])[1L]
    )
    # The final conditioner layer emits d2 * (3 * n_bins - 1) values per
    # transformed dimension, so its row count inverts to n_bins. Locate it by
    # suffix rather than assuming the index of the first layer's key.
    last <- grep("\\.conditioner\\.4\\.weight$", fk, value = TRUE)
    if (!is.null(out$n_layers) && !is.null(output_dim) && length(last) == out$n_layers) {
      d2 <- output_dim - floor(output_dim / 2)
      rows <- shape(last[[1L]])
      if (!is.null(rows) && length(rows) == 2L) {
        n_bins <- (rows[1L] / d2 + 1) / 3
        if (isTRUE(all.equal(n_bins, round(n_bins))) && round(n_bins) >= 2) {
          out$n_bins <- as.integer(round(n_bins))
        }
      }
    }
    return(out)
  }

  # --- MAF: blocks.<j>.layers.<i>.linear.weight
  # Permutation layers sit between blocks but hold no parameters, so they never
  # appear here: the block count is the count of distinct block indices.
  pat <- "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.linear\\.weight$"
  keys <- grep(pat, fk, value = TRUE)
  if (length(keys) > 0L) {
    ints <- grab_integers(keys, pat)
    if (is.null(ints)) return(list(style = "maf"))
    blocks <- dense_indices(ints[[1L]], start = 0, by = 2)
    layers <- dense_indices(ints[[2L]], start = 0, by = 1)
    if (is.null(blocks) || is.null(layers)) {
      return(list(style = "maf"))
    }
    return(list(
      style              = "maf",
      n_blocks           = length(blocks),
      n_layers_per_block = length(layers),
      hidden_dim         = shape(keys[[1L]])[1L]
    ))
  }

  list()
}

# Architecture fields that probotSave() infers from the model. They used to be
# arguments, which is how a checkpoint ended up carrying hidden_dim =
# c(736, 1472, 736) -- the *MDN/Point* three-layer shape -- for a 19-layer,
# hidden-512 RealNVP: probotSave() had both hidden_dim and hidden_dims and no
# `...`, so a mistyped argument silently partial-matched into the wrong slot.
# Passing one now raises an error that names it.
.probotInferredFields <- c(
  "mdn_components", "input_dim", "output_dim", "n_layers", "hidden_dim",
  "hidden_dims", "n_blocks", "n_layers_per_block", "n_bins", "tail_bound",
  "activation_name", "dropout", "soft_clamp", "flow_style"
)

# Map an activation function back to the name probotLoad() can rebuild it from.
# Matched by identity because the constructors store the function itself; these
# three are exactly the names probotLoad()'s switch knows. NULL means "not one of
# the recognised set", which the caller reports.
.probotActivationName <- function(activation) {
  nms <- c("relu", "gelu", "silu")
  fns <- list(torch::nnf_relu, torch::nnf_gelu, torch::nnf_silu)
  hit <- which(vapply(fns, function(f) identical(f, activation), logical(1)))
  if (length(hit) == 0L) return(NULL)
  nms[hit[1L]]
}

# Classify a model the way probotLoad() names it. MDN and Point modules carry the
# nn_module "class" tags given by their constructors ("mlp", "mlp_point"); those
# two names are distinct, so no order dependency between them.
.probotModelClassType <- function(model) {
  if (inherits(model, "mlp_point")) return("point")
  if (.probotIsFlow(model)) return("flow")
  if (inherits(model, "mlp")) return("mdn")
  NULL
}

# Layer widths of an MDN/Point MLP from its state_dict. Both are a flat
# nn_module_list named layers.<i>, with i 0-based, so the shapes give the full
# width chain exactly: the first layer's columns are input_dim and each row is
# that layer's output width.
.probotMLPShape <- function(state_dict) {
  pat <- "^layers\\.([0-9]+)\\.weight$"
  keys <- grep(pat, names(state_dict), value = TRUE)
  if (length(keys) == 0L) return(NULL)
  ints <- as.integer(sub(pat, "\\1", keys))
  if (anyNA(ints) || !identical(sort(ints), seq.int(0L, length(ints) - 1L))) return(NULL)
  keys <- keys[order(ints)]
  sh <- lapply(keys, function(k) as.integer(state_dict[[k]]$shape))
  list(
    input_dim  = sh[[1L]][2L],
    out_widths = vapply(sh, function(s) s[1L], integer(1))
  )
}

# Split an MDN head width into components: rows = K * (2D + 1). Both unknowns make
# the factorisation ambiguous -- for rows = 15 the valid splits are (K=1,D=7),
# (K=3,D=2) and (K=5,D=1), and nothing in the weight shapes distinguishes them --
# so this only applies when D is already known from the model object.
.probotMDNComponents <- function(rows, output_dim) {
  if (is.null(rows) || is.null(output_dim) || output_dim < 1) return(NULL)
  k <- rows / (2 * output_dim + 1)
  if (k < 1 || !isTRUE(all.equal(k, round(k)))) return(NULL)
  as.integer(round(k))
}

# The architecture keys probotSave() writes, in the order it writes them. Also the
# contract on .probotSaveArch()'s return: `arch$x <- NULL` *deletes* the element,
# and a missing key lets metadata$n_layers partially match n_layers_per_block --
# silently reading the wrong field, the same class of invisible mis-read that
# produced the corrupt checkpoint this whole path exists to survive.
.probotArchKeys <- c(
  "mdn_components", "input_dim", "output_dim", "n_layers", "hidden_dim",
  "hidden_dims", "n_blocks", "n_layers_per_block", "n_bins", "tail_bound",
  "soft_clamp", "flow_style", "loc_head", "loc_hidden_dims",
  "activation", "dropout"
)

# Public entry point: build the architecture metadata, then restore the full key
# set and order no matter which fields the builder could resolve.
.probotSaveArch <- function(model, model_type, state_dict) {
  built <- .probotSaveArchFields(model, model_type, state_dict)
  out <- stats::setNames(vector("list", length(.probotArchKeys)), .probotArchKeys)
  hit <- intersect(.probotArchKeys, names(built))
  out[hit] <- built[hit]
  # A field the builder invented (none today) is kept rather than dropped.
  rest <- setdiff(names(built), .probotArchKeys)
  out[rest] <- built[rest]
  out
}

# Build the architecture half of the metadata from the model itself.
#
# Two independent sources, cross-checked rather than trusted: the constructor
# arguments each in-tree model stores on self, and the tensor shapes in
# state_dict(). Shapes are the ground truth for depth and width; the stored
# scalars are the only way to recover soft_clamp, tail_bound and MDN's
# (components, output_dim) split for a single-hidden-layer network. A disagreement
# is reported, because it means the object being saved is not internally
# consistent. Absent fields stay absent; .probotSaveArch() normalises the key set.
.probotSaveArchFields <- function(model, model_type, state_dict) {
  arch <- list()

  check <- function(field, from_state) {
    got <- arch[[field]]
    if (is.null(from_state)) return(invisible(NULL))
    if (is.null(got)) {
      arch[[field]] <<- from_state
      return(invisible(NULL))
    }
    # Character fields (flow_style) must not go through all.equal's numeric path.
    same <- if (is.character(got) || is.character(from_state)) {
      identical(as.character(got), as.character(from_state))
    } else {
      isTRUE(all.equal(as.numeric(got), as.numeric(from_state),
                       check.attributes = FALSE))
    }
    if (!same) {
      warning("Model's stored ", field, " = ", paste(got, collapse = ", "),
              " disagrees with its weights (",
              paste(from_state, collapse = ", "), "); recording the weights.",
              call. = FALSE)
      arch[[field]] <<- from_state
    }
    invisible(NULL)
  }

  if (model_type %in% c("mdn", "point")) {
    arch$activation <- .probotActivationName(model$activation_fn)
    arch$dropout <- model$dropout_rate
    arch$input_dim <- model$input_dim
    arch$output_dim <- model$output_dim
    arch$hidden_dims <- model$hidden_dims
    arch$mdn_components <- model$mdn_components

    shp <- .probotMLPShape(state_dict)
    rows <- NULL
    if (!is.null(shp)) {
      check("input_dim", shp$input_dim)
      # The weight chain is input -> h1 -> ... -> hm -> head, so every layer but
      # the head is a hidden one. A network with no hidden layers has an empty
      # vector, which is a real architecture rather than a missing value, so it
      # is assigned rather than passed to check() as "could not determine".
      n_layers <- length(shp$out_widths)
      hidden <- if (n_layers > 1L) shp$out_widths[seq_len(n_layers - 1L)] else integer(0)
      if (is.null(arch$hidden_dims)) {
        arch$hidden_dims <- hidden
      } else {
        check("hidden_dims", hidden)
      }
      rows <- shp$out_widths[n_layers]
      if (model_type == "point" && is.null(arch$output_dim)) {
        arch$output_dim <- rows
      }
      if (model_type == "mdn" && is.null(arch$mdn_components)) {
        arch$mdn_components <- .probotMDNComponents(rows, arch$output_dim)
      }
    }

    # Final coherence check on the head width. Skipped when either unknown is
    # genuinely undetermined -- that case already warns via .probotRequired.
    d <- arch$output_dim
    k <- arch$mdn_components
    if (!is.null(rows) && !is.null(d)) {
      expect <- if (model_type == "mdn") {
        if (is.null(k)) NULL else k * (2 * d + 1)
      } else {
        d
      }
      if (!is.null(expect) &&
          !isTRUE(all.equal(as.numeric(rows), as.numeric(expect)))) {
        warning("Model's output layer has ", rows, " rows, which does not match the ",
                "stored output_dim = ", d,
                if (model_type == "mdn") paste0(" and mdn_components = ", k), ".",
                call. = FALSE)
      }
    }
    return(arch)
  }

  # --- flow ---
  # Unwrap the location head: the wrapper stores the head's own dimensions and the
  # style underneath is whatever base_flow is. probotFlowLoc's class matches none
  # of the style patterns, hence the unwrap rather than a class test on `model`.
  base <- model
  depth <- 0L
  while (inherits(base, "probotFlowLoc")) {
    depth <- depth + 1L
    if (depth > 10L) {
      stop("Encountered more than 10 nested 'probotFlowLoc' wrappers", call. = FALSE)
    }
    if (depth == 1L) {
      arch$loc_head <- TRUE
      arch$loc_hidden_dims <- model$loc_hidden_dims
    }
    base <- base$base_flow
  }
  arch$flow_style <- .probotFlowStyleName(model)
  arch$input_dim <- if (!is.null(model$input_dim)) model$input_dim else base$input_dim
  arch$output_dim <- if (!is.null(model$output_dim)) model$output_dim else base$output_dim
  arch$hidden_dim <- base$hidden_dim
  arch$soft_clamp <- base$soft_clamp
  arch$n_layers <- base$n_layers
  arch$n_blocks <- base$n_blocks
  arch$n_bins <- base$n_bins
  arch$tail_bound <- base$tail_bound

  # MAF keeps its width and per-block depth on the block, not the top module, and
  # its block list interleaves parameterless permutation layers.
  if (inherits(base, "probotFlowMAF") && length(base$blocks) > 0L) {
    blk <- base$blocks[[1L]]
    arch$n_layers_per_block <- blk$n_layers
    if (is.null(arch$hidden_dim)) arch$hidden_dim <- blk$hidden_dim
  }

  # Cross-check every shape-visible field against the tensors, using the same
  # reader probotLoad() relies on. Anything it disagrees about is recorded from
  # the tensors.
  derived <- .probotFlowArchFromState(state_dict, output_dim = arch$output_dim)
  names(derived)[names(derived) == "style"] <- "flow_style"
  for (fld in names(derived)) check(fld, derived[[fld]])

  arch
}

probotSave <- function(
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
) {
  # Anything supplied positionally beyond the fourth formal now lands in the
  # scaling-metadata slots: a legacy call written as
  # probotSave(model, opt, file, "mdn", mdn_components, input_dim, output_dim)
  # would bind mdn_components to col_means. Refuse it rather than mis-file it.
  # names() is NULL when every argument is positional, which must not count as
  # zero (`%in% NULL` is always FALSE).
  sc <- sys.call()
  snm <- names(sc)
  n_positional <- if (is.null(snm)) length(sc) - 1L else sum(snm == "" | is.na(snm))
  safe_positional <- 4L
  if (n_positional > safe_positional) {
    stop("probotSave(): only ", safe_positional, " arguments (model, optimizer, ",
         "filename, model_type) may be given positionally; everything after them ",
         "must be named. This call passed ", n_positional, " by position.",
         call. = FALSE)
  }

  dots <- list(...)
  retired <- intersect(names(dots), .probotInferredFields)
  if (length(retired) > 0L) {
    stop("probotSave() no longer takes ",
         paste(sQuote(sort(retired)), collapse = ", "),
         ". This metadata is read from `model` so that it cannot contradict the ",
         "weights being saved; remove ",
         if (length(retired) == 1L) "that argument" else "those arguments",
         " from the call.", call. = FALSE)
  }
  if (length(dots) > 0L) {
    # Unnamed extras reach here via do.call() with a partly-named list, where
    # names(dots) is "" rather than missing. Label those with R's own ..i
    # notation so the message is never blank.
    dnm <- names(dots)
    if (is.null(dnm)) dnm <- rep("", length(dots))
    dnm[!nzchar(dnm)] <- paste0("..", seq_along(dots))[!nzchar(dnm)]
    stop("probotSave() received unrecognised argument(s): ",
         paste(sQuote(dnm), collapse = ", "),
         ". Architecture metadata is inferred from `model`; the remaining ",
         "arguments are col_means, col_sds, col_names, training_history and ",
         "extra_metadata.", call. = FALSE)
  }

  # Determine the model type from the object, and validate any explicit claim
  # against it. model_type stays an argument because a hand-built nn_module
  # exposes no class tag to read, but a wrong value here would mislabel the
  # checkpoint and send probotLoad() down the wrong reconstruction branch.
  detected <- .probotModelClassType(model)
  if (is.null(model_type)) {
    if (is.null(detected)) {
      stop("probotSave() cannot tell which model type `model` is; pass ",
           "model_type = \"mdn\", \"point\" or \"flow\" explicitly.",
           call. = FALSE)
    }
    model_type <- detected
  } else {
    model_type <- match.arg(tolower(model_type), c("mdn", "point", "flow"))
    if (!is.null(detected) && !identical(detected, model_type)) {
      stop("model_type = ", sQuote(model_type), " describes a ", detected,
           " model, not the ", sQuote(model_type), " model it was given. ",
           "Omit model_type to have it inferred.", call. = FALSE)
    }
  }

  state_dict <- model$state_dict()
  arch <- .probotSaveArch(model, model_type, state_dict)

  required <- .probotRequired[[model_type]]
  missing_fields <- required[
    vapply(required, function(field) is.null(arch[[field]]), logical(1))
  ]
  if (length(missing_fields) > 0L) {
    warning(
      "Checkpoint metadata missing ",
      paste(missing_fields, collapse = ", "),
      ": this model does not expose them and they cannot be recovered from its ",
      "weight shapes. probotLoad() will be unable to auto-reconstruct it; use ",
      "probotLoadModel() with an explicit model_skeleton instead.",
      call. = FALSE
    )
  }

  # Scaling metadata is the one part of the checkpoint a model cannot know about
  # itself, so it is still supplied. Length is checked against the parameter space
  # because these vectors are the other thing a caller used to be able to get
  # wrong, and because a positional call written against the old signature would
  # otherwise bind them to arguments that no longer exist.
  if (is.null(col_names) && is.character(col_means)) {
    stop("col_means must be numeric; a character vector suggests it was given to ",
         "the wrong argument.", call. = FALSE)
  }
  if (!is.null(col_names) && !is.character(col_names)) {
    stop("col_names must be a character vector.", call. = FALSE)
  }
  n_out <- arch$output_dim
  scaling <- list(col_means = col_means, col_sds = col_sds, col_names = col_names)
  supplied <- scaling[!vapply(scaling, is.null, logical(1))]
  # Mutual consistency first: three vectors that disagree with each other are an
  # error, and reporting an output_dim mismatch on top of that would bury it.
  lens <- vapply(supplied, length, integer(1))
  if (length(lens) > 1L && length(unique(lens)) > 1L) {
    stop("col_means, col_sds and col_names must all have the same length (got ",
         paste(names(lens), lens, sep = " = ", collapse = ", "), ").",
         call. = FALSE)
  }
  if (!is.null(n_out)) {
    off <- names(supplied)[
      vapply(supplied, function(v) length(v) != n_out, logical(1))
    ]
    if (length(off) > 0L) {
      warning("These scaling metadata entries have length ",
              paste(vapply(supplied[off], length, integer(1)), collapse = ", "),
              " but the model's output_dim is ", n_out, ": ",
              paste(off, collapse = ", "), ". probotSamplePost*() and the ",
              "assessment functions unscale with these vectors.", call. = FALSE)
    }
  }

  metadata <- c(
    list(
      version         = "1.0",
      package_version = as.character(utils::packageVersion("ProBot")),
      model_type      = model_type
    ),
    # MDN/Point carry their depth as the hidden_dims vector and flows as the
    # scalar hidden_dim, matching the keys probotLoad() reads for each branch.
    # Writing each model's shape under its own key is what keeps a flow
    # checkpoint from ever holding a three-element vector again.
    arch,
    list(
      col_means       = col_means,
      col_sds         = col_sds,
      col_names       = col_names,
      training_history = training_history
    ),
    extra_metadata
  )

  torch_save(
    list(
      model_state     = state_dict,
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
    flow_style <- match.arg(tolower(flow_style), c("realnvp", "maf", "nsf",
                                                   "couple", "autoreg"))
    flow_style <- switch(flow_style,
      couple   = "realnvp",
      autoreg  = "maf",
      flow_style
    )
  }

  # For flow checkpoints saved before the dim_theta/dim_x -> input_dim/output_dim
  # rename, normalise legacy metadata keys so reconstruction still works. The
  # normalised values are written back to the checkpoint as well, so the metadata
  # returned always uses the current vocabulary -- the same promise probotSave()
  # now makes about the architecture fields.
  if (model_type == "flow") {
    if (is.null(meta$input_dim)  && !is.null(meta$dim_x)) {
      meta$input_dim <- meta$dim_x
      checkpoint$metadata$input_dim <- meta$dim_x
    }
    if (is.null(meta$output_dim) && !is.null(meta$dim_theta)) {
      meta$output_dim <- meta$dim_theta
      checkpoint$metadata$output_dim <- meta$dim_theta
    }
    # Normalise legacy flow_style values saved before the rename.
    if (!is.null(meta$flow_style)) {
      norm <- switch(meta$flow_style,
        couple  = "realnvp",
        autoreg = "maf",
        meta$flow_style
      )
      meta$flow_style <- norm
      checkpoint$metadata$flow_style <- norm
    }
  }

  required <- .probotRequired[[model_type]]
  missing_fields <- required[
    vapply(required, function(field) is.null(meta[[field]]), logical(1))
  ]
  if (length(missing_fields) > 0) {
    stop(
      "Cannot reconstruct model (type='", model_type, "'): metadata missing ",
      paste(missing_fields, collapse = ", "),
      ". These dimensions cannot be recovered from the weight shapes. Re-save ",
      "the model with a current probotSave(), or use probotLoadModel() with an ",
      "explicit model_skeleton."
    )
  }

  # Architecture from the tensors, not from the metadata. For flows the
  # state_dict fully determines style, depth and width, so it overrides
  # whatever probotSave() recorded; see .probotFlowArchFromState().
  if (model_type == "flow") {
    arch <- .probotFlowArchFromState(checkpoint$model_state,
                                     output_dim = meta$output_dim)
    # Rename the inferred `style` field to the metadata name it corresponds to,
    # so the merge loop below writes to meta$flow_style rather than inventing a
    # meta$style nobody reads.
    names(arch)[names(arch) == "style"] <- "flow_style"
    # An explicit caller override is not a disagreement with the weights, and it
    # is already applied where the skeleton is built, so leave it out of the
    # merge/warn loop entirely.
    if (!is.null(flow_style)) {
      arch$flow_style <- NULL
    }
    for (fld in names(arch)) {
      # An inference that came back NULL means "could not determine", not
      # "should be unset" -- leave the metadata value alone in that case.
      inf <- arch[[fld]]
      if (is.null(inf)) next
      met <- meta[[fld]]
      # Integer metadata versus double tensor shapes must not count as a
      # disagreement, so compare values rather than using identical().
      agree <- !is.null(met) &&
        ((is.numeric(met) && is.numeric(inf) &&
            length(met) == length(inf) &&
            isTRUE(all.equal(met, inf, check.attributes = FALSE))) ||
           isTRUE(identical(as.character(met), as.character(inf))))
      if (!agree) {
        if (!is.null(met)) {
          warning("Checkpoint metadata ", fld, " = ",
                  paste(met, collapse = ", "), " disagrees with the saved ",
                  "weights (", paste(inf, collapse = ", "), "); reconstructing ",
                  "from the weights.", call. = FALSE)
        }
        # Write the correction into the metadata that is returned as well as the
        # local copy used to build the skeleton: the caller's metadata should
        # describe the model they actually got.
        meta[[fld]] <- inf
        checkpoint$metadata[[fld]] <- inf
      }
    }
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
      input_dim   = meta$input_dim,
      output_dim  = meta$output_dim,
      n_layers    = if (!is.null(meta$n_layers)) meta$n_layers else 4,
      hidden_dim  = if (!is.null(meta$hidden_dim)) meta$hidden_dim else 32,
      # Pass n_blocks through untouched (NULL when absent) so probotMakeFlow()
      # can map the saved n_layers onto it for style = "maf". Substituting a
      # non-NULL default here would win over n_layers and reconstruct a flow
      # with the wrong depth, failing the state_dict load.
      n_blocks  = meta$n_blocks,
      n_layers_per_block = if (!is.null(meta$n_layers_per_block)) meta$n_layers_per_block else 2,
      n_bins = if (!is.null(meta$n_bins)) meta$n_bins else 8,
      tail_bound = if (!is.null(meta$tail_bound)) meta$tail_bound else 3,
      soft_clamp  = if (!is.null(meta$soft_clamp)) meta$soft_clamp else 3,
      # Priority: explicit override > saved metadata > "realnvp" (legacy default).
      style       = if (!is.null(flow_style)) flow_style else
                    if (!is.null(meta$flow_style)) meta$flow_style else "realnvp",
      # Absent in checkpoints saved before the location head existed, which is
      # exactly the FALSE default. The state_dict key namespaces differ between
      # headed and plain flows ("base_flow." prefix), so a wrong guess here
      # fails the strict load below rather than silently mis-reconstructing.
      loc_head    = isTRUE(meta$loc_head),
      loc_hidden_dims = if (!is.null(meta$loc_hidden_dims)) meta$loc_hidden_dims
                        else c(128, 128),
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
