library(testthat)
library(ProBot)
library(torch)

## ---- MDN save/load tests ----

test_that("probotSave creates a file (mdn)", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  probotSave(mdl, opt, filename = tmp)

  expect_true(file.exists(tmp))
})

test_that("probotSave warns when a legacy module's dimensions are ambiguous", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  # A bare "mlp"-classed module with a single layer. rows = K * (2D + 1) has
  # several valid splits and nothing on the object resolves them, so this is the
  # one case where probotSave() still cannot fill in the metadata.
  bare <- nn_module("mlp", initialize = function() {
    self$layers <- nn_module_list(list(nn_linear(3, 15)))
    self$to(device = "cpu")
  })()

  expect_warning(
    probotSave(bare, filename = tmp, model_type = "mdn"),
    "Checkpoint metadata missing"
  )
})

test_that("probotLoad auto-reconstructs MDN from checkpoint", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp)

  res <- probotLoad(tmp)
  expect_true("model" %in% names(res))
  expect_true("metadata" %in% names(res))
  expect_true(inherits(res$model, "nn_module"))
})

test_that("probotLoadModel loads into explicit skeleton (mdn)", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp)

  skeleton <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  res <- probotLoadModel(tmp, skeleton)
  expect_true("model" %in% names(res))
  expect_true(inherits(res$model, "nn_module"))
})

test_that("probotLoadModel errors without skeleton", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp)

  expect_error(probotLoadModel(tmp))
})

test_that("probotLoad errors on checkpoint without metadata", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  torch_save(list(model_state = list(), optimizer_state = NULL, metadata = NULL), tmp)

  expect_error(probotLoad(tmp), "metadata")
})

test_that("probotLoad errors when required metadata fields missing", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  torch_save(
    list(
      model_state = list(),
      optimizer_state = NULL,
      metadata = list(version = "1.0", model_type = "mdn")
    ),
    tmp
  )

  expect_error(probotLoad(tmp), "Cannot reconstruct model")
})

test_that("mdn save-load round-trip preserves predictions", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()
  probotSave(mdl, filename = tmp)

  inp <- matrix(rnorm(5 * 3), 5, 3)
  pred_orig <- probotPredictMDN(inp, mdl, 3)

  res <- probotLoad(tmp)
  res$model$eval()
  pred_loaded <- probotPredictMDN(inp, res$model, 3)

  expect_equal(
    as.array(pred_orig$mu$to(device = "cpu")),
    as.array(pred_loaded$mu$to(device = "cpu")),
    tolerance = 1e-6
  )
})

test_that("probotSave with different activation names (mdn)", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), activation = nnf_gelu, device = "cpu")()
  probotSave(mdl, filename = tmp)

  # The activation name is recovered from the stored function by identity.
  res <- probotLoad(tmp)
  expect_equal(res$metadata$activation, "gelu")
  expect_true(inherits(res$model, "nn_module"))
})

## ---- Point save/load tests ----

test_that("probotSave and Load work for Point model", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakePoint(3, 2, hidden_dims = c(8, 8), device = "cpu")()
  # model_type omitted: detected from the module's class tag.
  probotSave(mdl, filename = tmp)

  res <- probotLoad(tmp)
  expect_equal(res$metadata$model_type, "point")
  expect_equal(res$metadata$input_dim, 3L)
  expect_equal(res$metadata$output_dim, 2L)
  expect_equal(as.integer(res$metadata$hidden_dims), c(8L, 8L))
  expect_true(inherits(res$model, "nn_module"))
})

test_that("point save-load round-trip preserves predictions", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakePoint(3, 2, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()
  probotSave(mdl, filename = tmp)

  inp <- matrix(rnorm(5 * 3), 5, 3)
  pred_orig <- probotPredictPoint(inp, mdl)

  res <- probotLoad(tmp, device = "cpu")
  res$model$eval()
  pred_loaded <- probotPredictPoint(inp, res$model)

  expect_equal(as.matrix(pred_orig), as.matrix(pred_loaded), tolerance = 1e-6)
})

test_that("probotLoad handles old mdn checkpoints without model_type", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()

  # Save a checkpoint without model_type (old v1.0 format)
  torch_save(
    list(
      model_state     = mdl$state_dict(),
      optimizer_state = NULL,
      metadata = list(
        version        = "1.0",
        mdn_components = 3,
        input_dim      = 3,
        output_dim     = 2,
        hidden_dims    = c(8, 8),
        activation     = "relu",
        dropout        = 0
      )
    ),
    tmp
  )

  res <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res$model, "nn_module"))
  expect_equal(res$metadata$model_type, NULL)  # old checkpoint keeps NULL in metadata
})

## ---- Flow save/load tests ----

test_that("probotSave and Load work for Flow model", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  probotSave(mdl, filename = tmp)

  res <- probotLoad(tmp, device = "cpu")
  expect_equal(res$metadata$model_type, "flow")
  expect_equal(res$metadata$flow_style, "realnvp")
  expect_equal(res$metadata$n_layers, 2L)
  expect_equal(res$metadata$hidden_dim, 8L)
  expect_true(inherits(res$model, "nn_module"))
})

test_that("flow save-load round-trip preserves forward output", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()
  probotSave(mdl, filename = tmp)

  theta <- torch_randn(2, 4, device = "cpu")
  ctx   <- torch_randn(2, 3, device = "cpu")
  out_orig <- mdl$forward(theta, ctx)

  res <- probotLoad(tmp, device = "cpu")
  res$model$eval()
  out_loaded <- res$model$forward(theta, ctx)

  expect_equal(
    as.array(out_orig$z),
    as.array(out_loaded$z),
    tolerance = 1e-6
  )
  expect_equal(
    as.array(out_orig$log_det_jac),
    as.array(out_loaded$log_det_jac),
    tolerance = 1e-6
  )
})

test_that("probotSave/probotLoad round-trips flow_style for both architectures", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  # MAF: flow_style should be inferred from the model class and round-trip
  mdl_a <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 16,
                          device = "cpu", style = "maf")()
  probotSave(mdl_a, filename = tmp)

  cp <- torch_load(tmp, device = "cpu")
  expect_equal(cp$metadata$flow_style, "maf")
  # MAF depth is recorded in blocks, and n_layers stays NULL so it cannot
  # contradict the block count on load.
  expect_equal(cp$metadata$n_blocks, 2L)
  expect_null(cp$metadata$n_layers)

  res_a <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res_a$model, "probotFlowMAF"))

  # RealNVP: the default architecture
  mdl_c <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 8,
                          device = "cpu")()
  probotSave(mdl_c, filename = tmp)

  cp2 <- torch_load(tmp, device = "cpu")
  expect_equal(cp2$metadata$flow_style, "realnvp")

  res_c <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res_c$model, "probotFlowRealNVP"))
})

test_that("probotSave/probotLoad round-trips NSF configuration", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)
  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 8,
                        n_bins = 6, tail_bound = 4, style = "nsf", device = "cpu")()
  probotSave(mdl, filename = tmp)

  res <- probotLoad(tmp, device = "cpu")
  expect_equal(res$metadata$flow_style, "nsf")
  # tail_bound changes the maths, not the tensor sizes, so the stored scalar is
  # the only source for it; n_bins is additionally recoverable from the shape.
  expect_equal(res$metadata$n_bins, 6L)
  expect_equal(res$metadata$tail_bound, 4)
  expect_true(inherits(res$model, "probotFlowNSF"))
})

test_that("probotLoad defaults to RealNVP when flow_style metadata is absent", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  # Save a legacy-style checkpoint with no flow_style field.
  torch_save(
    list(
      model_state = mdl$state_dict(),
      optimizer_state = NULL,
      metadata = list(version = "1.0", model_type = "flow",
                      input_dim = 3, output_dim = 4, n_layers = 2, hidden_dim = 8)
    ),
    tmp
  )

  res <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res$model, "probotFlowRealNVP"))
})

test_that("probotSave rejects the architecture arguments it now infers", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2,
                        hidden_dim = 8, device = "cpu")()

  # Every one of these used to be silently acceptable and could contradict the
  # weights -- the exact failure mode behind a real checkpoint that shipped an
  # MDN-shaped hidden_dim vector for a 19-layer RealNVP.
  for (arg in c("hidden_dim", "n_layers", "input_dim", "output_dim",
                "flow_style", "soft_clamp", "n_bins", "tail_bound",
                "n_blocks", "n_layers_per_block")) {
    expect_error(
      do.call(probotSave, c(list(mdl, filename = tmp), setNames(list(99), arg))),
      paste0("no longer takes '", arg, "'"), fixed = TRUE,
      info = paste0("retired argument: ", arg)
    )
  }

  # MDN/Point-only fields are covered too, and a genuinely misspelled argument is
  # reported rather than absorbed.
  expect_error(probotSave(mdl, filename = tmp, hidden_dims = c(8, 8)),
               "no longer takes 'hidden_dims'")
  expect_error(probotSave(mdl, filename = tmp, col_mans = 1:4),
               "unrecognised argument")

  # model_type is still accepted when it agrees, and refused when it does not.
  expect_silent(probotSave(mdl, filename = tmp, model_type = "flow"))
  expect_error(probotSave(mdl, filename = tmp, model_type = "mdn"),
               "describes a flow model")
})

test_that("probotSave checks the scaling metadata it cannot infer", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakePoint(3, 4, hidden_dims = c(8, 8), device = "cpu")()

  # Scaling vectors are not knowable from the model, so they remain arguments --
  # but a length that contradicts output_dim is caught here instead of at
  # sampling time.
  expect_warning(
    probotSave(mdl, filename = tmp, col_means = 1:3),
    "length 3 but the model's output_dim is 4"
  )
  expect_error(
    probotSave(mdl, filename = tmp, col_means = 1:4, col_sds = 1:3),
    "same length"
  )
  # col_names belongs in col_names; a character vector in the numeric slot is a
  # positional-call mistake that would otherwise save silently.
  expect_error(
    probotSave(mdl, filename = tmp, col_means = letters[1:4]),
    "col_means must be numeric"
  )
  expect_error(
    probotSave(mdl, filename = tmp, col_names = 1:4),
    "col_names must be a character vector"
  )
  # Consistent metadata passes without comment.
  expect_silent(probotSave(mdl, filename = tmp, col_means = 1:4, col_sds = rep(2, 4),
                           col_names = paste0("p", 1:4)))
})

test_that("probotSave refuses more than four positional arguments", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()

  # Removing the architecture formals moves col_means into the 5th slot, so a
  # legacy positional call would bind mdn_components to it. The lengths differ
  # here, but the shape check is not what makes this safe -- the guard on the
  # call itself is, so the message must name the position count rather than
  # imply a scaling problem.
  expect_error(
    probotSave(mdl, NULL, tmp, "mdn", 3, 3, 2),
    "only 4 arguments .* may be given positionally"
  )
  expect_false(file.exists(tmp))

  # The four leading formals, and anything named, stay legal.
  expect_silent(probotSave(mdl, NULL, tmp, "mdn"))
  expect_silent(probotSave(mdl, NULL, tmp))
  expect_silent(probotSave(mdl, filename = tmp))
  # do.call() with an all-named list reports no positional arguments.
  expect_silent(do.call(probotSave, list(model = mdl, filename = tmp)))
})

test_that("probotSave always writes the full architecture key set", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  # `arch$x <- NULL` deletes the element rather than unsetting it, and a missing
  # key makes metadata$n_layers partially match n_layers_per_block -- an
  # invisible mis-read. Every architecture must therefore emit every key.
  full <- c("mdn_components", "input_dim", "output_dim", "n_layers", "hidden_dim",
            "hidden_dims", "n_blocks", "n_layers_per_block", "n_bins",
            "tail_bound", "soft_clamp", "flow_style", "loc_head",
            "loc_hidden_dims", "activation", "dropout")

  models <- list(
    mdn  = probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")(),
    point = probotMakePoint(3, 2, hidden_dims = c(8, 8), device = "cpu")(),
    realnvp = probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 8, device = "cpu")(),
    maf  = probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 16, style = "maf",
                          device = "cpu")(),
    nsf  = probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 16, style = "nsf",
                          device = "cpu")(),
    headed = probotMakeFlow(3, 4, n_layers = 2, hidden_dim = 16, style = "nsf",
                            loc_head = TRUE, device = "cpu")()
  )
  for (nm in names(models)) {
    probotSave(models[[nm]], filename = tmp)
    keys <- names(torch_load(tmp, device = "cpu")$metadata)
    expect_true(all(full %in% keys), info = nm)
  }

  # The specific trap: a MAF records depth in n_blocks and leaves n_layers NULL,
  # so a partial match would silently report the per-block layer count as the
  # block depth. An exact NULL key blocks that.
  probotSave(models$maf, filename = tmp)
  md <- torch_load(tmp, device = "cpu")$metadata
  expect_true("n_layers" %in% names(md))
  expect_null(md$n_layers)
  expect_equal(md$n_blocks, 2L)
  expect_equal(md$n_layers_per_block, 2L)
})

test_that("probotSave recovers what it can from weights when the model stores nothing", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  # Stands in for a model built by hand, or by a pre-change constructor: only
  # state_dict shapes are available. For a Point MLP the shapes give every
  # dimension exactly, so this must succeed in silence.
  bare_point <- nn_module("mlp_point", initialize = function() {
    self$layers <- nn_module_list(list(nn_linear(3, 8), nn_linear(8, 2)))
    self$activation_fn <- torch::nnf_gelu
    self$dropout_rate <- 0.2
    self$to(device = "cpu")
  })()

  expect_silent(probotSave(bare_point, filename = tmp))
  md <- torch_load(tmp, device = "cpu")$metadata
  expect_equal(md$model_type, "point")
  expect_equal(md$input_dim, 3L)
  expect_equal(as.integer(md$hidden_dims), 8L)
  expect_equal(md$output_dim, 2L)
  expect_equal(md$activation, "gelu")
  expect_equal(md$dropout, 0.2)
  expect_s3_class(probotLoad(tmp, device = "cpu")$model, "nn_module")

  # A RealNVP cannot: its conditioner input is d1 + input_dim, so the split
  # between the two is not recoverable from shapes. The save says so, naming
  # probotLoadModel() as the way out, rather than writing a guess.
  bare_rnvp <- nn_module("probotFlowRealNVP", initialize = function() {
    self$n_layers <- 2L
    for (i in 1:2) {
      self[[paste0("RealNVP_layer_", i)]] <- .probotRealNVPLayer(3, 4, 8,
                                                                 soft_clamp = 3)
    }
    self$to(device = "cpu")
  })()

  expect_warning(
    probotSave(bare_rnvp, filename = tmp),
    "Checkpoint metadata missing input_dim, output_dim"
  )
  md2 <- torch_load(tmp, device = "cpu")$metadata
  expect_equal(md2$flow_style, "realnvp")
  expect_equal(md2$n_layers, 2L)
  expect_equal(md2$hidden_dim, 8L)
})

test_that("probotLoad returns metadata in current vocabulary", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2,
                        n_blocks = 3, hidden_dim = 16, style = "maf",
                        device = "cpu")()
  # A checkpoint from before flow_style and the dim_x/dim_theta rename.
  torch_save(
    list(model_state = mdl$state_dict(), optimizer_state = NULL,
         metadata = list(version = "1.0", model_type = "flow",
                         dim_x = 3, dim_theta = 4, flow_style = "autoreg")),
    tmp
  )

  res <- probotLoad(tmp, device = "cpu")
  expect_s3_class(res$model, "probotFlowMAF")
  # The legacy spellings are normalised in the returned metadata too, so callers
  # never see a value they would have to translate again.
  expect_equal(res$metadata$flow_style, "maf")
  expect_equal(res$metadata$input_dim, 3)
  expect_equal(res$metadata$output_dim, 4)
})

test_that("probotLoad errors for flow with missing output_dim", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  torch_save(
    list(
      model_state = list(),
      optimizer_state = NULL,
      metadata = list(version = "1.0", model_type = "flow", input_dim = 3)
    ),
    tmp
  )

  expect_error(probotLoad(tmp), "Cannot reconstruct model")
})

## ---- Flow architecture read from the saved weights ----
#
# A real checkpoint shipped hidden_dim = c(736, 1472, 736) (an MDN-shaped vector)
# and n_layers = NULL for a 19-layer, hidden-512 RealNVP; reconstructing from that
# metadata died inside torch with "You should specify a single size argument".
# probotSave() now derives this from the model and can no longer be told
# otherwise, but checkpoints written by older builds still carry whatever their
# caller passed, so probotLoad() reads the architecture from the state_dict.

test_that("probotLoad recovers a RealNVP whose metadata is wrong or missing", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 5,
                        hidden_dim = 12, device = "cpu")()
  dev <- mdl$parameters[[1]]$device
  theta <- torch_randn(2, 4, device = dev)
  ctx <- torch_randn(2, 3, device = dev)
  mdl$eval()
  ref <- mdl$forward(theta, ctx)

  # Write deliberately corrupt metadata: vector hidden_dim, NULL n_layers.
  torch_save(
    list(
      model_state = mdl$state_dict(),
      optimizer_state = NULL,
      metadata = list(version = "1.0", model_type = "flow",
                      input_dim = 3, output_dim = 4,
                      n_layers = NULL, hidden_dim = c(8, 16, 8),
                      flow_style = "realnvp")
    ),
    tmp
  )

  expect_warning(res <- probotLoad(tmp, device = "cpu"), "disagrees with the")
  expect_equal(res$model$n_layers, 5)
  expect_equal(as.integer(res$model$RealNVP_layer_1$shift_scale_net[[1]]$weight$shape[1]), 12)
  res$model$eval()
  out <- res$model$forward(theta, ctx)
  expect_equal(as.array(out$z), as.array(ref$z), tolerance = 1e-6)
})

test_that("probotLoad infers flow style from weights when metadata lacks it", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  for (style in c("realnvp", "maf", "nsf")) {
    mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2,
                          hidden_dim = 16, style = style, device = "cpu")()
    probotSave(mdl, filename = tmp, model_type = "flow")
    # Strip the style/depth/width metadata so only the tensors identify it.
    cp <- torch_load(tmp, device = "cpu")
    cp$metadata$flow_style <- NULL
    cp$metadata$n_layers <- NULL
    cp$metadata$n_blocks <- NULL
    cp$metadata$hidden_dim <- NULL
    cp$metadata$n_bins <- NULL
    torch_save(cp, tmp)

    res <- probotLoad(tmp, device = "cpu")
    expect_s3_class(res$model, paste0("probotFlow", switch(style,
      realnvp = "RealNVP", maf = "MAF", nsf = "NSF")))
    expect_equal(res$metadata$flow_style, style)
  }
})

test_that("probotLoad infers MAF depth and NSF bins from the weights", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 3, n_blocks = 4,
                        hidden_dim = 24, style = "maf", device = "cpu")()
  torch_save(
    list(model_state = mdl$state_dict(), optimizer_state = NULL,
         metadata = list(version = "1.0", model_type = "flow",
                         input_dim = 3, output_dim = 4, flow_style = "maf")),
    tmp
  )
  res <- probotLoad(tmp, device = "cpu")
  expect_equal(res$model$n_blocks, 4)
  expect_equal(res$metadata$n_blocks, 4)
  expect_equal(res$metadata$n_layers_per_block, 2)

  # Odd output_dim -> d2 = 3, so the conditioner's last layer has
  # d2 * (3 * n_bins - 1) = 3 * 17 = 51 rows for n_bins = 6.
  mdl2 <- probotMakeFlow(input_dim = 3, output_dim = 5, n_layers = 2,
                         hidden_dim = 16, n_bins = 6, style = "nsf",
                         device = "cpu")()
  torch_save(
    list(model_state = mdl2$state_dict(), optimizer_state = NULL,
         metadata = list(version = "1.0", model_type = "flow",
                         input_dim = 3, output_dim = 5, flow_style = "nsf")),
    tmp
  )
  res2 <- probotLoad(tmp, device = "cpu")
  expect_equal(res2$metadata$n_bins, 6)
  expect_equal(res2$model$n_layers, 2)
})

test_that("an explicit flow_style override is not reported as a disagreement", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2,
                        hidden_dim = 8, device = "cpu")()
  probotSave(mdl, filename = tmp, model_type = "flow")

  # Overriding to a style the weights are not suppresses the warning and still
  # loads; overriding to the matching style is a no-op.
  expect_silent(probotLoad(tmp, device = "cpu", flow_style = "realnvp"))
  expect_error(probotLoad(tmp, device = "cpu", flow_style = "maf"))
})

test_that("probotLoad prefers the weights for a headed flow's style", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(input_dim = 3, output_dim = 4, n_layers = 2,
                        hidden_dim = 16, n_bins = 6, style = "nsf",
                        loc_head = TRUE, loc_hidden_dims = c(8, 8),
                        device = "cpu")()
  # Corrupt the recorded style but keep the head flags; the base_flow.*
  # namespace plus the tensor shapes must still reconstruct a headed NSF.
  probotSave(mdl, filename = tmp, model_type = "flow")
  cp <- torch_load(tmp, device = "cpu")
  cp$metadata$flow_style <- "realnvp"
  torch_save(cp, tmp)

  expect_warning(res <- probotLoad(tmp, device = "cpu"), "disagrees with the")
  expect_s3_class(res$model, "probotFlowLoc")
  expect_s3_class(res$model$base_flow, "probotFlowNSF")
  # The returned metadata describes the model the caller actually got.
  expect_equal(res$metadata$flow_style, "nsf")
})
