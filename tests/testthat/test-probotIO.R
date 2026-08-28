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

  probotSave(mdl, opt, filename = tmp,
             model_type = "mdn",
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

  expect_true(file.exists(tmp))
})

test_that("probotSave warns on missing metadata (mdn)", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()

  expect_warning(
    probotSave(mdl, filename = tmp, model_type = "mdn"),
    "Checkpoint metadata missing"
  )
})

test_that("probotLoad auto-reconstructs MDN from checkpoint", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp,
             model_type = "mdn",
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

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
  probotSave(mdl, filename = tmp,
             model_type = "mdn",
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

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
  probotSave(mdl, filename = tmp,
             model_type = "mdn",
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

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
  probotSave(mdl, filename = tmp,
             model_type = "mdn",
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

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
  probotSave(mdl, filename = tmp,
             model_type = "mdn",
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "gelu", dropout = 0)

  res <- probotLoad(tmp)
  expect_true(inherits(res$model, "nn_module"))
})

## ---- Point save/load tests ----

test_that("probotSave and Load work for Point model", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakePoint(3, 2, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp,
             model_type = "point",
             input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

  res <- probotLoad(tmp)
  expect_equal(res$metadata$model_type, "point")
  expect_true(inherits(res$model, "nn_module"))
})

test_that("point save-load round-trip preserves predictions", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakePoint(3, 2, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()
  probotSave(mdl, filename = tmp,
             model_type = "point",
             input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

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

  mdl <- probotMakeFlow(dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8, device = "cpu")()
  probotSave(mdl, filename = tmp,
             model_type = "flow",
             dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8, soft_clamp = 3)

  res <- probotLoad(tmp, device = "cpu")
  expect_equal(res$metadata$model_type, "flow")
  expect_true(inherits(res$model, "nn_module"))
})

test_that("flow save-load round-trip preserves forward output", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8, device = "cpu")()
  mdl$eval()
  probotSave(mdl, filename = tmp,
             model_type = "flow",
             dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8, soft_clamp = 3)

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

  # autoreg: flow_style should be inferred from the model class and round-trip
  mdl_a <- probotMakeFlow(dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 16,
                          device = "cpu", style = "autoreg")()
  probotSave(mdl_a, filename = tmp, model_type = "flow",
             dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 16, soft_clamp = 3)

  cp <- torch_load(tmp, device = "cpu")
  expect_equal(cp$metadata$flow_style, "autoreg")

  res_a <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res_a$model, "probotFlowAutoReg"))

  # couple: the default architecture
  mdl_c <- probotMakeFlow(dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8,
                          device = "cpu")()
  probotSave(mdl_c, filename = tmp, model_type = "flow",
             dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8, soft_clamp = 3)

  cp2 <- torch_load(tmp, device = "cpu")
  expect_equal(cp2$metadata$flow_style, "couple")

  res_c <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res_c$model, "probotMakeFlowCouple"))
})

test_that("probotLoad defaults to couple when flow_style metadata is absent", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8, device = "cpu")()
  # Save a legacy-style checkpoint with no flow_style field.
  torch_save(
    list(
      model_state = mdl$state_dict(),
      optimizer_state = NULL,
      metadata = list(version = "1.0", model_type = "flow",
                      dim_theta = 4, dim_x = 3, n_layers = 2, hidden_dim = 8)
    ),
    tmp
  )

  res <- probotLoad(tmp, device = "cpu")
  expect_true(inherits(res$model, "probotMakeFlowCouple"))
})

test_that("probotSave warns on missing metadata per model type", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeFlow(dim_theta = 2, dim_x = 3, n_layers = 2, hidden_dim = 8, device = "cpu")()

  expect_warning(
    probotSave(mdl, filename = tmp, model_type = "flow"),
    "Checkpoint metadata missing"
  )
})

test_that("probotLoad errors for flow with missing dim_theta", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  torch_save(
    list(
      model_state = list(),
      optimizer_state = NULL,
      metadata = list(version = "1.0", model_type = "flow", dim_x = 3)
    ),
    tmp
  )

  expect_error(probotLoad(tmp), "Cannot reconstruct model")
})
