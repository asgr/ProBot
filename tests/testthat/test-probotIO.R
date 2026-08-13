library(testthat)
library(ProBot)
library(torch)

test_that("probotSave creates a file", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  probotSave(mdl, opt, filename = tmp,
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

  expect_true(file.exists(tmp))
})

test_that("probotSave warns on missing metadata", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()

  expect_warning(
    probotSave(mdl, filename = tmp),
    "Checkpoint metadata missing"
  )
})

test_that("probotLoad auto-reconstructs model from checkpoint", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp,
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

  res <- probotLoad(tmp)
  expect_true("model" %in% names(res))
  expect_true("metadata" %in% names(res))
  expect_true(inherits(res$model, "nn_module"))
})

test_that("probotLoadModel loads into explicit skeleton", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  probotSave(mdl, filename = tmp,
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
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "relu", dropout = 0)

  expect_error(probotLoadModel(tmp))
})

test_that("probotLoad errors on checkpoint without metadata", {
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  # Save a bare dict without metadata
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
      metadata = list(version = "1.0")  # missing mdn_components, input_dim, output_dim
    ),
    tmp
  )

  expect_error(probotLoad(tmp), "Cannot reconstruct model")
})

test_that("save-load round-trip preserves predictions", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()
  probotSave(mdl, filename = tmp,
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

test_that("probotSave with different activation names", {
  set.seed(42)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)

  mdl <- probotMakeMDN(3, 2, 3, hidden_dims = c(8, 8), activation = nnf_gelu, device = "cpu")()
  probotSave(mdl, filename = tmp,
             mdn_components = 3, input_dim = 3, output_dim = 2,
             hidden_dims = c(8, 8), activation_name = "gelu", dropout = 0)

  res <- probotLoad(tmp)
  expect_true(inherits(res$model, "nn_module"))
})
