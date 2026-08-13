library(testthat)
library(ProBot)
library(torch)

test_that("probotDataLoader returns a dataloader object", {
  dat <- matrix(rnorm(100), 20, 5)
  tgt <- matrix(rnorm(40), 20, 2)
  dl <- probotDataLoader(dat, tgt, batch = 8)
  expect_true(!is.null(dl$dataset))
})

test_that("probotDataLoader respects train_idx", {
  dat <- matrix(rnorm(100), 20, 5)
  tgt <- matrix(rnorm(40), 20, 2)
  idx <- 1:10
  dl <- probotDataLoader(dat, tgt, batch = 8, train_idx = idx)
  expect_equal(length(dl$dataset), 10L)
})

test_that("probotDataLoader tensors are on correct device", {
  dat <- matrix(rnorm(100), 20, 5)
  tgt <- matrix(rnorm(40), 20, 2)
  dl <- probotDataLoader(dat, tgt, batch = 8, device = "cpu")
  expect_equal(
    as.character(dl$dataset$tensors[[1]]$device$type),
    "cpu"
  )
  expect_equal(
    as.character(dl$dataset$tensors[[2]]$device$type),
    "cpu"
  )
})

test_that("probotDataLoader with subset produces correct dataset size", {
  dat <- matrix(rnorm(200), 40, 5)
  tgt <- matrix(rnorm(80), 40, 2)
  idx <- sample(40, 25)
  dl <- probotDataLoader(dat, tgt, batch = 10, train_idx = idx)
  expect_equal(length(dl$dataset), 25L)
})
