probotSave <- function(
    model,
    optimizer = NULL,
    filename = "temp.pt",
    ...
){

  optimizer_state <- NULL

  if(!is.null(optimizer)){
    optimizer_state <- optimizer$state_dict()
  }

  torch_save(
    c(
      list(
        model = model$state_dict(),
        optimizer = optimizer_state
      ),
      list(...)
    ),
    filename
  )
}

probotLoad = function(filename, model_skeleton){
  model_skeleton$load_state_dict(
    torch_load(filename)
  )
  return(model_skeleton)
}
