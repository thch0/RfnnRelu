#normalize
normalize <- function(x,  minLim = min(x),maxLim = max(x)) {
  return((x - minLim) / (maxLim - minLim))
}
columnify <- function(x){
  if (!(is.matrix(x))) {x<-as.matrix(x)}
  unq <-  unique(x)
  out <- matrix(0,nrow = nrow(x),ncol=length(unq),dimnames = list(c(),unq))
  for(i in 1:length(unq)){ 
    out[which(x[,1] == unq[i]),i] <- 1
  }
  return(out)
}

fnnClassification <- function(name) {
  iterations <- -Inf
  get_iterations <- function(){return(iterations)}
  name <- name
  get_name <- function(){return(name)}
  get_description <- function(){cat(name,"\n",
                                    "iterations : ", iterations,"\n",
                                    "loss : ",loss,"\n")}
  
  loss <- Inf
  get_loss <- function(){return(loss)}
  
  model <- NULL
  predictions <- NULL
  get_predictions <- function(){return(predictions)}
  
  # Fonction d'activation sigmoïde
  sigmoid <- function(x) {
    1 / (1 + exp(-x))
  }
  
  softmax <- function(z) {
    exp_scores <- exp(z)
    return(exp_scores / rowSums(exp_scores))
  }
  
  # Fonction d'activation (Leaky ReLU)
  leaky_relu <- function(z, alpha=0.01) {
    pmax(alpha * z, z)
  }
  # Loss calculation - Cross Entropy
  calculate_ce_loss <- function(predictions, ground_truth) {
    loss <- -sum(ground_truth * log(predictions + 1e-15)) / nrow(ground_truth)
    return(loss)
  }
  # Loss calculation - Mean Squared Error
  calculate_mse_loss <- function(predictions, ground_truth) {
    loss <- mean((predictions - ground_truth)^2)
    return(loss)
  }
  
  # Loss calculation - Mean Absolute Error
  calculate_mae_loss <- function(predictions, ground_truth) {
    loss <- mean(abs(predictions - ground_truth))
    return(loss)
  }
  
  # Loss calculation - Huber Loss
  calculate_huber_loss <- function(predictions, ground_truth, delta = 1.0) {
    loss <- mean(ifelse(abs(predictions - ground_truth) <= delta, 0.5 * (predictions - ground_truth)^2, delta * (abs(predictions - ground_truth) - 0.5 * delta)))
    return(loss)
  }
  
  
  # Initialisation des poids et biais du réseau
  initialize_weights <- function(input_size, hidden_size, output_size) {
    # W1 <- matrix(runif(input_size * hidden_size), nrow = input_size,ncol = hidden_size)
    # b1 <- matrix(runif(hidden_size), nrow = 1, ncol = hidden_size)
    # W2 <- matrix(runif(hidden_size * output_size),nrow = hidden_size,ncol = output_size)
    # b2 <- matrix(runif(output_size), nrow = 1, ncol = output_size)
    W1 <- matrix(rnorm(input_size * hidden_size, mean = 0, sd = sqrt(2 / input_size)), nrow = input_size, ncol = hidden_size)
    b1 <- matrix(0, nrow = 1, ncol = hidden_size)
    W2 <- matrix(rnorm(hidden_size * output_size, mean = 0, sd = sqrt(2 / hidden_size)), nrow = hidden_size, ncol = output_size)
    b2 <- matrix(0, nrow = 1, ncol = output_size)
    return(list(W1 = W1,
                b1 = b1,
                W2 = W2,
                b2 = b2))
  }
  
  forward_propagation <- function(X, weights) {
    Z1 <- X %*% weights$W1 + matrix(weights$b1, nrow = nrow(X), ncol = ncol(weights$b1), byrow = TRUE)
    A1 <- leaky_relu(Z1)
    Z2 <- A1 %*% weights$W2 + matrix(weights$b2, nrow = nrow(A1), ncol = ncol(weights$b2), byrow = TRUE)
    A2 <- softmax(Z2)  # Softmax activation for multi-class
    
    return(list(Z1 = Z1, A1 = A1, Z2 = Z2, A2 = A2))
  }
  
  backward_propagation <- function(X, Y, weights, cache, learning_rate, alpha = 0.01, lambda = 0.01) {
    m <- nrow(X)  # Number of training examples
    dA2 <- cache$A2 - Y
    
    dZ2 <- dA2  # Derivative of cost w.r.t. Z2 (softmax layer)
    dW2 <- t(cache$A1) %*% dZ2 - lambda * weights$W2
    db2 <- matrix(colSums(dZ2) / m, nrow = 1, ncol = ncol(dZ2), byrow = TRUE)
    
    dA1 <- dZ2 %*% t(weights$W2)
    dZ1 <- ifelse(cache$A1 > 0, dA1, alpha * dA1)
    dW1 <- t(X) %*% dZ1 - lambda * weights$W1
    db1 <- matrix(colSums(dZ1) / m, nrow = 1, ncol = ncol(dZ1), byrow = TRUE)
    
    weights$W2 <- weights$W2 - learning_rate * dW2
    weights$b2 <- weights$b2 - learning_rate * db2
    weights$W1 <- weights$W1 - learning_rate * dW1
    weights$b1 <- weights$b1 - learning_rate * db1
    
    return(weights)
  }
  
  
  # Loss calculation - Mean Squared Error
  
  
  
  
  # Modified training function with loss calculation
  train_mlp <- function(X, Y, loss_function="mse",hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
    iterations <<- num_iterations
    input_size <- ncol(X)
    output_size <- ncol(Y)
    weights <- initialize_weights(input_size, hidden_size, output_size)
    calculate_loss <- switch(
      loss_function,
      # ce = calculate_ce_loss,
      mse = calculate_mse_loss,
      mae = calculate_mae_loss,
      huber = calculate_huber_loss,
      stop("Invalid loss function specified.")
    )
    previous_loss <- Inf
    for (i in 1:num_iterations) {
      cache <- forward_propagation(X, weights)
      predictions <- cache$A2
      
      # Calculate loss
      loss <<- calculate_loss(predictions, Y)
      cat("Iteration:", i, "Loss:", loss, "\n")
      
      # Check for convergence
      # if (abs(previous_loss - loss) < convergence_threshold) {
      if (is.finite(previous_loss) && is.finite(loss) && abs(previous_loss - loss) < convergence_threshold) {
        iterations <<- i
        cat("Convergence reached! \n")
        break
      }
      
      # Backpropagation
      weights <- backward_propagation(X, Y, weights, cache, learning_rate, lambda)
      if (is.nan(loss)) {
        break
      }
      
      previous_loss <- loss
    }
    
    cat("Final Loss : ", loss, "\n")
    cat("Iterations : ", iterations, "\n")
    
    model <<- weights
  }
  # train_mlp <- function(X, Y, loss_function = "ce",hidden_size = 4, learning_rate = 0.1, num_iterations = 1000,convergence_threshold = 0.0001,lambda = 0.01){
  #   model <<- NULL
  #   iterations <<- num_iterations
  #   input_size <- ncol(X)
  #   output_size <- ncol(Y)
  #   weights <- initialize_weights(input_size, hidden_size, output_size)
  #   
  #   # Choose the appropriate loss function based on the user's input
  #   calculate_loss <- switch(
  #     loss_function,
  #     # ce = calculate_ce_loss,
  #     mse = calculate_mse_loss,
  #     mae = calculate_mae_loss,
  #     huber = calculate_huber_loss,
  #     stop("Invalid loss function specified.")
  #   )
  #   previous_loss <- Inf
  #   for (i in 1:num_iterations) {
  #     cache <- forward_propagation(X, weights)
  #     predictions <- cache$A2
  #     
  #     # Calculate loss
  #     loss <<- calculate_loss(predictions, Y)
  #     cat("Iteration:", i, "Loss:", loss, "\n")
  #     # Check for convergence
  #     if (abs(previous_loss - loss) < convergence_threshold) {
  #       iterations <<- i
  #       cat("Convergence reached! \n")
  #       break
  #     }
  #     # if(is.nan(loss)){
  #     #   cat("Loss is Nan \n")
  #     #   loss <<- Inf
  #     #   break}
  #     # Backpropagation
  #     weights <- backward_propagation(X, Y, weights, cache, learning_rate,lambda)
  #     # if(loss<threshold) {
  #     #   iterations <<- i
  #     #   cat("Threshold reached ! \n")
  #     #   break}
  #   }
  #   cat("Loss : ",loss,"\n")
  #   cat("Iterations : ",iterations,"\n")
  #   model <<- weights
  #   # return(weights)
  # }
  
  train_mlpSilent <- function(X, Y,loss_function="mse", hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
    iterations <<- num_iterations
    input_size <- ncol(X)
    output_size <- ncol(Y)
    weights <- initialize_weights(input_size, hidden_size, output_size)
    calculate_loss <- switch(
      loss_function,
      # ce = calculate_ce_loss,
      mse = calculate_mse_loss,
      mae = calculate_mae_loss,
      huber = calculate_huber_loss,
      stop("Invalid loss function specified.")
    )
    previous_loss <- Inf
    previous_loss <- Inf
    
    for (i in 1:num_iterations) {
      cache <- forward_propagation(X, weights)
      predictions <- cache$A2
      
      # Calculate loss
      loss <<- calculate_loss(predictions, Y)
      # cat("Iteration:", i, "Loss:", loss, "\n")
      
      # Check for convergence
      # if (abs(previous_loss - loss) < convergence_threshold) {
      if (is.finite(previous_loss) && is.finite(loss) && abs(previous_loss - loss) < convergence_threshold) {
        iterations <<- i
        # cat("Convergence reached! \n")
        break
      }
      
      # Backpropagation
      weights <- backward_propagation(X, Y, weights, cache, learning_rate, lambda)
      if (is.nan(loss)) {
        break
      }
      
      previous_loss <- loss
    }
    
    # cat("Final Loss : ", loss, "\n")
    # cat("Iterations : ", iterations, "\n")
    
    model <<- weights
  }
  # Modified training function with loss calculation
  # train_mlpSilent <- function(X, Y, hidden_size = 4, learning_rate = 0.1, num_iterations = 1000,convergence_threshold = 0.0001, lambda = 0.01){
  #   model <<- NULL
  #   iterations <<- num_iterations
  #   input_size <- ncol(X)
  #   output_size <- ncol(Y)
  #   weights <- initialize_weights(input_size, hidden_size, output_size)
  #   calculate_loss <- switch(
  #     loss_function,
  #     # ce = calculate_ce_loss,
  #     mse = calculate_mse_loss,
  #     mae = calculate_mae_loss,
  #     huber = calculate_huber_loss,
  #     stop("Invalid loss function specified.")
  #   )
  #   previous_loss <- Inf
  #   for (i in 1:num_iterations) {
  #     cache <- forward_propagation(X, weights)
  #     weights2 <- backward_propagation(X, Y, weights, cache, learning_rate,lambda)
  #     predictions <- cache$A2
  #     # Calculate loss
  #     loss <<- calculate_loss(predictions, Y)
  #     # Check for convergence
  #     if (abs(previous_loss - loss) < convergence_threshold) {
  #       iterations <<- i
  #       cat("Convergence reached! \n")
  #       break
  #     }
  #     # if(is.nan(loss)){
  #     #   loss <<- Inf
  #     #   break}
  #     weights <- weights2
  #     # Backpropagation
  #     # if(loss<threshold) {
  #     #   iterations <<- i
  #     #   # cat("Threshold reached ! \n")
  #     #   break}
  #   }
  #   # cat("Loss : ",loss,"\n")
  #   # cat("Iterations : ",iterations,"\n")
  #   model <<- weights
  #   # return(weights)
  # }
  # Prédiction avec le réseau de neurones
  predict_mlp <- function(X) {
    cache <- forward_propagation(X, model)
    result <- cache$A2
    predictions <<- result
  }
  
  save_mlp <- function(modelName = name){saveRDS(model,modelName)}
  load_mlp <- function(model){model<<-readRDS(model)}
  
  return(list(
    predict = predict_mlp,
    train = train_mlp,
    trainSilent = train_mlpSilent,
    save = save_mlp,
    load = load_mlp,
    loss = get_loss,
    name= get_name,
    description = get_description,
    predictions = get_predictions,
    iterations = get_iterations))
}


fnnRegression <- function(name) {
  iterations <- -Inf
  get_iterations <- function(){return(iterations)}
  name <- name
  get_name <- function(){return(name)}
  get_description <- function(){cat(name,"\n",
                                    "iterations : ", iterations,"\n",
                                    "loss : ",loss,"\n")}
  
  loss <- Inf
  get_loss <- function(){return(loss)}
  
  model <- NULL
  predictions <- NULL
  get_predictions <- function(){return(predictions)}
  
  # Fonction d'activation sigmoïde
  sigmoid <- function(x) {
    1 / (1 + exp(-x))
  }
  
  # Fonction d'activation (Leaky ReLU)
  leaky_relu <- function(z, alpha=0.01) {
    pmax(alpha * z, z)
  }
  
  # Loss calculation - Cross Entropy
  calculate_ce_loss <- function(predictions, ground_truth) {
    loss <- -sum(ground_truth * log(predictions + 1e-15)) / nrow(ground_truth)
    return(loss)
  }
  # Loss calculation - Mean Squared Error
  calculate_mse_loss <- function(predictions, ground_truth) {
    loss <- mean((predictions - ground_truth)^2)
    return(loss)
  }
  
  # Loss calculation - Mean Absolute Error
  calculate_mae_loss <- function(predictions, ground_truth) {
    loss <- mean(abs(predictions - ground_truth))
    return(loss)
  }
  
  # Loss calculation - Huber Loss
  calculate_huber_loss <- function(predictions, ground_truth, delta = 1.0) {
    loss <- mean(ifelse(abs(predictions - ground_truth) <= delta, 0.5 * (predictions - ground_truth)^2, delta * (abs(predictions - ground_truth) - 0.5 * delta)))
    return(loss)
  }
  
  
  # Initialisation des poids et biais du réseau
  initialize_weights <- function(input_size, hidden_size, output_size) {
    # W1 <- matrix(runif(input_size * hidden_size), nrow = input_size,ncol = hidden_size)
    # b1 <- matrix(runif(hidden_size), nrow = 1, ncol = hidden_size)
    # W2 <- matrix(runif(hidden_size * output_size),nrow = hidden_size,ncol = output_size)
    # b2 <- matrix(runif(output_size), nrow = 1, ncol = output_size)
    W1 <- matrix(rnorm(input_size * hidden_size, mean = 0, sd = sqrt(2 / input_size)), nrow = input_size, ncol = hidden_size)
    b1 <- matrix(0, nrow = 1, ncol = hidden_size)
    W2 <- matrix(rnorm(hidden_size * output_size, mean = 0, sd = sqrt(2 / hidden_size)), nrow = hidden_size, ncol = output_size)
    b2 <- matrix(0, nrow = 1, ncol = output_size)
    return(list(W1 = W1,
                b1 = b1,
                W2 = W2,
                b2 = b2))
  }
  
  # Propagation avant
  forward_propagation <- function(X, weights) {
    Z1 <- X %*% weights$W1 + matrix(weights$b1, nrow = nrow(X), ncol = ncol(weights$b1), byrow = TRUE)
    A1 <- leaky_relu(Z1)
    Z2 <-A1 %*% weights$W2 + matrix(weights$b2, nrow = nrow(A1), ncol = ncol(weights$b2),byrow = TRUE)
    A2 <- leaky_relu(Z2)
    
    return(list(Z1 = Z1,
                A1 = A1,
                Z2 = Z2,
                A2 = A2))
  }
  
  backward_propagation <- function(X, Y, weights, cache, learning_rate, alpha = 0.01, lambda = 0.01) {
    m <- nrow(X)  # Number of training examples
    dA2 <- cache$A2 - Y  # Compute derivative of the cost with respect to the output activation of the last layer (using Leaky ReLU derivative)
    
    # Calculate the derivative of the cost with respect to the weighted sum of the last layer using Leaky ReLU derivative
    dZ2 <- ifelse(cache$A2 > 0, dA2, alpha * dA2)
    dW2 <- t(cache$A1) %*% dZ2 - lambda * weights$W2  # Include L2 penalty in weight update
    db2 <- matrix(colSums(dZ2) / m, nrow = 1, ncol = ncol(dZ2), byrow = TRUE)
    
    dA1 <- dZ2 %*% t(weights$W2)
    # Calculate the derivative of the cost with respect to the weighted sum of the first hidden layer using Leaky ReLU derivative
    dZ1 <- ifelse(cache$A1 > 0, dA1, alpha * dA1)
    dW1 <- t(X) %*% dZ1 - lambda * weights$W1  # Include L2 penalty in weight update
    
    db1 <- matrix(colSums(dZ1) / m, nrow = 1, ncol = ncol(dZ1), byrow = TRUE)
    
    # Update the weights and biases using gradient descent with L2 regularization
    weights$W2 <- weights$W2 - learning_rate * dW2
    weights$b2 <- weights$b2 - learning_rate * db2
    weights$W1 <- weights$W1 - learning_rate * dW1
    weights$b1 <- weights$b1 - learning_rate * db1
    
    return(weights)  # Return the updated weights and biases
  }
  
  # Batch Forward Propagation
  batch_forward_propagation <- function(X, weights) {
    Z1 <- X %*% weights$W1 + matrix(weights$b1, nrow = nrow(X), ncol = ncol(weights$b1), byrow = TRUE)
    A1 <- leaky_relu(Z1)
    Z2 <- A1 %*% weights$W2 + matrix(weights$b2, nrow = nrow(A1), ncol = ncol(weights$b2), byrow = TRUE)
    A2 <- leaky_relu(Z2)
    
    return(list(Z1 = Z1, A1 = A1, Z2 = Z2, A2 = A2))
  }
  
  # Batch Backward Propagation
  batch_backward_propagation <- function(X, Y, weights, cache, learning_rate, alpha = 0.01, lambda = 0.01) {
    m <- nrow(X)  # Number of training examples
    dA2 <- cache$A2 - Y  # Compute derivative of the cost with respect to the output activation of the last layer (using Leaky ReLU derivative)
    
    # Calculate the derivative of the cost with respect to the weighted sum of the last layer using Leaky ReLU derivative
    dZ2 <- ifelse(cache$A2 > 0, dA2, alpha * dA2)
    dW2 <- t(cache$A1) %*% dZ2 - lambda * weights$W2  # Include L2 penalty in weight update
    db2 <- matrix(colSums(dZ2) / m, nrow = 1, ncol = ncol(dZ2), byrow = TRUE)
    
    dA1 <- dZ2 %*% t(weights$W2)
    # Calculate the derivative of the cost with respect to the weighted sum of the first hidden layer using Leaky ReLU derivative
    dZ1 <- ifelse(cache$A1 > 0, dA1, alpha * dA1)
    dW1 <- t(X) %*% dZ1 - lambda * weights$W1  # Include L2 penalty in weight update
    
    db1 <- matrix(colSums(dZ1) / m, nrow = 1, ncol = ncol(dZ1), byrow = TRUE)
    
    # Update the weights and biases using gradient descent with L2 regularization
    weights$W2 <- weights$W2 - learning_rate * dW2
    weights$b2 <- weights$b2 - learning_rate * db2
    weights$W1 <- weights$W1 - learning_rate * dW1
    weights$b1 <- weights$b1 - learning_rate * db1
    
    return(weights)  # Return the updated weights and biases
  }
  
  # Modified training function with batch training
  train_mlp_batch <- function(X, Y,loss_function = "ce", batch_size = 32, hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
    model <<- NULL
    iterations <<- num_iterations
    input_size <- ncol(X)
    output_size <- ncol(Y)
    weights <- initialize_weights(input_size, hidden_size, output_size)
    
    # Choose the appropriate loss function based on the user's input
    calculate_loss <- switch(
      loss_function,
      # ce = calculate_ce_loss,
      mse = calculate_mse_loss,
      mae = calculate_mae_loss,
      huber = calculate_huber_loss,
      stop("Invalid loss function specified.")
    )
    previous_loss <- Inf
    for (i in 1:num_iterations) {
      for (batch_start in seq(1, nrow(X), batch_size)) {
        batch_end <- min(batch_start + batch_size - 1, nrow(X))
        X_batch <- X[batch_start:batch_end, ]
        Y_batch <- Y[batch_start:batch_end, ]
        
        cache <- batch_forward_propagation(X_batch, weights)
        predictions <- cache$A2
        
        # Calculate loss
        loss <<- calculate_loss(predictions, Y_batch)
        cat("Iteration:", i, "Batch:", batch_start, "-", batch_end, "Loss:", loss, "\n")
        # Check for convergence
        if (abs(previous_loss - loss) < convergence_threshold) {
          iterations <<- i
          cat("Convergence reached! \n")
          break
        }
        # 
        # if (is.nan(loss)) {
        #   cat("Loss is NaN \n")
        #   loss <<- Inf
        #   break
        # }
        
        # Backpropagation
        weights <- batch_backward_propagation(X_batch, Y_batch, weights, cache, learning_rate, lambda)
      }
      
      # if (loss < threshold) {
      #   iterations <<- i
      #   cat("Threshold reached ! \n")
      #   break
      # }
    }
    
    cat("Loss : ", loss, "\n")
    cat("Iterations : ", iterations, "\n")
    model <<- weights
  }
  
  
  # Modified training function with loss calculation
  train_mlp <- function(X, Y, loss_function="mse", hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
    iterations <<- num_iterations
    input_size <- ncol(X)
    output_size <- ncol(Y)
    weights <- initialize_weights(input_size, hidden_size, output_size)
    calculate_loss <- switch(
      loss_function,
      ce = calculate_ce_loss,
      mse = calculate_mse_loss,
      mae = calculate_mae_loss,
      huber = calculate_huber_loss,
      stop("Invalid loss function specified.")
    )
    previous_loss <- Inf
    
    for (i in 1:num_iterations) {
      cache <- forward_propagation(X, weights)
      predictions <- cache$A2
      
      # Calculate loss
      loss <<- calculate_loss(predictions, Y)
      cat("Iteration:", i, "Loss:", loss, "\n")
      
      # Check for convergence
      # if (abs(previous_loss - loss) < convergence_threshold) {
      if (is.finite(previous_loss) && is.finite(loss) && abs(previous_loss - loss) < convergence_threshold) {
        iterations <<- i
        cat("Convergence reached! \n")
        break
      }
      
      # Backpropagation
      weights <- backward_propagation(X, Y, weights, cache, learning_rate, lambda)
      if (loss < convergence_threshold) {
        iterations <<- i
        cat("Threshold reached ! \n")
        break
      }
      previous_loss <- loss
    }
    
    cat("Final Loss : ", loss, "\n")
    cat("Iterations : ", iterations, "\n")
    
    model <<- weights
  }
  
  # train_mlp <- function(X, Y,loss_function = "mse", hidden_size = 4, learning_rate = 0.1, num_iterations = 1000,convergence_threshold = 0.0001 ,lambda = 0.01){
  #   model <<- NULL
  #   iterations <<- num_iterations
  #   input_size <- ncol(X)
  #   output_size <- ncol(Y)
  #   weights <- initialize_weights(input_size, hidden_size, output_size)
  #   
  #   calculate_loss <- switch(
  #     loss_function,
  #     # ce = calculate_ce_loss,
  #     mse = calculate_mse_loss,
  #     mae = calculate_mae_loss,
  #     huber = calculate_huber_loss,
  #     stop("Invalid loss function specified.")
  #   )
  #   previous_loss <- Inf
  #   for (i in 1:num_iterations) {
  #     cache <- forward_propagation(X, weights)
  #     predictions <- cache$A2
  #     
  #     # Calculate loss
  #     loss <<- calculate_loss(predictions, Y)
  #     cat("Iteration:", i, "Loss:", loss, "\n")
  #     # Check for convergence
  #     if (abs(previous_loss - loss) < convergence_threshold) {
  #       iterations <<- i
  #       cat("Convergence reached! \n")
  #       break
  #     }
  #     # if(is.nan(loss)){
  #     #   cat("Loss is Nan \n")
  #     #   loss <<- Inf
  #     #   break}
  #     # Backpropagation
  #     weights <- backward_propagation(X, Y, weights, cache, learning_rate,lambda)
  #     # if(loss<threshold) {
  #     #   iterations <<- i
  #     #   cat("Threshold reached ! \n")
  #     #   break}
  #   }
  #   cat("Loss : ",loss,"\n")
  #   cat("Iterations : ",iterations,"\n")
  #   model <<- weights
  #   # return(weights)
  # }
  
  train_mlpSilent <- function(X, Y, loss_function="mse", hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
    iterations <<- num_iterations
    input_size <- ncol(X)
    output_size <- ncol(Y)
    weights <- initialize_weights(input_size, hidden_size, output_size)
    calculate_loss <- switch(
      loss_function,
      # ce = calculate_ce_loss,
      mse = calculate_mse_loss,
      mae = calculate_mae_loss,
      huber = calculate_huber_loss,
      stop("Invalid loss function specified.")
    )
    previous_loss <- Inf
    
    for (i in 1:num_iterations) {
      cache <- forward_propagation(X, weights)
      predictions <- cache$A2
      
      # Calculate loss
      loss <<- calculate_loss(predictions, Y)
      # cat("Iteration:", i, "Loss:", loss, "\n")
      
      # Check for convergence
      # if (abs(previous_loss - loss) < convergence_threshold) {
      if (is.finite(previous_loss) && is.finite(loss) && abs(previous_loss - loss) < convergence_threshold) {
        iterations <<- i
        # cat("Convergence reached! \n")
        break
      }
      
      # Backpropagation
      weights <- backward_propagation(X, Y, weights, cache, learning_rate, lambda)
      previous_loss <- loss
    }
    
    # cat("Final Loss : ", loss, "\n")
    # cat("Iterations : ", iterations, "\n")
    # 
    model <<- weights
  }
  # train_mlpSilent <- function(X, Y, loss_function = "mse",hidden_size = 4, learning_rate = 0.1, num_iterations = 1000,convergence_threshold = 0.0001,lambda = 0.01){
  #   model <<- NULL
  #   iterations <<- num_iterations
  #   input_size <- ncol(X)
  #   output_size <- ncol(Y)
  #   weights <- initialize_weights(input_size, hidden_size, output_size)
  #   calculate_loss <- switch(
  #     loss_function,
  #     # ce = calculate_ce_loss,
  #     mse = calculate_mse_loss,
  #     mae = calculate_mae_loss,
  #     huber = calculate_huber_loss,
  #     stop("Invalid loss function specified.")
  #   )
  #   previous_loss <- Inf
  #   for (i in 1:num_iterations) {
  #     cache <- forward_propagation(X, weights)
  #     predictions <- cache$A2
  #     
  #     # Calculate loss
  #     loss <<- calculate_loss(predictions, Y)
  #     # cat("Iteration:", i, "Loss:", loss, "\n")
  #     # Check for convergence
  #     if (abs(previous_loss - loss) < convergence_threshold) {
  #       iterations <<- i
  #       cat("Convergence reached! \n")
  #       break
  #     }
  #     # if(is.nan(loss)){
  #     #   # cat("Loss is Nan \n")
  #     #   loss <<- Inf
  #     #   break}
  #     # Backpropagation
  #     weights <- backward_propagation(X, Y, weights, cache, learning_rate,lambda)
  #     # if(loss<threshold) {
  #     #   iterations <<- i
  #     #   # cat("Threshold reached ! \n")
  #     #   break}
  #   }
  #   # cat("Loss : ",loss,"\n")
  #   # cat("Iterations : ",iterations,"\n")
  #   model <<- weights
  #   # return(weights)
  # }
  
  # Prédiction avec le réseau de neurones
  predict_mlp <- function(X) {
    cache <- forward_propagation(X, model)
    result <- cache$A2
    predictions <<- result
    # return(result)
  }
  
  save_mlp <- function(modelName = name){saveRDS(model,modelName)}
  load_mlp <- function(model){model<<-readRDS(model)}
  
  return(list(
    predict = predict_mlp,
    train = train_mlp,
    trainBatch = train_mlp_batch,
    trainSilent = train_mlpSilent,
    save = save_mlp,
    load = load_mlp,
    loss = get_loss,
    name= get_name,
    description = get_description,
    predictions = get_predictions,
    iterations = get_iterations))
}

fnnAutoencoder <- function(name) {
  iterations <- -Inf
  get_iterations <- function() { return(iterations) }
  name <- name
  get_name <- function() { return(name) }
  get_description <- function() {
    cat(name, "\n", "iterations : ", iterations, "\n", "loss : ", loss, "\n")
  }
  
  loss <- Inf
  get_loss <- function() { return(loss) }
  
  encoder <- NULL
  decoder <- NULL
  get_encoder <- function() { return(encoder) }
  get_decoder <- function() { return(decoder) }
  
  predictions <- NULL
  get_predictions <- function(){return(predictions)}
  
  # Fonction d'activation sigmoïde
  sigmoid <- function(x) {
    1 / (1 + exp(-x))
  }
  
  # Fonction d'activation (Leaky ReLU)
  leaky_relu <- function(z, alpha=0.01) {
    pmax(alpha * z, z)
  }
  
  
  # Initialisation des poids et biais du réseau
  initialize_weights <- function(input_size, hidden_size) {
    W_encode <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
    b_encode <- matrix(runif(hidden_size), nrow = 1, ncol = hidden_size)
    W_decode <- matrix(runif(hidden_size * input_size), nrow = hidden_size, ncol = input_size)
    b_decode <- matrix(runif(input_size), nrow = 1, ncol = input_size)
    
    return(list(W_encode = W_encode,
                b_encode = b_encode,
                W_decode = W_decode,
                b_decode = b_decode))
  }
  
  # Encoder function
  encode <- function(X, weights) {
    Z_encode <- X %*% weights$W_encode + matrix(weights$b_encode, nrow = nrow(X), ncol = ncol(weights$b_encode), byrow = TRUE)
    A_encode <- leaky_relu(Z_encode)
    
    return(list(Z_encode = Z_encode,
                A_encode = A_encode))
  }
  
  # Decoder function
  decode <- function(Z_encode, weights) {
    Z_decode <- Z_encode %*% weights$W_decode + matrix(weights$b_decode, nrow = nrow(Z_encode), ncol = ncol(weights$b_decode), byrow = TRUE)
    A_decode <- leaky_relu(Z_decode)
    
    return(list(Z_decode = Z_decode,
                A_decode = A_decode))
  }
  
  # Autoencoder forward propagation
  autoencoder_forward <- function(X, weights) {
    encoder_output <- encode(X, weights)
    decoder_output <- decode(encoder_output$A_encode, weights)
    
    return(list(encoder_output = encoder_output,
                decoder_output = decoder_output))
  }
  
  # Backward propagation for autoencoder
  autoencoder_backward <- function(X, weights, cache, learning_rate, alpha = 0.01, lambda = 0.01) {
    m <- nrow(X)  # Number of training examples
    
    # Decoder backpropagation
    dZ_decode <- cache$decoder_output$A_decode - X
    dW_decode <- t(cache$encoder_output$A_encode) %*% dZ_decode - lambda * weights$W_decode
    db_decode <- matrix(colSums(dZ_decode) / m, nrow = 1, ncol = ncol(dZ_decode), byrow = TRUE)
    
    # Encoder backpropagation
    dA_encode <- dZ_decode %*% t(weights$W_decode)
    dZ_encode <- ifelse(cache$encoder_output$A_encode > 0, dA_encode, alpha * dA_encode)
    dW_encode <- t(X) %*% dZ_encode - lambda * weights$W_encode
    db_encode <- matrix(colSums(dZ_encode) / m, nrow = 1, ncol = ncol(dZ_encode), byrow = TRUE)
    
    # Update weights
    weights$W_decode <- weights$W_decode - learning_rate * dW_decode
    weights$b_decode <- weights$b_decode - learning_rate * db_decode
    weights$W_encode <- weights$W_encode - learning_rate * dW_encode
    weights$b_encode <- weights$b_encode - learning_rate * db_encode
    
    return(weights)
  }
  
  # Loss calculation - Mean Squared Error
  calculate_loss <- function(predictions, ground_truth) {
    loss <- mean((predictions - ground_truth)^2)
    return(loss)
  }
  
  train_autoencoder <- function(X, hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
    iterations <<- num_iterations
    input_size <- ncol(X)
    weights <- initialize_weights(input_size, hidden_size)
    
    previous_loss <- Inf
    
    for (i in 1:num_iterations) {
      cache <- autoencoder_forward(X, weights)
      
      # Calculate loss
      loss <<- calculate_loss(cache$decoder_output$A_decode, X)
      cat("Iteration:", i, "Loss:", loss, "\n")
      
      # Check for convergence
      if (abs(previous_loss - loss) < convergence_threshold) {
        iterations <<- i
        cat("Convergence reached! \n")
        break
      }
      
      # Backpropagation
      weights <- autoencoder_backward(X, weights, cache, learning_rate, lambda)
      
      previous_loss <- loss
    }
    
    cat("Final Loss : ", loss, "\n")
    cat("Iterations : ", iterations, "\n")
    
    encoder <<- weights[c("W_encode", "b_encode")]
    decoder <<- weights[c("W_decode", "b_decode")]
  }
    predict_autoencoder <- function(X) {
      cache <- autoencoder_forward(X, list(W_encode = encoder$W_encode, b_encode = encoder$b_encode, W_decode = decoder$W_decode, b_decode = decoder$b_decode))
      result <- cache$decoder_output$A_decode
      predictions <<- result
      return(result)
    }
  
  # Modified training function with loss calculation
  # train_autoencoder <- function(X, hidden_size = 4,learning_rate = 0.1, num_iterations = 1000, threshold = 0.02, lambda = 0.01) {
  #   iterations <<- num_iterations
  #   input_size <- ncol(X)
  #   # hidden_size <- input_size / 2  # Choose an arbitrary size for the hidden layer
  #   weights <- initialize_weights(input_size, hidden_size)
  #   
  #   for (i in 1:num_iterations) {
  #     cache <- autoencoder_forward(X, weights)
  #     
  #     # Calculate loss
  #     loss <<- calculate_loss(cache$decoder_output$A_decode, X)
  #     cat("Iteration:", i, "Loss:", loss, "\n")
  #     
  #     # Backpropagation
  #     weights <- autoencoder_backward(X, weights, cache, learning_rate, lambda)
  #     if (loss < threshold) {
  #       iterations <<- i
  #       cat("Threshold reached ! \n")
  #       break
  #     }
  #   }
  #   cat("Loss : ", loss, "\n")
  #   cat("Iterations : ", iterations, "\n")
  #   encoder <<- weights[c("W_encode", "b_encode")]
  #   decoder <<- weights[c("W_decode", "b_decode")]
  # }
  # 
  # # Prédiction avec l'autoencoder
  # predict_autoencoder <- function(X) {
  #   cache <- autoencoder_forward(X, list(W_encode = encoder$W_encode, b_encode = encoder$b_encode, W_decode = decoder$W_decode, b_decode = decoder$b_decode))
  #   result <- cache$decoder_output$A_decode
  #   predictions <<- result
  #   return(result)
  # }
  
  save_autoencoder <- function(modelName = name) {
    saveRDS(list(encoder = encoder, decoder = decoder), modelName)
  }
  
  load_autoencoder <- function(model) {
    model <- readRDS(model)
    encoder <<- model$encoder
    decoder <<- model$decoder
  }
  
  return(list(
    predict = predict_autoencoder,
    predictions = get_predictions,
    train = train_autoencoder,
    save = save_autoencoder,
    load = load_autoencoder,
    loss = get_loss,
    name = get_name,
    description = get_description,
    iterations = get_iterations
  ))
}
# 
# fnnAutoencoder <- function(name) {
#   iterations <- -Inf
#   get_iterations <- function() { return(iterations) }
#   name <- name
#   get_name <- function() { return(name) }
#   get_description <- function() {
#     cat(name, "\n", "iterations : ", iterations, "\n", "loss : ", loss, "\n")
#   }
#   
#   loss <- Inf
#   get_loss <- function() { return(loss) }
#   
#   encoder <- NULL
#   decoder <- NULL
#   get_encoder <- function() { return(encoder) }
#   get_decoder <- function() { return(decoder) }
#   
#   predictions <- NULL
#   get_predictions <- function(){return(predictions)}
#   
#   # Fonction d'activation sigmoïde
#   sigmoid <- function(x) {
#     1 / (1 + exp(-x))
#   }
#   
#   # Fonction d'activation (Leaky ReLU)
#   leaky_relu <- function(z, alpha=0.01) {
#     pmax(alpha * z, z)
#   }
#   
#   
#   # Initialisation des poids et biais du réseau
#   initialize_weights <- function(input_size, hidden_size) {
#     # W_encode <- matrix(runif(input_size * hidden_size), nrow = input_size, ncol = hidden_size)
#     # b_encode <- matrix(runif(hidden_size), nrow = 1, ncol = hidden_size)
#     # W_decode <- matrix(runif(hidden_size * input_size), nrow = hidden_size, ncol = input_size)
#     # b_decode <- matrix(runif(input_size), nrow = 1, ncol = input_size)
#     # 
#     W_encode <- matrix(rnorm(input_size * hidden_size, mean = 0, sd = sqrt(2 / input_size)), nrow = input_size, ncol = hidden_size)
#     b_encode <- matrix(0, nrow = 1, ncol = hidden_size)
#     W_decode <- matrix(rnorm(hidden_size * output_size, mean = 0, sd = sqrt(2 / hidden_size)), nrow = hidden_size, ncol = output_size)
#     b_decode <- matrix(0, nrow = 1, ncol = output_size)
#     return(list(W_encode = W_encode,
#                 b_encode = b_encode,
#                 W_decode = W_decode,
#                 b_decode = b_decode))
#   }
#   
#   # Encoder function
#   encode <- function(X, weights) {
#     Z_encode <- X %*% weights$W_encode + matrix(weights$b_encode, nrow = nrow(X), ncol = ncol(weights$b_encode), byrow = TRUE)
#     A_encode <- leaky_relu(Z_encode)
#     
#     return(list(Z_encode = Z_encode,
#                 A_encode = A_encode))
#   }
#   
#   # Decoder function
#   decode <- function(Z_encode, weights) {
#     Z_decode <- Z_encode %*% weights$W_decode + matrix(weights$b_decode, nrow = nrow(Z_encode), ncol = ncol(weights$b_decode), byrow = TRUE)
#     A_decode <- leaky_relu(Z_decode)
#     
#     return(list(Z_decode = Z_decode,
#                 A_decode = A_decode))
#   }
#   
#   # Autoencoder forward propagation
#   autoencoder_forward <- function(X, weights) {
#     encoder_output <- encode(X, weights)
#     decoder_output <- decode(encoder_output$A_encode, weights)
#     
#     return(list(encoder_output = encoder_output,
#                 decoder_output = decoder_output))
#   }
#   
#   # Backward propagation for autoencoder
#   autoencoder_backward <- function(X, weights, cache, learning_rate, alpha = 0.01, lambda = 0.01) {
#     m <- nrow(X)  # Number of training examples
#     
#     # Decoder backpropagation
#     dZ_decode <- cache$decoder_output$A_decode - X
#     dW_decode <- t(cache$encoder_output$A_encode) %*% dZ_decode - lambda * weights$W_decode
#     db_decode <- matrix(colSums(dZ_decode) / m, nrow = 1, ncol = ncol(dZ_decode), byrow = TRUE)
#     
#     # Encoder backpropagation
#     dA_encode <- dZ_decode %*% t(weights$W_decode)
#     dZ_encode <- ifelse(cache$encoder_output$A_encode > 0, dA_encode, alpha * dA_encode)
#     dW_encode <- t(X) %*% dZ_encode - lambda * weights$W_encode
#     db_encode <- matrix(colSums(dZ_encode) / m, nrow = 1, ncol = ncol(dZ_encode), byrow = TRUE)
#     
#     # Update weights
#     weights$W_decode <- weights$W_decode - learning_rate * dW_decode
#     weights$b_decode <- weights$b_decode - learning_rate * db_decode
#     weights$W_encode <- weights$W_encode - learning_rate * dW_encode
#     weights$b_encode <- weights$b_encode - learning_rate * db_encode
#     
#     return(weights)
#   }
#   
#   # Loss calculation - Mean Squared Error
#   calculate_loss <- function(predictions, ground_truth) {
#     loss <- mean((predictions - ground_truth)^2)
#     return(loss)
#   }
#   
#   
#   # Modified training function with loss calculation
#   train_autoencoder <- function(X, hidden_size = 4, learning_rate = 0.1, num_iterations = 1000, convergence_threshold = 0.0001, lambda = 0.01) {
#     iterations <<- num_iterations
#     input_size <- ncol(X)
#     weights <- initialize_weights(input_size, hidden_size)
#     
#     previous_loss <- Inf
#     
#     for (i in 1:num_iterations) {
#       cache <- autoencoder_forward(X, weights)
#       
#       # Calculate loss
#       loss <<- calculate_loss(cache$decoder_output$A_decode, X)
#       cat("Iteration:", i, "Loss:", loss, "\n")
#       
#       # Check for convergence
#       if (abs(previous_loss - loss) < convergence_threshold) {
#         iterations <<- i
#         cat("Convergence reached! \n")
#         break
#       }
#       
#       # Backpropagation
#       weights <- autoencoder_backward(X, weights, cache, learning_rate, lambda)
#       
#       previous_loss <- loss
#     }
#     
#     cat("Final Loss : ", loss, "\n")
#     cat("Iterations : ", iterations, "\n")
#     
#     encoder <<- weights[c("W_encode", "b_encode")]
#     decoder <<- weights[c("W_decode", "b_decode")]
#   }
#   # train_autoencoder <- function(X, hidden_size = 4,learning_rate = 0.1, num_iterations = 1000, threshold = 0.02, lambda = 0.01) {
#   #   iterations <<- num_iterations
#   #   input_size <- ncol(X)
#   #   # hidden_size <- input_size / 2  # Choose an arbitrary size for the hidden layer
#   #   weights <- initialize_weights(input_size, hidden_size)
#   #   
#   #   for (i in 1:num_iterations) {
#   #     cache <- autoencoder_forward(X, weights)
#   #     
#   #     # Calculate loss
#   #     loss <<- calculate_loss(cache$decoder_output$A_decode, X)
#   #     cat("Iteration:", i, "Loss:", loss, "\n")
#   #     
#   #     # Backpropagation
#   #     weights <- autoencoder_backward(X, weights, cache, learning_rate, lambda)
#   #     if (loss < threshold) {
#   #       iterations <<- i
#   #       cat("Threshold reached ! \n")
#   #       break
#   #     }
#   #   }
#   #   cat("Loss : ", loss, "\n")
#   #   cat("Iterations : ", iterations, "\n")
#   #   encoder <<- weights[c("W_encode", "b_encode")]
#   #   decoder <<- weights[c("W_decode", "b_decode")]
#   # }
#   
#   
#   # Prédiction avec l'autoencoder
#   predict_autoencoder <- function(X) {
#     cache <- autoencoder_forward(X, list(W_encode = encoder$W_encode, b_encode = encoder$b_encode, W_decode = decoder$W_decode, b_decode = decoder$b_decode))
#     result <- cache$decoder_output$A_decode
#     predictions <<- result
#     return(result)
#   }
#   
#   save_autoencoder <- function(modelName = name) {
#     saveRDS(list(encoder = encoder, decoder = decoder), modelName)
#   }
#   
#   load_autoencoder <- function(model) {
#     model <- readRDS(model)
#     encoder <<- model$encoder
#     decoder <<- model$decoder
#   }
#   
#   return(list(
#     predict = predict_autoencoder,
#     predictions = get_predictions,
#     train = train_autoencoder,
#     save = save_autoencoder,
#     load = load_autoencoder,
#     loss = get_loss,
#     name = get_name,
#     description = get_description,
#     iterations = get_iterations
#   ))
# }
