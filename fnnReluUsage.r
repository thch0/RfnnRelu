source("R/fnnRelu.R")

x1 <- matrix(rnorm(10))
y1 <- matrix(sin(x1))
X <- normalize(x1)
Y <- normalize(y1)

fnn1 <- fnnRegression("fnn1")
fnn1$train(X,Y,
           hidden_size = 16,
           learning_rate = .005,
           num_iterations =6000,
           convergence_threshold = 0.000001,
           lambda = 0.01)

x1 <- matrix(sin(seq(20,70,by=.1)))
y1 <- matrix(sin(x1))
Xtest <- normalize(x1)
Ytest <- normalize(y1)

fnn1$predict(Xtest)
res <- matrix(c(Ytest,fnn1$predictions()),ncol=2)
plot(res[,1],cex=.5,col="blue")
lines(res[,1],cex=.1,col="blue")
points(res[,2],cex=.5,col="red")
lines(res[,2],cex=.1,col="red")

##init classifier
classifier <- fnnClassification("Iris_classifier")
##classifier data
data <- as.matrix(cbind(normalize(iris[,1:4]),columnify(iris[,5])))
id <- sample(1:nrow(data), 20/100*nrow(data))
trainX <- data[id,1:4]
trainY <- data[id,5:7]
testX <- data[-id,1:4]
testY <- data[-id,5:7]
#classifier training
classifier$train(trainX,trainY,
                 hidden_size = 6,
                 learning_rate = .01,
                 num_iterations = 1000,
                 convergence_threshold = 0.000001,
                 lambda= 0.01)
#classifier predictions
classifier$predict(testX)
#formating outputs...
round(matrix(c(classifier$predictions(),classifier$predictions()),ncol=6),digits=5)

classifier$predictions()
confidence_matrix <- table(data = testY, pred = round(classifier$predictions()))
plot(confidence_matrix,
     main=paste(classifier$name(),":",
                "\n",
                "False positives = ",
                round(confidence_matrix[1,2]*100/sum(confidence_matrix[1,]),digits = 2),"%",
                "\n",
                "False negatives = ",
                round(confidence_matrix[2,1]*100/sum(confidence_matrix[2,]),digits = 2),"%"))

fnn3 <- fnnAutoencoder("a1")
fnn3$train(trainX)
