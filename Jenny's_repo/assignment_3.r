# a. The number of models are:
# at least 1: 6
# at least 2: 15
# at least 3: 20
# at least 4: 15
# total: 56

# b. Measure of classification error: Misclassification rate (proportion of wrong predictions)

#c.
# load dataset
df = read.csv("madrid_houses_clean.csv")

# convert house price into categories: 0 means < 500.000 and 1 means ≥ 500.000
df$price_class = ifelse(df$buy_price >= 500000, 1, 0)

# convert categorical predictors into numerical variables
df[, c("has_lift", "is_renewal_needed", "has_parking")] = ifelse(df[, c("has_lift", "is_renewal_needed", "has_parking")] == "True", 1, 0)

# keep 6 predictors
df = df[ ,c("sq_mt_built", "n_rooms", "n_bathrooms", "has_lift",
             "is_renewal_needed", "has_parking", "price_class")]

# try with 5000 rows of df
df = df[2000:7000, ]
# perform LOOCV: loop 1-by-1 test row to get the result
n = nrow(df)
actual = c()
model.prob = c()

for (i in 1:n){
  train = df[-i, ]
  test  = df[i, ] # make sure test is a dataframe and not vector
  actual[i] = df$price_class[i]
  model = glm(price_class ~ ., data = train, family = binomial)
  model.prob[i] = predict(model, test, type = "response")
}
# calculate the prediction error
model.pred <- rep("0", length(model.prob)) # create a vector of 0
model.pred[model.prob > .5] <- "1" # if the prediction is larger than .5, change to 1
confusion_matrix = table(model.pred, actual) # confusion matrix
LOOCV.error = (confusion_matrix[1, 2] + confusion_matrix[2, 1])/length(actual) # LOOCV estimate of prediction error

# create a function
LOOCV = function(df){
  n = nrow(df)
  actual = c()
  model.prob = c()
  
  for (i in 1:n){
    train = df[-i, ]
    test  = df[i, ] # make sure test is a dataframe and not vector
    actual[i] = df$price_class[i]
    model = glm(price_class ~ ., data = train, family = binomial)
    model.prob[i] = predict(model, test, type = "response")
  }
  # calculate the prediction error
  model.pred <- rep("0", length(model.prob)) # create a vector of 0
  model.pred[model.prob > .5] <- "1" # if the prediction is larger than .5, change to 1
  confusion_matrix = table(model.pred, actual) # confusion matrix
  LOOCV.error = (confusion_matrix[1, 2] + confusion_matrix[2, 1])/length(actual) # LOOCV estimate of prediction error
}

# apply function to each model




