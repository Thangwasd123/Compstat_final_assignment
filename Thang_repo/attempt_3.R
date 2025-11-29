df <- read.csv("data/madrid_houses_clean.csv")
library(dplyr)

head(df)
nrow(df)
ncol(df)
colnames(df)
typeof(df)
df$price_class <- ifelse(df$buy_price >= 500000, 1, 0)
colnames(df)

trimmed_df <- df[,c(3,4,5,10,11,14,18)]
colnames(trimmed_df)

set.seed(20251126)

sample <- sample(c(TRUE, FALSE), nrow(trimmed_df), replace = TRUE, prob=c(0.7,0.3))
train <- trimmed_df[sample,]
test <- trimmed_df[!sample,]

print(nrow(test)/(nrow(train) + nrow(test)))

#install.packages("caret")
#install.packages("pROC")


# Get the proportion of each class in the original data
table(train$price_class)
prop.table(table(train$price_class))

library(sampling)

# Stratified sampling with exact sample size
sample_size <- 3000

# Create strata
strata_info <- strata(
  data = train,
  stratanames = "price_class",
  size = round(table(train$price_class) * sample_size / nrow(train)),
  method = "srswor"  # Simple random sampling without replacement
)

train_sampled <- train[strata_info$ID_unit, ]



library(caret)
library(pROC)

# ========================================
# STEP 1: Generate All Possible Models
# =======================================
outcome_name <- c("price_class")
predictor_names <- setdiff(c(colnames(train_sampled)),outcome_name)

all_models <- list()
model_count <- 0

for (k in 1:4){
  combinations <- combn(predictor_names, k, simplify= FALSE)
  all_models <- c(all_models, combinations)
  model_count <- model_count + choose(6, k)
}

cat("Total number of models to evaluate:", model_count, "\n")
cat("Breakdown: C(6,1)=", choose(6,1), 
    ", C(6,2)=", choose(6,2),
    ", C(6,3)=", choose(6,3), 
    ", C(6,4)=", choose(6,4), "\n\n")


# ========================================
# STEP 2: Implement a LOOCV procedure
# =======================================

#a. Measure of classification error: Cross Validation Loss (Sum of loss/loops)


#b. Model Selection

# Store results for each model
model_results <- data.frame(
  model_id = integer(),
  predictors = character(),
  num_predictors = integer(),
  cv_error = numeric(),
  stringsAsFactors = FALSE
)

# Manual LOOCV function
manual_loocv <- function(predictors, data) {
  n <- nrow(data)
  predictions <- numeric(n)
  actuals <- data$price_class
  
  # Loop through each observation
  for (i in 1:n) {
    # Create training set (all except i-th observation)
    train_fold <- data[-i, ]
    # Test set (only i-th observation)
    test_fold <- data[i, ]
    
    # Create formula
    formula_str <- paste("price_class ~", paste(predictors, collapse = " + "))
    
    # Fit logistic regression model
    model <- glm(as.formula(formula_str), 
                 data = train_fold, 
                 family = binomial(link = "logit"))
    
    # Predict on the left-out observation
    predictions[i] <- predict(model, newdata = test_fold, type = "response")
  }
  
  # Calculate Cross-Validation Error (Mean Squared Error)
  cv_error <- mean((actuals - predictions)^2)
  
  return(list(cv_error = cv_error, predictions = predictions))
}

# Loop through each model combination
cat("Starting LOOCV for", length(all_models), "models...\n\n")

for (i in 1:length(all_models)) {
  predictors <- all_models[[i]]
  
  cat("Model", i, "of", length(all_models), "- Predictors:", 
      paste(predictors, collapse = ", "), "\n")
  
  # Perform manual LOOCV
  loocv_results <- manual_loocv(predictors, train_sampled)
  
  # Store results
  model_results[i, "model_id"] <- i
  model_results[i, "predictors"] <- paste(predictors, collapse = ", ")
  model_results[i, "num_predictors"] <- length(predictors)
  model_results[i, "cv_error"] <- loocv_results$cv_error
}

# Sort by CV Error (lower is better)
model_results <- model_results[order(model_results$cv_error), ]

# Display top 10 models
cat("\n===========================================\n")
cat("Top 10 Models by CV Error (Lower is Better):\n")
cat("===========================================\n")
print(head(model_results, 10))

# Get the best model
best_model_predictors <- all_models[[model_results$model_id[1]]]
cat("\n===========================================\n")
cat("Best Model:\n")
cat("===========================================\n")
cat("Predictors:", paste(best_model_predictors, collapse = ", "), "\n")
cat("CV Error:", model_results$cv_error[1], "\n")
write.csv(model_results, "loocv_results.csv", row.names = FALSE)
library(ggplot2)

# Basic ggplot histogram
ggplot(model_results, aes(x = cv_error)) +
  geom_histogram(bins = 15, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = min(cv_error)), 
             color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = min(model_results$cv_error), y = Inf, 
           label = paste0("Best model\n(CV Error = ", round(min(model_results$cv_error), 6), ")"),
           hjust = -0.1, vjust = 1.5, color = "red", size = 4) +
  labs(title = "Distribution of LOOCV Classification Errors Across All Models",
       x = "Cross-Validated Classification Error (MSE)",
       y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

