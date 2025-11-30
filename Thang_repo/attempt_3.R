df <- read.csv("data/madrid_houses_clean.csv")
library(dplyr)

head(df)
nrow(df)
ncol(df)
colnames(df)
typeof(df)
df$price_class <- ifelse(df$buy_price >= 500000, 1, 0)
colnames(df)
df <- df[df$district == 2, ]
trimmed_df <- df[,c(3,4,5,10,11,14,18)]
colnames(trimmed_df)
nrow(trimmed_df)

set.seed(20251126)
#install.packages("caret")
#install.packages("pROC")

trimmed_df$has_lift <- as.numeric(as.logical(trimmed_df$has_lift))
trimmed_df$is_renewal_needed <- as.numeric(as.logical(trimmed_df$is_renewal_needed))
trimmed_df$has_parking <- as.numeric(as.logical(trimmed_df$has_parking))



library(caret)
library(pROC)

# ========================================
# STEP 1: Generate All Possible Models
# =======================================
outcome_name <- c("price_class")
predictor_names <- setdiff(c(colnames(trimmed_df)),outcome_name)

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
  loocv_results <- manual_loocv(predictors, trimmed_df)
  
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


# Fit the best model
model <- glm(price_class ~ n_rooms + n_bathrooms + is_renewal_needed + has_parking, 
             data = train, 
             family = binomial)

# Predict on test set
model.prob <- predict(model, test, type = "response")

# Calculate MSE on test set (matching your LOOCV approach)
actual <- test$price_class
test_mse <- mean((actual - model.prob)^2)

# Print results
cat("\n=================================\n")
cat("Test Set Performance:\n")
cat("=================================\n")
cat("Test MSE:", round(test_mse, 6), "\n")

# Summary of the model
cat("\n=================================\n")
cat("Model Summary:\n")
cat("=================================\n")
summary(model)




# ========================================
# DOUBLE/NESTED LOOCV IMPLEMENTATION
# ========================================

# Outer LOOCV function


double_loocv <- function(data, all_models) {
  n <- nrow(data)
  outer_predictions <- numeric(n)
  outer_actuals <- data$price_class  # <-- FIX: Add price_class
  selected_models <- character(n)
  
  cat("Starting Double LOOCV with", n, "outer iterations...\n\n")
  
  # OUTER LOOP: Leave one observation out
  for (i in 1:n) {
    if (i %% 100 == 0) cat("Outer iteration", i, "of", n, "\n")
    
    # Split data
    outer_train <- data[-i, ]
    outer_test <- data[i, ]
    
    # INNER LOOP: Model selection using LOOCV on outer_train
    inner_results <- data.frame(
      model_id = integer(),
      cv_error = numeric(),
      stringsAsFactors = FALSE
    )
    
    for (j in 1:length(all_models)) {
      predictors <- all_models[[j]]
      
      # Perform LOOCV on outer_train for this model
      inner_cv_error <- manual_loocv(predictors, outer_train)$cv_error
      
      inner_results[j, "model_id"] <- j
      inner_results[j, "cv_error"] <- inner_cv_error
    }
    
    # Select best model from inner LOOCV
    best_model_idx <- which.min(inner_results$cv_error)
    best_predictors <- all_models[[best_model_idx]]
    selected_models[i] <- paste(best_predictors, collapse = ", ")
    
    # Refit best model on full outer_train
    formula_str <- paste("price_class ~", paste(best_predictors, collapse = " + "))
    final_model <- glm(as.formula(formula_str), 
                       data = outer_train, 
                       family = binomial(link = "logit"))
    
    # Predict on outer_test (the single left-out observation)
    outer_predictions[i] <- predict(final_model, newdata = outer_test, type = "response")
  }
  
  # Calculate final CV error
  final_cv_error <- mean((outer_actuals - outer_predictions)^2)
  
  return(list(
    cv_error = final_cv_error,
    predictions = outer_predictions,
    selected_models = selected_models
  ))
}

# Run double LOOCV
cat("\n===========================================\n")
cat("Running Double/Nested LOOCV...\n")
cat("===========================================\n")

double_loocv_results <- double_loocv(trimmed_df, all_models)

cat("\n===========================================\n")
cat("Double LOOCV Results:\n")
cat("===========================================\n")
cat("Final CV Error (MSE):", double_loocv_results$cv_error, "\n")

# See which models were most frequently selected
cat("\nMost frequently selected models:\n")
print(head(sort(table(double_loocv_results$selected_models), decreasing = TRUE), 10))


 


