df <- read.csv("data/madrid_houses_clean.csv")
library(dplyr)

head(df)
nrow(df)
ncol(df)
colnames(df)
typeof(df)
df$price_label <- ifelse(df$buy_price >= 500000, 1, 0)
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

library(caret)
library(pROC)

# ========================================
# STEP 1: Generate All Possible Models
# =======================================
outcome_name <- c("price_label")
predictor_names <- setdiff(c(colnames(train)),outcome_name)

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
# STEP 2: Cross-Validation Setup
# ========================================

set.seed(20251126)
k_folds <- 10  # 10-fold CV

# Create fold indices
folds <- createFolds(train[[outcome_name]], k = k_folds, list = TRUE)
# ========================================
# STEP 3: Evaluate All Models
# ========================================

# ========================================
# STEP 3: Evaluate All Models
# ========================================

# Storage for results
cv_results <- data.frame(
  model_id = 1:length(all_models),
  predictors = character(length(all_models)),
  n_predictors = integer(length(all_models)),
  cv_accuracy = numeric(length(all_models)),
  cv_error = numeric(length(all_models)),
  cv_auc = numeric(length(all_models)),
  stringsAsFactors = FALSE
)

# Loop through all models
for (i in 1:length(all_models)) {
  
  predictors <- all_models[[i]]
  cv_results$predictors[i] <- paste(predictors, collapse = " + ")
  cv_results$n_predictors[i] <- length(predictors)
  
  # Storage for fold-level metrics
  fold_accuracy <- numeric(k_folds)
  fold_error <- numeric(k_folds)
  fold_auc <- numeric(k_folds)
  
  # K-fold cross-validation
  for (j in 1:k_folds) {
    
    # Split data
    train_indices <- unlist(folds[-j])
    test_indices <- folds[[j]]
    
    train_fold <- train[train_indices, ]
    test_fold <- train[test_indices, ]
    
    # Create formula
    formula_str <- paste(outcome_name, "~", paste(predictors, collapse = " + "))
    model_formula <- as.formula(formula_str)
    
    # Fit logistic regression
    model <- glm(model_formula, data = train_fold, family = binomial)
    
    # Predict on test fold
    predictions_prob <- predict(model, newdata = test_fold, type = "response")
    predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)
    
    # Calculate metrics
    fold_accuracy[j] <- mean(predictions_class == test_fold[[outcome_name]])
    fold_error[j] <- mean(predictions_class != test_fold[[outcome_name]])
    
    # Calculate AUC
    roc_obj <- roc(test_fold[[outcome_name]], predictions_prob, quiet = TRUE)
    fold_auc[j] <- auc(roc_obj)
  }
  
  # Average across folds
  cv_results$cv_accuracy[i] <- mean(fold_accuracy)
  cv_results$cv_error[i] <- mean(fold_error)
  cv_results$cv_auc[i] <- mean(fold_auc)
  
  # Progress indicator
  if (i %% 10 == 0) cat("Evaluated", i, "of", length(all_models), "models\n")
}

# ========================================
# STEP 4: Select Best Model
# ========================================

# Sort by CV accuracy (or you can use cv_auc or minimize cv_error)
cv_results_sorted <- cv_results[order(-cv_results$cv_accuracy), ]

cat("\n=== TOP 10 MODELS (by Accuracy) ===\n")
print(head(cv_results_sorted[, c("predictors", "n_predictors", "cv_accuracy", "cv_error", "cv_auc")], 10))

# Best model (maximum accuracy)
best_model_idx <- which.max(cv_results$cv_accuracy)
best_predictors <- all_models[[best_model_idx]]

cat("\n=== BEST MODEL ===\n")
cat("Predictors:", paste(best_predictors, collapse = " + "), "\n")
cat("CV Accuracy:", cv_results$cv_accuracy[best_model_idx], "\n")
cat("CV Error Rate:", cv_results$cv_error[best_model_idx], "\n")
cat("CV AUC:", cv_results$cv_auc[best_model_idx], "\n")
