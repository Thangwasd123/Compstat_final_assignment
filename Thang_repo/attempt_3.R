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

table(df$price_class)
print(nrow(test)/(nrow(train) + nrow(test)))

#install.packages("caret")
#install.packages("pROC")

library(caret)
library(pROC)

# ========================================
# STEP 1: Generate All Possible Models
# =======================================
outcome_name <- c("price_class")
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