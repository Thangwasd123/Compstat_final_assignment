df <- read.csv("data/madrid_houses_clean.csv")

head(df)
nrow(df)
ncol(df)
colnames(df)
typeof(df)

#Installing packages to observe missing data
#install.packages("Amelia")
library(Amelia)
missmap(df, main = "Missing values vs observed")
#No missing value found

#Filter for only neccessary predictors and target variable
trimmed_df <- df[,c(3,4,5,10,11,14)]
colnames(trimmed_df)
#Train test Split 70/30
set.seed(20251126)
sample <- sample(c(TRUE, FALSE), nrow(trimmed_df), replace = TRUE, prob=c(0.7,0.3))
train <- df[sample,]
test <- df[!sample,]

#sanity check
print(nrow(test)/(nrow(train) + nrow(test)))

#The six allowed predictors are:
#1. sq_mt_built
#2. n_rooms
#3. n_bathrooms
#4. has_lift
#5. is_renewal_needed
#6. has_parking


#First, checking data types of these columns
features_list = list("sq_mt_built", "n_rooms", "n_bathrooms", "has_lift", "is_renewal_needed", "has_parking")

typeof(trimmed_df$sq_mt_built)
typeof(trimmed_df$n_rooms)
typeof(trimmed_df$n_rooms)

trimmed_df$has_lift <- as.factor(trimmed_df$has_lift)
table(trimmed_df$has_lift)
typeof(trimmed_df$has_lift)

trimmed_df$is_renewal_needed = as.factor(trimmed_df$is_renewal_needed)
table(trimmed_df$is_renewal_needed)
typeof(trimmed_df$is_renewal_needed)

trimmed_df$has_parking = as.factor(trimmed_df$has_parking)
table_val <- table(trimmed_df$has_parking)
typeof(trimmed_df$has_parking)
print(as.integer(table_val[1]))

#Column selection -> What would be the most interesting to use for modelling?
#Use Entropy on factors 
#install.packages("entropy")
library(entropy)
entropy(table(trimmed_df$has_lift))
#0.623
entropy(table(trimmed_df$is_renewal_needed))
#0.477
entropy(table(trimmed_df$has_parking))
#0.65

#has_lift & has_parking are the two tops

#Use entropy on integer variable
#1. sq_mt_built
#2. n_rooms
#3. n_bathrooms
discr_sq_mt_built <- cut(trimmed_df$sq_mt_built, breaks = 10) #10 bins
entropy(table(discr_sq_mt_built))
#0.534

discr_n_rooms <- cut(trimmed_df$n_rooms, breaks = 10) #10 bins
entropy(table(discr_n_rooms))
#1.034

discr_n_bathrooms <- cut(trimmed_df$n_bathrooms, breaks = 10) #10 bins
entropy(table(discr_n_bathrooms))
#0.779

# Four top features with highest entropy are: n_rooms, n_bathrooms, has_lifts, has_bathrooms 



# To test for each scenario, we need a list of each possible predictor variable, simply by looping through each list

#1 Predictor Variable
features_list = list("sq_mt_built", "n_rooms", "n_bathrooms", "has_lift", "is_renewal_needed", "has_parking")

#2 Predictor Variable
two_predictors_list <- list()
for (i in 1:length(features_list)){
  feature_1 <- features_list[i]
  remaining_val <- setdiff(features_list, feature_1)  # Also fixed the order here
  for (j in 1:length(remaining_val)){
    feature_2 <- remaining_val[j]
    two_predictors_list<- append(two_predictors_list, paste(feature_1, feature_2, sep = ","))
  }
}

#3 Predictor Variable
three_predictors_list <- list()
for (i in 1:length(features_list)){
  feature_1 <- features_list[i]
  remaining_val <- setdiff(features_list, feature_1)  # Also fixed the order here
  for (j in 1:length(remaining_val)){
    feature_2 <- remaining_val[j]
    remaining_list <- setdiff(remaining_val, feature_2)
    for (k in 1:length(remaining_list)){
      feature_3 <- remaining_list[k]
      three_predictors_list <- append(paste(feature_1, feature_2, feature_3, sep = ","), three_predictors_list)
    }
  }
}


#4 Predictor Variable

four_predictors_list <- list()
for (i in 1:length(features_list)){
  feature_1 <- features_list[i]
  remaining_val <- setdiff(features_list, feature_1)  # Also fixed the order here
  for (j in 1:length(remaining_val)){
    feature_2 <- remaining_val[j]
    remaining_list <- setdiff(remaining_val, feature_2)
    for (k in 1:length(remaining_list)){
      feature_3 <- remaining_list[k]
      remaining_remains <- setdiff(feature_3, remaining_list)
    
      for (m in 1:length(remaining_remains)){
        feature_4 <- remaining_remains[m]
        print(paste(feature_1, feature_2, feature_3, feature_4, sep = ",")
      }
  }}}

"""
The method of the leave_one_out cross validation
Input a full dataframe
Extrapolate the index
Setup a list of all index
For each value in index:
Train with everything but that index, and test on that, and repeat untill all index has lapsed
"""






