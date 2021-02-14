# Movie Recommendation Capstone Project ----
# Author: Jose Caloca

# 1.0 Libraries ----
# We install and load the packages
start_time <- Sys.time()

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# 2.0 Get data ----


# For this project is used the MovieLens 10M dataset:
# download and further information: https://grouplens.org/datasets/movielens/10m/

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# 3.0 Analysis ----

# For the completion of this project we will be using machine learning algorithms and techniques. In order to achieve this, first we will split edx into training and test sets.

# 80% of the data will be used for training the algorithm and the outstanding 20% for testing. 

test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
test_set <- edx[test_index,]
train_set <- edx[-test_index,]

# We make sure to not include movies and users in the test set that are not in the training set

test_set <- test_set %>% semi_join(train_set, by = "movieId")
test_set <- test_set %>% semi_join(train_set, by = "userId")

# We check if there are any missing values (NA) in the training set 

sum(is.na(train_set$rating)) 

# There are not missing values in the training dataset

# The methodology followed improves the results of different algorithms (models) and make an ensemble of them, comparing the accuracies of the each of the models

# Since this project will be graded based on RMSE results, I will create a function for calculating it as follows:

RMSE <- function(actual, predicted){
  sqrt(mean((actual - predicted)^2))
}

# This first model will include only the mean of movie ratings and movie effects, b_i :=  rating - avg

mu <- mean(train_set$rating)

movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# We make the prediction using the mean of movie ratings and movie effects

pred_1 <- mu + test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  pull(b_i)

rmse1 <- RMSE(test_set$rating, pred_1)

# This first  model yields a RMSE of 0.9437144 

# Now we will add the user effects. User effects are, b_u := rating - avg - b_i

user_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>% #we need to join the tables in order to have access to b_i
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

pred_2 <- mu + test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(bu_bi = b_u + b_i) %>%
  pull(bu_bi)

rmse2 <- RMSE(test_set$rating, pred_2)

# This model yields a RMSE of 0.8661625 after adding the user effects. Next we regularize the movie effects and see if the RMSE improves

#the regularized movie effects is, b_i := sum(rating - mu)/(n + lambda)
# As we don't know the optimal value of lambda, we will use cross validation to obtain it as follows:

lambdas <- seq(0,10,0.1) 

sums <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), nums = n())

rmse_cv <- sapply(lambdas, function(lam){#this function makes predictions using different values of lambda
  predictions <- test_set %>%
    left_join(sums, by = "movieId") %>%
    mutate(b_i = s/(nums + lam)) %>%
    mutate(preds = mu + b_i) %>%
    pull(preds)
  return(RMSE(test_set$rating, predictions))
})


lambdas[which.min(rmse_cv)] # this tells us that the optimal lambda to use is 1.7
#next we will create the model using lambda = 1.7

lambda <- 1.7 

reg_movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(reg_bi = sum(rating - mu)/(n() + lambda))

pred_3 <- test_set %>%
  left_join(reg_movie_effects, by = "movieId") %>%
  mutate(preds = mu + reg_bi) %>%
  pull(preds)

rmse3 <- RMSE(test_set$rating, pred_3) # By following this method, the above model yields a RMSE of  0.9436775. This model didn't improve more as expected. Now we will regulariZe by the user effect to see if there is any difference in the RMSE.

# The regularized user effects equal, b_u := sum(rating - mu - reg_bi)/(n + lambda)
# as we don't know the optimal lamba, we will use cross validation to optain it, as follows:
# NOTE: The following chunck of code will take some time to run

rmses_cv2 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
    left_join(reg_user, by = "userId") %>%
    mutate(preds = mu + bi + bu) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})

lambdas[which.min(rmses_cv2)]

#the lambda that minimizes rmse for reg_user + reg_movie is 4.7

lambda2 <- 4.7

#now we run the model with lambda = 4.7 to get the rmse.

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda2 + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda2 + n()))

pred_4 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
  left_join(reg_user, by = "userId") %>%
  mutate(preds = mu + bi + bu) %>%
  pull(preds)

rmse4 <- RMSE(test_set$rating, pred_4)
rmse4

#This is the lowest RMSE yet: 0.8655424.


#Now I will attempt to make an ensemble of the 4 models that were used in this analysis.
#The idea is to average the ratings from the 4 models and use that as a prediction


pred_5 <- (pred_1 + pred_2 + pred_3 + pred_4)/4

rmse5 <- RMSE(test_set$rating, pred_5)

#this gave an rmse of 0.8847612, which is more than the previous rmse. 

#At the moment, regularized effects have yield better results, so we will regularize genre_effects
#reg_gen := sum(rating - mu - bi - bu)/(lambda + n)

# Regularized Movie + User + Genre Effects model

rmses_cv3 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    mutate(preds = mu + bi + bu + b_g) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})



lambdas[which.min(rmses_cv3)]

#the lambda that minimizes the RMSE for reg_user + reg_movie + reg_genres is 4.7

lambda3 <- 4.7

# now we run the model with the optimal value of lambda

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda3 + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda3 + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda3 + n()))

pred_6 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so we have access to both bi and bu
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  mutate(preds = mu + bi + bu + b_g) %>%
  pull(preds)

rmse6 <- RMSE(test_set$rating, pred_6)

#this model yield a lower RMSE of 0.8655406, although is not what it is expected to achieve. Therefore we check the results after regularizing  by year.

# Regularized Movie + User + Genre + Year Effects: 

#first, I will create a new column to represent the date for both test and train set

library(lubridate)
train_set <- train_set %>%
  mutate(date = as_datetime(timestamp), 
         year = year(date), 
         month = month(date)) 

test_set <- test_set %>%
  mutate(date = as_datetime(timestamp), 
         year = year(date), 
         month = month(date))

# As the train and test sets have a year and month column we will analyse the first model with year effects. Then by month effect.

#first we use cross validation to find the best lambda for the lowest RMSE
lambdas <- seq(0,10,0.25)
rmses_cv4 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  reg_year <- train_set %>% 
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% 
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(reg_year, by = "year") %>%
    mutate(preds = mu + bi + bu + b_g + b_y) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})

lambdas[which.min(rmses_cv4)] #The optimal lambda for this model is 4.75

#now we run the model with the optimal lambda and see what the RMSE is

lambda <- lambdas[which.min(rmses_cv4)]

reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

reg_year <- train_set %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lambda + n()))

pred_7 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% 
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  mutate(preds = mu + bi + bu + b_g + b_y) %>%
  pull(preds)

rmse7 <- RMSE(test_set$rating, pred_7)

# This model yields a RMSE of 0.8655207, Still not fulfilling the expectations. Therefore we check the model by adding year and month variables

# Regularized Movie + User + Genre + Year + Month Effects:

rmses_cv5 <- sapply(lambdas, function(lam){
  reg_movie <- train_set %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(lam + n())) #reg_movie will contain bi
  
  reg_user <- train_set %>%
    left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(lam + n()))
  
  reg_genre <- train_set %>%
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - bi - bu)/(lam + n()))
  
  reg_year <- train_set %>% 
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lam + n()))
  
  reg_month <- train_set %>% 
    left_join(reg_movie, by = "movieId") %>%
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(reg_year, by = "year") %>%
    group_by(month) %>%
    summarize(b_m = sum(rating - mu - bi - bu - b_g - b_y)/(lam + n()))
  
  predictions <- test_set %>%
    left_join(reg_movie, by = "movieId") %>% 
    left_join(reg_user, by = "userId") %>%
    left_join(reg_genre, by = "genres") %>%
    left_join(reg_year, by = "year") %>%
    left_join(reg_month, by = "month") %>%
    mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
    pull(preds)
  
  return(RMSE(test_set$rating, predictions))
})

lambdas[which.min(rmses_cv5)] #this lambda is also 4.75 (similar to previous models).


lambda <- lambdas[which.min(rmses_cv5)]
reg_movie <- train_set %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- train_set %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- train_set %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

reg_year <- train_set %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lambda + n()))

reg_month <- train_set %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  group_by(month) %>%
  summarize(b_m = sum(rating - mu - bi - bu - b_g - b_y)/(lambda + n()))

pred_8 <- test_set %>%
  left_join(reg_movie, by = "movieId") %>% 
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  left_join(reg_month, by = "month") %>%
  mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
  pull(preds)

rmse8 <- RMSE(test_set$rating, pred_8)

# This model yield a RMSE of 0.8655208, but still not low enough.


methods <- c("Movie Effects",
             "Movie + User Effects",
             "Reg Movie Effects",
             "Reg Movie/User Effects",
             "Ensemble of predictions 1,2,3,4(just the mean)",
             "Reg Movie/User/Genres Effects",
             "Reg Movie/User/Genres/Year Effects",
             "Reg Movie/User/Genres/Year/Month Effects")

rmse_table <- c(rmse1,
                rmse2,
                rmse3,
                rmse4,
                rmse5,
                rmse6,
                rmse7,
                rmse8)

pred_table <- c("pred_1",
                "pred_2",
                "pred_3",
                "pred_4",
                "pred_5",
                "pred_6",
                "pred_7",
                "pred_8")

model_comparison <- data.frame(method = methods, rmse = rmse_table, pred_index = pred_table)

model_comparison %>% arrange(rmse)


# 4.0 Validation----

# This project shows that the best method is all regularized features. So to get ready to have the final RMSE, we will add the year and month to edx and validation

edx <- edx %>%
  mutate(year = year(as_datetime(timestamp)), month = month(as_datetime(timestamp)))

validation <- validation %>%
  mutate(year = year(as_datetime(timestamp)), month = month(as_datetime(timestamp)))


lambda <- lambdas[which.min(rmses_cv5)] 

reg_movie <- edx %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(lambda + n())) #reg_movie will contain bi

reg_user <- edx %>%
  left_join(reg_movie, by = "movieId") %>% #need to join so that we have access to bi
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(lambda + n()))

reg_genre <- edx %>%
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - bi - bu)/(lambda + n()))

reg_year <- edx %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - bi - bu - b_g)/(lambda + n()))

reg_month <- edx %>% 
  left_join(reg_movie, by = "movieId") %>%
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  group_by(month) %>%
  summarize(b_m = sum(rating - mu - bi - bu - b_g - b_y)/(lambda + n()))

pred_final <- validation %>%
  left_join(reg_movie, by = "movieId") %>% 
  left_join(reg_user, by = "userId") %>%
  left_join(reg_genre, by = "genres") %>%
  left_join(reg_year, by = "year") %>%
  left_join(reg_month, by = "month") %>%
  mutate(preds = mu + bi + bu + b_g + b_y + b_m) %>%
  pull(preds)

final_rmse <- RMSE(validation$rating, pred_final)
final_rmse

#pred_final contains the final predictions using the validation set with a RMSE of 0.8647976.

end_time <- Sys.time()

# Time difference of 33.85353 mins
end_time - start_time
