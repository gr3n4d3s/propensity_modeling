## Read in packages
library(tidyverse)
library(tidymodels)
library(sparklyr)

## Read in data files
download.file(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip", "bank_data.zip", mode = "wb")
unzip("bank_data.zip", exdir = "data")
file.remove("bank_data.zip")

# normalize data
list.files("data")
bank_data <- read.csv("data/bank-full.csv", sep = ";") 

# take a quick peek at the data/explore
head(bank_data)
bank_data %>%
    summarise(across(everything(), ~sum(is.na(.))))
bank_data %>%
    {table(.$y)} %>%
    as.data.frame() %>%
    rename(outcome_y = Var1) %>%
    mutate(percentage = round((Freq/sum(Freq))*100, 2))

# maybe a quick little clean and normalize
# normalize column names, character fields, factor outcome 
# bin ages, balances etc. (all arbitrary for now)
bank_data <- bank_data %>%
    rename_all(~tolower(str_replace_all(., "\\W", ""))) %>%
    mutate(across(where(is.character), ~tolower(str_squish(.))),
           y = as.factor(if_else(str_detect(y, "y"), 1, 0)),
           age = as.factor(case_when(age < 30 ~ "<30",
                                     age >= 30 & age <= 50 ~ "30 - 50",
                                     age >= 51 & age <= 70 ~ "50 - 70",
                                     TRUE ~ "71+")),
           balance = round(balance, -3),
           duration = round(duration/60, 0))
str(bank_data)
# prep for models
# tidymodels
# split data
set.seed(1234)
split_bank <- initial_split(bank_data)

train_bank <- training(split_bank)
test_bank <- testing(split_bank)

# potentially cross validate
cvf <- vfold_cv(train_bank, v = 5)

# prepare model glm because why not
glm_model <- logistic_reg() %>%
    set_engine("glm") %>%
    set_mode("classification")

# recipe for model (using all predictors for outcome) ... no tuning 
bank_recipe <- recipe(y ~ ., data = train_bank) #%>% update_role()

# build workflow
bank_wflow <- workflow() %>%
    add_model(glm_model) %>%
    add_recipe(bank_recipe)

# fit model/train
fit_bank <- bank_wflow %>%
    fit_resamples(object = glm_model,
                  preprocessor = bank_recipe,
                  resamples = cvf,
                  metrics = metric_set(kap, accuracy))
# fit model to test data
banking_model_fit <- bank_wflow %>%
    last_fit(split = split_bank,
             metrics = metric_set(kap, accuracy))
# combine test with predicions
bank_test_pred <- bind_cols(
    test_bank,
    banking_model_fit %>% 
        collect_predictions() %>%
        select(.pred_class)
)
#check accuracy
bank_accuracy <- bank_test_pred %>%
    {table("Predicted" = .$.pred_class, "Observed" = .$y)} %>%
    as.data.frame()

### sparklyr

sc <- spark_connect(master = "local")

sprk_df <- bank_data %>%
    copy_to(sc, ., name = "sprk_df")

sprk_df_split <- sprk_df %>%
    compute("sprk_df_split") %>%
    sdf_random_split(test = .2, train = .8, seed = 1234)


sprk_rf_fit <- sprk_df_split$train %>%
    ml_random_forest(y ~ ., type = "regression")

sprk_rf_tfi <- ml_tree_feature_importance(sc = sc, model = sprk_rf_fit) %>%
    collect()

sprk_rf_prediction <- sprk_rf_fit %>% 
    ml_predict(sprk_df_split$test) %>%
    collect()

spark_disconnect(sc)

### refine tidymodel
glimpse(train_bank)

refine_bank_recipe <- training(split_bank) %>%
    recipe(y ~.) %>%
    step_corr(all_date_predictors()) %>%  # remove fields of high correlation
    step_center(where(is.numeric)) %>%    # normalize to have mean zero
    step_scale(where(is.numeric)) %>%     # normalize to sd dev of 1
    prep() # builds operation 

refine_bank_recipe

refine_bank_test_data <- refine_bank_recipe %>% 
    bake(testing(split_bank))

refine_bank_test_data

refine_bank_train_data <- juice(refine_bank_recipe)

bank_rf_model <- rand_forest(trees = 100, mode = "classification") %>%
    set_engine("randomForest") %>%
    fit(y ~ ., data = refine_bank_train_data)

bank_rf_outcome <- refine_bank_test_data %>%
    bind_cols( 
        bank_rf_model %>% 
            predict(refine_bank_test_data)
    ) 

metrics(bank_rf_outcome, truth = y, estimate = .pred_class)

refined_bank_accuracy <- bank_rf_outcome %>%
    {table("Predicted" = .$.pred_class, "Observed" = .$y)} %>%
    as.data.frame()
save(refined_bank_accuracy, file = "R Data/bank_table.Rdata")

