library(tidyverse)
library(mlr3verse)
library(mlr3tuning)
library(mlr3extralearners)
library(ranger)
library(DataExplorer)
library(sjPlot)
library(skimr)
library(ggridges)
library(GGally)

# Import Data
data_house<- read.csv("train.csv", stringsAsFactors = TRUE)

# Exploration and Visualization
plot_intro(data = data_house, geom_label_args = list(size=2.5))
skim_without_charts(data = data_house)
plot_histogram(data=data_house,nrow=3, ncol=3,
               geom_histogram_args = list(fill="blue"))
data_house%>%
  filter(city %in% c("Boston","Chicago","DC","LA","NYC","SF"))%>%
  ggplot() +
  geom_boxplot(aes(x=city, y=price, fill=city), show.legend=F) +
  ggtitle("Price Distribution for Each City") +
  ylab("House Price") +
  xlab("City") + 
  theme(plot.title = element_text(hjust = 0.5))+
  theme_bw()
ggcorr(data_house[,-1], method = c("everything","pearson"),geom = "tile")

# Pre-processing Data
data_house <- data_house %>% 
  select(-id) %>%
  select_if(is.numeric) %>%
  na.omit()
```

# Import learner to mlr3 ecosystem
task_house = TaskRegr$new(id="house", backend = data_house, target = "price")
model_rf <- lrn("regr.ranger", importance="impurity")

# Hyperparameter Tuning (using cross-validation 5 folds)
param_bound_rf <- ParamSet$new(params = list(ParamInt$new("mtry", 
                                                          lower = 2,
                                                          upper = 8),
                                             ParamInt$new("max.depth", 
                                                          lower = 1,
                                                          upper = 15)))
terminate = trm("evals", n_evals = 5) #stopping criteria
tuner <- tnr("random_search") #opr=timization
resample_inner = rsmp("cv", folds = 5)

model_rf_tune <- AutoTuner$new(learner = model_rf,
                               measure = msr("regr.mae"),
                               terminator = terminate,
                               resampling = resample_inner,
                               search_space = param_bound_rf,
                               tuner = tuner,
                               store_models = TRUE)

resample_outer = rsmp("cv", folds = 5)
set.seed(1)
resample_outer$instantiate(task = task_house)

# Model comparation before and after tuning (using mean absolute error)
model_house <- list(model_rf,
                    model_rf_tune)

design <- benchmark_grid(tasks = task_house,
                         learners = model_house,
                         resamplings = resample_outer)

lgr::get_logger("bbotk")$set_threshold("warn")
bmr = benchmark(design,store_models = TRUE)

result = bmr$aggregate(msr("regr.mae"))
result

# Optimized hyperparameter
get_param_res <- function(i){
  as.data.table(bmr)$learner[[i]]$tuning_result
}

best_rf_param =map_dfr(6:10,get_param_res)
best_rf_param

best_rf_param %>% slice_min(regr.mae)

best_rf_param_value <-  c(best_rf_param %>%
                            slice_min(regr.mae) %>%
                            pull(mtry),
                          best_rf_param %>%
                            slice_min(regr.mae) %>%
                            pull(max.depth))
best_rf_param_value

# Best Model interpretation
model_rf_best = lrn("regr.ranger", mtry=best_rf_param_value[1], 
                    max.depth=best_rf_param_value[2], importance="impurity")
model_rf_best$train(task = task_house)

model_rf_best$model$variable.importance
importance <- data.frame(Predictors = names(model_rf_best$model$variable.importance),
                         impurity = model_rf_best$model$variable.importance)
rownames(importance) <- NULL
importance %>% arrange(desc(impurity))

# Prediction
data_house_new <- read.csv("test.csv", stringsAsFactors = TRUE)
data_house_new <- data_house_new %>% 
  arrange(id) %>%
  select(-id) %>% 
  select_if(is.numeric)

# Handling null values
data_house_new[is.na(data_house_new)] = 0
data_house_new$bathrooms[data_house_new$bathrooms==0] = median(data_house_new$bathrooms)
data_house_new$host_response_rate[data_house_new$host_response_rate==0] = median(data_house_new$host_response_rate)
data_house_new$review_scores_rating[data_house_new$review_scores_rating==0] = median(data_house_new$review_scores_rating)
data_house_new$bedrooms[data_house_new$bedrooms==0] = median(data_house_new$bedrooms)
data_house_new$beds[data_house_new$beds==0] = median(data_house_new$beds)

prediction_rf_new <- model_rf_best$predict_newdata(newdata = data_house_new)
as.data.table(prediction_rf_new)
