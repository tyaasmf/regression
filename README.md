# About

This project is to help AirBNB suggesting the hosts about prices for the new listings by building a model using a newly listings dataset without any review! The modeling purpose is to predict prices to the hosts who sell their properties on AirBNB. The predictions then shall be the recommendation prices to keep the competitive value even before they have review. 

One of Statistical Machine Learning tools used in this project is Random Forest. This algorithm is one of the powerful methods to predict numerical or categorical response. Random Forest is an ensemble method which use the concept of aggregating prediction result from a set of decision tree to increase its performance. Compare to other ensemble methods, decision trees on Randow Forest would have different variables on each trees based on the randomization. This things construct an uncorrelated trees and reduce the variances.     

This project utilized the `mlr3` ecosystem to build the model. `mlr3` is efficient, object-oriented, extensible framework for classification, regression, survival analysis, and other statistical machine learning methods using R language. In this ecosystem, _mlr3_ can run various models simultaneously (for comparing purpose) and provide hyperparameter tuning, feature selection, and ensemble construction.

Dataset source: Kaggle.
