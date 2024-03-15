# Introduction

This document is a part of my application for the Data Scientist role at Ahrefs. 

## Objective

The project's goal is to construct an algorithm that effectively classifies images as relevant or irrelevant to specified queries.

## Scoring Criteria

Model evaluation hinges on the AUC ROC metric, with a benchmark established at an AUC of 0.5, representing the performance threshold our model aspires to exceed.

## Key Insights

1. The critical aspect of this project is a meticulous Exploratory Data Analysis (EDA) and Feature Engineering (FE). Many crucial features are buried inside seemingly unimportant/ unrelated information.
2. As expected, the best performing model is an Learning to Rank (LTR) algorithm (LGBM - LambdaRank). However, regular classification method such as CatBoost is able to perform with a comparable performance.
3. Despite showing similar performance results, both models focuses on very different feature sets. LGBM - LambdaRank focuses more on text-based features while CatBoost seems to prefer more subtle features such as image location and dimension.