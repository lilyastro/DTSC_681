# do all the model training here. 
from tkinter import FALSE
from prep.clean_data import CleanData
from prep.prep_data import PrepData
from decision_tree import DecisionTree
from random_forest import RandomForest
from k_nearest_neighbors import KNeighbors
import pandas as pd
from prep.performance_data import PerformanceData
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import streamlit as st


# Clean the Data
clean_data = CleanData('model/prep/Star99999_raw.csv')
df = clean_data.filter_df(['Vmag', 'Plx', 'e_Plx', 'B-V', 'SpType'])
df = clean_data.convert_datatypes()
df = clean_data.drop_nulls()
df = clean_data.drop_dupes()
df = clean_data.drop_outlier_plx()
df = clean_data.add_abs_mag()
df.reset_index(inplace=True, drop=True)

size_of_data = len(df)
# print(size_of_data)

# Prep the Data
prep_data = PrepData(df)
prep_data.classify()
df = prep_data.balance()
X, Y = prep_data.split(df, ['Vmag', 'Plx', 'B-V', 'e_Plx', 'B-V', 'Abs_Mag'], ['Target'])
X_encoded = prep_data.encode(X)
#train both types - scaled and non scaled. Scaled will be for K Neighbors
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = prep_data.train(X_encoded, Y, True)
X_train, X_test, y_train, y_test = prep_data.train(X_encoded, Y, False)

##########Initial Training of Data
benchmark = PerformanceData()
##random forest
# random_forest = RandomForest(n_estimators=100, max_depth=25, random_state=0)
# random_forest.train(X_train, y_train)
# preds = random_forest.predict(X_test)
# print(benchmark.get_performance_metrics(random_forest, y_test, preds))
# # 

# decision_tree = DecisionTree()
# decision_tree.train(X_train, y_train)
# preds = decision_tree.predict(X_test)
# print(benchmark.get_performance_metrics(decision_tree, y_test, preds))


# k_neighbors = KNeighbors(n_neighbors=5)
# k_neighbors.train(X_train_scaled, y_train_scaled)
# preds = k_neighbors.predict(X_test)
# print(benchmark.get_performance_metrics(k_neighbors, y_test, preds))

# perform a grid search for each algorithm to find the optimal hyper paramters

#DecisionTree
# dt_params = benchmark.grid_search(DecisionTreeClassifier(), 
# {
#     'max_depth': list(range(1, 15, 1)),
#     'min_samples_split': list(range(2, 20, 2)),
#     'splitter': ["best", "random"]
# }, X_train, y_train)
# print(dt_params)
# #RandomForest
# rf_params = benchmark.grid_search(RandomForestClassifier(), 
# {
#     'n_estimators': list(range(1, 100, 50)),
#     'max_depth': list(range(1, 15, 1)),
#     'random_state': list(range(0, 100, 25))
# }, 
# X_train, y_train)
# print(rf_params)
# #KNearest
# k_params = benchmark.grid_search(KNeighborsClassifier(), 
# {
#     'n_neighbors': list(range(1, 15, 1)),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }, 
# X_train_scaled, y_train_scaled)
# print(k_params)


# decision_tree = DecisionTree(max_depth = 9, min_samples_split=6, splitter='best')
# decision_tree.train(X_train, y_train)
# preds = decision_tree.predict(X_test)
# print(benchmark.get_performance_metrics(decision_tree, y_test, preds))

# random_forest = RandomForest(max_depth = 9, n_estimators=51, random_state=50)
# random_forest.train(X_train, y_train)
# preds = random_forest.predict(X_test)
# print(benchmark.get_performance_metrics(random_forest, y_test, preds))


# k_neighbors = KNeighbors(metric = 'euclidean', n_neighbors=14, weights='uniform')
# k_neighbors.train(X_train_scaled, y_train_scaled)
# preds = k_neighbors.predict(X_test)
# print(benchmark.get_performance_metrics(k_neighbors, y_test, preds))


# plot



## try on just 2 classes - dwarf and giant, not other. 
df = df[(df['Target'] == 1) | (df['Target'] == 0)]


X, Y = prep_data.split(df, ['Vmag', 'Plx', 'B-V', 'e_Plx', 'B-V', 'Abs_Mag'], ['Target'])
X_encoded = prep_data.encode(X)
#train both types - scaled and non scaled. Scaled will be for K Neighbors
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = prep_data.train(X_encoded, Y, True)
X_train, X_test, y_train, y_test = prep_data.train(X_encoded, Y, False)

##########Initial Training of Data
benchmark = PerformanceData()
##random forest
random_forest = RandomForest(n_estimators=100, max_depth=25, random_state=0)
random_forest.train(X_train, y_train)
preds = random_forest.predict(X_test)
print(benchmark.get_performance_metrics(random_forest, y_test, preds))
# 

decision_tree = DecisionTree()
decision_tree.train(X_train, y_train)
preds = decision_tree.predict(X_test)
print(benchmark.get_performance_metrics(decision_tree, y_test, preds))


k_neighbors = KNeighbors(n_neighbors=5)
k_neighbors.train(X_train_scaled, y_train_scaled)
preds = k_neighbors.predict(X_test)
print(benchmark.get_performance_metrics(k_neighbors, y_test, preds))

# perform a grid search for each algorithm to find the optimal hyper paramters

#DecisionTree
# dt_params = benchmark.grid_search(DecisionTreeClassifier(), 
# {
#     'max_depth': list(range(1, 15, 1)),
#     'min_samples_split': list(range(2, 20, 2)),
#     'splitter': ["best", "random"]
# }, X_train, y_train)
# print(dt_params)
# #RandomForest
# rf_params = benchmark.grid_search(RandomForestClassifier(), 
# {
#     'n_estimators': list(range(1, 100, 50)),
#     'max_depth': list(range(1, 15, 1)),
#     'random_state': list(range(0, 100, 25))
# }, 
# X_train, y_train)
# print(rf_params)
# #KNearest
# k_params = benchmark.grid_search(KNeighborsClassifier(), 
# {
#     'n_neighbors': list(range(1, 15, 1)),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }, 
# X_train_scaled, y_train_scaled)
# print(k_params)


decision_tree = DecisionTree(max_depth = 6, min_samples_split=18, splitter='best')
decision_tree.train(X_train, y_train)
preds = decision_tree.predict(X_test)
print(benchmark.get_performance_metrics(decision_tree, y_test, preds))

random_forest = RandomForest(max_depth = 10, n_estimators=51, random_state=75)
random_forest.train(X_train, y_train)
preds = random_forest.predict(X_test)
print(benchmark.get_performance_metrics(random_forest, y_test, preds))


k_neighbors = KNeighbors(metric = 'euclidean', n_neighbors=13, weights='uniform')
k_neighbors.train(X_train_scaled, y_train_scaled)
preds = k_neighbors.predict(X_test)
print(benchmark.get_performance_metrics(k_neighbors, y_test, preds))


