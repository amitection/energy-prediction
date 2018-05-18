#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:58:23 2018

@author: amit

AZURE MLP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import pickle
import os


model_name = "azure_mlp"
dataset = '../dataset'
saved_models = '../saved_models'


train = pd.read_csv(os.path.join(dataset, 'demand_train.csv'), parse_dates=['Time'])
X = train.drop(['Sum [kWh]', 'Time'], axis=1)

len_train = int(len(X) * 0.75)
len_valid = len(X) - len_train
test_fold = [-1]*len_train + [0]*len_valid
ps = PredefinedSplit(test_fold)


regr = MLPRegressor(solver='lbfgs', verbose=True)

hidden_layer_size = [(5,), (10,), (15,), (20,), (25,), (30,), (35,), (40,), (10,10), (20,20), (30,30), (40,40)]

param_grid = {'hidden_layer_sizes': hidden_layer_size,
             'alpha': [0.01, 0.1, 1.0, 10.0]}
regr_cv = GridSearchCV(estimator=regr,
            param_grid=param_grid,
            cv=ps,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1)

regr_pipe = Pipeline([('regr_cv', regr_cv)])
regr_pipe.fit(X, y=train['Sum [kWh]'])


with open(os.path.join(saved_models, model_name + '.pkl'), 'wb') as f:
    pickle.dump(regr_pipe, f)