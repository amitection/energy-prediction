#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:58:59 2018

@author: amit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
import pickle
import os


dataset = "../dataset"
saved_models = "../saved_models"
model_name = "azure_lr"

train = pd.read_csv(os.path.join(dataset, 'demand_train.csv'), parse_dates=['Time'])

X = train.drop(['Time','Sum [kWh]'], axis=1)

# One hot encode categorical features
cat_cols = ['hour', 'month', 'dayofweek']
cat_cols_idx = [X.columns.get_loc(c) for c in X.columns if c in cat_cols]
onehot = OneHotEncoder(categorical_features=cat_cols_idx, sparse=False)

# Linear Regression Model
regr = linear_model.LinearRegression(fit_intercept=True)

# Train-Test split
tscv = TimeSeriesSplit(n_splits=2)

#ts = train[['Time', 'Sum [kWh]']].copy()
#ts.reset_index(drop=True, inplace=True)
#
#for split_num, split_idx  in enumerate(tscv.split(ts)):
#    split_num = str(split_num)
#    train_idx = split_idx[0]
#    test_idx = split_idx[1]
#    ts['fold' + split_num] = "not used"
#    ts.loc[train_idx, 'fold' + split_num] = "train"
#    ts.loc[test_idx, 'fold' + split_num] = "test"
#    
#gs = gridspec.GridSpec(3,1)
#fig = plt.figure(figsize=(15, 10), tight_layout=True)


#ax = fig.add_subplot(gs[0])
#ax.plot(ts.loc[ts['fold0']=="train", "Time"], ts.loc[ts['fold0']=="train", "Sum [kWh]"], color='b')
#ax.plot(ts.loc[ts['fold0']=="test", "Time"], ts.loc[ts['fold0']=="test", "Sum [kWh]"], 'r')
#ax.plot(ts.loc[ts['fold0']=="not used", "Time"], ts.loc[ts['fold0']=="not used", "Sum [kWh]"], 'w')
#
#ax = fig.add_subplot(gs[1], sharex=ax)
#plt.plot(ts.loc[ts['fold1']=="train", "Time"], ts.loc[ts['fold1']=="train", "Sum [kWh]"], 'b')
#plt.plot(ts.loc[ts['fold1']=="test", "Time"], ts.loc[ts['fold1']=="test", "Sum [kWh]"], 'r')
#plt.plot(ts.loc[ts['fold1']=="not used", "Time"], ts.loc[ts['fold1']=="not used", "Sum [kWh]"], 'w')



regr_cv = RFECV(estimator=regr,
             cv=tscv,
             scoring='neg_mean_squared_error',
             verbose=2,
             n_jobs=-1)




# Create a Regressor Pipeline
regr_pipe = Pipeline([('onehot', onehot), ('rfecv', regr_cv)])

# Fit the regressor pipe
regr_pipe.fit(X, y=train['Sum [kWh]'])

# Save the trained model
with open(os.path.join(saved_models, model_name + '.pkl'), 'wb') as f:
    pickle.dump(regr_pipe, f)