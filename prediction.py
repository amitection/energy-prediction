#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:53:18 2018

@author: amit

Class responsible for energy consumption prediction
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd
import pickle
import os

class EnergyConsumptionPrediction:
    
    parent_dir = ''
    
    def __init__(self, parent_dir, model_name):
        print("Instantiating class...")
        
        # Initiate training   
        self.model = self.load_trained_model(parent_dir, model_name)
        
        # Load the Test Data
        self.test_data = self.__load_test_dataset(parent_dir)

        
    def load_trained_model(self, parent_dir, model_name):
        '''
        Loads a previously trained model or trains a new model.
        '''
        mpath = os.path.join(parent_dir, model_name + '.pkl')
        # If model already present, load the saved model
        if(os.path.exists(mpath)):
            # Load the trained model
            with open(mpath, 'rb') as f:
                model = pickle.load(f)
        else:
            # train the model if not already present
            print('Model ('+mpath+') not present. Training a new model...')
           
            model = self.__train(parent_dir, model_name)
            
        return model
        
    def predict(self, timestamp):
        datapoint = self.test_data.loc[self.test_data['Time'] == timestamp]
        X_test = datapoint.drop(['Sum [kWh]', 'Time'], axis = 1)
        pred = self.model.predict(X_test)
        pred=pred[0]
        print('True: '+str(datapoint['Sum [kWh]'])+' Predicted: '+str(pred))
        return pred
    
    
    def __train(self, parent_dir, model_name):
        '''
        Trains the model using the input time series
        '''
        print("Training initiated")
        
        train = self.__load_dataset(parent_dir)
        X_train = train.drop(['Sum [kWh]', 'Time'], axis=1)
        Y_train = train['Sum [kWh]']
        
        len_train = int(len(X_train) * 0.75)
        len_valid = len(X_train) - len_train
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
        regr_pipe.fit(X_train, Y_train)
        
        # Save the trained model
        with open(os.path.join(parent_dir, model_name + '.pkl'), 'wb') as f:
            pickle.dump(regr_pipe, f)
            
        return regr_pipe
        
        
    def __load_test_dataset(self, parent_dir):
        # load the test set
        test_data =  pd.read_csv(os.path.join(parent_dir, 'demand_test.csv'), parse_dates=['Time'])
        return test_data
        
    def __load_dataset(self, parent_dir):
        '''
        Load the dataset as a time series.
        '''
        train = pd.read_csv(os.path.join(parent_dir, 'demand_train.csv'), parse_dates=['Time'])
        X = train.drop(['Sum [kWh]', 'Time'], axis=1)
        return train