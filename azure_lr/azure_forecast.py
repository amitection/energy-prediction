#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:05:39 2018

@author: amit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

model_name = 'azure_mlp'
dataset = '../dataset'
saved_models = '../saved_models'
output = '../output'
H = 26

def generate_forecasts(test_df):
    '''
    The models trained in notebooks 2-7 are 'one-step' forecasts
    because they are trained to predict one time period into the 
    future. Here, we use the trained model recursively to predict
    multiple future time steps. At each iteration from time t+1
    to the forecast horizon H, the predictions from the previous
    steps become the lagged demand input features for subsequent
    predictions.
    '''
    
    predictions_df = test_df.copy()
    X_test = test_df.copy().drop(['Sum [kWh]', 'Time'], axis=1)
    
    # Iterate over future time steps
    for n in range(1, H+1):
        predictions_df['pred_t'+str(n)] = model.predict(X_test)
        
        # shift lagged demand features...
        shift_demand_features(X_test)
        
        # ...and replace demand_lag1 with latest prediction
        X_test['Sum [kWh]_lag1'] = predictions_df['pred_t'+str(n)]
        
    return predictions_df


def shift_demand_features(df):
    for i in range(H, 1, -1):
        df['Sum [kWh]_lag'+str(i)] = df['Sum [kWh]_lag'+str(i-1)]
        
        
        
        
if __name__=='__main__':
    

    # load the test set
    test = pd.read_csv(os.path.join(dataset, 'demand_test.csv'), parse_dates=['Time'])

    # Load trained model pipeline
    with open(os.path.join(saved_models, model_name + '.pkl'), 'rb') as f:
        model = pickle.load(f)

#    X_test = test.drop(['Sum [kWh]', 'Time'], axis = 1)
#    pred = model.predict(X_test.iloc[1].reshape(1, -1))
    
    
    # generate forecasts on the test set
    predictions_df = generate_forecasts(test)

    predictions_df.to_csv(os.path.join(output, model_name + '_predictions.csv'), float_format='%.4f', index=False)

#    # Store the trained model in the Outputs folder.
#    with open(os.path.join('.', output, model_name + '.pkl'), 'wb') as f:
#        pickle.dump(model, f)


    test = test[1000: 2000]
    predictions_df = predictions_df[1000: 2000]
    plt.plot(test['Time'], test['Sum [kWh]'], color='blue')
    plt.plot(predictions_df['Time'], predictions_df['pred_t1'], color = 'red')
    plt.savefig('output_figs/res.png', format='eps', dpi=1000)