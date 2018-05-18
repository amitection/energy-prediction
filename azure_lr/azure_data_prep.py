#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:32:27 2018

@author: amit

https://github.com/Azure/MachineLearningSamples-EnergyDemandTimeSeriesForecasting/blob/master/1-data-preparation.ipynb
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import LabelEncoder
import datetime
import os

dataset = "../dataset"

# Import Data
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %I:%M %p')
D = pd.read_csv("SumProfiles_1800s.Electricity-3years.csv", sep=';', 
                parse_dates=['Time'], date_parser=dateparse, usecols=['Time', 'Sum [kWh]'] )

ts = D
#ts.index = ts['Time']

# Histogram
#plt.hist(ts.dropna(), bins=100)
#plt.title('Demand distribution')
#plt.show()


# Autocorrelation plot
#autocorrelation_plot(ts['Sum [kWh]'].dropna())
#plt.xlim(0,48)
#plt.title('Auto-correlation of hourly demand over a 24 hour period')
#plt.show()
## 37 -11


ts_features = ts.copy()

ts_features['hour'] = ts_features['Time'].apply(lambda dt: ((dt.hour*60) + dt.minute) // 30)
ts_features['month'] = ts_features.Time.dt.month-1
ts_features['dayofweek'] = ts_features.Time.dt.dayofweek

def generate_lagged_features(df, var, max_lag):
    for t in range(1, max_lag+1):
        df[var+'_lag'+str(t)] = df[var].shift(t)
        
        
generate_lagged_features(ts_features, 'Sum [kWh]', 26)

# Drop rows with null values
ts_features.dropna(how='any', inplace=True)

# Split into train and test
train, test = (ts_features.loc[ts_features['Time']<'2016-01-01'], 
               ts_features.loc[ts_features['Time']>='2016-01-01'])
train.to_csv(os.path.join(dataset, 'demand_train.csv'), float_format='%.4f', index=False)
test.to_csv(os.path.join(dataset, 'demand_test.csv'), float_format='%.4f', index=False)