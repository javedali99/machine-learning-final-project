#  Created by Amanda Barroso on 3/26/21.
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

## Load St. Pete Water Level data
matstruct_contents = sio.loadmat('WaterLevel_St_Pete_hourly.mat')

raw_waterlevel= matstruct_contents['WaterLevel']
Time = matstruct_contents['Time']

plt.plot(Time,raw_waterlevel)
plt.show()

detrended = sio.loadmat('hourlydt.mat')

dt_water = detrended['hourlydt']
plt.plot(Time,dt_water)
plt.show()

## Load Cuxhaven storm surge data (because it is already prepared)
data_cux = pd.read_csv('cuxhaven_data.csv')

# features
data_cux.info()

time = data_cux['time hours since 1900-01-01 00:00:00.0']
u_wind = data_cux['u10 m s**-1']
v_wind = data_cux['v10 m s**-1']
mslp = data_cux['msl Pa']
weight = data_cux['Distance Weight']
surge = data_cux['surge m']

# directly rename the columns
data_cux.columns = ['time', 'u_wind', 'v_wind', 'mslp', 'weight', 'surge']

## Pre-processing
# remember to fix the wind vector calculation, and adjust the time to calendar

time_orig = pd.to_datetime('1900-01-01')
data_cux[surge.isna()]         # Check NaNs
inan = data_cux[surge.isna()].index   # index of NaNS

# storm surge time series data (reduced) --> where weight = 1 (every 10 values)
surge_ts = pd.DataFrame(data_cux.loc[ weight == weight.unique()[0] ] [['time', 'surge']])

# remove missing/NaN values
surge_ts.reset_index(inplace=True) # reset index for subsetting isnans
surge_ts.drop(['index'], axis = 1, inplace=True)
indx = surge_ts.loc[pd.isna(surge_ts['surge']), :].index  #index of 61 NaNs with weight = 1
#df_new.drop(indx, inplace=True) # This is for the time-lagged timeseries
surge_ts.drop(indx, inplace=True)

# remove NaNs from complete surge dataset
predict = data_cux[['surge']]
nani = predict.loc[pd.isna(predict['surge']), :].index  # index of 610 NaNs
predict.drop(nani, inplace=True)

#rmv = data_cux[['surge']].loc[pd.isna(data_cux[['surge']]['surge']),:].index

## Machine Learning Methods

###### RANDOM FOREST ######

## Apply the five-fold cross validation of the random forest learning algorithm to the training data to extract average classification accuracy

Predictors = pd.DataFrame(data_cux.drop(columns=['time',
                                'weight',
                                'surge']))  # input predictor variables (remove other features)
Inputs = Predictors.drop(inan,axis=0)       # remove NaNs from predictors
Target = predict     # Surge is what we want to predict

RF = RandomForestClassifier(criterion='gini')
#RF.fit(Inputs,Target)
# ValueError: Unknown label type: 'continuous'
#print(rf.score(Inputs,Target)) #Check model performance (1 being the best)

#rf_accuracy = cross_val_score(rf,Inputs,Target)
#avg_rf_accuracy = rf_accuracy.mean()

# Try Random Forest Regressor

# test on synthetic data
X, y = make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)
Regr = RandomForestRegressor(max_depth=2, random_state=0)
Regr.fit(X, y)
print(Regr.predict([[0, 0, 0, 0]]))

# use my data
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(Inputs, np.ravel(Target))
print(regr.predict([[0, 0, 0]]))  # Predict regression target for X.
regr.score(Inputs,np.ravel(Target))

##### SUPPORT VECTOR MACHINE #####

# attmept support vector regression
