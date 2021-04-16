#  Created by Amanda Barroso on 3/26/21.
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time
import math
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

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

# data_cux = pd.read_csv('cuxhaven_data.csv') # 2-year record

data_cux = pd.read_csv('cuxhaven_de.csv') # 2011-2015 record

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

## Understanding the function before applying it

# remember to fix the wind vector calculation, and adjust the time to calendar

#time_orig = pd.to_datetime('1900-01-01')
#data_cux[surge.isna()]         # Check NaNs
#inan = data_cux[surge.isna()].index   # index of NaNS

# storm surge time series data (reduced) --> where weight = 1 (every 10 values)
#surge_ts = pd.DataFrame(data_cux.loc[ weight == weight.unique()[0] ] [['time', 'surge']])

# remove missing/NaN values
#surge_ts.reset_index(inplace=True) # reset index for subsetting isnans
#surge_ts.drop(['index'], axis = 1, inplace=True)
#indx = surge_ts.loc[pd.isna(surge_ts['surge']), :].index  #index of 61 NaNs with weight = 1
##df_new.drop(indx, inplace=True) # This is for the time-lagged timeseries
#surge_ts.drop(indx, inplace=True)

# remove NaNs from complete surge dataset
#predict = data_cux[['surge']]
#predict.drop(inan, inplace=True)
##nani = predict.loc[pd.isna(predict['surge']), :].index  # index of 610 NaNs
##predict.drop(nani, inplace=True)

## Build a function for creating time lagged time series data
def time_lag(data, lags):
    """
    Transforms the dataset to  a time series of grid information and spits back the time lagged time series
    data - the full name of the csv file
    """
    time_orig = pd.to_datetime('1900-01-01')

    df = pd.read_csv(data)
    df.columns = ['time', 'wind_u10', 'wind_v10', 'slp', 'weight', 'surge']
    
    # reorganize the matrix
    df_new = df.loc[df['weight'] == df['weight'].unique()[0]]
    df_new.drop(['weight'], axis = 1, inplace=True) #, 'surge'
    
    for i in range(1,10):
        df_sub = df.loc[df['weight'] == df['weight'].unique()[i]]
        df_sub.drop(['weight', 'surge'], axis = 1, inplace=True)
        df_new = pd.merge(df_new, df_sub, on='time')
    
    
    # lag the time series data
    lagged_df = df_new.copy() # to prevent modifying original matrix
    for j in range(lags):
        #lagged.drop(j, axis = 0, inplace = True)
        lagged_df['time'] = lagged_df['time'] + 6  # 6-hourly
        
        # remove the last row since there is no match for it in df_new
        lagged_df.drop(lagged_df.tail(1).index.item(), axis = 0, inplace = True)
        
        # remove the topmost row from df_new to match lagged
        df_new.drop(df_new.head(1).index.item(), axis = 0, inplace = True)
        
        # merge lagged data with df_new
        df_new = pd.merge(df_new, lagged_df, on = 'time', how = 'outer', \
                       suffixes = ('_left', '_right'))
    
    df_new = df_new.T.reset_index(drop=True).T
    ind = df_new.loc[pd.isna(df_new[df_new.shape[1]-1]), :].index
    df_new.drop(ind, inplace=True)
    
    # storm surge time series data (reduced) --> where weight = 1 is closest to the gauge location (every 10 values)
    surge_ts = pd.DataFrame(df.loc[df['weight'] == \
                                df['weight'].unique()[0]][['time', 'surge']])
    # remove missing/NaN values
    surge_ts.reset_index(inplace=True) # reset index for subsetting isnans
    surge_ts.drop(['index'], axis = 1, inplace=True)
    indx = surge_ts.loc[pd.isna(surge_ts['surge']), :].index
    df_new.drop(indx, inplace=True)
    surge_ts.drop(indx, inplace=True)
    
    # filter surge according to df_new
    lagged_time = list(df_new[0])
    time_df_new = [float(x) for x in df_new[0]]
    time_surge_ts = [float(x) for x in surge_ts['time']]
    time_both = []
    for k in lagged_time:
        if ((k in time_df_new) & (k in time_surge_ts)):
            time_both.append(int(k))
            
    surge_ts = surge_ts[surge_ts['time'].isin(time_both)]
    
    dt = pd.DataFrame(columns = ['date']);
    for i in surge_ts.index:
        dt.loc[i, 'date'] = time_orig + \
            datetime.timedelta(hours = int(surge_ts.loc[i, 'time']))
            
    surge_ts['date'] = dt
    df_new = df_new[df_new[0].isin([x*1.0 for x in time_both])]
    df_new.drop(4, axis = 1, inplace = True) # remove the un-lagged surge data
    return df_new, surge_ts

## DATA PRE-PROCESSING

# 2-yr record un-lagged data
Predictors = pd.DataFrame(data_cux.drop(columns=['time',
                                'weight',
                                'surge']))  # input predictor variables (remove other features)
Inputs = Predictors.drop(inan,axis=0)       # remove NaNs from predictors
Target = predict     # Surge is what we want to predict

# Split data to training and test sets
# 2-year record un-lagged data
x_train, x_test, y_train, y_test, = \
            train_test_split(Inputs, Target, test_size = 0.3, random_state =42)

# Standardize the Training & Test Datasets
x_norm_train = preprocessing.scale(x_train)
x_norm_test = preprocessing.scale(x_test)


# Apply time-lag to the data
#data = 'cuxhaven_data.csv'      # 2-yr. record
data = 'cuxhaven_de.csv'       # 2011-2015 record
x, surge_w1 = time_lag(data, 5) # time-lagged data up to 6-hourly

# Split time-lagged data to training and test sets
# 2-yr. record
lx_train, lx_test, ly_train, ly_test, = \
            train_test_split(x, surge_w1, test_size = 0.3, random_state =42)

# Standardize the time-lagged Training & Test Datasets
lx_norm_train = preprocessing.scale(lx_train)
lx_norm_test = preprocessing.scale(lx_test)

## MACHINE LEARNING METHODS

###### RANDOM FOREST ######

# Try Random Forest Regressor

# use 2-year training data (unlagged)
ul_regr = RandomForestRegressor(max_depth=2, random_state=0)
ul_regr.fit(x_train, y_train)
print(ul_regr.predict([[0, 0, 0]]))  # Predict regression target for X.
ul_predictions = ul_regr.predict(x_test)
ul_regr.score(x_train,y_train)       # r^2 score

ul_rmse = np.sqrt(metrics.mean_squared_error(y_test, ul_predictions))

# adjust parameters
ul_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
ul_regressor.fit(x_train, np.ravel(y_train))
print(ul_regressor.predict([[0, 0, 0]]))  # Predict regression target for X.
ul_rpredictions = ul_regressor.predict(x_test)
print(ul_regressor.score(x_train,y_train)) # 0.93

rmse2 = np.sqrt(metrics.mean_squared_error(y_test, ul_rpredictions))
print(rmse2)  # 0.07


# Perform on the time-lagged data
# 2-yr
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(lx_norm_train, ly_train['surge'])  # Train the model with training dataset
print(_regr.predict([[0, 0, 0]]))  # Predict regression target for X.
predictions = regr.predict(lx_norm_test)    # Predict with the test dataset predictors (wind, mslp)
print(regr.score(lx_norm_train,ly_train['surge']))       # r^2 score
# Compare the surge values from the test dataset to the predicted surge values
lrmse = np.sqrt(metrics.mean_squared_error(ly_test['surge'], predictions)) #0.08

# Adjust parameters (BETTER)
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(lx_norm_train, ly_train['surge']) # Train the model with training dataset
#print(regressor.predict([[0, 0, 0]]))  # Predict regression target for X.
rpredictions = regressor.predict(lx_norm_test)
print(regressor.score(lx_norm_train, ly_train['surge']))  # r^2 score 0.96
# Compare the surge values from the test dataset to the predicted surge values
lrmse2 = np.sqrt(metrics.mean_squared_error(ly_test['surge'], rpredictions))
print(lrmse2) #0.059

# using full record (cuxhaven.de), the r^2 score goes to 0.97, and rmse goes to 0.15

# Plot results

y = ly_test[:]
y.reset_index(inplace=True)
y.drop(['index'], axis = 1, inplace=True)
# order dates chronologically
yy = y.sort_values(by='date')  # Test dataset in chronological order
lyy_test = ly_test.sort_values(by='date')
lyy_test.reset_index(inplace=True)
lyy_test.drop(['index'], axis = 1, inplace=True)


plt.figure(figsize=(14, 7))

plt.plot(surge_w1['date'],surge_w1['surge'], 'black') # un-split surge dataset

plt.plot(lyy_test['date'], yy['surge'], 'blue')
plt.plot(lyy_test['date'], rpredictions, 'green')
plt.legend(['Un-split observed surge dataset', 'Test Observed Surge', 'Predicted Surge (RFR)'], fontsize = 14)
plt.xlabel('Time')
plt.ylabel('Surge Height (m)')
plt.title("Observed vs. Test Predicted Storm Surge Height", fontsize=20, y=1.03)
plt.show()

# Evaluation metrics
RF_MSE = metrics.mean_squared_error(ly_test['surge'], rpredictions)
RF_MAE = metrics.mean_absolute_error(ly_test['surge'], rpredictions)
RF_r2 = regressor.score(lx_norm_train, ly_train['surge'])

##### SUPPORT VECTOR REGRESSION #####

# prepare data format

tr = ly_train[:]
# order dates chronologically
trai = tr.sort_values(by='date')  # Train dataset in chronological order
trai.reset_index(inplace=True)    # Fix new index
trai.drop(['index'], axis = 1, inplace=True)

X = lyy_test['date']

retry = trai['surge']       # surge in training data
horizontal = trai['date']   # date in training data

# attempt support vector regression
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

# Tune the hyperparameters
svr_rbf.get_params()

svr_params = {'kernel': ['rbf'], 'C': [0.1,1,10,20], 'gamma':[1, 0.1, 0.01, 0.001]}
tune = GridSearchCV(SVR(), svr_params, cv=5)
tune.fit(lx_norm_train,ly_train['surge'])
tune.cv_results_

print("Best score: ", tune.best_score_)         #0.727 (2-yr. data)
print("Best parameters: ", tune.best_params_)

# Try with the best parameters (2-yr data)
#svr_rbf = SVR(kernel='rbf', C=1, gamma=0.001)

# Best parameters for cuxhaven.de (~5yr. data)
# score: 0.831, Best parameters:  {'C': 10, 'gamma': 0.001
svr_rbf = SVR(kernel='rbf', C=10, gamma=0.001)

# RBF
svr_rbf.fit(lx_norm_train,ly_train['surge'])
pred_svr_rbf = svr_rbf.predict(lx_norm_test) # surge predictions by the svr_rbf model
print(svr_rbf.score(lx_norm_train, ly_train['surge'])) # Model Score R^2 of 0.777
# Compare the surge values from the test dataset to the predicted surge values
SR_rmse = np.sqrt(metrics.mean_squared_error(lyy_test['surge'], pred_svr_rbf))

# Plot results
plt.figure(figsize=(14, 7))
#plt.plot(surge_w1['date'],surge_w1['surge'], 'black') # un-split surge dataset
plt.plot(lyy_test['date'], lyy_test['surge'], 'blue') # test data (target: surge)

plt.plot(lyy_test['date'], pred_svr_rbf, 'red')

plt.scatter(horizontal[svr_rbf.support_], retry[svr_rbf.support_], \
            facecolor='none', edgecolor='green', )
plt.show()

# Evaluation metrics
SVR_MSE = metrics.mean_squared_error(ly_test['surge'], pred_svr_rbf)
SVR_MAE = metrics.mean_absolute_error(ly_test['surge'], pred_svr_rbf)
SVR_r2 = svr_rbf.score(lx_norm_train, ly_train['surge'])

# Linear
#lin_svr

lin_params = {'kernel': ['linear'], 'C': [0.1,1,10]}
ltune = GridSearchCV(SVR(), lin_params, cv=5)
ltune.fit(lx_norm_train,ly_train['surge'])
ltune.cv_results_

print("Best score: ", ltune.best_score_)         # 0.687
print("Best parameters: ", ltune.best_params_)   # C: 0.1


# Polynomial (quadratic degree 2)
poly_para = {'kernel': ['poly'], 'C': [0.1, 1, 10, 50], 'gamma': [1, 0.1, 0.01, 0.001], 'degree': [2]}
psvm = GridSearchCV(SVR(), poly_para, cv=5)
psvm.fit(lx_norm_train,ly_train['surge'])
psvm.cv_results_

print("Best score: ", psvm.best_score_)         # 0.258
print("Best parameters: ", psvm.best_params_)   #

# degree 3
poly3_para = {'kernel': ['poly'], 'C': [0.1, 1, 10, 50], 'gamma': [1, 0.1, 0.01, 0.001], 'degree': [3]}
p3svm = GridSearchCV(SVR(), poly3_para, cv=5)
p3svm.fit(lx_norm_train,ly_train['surge'])
p3svm.cv_results_

print("Best score: ", p3svm.best_score_)         # 0.53
print("Best parameters: ", p3svm.best_params_)

