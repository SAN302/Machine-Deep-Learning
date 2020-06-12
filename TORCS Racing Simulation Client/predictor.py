# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:32:10 2019

@author: SAN
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error



#def DataPrediction(data_f, pred_f_no):
    

data1 = pd.read_csv('All_data_train.csv')

data2 = pd.read_csv('All_data_test.csv')

data = data1.values
labels = data2.values

# dat1 = data[:,0:20]
# dat2 = data[:,64:78]
# data = np.concatenate((dat1,dat2), axis=1)
#data = data[:,0:78]
print 'data shape: ',data.shape
#print 'data: ',data

print 'labels shape: ',labels.shape
#print 'labels: ',labels

onelabel = labels[:,4]
onelabel = onelabel.reshape(-1, 1)
#print 'onelabel shape: ',onelabel.shape
#print 'onelabel: ',onelabel
X_train, X_test, y_train, y_test = train_test_split(data, onelabel, test_size=0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

regressor = LinearRegression()
regressor.fit(X_train,y_train)
accuracy = regressor.score(X_test,y_test)


# XGBModel = XGBRegressor()
# XGBModel.fit(X_train,y_train)
# XGBPredctions = XGBModel.predict(X_test)
# MAE = mean_absolute_error(y_test, XGBPredctions)

print 'XGBPredctions: ',XGBPredctions[5]
print 'Actual: ',y_test[5]
print 'XGBPredctions: ',XGBPredctions[32]
print 'Actual: ',y_test[32]
print 'XGBPredctions: ',XGBPredctions[555]
print 'Actual: ',y_test[555]
print 'XGBPredctions: ',XGBPredctions[14555]
print 'Actual: ',y_test[14555]
print 'XGBPredctions: ',XGBPredctions[20000]
print 'Actual: ',y_test[20000]

print 'MAE: ',MAE
print 'feature_importances_: ',XGBModel.feature_importances_
pyplot.bar(range(len(XGBModel.feature_importances_)), XGBModel.feature_importances_)
pyplot.show()



#print("Selected Features: %s" % fitter.support_)
#print(accuracy*100,'%')
#print X_test[0,:].reshape(1, -1).shape
#print X_test[0,:].reshape(1, -1)
#print rfe.predict(X_test[0,:].reshape(1, -1))

filename = 'steer_model.sav'
pickle.dump(XGBModel, open(filename, 'wb'))