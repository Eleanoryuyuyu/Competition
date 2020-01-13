# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:42:30 2019

@author: Administrator
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler




###############################################################################
data = pd.read_csv('../feature_combine.csv')


train = data.loc[:2750, 1:-1]
test = data.loc[2750:, 1:-1]
train_label = data.loc[:2750, '企业总评分']
test_label = data.loc[2750:, '企业总评分']


###############################################################################
def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree, \
                                           include_bias=False)
    linear_regression = SGDRegressor(**kwarg, alpha=0.001)
    pipeline = Pipeline([("polynomial_features", polynomial_features), \
                       ("linear_regression", linear_regression)])
    return pipeline


def sgdlinear(X_train, X_test, y_train, y_test):
    std_x = StandardScaler()
    X_train = std_x.fit_transform(X_train)
    X_test = std_x.transform(X_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1,1))
    y_test = std_y.transform(y_test.reshape(-1,1))
    return X_train, X_test, y_train, y_test
###############################################################################

train, test, train_label, test_label = sgdlinear(train.values, test.values, \
                                                 train_label.values, test_label.values)

degree = 2
penalty = 'l2'   
n_fold = 5
kf = KFold(n=train.shape[0], n_folds=n_fold, shuffle=True, random_state=2019)

n = 0
for index_train, index_eval in kf:
  
    x_train, x_eval = train[index_train], train[index_eval]
    y_train, y_eval = train_label[index_train], train_label[index_eval]
    
    model = polynomial_model(degree=degree, penalty=penalty) #多项式
    model.fit(x_train, y_train)
    testpreds = model.predict(test)
    
    if n > 0:
        totalpreds = totalpreds + testpreds
    else:
        totalpreds = testpreds
    
    n += 1
###############################################################################
totalpreds = totalpreds / n
lin_mse = mean_squared_error(test_label, totalpreds)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)



