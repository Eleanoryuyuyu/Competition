# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:35:17 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb


###############################################################################
data = pd.read_csv('../feature_combine.csv')

train = data.loc[:2750, 1:-1]
test = data.loc[2750:, 1:-1]
train_label = data.loc[:2750, '企业总评分']
test_label = data.loc[2750:, '企业总评分']
###############################################################################
params = { \
    'boosting_type': 'gbdt', \
    'objective': 'regression_l2', \
    'metric': 'l2', \
    'num_leaves': 32, \
    'learning_rate': 0.01, \
    'feature_fraction': 0.75, \
    'bagging_fraction': 0.8, \
    'bagging_freq': 5, \
    'verbose': 0, \
    'save_binary': True, \
    'min_data_in_leaf': 10, \
    'max_bin': 100, \
}

n_fold = 5
kf = KFold(n=train.shape[0], n_folds=n_fold, shuffle=True, random_state=2019)

n = 0
for index_train, index_eval in kf:

    x_train, x_eval = train.iloc[index_train], train.iloc[index_eval]
    y_train, y_eval = train_label[index_train], train_label[index_eval]
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_eval, y_eval, reference=lgb_train)
    
    gbm = lgb.train(params, \
                    lgb_train, \
                    num_boost_round=20000, \
                    valid_sets=[lgb_eval], \
                    verbose_eval=100, \
                    early_stopping_rounds=500)
    
    print('start predicting on test...')
    testpreds = gbm.predict(test.values, num_iteration=gbm.best_iteration)
    if n > 0:
        totalpreds = totalpreds + testpreds
    else:
        totalpreds = testpreds
    #gbm.save_model('lgb_model_fold_{}.txt'.format(n), num_iteration=gbm.best_iteration)
    n += 1
###############################################################################
totalpreds = totalpreds / n
lin_mse = mean_squared_error(test_label.values.reshape(-1, 1), totalpreds)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)








